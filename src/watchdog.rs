//! The wait-fed turn watchdog.
//!
//! A running turn nobody waits on is presumed abandoned: while a turn is
//! running, every attached `wait`/`wait_any` call — CLI or MCP `serve`
//! — feeds the watchdog; an idle session and a turn awaiting a permission
//! answer are not watched, and a turn that goes without an attached waiter
//! for `wait_grace` is cancelled and its session closed. The session
//! stays registered, adoptable. The grace clock starts when the turn
//! starts and whenever the last waiter detaches; a daemon supersede
//! restarts it once so a `wait` can reattach through the forwarding path.
//! It is driven by a timer and the notify edges, never by polling.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use tracing::warn;

use crate::state::{
    AppState, SessionEnd, Status, Subagent, end_session, end_target, next_turn_number,
};

/// The default grace period: a running turn with no attached waiter for
/// this long is cancelled and its session closed.
pub const WAIT_GRACE: Duration = Duration::from_secs(60);

/// One `wait`/`wait_any` call's attachment to a subagent's watchdog.
/// Held for the call's life: a finished call detaches, and a dead
/// caller's call detaches when its connection closes and the server
/// drops the in-flight call.
#[derive(Debug)]
pub struct Waiter(Arc<crate::state::SubRuntime>);

impl Waiter {
    /// Attach to `sub`'s running turn and wake the watchdog.
    pub fn attach(sub: &Subagent) -> Self {
        sub.rt.waiters.fetch_add(1, Ordering::Relaxed);
        sub.rt.watch.notify_waiters();
        Self(sub.rt.clone())
    }
}

impl Drop for Waiter {
    fn drop(&mut self) {
        self.0.waiters.fetch_sub(1, Ordering::Relaxed);
        self.0.watch.notify_waiters();
    }
}

/// One subagent's watchdog, spawned at launch and running until the
/// session leaves `live`. Each loop either sleeps out the rest of the
/// grace for the unwatched turn — waking early on any waiter or status
/// change — or waits for the next change while watched or idle.
pub async fn run(state: Arc<AppState>, sub: Arc<Subagent>) {
    let grace = state.wait_grace;
    // The unwatched turn and when its grace began: `None` while no turn
    // runs unwatched. A turn boundary — a chained turn under a still
    // `running` status — starts a fresh clock, as does the last
    // waiter's detach.
    let mut unwatched: Option<(u64, Instant)> = None;
    let mut superseded = false;
    loop {
        // Register the wake-ups before sampling state so a change
        // between the read and the await is not lost.
        let notified = sub.rt.notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        let fed = sub.rt.watch.notified();
        tokio::pin!(fed);
        fed.as_mut().enable();
        let progressed = state.handover.progressed.notified();
        tokio::pin!(progressed);
        progressed.as_mut().enable();

        if sub.rt.closing.get().is_some()
            || !state
                .live
                .lock()
                .expect("live poisoned")
                .contains_key(&sub.session_id)
        {
            return;
        }
        // A graceful restart is not waiter loss: the takeover restarts
        // this daemon's kept turns' grace clocks once, so a `wait` can
        // reattach through the successor's forwarding before a clock
        // armed before the restart fires.
        if !superseded && state.handover.superseded.load(Ordering::Relaxed) {
            superseded = true;
            unwatched = unwatched.map(|(turn, _)| (turn, Instant::now()));
        }
        // Only a running turn is watched. A turn blocked on a permission
        // answer burns nothing, and `wait` settles at once on it, so no
        // waiter can feed it while the coordinator decides.
        let turn = {
            let inner = sub.rt.inner.lock().expect("inner poisoned");
            match inner.status {
                Status::Running => Some(
                    inner
                        .current
                        .as_ref()
                        .map_or_else(|| next_turn_number(&inner), |turn| turn.n),
                ),
                _ => None,
            }
        };
        let wake = if let (Some(turn_n), 0) = (turn, sub.rt.waiters.load(Ordering::Relaxed)) {
            let since = match unwatched {
                Some((t, since)) if t == turn_n => since,
                _ => {
                    let since = Instant::now();
                    unwatched = Some((turn_n, since));
                    since
                }
            };
            grace.saturating_sub(since.elapsed())
        } else {
            unwatched = None;
            Duration::MAX
        };
        if wake.is_zero() {
            warn!(
                session_id = %sub.session_id,
                "no waiter fed the turn within the grace period; closing the session"
            );
            if let SessionEnd::Live(sub) = end_target(&state, &sub.session_id, false) {
                end_session(&state, &sub).await;
            }
            return;
        }
        tokio::select! {
            () = notified => {},
            () = fed => {},
            () = progressed => {},
            () = tokio::time::sleep(wake), if wake != Duration::MAX => {},
        }
    }
}
