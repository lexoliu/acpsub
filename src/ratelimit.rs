//! Provider rate limits: parking a turn the provider rate-limited, and
//! resuming it when the quota lifts.
//!
//! A `session/prompt` answered with a retryable `unavailable` JSON-RPC
//! error ends its turn in [`Status::RateLimited`] — not `failed` — and
//! registers the limit on the agent's [`QuotaScope`]: while it is active
//! every new prompt to every session in the scope parks instead of
//! burning a request. The resume scheduler wakes at each limit, then
//! resumes every parked session once its `resume_at` passes — the same
//! ACP session, `session/load` when the process is gone — with one fixed
//! continuation prompt, staggered so the first post-limit prompts do not
//! hit the freshly lifted quota in a single burst. Prompts parked while
//! the limit held run after the continuation.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use aither_acp::ClientError;
use serde_json::{Value, json};
use tracing::{debug, warn};

use crate::registry::Registry;
use crate::state::{AppState, Launch, Status, Subagent, launch, now, start_turn};

/// The JSON-RPC code a provider rate-limit rejection carries — inside
/// the spec's implementation-defined server-error range.
const RATE_LIMIT_CODE: i32 = -32010;

/// The fixed prompt every parked session resumes with.
pub(crate) const RESUME_PROMPT: &str = "The rate limit has lifted. Continue the task.";

/// How far apart the parked sessions of one scope resume — staggering
/// keeps the first post-limit prompts from hitting the freshly lifted
/// quota in a single burst.
const RESUME_STAGGER: Duration = Duration::from_secs(1);

/// The scheduler's longest sleep between checks: bounds how stale its
/// view of parked entries can get, and how long it lingers once it is the
/// only `AppState` holder left.
const SCHEDULER_POLL: Duration = Duration::from_millis(250);

/// The set of sessions a provider quota binds: the agent config key.
///
/// Every session of one config runs under that config's single provider
/// login, so a per-login limit stops them together. Kept as a type so a
/// richer key can join once a real source for one exists.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct QuotaScope {
    /// The agent config key whose sessions the limit binds.
    agent: String,
}

impl QuotaScope {
    /// The scope `agent`'s prompts draw on.
    #[must_use]
    pub fn for_agent(agent: &str) -> Self {
        Self {
            agent: agent.to_string(),
        }
    }
}

/// A rate limit in effect for one [`QuotaScope`]: every new prompt to a
/// session in the scope parks until `resume_at`.
#[derive(Debug, Clone)]
pub struct Limit {
    /// When the provider says the quota resets.
    pub resume_at: jiff::Timestamp,
    /// The provider's own message — the reason `wait`/`status` report.
    pub reason: String,
}

/// A rate limit parsed off a failed `session/prompt`.
#[derive(Debug, Clone)]
pub(crate) struct RateLimit {
    /// When the provider says the quota resets.
    pub(crate) resume_at: jiff::Timestamp,
    /// The provider's own message — the parked reason.
    pub(crate) reason: String,
}

/// Whether `error` is a provider rate limit, decided from its structured
/// fields alone: the server-range code plus a `data` block declaring the
/// failure `unavailable` and `retryable` — never the message text.
///
/// `resume_at` comes from the message's `(at HH:MM[:SS] UTC)` clause,
/// read as the next occurrence of that clock time. With no readable
/// clause there is nothing to schedule on: `None`, so the caller fails
/// the turn with the original error — the daemon never guesses a delay.
pub(crate) fn parse(error: &ClientError) -> Option<RateLimit> {
    let ClientError::JsonRpc(error) = error else {
        return None;
    };
    if error.code.0 != RATE_LIMIT_CODE {
        return None;
    }
    let data = error.data.as_ref()?;
    let tagged = data.get("cognition.ai/errorKind").and_then(Value::as_str) == Some("unavailable")
        && data.get("cognition.ai/retryable").and_then(Value::as_bool) == Some(true);
    if !tagged {
        return None;
    }
    Some(RateLimit {
        resume_at: parse_reset_at(&error.message)?,
        reason: error.message.clone(),
    })
}

/// The reset time a rate-limit message names in `(at HH:MM[:SS] UTC)`:
/// the clock fields inside the literal clause, on their next occurrence
/// — today when the time is still ahead, tomorrow when it has passed.
/// `None` when the clause is missing or unreadable.
fn parse_reset_at(message: &str) -> Option<jiff::Timestamp> {
    let start = message.find("(at ")? + 4;
    let end = message[start..].find(" UTC)")? + start;
    let clock = message[start..end].trim();
    let (hour, minute, second) = match clock.split(':').collect::<Vec<_>>().as_slice() {
        [h, m] => (parse_u8(h)?, parse_u8(m)?, 0),
        [h, m, s] => (parse_u8(h)?, parse_u8(m)?, parse_u8(s)?),
        _ => return None,
    };
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let now = jiff::Timestamp::now();
    let utc = jiff::tz::TimeZone::UTC;
    let mut at = now
        .to_zoned(utc.clone())
        .date()
        .at(
            i8::try_from(hour).ok()?,
            i8::try_from(minute).ok()?,
            i8::try_from(second).ok()?,
            0,
        )
        .to_zoned(utc)
        .ok()?;
    while at.timestamp() <= now {
        at = at.tomorrow().ok()?;
    }
    Some(at.timestamp())
}

/// A 0–255 field of a `HH:MM[:SS]` clock, accepting the bare digits only.
fn parse_u8(text: &str) -> Option<u8> {
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// Rebuild the quota gates from registry entries whose `rate_limited`
/// state survives a restart: a prompt against a still-limited scope keeps
/// parking. Entries whose `resume_at` already passed do not gate — they
/// stay parked for the scheduler to resume.
pub(crate) fn restored_limits(registry: &Registry) -> BTreeMap<QuotaScope, Limit> {
    let now = jiff::Timestamp::now();
    let mut limits = BTreeMap::new();
    for (_, entry) in registry.iter() {
        let Some(parked) = &entry.rate_limited else {
            continue;
        };
        let Ok(resume_at) = parked.resume_at.parse::<jiff::Timestamp>() else {
            continue;
        };
        if resume_at <= now {
            continue;
        }
        let limit = limits
            .entry(QuotaScope::for_agent(&entry.agent))
            .or_insert_with(|| Limit {
                resume_at,
                reason: parked.reason.clone(),
            });
        if resume_at > limit.resume_at {
            limit.resume_at = resume_at;
            limit.reason.clone_from(&parked.reason);
        }
    }
    limits
}

/// The quota gate for `agent`'s scope: `Some(limit)` while its reported
/// `resume_at` is still ahead — a prompt now would burn against an
/// already-known limit. An expired gate is dropped on read.
pub(crate) fn active_limit(state: &AppState, agent: &str) -> Option<Limit> {
    let mut limits = state.limits.lock().expect("limits poisoned");
    let scope = QuotaScope::for_agent(agent);
    match limits.get(&scope) {
        Some(limit) if limit.resume_at > jiff::Timestamp::now() => Some(limit.clone()),
        Some(_) => {
            limits.remove(&scope);
            None
        }
        None => None,
    }
}

/// What `park_prompt` reports back: where the prompt parked until.
pub(crate) struct Parked {
    /// When the parked session resumes.
    pub(crate) resume_at: jiff::Timestamp,
    /// The provider's own message — the parked reason.
    pub(crate) reason: String,
    /// The prompt's position in the parked queue — the resume's
    /// continuation runs ahead of all of them.
    pub(crate) position: usize,
}

/// Park `prompt` on `sub` while its quota scope is limited — or the
/// session is already parked: it joins the queue behind the scheduled
/// continuation, and a promptable session's status becomes
/// `rate_limited`. A session still running a turn (or waiting on a
/// permission) keeps its status; its parked prompt chains into the limit
/// the same way. `None` when nothing parks the prompt.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub(crate) async fn park_prompt(
    state: &Arc<AppState>,
    sub: &Arc<Subagent>,
    prompt: &str,
) -> Option<Parked> {
    let gate = active_limit(state, &sub.agent);
    let parked = {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        let (resume_at, reason) = match &inner.status {
            Status::RateLimited { resume_at, reason } => (*resume_at, reason.clone()),
            _ => gate.map(|limit| (limit.resume_at, limit.reason))?,
        };
        inner.queue.push_back(prompt.to_string());
        if inner.status.accepts_prompt() {
            inner.status = Status::RateLimited {
                resume_at,
                reason: reason.clone(),
            };
        }
        Parked {
            resume_at,
            reason,
            position: inner.queue.len(),
        }
    };
    record_park(state, sub, prompt, parked.resume_at, &parked.reason).await;
    ensure_scheduler(state);
    Some(parked)
}

/// Re-park a chained prompt whose turn popped into an active `limit`:
/// the queue's own pop reserved the slot, so the prompt returns to the
/// queue's front and the session turns `rate_limited` — the scheduler's
/// continuation comes first, queued prompts run after it.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub(crate) async fn repark_prompt(
    state: &Arc<AppState>,
    sub: &Arc<Subagent>,
    prompt: &str,
    limit: &Limit,
) {
    {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        inner.pending_turn_start = None;
        inner.status = Status::RateLimited {
            resume_at: limit.resume_at,
            reason: limit.reason.clone(),
        };
        inner.queue.push_front(prompt.to_string());
    }
    record_park(state, sub, prompt, limit.resume_at, &limit.reason).await;
    ensure_scheduler(state);
}

/// The park's records and wake-ups: the transcript's `park` record, the
/// registry's `rate_limited` flag, and a `notify` for `wait` callers.
async fn record_park(
    state: &Arc<AppState>,
    sub: &Subagent,
    prompt: &str,
    resume_at: jiff::Timestamp,
    reason: &str,
) {
    let record = json!({"ts": now(), "park": prompt, "resume_at": resume_at.to_string()});
    if let Err(error) = sub.rt.transcript.lock().await.append(&record).await {
        warn!(%error, "transcript write failed");
    }
    let reason = reason.to_string();
    let resume_at = resume_at.to_string();
    let session_id = sub.session_id.clone();
    let result = state
        .update_registry(|registry| {
            if let Some(entry) = registry.get_mut(&session_id) {
                entry.rate_limited = Some(crate::registry::RateLimited { resume_at, reason });
            }
        })
        .await;
    if let Err(error) = result {
        warn!(%error, "registry persist failed");
    }
    sub.rt.notify.notify_waiters();
    state.limits_notify.notify_waiters();
}

/// Register `hit` on `agent`'s quota scope: new prompts to every session
/// in the scope park until `resume_at`. A repeat report only ever pushes
/// the gate later — the freshest message becomes the reason — and every
/// parked session in the scope, live or registry-only, moves to the
/// gate's time.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub(crate) async fn note_limit(state: &Arc<AppState>, agent: &str, hit: &RateLimit) {
    let scope = QuotaScope::for_agent(agent);
    let resume_at = {
        let mut limits = state.limits.lock().expect("limits poisoned");
        let limit = limits.entry(scope).or_insert_with(|| Limit {
            resume_at: hit.resume_at,
            reason: hit.reason.clone(),
        });
        if hit.resume_at > limit.resume_at {
            limit.resume_at = hit.resume_at;
        }
        limit.reason.clone_from(&hit.reason);
        let resume_at = limit.resume_at;
        drop(limits);
        resume_at
    };
    for sub in state.live.lock().expect("live poisoned").values() {
        if sub.agent != agent {
            continue;
        }
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        if let Status::RateLimited { resume_at: at, .. } = &mut inner.status {
            *at = (*at).max(resume_at);
        }
        drop(inner);
        sub.rt.notify.notify_waiters();
    }
    let result = state
        .update_registry(|registry| {
            for (_, entry) in registry.iter_mut() {
                if entry.agent == agent
                    && let Some(parked) = entry.rate_limited.as_mut()
                    && let Ok(at) = parked.resume_at.parse::<jiff::Timestamp>()
                    && resume_at > at
                {
                    parked.resume_at = resume_at.to_string();
                }
            }
        })
        .await;
    if let Err(error) = result {
        warn!(%error, "registry persist failed");
    }
    state.limits_notify.notify_waiters();
    ensure_scheduler(state);
}

/// Spawn the resume scheduler once: `scheduler_started` guards it, so
/// `AppState::new`, `park_prompt`, and `note_limit` can all call it.
///
/// # Panics
///
/// Panics if called outside a Tokio runtime — every caller is async.
pub(crate) fn ensure_scheduler(state: &Arc<AppState>) {
    if state.scheduler_started.swap(true, Ordering::Relaxed) {
        return;
    }
    tokio::spawn(limit_scheduler(state.clone()));
}

/// The resume scheduler: wakes on every parked limit, resumes every
/// registry-parked session whose `resume_at` has passed — staggered so
/// they do not burst — then sleeps until the next one or the next
/// parked limit. Exits when nothing outside the task holds the
/// `AppState`: a `serve`/`daemon` shutdown, or a test dropping it.
async fn limit_scheduler(state: Arc<AppState>) {
    loop {
        if Arc::strong_count(&state) <= 1 {
            return;
        }
        let now = jiff::Timestamp::now();
        for (n, session_id) in parked_due(&state, now).iter().enumerate() {
            if n > 0 {
                tokio::time::sleep(RESUME_STAGGER).await;
            }
            resume_parked(&state, session_id).await;
        }
        let wake = parked_next(&state).map_or(SCHEDULER_POLL, |at| {
            Duration::from_secs_f64(at.duration_since(now).as_secs_f64().max(0.01))
        });
        tokio::select! {
            () = state.limits_notify.notified() => {},
            () = tokio::time::sleep(wake.min(SCHEDULER_POLL)) => {},
        }
    }
}

/// Every registry-parked session whose `resume_at` has passed, in
/// session-id order — stable, so the stagger is deterministic.
fn parked_due(state: &AppState, now: jiff::Timestamp) -> Vec<String> {
    state
        .registry
        .lock()
        .expect("registry poisoned")
        .iter()
        .filter(|(_, entry)| {
            entry
                .rate_limited
                .as_ref()
                .and_then(|parked| parked.resume_at.parse::<jiff::Timestamp>().ok())
                .is_some_and(|at| at <= now)
        })
        .map(|(session_id, _)| session_id.clone())
        .collect()
}

/// The earliest `resume_at` still parked — the scheduler's next wake.
fn parked_next(state: &AppState) -> Option<jiff::Timestamp> {
    state
        .registry
        .lock()
        .expect("registry poisoned")
        .iter()
        .filter_map(|(_, entry)| {
            entry
                .rate_limited
                .as_ref()?
                .resume_at
                .parse::<jiff::Timestamp>()
                .ok()
        })
        .min()
}

/// Resume one parked session in its own ACP session: the same client
/// while the process lives, `session/load` on a fresh process when it is
/// gone. The parked flag is consumed on the attempt — a session that
/// cannot resume reports `failed`, a dead one stays registered for a
/// manual `adopt`. A draining daemon leaves parked sessions alone: they
/// hand off to the next daemon, whose scheduler owns the resume.
async fn resume_parked(state: &Arc<AppState>, session_id: &str) {
    if state.draining() {
        return;
    }
    if let Ok(sub) = state.get(session_id) {
        let parked = {
            let mut inner = sub.rt.inner.lock().expect("inner poisoned");
            if matches!(inner.status, Status::RateLimited { .. }) {
                inner.status = Status::Idle;
                true
            } else {
                false
            }
        };
        if parked {
            clear_parked(state, session_id).await;
            debug!(%session_id, "resuming rate-limited session");
            if let Err(error) = start_turn(state.clone(), &sub, RESUME_PROMPT.to_string()).await {
                warn!(%session_id, %error, "rate-limited resume failed");
                sub.rt.transition(Status::Failed(format!(
                    "resume after rate limit failed: {error}"
                )));
            }
        } else if !matches!(
            sub.rt.inner.lock().expect("inner poisoned").status,
            Status::Exited(_)
        ) {
            // A live session that is neither parked nor a dead husk: the
            // flag is stale — the session moved on without the schedule.
            clear_parked(state, session_id).await;
        }
        return;
    }
    // The process is gone — resume the same ACP session on a fresh one
    // with `session/load`.
    let entry = state
        .registry
        .lock()
        .expect("registry poisoned")
        .get(session_id)
        .cloned();
    let Some(entry) = entry else {
        return;
    };
    let config = match state.load_config() {
        Ok(config) => config,
        Err(error) => {
            warn!(%session_id, %error, "cannot load config to resume rate-limited session");
            clear_parked(state, session_id).await;
            return;
        }
    };
    debug!(%session_id, agent = %entry.agent, "resuming rate-limited session via session/load");
    let result = launch(
        state,
        &config,
        Launch {
            agent: entry.agent,
            cwd: entry.cwd,
            prompt: Some(RESUME_PROMPT.to_string()),
            mode: entry.mode,
            model: entry.model,
            config: BTreeMap::new(),
            permission: None,
            load: Some(session_id.to_string()),
            // The registered owner carries over: a resumed session is
            // reaped when its coordinator exits, same as a fresh spawn.
            owner: entry.owner,
        },
    )
    .await;
    if let Err(error) = result {
        warn!(%session_id, %error, "rate-limited resume failed");
        clear_parked(state, session_id).await;
    }
}

/// Consume the registry's parked flag for a session — the resume
/// schedule does not fire twice.
async fn clear_parked(state: &Arc<AppState>, session_id: &str) {
    let result = state
        .update_registry(|registry| {
            if let Some(entry) = registry.get_mut(session_id) {
                entry.rate_limited = None;
            }
        })
        .await;
    if let Err(error) = result {
        warn!(%error, "registry persist failed");
    }
}
