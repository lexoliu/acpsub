//! Rolling transcript and registry retention.
//!
//! Every session that is not live and whose transcript has not been
//! written inside `[defaults] transcript_retention` loses both its
//! `<transcript_dir>/<id>.jsonl` and its registry entry. "Not live" means
//! no daemon holds the session — a running turn or a pending permission
//! is always live — and not owed anything by one: not in `live`, not
//! claimed by an in-flight `session/load` (`reserved`), not owned across
//! a generation boundary (`foreign`, `shipped`, `resuming`), and not
//! carrying a `resume` or `rate_limited` mark, which is a resume the
//! registry still owes it. Transcript files with no registry entry
//! expire on their own mtime.
//!
//! A pass runs when the daemon starts and then every [`PRUNE_INTERVAL`];
//! a superseded daemon skips its passes — the newest generation owns
//! `registry.json` — and resumes them if it reclaims the socket.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime};

use tracing::{info, warn};

use crate::error::{Error, Result};
use crate::state::AppState;

/// The default retention window: a non-live session whose transcript was
/// last written more than this long ago is pruned.
pub const TRANSCRIPT_RETENTION: Duration = Duration::from_secs(3 * 24 * 60 * 60);

/// How often the retention task prunes after the startup pass.
const PRUNE_INTERVAL: Duration = Duration::from_secs(60 * 60);

/// What one pass freed.
#[derive(Debug, Default)]
pub struct Pruned {
    /// Sessions whose transcript file and registry entry were removed.
    pub sessions: usize,
    /// Transcript files with no registry entry removed.
    pub orphans: usize,
    /// Transcript bytes freed.
    pub bytes: u64,
}

/// The retention task: a pass at startup — `interval`'s first tick is
/// immediate — then one every [`PRUNE_INTERVAL`], until the process
/// ends.
pub async fn run(state: Arc<AppState>) {
    let mut timer = tokio::time::interval(PRUNE_INTERVAL);
    loop {
        timer.tick().await;
        if state.handover.superseded.load(Ordering::Relaxed) {
            continue;
        }
        if let Err(error) = pass(&state).await {
            warn!(%error, "transcript retention pass failed");
        }
    }
}

/// One retention pass.
///
/// A registry entry expires when its transcript's mtime — or, with no
/// transcript, its `last_turn`/`created` — is older than
/// `state.transcript_retention`; a file with no entry expires on its own
/// mtime. Entry removals ride [`AppState::update_registry`], so the pass
/// is never a second registry writer, and a failure on either side is
/// self-healing: a lost transcript leaves the entry to its timestamps on
/// the next pass, a lost entry leaves the file an orphan.
///
/// # Errors
///
/// Returns an error if the transcript directory cannot be read or the
/// registry update cannot be persisted. A file that resists removal is
/// logged and skipped.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub async fn pass(state: &AppState) -> Result<Pruned> {
    let mut pruned = Pruned::default();
    // `now - window` unrepresentable means the window spans all recorded
    // time: nothing can be old enough.
    if let Some(cutoff) = SystemTime::now().checked_sub(state.transcript_retention) {
        let protected = protected(state);
        let (candidates, registered) = candidates(state, &protected);
        let (sessions, bytes) = prune_registered(state, cutoff, candidates).await?;
        pruned.sessions = sessions;
        pruned.bytes += bytes;
        let (orphans, bytes) = prune_orphans(state, cutoff, &registered, &protected).await?;
        pruned.orphans = orphans;
        pruned.bytes += bytes;
    }
    info!(
        sessions = pruned.sessions,
        orphans = pruned.orphans,
        bytes = pruned.bytes,
        "transcript retention pass"
    );
    Ok(pruned)
}

/// The registry's prune candidates: every non-live session, excluding
/// entries that carry a `resume` or `rate_limited` mark — a resume the
/// daemon still owes the session. Each comes with the entry's own
/// last-activity stamp (`last_turn`, else `created`) for the no-
/// transcript fallback. `registered` is the full id set for the orphan
/// scan.
fn candidates(
    state: &AppState,
    protected: &HashSet<String>,
) -> (Vec<(String, Option<jiff::Timestamp>)>, HashSet<String>) {
    let registry = state.registry.lock().expect("registry poisoned");
    let mut candidates = Vec::new();
    for (session_id, entry) in registry.iter() {
        if protected.contains(session_id) || entry.resume.is_some() || entry.rate_limited.is_some()
        {
            continue;
        }
        let stamped = entry
            .last_turn
            .as_deref()
            .unwrap_or(&entry.created)
            .parse::<jiff::Timestamp>()
            .ok();
        candidates.push((session_id.clone(), stamped));
    }
    let registered = registry.iter().map(|(id, _)| id.clone()).collect();
    drop(registry);
    (candidates, registered)
}

/// Remove the transcript and registry entry of every expired candidate;
/// returns the count and freed bytes.
async fn prune_registered(
    state: &AppState,
    cutoff: SystemTime,
    candidates: Vec<(String, Option<jiff::Timestamp>)>,
) -> Result<(usize, u64)> {
    let mut expired = Vec::new();
    let mut bytes = 0;
    for (session_id, stamped) in candidates {
        let path = state.transcript_dir.join(format!("{session_id}.jsonl"));
        let (gone, size) = match tokio::fs::metadata(&path).await {
            Ok(meta) => (
                meta.modified().is_ok_and(|mtime| mtime <= cutoff),
                meta.len(),
            ),
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                (stamped.is_some_and(|ts| SystemTime::from(ts) <= cutoff), 0)
            }
            Err(source) => {
                warn!(%session_id, %source, "cannot stat transcript");
                continue;
            }
        };
        // An adopt may have claimed the session while this pass awaited
        // the stat; re-read the protected set at the point of removal.
        if !gone || protected(state).contains(&session_id) {
            continue;
        }
        match tokio::fs::remove_file(&path).await {
            Ok(()) => bytes += size,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                warn!(%session_id, %source, "cannot remove transcript");
            }
        }
        expired.push(session_id);
    }
    let sessions = expired.len();
    if !expired.is_empty() {
        state
            .update_registry(|registry| {
                for session_id in &expired {
                    registry.remove(session_id);
                }
            })
            .await?;
    }
    Ok((sessions, bytes))
}

/// Remove every expired `*.jsonl` transcript with no registry entry;
/// returns the count and freed bytes. `protected` keeps a live or
/// in-flight launch's transcript safe even when its entry is not
/// written yet.
async fn prune_orphans(
    state: &AppState,
    cutoff: SystemTime,
    registered: &HashSet<String>,
    protected: &HashSet<String>,
) -> Result<(usize, u64)> {
    let read = |source| {
        Error::io(
            format!("cannot read {}", state.transcript_dir.display()),
            source,
        )
    };
    let mut dir = match tokio::fs::read_dir(&state.transcript_dir).await {
        Ok(dir) => dir,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Ok((0, 0));
        }
        Err(source) => return Err(read(source)),
    };
    let mut orphans = 0;
    let mut bytes = 0;
    while let Some(entry) = dir.next_entry().await.map_err(read)? {
        let Some(stem) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.strip_suffix(".jsonl"))
            .map(str::to_string)
        else {
            continue;
        };
        if registered.contains(&stem) || protected.contains(&stem) {
            continue;
        }
        let Ok(meta) = entry.metadata().await else {
            continue;
        };
        let expired = meta.is_file() && meta.modified().is_ok_and(|mtime| mtime <= cutoff);
        if !expired {
            continue;
        }
        match tokio::fs::remove_file(entry.path()).await {
            Ok(()) => {
                orphans += 1;
                bytes += meta.len();
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                warn!(session_id = %stem, %source, "cannot remove orphaned transcript");
            }
        }
    }
    Ok((orphans, bytes))
}

/// The session ids no pass may touch: this daemon's live sessions and
/// in-flight `session/load` launches, plus every session a generation
/// boundary still owns.
///
/// `live` is locked before `reserved`, their documented order; the
/// handover sets are leaf locks taken one at a time.
fn protected(state: &AppState) -> HashSet<String> {
    let live = state.live.lock().expect("live poisoned");
    let reserved = state.reserved.lock().expect("reserved poisoned");
    let mut set: HashSet<String> = live.keys().chain(reserved.keys()).cloned().collect();
    drop(reserved);
    drop(live);
    set.extend(
        state
            .handover
            .foreign
            .lock()
            .expect("foreign poisoned")
            .keys()
            .cloned(),
    );
    set.extend(
        state
            .handover
            .shipped
            .lock()
            .expect("shipped poisoned")
            .iter()
            .cloned(),
    );
    set.extend(
        state
            .handover
            .resuming
            .lock()
            .expect("resuming poisoned")
            .iter()
            .cloned(),
    );
    set
}
