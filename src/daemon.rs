//! The daemon: `acpsub` state served over a Unix socket.
//!
//! `acpsub daemon` is the session host for the CLI client commands
//! (`spawn`, `send`, `wait`, …). It holds every live ACP child process
//! and accepts one JSON-RPC/MCP connection per client invocation. Unlike
//! `serve`, it is not tied to a host's stdio and survives the
//! orchestrating agent's restarts — sessions stay adoptable through the
//! registry either way.
//!
//! An orphan reaper runs alongside the accept loop: a subagent that was
//! spawned with an `owner` pid is closed once that pid is dead, so a
//! session cannot keep running — and spending — after its coordinator
//! exited.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use aither_mcp::McpServer;
use aither_mcp::transport::StreamTransport;
use futures_lite::io::BufReader;
use tracing::{debug, error, info, warn};

use crate::error::{Error, Result};
use crate::ratelimit;
use crate::registry::RegistryEntry;
use crate::state::{
    AppState, Launch, ResumeFailure, ResumeReport, Status, Subagent, close_sub, launch, start_turn,
};
use crate::tools::build_tools;

/// How often the reaper checks owner pids.
const OWNER_CHECK: Duration = Duration::from_secs(2);
/// How often the drain task re-checks for running turns: a backstop for
/// the `progressed` notifications turn ends and live-map removals send.
const DRAIN_POLL: Duration = Duration::from_millis(100);

/// The default daemon socket: `~/.local/share/acpsub/daemon.sock`.
#[must_use]
pub fn default_socket_path() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".local/share/acpsub/daemon.sock"))
}

/// Serve the tool set over `socket_path` until the process is killed or a
/// `daemon/drain` call asks it to restart.
///
/// A stale socket file left by a dead daemon is unlinked and rebound; a
/// socket a live daemon still answers fails with [`Error::DaemonRunning`].
///
/// Before accepting clients the daemon re-adopts every session the
/// previous daemon marked for resume — a client that connects in the gap
/// waits in the socket backlog until resuming finishes. On a drain it
/// hands off: every live session's registry entry is marked, then the
/// sessions close and the process exits.
///
/// # Errors
///
/// Returns an error if the socket cannot be bound or its permissions set.
pub async fn run(state: Arc<AppState>, socket_path: &Path) -> Result<()> {
    let listener = bind(socket_path).await?;
    // The pid file lets a caller stop exactly this daemon:
    // `kill "$(cat daemon.sock.pid)"`.
    std::fs::write(
        socket_path.with_extension("sock.pid"),
        std::process::id().to_string(),
    )
    .map_err(|source| Error::io("cannot write daemon pid file", source))?;
    resume_marked(&state).await;
    tokio::spawn(reap_orphans(state.clone()));
    loop {
        if state.drain.quiesced.load(Ordering::Relaxed) {
            break;
        }
        tokio::select! {
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _)) => {
                        let state = state.clone();
                        tokio::spawn(async move {
                            if let Err(error) = serve_connection(stream, state).await {
                                debug!(%error, "client connection closed with error");
                            }
                        });
                    }
                    Err(error) => {
                        error!(%error, "socket accept failed");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
            () = state.drain.wake.notified() => {}
        }
    }
    handoff(&state).await;
    Ok(())
}

/// The drain task the `daemon/drain` tool spawns: wait until no turn is
/// running anywhere — queued prompts chain through the normal turn-end
/// path, so the wait covers them — then flag quiescence so the accept
/// loop hands off and exits. A `needs_permission` session is an in-flight
/// turn too: `permit` keeps working through the drain to unblock it.
pub(crate) async fn drain_sessions(state: Arc<AppState>) {
    info!("drain requested; waiting for in-flight turns to finish");
    loop {
        let running = state
            .live
            .lock()
            .expect("live poisoned")
            .values()
            .any(|sub| {
                let inner = sub.rt.inner.lock().expect("inner poisoned");
                matches!(inner.status, Status::Running | Status::NeedsPermission)
            });
        if !running {
            break;
        }
        tokio::select! {
            () = state.drain.progressed.notified() => {}
            () = tokio::time::sleep(DRAIN_POLL) => {}
        }
    }
    info!("all turns finished; closing sessions for restart");
    state.drain.quiesced.store(true, Ordering::Relaxed);
    state.drain.wake.notify_waiters();
}

/// The drain's handoff: close every live session, marking each registry
/// entry for the next daemon to resume — the entry keeps its agent,
/// model, mode, cwd and owner, and carries the session's accepted queued
/// prompts so they still run after the restart. Sessions a caller closed
/// mid-drain are already gone from the live map and stay unmarked.
async fn handoff(state: &Arc<AppState>) {
    let subs: Vec<Arc<Subagent>> = state
        .live
        .lock()
        .expect("live poisoned")
        .values()
        .cloned()
        .collect();
    for sub in subs {
        let queued: Vec<String> = sub
            .rt
            .inner
            .lock()
            .expect("inner poisoned")
            .queue
            .iter()
            .cloned()
            .collect();
        let session_id = sub.session_id.clone();
        let result = state
            .update_registry(|registry| {
                if let Some(entry) = registry.get_mut(&session_id) {
                    entry.resume = Some(crate::registry::Resume { queued });
                }
            })
            .await;
        if let Err(error) = result {
            warn!(%session_id, %error, "registry persist failed");
        }
        state
            .live
            .lock()
            .expect("live poisoned")
            .remove(&session_id);
        close_sub(&sub).await;
    }
}

/// Re-adopt every session the previous daemon marked for resume, before
/// the listener accepts clients: the same ids, owner, model, mode and
/// cwd, with accepted queued prompts replayed and a still-parked
/// session's rate-limit schedule restored. A failed `session/load` is
/// reported, not dropped — the registry entry and its resume mark stay,
/// so a later `adopt` or restart can retry.
async fn resume_marked(state: &Arc<AppState>) {
    let marked: Vec<(String, RegistryEntry)> = state
        .registry
        .lock()
        .expect("registry poisoned")
        .iter()
        .filter(|(_, entry)| entry.resume.is_some())
        .map(|(session_id, entry)| (session_id.clone(), entry.clone()))
        .collect();
    let mut report = ResumeReport::default();
    if !marked.is_empty() {
        match state.load_config() {
            Ok(config) => {
                for (session_id, entry) in marked {
                    resume_one(state, &config, session_id, entry, &mut report).await;
                }
            }
            Err(error) => {
                for (session_id, _) in marked {
                    report.failed.push(ResumeFailure {
                        session_id,
                        error: error.to_string(),
                    });
                }
            }
        }
    }
    if !report.resumed.is_empty() || !report.failed.is_empty() {
        info!(
            resumed = report.resumed.len(),
            failed = report.failed.len(),
            "session resume pass finished"
        );
    }
    *state.resume_report.lock().expect("resume report poisoned") = report;
}

/// Re-adopt one marked session on a fresh agent process, then restore
/// what the registry carried: the rate-limit schedule for a parked
/// session and the send-queue the drain could not run — kicked with the
/// same gate a fresh prompt sees, so a still-active limit re-parks it in
/// order instead of burning a request.
async fn resume_one(
    state: &Arc<AppState>,
    config: &crate::config::Config,
    session_id: String,
    entry: RegistryEntry,
    report: &mut ResumeReport,
) {
    let queued = entry
        .resume
        .as_ref()
        .map_or_else(Vec::new, |resume| resume.queued.clone());
    let result = launch(
        state,
        config,
        Launch {
            agent: entry.agent.clone(),
            cwd: entry.cwd.clone(),
            prompt: None,
            mode: entry.mode.clone(),
            model: entry.model.clone(),
            config: BTreeMap::new(),
            permission: None,
            load: Some(session_id.clone()),
            // The registered owner carries over: a resumed session is
            // reaped when its coordinator exits, same as a fresh spawn.
            owner: entry.owner,
        },
    )
    .await;
    let sub = match result {
        Ok(sub) => sub,
        Err(error) => {
            warn!(%session_id, %error, "session resume failed");
            report.failed.push(ResumeFailure {
                session_id,
                error: error.to_string(),
            });
            return;
        }
    };
    // A session parked at handoff keeps its schedule: `register_session`
    // consumed the flag, so restore it and re-park the status — the
    // scheduler resumes the session on its own when the quota lifts.
    if let Some(parked) = &entry.rate_limited
        && let Ok(resume_at) = parked.resume_at.parse::<jiff::Timestamp>()
    {
        sub.rt.inner.lock().expect("inner poisoned").status = Status::RateLimited {
            resume_at,
            reason: parked.reason.clone(),
        };
        let parked = parked.clone();
        let result = state
            .update_registry(|registry| {
                if let Some(entry) = registry.get_mut(&session_id) {
                    entry.rate_limited = Some(parked);
                }
            })
            .await;
        if let Err(error) = result {
            warn!(%session_id, %error, "registry persist failed");
        }
    }
    let first = {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        inner.queue.extend(queued);
        if inner.status.accepts_prompt() {
            inner.queue.pop_front()
        } else {
            None
        }
    };
    if let Some(prompt) = first {
        if let Some(limit) = ratelimit::active_limit(state, &sub.agent) {
            ratelimit::repark_prompt(state, &sub, &prompt, &limit).await;
        } else if let Err(error) = start_turn(state.clone(), &sub, prompt).await {
            warn!(%session_id, %error, "queued prompt failed to start after resume");
        }
    }
    report.resumed.push(session_id);
}

/// Bind the socket, reclaiming a stale one left by a dead daemon.
async fn bind(path: &Path) -> Result<async_net::unix::UnixListener> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|source| Error::io(format!("cannot create {}", parent.display()), source))?;
    }
    match async_net::unix::UnixListener::bind(path) {
        Ok(listener) => {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(|source| Error::io(format!("cannot chmod {}", path.display()), source))?;
            Ok(listener)
        }
        Err(bind_error) => {
            // A leftover socket file: probe it — if nothing answers, it is
            // stale and safe to replace.
            if async_net::unix::UnixStream::connect(path).await.is_ok() {
                return Err(Error::DaemonRunning(path.to_path_buf()));
            }
            std::fs::remove_file(path).map_err(|source| {
                Error::io(
                    format!("cannot remove stale socket {}", path.display()),
                    source,
                )
            })?;
            async_net::unix::UnixListener::bind(path)
                .map_err(|_| Error::io(format!("cannot bind {}", path.display()), bind_error))
        }
    }
}

/// One client connection: a fresh tool set over the shared state, served
/// until the client hangs up. The daemon adds its restart tools —
/// `daemon/drain` and `daemon/resumed` — on top of the shared set.
async fn serve_connection(
    stream: async_net::unix::UnixStream,
    state: Arc<AppState>,
) -> std::result::Result<(), aither_mcp::McpError> {
    let (read_half, write_half) = futures_lite::io::split(stream);
    let transport = StreamTransport::new(BufReader::new(read_half), write_half);
    let mut tools = build_tools(state.clone()).map_err(|error| {
        aither_mcp::McpError::InvalidConfig(format!("cannot build tools: {error}"))
    })?;
    crate::tools::register_daemon_tools(&mut tools, state).map_err(|error| {
        aither_mcp::McpError::InvalidConfig(format!("cannot build tools: {error}"))
    })?;
    let mut server = McpServer::new(transport, tools, "acpsub", env!("CARGO_PKG_VERSION"));
    server.run().await
}

/// Reap live subagents whose owning coordinator pid has exited.
async fn reap_orphans(state: Arc<AppState>) {
    loop {
        tokio::time::sleep(OWNER_CHECK).await;
        let orphans: Vec<(String, Arc<Subagent>)> = state
            .live
            .lock()
            .expect("live poisoned")
            .iter()
            .filter(|(_, sub)| sub.owner_dead())
            .map(|(id, sub)| (id.clone(), sub.clone()))
            .collect();
        for (session_id, sub) in orphans {
            let owner = sub.rt.owner.load(std::sync::atomic::Ordering::Relaxed);
            warn!(%session_id, owner, "coordinator pid is gone; closing subagent");
            state
                .live
                .lock()
                .expect("live poisoned")
                .remove(&session_id);
            close_sub(&sub).await;
        }
    }
}
