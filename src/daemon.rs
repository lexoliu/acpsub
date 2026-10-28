//! The daemon: `acpsub` state served over a Unix socket.
//!
//! `acpsub daemon` is the session host for the CLI client commands
//! (`spawn`, `send`, `wait`, …). It holds every live ACP child process
//! and accepts one JSON-RPC/MCP connection per client invocation. Unlike
//! `serve`, it is not tied to a host's stdio and survives the
//! orchestrating agent's restarts — sessions stay adoptable through the
//! registry either way.
//!
//! A watchdog feeds off `wait` calls: a running turn nobody is waiting
//! on is cancelled and its session closed once no `wait`/`wait_any`
//! caller has fed it for the grace period, so a turn cannot keep running
//! — and spending — after every caller walked away.
//!
//! A retention task prunes what the watchdog closes: a session that is
//! not live and whose transcript has not been written inside
//! `[defaults] transcript_retention` loses its transcript file and its
//! registry entry, at startup and then hourly. Only the newest
//! generation prunes — a superseded daemon owns neither the registry
//! file nor the sessions it handed off.
//!
//! # Restart: hand over the listener, not the turns
//!
//! Every daemon also binds a private generation socket
//! `daemon.<pid>.sock` beside `daemon.sock`. A new daemon takes the
//! public socket over — it binds a temp path and atomically renames it
//! onto `daemon.sock`, so connections the previous daemon already holds
//! (a blocked `wait`, say) keep working — then calls `daemon/takeover` on
//! the previous generation's private socket. The superseded daemon stops
//! persisting `registry.json`, hands off every session without a turn in
//! flight at once, and answers with the sessions it still owns. Those
//! stay live: calls for them on the new daemon forward to the
//! superseded one's private socket until a `daemon/handoff` (its turn
//! ended — the session is resumed here) or `daemon/release` (the session
//! ended for good) lands. The superseded daemon exits once it owns
//! nothing; its turn watchdogs keep working until then.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use aither_core::llm::tool::{Progress, ProgressSink, Tool, ToolContext, ToolResult, Tools};
use aither_mcp::McpServer;
use aither_mcp::protocol::{
    CallToolParams, CallToolResult, Content, InitializeParams, JsonRpcMessage, JsonRpcNotification,
    JsonRpcRequest, ProgressNotificationParams, ProgressToken, RequestId, RequestMeta,
};
use aither_mcp::transport::{BidirectionalTransport, StreamTransport, Transport};
use futures_lite::io::BufReader;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{debug, error, info, warn};

use crate::error::{Error, Result};
use crate::ratelimit;
use crate::registry::{Registry, RegistryEntry, Resume};
use crate::state::{
    AppState, Closing, HeldLock, Inner, Launch, ResumeFailure, ResumeReport, Status, Subagent,
    close_sub, launch, start_turn,
};
use crate::tools::build_tools;

/// Upper bound on dispatch reclassification while a session moves
/// between generations mid-call.
const DISPATCH_ATTEMPTS: usize = 8;

/// The default daemon socket: `~/.local/share/acpsub/daemon.sock`.
#[must_use]
pub fn default_socket_path() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".local/share/acpsub/daemon.sock"))
}

/// The private generation socket next to `socket_path`:
/// `daemon.<pid>.sock` for `daemon.sock`.
fn private_socket_path(socket_path: &Path, pid: u32) -> PathBuf {
    let stem = socket_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("daemon");
    socket_path.with_file_name(format!("{stem}.{pid}.sock"))
}

/// A session the superseded daemon still owns (or still forwards to an
/// even older generation): calls for it go to `socket` until it hands
/// off or releases.
#[derive(Debug, Serialize, Deserialize)]
struct KeptSession {
    /// Session id.
    session_id: String,
    /// Private socket of the daemon that owns it.
    socket: PathBuf,
    /// The owner's current registry entry, when it has one — applied so
    /// this daemon's registry view stays honest for `list`/`status`.
    entry: Option<RegistryEntry>,
}

/// A session handed off at takeover: the entry carries the resume mark
/// (and any accepted queued prompts) that [`resume_marked`] then loads.
#[derive(Debug, Serialize, Deserialize)]
struct HandedSession {
    /// Session id.
    session_id: String,
    /// Registry entry with `resume` set.
    entry: RegistryEntry,
}

/// `daemon/takeover`'s answer: the partition of its sessions at the
/// moment the newer daemon took the public socket.
#[derive(Debug, Serialize, Deserialize)]
struct TakeoverReport {
    /// Still owned — in-flight turn, launch, or forwarded from an older
    /// generation. They hand off one by one as they settle.
    kept: Vec<KeptSession>,
    /// Handed off immediately — no turn in flight.
    handed: Vec<HandedSession>,
}

/// `daemon/handoff`'s arguments: a kept session whose turn ended.
#[derive(Debug, Deserialize, JsonSchema)]
struct HandoffArgs {
    /// Session id being handed off.
    session_id: String,
    /// Registry entry with `resume` set — turn count, `last_turn`, the
    /// parked send-queue — applied here before the resume.
    entry: RegistryEntry,
}

/// `daemon/release`'s arguments: an owned session that ended without a
/// handoff (closed, forgotten, watchdog-cancelled, or a failed launch).
#[derive(Debug, Deserialize, JsonSchema)]
struct ReleaseArgs {
    /// Session id released.
    session_id: String,
    /// The owner's final registry entry, or `null` when the session was
    /// forgotten — the entry is removed here instead of replaced.
    entry: Option<RegistryEntry>,
}

/// `daemon/takeover`'s arguments.
#[derive(Debug, Deserialize, JsonSchema)]
struct TakeoverArgs {
    /// The superseding daemon's private socket — handoff and release
    /// reports go there.
    socket: PathBuf,
}

/// Serve the tool set over `socket_path` until the process is killed or
/// a newer daemon takes the socket over.
///
/// Two locks order daemon starts. The succession lock
/// (`daemon.sock.next`) is held from start until this daemon owns the
/// socket, so two starts never interleave — a second one fails with
/// [`Error::DaemonRunning`] before touching anything. The claim lock on
/// the pid file is held for as long as a daemon owns the socket: when it
/// is free this is a fresh start; when a live daemon holds it, that
/// daemon is superseded — the public socket is taken by an atomic rename
/// and the takeover protocol hands its sessions over, releasing its
/// claim to this daemon.
///
/// Before accepting clients the daemon finishes the takeover and
/// re-adopts every session marked for resume — a client that connects
/// in the gap waits in the socket backlog until resuming finishes.
///
/// # Errors
///
/// Returns an error if the socket cannot be bound or its permissions set.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub async fn run(state: Arc<AppState>, socket_path: &Path) -> Result<()> {
    let succession = lock_succession(socket_path)?;
    let private_path = private_socket_path(socket_path, std::process::id());
    // The private listener binds first so the takeover's handoff reports
    // have somewhere to land.
    let private_listener = bind(&private_path).await?;
    *state
        .handover
        .private_socket
        .lock()
        .expect("private socket poisoned") = Some(private_path.clone());
    // Any failure from here on exits this daemon. A predecessor this
    // daemon already superseded sees its successor gone and takes the
    // socket back.
    let served = serve(&state, socket_path, succession, &private_listener).await;
    let _ = std::fs::remove_file(&private_path);
    served
}

/// Lead the socket, then accept on it and on the private listener until
/// this daemon is superseded and owns nothing. A successor that dies
/// while this daemon still owns sessions makes it lead again.
async fn serve(
    state: &Arc<AppState>,
    socket_path: &Path,
    succession: HeldLock,
    private_listener: &async_net::unix::UnixListener,
) -> Result<()> {
    let mut listener = lead(state, socket_path, succession).await?;
    info!(socket = %socket_path.display(), "daemon listening");
    resume_marked(state).await;
    // Spawned only after `lead`: the takeover has populated `foreign`
    // and the resume pass its marks, so the first retention pass already
    // sees every session another generation still owns.
    tokio::spawn(crate::prune::run(state.clone()));
    let mut finished = state.handover.finished.subscribe();
    loop {
        if *finished.borrow() {
            break;
        }
        tokio::select! {
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _)) => {
                        let state = state.clone();
                        tokio::spawn(async move {
                            if let Err(error) = serve_connection(stream, state, false).await {
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
            accepted = private_listener.accept() => {
                match accepted {
                    Ok((stream, _)) => {
                        let state = state.clone();
                        tokio::spawn(async move {
                            if let Err(error) = serve_connection(stream, state, true).await {
                                debug!(%error, "daemon connection closed with error");
                            }
                        });
                    }
                    Err(error) => {
                        error!(%error, "private socket accept failed");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
            changed = finished.changed() => {
                if changed.is_err() {
                    break;
                }
            }
            () = state.handover.orphaned.notified() => {
                // Another start may be mid-succession (a client found the
                // dead socket and started a daemon): wait for its lock,
                // then lead past it.
                let succession = wait_succession(socket_path).await?;
                listener = reclaim(state, socket_path, succession).await?;
                info!(socket = %socket_path.display(), "successor died; daemon listening again");
            }
        }
    }
    // Superseded: the claim and the pid file are the successor's now.
    drop(listener);
    Ok(())
}

/// Become the daemon behind `socket_path`. Take the public socket over
/// atomically: bind a temp sibling and rename it onto `socket_path`.
/// Connections the previous daemon already holds keep working on its
/// bound socket. Every new connect reaches this daemon, which accepts
/// once the takeover and the resume pass are done. If a live daemon holds
/// the claim, supersede it. Then hold the claim, name this daemon in the
/// pid file, and release `succession`.
async fn lead(
    state: &Arc<AppState>,
    socket_path: &Path,
    succession: HeldLock,
) -> Result<async_net::unix::UnixListener> {
    let pid = std::process::id();
    let claim = open_claim(socket_path).await?;
    let tmp = socket_path.with_file_name(format!(".acpsub-{pid}.tmp"));
    let listener = bind(&tmp).await?;
    if let Err(source) = std::fs::rename(&tmp, socket_path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::io(
            format!("cannot take over {}", socket_path.display()),
            source,
        ));
    }
    let mut claim = match claim {
        Claim::Free(claim) => claim,
        Claim::Held { file, prev } => {
            supersede(state, &prev).await?;
            // The predecessor released its claim (`LOCK_UN`) before it
            // answered; the succession lock keeps every other start away
            // from it.
            HeldLock::try_new(file).map_err(|(_, error)| Error::Internal {
                session_id: String::new(),
                detail: format!("the superseded daemon kept its claim: {error}"),
            })?
        }
    };
    // The public socket accepts ⇒ the pid file names its daemon.
    let file = claim.file();
    file.set_len(0)
        // Reading the predecessor's pid moved the cursor.
        .and_then(|()| std::io::Seek::rewind(file))
        .and_then(|()| std::io::Write::write_all(file, pid.to_string().as_bytes()))
        .map_err(|source| Error::io("cannot write daemon pid file", source))?;
    *state.handover.claim.lock().expect("claim poisoned") = Some(claim);
    drop(succession);
    Ok(listener)
}

/// A superseded daemon whose successor died leads the socket again, as
/// the newest generation. It stops relaying to the dead successor and
/// writes the registry again, taking the dead successor's
/// `registry.json` as the record of every session it does not own
/// itself. Then it takes the socket back: it holds the claim if the claim
/// is free, or supersedes whichever daemon took it meanwhile. Sends that
/// queued while it was superseded start now.
async fn reclaim(
    state: &Arc<AppState>,
    socket_path: &Path,
    succession: HeldLock,
) -> Result<async_net::unix::UnixListener> {
    let path = state.registry_path.clone();
    let recorded = tokio::task::spawn_blocking(move || Registry::load(&path))
        .await
        .expect("registry load task panicked")?;
    let owned: HashSet<String> = {
        let live = state.live.lock().expect("live poisoned");
        let reserved = state.reserved.lock().expect("reserved poisoned");
        let foreign = state.handover.foreign.lock().expect("foreign poisoned");
        live.keys()
            .chain(reserved.keys())
            .chain(foreign.keys())
            .cloned()
            .collect()
    };
    *state.handover.next.lock().expect("next poisoned") = None;
    // What this daemon shipped died with the successor: those sessions
    // are registered, not live, like after any crash.
    state
        .handover
        .shipped
        .lock()
        .expect("shipped poisoned")
        .clear();
    state
        .update_registry(|registry| {
            let ours = std::mem::replace(registry, recorded);
            for (session_id, entry) in ours.iter() {
                if owned.contains(session_id) {
                    registry.insert(session_id.clone(), entry.clone());
                }
            }
            // Cleared under `registry_write`, before `update_registry`
            // reads the flag: this merge is the first write persisted
            // again, and no mutation serialized after it is dropped.
            state.handover.superseded.store(false, Ordering::Relaxed);
        })
        .await?;
    let listener = lead(state, socket_path, succession).await?;
    // Parked sessions were left to the dead successor's scheduler.
    state.limits_notify.notify_waiters();
    let subs: Vec<Arc<Subagent>> = state
        .live
        .lock()
        .expect("live poisoned")
        .values()
        .cloned()
        .collect();
    for sub in subs {
        kick_queue(state, &sub).await;
    }
    Ok(listener)
}

/// The socket's claim as this start found it.
enum Claim {
    /// Free: a fresh start, now holding it.
    Free(HeldLock),
    /// Held by the live daemon this start supersedes, named by its
    /// private socket. The file is the claim to take once it lets go.
    Held {
        /// The opened, unlocked pid file.
        file: std::fs::File,
        /// The holder's private socket.
        prev: PathBuf,
    },
}

/// Open the socket's claim — its pid file — and try its lock. A free
/// lock is a fresh start; a held one names the live daemon this one
/// supersedes, by its private socket.
///
/// # Errors
///
/// Returns [`Error::DaemonRunning`] when the claim is free yet the socket
/// still answers: a daemon that never took the claim — one built before
/// it existed — whose socket the takeover rename would steal, stranding
/// its sessions.
async fn open_claim(socket_path: &Path) -> Result<Claim> {
    let pid_path = socket_path.with_extension("sock.pid");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&pid_path)
        .map_err(|source| Error::io("cannot open daemon pid file", source))?;
    match HeldLock::try_new(file) {
        Ok(claim) => {
            if async_net::unix::UnixStream::connect(socket_path)
                .await
                .is_ok()
            {
                return Err(Error::DaemonRunning(socket_path.to_path_buf()));
            }
            Ok(Claim::Free(claim))
        }
        // A held claim names the live daemon this one supersedes.
        Err((mut file, std::fs::TryLockError::WouldBlock)) => {
            let mut text = String::new();
            std::io::Read::read_to_string(&mut file, &mut text)
                .map_err(|source| Error::io("cannot read daemon pid file", source))?;
            let prev = text
                .trim()
                .parse::<u32>()
                .map_err(|_| Error::DaemonRunning(socket_path.to_path_buf()))?;
            Ok(Claim::Held {
                file,
                prev: private_socket_path(socket_path, prev),
            })
        }
        Err((_, std::fs::TryLockError::Error(source))) => {
            Err(Error::io("cannot lock daemon pid file", source))
        }
    }
}

/// Open the succession lock file for `socket_path`: `daemon.sock.next`.
fn succession_file(socket_path: &Path) -> Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(socket_path.with_extension("sock.next"))
        .map_err(|source| Error::io("cannot open daemon succession lock", source))
}

/// Hold the succession lock for `socket_path` while this daemon starts:
/// of two starts at once only one proceeds, the other fails with
/// [`Error::DaemonRunning`].
fn lock_succession(socket_path: &Path) -> Result<HeldLock> {
    match HeldLock::try_new(succession_file(socket_path)?) {
        Ok(lock) => Ok(lock),
        Err((_, std::fs::TryLockError::WouldBlock)) => {
            Err(Error::DaemonRunning(socket_path.to_path_buf()))
        }
        Err((_, std::fs::TryLockError::Error(source))) => {
            Err(Error::io("cannot lock daemon succession lock", source))
        }
    }
}

/// Take the succession lock for a running daemon that must lead again.
/// A start that holds the lock gives it up once it leads the socket, or
/// when it exits.
async fn wait_succession(socket_path: &Path) -> Result<HeldLock> {
    let file = succession_file(socket_path)?;
    tokio::task::spawn_blocking(move || HeldLock::new(file))
        .await
        .expect("succession lock task panicked")
        .map_err(|source| Error::io("cannot lock daemon succession lock", source))
}

/// The new daemon's side of the takeover: call `daemon/takeover` on the
/// previous generation's private socket and apply the partition —
/// `kept` sessions stay owned (calls for them forward to the owning
/// daemon's private socket), `handed` ones land in the registry with
/// their resume marks, which [`resume_marked`] then re-adopts.
///
/// # Errors
///
/// Returns an error when the previous daemon cannot be reached, refuses
/// the takeover, or answers with a report that does not parse.
async fn supersede(state: &Arc<AppState>, prev_socket: &Path) -> Result<()> {
    let ours = state
        .handover
        .private_socket
        .lock()
        .expect("private socket poisoned")
        .clone()
        .expect("private socket bound");
    let report: TakeoverReport =
        peer_json(prev_socket, "daemon/takeover", json!({"socket": ours})).await?;
    {
        let mut foreign = state.handover.foreign.lock().expect("foreign poisoned");
        for kept in &report.kept {
            foreign.insert(kept.session_id.clone(), kept.socket.clone());
        }
    }
    *state.handover.prev.lock().expect("prev poisoned") = Some(prev_socket.to_path_buf());
    let kept_n = report.kept.len();
    let handed_n = report.handed.len();
    // Apply the owner's registry view: kept entries refresh this
    // daemon's copy (turn starts and ends the old daemon persisted after
    // it loaded); their resume mark is stripped — the owner still runs
    // them. Handed entries carry the mark that resumes them here.
    state
        .update_registry(|registry| {
            for kept in &report.kept {
                if let Some(mut entry) = kept.entry.clone() {
                    entry.resume = None;
                    registry.insert(kept.session_id.clone(), entry);
                }
            }
            for handed in &report.handed {
                registry.insert(handed.session_id.clone(), handed.entry.clone());
            }
        })
        .await?;
    info!(
        kept = kept_n,
        handed = handed_n,
        "superseded the previous daemon"
    );
    Ok(())
}

/// A superseded daemon's watcher: hands each session off as its turn
/// (or launch) settles, reports departures, and finishes once the
/// daemon owns nothing and relays for nobody — the accept loop then
/// exits. `handed` sessions were already shipped in the takeover report;
/// they are closed here. `owned` starts as the sessions the report named
/// as kept by this daemon, so one that ends before the first scan is
/// still reported as released.
///
/// `successor` is a connection held open to the successor's private
/// socket, which closes when the successor's process ends. The successor
/// cannot finish cleanly while it still has sessions of this daemon
/// listed as foreign. So if the connection closes while this daemon
/// still owns anything, the successor died. The watcher then wakes the
/// accept loop through `orphaned` to lead the socket again, and returns.
async fn handover_watch(
    state: Arc<AppState>,
    next: PathBuf,
    successor: async_net::unix::UnixStream,
    handed: Vec<Arc<Subagent>>,
    mut owned: HashSet<String>,
) {
    for sub in &handed {
        close_sub(sub).await;
    }
    let gone = closed(successor);
    tokio::pin!(gone);
    loop {
        // Register the waiter BEFORE the scan: a settle between the scan
        // and the await still wakes us.
        let notified = state.handover.progressed.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        let mut handoffs = Vec::new();
        let mut releases = Vec::new();
        {
            let live = state.live.lock().expect("live poisoned");
            let reserved = state.reserved.lock().expect("reserved poisoned");
            for (id, sub) in live.iter() {
                let in_flight = sub.rt.inner.lock().expect("inner poisoned").in_flight()
                    || sub.rt.closing.get().is_some();
                if in_flight {
                    owned.insert(id.clone());
                } else {
                    handoffs.push(sub.clone());
                }
            }
            for id in reserved.keys() {
                owned.insert(id.clone());
            }
            owned.retain(|id| {
                if live.contains_key(id) || reserved.contains_key(id) {
                    true
                } else {
                    releases.push(id.clone());
                    false
                }
            });
            drop(live);
            drop(reserved);
        }
        for sub in handoffs {
            let session_id = sub.session_id.clone();
            handoff_one(&state, &next, sub).await;
            owned.remove(&session_id);
        }
        for session_id in releases {
            released_one(&state, &next, &session_id).await;
        }
        // Sessions this daemon relays for an older generation keep it
        // alive: their handoff and release still arrive here.
        let done = {
            state.live.lock().expect("live poisoned").is_empty()
                && state.reserved.lock().expect("reserved poisoned").is_empty()
                && state
                    .handover
                    .foreign
                    .lock()
                    .expect("foreign poisoned")
                    .is_empty()
        };
        if done {
            info!("no sessions left; superseded daemon exiting");
            let _ = state.handover.finished.send(true);
            return;
        }
        // Death first: a report the dead successor refused wakes
        // `progressed` too, and must not spin the scan.
        tokio::select! {
            biased;
            () = &mut gone => {
                warn!(successor = %next.display(), "the next daemon died; leading again");
                state.handover.orphaned.notify_one();
                return;
            }
            () = notified => {}
        }
    }
}

/// Resolve once `stream` reaches EOF or fails: its peer's end closed. The
/// peer never writes on a liveness connection, so any bytes are drained.
async fn closed(mut stream: async_net::unix::UnixStream) {
    let mut buf = [0u8; 64];
    while let Ok(n) = futures_lite::AsyncReadExt::read(&mut stream, &mut buf).await {
        if n == 0 {
            return;
        }
    }
}

/// The resume mark a handing-over daemon leaves for a session: its queued
/// sends and how its last turn ended.
fn resume_mark(inner: &Inner) -> Resume {
    Resume {
        queued: inner.queue.iter().cloned().collect(),
        failed: None,
        last_turn: inner.status.last_turn(),
    }
}

/// Ship one settled session to the next daemon: mark it `HandingOff` so
/// no turn can start mid-ship — the mark is set under the same `inner`
/// lock `begin_turn`'s gate reads it with, and the send-queue snapshot it
/// guards is taken in the same critical section — report `daemon/handoff`
/// with its final registry entry — turn count, `last_turn`, the parked
/// send-queue as the resume's `queued` — then close it. A `send` refused
/// by the mark holds until this resolves: `handover.progressed` wakes it
/// once `live` drops the session, or once a failed report clears the
/// mark. On a report failure the session stays owned: the next scan
/// retries.
async fn handoff_one(state: &Arc<AppState>, next: &Path, sub: Arc<Subagent>) {
    let session_id = sub.session_id.clone();
    let entry = {
        let mut registry = state.registry.lock().expect("registry poisoned");
        registry.get_mut(&session_id).map(|entry| {
            let mut entry = entry.clone();
            let inner = sub.rt.inner.lock().expect("inner poisoned");
            // `HandingOff` under the `inner` lock the gate checks: a send
            // serialized after this point refuses and holds; one before it
            // started its turn — which keeps the session in flight, never
            // handed over.
            sub.rt.closing.set(Closing::HandingOff);
            let mark = resume_mark(&inner);
            // A live parked session's queue is the truth: the parked
            // record's persisted copy can lag a `repark` that already
            // requeued the prompt, so the shipped record mirrors the
            // queue as it stands.
            if matches!(inner.status, Status::RateLimited { .. })
                && let Some(parked) = entry.rate_limited.as_mut()
            {
                parked.queued.clone_from(&mark.queued);
            }
            entry.resume = Some(mark);
            drop(inner);
            entry
        })
    };
    let Some(entry) = entry else {
        // Never registered — nothing to resume; just end it.
        state
            .live
            .lock()
            .expect("live poisoned")
            .remove(&session_id);
        close_sub(&sub).await;
        return;
    };
    let outcome: Result<Value> = peer_json(
        next,
        "daemon/handoff",
        json!({"session_id": session_id, "entry": entry}),
    )
    .await;
    match outcome {
        Ok(_) => {
            // Recorded before `live` drops the session: a call that misses
            // it in `live` already sees the ship.
            state
                .handover
                .shipped
                .lock()
                .expect("shipped poisoned")
                .insert(session_id.clone());
            state
                .live
                .lock()
                .expect("live poisoned")
                .remove(&session_id);
            // Wake sends holding on the `HandingOff` mark: their refusal
            // now routes to the session's new owner.
            state.handover.progressed.notify_waiters();
            close_sub(&sub).await;
            info!(%session_id, "session handed off to the next daemon");
        }
        Err(error) => {
            error!(%session_id, %error, "handoff report failed; keeping the session");
            sub.rt.closing.clear_handoff();
            // Wake sends holding on the mark: the session stays owned and
            // their retry proceeds here.
            state.handover.progressed.notify_waiters();
        }
    }
}

/// Report a session that left this daemon without a handoff — closed,
/// forgotten, watchdog-cancelled, or a launch that never landed. The shipped entry
/// is the owner's last registry state; `null` means `forget` removed it.
async fn released_one(state: &Arc<AppState>, next: &Path, session_id: &str) {
    let entry = state
        .registry
        .lock()
        .expect("registry poisoned")
        .get(session_id)
        .cloned();
    let outcome: Result<Value> = peer_json(
        next,
        "daemon/release",
        json!({"session_id": session_id, "entry": entry}),
    )
    .await;
    if let Err(error) = outcome {
        error!(%session_id, %error, "release report failed");
    }
}

/// Resume one handed-off session in the background: `session/load` on a
/// fresh agent process, same as the startup pass. `resuming` keeps the
/// id claimed until the launch settles so a call that lands meanwhile
/// waits instead of seeing an unknown session.
async fn resume_handed(state: Arc<AppState>, session_id: String, entry: RegistryEntry) {
    let mut report = ResumeReport::default();
    match state.load_config() {
        Ok(config) => {
            resume_one(&state, &config, session_id.clone(), entry, &mut report).await;
        }
        Err(error) => {
            report.failed.push(ResumeFailure {
                session_id: session_id.clone(),
                error: error.to_string(),
            });
        }
    }
    {
        let mut state_report = state.resume_report.lock().expect("resume report poisoned");
        state_report.resumed.extend(report.resumed);
        state_report.failed.extend(report.failed);
    }
    state
        .handover
        .resuming
        .lock()
        .expect("resuming poisoned")
        .remove(&session_id);
    state.handover.progressed.notify_waiters();
}

/// Re-adopt every session marked for resume — the previous daemon's
/// `handed` list and the marks a dead daemon left — before the listener
/// accepts clients: the same ids, model, mode and cwd, with
/// accepted queued prompts replayed and a still-parked session's
/// rate-limit schedule restored. A failed `session/load` is reported,
/// not dropped — the registry entry and its resume mark stay, so a
/// later `adopt` or restart can retry.
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
/// session and the send-queue the previous daemon could not run —
/// kicked with the same gate a fresh prompt sees, so a still-active
/// limit re-parks it in order instead of burning a request.
async fn resume_one(
    state: &Arc<AppState>,
    config: &crate::config::Config,
    session_id: String,
    entry: RegistryEntry,
    report: &mut ResumeReport,
) {
    // A parked session's queue lives in its parked record — a dead
    // process took the send-queue with it, so the registry copy is the
    // only one there. An unparked session's queued sends ride the
    // resume mark.
    let queued = match &entry.rate_limited {
        Some(parked) => parked.queued.clone(),
        None => entry
            .resume
            .as_ref()
            .map_or_else(Vec::new, |resume| resume.queued.clone()),
    };
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
        },
    )
    .await;
    let sub = match result {
        Ok(sub) => sub,
        Err(error) => {
            // A `close` that landed while the resume's `session/load`
            // was in flight ended the session on purpose — its mark was
            // already consumed, so this is not a resume failure.
            if matches!(error, Error::ClosedWhileLaunching(_)) {
                info!(%session_id, "session closed while its resume was in flight");
                return;
            }
            warn!(%session_id, %error, "session resume failed");
            let message = error.to_string();
            // The mark stays — the next restart retries the load — and
            // records the failure so `list`/`status` surface it.
            let persist = state
                .update_registry(|registry| {
                    if let Some(entry) = registry.get_mut(&session_id)
                        && let Some(resume) = entry.resume.as_mut()
                    {
                        resume.failed = Some(message.clone());
                    }
                })
                .await;
            if let Err(error) = persist {
                warn!(%session_id, %error, "registry persist failed");
            }
            report.failed.push(ResumeFailure {
                session_id,
                error: message,
            });
            return;
        }
    };
    // The handing-over daemon's last turn outcome carries over, so a
    // `wait` on that turn still sees it end rather than a fresh `idle`.
    if let Some(last_turn) = entry
        .resume
        .as_ref()
        .and_then(|resume| resume.last_turn.clone())
    {
        sub.rt.inner.lock().expect("inner poisoned").status = Status::from(last_turn);
    }
    // A session parked at handoff keeps its schedule: `register_session`
    // consumed the flag, so restore it and re-park the status — the
    // scheduler resumes the session on its own when the quota lifts.
    if let Some(parked) = &entry.rate_limited
        && let Ok(resume_at) = parked.resume_at.parse::<jiff::Timestamp>()
    {
        sub.rt.inner.lock().expect("inner poisoned").status = Status::RateLimited {
            resume_at,
            reason: parked.reason.clone(),
            plan: parked.plan,
        };
        let parked = parked.clone();
        let result = state
            .update_registry(|registry| {
                // The closing check runs under `registry_write`, serialized
                // against `unschedule_resume`: a `close` that already
                // stored the flag keeps its cleared mark.
                if let Some(entry) = registry.get_mut(&session_id)
                    && sub.rt.closing.get().is_none()
                {
                    entry.rate_limited = Some(parked);
                }
            })
            .await;
        if let Err(error) = result {
            warn!(%session_id, %error, "registry persist failed");
        }
    }
    sub.rt
        .inner
        .lock()
        .expect("inner poisoned")
        .queue
        .extend(queued);
    kick_queue(state, &sub).await;
    report.resumed.push(session_id);
}

/// Start a settled session's first queued send, with the same gate a
/// fresh prompt sees: a still-active limit re-parks it in order instead
/// of burning a request.
async fn kick_queue(state: &Arc<AppState>, sub: &Arc<Subagent>) {
    let first = {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        if inner.status.accepts_prompt()
            && inner.current.is_none()
            && inner.steer_pending.is_empty()
            && sub.rt.closing.get().is_none()
        {
            inner.queue.pop_front()
        } else {
            None
        }
    };
    if let Some(prompt) = first {
        if let Some(limit) = ratelimit::active_limit(state, &sub.agent) {
            ratelimit::repark_prompt(state, sub, &prompt, &limit).await;
        } else if let Err(error) = start_turn(state.clone(), sub, prompt).await {
            warn!(session_id = %sub.session_id, %error, "queued prompt failed to start");
        }
    }
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
        Err(bind_error) if bind_error.kind() == std::io::ErrorKind::AddrInUse => {
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
        Err(source) => Err(Error::io(format!("cannot bind {}", path.display()), source)),
    }
}

/// The tool set one connection serves.
///
/// `internal` is the private generation socket: on top of the daemon
/// set it exposes `daemon/takeover`, `daemon/handoff` and
/// `daemon/release`. The public socket wraps the daemon set with the
/// forwarding dispatch, so calls for sessions an older generation still
/// owns reach it.
fn local_tools(state: &Arc<AppState>, internal: bool) -> aither_core::Result<Tools> {
    let mut tools = build_tools(state.clone())?;
    crate::tools::register_daemon_tools(&mut tools, state.clone())?;
    if internal {
        register_internal_tools(&mut tools, state.clone())?;
    }
    Ok(tools)
}

/// One client connection: a fresh tool set over the shared state, served
/// until the client hangs up.
async fn serve_connection(
    stream: async_net::unix::UnixStream,
    state: Arc<AppState>,
    internal: bool,
) -> std::result::Result<(), aither_mcp::McpError> {
    let (read_half, write_half) = futures_lite::io::split(stream);
    let transport = StreamTransport::new(BufReader::new(read_half), write_half);
    let tools = if internal {
        local_tools(&state, true)
    } else {
        forwarding_tools(&state)
    }
    .map_err(|error| aither_mcp::McpError::InvalidConfig(format!("cannot build tools: {error}")))?;
    // `serverInfo.version` doubles as this daemon's instance identity:
    // the pid — already the generation's name (`daemon.<pid>.sock`, the
    // pid file) — rides as semver build metadata, so a client retrying a
    // dropped call can tell the process that dropped it from its
    // replacement.
    let version = format!(
        "{}+daemon.{}",
        env!("CARGO_PKG_VERSION"),
        std::process::id()
    );
    let mut server = McpServer::new(transport, tools, "acpsub", version);
    server.run().await
}

// ---------------------------------------------------------------------------
// daemon/takeover / daemon/handoff / daemon/release — the generation
// protocol, served on the private socket only
// ---------------------------------------------------------------------------

/// Register the generation-protocol tools.
fn register_internal_tools(tools: &mut Tools, state: Arc<AppState>) -> aither_core::Result<()> {
    tools.register(TakeoverTool(state.clone()))?;
    tools.register(HandoffTool(state.clone()))?;
    tools.register(ReleaseTool(state))?;
    Ok(())
}

/// Tell this daemon it is superseded: a newer daemon took the public
/// socket. This daemon stops persisting `registry.json`, hands every
/// session without a turn in flight over at once — the answer carries
/// each with its resume mark — and keeps only sessions with a turn or
/// launch in flight (named with this daemon's private socket, or the
/// older owner's for sessions this daemon itself forwards). It reports
/// each kept session again as it settles (`daemon/handoff`) or ends
/// (`daemon/release`), then exits.
struct TakeoverTool(Arc<AppState>);

impl Tool for TakeoverTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("daemon/takeover")
    }
    type Arguments = TakeoverArgs;
    type Res = Value;

    async fn call(&self, args: TakeoverArgs, _cx: ToolContext) -> aither_core::Result<Value> {
        let state = &self.0;
        // The liveness connection to the successor: its private listener
        // is bound before it calls this, and the connection closes when
        // the successor's process ends. A successor that cannot be reached
        // is refused before anything commits to it.
        let successor = async_net::unix::UnixStream::connect(&args.socket)
            .await
            .map_err(|source| Error::DaemonConnect {
                path: args.socket.clone(),
                source,
            })?;
        // `registry_write` serializes the flag against in-flight
        // persists: a mutation serialized after this point stops writing
        // and rides the handoff or release report instead.
        let _write = state.registry_write.lock().await;
        if state.handover.superseded.swap(true, Ordering::Relaxed) {
            return Err(Error::io(
                "already superseded",
                std::io::Error::other("two successors"),
            )
            .into());
        }
        *state.handover.next.lock().expect("next poisoned") = Some(args.socket.clone());
        let own_socket = state
            .handover
            .private_socket
            .lock()
            .expect("private socket poisoned")
            .clone()
            .expect("private socket bound");
        let Partition {
            kept,
            handed,
            handed_subs,
        } = partition(state, &own_socket)?;
        info!(
            kept = kept.len(),
            handed = handed.len(),
            "superseded by a newer daemon"
        );
        let owned = kept
            .iter()
            .filter(|kept| kept.socket == own_socket)
            .map(|kept| kept.session_id.clone())
            .collect();
        tokio::spawn(handover_watch(
            state.clone(),
            args.socket,
            successor,
            handed_subs,
            owned,
        ));
        // The successor takes the socket's claim the moment this answer
        // lands: release it first. The `HeldLock` drop unlocks the
        // description, so a child mid-spawn cannot keep the claim held.
        drop(state.handover.claim.lock().expect("claim poisoned").take());
        // Wake the turn watchdogs and resuming-call waiters: a kept
        // turn's grace clock restarts now — the restart is not waiter
        // loss, and a `wait` gets a fresh window to reattach through the
        // successor's forwarding.
        state.handover.progressed.notify_waiters();
        serde_json::to_value(TakeoverReport { kept, handed }).map_err(Into::into)
    }
}

/// [`TakeoverTool`]'s partition of the live sessions.
struct Partition {
    /// Still owned: a turn or launch in flight, or forwarded further back.
    kept: Vec<KeptSession>,
    /// Shipped in the report with their resume marks.
    handed: Vec<HandedSession>,
    /// The handed sessions' subagents, closed by the handover watcher.
    handed_subs: Vec<Arc<Subagent>>,
}

/// Split the live sessions at takeover: every session without a turn in
/// flight is handed over (and leaves `live`), the rest stay owned.
///
/// # Errors
///
/// Returns [`Error::Internal`] for a live session missing from the
/// registry.
fn partition(state: &AppState, own_socket: &Path) -> Result<Partition> {
    let mut kept = Vec::new();
    let mut handed = Vec::new();
    let mut handed_subs = Vec::new();
    let mut live = state.live.lock().expect("live poisoned");
    let reserved = state.reserved.lock().expect("reserved poisoned");
    let registry = state.registry.lock().expect("registry poisoned");
    for (id, sub) in live.iter() {
        let inner = sub.rt.inner.lock().expect("inner poisoned");
        let in_flight = inner.in_flight() || sub.rt.closing.get().is_some();
        // A handed session takes no new turn here: `begin_turn`'s gate
        // checks `closing` under this same lock.
        if !in_flight {
            sub.rt.closing.set(Closing::HandingOff);
        }
        let mark = resume_mark(&inner);
        // A live parked session's queue is the truth: the parked
        // record's persisted copy can lag a `repark` that already
        // requeued the prompt, so the shipped record mirrors the queue
        // as it stands.
        let parked_queue =
            matches!(inner.status, Status::RateLimited { .. }).then(|| mark.queued.clone());
        drop(inner);
        let entry = registry.get(id).cloned();
        if in_flight {
            kept.push(KeptSession {
                session_id: id.clone(),
                socket: own_socket.to_path_buf(),
                entry,
            });
        } else {
            let mut entry = entry.ok_or_else(|| Error::Internal {
                session_id: id.clone(),
                detail: "live session is not registered".to_string(),
            })?;
            if let (Some(queued), Some(parked)) = (parked_queue, entry.rate_limited.as_mut()) {
                parked.queued = queued;
            }
            entry.resume = Some(mark);
            handed.push(HandedSession {
                session_id: id.clone(),
                entry,
            });
            handed_subs.push(sub.clone());
        }
    }
    // A launch in flight keeps its session until it settles.
    for id in reserved.keys() {
        kept.push(KeptSession {
            session_id: id.clone(),
            socket: own_socket.to_path_buf(),
            entry: registry.get(id).cloned(),
        });
    }
    // Sessions this daemon itself forwards to an older generation stay
    // forwarded — the successor gets their owner, and this daemon relays
    // their handoff.
    for (id, socket) in state
        .handover
        .foreign
        .lock()
        .expect("foreign poisoned")
        .iter()
    {
        kept.push(KeptSession {
            session_id: id.clone(),
            socket: socket.clone(),
            entry: registry.get(id).cloned(),
        });
    }
    drop(registry);
    drop(reserved);
    let mut shipped = state.handover.shipped.lock().expect("shipped poisoned");
    for sub in &handed_subs {
        shipped.insert(sub.session_id.clone());
        live.remove(&sub.session_id);
    }
    drop(shipped);
    drop(live);
    Ok(Partition {
        kept,
        handed,
        handed_subs,
    })
}

/// A kept session settled: its turn ended, and the owner hands it over.
/// The entry — already carrying the resume mark and the parked
/// send-queue — replaces this daemon's copy, then the session resumes
/// here via `session/load`.
struct HandoffTool(Arc<AppState>);

impl Tool for HandoffTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("daemon/handoff")
    }
    type Arguments = HandoffArgs;
    type Res = Value;

    async fn call(&self, args: HandoffArgs, _cx: ToolContext) -> aither_core::Result<Value> {
        let state = &self.0;
        // Superseded itself: the session is relayed to the newer
        // generation rather than resumed here.
        let next = state.handover.next.lock().expect("next poisoned").clone();
        if let Some(next) = next {
            let relayed = relay(
                &next,
                "daemon/handoff",
                serde_json::to_value(&RelayHandoff {
                    session_id: &args.session_id,
                    entry: &args.entry,
                })?,
            )
            .await?;
            stop_relaying(state, &args.session_id);
            return Ok(relayed);
        }
        let session_id = args.session_id;
        let entry = args.entry;
        // Claimed as resuming before it is registered, and registered
        // before the forward drops: `route` then never sees the session
        // registered yet neither forwarded nor resuming — a local call in
        // that gap would find no live session. Until the forward drops,
        // calls still reach the owner, which keeps the session live until
        // this answer lands.
        state
            .handover
            .resuming
            .lock()
            .expect("resuming poisoned")
            .insert(session_id.clone());
        let registered = state
            .update_registry(|registry| {
                registry.insert(session_id.clone(), entry.clone());
            })
            .await;
        if let Err(error) = registered {
            state
                .handover
                .resuming
                .lock()
                .expect("resuming poisoned")
                .remove(&session_id);
            state.handover.progressed.notify_waiters();
            return Err(error.into());
        }
        state
            .handover
            .foreign
            .lock()
            .expect("foreign poisoned")
            .remove(&session_id);
        info!(%session_id, "session handed over; resuming");
        tokio::spawn(resume_handed(state.clone(), session_id, entry));
        Ok(json!({}))
    }
}

/// `daemon/handoff`'s relay body — borrows the args so the call needs
/// no extra copy.
#[derive(Serialize)]
struct RelayHandoff<'a> {
    /// Session id.
    session_id: &'a str,
    /// Registry entry with `resume` set.
    entry: &'a RegistryEntry,
}

/// An owned session ended without a handoff: the owner reports its
/// last registry entry (`null` when `forget` removed it), and the
/// session leaves the forwarding map.
struct ReleaseTool(Arc<AppState>);

impl Tool for ReleaseTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("daemon/release")
    }
    type Arguments = ReleaseArgs;
    type Res = Value;

    async fn call(&self, args: ReleaseArgs, _cx: ToolContext) -> aither_core::Result<Value> {
        let state = &self.0;
        let next = state.handover.next.lock().expect("next poisoned").clone();
        if let Some(next) = next {
            let relayed = relay(
                &next,
                "daemon/release",
                serde_json::to_value(&RelayRelease {
                    session_id: &args.session_id,
                    entry: args.entry.as_ref(),
                })?,
            )
            .await?;
            stop_relaying(state, &args.session_id);
            return Ok(relayed);
        }
        let session_id = args.session_id;
        // The final entry lands before the forward drops, so a call in
        // between answers from the owner or from this registry, never
        // from neither.
        state
            .update_registry(|registry| {
                if let Some(entry) = args.entry {
                    registry.insert(session_id.clone(), entry);
                } else {
                    registry.remove(&session_id);
                }
            })
            .await?;
        state
            .handover
            .foreign
            .lock()
            .expect("foreign poisoned")
            .remove(&session_id);
        Ok(json!({}))
    }
}

/// `daemon/release`'s relay body.
#[derive(Serialize)]
struct RelayRelease<'a> {
    /// Session id.
    session_id: &'a str,
    /// Final registry entry, or none.
    entry: Option<&'a RegistryEntry>,
}

/// Relay one generation-protocol call to the newer daemon that
/// superseded this one and pass its result back.
async fn relay(next: &Path, tool: &str, args: Value) -> aither_core::Result<Value> {
    peer_json(next, tool, args).await.map_err(Into::into)
}

/// A relayed session left this daemon's forwarding map: wake the
/// handover watcher, which exits once nothing is left to relay.
fn stop_relaying(state: &AppState, session_id: &str) {
    state
        .handover
        .foreign
        .lock()
        .expect("foreign poisoned")
        .remove(session_id);
    state.handover.progressed.notify_waiters();
}

/// One generation-protocol call on a peer's private socket, its JSON
/// answer decoded. A refusal is an error, never a reply to read past.
///
/// # Errors
///
/// Returns an error when the peer is unreachable, refuses the call, or
/// answers with JSON that does not decode as `T`.
async fn peer_json<T: serde::de::DeserializeOwned>(
    socket: &Path,
    tool: &str,
    args: Value,
) -> Result<T> {
    let mut cx = ToolContext::new();
    let result = peer_call(socket, tool, args, &mut cx).await?;
    let text = result_text(&result);
    if result.is_error {
        return Err(Error::PeerRefused {
            path: socket.to_path_buf(),
            tool: tool.to_string(),
            detail: text,
        });
    }
    serde_json::from_str(&text).map_err(|source| {
        Error::io(
            format!("cannot parse {tool} answer from {}", socket.display()),
            std::io::Error::other(source),
        )
    })
}

// ---------------------------------------------------------------------------
// Forwarding: calls for sessions an older generation still owns
// ---------------------------------------------------------------------------

/// Tools addressed at one or more sessions — the ones a call can carry
/// to a session the previous generation still owns.
const SESSION_TOOLS: &[&str] = &[
    "wait",
    "status",
    "result",
    "transcript",
    "send",
    "cancel",
    "permit",
    "close",
    "forget",
    "adopt",
];

/// Wrap the local tool set so session-addressed calls forward to the
/// daemon that still owns the session. Every tool is re-registered as a
/// dynamic handler that either forwards or delegates to the local set.
fn forwarding_tools(state: &Arc<AppState>) -> aither_core::Result<Tools> {
    let inner = Arc::new(local_tools(state, false)?);
    let mut tools = Tools::new();
    for definition in inner.definitions() {
        let name = definition.name().to_string();
        let inner = inner.clone();
        let state = state.clone();
        tools.register_dyn(definition, move |args: &str, cx: ToolContext| {
            let inner = inner.clone();
            let state = state.clone();
            let name = name.clone();
            let args = args.to_string();
            Box::pin(dispatch(state, inner, name, args, cx))
        })?;
    }
    Ok(tools)
}

/// The session ids a call addresses: `session_id` for the single-session
/// tools, `session_ids` for `wait_any`. Anything else is local work.
fn call_sessions(name: &str, args: &Value) -> Vec<String> {
    if name == "wait_any" {
        args.get("session_ids")
            .and_then(Value::as_array)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| id.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    } else if SESSION_TOOLS.contains(&name) {
        args.get("session_id")
            .and_then(Value::as_str)
            .map(|id| vec![id.to_string()])
            .unwrap_or_default()
    } else {
        Vec::new()
    }
}

/// A `ProgressSink` that fans reports into a channel so every dispatch
/// branch can report while the caller's `ToolContext` stays single-owner.
struct ProgressTx(tokio::sync::mpsc::UnboundedSender<Progress>);

impl ProgressSink for ProgressTx {
    async fn report(&mut self, progress: Progress) {
        let _ = self.0.send(progress);
    }
}

/// The context a dispatch branch runs with: listening branches fan into
/// `tx`; a non-listening call gets a plain context — and so does a
/// forwarded call whose caller does not listen, keeping the peer's own
/// no-listener limits honest.
fn branch_cx(listening: bool, tx: &tokio::sync::mpsc::UnboundedSender<Progress>) -> ToolContext {
    if listening {
        ToolContext::with_progress(ProgressTx(tx.clone()))
    } else {
        ToolContext::new()
    }
}

/// Run one tool call, forwarding session-addressed calls to the daemon
/// generation that still owns the session. Local work — `spawn`, `list`,
/// sessions already handed over — runs here.
///
/// A forwarded call that loses its target mid-flight — the owner
/// released or handed the session over while the call was on the wire —
/// is reclassified and retried locally, never surfaced as a restart
/// error.
async fn dispatch(
    state: Arc<AppState>,
    inner: Arc<Tools>,
    name: String,
    args: String,
    mut cx: ToolContext,
) -> aither_core::Result<ToolResult> {
    let listening = cx.is_listening();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Progress>();
    let pipe = tokio::spawn(async move {
        while let Some(progress) = rx.recv().await {
            let _ = cx.report_progress(progress).await;
        }
    });
    let result = dispatch_inner(&state, &inner, &name, &args, listening, tx).await;
    let _ = pipe.await;
    result
}

/// The dispatch loop behind [`dispatch`]: classify the call's sessions,
/// wait out a mid-resume one, forward foreign ones, run local ones.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per dispatch case; the wait_any split needs the room"
)]
async fn dispatch_inner(
    state: &Arc<AppState>,
    inner: &Arc<Tools>,
    name: &str,
    args: &str,
    listening: bool,
    tx: tokio::sync::mpsc::UnboundedSender<Progress>,
) -> aither_core::Result<ToolResult> {
    let parsed: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
    let ids = call_sessions(name, &parsed);
    for _ in 0..DISPATCH_ATTEMPTS {
        let mut foreign: BTreeMap<PathBuf, Vec<String>> = BTreeMap::new();
        let mut local = Vec::new();
        let mut resuming = false;
        for id in &ids {
            match route(state, id) {
                Route::Owner(socket) => foreign.entry(socket).or_default().push(id.clone()),
                Route::Resuming => resuming = true,
                Route::Local => local.push(id.clone()),
            }
        }
        if resuming {
            // A handoff landed but the `session/load` is still in flight:
            // wait for it to settle, then reclassify.
            let notified = state.handover.progressed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if ids.iter().any(|id| {
                state
                    .handover
                    .resuming
                    .lock()
                    .expect("resuming poisoned")
                    .contains(id)
            }) {
                notified.await;
            }
            continue;
        }
        match (foreign.len(), local.is_empty()) {
            // Everything local — the common case.
            (0, _) => {
                return inner.call(name, args, branch_cx(listening, &tx)).await;
            }
            // One owner for every addressed session: forward the call
            // as-is.
            (1, true) => {
                let (socket, kept) = foreign.iter().next().expect("one entry");
                let mut branch = branch_cx(listening, &tx);
                match peer_call(socket, name, parsed.clone(), &mut branch).await {
                    Ok(result) => {
                        if result.is_error && !still_owned(state, socket, kept) {
                            // The session moved mid-call — reclassify.
                            continue;
                        }
                        return Ok(tool_result(&result));
                    }
                    Err(error) => {
                        // The owner exited: it owns nothing any more.
                        if matches!(error, Error::DaemonConnect { .. }) {
                            owner_gone(state, socket);
                            continue;
                        }
                        if !still_owned(state, socket, kept) {
                            continue;
                        }
                        return Err(error.into());
                    }
                }
            }
            // `wait_any` mixing local and foreign sessions: every
            // branch watches its subset; the first answer wins.
            _ => {
                let mut calls: Vec<
                    std::pin::Pin<
                        Box<dyn Future<Output = aither_core::Result<ToolResult>> + Send + '_>,
                    >,
                > = Vec::new();
                if !local.is_empty() {
                    let mut subset = parsed.clone();
                    subset["session_ids"] = json!(local);
                    let local_args = subset.to_string();
                    let branch = branch_cx(listening, &tx);
                    calls.push(Box::pin(async move {
                        inner.call(name, &local_args, branch).await
                    }));
                }
                for (socket, kept) in &foreign {
                    let mut subset = parsed.clone();
                    subset["session_ids"] = json!(kept);
                    calls.push(Box::pin(peer_result(
                        socket,
                        name,
                        subset,
                        branch_cx(listening, &tx),
                    )));
                }
                let race = calls
                    .into_iter()
                    .reduce(|left, right| Box::pin(futures_lite::future::race(left, right)))
                    .expect("at least one branch");
                let result = race.await;
                return match result {
                    Ok(result) => Ok(result),
                    Err(error) => {
                        // The winning branch lost its target: retry
                        // rather than guess which result the caller saw.
                        if foreign
                            .iter()
                            .all(|(socket, kept)| !still_owned(state, socket, kept))
                        {
                            continue;
                        }
                        Err(error)
                    }
                };
            }
        }
    }
    Err(aither_core::Error::msg(
        "handover dispatch did not settle: the session keeps moving",
    ))
}

/// Where a call for one session goes right now.
enum Route {
    /// This daemon serves it.
    Local,
    /// A handoff landed and its `session/load` is still in flight.
    Resuming,
    /// Another generation's daemon owns it, by its private socket.
    Owner(PathBuf),
}

/// Route one session id. A session another generation owns forwards to
/// it; one this daemon has never heard of — not live, launching, or
/// registered — forwards to the superseded generation while it still
/// runs, since it may own a session spawned on a connection it accepted
/// before the takeover, which reaches this daemon only by handoff or
/// release. Nothing about that guess is stored: the next call routes
/// afresh.
fn route(state: &AppState, id: &str) -> Route {
    let prev = state.handover.prev.lock().expect("prev poisoned").clone();
    // Checked before `foreign` is taken: `partition` holds `live`,
    // `reserved` and `registry` while it takes `foreign`.
    let unknown = prev.is_some()
        && !state.live.lock().expect("live poisoned").contains_key(id)
        && !state
            .reserved
            .lock()
            .expect("reserved poisoned")
            .contains_key(id)
        && state
            .registry
            .lock()
            .expect("registry poisoned")
            .get(id)
            .is_none();
    let foreign = state.handover.foreign.lock().expect("foreign poisoned");
    let resuming = state.handover.resuming.lock().expect("resuming poisoned");
    match (
        foreign.get(id),
        resuming.contains(id),
        prev.filter(|_| unknown),
    ) {
        (Some(owner), _, _) => Route::Owner(owner.clone()),
        (None, true, _) => Route::Resuming,
        (None, false, Some(prev)) => Route::Owner(prev),
        (None, false, None) => Route::Local,
    }
}

/// A forwarding target stopped answering: the daemon behind it exited
/// and owns nothing, so every session routed there is local again.
fn owner_gone(state: &AppState, socket: &Path) {
    state
        .handover
        .foreign
        .lock()
        .expect("foreign poisoned")
        .retain(|_, owner| owner != socket);
    let mut prev = state.handover.prev.lock().expect("prev poisoned");
    if prev.as_deref() == Some(socket) {
        *prev = None;
    }
}

/// `peer_call` mapped onto the local result type.
async fn peer_result(
    socket: &Path,
    name: &str,
    args: Value,
    mut cx: ToolContext,
) -> aither_core::Result<ToolResult> {
    peer_call(socket, name, args, &mut cx)
        .await
        .map(|result| tool_result(&result))
        .map_err(aither_core::Error::from)
}

/// Whether every id in `kept` still routes to `socket`'s daemon — false
/// means the session moved and the call should reclassify.
fn still_owned(state: &AppState, socket: &Path, kept: &[String]) -> bool {
    kept.iter()
        .all(|id| matches!(route(state, id), Route::Owner(owner) if owner == socket))
}

/// A peer's `CallToolResult` on the local `ToolResult` scale: the daemon
/// tools render text only, so the contents concatenate.
fn tool_result(result: &CallToolResult) -> ToolResult {
    let text = result_text(result);
    if result.is_error {
        ToolResult::error(text)
    } else if text.is_empty() {
        ToolResult::Done
    } else {
        ToolResult::text(text)
    }
}

/// The text a `CallToolResult` carries — every daemon tool answer is
/// one JSON text item.
fn result_text(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|item| match item {
            Content::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// One call on a peer daemon's private socket: connect, MCP handshake,
/// `tools/call`. A progress token rides when `cx` listens, and the
/// peer's progress notifications pipe back into it.
async fn peer_call(
    socket: &Path,
    tool: &str,
    arguments: Value,
    cx: &mut ToolContext,
) -> Result<CallToolResult> {
    let stream = async_net::unix::UnixStream::connect(socket)
        .await
        .map_err(|source| Error::DaemonConnect {
            path: socket.to_path_buf(),
            source,
        })?;
    let (read_half, write_half) = futures_lite::io::split(stream);
    let mut transport = StreamTransport::new(BufReader::new(read_half), write_half);
    let disconnected = |error: aither_mcp::McpError| Error::DaemonDisconnected {
        path: socket.to_path_buf(),
        source: std::io::Error::other(error),
    };
    transport
        .request(JsonRpcRequest::with_params(
            0i64,
            "initialize",
            InitializeParams::default(),
        ))
        .await
        .map_err(disconnected)?;
    transport
        .notify(JsonRpcNotification::new("notifications/initialized"))
        .await
        .map_err(disconnected)?;
    let token = cx
        .is_listening()
        .then(|| ProgressToken::String("acpsub-handover".to_string()));
    transport
        .send_request(JsonRpcRequest::with_params(
            1i64,
            "tools/call",
            CallToolParams {
                name: tool.to_string(),
                arguments,
                meta: Some(RequestMeta {
                    progress_token: token.clone(),
                }),
            },
        ))
        .await
        .map_err(disconnected)?;
    loop {
        match transport.recv().await.map_err(disconnected)? {
            Some(JsonRpcMessage::Response(response)) if response.id == RequestId::Number(1) => {
                let result = response
                    .into_result()
                    .map_err(|error| Error::DaemonDisconnected {
                        path: socket.to_path_buf(),
                        source: std::io::Error::other(error),
                    })?;
                return serde_json::from_value(result).map_err(|source| {
                    Error::io("cannot parse peer result", std::io::Error::other(source))
                });
            }
            Some(JsonRpcMessage::Notification(notification)) => {
                if notification.method == "notifications/progress" && cx.is_listening() {
                    let params = notification
                        .params
                        .and_then(|params| {
                            serde_json::from_value::<ProgressNotificationParams>(params).ok()
                        })
                        .filter(|params| Some(&params.progress_token) == token.as_ref());
                    if let Some(params) = params {
                        let mut progress = Progress::new(params.progress);
                        if let Some(total) = params.total {
                            progress = progress.with_total(total);
                        }
                        if let Some(message) = params.message {
                            progress = progress.with_message(message);
                        }
                        let _ = cx.report_progress(progress).await;
                    }
                }
            }
            Some(_) => {}
            None => return Err(disconnected(aither_mcp::McpError::ConnectionClosed)),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// A released claim is free at once, even while another copy of its
    /// descriptor lives on, as it does in a child mid-spawn. Closing a
    /// locked descriptor alone would leave the lock held until every
    /// copy closed.
    #[test]
    fn released_lock_is_free_while_a_descriptor_copy_lives() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("daemon.sock.pid");
        let open = || {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
                .expect("open claim")
        };
        let mut claim = HeldLock::try_new(open()).expect("take claim");
        let copy = claim.file().try_clone().expect("copy descriptor");
        drop(claim);
        let next = HeldLock::try_new(open());
        assert!(next.is_ok(), "released claim still held");
        drop(copy);
    }

    /// A socket path over the platform's `sun_path` limit fails `bind`
    /// with a non-`AddrInUse` error: the error must report the bind
    /// failure, not a stale-socket removal.
    #[tokio::test]
    async fn bind_reports_overlong_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("s".repeat(256));
        let error = bind(&path).await.expect_err("overlong path bound");
        match error {
            Error::Io { context, source } => {
                assert!(context.starts_with("cannot bind"), "{context}");
                assert_ne!(source.kind(), std::io::ErrorKind::AddrInUse);
            }
            other => panic!("expected io error, got {other}"),
        }
    }
}
