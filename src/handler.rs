//! The per-subagent client hooks: applies the permission policy, serves
//! `terminal/*` and — unless the agent's file access is off — `fs/*`
//! requests, and drives the session's event stream into the turn state.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aither_acp::client::{FsHandler, PermissionHandler, V1Terminal};
use aither_acp::rpc::RpcError;
use aither_acp::state::{MessageKind, SessionEvent};
use aither_acp::wire::v1;
use aither_acp::{
    ClientSession, PermissionOutcome, PermissionRequest,
    domain::{CancelledOutcome, Open, Plan, SelectedOutcome, ToolCall},
};
use tokio::sync::oneshot;

use crate::config::{FileAccess, PermissionPolicy};
use crate::permission::PermissionRequest as SubPermissionRequest;
use crate::services::{ServiceCall, ServiceCalls};
use crate::state::{
    Inner, Status, SubRuntime, ToolCallSummary, Turn, oldest_steer, terminal_status,
};
use crate::terminal::Terminal;

/// Answers `session/request_permission` through the host's services.
#[derive(Debug)]
pub(crate) struct SubPermissions {
    /// Shared mutable state.
    rt: Arc<SubRuntime>,
    /// Session working directory.
    cwd: PathBuf,
    /// Permission policy for `session/request_permission`.
    permission: PermissionPolicy,
    /// The host's services, which answer permission requests.
    services: ServiceCalls,
}

impl SubPermissions {
    /// Construct the permission hook for a subagent being launched.
    #[must_use]
    pub(crate) const fn new(
        rt: Arc<SubRuntime>,
        cwd: PathBuf,
        permission: PermissionPolicy,
        services: ServiceCalls,
    ) -> Self {
        Self {
            rt,
            cwd,
            permission,
            services,
        }
    }
}

impl PermissionHandler for SubPermissions {
    async fn request(&self, request: PermissionRequest) -> PermissionOutcome {
        let (reply, answer) = oneshot::channel();
        let request = SubPermissionRequest::new(
            self.rt.session_label(),
            self.cwd.clone(),
            self.permission,
            request,
            self.rt.clone(),
        );
        if self
            .services
            .send(ServiceCall::Permission { request, reply })
            .is_ok()
        {
            answer
                .await
                .unwrap_or(PermissionOutcome::Cancelled(CancelledOutcome {
                    meta: None,
                }))
        } else {
            PermissionOutcome::Cancelled(CancelledOutcome { meta: None })
        }
    }
}

/// Serves the agent's `fs/*` requests under the session's file-access
/// policy.
#[derive(Debug)]
pub(crate) struct SubFs {
    /// Session working directory; fs paths must stay under it under
    /// [`FileAccess::Cwd`].
    cwd: PathBuf,
    /// What the `fs/*` methods serve.
    file_access: FileAccess,
}

impl SubFs {
    /// Construct the fs hook for a subagent being launched.
    #[must_use]
    pub(crate) const fn new(cwd: PathBuf, file_access: FileAccess) -> Self {
        Self { cwd, file_access }
    }
}

impl FsHandler for SubFs {
    async fn read(
        &self,
        params: v1::ReadTextFileParams,
    ) -> Result<v1::ReadTextFileResult, RpcError> {
        let path = self.resolve("fs/read_text_file", &params.path)?;
        let text = tokio::fs::read_to_string(&path).await.map_err(|source| {
            RpcError::internal_error(format!("cannot read {}: {source}", path.display()))
        })?;
        let content = slice_lines(&text, params.line, params.limit);
        Ok(v1::ReadTextFileResult {
            content,
            meta: None,
        })
    }

    async fn write(
        &self,
        params: v1::WriteTextFileParams,
    ) -> Result<v1::WriteTextFileResult, RpcError> {
        let path = self.resolve("fs/write_text_file", &params.path)?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|source| {
                RpcError::internal_error(format!("cannot create {}: {source}", parent.display()))
            })?;
        }
        tokio::fs::write(&path, &params.content)
            .await
            .map_err(|source| {
                RpcError::internal_error(format!("cannot write {}: {source}", path.display()))
            })?;
        Ok(v1::WriteTextFileResult { meta: None })
    }
}

impl SubFs {
    /// Resolve the path of a `method` request against the session cwd and
    /// enforce the file access: every request is refused when it is off,
    /// and one outside the cwd when it is confined there. Symlinks are
    /// resolved when the path (or its parent) exists.
    fn resolve(&self, method: &str, path: &Path) -> Result<PathBuf, RpcError> {
        if self.file_access == FileAccess::Off {
            return Err(RpcError::method_not_found(method));
        }
        let candidate = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        };
        let normalized = canonicalize_lenient(&candidate);
        if self.file_access == FileAccess::Cwd && !normalized.starts_with(&self.cwd) {
            return Err(RpcError::internal_error(format!(
                "{} is outside the session cwd {}",
                path.display(),
                self.cwd.display()
            )));
        }
        Ok(normalized)
    }
}

/// Serves the agent's `terminal/*` requests through the host's executor.
#[derive(Debug)]
pub(crate) struct SubTerminals {
    /// Shared mutable state (the terminals registry).
    rt: Arc<SubRuntime>,
    /// The host's services, which start terminal commands.
    services: ServiceCalls,
}

impl SubTerminals {
    /// Construct the terminal hook for a subagent being launched.
    #[must_use]
    pub(crate) const fn new(rt: Arc<SubRuntime>, services: ServiceCalls) -> Self {
        Self { rt, services }
    }
}

impl V1Terminal for SubTerminals {
    async fn create(
        &self,
        params: v1::TerminalCreateParams,
    ) -> Result<v1::TerminalCreateResult, RpcError> {
        let (reply, answer) = oneshot::channel();
        self.services
            .send(ServiceCall::Terminal { params, reply })
            .map_err(|_| RpcError::internal_error("the session's services have shut down"))?;
        answer.await.map_err(|_| {
            RpcError::internal_error("the session's services dropped the terminal request")
        })?
    }

    async fn output(
        &self,
        params: v1::TerminalOutputParams,
    ) -> Result<v1::TerminalOutputResult, RpcError> {
        Ok(self.terminal(&params.terminal_id).await?.output())
    }

    async fn wait_for_exit(
        &self,
        params: v1::TerminalWaitForExitParams,
    ) -> Result<v1::TerminalExitStatus, RpcError> {
        self.terminal(&params.terminal_id)
            .await?
            .wait_for_exit()
            .await
    }

    async fn kill(
        &self,
        params: v1::TerminalKillParams,
    ) -> Result<v1::TerminalKillResult, RpcError> {
        self.terminal(&params.terminal_id).await?.kill();
        Ok(v1::TerminalKillResult { meta: None })
    }

    async fn release(
        &self,
        params: v1::TerminalReleaseParams,
    ) -> Result<v1::TerminalReleaseResult, RpcError> {
        // Kill explicitly, not by drop: orphaned grandchildren hold the
        // output pipes open, which keeps pump-held `Arc`s — and therefore
        // the terminal — alive until they exit.
        let terminal = self.rt.terminals.lock().await.remove(&params.terminal_id);
        if let Some(terminal) = terminal {
            terminal.kill();
        }
        Ok(v1::TerminalReleaseResult { meta: None })
    }
}

impl SubTerminals {
    /// Look up a terminal by id.
    async fn terminal(&self, id: &str) -> Result<Arc<Terminal>, RpcError> {
        self.rt
            .terminals
            .lock()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| RpcError::invalid_params(format!("unknown terminal '{id}'")))
    }
}

/// Whether an event signals agent turn activity — the materialization
/// triggers for a steered prompt's turn. Session-state events (config,
/// commands, info, usage, replay) do not count: they arrive outside
/// turns too.
const fn turn_activity(event: &SessionEvent) -> bool {
    matches!(
        event,
        SessionEvent::MessageDelta { .. }
            | SessionEvent::ToolCall(_)
            | SessionEvent::Plan { .. }
            | SessionEvent::TerminalUpsert(_)
            | SessionEvent::TerminalOutput { .. }
            | SessionEvent::Subagent(_)
    )
}

/// Drive a session's event stream into the turn state: materialize a
/// `current` turn when an untracked update arrives (a steered prompt
/// running as a turn of its own), then apply the event's data.
///
/// Runs until the event stream ends (connection or session gone).
pub(crate) async fn drive_events(session: ClientSession, rt: Arc<SubRuntime>) {
    use futures_lite::StreamExt;
    let mut events = session.subscribe();
    // This stream's own position: the last `TurnEnded` it consumed. Only
    // events after it can open a steered turn — the tee's wire boundary
    // (`inner.turn_ended`) races ahead, so `drive_events` tracks its own.
    let mut seen_end: Option<u64> = None;
    while let Some(event) = events.next().await {
        if turn_activity(&event) {
            // Untracked traffic means the agent is running a turn we did
            // not open: a steered prompt that landed after its target
            // ended. Only traffic that postdates that turn's own end can
            // open one — the broadcast reports `TurnEnded` before a next
            // turn's updates arrive, so an event still queued behind it
            // (a v1-synthesized state event, say) belongs to the settled
            // turn. While the settle holds `current`, the steered turn
            // stages into `next_turn`.
            let mut inner = rt.inner.lock().expect("inner poisoned");
            let post_end = inner
                .current
                .as_ref()
                .is_some_and(|turn| seen_end == Some(turn.n))
                || (inner.current.is_none()
                    && seen_end.is_some()
                    && seen_end == inner.turns.last().map(|turn| turn.n));
            if post_end && let Some(owner) = oldest_steer(&inner) {
                let staged = Turn {
                    n: inner.turn_offset
                        + inner.turns.len() as u64
                        + 1
                        + u64::from(inner.current.is_some()),
                    started_at: Some(jiff::Timestamp::now()),
                    ..Turn::default()
                };
                inner.pending_turn_start = None;
                inner.steer_owner = Some(owner);
                if inner.current.is_none() {
                    inner.current = Some(staged);
                    inner.status = Status::Running;
                } else if inner.next_turn.is_none() {
                    inner.next_turn = Some(staged);
                }
                drop(inner);
                rt.notify.notify_waiters();
            } else {
                drop(inner);
            }
        }
        let mut inner = rt.inner.lock().expect("inner poisoned");
        // Mode and config updates are agent-reported session state, not
        // turn data: they apply even when no turn is open (e.g. during the
        // launch handshake, where `set_mode` is answered by an update).
        match &event {
            SessionEvent::ModeChanged(mode) => {
                inner.current_mode = Some(mode.clone());
            }
            SessionEvent::ConfigChanged => {
                inner
                    .config_options
                    .clone_from(&session.state().config_options);
            }
            // The broadcast's own end marker: untracked traffic after it
            // belongs to a next turn, traffic queued behind it still to
            // the settled one — the steer-turn materialization gates on
            // it, not just on `current` being free.
            SessionEvent::TurnEnded { .. } => {
                seen_end = inner
                    .current
                    .as_ref()
                    .or_else(|| inner.turns.last())
                    .map(|turn| turn.n);
            }
            _ => {}
        }
        // A chunk arriving just behind the turn's end still belongs to
        // that turn — and one staged past it (`next_turn`) belongs to the
        // steered turn it opens: `finish_turn` may clear `current` before
        // this task drains events that preceded the end on the wire.
        let Inner {
            next_turn,
            current,
            turns,
            ..
        } = &mut *inner;
        let turn = next_turn
            .as_mut()
            .or(current.as_mut())
            .or_else(|| turns.last_mut());
        let Some(turn) = turn else {
            continue;
        };
        match event {
            SessionEvent::MessageDelta {
                kind: MessageKind::Agent,
                text: Some(text),
                ..
            } => turn.reply.push_str(&text),
            SessionEvent::ToolCall(call) => note_tool_call(turn, call),
            SessionEvent::Plan { plan, .. } => {
                let Plan::Items(items) = plan;
                turn.plan = items.entries;
            }
            SessionEvent::ModeChanged(mode) => {
                turn.mode = Some(mode.as_str().to_string());
            }
            _ => {}
        }
        drop(inner);
    }
}

/// Record a `tool_call_update`/`tool_call_content_chunk` event: upsert
/// the call's summary from the materialized snapshot, stamping
/// `ended_at` when a terminal status first shows.
pub(crate) fn note_tool_call(turn: &mut Turn, call: ToolCall) {
    let seen = jiff::Timestamp::now();
    let id = call.tool_call_id.as_str().to_string();
    turn.latest_tool_call = Some(id.clone());
    let status = call.status.and_then(Open::into_known);
    let entry = turn
        .tool_calls
        .entry(id.clone())
        .or_insert_with(|| ToolCallSummary {
            id,
            title: String::new(),
            kind: None,
            status: None,
            locations: Vec::new(),
            started_at: seen,
            ended_at: None,
        });
    if let Some(status) = status {
        entry.status = Some(status);
        if terminal_status(status) && entry.ended_at.is_none() {
            entry.ended_at = Some(seen);
        }
    }
    if !call.title.is_empty() {
        entry.title = call.title;
    }
    if let Some(kind) = call.kind {
        entry.kind = kind.into_known();
    }
    if !call.locations.is_empty() {
        entry.locations = call.locations;
    }
}

/// Canonicalize a path that may not exist: the path itself, else its parent
/// plus the file name, else a lexical `.`/`..` normalization.
fn canonicalize_lenient(path: &Path) -> PathBuf {
    if let Ok(resolved) = path.canonicalize() {
        return resolved;
    }
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name())
        && let Ok(resolved) = parent.canonicalize()
    {
        return resolved.join(name);
    }
    absolutize(path)
}

/// Lexically normalize a path: resolve `.`/`..` without touching the
/// filesystem.
fn absolutize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

/// Apply the `line` (1-based)/`limit` window to file text.
fn slice_lines(text: &str, line: Option<u32>, limit: Option<u32>) -> String {
    let start = line.map_or(0, |line| line.saturating_sub(1) as usize);
    let lines: Vec<&str> = text.lines().skip(start).collect();
    let window = match limit {
        Some(limit) => &lines[..(limit as usize).min(lines.len())],
        None => &lines,
    };
    window.join("\n")
}

/// The option id a `selected` permission outcome reports.
#[must_use]
pub(crate) fn selected(option_id: &str) -> PermissionOutcome {
    PermissionOutcome::Selected(SelectedOutcome {
        option_id: option_id.into(),
        meta: None,
    })
}

/// The `cancelled` permission outcome.
#[must_use]
pub(crate) const fn cancelled() -> PermissionOutcome {
    PermissionOutcome::Cancelled(CancelledOutcome { meta: None })
}
