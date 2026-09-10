//! The per-subagent [`ClientHandler`]: transcripts every `session/update`,
//! applies the permission policy, and serves `fs/*`/`terminal/*` requests.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use aither_acp::{
    ClientCapabilities, ClientHandler, ContentBlock, FileSystemCapability, ReadTextFileParams,
    ReadTextFileResult, RequestPermissionOutcome, RequestPermissionParams, RequestPermissionResult,
    SessionNotification, SessionUpdate, TerminalCreateParams, TerminalCreateResult,
    TerminalExitStatus, TerminalKillParams, TerminalKillResult, TerminalOutputParams,
    TerminalOutputResult, TerminalReleaseParams, TerminalReleaseResult, TerminalWaitForExitParams,
    WriteTextFileParams, WriteTextFileResult,
};
use aither_mcp::protocol::JsonRpcError;
use tokio::sync::oneshot;
use tracing::{debug, warn};

use crate::config::PermissionPolicy;
use crate::state::{PendingPermission, Status, SubRuntime, ToolCallSummary, now};
use crate::terminal::{self, Terminal};

/// Handles one subagent's agent-to-client traffic.
#[derive(Debug)]
pub struct SubagentHandler {
    /// Subagent name (for diagnostics).
    name: String,
    /// Shared mutable state.
    rt: Arc<SubRuntime>,
    /// Session working directory; fs paths must stay under it unless
    /// `allow_outside_cwd`.
    cwd: PathBuf,
    /// Permission policy for `session/request_permission`.
    permission: PermissionPolicy,
    /// Allow fs access outside `cwd`.
    allow_outside_cwd: bool,
}

impl SubagentHandler {
    /// Construct the handler for a subagent being launched.
    #[must_use]
    pub const fn new(
        name: String,
        rt: Arc<SubRuntime>,
        cwd: PathBuf,
        permission: PermissionPolicy,
        allow_outside_cwd: bool,
    ) -> Self {
        Self {
            name,
            rt,
            cwd,
            permission,
            allow_outside_cwd,
        }
    }
}

impl ClientHandler for SubagentHandler {
    fn capabilities(&self) -> ClientCapabilities {
        ClientCapabilities {
            fs: FileSystemCapability {
                read_text_file: true,
                write_text_file: true,
            },
            terminal: true,
            meta: None,
        }
    }

    #[expect(
        clippy::significant_drop_tightening,
        reason = "the `inner` guard must stay alive while `turn` borrows it"
    )]
    async fn session_update(&self, notification: SessionNotification) {
        let turn_n = self
            .rt
            .inner
            .lock()
            .expect("inner poisoned")
            .current
            .as_ref()
            .map_or(0, |turn| turn.n);
        let record = serde_json::json!({
            "ts": now(),
            "turn": turn_n,
            "update": notification.update,
        });
        if let Err(error) = self.rt.transcript.lock().await.append(&record).await {
            warn!(subagent = %self.name, %error, "transcript append failed");
        }
        let mut inner = self.rt.inner.lock().expect("inner poisoned");
        let Some(turn) = inner.current.as_mut() else {
            return;
        };
        match notification.update {
            SessionUpdate::AgentMessageChunk(chunk) => {
                if let ContentBlock::Text(text) = chunk.content {
                    turn.reply.push_str(&text.text);
                }
            }
            SessionUpdate::ToolCall(call) => {
                turn.tool_calls.insert(
                    call.tool_call_id.clone(),
                    ToolCallSummary {
                        id: call.tool_call_id,
                        title: call.title,
                        kind: call.kind,
                        status: call.status,
                        locations: call.locations,
                    },
                );
            }
            SessionUpdate::ToolCallUpdate(update) => {
                let entry = turn
                    .tool_calls
                    .entry(update.tool_call_id.clone())
                    .or_insert_with(|| ToolCallSummary {
                        id: update.tool_call_id.clone(),
                        title: String::new(),
                        kind: None,
                        status: None,
                        locations: Vec::new(),
                    });
                if let Some(status) = update.status {
                    entry.status = Some(status);
                }
                if let Some(title) = update.title {
                    entry.title = title;
                }
                if let Some(kind) = update.kind {
                    entry.kind = Some(kind);
                }
                if let Some(locations) = update.locations {
                    entry.locations = locations;
                }
            }
            SessionUpdate::Plan(plan) => turn.plan = plan.entries,
            SessionUpdate::CurrentModeUpdate(mode) => {
                turn.mode = Some(mode.current_mode_id);
            }
            _ => {}
        }
    }

    async fn request_permission(
        &self,
        params: RequestPermissionParams,
    ) -> Result<RequestPermissionResult, JsonRpcError> {
        let outcome = match self.permission {
            PermissionPolicy::Allow => params
                .options
                .iter()
                .find(|o| {
                    matches!(
                        o.kind,
                        aither_acp::PermissionOptionKind::AllowOnce
                            | aither_acp::PermissionOptionKind::AllowAlways
                    )
                })
                .map_or(RequestPermissionOutcome::Cancelled, |o| {
                    RequestPermissionOutcome::Selected {
                        option_id: o.option_id.clone(),
                    }
                }),
            PermissionPolicy::Deny => params
                .options
                .iter()
                .find(|o| {
                    matches!(
                        o.kind,
                        aither_acp::PermissionOptionKind::RejectOnce
                            | aither_acp::PermissionOptionKind::RejectAlways
                    )
                })
                .map_or(RequestPermissionOutcome::Cancelled, |o| {
                    RequestPermissionOutcome::Selected {
                        option_id: o.option_id.clone(),
                    }
                }),
            PermissionPolicy::Ask => self.queue_permission(params).await,
        };
        Ok(RequestPermissionResult {
            outcome,
            meta: None,
        })
    }

    async fn read_text_file(
        &self,
        params: ReadTextFileParams,
    ) -> Result<ReadTextFileResult, JsonRpcError> {
        let path = self.resolve(&params.path)?;
        let text = tokio::fs::read_to_string(&path).await.map_err(|source| {
            JsonRpcError::internal_error(format!("cannot read {}: {source}", path.display()))
        })?;
        let content = slice_lines(&text, params.line, params.limit);
        Ok(ReadTextFileResult {
            content,
            meta: None,
        })
    }

    async fn write_text_file(
        &self,
        params: WriteTextFileParams,
    ) -> Result<WriteTextFileResult, JsonRpcError> {
        let path = self.resolve(&params.path)?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|source| {
                JsonRpcError::internal_error(format!(
                    "cannot create {}: {source}",
                    parent.display()
                ))
            })?;
        }
        tokio::fs::write(&path, &params.content)
            .await
            .map_err(|source| {
                JsonRpcError::internal_error(format!("cannot write {}: {source}", path.display()))
            })?;
        Ok(WriteTextFileResult { meta: None })
    }

    async fn terminal_create(
        &self,
        params: TerminalCreateParams,
    ) -> Result<TerminalCreateResult, JsonRpcError> {
        terminal::create(&self.rt.terminals, &self.rt.next_term, params, &self.cwd).await
    }

    async fn terminal_output(
        &self,
        params: TerminalOutputParams,
    ) -> Result<TerminalOutputResult, JsonRpcError> {
        Ok(self.terminal(&params.terminal_id).await?.output())
    }

    async fn terminal_wait_for_exit(
        &self,
        params: TerminalWaitForExitParams,
    ) -> Result<TerminalExitStatus, JsonRpcError> {
        self.terminal(&params.terminal_id)
            .await?
            .wait_for_exit()
            .await
    }

    async fn terminal_kill(
        &self,
        params: TerminalKillParams,
    ) -> Result<TerminalKillResult, JsonRpcError> {
        self.terminal(&params.terminal_id).await?.kill().await?;
        Ok(TerminalKillResult { meta: None })
    }

    async fn terminal_release(
        &self,
        params: TerminalReleaseParams,
    ) -> Result<TerminalReleaseResult, JsonRpcError> {
        // Dropping the entry kills the child if still running (kill_on_drop).
        self.rt.terminals.lock().await.remove(&params.terminal_id);
        Ok(TerminalReleaseResult { meta: None })
    }
}

impl SubagentHandler {
    /// Queue a permission request and await the `permit` tool's answer.
    async fn queue_permission(&self, params: RequestPermissionParams) -> RequestPermissionOutcome {
        let (tx, rx) = oneshot::channel();
        let request_id = {
            let mut inner = self.rt.inner.lock().expect("inner poisoned");
            let id = format!("perm-{}", self.rt.next_perm.fetch_add(1, Ordering::Relaxed));
            inner.pending.push(PendingPermission {
                request_id: id.clone(),
                tool_call: params.tool_call.clone(),
                options: params.options,
                answer: tx,
            });
            inner.status = Status::NeedsPermission;
            id
        };
        debug!(subagent = %self.name, %request_id, "permission request queued");
        self.rt.notify.notify_waiters();
        let outcome = rx.await.unwrap_or(RequestPermissionOutcome::Cancelled);
        {
            let mut inner = self.rt.inner.lock().expect("inner poisoned");
            if inner.pending.is_empty() && matches!(inner.status, Status::NeedsPermission) {
                inner.status = Status::Running;
            }
        }
        self.rt.notify.notify_waiters();
        outcome
    }

    /// Look up a terminal by id.
    async fn terminal(&self, id: &str) -> Result<Arc<Terminal>, JsonRpcError> {
        self.rt
            .terminals
            .lock()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| JsonRpcError::invalid_params(format!("unknown terminal '{id}'")))
    }

    /// Resolve `path` against the session cwd and enforce the
    /// `allow_outside_cwd` boundary. Symlinks are resolved when the path (or
    /// its parent) exists.
    fn resolve(&self, path: &Path) -> Result<PathBuf, JsonRpcError> {
        let candidate = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        };
        let normalized = canonicalize_lenient(&candidate);
        if !self.allow_outside_cwd && !normalized.starts_with(&self.cwd) {
            return Err(JsonRpcError::internal_error(format!(
                "{} is outside the session cwd {}",
                path.display(),
                self.cwd.display()
            )));
        }
        Ok(normalized)
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
