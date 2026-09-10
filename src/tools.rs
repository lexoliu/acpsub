//! The MCP tools: one `Tool` impl per subagent operation.
//!
//! Each tool's `Arguments` rustdoc is the description the orchestrating model
//! reads, so they are written as instructions, not labels.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use aither_acp::{ConfigOption, ConfigOptionValue, ConfigSelectOptions, RequestPermissionOutcome};
use aither_core::llm::tool::{Tool, Tools};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::debug;

use crate::config::{ConfigValue, PermissionPolicy};
use crate::error::{Error, Result};
use crate::registry::RegistryEntry;
use crate::state::{AppState, Launch, Status, Subagent, launch, start_turn};
use crate::transcript::{RenderOptions, render};

/// `wait`'s default timeout.
const DEFAULT_WAIT_SECS: u64 = 600;
/// `wait`'s maximum timeout.
const MAX_WAIT_SECS: u64 = 3600;
/// `wait_any` polls the named subagents this often.
const WAIT_ANY_POLL: Duration = Duration::from_millis(50);

/// Build the full tool set over `state`.
///
/// # Errors
///
/// Returns an error if a tool fails to register (duplicate name or empty
/// description — both would be a bug in this crate).
pub fn build_tools(state: Arc<AppState>) -> aither_core::Result<Tools> {
    let mut tools = Tools::new();
    tools.register(SpawnTool(state.clone()))?;
    tools.register(SendTool(state.clone()))?;
    tools.register(WaitTool(state.clone()))?;
    tools.register(WaitAnyTool(state.clone()))?;
    tools.register(StatusTool(state.clone()))?;
    tools.register(ResultTool(state.clone()))?;
    tools.register(CancelTool(state.clone()))?;
    tools.register(PermitTool(state.clone()))?;
    tools.register(TranscriptTool(state.clone()))?;
    tools.register(ListTool(state.clone()))?;
    tools.register(CloseTool(state.clone()))?;
    tools.register(ForgetTool(state.clone()))?;
    tools.register(AgentsTool(state))?;
    Ok(tools)
}

/// The `spawn` tool's state summary.
fn spawned_view(sub: &Subagent) -> Value {
    let (session_id, state) = {
        let inner = sub.rt.inner.lock().expect("inner poisoned");
        (inner.session_id.clone(), inner.status.name())
    };
    json!({
        "name": sub.name,
        "agent": sub.agent,
        "session_id": session_id,
        "state": state,
        "cwd": sub.cwd,
        "transcript": sub.transcript_path,
    })
}

/// A `wait`-style result for one subagent.
fn wait_view(name: &str, sub: &Subagent, started: Instant) -> Value {
    let (status, reply, tool_calls, pending) = {
        let inner = sub.rt.inner.lock().expect("inner poisoned");
        let turn = inner.current.as_ref().or_else(|| inner.turns.last());
        let pending = inner.pending.first().map(|perm| {
            json!({
                "request_id": perm.request_id,
                "tool_call": {
                    "id": perm.tool_call.tool_call_id,
                    "title": perm.tool_call.title,
                    "kind": perm.tool_call.kind,
                    "status": perm.tool_call.status,
                },
                "options": perm.options.iter().map(|o| json!({
                    "option_id": o.option_id,
                    "name": o.name,
                    "kind": o.kind,
                })).collect::<Vec<_>>(),
            })
        });
        (
            inner.status.clone(),
            turn.map_or_else(String::new, |turn| turn.reply.clone()),
            turn.map_or_else(Vec::new, |turn| {
                turn.tool_calls.values().cloned().collect::<Vec<_>>()
            }),
            pending,
        )
    };
    let mut view = json!({
        "name": name,
        "state": status.name(),
        "reply": reply,
        "tool_calls": tool_calls,
        "elapsed_secs": started.elapsed().as_secs_f64(),
    });
    match &status {
        Status::Done(reason) => {
            view["stop_reason"] = serde_json::to_value(reason).unwrap_or_default();
        }
        Status::Cancelled => {
            view["stop_reason"] = json!("cancelled");
        }
        Status::Failed(reason) => {
            view["error"] = json!(reason);
        }
        _ => {}
    }
    if let Some(pending) = pending {
        view["pending_permission"] = pending;
    }
    view
}

/// Close procedure shared by the `close` and `forget` tools.
async fn close_sub(sub: &Subagent) {
    sub.rt.closing.store(true, Ordering::Relaxed);
    {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        for pending in inner.pending.drain(..) {
            let _ = pending.answer.send(RequestPermissionOutcome::Cancelled);
        }
        if matches!(inner.status, Status::Running | Status::NeedsPermission) {
            inner.status = Status::Cancelled;
        }
    }
    sub.client.clone().close();
    let conn = sub.conn.lock().expect("conn poisoned").take();
    if let Some(conn) = conn
        && let Err(error) = conn.await
    {
        debug!(%error, "connection task join failed");
    }
    let pump = sub.stderr_pump.lock().expect("stderr pump poisoned").take();
    if let Some(pump) = pump {
        pump.abort();
    }
}

// ---------------------------------------------------------------------------
// spawn
// ---------------------------------------------------------------------------

/// Spawn a named subagent: start the configured agent process, open an ACP
/// session in `cwd`, and send `prompt` as its first turn.
///
/// Returns immediately with the session id; the turn runs in the background.
/// Use `wait` for the reply, `send` for follow-ups, `transcript` for the full
/// log. Spawning a name that is registered to a previous session resumes it
/// with `session/load` when the agent supports it; otherwise it errors —
/// pass `replace: true` to start over.
#[derive(Debug, Deserialize, JsonSchema)]
struct SpawnArgs {
    /// Name for this subagent (`[A-Za-z0-9._-]+`). Unique while alive; a
    /// registered name resumes its session.
    name: String,
    /// Configured agent key (`[agents.<key>]` in the config file).
    agent: String,
    /// Working directory of the session. Agent fs/terminal access is confined
    /// to it unless the agent config sets `allow_outside_cwd`.
    cwd: PathBuf,
    /// The first turn's prompt — the task for the subagent.
    prompt: String,
    /// Session mode to set (`session/set_mode`), overriding the agent's
    /// configured `mode`.
    mode: Option<String>,
    /// Session config options to set (`session/set_config_option`), merged
    /// over the agent's configured `config`: option id → string or boolean.
    config: Option<BTreeMap<String, ConfigValue>>,
    /// Permission policy for this subagent: `allow` auto-approves, `deny`
    /// auto-rejects, `ask` queues requests for the `permit` tool.
    permission: Option<PermissionPolicy>,
    /// Forget the registered session under `name` and start fresh.
    replace: Option<bool>,
}

struct SpawnTool(Arc<AppState>);

impl Tool for SpawnTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("spawn")
    }
    type Arguments = SpawnArgs;
    type Res = Value;

    async fn call(&self, args: SpawnArgs) -> aither_core::Result<Value> {
        let sub = launch(
            &self.0,
            Launch {
                name: args.name,
                agent: args.agent,
                cwd: args.cwd,
                prompt: args.prompt,
                mode: args.mode,
                config: args.config.unwrap_or_default(),
                permission: args.permission,
                replace: args.replace.unwrap_or(false),
            },
        )
        .await?;
        Ok(spawned_view(&sub))
    }
}

// ---------------------------------------------------------------------------
// send
// ---------------------------------------------------------------------------

/// Send a follow-up prompt to a live subagent: a new turn on the same session.
///
/// Only valid when the subagent is `idle`, `done`, or `cancelled`; a `running`
/// or `needs_permission` subagent must be waited on or cancelled first.
/// Returns immediately; use `wait` for the reply.
#[derive(Debug, Deserialize, JsonSchema)]
struct SendArgs {
    /// Subagent name.
    name: String,
    /// The prompt for the next turn.
    prompt: String,
}

struct SendTool(Arc<AppState>);

impl Tool for SendTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("send")
    }
    type Arguments = SendArgs;
    type Res = Value;

    async fn call(&self, args: SendArgs) -> aither_core::Result<Value> {
        let sub = self.0.get(&args.name)?;
        start_turn(self.0.clone(), &sub, args.prompt).await?;
        Ok(json!({"name": args.name, "state": "running"}))
    }
}

// ---------------------------------------------------------------------------
// wait / wait_any
// ---------------------------------------------------------------------------

/// Block until a subagent's turn ends, a permission request needs an answer,
/// or the timeout expires.
///
/// Returns the subagent state (`done`/`cancelled`/`failed`/
/// `needs_permission`/`running`), the turn's `reply` (concatenated agent
/// message text), its tool calls, and — when `needs_permission` — the
/// `pending_permission` request to answer with `permit`.
#[derive(Debug, Deserialize, JsonSchema)]
struct WaitArgs {
    /// Subagent name.
    name: String,
    /// Seconds to wait (default 600, max 3600). On expiry the result reports
    /// the still-current state.
    timeout_secs: Option<u64>,
}

struct WaitTool(Arc<AppState>);

impl Tool for WaitTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("wait")
    }
    type Arguments = WaitArgs;
    type Res = Value;

    async fn call(&self, args: WaitArgs) -> aither_core::Result<Value> {
        let sub = self.0.get(&args.name)?;
        let timeout = Duration::from_secs(
            args.timeout_secs
                .unwrap_or(DEFAULT_WAIT_SECS)
                .min(MAX_WAIT_SECS),
        );
        let started = Instant::now();
        loop {
            {
                let inner = sub.rt.inner.lock().expect("inner poisoned");
                if !matches!(inner.status, Status::Running) {
                    break;
                }
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                break;
            }
            tokio::select! {
                () = sub.rt.notify.notified() => {},
                () = tokio::time::sleep(remaining.min(Duration::from_millis(250))) => {},
            }
        }
        Ok(wait_view(&args.name, &sub, started))
    }
}

/// Block until the first of several subagents leaves `running`, then return
/// that one's wait-style result.
#[derive(Debug, Deserialize, JsonSchema)]
struct WaitAnyArgs {
    /// Subagent names to watch.
    names: Vec<String>,
    /// Seconds to wait (default 600, max 3600).
    timeout_secs: Option<u64>,
}

struct WaitAnyTool(Arc<AppState>);

impl Tool for WaitAnyTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("wait_any")
    }
    type Arguments = WaitAnyArgs;
    type Res = Value;

    async fn call(&self, args: WaitAnyArgs) -> aither_core::Result<Value> {
        let subs = args
            .names
            .iter()
            .map(|name| self.0.get(name).map(|sub| (name.clone(), sub)))
            .collect::<Result<Vec<_>>>()?;
        let timeout = Duration::from_secs(
            args.timeout_secs
                .unwrap_or(DEFAULT_WAIT_SECS)
                .min(MAX_WAIT_SECS),
        );
        let started = Instant::now();
        loop {
            for (name, sub) in &subs {
                let done = {
                    let inner = sub.rt.inner.lock().expect("inner poisoned");
                    !matches!(inner.status, Status::Running)
                };
                if done {
                    return Ok(wait_view(name, sub, started));
                }
            }
            if timeout.saturating_sub(started.elapsed()).is_zero() {
                return Ok(json!({
                    "state": "running",
                    "names": args.names,
                    "elapsed_secs": started.elapsed().as_secs_f64(),
                }));
            }
            tokio::time::sleep(WAIT_ANY_POLL).await;
        }
    }
}

// ---------------------------------------------------------------------------
// status / result
// ---------------------------------------------------------------------------

/// Report a subagent's state: live status, session id, turn count, last stop
/// reason, transcript path; or its registry entry when closed.
#[derive(Debug, Deserialize, JsonSchema)]
struct StatusArgs {
    /// Subagent name.
    name: String,
}

struct StatusTool(Arc<AppState>);

impl Tool for StatusTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("status")
    }
    type Arguments = StatusArgs;
    type Res = Value;

    fn call(&self, args: StatusArgs) -> impl Future<Output = aither_core::Result<Value>> + Send {
        std::future::ready(self.status(args).map_err(Into::into))
    }
}

impl StatusTool {
    /// Synchronous body: `status` only reads shared state.
    fn status(&self, args: StatusArgs) -> Result<Value> {
        let sub = self
            .0
            .live
            .lock()
            .expect("live poisoned")
            .get(&args.name)
            .cloned();
        if let Some(sub) = sub {
            let (status, session_id, turns, last_stop, pending) = {
                let inner = sub.rt.inner.lock().expect("inner poisoned");
                (
                    inner.status.clone(),
                    inner.session_id.clone(),
                    inner.turn_offset + inner.turns.len() as u64,
                    inner.turns.last().and_then(|turn| turn.stop_reason),
                    inner
                        .pending
                        .iter()
                        .map(|p| p.request_id.clone())
                        .collect::<Vec<_>>(),
                )
            };
            let mut view = json!({
                "name": sub.name,
                "agent": sub.agent,
                "live": true,
                "state": status.name(),
                "session_id": session_id,
                "cwd": sub.cwd,
                "turns": turns,
                "transcript": sub.transcript_path,
                "permission": serde_json::to_value(sub.permission).unwrap_or_default(),
                "pending_permissions": pending,
            });
            if let Some(reason) = last_stop {
                view["last_stop_reason"] = serde_json::to_value(reason).unwrap_or_default();
            }
            if let Status::Failed(reason) = &status {
                view["error"] = json!(reason);
            }
            return Ok(view);
        }
        let entry = self
            .0
            .registry
            .lock()
            .expect("registry poisoned")
            .get(&args.name)
            .cloned();
        if let Some(entry) = entry {
            return Ok(json!({
                "name": args.name,
                "agent": entry.agent,
                "live": false,
                "state": "closed",
                "session_id": entry.session_id,
                "cwd": entry.cwd,
                "turns": entry.turns,
                "created": entry.created,
                "last_turn": entry.last_turn,
            }));
        }
        Err(Error::UnknownSubagent(args.name))
    }
}

/// Return a turn's reply: the last turn by default, or turn `n` (1-based).
#[derive(Debug, Deserialize, JsonSchema)]
struct ResultArgs {
    /// Subagent name.
    name: String,
    /// 1-based turn number; defaults to the latest turn.
    turn: Option<u64>,
}

struct ResultTool(Arc<AppState>);

impl Tool for ResultTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("result")
    }
    type Arguments = ResultArgs;
    type Res = Value;

    fn call(&self, args: ResultArgs) -> impl Future<Output = aither_core::Result<Value>> + Send {
        std::future::ready(self.result(args).map_err(Into::into))
    }
}

impl ResultTool {
    /// Synchronous body: `result` only reads shared state.
    fn result(&self, args: ResultArgs) -> Result<Value> {
        let sub = self.0.get(&args.name)?;
        let (n, reply, stop_reason) = {
            let inner = sub.rt.inner.lock().expect("inner poisoned");
            let turn = args.turn.map_or_else(
                || inner.current.as_ref().or_else(|| inner.turns.last()),
                |n| {
                    n.checked_sub(inner.turn_offset + 1)
                        .and_then(|i| inner.turns.get(usize::try_from(i).unwrap_or(usize::MAX)))
                        .or_else(|| inner.current.as_ref().filter(|turn| turn.n == n))
                },
            );
            match turn {
                Some(turn) => (turn.n, turn.reply.clone(), turn.stop_reason),
                None => {
                    return Err(Error::NoSuchTurn {
                        name: args.name,
                        turn: args.turn.unwrap_or(0),
                        have: inner.turn_offset + inner.turns.len() as u64,
                    });
                }
            }
        };
        let mut view = json!({
            "name": args.name,
            "turn": n,
            "reply": reply,
        });
        if let Some(reason) = stop_reason {
            view["stop_reason"] = serde_json::to_value(reason).unwrap_or_default();
        }
        Ok(view)
    }
}

// ---------------------------------------------------------------------------
// cancel / permit
// ---------------------------------------------------------------------------

/// Cancel a subagent's running turn: answers every queued permission request
/// `cancelled`, then sends `session/cancel`. The turn ends with the agent's
/// `cancelled` stop reason.
#[derive(Debug, Deserialize, JsonSchema)]
struct CancelArgs {
    /// Subagent name.
    name: String,
}

struct CancelTool(Arc<AppState>);

impl Tool for CancelTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("cancel")
    }
    type Arguments = CancelArgs;
    type Res = Value;

    async fn call(&self, args: CancelArgs) -> aither_core::Result<Value> {
        let sub = self.0.get(&args.name)?;
        {
            let mut inner = sub.rt.inner.lock().expect("inner poisoned");
            if !matches!(inner.status, Status::Running | Status::NeedsPermission) {
                return Err(Error::NotRunning(args.name).into());
            }
            for pending in inner.pending.drain(..) {
                let _ = pending.answer.send(RequestPermissionOutcome::Cancelled);
            }
        }
        let session_id = sub
            .rt
            .inner
            .lock()
            .expect("inner poisoned")
            .session_id
            .clone();
        sub.client
            .cancel(&session_id)
            .await
            .map_err(|source| Error::Agent {
                agent: args.name.clone(),
                source,
            })?;
        Ok(json!({"name": args.name, "cancelled": true}))
    }
}

/// Answer a permission request queued by the `ask` policy.
///
/// `request_id` and `option_id` come from `wait`'s `pending_permission` field.
#[derive(Debug, Deserialize, JsonSchema)]
struct PermitArgs {
    /// Subagent name.
    name: String,
    /// Pending permission request id (`perm-N`).
    request_id: String,
    /// The option id to select (one of the request's `options`).
    option_id: String,
}

struct PermitTool(Arc<AppState>);

impl Tool for PermitTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("permit")
    }
    type Arguments = PermitArgs;
    type Res = Value;

    fn call(&self, args: PermitArgs) -> impl Future<Output = aither_core::Result<Value>> + Send {
        std::future::ready(self.permit(&args).map_err(Into::into))
    }
}

impl PermitTool {
    /// Synchronous body: `permit` only touches shared state and a oneshot.
    fn permit(&self, args: &PermitArgs) -> Result<Value> {
        let sub = self.0.get(&args.name)?;
        let pending = {
            let mut inner = sub.rt.inner.lock().expect("inner poisoned");
            let Some(position) = inner
                .pending
                .iter()
                .position(|p| p.request_id == args.request_id)
            else {
                return Err(Error::NoSuchPermission {
                    name: args.name.clone(),
                    request: args.request_id.clone(),
                });
            };
            let pending = &inner.pending[position];
            if pending
                .options
                .iter()
                .all(|o| o.option_id != args.option_id)
            {
                return Err(Error::BadOption {
                    request: args.request_id.clone(),
                    option: args.option_id.clone(),
                    offered: pending
                        .options
                        .iter()
                        .map(|o| o.option_id.clone())
                        .collect(),
                });
            }
            inner.pending.remove(position)
        };
        let _ = pending.answer.send(RequestPermissionOutcome::Selected {
            option_id: args.option_id.clone(),
        });
        // The handler recomputes this too once its await resumes, but callers
        // read the state as soon as `permit` returns.
        {
            let mut inner = sub.rt.inner.lock().expect("inner poisoned");
            if inner.pending.is_empty() && matches!(inner.status, Status::NeedsPermission) {
                inner.status = Status::Running;
            }
        }
        sub.rt.notify.notify_waiters();
        Ok(json!({
            "name": args.name,
            "request_id": args.request_id,
            "option_id": args.option_id,
        }))
    }
}

// ---------------------------------------------------------------------------
// transcript / list / close / forget / agents
// ---------------------------------------------------------------------------

/// Render a subagent's transcript: every prompt, streamed message, thinking
/// chunk (with `thinking`), tool call and result, plan, and mode change, in
/// order.
#[derive(Debug, Deserialize, JsonSchema)]
struct TranscriptArgs {
    /// Subagent name.
    name: String,
    /// Start rendering at record N (skip the first N records).
    from: Option<usize>,
    /// Show only the last N records.
    tail: Option<usize>,
    /// Do not clip long values (default clips at 400 chars).
    full: Option<bool>,
    /// Include `agent_thought_chunk` records.
    thinking: Option<bool>,
}

struct TranscriptTool(Arc<AppState>);

impl Tool for TranscriptTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("transcript")
    }
    type Arguments = TranscriptArgs;
    type Res = String;

    async fn call(&self, args: TranscriptArgs) -> aither_core::Result<String> {
        let path = match self.0.live.lock().expect("live poisoned").get(&args.name) {
            Some(sub) => sub.transcript_path.clone(),
            None => self
                .0
                .config
                .defaults
                .transcript_dir
                .join(format!("{}.jsonl", args.name)),
        };
        let text =
            tokio::fs::read_to_string(&path)
                .await
                .map_err(|source| match source.kind() {
                    std::io::ErrorKind::NotFound => Error::NoTranscript(path.clone()),
                    _ => Error::io(format!("cannot read {}", path.display()), source),
                })?;
        Ok(render(
            &text,
            &RenderOptions {
                from: args.from.unwrap_or(0),
                tail: args.tail,
                full: args.full.unwrap_or(false),
                thinking: args.thinking.unwrap_or(false),
            },
        ))
    }
}

/// List subagents: live ones with their state, and registered (closed but
/// resumable) ones from the registry.
#[derive(Debug, Deserialize, JsonSchema)]
struct ListArgs {}

struct ListTool(Arc<AppState>);

/// `list` view of a live subagent.
fn live_view(name: &str, sub: &Subagent) -> Value {
    let (state, session_id, turns) = {
        let inner = sub.rt.inner.lock().expect("inner poisoned");
        (
            inner.status.name(),
            inner.session_id.clone(),
            inner.turn_offset + inner.turns.len() as u64,
        )
    };
    json!({
        "name": name,
        "agent": sub.agent,
        "live": true,
        "state": state,
        "session_id": session_id,
        "cwd": sub.cwd,
        "turns": turns,
    })
}

/// `list` view of a registered but closed subagent.
fn closed_view(name: &str, entry: &RegistryEntry) -> Value {
    json!({
        "name": name,
        "agent": entry.agent,
        "live": false,
        "state": "closed",
        "session_id": entry.session_id,
        "cwd": entry.cwd,
        "turns": entry.turns,
    })
}

impl Tool for ListTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("list")
    }
    type Arguments = ListArgs;
    type Res = Value;

    fn call(&self, _args: ListArgs) -> impl Future<Output = aither_core::Result<Value>> + Send {
        std::future::ready(Ok(self.list()))
    }
}

impl ListTool {
    /// Synchronous body: `list` only reads shared state.
    fn list(&self) -> Value {
        let (live_views, registered) = {
            let live = self.0.live.lock().expect("live poisoned");
            let registry = self.0.registry.lock().expect("registry poisoned");
            (
                live.iter()
                    .map(|(name, sub)| (name.clone(), live_view(name, sub)))
                    .collect::<BTreeMap<_, _>>(),
                registry
                    .iter()
                    .map(|(name, entry)| (name.clone(), entry.clone()))
                    .collect::<BTreeMap<_, _>>(),
            )
        };
        let mut by_name: BTreeMap<String, Value> = registered
            .iter()
            .map(|(name, entry)| (name.clone(), closed_view(name, entry)))
            .collect();
        by_name.extend(live_views);
        json!({"subagents": by_name.into_values().collect::<Vec<_>>()})
    }
}

/// Close a subagent: end the agent process. The registry entry (and so the
/// session's resumability) is kept; use `forget` to remove it.
#[derive(Debug, Deserialize, JsonSchema)]
struct CloseArgs {
    /// Subagent name.
    name: String,
}

struct CloseTool(Arc<AppState>);

impl Tool for CloseTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("close")
    }
    type Arguments = CloseArgs;
    type Res = Value;

    async fn call(&self, args: CloseArgs) -> aither_core::Result<Value> {
        let sub = {
            self.0
                .live
                .lock()
                .expect("live poisoned")
                .remove(&args.name)
        };
        match sub {
            Some(sub) => {
                close_sub(&sub).await;
                Ok(json!({"name": args.name, "closed": true}))
            }
            None if self
                .0
                .registry
                .lock()
                .expect("registry poisoned")
                .get(&args.name)
                .is_some() =>
            {
                Ok(json!({"name": args.name, "closed": true, "was_live": false}))
            }
            None => Err(Error::UnknownSubagent(args.name).into()),
        }
    }
}

/// Forget a subagent: close it if live and remove its registry entry. The
/// transcript file is kept.
#[derive(Debug, Deserialize, JsonSchema)]
struct ForgetArgs {
    /// Subagent name.
    name: String,
}

struct ForgetTool(Arc<AppState>);

impl Tool for ForgetTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("forget")
    }
    type Arguments = ForgetArgs;
    type Res = Value;

    async fn call(&self, args: ForgetArgs) -> aither_core::Result<Value> {
        let sub = {
            self.0
                .live
                .lock()
                .expect("live poisoned")
                .remove(&args.name)
        };
        if let Some(sub) = &sub {
            close_sub(sub).await;
        }
        let registered = self
            .0
            .registry
            .lock()
            .expect("registry poisoned")
            .remove(&args.name);
        if sub.is_none() && !registered {
            return Err(Error::UnknownSubagent(args.name).into());
        }
        self.0.save_registry().await?;
        Ok(json!({"name": args.name, "forgotten": true}))
    }
}

/// List the configured agents; for agents with a live subagent also report
/// their `agentInfo`, modes, and config options.
#[derive(Debug, Deserialize, JsonSchema)]
struct AgentsArgs {}

struct AgentsTool(Arc<AppState>);

impl Tool for AgentsTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("agents")
    }
    type Arguments = AgentsArgs;
    type Res = Value;

    fn call(&self, _args: AgentsArgs) -> impl Future<Output = aither_core::Result<Value>> + Send {
        std::future::ready(Ok(self.agents()))
    }
}

impl AgentsTool {
    /// Synchronous body: `agents` only reads shared state.
    fn agents(&self) -> Value {
        // Per-agent: names of its live subagents and the info the first such
        // process reported (agent_info, modes, config options).
        let live: Vec<Arc<Subagent>> = self
            .0
            .live
            .lock()
            .expect("live poisoned")
            .values()
            .cloned()
            .collect();
        let mut out = Vec::new();
        for (name, agent) in &self.0.config.agents {
            let subs: Vec<&str> = live
                .iter()
                .filter(|sub| sub.agent == *name)
                .map(|sub| sub.name.as_str())
                .collect();
            let mut view = json!({
                "name": name,
                "command": agent.command,
                "args": agent.args,
                "mode": agent.mode,
                "allow_outside_cwd": agent.allow_outside_cwd,
                "subagents": subs,
            });
            // Agent info is known only once a process initialized.
            if let Some(sub) = live.iter().find(|sub| sub.agent == *name) {
                let (info, modes, options) = {
                    let inner = sub.rt.inner.lock().expect("inner poisoned");
                    (
                        inner
                            .agent_info
                            .as_ref()
                            .map(|info| serde_json::to_value(info).unwrap_or_default()),
                        inner.modes.as_ref().map(|modes| {
                            json!({
                                "current": modes.current_mode_id,
                                "available": modes.available_modes.iter().map(|m| json!({
                                    "id": m.id, "name": m.name,
                                })).collect::<Vec<_>>(),
                            })
                        }),
                        inner
                            .config_options
                            .iter()
                            .map(config_option_view)
                            .collect::<Vec<_>>(),
                    )
                };
                if let Some(info) = info {
                    view["agent_info"] = info;
                }
                if let Some(modes) = modes {
                    view["modes"] = modes;
                }
                if !options.is_empty() {
                    view["config_options"] = json!(options);
                }
            }
            out.push(view);
        }
        json!({"agents": out})
    }
}

/// Compact view of a session `ConfigOption`.
fn config_option_view(option: &ConfigOption) -> Value {
    let current = option.current_value.as_ref().map(|value| match value {
        ConfigOptionValue::Selected(id) => json!(id),
        ConfigOptionValue::Toggle(flag) => json!(flag),
    });
    let options = option.options.as_ref().map(|options| match options {
        ConfigSelectOptions::Flat(flat) => {
            json!(flat.iter().map(|o| json!({"value": o.value, "name": o.name})).collect::<Vec<_>>())
        }
        ConfigSelectOptions::Grouped(groups) => json!(groups.iter().map(|g| json!({
            "group": g.group,
            "name": g.name,
            "options": g.options.iter().map(|o| json!({"value": o.value, "name": o.name})).collect::<Vec<_>>(),
        })).collect::<Vec<_>>()),
    });
    json!({
        "id": option.id,
        "name": option.name,
        "type": option.kind,
        "category": option.category,
        "current_value": current,
        "options": options,
    })
}
