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
use crate::state::{
    AppState, Gate, Launch, Status, Subagent, begin_turn, launch, prompt_slot_free,
    record_queue_dropped, spawn_turn_task, start_turn, steer_resolved, steer_turn, steer_waiter,
};
use crate::transcript::{RenderOptions, render};

/// `wait`'s minimum timeout: shorter polls belong to `status`.
const MIN_WAIT_SECS: u64 = 60;
/// `wait`'s default timeout.
const DEFAULT_WAIT_SECS: u64 = 600;
/// `wait`'s maximum timeout.
const MAX_WAIT_SECS: u64 = 3600;
/// `wait_any` polls the named subagents this often.
const WAIT_ANY_POLL: Duration = Duration::from_millis(50);
/// How long `send` waits for a synchronous rejection of a steer before
/// reporting `steered`; an accepted steer resolves only at turn end, so a
/// longer wait would just stall the caller.
const STEER_ACK: Duration = Duration::from_millis(250);

/// Build the full tool set over `state`.
///
/// # Errors
///
/// Returns an error if a tool fails to register (duplicate name or empty
/// description — both would be a bug in this crate).
pub fn build_tools(state: Arc<AppState>) -> aither_core::Result<Tools> {
    let mut tools = Tools::new();
    tools.register(SpawnTool(state.clone()))?;
    tools.register(AdoptTool(state.clone()))?;
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

/// The `spawn`/`adopt` tool's state summary.
fn spawned_view(sub: &Subagent) -> Value {
    let state = sub.rt.inner.lock().expect("inner poisoned").status.name();
    json!({
        "session_id": sub.session_id,
        "agent": sub.agent,
        "state": state,
        "cwd": sub.cwd,
        "transcript": sub.transcript_path,
    })
}

/// A `wait`-style result for one subagent.
fn wait_view(sub: &Subagent, started: Instant) -> Value {
    let (status, reply, tool_calls, pending, turn_n, queued) = {
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
            turn.map(|turn| turn.n),
            inner.queue.len(),
        )
    };
    let mut view = json!({
        "session_id": sub.session_id,
        "state": status.name(),
        "turn": turn_n,
        "reply": reply,
        "tool_calls": tool_calls,
        "queued": queued,
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
        inner.queue.clear();
        inner.steer_pending.clear();
        inner.steer_owner = None;
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
// spawn / adopt
// ---------------------------------------------------------------------------

/// Spawn a subagent: start the configured agent process, open a new ACP
/// session in `cwd`, and send `prompt` as its first turn.
///
/// Returns immediately with `session_id` — the handle every other tool
/// addresses (`wait` for the reply, `send` for follow-ups, `transcript` for
/// the full log). The turn runs in the background. To take over an existing
/// session instead of starting a new one, use `adopt`.
#[derive(Debug, Deserialize, JsonSchema)]
struct SpawnArgs {
    /// Configured agent key (`[agents.<key>]` in the config file). Optional:
    /// falls back to `[defaults] agent`, then to the only configured agent.
    agent: Option<String>,
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
}

struct SpawnTool(Arc<AppState>);

impl Tool for SpawnTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("spawn")
    }
    type Arguments = SpawnArgs;
    type Res = Value;

    async fn call(&self, args: SpawnArgs) -> aither_core::Result<Value> {
        let (agent, _) = self.0.config.resolve_agent(args.agent.as_deref())?;
        let sub = launch(
            &self.0,
            Launch {
                agent: agent.clone(),
                cwd: args.cwd,
                prompt: Some(args.prompt),
                mode: args.mode,
                config: args.config.unwrap_or_default(),
                permission: args.permission,
                load: None,
            },
        )
        .await?;
        Ok(spawned_view(&sub))
    }
}

/// Adopt an existing session: start the agent process and bind it to a
/// session created earlier (by `spawn`, or outside acpsub — e.g. a `devin`
/// CLI session) via `session/load`.
///
/// The session becomes a live subagent addressed by `session_id` — `send`,
/// `wait`, `transcript`, and the rest work as usual. When `prompt` is given
/// its turn starts immediately; otherwise the session idles until `send`.
///
/// `cwd` is optional: it is taken from the registry for sessions acpsub
/// knows, or discovered from the agent's session database (e.g. devin's
/// local store). Only pass it when discovery cannot find the session.
#[derive(Debug, Deserialize, JsonSchema)]
struct AdoptArgs {
    /// The existing ACP session id to take over.
    session_id: String,
    /// First turn's prompt after adopting. Omit to bind the session without
    /// starting a turn — direct it later with `send`.
    prompt: Option<String>,
    /// Configured agent key. Optional: a registered session resolves to the
    /// agent that created it; otherwise `[defaults] agent`, then the only
    /// configured agent.
    agent: Option<String>,
    /// Session working directory. Usually omitted — read from the registry
    /// or the agent's session database. Required only when neither knows the
    /// session.
    cwd: Option<PathBuf>,
    /// Permission policy override, as in `spawn`.
    permission: Option<PermissionPolicy>,
}

struct AdoptTool(Arc<AppState>);

impl Tool for AdoptTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("adopt")
    }
    type Arguments = AdoptArgs;
    type Res = Value;

    async fn call(&self, args: AdoptArgs) -> aither_core::Result<Value> {
        let registered = self
            .0
            .registry
            .lock()
            .expect("registry poisoned")
            .get(&args.session_id)
            .cloned();
        if let (Some(requested), Some(entry)) = (&args.agent, &registered)
            && *requested != entry.agent
        {
            return Err(Error::AgentMismatch {
                session_id: args.session_id,
                registered: entry.agent.clone(),
                requested: requested.clone(),
            }
            .into());
        }
        let requested = args
            .agent
            .as_deref()
            .or_else(|| registered.as_ref().map(|entry| entry.agent.as_str()));
        let (agent, agent_cfg) = self.0.config.resolve_agent(requested)?;
        let cwd = if let Some(cwd) = args
            .cwd
            .or_else(|| registered.as_ref().map(|entry| entry.cwd.clone()))
        {
            cwd
        } else {
            let unknown = |reason: String| Error::SessionCwdUnknown {
                session_id: args.session_id.clone(),
                reason,
            };
            match agent_cfg.session_db_path() {
                Some(db) => match crate::state::discover_session_cwd(&db, &args.session_id) {
                    Ok(Some(cwd)) => cwd,
                    Ok(None) => {
                        return Err(unknown(format!("not found in {}", db.display())).into());
                    }
                    Err(error) => return Err(unknown(error.to_string()).into()),
                },
                None => {
                    return Err(unknown("no session database configured".to_string()).into());
                }
            }
        };
        let sub = launch(
            &self.0,
            Launch {
                agent: agent.clone(),
                cwd,
                prompt: args.prompt,
                mode: None,
                config: BTreeMap::new(),
                permission: args.permission,
                load: Some(args.session_id),
            },
        )
        .await?;
        Ok(spawned_view(&sub))
    }
}

// ---------------------------------------------------------------------------
// send
// ---------------------------------------------------------------------------

/// Send a prompt to a live subagent. `policy` decides what a `running` (or
/// `needs_permission`) subagent does with it; nothing is queued or injected
/// unless you ask.
///
/// Returns immediately; use `wait` for the reply.
#[derive(Debug, Deserialize, JsonSchema)]
struct SendArgs {
    /// Session id, as returned by `spawn`/`adopt`.
    session_id: String,
    /// The prompt for the next turn.
    prompt: String,
    /// Required: how to deliver while a turn is running.
    ///
    /// `try` — the original behavior: error unless the subagent is `idle`,
    /// `done`, or `cancelled`. `queued` — park the prompt on a FIFO queue;
    /// it fires as the next turn when the current one ends, and is dropped
    /// if that turn is cancelled or fails, or on `cancel`/`close`/`forget`.
    /// `steer` — inject the prompt into the running turn (a second
    /// `session/prompt` while it is in flight); agents that support
    /// mid-turn injection fold it into the active task, agents that do not
    /// surface an error.
    policy: SendPolicy,
}

/// Delivery policy for `send` while a turn is running — no default, the
/// caller chooses explicitly.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum SendPolicy {
    /// Error when the subagent is not `idle`, `done`, or `cancelled`.
    Try,
    /// Park the prompt on the subagent's queue; it fires as the next turn
    /// when the current one ends.
    Queued,
    /// Inject the prompt into the running turn as a steering message.
    Steer,
}

struct SendTool(Arc<AppState>);

impl Tool for SendTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("send")
    }
    type Arguments = SendArgs;
    type Res = Value;

    async fn call(&self, args: SendArgs) -> aither_core::Result<Value> {
        let sub = self.0.get(&args.session_id)?;
        match args.policy {
            SendPolicy::Try => {
                start_turn(self.0.clone(), &sub, args.prompt).await?;
            }
            SendPolicy::Queued => {
                // Promptable → start the turn, otherwise park the prompt. A
                // concurrent `send` can win the start between the check and
                // `start_turn`; retry then lands in the queue.
                enum Action {
                    Fresh,
                    Queued(usize),
                }
                loop {
                    let action = {
                        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
                        if prompt_slot_free(&inner) && inner.queue.is_empty() {
                            Action::Fresh
                        } else if matches!(inner.status, Status::Failed(_)) {
                            return Err(Error::NotPromptable {
                                session_id: args.session_id.clone(),
                                status: inner.status.name().to_string(),
                            }
                            .into());
                        } else {
                            inner.queue.push_back(args.prompt.clone());
                            Action::Queued(inner.queue.len())
                        }
                    };
                    match action {
                        Action::Queued(position) => {
                            sub.rt.notify.notify_waiters();
                            return Ok(json!({
                                "session_id": args.session_id,
                                "state": "queued",
                                "position": position,
                            }));
                        }
                        Action::Fresh => {
                            match start_turn(self.0.clone(), &sub, args.prompt.clone()).await {
                                Ok(()) => break,
                                Err(Error::NotPromptable { .. }) => {}
                                Err(error) => return Err(error.into()),
                            }
                        }
                    }
                }
            }
            SendPolicy::Steer => {
                let wire = sub.rt.prompt_send.lock().await;
                let status = {
                    let inner = sub.rt.inner.lock().expect("inner poisoned");
                    inner.status.clone()
                };
                match status {
                    Status::Idle | Status::Done(_) | Status::Cancelled => {
                        let (turn_n, fut) =
                            begin_turn(&sub, args.prompt, Gate::Check, false).await?;
                        spawn_turn_task(self.0.clone(), &sub, fut, turn_n);
                    }
                    Status::Running | Status::NeedsPermission => {
                        let (steer_id, turn_n, mut fut) = steer_turn(&sub, args.prompt).await?;
                        drop(wire);
                        // An accepted steer resolves only at turn end; a
                        // rejection lands almost at once. Give the agent a
                        // beat to reject before reporting `steered`.
                        if let Ok(outcome) = tokio::time::timeout(STEER_ACK, &mut fut).await {
                            if let Err(source) =
                                steer_resolved(self.0.clone(), &sub, steer_id, turn_n, outcome)
                                    .await
                            {
                                return Err(Error::Agent {
                                    agent: sub.agent.clone(),
                                    source,
                                }
                                .into());
                            }
                        } else {
                            tokio::spawn(steer_waiter(
                                self.0.clone(),
                                sub.clone(),
                                steer_id,
                                turn_n,
                                fut,
                            ));
                        }
                        return Ok(json!({
                            "session_id": args.session_id,
                            "state": "steered",
                            "turn": turn_n,
                        }));
                    }
                    Status::Failed(_) => {
                        drop(wire);
                        return Err(Error::NotPromptable {
                            session_id: args.session_id.clone(),
                            status: status.name().to_string(),
                        }
                        .into());
                    }
                }
            }
        }
        Ok(json!({"session_id": args.session_id, "state": "running"}))
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
    /// Session id, as returned by `spawn`/`adopt`.
    session_id: String,
    /// Seconds to wait (default 600, min 60, max 3600). Prefer long waits —
    /// 300–1800 (5–30 min): the block is event-driven and returns early on
    /// any state change, so a generous timeout is free, while every expiry
    /// costs a model turn just to re-issue the wait on a still-`running`
    /// result. Keep it under 1800 to stay inside the 30-min prompt-cache
    /// TTL. On expiry the result reports the still-current state. For an
    /// instant check use `status` — timeouts under 60 are rejected.
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
        if let Some(secs) = args.timeout_secs
            && secs < MIN_WAIT_SECS
        {
            return Err(Error::TimeoutBelowMin { got: secs }.into());
        }
        let sub = self.0.get(&args.session_id)?;
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
        Ok(wait_view(&sub, started))
    }
}

/// Block until the first of several subagents leaves `running`, then return
/// that one's wait-style result.
#[derive(Debug, Deserialize, JsonSchema)]
struct WaitAnyArgs {
    /// Session ids to watch.
    session_ids: Vec<String>,
    /// Seconds to wait (default 600, min 60, max 3600). Prefer 300–1800
    /// (5–30 min), as with `wait`: early return on the first finisher is
    /// free, but each expiry burns a model turn re-issuing the wait.
    /// Timeouts under 60 are rejected — use `status` for instant checks.
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
        if let Some(secs) = args.timeout_secs
            && secs < MIN_WAIT_SECS
        {
            return Err(Error::TimeoutBelowMin { got: secs }.into());
        }
        let subs = args
            .session_ids
            .iter()
            .map(|session_id| self.0.get(session_id))
            .collect::<Result<Vec<_>>>()?;
        let timeout = Duration::from_secs(
            args.timeout_secs
                .unwrap_or(DEFAULT_WAIT_SECS)
                .min(MAX_WAIT_SECS),
        );
        let started = Instant::now();
        loop {
            for sub in &subs {
                let done = {
                    let inner = sub.rt.inner.lock().expect("inner poisoned");
                    !matches!(inner.status, Status::Running)
                };
                if done {
                    return Ok(wait_view(sub, started));
                }
            }
            if timeout.saturating_sub(started.elapsed()).is_zero() {
                return Ok(json!({
                    "state": "running",
                    "session_ids": args.session_ids,
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

/// Report a subagent's state: live status, turn count, last stop reason,
/// transcript path; or its registry entry when closed.
#[derive(Debug, Deserialize, JsonSchema)]
struct StatusArgs {
    /// Session id, as returned by `spawn`/`adopt`.
    session_id: String,
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
            .get(&args.session_id)
            .cloned();
        if let Some(sub) = sub {
            let (status, turns, last_stop, pending, queued) = {
                let inner = sub.rt.inner.lock().expect("inner poisoned");
                (
                    inner.status.clone(),
                    inner.turn_offset + inner.turns.len() as u64,
                    inner.turns.last().and_then(|turn| turn.stop_reason),
                    inner
                        .pending
                        .iter()
                        .map(|p| p.request_id.clone())
                        .collect::<Vec<_>>(),
                    inner.queue.iter().cloned().collect::<Vec<_>>(),
                )
            };
            let mut view = json!({
                "session_id": sub.session_id,
                "agent": sub.agent,
                "live": true,
                "state": status.name(),
                "cwd": sub.cwd,
                "turns": turns,
                "queued": queued,
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
            .get(&args.session_id)
            .cloned();
        if let Some(entry) = entry {
            return Ok(json!({
                "session_id": args.session_id,
                "agent": entry.agent,
                "live": false,
                "state": "closed",
                "cwd": entry.cwd,
                "turns": entry.turns,
                "created": entry.created,
                "last_turn": entry.last_turn,
            }));
        }
        Err(Error::UnknownSession(args.session_id))
    }
}

/// Return a turn's reply: the last turn by default, or turn `n` (1-based).
#[derive(Debug, Deserialize, JsonSchema)]
struct ResultArgs {
    /// Session id, as returned by `spawn`/`adopt`.
    session_id: String,
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
        std::future::ready(self.result(&args).map_err(Into::into))
    }
}

impl ResultTool {
    /// Synchronous body: `result` only reads shared state.
    fn result(&self, args: &ResultArgs) -> Result<Value> {
        let sub = self.0.get(&args.session_id)?;
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
                        session_id: args.session_id.clone(),
                        turn: args.turn.unwrap_or(0),
                        have: inner.turn_offset + inner.turns.len() as u64,
                    });
                }
            }
        };
        let mut view = json!({
            "session_id": args.session_id,
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
/// `cancelled`, drops prompts parked by `send` with `policy: "queued"`, then
/// sends `session/cancel`. The turn ends with the agent's `cancelled` stop
/// reason.
#[derive(Debug, Deserialize, JsonSchema)]
struct CancelArgs {
    /// Session id, as returned by `spawn`/`adopt`.
    session_id: String,
}

struct CancelTool(Arc<AppState>);

impl Tool for CancelTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("cancel")
    }
    type Arguments = CancelArgs;
    type Res = Value;

    async fn call(&self, args: CancelArgs) -> aither_core::Result<Value> {
        let sub = self.0.get(&args.session_id)?;
        let (dropped, turn_n) = {
            let mut inner = sub.rt.inner.lock().expect("inner poisoned");
            if !matches!(inner.status, Status::Running | Status::NeedsPermission) {
                return Err(Error::NotRunning(args.session_id.clone()).into());
            }
            for pending in inner.pending.drain(..) {
                let _ = pending.answer.send(RequestPermissionOutcome::Cancelled);
            }
            (
                std::mem::take(&mut inner.queue),
                inner.current.as_ref().map(|turn| turn.n),
            )
        };
        record_queue_dropped(&sub.rt, turn_n, dropped).await;
        sub.rt.notify.notify_waiters();
        sub.client
            .cancel(&sub.session_id)
            .await
            .map_err(|source| Error::Agent {
                agent: args.session_id.clone(),
                source,
            })?;
        Ok(json!({"session_id": args.session_id, "cancelled": true}))
    }
}

/// Answer a permission request queued by the `ask` policy.
///
/// `request_id` and `option_id` come from `wait`'s `pending_permission` field.
#[derive(Debug, Deserialize, JsonSchema)]
#[allow(clippy::struct_field_names)]
struct PermitArgs {
    /// Session id, as returned by `spawn`/`adopt`.
    session_id: String,
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
        let sub = self.0.get(&args.session_id)?;
        let pending = {
            let mut inner = sub.rt.inner.lock().expect("inner poisoned");
            let Some(position) = inner
                .pending
                .iter()
                .position(|p| p.request_id == args.request_id)
            else {
                return Err(Error::NoSuchPermission {
                    session_id: args.session_id.clone(),
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
            "session_id": args.session_id,
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
    /// Session id, as returned by `spawn`/`adopt`.
    session_id: String,
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
        let path = match self
            .0
            .live
            .lock()
            .expect("live poisoned")
            .get(&args.session_id)
        {
            Some(sub) => sub.transcript_path.clone(),
            None => self
                .0
                .config
                .defaults
                .transcript_dir
                .join(format!("{}.jsonl", args.session_id)),
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
fn live_view(sub: &Subagent) -> Value {
    let (state, turns) = {
        let inner = sub.rt.inner.lock().expect("inner poisoned");
        (
            inner.status.name(),
            inner.turn_offset + inner.turns.len() as u64,
        )
    };
    json!({
        "session_id": sub.session_id,
        "agent": sub.agent,
        "live": true,
        "state": state,
        "cwd": sub.cwd,
        "turns": turns,
    })
}

/// `list` view of a registered but closed session.
fn closed_view(session_id: &str, entry: &RegistryEntry) -> Value {
    json!({
        "session_id": session_id,
        "agent": entry.agent,
        "live": false,
        "state": "closed",
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
                    .map(|(session_id, sub)| (session_id.clone(), live_view(sub)))
                    .collect::<BTreeMap<_, _>>(),
                registry
                    .iter()
                    .map(|(session_id, entry)| (session_id.clone(), entry.clone()))
                    .collect::<BTreeMap<_, _>>(),
            )
        };
        let mut by_session: BTreeMap<String, Value> = registered
            .iter()
            .map(|(session_id, entry)| (session_id.clone(), closed_view(session_id, entry)))
            .collect();
        by_session.extend(live_views);
        json!({"subagents": by_session.into_values().collect::<Vec<_>>()})
    }
}

/// Close a subagent: end the agent process. The registry entry (and so the
/// session's resumability via `adopt`) is kept; use `forget` to remove it.
#[derive(Debug, Deserialize, JsonSchema)]
struct CloseArgs {
    /// Session id, as returned by `spawn`/`adopt`.
    session_id: String,
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
                .remove(&args.session_id)
        };
        match sub {
            Some(sub) => {
                close_sub(&sub).await;
                Ok(json!({"session_id": args.session_id, "closed": true}))
            }
            None if self
                .0
                .registry
                .lock()
                .expect("registry poisoned")
                .get(&args.session_id)
                .is_some() =>
            {
                Ok(json!({"session_id": args.session_id, "closed": true, "was_live": false}))
            }
            None => Err(Error::UnknownSession(args.session_id).into()),
        }
    }
}

/// Forget a subagent: close it if live and remove its registry entry. The
/// transcript file is kept.
#[derive(Debug, Deserialize, JsonSchema)]
struct ForgetArgs {
    /// Session id, as returned by `spawn`/`adopt`.
    session_id: String,
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
                .remove(&args.session_id)
        };
        if let Some(sub) = &sub {
            close_sub(sub).await;
        }
        let mut registered = false;
        self.0
            .update_registry(|registry| {
                registered = registry.remove(&args.session_id);
            })
            .await?;
        if sub.is_none() && !registered {
            return Err(Error::UnknownSession(args.session_id).into());
        }
        Ok(json!({"session_id": args.session_id, "forgotten": true}))
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
        // Per-agent: ids of its live subagents and the info the first such
        // process reported (agent_info, modes, config options).
        let live: Vec<Arc<Subagent>> = self
            .0
            .live
            .lock()
            .expect("live poisoned")
            .values()
            .cloned()
            .collect();
        let implicit_default = self
            .0
            .config
            .resolve_agent(None)
            .map(|(name, _)| name.clone())
            .ok();
        let mut out = Vec::new();
        for (name, agent) in &self.0.config.agents {
            let subs: Vec<&str> = live
                .iter()
                .filter(|sub| sub.agent == *name)
                .map(|sub| sub.session_id.as_str())
                .collect();
            let mut view = json!({
                "name": name,
                "command": agent.command,
                "args": agent.args,
                "mode": agent.mode,
                "allow_outside_cwd": agent.allow_outside_cwd,
                "default": implicit_default.as_ref() == Some(name),
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
