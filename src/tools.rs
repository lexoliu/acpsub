//! The MCP tools: one `Tool` impl per subagent operation.
//!
//! Each tool's `Arguments` rustdoc is the description the orchestrating model
//! reads, so they are written as instructions, not labels.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aither_acp::{
    ConfigOption, ConfigOptionValue, ConfigSelectOptions, RequestPermissionOutcome, StopReason,
    ToolKind,
};
use aither_core::llm::tool::{Progress, Tool, ToolContext, Tools};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::{ConfigValue, PermissionPolicy};
use crate::error::{Error, Result};
use crate::registry::RegistryEntry;
use crate::state::{
    AppState, Gate, Launch, Status, Subagent, ToolCallSummary, Turn, begin_turn, close_sub, launch,
    next_turn_number, prompt_slot_free, record_queue_dropped, spawn_turn_task, start_turn,
    steer_resolved, steer_turn, steer_waiter,
};
use crate::transcript::{RenderOptions, render};

/// `wait_any` polls the named subagents this often.
const WAIT_ANY_POLL: Duration = Duration::from_millis(50);
/// How often `wait`/`wait_any` report progress to the caller while
/// blocked.
///
/// MCP hosts abandon a `tools/call` that produces neither a response nor
/// progress — Claude Code aborts one after 1800 s of silence — so the
/// interval sits far below that limit: frequent enough that the request is
/// never anywhere near idle, sparse enough that the notifications stay
/// noise.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(10);
/// How long `send` waits for a synchronous rejection of a steer before
/// reporting `steered`; an accepted steer resolves only at turn end, so a
/// longer wait would just stall the caller.
const STEER_ACK: Duration = Duration::from_millis(250);
/// Largest `expect_secs` a `wait`/`wait_any` accepts when the call has no
/// progress channel — [`ToolContext::is_listening`] is false.
///
/// The MCP progress specification ties `notifications/progress` to the
/// request's `progressToken`, so a call without one has no keep-alive: it
/// produces nothing on the wire until it returns, and a host abandons a
/// `tools/call` idle for 1800 s (Claude Code's limit). 600 s stays far
/// below that — the call returns long before even a stricter host would
/// give up — while an expectation beyond it could never come back and is
/// rejected up front.
/// <https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/progress>
const NO_LISTENER_EXPECT_LIMIT: u64 = 600;

/// Build the full tool set over `state`.
///
/// # Errors
///
/// Returns an error if a tool fails to register (duplicate name or empty
/// description — both would be a bug in this crate).
pub fn build_tools(state: Arc<AppState>) -> aither_core::Result<Tools> {
    build_tools_with_progress_interval(state, PROGRESS_INTERVAL)
}

/// [`build_tools`] with an explicit report interval: the seam tests
/// inject a short interval through — production callers use
/// [`build_tools`], which reports every `PROGRESS_INTERVAL`.
///
/// # Errors
///
/// Returns an error if a tool fails to register (duplicate name or empty
/// description — both would be a bug in this crate).
pub fn build_tools_with_progress_interval(
    state: Arc<AppState>,
    progress_interval: Duration,
) -> aither_core::Result<Tools> {
    let mut tools = Tools::new();
    tools.register(SpawnTool(state.clone()))?;
    tools.register(AdoptTool(state.clone()))?;
    tools.register(SendTool(state.clone()))?;
    tools.register(WaitTool {
        state: state.clone(),
        interval: progress_interval,
    })?;
    tools.register(WaitAnyTool {
        state: state.clone(),
        interval: progress_interval,
    })?;
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

/// The daemon-only tools, registered by the socket server on top of the
/// shared set: `daemon/drain` and `daemon/resumed` drive the graceful
/// restart `acpsub restart` performs.
///
/// # Errors
///
/// Returns an error if a tool fails to register.
pub fn register_daemon_tools(tools: &mut Tools, state: Arc<AppState>) -> aither_core::Result<()> {
    tools.register(DrainTool(state.clone()))?;
    tools.register(ResumedTool(state))?;
    Ok(())
}

/// The `spawn`/`adopt` tool's state summary.
fn spawned_view(sub: &Subagent) -> Value {
    let inner = sub.rt.inner.lock().expect("inner poisoned");
    let (state, parked) = match &inner.status {
        Status::RateLimited { resume_at, reason } => (
            inner.status.name(),
            Some((resume_at.to_string(), reason.clone())),
        ),
        _ => (inner.status.name(), None),
    };
    drop(inner);
    let mut view = json!({
        "session_id": sub.session_id,
        "agent": sub.agent,
        "state": state,
        "model": sub.model,
        "mode": sub.mode,
        "cwd": sub.cwd,
        "transcript": sub.transcript_path,
    });
    if let Some((resume_at, reason)) = parked {
        view["resume_at"] = json!(resume_at);
        view["reason"] = json!(reason);
    }
    view
}

/// The turn `wait`/`wait_any` binds to at call time: the number of the
/// turn in flight, or — during the chained-turn gap, when the queue pop
/// already reserved the slot but `begin_turn` has not stamped `current`
/// yet — of the turn that is about to start. `None` when the subagent is
/// not `running`: the call returns at once, unchanged. A queued turn that
/// starts afterwards does not extend the wait.
///
/// # Errors
///
/// Returns [`Error::Internal`] when the status is `running` but no turn
/// start is recorded — a broken invariant, never a case to wait through
/// unbounded.
fn awaited_turn(sub: &Subagent) -> Result<Option<u64>> {
    let inner = sub.rt.inner.lock().expect("inner poisoned");
    if !matches!(inner.status, Status::Running) {
        return Ok(None);
    }
    if let Some(turn) = &inner.current {
        return Ok(Some(turn.n));
    }
    if inner.pending_turn_start.is_some() {
        return Ok(Some(next_turn_number(&inner)));
    }
    drop(inner);
    Err(Error::Internal {
        session_id: sub.session_id.clone(),
        detail: "status is running but no turn start is recorded".to_string(),
    })
}

/// What the turn a wait is bound to is doing.
enum Awaited {
    /// Still in flight: its elapsed seconds, measured from its own
    /// recorded start.
    Running(f64),
    /// The awaited turn ended `rate_limited`: `active` is its own elapsed
    /// seconds before the park — the only part that counts against
    /// `expect_secs` — and `next` is the turn the resume's continuation
    /// takes, which the wait rebinds to.
    Parked {
        /// The awaited turn's elapsed seconds before it parked.
        active: f64,
        /// The turn number the resume's continuation takes.
        next: u64,
    },
    /// The awaited turn has not materialized and the session is parked
    /// rate-limited: the wait freezes — it must not end, and the pause
    /// does not count against `expect_secs`.
    Paused,
    /// No longer the in-flight turn — settled into `turns`, superseded by
    /// a chained or materialized turn, parked on a permission request
    /// (the wait returns to report it), or gone with the session.
    Ended,
}

/// Poll the turn `n` a wait is bound to: its elapsed seconds while it is
/// still in flight, [`Awaited::Ended`] once it is not. Tracking the turn
/// number — not "whatever is running now" — is what keeps a queued turn
/// chaining after the awaited one from extending the wait or resetting
/// its progress.
///
/// # Errors
///
/// Returns [`Error::Internal`] when the awaited turn is in flight but has
/// no recorded start.
fn awaited_elapsed(sub: &Subagent, n: u64) -> Result<Awaited> {
    let inner = sub.rt.inner.lock().expect("inner poisoned");
    let missing_start = || Error::Internal {
        session_id: sub.session_id.clone(),
        detail: "status is running but no turn start is recorded".to_string(),
    };
    if let Some(turn) = inner.current.as_ref().filter(|turn| turn.n == n) {
        // Only `running` means still in flight: `needs_permission` parks
        // the turn on a request the wait returns to report, and `close`
        // cancels the session without clearing `current`.
        if !matches!(inner.status, Status::Running) {
            return Ok(Awaited::Ended);
        }
        let Some(started) = turn.started_at.or(inner.pending_turn_start) else {
            return Err(missing_start());
        };
        return Ok(Awaited::Running(
            jiff::Timestamp::now().duration_since(started).as_secs_f64(),
        ));
    }
    // The awaited turn settled: `rate_limited` means the wait follows
    // the resume's turn — the park itself does not count against
    // `expect_secs`. A settled turn that already chained into `current`
    // was caught by the arm above.
    if let Some(turn) = inner.turns.iter().find(|turn| turn.n == n)
        && turn.rate_limited
    {
        let active = turn
            .started_at
            .zip(turn.ended_at)
            .map_or(0.0, |(started, ended)| {
                ended.duration_since(started).as_secs_f64()
            });
        return Ok(Awaited::Parked {
            active,
            next: next_turn_number(&inner),
        });
    }
    // The gap the awaited turn was predicted in: the slot is still
    // reserved (`Running`, no `current` yet) until `begin_turn` runs.
    if matches!(inner.status, Status::Running)
        && inner.current.is_none()
        && next_turn_number(&inner) == n
    {
        let Some(started) = inner.pending_turn_start else {
            return Err(missing_start());
        };
        return Ok(Awaited::Running(
            jiff::Timestamp::now().duration_since(started).as_secs_f64(),
        ));
    }
    // The awaited turn is a resume the session is still parked for.
    if matches!(inner.status, Status::RateLimited { .. }) {
        return Ok(Awaited::Paused);
    }
    drop(inner);
    Ok(Awaited::Ended)
}

/// The progress report a blocked wait sends for `sub`: `progress` is the
/// turn's elapsed seconds against `total = expect_secs`, and the message
/// names the session and its latest tool call title.
fn wait_progress(sub: &Subagent, elapsed: f64, expect_secs: u64) -> Progress {
    let title = {
        let inner = sub.rt.inner.lock().expect("inner poisoned");
        inner.current.as_ref().and_then(|turn| {
            turn.latest_tool_call
                .as_ref()
                .and_then(|id| turn.tool_calls.get(id))
                .map(|call| call.title.clone())
        })
    };
    let message = title.map_or_else(
        || sub.session_id.clone(),
        |title| format!("{}: {title}", sub.session_id),
    );
    #[expect(clippy::cast_precision_loss, reason = "sub-second precision is enough")]
    Progress::new(elapsed)
        .with_total(expect_secs as f64)
        .with_message(message)
}

/// Reject a blocked wait that could never return: without a progress
/// listener the call stays silent until it returns, and a host abandons a
/// silent `tools/call` long before an expectation beyond
/// [`NO_LISTENER_EXPECT_LIMIT`] plays out.
///
/// # Errors
///
/// Returns [`Error::ExpectExceedsNoToken`] when `!cx.is_listening()` and
/// `expect_secs` exceeds the ceiling.
const fn check_expect(cx: &ToolContext, expect_secs: u64) -> Result<()> {
    if !cx.is_listening() && expect_secs > NO_LISTENER_EXPECT_LIMIT {
        return Err(Error::ExpectExceedsNoToken {
            expect_secs,
            ceiling_secs: NO_LISTENER_EXPECT_LIMIT,
        });
    }
    Ok(())
}

/// How a `wait`/`wait_any` ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaitOutcome {
    /// The awaited turn left `running` — ended, parked on a permission,
    /// or gone with the session — and the result reports that state.
    Ended,
    /// The awaited turn outlived `expect_secs`: the result reports
    /// `overrun`, the turn's own elapsed time, its latest tool call, and
    /// the activity digest.
    Overrun,
    /// The call's own `max_wait_secs` bound passed first: the result
    /// reports `running` with the activity digest.
    MaxWait,
}

/// The activity digest a `wait`/`wait_any` result carries on `overrun` and
/// on a `max_wait_secs` timeout, measured over the window from when the
/// wait call began to now. Every value comes from the tool calls'
/// recorded `started_at`/`ended_at`; nothing is classified from command
/// text.
///
/// A call counts toward the window when any of its recorded lifetime
/// overlaps it — a call that ended before the wait began is excluded, and
/// a still-running call's duration is measured to now.
#[derive(Debug, Serialize)]
struct ActivityDigest {
    /// The window's length in seconds.
    window_secs: f64,
    /// Calls whose recorded lifetime overlaps the window.
    tool_calls: u64,
    /// Fraction of window wall time spent inside calls — above 1.0 when
    /// calls overlap.
    tool_call_share: f64,
    /// The five longest calls by in-window duration.
    longest_tool_calls: Vec<DigestCall>,
    /// Every title seen three or more times, with its count.
    repeated_titles: Vec<DigestTitle>,
    /// Seconds since the most recent edit-kind call (`edit`/`delete`/`move`)
    /// finished — about 0 while one is still running; `null` when none
    /// overlaps the window.
    secs_since_last_edit: Option<f64>,
    /// The turn's latest tool call summary.
    latest_tool_call: Option<ToolCallSummary>,
}

/// A `longest_tool_calls` entry.
#[derive(Debug, Serialize)]
struct DigestCall {
    /// Tool call id.
    id: String,
    /// Human-readable title.
    title: String,
    /// The call's in-window duration in seconds.
    duration_secs: f64,
}

/// A `repeated_titles` entry.
#[derive(Debug, Serialize)]
struct DigestTitle {
    /// The repeated title.
    title: String,
    /// How many in-window calls carry it.
    count: u64,
}

/// Build the [`ActivityDigest`] of `turn` over the window from
/// `window_start` to now.
fn activity_digest(turn: Option<&Turn>, window_start: jiff::Timestamp) -> ActivityDigest {
    let now = jiff::Timestamp::now();
    let window_secs = now.duration_since(window_start).as_secs_f64();
    let mut tool_calls = 0_u64;
    let mut busy_secs = 0.0_f64;
    let mut longest: Vec<(&ToolCallSummary, f64)> = Vec::new();
    let mut titles: BTreeMap<&str, u64> = BTreeMap::new();
    let mut last_edit_end: Option<jiff::Timestamp> = None;
    if let Some(turn) = turn {
        for call in turn.tool_calls.values() {
            let call_end = call.ended_at.unwrap_or(now);
            if call_end <= window_start {
                continue;
            }
            tool_calls += 1;
            let duration = call_end
                .duration_since(call.started_at.max(window_start))
                .as_secs_f64();
            busy_secs += duration;
            longest.push((call, duration));
            *titles.entry(call.title.as_str()).or_default() += 1;
            if matches!(
                call.kind,
                Some(ToolKind::Edit | ToolKind::Delete | ToolKind::Move)
            ) {
                last_edit_end = Some(last_edit_end.map_or(call_end, |prev| prev.max(call_end)));
            }
        }
    }
    longest.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut repeated: Vec<(&str, u64)> = titles
        .into_iter()
        .filter(|(_, count)| *count >= 3)
        .collect();
    repeated.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ActivityDigest {
        window_secs,
        tool_calls,
        tool_call_share: if window_secs > 0.0 {
            busy_secs / window_secs
        } else {
            0.0
        },
        longest_tool_calls: longest
            .iter()
            .take(5)
            .map(|(call, duration)| DigestCall {
                id: call.id.clone(),
                title: call.title.clone(),
                duration_secs: *duration,
            })
            .collect(),
        repeated_titles: repeated
            .iter()
            .map(|(title, count)| DigestTitle {
                title: (*title).to_string(),
                count: *count,
            })
            .collect(),
        secs_since_last_edit: last_edit_end.map(|end| now.duration_since(end).as_secs_f64()),
        latest_tool_call: turn.and_then(|turn| {
            turn.latest_tool_call
                .as_ref()
                .and_then(|id| turn.tool_calls.get(id))
                .cloned()
        }),
    }
}

/// The `pending_permission` field of a wait result, for the first queued
/// permission request.
fn pending_view(perm: &crate::state::PendingPermission) -> Value {
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
}

/// The `send`/`spawn`/`adopt` result for a prompt parked by a rate limit:
/// where it parks and for how long — the continuation the scheduler sends
/// at `resume_at` runs ahead of the queued prompts.
fn parked_view(session_id: &str, parked: &crate::ratelimit::Parked) -> Value {
    json!({
        "session_id": session_id,
        "state": "rate_limited",
        "resume_at": parked.resume_at.to_string(),
        "reason": parked.reason,
        "position": parked.position,
    })
}

/// A `wait`-style result for one subagent. `overrun` marks that the turn
/// outlived `expect_secs`: the result reports `overrun`, the turn's own
/// elapsed time, and its latest tool call. `awaited` is the turn the call
/// bound to (see [`awaited_turn`]); `None` reports the latest turn, the
/// shape a wait on a non-running subagent has always had. `window_start`
/// is when the wait call began — the base of the activity digest an
/// `Overrun`/`MaxWait` outcome carries; `parked_secs` is the active
/// seconds an earlier bound turn banked before a rate-limit park, folded
/// into the reported `elapsed_secs` on an overrun.
fn wait_view(
    sub: &Subagent,
    started: Instant,
    outcome: WaitOutcome,
    awaited: Option<u64>,
    window_start: jiff::Timestamp,
    parked_secs: f64,
) -> Value {
    let (status, reply, tool_calls, pending, turn_n, queued, elapsed, latest, ended_reason, digest) = {
        let inner = sub.rt.inner.lock().expect("inner poisoned");
        // The awaited turn's own record — the same lookup `result` uses
        // for an explicit turn number: `turns` once settled, `current`
        // while it is still in flight. A queued turn that already chained
        // into `current` must not shadow it.
        let turn = awaited.map_or_else(
            || inner.current.as_ref().or_else(|| inner.turns.last()),
            |n| {
                inner
                    .turns
                    .iter()
                    .find(|turn| turn.n == n)
                    .or_else(|| inner.current.as_ref().filter(|turn| turn.n == n))
            },
        );
        let digest = match outcome {
            WaitOutcome::Ended => None,
            WaitOutcome::Overrun | WaitOutcome::MaxWait => {
                Some(activity_digest(turn, window_start))
            }
        };
        let pending = inner.pending.first().map(pending_view);
        (
            inner.status.clone(),
            turn.map_or_else(String::new, |turn| turn.reply.clone()),
            turn.map_or_else(Vec::new, |turn| {
                turn.tool_calls.values().cloned().collect::<Vec<_>>()
            }),
            pending,
            turn.map(|turn| turn.n).or(awaited),
            inner.queue.len(),
            turn.and_then(|turn| turn.started_at).map(|started| {
                parked_secs + jiff::Timestamp::now().duration_since(started).as_secs_f64()
            }),
            turn.and_then(|turn| {
                turn.latest_tool_call
                    .as_ref()
                    .and_then(|id| turn.tool_calls.get(id))
                    .cloned()
            }),
            turn.and_then(|turn| turn.stop_reason)
                .filter(|_| awaited.is_some()),
            digest,
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
    if outcome == WaitOutcome::Overrun {
        view["state"] = json!("overrun");
        if let Some(elapsed) = elapsed {
            view["elapsed_secs"] = json!(elapsed);
        }
        if let Some(latest) = latest {
            view["latest_tool_call"] = serde_json::to_value(latest)
                .expect("ToolCallSummary holds only owned primitives; serialization cannot fail");
        }
    } else if let Some(reason) = ended_reason {
        // The awaited turn's own end: its stop reason wins over the
        // session status, which may already show the turn that chained
        // after it.
        view["state"] = json!(if matches!(reason, StopReason::Cancelled) {
            "cancelled"
        } else {
            "done"
        });
        view["stop_reason"] = serde_json::to_value(reason).unwrap_or_default();
    } else {
        match &status {
            Status::Done(reason) => {
                view["stop_reason"] = serde_json::to_value(reason).unwrap_or_default();
            }
            Status::Cancelled => {
                view["stop_reason"] = json!("cancelled");
            }
            Status::Failed(reason) | Status::Exited(reason) => {
                view["error"] = json!(reason);
            }
            _ => {}
        }
    }
    // A parked session reports where the wait stands regardless of how
    // the wait ended: `rate_limited`, the reset time, and the provider's
    // own reason.
    if let Status::RateLimited { resume_at, reason } = &status {
        view["resume_at"] = json!(resume_at.to_string());
        view["reason"] = json!(reason);
    }
    if let Some(pending) = pending {
        view["pending_permission"] = pending;
    }
    if let Some(digest) = digest {
        view["digest"] = serde_json::to_value(digest)
            .expect("ActivityDigest holds only owned primitives; serialization cannot fail");
    }
    view
}

// ---------------------------------------------------------------------------
// spawn / adopt
// ---------------------------------------------------------------------------

/// Spawn a subagent: start the configured agent process, open a new ACP
/// session in `cwd`, and send `prompt` as its first turn.
///
/// Returns immediately with `session_id` — the handle every other tool
/// addresses (`wait` for the reply, `send` for follow-ups, `transcript` for
/// the full log), plus the `model` and `mode` the agent accepted (`null`
/// for an agent that advertises neither). The turn runs in the background.
/// To take over an existing session instead of starting a new one, use
/// `adopt`.
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
    /// The model to run, set via `session/set_config_option` on the
    /// `model` option. Required when the agent advertises a `model`
    /// config option; must be omitted when it advertises none — passing
    /// one then is an error. When given, the value must be one the agent
    /// advertises; the value it reports as current comes back in the
    /// result.
    model: Option<String>,
    /// The session mode to activate (`session/set_mode`). Required when
    /// the agent advertises session modes; must be omitted when it
    /// advertises none — passing one then is an error and no `set_mode`
    /// call is made. When given, the value must be one of the modes the
    /// agent advertises; the mode it reports as current comes back in the
    /// result.
    mode: Option<String>,
    /// Extra session config options to set (`session/set_config_option`),
    /// applied after `model`: option id → string or boolean. Every id and
    /// value must be one the agent advertises.
    config: Option<BTreeMap<String, ConfigValue>>,
    /// Permission policy for this subagent: `allow` auto-approves, `deny`
    /// auto-rejects, `ask` queues requests for the `permit` tool.
    permission: Option<PermissionPolicy>,
    /// Pid of the coordinator process this subagent belongs to. When the
    /// server runs as a daemon it reaps the subagent once that pid dies,
    /// so a session cannot outlive its coordinator as an orphan. Omit to
    /// leave the subagent unowned.
    owner: Option<u32>,
}

struct SpawnTool(Arc<AppState>);

impl Tool for SpawnTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("spawn")
    }
    type Arguments = SpawnArgs;
    type Res = Value;

    async fn call(&self, args: SpawnArgs, _cx: ToolContext) -> aither_core::Result<Value> {
        if self.0.draining() {
            return Err(restarting().into());
        }
        let config = self.0.load_config()?;
        let (agent, _) = config.resolve_agent(args.agent.as_deref())?;
        let sub = launch(
            &self.0,
            &config,
            Launch {
                agent: agent.clone(),
                cwd: args.cwd,
                prompt: Some(args.prompt),
                mode: args.mode,
                model: args.model,
                config: args.config.unwrap_or_default(),
                permission: args.permission,
                load: None,
                owner: args.owner,
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
/// A live session whose agent process exited (`exited` state) is reaped by
/// `adopt` — no `close` needed first.
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
    /// The model to run, as in `spawn`: required when the agent advertises
    /// a `model` config option, must be omitted when it advertises none.
    model: Option<String>,
    /// The session mode to activate, as in `spawn`: required when the agent
    /// advertises session modes, must be omitted when it advertises none.
    mode: Option<String>,
    /// Permission policy override, as in `spawn`.
    permission: Option<PermissionPolicy>,
    /// Owning coordinator pid, as in `spawn`.
    owner: Option<u32>,
}

struct AdoptTool(Arc<AppState>);

impl Tool for AdoptTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("adopt")
    }
    type Arguments = AdoptArgs;
    type Res = Value;

    async fn call(&self, args: AdoptArgs, _cx: ToolContext) -> aither_core::Result<Value> {
        if self.0.draining() {
            return Err(restarting().into());
        }
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
        let config = self.0.load_config()?;
        let (agent, agent_cfg) = config.resolve_agent(requested)?;
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
            &config,
            Launch {
                agent: agent.clone(),
                cwd,
                prompt: args.prompt,
                mode: args.mode,
                model: args.model,
                config: BTreeMap::new(),
                permission: args.permission,
                load: Some(args.session_id),
                owner: args.owner,
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
/// unless you ask. A `failed` subagent — the last turn errored but the
/// agent is still live — takes the prompt as a fresh turn, like `done`;
/// an `exited` one (the agent process is gone) errors — `adopt` it instead.
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
    /// `done`, `cancelled`, or `failed`. `queued` — park the prompt on a
    /// FIFO queue; it fires as the next turn when the current one ends, and
    /// is dropped if that turn is cancelled or fails, or on
    /// `cancel`/`close`/`forget`.
    /// `steer` — inject the prompt into the running turn (a second
    /// `session/prompt` while it is in flight); agents that support
    /// mid-turn injection fold it into the active task, agents that do not
    /// surface an error. How the request resolves depends on the agent's
    /// configured `steer` semantics: an `answered` agent (the default)
    /// answers every steered prompt, while a `folded` agent (codex-acp)
    /// never answers one it folded into a turn — acpsub settles such a
    /// steer when its target turn ends. On either, a steer that landed
    /// behind the turn's end still runs and resolves as a turn of its own.
    policy: SendPolicy,
}

/// Delivery policy for `send` while a turn is running — no default, the
/// caller chooses explicitly.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum SendPolicy {
    /// Error when the subagent is not `idle`, `done`, `cancelled`, or
    /// `failed`.
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

    async fn call(&self, args: SendArgs, _cx: ToolContext) -> aither_core::Result<Value> {
        // A draining daemon refuses sends that would start a new turn;
        // a steer into a turn that is still running still goes through —
        // it does not extend the drain.
        if self.0.draining() && !matches!(args.policy, SendPolicy::Steer) {
            return Err(restarting().into());
        }
        let sub = self.0.get(&args.session_id)?;
        // A quota gate for this agent's scope parks the prompt instead of
        // letting it burn a request that is bound to fail.
        if let Some(parked) = crate::ratelimit::park_prompt(&self.0, &sub, &args.prompt).await {
            return Ok(parked_view(&args.session_id, &parked));
        }
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
                        } else if matches!(inner.status, Status::Exited(_)) {
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
            SendPolicy::Steer => return steer_send(&self.0, &sub, &args).await,
        }
        Ok(json!({"session_id": args.session_id, "state": "running"}))
    }
}

/// `send` with `policy: "steer"`: inject the prompt into the running
/// turn as a steering message — a second `session/prompt` while it is in
/// flight. On a promptable session it just runs as the next turn; a
/// parked or dead one is [`Error::NotPromptable`] (`park_prompt` in the
/// caller has already parked prompts while a limit holds, so the
/// `rate_limited` arm is unreachable but kept for completeness).
async fn steer_send(
    state: &Arc<AppState>,
    sub: &Arc<Subagent>,
    args: &SendArgs,
) -> aither_core::Result<Value> {
    let wire = sub.rt.prompt_send.lock().await;
    let status = {
        let inner = sub.rt.inner.lock().expect("inner poisoned");
        inner.status.clone()
    };
    match status {
        Status::Idle | Status::Done(_) | Status::Cancelled | Status::Failed(_) => {
            let (turn_n, fut) =
                begin_turn(state, sub, args.prompt.clone(), Gate::Check, false).await?;
            spawn_turn_task(state.clone(), sub, fut, turn_n);
            drop(wire);
            Ok(json!({"session_id": args.session_id, "state": "running"}))
        }
        Status::Running | Status::NeedsPermission => {
            let (steer_id, turn_n, mut fut) = steer_turn(sub, args.prompt.clone()).await?;
            drop(wire);
            // An accepted steer resolves only at turn end; a rejection
            // lands almost at once. Give the agent a beat to reject
            // before reporting `steered`.
            if let Ok(outcome) = tokio::time::timeout(STEER_ACK, &mut fut).await {
                if let Err(source) =
                    steer_resolved(state.clone(), sub, steer_id, turn_n, outcome).await
                {
                    return Err(Error::Agent {
                        agent: sub.agent.clone(),
                        source,
                    }
                    .into());
                }
            } else {
                tokio::spawn(steer_waiter(
                    state.clone(),
                    sub.clone(),
                    steer_id,
                    turn_n,
                    fut,
                ));
            }
            Ok(json!({
                "session_id": args.session_id,
                "state": "steered",
                "turn": turn_n,
            }))
        }
        Status::RateLimited { .. } | Status::Exited(_) => {
            drop(wire);
            Err(Error::NotPromptable {
                session_id: args.session_id.clone(),
                status: status.name().to_string(),
            }
            .into())
        }
    }
}

// ---------------------------------------------------------------------------
// wait / wait_any
// ---------------------------------------------------------------------------

/// Block until the turn that was running when the call began ends, a
/// permission request needs an answer, or the turn has run past
/// `expect_secs`. The wait binds to that turn's number: a queued or
/// steered turn that starts afterwards does not extend it.
///
/// Returns the subagent state (`done`/`cancelled`/`failed`/`exited`/
/// `needs_permission`), the turn's `reply` (concatenated agent message
/// text), its tool calls, and — when `needs_permission` — the
/// `pending_permission` request to answer with `permit`. When the turn has
/// run longer than `expect_secs` the result is `overrun`, carrying the
/// turn's elapsed time and its latest tool call: an overrun is to be
/// investigated (`status`, `transcript`, a `send` steer), not re-waited
/// with a larger expectation.
///
/// Both an `overrun` result and a `max_wait_secs` timeout carry an
/// activity `digest` measured over the window since the wait began —
/// its fields are documented on [`ActivityDigest`].
#[derive(Debug, Deserialize, JsonSchema)]
struct WaitArgs {
    /// Session id, as returned by `spawn`/`adopt`.
    session_id: String,
    /// Required: how long the awaited turn is expected to take, in seconds.
    /// Measured from the turn's recorded start — re-issuing a wait never
    /// extends it. The wait returns early on any state change, so an
    /// accurate expectation is free.
    expect_secs: u64,
    /// Optional: the wait call's own deadline in seconds, measured from
    /// when the wait began — unlike `expect_secs` it bounds the wait, not
    /// the turn. When it passes before the awaited turn ends or overruns,
    /// the wait returns `state: "running"` with an activity `digest` of
    /// the window (fields documented on `ActivityDigest`), rather than
    /// dying silent under a caller-side kill limit — set it just below
    /// the caller's own timeout.
    max_wait_secs: Option<u64>,
}

struct WaitTool {
    state: Arc<AppState>,
    interval: Duration,
}

impl Tool for WaitTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("wait")
    }
    type Arguments = WaitArgs;
    type Res = Value;

    async fn call(&self, args: WaitArgs, mut cx: ToolContext) -> aither_core::Result<Value> {
        check_expect(&cx, args.expect_secs)?;
        let sub = self.state.get(&args.session_id)?;
        let started = Instant::now();
        let window_start = jiff::Timestamp::now();
        let Some(mut awaited) = awaited_turn(&sub)? else {
            return Ok(wait_view(
                &sub,
                started,
                WaitOutcome::Ended,
                None,
                window_start,
                0.0,
            ));
        };
        // The wait's own deadline: `Instant` can represent anything the
        // caller would wait through; an overflowing bound is no bound.
        let deadline = args
            .max_wait_secs
            .and_then(|secs| started.checked_add(Duration::from_secs(secs)));
        #[expect(clippy::cast_precision_loss, reason = "sub-second precision is enough")]
        let expect = args.expect_secs as f64;
        let interval = self.interval;
        let mut next_report = started;
        // Seconds the awaited turn ran before it parked rate-limited —
        // they count once against `expect_secs`; the pause never does.
        let mut parked_secs = 0.0;
        let outcome = loop {
            match awaited_elapsed(&sub, awaited)? {
                Awaited::Ended => break WaitOutcome::Ended,
                Awaited::Parked { active, next } => {
                    parked_secs += active;
                    awaited = next;
                }
                Awaited::Running(elapsed) => {
                    let elapsed = parked_secs + elapsed;
                    if elapsed >= expect {
                        break WaitOutcome::Overrun;
                    }
                    let now = Instant::now();
                    if deadline.is_some_and(|deadline| now >= deadline) {
                        break WaitOutcome::MaxWait;
                    }
                    if now >= next_report {
                        cx.report_progress(wait_progress(&sub, elapsed, args.expect_secs))
                            .await?;
                        next_report = now + interval;
                    }
                    let wake = (expect - elapsed)
                        .min(0.25)
                        .min(next_report.saturating_duration_since(now).as_secs_f64())
                        .min(deadline.map_or(f64::MAX, |deadline| {
                            deadline.saturating_duration_since(now).as_secs_f64()
                        }));
                    tokio::select! {
                        () = sub.rt.notify.notified() => {},
                        () = tokio::time::sleep(Duration::from_secs_f64(wake)) => {},
                    }
                }
                // The session is parked rate-limited and the awaited turn
                // is the resume's continuation: freeze — bound by the
                // wait's own deadline alone. No turn progress happens
                // while parked, so there is nothing new to report.
                Awaited::Paused => {
                    let now = Instant::now();
                    if deadline.is_some_and(|deadline| now >= deadline) {
                        break WaitOutcome::MaxWait;
                    }
                    let wake = 0.25_f64.min(deadline.map_or(f64::MAX, |deadline| {
                        deadline.saturating_duration_since(now).as_secs_f64()
                    }));
                    tokio::select! {
                        () = sub.rt.notify.notified() => {},
                        () = tokio::time::sleep(Duration::from_secs_f64(wake)) => {},
                    }
                }
            }
        };
        Ok(wait_view(
            &sub,
            started,
            outcome,
            Some(awaited),
            window_start,
            parked_secs,
        ))
    }
}

/// Block until the first of several subagents' awaited turn ends — each
/// session binds to the turn running when the call began — a permission
/// request needs an answer, or an `overrun` once a turn has run past
/// `expect_secs` — then return that one's wait-style result. An overrun
/// is to be investigated, not re-waited with a larger number.
///
/// On a `max_wait_secs` timeout the result is the longest-running watched
/// turn's `running` state with its activity digest — the same shape
/// `wait` reports, documented there.
#[derive(Debug, Deserialize, JsonSchema)]
struct WaitAnyArgs {
    /// Session ids to watch.
    session_ids: Vec<String>,
    /// Required: how long the awaited turns are expected to take, in
    /// seconds. Measured from each turn's recorded start — re-issuing a
    /// wait never extends it.
    expect_secs: u64,
    /// Optional: the wait call's own deadline in seconds, as in `wait` —
    /// on expiry the result is the longest-running watched turn's
    /// `running` state with its activity `digest`.
    max_wait_secs: Option<u64>,
}

struct WaitAnyTool {
    state: Arc<AppState>,
    interval: Duration,
}

impl Tool for WaitAnyTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("wait_any")
    }
    type Arguments = WaitAnyArgs;
    type Res = Value;

    async fn call(&self, args: WaitAnyArgs, mut cx: ToolContext) -> aither_core::Result<Value> {
        check_expect(&cx, args.expect_secs)?;
        let subs = args
            .session_ids
            .iter()
            .map(|session_id| self.state.get(session_id))
            .collect::<Result<Vec<_>>>()?;
        let started = Instant::now();
        let window_start = jiff::Timestamp::now();
        // Each watched session binds to the turn in flight at call time,
        // the same rule `wait` applies to one.
        let mut awaited = subs
            .iter()
            .map(|sub| awaited_turn(sub))
            .collect::<Result<Vec<_>>>()?;
        let deadline = args
            .max_wait_secs
            .and_then(|secs| started.checked_add(Duration::from_secs(secs)));
        #[expect(clippy::cast_precision_loss, reason = "sub-second precision is enough")]
        let expect = args.expect_secs as f64;
        let interval = self.interval;
        let mut next_report = started;
        // Per-session active seconds before a rate-limit park — the same
        // rebind `wait` applies to one turn.
        let mut parked_secs = vec![0.0; subs.len()];
        loop {
            let mut poll = WAIT_ANY_POLL;
            // Report the longest-running watched turn — by index, so a
            // rebind on park can update `awaited` in place.
            let mut longest: Option<(f64, usize)> = None;
            for index in 0..subs.len() {
                let sub = &subs[index];
                let Some(n) = awaited[index] else {
                    return Ok(wait_view(
                        sub,
                        started,
                        WaitOutcome::Ended,
                        None,
                        window_start,
                        0.0,
                    ));
                };
                match awaited_elapsed(sub, n)? {
                    Awaited::Ended => {
                        return Ok(wait_view(
                            sub,
                            started,
                            WaitOutcome::Ended,
                            Some(n),
                            window_start,
                            parked_secs[index],
                        ));
                    }
                    Awaited::Parked { active, next } => {
                        // The awaited turn ended rate-limited: its
                        // pre-park time counts once, and the wait follows
                        // the resume's turn.
                        parked_secs[index] += active;
                        awaited[index] = Some(next);
                    }
                    // Parked rate-limited: nothing running to compare.
                    Awaited::Paused => {}
                    Awaited::Running(elapsed) => {
                        let elapsed = parked_secs[index] + elapsed;
                        if elapsed >= expect {
                            return Ok(wait_view(
                                sub,
                                started,
                                WaitOutcome::Overrun,
                                Some(n),
                                window_start,
                                parked_secs[index],
                            ));
                        }
                        poll = poll.min(Duration::from_secs_f64(expect - elapsed));
                        if longest.is_none_or(|(max, _)| elapsed > max) {
                            longest = Some((elapsed, index));
                        }
                    }
                }
            }
            let now = Instant::now();
            if deadline.is_some_and(|deadline| now >= deadline)
                && let Some(index) = max_wait_index(longest, &awaited)
            {
                return Ok(wait_view(
                    &subs[index],
                    started,
                    WaitOutcome::MaxWait,
                    awaited[index],
                    window_start,
                    parked_secs[index],
                ));
            }
            if now >= next_report
                && let Some((elapsed, index)) = longest
            {
                cx.report_progress(wait_progress(&subs[index], elapsed, args.expect_secs))
                    .await?;
                next_report = now + interval;
            }
            tokio::time::sleep(poll.min(next_report.saturating_duration_since(now)).min(
                deadline.map_or(Duration::MAX, |deadline| {
                    deadline.saturating_duration_since(now)
                }),
            ))
            .await;
        }
    }
}

/// Which watched session a `max_wait_secs` expiry reports: the
/// longest-running turn, else the first still-bound one — a parked
/// session reports its `rate_limited` state either way.
fn max_wait_index(longest: Option<(f64, usize)>, awaited: &[Option<u64>]) -> Option<usize> {
    longest
        .map(|(_, index)| index)
        .or_else(|| awaited.iter().position(Option::is_some))
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

    fn call(
        &self,
        args: StatusArgs,
        _cx: ToolContext,
    ) -> impl Future<Output = aither_core::Result<Value>> + Send {
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
                "model": sub.model,
                "mode": sub.mode,
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
            if let Status::Failed(reason) | Status::Exited(reason) = &status {
                view["error"] = json!(reason);
            }
            if let Status::RateLimited { resume_at, reason } = &status {
                view["resume_at"] = json!(resume_at.to_string());
                view["reason"] = json!(reason);
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
            let mut view = json!({
                "session_id": args.session_id,
                "agent": entry.agent,
                "live": false,
                "state": "closed",
                "cwd": entry.cwd,
                "turns": entry.turns,
                "created": entry.created,
                "last_turn": entry.last_turn,
            });
            // A parked session's schedule outlives its process: report
            // `rate_limited` over `closed` so the resume reads clearly.
            if let Some(parked) = &entry.rate_limited {
                view["state"] = json!("rate_limited");
                view["resume_at"] = json!(parked.resume_at);
                view["reason"] = json!(parked.reason);
            }
            return Ok(view);
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

    fn call(
        &self,
        args: ResultArgs,
        _cx: ToolContext,
    ) -> impl Future<Output = aither_core::Result<Value>> + Send {
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

    async fn call(&self, args: CancelArgs, _cx: ToolContext) -> aither_core::Result<Value> {
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
            .cancel(sub.session_id.as_str())
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

    fn call(
        &self,
        args: PermitArgs,
        _cx: ToolContext,
    ) -> impl Future<Output = aither_core::Result<Value>> + Send {
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

    async fn call(&self, args: TranscriptArgs, _cx: ToolContext) -> aither_core::Result<String> {
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
    let (state, turns, parked) = {
        let inner = sub.rt.inner.lock().expect("inner poisoned");
        (
            inner.status.name(),
            inner.turn_offset + inner.turns.len() as u64,
            match &inner.status {
                Status::RateLimited { resume_at, reason } => {
                    Some((resume_at.to_string(), reason.clone()))
                }
                _ => None,
            },
        )
    };
    let mut view = json!({
        "session_id": sub.session_id,
        "agent": sub.agent,
        "live": true,
        "state": state,
        "cwd": sub.cwd,
        "turns": turns,
    });
    if let Some((resume_at, reason)) = parked {
        view["resume_at"] = json!(resume_at);
        view["reason"] = json!(reason);
    }
    view
}

/// `list` view of a registered but closed session — a parked one reports
/// `rate_limited` with its reset time; the schedule survives the process.
fn closed_view(session_id: &str, entry: &RegistryEntry) -> Value {
    let mut view = json!({
        "session_id": session_id,
        "agent": entry.agent,
        "live": false,
        "state": "closed",
        "cwd": entry.cwd,
        "turns": entry.turns,
    });
    if let Some(parked) = &entry.rate_limited {
        view["state"] = json!("rate_limited");
        view["resume_at"] = json!(parked.resume_at);
        view["reason"] = json!(parked.reason);
    }
    view
}

impl Tool for ListTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("list")
    }
    type Arguments = ListArgs;
    type Res = Value;

    fn call(
        &self,
        _args: ListArgs,
        _cx: ToolContext,
    ) -> impl Future<Output = aither_core::Result<Value>> + Send {
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

    async fn call(&self, args: CloseArgs, _cx: ToolContext) -> aither_core::Result<Value> {
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

    async fn call(&self, args: ForgetArgs, _cx: ToolContext) -> aither_core::Result<Value> {
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

    fn call(
        &self,
        _args: AgentsArgs,
        _cx: ToolContext,
    ) -> impl Future<Output = aither_core::Result<Value>> + Send {
        std::future::ready(self.agents().map_err(Into::into))
    }
}

impl AgentsTool {
    /// Synchronous body: `agents` re-reads the config file and reads shared
    /// state.
    fn agents(&self) -> Result<Value> {
        let config = self.0.load_config()?;
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
        let implicit_default = config
            .resolve_agent(None)
            .map(|(name, _)| name.clone())
            .ok();
        let mut out = Vec::new();
        for (name, agent) in &config.agents {
            let subs: Vec<&str> = live
                .iter()
                .filter(|sub| sub.agent == *name)
                .map(|sub| sub.session_id.as_str())
                .collect();
            let mut view = json!({
                "name": name,
                "command": agent.command,
                "args": agent.args,
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
        Ok(json!({"agents": out}))
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

// ---------------------------------------------------------------------------
// daemon/drain / daemon/resumed
// ---------------------------------------------------------------------------

/// The `restarting` refusal for work a draining daemon will not start.
fn restarting() -> Error {
    Error::Restarting {
        detail: "the daemon is draining for a restart; new work is refused".to_string(),
    }
}

/// Begin a graceful daemon restart: stop accepting new turns and
/// session-binds — every call that would start one is refused with a
/// `restarting` error — let in-flight turns and already-accepted queued
/// prompts finish, then close every live session marked for resume and
/// exit. The next daemon re-adopts them; `daemon/resumed` reports how the
/// resume went. Refused while a drain is already in progress.
#[derive(Debug, Deserialize, JsonSchema)]
struct DrainArgs {}

struct DrainTool(Arc<AppState>);

impl Tool for DrainTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("daemon/drain")
    }
    type Arguments = DrainArgs;
    type Res = Value;

    fn call(
        &self,
        _args: DrainArgs,
        _cx: ToolContext,
    ) -> impl Future<Output = aither_core::Result<Value>> + Send {
        std::future::ready(self.drain().map_err(Into::into))
    }
}

impl DrainTool {
    /// Synchronous body: the drain runs in its own task, so the call only
    /// flips the flag and spawns it.
    fn drain(&self) -> Result<Value> {
        if self
            .0
            .drain
            .requested
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            return Err(Error::Restarting {
                detail: "a drain is already in progress".to_string(),
            });
        }
        tokio::spawn(crate::daemon::drain_sessions(self.0.clone()));
        Ok(json!({"draining": true}))
    }
}

/// Report what this daemon's startup resume pass did: which marked
/// sessions came back live via `session/load` and which failed, with the
/// agent's error. Both lists are empty on a daemon that did not start
/// from a drain.
#[derive(Debug, Deserialize, JsonSchema)]
struct ResumedArgs {}

struct ResumedTool(Arc<AppState>);

impl Tool for ResumedTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("daemon/resumed")
    }
    type Arguments = ResumedArgs;
    type Res = Value;

    fn call(
        &self,
        _args: ResumedArgs,
        _cx: ToolContext,
    ) -> impl Future<Output = aither_core::Result<Value>> + Send {
        let report = self
            .0
            .resume_report
            .lock()
            .expect("resume report poisoned")
            .clone();
        std::future::ready(Ok(serde_json::to_value(report).unwrap_or_default()))
    }
}
