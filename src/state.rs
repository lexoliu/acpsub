//! The subagent model: per-subagent runtime state, the shared app state, and
//! the spawn/turn machinery.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aither_acp::{
    AcpClient, ClientError, ConfigOption, ConfigOptionValue, ConfigSelectOptions, ContentBlock,
    Implementation, InitializeResult, PlanEntry, PromptParams, PromptResult,
    RequestPermissionOutcome, SessionLoadParams, SessionModeState, SessionNewParams,
    SessionSetConfigOptionParams, SessionSetModeParams, StopReason, TextContent, ToolCall,
    ToolCallLocation, ToolCallStatus, ToolKind,
};
use aither_mcp::transport::ChildProcessTransport;
use serde::Serialize;
use tokio::sync::{Notify, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::config::{AgentConfig, Config, ConfigValue, PermissionPolicy, SteerSemantics};
use crate::error::{Error, Result};
use crate::handler::SubagentHandler;
use crate::registry::{Registry, RegistryEntry, persist};
use crate::terminal::Terminals;
use crate::transcript::TranscriptWriter;

/// Number of agent stderr lines retained for error reporting.
const STDERR_TAIL_LINES: usize = 100;
/// Number of tail lines quoted in a failure reason.
const STDERR_QUOTE_LINES: usize = 20;

/// Shared app state behind every tool.
#[derive(Debug)]
pub struct AppState {
    /// Path of the config file; re-read on every `spawn`/`adopt`/`agents`
    /// call so the running server always reflects the file on disk.
    pub config_path: PathBuf,
    /// Directory holding `<session_id>.jsonl` transcript files (from the
    /// config loaded at startup).
    pub transcript_dir: PathBuf,
    /// Live subagents by session id.
    pub live: Mutex<HashMap<String, Arc<Subagent>>>,
    /// Session ids reserved by an in-flight `adopt`, before the subagent is
    /// live. Locked after `live` wherever both are held.
    pub reserved: Mutex<HashSet<String>>,
    /// The persisted session id → session registry.
    pub registry: Mutex<Registry>,
    /// Serializes registry mutations with their `persist` so the file can
    /// never be overwritten by an older snapshot out of order.
    pub registry_write: tokio::sync::Mutex<()>,
    /// Path of the registry file (`config.defaults.registry`).
    pub registry_path: PathBuf,
}

impl AppState {
    /// Build app state, loading the registry file if it exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the registry file exists but cannot be parsed.
    ///
    /// # Panics
    ///
    /// Panics if the registry mutex is poisoned.
    pub fn new(config: Config, config_path: PathBuf) -> Result<Arc<Self>> {
        let registry_path = config.defaults.registry;
        Ok(Arc::new(Self {
            config_path,
            transcript_dir: config.defaults.transcript_dir,
            live: Mutex::new(HashMap::new()),
            reserved: Mutex::new(HashSet::new()),
            registry: Mutex::new(Registry::load(&registry_path)?),
            registry_write: tokio::sync::Mutex::new(()),
            registry_path,
        }))
    }

    /// Re-read and parse the config file.
    ///
    /// `spawn`, `adopt`, and `agents` call this so a running server always
    /// reflects the file on disk: edits take effect without a restart, and
    /// a missing or unparsable file is an error — never a fallback to a
    /// previously loaded config.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigMissing`], [`Error::ConfigRead`], or
    /// [`Error::ConfigParse`].
    pub fn load_config(&self) -> Result<Config> {
        Config::load(&self.config_path)
    }

    /// Get a live subagent by session id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownSession`] when the session is not live.
    ///
    /// # Panics
    ///
    /// Panics if the live mutex is poisoned.
    pub fn get(&self, session_id: &str) -> Result<Arc<Subagent>> {
        self.live
            .lock()
            .expect("live poisoned")
            .get(session_id)
            .cloned()
            .ok_or_else(|| Error::UnknownSession(session_id.to_string()))
    }

    /// Mutate the registry, then persist the result.
    ///
    /// `registry_write` is held across both steps, so concurrent tool calls
    /// can interleave freely without an older snapshot landing on disk after
    /// a newer one.
    ///
    /// # Errors
    ///
    /// Returns an error if the registry cannot be serialized or written.
    ///
    /// # Panics
    ///
    /// Panics if the registry mutex is poisoned.
    pub async fn update_registry(&self, mutate: impl FnOnce(&mut Registry)) -> Result<()> {
        let _write = self.registry_write.lock().await;
        let snapshot = {
            let mut registry = self.registry.lock().expect("registry poisoned");
            mutate(&mut registry);
            registry.snapshot()
        }?;
        persist(&self.registry_path, snapshot).await
    }
}

/// Subagent lifecycle status.
#[derive(Debug, Clone)]
pub enum Status {
    /// Session established, no turn has run yet.
    Idle,
    /// A `session/prompt` turn is in flight.
    Running,
    /// A permission request is queued for the `permit` tool.
    NeedsPermission,
    /// The last turn finished; carries its stop reason.
    Done(StopReason),
    /// The last turn was cancelled.
    Cancelled,
    /// The last turn errored while the agent process stays alive; the same
    /// session takes another prompt. Carries the reason.
    Failed(String),
    /// The agent process exited; the session is a husk `adopt` reaps (or
    /// `close`/`forget` removes). Carries the reason.
    Exited(String),
}

impl Status {
    /// Stable lowercase name used in tool output.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Running => "running",
            Self::NeedsPermission => "needs_permission",
            Self::Done(_) => "done",
            Self::Cancelled => "cancelled",
            Self::Failed(_) => "failed",
            Self::Exited(_) => "exited",
        }
    }

    /// Whether a `send`/`spawn` prompt may start a new turn: `failed`
    /// qualifies because the agent process is still alive — the turn
    /// errored, not the session.
    #[must_use]
    pub const fn accepts_prompt(&self) -> bool {
        matches!(
            self,
            Self::Idle | Self::Done(_) | Self::Cancelled | Self::Failed(_)
        )
    }
}

/// A tool call observed in a turn, for `wait`/`status` output.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallSummary {
    /// Tool call id.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Tool kind, `snake_case`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<ToolKind>,
    /// Latest status, `snake_case`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ToolCallStatus>,
    /// Affected locations.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<ToolCallLocation>,
    /// When the call first appeared in a `session/update` — the start every
    /// digest duration is measured from. Always set: a summary exists only
    /// because an update carrying the call's id was seen.
    pub started_at: jiff::Timestamp,
    /// When the call first reached a terminal status (`completed` or
    /// `failed`); absent while it is still running.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<jiff::Timestamp>,
}

/// Whether a tool call status ends the call — `completed` or `failed`.
/// `pending`/`in_progress` (or no status) leave it open.
pub(crate) const fn terminal_status(status: ToolCallStatus) -> bool {
    matches!(status, ToolCallStatus::Completed | ToolCallStatus::Failed)
}

/// One prompt turn's accumulated state.
#[derive(Debug, Default)]
pub struct Turn {
    /// 1-based turn number.
    pub n: u64,
    /// Wall-clock start — recorded in the registry as `turn_started` and
    /// the base `wait`'s `expect_secs` is measured from.
    pub started_at: Option<jiff::Timestamp>,
    /// Concatenated `agent_message_chunk` text — the turn's reply.
    pub reply: String,
    /// Tool calls by id.
    pub tool_calls: BTreeMap<String, ToolCallSummary>,
    /// Id of the tool call most recently seen in an update.
    pub latest_tool_call: Option<String>,
    /// Latest plan entries.
    pub plan: Vec<PlanEntry>,
    /// Latest mode id seen in `current_mode_update`.
    pub mode: Option<String>,
    /// Stop reason once the turn ended.
    pub stop_reason: Option<StopReason>,
}

/// A permission request queued by the `ask` policy.
#[derive(Debug)]
pub struct PendingPermission {
    /// Minted request id (`perm-N`), passed to the `permit` tool.
    pub request_id: String,
    /// The tool call the agent wants to run.
    pub tool_call: ToolCall,
    /// Options the agent offered.
    pub options: Vec<aither_acp::PermissionOption>,
    /// Channel answering the agent's request.
    pub answer: oneshot::Sender<RequestPermissionOutcome>,
}

/// Mutable subagent state shared between the tools and the handler.
#[derive(Debug)]
pub struct Inner {
    /// Lifecycle status.
    pub status: Status,
    /// ACP session id.
    pub session_id: String,
    /// `agentInfo` from `initialize`.
    pub agent_info: Option<Implementation>,
    /// Modes reported by `session/new`/`session/load`.
    pub modes: Option<SessionModeState>,
    /// Config options reported by the session calls.
    pub config_options: Vec<ConfigOption>,
    /// Turns from a resumed session's earlier life (registry `turns`); in-memory
    /// `turns` only holds this process's turns.
    pub turn_offset: u64,
    /// Completed turns.
    pub turns: Vec<Turn>,
    /// The running turn, if any.
    pub current: Option<Turn>,
    /// Start recorded for a turn whose slot was reserved by a queue pop
    /// (`settle_turn`/`steer_resolved`) but whose `begin_turn` has not run
    /// yet — the status is `Running` throughout the gap, so `wait`'s
    /// `expect_secs` budget measures from it until `current` exists.
    pub pending_turn_start: Option<jiff::Timestamp>,
    /// Permission requests awaiting `permit`, oldest first.
    pub pending: Vec<PendingPermission>,
    /// Prompts parked by `send` with `policy: "queued"`, oldest first. Fired
    /// one at a time as each turn ends; dropped when a turn ends `cancelled`
    /// or fails, and on `cancel`/`close`/`forget`/agent disconnect.
    pub queue: VecDeque<String>,
    /// Ids of steered prompts whose `session/prompt` result is still in
    /// flight and still holds the prompt slot, oldest first — the wire
    /// order their turns would run in. A steer that lands after its target
    /// turn ended runs as a turn of its own; the first untracked
    /// `session/update` materializes a `current` for it (see
    /// `steer_owner`), and the queue holds until they resolve.
    pub steer_pending: VecDeque<u64>,
    /// Steers settled as folded at their target turn's end whose
    /// `session/prompt` response is still outstanding, oldest first —
    /// always older than every `steer_pending` id. Only a
    /// [`SteerSemantics::Folded`] agent produces these; the contract says
    /// they never resolve — but when one was actually a late arrival that
    /// runs as a turn of its own, the first untracked `session/update`
    /// still materializes a `current` owned by it and its response
    /// settles that turn.
    pub steer_folded: VecDeque<u64>,
    /// The pending steer that owns the materialized `current` turn — set
    /// when `current` was not opened by a `send`/`spawn`/`adopt` prompt but
    /// synthesized for untracked `session/update` traffic.
    pub steer_owner: Option<u64>,
}

/// Everything the handler, tools, and background tasks share about one
/// subagent. Held by `Arc` inside both [`Subagent`] and [`SubagentHandler`].
#[derive(Debug)]
pub struct SubRuntime {
    /// State machine and turn data.
    pub inner: Mutex<Inner>,
    /// Wakes `wait`/`wait_any` callers on every status change.
    pub notify: Notify,
    /// Serializes every `session/prompt` wire-send — turn prompts, queued
    /// handoffs, and steers — so a steer can never overtake the prompt of
    /// the turn it steers into and the between-turns transition stays atomic.
    /// `Arc` so a queued-turn handoff can move the owned guard into the
    /// chained task.
    pub prompt_send: Arc<tokio::sync::Mutex<()>>,
    /// The agent's declared steer contract (config `steer`): what a
    /// `session/prompt` sent mid-turn becomes.
    pub steer: SteerSemantics,
    /// The JSONL transcript file.
    pub transcript: tokio::sync::Mutex<TranscriptWriter>,
    /// Live terminals by id.
    pub terminals: Terminals,
    /// Last `STDERR_TAIL_LINES` lines of agent stderr.
    pub stderr_tail: Mutex<VecDeque<String>>,
    /// Permission request id counter.
    pub next_perm: AtomicU64,
    /// Terminal id counter.
    pub next_term: AtomicU64,
    /// Steer id counter.
    pub next_steer: AtomicU64,
    /// Set by `close`/`forget` before the connection is shut down.
    pub closing: AtomicBool,
    /// Process id of the coordinator that owns this subagent, or 0 when
    /// none was claimed at `spawn`/`adopt`. The daemon reaps a subagent
    /// whose owner is dead, so sessions cannot outlive their coordinator
    /// as unsupervised orphans.
    pub owner: AtomicU32,
}

impl SubRuntime {
    /// Replace the status and wake waiters.
    ///
    /// # Panics
    ///
    /// Panics if the inner mutex is poisoned.
    pub fn transition(&self, status: Status) {
        self.inner.lock().expect("inner poisoned").status = status;
        self.notify.notify_waiters();
    }

    /// The stderr tail, newest last.
    ///
    /// # Panics
    ///
    /// Panics if the tail mutex is poisoned.
    pub fn stderr_tail(&self) -> Vec<String> {
        self.stderr_tail
            .lock()
            .expect("stderr tail poisoned")
            .iter()
            .cloned()
            .collect()
    }

    /// The session id for log fields; empty until the session is
    /// established.
    ///
    /// # Panics
    ///
    /// Panics if the inner mutex is poisoned.
    pub fn session_label(&self) -> String {
        self.inner
            .lock()
            .expect("inner poisoned")
            .session_id
            .clone()
    }
}

/// A live subagent: one agent process, one ACP session, one transcript.
#[derive(Debug)]
pub struct Subagent {
    /// ACP session id — the handle every other tool addresses.
    pub session_id: String,
    /// Agent config key.
    pub agent: String,
    /// Session working directory.
    pub cwd: PathBuf,
    /// Effective permission policy.
    pub permission: PermissionPolicy,
    /// Whether agent fs/* requests may leave `cwd`.
    pub allow_outside_cwd: bool,
    /// The model the agent accepted (`session/set_config_option` `model`),
    /// `None` when the agent advertises no `model` option.
    pub model: Option<serde_json::Value>,
    /// The mode the agent accepted (`session/set_mode`), `None` when the
    /// agent advertises no session modes.
    pub mode: Option<String>,
    /// The transcript file path.
    pub transcript_path: PathBuf,
    /// The ACP client handle.
    pub client: AcpClient<SubagentHandler>,
    /// Connection driver task; also marks the subagent `exited` on disconnect.
    /// Taken by `close`/`forget` to await shutdown.
    pub conn: Mutex<Option<JoinHandle<()>>>,
    /// stderr pump task; taken by `close`/`forget`.
    pub stderr_pump: Mutex<Option<JoinHandle<()>>>,
    /// Shared mutable state.
    pub rt: Arc<SubRuntime>,
}

impl Subagent {
    /// Whether the coordinator process that claimed this subagent at
    /// `spawn`/`adopt` is gone — meaning the daemon may reap it. Always
    /// `false` for an unowned subagent.
    ///
    /// # Panics
    ///
    /// Panics never; a 0 owner means "unowned".
    #[must_use]
    pub fn owner_dead(&self) -> bool {
        let pid = self.rt.owner.load(Ordering::Relaxed);
        pid != 0 && !pid_alive(pid)
    }
}

/// Whether `pid` exists on this machine: `kill(pid, 0)` reports
/// `ESRCH` for a dead pid and `EPERM` for a live one we may not signal.
#[must_use]
pub fn pid_alive(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    // SAFETY: signal 0 performs liveness checking only; nothing is sent.
    (unsafe { libc::kill(pid, 0) } == 0)
        || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Current RFC 3339 timestamp.
#[must_use]
pub fn now() -> String {
    jiff::Timestamp::now().to_string()
}

/// Reject an empty session id.
///
/// # Errors
///
/// Returns [`Error::EmptySessionId`] for an empty id.
pub const fn validate_session_id(session_id: &str) -> Result<()> {
    if session_id.is_empty() {
        Err(Error::EmptySessionId)
    } else {
        Ok(())
    }
}

/// Arguments for [`launch`], as given by the `spawn` and `adopt` tools.
#[derive(Debug)]
pub struct Launch {
    /// Agent config key (already resolved).
    pub agent: String,
    /// Session working directory (already resolved).
    pub cwd: PathBuf,
    /// First turn's prompt, if the launch should start one. `adopt` may bind
    /// the session without prompting.
    pub prompt: Option<String>,
    /// Session mode to activate with `session/set_mode`. Required when the
    /// agent advertises a mode list; must be `None` when it advertises
    /// none — no `set_mode` call is made then.
    pub mode: Option<String>,
    /// Session model to set with `session/set_config_option` on the `model`
    /// option. Required when the agent advertises a `model` option; must be
    /// `None` when it advertises none.
    pub model: Option<String>,
    /// Extra `session/set_config_option` values applied after `model`.
    pub config: BTreeMap<String, ConfigValue>,
    /// Permission policy override.
    pub permission: Option<PermissionPolicy>,
    /// `adopt`: the existing session id to load instead of `session/new`.
    pub load: Option<String>,
    /// Owning coordinator pid, for the daemon's orphan reaper. `None`
    /// leaves the subagent unowned — nothing reaps it.
    pub owner: Option<u32>,
}

/// Spawn an agent process, run the ACP handshake, and register the subagent.
///
/// When `prompt` is set, starts the first turn. Returns once the session is
/// established (and the prompt, if any, is in flight).
///
/// # Errors
///
/// Returns [`Error::EmptySessionId`], [`Error::UnknownAgent`],
/// [`Error::SessionLive`], [`Error::LoadUnsupported`],
/// [`Error::SpawnFailed`], or [`Error::Agent`] when the ACP handshake fails.
/// The spawned process is killed on any error.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub async fn launch(state: &Arc<AppState>, config: &Config, args: Launch) -> Result<Arc<Subagent>> {
    let agent_cfg = config.agent(&args.agent)?.clone();
    let cwd = args.cwd.canonicalize().map_err(|source| {
        Error::io(format!("cannot resolve cwd {}", args.cwd.display()), source)
    })?;
    // Claim the session id atomically across the live and in-flight maps:
    // the MCP server runs `tools/call`s concurrently, so two `adopt`s of one
    // session could interleave between this check and `live.insert` below.
    // An `exited` live entry is a husk — its agent process is already gone —
    // so `adopt` reaps it here rather than requiring a `close` first.
    if let Some(session_id) = &args.load {
        validate_session_id(session_id)?;
        let reap = {
            let mut live = state.live.lock().expect("live poisoned");
            let mut reserved = state.reserved.lock().expect("reserved poisoned");
            let husk = live.get(session_id).is_some_and(|sub| {
                matches!(
                    sub.rt.inner.lock().expect("inner poisoned").status,
                    Status::Exited(_)
                )
            });
            if (live.contains_key(session_id) && !husk) || !reserved.insert(session_id.clone()) {
                return Err(Error::SessionLive(session_id.clone()));
            }
            drop(reserved);
            if husk { live.remove(session_id) } else { None }
        };
        if let Some(sub) = reap {
            close_sub(&sub).await;
        }
    }
    let result = launch_inner(state, config, &args, agent_cfg, cwd).await;
    if result.is_err()
        && let Some(session_id) = &args.load
    {
        state
            .reserved
            .lock()
            .expect("reserved poisoned")
            .remove(session_id);
    }
    result
}

/// Everything `launch` does once preflight passed: spawn the process, run
/// the handshake, register and go live, start the first turn if prompted.
async fn launch_inner(
    state: &Arc<AppState>,
    config: &Config,
    args: &Launch,
    agent_cfg: AgentConfig,
    cwd: PathBuf,
) -> Result<Arc<Subagent>> {
    let permission = args
        .permission
        .or(agent_cfg.permission)
        .unwrap_or(config.defaults.permission);
    let rt = Arc::new(SubRuntime {
        inner: Mutex::new(Inner {
            status: Status::Idle,
            // `adopt` knows its session id up front; `spawn` fills it in
            // during the handshake.
            session_id: args.load.clone().unwrap_or_default(),
            agent_info: None,
            modes: None,
            config_options: Vec::new(),
            turn_offset: 0,
            turns: Vec::new(),
            current: None,
            pending_turn_start: None,
            pending: Vec::new(),
            queue: VecDeque::new(),
            steer_pending: VecDeque::new(),
            steer_folded: VecDeque::new(),
            steer_owner: None,
        }),
        notify: Notify::new(),
        prompt_send: Arc::new(tokio::sync::Mutex::new(())),
        steer: agent_cfg.steer,
        transcript: tokio::sync::Mutex::new(TranscriptWriter::deferred()),
        terminals: Terminals::new(HashMap::new()),
        stderr_tail: Mutex::new(VecDeque::new()),
        next_perm: AtomicU64::new(1),
        next_steer: AtomicU64::new(1),
        next_term: AtomicU64::new(1),
        closing: AtomicBool::new(false),
        owner: AtomicU32::new(args.owner.unwrap_or(0)),
    });

    let handler = SubagentHandler::new(
        rt.clone(),
        cwd.clone(),
        permission,
        agent_cfg.allow_outside_cwd,
    );
    let (client, conn, stderr_pump) = connect(&agent_cfg, &cwd, handler, rt.clone())?;

    // On any handshake failure, kill the child before returning the error.
    let accepted = match handshake(state, &client, &rt, args).await {
        Ok(accepted) => accepted,
        Err(error) => {
            rt.closing.store(true, Ordering::Relaxed);
            client.close();
            let _ = conn.await;
            return Err(error);
        }
    };
    let session_id = accepted.session_id.clone();

    let transcript_path = config
        .defaults
        .transcript_dir
        .join(format!("{session_id}.jsonl"));
    if let Err(source) = rt.transcript.lock().await.bind(&transcript_path).await {
        rt.closing.store(true, Ordering::Relaxed);
        client.close();
        let _ = conn.await;
        return Err(Error::io(
            format!("cannot open {}", transcript_path.display()),
            source,
        ));
    }

    let sub = Arc::new(Subagent {
        session_id: session_id.clone(),
        agent: args.agent.clone(),
        cwd,
        permission,
        allow_outside_cwd: agent_cfg.allow_outside_cwd,
        model: accepted.model,
        mode: accepted.mode,
        transcript_path,
        client,
        conn: Mutex::new(Some(conn)),
        stderr_pump: Mutex::new(Some(stderr_pump)),
        rt,
    });
    go_live(state, &sub).await?;
    if let Some(session_id) = &args.load {
        state
            .reserved
            .lock()
            .expect("reserved poisoned")
            .remove(session_id);
    }
    if let Some(prompt) = &args.prompt
        && let Err(error) = start_turn(state.clone(), &sub, prompt.clone()).await
    {
        state
            .live
            .lock()
            .expect("live poisoned")
            .remove(&session_id);
        return Err(error);
    }
    Ok(sub)
}

/// Insert the subagent into the live map, refusing (and killing its
/// process) when an agent produced a session id that is already live.
async fn go_live(state: &Arc<AppState>, sub: &Arc<Subagent>) -> Result<()> {
    let collision = {
        let mut live = state.live.lock().expect("live poisoned");
        if live.contains_key(&sub.session_id) {
            true
        } else {
            live.insert(sub.session_id.clone(), sub.clone());
            false
        }
    };
    if !collision {
        return Ok(());
    }
    // A second live session with the same id (an agent reusing ids) would
    // shadow the first; refuse and kill this process.
    sub.rt.closing.store(true, Ordering::Relaxed);
    sub.client.clone().close();
    let conn = sub.conn.lock().expect("conn poisoned").take();
    if let Some(conn) = conn {
        let _ = conn.await;
    }
    Err(Error::SessionLive(sub.session_id.clone()))
}

/// Spawn the child process, capture its stderr, and connect the ACP client.
///
/// # Errors
///
/// Returns [`Error::SpawnFailed`] when the process cannot be spawned, or
/// [`Error::Agent`] when the transport cannot be built from it.
fn connect(
    agent_cfg: &AgentConfig,
    cwd: &std::path::Path,
    handler: SubagentHandler,
    rt: Arc<SubRuntime>,
) -> Result<(AcpClient<SubagentHandler>, JoinHandle<()>, JoinHandle<()>)> {
    let mut command = async_process::Command::new(&agent_cfg.command);
    command
        .args(&agent_cfg.args)
        .envs(agent_cfg.env.clone())
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|source| Error::SpawnFailed {
        agent: agent_cfg.command.clone(),
        command: format!("{} {}", agent_cfg.command, agent_cfg.args.join(" ")),
        source,
    })?;
    let stderr = child.stderr.take().ok_or_else(|| {
        Error::io(
            "agent stderr pipe",
            std::io::Error::other("stderr was not piped"),
        )
    })?;
    let transport = ChildProcessTransport::from_child(child)
        .map_err(|error| ClientError::Transport(error.to_string()))
        .map_err(|source| Error::Agent {
            agent: agent_cfg.command.clone(),
            source,
        })?;
    let (client, connection) = AcpClient::connect(transport, handler);
    let conn_rt = rt.clone();
    let conn = tokio::spawn(async move {
        connection.await;
        on_disconnect(&conn_rt);
    });
    let stderr_pump = tokio::spawn(pump_stderr(stderr, rt));
    Ok((client, conn, stderr_pump))
}

/// What the agent accepted during the handshake: the session id and the
/// effective model and mode, read back from the `session/set_mode` /
/// `session/set_config_option` answers — the agent's own report, not the
/// requested strings echoed back. `model`/`mode` are `None` for an agent
/// that advertises neither a `model` option nor session modes.
struct Accepted {
    /// ACP session id (agent-assigned for `session/new`, the adopted id for
    /// `session/load`).
    session_id: String,
    /// The `model` option's current value after `set_config_option`.
    model: Option<serde_json::Value>,
    /// The session's current mode after `set_mode`, from the agent's
    /// `current_mode_update` or its acceptance of the request.
    mode: Option<String>,
}

/// `session/new` or `session/load` — the latter only when the agent
/// advertised `loadSession`. Returns the session id and the modes and
/// config options the agent advertised.
async fn open_session(
    client: &AcpClient<SubagentHandler>,
    init: &InitializeResult,
    args: &Launch,
    agent_error: &(impl Fn(ClientError) -> Error + Sync),
) -> Result<(String, Option<SessionModeState>, Option<Vec<ConfigOption>>)> {
    if let Some(load) = &args.load {
        if !init.agent_capabilities.load_session {
            return Err(Error::LoadUnsupported(args.agent.clone()));
        }
        let result = client
            .load_session(SessionLoadParams::new(load.clone(), args.cwd.clone()))
            .await
            .map_err(agent_error)?;
        Ok((load.clone(), result.modes, result.config_options))
    } else {
        let result = client
            .new_session(SessionNewParams::new(args.cwd.clone()))
            .await
            .map_err(agent_error)?;
        Ok((result.session_id, result.modes, result.config_options))
    }
}

/// `initialize`, `session/new` or `session/load`, `set_mode` when a mode
/// was given, and the `set_config_option` calls; upserts the registry
/// entry. Returns the session id and the model/mode the agent accepted.
async fn handshake(
    state: &Arc<AppState>,
    client: &AcpClient<SubagentHandler>,
    rt: &SubRuntime,
    args: &Launch,
) -> Result<Accepted> {
    let agent_error = |source: ClientError| Error::Agent {
        agent: args.agent.clone(),
        source,
    };
    let init = client.initialize().await.map_err(agent_error)?;
    let (session_id, modes, config_options) =
        open_session(client, &init, args, &agent_error).await?;
    // Check every requested value against what the agent advertised before
    // applying anything: an unverifiable or unknown value fails the launch
    // and the process is closed by the caller, so no half-configured
    // session is registered.
    let config_options = config_options.unwrap_or_default();
    let (modes, options) = check_advertised(&args.agent, args, modes, &config_options)?;
    let current_mode = modes.as_ref().map(|modes| modes.current_mode_id.clone());
    let turn_offset = state
        .registry
        .lock()
        .expect("registry poisoned")
        .get(&session_id)
        .map_or(0, |entry| entry.turns);
    {
        let mut inner = rt.inner.lock().expect("inner poisoned");
        inner.session_id.clone_from(&session_id);
        inner.agent_info = init.agent_info;
        inner.modes = modes;
        inner.config_options = config_options;
        inner.turn_offset = turn_offset;
    }
    // `set_mode` runs only for a requested mode — `check_advertised`
    // already paired it with an advertised mode list.
    if let Some(mode) = args.mode.as_deref() {
        client
            .set_mode(SessionSetModeParams::new(
                session_id.clone(),
                mode.to_string(),
            ))
            .await
            .map_err(agent_error)?;
        note_set_mode(
            &mut rt.inner.lock().expect("inner poisoned"),
            current_mode.as_deref(),
            mode,
        );
    }
    let mut accepted_model = options.get("model").map(|value| match value {
        ConfigValue::Select(value) => serde_json::Value::from(value.clone()),
        ConfigValue::Toggle(flag) => serde_json::Value::from(*flag),
    });
    for (id, value) in options {
        let value = match value {
            ConfigValue::Select(value) => aither_acp::SessionConfigValue::from(value),
            ConfigValue::Toggle(value) => aither_acp::SessionConfigValue::from(value),
        };
        let updated = client
            .set_config_option(SessionSetConfigOptionParams::new(
                session_id.clone(),
                id.clone(),
                value,
            ))
            .await
            .map_err(agent_error)?;
        // The option list the agent returns is what it accepted; the
        // reported model is the `model` option's value after that call.
        if id == "model"
            && let Some(current) = updated
                .iter()
                .find(|option| option.id == "model")
                .and_then(|option| option.current_value.as_ref())
            && let Ok(value) = serde_json::to_value(current)
        {
            accepted_model = Some(value);
        }
        rt.inner.lock().expect("inner poisoned").config_options = updated;
    }
    let accepted_mode = confirmed_mode(&mut rt.inner.lock().expect("inner poisoned"));
    // A registered session keeps its `created` timestamp and turn count;
    // a new one gets a fresh entry.
    let agent = args.agent.clone();
    let cwd = args.cwd.clone();
    state
        .update_registry(|registry| match registry.get_mut(&session_id) {
            Some(entry) => {
                entry.agent.clone_from(&agent);
                entry.cwd.clone_from(&cwd);
            }
            None => registry.insert(
                session_id.clone(),
                RegistryEntry {
                    agent: agent.clone(),
                    cwd: cwd.clone(),
                    created: now(),
                    last_turn: None,
                    turn_started: None,
                    turns: 0,
                },
            ),
        })
        .await?;
    debug!(%session_id, agent = %args.agent, "subagent session established");
    Ok(Accepted {
        session_id,
        model: accepted_model,
        mode: accepted_mode,
    })
}

/// Check `args`' mode, model and config values against what the agent
/// advertised in `session/new`/`session/load`, before anything is applied.
/// Each of `mode`/`model` pairs with its advertisement: required when the
/// agent advertises the surface (a mode list / a `model` config option),
/// rejected when passed to an agent that advertises none, and otherwise
/// checked the same as before — a mode the agent does not list, a config
/// id it does not advertise, or a select value outside the advertised set
/// fails the launch.
///
/// Returns the advertised mode state (when there is one) and the merged
/// `model` + `config` option map to apply.
fn check_advertised(
    agent: &str,
    args: &Launch,
    modes: Option<SessionModeState>,
    config_options: &[ConfigOption],
) -> Result<(Option<SessionModeState>, BTreeMap<String, ConfigValue>)> {
    let modes = match (modes, args.mode.as_deref()) {
        (Some(modes), Some(mode)) => {
            let valid: Vec<String> = modes
                .available_modes
                .iter()
                .map(|mode| mode.id.clone())
                .collect();
            if !valid.iter().any(|id| id == mode) {
                return Err(Error::UnknownMode {
                    agent: agent.to_string(),
                    value: mode.to_string(),
                    valid,
                });
            }
            Some(modes)
        }
        (Some(modes), None) => {
            return Err(Error::ModeRequired {
                agent: agent.to_string(),
                valid: modes
                    .available_modes
                    .iter()
                    .map(|mode| mode.id.clone())
                    .collect(),
            });
        }
        (None, Some(mode)) => {
            return Err(Error::ModesNotAdvertised {
                agent: agent.to_string(),
                value: mode.to_string(),
            });
        }
        (None, None) => None,
    };
    // `model` pairs with the `model` config option the same way: required
    // when the agent advertises one, rejected when given to an agent that
    // advertises none. A `config`-supplied `model` key counts as given.
    let mut options = args.config.clone();
    if let Some(model) = args.model.as_deref() {
        options.insert("model".to_string(), ConfigValue::Select(model.to_string()));
    }
    if config_options.iter().any(|option| option.id == "model") {
        if !options.contains_key("model") {
            return Err(Error::ModelRequired {
                agent: agent.to_string(),
            });
        }
    } else if let Some(model) = args.model.as_deref() {
        return Err(Error::ModelNotAdvertised {
            agent: agent.to_string(),
            value: model.to_string(),
        });
    }
    for (id, value) in &options {
        check_config_option(agent, config_options, id, value)?;
    }
    Ok((modes, options))
}

/// Record the mode `set_mode` left in effect: a `current_mode_update`
/// received while the call was in flight already reported it; with none,
/// the agent's acceptance of the request stands.
fn note_set_mode(inner: &mut Inner, prior: Option<&str>, requested: &str) {
    let modes = inner.modes.as_mut().expect("modes recorded");
    if Some(modes.current_mode_id.as_str()) == prior {
        requested.clone_into(&mut modes.current_mode_id);
    }
}

/// The agent-confirmed mode after every set call: a `mode` config option's
/// current value, when the agent exposes one, is its own post-set report —
/// it can differ from the request when `set_mode` succeeded but kept
/// another mode. Without such an option the modes state's current id —
/// a `current_mode_update` when one arrived, else the accepted request —
/// is the report.
fn confirmed_mode(inner: &mut Inner) -> Option<String> {
    let confirmed = inner
        .config_options
        .iter()
        .find(|option| option.id == "mode" || option.category.as_deref() == Some("mode"))
        .and_then(|option| option.current_value.as_ref())
        .and_then(|value| match value {
            ConfigOptionValue::Selected(id) => Some(id.clone()),
            ConfigOptionValue::Toggle(_) => None,
        });
    match inner.modes.as_mut() {
        Some(modes) => {
            if let Some(confirmed) = confirmed {
                modes.current_mode_id = confirmed;
            }
            Some(modes.current_mode_id.clone())
        }
        // An agent with no mode list can still report a `mode` config
        // option's current value; otherwise there is no mode to report.
        None => confirmed,
    }
}

/// Check a `model`/`config` value against the options the agent advertised
/// in `session/new`/`session/load`: the option must be advertised at all,
/// and a `select` value must be one of its listed values — an option with
/// no advertised values leaves the request uncheckable.
fn check_config_option(
    agent: &str,
    options: &[ConfigOption],
    id: &str,
    value: &ConfigValue,
) -> Result<()> {
    let option = options
        .iter()
        .find(|option| option.id == id)
        .ok_or_else(|| Error::UnknownConfigOption {
            agent: agent.to_string(),
            id: id.to_string(),
            known: options.iter().map(|option| option.id.clone()).collect(),
        })?;
    let valid: Vec<String> = option
        .options
        .as_ref()
        .map(|options| match options {
            ConfigSelectOptions::Flat(flat) => {
                flat.iter().map(|option| option.value.clone()).collect()
            }
            ConfigSelectOptions::Grouped(groups) => groups
                .iter()
                .flat_map(|group| group.options.iter().map(|option| option.value.clone()))
                .collect(),
        })
        .unwrap_or_default();
    let uncheckable = |value: String| Error::ConfigOptionValuesMissing {
        agent: agent.to_string(),
        id: id.to_string(),
        value,
    };
    let unknown = |value: String| Error::UnknownConfigValue {
        agent: agent.to_string(),
        id: id.to_string(),
        value,
        valid: valid.clone(),
    };
    match value {
        ConfigValue::Select(value) => {
            if valid.is_empty() {
                Err(uncheckable(value.clone()))
            } else if valid.contains(value) {
                Ok(())
            } else {
                Err(unknown(value.clone()))
            }
        }
        // A `boolean` option advertises no `options`; true/false is its
        // whole domain. For anything else a toggle cannot be checked.
        ConfigValue::Toggle(flag) => {
            if option.kind.as_deref() == Some("boolean")
                || matches!(option.current_value, Some(ConfigOptionValue::Toggle(_)))
            {
                Ok(())
            } else if valid.is_empty() {
                Err(uncheckable(flag.to_string()))
            } else {
                Err(unknown(flag.to_string()))
            }
        }
    }
}

/// Connection task epilogue: a dead agent marks the subagent `exited` unless
/// `close`/`forget` set the closing flag first. A `failed` turn's reason is
/// folded into the exit reason — the exit is the terminal fact.
fn on_disconnect(rt: &SubRuntime) {
    if rt.closing.load(Ordering::Relaxed) {
        return;
    }
    let mut inner = rt.inner.lock().expect("inner poisoned");
    let reason = match &inner.status {
        Status::Failed(reason) => format!("agent process exited (earlier failure: {reason})"),
        _ => "agent process exited".to_string(),
    };
    inner.status = Status::Exited(reason);
    inner.pending.clear();
    inner.queue.clear();
    inner.steer_pending.clear();
    inner.steer_folded.clear();
    inner.steer_owner = None;
    inner.current = None;
    drop(inner);
    rt.notify.notify_waiters();
}

/// Close a subagent: drop pending requests and queues, close the client, and
/// take the connection and stderr tasks down.
///
/// Shared by the `close` and `forget` tools, by `adopt` reaping an `exited`
/// husk, and by the daemon's orphan reaper.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub async fn close_sub(sub: &Subagent) {
    sub.rt.closing.store(true, Ordering::Relaxed);
    {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        for pending in inner.pending.drain(..) {
            let _ = pending.answer.send(RequestPermissionOutcome::Cancelled);
        }
        inner.queue.clear();
        inner.steer_pending.clear();
        inner.steer_folded.clear();
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

/// Pump agent stderr into the tail buffer and tracing.
async fn pump_stderr(stderr: async_process::ChildStderr, rt: Arc<SubRuntime>) {
    use futures_lite::{AsyncBufReadExt, StreamExt};
    let mut lines = futures_lite::io::BufReader::new(stderr).lines();
    while let Some(line) = lines.next().await {
        match line {
            Ok(line) => {
                debug!(subagent = %rt.session_label(), "agent stderr: {line}");
                let mut tail = rt.stderr_tail.lock().expect("stderr tail poisoned");
                if tail.len() >= STDERR_TAIL_LINES {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
            Err(error) => {
                debug!(subagent = %rt.session_label(), %error, "agent stderr read failed");
                return;
            }
        }
    }
}

/// Whether [`begin_turn`] verifies the status before starting a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gate {
    /// The status must accept a prompt — the `send`/`spawn`/`adopt` paths.
    Check,
    /// The queued-turn handoff already reserved the slot: the status is
    /// `Running` with no current turn.
    Chained,
}

/// Whether the status accepts a prompt and no steered prompt is still in
/// flight on the wire — its turn, if the agent runs it as one, comes first.
pub(crate) fn prompt_slot_free(inner: &Inner) -> bool {
    inner.status.accepts_prompt() && inner.steer_pending.is_empty()
}

/// The oldest steer that could still produce turn traffic. Settled-folded
/// steers precede every pending one — a steer moves to `steer_folded` only
/// at a turn's end, before any newer steer exists — so the folded queue's
/// front wins.
pub(crate) fn oldest_steer(inner: &Inner) -> Option<u64> {
    inner
        .steer_folded
        .front()
        .copied()
        .or_else(|| inner.steer_pending.front().copied())
}

/// Begin a prompt turn on an established session.
///
/// Serializes through `rt.prompt_send`, requires a promptable status, and
/// spawns the task that records the turn's end.
///
/// # Errors
///
/// Returns [`Error::NotPromptable`] when the status does not accept a prompt,
/// or an I/O error if the transcript record cannot be written.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub async fn start_turn(state: Arc<AppState>, sub: &Arc<Subagent>, prompt: String) -> Result<()> {
    let _wire = sub.rt.prompt_send.lock().await;
    let (turn_n, fut) = begin_turn(&state, sub, prompt, Gate::Check, false).await?;
    spawn_turn_task(state, sub, fut, turn_n);
    Ok(())
}

/// The number the next turn takes when its slot is stamped: the settled
/// count plus one, shifted by `turn_offset`. `begin_turn` assigns it when
/// `current` is stamped; callers that predict the next turn during the
/// reserved-slot gap must use this same function so the two numberings
/// cannot drift apart.
pub(crate) const fn next_turn_number(inner: &Inner) -> u64 {
    inner.turn_offset + inner.turns.len() as u64 + 1
}

/// The shared tail of `start_turn`, `send`'s steer-fallback, and the
/// queued-turn handoff: marks the turn running, writes the `prompt`
/// transcript record, and enqueues `session/prompt`.
///
/// The caller must hold `rt.prompt_send`: serializing every prompt's
/// wire-send is what keeps a steer behind the prompt of the turn it steers
/// into, and what keeps the between-turns handoff atomic against `send`.
/// The caller also spawns [`turn_task`] on the returned future — see
/// [`spawn_turn_task`].
///
/// `queued` marks the transcript record as queue-fired (rendered
/// `USER (queued)`).
///
/// Returns the turn's number and its `session/prompt` future.
///
/// # Errors
///
/// Returns [`Error::NotPromptable`] when the gate rejects the current
/// status, or an I/O error if the transcript record cannot be written — in
/// which case the subagent is marked `failed`, since the turn can neither
/// run nor be logged.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub(crate) async fn begin_turn(
    state: &Arc<AppState>,
    sub: &Arc<Subagent>,
    prompt: String,
    gate: Gate,
    queued: bool,
) -> Result<(u64, PromptFut)> {
    let started = jiff::Timestamp::now();
    let turn_n = {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        let ready = match gate {
            Gate::Check => prompt_slot_free(&inner),
            Gate::Chained => matches!(inner.status, Status::Running) && inner.current.is_none(),
        };
        if !ready {
            let status = if inner.status.accepts_prompt() {
                "steer in flight".to_string()
            } else {
                inner.status.name().to_string()
            };
            return Err(Error::NotPromptable {
                session_id: sub.session_id.clone(),
                status,
            });
        }
        let n = next_turn_number(&inner);
        inner.current = Some(Turn {
            n,
            started_at: Some(started),
            ..Turn::default()
        });
        inner.pending_turn_start = None;
        inner.status = Status::Running;
        n
    };
    let mut record = serde_json::json!({"ts": now(), "turn": turn_n, "prompt": prompt});
    if queued {
        record["queued"] = serde_json::json!(true);
    }
    if let Err(source) = sub.rt.transcript.lock().await.append(&record).await {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        inner.current = None;
        // `failed`, not `exited`: the agent process is still alive — the
        // turn could not be logged, but the next `send` retries the write
        // and either runs the turn or reports the I/O error again. A
        // disconnect that landed in between keeps `exited`.
        if !matches!(inner.status, Status::Exited(_)) {
            inner.status = Status::Failed(format!("cannot write transcript: {source}"));
        }
        drop(inner);
        sub.rt.notify.notify_waiters();
        return Err(Error::io("cannot write transcript", source));
    }
    // The registry records the turn's start: `wait`'s `expect_secs` budget
    // is measured from it and survives a server restart.
    if let Err(error) = state
        .update_registry(|registry| {
            if let Some(entry) = registry.get_mut(&sub.session_id) {
                entry.turn_started = Some(started.to_string());
            }
        })
        .await
    {
        warn!(%error, "registry persist failed");
    }
    sub.rt.notify.notify_waiters();

    let fut = enqueue_prompt(&sub.client, &sub.session_id, prompt).await;
    Ok((turn_n, fut))
}

/// Spawn the turn-end task for a [`begin_turn`] result.
pub(crate) fn spawn_turn_task(
    state: Arc<AppState>,
    sub: &Arc<Subagent>,
    prompt: PromptFut,
    turn_n: u64,
) {
    tokio::spawn(turn_task(
        state,
        prompt,
        turn_n,
        sub.rt.clone(),
        sub.session_id.clone(),
    ));
}

/// Build a `session/prompt` call and drive it once so the request is on the
/// wire: `AcpClient` enqueues the request on its unbounded outbound channel
/// inside the first poll, so a `cancel` or steer issued right after cannot
/// overtake the prompt it is meant to follow.
async fn enqueue_prompt(
    client: &AcpClient<SubagentHandler>,
    session_id: &str,
    prompt: String,
) -> PromptFut {
    let client = client.clone();
    let session_id = session_id.to_string();
    let mut prompt: PromptFut = Box::pin(async move {
        client
            .prompt(PromptParams::new(
                session_id,
                vec![ContentBlock::Text(TextContent {
                    text: prompt,
                    annotations: None,
                    meta: None,
                })],
            ))
            .await
    });
    let ready = futures_lite::future::poll_fn(|cx| match prompt.as_mut().poll(cx) {
        std::task::Poll::Ready(outcome) => std::task::Poll::Ready(Some(outcome)),
        std::task::Poll::Pending => std::task::Poll::Ready(None),
    })
    .await;
    // A synchronously-failed send resolves on the first poll; wrap the
    // outcome so the same turn-end path handles it.
    ready.map_or(prompt, |outcome| Box::pin(async move { outcome }))
}

/// Inject `prompt` into the running turn as a steering message: a second
/// `session/prompt` on the wire while the turn is in flight.
///
/// The caller holds `rt.prompt_send`, so the steer lands on the wire behind
/// the running turn's own prompt. Agents that support mid-turn injection
/// (devin treats it as an injected user message steering the active task)
/// fold the text into the turn; the returned future resolves with that
/// turn's own result — or with a rejection from agents that do not accept a
/// concurrent prompt, or not at all on a [`SteerSemantics::Folded`] agent,
/// where [`settle_turn`] settles it at the turn's end. A steer that lands
/// after the target turn ended runs as a turn of its own:
/// [`steer_resolved`] reconciles which it became.
///
/// Returns the steer id, the target turn's number, and the prompt call's
/// future, which the caller either awaits briefly for a synchronous
/// rejection or hands to [`steer_waiter`].
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub(crate) async fn steer_turn(
    sub: &Arc<Subagent>,
    prompt: String,
) -> Result<(u64, u64, PromptFut)> {
    let (steer_id, turn_n) = {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        let steer_id = sub.rt.next_steer.fetch_add(1, Ordering::Relaxed);
        inner.steer_pending.push_back(steer_id);
        (steer_id, inner.current.as_ref().map_or(0, |turn| turn.n))
    };
    let record = serde_json::json!({"ts": now(), "turn": turn_n, "steer": prompt});
    if let Err(error) = sub.rt.transcript.lock().await.append(&record).await {
        warn!(%error, "transcript write failed");
    }
    let fut = enqueue_prompt(&sub.client, &sub.session_id, prompt).await;
    Ok((steer_id, turn_n, fut))
}

/// The `steer_end` transcript record for a resolved steer call.
pub(crate) fn steer_end_record(
    turn_n: u64,
    outcome: &std::result::Result<PromptResult, ClientError>,
) -> serde_json::Value {
    match outcome {
        Ok(result) => serde_json::json!({
            "ts": now(),
            "turn": turn_n,
            "steer_end": serde_json::to_value(result.stop_reason).unwrap_or_default(),
        }),
        Err(error) => serde_json::json!({
            "ts": now(),
            "turn": turn_n,
            "steer_end": "error",
            "error": error.to_string(),
        }),
    }
}

/// Reconcile a steered prompt's resolution with the state machine.
///
/// Three fates: the agent folded it into the turn it was sent into — the
/// tracked turn's own `turn_task` owns that end, so only the `steer_end`
/// record is written; the agent ran it as a turn of its own — untracked
/// `session/update` traffic materialized a `current` owned by this steer,
/// so its resolution IS that turn's end and [`finish_turn`] settles it;
/// or it errored — a rejection transfers ownership of a materialized turn
/// to the next pending steer, since the traffic cannot be the rejected
/// one's. A steer already settled `folded` resolves through the same
/// paths: it left `steer_pending` at its target turn's end, but a late
/// arrival's own turn still closes here. When the last in-flight steer
/// resolves without owning a turn and the queue was held for it, the
/// first queued prompt chains here.
///
/// Returns the outcome back to the caller (a synchronous `send(steer)` path
/// reports rejections from it).
pub(crate) async fn steer_resolved(
    state: Arc<AppState>,
    sub: &Arc<Subagent>,
    steer_id: u64,
    target_turn: u64,
    outcome: std::result::Result<PromptResult, ClientError>,
) -> std::result::Result<PromptResult, ClientError> {
    let (mine, kick, turn_n) = {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        inner.steer_pending.retain(|&id| id != steer_id);
        inner.steer_folded.retain(|&id| id != steer_id);
        let mine = if inner.steer_owner == Some(steer_id) {
            inner.steer_owner = None;
            if outcome.is_err()
                && let Some(owner) = oldest_steer(&inner)
            {
                inner.steer_owner = Some(owner);
                false
            } else {
                true
            }
        } else {
            false
        };
        let kick = !mine
            && inner.steer_pending.is_empty()
            && inner.current.is_none()
            && inner.status.accepts_prompt()
            && !inner.queue.is_empty();
        let prompt = if kick {
            inner.status = Status::Running;
            inner.pending_turn_start = Some(jiff::Timestamp::now());
            inner.queue.pop_front()
        } else {
            None
        };
        let turn_n = inner.current.as_ref().map_or(target_turn, |turn| turn.n);
        drop(inner);
        (mine, prompt, turn_n)
    };
    if mine {
        let returned = outcome.clone();
        finish_turn(
            state,
            sub.rt.clone(),
            sub.session_id.clone(),
            turn_n,
            outcome,
        )
        .await;
        let record = steer_end_record(turn_n, &returned);
        if let Err(error) = sub.rt.transcript.lock().await.append(&record).await {
            warn!(%error, "transcript write failed");
        }
        returned
    } else {
        if let Some(prompt) = kick {
            let wire = sub.rt.prompt_send.clone().lock_owned().await;
            spawn_chained(state, sub.session_id.clone(), prompt, wire);
        }
        let record = steer_end_record(turn_n, &outcome);
        if let Err(error) = sub.rt.transcript.lock().await.append(&record).await {
            warn!(%error, "transcript write failed");
        }
        outcome
    }
}

/// Await a steered prompt's resolution, then [`steer_resolved`] it.
pub(crate) async fn steer_waiter(
    state: Arc<AppState>,
    sub: Arc<Subagent>,
    steer_id: u64,
    target_turn: u64,
    prompt: PromptFut,
) {
    let outcome = prompt.await;
    let _ = steer_resolved(state, &sub, steer_id, target_turn, outcome).await;
}

/// The in-flight `session/prompt` call, already enqueued by `start_turn`.
pub(crate) type PromptFut =
    Pin<Box<dyn Future<Output = std::result::Result<PromptResult, ClientError>> + Send>>;

/// Push the finished turn into `turns` and pick the follow-up state: a
/// `cancelled` turn drops the queue, a queued prompt becomes the next turn
/// (the status stays `running` — no `done` interlude), anything else settles
/// `done`. A steered prompt still in flight may materialize as a turn ahead
/// of the queue on the wire, so the queue pops only once they all resolve.
/// Returns the prompt to chain, the prompts the queue dropped, and the ids
/// of steers settled as folded.
fn settle_turn(
    rt: &SubRuntime,
    reason: StopReason,
) -> (Option<String>, VecDeque<String>, Vec<u64>) {
    let mut inner = rt.inner.lock().expect("inner poisoned");
    if let Some(mut turn) = inner.current.take() {
        turn.stop_reason = Some(reason);
        inner.turns.push(turn);
    }
    let folded = fold_pending_steers(rt, &mut inner);
    let (next, dropped) = if matches!(inner.status, Status::Exited(_)) {
        // The process died while the turn was in flight; `on_disconnect`
        // already cleared the queue and steers and owns the status.
        (None, VecDeque::new())
    } else if reason == StopReason::Cancelled {
        inner.status = Status::Cancelled;
        (None, std::mem::take(&mut inner.queue))
    } else if inner.steer_pending.is_empty()
        && let Some(next) = inner.queue.pop_front()
    {
        // The chained turn's slot is reserved now; `begin_turn` stamps its
        // own start. `Running` must never lack a recorded start — `wait`'s
        // budget would otherwise have nothing to measure from.
        inner.pending_turn_start = Some(jiff::Timestamp::now());
        (Some(next), VecDeque::new())
    } else {
        inner.status = Status::Done(reason);
        (None, VecDeque::new())
    };
    drop(inner);
    (next, dropped, folded)
}

/// A `folded`-semantics agent never answers a steered prompt it folded
/// into the turn that just ended: settle every still-pending steer as
/// folded so the prompt slot frees and the queue can chain. The ids move
/// to `steer_folded`, not oblivion — a steer that actually landed late
/// runs as a turn of its own and still needs to own it.
///
/// Returns the settled steer ids for their `steer_end: folded` records.
fn fold_pending_steers(rt: &SubRuntime, inner: &mut Inner) -> Vec<u64> {
    if rt.steer != SteerSemantics::Folded {
        return Vec::new();
    }
    let folded: Vec<u64> = inner.steer_pending.drain(..).collect();
    inner.steer_folded.extend(folded.iter().copied());
    folded
}

/// Push the finished turn into `turns`, mark the subagent `failed`, and
/// return the dropped queue and the ids of steers settled as folded. A turn
/// error that lands after `on_disconnect` does not downgrade `exited` — the
/// dead process cannot take the prompt `failed` would invite.
fn fail_turn(rt: &SubRuntime, reason: String) -> (VecDeque<String>, Vec<u64>) {
    let mut inner = rt.inner.lock().expect("inner poisoned");
    if let Some(turn) = inner.current.take() {
        inner.turns.push(turn);
    }
    let folded = fold_pending_steers(rt, &mut inner);
    if !matches!(inner.status, Status::Exited(_)) {
        inner.status = Status::Failed(reason);
    }
    (std::mem::take(&mut inner.queue), folded)
}

/// Append a `queue_dropped` transcript record, when any prompts dropped.
pub(crate) async fn record_queue_dropped(
    rt: &SubRuntime,
    turn_n: Option<u64>,
    prompts: VecDeque<String>,
) {
    if prompts.is_empty() {
        return;
    }
    let mut record = serde_json::json!({"ts": now(), "queue_dropped": prompts});
    if let Some(turn_n) = turn_n {
        record["turn"] = serde_json::json!(turn_n);
    }
    if let Err(error) = rt.transcript.lock().await.append(&record).await {
        warn!(%error, "transcript write failed");
    }
}

/// Append a `steer_end: folded` record for each steer settled as folded
/// at a turn's end.
async fn record_steers_folded(rt: &SubRuntime, turn_n: u64, steers: Vec<u64>) {
    for _ in steers {
        let record = serde_json::json!({
            "ts": now(),
            "turn": turn_n,
            "steer_end": "folded",
        });
        if let Err(error) = rt.transcript.lock().await.append(&record).await {
            warn!(%error, "transcript write failed");
        }
    }
}

/// Await `session/prompt`, then [`finish_turn`].
async fn turn_task(
    state: Arc<AppState>,
    prompt: PromptFut,
    turn_n: u64,
    rt: Arc<SubRuntime>,
    session_id: String,
) {
    let outcome = prompt.await;
    finish_turn(state, rt, session_id, turn_n, outcome).await;
}

/// Spawn the queued-turn handoff: the wire guard moves into the task so the
/// queued prompt reaches the wire ahead of any `send` that lands between
/// the turns.
fn spawn_chained(
    state: Arc<AppState>,
    session_id: String,
    prompt: String,
    wire: tokio::sync::OwnedMutexGuard<()>,
) {
    tokio::spawn(async move {
        let _wire = wire;
        match state.get(&session_id) {
            Ok(sub) => match begin_turn(&state, &sub, prompt, Gate::Chained, true).await {
                Ok((n, fut)) => spawn_turn_task(state, &sub, fut, n),
                Err(error) => warn!(%error, "queued turn failed to start"),
            },
            Err(error) => warn!(%error, "queued prompt dropped: subagent gone"),
        }
    });
}

/// The shared settlement tail of `turn_task` and a steer-owned turn's
/// [`steer_resolved`]: record the turn's end in the transcript, settle or
/// fail state, update the registry, and wake waiters.
///
/// While this task holds `rt.prompt_send` the between-turns transition is
/// atomic: a `send`/`steer` sees either the still-`running` pre-transition
/// state or the settled post-transition one, never the gap. A queued prompt
/// popped here becomes the next turn without a `done` interlude; a
/// `cancelled` or failed turn drops the queue instead.
async fn finish_turn(
    state: Arc<AppState>,
    rt: Arc<SubRuntime>,
    session_id: String,
    turn_n: u64,
    outcome: std::result::Result<PromptResult, ClientError>,
) {
    let wire = rt.prompt_send.clone().lock_owned().await;
    match outcome {
        Ok(result) => {
            let reason = result.stop_reason;
            let record = serde_json::json!({
                "ts": now(),
                "turn": turn_n,
                "stop_reason": serde_json::to_value(reason).unwrap_or_default(),
            });
            if let Err(error) = rt.transcript.lock().await.append(&record).await {
                warn!(%error, "transcript write failed");
            }
            let (next, dropped, folded) = settle_turn(&rt, reason);
            record_steers_folded(&rt, turn_n, folded).await;
            if let Some(prompt) = next {
                spawn_chained(state.clone(), session_id.clone(), prompt, wire);
            } else {
                drop(wire);
            }
            record_queue_dropped(&rt, Some(turn_n), dropped).await;
        }
        Err(error) => {
            let mut reason = error.to_string();
            let tail = rt.stderr_tail();
            if !tail.is_empty() {
                let start = tail.len().saturating_sub(STDERR_QUOTE_LINES);
                reason.push_str("\nagent stderr (tail):\n");
                reason.push_str(&tail[start..].join("\n"));
            }
            let record = serde_json::json!({
                "ts": now(),
                "turn": turn_n,
                "stop_reason": "error",
                "error": reason,
            });
            if let Err(error) = rt.transcript.lock().await.append(&record).await {
                warn!(%error, "transcript write failed");
            }
            let (dropped, folded) = fail_turn(&rt, reason);
            record_steers_folded(&rt, turn_n, folded).await;
            record_queue_dropped(&rt, Some(turn_n), dropped).await;
            drop(wire);
        }
    }
    // Update the registry: turn count and last_turn.
    let turns = {
        let inner = rt.inner.lock().expect("inner poisoned");
        inner.turn_offset + inner.turns.len() as u64
    };
    let result = state
        .update_registry(|registry| {
            if let Some(entry) = registry.get_mut(&session_id) {
                entry.turns = turns;
                entry.last_turn = Some(now());
                entry.turn_started = None;
            }
        })
        .await;
    if let Err(error) = result {
        warn!(%error, "registry persist failed");
    }
    rt.notify.notify_waiters();
}

/// Best-effort `cwd` discovery for `adopt`.
///
/// Reads the session's working directory out of the agent's local session
/// database (devin CLI layout — a `sessions` table with `id` and
/// `working_directory` columns). Returns `Ok(None)` when the database has
/// no row for the session.
///
/// # Errors
///
/// Returns an error when the database cannot be opened or queried.
pub fn discover_session_cwd(db: &std::path::Path, session_id: &str) -> Result<Option<PathBuf>> {
    use rusqlite::OptionalExtension;
    let context = || format!("cannot query {}", db.display());
    let conn =
        rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(|source| Error::io(context(), std::io::Error::other(source)))?;
    conn.query_row(
        "SELECT working_directory FROM sessions WHERE id = ?1",
        [session_id],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map(|cwd| cwd.map(PathBuf::from))
    .map_err(|source| Error::io(context(), std::io::Error::other(source)))
}
