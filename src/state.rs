//! The subagent model: per-subagent runtime state, the shared app state, and
//! the spawn/turn machinery.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aither_acp::{
    AcpClient, AgentCapabilities, ClientError, ConfigOption, ContentBlock, Implementation,
    InitializeResult, PlanEntry, PromptParams, PromptResult, RequestPermissionOutcome,
    ResponseFuture, SessionLoadParams, SessionModeState, SessionNewParams, SessionResumeParams,
    SessionSetConfigOptionParams, SessionSetModeParams, StopReason, TextContent, ToolCall,
    ToolCallLocation, ToolCallStatus, ToolKind,
};
use aither_mcp::transport::ChildProcessTransport;
use serde::Serialize;
use tokio::sync::{Notify, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::config::{AgentConfig, Config, ConfigValue, PermissionPolicy};
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
    /// The loaded config file.
    pub config: Config,
    /// Live subagents by name.
    pub live: Mutex<HashMap<String, Arc<Subagent>>>,
    /// Names reserved by an in-flight `spawn`, before the subagent is live.
    /// Locked after `live` wherever both are held.
    pub reserved: Mutex<HashSet<String>>,
    /// The persisted name → session registry.
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
    pub fn new(config: Config) -> Result<Arc<Self>> {
        let registry_path = config.defaults.registry.clone();
        Ok(Arc::new(Self {
            config,
            live: Mutex::new(HashMap::new()),
            reserved: Mutex::new(HashSet::new()),
            registry: Mutex::new(Registry::load(&registry_path)?),
            registry_write: tokio::sync::Mutex::new(()),
            registry_path,
        }))
    }

    /// Get a live subagent by name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownSubagent`] when the name is not live.
    ///
    /// # Panics
    ///
    /// Panics if the live mutex is poisoned.
    pub fn get(&self, name: &str) -> Result<Arc<Subagent>> {
        self.live
            .lock()
            .expect("live poisoned")
            .get(name)
            .cloned()
            .ok_or_else(|| Error::UnknownSubagent(name.to_string()))
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
    /// The agent errored or exited; carries the reason.
    Failed(String),
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
        }
    }

    /// Whether a `send`/`spawn` prompt may start a new turn.
    #[must_use]
    pub const fn accepts_prompt(&self) -> bool {
        matches!(self, Self::Idle | Self::Done(_) | Self::Cancelled)
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
}

/// One prompt turn's accumulated state.
#[derive(Debug, Default)]
pub struct Turn {
    /// 1-based turn number.
    pub n: u64,
    /// Concatenated `agent_message_chunk` text — the turn's reply.
    pub reply: String,
    /// Tool calls by id.
    pub tool_calls: BTreeMap<String, ToolCallSummary>,
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
    /// Permission requests awaiting `permit`, oldest first.
    pub pending: Vec<PendingPermission>,
    /// Follow-up prompts sent while a turn was running; drained oldest
    /// first when the turn completes, dropped on cancel or failure.
    pub queued: VecDeque<String>,
    /// Whether the running turn's `session/prompt` has been written to the
    /// connection. `cancel` waits for this: a `session/cancel` that overtakes
    /// its prompt on the wire is ignored by agents, leaving the turn running.
    pub prompt_sent: bool,
    /// `agentCapabilities` from `initialize`; drives optional features such
    /// as the fork strategy.
    pub agent_capabilities: AgentCapabilities,
}

/// Everything the handler, tools, and background tasks share about one
/// subagent. Held by `Arc` inside both [`Subagent`] and [`SubagentHandler`].
#[derive(Debug)]
pub struct SubRuntime {
    /// State machine and turn data.
    pub inner: Mutex<Inner>,
    /// Wakes `wait`/`wait_any` callers on every status change.
    pub notify: Notify,
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
    /// Set by `close`/`forget` before the connection is shut down.
    pub closing: AtomicBool,
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
}

/// A live subagent: one agent process, one ACP session, one transcript.
#[derive(Debug)]
pub struct Subagent {
    /// Caller-chosen name.
    pub name: String,
    /// Agent config key.
    pub agent: String,
    /// Session working directory.
    pub cwd: PathBuf,
    /// Effective permission policy.
    pub permission: PermissionPolicy,
    /// Whether agent fs/* requests may leave `cwd`.
    pub allow_outside_cwd: bool,
    /// The transcript file path.
    pub transcript_path: PathBuf,
    /// Name this session was forked from, when the subagent is a `fork`.
    pub forked_from: Option<String>,
    /// The ACP client handle.
    pub client: AcpClient<SubagentHandler>,
    /// Connection driver task; also marks the subagent failed on disconnect.
    /// Taken by `close`/`forget` to await shutdown.
    pub conn: Mutex<Option<JoinHandle<()>>>,
    /// stderr pump task; taken by `close`/`forget`.
    pub stderr_pump: Mutex<Option<JoinHandle<()>>>,
    /// Shared mutable state.
    pub rt: Arc<SubRuntime>,
}

/// Current RFC 3339 timestamp.
#[must_use]
pub fn now() -> String {
    jiff::Timestamp::now().to_string()
}

/// Enforce the `[A-Za-z0-9._-]+` name rule.
///
/// # Errors
///
/// Returns [`Error::InvalidName`] for an empty or out-of-charset name.
pub fn validate_name(name: &str) -> Result<()> {
    if !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        Ok(())
    } else {
        Err(Error::InvalidName(name.to_string()))
    }
}

/// Arguments for [`launch`], as given by the `spawn` tool.
#[derive(Debug)]
pub struct Launch {
    /// Subagent name.
    pub name: String,
    /// Agent config key.
    pub agent: String,
    /// Session working directory.
    pub cwd: PathBuf,
    /// First turn's prompt.
    pub prompt: String,
    /// `session/set_mode` override; falls back to the agent's configured mode.
    pub mode: Option<String>,
    /// `session/set_config_option` overrides merged over the agent's `config`.
    pub config: BTreeMap<String, ConfigValue>,
    /// Permission policy override.
    pub permission: Option<PermissionPolicy>,
    /// Forget a registered session of this name instead of resuming it.
    pub replace: bool,
}

/// Spawn an agent process, run the ACP handshake, register the subagent, and
/// start the first turn. Returns once the prompt is in flight.
///
/// # Errors
///
/// Returns [`Error::InvalidName`], [`Error::UnknownAgent`],
/// [`Error::NameTaken`], [`Error::NameRegistered`], [`Error::SpawnFailed`], or
/// [`Error::Agent`] when the ACP handshake fails. The spawned process is
/// killed on any error.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub async fn launch(state: &Arc<AppState>, args: Launch) -> Result<Arc<Subagent>> {
    let (agent_cfg, cwd, prior) = preflight(state, &args)?;
    // `preflight` reserved the name; release the reservation on any failure.
    let result = launch_reserved(state, &args, agent_cfg, cwd, prior).await;
    if result.is_err() {
        state
            .reserved
            .lock()
            .expect("reserved poisoned")
            .remove(&args.name);
    }
    result
}

/// Everything `launch` does once the name is reserved: wire the subagent up,
/// then start the first turn.
async fn launch_reserved(
    state: &Arc<AppState>,
    args: &Launch,
    agent_cfg: AgentConfig,
    cwd: PathBuf,
    prior: Option<RegistryEntry>,
) -> Result<Arc<Subagent>> {
    let permission = args
        .permission
        .or(agent_cfg.permission)
        .unwrap_or(state.config.defaults.permission);
    let sub = wire_subagent(state, args, &agent_cfg, cwd, permission, prior).await?;
    if let Err(error) = start_turn(state.clone(), &sub, args.prompt.clone()).await {
        state.live.lock().expect("live poisoned").remove(&args.name);
        return Err(error);
    }
    Ok(sub)
}

/// Spawn the process, run the handshake, and put the subagent live.
///
/// Shared by `launch_reserved` (spawn) and `fork_launch_reserved` (fork):
/// `prior` drives `session/load`/`session/resume` in the handshake and
/// carries `forked_from` lineage. The first prompt is NOT sent — callers
/// start the turn themselves.
async fn wire_subagent(
    state: &Arc<AppState>,
    args: &Launch,
    agent_cfg: &AgentConfig,
    cwd: PathBuf,
    permission: PermissionPolicy,
    prior: Option<RegistryEntry>,
) -> Result<Arc<Subagent>> {
    let transcript_path = state
        .config
        .defaults
        .transcript_dir
        .join(format!("{}.jsonl", args.name));
    let transcript = TranscriptWriter::create(&transcript_path)
        .await
        .map_err(|source| {
            Error::io(format!("cannot open {}", transcript_path.display()), source)
        })?;
    let rt = Arc::new(SubRuntime {
        inner: Mutex::new(Inner {
            status: Status::Idle,
            session_id: String::new(),
            agent_info: None,
            modes: None,
            config_options: Vec::new(),
            turn_offset: 0,
            turns: Vec::new(),
            current: None,
            pending: Vec::new(),
            queued: VecDeque::new(),
            prompt_sent: false,
            agent_capabilities: AgentCapabilities::default(),
        }),
        notify: Notify::new(),
        transcript: tokio::sync::Mutex::new(transcript),
        terminals: Terminals::new(HashMap::new()),
        stderr_tail: Mutex::new(VecDeque::new()),
        next_perm: AtomicU64::new(1),
        next_term: AtomicU64::new(1),
        closing: AtomicBool::new(false),
    });

    let handler = SubagentHandler::new(
        args.name.clone(),
        rt.clone(),
        cwd.clone(),
        permission,
        agent_cfg.allow_outside_cwd,
    );
    let (client, conn, stderr_pump) =
        connect(agent_cfg, &cwd, handler, rt.clone(), args.name.clone())?;

    let forked_from = prior.as_ref().and_then(|entry| entry.forked_from.clone());
    // On any handshake failure, kill the child before returning the error.
    let result = handshake(state, &client, &rt, args, agent_cfg, prior).await;
    if let Err(error) = result {
        rt.closing.store(true, Ordering::Relaxed);
        client.close();
        let _ = conn.await;
        return Err(error);
    }

    let sub = Arc::new(Subagent {
        name: args.name.clone(),
        agent: args.agent.clone(),
        cwd,
        permission,
        allow_outside_cwd: agent_cfg.allow_outside_cwd,
        transcript_path,
        forked_from,
        client,
        conn: Mutex::new(Some(conn)),
        stderr_pump: Mutex::new(Some(stderr_pump)),
        rt,
    });
    // Once the name is in `live`, `preflight` reports it taken even before
    // the reservation is released — order matters.
    state
        .live
        .lock()
        .expect("live poisoned")
        .insert(args.name.clone(), sub.clone());
    state
        .reserved
        .lock()
        .expect("reserved poisoned")
        .remove(&args.name);
    Ok(sub)
}

/// Everything checked before a process starts: name validity, agent lookup,
/// cwd resolution, live-name collision, and the prior registry entry (unless
/// `replace`). A registered name whose agent differs is rejected up front.
///
/// # Errors
///
/// Returns [`Error::InvalidName`], [`Error::UnknownAgent`],
/// [`Error::NameTaken`], [`Error::NameRegistered`], or an I/O error resolving
/// `cwd`.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
fn preflight(
    state: &Arc<AppState>,
    args: &Launch,
) -> Result<(AgentConfig, PathBuf, Option<RegistryEntry>)> {
    validate_name(&args.name)?;
    let agent_cfg = state.config.agent(&args.agent)?.clone();
    let cwd = args.cwd.canonicalize().map_err(|source| {
        Error::io(format!("cannot resolve cwd {}", args.cwd.display()), source)
    })?;
    // Claim the name atomically across the live and in-flight maps: the MCP
    // server runs `tools/call`s concurrently, so two `spawn`s of one name can
    // interleave between this check and `live.insert` in `launch_reserved`.
    {
        let live = state.live.lock().expect("live poisoned");
        let mut reserved = state.reserved.lock().expect("reserved poisoned");
        if live.contains_key(&args.name) || !reserved.insert(args.name.clone()) {
            return Err(Error::NameTaken(args.name.clone()));
        }
    }
    let prior = if args.replace {
        None
    } else {
        state
            .registry
            .lock()
            .expect("registry poisoned")
            .get(&args.name)
            .cloned()
    };
    // A name registered to a different agent cannot resume that session.
    if let Some(entry) = &prior
        && entry.agent != args.agent
    {
        return Err(Error::NameRegistered(args.name.clone()));
    }
    Ok((agent_cfg, cwd, prior))
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
    name: String,
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
    let stderr_pump = tokio::spawn(pump_stderr(stderr, rt, name));
    Ok((client, conn, stderr_pump))
}

/// `initialize`, `session/new` or `session/load`, `set_mode`, and the
/// `set_config_option` calls. Also writes/refreshes the registry entry.
async fn handshake(
    state: &Arc<AppState>,
    client: &AcpClient<SubagentHandler>,
    rt: &SubRuntime,
    args: &Launch,
    agent_cfg: &AgentConfig,
    prior: Option<RegistryEntry>,
) -> Result<()> {
    let agent_error = |source: ClientError| Error::Agent {
        agent: args.name.clone(),
        source,
    };
    let init = client.initialize().await.map_err(agent_error)?;
    let (session_id, modes, config_options) =
        open_session(client, args, &init, prior.as_ref()).await?;
    {
        let mut inner = rt.inner.lock().expect("inner poisoned");
        inner.session_id.clone_from(&session_id);
        inner.agent_info = init.agent_info;
        inner.agent_capabilities = init.agent_capabilities;
        inner.modes = modes;
        inner.config_options = config_options.unwrap_or_default();
        inner.turn_offset = prior.as_ref().map_or(0, |entry| entry.turns);
    }
    if let Some(mode) = args.mode.clone().or_else(|| agent_cfg.mode.clone()) {
        client
            .set_mode(SessionSetModeParams {
                session_id: session_id.clone(),
                mode_id: mode.clone(),
                meta: None,
            })
            .await
            .map_err(agent_error)?;
        let mut inner = rt.inner.lock().expect("inner poisoned");
        if let Some(modes) = &mut inner.modes {
            modes.current_mode_id = mode;
        }
    }
    let mut options = agent_cfg.config.clone();
    options.extend(args.config.clone());
    for (id, value) in options {
        let value = match value {
            ConfigValue::Select(value) => aither_acp::SessionConfigValue::from(value),
            ConfigValue::Toggle(value) => aither_acp::SessionConfigValue::from(value),
        };
        let updated = client
            .set_config_option(SessionSetConfigOptionParams {
                session_id: session_id.clone(),
                config_id: id,
                value,
                meta: None,
            })
            .await
            .map_err(agent_error)?;
        rt.inner.lock().expect("inner poisoned").config_options = updated;
    }
    // Only a fresh session writes a new entry; a resumed one keeps its
    // `created` timestamp and turn count.
    if prior.is_none() {
        let entry = RegistryEntry {
            agent: args.agent.clone(),
            session_id: session_id.clone(),
            cwd: args.cwd.clone(),
            created: now(),
            last_turn: None,
            turns: 0,
            forked_from: None,
        };
        state
            .update_registry(|registry| registry.insert(args.name.clone(), entry))
            .await?;
    }
    debug!(name = %args.name, %session_id, "subagent session established");
    Ok(())
}

/// Resume the session named by `prior` (`session/load` when advertised,
/// else `session/resume`), or start a fresh `session/new`.
///
/// # Errors
///
/// Returns [`Error::NameRegistered`] when a prior session exists but the
/// agent can neither load nor resume it, or the agent error on failure.
async fn open_session(
    client: &AcpClient<SubagentHandler>,
    args: &Launch,
    init: &InitializeResult,
    prior: Option<&RegistryEntry>,
) -> Result<(String, Option<SessionModeState>, Option<Vec<ConfigOption>>)> {
    let agent_error = |source: ClientError| Error::Agent {
        agent: args.name.clone(),
        source,
    };
    let Some(entry) = prior else {
        let result = client
            .new_session(SessionNewParams {
                cwd: args.cwd.clone(),
                mcp_servers: vec![],
                additional_directories: vec![],
                meta: None,
            })
            .await
            .map_err(agent_error)?;
        return Ok((result.session_id, result.modes, result.config_options));
    };
    let (modes, config_options) = if init.agent_capabilities.load_session {
        let result = client
            .load_session(SessionLoadParams {
                session_id: entry.session_id.clone(),
                cwd: args.cwd.clone(),
                mcp_servers: vec![],
                additional_directories: vec![],
                meta: None,
            })
            .await
            .map_err(agent_error)?;
        (result.modes, result.config_options)
    } else if init
        .agent_capabilities
        .session_capabilities
        .resume
        .is_some()
    {
        let result = client
            .resume_session(SessionResumeParams {
                session_id: entry.session_id.clone(),
                cwd: args.cwd.clone(),
                mcp_servers: vec![],
                additional_directories: vec![],
                meta: None,
            })
            .await
            .map_err(agent_error)?;
        (result.modes, result.config_options)
    } else {
        return Err(Error::NameRegistered(args.name.clone()));
    };
    Ok((entry.session_id.clone(), modes, config_options))
}

/// Connection task epilogue: a dead agent marks the subagent failed unless
/// `close`/`forget` set the closing flag first.
fn on_disconnect(rt: &SubRuntime) {
    if rt.closing.load(Ordering::Relaxed) {
        return;
    }
    let mut inner = rt.inner.lock().expect("inner poisoned");
    if matches!(inner.status, Status::Failed(_)) {
        return;
    }
    inner.status = Status::Failed("agent process exited".to_string());
    inner.pending.clear();
    inner.current = None;
    drop(inner);
    rt.notify.notify_waiters();
}

/// Pump agent stderr into the tail buffer and tracing.
async fn pump_stderr(stderr: async_process::ChildStderr, rt: Arc<SubRuntime>, name: String) {
    use futures_lite::{AsyncBufReadExt, StreamExt};
    let mut lines = futures_lite::io::BufReader::new(stderr).lines();
    while let Some(line) = lines.next().await {
        match line {
            Ok(line) => {
                debug!(subagent = %name, "agent stderr: {line}");
                let mut tail = rt.stderr_tail.lock().expect("stderr tail poisoned");
                if tail.len() >= STDERR_TAIL_LINES {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
            Err(error) => {
                debug!(subagent = %name, %error, "agent stderr read failed");
                return;
            }
        }
    }
}

/// Begin a prompt turn on an established session.
///
/// The caller has already checked the status accepts a prompt. Writes the
/// `prompt` transcript record, marks the turn running, and spawns the task
/// that awaits `session/prompt` and records the turn's end.
///
/// # Errors
///
/// Returns [`Error::NotPromptable`] when the subagent is failed, or an I/O
/// error if the transcript record cannot be written.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub async fn start_turn(
    state: Arc<AppState>,
    sub: &Arc<Subagent>,
    prompt: String,
) -> Result<SendOutcome> {
    enum Action {
        Start(u64),
        Queued(usize),
        Reject(String),
    }
    let action = {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        let action = match inner.status {
            // A turn in flight (`current` set covers `running` and
            // `needs_permission`), or a non-empty queue in the `done` drain
            // gap: the live turn task drains in FIFO order, so a direct
            // start would overtake prompts parked earlier.
            _ if inner.current.is_some() || !inner.queued.is_empty() => {
                inner.queued.push_back(prompt.clone());
                Action::Queued(inner.queued.len())
            }
            _ if !inner.status.accepts_prompt() => Action::Reject(inner.status.name().to_string()),
            _ => {
                let n = inner.turn_offset + inner.turns.len() as u64 + 1;
                inner.current = Some(Turn {
                    n,
                    ..Turn::default()
                });
                inner.status = Status::Running;
                inner.prompt_sent = false;
                Action::Start(n)
            }
        };
        drop(inner);
        action
    };
    let turn_n = match action {
        Action::Queued(position) => return Ok(SendOutcome::Queued(position)),
        Action::Reject(status) => {
            return Err(Error::NotPromptable {
                name: sub.name.clone(),
                status,
            });
        }
        Action::Start(n) => n,
    };
    sub.rt
        .transcript
        .lock()
        .await
        .append(&serde_json::json!({"ts": now(), "turn": turn_n, "prompt": prompt}))
        .await
        .map_err(|source| Error::io("cannot write transcript", source))?;
    sub.rt.notify.notify_waiters();

    // Push the prompt request now, before this call returns: a `session/
    // cancel` or a `send` decision made after `Running` was observed must
    // never reach the wire ahead of the prompt it acts on.
    let session_id = sub
        .rt
        .inner
        .lock()
        .expect("inner poisoned")
        .session_id
        .clone();
    let turn = match sub
        .client
        .start_prompt(&PromptParams {
            session_id: session_id.clone(),
            prompt: vec![ContentBlock::Text(TextContent {
                text: prompt,
                annotations: None,
                meta: None,
            })],
            meta: None,
        })
        .await
    {
        Ok(turn) => turn,
        Err(error) => {
            let mut inner = sub.rt.inner.lock().expect("inner poisoned");
            if let Some(turn) = inner.current.take() {
                inner.turns.push(turn);
            }
            inner.status = Status::Failed(error.to_string());
            drop(inner);
            sub.rt.notify.notify_waiters();
            return Err(Error::Agent {
                agent: sub.name.clone(),
                source: error,
            });
        }
    };
    sub.rt.inner.lock().expect("inner poisoned").prompt_sent = true;
    sub.rt.notify.notify_waiters();

    let client = sub.client.clone();
    let rt = sub.rt.clone();
    let name = sub.name.clone();
    tokio::spawn(turn_task(state, client, session_id, turn, turn_n, rt, name));
    Ok(SendOutcome::Running)
}

/// Await the in-flight `session/prompt`, record the turn's end, then keep
/// looping while a completed turn leaves queued prompts behind: each queued
/// prompt becomes the next turn in this same task, so no two turns ever run
/// concurrently.
async fn turn_task(
    state: Arc<AppState>,
    client: AcpClient<SubagentHandler>,
    session_id: String,
    mut turn: ResponseFuture<PromptResult>,
    mut turn_n: u64,
    rt: Arc<SubRuntime>,
    name: String,
) {
    loop {
        let outcome = (&mut turn).await;
        settle_turn(&rt, turn_n, outcome).await;
        // Update the registry: turn count and last_turn.
        let turns = {
            let inner = rt.inner.lock().expect("inner poisoned");
            inner.turn_offset + inner.turns.len() as u64
        };
        let result = state
            .update_registry(|registry| {
                if let Some(entry) = registry.get_mut(&name) {
                    entry.turns = turns;
                    entry.last_turn = Some(now());
                }
            })
            .await;
        if let Err(error) = result {
            warn!(%error, "registry persist failed");
        }
        rt.notify.notify_waiters();
        // A completed turn drains the queue into the next turn inside this
        // task; cancelled or failed turns drop it — the lead re-sends what it
        // still wants. When a `send` slipped into the Done window and already
        // started its own turn (`current.is_some()`), that task drains the
        // queue at its own end, so this one leaves it alone.
        let next = {
            let mut inner = rt.inner.lock().expect("inner poisoned");
            match inner.status {
                Status::Done(_) if inner.current.is_none() => {
                    inner.queued.pop_front().map(|queued| {
                        let n = inner.turn_offset + inner.turns.len() as u64 + 1;
                        inner.current = Some(Turn {
                            n,
                            ..Turn::default()
                        });
                        inner.status = Status::Running;
                        inner.prompt_sent = false;
                        (queued, n)
                    })
                }
                // Cancelled or failed: stale briefs are dropped; the lead
                // re-sends what it still wants. `running`/`needs_permission`
                // here means a `send` in the `done` window already started
                // the next turn — its own task drains the queue at its end.
                Status::Cancelled | Status::Failed(_) => {
                    inner.queued.clear();
                    None
                }
                _ => None,
            }
        };
        let Some((queued, n)) = next else {
            break;
        };
        if let Err(error) = rt
            .transcript
            .lock()
            .await
            .append(&serde_json::json!({"ts": now(), "turn": n, "prompt": queued}))
            .await
        {
            warn!(%error, "transcript write failed");
        }
        rt.notify.notify_waiters();
        // Same ordering rule as `start_turn`: the prompt request is pushed
        // inside this task before the next iteration, so a `cancel` that saw
        // `Running` cannot overtake it on the wire.
        match client
            .start_prompt(&PromptParams {
                session_id: session_id.clone(),
                prompt: vec![ContentBlock::Text(TextContent {
                    text: queued,
                    annotations: None,
                    meta: None,
                })],
                meta: None,
            })
            .await
        {
            Ok(next_turn) => {
                rt.inner.lock().expect("inner poisoned").prompt_sent = true;
                rt.notify.notify_waiters();
                turn = next_turn;
                turn_n = n;
            }
            Err(error) => {
                let reason = error.to_string();
                let record = serde_json::json!({
                    "ts": now(),
                    "turn": n,
                    "stop_reason": "error",
                    "error": reason,
                });
                if let Err(error) = rt.transcript.lock().await.append(&record).await {
                    warn!(%error, "transcript write failed");
                }
                let mut inner = rt.inner.lock().expect("inner poisoned");
                if let Some(turn) = inner.current.take() {
                    inner.turns.push(turn);
                }
                inner.status = Status::Failed(reason);
                inner.queued.clear();
                drop(inner);
                rt.notify.notify_waiters();
                break;
            }
        }
    }
}

/// Record a finished turn: transcript end record, the completed `Turn`, and
/// the terminal status (`done`, `cancelled`, or `failed`).
async fn settle_turn(
    rt: &SubRuntime,
    turn_n: u64,
    outcome: std::result::Result<PromptResult, ClientError>,
) {
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
            let mut inner = rt.inner.lock().expect("inner poisoned");
            if let Some(mut turn) = inner.current.take() {
                turn.stop_reason = Some(reason);
                inner.turns.push(turn);
            }
            inner.status = if reason == StopReason::Cancelled {
                Status::Cancelled
            } else {
                Status::Done(reason)
            };
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
            let mut inner = rt.inner.lock().expect("inner poisoned");
            if let Some(turn) = inner.current.take() {
                inner.turns.push(turn);
            }
            inner.status = Status::Failed(reason);
        }
    }
}

/// What a `send` did with the prompt.
#[derive(Debug)]
pub enum SendOutcome {
    /// The prompt started a turn immediately.
    Running,
    /// The prompt was parked behind the running turn; carries its 1-based
    /// queue position.
    Queued(usize),
}

/// Send a follow-up prompt: a new turn on the same session.
///
/// When the subagent is running or awaiting a permission answer, the prompt
/// is parked and fires in order once the turn completes; a cancelled or
/// failed turn drops the queue.
///
/// # Errors
///
/// Returns [`Error::UnknownSubagent`] when the name is not live, or
/// [`Error::NotPromptable`] when the subagent failed.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub async fn send(state: &Arc<AppState>, name: &str, prompt: String) -> Result<SendOutcome> {
    let sub = state.get(name)?;
    start_turn(state.clone(), &sub, prompt).await
}

/// Arguments for [`fork`], as given by the `fork` tool.
#[derive(Debug)]
pub struct ForkArgs {
    /// Source subagent name; must be live.
    pub name: String,
    /// Name for the forked subagent; defaults to `<name>-fork` (with `-N`
    /// suffixes on collision).
    pub new_name: Option<String>,
    /// First turn for the fork, sent once it is live.
    pub prompt: Option<String>,
    /// 1-based history step to fork from; defaults to the latest.
    pub step: Option<u64>,
}

/// Fork a live subagent's session: clone its history into a new session,
/// attach it to a fresh agent process, and register it under a new name.
///
/// The fork keeps the source's agent, cwd, and permission policy. The
/// source is untouched and stays usable.
///
/// Fork strategy, by advertised capability: `sessionCapabilities.fork`
/// (`session/fork`) first, then devin's `_cognition.ai/revert/forkFromStep`.
/// Neither present is a [`Error::ForkUnsupported`] error — acpsub does not
/// fake a fork by respawning.
///
/// # Errors
///
/// Returns [`Error::UnknownSubagent`] when the source is not live,
/// [`Error::ForkUnsupported`] when the agent cannot fork,
/// [`Error::ForkStep`] when `step` names no forkable step, or the usual
/// launch errors.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub async fn fork(state: &Arc<AppState>, args: ForkArgs) -> Result<Arc<Subagent>> {
    let source = state.get(&args.name)?;
    let new_name = match &args.new_name {
        Some(name) => {
            validate_name(name)?;
            name.clone()
        }
        None => args.name.clone() + "-fork",
    };
    // A forked session id, minted on the source's live connection.
    let forked_id = mint_fork(&source, args.step).await?;
    fork_launch(state, &source, new_name, forked_id, args.prompt).await
}

/// Run the source agent's fork method and return the new session id.
async fn mint_fork(source: &Subagent, step: Option<u64>) -> Result<String> {
    let (session_id, caps) = {
        let inner = source.rt.inner.lock().expect("inner poisoned");
        (inner.session_id.clone(), inner.agent_capabilities.clone())
    };
    let agent_error = |source_err: ClientError| Error::Agent {
        agent: source.name.clone(),
        source: source_err,
    };
    if caps.session_capabilities.extra.contains_key("fork") {
        let result: serde_json::Value = source
            .client
            .request(
                "session/fork",
                &serde_json::json!({
                    "sessionId": session_id,
                    "cwd": source.cwd,
                    "mcpServers": [],
                    "additionalDirectories": [],
                }),
            )
            .await
            .map_err(agent_error)?;
        return result
            .get("sessionId")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| Error::ForkUnsupported {
                name: source.name.clone(),
                agent: source.agent.clone(),
            });
    }
    if aither_acp::vendor::devin::supports(&caps, aither_acp::vendor::devin::capability::REVERT) {
        return mint_fork_devin(source, &session_id, step).await;
    }
    Err(Error::ForkUnsupported {
        name: source.name.clone(),
        agent: source.agent.clone(),
    })
}

/// Devin path: resolve `step` to a `forkTargetNodeId` via
/// `revert/listSteps`, then `revert/forkFromStep` mints the session.
async fn mint_fork_devin(source: &Subagent, session_id: &str, step: Option<u64>) -> Result<String> {
    use aither_acp::vendor::devin::revert;
    let agent_error = |source_err: ClientError| Error::Agent {
        agent: source.name.clone(),
        source: source_err,
    };
    let steps: revert::ListStepsResult = source
        .client
        .request(
            revert::LIST_STEPS_METHOD,
            &revert::ListStepsParams {
                session_id: session_id.to_string(),
            },
        )
        .await
        .map_err(agent_error)?;
    let target = match step {
        Some(n) => steps.steps.iter().find(|s| s.step_number == n),
        None => steps
            .steps
            .iter()
            .rev()
            .find(|s| s.fork_target_node_id.is_some()),
    }
    .ok_or_else(|| Error::ForkStep {
        name: source.name.clone(),
        step: step.unwrap_or_else(|| steps.steps.last().map_or(0, |s| s.step_number)),
        have: steps.steps.len() as u64,
    })?;
    let node = target.fork_target_node_id.ok_or_else(|| Error::ForkStep {
        name: source.name.clone(),
        step: target.step_number,
        have: steps.steps.len() as u64,
    })?;
    let result: revert::ForkFromStepResult = source
        .client
        .request(
            revert::FORK_FROM_STEP_METHOD,
            &revert::ForkFromStepParams {
                session_id: session_id.to_string(),
                target_node_id: node,
            },
        )
        .await
        .map_err(agent_error)?;
    Ok(result.forked_session_id)
}

/// Launch the forked session on its own process, mirroring `launch`.
async fn fork_launch(
    state: &Arc<AppState>,
    source: &Subagent,
    name: String,
    forked_id: String,
    prompt: Option<String>,
) -> Result<Arc<Subagent>> {
    // Reserve the name; collide like spawn does.
    {
        let live = state.live.lock().expect("live poisoned");
        let mut reserved = state.reserved.lock().expect("reserved poisoned");
        if live.contains_key(&name) || !reserved.insert(name.clone()) {
            return Err(Error::NameTaken(name));
        }
    }
    let result = fork_launch_reserved(state, source, &name, &forked_id, prompt).await;
    if result.is_err() {
        state
            .reserved
            .lock()
            .expect("reserved poisoned")
            .remove(&name);
    }
    result
}

/// `fork_launch` once the name is reserved.
async fn fork_launch_reserved(
    state: &Arc<AppState>,
    source: &Subagent,
    name: &str,
    forked_id: &str,
    prompt: Option<String>,
) -> Result<Arc<Subagent>> {
    // The name must also be free in the registry: a registered name means a
    // resumable session already exists under it.
    if state
        .registry
        .lock()
        .expect("registry poisoned")
        .get(name)
        .is_some()
    {
        return Err(Error::NameRegistered(name.to_string()));
    }
    let agent_cfg = state.config.agent(&source.agent)?.clone();
    let source_turns = {
        let inner = source.rt.inner.lock().expect("inner poisoned");
        inner.turn_offset + inner.turns.len() as u64
    };
    // A synthetic prior entry routes the handshake through `session/load`
    // (or `session/resume`) on the forked id, and carries the lineage into
    // the registry insert below.
    let prior = RegistryEntry {
        agent: source.agent.clone(),
        session_id: forked_id.to_string(),
        cwd: source.cwd.clone(),
        created: now(),
        last_turn: None,
        turns: source_turns,
        forked_from: Some(source.name.clone()),
    };
    let launch_args = Launch {
        name: name.to_string(),
        agent: source.agent.clone(),
        cwd: source.cwd.clone(),
        prompt: String::new(),
        mode: None,
        config: BTreeMap::new(),
        permission: Some(source.permission),
        replace: false,
    };
    let sub = wire_subagent(
        state,
        &launch_args,
        &agent_cfg,
        source.cwd.clone(),
        source.permission,
        Some(prior.clone()),
    )
    .await?;
    // The handshake skips the registry write for a prior entry, so the
    // fork's lineage entry goes in once the session is established.
    state
        .update_registry(|registry| registry.insert(name.to_string(), prior))
        .await?;
    if let Some(prompt) = prompt
        && let Err(error) = start_turn(state.clone(), &sub, prompt).await
    {
        state.live.lock().expect("live poisoned").remove(name);
        return Err(error);
    }
    Ok(sub)
}
