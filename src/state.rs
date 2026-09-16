//! The subagent model: per-subagent runtime state, the shared app state, and
//! the spawn/turn machinery.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aither_acp::{
    AcpClient, ClientError, ConfigOption, ContentBlock, Implementation, PlanEntry, PromptResult,
    RequestPermissionOutcome, SessionModeState, StopReason, TextContent, ToolCall,
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
    /// The transcript file path.
    pub transcript_path: PathBuf,
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
    /// `session/set_mode` override; falls back to the agent's configured mode.
    pub mode: Option<String>,
    /// `session/set_config_option` overrides merged over the agent's `config`.
    pub config: BTreeMap<String, ConfigValue>,
    /// Permission policy override.
    pub permission: Option<PermissionPolicy>,
    /// `adopt`: the existing session id to load instead of `session/new`.
    pub load: Option<String>,
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
pub async fn launch(state: &Arc<AppState>, args: Launch) -> Result<Arc<Subagent>> {
    let agent_cfg = state.config.agent(&args.agent)?.clone();
    let cwd = args.cwd.canonicalize().map_err(|source| {
        Error::io(format!("cannot resolve cwd {}", args.cwd.display()), source)
    })?;
    // Claim the session id atomically across the live and in-flight maps:
    // the MCP server runs `tools/call`s concurrently, so two `adopt`s of one
    // session could interleave between this check and `live.insert` below.
    if let Some(session_id) = &args.load {
        validate_session_id(session_id)?;
        let live = state.live.lock().expect("live poisoned");
        let mut reserved = state.reserved.lock().expect("reserved poisoned");
        if live.contains_key(session_id) || !reserved.insert(session_id.clone()) {
            return Err(Error::SessionLive(session_id.clone()));
        }
    }
    let result = launch_inner(state, &args, agent_cfg, cwd).await;
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
    args: &Launch,
    agent_cfg: AgentConfig,
    cwd: PathBuf,
) -> Result<Arc<Subagent>> {
    let permission = args
        .permission
        .or(agent_cfg.permission)
        .unwrap_or(state.config.defaults.permission);
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
            pending: Vec::new(),
        }),
        notify: Notify::new(),
        transcript: tokio::sync::Mutex::new(TranscriptWriter::deferred()),
        terminals: Terminals::new(HashMap::new()),
        stderr_tail: Mutex::new(VecDeque::new()),
        next_perm: AtomicU64::new(1),
        next_term: AtomicU64::new(1),
        closing: AtomicBool::new(false),
    });

    let handler = SubagentHandler::new(
        rt.clone(),
        cwd.clone(),
        permission,
        agent_cfg.allow_outside_cwd,
    );
    let (client, conn, stderr_pump) = connect(&agent_cfg, &cwd, handler, rt.clone())?;

    // On any handshake failure, kill the child before returning the error.
    let session_id = match handshake(state, &client, &rt, args, &agent_cfg).await {
        Ok(session_id) => session_id,
        Err(error) => {
            rt.closing.store(true, Ordering::Relaxed);
            client.close();
            let _ = conn.await;
            return Err(error);
        }
    };

    let transcript_path = state
        .config
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

/// `initialize`, `session/new` or `session/load`, `set_mode`, and the
/// `set_config_option` calls; upserts the registry entry. Returns the
/// session id (agent-assigned for `session/new`, the adopted id for
/// `session/load`).
async fn handshake(
    state: &Arc<AppState>,
    client: &AcpClient<SubagentHandler>,
    rt: &SubRuntime,
    args: &Launch,
    agent_cfg: &AgentConfig,
) -> Result<String> {
    let agent_error = |source: ClientError| Error::Agent {
        agent: args.agent.clone(),
        source,
    };
    let init = client.initialize().await.map_err(agent_error)?;
    let (session_id, modes, config_options) = if let Some(load) = &args.load {
        if !init.agent_capabilities.load_session {
            return Err(Error::LoadUnsupported(args.agent.clone()));
        }
        let result = client
            .load_session(load, args.cwd.clone(), vec![])
            .await
            .map_err(agent_error)?;
        (load.clone(), result.modes, result.config_options)
    } else {
        let result = client
            .new_session(args.cwd.clone(), vec![])
            .await
            .map_err(agent_error)?;
        (result.session_id, result.modes, result.config_options)
    };
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
        inner.config_options = config_options.unwrap_or_default();
        inner.turn_offset = turn_offset;
    }
    if let Some(mode) = args.mode.clone().or_else(|| agent_cfg.mode.clone()) {
        client
            .set_mode(&session_id, &mode)
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
            .set_config_option(&session_id, &id, value)
            .await
            .map_err(agent_error)?;
        rt.inner.lock().expect("inner poisoned").config_options = updated;
    }
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
                    turns: 0,
                },
            ),
        })
        .await?;
    debug!(%session_id, agent = %args.agent, "subagent session established");
    Ok(session_id)
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

/// Begin a prompt turn on an established session.
///
/// The caller has already checked the status accepts a prompt. Writes the
/// `prompt` transcript record, marks the turn running, and spawns the task
/// that awaits `session/prompt` and records the turn's end.
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
    let turn_n = {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        if !inner.status.accepts_prompt() {
            return Err(Error::NotPromptable {
                session_id: sub.session_id.clone(),
                status: inner.status.name().to_string(),
            });
        }
        let n = inner.turn_offset + inner.turns.len() as u64 + 1;
        inner.current = Some(Turn {
            n,
            ..Turn::default()
        });
        inner.status = Status::Running;
        drop(inner);
        n
    };
    sub.rt
        .transcript
        .lock()
        .await
        .append(&serde_json::json!({"ts": now(), "turn": turn_n, "prompt": prompt}))
        .await
        .map_err(|source| Error::io("cannot write transcript", source))?;
    sub.rt.notify.notify_waiters();

    let client = sub.client.clone();
    let session_id = sub.session_id.clone();
    let rt = sub.rt.clone();
    let task_session_id = session_id.clone();
    let mut prompt: PromptFut = Box::pin(async move {
        client
            .prompt(
                &session_id,
                vec![ContentBlock::Text(TextContent {
                    text: prompt,
                    annotations: None,
                })],
            )
            .await
    });
    // Drive the prompt future once before returning: `AcpClient` enqueues
    // the request on its unbounded outbound channel inside the first poll,
    // so `session/prompt` is on the wire ahead of anything a later tool
    // call sends — a `cancel` issued right after `spawn`/`send` cannot
    // overtake the prompt it is meant to stop.
    let ready = futures_lite::future::poll_fn(|cx| match prompt.as_mut().poll(cx) {
        std::task::Poll::Ready(outcome) => std::task::Poll::Ready(Some(outcome)),
        std::task::Poll::Pending => std::task::Poll::Ready(None),
    })
    .await;
    // A synchronously-failed send resolves on the first poll; wrap the
    // outcome so the same turn-end path handles it.
    let prompt: PromptFut = ready.map_or(prompt, |outcome| Box::pin(async move { outcome }));
    tokio::spawn(turn_task(state, prompt, turn_n, rt, task_session_id));
    Ok(())
}

/// The in-flight `session/prompt` call, already enqueued by `start_turn`.
type PromptFut =
    Pin<Box<dyn Future<Output = std::result::Result<PromptResult, ClientError>> + Send>>;

/// Await `session/prompt`, record the turn's end in the transcript, state,
/// and registry, then wake waiters.
async fn turn_task(
    state: Arc<AppState>,
    prompt: PromptFut,
    turn_n: u64,
    rt: Arc<SubRuntime>,
    session_id: String,
) {
    let outcome = prompt.await;
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
