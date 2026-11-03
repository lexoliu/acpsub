//! The typed session API for a host process embedding acpsub.
//!
//! A [`Host`] runs ACP agents as sessions without going through the MCP
//! tool layer: [`Host::spawn`] starts an agent command and opens a
//! session, [`Session::prompt`] runs one turn to its end,
//! [`Session::cancel`] stops it, [`Session::close`] ends the session, and
//! [`probe`] checks that a command speaks ACP. The tools and the CLI run
//! on the same machinery, so a session spawned here is visible to them
//! and the other way round.
//!
//! The host owns where things live through its [`Config`]: `transcript_dir`
//! and `registry` point at the host's own storage, and the
//! [`ConfigSource`] it hands the [`AppState`] supplies that config on
//! every spawn.
//!
//! Dropping a [`Session`] does not close it: closing is asynchronous and
//! fallible, and a session stays resumable (`adopt`) after it, so ending
//! one silently from `Drop` would hide both. A host calls
//! [`Session::close`].

use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use aither_acp::{InitializeResult, StopReason};
use tokio::task::JoinHandle;

use crate::config::{AgentConfig, Config, ConfigSource, ConfigValue, PermissionPolicy};
use crate::error::{Error, Result};
use crate::handler::SubagentHandler;
use crate::state::{AppState, Launch, Subagent, connect, launch, new_runtime, start_turn};
use crate::tools::{
    WaitBound, WaitOutcome, WaitReporter, awaited_turn, cancel_turn, close_session, turn_outcome,
    wait_for_turn,
};
use crate::watchdog::Waiter;

/// What a session is spawned with.
#[derive(Debug, Clone)]
pub struct SessionSpec {
    /// Configured agent key; `None` falls back to `[defaults] agent`, then
    /// to the only configured agent.
    pub agent: Option<String>,
    /// Working directory of the session.
    pub cwd: PathBuf,
    /// The model to run. Required when the agent advertises a `model`
    /// config option, an error when it advertises none, and it must be one
    /// the agent advertises.
    pub model: Option<String>,
    /// The session mode to activate, under the same rules as `model`
    /// against the agent's advertised modes.
    pub mode: Option<String>,
    /// Extra `session/set_config_option` values applied after `model`;
    /// every id and value must be one the agent advertises.
    pub config: BTreeMap<String, ConfigValue>,
    /// Permission policy for this session; the agent's, then the
    /// defaults' apply when `None`. A host drives turns with
    /// [`Session::prompt`], which has no way to answer a request the
    /// `ask` policy queues, so it picks `allow` or `deny` here.
    pub permission: Option<PermissionPolicy>,
}

impl SessionSpec {
    /// A spec for the agent configured as `agent`, running in `cwd`, with
    /// no model, mode, extra options or permission override.
    #[must_use]
    pub fn new(agent: impl Into<String>, cwd: impl Into<PathBuf>) -> Self {
        Self {
            agent: Some(agent.into()),
            cwd: cwd.into(),
            model: None,
            mode: None,
            config: BTreeMap::new(),
            permission: None,
        }
    }
}

/// Start the session `spec` describes, with `prompt` as its first turn
/// when given.
///
/// Shared by the `spawn` tool and [`Host::spawn`].
///
/// # Errors
///
/// The config source's error, [`Error::UnknownAgent`] or
/// [`Error::AgentUnspecified`] for an agent that cannot be resolved, the
/// quota refusal of a rate-limited scope, and everything [`launch`]
/// reports.
pub(crate) async fn spawn_session(
    state: &Arc<AppState>,
    spec: SessionSpec,
    prompt: Option<String>,
) -> Result<Arc<Subagent>> {
    let config = state.load_config().await?;
    let (agent, _) = config.resolve_agent(spec.agent.as_deref())?;
    let agent = agent.clone();
    // A limited quota scope refuses the prompt before any process
    // starts: it would burn against a limit the daemon already knows.
    crate::ratelimit::refuse_if_limited(state, &agent)?;
    launch(
        state,
        &config,
        Launch {
            agent,
            cwd: spec.cwd,
            prompt,
            mode: spec.mode,
            model: spec.model,
            config: spec.config,
            permission: spec.permission,
            load: None,
        },
    )
    .await
}

/// A host's entry point: spawns sessions and probes agent commands.
#[derive(Debug, Clone)]
pub struct Host {
    state: Arc<AppState>,
}

impl Host {
    /// A host over an existing app state.
    #[must_use]
    pub const fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }

    /// A host that builds its own app state from `config` — which fixes
    /// the transcript directory, the registry file and the watchdog
    /// grace — and re-reads `source` on every spawn.
    ///
    /// # Errors
    ///
    /// Returns an error if the registry file exists but cannot be parsed.
    pub fn from_config(config: Config, source: impl ConfigSource + 'static) -> Result<Self> {
        AppState::new(config, source).map(Self::new)
    }

    /// The app state under this host, for building the MCP tools over the
    /// same sessions.
    #[must_use]
    pub const fn state(&self) -> &Arc<AppState> {
        &self.state
    }

    /// Start an agent process, open a session in `spec.cwd` and return its
    /// handle. No turn starts: [`Session::prompt`] runs the first one.
    ///
    /// # Errors
    ///
    /// Exactly the errors `spawn` reports: [`Error::UnknownAgent`],
    /// [`Error::SpawnFailed`], [`Error::Agent`] for a failed handshake, and
    /// the model and mode checks — a value the agent advertises but the
    /// spec omits, or one the spec gives but the agent does not advertise,
    /// is an error. The process is killed on any error.
    pub fn spawn(&self, spec: SessionSpec) -> impl Future<Output = Result<Session>> + Send {
        let state = self.state.clone();
        async move {
            let sub = spawn_session(&state, spec, None).await?;
            Ok(Session { state, sub })
        }
    }
}

/// How a turn ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnEnd {
    /// The agent answered the prompt with a stop reason —
    /// [`StopReason::Cancelled`] after [`Session::cancel`] or a close.
    Stopped(StopReason),
    /// The turn errored while the agent process stays alive; the session
    /// takes another prompt.
    Failed(String),
    /// The agent process exited.
    Exited(String),
    /// The agent's request for permission waits for an answer the
    /// `ask` policy leaves to the `permit` tool.
    NeedsPermission {
        /// The request's id, as `permit` takes it.
        request_id: String,
        /// The tool call the agent wants to run.
        title: String,
    },
    /// The provider rate-limited the turn: the session parks until
    /// `resume_at`, when the daemon-side scheduler resumes it.
    RateLimited {
        /// When the provider says the quota resets.
        resume_at: jiff::Timestamp,
        /// The provider's own message.
        reason: String,
    },
}

/// What a finished turn reports: the data `wait` reports, typed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnOutcome {
    /// The turn's number within the session.
    pub turn: u64,
    /// The turn's final assistant message: the concatenated message
    /// chunks it streamed.
    pub reply: String,
    /// How the turn ended.
    pub end: TurnEnd,
}

/// One running agent session.
///
/// Dropping the handle leaves the session open; call
/// [`close`](Self::close) to end it.
#[derive(Debug)]
#[must_use = "a session stays open until `close` is called"]
pub struct Session {
    state: Arc<AppState>,
    sub: Arc<Subagent>,
}

impl Session {
    /// The agent-assigned session id: the handle every tool addresses.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.sub.session_id
    }

    /// The model the agent accepted — a string for a select option — or
    /// `None` when the agent advertises no `model` option.
    #[must_use]
    pub fn model(&self) -> Option<&serde_json::Value> {
        self.sub.model.as_ref()
    }

    /// The mode the agent accepted, or `None` when it advertises none.
    #[must_use]
    pub fn mode(&self) -> Option<&str> {
        self.sub.mode.as_deref()
    }

    /// Start a turn with `text` and resolve when it ends.
    ///
    /// While the returned future is pending it counts as an attached
    /// `wait` for the turn watchdog, so a turn a host awaits is never
    /// cancelled for lack of a waiter. Dropping the future stops
    /// waiting, not the turn: [`cancel`](Self::cancel) stops it.
    ///
    /// # Errors
    ///
    /// Returns the quota refusal of a rate-limited scope,
    /// [`Error::NotPromptable`] when the session does not accept a prompt
    /// (a turn is already running, or the agent exited), and an I/O error
    /// when the transcript cannot be written.
    pub fn prompt(
        &self,
        text: impl Into<String>,
    ) -> impl Future<Output = Result<TurnOutcome>> + Send {
        let state = self.state.clone();
        let sub = self.sub.clone();
        let text = text.into();
        async move {
            crate::ratelimit::refuse_if_limited(&state, &sub.agent)?;
            // Attached before the turn starts: the watchdog never sees
            // the turn unwatched.
            let _waiter = Waiter::attach(&sub);
            let started = Instant::now();
            start_turn(state, &sub, text).await?;
            let awaited = awaited_turn(&sub)?;
            let outcome = match awaited {
                Some(awaited) => {
                    let bound = WaitBound {
                        expect: f64::INFINITY,
                        deadline: None,
                    };
                    wait_for_turn(&sub, awaited, started, &bound, &mut Silent).await?
                }
                None => WaitOutcome::Ended,
            };
            turn_outcome(&sub, awaited, outcome)
        }
    }

    /// Stop the turn in flight; the pending [`prompt`](Self::prompt)
    /// resolves with [`StopReason::Cancelled`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotRunning`] when no turn is in flight and
    /// [`Error::Agent`] when the agent rejects the cancel.
    pub fn cancel(&self) -> impl Future<Output = Result<()>> + Send {
        let sub = self.sub.clone();
        async move { cancel_turn(&sub).await }
    }

    /// End the session: tear the agent process down. The registry entry
    /// stays, so the session can be adopted again.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownSession`] when the session was already
    /// closed or forgotten.
    pub async fn close(self) -> Result<()> {
        close_session(&self.state, &self.sub.session_id)
            .await
            .map(|_| ())
    }
}

/// The reporter of a wait nobody listens to.
struct Silent;

impl WaitReporter for Silent {
    type Error = Error;

    fn interval(&self) -> Option<std::time::Duration> {
        None
    }

    fn report(
        &mut self,
        _: &Subagent,
        _: Instant,
        _: f64,
    ) -> impl Future<Output = Result<()>> + Send {
        std::future::ready(Ok(()))
    }
}

/// Start `agent`'s command, send it ACP `initialize` and return what it
/// answers — its name, version and capabilities — then kill it. This is
/// how a host checks that a command speaks ACP.
///
/// The command runs in the current directory. A command that exits or
/// answers something other than ACP fails the probe; one that stays
/// silent leaves it pending, so a host bounds the call with its own
/// timeout. Dropping the future kills the process.
///
/// # Errors
///
/// Returns [`Error::SpawnFailed`] when the process cannot start and
/// [`Error::Agent`] when it does not answer `initialize`.
pub fn probe(agent: &AgentConfig) -> impl Future<Output = Result<InitializeResult>> + Send {
    let agent = agent.clone();
    async move {
        let cwd = std::env::current_dir()
            .map_err(|source| Error::io("cannot resolve the current directory", source))?;
        let rt = new_runtime(String::new(), agent.steer);
        let handler = SubagentHandler::new(rt.clone(), cwd.clone(), PermissionPolicy::Deny, false);
        let (client, conn, pump) = connect(&agent, &cwd, handler, rt.clone())?;
        let mut process = Probed {
            rt,
            client: client.clone(),
            conn: Some(conn),
            pump,
        };
        let answer = client.initialize().await.map_err(|source| Error::Agent {
            agent: agent.command.clone(),
            source,
        });
        process.finish().await;
        answer
    }
}

/// The process a [`probe`] started: killed when the probe ends or its
/// future is dropped mid-flight.
struct Probed {
    rt: Arc<crate::state::SubRuntime>,
    client: aither_acp::AcpClient<SubagentHandler>,
    conn: Option<JoinHandle<()>>,
    pump: JoinHandle<()>,
}

impl Probed {
    /// Kill the process tree and take the connection down.
    fn kill(&self) {
        self.rt.closing.set(crate::state::Closing::Ending);
        self.rt.kill_process_tree();
        self.client.clone().close();
        self.pump.abort();
    }

    /// Kill, then wait for the connection task to end.
    async fn finish(&mut self) {
        self.kill();
        if let Some(conn) = self.conn.take() {
            let _ = conn.await;
        }
    }
}

impl Drop for Probed {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.kill();
            conn.abort();
        }
    }
}
