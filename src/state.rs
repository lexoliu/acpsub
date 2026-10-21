//! The subagent model: per-subagent runtime state, the shared app state, and
//! the spawn/turn machinery.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque, hash_map::Entry};
use std::future::Future;
#[cfg(unix)]
use std::os::unix::process::CommandExt as _;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_acp::{
    AcpClient, ClientError, ConfigOption, ConfigOptionValue, ConfigSelectOptions, ContentBlock,
    Implementation, InitializeResult, PlanEntry, PromptParams, PromptResult,
    RequestPermissionOutcome, SessionLoadParams, SessionModeState, SessionNewParams,
    SessionSetConfigOptionParams, SessionSetModeParams, SessionUpdate, StopReason, TextContent,
    ToolCall, ToolCallLocation, ToolCallStatus, ToolKind,
};
use aither_mcp::transport::ChildProcessTransport;
use serde::Serialize;
use tokio::sync::{Notify, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::config::{AgentConfig, Config, ConfigValue, PermissionPolicy, SteerSemantics};
use crate::error::{Error, Result};
use crate::handler::{SubagentHandler, note_tool_call, note_tool_call_update};
use crate::registry::{LastTurn, Registry, RegistryEntry, persist};
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
    /// Session ids claimed by an in-flight `session/load` launch — an
    /// `adopt` or a daemon resume — before the subagent is live. A
    /// `close`/`forget` landing while the id is held marks the
    /// reservation; the launch ends the session instead of returning it
    /// live. Locked after `live` wherever both are held.
    pub reserved: Mutex<HashMap<String, Reservation>>,
    /// The persisted session id → session registry.
    pub registry: Mutex<Registry>,
    /// Serializes registry mutations with their `persist` so the file can
    /// never be overwritten by an older snapshot out of order.
    pub registry_write: tokio::sync::Mutex<()>,
    /// Path of the registry file (`config.defaults.registry`).
    pub registry_path: PathBuf,
    /// Provider rate limits in effect, by the quota scope they bind:
    /// while a scope's `resume_at` is still ahead, new prompts to every
    /// session in it park instead of burning a request that is bound to
    /// fail.
    pub limits: Mutex<BTreeMap<crate::ratelimit::QuotaScope, crate::ratelimit::Limit>>,
    /// Wakes the resume scheduler when a limit is registered or a session
    /// parks.
    pub limits_notify: Notify,
    /// Set once the resume scheduler is running.
    pub scheduler_started: AtomicBool,
    /// How long a running turn may go without an attached `wait`/
    /// `wait_any` caller before the watchdog cancels it and closes the
    /// session — [`crate::watchdog::WAIT_GRACE`] unless the config sets
    /// `[defaults] wait_grace`.
    pub wait_grace: Duration,
    /// Daemon-generation handover coordination.
    pub handover: Handover,
    /// What this daemon's startup resume pass did — filled by
    /// [`crate::daemon::run`] before it accepts clients; empty everywhere
    /// else.
    pub resume_report: Mutex<ResumeReport>,
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
        let registry = Registry::load(&registry_path)?;
        // Quota gates survive a restart through the registry's parked
        // flags: a prompt against a still-limited scope keeps parking.
        let limits = crate::ratelimit::restored_limits(&registry);
        let state = Arc::new(Self {
            config_path,
            transcript_dir: config.defaults.transcript_dir,
            live: Mutex::new(HashMap::new()),
            reserved: Mutex::new(HashMap::new()),
            registry: Mutex::new(registry),
            registry_write: tokio::sync::Mutex::new(()),
            registry_path,
            limits: Mutex::new(limits),
            limits_notify: Notify::new(),
            scheduler_started: AtomicBool::new(false),
            wait_grace: config.defaults.wait_grace,
            handover: Handover::default(),
            resume_report: Mutex::new(ResumeReport::default()),
        });
        if tokio::runtime::Handle::try_current().is_ok() {
            crate::ratelimit::ensure_scheduler(&state);
        }
        Ok(state)
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
    /// Returns [`Error::HandedOff`] when this daemon handed the session
    /// off to the next generation, and [`Error::UnknownSession`] when the
    /// session is otherwise not live.
    ///
    /// # Panics
    ///
    /// Panics if the live mutex is poisoned.
    pub fn get(&self, session_id: &str) -> Result<Arc<Subagent>> {
        let sub = self
            .live
            .lock()
            .expect("live poisoned")
            .get(session_id)
            .cloned();
        if let Some(sub) = sub {
            return Ok(sub);
        }
        self.not_shipped(session_id)?;
        Err(Error::UnknownSession(session_id.to_string()))
    }

    /// Refuse a session this daemon handed off to the next generation.
    /// A caller checks it after finding the session missing from `live`:
    /// a ship is recorded before the session leaves `live`, so the miss
    /// already sees it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::HandedOff`] when the session was shipped.
    ///
    /// # Panics
    ///
    /// Panics if the shipped mutex is poisoned.
    pub fn not_shipped(&self, session_id: &str) -> Result<()> {
        if self
            .handover
            .shipped
            .lock()
            .expect("shipped poisoned")
            .contains(session_id)
        {
            return Err(Error::HandedOff(session_id.to_string()));
        }
        Ok(())
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
            // A superseded daemon stops persisting — the newest generation
            // owns `registry.json`; this mutation still lands in memory and
            // rides the next handoff or release report to it. The flag is
            // set under `registry_write`, so a mutation serialized after
            // the takeover is always reported rather than written.
            if self.handover.superseded.load(Ordering::Relaxed) {
                return Ok(());
            }
            registry.snapshot()
        }?;
        persist(&self.registry_path, snapshot).await
    }
}

/// A `session/load` launch's claim on a session id.
///
/// Held from `launch`'s preflight until the session either goes live or
/// fails, so a `close` or `forget` landing mid-launch is recorded
/// instead of overtaken by the launch.
#[derive(Debug, Default)]
pub struct Reservation {
    /// `close`/`forget` arrived while the launch was in flight: the
    /// launch ends the session it just established rather than
    /// returning it live.
    pub closed: bool,
    /// The in-flight end was a `forget`: the launch drops the session's
    /// registry entry with it instead of leaving it adoptable.
    pub forgotten: bool,
}

/// Daemon-generation handover coordination.
///
/// A restart starts the next daemon first: it takes the public socket
/// over, then calls `daemon/takeover` on this daemon's private socket,
/// which sets `superseded`. The superseded daemon stops persisting the
/// registry, keeps only sessions with a turn or launch in flight, hands
/// each off as it settles — woken by `progressed`, which fires on every
/// turn end, launch settle, and session departure — and publishes
/// `finished` once it owns nothing, so the accept loop exits.
///
/// On the newest daemon `foreign` maps sessions an older generation
/// still owns to that daemon's private socket — calls for them forward
/// there until a `daemon/handoff` or `daemon/release` lands — and
/// `resuming` holds the ids of sessions mid-`session/load` after a
/// handoff.
#[derive(Debug)]
pub struct Handover {
    /// A newer daemon told this one it is superseded. Registry persists
    /// stop; in-flight sessions hand off as they settle.
    pub superseded: AtomicBool,
    /// The superseding daemon's private socket — handoff and release
    /// reports go here while `superseded` holds. Also the relay target
    /// for handovers this daemon receives after being superseded itself.
    pub next: Mutex<Option<PathBuf>>,
    /// Sessions another generation still owns, by that daemon's private
    /// socket — calls for them forward there until the owner hands them
    /// off or releases them.
    pub foreign: Mutex<HashMap<String, PathBuf>>,
    /// The superseded generation's private socket while it may still own
    /// sessions this daemon has not heard of — a spawn it accepted on a
    /// connection opened before the takeover. A call for a session this
    /// daemon does not know forwards there; cleared once it stops
    /// answering.
    pub prev: Mutex<Option<PathBuf>>,
    /// Sessions this superseded daemon handed off to `next`. A call that
    /// still reaches it for one — sent before the successor dropped its
    /// forward, or straight to this daemon's private socket — refuses with
    /// [`Error::HandedOff`] instead of answering from what this daemon
    /// last knew, so a forwarding successor routes the call to the
    /// session's new owner. Recorded before the session leaves `live`.
    pub shipped: Mutex<HashSet<String>>,
    /// Sessions mid-resume after a handoff — a call for one waits on
    /// `progressed` until the launch settles.
    pub resuming: Mutex<HashSet<String>>,
    /// Wakes the handover watcher on a superseded daemon and the
    /// resuming-call waiters on the newest one.
    pub progressed: Notify,
    /// `true` once a superseded daemon owns nothing — the accept loop
    /// exits.
    pub finished: tokio::sync::watch::Sender<bool>,
    /// The successor died while this superseded daemon still owned
    /// sessions: the accept loop takes the socket back. Fired with
    /// `notify_one`, so the permit is kept until the loop polls it.
    pub orphaned: Notify,
    /// This daemon's private socket path — set by `daemon::run` once
    /// bound; `daemon/takeover` names it for the sessions it keeps.
    pub private_socket: Mutex<Option<PathBuf>>,
    /// The locked pid file: this daemon's claim on the socket, held while
    /// it owns the socket and released to its successor by
    /// `daemon/takeover`.
    pub claim: Mutex<Option<HeldLock>>,
}

/// An exclusive `flock` this process holds, released on drop with
/// `LOCK_UN`.
///
/// Closing the descriptor does not release the lock. The lock belongs to
/// the open file description, and a child this process is spawning
/// holds a copy of every descriptor until its exec closes the
/// close-on-exec ones, so the lock outlives the close until then.
/// `LOCK_UN` removes the lock from the description whatever copies
/// exist, so the next holder can take it as soon as this one lets go.
#[derive(Debug)]
pub struct HeldLock(std::fs::File);

impl HeldLock {
    /// Take the lock on `file` without blocking.
    ///
    /// # Errors
    ///
    /// Returns the file back with the lock error — `WouldBlock` while
    /// another description holds it.
    pub fn try_new(
        file: std::fs::File,
    ) -> std::result::Result<Self, (std::fs::File, std::fs::TryLockError)> {
        match file.try_lock() {
            Ok(()) => Ok(Self(file)),
            Err(error) => Err((file, error)),
        }
    }

    /// Take the lock on `file`, blocking until its holder releases it.
    ///
    /// # Errors
    ///
    /// Returns the `flock` error.
    pub fn new(file: std::fs::File) -> std::io::Result<Self> {
        file.lock()?;
        Ok(Self(file))
    }

    /// The locked file.
    pub const fn file(&mut self) -> &mut std::fs::File {
        &mut self.0
    }
}

impl Drop for HeldLock {
    fn drop(&mut self) {
        if let Err(error) = self.0.unlock() {
            tracing::error!(%error, "cannot release a file lock");
        }
    }
}

impl Default for Handover {
    fn default() -> Self {
        Self {
            superseded: AtomicBool::new(false),
            next: Mutex::new(None),
            foreign: Mutex::new(HashMap::new()),
            prev: Mutex::new(None),
            shipped: Mutex::new(HashSet::new()),
            resuming: Mutex::new(HashSet::new()),
            progressed: Notify::new(),
            finished: tokio::sync::watch::channel(false).0,
            orphaned: Notify::new(),
            private_socket: Mutex::new(None),
            claim: Mutex::new(None),
        }
    }
}

/// What the daemon's startup resume pass did, reported by
/// `daemon/resumed`.
#[derive(Debug, Default, Clone, Serialize)]
pub struct ResumeReport {
    /// Session ids re-adopted via `session/load`.
    pub resumed: Vec<String>,
    /// Sessions whose resume failed, with the agent's error.
    pub failed: Vec<ResumeFailure>,
    /// Sessions the previous generation still owns or that are
    /// mid-resume — they land here when their turn ends and the handoff
    /// completes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending: Vec<String>,
}

/// One session's failed resume in a [`ResumeReport`].
#[derive(Debug, Clone, Serialize)]
pub struct ResumeFailure {
    /// Session id.
    pub session_id: String,
    /// The launch error — the agent's own message when it produced one.
    pub error: String,
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
    /// The agent's provider rate-limited the last turn; the session parks
    /// until `resume_at` and the daemon resumes it on its own — with a
    /// continuation when a turn was interrupted, else with its first
    /// queued prompt. Carries the resume time, the provider's reason and
    /// what the resume sends.
    RateLimited {
        /// When the provider says the quota resets.
        resume_at: jiff::Timestamp,
        /// The provider's own message.
        reason: String,
        /// What the resume sends first.
        plan: crate::registry::ResumePlan,
    },
    /// The agent process exited; the session is a husk `adopt` reaps (or
    /// `close`/`forget` removes). Carries the reason.
    Exited(String),
}

impl From<LastTurn> for Status {
    fn from(last_turn: LastTurn) -> Self {
        match last_turn {
            LastTurn::Done { stop_reason } => Self::Done(stop_reason),
            LastTurn::Cancelled => Self::Cancelled,
            LastTurn::Failed { reason } => Self::Failed(reason),
        }
    }
}

impl Status {
    /// How the last turn ended, when this status reports a finished one —
    /// what a handoff carries to the next daemon.
    #[must_use]
    pub fn last_turn(&self) -> Option<LastTurn> {
        match self {
            Self::Done(stop_reason) => Some(LastTurn::Done {
                stop_reason: *stop_reason,
            }),
            Self::Cancelled => Some(LastTurn::Cancelled),
            Self::Failed(reason) => Some(LastTurn::Failed {
                reason: reason.clone(),
            }),
            Self::Idle
            | Self::Running
            | Self::NeedsPermission
            | Self::RateLimited { .. }
            | Self::Exited(_) => None,
        }
    }

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
            Self::RateLimited { .. } => "rate_limited",
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
    /// When the turn left the wire — the end of its active span: a wait
    /// that followed the turn through a park measures the pre-park part
    /// from `started_at` to here.
    pub ended_at: Option<jiff::Timestamp>,
    /// The turn ended `rate_limited`: the session parks until its scope's
    /// `resume_at`, and a wait bound to it follows the resume's turn —
    /// the pause does not count against `expect_secs`.
    pub rate_limited: bool,
    /// Resolutions of the steers that targeted this turn, in the order
    /// they resolved — `wait` lists them.
    pub steers: Vec<SteerEnd>,
}

/// A steered `session/prompt` still in flight on the wire.
#[derive(Debug)]
pub struct Steer {
    /// The steer id `send` reported.
    pub id: u64,
    /// The turn it targeted — the `current` turn's number when sent.
    pub target: u64,
}

/// A steered prompt's resolution on the turn it targeted, for `wait`'s
/// `steers` list.
#[derive(Debug, Clone, Serialize)]
pub struct SteerEnd {
    /// The steer id `send` reported.
    pub steer_id: u64,
    /// What the steer became.
    #[serde(flatten)]
    pub outcome: SteerOutcome,
}

/// How a steered prompt resolved.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum SteerOutcome {
    /// The agent folded it into the turn it targeted — answered with that
    /// turn's end, or settled at it on a `folded` agent.
    Folded,
    /// The agent rejected its `session/prompt`.
    Rejected {
        /// The agent's error text.
        error: String,
    },
    /// It landed behind the target turn's end and ran as a turn of its
    /// own.
    Turn {
        /// The turn it ran as.
        turn: u64,
    },
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
    /// Steered prompts whose `session/prompt` result is still in
    /// flight and still holds the prompt slot, oldest first — the wire
    /// order their turns would run in. A steer that lands after its target
    /// turn ended runs as a turn of its own; the first untracked
    /// `session/update` materializes a `current` for it (see
    /// `steer_owner`), and the queue holds until they resolve.
    pub steer_pending: VecDeque<Steer>,
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
    /// The turn whose end is settled here but not yet recorded in the
    /// registry. Until `finish_turn` records it, every client-facing
    /// report treats the turn as still in flight: `wait` and `wait_any`
    /// stay bound to it, `status`, `list` and `result` hold their read,
    /// and the handover keeps the session — so no turn is reported ended
    /// that a crash would lose, or shipped without its count.
    pub recording: Option<u64>,
}

impl Inner {
    /// Whether the session has work in flight a handover must wait out: a
    /// running turn, a permission request, an unresolved steer, or a turn
    /// end not yet recorded.
    pub(crate) fn in_flight(&self) -> bool {
        matches!(self.status, Status::Running | Status::NeedsPermission)
            || !self.steer_pending.is_empty()
            || self.recording.is_some()
    }
}

/// Why a session refuses new work: the teardown that claimed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Closing {
    /// `close`/`forget`/`close_sub` teardown — the session is ending on
    /// this daemon.
    Ending = 1,
    /// `handoff_one` is shipping the session to the next daemon
    /// generation; a failed report clears the mark, and a refusal it
    /// caused holds until the ship resolves.
    HandingOff = 2,
}

impl Closing {
    /// The status a refusal caused by this mark reports.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Ending => "closing",
            Self::HandingOff => "handing off",
        }
    }
}

/// The session's closing mark — `Option<Closing>` in a lock-free cell.
#[derive(Debug, Default)]
pub struct ClosingFlag(AtomicU8);

impl ClosingFlag {
    /// The current mark, if the session is closing.
    pub(crate) fn get(&self) -> Option<Closing> {
        match self.0.load(Ordering::Relaxed) {
            1 => Some(Closing::Ending),
            2 => Some(Closing::HandingOff),
            _ => None,
        }
    }

    /// Set the mark.
    pub(crate) fn set(&self, closing: Closing) {
        self.0.store(closing as u8, Ordering::Relaxed);
    }

    /// Clear a `HandingOff` mark after a failed ship — a `close` that
    /// landed meanwhile keeps its `Ending`.
    pub(crate) fn clear_handoff(&self) {
        let _ = self.0.compare_exchange(
            Closing::HandingOff as u8,
            0,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }
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
    /// The teardown that claimed the session — `close`/`forget` before
    /// the connection is shut down (`Ending`), or a `handoff_one` ship
    /// to the next daemon generation (`HandingOff`).
    pub closing: ClosingFlag,
    /// `wait`/`wait_any` calls attached to this subagent — each live one
    /// feeds the turn watchdog.
    pub waiters: AtomicU64,
    /// Wakes the turn watchdog on a waiter attach or detach.
    pub watch: Notify,
    /// The agent child process id — also its process group id, since the
    /// child is spawned with `process_group(0)`. 0 until spawned.
    pub agent_pid: AtomicU32,
}

impl SubRuntime {
    /// Read `inner` once no turn end is awaiting its registry record — the
    /// read and the check share one critical section, so what `read` sees
    /// is durable. Waits on `notify`, which `finish_turn` fires once the
    /// record lands.
    ///
    /// # Panics
    ///
    /// Panics if the inner mutex is poisoned.
    pub async fn read_recorded<R>(&self, mut read: impl FnMut(&Inner) -> R) -> R {
        loop {
            // Armed before the check, so a record landing in between
            // still wakes this read.
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let inner = self.inner.lock().expect("inner poisoned");
                if inner.recording.is_none() {
                    return read(&inner);
                }
            }
            notified.await;
        }
    }

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

    /// SIGKILL the agent's whole process group — the agent leads the group
    /// (`process_group(0)`), so this sweeps the `bash -c` children it
    /// exec'd, including ones already orphaned to launchd.
    pub fn kill_process_tree(&self) {
        kill_process_group(self.agent_pid.load(Ordering::Relaxed));
    }
}

/// SIGKILL the process group led by `pgid`. A process group outlives its
/// leader, so this still reaches a dead agent's leaked descendants. Probes
/// the group first; a pgid meanwhile recycled by a foreign group leader is
/// the accepted residual.
///
/// On unix; a no-op elsewhere.
#[cfg(unix)]
pub(crate) fn kill_process_group(pgid: u32) {
    if pgid == 0 {
        return;
    }
    // SAFETY: kill(2) only delivers a signal; a stale pgid is narrowed by
    // the existence probe.
    unsafe {
        let pgid = pgid as libc::pid_t;
        if libc::kill(-pgid, 0) == 0 {
            libc::kill(-pgid, libc::SIGKILL);
        }
    }
}

/// No-op on platforms without process groups.
#[cfg(not(unix))]
pub(crate) fn kill_process_group(_pgid: u32) {}

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

/// Current RFC 3339 timestamp.
#[must_use]
pub fn now() -> String {
    jiff::Timestamp::now().to_string()
}

/// Rebuild the settled turns of a session's earlier life from its JSONL
/// transcript: every record carries its turn number, so replaying the
/// `update` payloads through the same summarizers the handler applies
/// restores each turn's reply, tool calls, plan and stop reason.
fn replay_transcript(text: &str) -> Vec<Turn> {
    let mut turns: BTreeMap<u64, Turn> = BTreeMap::new();
    for line in text.lines() {
        let Ok(record) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let Some(n) = record
            .get("turn")
            .and_then(serde_json::Value::as_u64)
            .filter(|n| *n > 0)
        else {
            continue;
        };
        let ts = record
            .get("ts")
            .and_then(serde_json::Value::as_str)
            .and_then(|ts| ts.parse::<jiff::Timestamp>().ok());
        let turn = turns.entry(n).or_insert_with(|| Turn {
            n,
            started_at: ts,
            ..Turn::default()
        });
        // The `prompt` record opens a turn; steer-materialized turns keep
        // their first record's timestamp.
        if record.get("prompt").is_some() {
            turn.started_at = ts;
        }
        if let Some(update) = record.get("update")
            && let Ok(update) = serde_json::from_value::<SessionUpdate>(update.clone())
        {
            match update {
                SessionUpdate::AgentMessageChunk(chunk) => {
                    if let ContentBlock::Text(text) = chunk.content {
                        turn.reply.push_str(&text.text);
                    }
                }
                SessionUpdate::ToolCall(call) => note_tool_call(turn, call),
                SessionUpdate::ToolCallUpdate(update) => note_tool_call_update(turn, update),
                SessionUpdate::Plan(plan) => turn.plan = plan.entries,
                SessionUpdate::CurrentModeUpdate(mode) => {
                    turn.mode = Some(mode.current_mode_id);
                }
                _ => {}
            }
        }
        if let Some(reason) = record.get("stop_reason") {
            turn.ended_at = ts.or(turn.ended_at);
            match reason {
                serde_json::Value::String(reason) if reason == "rate_limited" => {
                    turn.rate_limited = true;
                }
                other => {
                    if let Ok(reason) = serde_json::from_value::<StopReason>(other.clone()) {
                        turn.stop_reason = Some(reason);
                    }
                }
            }
        }
    }
    turns.into_values().collect()
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
/// [`Error::SpawnFailed`], [`Error::ClosedWhileLaunching`] when a `close`
/// landed while a `session/load` launch was in flight, or
/// [`Error::Agent`] when the ACP handshake fails.
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
            if live.contains_key(session_id) && !husk {
                return Err(Error::SessionLive(session_id.clone()));
            }
            match reserved.entry(session_id.clone()) {
                Entry::Occupied(_) => return Err(Error::SessionLive(session_id.clone())),
                Entry::Vacant(slot) => {
                    slot.insert(Reservation::default());
                }
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
        state.handover.progressed.notify_waiters();
    }
    result
}

/// The per-session runtime bundle for a just-connected agent: `adopt`
/// knows its session id up front; `spawn` fills it in during the
/// handshake.
fn new_runtime(args: &Launch, steer: SteerSemantics) -> Arc<SubRuntime> {
    Arc::new(SubRuntime {
        inner: Mutex::new(Inner {
            status: Status::Idle,
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
            recording: None,
        }),
        notify: Notify::new(),
        prompt_send: Arc::new(tokio::sync::Mutex::new(())),
        steer,
        transcript: tokio::sync::Mutex::new(TranscriptWriter::deferred()),
        terminals: Terminals::new(HashMap::new()),
        stderr_tail: Mutex::new(VecDeque::new()),
        next_perm: AtomicU64::new(1),
        next_steer: AtomicU64::new(1),
        next_term: AtomicU64::new(1),
        closing: ClosingFlag::default(),
        waiters: AtomicU64::new(0),
        watch: Notify::new(),
        agent_pid: AtomicU32::new(0),
    })
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
    let rt = new_runtime(args, agent_cfg.steer);

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
            rt.closing.set(Closing::Ending);
            rt.kill_process_tree();
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
        rt.closing.set(Closing::Ending);
        rt.kill_process_tree();
        client.close();
        let _ = conn.await;
        return Err(Error::io(
            format!("cannot open {}", transcript_path.display()),
            source,
        ));
    }
    // A resumed session's earlier turns replay from the transcript: the
    // file carries every turn's prompts, updates and stop reason, so
    // `result`, `status` and `wait` answer for the turns that ran before
    // the restart or adopt — not only the ones this process sees. The
    // registry's turn count stays the numbering authority, so the next
    // turn still takes `turns + 1` even if the file is short.
    if args.load.is_some() {
        let restored = match tokio::fs::read_to_string(&transcript_path).await {
            Ok(text) => replay_transcript(&text),
            Err(error) => {
                if error.kind() != std::io::ErrorKind::NotFound {
                    warn!(%error, "cannot read transcript for turn replay");
                }
                Vec::new()
            }
        };
        if !restored.is_empty() {
            let mut inner = rt.inner.lock().expect("inner poisoned");
            inner.turn_offset = inner.turn_offset.saturating_sub(restored.len() as u64);
            inner.turns = restored;
        }
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
    tokio::spawn(crate::watchdog::run(state.clone(), sub.clone()));
    if let Some(session_id) = &args.load {
        release_reservation(state, &sub, session_id).await?;
    }
    if let Some(prompt) = &args.prompt {
        // A quota gate for this agent parks the prompt instead of letting
        // it burn a request that is bound to fail.
        if crate::ratelimit::park_prompt(state, &sub, prompt)
            .await
            .is_none()
            && let Err(error) = start_turn(state.clone(), &sub, prompt.clone()).await
        {
            state
                .live
                .lock()
                .expect("live poisoned")
                .remove(&session_id);
            return Err(error);
        }
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
    sub.rt.closing.set(Closing::Ending);
    sub.rt.kill_process_tree();
    sub.client.clone().close();
    let conn = sub.conn.lock().expect("conn poisoned").take();
    if let Some(conn) = conn {
        let _ = conn.await;
    }
    Err(Error::SessionLive(sub.session_id.clone()))
}

/// Release a `session/load` launch's claim on `session_id`, settling a
/// `close`/`forget` that landed while the launch was in flight.
///
/// The release and the flag read sit in one critical section holding
/// `live` then `reserved` — the lock order `launch`'s claim and
/// `close`/`forget` both take — so the end is serialized either before
/// `go_live` (the reservation carries its mark and the teardown is this
/// launch's) or after it (the end took the ordinary live path and the
/// session is already ending). Either way the result is
/// [`Error::ClosedWhileLaunching`]; only a clean reservation and a still
/// present live entry let the launch proceed.
async fn release_reservation(
    state: &Arc<AppState>,
    sub: &Arc<Subagent>,
    session_id: &str,
) -> Result<()> {
    /// What the release found.
    enum Released {
        /// No end landed — the session stays live.
        Clean,
        /// An end marked the reservation: this launch removed the
        /// session from `live` and ends it below.
        Marked,
        /// An end took the ordinary live path in the gap between
        /// `go_live` and the release — it is ending the session.
        Taken,
    }
    let (released, forgotten) = {
        let mut live = state.live.lock().expect("live poisoned");
        // `end_target` locks `live` first, so holding it alone still
        // serializes this section against a `close`/`forget`.
        let reservation = state
            .reserved
            .lock()
            .expect("reserved poisoned")
            .remove(session_id)
            .expect("reservation held until release");
        // A second `close`/`forget` that landed after `go_live` may hold
        // the teardown already: only whoever removed the live entry ends
        // the session.
        let released = if reservation.closed {
            if live.remove(session_id).is_some() {
                Released::Marked
            } else {
                Released::Taken
            }
        } else if live.contains_key(session_id) {
            Released::Clean
        } else {
            Released::Taken
        };
        (released, reservation.forgotten)
    };
    match released {
        Released::Clean => {
            state.handover.progressed.notify_waiters();
            return Ok(());
        }
        Released::Marked => end_session(state, sub).await,
        Released::Taken => {}
    }
    // A `forget` takes the registry entry with the session: the
    // handshake's `register_session` may have re-created it after
    // `forget`'s own removal.
    if forgotten {
        let result = state
            .update_registry(|registry| {
                registry.remove(session_id);
            })
            .await;
        if let Err(error) = result {
            warn!(%session_id, %error, "registry persist failed");
        }
    }
    state.handover.progressed.notify_waiters();
    Err(Error::ClosedWhileLaunching(session_id.to_string()))
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
    let mut std_command = std::process::Command::new(&agent_cfg.command);
    std_command
        .args(&agent_cfg.args)
        .envs(agent_cfg.env.clone())
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // The agent leads its own process group so `kill_process_group` can
    // reap every `bash -c` descendant it spawned — orphaned ones reparent
    // to launchd but stay in the group.
    #[cfg(unix)]
    std_command.process_group(0);
    // `From` resets async-process's stdio flags, so re-apply the pipes on
    // the wrapper or `spawn` downgrades them to `inherit`.
    let mut command = async_process::Command::from(std_command);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|source| Error::SpawnFailed {
        agent: agent_cfg.command.clone(),
        command: format!("{} {}", agent_cfg.command, agent_cfg.args.join(" ")),
        source,
    })?;
    rt.agent_pid.store(child.id(), Ordering::Relaxed);
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
    register_session(state, &session_id, args).await?;
    debug!(%session_id, agent = %args.agent, "subagent session established");
    Ok(Accepted {
        session_id,
        model: accepted_model,
        mode: accepted_mode,
    })
}

/// Upsert `args`' session in the registry after a handshake: an existing
/// entry keeps its `created` timestamp and turn count; a new one gets a
/// fresh entry. Model and mode are recorded for a `session/load`
/// resume; a parked schedule the entry carried is consumed — the session
/// is live again.
///
/// # Errors
///
/// Returns an error when the registry cannot be persisted.
async fn register_session(state: &Arc<AppState>, session_id: &str, args: &Launch) -> Result<()> {
    state
        .update_registry(|registry| match registry.get_mut(session_id) {
            Some(entry) => {
                entry.agent.clone_from(&args.agent);
                entry.cwd.clone_from(&args.cwd);
                entry.model.clone_from(&args.model);
                entry.mode.clone_from(&args.mode);
                entry.rate_limited = None;
                entry.resume = None;
            }
            None => registry.insert(
                session_id.to_string(),
                RegistryEntry {
                    agent: args.agent.clone(),
                    cwd: args.cwd.clone(),
                    created: now(),
                    last_turn: None,
                    turn_started: None,
                    turns: 0,
                    model: args.model.clone(),
                    mode: args.mode.clone(),
                    rate_limited: None,
                    resume: None,
                },
            ),
        })
        .await
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
    if rt.closing.get().is_some() {
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
    // The agent died unsupervised: its exec'd children outlive it as
    // launchd orphans unless the group is swept.
    rt.kill_process_tree();
    rt.notify.notify_waiters();
}

/// Where ending a session id found it.
#[derive(Debug)]
pub(crate) enum SessionEnd {
    /// A live subagent — removed from `live`; the caller tears it down.
    Live(Arc<Subagent>),
    /// A session id claimed by an in-flight `session/load` launch: the
    /// reservation is marked so the launch ends the session as it lands
    /// instead of overtaking this end.
    Launching,
    /// Neither live nor claimed.
    Missing,
}

/// Find where ending `session_id` lands.
///
/// Removes the id from `live`, or marks its in-flight launch's
/// reservation closed, under one `live`-then-`reserved` critical
/// section — the lock order `launch`'s claim and [`release_reservation`]
/// take — so the end is serialized against the release: it either marks
/// the reservation before `go_live` or finds the session live after it,
/// never slipping between the two.
///
/// `forgotten` is `forget`'s end: the launch drops the registry entry
/// with the session instead of leaving it adoptable.
///
/// Shared by the `close` and `forget` tools.
#[expect(
    clippy::option_if_let_else,
    reason = "the else arm mutates under the `reserved` lock — a match keeps the lock order legible"
)]
pub(crate) fn end_target(state: &AppState, session_id: &str, forgotten: bool) -> SessionEnd {
    let mut live = state.live.lock().expect("live poisoned");
    let mut reserved = state.reserved.lock().expect("reserved poisoned");
    let end = match live.remove(session_id) {
        Some(sub) => SessionEnd::Live(sub),
        None => match reserved.get_mut(session_id) {
            Some(reservation) => {
                reservation.closed = true;
                reservation.forgotten |= forgotten;
                SessionEnd::Launching
            }
            None => SessionEnd::Missing,
        },
    };
    drop(reserved);
    drop(live);
    state.handover.progressed.notify_waiters();
    end
}

/// Close a subagent: drop pending requests and queues, close the client, and
/// take the connection and stderr tasks down.
///
/// Shared by the `close` and `forget` tools, by `adopt` reaping an `exited`
/// husk, and by the turn watchdog.
///
/// # Panics
///
/// Panics if a state mutex is poisoned.
pub async fn close_sub(sub: &Subagent) {
    sub.rt.closing.set(Closing::Ending);
    sub.rt.kill_process_tree();
    let terminals: Vec<_> = sub
        .rt
        .terminals
        .lock()
        .await
        .drain()
        .map(|(_, t)| t)
        .collect();
    for terminal in terminals {
        let _ = terminal.kill().await;
    }
    {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        for pending in inner.pending.drain(..) {
            let _ = pending.answer.send(RequestPermissionOutcome::Cancelled);
        }
        inner.queue.clear();
        inner.steer_pending.clear();
        inner.steer_folded.clear();
        inner.steer_owner = None;
        if matches!(
            inner.status,
            Status::Running | Status::NeedsPermission | Status::RateLimited { .. }
        ) {
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

/// Consume every registry mark that would bring `session_id` back on its
/// own.
///
/// Both schedules go: the rate-limit park the resume scheduler fires at
/// `resume_at`, and the restart `resume` mark the next daemon re-adopts
/// on. A session ended on purpose is done — the entry stays registered
/// for a manual `adopt`, but no schedule may relaunch it.
///
/// Shared by every path that ends a session deliberately: the `close`
/// tool and the turn watchdog.
pub async fn unschedule_resume(state: &Arc<AppState>, session_id: &str) {
    let result = state
        .update_registry(|registry| {
            if let Some(entry) = registry.get_mut(session_id) {
                entry.rate_limited = None;
                entry.resume = None;
            }
        })
        .await;
    if let Err(error) = result {
        warn!(%session_id, %error, "registry persist failed");
    }
}

/// End `sub`'s session for good: flag it closing, consume every registry
/// mark that would bring the session back, then tear the agent process
/// down.
///
/// `closing` is stored before the marks clear so a mark write serialized
/// after this update sees the flag and skips; one serialized before it is
/// consumed by `unschedule_resume`. `close_sub` stores the flag again —
/// idempotent — as part of teardown.
///
/// Shared by the paths that end a live session on purpose: the `close`
/// tool and the turn watchdog. `close` on a registry-only
/// session — no `Subagent`, nothing to flag or tear down — calls
/// `unschedule_resume` alone.
pub async fn end_session(state: &Arc<AppState>, sub: &Subagent) {
    sub.rt.closing.set(Closing::Ending);
    unschedule_resume(state, &sub.session_id).await;
    close_sub(sub).await;
    state.handover.progressed.notify_waiters();
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
        .or_else(|| inner.steer_pending.front().map(|steer| steer.id))
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
    /// The gate's verdict under `inner`.
    enum Verdict {
        /// The turn starts with this number and start stamp.
        Start(u64, jiff::Timestamp),
        /// `HandingOff` is set: hold until `handoff_one` resolves.
        HoldHandoff,
    }
    let (turn_n, started) = loop {
        let verdict = {
            let mut inner = sub.rt.inner.lock().expect("inner poisoned");
            let closing = sub.rt.closing.get();
            let ready = match gate {
                // `closing` also blocks a fresh turn: a session mid-handoff
                // to the next daemon generation must not start one here.
                Gate::Check => prompt_slot_free(&inner) && closing.is_none(),
                Gate::Chained => matches!(inner.status, Status::Running) && inner.current.is_none(),
            };
            if ready {
                let n = next_turn_number(&inner);
                let started = jiff::Timestamp::now();
                inner.current = Some(Turn {
                    n,
                    started_at: Some(started),
                    ..Turn::default()
                });
                inner.pending_turn_start = None;
                inner.status = Status::Running;
                drop(inner);
                Verdict::Start(n, started)
            } else if gate == Gate::Check && closing == Some(Closing::HandingOff) {
                Verdict::HoldHandoff
            } else {
                let status = closing.map_or_else(
                    || {
                        if inner.status.accepts_prompt() {
                            "steer in flight".to_string()
                        } else {
                            inner.status.name().to_string()
                        }
                    },
                    |closing| closing.name().to_string(),
                );
                return Err(Error::NotPromptable {
                    session_id: sub.session_id.clone(),
                    status,
                });
            }
        };
        match verdict {
            Verdict::Start(n, started) => break (n, started),
            // A `HandingOff` refusal holds until `handoff_one` resolves
            // rather than answering at once: the successor removed its
            // `foreign` route before it answered the handoff, so a refusal
            // that arrives only after the ship lands `still_owned ==
            // false` there and re-routes the call; a failed ship clears
            // the mark and the send retries the gate.
            Verdict::HoldHandoff if wait_out_handoff(state, sub).await => {}
            Verdict::HoldHandoff => return Err(Error::HandedOff(sub.session_id.clone())),
        }
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

/// Wait out `sub`'s `HandingOff` mark: `handoff_one` resolves by removing
/// the session from `live` — the caller's refusal then routes onward — or
/// by clearing the mark after a failed ship — the caller retries. Both
/// wakes ride `handover.progressed`; no polling, no sleep.
///
/// Returns `false` once the session has left `live` — it shipped.
pub(crate) async fn wait_out_handoff(state: &AppState, sub: &Arc<Subagent>) -> bool {
    loop {
        // Register the waiter BEFORE the check: a resolve between them
        // still wakes us.
        let notified = state.handover.progressed.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if !state
            .live
            .lock()
            .expect("live poisoned")
            .contains_key(&sub.session_id)
        {
            return false;
        }
        if sub.rt.closing.get() != Some(Closing::HandingOff) {
            return true;
        }
        notified.await;
    }
}

/// Spawn the turn-end task for a [`begin_turn`] result.
pub(crate) fn spawn_turn_task(
    state: Arc<AppState>,
    sub: &Arc<Subagent>,
    prompt: PromptFut,
    turn_n: u64,
) {
    tokio::spawn(turn_task(state, prompt, turn_n, sub.clone()));
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
/// future, which the caller hands to [`steer_waiter`].
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
        // In the queue handoff's gap (`Running`, no `current` yet) the
        // slot is reserved for the next turn — the one a wait binds to.
        let target = inner
            .current
            .as_ref()
            .map_or_else(|| next_turn_number(&inner), |turn| turn.n);
        inner.steer_pending.push_back(Steer {
            id: steer_id,
            target,
        });
        drop(inner);
        (steer_id, target)
    };
    let record =
        serde_json::json!({"ts": now(), "turn": turn_n, "steer": prompt, "steer_id": steer_id});
    if let Err(error) = sub.rt.transcript.lock().await.append(&record).await {
        warn!(%error, "transcript write failed");
    }
    let fut = enqueue_prompt(&sub.client, &sub.session_id, prompt).await;
    Ok((steer_id, turn_n, fut))
}

/// The `steer_end` transcript record for a resolved steer call.
pub(crate) fn steer_end_record(
    steer_id: u64,
    turn_n: u64,
    outcome: &std::result::Result<PromptResult, ClientError>,
) -> serde_json::Value {
    match outcome {
        Ok(result) => serde_json::json!({
            "ts": now(),
            "turn": turn_n,
            "steer_id": steer_id,
            "steer_end": serde_json::to_value(result.stop_reason).unwrap_or_default(),
        }),
        Err(error) => serde_json::json!({
            "ts": now(),
            "turn": turn_n,
            "steer_id": steer_id,
            "steer_end": "error",
            "error": error.to_string(),
        }),
    }
}

/// Note a steer's resolution on its target turn's record — `current`
/// while that turn is still in flight (a synchronous rejection), else its
/// settled record in `turns`. The first resolution stands: a steer already
/// settled `folded` at its target's end keeps that outcome when its own
/// turn's response lands later. A target turn that no longer exists was
/// discarded with a dead agent (`on_disconnect` drops `current`), and
/// there is no record left to report on.
fn record_steer_end(inner: &mut Inner, steer_id: u64, target_turn: u64, outcome: SteerOutcome) {
    let turn = if inner
        .current
        .as_ref()
        .is_some_and(|turn| turn.n == target_turn)
    {
        inner.current.as_mut()
    } else {
        inner.turns.iter_mut().find(|turn| turn.n == target_turn)
    };
    let Some(turn) = turn else {
        return;
    };
    if turn.steers.iter().any(|end| end.steer_id == steer_id) {
        return;
    }
    turn.steers.push(SteerEnd { steer_id, outcome });
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
/// The resolution lands on the target turn's `steers` list — what `wait`
/// reports — and the call's `rt.notify` wake is what releases a wait
/// holding for it.
pub(crate) async fn steer_resolved(
    state: Arc<AppState>,
    sub: &Arc<Subagent>,
    steer_id: u64,
    target_turn: u64,
    outcome: std::result::Result<PromptResult, ClientError>,
) {
    // A steered prompt rejected by a rate limit tells the scope about the
    // quota: later prompts park instead of burning a request. The steer
    // itself resolves as the rejection it was.
    if let Err(error) = &outcome
        && let Some(hit) = crate::ratelimit::parse(error)
    {
        crate::ratelimit::note_limit(&state, &sub.agent, &hit).await;
    }
    let (mine, kick, turn_n) = {
        let mut inner = sub.rt.inner.lock().expect("inner poisoned");
        inner.steer_pending.retain(|steer| steer.id != steer_id);
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
        // An answer outside a turn of its own means the agent folded the
        // steer into the turn it targeted.
        let resolution = match (&outcome, inner.current.as_ref().filter(|_| mine)) {
            (Err(error), _) => SteerOutcome::Rejected {
                error: error.to_string(),
            },
            (Ok(_), Some(turn)) => SteerOutcome::Turn { turn: turn.n },
            (Ok(_), None) => SteerOutcome::Folded,
        };
        record_steer_end(&mut inner, steer_id, target_turn, resolution);
        let kick = !mine
            && inner.steer_pending.is_empty()
            && inner.current.is_none()
            && inner.status.accepts_prompt()
            && !state.handover.superseded.load(Ordering::Relaxed)
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
        finish_turn(state, sub, turn_n, outcome.clone()).await;
        let record = steer_end_record(steer_id, turn_n, &outcome);
        if let Err(error) = sub.rt.transcript.lock().await.append(&record).await {
            warn!(%error, "transcript write failed");
        }
    } else {
        // The resolution is also the event the handover watcher needs: a
        // steer that resolves after its turn settled was the last
        // in-flight work keeping the session on a superseded daemon —
        // its handoff can now run. `finish_turn` notifies on the `mine`
        // path.
        state.handover.progressed.notify_waiters();
        if let Some(prompt) = kick {
            let wire = sub.rt.prompt_send.clone().lock_owned().await;
            spawn_chained(state, sub.session_id.clone(), prompt, wire);
        }
        let record = steer_end_record(steer_id, turn_n, &outcome);
        if let Err(error) = sub.rt.transcript.lock().await.append(&record).await {
            warn!(%error, "transcript write failed");
        }
        // The resolution is the event a wait bound to the target turn is
        // holding for (`finish_turn` notifies on the `mine` path).
        sub.rt.notify.notify_waiters();
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
    steer_resolved(state, &sub, steer_id, target_turn, outcome).await;
}

/// The in-flight `session/prompt` call, already enqueued by `start_turn`.
pub(crate) type PromptFut =
    Pin<Box<dyn Future<Output = std::result::Result<PromptResult, ClientError>> + Send>>;

/// Push the finished turn into `turns` and pick the follow-up state: a
/// `cancelled` turn drops the queue, a queued prompt becomes the next turn
/// (the status stays `running` — no `done` interlude), anything else settles
/// `done`. A steered prompt still in flight may materialize as a turn ahead
/// of the queue on the wire, so the queue pops only once they all resolve.
/// On a superseded daemon the queue never chains — it rides the session's
/// handoff to the next generation instead. Returns the prompt to chain, the
/// prompts the queue dropped, and the ids of steers settled as folded.
fn settle_turn(
    rt: &SubRuntime,
    reason: StopReason,
    superseded: bool,
) -> (Option<String>, VecDeque<String>, Vec<u64>) {
    let mut inner = rt.inner.lock().expect("inner poisoned");
    if let Some(mut turn) = inner.current.take() {
        turn.stop_reason = Some(reason);
        turn.ended_at = Some(jiff::Timestamp::now());
        inner.recording = Some(turn.n);
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
    } else if !superseded
        && inner.steer_pending.is_empty()
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
    let folded: Vec<Steer> = inner.steer_pending.drain(..).collect();
    for steer in &folded {
        record_steer_end(inner, steer.id, steer.target, SteerOutcome::Folded);
    }
    inner
        .steer_folded
        .extend(folded.iter().map(|steer| steer.id));
    folded.iter().map(|steer| steer.id).collect()
}

/// Push the finished turn into `turns`, mark the subagent `failed`, and
/// return the dropped queue and the ids of steers settled as folded. A turn
/// error that lands after `on_disconnect` does not downgrade `exited` — the
/// dead process cannot take the prompt `failed` would invite.
fn fail_turn(rt: &SubRuntime, reason: String) -> (VecDeque<String>, Vec<u64>) {
    let mut inner = rt.inner.lock().expect("inner poisoned");
    if let Some(mut turn) = inner.current.take() {
        turn.ended_at = Some(jiff::Timestamp::now());
        inner.recording = Some(turn.n);
        inner.turns.push(turn);
    }
    let folded = fold_pending_steers(rt, &mut inner);
    if !matches!(inner.status, Status::Exited(_)) {
        inner.status = Status::Failed(reason);
    }
    (std::mem::take(&mut inner.queue), folded)
}

/// Push the finished turn into `turns` marked `rate_limited`, park the
/// session until its scope's `resume_at`, and return the steers settled
/// as folded. Unlike [`fail_turn`] the queue survives: prompts parked by
/// the limit run after the resume's continuation.
fn park_turn(rt: &SubRuntime, hit: &crate::ratelimit::RateLimit) -> Vec<u64> {
    let mut inner = rt.inner.lock().expect("inner poisoned");
    if let Some(mut turn) = inner.current.take() {
        turn.ended_at = Some(jiff::Timestamp::now());
        turn.rate_limited = true;
        inner.recording = Some(turn.n);
        inner.turns.push(turn);
    }
    let folded = fold_pending_steers(rt, &mut inner);
    inner.pending_turn_start = None;
    if !matches!(inner.status, Status::Exited(_)) {
        inner.status = Status::RateLimited {
            resume_at: hit.resume_at,
            reason: hit.reason.clone(),
            plan: crate::registry::ResumePlan::Continue,
        };
    }
    folded
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
    for steer_id in steers {
        let record = serde_json::json!({
            "ts": now(),
            "turn": turn_n,
            "steer_id": steer_id,
            "steer_end": "folded",
        });
        if let Err(error) = rt.transcript.lock().await.append(&record).await {
            warn!(%error, "transcript write failed");
        }
    }
}

/// Await `session/prompt`, then [`finish_turn`].
async fn turn_task(state: Arc<AppState>, prompt: PromptFut, turn_n: u64, sub: Arc<Subagent>) {
    let outcome = prompt.await;
    finish_turn(state, &sub, turn_n, outcome).await;
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
            Ok(sub) => {
                // A limit learned since the prompt was queued: re-park it
                // at the queue's front — the scheduler's continuation
                // comes first, queued prompts run after it.
                if let Some(limit) = crate::ratelimit::active_limit(&state, &sub.agent) {
                    crate::ratelimit::repark_prompt(&state, &sub, &prompt, &limit).await;
                    return;
                }
                match begin_turn(&state, &sub, prompt, Gate::Chained, true).await {
                    Ok((n, fut)) => spawn_turn_task(state, &sub, fut, n),
                    Err(error) => warn!(%error, "queued turn failed to start"),
                }
            }
            Err(error) => warn!(%error, "queued prompt dropped: subagent gone"),
        }
    });
}

/// The shared settlement tail of `turn_task` and a steer-owned turn's
/// [`steer_resolved`]: record the turn's end in the transcript, settle or
/// fail state, record the end in the registry, and wake waiters.
///
/// The settle marks the turn `recording`, and the mark clears only once
/// the registry holds the end: client-facing reports treat the turn as
/// in flight until then, so an end they report is durable.
///
/// This task holds `rt.prompt_send` until the end is recorded, so the
/// between-turns transition is atomic: a `send`/`steer` sees either the
/// still-`running` pre-transition state or the recorded post-transition
/// one, never the gap, and no other turn end can settle in between. A
/// queued prompt popped here becomes the next turn without a `done`
/// interlude, starting once the end is recorded — the record never clears
/// the next turn's `turn_started`; a `cancelled` or failed turn drops the
/// queue instead.
async fn finish_turn(
    state: Arc<AppState>,
    sub: &Arc<Subagent>,
    turn_n: u64,
    outcome: std::result::Result<PromptResult, ClientError>,
) {
    let rt = &sub.rt;
    let session_id = &sub.session_id;
    let wire = rt.prompt_send.clone().lock_owned().await;
    // A provider rate limit ends the turn parked, not failed — the
    // session resumes on its own when the quota lifts.
    let hit = outcome.as_ref().err().and_then(crate::ratelimit::parse);
    let mut chained = None;
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
            let (next, dropped, folded) = settle_turn(
                rt,
                reason,
                state.handover.superseded.load(Ordering::Relaxed),
            );
            record_steers_folded(rt, turn_n, folded).await;
            record_queue_dropped(rt, Some(turn_n), dropped).await;
            chained = next;
        }
        Err(error) => {
            let mut reason = error.to_string();
            let tail = rt.stderr_tail();
            if !tail.is_empty() {
                let start = tail.len().saturating_sub(STDERR_QUOTE_LINES);
                reason.push_str("\nagent stderr (tail):\n");
                reason.push_str(&tail[start..].join("\n"));
            }
            let mut record = serde_json::json!({
                "ts": now(),
                "turn": turn_n,
                "stop_reason": "error",
                "error": reason,
            });
            if let Some(hit) = &hit {
                record["stop_reason"] = serde_json::json!("rate_limited");
                record["resume_at"] = serde_json::json!(hit.resume_at.to_string());
            }
            if let Err(error) = rt.transcript.lock().await.append(&record).await {
                warn!(%error, "transcript write failed");
            }
            if let Some(hit) = &hit {
                let folded = park_turn(rt, hit);
                record_steers_folded(rt, turn_n, folded).await;
            } else {
                let (dropped, folded) = fail_turn(rt, reason);
                record_steers_folded(rt, turn_n, folded).await;
                record_queue_dropped(rt, Some(turn_n), dropped).await;
            }
        }
    }
    record_turn_end(&state, rt, session_id, hit.as_ref()).await;
    rt.inner.lock().expect("inner poisoned").recording = None;
    if let Some(prompt) = chained {
        spawn_chained(state.clone(), session_id.clone(), prompt, wire);
    } else {
        drop(wire);
    }
    // After the record: the scheduler this wakes finds the parked flag,
    // and a later scope `resume_at` moves this session's record too.
    if let Some(hit) = &hit {
        crate::ratelimit::note_limit(&state, &sub.agent, hit).await;
    }
    rt.notify.notify_waiters();
    state.handover.progressed.notify_waiters();
}

/// Persist a turn's end: the count and `last_turn` stamp — and the
/// parked flag that keeps a rate-limited session's schedule across a
/// daemon restart, carrying the send-queue it leaves behind. A turn
/// that settles unparked disarms the flag: a parked queue chained into
/// turns while the limit held would otherwise rerun prompts that
/// already ran.
async fn record_turn_end(
    state: &Arc<AppState>,
    rt: &SubRuntime,
    session_id: &str,
    hit: Option<&crate::ratelimit::RateLimit>,
) {
    let (turns, queue) = {
        let inner = rt.inner.lock().expect("inner poisoned");
        (
            inner.turn_offset + inner.turns.len() as u64,
            inner.queue.clone(),
        )
    };
    let parked = hit.map(|hit| crate::registry::RateLimited {
        resume_at: hit.resume_at.to_string(),
        reason: hit.reason.clone(),
        plan: crate::registry::ResumePlan::Continue,
        queued: queue.into_iter().collect(),
    });
    let result = state
        .update_registry(|registry| {
            if let Some(entry) = registry.get_mut(session_id) {
                entry.turns = turns;
                entry.last_turn = Some(now());
                entry.turn_started = None;
                // The closing check runs under `registry_write`, serialized
                // against `unschedule_resume`: a `close` that already
                // stored the flag must not see its cleared mark rewritten.
                if let Some(parked) = parked
                    && rt.closing.get().is_none()
                {
                    entry.rate_limited = Some(parked);
                } else if !matches!(
                    rt.inner.lock().expect("inner poisoned").status,
                    Status::RateLimited { .. } | Status::Exited(_)
                ) {
                    // The armed schedule is spent once a turn settles
                    // unparked. `rate_limited` keeps its own record —
                    // a gated `send` can park the settled session
                    // between the settle and this write — and `exited` keeps the flag for the
                    // fresh-process resume.
                    entry.rate_limited = None;
                }
            }
        })
        .await;
    if let Err(error) = result {
        warn!(%error, "registry persist failed");
    }
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
