//! Typed errors for acpsub.

use std::path::PathBuf;

use thiserror::Error;

/// Errors produced by acpsub operations.
///
/// Tools convert these into `aither_core::Error` at the tool boundary so the
/// exact cause reaches the orchestrating model as a tool error.
#[derive(Debug, Error)]
pub enum Error {
    /// The config file does not exist.
    #[error("config file not found: {0}")]
    ConfigMissing(PathBuf),

    /// The config file exists but cannot be read.
    #[error("cannot read config {path}: {source}")]
    ConfigRead {
        /// Path of the config file.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// The config file exists but does not parse.
    #[error("cannot parse config {path}: {source}")]
    ConfigParse {
        /// Path of the config file.
        path: PathBuf,
        /// Underlying TOML parse error.
        source: toml::de::Error,
    },

    /// A `~` path was configured but no home directory is known.
    #[error("cannot expand '~' in configured path '{0}': no home directory")]
    NoHome(String),

    /// The named agent is not in the config file.
    #[error("unknown agent '{name}'{}", if configured.is_empty() { String::new() } else { format!(" (configured: {})", configured.join(", ")) })]
    UnknownAgent {
        /// The agent key that was not found.
        name: String,
        /// Configured agent keys.
        configured: Vec<String>,
    },

    /// No agent was passed and none could be inferred.
    #[error("no agent specified{}", if configured.is_empty() { "; none are configured".to_string() } else { format!("; set 'agent' in [defaults] or pass one of: {}", configured.join(", ")) })]
    AgentUnspecified {
        /// Configured agent keys.
        configured: Vec<String>,
    },

    /// The session id is neither live nor registered.
    #[error("unknown session '{0}'")]
    UnknownSession(String),

    /// The session id is empty.
    #[error("session id must not be empty")]
    EmptySessionId,

    /// A live subagent already holds the session id.
    #[error("session '{0}' is already live; close it first")]
    SessionLive(String),

    /// The agent does not advertise the `loadSession` capability.
    #[error("agent '{0}' does not support session/load; it cannot adopt sessions")]
    LoadUnsupported(String),

    /// The registry says the session belongs to a different agent.
    #[error("session '{session_id}' is registered to agent '{registered}', not '{requested}'")]
    AgentMismatch {
        /// The session being adopted.
        session_id: String,
        /// The agent the registry recorded.
        registered: String,
        /// The agent the caller asked for.
        requested: String,
    },

    /// `adopt` could not determine the session's working directory.
    #[error("cannot determine cwd for session '{session_id}' ({reason}); pass `cwd` explicitly")]
    SessionCwdUnknown {
        /// The session being adopted.
        session_id: String,
        /// Why discovery failed.
        reason: String,
    },

    /// The subagent is not in a state that accepts a prompt.
    #[error("session '{session_id}' is {status}; cannot accept a prompt now")]
    NotPromptable {
        /// Session id.
        session_id: String,
        /// Current status.
        status: String,
    },

    /// A `wait`/`wait_any` expected longer than the ceiling for a call
    /// without a progress channel.
    #[error(
        "expect_secs {expect_secs} exceeds the {ceiling_secs} s ceiling: the caller sent no progress token, so the call produces no keep-alive and the host would abandon it first"
    )]
    ExpectExceedsNoToken {
        /// The requested `expect_secs`.
        expect_secs: u64,
        /// The ceiling it exceeded.
        ceiling_secs: u64,
    },

    /// The subagent has no running turn to cancel.
    #[error("session '{0}' has no running turn")]
    NotRunning(String),

    /// The named permission request is not pending on this subagent.
    #[error("session '{session_id}' has no pending permission request '{request}'")]
    NoSuchPermission {
        /// Session id.
        session_id: String,
        /// Permission request id.
        request: String,
    },

    /// The chosen option was not offered by the permission request.
    #[error("option '{option}' was not offered by permission request '{request}' (offered: {})", .offered.join(", "))]
    BadOption {
        /// Permission request id.
        request: String,
        /// The option id that was rejected.
        option: String,
        /// Option ids that were offered.
        offered: Vec<String>,
    },

    /// The requested turn does not exist.
    #[error("session '{session_id}' has no turn {turn} (has {have})")]
    NoSuchTurn {
        /// Session id.
        session_id: String,
        /// Requested 1-based turn number.
        turn: u64,
        /// Number of turns the subagent has completed.
        have: u64,
    },

    /// The transcript file is missing.
    #[error("no transcript at {0}")]
    NoTranscript(PathBuf),

    /// The agent process could not be started.
    #[error("cannot spawn agent '{agent}' ({command}): {source}")]
    SpawnFailed {
        /// Agent config key.
        agent: String,
        /// Command that failed to start.
        command: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// An ACP client call failed.
    #[error("agent '{agent}': {source}")]
    Agent {
        /// Subagent name.
        agent: String,
        /// The client error.
        source: aither_acp::ClientError,
    },

    /// A config option value had an unsupported type.
    #[error("config option '{0}' must be a string or boolean")]
    BadConfigValue(String),

    /// The agent advertised no mode list, so `mode` cannot be checked.
    #[error("agent '{agent}' advertises no session modes; cannot check mode '{value}'")]
    ModesNotAdvertised {
        /// Agent config key.
        agent: String,
        /// The requested mode.
        value: String,
    },

    /// The requested mode is not among the agent's advertised modes.
    #[error("unknown mode '{value}' for agent '{agent}'{}", if valid.is_empty() { "; it advertises none".to_string() } else { format!(" (valid: {})", valid.join(", ")) })]
    UnknownMode {
        /// Agent config key.
        agent: String,
        /// The rejected mode.
        value: String,
        /// Mode ids the agent advertised.
        valid: Vec<String>,
    },

    /// The agent advertises no config option under this id.
    #[error("agent '{agent}' advertises no config option '{id}'{}", if known.is_empty() { "; it advertises none".to_string() } else { format!(" (known: {})", known.join(", ")) })]
    UnknownConfigOption {
        /// Agent config key.
        agent: String,
        /// The rejected config option id.
        id: String,
        /// Option ids the agent advertised.
        known: Vec<String>,
    },

    /// The config option exists but the agent advertises no values for it.
    #[error(
        "agent '{agent}' advertises no values for config option '{id}'; cannot check '{value}'"
    )]
    ConfigOptionValuesMissing {
        /// Agent config key.
        agent: String,
        /// The config option id.
        id: String,
        /// The requested value.
        value: String,
    },

    /// The requested value is not among the option's advertised values.
    #[error("unknown value '{value}' for config option '{id}' on agent '{agent}' (valid: {})", .valid.join(", "))]
    UnknownConfigValue {
        /// Agent config key.
        agent: String,
        /// The config option id.
        id: String,
        /// The rejected value.
        value: String,
        /// Values the agent advertised.
        valid: Vec<String>,
    },

    /// The registry file exists but does not parse.
    #[error("cannot parse {path}: {source}")]
    RegistryParse {
        /// Path of the registry file.
        path: PathBuf,
        /// Underlying JSON error.
        source: serde_json::Error,
    },

    /// An internal invariant was violated — an acpsub bug, not bad input.
    #[error("internal error on session '{session_id}': {detail}")]
    Internal {
        /// Session id.
        session_id: String,
        /// What invariant broke.
        detail: String,
    },

    /// A filesystem or registry I/O error.
    #[error("{context}: {source}")]
    Io {
        /// What was being written or read.
        context: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
}

impl Error {
    /// Wrap an I/O error with the operation that failed.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}

/// Result alias for acpsub operations.
pub type Result<T> = std::result::Result<T, Error>;
