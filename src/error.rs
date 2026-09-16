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

    /// `wait`/`wait_any` received a `timeout_secs` below the minimum.
    #[error("timeout_secs {got} is below the 60s minimum; use `status` for an instant state check")]
    TimeoutBelowMin {
        /// The rejected timeout.
        got: u64,
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

    /// The registry file exists but does not parse.
    #[error("cannot parse {path}: {source}")]
    RegistryParse {
        /// Path of the registry file.
        path: PathBuf,
        /// Underlying JSON error.
        source: serde_json::Error,
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
