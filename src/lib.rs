//! # acpsub
//!
//! `acpsub` runs any [ACP](https://agentclientprotocol.com) agent as a named,
//! resumable subagent behind MCP tools. An orchestrating agent (Claude Code,
//! aither, …) calls `spawn` with a configured agent name; `acpsub` starts the
//! process, opens an ACP session, streams every `session/update` to a
//! per-subagent JSONL transcript, and serves the agent's `fs/*`, `terminal/*`,
//! and permission requests under a configurable policy.
//!
//! Subagents persist in a registry keyed by name, so a `spawn` on a
//! previously-registered name resumes the session with `session/load` when
//! the agent supports it.

pub mod config;
pub mod error;
pub mod handler;
pub mod registry;
pub mod state;
pub mod terminal;
pub mod tools;
pub mod transcript;

pub use config::{AgentConfig, Config, Defaults, PermissionPolicy, default_config_path};
pub use error::{Error, Result};
pub use registry::{Registry, RegistryEntry};
pub use state::{AppState, Status, Subagent};
pub use tools::build_tools;
pub use transcript::{RenderOptions, TranscriptWriter, render};
