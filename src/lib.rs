//! # acpsub
//!
//! `acpsub` runs any [ACP](https://agentclientprotocol.com) agent as a
//! resumable subagent behind MCP tools. An orchestrating agent calls
//! `spawn` with a configured agent key; `acpsub` starts the process, opens
//! an ACP session, streams every `session/update` to a per-session JSONL
//! transcript, and serves the agent's `fs/*`, `terminal/*`, and permission
//! requests under a configurable policy.
//!
//! The agent-assigned `session_id` is the handle every tool addresses.
//! Sessions persist in a registry keyed by that id, so `adopt` can resume
//! one — whether acpsub spawned it or it was created elsewhere — with
//! `session/load` when the agent supports it.

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
