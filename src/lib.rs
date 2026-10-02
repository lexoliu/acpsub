//! # acpsub
//!
//! `acpsub` runs any [ACP](https://agentclientprotocol.com) agent as a
//! resumable subagent. The session host is either `acpsub serve` (an MCP
//! server over stdio, launched by an orchestrating agent) or
//! `acpsub daemon` (the same tool set over a Unix socket, driven by the
//! `acpsub` CLI). It starts the process, opens an ACP session, streams
//! every `session/update` to a per-session JSONL transcript, and serves
//! the agent's `fs/*`, `terminal/*`, and permission requests under a
//! configurable policy.
//!
//! The agent-assigned `session_id` is the handle every operation
//! addresses. Sessions persist in a registry keyed by that id, so `adopt`
//! can resume one — whether acpsub spawned it or it was created elsewhere
//! — with `session/load` when the agent supports it.

pub mod client;
pub mod config;
pub mod daemon;
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
pub use tools::{build_tools, build_tools_with_progress_interval};
pub use transcript::{RenderOptions, TranscriptWriter, render};
