//! The daemon: `acpsub` state served over a Unix socket.
//!
//! `acpsub daemon` is the session host for the CLI client commands
//! (`spawn`, `send`, `wait`, …). It holds every live ACP child process
//! and accepts one JSON-RPC/MCP connection per client invocation. Unlike
//! `serve`, it is not tied to a host's stdio and survives the
//! orchestrating agent's restarts — sessions stay adoptable through the
//! registry either way.
//!
//! An orphan reaper runs alongside the accept loop: a subagent that was
//! spawned with an `owner` pid is closed once that pid is dead, so a
//! session cannot keep running — and spending — after its coordinator
//! exited.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use aither_mcp::McpServer;
use aither_mcp::transport::StreamTransport;
use futures_lite::io::BufReader;
use tracing::{debug, error, warn};

use crate::error::{Error, Result};
use crate::state::{AppState, Subagent, close_sub};
use crate::tools::build_tools;

/// How often the reaper checks owner pids.
const OWNER_CHECK: Duration = Duration::from_secs(2);

/// The default daemon socket: `~/.local/share/acpsub/daemon.sock`.
#[must_use]
pub fn default_socket_path() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".local/share/acpsub/daemon.sock"))
}

/// Serve the tool set over `socket_path` until the process is killed.
///
/// A stale socket file left by a dead daemon is unlinked and rebound; a
/// socket a live daemon still answers fails with [`Error::DaemonRunning`].
///
/// # Errors
///
/// Returns an error if the socket cannot be bound or its permissions set.
pub async fn run(state: Arc<AppState>, socket_path: &Path) -> Result<()> {
    let listener = bind(socket_path).await?;
    // The pid file lets a caller stop exactly this daemon:
    // `kill "$(cat daemon.sock.pid)"`.
    std::fs::write(
        socket_path.with_extension("sock.pid"),
        std::process::id().to_string(),
    )
    .map_err(|source| Error::io("cannot write daemon pid file", source))?;
    tokio::spawn(reap_orphans(state.clone()));
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let state = state.clone();
                tokio::spawn(async move {
                    if let Err(error) = serve_connection(stream, state).await {
                        debug!(%error, "client connection closed with error");
                    }
                });
            }
            Err(error) => {
                error!(%error, "socket accept failed");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

/// Bind the socket, reclaiming a stale one left by a dead daemon.
async fn bind(path: &Path) -> Result<async_net::unix::UnixListener> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|source| Error::io(format!("cannot create {}", parent.display()), source))?;
    }
    match async_net::unix::UnixListener::bind(path) {
        Ok(listener) => {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(|source| Error::io(format!("cannot chmod {}", path.display()), source))?;
            Ok(listener)
        }
        Err(bind_error) => {
            // A leftover socket file: probe it — if nothing answers, it is
            // stale and safe to replace.
            if async_net::unix::UnixStream::connect(path).await.is_ok() {
                return Err(Error::DaemonRunning(path.to_path_buf()));
            }
            std::fs::remove_file(path).map_err(|source| {
                Error::io(
                    format!("cannot remove stale socket {}", path.display()),
                    source,
                )
            })?;
            async_net::unix::UnixListener::bind(path)
                .map_err(|_| Error::io(format!("cannot bind {}", path.display()), bind_error))
        }
    }
}

/// One client connection: a fresh tool set over the shared state, served
/// until the client hangs up.
async fn serve_connection(
    stream: async_net::unix::UnixStream,
    state: Arc<AppState>,
) -> std::result::Result<(), aither_mcp::McpError> {
    let (read_half, write_half) = futures_lite::io::split(stream);
    let transport = StreamTransport::new(BufReader::new(read_half), write_half);
    let tools = build_tools(state).map_err(|error| {
        aither_mcp::McpError::InvalidConfig(format!("cannot build tools: {error}"))
    })?;
    let mut server = McpServer::new(transport, tools, "acpsub", env!("CARGO_PKG_VERSION"));
    server.run().await
}

/// Reap live subagents whose owning coordinator pid has exited.
async fn reap_orphans(state: Arc<AppState>) {
    loop {
        tokio::time::sleep(OWNER_CHECK).await;
        let orphans: Vec<(String, Arc<Subagent>)> = state
            .live
            .lock()
            .expect("live poisoned")
            .iter()
            .filter(|(_, sub)| sub.owner_dead())
            .map(|(id, sub)| (id.clone(), sub.clone()))
            .collect();
        for (session_id, sub) in orphans {
            let owner = sub.rt.owner.load(std::sync::atomic::Ordering::Relaxed);
            warn!(%session_id, owner, "coordinator pid is gone; closing subagent");
            state
                .live
                .lock()
                .expect("live poisoned")
                .remove(&session_id);
            close_sub(&sub).await;
        }
    }
}
