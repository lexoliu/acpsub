//! The CLI's path to the daemon: one MCP `tools/call` per invocation.
//!
//! A CLI command connects to the daemon's Unix socket, performs the MCP
//! handshake, calls a single tool, prints the result, and exits. Every
//! call carries a `progressToken`, so a blocking `wait` is never capped
//! by the no-listener `expect_secs` ceiling — the client simply discards
//! the progress notifications; on a terminal or in a background task the
//! process exit itself is the signal.

use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use aither_mcp::protocol::{
    CallToolParams, CallToolResult, InitializeParams, JsonRpcNotification, JsonRpcRequest,
    McpError, ProgressToken, RequestMeta,
};
use aither_mcp::transport::{StreamTransport, Transport};
use futures_lite::io::BufReader;
use serde_json::Value;
use tracing::debug;

use crate::error::{Error, Result};

/// How long to wait for an auto-started daemon to begin answering.
const DAEMON_STARTUP_TIMEOUT: Duration = Duration::from_secs(10);
/// Connect retry interval while the daemon is starting.
const DAEMON_STARTUP_POLL: Duration = Duration::from_millis(50);

/// Call `tool` on the daemon at `socket` with `arguments` and return its
/// result.
///
/// When no daemon answers the socket, one is started automatically
/// (`acpsub daemon --socket …`), so the CLI works with no setup step.
/// `config` is forwarded to the daemon when given.
///
/// # Errors
///
/// Returns [`Error::DaemonConnect`] when the socket is unreachable and the
/// daemon cannot be started, or an error when the daemon fails the call.
pub async fn call(
    socket: &Path,
    tool: &str,
    arguments: Value,
    config: Option<&Path>,
) -> Result<CallToolResult> {
    let stream = connect_or_start(socket, config).await?;
    let (read_half, write_half) = futures_lite::io::split(stream);
    let mut transport = StreamTransport::new(BufReader::new(read_half), write_half);
    transport
        .request(JsonRpcRequest::with_params(
            0i64,
            "initialize",
            InitializeParams::default(),
        ))
        .await
        .map_err(rpc_failure)?;
    transport
        .notify(JsonRpcNotification::new("notifications/initialized"))
        .await
        .map_err(rpc_failure)?;
    let params = CallToolParams {
        name: tool.to_string(),
        arguments,
        meta: Some(RequestMeta {
            progress_token: Some(ProgressToken::String("acpsub-cli".to_string())),
        }),
    };
    let response = transport
        .request(JsonRpcRequest::with_params(1i64, "tools/call", params))
        .await
        .map_err(rpc_failure)?;
    let result = response
        .into_result()
        .map_err(|error| Error::io("daemon rejected the call", std::io::Error::other(error)))?;
    serde_json::from_value(result)
        .map_err(|source| Error::io("cannot parse daemon result", std::io::Error::other(source)))
}

/// Connect to the daemon socket, starting a daemon when none answers.
async fn connect_or_start(
    socket: &Path,
    config: Option<&Path>,
) -> Result<async_net::unix::UnixStream> {
    match async_net::unix::UnixStream::connect(socket).await {
        Ok(stream) => return Ok(stream),
        Err(source)
            if matches!(
                source.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
            ) =>
        {
            start_daemon(socket, config)?;
        }
        Err(source) => {
            return Err(Error::DaemonConnect {
                path: socket.to_path_buf(),
                source,
            });
        }
    }
    let deadline = std::time::Instant::now() + DAEMON_STARTUP_TIMEOUT;
    loop {
        match async_net::unix::UnixStream::connect(socket).await {
            Ok(stream) => return Ok(stream),
            Err(source)
                if matches!(
                    source.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                ) =>
            {
                if std::time::Instant::now() >= deadline {
                    return Err(Error::DaemonConnect {
                        path: socket.to_path_buf(),
                        source,
                    });
                }
                tokio::time::sleep(DAEMON_STARTUP_POLL).await;
            }
            Err(source) => {
                return Err(Error::DaemonConnect {
                    path: socket.to_path_buf(),
                    source,
                });
            }
        }
    }
}

/// Spawn `acpsub daemon` detached: its own process group, no inherited
/// stdio, so it survives the CLI process that launched it.
fn start_daemon(socket: &Path, config: Option<&Path>) -> Result<()> {
    let exe = std::env::current_exe()
        .map_err(|source| Error::io("cannot locate the acpsub executable", source))?;
    let mut command = std::process::Command::new(exe);
    command.arg("daemon").arg("--socket").arg(socket);
    if let Some(config) = config {
        command.arg("--config").arg(config);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0);
    debug!(socket = %socket.display(), "starting acpsub daemon");
    command
        .spawn()
        .map_err(|source| Error::io("cannot spawn acpsub daemon", source))?;
    Ok(())
}

/// Map a transport/protocol failure to a daemon-flavored error.
fn rpc_failure(error: McpError) -> Error {
    Error::io("daemon call failed", std::io::Error::other(error))
}
