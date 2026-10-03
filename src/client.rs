//! The CLI's path to the daemon: one MCP `tools/call` per invocation.
//!
//! A CLI command connects to the daemon's Unix socket, performs the MCP
//! handshake, calls a single tool, prints the result, and exits. Every
//! call carries a `progressToken`, so a blocking `wait` is never capped
//! by the no-listener `expect_secs` ceiling — the client simply discards
//! the progress notifications; on a terminal or in a background task the
//! process exit itself is the signal.
//!
//! Two restart behaviors ride on top. A read-only call whose connection
//! a draining daemon closes is not a failure: it reconnects to the next
//! daemon and re-issues the call — a `wait` keeps its `expect_secs`
//! budget because the budget measures from the turn's recorded start,
//! and a turn that already ended answers from the resumed session's
//! restored state. A `spawn`/`send`/`adopt` refused with `restarting`
//! waits for the old daemon to exit, then retries once.

use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use aither_mcp::protocol::{
    CallToolParams, CallToolResult, Content, InitializeParams, JsonRpcNotification, JsonRpcRequest,
    McpError, ProgressToken, RequestMeta,
};
use aither_mcp::transport::{StreamTransport, Transport};
use futures_lite::io::{AsyncReadExt, BufReader};
use serde_json::Value;
use tracing::debug;

use crate::error::{Error, Result};

/// How long to wait for an auto-started daemon to begin answering.
const DAEMON_STARTUP_TIMEOUT: Duration = Duration::from_secs(10);
/// Connect retry interval while the daemon is starting, and the poll
/// interval for every restart wait.
const DAEMON_STARTUP_POLL: Duration = Duration::from_millis(50);
/// How long a restart-following call waits for the next daemon. A drain
/// blocks on the longest running turn, so the bound is generous.
const RESTART_FOLLOW: Duration = Duration::from_secs(3600);

/// Tools that only read daemon state: a call dropped mid-flight by an
/// exiting daemon is safe to re-issue against the next one.
const IDEMPOTENT: &[&str] = &[
    "wait",
    "wait_any",
    "status",
    "result",
    "transcript",
    "list",
    "agents",
    "daemon/resumed",
];
/// Tools that start work: refused with `restarting` during a drain, they
/// wait for the next daemon and retry once.
const RETRY_RESTART: &[&str] = &["spawn", "send", "adopt"];

/// Call `tool` on the daemon at `socket` with `arguments` and return its
/// result.
///
/// When no daemon answers the socket, one is started automatically
/// (`acpsub daemon --socket …`), so the CLI works with no setup step.
/// `config` is forwarded to the daemon when it has to be started for this
/// call.
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
    let deadline = std::time::Instant::now() + RESTART_FOLLOW;
    let mut restarted = false;
    loop {
        match call_once(socket, tool, &arguments, config).await {
            Ok(result) => {
                // A turn-starting call refused by a draining daemon waits
                // for the replacement and retries once — the spawn/send
                // does not fail, it lands on the next daemon.
                if RETRY_RESTART.contains(&tool)
                    && !restarted
                    && is_restarting(&result)
                    && std::time::Instant::now() < deadline
                {
                    restarted = true;
                    wait_for_next_daemon(socket).await?;
                    continue;
                }
                return Ok(result);
            }
            // A read-only call dropped by an exiting daemon is a restart,
            // not a failure: re-issue it at once — the next connect lands
            // in the old listener's backlog while it exits, or reaches —
            // or auto-starts — the replacement. A call that never reached
            // the daemon stays an error.
            Err(Error::DaemonDisconnected { .. })
                if IDEMPOTENT.contains(&tool) && std::time::Instant::now() < deadline => {}
            Err(error) => return Err(error),
        }
    }
}

/// Whether the tool result is the draining daemon's `restarting`
/// refusal. The wire carries tool errors as text only, so the refusal is
/// a JSON object whose `error` discriminant is checked exactly — a
/// coincidental "restarting" in an agent's message does not count.
fn is_restarting(result: &CallToolResult) -> bool {
    result.is_error
        && result
            .content
            .iter()
            .any(|content| matches!(content, Content::Text(text) if restarting_marker(&text.text)))
}

/// The structured refusal's discriminant check.
fn restarting_marker(text: &str) -> bool {
    serde_json::from_str::<Value>(text)
        .ok()
        .as_ref()
        .and_then(|value| value.get("error"))
        .and_then(Value::as_str)
        == Some("restarting")
}

/// One connect-handshake-call round against the daemon at `socket`.
async fn call_once(
    socket: &Path,
    tool: &str,
    arguments: &Value,
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
        .map_err(|error| disconnected(socket, error))?;
    transport
        .notify(JsonRpcNotification::new("notifications/initialized"))
        .await
        .map_err(|error| disconnected(socket, error))?;
    let params = CallToolParams {
        name: tool.to_string(),
        arguments: arguments.clone(),
        meta: Some(RequestMeta {
            progress_token: Some(ProgressToken::String("acpsub-cli".to_string())),
        }),
    };
    let response = transport
        .request(JsonRpcRequest::with_params(1i64, "tools/call", params))
        .await
        .map_err(|error| disconnected(socket, error))?;
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

/// Wait until the daemon generation serving `socket` changes.
///
/// The draining daemon's exit closes every client stream: a fresh
/// connection's read end hits EOF the moment the process is gone — the
/// real signal, not a poll. The connect can also land on a replacement
/// another caller already auto-started: the daemon writes its pid file
/// before binding, so a changed pid means this stream reached the next
/// generation and there is nothing left to wait for. When the pid file
/// cannot tell, the stream is awaited anyway — a misattributed peer only
/// costs the deadline.
///
/// # Errors
///
/// Returns [`Error::DaemonConnect`] on a socket error, or an error when
/// the daemon is still draining at the deadline.
pub async fn wait_for_next_daemon(socket: &Path) -> Result<()> {
    let pid_path = socket.with_extension("sock.pid");
    let was = daemon_pid(&pid_path);
    let mut stream = match async_net::unix::UnixStream::connect(socket).await {
        Ok(stream) => stream,
        // Nothing answers: the draining daemon already exited.
        Err(source)
            if matches!(
                source.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
            ) =>
        {
            return Ok(());
        }
        Err(source) => {
            return Err(Error::DaemonConnect {
                path: socket.to_path_buf(),
                source,
            });
        }
    };
    if daemon_pid(&pid_path).is_some_and(|now| was.is_some_and(|was| now != was)) {
        return Ok(());
    }
    // The server never writes unprompted, so a completed read means the
    // stream ended — the daemon is gone.
    let mut byte = [0_u8; 1];
    match tokio::time::timeout(RESTART_FOLLOW, stream.read(&mut byte)).await {
        Err(_deadline) => Err(Error::io(
            format!("the daemon at {} is still draining", socket.display()),
            std::io::Error::other("drain deadline"),
        )),
        // EOF, reset, or a byte the daemon never sends — the stream
        // resolved, the draining daemon is gone.
        Ok(_) => Ok(()),
    }
}

/// The pid a daemon's pid file currently names, when it is readable.
fn daemon_pid(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|pid| pid.trim().parse().ok())
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

/// A transport failure mid-call: the connection is gone — a draining
/// daemon's exit is the common cause.
fn disconnected(socket: &Path, error: McpError) -> Error {
    Error::DaemonDisconnected {
        path: socket.to_path_buf(),
        source: std::io::Error::other(error),
    }
}
