//! The CLI's path to the daemon: one MCP `tools/call` per invocation.
//!
//! A CLI command connects to the daemon's Unix socket, performs the MCP
//! handshake, calls a single tool, prints the result, and exits. Every
//! call carries a `progressToken`, so a blocking `wait` is never capped
//! by the no-listener `expect_secs` ceiling — the client simply discards
//! the progress notifications; on a terminal or in a background task the
//! process exit itself is the signal.
//!
//! A read-only call whose connection a superseded daemon's exit closes
//! is not a failure: it reconnects to the next daemon and re-issues the
//! call — a `wait` keeps its `expect_secs` budget because the budget
//! measures from the turn's recorded start, and a turn that already
//! ended answers from the resumed session's restored state.

use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use aither_mcp::protocol::{
    CallToolParams, CallToolResult, InitializeParams, JsonRpcNotification, JsonRpcRequest,
    McpError, ProgressToken, RequestMeta,
};
use aither_mcp::transport::{StreamTransport, Transport};
use futures_lite::io::BufReader;
use serde::Deserialize;
use serde_json::Value;
use tracing::debug;

use crate::error::{Error, Result};

/// How long to wait for an auto-started daemon to begin answering.
const DAEMON_STARTUP_TIMEOUT: Duration = Duration::from_secs(10);
/// Connect retry interval while the daemon is starting.
const DAEMON_STARTUP_POLL: Duration = Duration::from_millis(50);

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
    // The instance that dropped this call, how it dropped it, and when.
    // The call is re-issued only when a different instance answers the
    // retry — a restart. The same instance still serving means a live
    // daemon dropped a call it stays up to drop — a handler panic, not an
    // exit — and re-issuing would loop on the same failure.
    let mut dropped: Option<Dropped> = None;
    loop {
        let (mut transport, instance) = match handshake(socket, config).await {
            Ok(pair) => pair,
            // The connection died before `initialize` could name its
            // daemon. That is the gap between generations only while a
            // dropper is on its way out, and only for as long as a
            // replacement takes to start; anything else is a daemon that
            // fails every handshake.
            Err(Error::DaemonDisconnected { .. })
                if dropped
                    .as_ref()
                    .is_some_and(|drop| drop.at.elapsed() < DAEMON_STARTUP_TIMEOUT) =>
            {
                continue;
            }
            Err(error) => return Err(error),
        };
        // A read-only call dropped by an exiting daemon is a restart,
        // not a failure: re-issue it once the handshake names a new
        // instance — the next connect lands in the old listener's
        // backlog while it exits, or reaches — or auto-starts — the
        // replacement. The same instance answering again means the
        // dropper never left: the call is refused before it is sent.
        if let Some(drop) = dropped.take_if(|drop| drop.instance == instance) {
            return Err(Error::DaemonDroppedCall {
                path: socket.to_path_buf(),
                tool: tool.to_string(),
                source: drop.source,
            });
        }
        match call_tool(&mut transport, socket, tool, &arguments).await {
            Err(Error::DaemonDisconnected { source, .. }) if IDEMPOTENT.contains(&tool) => {
                dropped = Some(Dropped {
                    instance,
                    source,
                    at: Instant::now(),
                });
            }
            result => return result,
        }
    }
}

/// An idempotent call the daemon dropped: which instance, the transport
/// error, and when — the start of the window a restart may take.
struct Dropped {
    instance: ServerIdentity,
    source: std::io::Error,
    at: Instant,
}

/// The daemon instance a connection's `initialize` answer reports —
/// `serverInfo`. A daemon reports the same identity on every connection
/// it serves; its version carries the pid as semver build metadata
/// (`<version>+daemon.<pid>`), so a restarted daemon is another
/// identity.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ServerIdentity {
    /// `serverInfo.name`.
    name: String,
    /// `serverInfo.version`.
    #[serde(default)]
    version: Option<String>,
}

/// The client end of one daemon connection.
type DaemonConn = StreamTransport<
    BufReader<futures_lite::io::ReadHalf<async_net::unix::UnixStream>>,
    futures_lite::io::WriteHalf<async_net::unix::UnixStream>,
>;

/// Connect to the daemon at `socket` and run `initialize`: the transport
/// and the instance `serverInfo` reports. A daemon always names itself,
/// so an answer without a readable `serverInfo` is an error.
async fn handshake(socket: &Path, config: Option<&Path>) -> Result<(DaemonConn, ServerIdentity)> {
    let stream = connect_or_start(socket, config).await?;
    let (read_half, write_half) = futures_lite::io::split(stream);
    let mut transport = StreamTransport::new(BufReader::new(read_half), write_half);
    let response = transport
        .request(JsonRpcRequest::with_params(
            0i64,
            "initialize",
            InitializeParams::default(),
        ))
        .await
        .map_err(|error| disconnected(socket, error))?;
    let identity = response
        .into_result()
        .map_err(|error| format!("initialize failed: {error:?}"))
        .and_then(|result| {
            serde_json::from_value(result["serverInfo"].clone())
                .map_err(|error| format!("unreadable serverInfo: {error}"))
        })
        .map_err(|detail| Error::DaemonIdentity {
            path: socket.to_path_buf(),
            detail,
        })?;
    Ok((transport, identity))
}

/// `notifications/initialized` then the `tools/call`, on a handshaken
/// connection.
async fn call_tool(
    transport: &mut DaemonConn,
    socket: &Path,
    tool: &str,
    arguments: &Value,
) -> Result<CallToolResult> {
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
            // Detached: dropping the handle leaves the daemon running.
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
///
/// # Errors
///
/// Returns an error if the daemon process cannot be spawned.
pub fn start_daemon(socket: &Path, config: Option<&Path>) -> Result<std::process::Child> {
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
        .map_err(|source| Error::io("cannot spawn acpsub daemon", source))
}

/// A transport failure mid-call: the connection is gone — a superseded
/// daemon's exit is the common cause.
fn disconnected(socket: &Path, error: McpError) -> Error {
    Error::DaemonDisconnected {
        path: socket.to_path_buf(),
        source: std::io::Error::other(error),
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::time::Duration;

    use serde_json::json;
    use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

    use super::*;

    /// What a fake daemon reports back to its test, per connection.
    #[derive(Debug, PartialEq, Eq)]
    enum FakeEvent {
        /// `initialize` answered.
        Handshaken,
        /// `tools/call` received — what follows is the connection's
        /// `Answer`.
        Called,
    }

    /// What one connection's fake daemon does.
    #[derive(Clone)]
    enum Answer {
        /// Handshake, then close on `tools/call` — a handler panic: the
        /// daemon stays up, the call is gone.
        DropCall,
        /// Handshake, then answer `tools/call` with this result document.
        Result(Value),
        /// Close the connection at once — the drained backlog of a daemon
        /// that is on its way out.
        Close,
    }

    /// A fake daemon bound at a path: every `initialize` is answered with
    /// the given `serverInfo` — the instance identity `call` compares —
    /// and each connection then behaves as its `Answer` says, in connect
    /// order, the last one repeating once they run out.
    struct FakeDaemon {
        /// What the fake did, in order.
        events: UnboundedReceiver<FakeEvent>,
    }

    impl FakeDaemon {
        /// Bind `path` and serve `answers` — each `initialize` reports
        /// `serverInfo` `{name, version}`.
        fn bind(path: &Path, name: &str, version: &str, answers: &[Answer]) -> Self {
            assert!(!answers.is_empty(), "a fake daemon needs an answer");
            let listener = UnixListener::bind(path).expect("bind fake daemon");
            let (tx, events) = unbounded_channel();
            let answers = answers.to_vec();
            let name = name.to_string();
            let version = version.to_string();
            std::thread::spawn(move || {
                for (count, stream) in listener.incoming().enumerate() {
                    let Ok(stream) = stream else { continue };
                    let answer = answers[count.min(answers.len() - 1)].clone();
                    let (tx, name, version) = (tx.clone(), name.clone(), version.clone());
                    std::thread::spawn(move || serve_conn(stream, &name, &version, &answer, &tx));
                }
            });
            Self { events }
        }

        /// Assert the next reported event is `want`, bounded.
        async fn next(&mut self, want: FakeEvent) {
            let event = tokio::time::timeout(Duration::from_secs(10), self.events.recv())
                .await
                .expect("fake daemon event never arrived")
                .expect("fake daemon stopped reporting");
            assert_eq!(event, want);
        }

        /// The events buffered so far — for checking nothing followed.
        fn drained(&mut self) -> Vec<FakeEvent> {
            let mut seen = Vec::new();
            while let Ok(event) = self.events.try_recv() {
                seen.push(event);
            }
            seen
        }
    }

    /// One fake-daemon connection: the MCP handshake, then the `Answer`.
    fn serve_conn(
        stream: UnixStream,
        name: &str,
        version: &str,
        answer: &Answer,
        events: &UnboundedSender<FakeEvent>,
    ) {
        // A dying daemon's backlog entry: accepted, gone at once.
        if matches!(answer, Answer::Close) {
            return;
        }
        let mut writer = stream.try_clone().expect("clone stream");
        let mut reader = BufReader::new(stream);
        let mut respond = |id: &Value, result: Value| {
            let mut line = json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string();
            line.push('\n');
            writer.write_all(line.as_bytes()).expect("write response");
        };
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => return,
                Ok(_) => {}
            }
            let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            let Some(id) = msg.get("id").cloned() else {
                continue;
            };
            match msg["method"].as_str() {
                Some("initialize") => {
                    let _ = events.send(FakeEvent::Handshaken);
                    respond(
                        &id,
                        json!({
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "serverInfo": {"name": name, "version": version},
                        }),
                    );
                }
                Some("tools/call") => {
                    let _ = events.send(FakeEvent::Called);
                    match answer {
                        // The panic: the daemon lives, the call is gone.
                        Answer::DropCall => return,
                        Answer::Result(result) => respond(&id, result.clone()),
                        Answer::Close => unreachable!("checked on entry"),
                    }
                }
                _ => {}
            }
        }
    }

    /// A live daemon that drops an idempotent call and stays up answers
    /// the retry's handshake with its own identity: `call` fails with
    /// `DaemonDroppedCall` naming the tool, and never sends the second
    /// `tools/call` — the same instance dropping it again would be the
    /// same failure forever.
    #[tokio::test]
    async fn same_instance_drop_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let socket = dir.path().join("daemon.sock");
        let mut daemon = FakeDaemon::bind(&socket, "acpsub", "0.0.0+daemon.1", &[Answer::DropCall]);

        let error = call(&socket, "list", json!({}), None)
            .await
            .expect_err("a call dropped by a live daemon must fail");
        match &error {
            Error::DaemonDroppedCall { tool, .. } => assert_eq!(tool, "list"),
            other => panic!("expected DaemonDroppedCall, got {other}"),
        }
        assert!(error.to_string().contains("list"), "{error}");

        // The retry stopped at the handshake: one `initialize` on the
        // second connection, never a second `tools/call`.
        assert_eq!(
            daemon.drained(),
            [
                FakeEvent::Handshaken,
                FakeEvent::Called,
                FakeEvent::Handshaken,
            ]
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(500), daemon.events.recv())
                .await
                .is_err(),
            "a third connection reached the daemon"
        );
    }

    /// A call dropped by an exiting daemon completes against its
    /// replacement: the retry's handshake reports another instance, so
    /// `call` re-issues. The replacement takes the socket the way a real
    /// restart does — bound on a sibling path, renamed over the public
    /// one — while the old daemon's remaining backlog closes connections
    /// before the handshake.
    #[tokio::test]
    async fn drop_by_exiting_daemon_completes_against_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let socket = dir.path().join("daemon.sock");
        let mut old = FakeDaemon::bind(
            &socket,
            "acpsub",
            "0.0.0+daemon.1",
            &[Answer::DropCall, Answer::Close],
        );
        let task = {
            let socket = socket.clone();
            tokio::spawn(async move { call(&socket, "list", json!({}), None).await })
        };

        // The drop is the restart's start: the replacement binds a
        // sibling and takes the public socket by rename.
        old.next(FakeEvent::Handshaken).await;
        old.next(FakeEvent::Called).await;
        let next = dir.path().join("next.sock");
        let mut new = FakeDaemon::bind(
            &next,
            "acpsub",
            "0.0.0+daemon.2",
            &[Answer::Result(json!({
                "content": [{"type": "text", "text": "{\"subagents\":[]}"}],
                "isError": false,
            }))],
        );
        std::fs::rename(&next, &socket).expect("take the socket over");

        let result = task
            .await
            .expect("call task panicked")
            .expect("call completes against the replacement");
        let text = serde_json::to_value(&result).expect("result serializes");
        assert!(text.to_string().contains("subagents"), "{text}");

        // The retry reached the replacement exactly once: handshake,
        // then the re-issued call it answered.
        new.next(FakeEvent::Handshaken).await;
        new.next(FakeEvent::Called).await;
        // The old daemon saw the one call it dropped and nothing more:
        // every later connection closed before the handshake.
        assert_eq!(old.drained(), []);
    }
}
