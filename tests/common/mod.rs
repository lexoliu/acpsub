//! Shared helpers for the integration tests.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use acpsub::config::{AgentConfig, SteerSemantics};
use acpsub::{AppState, Config};
use aither_core::llm::tool::{ToolContext, ToolResult, Tools};
use serde_json::{Value, json};

/// Run `command` until it exits, its stdout and stderr captured in
/// anonymous temp files. The client's exit is the completion signal, not
/// EOF on a pipe: on macOS std creates a `Command`'s pipes with `pipe()`
/// and marks them close-on-exec only afterwards, so a client spawned at
/// that moment on another thread inherits the write end and holds the
/// pipe open — and with it `output()` — until it exits, or for good once
/// it starts a daemon. A `restart` racing a blocked `wait` stalled that
/// way until the wait ran into its expectation.
pub fn run_to_exit(mut command: Command) -> std::io::Result<Output> {
    let mut stdout = tempfile::tempfile()?;
    let mut stderr = tempfile::tempfile()?;
    let status = command
        .stdin(Stdio::null())
        .stdout(stdout.try_clone()?)
        .stderr(stderr.try_clone()?)
        .status()?;
    // The child wrote through a shared file description: rewind to read
    // what it wrote.
    let captured = |file: &mut File| -> std::io::Result<Vec<u8>> {
        file.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(bytes)
    };
    Ok(Output {
        status,
        stdout: captured(&mut stdout)?,
        stderr: captured(&mut stderr)?,
    })
}

/// Path to the fake agent script.
pub fn fake_agent_script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fake_agent.py")
}

/// Whether `python3` is available; python-based tests skip otherwise.
pub fn python3() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// The `fake` agent config: `python3 tests/fake_agent.py` (loadSession,
/// accepts `set_mode`/`set_config_option`).
pub fn fake_agent_config() -> AgentConfig {
    AgentConfig {
        command: "python3".to_string(),
        args: vec![fake_agent_script().to_string_lossy().into_owned()],
        env: BTreeMap::new(),
        allow_outside_cwd: false,
        permission: None,
        steer: SteerSemantics::Answered,
        sessions_db: None,
    }
}

/// A rate-limit agent whose provider window (3600 s) outlasts any test,
/// and whose error answer holds until `gate` exists: the test decides when
/// the awaited turn ends, never a delay, and a parked session is released
/// by the test's own event. A prompt `gated <path>` holds its turn open
/// until that file exists.
pub fn gated_limit_agent(gate: &Path) -> AgentConfig {
    AgentConfig {
        env: BTreeMap::from([
            ("FAKE_RATE_LIMIT".to_string(), "3600".to_string()),
            (
                "FAKE_RATE_LIMIT_GATE".to_string(),
                gate.to_string_lossy().into_owned(),
            ),
        ]),
        ..fake_agent_config()
    }
}

/// `gated_limit_agent` plus a hold on the limit itself: while `hold` is
/// absent every prompt errors again — each resume the park schedules is
/// answered with the limit still standing — so the parked phase ends on
/// the test's event, never on a window's end.
pub fn held_limit_agent(gate: &Path, hold: &Path) -> AgentConfig {
    let mut agent = gated_limit_agent(gate);
    agent.env.insert(
        "FAKE_RATE_LIMIT_HOLD".to_string(),
        hold.to_string_lossy().into_owned(),
    );
    agent
}

/// Write `dir/config.toml` for `agents` under `[defaults]` pointing at
/// `dir` — `wait_grace` seconds when given — and return the config
/// file's path.
pub fn write_config(
    dir: &Path,
    default_agent: Option<&str>,
    agents: &BTreeMap<String, AgentConfig>,
    wait_grace: Option<f64>,
) -> PathBuf {
    let mut text = format!(
        "[defaults]\npermission = \"allow\"\ntranscript_dir = \"{}\"\nregistry = \"{}\"\n",
        dir.join("transcripts").display(),
        dir.join("registry.json").display(),
    );
    if let Some(wait_grace) = wait_grace {
        let _ = writeln!(text, "wait_grace = {wait_grace}");
    }
    if let Some(agent) = default_agent {
        let _ = writeln!(text, "agent = \"{agent}\"");
    }
    for (name, agent) in agents {
        let _ = write!(text, "\n[agents.{name}]\ncommand = {:?}", agent.command);
        if !agent.args.is_empty() {
            let args = agent
                .args
                .iter()
                .map(|arg| format!("{arg:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = write!(text, "\nargs = [{args}]");
        }
        if !agent.env.is_empty() {
            let env = agent
                .env
                .iter()
                .map(|(key, value)| format!("{key} = {value:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = write!(text, "\nenv = {{ {env} }}");
        }
        if agent.allow_outside_cwd {
            text.push_str("\nallow_outside_cwd = true");
        }
        if let Some(permission) = agent.permission {
            let name = serde_json::to_value(permission)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            let _ = write!(text, "\npermission = \"{name}\"");
        }
        if agent.steer != SteerSemantics::Answered {
            let name = serde_json::to_value(agent.steer)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            let _ = write!(text, "\nsteer = \"{name}\"");
        }
        if let Some(db) = &agent.sessions_db {
            let _ = write!(text, "\nsessions_db = {:?}", db.display().to_string());
        }
        text.push('\n');
    }
    let path = dir.join("config.toml");
    std::fs::write(&path, text).expect("write config");
    path
}

/// Build an app state rooted at `dir`, with a `fake` agent (loadSession,
/// accepts `set_mode`/`set_config_option`), a `noload` agent (same script
/// with `FAKE_NO_LOAD=1`), a `nosteer` agent (`FAKE_NO_STEER=1`, rejects a
/// concurrent prompt), a `queueagent` (`FAKE_QUEUE_PROMPTS=1`, parks a
/// concurrent prompt and runs it as its own turn), a `foldagent`
/// (`FAKE_FOLD_STEER=1` and `steer = "folded"`, folds a concurrent prompt
/// into the running turn and never answers it), two rate-limit agents
/// (`ratelimit` answers "ratelimit" prompts — then every prompt until a
/// ~4 s window lifts — with the structured -32010 shape whose message
/// clause names the reset; `vaguelimit` omits the clause, leaving no
/// readable reset at all), and a `wideopen` agent allowed outside its
/// cwd.
pub fn test_state(dir: &Path) -> (Arc<AppState>, Tools) {
    let mut agents = BTreeMap::new();
    let fake = fake_agent_config();
    agents.insert("fake".to_string(), fake.clone());
    agents.insert(
        "ratelimit".to_string(),
        AgentConfig {
            env: BTreeMap::from([("FAKE_RATE_LIMIT".to_string(), "4".to_string())]),
            ..fake.clone()
        },
    );
    agents.insert(
        "vaguelimit".to_string(),
        AgentConfig {
            env: BTreeMap::from([
                ("FAKE_RATE_LIMIT".to_string(), "4".to_string()),
                ("FAKE_RATE_LIMIT_VAGUE".to_string(), "1".to_string()),
            ]),
            ..fake.clone()
        },
    );
    agents.insert(
        "noload".to_string(),
        AgentConfig {
            env: BTreeMap::from([("FAKE_NO_LOAD".to_string(), "1".to_string())]),
            ..fake.clone()
        },
    );
    agents.insert(
        "nosteer".to_string(),
        AgentConfig {
            env: BTreeMap::from([("FAKE_NO_STEER".to_string(), "1".to_string())]),
            ..fake.clone()
        },
    );
    agents.insert(
        "queueagent".to_string(),
        AgentConfig {
            env: BTreeMap::from([("FAKE_QUEUE_PROMPTS".to_string(), "1".to_string())]),
            ..fake.clone()
        },
    );
    agents.insert(
        "foldagent".to_string(),
        AgentConfig {
            env: BTreeMap::from([("FAKE_FOLD_STEER".to_string(), "1".to_string())]),
            steer: SteerSemantics::Folded,
            ..fake.clone()
        },
    );
    agents.insert(
        "wideopen".to_string(),
        AgentConfig {
            allow_outside_cwd: true,
            ..fake
        },
    );
    test_state_with_agents(dir, &agents)
}

/// Build an app state rooted at `dir` over an explicit agent map.
pub fn test_state_with_agents(
    dir: &Path,
    agents: &BTreeMap<String, AgentConfig>,
) -> (Arc<AppState>, Tools) {
    test_state_full(dir, None, agents)
}

/// Build an app state with an explicit default agent (or none).
pub fn test_state_full(
    dir: &Path,
    default_agent: Option<&str>,
    agents: &BTreeMap<String, AgentConfig>,
) -> (Arc<AppState>, Tools) {
    let path = write_config(dir, default_agent, agents, None);
    let config = Config::load(&path).expect("config loads");
    let state = AppState::new(config, path).expect("app state");
    // The test-harness constructor: a short report interval so progress
    // tests see several reports inside a test-length wait.
    let tools =
        acpsub::build_tools_with_progress_interval(state.clone(), Duration::from_millis(50))
            .expect("tools");
    (state, tools)
}

/// Call a tool; errors surface as `Err(message)`.
pub async fn call(tools: &Tools, name: &str, args: Value) -> Result<ToolResult, String> {
    call_with(tools, name, args, ToolContext::new()).await
}

/// Call a tool with an explicit context — progress tests attach a
/// listening sink.
pub async fn call_with(
    tools: &Tools,
    name: &str,
    args: Value,
    cx: ToolContext,
) -> Result<ToolResult, String> {
    tools
        .call(name, &args.to_string(), cx)
        .await
        .map_err(|error| error.to_string())
}

/// Call a tool and unwrap a non-error result as JSON — asserts the
/// tool's `json` form, which is what every assertion here describes.
pub async fn call_json(tools: &Tools, name: &str, mut args: Value) -> Value {
    args["json"] = json!(true);
    let result = call(tools, name, args)
        .await
        .unwrap_or_else(|e| panic!("{name} failed: {e}"));
    assert!(
        !result.is_error(),
        "{name} returned error: {}",
        result.error_message().unwrap_or("?")
    );
    let text = result.render_for_model().expect("render");
    serde_json::from_str(&text).expect("tool result is json")
}

/// Call a tool and return the error message.
pub async fn call_err(tools: &Tools, name: &str, args: Value) -> String {
    match call(tools, name, args).await {
        Err(error) => error,
        Ok(result) if result.is_error() => {
            result.error_message().expect("error message").to_string()
        }
        Ok(result) => panic!(
            "expected error, got {}",
            result.render_for_model().unwrap_or_default()
        ),
    }
}

/// Call a tool returning plain text.
pub async fn call_text(tools: &Tools, name: &str, args: Value) -> String {
    let result = call(tools, name, args).await.expect("call failed");
    result.render_for_model().expect("render")
}

/// Spawn args for the fake agent rooted at `cwd`.
pub fn spawn_args(cwd: &Path, prompt: &str) -> Value {
    json!({
        "agent": "fake",
        "cwd": cwd,
        "prompt": prompt,
        "model": "b",
        "mode": "bypass",
    })
}

/// Spawn and return the assigned session id.
pub async fn spawn_id(tools: &Tools, args: Value) -> String {
    let spawned = call_json(tools, "spawn", args).await;
    spawned["session_id"]
        .as_str()
        .expect("session_id")
        .to_string()
}

/// `wait` with an expected turn duration.
pub async fn wait(tools: &Tools, session_id: &str, expect_secs: u64) -> Value {
    call_json(
        tools,
        "wait",
        json!({"session_id": session_id, "expect_secs": expect_secs}),
    )
    .await
}

/// How long teardown waits for a daemon to die, or for the clients a
/// test left in flight to give up — each gives up within the client's
/// own daemon-startup window once no daemon can serve.
const TEARDOWN: Duration = Duration::from_secs(30);
/// Retry interval while a signalled daemon still holds the socket's
/// claim: its exit has no event a test can block on without risking a
/// wait on a newer holder.
const TEARDOWN_POLL: Duration = Duration::from_millis(10);

/// Owns every daemon that serves `socket` — the one the test started
/// and any a client auto-started behind it, during a restart or after a
/// crash — and stops them all on drop: pass, fail or panic.
///
/// A daemon holds an exclusive lock on its pid file for as long as it
/// lives, and the file names it. Teardown takes that lock itself,
/// terminating each holder the file names, then keeps it until every
/// client the test left in flight has exited: an auto-start one of them
/// attempts meanwhile finds the claim taken and exits on its own.
#[derive(Debug)]
pub struct DaemonGuard {
    socket: PathBuf,
    /// The daemon the test spawned itself, reaped once the claim is held.
    child: Option<std::process::Child>,
    /// Cloned into each in-flight client; the channel disconnects once
    /// every clone is gone.
    in_flight: Option<std::sync::mpsc::Sender<std::convert::Infallible>>,
    clients_done: std::sync::mpsc::Receiver<std::convert::Infallible>,
}

/// Held by a client the test runs off its own thread, for as long as
/// that client runs.
pub struct ClientInFlight {
    _token: std::sync::mpsc::Sender<std::convert::Infallible>,
}

impl DaemonGuard {
    /// Guard `socket`, owning `child` when the test spawned the daemon
    /// itself.
    pub fn new(socket: PathBuf, child: Option<std::process::Child>) -> Self {
        let (in_flight, clients_done) = std::sync::mpsc::channel();
        Self {
            socket,
            child,
            in_flight: Some(in_flight),
            clients_done,
        }
    }

    /// A token a client running off the test thread holds until it
    /// exits: a client a panic leaves running can still auto-start a
    /// daemon, so teardown waits for it under the claim.
    pub fn in_flight(&self) -> ClientInFlight {
        ClientInFlight {
            _token: self.in_flight.clone().expect("guard is live"),
        }
    }

    /// Take the socket's claim — the lock on its pid file — terminating
    /// each daemon that holds it. A holder that just took the lock may
    /// not have written its pid yet, so the file is re-read until it
    /// names one; each pid is signalled once.
    fn claim(&self) -> std::fs::File {
        let path = self.socket.with_extension("sock.pid");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .expect("open daemon pid file");
        let deadline = Instant::now() + TEARDOWN;
        let mut signalled = None;
        loop {
            match file.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) => {}
                Err(std::fs::TryLockError::Error(error)) => {
                    panic!("cannot lock {}: {error}", path.display())
                }
            }
            let holder = std::fs::read_to_string(&path)
                .expect("read daemon pid file")
                .trim()
                .parse::<i32>()
                .ok();
            if let Some(pid) = holder
                && signalled != Some(pid)
            {
                // SAFETY: kill(2) only delivers a signal; the pid is the
                // lock holder's own record of itself.
                unsafe { libc::kill(pid, libc::SIGTERM) };
                signalled = Some(pid);
            }
            assert!(
                Instant::now() < deadline,
                "daemon {holder:?} still holds {}",
                path.display()
            );
            std::thread::sleep(TEARDOWN_POLL);
        }
        // No daemon serves the socket now: the file names none.
        file.set_len(0).expect("clear daemon pid file");
        file
    }
}

impl DaemonGuard {
    /// Terminate every superseded generation still serving the socket's
    /// sessions — each one binds `<socket stem>.<pid>.sock` beside the
    /// socket for as long as it runs, after giving its claim up.
    fn stop_superseded(&self) {
        let dir = self.socket.parent().expect("socket has a directory");
        let stem = self
            .socket
            .file_stem()
            .and_then(|stem| stem.to_str())
            .expect("socket stem");
        for entry in std::fs::read_dir(dir).expect("read socket directory") {
            let name = entry.expect("directory entry").file_name();
            let pid = name
                .to_str()
                .and_then(|name| name.strip_prefix(stem))
                .and_then(|rest| rest.strip_prefix('.'))
                .and_then(|rest| rest.strip_suffix(".sock"))
                .and_then(|pid| pid.parse::<i32>().ok());
            if let Some(pid) = pid {
                // SAFETY: kill(2) only delivers a signal; the pid is the
                // generation's own record of itself.
                unsafe { libc::kill(pid, libc::SIGTERM) };
            }
        }
    }
}

impl Drop for DaemonGuard {
    fn drop(&mut self) {
        let claim = self.claim();
        // Every client still in flight finds no daemon and none it
        // starts can take the claim, so each gives up and exits.
        self.in_flight = None;
        match self.clients_done.recv_timeout(TEARDOWN) {
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                panic!("clients still in flight on {}", self.socket.display())
            }
            Ok(never) => match never {},
        }
        self.stop_superseded();
        if let Some(child) = &mut self.child {
            // Already terminated when it held the claim; killed here when
            // it never took it.
            let _ = child.kill();
            child.wait().expect("reap test daemon");
        }
        drop(claim);
    }
}

/// Block on `rt.notify` until the live subagent's status is `want`. For
/// transitions that fire off the turn path — e.g. the `exited` mark a dead
/// agent's connection task stamps, which a `wait` call may race — where a
/// single `wait` result cannot be relied on. The timeout only turns a hang
/// into a failure.
pub async fn status_becomes(state: &AppState, session_id: &str, want: &str) {
    let sub = state.get(session_id).expect("live subagent");
    let reached = async {
        loop {
            let notified = sub.rt.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if sub.rt.inner.lock().expect("inner poisoned").status.name() == want {
                return;
            }
            notified.await;
        }
    };
    tokio::time::timeout(Duration::from_secs(30), reached)
        .await
        .unwrap_or_else(|_| panic!("session {session_id} never reached '{want}'"));
}
