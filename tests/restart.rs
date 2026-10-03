//! Graceful daemon restart through the socket: `acpsub daemon` drains its
//! in-flight turns, marks every live session for resume, exits, and the
//! auto-started replacement re-adopts them — while client calls in flight
//! across the handoff reconnect and retry instead of failing.

#[expect(
    dead_code,
    reason = "helpers shared across test binaries; this one uses a subset"
)]
mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::*;
use serde_json::Value;

/// The agent set the restart tests need: `fake` (loadSession), `noload`
/// (same script with `FAKE_NO_LOAD=1`, so `session/load` is refused) and
/// `slowlimit` (every "ratelimit" prompt parks the turn after a 3 s delay
/// and keeps the session parked for an 8 s window — long enough to drain
/// and restart inside it).
fn agents() -> BTreeMap<String, acpsub::config::AgentConfig> {
    let fake = fake_agent_config();
    let mut agents = BTreeMap::new();
    agents.insert("fake".to_string(), fake.clone());
    agents.insert(
        "noload".to_string(),
        acpsub::config::AgentConfig {
            env: BTreeMap::from([("FAKE_NO_LOAD".to_string(), "1".to_string())]),
            ..fake.clone()
        },
    );
    agents.insert(
        "slowlimit".to_string(),
        acpsub::config::AgentConfig {
            env: BTreeMap::from([
                ("FAKE_RATE_LIMIT".to_string(), "8".to_string()),
                ("FAKE_RATE_LIMIT_DELAY".to_string(), "3".to_string()),
            ]),
            ..fake
        },
    );
    agents
}

/// A running `acpsub daemon` on a tempdir socket; killed on drop — which
/// covers its restarted replacement too, through the pidfile.
struct Daemon {
    child: Child,
    socket: PathBuf,
    config: PathBuf,
    log: PathBuf,
}

impl Daemon {
    /// Write the test config into `dir` and start the daemon, waiting for
    /// the socket to exist — the real "listening" signal, never a client
    /// call (a client would auto-start a daemon and hide a startup
    /// failure).
    fn start(dir: &Path) -> Self {
        let config = write_config(dir, Some("fake"), &agents());
        let socket = dir.join("daemon.sock");
        let log = dir.join("daemon.log");
        let child = Command::new(env!("CARGO_BIN_EXE_acpsub"))
            .arg("daemon")
            .arg("--socket")
            .arg(&socket)
            .arg("--config")
            .arg(&config)
            .arg("--log-file")
            .arg(&log)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn acpsub daemon");
        let daemon = Self {
            child,
            socket,
            config,
            log,
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        while !daemon.socket.exists() {
            assert!(
                Instant::now() < deadline,
                "daemon never bound {}",
                daemon.socket.display()
            );
            std::thread::sleep(Duration::from_millis(50));
        }
        daemon
    }

    /// The daemon's pid — after a restart it is the replacement's.
    fn pid(&self) -> u32 {
        std::fs::read_to_string(self.socket.with_extension("sock.pid"))
            .expect("pidfile")
            .trim()
            .parse()
            .expect("pidfile is a pid")
    }

    /// Run a client subcommand against this daemon.
    fn run(&self, args: &[&str]) -> std::io::Result<std::process::Output> {
        Command::new(env!("CARGO_BIN_EXE_acpsub"))
            .args(args)
            .arg("--socket")
            .arg(&self.socket)
            .arg("--config")
            .arg(&self.config)
            .output()
    }

    /// Run a client subcommand in a thread and return its handle.
    fn run_thread(&self, args: &[&str]) -> std::thread::JoinHandle<std::process::Output> {
        let exe = PathBuf::from(env!("CARGO_BIN_EXE_acpsub"));
        let socket = self.socket.clone();
        let config = self.config.clone();
        let args: Vec<String> = args.iter().map(|arg| (*arg).to_string()).collect();
        std::thread::spawn(move || {
            Command::new(exe)
                .args(&args)
                .arg("--socket")
                .arg(&socket)
                .arg("--config")
                .arg(&config)
                .output()
                .expect("client runs")
        })
    }

    /// Run a client subcommand and return its stdout as JSON.
    fn call_json(&self, args: &[&str]) -> Value {
        let output = self.run(args).expect("client runs");
        assert!(
            output.status.success(),
            "{args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).expect("client output is json")
    }

    /// Poll `status` for the session until it reports `want` — for
    /// transitions a single call cannot wait on, like a rate-limit park
    /// or a scheduler-driven resume.
    fn status_becomes(&self, session_id: &str, want: &str) -> Value {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let status = self.call_json(&["status", session_id]);
            if status["state"] == want {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "session {session_id} never reached '{want}': {status}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Wait until the daemon log mentions `marker` — the real readiness
    /// signal that the drain began.
    fn wait_log(&self, marker: &str) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let text = std::fs::read_to_string(&self.log).unwrap_or_default();
            if text.contains(marker) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "daemon log never showed '{marker}':\n{text}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        // A restarted daemon replaced the original child; its pid is in
        // the pidfile. Kill it so no test daemon outlives the test.
        if self.socket.with_extension("sock.pid").exists() {
            let pid = self.pid().cast_signed();
            unsafe { libc::kill(pid, libc::SIGTERM) };
        }
    }
}

/// A drain with one running turn and one idle session: the turn finishes
/// with its result recorded, and both sessions come back live under the
/// same ids and owner.
#[test]
fn drain_resumes_sessions() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path());
    let owner = std::process::id().to_string();

    let running = daemon.call_json(&[
        "spawn",
        "--cwd",
        &dir.path().to_string_lossy(),
        "--model",
        "b",
        "--mode",
        "bypass",
        "--owner",
        &owner,
        "--prompt",
        "gather",
    ]);
    let a = running["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();

    let idle = daemon.call_json(&[
        "spawn",
        "--cwd",
        &dir.path().to_string_lossy(),
        "--model",
        "b",
        "--mode",
        "bypass",
        "--owner",
        &owner,
        "--prompt",
        "hi",
    ]);
    let b = idle["session_id"].as_str().expect("session_id").to_string();
    daemon.status_becomes(&b, "done");

    let restart = daemon.run_thread(&["restart"]);
    // The drain blocks on A's open turn until it is steered to an end.
    daemon.wait_log("drain requested");

    let steered = daemon.call_json(&["send", &a, "--policy", "steer", "--prompt", "go"]);
    assert_eq!(steered["state"], "steered", "{steered}");

    let output = restart.join().expect("restart thread");
    assert!(
        output.status.success(),
        "restart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).expect("restart output is json");
    let resumed: Vec<&str> = report["resumed"]
        .as_array()
        .expect("resumed")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    assert!(resumed.contains(&a.as_str()), "{report}");
    assert!(resumed.contains(&b.as_str()), "{report}");
    assert_eq!(
        report["failed"].as_array().expect("failed").len(),
        0,
        "{report}"
    );

    // Both sessions are live under the same ids; A's turn result answers
    // from the restored turns, and the owner carried over so the reaper
    // still covers them.
    for sid in [&a, &b] {
        let status = daemon.status_becomes(sid, "idle");
        assert_eq!(status["live"], true, "{status}");
        assert_eq!(status["turns"], 1, "{status}");
    }
    let result = daemon.call_json(&["result", &a, "--turn", "1"]);
    assert_eq!(result["stop_reason"], "end_turn", "{result}");
    assert!(
        result["reply"]
            .as_str()
            .expect("reply")
            .contains("steer:go"),
        "{result}"
    );

    let registry: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("registry.json")).unwrap())
            .expect("registry is json");
    for sid in [&a, &b] {
        assert_eq!(
            registry[sid]["owner"].as_u64().expect("owner"),
            u64::from(std::process::id()),
            "{registry}"
        );
    }
}

/// A `send` issued during the drain is refused with `restarting`; the
/// client waits for the next daemon and retries — it does not fail.
#[test]
fn send_during_drain_retries() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path());

    let running = daemon.call_json(&[
        "spawn",
        "--cwd",
        &dir.path().to_string_lossy(),
        "--model",
        "b",
        "--mode",
        "bypass",
        "--prompt",
        "gather",
    ]);
    let a = running["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();

    let idle = daemon.call_json(&[
        "spawn",
        "--cwd",
        &dir.path().to_string_lossy(),
        "--model",
        "b",
        "--mode",
        "bypass",
        "--prompt",
        "hi",
    ]);
    let b = idle["session_id"].as_str().expect("session_id").to_string();
    daemon.status_becomes(&b, "done");

    let restart = daemon.run_thread(&["restart"]);
    daemon.wait_log("drain requested");

    // Issued while the drain is in progress: the call is refused, waits
    // out the restart, and lands on the next daemon.
    let send = daemon.run_thread(&["send", &b, "--policy", "try", "--prompt", "after"]);

    let steered = daemon.call_json(&["send", &a, "--policy", "steer", "--prompt", "go"]);
    assert_eq!(steered["state"], "steered", "{steered}");

    let output = send.join().expect("send thread");
    assert!(
        output.status.success(),
        "send failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let retried: Value = serde_json::from_slice(&output.stdout).expect("send output is json");
    assert_eq!(retried["state"], "running", "{retried}");

    let done = daemon.call_json(&["wait", &b, "--expect", "60"]);
    assert_eq!(done["state"], "done", "{done}");
    assert_eq!(done["reply"], "Hello abworld", "{done}");

    let output = restart.join().expect("restart thread");
    assert!(output.status.success(), "restart failed");
}

/// A `wait` blocked across the restart treats the dropped connection as
/// a restart, not a failure: it reconnects to the next daemon and answers
/// with the awaited turn's recorded result.
#[test]
fn wait_survives_restart() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path());

    // A "ratelimit" prompt keeps the turn running ~3 s, then parks the
    // session for an 8 s window — the bound wait follows the park and
    // stays blocked through the drain and the exit.
    let spawned = daemon.call_json(&[
        "spawn",
        "--cwd",
        &dir.path().to_string_lossy(),
        "--agent",
        "slowlimit",
        "--model",
        "b",
        "--mode",
        "bypass",
        "--prompt",
        "ratelimit",
    ]);
    let a = spawned["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();

    let wait = daemon.run_thread(&["wait", &a, "--expect", "60"]);
    daemon.status_becomes(&a, "rate_limited");

    let restart = daemon.run_thread(&["restart"]);
    let output = restart.join().expect("restart thread");
    assert!(
        output.status.success(),
        "restart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The parked session is quiescent work — the drain did not wait for
    // its window — and it came back re-parked under the same id.
    let report: Value = serde_json::from_slice(&output.stdout).expect("restart output is json");
    assert!(
        report["resumed"]
            .as_array()
            .expect("resumed")
            .iter()
            .filter_map(Value::as_str)
            .any(|sid| sid == a),
        "{report}"
    );

    let output = wait.join().expect("wait thread");
    assert!(
        output.status.success(),
        "wait failed across the restart: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let done: Value = serde_json::from_slice(&output.stdout).expect("wait output is json");
    // The re-issued wait lands inside the still-open window and answers
    // with the awaited turn's recorded end.
    assert_eq!(done["state"], "rate_limited", "{done}");
    assert!(done["resume_at"].is_string(), "{done}");

    // And the park's own schedule kept running on the next daemon: the
    // scheduler resumes the session and the turn completes.
    daemon.status_becomes(&a, "done");
}

/// A session whose `session/load` fails is reported in the resume report
/// and is not silently dropped from the registry.
#[test]
fn failed_resume_is_reported() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path());

    // `noload` never advertises loadSession: the resume's session/load
    // must fail.
    let spawned = daemon.call_json(&[
        "spawn",
        "--cwd",
        &dir.path().to_string_lossy(),
        "--agent",
        "noload",
        "--model",
        "b",
        "--mode",
        "bypass",
        "--prompt",
        "hi",
    ]);
    let c = spawned["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();
    daemon.status_becomes(&c, "done");

    let output = daemon.run(&["restart"]).expect("restart runs");
    assert!(
        output.status.success(),
        "restart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).expect("restart output is json");
    assert_eq!(
        report["resumed"].as_array().expect("resumed").len(),
        0,
        "{report}"
    );
    let failed: Vec<&Value> = report["failed"]
        .as_array()
        .expect("failed")
        .iter()
        .collect();
    assert_eq!(failed.len(), 1, "{report}");
    assert_eq!(failed[0]["session_id"], c, "{report}");
    assert!(
        failed[0]["error"]
            .as_str()
            .expect("error")
            .contains("session/load"),
        "{report}"
    );

    // The session stays registered — closed, not dropped — so a later
    // adopt or restart can still reach it.
    let status = daemon.call_json(&["status", &c]);
    assert_eq!(status["live"], false, "{status}");
    let registry: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("registry.json")).unwrap())
            .expect("registry is json");
    assert!(registry.get(&c).is_some(), "{registry}");
}
