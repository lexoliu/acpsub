//! Daemon generation handover through the socket: `acpsub restart`
//! starts the next daemon, which takes `daemon.sock` over and resumes
//! every session the old one was not running — while in-flight turns
//! finish in the old daemon, calls for its sessions forward to it, and
//! client connections it already holds keep working.
//!
//! Sequencing is signal-driven, never timed: daemon readiness is the
//! `daemon listening` line on its stderr, a turn's end is observed
//! through `acpsub wait`, the restart's own completion is the handover
//! ordering, and a session's arrival on the new daemon is `status`
//! reporting it live again.

#[expect(
    dead_code,
    reason = "helpers shared across test binaries; this one uses a subset"
)]
mod common;

use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::*;
use serde_json::Value;

/// The agent set the restart tests need: `fake` (loadSession), `noload`
/// (same script with `FAKE_NO_LOAD=1`, so `session/load` is refused) and
/// `slowlimit` (every "ratelimit" prompt parks the turn after a 3 s delay
/// and keeps the session parked for an 8 s window — long enough to hand
/// over inside it).
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

/// A running `acpsub daemon` on a tempdir socket; its guard stops it on
/// drop, along with every replacement a restart auto-started.
struct Daemon {
    guard: DaemonGuard,
    socket: PathBuf,
    config: PathBuf,
    /// Lines the daemon writes to stderr, in order.
    lines: Receiver<String>,
    /// Every line seen so far — dumped when a wait times out.
    seen: Arc<Mutex<Vec<String>>>,
}

impl Daemon {
    /// Write the test config into `dir` and start the daemon, waiting for
    /// its "listening" log line — the real readiness signal, never a
    /// client call (a client would auto-start a daemon and hide a startup
    /// failure).
    fn start(dir: &Path) -> Self {
        let config = write_config(dir, Some("fake"), &agents());
        let socket = dir.join("daemon.sock");
        let mut child = Command::new(env!("CARGO_BIN_EXE_acpsub"))
            .arg("daemon")
            .arg("--socket")
            .arg(&socket)
            .arg("--config")
            .arg(&config)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn acpsub daemon");
        let stderr = child.stderr.take().expect("stderr piped");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (tx, lines) = std::sync::mpsc::channel();
        let captured = seen.clone();
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                captured.lock().expect("seen").push(line.clone());
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let daemon = Self {
            guard: DaemonGuard::new(socket.clone(), Some(child)),
            socket,
            config,
            lines,
            seen,
        };
        daemon.wait_marker("daemon listening");
        daemon
    }

    /// Wait until the daemon writes `marker` to stderr — bounded; on
    /// timeout the captured output is the diagnostic.
    fn wait_marker(&self, marker: &str) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match self.lines.recv_timeout(remaining) {
                Ok(line) if line.contains(marker) => return,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        panic!(
            "daemon never logged '{marker}':\n{}",
            self.seen.lock().expect("seen").join("\n")
        );
    }

    /// Run a client subcommand against this daemon.
    fn run(&self, args: &[&str]) -> std::io::Result<Output> {
        run_to_exit(self.command(args))
    }

    /// Run a client subcommand in a thread and return its handle. The
    /// thread holds an in-flight token until the client exits, so a
    /// teardown the test panics into waits the client out.
    fn run_thread(&self, args: &[&str]) -> std::thread::JoinHandle<Output> {
        let command = self.command(args);
        let in_flight = self.guard.in_flight();
        std::thread::spawn(move || {
            let _in_flight = in_flight;
            run_to_exit(command).expect("client runs")
        })
    }

    /// A client subcommand addressed to this daemon.
    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_acpsub"));
        command
            .args(args)
            .arg("--socket")
            .arg(&self.socket)
            .arg("--config")
            .arg(&self.config);
        command
    }

    /// Run a client subcommand and return its stdout as JSON.
    fn call_json(&self, args: &[&str]) -> Value {
        let output = self.run(args).expect("client runs");
        assert!(
            output.status.success(),
            "{args:?} failed: {}\nfirst daemon log:\n{}",
            String::from_utf8_lossy(&output.stderr),
            self.seen.lock().expect("seen").join("\n")
        );
        serde_json::from_slice(&output.stdout).expect("client output is json")
    }

    /// Poll `status` until `live` reports `want` or the deadline passes:
    /// the real signal that a handoff — or a release — landed, bounded
    /// instead of slept.
    fn status_until(&self, session_id: &str, reached: impl Fn(&Value) -> bool) -> Value {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let status = self.call_json(&["status", session_id]);
            if reached(&status) || Instant::now() >= deadline {
                return status;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// A turn in flight during the restart finishes in the old daemon; its
/// session then comes back live under the same id on the new one, while
/// an idle session was handed over and resumed immediately.
#[test]
fn inflight_turn_hands_off() {
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
    // The turn's end is `wait`'s return, not a status poll.
    let ended = daemon.call_json(&["wait", &b, "--expect", "60"]);
    assert_eq!(ended["state"], "done", "{ended}");

    // The restart completes while A's turn still runs: the new daemon is
    // serving and reports A as still owned by the old one, and the idle
    // session already resumed.
    let output = daemon.run(&["restart"]).expect("restart runs");
    assert!(
        output.status.success(),
        "restart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).expect("restart output is json");
    assert!(
        report["resumed"]
            .as_array()
            .expect("resumed")
            .iter()
            .filter_map(Value::as_str)
            .any(|sid| sid == b),
        "{report}"
    );
    assert!(
        report["pending"]
            .as_array()
            .expect("pending")
            .iter()
            .filter_map(Value::as_str)
            .any(|sid| sid == a),
        "{report}"
    );

    // A's turn ends in the old daemon — its result is recorded — and the
    // session hands over and resumes on the new one.
    let steered = daemon.call_json(&["send", &a, "--policy", "steer", "--prompt", "go"]);
    assert_eq!(steered["state"], "submitted", "{steered}");
    let done = daemon.call_json(&["wait", &a, "--expect", "60"]);
    assert_eq!(done["state"], "done", "{done}");
    assert!(
        done["reply"].as_str().expect("reply").contains("steer:go"),
        "{done}"
    );

    // The handoff itself has no client signal; `status` going live on
    // the new daemon is the observable arrival.
    let status = daemon.status_until(&a, |status| status["live"] == true);
    assert_eq!(status["state"], "done", "{status}");
    assert_eq!(status["turns"], 1, "{status}");

    // And it is a real session again: prompts run on the new daemon.
    let sent = daemon.call_json(&["send", &a, "--policy", "try", "--prompt", "again"]);
    assert_eq!(sent["state"], "running", "{sent}");
    let done = daemon.call_json(&["wait", &a, "--expect", "60"]);
    assert_eq!(done["state"], "done", "{done}");

    let registry: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("registry.json")).unwrap())
            .expect("registry is json");
    let entries = registry.as_object().expect("registry entries");
    assert_eq!(entries.len(), 2, "{registry}");
    assert!(entries.contains_key(&a), "{registry}");
    assert!(entries.contains_key(&b), "{registry}");
    // The old daemon's turn and the new daemon's both count: the single
    // registry writer lost neither.
    assert_eq!(
        registry[&a]["turns"].as_u64().expect("turns"),
        2,
        "{registry}"
    );
}

/// A spawn issued while a turn is in flight during the restart is served
/// at once — no restart gate, no wait for the old daemon's turn.
#[test]
fn spawn_during_restart_serves_immediately() {
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

    // The spawn races the restart: whichever generation it lands on, it
    // runs — the turn in flight on A does not block it.
    let restart = daemon.run_thread(&["restart"]);
    let spawn = daemon.run_thread(&[
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

    let spawned: Value = serde_json::from_slice(&spawn.join().expect("spawn thread").stdout)
        .expect("spawn output is json");
    assert_eq!(spawned["state"], "running", "{spawned}");
    let b = spawned["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();
    let done = daemon.call_json(&["wait", &b, "--expect", "60"]);
    assert_eq!(done["state"], "done", "{done}");
    assert_eq!(done["reply"], "Hello abworld", "{done}");

    let output = restart.join().expect("restart thread");
    assert!(
        output.status.success(),
        "restart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // A's old-daemon turn still finishes and its session still follows.
    let steered = daemon.call_json(&["send", &a, "--policy", "steer", "--prompt", "go"]);
    assert_eq!(steered["state"], "submitted", "{steered}");
    let done = daemon.call_json(&["wait", &a, "--expect", "60"]);
    assert_eq!(done["state"], "done", "{done}");
    let status = daemon.status_until(&a, |status| status["live"] == true);
    assert_eq!(status["live"], true, "{status}");
}

/// A `wait` bound on the old daemon's connection survives the socket
/// rename — the answer comes from the turn that ended in the old daemon.
#[test]
fn wait_blocked_across_restart() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path());

    let spawned = daemon.call_json(&[
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
    let a = spawned["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();

    // Bound while the turn runs; the connection lands on the old
    // daemon's listener before the rename takes the socket over — or
    // forwards through the new one if it is late. Either way it must
    // answer, not fail.
    let wait = daemon.run_thread(&["wait", &a, "--expect", "60"]);
    let restart = daemon.run_thread(&["restart"]);

    let output = restart.join().expect("restart thread");
    assert!(
        output.status.success(),
        "restart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let steered = daemon.call_json(&["send", &a, "--policy", "steer", "--prompt", "go"]);
    assert_eq!(steered["state"], "submitted", "{steered}");

    let output = wait.join().expect("wait thread");
    assert!(
        output.status.success(),
        "wait failed across the restart: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let done: Value = serde_json::from_slice(&output.stdout).expect("wait output is json");
    assert_eq!(done["state"], "done", "{done}");
    assert!(
        done["reply"].as_str().expect("reply").contains("steer:go"),
        "{done}"
    );
}

/// Calls for a session the old generation still owns forward to it:
/// `status` answers the live running state only the owner can have, and
/// `cancel` ends its turn there — the session then follows.
#[test]
fn forwarded_calls_for_old_owned_session() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path());

    // "wait" never completes until cancelled: the session stays
    // old-owned until its turn ends.
    let spawned = daemon.call_json(&[
        "spawn",
        "--cwd",
        &dir.path().to_string_lossy(),
        "--model",
        "b",
        "--mode",
        "bypass",
        "--prompt",
        "wait",
    ]);
    let a = spawned["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();

    let output = daemon.run(&["restart"]).expect("restart runs");
    assert!(
        output.status.success(),
        "restart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The session is not live on the new daemon — a live "running"
    // answer could only come from the owner the call forwarded to.
    let status = daemon.call_json(&["status", &a]);
    assert_eq!(status["live"], true, "{status}");
    assert_eq!(status["state"], "running", "{status}");

    // The cancel forwards too: it ends the turn in the old daemon, and
    // the session hands over.
    let cancelled = daemon.call_json(&["cancel", &a]);
    assert_eq!(cancelled["cancelled"], true, "{cancelled}");

    // The cancelled turn's end is `wait`'s return, wherever the session
    // is by then.
    let ended = daemon.call_json(&["wait", &a, "--expect", "60"]);
    assert_eq!(ended["state"], "cancelled", "{ended}");
}

/// A parked session is handed over with its schedule: after the restart
/// it is live on the new daemon, still `rate_limited`, scheduled by the
/// new daemon's own scheduler.
#[test]
fn parked_session_follows_restart() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path());

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

    // `wait` follows a park through to the resumed turn, so the park is
    // observed on `status` — quiescent work that hands off at once.
    let parked = daemon.status_until(&a, |status| status["state"] == "rate_limited");
    assert_eq!(parked["state"], "rate_limited", "{parked}");

    let output = daemon.run(&["restart"]).expect("restart runs");
    assert!(
        output.status.success(),
        "restart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
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

    let status = daemon.status_until(&a, |status| status["live"] == true);
    assert_eq!(status["state"], "rate_limited", "{status}");
    assert!(status["resume_at"].is_string(), "{status}");
}

/// A prompt parked before its turn started stays marked as the resume's
/// first turn across a daemon restart: the new daemon runs it without a
/// continuation, and `wait` blocks until that turn ends.
#[test]
fn parked_prompt_resumes_without_continuation_after_restart() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path());
    let spawn = |prompt: &str| {
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
            prompt,
        ]);
        spawned["session_id"]
            .as_str()
            .expect("session_id")
            .to_string()
    };
    let trigger = spawn("ratelimit");
    let parked = daemon.status_until(&trigger, |status| status["state"] == "rate_limited");
    assert_eq!(parked["state"], "rate_limited", "{parked}");

    // No turn of this session ever started: its prompt is parked.
    let b = spawn("hi");
    let status = daemon.status_until(&b, |status| status["state"] == "rate_limited");
    assert_eq!(status["state"], "rate_limited", "{status}");

    let output = daemon.run(&["restart"]).expect("restart runs");
    assert!(
        output.status.success(),
        "restart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let status = daemon.status_until(&b, |status| status["live"] == true);
    assert_eq!(status["state"], "rate_limited", "{status}");

    let ended = daemon.call_json(&["wait", &b, "--expect", "60"]);
    assert_eq!(ended["state"], "done", "{ended}");
    assert_eq!(ended["turn"], 1, "{ended}");
    let status = daemon.call_json(&["status", &b]);
    let path = status["transcript"].as_str().expect("transcript path");
    let text = std::fs::read_to_string(path).expect("transcript file");
    assert!(text.contains("\"prompt\":\"hi\""), "{text}");
    assert!(!text.contains("rate limit has lifted"), "{text}");
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
    let ended = daemon.call_json(&["wait", &c, "--expect", "60"]);
    assert_eq!(ended["state"], "done", "{ended}");

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

    // The failure is visible, not just logged: `status` reports the
    // marked session as `resume_failed` with the agent's error, and the
    // registry keeps the mark so the next restart retries the load.
    let status = daemon.call_json(&["status", &c]);
    assert_eq!(status["live"], false, "{status}");
    assert_eq!(status["state"], "resume_failed", "{status}");
    assert!(
        status["error"]
            .as_str()
            .expect("error")
            .contains("session/load"),
        "{status}"
    );
    let registry: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("registry.json")).unwrap())
            .expect("registry is json");
    assert!(registry[&c]["resume"]["failed"].is_string(), "{registry}");
}
