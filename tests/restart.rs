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
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{Receiver, SyncSender};
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
    /// The started daemon's pid — its private socket is
    /// `daemon.<pid>.sock` beside the public one.
    pid: u32,
    /// Lines the daemon writes to stderr, in order.
    lines: Receiver<String>,
    /// Every line seen so far — dumped when a wait times out.
    seen: Arc<Mutex<Vec<String>>>,
}

impl Daemon {
    /// Write the test config into `dir` and start the daemon.
    fn start(dir: &Path) -> Self {
        Self::start_with(dir, &agents())
    }

    /// `start` over an explicit agent map, waiting for the "listening"
    /// log line — the real readiness signal, never a client call (a
    /// client would auto-start a daemon and hide a startup failure).
    fn start_with(dir: &Path, agents: &BTreeMap<String, acpsub::config::AgentConfig>) -> Self {
        let config = write_config(dir, Some("fake"), agents, None);
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
        let pid = child.id();
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
            pid,
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
    fn run(&self, args: &[&str]) -> std::io::Result<std::process::Output> {
        self.command(args).output()
    }

    /// Run a client subcommand in a thread and return its handle. The
    /// thread holds an in-flight token until the client exits, so a
    /// teardown the test panics into waits the client out.
    fn run_thread(&self, args: &[&str]) -> std::thread::JoinHandle<std::process::Output> {
        let mut command = self.command(args);
        let in_flight = self.guard.in_flight();
        std::thread::spawn(move || {
            let _in_flight = in_flight;
            command.output().expect("client runs")
        })
    }

    /// `run_thread` against `socket` instead of the public one — the
    /// daemon's private generation socket serves the same local tools
    /// without the forwarding dispatch.
    fn run_thread_on(
        &self,
        socket: &Path,
        args: &[&str],
    ) -> std::thread::JoinHandle<std::process::Output> {
        let mut command = self.command_on(socket, args);
        let in_flight = self.guard.in_flight();
        std::thread::spawn(move || {
            let _in_flight = in_flight;
            command.output().expect("client runs")
        })
    }

    /// A client subcommand addressed to this daemon.
    fn command(&self, args: &[&str]) -> Command {
        self.command_on(&self.socket.clone(), args)
    }

    /// `command` against an explicit socket path.
    fn command_on(&self, socket: &Path, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_acpsub"));
        command
            .args(args)
            .arg("--socket")
            .arg(socket)
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

    // A `status` for the session answers `live: true` from the old owner
    // while the handoff is still in flight, so the ship is observed on
    // the old daemon's own log line — the handoff's real signal.
    daemon.wait_marker("session handed off to the next daemon");
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

/// A steer that resolves after its turn settled still frees the session:
/// the handover watcher must wake on the resolution, or the session
/// stays on the old daemon forever. `FAKE_LATE_STEER` holds the steer's
/// answer past the turn's end until the gate file exists — the ordering
/// is the agent's, not a delay's.
#[test]
fn late_steer_resolution_still_hands_off() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let gate = dir.path().join("steer-gate");
    let mut agents = agents();
    agents.insert(
        "latesteer".to_string(),
        acpsub::config::AgentConfig {
            env: BTreeMap::from([(
                "FAKE_LATE_STEER".to_string(),
                gate.to_string_lossy().into_owned(),
            )]),
            ..fake_agent_config()
        },
    );
    let daemon = Daemon::start_with(dir.path(), &agents);

    let spawned = daemon.call_json(&[
        "spawn",
        "--cwd",
        &dir.path().to_string_lossy(),
        "--agent",
        "latesteer",
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

    let output = daemon.run(&["restart"]).expect("restart runs");
    assert!(
        output.status.success(),
        "restart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The steer ends the "gather" turn; its own request answer waits on
    // the gate — the turn settles with the steer still in flight, which
    // keeps the session on the old daemon.
    let steered = daemon.call_json(&["send", &a, "--policy", "steer", "--prompt", "go"]);
    assert_eq!(steered["state"], "submitted", "{steered}");
    let status = daemon.status_until(&a, |status| status["state"] == "done");
    assert_eq!(status["state"], "done", "{status}");

    // The resolution frees the session: the ship follows on its own.
    std::fs::write(&gate, "").expect("release the steer");
    daemon.wait_marker("session handed off to the next daemon");
    let status = daemon.status_until(&a, |status| status["live"] == true);
    assert_eq!(status["state"], "done", "{status}");
}

/// A `send` that reaches the old owner while its `daemon/handoff` report
/// is still unanswered holds rather than failing: the refusal lands only
/// once the ship resolves — `still_owned == false` on the successor by
/// then — or the ship fails and the send proceeds on the session's real
/// owner. Both refuse with the true reason, never "steer in flight".
///
/// The test stands in for the successor: `daemon/takeover` on the old
/// daemon's private socket names a socket this test serves, so the
/// `daemon/handoff` report waits on the test's answer — the window a
/// forwarded send falls into, held open deterministically. Session `b`
/// keeps a turn in flight forever so the old daemon still owns something
/// (and keeps serving) after `a` ships.
#[test]
fn send_during_handoff_waits_for_the_ship() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path());
    let old_private = dir.path().join(format!("daemon.{}.sock", daemon.pid));

    // The fake successor: a socket the test answers generation-protocol
    // calls on, each answer gated by the test.
    let next_socket = dir.path().join("next.sock");
    let calls = serve_fake_successor(&next_socket);

    let spawn = |prompt: &str| {
        let spawned = daemon.call_json(&[
            "spawn",
            "--cwd",
            &dir.path().to_string_lossy(),
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
    let a = spawn("gather");
    let _b = spawn("wait");

    // Supersede the old daemon, naming the fake successor: both running
    // sessions stay owned.
    let report = private_call(
        &old_private,
        "daemon/takeover",
        &serde_json::json!({"socket": next_socket}),
    );
    let report: Value = serde_json::from_str(&report).expect("takeover report is json");
    assert!(
        report["kept"]
            .as_array()
            .expect("kept")
            .iter()
            .filter_map(|kept| kept["session_id"].as_str())
            .any(|id| id == a),
        "{report}"
    );

    // A's turn ends in the old daemon and the watcher ships it: the
    // `daemon/handoff` report reaching the fake successor proves the
    // session's `HandingOff` mark is already set.
    let steered = daemon.call_json(&["send", &a, "--policy", "steer", "--prompt", "go"]);
    assert_eq!(steered["state"], "submitted", "{steered}");
    let call = next_call(&calls);
    assert_eq!(call.tool, "daemon/handoff");
    assert_eq!(call.arguments["session_id"], a, "{:?}", call.arguments);

    // A send landing in the window holds instead of erroring at once.
    let send = daemon.run_thread_on(
        &old_private,
        &["send", &a, "--policy", "try", "--prompt", "again"],
    );
    if finished_within(&send, Duration::from_millis(500)) {
        let output = send.join().expect("send thread");
        panic!("send returned before the handoff resolved: {output:?}");
    }

    // The ship resolves; the held send answers with the true reason.
    call.answer(r#"{"content":[{"type":"text","text":"{}"}]}"#);
    let output = send.join().expect("send thread");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{stderr}");
    assert!(stderr.contains("handing off"), "{stderr}");
    assert!(!stderr.contains("steer in flight"), "{stderr}");
    // `a` ships for real — consume its marker so the one waited on below
    // is `c`'s.
    daemon.wait_marker("session handed off to the next daemon");

    // A refused ship clears the mark and the watcher retries: `c`
    // settles, its first report is refused, and the rescan's second
    // `daemon/handoff` is the deterministic proof the mark cleared. A
    // queued send landing in that retry's window holds the same way.
    let c = spawn("hi");
    let ended = daemon.call_json(&["wait", &c, "--expect", "60"]);
    assert_eq!(ended["state"], "done", "{ended}");
    let call = next_call(&calls);
    assert_eq!(call.tool, "daemon/handoff");
    assert_eq!(call.arguments["session_id"], c, "{:?}", call.arguments);
    call.answer(r#"{"content":[{"type":"text","text":"refused for the test"}],"isError":true}"#);
    let call = next_call(&calls);
    assert_eq!(call.tool, "daemon/handoff");
    assert_eq!(call.arguments["session_id"], c, "{:?}", call.arguments);

    let send = daemon.run_thread_on(
        &old_private,
        &["send", &c, "--policy", "queued", "--prompt", "again"],
    );
    if finished_within(&send, Duration::from_millis(500)) {
        let output = send.join().expect("send thread");
        panic!("send returned before the handoff resolved: {output:?}");
    }
    call.answer(r#"{"content":[{"type":"text","text":"{}"}]}"#);
    let output = send.join().expect("send thread");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{stderr}");
    assert!(stderr.contains("handing off"), "{stderr}");
    assert!(!stderr.contains("steer in flight"), "{stderr}");
    daemon.wait_marker("session handed off to the next daemon");
}

/// Whether `handle` finished within `dur`: the bounded check that a call
/// is still in flight — asserting a hold, never ordering two events.
fn finished_within(handle: &std::thread::JoinHandle<std::process::Output>, dur: Duration) -> bool {
    let deadline = Instant::now() + dur;
    while !handle.is_finished() {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    true
}

/// One generation-protocol call the fake successor received; the test
/// answers it through `answer`.
struct NextCall {
    /// The tool the daemon called (`daemon/handoff`, `daemon/release`).
    tool: String,
    /// The call's `arguments`.
    arguments: Value,
    /// Where the answer goes.
    answer: SyncSender<Value>,
}

impl NextCall {
    /// Answer the call with a `CallToolResult` JSON document.
    fn answer(self, result: &str) {
        self.answer
            .send(serde_json::from_str(result).expect("result is json"))
            .expect("fake successor still serving");
    }
}

/// Take the next generation-protocol call the fake successor received.
fn next_call(calls: &Receiver<NextCall>) -> NextCall {
    calls
        .recv_timeout(Duration::from_secs(15))
        .expect("generation-protocol call never arrived")
}

/// Bind `socket` and serve the daemon's generation protocol on it: every
/// `tools/call` arrives on the returned channel and waits for the test's
/// answer — the handoff window, held open as long as the test wants.
fn serve_fake_successor(socket: &Path) -> Receiver<NextCall> {
    let listener = UnixListener::bind(socket).expect("bind fake successor");
    let (tx, calls) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let tx = tx.clone();
            std::thread::spawn(move || serve_successor_conn(stream, &tx));
        }
    });
    calls
}

/// One fake-successor connection: the MCP handshake, then each
/// `tools/call` forwarded to the test and answered with what it sends
/// back.
fn serve_successor_conn(stream: UnixStream, calls: &std::sync::mpsc::Sender<NextCall>) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut writer = stream;
    let mut respond = |id: &Value, result: Value| {
        let mut line =
            serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string();
        line.push('\n');
        writer
            .write_all(line.as_bytes())
            .expect("write fake successor response");
    };
    let mut line = String::new();
    loop {
        line.clear();
        let Ok(read) = reader.read_line(&mut line) else {
            return;
        };
        if read == 0 {
            return;
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(id) = msg.get("id").cloned() else {
            continue;
        };
        match msg["method"].as_str() {
            Some("initialize") => respond(
                &id,
                serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "fake-successor", "version": "0"},
                }),
            ),
            Some("tools/call") => {
                let (tx, rx) = std::sync::mpsc::sync_channel(0);
                if calls
                    .send(NextCall {
                        tool: msg["params"]["name"].as_str().unwrap_or("?").to_string(),
                        arguments: msg["params"]["arguments"].clone(),
                        answer: tx,
                    })
                    .is_err()
                {
                    return;
                }
                let Ok(result) = rx.recv() else {
                    return;
                };
                respond(&id, result);
            }
            _ => {}
        }
    }
}

/// One newline-delimited JSON-RPC `tools/call` on a daemon's private
/// socket — the generation protocol's client side, against the local
/// tool set the private listener serves.
fn private_call(socket: &Path, tool: &str, arguments: &Value) -> String {
    let mut stream = UnixStream::connect(socket).expect("connect private socket");
    let mut line = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "restart-test", "version": "0"},
        },
    })
    .to_string();
    line.push('\n');
    stream.write_all(line.as_bytes()).expect("write initialize");
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut read = String::new();
    reader.read_line(&mut read).expect("initialize response");
    let mut line = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    })
    .to_string();
    line.push('\n');
    stream
        .write_all(line.as_bytes())
        .expect("write initialized");
    let mut line = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    })
    .to_string();
    line.push('\n');
    stream.write_all(line.as_bytes()).expect("write tools/call");
    loop {
        read.clear();
        let n = reader.read_line(&mut read).expect("call response");
        assert!(n > 0, "daemon closed the private connection");
        let msg: Value = serde_json::from_str(&read).expect("response is json");
        if msg.get("id") != Some(&Value::from(1)) {
            continue;
        }
        assert!(msg["error"].is_null(), "{tool} refused: {}", msg["error"]);
        let text = msg["result"]["content"]
            .as_array()
            .and_then(|content| content.first())
            .and_then(|item| item["text"].as_str())
            .expect("call result text");
        assert!(
            !msg["result"]["isError"].as_bool().unwrap_or(false),
            "{tool} errored: {text}"
        );
        return text.to_string();
    }
}