//! The wait-fed watchdog through the daemon socket: a running turn
//! nobody waits on is cancelled and its session closed once no
//! `wait`/`wait_any` caller has fed it for `wait_grace`. An attached
//! wait — CLI or MCP — keeps it alive; an idle session burns nothing;
//! a restart is not waiter loss.
//!
//! The grace is injected through the config file's `[defaults]
//! wait_grace`, so no test sleeps the real 60 s.

#[expect(
    dead_code,
    reason = "helpers shared across test binaries; this one uses a subset"
)]
mod common;

use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::*;
use serde_json::Value;

/// A running `acpsub daemon` on a tempdir socket with an injected
/// `wait_grace`; its guard stops it on drop.
struct Daemon {
    _guard: DaemonGuard,
    socket: PathBuf,
    config: PathBuf,
}

impl Daemon {
    /// Write the test config into `dir` with `wait_grace` and start the
    /// daemon, waiting for the socket file — never a client call (a
    /// client would auto-start a daemon and hide a startup failure).
    fn start(dir: &Path, wait_grace: f64) -> Self {
        let mut agents = BTreeMap::new();
        agents.insert("fake".to_string(), fake_agent_config());
        let config = write_config(dir, Some("fake"), &agents, Some(wait_grace));
        let socket = dir.join("daemon.sock");
        let child = Command::new(env!("CARGO_BIN_EXE_acpsub"))
            .arg("daemon")
            .arg("--socket")
            .arg(&socket)
            .arg("--config")
            .arg(&config)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn acpsub daemon");
        let daemon = Self {
            _guard: DaemonGuard::new(socket.clone(), Some(child)),
            socket,
            config,
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

    /// Run a client subcommand to completion.
    fn run(&self, args: &[&str]) -> std::io::Result<std::process::Output> {
        run_to_exit(self.command(args))
    }

    /// Run a client subcommand and return its stdout as JSON.
    fn call_json(&self, args: &[&str]) -> Value {
        let mut args = args.to_vec();
        args.push("--json");
        let output = self.run(&args).expect("client runs");
        assert!(
            output.status.success(),
            "{args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).expect("client output is json")
    }

    /// Spawn a subagent on the `fake` agent; its first prompt is `prompt`.
    fn spawn(&self, dir: &Path, prompt: &str) -> String {
        let spawned = self.call_json(&[
            "spawn",
            "--cwd",
            &dir.to_string_lossy(),
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
    }

    /// Poll `status` until `reached` holds or the deadline passes.
    fn status_until(&self, session_id: &str, reached: impl Fn(&Value) -> bool) -> Value {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let status = self.call_json(&["status", session_id]);
            if reached(&status) || Instant::now() >= deadline {
                return status;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// A turn with an attached `wait-any` survives past the grace period:
/// the wait call feeds the watchdog for as long as it lives.
#[test]
fn watched_turn_survives_grace() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path(), 1.5);
    let sid = daemon.spawn(dir.path(), "wait");

    // "wait" never completes until cancelled: the turn runs as long as
    // the test lets it. The attached wait-any must keep it alive well
    // past the grace period.
    let mut waiter = daemon
        .command(&["wait-any", &sid, "--expect", "60"])
        .spawn()
        .expect("spawn wait-any client");
    std::thread::sleep(Duration::from_secs(5));
    let status = daemon.call_json(&["status", &sid]);
    assert_eq!(status["live"], true, "{status}");
    assert_eq!(status["state"], "running", "{status}");

    let cancelled = daemon.call_json(&["cancel", &sid]);
    assert_eq!(cancelled["cancelled"], true, "{cancelled}");
    waiter.wait().expect("reap wait-any client");
}

/// A turn whose waiter is killed is cancelled once the grace passes:
/// the dead client's connection closing is the detach the watchdog
/// sees. The session stays registered, adoptable.
#[test]
fn dead_waiter_turn_is_cancelled() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path(), 1.5);
    let sid = daemon.spawn(dir.path(), "wait");

    let mut waiter = daemon
        .command(&["wait", &sid, "--expect", "60", "--json"])
        .spawn()
        .expect("spawn wait client");
    // Let the client's tools/call land so it is attached.
    std::thread::sleep(Duration::from_millis(500));
    waiter.kill().expect("kill wait client");
    waiter.wait().expect("reap wait client");

    let status = daemon.status_until(&sid, |status| status["live"] == false);
    assert_eq!(status["live"], false, "{status}");
    assert_eq!(status["state"], "closed", "{status}");
}

/// A turn awaiting a permission answer is not watched: `wait` settles on
/// it at once, so the coordinator may take longer than the grace to
/// decide, and the turn still completes after `permit`.
#[test]
fn permission_pending_turn_outlives_grace() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path(), 1.5);
    let spawned = daemon.call_json(&[
        "spawn",
        "--cwd",
        &dir.path().to_string_lossy(),
        "--model",
        "b",
        "--mode",
        "bypass",
        "--permission",
        "ask",
        "--prompt",
        "PERMISSION go",
    ]);
    let sid = spawned["session_id"].as_str().expect("session_id");
    let waiting = daemon.call_json(&["wait", sid, "--expect", "60"]);
    assert_eq!(waiting["state"], "needs_permission", "{waiting}");
    let request_id = waiting["pending_permission"]["request_id"]
        .as_str()
        .expect("request_id");

    std::thread::sleep(Duration::from_secs(4));
    let status = daemon.call_json(&["status", sid]);
    assert_eq!(status["live"], true, "{status}");
    assert_eq!(status["state"], "needs_permission", "{status}");

    daemon.call_json(&[
        "permit",
        sid,
        "--request",
        request_id,
        "--option",
        "allow-1",
    ]);
    let done = daemon.call_json(&["wait", sid, "--expect", "60"]);
    assert_eq!(done["state"], "done", "{done}");
    assert!(
        done["reply"]
            .as_str()
            .expect("reply")
            .contains("perm:allow-1"),
        "{done}"
    );
}

/// An idle session is never reaped: with no turn in flight the
/// watchdog has nothing to cancel.
#[test]
fn idle_session_is_never_reaped() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path(), 1.5);
    let sid = daemon.spawn(dir.path(), "hi");
    let done = daemon.call_json(&["wait", &sid, "--expect", "60"]);
    assert_eq!(done["state"], "done", "{done}");

    std::thread::sleep(Duration::from_secs(4));
    let status = daemon.call_json(&["status", &sid]);
    assert_eq!(status["live"], true, "{status}");
    assert_eq!(status["state"], "done", "{status}");
}

/// A restart handoff does not cancel turns: an unwatched turn kept by
/// the old generation gets a fresh grace clock at the takeover, so it
/// outlives its turn-start deadline and a `wait` reattaching through the
/// new daemon's forwarding still collects it.
#[test]
fn restart_handoff_does_not_cancel_turns() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let grace = 5.0;
    let daemon = Daemon::start(dir.path(), grace);
    // "gather" holds its turn open until a steer arrives. Nothing waits on
    // it, so its grace clock runs from the turn start.
    let sid = daemon.spawn(dir.path(), "gather");
    std::thread::sleep(Duration::from_secs_f64(grace * 0.6));

    let output = daemon.run(&["restart"]).expect("restart runs");
    assert!(
        output.status.success(),
        "restart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Past the turn-start deadline but inside a grace period restarted at
    // the takeover: a watchdog that kept the old clock, or counted the
    // handoff as waiter loss, has cancelled the turn by now.
    std::thread::sleep(Duration::from_secs_f64(grace * 0.6));
    let status = daemon.call_json(&["status", &sid]);
    assert_eq!(status["live"], true, "{status}");
    assert_eq!(status["state"], "running", "{status}");

    // Reattach through the new daemon's forwarding and end the turn.
    let waiter = daemon
        .command(&["wait", &sid, "--expect", "60", "--json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn wait client");
    std::thread::sleep(Duration::from_millis(500));
    let steered = daemon.call_json(&["send", &sid, "--policy", "steer", "--prompt", "go"]);
    assert_eq!(steered["state"], "submitted", "{steered}");
    let output = waiter.wait_with_output().expect("wait client runs");
    assert!(
        output.status.success(),
        "wait failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let done: Value = serde_json::from_slice(&output.stdout).expect("wait output is json");
    assert_eq!(done["state"], "done", "{done}");
    assert!(
        done["reply"].as_str().expect("reply").contains("steer:go"),
        "{done}"
    );
}
