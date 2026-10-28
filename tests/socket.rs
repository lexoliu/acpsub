//! End-to-end test through the daemon socket: `acpsub daemon` as a child
//! process, and the CLI client commands connecting over it.

#[expect(
    dead_code,
    reason = "helpers shared across test binaries; this one uses a subset"
)]
mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::*;
use serde_json::Value;

/// A running `acpsub daemon` on a tempdir socket; its guard stops it on
/// drop, along with any daemon a client auto-started behind it.
struct Daemon {
    _guard: DaemonGuard,
    socket: PathBuf,
    config: PathBuf,
}

impl Daemon {
    /// Write a one-agent config into `dir` and start the daemon.
    fn start(dir: &Path) -> Self {
        let mut agents = BTreeMap::new();
        agents.insert("fake".to_string(), fake_agent_config());
        let config = write_config(dir, Some("fake"), &agents, None);
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
            _guard: DaemonGuard::new(socket.clone(), Some(child)),
            socket,
            config,
        };
        // Wait for the socket file: it exists the moment the daemon bound
        // and is listening. Never invoke a client here — a client
        // auto-starts a daemon when the socket is absent, which would hide
        // a startup failure.
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

    /// Run a client subcommand against this daemon.
    fn run(&self, args: &[&str]) -> std::io::Result<std::process::Output> {
        let mut command = Command::new(env!("CARGO_BIN_EXE_acpsub"));
        command
            .args(args)
            .arg("--socket")
            .arg(&self.socket)
            .arg("--config")
            .arg(&self.config);
        run_to_exit(command)
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
}

#[test]
fn daemon_spawn_wait_result_close() {
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
        "hi",
    ]);
    assert_eq!(spawned["state"], "running", "{spawned}");
    let sid = spawned["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();

    let done = daemon.call_json(&["wait", &sid, "--expect", "60"]);
    assert_eq!(done["state"], "done", "{done}");
    assert_eq!(done["reply"], "Hello abworld", "{done}");

    let result = daemon.call_json(&["result", &sid]);
    assert_eq!(result["reply"], "Hello abworld", "{result}");

    let status = daemon.call_json(&["status", &sid]);
    assert_eq!(status["state"], "done", "{status}");

    let list = daemon.call_json(&["list"]);
    let ids: Vec<&str> = list["subagents"]
        .as_array()
        .expect("subagents")
        .iter()
        .filter_map(|s| s["session_id"].as_str())
        .collect();
    assert!(ids.contains(&sid.as_str()), "{ids:?}");

    let closed = daemon.call_json(&["close", &sid]);
    assert_eq!(closed["closed"], true, "{closed}");
}

/// A `wait` that returns with the turn still running repeats the
/// watchdog's cancellation warning on stderr, so a host that surfaces
/// only the background task's exit still shows it (#107).
#[test]
fn wait_overrun_writes_warning_to_stderr() {
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
        "wait",
    ]);
    let sid = spawned["session_id"].as_str().expect("session_id");

    let output = daemon
        .run(&["wait", sid, "--expect", "1"])
        .expect("client runs");
    assert!(
        output.status.success(),
        "wait failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("is still running"), "{stdout}");
    assert!(stdout.contains("now has no waiter"), "{stdout}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("now has no waiter"), "{stderr}");
    assert!(stderr.contains("the watchdog cancels the turn"), "{stderr}");
    assert!(stderr.contains(&format!("acpsub wait {sid}")), "{stderr}");
    assert!(stderr.contains("take it over"), "{stderr}");
    assert!(stderr.contains(&format!("acpsub cancel {sid}")), "{stderr}");

    // --json carries the deadline as a timestamp field; the warning
    // still goes to stderr.
    let output = daemon
        .run(&["wait", sid, "--expect", "60", "--max-wait", "1", "--json"])
        .expect("client runs");
    assert!(output.status.success());
    let out: Value = serde_json::from_slice(&output.stdout).expect("json");
    assert_eq!(out["state"], "running", "{out}");
    assert!(out["unwatched_cancel_at"].is_string(), "{out}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("now has no waiter"), "{stderr}");

    daemon.call_json(&["cancel", sid]);
}

/// `--expect`/`--max-wait` over 1800 s (30 minutes) are rejected unless
/// `--long` is passed; the limit itself passes, and `--long` lifts it.
#[test]
fn wait_expect_limit() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(dir.path());
    // "hi" ends at once, so an accepted wait returns immediately.
    let spawned = daemon.call_json(&[
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
    let sid = spawned["session_id"].as_str().expect("session_id");

    for (args, field) in [
        (vec!["wait", sid, "--expect", "1801"], "expect_secs"),
        (vec!["wait-any", sid, "--expect", "1801"], "expect_secs"),
        (
            vec!["wait", sid, "--expect", "60", "--max-wait", "1801"],
            "max_wait_secs",
        ),
        (
            vec!["wait-any", sid, "--expect", "60", "--max-wait", "1801"],
            "max_wait_secs",
        ),
    ] {
        let output = daemon.run(&args).expect("client runs");
        assert!(!output.status.success(), "{args:?} unexpectedly succeeded");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(field), "{args:?}: {stderr}");
        assert!(stderr.contains("1801"), "{args:?}: {stderr}");
        assert!(stderr.contains("1800"), "{args:?}: {stderr}");
        assert!(stderr.contains("--long"), "{args:?}: {stderr}");
    }

    for args in [
        vec!["wait", sid, "--expect", "1800"],
        vec!["wait-any", sid, "--expect", "1800"],
        vec!["wait", sid, "--expect", "7200", "--long"],
        vec!["wait-any", sid, "--expect", "7200", "--long"],
        vec![
            "wait",
            sid,
            "--expect",
            "60",
            "--max-wait",
            "7200",
            "--long",
        ],
    ] {
        let done = daemon.call_json(&args);
        assert_eq!(done["state"], "done", "{args:?}: {done}");
    }
}

#[test]
fn client_starts_daemon_when_absent() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let mut agents = BTreeMap::new();
    agents.insert("fake".to_string(), fake_agent_config());
    let config = write_config(dir.path(), Some("fake"), &agents, None);
    let socket = dir.path().join("daemon.sock");
    // The auto-started daemon is detached: the guard stops it through
    // its pid file, whatever the assertions below do.
    let _guard = DaemonGuard::new(socket.clone(), None);
    // No daemon running: `list` must spawn one and still succeed.
    let mut command = Command::new(env!("CARGO_BIN_EXE_acpsub"));
    command
        .args(["list", "--json", "--socket"])
        .arg(&socket)
        .arg("--config")
        .arg(&config);
    let output = run_to_exit(command).expect("client runs");
    assert!(
        output.status.success(),
        "auto-start failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let list: Value = serde_json::from_slice(&output.stdout).expect("json");
    assert!(list["subagents"].is_array(), "{list}");
}
