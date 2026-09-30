//! End-to-end test through the real MCP transport: `acpsub serve` as a child
//! process, JSON-RPC over stdio.

#[expect(
    dead_code,
    reason = "helpers shared across test binaries; this one uses a subset"
)]
mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};

use common::*;
use serde_json::{Value, json};

struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    next_id: i64,
}

impl Server {
    fn spawn(config: &std::path::Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_acpsub"))
            .args(["serve", "--config"])
            .arg(config)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn acpsub serve");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        Self {
            child,
            stdin,
            stdout,
            next_id: 1,
        }
    }

    /// Send a request and read lines until its response arrives.
    fn request(&mut self, method: &str, params: &Value) -> Value {
        self.request_notified(method, params).0
    }

    /// Send a request and read lines until its response arrives, collecting
    /// every notification seen meanwhile.
    fn request_notified(&mut self, method: &str, params: &Value) -> (Value, Vec<Value>) {
        let id = self.next_id;
        self.next_id += 1;
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        writeln!(self.stdin, "{request}").expect("write request");
        self.stdin.flush().expect("flush");
        let mut notifications = Vec::new();
        loop {
            let mut line = String::new();
            self.stdout.read_line(&mut line).expect("read response");
            let msg: Value = serde_json::from_str(line.trim()).expect("response is json");
            if msg.get("id").and_then(Value::as_i64) == Some(id) {
                return (msg, notifications);
            }
            if msg.get("method").is_some() {
                notifications.push(msg);
            }
        }
    }

    fn notify(&mut self, method: &str) {
        let notification = json!({"jsonrpc": "2.0", "method": method});
        writeln!(self.stdin, "{notification}").expect("write notification");
        self.stdin.flush().expect("flush");
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Extract the JSON a tool call returned in `content[0].text`.
fn tool_json(response: &Value) -> Value {
    let result = &response["result"];
    assert_ne!(
        result["isError"].as_bool(),
        Some(true),
        "tool error: {result}"
    );
    let text = result["content"][0]["text"].as_str().expect("text content");
    serde_json::from_str(text).expect("tool result is json")
}

#[test]
fn mcp_end_to_end() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.toml");
    std::fs::write(
        &config,
        format!(
            r#"
[defaults]
permission = "allow"
transcript_dir = "{dir}/transcripts"
registry = "{dir}/registry.json"

[agents.fake]
command = "python3"
args = ["{script}"]
"#,
            dir = dir.path().display(),
            script = fake_agent_script().display(),
        ),
    )
    .unwrap();

    let mut server = Server::spawn(&config);
    let init = server.request(
        "initialize",
        &json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "e2e", "version": "0"},
        }),
    );
    assert_eq!(init["result"]["serverInfo"]["name"], "acpsub", "{init}");
    server.notify("notifications/initialized");

    let tools = server.request("tools/list", &json!({}));
    let names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    for expected in [
        "spawn",
        "adopt",
        "send",
        "wait",
        "wait_any",
        "status",
        "result",
        "cancel",
        "permit",
        "transcript",
        "list",
        "close",
        "forget",
        "agents",
    ] {
        assert!(
            names.contains(&expected),
            "missing tool {expected} in {names:?}"
        );
    }

    // One configured agent: `agent` may be omitted — the only choice is the
    // implicit default.
    let spawned = server.request(
        "tools/call",
        &json!({
            "name": "spawn",
            "arguments": {"cwd": dir.path(), "prompt": "hi", "model": "b", "mode": "bypass"},
        }),
    );
    let spawned = tool_json(&spawned);
    assert_eq!(spawned["state"], "running");
    let session_id = spawned["session_id"].as_str().expect("session_id");
    assert!(session_id.starts_with("sess-"), "{session_id}");

    let waited = server.request(
        "tools/call",
        &json!({"name": "wait", "arguments": {"session_id": session_id, "expect_secs": 60}}),
    );
    let waited = tool_json(&waited);
    assert_eq!(waited["state"], "done", "{waited}");
    assert_eq!(waited["reply"], "Hello abworld");

    let list = server.request("tools/call", &json!({"name": "list", "arguments": {}}));
    let list = tool_json(&list);
    assert_eq!(list["subagents"][0]["session_id"], session_id);
}

/// A `tools/call` carrying `_meta.progressToken` gets `notifications/progress`
/// on the wire while `wait` blocks — the keep-alive MCP hosts need.
#[test]
fn mcp_wait_reports_progress() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.toml");
    std::fs::write(
        &config,
        format!(
            r#"
[defaults]
permission = "allow"
transcript_dir = "{dir}/transcripts"
registry = "{dir}/registry.json"

[agents.fake]
command = "python3"
args = ["{script}"]
"#,
            dir = dir.path().display(),
            script = fake_agent_script().display(),
        ),
    )
    .unwrap();

    let mut server = Server::spawn(&config);
    server.request(
        "initialize",
        &json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "e2e", "version": "0"},
        }),
    );
    server.notify("notifications/initialized");

    // The fake agent's "wait" turn never completes: the wait returns on
    // its own `expect_secs` overrun, having reported meanwhile. The
    // production interval is 10 s, so 12 s sees the first report and the
    // first interval repeat.
    let spawned = server.request(
        "tools/call",
        &json!({
            "name": "spawn",
            "arguments": {"cwd": dir.path(), "prompt": "wait", "model": "b", "mode": "bypass"},
        }),
    );
    let session_id = tool_json(&spawned)["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();

    let (waited, notifications) = server.request_notified(
        "tools/call",
        &json!({
            "name": "wait",
            "arguments": {"session_id": session_id, "expect_secs": 12},
            "_meta": {"progressToken": "tok-1"},
        }),
    );

    let progress: Vec<&Value> = notifications
        .iter()
        .filter(|msg| msg["method"] == "notifications/progress")
        .collect();
    assert!(
        progress.len() >= 2,
        "expected the first report and an interval repeat, got {notifications:?}"
    );
    let values: Vec<f64> = progress
        .iter()
        .map(|msg| msg["params"]["progress"].as_f64().expect("progress"))
        .collect();
    assert!(
        values.windows(2).all(|w| w[1] > w[0]),
        "progress must increase: {values:?}"
    );
    for msg in &progress {
        assert_eq!(msg["params"]["progressToken"], "tok-1", "{msg}");
        assert_eq!(msg["params"]["total"].as_f64(), Some(12.0), "{msg}");
        let message = msg["params"]["message"].as_str().expect("message");
        assert!(message.contains(&session_id), "{message}");
    }
    // The first report can precede the agent's tool_call update; later ones
    // carry its title.
    assert!(
        progress.iter().any(|msg| msg["params"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("fake-tool"))),
        "no message names the latest tool call: {progress:?}"
    );
    let waited = tool_json(&waited);
    assert_eq!(waited["state"], "overrun", "{waited}");
}
