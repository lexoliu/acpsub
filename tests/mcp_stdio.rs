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
        loop {
            let mut line = String::new();
            self.stdout.read_line(&mut line).expect("read response");
            let msg: Value = serde_json::from_str(line.trim()).expect("response is json");
            if msg.get("id").and_then(Value::as_i64) == Some(id) {
                return msg;
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

    let spawned = server.request(
        "tools/call",
        &json!({
            "name": "spawn",
            "arguments": {"name": "e2e", "agent": "fake", "cwd": dir.path(), "prompt": "hi"},
        }),
    );
    let spawned = tool_json(&spawned);
    assert_eq!(spawned["state"], "running");
    assert_eq!(spawned["session_id"], "sess-1");

    let waited = server.request(
        "tools/call",
        &json!({"name": "wait", "arguments": {"name": "e2e", "timeout_secs": 30}}),
    );
    let waited = tool_json(&waited);
    assert_eq!(waited["state"], "done", "{waited}");
    assert_eq!(waited["reply"], "Hello abworld");

    let list = server.request("tools/call", &json!({"name": "list", "arguments": {}}));
    let list = tool_json(&list);
    assert_eq!(list["subagents"][0]["name"], "e2e");
}
