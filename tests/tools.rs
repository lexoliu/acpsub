//! Integration tests: the tool set exercised in-process against
//! `tests/fake_agent.py`.

mod common;

use common::*;
use serde_json::json;

#[tokio::test(flavor = "multi_thread")]
async fn spawn_wait_reply() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let spawned = call_json(&tools, "spawn", spawn_args("a", dir.path(), "hi")).await;
    assert_eq!(spawned["name"], "a");
    assert_eq!(spawned["state"], "running");
    assert_eq!(spawned["session_id"], "sess-1");

    let done = wait(&tools, "a", 30).await;
    assert_eq!(done["state"], "done");
    assert_eq!(done["stop_reason"], "end_turn");
    assert_eq!(done["reply"], "Hello abworld", "{done}");
    assert_eq!(done["tool_calls"][0]["id"], "tc-1");
    assert_eq!(done["tool_calls"][0]["status"], "completed");

    let result = call_json(&tools, "result", json!({"name": "a"})).await;
    assert_eq!(result["reply"], "Hello abworld");
    assert_eq!(result["turn"], 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn send_followup_turn() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("b", dir.path(), "hi")).await;
    wait(&tools, "b", 30).await;

    call_json(&tools, "send", json!({"name": "b", "prompt": "again"})).await;
    let done = wait(&tools, "b", 30).await;
    assert_eq!(done["state"], "done");

    let status = call_json(&tools, "status", json!({"name": "b"})).await;
    assert_eq!(status["turns"], 2);
    assert_eq!(status["state"], "done");
    assert_eq!(status["last_stop_reason"], "end_turn");

    let first = call_json(&tools, "result", json!({"name": "b", "turn": 1})).await;
    assert_eq!(first["turn"], 1);
    let missing = call_err(&tools, "result", json!({"name": "b", "turn": 9})).await;
    assert!(missing.contains("no turn 9"), "{missing}");
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_mid_turn() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("c", dir.path(), "wait")).await;
    let running = wait(&tools, "c", 2).await;
    assert_eq!(running["state"], "running");

    call_json(&tools, "cancel", json!({"name": "c"})).await;
    let done = wait(&tools, "c", 30).await;
    assert_eq!(done["state"], "cancelled", "{done}");
    assert_eq!(done["stop_reason"], "cancelled");
}

#[tokio::test(flavor = "multi_thread")]
async fn permission_ask_then_permit() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let mut args = spawn_args("d", dir.path(), "PERMISSION go");
    args["permission"] = json!("ask");
    call_json(&tools, "spawn", args).await;

    let waiting = wait(&tools, "d", 30).await;
    assert_eq!(waiting["state"], "needs_permission", "{waiting}");
    let pending = &waiting["pending_permission"];
    assert_eq!(pending["tool_call"]["id"], "tc-1");
    let request_id = pending["request_id"].as_str().unwrap();

    let bad = call_err(
        &tools,
        "permit",
        json!({"name": "d", "request_id": request_id, "option_id": "nope"}),
    )
    .await;
    assert!(bad.contains("not offered"), "{bad}");

    call_json(
        &tools,
        "permit",
        json!({"name": "d", "request_id": request_id, "option_id": "allow-1"}),
    )
    .await;
    let done = wait(&tools, "d", 30).await;
    assert_eq!(done["state"], "done", "{done}");
    assert!(
        done["reply"].as_str().unwrap().contains("perm:allow-1"),
        "{done}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn permission_deny_policy() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let mut args = spawn_args("e", dir.path(), "PERMISSION go");
    args["permission"] = json!("deny");
    call_json(&tools, "spawn", args).await;
    let done = wait(&tools, "e", 30).await;
    assert_eq!(done["state"], "done", "{done}");
    assert!(
        done["reply"].as_str().unwrap().contains("perm:deny-1"),
        "{done}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn fs_read_inside_and_outside_cwd() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let inside = dir.path().join("inner.txt");
    std::fs::write(&inside, "inner-content").unwrap();
    let (_state, tools) = test_state(dir.path());

    call_json(
        &tools,
        "spawn",
        spawn_args("f", dir.path(), &format!("READ {}", inside.display())),
    )
    .await;
    let done = wait(&tools, "f", 30).await;
    assert!(
        done["reply"].as_str().unwrap().contains("fs:inner-content"),
        "{done}"
    );

    call_json(
        &tools,
        "spawn",
        spawn_args("g", dir.path(), "READ /etc/hosts"),
    )
    .await;
    let done = wait(&tools, "g", 30).await;
    let reply = done["reply"].as_str().unwrap();
    assert!(reply.contains("fserr:"), "{reply}");
    assert!(reply.contains("outside the session cwd"), "{reply}");

    // The wideopen agent may leave its cwd.
    call_json(
        &tools,
        "spawn",
        json!({"name": "h", "agent": "wideopen", "cwd": dir.path(), "prompt": "READ /etc/hosts"}),
    )
    .await;
    let done = wait(&tools, "h", 30).await;
    assert!(done["reply"].as_str().unwrap().contains("fs:"), "{done}");
    assert!(
        !done["reply"].as_str().unwrap().contains("fserr:"),
        "{done}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn terminal_run() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(
        &tools,
        "spawn",
        spawn_args("t", dir.path(), "RUN echo hello-term"),
    )
    .await;
    let done = wait(&tools, "t", 30).await;
    let reply = done["reply"].as_str().unwrap().to_string();
    assert!(reply.contains("term:hello-term"), "{reply}");
    assert!(reply.contains("exit:0"), "{reply}");
}

#[tokio::test(flavor = "multi_thread")]
async fn transcript_rendering() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("tr", dir.path(), "hi")).await;
    wait(&tools, "tr", 30).await;

    let text = call_text(&tools, "transcript", json!({"name": "tr"})).await;
    assert!(text.contains("=== [turn 1] USER\nhi"), "{text}");
    assert!(text.contains("--- ASSISTANT\nHello"), "{text}");
    assert!(text.contains("--- ASSISTANT\nab"), "{text}");
    assert!(text.contains("--- ASSISTANT\nworld"), "{text}");
    assert!(text.contains("--- CALL execute fake-tool (tc-1)"), "{text}");
    assert!(text.contains("--- RESULT (tc-1)"), "{text}");
    assert!(text.contains("--- PLAN\n"), "{text}");
    assert!(text.contains("=== [turn 1] END end_turn"), "{text}");
    assert!(!text.contains("THINKING"), "{text}");

    let with_thinking = call_text(
        &tools,
        "transcript",
        json!({"name": "tr", "thinking": true}),
    )
    .await;
    assert!(
        with_thinking.contains("--- THINKING\nthinking about it"),
        "{with_thinking}"
    );

    let tail = call_text(&tools, "transcript", json!({"name": "tr", "tail": 1})).await;
    assert!(tail.contains("END end_turn"), "{tail}");
    assert!(!tail.contains("USER"), "{tail}");
}

#[tokio::test(flavor = "multi_thread")]
async fn registry_resume_via_load() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("res", dir.path(), "hi")).await;
    wait(&tools, "res", 30).await;
    call_json(&tools, "close", json!({"name": "res"})).await;

    // Registered but not live: the fake agent advertises loadSession, so the
    // second spawn runs session/load and keeps the session id.
    let resumed = call_json(&tools, "spawn", spawn_args("res", dir.path(), "again")).await;
    assert_eq!(resumed["session_id"], "sess-1");
    let done = wait(&tools, "res", 30).await;
    assert_eq!(done["state"], "done");

    let status = call_json(&tools, "status", json!({"name": "res"})).await;
    assert_eq!(status["turns"], 2);
    let text = call_text(&tools, "transcript", json!({"name": "res"})).await;
    assert!(text.contains("--- USER\nloaded user message"), "{text}");
}

#[tokio::test(flavor = "multi_thread")]
async fn registry_no_load_requires_replace() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let mut args = spawn_args("nl", dir.path(), "hi");
    args["agent"] = json!("noload");
    call_json(&tools, "spawn", args.clone()).await;
    wait(&tools, "nl", 30).await;
    call_json(&tools, "close", json!({"name": "nl"})).await;

    let err = call_err(&tools, "spawn", args.clone()).await;
    assert!(err.contains("already registered"), "{err}");

    // replace=true starts a fresh session: the transcript gains a second turn
    // but no session/load replay record.
    args["replace"] = json!(true);
    call_json(&tools, "spawn", args).await;
    let done = wait(&tools, "nl", 30).await;
    assert_eq!(done["state"], "done");
    let text = call_text(&tools, "transcript", json!({"name": "nl"})).await;
    assert_eq!(text.matches("=== [turn 1] USER").count(), 2, "{text}");
    assert!(!text.contains("loaded user message"), "{text}");
}

#[tokio::test(flavor = "multi_thread")]
async fn list_and_unknown_names() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("l1", dir.path(), "hi")).await;
    wait(&tools, "l1", 30).await;

    let list = call_json(&tools, "list", json!({})).await;
    let subs = list["subagents"].as_array().unwrap();
    assert!(subs.iter().any(|s| s["name"] == "l1" && s["live"] == true));

    let err = call_err(&tools, "status", json!({"name": "nobody"})).await;
    assert!(err.contains("unknown subagent"), "{err}");
    let err = call_err(&tools, "spawn", spawn_args("bad name!", dir.path(), "hi")).await;
    assert!(err.contains("invalid subagent name"), "{err}");
    let err = call_err(
        &tools,
        "spawn",
        json!({"name": "x", "agent": "nobody", "cwd": dir.path(), "prompt": "hi"}),
    )
    .await;
    assert!(err.contains("unknown agent 'nobody'"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn wait_any_returns_first_done() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("w1", dir.path(), "wait")).await;
    call_json(&tools, "spawn", spawn_args("w2", dir.path(), "hi")).await;
    let first = call_json(
        &tools,
        "wait_any",
        json!({"names": ["w1", "w2"], "timeout_secs": 30}),
    )
    .await;
    assert_eq!(first["name"], "w2", "{first}");
    assert_eq!(first["state"], "done");
    call_json(&tools, "cancel", json!({"name": "w1"})).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn forget_removes_registry_entry() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("fg", dir.path(), "hi")).await;
    wait(&tools, "fg", 30).await;
    call_json(&tools, "forget", json!({"name": "fg"})).await;
    let err = call_err(&tools, "status", json!({"name": "fg"})).await;
    assert!(err.contains("unknown subagent"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn agents_tool_reports_initialized() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("ag", dir.path(), "hi")).await;
    wait(&tools, "ag", 30).await;
    let agents = call_json(&tools, "agents", json!({})).await;
    let fake = agents["agents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["name"] == "fake")
        .expect("fake agent");
    assert_eq!(fake["agent_info"]["name"], "fake-agent-py", "{fake}");
    assert_eq!(fake["modes"]["current"], "bypass", "{fake}");
    assert_eq!(fake["config_options"][0]["id"], "model", "{fake}");
    assert_eq!(fake["subagents"], json!(["ag"]));
}
