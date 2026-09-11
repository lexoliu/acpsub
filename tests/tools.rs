//! Integration tests: the tool set exercised in-process against
//! `tests/fake_agent.py`.

mod common;

use std::collections::BTreeMap;

use acpsub::config::AgentConfig;
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

/// The MCP server runs `tools/call`s concurrently: a blocked `wait` must
/// return when a `cancel` from another call lands.
#[tokio::test(flavor = "multi_thread")]
async fn wait_returns_when_cancel_runs_concurrently() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let tools = std::sync::Arc::new(tools);
    call_json(&tools, "spawn", spawn_args("w", dir.path(), "wait")).await;

    let waiter = {
        let tools = tools.clone();
        tokio::spawn(async move { wait(&tools, "w", 30).await })
    };
    // Let the prompt reach the agent so the cancel answers it.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    call_json(&tools, "cancel", json!({"name": "w"})).await;

    let done = waiter.await.expect("wait task panicked");
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

/// `devin acp` sends the whole shell command line as `command` with no
/// `args`; it must run through a shell, not be spawned as an executable.
#[tokio::test(flavor = "multi_thread")]
async fn terminal_run_shell_line() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(
        &tools,
        "spawn",
        spawn_args("t", dir.path(), "RUNLINE echo shell-$((40 + 2))"),
    )
    .await;
    let done = wait(&tools, "t", 30).await;
    let reply = done["reply"].as_str().unwrap().to_string();
    assert!(reply.contains("term:shell-42"), "{reply}");
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

/// The `die` prompt makes the agent exit mid-turn: the turn fails, the
/// subagent reports `failed`, and the registry entry survives so the session
/// can be resumed after `close`.
#[tokio::test(flavor = "multi_thread")]
async fn agent_death_mid_turn_marks_failed() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("d", dir.path(), "die")).await;
    let done = wait(&tools, "d", 30).await;
    assert_eq!(done["state"], "failed", "{done}");
    assert!(
        done["error"].as_str().is_some_and(|e| !e.is_empty()),
        "{done}"
    );

    let status = call_json(&tools, "status", json!({"name": "d"})).await;
    assert_eq!(status["state"], "failed");
    assert_eq!(status["live"], true);

    // The registry entry outlives the process: after `close` the name resumes
    // its session via session/load.
    call_json(&tools, "close", json!({"name": "d"})).await;
    let resumed = call_json(&tools, "spawn", spawn_args("d", dir.path(), "hi")).await;
    assert_eq!(resumed["session_id"], "sess-1");
    let done = wait(&tools, "d", 30).await;
    assert_eq!(done["state"], "done");
}

/// `KILL <cmd>` has the agent create a terminal running `sleep 60`, call
/// `terminal/kill`, then `wait_for_exit`: the signal exit reaches the reply.
#[tokio::test(flavor = "multi_thread")]
async fn terminal_kill_reports_signal_exit() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(
        &tools,
        "spawn",
        spawn_args("k", dir.path(), "KILL sleep 60"),
    )
    .await;
    let done = wait(&tools, "k", 30).await;
    assert_eq!(done["state"], "done", "{done}");
    let reply = done["reply"].as_str().unwrap();
    assert!(reply.contains("exit:None"), "{reply}");
    assert!(reply.contains("signal:9"), "{reply}");
}

/// `close` ends the process but keeps the registry entry: the name still
/// lists as `closed`, and `send` to it errors.
#[tokio::test(flavor = "multi_thread")]
async fn close_keeps_name_registered() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("cl", dir.path(), "hi")).await;
    wait(&tools, "cl", 30).await;

    let closed = call_json(&tools, "close", json!({"name": "cl"})).await;
    assert_eq!(closed["closed"], true);

    let status = call_json(&tools, "status", json!({"name": "cl"})).await;
    assert_eq!(status["live"], false);
    assert_eq!(status["state"], "closed");
    let list = call_json(&tools, "list", json!({})).await;
    let entry = list["subagents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"] == "cl")
        .expect("cl still listed");
    assert_eq!(entry["live"], false);
    assert_eq!(entry["state"], "closed");

    let err = call_err(&tools, "send", json!({"name": "cl", "prompt": "again"})).await;
    assert!(err.contains("unknown subagent"), "{err}");
}

/// `WRITE <path>` has the agent call `fs/write_text_file`: writes inside the
/// session cwd succeed, writes outside are refused unless the agent sets
/// `allow_outside_cwd`.
#[tokio::test(flavor = "multi_thread")]
async fn fs_write_inside_and_outside_cwd() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let nested = dir.path().join("nested");
    std::fs::create_dir(&nested).unwrap();
    let inside = nested.join("written.txt");
    let (_state, tools) = test_state(dir.path());

    call_json(
        &tools,
        "spawn",
        spawn_args("wi", dir.path(), &format!("WRITE {}", inside.display())),
    )
    .await;
    let done = wait(&tools, "wi", 30).await;
    assert!(done["reply"].as_str().unwrap().contains("fsw:ok"), "{done}");
    assert_eq!(
        std::fs::read_to_string(&inside).unwrap(),
        "written-by-fake-agent"
    );

    let outside = tempfile::tempdir().unwrap();
    let outside_file = outside.path().join("nope.txt");
    call_json(
        &tools,
        "spawn",
        spawn_args(
            "wo",
            dir.path(),
            &format!("WRITE {}", outside_file.display()),
        ),
    )
    .await;
    let done = wait(&tools, "wo", 30).await;
    let reply = done["reply"].as_str().unwrap();
    assert!(reply.contains("fswerr:"), "{reply}");
    assert!(reply.contains("outside the session cwd"), "{reply}");
    assert!(!outside_file.exists());

    // The wideopen agent may write outside its cwd.
    call_json(
        &tools,
        "spawn",
        json!({"name": "ww", "agent": "wideopen", "cwd": dir.path(),
               "prompt": format!("WRITE {}", outside_file.display())}),
    )
    .await;
    let done = wait(&tools, "ww", 30).await;
    assert!(done["reply"].as_str().unwrap().contains("fsw:ok"), "{done}");
    assert_eq!(
        std::fs::read_to_string(&outside_file).unwrap(),
        "written-by-fake-agent"
    );
}

/// A name held by a live subagent cannot be spawned again; once `forget`
/// releases it, the name is reusable.
#[tokio::test(flavor = "multi_thread")]
async fn duplicate_live_name_rejected_until_forgotten() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("dup", dir.path(), "hi")).await;
    wait(&tools, "dup", 30).await;

    let err = call_err(&tools, "spawn", spawn_args("dup", dir.path(), "again")).await;
    assert!(err.contains("already running"), "{err}");

    call_json(&tools, "forget", json!({"name": "dup"})).await;
    let spawned = call_json(&tools, "spawn", spawn_args("dup", dir.path(), "again")).await;
    assert_eq!(spawned["state"], "running");
    let done = wait(&tools, "dup", 30).await;
    assert_eq!(done["state"], "done");
}

/// `send` only accepts `idle`/`done`/`cancelled` subagents; a `running` one
/// rejects the prompt.
#[tokio::test(flavor = "multi_thread")]
async fn send_to_running_subagent_rejected() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("run", dir.path(), "wait")).await;

    let err = call_err(
        &tools,
        "send",
        json!({"name": "run", "prompt": "more work"}),
    )
    .await;
    assert!(err.contains("cannot accept a prompt"), "{err}");

    call_json(&tools, "cancel", json!({"name": "run"})).await;
}

/// An agent whose command does not exist fails `spawn` cleanly and releases
/// the name it reserved.
#[tokio::test(flavor = "multi_thread")]
async fn spawn_with_missing_command_fails() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let agents = BTreeMap::from([
        ("fake".to_string(), fake_agent_config()),
        (
            "missing".to_string(),
            AgentConfig {
                command: "acpsub-no-such-binary".to_string(),
                ..fake_agent_config()
            },
        ),
    ]);
    let (_state, tools) = test_state_with_agents(dir.path(), agents);

    let err = call_err(
        &tools,
        "spawn",
        json!({"name": "b", "agent": "missing", "cwd": dir.path(), "prompt": "hi"}),
    )
    .await;
    assert!(err.contains("cannot spawn agent"), "{err}");

    // The failed spawn released the name: a working agent can take it.
    let spawned = call_json(&tools, "spawn", spawn_args("b", dir.path(), "hi")).await;
    assert_eq!(spawned["state"], "running");
    wait(&tools, "b", 30).await;
}

/// Two concurrent `spawn` calls for one name: the reservation is atomic, so
/// exactly one succeeds and the other is told the name is taken.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_spawn_same_name_runs_once() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let tools = std::sync::Arc::new(tools);
    let cwd = dir.path().to_path_buf();
    let first = {
        let tools = tools.clone();
        let cwd = cwd.clone();
        tokio::spawn(async move { call(&tools, "spawn", spawn_args("cc", &cwd, "hi")).await })
    };
    let second = {
        let tools = tools.clone();
        tokio::spawn(async move { call(&tools, "spawn", spawn_args("cc", &cwd, "hi")).await })
    };

    let mut succeeded = 0;
    let mut errors = Vec::new();
    for outcome in [first.await.expect("task"), second.await.expect("task")] {
        match outcome {
            Ok(result) if result.is_error() => {
                errors.push(result.error_message().unwrap_or("?").to_string());
            }
            Ok(_) => succeeded += 1,
            Err(error) => errors.push(error),
        }
    }
    assert_eq!(succeeded, 1, "{errors:?}");
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("already running"), "{}", errors[0]);

    let done = wait(&tools, "cc", 30).await;
    assert_eq!(done["state"], "done");
}

/// `seq 1 5000` emits ~23 KB, over the 4096-byte `outputByteLimit` the fake
/// agent requests: `terminal/output` reports `truncated` and retains the
/// newest bytes.
#[tokio::test(flavor = "multi_thread")]
async fn terminal_output_truncates_to_tail() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(
        &tools,
        "spawn",
        spawn_args("big", dir.path(), "RUNLINE seq 1 5000"),
    )
    .await;
    let done = wait(&tools, "big", 30).await;
    let reply = done["reply"].as_str().unwrap();
    assert!(reply.contains("trunc"), "{reply}");
    assert!(reply.contains("5000"), "{reply}");
    // The oldest bytes were evicted: the retained tail starts mid-sequence.
    assert!(!reply.contains("term:1\n2\n3"), "{reply}");
}

/// After `cancel`, `result` still returns the turn and `transcript` renders
/// its `cancelled` end record.
#[tokio::test(flavor = "multi_thread")]
async fn result_and_transcript_after_cancel() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("cx", dir.path(), "wait")).await;
    wait(&tools, "cx", 2).await;
    call_json(&tools, "cancel", json!({"name": "cx"})).await;
    let done = wait(&tools, "cx", 30).await;
    assert_eq!(done["state"], "cancelled");

    let result = call_json(&tools, "result", json!({"name": "cx"})).await;
    assert_eq!(result["turn"], 1);
    assert_eq!(result["reply"], "Hello ");
    assert_eq!(result["stop_reason"], "cancelled");

    let text = call_text(&tools, "transcript", json!({"name": "cx"})).await;
    assert!(text.contains("=== [turn 1] END cancelled"), "{text}");
}

/// `permit` with a request id that is not pending errors cleanly.
#[tokio::test(flavor = "multi_thread")]
async fn permit_with_unknown_request_id() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    call_json(&tools, "spawn", spawn_args("p", dir.path(), "hi")).await;
    wait(&tools, "p", 30).await;

    let err = call_err(
        &tools,
        "permit",
        json!({"name": "p", "request_id": "perm-99", "option_id": "allow-1"}),
    )
    .await;
    assert!(err.contains("no pending permission request"), "{err}");
}
