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
    let spawned = call_json(&tools, "spawn", spawn_args(dir.path(), "hi")).await;
    assert_eq!(spawned["state"], "running");
    let sid = spawned["session_id"].as_str().expect("session_id");
    assert!(sid.starts_with("sess-"), "{sid}");
    assert!(spawned.get("name").is_none(), "no name handle: {spawned}");

    let done = wait(&tools, sid, 60).await;
    assert_eq!(done["state"], "done");
    assert_eq!(done["stop_reason"], "end_turn");
    assert_eq!(done["reply"], "Hello abworld", "{done}");
    assert_eq!(done["tool_calls"][0]["id"], "tc-1");
    assert_eq!(done["tool_calls"][0]["status"], "completed");

    let result = call_json(&tools, "result", json!({"session_id": sid})).await;
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
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;

    call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "again"}),
    )
    .await;
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done");

    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["turns"], 2);
    assert_eq!(status["state"], "done");
    assert_eq!(status["last_stop_reason"], "end_turn");

    let first = call_json(
        &tools,
        "result",
        json!({"session_id": sid.as_str(), "turn": 1}),
    )
    .await;
    assert_eq!(first["turn"], 1);
    let missing = call_err(
        &tools,
        "result",
        json!({"session_id": sid.as_str(), "turn": 9}),
    )
    .await;
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
    let sid = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;
    let running = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(running["state"], "running");

    call_json(&tools, "cancel", json!({"session_id": sid.as_str()})).await;
    let done = wait(&tools, &sid, 60).await;
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
    let sid = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;

    let waiter = {
        let tools = tools.clone();
        let sid = sid.clone();
        tokio::spawn(async move { wait(&tools, &sid, 60).await })
    };
    // Let the prompt reach the agent so the cancel answers it.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    call_json(&tools, "cancel", json!({"session_id": sid.as_str()})).await;

    let done = waiter.await.expect("wait task panicked");
    assert_eq!(done["state"], "cancelled", "{done}");
    assert_eq!(done["stop_reason"], "cancelled");
}

/// Timeouts under the 60s floor are rejected on both wait tools; an instant
/// check is `status`'s job.
#[tokio::test(flavor = "multi_thread")]
async fn wait_rejects_timeout_below_min() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;

    let err = call_err(
        &tools,
        "wait",
        json!({"session_id": sid.as_str(), "timeout_secs": 30}),
    )
    .await;
    assert!(err.contains("below the 60s minimum"), "{err}");
    let err = call_err(
        &tools,
        "wait_any",
        json!({"session_ids": [sid.as_str()], "timeout_secs": 1}),
    )
    .await;
    assert!(err.contains("below the 60s minimum"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn permission_ask_then_permit() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let mut args = spawn_args(dir.path(), "PERMISSION go");
    args["permission"] = json!("ask");
    let sid = spawn_id(&tools, args).await;

    let waiting = wait(&tools, &sid, 60).await;
    assert_eq!(waiting["state"], "needs_permission", "{waiting}");
    let pending = &waiting["pending_permission"];
    assert_eq!(pending["tool_call"]["id"], "tc-1");
    let request_id = pending["request_id"].as_str().unwrap();

    let bad = call_err(
        &tools,
        "permit",
        json!({"session_id": sid.as_str(), "request_id": request_id, "option_id": "nope"}),
    )
    .await;
    assert!(bad.contains("not offered"), "{bad}");

    call_json(
        &tools,
        "permit",
        json!({"session_id": sid.as_str(), "request_id": request_id, "option_id": "allow-1"}),
    )
    .await;
    let done = wait(&tools, &sid, 60).await;
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
    let mut args = spawn_args(dir.path(), "PERMISSION go");
    args["permission"] = json!("deny");
    let sid = spawn_id(&tools, args).await;
    let done = wait(&tools, &sid, 60).await;
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

    let sid = spawn_id(
        &tools,
        spawn_args(dir.path(), &format!("READ {}", inside.display())),
    )
    .await;
    let done = wait(&tools, &sid, 60).await;
    assert!(
        done["reply"].as_str().unwrap().contains("fs:inner-content"),
        "{done}"
    );

    let sid = spawn_id(&tools, spawn_args(dir.path(), "READ /etc/hosts")).await;
    let done = wait(&tools, &sid, 60).await;
    let reply = done["reply"].as_str().unwrap();
    assert!(reply.contains("fserr:"), "{reply}");
    assert!(reply.contains("outside the session cwd"), "{reply}");

    // The wideopen agent may leave its cwd.
    let sid = spawn_id(
        &tools,
        json!({"agent": "wideopen", "cwd": dir.path(), "prompt": "READ /etc/hosts"}),
    )
    .await;
    let done = wait(&tools, &sid, 60).await;
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
    let sid = spawn_id(&tools, spawn_args(dir.path(), "RUN echo hello-term")).await;
    let done = wait(&tools, &sid, 60).await;
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
    let sid = spawn_id(
        &tools,
        spawn_args(dir.path(), "RUNLINE echo shell-$((40 + 2))"),
    )
    .await;
    let done = wait(&tools, &sid, 60).await;
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
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;

    let text = call_text(&tools, "transcript", json!({"session_id": sid.as_str()})).await;
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
        json!({"session_id": sid.as_str(), "thinking": true}),
    )
    .await;
    assert!(
        with_thinking.contains("--- THINKING\nthinking about it"),
        "{with_thinking}"
    );

    let tail = call_text(
        &tools,
        "transcript",
        json!({"session_id": sid.as_str(), "tail": 1}),
    )
    .await;
    assert!(tail.contains("END end_turn"), "{tail}");
    assert!(!tail.contains("USER"), "{tail}");
}

/// `adopt` on a registered session resumes it via `session/load` — the only
/// resume path; `spawn` always starts fresh.
#[tokio::test(flavor = "multi_thread")]
async fn adopt_registered_session_resumes() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;
    call_json(&tools, "close", json!({"session_id": sid.as_str()})).await;

    // Registry knows the session: agent and cwd resolve without arguments.
    let adopted = call_json(
        &tools,
        "adopt",
        json!({"session_id": sid.as_str(), "prompt": "again"}),
    )
    .await;
    assert_eq!(adopted["session_id"], sid);
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done");

    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["turns"], 2);
    let text = call_text(&tools, "transcript", json!({"session_id": sid.as_str()})).await;
    assert!(text.contains("--- USER\nloaded user message"), "{text}");
}

/// `adopt` without `prompt` binds the session and leaves it idle; `send`
/// starts the first turn.
#[tokio::test(flavor = "multi_thread")]
async fn adopt_without_prompt_idles_until_send() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;
    call_json(&tools, "close", json!({"session_id": sid.as_str()})).await;

    let adopted = call_json(&tools, "adopt", json!({"session_id": sid.as_str()})).await;
    assert_eq!(adopted["state"], "idle", "{adopted}");

    call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "again"}),
    )
    .await;
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done");
    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["turns"], 2);
}

/// An agent without `loadSession` cannot adopt.
#[tokio::test(flavor = "multi_thread")]
async fn adopt_requires_load_capability() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let mut args = spawn_args(dir.path(), "hi");
    args["agent"] = json!("noload");
    let sid = spawn_id(&tools, args).await;
    wait(&tools, &sid, 60).await;
    call_json(&tools, "close", json!({"session_id": sid.as_str()})).await;

    let err = call_err(&tools, "adopt", json!({"session_id": sid.as_str()})).await;
    assert!(err.contains("does not support session/load"), "{err}");
}

/// `adopt` on a session acpsub never saw needs only an explicit `cwd`.
#[tokio::test(flavor = "multi_thread")]
async fn adopt_external_session_with_explicit_cwd() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let adopted = call_json(
        &tools,
        "adopt",
        json!({"session_id": "ext-42", "agent": "fake",
               "cwd": dir.path(), "prompt": "hi"}),
    )
    .await;
    assert_eq!(adopted["session_id"], "ext-42");
    let done = wait(&tools, "ext-42", 60).await;
    assert_eq!(done["state"], "done");
}

/// With no registry entry and no session database, `adopt` cannot guess the
/// cwd and asks for it.
#[tokio::test(flavor = "multi_thread")]
async fn adopt_without_discoverable_cwd_errors() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let err = call_err(
        &tools,
        "adopt",
        json!({"session_id": "ghost", "agent": "fake"}),
    )
    .await;
    assert!(err.contains("cannot determine cwd"), "{err}");
    assert!(err.contains("pass `cwd` explicitly"), "{err}");
}

/// A `sessions_db` maps unknown session ids to their working directory, so
/// `adopt` does not need `cwd`.
#[tokio::test(flavor = "multi_thread")]
async fn adopt_discovers_cwd_from_sessions_db() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("sessions.db");
    {
        let conn = rusqlite::Connection::open(&db_path).expect("open db");
        conn.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, working_directory TEXT NOT NULL)",
            [],
        )
        .expect("create table");
        conn.execute(
            "INSERT INTO sessions (id, working_directory) VALUES (?1, ?2)",
            rusqlite::params!["ext-9", dir.path().to_string_lossy()],
        )
        .expect("insert row");
    }
    let fake = AgentConfig {
        sessions_db: Some(db_path),
        ..fake_agent_config()
    };
    let (_state, tools) =
        test_state_with_agents(dir.path(), BTreeMap::from([("fake".to_string(), fake)]));

    let adopted = call_json(
        &tools,
        "adopt",
        json!({"session_id": "ext-9", "prompt": "hi"}),
    )
    .await;
    assert_eq!(adopted["session_id"], "ext-9");
    let done = wait(&tools, "ext-9", 60).await;
    assert_eq!(done["state"], "done");
    let status = call_json(&tools, "status", json!({"session_id": "ext-9"})).await;
    assert_eq!(
        status["cwd"].as_str().map(std::path::Path::new),
        Some(dir.path().canonicalize().unwrap().as_path()),
        "{status}"
    );
}

/// `adopt` on a live session id is refused; so is a second concurrent one.
#[tokio::test(flavor = "multi_thread")]
async fn adopt_rejects_live_and_concurrent() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;

    let err = call_err(&tools, "adopt", json!({"session_id": sid.as_str()})).await;
    assert!(err.contains("already live"), "{err}");

    call_json(&tools, "close", json!({"session_id": sid.as_str()})).await;
    let tools = std::sync::Arc::new(tools);
    let first = {
        let tools = tools.clone();
        let sid = sid.clone();
        tokio::spawn(
            async move { call(&tools, "adopt", json!({"session_id": sid.as_str()})).await },
        )
    };
    let second = {
        let tools = tools.clone();
        tokio::spawn(
            async move { call(&tools, "adopt", json!({"session_id": sid.as_str()})).await },
        )
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
    assert!(errors[0].contains("already live"), "{}", errors[0]);
}

/// The registry records which agent owns a session; an explicit `agent`
/// that disagrees is rejected before any process starts.
#[tokio::test(flavor = "multi_thread")]
async fn adopt_agent_mismatch_rejected() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;
    call_json(&tools, "close", json!({"session_id": sid.as_str()})).await;

    let err = call_err(
        &tools,
        "adopt",
        json!({"session_id": sid.as_str(), "agent": "noload"}),
    )
    .await;
    assert!(err.contains("registered to agent 'fake'"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn adopt_empty_session_id_rejected() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let err = call_err(
        &tools,
        "adopt",
        json!({"session_id": "", "agent": "fake", "cwd": dir.path()}),
    )
    .await;
    assert!(err.contains("must not be empty"), "{err}");
}

/// `agent` is optional: `[defaults] agent` wins, then a single configured
/// agent; several agents without a default is an error.
#[tokio::test(flavor = "multi_thread")]
async fn default_agent_resolution() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();

    // No default, three agents: omitting `agent` errors.
    let (_state, tools) = test_state(dir.path());
    let err = call_err(&tools, "spawn", json!({"cwd": dir.path(), "prompt": "hi"})).await;
    assert!(err.contains("no agent specified"), "{err}");

    // A configured default is used.
    let (_state, tools) = test_state_full(
        dir.path(),
        Some("fake"),
        BTreeMap::from([
            ("fake".to_string(), fake_agent_config()),
            (
                "other".to_string(),
                AgentConfig {
                    command: "acpsub-no-such-binary".to_string(),
                    ..fake_agent_config()
                },
            ),
        ]),
    );
    let sid = spawn_id(&tools, json!({"cwd": dir.path(), "prompt": "hi"})).await;
    wait(&tools, &sid, 60).await;

    // A single configured agent is the implicit default.
    let dir2 = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state_with_agents(
        dir2.path(),
        BTreeMap::from([("fake".to_string(), fake_agent_config())]),
    );
    let sid = spawn_id(&tools, json!({"cwd": dir2.path(), "prompt": "hi"})).await;
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done");
}

#[tokio::test(flavor = "multi_thread")]
async fn list_and_unknown_sessions() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;

    let list = call_json(&tools, "list", json!({})).await;
    let subs = list["subagents"].as_array().unwrap();
    assert!(
        subs.iter()
            .any(|s| s["session_id"] == sid && s["live"] == true)
    );

    let err = call_err(&tools, "status", json!({"session_id": "nobody"})).await;
    assert!(err.contains("unknown session"), "{err}");
    let err = call_err(
        &tools,
        "spawn",
        json!({"agent": "nobody", "cwd": dir.path(), "prompt": "hi"}),
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
    let sid1 = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;
    let sid2 = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    let first = call_json(
        &tools,
        "wait_any",
        json!({"session_ids": [sid1.as_str(), sid2.as_str()], "timeout_secs": 60}),
    )
    .await;
    assert_eq!(first["session_id"], sid2, "{first}");
    assert_eq!(first["state"], "done");
    call_json(&tools, "cancel", json!({"session_id": sid1})).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn forget_removes_registry_entry() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;
    call_json(&tools, "forget", json!({"session_id": sid.as_str()})).await;
    let err = call_err(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert!(err.contains("unknown session"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn agents_tool_reports_initialized() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;
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
    assert_eq!(fake["subagents"], json!([sid]));
}

/// The `die` prompt makes the agent exit mid-turn: the turn fails, the
/// subagent reports `failed`, and the registry entry survives so the session
/// can be adopted after `close`.
#[tokio::test(flavor = "multi_thread")]
async fn agent_death_mid_turn_marks_failed() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "die")).await;
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "failed", "{done}");
    assert!(
        done["error"].as_str().is_some_and(|e| !e.is_empty()),
        "{done}"
    );

    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["state"], "failed");
    assert_eq!(status["live"], true);

    // The registry entry outlives the process: after `close` the session is
    // adoptable again via session/load.
    call_json(&tools, "close", json!({"session_id": sid.as_str()})).await;
    let adopted = call_json(
        &tools,
        "adopt",
        json!({"session_id": sid.as_str(), "prompt": "hi"}),
    )
    .await;
    assert_eq!(adopted["session_id"], sid);
    let done = wait(&tools, &sid, 60).await;
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
    let sid = spawn_id(&tools, spawn_args(dir.path(), "KILL sleep 60")).await;
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done", "{done}");
    let reply = done["reply"].as_str().unwrap();
    assert!(reply.contains("exit:None"), "{reply}");
    assert!(reply.contains("signal:9"), "{reply}");
}

/// `close` ends the process but keeps the registry entry: the session still
/// lists as `closed`, and `send` to it errors.
#[tokio::test(flavor = "multi_thread")]
async fn close_keeps_session_registered() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;

    let closed = call_json(&tools, "close", json!({"session_id": sid.as_str()})).await;
    assert_eq!(closed["closed"], true);

    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["live"], false);
    assert_eq!(status["state"], "closed");
    let list = call_json(&tools, "list", json!({})).await;
    let entry = list["subagents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["session_id"] == sid)
        .expect("session still listed");
    assert_eq!(entry["live"], false);
    assert_eq!(entry["state"], "closed");

    let err = call_err(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "again"}),
    )
    .await;
    assert!(err.contains("unknown session"), "{err}");
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

    let sid = spawn_id(
        &tools,
        spawn_args(dir.path(), &format!("WRITE {}", inside.display())),
    )
    .await;
    let done = wait(&tools, &sid, 60).await;
    assert!(done["reply"].as_str().unwrap().contains("fsw:ok"), "{done}");
    assert_eq!(
        std::fs::read_to_string(&inside).unwrap(),
        "written-by-fake-agent"
    );

    let outside = tempfile::tempdir().unwrap();
    let outside_file = outside.path().join("nope.txt");
    let sid = spawn_id(
        &tools,
        spawn_args(dir.path(), &format!("WRITE {}", outside_file.display())),
    )
    .await;
    let done = wait(&tools, &sid, 60).await;
    let reply = done["reply"].as_str().unwrap();
    assert!(reply.contains("fswerr:"), "{reply}");
    assert!(reply.contains("outside the session cwd"), "{reply}");
    assert!(!outside_file.exists());

    // The wideopen agent may write outside its cwd.
    let sid = spawn_id(
        &tools,
        json!({"agent": "wideopen", "cwd": dir.path(),
               "prompt": format!("WRITE {}", outside_file.display())}),
    )
    .await;
    let done = wait(&tools, &sid, 60).await;
    assert!(done["reply"].as_str().unwrap().contains("fsw:ok"), "{done}");
    assert_eq!(
        std::fs::read_to_string(&outside_file).unwrap(),
        "written-by-fake-agent"
    );
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
    let sid = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;

    let err = call_err(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "more work"}),
    )
    .await;
    assert!(err.contains("cannot accept a prompt"), "{err}");

    call_json(&tools, "cancel", json!({"session_id": sid.as_str()})).await;
}

/// An agent whose command does not exist fails `spawn` cleanly.
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
        json!({"agent": "missing", "cwd": dir.path(), "prompt": "hi"}),
    )
    .await;
    assert!(err.contains("cannot spawn agent"), "{err}");

    // A working agent still spawns.
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    let done = wait(&tools, &sid, 60).await;
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
    let sid = spawn_id(&tools, spawn_args(dir.path(), "RUNLINE seq 1 5000")).await;
    let done = wait(&tools, &sid, 60).await;
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
    let sid = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;
    let running = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(running["state"], "running", "{running}");
    call_json(&tools, "cancel", json!({"session_id": sid.as_str()})).await;
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "cancelled");

    let result = call_json(&tools, "result", json!({"session_id": sid.as_str()})).await;
    assert_eq!(result["turn"], 1);
    assert_eq!(result["reply"], "Hello ");
    assert_eq!(result["stop_reason"], "cancelled");

    let text = call_text(&tools, "transcript", json!({"session_id": sid.as_str()})).await;
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
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;

    let err = call_err(
        &tools,
        "permit",
        json!({"session_id": sid.as_str(), "request_id": "perm-99", "option_id": "allow-1"}),
    )
    .await;
    assert!(err.contains("no pending permission request"), "{err}");
}
