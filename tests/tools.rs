//! Integration tests: the tool set exercised in-process against
//! `tests/fake_agent.py`.

mod common;

use std::collections::BTreeMap;

use acpsub::Status;
use acpsub::config::AgentConfig;
use aither_core::llm::tool::Tools;
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
    // The result reports the model and mode the agent accepted.
    assert_eq!(spawned["model"], "b", "{spawned}");
    assert_eq!(spawned["mode"], "bypass", "{spawned}");

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
        json!({"session_id": sid.as_str(), "prompt": "again", "policy": "try"}),
    )
    .await;
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done");

    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["turns"], 2);
    assert_eq!(status["state"], "done");
    assert_eq!(status["last_stop_reason"], "end_turn");
    assert_eq!(status["model"], "b", "{status}");
    assert_eq!(status["mode"], "bypass", "{status}");

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

/// `policy` is a required argument: nothing is queued or injected unless
/// the caller says so.
#[tokio::test(flavor = "multi_thread")]
async fn send_requires_explicit_policy() {
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
        "send",
        json!({"session_id": sid.as_str(), "prompt": "again"}),
    )
    .await;
    assert!(err.contains("policy"), "{err}");
}

/// `policy: "try"` on a running subagent is the original rejection.
#[tokio::test(flavor = "multi_thread")]
async fn send_try_rejects_running() {
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
        json!({"session_id": sid.as_str(), "prompt": "hi", "policy": "try"}),
    )
    .await;
    assert!(err.contains("running"), "{err}");
    assert!(err.contains("cannot accept a prompt"), "{err}");
    call_json(&tools, "cancel", json!({"session_id": sid.as_str()})).await;
}

/// `policy: "queued"` parks the prompt; it fires as the next turn when the
/// running one ends — here ended by a steer into the `gather` prompt.
#[tokio::test(flavor = "multi_thread")]
async fn send_queued_fires_after_turn() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "gather")).await;

    let queued = call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "second", "policy": "queued"}),
    )
    .await;
    assert_eq!(queued["state"], "queued", "{queued}");
    assert_eq!(queued["position"], 1);

    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["queued"], json!(["second"]), "{status}");

    let steered = call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "kick", "policy": "steer"}),
    )
    .await;
    assert_eq!(steered["state"], "steered", "{steered}");
    assert_eq!(steered["turn"], 1);

    // The queued prompt runs as turn 2 with no done interlude: the state
    // stays `running` through the handoff, so `done` can only mean turn 2
    // ended. A bare `wait` could instead bind to turn 1 and return before
    // the chain completed.
    status_becomes(&state, &sid, "done").await;
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done", "{done}");
    assert_eq!(done["turn"], 2, "{done}");
    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["turns"], 2);
    assert_eq!(status["queued"], json!([]));

    let first = call_json(
        &tools,
        "result",
        json!({"session_id": sid.as_str(), "turn": 1}),
    )
    .await;
    assert!(
        first["reply"].as_str().unwrap().contains("steer:kick"),
        "{first}"
    );

    let text = call_text(&tools, "transcript", json!({"session_id": sid.as_str()})).await;
    assert!(text.contains("=== [turn 1] STEER\nkick"), "{text}");
    assert!(text.contains("=== [turn 1] STEER END end_turn"), "{text}");
    assert!(
        text.contains("=== [turn 2] USER (queued)\nsecond"),
        "{text}"
    );
}

/// Queued prompts drop when the running turn is cancelled.
#[tokio::test(flavor = "multi_thread")]
async fn send_queued_dropped_on_cancel() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;

    for (prompt, position) in [("next", 1), ("third", 2)] {
        let queued = call_json(
            &tools,
            "send",
            json!({"session_id": sid.as_str(), "prompt": prompt, "policy": "queued"}),
        )
        .await;
        assert_eq!(queued["state"], "queued");
        assert_eq!(queued["position"], position);
    }

    call_json(&tools, "cancel", json!({"session_id": sid.as_str()})).await;
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "cancelled", "{done}");

    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["queued"], json!([]), "{status}");
    assert_eq!(status["turns"], 1);

    let text = call_text(&tools, "transcript", json!({"session_id": sid.as_str()})).await;
    assert!(text.contains("QUEUE DROPPED"), "{text}");
    assert!(text.contains("next"), "{text}");
    assert!(text.contains("third"), "{text}");

    // A cancelled subagent accepts a fresh prompt.
    let sent = call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "hi", "policy": "try"}),
    )
    .await;
    assert_eq!(sent["state"], "running");
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done");
}

/// Queued prompts drop when the agent dies; the steer that kills it
/// surfaces its error to the `send` caller and the session ends `exited`.
#[tokio::test(flavor = "multi_thread")]
async fn send_queued_dropped_on_failure() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "gather")).await;

    let queued = call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "next", "policy": "queued"}),
    )
    .await;
    assert_eq!(queued["state"], "queued");

    // "die" exits the agent process; both the turn and the steer error out.
    let err = call_err(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "die", "policy": "steer"}),
    )
    .await;
    assert_ne!(err, "");

    let done = wait(&tools, &sid, 60).await;
    // The turn error and the process exit race; `exited` always wins.
    assert!(
        matches!(done["state"].as_str(), Some("failed" | "exited")),
        "{done}"
    );
    status_becomes(&state, &sid, "exited").await;
    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["state"], "exited", "{status}");
    assert_eq!(status["queued"], json!([]), "{status}");
}

/// A turn that fails while the agent process stays alive leaves the session
/// `failed` — resumable: `send` starts the next turn on the same ACP
/// session and its reply arrives (issue #51).
#[tokio::test(flavor = "multi_thread")]
async fn send_resumes_after_failed_turn() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    // "fail" makes the agent answer session/prompt with a JSON-RPC error
    // while staying alive — a dropped model stream, not a dead process.
    let sid = spawn_id(&tools, spawn_args(dir.path(), "fail")).await;
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "failed", "{done}");

    let sent = call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "hi", "policy": "try"}),
    )
    .await;
    assert_eq!(sent["state"], "running", "{sent}");
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done", "{done}");
    assert_eq!(done["reply"], "Hello abworld", "{done}");

    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["turns"], 2, "{status}");
    assert_eq!(status["state"], "done", "{status}");
}

/// A `steer` on a promptable subagent just starts the turn.
#[tokio::test(flavor = "multi_thread")]
async fn send_steer_on_idle_starts_turn() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;

    let sent = call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "again", "policy": "steer"}),
    )
    .await;
    assert_eq!(sent["state"], "running", "{sent}");
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done");
}

/// An agent that rejects a concurrent prompt surfaces the rejection.
#[tokio::test(flavor = "multi_thread")]
async fn send_steer_rejection_errors() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let mut args = spawn_args(dir.path(), "wait");
    args["agent"] = json!("nosteer");
    let sid = spawn_id(&tools, args).await;

    let err = call_err(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "hi", "policy": "steer"}),
    )
    .await;
    assert!(err.contains("prompt in flight"), "{err}");
    call_json(&tools, "cancel", json!({"session_id": sid.as_str()})).await;
}

/// On an agent that serializes a concurrent prompt instead of injecting it,
/// a steer that lands behind the running turn becomes a turn of its own:
/// acpsub materializes it from the untracked `session/update` traffic and
/// settles it when the steer's `session/prompt` resolves. The client-side
/// queue holds until the steer-turn ends — it was sent after the steer on
/// the wire — then chains without a `done` interlude.
#[tokio::test(flavor = "multi_thread")]
async fn send_steer_becomes_turn_on_serializing_agent() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());
    let mut args = spawn_args(dir.path(), "hold");
    args["agent"] = json!("queueagent");
    let sid = spawn_id(&tools, args).await;

    // Turn 1 holds; park the queued prompt client-side while it runs.
    let queued = call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "RUN echo queue-fired", "policy": "queued"}),
    )
    .await;
    assert_eq!(queued["state"], "queued", "{queued}");

    // The steer parks agent-side, releasing turn 1; it then runs as turn 2.
    let steered = call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "PERMISSION", "policy": "steer"}),
    )
    .await;
    assert_eq!(steered["state"], "steered", "{steered}");
    assert_eq!(steered["turn"], 1, "{steered}");

    // Wait on the real signal, not a poll deadline: `rt.notify` fires when
    // turn 1 settles, when the steered prompt materializes as its own turn,
    // and when each turn ends — so block on it until all three turns are
    // recorded. The timeout only turns a hang into a failure.
    let sub = state.get(&sid).expect("live subagent");
    let three_turns = async {
        loop {
            let notified = sub.rt.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let inner = sub.rt.inner.lock().expect("inner poisoned");
                if inner.turns.len() == 3 && matches!(inner.status, Status::Done(_)) {
                    return;
                }
            }
            notified.await;
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(60), three_turns)
        .await
        .expect("subagent did not record three turns");

    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["state"], "done", "{status}");
    assert_eq!(status["turns"], 3, "{status}");
    assert_eq!(status["queued"], json!([]), "{status}");

    let second = call_json(
        &tools,
        "result",
        json!({"session_id": sid.as_str(), "turn": 2}),
    )
    .await;
    assert!(
        second["reply"].as_str().unwrap().contains("perm:allow-1"),
        "{second}"
    );
    let third = call_json(
        &tools,
        "result",
        json!({"session_id": sid.as_str(), "turn": 3}),
    )
    .await;
    assert!(
        third["reply"]
            .as_str()
            .unwrap()
            .contains("term:queue-fired"),
        "{third}"
    );
}

/// On an agent declared `steer = "folded"`, a steer the agent folds into
/// the running turn is never answered: when the target turn ends the steer
/// settles as folded (`steer_end: folded`), the prompt slot frees, and a
/// later `send` runs — the deadlock of issue #47.
#[tokio::test(flavor = "multi_thread")]
async fn send_steer_folded_settles_at_turn_end() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let mut args = spawn_args(dir.path(), "gather");
    args["agent"] = json!("foldagent");
    let sid = spawn_id(&tools, args).await;

    // "gather" holds turn 1 open until the steer arrives; the agent folds
    // it in, ends the turn, and never answers the steer's session/prompt.
    let steered = call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "kick", "policy": "steer"}),
    )
    .await;
    assert_eq!(steered["state"], "steered", "{steered}");
    assert_eq!(steered["turn"], 1, "{steered}");

    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done", "{done}");
    assert!(
        done["reply"].as_str().unwrap().contains("steer:kick"),
        "{done}"
    );

    // The folded steer no longer blocks the slot: a `try` prompt runs.
    let sent = call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "hi", "policy": "try"}),
    )
    .await;
    assert_eq!(sent["state"], "running", "{sent}");
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done");

    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["turns"], 2, "{status}");
    let text = call_text(&tools, "transcript", json!({"session_id": sid.as_str()})).await;
    assert!(text.contains("=== [turn 1] STEER\nkick"), "{text}");
    assert!(text.contains("=== [turn 1] STEER END folded"), "{text}");
    assert!(text.contains("=== [turn 2] USER\nhi"), "{text}");
}

/// On a `folded` agent, a steer that landed behind its target turn's end
/// still runs as a turn of its own: its untracked `session/update` traffic
/// materializes a `current` owned by the settled-folded steer, and the
/// steer's own `session/prompt` response settles that turn.
#[tokio::test(flavor = "multi_thread")]
async fn send_steer_folded_late_becomes_own_turn() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());
    let mut args = spawn_args(dir.path(), "wait");
    args["agent"] = json!("foldagent");
    let sid = spawn_id(&tools, args).await;

    // "LATE" marks the steer that reaches the agent after turn 1 ended:
    // the fake ends turn 1, then runs the steer as its own turn whose
    // response resolves normally. The PERMISSION marker stalls that turn
    // on a permission round-trip, so acpsub settles turn 1 — and the
    // settled-folded steer owns the materialized turn — before the
    // steer's updates finish arriving.
    let steered = call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "LATE PERMISSION go", "policy": "steer"}),
    )
    .await;
    assert_eq!(steered["state"], "steered", "{steered}");

    // Turn 1 ends, the settled-folded steer's own turn materializes from
    // its updates, and the steer's response settles it — wait on the
    // notify signal until both turns are recorded. The timeout only turns
    // a hang into a failure.
    let sub = state.get(&sid).expect("live subagent");
    let two_turns = async {
        loop {
            let notified = sub.rt.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let inner = sub.rt.inner.lock().expect("inner poisoned");
                if inner.turns.len() == 2 && matches!(inner.status, Status::Done(_)) {
                    return;
                }
            }
            notified.await;
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(60), two_turns)
        .await
        .expect("subagent did not record two turns");

    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["state"], "done", "{status}");
    assert_eq!(status["turns"], 2, "{status}");

    let second = call_json(
        &tools,
        "result",
        json!({"session_id": sid.as_str(), "turn": 2}),
    )
    .await;
    assert!(
        second["reply"].as_str().unwrap().contains("perm:allow-1"),
        "{second}"
    );

    let text = call_text(&tools, "transcript", json!({"session_id": sid.as_str()})).await;
    assert!(
        text.contains("=== [turn 1] STEER\nLATE PERMISSION go"),
        "{text}"
    );
    assert!(text.contains("=== [turn 1] STEER END folded"), "{text}");
    assert!(text.contains("=== [turn 2] END end_turn"), "{text}");
    assert!(text.contains("=== [turn 2] STEER END end_turn"), "{text}");

    // The slot is free afterwards: a `try` prompt runs turn 3.
    let sent = call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "again", "policy": "try"}),
    )
    .await;
    assert_eq!(sent["state"], "running", "{sent}");
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done", "{done}");
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

/// `wait` returns `overrun` once the turn has run past `expect_secs` —
/// measured from the turn's start — reporting the turn's elapsed time and
/// its latest tool call. The turn keeps running underneath.
#[tokio::test(flavor = "multi_thread")]
async fn wait_overruns_past_expected_duration() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;

    let out = wait(&tools, &sid, 1).await;
    assert_eq!(out["state"], "overrun", "{out}");
    assert!(
        out["elapsed_secs"].as_f64().expect("elapsed_secs") >= 1.0,
        "{out}"
    );
    assert_eq!(out["latest_tool_call"]["id"], "tc-1", "{out}");

    // The subagent itself is still running — overrun is a report, not a
    // state transition.
    let status = call_json(&tools, "status", json!({"session_id": sid})).await;
    assert_eq!(status["state"], "running", "{status}");
    call_json(&tools, "cancel", json!({"session_id": sid})).await;
}

/// A turn that finishes within `expect_secs` returns its normal terminal
/// state.
#[tokio::test(flavor = "multi_thread")]
async fn wait_within_expected_duration_returns_done() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done", "{done}");
    assert_eq!(done["stop_reason"], "end_turn");
}

/// Re-issuing `wait` cannot extend the budget: both waits overrun at the
/// same wall-clock point, measured from the turn's recorded start.
#[tokio::test(flavor = "multi_thread")]
async fn wait_overrun_is_measured_from_turn_start() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;

    let first = wait(&tools, &sid, 2).await;
    assert_eq!(first["state"], "overrun", "{first}");

    // The budget ran out at turn_start + 2s: a second wait overruns at
    // once rather than getting a fresh 2 seconds.
    let reissued = std::time::Instant::now();
    let second = wait(&tools, &sid, 2).await;
    assert!(
        reissued.elapsed() < std::time::Duration::from_secs(1),
        "re-issued wait must not extend the budget: {second}"
    );
    assert_eq!(second["state"], "overrun", "{second}");
    assert!(
        second["elapsed_secs"].as_f64().expect("elapsed_secs")
            > first["elapsed_secs"].as_f64().expect("elapsed_secs"),
        "elapsed keeps growing from the turn's start: {second}"
    );
    call_json(&tools, "cancel", json!({"session_id": sid})).await;
}

/// `expect_secs` is required on both wait tools — there is no default.
#[tokio::test(flavor = "multi_thread")]
async fn wait_requires_expect_secs() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;

    let err = call_err(&tools, "wait", json!({"session_id": sid.as_str()})).await;
    assert!(err.contains("expect_secs"), "{err}");
    let err = call_err(&tools, "wait_any", json!({"session_ids": [sid.as_str()]})).await;
    assert!(err.contains("expect_secs"), "{err}");
    call_json(&tools, "cancel", json!({"session_id": sid})).await;
}

/// A `wait` spanning a queued-turn handoff sees a recorded start through
/// the gap: the chained turn's budget is measured from the moment its
/// slot was reserved — `running` never lacks a turn start.
#[tokio::test(flavor = "multi_thread")]
async fn wait_overruns_on_chained_turn() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());
    // "gather" holds turn 1 open until a steered prompt arrives.
    let sid = spawn_id(&tools, spawn_args(dir.path(), "gather")).await;

    // Park a prompt client-side, then finish turn 1 with a steer: the
    // queued prompt chains as turn 2 — a "wait" turn that never ends.
    call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "wait", "policy": "queued"}),
    )
    .await;
    call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "go", "policy": "steer"}),
    )
    .await;

    // `wait` binds to the turn in flight when it is called: hold it until
    // turn 1 has settled — `rt.notify` fires at the settle — so the call
    // awaits the chained turn 2 whether or not `begin_turn` has run yet.
    let sub = state.get(&sid).expect("live subagent");
    let turn_one_settled = async {
        loop {
            let notified = sub.rt.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let inner = sub.rt.inner.lock().expect("inner poisoned");
                if inner.turns.len() == 1 {
                    return;
                }
            }
            notified.await;
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(60), turn_one_settled)
        .await
        .expect("turn 1 did not settle");

    let out = wait(&tools, &sid, 2).await;
    assert_eq!(out["state"], "overrun", "{out}");
    assert!(
        out["elapsed_secs"].as_f64().expect("elapsed_secs") >= 2.0,
        "{out}"
    );
    call_json(&tools, "cancel", json!({"session_id": sid})).await;
}

/// A steered prompt that the agent runs as a turn of its own materializes
/// a `current` with a recorded start; `wait` budgets from it.
#[tokio::test(flavor = "multi_thread")]
async fn wait_overruns_on_materialized_steer_turn() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());
    // queueagent parks a concurrent prompt agent-side and runs it as its
    // own turn after "hold" ends.
    let mut args = spawn_args(dir.path(), "hold");
    args["agent"] = json!("queueagent");
    let sid = spawn_id(&tools, args).await;
    call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "wait", "policy": "steer"}),
    )
    .await;

    // `wait` binds to the turn in flight when it is called: hold it until
    // the steered prompt has materialized as turn 2 — `rt.notify` fires
    // on materialization — so the call awaits it rather than turn 1.
    let sub = state.get(&sid).expect("live subagent");
    let steer_turn_open = async {
        loop {
            let notified = sub.rt.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let inner = sub.rt.inner.lock().expect("inner poisoned");
                if inner.current.as_ref().is_some_and(|turn| turn.n == 2) {
                    return;
                }
            }
            notified.await;
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(60), steer_turn_open)
        .await
        .expect("steered prompt did not materialize as turn 2");

    let out = wait(&tools, &sid, 2).await;
    assert_eq!(out["state"], "overrun", "{out}");
    assert!(
        out["elapsed_secs"].as_f64().expect("elapsed_secs") >= 2.0,
        "{out}"
    );
    call_json(&tools, "cancel", json!({"session_id": sid})).await;
}

/// `wait_any` reports the first subagent whose turn overruns, with that
/// turn's elapsed time and latest tool call.
#[tokio::test(flavor = "multi_thread")]
async fn wait_any_reports_overrun() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;

    let out = call_json(
        &tools,
        "wait_any",
        json!({"session_ids": [sid.as_str()], "expect_secs": 1}),
    )
    .await;
    assert_eq!(out["state"], "overrun", "{out}");
    assert_eq!(out["session_id"], sid, "{out}");
    assert!(
        out["elapsed_secs"].as_f64().expect("elapsed_secs") >= 1.0,
        "{out}"
    );
    assert_eq!(out["latest_tool_call"]["id"], "tc-1", "{out}");
    call_json(&tools, "cancel", json!({"session_id": sid})).await;
}

/// `--max-wait` is the wait's own deadline: it returns `state: "running"`
/// with an activity digest measured from the tool calls' recorded
/// timestamps — the still-open call named as the longest, repeated titles
/// counted, and the mid-window edit's end measured.
#[tokio::test(flavor = "multi_thread")]
async fn wait_max_wait_returns_activity_digest() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "loops")).await;

    let out = call_json(
        &tools,
        "wait",
        json!({"session_id": sid.as_str(), "expect_secs": 60, "max_wait_secs": 2}),
    )
    .await;
    assert_eq!(out["state"], "running", "{out}");
    let digest = &out["digest"];
    let window = digest["window_secs"].as_f64().expect("window_secs");
    assert!((1.9..10.0).contains(&window), "{digest}");
    // tc-1, tc-edit, tc-sleep and the three "run check" calls all overlap
    // the window.
    assert_eq!(digest["tool_calls"], 6, "{digest}");
    // tc-sleep is the only call still open at the deadline: it owns the
    // window's wall time and the top of the longest list.
    let longest = digest["longest_tool_calls"]
        .as_array()
        .expect("longest_tool_calls");
    assert_eq!(longest[0]["id"], "tc-sleep", "{digest}");
    assert_eq!(longest[0]["title"], "run sleep 300", "{digest}");
    // Its recorded start can land inside the window when update traffic
    // lags under load; it still owns most of the window's wall time.
    assert!(
        longest[0]["duration_secs"].as_f64().expect("duration_secs") >= window / 2.0,
        "{digest}"
    );
    assert_eq!(
        digest["repeated_titles"],
        json!([{"title": "run check", "count": 3}]),
        "{digest}"
    );
    // The edit finished ~0.3 s in, inside the window.
    let since_edit = digest["secs_since_last_edit"]
        .as_f64()
        .expect("secs_since_last_edit");
    assert!(since_edit > 0.1 && since_edit < window, "{digest}");
    assert_eq!(digest["latest_tool_call"]["id"], "tc-edit", "{digest}");
    // Same lag margin: the open call plus the two ~0.3 s calls put most of
    // the window's wall time inside tool calls.
    assert!(
        digest["tool_call_share"].as_f64().expect("tool_call_share") >= 0.5,
        "{digest}"
    );

    // An overrun carries the same digest — over that wait's own window,
    // where only the still-running call remains in scope.
    let over = call_json(
        &tools,
        "wait",
        json!({"session_id": sid.as_str(), "expect_secs": 1}),
    )
    .await;
    assert_eq!(over["state"], "overrun", "{over}");
    let digest = &over["digest"];
    assert_eq!(digest["tool_calls"], 1, "{digest}");
    assert_eq!(
        digest["longest_tool_calls"][0]["id"], "tc-sleep",
        "{digest}"
    );
    assert_eq!(digest["repeated_titles"], json!([]), "{digest}");

    // `wait_any` takes the same bound and reports the same digest.
    let any = call_json(
        &tools,
        "wait_any",
        json!({"session_ids": [sid.as_str()], "expect_secs": 60, "max_wait_secs": 1}),
    )
    .await;
    assert_eq!(any["state"], "running", "{any}");
    assert_eq!(any["session_id"], sid, "{any}");
    assert_eq!(any["digest"]["tool_calls"], 1, "{any}");
    assert_eq!(
        any["digest"]["longest_tool_calls"][0]["id"], "tc-sleep",
        "{any}"
    );
    call_json(&tools, "cancel", json!({"session_id": sid})).await;
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
        json!({"agent": "wideopen", "cwd": dir.path(), "prompt": "READ /etc/hosts",
               "model": "b", "mode": "bypass"}),
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
        json!({"session_id": sid.as_str(), "prompt": "again", "model": "b", "mode": "bypass"}),
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

    let adopted = call_json(
        &tools,
        "adopt",
        json!({"session_id": sid.as_str(), "model": "b", "mode": "bypass"}),
    )
    .await;
    assert_eq!(adopted["state"], "idle", "{adopted}");

    call_json(
        &tools,
        "send",
        json!({"session_id": sid.as_str(), "prompt": "again", "policy": "try"}),
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

    let err = call_err(
        &tools,
        "adopt",
        json!({"session_id": sid.as_str(), "model": "b", "mode": "bypass"}),
    )
    .await;
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
               "cwd": dir.path(), "prompt": "hi", "model": "b", "mode": "bypass"}),
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
        json!({"session_id": "ghost", "agent": "fake", "model": "b", "mode": "bypass"}),
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
        test_state_with_agents(dir.path(), &BTreeMap::from([("fake".to_string(), fake)]));

    let adopted = call_json(
        &tools,
        "adopt",
        json!({"session_id": "ext-9", "prompt": "hi", "model": "b", "mode": "bypass"}),
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

    let err = call_err(
        &tools,
        "adopt",
        json!({"session_id": sid.as_str(), "model": "b", "mode": "bypass"}),
    )
    .await;
    assert!(err.contains("already live"), "{err}");

    call_json(&tools, "close", json!({"session_id": sid.as_str()})).await;
    let tools = std::sync::Arc::new(tools);
    let first = {
        let tools = tools.clone();
        let sid = sid.clone();
        tokio::spawn(async move {
            call(
                &tools,
                "adopt",
                json!({"session_id": sid.as_str(), "model": "b", "mode": "bypass"}),
            )
            .await
        })
    };
    let second = {
        let tools = tools.clone();
        tokio::spawn(async move {
            call(
                &tools,
                "adopt",
                json!({"session_id": sid.as_str(), "model": "b", "mode": "bypass"}),
            )
            .await
        })
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
        json!({"session_id": sid.as_str(), "agent": "noload", "model": "b", "mode": "bypass"}),
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
        json!({"session_id": "", "agent": "fake", "cwd": dir.path(),
               "model": "b", "mode": "bypass"}),
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
    let err = call_err(
        &tools,
        "spawn",
        json!({"cwd": dir.path(), "prompt": "hi", "model": "b", "mode": "bypass"}),
    )
    .await;
    assert!(err.contains("no agent specified"), "{err}");

    // A configured default is used.
    let (_state, tools) = test_state_full(
        dir.path(),
        Some("fake"),
        &BTreeMap::from([
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
    let sid = spawn_id(
        &tools,
        json!({"cwd": dir.path(), "prompt": "hi", "model": "b", "mode": "bypass"}),
    )
    .await;
    wait(&tools, &sid, 60).await;

    // A single configured agent is the implicit default.
    let dir2 = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state_with_agents(
        dir2.path(),
        &BTreeMap::from([("fake".to_string(), fake_agent_config())]),
    );
    let sid = spawn_id(
        &tools,
        json!({"cwd": dir2.path(), "prompt": "hi", "model": "b", "mode": "bypass"}),
    )
    .await;
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
        json!({"agent": "nobody", "cwd": dir.path(), "prompt": "hi",
               "model": "b", "mode": "bypass"}),
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
        json!({"session_ids": [sid1.as_str(), sid2.as_str()], "expect_secs": 60}),
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
/// subagent settles `exited`, and the registry entry survives so the session
/// stays adoptable after `close`.
#[tokio::test(flavor = "multi_thread")]
async fn agent_death_mid_turn_marks_failed() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "die")).await;
    let done = wait(&tools, &sid, 60).await;
    // The turn error and the process exit race; `exited` always wins.
    assert!(
        matches!(done["state"].as_str(), Some("failed" | "exited")),
        "{done}"
    );
    status_becomes(&state, &sid, "exited").await;

    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["state"], "exited");
    assert_eq!(status["live"], true);
    assert!(
        status["error"].as_str().is_some_and(|e| !e.is_empty()),
        "{status}"
    );

    // The registry entry outlives the process: after `close` the session is
    // adoptable again via session/load.
    call_json(&tools, "close", json!({"session_id": sid.as_str()})).await;
    let adopted = call_json(
        &tools,
        "adopt",
        json!({"session_id": sid.as_str(), "prompt": "hi", "model": "b", "mode": "bypass"}),
    )
    .await;
    assert_eq!(adopted["session_id"], sid);
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done");
}

/// An `exited` subagent is no longer live: `adopt` reaps the dead runtime
/// and loads the session on a fresh process — no `close` first (issue #51).
#[tokio::test(flavor = "multi_thread")]
async fn adopt_reaps_exited_subagent() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());
    let sid = spawn_id(&tools, spawn_args(dir.path(), "die")).await;
    wait(&tools, &sid, 60).await;
    status_becomes(&state, &sid, "exited").await;

    let adopted = call_json(
        &tools,
        "adopt",
        json!({"session_id": sid.as_str(), "prompt": "hi", "model": "b", "mode": "bypass"}),
    )
    .await;
    assert_eq!(adopted["session_id"], sid, "{adopted}");
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done", "{done}");
    assert_eq!(done["reply"], "Hello abworld", "{done}");
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
        json!({"session_id": sid.as_str(), "prompt": "again", "policy": "try"}),
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
               "prompt": format!("WRITE {}", outside_file.display()),
               "model": "b", "mode": "bypass"}),
    )
    .await;
    let done = wait(&tools, &sid, 60).await;
    assert!(done["reply"].as_str().unwrap().contains("fsw:ok"), "{done}");
    assert_eq!(
        std::fs::read_to_string(&outside_file).unwrap(),
        "written-by-fake-agent"
    );
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
    let (_state, tools) = test_state_with_agents(dir.path(), &agents);

    let err = call_err(
        &tools,
        "spawn",
        json!({"agent": "missing", "cwd": dir.path(), "prompt": "hi",
               "model": "b", "mode": "bypass"}),
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

/// `model` and `mode` are required `spawn`/`adopt` arguments: a missing one
/// is a tool error naming the field.
#[tokio::test(flavor = "multi_thread")]
async fn spawn_and_adopt_require_model_and_mode() {
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    for (tool, base) in [
        (
            "spawn",
            json!({"agent": "fake", "cwd": dir.path(), "prompt": "hi"}),
        ),
        (
            "adopt",
            json!({"agent": "fake", "session_id": "s-1", "cwd": dir.path()}),
        ),
    ] {
        let mut missing_model = base.clone();
        missing_model["mode"] = json!("bypass");
        let err = call_err(&tools, tool, missing_model).await;
        assert!(err.contains("model"), "{tool}: {err}");

        let mut missing_mode = base.clone();
        missing_mode["model"] = json!("b");
        let err = call_err(&tools, tool, missing_mode).await;
        assert!(err.contains("mode"), "{tool}: {err}");
    }
}

/// `spawn` validates `mode`, `model`, and `config` values against what the
/// agent advertised in `session/new`: an unknown value fails naming itself
/// and the valid choices, and the spawned process is closed — nothing stays
/// live or registered.
#[tokio::test(flavor = "multi_thread")]
async fn spawn_rejects_unadvertised_mode_model_config() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());

    let mut args = spawn_args(dir.path(), "hi");
    args["mode"] = json!("dangerous");
    let err = call_err(&tools, "spawn", args).await;
    assert!(err.contains("dangerous"), "{err}");
    assert!(err.contains("default"), "{err}");
    assert!(err.contains("bypass"), "{err}");

    let mut args = spawn_args(dir.path(), "hi");
    args["model"] = json!("hal-9000");
    let err = call_err(&tools, "spawn", args).await;
    assert!(err.contains("hal-9000"), "{err}");
    assert!(err.contains("valid: a, b"), "{err}");

    // A `config` key the agent does not advertise is rejected as well.
    let mut args = spawn_args(dir.path(), "hi");
    args["config"] = json!({"thinking": "high"});
    let err = call_err(&tools, "spawn", args).await;
    assert!(err.contains("thinking"), "{err}");
    assert!(err.contains("known: model"), "{err}");

    // Nothing survived: no live subagent and no registry entry.
    assert!(
        state.live.lock().expect("live").is_empty(),
        "no live subagent"
    );
    assert!(
        state.reserved.lock().expect("reserved").is_empty(),
        "no reserved session id"
    );
    assert!(
        !dir.path().join("registry.json").exists(),
        "nothing registered"
    );

    // A valid pair spawns and reports the agent's current values.
    let spawned = call_json(&tools, "spawn", spawn_args(dir.path(), "hi")).await;
    assert_eq!(spawned["model"], "b", "{spawned}");
    assert_eq!(spawned["mode"], "bypass", "{spawned}");
}

/// The reported `mode`/`model` are the agent-confirmed values, not the
/// request echoed back: `FAKE_MODE_IGNORED` makes `session/set_mode` a
/// silent no-op, so the `mode` config option keeps reporting the mode the
/// session actually runs.
#[tokio::test(flavor = "multi_thread")]
async fn spawn_reports_agent_confirmed_mode() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let agents = BTreeMap::from([(
        "stuck".to_string(),
        AgentConfig {
            env: BTreeMap::from([("FAKE_MODE_IGNORED".to_string(), "1".to_string())]),
            ..fake_agent_config()
        },
    )]);
    let (_state, tools) = test_state_with_agents(dir.path(), &agents);

    let mut args = spawn_args(dir.path(), "hi");
    args["agent"] = json!("stuck");
    let spawned = call_json(&tools, "spawn", args).await;
    assert_eq!(spawned["mode"], "default", "{spawned}");
    assert_eq!(spawned["model"], "b", "{spawned}");

    // The `agents` view reports the same agent-confirmed mode.
    let listed = call_json(&tools, "agents", json!({})).await;
    let stuck = listed["agents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["name"] == "stuck")
        .expect("stuck agent");
    assert_eq!(stuck["modes"]["current"], "default", "{stuck}");
}

/// An agent that advertises no mode list, or a `model` option with no
/// values, leaves the parameter uncheckable — the call fails instead of
/// proceeding unverified.
#[tokio::test(flavor = "multi_thread")]
async fn spawn_rejects_unverifiable_mode_and_model() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let fake = fake_agent_config();
    let agents = BTreeMap::from([
        (
            "nomodes".to_string(),
            AgentConfig {
                env: BTreeMap::from([("FAKE_NO_MODES".to_string(), "1".to_string())]),
                ..fake.clone()
            },
        ),
        (
            "noopts".to_string(),
            AgentConfig {
                env: BTreeMap::from([("FAKE_NO_OPTIONS".to_string(), "1".to_string())]),
                ..fake
            },
        ),
    ]);
    let (state, tools) = test_state_with_agents(dir.path(), &agents);

    let mut args = spawn_args(dir.path(), "hi");
    args["agent"] = json!("nomodes");
    let err = call_err(&tools, "spawn", args).await;
    assert!(err.contains("advertises no session modes"), "{err}");

    let mut args = spawn_args(dir.path(), "hi");
    args["agent"] = json!("noopts");
    let err = call_err(&tools, "spawn", args).await;
    assert!(err.contains("cannot check 'b'"), "{err}");

    assert!(state.live.lock().expect("live").is_empty());
}

/// `adopt` applies the same check to `session/load` results: a bad mode is
/// rejected and the reserved session id is freed for a valid retry.
#[tokio::test(flavor = "multi_thread")]
async fn adopt_rejects_unadvertised_mode() {
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
        json!({"session_id": sid.as_str(), "model": "b", "mode": "dangerous"}),
    )
    .await;
    assert!(err.contains("dangerous"), "{err}");
    assert!(err.contains("bypass"), "{err}");

    // The failed adopt released the session id: a valid pair adopts and
    // reports the agent's current mode and model.
    let adopted = call_json(
        &tools,
        "adopt",
        json!({"session_id": sid.as_str(), "model": "b", "mode": "bypass"}),
    )
    .await;
    assert_eq!(adopted["session_id"], sid);
    assert_eq!(adopted["mode"], "bypass", "{adopted}");
    assert_eq!(adopted["model"], "b", "{adopted}");
}

/// An agent that advertises neither session modes nor a `model` config
/// option (e.g. `grok agent stdio`) takes neither argument: `spawn`/`adopt`
/// with them omitted succeed and report `null` for both, while passing one
/// is an error naming the missing advertisement.
#[tokio::test(flavor = "multi_thread")]
async fn spawn_adopt_agent_without_modes_or_model() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let fake = fake_agent_config();
    let agents = BTreeMap::from([
        (
            "nomodelmode".to_string(),
            AgentConfig {
                env: BTreeMap::from([
                    ("FAKE_NO_MODES".to_string(), "1".to_string()),
                    ("FAKE_NO_MODEL".to_string(), "1".to_string()),
                ]),
                ..fake.clone()
            },
        ),
        ("fake".to_string(), fake),
    ]);
    let (_state, tools) = test_state_with_agents(dir.path(), &agents);
    let spawn_args = |extra: serde_json::Value| {
        let mut args = json!({
            "agent": "nomodelmode",
            "cwd": dir.path(),
            "prompt": "hi",
        });
        for (key, value) in extra.as_object().expect("object") {
            args[key] = value.clone();
        }
        args
    };

    // Omitted mode/model spawn and report `null`.
    let spawned = call_json(&tools, "spawn", spawn_args(json!({}))).await;
    assert!(spawned["mode"].is_null(), "{spawned}");
    assert!(spawned["model"].is_null(), "{spawned}");
    let sid = spawned["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();
    let status = call_json(&tools, "status", json!({"session_id": sid})).await;
    assert!(status["mode"].is_null(), "{status}");
    assert!(status["model"].is_null(), "{status}");
    wait(&tools, &sid, 60).await;
    call_json(&tools, "close", json!({"session_id": sid})).await;

    // Adopt likewise takes neither; passing one is an error that names the
    // missing advertisement, and the session stays adoptable.
    let err = call_err(
        &tools,
        "adopt",
        json!({"session_id": sid.as_str(), "mode": "bypass"}),
    )
    .await;
    assert!(err.contains("advertises no session modes"), "{err}");
    let adopted = call_json(&tools, "adopt", json!({"session_id": sid.as_str()})).await;
    assert!(adopted["mode"].is_null(), "{adopted}");
    assert!(adopted["model"].is_null(), "{adopted}");

    // spawn errors the same way on each argument.
    let err = call_err(&tools, "spawn", spawn_args(json!({"mode": "bypass"}))).await;
    assert!(err.contains("advertises no session modes"), "{err}");
    let err = call_err(&tools, "spawn", spawn_args(json!({"model": "b"}))).await;
    assert!(err.contains("advertises no 'model' config option"), "{err}");

    // The reverse still holds: omitting `mode` or `model` for an agent
    // that advertises them is an error naming the required argument.
    let err = call_err(
        &tools,
        "spawn",
        json!({"agent": "fake", "cwd": dir.path(), "prompt": "hi", "model": "b"}),
    )
    .await;
    assert!(err.contains("'mode' is required"), "{err}");
    let err = call_err(
        &tools,
        "spawn",
        json!({"agent": "fake", "cwd": dir.path(), "prompt": "hi", "mode": "bypass"}),
    )
    .await;
    assert!(err.contains("'model' is required"), "{err}");
}

/// The config file is re-read per call: an agent removed from it is gone
/// from `agents` and unspawnable without a restart.
#[tokio::test(flavor = "multi_thread")]
async fn removed_agent_disappears_without_restart() {
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let listed = call_json(&tools, "agents", json!({})).await;
    assert!(
        listed["agents"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["name"] == "fake"),
        "{listed}"
    );

    write_config(
        dir.path(),
        None,
        &BTreeMap::from([("other".to_string(), fake_agent_config())]),
    );
    let listed = call_json(&tools, "agents", json!({})).await;
    let names: Vec<&str> = listed["agents"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|a| a["name"].as_str())
        .collect();
    assert_eq!(names, ["other"], "{listed}");
    let err = call_err(&tools, "spawn", spawn_args(dir.path(), "hi")).await;
    assert!(err.contains("unknown agent 'fake'"), "{err}");
}

/// A config file that fails to parse is a tool error — never a fallback to
/// a previously loaded config.
#[tokio::test(flavor = "multi_thread")]
async fn broken_config_is_a_tool_error() {
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    std::fs::write(dir.path().join("config.toml"), "not toml [[[").unwrap();
    let err = call_err(&tools, "spawn", spawn_args(dir.path(), "hi")).await;
    assert!(err.contains("cannot parse config"), "{err}");
    let err = call_err(&tools, "agents", json!({})).await;
    assert!(err.contains("cannot parse config"), "{err}");
}

// ---------------------------------------------------------------------
// Rate-limited prompts park, then resume on their own (#56)
// ---------------------------------------------------------------------

/// A turn the provider rate-limits parks in `rate_limited` carrying the
/// structured reset — never `failed` — and the session resumes on its
/// own at `resume_at`, in the same ACP session, on the scheduler's
/// continuation prompt.
#[tokio::test(flavor = "multi_thread")]
async fn rate_limit_parks_turn_and_resumes() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());
    let sid = spawn_id(
        &tools,
        json!({"agent": "ratelimit", "cwd": dir.path(), "prompt": "ratelimit", "model": "b", "mode": "bypass"}),
    )
    .await;

    status_becomes(&state, sid.as_str(), "rate_limited").await;
    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["state"], "rate_limited", "{status}");
    let resume_at = status["resume_at"].as_str().expect("resume_at");
    assert!(resume_at.parse::<jiff::Timestamp>().is_ok(), "{resume_at}");
    assert!(
        status["reason"].as_str().unwrap().contains("rate limit"),
        "{status}"
    );
    // The registry carries the schedule — a restart picks it up from
    // disk alone (the write races this check, so poll for it).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let registry = loop {
        let text = std::fs::read_to_string(dir.path().join("registry.json")).unwrap();
        if text.contains("\"rate_limited\"") {
            break text;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "parked schedule never persisted: {text}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    };
    assert!(registry.contains(resume_at), "{registry}");

    // No coordinator call moves it: the scheduler fires the
    // continuation at resume_at, in this session.
    let text = transcript_eventually(&tools, sid.as_str(), "rate limit has lifted").await;
    assert!(text.contains("END rate_limited"), "{text}");
    assert!(text.contains("END end_turn"), "{text}");
}

/// Poll a session's rendered transcript until it contains `needle`,
/// returning the text. For states nothing notifies — a queued prompt
/// chained after a resume leaves the status `done` twice.
async fn transcript_eventually(tools: &Tools, session_id: &str, needle: &str) -> String {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let text = call_text(tools, "transcript", json!({"session_id": session_id})).await;
        if text.contains(needle) {
            return text;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "transcript never showed {needle:?}: {text}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// Every prompt the scope's limit would hit parks instead: a `send`
/// into a parked session and a different session's new prompt both join
/// the parked queue — and both run after the continuation, in their own
/// sessions.
#[tokio::test(flavor = "multi_thread")]
async fn rate_limit_parks_send_and_spawn() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());
    let sid_a = spawn_id(
        &tools,
        json!({"agent": "ratelimit", "cwd": dir.path(), "prompt": "ratelimit", "model": "b", "mode": "bypass"}),
    )
    .await;
    status_becomes(&state, sid_a.as_str(), "rate_limited").await;

    let sent = call_json(
        &tools,
        "send",
        json!({"session_id": sid_a.as_str(), "prompt": "queued one", "policy": "queued"}),
    )
    .await;
    assert_eq!(sent["state"], "rate_limited", "{sent}");
    assert_eq!(sent["position"], 1, "{sent}");
    assert!(
        sent["resume_at"]
            .as_str()
            .unwrap()
            .parse::<jiff::Timestamp>()
            .is_ok(),
        "{sent}"
    );

    // The quota binds the login, not the session: another session's
    // prompt parks without ever reaching its agent.
    let spawned = call_json(
        &tools,
        "spawn",
        json!({"agent": "ratelimit", "cwd": dir.path(), "prompt": "hi", "model": "b", "mode": "bypass"}),
    )
    .await;
    assert_eq!(spawned["state"], "rate_limited", "{spawned}");
    assert!(spawned["resume_at"].is_string(), "{spawned}");
    let sid_b = spawned["session_id"].as_str().unwrap().to_string();

    // Both sessions resume on their own and run the continuation
    // first, then their parked prompts — under the session ids they
    // started with.
    let text = transcript_eventually(&tools, sid_a.as_str(), "USER (queued)\nqueued one").await;
    let continuation = text.find("rate limit has lifted").expect("continuation");
    let queued = text
        .find("USER (queued)\nqueued one")
        .expect("queued prompt");
    assert!(continuation < queued, "continuation ran late: {text}");
    assert!(text.contains("END end_turn"), "{text}");

    let text = transcript_eventually(&tools, sid_b.as_str(), "USER (queued)\nhi").await;
    let continuation = text.find("rate limit has lifted").expect("continuation");
    let queued = text.find("USER (queued)\nhi").expect("queued prompt");
    assert!(continuation < queued, "continuation ran late: {text}");
    assert!(text.contains("END end_turn"), "{text}");
    assert_eq!(state.get(sid_b.as_str()).expect("live").agent, "ratelimit");
}

/// A `wait` bound to a turn that ends rate-limited stays bound through
/// the pause: it does not return early, and the parked seconds do not
/// count against `expect_secs` — the resume's continuation ends it.
#[tokio::test(flavor = "multi_thread")]
async fn wait_spans_rate_limit_pause() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(
        &tools,
        json!({"agent": "ratelimit", "cwd": dir.path(), "prompt": "ratelimit", "model": "b", "mode": "bypass"}),
    )
    .await;

    // Bind while the turn is still in flight — the fake takes ~0.35 s
    // to answer with the rate-limit error.
    let started = std::time::Instant::now();
    let out = call_json(
        &tools,
        "wait",
        json!({"session_id": sid.as_str(), "expect_secs": 2, "max_wait_secs": 30}),
    )
    .await;
    let wall = started.elapsed().as_secs_f64();
    assert_eq!(out["state"], "done", "{out}");
    assert_eq!(out["stop_reason"], "end_turn", "{out}");
    assert!(out["turn"].as_u64().unwrap() >= 2, "{out}");
    // The window lasts ~4 s: returning before it lifted would be early.
    assert!(wall >= 3.0, "wait returned early: {wall}s");
    // ~0.5 s of turn time ran against expect_secs=2 — had the ~4 s
    // pause counted, the wait would have overrun.
    assert!(out.get("overrun").is_none(), "{out}");
}

/// A parked session whose process is gone — here the whole state is
/// dropped and rebuilt over the same registry, the daemon-restart case —
/// resumes at `resume_at` through `session/load`, in the same session.
#[tokio::test(flavor = "multi_thread")]
async fn rate_limit_resumes_after_restart() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());
    let sid = spawn_id(
        &tools,
        json!({"agent": "ratelimit", "cwd": dir.path(), "prompt": "ratelimit", "model": "b", "mode": "bypass"}),
    )
    .await;
    status_becomes(&state, sid.as_str(), "rate_limited").await;
    call_json(&tools, "close", json!({"session_id": sid.as_str()})).await;

    // Down and back up: the registry file is all that survives.
    drop(tools);
    drop(state);
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    let (state, tools) = test_state(dir.path());

    // The schedule survived: the session is back — resumed through
    // session/load — once resume_at passes.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while state.get(sid.as_str()).is_err() {
        assert!(
            std::time::Instant::now() < deadline,
            "parked session never resumed"
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let text = transcript_eventually(&tools, sid.as_str(), "rate limit has lifted").await;
    // session/load replays history — proof the resume re-bound the same
    // ACP session rather than spawning a new one.
    assert!(text.contains("loaded user message"), "{text}");
    status_becomes(&state, sid.as_str(), "done").await;
}

/// A rate-limit report whose reset is unreadable — no structured
/// retry-after, no `(at HH:MM UTC)` clause — fails the turn with the
/// original error rather than guessing a delay.
#[tokio::test(flavor = "multi_thread")]
async fn rate_limit_without_reset_fails() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let sid = spawn_id(
        &tools,
        json!({"agent": "vaguelimit", "cwd": dir.path(), "prompt": "ratelimit", "model": "b", "mode": "bypass"}),
    )
    .await;

    let out = wait(&tools, sid.as_str(), 60).await;
    assert_eq!(out["state"], "failed", "{out}");
    assert!(
        out["error"].as_str().unwrap().contains("rate limit"),
        "{out}"
    );
    // And no parking happened: the session reports a plain failure,
    // not a `rate_limited` schedule.
    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert!(status.get("resume_at").is_none(), "{status}");
}

/// A session the scheduler resumes through `session/load` — the
/// process was gone — keeps the coordinator pid that owned it, so the
/// reaper still sees an owner.
#[tokio::test(flavor = "multi_thread")]
async fn rate_limit_resume_keeps_owner() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = test_state(dir.path());
    let sid = spawn_id(
        &tools,
        json!({"agent": "ratelimit", "cwd": dir.path(), "prompt": "ratelimit", "model": "b", "mode": "bypass", "owner": 424_242}),
    )
    .await;
    status_becomes(&state, sid.as_str(), "rate_limited").await;
    call_json(&tools, "close", json!({"session_id": sid.as_str()})).await;

    drop(tools);
    drop(state);
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    let (state, _tools) = test_state(dir.path());

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let owner = loop {
        if let Ok(sub) = state.get(sid.as_str()) {
            break sub.rt.owner.load(std::sync::atomic::Ordering::Relaxed);
        }
        assert!(
            std::time::Instant::now() < deadline,
            "parked session never resumed"
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    };
    assert_eq!(owner, 424_242);
    status_becomes(&state, sid.as_str(), "done").await;
}

/// A draining daemon refuses new work: `spawn`, `send` and `adopt` get
/// the `restarting` error the client retries on, read-only calls keep
/// working, and a second drain is refused outright.
#[tokio::test(flavor = "multi_thread")]
async fn drain_refuses_new_work() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, mut tools) = test_state(dir.path());
    acpsub::tools::register_daemon_tools(&mut tools, state.clone()).expect("daemon tools");

    let draining = call_json(&tools, "daemon/drain", json!({})).await;
    assert_eq!(draining["draining"], true, "{draining}");

    for (tool, args) in [
        ("spawn", spawn_args(dir.path(), "hi")),
        (
            "send",
            json!({"session_id": "s", "prompt": "hi", "policy": "try"}),
        ),
        ("adopt", json!({"session_id": "s"})),
    ] {
        let err = call_err(&tools, tool, args).await;
        assert!(err.contains("restarting"), "{tool}: {err}");
    }

    call_json(&tools, "list", json!({})).await;

    let again = call_err(&tools, "daemon/drain", json!({})).await;
    assert!(again.contains("already in progress"), "{again}");
}
