//! Integration tests: the tool set exercised in-process against
//! `tests/fake_agent.py`.

mod common;

use std::collections::BTreeMap;

use acpsub::Status;
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
    let (_state, tools) = test_state(dir.path());
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

    // The queued prompt runs as turn 2 with no done interlude; `wait`
    // returns when it ends.
    let done = wait(&tools, &sid, 60).await;
    assert_eq!(done["state"], "done", "{done}");
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

/// Queued prompts drop when the turn fails; the steer that kills the agent
/// surfaces its error to the `send` caller.
#[tokio::test(flavor = "multi_thread")]
async fn send_queued_dropped_on_failure() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
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
    assert_eq!(done["state"], "failed", "{done}");
    let status = call_json(&tools, "status", json!({"session_id": sid.as_str()})).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["queued"], json!([]), "{status}");
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
    let (_state, tools) = test_state(dir.path());
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
    let (_state, tools) = test_state(dir.path());
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
        json!({"session_id": sid.as_str(), "prompt": "hi", "model": "b", "mode": "bypass"}),
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
        ("spawn", json!({"cwd": dir.path(), "prompt": "hi"})),
        ("adopt", json!({"session_id": "s-1", "cwd": dir.path()})),
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
    assert!(err.contains("cannot check mode 'bypass'"), "{err}");

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
