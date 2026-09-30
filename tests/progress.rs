//! Progress while `wait`/`wait_any` block: an in-process sink captures the
//! reports at the short interval the test harness injects through
//! [`acpsub::build_tools_with_progress_interval`], and asserts they keep
//! coming for as long as the turn runs. The stdio observation is in
//! `mcp_stdio.rs`.

#[expect(
    dead_code,
    reason = "helpers shared across test binaries; this one uses a subset"
)]
mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use aither_core::llm::tool::{Progress, ProgressSink, ToolContext, ToolResult, Tools};
use common::*;
use serde_json::{Value, json};
use tokio::sync::mpsc;

/// The interval `tests/common` builds the tools with — small enough to
/// observe several reports inside a test-length wait.
const TEST_INTERVAL_MS: u64 = 50;
/// The `expect_secs` the tests wait against: beyond the no-listener
/// ceiling, so only a progress-reporting call may proceed — and only the
/// test's own `cancel` ends it.
const EXPECT_SECS: u64 = 3600;

/// Every report goes to the test's channel, stamped on arrival.
struct Collect(mpsc::UnboundedSender<(Progress, Instant)>);

impl ProgressSink for Collect {
    async fn report(&mut self, progress: Progress) {
        let _ = self.0.send((progress, Instant::now()));
    }
}

/// Run a blocked `wait`/`wait_any` call against `session_id`, collecting
/// the reports its context delivers; returns after three arrive and
/// `cancel` ends the turn and the call.
async fn reports_while_blocked(
    tools: &Arc<Tools>,
    name: &str,
    args: Value,
    session_id: &str,
) -> Vec<(Progress, Instant)> {
    let (tx, mut reports) = mpsc::unbounded_channel();
    let waiting = {
        let tools = Arc::clone(tools);
        let name = name.to_string();
        tokio::spawn(async move {
            call_with(&tools, &name, args, ToolContext::with_progress(Collect(tx))).await
        })
    };
    let mut seen = Vec::new();
    while seen.len() < 3 {
        seen.push(
            tokio::time::timeout(Duration::from_secs(5), reports.recv())
                .await
                .expect("a progress report within 5 s")
                .expect("sink alive while the call waits"),
        );
    }
    call(tools, "cancel", json!({"session_id": session_id}))
        .await
        .expect("cancel");
    let result: ToolResult = waiting.await.expect("wait task").expect("wait call");
    assert!(!result.is_error(), "call returned an error result");
    seen
}

/// The reports a blocked wait must produce: strictly increasing elapsed
/// seconds against `total = expect_secs`, at the interval, naming the
/// session and its latest tool call.
#[expect(
    clippy::cast_precision_loss,
    reason = "the test's small second counts are exactly representable"
)]
fn assert_reports(seen: &[(Progress, Instant)], session_id: &str) {
    let progress: Vec<f64> = seen.iter().map(|(p, _)| p.progress()).collect();
    assert!(
        progress.windows(2).all(|w| w[1] > w[0]),
        "progress must increase: {progress:?}"
    );
    for (p, _) in seen {
        assert_eq!(p.total(), Some(EXPECT_SECS as f64), "total is expect_secs");
        let message = p.message().expect("progress message");
        assert!(message.contains(session_id), "{message}");
    }
    // The first report can precede the agent's tool_call update; later ones
    // carry its title.
    assert!(
        seen.iter()
            .any(|(p, _)| p.message().is_some_and(|m| m.contains("fake-tool"))),
        "no message names the latest tool call: {seen:?}"
    );
    for gap in seen.windows(2).map(|w| w[1].1.duration_since(w[0].1)) {
        assert!(
            gap >= Duration::from_millis(TEST_INTERVAL_MS / 2) && gap < Duration::from_secs(2),
            "report gap {gap:?} vs interval {TEST_INTERVAL_MS} ms",
        );
    }
}

/// Without a progress token a wait whose expectation passes the
/// no-listener ceiling fails up front: it could never return before the
/// host abandons the silent call.
#[tokio::test]
async fn silent_long_expect_is_rejected() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let session_id = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;

    for (name, args) in [
        (
            "wait",
            json!({"session_id": session_id, "expect_secs": EXPECT_SECS}),
        ),
        (
            "wait_any",
            json!({"session_ids": [session_id], "expect_secs": EXPECT_SECS}),
        ),
    ] {
        let error = call_err(&tools, name, args).await;
        assert!(error.contains("3600"), "{name}: {error}");
        assert!(error.contains("600"), "{name}: {error}");
        assert!(error.contains("progress token"), "{name}: {error}");
    }
}

/// A short expectation still waits normally with no progress token: it
/// returns — here `overrun` — before any host idle limit could bite.
#[tokio::test]
async fn silent_short_expect_proceeds() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let session_id = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;

    let waited = wait(&tools, &session_id, 2).await;
    assert_eq!(waited["state"], "overrun", "{waited}");
}

/// `wait` reports the turn's elapsed time every interval while blocked.
#[tokio::test]
async fn wait_reports_progress_at_the_interval() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let tools = Arc::new(tools);
    let session_id = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;
    let seen = reports_while_blocked(
        &tools,
        "wait",
        json!({"session_id": session_id, "expect_secs": EXPECT_SECS}),
        &session_id,
    )
    .await;
    assert_reports(&seen, &session_id);
}

/// `wait_any` reports the longest-running watched turn's elapsed time
/// every interval while blocked.
#[tokio::test]
async fn wait_any_reports_progress_at_the_interval() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_state, tools) = test_state(dir.path());
    let tools = Arc::new(tools);
    let session_id = spawn_id(&tools, spawn_args(dir.path(), "wait")).await;
    let seen = reports_while_blocked(
        &tools,
        "wait_any",
        json!({"session_ids": [session_id], "expect_secs": EXPECT_SECS}),
        &session_id,
    )
    .await;
    assert_reports(&seen, &session_id);
}
