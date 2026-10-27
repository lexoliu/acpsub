//! Transcript retention: a prune pass removes expired non-live sessions —
//! transcript file and registry entry — and expired transcript files with
//! no registry entry, and leaves live and recent sessions alone.

#[expect(
    dead_code,
    reason = "helpers shared across test binaries; this one uses a subset"
)]
mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use acpsub::registry::RegistryEntry;
use acpsub::{AppState, Config};
use aither_core::llm::tool::Tools;
use common::*;
use serde_json::json;

/// Build an app state rooted at `dir` with the `fake` agent and
/// `[defaults] transcript_retention` set to `retention` (a friendly
/// duration like "1h").
fn prune_state(dir: &Path, retention: &str) -> (Arc<AppState>, Tools) {
    let mut agents = BTreeMap::new();
    agents.insert("fake".to_string(), fake_agent_config());
    let path = write_config(dir, Some("fake"), &agents, None);
    // `write_config` has no retention knob: the line lands at the head of
    // `[defaults]`, before the first `[agents.*]` section.
    let text = std::fs::read_to_string(&path)
        .expect("read config")
        .replacen(
            "[defaults]\n",
            &format!("[defaults]\ntranscript_retention = \"{retention}\"\n"),
            1,
        );
    std::fs::write(&path, text).expect("write config");
    let config = Config::load(&path).expect("config loads");
    let state = AppState::new(config, path).expect("app state");
    let tools = acpsub::build_tools(state.clone()).expect("tools");
    (state, tools)
}

/// A closed session's registry entry, `created_ago` seconds old.
fn entry(created_ago: u64) -> RegistryEntry {
    let created = SystemTime::now() - Duration::from_secs(created_ago);
    RegistryEntry {
        agent: "fake".to_string(),
        cwd: PathBuf::from("/tmp"),
        created: jiff::Timestamp::try_from(created)
            .expect("timestamp")
            .to_string(),
        last_turn: None,
        turn_started: None,
        turns: 0,
        model: None,
        mode: None,
        rate_limited: None,
        resume: None,
    }
}

/// Register a closed session.
async fn register(state: &AppState, session_id: &str, entry: RegistryEntry) {
    state
        .update_registry(|registry| registry.insert(session_id.to_string(), entry))
        .await
        .expect("registry update");
}

/// Set `path`'s mtime to `ago` seconds back.
fn age(path: &Path, ago: u64) {
    let file = std::fs::File::options()
        .write(true)
        .open(path)
        .expect("open transcript");
    file.set_modified(SystemTime::now() - Duration::from_secs(ago))
        .expect("set mtime");
}

/// Inside versus outside the window, registered versus orphaned.
#[tokio::test(flavor = "multi_thread")]
async fn expired_sessions_and_orphans_are_pruned() {
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = prune_state(dir.path(), "1h");
    let transcripts = dir.path().join("transcripts");
    std::fs::create_dir_all(&transcripts).expect("transcript dir");
    let two_hours = 2 * 60 * 60;
    let write = |name: &str| {
        let path = transcripts.join(name);
        std::fs::write(&path, "{}\n").expect("write transcript");
        path
    };

    // Registered and expired: transcript and entry both go.
    age(&write("gone.jsonl"), two_hours);
    register(&state, "gone", entry(two_hours)).await;
    // Registered and fresh: both stay.
    write("kept.jsonl");
    register(&state, "kept", entry(0)).await;
    // A registered session with no transcript falls back to the entry's
    // own stamps: the stale one goes, the fresh one stays.
    register(&state, "stale-entry", entry(two_hours)).await;
    register(&state, "fresh-entry", entry(0)).await;
    // Orphan files expire on their own mtime.
    age(&write("orphan-old.jsonl"), two_hours);
    write("orphan-new.jsonl");

    let pruned = acpsub::prune::pass(&state).await.expect("prune pass");
    assert_eq!(pruned.sessions, 2);
    assert_eq!(pruned.orphans, 1);
    assert_eq!(pruned.bytes, 6, "two 3-byte transcripts freed");
    assert!(!transcripts.join("gone.jsonl").exists());
    assert!(!transcripts.join("orphan-old.jsonl").exists());
    assert!(transcripts.join("kept.jsonl").exists());
    assert!(transcripts.join("orphan-new.jsonl").exists());
    {
        let registry = state.registry.lock().expect("registry poisoned");
        assert!(registry.get("gone").is_none());
        assert!(registry.get("stale-entry").is_none());
        assert!(registry.get("kept").is_some());
        assert!(registry.get("fresh-entry").is_some());
        drop(registry);
    }
    // A pruned id reads as unknown everywhere — never a missing-file panic.
    let err = call_err(&tools, "status", json!({"session_id": "gone"})).await;
    assert!(err.contains("unknown session"), "{err}");
    let err = call_err(&tools, "transcript", json!({"session_id": "gone"})).await;
    assert!(err.contains("no transcript"), "{err}");
    let err = call_err(&tools, "adopt", json!({"session_id": "gone"})).await;
    assert!(err.contains("cannot determine cwd"), "{err}");
}

/// A live session is never pruned, however old its transcript reads;
/// once closed it is an ordinary expired session.
#[tokio::test(flavor = "multi_thread")]
async fn live_sessions_are_never_pruned() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (state, tools) = prune_state(dir.path(), "1h");
    let sid = spawn_id(&tools, spawn_args(dir.path(), "hi")).await;
    wait(&tools, &sid, 60).await;
    let transcript = dir.path().join("transcripts").join(format!("{sid}.jsonl"));
    assert!(transcript.exists());
    age(&transcript, 2 * 60 * 60);

    let pruned = acpsub::prune::pass(&state).await.expect("prune pass");
    assert_eq!(pruned.sessions, 0);
    assert!(transcript.exists());
    assert!(state.live.lock().expect("live poisoned").contains_key(&sid));

    // Closed, the same file is unprotected.
    call_json(&tools, "close", json!({"session_id": sid})).await;
    let pruned = acpsub::prune::pass(&state).await.expect("prune pass");
    assert_eq!(pruned.sessions, 1);
    assert!(!transcript.exists());
    assert!(
        state
            .registry
            .lock()
            .expect("registry poisoned")
            .get(&sid)
            .is_none()
    );
}
