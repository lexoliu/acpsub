//! The library API: a host drives sessions through `Host`, `Session`
//! and `probe` against `tests/fake_agent.py`, with its config built in
//! memory and supplied by its own `ConfigSource`.

#[expect(
    dead_code,
    reason = "helpers shared across test binaries; this one uses a subset"
)]
mod common;

use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use acpsub::config::{AgentConfig, ConfigSource, Defaults, PermissionPolicy};
use acpsub::{Config, Error, Host, SessionSpec, TurnEnd, probe};
use aither_acp::StopReason;
use common::*;

/// A host-supplied source: an in-memory config, counting its loads.
struct MemorySource {
    config: Config,
    loads: Arc<AtomicUsize>,
}

impl ConfigSource for MemorySource {
    fn load(&self) -> impl Future<Output = acpsub::Result<Config>> + Send {
        self.loads.fetch_add(1, Ordering::SeqCst);
        std::future::ready(Ok(self.config.clone()))
    }
}

/// A host over an in-memory config rooted at `dir` with one agent,
/// `fake`, and the given watchdog grace.
fn host(dir: &Path, agent: AgentConfig, wait_grace: Duration) -> (Host, Arc<AtomicUsize>) {
    let mut defaults = Defaults::new(dir.join("transcripts"), dir.join("registry.json"));
    defaults.permission = PermissionPolicy::Allow;
    defaults.wait_grace = wait_grace;
    let config = Config::new(defaults).with_agent("fake", agent);
    let loads = Arc::new(AtomicUsize::new(0));
    let source = MemorySource {
        config: config.clone(),
        loads: loads.clone(),
    };
    let host = Host::from_config(config, source).expect("host");
    (host, loads)
}

fn spec(dir: &Path) -> SessionSpec {
    let mut spec = SessionSpec::new("fake", dir);
    spec.model = Some("b".to_string());
    spec.mode = Some("bypass".to_string());
    spec
}

#[tokio::test(flavor = "multi_thread")]
async fn spawn_prompt_twice_close() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (host, loads) = host(dir.path(), fake_agent_config(), Duration::from_secs(60));
    let session = host.spawn(spec(dir.path())).await.expect("spawn");
    assert_eq!(loads.load(Ordering::SeqCst), 1, "spawn reads the source");
    assert!(session.session_id().starts_with("sess-"));
    assert_eq!(session.model(), Some(&serde_json::json!("b")));
    assert_eq!(session.mode(), Some("bypass"));

    let first = session.prompt("hi").await.expect("first prompt");
    assert_eq!(first.turn, 1);
    assert_eq!(first.end, TurnEnd::Stopped(StopReason::EndTurn));
    assert!(first.reply.starts_with("Hello"), "{}", first.reply);

    // The same session takes a second turn.
    let second = session.prompt("again").await.expect("second prompt");
    assert_eq!(second.turn, 2);
    assert_eq!(second.end, TurnEnd::Stopped(StopReason::EndTurn));

    let id = session.session_id().to_string();
    session.close().await.expect("close");
    assert!(
        matches!(host.state().get(&id), Err(Error::UnknownSession(_))),
        "a closed session is no longer live"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_ends_the_pending_prompt() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (host, _) = host(dir.path(), fake_agent_config(), Duration::from_secs(60));
    let session = Arc::new(host.spawn(spec(dir.path())).await.expect("spawn"));

    // "wait" never completes until the agent is cancelled.
    let turn = tokio::spawn({
        let session = session.clone();
        async move { session.prompt("wait").await }
    });
    // Cancel once the turn runs: `cancel` refuses before that.
    let mut cancelled = false;
    for _ in 0..200 {
        if session.cancel().await.is_ok() {
            cancelled = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(cancelled, "the turn never started");
    let outcome = turn.await.unwrap().expect("prompt");
    assert_eq!(outcome.end, TurnEnd::Stopped(StopReason::Cancelled));

    // Nothing runs now: cancel is an error, and the session takes a turn.
    assert!(matches!(session.cancel().await, Err(Error::NotRunning(_))));
    let next = session.prompt("hi").await.expect("prompt after cancel");
    assert_eq!(next.end, TurnEnd::Stopped(StopReason::EndTurn));
    Arc::into_inner(session)
        .expect("sole handle")
        .close()
        .await
        .expect("close");
}

/// A cancel accepted before the turn's prompt is on the wire is sent
/// right behind the prompt instead of being lost. Holding the transcript
/// lock parks `begin_turn` after it marks the turn `Running` and before
/// it enqueues `session/prompt`.
#[tokio::test(flavor = "multi_thread")]
async fn cancel_before_the_prompt_is_on_the_wire() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (host, _) = host(dir.path(), fake_agent_config(), Duration::from_secs(60));
    let session = Arc::new(host.spawn(spec(dir.path())).await.expect("spawn"));
    let sub = host.state().get(session.session_id()).expect("live");

    let guard = sub.rt.transcript.lock().await;
    let turn = tokio::spawn({
        let session = session.clone();
        async move { session.prompt("wait").await }
    });
    // `begin_turn` is parked on the transcript once the turn is current.
    let mut cancelled = false;
    for _ in 0..200 {
        if session.cancel().await.is_ok() {
            cancelled = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(cancelled, "the turn never started");
    assert!(
        sub.rt
            .inner
            .lock()
            .unwrap()
            .current
            .as_ref()
            .is_some_and(|turn| !turn.on_wire && turn.cancel_requested),
        "the cancel must land inside the window"
    );
    drop(guard);

    let outcome = tokio::time::timeout(Duration::from_secs(30), turn)
        .await
        .expect("the cancelled turn hung")
        .unwrap()
        .expect("prompt");
    assert_eq!(outcome.end, TurnEnd::Stopped(StopReason::Cancelled));
}

#[tokio::test(flavor = "multi_thread")]
async fn awaited_prompt_feeds_the_watchdog() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // A grace far shorter than the turn: only the pending `prompt`
    // keeps the watchdog from cancelling it.
    let (host, _) = host(dir.path(), fake_agent_config(), Duration::from_millis(300));
    let session = host.spawn(spec(dir.path())).await.expect("spawn");
    let gate = dir.path().join("gate");
    let release = gate.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(1500)).await;
        std::fs::write(release, "go").unwrap();
    });
    let outcome = session
        .prompt(format!("gated {}", gate.display()))
        .await
        .expect("prompt");
    assert_eq!(outcome.end, TurnEnd::Stopped(StopReason::EndTurn));
    session.close().await.expect("close");
}

#[tokio::test(flavor = "multi_thread")]
async fn spawn_errors_like_launch() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (host, _) = host(dir.path(), fake_agent_config(), Duration::from_secs(60));

    // Advertised but not given.
    let mut missing = spec(dir.path());
    missing.model = None;
    let error = host.spawn(missing).await.expect_err("model is required");
    assert!(error.to_string().contains("model"), "{error}");
    let mut missing = spec(dir.path());
    missing.mode = None;
    let error = host.spawn(missing).await.expect_err("mode is required");
    assert!(error.to_string().contains("mode"), "{error}");

    // Given but not in the advertised list.
    let mut unknown = spec(dir.path());
    unknown.model = Some("nope".to_string());
    host.spawn(unknown).await.expect_err("unknown model");

    // Given but nothing advertised.
    let mut bare = fake_agent_config();
    bare.env.insert("FAKE_NO_MODEL".into(), "1".into());
    bare.env.insert("FAKE_NO_MODES".into(), "1".into());
    let (bare_host, _) = host_with(dir.path(), bare);
    bare_host
        .spawn(spec(dir.path()))
        .await
        .expect_err("model and mode are not advertised");

    // An unconfigured agent.
    let mut other = spec(dir.path());
    other.agent = Some("ghost".to_string());
    assert!(matches!(
        host.spawn(other).await,
        Err(Error::UnknownAgent { .. })
    ));
}

fn host_with(dir: &Path, agent: AgentConfig) -> (Host, Arc<AtomicUsize>) {
    host(dir, agent, Duration::from_secs(60))
}

#[tokio::test(flavor = "multi_thread")]
async fn probe_tells_acp_from_other_commands() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let init = probe(&fake_agent_config()).await.expect("fake speaks ACP");
    let info = init.agent_info.expect("agent info");
    assert_eq!(info.name, "fake-agent-py");
    assert!(init.agent_capabilities.load_session);

    // Exits at once.
    let silent = AgentConfig::new("true");
    assert!(matches!(probe(&silent).await, Err(Error::Agent { .. })));
    // Answers with something that is not JSON-RPC.
    let mut chatty = AgentConfig::new("echo");
    chatty.args = vec!["not acp".to_string()];
    assert!(matches!(probe(&chatty).await, Err(Error::Agent { .. })));
    // Does not exist.
    let missing = AgentConfig::new("/nonexistent/acp-agent");
    assert!(matches!(
        probe(&missing).await,
        Err(Error::SpawnFailed { .. })
    ));
}
