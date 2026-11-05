//! A host supplies the executor that starts the agent and its terminal
//! commands, and the handler that answers its permission requests;
//! without them acpsub runs both on the host as before.

#[expect(
    dead_code,
    reason = "helpers shared across test binaries; this one uses a subset"
)]
mod common;

use std::future::Future;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use acpsub::config::{ConfigSource, Defaults, PermissionPolicy};
use acpsub::{
    Config, Executor, Host, HostExecutor, PermissionHandler, PermissionRequest, Process, Purpose,
    Services, SessionSpec, SpawnSpec, TurnEnd, probe_with,
};
use aither_acp::{RequestPermissionOutcome, StopReason};
use common::*;

struct MemorySource(Config);

impl ConfigSource for MemorySource {
    fn load(&self) -> impl Future<Output = acpsub::Result<Config>> + Send {
        std::future::ready(Ok(self.0.clone()))
    }
}

/// An executor that records every spawn and runs it on the host, except
/// that a terminal command is replaced by `terminal_script` when given.
#[derive(Clone, Default)]
struct Recording {
    spawns: Arc<Mutex<Vec<SpawnSpec>>>,
    terminal_script: Option<String>,
}

impl Executor for Recording {
    fn spawn(&self, mut spec: SpawnSpec) -> impl Future<Output = io::Result<Process>> + Send {
        self.spawns.lock().unwrap().push(spec.clone());
        if let (Purpose::Terminal, Some(script)) = (spec.purpose, &self.terminal_script) {
            spec.program = "sh".to_string();
            spec.args = vec!["-c".to_string(), script.clone()];
        }
        HostExecutor.spawn(spec)
    }
}

/// A handler that records its requests and always selects `option`.
#[derive(Clone)]
struct Selecting {
    requests: Arc<Mutex<Vec<PermissionRequest>>>,
    option: &'static str,
}

impl PermissionHandler for Selecting {
    fn decide(
        &self,
        request: PermissionRequest,
    ) -> impl Future<Output = RequestPermissionOutcome> + Send {
        self.requests.lock().unwrap().push(request);
        std::future::ready(RequestPermissionOutcome::Selected {
            option_id: self.option.into(),
        })
    }
}

fn recorded(spawns: &Mutex<Vec<SpawnSpec>>) -> Vec<SpawnSpec> {
    spawns.lock().unwrap().clone()
}

fn host(dir: &Path, policy: PermissionPolicy, services: Services) -> Host {
    let mut defaults = Defaults::new(dir.join("transcripts"), dir.join("registry.json"));
    defaults.permission = policy;
    defaults.wait_grace = Duration::from_secs(60);
    let config = Config::new(defaults).with_agent("fake", fake_agent_config());
    Host::from_config_with(config.clone(), MemorySource(config), services).expect("host")
}

fn spec(dir: &Path) -> SessionSpec {
    let mut spec = SessionSpec::new("fake", dir);
    spec.model = Some("b".to_string());
    spec.mode = Some("bypass".to_string());
    spec
}

#[tokio::test(flavor = "multi_thread")]
async fn supplied_executor_starts_the_agent_and_its_terminal_commands() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().canonicalize().unwrap();
    let executor = Recording {
        terminal_script: Some("echo via-executor".to_string()),
        ..Recording::default()
    };
    let spawns = executor.spawns.clone();
    let host = host(
        dir.path(),
        PermissionPolicy::Allow,
        Services::default().with_executor(executor),
    );
    let session = host.spawn(spec(&cwd)).await.expect("spawn");

    let seen = recorded(&spawns);
    assert_eq!(seen.len(), 1, "only the agent so far: {seen:?}");
    let agent = &seen[0];
    assert_eq!(agent.purpose, Purpose::Agent);
    assert_eq!(agent.program, "python3");
    assert_eq!(agent.args, [fake_agent_script().to_string_lossy()]);
    assert_eq!(agent.cwd, cwd);

    // The ACP stream ran over the executor's pipes: the turn completes, and
    // its terminal command ran where the executor put it.
    let turn = session.prompt("RUN echo asked").await.expect("prompt");
    assert_eq!(turn.end, TurnEnd::Stopped(StopReason::EndTurn));
    assert!(turn.reply.contains("term:via-executor"), "{}", turn.reply);
    let seen = recorded(&spawns);
    assert_eq!(seen.len(), 2, "{seen:?}");
    let terminal = &seen[1];
    assert_eq!(terminal.purpose, Purpose::Terminal);
    assert_eq!(terminal.program, "echo");
    assert_eq!(terminal.args, ["asked"]);
    assert_eq!(terminal.cwd, cwd);

    // A command line without arguments goes through a shell.
    let turn = session
        .prompt("RUNLINE echo one && echo two")
        .await
        .expect("prompt");
    assert!(turn.reply.contains("term:via-executor"), "{}", turn.reply);
    let seen = recorded(&spawns);
    assert_eq!(seen[2].program, "bash");
    assert_eq!(seen[2].args, ["-c", "echo one && echo two"]);
    session.close().await.expect("close");
}

#[tokio::test(flavor = "multi_thread")]
async fn supplied_permission_handler_answers_instead_of_the_policy() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().canonicalize().unwrap();
    let handler = Selecting {
        requests: Arc::default(),
        option: "allow-2",
    };
    let requests = handler.requests.clone();
    // The policy would deny; the handler's answer wins.
    let host = host(
        dir.path(),
        PermissionPolicy::Deny,
        Services::default().with_permissions(handler),
    );
    let session = host.spawn(spec(&cwd)).await.expect("spawn");
    let turn = session.prompt("PERMISSION").await.expect("prompt");
    assert_eq!(turn.end, TurnEnd::Stopped(StopReason::EndTurn));
    assert!(turn.reply.contains("perm:allow-2"), "{}", turn.reply);

    let seen = requests.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    let request = &seen[0];
    assert_eq!(request.session_id, session.session_id());
    assert_eq!(request.cwd, cwd);
    assert_eq!(request.params.tool_call.tool_call_id, "tc-1");
    let options: Vec<_> = request
        .params
        .options
        .iter()
        .map(|option| option.option_id.clone())
        .collect();
    assert_eq!(options, ["allow-1", "allow-2", "deny-1"]);
    session.close().await.expect("close");
}

#[tokio::test(flavor = "multi_thread")]
async fn default_services_run_on_the_host_under_the_policy() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().canonicalize().unwrap();
    let host = host(dir.path(), PermissionPolicy::Deny, Services::default());
    let session = host.spawn(spec(&cwd)).await.expect("spawn");
    let turn = session.prompt("RUN echo on-host").await.expect("prompt");
    assert!(turn.reply.contains("term:on-host"), "{}", turn.reply);
    let turn = session.prompt("PERMISSION").await.expect("prompt");
    assert!(turn.reply.contains("perm:deny-1"), "{}", turn.reply);
    session.close().await.expect("close");
}

#[tokio::test(flavor = "multi_thread")]
async fn probe_runs_through_the_supplied_executor() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let executor = Recording::default();
    let spawns = executor.spawns.clone();
    let services = Services::default().with_executor(executor);
    let init = probe_with(&fake_agent_config(), &services)
        .await
        .expect("fake speaks ACP");
    assert!(init.agent_info.is_some());
    let seen = recorded(&spawns);
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].purpose, Purpose::Agent);
}
