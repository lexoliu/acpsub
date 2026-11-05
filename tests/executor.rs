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
    AgentProcess, Config, Executor, Host, HostExecutor, PermissionHandler, PermissionRequest,
    PolicyPermissions, Services, SessionSpec, SpawnSpec, TerminalProcess, TurnEnd, probe_with,
};
use aither_acp::{RequestPermissionOutcome, StopReason};
use common::*;

struct MemorySource(Config);

impl ConfigSource for MemorySource {
    fn load(&self) -> impl Future<Output = acpsub::Result<Config>> + Send {
        std::future::ready(Ok(self.0.clone()))
    }
}

/// Which start an executor was asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Start {
    Agent,
    Terminal,
}

/// An executor that records every spawn and runs it on the host, except
/// that a terminal command is replaced by `terminal_script` when given.
#[derive(Clone, Default)]
struct Recording {
    spawns: Arc<Mutex<Vec<(Start, SpawnSpec)>>>,
    terminal_script: Option<String>,
}

impl Executor for Recording {
    type Process = <HostExecutor as Executor>::Process;
    type Stdin = <HostExecutor as Executor>::Stdin;
    type Stdout = <HostExecutor as Executor>::Stdout;
    type Stderr = <HostExecutor as Executor>::Stderr;

    fn spawn_agent(
        &self,
        spec: SpawnSpec,
    ) -> impl Future<Output = io::Result<AgentProcess<Self>>> + Send {
        self.spawns
            .lock()
            .unwrap()
            .push((Start::Agent, spec.clone()));
        async move {
            let process = HostExecutor.spawn_agent(spec).await?;
            Ok(AgentProcess {
                stdin: process.stdin,
                stdout: process.stdout,
                stderr: process.stderr,
                control: process.control,
            })
        }
    }

    fn spawn_terminal(
        &self,
        mut spec: SpawnSpec,
    ) -> impl Future<Output = io::Result<TerminalProcess<Self>>> + Send {
        self.spawns
            .lock()
            .unwrap()
            .push((Start::Terminal, spec.clone()));
        if let Some(script) = &self.terminal_script {
            spec.program = "sh".to_string();
            spec.args = vec!["-c".to_string(), script.clone()];
        }
        async move {
            let process = HostExecutor.spawn_terminal(spec).await?;
            Ok(TerminalProcess {
                stdout: process.stdout,
                stderr: process.stderr,
                control: process.control,
            })
        }
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

fn recorded(spawns: &Mutex<Vec<(Start, SpawnSpec)>>) -> Vec<(Start, SpawnSpec)> {
    spawns.lock().unwrap().clone()
}

fn default_services() -> Services {
    Services::<HostExecutor, PolicyPermissions>::default()
}

fn host<E: Executor, P: PermissionHandler>(
    dir: &Path,
    policy: PermissionPolicy,
    services: Services<E, P>,
) -> Host<E, P> {
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
        default_services().with_executor(executor),
    );
    let session = host.spawn(spec(&cwd)).await.expect("spawn");

    let seen = recorded(&spawns);
    assert_eq!(seen.len(), 1, "only the agent so far: {seen:?}");
    let (start, agent) = &seen[0];
    assert_eq!(*start, Start::Agent);
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
    let (start, terminal) = &seen[1];
    assert_eq!(*start, Start::Terminal);
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
    assert_eq!(seen[2].0, Start::Terminal);
    assert_eq!(seen[2].1.program, "bash");
    assert_eq!(seen[2].1.args, ["-c", "echo one && echo two"]);
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
        default_services().with_permissions(handler),
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
    assert_eq!(request.policy, PermissionPolicy::Deny);
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
    let host = host(dir.path(), PermissionPolicy::Deny, default_services());
    let session = host.spawn(spec(&cwd)).await.expect("spawn");
    let turn = session.prompt("RUN echo on-host").await.expect("prompt");
    assert!(turn.reply.contains("term:on-host"), "{}", turn.reply);
    let turn = session.prompt("PERMISSION").await.expect("prompt");
    assert!(turn.reply.contains("perm:deny-1"), "{}", turn.reply);
    session.close().await.expect("close");
}

/// Start `sleep` on a runtime of its own, supervise it, run `act` on its
/// handle, drop the runtime at once, and report whether the process
/// group is still there afterwards.
fn group_survives_runtime_shutdown(act: impl FnOnce(&acpsub::ProcessHandle)) -> bool {
    use futures_lite::io::{AsyncBufReadExt, BufReader};

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let pid = runtime.block_on(async {
        let process = HostExecutor
            .spawn_agent(SpawnSpec {
                program: "sh".into(),
                args: vec!["-c".into(), "echo $$; exec sleep 30".into()],
                env: std::collections::BTreeMap::new(),
                cwd: std::env::temp_dir(),
            })
            .await
            .unwrap();
        let mut line = String::new();
        BufReader::new(process.stdout)
            .read_line(&mut line)
            .await
            .unwrap();
        let handle = acpsub::ProcessHandle::supervise(process.control);
        act(&handle);
        line.trim().parse::<libc::pid_t>().unwrap()
    });
    // Shutdown right after: the supervising task may never run again.
    drop(runtime);
    // SAFETY: kill(2) with signal 0 only probes the group.
    let alive = || unsafe { libc::kill(-pid, 0) == 0 };
    for _ in 0..100 {
        if !alive() {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    // SAFETY: the test's own leftover group.
    unsafe { libc::kill(-pid, libc::SIGKILL) };
    true
}

#[test]
fn killed_handle_then_runtime_shutdown_leaves_no_process() {
    assert!(!group_survives_runtime_shutdown(
        acpsub::ProcessHandle::kill
    ));
}

#[test]
fn runtime_shutdown_kills_a_supervised_process() {
    assert!(!group_survives_runtime_shutdown(|_| {}));
}

#[tokio::test(flavor = "multi_thread")]
async fn probe_runs_through_the_supplied_executor() {
    if !python3() {
        eprintln!("skipping: python3 not found");
        return;
    }
    let executor = Recording::default();
    let spawns = executor.spawns.clone();
    let init = probe_with(
        &fake_agent_config(),
        default_services().with_executor(executor),
    )
    .await
    .expect("fake speaks ACP");
    assert!(init.agent_info.is_some());
    let seen = recorded(&spawns);
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].0, Start::Agent);
}
