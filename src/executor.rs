//! Where acpsub starts processes.
//!
//! Every process acpsub starts goes through an [`Executor`]: the agent
//! itself ([`Executor::spawn_agent`]) and each command the agent runs
//! through the ACP `terminal/*` requests ([`Executor::spawn_terminal`]).
//! The default, [`HostExecutor`], starts them as plain child processes of
//! acpsub, each leading its own process group. A host that keeps its
//! agents inside a sandbox supplies an executor that starts them there
//! instead, through [`Services`](crate::Services).
//!
//! The contract is process-spawn semantics: a [`SpawnSpec`] in, a process
//! out — the standard streams as async byte streams, and a
//! [`ProcessControl`] to kill the process tree and to wait for its exit.
//! The stream and control types are associated types of the executor, so
//! nothing is boxed.

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::io;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};

use futures_lite::io::{AsyncRead, AsyncWrite};
use tokio::sync::{mpsc, watch};

/// A process to start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnSpec {
    /// The executable, resolved by the executor the way its platform
    /// resolves a command name.
    pub program: String,
    /// Arguments after the program.
    pub args: Vec<String>,
    /// Variables set on top of the executor's own base environment.
    pub env: BTreeMap<String, String>,
    /// The working directory.
    pub cwd: PathBuf,
}

/// The control of a started process.
///
/// Both methods take `&self`: acpsub kills from synchronous code while
/// another task waits.
pub trait ProcessControl: Send + Sync + 'static {
    /// SIGKILL the process and everything it started — its whole process
    /// tree, including descendants the process orphaned. Idempotent, and
    /// harmless once the process is gone.
    fn kill(&self);

    /// Resolve when the process exits. Callable again after it resolved,
    /// and then yields the same status.
    fn wait(&self) -> impl Future<Output = io::Result<ExitStatus>> + Send;
}

/// The agent process an [`Executor`] started.
#[must_use = "dropping a process drops its streams"]
pub struct AgentProcess<E: Executor + ?Sized> {
    /// Its standard input: the ACP stream towards the agent.
    pub stdin: E::Stdin,
    /// Its standard output: the ACP stream from the agent.
    pub stdout: E::Stdout,
    /// Its standard error.
    pub stderr: E::Stderr,
    /// Kill and wait.
    pub control: E::Process,
}

impl<E: Executor + ?Sized> fmt::Debug for AgentProcess<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgentProcess")
            .finish_non_exhaustive()
    }
}

/// The process of a terminal command an [`Executor`] started. Its stdin
/// is closed: a terminal command reads nothing.
#[must_use = "dropping a process drops its streams"]
pub struct TerminalProcess<E: Executor + ?Sized> {
    /// Its standard output, captured as the terminal's output.
    pub stdout: E::Stdout,
    /// Its standard error, captured as the terminal's output.
    pub stderr: E::Stderr,
    /// Kill and wait.
    pub control: E::Process,
}

impl<E: Executor + ?Sized> fmt::Debug for TerminalProcess<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TerminalProcess")
            .finish_non_exhaustive()
    }
}

/// Starts the processes acpsub runs.
pub trait Executor: Send + Sync + 'static {
    /// Kill and wait for a started process.
    type Process: ProcessControl;
    /// A process's standard input.
    type Stdin: AsyncWrite + Send + Unpin + 'static;
    /// A process's standard output.
    type Stdout: AsyncRead + Send + Unpin + 'static;
    /// A process's standard error.
    type Stderr: AsyncRead + Send + Unpin + 'static;

    /// Start the agent: stdin, stdout and stderr piped.
    ///
    /// # Errors
    ///
    /// Whatever prevents the process from starting; acpsub reports it as
    /// a failed spawn.
    fn spawn_agent(
        &self,
        spec: SpawnSpec,
    ) -> impl Future<Output = io::Result<AgentProcess<Self>>> + Send;

    /// Start a terminal command: stdin closed, stdout and stderr piped.
    ///
    /// # Errors
    ///
    /// Whatever prevents the process from starting; acpsub reports it as
    /// a failed `terminal/create`.
    fn spawn_terminal(
        &self,
        spec: SpawnSpec,
    ) -> impl Future<Output = io::Result<TerminalProcess<Self>>> + Send;
}

/// A started process under supervision, cheap to clone and share.
///
/// A task owns the process's [`ProcessControl`]: it forwards kills,
/// publishes the exit status once, and kills the tree when the last
/// handle is dropped. Handles therefore kill from synchronous code and
/// read the status without a lock.
#[derive(Clone)]
pub struct ProcessHandle {
    kill: mpsc::UnboundedSender<()>,
    exit: watch::Receiver<Option<ExitStatus>>,
}

impl fmt::Debug for ProcessHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessHandle")
            .field("exit", &*self.exit.borrow())
            .finish_non_exhaustive()
    }
}

impl ProcessHandle {
    /// Supervise `control` on the current tokio runtime.
    pub fn supervise(control: impl ProcessControl) -> Self {
        let (kill, mut requests) = mpsc::unbounded_channel();
        let (publish, exit) = watch::channel(None);
        // Wrapped before the spawn, so the tree is killed when the task ends
        // or is dropped unpolled, as a runtime shutting down does.
        let control = KillOnDrop(control);
        tokio::spawn(async move {
            let mut waiting = std::pin::pin!(control.0.wait());
            let mut exited = false;
            loop {
                tokio::select! {
                    request = requests.recv() => {
                        if request.is_none() {
                            return;
                        }
                        control.0.kill();
                    }
                    status = &mut waiting, if !exited => {
                        exited = true;
                        if let Ok(status) = status {
                            let _ = publish.send(Some(status));
                        }
                    }
                }
            }
        });
        Self { kill, exit }
    }

    /// Kill the process tree. Asynchronous: the supervising task carries
    /// it out. Harmless once the process is gone.
    pub fn kill(&self) {
        let _ = self.kill.send(());
    }

    /// The exit status if the process has exited and was reaped.
    #[must_use]
    pub fn exited(&self) -> Option<ExitStatus> {
        *self.exit.borrow()
    }

    /// Resolve when the process exits.
    ///
    /// # Errors
    ///
    /// Returns an error when the process's status could not be collected.
    pub async fn wait(&self) -> io::Result<ExitStatus> {
        let mut exit = self.exit.clone();
        let status = exit
            .wait_for(Option::is_some)
            .await
            .map_err(|_| io::Error::other("the process's exit status could not be collected"))?;
        (*status).ok_or_else(|| io::Error::other("the process's exit status is missing"))
    }
}

/// A [`ProcessControl`] that kills its process tree when dropped. The kill
/// also sweeps descendants a reaped leader orphaned, and is harmless when
/// the group is gone.
struct KillOnDrop<C: ProcessControl>(C);

impl<C: ProcessControl> Drop for KillOnDrop<C> {
    fn drop(&mut self) {
        self.0.kill();
    }
}

/// The default [`Executor`]: plain child processes of acpsub, on the
/// host, each leading its own process group on unix so a kill reaches
/// every descendant.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostExecutor;

impl Executor for HostExecutor {
    type Process = HostProcess;
    type Stdin = async_process::ChildStdin;
    type Stdout = async_process::ChildStdout;
    type Stderr = async_process::ChildStderr;

    fn spawn_agent(
        &self,
        spec: SpawnSpec,
    ) -> impl Future<Output = io::Result<AgentProcess<Self>>> + Send {
        std::future::ready(spawn_agent_on_host(&spec))
    }

    fn spawn_terminal(
        &self,
        spec: SpawnSpec,
    ) -> impl Future<Output = io::Result<TerminalProcess<Self>>> + Send {
        std::future::ready(spawn_terminal_on_host(&spec))
    }
}

fn host_command(spec: &SpawnSpec) -> async_process::Command {
    let mut std_command = std::process::Command::new(&spec.program);
    std_command
        .args(&spec.args)
        .envs(&spec.env)
        .current_dir(&spec.cwd);
    // The process leads its own group so `kill` reaps every descendant —
    // orphaned ones reparent to launchd but stay in the group.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        std_command.process_group(0);
    }
    // `From` resets async-process's stdio flags, so the pipes are set on
    // the wrapper.
    let mut command = async_process::Command::from(std_command);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    command
}

fn spawn_agent_on_host(spec: &SpawnSpec) -> io::Result<AgentProcess<HostExecutor>> {
    let mut command = host_command(spec);
    command.stdin(Stdio::piped());
    let mut child = command.spawn()?;
    let missing = |stream: &str| io::Error::other(format!("{stream} was not piped"));
    let stdin = child.stdin.take().ok_or_else(|| missing("stdin"))?;
    let stdout = child.stdout.take().ok_or_else(|| missing("stdout"))?;
    let stderr = child.stderr.take().ok_or_else(|| missing("stderr"))?;
    Ok(AgentProcess {
        stdin,
        stdout,
        stderr,
        control: HostProcess::reap(child),
    })
}

fn spawn_terminal_on_host(spec: &SpawnSpec) -> io::Result<TerminalProcess<HostExecutor>> {
    let mut command = host_command(spec);
    command.stdin(Stdio::null()).kill_on_drop(true);
    let mut child = command.spawn()?;
    let missing = |stream: &str| io::Error::other(format!("{stream} was not piped"));
    let stdout = child.stdout.take().ok_or_else(|| missing("stdout"))?;
    let stderr = child.stderr.take().ok_or_else(|| missing("stderr"))?;
    Ok(TerminalProcess {
        stdout,
        stderr,
        control: HostProcess::reap(child),
    })
}

/// A process started by [`HostExecutor`]: a reaper task owns the child
/// and publishes its status once; the handle signals the process group.
#[derive(Debug)]
pub struct HostProcess {
    /// The child's process group id — the child leads it.
    pgid: u32,
    exit: watch::Receiver<Option<ExitStatus>>,
}

impl HostProcess {
    fn reap(mut child: async_process::Child) -> Self {
        let pgid = child.id();
        let (publish, exit) = watch::channel(None);
        tokio::spawn(async move {
            if let Ok(status) = child.status().await {
                let _ = publish.send(Some(status));
            }
        });
        Self { pgid, exit }
    }
}

impl ProcessControl for HostProcess {
    fn kill(&self) {
        kill_process_group(self.pgid);
    }

    async fn wait(&self) -> io::Result<ExitStatus> {
        let mut exit = self.exit.clone();
        let status = exit
            .wait_for(Option::is_some)
            .await
            .map_err(|_| io::Error::other("the child's exit status could not be collected"))?;
        (*status).ok_or_else(|| io::Error::other("the child's exit status is missing"))
    }
}

/// SIGKILL the process group led by `pgid`. A process group outlives its
/// leader, so this still reaches a dead agent's leaked descendants. Probes
/// the group first; a pgid meanwhile recycled by a foreign group leader is
/// the accepted residual.
///
/// On unix; a no-op elsewhere.
#[cfg(unix)]
fn kill_process_group(pgid: u32) {
    if pgid == 0 {
        return;
    }
    // SAFETY: kill(2) only delivers a signal; a stale pgid is narrowed by
    // the existence probe.
    unsafe {
        let pgid = pgid as libc::pid_t;
        if libc::kill(-pgid, 0) == 0 {
            libc::kill(-pgid, libc::SIGKILL);
        }
    }
}

/// No-op on platforms without process groups.
#[cfg(not(unix))]
fn kill_process_group(_pgid: u32) {}
