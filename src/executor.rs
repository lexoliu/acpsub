//! Where acpsub starts processes.
//!
//! Every process acpsub starts goes through an [`Executor`]: the agent
//! itself ([`Purpose::Agent`]) and each command the agent runs through
//! the ACP `terminal/*` requests ([`Purpose::Terminal`]). The default,
//! [`HostExecutor`], starts them as plain child processes of acpsub, each
//! leading its own process group. A host that keeps its agents inside a
//! sandbox supplies an executor that starts them there instead, through
//! [`Services`](crate::Services).
//!
//! The contract is process-spawn semantics: a [`SpawnSpec`] in, a
//! [`Process`] out — the standard streams as async byte streams, and a
//! [`ProcessControl`] to kill the process tree and to wait for its exit.

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::io;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};

use futures_lite::io::{AsyncRead, AsyncWrite};

/// What a process is for, which fixes how its standard input is wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// The ACP agent: its stdin and stdout carry the ACP stream, so
    /// [`Process::stdin`] is `Some`.
    Agent,
    /// A command the agent asked to run through `terminal/create`: its
    /// stdin is closed ([`Process::stdin`] is `None`) and its stdout and
    /// stderr are captured as the terminal's output.
    Terminal,
}

/// A process to start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnSpec {
    /// What the process is for.
    pub purpose: Purpose,
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

/// A boxed future, as [`ProcessControl`] returns them to stay object-safe.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The standard input of a [`Process`].
pub type ProcessStdin = Box<dyn AsyncWrite + Send + Unpin>;

/// An output stream of a [`Process`].
pub type ProcessOutput = Box<dyn AsyncRead + Send + Unpin>;

/// The handle on a started process.
///
/// All methods take `&self`: acpsub kills from synchronous code and from
/// other tasks while one task waits.
pub trait ProcessControl: Send + Sync {
    /// SIGKILL the process and everything it started — its whole process
    /// tree, including descendants the process orphaned. Idempotent, and
    /// harmless once the process is gone.
    fn kill(&self);

    /// The exit status if the process has already exited, else `None`.
    ///
    /// # Errors
    ///
    /// Whatever the platform reports when the status cannot be read.
    fn try_wait(&self) -> io::Result<Option<ExitStatus>>;

    /// Resolve when the process exits. Callable again after it
    /// resolved, and then yields the same status.
    fn wait(&self) -> BoxFuture<'_, io::Result<ExitStatus>>;
}

/// A shared [`ProcessControl`].
#[derive(Clone)]
pub struct ProcessHandle(Arc<dyn ProcessControl>);

impl fmt::Debug for ProcessHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessHandle")
            .finish_non_exhaustive()
    }
}

impl ProcessHandle {
    /// Wrap `control`.
    #[must_use]
    pub fn new(control: impl ProcessControl + 'static) -> Self {
        Self(Arc::new(control))
    }

    /// See [`ProcessControl::kill`].
    pub fn kill(&self) {
        self.0.kill();
    }

    /// See [`ProcessControl::try_wait`].
    ///
    /// # Errors
    ///
    /// Whatever the control reports.
    pub fn try_wait(&self) -> io::Result<Option<ExitStatus>> {
        self.0.try_wait()
    }

    /// See [`ProcessControl::wait`].
    pub fn wait(&self) -> BoxFuture<'_, io::Result<ExitStatus>> {
        self.0.wait()
    }
}

/// A started process.
#[must_use = "dropping a process drops its streams"]
pub struct Process {
    /// Its standard input; `Some` exactly for [`Purpose::Agent`].
    pub stdin: Option<ProcessStdin>,
    /// Its standard output.
    pub stdout: ProcessOutput,
    /// Its standard error.
    pub stderr: ProcessOutput,
    /// Kill and wait.
    pub control: ProcessHandle,
}

impl fmt::Debug for Process {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Process").finish_non_exhaustive()
    }
}

/// Starts the processes acpsub runs.
pub trait Executor: Send + Sync {
    /// Start the process `spec` describes.
    ///
    /// # Errors
    ///
    /// Whatever prevents the process from starting; acpsub reports it as
    /// a failed spawn (agent) or a failed `terminal/create` (terminal).
    fn spawn(&self, spec: SpawnSpec) -> impl Future<Output = io::Result<Process>> + Send;
}

/// Object-safe twin of [`Executor`].
trait ExecutorImpl: Send + Sync {
    fn spawn(&self, spec: SpawnSpec) -> BoxFuture<'_, io::Result<Process>>;
}

impl<T: Executor> ExecutorImpl for T {
    fn spawn(&self, spec: SpawnSpec) -> BoxFuture<'_, io::Result<Process>> {
        Box::pin(Executor::spawn(self, spec))
    }
}

/// A type-erased [`Executor`], as [`crate::AppState`] stores it. The
/// default is [`HostExecutor`].
#[derive(Clone)]
pub struct AnyExecutor {
    inner: Arc<dyn ExecutorImpl>,
}

impl fmt::Debug for AnyExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnyExecutor")
            .finish_non_exhaustive()
    }
}

impl Default for AnyExecutor {
    fn default() -> Self {
        Self::new(HostExecutor)
    }
}

impl AnyExecutor {
    /// Erase the concrete type of `executor`.
    #[must_use]
    pub fn new(executor: impl Executor + 'static) -> Self {
        Self {
            inner: Arc::new(executor),
        }
    }

    /// See [`Executor::spawn`].
    pub fn spawn(&self, spec: SpawnSpec) -> impl Future<Output = io::Result<Process>> + Send + '_ {
        self.inner.spawn(spec)
    }
}

/// The default [`Executor`]: plain child processes of acpsub, on the
/// host, each leading its own process group on unix so a kill reaches
/// every descendant.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostExecutor;

impl Executor for HostExecutor {
    fn spawn(&self, spec: SpawnSpec) -> impl Future<Output = io::Result<Process>> + Send {
        std::future::ready(spawn_on_host(&spec))
    }
}

fn spawn_on_host(spec: &SpawnSpec) -> io::Result<Process> {
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
    match spec.purpose {
        Purpose::Agent => {
            command.stdin(Stdio::piped());
        }
        Purpose::Terminal => {
            command.stdin(Stdio::null()).kill_on_drop(true);
        }
    }
    let mut child = command.spawn()?;
    let missing = |stream: &str| io::Error::other(format!("{stream} was not piped"));
    let stdin = match spec.purpose {
        Purpose::Agent => {
            Some(Box::new(child.stdin.take().ok_or_else(|| missing("stdin"))?) as ProcessStdin)
        }
        Purpose::Terminal => None,
    };
    let stdout = Box::new(child.stdout.take().ok_or_else(|| missing("stdout"))?) as ProcessOutput;
    let stderr = Box::new(child.stderr.take().ok_or_else(|| missing("stderr"))?) as ProcessOutput;
    Ok(Process {
        stdin,
        stdout,
        stderr,
        control: ProcessHandle::new(HostProcess {
            pgid: child.id(),
            child: tokio::sync::Mutex::new(child),
            status: Mutex::new(None),
        }),
    })
}

/// A process started by [`HostExecutor`].
struct HostProcess {
    /// The child's process group id — the child leads it.
    pgid: u32,
    child: tokio::sync::Mutex<async_process::Child>,
    /// The status once reaped: a reaped child cannot be asked again.
    status: Mutex<Option<ExitStatus>>,
}

impl HostProcess {
    fn recorded(&self) -> Option<ExitStatus> {
        *self.status.lock().expect("status poisoned")
    }

    fn record(&self, status: ExitStatus) -> ExitStatus {
        *self.status.lock().expect("status poisoned") = Some(status);
        status
    }
}

impl ProcessControl for HostProcess {
    fn kill(&self) {
        kill_process_group(self.pgid);
        // The group kill reached the child on unix; elsewhere this is the
        // kill. A child being waited on is already covered by the former.
        if let Ok(mut child) = self.child.try_lock() {
            let _ = child.kill();
        }
    }

    fn try_wait(&self) -> io::Result<Option<ExitStatus>> {
        if let Some(status) = self.recorded() {
            return Ok(Some(status));
        }
        let Ok(mut child) = self.child.try_lock() else {
            return Ok(None);
        };
        Ok(child.try_status()?.map(|status| self.record(status)))
    }

    fn wait(&self) -> BoxFuture<'_, io::Result<ExitStatus>> {
        Box::pin(async move {
            if let Some(status) = self.recorded() {
                return Ok(status);
            }
            let mut child = self.child.lock().await;
            if let Some(status) = self.recorded() {
                return Ok(status);
            }
            let status = child.status().await?;
            drop(child);
            Ok(self.record(status))
        })
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
