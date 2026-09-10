//! Per-subagent terminals: child processes backing the `terminal/*`
//! agent-to-client requests.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aither_acp::{
    TerminalCreateParams, TerminalCreateResult, TerminalExitStatus, TerminalOutputResult,
};
use aither_mcp::protocol::JsonRpcError;
use tokio::io::AsyncReadExt;

/// Output retained when `output_byte_limit` is unset: 1 MiB.
const DEFAULT_OUTPUT_LIMIT: usize = 1 << 20;

/// Live terminals of one subagent, keyed by terminal id.
pub type Terminals = tokio::sync::Mutex<HashMap<String, Arc<Terminal>>>;

/// A bounded tail buffer: once full, new bytes evict the oldest.
#[derive(Debug)]
struct Tail {
    data: Vec<u8>,
    cap: usize,
    truncated: bool,
}

impl Tail {
    fn push(&mut self, bytes: &[u8]) {
        if bytes.len() >= self.cap {
            self.data.clear();
            self.data
                .extend_from_slice(&bytes[bytes.len() - self.cap..]);
            self.truncated = true;
            return;
        }
        let overflow = (self.data.len() + bytes.len()).saturating_sub(self.cap);
        if overflow > 0 {
            self.data.drain(..overflow);
            self.truncated = true;
        }
        self.data.extend_from_slice(bytes);
    }
}

/// One spawned terminal process.
#[derive(Debug)]
pub struct Terminal {
    /// The child; `None` once reaped by `wait_for_exit`/`kill`.
    child: tokio::sync::Mutex<Option<tokio::process::Child>>,
    /// Merged stdout+stderr tail, bounded by `output_byte_limit`.
    output: Mutex<Tail>,
    /// Exit status once the process has exited.
    status: Mutex<Option<TerminalExitStatus>>,
}

impl Terminal {
    /// Snapshot of the retained output and exit state, for `terminal/output`.
    ///
    /// # Panics
    ///
    /// Panics if a terminal mutex is poisoned.
    pub fn output(&self) -> TerminalOutputResult {
        let (output, truncated) = {
            let tail = self.output.lock().expect("terminal output poisoned");
            (
                String::from_utf8_lossy(&tail.data).into_owned(),
                tail.truncated,
            )
        };
        let exit_status = self
            .status
            .lock()
            .expect("terminal status poisoned")
            .clone();
        TerminalOutputResult {
            output,
            truncated,
            exit_status,
            meta: None,
        }
    }

    /// Await process exit, for `terminal/wait_for_exit`. Idempotent: callers
    /// after the first get the recorded status.
    ///
    /// # Errors
    ///
    /// Returns an internal error if waiting on the child fails.
    ///
    /// # Panics
    ///
    /// Panics if the status mutex is poisoned.
    pub async fn wait_for_exit(&self) -> Result<TerminalExitStatus, JsonRpcError> {
        let status = self
            .status
            .lock()
            .expect("terminal status poisoned")
            .clone();
        if let Some(status) = status {
            return Ok(status);
        }
        let mut guard = self.child.lock().await;
        let Some(child) = guard.as_mut() else {
            return Err(JsonRpcError::internal_error(
                "terminal vanished before exit status was recorded",
            ));
        };
        let status = exit_status(child.wait().await.map_err(|source| {
            JsonRpcError::internal_error(format!("cannot wait for terminal: {source}"))
        })?);
        *guard = None;
        drop(guard);
        *self.status.lock().expect("terminal status poisoned") = Some(status.clone());
        Ok(status)
    }

    /// SIGKILL the process, for `terminal/kill`. Succeeds on an already-dead
    /// terminal.
    ///
    /// # Errors
    ///
    /// Returns an internal error if the kill signal fails.
    pub async fn kill(&self) -> Result<(), JsonRpcError> {
        let mut guard = self.child.lock().await;
        if let Some(child) = guard.as_mut() {
            child.start_kill().map_err(|source| {
                JsonRpcError::internal_error(format!("cannot kill terminal: {source}"))
            })?;
        }
        drop(guard);
        Ok(())
    }
}

/// A pump appending one pipe's bytes into the terminal's tail buffer.
///
/// Dropping the future is safe: whatever was already read stays recorded.
fn pump(mut pipe: impl tokio::io::AsyncRead + Unpin + Send + 'static, terminal: Arc<Terminal>) {
    tokio::spawn(async move {
        let mut buf = [0u8; 8192];
        loop {
            match pipe.read(&mut buf).await {
                Ok(0) | Err(_) => return,
                Ok(n) => terminal
                    .output
                    .lock()
                    .expect("terminal output poisoned")
                    .push(&buf[..n]),
            }
        }
    });
}

/// Register the pumps for a fresh child; call before storing it.
fn tee_pipes(child: &mut tokio::process::Child, terminal: &Arc<Terminal>) {
    if let Some(stdout) = child.stdout.take() {
        pump(stdout, terminal.clone());
    }
    if let Some(stderr) = child.stderr.take() {
        pump(stderr, terminal.clone());
    }
}

/// Convert a process exit into the wire shape.
fn exit_status(status: std::process::ExitStatus) -> TerminalExitStatus {
    #[cfg(unix)]
    let signal = {
        use std::os::unix::process::ExitStatusExt;
        status.signal().map(|signal| signal.to_string())
    };
    #[cfg(not(unix))]
    let signal = None;
    TerminalExitStatus {
        exit_code: status.code().map(i64::from),
        signal,
        meta: None,
    }
}

/// Spawn and fully wire a terminal for `terminal/create`: command, output
/// pumps, registration under a fresh `term-N` id.
///
/// `cwd` is the session working directory used when the params carry none.
///
/// # Errors
///
/// Returns an internal error if the command cannot be spawned.
pub async fn create(
    terminals: &Terminals,
    next_id: &AtomicU64,
    params: TerminalCreateParams,
    cwd: &std::path::Path,
) -> Result<TerminalCreateResult, JsonRpcError> {
    let mut command = tokio::process::Command::new(&params.command);
    command
        .args(&params.args)
        .envs(
            params
                .env
                .iter()
                .map(|var| (var.name.clone(), var.value.clone())),
        )
        .current_dir(params.cwd.clone().unwrap_or_else(|| cwd.to_path_buf()))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn().map_err(|source| {
        JsonRpcError::internal_error(format!("cannot spawn '{}': {source}", params.command))
    })?;
    let terminal = Arc::new(Terminal {
        child: tokio::sync::Mutex::new(None),
        output: Mutex::new(Tail {
            data: Vec::new(),
            cap: params
                .output_byte_limit
                .map_or(DEFAULT_OUTPUT_LIMIT, |limit| {
                    usize::try_from(limit).unwrap_or(usize::MAX)
                }),
            truncated: false,
        }),
        status: Mutex::new(None),
    });
    tee_pipes(&mut child, &terminal);
    *terminal.child.lock().await = Some(child);
    let id = format!("term-{}", next_id.fetch_add(1, Ordering::Relaxed));
    terminals.lock().await.insert(id.clone(), terminal);
    Ok(TerminalCreateResult {
        terminal_id: id,
        meta: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_keeps_newest_bytes() {
        let mut tail = Tail {
            data: Vec::new(),
            cap: 5,
            truncated: false,
        };
        tail.push(b"abc");
        assert_eq!(tail.data, b"abc");
        assert!(!tail.truncated);
        tail.push(b"defgh");
        assert_eq!(tail.data, b"defgh");
        assert!(tail.truncated);
        tail.push(b"123456");
        assert_eq!(tail.data, b"23456");
    }
}
