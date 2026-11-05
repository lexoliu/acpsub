//! Per-subagent terminals: child processes backing the `terminal/*`
//! agent-to-client requests.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aither_acp::{
    TerminalCreateParams, TerminalCreateResult, TerminalExitStatus, TerminalOutputResult,
};
use aither_mcp::protocol::JsonRpcError;
use futures_lite::io::AsyncReadExt;

use crate::executor::{AnyExecutor, ProcessHandle, ProcessOutput, Purpose, SpawnSpec};

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
    /// Kills the process tree and waits for its exit.
    control: ProcessHandle,
    /// Merged stdout+stderr tail, bounded by `output_byte_limit`.
    output: Mutex<Tail>,
    /// Exit status once the process has exited.
    status: Mutex<Option<TerminalExitStatus>>,
}

impl Drop for Terminal {
    /// Kill the whole tree so descendants the leader orphaned die with it.
    fn drop(&mut self) {
        self.control.kill();
    }
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
        let status = exit_status(self.control.wait().await.map_err(|source| {
            JsonRpcError::internal_error(format!("cannot wait for terminal: {source}"))
        })?);
        *self.status.lock().expect("terminal status poisoned") = Some(status.clone());
        Ok(status)
    }

    /// SIGKILL the process tree, for `terminal/kill`. Succeeds on an
    /// already-dead terminal.
    pub fn kill(&self) {
        self.control.kill();
    }
}

/// A pump appending one pipe's bytes into the terminal's tail buffer.
///
/// Dropping the future is safe: whatever was already read stays recorded.
fn pump(mut pipe: ProcessOutput, terminal: Arc<Terminal>) {
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
/// The process is started through `executor`.
///
/// # Errors
///
/// Returns an internal error if the command cannot be spawned.
pub async fn create(
    terminals: &Terminals,
    next_id: &AtomicU64,
    params: TerminalCreateParams,
    cwd: &std::path::Path,
    executor: &AnyExecutor,
) -> Result<TerminalCreateResult, JsonRpcError> {
    // With no `args`, `command` is a shell command line (agents like `devin
    // acp` send e.g. "cd /repo && cargo test"), so run it through a shell.
    // With `args`, `command` is the executable per the ACP wire contract.
    let (program, args) = if params.args.is_empty() {
        (
            "bash".to_string(),
            vec!["-c".to_string(), params.command.clone()],
        )
    } else {
        (params.command.clone(), params.args.clone())
    };
    let process = executor
        .spawn(SpawnSpec {
            purpose: Purpose::Terminal,
            program,
            args,
            env: params
                .env
                .iter()
                .map(|var| (var.name.clone(), var.value.clone()))
                .collect(),
            cwd: params.cwd.clone().unwrap_or_else(|| cwd.to_path_buf()),
        })
        .await
        .map_err(|source| {
            JsonRpcError::internal_error(format!("cannot spawn '{}': {source}", params.command))
        })?;
    let terminal = Arc::new(Terminal {
        control: process.control,
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
    pump(process.stdout, terminal.clone());
    pump(process.stderr, terminal.clone());
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
