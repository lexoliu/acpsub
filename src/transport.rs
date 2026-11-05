//! The ACP transport over an executor-started agent process.

use std::fmt;
use std::process::ExitStatus;

use aither_mcp::protocol::{
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError,
};
use aither_mcp::transport::{BidirectionalTransport, StreamTransport, Transport};
use futures_lite::io::BufReader;

use crate::executor::{ProcessHandle, ProcessOutput, ProcessStdin};

type Result<T> = std::result::Result<T, McpError>;

/// Newline-delimited JSON-RPC over the agent's stdout and stdin. Closing
/// it, or asking for the exit status of a process that is still running,
/// kills the agent's process tree and reaps it.
pub struct ProcessTransport {
    stream: StreamTransport<BufReader<ProcessOutput>, ProcessStdin>,
    process: ProcessHandle,
}

impl fmt::Debug for ProcessTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessTransport")
            .finish_non_exhaustive()
    }
}

impl ProcessTransport {
    pub fn new(stdout: ProcessOutput, stdin: ProcessStdin, process: ProcessHandle) -> Self {
        Self {
            stream: StreamTransport::new(BufReader::new(stdout), stdin),
            process,
        }
    }
}

impl Transport for ProcessTransport {
    fn request(
        &mut self,
        req: JsonRpcRequest,
    ) -> impl Future<Output = Result<JsonRpcResponse>> + Send {
        self.stream.request(req)
    }

    fn notify(&mut self, notif: JsonRpcNotification) -> impl Future<Output = Result<()>> + Send {
        self.stream.notify(notif)
    }

    async fn close(&mut self) -> Result<()> {
        let closed = self.stream.close().await;
        self.process.kill();
        let _ = self.process.wait().await;
        closed
    }

    async fn exit_status(&mut self) -> Option<ExitStatus> {
        if let Ok(Some(status)) = self.process.try_wait() {
            Some(status)
        } else {
            // The pipe is gone but the process is still running; it can no
            // longer be reached, so collect a real status by reaping it.
            self.process.kill();
            self.process.wait().await.ok()
        }
    }
}

impl BidirectionalTransport for ProcessTransport {
    fn recv(&mut self) -> impl Future<Output = Result<Option<JsonRpcMessage>>> + Send {
        self.stream.recv()
    }

    fn respond(&mut self, response: JsonRpcResponse) -> impl Future<Output = Result<()>> + Send {
        self.stream.respond(response)
    }

    fn send_request(&mut self, req: JsonRpcRequest) -> impl Future<Output = Result<()>> + Send {
        self.stream.send_request(req)
    }
}
