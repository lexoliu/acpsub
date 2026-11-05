//! The ACP transport over an executor-started agent process.

use std::fmt;
use std::process::ExitStatus;

use aither_mcp::protocol::{
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError,
};
use aither_mcp::transport::{BidirectionalTransport, StreamTransport, Transport};
use futures_lite::io::{AsyncRead, AsyncWrite, BufReader};

use crate::executor::ProcessHandle;

type Result<T> = std::result::Result<T, McpError>;

/// Newline-delimited JSON-RPC over the agent's stdout and stdin. Closing
/// it, or asking for the exit status of a process that is still running,
/// kills the agent's process tree and reaps it.
pub struct ProcessTransport<R, W> {
    stream: StreamTransport<BufReader<R>, W>,
    process: ProcessHandle,
}

impl<R, W> fmt::Debug for ProcessTransport<R, W> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessTransport")
            .finish_non_exhaustive()
    }
}

impl<R: AsyncRead + Send + Unpin, W: AsyncWrite + Send + Unpin> ProcessTransport<R, W> {
    /// A transport reading `stdout` and writing `stdin` of `process`.
    pub fn new(stdout: R, stdin: W, process: ProcessHandle) -> Self {
        Self {
            stream: StreamTransport::new(BufReader::new(stdout), stdin),
            process,
        }
    }
}

impl<R: AsyncRead + Send + Unpin, W: AsyncWrite + Send + Unpin> Transport
    for ProcessTransport<R, W>
{
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
        if let Some(status) = self.process.exited() {
            Some(status)
        } else {
            // The pipe is gone but the process is still running; it can no
            // longer be reached, so collect a real status by reaping it.
            self.process.kill();
            self.process.wait().await.ok()
        }
    }
}

impl<R: AsyncRead + Send + Unpin, W: AsyncWrite + Send + Unpin> BidirectionalTransport
    for ProcessTransport<R, W>
{
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
