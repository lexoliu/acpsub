//! The ACP transport over an executor-started agent process.

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::process::ExitStatus;
use std::sync::{Arc, Mutex};

use aither_acp::Lines;
use aither_acp::rpc::{Inbound, Outbound, RequestId, RpcError, Transport};
use futures_lite::io::{AsyncRead, AsyncWrite};
use serde_json::Value;
use tokio::sync::Notify;

use crate::executor::ProcessHandle;

type Result<T> = std::result::Result<T, RpcError>;

/// What the wire tee queues: `session/update` payloads for the
/// transcript, and the boundary a `session/prompt` response marks — an
/// update before it belongs to the turn the response ended, one after it
/// to the turn that follows.
#[derive(Debug)]
pub enum TeeItem {
    /// A `session/update` payload, in wire order.
    Update(Value),
    /// A `session/prompt` response passed here.
    PromptEnd,
}

/// A queue of `session/update` payloads teed off the wire for the
/// transcript, plus the signal that wakes the flusher.
///
/// The transport pushes under a plain mutex — it cannot await — and the
/// session drains the queue inside the transcript's write lock, so every
/// teed update lands ahead of the record a later wire message (the prompt
/// response's turn end, say) produces.
#[derive(Debug, Clone)]
pub struct UpdateTee {
    /// Teed items, in wire order.
    pub queue: Arc<Mutex<VecDeque<TeeItem>>>,
    /// Wakes the flusher task.
    pub notify: Arc<Notify>,
}

impl UpdateTee {
    /// A fresh, empty tee.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
        }
    }
}

/// Newline-delimited JSON-RPC over the agent's stdout and stdin, with a
/// tee on inbound `session/update` notifications for the transcript.
/// Closing it, or asking for the exit status of a process that is still
/// running, kills the agent's process tree and reaps it.
pub struct ProcessTransport<R, W> {
    lines: Lines<R, W>,
    process: ProcessHandle,
    tee: Option<UpdateTee>,
    /// `session/prompt` request ids still unanswered: their responses
    /// mark turn boundaries in `tee`.
    prompt_ids: Arc<Mutex<HashSet<RequestId>>>,
}

impl<R, W> fmt::Debug for ProcessTransport<R, W> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessTransport")
            .finish_non_exhaustive()
    }
}

impl<R: AsyncRead + Send + Unpin, W: AsyncWrite + Send + Unpin> ProcessTransport<R, W> {
    /// A transport reading `stdout` and writing `stdin` of `process`,
    /// teeing every `session/update` to `tee`.
    pub fn new(stdout: R, stdin: W, process: ProcessHandle, tee: UpdateTee) -> Self {
        Self {
            lines: Lines::new(stdout, stdin),
            process,
            tee: Some(tee),
            prompt_ids: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Push a teed item and wake the flusher.
    fn tee_push(&self, item: TeeItem) {
        if let Some(tee) = &self.tee {
            tee.queue
                .lock()
                .expect("tee queue poisoned")
                .push_back(item);
            tee.notify.notify_one();
        }
    }
}

impl<R: AsyncRead + Send + Unpin, W: AsyncWrite + Send + Unpin> Transport
    for ProcessTransport<R, W>
{
    async fn recv(&mut self) -> Result<Option<Inbound>> {
        let inbound = self.lines.recv().await?;
        match &inbound {
            Some(Inbound::Notification(notification))
                if notification.method == "session/update" =>
            {
                if let Some(update) = notification
                    .params
                    .as_ref()
                    .and_then(|params| params.get("update"))
                {
                    self.tee_push(TeeItem::Update(update.clone()));
                }
            }
            Some(Inbound::Response(response)) => {
                let boundary = response.id.as_ref().is_some_and(|id| {
                    self.prompt_ids
                        .lock()
                        .expect("prompt ids poisoned")
                        .remove(id)
                });
                if boundary {
                    self.tee_push(TeeItem::PromptEnd);
                }
            }
            _ => {}
        }
        Ok(inbound)
    }

    fn send(&mut self, msg: &Outbound) -> impl Future<Output = Result<()>> + Send {
        // Track `session/prompt` request ids so their responses mark the
        // tee's turn boundary — updates after it belong to a next turn.
        if let Outbound::Request(request) = msg
            && request.method == "session/prompt"
        {
            self.prompt_ids
                .lock()
                .expect("prompt ids poisoned")
                .insert(request.id.clone());
        }
        self.lines.send(msg)
    }

    async fn close(&mut self) -> Result<()> {
        let closed = self.lines.close().await;
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
