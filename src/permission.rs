//! Who answers an agent's permission requests.
//!
//! Every `session/request_permission` goes through a
//! [`PermissionHandler`]. The default, [`PolicyPermissions`], answers from
//! the session's [`PermissionPolicy`]. A host that has its own permission
//! handling supplies a handler through [`Services`](crate::Services).

use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use aither_acp::domain::PermissionOptionKind;
use aither_acp::{Open, PermissionOutcome, domain::PermissionRequest as AgentRequest};
use tokio::sync::oneshot;
use tracing::debug;

use crate::config::PermissionPolicy;
use crate::handler;
use crate::state::{PendingPermission, Status, SubRuntime};

/// One permission request of an agent.
#[derive(Clone)]
pub struct PermissionRequest {
    /// The session asking; empty while the agent has not assigned its
    /// id yet, during the launch handshake.
    pub session_id: String,
    /// The session's working directory.
    pub cwd: PathBuf,
    /// The session's permission policy.
    pub policy: PermissionPolicy,
    /// The request as the agent sent it: title, subject and the options
    /// to choose from.
    pub params: AgentRequest,
    rt: Arc<SubRuntime>,
}

impl fmt::Debug for PermissionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PermissionRequest")
            .field("session_id", &self.session_id)
            .field("cwd", &self.cwd)
            .field("policy", &self.policy)
            .field("params", &self.params)
            .finish_non_exhaustive()
    }
}

impl PermissionRequest {
    pub(crate) const fn new(
        session_id: String,
        cwd: PathBuf,
        policy: PermissionPolicy,
        params: AgentRequest,
        rt: Arc<SubRuntime>,
    ) -> Self {
        Self {
            session_id,
            cwd,
            policy,
            params,
            rt,
        }
    }

    /// Park the request for the `permit` tool and resolve with its
    /// answer — what the `ask` policy does. The session reports
    /// `needs_permission` meanwhile.
    ///
    /// # Panics
    ///
    /// Panics if a state mutex is poisoned.
    pub async fn queue_for_permit(self) -> PermissionOutcome {
        let rt = self.rt;
        let (tx, rx) = oneshot::channel();
        let request_id = {
            let mut inner = rt.inner.lock().expect("inner poisoned");
            let id = format!("perm-{}", rt.next_perm.fetch_add(1, Ordering::Relaxed));
            let tool_call_id = self
                .params
                .subject
                .as_ref()
                .and_then(|subject| match subject {
                    Open::Known(aither_acp::domain::PermissionSubject::ToolCall(call)) => {
                        Some(call.tool_call.tool_call_id.as_str().to_string())
                    }
                    _ => None,
                });
            inner.pending.push(PendingPermission {
                request_id: id.clone(),
                title: self.params.title.clone(),
                tool_call_id,
                options: self.params.options.clone(),
                answer: tx,
            });
            inner.status = Status::NeedsPermission;
            id
        };
        debug!(subagent = %rt.session_label(), %request_id, "permission request queued");
        rt.notify.notify_waiters();
        let outcome = rx.await.unwrap_or_else(|_| handler::cancelled());
        {
            let mut inner = rt.inner.lock().expect("inner poisoned");
            if inner.pending.is_empty() && matches!(inner.status, Status::NeedsPermission) {
                inner.status = Status::Running;
            }
        }
        rt.notify.notify_waiters();
        outcome
    }
}

/// Answers permission requests for a host.
pub trait PermissionHandler: Send + Sync + 'static {
    /// The outcome for `request`: [`PermissionOutcome::Selected`]
    /// with one of the request's option ids, or
    /// [`PermissionOutcome::Cancelled`].
    fn decide(&self, request: PermissionRequest) -> impl Future<Output = PermissionOutcome> + Send;
}

/// The default [`PermissionHandler`]: answers from the session's
/// [`PermissionPolicy`] — `allow` picks the first allow option, `deny` the
/// first reject option, `ask` queues the request for the `permit` tool.
#[derive(Debug, Clone, Copy, Default)]
pub struct PolicyPermissions;

impl PermissionHandler for PolicyPermissions {
    async fn decide(&self, request: PermissionRequest) -> PermissionOutcome {
        let kinds = match request.policy {
            PermissionPolicy::Allow => [
                PermissionOptionKind::AllowOnce,
                PermissionOptionKind::AllowAlways,
            ],
            PermissionPolicy::Deny => [
                PermissionOptionKind::RejectOnce,
                PermissionOptionKind::RejectAlways,
            ],
            PermissionPolicy::Ask => return request.queue_for_permit().await,
        };
        request
            .params
            .options
            .iter()
            .find(|option| matches!(&option.kind, Open::Known(kind) if kinds.contains(kind)))
            .map_or_else(handler::cancelled, |option| {
                handler::selected(option.option_id.as_str())
            })
    }
}
