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

use aither_acp::{PermissionOptionKind, RequestPermissionOutcome, RequestPermissionParams};
use tokio::sync::oneshot;
use tracing::debug;

use crate::config::PermissionPolicy;
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
    /// The request as the agent sent it: the tool call and the options to
    /// choose from.
    pub params: RequestPermissionParams,
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
        params: RequestPermissionParams,
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
    pub async fn queue_for_permit(self) -> RequestPermissionOutcome {
        let rt = self.rt;
        let (tx, rx) = oneshot::channel();
        let request_id = {
            let mut inner = rt.inner.lock().expect("inner poisoned");
            let id = format!("perm-{}", rt.next_perm.fetch_add(1, Ordering::Relaxed));
            inner.pending.push(PendingPermission {
                request_id: id.clone(),
                tool_call: self.params.tool_call.clone(),
                options: self.params.options,
                answer: tx,
            });
            inner.status = Status::NeedsPermission;
            id
        };
        debug!(subagent = %rt.session_label(), %request_id, "permission request queued");
        rt.notify.notify_waiters();
        let outcome = rx.await.unwrap_or(RequestPermissionOutcome::Cancelled);
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
    /// The outcome for `request`: [`RequestPermissionOutcome::Selected`]
    /// with one of the request's option ids, or
    /// [`RequestPermissionOutcome::Cancelled`].
    fn decide(
        &self,
        request: PermissionRequest,
    ) -> impl Future<Output = RequestPermissionOutcome> + Send;
}

/// The default [`PermissionHandler`]: answers from the session's
/// [`PermissionPolicy`] — `allow` picks the first allow option, `deny` the
/// first reject option, `ask` queues the request for the `permit` tool.
#[derive(Debug, Clone, Copy, Default)]
pub struct PolicyPermissions;

impl PermissionHandler for PolicyPermissions {
    async fn decide(&self, request: PermissionRequest) -> RequestPermissionOutcome {
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
            .find(|option| kinds.contains(&option.kind))
            .map_or(RequestPermissionOutcome::Cancelled, |option| {
                RequestPermissionOutcome::Selected {
                    option_id: option.option_id.clone(),
                }
            })
    }
}
