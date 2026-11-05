//! Who answers an agent's permission requests.
//!
//! By default acpsub answers `session/request_permission` itself from the
//! session's [`PermissionPolicy`](crate::PermissionPolicy). A host that
//! has its own permission handling supplies a [`PermissionHandler`]
//! through [`Services`](crate::Services); it then answers every request,
//! and the policy is not consulted.

use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use aither_acp::{RequestPermissionOutcome, RequestPermissionParams};

use crate::executor::BoxFuture;

/// One permission request of an agent.
#[derive(Debug, Clone)]
pub struct PermissionRequest {
    /// The session asking; empty while the agent has not assigned its
    /// id yet, during the launch handshake.
    pub session_id: String,
    /// The session's working directory.
    pub cwd: PathBuf,
    /// The request as the agent sent it: the tool call and the options to
    /// choose from.
    pub params: RequestPermissionParams,
}

/// Answers permission requests for a host.
pub trait PermissionHandler: Send + Sync {
    /// The outcome for `request`: [`RequestPermissionOutcome::Selected`]
    /// with one of the request's option ids, or
    /// [`RequestPermissionOutcome::Cancelled`].
    fn decide(
        &self,
        request: PermissionRequest,
    ) -> impl Future<Output = RequestPermissionOutcome> + Send;
}

/// Object-safe twin of [`PermissionHandler`].
trait PermissionHandlerImpl: Send + Sync {
    fn decide(&self, request: PermissionRequest) -> BoxFuture<'_, RequestPermissionOutcome>;
}

impl<T: PermissionHandler> PermissionHandlerImpl for T {
    fn decide(&self, request: PermissionRequest) -> BoxFuture<'_, RequestPermissionOutcome> {
        Box::pin(PermissionHandler::decide(self, request))
    }
}

/// A type-erased [`PermissionHandler`].
#[derive(Clone)]
pub struct AnyPermissionHandler {
    inner: Arc<dyn PermissionHandlerImpl>,
}

impl fmt::Debug for AnyPermissionHandler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnyPermissionHandler")
            .finish_non_exhaustive()
    }
}

impl AnyPermissionHandler {
    /// Erase the concrete type of `handler`.
    #[must_use]
    pub fn new(handler: impl PermissionHandler + 'static) -> Self {
        Self {
            inner: Arc::new(handler),
        }
    }

    /// See [`PermissionHandler::decide`].
    pub fn decide(
        &self,
        request: PermissionRequest,
    ) -> impl Future<Output = RequestPermissionOutcome> + Send + '_ {
        self.inner.decide(request)
    }
}
