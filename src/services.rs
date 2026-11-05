//! What a host plugs into acpsub's sessions.

use std::path::PathBuf;
use std::sync::Arc;

use aither_acp::{RequestPermissionOutcome, TerminalCreateParams, TerminalCreateResult};
use aither_mcp::protocol::JsonRpcError;
use tokio::sync::{mpsc, oneshot};

use crate::executor::{Executor, HostExecutor};
use crate::permission::{PermissionHandler, PermissionRequest, PolicyPermissions};
use crate::state::SubRuntime;

/// The host-supplied parts of a session's environment.
///
/// The default keeps acpsub's own behaviour: processes run on the host
/// through [`HostExecutor`], and permission requests are answered from the
/// session's policy by [`PolicyPermissions`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Services<E = HostExecutor, P = PolicyPermissions> {
    /// Starts the agent process and the agent's terminal commands.
    pub executor: E,
    /// Answers the agent's permission requests.
    pub permissions: P,
}

impl<E: Executor, P: PermissionHandler> Services<E, P> {
    /// Start agents and their terminal commands through `executor`.
    #[must_use]
    pub fn with_executor<E2: Executor>(self, executor: E2) -> Services<E2, P> {
        Services {
            executor,
            permissions: self.permissions,
        }
    }

    /// Answer permission requests through `permissions`.
    #[must_use]
    pub fn with_permissions<P2: PermissionHandler>(self, permissions: P2) -> Services<E, P2> {
        Services {
            executor: self.executor,
            permissions,
        }
    }
}

/// A request of the agent that the host's services answer.
#[derive(Debug)]
pub(crate) enum ServiceCall {
    /// `terminal/create`.
    Terminal {
        params: TerminalCreateParams,
        reply: oneshot::Sender<Result<TerminalCreateResult, JsonRpcError>>,
    },
    /// `session/request_permission`.
    Permission {
        request: PermissionRequest,
        reply: oneshot::Sender<RequestPermissionOutcome>,
    },
}

/// The sending half of a session's service channel.
pub(crate) type ServiceCalls = mpsc::UnboundedSender<ServiceCall>;

/// Answer a session's [`ServiceCall`]s through `services` until the
/// session's handler is dropped.
///
/// The handler of a session is one concrete type whatever the host plugs
/// in, so it reaches the generic services over this channel. Each call
/// runs on its own task: a permission request waiting for the `permit`
/// tool must not hold back a terminal.
pub(crate) async fn serve<E: Executor, P: PermissionHandler>(
    services: Arc<Services<E, P>>,
    cwd: PathBuf,
    rt: Arc<SubRuntime>,
    mut calls: mpsc::UnboundedReceiver<ServiceCall>,
) {
    while let Some(call) = calls.recv().await {
        let services = services.clone();
        match call {
            ServiceCall::Terminal { params, reply } => {
                let cwd = cwd.clone();
                let rt = rt.clone();
                tokio::spawn(async move {
                    let created = crate::terminal::create(
                        &rt.terminals,
                        &rt.next_term,
                        params,
                        &cwd,
                        &services.executor,
                    )
                    .await;
                    let _ = reply.send(created);
                });
            }
            ServiceCall::Permission { request, reply } => {
                tokio::spawn(async move {
                    let _ = reply.send(services.permissions.decide(request).await);
                });
            }
        }
    }
}
