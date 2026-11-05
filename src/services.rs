//! What a host plugs into acpsub's sessions.

use crate::executor::{AnyExecutor, Executor};
use crate::permission::{AnyPermissionHandler, PermissionHandler};

/// The host-supplied parts of a session's environment.
///
/// The default keeps acpsub's own behaviour: processes run on the host through
/// [`HostExecutor`](crate::executor::HostExecutor), and permission
/// requests are answered from the session's policy.
#[derive(Debug, Clone, Default)]
pub struct Services {
    /// Starts the agent process and the agent's terminal commands.
    pub executor: AnyExecutor,
    /// Answers permission requests instead of the session's policy.
    pub permissions: Option<AnyPermissionHandler>,
}

impl Services {
    /// Start agents and their terminal commands through `executor`.
    #[must_use]
    pub fn with_executor(mut self, executor: impl Executor + 'static) -> Self {
        self.executor = AnyExecutor::new(executor);
        self
    }

    /// Answer permission requests through `handler`.
    #[must_use]
    pub fn with_permissions(mut self, handler: impl PermissionHandler + 'static) -> Self {
        self.permissions = Some(AnyPermissionHandler::new(handler));
        self
    }
}
