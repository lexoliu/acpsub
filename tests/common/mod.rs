//! Shared helpers for the integration tests.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use acpsub::config::{AgentConfig, ConfigValue};
use acpsub::{AppState, Config, Defaults, PermissionPolicy};
use aither_core::llm::tool::{ToolResult, Tools};
use serde_json::{Value, json};

/// Path to the fake agent script.
pub fn fake_agent_script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fake_agent.py")
}

/// Whether `python3` is available; python-based tests skip otherwise.
pub fn python3() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Build an app state rooted at `dir`, with a `fake` agent (loadSession,
/// accepts `set_mode`/`set_config_option`), a `noload` agent (same script with
/// `FAKE_NO_LOAD=1`), and a `wideopen` agent allowed outside its cwd.
pub fn test_state(dir: &Path) -> (Arc<AppState>, Tools) {
    let script = fake_agent_script();
    let mut agents = BTreeMap::new();
    let fake = AgentConfig {
        command: "python3".to_string(),
        args: vec![script.to_string_lossy().into_owned()],
        env: BTreeMap::new(),
        mode: Some("bypass".to_string()),
        config: BTreeMap::from([("model".to_string(), ConfigValue::Select("b".to_string()))]),
        allow_outside_cwd: false,
        permission: None,
    };
    agents.insert("fake".to_string(), fake.clone());
    agents.insert(
        "noload".to_string(),
        AgentConfig {
            env: BTreeMap::from([("FAKE_NO_LOAD".to_string(), "1".to_string())]),
            ..fake.clone()
        },
    );
    agents.insert(
        "wideopen".to_string(),
        AgentConfig {
            allow_outside_cwd: true,
            ..fake
        },
    );
    let config = Config {
        defaults: Defaults {
            permission: PermissionPolicy::Allow,
            transcript_dir: dir.join("transcripts"),
            registry: dir.join("registry.json"),
        },
        agents,
    };
    let state = AppState::new(config).expect("app state");
    let tools = acpsub::build_tools(state.clone()).expect("tools");
    (state, tools)
}

/// Call a tool; errors surface as `Err(message)`.
pub async fn call(tools: &Tools, name: &str, args: Value) -> Result<ToolResult, String> {
    tools
        .call(name, &args.to_string())
        .await
        .map_err(|error| error.to_string())
}

/// Call a tool and unwrap a non-error result as JSON.
pub async fn call_json(tools: &Tools, name: &str, args: Value) -> Value {
    let result = call(tools, name, args)
        .await
        .unwrap_or_else(|e| panic!("{name} failed: {e}"));
    assert!(
        !result.is_error(),
        "{name} returned error: {}",
        result.error_message().unwrap_or("?")
    );
    let text = result.render_for_model().expect("render");
    serde_json::from_str(&text).expect("tool result is json")
}

/// Call a tool and return the error message.
pub async fn call_err(tools: &Tools, name: &str, args: Value) -> String {
    match call(tools, name, args).await {
        Err(error) => error,
        Ok(result) if result.is_error() => {
            result.error_message().expect("error message").to_string()
        }
        Ok(result) => panic!(
            "expected error, got {}",
            result.render_for_model().unwrap_or_default()
        ),
    }
}

/// Call a tool returning plain text.
pub async fn call_text(tools: &Tools, name: &str, args: Value) -> String {
    let result = call(tools, name, args).await.expect("call failed");
    result.render_for_model().expect("render")
}

/// Spawn args for the fake agent rooted at `cwd`.
pub fn spawn_args(name: &str, cwd: &Path, prompt: &str) -> Value {
    json!({
        "name": name,
        "agent": "fake",
        "cwd": cwd,
        "prompt": prompt,
    })
}

/// `wait` with a bounded timeout.
pub async fn wait(tools: &Tools, name: &str, timeout_secs: u64) -> Value {
    call_json(
        tools,
        "wait",
        json!({"name": name, "timeout_secs": timeout_secs}),
    )
    .await
}
