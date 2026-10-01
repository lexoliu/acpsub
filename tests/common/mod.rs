//! Shared helpers for the integration tests.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use acpsub::config::{AgentConfig, SteerSemantics};
use acpsub::{AppState, Config};
use aither_core::llm::tool::{ToolContext, ToolResult, Tools};
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

/// The `fake` agent config: `python3 tests/fake_agent.py` (loadSession,
/// accepts `set_mode`/`set_config_option`).
pub fn fake_agent_config() -> AgentConfig {
    AgentConfig {
        command: "python3".to_string(),
        args: vec![fake_agent_script().to_string_lossy().into_owned()],
        env: BTreeMap::new(),
        allow_outside_cwd: false,
        permission: None,
        steer: SteerSemantics::Answered,
        sessions_db: None,
    }
}

/// Write `dir/config.toml` for `agents` under `[defaults]` pointing at
/// `dir`, and return the config file's path.
pub fn write_config(
    dir: &Path,
    default_agent: Option<&str>,
    agents: &BTreeMap<String, AgentConfig>,
) -> PathBuf {
    let mut text = format!(
        "[defaults]\npermission = \"allow\"\ntranscript_dir = \"{}\"\nregistry = \"{}\"\n",
        dir.join("transcripts").display(),
        dir.join("registry.json").display(),
    );
    if let Some(agent) = default_agent {
        let _ = writeln!(text, "agent = \"{agent}\"");
    }
    for (name, agent) in agents {
        let _ = write!(text, "\n[agents.{name}]\ncommand = {:?}", agent.command);
        if !agent.args.is_empty() {
            let args = agent
                .args
                .iter()
                .map(|arg| format!("{arg:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = write!(text, "\nargs = [{args}]");
        }
        if !agent.env.is_empty() {
            let env = agent
                .env
                .iter()
                .map(|(key, value)| format!("{key} = {value:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = write!(text, "\nenv = {{ {env} }}");
        }
        if agent.allow_outside_cwd {
            text.push_str("\nallow_outside_cwd = true");
        }
        if let Some(permission) = agent.permission {
            let name = serde_json::to_value(permission)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            let _ = write!(text, "\npermission = \"{name}\"");
        }
        if agent.steer != SteerSemantics::Answered {
            let name = serde_json::to_value(agent.steer)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            let _ = write!(text, "\nsteer = \"{name}\"");
        }
        if let Some(db) = &agent.sessions_db {
            let _ = write!(text, "\nsessions_db = {:?}", db.display().to_string());
        }
        text.push('\n');
    }
    let path = dir.join("config.toml");
    std::fs::write(&path, text).expect("write config");
    path
}

/// Build an app state rooted at `dir`, with a `fake` agent (loadSession,
/// accepts `set_mode`/`set_config_option`), a `noload` agent (same script
/// with `FAKE_NO_LOAD=1`), a `nosteer` agent (`FAKE_NO_STEER=1`, rejects a
/// concurrent prompt), a `queueagent` (`FAKE_QUEUE_PROMPTS=1`, parks a
/// concurrent prompt and runs it as its own turn), a `foldagent`
/// (`FAKE_FOLD_STEER=1` and `steer = "folded"`, folds a concurrent prompt
/// into the running turn and never answers it), and a `wideopen` agent
/// allowed outside its cwd.
pub fn test_state(dir: &Path) -> (Arc<AppState>, Tools) {
    let mut agents = BTreeMap::new();
    let fake = fake_agent_config();
    agents.insert("fake".to_string(), fake.clone());
    agents.insert(
        "noload".to_string(),
        AgentConfig {
            env: BTreeMap::from([("FAKE_NO_LOAD".to_string(), "1".to_string())]),
            ..fake.clone()
        },
    );
    agents.insert(
        "nosteer".to_string(),
        AgentConfig {
            env: BTreeMap::from([("FAKE_NO_STEER".to_string(), "1".to_string())]),
            ..fake.clone()
        },
    );
    agents.insert(
        "queueagent".to_string(),
        AgentConfig {
            env: BTreeMap::from([("FAKE_QUEUE_PROMPTS".to_string(), "1".to_string())]),
            ..fake.clone()
        },
    );
    agents.insert(
        "foldagent".to_string(),
        AgentConfig {
            env: BTreeMap::from([("FAKE_FOLD_STEER".to_string(), "1".to_string())]),
            steer: SteerSemantics::Folded,
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
    test_state_with_agents(dir, &agents)
}

/// Build an app state rooted at `dir` over an explicit agent map.
pub fn test_state_with_agents(
    dir: &Path,
    agents: &BTreeMap<String, AgentConfig>,
) -> (Arc<AppState>, Tools) {
    test_state_full(dir, None, agents)
}

/// Build an app state with an explicit default agent (or none).
pub fn test_state_full(
    dir: &Path,
    default_agent: Option<&str>,
    agents: &BTreeMap<String, AgentConfig>,
) -> (Arc<AppState>, Tools) {
    let path = write_config(dir, default_agent, agents);
    let config = Config::load(&path).expect("config loads");
    let state = AppState::new(config, path).expect("app state");
    // The test-harness constructor: a short report interval so progress
    // tests see several reports inside a test-length wait.
    let tools =
        acpsub::build_tools_with_progress_interval(state.clone(), Duration::from_millis(50))
            .expect("tools");
    (state, tools)
}

/// Call a tool; errors surface as `Err(message)`.
pub async fn call(tools: &Tools, name: &str, args: Value) -> Result<ToolResult, String> {
    call_with(tools, name, args, ToolContext::new()).await
}

/// Call a tool with an explicit context — progress tests attach a
/// listening sink.
pub async fn call_with(
    tools: &Tools,
    name: &str,
    args: Value,
    cx: ToolContext,
) -> Result<ToolResult, String> {
    tools
        .call(name, &args.to_string(), cx)
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
pub fn spawn_args(cwd: &Path, prompt: &str) -> Value {
    json!({
        "agent": "fake",
        "cwd": cwd,
        "prompt": prompt,
        "model": "b",
        "mode": "bypass",
    })
}

/// Spawn and return the assigned session id.
pub async fn spawn_id(tools: &Tools, args: Value) -> String {
    let spawned = call_json(tools, "spawn", args).await;
    spawned["session_id"]
        .as_str()
        .expect("session_id")
        .to_string()
}

/// `wait` with an expected turn duration.
pub async fn wait(tools: &Tools, session_id: &str, expect_secs: u64) -> Value {
    call_json(
        tools,
        "wait",
        json!({"session_id": session_id, "expect_secs": expect_secs}),
    )
    .await
}
