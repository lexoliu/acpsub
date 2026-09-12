//! Config file loading (`~/.config/acpsub/config.toml`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Error, Result};

/// Default transcript directory under the home directory.
const DEFAULT_TRANSCRIPT_DIR: &str = "~/.local/share/acpsub/transcripts";
/// Default registry file under the home directory.
const DEFAULT_REGISTRY: &str = "~/.local/share/acpsub/registry.json";

/// The default config file path: `~/.config/acpsub/config.toml`.
#[must_use]
pub fn default_config_path() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".config/acpsub/config.toml"))
}

/// Loaded acpsub configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Server-wide defaults.
    pub defaults: Defaults,
    /// Configured agents keyed by name.
    pub agents: BTreeMap<String, AgentConfig>,
}

impl Config {
    /// Load and parse a config file.
    ///
    /// A missing file is an error naming the path; so is a file that does not
    /// parse.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigMissing`], [`Error::ConfigRead`], or
    /// [`Error::ConfigParse`].
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(Error::ConfigMissing(path.to_path_buf()));
        }
        let text = std::fs::read_to_string(path).map_err(|source| Error::ConfigRead {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text).map_err(|source| Error::ConfigParse {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Parse a config from TOML text.
    ///
    /// # Errors
    ///
    /// Returns a [`toml::de::Error`] when the text is not a valid config.
    pub fn parse(text: &str) -> std::result::Result<Self, toml::de::Error> {
        let raw: RawConfig = toml::from_str(text)?;
        Ok(raw.into_config())
    }

    /// Look up an agent by name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownAgent`] listing the configured keys.
    pub fn agent(&self, name: &str) -> Result<&AgentConfig> {
        self.agents.get(name).ok_or_else(|| Error::UnknownAgent {
            name: name.to_string(),
            configured: self.agent_names(),
        })
    }

    /// Sorted configured agent names.
    #[must_use]
    pub fn agent_names(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }
}

/// Server-wide defaults.
#[derive(Debug, Clone)]
pub struct Defaults {
    /// Permission policy used when neither the agent entry nor the spawn call
    /// overrides it.
    pub permission: PermissionPolicy,
    /// Directory holding `<name>.jsonl` transcript files.
    pub transcript_dir: PathBuf,
    /// Path of the subagent registry JSON file.
    pub registry: PathBuf,
}

/// How `session/request_permission` requests are answered.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum PermissionPolicy {
    /// Auto-select the first `allow_once` option, else `allow_always`.
    Allow,
    /// Auto-select the first `reject_once` option, else `reject_always`.
    Deny,
    /// Queue the request until the `permit` tool answers it.
    #[default]
    Ask,
}

/// A configured ACP agent: the command to spawn plus session options.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Program to spawn (resolved through `PATH`).
    pub command: String,
    /// Arguments to the program.
    pub args: Vec<String>,
    /// Extra environment variables for the agent process.
    pub env: BTreeMap<String, String>,
    /// Session mode to activate with `session/set_mode` after `session/new` or
    /// `session/load`.
    pub mode: Option<String>,
    /// Session config options applied with `session/set_config_option`, after
    /// `set_mode`.
    pub config: BTreeMap<String, ConfigValue>,
    /// Whether the agent may read/write files outside the session `cwd`.
    pub allow_outside_cwd: bool,
    /// Per-agent permission policy override.
    pub permission: Option<PermissionPolicy>,
}

/// A session config option value: a select value id or a boolean.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ConfigValue {
    /// Select option: the value id to choose.
    Select(String),
    /// Boolean option: the flag state.
    Toggle(bool),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    #[serde(default)]
    defaults: RawDefaults,
    #[serde(default)]
    agents: BTreeMap<String, RawAgentConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDefaults {
    permission: Option<PermissionPolicy>,
    transcript_dir: Option<String>,
    registry: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAgentConfig {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    mode: Option<String>,
    #[serde(default)]
    config: BTreeMap<String, ConfigValue>,
    #[serde(default)]
    allow_outside_cwd: bool,
    permission: Option<PermissionPolicy>,
}

impl RawConfig {
    fn into_config(self) -> Config {
        Config {
            defaults: Defaults {
                permission: self.defaults.permission.unwrap_or_default(),
                transcript_dir: expand_tilde(
                    self.defaults
                        .transcript_dir
                        .as_deref()
                        .unwrap_or(DEFAULT_TRANSCRIPT_DIR),
                ),
                registry: expand_tilde(
                    self.defaults
                        .registry
                        .as_deref()
                        .unwrap_or(DEFAULT_REGISTRY),
                ),
            },
            agents: self
                .agents
                .into_iter()
                .map(|(name, agent)| (name, agent.into_config()))
                .collect(),
        }
    }
}

impl RawAgentConfig {
    fn into_config(self) -> AgentConfig {
        AgentConfig {
            command: self.command,
            args: self.args,
            env: self.env,
            mode: self.mode,
            config: self.config,
            allow_outside_cwd: self.allow_outside_cwd,
            permission: self.permission,
        }
    }
}

/// Expand a leading `~` or `~/` to the home directory.
///
/// Without a home directory the path is left literal — a missing directory
/// then fails later at first use with the literal path in the message.
#[must_use]
pub fn expand_tilde(path: &str) -> PathBuf {
    match path.strip_prefix("~") {
        Some(rest) if rest.is_empty() || rest.starts_with('/') => dirs::home_dir().map_or_else(
            || PathBuf::from(path),
            |home| home.join(rest.trim_start_matches('/')),
        ),
        _ => PathBuf::from(path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_config() {
        let text = r#"
[defaults]
permission = "allow"
transcript_dir = "/tmp/acpsub-t"
registry = "/tmp/acpsub-registry.json"

[agents.devin]
command = "devin"
args = ["acp"]
mode = "bypass"
config = { model = "swe-2-max" }
allow_outside_cwd = false

[agents.claude]
command = "npx"
args = ["-y", "@zed-industries/claude-code-acp"]
mode = "bypassPermissions"
permission = "ask"
env = { FOO = "bar" }
"#;
        let config = Config::parse(text).expect("config parses");
        assert_eq!(config.defaults.permission, PermissionPolicy::Allow);
        assert_eq!(
            config.defaults.transcript_dir,
            PathBuf::from("/tmp/acpsub-t")
        );
        assert_eq!(config.agents.len(), 2);
        let devin = &config.agents["devin"];
        assert_eq!(devin.command, "devin");
        assert_eq!(devin.args, ["acp"]);
        assert_eq!(devin.mode.as_deref(), Some("bypass"));
        assert_eq!(
            devin.config["model"],
            ConfigValue::Select("swe-2-max".to_string())
        );
        let claude = &config.agents["claude"];
        assert_eq!(claude.permission, Some(PermissionPolicy::Ask));
        assert_eq!(claude.env["FOO"], "bar");
    }

    #[test]
    fn defaults_apply_when_sections_missing() {
        let config = Config::parse("[agents.a]\ncommand = \"x\"\n").expect("config parses");
        assert_eq!(config.defaults.permission, PermissionPolicy::Ask);
        assert!(
            config
                .defaults
                .transcript_dir
                .ends_with("acpsub/transcripts")
        );
        assert_eq!(config.agents["a"].args, Vec::<String>::new());
        assert!(!config.agents["a"].allow_outside_cwd);
    }

    #[test]
    fn tilde_expands_to_home() {
        let expanded = expand_tilde("~/x/y");
        let home = dirs::home_dir().expect("home dir");
        assert_eq!(expanded, home.join("x/y"));
        assert_eq!(expand_tilde("/abs/path"), PathBuf::from("/abs/path"));
        assert_eq!(
            expand_tilde("~other/x"),
            PathBuf::from("~other/x"),
            "only ~ and ~/ expand"
        );
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let err = Config::parse("[agents.a]\ncommand = \"x\"\nbogus = 1\n")
            .expect_err("unknown key rejected");
        assert!(err.to_string().contains("bogus"));
    }

    #[test]
    fn missing_file_is_named() {
        let err = Config::load(Path::new("/nonexistent/acpsub/config.toml"))
            .expect_err("missing file fails");
        assert!(matches!(err, Error::ConfigMissing(_)));
        assert!(err.to_string().contains("/nonexistent/acpsub/config.toml"));
    }
}
