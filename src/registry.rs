//! The session registry: a JSON file mapping session ids to sessions.
//!
//! The registry survives server restarts; a registered session can be
//! resumed with `session/load` (via the `adopt` tool) when the agent
//! supports it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Counter making each registry temp file unique within a process.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// One registered session: everything needed to resume it, keyed in the
/// registry by its ACP session id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    /// Agent config key that owns the session.
    pub agent: String,
    /// Session working directory.
    pub cwd: PathBuf,
    /// RFC 3339 timestamp of when the session was first registered.
    pub created: String,
    /// RFC 3339 timestamp of the last completed turn, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_turn: Option<String>,
    /// Number of completed turns.
    #[serde(default)]
    pub turns: u64,
}

/// The session id → entry map backed by `registry.json`.
#[derive(Debug, Default)]
pub struct Registry {
    entries: BTreeMap<String, RegistryEntry>,
}

impl Registry {
    /// Load the registry file; a missing file yields an empty registry.
    ///
    /// Registries written by older acpsub versions were keyed by a
    /// caller-chosen name and carried the session id inside each entry;
    /// those are re-keyed by session id on load.
    ///
    /// A file that exists but does not parse is an error naming the path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RegistryParse`] or an I/O error.
    pub fn load(path: &Path) -> Result<Self> {
        let parse = |source| Error::RegistryParse {
            path: path.to_path_buf(),
            source,
        };
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let raw: BTreeMap<String, serde_json::Value> =
                    serde_json::from_str(&text).map_err(parse)?;
                let mut entries = BTreeMap::new();
                for (key, value) in raw {
                    // Old format: the session id lived inside the entry.
                    let session_id = value
                        .get("session_id")
                        .and_then(serde_json::Value::as_str)
                        .map_or_else(|| key.clone(), str::to_string);
                    let entry: RegistryEntry = serde_json::from_value(value).map_err(parse)?;
                    entries.insert(session_id, entry);
                }
                Ok(Self { entries })
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(Error::io(format!("cannot read {}", path.display()), source)),
        }
    }

    /// Look up a registered entry by session id.
    #[must_use]
    pub fn get(&self, session_id: &str) -> Option<&RegistryEntry> {
        self.entries.get(session_id)
    }

    /// Look up a registered entry mutably.
    pub fn get_mut(&mut self, session_id: &str) -> Option<&mut RegistryEntry> {
        self.entries.get_mut(session_id)
    }

    /// Insert or replace the entry for a session id.
    pub fn insert(&mut self, session_id: String, entry: RegistryEntry) {
        self.entries.insert(session_id, entry);
    }

    /// Remove an entry, returning whether one existed.
    pub fn remove(&mut self, session_id: &str) -> bool {
        self.entries.remove(session_id).is_some()
    }

    /// Iterate over entries by session id.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &RegistryEntry)> {
        self.entries.iter()
    }

    /// Serialize the registry for persistence.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn snapshot(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.entries)
            .map_err(|source| Error::io("cannot serialize registry", source.into()))
    }
}

/// Persist a registry snapshot atomically: write a sibling temp file, then
/// rename it over the target. Runs on `spawn_blocking` so callers stay async.
///
/// # Errors
///
/// Returns an error if the parent directory cannot be created or the write or
/// rename fails.
pub async fn persist(path: &Path, snapshot: String) -> Result<()> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| {
                Error::io(format!("cannot create {}", parent.display()), source)
            })?;
        }
        let tmp = path.with_file_name(format!(
            ".{}.{}.tmp",
            path.file_name().map_or_else(
                || "registry".to_string(),
                |name| name.to_string_lossy().into_owned()
            ),
            std::process::id().to_string()
                + "-"
                + &TMP_COUNTER.fetch_add(1, Ordering::Relaxed).to_string()
        ));
        std::fs::write(&tmp, snapshot)
            .and_then(|()| std::fs::rename(&tmp, &path))
            .map_err(|source| Error::io(format!("cannot write {}", path.display()), source))
    })
    .await
    .map_err(|join| Error::io("registry writer task failed", std::io::Error::other(join)))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_round_trips() {
        let mut registry = Registry::default();
        registry.insert(
            "sess-1".to_string(),
            RegistryEntry {
                agent: "devin".to_string(),
                cwd: PathBuf::from("/tmp"),
                created: "2026-09-10T00:00:00Z".to_string(),
                last_turn: None,
                turns: 2,
            },
        );
        let snapshot = registry.snapshot().expect("snapshot");
        let loaded: BTreeMap<String, RegistryEntry> =
            serde_json::from_str(&snapshot).expect("parse snapshot");
        assert_eq!(loaded["sess-1"].turns, 2);
        assert_eq!(loaded["sess-1"].agent, "devin");
    }

    #[test]
    fn loads_name_keyed_legacy_format() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("registry.json");
        std::fs::write(
            &path,
            r#"{"logical-name": {"agent": "devin", "session_id": "sess-9",
                "cwd": "/tmp/x", "created": "t", "turns": 3}}"#,
        )
        .expect("write");
        let registry = Registry::load(&path).expect("load migrates");
        let entry = registry.get("sess-9").expect("re-keyed by session id");
        assert_eq!(entry.agent, "devin");
        assert_eq!(entry.turns, 3);
        assert!(registry.get("logical-name").is_none());
    }

    #[test]
    fn missing_registry_loads_empty() {
        let registry = Registry::load(Path::new("/nonexistent/registry.json")).expect("load");
        assert!(registry.iter().next().is_none());
    }

    #[tokio::test]
    async fn persist_writes_and_reloads() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sub/registry.json");
        let mut registry = Registry::default();
        registry.insert(
            "s".to_string(),
            RegistryEntry {
                agent: "fake".to_string(),
                cwd: PathBuf::from("/tmp"),
                created: "t".to_string(),
                last_turn: Some("t2".to_string()),
                turns: 1,
            },
        );
        persist(&path, registry.snapshot().expect("snapshot"))
            .await
            .expect("persist");
        let loaded = Registry::load(&path).expect("reload");
        assert_eq!(loaded.get("s").expect("entry").turns, 1);
    }
}
