//! The subagent registry: a JSON file mapping names to sessions.
//!
//! The registry survives server restarts; a name registered to a session can
//! be resumed with `session/load` when the agent supports it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Counter making each registry temp file unique within a process.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// One registered subagent: everything needed to resume its session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    /// Agent config key that owns the session.
    pub agent: String,
    /// ACP session id returned by `session/new`.
    pub session_id: String,
    /// Session working directory.
    pub cwd: PathBuf,
    /// RFC 3339 timestamp of when the name was first registered.
    pub created: String,
    /// RFC 3339 timestamp of the last completed turn, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_turn: Option<String>,
    /// Number of completed turns.
    #[serde(default)]
    pub turns: u64,
}

/// The name → entry map backed by `registry.json`.
#[derive(Debug, Default)]
pub struct Registry {
    entries: BTreeMap<String, RegistryEntry>,
}

impl Registry {
    /// Load the registry file; a missing file yields an empty registry.
    ///
    /// A file that exists but does not parse is an error naming the path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RegistryParse`] or an I/O error.
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let entries =
                    serde_json::from_str(&text).map_err(|source| Error::RegistryParse {
                        path: path.to_path_buf(),
                        source,
                    })?;
                Ok(Self { entries })
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(Error::io(format!("cannot read {}", path.display()), source)),
        }
    }

    /// Look up a registered entry.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&RegistryEntry> {
        self.entries.get(name)
    }

    /// Look up a registered entry mutably.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut RegistryEntry> {
        self.entries.get_mut(name)
    }

    /// Insert or replace an entry.
    pub fn insert(&mut self, name: String, entry: RegistryEntry) {
        self.entries.insert(name, entry);
    }

    /// Remove an entry, returning whether one existed.
    pub fn remove(&mut self, name: &str) -> bool {
        self.entries.remove(name).is_some()
    }

    /// Iterate over entries by name.
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
            "dev".to_string(),
            RegistryEntry {
                agent: "devin".to_string(),
                session_id: "sess-1".to_string(),
                cwd: PathBuf::from("/tmp"),
                created: "2026-09-10T00:00:00Z".to_string(),
                last_turn: None,
                turns: 2,
            },
        );
        let snapshot = registry.snapshot().expect("snapshot");
        let loaded: BTreeMap<String, RegistryEntry> =
            serde_json::from_str(&snapshot).expect("parse snapshot");
        assert_eq!(loaded["dev"].session_id, "sess-1");
        assert_eq!(loaded["dev"].turns, 2);
        assert_eq!(loaded["dev"].agent, "devin");
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
            "a".to_string(),
            RegistryEntry {
                agent: "fake".to_string(),
                session_id: "s".to_string(),
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
        assert_eq!(loaded.get("a").expect("entry").turns, 1);
    }
}
