//! Snapshot: save/load full database state to/from disk.

use crate::error::{Result, VectorDbError};
use crate::persistence::serialization::{self, DatabaseSnapshot};
use std::fs;
use std::path::{Path, PathBuf};

/// Manages saving and loading database snapshots.
pub struct SnapshotManager {
    dir: PathBuf,
}

impl SnapshotManager {
    /// Create a snapshot manager for the given directory.
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    fn snapshot_path(&self) -> PathBuf {
        self.dir.join("snapshot.bin")
    }

    fn manifest_path(&self) -> PathBuf {
        self.dir.join("manifest.json")
    }

    /// Save a database snapshot to disk.
    pub fn save(&self, snapshot: &DatabaseSnapshot) -> Result<()> {
        // Write snapshot data (bincode)
        let data = serialization::to_bincode(snapshot)?;
        fs::write(self.snapshot_path(), &data)?;

        // Write manifest (JSON) for human-readable metadata
        let manifest = serde_json::json!({
            "vector_count": snapshot.vectors.len(),
            "next_id": snapshot.next_id,
            "dimension": snapshot.dimension,
        });
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| VectorDbError::SerializationError(e.to_string()))?;
        fs::write(self.manifest_path(), &manifest_bytes)?;

        Ok(())
    }

    /// Load a database snapshot from disk, or return None if no snapshot exists.
    pub fn load(&self) -> Result<Option<DatabaseSnapshot>> {
        let path = self.snapshot_path();
        if !path.exists() {
            return Ok(None);
        }

        let data = fs::read(&path)?;
        let snapshot: DatabaseSnapshot = serialization::from_bincode(&data)?;
        Ok(Some(snapshot))
    }

    /// Check if a snapshot exists.
    pub fn exists(&self) -> bool {
        self.snapshot_path().exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::serialization::SerializedVector;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_save_and_load() {
        let dir = TempDir::new().unwrap();
        let mgr = SnapshotManager::new(dir.path().join("db")).unwrap();

        let snapshot = DatabaseSnapshot {
            vectors: vec![
                SerializedVector {
                    internal_id: 0,
                    string_id: "v1".to_string(),
                    data: vec![1.0, 2.0, 3.0],
                },
                SerializedVector {
                    internal_id: 1,
                    string_id: "v2".to_string(),
                    data: vec![4.0, 5.0, 6.0],
                },
            ],
            metadata: HashMap::new(),
            next_id: 2,
            dimension: Some(3),
        };

        mgr.save(&snapshot).unwrap();
        assert!(mgr.exists());

        let loaded = mgr.load().unwrap().unwrap();
        assert_eq!(loaded.vectors.len(), 2);
        assert_eq!(loaded.next_id, 2);
        assert_eq!(loaded.dimension, Some(3));
        assert_eq!(loaded.vectors[0].string_id, "v1");
        assert_eq!(loaded.vectors[1].data, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_load_nonexistent() {
        let dir = TempDir::new().unwrap();
        let mgr = SnapshotManager::new(dir.path().join("empty")).unwrap();
        assert!(!mgr.exists());
        assert!(mgr.load().unwrap().is_none());
    }
}
