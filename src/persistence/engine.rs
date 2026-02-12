//! Storage engine: combines WAL + snapshots for crash-safe persistence.

use crate::distance::DistanceMetric;
use crate::error::Result;
use crate::flat_index::FlatIndex;
use crate::persistence::serialization::{DatabaseSnapshot, SerializedVector};
use crate::persistence::snapshot::SnapshotManager;
use crate::persistence::wal::{WalEntry, WriteAheadLog};
use crate::storage::{Metadata, VectorStore};
use crate::vector::Vector;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Configuration for the storage engine.
pub struct EngineConfig {
    /// Checkpoint after this many WAL entries.
    pub checkpoint_interval: usize,
    /// Distance metric.
    pub metric: DistanceMetric,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: 1000,
            metric: DistanceMetric::Euclidean,
        }
    }
}

/// Persistent storage engine wrapping a VectorStore with WAL + snapshot.
pub struct StorageEngine {
    store: VectorStore<FlatIndex>,
    wal: WriteAheadLog,
    snapshot_mgr: SnapshotManager,
    #[allow(dead_code)]
    data_dir: PathBuf,
    wal_count: usize,
    config: EngineConfig,
}

impl StorageEngine {
    /// Open or create a persistent database at the given directory.
    pub fn open(data_dir: impl AsRef<Path>, config: EngineConfig) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;

        let snapshot_mgr = SnapshotManager::new(&data_dir)?;
        let wal = WriteAheadLog::open(data_dir.join("wal.log"))?;
        let mut store = VectorStore::with_flat_index(config.metric);

        // Load snapshot if available
        if let Some(snapshot) = snapshot_mgr.load()? {
            Self::apply_snapshot(&mut store, &snapshot)?;
        }

        // Replay WAL on top of snapshot
        let entries = wal.replay()?;
        for entry in &entries {
            Self::apply_wal_entry(&mut store, entry)?;
        }

        let wal_count = entries.len();

        Ok(Self {
            store,
            wal,
            snapshot_mgr,
            data_dir,
            wal_count,
            config,
        })
    }

    /// Apply a snapshot to restore store state.
    fn apply_snapshot(
        store: &mut VectorStore<FlatIndex>,
        snapshot: &DatabaseSnapshot,
    ) -> Result<()> {
        for sv in &snapshot.vectors {
            if !sv.data.is_empty() {
                let vector = Vector::new(sv.data.clone());
                store.insert(&sv.string_id, vector)?;
            }
        }
        Ok(())
    }

    /// Apply a single WAL entry to the store.
    fn apply_wal_entry(store: &mut VectorStore<FlatIndex>, entry: &WalEntry) -> Result<()> {
        match entry {
            WalEntry::Insert {
                string_id, data, ..
            } => {
                let vector = Vector::new(data.clone());
                store.insert(string_id.as_str(), vector)?;
            }
            WalEntry::Delete { string_id } => {
                let _ = store.delete(string_id);
            }
            WalEntry::Checkpoint => {}
        }
        Ok(())
    }

    /// Insert a vector, writing to WAL first.
    pub fn insert(&mut self, id: impl Into<String>, vector: Vector) -> Result<()> {
        let id = id.into();
        let data = vector.as_slice().to_vec();

        // WAL first
        self.wal.append(&WalEntry::Insert {
            string_id: id.clone(),
            internal_id: 0,
            data,
        })?;

        // Then apply
        self.store.insert(&id, vector)?;
        self.wal_count += 1;
        self.maybe_checkpoint()?;

        Ok(())
    }

    /// Insert a vector with metadata.
    pub fn insert_with_metadata(
        &mut self,
        id: impl Into<String>,
        vector: Vector,
        metadata: Metadata,
    ) -> Result<()> {
        let id = id.into();
        let data = vector.as_slice().to_vec();

        self.wal.append(&WalEntry::Insert {
            string_id: id.clone(),
            internal_id: 0,
            data,
        })?;

        self.store.insert_with_metadata(&id, vector, metadata)?;
        self.wal_count += 1;
        self.maybe_checkpoint()?;

        Ok(())
    }

    /// Delete a vector, writing to WAL first.
    pub fn delete(&mut self, id: &str) -> Result<Vector> {
        self.wal.append(&WalEntry::Delete {
            string_id: id.to_string(),
        })?;

        let result = self.store.delete(id)?;
        self.wal_count += 1;
        self.maybe_checkpoint()?;

        Ok(result)
    }

    /// Search for the k nearest neighbors.
    pub fn search(
        &self,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<crate::storage::SearchResult>> {
        self.store.search(query, k)
    }

    /// Get the number of vectors.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// List all vector IDs.
    pub fn list_ids(&self) -> Vec<String> {
        self.store.list_ids()
    }

    /// Force a checkpoint: snapshot + truncate WAL.
    pub fn checkpoint(&mut self) -> Result<()> {
        let snapshot = self.build_snapshot();
        self.snapshot_mgr.save(&snapshot)?;

        self.wal.append(&WalEntry::Checkpoint)?;
        self.wal.truncate()?;
        self.wal_count = 0;

        Ok(())
    }

    /// Check if we should checkpoint based on WAL size.
    fn maybe_checkpoint(&mut self) -> Result<()> {
        if self.wal_count >= self.config.checkpoint_interval {
            self.checkpoint()?;
        }
        Ok(())
    }

    /// Build a snapshot from current store state, including actual vector data.
    fn build_snapshot(&self) -> DatabaseSnapshot {
        let id_map = self.store.internal_to_string_ids();
        let index = self.store.index();

        let vectors: Vec<SerializedVector> = index
            .iter()
            .filter_map(|(&internal_id, vector)| {
                id_map.get(&internal_id).map(|string_id| SerializedVector {
                    internal_id,
                    string_id: string_id.clone(),
                    data: vector.as_slice().to_vec(),
                })
            })
            .collect();

        DatabaseSnapshot {
            vectors,
            metadata: HashMap::new(),
            next_id: self.store.len(),
            dimension: self.store.dimension(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_engine_insert_and_search() {
        let dir = TempDir::new().unwrap();
        let config = EngineConfig {
            checkpoint_interval: 100,
            metric: DistanceMetric::Euclidean,
        };
        let mut engine = StorageEngine::open(dir.path().join("db"), config).unwrap();

        engine
            .insert("v1", Vector::new(vec![1.0, 0.0, 0.0]))
            .unwrap();
        engine
            .insert("v2", Vector::new(vec![0.0, 1.0, 0.0]))
            .unwrap();

        let results = engine
            .search(&Vector::new(vec![1.0, 0.0, 0.0]), 1)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_engine_wal_recovery() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("db");

        // Insert data
        {
            let config = EngineConfig {
                checkpoint_interval: 10000,
                metric: DistanceMetric::Euclidean,
            };
            let mut engine = StorageEngine::open(&db_path, config).unwrap();
            engine
                .insert("v1", Vector::new(vec![1.0, 2.0, 3.0]))
                .unwrap();
            engine
                .insert("v2", Vector::new(vec![4.0, 5.0, 6.0]))
                .unwrap();
            engine
                .insert("v3", Vector::new(vec![7.0, 8.0, 9.0]))
                .unwrap();
            assert_eq!(engine.len(), 3);
        }

        // Reopen — should recover from WAL
        {
            let config = EngineConfig {
                checkpoint_interval: 10000,
                metric: DistanceMetric::Euclidean,
            };
            let engine = StorageEngine::open(&db_path, config).unwrap();
            assert_eq!(engine.len(), 3);
        }
    }

    #[test]
    fn test_engine_checkpoint_and_recovery() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("db");

        // Insert and checkpoint
        {
            let config = EngineConfig {
                checkpoint_interval: 2,
                metric: DistanceMetric::Euclidean,
            };
            let mut engine = StorageEngine::open(&db_path, config).unwrap();
            engine
                .insert("v1", Vector::new(vec![1.0, 0.0]))
                .unwrap();
            engine
                .insert("v2", Vector::new(vec![0.0, 1.0]))
                .unwrap();
            // After 2 inserts, checkpoint should have happened
            engine
                .insert("v3", Vector::new(vec![1.0, 1.0]))
                .unwrap();
            assert_eq!(engine.len(), 3);
        }

        // Reopen — should recover from snapshot + WAL
        {
            let config = EngineConfig {
                checkpoint_interval: 10000,
                metric: DistanceMetric::Euclidean,
            };
            let engine = StorageEngine::open(&db_path, config).unwrap();
            assert_eq!(engine.len(), 3);
        }
    }

    #[test]
    fn test_engine_delete_and_recovery() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("db");

        {
            let config = EngineConfig {
                checkpoint_interval: 10000,
                metric: DistanceMetric::Euclidean,
            };
            let mut engine = StorageEngine::open(&db_path, config).unwrap();
            engine
                .insert("v1", Vector::new(vec![1.0, 0.0]))
                .unwrap();
            engine
                .insert("v2", Vector::new(vec![0.0, 1.0]))
                .unwrap();
            engine.delete("v1").unwrap();
            assert_eq!(engine.len(), 1);
        }

        {
            let config = EngineConfig {
                checkpoint_interval: 10000,
                metric: DistanceMetric::Euclidean,
            };
            let engine = StorageEngine::open(&db_path, config).unwrap();
            assert_eq!(engine.len(), 1);
        }
    }

    #[test]
    fn test_engine_1000_vectors_recovery() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("db");

        {
            let config = EngineConfig {
                checkpoint_interval: 500,
                metric: DistanceMetric::Euclidean,
            };
            let mut engine = StorageEngine::open(&db_path, config).unwrap();
            for i in 0..1000 {
                engine
                    .insert(
                        format!("v{}", i),
                        Vector::new(vec![i as f32, (i * 2) as f32]),
                    )
                    .unwrap();
            }
            assert_eq!(engine.len(), 1000);
        }

        {
            let config = EngineConfig {
                checkpoint_interval: 10000,
                metric: DistanceMetric::Euclidean,
            };
            let engine = StorageEngine::open(&db_path, config).unwrap();
            assert_eq!(engine.len(), 1000);
        }
    }
}
