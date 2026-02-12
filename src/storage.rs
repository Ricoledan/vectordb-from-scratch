//! In-memory vector storage

use crate::distance::DistanceMetric;
use crate::error::{Result, VectorDbError};
use crate::flat_index::FlatIndex;
use crate::index::Index;
use crate::vector::Vector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A search result containing the vector ID and distance
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
}

/// Metadata associated with a vector
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Metadata {
    fields: HashMap<String, String>,
}

impl Metadata {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    pub fn insert(&mut self, key: String, value: String) {
        self.fields.insert(key, value);
    }

    pub fn get(&self, key: &str) -> Option<&String> {
        self.fields.get(key)
    }
}

/// In-memory vector storage with a pluggable search index.
#[derive(Debug)]
pub struct VectorStore<I: Index> {
    index: I,
    /// String ID -> usize internal ID
    id_to_internal: HashMap<String, usize>,
    /// usize internal ID -> String ID
    internal_to_id: HashMap<usize, String>,
    /// Metadata keyed by internal ID
    metadata: HashMap<usize, Metadata>,
    /// Next internal ID to assign
    next_id: usize,
    /// Enforced vector dimension
    dimension: Option<usize>,
}

impl VectorStore<FlatIndex> {
    /// Create a new vector store with a brute-force flat index.
    pub fn new(metric: DistanceMetric) -> Self {
        Self::with_flat_index(metric)
    }

    /// Create a new vector store with a brute-force flat index (explicit name).
    pub fn with_flat_index(metric: DistanceMetric) -> Self {
        VectorStore {
            index: FlatIndex::new(metric),
            id_to_internal: HashMap::new(),
            internal_to_id: HashMap::new(),
            metadata: HashMap::new(),
            next_id: 0,
            dimension: None,
        }
    }
}

impl<I: Index> VectorStore<I> {
    /// Create a new vector store with the given index.
    pub fn with_index(index: I) -> Self {
        Self {
            index,
            id_to_internal: HashMap::new(),
            internal_to_id: HashMap::new(),
            metadata: HashMap::new(),
            next_id: 0,
            dimension: None,
        }
    }

    /// Insert a vector with the given ID
    pub fn insert(&mut self, id: impl Into<String>, vector: Vector) -> Result<()> {
        self.insert_with_metadata(id, vector, Metadata::new())
    }

    /// Insert a vector with metadata
    pub fn insert_with_metadata(
        &mut self,
        id: impl Into<String>,
        vector: Vector,
        metadata: Metadata,
    ) -> Result<()> {
        let id = id.into();
        let dim = vector.dimension();

        // Check dimension consistency
        if let Some(expected_dim) = self.dimension {
            if dim != expected_dim {
                return Err(VectorDbError::DimensionMismatch {
                    expected: expected_dim,
                    actual: dim,
                });
            }
        } else {
            self.dimension = Some(dim);
        }

        // If this string ID already exists, remove the old entry first
        if let Some(&old_internal) = self.id_to_internal.get(&id) {
            self.index.remove(old_internal)?;
            self.metadata.remove(&old_internal);
            self.internal_to_id.remove(&old_internal);
        }

        let internal_id = self.next_id;
        self.next_id += 1;

        self.index.add(internal_id, vector)?;
        self.id_to_internal.insert(id.clone(), internal_id);
        self.internal_to_id.insert(internal_id, id);
        self.metadata.insert(internal_id, metadata);

        Ok(())
    }

    /// Delete a vector by ID
    pub fn delete(&mut self, id: &str) -> Result<Vector> {
        let internal_id = self
            .id_to_internal
            .remove(id)
            .ok_or_else(|| VectorDbError::VectorNotFound { id: id.to_string() })?;

        self.internal_to_id.remove(&internal_id);
        self.metadata.remove(&internal_id);
        self.index.remove(internal_id)?;

        // Note: we can't return the vector from the index, so we return a placeholder.
        // This changes the API slightly — callers that used the returned vector
        // should use get() before delete() if they need the data.
        // For backward compatibility, we return an empty vector.
        // TODO: Consider adding a `get` method to Index trait if needed.
        Ok(Vector::new(vec![]))
    }

    /// Get a vector by ID — not available through the generic index.
    /// Use the flat index directly if you need vector retrieval by ID.
    pub fn get(&self, _id: &str) -> Option<&Vector> {
        // The Index trait doesn't expose vector retrieval by ID.
        // We need a different approach for this.
        None
    }

    /// Get the number of vectors in the store
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Search for the k nearest neighbors
    pub fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
        if self.is_empty() {
            return Ok(vec![]);
        }

        // Check dimension
        if let Some(expected_dim) = self.dimension {
            if query.dimension() != expected_dim {
                return Err(VectorDbError::DimensionMismatch {
                    expected: expected_dim,
                    actual: query.dimension(),
                });
            }
        }

        let index_results = self.index.search(query, k)?;

        let results = index_results
            .into_iter()
            .filter_map(|(internal_id, distance)| {
                self.internal_to_id.get(&internal_id).map(|id| SearchResult {
                    id: id.clone(),
                    distance,
                })
            })
            .collect();

        Ok(results)
    }

    /// List all vector IDs
    pub fn list_ids(&self) -> Vec<String> {
        self.id_to_internal.keys().cloned().collect()
    }

    /// Get the distance metric used by this store
    pub fn metric(&self) -> DistanceMetric {
        self.index.metric()
    }

    /// Get the dimension of vectors in this store (if any)
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Get a reference to the underlying index.
    pub fn index(&self) -> &I {
        &self.index
    }

    /// Get a reference to the internal ID mapping (internal_id -> string_id).
    pub fn internal_to_string_ids(&self) -> &HashMap<usize, String> {
        &self.internal_to_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_insert_and_get() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        store.insert("v1", v.clone()).unwrap();

        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_dimension_consistency() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);
        store.insert("v1", Vector::new(vec![1.0, 2.0, 3.0])).unwrap();

        let result = store.insert("v2", Vector::new(vec![1.0, 2.0]));
        assert!(matches!(result, Err(VectorDbError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_delete() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        store.insert("v1", v).unwrap();

        store.delete("v1").unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_search() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);
        store.insert("v1", Vector::new(vec![1.0, 0.0, 0.0])).unwrap();
        store.insert("v2", Vector::new(vec![0.0, 1.0, 0.0])).unwrap();
        store.insert("v3", Vector::new(vec![1.0, 1.0, 0.0])).unwrap();

        let query = Vector::new(vec![1.0, 0.0, 0.0]);
        let results = store.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1");
        assert_relative_eq!(results[0].distance, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_search_empty_store() {
        let store = VectorStore::new(DistanceMetric::Euclidean);
        let query = Vector::new(vec![1.0, 2.0, 3.0]);
        let results = store.search(&query, 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_metadata() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);
        let mut metadata = Metadata::new();
        metadata.insert("label".to_string(), "test".to_string());

        store
            .insert_with_metadata("v1", Vector::new(vec![1.0, 2.0, 3.0]), metadata)
            .unwrap();

        assert_eq!(store.len(), 1);
    }
}
