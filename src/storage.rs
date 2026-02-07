//! In-memory vector storage

use crate::distance::DistanceMetric;
use crate::error::{Result, VectorDbError};
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

/// A stored vector with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredVector {
    vector: Vector,
    metadata: Metadata,
}

/// In-memory vector storage
#[derive(Debug)]
pub struct VectorStore {
    vectors: HashMap<String, StoredVector>,
    metric: DistanceMetric,
    dimension: Option<usize>,
}

impl VectorStore {
    /// Create a new empty vector store with the specified distance metric
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            vectors: HashMap::new(),
            metric,
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

        self.vectors.insert(id, StoredVector { vector, metadata });
        Ok(())
    }

    /// Delete a vector by ID
    pub fn delete(&mut self, id: &str) -> Result<Vector> {
        self.vectors
            .remove(id)
            .map(|stored| stored.vector)
            .ok_or_else(|| VectorDbError::VectorNotFound { id: id.to_string() })
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Option<&Vector> {
        self.vectors.get(id).map(|stored| &stored.vector)
    }

    /// Get the number of vectors in the store
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Search for the k nearest neighbors (brute force)
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

        let mut results: Vec<SearchResult> = self
            .vectors
            .iter()
            .map(|(id, stored)| {
                let distance = self.metric.distance(query, &stored.vector)?;
                Ok(SearchResult {
                    id: id.clone(),
                    distance,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Sort by distance (ascending)
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        // Return top k results
        results.truncate(k);
        Ok(results)
    }

    /// List all vector IDs
    pub fn list_ids(&self) -> Vec<String> {
        self.vectors.keys().cloned().collect()
    }

    /// Get the distance metric used by this store
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Get the dimension of vectors in this store (if any)
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
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

        let retrieved = store.get("v1").unwrap();
        assert_eq!(retrieved.as_slice(), v.as_slice());
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

        let deleted = store.delete("v1").unwrap();
        assert_eq!(deleted.as_slice(), &[1.0, 2.0, 3.0]);
        assert!(store.get("v1").is_none());
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
