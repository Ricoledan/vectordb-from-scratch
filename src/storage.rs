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

    pub fn fields(&self) -> &HashMap<String, String> {
        &self.fields
    }
}

/// A filter for metadata-based search narrowing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum MetadataFilter {
    /// Field equals a specific value.
    Eq { field: String, value: String },
    /// Field does not equal a specific value.
    Ne { field: String, value: String },
    /// Field exists (has any value).
    Exists { field: String },
    /// All sub-filters must match.
    And { filters: Vec<MetadataFilter> },
    /// At least one sub-filter must match.
    Or { filters: Vec<MetadataFilter> },
}

impl MetadataFilter {
    /// Returns true if the given metadata satisfies this filter.
    pub fn matches(&self, metadata: &Metadata) -> bool {
        match self {
            MetadataFilter::Eq { field, value } => metadata.get(field) == Some(value),
            MetadataFilter::Ne { field, value } => metadata.get(field) != Some(value),
            MetadataFilter::Exists { field } => metadata.get(field).is_some(),
            MetadataFilter::And { filters } => filters.iter().all(|f| f.matches(metadata)),
            MetadataFilter::Or { filters } => filters.iter().any(|f| f.matches(metadata)),
        }
    }
}

/// An item for batch insertion.
#[derive(Debug, Clone)]
pub struct BatchInsertItem {
    pub id: String,
    pub vector: Vector,
    pub metadata: Metadata,
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

    /// Delete a vector by ID, returning the vector data.
    pub fn delete(&mut self, id: &str) -> Result<Vector> {
        let internal_id = self
            .id_to_internal
            .remove(id)
            .ok_or_else(|| VectorDbError::VectorNotFound { id: id.to_string() })?;

        let vector = self
            .index
            .get_vector(internal_id)
            .cloned()
            .unwrap_or_else(|| Vector::new(vec![]));

        self.internal_to_id.remove(&internal_id);
        self.metadata.remove(&internal_id);
        self.index.remove(internal_id)?;

        Ok(vector)
    }

    /// Get a vector by ID.
    pub fn get(&self, id: &str) -> Option<&Vector> {
        let &internal_id = self.id_to_internal.get(id)?;
        self.index.get_vector(internal_id)
    }

    /// Get metadata for a vector by ID.
    pub fn get_metadata(&self, id: &str) -> Option<&Metadata> {
        let &internal_id = self.id_to_internal.get(id)?;
        self.metadata.get(&internal_id)
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

    /// Search for the k nearest neighbors that match the given metadata filter.
    /// Uses post-filtering with 3x over-fetch to compensate for filtered-out results.
    pub fn search_with_filter(
        &self,
        query: &Vector,
        k: usize,
        filter: &MetadataFilter,
    ) -> Result<Vec<SearchResult>> {
        if self.is_empty() {
            return Ok(vec![]);
        }

        if let Some(expected_dim) = self.dimension {
            if query.dimension() != expected_dim {
                return Err(VectorDbError::DimensionMismatch {
                    expected: expected_dim,
                    actual: query.dimension(),
                });
            }
        }

        // Over-fetch 3x to compensate for filtered-out results
        let fetch_k = (k * 3).max(k).min(self.len());
        let index_results = self.index.search(query, fetch_k)?;

        let results: Vec<SearchResult> = index_results
            .into_iter()
            .filter_map(|(internal_id, distance)| {
                let string_id = self.internal_to_id.get(&internal_id)?;
                let meta = self.metadata.get(&internal_id)?;
                if filter.matches(meta) {
                    Some(SearchResult {
                        id: string_id.clone(),
                        distance,
                    })
                } else {
                    None
                }
            })
            .take(k)
            .collect();

        Ok(results)
    }

    /// Insert a batch of vectors. Stops at the first error and returns it.
    pub fn insert_batch(&mut self, items: Vec<BatchInsertItem>) -> Result<()> {
        for item in items {
            self.insert_with_metadata(item.id, item.vector, item.metadata)?;
        }
        Ok(())
    }

    /// Search for k nearest neighbors for multiple queries at once.
    /// Returns one result set per query.
    pub fn search_batch(
        &self,
        queries: &[(Vector, usize)],
    ) -> Result<Vec<Vec<SearchResult>>> {
        queries
            .iter()
            .map(|(query, k)| self.search(query, *k))
            .collect()
    }

    /// Search for k nearest neighbors with a metadata filter for multiple queries.
    pub fn search_batch_with_filter(
        &self,
        queries: &[(Vector, usize)],
        filter: &MetadataFilter,
    ) -> Result<Vec<Vec<SearchResult>>> {
        queries
            .iter()
            .map(|(query, k)| self.search_with_filter(query, *k, filter))
            .collect()
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
    fn test_get_returns_vector() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        store.insert("v1", v.clone()).unwrap();

        assert_eq!(store.get("v1"), Some(&v));
        assert_eq!(store.get("nonexistent"), None);
    }

    #[test]
    fn test_delete_returns_vector() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        store.insert("v1", v.clone()).unwrap();

        let deleted = store.delete("v1").unwrap();
        assert_eq!(deleted, v);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_get_metadata() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);
        let mut meta = Metadata::new();
        meta.insert("color".to_string(), "red".to_string());
        store
            .insert_with_metadata("v1", Vector::new(vec![1.0, 2.0, 3.0]), meta)
            .unwrap();

        let m = store.get_metadata("v1").unwrap();
        assert_eq!(m.get("color"), Some(&"red".to_string()));
        assert!(store.get_metadata("nonexistent").is_none());
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

    // --- MetadataFilter tests ---

    #[test]
    fn test_filter_eq() {
        let mut meta = Metadata::new();
        meta.insert("color".to_string(), "red".to_string());

        let f = MetadataFilter::Eq {
            field: "color".to_string(),
            value: "red".to_string(),
        };
        assert!(f.matches(&meta));

        let f2 = MetadataFilter::Eq {
            field: "color".to_string(),
            value: "blue".to_string(),
        };
        assert!(!f2.matches(&meta));
    }

    #[test]
    fn test_filter_ne() {
        let mut meta = Metadata::new();
        meta.insert("color".to_string(), "red".to_string());

        let f = MetadataFilter::Ne {
            field: "color".to_string(),
            value: "blue".to_string(),
        };
        assert!(f.matches(&meta));

        let f2 = MetadataFilter::Ne {
            field: "color".to_string(),
            value: "red".to_string(),
        };
        assert!(!f2.matches(&meta));
    }

    #[test]
    fn test_filter_exists() {
        let mut meta = Metadata::new();
        meta.insert("color".to_string(), "red".to_string());

        let f = MetadataFilter::Exists {
            field: "color".to_string(),
        };
        assert!(f.matches(&meta));

        let f2 = MetadataFilter::Exists {
            field: "size".to_string(),
        };
        assert!(!f2.matches(&meta));
    }

    #[test]
    fn test_filter_and() {
        let mut meta = Metadata::new();
        meta.insert("color".to_string(), "red".to_string());
        meta.insert("size".to_string(), "large".to_string());

        let f = MetadataFilter::And {
            filters: vec![
                MetadataFilter::Eq {
                    field: "color".to_string(),
                    value: "red".to_string(),
                },
                MetadataFilter::Eq {
                    field: "size".to_string(),
                    value: "large".to_string(),
                },
            ],
        };
        assert!(f.matches(&meta));

        let f2 = MetadataFilter::And {
            filters: vec![
                MetadataFilter::Eq {
                    field: "color".to_string(),
                    value: "red".to_string(),
                },
                MetadataFilter::Eq {
                    field: "size".to_string(),
                    value: "small".to_string(),
                },
            ],
        };
        assert!(!f2.matches(&meta));
    }

    #[test]
    fn test_filter_or() {
        let mut meta = Metadata::new();
        meta.insert("color".to_string(), "red".to_string());

        let f = MetadataFilter::Or {
            filters: vec![
                MetadataFilter::Eq {
                    field: "color".to_string(),
                    value: "red".to_string(),
                },
                MetadataFilter::Eq {
                    field: "color".to_string(),
                    value: "blue".to_string(),
                },
            ],
        };
        assert!(f.matches(&meta));

        let f2 = MetadataFilter::Or {
            filters: vec![
                MetadataFilter::Eq {
                    field: "color".to_string(),
                    value: "green".to_string(),
                },
                MetadataFilter::Eq {
                    field: "color".to_string(),
                    value: "blue".to_string(),
                },
            ],
        };
        assert!(!f2.matches(&meta));
    }

    #[test]
    fn test_search_with_filter_matching() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);

        let mut m1 = Metadata::new();
        m1.insert("color".to_string(), "red".to_string());
        store
            .insert_with_metadata("v1", Vector::new(vec![1.0, 0.0, 0.0]), m1)
            .unwrap();

        let mut m2 = Metadata::new();
        m2.insert("color".to_string(), "blue".to_string());
        store
            .insert_with_metadata("v2", Vector::new(vec![0.9, 0.1, 0.0]), m2)
            .unwrap();

        let mut m3 = Metadata::new();
        m3.insert("color".to_string(), "red".to_string());
        store
            .insert_with_metadata("v3", Vector::new(vec![0.0, 1.0, 0.0]), m3)
            .unwrap();

        let query = Vector::new(vec![1.0, 0.0, 0.0]);
        let filter = MetadataFilter::Eq {
            field: "color".to_string(),
            value: "red".to_string(),
        };
        let results = store.search_with_filter(&query, 10, &filter).unwrap();

        assert_eq!(results.len(), 2);
        // All results should have color=red
        for r in &results {
            assert!(r.id == "v1" || r.id == "v3");
        }
    }

    #[test]
    fn test_search_with_filter_none_matching() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);

        let mut m1 = Metadata::new();
        m1.insert("color".to_string(), "red".to_string());
        store
            .insert_with_metadata("v1", Vector::new(vec![1.0, 0.0, 0.0]), m1)
            .unwrap();

        let query = Vector::new(vec![1.0, 0.0, 0.0]);
        let filter = MetadataFilter::Eq {
            field: "color".to_string(),
            value: "green".to_string(),
        };
        let results = store.search_with_filter(&query, 10, &filter).unwrap();
        assert!(results.is_empty());
    }

    // --- Batch operation tests ---

    #[test]
    fn test_batch_insert() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);
        let items = vec![
            BatchInsertItem {
                id: "v1".to_string(),
                vector: Vector::new(vec![1.0, 0.0, 0.0]),
                metadata: Metadata::new(),
            },
            BatchInsertItem {
                id: "v2".to_string(),
                vector: Vector::new(vec![0.0, 1.0, 0.0]),
                metadata: Metadata::new(),
            },
        ];
        store.insert_batch(items).unwrap();
        assert_eq!(store.len(), 2);
        assert!(store.get("v1").is_some());
        assert!(store.get("v2").is_some());
    }

    #[test]
    fn test_batch_insert_dim_mismatch() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);
        let items = vec![
            BatchInsertItem {
                id: "v1".to_string(),
                vector: Vector::new(vec![1.0, 0.0, 0.0]),
                metadata: Metadata::new(),
            },
            BatchInsertItem {
                id: "v2".to_string(),
                vector: Vector::new(vec![0.0, 1.0]), // wrong dimension
                metadata: Metadata::new(),
            },
        ];
        let result = store.insert_batch(items);
        assert!(matches!(
            result,
            Err(VectorDbError::DimensionMismatch { .. })
        ));
        // First vector should still have been inserted
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_batch_search() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);
        store
            .insert("v1", Vector::new(vec![1.0, 0.0, 0.0]))
            .unwrap();
        store
            .insert("v2", Vector::new(vec![0.0, 1.0, 0.0]))
            .unwrap();

        let queries = vec![
            (Vector::new(vec![1.0, 0.0, 0.0]), 1),
            (Vector::new(vec![0.0, 1.0, 0.0]), 1),
        ];
        let results = store.search_batch(&queries).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0].id, "v1");
        assert_eq!(results[1][0].id, "v2");
    }

    #[test]
    fn test_batch_search_with_filter() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);

        let mut m1 = Metadata::new();
        m1.insert("color".to_string(), "red".to_string());
        store
            .insert_with_metadata("v1", Vector::new(vec![1.0, 0.0, 0.0]), m1)
            .unwrap();

        let mut m2 = Metadata::new();
        m2.insert("color".to_string(), "blue".to_string());
        store
            .insert_with_metadata("v2", Vector::new(vec![0.0, 1.0, 0.0]), m2)
            .unwrap();

        let queries = vec![
            (Vector::new(vec![1.0, 0.0, 0.0]), 10),
            (Vector::new(vec![0.0, 1.0, 0.0]), 10),
        ];
        let filter = MetadataFilter::Eq {
            field: "color".to_string(),
            value: "red".to_string(),
        };
        let results = store.search_batch_with_filter(&queries, &filter).unwrap();
        assert_eq!(results.len(), 2);
        // Both queries should only return v1 (the red one)
        assert_eq!(results[0].len(), 1);
        assert_eq!(results[0][0].id, "v1");
        assert_eq!(results[1].len(), 1);
        assert_eq!(results[1][0].id, "v1");
    }

    #[test]
    fn test_search_with_filter_all_matching() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);

        let mut m1 = Metadata::new();
        m1.insert("color".to_string(), "red".to_string());
        store
            .insert_with_metadata("v1", Vector::new(vec![1.0, 0.0, 0.0]), m1)
            .unwrap();

        let mut m2 = Metadata::new();
        m2.insert("color".to_string(), "red".to_string());
        store
            .insert_with_metadata("v2", Vector::new(vec![0.0, 1.0, 0.0]), m2)
            .unwrap();

        let query = Vector::new(vec![1.0, 0.0, 0.0]);
        let filter = MetadataFilter::Eq {
            field: "color".to_string(),
            value: "red".to_string(),
        };
        let results = store.search_with_filter(&query, 10, &filter).unwrap();
        assert_eq!(results.len(), 2);
    }
}
