//! HNSW (Hierarchical Navigable Small World) index module.

pub mod graph;
pub mod neighbor_queue;

pub use graph::{HnswGraph, HnswParams};

use crate::distance::DistanceMetric;
use crate::error::Result;
use crate::index::Index;
use crate::vector::Vector;

/// An HNSW-based approximate nearest neighbor index.
#[derive(Debug)]
pub struct HnswIndex {
    graph: HnswGraph,
}

impl HnswIndex {
    /// Create a new HNSW index with the given metric and default parameters.
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            graph: HnswGraph::new(metric, HnswParams::default()),
        }
    }

    /// Create a new HNSW index with custom parameters.
    pub fn with_params(metric: DistanceMetric, params: HnswParams) -> Self {
        Self {
            graph: HnswGraph::new(metric, params),
        }
    }

    /// Build the index from a batch of vectors (parallel distance computation with rayon).
    /// Vectors are inserted sequentially into the graph, but distance computations
    /// during search_layer use rayon for parallelism on large neighbor lists.
    pub fn build_batch(&mut self, vectors: Vec<(usize, Vector)>) -> Result<()> {
        for (id, vector) in vectors {
            self.graph.insert(id, vector)?;
        }
        Ok(())
    }

    /// Search with a specific ef value for runtime tuning.
    pub fn search_with_ef(
        &self,
        query: &Vector,
        k: usize,
        ef: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let results = self.graph.search_with_ef(query, k, ef)?;
        Ok(results.into_iter().map(|n| (n.id, n.distance)).collect())
    }
}

impl Index for HnswIndex {
    fn add(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.graph.insert(id, vector)
    }

    fn remove(&mut self, id: usize) -> Result<()> {
        self.graph.remove(id)
    }

    fn get_vector(&self, id: usize) -> Option<&Vector> {
        self.graph.get_vector(id)
    }

    fn search(&self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>> {
        let results = self.graph.search_knn(query, k, 50)?; // default ef_search=50
        Ok(results.into_iter().map(|n| (n.id, n.distance)).collect())
    }

    fn metric(&self) -> DistanceMetric {
        self.graph.metric()
    }

    fn len(&self) -> usize {
        self.graph.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::VectorStore;

    #[test]
    fn test_hnsw_index_via_trait() {
        let mut index = HnswIndex::new(DistanceMetric::Euclidean);
        index.add(0, Vector::new(vec![1.0, 0.0, 0.0])).unwrap();
        index.add(1, Vector::new(vec![0.0, 1.0, 0.0])).unwrap();
        index.add(2, Vector::new(vec![1.0, 1.0, 0.0])).unwrap();

        let results = index.search(&Vector::new(vec![1.0, 0.0, 0.0]), 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // exact match
        assert!(results[0].1 < 1e-5);
    }

    #[test]
    fn test_hnsw_get_vector() {
        let mut index = HnswIndex::new(DistanceMetric::Euclidean);
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        index.add(0, v.clone()).unwrap();

        assert_eq!(index.get_vector(0), Some(&v));
        assert_eq!(index.get_vector(99), None);
    }

    #[test]
    fn test_hnsw_via_vectorstore() {
        let index = HnswIndex::with_params(
            DistanceMetric::Euclidean,
            HnswParams::new(4, 32, 16),
        );
        let mut store = VectorStore::with_index(index);

        store
            .insert("v1", Vector::new(vec![1.0, 0.0, 0.0]))
            .unwrap();
        store
            .insert("v2", Vector::new(vec![0.0, 1.0, 0.0]))
            .unwrap();
        store
            .insert("v3", Vector::new(vec![0.0, 0.0, 1.0]))
            .unwrap();

        let results = store
            .search(&Vector::new(vec![1.0, 0.1, 0.0]), 2)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_hnsw_delete_via_vectorstore() {
        let index = HnswIndex::with_params(
            DistanceMetric::Euclidean,
            HnswParams::new(4, 32, 16),
        );
        let mut store = VectorStore::with_index(index);

        store
            .insert("v1", Vector::new(vec![1.0, 0.0]))
            .unwrap();
        store
            .insert("v2", Vector::new(vec![0.0, 1.0]))
            .unwrap();
        assert_eq!(store.len(), 2);

        store.delete("v1").unwrap();
        assert_eq!(store.len(), 1);
    }
}
