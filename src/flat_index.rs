//! Brute-force flat index — O(n) k-NN search

use std::collections::HashMap;

use crate::distance::DistanceMetric;
use crate::error::Result;
use crate::index::Index;
use crate::vector::Vector;

/// A flat (brute-force) index that computes distance to every stored vector.
#[derive(Debug)]
pub struct FlatIndex {
    vectors: HashMap<usize, Vector>,
    metric: DistanceMetric,
}

impl FlatIndex {
    /// Create a new empty flat index with the given distance metric.
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            vectors: HashMap::new(),
            metric,
        }
    }

    /// Get a vector by internal ID.
    pub fn get_vector(&self, id: usize) -> Option<&Vector> {
        self.vectors.get(&id)
    }

    /// Iterate over all (id, vector) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &Vector)> {
        self.vectors.iter()
    }
}

impl Index for FlatIndex {
    fn add(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.vectors.insert(id, vector);
        Ok(())
    }

    fn remove(&mut self, id: usize) -> Result<()> {
        self.vectors.remove(&id);
        Ok(())
    }

    fn get_vector(&self, id: usize) -> Option<&Vector> {
        self.vectors.get(&id)
    }

    fn search(&self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>> {
        let mut results: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .map(|(&id, vec)| {
                let distance = self.metric.distance(query, vec)?;
                Ok((id, distance))
            })
            .collect::<Result<Vec<_>>>()?;

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        Ok(results)
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_index_basic() {
        let mut index = FlatIndex::new(DistanceMetric::Euclidean);
        index.add(0, Vector::new(vec![1.0, 0.0, 0.0])).unwrap();
        index.add(1, Vector::new(vec![0.0, 1.0, 0.0])).unwrap();
        index.add(2, Vector::new(vec![1.0, 1.0, 0.0])).unwrap();

        let query = Vector::new(vec![1.0, 0.0, 0.0]);
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // exact match
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn test_flat_index_get_vector() {
        let mut index = FlatIndex::new(DistanceMetric::Euclidean);
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        index.add(0, v.clone()).unwrap();

        assert_eq!(index.get_vector(0), Some(&v));
        assert_eq!(index.get_vector(99), None);
    }

    #[test]
    fn test_flat_index_remove() {
        let mut index = FlatIndex::new(DistanceMetric::Euclidean);
        index.add(0, Vector::new(vec![1.0, 0.0])).unwrap();
        index.add(1, Vector::new(vec![0.0, 1.0])).unwrap();
        assert_eq!(index.len(), 2);

        index.remove(0).unwrap();
        assert_eq!(index.len(), 1);
    }
}
