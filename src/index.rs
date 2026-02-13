//! Index trait for pluggable search backends

use crate::distance::DistanceMetric;
use crate::error::Result;
use crate::vector::Vector;

/// A search index that supports insertion, removal, and k-NN search.
///
/// Implementations use `usize` internal IDs for cache efficiency;
/// the `VectorStore` handles String-to-usize mapping.
pub trait Index {
    /// Add a vector with the given internal ID.
    fn add(&mut self, id: usize, vector: Vector) -> Result<()>;

    /// Remove the vector with the given internal ID.
    fn remove(&mut self, id: usize) -> Result<()>;

    /// Search for the `k` nearest neighbors of `query`.
    /// Returns a Vec of `(id, distance)` pairs sorted by distance ascending.
    fn search(&self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>>;

    /// Retrieve a vector by its internal ID.
    fn get_vector(&self, id: usize) -> Option<&Vector>;

    /// The distance metric used by this index.
    fn metric(&self) -> DistanceMetric;

    /// The number of vectors in this index.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
