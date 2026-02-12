//! # VectorDB From Scratch
//!
//! A vector database implementation in Rust for learning purposes.
//!
//! This library provides:
//! - Vector storage and management
//! - Distance metrics (Euclidean, Cosine, Dot Product)
//! - Pluggable search indexes (FlatIndex, HNSW)
//! - Persistence layer
//!
//! ## Example
//!
//! ```rust
//! use vectordb_from_scratch::vector::Vector;
//! use vectordb_from_scratch::storage::VectorStore;
//! use vectordb_from_scratch::distance::DistanceMetric;
//!
//! // Create a vector store with brute-force flat index
//! let mut store = VectorStore::new(DistanceMetric::Euclidean);
//!
//! // Insert vectors
//! let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
//! store.insert("v1", v1);
//!
//! // Search for similar vectors
//! let query = Vector::new(vec![1.1, 2.1, 3.1]);
//! let results = store.search(&query, 5);
//! ```

pub mod vector;
pub mod storage;
pub mod distance;
pub mod error;
pub mod index;
pub mod flat_index;
pub mod hnsw;
pub mod persistence;
pub mod server;
pub mod metrics;

pub use vector::Vector;
pub use storage::VectorStore;
pub use distance::DistanceMetric;
pub use error::{VectorDbError, Result};
pub use index::Index;
pub use flat_index::FlatIndex;
pub use hnsw::{HnswIndex, HnswParams};
