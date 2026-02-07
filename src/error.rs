//! Error types for the vector database

use thiserror::Error;

/// Result type alias for VectorDB operations
pub type Result<T> = std::result::Result<T, VectorDbError>;

/// Error types that can occur in VectorDB operations
#[derive(Error, Debug)]
pub enum VectorDbError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Vector not found: {id}")]
    VectorNotFound { id: String },

    #[error("Invalid vector: {reason}")]
    InvalidVector { reason: String },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Index error: {0}")]
    IndexError(String),
}
