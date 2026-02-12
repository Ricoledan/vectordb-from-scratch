//! Serialization utilities: bincode for vectors/graph, JSON for metadata/config.

use crate::error::{Result, VectorDbError};
use crate::vector::Vector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Serializable representation of a stored vector with its ID mapping.
#[derive(Debug, Serialize, Deserialize)]
pub struct SerializedVector {
    pub internal_id: usize,
    pub string_id: String,
    pub data: Vec<f32>,
}

/// Serializable representation of the full database state.
#[derive(Debug, Serialize, Deserialize)]
pub struct DatabaseSnapshot {
    pub vectors: Vec<SerializedVector>,
    pub metadata: HashMap<usize, HashMap<String, String>>,
    pub next_id: usize,
    pub dimension: Option<usize>,
}

/// Encode data to bincode bytes.
pub fn to_bincode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    bincode::serialize(value).map_err(|e| VectorDbError::SerializationError(e.to_string()))
}

/// Decode data from bincode bytes.
pub fn from_bincode<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T> {
    bincode::deserialize(bytes).map_err(|e| VectorDbError::SerializationError(e.to_string()))
}

/// Encode data to JSON bytes.
pub fn to_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|e| VectorDbError::SerializationError(e.to_string()))
}

/// Decode data from JSON bytes.
pub fn from_json<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|e| VectorDbError::SerializationError(e.to_string()))
}

/// Convert a Vector to a serializable form.
pub fn serialize_vector(internal_id: usize, string_id: &str, vector: &Vector) -> SerializedVector {
    SerializedVector {
        internal_id,
        string_id: string_id.to_string(),
        data: vector.as_slice().to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bincode_roundtrip() {
        let sv = SerializedVector {
            internal_id: 42,
            string_id: "test".to_string(),
            data: vec![1.0, 2.0, 3.0],
        };
        let bytes = to_bincode(&sv).unwrap();
        let decoded: SerializedVector = from_bincode(&bytes).unwrap();
        assert_eq!(decoded.internal_id, 42);
        assert_eq!(decoded.string_id, "test");
        assert_eq!(decoded.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_json_roundtrip() {
        let sv = SerializedVector {
            internal_id: 1,
            string_id: "hello".to_string(),
            data: vec![0.5, 1.5],
        };
        let bytes = to_json(&sv).unwrap();
        let decoded: SerializedVector = from_json(&bytes).unwrap();
        assert_eq!(decoded.internal_id, 1);
        assert_eq!(decoded.string_id, "hello");
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let snapshot = DatabaseSnapshot {
            vectors: vec![
                SerializedVector {
                    internal_id: 0,
                    string_id: "v1".to_string(),
                    data: vec![1.0, 2.0],
                },
            ],
            metadata: HashMap::new(),
            next_id: 1,
            dimension: Some(2),
        };
        let bytes = to_bincode(&snapshot).unwrap();
        let decoded: DatabaseSnapshot = from_bincode(&bytes).unwrap();
        assert_eq!(decoded.vectors.len(), 1);
        assert_eq!(decoded.next_id, 1);
        assert_eq!(decoded.dimension, Some(2));
    }
}
