//! Distance metrics for vector similarity

use crate::error::{Result, VectorDbError};
use crate::vector::Vector;
use serde::{Deserialize, Serialize};

/// Distance metrics for measuring vector similarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance
    Euclidean,
    /// Cosine similarity (converted to distance: 1 - similarity)
    Cosine,
    /// Dot product (negated for minimum distance)
    DotProduct,
}

impl DistanceMetric {
    /// Compute the distance between two vectors using this metric
    pub fn distance(&self, v1: &Vector, v2: &Vector) -> Result<f32> {
        if !v1.has_same_dimension(v2) {
            return Err(VectorDbError::DimensionMismatch {
                expected: v1.dimension(),
                actual: v2.dimension(),
            });
        }

        match self {
            DistanceMetric::Euclidean => Ok(euclidean_distance(v1, v2)),
            DistanceMetric::Cosine => cosine_distance(v1, v2),
            DistanceMetric::DotProduct => Ok(-dot_product(v1, v2)),
        }
    }
}

/// Compute Euclidean (L2) distance between two vectors
pub fn euclidean_distance(v1: &Vector, v2: &Vector) -> f32 {
    v1.as_slice()
        .iter()
        .zip(v2.as_slice().iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute cosine distance between two vectors (1 - cosine similarity)
pub fn cosine_distance(v1: &Vector, v2: &Vector) -> Result<f32> {
    let norm1 = v1.norm();
    let norm2 = v2.norm();

    if norm1 == 0.0 || norm2 == 0.0 {
        return Err(VectorDbError::InvalidVector {
            reason: "Cannot compute cosine distance with zero vector".to_string(),
        });
    }

    let dot = dot_product(v1, v2);
    let similarity = dot / (norm1 * norm2);

    // Clamp to [-1, 1] to handle floating point errors
    let similarity = similarity.clamp(-1.0, 1.0);

    Ok(1.0 - similarity)
}

/// Compute dot product of two vectors
pub fn dot_product(v1: &Vector, v2: &Vector) -> f32 {
    v1.as_slice()
        .iter()
        .zip(v2.as_slice().iter())
        .map(|(a, b)| a * b)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_euclidean_distance() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let dist = euclidean_distance(&v1, &v2);
        assert_relative_eq!(dist, 5.196152, epsilon = 1e-5);
    }

    #[test]
    fn test_euclidean_same_vector() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        let dist = euclidean_distance(&v, &v);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let dot = dot_product(&v1, &v2);
        assert_relative_eq!(dot, 32.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let v1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(vec![1.0, 0.0, 0.0]);
        let dist = cosine_distance(&v1, &v2).unwrap();
        assert_relative_eq!(dist, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let v1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(vec![0.0, 1.0, 0.0]);
        let dist = cosine_distance(&v1, &v2).unwrap();
        assert_relative_eq!(dist, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let v1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(vec![-1.0, 0.0, 0.0]);
        let dist = cosine_distance(&v1, &v2).unwrap();
        assert_relative_eq!(dist, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_distance_metric_euclidean() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let dist = DistanceMetric::Euclidean.distance(&v1, &v2).unwrap();
        assert_relative_eq!(dist, 5.196152, epsilon = 1e-5);
    }

    #[test]
    fn test_dimension_mismatch() {
        let v1 = Vector::new(vec![1.0, 2.0]);
        let v2 = Vector::new(vec![1.0, 2.0, 3.0]);
        assert!(matches!(
            DistanceMetric::Euclidean.distance(&v1, &v2),
            Err(VectorDbError::DimensionMismatch { .. })
        ));
    }
}
