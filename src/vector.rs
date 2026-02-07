//! Vector type and operations

use crate::error::{Result, VectorDbError};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub, Mul};

/// A vector in n-dimensional space
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vector {
    data: Vec<f32>,
}

impl Vector {
    /// Create a new vector from a Vec<f32>
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Get the dimension of the vector
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Get the underlying data as a slice
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Check if this vector has the same dimension as another
    pub fn has_same_dimension(&self, other: &Vector) -> bool {
        self.dimension() == other.dimension()
    }

    /// Compute the L2 norm (magnitude) of the vector
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize the vector to unit length
    pub fn normalize(&mut self) -> Result<()> {
        let norm = self.norm();
        if norm == 0.0 {
            return Err(VectorDbError::InvalidVector {
                reason: "Cannot normalize zero vector".to_string(),
            });
        }
        for x in &mut self.data {
            *x /= norm;
        }
        Ok(())
    }

    /// Create a normalized copy of the vector
    pub fn normalized(&self) -> Result<Vector> {
        let mut v = self.clone();
        v.normalize()?;
        Ok(v)
    }

    /// Parse a vector from a comma-separated string
    pub fn from_str(s: &str) -> Result<Self> {
        let data: Result<Vec<f32>> = s
            .split(',')
            .map(|x| {
                x.trim()
                    .parse::<f32>()
                    .map_err(|_| VectorDbError::InvalidVector {
                        reason: format!("Invalid float: {}", x),
                    })
            })
            .collect();
        Ok(Vector::new(data?))
    }
}

impl Add for Vector {
    type Output = Result<Vector>;

    fn add(self, other: Vector) -> Result<Vector> {
        if !self.has_same_dimension(&other) {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.dimension(),
                actual: other.dimension(),
            });
        }
        Ok(Vector::new(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        ))
    }
}

impl Sub for Vector {
    type Output = Result<Vector>;

    fn sub(self, other: Vector) -> Result<Vector> {
        if !self.has_same_dimension(&other) {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.dimension(),
                actual: other.dimension(),
            });
        }
        Ok(Vector::new(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
        ))
    }
}

impl Mul<f32> for Vector {
    type Output = Vector;

    fn mul(self, scalar: f32) -> Vector {
        Vector::new(self.data.iter().map(|x| x * scalar).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vector_creation() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.dimension(), 3);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_norm() {
        let v = Vector::new(vec![3.0, 4.0]);
        assert_relative_eq!(v.norm(), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_vector_normalize() {
        let mut v = Vector::new(vec![3.0, 4.0]);
        v.normalize().unwrap();
        assert_relative_eq!(v.norm(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(v.as_slice()[0], 0.6, epsilon = 1e-6);
        assert_relative_eq!(v.as_slice()[1], 0.8, epsilon = 1e-6);
    }

    #[test]
    fn test_vector_addition() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let result = (v1 + v2).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_subtraction() {
        let v1 = Vector::new(vec![4.0, 5.0, 6.0]);
        let v2 = Vector::new(vec![1.0, 2.0, 3.0]);
        let result = (v1 - v2).unwrap();
        assert_eq!(result.as_slice(), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_vector_scalar_multiplication() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        let result = v * 2.0;
        assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_from_str() {
        let v = Vector::from_str("1.0, 2.0, 3.0").unwrap();
        assert_eq!(v.dimension(), 3);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let v1 = Vector::new(vec![1.0, 2.0]);
        let v2 = Vector::new(vec![1.0, 2.0, 3.0]);
        assert!(matches!(v1 + v2, Err(VectorDbError::DimensionMismatch { .. })));
    }
}
