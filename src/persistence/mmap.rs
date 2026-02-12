//! Memory-mapped vector storage for large datasets.
//!
//! Stores vectors in a flat binary file where each vector is stored as
//! contiguous f32 values. Uses regular file I/O for writes and can optionally
//! use memory mapping for reads.

use crate::error::{Result, VectorDbError};
use crate::vector::Vector;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Header written at the start of the file.
/// [dimension: u32][count: u32]
const HEADER_SIZE: usize = 8;

/// Memory-mapped (or file-backed) vector storage.
pub struct MmapVectorStorage {
    path: PathBuf,
    dimension: usize,
    count: usize,
}

impl MmapVectorStorage {
    /// Create a new storage file.
    pub fn create(path: impl AsRef<Path>, dimension: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        let header = Self::encode_header(dimension, 0);
        file.write_all(&header)?;
        file.sync_all()?;

        Ok(Self {
            path,
            dimension,
            count: 0,
        })
    }

    /// Open an existing storage file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;

        let mut header = [0u8; HEADER_SIZE];
        file.read_exact(&mut header).map_err(|_| {
            VectorDbError::StorageError("File too small for header".to_string())
        })?;

        let (dimension, count) = Self::decode_header(&header);

        Ok(Self {
            path,
            dimension,
            count,
        })
    }

    /// Append a vector to the file.
    pub fn append(&mut self, vector: &Vector) -> Result<usize> {
        if vector.dimension() != self.dimension {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.dimension(),
            });
        }

        let mut file = OpenOptions::new().read(true).write(true).open(&self.path)?;

        // Seek to end of data
        let vec_bytes = self.dimension * 4;
        let offset = (HEADER_SIZE + self.count * vec_bytes) as u64;
        file.seek(SeekFrom::Start(offset))?;

        // Write vector data as little-endian f32s
        for &val in vector.as_slice() {
            file.write_all(&val.to_le_bytes())?;
        }

        // Update header count
        self.count += 1;
        let header = Self::encode_header(self.dimension, self.count);
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;

        file.sync_all()?;

        Ok(self.count - 1)
    }

    /// Read a vector by index.
    pub fn get(&self, index: usize) -> Result<Vector> {
        if index >= self.count {
            return Err(VectorDbError::IndexError(format!(
                "Index {} out of range (count={})",
                index, self.count
            )));
        }

        let mut file = File::open(&self.path)?;

        let vec_bytes = self.dimension * 4;
        let offset = (HEADER_SIZE + index * vec_bytes) as u64;
        file.seek(SeekFrom::Start(offset))?;

        let mut data = Vec::with_capacity(self.dimension);
        for _ in 0..self.dimension {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            data.push(f32::from_le_bytes(buf));
        }

        Ok(Vector::new(data))
    }

    /// Try to memory-map the file for read-only access (best-effort).
    /// Falls back to regular file I/O if mmap is unavailable.
    pub fn get_mmap(&self, index: usize) -> Result<Vector> {
        if index >= self.count {
            return Err(VectorDbError::IndexError(format!(
                "Index {} out of range (count={})",
                index, self.count
            )));
        }

        let file = File::open(&self.path)?;
        match unsafe { memmap2::Mmap::map(&file) } {
            Ok(mmap) => {
                let vec_bytes = self.dimension * 4;
                let offset = HEADER_SIZE + index * vec_bytes;

                let mut data = Vec::with_capacity(self.dimension);
                for i in 0..self.dimension {
                    let byte_offset = offset + i * 4;
                    let bytes: [u8; 4] =
                        mmap[byte_offset..byte_offset + 4].try_into().unwrap();
                    data.push(f32::from_le_bytes(bytes));
                }
                Ok(Vector::new(data))
            }
            Err(_) => self.get(index), // Fallback to regular I/O
        }
    }

    /// Get the number of stored vectors.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    fn encode_header(dimension: usize, count: usize) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&(dimension as u32).to_le_bytes());
        buf[4..8].copy_from_slice(&(count as u32).to_le_bytes());
        buf
    }

    fn decode_header(data: &[u8]) -> (usize, usize) {
        let dimension = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        (dimension, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_mmap_create_and_append() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");

        let mut storage = MmapVectorStorage::create(&path, 3).unwrap();
        storage
            .append(&Vector::new(vec![1.0, 2.0, 3.0]))
            .unwrap();
        storage
            .append(&Vector::new(vec![4.0, 5.0, 6.0]))
            .unwrap();
        assert_eq!(storage.count(), 2);

        let v0 = storage.get(0).unwrap();
        assert_eq!(v0.as_slice(), &[1.0, 2.0, 3.0]);

        let v1 = storage.get(1).unwrap();
        assert_eq!(v1.as_slice(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mmap_reopen() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");

        {
            let mut storage = MmapVectorStorage::create(&path, 2).unwrap();
            storage.append(&Vector::new(vec![1.5, 2.5])).unwrap();
            storage.append(&Vector::new(vec![3.5, 4.5])).unwrap();
        }

        let storage = MmapVectorStorage::open(&path).unwrap();
        assert_eq!(storage.count(), 2);
        assert_eq!(storage.dimension(), 2);

        let v = storage.get(1).unwrap();
        assert_eq!(v.as_slice(), &[3.5, 4.5]);
    }

    #[test]
    fn test_mmap_dimension_mismatch() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");

        let mut storage = MmapVectorStorage::create(&path, 3).unwrap();
        let result = storage.append(&Vector::new(vec![1.0, 2.0]));
        assert!(result.is_err());
    }
}
