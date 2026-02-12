//! Write-Ahead Log (WAL) for crash recovery.
//!
//! Each entry is written as: [length: u32][crc32: u32][payload: bincode(WalEntry)]
//! The WAL is append-only and fsynced after each write.

use crate::error::{Result, VectorDbError};
use crate::persistence::serialization;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

/// A single WAL entry.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum WalEntry {
    Insert {
        string_id: String,
        internal_id: usize,
        data: Vec<f32>,
    },
    Delete {
        string_id: String,
    },
    Checkpoint,
}

/// Write-Ahead Log file manager.
pub struct WriteAheadLog {
    path: PathBuf,
    file: File,
}

impl WriteAheadLog {
    /// Open (or create) a WAL file at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self { path, file })
    }

    /// Append an entry to the WAL and fsync.
    pub fn append(&mut self, entry: &WalEntry) -> Result<()> {
        let payload = serialization::to_bincode(entry)?;
        let crc = crc32fast::hash(&payload);
        let len = payload.len() as u32;

        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&crc.to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.sync()?;

        Ok(())
    }

    /// Fsync the WAL file.
    pub fn sync(&self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }

    /// Replay all valid entries from the WAL.
    /// Stops at the first corrupted or incomplete entry (crash tolerance).
    pub fn replay(&self) -> Result<Vec<WalEntry>> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        loop {
            // Read length
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(VectorDbError::IoError(e)),
            }
            let len = u32::from_le_bytes(len_buf) as usize;

            // Read CRC
            let mut crc_buf = [0u8; 4];
            match reader.read_exact(&mut crc_buf) {
                Ok(()) => {}
                Err(_) => break, // Truncated — stop
            }
            let expected_crc = u32::from_le_bytes(crc_buf);

            // Read payload
            let mut payload = vec![0u8; len];
            match reader.read_exact(&mut payload) {
                Ok(()) => {}
                Err(_) => break, // Truncated — stop
            }

            // Verify CRC
            let actual_crc = crc32fast::hash(&payload);
            if actual_crc != expected_crc {
                break; // Corrupted — stop
            }

            // Deserialize
            match serialization::from_bincode::<WalEntry>(&payload) {
                Ok(entry) => entries.push(entry),
                Err(_) => break, // Corrupted — stop
            }
        }

        Ok(entries)
    }

    /// Truncate the WAL file (after a successful checkpoint).
    pub fn truncate(&mut self) -> Result<()> {
        self.file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_wal_write_and_replay() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let mut wal = WriteAheadLog::open(&wal_path).unwrap();
            wal.append(&WalEntry::Insert {
                string_id: "v1".to_string(),
                internal_id: 0,
                data: vec![1.0, 2.0, 3.0],
            })
            .unwrap();
            wal.append(&WalEntry::Insert {
                string_id: "v2".to_string(),
                internal_id: 1,
                data: vec![4.0, 5.0, 6.0],
            })
            .unwrap();
            wal.append(&WalEntry::Delete {
                string_id: "v1".to_string(),
            })
            .unwrap();
        }

        let wal = WriteAheadLog::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 3);

        assert!(matches!(&entries[0], WalEntry::Insert { string_id, .. } if string_id == "v1"));
        assert!(matches!(&entries[1], WalEntry::Insert { string_id, .. } if string_id == "v2"));
        assert!(matches!(&entries[2], WalEntry::Delete { string_id } if string_id == "v1"));
    }

    #[test]
    fn test_wal_truncated_entry() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write a valid entry
        {
            let mut wal = WriteAheadLog::open(&wal_path).unwrap();
            wal.append(&WalEntry::Insert {
                string_id: "v1".to_string(),
                internal_id: 0,
                data: vec![1.0],
            })
            .unwrap();
        }

        // Append garbage (simulates a crash mid-write)
        {
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            file.write_all(&[0xFF, 0xFF, 0xFF]).unwrap();
        }

        let wal = WriteAheadLog::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1); // Only the valid entry
    }

    #[test]
    fn test_wal_truncate() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut wal = WriteAheadLog::open(&wal_path).unwrap();
        wal.append(&WalEntry::Checkpoint).unwrap();
        assert_eq!(wal.replay().unwrap().len(), 1);

        wal.truncate().unwrap();
        // Re-open to replay
        let wal = WriteAheadLog::open(&wal_path).unwrap();
        assert_eq!(wal.replay().unwrap().len(), 0);
    }
}
