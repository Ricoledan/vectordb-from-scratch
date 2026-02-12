//! Persistence layer: WAL, snapshots, and crash recovery.

pub mod serialization;
pub mod wal;
pub mod snapshot;
pub mod engine;
pub mod mmap;
