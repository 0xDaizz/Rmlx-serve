//! Error types for the cache crate.

use std::path::PathBuf;

/// Errors that can occur during KV cache operations.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// Failed to allocate a cache block or buffer.
    #[error("allocation failed: {0}")]
    AllocationFailed(String),

    /// The cache has reached its maximum capacity.
    #[error("capacity exceeded: {0}")]
    CapacityExceeded(String),

    /// An invalid block ID was referenced.
    #[error("invalid block id: {0}")]
    InvalidBlockId(usize),

    /// Internal cache state is corrupted.
    #[error("cache corrupted: {0}")]
    CacheCorrupted(String),

    /// Eviction failed (e.g., no blocks eligible for eviction).
    #[error("eviction failed: {0}")]
    EvictionFailed(String),

    /// Error during cache serialization or deserialization.
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// I/O error (file read/write).
    #[error("io error at {path:?}: {source}")]
    IoError {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl From<std::io::Error> for CacheError {
    fn from(err: std::io::Error) -> Self {
        CacheError::IoError {
            path: PathBuf::new(),
            source: err,
        }
    }
}

impl From<serde_json::Error> for CacheError {
    fn from(err: serde_json::Error) -> Self {
        CacheError::SerializationError(err.to_string())
    }
}

/// Convenience alias for `Result<T, CacheError>`.
pub type Result<T> = std::result::Result<T, CacheError>;
