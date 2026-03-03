//! Error types for weight loading and processing.

use std::path::PathBuf;

/// Errors that can occur during weight loading and model construction.
#[derive(Debug, thiserror::Error)]
pub enum WeightError {
    /// The config.json file was not found at the expected path.
    #[error("config.json not found at {0}")]
    ConfigNotFound(PathBuf),

    /// The config.json file could not be parsed or contains invalid values.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// A safetensors file could not be read or parsed.
    #[error("safetensors error: {0}")]
    SafetensorsError(String),

    /// A referenced shard file was not found on disk.
    #[error("shard not found: {path}")]
    ShardNotFound { path: PathBuf },

    /// The dtype of a loaded tensor does not match the expected dtype.
    #[error("dtype mismatch for {name}: expected {expected}, got {actual}")]
    DTypeMismatch {
        name: String,
        expected: String,
        actual: String,
    },

    /// A required weight tensor is missing from the safetensors file(s).
    #[error("missing weight: {0}")]
    MissingWeight(String),

    /// An error occurred during quantization processing.
    #[error("quantization error: {0}")]
    QuantizationError(String),

    /// An I/O error occurred while reading files.
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    /// An error from the RMLX kernel layer (shape validation, etc.).
    #[error("kernel error: {0}")]
    KernelError(#[from] rmlx_core::KernelError),
}

/// Convenience alias for weight-loading results.
pub type Result<T> = std::result::Result<T, WeightError>;
