//! Error types for model loading and inference.

use std::path::PathBuf;

/// Errors that can occur during model loading, construction, or inference.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// The model directory or config file was not found.
    #[error("model not found: {0}")]
    NotFound(PathBuf),

    /// The model configuration is invalid or unsupported.
    #[error("invalid model config: {0}")]
    InvalidConfig(String),

    /// The requested model architecture is not supported.
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    /// An error occurred during weight loading.
    #[error("weight error: {0}")]
    WeightError(#[from] rmlx_serve_weights::WeightError),

    /// A kernel or compute error occurred during inference.
    #[error("kernel error: {0}")]
    KernelError(#[from] rmlx_core::KernelError),

    /// An error occurred acquiring or using the Metal GPU device.
    #[error("device error: {0}")]
    DeviceError(String),

    /// The model forward pass produced an unexpected output shape.
    #[error("shape error: {0}")]
    ShapeError(String),

    /// An I/O error occurred while reading model files.
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON parsing error (config.json, etc.).
    #[error("json error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Convenience alias for model results.
pub type Result<T> = std::result::Result<T, ModelError>;
