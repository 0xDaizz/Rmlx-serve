//! Sampling error types.

use thiserror::Error;

/// Errors that can occur during sampling configuration or execution.
#[derive(Debug, Error)]
pub enum SamplingError {
    /// Temperature must be non-negative.
    #[error("invalid temperature: {0} (must be >= 0.0)")]
    InvalidTemperature(f32),

    /// Top-p must be in (0.0, 1.0].
    #[error("invalid top_p: {0} (must be in (0.0, 1.0])")]
    InvalidTopP(f32),

    /// Top-k must be non-negative (0 disables).
    #[error("invalid top_k: {0} (must be >= 0)")]
    InvalidTopK(u32),

    /// Catch-all for other invalid parameter combinations.
    #[error("invalid sampling parameter: {0}")]
    InvalidParameter(String),
}
