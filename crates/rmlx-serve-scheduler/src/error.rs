//! Error types for the scheduler crate.

use thiserror::Error;

/// Errors that can occur during scheduling and batch generation.
#[derive(Debug, Error)]
pub enum SchedulerError {
    /// The request queue is full; cannot accept more requests.
    #[error("queue full: {0}")]
    QueueFull(String),

    /// The request is invalid (e.g., prompt too long, invalid parameters).
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// KV cache capacity is exhausted; no room for new sequences.
    #[error("cache exhausted: {0}")]
    CacheExhausted(String),

    /// An error occurred during model forward pass.
    #[error("model error: {0}")]
    ModelError(String),

    /// An error occurred during batch processing (prefill or decode).
    #[error("batch error: {0}")]
    BatchError(String),

    /// Catch-all for unexpected internal errors.
    #[error("internal error: {0}")]
    InternalError(String),
}

impl From<rmlx_serve_models::ModelError> for SchedulerError {
    fn from(err: rmlx_serve_models::ModelError) -> Self {
        SchedulerError::ModelError(err.to_string())
    }
}

impl From<rmlx_core::KernelError> for SchedulerError {
    fn from(err: rmlx_core::KernelError) -> Self {
        SchedulerError::ModelError(err.to_string())
    }
}
