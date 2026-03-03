//! Error types for speculative decoding.

use thiserror::Error;

/// Errors that can occur during speculative decoding.
#[derive(Debug, Error)]
pub enum SpecError {
    /// The draft proposer failed to generate candidate tokens.
    #[error("proposal failed: {0}")]
    ProposalFailed(String),

    /// Verification of draft tokens against the target model failed.
    #[error("verification failed: {0}")]
    VerificationFailed(String),

    /// An error occurred in the draft model during forward pass or cache creation.
    #[error("draft model error: {0}")]
    DraftModelError(String),

    /// A KV cache operation failed during speculative decoding.
    #[error("cache error: {0}")]
    CacheError(String),

    /// The speculative decoding configuration is invalid.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// Convenience alias for speculative decoding results.
pub type Result<T> = std::result::Result<T, SpecError>;
