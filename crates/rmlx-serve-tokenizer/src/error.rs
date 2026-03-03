//! Error types for the tokenizer crate.

use std::path::PathBuf;

/// All error variants that can occur in the tokenizer crate.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    /// Failed to load a tokenizer or its configuration from disk.
    #[error("failed to load tokenizer from {path}: {reason}")]
    LoadFailed {
        path: PathBuf,
        reason: String,
    },

    /// Encoding text into token IDs failed.
    #[error("encoding failed: {0}")]
    EncodeFailed(String),

    /// Decoding token IDs back into text failed.
    #[error("decoding failed: {0}")]
    DecodeFailed(String),

    /// Chat-template rendering failed.
    #[error("template rendering failed: {0}")]
    TemplateFailed(String),

    /// The configuration file was malformed or missing required fields.
    #[error("invalid tokenizer configuration: {0}")]
    InvalidConfig(String),
}

/// Convenience alias used throughout this crate.
pub type Result<T> = std::result::Result<T, TokenizerError>;
