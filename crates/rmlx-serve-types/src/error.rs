//! Error types for the rmlx-serve engine.

use std::fmt;

/// Top-level error type covering all failure modes in the serving stack.
#[derive(Debug, thiserror::Error)]
pub enum LmError {
    /// Failed to load or initialise the model architecture.
    #[error("model error: {0}")]
    Model(String),

    /// Tokenizer initialisation or encoding/decoding failure.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Failed to load model weights from disk or network.
    #[error("weight loading error: {0}")]
    WeightLoading(String),

    /// KV-cache allocation or management failure.
    #[error("cache error: {0}")]
    Cache(String),

    /// Scheduler policy or capacity error.
    #[error("scheduler error: {0}")]
    Scheduler(String),

    /// Sampling or logit processing error.
    #[error("sampling error: {0}")]
    Sampling(String),

    /// Engine-level orchestration error.
    #[error("engine error: {0}")]
    Engine(String),

    /// Distributed / multi-device communication error.
    #[error("distributed error: {0}")]
    Distributed(String),

    /// Standard I/O error wrapper.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration validation error.
    #[error("config error: {0}")]
    Config(String),

    /// HTTP / API layer error.
    #[error("api error: {0}")]
    Api(String),

    /// Tool-call JSON parsing error.
    #[error("tool parsing error: {0}")]
    ToolParsing(String),

    /// Speculative decoding error.
    #[error("speculative decoding error: {0}")]
    Speculative(String),

    /// An argument supplied by the caller was invalid.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Catch-all for internal / unexpected failures.
    #[error("internal error: {0}")]
    Internal(String),
}

impl LmError {
    /// Convenience constructor for [`LmError::InvalidArgument`].
    pub fn invalid_argument(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }

    /// Convenience constructor for [`LmError::Internal`].
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}

// Allow converting serde_json errors into LmError for ergonomic `?` usage.
impl From<serde_json::Error> for LmError {
    fn from(err: serde_json::Error) -> Self {
        Self::Api(fmt::format(format_args!("json error: {err}")))
    }
}
