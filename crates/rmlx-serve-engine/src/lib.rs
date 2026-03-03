//! rmlx-serve-engine: Core inference engine for the rmlx-serve stack.
//!
//! This crate provides:
//!
//! - **[`Engine`]** -- The core trait that the API server and other consumers
//!   programme against, allowing different backends to be plugged in.
//!
//! - **[`SimpleEngine`]** -- Single-request, non-batched engine. Processes
//!   one request at a time with the model locked behind a `tokio::sync::Mutex`.
//!   Best for CLI tools, testing, and low-concurrency scenarios.
//!
//! - **[`BatchedEngine`]** -- Continuous-batching engine backed by the
//!   [`Scheduler`](rmlx_serve_scheduler::Scheduler) and
//!   [`BatchGenerator`](rmlx_serve_scheduler::BatchGenerator). Runs a
//!   background tokio task that manages multiple concurrent requests,
//!   performing prefill and decode in batched steps.
//!
//! - **[`generate_text`]** -- Convenience function for one-shot CLI generation.

pub mod batched;
pub mod error;
pub mod generation;
pub mod simple;
pub mod traits;

use async_trait::async_trait;
use rmlx_serve_types::{Request, RequestOutput};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

// Re-export concrete implementations.
pub use batched::BatchedEngine;
pub use generation::{generate_text, GenerationResponse};
pub use simple::SimpleEngine;

// ---------------------------------------------------------------------------
// EngineHealth
// ---------------------------------------------------------------------------

/// Health status returned by the `/health` endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineHealth {
    /// Whether the engine is ready to serve requests.
    pub is_ready: bool,

    /// Human-readable status message.
    pub status: String,

    /// Model name / identifier.
    pub model: String,

    /// Number of requests currently in flight.
    pub active_requests: usize,
}

// ---------------------------------------------------------------------------
// EngineStats
// ---------------------------------------------------------------------------

/// Runtime statistics exposed via the `/metrics` endpoint.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EngineStats {
    /// Total number of requests received since startup.
    pub total_requests: u64,

    /// Number of requests currently being processed.
    pub active_requests: u64,

    /// Total prompt tokens processed.
    pub total_prompt_tokens: u64,

    /// Total completion tokens generated.
    pub total_completion_tokens: u64,

    /// Average time-to-first-token in milliseconds.
    pub avg_ttft_ms: f64,

    /// Average tokens per second across all completed requests.
    pub avg_tps: f64,

    /// Engine uptime in seconds.
    pub uptime_secs: f64,
}

// ---------------------------------------------------------------------------
// Engine trait
// ---------------------------------------------------------------------------

/// The core inference engine trait.
///
/// All interactions from the HTTP API layer go through this trait, allowing
/// different backends (MLX, mock, etc.) to be plugged in.
#[async_trait]
pub trait Engine: Send + Sync + 'static {
    /// Return the model name / identifier this engine is serving.
    fn model_name(&self) -> &str;

    /// Run non-streaming inference: submit a request and wait for the final
    /// output.
    async fn generate(&self, request: Request) -> Result<RequestOutput, EngineError>;

    /// Run streaming inference: submit a request and receive incremental
    /// outputs via a channel receiver.
    async fn generate_stream(
        &self,
        request: Request,
    ) -> Result<mpsc::UnboundedReceiver<RequestOutput>, EngineError>;

    /// Report the current health status of the engine.
    async fn health(&self) -> EngineHealth;

    /// Return aggregate statistics for the engine.
    fn get_stats(&self) -> EngineStats;

    /// Encode text to token IDs using the engine's tokenizer.
    fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError>;

    /// Decode token IDs back to text using the engine's tokenizer.
    fn decode(&self, token_ids: &[u32]) -> Result<String, EngineError>;
}

// ---------------------------------------------------------------------------
// EngineError
// ---------------------------------------------------------------------------

/// Errors that can occur during engine operations.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// Failed to load or run the model.
    #[error("model error: {0}")]
    Model(String),

    /// Tokenizer encoding/decoding failure.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Invalid request (bad parameters, empty prompt, etc.).
    #[error("request error: {0}")]
    Request(String),

    /// The engine cannot accept more requests (scheduler queue full, etc.).
    #[error("capacity exceeded: {0}")]
    CapacityExceeded(String),

    /// The request was cancelled (aborted by the caller or timed out).
    #[error("cancelled: {0}")]
    Cancelled(String),

    /// Catch-all for unexpected internal failures.
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<rmlx_serve_models::ModelError> for EngineError {
    fn from(err: rmlx_serve_models::ModelError) -> Self {
        EngineError::Model(err.to_string())
    }
}

impl From<rmlx_serve_tokenizer::TokenizerError> for EngineError {
    fn from(err: rmlx_serve_tokenizer::TokenizerError) -> Self {
        EngineError::Tokenizer(err.to_string())
    }
}

impl From<rmlx_serve_scheduler::SchedulerError> for EngineError {
    fn from(err: rmlx_serve_scheduler::SchedulerError) -> Self {
        EngineError::Internal(format!("scheduler: {err}"))
    }
}

impl From<rmlx_core::KernelError> for EngineError {
    fn from(err: rmlx_core::KernelError) -> Self {
        EngineError::Internal(format!("kernel: {err}"))
    }
}
