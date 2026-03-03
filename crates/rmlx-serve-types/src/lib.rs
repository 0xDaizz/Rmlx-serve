//! rmlx-serve-types: Shared type definitions for the rmlx-serve inference engine.
//!
//! This crate provides all the common types used across rmlx-serve crates,
//! including request/response types, OpenAI-compatible API types, Anthropic API types,
//! configuration structures, error types, and MCP (Model Context Protocol) types.
//!
//! This crate has NO rmlx dependencies -- only serde, serde_json, uuid, and thiserror.

pub mod anthropic;
pub mod config;
pub mod error;
pub mod mcp;
pub mod openai;
pub mod request;

// Re-export the most commonly used types at crate root for convenience.
pub use config::{CacheConfig, EngineConfig, SchedulerConfig, ServerConfig};
pub use error::LmError;
pub use request::{
    CompletionOutput, FinishReason, Request, RequestId, RequestMetrics, RequestOutput,
    RequestStatus, SamplingParams, TokenLogprob,
};
