//! rmlx-serve-models: Model loading, architecture wrappers, and inference traits.
//!
//! This crate bridges the gap between raw RMLX neural network components
//! (`rmlx-nn`) and the serving engine. It provides:
//!
//! - **[`LlmModel`]** — A trait defining the interface for LLM inference,
//!   including forward passes with KV cache, RoPE, and causal masking.
//!
//! - **[`TransformerLlm`]** — The primary model wrapper that handles RoPE
//!   frequency precomputation, causal mask creation, cache management, and
//!   last-token logit extraction.
//!
//! - **[`ModelRegistry`]** — Dynamic architecture selection based on
//!   `model_type` from config.json.
//!
//! - **[`load_model`]** — High-level entry point for loading a model from disk.
//!
//! - **[`rope`]** and **[`mask`]** — Utility modules for RoPE frequency
//!   computation and attention mask generation.
//!
//! # Quick Start
//!
//! ```ignore
//! use rmlx_serve_models::load_model;
//!
//! let (model, config) = load_model("/path/to/llama-3-8b")?;
//! let device = rmlx_metal::GpuDevice::system_default()?;
//! let mut cache = model.make_cache(device.raw());
//! // ... run inference with model.forward(...)
//! ```

pub mod error;
pub mod loader;
pub mod mask;
pub mod registry;
pub mod rope;
pub mod traits;
pub mod transformer;

// ── Re-exports for convenience ──
pub use error::ModelError;
pub use loader::{load_model, sharded_load};
pub use registry::ModelRegistry;
pub use traits::LlmModel;
pub use transformer::TransformerLlm;
