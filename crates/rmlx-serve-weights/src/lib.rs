//! rmlx-serve-weights: Model weight loading, mapping, and management.
//!
//! This crate handles the complete pipeline from HuggingFace model files on
//! disk to a fully constructed `TransformerModel` on GPU:
//!
//! 1. **Config** — Parse `config.json` into a `ModelConfig`, then convert to
//!    an RMLX `TransformerConfig`.
//! 2. **Loader** — Read safetensors files (single or sharded) and create
//!    `rmlx_core::Array` tensors on the Metal GPU.
//! 3. **Mapper** — Translate HuggingFace weight names (e.g.,
//!    `model.layers.5.self_attn.q_proj.weight`) to RMLX model components.
//! 4. **Sanitize** — Remove unnecessary tensors (inv_freq), handle tied
//!    embeddings, strip bias from bias-free models.
//! 5. **Quantization** — Detect and unpack AWQ/GPTQ quantized weights.
//!
//! # Quick Start
//!
//! ```ignore
//! use rmlx_serve_weights::load_model;
//!
//! let (model, config) = load_model("/path/to/model")?;
//! // model is a TransformerModel ready for inference
//! ```

pub mod config;
pub mod error;
pub mod loader;
pub mod mapper;
pub mod quantization;
pub mod sanitize;

// ── Re-exports for convenience ──
pub use config::{ModelConfig, QuantizationConfig};
pub use error::WeightError;
pub use loader::{load_model, load_safetensors};
pub use mapper::{
    create_weight_mapper, MappedWeight, WeightComponent, WeightMapper,
    DeepSeekWeightMapper, LlamaWeightMapper, MixtralWeightMapper, QwenWeightMapper,
};
pub use quantization::{detect_quantization, QuantMethod};
pub use sanitize::sanitize_weights;
