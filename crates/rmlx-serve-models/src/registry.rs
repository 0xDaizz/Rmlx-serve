//! Model registry for dynamic architecture selection.
//!
//! The `ModelRegistry` maps model type strings (from config.json's `model_type`
//! field) to loader functions. It comes pre-populated with loaders for all
//! supported architectures and can be extended with custom loaders.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use rmlx_metal::GpuDevice;
use rmlx_nn::{FeedForwardType, MoeConfig, TransformerConfig};
use rmlx_serve_weights::ModelConfig;
use tracing::{debug, info};

use crate::error::{ModelError, Result};
use crate::traits::LlmModel;
use crate::transformer::TransformerLlm;

/// Type alias for a model loader function.
///
/// Takes a model directory path and a GPU device, returns a boxed `LlmModel`.
type LoaderFn = dyn Fn(&Path, GpuDevice) -> Result<Box<dyn LlmModel>> + Send + Sync;

/// Registry mapping model architecture names to their loader functions.
///
/// The registry is populated with default loaders for all supported
/// architectures at construction time. Additional architectures can be
/// registered dynamically.
pub struct ModelRegistry {
    loaders: HashMap<String, Arc<LoaderFn>>,
}

impl ModelRegistry {
    /// Create a new registry with default architecture loaders.
    ///
    /// Registers loaders for: "llama", "qwen2", "mixtral", "deepseek_v3".
    pub fn new() -> Self {
        let mut registry = Self {
            loaders: HashMap::new(),
        };

        registry.register("llama", load_transformer_model);
        registry.register("qwen2", load_transformer_model);
        registry.register("mixtral", load_transformer_model);
        registry.register("deepseek_v3", load_transformer_model);

        // Common aliases
        registry.register("deepseek_v2", load_transformer_model);
        registry.register("mistral", load_transformer_model);

        debug!(
            architectures = ?registry.supported_types(),
            "model registry initialized"
        );

        registry
    }

    /// Register a loader for a model type.
    ///
    /// If a loader for this model type already exists, it is replaced.
    pub fn register<F>(&mut self, model_type: &str, loader: F)
    where
        F: Fn(&Path, GpuDevice) -> Result<Box<dyn LlmModel>> + Send + Sync + 'static,
    {
        self.loaders
            .insert(model_type.to_string(), Arc::new(loader));
    }

    /// Load a model from a directory.
    ///
    /// Reads `config.json` to determine the architecture, then dispatches to
    /// the appropriate registered loader.
    ///
    /// # Arguments
    /// * `model_path` - Path to a directory containing `config.json` and
    ///   safetensors weight files.
    ///
    /// # Errors
    /// Returns `ModelError::UnsupportedArchitecture` if the model type from
    /// config.json is not registered, or propagates any loader errors.
    pub fn load(&self, model_path: &Path) -> Result<Box<dyn LlmModel>> {
        let config = ModelConfig::from_path(model_path)?;

        info!(
            model_type = config.model_type.as_str(),
            path = %model_path.display(),
            "loading model from registry"
        );

        let loader = self
            .loaders
            .get(&config.model_type)
            .ok_or_else(|| {
                ModelError::UnsupportedArchitecture(format!(
                    "unsupported model_type '{}'; supported: {:?}",
                    config.model_type,
                    self.supported_types()
                ))
            })?
            .clone();

        let device = GpuDevice::system_default()
            .map_err(|e| ModelError::DeviceError(format!("failed to acquire Metal device: {e}")))?;

        loader(model_path, device)
    }

    /// List all registered model types.
    pub fn supported_types(&self) -> Vec<String> {
        let mut types: Vec<String> = self.loaders.keys().cloned().collect();
        types.sort();
        types
    }

    /// Check whether a given model type is supported.
    pub fn is_supported(&self, model_type: &str) -> bool {
        self.loaders.contains_key(model_type)
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Generic transformer model loader.
///
/// This is the shared loader used for all standard transformer architectures
/// (Llama, Qwen2, Mixtral, DeepSeek). It:
/// 1. Reads `config.json` and converts it to a `TransformerConfig`
/// 2. Loads safetensors weights via `rmlx_serve_weights::load_model`
/// 3. Wraps the loaded model in a `TransformerLlm`
fn load_transformer_model(model_path: &Path, device: GpuDevice) -> Result<Box<dyn LlmModel>> {
    let (model, model_config) = rmlx_serve_weights::load_model(model_path)?;

    let transformer_config = model_config.to_transformer_config()?;

    info!(
        model_type = model_config.model_type.as_str(),
        hidden_size = transformer_config.hidden_size,
        num_layers = transformer_config.num_layers,
        vocab_size = transformer_config.vocab_size,
        "loaded transformer model"
    );

    let llm = TransformerLlm::new(model, transformer_config, device)?;
    Ok(Box::new(llm))
}

/// Duplicate a `TransformerConfig` (needed because `TransformerConfig` does not
/// implement `Clone` — it lives in `rmlx-nn` and we cannot add an orphan impl).
#[allow(dead_code)]
fn duplicate_config(src: &TransformerConfig) -> TransformerConfig {
    let ff_type = match &src.ff_type {
        FeedForwardType::Dense { intermediate_dim } => FeedForwardType::Dense {
            intermediate_dim: *intermediate_dim,
        },
        FeedForwardType::MoE { config } => FeedForwardType::MoE {
            config: MoeConfig {
                num_experts: config.num_experts,
                num_experts_per_token: config.num_experts_per_token,
                hidden_dim: config.hidden_dim,
                intermediate_dim: config.intermediate_dim,
                capacity_factor: config.capacity_factor,
            },
        },
    };
    TransformerConfig {
        hidden_size: src.hidden_size,
        num_heads: src.num_heads,
        num_kv_heads: src.num_kv_heads,
        head_dim: src.head_dim,
        num_layers: src.num_layers,
        vocab_size: src.vocab_size,
        max_seq_len: src.max_seq_len,
        rope_theta: src.rope_theta,
        rms_norm_eps: src.rms_norm_eps,
        ff_type,
    }
}
