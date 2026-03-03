//! High-level model loading entry points.
//!
//! These functions provide the simplest way to load a model from disk.
//! For more control over the loading process (e.g., custom architectures),
//! use [`ModelRegistry`](crate::ModelRegistry) directly.

use std::path::Path;

use rmlx_serve_weights::ModelConfig;
use tracing::info;

use crate::error::{ModelError, Result};
use crate::registry::ModelRegistry;
use crate::traits::LlmModel;

/// Load a model from a directory containing `config.json` and safetensors files.
///
/// This is the primary entry point for model loading. It:
/// 1. Reads `config.json` to determine the model architecture
/// 2. Uses the default `ModelRegistry` to find the appropriate loader
/// 3. Loads weights, builds the model, and precomputes RoPE frequencies
///
/// # Arguments
/// * `path` - Path to a model directory (e.g., a HuggingFace model download).
///
/// # Returns
/// A tuple of `(model, config)` where `model` implements [`LlmModel`] and
/// `config` contains the parsed HuggingFace configuration.
///
/// # Example
/// ```ignore
/// use rmlx_serve_models::load_model;
///
/// let (model, config) = load_model("/path/to/llama-3-8b")?;
/// println!("Loaded {} with {} layers", config.model_type, model.num_layers());
/// ```
pub fn load_model(path: impl AsRef<Path>) -> Result<(Box<dyn LlmModel>, ModelConfig)> {
    let model_path = path.as_ref();

    info!(path = %model_path.display(), "loading model");

    // Parse config first to return it alongside the model
    let config = ModelConfig::from_path(model_path)?;

    info!(
        model_type = config.model_type.as_str(),
        hidden_size = ?config.hidden_size,
        num_layers = ?config.num_hidden_layers,
        vocab_size = ?config.vocab_size,
        "parsed model config"
    );

    // Use the registry to load the model
    let registry = ModelRegistry::new();
    let model = registry.load(model_path)?;

    Ok((model, config))
}

/// Load a model shard for distributed inference.
///
/// Loads only the weights belonging to `shard_id` out of `num_shards` total.
/// Linear layers are split by output dimension across shards.
///
/// # Arguments
/// * `path` - Path to the model directory.
/// * `num_shards` - Total number of shards (GPUs) participating.
/// * `shard_id` - Zero-based index of this shard.
///
/// # Returns
/// A model containing only this shard's portion of the weights.
///
/// # Note
/// This is a placeholder for distributed inference support. Full implementation
/// requires the `rmlx-distributed` crate for cross-device communication.
pub fn sharded_load(
    path: impl AsRef<Path>,
    num_shards: usize,
    shard_id: usize,
) -> Result<Box<dyn LlmModel>> {
    let model_path = path.as_ref();

    info!(
        path = %model_path.display(),
        num_shards = num_shards,
        shard_id = shard_id,
        "sharded model loading requested"
    );

    if num_shards == 0 {
        return Err(ModelError::InvalidConfig("num_shards must be > 0".into()));
    }
    if shard_id >= num_shards {
        return Err(ModelError::InvalidConfig(format!(
            "shard_id ({}) must be < num_shards ({})",
            shard_id, num_shards
        )));
    }

    // For single-shard case, just load normally
    if num_shards == 1 {
        let (model, _config) = load_model(model_path)?;
        return Ok(model);
    }

    // Multi-shard distributed loading is not yet implemented.
    // This requires rmlx-distributed for cross-device tensor parallelism.
    todo!(
        "Distributed model loading requires rmlx-distributed. \
         Requested shard {}/{} for model at {}",
        shard_id,
        num_shards,
        model_path.display()
    )
}
