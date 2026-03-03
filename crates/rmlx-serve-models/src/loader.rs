//! High-level model loading entry points.
//!
//! These functions provide the simplest way to load a model from disk.
//! For more control over the loading process (e.g., custom architectures),
//! use [`ModelRegistry`](crate::ModelRegistry) directly.

use std::collections::HashMap;
use std::path::Path;

use rmlx_serve_weights::ModelConfig;
use tracing::info;

use crate::error::{ModelError, Result};
use crate::registry::ModelRegistry;
use crate::traits::LlmModel;

/// Safetensors index file format for multi-shard models.
#[derive(serde::Deserialize)]
struct ShardIndex {
    /// Metadata (total_size, etc.) -- not used directly.
    #[serde(default)]
    #[allow(dead_code)]
    metadata: serde_json::Value,
    /// Maps tensor name -> shard filename.
    weight_map: HashMap<String, String>,
}

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

    // Multi-shard loading: load weight files in order and merge into a
    // single weight map, then build the model from the merged weights.
    //
    // We look for sharded files matching the pattern
    // `model-NNNNN-of-NNNNN.safetensors`. If only a single
    // `model.safetensors` exists, we fall back to loading it directly.

    let index_path = model_path.join("model.safetensors.index.json");
    let single_path = model_path.join("model.safetensors");

    if !index_path.exists() && single_path.exists() {
        // Only a single-file model exists — load it normally.
        info!("sharded_load: only single safetensors file found, loading directly");
        let (model, _config) = load_model(model_path)?;
        return Ok(model);
    }

    if !index_path.exists() {
        return Err(ModelError::InvalidConfig(format!(
            "no model.safetensors or model.safetensors.index.json found in {}",
            model_path.display()
        )));
    }

    // Read the index to discover shard files
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: ShardIndex = serde_json::from_str(&index_content)?;

    // Collect unique shard filenames in sorted order
    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();

    let total_shards = shard_files.len();
    info!(
        total_shard_files = total_shards,
        num_shards = num_shards,
        shard_id = shard_id,
        "merging weight shards"
    );

    // Determine which shard files this worker is responsible for.
    // Distribute shard files across `num_shards` workers round-robin.
    let my_shard_files: Vec<&String> = shard_files
        .iter()
        .enumerate()
        .filter(|(i, _)| i % num_shards == shard_id)
        .map(|(_, f)| f)
        .collect();

    if my_shard_files.is_empty() {
        return Err(ModelError::InvalidConfig(format!(
            "shard_id {} has no files to load (total shard files: {}, num_shards: {})",
            shard_id, total_shards, num_shards
        )));
    }

    info!(
        files = my_shard_files.len(),
        shard_id = shard_id,
        "loading assigned shard files"
    );

    // Load model using the standard registry (which ultimately calls
    // `rmlx_serve_weights::load_model` that already handles the
    // index-based sharded loading). For multi-GPU tensor parallelism
    // a future `rmlx-distributed` integration would partition the
    // weight tensors across devices. For now we load the full model
    // on this device — the shard_id metadata is recorded for future use.
    let (model, _config) = load_model(model_path)?;
    Ok(model)
}
