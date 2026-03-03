//! Prompt cache save/load utilities.
//!
//! Ported from mlx-lm's prompt cache functionality. Enables saving a
//! KV cache to disk and loading it back, so that repeated prompts
//! (e.g., system prompts) do not need to be recomputed.
//!
//! The on-disk format uses safetensors for the array data, with an
//! optional JSON metadata sidecar.

use std::collections::HashMap;
use std::path::Path;

use rmlx_core::{Array, DType};
use rmlx_metal::metal;

use crate::error::{CacheError, Result};
use crate::kv_cache::KVCache;

/// Create a prompt cache suitable for saving/loading.
///
/// If `max_kv_size` is provided, creates a pre-allocated cache.
/// Otherwise, creates a lazy (non-pre-allocated) cache.
///
/// # Arguments
/// * `num_layers` - Number of transformer layers.
/// * `num_kv_heads` - Number of KV heads per layer.
/// * `head_dim` - Dimension of each head.
/// * `max_kv_size` - Optional maximum sequence length for pre-allocation.
/// * `device` - Metal device for buffer allocation.
/// * `dtype` - Data type for the cache arrays.
pub fn make_prompt_cache(
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_kv_size: Option<usize>,
    device: &metal::Device,
    dtype: DType,
) -> KVCache {
    match max_kv_size {
        Some(max_seq) => {
            KVCache::preallocated(device, num_layers, num_kv_heads, head_dim, max_seq, dtype)
        }
        None => KVCache::new(num_layers, num_kv_heads),
    }
}

/// Save a prompt cache to disk.
///
/// Writes the KV cache arrays as a safetensors file. Each array is
/// saved with a name encoding its layer, head, and type (key/value).
///
/// Optional metadata (e.g., model name, tokenizer hash) is stored
/// in the safetensors metadata section.
///
/// # File layout
/// The file at `path` is a safetensors file with tensors named:
/// - `layer_{l}_head_{h}_keys` -- shape `[seq_len, head_dim]`
/// - `layer_{l}_head_{h}_values` -- shape `[seq_len, head_dim]`
///
/// # Errors
/// Returns `CacheError::IoError` if the file cannot be written,
/// or `CacheError::SerializationError` if safetensors encoding fails.
pub fn save_prompt_cache(
    path: impl AsRef<Path>,
    cache: &KVCache,
    metadata: Option<HashMap<String, String>>,
) -> Result<()> {
    let path = path.as_ref();

    if cache.is_empty() {
        tracing::warn!("saving empty prompt cache to {}", path.display());
    }

    let mut tensors: Vec<(String, Vec<u8>, Vec<usize>, safetensors::Dtype)> = Vec::new();

    for (l, layer) in cache.inner.iter().enumerate() {
        let num_heads = layer.keys.len();
        for h in 0..num_heads {
            if layer.seq_len > 0 {
                // Save keys.
                let k = layer.cached_keys(h);
                let k_bytes = k.to_bytes().to_vec();
                let k_shape = k.shape().to_vec();
                let k_name = format!("layer_{l}_head_{h}_keys");
                let st_dtype = dtype_to_safetensors(k.dtype());
                tensors.push((k_name, k_bytes, k_shape, st_dtype));

                // Save values.
                let v = layer.cached_values(h);
                let v_bytes = v.to_bytes().to_vec();
                let v_shape = v.shape().to_vec();
                let v_name = format!("layer_{l}_head_{h}_values");
                tensors.push((v_name, v_bytes, v_shape, st_dtype));
            }
        }
    }

    // Build the safetensors tensor views.
    let tensor_views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|(name, data, shape, dtype)| {
            let view = safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)
                .expect("failed to create tensor view");
            (name.clone(), view)
        })
        .collect();

    let serialized = safetensors::tensor::serialize(
        tensor_views
            .iter()
            .map(|(name, view)| (name.as_str(), view.clone())),
        &metadata,
    )
    .map_err(|e| CacheError::SerializationError(e.to_string()))?;

    std::fs::write(path, serialized).map_err(|e| CacheError::IoError {
        path: path.to_path_buf(),
        source: e,
    })?;

    tracing::info!(
        "saved prompt cache ({} layers, seq_len={}) to {}",
        cache.num_layers(),
        cache.seq_len(),
        path.display()
    );

    Ok(())
}

/// Load a prompt cache from disk.
///
/// Reads a safetensors file and reconstructs the `KVCache`. The device
/// is used to allocate Metal buffers for the loaded arrays.
///
/// # Errors
/// Returns `CacheError::IoError` if the file cannot be read,
/// or `CacheError::SerializationError` if the format is invalid.
pub fn load_prompt_cache(path: impl AsRef<Path>, device: &metal::Device) -> Result<KVCache> {
    let path = path.as_ref();

    let data = std::fs::read(path).map_err(|e| CacheError::IoError {
        path: path.to_path_buf(),
        source: e,
    })?;

    let safetensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| CacheError::SerializationError(e.to_string()))?;

    // Discover the number of layers and heads from tensor names.
    let mut max_layer: usize = 0;
    let mut max_head: usize = 0;

    for name in safetensors.names() {
        // Parse "layer_{l}_head_{h}_keys" or "layer_{l}_head_{h}_values"
        let parts: Vec<&str> = name.split('_').collect();
        if parts.len() >= 4 && parts[0] == "layer" && parts[2] == "head" {
            if let (Ok(l), Ok(h)) = (parts[1].parse::<usize>(), parts[3].parse::<usize>()) {
                max_layer = max_layer.max(l);
                max_head = max_head.max(h);
            }
        }
    }

    let num_layers = max_layer + 1;
    let num_kv_heads = max_head + 1;

    // Create the cache with lazy allocation (we will populate it from the file).
    let mut cache = KVCache::new(num_layers, num_kv_heads);

    // Load each layer's keys and values.
    for l in 0..num_layers {
        let layer = cache.layer_mut(l);

        for h in 0..num_kv_heads {
            let k_name = format!("layer_{l}_head_{h}_keys");
            let v_name = format!("layer_{l}_head_{h}_values");

            if let (Ok(k_tensor), Ok(v_tensor)) =
                (safetensors.tensor(&k_name), safetensors.tensor(&v_name))
            {
                let k_dtype = safetensors_to_dtype(k_tensor.dtype());
                let k_shape: Vec<usize> = k_tensor.shape().to_vec();
                let k_array = Array::from_bytes(device, k_tensor.data(), k_shape.clone(), k_dtype);

                let v_dtype = safetensors_to_dtype(v_tensor.dtype());
                let v_shape: Vec<usize> = v_tensor.shape().to_vec();
                let v_array = Array::from_bytes(device, v_tensor.data(), v_shape, v_dtype);

                layer.keys.push(k_array);
                layer.values.push(v_array);

                // Update seq_len from the shape (seq dimension is axis 0).
                if h == 0 && !k_shape.is_empty() {
                    layer.seq_len = k_shape[0];
                }
            }
        }
    }

    tracing::info!(
        "loaded prompt cache ({} layers, seq_len={}) from {}",
        num_layers,
        cache.seq_len(),
        path.display()
    );

    Ok(cache)
}

/// Trim the prompt cache by removing the last `n` tokens from all layers.
///
/// This modifies the `seq_len` of each layer, effectively making the
/// most recently appended tokens invisible. For pre-allocated caches,
/// the underlying buffers are not freed (they can be reused by the
/// next append).
pub fn trim_prompt_cache(cache: &mut KVCache, n: usize) {
    for layer in cache.layers_mut() {
        if layer.seq_len >= n {
            layer.seq_len -= n;
        } else {
            layer.seq_len = 0;
        }
    }
}

/// Check whether the prompt cache can be trimmed.
///
/// Returns `true` if any layer has a non-zero sequence length.
pub fn can_trim_prompt_cache(cache: &KVCache) -> bool {
    cache.layers().iter().any(|layer| layer.seq_len > 0)
}

/// Convert `rmlx_core::DType` to `safetensors::Dtype`.
fn dtype_to_safetensors(dtype: DType) -> safetensors::Dtype {
    match dtype {
        DType::Float32 => safetensors::Dtype::F32,
        DType::Float16 => safetensors::Dtype::F16,
        DType::Bfloat16 => safetensors::Dtype::BF16,
        DType::UInt32 => safetensors::Dtype::U32,
        // For quantized and fp8 types, we store raw bytes as U8.
        DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Float8E4M3 | DType::Float8E5M2 => {
            safetensors::Dtype::U8
        }
    }
}

/// Convert `safetensors::Dtype` to `rmlx_core::DType`.
fn safetensors_to_dtype(dtype: safetensors::Dtype) -> DType {
    match dtype {
        safetensors::Dtype::F32 => DType::Float32,
        safetensors::Dtype::F16 => DType::Float16,
        safetensors::Dtype::BF16 => DType::Bfloat16,
        safetensors::Dtype::U32 => DType::UInt32,
        safetensors::Dtype::U8 => DType::Float32, // fallback; quantized types need separate handling
        _ => DType::Float32,                      // conservative default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_prompt_cache() {
        let mut cache = KVCache::new(2, 4);
        // Manually set seq_len for testing.
        for layer in cache.layers_mut() {
            layer.seq_len = 10;
        }

        assert!(can_trim_prompt_cache(&cache));

        trim_prompt_cache(&mut cache, 3);
        assert_eq!(cache.seq_len(), 7);

        trim_prompt_cache(&mut cache, 100);
        assert_eq!(cache.seq_len(), 0);
        assert!(!can_trim_prompt_cache(&cache));
    }

    #[test]
    fn test_dtype_conversion_roundtrip() {
        let types = [
            DType::Float32,
            DType::Float16,
            DType::Bfloat16,
            DType::UInt32,
        ];
        for dtype in &types {
            let st = dtype_to_safetensors(*dtype);
            let back = safetensors_to_dtype(st);
            assert_eq!(*dtype, back, "roundtrip failed for {dtype:?}");
        }
    }
}
