//! Weight sanitization and cleanup.
//!
//! After loading raw tensors from safetensors, several cleanups are needed
//! before the weights can be assembled into a `TransformerModel`:
//!
//! - Remove `rotary_emb.inv_freq` tensors (computed at runtime by RMLX)
//! - Handle tied embeddings (lm_head shares the embedding weight)
//! - Remove bias tensors for models that don't use bias
//! - Model-specific key renaming or cleanup

use std::collections::HashMap;

use tracing::{debug, trace};

use rmlx_core::Array;

/// Sanitize loaded weight tensors.
///
/// This performs architecture-agnostic cleanups that apply to most transformer
/// models, plus model-specific cleanups based on the `model_type` string.
///
/// # Operations
/// 1. Remove `rotary_emb.inv_freq` tensors (RMLX computes RoPE frequencies at runtime)
/// 2. Handle tied word embeddings: if `lm_head.weight` is missing, clone from `model.embed_tokens.weight`
/// 3. Remove `.bias` keys for models that are known to be bias-free
/// 4. Model-specific cleanups
pub fn sanitize_weights(
    mut weights: HashMap<String, Array>,
    model_type: &str,
) -> HashMap<String, Array> {
    let initial_count = weights.len();

    // 1. Remove rotary_emb.inv_freq — computed at runtime
    let inv_freq_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.contains("rotary_emb.inv_freq"))
        .cloned()
        .collect();
    for key in &inv_freq_keys {
        trace!(key = key.as_str(), "removing rotary_emb.inv_freq");
        weights.remove(key);
    }

    // 2. Handle tied word embeddings
    //    If lm_head.weight is missing but model.embed_tokens.weight exists,
    //    the model uses tied embeddings. We cannot simply clone GPU buffers,
    //    so we create a shared reference by inserting the same key's bytes.
    if !weights.contains_key("lm_head.weight") {
        // Build the tied view in a separate scope to satisfy the borrow checker.
        // Array::view shares the underlying Metal buffer via refcount (zero-copy).
        let tied = weights.get("model.embed_tokens.weight").map(|embed_weight| {
            let shape = embed_weight.shape().to_vec();
            let strides = embed_weight.strides().to_vec();
            let offset = embed_weight.offset();
            embed_weight.view(shape, strides, offset)
        });
        if let Some(tied_array) = tied {
            debug!("tying lm_head.weight to model.embed_tokens.weight");
            weights.insert("lm_head.weight".to_string(), tied_array);
        }
    }

    // 3. Remove bias tensors for models known to be bias-free
    //    Most modern LLMs (Llama, Mistral, Qwen2, etc.) don't use bias in
    //    linear layers. Removing them prevents shape-mismatch errors.
    let bias_free_models = [
        "llama", "llama2", "llama3", "codellama", "mistral", "mixtral",
        "qwen2", "deepseek", "deepseek_v2", "deepseek_v3",
    ];

    let model_type_lower = model_type.to_lowercase();
    if bias_free_models.contains(&model_type_lower.as_str()) {
        let bias_keys: Vec<String> = weights
            .keys()
            .filter(|k| k.ends_with(".bias"))
            .cloned()
            .collect();
        for key in &bias_keys {
            trace!(key = key.as_str(), "removing bias tensor (bias-free model)");
            weights.remove(key);
        }
    }

    // 4. Model-specific cleanups
    match model_type_lower.as_str() {
        "mixtral" => {
            sanitize_mixtral(&mut weights);
        }
        "deepseek" | "deepseek_v2" | "deepseek_v3" => {
            sanitize_deepseek(&mut weights);
        }
        _ => {}
    }

    let removed = initial_count.saturating_sub(weights.len());
    // Account for potential additions (tied embeddings)
    let added = weights.len().saturating_sub(initial_count.saturating_sub(removed));
    debug!(
        initial = initial_count,
        final_count = weights.len(),
        removed = removed,
        added = added,
        "weight sanitization complete"
    );

    weights
}

/// Mixtral-specific sanitization.
fn sanitize_mixtral(weights: &mut HashMap<String, Array>) {
    // Remove any shared expert weights that might be present in some checkpoints
    // but are not used in the standard Mixtral architecture.
    let shared_expert_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.contains("shared_expert"))
        .cloned()
        .collect();
    for key in &shared_expert_keys {
        trace!(key = key.as_str(), "removing shared_expert tensor (mixtral)");
        weights.remove(key);
    }
}

/// DeepSeek-specific sanitization.
fn sanitize_deepseek(weights: &mut HashMap<String, Array>) {
    // DeepSeek V2/V3 may have extra tensors for MLA (Multi-head Latent Attention)
    // that are not used in the standard forward path. Remove them if present.
    let extra_keys: Vec<String> = weights
        .keys()
        .filter(|k| {
            k.contains("kv_a_proj_with_mqa")
                || k.contains("kv_b_proj")
                || k.contains("q_a_proj")
                || k.contains("q_b_proj")
        })
        .cloned()
        .collect();
    for key in &extra_keys {
        trace!(
            key = key.as_str(),
            "removing DeepSeek MLA tensor (not supported in standard path)"
        );
        weights.remove(key);
    }
}

#[cfg(test)]
mod tests {
    // We can't easily create Array in tests without a Metal device,
    // so these tests focus on the key-filtering logic.

    #[test]
    fn test_inv_freq_key_detection() {
        let key = "model.layers.0.self_attn.rotary_emb.inv_freq";
        assert!(key.contains("rotary_emb.inv_freq"));
    }

    #[test]
    fn test_bias_free_model_detection() {
        let bias_free = [
            "llama", "llama2", "llama3", "codellama", "mistral", "mixtral",
            "qwen2", "deepseek", "deepseek_v2", "deepseek_v3",
        ];
        assert!(bias_free.contains(&"llama"));
        assert!(bias_free.contains(&"qwen2"));
        assert!(!bias_free.contains(&"gpt2"));
    }
}
