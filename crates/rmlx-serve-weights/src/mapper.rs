//! Weight name mapping from HuggingFace naming conventions to RMLX components.
//!
//! Each model family (Llama, Qwen2, Mixtral, DeepSeek) uses slightly different
//! naming for its weight tensors. The `WeightMapper` trait abstracts this so the
//! loader can work with any supported architecture.

use tracing::trace;

/// Identifies which component of the model a weight tensor belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightComponent {
    /// Token embedding table: [vocab_size, hidden_size]
    Embedding,
    /// Language model output head: [vocab_size, hidden_size]
    LmHead,
    /// Final RMS normalization weight: [hidden_size]
    FinalNorm,
    /// Pre-attention RMS norm for a given layer: [hidden_size]
    AttnNorm { layer: usize },
    /// Pre-FFN RMS norm for a given layer: [hidden_size]
    FfnNorm { layer: usize },
    /// Query projection: [num_heads * head_dim, hidden_size]
    QProj { layer: usize },
    /// Key projection: [num_kv_heads * head_dim, hidden_size]
    KProj { layer: usize },
    /// Value projection: [num_kv_heads * head_dim, hidden_size]
    VProj { layer: usize },
    /// Output projection: [hidden_size, num_heads * head_dim]
    OProj { layer: usize },
    /// SwiGLU gate projection: [intermediate_size, hidden_size]
    GateProj { layer: usize },
    /// SwiGLU up projection: [intermediate_size, hidden_size]
    UpProj { layer: usize },
    /// SwiGLU down projection: [hidden_size, intermediate_size]
    DownProj { layer: usize },
    /// MoE router / gate weight: [num_experts, hidden_size]
    GateWeight { layer: usize },
    /// MoE expert gate projection: [intermediate_size, hidden_size]
    ExpertGateProj { layer: usize, expert: usize },
    /// MoE expert up projection: [intermediate_size, hidden_size]
    ExpertUpProj { layer: usize, expert: usize },
    /// MoE expert down projection: [hidden_size, intermediate_size]
    ExpertDownProj { layer: usize, expert: usize },
}

/// A resolved weight mapping: the component and optional layer index.
#[derive(Debug, Clone)]
pub struct MappedWeight {
    /// Layer index (None for embedding, lm_head, final_norm).
    pub layer_idx: Option<usize>,
    /// Which model component this weight belongs to.
    pub component: WeightComponent,
}

/// Trait for mapping HuggingFace weight names to RMLX model components.
pub trait WeightMapper: Send + Sync {
    /// Map a HuggingFace tensor name to its RMLX component.
    ///
    /// Returns `None` if the tensor is not recognized or should be skipped
    /// (e.g., rotary_emb.inv_freq).
    fn map_name(&self, hf_name: &str) -> Option<MappedWeight>;

    /// The model type this mapper handles (e.g., "llama", "qwen2").
    fn model_type(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Helper: parse layer index from a dotted path like "model.layers.5.self_attn..."
// ---------------------------------------------------------------------------

/// Extract a layer index from a segment like "layers.{i}" in a dotted name.
fn parse_layer_index(name: &str, prefix: &str) -> Option<usize> {
    let rest = name.strip_prefix(prefix)?;
    let dot_pos = rest.find('.')?;
    rest[..dot_pos].parse().ok()
}

/// Extract an expert index from a segment like "experts.{e}" in the remainder.
fn parse_expert_index(remainder: &str) -> Option<(usize, &str)> {
    let rest = remainder.strip_prefix("experts.")?;
    let dot_pos = rest.find('.')?;
    let idx: usize = rest[..dot_pos].parse().ok()?;
    Some((idx, &rest[dot_pos + 1..]))
}

// ===========================================================================
// Llama weight mapper
// ===========================================================================

/// Weight mapper for Llama / Llama 2 / Llama 3 models.
///
/// HF naming convention:
/// - `model.embed_tokens.weight`
/// - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
/// - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
/// - `model.layers.{i}.input_layernorm.weight`
/// - `model.layers.{i}.post_attention_layernorm.weight`
/// - `model.norm.weight`
/// - `lm_head.weight`
pub struct LlamaWeightMapper;

impl WeightMapper for LlamaWeightMapper {
    fn map_name(&self, hf_name: &str) -> Option<MappedWeight> {
        map_llama_style(hf_name)
    }

    fn model_type(&self) -> &str {
        "llama"
    }
}

/// Shared Llama-style mapping used by Llama and Qwen2.
fn map_llama_style(hf_name: &str) -> Option<MappedWeight> {
    // Embedding
    if hf_name == "model.embed_tokens.weight" {
        return Some(MappedWeight {
            layer_idx: None,
            component: WeightComponent::Embedding,
        });
    }

    // LM head
    if hf_name == "lm_head.weight" {
        return Some(MappedWeight {
            layer_idx: None,
            component: WeightComponent::LmHead,
        });
    }

    // Final norm
    if hf_name == "model.norm.weight" {
        return Some(MappedWeight {
            layer_idx: None,
            component: WeightComponent::FinalNorm,
        });
    }

    // Layer-specific weights
    if let Some(layer) = parse_layer_index(hf_name, "model.layers.") {
        let layer_prefix = format!("model.layers.{layer}.");
        let suffix = hf_name.strip_prefix(&layer_prefix)?;

        let component = match suffix {
            // Attention projections
            "self_attn.q_proj.weight" => WeightComponent::QProj { layer },
            "self_attn.k_proj.weight" => WeightComponent::KProj { layer },
            "self_attn.v_proj.weight" => WeightComponent::VProj { layer },
            "self_attn.o_proj.weight" => WeightComponent::OProj { layer },

            // Dense FFN projections
            "mlp.gate_proj.weight" => WeightComponent::GateProj { layer },
            "mlp.up_proj.weight" => WeightComponent::UpProj { layer },
            "mlp.down_proj.weight" => WeightComponent::DownProj { layer },

            // Layer norms
            "input_layernorm.weight" => WeightComponent::AttnNorm { layer },
            "post_attention_layernorm.weight" => WeightComponent::FfnNorm { layer },

            _ => {
                trace!(name = hf_name, "skipping unrecognized llama weight");
                return None;
            }
        };

        return Some(MappedWeight {
            layer_idx: Some(layer),
            component,
        });
    }

    trace!(name = hf_name, "skipping unrecognized weight");
    None
}

// ===========================================================================
// Qwen2 weight mapper
// ===========================================================================

/// Weight mapper for Qwen2 models.
///
/// Qwen2 uses the same naming convention as Llama.
pub struct QwenWeightMapper;

impl WeightMapper for QwenWeightMapper {
    fn map_name(&self, hf_name: &str) -> Option<MappedWeight> {
        map_llama_style(hf_name)
    }

    fn model_type(&self) -> &str {
        "qwen2"
    }
}

// ===========================================================================
// Mixtral weight mapper
// ===========================================================================

/// Weight mapper for Mixtral (Mixture of Experts) models.
///
/// Like Llama for attention, but the FFN uses a sparse MoE block:
/// - `model.layers.{i}.block_sparse_moe.gate.weight`       (router)
/// - `model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight`  (gate_proj)
/// - `model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight`  (down_proj)
/// - `model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight`  (up_proj)
pub struct MixtralWeightMapper;

impl WeightMapper for MixtralWeightMapper {
    fn map_name(&self, hf_name: &str) -> Option<MappedWeight> {
        // Non-layer weights are identical to Llama
        if hf_name == "model.embed_tokens.weight" {
            return Some(MappedWeight {
                layer_idx: None,
                component: WeightComponent::Embedding,
            });
        }
        if hf_name == "lm_head.weight" {
            return Some(MappedWeight {
                layer_idx: None,
                component: WeightComponent::LmHead,
            });
        }
        if hf_name == "model.norm.weight" {
            return Some(MappedWeight {
                layer_idx: None,
                component: WeightComponent::FinalNorm,
            });
        }

        if let Some(layer) = parse_layer_index(hf_name, "model.layers.") {
            let layer_prefix = format!("model.layers.{layer}.");
            let suffix = hf_name.strip_prefix(&layer_prefix)?;

            let component = match suffix {
                // Attention projections (same as Llama)
                "self_attn.q_proj.weight" => WeightComponent::QProj { layer },
                "self_attn.k_proj.weight" => WeightComponent::KProj { layer },
                "self_attn.v_proj.weight" => WeightComponent::VProj { layer },
                "self_attn.o_proj.weight" => WeightComponent::OProj { layer },

                // Layer norms (same as Llama)
                "input_layernorm.weight" => WeightComponent::AttnNorm { layer },
                "post_attention_layernorm.weight" => WeightComponent::FfnNorm { layer },

                // MoE gate (router)
                "block_sparse_moe.gate.weight" => WeightComponent::GateWeight { layer },

                _ => {
                    // MoE expert weights: block_sparse_moe.experts.{e}.{w1,w2,w3}.weight
                    if let Some(expert_rest) =
                        suffix.strip_prefix("block_sparse_moe.")
                    {
                        if let Some((expert, weight_name)) = parse_expert_index(expert_rest) {
                            match weight_name {
                                "w1.weight" => {
                                    WeightComponent::ExpertGateProj { layer, expert }
                                }
                                "w2.weight" => {
                                    WeightComponent::ExpertDownProj { layer, expert }
                                }
                                "w3.weight" => {
                                    WeightComponent::ExpertUpProj { layer, expert }
                                }
                                _ => {
                                    trace!(
                                        name = hf_name,
                                        "skipping unrecognized mixtral expert weight"
                                    );
                                    return None;
                                }
                            }
                        } else {
                            trace!(
                                name = hf_name,
                                "skipping unrecognized mixtral MoE weight"
                            );
                            return None;
                        }
                    } else {
                        trace!(name = hf_name, "skipping unrecognized mixtral weight");
                        return None;
                    }
                }
            };

            return Some(MappedWeight {
                layer_idx: Some(layer),
                component,
            });
        }

        trace!(name = hf_name, "skipping unrecognized weight");
        None
    }

    fn model_type(&self) -> &str {
        "mixtral"
    }
}

// ===========================================================================
// DeepSeek weight mapper
// ===========================================================================

/// Weight mapper for DeepSeek V2 / V3 MoE models.
///
/// Similar to Mixtral but uses different MoE naming:
/// - `model.layers.{i}.mlp.gate.weight`                    (router)
/// - `model.layers.{i}.mlp.experts.{e}.gate_proj.weight`   (gate_proj)
/// - `model.layers.{i}.mlp.experts.{e}.up_proj.weight`     (up_proj)
/// - `model.layers.{i}.mlp.experts.{e}.down_proj.weight`   (down_proj)
///
/// Some DeepSeek layers are dense (no MoE), using standard mlp.{gate,up,down}_proj.
pub struct DeepSeekWeightMapper;

impl WeightMapper for DeepSeekWeightMapper {
    fn map_name(&self, hf_name: &str) -> Option<MappedWeight> {
        // Non-layer weights
        if hf_name == "model.embed_tokens.weight" {
            return Some(MappedWeight {
                layer_idx: None,
                component: WeightComponent::Embedding,
            });
        }
        if hf_name == "lm_head.weight" {
            return Some(MappedWeight {
                layer_idx: None,
                component: WeightComponent::LmHead,
            });
        }
        if hf_name == "model.norm.weight" {
            return Some(MappedWeight {
                layer_idx: None,
                component: WeightComponent::FinalNorm,
            });
        }

        if let Some(layer) = parse_layer_index(hf_name, "model.layers.") {
            let layer_prefix = format!("model.layers.{layer}.");
            let suffix = hf_name.strip_prefix(&layer_prefix)?;

            let component = match suffix {
                // Attention (same as Llama)
                "self_attn.q_proj.weight" => WeightComponent::QProj { layer },
                "self_attn.k_proj.weight" => WeightComponent::KProj { layer },
                "self_attn.v_proj.weight" => WeightComponent::VProj { layer },
                "self_attn.o_proj.weight" => WeightComponent::OProj { layer },

                // Layer norms
                "input_layernorm.weight" => WeightComponent::AttnNorm { layer },
                "post_attention_layernorm.weight" => WeightComponent::FfnNorm { layer },

                // Dense MLP (for non-MoE layers in DeepSeek)
                "mlp.gate_proj.weight" => WeightComponent::GateProj { layer },
                "mlp.up_proj.weight" => WeightComponent::UpProj { layer },
                "mlp.down_proj.weight" => WeightComponent::DownProj { layer },

                // MoE gate (router)
                "mlp.gate.weight" => WeightComponent::GateWeight { layer },

                _ => {
                    // MoE expert weights: mlp.experts.{e}.{gate,up,down}_proj.weight
                    if let Some(expert_rest) = suffix.strip_prefix("mlp.") {
                        if let Some((expert, weight_name)) = parse_expert_index(expert_rest) {
                            match weight_name {
                                "gate_proj.weight" => {
                                    WeightComponent::ExpertGateProj { layer, expert }
                                }
                                "up_proj.weight" => {
                                    WeightComponent::ExpertUpProj { layer, expert }
                                }
                                "down_proj.weight" => {
                                    WeightComponent::ExpertDownProj { layer, expert }
                                }
                                _ => {
                                    trace!(
                                        name = hf_name,
                                        "skipping unrecognized deepseek expert weight"
                                    );
                                    return None;
                                }
                            }
                        } else {
                            trace!(
                                name = hf_name,
                                "skipping unrecognized deepseek MLP weight"
                            );
                            return None;
                        }
                    } else {
                        trace!(name = hf_name, "skipping unrecognized deepseek weight");
                        return None;
                    }
                }
            };

            return Some(MappedWeight {
                layer_idx: Some(layer),
                component,
            });
        }

        trace!(name = hf_name, "skipping unrecognized weight");
        None
    }

    fn model_type(&self) -> &str {
        "deepseek"
    }
}

// ===========================================================================
// Factory function
// ===========================================================================

/// Create the appropriate weight mapper for a given model type string.
///
/// Recognized model types:
/// - "llama" / "llama2" / "llama3" / "codellama" -> LlamaWeightMapper
/// - "qwen2" / "qwen2_moe" -> QwenWeightMapper
/// - "mixtral" -> MixtralWeightMapper
/// - "deepseek" / "deepseek_v2" / "deepseek_v3" -> DeepSeekWeightMapper
///
/// Falls back to LlamaWeightMapper for unrecognized types, since many models
/// follow the Llama naming convention.
pub fn create_weight_mapper(model_type: &str) -> Box<dyn WeightMapper> {
    match model_type.to_lowercase().as_str() {
        "llama" | "llama2" | "llama3" | "codellama" => Box::new(LlamaWeightMapper),
        "qwen2" | "qwen2_moe" => Box::new(QwenWeightMapper),
        "mixtral" => Box::new(MixtralWeightMapper),
        "deepseek" | "deepseek_v2" | "deepseek_v3" => Box::new(DeepSeekWeightMapper),
        _ => {
            tracing::warn!(
                model_type = model_type,
                "unrecognized model type, falling back to Llama weight mapper"
            );
            Box::new(LlamaWeightMapper)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_mapper_embedding() {
        let mapper = LlamaWeightMapper;
        let mapped = mapper.map_name("model.embed_tokens.weight").unwrap();
        assert_eq!(mapped.component, WeightComponent::Embedding);
        assert_eq!(mapped.layer_idx, None);
    }

    #[test]
    fn test_llama_mapper_lm_head() {
        let mapper = LlamaWeightMapper;
        let mapped = mapper.map_name("lm_head.weight").unwrap();
        assert_eq!(mapped.component, WeightComponent::LmHead);
    }

    #[test]
    fn test_llama_mapper_attention() {
        let mapper = LlamaWeightMapper;

        let mapped = mapper
            .map_name("model.layers.5.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(mapped.component, WeightComponent::QProj { layer: 5 });
        assert_eq!(mapped.layer_idx, Some(5));

        let mapped = mapper
            .map_name("model.layers.0.self_attn.k_proj.weight")
            .unwrap();
        assert_eq!(mapped.component, WeightComponent::KProj { layer: 0 });
    }

    #[test]
    fn test_llama_mapper_ffn() {
        let mapper = LlamaWeightMapper;

        let mapped = mapper
            .map_name("model.layers.3.mlp.gate_proj.weight")
            .unwrap();
        assert_eq!(mapped.component, WeightComponent::GateProj { layer: 3 });
    }

    #[test]
    fn test_llama_mapper_norms() {
        let mapper = LlamaWeightMapper;

        let mapped = mapper.map_name("model.norm.weight").unwrap();
        assert_eq!(mapped.component, WeightComponent::FinalNorm);

        let mapped = mapper
            .map_name("model.layers.2.input_layernorm.weight")
            .unwrap();
        assert_eq!(mapped.component, WeightComponent::AttnNorm { layer: 2 });

        let mapped = mapper
            .map_name("model.layers.2.post_attention_layernorm.weight")
            .unwrap();
        assert_eq!(mapped.component, WeightComponent::FfnNorm { layer: 2 });
    }

    #[test]
    fn test_llama_mapper_skips_inv_freq() {
        let mapper = LlamaWeightMapper;
        assert!(mapper
            .map_name("model.layers.0.self_attn.rotary_emb.inv_freq")
            .is_none());
    }

    #[test]
    fn test_mixtral_mapper_moe_gate() {
        let mapper = MixtralWeightMapper;

        let mapped = mapper
            .map_name("model.layers.1.block_sparse_moe.gate.weight")
            .unwrap();
        assert_eq!(mapped.component, WeightComponent::GateWeight { layer: 1 });
    }

    #[test]
    fn test_mixtral_mapper_expert_weights() {
        let mapper = MixtralWeightMapper;

        let mapped = mapper
            .map_name("model.layers.2.block_sparse_moe.experts.3.w1.weight")
            .unwrap();
        assert_eq!(
            mapped.component,
            WeightComponent::ExpertGateProj {
                layer: 2,
                expert: 3
            }
        );

        let mapped = mapper
            .map_name("model.layers.2.block_sparse_moe.experts.3.w2.weight")
            .unwrap();
        assert_eq!(
            mapped.component,
            WeightComponent::ExpertDownProj {
                layer: 2,
                expert: 3
            }
        );

        let mapped = mapper
            .map_name("model.layers.2.block_sparse_moe.experts.3.w3.weight")
            .unwrap();
        assert_eq!(
            mapped.component,
            WeightComponent::ExpertUpProj {
                layer: 2,
                expert: 3
            }
        );
    }

    #[test]
    fn test_deepseek_mapper_moe() {
        let mapper = DeepSeekWeightMapper;

        let mapped = mapper
            .map_name("model.layers.4.mlp.gate.weight")
            .unwrap();
        assert_eq!(mapped.component, WeightComponent::GateWeight { layer: 4 });

        let mapped = mapper
            .map_name("model.layers.4.mlp.experts.7.gate_proj.weight")
            .unwrap();
        assert_eq!(
            mapped.component,
            WeightComponent::ExpertGateProj {
                layer: 4,
                expert: 7
            }
        );
    }

    #[test]
    fn test_create_weight_mapper() {
        assert_eq!(create_weight_mapper("llama").model_type(), "llama");
        assert_eq!(create_weight_mapper("qwen2").model_type(), "qwen2");
        assert_eq!(create_weight_mapper("mixtral").model_type(), "mixtral");
        assert_eq!(create_weight_mapper("deepseek_v2").model_type(), "deepseek");
        // Fallback
        assert_eq!(create_weight_mapper("unknown_model").model_type(), "llama");
    }
}
