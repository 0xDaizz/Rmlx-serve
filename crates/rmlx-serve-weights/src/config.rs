//! Model configuration parsed from HuggingFace config.json files.
//!
//! Supports Llama, Qwen2, Mixtral, DeepSeek, and other transformer architectures
//! that share the standard HF config schema.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::debug;

use rmlx_core::DType;
use rmlx_nn::{FeedForwardType, MoeConfig, TransformerConfig};

use crate::error::{Result, WeightError};

/// Convert a PyTorch dtype string (from HuggingFace config.json) to an RMLX DType.
///
/// Handles common aliases used in HuggingFace model configs. Returns `None` for
/// types that have no RMLX equivalent (e.g., integer types not supported on the
/// Metal compute backend).
pub fn torch_dtype_to_rmlx(dtype_str: &str) -> Option<DType> {
    match dtype_str {
        "float16" | "fp16" => Some(DType::Float16),
        "bfloat16" | "bf16" => Some(DType::Bfloat16),
        "float32" | "fp32" | "float" => Some(DType::Float32),
        // Integer types: rmlx-core currently only exposes UInt32.
        // Int8, UInt8, Int16, UInt16, and Int32 are not available as DType variants
        // in rmlx-core's Metal-backed array system.
        "uint32" | "u32" => Some(DType::UInt32),
        _ => None,
    }
}

/// Quantization configuration embedded in config.json (AWQ, GPTQ, etc.).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QuantizationConfig {
    /// Quantization method name (e.g., "awq", "gptq", "bitsandbytes").
    pub quant_method: String,
    /// Number of bits per weight (typically 4 or 8).
    pub bits: Option<usize>,
    /// Group size for block quantization (typically 128).
    pub group_size: Option<usize>,
}

/// Model configuration parsed from a HuggingFace config.json file.
///
/// Fields are largely optional to accommodate the many variations across model
/// families. `model_type` is the one required field that determines which weight
/// mapper and architecture-specific logic to use.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    /// Model family identifier (e.g., "llama", "qwen2", "mixtral", "deepseek_v2").
    pub model_type: String,

    /// Hidden dimension of the transformer.
    #[serde(default)]
    pub hidden_size: Option<usize>,

    /// Number of transformer layers.
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,

    /// Number of attention heads.
    #[serde(default)]
    pub num_attention_heads: Option<usize>,

    /// Number of key-value heads (for GQA). Defaults to num_attention_heads.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Intermediate (FFN) dimension.
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// Vocabulary size.
    #[serde(default)]
    pub vocab_size: Option<usize>,

    /// Maximum position embeddings (context length).
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    /// Base frequency for RoPE.
    #[serde(default)]
    pub rope_theta: Option<f64>,

    /// Epsilon for RMS normalization.
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,

    /// Per-head dimension (overrides hidden_size / num_attention_heads).
    #[serde(default)]
    pub head_dim: Option<usize>,

    /// Number of experts in MoE models.
    #[serde(default)]
    pub num_local_experts: Option<usize>,

    /// Number of experts selected per token in MoE models.
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,

    /// Embedded quantization configuration (AWQ, GPTQ, etc.).
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,

    /// Whether lm_head shares weights with the embedding table.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,

    /// RoPE scaling configuration (varies by model family).
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    /// List of architecture class names (e.g., ["LlamaForCausalLM"]).
    #[serde(default)]
    pub architectures: Option<Vec<String>>,

    /// PyTorch dtype string (e.g., "float16", "bfloat16").
    #[serde(default)]
    pub torch_dtype: Option<String>,

    /// Whether the model uses attention bias (q/k/v projections).
    /// Some models like Qwen2 use bias, while Llama/Mistral don't.
    #[serde(default)]
    pub attention_bias: Option<bool>,
}

impl ModelConfig {
    /// Load a ModelConfig from a directory containing config.json.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let config_path = path.as_ref().join("config.json");
        if !config_path.exists() {
            return Err(WeightError::ConfigNotFound(config_path));
        }

        let content = std::fs::read_to_string(&config_path)?;
        let config: ModelConfig = serde_json::from_str(&content).map_err(|e| {
            WeightError::InvalidConfig(format!("failed to parse config.json: {e}"))
        })?;

        debug!(
            model_type = %config.model_type,
            hidden_size = ?config.hidden_size,
            num_layers = ?config.num_hidden_layers,
            num_heads = ?config.num_attention_heads,
            vocab_size = ?config.vocab_size,
            "loaded model config"
        );

        Ok(config)
    }

    /// Effective number of key-value heads (falls back to num_attention_heads).
    pub fn effective_num_kv_heads(&self) -> usize {
        self.num_key_value_heads
            .or(self.num_attention_heads)
            .unwrap_or(1)
    }

    /// Effective per-head dimension (falls back to hidden_size / num_attention_heads).
    pub fn effective_head_dim(&self) -> usize {
        if let Some(hd) = self.head_dim {
            return hd;
        }
        let hidden = self.hidden_size.unwrap_or(0);
        let heads = self.num_attention_heads.unwrap_or(1);
        if heads == 0 {
            return 0;
        }
        hidden / heads
    }

    /// Get the model's preferred DType based on `torch_dtype` from config.json.
    ///
    /// Returns `None` if `torch_dtype` is not set or maps to an unsupported type.
    pub fn dtype(&self) -> Option<DType> {
        self.torch_dtype
            .as_deref()
            .and_then(torch_dtype_to_rmlx)
    }

    /// Convert this HuggingFace config into the RMLX TransformerConfig.
    pub fn to_transformer_config(&self) -> Result<TransformerConfig> {
        let hidden_size = self.hidden_size.ok_or_else(|| {
            WeightError::InvalidConfig("missing hidden_size".into())
        })?;
        let num_heads = self.num_attention_heads.ok_or_else(|| {
            WeightError::InvalidConfig("missing num_attention_heads".into())
        })?;
        let num_layers = self.num_hidden_layers.ok_or_else(|| {
            WeightError::InvalidConfig("missing num_hidden_layers".into())
        })?;
        let vocab_size = self.vocab_size.ok_or_else(|| {
            WeightError::InvalidConfig("missing vocab_size".into())
        })?;

        let num_kv_heads = self.effective_num_kv_heads();
        let head_dim = self.effective_head_dim();
        let max_seq_len = self.max_position_embeddings.unwrap_or(4096);
        let rope_theta = self.rope_theta.unwrap_or(10000.0) as f32;
        let rms_norm_eps = self.rms_norm_eps.unwrap_or(1e-5) as f32;

        let ff_type = if let (Some(num_experts), Some(experts_per_tok)) =
            (self.num_local_experts, self.num_experts_per_tok)
        {
            let intermediate_size = self.intermediate_size.ok_or_else(|| {
                WeightError::InvalidConfig("missing intermediate_size for MoE model".into())
            })?;
            FeedForwardType::MoE {
                config: MoeConfig {
                    num_experts,
                    num_experts_per_token: experts_per_tok,
                    hidden_dim: hidden_size,
                    intermediate_dim: intermediate_size,
                    capacity_factor: 1.0,
                },
            }
        } else {
            let intermediate_dim = self.intermediate_size.ok_or_else(|| {
                WeightError::InvalidConfig("missing intermediate_size".into())
            })?;
            FeedForwardType::Dense { intermediate_dim }
        };

        Ok(TransformerConfig {
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            ff_type,
        })
    }
}
