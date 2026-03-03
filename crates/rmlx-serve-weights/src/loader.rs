//! Model weight loading from safetensors files.
//!
//! Supports both single-file models (`model.safetensors`) and sharded models
//! (`model.safetensors.index.json` + `model-00001-of-NNNNN.safetensors`).
//!
//! The loading pipeline:
//! 1. Read safetensors bytes from disk
//! 2. Parse tensor metadata (name, dtype, shape, offsets)
//! 3. Copy tensor data to Metal GPU buffers as `rmlx_core::Array`
//! 4. Sanitize weights (remove inv_freq, handle tied embeddings, etc.)
//! 5. Map weight names to model components via `WeightMapper`
//! 6. Assemble components into a `TransformerModel`

use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;
use tracing::{debug, info, warn};

use rmlx_core::{Array, DType};
use rmlx_metal::GpuDevice;
use rmlx_nn::{
    Attention, AttentionConfig, Embedding, EmbeddingConfig, FeedForward, FeedForwardType, Linear,
    LinearConfig, MoeConfig, MoeLayer, TransformerBlock, TransformerConfig, TransformerModel,
};

use crate::config::ModelConfig;
use crate::error::{Result, WeightError};
use crate::mapper::{self, WeightComponent, WeightMapper};
use crate::quantization;
use crate::sanitize;

/// Safetensors index file format for multi-shard models.
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    /// Metadata (total_size, etc.) — not used directly.
    #[serde(default)]
    #[allow(dead_code)]
    metadata: serde_json::Value,
    /// Maps tensor name -> shard filename.
    weight_map: HashMap<String, String>,
}

/// Convert a safetensors dtype string to an RMLX DType.
fn safetensors_dtype_to_rmlx(dtype: safetensors::Dtype) -> Option<DType> {
    match dtype {
        safetensors::Dtype::F32 => Some(DType::Float32),
        safetensors::Dtype::F16 => Some(DType::Float16),
        safetensors::Dtype::BF16 => Some(DType::Bfloat16),
        safetensors::Dtype::U32 => Some(DType::UInt32),
        _ => None,
    }
}

/// Load all tensors from safetensors file(s) into a HashMap of Arrays on GPU.
///
/// Handles two cases:
/// 1. Single file: `{path}/model.safetensors`
/// 2. Sharded: `{path}/model.safetensors.index.json` pointing to multiple shard files
///
/// Returns a map from tensor name to Array allocated on the given Metal device.
pub fn load_safetensors(
    path: impl AsRef<Path>,
    device: &metal::Device,
) -> Result<HashMap<String, Array>> {
    let base_path = path.as_ref();

    // Check for single-file model first
    let single_path = base_path.join("model.safetensors");
    if single_path.exists() {
        info!(path = %single_path.display(), "loading single-file safetensors");
        return load_single_safetensors(&single_path, device);
    }

    // Check for sharded model
    let index_path = base_path.join("model.safetensors.index.json");
    if index_path.exists() {
        info!(path = %index_path.display(), "loading sharded safetensors");
        return load_sharded_safetensors(base_path, &index_path, device);
    }

    Err(WeightError::SafetensorsError(format!(
        "no model.safetensors or model.safetensors.index.json found in {}",
        base_path.display()
    )))
}

/// Load tensors from a single safetensors file.
fn load_single_safetensors(
    file_path: &Path,
    device: &metal::Device,
) -> Result<HashMap<String, Array>> {
    let bytes = std::fs::read(file_path)?;
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|e| WeightError::SafetensorsError(format!("failed to parse safetensors: {e}")))?;

    let mut result = HashMap::new();

    for (name, view) in tensors.tensors() {
        let rmlx_dtype = match safetensors_dtype_to_rmlx(view.dtype()) {
            Some(dt) => dt,
            None => {
                warn!(
                    name = name.as_str(),
                    dtype = ?view.dtype(),
                    "skipping tensor with unsupported dtype"
                );
                continue;
            }
        };

        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();

        let array = Array::from_bytes(device, data, shape, rmlx_dtype);
        debug!(
            name = name.as_str(),
            dtype = %rmlx_dtype,
            shape = ?array.shape(),
            bytes = data.len(),
            "loaded tensor"
        );
        result.insert(name.to_string(), array);
    }

    info!(count = result.len(), "loaded tensors from single safetensors file");
    Ok(result)
}

/// Load tensors from a sharded safetensors model.
fn load_sharded_safetensors(
    base_path: &Path,
    index_path: &Path,
    device: &metal::Device,
) -> Result<HashMap<String, Array>> {
    let index_content = std::fs::read_to_string(index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)
        .map_err(|e| WeightError::SafetensorsError(format!("failed to parse index: {e}")))?;

    // Determine unique shard filenames
    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();

    info!(
        shards = shard_files.len(),
        tensors = index.weight_map.len(),
        "loading sharded model"
    );

    let mut result = HashMap::new();

    // Load each shard file
    for shard_name in &shard_files {
        let shard_path = base_path.join(shard_name);
        if !shard_path.exists() {
            return Err(WeightError::ShardNotFound {
                path: shard_path,
            });
        }

        debug!(shard = shard_name.as_str(), "loading shard");
        let bytes = std::fs::read(&shard_path)?;
        let tensors = SafeTensors::deserialize(&bytes).map_err(|e| {
            WeightError::SafetensorsError(format!("failed to parse shard {shard_name}: {e}"))
        })?;

        // Only load tensors that belong to this shard according to the index
        for (tensor_name, shard_file) in &index.weight_map {
            if shard_file != shard_name {
                continue;
            }

            let view = match tensors.tensor(tensor_name) {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        name = tensor_name.as_str(),
                        shard = shard_name.as_str(),
                        error = %e,
                        "tensor listed in index but not found in shard"
                    );
                    continue;
                }
            };

            let rmlx_dtype = match safetensors_dtype_to_rmlx(view.dtype()) {
                Some(dt) => dt,
                None => {
                    warn!(
                        name = tensor_name.as_str(),
                        dtype = ?view.dtype(),
                        "skipping tensor with unsupported dtype"
                    );
                    continue;
                }
            };

            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            let array = Array::from_bytes(device, data, shape, rmlx_dtype);
            debug!(
                name = tensor_name.as_str(),
                dtype = %rmlx_dtype,
                shape = ?array.shape(),
                bytes = data.len(),
                "loaded tensor from shard"
            );
            result.insert(tensor_name.clone(), array);
        }
    }

    info!(
        count = result.len(),
        expected = index.weight_map.len(),
        "loaded tensors from sharded safetensors"
    );

    Ok(result)
}

/// Load a complete TransformerModel from a model directory.
///
/// This is the main entry point for model loading. It:
/// 1. Reads config.json to determine architecture and dimensions
/// 2. Loads all safetensors weight files
/// 3. Applies quantization transforms if needed
/// 4. Sanitizes weights (remove inv_freq, handle tied embeddings, etc.)
/// 5. Maps weights to model components
/// 6. Assembles a `TransformerModel` ready for inference
///
/// # Arguments
/// * `path` - Path to a directory containing config.json and model.safetensors (or sharded files)
///
/// # Returns
/// A tuple of (TransformerModel, ModelConfig).
pub fn load_model(path: impl AsRef<Path>) -> Result<(TransformerModel, ModelConfig)> {
    let model_path = path.as_ref();
    info!(path = %model_path.display(), "loading model");

    // 1. Load config
    let config = ModelConfig::from_path(model_path)?;
    let transformer_config = config.to_transformer_config()?;

    // 2. Get GPU device
    let gpu = GpuDevice::system_default().map_err(|e| {
        WeightError::InvalidConfig(format!("failed to acquire Metal device: {e}"))
    })?;
    let device = gpu.raw();

    // 3. Load safetensors
    let mut weights = load_safetensors(model_path, device)?;

    // 4. Apply quantization if needed
    if let Some(ref qconfig) = config.quantization_config {
        let method = quantization::detect_quantization(&config);
        if method != quantization::QuantMethod::None {
            info!(method = ?method, "applying quantization transforms");
            quantization::apply_quantization(&mut weights, qconfig, device);
        }
    }

    // 5. Sanitize weights
    weights = sanitize::sanitize_weights(weights, &config.model_type);

    // 6. Map and assemble
    let mapper = mapper::create_weight_mapper(&config.model_type);
    let model = assemble_model(weights, &transformer_config, mapper.as_ref(), device)?;

    info!(
        model_type = config.model_type.as_str(),
        num_layers = transformer_config.num_layers,
        hidden_size = transformer_config.hidden_size,
        vocab_size = transformer_config.vocab_size,
        "model loaded successfully"
    );

    Ok((model, config))
}

/// Assemble a TransformerModel from mapped weight tensors.
fn assemble_model(
    weights: HashMap<String, Array>,
    config: &TransformerConfig,
    mapper: &dyn WeightMapper,
    device: &metal::Device,
) -> Result<TransformerModel> {
    // Categorize all weights by their component
    let mut embedding_weight: Option<Array> = None;
    let mut lm_head_weight: Option<Array> = None;
    let mut final_norm_weight: Option<Array> = None;

    // Per-layer storage — Array doesn't implement Clone, so we build Vecs via iterators
    let num_layers = config.num_layers;
    let mut attn_norm: Vec<Option<Array>> = (0..num_layers).map(|_| None).collect();
    let mut ffn_norm: Vec<Option<Array>> = (0..num_layers).map(|_| None).collect();
    let mut q_proj: Vec<Option<Array>> = (0..num_layers).map(|_| None).collect();
    let mut k_proj: Vec<Option<Array>> = (0..num_layers).map(|_| None).collect();
    let mut v_proj: Vec<Option<Array>> = (0..num_layers).map(|_| None).collect();
    let mut o_proj: Vec<Option<Array>> = (0..num_layers).map(|_| None).collect();
    let mut gate_proj: Vec<Option<Array>> = (0..num_layers).map(|_| None).collect();
    let mut up_proj: Vec<Option<Array>> = (0..num_layers).map(|_| None).collect();
    let mut down_proj: Vec<Option<Array>> = (0..num_layers).map(|_| None).collect();

    // MoE storage
    let mut moe_gate: Vec<Option<Array>> = (0..num_layers).map(|_| None).collect();
    // expert_weights[layer][expert] = (gate, up, down)
    let mut expert_gate_proj: Vec<HashMap<usize, Array>> =
        (0..num_layers).map(|_| HashMap::new()).collect();
    let mut expert_up_proj: Vec<HashMap<usize, Array>> =
        (0..num_layers).map(|_| HashMap::new()).collect();
    let mut expert_down_proj: Vec<HashMap<usize, Array>> =
        (0..num_layers).map(|_| HashMap::new()).collect();

    // Map each weight to its component
    for (name, array) in weights {
        let mapped = match mapper.map_name(&name) {
            Some(m) => m,
            None => {
                debug!(name = name.as_str(), "unmapped weight, skipping");
                continue;
            }
        };

        match mapped.component {
            WeightComponent::Embedding => {
                embedding_weight = Some(array);
            }
            WeightComponent::LmHead => {
                lm_head_weight = Some(array);
            }
            WeightComponent::FinalNorm => {
                final_norm_weight = Some(array);
            }
            WeightComponent::AttnNorm { layer } => {
                if layer < num_layers {
                    attn_norm[layer] = Some(array);
                }
            }
            WeightComponent::FfnNorm { layer } => {
                if layer < num_layers {
                    ffn_norm[layer] = Some(array);
                }
            }
            WeightComponent::QProj { layer } => {
                if layer < num_layers {
                    q_proj[layer] = Some(array);
                }
            }
            WeightComponent::KProj { layer } => {
                if layer < num_layers {
                    k_proj[layer] = Some(array);
                }
            }
            WeightComponent::VProj { layer } => {
                if layer < num_layers {
                    v_proj[layer] = Some(array);
                }
            }
            WeightComponent::OProj { layer } => {
                if layer < num_layers {
                    o_proj[layer] = Some(array);
                }
            }
            WeightComponent::GateProj { layer } => {
                if layer < num_layers {
                    gate_proj[layer] = Some(array);
                }
            }
            WeightComponent::UpProj { layer } => {
                if layer < num_layers {
                    up_proj[layer] = Some(array);
                }
            }
            WeightComponent::DownProj { layer } => {
                if layer < num_layers {
                    down_proj[layer] = Some(array);
                }
            }
            WeightComponent::GateWeight { layer } => {
                if layer < num_layers {
                    moe_gate[layer] = Some(array);
                }
            }
            WeightComponent::ExpertGateProj { layer, expert } => {
                if layer < num_layers {
                    expert_gate_proj[layer].insert(expert, array);
                }
            }
            WeightComponent::ExpertUpProj { layer, expert } => {
                if layer < num_layers {
                    expert_up_proj[layer].insert(expert, array);
                }
            }
            WeightComponent::ExpertDownProj { layer, expert } => {
                if layer < num_layers {
                    expert_down_proj[layer].insert(expert, array);
                }
            }
        }
    }

    // Validate required components
    let embed_w = embedding_weight.ok_or(WeightError::MissingWeight(
        "model.embed_tokens.weight".into(),
    ))?;
    let lm_head_w = lm_head_weight.ok_or(WeightError::MissingWeight("lm_head.weight".into()))?;
    let final_norm_w =
        final_norm_weight.ok_or(WeightError::MissingWeight("model.norm.weight".into()))?;

    // Build embedding
    let embedding = Embedding::from_array(
        EmbeddingConfig {
            vocab_size: config.vocab_size,
            embed_dim: config.hidden_size,
        },
        embed_w,
    )?;

    // Build lm_head
    let lm_head = Linear::from_arrays(
        LinearConfig {
            in_features: config.hidden_size,
            out_features: config.vocab_size,
            has_bias: false,
        },
        lm_head_w,
        None,
    )?;

    // Build transformer blocks
    let mut layers = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let layer_attn_norm = attn_norm[i].take().ok_or_else(|| {
            WeightError::MissingWeight(format!("model.layers.{i}.input_layernorm.weight"))
        })?;
        let layer_ffn_norm = ffn_norm[i].take().ok_or_else(|| {
            WeightError::MissingWeight(format!(
                "model.layers.{i}.post_attention_layernorm.weight"
            ))
        })?;

        // Build attention
        let layer_q = q_proj[i].take().ok_or_else(|| {
            WeightError::MissingWeight(format!("model.layers.{i}.self_attn.q_proj.weight"))
        })?;
        let layer_k = k_proj[i].take().ok_or_else(|| {
            WeightError::MissingWeight(format!("model.layers.{i}.self_attn.k_proj.weight"))
        })?;
        let layer_v = v_proj[i].take().ok_or_else(|| {
            WeightError::MissingWeight(format!("model.layers.{i}.self_attn.v_proj.weight"))
        })?;
        let layer_o = o_proj[i].take().ok_or_else(|| {
            WeightError::MissingWeight(format!("model.layers.{i}.self_attn.o_proj.weight"))
        })?;

        let hidden_size = config.hidden_size;
        let kv_size = config.num_kv_heads * config.head_dim;

        let attn_config = AttentionConfig {
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            max_seq_len: config.max_seq_len,
            rope_theta: config.rope_theta,
        };

        let q_linear = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_size,
                out_features: config.num_heads * config.head_dim,
                has_bias: false,
            },
            layer_q,
            None,
        )?;
        let k_linear = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_size,
                out_features: kv_size,
                has_bias: false,
            },
            layer_k,
            None,
        )?;
        let v_linear = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_size,
                out_features: kv_size,
                has_bias: false,
            },
            layer_v,
            None,
        )?;
        let o_linear = Linear::from_arrays(
            LinearConfig {
                in_features: config.num_heads * config.head_dim,
                out_features: hidden_size,
                has_bias: false,
            },
            layer_o,
            None,
        )?;

        let attention = Attention::from_layers(attn_config, q_linear, k_linear, v_linear, o_linear)?;

        // Build FFN: dense or MoE depending on config and available weights
        let ffn = build_ffn(
            i,
            config,
            &mut gate_proj,
            &mut up_proj,
            &mut down_proj,
            &mut moe_gate,
            &mut expert_gate_proj,
            &mut expert_up_proj,
            &mut expert_down_proj,
            device,
        )?;

        let block = TransformerBlock::from_parts(
            i,
            attention,
            ffn,
            layer_attn_norm,
            layer_ffn_norm,
            config.rms_norm_eps,
        );
        layers.push(block);
    }

    let model = TransformerModel::from_parts(duplicate_config(config), embedding, layers, final_norm_w, lm_head)?;
    Ok(model)
}

/// Build the feed-forward network for a layer (dense or MoE).
#[allow(clippy::too_many_arguments)]
fn build_ffn(
    layer_idx: usize,
    config: &TransformerConfig,
    gate_proj: &mut [Option<Array>],
    up_proj: &mut [Option<Array>],
    down_proj: &mut [Option<Array>],
    moe_gate: &mut [Option<Array>],
    expert_gate_proj: &mut [HashMap<usize, Array>],
    expert_up_proj: &mut [HashMap<usize, Array>],
    expert_down_proj: &mut [HashMap<usize, Array>],
    _device: &metal::Device,
) -> Result<FeedForward> {
    let hidden_size = config.hidden_size;

    // Check if this layer has MoE weights
    let has_moe = moe_gate[layer_idx].is_some() && !expert_gate_proj[layer_idx].is_empty();

    if has_moe {
        // MoE FFN
        let (num_experts, intermediate_dim) = match &config.ff_type {
            FeedForwardType::MoE { config: moe_cfg } => {
                (moe_cfg.num_experts, moe_cfg.intermediate_dim)
            }
            FeedForwardType::Dense { intermediate_dim } => {
                // Infer expert count from available weights
                let ne = expert_gate_proj[layer_idx].len();
                (ne, *intermediate_dim)
            }
        };

        let experts_per_tok = match &config.ff_type {
            FeedForwardType::MoE { config: moe_cfg } => moe_cfg.num_experts_per_token,
            _ => 2, // default
        };

        let moe_config = MoeConfig {
            num_experts,
            num_experts_per_token: experts_per_tok,
            hidden_dim: hidden_size,
            intermediate_dim,
            capacity_factor: 1.0,
        };

        // Build gate (router)
        let gate_weight = moe_gate[layer_idx].take().ok_or_else(|| {
            WeightError::MissingWeight(format!("layer {layer_idx} MoE gate weight"))
        })?;
        let gate_linear = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_size,
                out_features: num_experts,
                has_bias: false,
            },
            gate_weight,
            None,
        )?;

        // Build experts
        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let eg = expert_gate_proj[layer_idx].remove(&e).ok_or_else(|| {
                WeightError::MissingWeight(format!(
                    "layer {layer_idx} expert {e} gate_proj weight"
                ))
            })?;
            let eu = expert_up_proj[layer_idx].remove(&e).ok_or_else(|| {
                WeightError::MissingWeight(format!(
                    "layer {layer_idx} expert {e} up_proj weight"
                ))
            })?;
            let ed = expert_down_proj[layer_idx].remove(&e).ok_or_else(|| {
                WeightError::MissingWeight(format!(
                    "layer {layer_idx} expert {e} down_proj weight"
                ))
            })?;

            let expert_gate = Linear::from_arrays(
                LinearConfig {
                    in_features: hidden_size,
                    out_features: intermediate_dim,
                    has_bias: false,
                },
                eg,
                None,
            )?;
            let expert_up = Linear::from_arrays(
                LinearConfig {
                    in_features: hidden_size,
                    out_features: intermediate_dim,
                    has_bias: false,
                },
                eu,
                None,
            )?;
            let expert_down = Linear::from_arrays(
                LinearConfig {
                    in_features: intermediate_dim,
                    out_features: hidden_size,
                    has_bias: false,
                },
                ed,
                None,
            )?;

            experts.push(rmlx_nn::moe::Expert {
                gate_proj: expert_gate,
                up_proj: expert_up,
                down_proj: expert_down,
            });
        }

        let moe_layer = MoeLayer::from_layers(moe_config, gate_linear, experts)?;
        Ok(FeedForward::MoE(moe_layer))
    } else {
        // Dense FFN
        let intermediate_dim = match &config.ff_type {
            FeedForwardType::Dense { intermediate_dim } => *intermediate_dim,
            FeedForwardType::MoE { config: moe_cfg } => moe_cfg.intermediate_dim,
        };

        let gp = gate_proj[layer_idx].take().ok_or_else(|| {
            WeightError::MissingWeight(format!(
                "model.layers.{layer_idx}.mlp.gate_proj.weight"
            ))
        })?;
        let up = up_proj[layer_idx].take().ok_or_else(|| {
            WeightError::MissingWeight(format!(
                "model.layers.{layer_idx}.mlp.up_proj.weight"
            ))
        })?;
        let dp = down_proj[layer_idx].take().ok_or_else(|| {
            WeightError::MissingWeight(format!(
                "model.layers.{layer_idx}.mlp.down_proj.weight"
            ))
        })?;

        let gate_linear = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_size,
                out_features: intermediate_dim,
                has_bias: false,
            },
            gp,
            None,
        )?;
        let up_linear = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_size,
                out_features: intermediate_dim,
                has_bias: false,
            },
            up,
            None,
        )?;
        let down_linear = Linear::from_arrays(
            LinearConfig {
                in_features: intermediate_dim,
                out_features: hidden_size,
                has_bias: false,
            },
            dp,
            None,
        )?;

        Ok(FeedForward::Dense {
            gate_proj: gate_linear,
            up_proj: up_linear,
            down_proj: down_linear,
        })
    }
}

/// Create a duplicate TransformerConfig (TransformerConfig doesn't derive Clone
/// since it lives in rmlx-nn, and we can't add an orphan impl).
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
