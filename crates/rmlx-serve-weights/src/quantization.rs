//! Quantization detection and weight transformation utilities.
//!
//! Supports AWQ, GPTQ, and BitsAndBytes quantization formats commonly used in
//! HuggingFace models. When a quantized model is detected, raw packed weights
//! are unpacked into a format suitable for RMLX GPU kernels.

use std::collections::HashMap;

use tracing::{debug, warn};

use rmlx_core::{Array, DType};

use crate::config::{ModelConfig, QuantizationConfig};

/// Detected quantization method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantMethod {
    /// No quantization (fp16/bf16/fp32 weights).
    None,
    /// Activation-aware Weight Quantization.
    AWQ,
    /// GPTQ post-training quantization.
    GPTQ,
    /// BitsAndBytes quantization (nf4/int8).
    BitsAndBytes,
}

/// Detect the quantization method from a ModelConfig.
///
/// Returns `QuantMethod::None` if no quantization config is present or the
/// method string is unrecognized.
pub fn detect_quantization(config: &ModelConfig) -> QuantMethod {
    match &config.quantization_config {
        Some(qc) => match qc.quant_method.to_lowercase().as_str() {
            "awq" => {
                debug!(bits = ?qc.bits, group_size = ?qc.group_size, "detected AWQ quantization");
                QuantMethod::AWQ
            }
            "gptq" => {
                debug!(bits = ?qc.bits, group_size = ?qc.group_size, "detected GPTQ quantization");
                QuantMethod::GPTQ
            }
            "bitsandbytes" => {
                debug!(bits = ?qc.bits, "detected BitsAndBytes quantization");
                QuantMethod::BitsAndBytes
            }
            other => {
                warn!(
                    method = other,
                    "unrecognized quantization method, treating as unquantized"
                );
                QuantMethod::None
            }
        },
        None => QuantMethod::None,
    }
}

/// Unpack AWQ-quantized weights from a packed integer format.
///
/// AWQ packs multiple low-bit weights into 32-bit integers. This function
/// extracts individual weight values and converts them to f32.
///
/// # Arguments
/// * `packed` - Raw bytes of the packed weight tensor (u32 values).
/// * `bits` - Number of bits per weight (typically 4).
/// * `group_size` - Number of weights sharing a single scale/zero-point.
///
/// # Returns
/// A vector of f32 values representing the unpacked weights.
pub fn unpack_awq_weights(packed: &[u8], bits: usize, _group_size: usize) -> Vec<f32> {
    if bits == 0 || bits > 32 {
        return Vec::new();
    }

    let elements_per_u32 = 32 / bits;
    let mask = (1u32 << bits) - 1;

    // Interpret packed bytes as u32 values (little-endian)
    let num_u32s = packed.len() / 4;
    let mut result = Vec::with_capacity(num_u32s * elements_per_u32);

    for i in 0..num_u32s {
        let offset = i * 4;
        let packed_val = u32::from_le_bytes([
            packed[offset],
            packed[offset + 1],
            packed[offset + 2],
            packed[offset + 3],
        ]);

        #[allow(clippy::needless_range_loop)]
        for j in 0..elements_per_u32 {
            // AWQ uses interleaved bit packing order for 4-bit quantization
            let shift = if bits == 4 {
                const AWQ_4BIT_ORDER: [usize; 8] = [0, 16, 4, 20, 8, 24, 12, 28];
                AWQ_4BIT_ORDER[j]
            } else {
                j * bits
            };
            let val = (packed_val >> shift) & mask;
            // AWQ weights are unsigned; the zero-point tensor handles centering
            result.push(val as f32);
        }
    }

    result
}

/// Transform GPTQ-quantized weights from packed format.
///
/// GPTQ uses a similar packing scheme to AWQ, with weights stored as packed
/// integers alongside scale and zero-point tensors. This function unpacks
/// the integer weights.
///
/// # Arguments
/// * `packed` - Raw bytes of the packed weight tensor.
/// * `bits` - Number of bits per weight (typically 4).
/// * `group_size` - Number of weights sharing a single scale/zero-point.
///
/// # Returns
/// A vector of f32 values representing the unpacked weights.
pub fn transform_gptq_weights(packed: &[u8], bits: usize, _group_size: usize) -> Vec<f32> {
    if bits == 0 || bits > 32 {
        return Vec::new();
    }

    let elements_per_u32 = 32 / bits;
    let mask = (1u32 << bits) - 1;

    let num_u32s = packed.len() / 4;
    let mut result = Vec::with_capacity(num_u32s * elements_per_u32);

    for i in 0..num_u32s {
        let offset = i * 4;
        let packed_val = u32::from_le_bytes([
            packed[offset],
            packed[offset + 1],
            packed[offset + 2],
            packed[offset + 3],
        ]);

        // GPTQ also uses interleaved bit packing order for 4-bit quantization
        #[allow(clippy::needless_range_loop)]
        for j in 0..elements_per_u32 {
            let shift = if bits == 4 {
                const GPTQ_4BIT_ORDER: [usize; 8] = [0, 16, 4, 20, 8, 24, 12, 28];
                GPTQ_4BIT_ORDER[j]
            } else {
                j * bits
            };
            let val = (packed_val >> shift) & mask;
            result.push(val as f32);
        }
    }

    result
}

/// Apply quantization transforms to weight tensors in-place.
///
/// For quantized models (AWQ, GPTQ), this processes the packed weight tensors
/// and replaces them with unpacked f32 tensors suitable for RMLX kernels.
///
/// This function looks for common quantization tensor patterns:
/// - `*.qweight` - packed quantized weights
/// - `*.qzeros` - zero-point values
/// - `*.scales` - per-group scale factors
///
/// The unpacked weight is computed as: `(qweight - qzeros) * scales`
pub fn apply_quantization(
    weights: &mut HashMap<String, Array>,
    config: &QuantizationConfig,
    device: &metal::Device,
) {
    let bits = config.bits.unwrap_or(4);
    let group_size = config.group_size.unwrap_or(128);
    let method = config.quant_method.to_lowercase();

    // Collect names of qweight tensors to process
    let qweight_names: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".qweight"))
        .cloned()
        .collect();

    if qweight_names.is_empty() {
        debug!("no .qweight tensors found, skipping quantization transform");
        return;
    }

    debug!(
        count = qweight_names.len(),
        method = method.as_str(),
        bits = bits,
        group_size = group_size,
        "applying quantization transforms"
    );

    for qweight_name in &qweight_names {
        // Derive related tensor names
        let base = qweight_name.strip_suffix(".qweight").unwrap();
        let scales_name = format!("{base}.scales");
        let zeros_name = format!("{base}.qzeros");

        // Get the packed weight bytes
        let qweight = match weights.get(qweight_name) {
            Some(w) => w,
            None => continue,
        };
        let packed_bytes = qweight.to_bytes();

        // Unpack based on method
        let unpacked = match method.as_str() {
            "awq" => unpack_awq_weights(packed_bytes, bits, group_size),
            "gptq" => transform_gptq_weights(packed_bytes, bits, group_size),
            _ => continue,
        };

        if unpacked.is_empty() {
            warn!(
                name = qweight_name.as_str(),
                "unpacking produced zero elements"
            );
            continue;
        }

        // Determine the unpacked shape: the qweight shape's column count expands
        // by the packing ratio (32 / bits).
        let qw_shape = qweight.shape().to_vec();
        let packing_ratio = 32 / bits;
        let (rows, cols) = if qw_shape.len() == 2 {
            (qw_shape[0], qw_shape[1] * packing_ratio)
        } else {
            (1, unpacked.len())
        };

        // Apply scales and zeros if available
        let dequantized = if let (Some(scales), Some(zeros)) =
            (weights.get(&scales_name), weights.get(&zeros_name))
        {
            let scales_f32: Vec<f32> = read_as_f32(scales);
            let zeros_f32: Vec<f32> = read_as_f32(zeros);
            apply_scales_and_zeros(&unpacked, &scales_f32, &zeros_f32, group_size, cols)
        } else if let Some(scales) = weights.get(&scales_name) {
            let scales_f32: Vec<f32> = read_as_f32(scales);
            apply_scales_only(&unpacked, &scales_f32, group_size)
        } else {
            unpacked
        };

        // For AWQ, transpose from [rows, cols] to [cols, rows]
        let (out_shape, final_data) = if method == "awq" && qw_shape.len() == 2 {
            let mut transposed = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    transposed[c * rows + r] = dequantized[r * cols + c];
                }
            }
            (vec![cols, rows], transposed)
        } else {
            let out_shape = if qw_shape.len() == 2 {
                vec![rows, cols]
            } else {
                vec![dequantized.len()]
            };
            let numel: usize = out_shape.iter().product();
            let data = if dequantized.len() > numel {
                dequantized[..numel].to_vec()
            } else {
                dequantized
            };
            (out_shape, data)
        };

        // Replace the weight name (remove .qweight, add .weight)
        let weight_name = format!("{base}.weight");
        let arr = Array::from_slice(device, &final_data, out_shape);
        weights.insert(weight_name, arr);

        // Remove the quantization-specific tensors
        weights.remove(qweight_name);
        weights.remove(&scales_name);
        weights.remove(&zeros_name);
    }
}

/// Read an Array as f32 values, converting from f16/bf16 if needed.
fn read_as_f32(arr: &Array) -> Vec<f32> {
    match arr.dtype() {
        DType::Float32 => arr.to_vec_checked::<f32>(),
        DType::Float16 | DType::Bfloat16 => {
            // Read raw bytes and convert element by element
            let bytes = arr.to_bytes();
            let numel = arr.numel();
            let mut result = Vec::with_capacity(numel);
            for i in 0..numel {
                let offset = i * 2;
                let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                let val = match arr.dtype() {
                    DType::Float16 => f16_to_f32(bits),
                    DType::Bfloat16 => bf16_to_f32(bits),
                    _ => unreachable!(),
                };
                result.push(val);
            }
            result
        }
        _ => Vec::new(),
    }
}

/// Convert a half-precision (IEEE 754 binary16) bit pattern to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: renormalize
            let mut f = frac;
            let mut e = 0i32;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3FF;
            let f32_exp = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (f << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
    } else {
        // Normal
        let f32_exp = exp + (127 - 15);
        f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
    }
}

/// Convert a bfloat16 bit pattern to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    // bfloat16 is the upper 16 bits of a float32, so just shift left.
    f32::from_bits((bits as u32) << 16)
}

/// Apply per-group scales and zero-points to unpacked weights.
///
/// Scales and zeros are 2D tensors with shape `[num_groups, out_features]`.
/// `num_cols` is the number of columns in the unpacked weight matrix (out_features).
fn apply_scales_and_zeros(
    unpacked: &[f32],
    scales: &[f32],
    zeros: &[f32],
    group_size: usize,
    num_cols: usize,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(unpacked.len());
    for (i, &val) in unpacked.iter().enumerate() {
        let row = i / num_cols;
        let col = i % num_cols;
        let group_row = row / group_size;
        // scales/zeros layout: [num_groups, num_cols] => index = group_row * num_cols + col
        let scale_idx = group_row * num_cols + col;
        let scale = scales.get(scale_idx).copied().unwrap_or(1.0);
        let zero = zeros.get(scale_idx).copied().unwrap_or(0.0);
        result.push((val - zero) * scale);
    }
    result
}

/// Apply per-group scales only (no zero-points) to unpacked weights.
fn apply_scales_only(unpacked: &[f32], scales: &[f32], group_size: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(unpacked.len());
    for (i, &val) in unpacked.iter().enumerate() {
        let group = i / group_size;
        let scale = scales.get(group).copied().unwrap_or(1.0);
        result.push(val * scale);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_quantization_none() {
        let config = ModelConfig {
            model_type: "llama".into(),
            hidden_size: Some(4096),
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: None,
            intermediate_size: Some(11008),
            vocab_size: Some(32000),
            max_position_embeddings: Some(4096),
            rope_theta: None,
            rms_norm_eps: None,
            head_dim: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            quantization_config: None,
            tie_word_embeddings: None,
            rope_scaling: None,
            architectures: None,
            torch_dtype: None,
            attention_bias: None,
        };
        assert_eq!(detect_quantization(&config), QuantMethod::None);
    }

    #[test]
    fn test_detect_quantization_awq() {
        let config = ModelConfig {
            model_type: "llama".into(),
            hidden_size: Some(4096),
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: None,
            intermediate_size: Some(11008),
            vocab_size: Some(32000),
            max_position_embeddings: Some(4096),
            rope_theta: None,
            rms_norm_eps: None,
            head_dim: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            quantization_config: Some(QuantizationConfig {
                quant_method: "awq".into(),
                bits: Some(4),
                group_size: Some(128),
            }),
            tie_word_embeddings: None,
            rope_scaling: None,
            architectures: None,
            torch_dtype: None,
            attention_bias: None,
        };
        assert_eq!(detect_quantization(&config), QuantMethod::AWQ);
    }

    #[test]
    fn test_unpack_awq_4bit() {
        // Test AWQ interleaved 4-bit unpacking.
        // AWQ 4-bit order: shifts [0, 16, 4, 20, 8, 24, 12, 28]
        // Pack known values into a u32 at the interleaved positions.
        // Place value 3 at bits[0..4], value 5 at bits[16..20],
        // value 7 at bits[4..8], value 9 at bits[20..24],
        // value 1 at bits[8..12], value 2 at bits[24..28],
        // value 4 at bits[12..16], value 6 at bits[28..32]
        let packed_val: u32 = (3 << 0)
            | (7 << 4)
            | (1 << 8)
            | (4 << 12)
            | (5 << 16)
            | (9 << 20)
            | (2 << 24)
            | (6 << 28);
        let packed = packed_val.to_le_bytes().to_vec();
        let result = unpack_awq_weights(&packed, 4, 128);
        assert_eq!(result.len(), 8); // 32/4 = 8 values per u32
                                     // Values extracted in interleaved order: shifts [0,16,4,20,8,24,12,28]
                                     // Unsigned (no sign centering)
        assert_eq!(result[0], 3.0); // shift 0
        assert_eq!(result[1], 5.0); // shift 16
        assert_eq!(result[2], 7.0); // shift 4
        assert_eq!(result[3], 9.0); // shift 20
        assert_eq!(result[4], 1.0); // shift 8
        assert_eq!(result[5], 2.0); // shift 24
        assert_eq!(result[6], 4.0); // shift 12
        assert_eq!(result[7], 6.0); // shift 28
    }

    #[test]
    fn test_bf16_to_f32() {
        // 1.0 in bf16 is 0x3F80 (same upper bits as f32 1.0 = 0x3F800000)
        let val = bf16_to_f32(0x3F80);
        assert!((val - 1.0).abs() < 1e-6);

        // 0.0
        let val = bf16_to_f32(0x0000);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_f16_to_f32() {
        // 1.0 in f16 is 0x3C00
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 1e-6);

        // 0.0
        let val = f16_to_f32(0x0000);
        assert_eq!(val, 0.0);

        // -1.0 in f16 is 0xBC00
        let val = f16_to_f32(0xBC00);
        assert!((val + 1.0).abs() < 1e-6);
    }
}
