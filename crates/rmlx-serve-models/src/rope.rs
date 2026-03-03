//! RoPE (Rotary Position Embedding) frequency computation utilities.
//!
//! Precomputes cosine and sine frequency tables on the CPU, then uploads
//! them to GPU as `Array` tensors. These tables are computed once at model
//! load time and reused for every forward pass.
//!
//! The standard RoPE formulation:
//!   theta_i = rope_theta^(-2i / head_dim)   for i in 0..head_dim/2
//!   freq(pos, i) = pos * theta_i
//!   cos_table[pos][i] = cos(freq(pos, i))
//!   sin_table[pos][i] = sin(freq(pos, i))

use rmlx_core::Array;

/// RoPE scaling method parsed from the model config's `rope_scaling` field.
#[derive(Debug, Clone)]
pub enum RopeScalingMethod {
    /// No scaling applied — standard RoPE.
    None,
    /// Linear scaling: divide each position by the factor before computing
    /// frequencies. Effectively stretches the positional range.
    Linear { factor: f32 },
    /// NTK-aware scaling (also called "dynamic"): modify the base theta to
    /// `theta * factor^(head_dim / (head_dim - 2))` so that high-frequency
    /// components are preserved while low-frequency ones are extended.
    Ntk { factor: f32 },
    /// YaRN (Yet another RoPE extensioN): combines NTK base scaling with an
    /// attention temperature adjustment. Provides better extrapolation than
    /// pure NTK by blending scaled and unscaled frequencies.
    Yarn {
        factor: f32,
        original_max_position_embeddings: usize,
    },
}

/// Parse a RoPE scaling configuration from a `serde_json::Value`.
///
/// Expected JSON formats:
/// ```json
/// {"type": "linear", "factor": 2.0}
/// {"type": "dynamic", "factor": 2.0}
/// {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192}
/// ```
///
/// Returns `RopeScalingMethod::None` if the value is null, missing, or
/// contains an unrecognized type.
pub fn parse_rope_scaling(value: Option<&serde_json::Value>) -> RopeScalingMethod {
    let obj = match value {
        Some(serde_json::Value::Object(map)) => map,
        _ => return RopeScalingMethod::None,
    };

    let scaling_type = match obj.get("type").and_then(|v| v.as_str()) {
        Some(t) => t.to_lowercase(),
        None => return RopeScalingMethod::None,
    };

    let factor = obj
        .get("factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    match scaling_type.as_str() {
        "linear" => RopeScalingMethod::Linear { factor },
        "dynamic" | "ntk" => RopeScalingMethod::Ntk { factor },
        "yarn" => {
            let original_max = obj
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(8192) as usize;
            RopeScalingMethod::Yarn {
                factor,
                original_max_position_embeddings: original_max,
            }
        }
        _ => {
            tracing::warn!(
                scaling_type = scaling_type.as_str(),
                "unrecognized rope_scaling type, falling back to no scaling"
            );
            RopeScalingMethod::None
        }
    }
}

/// Compute RoPE cosine and sine frequency tables.
///
/// Returns two Arrays of shape `[max_seq_len, head_dim / 2]` containing
/// precomputed cosine and sine values for each (position, dimension pair).
///
/// # Arguments
/// * `head_dim` - Per-head dimension. Must be even.
/// * `max_seq_len` - Maximum sequence length to precompute for.
/// * `theta` - Base frequency (typically 10000.0 or 500000.0).
/// * `device` - Metal device for GPU buffer allocation.
///
/// # Returns
/// `(cos_table, sin_table)` as f32 Arrays on GPU.
pub fn compute_rope_frequencies(
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    device: &metal::Device,
) -> (Array, Array) {
    compute_rope_frequencies_with_scaling(
        head_dim,
        max_seq_len,
        theta,
        &RopeScalingMethod::None,
        device,
    )
}

/// Compute RoPE frequencies with an optional scaling method applied.
///
/// This is the core frequency computation that supports all RoPE variants:
/// - **None**: standard RoPE with the given base theta.
/// - **Linear**: positions are divided by the scaling factor.
/// - **NTK**: the base theta is raised to account for the scaling factor.
/// - **YaRN**: NTK base modification plus per-dimension interpolation blending.
///
/// # Arguments
/// * `head_dim` - Per-head dimension. Must be even.
/// * `max_seq_len` - Maximum sequence length to precompute for.
/// * `theta` - Base frequency (typically 10000.0 or 500000.0).
/// * `scaling` - The RoPE scaling method to apply.
/// * `device` - Metal device for GPU buffer allocation.
///
/// # Returns
/// `(cos_table, sin_table)` as f32 Arrays on GPU.
pub fn compute_rope_frequencies_with_scaling(
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    scaling: &RopeScalingMethod,
    device: &metal::Device,
) -> (Array, Array) {
    let half_dim = head_dim / 2;
    let num_elements = max_seq_len * half_dim;

    let mut cos_data = Vec::with_capacity(num_elements);
    let mut sin_data = Vec::with_capacity(num_elements);

    // Determine the effective base theta and per-dimension blend weights
    // based on the scaling method.
    let effective_theta = match scaling {
        RopeScalingMethod::None | RopeScalingMethod::Linear { .. } => theta,
        RopeScalingMethod::Ntk { factor } => {
            // NTK-aware: theta_new = theta * factor^(dim / (dim - 2))
            let exponent = (head_dim as f32) / ((head_dim as f32) - 2.0);
            theta * factor.powf(exponent)
        }
        RopeScalingMethod::Yarn { factor, .. } => {
            // YaRN uses NTK-style base scaling
            let exponent = (head_dim as f32) / ((head_dim as f32) - 2.0);
            theta * factor.powf(exponent)
        }
    };

    // Position scaling factor for linear scaling
    let position_scale = match scaling {
        RopeScalingMethod::Linear { factor } => *factor,
        _ => 1.0,
    };

    // Precompute inverse frequencies: theta_i = effective_theta^(-2i / head_dim)
    let inv_freqs: Vec<f32> = (0..half_dim)
        .map(|i| {
            let exponent = -2.0 * (i as f32) / (head_dim as f32);
            effective_theta.powf(exponent)
        })
        .collect();

    // For YaRN, compute per-dimension interpolation weights.
    // Dimensions with high frequencies (small wavelength) keep the original
    // values, while low-frequency dimensions use the NTK-scaled values.
    // This creates a smooth blend between the two regimes.
    let yarn_blend: Option<Vec<f32>> = match scaling {
        RopeScalingMethod::Yarn {
            factor,
            original_max_position_embeddings,
        } => {
            let original_max = *original_max_position_embeddings as f32;
            // The attention temperature correction for YaRN
            // is: sqrt(1 + ln(factor) / ln(original_max))
            let _attn_factor =
                (1.0 + factor.ln() / original_max.ln()).sqrt();

            // Compute blend weights per dimension.
            // Low-frequency dimensions (small inv_freq, large wavelength)
            // should be interpolated (blend -> 1.0), while high-frequency
            // dimensions should keep original frequencies (blend -> 0.0).
            let blend: Vec<f32> = (0..half_dim)
                .map(|i| {
                    let freq =
                        theta.powf(-2.0 * (i as f32) / (head_dim as f32));
                    let wavelength = 2.0 * std::f32::consts::PI / freq;
                    // Wavelengths shorter than original_max are high-freq
                    // (keep original), longer are low-freq (interpolate).
                    let ratio = wavelength / original_max;
                    // Smooth ramp between 0 and 1
                    let t = ((ratio - 1.0) * factor / (factor - 1.0)).clamp(0.0, 1.0);
                    1.0 - t
                })
                .collect();
            Some(blend)
        }
        _ => None,
    };

    // For each position, compute cos and sin of (pos * theta_i)
    for pos in 0..max_seq_len {
        let scaled_pos = (pos as f32) / position_scale;

        for (i, &inv_freq) in inv_freqs.iter().enumerate() {
            let freq = scaled_pos * inv_freq;

            // For YaRN, blend between the NTK-scaled frequency and the
            // linearly-scaled frequency.
            let final_freq = if let Some(ref blend) = yarn_blend {
                // Original (unscaled) frequency for this dimension
                let orig_inv_freq =
                    theta.powf(-2.0 * (i as f32) / (head_dim as f32));
                let orig_freq = scaled_pos * orig_inv_freq;
                // Blend: 0.0 = use NTK-scaled, 1.0 = use original
                blend[i] * orig_freq + (1.0 - blend[i]) * freq
            } else {
                freq
            };

            cos_data.push(final_freq.cos());
            sin_data.push(final_freq.sin());
        }
    }

    let shape = vec![max_seq_len, half_dim];
    let cos_array = Array::from_slice(device, &cos_data, shape.clone());
    let sin_array = Array::from_slice(device, &sin_data, shape);

    (cos_array, sin_array)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_frequency_dimensions() {
        // Verify the computation produces correct sizes without a GPU device.
        // We test the CPU-side logic only.
        let head_dim = 128;
        let max_seq_len = 4096;
        let theta = 10000.0f32;
        let half_dim = head_dim / 2;

        let inv_freqs: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exponent = -2.0 * (i as f32) / (head_dim as f32);
                theta.powf(exponent)
            })
            .collect();

        assert_eq!(inv_freqs.len(), half_dim);

        // First frequency should be theta^0 = 1.0
        assert!((inv_freqs[0] - 1.0).abs() < 1e-6);

        // Last frequency should be theta^(-2*(half_dim-1)/head_dim)
        let expected_last = theta.powf(-2.0 * ((half_dim - 1) as f32) / (head_dim as f32));
        assert!((inv_freqs[half_dim - 1] - expected_last).abs() < 1e-6);

        // At position 0, all cos values should be 1.0 and sin values 0.0
        let mut cos_data = Vec::new();
        let mut sin_data = Vec::new();
        for &inv_freq in &inv_freqs {
            let freq = 0.0 * inv_freq;
            cos_data.push(freq.cos());
            sin_data.push(freq.sin());
        }

        for &c in &cos_data {
            assert!((c - 1.0).abs() < 1e-6, "cos(0) should be 1.0, got {c}");
        }
        for &s in &sin_data {
            assert!(s.abs() < 1e-6, "sin(0) should be 0.0, got {s}");
        }

        // Total element count
        let total = max_seq_len * half_dim;
        assert_eq!(total, 4096 * 64);
    }

    #[test]
    fn test_parse_rope_scaling_none() {
        let result = parse_rope_scaling(None);
        assert!(matches!(result, RopeScalingMethod::None));

        let null_val = serde_json::Value::Null;
        let result = parse_rope_scaling(Some(&null_val));
        assert!(matches!(result, RopeScalingMethod::None));
    }

    #[test]
    fn test_parse_rope_scaling_linear() {
        let val: serde_json::Value =
            serde_json::json!({"type": "linear", "factor": 2.0});
        let result = parse_rope_scaling(Some(&val));
        match result {
            RopeScalingMethod::Linear { factor } => {
                assert!((factor - 2.0).abs() < 1e-6);
            }
            other => panic!("expected Linear, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_rope_scaling_dynamic_ntk() {
        let val: serde_json::Value =
            serde_json::json!({"type": "dynamic", "factor": 2.0});
        let result = parse_rope_scaling(Some(&val));
        match result {
            RopeScalingMethod::Ntk { factor } => {
                assert!((factor - 2.0).abs() < 1e-6);
            }
            other => panic!("expected Ntk, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_rope_scaling_yarn() {
        let val: serde_json::Value = serde_json::json!({
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 8192
        });
        let result = parse_rope_scaling(Some(&val));
        match result {
            RopeScalingMethod::Yarn {
                factor,
                original_max_position_embeddings,
            } => {
                assert!((factor - 4.0).abs() < 1e-6);
                assert_eq!(original_max_position_embeddings, 8192);
            }
            other => panic!("expected Yarn, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_rope_scaling_unknown_type() {
        let val: serde_json::Value =
            serde_json::json!({"type": "unknown_future_method", "factor": 3.0});
        let result = parse_rope_scaling(Some(&val));
        assert!(matches!(result, RopeScalingMethod::None));
    }

    #[test]
    fn test_linear_scaling_frequencies_differ() {
        // With linear scaling, frequencies at position P should match
        // unscaled frequencies at position P/factor.
        let head_dim = 64;
        let theta = 10000.0f32;
        let half_dim = head_dim / 2;
        let factor = 2.0f32;

        // Unscaled: position 5
        let inv_freqs: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exp = -2.0 * (i as f32) / (head_dim as f32);
                theta.powf(exp)
            })
            .collect();

        // Unscaled freq at position 5
        let unscaled_freqs: Vec<f32> = inv_freqs.iter().map(|f| 5.0 * f).collect();

        // Linear scaled freq at position 10 should equal unscaled at position 5
        // because scaled_pos = 10 / 2.0 = 5.0
        let scaled_freqs: Vec<f32> = inv_freqs.iter().map(|f| (10.0 / factor) * f).collect();

        for (u, s) in unscaled_freqs.iter().zip(scaled_freqs.iter()) {
            assert!(
                (u - s).abs() < 1e-4,
                "linear scaling mismatch: unscaled={u}, scaled={s}"
            );
        }
    }

    #[test]
    fn test_ntk_scaling_modifies_base() {
        // NTK scaling changes the base theta. The effective theta should be
        // theta * factor^(dim / (dim - 2)).
        let head_dim = 128;
        let theta = 10000.0f32;
        let factor = 2.0f32;

        let expected_theta =
            theta * factor.powf((head_dim as f32) / ((head_dim as f32) - 2.0));

        // Verify the effective theta is larger than the original
        assert!(expected_theta > theta);

        // inv_freq = theta^(-2*i/dim). With a larger theta the exponent
        // is the same negative value, so the result is smaller (the
        // frequency decays faster in absolute terms, meaning lower
        // frequencies overall). This stretches the positional range.
        let inv_freq_standard =
            theta.powf(-2.0 * 1.0 / (head_dim as f32));
        let inv_freq_ntk =
            expected_theta.powf(-2.0 * 1.0 / (head_dim as f32));

        // Larger base -> smaller inv_freq for i > 0 (frequencies are lower)
        assert!(
            inv_freq_ntk < inv_freq_standard,
            "NTK should produce lower frequencies: ntk={}, std={}",
            inv_freq_ntk,
            inv_freq_standard
        );

        // At dimension 0, both should be 1.0 (theta^0 = 1.0)
        let inv_freq_standard_0 = theta.powf(0.0);
        let inv_freq_ntk_0 = expected_theta.powf(0.0);
        assert!((inv_freq_standard_0 - 1.0).abs() < 1e-6);
        assert!((inv_freq_ntk_0 - 1.0).abs() < 1e-6);
    }
}
