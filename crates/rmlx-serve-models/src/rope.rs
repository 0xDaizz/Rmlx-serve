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
    let half_dim = head_dim / 2;
    let num_elements = max_seq_len * half_dim;

    let mut cos_data = Vec::with_capacity(num_elements);
    let mut sin_data = Vec::with_capacity(num_elements);

    // Precompute inverse frequencies: theta_i = theta^(-2i / head_dim)
    let inv_freqs: Vec<f32> = (0..half_dim)
        .map(|i| {
            let exponent = -2.0 * (i as f32) / (head_dim as f32);
            theta.powf(exponent)
        })
        .collect();

    // For each position, compute cos and sin of (pos * theta_i)
    for pos in 0..max_seq_len {
        for &inv_freq in &inv_freqs {
            let freq = (pos as f32) * inv_freq;
            cos_data.push(freq.cos());
            sin_data.push(freq.sin());
        }
    }

    let shape = vec![max_seq_len, half_dim];
    let cos_array = Array::from_slice(device, &cos_data, shape.clone());
    let sin_array = Array::from_slice(device, &sin_data, shape);

    (cos_array, sin_array)
}

#[cfg(test)]
mod tests {
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
}
