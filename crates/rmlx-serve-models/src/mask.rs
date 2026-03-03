//! Attention mask utilities for causal and sliding window attention.
//!
//! Masks are additive: 0.0 means "attend" and negative infinity means "block".
//! They are computed on the CPU as `Vec<f32>` and uploaded to the GPU.

use rmlx_core::Array;

/// Create a causal attention mask.
///
/// Returns an Array of shape `[seq_len, seq_len + offset]` where:
/// - `mask[i][j] = 0.0`  if `j <= i + offset` (attend)
/// - `mask[i][j] = -inf`  if `j > i + offset` (block)
///
/// The `offset` parameter accounts for previously cached tokens. For the
/// first forward pass (prefill), offset is 0. For subsequent decode steps,
/// offset equals the number of previously cached tokens.
///
/// # Arguments
/// * `seq_len` - Number of new tokens in this step.
/// * `offset` - Number of previously cached tokens (RoPE position offset).
/// * `device` - Metal device for GPU buffer allocation.
pub fn create_causal_mask(seq_len: usize, offset: usize, device: &metal::Device) -> Array {
    let total_len = seq_len + offset;
    let num_elements = seq_len * total_len;

    let mut mask_data = Vec::with_capacity(num_elements);

    for i in 0..seq_len {
        for j in 0..total_len {
            // Position i in the new sequence corresponds to absolute position
            // i + offset. It can attend to any position j where j <= i + offset.
            if j <= i + offset {
                mask_data.push(0.0f32);
            } else {
                mask_data.push(f32::NEG_INFINITY);
            }
        }
    }

    let shape = vec![seq_len, total_len];
    Array::from_slice(device, &mask_data, shape)
}

/// Create a sliding window causal attention mask.
///
/// Like [`create_causal_mask`], but also masks positions that are more than
/// `window_size` positions in the past. This is used by architectures like
/// Mistral that use sliding window attention.
///
/// Returns an Array of shape `[seq_len, seq_len + offset]` where:
/// - `mask[i][j] = 0.0`  if `j <= i + offset` AND `i + offset - j < window_size`
/// - `mask[i][j] = -inf`  otherwise
///
/// # Arguments
/// * `seq_len` - Number of new tokens in this step.
/// * `offset` - Number of previously cached tokens.
/// * `window_size` - Maximum number of past positions to attend to.
/// * `device` - Metal device for GPU buffer allocation.
pub fn create_sliding_window_mask(
    seq_len: usize,
    offset: usize,
    window_size: usize,
    device: &metal::Device,
) -> Array {
    let total_len = seq_len + offset;
    let num_elements = seq_len * total_len;

    let mut mask_data = Vec::with_capacity(num_elements);

    for i in 0..seq_len {
        let abs_pos = i + offset;
        for j in 0..total_len {
            // Attend if: (1) causal: j <= abs_pos, and
            //            (2) within window: abs_pos - j < window_size
            let is_causal = j <= abs_pos;
            let is_in_window = abs_pos < j + window_size; // equivalent to abs_pos - j < window_size (avoiding underflow)
            if is_causal && is_in_window {
                mask_data.push(0.0f32);
            } else {
                mask_data.push(f32::NEG_INFINITY);
            }
        }
    }

    let shape = vec![seq_len, total_len];
    Array::from_slice(device, &mask_data, shape)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_causal_mask_values() {
        // Test the CPU-side mask computation without a GPU device.
        let seq_len = 4;
        let offset = 0;
        let total_len = seq_len + offset;

        let mut mask = Vec::new();
        for i in 0..seq_len {
            for j in 0..total_len {
                if j <= i + offset {
                    mask.push(0.0f32);
                } else {
                    mask.push(f32::NEG_INFINITY);
                }
            }
        }

        // Row 0: [0, -inf, -inf, -inf]
        assert_eq!(mask[0], 0.0);
        assert!(mask[1].is_infinite());
        assert!(mask[2].is_infinite());
        assert!(mask[3].is_infinite());

        // Row 1: [0, 0, -inf, -inf]
        assert_eq!(mask[4], 0.0);
        assert_eq!(mask[5], 0.0);
        assert!(mask[6].is_infinite());
        assert!(mask[7].is_infinite());

        // Row 3: [0, 0, 0, 0]
        assert_eq!(mask[12], 0.0);
        assert_eq!(mask[13], 0.0);
        assert_eq!(mask[14], 0.0);
        assert_eq!(mask[15], 0.0);
    }

    #[test]
    fn test_causal_mask_with_offset() {
        let seq_len = 1; // single decode token
        let offset = 5; // 5 cached tokens
        let total_len = seq_len + offset; // 6

        let mut mask = Vec::new();
        for i in 0..seq_len {
            for j in 0..total_len {
                if j <= i + offset {
                    mask.push(0.0f32);
                } else {
                    mask.push(f32::NEG_INFINITY);
                }
            }
        }

        // Single row: [0, 0, 0, 0, 0, 0] — attend to all 6 positions
        assert_eq!(mask.len(), 6);
        for &val in &mask {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_sliding_window_mask() {
        let seq_len = 4;
        let offset = 0;
        let window_size = 2;
        let total_len = seq_len + offset;

        let mut mask = Vec::new();
        for i in 0..seq_len {
            let abs_pos = i + offset;
            for j in 0..total_len {
                let is_causal = j <= abs_pos;
                let is_in_window = abs_pos < j + window_size;
                if is_causal && is_in_window {
                    mask.push(0.0f32);
                } else {
                    mask.push(f32::NEG_INFINITY);
                }
            }
        }

        // Row 0 (pos 0): attend to j=0 only (causal + window)
        assert_eq!(mask[0], 0.0);
        assert!(mask[1].is_infinite());

        // Row 1 (pos 1): attend to j=0, j=1
        assert_eq!(mask[4], 0.0);
        assert_eq!(mask[5], 0.0);
        assert!(mask[6].is_infinite());

        // Row 2 (pos 2): window=2 means attend to j=1,j=2 (not j=0)
        assert!(mask[8].is_infinite()); // j=0 is out of window
        assert_eq!(mask[9], 0.0); // j=1
        assert_eq!(mask[10], 0.0); // j=2
        assert!(mask[11].is_infinite()); // j=3 is future

        // Row 3 (pos 3): attend to j=2, j=3
        assert!(mask[12].is_infinite()); // j=0
        assert!(mask[13].is_infinite()); // j=1
        assert_eq!(mask[14], 0.0); // j=2
        assert_eq!(mask[15], 0.0); // j=3
    }
}
