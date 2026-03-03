//! Common cache operations and traits.
//!
//! Provides the [`CacheOps`] trait for unified cache interaction (update + fetch),
//! and utility functions like [`make_causal_mask`] for attention mask generation.

/// Trait for caches that support an atomic update-and-fetch operation.
///
/// This is useful for single-layer cache access patterns where the caller wants
/// to write new key/value data and immediately retrieve the full cache state
/// (including the newly written data) in one call.
pub trait CacheOps {
    /// Atomically write new KV data for a single head at the given position and
    /// return the full cached key and value slices (including the new data).
    ///
    /// # Arguments
    /// * `key` - New key data for a single token, length `head_dim`.
    /// * `value` - New value data for a single token, length `head_dim`.
    /// * `position` - The logical position (token index) to write into.
    ///
    /// # Returns
    /// A tuple of `(keys, values)` slices covering the full cache from position 0
    /// through (and including) the newly written position.
    fn update_and_fetch(&mut self, key: &[f32], value: &[f32], position: usize)
        -> (&[f32], &[f32]);
}

/// Generate a causal attention mask for the given sequence length.
///
/// Returns a `seq_len x (seq_len + offset)` matrix where position `(i, j)`
/// is `0.0` if token `i` can attend to position `j`, and `-f32::INFINITY`
/// (masked) otherwise. The mask is lower-triangular: token `i` can attend
/// to positions `0..=(i + offset)`.
///
/// The `offset` parameter accounts for cached KV tokens. When generating
/// token at position `p` with `offset` cached tokens, the mask ensures
/// the new token can attend to all `offset + seq_len` positions up to
/// its own position.
///
/// # Examples
/// ```
/// use rmlx_serve_cache::make_causal_mask;
///
/// // Simple 3-token mask with no offset
/// let mask = make_causal_mask(3, 0);
/// assert_eq!(mask.len(), 3);
/// assert_eq!(mask[0].len(), 3);
/// // Token 0 can only see position 0
/// assert_eq!(mask[0][0], 0.0);
/// assert!(mask[0][1].is_infinite());
/// // Token 2 can see positions 0, 1, 2
/// assert_eq!(mask[2], vec![0.0, 0.0, 0.0]);
/// ```
pub fn make_causal_mask(seq_len: usize, offset: usize) -> Vec<Vec<f32>> {
    let total_len = seq_len + offset;
    let mut mask = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        let mut row = Vec::with_capacity(total_len);
        for j in 0..total_len {
            // Token at query position `i` (logical position `i + offset`) can
            // attend to key position `j` if `j <= i + offset`.
            if j <= i + offset {
                row.push(0.0);
            } else {
                row.push(f32::NEG_INFINITY);
            }
        }
        mask.push(row);
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_no_offset() {
        let mask = make_causal_mask(3, 0);
        assert_eq!(mask.len(), 3);
        // Row 0: can see [0] only
        assert_eq!(mask[0], vec![0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        // Row 1: can see [0, 1]
        assert_eq!(mask[1], vec![0.0, 0.0, f32::NEG_INFINITY]);
        // Row 2: can see [0, 1, 2]
        assert_eq!(mask[2], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_causal_mask_with_offset() {
        // 2 new tokens, 3 cached tokens
        let mask = make_causal_mask(2, 3);
        assert_eq!(mask.len(), 2);
        assert_eq!(mask[0].len(), 5); // 2 + 3
                                      // Row 0 (logical pos 3): can see [0..3]
        assert_eq!(mask[0], vec![0.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY]);
        // Row 1 (logical pos 4): can see [0..4]
        assert_eq!(mask[1], vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_causal_mask_single_token() {
        let mask = make_causal_mask(1, 5);
        assert_eq!(mask.len(), 1);
        assert_eq!(mask[0].len(), 6);
        // Single decoding token can attend to all 6 positions
        assert_eq!(mask[0], vec![0.0; 6]);
    }

    #[test]
    fn test_causal_mask_empty() {
        let mask = make_causal_mask(0, 0);
        assert!(mask.is_empty());
    }
}
