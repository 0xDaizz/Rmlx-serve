//! Rotating (circular buffer) KV cache.
//!
//! Ported from mlx-lm's `cache.py` `RotatingKVCache`. This cache has a fixed
//! maximum size and overwrites the oldest entries (beyond `keep`) when full.
//! The `keep` parameter ensures that the first `keep` tokens (typically the
//! system prompt) are never overwritten.

/// Circular buffer KV cache with a fixed maximum size.
///
/// Stores key/value data in CPU memory as flat `f32` vectors.
/// Each layer has `num_kv_heads` heads, each head stores up to
/// `max_size * head_dim` floats in a circular buffer.
///
/// The first `keep` tokens are pinned and never overwritten.
/// New tokens beyond `max_size` overwrite the oldest non-pinned entries.
pub struct RotatingKVCache {
    /// `[layer][head][max_size * head_dim]` -- key storage.
    keys: Vec<Vec<Vec<f32>>>,
    /// `[layer][head][max_size * head_dim]` -- value storage.
    values: Vec<Vec<Vec<f32>>>,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_size: usize,
    /// Number of leading tokens that are always kept (never overwritten).
    keep: usize,
    /// Write offset into the rotating region (0-based within the rotating window).
    offset: usize,
    /// Total number of tokens that have been written (may exceed max_size).
    written: usize,
}

impl RotatingKVCache {
    /// Create a new rotating KV cache.
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers.
    /// * `num_kv_heads` - Number of KV heads per layer.
    /// * `head_dim` - Dimension of each head.
    /// * `max_size` - Maximum number of tokens the cache can hold.
    /// * `keep` - Number of leading tokens to always keep (never overwrite).
    ///
    /// # Panics
    /// Panics if `keep >= max_size`.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_size: usize,
        keep: usize,
    ) -> Self {
        assert!(
            keep < max_size,
            "keep ({keep}) must be less than max_size ({max_size})"
        );

        let buf_size = max_size * head_dim;
        let keys = (0..num_layers)
            .map(|_| (0..num_kv_heads).map(|_| vec![0.0f32; buf_size]).collect())
            .collect();
        let values = (0..num_layers)
            .map(|_| (0..num_kv_heads).map(|_| vec![0.0f32; buf_size]).collect())
            .collect();

        Self {
            keys,
            values,
            num_layers,
            num_kv_heads,
            head_dim,
            max_size,
            keep,
            offset: 0,
            written: 0,
        }
    }

    /// Append new key/value data for a single layer.
    ///
    /// `new_keys` and `new_values` are flat arrays of length
    /// `num_kv_heads * new_tokens * head_dim`, laid out as
    /// `[head0_tok0, head0_tok1, ..., head1_tok0, ...]`.
    ///
    /// If the cache is not yet full, tokens are appended linearly.
    /// Once full, tokens rotate through the non-pinned region.
    pub fn append(
        &mut self,
        layer: usize,
        new_keys: &[f32],
        new_values: &[f32],
        new_tokens: usize,
    ) {
        let expected_len = self.num_kv_heads * new_tokens * self.head_dim;
        assert_eq!(
            new_keys.len(),
            expected_len,
            "new_keys length mismatch: expected {expected_len}, got {}",
            new_keys.len()
        );
        assert_eq!(
            new_values.len(),
            expected_len,
            "new_values length mismatch: expected {expected_len}, got {}",
            new_values.len()
        );

        let hd = self.head_dim;
        let rotating_size = self.max_size - self.keep;

        for tok in 0..new_tokens {
            // Determine the write position for this token.
            let write_pos = if self.written < self.max_size {
                // Still filling the buffer linearly.
                self.written
            } else {
                // Rotating region: write at keep + offset.
                self.keep + self.offset
            };

            for head in 0..self.num_kv_heads {
                let src_start = (head * new_tokens + tok) * hd;
                let dst_start = write_pos * hd;

                self.keys[layer][head][dst_start..dst_start + hd]
                    .copy_from_slice(&new_keys[src_start..src_start + hd]);
                self.values[layer][head][dst_start..dst_start + hd]
                    .copy_from_slice(&new_values[src_start..src_start + hd]);
            }

            if self.written >= self.max_size {
                self.offset = (self.offset + 1) % rotating_size;
            }
            self.written += 1;
        }
    }

    /// Get the cached keys for a layer in logical order.
    ///
    /// Returns one `Vec<f32>` per head, each of length `seq_len * head_dim`,
    /// with tokens in their logical (not physical) order.
    pub fn get_keys(&self, layer: usize) -> Vec<Vec<f32>> {
        self.get_logical_order(layer, &self.keys)
    }

    /// Get the cached values for a layer in logical order.
    ///
    /// Returns one `Vec<f32>` per head, each of length `seq_len * head_dim`,
    /// with tokens in their logical (not physical) order.
    pub fn get_values(&self, layer: usize) -> Vec<Vec<f32>> {
        self.get_logical_order(layer, &self.values)
    }

    /// Internal: reconstruct logical ordering from the circular buffer.
    fn get_logical_order(&self, layer: usize, storage: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
        let hd = self.head_dim;
        let effective_len = self.seq_len();
        let mut result = Vec::with_capacity(self.num_kv_heads);

        for buf in storage[layer].iter().take(self.num_kv_heads) {
            let mut out = Vec::with_capacity(effective_len * hd);

            if self.written <= self.max_size {
                // No rotation happened yet: data is already in order.
                out.extend_from_slice(&buf[..effective_len * hd]);
            } else {
                // First: the pinned `keep` tokens.
                out.extend_from_slice(&buf[..self.keep * hd]);

                // Then: the rotating region in logical order.
                // The oldest non-overwritten token is at position `keep + offset`.
                let rotating_size = self.max_size - self.keep;
                let rotating_start = self.keep * hd;
                for i in 0..rotating_size {
                    let physical_idx = (self.offset + i) % rotating_size;
                    let start = rotating_start + physical_idx * hd;
                    out.extend_from_slice(&buf[start..start + hd]);
                }
            }

            result.push(out);
        }

        result
    }

    /// Current effective sequence length.
    ///
    /// Capped at `max_size` once the buffer is full.
    pub fn seq_len(&self) -> usize {
        self.written.min(self.max_size)
    }

    /// Remove the oldest `n` tokens from the rotating region.
    ///
    /// This effectively "forgets" the oldest tokens beyond the `keep` region.
    /// Does not actually zero the memory -- just adjusts bookkeeping.
    ///
    /// # Panics
    /// Panics if `n` exceeds the number of trimmable tokens.
    pub fn trim(&mut self, n: usize) {
        let trimmable = self.seq_len().saturating_sub(self.keep);
        assert!(
            n <= trimmable,
            "cannot trim {n} tokens: only {trimmable} trimmable"
        );

        if self.written <= self.max_size {
            // Not yet rotating. We can just reduce `written`.
            // Move the offset forward so get_logical_order still works when
            // we later start rotating. For the linear case, we shift the
            // rotating region start by advancing `offset` and adjusting
            // `written`.
            //
            // Simplest approach: reduce written, which reduces seq_len.
            // The trimmed tokens are at positions [keep .. keep + n).
            // We need to shift data in each head buffer to close the gap.
            let hd = self.head_dim;
            for layer_bufs in &mut self.keys {
                for head_buf in layer_bufs {
                    let src_start = (self.keep + n) * hd;
                    let dst_start = self.keep * hd;
                    let remaining = (self.written - self.keep - n) * hd;
                    if remaining > 0 {
                        head_buf.copy_within(src_start..src_start + remaining, dst_start);
                    }
                }
            }
            for layer_bufs in &mut self.values {
                for head_buf in layer_bufs {
                    let src_start = (self.keep + n) * hd;
                    let dst_start = self.keep * hd;
                    let remaining = (self.written - self.keep - n) * hd;
                    if remaining > 0 {
                        head_buf.copy_within(src_start..src_start + remaining, dst_start);
                    }
                }
            }
            self.written -= n;
        } else {
            // Already rotating. Advance the offset to skip the oldest tokens.
            let rotating_size = self.max_size - self.keep;
            self.offset = (self.offset + n) % rotating_size;
            // Conceptually the written count stays the same but the effective
            // window shifts. We reduce written to reflect fewer visible tokens.
            // Since seq_len = min(written, max_size), and we want to reduce
            // the effective seq_len by n, we reduce written by n.
            self.written -= n;
        }
    }

    /// Whether the cache can be trimmed (has tokens beyond `keep`).
    pub fn can_trim(&self) -> bool {
        self.seq_len() > self.keep
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Maximum cache size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Number of pinned (always-kept) leading tokens.
    pub fn keep(&self) -> usize {
        self.keep
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_rotating_cache() {
        let cache = RotatingKVCache::new(2, 4, 8, 16, 2);
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.num_layers(), 2);
        assert_eq!(cache.max_size(), 16);
        assert_eq!(cache.keep(), 2);
        assert!(!cache.can_trim());
    }

    #[test]
    fn test_append_linear() {
        let mut cache = RotatingKVCache::new(1, 1, 2, 4, 0);
        // Append 2 tokens: [1.0, 2.0, 3.0, 4.0] for 1 head, 2 tokens, head_dim=2
        let keys = vec![1.0, 2.0, 3.0, 4.0];
        let values = vec![10.0, 20.0, 30.0, 40.0];
        cache.append(0, &keys, &values, 2);
        assert_eq!(cache.seq_len(), 2);

        let k = cache.get_keys(0);
        assert_eq!(k.len(), 1); // 1 head
        assert_eq!(k[0], vec![1.0, 2.0, 3.0, 4.0]);

        let v = cache.get_values(0);
        assert_eq!(v[0], vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_append_with_rotation() {
        // max_size=3, keep=1, head_dim=1, 1 head, 1 layer
        let mut cache = RotatingKVCache::new(1, 1, 1, 3, 1);

        // Fill: token 0 (pinned), token 1, token 2
        for i in 0..3 {
            let keys = vec![(i + 1) as f32];
            let values = vec![(i + 1) as f32 * 10.0];
            cache.append(0, &keys, &values, 1);
        }
        assert_eq!(cache.seq_len(), 3);

        // Logical order: [1, 2, 3]
        let k = cache.get_keys(0);
        assert_eq!(k[0], vec![1.0, 2.0, 3.0]);

        // Now add token 4 -- should overwrite token 1 (pos 1, the oldest non-pinned)
        cache.append(0, &[4.0], &[40.0], 1);
        assert_eq!(cache.seq_len(), 3); // still capped at max_size

        // Logical order: [1(pinned), 3, 4] (token 2 was overwritten, then
        // offset advanced, so 3 is oldest, then 4 is newest)
        let k = cache.get_keys(0);
        assert_eq!(k[0], vec![1.0, 3.0, 4.0]);
    }

    #[test]
    fn test_trim() {
        let mut cache = RotatingKVCache::new(1, 1, 1, 8, 2);
        // Fill 6 tokens
        for i in 0..6 {
            cache.append(0, &[(i + 1) as f32], &[(i + 1) as f32], 1);
        }
        assert_eq!(cache.seq_len(), 6);
        assert!(cache.can_trim());

        // Trim 2 oldest non-pinned tokens
        cache.trim(2);
        assert_eq!(cache.seq_len(), 4);

        // Remaining should be: [1, 2, 5, 6] (kept=1,2; trimmed=3,4; remaining=5,6)
        let k = cache.get_keys(0);
        assert_eq!(k[0], vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "keep (5) must be less than max_size (5)")]
    fn test_keep_equals_max_size_panics() {
        let _ = RotatingKVCache::new(1, 1, 1, 5, 5);
    }
}
