//! Standard KV cache wrapping `rmlx_nn::LayerKvCache`.
//!
//! This is the primary cache type, providing a multi-layer wrapper around
//! the per-layer KV cache from `rmlx_nn`. Ported from mlx-lm's `cache.py`
//! `KVCache` class.

use rmlx_core::DType;
use rmlx_metal::metal;
use rmlx_nn::LayerKvCache;

/// Standard KV cache for transformer inference.
///
/// Wraps a `Vec<LayerKvCache>`, one per transformer layer. Provides
/// convenience methods for creating, querying, and mutating the cache
/// across all layers.
pub struct KVCache {
    /// Per-layer caches. Public for direct iteration when needed.
    pub inner: Vec<LayerKvCache>,
    num_layers: usize,
}

impl KVCache {
    /// Create a new empty (non-pre-allocated) KV cache.
    ///
    /// Each layer starts with zero cached tokens and no allocated buffers.
    /// Buffers will be allocated on first `append` call via the legacy
    /// concat path in `LayerKvCache`.
    pub fn new(num_layers: usize, num_kv_heads: usize) -> Self {
        let inner = (0..num_layers)
            .map(|_| LayerKvCache::new(num_kv_heads))
            .collect();
        Self { inner, num_layers }
    }

    /// Create a pre-allocated KV cache with room for `max_seq_len` tokens.
    ///
    /// Each layer gets a contiguous `[max_seq_len, head_dim]` buffer per
    /// KV head, allocated up front on the given Metal device. Subsequent
    /// `append` calls write into the next slot(s) with O(1) cost per token.
    pub fn preallocated(
        device: &metal::Device,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> Self {
        let inner = (0..num_layers)
            .map(|_| {
                LayerKvCache::preallocated(device, num_kv_heads, head_dim, max_seq_len, dtype)
            })
            .collect();
        Self { inner, num_layers }
    }

    /// Current sequence length (number of cached tokens).
    ///
    /// Returns the `seq_len` of the first layer. All layers are assumed
    /// to have the same sequence length.
    pub fn seq_len(&self) -> usize {
        self.inner.first().map_or(0, |layer| layer.seq_len)
    }

    /// Whether the cache is empty (no tokens cached in any layer).
    pub fn is_empty(&self) -> bool {
        self.inner.first().map_or(true, |layer| layer.is_empty())
    }

    /// Number of transformer layers in this cache.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Mutable reference to a single layer's cache.
    ///
    /// # Panics
    /// Panics if `idx >= num_layers`.
    pub fn layer_mut(&mut self, idx: usize) -> &mut LayerKvCache {
        &mut self.inner[idx]
    }

    /// Mutable slice of all layer caches.
    pub fn layers_mut(&mut self) -> &mut [LayerKvCache] {
        &mut self.inner
    }

    /// Immutable reference to a single layer's cache.
    ///
    /// # Panics
    /// Panics if `idx >= num_layers`.
    pub fn layer(&self, idx: usize) -> &LayerKvCache {
        &self.inner[idx]
    }

    /// Immutable slice of all layer caches.
    pub fn layers(&self) -> &[LayerKvCache] {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty_cache() {
        let cache = KVCache::new(32, 8);
        assert_eq!(cache.num_layers(), 32);
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_layer_access() {
        let mut cache = KVCache::new(4, 8);
        assert_eq!(cache.layers().len(), 4);
        assert_eq!(cache.layers_mut().len(), 4);
        let _ = cache.layer_mut(0);
        let _ = cache.layer(3);
    }
}
