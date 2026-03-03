//! rmlx-serve-cache: KV cache management for rmlx-serve inference engine.
//!
//! This crate provides multiple KV cache strategies for transformer inference:
//!
//! - [`KVCache`] -- Standard KV cache wrapping `rmlx_nn::LayerKvCache` with
//!   optional pre-allocation for O(1) append.
//! - [`RotatingKVCache`] -- Circular buffer cache with a fixed maximum size,
//!   supporting pinned leading tokens (e.g., system prompts).
//! - [`QuantizedKVCache`] -- Memory-efficient cache with 4-bit or 8-bit
//!   quantized storage and per-group scale factors.
//! - [`BatchKVCache`] -- Manages KV caches for multiple sequences in a batch.
//! - [`PagedCacheManager`] -- Block-level allocation with reference counting
//!   and copy-on-write, ported from vllm-mlx.
//! - [`PrefixCacheManager`] -- Trie-based prefix matching for KV cache reuse.
//! - [`MemoryAwarePrefixCache`] -- Prefix cache with system memory monitoring.
//!
//! Prompt cache save/load utilities are provided via the [`prompt_cache`] module.

pub mod batch_cache;
pub mod cache_ops;
pub mod error;
pub mod kv_cache;
pub mod memory;
pub mod paged_cache;
pub mod prefix_cache;
pub mod prompt_cache;
pub mod quantized_cache;
pub mod rotating_cache;

// ── Re-exports of core types ──
pub use batch_cache::BatchKVCache;
pub use cache_ops::{make_causal_mask, CacheOps};
pub use error::{CacheError, Result};
pub use kv_cache::KVCache;
pub use memory::MemoryAwarePrefixCache;
pub use paged_cache::{KVCacheBlock, PagedCacheManager};
pub use prefix_cache::PrefixCacheManager;
pub use prompt_cache::{
    can_trim_prompt_cache, load_prompt_cache, make_prompt_cache, save_prompt_cache,
    trim_prompt_cache,
};
pub use quantized_cache::QuantizedKVCache;
pub use rotating_cache::RotatingKVCache;
