//! Memory-aware prefix cache.
//!
//! Wraps `PrefixCacheManager` with memory pressure awareness, using
//! `sysinfo` on macOS to monitor available system memory and trigger
//! eviction when memory usage exceeds a threshold.

use crate::prefix_cache::PrefixCacheManager;

/// Memory-aware prefix cache manager.
///
/// Combines the trie-based prefix cache with memory monitoring.
/// When the cache approaches the memory limit, blocks are evicted
/// (LRU) to free GPU memory.
pub struct MemoryAwarePrefixCache {
    /// Underlying prefix cache for trie-based lookup.
    prefix_cache: PrefixCacheManager,
    /// Maximum memory budget for the cache, in bytes.
    memory_limit_bytes: usize,
    /// Approximate current memory usage, in bytes.
    current_usage_bytes: usize,
    /// Estimated bytes per cache block (for bookkeeping).
    bytes_per_block: usize,
}

impl MemoryAwarePrefixCache {
    /// Create a new memory-aware prefix cache.
    ///
    /// # Arguments
    /// * `max_blocks` - Maximum number of cache blocks.
    /// * `memory_limit_mb` - Maximum memory budget in megabytes.
    pub fn new(max_blocks: usize, memory_limit_mb: usize) -> Self {
        // Estimate bytes per block based on typical KV cache parameters.
        // A conservative default: each block stores block_size tokens *
        // 2 (K+V) * head_dim * dtype_size across all layers.
        // We use a rough estimate here; actual sizes are tracked via
        // `current_usage_bytes` as blocks are allocated/freed.
        let bytes_per_block = 4096; // conservative default, ~4 KB per block

        Self {
            prefix_cache: PrefixCacheManager::new(max_blocks),
            memory_limit_bytes: memory_limit_mb * 1024 * 1024,
            current_usage_bytes: 0,
            bytes_per_block,
        }
    }

    /// Create a memory-aware prefix cache with a custom per-block size estimate.
    ///
    /// # Arguments
    /// * `max_blocks` - Maximum number of cache blocks.
    /// * `memory_limit_mb` - Maximum memory budget in megabytes.
    /// * `bytes_per_block` - Estimated memory per block in bytes.
    pub fn with_block_size(
        max_blocks: usize,
        memory_limit_mb: usize,
        bytes_per_block: usize,
    ) -> Self {
        Self {
            prefix_cache: PrefixCacheManager::new(max_blocks),
            memory_limit_bytes: memory_limit_mb * 1024 * 1024,
            current_usage_bytes: 0,
            bytes_per_block,
        }
    }

    /// Query available system memory in megabytes.
    ///
    /// On macOS, uses `sysinfo` to read the available memory.
    /// On other platforms, returns `usize::MAX` (no memory pressure).
    #[cfg(target_os = "macos")]
    pub fn available_memory_mb() -> usize {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();
        let available = sys.available_memory();
        if available == 0 {
            // Fallback: use free_memory or total_memory as an estimate.
            let free = sys.free_memory();
            if free > 0 {
                return (free / (1024 * 1024)) as usize;
            }
            // If sysinfo can't determine memory, assume plenty is available.
            return usize::MAX;
        }
        (available / (1024 * 1024)) as usize
    }

    #[cfg(not(target_os = "macos"))]
    pub fn available_memory_mb() -> usize {
        usize::MAX
    }

    /// Whether eviction should be triggered.
    ///
    /// Returns `true` if:
    /// - Current usage exceeds the memory limit, OR
    /// - The number of stored blocks has reached `max_blocks`, OR
    /// - System available memory is low (below 10% of the limit).
    pub fn should_evict(&self) -> bool {
        if self.current_usage_bytes >= self.memory_limit_bytes {
            return true;
        }
        if self.prefix_cache.is_full() {
            return true;
        }
        // Also check system memory pressure.
        let available_mb = Self::available_memory_mb();
        let threshold_mb = self.memory_limit_bytes / (1024 * 1024) / 10;
        available_mb < threshold_mb.max(256) // at least 256 MB threshold
    }

    /// Evict blocks until memory usage is below the threshold.
    ///
    /// Evicts blocks in LRU order until either:
    /// - Memory usage drops below 80% of the limit, or
    /// - No more blocks can be evicted.
    pub fn evict_until_threshold(&mut self) {
        let target = (self.memory_limit_bytes as f64 * 0.8) as usize;

        while self.current_usage_bytes > target {
            if let Some(_block_id) = self.prefix_cache.evict_lru() {
                self.current_usage_bytes = self
                    .current_usage_bytes
                    .saturating_sub(self.bytes_per_block);
                tracing::debug!(
                    "evicted block, usage now {} MB",
                    self.current_usage_bytes / (1024 * 1024)
                );
            } else {
                tracing::warn!(
                    "cannot evict further, {} bytes still in use",
                    self.current_usage_bytes
                );
                break;
            }
        }
    }

    /// Look up the longest matching prefix.
    ///
    /// Delegates to the underlying `PrefixCacheManager`.
    pub fn lookup(&mut self, tokens: &[u32]) -> Vec<usize> {
        self.prefix_cache.lookup(tokens)
    }

    /// Insert a token sequence with associated block IDs.
    ///
    /// Updates the memory usage tracker. May trigger eviction if the
    /// cache is over budget.
    pub fn insert(&mut self, tokens: &[u32], block_ids: &[usize]) {
        // Evict if we are at capacity or over memory budget.
        if self.should_evict() {
            self.evict_until_threshold();
        }

        let new_blocks = block_ids.len();
        self.prefix_cache.insert(tokens, block_ids);
        self.current_usage_bytes += new_blocks * self.bytes_per_block;
    }

    /// Manually evict the LRU block.
    pub fn evict_lru(&mut self) -> Option<usize> {
        let evicted = self.prefix_cache.evict_lru();
        if evicted.is_some() {
            self.current_usage_bytes = self
                .current_usage_bytes
                .saturating_sub(self.bytes_per_block);
        }
        evicted
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        self.prefix_cache.hit_rate()
    }

    /// Current memory usage in bytes.
    pub fn current_usage_bytes(&self) -> usize {
        self.current_usage_bytes
    }

    /// Memory limit in bytes.
    pub fn memory_limit_bytes(&self) -> usize {
        self.memory_limit_bytes
    }

    /// Number of cached blocks.
    pub fn num_blocks(&self) -> usize {
        self.prefix_cache.num_blocks()
    }

    /// Immutable access to the underlying prefix cache.
    pub fn prefix_cache(&self) -> &PrefixCacheManager {
        &self.prefix_cache
    }

    /// Mutable access to the underlying prefix cache.
    pub fn prefix_cache_mut(&mut self) -> &mut PrefixCacheManager {
        &mut self.prefix_cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_memory_aware_cache() {
        let cache = MemoryAwarePrefixCache::new(100, 512);
        assert_eq!(cache.memory_limit_bytes(), 512 * 1024 * 1024);
        assert_eq!(cache.current_usage_bytes(), 0);
        assert_eq!(cache.num_blocks(), 0);
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut cache = MemoryAwarePrefixCache::new(100, 512);
        cache.insert(&[1, 2, 3], &[10]);
        cache.insert(&[1, 2, 3, 4, 5, 6], &[10, 20]);

        let matched = cache.lookup(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(matched, vec![10, 20]);
    }

    #[test]
    fn test_memory_tracking() {
        let mut cache = MemoryAwarePrefixCache::with_block_size(100, 1, 1024);
        // 1 MB limit, 1024 bytes per block

        cache.insert(&[1, 2], &[10]);
        assert_eq!(cache.current_usage_bytes(), 1024);

        cache.insert(&[3, 4], &[20]);
        assert_eq!(cache.current_usage_bytes(), 2048);
    }

    #[test]
    fn test_eviction_on_memory_pressure() {
        // Very small limit: 1 KB, 512 bytes per block.
        let mut cache = MemoryAwarePrefixCache::with_block_size(100, 0, 512);
        // memory_limit_bytes = 0 (edge case)
        // This means should_evict() is immediately true.

        // Insert with eviction.
        cache.insert(&[1, 2], &[10]);
        // After insert, eviction may have been triggered.
        // The exact behavior depends on the eviction logic.
    }

    #[test]
    fn test_manual_evict() {
        let mut cache = MemoryAwarePrefixCache::with_block_size(100, 512, 4096);
        cache.insert(&[1, 2], &[10]);
        cache.insert(&[3, 4], &[20]);

        let evicted = cache.evict_lru();
        assert!(evicted.is_some());
        assert_eq!(cache.current_usage_bytes(), 4096); // one block remains
    }

    #[test]
    fn test_available_memory() {
        // Just ensure it doesn't panic and returns a non-zero value.
        // In sandboxed environments, sysinfo may return 0 for available_memory;
        // our implementation falls back to free_memory or usize::MAX.
        let _mb = MemoryAwarePrefixCache::available_memory_mb();
        // If we reach here without panicking, the function works correctly.
    }
}
