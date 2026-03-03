//! Batch KV cache management.
//!
//! Manages KV caches for multiple sequences in a batch, supporting
//! insertion, removal, and compaction of per-sequence caches.

use crate::kv_cache::KVCache;

/// Manages KV caches for multiple sequences in a batch.
///
/// Each sequence in a batch gets its own `KVCache` instance, indexed
/// by sequence position. Supports sparse insertion (not all slots need
/// to be filled), removal, and compaction to eliminate gaps.
pub struct BatchKVCache {
    /// Sparse map of sequence index to cache. Uses `Option` to allow gaps.
    caches: Vec<Option<KVCache>>,
    /// Maximum number of sequences this batch can hold.
    capacity: usize,
}

impl BatchKVCache {
    /// Create a new batch cache with the given capacity.
    ///
    /// All slots are initially empty.
    pub fn new(capacity: usize) -> Self {
        let caches = (0..capacity).map(|_| None).collect();
        Self { caches, capacity }
    }

    /// Insert a cache at the given sequence index.
    ///
    /// Overwrites any existing cache at that index.
    ///
    /// # Panics
    /// Panics if `idx >= capacity`.
    pub fn insert(&mut self, idx: usize, cache: KVCache) {
        assert!(
            idx < self.capacity,
            "batch cache index {idx} exceeds capacity {}",
            self.capacity
        );
        self.caches[idx] = Some(cache);
    }

    /// Get a mutable reference to the cache at the given index.
    ///
    /// Returns `None` if the slot is empty or `idx >= capacity`.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut KVCache> {
        self.caches.get_mut(idx).and_then(|slot| slot.as_mut())
    }

    /// Get an immutable reference to the cache at the given index.
    ///
    /// Returns `None` if the slot is empty or `idx >= capacity`.
    pub fn get(&self, idx: usize) -> Option<&KVCache> {
        self.caches.get(idx).and_then(|slot| slot.as_ref())
    }

    /// Remove and return the cache at the given index.
    ///
    /// Returns `None` if the slot was already empty or `idx >= capacity`.
    pub fn remove(&mut self, idx: usize) -> Option<KVCache> {
        if idx >= self.capacity {
            return None;
        }
        self.caches[idx].take()
    }

    /// Number of active (non-empty) sequence caches.
    pub fn active_count(&self) -> usize {
        self.caches.iter().filter(|s| s.is_some()).count()
    }

    /// Maximum capacity of this batch cache.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Compact the cache by removing gaps.
    ///
    /// Moves all non-empty caches to the front of the array, maintaining
    /// their relative order. After compaction, `active_count()` caches
    /// occupy indices `0..active_count()`.
    pub fn compact(&mut self) {
        // Collect all active caches while preserving order.
        let active: Vec<KVCache> = self.caches.iter_mut().filter_map(|s| s.take()).collect();
        let active_len = active.len();

        // Place them back at the front.
        for (i, cache) in active.into_iter().enumerate() {
            self.caches[i] = Some(cache);
        }

        // Ensure remaining slots are empty (they already are after `take`).
        for slot in &mut self.caches[active_len..] {
            *slot = None;
        }
    }

    /// Iterate over all active (index, cache) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &KVCache)> {
        self.caches
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|c| (i, c)))
    }

    /// Iterate mutably over all active (index, cache) pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut KVCache)> {
        self.caches
            .iter_mut()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_mut().map(|c| (i, c)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_batch_cache() {
        let batch = BatchKVCache::new(8);
        assert_eq!(batch.capacity(), 8);
        assert_eq!(batch.active_count(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut batch = BatchKVCache::new(4);
        let cache = KVCache::new(2, 4);
        batch.insert(1, cache);

        assert_eq!(batch.active_count(), 1);
        assert!(batch.get(0).is_none());
        assert!(batch.get(1).is_some());
        assert!(batch.get_mut(1).is_some());
    }

    #[test]
    fn test_remove() {
        let mut batch = BatchKVCache::new(4);
        batch.insert(0, KVCache::new(2, 4));
        batch.insert(2, KVCache::new(2, 4));

        assert_eq!(batch.active_count(), 2);

        let removed = batch.remove(0);
        assert!(removed.is_some());
        assert_eq!(batch.active_count(), 1);

        let removed = batch.remove(0);
        assert!(removed.is_none());
    }

    #[test]
    fn test_compact() {
        let mut batch = BatchKVCache::new(4);
        batch.insert(0, KVCache::new(2, 4));
        batch.insert(2, KVCache::new(2, 8));
        batch.insert(3, KVCache::new(2, 16));

        // Remove slot 0, leaving gaps at 0 and 1.
        batch.remove(0);
        assert_eq!(batch.active_count(), 2);

        batch.compact();
        assert_eq!(batch.active_count(), 2);

        // After compaction, caches should be at indices 0 and 1.
        assert!(batch.get(0).is_some());
        assert!(batch.get(1).is_some());
        assert!(batch.get(2).is_none());
        assert!(batch.get(3).is_none());
    }

    #[test]
    fn test_iter() {
        let mut batch = BatchKVCache::new(4);
        batch.insert(1, KVCache::new(2, 4));
        batch.insert(3, KVCache::new(2, 8));

        let indices: Vec<usize> = batch.iter().map(|(i, _)| i).collect();
        assert_eq!(indices, vec![1, 3]);
    }

    #[test]
    #[should_panic(expected = "batch cache index 4 exceeds capacity 4")]
    fn test_insert_out_of_bounds() {
        let mut batch = BatchKVCache::new(4);
        batch.insert(4, KVCache::new(1, 1));
    }
}
