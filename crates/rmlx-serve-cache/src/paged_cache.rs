//! Paged KV cache manager.
//!
//! Ported and simplified from vllm-mlx's paged cache implementation.
//! Manages a pool of fixed-size cache blocks with reference counting,
//! copy-on-write semantics, and content-based hashing for block sharing.

use std::collections::{HashMap, VecDeque};

/// A single block in the paged KV cache.
///
/// Each block holds a fixed number of token positions (`block_size`).
/// Blocks are reference-counted to enable sharing between sequences
/// that have identical prefixes (via content hashing).
pub struct KVCacheBlock {
    /// Unique identifier for this block (index into the block pool).
    pub block_id: usize,
    /// Number of active references to this block.
    pub ref_count: usize,
    /// Content hash for prefix sharing. `None` if the block has been
    /// modified since last hashing (or never hashed).
    pub hash: Option<u64>,
    /// Token IDs stored in this block, up to `block_size`.
    pub tokens: Vec<u32>,
    /// Number of positions that have been filled with data.
    pub num_filled: usize,
}

impl KVCacheBlock {
    /// Create a new empty block.
    fn new(block_id: usize, block_size: usize) -> Self {
        Self {
            block_id,
            ref_count: 0,
            hash: None,
            tokens: Vec::with_capacity(block_size),
            num_filled: 0,
        }
    }

    /// Whether this block is completely filled.
    pub fn is_full(&self, block_size: usize) -> bool {
        self.num_filled >= block_size
    }

    /// Whether this block has no data.
    pub fn is_empty(&self) -> bool {
        self.num_filled == 0
    }

    /// Reset the block to an empty state (for reuse after freeing).
    fn reset(&mut self) {
        self.ref_count = 0;
        self.hash = None;
        self.tokens.clear();
        self.num_filled = 0;
    }
}

/// Paged KV cache block manager.
///
/// Manages a fixed pool of `KVCacheBlock`s with:
/// - Block allocation/deallocation via a free list.
/// - Reference counting for shared blocks.
/// - Copy-on-write: when a shared block needs modification, it is copied.
/// - Content-based hashing for prefix cache deduplication.
///
/// This is the core block management layer. It does not store actual
/// KV tensor data -- that responsibility belongs to the GPU-side cache
/// tensors. This manager tracks which blocks are allocated, shared,
/// and how they map to token sequences.
pub struct PagedCacheManager {
    /// Number of token positions per block.
    block_size: usize,
    /// Total number of blocks in the pool.
    num_blocks: usize,
    /// Queue of free block IDs available for allocation.
    free_blocks: VecDeque<usize>,
    /// All blocks in the pool (indexed by block_id).
    blocks: Vec<KVCacheBlock>,
    /// Map from content hash to block_id for prefix sharing.
    hash_to_block: HashMap<u64, usize>,
}

impl PagedCacheManager {
    /// Create a new paged cache manager.
    ///
    /// # Arguments
    /// * `block_size` - Number of token positions per block.
    /// * `num_blocks` - Total number of blocks in the pool.
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        let blocks: Vec<KVCacheBlock> = (0..num_blocks)
            .map(|id| KVCacheBlock::new(id, block_size))
            .collect();

        let free_blocks: VecDeque<usize> = (0..num_blocks).collect();

        Self {
            block_size,
            num_blocks,
            free_blocks,
            blocks,
            hash_to_block: HashMap::new(),
        }
    }

    /// Allocate a free block and return its block_id.
    ///
    /// The block's reference count is set to 1. Returns `None` if no
    /// free blocks are available.
    pub fn allocate(&mut self) -> Option<usize> {
        let block_id = self.free_blocks.pop_front()?;
        self.blocks[block_id].reset();
        self.blocks[block_id].ref_count = 1;
        Some(block_id)
    }

    /// Free a block, decrementing its reference count.
    ///
    /// If the reference count reaches zero, the block is returned to
    /// the free list and any associated hash mapping is removed.
    ///
    /// # Panics
    /// Panics if `block_id >= num_blocks` or the block is already free
    /// (ref_count == 0).
    pub fn free(&mut self, block_id: usize) {
        assert!(
            block_id < self.num_blocks,
            "block_id {block_id} out of range (num_blocks = {})",
            self.num_blocks
        );
        let block = &mut self.blocks[block_id];
        assert!(
            block.ref_count > 0,
            "double free: block {block_id} already has ref_count 0"
        );

        block.ref_count -= 1;
        if block.ref_count == 0 {
            // Remove hash mapping if present.
            if let Some(hash) = block.hash {
                // Only remove from the map if this block is the current holder.
                if self.hash_to_block.get(&hash) == Some(&block_id) {
                    self.hash_to_block.remove(&hash);
                }
            }
            block.reset();
            self.free_blocks.push_back(block_id);
        }
    }

    /// Get the reference count of a block.
    ///
    /// Returns 0 for free blocks.
    ///
    /// # Panics
    /// Panics if `block_id >= num_blocks`.
    pub fn ref_count(&self, block_id: usize) -> usize {
        assert!(
            block_id < self.num_blocks,
            "block_id {block_id} out of range"
        );
        self.blocks[block_id].ref_count
    }

    /// Increment the reference count of a block.
    ///
    /// Used when a new sequence wants to share an existing block
    /// (e.g., for prefix caching).
    ///
    /// # Panics
    /// Panics if `block_id >= num_blocks` or the block is free.
    pub fn increment_ref(&mut self, block_id: usize) {
        assert!(
            block_id < self.num_blocks,
            "block_id {block_id} out of range"
        );
        assert!(
            self.blocks[block_id].ref_count > 0,
            "cannot increment ref on free block {block_id}"
        );
        self.blocks[block_id].ref_count += 1;
    }

    /// Perform copy-on-write for a shared block.
    ///
    /// If the block has `ref_count > 1`, allocates a new block, copies
    /// the metadata (tokens, num_filled), decrements the original block's
    /// ref_count, and returns the new block_id.
    ///
    /// If the block has `ref_count == 1`, it is not shared and no copy
    /// is needed -- returns `None` (the caller can write in-place).
    ///
    /// Returns `Some(new_block_id)` if a copy was made, or `None` if
    /// the block is not shared and can be modified in-place.
    ///
    /// Returns `None` also if no free blocks are available for the copy.
    pub fn copy_on_write(&mut self, block_id: usize) -> Option<usize> {
        assert!(
            block_id < self.num_blocks,
            "block_id {block_id} out of range"
        );

        if self.blocks[block_id].ref_count <= 1 {
            // Not shared, no copy needed.
            return None;
        }

        // Allocate a new block for the copy.
        let new_id = self.allocate()?;

        // Copy metadata from the original block.
        let tokens = self.blocks[block_id].tokens.clone();
        let num_filled = self.blocks[block_id].num_filled;

        self.blocks[new_id].tokens = tokens;
        self.blocks[new_id].num_filled = num_filled;
        // New block has no hash (it is about to be modified).
        self.blocks[new_id].hash = None;

        // Decrement the original block's reference count.
        self.blocks[block_id].ref_count -= 1;

        Some(new_id)
    }

    /// Number of free (unallocated) blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Number of allocated (in-use) blocks.
    pub fn num_allocated_blocks(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }

    /// Total number of blocks in the pool.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Block size (number of token positions per block).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get an immutable reference to a block.
    ///
    /// Returns `None` if `block_id >= num_blocks`.
    pub fn get_block(&self, block_id: usize) -> Option<&KVCacheBlock> {
        self.blocks.get(block_id)
    }

    /// Get a mutable reference to a block.
    ///
    /// Returns `None` if `block_id >= num_blocks`.
    pub fn get_block_mut(&mut self, block_id: usize) -> Option<&mut KVCacheBlock> {
        self.blocks.get_mut(block_id)
    }

    /// Register a content hash for a block.
    ///
    /// This enables future lookups via `lookup_hash`. If a different block
    /// was previously registered with the same hash, the old mapping is
    /// replaced.
    pub fn register_hash(&mut self, block_id: usize, hash: u64) {
        assert!(
            block_id < self.num_blocks,
            "block_id {block_id} out of range"
        );
        self.blocks[block_id].hash = Some(hash);
        self.hash_to_block.insert(hash, block_id);
    }

    /// Look up a block by content hash.
    ///
    /// Returns the block_id if a block with the given hash exists and
    /// is still allocated (ref_count > 0).
    pub fn lookup_hash(&self, hash: u64) -> Option<usize> {
        self.hash_to_block
            .get(&hash)
            .copied()
            .filter(|&id| self.blocks[id].ref_count > 0)
    }

    /// Append tokens to a block.
    ///
    /// # Panics
    /// Panics if the block would overflow `block_size`.
    pub fn append_tokens(&mut self, block_id: usize, tokens: &[u32]) {
        let block = &mut self.blocks[block_id];
        assert!(
            block.num_filled + tokens.len() <= self.block_size,
            "block {block_id} overflow: {} filled + {} new > {} block_size",
            block.num_filled,
            tokens.len(),
            self.block_size
        );
        block.tokens.extend_from_slice(tokens);
        block.num_filled += tokens.len();
        // Invalidate hash since content changed.
        block.hash = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_manager() {
        let mgr = PagedCacheManager::new(16, 100);
        assert_eq!(mgr.block_size(), 16);
        assert_eq!(mgr.num_blocks(), 100);
        assert_eq!(mgr.num_free_blocks(), 100);
        assert_eq!(mgr.num_allocated_blocks(), 0);
    }

    #[test]
    fn test_allocate_and_free() {
        let mut mgr = PagedCacheManager::new(16, 4);

        let b0 = mgr.allocate().unwrap();
        assert_eq!(mgr.ref_count(b0), 1);
        assert_eq!(mgr.num_free_blocks(), 3);

        let b1 = mgr.allocate().unwrap();
        let b2 = mgr.allocate().unwrap();
        let b3 = mgr.allocate().unwrap();
        assert_eq!(mgr.num_free_blocks(), 0);

        // All blocks exhausted.
        assert!(mgr.allocate().is_none());

        // Free one block.
        mgr.free(b1);
        assert_eq!(mgr.num_free_blocks(), 1);
        assert_eq!(mgr.ref_count(b1), 0);

        // Re-allocate.
        let b4 = mgr.allocate().unwrap();
        assert_eq!(b4, b1); // reused the freed block

        // Free remaining.
        mgr.free(b0);
        mgr.free(b2);
        mgr.free(b3);
        mgr.free(b4);
        assert_eq!(mgr.num_free_blocks(), 4);
    }

    #[test]
    fn test_ref_counting() {
        let mut mgr = PagedCacheManager::new(16, 4);
        let b0 = mgr.allocate().unwrap();

        assert_eq!(mgr.ref_count(b0), 1);
        mgr.increment_ref(b0);
        assert_eq!(mgr.ref_count(b0), 2);
        mgr.increment_ref(b0);
        assert_eq!(mgr.ref_count(b0), 3);

        // Free decrements.
        mgr.free(b0);
        assert_eq!(mgr.ref_count(b0), 2);
        assert_eq!(mgr.num_free_blocks(), 3); // not returned yet

        mgr.free(b0);
        assert_eq!(mgr.ref_count(b0), 1);

        mgr.free(b0);
        assert_eq!(mgr.ref_count(b0), 0);
        assert_eq!(mgr.num_free_blocks(), 4); // now returned
    }

    #[test]
    fn test_copy_on_write_shared() {
        let mut mgr = PagedCacheManager::new(16, 4);
        let b0 = mgr.allocate().unwrap();

        // Add some tokens.
        mgr.append_tokens(b0, &[1, 2, 3]);

        // Share the block.
        mgr.increment_ref(b0);
        assert_eq!(mgr.ref_count(b0), 2);

        // CoW should create a new block.
        let new_id = mgr.copy_on_write(b0).unwrap();
        assert_ne!(new_id, b0);
        assert_eq!(mgr.ref_count(b0), 1); // decremented
        assert_eq!(mgr.ref_count(new_id), 1);

        // New block should have the same tokens.
        let new_block = mgr.get_block(new_id).unwrap();
        assert_eq!(new_block.tokens, vec![1, 2, 3]);
        assert_eq!(new_block.num_filled, 3);
    }

    #[test]
    fn test_copy_on_write_unshared() {
        let mut mgr = PagedCacheManager::new(16, 4);
        let b0 = mgr.allocate().unwrap();

        // Not shared, CoW returns None.
        assert!(mgr.copy_on_write(b0).is_none());
    }

    #[test]
    fn test_hash_lookup() {
        let mut mgr = PagedCacheManager::new(16, 4);
        let b0 = mgr.allocate().unwrap();

        mgr.register_hash(b0, 12345);
        assert_eq!(mgr.lookup_hash(12345), Some(b0));
        assert_eq!(mgr.lookup_hash(99999), None);

        // Free the block -- lookup should return None.
        mgr.free(b0);
        assert_eq!(mgr.lookup_hash(12345), None);
    }

    #[test]
    fn test_append_tokens() {
        let mut mgr = PagedCacheManager::new(4, 2);
        let b0 = mgr.allocate().unwrap();

        mgr.append_tokens(b0, &[10, 20]);
        let block = mgr.get_block(b0).unwrap();
        assert_eq!(block.tokens, vec![10, 20]);
        assert_eq!(block.num_filled, 2);
        assert!(!block.is_full(4));

        mgr.append_tokens(b0, &[30, 40]);
        let block = mgr.get_block(b0).unwrap();
        assert_eq!(block.tokens, vec![10, 20, 30, 40]);
        assert_eq!(block.num_filled, 4);
        assert!(block.is_full(4));
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn test_append_tokens_overflow() {
        let mut mgr = PagedCacheManager::new(2, 1);
        let b0 = mgr.allocate().unwrap();
        mgr.append_tokens(b0, &[1, 2, 3]); // 3 > block_size of 2
    }

    #[test]
    #[should_panic(expected = "double free")]
    fn test_double_free_panics() {
        let mut mgr = PagedCacheManager::new(16, 2);
        let b0 = mgr.allocate().unwrap();
        mgr.free(b0);
        mgr.free(b0); // should panic
    }

    #[test]
    fn test_get_block() {
        let mgr = PagedCacheManager::new(16, 2);
        assert!(mgr.get_block(0).is_some());
        assert!(mgr.get_block(1).is_some());
        assert!(mgr.get_block(2).is_none());
    }
}
