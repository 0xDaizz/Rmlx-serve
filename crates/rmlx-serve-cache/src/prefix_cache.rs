//! Trie-based prefix cache for KV cache block reuse.
//!
//! Enables efficient prefix matching: given a token sequence, find the
//! longest prefix that has already been cached. This avoids recomputing
//! KV values for shared prefixes (e.g., system prompts, few-shot examples).

use std::collections::HashMap;

/// A node in the prefix trie.
///
/// Each node represents a token position. Children map token IDs to
/// the index of the next node in the `nodes` arena. Leaf or internal
/// nodes may have an associated cache block ID.
struct TrieNode {
    /// Map from token_id to the index of the child node in the arena.
    children: HashMap<u32, usize>,
    /// Cache block ID associated with this node (if this position marks
    /// the end of a cached block's token range).
    block_id: Option<usize>,
    /// Monotonically increasing counter value at last access (for LRU eviction).
    last_access: u64,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            block_id: None,
            last_access: 0,
        }
    }
}

/// Trie-based prefix cache manager.
///
/// Stores a trie of token sequences, where each path from root to a node
/// corresponds to a token prefix. Nodes at block boundaries store the
/// block_id of the cached KV block covering that range of tokens.
///
/// Supports:
/// - **Lookup**: Find the longest cached prefix for a token sequence.
/// - **Insert**: Record a new token sequence and its associated block IDs.
/// - **LRU eviction**: Evict the least recently used leaf block.
pub struct PrefixCacheManager {
    /// Arena of trie nodes. Index 0 is always the root.
    nodes: Vec<TrieNode>,
    /// Monotonically increasing counter for tracking access recency.
    access_counter: u64,
    /// Maximum number of blocks that can be stored in the cache.
    max_blocks: usize,
    /// Number of blocks currently stored.
    num_blocks: usize,
    /// Total lookups performed (for hit rate calculation).
    total_lookups: u64,
    /// Total tokens that were cache hits.
    hit_tokens: u64,
    /// Total tokens looked up.
    lookup_tokens: u64,
}

impl PrefixCacheManager {
    /// Create a new prefix cache manager.
    ///
    /// # Arguments
    /// * `max_blocks` - Maximum number of cache blocks to store.
    pub fn new(max_blocks: usize) -> Self {
        let root = TrieNode::new();
        Self {
            nodes: vec![root],
            access_counter: 0,
            max_blocks,
            num_blocks: 0,
            total_lookups: 0,
            hit_tokens: 0,
            lookup_tokens: 0,
        }
    }

    /// Look up the longest matching prefix for the given token sequence.
    ///
    /// Returns a list of block IDs for the matched prefix, in order.
    /// Updates the access counter on all matched nodes (for LRU tracking).
    ///
    /// If no prefix matches, returns an empty vector.
    pub fn lookup(&mut self, tokens: &[u32]) -> Vec<usize> {
        self.total_lookups += 1;
        self.lookup_tokens += tokens.len() as u64;

        let mut block_ids = Vec::new();
        let mut current = 0; // root node index

        self.access_counter += 1;
        let access_time = self.access_counter;

        for &token in tokens {
            if let Some(&child_idx) = self.nodes[current].children.get(&token) {
                current = child_idx;
                // Update access time.
                self.nodes[current].last_access = access_time;

                // If this node has a block_id, record it.
                if let Some(bid) = self.nodes[current].block_id {
                    block_ids.push(bid);
                }
            } else {
                // No match beyond this point.
                break;
            }
        }

        self.hit_tokens += block_ids.len() as u64;
        block_ids
    }

    /// Insert a token sequence and its associated block IDs into the trie.
    ///
    /// The `tokens` slice and `block_ids` slice should correspond: each
    /// block_id is associated with a range of tokens. Block boundaries
    /// are placed at even intervals (`tokens.len() / block_ids.len()`
    /// tokens per block), or the caller can provide token-per-block
    /// boundaries explicitly by ensuring `tokens.len()` is divisible
    /// by `block_ids.len()`.
    ///
    /// If `tokens.len()` is not evenly divisible by `block_ids.len()`,
    /// the last block covers the remaining tokens.
    pub fn insert(&mut self, tokens: &[u32], block_ids: &[usize]) {
        if tokens.is_empty() || block_ids.is_empty() {
            return;
        }

        let tokens_per_block = if block_ids.len() == 1 {
            tokens.len()
        } else {
            tokens.len() / block_ids.len()
        };

        self.access_counter += 1;
        let access_time = self.access_counter;

        let mut current = 0; // root
        let mut block_idx = 0;

        for (i, &token) in tokens.iter().enumerate() {
            let next = if let Some(&child_idx) = self.nodes[current].children.get(&token) {
                child_idx
            } else {
                // Create a new node.
                let new_idx = self.nodes.len();
                self.nodes.push(TrieNode::new());
                self.nodes[current].children.insert(token, new_idx);
                new_idx
            };

            self.nodes[next].last_access = access_time;

            // Check if this token position is a block boundary.
            let is_boundary = if block_idx < block_ids.len() {
                if block_idx == block_ids.len() - 1 {
                    // Last block: covers remaining tokens.
                    i == tokens.len() - 1
                } else {
                    (i + 1) == (block_idx + 1) * tokens_per_block
                }
            } else {
                false
            };

            if is_boundary && block_idx < block_ids.len() {
                let had_block = self.nodes[next].block_id.is_some();
                self.nodes[next].block_id = Some(block_ids[block_idx]);
                if !had_block {
                    self.num_blocks += 1;
                }
                block_idx += 1;
            }

            current = next;
        }
    }

    /// Evict the least recently used leaf block.
    ///
    /// Finds the trie node with the smallest `last_access` value that
    /// has a `block_id`, removes that block_id, and returns it.
    ///
    /// Returns `None` if no blocks are stored.
    pub fn evict_lru(&mut self) -> Option<usize> {
        if self.num_blocks == 0 {
            return None;
        }

        // Find the node with the smallest last_access that has a block_id.
        // Prefer leaf nodes (nodes with no children that have block_ids)
        // to avoid fragmenting the trie.
        let mut best_idx = None;
        let mut best_access = u64::MAX;
        let mut best_is_leaf = false;

        for (idx, node) in self.nodes.iter().enumerate() {
            if node.block_id.is_some() {
                let is_leaf = node.children.is_empty();
                // Prefer leaves over internal nodes; among the same type,
                // prefer the one with the oldest access time.
                let better = match (is_leaf, best_is_leaf) {
                    (true, false) => true,
                    (false, true) => false,
                    _ => node.last_access < best_access,
                };
                if better {
                    best_idx = Some(idx);
                    best_access = node.last_access;
                    best_is_leaf = is_leaf;
                }
            }
        }

        if let Some(idx) = best_idx {
            let block_id = self.nodes[idx].block_id.take();
            self.num_blocks -= 1;

            // If this was a leaf node, try to clean up the trie by removing
            // it from its parent. We do a simple cleanup: find the parent
            // and remove the child entry if the node is now empty.
            if best_is_leaf {
                self.prune_leaf(idx);
            }

            block_id
        } else {
            None
        }
    }

    /// Remove a leaf node from its parent if it has no block_id and no children.
    fn prune_leaf(&mut self, leaf_idx: usize) {
        if leaf_idx == 0 {
            return; // never remove root
        }

        let node = &self.nodes[leaf_idx];
        if !node.children.is_empty() || node.block_id.is_some() {
            return; // not pruneable
        }

        // Find the parent by scanning (O(n) but trie is typically small).
        for parent_idx in 0..self.nodes.len() {
            let parent = &mut self.nodes[parent_idx];
            let mut to_remove = None;
            for (&token_id, &child_idx) in &parent.children {
                if child_idx == leaf_idx {
                    to_remove = Some(token_id);
                    break;
                }
            }
            if let Some(token_id) = to_remove {
                parent.children.remove(&token_id);
                break;
            }
        }
    }

    /// Cache hit rate (0.0 to 1.0).
    ///
    /// Computed as the ratio of hit tokens to total lookup tokens.
    /// Returns 0.0 if no lookups have been performed.
    pub fn hit_rate(&self) -> f64 {
        if self.lookup_tokens == 0 {
            return 0.0;
        }
        self.hit_tokens as f64 / self.lookup_tokens as f64
    }

    /// Number of blocks currently stored in the cache.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Maximum number of blocks allowed.
    pub fn max_blocks(&self) -> usize {
        self.max_blocks
    }

    /// Whether the cache is at capacity.
    pub fn is_full(&self) -> bool {
        self.num_blocks >= self.max_blocks
    }

    /// Number of nodes in the trie (including root).
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_prefix_cache() {
        let cache = PrefixCacheManager::new(100);
        assert_eq!(cache.num_blocks(), 0);
        assert_eq!(cache.max_blocks(), 100);
        assert!(!cache.is_full());
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut cache = PrefixCacheManager::new(100);

        // Insert a sequence of 4 tokens with 2 blocks.
        let tokens = vec![10, 20, 30, 40];
        let block_ids = vec![0, 1];
        cache.insert(&tokens, &block_ids);

        // Full prefix match.
        let matched = cache.lookup(&tokens);
        assert_eq!(matched, vec![0, 1]);

        // Partial prefix match (first 2 tokens = first block).
        let matched = cache.lookup(&[10, 20]);
        assert_eq!(matched, vec![0]);

        // Partial prefix match (first 3 tokens = still just first block boundary).
        let matched = cache.lookup(&[10, 20, 30]);
        assert_eq!(matched, vec![0]);

        // No match.
        let matched = cache.lookup(&[99, 88]);
        assert_eq!(matched, Vec::<usize>::new());
    }

    #[test]
    fn test_single_block_insert() {
        let mut cache = PrefixCacheManager::new(100);

        let tokens = vec![1, 2, 3];
        let block_ids = vec![42];
        cache.insert(&tokens, &block_ids);

        // Block is at the end of the token sequence.
        let matched = cache.lookup(&[1, 2, 3]);
        assert_eq!(matched, vec![42]);

        // Partial match -- no block boundary hit.
        let matched = cache.lookup(&[1, 2]);
        assert_eq!(matched, Vec::<usize>::new());
    }

    #[test]
    fn test_overlapping_prefixes() {
        let mut cache = PrefixCacheManager::new(100);

        // Insert "hello" prefix.
        cache.insert(&[1, 2, 3], &[10]);
        // Insert "hello world" prefix (shares first 3 tokens).
        cache.insert(&[1, 2, 3, 4, 5, 6], &[10, 20]);

        // Lookup the longer sequence.
        let matched = cache.lookup(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(matched, vec![10, 20]);

        // Lookup just the shorter prefix.
        let matched = cache.lookup(&[1, 2, 3]);
        assert_eq!(matched, vec![10]);
    }

    #[test]
    fn test_evict_lru() {
        let mut cache = PrefixCacheManager::new(2);

        cache.insert(&[1, 2], &[10]);
        cache.insert(&[3, 4], &[20]);
        assert_eq!(cache.num_blocks(), 2);

        // Access the second one to make the first the LRU.
        let _ = cache.lookup(&[3, 4]);

        // Evict should remove the LRU block (block 10, from sequence [1,2]).
        let evicted = cache.evict_lru();
        assert_eq!(evicted, Some(10));
        assert_eq!(cache.num_blocks(), 1);

        // The second block should still be accessible.
        let matched = cache.lookup(&[3, 4]);
        assert_eq!(matched, vec![20]);
    }

    #[test]
    fn test_evict_empty() {
        let mut cache = PrefixCacheManager::new(10);
        assert_eq!(cache.evict_lru(), None);
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = PrefixCacheManager::new(100);
        cache.insert(&[1, 2, 3], &[10]);

        // First lookup: 3 tokens, all miss (block at end).
        // Actually, the block_id is at token 3, so we get 1 hit out of 3 tokens.
        let _ = cache.lookup(&[1, 2, 3]);
        // hit_tokens = 1 (one block matched), lookup_tokens = 3
        assert!(cache.hit_rate() > 0.0);

        // Lookup with no match.
        let _ = cache.lookup(&[99, 88, 77]);
        // Now: hit_tokens = 1, lookup_tokens = 6
    }

    #[test]
    fn test_insert_empty() {
        let mut cache = PrefixCacheManager::new(100);
        cache.insert(&[], &[]);
        assert_eq!(cache.num_blocks(), 0);

        cache.insert(&[1, 2], &[]);
        assert_eq!(cache.num_blocks(), 0);
    }

    #[test]
    fn test_branching_trie() {
        let mut cache = PrefixCacheManager::new(100);

        // Two sequences that share a prefix then diverge.
        cache.insert(&[1, 2, 3, 4], &[10, 11]);
        cache.insert(&[1, 2, 5, 6], &[10, 12]);

        // Common prefix matches.
        let matched = cache.lookup(&[1, 2, 3, 4]);
        assert_eq!(matched, vec![10, 11]);

        let matched = cache.lookup(&[1, 2, 5, 6]);
        assert_eq!(matched, vec![10, 12]);

        // Shared prefix only.
        let matched = cache.lookup(&[1, 2]);
        assert_eq!(matched, vec![10]);
    }
}
