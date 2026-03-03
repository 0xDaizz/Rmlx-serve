//! Expert Parallelism (EP) adapter for MoE models.
//!
//! Distributes expert computation across multiple devices/processes.
//! When running with a single rank (`num_ranks == 1`), the adapter operates in
//! pass-through mode with zero overhead.  Multi-rank mode logs warnings until a
//! real distributed transport is wired in.

use std::collections::HashMap;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// EPConfig
// ---------------------------------------------------------------------------

/// Expert parallelism configuration.
#[derive(Debug, Clone)]
pub struct EPConfig {
    /// Number of ranks (devices/processes).
    pub num_ranks: usize,
    /// Current rank index.
    pub rank: usize,
    /// Total number of experts in the model.
    pub total_experts: usize,
    /// Backend for communication (`"gloo"`, `"nccl"`, `"mpi"`).
    pub backend: String,
}

// ---------------------------------------------------------------------------
// ExpertPartition
// ---------------------------------------------------------------------------

/// Maps experts to ranks for load balancing.
pub struct ExpertPartition {
    /// expert_id -> rank assignment
    pub expert_to_rank: HashMap<usize, usize>,
    /// List of expert IDs that are local to this rank.
    pub local_experts: Vec<usize>,
}

impl ExpertPartition {
    /// Create a round-robin partition of experts across ranks.
    ///
    /// Expert `i` is assigned to rank `i % num_ranks`.
    pub fn round_robin(total_experts: usize, num_ranks: usize, rank: usize) -> Self {
        let mut expert_to_rank = HashMap::with_capacity(total_experts);
        let mut local_experts = Vec::new();

        for expert_id in 0..total_experts {
            let assigned_rank = expert_id % num_ranks;
            expert_to_rank.insert(expert_id, assigned_rank);
            if assigned_rank == rank {
                local_experts.push(expert_id);
            }
        }

        debug!(
            rank,
            num_ranks,
            total_experts,
            num_local = local_experts.len(),
            "expert partition created (round-robin)"
        );

        Self {
            expert_to_rank,
            local_experts,
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch / Combine metadata
// ---------------------------------------------------------------------------

/// Metadata produced by [`EPAdapter::dispatch_exchange`] that is needed by
/// [`EPAdapter::combine_exchange`] to reassemble the final output.
#[derive(Debug, Clone)]
pub struct DispatchMeta {
    /// For each input token index, which rank it was dispatched to.
    pub token_destinations: Vec<usize>,
    /// The number of input tokens (needed for reassembly).
    pub num_tokens: usize,
    /// Dimensionality of each token vector.
    pub token_dim: usize,
}

/// Result of the dispatch phase.
#[derive(Debug)]
pub struct DispatchResult {
    /// Tokens that this rank should process locally.
    /// Flat buffer of shape `[num_local_tokens, token_dim]`.
    pub local_tokens: Vec<f32>,
    /// The expert index for each local token.
    pub local_expert_indices: Vec<usize>,
    /// Metadata needed for the combine phase.
    pub meta: DispatchMeta,
}

// ---------------------------------------------------------------------------
// EPAdapter
// ---------------------------------------------------------------------------

/// EP Adapter handles the dispatch-compute-combine cycle.
pub struct EPAdapter {
    config: EPConfig,
    partition: ExpertPartition,
}

impl EPAdapter {
    /// Create a new EP adapter with the given configuration.
    ///
    /// The expert partition is computed via round-robin assignment.
    pub fn new(config: EPConfig) -> Self {
        let partition =
            ExpertPartition::round_robin(config.total_experts, config.num_ranks, config.rank);

        info!(
            rank = config.rank,
            num_ranks = config.num_ranks,
            total_experts = config.total_experts,
            local_experts = ?partition.local_experts,
            backend = %config.backend,
            "EPAdapter initialised"
        );

        if config.num_ranks > 1 {
            warn!(
                "multi-rank EP is not yet backed by a real transport; \
                 dispatch/combine will only work correctly for num_ranks=1"
            );
        }

        Self { config, partition }
    }

    /// Phase 1: Dispatch tokens to the rank that owns each expert.
    ///
    /// `tokens` is a flat buffer of shape `[num_tokens, token_dim]` and
    /// `expert_indices` has length `num_tokens` indicating the chosen expert
    /// for each token.
    ///
    /// Returns a [`DispatchResult`] containing only the tokens assigned to
    /// this rank, plus metadata for reassembly in [`combine_exchange`].
    pub fn dispatch_exchange(&self, tokens: &[f32], expert_indices: &[usize]) -> DispatchResult {
        let num_tokens = expert_indices.len();
        assert!(num_tokens > 0, "dispatch_exchange called with zero tokens");

        let token_dim = tokens.len() / num_tokens;
        assert_eq!(
            tokens.len(),
            num_tokens * token_dim,
            "token buffer length must be divisible by num_tokens"
        );

        // Build destination list.
        let token_destinations: Vec<usize> = expert_indices
            .iter()
            .map(|&eid| {
                *self
                    .partition
                    .expert_to_rank
                    .get(&eid)
                    .unwrap_or_else(|| panic!("expert_id {eid} out of range"))
            })
            .collect();

        // Filter tokens destined for this rank.
        let my_rank = self.config.rank;
        let mut local_tokens = Vec::new();
        let mut local_expert_indices = Vec::new();

        for (idx, &dest) in token_destinations.iter().enumerate() {
            if dest == my_rank {
                let start = idx * token_dim;
                let end = start + token_dim;
                local_tokens.extend_from_slice(&tokens[start..end]);
                local_expert_indices.push(expert_indices[idx]);
            }
        }

        if self.config.num_ranks > 1 {
            // In a real implementation we would send non-local tokens to their
            // destination ranks via all-to-all communication here.
            let remote_count = num_tokens - local_expert_indices.len();
            if remote_count > 0 {
                warn!(
                    rank = my_rank,
                    remote_count,
                    "dropping {remote_count} tokens destined for remote ranks \
                     (distributed transport not yet implemented)"
                );
            }
        }

        debug!(
            rank = my_rank,
            total = num_tokens,
            local = local_expert_indices.len(),
            "dispatch complete"
        );

        DispatchResult {
            local_tokens,
            local_expert_indices,
            meta: DispatchMeta {
                token_destinations,
                num_tokens,
                token_dim,
            },
        }
    }

    // Phase 2: Run local expert computation (called by the model).
    // This adapter does not perform the actual forward pass -- the caller
    // invokes the model's expert layers with the local tokens returned by
    // `dispatch_exchange`.

    /// Phase 3: Combine results from all ranks.
    ///
    /// `local_results` is the flat output buffer from the local expert forward
    /// pass, with shape `[num_local_tokens, token_dim]`.
    ///
    /// Returns a flat buffer of shape `[num_tokens, token_dim]` with all
    /// tokens reassembled in their original order.
    pub fn combine_exchange(
        &self,
        local_results: &[f32],
        dispatch_meta: &DispatchMeta,
    ) -> Vec<f32> {
        let DispatchMeta {
            token_destinations,
            num_tokens,
            token_dim,
        } = dispatch_meta;

        let my_rank = self.config.rank;

        // Pre-allocate the full output buffer (zeros for tokens we didn't process).
        let mut output = vec![0.0f32; num_tokens * token_dim];

        // Place local results back into their original positions.
        let mut local_cursor = 0;
        for (idx, &dest) in token_destinations.iter().enumerate() {
            if dest == my_rank {
                let src_start = local_cursor * token_dim;
                let dst_start = idx * token_dim;
                output[dst_start..dst_start + token_dim]
                    .copy_from_slice(&local_results[src_start..src_start + token_dim]);
                local_cursor += 1;
            }
        }

        if self.config.num_ranks > 1 {
            // In a real implementation we would receive remote results via
            // all-to-all and fill in the remaining slots.
            warn!(
                rank = my_rank,
                "combine_exchange: remote results not available \
                 (distributed transport not yet implemented); \
                 output will contain zeros for non-local experts"
            );
        }

        debug!(rank = my_rank, num_tokens, "combine complete");

        output
    }

    /// Check if this rank owns the given expert.
    pub fn is_local_expert(&self, expert_id: usize) -> bool {
        self.partition
            .expert_to_rank
            .get(&expert_id)
            .is_some_and(|&r| r == self.config.rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_rank_roundtrip() {
        let config = EPConfig {
            num_ranks: 1,
            rank: 0,
            total_experts: 8,
            backend: "local".into(),
        };

        let adapter = EPAdapter::new(config);

        // 4 tokens, dim=3, each routed to a different expert.
        let tokens: Vec<f32> = vec![
            1.0, 2.0, 3.0, // token 0 -> expert 0
            4.0, 5.0, 6.0, // token 1 -> expert 1
            7.0, 8.0, 9.0, // token 2 -> expert 2
            10.0, 11.0, 12.0, // token 3 -> expert 3
        ];
        let expert_indices = vec![0, 1, 2, 3];

        let dispatch = adapter.dispatch_exchange(&tokens, &expert_indices);

        // Single rank: all tokens are local.
        assert_eq!(dispatch.local_tokens.len(), 12);
        assert_eq!(dispatch.local_expert_indices.len(), 4);

        // Simulate expert computation (identity).
        let local_results = dispatch.local_tokens.clone();
        let output = adapter.combine_exchange(&local_results, &dispatch.meta);

        assert_eq!(output, tokens);
    }

    #[test]
    fn partition_round_robin() {
        let partition = ExpertPartition::round_robin(8, 4, 1);
        // Experts 1, 5 should be on rank 1.
        assert_eq!(partition.local_experts, vec![1, 5]);
        assert_eq!(partition.expert_to_rank[&0], 0);
        assert_eq!(partition.expert_to_rank[&1], 1);
        assert_eq!(partition.expert_to_rank[&5], 1);
        assert_eq!(partition.expert_to_rank[&7], 3);
    }

    #[test]
    fn is_local_expert_check() {
        let config = EPConfig {
            num_ranks: 4,
            rank: 2,
            total_experts: 8,
            backend: "nccl".into(),
        };
        let adapter = EPAdapter::new(config);

        assert!(adapter.is_local_expert(2));
        assert!(adapter.is_local_expert(6));
        assert!(!adapter.is_local_expert(0));
        assert!(!adapter.is_local_expert(1));
    }
}
