//! Communication primitives for distributed inference.
//!
//! Currently provides stub implementations that work for single-process usage.
//! When the `distributed` feature is enabled the real transport from
//! `rmlx-distributed` can be wired in.

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur in distributed operations.
#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    #[error("communication error: {0}")]
    Communication(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("not yet implemented: {0}")]
    Unimplemented(String),
}

// ---------------------------------------------------------------------------
// Communicator
// ---------------------------------------------------------------------------

/// Communication primitives for distributed inference.
pub struct Communicator {
    rank: usize,
    world_size: usize,
}

impl Communicator {
    /// Create a new communicator for the given rank within `world_size` peers.
    pub fn new(rank: usize, world_size: usize) -> Self {
        debug!(rank, world_size, "communicator created");
        Self { rank, world_size }
    }

    /// Return this process's rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Return the total number of peers.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Whether this process is the leader (rank 0).
    pub fn is_leader(&self) -> bool {
        self.rank == 0
    }

    /// Broadcast a step plan from the leader to all workers.
    ///
    /// For `world_size == 1` this is a no-op (the leader already has the plan).
    /// For larger world sizes this currently returns an error because the
    /// network transport is not yet wired in.
    pub async fn broadcast_plan(&self, _plan: &StepPlan) -> Result<(), DistributedError> {
        if self.world_size == 1 {
            debug!("broadcast_plan: single-rank, no-op");
            return Ok(());
        }

        warn!(
            rank = self.rank,
            world_size = self.world_size,
            "broadcast_plan: multi-rank transport not yet implemented"
        );
        Err(DistributedError::Unimplemented(
            "multi-rank broadcast not yet implemented".into(),
        ))
    }

    /// All-reduce sum across ranks (for tensor parallelism).
    ///
    /// For `world_size == 1` the data is already the global result.
    /// For larger world sizes this currently returns an error.
    pub async fn all_sum(&self, data: &mut [f32]) -> Result<(), DistributedError> {
        if self.world_size == 1 {
            // Nothing to reduce -- data is already the global result.
            return Ok(());
        }

        warn!(
            rank = self.rank,
            world_size = self.world_size,
            len = data.len(),
            "all_sum: multi-rank transport not yet implemented"
        );
        Err(DistributedError::Unimplemented(
            "multi-rank all_sum not yet implemented".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// StepPlan / StepType
// ---------------------------------------------------------------------------

/// A plan describing what each worker should do in this step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepPlan {
    /// The kind of work to perform.
    pub step_type: StepType,
    /// Sequence IDs included in this batch.
    pub sequence_ids: Vec<u64>,
    /// Input token IDs for each sequence.
    pub input_tokens: Vec<Vec<u32>>,
}

/// The type of step to execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    /// Full prefill of new sequences.
    Prefill,
    /// Autoregressive decode step.
    Decode,
    /// Graceful shutdown signal.
    Shutdown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn single_rank_broadcast_is_noop() {
        let comm = Communicator::new(0, 1);
        let plan = StepPlan {
            step_type: StepType::Decode,
            sequence_ids: vec![1],
            input_tokens: vec![vec![42]],
        };
        assert!(comm.broadcast_plan(&plan).await.is_ok());
    }

    #[tokio::test]
    async fn single_rank_all_sum_is_noop() {
        let comm = Communicator::new(0, 1);
        let mut data = vec![1.0, 2.0, 3.0];
        assert!(comm.all_sum(&mut data).await.is_ok());
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn multi_rank_broadcast_returns_error() {
        let comm = Communicator::new(0, 4);
        let plan = StepPlan {
            step_type: StepType::Prefill,
            sequence_ids: vec![],
            input_tokens: vec![],
        };
        assert!(comm.broadcast_plan(&plan).await.is_err());
    }
}
