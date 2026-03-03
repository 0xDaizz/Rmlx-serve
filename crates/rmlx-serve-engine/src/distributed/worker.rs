//! Worker loop for non-leader ranks.
//!
//! Receives plans from the leader, executes forward passes, and returns
//! results. The transport wiring is still pending, so this worker currently
//! fails fast in multi-rank mode instead of looping forever.

use tracing::{info, warn};

use super::communicator::{Communicator, DistributedError};

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Worker loop for non-leader ranks.
///
/// Receives plans from the leader, executes forward passes, returns results.
pub struct Worker {
    rank: usize,
    communicator: Communicator,
}

impl Worker {
    /// Create a new worker for the given rank.
    pub fn new(rank: usize, communicator: Communicator) -> Self {
        Self { rank, communicator }
    }

    /// Main worker loop: receive plan -> execute -> send results.
    ///
    /// In the current stub implementation this fails fast for multi-rank mode
    /// so launcher callers get an explicit error instead of a hung process.
    pub async fn run(&self) -> Result<(), DistributedError> {
        info!(rank = self.rank, "worker started, waiting for plans");

        if self.communicator.world_size() <= 1 {
            warn!(
                rank = self.rank,
                "worker mode started with world_size<=1; nothing to do"
            );
            return Ok(());
        }

        Err(DistributedError::Unimplemented(
            "worker transport loop is not wired yet for multi-rank execution".to_string(),
        ))
    }

    /// Return this worker's rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Return a reference to the underlying communicator.
    pub fn communicator(&self) -> &Communicator {
        &self.communicator
    }
}
