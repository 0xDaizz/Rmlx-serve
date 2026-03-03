//! Worker loop for non-leader ranks.
//!
//! Receives plans from the leader, executes forward passes, and returns
//! results. Currently a stub that logs and sleeps until the distributed
//! transport is available.

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
    /// In the current stub implementation this simply logs a warning and
    /// sleeps in a loop until the distributed communicator is fully wired.
    pub async fn run(&self) -> Result<(), DistributedError> {
        info!(rank = self.rank, "worker started, waiting for plans");

        loop {
            // In production: receive plan from leader via communicator.
            // For now: stub that logs and yields.
            warn!(
                rank = self.rank,
                "distributed worker loop not yet connected to communicator"
            );
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
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
