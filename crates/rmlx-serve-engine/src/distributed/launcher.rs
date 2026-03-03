//! Launches distributed inference across multiple processes/machines.
//!
//! Provides [`DistributedLauncher`] which can spawn worker processes either
//! locally or via SSH for multi-machine setups.

use std::process::Stdio;
use tokio::process::Child;
use tracing::info;

use super::communicator::DistributedError;

// ---------------------------------------------------------------------------
// DistributedLauncher
// ---------------------------------------------------------------------------

/// Launches distributed inference across multiple processes/machines.
pub struct DistributedLauncher {
    /// Number of ranks (processes) to launch.
    pub num_ranks: usize,
    /// Optional hostfile for multi-machine launch (one host per line).
    pub hostfile: Option<String>,
    /// Communication backend (`"gloo"`, `"nccl"`, `"mpi"`).
    pub backend: String,
}

impl DistributedLauncher {
    /// Launch worker processes.
    ///
    /// For single-machine setups (`hostfile` is `None`), workers are spawned as
    /// child processes on the local machine. For multi-machine setups the
    /// hostfile is read and workers are launched via SSH.
    ///
    /// Returns a `Vec<Child>` with handles to all spawned worker processes
    /// (rank 0 is assumed to run in the current process and is **not** spawned).
    pub async fn launch(
        &self,
        model_path: &str,
    ) -> Result<Vec<Child>, DistributedError> {
        if self.num_ranks <= 1 {
            info!("single-rank launch; no workers to spawn");
            return Ok(Vec::new());
        }

        info!(
            num_ranks = self.num_ranks,
            backend = %self.backend,
            hostfile = ?self.hostfile,
            "launching distributed workers"
        );

        let mut children = Vec::with_capacity(self.num_ranks - 1);

        // Determine the executable to run for each worker.
        let exe = std::env::current_exe().map_err(|e| {
            DistributedError::Communication(format!("failed to determine current exe: {e}"))
        })?;

        for rank in 1..self.num_ranks {
            let child = if let Some(ref hostfile) = self.hostfile {
                // Multi-machine: launch via SSH.
                let hosts = self.read_hostfile(hostfile)?;
                let host = &hosts[(rank - 1) % hosts.len()];

                info!(rank, host = %host, "launching remote worker via SSH");

                tokio::process::Command::new("ssh")
                    .arg(host)
                    .arg(exe.to_string_lossy().as_ref())
                    .arg("--distributed-worker")
                    .arg("--rank")
                    .arg(rank.to_string())
                    .arg("--world-size")
                    .arg(self.num_ranks.to_string())
                    .arg("--backend")
                    .arg(&self.backend)
                    .arg("--model")
                    .arg(model_path)
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|e| {
                        DistributedError::Communication(format!(
                            "failed to launch worker rank {rank} on {host}: {e}"
                        ))
                    })?
            } else {
                // Single-machine: spawn local process.
                info!(rank, "launching local worker process");

                tokio::process::Command::new(&exe)
                    .arg("--distributed-worker")
                    .arg("--rank")
                    .arg(rank.to_string())
                    .arg("--world-size")
                    .arg(self.num_ranks.to_string())
                    .arg("--backend")
                    .arg(&self.backend)
                    .arg("--model")
                    .arg(model_path)
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|e| {
                        DistributedError::Communication(format!(
                            "failed to launch local worker rank {rank}: {e}"
                        ))
                    })?
            };

            children.push(child);
        }

        info!(
            spawned = children.len(),
            "all worker processes launched"
        );

        Ok(children)
    }

    /// Parse a hostfile (one hostname per line, ignoring blank lines and `#` comments).
    fn read_hostfile(&self, path: &str) -> Result<Vec<String>, DistributedError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            DistributedError::Communication(format!("failed to read hostfile {path}: {e}"))
        })?;

        let hosts: Vec<String> = content
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.to_string())
            .collect();

        if hosts.is_empty() {
            return Err(DistributedError::Communication(format!(
                "hostfile {path} contains no hosts"
            )));
        }

        Ok(hosts)
    }
}
