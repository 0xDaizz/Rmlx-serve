//! Distributed inference support.
//!
//! Provides the communication primitives, worker loop, and launcher for
//! running model inference across multiple processes or machines.

pub mod communicator;
pub mod launcher;
pub mod worker;

pub use communicator::{Communicator, DistributedError, StepPlan, StepType};
pub use launcher::DistributedLauncher;
pub use worker::Worker;
