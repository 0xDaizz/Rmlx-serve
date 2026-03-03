//! rmlx-serve-scheduler: Request scheduling and batch generation for the
//! rmlx-serve inference engine.
//!
//! This crate integrates concepts from two sources:
//!
//! - **mlx-lm's `BatchGenerator`**: Low-level batched prefill and decode
//!   operations. Manages a prefill queue and an active decode batch,
//!   running model forward passes and applying sampling at each step.
//!
//! - **vllm-mlx's `Scheduler`**: High-level request admission control,
//!   queue management, and request lifecycle tracking. Maps between
//!   user-facing request IDs and internal sequence IDs.
//!
//! # Architecture
//!
//! ```text
//!   Requests ──> Scheduler ──> BatchGenerator ──> Model
//!                   │               │
//!                   │            prefill_queue ──> decode_batch
//!                   │
//!                waiting ──> running (seq_id mapping)
//! ```
//!
//! The `Scheduler` sits between the engine/API layer and the `BatchGenerator`.
//! It validates requests, creates samplers and logits processors from
//! `SamplingParams`, manages KV cache allocation, and maps batch generator
//! responses back to request IDs.

pub mod batch;
pub mod batch_generator;
pub mod error;
pub mod policy;
pub mod scheduler;

// Re-export the main public API at crate root.
pub use batch::{Batch, SequenceId, SequenceState};
pub use batch_generator::{BatchGenerator, BatchGeneratorConfig, BatchResponse, BatchStats};
pub use error::SchedulerError;
pub use policy::SchedulingPolicy;
pub use scheduler::{ScheduledResponse, Scheduler, SchedulerOutput, WaitingRequest};
