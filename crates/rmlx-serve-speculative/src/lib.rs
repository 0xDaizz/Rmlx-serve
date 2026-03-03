//! rmlx-serve-speculative: Speculative decoding for the rmlx-serve inference engine.
//!
//! This crate implements speculative decoding, a technique that accelerates
//! autoregressive LLM inference by using a fast "draft" source to propose
//! multiple tokens at once, then verifying them against the target model
//! in a single batched forward pass.
//!
//! # Architecture
//!
//! The system is composed of three layers:
//!
//! 1. **Proposers** ([`Proposer`] trait) -- generate candidate tokens:
//!    - [`NgramProposer`] -- zero-cost n-gram pattern matching from context
//!    - [`DraftModelProposer`] -- uses a smaller LLM for higher quality drafts
//!    - [`MtpProposer`] -- multi-token prediction (requires engine integration)
//!
//! 2. **Verification** ([`RejectionSampler`]) -- compares draft proposals
//!    against the target model using either greedy or stochastic rejection
//!    sampling. The stochastic variant preserves the target model's exact
//!    output distribution.
//!
//! 3. **Runtime** ([`SpecDecodeRuntime`]) -- orchestrates the proposal-verify
//!    loop, tracks performance metrics, and auto-disables speculation when
//!    the acceptance rate drops below a configurable threshold.
//!
//! # Example
//!
//! ```ignore
//! use rmlx_serve_speculative::{
//!     NgramProposer, SpecDecodeConfig, SpecDecodeRuntime, SpecMethod,
//! };
//!
//! let config = SpecDecodeConfig {
//!     num_speculative_tokens: 5,
//!     method: SpecMethod::Ngram { n: 3 },
//!     auto_disable_threshold: 0.2,
//!     probe_interval: 50,
//! };
//!
//! let proposer = NgramProposer::new(3);
//! let mut runtime = SpecDecodeRuntime::new(config, Box::new(proposer));
//!
//! // During generation:
//! let context = vec![1u32, 2, 3, 4, 5];
//! let result = runtime.step(&context, &|draft_tokens| {
//!     // Run target model on draft tokens, return probabilities
//!     // Shape: [k+1][vocab_size]
//!     vec![vec![0.0; vocab_size]; draft_tokens.len() + 1]
//! });
//! ```

pub mod draft_model;
pub mod error;
pub mod metrics;
pub mod mtp;
pub mod ngram;
pub mod proposal;
pub mod rejection;
pub mod runtime;

// ── Re-exports of core types ──
pub use draft_model::DraftModelProposer;
pub use error::{Result, SpecError};
pub use metrics::SpecDecodeMetrics;
pub use mtp::MtpProposer;
pub use ngram::NgramProposer;
pub use proposal::{Proposal, Proposer};
pub use rejection::{RejectionSampler, VerificationResult};
pub use runtime::{SpecDecodeConfig, SpecDecodeRuntime, SpecMethod};
