//! rmlx-serve-sampling: Token sampling strategies for the rmlx-serve inference engine.
//!
//! This crate provides all sampling logic needed to select the next token from
//! a vocabulary distribution. It operates entirely on CPU `f32` slices -- the
//! caller is responsible for copying logits from GPU memory before invoking
//! these functions.
//!
//! # Architecture
//!
//! The sampling pipeline is split into two phases:
//!
//! 1. **Logits processors** ([`LogitsProcessor`] trait) -- context-dependent
//!    mutations applied before sampling. These include logit bias, repetition
//!    penalty, and frequency/presence penalties. They are created via
//!    [`make_logits_processors`].
//!
//! 2. **Sampler** -- a closure `&[f32] -> u32` that applies temperature
//!    scaling, top-k, top-p, min-p filtering and then performs either greedy
//!    (argmax) or categorical (multinomial) sampling. Created via
//!    [`make_sampler`].
//!
//! # Example
//!
//! ```ignore
//! use rmlx_serve_types::SamplingParams;
//! use rmlx_serve_sampling::{make_sampler, make_logits_processors};
//!
//! let params = SamplingParams { temperature: 0.8, top_p: 0.95, ..Default::default() };
//! let processors = make_logits_processors(&params);
//! let sample = make_sampler(&params);
//!
//! // During generation:
//! let mut logits: Vec<f32> = get_logits_from_model(); // hypothetical
//! let context_token_ids: &[u32] = &[/* previously generated tokens */];
//!
//! for proc in &processors {
//!     proc.process(&mut logits, context_token_ids);
//! }
//!
//! let next_token_id = sample(&logits);
//! ```

pub mod error;
pub mod pipeline;
pub mod processors;
pub mod sampler;

// Re-export the main public API at crate root.
pub use error::SamplingError;
pub use pipeline::{make_logits_processors, make_sampler, SamplerFn};
pub use processors::LogitsProcessor;
pub use sampler::{categorical, greedy, log_softmax, softmax, top_logprobs};
