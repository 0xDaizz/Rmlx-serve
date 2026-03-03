//! Draft model proposer for speculative decoding.
//!
//! Uses a smaller LLM (the "draft" model) to generate speculative tokens.
//! The draft model runs `k` autoregressive forward passes to produce
//! candidate tokens along with their full probability distributions.
//! These are then verified by the larger target model in a single
//! batched forward pass.

use std::sync::Arc;

use rmlx_core::KernelRegistry;
use rmlx_metal::metal;
use rmlx_metal::GpuDevice;
use rmlx_nn::LayerKvCache;
use rmlx_serve_models::LlmModel;
use rmlx_serve_sampling::softmax;

use crate::error::SpecError;
use crate::proposal::{Proposal, Proposer};

/// Speculative proposer that uses a smaller draft model.
///
/// Runs `k` sequential forward passes through a lightweight draft model
/// to produce token candidates and probability distributions. The draft
/// model should be significantly faster than the target model (e.g., a
/// 1B parameter model drafting for a 70B target).
pub struct DraftModelProposer {
    /// The draft model implementing the LlmModel trait.
    model: Box<dyn LlmModel>,
    /// Kernel registry for dispatching Metal compute pipelines.
    registry: Arc<KernelRegistry>,
    /// Metal command queue for GPU work submission.
    queue: metal::CommandQueue,
    /// GPU device for cache allocation.
    device: GpuDevice,
}

impl DraftModelProposer {
    /// Create a new draft model proposer.
    ///
    /// # Arguments
    /// * `model` - A smaller LLM to use as the draft model.
    /// * `registry` - Shared kernel registry for Metal compute pipelines.
    /// * `queue` - Metal command queue for GPU submissions.
    /// * `device` - GPU device for allocating KV cache buffers.
    pub fn new(
        model: Box<dyn LlmModel>,
        registry: Arc<KernelRegistry>,
        queue: metal::CommandQueue,
        device: GpuDevice,
    ) -> Self {
        Self {
            model,
            registry,
            queue,
            device,
        }
    }
}

impl Proposer for DraftModelProposer {
    fn propose(&mut self, context_tokens: &[u32], k: usize) -> Result<Proposal, SpecError> {
        if context_tokens.is_empty() {
            return Err(SpecError::ProposalFailed(
                "context_tokens must not be empty".into(),
            ));
        }

        // Create a fresh KV cache for the draft model for this proposal.
        // We do a full prefill of the context, then generate k tokens.
        let mut cache: Vec<LayerKvCache> = self.model.make_cache(self.device.raw());

        // Prefill: run the entire context through the draft model to populate
        // the KV cache.
        let prefill_logits = self
            .model
            .forward(context_tokens, Some(&mut cache), &self.registry, &self.queue)
            .map_err(|e| SpecError::DraftModelError(format!("prefill failed: {e}")))?;

        // Extract the logits from the prefill step (last token position).
        let prefill_logits_vec: Vec<f32> = prefill_logits.to_vec_checked();
        let prefill_probs = softmax(&prefill_logits_vec);
        let first_token = rmlx_serve_sampling::greedy(&prefill_logits_vec);

        let mut token_ids = Vec::with_capacity(k);
        let mut probabilities = Vec::with_capacity(k);

        token_ids.push(first_token);
        probabilities.push(prefill_probs);

        // Generate remaining k-1 tokens autoregressively.
        let mut current_token = first_token;
        for _ in 1..k {
            let logits = self
                .model
                .forward(&[current_token], Some(&mut cache), &self.registry, &self.queue)
                .map_err(|e| SpecError::DraftModelError(format!("decode step failed: {e}")))?;

            let logits_vec: Vec<f32> = logits.to_vec_checked();
            let probs = softmax(&logits_vec);
            let next_token = rmlx_serve_sampling::greedy(&logits_vec);

            token_ids.push(next_token);
            probabilities.push(probs);
            current_token = next_token;
        }

        Ok(Proposal {
            token_ids,
            probabilities,
        })
    }

    fn reset(&mut self) {
        // No persistent state to reset -- we create a fresh cache each
        // time propose() is called.
    }

    fn name(&self) -> &str {
        "draft_model"
    }
}
