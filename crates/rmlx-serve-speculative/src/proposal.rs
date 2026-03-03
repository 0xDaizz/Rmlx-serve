//! Proposer trait and Proposal type for speculative token generation.
//!
//! A [`Proposer`] generates candidate (draft) tokens that the target model
//! will verify. Different strategies implement this trait: n-gram lookup,
//! a smaller draft model, or multi-token prediction heads.

use crate::error::SpecError;

/// A batch of speculative token proposals from a draft source.
///
/// Contains the proposed token IDs and their associated probability
/// distributions. The target model will verify these against its own
/// distributions using rejection sampling.
#[derive(Debug, Clone)]
pub struct Proposal {
    /// Proposed token IDs, length `k` (number of speculative tokens).
    pub token_ids: Vec<u32>,

    /// Draft model probability distributions at each speculative position.
    ///
    /// Shape: `[k][vocab_size]`. Each inner `Vec<f32>` is a full probability
    /// distribution over the vocabulary for that position. For simple
    /// proposers (e.g., n-gram), these may be one-hot distributions.
    pub probabilities: Vec<Vec<f32>>,
}

/// Trait for speculative token proposers.
///
/// A proposer generates `k` candidate tokens given a context sequence.
/// Different implementations trade off quality (acceptance rate) against
/// cost (time to generate proposals).
pub trait Proposer: Send {
    /// Generate up to `k` speculative token proposals given the context.
    ///
    /// # Arguments
    /// * `context_tokens` - All tokens generated so far (prompt + generated).
    /// * `k` - Maximum number of speculative tokens to propose.
    ///
    /// # Returns
    /// A [`Proposal`] containing token IDs and their probability distributions,
    /// or an error if the proposer cannot generate candidates.
    fn propose(&mut self, context_tokens: &[u32], k: usize) -> Result<Proposal, SpecError>;

    /// Reset any internal state (e.g., clear draft model KV cache).
    ///
    /// Called when starting a new generation sequence or when the context
    /// has changed in a way that invalidates prior state.
    fn reset(&mut self);

    /// Human-readable name identifying this proposer strategy.
    fn name(&self) -> &str;
}
