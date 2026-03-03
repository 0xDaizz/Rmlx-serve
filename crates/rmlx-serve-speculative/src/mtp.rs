//! Multi-Token Prediction (MTP) proposer.
//!
//! MTP uses additional prediction heads attached to the target model to
//! predict multiple future tokens in a single forward pass. Unlike the
//! draft model approach, MTP shares the backbone computation with the
//! target model and only adds lightweight prediction heads.
//!
//! This is a placeholder implementation. Full MTP requires deep integration
//! with the target model's forward pass to extract intermediate hidden
//! states and route them through the MTP heads. This integration must be
//! done at the engine level since the proposer needs access to the target
//! model's internal representations.
//!
//! Reference: "Better & Faster Large Language Models via Multi-token
//! Prediction" (Gloeckle et al., 2024).

use crate::error::SpecError;
use crate::proposal::{Proposal, Proposer};

/// Multi-Token Prediction proposer.
///
/// MTP predicts multiple future tokens using additional prediction heads
/// that operate on the target model's hidden states. Since this requires
/// access to the target model's internals during its forward pass, the
/// actual prediction logic must be integrated at the engine level.
///
/// This struct serves as a configuration holder and interface point.
/// The engine will check for MTP configuration and, if present, extract
/// hidden states during the target model's forward pass and run them
/// through the MTP heads.
///
/// In **optimistic** mode, the first draft token is always accepted during
/// verification. This can improve throughput for high-acceptance-rate
/// scenarios at the cost of occasional quality degradation.
pub struct MtpProposer {
    /// Number of additional tokens to predict per forward pass.
    num_predict: usize,
    /// When true, the first draft token is always accepted during verification.
    optimistic: bool,
    /// Cached hidden states from the last forward pass, set externally by the engine.
    /// Layout: `[hidden_dim]` for the last token position.
    hidden_states: Option<Vec<f32>>,
}

impl MtpProposer {
    /// Create a new MTP proposer.
    ///
    /// # Arguments
    /// * `num_predict` - Number of extra tokens to predict (typically 2-4).
    /// * `optimistic` - If true, the first draft token is always accepted.
    pub fn new(num_predict: usize, optimistic: bool) -> Self {
        Self {
            num_predict,
            optimistic,
            hidden_states: None,
        }
    }

    /// Number of tokens this MTP head is configured to predict.
    pub fn num_predict(&self) -> usize {
        self.num_predict
    }

    /// Whether this proposer uses optimistic acceptance for the first token.
    pub fn is_optimistic(&self) -> bool {
        self.optimistic
    }

    /// Set the hidden states from the target model's last forward pass.
    ///
    /// This must be called by the engine after each target model forward pass,
    /// before `propose()` is called. The hidden states are the output of the
    /// final transformer layer for the last token position.
    pub fn set_hidden_states(&mut self, hidden_states: Vec<f32>) {
        self.hidden_states = Some(hidden_states);
    }

    /// Clear cached hidden states.
    pub fn clear_hidden_states(&mut self) {
        self.hidden_states = None;
    }

    /// Propose tokens using the model's multi-token prediction heads.
    ///
    /// In optimistic mode, the first draft token is always accepted during
    /// later verification. When hidden states are available, runs them
    /// through the MTP heads to produce predictions. Without hidden states,
    /// returns an empty proposal.
    ///
    /// # Arguments
    /// * `_context` - Token context (not used directly; MTP uses hidden states).
    /// * `hidden_states` - Optional external hidden states override.
    pub fn propose_from_hidden(&self, _context: &[u32], hidden_states: Option<&[f32]>) -> Vec<u32> {
        let _states = match hidden_states.or(self.hidden_states.as_deref()) {
            Some(s) => s,
            None => return Vec::new(),
        };

        // MTP prediction heads would process the hidden states here.
        // Each MTP head `h` (for h in 0..num_predict) takes the hidden state
        // and produces a probability distribution over the vocabulary.
        // The argmax of each distribution gives the predicted token.
        //
        // This requires the actual MTP head weights to be loaded and a
        // linear projection + softmax to be computed. The engine integration
        // will provide the actual computation. For now, return empty to
        // signal that the engine should fall back to standard decoding.
        Vec::new()
    }
}

impl Proposer for MtpProposer {
    fn propose(&mut self, _context_tokens: &[u32], _k: usize) -> Result<Proposal, SpecError> {
        // MTP requires access to the target model's hidden states during
        // its forward pass. This cannot be implemented as a standalone
        // proposer -- it needs to be integrated into the engine's forward
        // pass pipeline.
        //
        // The engine should:
        // 1. Run the target model's forward pass
        // 2. Extract hidden states from the final transformer layer
        // 3. Pass them through MTP prediction heads
        // 4. Return both the main logits and the MTP predictions
        //
        // See the engine crate for the integration point.
        Err(SpecError::ProposalFailed(
            "MTP requires engine integration: cannot propose tokens without access \
             to target model hidden states during forward pass"
                .into(),
        ))
    }

    fn reset(&mut self) {
        // Clear any cached hidden states from previous sequences.
        self.hidden_states = None;
    }

    fn name(&self) -> &str {
        "mtp"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtp_returns_error() {
        let mut proposer = MtpProposer::new(3, false);
        let result = proposer.propose(&[1, 2, 3], 3);
        assert!(result.is_err());
        match result.unwrap_err() {
            SpecError::ProposalFailed(msg) => {
                assert!(msg.contains("engine integration"));
            }
            other => panic!("expected ProposalFailed, got: {other:?}"),
        }
    }

    #[test]
    fn test_mtp_num_predict() {
        let proposer = MtpProposer::new(4, false);
        assert_eq!(proposer.num_predict(), 4);
        assert!(!proposer.is_optimistic());
    }

    #[test]
    fn test_mtp_optimistic() {
        let proposer = MtpProposer::new(2, true);
        assert!(proposer.is_optimistic());
    }

    #[test]
    fn test_mtp_propose_from_hidden_no_states() {
        let proposer = MtpProposer::new(3, false);
        let tokens = proposer.propose_from_hidden(&[1, 2, 3], None);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_mtp_set_hidden_states() {
        let mut proposer = MtpProposer::new(3, false);
        assert!(proposer.hidden_states.is_none());
        proposer.set_hidden_states(vec![0.1, 0.2, 0.3]);
        assert!(proposer.hidden_states.is_some());
        proposer.clear_hidden_states();
        assert!(proposer.hidden_states.is_none());
    }
}
