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

/// Multi-Token Prediction proposer (placeholder).
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
pub struct MtpProposer {
    /// Number of additional tokens to predict per forward pass.
    num_predict: usize,
}

impl MtpProposer {
    /// Create a new MTP proposer.
    ///
    /// # Arguments
    /// * `num_predict` - Number of extra tokens to predict (typically 2-4).
    pub fn new(num_predict: usize) -> Self {
        Self { num_predict }
    }

    /// Number of tokens this MTP head is configured to predict.
    pub fn num_predict(&self) -> usize {
        self.num_predict
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
        // No state to reset -- MTP heads are stateless prediction layers.
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
        let mut proposer = MtpProposer::new(3);
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
        let proposer = MtpProposer::new(4);
        assert_eq!(proposer.num_predict(), 4);
    }
}
