//! Speculative decoding runtime coordinator.
//!
//! Orchestrates the proposal-verification loop: obtains draft tokens from a
//! [`Proposer`], verifies them against the target model's distributions via
//! a [`RejectionSampler`], tracks metrics, and auto-disables speculation
//! when acceptance rates drop below a configurable threshold.

use tracing::{debug, info, warn};

use crate::error::SpecError;
use crate::metrics::SpecDecodeMetrics;
use crate::proposal::Proposer;
use crate::rejection::RejectionSampler;

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpecDecodeConfig {
    /// Number of speculative tokens to propose per step (`k`).
    ///
    /// Higher values amortize the overhead of the verification pass but risk
    /// more wasted work if the acceptance rate is low. Typical values: 3-8.
    pub num_speculative_tokens: usize,

    /// Which speculation method to use.
    pub method: SpecMethod,

    /// Automatically disable speculative decoding if the windowed acceptance
    /// rate drops below this threshold. Set to 0.0 to never auto-disable.
    ///
    /// Typical value: 0.2 (disable if fewer than 20% of tokens are accepted).
    pub auto_disable_threshold: f32,

    /// Number of steps between probe attempts when auto-disabled.
    ///
    /// After auto-disabling, the runtime will periodically re-enable
    /// speculation for one step to check if the acceptance rate has improved
    /// (e.g., the model entered a more predictable region). Set to 0 to
    /// never probe.
    pub probe_interval: usize,
}

/// Speculative decoding method selector.
#[derive(Debug, Clone)]
pub enum SpecMethod {
    /// N-gram table lookup. Cheap but limited to patterns seen in context.
    Ngram {
        /// N-gram order (typically 3-5).
        n: usize,
    },
    /// Use a separate smaller draft model.
    DraftModel,
    /// Multi-token prediction using target model's MTP heads.
    Mtp {
        /// Number of tokens the MTP head predicts.
        num_predict: usize,
    },
}

/// Speculative decoding runtime.
///
/// Coordinates the full speculative decoding loop:
/// 1. Propose `k` draft tokens using the configured [`Proposer`].
/// 2. Obtain target model probabilities for all draft positions.
/// 3. Verify using [`RejectionSampler`].
/// 4. Track metrics and auto-disable if acceptance rate is too low.
pub struct SpecDecodeRuntime {
    /// The token proposer (n-gram, draft model, or MTP).
    proposer: Box<dyn Proposer>,
    /// Rejection sampler for verification.
    sampler: RejectionSampler,
    /// Runtime configuration.
    config: SpecDecodeConfig,
    /// Performance metrics.
    metrics: SpecDecodeMetrics,
    /// Whether speculative decoding is currently active.
    enabled: bool,
    /// Steps since the last probe attempt (used for re-enabling after auto-disable).
    steps_since_probe: usize,
}

impl SpecDecodeRuntime {
    /// Create a new speculative decoding runtime.
    ///
    /// # Arguments
    /// * `config` - Configuration controlling `k`, method, and auto-disable.
    /// * `proposer` - The token proposer to use.
    pub fn new(config: SpecDecodeConfig, proposer: Box<dyn Proposer>) -> Self {
        let use_greedy = matches!(config.method, SpecMethod::Ngram { .. });
        let sampler = RejectionSampler::new(use_greedy);

        info!(
            method = proposer.name(),
            k = config.num_speculative_tokens,
            threshold = config.auto_disable_threshold,
            "initialized speculative decoding runtime"
        );

        Self {
            proposer,
            sampler,
            config,
            metrics: SpecDecodeMetrics::new(50), // 50-step rolling window
            enabled: true,
            steps_since_probe: 0,
        }
    }

    /// Run one speculative decoding step.
    ///
    /// Proposes `k` draft tokens, gets target probabilities for verification,
    /// runs rejection sampling, and returns the accepted tokens.
    ///
    /// # Arguments
    /// * `context_tokens` - All tokens generated so far.
    /// * `target_probs_fn` - Callback that runs the target model and returns
    ///   probability distributions. Given draft token IDs (appended to
    ///   context), returns `[k+1][vocab_size]` probability distributions:
    ///   one for each draft position plus one for the bonus token position.
    ///
    /// # Returns
    /// A vector of accepted tokens (including the bonus token), or an error
    /// if speculation is disabled or the proposer fails.
    pub fn step(
        &mut self,
        context_tokens: &[u32],
        target_probs_fn: &dyn Fn(&[u32]) -> Vec<Vec<f32>>,
    ) -> Result<Vec<u32>, SpecError> {
        // Check if we should re-enable for a probe attempt.
        let is_probe = !self.enabled;
        if !self.enabled {
            if self.config.probe_interval > 0 {
                self.steps_since_probe += 1;
                if self.steps_since_probe >= self.config.probe_interval {
                    debug!("probing: re-enabling speculative decoding for one step");
                    self.steps_since_probe = 0;
                    // Temporarily enable for this step. We'll check metrics
                    // after and potentially disable again.
                } else {
                    return Err(SpecError::ProposalFailed(
                        "speculative decoding is auto-disabled".into(),
                    ));
                }
            } else {
                return Err(SpecError::ProposalFailed(
                    "speculative decoding is disabled".into(),
                ));
            }
        }

        let k = self.config.num_speculative_tokens;

        // Step 1: Propose k draft tokens.
        let proposal = self.proposer.propose(context_tokens, k)?;
        let num_proposed = proposal.token_ids.len();

        if num_proposed == 0 {
            return Err(SpecError::ProposalFailed(
                "proposer returned zero tokens".into(),
            ));
        }

        debug!(
            proposer = self.proposer.name(),
            num_proposed,
            "generated draft tokens"
        );

        // Step 2: Get target model probabilities for all draft positions.
        // The target model evaluates all draft tokens in a single forward
        // pass and returns probability distributions at each position.
        let target_probs = target_probs_fn(&proposal.token_ids);

        if target_probs.len() < num_proposed {
            return Err(SpecError::VerificationFailed(format!(
                "target_probs_fn returned {} distributions, need at least {}",
                target_probs.len(),
                num_proposed
            )));
        }

        // Step 3: Verify via rejection sampling.
        let result = self.sampler.verify(
            &target_probs,
            &proposal.probabilities,
            &proposal.token_ids,
        );

        debug!(
            num_proposed,
            num_accepted = result.num_accepted,
            has_bonus = result.bonus_token.is_some(),
            "verification complete"
        );

        // Step 4: Update metrics.
        self.metrics.record(num_proposed, result.num_accepted);

        // Step 5: Auto-disable check.
        let windowed_rate = self.metrics.windowed_acceptance_rate();
        if self.config.auto_disable_threshold > 0.0
            && self.metrics.total_steps() >= 10
            && windowed_rate < self.config.auto_disable_threshold as f64
        {
            warn!(
                windowed_rate = format!("{:.2}", windowed_rate),
                threshold = self.config.auto_disable_threshold,
                "auto-disabling speculative decoding due to low acceptance rate"
            );
            self.enabled = false;
            self.steps_since_probe = 0;
        }

        // Step 6: After a probe, check if acceptance rate warrants reactivation.
        if is_probe {
            let probe_rate = self.metrics.windowed_acceptance_rate();
            if self.config.auto_disable_threshold > 0.0
                && probe_rate >= self.config.auto_disable_threshold as f64
            {
                info!(
                    probe_rate = format!("{:.2}", probe_rate),
                    threshold = self.config.auto_disable_threshold,
                    "probe succeeded: re-enabling speculative decoding"
                );
                self.enabled = true;
                self.steps_since_probe = 0;
            } else {
                debug!(
                    probe_rate = format!("{:.2}", probe_rate),
                    threshold = self.config.auto_disable_threshold,
                    "probe failed: keeping speculative decoding disabled"
                );
            }
        }

        // Build the final output: accepted tokens + bonus token.
        let mut output = result.accepted_tokens;
        if let Some(bonus) = result.bonus_token {
            output.push(bonus);
        }

        Ok(output)
    }

    /// Whether speculative decoding is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Access the current performance metrics.
    pub fn metrics(&self) -> &SpecDecodeMetrics {
        &self.metrics
    }

    /// Manually enable speculative decoding.
    pub fn enable(&mut self) {
        self.enabled = true;
        self.steps_since_probe = 0;
        info!("speculative decoding manually enabled");
    }

    /// Manually disable speculative decoding.
    pub fn disable(&mut self) {
        self.enabled = false;
        info!("speculative decoding manually disabled");
    }

    /// Reset the proposer and metrics. Call when starting a new sequence.
    pub fn reset(&mut self) {
        self.proposer.reset();
        self.metrics = SpecDecodeMetrics::new(50);
        self.enabled = true;
        self.steps_since_probe = 0;
        debug!("speculative decoding runtime reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proposal::Proposal;

    /// A test proposer that returns fixed tokens.
    struct FixedProposer {
        tokens: Vec<u32>,
        probs: Vec<Vec<f32>>,
    }

    impl FixedProposer {
        fn new(tokens: Vec<u32>, vocab_size: usize) -> Self {
            let probs = tokens
                .iter()
                .map(|&t| {
                    let mut p = vec![0.0; vocab_size];
                    if (t as usize) < vocab_size {
                        p[t as usize] = 1.0;
                    }
                    p
                })
                .collect();
            Self { tokens, probs }
        }
    }

    impl Proposer for FixedProposer {
        fn propose(
            &mut self,
            _context_tokens: &[u32],
            k: usize,
        ) -> Result<Proposal, SpecError> {
            let n = k.min(self.tokens.len());
            Ok(Proposal {
                token_ids: self.tokens[..n].to_vec(),
                probabilities: self.probs[..n].to_vec(),
            })
        }

        fn reset(&mut self) {}

        fn name(&self) -> &str {
            "fixed_test"
        }
    }

    fn make_config(k: usize) -> SpecDecodeConfig {
        // Use Ngram method so that the runtime creates a greedy RejectionSampler,
        // making test assertions deterministic.
        SpecDecodeConfig {
            num_speculative_tokens: k,
            method: SpecMethod::Ngram { n: 3 },
            auto_disable_threshold: 0.0, // disable auto-disable for tests
            probe_interval: 0,
        }
    }

    #[test]
    fn test_runtime_all_accepted() {
        let proposer = FixedProposer::new(vec![1, 2, 3], 4);
        let config = make_config(3);
        let mut runtime = SpecDecodeRuntime::new(config, Box::new(proposer));

        // Target model agrees with all draft tokens.
        let target_fn = |_draft: &[u32]| -> Vec<Vec<f32>> {
            vec![
                vec![0.05, 0.85, 0.05, 0.05], // argmax=1
                vec![0.05, 0.05, 0.85, 0.05], // argmax=2
                vec![0.05, 0.05, 0.05, 0.85], // argmax=3
                vec![0.85, 0.05, 0.05, 0.05], // bonus: argmax=0
            ]
        };

        let result = runtime.step(&[0], &target_fn).unwrap();
        // All 3 accepted + bonus token 0
        assert_eq!(result, vec![1, 2, 3, 0]);
        assert!(runtime.is_enabled());
    }

    #[test]
    fn test_runtime_partial_acceptance() {
        let proposer = FixedProposer::new(vec![1, 2, 3], 4);
        let config = make_config(3);
        let mut runtime = SpecDecodeRuntime::new(config, Box::new(proposer));

        // Target model disagrees at position 1.
        let target_fn = |_draft: &[u32]| -> Vec<Vec<f32>> {
            vec![
                vec![0.05, 0.85, 0.05, 0.05], // argmax=1 (match)
                vec![0.85, 0.05, 0.05, 0.05], // argmax=0 (mismatch with draft token 2)
                vec![0.05, 0.05, 0.05, 0.85], // not reached
                vec![0.85, 0.05, 0.05, 0.05], // not reached
            ]
        };

        let result = runtime.step(&[0], &target_fn).unwrap();
        // Token 1 accepted, token 2 rejected -> bonus is 0 (target argmax at rejection)
        assert_eq!(result, vec![1, 0]);
    }

    #[test]
    fn test_runtime_disabled() {
        let proposer = FixedProposer::new(vec![1], 4);
        let mut config = make_config(1);
        config.probe_interval = 0;
        let mut runtime = SpecDecodeRuntime::new(config, Box::new(proposer));

        runtime.disable();
        assert!(!runtime.is_enabled());

        let target_fn = |_: &[u32]| -> Vec<Vec<f32>> { vec![] };
        let result = runtime.step(&[0], &target_fn);
        assert!(result.is_err());
    }

    #[test]
    fn test_runtime_auto_disable() {
        let proposer = FixedProposer::new(vec![1, 2, 3], 4);
        let mut config = make_config(3);
        config.auto_disable_threshold = 0.5; // disable if < 50% acceptance
        let mut runtime = SpecDecodeRuntime::new(config, Box::new(proposer));

        // Target always rejects everything (argmax is always 0, draft is 1).
        let target_fn = |_draft: &[u32]| -> Vec<Vec<f32>> {
            vec![
                vec![0.85, 0.05, 0.05, 0.05],
                vec![0.85, 0.05, 0.05, 0.05],
                vec![0.85, 0.05, 0.05, 0.05],
                vec![0.85, 0.05, 0.05, 0.05],
            ]
        };

        // Run enough steps to trigger auto-disable (needs >= 10 steps).
        for _ in 0..15 {
            let _ = runtime.step(&[0], &target_fn);
        }

        assert!(!runtime.is_enabled(), "should auto-disable after low acceptance");
    }

    #[test]
    fn test_runtime_reset() {
        let proposer = FixedProposer::new(vec![1], 4);
        let config = make_config(1);
        let mut runtime = SpecDecodeRuntime::new(config, Box::new(proposer));

        runtime.disable();
        assert!(!runtime.is_enabled());

        runtime.reset();
        assert!(runtime.is_enabled());
        assert_eq!(runtime.metrics().total_steps(), 0);
    }
}
