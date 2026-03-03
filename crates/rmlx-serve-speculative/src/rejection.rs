//! Rejection sampler for verifying speculative tokens.
//!
//! Implements the core verification algorithm from "Fast Inference from
//! Transformers via Speculative Decoding" (Leviathan et al., 2023) and
//! "Accelerating Large Language Model Decoding with Speculative Sampling"
//! (Chen et al., 2023).
//!
//! Two modes are supported:
//!
//! - **Greedy**: Accept a draft token if it matches the target model's argmax
//!   at that position. Simple and deterministic, but may reject tokens that
//!   would have been acceptable under stochastic sampling.
//!
//! - **Stochastic**: Accept each draft token with probability
//!   `min(1, p_target / p_draft)`. On rejection, sample from the adjusted
//!   distribution `normalize(max(0, p_target - p_draft))`. This preserves
//!   the target model's exact output distribution.
//!
//! In both modes, after processing all draft tokens (either all accepted or
//! first rejection), a "bonus" token is sampled from the target model's
//! distribution at the next position.

use rand::Rng;

use rmlx_serve_sampling::greedy;

/// Result of verifying draft tokens against the target model.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Tokens that were accepted (subset of draft tokens, in order).
    pub accepted_tokens: Vec<u32>,
    /// Number of accepted tokens (== `accepted_tokens.len()`).
    pub num_accepted: usize,
    /// An additional token sampled from the target model's distribution
    /// at the position after the last accepted token. This "bonus" token
    /// is always produced, even if zero draft tokens were accepted.
    pub bonus_token: Option<u32>,
}

/// Rejection sampler for speculative decoding verification.
///
/// Compares draft model proposals against the target model's probability
/// distributions and determines which tokens to accept.
pub struct RejectionSampler {
    /// If true, use greedy (argmax) verification. If false, use stochastic
    /// rejection sampling that preserves the target distribution.
    use_greedy: bool,
}

impl RejectionSampler {
    /// Create a new rejection sampler.
    ///
    /// # Arguments
    /// * `greedy` - If `true`, accept only when the target model's argmax
    ///   matches the draft token. If `false`, use stochastic rejection
    ///   sampling with probability `min(1, p_target/p_draft)`.
    pub fn new(greedy: bool) -> Self {
        Self {
            use_greedy: greedy,
        }
    }

    /// Verify draft tokens against target model probability distributions.
    ///
    /// # Arguments
    /// * `target_probs` - Target model probabilities at each draft position.
    ///   Shape: `[k][vocab_size]`. Each entry is a probability distribution
    ///   over the full vocabulary. An extra entry at index `k` (if present)
    ///   provides the target distribution for the bonus token position.
    /// * `draft_probs` - Draft model probabilities at each position.
    ///   Shape: `[k][vocab_size]`. Empty inner vecs are treated as one-hot
    ///   distributions at the corresponding `draft_tokens[i]` (used by
    ///   n-gram proposers).
    /// * `draft_tokens` - The proposed token IDs, length `k`.
    ///
    /// # Returns
    /// A [`VerificationResult`] containing accepted tokens and a bonus token.
    ///
    /// # Panics
    /// Panics if `draft_tokens.len()` exceeds `target_probs.len()`.
    pub fn verify(
        &self,
        target_probs: &[Vec<f32>],
        draft_probs: &[Vec<f32>],
        draft_tokens: &[u32],
    ) -> VerificationResult {
        assert!(
            draft_tokens.len() <= target_probs.len(),
            "need at least {} target prob distributions, got {}",
            draft_tokens.len(),
            target_probs.len()
        );

        if self.use_greedy {
            self.verify_greedy(target_probs, draft_tokens)
        } else {
            self.verify_stochastic(target_probs, draft_probs, draft_tokens)
        }
    }

    /// Greedy verification: accept while target argmax equals draft token.
    fn verify_greedy(
        &self,
        target_probs: &[Vec<f32>],
        draft_tokens: &[u32],
    ) -> VerificationResult {
        let k = draft_tokens.len();
        let mut accepted_tokens = Vec::with_capacity(k);

        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            let target_argmax = greedy(&target_probs[i]);
            if target_argmax == draft_token {
                accepted_tokens.push(draft_token);
            } else {
                // First rejection -- the target model's argmax at this
                // position becomes the bonus token.
                let num_accepted = accepted_tokens.len();
                return VerificationResult {
                    accepted_tokens,
                    num_accepted,
                    bonus_token: Some(target_argmax),
                };
            }
        }

        // All draft tokens accepted. Bonus token comes from the target
        // distribution at position k (one past the last draft token).
        let num_accepted = accepted_tokens.len();
        let bonus_token = if target_probs.len() > k {
            Some(greedy(&target_probs[k]))
        } else {
            None
        };

        VerificationResult {
            accepted_tokens,
            num_accepted,
            bonus_token,
        }
    }

    /// Stochastic rejection sampling that preserves the target distribution.
    ///
    /// For each draft position i:
    ///   1. Draw r ~ Uniform(0, 1)
    ///   2. If r < p_target[draft_token[i]] / p_draft[draft_token[i]]:
    ///      accept the token
    ///   3. Otherwise: reject, sample from normalized max(0, p_target - p_draft)
    ///
    /// After all k tokens are accepted (or first rejection), sample a bonus
    /// token from the target distribution at the next position.
    fn verify_stochastic(
        &self,
        target_probs: &[Vec<f32>],
        draft_probs: &[Vec<f32>],
        draft_tokens: &[u32],
    ) -> VerificationResult {
        let k = draft_tokens.len();
        let mut rng = rand::thread_rng();
        let mut accepted_tokens = Vec::with_capacity(k);

        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            let target_p = &target_probs[i];
            let token_idx = draft_token as usize;

            // Get target probability for the draft token.
            let p_target = if token_idx < target_p.len() {
                target_p[token_idx]
            } else {
                0.0
            };

            // Get draft probability for the draft token.
            // Empty draft_probs means one-hot at draft_token (n-gram proposer).
            let p_draft = if i < draft_probs.len() && !draft_probs[i].is_empty() {
                if token_idx < draft_probs[i].len() {
                    draft_probs[i][token_idx]
                } else {
                    0.0
                }
            } else {
                // One-hot: draft probability is 1.0 for the proposed token.
                1.0
            };

            // Acceptance probability: min(1, p_target / p_draft).
            let accept_prob = if p_draft > 1e-10 {
                (p_target / p_draft).min(1.0)
            } else if p_target > 0.0 {
                // Draft assigned zero probability but target didn't -- always accept.
                1.0
            } else {
                // Both zero -- this token has zero probability in both models.
                // Accept it (it won't affect the distribution).
                1.0
            };

            let r: f32 = rng.gen();
            if r < accept_prob {
                accepted_tokens.push(draft_token);
            } else {
                // Rejection: sample from the adjusted distribution
                // adjusted[v] = max(0, p_target[v] - p_draft[v])
                let bonus = self.sample_adjusted(target_p, draft_probs, i, &mut rng);
                let num_accepted = accepted_tokens.len();
                return VerificationResult {
                    accepted_tokens,
                    num_accepted,
                    bonus_token: Some(bonus),
                };
            }
        }

        // All k tokens accepted. Sample bonus from target distribution at
        // position k.
        let num_accepted = accepted_tokens.len();
        let bonus_token = if target_probs.len() > k {
            Some(self.sample_from_probs(&target_probs[k], &mut rng))
        } else {
            None
        };

        VerificationResult {
            accepted_tokens,
            num_accepted,
            bonus_token,
        }
    }

    /// Sample from the adjusted distribution: normalize(max(0, p_target - p_draft)).
    fn sample_adjusted(
        &self,
        target_p: &[f32],
        draft_probs: &[Vec<f32>],
        position: usize,
        rng: &mut impl Rng,
    ) -> u32 {
        let vocab_size = target_p.len();
        let mut adjusted = Vec::with_capacity(vocab_size);

        let has_draft_probs =
            position < draft_probs.len() && !draft_probs[position].is_empty();

        for v in 0..vocab_size {
            let p_t = target_p[v];
            let p_d = if has_draft_probs && v < draft_probs[position].len() {
                draft_probs[position][v]
            } else {
                0.0
            };
            adjusted.push((p_t - p_d).max(0.0));
        }

        // Normalize the adjusted distribution.
        let sum: f32 = adjusted.iter().sum();
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for p in adjusted.iter_mut() {
                *p *= inv_sum;
            }
        } else {
            // Fallback: if adjusted distribution is all zeros (target == draft
            // everywhere), sample uniformly from the target distribution.
            return self.sample_from_probs(target_p, rng);
        }

        self.sample_from_probs(&adjusted, rng)
    }

    /// Sample a token from a probability distribution using inverse CDF.
    fn sample_from_probs(&self, probs: &[f32], rng: &mut impl Rng) -> u32 {
        let u: f32 = rng.gen();
        let mut cumulative = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                return i as u32;
            }
        }
        // Fallback for floating-point edge cases.
        (probs.len().saturating_sub(1)) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_all_accepted() {
        let sampler = RejectionSampler::new(true);

        // Target argmax at each position matches draft tokens.
        let target_probs = vec![
            vec![0.1, 0.8, 0.1], // argmax = 1
            vec![0.1, 0.1, 0.8], // argmax = 2
            vec![0.7, 0.2, 0.1], // argmax = 0 (bonus position)
        ];
        let draft_probs = vec![vec![0.1, 0.7, 0.2], vec![0.2, 0.1, 0.7]];
        let draft_tokens = vec![1, 2];

        let result = sampler.verify(&target_probs, &draft_probs, &draft_tokens);
        assert_eq!(result.num_accepted, 2);
        assert_eq!(result.accepted_tokens, vec![1, 2]);
        assert_eq!(result.bonus_token, Some(0));
    }

    #[test]
    fn test_greedy_first_rejection() {
        let sampler = RejectionSampler::new(true);

        // Target argmax at position 0 is 0, but draft says 1 -> reject immediately.
        let target_probs = vec![
            vec![0.8, 0.1, 0.1], // argmax = 0
            vec![0.1, 0.1, 0.8], // never reached
        ];
        let draft_probs = vec![vec![0.1, 0.7, 0.2], vec![0.2, 0.1, 0.7]];
        let draft_tokens = vec![1, 2];

        let result = sampler.verify(&target_probs, &draft_probs, &draft_tokens);
        assert_eq!(result.num_accepted, 0);
        assert!(result.accepted_tokens.is_empty());
        // Bonus token is the target argmax at the rejection position.
        assert_eq!(result.bonus_token, Some(0));
    }

    #[test]
    fn test_greedy_partial_acceptance() {
        let sampler = RejectionSampler::new(true);

        let target_probs = vec![
            vec![0.1, 0.8, 0.1], // argmax = 1 (matches draft)
            vec![0.8, 0.1, 0.1], // argmax = 0 (does NOT match draft token 2)
        ];
        let draft_probs = vec![vec![0.1, 0.7, 0.2], vec![0.2, 0.1, 0.7]];
        let draft_tokens = vec![1, 2];

        let result = sampler.verify(&target_probs, &draft_probs, &draft_tokens);
        assert_eq!(result.num_accepted, 1);
        assert_eq!(result.accepted_tokens, vec![1]);
        assert_eq!(result.bonus_token, Some(0));
    }

    #[test]
    fn test_stochastic_high_target_prob() {
        let sampler = RejectionSampler::new(false);

        // Target prob >> draft prob for all draft tokens -> very likely to accept.
        let target_probs = vec![
            vec![0.01, 0.98, 0.01], // draft token 1 has high target prob
            vec![0.01, 0.01, 0.98], // draft token 2 has high target prob
            vec![0.5, 0.3, 0.2],    // bonus position
        ];
        let draft_probs = vec![
            vec![0.3, 0.4, 0.3], // draft token 1 has lower prob
            vec![0.3, 0.3, 0.4], // draft token 2 has lower prob
        ];
        let draft_tokens = vec![1, 2];

        // Run many times -- should almost always accept both.
        let mut all_accepted_count = 0;
        for _ in 0..100 {
            let result = sampler.verify(&target_probs, &draft_probs, &draft_tokens);
            if result.num_accepted == 2 {
                all_accepted_count += 1;
            }
        }
        // With these probabilities, acceptance should be very high.
        assert!(
            all_accepted_count > 80,
            "expected most runs to accept all tokens, got {all_accepted_count}/100"
        );
    }

    #[test]
    fn test_stochastic_zero_target_prob() {
        let sampler = RejectionSampler::new(false);

        // Target gives zero probability to draft token -> always reject.
        let target_probs = vec![
            vec![0.5, 0.0, 0.5], // draft token 1 has zero target prob
        ];
        let draft_probs = vec![
            vec![0.2, 0.6, 0.2], // draft token 1 has high draft prob
        ];
        let draft_tokens = vec![1];

        let result = sampler.verify(&target_probs, &draft_probs, &draft_tokens);
        assert_eq!(result.num_accepted, 0);
        assert!(result.bonus_token.is_some());
        // Bonus should be sampled from adjusted = max(0, target - draft)
        // adjusted = [0.3, 0.0, 0.3] -> normalized to [0.5, 0.0, 0.5]
        // So bonus should be 0 or 2, never 1.
        let bonus = result.bonus_token.unwrap();
        assert!(bonus == 0 || bonus == 2, "bonus was {bonus}, expected 0 or 2");
    }

    #[test]
    fn test_empty_draft_probs_treated_as_one_hot() {
        let sampler = RejectionSampler::new(false);

        // Empty draft probs (n-gram style) -- treated as one-hot.
        let target_probs = vec![
            vec![0.1, 0.8, 0.1], // draft token 1 has 0.8 target prob
            vec![0.5, 0.3, 0.2], // bonus position
        ];
        let draft_probs = vec![vec![]]; // empty = one-hot at draft_token
        let draft_tokens = vec![1];

        // accept_prob = min(1, 0.8/1.0) = 0.8 -- should accept ~80% of the time.
        let mut accept_count = 0;
        for _ in 0..1000 {
            let result = sampler.verify(&target_probs, &draft_probs, &draft_tokens);
            if result.num_accepted == 1 {
                accept_count += 1;
            }
        }
        // Should be roughly 800/1000 with some variance.
        assert!(
            accept_count > 700 && accept_count < 900,
            "expected ~800 acceptances, got {accept_count}/1000"
        );
    }
}
