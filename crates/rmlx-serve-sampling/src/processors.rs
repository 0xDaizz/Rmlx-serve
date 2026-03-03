//! Logits processors: trait definition and concrete implementations.
//!
//! Each processor mutates a logit slice in-place before sampling.

use std::collections::{HashMap, HashSet};

use rand::Rng;

use crate::sampler::softmax;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A processor that mutates logits in-place before the final sampling step.
///
/// `token_ids` is the list of token ids generated so far (the context), which
/// some processors (e.g. repetition penalty) use to decide how to modify logits.
pub trait LogitsProcessor: Send {
    fn process(&self, logits: &mut [f32], token_ids: &[u32]);
}

// ---------------------------------------------------------------------------
// TemperatureProcessor
// ---------------------------------------------------------------------------

/// Divides all logits by the given temperature.
///
/// Higher temperature flattens the distribution (more random); lower temperature
/// sharpens it (more deterministic). Temperature must be > 0.
pub struct TemperatureProcessor {
    pub temperature: f32,
}

impl LogitsProcessor for TemperatureProcessor {
    fn process(&self, logits: &mut [f32], _token_ids: &[u32]) {
        if self.temperature > 0.0 && (self.temperature - 1.0).abs() > f32::EPSILON {
            let inv_temp = 1.0 / self.temperature;
            for l in logits.iter_mut() {
                *l *= inv_temp;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TopKProcessor
// ---------------------------------------------------------------------------

/// Keeps only the top-k logits; sets the rest to negative infinity.
pub struct TopKProcessor {
    pub k: u32,
}

impl LogitsProcessor for TopKProcessor {
    fn process(&self, logits: &mut [f32], _token_ids: &[u32]) {
        let k = self.k as usize;
        if k == 0 || k >= logits.len() {
            return;
        }

        // Find the k-th largest value via partial sort on indices.
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let threshold = logits[indices[k - 1]];

        // Count how many elements are >= threshold to handle ties.
        let count_above = logits.iter().filter(|&&v| v >= threshold).count();

        if count_above > k {
            // There are ties at the boundary. We need to keep exactly k tokens.
            // Keep all tokens strictly above threshold, plus enough at the threshold.
            let strictly_above = logits.iter().filter(|&&v| v > threshold).count();
            let ties_to_keep = k - strictly_above;
            let mut ties_kept = 0;
            for l in logits.iter_mut() {
                if *l > threshold {
                    // keep
                } else if *l == threshold && ties_kept < ties_to_keep {
                    ties_kept += 1;
                    // keep
                } else {
                    *l = f32::NEG_INFINITY;
                }
            }
        } else {
            for l in logits.iter_mut() {
                if *l < threshold {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TopPProcessor
// ---------------------------------------------------------------------------

/// Nucleus (top-p) sampling: keeps the smallest set of tokens whose cumulative
/// probability mass is >= p, sets the rest to negative infinity.
pub struct TopPProcessor {
    pub p: f32,
}

impl LogitsProcessor for TopPProcessor {
    fn process(&self, logits: &mut [f32], _token_ids: &[u32]) {
        if self.p >= 1.0 || logits.is_empty() {
            return;
        }

        // Convert logits to probabilities.
        let probs = softmax(logits);

        // Build (index, probability) pairs and sort by probability descending.
        let mut indexed: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Cumulative sum; once cumsum > p, mask remaining tokens.
        let mut cumsum = 0.0f32;
        let mut keep = vec![false; logits.len()];

        for &(idx, prob) in &indexed {
            cumsum += prob;
            keep[idx] = true;
            if cumsum >= self.p {
                break;
            }
        }

        for (i, l) in logits.iter_mut().enumerate() {
            if !keep[i] {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MinPProcessor
// ---------------------------------------------------------------------------

/// Minimum probability filtering: removes tokens with probability less than
/// `p * max_prob`.
pub struct MinPProcessor {
    pub p: f32,
    /// The minimum number of tokens that must survive filtering regardless of
    /// the probability threshold. Defaults to 1.
    pub min_tokens_to_keep: usize,
}

impl MinPProcessor {
    /// Creates a new `MinPProcessor` with the given probability threshold and
    /// a default `min_tokens_to_keep` of 1.
    pub fn new(p: f32) -> Self {
        Self {
            p,
            min_tokens_to_keep: 1,
        }
    }
}

impl LogitsProcessor for MinPProcessor {
    fn process(&self, logits: &mut [f32], _token_ids: &[u32]) {
        if self.p <= 0.0 || logits.is_empty() {
            return;
        }

        let probs = softmax(logits);

        let max_prob = probs
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let threshold = self.p * max_prob;

        // Count how many tokens are above the threshold.
        let count_above = probs.iter().filter(|&&p| p >= threshold).count();

        // If fewer tokens survive than `min_tokens_to_keep`, skip filtering
        // entirely so that at least that many tokens remain.
        if count_above < self.min_tokens_to_keep {
            return;
        }

        // Build a list of (index, prob) sorted by descending probability so we
        // can identify the top-N tokens to protect.
        let mut indexed: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Collect the indices of the top `min_tokens_to_keep` tokens that must
        // never be filtered out.
        let keep_set: HashSet<usize> = indexed
            .iter()
            .take(self.min_tokens_to_keep)
            .map(|&(i, _)| i)
            .collect();

        for (i, l) in logits.iter_mut().enumerate() {
            if probs[i] < threshold && !keep_set.contains(&i) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RepetitionPenaltyProcessor
// ---------------------------------------------------------------------------

/// Repetition penalty: for tokens present in the recent context, penalises
/// their logits direction-dependently.
///
/// - If logit > 0: divide by `penalty`
/// - If logit < 0: multiply by `penalty`
///
/// This ensures that high penalties always push the token *away* from being
/// sampled, regardless of logit sign.
///
/// Only the most recent `context_size` tokens are considered.
pub struct RepetitionPenaltyProcessor {
    pub penalty: f32,
    pub context_size: usize,
}

impl LogitsProcessor for RepetitionPenaltyProcessor {
    fn process(&self, logits: &mut [f32], token_ids: &[u32]) {
        if (self.penalty - 1.0).abs() < f32::EPSILON || token_ids.is_empty() {
            return;
        }

        // Consider only the last `context_size` tokens.
        let start = token_ids.len().saturating_sub(self.context_size);
        let context = &token_ids[start..];

        // Collect unique token ids in the context window.
        let mut seen = std::collections::HashSet::new();
        for &tid in context {
            seen.insert(tid);
        }

        for tid in seen {
            let idx = tid as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.penalty;
                } else {
                    logits[idx] *= self.penalty;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FrequencyPresencePenaltyProcessor
// ---------------------------------------------------------------------------

/// Applies frequency and presence penalties based on token counts in the
/// generated output so far.
///
/// For each token id in `token_ids`:
///   `logit -= frequency_penalty * count + presence_penalty * (count > 0 ? 1 : 0)`
pub struct FrequencyPresencePenaltyProcessor {
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

impl LogitsProcessor for FrequencyPresencePenaltyProcessor {
    fn process(&self, logits: &mut [f32], token_ids: &[u32]) {
        if token_ids.is_empty() {
            return;
        }

        // Count occurrences.
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &tid in token_ids {
            *counts.entry(tid).or_insert(0) += 1;
        }

        for (&tid, &count) in &counts {
            let idx = tid as usize;
            if idx < logits.len() {
                logits[idx] -= self.frequency_penalty * count as f32
                    + self.presence_penalty;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LogitBiasProcessor
// ---------------------------------------------------------------------------

/// Adds per-token additive bias to logits.
pub struct LogitBiasProcessor {
    pub bias: HashMap<u32, f32>,
}

impl LogitsProcessor for LogitBiasProcessor {
    fn process(&self, logits: &mut [f32], _token_ids: &[u32]) {
        for (&tid, &b) in &self.bias {
            let idx = tid as usize;
            if idx < logits.len() {
                logits[idx] += b;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// XtcProcessor
// ---------------------------------------------------------------------------

/// eXtraordinary Token Choice (XTC): with a given probability, removes all
/// tokens above the threshold probability except the one with the *minimum*
/// probability among those above the threshold, encouraging diversity.
///
/// More precisely:
/// 1. Compute probabilities via softmax.
/// 2. Identify tokens whose probability >= `threshold`.
/// 3. With probability `self.probability`, set all of those tokens' logits to
///    -inf EXCEPT the one with the lowest probability among them.
pub struct XtcProcessor {
    pub probability: f32,
    pub threshold: f32,
    /// Token ids that should never be masked by XTC (e.g. EOS, newline tokens).
    pub excluded_token_ids: HashSet<u32>,
}

impl LogitsProcessor for XtcProcessor {
    fn process(&self, logits: &mut [f32], _token_ids: &[u32]) {
        if self.probability <= 0.0 || self.threshold <= 0.0 || logits.is_empty() {
            return;
        }

        // Roll the dice: only activate with the configured probability.
        let mut rng = rand::thread_rng();
        let roll: f32 = rng.gen();
        if roll >= self.probability {
            return;
        }

        let probs = softmax(logits);

        // Find tokens above the threshold, excluding special tokens (EOS,
        // newline, etc.) which should never be masked by XTC.
        let above_threshold: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .filter(|(i, &p)| p >= self.threshold && !self.excluded_token_ids.contains(&(*i as u32)))
            .map(|(i, &p)| (i, p))
            .collect();

        // Need at least 2 tokens above threshold to do anything meaningful.
        if above_threshold.len() < 2 {
            return;
        }

        // Find the one with the minimum probability (the "extraordinary" choice).
        let min_idx = above_threshold
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(i, _)| i)
            .unwrap();

        // Set all above-threshold tokens to -inf except the minimum one.
        for &(idx, _) in &above_threshold {
            if idx != min_idx {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature() {
        let mut logits = vec![2.0f32, 4.0, 6.0];
        let proc = TemperatureProcessor { temperature: 2.0 };
        proc.process(&mut logits, &[]);
        assert!((logits[0] - 1.0).abs() < 1e-5);
        assert!((logits[1] - 2.0).abs() < 1e-5);
        assert!((logits[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_k() {
        let mut logits = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
        let proc = TopKProcessor { k: 2 };
        proc.process(&mut logits, &[]);
        // Top 2 are indices 1 (5.0) and 4 (4.0).
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(logits[4], 4.0);
    }

    #[test]
    fn test_top_p() {
        // Softmax of [10, 1, 1, 1] heavily favours index 0.
        let mut logits = vec![10.0f32, 1.0, 1.0, 1.0];
        let proc = TopPProcessor { p: 0.9 };
        proc.process(&mut logits, &[]);
        // Index 0 should be kept, most others masked.
        assert!(logits[0] > f32::NEG_INFINITY);
    }

    #[test]
    fn test_min_p() {
        let mut logits = vec![10.0f32, 1.0, 0.0, -10.0];
        let proc = MinPProcessor::new(0.1);
        proc.process(&mut logits, &[]);
        // Token 0 is max prob; tokens with prob < 0.1 * max_prob should be masked.
        assert!(logits[0] > f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![2.0f32, -1.0, 3.0];
        let proc = RepetitionPenaltyProcessor {
            penalty: 2.0,
            context_size: 100,
        };
        proc.process(&mut logits, &[0, 1]);
        // Token 0: positive, divided by 2 => 1.0
        assert!((logits[0] - 1.0).abs() < 1e-5);
        // Token 1: negative, multiplied by 2 => -2.0
        assert!((logits[1] - (-2.0)).abs() < 1e-5);
        // Token 2: not in context, unchanged.
        assert!((logits[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_frequency_presence_penalty() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let proc = FrequencyPresencePenaltyProcessor {
            frequency_penalty: 0.5,
            presence_penalty: 0.25,
        };
        // Token 0 appears 3 times, token 1 appears 1 time.
        proc.process(&mut logits, &[0, 0, 0, 1]);
        // Token 0: 1.0 - 0.5*3 - 0.25 = 1.0 - 1.5 - 0.25 = -0.75
        assert!((logits[0] - (-0.75)).abs() < 1e-5);
        // Token 1: 2.0 - 0.5*1 - 0.25 = 1.25
        assert!((logits[1] - 1.25).abs() < 1e-5);
        // Token 2: unchanged.
        assert!((logits[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_logit_bias() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let bias = HashMap::from([(0, 10.0), (2, -5.0)]);
        let proc = LogitBiasProcessor { bias };
        proc.process(&mut logits, &[]);
        assert!((logits[0] - 11.0).abs() < 1e-5);
        assert!((logits[1] - 2.0).abs() < 1e-5);
        assert!((logits[2] - (-2.0)).abs() < 1e-5);
    }
}
