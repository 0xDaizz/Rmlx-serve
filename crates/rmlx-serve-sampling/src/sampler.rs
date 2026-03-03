//! Core sampling functions: greedy, categorical, softmax, log_softmax, top_logprobs.
//!
//! All operations work on CPU `f32` slices.

use rand::Rng;

/// Greedy (argmax) sampling: returns the index of the largest logit.
///
/// If the slice is empty, returns 0 (caller should ensure non-empty logits).
pub fn greedy(logits: &[f32]) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    let mut best_idx = 0u32;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Numerically stable softmax: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut probs = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &l in logits {
        let e = (l - max_val).exp();
        probs.push(e);
        sum += e;
    }

    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv_sum;
        }
    }

    probs
}

/// Numerically stable log-softmax: `log_softmax(x)_i = x_i - max(x) - log(sum(exp(x_j - max(x))))`.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let log_sum_exp = logits
        .iter()
        .map(|&l| (l - max_val).exp())
        .sum::<f32>()
        .ln();

    logits
        .iter()
        .map(|&l| l - max_val - log_sum_exp)
        .collect()
}

/// Returns the top-k (token_id, log_prob) pairs sorted by descending log probability.
///
/// `k` is clamped to `logits.len()`.
pub fn top_logprobs(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    if logits.is_empty() || k == 0 {
        return Vec::new();
    }

    let log_probs = log_softmax(logits);
    let k = k.min(logits.len());

    // Build (token_id, log_prob) pairs and partial-sort to get top-k.
    let mut indexed: Vec<(u32, f32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &lp)| (i as u32, lp))
        .collect();

    // Partial sort: move the k largest to the front.
    indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    indexed.truncate(k);
    // Sort the top-k by descending log-prob.
    indexed.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    indexed
}

/// Multinomial (categorical) sampling from a probability distribution.
///
/// `logits` are raw logits; this function applies softmax internally then samples.
pub fn categorical(logits: &[f32], rng: &mut impl Rng) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    let probs = softmax(logits);

    // Use a uniform [0, 1) draw and scan the CDF.
    let u: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if u < cumulative {
            return i as u32;
        }
    }

    // Fallback: floating-point rounding edge case -- return last token.
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_basic() {
        let logits = [1.0f32, 3.0, 2.0, 0.5];
        assert_eq!(greedy(&logits), 1);
    }

    #[test]
    fn test_greedy_empty() {
        assert_eq!(greedy(&[]), 0);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_monotonic() {
        let logits = [1.0f32, 2.0, 3.0];
        let probs = softmax(&logits);
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_log_softmax_consistent_with_softmax() {
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let log_probs = log_softmax(&logits);
        for (lp, p) in log_probs.iter().zip(probs.iter()) {
            assert!((lp.exp() - p).abs() < 1e-5);
        }
    }

    #[test]
    fn test_top_logprobs_ordering() {
        let logits = [1.0f32, 5.0, 3.0, 2.0];
        let top = top_logprobs(&logits, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1); // token 1 has highest logit
        assert_eq!(top[1].0, 2); // token 2 has second highest
    }

    #[test]
    fn test_categorical_returns_valid_index() {
        let logits = [1.0f32, 2.0, 3.0];
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let idx = categorical(&logits, &mut rng);
            assert!(idx < 3);
        }
    }
}
