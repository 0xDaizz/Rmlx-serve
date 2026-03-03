//! N-gram based speculative token proposer.
//!
//! Uses a hash table of n-gram patterns built from previously generated tokens
//! to predict future tokens. This is a lightweight, zero-cost proposer that
//! works well for repetitive text (code, structured output, etc.) but has
//! lower acceptance rates for creative or novel text.
//!
//! Based on the approach described in "REST: Retrieval-Based Speculative
//! Decoding" and similar n-gram speculation techniques.

use std::collections::HashMap;

use crate::error::SpecError;
use crate::proposal::{Proposal, Proposer};

/// Speculative proposer based on n-gram pattern matching.
///
/// Maintains a table mapping (n-1)-token sequences to observed next tokens.
/// During proposal, looks up the most recent context suffix and follows
/// the chain greedily for `k` steps.
pub struct NgramProposer {
    /// The n-gram order (e.g., 3 means trigrams: 2 context tokens -> 1 predicted).
    n: usize,
    /// Mapping from (n-1)-token prefix to list of observed next tokens.
    /// When multiple next tokens exist, the most recently observed one wins
    /// (last element in the Vec).
    table: HashMap<Vec<u32>, Vec<u32>>,
    /// Minimum number of context tokens required before proposing.
    min_context: usize,
    /// Number of tokens already indexed from the context. This enables
    /// incremental updates: on each call to `update_incremental`, only the
    /// new n-grams (those involving tokens at positions >= `indexed_up_to`)
    /// are added to the table, avoiding redundant work.
    indexed_up_to: usize,
}

impl NgramProposer {
    /// Create a new n-gram proposer with the given n-gram order.
    ///
    /// # Arguments
    /// * `n` - The n-gram size (typically 3-5). The proposer uses (n-1) context
    ///   tokens to predict the next token.
    ///
    /// # Panics
    /// Panics if `n < 2` since at least a bigram is needed.
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "n-gram order must be at least 2, got {n}");
        Self {
            n,
            table: HashMap::new(),
            min_context: n - 1,
            indexed_up_to: 0,
        }
    }

    /// Update the n-gram table with newly generated tokens.
    ///
    /// Scans all (n-1)-grams in the token sequence and records the following
    /// token. Should be called after each generation step with the full
    /// token history (or at least the last `n` tokens).
    pub fn update(&mut self, tokens: &[u32]) {
        if tokens.len() < self.n {
            return;
        }
        // Build n-grams from all windows of size n.
        for window in tokens.windows(self.n) {
            let prefix = window[..self.n - 1].to_vec();
            let next_token = window[self.n - 1];
            let entry = self.table.entry(prefix).or_default();
            // Append the next token. The most recent occurrence will be at the
            // end of the vec, which is what we pick during greedy proposal.
            entry.push(next_token);
        }
    }

    /// Incrementally update the n-gram table with only the new tokens since
    /// the last call.
    ///
    /// This avoids re-scanning the entire token history on every propose() call.
    /// Only n-gram windows that include at least one new token are indexed.
    fn update_incremental(&mut self, tokens: &[u32]) {
        if tokens.len() < self.n {
            return;
        }

        // Determine the start of the first new window. Each window of size `n`
        // ending at index `i` starts at index `i - n + 1`. We need to index
        // all windows whose last element is at position >= indexed_up_to.
        // The window ending at position `p` starts at `p - n + 1`.
        let start = if self.indexed_up_to >= self.n {
            // We already indexed windows ending before indexed_up_to.
            // The first new window starts at indexed_up_to - (n - 1).
            self.indexed_up_to - (self.n - 1)
        } else {
            0
        };

        for window in tokens[start..].windows(self.n) {
            let prefix = window[..self.n - 1].to_vec();
            let next_token = window[self.n - 1];
            self.table.entry(prefix).or_default().push(next_token);
        }

        self.indexed_up_to = tokens.len();
    }

    /// Look up the next token for a given (n-1)-token prefix.
    ///
    /// Returns the most recently observed next token (last in the list),
    /// or `None` if the prefix has not been seen.
    fn lookup(&self, prefix: &[u32]) -> Option<u32> {
        self.table
            .get(prefix)
            .and_then(|nexts| nexts.last().copied())
    }
}

impl Proposer for NgramProposer {
    fn propose(&mut self, context_tokens: &[u32], k: usize) -> Result<Proposal, SpecError> {
        if context_tokens.len() < self.min_context {
            return Err(SpecError::ProposalFailed(format!(
                "need at least {} context tokens for {}-gram, got {}",
                self.min_context,
                self.n,
                context_tokens.len()
            )));
        }

        // Incrementally update the table with only new tokens since the last
        // call, avoiding redundant re-scanning of the full context history.
        self.update_incremental(context_tokens);

        let prefix_len = self.n - 1;
        let mut token_ids = Vec::with_capacity(k);
        let mut probabilities = Vec::with_capacity(k);

        // Start with the last (n-1) tokens from context as our lookup key.
        let mut current_prefix: Vec<u32> =
            context_tokens[context_tokens.len() - prefix_len..].to_vec();

        for _ in 0..k {
            match self.lookup(&current_prefix) {
                Some(next_token) => {
                    token_ids.push(next_token);

                    // N-gram proposals use one-hot probability distributions
                    // since we have no real probability model -- the chosen
                    // token gets probability 1.0, all others get 0.0.
                    // We use an empty vec as a sentinel for "one-hot at token_id".
                    // The rejection sampler handles this specially.
                    //
                    // For correctness with the rejection sampler, we do NOT
                    // allocate a full vocab-size vector here (we don't know
                    // the vocab size). Instead we store an empty vec and the
                    // rejection sampler treats empty draft probs as one-hot.
                    probabilities.push(Vec::new());

                    // Advance the prefix window.
                    current_prefix.remove(0);
                    current_prefix.push(next_token);
                }
                None => {
                    // No match found -- stop proposing early.
                    break;
                }
            }
        }

        if token_ids.is_empty() {
            return Err(SpecError::ProposalFailed(
                "no n-gram matches found for current context".into(),
            ));
        }

        Ok(Proposal {
            token_ids,
            probabilities,
        })
    }

    fn reset(&mut self) {
        self.table.clear();
        self.indexed_up_to = 0;
    }

    fn name(&self) -> &str {
        "ngram"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_basic() {
        let mut proposer = NgramProposer::new(3);

        // Feed a repeating sequence: 1 2 3 1 2 3 1 2 3
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        proposer.update(&tokens);

        // Context ends with [2, 3], trigram should predict 1.
        let result = proposer.propose(&[2, 3], 1);
        assert!(result.is_ok());
        let proposal = result.unwrap();
        assert_eq!(proposal.token_ids, vec![1]);
    }

    #[test]
    fn test_ngram_chain() {
        let mut proposer = NgramProposer::new(3);

        let tokens = vec![10, 20, 30, 40, 50, 10, 20, 30, 40, 50];
        proposer.update(&tokens);

        // Context: [20, 30] -> should predict 40, then [30, 40] -> 50, etc.
        let result = proposer.propose(&[10, 20, 30], 3);
        assert!(result.is_ok());
        let proposal = result.unwrap();
        assert_eq!(proposal.token_ids, vec![40, 50, 10]);
    }

    #[test]
    fn test_ngram_insufficient_context() {
        let mut proposer = NgramProposer::new(4);

        // Only 2 tokens, need 3 for 4-gram.
        let result = proposer.propose(&[1, 2], 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_ngram_no_match() {
        let mut proposer = NgramProposer::new(3);

        let tokens = vec![1, 2, 3];
        proposer.update(&tokens);

        // Context [5, 6] has never been seen.
        let result = proposer.propose(&[5, 6], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_ngram_reset() {
        let mut proposer = NgramProposer::new(3);

        let tokens = vec![1, 2, 3, 1, 2, 3];
        proposer.update(&tokens);

        proposer.reset();

        // After reset, table is empty so no matches.
        let result = proposer.propose(&[1, 2], 1);
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "n-gram order must be at least 2")]
    fn test_ngram_invalid_n() {
        NgramProposer::new(1);
    }
}
