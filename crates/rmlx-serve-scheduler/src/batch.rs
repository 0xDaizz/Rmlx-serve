//! Batch management for grouped sequence processing.
//!
//! Ported from mlx-lm's `BatchGenerator` `Batch` abstraction. A `Batch`
//! holds the state of all active sequences being decoded together.

use rmlx_serve_sampling::LogitsProcessor;
use rmlx_serve_types::FinishReason;

/// Unique identifier for a sequence within the batch generator.
pub type SequenceId = u64;

/// Per-sequence generation state tracked within a batch.
pub struct SequenceState {
    /// Unique identifier for this sequence across all batches.
    pub uid: SequenceId,

    /// All token ids generated so far (excluding the original prompt).
    pub token_ids: Vec<u32>,

    /// The most recently generated token.
    pub current_token: u32,

    /// Maximum number of tokens to generate for this sequence.
    pub max_tokens: usize,

    /// How many tokens have been generated so far.
    pub num_generated: usize,

    /// The sampler closure: takes a logits slice and returns the sampled token id.
    pub sampler: Box<dyn Fn(&[f32]) -> u32 + Send>,

    /// Context-dependent logits processors (repetition penalty, etc.).
    pub logits_processors: Vec<Box<dyn LogitsProcessor>>,

    /// If `Some(k)`, return the top-k log-probabilities at each step.
    pub logprobs_count: Option<usize>,

    /// Why this sequence stopped generating, if it has.
    pub finish_reason: Option<FinishReason>,

    /// Stop token ids -- generation halts when any of these are sampled.
    pub stop_token_ids: Vec<u32>,
}

/// A collection of active sequences being decoded together.
///
/// Provides efficient insertion, removal, filtering, and iteration over
/// the set of in-flight sequences. This mirrors the `Batch` from mlx-lm's
/// `BatchGenerator`.
pub struct Batch {
    sequences: Vec<SequenceState>,
}

impl Batch {
    /// Create a new empty batch.
    pub fn new() -> Self {
        Self {
            sequences: Vec::new(),
        }
    }

    /// Number of sequences currently in the batch.
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Whether the batch is empty (no active sequences).
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Add a sequence to the batch.
    pub fn add(&mut self, state: SequenceState) {
        self.sequences.push(state);
    }

    /// Remove a sequence by its uid. Returns the removed state if found.
    pub fn remove(&mut self, uid: SequenceId) -> Option<SequenceState> {
        if let Some(pos) = self.sequences.iter().position(|s| s.uid == uid) {
            Some(self.sequences.swap_remove(pos))
        } else {
            None
        }
    }

    /// Filter the batch: keep only sequences where `keep[i]` is `true`.
    ///
    /// The `keep` slice must have the same length as the batch. Sequences
    /// at positions where `keep[i]` is `false` are dropped.
    pub fn filter(&mut self, keep: &[bool]) {
        assert_eq!(
            keep.len(),
            self.sequences.len(),
            "filter: keep slice length ({}) must match batch size ({})",
            keep.len(),
            self.sequences.len(),
        );

        let mut write = 0;
        for read in 0..self.sequences.len() {
            if keep[read] {
                if write != read {
                    self.sequences.swap(write, read);
                }
                write += 1;
            }
        }
        self.sequences.truncate(write);
    }

    /// Merge another batch into this one, consuming it.
    pub fn extend(&mut self, other: Batch) {
        self.sequences.extend(other.sequences);
    }

    /// Return the uids of all sequences in the batch.
    pub fn uids(&self) -> Vec<SequenceId> {
        self.sequences.iter().map(|s| s.uid).collect()
    }

    /// Return the current (most recent) token for each sequence.
    pub fn current_tokens(&self) -> Vec<u32> {
        self.sequences.iter().map(|s| s.current_token).collect()
    }

    /// Immutable iterator over all sequences.
    pub fn iter(&self) -> impl Iterator<Item = &SequenceState> {
        self.sequences.iter()
    }

    /// Mutable iterator over all sequences.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut SequenceState> {
        self.sequences.iter_mut()
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_state(uid: SequenceId, token: u32) -> SequenceState {
        SequenceState {
            uid,
            token_ids: vec![],
            current_token: token,
            max_tokens: 100,
            num_generated: 0,
            sampler: Box::new(|logits: &[f32]| {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            }),
            logits_processors: vec![],
            logprobs_count: None,
            finish_reason: None,
            stop_token_ids: vec![],
        }
    }

    #[test]
    fn test_new_batch_is_empty() {
        let batch = Batch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_add_and_len() {
        let mut batch = Batch::new();
        batch.add(make_test_state(1, 10));
        batch.add(make_test_state(2, 20));
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut batch = Batch::new();
        batch.add(make_test_state(1, 10));
        batch.add(make_test_state(2, 20));
        batch.add(make_test_state(3, 30));

        let removed = batch.remove(2);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().uid, 2);
        assert_eq!(batch.len(), 2);

        let not_found = batch.remove(99);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_filter() {
        let mut batch = Batch::new();
        batch.add(make_test_state(1, 10));
        batch.add(make_test_state(2, 20));
        batch.add(make_test_state(3, 30));

        batch.filter(&[true, false, true]);
        assert_eq!(batch.len(), 2);

        let uids = batch.uids();
        assert!(uids.contains(&1));
        assert!(uids.contains(&3));
        assert!(!uids.contains(&2));
    }

    #[test]
    fn test_extend() {
        let mut batch1 = Batch::new();
        batch1.add(make_test_state(1, 10));

        let mut batch2 = Batch::new();
        batch2.add(make_test_state(2, 20));
        batch2.add(make_test_state(3, 30));

        batch1.extend(batch2);
        assert_eq!(batch1.len(), 3);
    }

    #[test]
    fn test_uids_and_current_tokens() {
        let mut batch = Batch::new();
        batch.add(make_test_state(10, 100));
        batch.add(make_test_state(20, 200));

        assert_eq!(batch.uids(), vec![10, 20]);
        assert_eq!(batch.current_tokens(), vec![100, 200]);
    }
}
