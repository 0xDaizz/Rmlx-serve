//! Metrics tracking for speculative decoding performance.
//!
//! Tracks acceptance rates (both cumulative and windowed), total tokens
//! proposed and accepted, and estimated tokens saved. These metrics are
//! used by the runtime to auto-disable speculative decoding when the
//! acceptance rate drops below a configured threshold.

use std::collections::VecDeque;

/// Performance metrics for speculative decoding.
///
/// Tracks both all-time cumulative statistics and a rolling window of
/// recent performance. The windowed metrics are more responsive to
/// changes in acceptance rate and are used for auto-disable decisions.
pub struct SpecDecodeMetrics {
    /// Total tokens proposed across all steps.
    total_proposed: u64,
    /// Total tokens accepted across all steps.
    total_accepted: u64,
    /// Total number of speculative decoding steps executed.
    total_steps: u64,
    /// Rolling window of per-step proposed counts.
    window_proposed: VecDeque<u64>,
    /// Rolling window of per-step accepted counts.
    window_accepted: VecDeque<u64>,
    /// Maximum number of entries in the rolling window.
    window_size: usize,
}

impl SpecDecodeMetrics {
    /// Create a new metrics tracker with the given rolling window size.
    ///
    /// # Arguments
    /// * `window_size` - Number of recent steps to keep in the rolling window.
    ///   A larger window provides more stable estimates; a smaller window
    ///   responds faster to changes.
    pub fn new(window_size: usize) -> Self {
        Self {
            total_proposed: 0,
            total_accepted: 0,
            total_steps: 0,
            window_proposed: VecDeque::with_capacity(window_size),
            window_accepted: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Record the results of one speculative decoding step.
    ///
    /// # Arguments
    /// * `proposed` - Number of tokens proposed in this step (typically `k`).
    /// * `accepted` - Number of tokens accepted by the verifier.
    pub fn record(&mut self, proposed: usize, accepted: usize) {
        let proposed = proposed as u64;
        let accepted = accepted as u64;

        self.total_proposed += proposed;
        self.total_accepted += accepted;
        self.total_steps += 1;

        // Maintain rolling window.
        if self.window_proposed.len() >= self.window_size {
            self.window_proposed.pop_front();
            self.window_accepted.pop_front();
        }
        self.window_proposed.push_back(proposed);
        self.window_accepted.push_back(accepted);
    }

    /// Overall acceptance rate across all steps.
    ///
    /// Returns 0.0 if no tokens have been proposed.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_proposed == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_proposed as f64
    }

    /// Acceptance rate over the most recent window of steps.
    ///
    /// More responsive to recent changes than `acceptance_rate()`.
    /// Returns 0.0 if the window is empty.
    pub fn windowed_acceptance_rate(&self) -> f64 {
        let proposed: u64 = self.window_proposed.iter().sum();
        if proposed == 0 {
            return 0.0;
        }
        let accepted: u64 = self.window_accepted.iter().sum();
        accepted as f64 / proposed as f64
    }

    /// Total tokens saved by speculative decoding.
    ///
    /// Each accepted token would have required a separate forward pass
    /// through the target model. This returns the total number of accepted
    /// tokens, representing the number of target model forward passes saved.
    pub fn tokens_saved(&self) -> u64 {
        self.total_accepted
    }

    /// Total number of speculative decoding steps executed.
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let metrics = SpecDecodeMetrics::new(10);
        assert_eq!(metrics.acceptance_rate(), 0.0);
        assert_eq!(metrics.windowed_acceptance_rate(), 0.0);
        assert_eq!(metrics.tokens_saved(), 0);
        assert_eq!(metrics.total_steps(), 0);
    }

    #[test]
    fn test_record_and_rates() {
        let mut metrics = SpecDecodeMetrics::new(10);

        metrics.record(5, 3); // 60% acceptance
        assert_eq!(metrics.total_steps(), 1);
        assert_eq!(metrics.tokens_saved(), 3);
        assert!((metrics.acceptance_rate() - 0.6).abs() < 1e-10);
        assert!((metrics.windowed_acceptance_rate() - 0.6).abs() < 1e-10);

        metrics.record(5, 5); // 100% acceptance
        assert_eq!(metrics.total_steps(), 2);
        assert_eq!(metrics.tokens_saved(), 8);
        // Overall: 8/10 = 80%
        assert!((metrics.acceptance_rate() - 0.8).abs() < 1e-10);
        // Windowed: same as overall since window hasn't filled up
        assert!((metrics.windowed_acceptance_rate() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_window_eviction() {
        let mut metrics = SpecDecodeMetrics::new(2);

        metrics.record(5, 5); // 100%
        metrics.record(5, 0); // 0%
        metrics.record(5, 3); // 60%

        // Window should contain only the last 2 entries: [0%, 60%]
        // Windowed: 3/10 = 30%
        assert!((metrics.windowed_acceptance_rate() - 0.3).abs() < 1e-10);

        // Overall: 8/15 = 53.3%
        assert!((metrics.acceptance_rate() - 8.0 / 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_proposed() {
        let mut metrics = SpecDecodeMetrics::new(10);
        metrics.record(0, 0);
        assert_eq!(metrics.acceptance_rate(), 0.0);
        assert_eq!(metrics.windowed_acceptance_rate(), 0.0);
    }
}
