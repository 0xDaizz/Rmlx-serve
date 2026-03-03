//! Scheduling policies for ordering waiting requests.

use std::collections::VecDeque;

use crate::scheduler::WaitingRequest;

/// Scheduling policy for ordering requests in the waiting queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First Come First Served -- requests are processed in arrival order.
    Fcfs,

    /// Priority-based -- requests are ordered by arrival time (earliest first),
    /// then by prompt length (shorter prompts first for faster turnover).
    Priority,
}

impl SchedulingPolicy {
    /// Parse a scheduling policy from a string (e.g., from config).
    pub fn from_str_config(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "priority" => SchedulingPolicy::Priority,
            _ => SchedulingPolicy::Fcfs,
        }
    }
}

/// Sort the waiting request queue according to the given policy.
///
/// For `Fcfs`, requests are already in FIFO order so no sorting is needed.
/// For `Priority`, requests are sorted by arrival time (earliest first),
/// breaking ties by prompt length (shorter first).
pub fn sort_waiting_requests(requests: &mut VecDeque<WaitingRequest>, policy: SchedulingPolicy) {
    match policy {
        SchedulingPolicy::Fcfs => {
            // Already in FIFO order from VecDeque push_back semantics.
        }
        SchedulingPolicy::Priority => {
            // Convert to vec, sort, convert back to deque.
            let mut vec: Vec<WaitingRequest> = requests.drain(..).collect();
            vec.sort_by(|a, b| {
                // Primary: arrival time (earliest first).
                let time_cmp = a
                    .request
                    .arrival_time
                    .partial_cmp(&b.request.arrival_time)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if time_cmp != std::cmp::Ordering::Equal {
                    return time_cmp;
                }
                // Secondary: prompt length (shorter first for faster turnover).
                a.request
                    .prompt_token_ids
                    .len()
                    .cmp(&b.request.prompt_token_ids.len())
            });
            requests.extend(vec);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmlx_serve_types::{Request, SamplingParams};

    fn make_waiting(arrival: f64, prompt_len: usize) -> WaitingRequest {
        let prompt = vec![0u32; prompt_len];
        let mut req = Request::new(prompt, SamplingParams::default(), arrival);
        req.arrival_time = arrival;
        WaitingRequest {
            request: req,
            sampler: Box::new(|logits: &[f32]| {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            }),
            logits_processors: vec![],
        }
    }

    #[test]
    fn test_fcfs_preserves_order() {
        let mut queue = VecDeque::new();
        queue.push_back(make_waiting(3.0, 10));
        queue.push_back(make_waiting(1.0, 5));
        queue.push_back(make_waiting(2.0, 20));

        sort_waiting_requests(&mut queue, SchedulingPolicy::Fcfs);

        // FCFS: order should be unchanged.
        assert!((queue[0].request.arrival_time - 3.0).abs() < f64::EPSILON);
        assert!((queue[1].request.arrival_time - 1.0).abs() < f64::EPSILON);
        assert!((queue[2].request.arrival_time - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_priority_sorts_by_arrival_time() {
        let mut queue = VecDeque::new();
        queue.push_back(make_waiting(3.0, 10));
        queue.push_back(make_waiting(1.0, 10));
        queue.push_back(make_waiting(2.0, 10));

        sort_waiting_requests(&mut queue, SchedulingPolicy::Priority);

        // Priority: earliest arrival first.
        assert!((queue[0].request.arrival_time - 1.0).abs() < f64::EPSILON);
        assert!((queue[1].request.arrival_time - 2.0).abs() < f64::EPSILON);
        assert!((queue[2].request.arrival_time - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_priority_breaks_ties_by_prompt_length() {
        let mut queue = VecDeque::new();
        queue.push_back(make_waiting(1.0, 100));
        queue.push_back(make_waiting(1.0, 10));
        queue.push_back(make_waiting(1.0, 50));

        sort_waiting_requests(&mut queue, SchedulingPolicy::Priority);

        // Same arrival time: shorter prompt first.
        assert_eq!(queue[0].request.prompt_token_ids.len(), 10);
        assert_eq!(queue[1].request.prompt_token_ids.len(), 50);
        assert_eq!(queue[2].request.prompt_token_ids.len(), 100);
    }

    #[test]
    fn test_from_str_config() {
        assert_eq!(
            SchedulingPolicy::from_str_config("fcfs"),
            SchedulingPolicy::Fcfs
        );
        assert_eq!(
            SchedulingPolicy::from_str_config("priority"),
            SchedulingPolicy::Priority
        );
        assert_eq!(
            SchedulingPolicy::from_str_config("PRIORITY"),
            SchedulingPolicy::Priority
        );
        assert_eq!(
            SchedulingPolicy::from_str_config("unknown"),
            SchedulingPolicy::Fcfs
        );
    }
}
