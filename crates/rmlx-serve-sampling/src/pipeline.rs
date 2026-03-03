//! Pipeline composition: builds a sampler closure and logit processor chain
//! from [`SamplingParams`].

use std::sync::Mutex;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rmlx_serve_types::SamplingParams;

use crate::processors::*;
use crate::sampler;

/// A boxed sampler closure: takes a logits slice and returns the sampled token id.
pub type SamplerFn = Box<dyn Fn(&[f32]) -> u32 + Send>;

/// Construct a sampling closure from the given parameters.
///
/// The returned closure accepts a logit slice (already copied from GPU to CPU)
/// and returns the sampled token id. It is `Send` so it can be moved to worker
/// threads.
///
/// Internally the closure:
/// 1. Clones the logits to a mutable buffer.
/// 2. Applies top-p (nucleus) filtering (if `top_p < 1.0`).
/// 3. Applies min-p filtering (if `min_p > 0.0`).
/// 4. Applies XTC filtering (if `xtc_probability > 0.0` and `xtc_threshold > 0.0`).
/// 5. Applies top-k filtering (if `top_k > 0`).
/// 6. Applies temperature scaling.
/// 7. Samples: greedy if temperature ~= 0, otherwise categorical.
pub fn make_sampler(params: &SamplingParams) -> SamplerFn {
    let temperature = params.temperature;
    let top_k = params.top_k;
    let top_p = params.top_p;
    let min_p = params.min_p;
    let xtc_probability = params.xtc_probability;
    let xtc_threshold = params.xtc_threshold;
    let stop_token_ids: std::collections::HashSet<u32> =
        params.stop_token_ids.iter().copied().collect();

    // Use greedy decoding when temperature is effectively zero.
    if temperature < 1e-6 {
        return Box::new(move |logits: &[f32]| sampler::greedy(logits));
    }

    // Build the RNG. If a seed is provided, use it for reproducibility.
    let rng: Mutex<StdRng> = match params.seed {
        Some(seed) => Mutex::new(StdRng::seed_from_u64(seed)),
        None => Mutex::new(StdRng::from_entropy()),
    };

    Box::new(move |logits: &[f32]| {
        let mut buf = logits.to_vec();

        // 1. Top-p (nucleus) filtering.
        if top_p < 1.0 {
            let topp_proc = TopPProcessor { p: top_p };
            LogitsProcessor::process(&topp_proc, &mut buf, &[]);
        }

        // 2. Min-p filtering.
        if min_p > 0.0 {
            let minp_proc = MinPProcessor::new(min_p);
            LogitsProcessor::process(&minp_proc, &mut buf, &[]);
        }

        // 3. XTC filtering.
        if xtc_probability > 0.0 && xtc_threshold > 0.0 {
            let xtc_proc = XtcProcessor {
                probability: xtc_probability,
                threshold: xtc_threshold,
                excluded_token_ids: stop_token_ids.clone(),
            };
            LogitsProcessor::process(&xtc_proc, &mut buf, &[]);
        }

        // 4. Top-k filtering.
        if top_k > 0 {
            let topk_proc = TopKProcessor { k: top_k };
            LogitsProcessor::process(&topk_proc, &mut buf, &[]);
        }

        // 5. Temperature scaling (applied after all filters, before sampling).
        let temp_proc = TemperatureProcessor { temperature };
        LogitsProcessor::process(&temp_proc, &mut buf, &[]);

        // 6. Categorical sampling.
        let mut rng_guard = rng.lock().unwrap();
        sampler::categorical(&buf, &mut *rng_guard)
    })
}

/// Build the chain of logit processors that should be applied *before* the
/// sampler's own processing (temperature, top-k, top-p, etc.).
///
/// These processors handle things like logit bias, repetition penalty, and
/// frequency/presence penalties that depend on the generation context.
pub fn make_logits_processors(params: &SamplingParams) -> Vec<Box<dyn LogitsProcessor>> {
    let mut processors: Vec<Box<dyn LogitsProcessor>> = Vec::new();

    // Logit bias.
    if !params.logit_bias.is_empty() {
        processors.push(Box::new(LogitBiasProcessor {
            bias: params.logit_bias.clone(),
        }));
    }

    // Repetition penalty.
    if (params.repetition_penalty - 1.0).abs() > f32::EPSILON {
        processors.push(Box::new(RepetitionPenaltyProcessor {
            penalty: params.repetition_penalty,
            // Default context window for repetition penalty: 20 tokens.
            // This could be made configurable in the future.
            context_size: 20,
        }));
    }

    // Frequency and presence penalties.
    if params.frequency_penalty.abs() > f32::EPSILON || params.presence_penalty.abs() > f32::EPSILON
    {
        processors.push(Box::new(FrequencyPresencePenaltyProcessor {
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
        }));
    }

    processors
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_make_sampler_greedy() {
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let sample = make_sampler(&params);
        let logits = vec![1.0f32, 5.0, 3.0, 2.0];
        assert_eq!(sample(&logits), 1);
    }

    #[test]
    fn test_make_sampler_seeded_deterministic() {
        let params = SamplingParams {
            temperature: 0.8,
            seed: Some(42),
            ..Default::default()
        };
        let sample1 = make_sampler(&params);
        let sample2 = make_sampler(&params);
        let logits = vec![1.0f32, 1.0, 1.0, 1.0];
        // Same seed should produce same result.
        assert_eq!(sample1(&logits), sample2(&logits));
    }

    #[test]
    fn test_make_logits_processors_empty() {
        let params = SamplingParams::default();
        let procs = make_logits_processors(&params);
        assert!(procs.is_empty());
    }

    #[test]
    fn test_make_logits_processors_with_bias() {
        let params = SamplingParams {
            logit_bias: HashMap::from([(0, 1.0)]),
            ..Default::default()
        };
        let procs = make_logits_processors(&params);
        assert_eq!(procs.len(), 1);
    }

    #[test]
    fn test_make_logits_processors_all() {
        let params = SamplingParams {
            logit_bias: HashMap::from([(0, 1.0)]),
            repetition_penalty: 1.5,
            frequency_penalty: 0.5,
            presence_penalty: 0.3,
            ..Default::default()
        };
        let procs = make_logits_processors(&params);
        assert_eq!(procs.len(), 3);
    }
}
