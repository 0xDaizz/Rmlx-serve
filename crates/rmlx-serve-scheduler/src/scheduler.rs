//! Scheduler: high-level request admission and lifecycle management.
//!
//! Ported from vllm-mlx's `Scheduler`. Manages the waiting queue,
//! running request set, and delegates actual token generation to the
//! [`BatchGenerator`]. The scheduler handles:
//!
//! - Request admission control (max_num_seqs, prompt length validation)
//! - Mapping between request IDs and sequence IDs
//! - Sampler and logits processor creation from sampling parameters
//! - Tracking request completion and finish reasons

use std::collections::{HashMap, VecDeque};

use tracing::{debug, info, trace, warn};

use rmlx_core::KernelRegistry;
use rmlx_serve_cache::{BatchKVCache, KVCache, PrefixCacheManager};
use rmlx_serve_models::LlmModel;
use rmlx_serve_sampling::{make_logits_processors, make_sampler, LogitsProcessor};
use rmlx_serve_types::config::SchedulerConfig;
use rmlx_serve_types::{FinishReason, Request, RequestId};

use crate::batch::SequenceId;
use crate::batch_generator::{BatchGenerator, BatchGeneratorConfig};
use crate::error::SchedulerError;
use crate::policy::{sort_waiting_requests, SchedulingPolicy};

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// The output from one scheduler step.
pub struct SchedulerOutput {
    /// Per-sequence responses produced in this step.
    pub responses: Vec<ScheduledResponse>,

    /// Number of sequences that were prefilled in this step.
    pub num_prefill: usize,

    /// Number of sequences that were decoded in this step.
    pub num_decode: usize,
}

/// A single-token response mapped back to a request ID.
pub struct ScheduledResponse {
    /// The originating request ID.
    pub request_id: RequestId,

    /// The batch-generator sequence ID.
    pub sequence_id: SequenceId,

    /// The generated token.
    pub token: u32,

    /// Optional top-k (token_id, log_prob) pairs.
    pub logprobs: Option<Vec<(u32, f32)>>,

    /// Set when this sequence has finished generating.
    pub finish_reason: Option<FinishReason>,
}

// ---------------------------------------------------------------------------
// Internal request tracking
// ---------------------------------------------------------------------------

/// A sampler function that takes logits and returns a token ID.
pub type SamplerFn = Box<dyn Fn(&[f32]) -> u32 + Send>;

/// A request waiting to be admitted into the batch generator.
pub struct WaitingRequest {
    /// The original request.
    pub request: Request,

    /// Pre-built sampler closure.
    pub sampler: SamplerFn,

    /// Pre-built logits processors.
    pub logits_processors: Vec<Box<dyn LogitsProcessor>>,
}

/// A request currently being processed by the batch generator.
struct RunningRequest {
    /// The original request ID.
    _request_id: RequestId,

    /// The sequence ID assigned by the batch generator.
    sequence_id: SequenceId,

    /// Number of tokens generated so far.
    tokens_generated: usize,

    /// Maximum tokens to generate.
    _max_tokens: usize,
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// High-level request scheduler for the inference engine.
///
/// Sits between the API layer (which submits `Request` objects) and the
/// `BatchGenerator` (which handles low-level batch prefill/decode). The
/// scheduler is responsible for:
///
/// 1. Validating and queuing incoming requests.
/// 2. Admission control: deciding which waiting requests to admit based
///    on capacity (max_num_seqs, memory).
/// 3. Creating KV caches, samplers, and logits processors from request params.
/// 4. Mapping batch generator sequence IDs back to request IDs.
/// 5. Tracking request completion and cleanup.
pub struct Scheduler {
    /// Scheduler configuration (limits, policy, etc.).
    config: SchedulerConfig,

    /// The batch generator that handles prefill and decode.
    batch_generator: BatchGenerator,

    /// KV caches for all active sequences, indexed by slot.
    batch_cache: BatchKVCache,

    /// Requests waiting to be admitted.
    waiting: VecDeque<WaitingRequest>,

    /// Requests currently running (mapped by request ID).
    running: HashMap<RequestId, RunningRequest>,

    /// Reverse mapping: sequence ID -> request ID.
    seq_to_request: HashMap<SequenceId, RequestId>,

    /// Optional prefix cache for KV cache reuse.
    prefix_cache: Option<PrefixCacheManager>,

    /// Scheduling policy.
    policy: SchedulingPolicy,

    /// Model metadata needed for cache creation.
    num_layers: usize,
    num_kv_heads: usize,
}

impl Scheduler {
    /// Create a new scheduler.
    ///
    /// # Arguments
    /// * `config` - Scheduler configuration (max_num_seqs, policy, etc.).
    /// * `model` - Reference to the model (for cache dimensions).
    /// * `device` - Metal device for KV cache allocation.
    pub fn new(config: SchedulerConfig, model: &dyn LlmModel, _device: &metal::Device) -> Self {
        let batch_config = BatchGeneratorConfig {
            prefill_batch_size: config.max_num_seqs.min(4),
            completion_batch_size: config.max_num_seqs,
            prefill_step_size: if config.enable_chunked_prefill {
                config.max_prefill_chunk_size
            } else {
                config.max_model_len
            },
        };

        let batch_generator = BatchGenerator::new(batch_config);

        // Create the batch KV cache with capacity for max_num_seqs.
        let batch_cache = BatchKVCache::new(config.max_num_seqs);

        let policy = SchedulingPolicy::from_str_config(&config.policy);

        // Prefix cache is not created by default here; the engine layer can
        // set one up via `prefix_cache_mut()` if `CacheConfig::enable_prefix_caching`
        // is true. The scheduler only manages the scheduling policy and batch
        // generation, not cache-level features.
        let prefix_cache: Option<PrefixCacheManager> = None;

        info!(
            max_num_seqs = config.max_num_seqs,
            max_model_len = config.max_model_len,
            policy = ?policy,
            "scheduler initialized"
        );

        Self {
            config,
            batch_generator,
            batch_cache,
            waiting: VecDeque::new(),
            running: HashMap::new(),
            seq_to_request: HashMap::new(),
            prefix_cache,
            policy,
            num_layers: model.num_layers(),
            num_kv_heads: model.num_kv_heads(),
        }
    }

    /// Add a request to the scheduler.
    ///
    /// The request is validated, and a sampler and logits processors are
    /// created from its sampling parameters. The request is then placed
    /// in the waiting queue.
    pub fn add_request(&mut self, request: Request) {
        let prompt_len = request.prompt_token_ids.len();

        // Validate prompt length.
        if prompt_len > self.config.max_prompt_len {
            warn!(
                request_id = %request.id,
                prompt_len,
                max = self.config.max_prompt_len,
                "request rejected: prompt too long"
            );
            return;
        }

        if prompt_len == 0 {
            warn!(
                request_id = %request.id,
                "request rejected: empty prompt"
            );
            return;
        }

        // Build sampler and logits processors from sampling params.
        let sampler = make_sampler(&request.sampling_params);
        let logits_processors = make_logits_processors(&request.sampling_params);

        debug!(
            request_id = %request.id,
            prompt_len,
            max_tokens = request.sampling_params.max_tokens,
            "request added to waiting queue"
        );

        self.waiting.push_back(WaitingRequest {
            request,
            sampler,
            logits_processors,
        });
    }

    /// Abort a request by its ID.
    ///
    /// Removes the request from the waiting queue or running set. If the
    /// request is running, its sequence is also removed from the batch
    /// generator and its KV cache slot is freed.
    pub fn abort_request(&mut self, request_id: &RequestId) {
        // Check waiting queue.
        if let Some(pos) = self
            .waiting
            .iter()
            .position(|w| w.request.id == *request_id)
        {
            self.waiting.remove(pos);
            debug!(request_id = %request_id, "aborted waiting request");
            return;
        }

        // Check running requests.
        if let Some(running) = self.running.remove(request_id) {
            let seq_id = running.sequence_id;
            self.seq_to_request.remove(&seq_id);
            self.batch_generator.remove(&[seq_id]);

            debug!(
                request_id = %request_id,
                sequence_id = seq_id,
                "aborted running request"
            );
        }
    }

    /// Run one scheduling step.
    ///
    /// This is the main entry point called by the engine on each iteration:
    ///
    /// 1. Sort waiting requests according to the scheduling policy.
    /// 2. Admit waiting requests up to `max_num_seqs` (create caches, samplers).
    /// 3. Call `batch_generator.step()` to process prefill and decode.
    /// 4. Map batch responses back to request IDs.
    /// 5. Handle finished sequences (update running set, emit finish reasons).
    /// 6. Return `SchedulerOutput`.
    pub fn step(
        &mut self,
        model: &dyn LlmModel,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<SchedulerOutput, SchedulerError> {
        // ── Step 1: Sort waiting requests ──
        sort_waiting_requests(&mut self.waiting, self.policy);

        // ── Step 2: Admit waiting requests ──
        let current_running = self.running.len();
        let can_admit = self.config.max_num_seqs.saturating_sub(current_running);
        let to_admit = can_admit.min(self.waiting.len());

        let num_prefill = to_admit;

        if to_admit > 0 {
            let mut prompts = Vec::with_capacity(to_admit);
            let mut max_tokens_vec = Vec::with_capacity(to_admit);
            let mut caches = Vec::with_capacity(to_admit);
            let mut samplers = Vec::with_capacity(to_admit);
            let mut logits_procs = Vec::with_capacity(to_admit);
            let mut logprobs_counts = Vec::with_capacity(to_admit);
            let mut stop_token_ids = Vec::with_capacity(to_admit);
            let mut admitted_requests: Vec<(RequestId, usize)> = Vec::with_capacity(to_admit);

            for _ in 0..to_admit {
                let waiting = self.waiting.pop_front().unwrap();
                let request = waiting.request;
                let request_id = request.id;
                let max_tokens = request.sampling_params.max_tokens;
                let prompt = request.prompt_token_ids.clone();
                let prompt_len = prompt.len();
                let logprobs = request.sampling_params.logprobs;
                let stops = request.sampling_params.stop_token_ids.clone();

                // Create a KV cache for this sequence.
                let cache = KVCache::new(self.num_layers, self.num_kv_heads);

                prompts.push(prompt);
                max_tokens_vec.push(max_tokens);
                caches.push(cache);
                samplers.push(waiting.sampler);
                logits_procs.push(waiting.logits_processors);
                logprobs_counts.push(logprobs);
                stop_token_ids.push(stops);
                admitted_requests.push((request_id, max_tokens));

                trace!(
                    request_id = %request_id,
                    prompt_len,
                    max_tokens,
                    "admitting request"
                );
            }

            // Insert into the batch generator.
            let seq_ids = self.batch_generator.insert(
                prompts,
                max_tokens_vec,
                caches,
                samplers,
                logits_procs,
                logprobs_counts,
                stop_token_ids,
            );

            // Track the admitted requests.
            for (seq_id, (request_id, max_tokens)) in seq_ids.iter().zip(admitted_requests.iter()) {
                self.running.insert(
                    *request_id,
                    RunningRequest {
                        _request_id: *request_id,
                        sequence_id: *seq_id,
                        tokens_generated: 0,
                        _max_tokens: *max_tokens,
                    },
                );
                self.seq_to_request.insert(*seq_id, *request_id);
            }

            debug!(
                admitted = to_admit,
                total_running = self.running.len(),
                remaining_waiting = self.waiting.len(),
                "admitted requests"
            );
        }

        // ── Step 3: Run batch generator step ──
        let batch_responses =
            self.batch_generator
                .step(model, registry, queue, &mut self.batch_cache)?;

        let num_decode = batch_responses
            .iter()
            .filter(|r| r.finish_reason.is_none())
            .count();

        // ── Step 4: Map responses back to request IDs ──
        let mut scheduled_responses = Vec::with_capacity(batch_responses.len());
        let mut finished_request_ids = Vec::new();

        for resp in &batch_responses {
            let request_id = match self.seq_to_request.get(&resp.uid) {
                Some(id) => *id,
                None => {
                    warn!(
                        sequence_id = resp.uid,
                        "batch response for unknown sequence"
                    );
                    continue;
                }
            };

            // Update running request state.
            if let Some(running) = self.running.get_mut(&request_id) {
                running.tokens_generated += 1;
            }

            scheduled_responses.push(ScheduledResponse {
                request_id,
                sequence_id: resp.uid,
                token: resp.token,
                logprobs: resp.logprobs.clone(),
                finish_reason: resp.finish_reason,
            });

            // ── Step 5: Handle finished sequences ──
            if resp.finish_reason.is_some() {
                finished_request_ids.push((request_id, resp.uid));
            }
        }

        // Clean up finished requests.
        for (request_id, seq_id) in &finished_request_ids {
            self.running.remove(request_id);
            self.seq_to_request.remove(seq_id);

            debug!(
                request_id = %request_id,
                sequence_id = seq_id,
                "request finished"
            );
        }

        // ── Step 6: Return output ──
        Ok(SchedulerOutput {
            responses: scheduled_responses,
            num_prefill,
            num_decode,
        })
    }

    /// Number of requests waiting to be admitted.
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Number of requests currently running.
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Whether the scheduler has no pending or running work.
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty() && self.batch_generator.is_idle()
    }

    /// Whether there is any pending work (waiting or running requests,
    /// or pending prefills in the batch generator).
    pub fn has_pending_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty() || !self.batch_generator.is_idle()
    }

    /// Reference to the batch generator's statistics.
    pub fn stats(&self) -> &crate::batch_generator::BatchStats {
        self.batch_generator.stats()
    }

    /// Reference to the prefix cache manager, if enabled.
    pub fn prefix_cache(&self) -> Option<&PrefixCacheManager> {
        self.prefix_cache.as_ref()
    }

    /// Mutable reference to the prefix cache manager, if enabled.
    pub fn prefix_cache_mut(&mut self) -> Option<&mut PrefixCacheManager> {
        self.prefix_cache.as_mut()
    }

    /// Set the prefix cache manager. Called by the engine during initialization
    /// when `CacheConfig::enable_prefix_caching` is true.
    pub fn set_prefix_cache(&mut self, cache: PrefixCacheManager) {
        self.prefix_cache = Some(cache);
    }

    /// Reference to the scheduler configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}
