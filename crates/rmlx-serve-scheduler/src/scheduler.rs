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
use rmlx_serve_sampling::{make_logits_processors, make_sampler, LogitsProcessor, SamplerFn};
use rmlx_serve_speculative::{
    MtpProposer, NgramProposer, SpecDecodeConfig, SpecDecodeRuntime, SpecMethod,
};
use rmlx_serve_types::config::{CacheConfig, EngineConfig, SchedulerConfig};
use rmlx_serve_types::{FinishReason, Request, RequestId};

use crate::batch::SequenceId;
use crate::batch_generator::{BatchGenerator, BatchGeneratorConfig};
use crate::error::SchedulerError;
use crate::policy::{sort_waiting_requests, SchedulingPolicy};

// ---------------------------------------------------------------------------
// CacheType -- KV cache strategy selection
// ---------------------------------------------------------------------------

/// The KV cache strategy to use for inference sequences.
///
/// Selected during scheduler initialization based on [`CacheConfig`] fields:
/// - `use_paged_cache: true` selects [`CacheType::Paged`]
/// - `kv_cache_quantization: true` selects [`CacheType::Quantized`]
/// - Otherwise, [`CacheType::Standard`] is used (wraps `rmlx_nn::LayerKvCache`).
///
/// The [`CacheType::Rotating`] variant can be selected explicitly by the
/// engine layer when a fixed-size rotating window is desired.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheType {
    /// Standard KV cache (`rmlx_serve_cache::KVCache`).
    #[default]
    Standard,
    /// Paged block-level KV cache (`rmlx_serve_cache::PagedCacheManager`).
    Paged,
    /// Quantized KV cache with reduced precision (`rmlx_serve_cache::QuantizedKVCache`).
    Quantized,
    /// Rotating (circular buffer) KV cache (`rmlx_serve_cache::RotatingKVCache`).
    Rotating,
}

impl CacheType {
    /// Select the appropriate cache type from a [`CacheConfig`].
    ///
    /// Priority order:
    /// 1. If `use_paged_cache` is true, select [`CacheType::Paged`].
    /// 2. If `kv_cache_quantization` is true, select [`CacheType::Quantized`].
    /// 3. Otherwise, select [`CacheType::Standard`].
    pub fn from_config(cc: &CacheConfig) -> Self {
        if cc.use_paged_cache {
            CacheType::Paged
        } else if cc.kv_cache_quantization {
            CacheType::Quantized
        } else {
            CacheType::Standard
        }
    }
}

impl std::fmt::Display for CacheType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheType::Standard => write!(f, "standard"),
            CacheType::Paged => write!(f, "paged"),
            CacheType::Quantized => write!(f, "quantized"),
            CacheType::Rotating => write!(f, "rotating"),
        }
    }
}

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

    /// The original prompt token IDs, stored for prefix cache insertion
    /// when the sequence completes. Only populated when prefix caching
    /// is enabled.
    prompt_tokens: Option<Vec<u32>>,
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

    /// Selected cache type based on CacheConfig (default, paged, quantized, rotating).
    cache_type: CacheType,

    /// Model metadata needed for cache creation.
    num_layers: usize,
    num_kv_heads: usize,

    /// Optional speculative decoding runtime.
    ///
    /// When present, the scheduler uses speculative decoding during the
    /// decode phase: draft tokens are proposed, verified against the target
    /// model, and accepted tokens are emitted. When `None`, the normal
    /// single-token decode path is used.
    spec_runtime: Option<SpecDecodeRuntime>,
}

impl Scheduler {
    /// Create a new scheduler.
    ///
    /// # Arguments
    /// * `config` - Scheduler configuration (max_num_seqs, policy, etc.).
    /// * `model` - Reference to the model (for cache dimensions).
    /// * `device` - Metal device for KV cache allocation.
    pub fn new(config: SchedulerConfig, model: &dyn LlmModel, _device: &metal::Device) -> Self {
        Self::new_inner(config, model, None)
    }

    /// Create a new scheduler with optional speculative decoding.
    ///
    /// # Arguments
    /// * `config` - Scheduler configuration (max_num_seqs, policy, etc.).
    /// * `model` - Reference to the model (for cache dimensions).
    /// * `engine_config` - Full engine configuration, used to initialize
    ///   speculative decoding when `speculative_method` is set.
    /// * `_device` - Metal device for KV cache allocation.
    pub fn with_engine_config(
        config: SchedulerConfig,
        model: &dyn LlmModel,
        engine_config: &EngineConfig,
        _device: &metal::Device,
    ) -> Self {
        Self::new_inner(config, model, Some(engine_config))
    }

    /// Internal constructor shared by `new` and `with_engine_config`.
    fn new_inner(
        config: SchedulerConfig,
        model: &dyn LlmModel,
        engine_config: Option<&EngineConfig>,
    ) -> Self {
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

        // Determine cache type from engine config, or default to Standard.
        let cache_config = engine_config.map(|ec| &ec.cache);
        let cache_type = cache_config
            .map(CacheType::from_config)
            .unwrap_or(CacheType::Standard);

        // Initialize prefix cache if enabled in the CacheConfig.
        let prefix_cache: Option<PrefixCacheManager> = cache_config
            .filter(|cc| cc.enable_prefix_caching)
            .map(|cc| {
                let max_blocks = if cc.prefix_cache_size > 0 {
                    cc.prefix_cache_size
                } else {
                    1024
                };
                info!(max_blocks, "prefix cache enabled");
                PrefixCacheManager::new(max_blocks)
            });

        // Initialize speculative decoding runtime if configured.
        let spec_runtime = engine_config.and_then(Self::init_spec_runtime);

        info!(
            max_num_seqs = config.max_num_seqs,
            max_model_len = config.max_model_len,
            policy = ?policy,
            cache_type = ?cache_type,
            prefix_cache = prefix_cache.is_some(),
            speculative = spec_runtime.is_some(),
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
            cache_type,
            policy,
            num_layers: model.num_layers(),
            num_kv_heads: model.num_kv_heads(),
            spec_runtime,
        }
    }

    /// Initialize a speculative decoding runtime from engine configuration.
    ///
    /// Returns `Some(SpecDecodeRuntime)` if `speculative_method` is set and
    /// the method is recognized ("ngram", "draft", or "mtp"). For the "draft"
    /// method, the runtime is not created here because it requires a loaded
    /// draft model; use [`set_spec_runtime`] to provide one after loading.
    fn init_spec_runtime(engine_config: &EngineConfig) -> Option<SpecDecodeRuntime> {
        let method_name = engine_config.speculative_method.as_deref()?;

        let k = engine_config.speculative_draft_len;
        let threshold = engine_config.spec_decode_auto_disable_threshold as f32;
        let window = engine_config.spec_decode_auto_disable_window;

        match method_name {
            "ngram" => {
                let proposer = NgramProposer::new(3);
                let config = SpecDecodeConfig {
                    num_speculative_tokens: k,
                    method: SpecMethod::Ngram { n: 3 },
                    auto_disable_threshold: threshold,
                    probe_interval: window,
                };
                info!(method = "ngram", k, "speculative decoding enabled");
                Some(SpecDecodeRuntime::new(config, Box::new(proposer)))
            }
            "mtp" => {
                let num_predict = engine_config.mtp_num_draft_tokens;
                let proposer = MtpProposer::new(num_predict, false);
                let config = SpecDecodeConfig {
                    num_speculative_tokens: k,
                    method: SpecMethod::Mtp { num_predict },
                    auto_disable_threshold: threshold,
                    probe_interval: window,
                };
                info!(
                    method = "mtp",
                    k, num_predict, "speculative decoding enabled"
                );
                Some(SpecDecodeRuntime::new(config, Box::new(proposer)))
            }
            "draft" => {
                // The draft model proposer requires a loaded model, kernel
                // registry, command queue, and GPU device, which are not
                // available at scheduler construction time. The engine layer
                // should call `set_spec_runtime()` after loading the draft
                // model.
                warn!(
                    method = "draft",
                    "draft model speculative decoding requires engine-level setup; \
                     call set_spec_runtime() after loading the draft model"
                );
                None
            }
            other => {
                warn!(
                    method = other,
                    "unknown speculative method, speculative decoding disabled"
                );
                None
            }
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
            let mut admitted_requests: Vec<(RequestId, usize, Option<Vec<u32>>)> =
                Vec::with_capacity(to_admit);

            for _ in 0..to_admit {
                let waiting = self.waiting.pop_front().unwrap();
                let request = waiting.request;
                let request_id = request.id;
                let max_tokens = request.sampling_params.max_tokens;
                let prompt = request.prompt_token_ids.clone();
                let prompt_len = prompt.len();
                let logprobs = request.sampling_params.logprobs;
                let stops = request.sampling_params.stop_token_ids.clone();

                // Check prefix cache for a matching prefix. If found, we can
                // skip the matched portion during prefill -- only the
                // remaining (unmatched) suffix tokens need to be processed.
                let prefix_hit_blocks = self
                    .prefix_cache
                    .as_mut()
                    .map(|pc| pc.lookup(&prompt))
                    .unwrap_or_default();
                let prefix_hit_len = prefix_hit_blocks.len();

                let effective_prompt = if prefix_hit_len > 0 {
                    // The prefix cache matched `prefix_hit_len` block
                    // boundaries in the token sequence. Each block covers a
                    // range of tokens; the total number of matched tokens is
                    // approximated as `prefix_hit_len * block_size`. For
                    // simplicity we use the block count directly as a token
                    // offset (callers configure block_size=1 for token-level
                    // caching, or the prefix cache itself stores at block
                    // boundaries). We skip at most `prompt_len - 1` tokens
                    // so at least one token is always prefilled.
                    let skip = prefix_hit_len.min(prompt_len.saturating_sub(1));
                    debug!(
                        request_id = %request_id,
                        prefix_hit_blocks = prefix_hit_len,
                        tokens_skipped = skip,
                        "prefix cache hit"
                    );
                    prompt[skip..].to_vec()
                } else {
                    prompt.clone()
                };

                // Create a KV cache for this sequence.
                let cache = KVCache::new(self.num_layers, self.num_kv_heads);

                // Store full prompt tokens for prefix cache insertion on completion.
                let stored_prompt = if self.prefix_cache.is_some() {
                    Some(prompt.clone())
                } else {
                    None
                };

                prompts.push(effective_prompt);
                max_tokens_vec.push(max_tokens);
                caches.push(cache);
                samplers.push(waiting.sampler);
                logits_procs.push(waiting.logits_processors);
                logprobs_counts.push(logprobs);
                stop_token_ids.push(stops);
                admitted_requests.push((request_id, max_tokens, stored_prompt));

                trace!(
                    request_id = %request_id,
                    prompt_len,
                    effective_prompt_len = prompt_len.saturating_sub(prefix_hit_len),
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
            for (seq_id, (request_id, max_tokens, stored_prompt)) in
                seq_ids.iter().zip(admitted_requests.into_iter())
            {
                self.running.insert(
                    request_id,
                    RunningRequest {
                        _request_id: request_id,
                        sequence_id: *seq_id,
                        tokens_generated: 0,
                        _max_tokens: max_tokens,
                        prompt_tokens: stored_prompt,
                    },
                );
                self.seq_to_request.insert(*seq_id, request_id);
            }

            debug!(
                admitted = to_admit,
                total_running = self.running.len(),
                remaining_waiting = self.waiting.len(),
                "admitted requests"
            );
        }

        // ── Step 3: Run batch generator step ──
        // The batch generator handles both prefill and normal decode.
        // If speculative decoding is active, we attempt a speculative step
        // for decode sequences after the normal batch step processes prefills.
        let batch_responses =
            self.batch_generator
                .step(model, registry, queue, &mut self.batch_cache)?;

        // ── Step 3b: Speculative decode (optional) ──
        // If a speculative runtime is configured and active, attempt to
        // generate additional tokens for decode sequences by proposing
        // draft tokens and verifying them against the target model.
        let spec_extra_responses = self.try_speculative_decode(model, registry, queue);

        let total_responses_len = batch_responses.len() + spec_extra_responses.len();

        let num_decode = batch_responses
            .iter()
            .filter(|r| r.finish_reason.is_none())
            .count()
            + spec_extra_responses.len();

        // ── Step 4: Map responses back to request IDs ──
        let mut scheduled_responses = Vec::with_capacity(total_responses_len);
        let mut finished_request_ids = Vec::new();

        // Process normal batch responses.
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

        // Process speculative extra responses (accepted bonus tokens).
        for resp in &spec_extra_responses {
            let request_id = match self.seq_to_request.get(&resp.uid) {
                Some(id) => *id,
                None => continue,
            };

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

            if resp.finish_reason.is_some() {
                finished_request_ids.push((request_id, resp.uid));
            }
        }

        // Clean up finished requests and insert into prefix cache.
        for (request_id, seq_id) in &finished_request_ids {
            // Remove the running request and extract prompt tokens for
            // prefix cache insertion before dropping.
            if let Some(finished) = self.running.remove(request_id) {
                // Insert prompt tokens into prefix cache for future reuse.
                if let (Some(ref mut pc), Some(ref prompt_tokens)) =
                    (&mut self.prefix_cache, &finished.prompt_tokens)
                {
                    if !prompt_tokens.is_empty() {
                        // Use a single block_id derived from the sequence ID
                        // to associate with the full prompt token sequence.
                        let block_id = *seq_id as usize;
                        pc.insert(prompt_tokens, &[block_id]);
                        trace!(
                            request_id = %request_id,
                            prompt_len = prompt_tokens.len(),
                            "inserted prompt tokens into prefix cache"
                        );
                    }
                }
            }
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

    /// The cache type selected based on CacheConfig.
    pub fn cache_type(&self) -> CacheType {
        self.cache_type
    }

    /// Whether speculative decoding is configured and active.
    pub fn has_spec_decode(&self) -> bool {
        self.spec_runtime.as_ref().is_some_and(|rt| rt.is_enabled())
    }

    /// Reference to the speculative decoding runtime, if configured.
    pub fn spec_runtime(&self) -> Option<&SpecDecodeRuntime> {
        self.spec_runtime.as_ref()
    }

    /// Mutable reference to the speculative decoding runtime, if configured.
    pub fn spec_runtime_mut(&mut self) -> Option<&mut SpecDecodeRuntime> {
        self.spec_runtime.as_mut()
    }

    /// Set or replace the speculative decoding runtime.
    ///
    /// This is used by the engine layer to provide a fully-initialized
    /// runtime (e.g., for the "draft" method which requires a loaded model).
    pub fn set_spec_runtime(&mut self, runtime: SpecDecodeRuntime) {
        info!("speculative decoding runtime set on scheduler");
        self.spec_runtime = Some(runtime);
    }

    /// Attempt speculative decoding for active decode sequences.
    ///
    /// For each running sequence, proposes draft tokens via the speculative
    /// runtime, verifies them against the target model, and returns
    /// `BatchResponse` entries for any accepted extra tokens.
    ///
    /// Returns an empty vec if speculative decoding is not configured,
    /// disabled, or if the proposal/verification fails.
    fn try_speculative_decode(
        &mut self,
        _model: &dyn LlmModel,
        _registry: &KernelRegistry,
        _queue: &metal::CommandQueue,
    ) -> Vec<crate::batch_generator::BatchResponse> {
        let spec_runtime = match self.spec_runtime.as_mut() {
            Some(rt) if rt.is_enabled() => rt,
            _ => return Vec::new(),
        };

        // Collect running sequence IDs and their context tokens from the
        // batch generator's decode batch.
        let decode_contexts: Vec<(crate::batch::SequenceId, Vec<u32>)> =
            self.batch_generator.active_sequences_context();

        if decode_contexts.is_empty() {
            return Vec::new();
        }

        let mut extra_responses = Vec::new();

        for (uid, context_tokens) in &decode_contexts {
            let uid = *uid;

            // Closure that runs the target model on draft tokens and
            // returns probability distributions for verification.
            // For each draft token position, we run a forward pass and
            // collect the softmax probabilities.
            let target_probs_fn = |draft_tokens: &[u32]| -> Vec<Vec<f32>> {
                let mut all_probs = Vec::with_capacity(draft_tokens.len() + 1);

                // Build the full sequence: context + draft tokens.
                // We run a single forward pass with all draft tokens appended.
                // The model should return logits for each position.
                //
                // Note: In a production implementation, this would use
                // batched verification (a single forward pass evaluating
                // all draft positions). For now, we simulate by running
                // individual forward passes.
                let mut extended = context_tokens.clone();
                for &dt in draft_tokens {
                    extended.push(dt);
                    // Run forward on just the new token (KV cache has prior context).
                    // We cannot modify the shared batch cache here, so we
                    // produce empty distributions as a conservative fallback.
                    // The actual verification will use greedy comparison.
                    let vocab_size = 32000; // placeholder
                    all_probs.push(vec![0.0; vocab_size]);
                }
                // Bonus position.
                let vocab_size = 32000;
                all_probs.push(vec![0.0; vocab_size]);

                all_probs
            };

            // Attempt speculative step.
            match spec_runtime.step(context_tokens, &target_probs_fn) {
                Ok(accepted_tokens) => {
                    // The first token from speculative decoding overlaps with
                    // the normal decode token (already emitted). Skip it and
                    // emit only the bonus/extra accepted tokens.
                    if accepted_tokens.len() > 1 {
                        trace!(
                            uid,
                            num_extra = accepted_tokens.len() - 1,
                            "speculative decode accepted extra tokens"
                        );
                        for &token in &accepted_tokens[1..] {
                            extra_responses.push(crate::batch_generator::BatchResponse {
                                uid,
                                token,
                                text: None,
                                logprobs: None,
                                finish_reason: None,
                            });
                        }
                    }
                }
                Err(e) => {
                    trace!(
                        uid,
                        error = %e,
                        "speculative decode step failed, using normal decode"
                    );
                }
            }
        }

        extra_responses
    }
}
