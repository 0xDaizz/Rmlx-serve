//! BatchGenerator: low-level batch-oriented prefill and decode engine.
//!
//! Ported from mlx-lm's `BatchGenerator`. Manages a queue of sequences
//! waiting for prefill and a batch of sequences actively decoding.
//! Each call to `step()` prefills new sequences (up to limits) then
//! decodes one token for all active sequences.

use std::collections::VecDeque;

use tracing::{debug, trace, warn};

use rmlx_core::{Array, DType, KernelRegistry};
use rmlx_serve_cache::{BatchKVCache, KVCache};
use rmlx_serve_models::LlmModel;
use rmlx_serve_sampling::{top_logprobs, LogitsProcessor};
use rmlx_serve_types::FinishReason;

use crate::batch::{Batch, SequenceId, SequenceState};
use crate::error::SchedulerError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the batch generator.
#[derive(Clone, Debug)]
pub struct BatchGeneratorConfig {
    /// Maximum number of sequences to prefill in a single step.
    pub prefill_batch_size: usize,

    /// Maximum number of sequences in the decode batch at once.
    pub completion_batch_size: usize,

    /// Maximum number of tokens to process in a single prefill step
    /// (chunked prefill). If a prompt exceeds this, it is processed
    /// in multiple steps.
    pub prefill_step_size: usize,
}

impl Default for BatchGeneratorConfig {
    fn default() -> Self {
        Self {
            prefill_batch_size: 4,
            completion_batch_size: 256,
            prefill_step_size: 2048,
        }
    }
}

// ---------------------------------------------------------------------------
// Response and Stats
// ---------------------------------------------------------------------------

/// A single-token response from the batch generator for one sequence.
pub struct BatchResponse {
    /// The sequence id this response belongs to.
    pub uid: SequenceId,

    /// The token that was generated.
    pub token: u32,

    /// Optional text representation (set later by the engine after detokenization).
    pub text: Option<String>,

    /// Optional top-k (token_id, log_prob) pairs if logprobs were requested.
    pub logprobs: Option<Vec<(u32, f32)>>,

    /// Set when this sequence has finished generating.
    pub finish_reason: Option<FinishReason>,
}

/// Aggregate statistics for the batch generator.
#[derive(Clone, Debug)]
pub struct BatchStats {
    /// Total prompt tokens processed (prefilled).
    pub prompt_tokens: usize,

    /// Total generation tokens produced (decoded).
    pub generation_tokens: usize,

    /// Number of sequences currently in the decode batch.
    pub active_sequences: usize,

    /// Prompt processing throughput (tokens per second).
    pub prompt_tps: f64,

    /// Generation throughput (tokens per second).
    pub generation_tps: f64,
}

impl Default for BatchStats {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            generation_tokens: 0,
            active_sequences: 0,
            prompt_tps: 0.0,
            generation_tps: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill entry
// ---------------------------------------------------------------------------

/// An entry waiting to be prefilled before joining the decode batch.
struct PrefillEntry {
    /// The sequence identifier assigned by the batch generator.
    uid: SequenceId,

    /// Full prompt token ids.
    prompt_tokens: Vec<u32>,

    /// Maximum number of tokens to generate after prefill.
    max_tokens: usize,

    /// Pre-created KV cache for this sequence.
    cache: KVCache,

    /// The sampler closure.
    sampler: Box<dyn Fn(&[f32]) -> u32 + Send>,

    /// Context-dependent logits processors.
    logits_processors: Vec<Box<dyn LogitsProcessor>>,

    /// Number of top log-probabilities to return per token.
    logprobs_count: Option<usize>,

    /// Stop token ids that signal end of generation.
    stop_token_ids: Vec<u32>,
}

// ---------------------------------------------------------------------------
// BatchGenerator
// ---------------------------------------------------------------------------

/// Low-level batch-oriented generation engine.
///
/// Manages the lifecycle of sequences from prefill queue through active
/// decode batch to completion. The key insight is that prefill (processing
/// the full prompt) and decode (generating one token at a time) are
/// fundamentally different operations with different batch characteristics.
///
/// The `step()` method:
/// 1. Admits pending sequences from the prefill queue (up to limits).
/// 2. Runs model forward for each newly prefilled sequence.
/// 3. Runs model forward for all active decode sequences.
/// 4. Applies logits processors and samplers.
/// 5. Checks stop conditions.
/// 6. Returns responses for all sequences that generated a token.
pub struct BatchGenerator {
    /// Sequences waiting to be prefilled (FIFO).
    prefill_queue: VecDeque<PrefillEntry>,

    /// Sequences actively being decoded (one token per step).
    decode_batch: Batch,

    /// Configuration limits.
    config: BatchGeneratorConfig,

    /// Monotonically increasing uid counter.
    next_uid: SequenceId,

    /// Cumulative statistics.
    stats: BatchStats,

    /// When the generator was created (for throughput calculations).
    start_time: std::time::Instant,
}

impl BatchGenerator {
    /// Create a new batch generator with the given configuration.
    pub fn new(config: BatchGeneratorConfig) -> Self {
        Self {
            prefill_queue: VecDeque::new(),
            decode_batch: Batch::new(),
            config,
            next_uid: 0,
            stats: BatchStats::default(),
            start_time: std::time::Instant::now(),
        }
    }

    /// Insert new sequences for generation.
    ///
    /// Each call provides parallel vectors of prompt tokens, generation limits,
    /// pre-allocated KV caches, samplers, logits processors, and logprobs counts.
    /// Returns the assigned sequence ids.
    pub fn insert(
        &mut self,
        prompts: Vec<Vec<u32>>,
        max_tokens: Vec<usize>,
        caches: Vec<KVCache>,
        samplers: Vec<Box<dyn Fn(&[f32]) -> u32 + Send>>,
        logits_processors: Vec<Vec<Box<dyn LogitsProcessor>>>,
        logprobs_counts: Vec<Option<usize>>,
        stop_token_ids: Vec<Vec<u32>>,
    ) -> Vec<SequenceId> {
        let n = prompts.len();
        debug_assert_eq!(n, max_tokens.len());
        debug_assert_eq!(n, caches.len());
        debug_assert_eq!(n, samplers.len());
        debug_assert_eq!(n, logits_processors.len());
        debug_assert_eq!(n, logprobs_counts.len());
        debug_assert_eq!(n, stop_token_ids.len());

        let mut uids = Vec::with_capacity(n);

        // Consume all inputs in parallel via into_iter().
        let prompts_iter = prompts.into_iter();
        let max_iter = max_tokens.into_iter();
        let cache_iter = caches.into_iter();
        let sampler_iter = samplers.into_iter();
        let lp_iter = logits_processors.into_iter();
        let logprobs_iter = logprobs_counts.into_iter();
        let stop_iter = stop_token_ids.into_iter();

        for (((((prompt, mt), cache), sampler), lps), (lpc, stop)) in prompts_iter
            .zip(max_iter)
            .zip(cache_iter)
            .zip(sampler_iter)
            .zip(lp_iter)
            .zip(logprobs_iter.zip(stop_iter))
        {
            let uid = self.next_uid;
            self.next_uid += 1;

            self.prefill_queue.push_back(PrefillEntry {
                uid,
                prompt_tokens: prompt,
                max_tokens: mt,
                cache,
                sampler,
                logits_processors: lps,
                logprobs_count: lpc,
                stop_token_ids: stop,
            });

            uids.push(uid);
        }

        debug!(
            count = uids.len(),
            pending = self.prefill_queue.len(),
            "inserted sequences into prefill queue"
        );

        uids
    }

    /// Remove completed or aborted sequences by uid.
    ///
    /// Searches both the prefill queue and the decode batch. Returns the
    /// KV cache for each uid if found (so the caller can reclaim memory).
    pub fn remove(&mut self, uids: &[SequenceId]) -> Vec<Option<KVCache>> {
        let mut results = Vec::with_capacity(uids.len());

        for &uid in uids {
            // Check prefill queue first.
            if let Some(pos) = self.prefill_queue.iter().position(|e| e.uid == uid) {
                let entry = self.prefill_queue.remove(pos).unwrap();
                results.push(Some(entry.cache));
                continue;
            }

            // Check decode batch -- we can't recover the cache directly from
            // the batch (it's in BatchKVCache managed by the scheduler), so
            // just remove the sequence state.
            if self.decode_batch.remove(uid).is_some() {
                results.push(None);
            } else {
                results.push(None);
            }
        }

        results
    }

    /// Process one generation step.
    ///
    /// This is the core loop:
    /// 1. Move pending sequences from the prefill queue into the decode batch,
    ///    running the model forward pass on their full prompts (prefill).
    /// 2. For all active decode sequences, run the model forward pass with
    ///    their current (single) token to generate the next token.
    /// 3. Apply logits processors and samplers to select the next token.
    /// 4. Check stop conditions (max_tokens, stop token ids).
    /// 5. Return a `BatchResponse` for each sequence that produced a token.
    pub fn step(
        &mut self,
        model: &dyn LlmModel,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
        batch_cache: &mut BatchKVCache,
    ) -> Result<Vec<BatchResponse>, SchedulerError> {
        let mut responses = Vec::new();

        // ── Phase 1: Prefill ──
        // Admit sequences from the prefill queue up to batch limits.
        let available_slots = self
            .config
            .completion_batch_size
            .saturating_sub(self.decode_batch.len());
        let prefill_count = available_slots.min(self.config.prefill_batch_size);

        let mut newly_prefilled: Vec<SequenceId> = Vec::new();

        for _ in 0..prefill_count {
            if self.prefill_queue.is_empty() {
                break;
            }

            let entry = self.prefill_queue.pop_front().unwrap();
            let uid = entry.uid;
            let prompt_len = entry.prompt_tokens.len();

            trace!(uid, prompt_len, "prefilling sequence");

            // Find a free slot in the batch cache for this sequence.
            let batch_idx = self.find_free_cache_slot(batch_cache);

            // Insert the KV cache into the batch cache.
            batch_cache.insert(batch_idx, entry.cache);

            // Run model forward on the full prompt to populate the KV cache.
            let cache_slot = batch_cache.get_mut(batch_idx);
            let cache_layers = cache_slot.map(|c| &mut c.inner);

            let logits_array = model
                .forward(&entry.prompt_tokens, cache_layers, registry, queue)
                .map_err(|e| SchedulerError::ModelError(e.to_string()))?;

            // Extract logits as f32 on CPU. The model returns [1, vocab_size].
            let logits_f32 = self.extract_logits(&logits_array)?;

            // Apply logits processors.
            let mut logits_buf = logits_f32;
            for proc in &entry.logits_processors {
                proc.process(&mut logits_buf, &entry.prompt_tokens);
            }

            // Compute top logprobs if requested.
            let top_lps = entry
                .logprobs_count
                .map(|k| top_logprobs(&logits_buf, k));

            // Sample the first generated token.
            let first_token = (entry.sampler)(&logits_buf);

            // Track prompt tokens.
            self.stats.prompt_tokens += prompt_len;

            // Check if the first token is a stop token.
            let is_stop = entry.stop_token_ids.contains(&first_token);
            let finish = if is_stop {
                Some(FinishReason::Stop)
            } else {
                None
            };

            // Create sequence state and add to decode batch.
            let state = SequenceState {
                uid,
                token_ids: vec![first_token],
                current_token: first_token,
                max_tokens: entry.max_tokens,
                num_generated: 1,
                sampler: entry.sampler,
                logits_processors: entry.logits_processors,
                logprobs_count: entry.logprobs_count,
                finish_reason: finish,
                stop_token_ids: entry.stop_token_ids,
            };

            self.decode_batch.add(state);
            newly_prefilled.push(uid);

            // Emit the first token response.
            responses.push(BatchResponse {
                uid,
                token: first_token,
                text: None,
                logprobs: top_lps,
                finish_reason: finish,
            });

            self.stats.generation_tokens += 1;
        }

        // ── Phase 2: Decode ──
        // Process all active sequences that were NOT just prefilled and are
        // not finished.
        let decode_uids: Vec<SequenceId> = self
            .decode_batch
            .iter()
            .filter(|s| s.finish_reason.is_none() && !newly_prefilled.contains(&s.uid))
            .map(|s| s.uid)
            .collect();

        if !decode_uids.is_empty() {
            // For each active decode sequence, run model forward with its
            // current token and its cache.
            for &uid in &decode_uids {
                // Find the sequence in the batch.
                let seq = match self.decode_batch.iter().find(|s| s.uid == uid) {
                    Some(s) => s,
                    None => continue,
                };

                let current_token = seq.current_token;

                // Find its cache slot in the batch cache. The slot index
                // matches the position we assigned during prefill. We search
                // for a cache that has the right sequence length.
                let batch_idx = self.find_cache_slot_for_uid(uid, batch_cache);

                if let Some(idx) = batch_idx {
                    let cache_slot = batch_cache.get_mut(idx);
                    let cache_layers = cache_slot.map(|c| &mut c.inner);

                    let logits_array = model
                        .forward(&[current_token], cache_layers, registry, queue)
                        .map_err(|e| SchedulerError::ModelError(e.to_string()))?;

                    let logits_f32 = self.extract_logits(&logits_array)?;

                    // Get the mutable sequence state to apply processors.
                    let seq_mut = match self.decode_batch.iter_mut().find(|s| s.uid == uid) {
                        Some(s) => s,
                        None => continue,
                    };

                    // Apply logits processors with context of generated tokens.
                    let mut logits_buf = logits_f32;
                    for proc in &seq_mut.logits_processors {
                        proc.process(&mut logits_buf, &seq_mut.token_ids);
                    }

                    // Compute top logprobs if requested.
                    let top_lps = seq_mut
                        .logprobs_count
                        .map(|k| top_logprobs(&logits_buf, k));

                    // Sample next token.
                    let next_token = (seq_mut.sampler)(&logits_buf);

                    // Update sequence state.
                    seq_mut.token_ids.push(next_token);
                    seq_mut.current_token = next_token;
                    seq_mut.num_generated += 1;

                    // Check stop conditions.
                    let is_stop_token = seq_mut.stop_token_ids.contains(&next_token);
                    let hit_max = seq_mut.num_generated >= seq_mut.max_tokens;

                    let finish = if is_stop_token {
                        Some(FinishReason::Stop)
                    } else if hit_max {
                        Some(FinishReason::Length)
                    } else {
                        None
                    };

                    if finish.is_some() {
                        seq_mut.finish_reason = finish;
                    }

                    responses.push(BatchResponse {
                        uid,
                        token: next_token,
                        text: None,
                        logprobs: top_lps,
                        finish_reason: finish,
                    });

                    self.stats.generation_tokens += 1;
                } else {
                    warn!(uid, "no cache slot found for decode sequence");
                }
            }
        }

        // ── Phase 3: Clean up finished sequences ──
        let finished_uids: Vec<SequenceId> = self
            .decode_batch
            .iter()
            .filter(|s| s.finish_reason.is_some())
            .map(|s| s.uid)
            .collect();

        if !finished_uids.is_empty() {
            let keep: Vec<bool> = self
                .decode_batch
                .iter()
                .map(|s| s.finish_reason.is_none())
                .collect();
            self.decode_batch.filter(&keep);

            debug!(
                finished = finished_uids.len(),
                remaining = self.decode_batch.len(),
                "removed finished sequences from decode batch"
            );
        }

        // Update stats.
        self.stats.active_sequences = self.decode_batch.len();
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.stats.prompt_tps = self.stats.prompt_tokens as f64 / elapsed;
            self.stats.generation_tps = self.stats.generation_tokens as f64 / elapsed;
        }

        Ok(responses)
    }

    /// Reference to the current statistics.
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Number of sequences currently in the decode batch.
    pub fn active_count(&self) -> usize {
        self.decode_batch.len()
    }

    /// Number of sequences waiting to be prefilled.
    pub fn pending_count(&self) -> usize {
        self.prefill_queue.len()
    }

    /// Whether there is no pending or active work.
    pub fn is_idle(&self) -> bool {
        self.prefill_queue.is_empty() && self.decode_batch.is_empty()
    }

    // ── Internal helpers ──

    /// Extract logits from a model output Array as f32 on CPU.
    ///
    /// The model returns logits with shape `[1, vocab_size]`. We extract
    /// them as a flat `Vec<f32>`.
    fn extract_logits(&self, logits_array: &Array) -> Result<Vec<f32>, SchedulerError> {
        // The model output may be Float16 or Float32. We need f32 for sampling.
        let dtype = logits_array.dtype();
        match dtype {
            DType::Float32 => Ok(logits_array.to_vec_checked::<f32>()),
            DType::Float16 => {
                // Read raw bytes and convert f16 -> f32.
                let bytes = logits_array.to_bytes();
                let numel = logits_array.numel();
                let mut result = Vec::with_capacity(numel);
                for i in 0..numel {
                    let offset = i * 2;
                    if offset + 2 <= bytes.len() {
                        let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                        result.push(f16_to_f32(bits));
                    }
                }
                Ok(result)
            }
            DType::Bfloat16 => {
                // Read raw bytes and convert bf16 -> f32.
                let bytes = logits_array.to_bytes();
                let numel = logits_array.numel();
                let mut result = Vec::with_capacity(numel);
                for i in 0..numel {
                    let offset = i * 2;
                    if offset + 2 <= bytes.len() {
                        let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                        result.push(bf16_to_f32(bits));
                    }
                }
                Ok(result)
            }
            other => Err(SchedulerError::BatchError(format!(
                "unsupported logits dtype: {:?}",
                other
            ))),
        }
    }

    /// Find a free slot in the batch cache for a new sequence.
    fn find_free_cache_slot(&self, batch_cache: &BatchKVCache) -> usize {
        for idx in 0..batch_cache.capacity() {
            if batch_cache.get(idx).is_none() {
                return idx;
            }
        }
        // If no free slot, return the next index after active count.
        // The caller should have ensured capacity.
        batch_cache.active_count()
    }

    /// Find the cache slot for a given sequence uid.
    ///
    /// Since we track uid-to-slot mapping implicitly via batch ordering,
    /// we search the decode batch for the sequence's position and map
    /// it to the corresponding batch cache slot. This uses a linear scan
    /// which is acceptable for typical batch sizes (< 256).
    fn find_cache_slot_for_uid(
        &self,
        uid: SequenceId,
        batch_cache: &BatchKVCache,
    ) -> Option<usize> {
        // The slot index corresponds to the sequence's position in the
        // batch at the time it was added. We iterate active cache slots
        // and match by position in the decode batch.
        //
        // Since sequences may have been removed (creating gaps), we need
        // a more robust approach: we maintain that the uid at decode_batch
        // position i corresponds to batch_cache slot i (after compaction).
        // For simplicity, we map the position in the batch to the i-th
        // active cache slot.
        let pos = self.decode_batch.iter().position(|s| s.uid == uid)?;

        // Find the pos-th active slot in the batch cache.
        let mut count = 0;
        for idx in 0..batch_cache.capacity() {
            if batch_cache.get(idx).is_some() {
                if count == pos {
                    return Some(idx);
                }
                count += 1;
            }
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Float conversion helpers
// ---------------------------------------------------------------------------

/// Convert an IEEE 754 half-precision float (f16) bit pattern to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero (positive or negative).
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: normalize.
            let mut m = mantissa;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let f32_exp = ((e + 127) as u32) & 0xff;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exponent == 31 {
        // Inf or NaN.
        if mantissa == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (mantissa << 13))
        }
    } else {
        // Normal number.
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
    }
}

/// Convert a bfloat16 bit pattern to f32 (simply shift left by 16 bits).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
