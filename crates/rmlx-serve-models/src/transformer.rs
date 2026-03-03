//! TransformerLlm: wraps `rmlx_nn::TransformerModel` with RoPE precomputation,
//! causal mask creation, cache management, and last-token logit extraction.

use rmlx_core::{Array, DType, KernelRegistry};
use rmlx_metal::GpuDevice;
use rmlx_nn::{LayerKvCache, TransformerConfig, TransformerModel};
use tracing::{debug, info};

use crate::error::{ModelError, Result};
use crate::mask;
use crate::rope;
use crate::rope::RopeScalingMethod;
use crate::traits::LlmModel;

/// Default prefill chunk size in tokens. Sequences longer than this are
/// processed in chunks to limit peak memory usage.
const DEFAULT_PREFILL_CHUNK_SIZE: usize = 512;

/// A fully-initialized transformer LLM ready for inference.
///
/// Wraps an `rmlx_nn::TransformerModel` and manages:
/// 1. **RoPE frequencies** — precomputed at construction time for all positions
///    up to `max_seq_len`.
/// 2. **Causal masks** — created per forward pass based on sequence length and
///    cache offset.
/// 3. **KV cache** — preallocated per-layer caches for incremental decoding.
/// 4. **Last-token extraction** — during decode steps, only the last position's
///    logits are returned to avoid redundant computation.
pub struct TransformerLlm {
    /// The underlying RMLX transformer model with loaded weights.
    model: TransformerModel,
    /// Model configuration (dimensions, layer count, etc.).
    config: TransformerConfig,
    /// Precomputed cosine RoPE frequency table: [max_seq_len, head_dim/2].
    cos_freqs: Array,
    /// Precomputed sine RoPE frequency table: [max_seq_len, head_dim/2].
    sin_freqs: Array,
    /// GPU device reference for buffer allocation.
    device: GpuDevice,
    /// Maximum number of tokens to process in a single prefill chunk.
    /// Sequences longer than this are split into chunks to limit peak
    /// memory usage. Defaults to [`DEFAULT_PREFILL_CHUNK_SIZE`].
    prefill_chunk_size: usize,
}

impl TransformerLlm {
    /// Create a new `TransformerLlm` from a loaded model, config, and device.
    ///
    /// Precomputes RoPE frequency tables (with no scaling) and validates the
    /// configuration. See [`Self::with_rope_scaling`] for RoPE scaling support.
    pub fn new(
        model: TransformerModel,
        config: TransformerConfig,
        device: GpuDevice,
    ) -> Result<Self> {
        Self::with_rope_scaling(model, config, device, RopeScalingMethod::None)
    }

    /// Create a new `TransformerLlm` with explicit RoPE scaling.
    ///
    /// Precomputes RoPE frequency tables using the given scaling method and
    /// validates the configuration.
    pub fn with_rope_scaling(
        model: TransformerModel,
        config: TransformerConfig,
        device: GpuDevice,
        rope_scaling: RopeScalingMethod,
    ) -> Result<Self> {
        config.validate().map_err(ModelError::KernelError)?;

        info!(
            hidden_size = config.hidden_size,
            num_layers = config.num_layers,
            num_heads = config.num_heads,
            num_kv_heads = config.num_kv_heads,
            head_dim = config.head_dim,
            max_seq_len = config.max_seq_len,
            vocab_size = config.vocab_size,
            rope_theta = config.rope_theta,
            rope_scaling = ?rope_scaling,
            "initializing TransformerLlm"
        );

        let (cos_freqs, sin_freqs) =
            Self::precompute_rope_frequencies(&config, &rope_scaling, &device);

        debug!(
            cos_shape = ?cos_freqs.shape(),
            sin_shape = ?sin_freqs.shape(),
            "RoPE frequencies precomputed"
        );

        Ok(Self {
            model,
            config,
            cos_freqs,
            sin_freqs,
            device,
            prefill_chunk_size: DEFAULT_PREFILL_CHUNK_SIZE,
        })
    }

    /// Set the maximum prefill chunk size (in tokens).
    ///
    /// When the input sequence to `forward()` exceeds this size, it will be
    /// processed in chunks to limit peak memory consumption.
    pub fn set_prefill_chunk_size(&mut self, chunk_size: usize) {
        self.prefill_chunk_size = chunk_size.max(1);
    }

    /// Precompute RoPE frequency tables for all positions up to `max_seq_len`.
    ///
    /// Supports standard RoPE as well as linear, NTK-aware, and YaRN scaling
    /// methods via the `scaling` parameter.
    fn precompute_rope_frequencies(
        config: &TransformerConfig,
        scaling: &RopeScalingMethod,
        device: &GpuDevice,
    ) -> (Array, Array) {
        rope::compute_rope_frequencies_with_scaling(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            scaling,
            device.raw(),
        )
    }

    /// Create a causal attention mask for the given sequence length and cache offset.
    ///
    /// The mask is a lower-triangular matrix where positions above the diagonal
    /// (accounting for the offset) are set to negative infinity.
    fn create_causal_mask(seq_len: usize, offset: usize, device: &metal::Device) -> Array {
        mask::create_causal_mask(seq_len, offset, device)
    }

    /// Reference to the underlying `GpuDevice`.
    pub fn gpu_device(&self) -> &GpuDevice {
        &self.device
    }
}

impl LlmModel for TransformerLlm {
    fn forward(
        &self,
        token_ids: &[u32],
        mut cache: Option<&mut Vec<LayerKvCache>>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> std::result::Result<Array, ModelError> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(ModelError::ShapeError(
                "forward: token_ids must not be empty".into(),
            ));
        }

        // 1. Determine cache offset (number of previously cached tokens)
        let offset = cache
            .as_ref()
            .and_then(|c| c.first())
            .map(|layer_cache| layer_cache.position_offset())
            .unwrap_or(0);

        // Validate we won't exceed max_seq_len
        if offset + seq_len > self.config.max_seq_len {
            return Err(ModelError::ShapeError(format!(
                "sequence length {} + offset {} = {} exceeds max_seq_len {}",
                seq_len,
                offset,
                offset + seq_len,
                self.config.max_seq_len
            )));
        }

        // ── Chunked prefill ──────────────────────────────────────────
        // If the sequence is longer than `prefill_chunk_size` and we have
        // a KV cache, split into chunks. Each chunk runs a forward pass
        // and updates the cache so the next chunk sees the previously
        // computed K/V entries. This limits peak memory proportional to
        // chunk_size rather than full seq_len.
        if seq_len > self.prefill_chunk_size {
            if let Some(cache) = cache.as_mut() {
                let mut last_logits: Option<Array> = None;

                let mut chunk_start = 0;
                while chunk_start < seq_len {
                    let chunk_end = (chunk_start + self.prefill_chunk_size).min(seq_len);
                    let chunk_tokens = &token_ids[chunk_start..chunk_end];
                    let chunk_len = chunk_tokens.len();

                    // Current offset includes both the original offset and
                    // tokens from previous chunks in this call.
                    let chunk_offset = offset + chunk_start;

                    // Slice RoPE frequencies for this chunk
                    let cos_slice = self
                        .cos_freqs
                        .slice(0, chunk_offset, chunk_offset + chunk_len)
                        .map_err(ModelError::KernelError)?;
                    let sin_slice = self
                        .sin_freqs
                        .slice(0, chunk_offset, chunk_offset + chunk_len)
                        .map_err(ModelError::KernelError)?;

                    // Build causal mask for this chunk
                    let causal_mask =
                        Self::create_causal_mask(chunk_len, chunk_offset, self.device.raw());

                    // Run forward pass for this chunk (updates cache in place)
                    let logits = self.model.forward(
                        chunk_tokens,
                        Some(&cos_slice),
                        Some(&sin_slice),
                        Some(&causal_mask),
                        Some(cache),
                        registry,
                        queue,
                    )?;

                    debug!(
                        chunk_start = chunk_start,
                        chunk_end = chunk_end,
                        chunk_offset = chunk_offset,
                        "processed prefill chunk"
                    );

                    last_logits = Some(logits);
                    chunk_start = chunk_end;
                }

                // Extract last token logits from the final chunk's output
                let logits = last_logits.expect("at least one chunk must be processed");
                let final_chunk_len =
                    seq_len - (seq_len / self.prefill_chunk_size) * self.prefill_chunk_size;
                let final_chunk_len = if final_chunk_len == 0 {
                    self.prefill_chunk_size
                } else {
                    final_chunk_len
                };
                if final_chunk_len > 1 {
                    let last = logits
                        .slice(0, final_chunk_len - 1, final_chunk_len)
                        .map_err(ModelError::KernelError)?;
                    return Ok(last);
                } else {
                    return Ok(logits);
                }
            }
        }

        // ── Standard (non-chunked) forward pass ──────────────────────

        // 2. Slice precomputed RoPE frequencies for [offset..offset+seq_len]
        let cos_slice = self
            .cos_freqs
            .slice(0, offset, offset + seq_len)
            .map_err(ModelError::KernelError)?;
        let sin_slice = self
            .sin_freqs
            .slice(0, offset, offset + seq_len)
            .map_err(ModelError::KernelError)?;

        // 3. Build causal mask for this step
        let causal_mask = Self::create_causal_mask(seq_len, offset, self.device.raw());

        // 4. Run the transformer forward pass
        let logits = self.model.forward(
            token_ids,
            Some(&cos_slice),
            Some(&sin_slice),
            Some(&causal_mask),
            cache,
            registry,
            queue,
        )?;

        // 5. Extract last token logits only (optimization for decode steps).
        //    The model returns [seq_len, vocab_size]; we want [1, vocab_size].
        if seq_len > 1 {
            let last_logits = logits
                .slice(0, seq_len - 1, seq_len)
                .map_err(ModelError::KernelError)?;
            Ok(last_logits)
        } else {
            // Already [1, vocab_size]
            Ok(logits)
        }
    }

    fn make_cache(&self, device: &metal::Device) -> Vec<LayerKvCache> {
        (0..self.config.num_layers)
            .map(|_| {
                LayerKvCache::preallocated(
                    device,
                    self.config.num_kv_heads,
                    self.config.head_dim,
                    self.config.max_seq_len,
                    DType::Float16,
                )
            })
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn head_dim(&self) -> usize {
        self.config.head_dim
    }

    fn num_kv_heads(&self) -> usize {
        self.config.num_kv_heads
    }

    fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    fn config(&self) -> &TransformerConfig {
        &self.config
    }
}
