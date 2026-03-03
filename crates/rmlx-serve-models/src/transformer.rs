//! TransformerLlm: wraps `rmlx_nn::TransformerModel` with RoPE precomputation,
//! causal mask creation, cache management, and last-token logit extraction.

use rmlx_core::{Array, DType, KernelRegistry};
use rmlx_metal::GpuDevice;
use rmlx_nn::{LayerKvCache, TransformerConfig, TransformerModel};
use tracing::{debug, info};

use crate::error::{ModelError, Result};
use crate::mask;
use crate::rope;
use crate::traits::LlmModel;

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
}

impl TransformerLlm {
    /// Create a new `TransformerLlm` from a loaded model, config, and device.
    ///
    /// Precomputes RoPE frequency tables and validates the configuration.
    pub fn new(
        model: TransformerModel,
        config: TransformerConfig,
        device: GpuDevice,
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
            "initializing TransformerLlm"
        );

        let (cos_freqs, sin_freqs) = Self::precompute_rope_frequencies(&config, &device);

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
        })
    }

    /// Precompute RoPE frequency tables for all positions up to `max_seq_len`.
    ///
    /// Uses the formula:
    ///   theta_i = rope_theta^(-2i / head_dim)  for i in 0..head_dim/2
    ///   cos_freqs[pos][i] = cos(pos * theta_i)
    ///   sin_freqs[pos][i] = sin(pos * theta_i)
    fn precompute_rope_frequencies(
        config: &TransformerConfig,
        device: &GpuDevice,
    ) -> (Array, Array) {
        rope::compute_rope_frequencies(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
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
        cache: Option<&mut Vec<LayerKvCache>>,
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
