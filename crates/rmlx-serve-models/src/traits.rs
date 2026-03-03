//! Core trait defining the interface for LLM models in the serving stack.

use rmlx_core::{Array, KernelRegistry};
use rmlx_nn::{LayerKvCache, TransformerConfig};

use crate::error::ModelError;

/// Trait defining the interface for a large language model used in inference.
///
/// Implementors wrap a specific model architecture (e.g., `TransformerModel`)
/// and handle RoPE frequency precomputation, causal mask creation, cache
/// management, and last-token logit extraction.
///
/// All methods that perform GPU work take a `KernelRegistry` and
/// `metal::CommandQueue` to dispatch Metal compute commands.
pub trait LlmModel: Send {
    /// Run a forward pass through the model.
    ///
    /// # Arguments
    /// * `token_ids` - Input token indices for this step. For prefill, this is
    ///   the full prompt; for decode steps, this is typically a single token.
    /// * `cache` - Optional mutable reference to per-layer KV caches. When
    ///   provided, the model appends new K/V entries and uses cached history.
    /// * `registry` - Kernel registry for dispatching Metal compute pipelines.
    /// * `queue` - Metal command queue for GPU command submission.
    ///
    /// # Returns
    /// Logits for the **last token only** as an `Array` with shape
    /// `[1, vocab_size]`. During decode steps this avoids recomputing logits
    /// for cached positions.
    fn forward(
        &self,
        token_ids: &[u32],
        cache: Option<&mut Vec<LayerKvCache>>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, ModelError>;

    /// Create a fresh set of pre-allocated KV caches for all layers.
    ///
    /// Each layer gets a `LayerKvCache` with room for `max_seq_len` tokens,
    /// using Float16 precision.
    fn make_cache(&self, device: &metal::Device) -> Vec<LayerKvCache>;

    /// Number of transformer layers in the model.
    fn num_layers(&self) -> usize;

    /// Hidden dimension of the model (embedding width).
    fn hidden_size(&self) -> usize;

    /// Vocabulary size (number of output logits).
    fn vocab_size(&self) -> usize;

    /// Per-head dimension for attention.
    fn head_dim(&self) -> usize;

    /// Number of key-value heads (may differ from query heads in GQA models).
    fn num_kv_heads(&self) -> usize;

    /// Maximum sequence length the model supports.
    fn max_seq_len(&self) -> usize;

    /// Reference to the underlying transformer configuration.
    fn config(&self) -> &TransformerConfig;
}
