//! Hardware profiling and optimization hints.
//!
//! Provides types for describing the hardware capabilities of the
//! current system and generating optimization hints for the engine.

use serde::{Deserialize, Serialize};

/// A profile of the hardware available on this system.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// GPU device name.
    pub device_name: String,

    /// Total GPU memory in bytes.
    pub total_memory: u64,

    /// Available GPU memory in bytes.
    pub available_memory: u64,

    /// Number of GPU compute units / cores.
    pub compute_units: u32,
}

/// Optimization hints derived from the hardware profile.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OptimizationHints {
    /// Recommended maximum batch size.
    pub max_batch_size: usize,

    /// Recommended maximum sequence length.
    pub max_seq_len: usize,

    /// Whether to enable KV cache quantization.
    pub use_kv_quantization: bool,

    /// Whether to enable paged attention.
    pub use_paged_attention: bool,
}
