//! Configuration types for the rmlx-serve engine.

use serde::{Deserialize, Serialize};

// ===========================================================================
// SchedulerConfig
// ===========================================================================

/// Configuration for the request scheduler.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum number of sequences that can be in-flight simultaneously.
    pub max_num_seqs: usize,

    /// Maximum number of tokens in a single batch (prompt + generation).
    pub max_num_batched_tokens: usize,

    /// Maximum model context length (in tokens). Requests exceeding this are
    /// rejected.
    pub max_model_len: usize,

    /// Maximum number of tokens in a single prompt.
    pub max_prompt_len: usize,

    /// Whether to enable chunked prefill: large prompts are split into chunks
    /// and interleaved with decode steps for better latency.
    pub enable_chunked_prefill: bool,

    /// Maximum chunk size (in tokens) when chunked prefill is enabled.
    pub max_prefill_chunk_size: usize,

    /// Scheduling policy: `"fcfs"` (first-come-first-served) or `"priority"`.
    pub policy: String,

    /// Number of scheduler steps to look ahead when pre-allocating KV cache
    /// blocks. Higher values reduce stalls but may waste memory.
    pub lookahead_slots: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            max_model_len: 4096,
            max_prompt_len: 4096,
            enable_chunked_prefill: false,
            max_prefill_chunk_size: 2048,
            policy: "fcfs".to_string(),
            lookahead_slots: 0,
        }
    }
}

// ===========================================================================
// CacheConfig
// ===========================================================================

/// Configuration for the KV-cache (paged attention).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Block size in number of tokens per block.
    pub block_size: usize,

    /// Fraction of device memory (0.0 to 1.0) reserved for KV-cache.
    pub gpu_memory_utilization: f64,

    /// Swap space on CPU (in GiB) for sequence preemption.
    pub swap_space_gib: f64,

    /// Number of KV-cache layers. `None` means auto-detect from model config.
    pub num_layers: Option<usize>,

    /// Number of attention heads for the KV-cache. `None` = auto-detect.
    pub num_kv_heads: Option<usize>,

    /// Head dimension. `None` = auto-detect.
    pub head_dim: Option<usize>,

    /// Data type for cached keys/values: `"f16"`, `"bf16"`, `"f32"`, `"f8e4m3"`.
    pub dtype: String,

    /// Enable prefix caching (automatic prompt prefix sharing).
    pub enable_prefix_caching: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            gpu_memory_utilization: 0.90,
            swap_space_gib: 4.0,
            num_layers: None,
            num_kv_heads: None,
            head_dim: None,
            dtype: "f16".to_string(),
            enable_prefix_caching: false,
        }
    }
}

// ===========================================================================
// EngineConfig
// ===========================================================================

/// Configuration for the inference engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Path or HuggingFace repo id for the model.
    pub model: String,

    /// Optional path or repo id for the tokenizer (defaults to `model`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,

    /// Revision / branch for the model repo.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,

    /// Data type for model weights: `"f16"`, `"bf16"`, `"f32"`, `"auto"`.
    pub dtype: String,

    /// Number of tensor-parallel shards. 1 = single device.
    pub tensor_parallel_size: usize,

    /// Number of pipeline-parallel stages. 1 = no pipeline parallelism.
    pub pipeline_parallel_size: usize,

    /// Quantization method, if any: `"gptq"`, `"awq"`, `"fp8"`, etc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,

    /// Maximum model context length override. `None` = use the model's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,

    /// Whether to trust remote code when loading the model.
    pub trust_remote_code: bool,

    /// Random seed for weight initialisation (if applicable).
    pub seed: u64,

    /// Scheduler configuration.
    pub scheduler: SchedulerConfig,

    /// KV-cache configuration.
    pub cache: CacheConfig,

    /// Enable speculative decoding.
    pub enable_speculative: bool,

    /// Draft model path for speculative decoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_model: Option<String>,

    /// Number of draft tokens for speculative decoding.
    pub speculative_draft_len: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            tokenizer: None,
            revision: None,
            dtype: "auto".to_string(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            quantization: None,
            max_model_len: None,
            trust_remote_code: false,
            seed: 0,
            scheduler: SchedulerConfig::default(),
            cache: CacheConfig::default(),
            enable_speculative: false,
            speculative_model: None,
            speculative_draft_len: 5,
        }
    }
}

// ===========================================================================
// ServerConfig
// ===========================================================================

/// Configuration for the HTTP server layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host address to bind to.
    pub host: String,

    /// Port to listen on.
    pub port: u16,

    /// Maximum number of concurrent connections.
    pub max_connections: usize,

    /// Request timeout in seconds.
    pub request_timeout_secs: u64,

    /// CORS allowed origins. Empty = disallow all cross-origin requests.
    /// Use `["*"]` to allow all.
    #[serde(default)]
    pub cors_allowed_origins: Vec<String>,

    /// Optional API key for authentication. `None` = no auth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Whether to enable the OpenAI-compatible API endpoints.
    pub enable_openai_api: bool,

    /// Whether to enable the Anthropic-compatible API endpoints.
    pub enable_anthropic_api: bool,

    /// Whether to expose Prometheus-compatible metrics at `/metrics`.
    pub enable_metrics: bool,

    /// Log level: `"trace"`, `"debug"`, `"info"`, `"warn"`, `"error"`.
    pub log_level: String,

    /// Optional SSL certificate path for HTTPS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssl_cert_path: Option<String>,

    /// Optional SSL key path for HTTPS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssl_key_path: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            max_connections: 1024,
            request_timeout_secs: 300,
            cors_allowed_origins: Vec::new(),
            api_key: None,
            enable_openai_api: true,
            enable_anthropic_api: false,
            enable_metrics: true,
            log_level: "info".to_string(),
            ssl_cert_path: None,
            ssl_key_path: None,
        }
    }
}
