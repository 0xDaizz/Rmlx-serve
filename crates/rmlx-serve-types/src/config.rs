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

    /// Maximum number of sequences to prefill in a single batch.
    pub prefill_batch_size: usize,

    /// Maximum number of sequences in a single decode batch.
    pub completion_batch_size: usize,

    /// Emit a streaming token every N decode steps.
    pub stream_interval: usize,
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
            prefill_batch_size: 8,
            completion_batch_size: 32,
            stream_interval: 1,
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

    /// Maximum number of prefix cache entries.
    pub prefix_cache_size: usize,

    /// Explicit memory budget (in MiB) for the KV-cache. When set, overrides
    /// `cache_memory_percent`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_memory_mb: Option<usize>,

    /// Fraction of device memory to use for KV-cache (0.0 to 1.0).
    /// Only used when `cache_memory_mb` is `None`.
    pub cache_memory_percent: f64,

    /// Use memory-aware cache eviction (considers actual memory pressure).
    pub use_memory_aware_cache: bool,

    /// Enable KV-cache quantization to reduce memory usage.
    pub kv_cache_quantization: bool,

    /// Number of bits for KV-cache quantization (e.g. 4, 8).
    pub kv_cache_quantization_bits: u8,

    /// Group size for KV-cache quantization.
    pub kv_cache_quantization_group_size: usize,

    /// Minimum number of tokens before applying KV-cache quantization.
    pub kv_cache_min_quantize_tokens: usize,

    /// Enable paged KV-cache (vLLM-style block management).
    pub use_paged_cache: bool,

    /// Block size (in tokens) for the paged KV-cache.
    pub paged_cache_block_size: usize,

    /// Maximum number of paged cache blocks.
    pub max_cache_blocks: usize,
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
            prefix_cache_size: 100,
            cache_memory_mb: None,
            cache_memory_percent: 0.20,
            use_memory_aware_cache: true,
            kv_cache_quantization: false,
            kv_cache_quantization_bits: 8,
            kv_cache_quantization_group_size: 64,
            kv_cache_min_quantize_tokens: 256,
            use_paged_cache: false,
            paged_cache_block_size: 64,
            max_cache_blocks: 1000,
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

    /// Speculative decoding method name (e.g. "ngram", "draft", "mtp").
    /// `None` means speculative decoding is disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_method: Option<String>,

    /// Path or repo id for the draft model (speculative decoding).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_model: Option<String>,

    /// Path or repo id for the MTP (multi-token prediction) model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp_model: Option<String>,

    /// Enable multi-token prediction.
    pub enable_mtp: bool,

    /// Number of draft tokens for MTP.
    pub mtp_num_draft_tokens: usize,

    /// Use optimistic acceptance for MTP (accept all draft tokens greedily).
    pub mtp_optimistic: bool,

    /// Disable speculative decoding when batch size exceeds this value.
    /// `None` means never auto-disable based on batch size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_decode_disable_batch_size: Option<usize>,

    /// Acceptance rate threshold below which speculative decoding is
    /// automatically disabled.
    pub spec_decode_auto_disable_threshold: f64,

    /// Rolling window size (in steps) used to compute the acceptance rate
    /// for auto-disable.
    pub spec_decode_auto_disable_window: usize,

    /// Enable extended-thinking / chain-of-thought support.
    pub enable_thinking: bool,

    /// Reasoning output parser to use (e.g. "deepseek").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,

    /// Enable automatic tool-call detection in model output.
    pub enable_auto_tool_choice: bool,

    /// Tool-call output parser to use (e.g. "hermes", "llama3").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_parser: Option<String>,

    /// Multimodal LLM model identifier (e.g. "llava", "qwen-vl").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mllm: Option<String>,
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
            speculative_method: None,
            draft_model: None,
            mtp_model: None,
            enable_mtp: false,
            mtp_num_draft_tokens: 1,
            mtp_optimistic: false,
            spec_decode_disable_batch_size: None,
            spec_decode_auto_disable_threshold: 0.4,
            spec_decode_auto_disable_window: 50,
            enable_thinking: false,
            reasoning_parser: None,
            enable_auto_tool_choice: false,
            tool_call_parser: None,
            mllm: None,
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

    /// Rate limit: maximum requests per second per client. 0 = no limit.
    pub rate_limit: usize,

    /// Maximum number of tokens the server will generate per request.
    pub max_tokens: usize,

    /// Default sampling temperature applied when the client does not specify one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_temperature: Option<f64>,

    /// Default top-p value applied when the client does not specify one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_top_p: Option<f64>,

    /// Path to an MCP configuration file (`mcp.json`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_config: Option<String>,

    /// Optional embedding model path or repo id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
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
            rate_limit: 0,
            max_tokens: 32768,
            default_temperature: None,
            default_top_p: None,
            mcp_config: None,
            embedding_model: None,
        }
    }
}
