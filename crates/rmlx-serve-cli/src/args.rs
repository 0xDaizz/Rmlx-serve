//! CLI argument definitions for the `serve` subcommand.
//!
//! This module owns the [`ServeArgs`] struct and its conversion methods into
//! the config types defined in `rmlx-serve-types`.  It is kept separate from
//! `serve.rs` so that engine refactors and flag additions can proceed without
//! merge conflicts.

use rmlx_serve_types::config::{CacheConfig, EngineConfig, SchedulerConfig, ServerConfig};

// ===========================================================================
// ServeArgs
// ===========================================================================

/// Arguments for the `serve` subcommand.
#[derive(clap::Args, Debug, Clone)]
pub struct ServeArgs {
    // -- Server -----------------------------------------------------------

    /// Path or HuggingFace repo id for the model to serve.
    #[arg(short = 'm', long, required = true)]
    pub model: String,

    /// Optional path or repo id for the tokenizer (defaults to model path).
    #[arg(long)]
    pub tokenizer: Option<String>,

    /// Host address to bind the HTTP server to.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// Optional API key for bearer-token authentication.
    #[arg(long)]
    pub api_key: Option<String>,

    /// Enable continuous batching mode (BatchedEngine). When false, uses
    /// SimpleEngine which processes one request at a time.
    #[arg(long, default_value_t = false)]
    pub continuous_batching: bool,

    /// Maximum number of sequences in flight simultaneously.
    #[arg(long, default_value_t = 256)]
    pub max_num_seqs: usize,

    /// Maximum number of tokens in a single batch (prompt + generation).
    #[arg(long, default_value_t = 8192)]
    pub max_num_batched_tokens: usize,

    /// Weight data type: "float16", "bfloat16", "float32", or "auto".
    #[arg(long)]
    pub dtype: Option<String>,

    // -- Batching ----------------------------------------------------------

    /// Maximum number of sequences to prefill in a single batch.
    #[arg(long, default_value_t = 8)]
    pub prefill_batch_size: usize,

    /// Maximum number of sequences in a single decode batch.
    #[arg(long, default_value_t = 32)]
    pub completion_batch_size: usize,

    /// Emit a streaming token every N decode steps.
    #[arg(long, default_value_t = 1)]
    pub stream_interval: usize,

    /// Chunked prefill token budget (0 = disabled).
    #[arg(long, default_value_t = 0)]
    pub chunked_prefill_tokens: usize,

    // -- Cache -------------------------------------------------------------

    /// Maximum number of prefix cache entries.
    #[arg(long, default_value_t = 100)]
    pub prefix_cache_size: usize,

    /// Fraction of device memory to use for KV-cache (0.0 to 1.0).
    #[arg(long, default_value_t = 0.20)]
    pub cache_memory_percent: f64,

    /// Explicit memory budget (MiB) for KV-cache. Overrides --cache-memory-percent.
    #[arg(long)]
    pub cache_memory_mb: Option<usize>,

    /// Disable memory-aware cache eviction.
    #[arg(long, default_value_t = false)]
    pub no_memory_aware_cache: bool,

    /// Enable KV-cache quantization.
    #[arg(long, default_value_t = false)]
    pub kv_cache_quantization: bool,

    /// Number of bits for KV-cache quantization.
    #[arg(long, default_value_t = 8)]
    pub kv_cache_quantization_bits: u8,

    /// Group size for KV-cache quantization.
    #[arg(long, default_value_t = 64)]
    pub kv_cache_quantization_group_size: usize,

    /// Minimum number of tokens before applying KV-cache quantization.
    #[arg(long, default_value_t = 256)]
    pub kv_cache_min_quantize_tokens: usize,

    /// Enable paged KV-cache (vLLM-style block management).
    #[arg(long, default_value_t = false)]
    pub use_paged_cache: bool,

    /// Block size (in tokens) for paged KV-cache.
    #[arg(long, default_value_t = 64)]
    pub paged_cache_block_size: usize,

    /// Maximum number of paged cache blocks.
    #[arg(long, default_value_t = 1000)]
    pub max_cache_blocks: usize,

    /// Enable automatic prefix caching (prompt prefix sharing).
    #[arg(long, default_value_t = true)]
    pub enable_prefix_caching: bool,

    // -- Generation --------------------------------------------------------

    /// Maximum number of tokens the server will generate per request.
    #[arg(long, default_value_t = 32768)]
    pub max_tokens: usize,

    /// Default sampling temperature when the client does not specify one.
    #[arg(long)]
    pub default_temperature: Option<f64>,

    /// Default top-p value when the client does not specify one.
    #[arg(long)]
    pub default_top_p: Option<f64>,

    // -- Security ----------------------------------------------------------

    /// Rate limit: maximum requests per second per client. 0 = no limit.
    #[arg(long, default_value_t = 0)]
    pub rate_limit: usize,

    /// Request timeout in seconds.
    #[arg(long, default_value_t = 300.0)]
    pub timeout: f64,

    // -- Speculative decoding ----------------------------------------------

    /// Speculative decoding method: "ngram", "draft", or "mtp".
    #[arg(long)]
    pub speculative_method: Option<String>,

    /// Disable speculative decoding when batch size exceeds this value.
    #[arg(long)]
    pub spec_decode_disable_batch_size: Option<usize>,

    /// Path or repo id for the draft model (speculative decoding).
    #[arg(long)]
    pub draft_model: Option<String>,

    /// Path or repo id for the MTP (multi-token prediction) model.
    #[arg(long)]
    pub mtp_model: Option<String>,

    /// Acceptance rate threshold below which speculative decoding is auto-disabled.
    #[arg(long, default_value_t = 0.4)]
    pub spec_decode_auto_disable_threshold: f64,

    /// Rolling window size (in steps) for computing speculative acceptance rate.
    #[arg(long, default_value_t = 50)]
    pub spec_decode_auto_disable_window: usize,

    /// Number of tokens to speculatively draft per step.
    #[arg(long, default_value_t = 5)]
    pub num_speculative_tokens: usize,

    // -- MTP ---------------------------------------------------------------

    /// Enable multi-token prediction.
    #[arg(long, default_value_t = false)]
    pub enable_mtp: bool,

    /// Number of draft tokens for MTP.
    #[arg(long, default_value_t = 1)]
    pub mtp_num_draft_tokens: usize,

    /// Use optimistic acceptance for MTP (accept all draft tokens greedily).
    #[arg(long, default_value_t = false)]
    pub mtp_optimistic: bool,

    // -- Tool / Reasoning --------------------------------------------------

    /// Enable automatic tool-call detection in model output.
    #[arg(long, default_value_t = false)]
    pub enable_auto_tool_choice: bool,

    /// Tool-call output parser: "hermes", "llama", "mistral", "deepseek", etc.
    #[arg(long)]
    pub tool_call_parser: Option<String>,

    /// Enable extended-thinking / chain-of-thought support.
    #[arg(long, default_value_t = false)]
    pub enable_thinking: bool,

    /// Reasoning output parser: "deepseek", "harmony", etc.
    #[arg(long)]
    pub reasoning_parser: Option<String>,

    // -- Other -------------------------------------------------------------

    /// Path to an MCP configuration file (mcp.json).
    #[arg(long)]
    pub mcp_config: Option<String>,

    /// Optional embedding model path or repo id.
    #[arg(long)]
    pub embedding_model: Option<String>,

    /// Multimodal LLM model identifier (e.g. "llava", "qwen-vl").
    #[arg(long)]
    pub mllm: Option<String>,

    // -- Distributed -------------------------------------------------------

    /// Enable distributed inference across multiple devices/nodes.
    #[arg(long, default_value_t = false)]
    pub distributed: bool,

    /// Distributed backend: "gloo", "nccl", etc.
    #[arg(long, default_value = "gloo")]
    pub dist_backend: String,

    /// Number of ranks (processes) for distributed inference.
    #[arg(long)]
    pub dist_num_ranks: Option<usize>,

    /// Path to a hostfile for multi-node distributed inference.
    #[arg(long)]
    pub dist_hostfile: Option<String>,

    /// Enable expert parallelism (for MoE models).
    #[arg(long, default_value_t = false)]
    pub expert_parallel: bool,

    /// Kernel backend for expert parallelism.
    #[arg(long)]
    pub ep_kernel_backend: Option<String>,
}

// ===========================================================================
// Conversion methods
// ===========================================================================

impl ServeArgs {
    /// Build an [`EngineConfig`] from the CLI arguments.
    pub fn to_engine_config(&self) -> Result<EngineConfig, String> {
        let dtype = match self.dtype.as_deref() {
            Some("float16") => "f16",
            Some("bfloat16") => "bf16",
            Some("float32") => "f32",
            Some("auto") | None => "auto",
            Some(other) => {
                return Err(format!(
                    "unsupported --dtype value: {other:?}. \
                     Use \"float16\", \"bfloat16\", \"float32\", or \"auto\"."
                ));
            }
        };

        let enable_speculative = self.speculative_method.is_some();

        let enable_chunked_prefill = self.chunked_prefill_tokens > 0;
        let max_prefill_chunk_size = if enable_chunked_prefill {
            self.chunked_prefill_tokens
        } else {
            2048 // default from SchedulerConfig
        };

        Ok(EngineConfig {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            revision: None,
            dtype: dtype.to_string(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            quantization: None,
            max_model_len: None,
            trust_remote_code: false,
            seed: 0,
            scheduler: SchedulerConfig {
                max_num_seqs: self.max_num_seqs,
                max_num_batched_tokens: self.max_num_batched_tokens,
                prefill_batch_size: self.prefill_batch_size,
                completion_batch_size: self.completion_batch_size,
                stream_interval: self.stream_interval,
                enable_chunked_prefill,
                max_prefill_chunk_size,
                ..SchedulerConfig::default()
            },
            cache: self.to_cache_config(),
            enable_speculative,
            speculative_model: None,
            speculative_draft_len: self.num_speculative_tokens,
            speculative_method: self.speculative_method.clone(),
            draft_model: self.draft_model.clone(),
            mtp_model: self.mtp_model.clone(),
            enable_mtp: self.enable_mtp,
            mtp_num_draft_tokens: self.mtp_num_draft_tokens,
            mtp_optimistic: self.mtp_optimistic,
            spec_decode_disable_batch_size: self.spec_decode_disable_batch_size,
            spec_decode_auto_disable_threshold: self.spec_decode_auto_disable_threshold,
            spec_decode_auto_disable_window: self.spec_decode_auto_disable_window,
            enable_thinking: self.enable_thinking,
            reasoning_parser: self.reasoning_parser.clone(),
            enable_auto_tool_choice: self.enable_auto_tool_choice,
            tool_call_parser: self.tool_call_parser.clone(),
            mllm: self.mllm.clone(),
        })
    }

    /// Build a [`SchedulerConfig`] from the CLI arguments.
    pub fn to_scheduler_config(&self) -> SchedulerConfig {
        let enable_chunked_prefill = self.chunked_prefill_tokens > 0;
        let max_prefill_chunk_size = if enable_chunked_prefill {
            self.chunked_prefill_tokens
        } else {
            2048
        };

        SchedulerConfig {
            max_num_seqs: self.max_num_seqs,
            max_num_batched_tokens: self.max_num_batched_tokens,
            prefill_batch_size: self.prefill_batch_size,
            completion_batch_size: self.completion_batch_size,
            stream_interval: self.stream_interval,
            enable_chunked_prefill,
            max_prefill_chunk_size,
            ..SchedulerConfig::default()
        }
    }

    /// Build a [`CacheConfig`] from the CLI arguments.
    pub fn to_cache_config(&self) -> CacheConfig {
        CacheConfig {
            enable_prefix_caching: self.enable_prefix_caching,
            prefix_cache_size: self.prefix_cache_size,
            cache_memory_mb: self.cache_memory_mb,
            cache_memory_percent: self.cache_memory_percent,
            use_memory_aware_cache: !self.no_memory_aware_cache,
            kv_cache_quantization: self.kv_cache_quantization,
            kv_cache_quantization_bits: self.kv_cache_quantization_bits,
            kv_cache_quantization_group_size: self.kv_cache_quantization_group_size,
            kv_cache_min_quantize_tokens: self.kv_cache_min_quantize_tokens,
            use_paged_cache: self.use_paged_cache,
            paged_cache_block_size: self.paged_cache_block_size,
            max_cache_blocks: self.max_cache_blocks,
            // Derive gpu_memory_utilization from cache_memory_mb when provided
            gpu_memory_utilization: self
                .cache_memory_mb
                .map(|mb| {
                    let fraction = (mb as f64) / (16.0 * 1024.0);
                    fraction.clamp(0.05, 0.99)
                })
                .unwrap_or(0.90),
            ..CacheConfig::default()
        }
    }

    /// Build a [`ServerConfig`] from the CLI arguments.
    pub fn to_server_config(&self) -> ServerConfig {
        ServerConfig {
            host: self.host.clone(),
            port: self.port,
            api_key: self.api_key.clone(),
            rate_limit: self.rate_limit,
            request_timeout_secs: self.timeout as u64,
            max_tokens: self.max_tokens,
            default_temperature: self.default_temperature,
            default_top_p: self.default_top_p,
            mcp_config: self.mcp_config.clone(),
            embedding_model: self.embedding_model.clone(),
            enable_openai_api: true,
            enable_anthropic_api: true,
            enable_metrics: true,
            ..ServerConfig::default()
        }
    }
}
