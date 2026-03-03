//! `rmlx-serve serve` subcommand -- start the HTTP API server.

use rmlx_serve_types::config::{CacheConfig, EngineConfig, SchedulerConfig, ServerConfig};
use tracing::info;

/// Arguments for the `serve` subcommand.
#[derive(clap::Args, Debug)]
pub struct ServeArgs {
    /// Path or HuggingFace repo id for the model to serve.
    #[arg(long, required = true)]
    pub model: String,

    /// Host address to bind the HTTP server to.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on.
    #[arg(long, default_value = "8000")]
    pub port: u16,

    /// Enable continuous batching mode (BatchedEngine). When false, uses
    /// SimpleEngine which processes one request at a time.
    #[arg(long, default_value = "false")]
    pub continuous_batching: bool,

    /// Maximum number of sequences in flight simultaneously.
    #[arg(long, default_value = "256")]
    pub max_num_seqs: usize,

    /// Maximum number of tokens in a single batch (prompt + generation).
    #[arg(long, default_value = "8192")]
    pub max_num_batched_tokens: usize,

    /// Optional API key for bearer-token authentication.
    #[arg(long)]
    pub api_key: Option<String>,

    /// Disable automatic prefix caching.
    #[arg(long, default_value = "false")]
    pub no_prefix_cache: bool,

    /// Memory budget for KV-cache in MiB. If not set, auto-detected.
    #[arg(long)]
    pub cache_memory_mb: Option<usize>,

    /// KV-cache quantization level: "4bit" or "8bit".
    #[arg(long)]
    pub kv_cache_quantization: Option<String>,

    /// Enable automatic tool-call detection in model output.
    #[arg(long, default_value = "false")]
    pub enable_auto_tool_choice: bool,

    /// Tool-call output parser to use (e.g. "hermes", "llama3").
    #[arg(long)]
    pub tool_call_parser: Option<String>,

    /// Enable extended-thinking / chain-of-thought support.
    #[arg(long, default_value = "false")]
    pub enable_thinking: bool,

    /// Reasoning output parser to use (e.g. "deepseek").
    #[arg(long)]
    pub reasoning_parser: Option<String>,

    /// Speculative decoding method: "ngram", "draft", or "mtp".
    #[arg(long)]
    pub speculative_method: Option<String>,

    /// Number of tokens to speculatively draft per step.
    #[arg(long, default_value = "5")]
    pub num_speculative_tokens: usize,

    /// Weight data type: "float16", "bfloat16", "float32", or "auto".
    #[arg(long)]
    pub dtype: Option<String>,
}

/// Run the `serve` subcommand.
pub async fn run_serve(args: ServeArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // -----------------------------------------------------------------------
    // 1. Build EngineConfig from CLI arguments
    // -----------------------------------------------------------------------
    let cache_dtype = match args.kv_cache_quantization.as_deref() {
        Some("4bit") => "f8e4m3".to_string(),
        Some("8bit") => "f16".to_string(),
        Some(other) => {
            return Err(format!(
                "unsupported --kv-cache-quantization value: {other:?}. Use \"4bit\" or \"8bit\"."
            )
            .into());
        }
        None => "f16".to_string(),
    };

    let dtype = match args.dtype.as_deref() {
        Some("float16") => "f16",
        Some("bfloat16") => "bf16",
        Some("float32") => "f32",
        Some("auto") | None => "auto",
        Some(other) => {
            return Err(format!(
                "unsupported --dtype value: {other:?}. \
                 Use \"float16\", \"bfloat16\", \"float32\", or \"auto\"."
            )
            .into());
        }
    };

    let enable_speculative = args.speculative_method.is_some();

    let engine_config = EngineConfig {
        model: args.model.clone(),
        tokenizer: None,
        revision: None,
        dtype: dtype.to_string(),
        tensor_parallel_size: 1,
        pipeline_parallel_size: 1,
        quantization: None,
        max_model_len: None,
        trust_remote_code: false,
        seed: 0,
        scheduler: SchedulerConfig {
            max_num_seqs: args.max_num_seqs,
            max_num_batched_tokens: args.max_num_batched_tokens,
            ..SchedulerConfig::default()
        },
        cache: CacheConfig {
            dtype: cache_dtype,
            enable_prefix_caching: !args.no_prefix_cache,
            gpu_memory_utilization: args
                .cache_memory_mb
                .map(|mb| {
                    // Rough heuristic: assume the device has ~16 GiB;
                    // convert the user's MiB budget into a 0..1 fraction.
                    // Real implementation should query actual device memory.
                    let fraction = (mb as f64) / (16.0 * 1024.0);
                    fraction.clamp(0.05, 0.99)
                })
                .unwrap_or(0.90),
            ..CacheConfig::default()
        },
        enable_speculative,
        speculative_model: None,
        speculative_draft_len: args.num_speculative_tokens,
    };

    // -----------------------------------------------------------------------
    // 2. Build ServerConfig
    // -----------------------------------------------------------------------
    let server_config = ServerConfig {
        host: args.host.clone(),
        port: args.port,
        api_key: args.api_key.clone(),
        enable_openai_api: true,
        enable_anthropic_api: true,
        enable_metrics: true,
        ..ServerConfig::default()
    };

    // -----------------------------------------------------------------------
    // 3. Print startup banner
    // -----------------------------------------------------------------------
    let batching_mode = if args.continuous_batching {
        "continuous"
    } else {
        "simple"
    };

    info!("=========================================================");
    info!("  rmlx-serve  --  LLM serving on Apple Silicon");
    info!("=========================================================");
    info!("model          : {}", args.model);
    info!("dtype          : {}", dtype);
    info!("batching       : {}", batching_mode);
    info!("max_num_seqs   : {}", args.max_num_seqs);
    info!("max_batch_tok  : {}", args.max_num_batched_tokens);
    info!("prefix_cache   : {}", !args.no_prefix_cache);
    info!("speculative    : {:?}", args.speculative_method);
    info!("spec_tokens    : {}", args.num_speculative_tokens);
    info!("thinking       : {}", args.enable_thinking);
    info!("auto_tool_call : {}", args.enable_auto_tool_choice);
    info!(
        "listen         : http://{}:{}",
        args.host, args.port
    );
    if args.api_key.is_some() {
        info!("auth           : bearer-token enabled");
    }
    info!("---------------------------------------------------------");

    // -----------------------------------------------------------------------
    // 4. Create engine and start server
    // -----------------------------------------------------------------------
    //
    // When rmlx_serve_engine provides concrete SimpleEngine / BatchedEngine
    // implementations, the code below should be:
    //
    //   let engine: Arc<dyn Engine> = if args.continuous_batching {
    //       Arc::new(BatchedEngine::new(engine_config).await?)
    //   } else {
    //       Arc::new(SimpleEngine::new(engine_config).await?)
    //   };
    //   rmlx_serve_api::serve(engine, server_config).await?;
    //
    // For now we use a StubEngine that validates the full startup path.

    let engine = std::sync::Arc::new(crate::stub_engine::StubEngine::new(
        engine_config,
    ));

    info!("engine initialised (stub mode -- no real model loaded)");
    info!("starting HTTP server on {}:{} ...", server_config.host, server_config.port);

    // When rmlx_serve_api::serve() is implemented, replace the block below:
    //   rmlx_serve_api::serve(engine, server_config).await?;

    // Placeholder: bind a minimal health-check server so the binary is functional.
    use axum::{routing::get, Json, Router};
    use std::net::SocketAddr;

    let engine_ref = engine.clone();
    let app = Router::new()
        .route(
            "/health",
            get({
                let engine = engine_ref.clone();
                move || {
                    let engine = engine.clone();
                    async move {
                        let health = rmlx_serve_engine::Engine::health(engine.as_ref()).await;
                        Json(health)
                    }
                }
            }),
        )
        .route(
            "/v1/models",
            get({
                let engine = engine_ref.clone();
                move || {
                    let engine = engine.clone();
                    async move {
                        let name = rmlx_serve_engine::Engine::model_name(engine.as_ref());
                        Json(serde_json::json!({
                            "object": "list",
                            "data": [{
                                "id": name,
                                "object": "model",
                                "owned_by": "rmlx-serve",
                            }]
                        }))
                    }
                }
            }),
        );

    let addr: SocketAddr = format!("{}:{}", server_config.host, server_config.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("server listening on {}", addr);

    axum::serve(listener, app.into_make_service())
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;

    Ok(())
}
