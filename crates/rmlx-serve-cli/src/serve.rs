//! `rmlx-serve serve` subcommand -- start the HTTP API server.

use crate::args::ServeArgs;
use rmlx_serve_engine::{BatchedEngine, Engine, SimpleEngine};
use std::sync::Arc;
use tracing::info;

/// Run the `serve` subcommand.
pub async fn run_serve(args: ServeArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // -----------------------------------------------------------------------
    // 1. Build EngineConfig from CLI arguments
    // -----------------------------------------------------------------------
    let engine_config = args
        .to_engine_config()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    // -----------------------------------------------------------------------
    // 2. Build ServerConfig
    // -----------------------------------------------------------------------
    let server_config = args.to_server_config();

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
    info!("dtype          : {}", engine_config.dtype);
    info!("batching       : {}", batching_mode);
    info!("max_num_seqs   : {}", args.max_num_seqs);
    info!("max_batch_tok  : {}", args.max_num_batched_tokens);
    info!("prefix_cache   : {}", args.enable_prefix_caching);
    info!("speculative    : {:?}", args.speculative_method);
    info!("spec_tokens    : {}", args.num_speculative_tokens);
    info!("thinking       : {}", args.enable_thinking);
    info!("auto_tool_call : {}", args.enable_auto_tool_choice);
    info!("listen         : http://{}:{}", args.host, args.port);
    if args.api_key.is_some() {
        info!("auth           : bearer-token enabled");
    }
    info!("---------------------------------------------------------");

    // -----------------------------------------------------------------------
    // 4. Create engine and start server
    // -----------------------------------------------------------------------
    let engine: Arc<dyn Engine> = if args.continuous_batching {
        info!("initialising BatchedEngine ...");
        Arc::new(BatchedEngine::new(engine_config).await?)
    } else {
        info!("initialising SimpleEngine ...");
        Arc::new(SimpleEngine::new(engine_config).await?)
    };

    info!("engine initialised, model loaded");
    info!(
        "starting HTTP server on {}:{} ...",
        server_config.host, server_config.port
    );

    rmlx_serve_api::serve(engine, server_config).await?;

    Ok(())
}
