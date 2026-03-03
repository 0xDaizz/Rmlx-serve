//! `rmlx-serve-api` -- HTTP API server for rmlx-serve.
//!
//! This crate implements an OpenAI-compatible (and Anthropic-compatible) HTTP
//! API server, ported from the vllm-mlx `server.py`.  It exposes:
//!
//! - `POST /v1/chat/completions` -- chat completions (streaming + non-streaming)
//! - `POST /v1/completions` -- legacy text completions
//! - `GET  /v1/models` -- model listing
//! - `POST /v1/embeddings` -- embeddings (placeholder)
//! - `POST /v1/messages` -- Anthropic Messages API adapter
//! - `GET  /v1/mcp/tools` -- MCP tool listing (placeholder)
//! - `GET  /v1/mcp/servers` -- MCP server listing (placeholder)
//! - `GET  /health` -- health check
//! - `GET  /metrics` -- engine statistics
//!
//! # Usage
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use rmlx_serve_api::serve;
//! use rmlx_serve_types::config::ServerConfig;
//!
//! let engine: Arc<dyn rmlx_serve_engine::Engine> = /* ... */;
//! let config = ServerConfig::default();
//! serve(engine, config).await.unwrap();
//! ```

pub mod convert;
pub mod error;
pub mod handlers;
pub mod mcp;
pub mod middleware;
pub mod router;
pub mod sse;
pub mod state;

pub use error::ApiError;
pub use router::create_router;
pub use state::AppState;

use std::sync::Arc;

use rmlx_serve_engine::Engine;
use rmlx_serve_types::config::ServerConfig;

/// Start the HTTP API server.
///
/// This is the main entry point. It constructs the application state, builds
/// the Axum router with all routes and middleware, binds to the configured
/// host:port, and serves requests until the process is shut down.
pub async fn serve(engine: Arc<dyn Engine>, config: ServerConfig) -> Result<(), ApiError> {
    let state = AppState::new(engine, config.clone());
    let app = create_router(state);

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("rmlx-serve listening on {}", addr);

    axum::serve(listener, app)
        .await
        .map_err(|e| ApiError::InternalError(format!("server error: {e}")))?;

    Ok(())
}
