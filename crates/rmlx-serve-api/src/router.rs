//! Axum router construction.

use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::handlers;
use crate::middleware::auth_middleware;
use crate::state::AppState;

/// Build the complete Axum router with all routes, middleware, and layers.
pub fn create_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::permissive();
    let trace = TraceLayer::new_for_http();
    let timeout_secs = state.config.request_timeout_secs;

    // Core API routes.
    let api_routes = Router::new()
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/completions", post(handlers::completions))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/messages", post(handlers::anthropic_messages))
        .route("/v1/mcp/tools", get(handlers::mcp_list_tools))
        .route("/v1/mcp/servers", get(handlers::mcp_list_servers));

    // Conditionally apply auth middleware only if an API key is configured.
    let api_routes = if state.config.api_key.is_some() {
        api_routes.route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
    } else {
        api_routes
    };

    // Health and metrics endpoints are not behind auth.
    let infra_routes = Router::new()
        .route("/health", get(handlers::health_check))
        .route("/metrics", get(handlers::metrics));

    api_routes
        .merge(infra_routes)
        .layer(cors)
        .layer(trace)
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(timeout_secs),
        ))
        .with_state(state)
}
