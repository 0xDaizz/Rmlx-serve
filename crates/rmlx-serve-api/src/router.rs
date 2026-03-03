//! Axum router construction.

use std::sync::Arc;
use std::time::Duration;

use axum::http::header::HeaderValue;
use axum::http::Method;
use axum::http::StatusCode;
use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::handlers;
use crate::middleware::{auth_middleware, rate_limit_middleware};
use crate::state::AppState;

/// Build the complete Axum router with all routes, middleware, and layers.
pub fn create_router(state: Arc<AppState>) -> Router {
    let cors = build_cors_layer(&state.config.cors_allowed_origins);
    let trace = TraceLayer::new_for_http();
    let timeout_secs = state.config.request_timeout_secs;
    let max_connections = state.config.max_connections;

    // Core API routes.
    let mut api_routes = Router::new()
        .route("/v1/mcp/tools", post(handlers::mcp_list_tools))
        .route("/v1/mcp/execute", post(handlers::mcp_execute_tool))
        .route("/v1/mcp/servers", get(handlers::mcp_list_servers));

    if state.config.enable_openai_api {
        api_routes = api_routes
            .route("/v1/chat/completions", post(handlers::chat_completions))
            .route("/v1/completions", post(handlers::completions))
            .route("/v1/models", get(handlers::list_models))
            .route("/v1/embeddings", post(handlers::embeddings));
    }

    if state.config.enable_anthropic_api {
        api_routes = api_routes.route("/v1/messages", post(handlers::anthropic_messages));
    }

    // Conditionally apply auth middleware only if an API key is configured.
    let api_routes = if state.config.api_key.is_some() {
        api_routes.route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
    } else {
        api_routes
    };

    // Conditionally apply rate-limiting middleware when rate_limit > 0.
    let api_routes = if state.config.rate_limit > 0 {
        api_routes.route_layer(middleware::from_fn_with_state(
            state.clone(),
            rate_limit_middleware,
        ))
    } else {
        api_routes
    };

    // Health endpoint is always enabled.
    let mut infra_routes = Router::new().route("/health", get(handlers::health_check));
    if state.config.enable_metrics {
        infra_routes = infra_routes.route("/metrics", get(handlers::metrics));
    }

    let app = api_routes
        .merge(infra_routes)
        .layer(cors)
        .layer(trace)
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(timeout_secs),
        ));

    if max_connections > 0 {
        app.layer(ConcurrencyLimitLayer::new(max_connections))
            .with_state(state)
    } else {
        app.with_state(state)
    }
}

fn build_cors_layer(origins: &[String]) -> CorsLayer {
    let base = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    if origins.is_empty() {
        return base;
    }

    if origins.iter().any(|o| o == "*") {
        return base.allow_origin(Any);
    }

    let mut parsed = Vec::new();
    for origin in origins {
        match HeaderValue::from_str(origin) {
            Ok(v) => parsed.push(v),
            Err(_) => tracing::warn!(origin = %origin, "invalid CORS origin ignored"),
        }
    }

    if parsed.is_empty() {
        base
    } else {
        base.allow_origin(parsed)
    }
}
