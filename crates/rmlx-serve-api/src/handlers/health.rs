//! `/health` and `/metrics` handlers.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use rmlx_serve_engine::{EngineHealth, EngineStats};

use crate::state::AppState;

/// Handle `GET /health`.
///
/// Returns the engine's current health status.
pub async fn health_check(State(state): State<Arc<AppState>>) -> Json<EngineHealth> {
    Json(state.engine.health().await)
}

/// Handle `GET /metrics`.
///
/// Returns aggregate engine statistics.
pub async fn metrics(State(state): State<Arc<AppState>>) -> Json<EngineStats> {
    Json(state.engine.get_stats())
}
