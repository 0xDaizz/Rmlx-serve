//! `/v1/models` handler.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::state::AppState;

/// Handle `GET /v1/models`.
///
/// Returns an OpenAI-compatible model list containing the single model
/// served by this engine.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [
            {
                "id": state.engine.model_name(),
                "object": "model",
                "created": 0,
                "owned_by": "rmlx-serve"
            }
        ]
    }))
}
