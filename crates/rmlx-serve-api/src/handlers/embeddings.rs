//! `/v1/embeddings` handler (placeholder).

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use rmlx_serve_types::openai::{EmbeddingRequest, EmbeddingResponse};

use crate::error::ApiError;
use crate::state::AppState;

/// Handle `POST /v1/embeddings`.
///
/// Currently returns an error since embeddings are not yet supported.
pub async fn embeddings(
    State(_state): State<Arc<AppState>>,
    Json(_req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    Err(ApiError::InvalidRequest(
        "Embeddings are not yet supported by this server.".into(),
    ))
}
