//! Authentication middleware.
//!
//! When the server is configured with an `api_key`, this middleware checks
//! every request for a valid `Authorization: Bearer <key>` header.

use std::sync::Arc;

use axum::extract::State;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::Response;

use crate::error::ApiError;
use crate::state::AppState;

/// Axum middleware that validates the `Authorization` header when an API key
/// is configured in the server config.
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, ApiError> {
    if let Some(ref expected_key) = state.config.api_key {
        let auth_header = request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());

        match auth_header {
            Some(header) if header.starts_with("Bearer ") => {
                let provided_key = &header["Bearer ".len()..];
                if provided_key != expected_key.as_str() {
                    return Err(ApiError::Unauthorized("Invalid API key provided.".into()));
                }
            }
            _ => {
                return Err(ApiError::Unauthorized(
                    "Missing or malformed Authorization header. Expected: Bearer <api_key>".into(),
                ));
            }
        }
    }

    Ok(next.run(request).await)
}
