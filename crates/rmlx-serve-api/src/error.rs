//! API error types with OpenAI-compatible JSON error responses.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Errors that can occur in the API layer.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// The request body is malformed or contains invalid parameters.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// An error occurred in the inference engine.
    #[error("engine error: {0}")]
    EngineError(String),

    /// The request lacks valid authentication credentials.
    #[error("unauthorized: {0}")]
    Unauthorized(String),

    /// The requested resource was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// The client has exceeded the rate limit.
    #[error("rate limited: {0}")]
    RateLimited(String),

    /// An unexpected internal error.
    #[error("internal error: {0}")]
    InternalError(String),

    /// JSON serialization / deserialization failed.
    #[error("serialization error: {0}")]
    SerializationError(String),
}

impl ApiError {
    /// Map this error to an HTTP status code.
    fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            Self::EngineError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::RateLimited(_) => StatusCode::TOO_MANY_REQUESTS,
            Self::InternalError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::SerializationError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Map this error to an OpenAI-compatible error type string.
    fn error_type(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => "invalid_request_error",
            Self::EngineError(_) => "server_error",
            Self::Unauthorized(_) => "authentication_error",
            Self::NotFound(_) => "not_found_error",
            Self::RateLimited(_) => "rate_limit_error",
            Self::InternalError(_) => "server_error",
            Self::SerializationError(_) => "server_error",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = json!({
            "error": {
                "message": self.to_string(),
                "type": self.error_type(),
                "param": null,
                "code": null,
            }
        });

        (status, axum::Json(body)).into_response()
    }
}

impl From<std::io::Error> for ApiError {
    fn from(err: std::io::Error) -> Self {
        Self::InternalError(err.to_string())
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationError(err.to_string())
    }
}

impl From<rmlx_serve_engine::EngineError> for ApiError {
    fn from(err: rmlx_serve_engine::EngineError) -> Self {
        match &err {
            rmlx_serve_engine::EngineError::Request(msg) => {
                Self::InvalidRequest(msg.clone())
            }
            rmlx_serve_engine::EngineError::CapacityExceeded(msg) => {
                Self::RateLimited(msg.clone())
            }
            _ => Self::EngineError(err.to_string()),
        }
    }
}
