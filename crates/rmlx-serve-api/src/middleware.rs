//! Authentication and rate-limiting middleware.
//!
//! - **Auth**: When the server is configured with an `api_key`, the
//!   [`auth_middleware`] checks every request for a valid
//!   `Authorization: Bearer <key>` header.
//!
//! - **Rate limiting**: When `rate_limit > 0` in [`ServerConfig`], the
//!   [`rate_limit_middleware`] enforces a sliding-window limit per client IP,
//!   returning 429 Too Many Requests when exceeded.

use std::net::IpAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{ConnectInfo, State};
use axum::http::Request;
use axum::middleware::Next;
use axum::response::Response;
use dashmap::DashMap;
use subtle::ConstantTimeEq;

use crate::error::ApiError;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Authentication middleware
// ---------------------------------------------------------------------------

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
                if !bool::from(provided_key.as_bytes().ct_eq(expected_key.as_bytes())) {
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

// ---------------------------------------------------------------------------
// Rate limiting middleware (P1 Fix 3)
// ---------------------------------------------------------------------------

/// Per-client sliding-window rate limiter state.
///
/// Each entry stores `(request_count, window_start)`.  When a new request
/// arrives:
///   1. If the window has expired (> 1 second old), reset the counter.
///   2. If the counter is below the limit, increment and allow.
///   3. Otherwise, return 429.
pub struct RateLimiter {
    /// Map from client IP to (count, window_start).
    clients: DashMap<IpAddr, (usize, Instant)>,
    /// Maximum requests per second (per client IP). 0 = unlimited.
    limit: usize,
}

impl RateLimiter {
    /// Create a new rate limiter with the given per-second limit.
    pub fn new(limit: usize) -> Self {
        Self {
            clients: DashMap::new(),
            limit,
        }
    }

    /// Check whether a request from `ip` should be allowed.
    /// Returns `true` if allowed, `false` if rate-limited.
    pub fn check(&self, ip: IpAddr) -> bool {
        if self.limit == 0 {
            return true;
        }

        let now = Instant::now();
        let mut entry = self.clients.entry(ip).or_insert((0, now));
        let (ref mut count, ref mut window_start) = *entry;

        // If the window has elapsed (> 1 second), reset.
        if now.duration_since(*window_start).as_secs_f64() >= 1.0 {
            *count = 1;
            *window_start = now;
            return true;
        }

        if *count < self.limit {
            *count += 1;
            true
        } else {
            false
        }
    }
}

/// Axum middleware that enforces per-IP rate limiting using a sliding window.
///
/// When `rate_limit > 0` in the server config, this middleware rejects
/// requests that exceed the limit with 429 Too Many Requests.
pub async fn rate_limit_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, ApiError> {
    if state.config.rate_limit > 0 {
        // Try to extract the client IP from ConnectInfo, falling back to
        // optionally trusting X-Forwarded-For only when explicitly enabled.
        let connect_ip = request
            .extensions()
            .get::<ConnectInfo<std::net::SocketAddr>>()
            .map(|ci| ci.0.ip());

        let forwarded_ip = if state.config.trust_x_forwarded_for {
            request
                .headers()
                .get("x-forwarded-for")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.split(',').next())
                .and_then(|s| s.trim().parse::<IpAddr>().ok())
        } else {
            None
        };

        let client_ip = connect_ip
            .or(forwarded_ip)
            .unwrap_or(IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED));

        if !state.rate_limiter.check(client_ip) {
            return Err(ApiError::RateLimited(format!(
                "Rate limit exceeded ({} requests/s). Please slow down.",
                state.config.rate_limit
            )));
        }
    }

    Ok(next.run(request).await)
}
