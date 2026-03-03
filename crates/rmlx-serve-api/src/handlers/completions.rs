//! `/v1/completions` handler (legacy text completions).

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use rmlx_serve_types::openai::{CompletionPrompt, CompletionRequest, CompletionResponse};
use rmlx_serve_types::{Request, SamplingParams};

use crate::convert::{arrival_time, internal_to_completion_response, unix_timestamp};
use crate::error::ApiError;
use crate::state::AppState;

/// Handle `POST /v1/completions`.
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    state.request_count.fetch_add(1, Ordering::Relaxed);

    // Validate basic parameters.
    if let Some(t) = req.temperature {
        if !(0.0..=2.0).contains(&t) {
            return Err(ApiError::InvalidRequest(format!(
                "temperature must be between 0.0 and 2.0, got {t}"
            )));
        }
    }
    if let Some(p) = req.top_p {
        if !(0.0..=1.0).contains(&p) {
            return Err(ApiError::InvalidRequest(format!(
                "top_p must be between 0.0 and 1.0, got {p}"
            )));
        }
    }

    // Extract prompt text and encode.
    let prompt_text = match &req.prompt {
        CompletionPrompt::Single(s) => s.clone(),
        CompletionPrompt::Multiple(v) => v.join(""),
        CompletionPrompt::TokenIds(_) | CompletionPrompt::BatchTokenIds(_) => {
            return Err(ApiError::InvalidRequest(
                "Token ID prompts are not yet supported for /v1/completions".into(),
            ));
        }
    };

    let token_ids = state
        .engine
        .encode(&prompt_text)
        .map_err(|e| ApiError::EngineError(format!("tokenization failed: {e}")))?;

    let prompt_tokens = token_ids.len();

    // Build sampling params.
    let mut params = SamplingParams::default();
    if let Some(t) = req.temperature {
        params.temperature = t;
    }
    if let Some(p) = req.top_p {
        params.top_p = p;
    }
    if let Some(n) = req.n {
        params.n = n;
    }
    if let Some(max) = req.max_tokens {
        params.max_tokens = max;
    }
    if let Some(pp) = req.presence_penalty {
        params.presence_penalty = pp;
    }
    if let Some(fp) = req.frequency_penalty {
        params.frequency_penalty = fp;
    }
    if let Some(ref stop) = req.stop {
        params.stop = stop.clone().into_vec();
    }
    if let Some(seed) = req.seed {
        params.seed = Some(seed);
    }

    // Build internal request.
    let internal = Request::new(token_ids, params, arrival_time());

    // Generate.
    let output = state.engine.generate(internal).await?;
    let created = unix_timestamp();
    let response = internal_to_completion_response(&output, &req.model, created, prompt_tokens);

    Ok(Json(response))
}
