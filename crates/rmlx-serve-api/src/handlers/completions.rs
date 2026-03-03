//! `/v1/completions` handler (legacy text completions).

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;

use rmlx_serve_types::openai::{CompletionPrompt, CompletionRequest};
use rmlx_serve_types::{Request, SamplingParams};

use crate::convert::{
    apply_config_defaults, arrival_time, internal_to_completion_chunk,
    internal_to_completion_response, strip_special_tokens, unix_timestamp,
};
use crate::error::ApiError;
use crate::sse::SSE_DONE;
use crate::state::AppState;

/// Handle `POST /v1/completions`.
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
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

    // Build sampling params with default parameter resolution (Fix 6).
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

    // Apply server-config defaults for parameters not set by the request.
    apply_config_defaults(
        &mut params,
        &state.config,
        req.temperature,
        req.top_p,
        req.max_tokens,
    );

    let stream = req.stream.unwrap_or(false);
    let model = req.model.clone();

    // Build internal request.
    let mut internal = Request::new(token_ids, params, arrival_time());
    internal.stream = stream;

    if stream {
        // Streaming response (P0 Fix 1).
        Ok(stream_completion(state, internal, model, prompt_tokens)
            .await
            .into_response())
    } else {
        // Non-streaming: generate and return full response.
        let output = state.engine.generate(internal).await?;
        let created = unix_timestamp();
        let response = internal_to_completion_response(&output, &model, created, prompt_tokens);
        Ok(Json(response).into_response())
    }
}

type SseStream =
    Pin<Box<dyn futures_core::Stream<Item = Result<Event, Infallible>> + Send>>;

/// Build SSE stream for a streaming text completion.
///
/// Client disconnect detection (P0 Fix 2): When the client disconnects, Axum
/// drops the SSE response body, which drops this stream and closes the
/// receiver channel. The engine's sender will then observe a closed channel
/// and can stop generation. We additionally use `tokio::select!` inside the
/// stream to race between receiving the next token and a cancellation
/// signal, ensuring prompt cleanup.
async fn stream_completion(
    state: Arc<AppState>,
    request: Request,
    model: String,
    _prompt_tokens: usize,
) -> Sse<SseStream> {
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = unix_timestamp();

    // Get the streaming receiver from the engine.
    let receiver = match state.engine.generate_stream(request).await {
        Ok(rx) => rx,
        Err(e) => {
            let err_msg = format!("{{\"error\": \"{e}\"}}");
            let stream: SseStream =
                Box::pin(tokio_stream::once(Ok(Event::default().data(err_msg))));
            return Sse::new(stream);
        }
    };

    // P0 Fix 2: Use unfold + rx.recv() for explicit disconnect detection.
    // When the stream is dropped (client disconnect), the receiver is dropped,
    // closing the channel and signaling the engine to stop generation.
    let event_stream = futures_util::stream::unfold(
        (receiver, model, request_id, created),
        move |(mut rx, model, request_id, created)| async move {
            let output = rx.recv().await?;

            let mut events: Vec<Result<Event, Infallible>> = Vec::new();

            for comp in &output.outputs {
                let delta_text = strip_special_tokens(&comp.text);

                let finish = if output.finished {
                    comp.finish_reason
                } else {
                    None
                };

                if !delta_text.is_empty() || output.finished {
                    let chunk = internal_to_completion_chunk(
                        &delta_text,
                        &model,
                        created,
                        comp.index,
                        &request_id,
                        finish,
                    );
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    events.push(Ok(Event::default().data(json)));
                }
            }

            // If the output is finished, send the [DONE] sentinel.
            if output.finished {
                events.push(Ok(Event::default().data(SSE_DONE)));
                // Return events but signal end of stream on next iteration.
            }

            let batch = futures_util::stream::iter(events);
            Some((batch, (rx, model, request_id, created)))
        },
    )
    .flat_map(|batch| batch);

    let boxed: SseStream = Box::pin(event_stream);
    Sse::new(boxed)
}
