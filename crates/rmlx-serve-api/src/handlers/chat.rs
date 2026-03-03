//! `/v1/chat/completions` handler.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use rmlx_serve_types::openai::ChatCompletionRequest;

use crate::convert::{
    chat_messages_to_template, chat_request_to_internal, internal_to_chat_chunk,
    internal_to_chat_response, unix_timestamp, validate_chat_request,
};
use crate::error::ApiError;
use crate::sse::SSE_DONE;
use crate::state::AppState;

/// Handle `POST /v1/chat/completions`.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    // Bump request counter.
    state.request_count.fetch_add(1, Ordering::Relaxed);

    // 1. Validate request parameters.
    validate_chat_request(&req).map_err(ApiError::InvalidRequest)?;

    // 2. Convert messages to a prompt string via chat template then encode.
    let template_messages = chat_messages_to_template(&req.messages);

    // Build a simple ChatML-like prompt from the template messages.
    // In production this would use the model's Jinja2 chat template.
    let prompt = build_prompt_from_messages(&template_messages);

    let token_ids = state
        .engine
        .encode(&prompt)
        .map_err(|e| ApiError::EngineError(format!("tokenization failed: {e}")))?;

    let prompt_tokens = token_ids.len();

    // 3. Build internal request.
    let internal_request = chat_request_to_internal(&req, token_ids);
    let model = req.model.clone();

    // 4. Streaming vs non-streaming.
    if req.stream.unwrap_or(false) {
        Ok(
            stream_chat_completion(state, internal_request, model, prompt_tokens)
                .await
                .into_response(),
        )
    } else {
        // Non-streaming: generate and return full response.
        let output = state.engine.generate(internal_request).await?;
        let created = unix_timestamp();
        let response = internal_to_chat_response(&output, &model, created, prompt_tokens);
        Ok(Json(response).into_response())
    }
}

type SseStream = Pin<Box<dyn futures_core::Stream<Item = Result<Event, Infallible>> + Send>>;

/// Build SSE stream for a streaming chat completion.
async fn stream_chat_completion(
    state: Arc<AppState>,
    request: rmlx_serve_types::Request,
    model: String,
    _prompt_tokens: usize,
) -> Sse<SseStream> {
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = unix_timestamp();

    // Get the streaming receiver from the engine.
    let receiver = match state.engine.generate_stream(request).await {
        Ok(rx) => rx,
        Err(e) => {
            // If we fail to start streaming, emit an error event then close.
            let err_msg = format!("{{\"error\": \"{e}\"}}");
            let stream: SseStream =
                Box::pin(tokio_stream::once(Ok(Event::default().data(err_msg))));
            return Sse::new(stream);
        }
    };

    let rx_stream = UnboundedReceiverStream::new(receiver);

    let mut is_first = true;
    let event_stream = rx_stream.flat_map(move |output| {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();

        for comp in &output.outputs {
            let delta_text = &comp.text;
            let finish = if output.finished {
                comp.finish_reason
            } else {
                None
            };

            let chunk = internal_to_chat_chunk(
                delta_text,
                &model,
                created,
                comp.index,
                &request_id,
                finish,
                is_first,
            );

            is_first = false;

            let json = serde_json::to_string(&chunk).unwrap_or_default();
            events.push(Ok(Event::default().data(json)));
        }

        // If the output is finished, send the [DONE] sentinel.
        if output.finished {
            events.push(Ok(Event::default().data(SSE_DONE)));
        }

        futures_util::stream::iter(events)
    });

    let boxed: SseStream = Box::pin(event_stream);
    Sse::new(boxed)
}

/// Build a prompt string from template messages.
///
/// This is a simple fallback that constructs a ChatML-style prompt.
/// In a full deployment, the engine's chat template would be used instead.
fn build_prompt_from_messages(messages: &[rmlx_serve_tokenizer::TemplateMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}
