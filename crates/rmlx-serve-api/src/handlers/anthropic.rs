//! `/v1/messages` handler (Anthropic Messages API adapter).

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

use rmlx_serve_types::anthropic::{
    AnthropicContentDelta, AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicRole,
    AnthropicStreamEvent, AnthropicUsage, StopReason,
};

use crate::convert::{
    anthropic_to_chat_request, chat_messages_to_template, chat_request_to_internal,
    chat_response_to_anthropic, internal_to_chat_response, unix_timestamp, validate_chat_request,
};
use crate::error::ApiError;
use crate::state::AppState;

/// Handle `POST /v1/messages` (Anthropic Messages API).
pub async fn anthropic_messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnthropicMessagesRequest>,
) -> Result<Response, ApiError> {
    state.request_count.fetch_add(1, Ordering::Relaxed);

    // Convert Anthropic request to an internal ChatCompletionRequest.
    let chat_req = anthropic_to_chat_request(&req);

    // Validate the converted request.
    validate_chat_request(&chat_req).map_err(ApiError::InvalidRequest)?;

    // Encode prompt.
    let template_messages = chat_messages_to_template(&chat_req.messages);
    let prompt = build_prompt_from_messages(&template_messages);
    let token_ids = state
        .engine
        .encode(&prompt)
        .map_err(|e| ApiError::EngineError(format!("tokenization failed: {e}")))?;

    let prompt_tokens = token_ids.len();
    let internal_request = chat_request_to_internal(&chat_req, token_ids);
    let model = req.model.clone();

    if req.stream.unwrap_or(false) {
        Ok(
            stream_anthropic_response(state, internal_request, model, prompt_tokens)
                .await
                .into_response(),
        )
    } else {
        // Non-streaming: generate, convert via OpenAI format, then to Anthropic.
        let output = state.engine.generate(internal_request).await?;
        let created = unix_timestamp();
        let chat_response = internal_to_chat_response(&output, &model, created, prompt_tokens, None, None);
        let anthropic_response = chat_response_to_anthropic(&chat_response);
        Ok(Json(anthropic_response).into_response())
    }
}

type SseStream = Pin<Box<dyn futures_core::Stream<Item = Result<Event, Infallible>> + Send>>;

/// Build SSE stream for Anthropic streaming format.
async fn stream_anthropic_response(
    state: Arc<AppState>,
    request: rmlx_serve_types::Request,
    model: String,
    prompt_tokens: usize,
) -> Sse<SseStream> {
    let receiver = match state.engine.generate_stream(request).await {
        Ok(rx) => rx,
        Err(e) => {
            let err_json = serde_json::to_string(&serde_json::json!({
                "type": "error",
                "error": { "type": "server_error", "message": e.to_string() }
            }))
            .unwrap_or_default();
            let stream: SseStream =
                Box::pin(tokio_stream::once(Ok(Event::default().data(err_json))));
            return Sse::new(stream);
        }
    };

    let rx_stream = UnboundedReceiverStream::new(receiver);

    // Emit message_start first.
    let message_start = AnthropicStreamEvent::MessageStart {
        message: AnthropicMessagesResponse {
            id: format!("msg_{}", uuid::Uuid::new_v4()),
            response_type: "message".to_string(),
            role: AnthropicRole::Assistant,
            content: vec![],
            model: model.clone(),
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: prompt_tokens,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        },
    };

    let start_json = serde_json::to_string(&message_start).unwrap_or_default();
    let start_event: Result<Event, Infallible> =
        Ok(Event::default().event("message_start").data(start_json));

    // Content block start.
    let block_start = serde_json::json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": { "type": "text", "text": "" }
    });
    let block_start_json = serde_json::to_string(&block_start).unwrap_or_default();
    let block_start_event: Result<Event, Infallible> = Ok(Event::default()
        .event("content_block_start")
        .data(block_start_json));

    let prefix_stream = futures_util::stream::iter(vec![start_event, block_start_event]);

    let mut total_output_tokens: usize = 0;

    let content_stream = rx_stream.flat_map(move |output| {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();

        for comp in &output.outputs {
            if !comp.text.is_empty() {
                let delta = AnthropicContentDelta::TextDelta {
                    text: comp.text.clone(),
                };
                let delta_event = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": delta,
                });
                let json = serde_json::to_string(&delta_event).unwrap_or_default();
                events.push(Ok(Event::default().event("content_block_delta").data(json)));
            }
            total_output_tokens += comp.token_ids.len();
        }

        if output.finished {
            // Content block stop.
            let block_stop = serde_json::json!({
                "type": "content_block_stop",
                "index": 0,
            });
            events.push(Ok(Event::default()
                .event("content_block_stop")
                .data(serde_json::to_string(&block_stop).unwrap_or_default())));

            // Message delta with stop reason.
            let stop_reason =
                output
                    .outputs
                    .first()
                    .and_then(|c| c.finish_reason)
                    .map(|r| match r {
                        rmlx_serve_types::FinishReason::Stop => StopReason::EndTurn,
                        rmlx_serve_types::FinishReason::Length => StopReason::MaxTokens,
                        rmlx_serve_types::FinishReason::ToolCall => StopReason::ToolUse,
                        rmlx_serve_types::FinishReason::Error => StopReason::EndTurn,
                    });

            let msg_delta = serde_json::json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": null,
                },
                "usage": {
                    "output_tokens": total_output_tokens,
                },
            });
            events.push(Ok(Event::default()
                .event("message_delta")
                .data(serde_json::to_string(&msg_delta).unwrap_or_default())));

            // Message stop.
            let msg_stop = serde_json::json!({ "type": "message_stop" });
            events.push(Ok(Event::default()
                .event("message_stop")
                .data(serde_json::to_string(&msg_stop).unwrap_or_default())));
        }

        futures_util::stream::iter(events)
    });

    let combined: SseStream = Box::pin(prefix_stream.chain(content_stream));
    Sse::new(combined)
}

/// Build a prompt string from template messages (same as in chat handler).
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
