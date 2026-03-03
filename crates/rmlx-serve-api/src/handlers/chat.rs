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

use rmlx_serve_types::openai::{ChatCompletionRequest, ChatContent, ChatMessage, ChatRole};

use crate::convert::{
    apply_config_defaults, chat_messages_to_template, chat_request_to_internal,
    delta_tool_call_to_openai, final_usage_chunk, internal_to_chat_chunk,
    internal_to_chat_response, strip_special_tokens, unix_timestamp, validate_chat_request,
};
use crate::error::ApiError;
use crate::sse::SSE_DONE;
use crate::state::AppState;

/// Handle `POST /v1/chat/completions`.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(mut req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    // Bump request counter.
    state.request_count.fetch_add(1, Ordering::Relaxed);

    // 1. Validate request parameters.
    validate_chat_request(&req).map_err(ApiError::InvalidRequest)?;

    // P1 Fix 5: Structured output (response_format).
    // Inject system-level instructions when the client requests JSON output.
    apply_response_format_hint(&mut req);

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
    let mut internal_request = chat_request_to_internal(&req, token_ids);

    // P1 Fix 6: Apply server-config defaults for parameters not set by the request.
    apply_config_defaults(
        &mut internal_request.sampling_params,
        &state.config,
        req.temperature,
        req.top_p,
        req.max_tokens,
    );

    let model = req.model.clone();

    // Resolve the configured tool-call / reasoning parser names from AppState.
    let tool_parser_name: Option<String> = state.tool_parser_name.clone();
    let reasoning_parser_name: Option<String> = state.reasoning_parser_name.clone();

    // P1 Fix 4: Check if the client wants usage stats in the final streaming chunk.
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|opts| opts.include_usage);

    // 4. Streaming vs non-streaming.
    if req.stream.unwrap_or(false) {
        Ok(stream_chat_completion(
            state,
            internal_request,
            model,
            prompt_tokens,
            tool_parser_name,
            reasoning_parser_name,
            include_usage,
        )
        .await
        .into_response())
    } else {
        // Non-streaming: generate and return full response.
        let output = state.engine.generate(internal_request).await?;

        // Apply tool-call parsing to each output.
        let tool_results = tool_parser_name.as_deref().and_then(|name| {
            state.tool_parser_registry.get(name).map(|parser| {
                output
                    .outputs
                    .iter()
                    .map(|comp| {
                        let cleaned = strip_special_tokens(&comp.text);
                        parser.parse(&cleaned)
                    })
                    .collect::<Vec<_>>()
            })
        });

        // Apply reasoning parsing to each output.
        let reasoning_results = reasoning_parser_name.as_deref().and_then(|name| {
            state.reasoning_parser_registry.get(name).map(|parser| {
                output
                    .outputs
                    .iter()
                    .map(|comp| {
                        let cleaned = strip_special_tokens(&comp.text);
                        parser.parse(&cleaned)
                    })
                    .collect::<Vec<_>>()
            })
        });

        let created = unix_timestamp();
        let response = internal_to_chat_response(
            &output,
            &model,
            created,
            prompt_tokens,
            tool_results.as_deref(),
            reasoning_results.as_deref(),
        );
        Ok(Json(response).into_response())
    }
}

type SseStream =
    Pin<Box<dyn futures_core::Stream<Item = Result<Event, Infallible>> + Send>>;

/// Mutable state carried across streaming iterations via `unfold`.
struct ChatStreamState {
    rx: tokio::sync::mpsc::UnboundedReceiver<rmlx_serve_types::RequestOutput>,
    model: String,
    request_id: String,
    created: u64,
    prompt_tokens: usize,
    include_usage: bool,
    tool_parser: Option<Box<dyn rmlx_serve_tools::ToolCallParser>>,
    reasoning_parser: Option<Box<dyn rmlx_serve_tools::ReasoningParser>>,
    accumulated: std::collections::HashMap<usize, String>,
    is_first: bool,
    total_completion_tokens: usize,
}

/// Build SSE stream for a streaming chat completion.
///
/// P0 Fix 2 (disconnect detection): The stream is driven by
/// `futures_util::stream::unfold` calling `rx.recv()`. When the client
/// disconnects, Axum drops the SSE response body, which drops this stream
/// and closes the receiver channel. The engine observes the closed channel
/// and stops generation.
///
/// P1 Fix 4 (stream_options.include_usage): When `include_usage` is true,
/// an additional chunk with usage statistics is sent before `[DONE]`.
async fn stream_chat_completion(
    state: Arc<AppState>,
    request: rmlx_serve_types::Request,
    model: String,
    prompt_tokens: usize,
    tool_parser_name: Option<String>,
    reasoning_parser_name: Option<String>,
    include_usage: bool,
) -> Sse<SseStream> {
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = unix_timestamp();

    // Create mutable streaming parsers if configured.
    let tool_parser = tool_parser_name
        .as_deref()
        .and_then(|name| state.tool_parser_registry.get(name));
    let reasoning_parser = reasoning_parser_name
        .as_deref()
        .and_then(|name| state.reasoning_parser_registry.get(name));

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

    let init_state = ChatStreamState {
        rx: receiver,
        model,
        request_id,
        created,
        prompt_tokens,
        include_usage,
        tool_parser,
        reasoning_parser,
        accumulated: std::collections::HashMap::new(),
        is_first: true,
        total_completion_tokens: 0,
    };

    // P0 Fix 2: Use unfold + rx.recv() for explicit disconnect detection.
    // When the stream is dropped (client disconnect), the receiver is dropped,
    // closing the channel and signaling the engine to stop.
    let event_stream = futures_util::stream::unfold(init_state, |mut st| async move {
        let output = st.rx.recv().await?;

        let mut events: Vec<Result<Event, Infallible>> = Vec::new();

        for comp in &output.outputs {
            let raw_delta = strip_special_tokens(&comp.text);

            // Track completion tokens for usage reporting.
            st.total_completion_tokens += comp.token_ids.len();

            let finish = if output.finished {
                comp.finish_reason
            } else {
                None
            };

            // --- Reasoning parser (streaming) ---
            let delta_after_reasoning = if let Some(ref mut rp) = st.reasoning_parser {
                match rp.parse_streaming(&raw_delta) {
                    Some(rr) => rr.content,
                    None => String::new(),
                }
            } else {
                raw_delta.clone()
            };

            // --- Tool parser (streaming) ---
            if let Some(ref mut tp) = st.tool_parser {
                let prev = st.accumulated.get(&comp.index).cloned().unwrap_or_default();
                let curr = format!("{}{}", prev, delta_after_reasoning);
                let sp = tp.parse_streaming(&prev, &curr, &delta_after_reasoning);
                *st.accumulated.entry(comp.index).or_default() = curr;

                // Emit content delta if present.
                if let Some(ref content_delta) = sp.content {
                    if !content_delta.is_empty() {
                        let chunk = internal_to_chat_chunk(
                            content_delta,
                            &st.model,
                            st.created,
                            comp.index,
                            &st.request_id,
                            if sp.tool_calls.is_empty() { finish } else { None },
                            st.is_first,
                        );
                        st.is_first = false;
                        let json = serde_json::to_string(&chunk).unwrap_or_default();
                        events.push(Ok(Event::default().data(json)));
                    }
                }

                // Emit tool-call deltas if present.
                if !sp.tool_calls.is_empty() {
                    let tc_deltas: Vec<rmlx_serve_types::openai::ToolCallDelta> = sp
                        .tool_calls
                        .iter()
                        .map(delta_tool_call_to_openai)
                        .collect();

                    let tc_finish = if sp.finished { Some("tool_calls") } else { None };

                    let chunk = rmlx_serve_types::openai::ChatCompletionChunk {
                        id: st.request_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: st.created,
                        model: st.model.clone(),
                        choices: vec![rmlx_serve_types::openai::ChatChunkChoice {
                            index: comp.index,
                            delta: rmlx_serve_types::openai::ChatDelta {
                                role: if st.is_first {
                                    Some(rmlx_serve_types::openai::ChatRole::Assistant)
                                } else {
                                    None
                                },
                                content: None,
                                tool_calls: Some(tc_deltas),
                            },
                            finish_reason: tc_finish.map(|s| s.to_string()),
                            logprobs: None,
                        }],
                        usage: None,
                        system_fingerprint: Some("rmlx-serve".to_string()),
                    };
                    st.is_first = false;
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    events.push(Ok(Event::default().data(json)));
                }
            } else {
                // No tool parser -- emit the delta directly.
                if !delta_after_reasoning.is_empty() || output.finished {
                    let chunk = internal_to_chat_chunk(
                        &delta_after_reasoning,
                        &st.model,
                        st.created,
                        comp.index,
                        &st.request_id,
                        finish,
                        st.is_first,
                    );
                    st.is_first = false;
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    events.push(Ok(Event::default().data(json)));
                }
            }
        }

        // If the output is finished:
        if output.finished {
            // P1 Fix 4: Send usage chunk if include_usage was requested.
            if st.include_usage {
                let usage_chunk = final_usage_chunk(
                    &st.model,
                    st.created,
                    &st.request_id,
                    st.prompt_tokens,
                    st.total_completion_tokens,
                );
                let json = serde_json::to_string(&usage_chunk).unwrap_or_default();
                events.push(Ok(Event::default().data(json)));
            }

            // Send the [DONE] sentinel.
            events.push(Ok(Event::default().data(SSE_DONE)));
        }

        Some((futures_util::stream::iter(events), st))
    })
    .flat_map(|batch| batch);

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

// ---------------------------------------------------------------------------
// P1 Fix 5: Structured output (response_format)
// ---------------------------------------------------------------------------

/// Inject system-prompt hints based on `response_format`.
///
/// - `{"type": "json_object"}` -- prepend a system message instructing the
///   model to respond with valid JSON.
/// - `{"type": "json_schema", "json_schema": {...}}` -- prepend a system
///   message that includes the schema the output must conform to.
/// - `{"type": "text"}` or absent -- no-op.
fn apply_response_format_hint(req: &mut ChatCompletionRequest) {
    let format = match &req.response_format {
        Some(f) => f,
        None => return,
    };

    let hint = match format.format_type.as_str() {
        "json_object" => {
            "You must respond with valid JSON. Do not include any text outside the JSON object."
                .to_string()
        }
        "json_schema" => {
            if let Some(ref schema) = format.json_schema {
                format!(
                    "You must respond with valid JSON that conforms to the following JSON Schema:\n\n```json\n{}\n```\n\nDo not include any text outside the JSON object.",
                    serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string())
                )
            } else {
                "You must respond with valid JSON. Do not include any text outside the JSON object."
                    .to_string()
            }
        }
        _ => return, // "text" or unknown -- no hint needed.
    };

    // Prepend or merge into the first system message.
    let has_system = req
        .messages
        .first()
        .is_some_and(|m| m.role == ChatRole::System);

    if has_system {
        // Append the hint to the existing system message.
        if let Some(ref mut content) = req.messages[0].content {
            match content {
                ChatContent::Text(ref mut t) => {
                    t.push_str("\n\n");
                    t.push_str(&hint);
                }
                ChatContent::Parts(_) => {
                    // For multi-part content, add a new text system message instead.
                    req.messages.insert(
                        1,
                        ChatMessage {
                            role: ChatRole::System,
                            content: Some(ChatContent::Text(hint)),
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                        },
                    );
                }
            }
        }
    } else {
        // Insert a new system message at the front.
        req.messages.insert(
            0,
            ChatMessage {
                role: ChatRole::System,
                content: Some(ChatContent::Text(hint)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        );
    }
}
