//! Request / response conversion utilities.
//!
//! This module converts between the OpenAI/Anthropic API types and the
//! internal engine types ([`Request`], [`RequestOutput`], etc.).

use std::time::{SystemTime, UNIX_EPOCH};

use rmlx_serve_types::anthropic::{
    AnthropicContent, AnthropicMessageContent, AnthropicMessagesRequest,
    AnthropicMessagesResponse, AnthropicRole, AnthropicSystemPrompt, AnthropicUsage, StopReason,
};
use rmlx_serve_tools::types::{ParsedToolCall, ReasoningParseResult, ToolCallParseResult};
use rmlx_serve_types::openai::{
    ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatContent, ChatDelta, ChatMessage, ChatRole, CompletionChoice,
    CompletionResponse, FunctionCall, FunctionCallDelta, StopCondition, ToolCall, ToolCallDelta,
    Usage,
};
use rmlx_serve_types::{FinishReason, Request, RequestOutput, SamplingParams};

use rmlx_serve_tokenizer::TemplateMessage;

// ---------------------------------------------------------------------------
// Timestamp helper
// ---------------------------------------------------------------------------

/// Current Unix timestamp in seconds.
pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Current time as fractional seconds (for `Request::arrival_time`).
pub fn arrival_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ---------------------------------------------------------------------------
// Chat messages -> TemplateMessage
// ---------------------------------------------------------------------------

/// Convert OpenAI-style [`ChatMessage`]s into [`TemplateMessage`]s for the
/// chat template renderer.
pub fn chat_messages_to_template(messages: &[ChatMessage]) -> Vec<TemplateMessage> {
    messages
        .iter()
        .map(|m| {
            let role = match m.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => "tool",
            };
            let content = match &m.content {
                Some(ChatContent::Text(t)) => t.clone(),
                Some(ChatContent::Parts(parts)) => {
                    // Concatenate text parts; skip non-text content for the
                    // template (the engine does not support images yet).
                    parts
                        .iter()
                        .filter_map(|p| {
                            if let rmlx_serve_types::openai::ContentPart::Text { text } = p {
                                Some(text.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("")
                }
                None => String::new(),
            };
            TemplateMessage {
                role: role.to_string(),
                content,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// SamplingParams from ChatCompletionRequest
// ---------------------------------------------------------------------------

/// Build [`SamplingParams`] from a [`ChatCompletionRequest`].
pub fn sampling_params_from_chat(req: &ChatCompletionRequest) -> SamplingParams {
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
    if let Some(true) = req.logprobs {
        params.logprobs = Some(req.top_logprobs.unwrap_or(5));
    }
    if let Some(ref bias) = req.logit_bias {
        // OpenAI uses string keys for token IDs; convert to u32.
        params.logit_bias = bias
            .iter()
            .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
            .collect();
    }

    params
}

// ---------------------------------------------------------------------------
// ChatCompletionRequest -> internal Request
// ---------------------------------------------------------------------------

/// Build an internal [`Request`] from a [`ChatCompletionRequest`] and
/// pre-encoded prompt token IDs.
pub fn chat_request_to_internal(
    req: &ChatCompletionRequest,
    token_ids: Vec<u32>,
) -> Request {
    let sampling = sampling_params_from_chat(req);
    let stream = req.stream.unwrap_or(false);

    let mut internal = Request::new(token_ids, sampling, arrival_time());
    internal.stream = stream;
    internal
}

// ---------------------------------------------------------------------------
// RequestOutput -> ChatCompletionResponse
// ---------------------------------------------------------------------------

/// Convert engine output to an OpenAI-format chat completion response.
///
/// If `tool_results` is provided (one per output), tool calls are populated
/// in the response message and `finish_reason` is set to `"tool_calls"`.
///
/// If `reasoning_results` is provided (one per output), the content text is
/// replaced with the reasoning-cleaned content (reasoning is currently logged
/// but not surfaced in the response -- a future extension can add a
/// `reasoning_content` field).
pub fn internal_to_chat_response(
    output: &RequestOutput,
    model: &str,
    created: u64,
    prompt_tokens: usize,
    tool_results: Option<&[ToolCallParseResult]>,
    reasoning_results: Option<&[ReasoningParseResult]>,
) -> ChatCompletionResponse {
    let choices: Vec<ChatChoice> = output
        .outputs
        .iter()
        .enumerate()
        .map(|(i, comp)| {
            // Determine content text -- start with generated text, cleaned of
            // special tokens.
            let raw_text = strip_special_tokens(&comp.text);

            // Apply reasoning extraction if available.
            let (content_text, _reasoning) = if let Some(rr) = reasoning_results.and_then(|r| r.get(i)) {
                (rr.content.clone(), rr.thinking.clone())
            } else {
                (raw_text.clone(), None)
            };

            // Apply tool-call extraction if available.
            let (final_content, tool_calls, finish) = if let Some(tr) = tool_results.and_then(|t| t.get(i)) {
                if !tr.tool_calls.is_empty() {
                    let oai_calls: Vec<ToolCall> = tr
                        .tool_calls
                        .iter()
                        .map(parsed_tool_call_to_openai)
                        .collect();
                    // Use the content remaining after tool-call extraction;
                    // if it is empty, omit it.
                    let remaining = tr.content.as_deref().unwrap_or("").trim();
                    let content = if remaining.is_empty() {
                        None
                    } else {
                        Some(ChatContent::Text(remaining.to_string()))
                    };
                    (content, Some(oai_calls), Some("tool_calls".to_string()))
                } else {
                    // No tool calls detected -- pass content through.
                    let content = if content_text.is_empty() {
                        None
                    } else {
                        Some(ChatContent::Text(content_text))
                    };
                    (content, None, comp.finish_reason.map(finish_reason_to_string))
                }
            } else {
                let content = if content_text.is_empty() {
                    None
                } else {
                    Some(ChatContent::Text(content_text))
                };
                (content, None, comp.finish_reason.map(finish_reason_to_string))
            };

            ChatChoice {
                index: comp.index,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: final_content,
                    name: None,
                    tool_calls,
                    tool_call_id: None,
                },
                finish_reason: finish,
                logprobs: None,
            }
        })
        .collect();

    let completion_tokens: usize = output
        .outputs
        .iter()
        .map(|c| c.token_ids.len())
        .sum();

    ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created,
        model: model.to_string(),
        choices,
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        system_fingerprint: Some("rmlx-serve".to_string()),
    }
}

// ---------------------------------------------------------------------------
// RequestOutput -> ChatCompletionChunk (streaming)
// ---------------------------------------------------------------------------

/// Build a streaming chunk for a single incremental output.
pub fn internal_to_chat_chunk(
    delta_text: &str,
    model: &str,
    created: u64,
    index: usize,
    request_id: &str,
    finish_reason: Option<FinishReason>,
    is_first: bool,
) -> ChatCompletionChunk {
    let delta = if is_first {
        ChatDelta {
            role: Some(ChatRole::Assistant),
            content: if delta_text.is_empty() {
                None
            } else {
                Some(delta_text.to_string())
            },
            tool_calls: None,
        }
    } else {
        ChatDelta {
            role: None,
            content: if delta_text.is_empty() {
                None
            } else {
                Some(delta_text.to_string())
            },
            tool_calls: None,
        }
    };

    ChatCompletionChunk {
        id: request_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index,
            delta,
            finish_reason: finish_reason.map(finish_reason_to_string),
            logprobs: None,
        }],
        usage: None,
        system_fingerprint: Some("rmlx-serve".to_string()),
    }
}

/// Build the final streaming chunk with usage info.
pub fn final_usage_chunk(
    model: &str,
    created: u64,
    request_id: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: request_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![],
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }),
        system_fingerprint: Some("rmlx-serve".to_string()),
    }
}

// ---------------------------------------------------------------------------
// CompletionResponse builder
// ---------------------------------------------------------------------------

/// Convert engine output to an OpenAI-format text completion response.
pub fn internal_to_completion_response(
    output: &RequestOutput,
    model: &str,
    created: u64,
    prompt_tokens: usize,
) -> CompletionResponse {
    let choices: Vec<CompletionChoice> = output
        .outputs
        .iter()
        .map(|comp| CompletionChoice {
            index: comp.index,
            text: comp.text.clone(),
            finish_reason: comp.finish_reason.map(finish_reason_to_string),
            logprobs: None,
        })
        .collect();

    let completion_tokens: usize = output
        .outputs
        .iter()
        .map(|c| c.token_ids.len())
        .sum();

    CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices,
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        system_fingerprint: Some("rmlx-serve".to_string()),
    }
}

// ---------------------------------------------------------------------------
// CompletionChunk builder (streaming text completions)
// ---------------------------------------------------------------------------

/// Build a streaming chunk for a single incremental text completion output.
pub fn internal_to_completion_chunk(
    delta_text: &str,
    model: &str,
    created: u64,
    index: usize,
    request_id: &str,
    finish_reason: Option<FinishReason>,
) -> rmlx_serve_types::openai::CompletionChunk {
    rmlx_serve_types::openai::CompletionChunk {
        id: request_id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![rmlx_serve_types::openai::CompletionChunkChoice {
            index,
            text: delta_text.to_string(),
            finish_reason: finish_reason.map(finish_reason_to_string),
            logprobs: None,
        }],
        usage: None,
        system_fingerprint: Some("rmlx-serve".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Anthropic <-> OpenAI conversions
// ---------------------------------------------------------------------------

/// Convert an [`AnthropicMessagesRequest`] to a [`ChatCompletionRequest`].
pub fn anthropic_to_chat_request(req: &AnthropicMessagesRequest) -> ChatCompletionRequest {
    let mut messages: Vec<ChatMessage> = Vec::new();

    // Insert system prompt as a system message if present.
    if let Some(ref system) = req.system {
        let system_text = match system {
            AnthropicSystemPrompt::Text(t) => t.clone(),
            AnthropicSystemPrompt::Blocks(blocks) => blocks
                .iter()
                .map(|b| match b {
                    rmlx_serve_types::anthropic::AnthropicSystemBlock::Text {
                        text, ..
                    } => text.as_str(),
                })
                .collect::<Vec<_>>()
                .join("\n"),
        };
        messages.push(ChatMessage {
            role: ChatRole::System,
            content: Some(ChatContent::Text(system_text)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Convert each Anthropic message.
    for msg in &req.messages {
        let role = match msg.role {
            AnthropicRole::User => ChatRole::User,
            AnthropicRole::Assistant => ChatRole::Assistant,
        };

        let content_text = match &msg.content {
            AnthropicMessageContent::Text(t) => t.clone(),
            AnthropicMessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    AnthropicContent::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        };

        messages.push(ChatMessage {
            role,
            content: Some(ChatContent::Text(content_text)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    let stop = req
        .stop_sequences
        .as_ref()
        .map(|seqs| StopCondition::Multiple(seqs.clone()));

    ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        n: None,
        stream: req.stream,
        stop,
        max_tokens: Some(req.max_tokens),
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: req
            .metadata
            .as_ref()
            .and_then(|m| m.user_id.clone()),
        tools: None,
        tool_choice: None,
        response_format: None,
        seed: None,
        logprobs: None,
        top_logprobs: None,
        stream_options: None,
    }
}

/// Convert a [`ChatCompletionResponse`] to an [`AnthropicMessagesResponse`].
pub fn chat_response_to_anthropic(
    resp: &ChatCompletionResponse,
) -> AnthropicMessagesResponse {
    let content: Vec<AnthropicContent> = resp
        .choices
        .iter()
        .filter_map(|choice| {
            choice.message.content.as_ref().map(|c| {
                let text = match c {
                    ChatContent::Text(t) => t.clone(),
                    ChatContent::Parts(parts) => parts
                        .iter()
                        .filter_map(|p| {
                            if let rmlx_serve_types::openai::ContentPart::Text { text } = p {
                                Some(text.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                };
                AnthropicContent::Text { text }
            })
        })
        .collect();

    let stop_reason = resp
        .choices
        .first()
        .and_then(|c| c.finish_reason.as_deref())
        .map(|r| match r {
            "stop" => StopReason::EndTurn,
            "length" => StopReason::MaxTokens,
            "tool_calls" => StopReason::ToolUse,
            _ => StopReason::EndTurn,
        });

    AnthropicMessagesResponse {
        id: format!("msg_{}", uuid::Uuid::new_v4()),
        response_type: "message".to_string(),
        role: AnthropicRole::Assistant,
        content,
        model: resp.model.clone(),
        stop_reason,
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: resp.usage.prompt_tokens,
            output_tokens: resp.usage.completion_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        },
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate a [`ChatCompletionRequest`] and return errors if any fields are
/// out of range.
pub fn validate_chat_request(req: &ChatCompletionRequest) -> Result<(), String> {
    if req.messages.is_empty() {
        return Err("messages must not be empty".to_string());
    }

    if let Some(t) = req.temperature {
        if !(0.0..=2.0).contains(&t) {
            return Err(format!(
                "temperature must be between 0.0 and 2.0, got {t}"
            ));
        }
    }

    if let Some(p) = req.top_p {
        if !(0.0..=1.0).contains(&p) {
            return Err(format!("top_p must be between 0.0 and 1.0, got {p}"));
        }
    }

    if let Some(n) = req.n {
        if n == 0 {
            return Err("n must be at least 1".to_string());
        }
        if n > 16 {
            return Err(format!("n must not exceed 16, got {n}"));
        }
    }

    if let Some(max) = req.max_tokens {
        if max == 0 {
            return Err("max_tokens must be at least 1".to_string());
        }
    }

    if let Some(pp) = req.presence_penalty {
        if !(-2.0..=2.0).contains(&pp) {
            return Err(format!(
                "presence_penalty must be between -2.0 and 2.0, got {pp}"
            ));
        }
    }

    if let Some(fp) = req.frequency_penalty {
        if !(-2.0..=2.0).contains(&fp) {
            return Err(format!(
                "frequency_penalty must be between -2.0 and 2.0, got {fp}"
            ));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Default parameter resolution (Fix 6)
// ---------------------------------------------------------------------------

/// Resolve a sampling parameter using the fallback chain:
/// request value > server config default > hardcoded fallback.
///
/// Returns `request_val` if `Some`, otherwise `config_val` if `Some`,
/// otherwise `fallback`.
pub fn resolve_param<T>(request_val: Option<T>, config_val: Option<T>, fallback: T) -> T {
    request_val.or(config_val).unwrap_or(fallback)
}

/// Apply server-config defaults to [`SamplingParams`] that were not set by
/// the request. This implements the parameter fallback chain:
///   request params > CLI defaults (ServerConfig) > hardcoded fallback.
pub fn apply_config_defaults(
    params: &mut SamplingParams,
    config: &rmlx_serve_types::config::ServerConfig,
    req_temperature: Option<f32>,
    req_top_p: Option<f32>,
    req_max_tokens: Option<usize>,
) {
    // Temperature: request > config.default_temperature > SamplingParams::default (1.0)
    if req_temperature.is_none() {
        if let Some(cfg_temp) = config.default_temperature {
            params.temperature = cfg_temp as f32;
        }
        // else keep SamplingParams::default value
    }

    // Top-p: request > config.default_top_p > SamplingParams::default (1.0)
    if req_top_p.is_none() {
        if let Some(cfg_top_p) = config.default_top_p {
            params.top_p = cfg_top_p as f32;
        }
    }

    // Max tokens: request > config.max_tokens > SamplingParams::default (256)
    if req_max_tokens.is_none() && config.max_tokens > 0 {
        params.max_tokens = config.max_tokens;
    }
}

/// Convert an internal [`FinishReason`] to the OpenAI string representation.
pub fn finish_reason_to_string(reason: FinishReason) -> String {
    match reason {
        FinishReason::Stop => "stop".to_string(),
        FinishReason::Length => "length".to_string(),
        FinishReason::ToolCall => "tool_calls".to_string(),
        FinishReason::Error => "stop".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Special token stripping
// ---------------------------------------------------------------------------

/// Common special / end-of-sequence tokens that should be stripped from
/// generation output before returning to the client.
const SPECIAL_TOKENS: &[&str] = &[
    "<|im_end|>",
    "<|eot_id|>",
    "<|end|>",
    "<|endoftext|>",
];

/// Strip common special tokens from generated text.
pub fn strip_special_tokens(text: &str) -> String {
    let mut result = text.to_string();
    for token in SPECIAL_TOKENS {
        result = result.replace(token, "");
    }
    result
}

// ---------------------------------------------------------------------------
// Tool call conversion helpers
// ---------------------------------------------------------------------------

/// Convert a [`ParsedToolCall`] into the OpenAI [`ToolCall`] type.
pub fn parsed_tool_call_to_openai(tc: &ParsedToolCall) -> ToolCall {
    ToolCall {
        id: tc.id.clone(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
        },
    }
}

/// Convert a [`rmlx_serve_tools::types::DeltaToolCall`] into the OpenAI
/// [`ToolCallDelta`] type.
pub fn delta_tool_call_to_openai(dtc: &rmlx_serve_tools::types::DeltaToolCall) -> ToolCallDelta {
    ToolCallDelta {
        index: dtc.index,
        id: dtc.id.clone(),
        call_type: dtc.id.as_ref().map(|_| "function".to_string()),
        function: if dtc.name.is_some() || dtc.arguments.is_some() {
            Some(FunctionCallDelta {
                name: dtc.name.clone(),
                arguments: dtc.arguments.clone(),
            })
        } else {
            None
        },
    }
}
