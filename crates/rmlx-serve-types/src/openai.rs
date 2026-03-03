//! OpenAI-compatible API request and response types.
//!
//! These types mirror the OpenAI REST API schema so that rmlx-serve can act as
//! a drop-in replacement for the OpenAI API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===========================================================================
// Shared enums
// ===========================================================================

/// The role of a participant in a chat conversation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A single part inside a multi-part content array.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text content.
    Text { text: String },
    /// An image specified by URL (or base64 data-URI).
    ImageUrl { image_url: ImageUrl },
    /// A video specified by URL.
    Video { video: MediaUrl },
    /// An audio clip specified by URL.
    Audio { audio: MediaUrl },
}

/// URL reference for an image, with optional detail level.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Generic URL reference for video / audio media.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MediaUrl {
    pub url: String,
}

/// The content of a chat message -- either a plain string or multipart array.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    /// Simple text content.
    Text(String),
    /// Multipart content (text, images, video, audio).
    Parts(Vec<ContentPart>),
}

// ===========================================================================
// Chat message
// ===========================================================================

/// A single message in a chat conversation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of the author of this message.
    pub role: ChatRole,

    /// The content of the message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatContent>,

    /// An optional name for the participant (used with tool role).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool calls requested by the assistant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// When role is `tool`, the id of the tool call this message responds to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

// ===========================================================================
// Tools
// ===========================================================================

/// A tool definition that the model may call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    /// Currently always `"function"`.
    #[serde(rename = "type")]
    pub tool_type: String,

    /// The function description.
    pub function: FunctionDefinition,
}

/// Description of a callable function exposed as a tool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// The name of the function.
    pub name: String,

    /// Human-readable description of what the function does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema describing the function parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,

    /// Whether to enable strict schema adherence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A tool invocation produced by the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique id for this tool call.
    pub id: String,

    /// Currently always `"function"`.
    #[serde(rename = "type")]
    pub call_type: String,

    /// The function name and serialised arguments.
    pub function: FunctionCall,
}

/// The function name + arguments inside a [`ToolCall`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function to invoke.
    pub name: String,

    /// JSON-encoded arguments.
    pub arguments: String,
}

/// How the model should choose tools.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// A keyword: `"none"`, `"auto"`, or `"required"`.
    Mode(String),
    /// Force a specific function.
    Function {
        #[serde(rename = "type")]
        choice_type: String,
        function: ToolChoiceFunction,
    },
}

/// Identifies a specific function when using [`ToolChoice::Function`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

// ===========================================================================
// Response format
// ===========================================================================

/// Specifies the desired response format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseFormat {
    /// `"text"` or `"json_object"` or `"json_schema"`.
    #[serde(rename = "type")]
    pub format_type: String,

    /// Optional JSON schema when `format_type` is `"json_schema"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
}

// ===========================================================================
// Usage
// ===========================================================================

/// Token usage statistics returned in API responses.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ===========================================================================
// Chat Completion (non-streaming)
// ===========================================================================

/// `POST /v1/chat/completions` request body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier.
    pub model: String,

    /// The conversation messages.
    pub messages: Vec<ChatMessage>,

    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling probability mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Number of completions to generate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,

    /// Whether to stream back partial progress via SSE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Stop sequences.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopCondition>,

    /// Maximum number of tokens to generate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,

    /// Presence penalty (-2.0 to 2.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty (-2.0 to 2.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Token logit biases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,

    /// User identifier for abuse detection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Tool definitions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// Tool selection strategy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Response format control.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Random seed for deterministic generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Whether to return log-probabilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// Number of top log-probs to return per token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<usize>,

    /// Options for streaming responses (e.g., include usage in final chunk).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
}

/// Options controlling streaming behavior.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamOptions {
    /// When `true`, the final streaming chunk includes a `usage` field with
    /// prompt/completion/total token counts.
    #[serde(default)]
    pub include_usage: bool,
}

/// Stop condition can be a single string or an array of strings.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopCondition {
    Single(String),
    Multiple(Vec<String>),
}

impl StopCondition {
    /// Flatten into a `Vec<String>` regardless of variant.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multiple(v) => v,
        }
    }
}

/// `POST /v1/chat/completions` response body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique response id (e.g. `chatcmpl-...`).
    pub id: String,

    /// Always `"chat.completion"`.
    pub object: String,

    /// Unix timestamp (seconds) of when the response was created.
    pub created: u64,

    /// Model used for this completion.
    pub model: String,

    /// Completion choices.
    pub choices: Vec<ChatChoice>,

    /// Token usage.
    pub usage: Usage,

    /// System fingerprint for reproducibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// A single choice in a chat completion response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatChoice {
    /// Zero-based index.
    pub index: usize,

    /// The assistant message.
    pub message: ChatMessage,

    /// Why generation stopped.
    pub finish_reason: Option<String>,

    /// Log-probability info.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Log-probability details for a choice.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChoiceLogprobs {
    /// Per-token log-probs.
    #[serde(default)]
    pub content: Option<Vec<TokenLogprobInfo>>,
}

/// Log-probability information for a single token.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenLogprobInfo {
    /// The token string.
    pub token: String,
    /// The log-probability.
    pub logprob: f32,
    /// Byte offsets (UTF-8).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    /// Top alternatives at this position.
    #[serde(default)]
    pub top_logprobs: Vec<TopLogprobEntry>,
}

/// A single entry in the top-logprobs list.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopLogprobEntry {
    pub token: String,
    pub logprob: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

// ===========================================================================
// Chat Completion Streaming (SSE chunks)
// ===========================================================================

/// A single SSE chunk for streaming chat completions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,

    /// Always `"chat.completion.chunk"`.
    pub object: String,

    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,

    /// Usage is optionally included in the final chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// A choice inside a streaming chat chunk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// The delta payload inside a streaming chat chunk choice.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChatDelta {
    /// Set in the first chunk to indicate the role.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<ChatRole>,

    /// Incremental text content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Incremental tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Incremental tool-call information in a streaming delta.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,

    /// Set in the first chunk for this tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
    pub call_type: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// Incremental function-call data inside a tool-call delta.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    /// Set in the first chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Incremental JSON arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ===========================================================================
// Text Completion (non-streaming)
// ===========================================================================

/// `POST /v1/completions` request body (legacy completions endpoint).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,

    /// The prompt to complete. Can be a string or token array.
    pub prompt: CompletionPrompt,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<usize>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopCondition>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Prompt input for the legacy completions endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Single(String),
    Multiple(Vec<String>),
    TokenIds(Vec<u32>),
    BatchTokenIds(Vec<Vec<u32>>),
}

/// `POST /v1/completions` response body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,

    /// Always `"text_completion"`.
    pub object: String,

    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// A single choice in a text completion response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogprobs>,
}

/// Log-probability info for text completions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionLogprobs {
    #[serde(default)]
    pub tokens: Vec<String>,
    #[serde(default)]
    pub token_logprobs: Vec<Option<f32>>,
    #[serde(default)]
    pub top_logprobs: Vec<Option<HashMap<String, f32>>>,
    #[serde(default)]
    pub text_offset: Vec<usize>,
}

// ===========================================================================
// Text Completion Streaming (SSE chunks)
// ===========================================================================

/// A single SSE chunk for streaming text completions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionChunk {
    pub id: String,

    /// Always `"text_completion"`.
    pub object: String,

    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// A choice inside a streaming text completion chunk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionChunkChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogprobs>,
}

// ===========================================================================
// Embeddings
// ===========================================================================

/// `POST /v1/embeddings` request body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,

    /// Input text(s) to embed.
    pub input: EmbeddingInput,

    /// Optional encoding format: `"float"` (default) or `"base64"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,

    /// Optional dimensionality for models that support it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<usize>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Input for the embeddings endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
    TokenIds(Vec<u32>),
    BatchTokenIds(Vec<Vec<u32>>),
}

/// `POST /v1/embeddings` response body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Always `"list"`.
    pub object: String,

    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// A single embedding vector in the response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingData {
    /// Always `"embedding"`.
    pub object: String,

    /// The embedding vector.
    pub embedding: Vec<f32>,

    /// Index in the input array.
    pub index: usize,
}

/// Usage statistics for embeddings.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}
