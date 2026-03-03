//! Anthropic Messages API request and response types.
//!
//! These types mirror the Anthropic Messages API schema so that rmlx-serve
//! can also serve requests from Anthropic-compatible clients.

use serde::{Deserialize, Serialize};

// ===========================================================================
// Roles and stop reasons
// ===========================================================================

/// Role for Anthropic messages.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    User,
    Assistant,
}

/// Why a response stopped.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model reached a natural stopping point.
    EndTurn,
    /// A stop sequence was matched.
    StopSequence,
    /// The `max_tokens` limit was hit.
    MaxTokens,
    /// The model wants to use a tool.
    ToolUse,
}

// ===========================================================================
// Content blocks
// ===========================================================================

/// A content block inside an Anthropic message.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContent {
    /// Plain text content.
    Text { text: String },

    /// A tool-use request from the assistant.
    ToolUse {
        /// Unique id for this tool invocation.
        id: String,
        /// Tool name.
        name: String,
        /// JSON-encoded input arguments.
        input: serde_json::Value,
    },

    /// A tool result sent by the user.
    ToolResult {
        /// The `tool_use` id this result corresponds to.
        tool_use_id: String,
        /// The content returned by the tool.
        content: ToolResultContent,
        /// Whether the tool invocation was an error.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },

    /// Extended thinking content (chain-of-thought).
    Thinking {
        /// The thinking text.
        thinking: String,
        /// Opaque signature for verification.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

/// Tool result content can be a simple string or structured content blocks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    /// Simple text result.
    Text(String),
    /// Structured content blocks.
    Blocks(Vec<ToolResultBlock>),
}

/// A block inside a structured tool result.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    Text { text: String },
    Image { source: ImageSource },
}

/// Source for an inline image in a tool result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageSource {
    /// Always `"base64"`.
    #[serde(rename = "type")]
    pub source_type: String,
    /// MIME type, e.g. `"image/png"`.
    pub media_type: String,
    /// Base64-encoded image data.
    pub data: String,
}

// ===========================================================================
// Messages
// ===========================================================================

/// A single message in an Anthropic conversation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: AnthropicRole,
    pub content: AnthropicMessageContent,
}

/// Message content: either a plain string or an array of content blocks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    /// Simple text.
    Text(String),
    /// Structured content blocks.
    Blocks(Vec<AnthropicContent>),
}

// ===========================================================================
// Tool definitions
// ===========================================================================

/// An Anthropic tool definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicTool {
    /// The tool name.
    pub name: String,

    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema for the input parameters.
    pub input_schema: serde_json::Value,
}

/// How the model should handle tool use.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    /// Let the model decide.
    Auto,
    /// The model must call at least one tool.
    Any,
    /// Force a specific tool.
    Tool { name: String },
}

// ===========================================================================
// Request
// ===========================================================================

/// `POST /v1/messages` request body (Anthropic Messages API).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicMessagesRequest {
    /// Model identifier.
    pub model: String,

    /// The conversation messages.
    pub messages: Vec<AnthropicMessage>,

    /// Maximum tokens to generate.
    pub max_tokens: usize,

    /// Optional system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<AnthropicSystemPrompt>,

    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus (top-p) sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Stop sequences.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Whether to stream the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Tool definitions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,

    /// Tool choice strategy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,

    /// Metadata for the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata>,
}

/// System prompt can be a string or structured content blocks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystemPrompt {
    Text(String),
    Blocks(Vec<AnthropicSystemBlock>),
}

/// A block in a structured system prompt.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicSystemBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// Cache control hints for Anthropic's prompt caching.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub control_type: String,
}

/// Metadata attached to a request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicMetadata {
    /// User identifier for abuse detection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

// ===========================================================================
// Response
// ===========================================================================

/// `POST /v1/messages` response body (Anthropic Messages API).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicMessagesResponse {
    /// Unique response id (e.g. `msg_...`).
    pub id: String,

    /// Always `"message"`.
    #[serde(rename = "type")]
    pub response_type: String,

    /// Always `"assistant"`.
    pub role: AnthropicRole,

    /// The response content blocks.
    pub content: Vec<AnthropicContent>,

    /// The model that generated this response.
    pub model: String,

    /// Why generation stopped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,

    /// Which stop sequence was matched, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,

    /// Token usage statistics.
    pub usage: AnthropicUsage,
}

/// Token usage for Anthropic responses.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,

    /// Tokens read from cache (prompt caching).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<usize>,

    /// Tokens served from cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<usize>,
}

// ===========================================================================
// Streaming events
// ===========================================================================

/// Server-sent event types for Anthropic streaming.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicStreamEvent {
    /// The message has started.
    MessageStart { message: AnthropicMessagesResponse },

    /// A new content block has started.
    ContentBlockStart {
        index: usize,
        content_block: AnthropicContent,
    },

    /// Incremental update to a content block.
    ContentBlockDelta {
        index: usize,
        delta: AnthropicContentDelta,
    },

    /// A content block has finished.
    ContentBlockStop { index: usize },

    /// Incremental update to the message (e.g. usage).
    MessageDelta {
        delta: AnthropicMessageDelta,
        usage: AnthropicDeltaUsage,
    },

    /// The message is complete.
    MessageStop,

    /// A ping to keep the connection alive.
    Ping,
}

/// Delta for content block updates.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
}

/// Delta for message-level updates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicMessageDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

/// Usage stats in a message delta.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AnthropicDeltaUsage {
    pub output_tokens: usize,
}
