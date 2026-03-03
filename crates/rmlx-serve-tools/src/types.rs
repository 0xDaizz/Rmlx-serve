//! Shared types for tool call parsing and reasoning extraction.

use serde::{Deserialize, Serialize};

/// A fully parsed tool/function call extracted from model output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedToolCall {
    /// Unique identifier for this tool call (e.g., "call_abc123").
    pub id: String,
    /// Name of the function to invoke.
    pub name: String,
    /// Arguments as a JSON string.
    pub arguments: String,
}

/// Result of parsing a complete model output for tool calls.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ToolCallParseResult {
    /// Extracted tool calls, if any.
    pub tool_calls: Vec<ParsedToolCall>,
    /// Non-tool-call content (text before/after tool call markers).
    pub content: Option<String>,
}

/// Result of parsing reasoning/thinking blocks from model output.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReasoningParseResult {
    /// Extracted thinking/reasoning content, if any.
    pub thinking: Option<String>,
    /// The remaining content after thinking blocks are removed.
    pub content: String,
}

/// A delta (incremental) tool call update for streaming responses.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeltaToolCall {
    /// Zero-based index identifying which tool call this delta belongs to.
    pub index: usize,
    /// Tool call id (sent only in the first delta for a given tool call).
    pub id: Option<String>,
    /// Function name (sent only in the first delta for a given tool call).
    pub name: Option<String>,
    /// Incremental arguments fragment.
    pub arguments: Option<String>,
}

/// Result of a streaming parse step.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StreamingParseResult {
    /// Non-tool-call content delta.
    pub content: Option<String>,
    /// Incremental tool call updates.
    pub tool_calls: Vec<DeltaToolCall>,
    /// Whether all tool calls are fully received.
    pub finished: bool,
}
