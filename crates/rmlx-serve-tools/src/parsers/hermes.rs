//! Hermes-style tool call parser.
//!
//! Detects `<tool_call>{"name": ..., "arguments": ...}</tool_call>` patterns.
//! Also strips `<think>...</think>` blocks.

use regex::Regex;
use std::sync::LazyLock;
use tracing::debug;

use crate::parsers::utils::{generate_tool_call_id, strip_think_tags};
use crate::tool_parser::ToolCallParser;
use crate::types::{DeltaToolCall, ParsedToolCall, StreamingParseResult, ToolCallParseResult};

static TOOL_CALL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").unwrap());

/// Parser for Hermes-style `<tool_call>` blocks.
pub struct HermesToolParser {
    /// Accumulated text for streaming.
    buffer: String,
    /// Number of tool calls already emitted during streaming.
    current_tool_count: usize,
    /// Whether we have detected we are inside a tool call block.
    in_tool_call: bool,
}

impl HermesToolParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            current_tool_count: 0,
            in_tool_call: false,
        }
    }

    fn parse_tool_call_json(json_str: &str) -> Option<ParsedToolCall> {
        let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
        let obj = v.as_object()?;
        let name = obj.get("name")?.as_str()?.to_string();
        let args = obj
            .get("arguments")
            .or_else(|| obj.get("parameters"))
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let arguments = if args.is_string() {
            args.as_str().unwrap().to_string()
        } else {
            serde_json::to_string(&args).unwrap_or_default()
        };
        Some(ParsedToolCall {
            id: generate_tool_call_id(),
            name,
            arguments,
        })
    }
}

impl Default for HermesToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for HermesToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        // Strip think tags first
        let cleaned = strip_think_tags(text);

        let mut tool_calls = Vec::new();
        let mut content_parts = Vec::new();
        let mut last_end = 0;

        for cap in TOOL_CALL_RE.captures_iter(&cleaned) {
            let full_match = cap.get(0).unwrap();
            let json_str = cap.get(1).unwrap().as_str();

            // Collect text before this tool call
            let before = &cleaned[last_end..full_match.start()];
            let trimmed = before.trim();
            if !trimmed.is_empty() {
                content_parts.push(trimmed.to_string());
            }
            last_end = full_match.end();

            match Self::parse_tool_call_json(json_str) {
                Some(tc) => {
                    debug!(name = %tc.name, "parsed Hermes tool call");
                    tool_calls.push(tc);
                }
                None => {
                    debug!(json = json_str, "failed to parse Hermes tool call JSON");
                }
            }
        }

        // Collect any trailing text
        let trailing = cleaned[last_end..].trim();
        if !trailing.is_empty() {
            content_parts.push(trailing.to_string());
        }

        let content = if content_parts.is_empty() {
            None
        } else {
            Some(content_parts.join("\n"))
        };

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.buffer.push_str(delta);
        let text = strip_think_tags(curr);

        // Check if we're currently in or entering a tool_call block
        if text.contains("<tool_call>") {
            self.in_tool_call = true;
        }

        if !self.in_tool_call {
            // No tool call detected yet, pass through as content
            return StreamingParseResult {
                content: Some(delta.to_string()),
                tool_calls: vec![],
                finished: false,
            };
        }

        // Try to parse completed tool calls from the full text so far
        let mut tool_calls = Vec::new();
        let mut count = 0;
        for cap in TOOL_CALL_RE.captures_iter(&text) {
            count += 1;
            if count > self.current_tool_count {
                let json_str = cap.get(1).unwrap().as_str();
                if let Some(tc) = Self::parse_tool_call_json(json_str) {
                    tool_calls.push(DeltaToolCall {
                        index: self.current_tool_count,
                        id: Some(tc.id),
                        name: Some(tc.name),
                        arguments: Some(tc.arguments),
                    });
                    self.current_tool_count += 1;
                }
            }
        }

        // Check if the last tool call block is closed
        let finished = if let Some(last_open) = text.rfind("<tool_call>") {
            text[last_open..].contains("</tool_call>")
        } else {
            false
        };

        // Extract content before first tool call
        let content = if let Some(idx) = text.find("<tool_call>") {
            let before = text[..idx].trim();
            if !before.is_empty() && self.current_tool_count <= 1 {
                // Only emit content on the first pass
                None
            } else {
                None
            }
        } else {
            None
        };

        StreamingParseResult {
            content,
            tool_calls,
            finished,
        }
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.current_tool_count = 0;
        self.in_tool_call = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_tool_call() {
        let parser = HermesToolParser::new();
        let text =
            r#"<tool_call>{"name": "get_weather", "arguments": {"city": "London"}}</tool_call>"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("London"));
        assert!(result.content.is_none());
    }

    #[test]
    fn test_multiple_tool_calls() {
        let parser = HermesToolParser::new();
        let text = r#"<tool_call>{"name": "get_weather", "arguments": {"city": "London"}}</tool_call>
<tool_call>{"name": "get_time", "arguments": {"timezone": "UTC"}}</tool_call>"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[1].name, "get_time");
    }

    #[test]
    fn test_with_content_and_tool_call() {
        let parser = HermesToolParser::new();
        let text = r#"Let me check the weather for you.
<tool_call>{"name": "get_weather", "arguments": {"city": "London"}}</tool_call>"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.is_some());
        assert!(result.content.unwrap().contains("check the weather"));
    }

    #[test]
    fn test_with_think_tags() {
        let parser = HermesToolParser::new();
        let text = r#"<think>I need to check the weather</think>
<tool_call>{"name": "get_weather", "arguments": {"city": "London"}}</tool_call>"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_no_tool_calls() {
        let parser = HermesToolParser::new();
        let text = "Hello, how can I help you?";
        let result = parser.parse(text);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content.unwrap(), "Hello, how can I help you?");
    }
}
