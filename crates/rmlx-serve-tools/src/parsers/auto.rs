//! Auto-detection tool call parser.
//!
//! Attempts to detect tool calls using multiple heuristics:
//! 1. JSON array of tool calls `[{"name": ..., "arguments": ...}, ...]`
//! 2. Single JSON object tool call `{"name": ..., "arguments": ...}`
//! 3. Known marker patterns from other parsers
//! 4. Fallback: no tool calls

use tracing::debug;

use crate::parsers::utils::{
    extract_json_array, extract_json_objects, parse_json_tool_call, strip_think_tags,
};
use crate::tool_parser::ToolCallParser;
use crate::types::{
    DeltaToolCall, ParsedToolCall, StreamingParseResult, ToolCallParseResult,
};

/// Automatic tool call parser that tries multiple detection strategies.
pub struct AutoToolParser {
    buffer: String,
    current_tool_count: usize,
    detected_calls: Vec<ParsedToolCall>,
}

impl AutoToolParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            current_tool_count: 0,
            detected_calls: Vec::new(),
        }
    }

    fn detect_tool_calls(text: &str) -> Vec<ParsedToolCall> {
        // Strategy 1: Try to find and parse a JSON array of tool calls
        if let Some(arr_str) = extract_json_array(text) {
            if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(&arr_str) {
                let mut calls = Vec::new();
                for item in &arr {
                    if let Some(obj) = item.as_object() {
                        if obj.contains_key("name") {
                            let json_str = serde_json::to_string(item).unwrap_or_default();
                            if let Some(tc) = parse_json_tool_call(&json_str) {
                                calls.push(tc);
                            }
                        }
                    }
                }
                if !calls.is_empty() {
                    return calls;
                }
            }
        }

        // Strategy 2: Try individual JSON objects
        let json_objs = extract_json_objects(text);
        let mut calls = Vec::new();
        for obj_str in &json_objs {
            // Check if this looks like a tool call (has "name" key)
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(obj_str) {
                if let Some(obj) = v.as_object() {
                    // Must have "name" and either "arguments" or "parameters"
                    if obj.contains_key("name")
                        && (obj.contains_key("arguments") || obj.contains_key("parameters"))
                    {
                        if let Some(tc) = parse_json_tool_call(obj_str) {
                            calls.push(tc);
                        }
                    }
                    // Also check for {"tool_calls": [...]} wrapper
                    else if let Some(arr) = obj.get("tool_calls").and_then(|v| v.as_array()) {
                        for item in arr {
                            let item_str = serde_json::to_string(item).unwrap_or_default();
                            if let Some(tc) = parse_json_tool_call(&item_str) {
                                calls.push(tc);
                            }
                        }
                    }
                }
            }
        }

        calls
    }
}

impl Default for AutoToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for AutoToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        let cleaned = strip_think_tags(text);
        let tool_calls = Self::detect_tool_calls(&cleaned);

        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "auto-detected tool calls");
        }

        let content = if tool_calls.is_empty() {
            let trimmed = cleaned.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        } else {
            // Try to extract content that's not part of JSON
            let trimmed = cleaned.trim();
            // If the entire text is JSON, no content
            if trimmed.starts_with('[') || trimmed.starts_with('{') {
                None
            } else {
                // Find where the JSON starts and get content before it
                let first_brace = trimmed.find('{').or_else(|| trimmed.find('['));
                if let Some(idx) = first_brace {
                    let before = trimmed[..idx].trim();
                    if before.is_empty() {
                        None
                    } else {
                        Some(before.to_string())
                    }
                } else {
                    None
                }
            }
        };

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.buffer.push_str(delta);
        let text = strip_think_tags(curr);

        let all_calls = Self::detect_tool_calls(&text);

        if all_calls.is_empty() {
            // No tool calls detected yet; pass through as content
            return StreamingParseResult {
                content: Some(delta.to_string()),
                tool_calls: vec![],
                finished: false,
            };
        }

        let mut new_deltas = Vec::new();
        for (i, tc) in all_calls.iter().enumerate() {
            if i >= self.current_tool_count {
                new_deltas.push(DeltaToolCall {
                    index: i,
                    id: Some(tc.id.clone()),
                    name: Some(tc.name.clone()),
                    arguments: Some(tc.arguments.clone()),
                });
                self.current_tool_count = i + 1;
            }
        }

        self.detected_calls = all_calls;

        // Simple heuristic for finished: text ends with } or ]
        let finished = text.trim().ends_with('}') || text.trim().ends_with(']');

        StreamingParseResult {
            content: None,
            tool_calls: new_deltas,
            finished,
        }
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.current_tool_count = 0;
        self.detected_calls.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_array() {
        let parser = AutoToolParser::new();
        let text =
            r#"[{"name": "get_weather", "arguments": {"city": "London"}}, {"name": "get_time", "arguments": {"tz": "UTC"}}]"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[1].name, "get_time");
    }

    #[test]
    fn test_single_json_object() {
        let parser = AutoToolParser::new();
        let text = r#"{"name": "get_weather", "arguments": {"city": "London"}}"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_tool_calls_wrapper() {
        let parser = AutoToolParser::new();
        let text =
            r#"{"tool_calls": [{"name": "search", "arguments": {"q": "rust"}}]}"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn test_no_tool_calls() {
        let parser = AutoToolParser::new();
        let text = "Just a regular message with no JSON.";
        let result = parser.parse(text);
        assert!(result.tool_calls.is_empty());
        assert!(result.content.is_some());
    }

    #[test]
    fn test_json_without_tool_call_shape() {
        let parser = AutoToolParser::new();
        let text = r#"{"key": "value", "count": 42}"#;
        let result = parser.parse(text);
        // This JSON doesn't have "name" + "arguments/parameters", so no tool calls
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn test_with_parameters_key() {
        let parser = AutoToolParser::new();
        let text = r#"{"name": "search", "parameters": {"query": "hello"}}"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn test_with_think_tags() {
        let parser = AutoToolParser::new();
        let text = r#"<think>Let me think...</think>{"name": "search", "arguments": {"q": "test"}}"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
    }
}
