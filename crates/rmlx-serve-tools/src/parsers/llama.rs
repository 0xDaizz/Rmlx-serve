//! Llama-style tool call parser.
//!
//! Detects:
//! 1. `<|python_tag|>` followed by JSON tool call(s)
//! 2. Direct JSON `{"name": ..., "parameters": ...}` patterns

use regex::Regex;
use std::sync::LazyLock;
use tracing::debug;

use crate::parsers::utils::{
    extract_json_objects, generate_tool_call_id, parse_json_tool_call, strip_think_tags,
};
use crate::tool_parser::ToolCallParser;
use crate::types::{DeltaToolCall, ParsedToolCall, StreamingParseResult, ToolCallParseResult};

static PYTHON_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<\|python_tag\|>\s*(.*)").unwrap());

/// Parser for Llama-style tool calls.
pub struct LlamaToolParser {
    buffer: String,
    current_tool_count: usize,
    in_tool_call: bool,
}

impl LlamaToolParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            current_tool_count: 0,
            in_tool_call: false,
        }
    }

    fn extract_tool_calls(text: &str) -> Vec<ParsedToolCall> {
        let mut calls = Vec::new();

        // Check for <|python_tag|> prefix
        let json_text = if let Some(cap) = PYTHON_TAG_RE.captures(text) {
            cap.get(1).unwrap().as_str().to_string()
        } else {
            text.to_string()
        };

        // Try to parse as JSON array first
        if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(json_text.trim()) {
            for item in arr {
                if let Some(obj) = item.as_object() {
                    if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
                        let args = obj
                            .get("parameters")
                            .or_else(|| obj.get("arguments"))
                            .cloned()
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        let arguments = if args.is_string() {
                            args.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string(&args).unwrap_or_default()
                        };
                        calls.push(ParsedToolCall {
                            id: generate_tool_call_id(),
                            name: name.to_string(),
                            arguments,
                        });
                    }
                }
            }
            return calls;
        }

        // Try extracting individual JSON objects
        let json_objs = extract_json_objects(&json_text);
        for obj_str in json_objs {
            if let Some(tc) = parse_json_tool_call(&obj_str) {
                calls.push(tc);
            }
        }

        calls
    }
}

impl Default for LlamaToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for LlamaToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        let cleaned = strip_think_tags(text);

        // Check if there's a python_tag or tool-call-like JSON
        let has_python_tag = cleaned.contains("<|python_tag|>");

        let tool_calls = Self::extract_tool_calls(&cleaned);

        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "parsed Llama tool calls");
        }

        // Extract content: text before <|python_tag|> or before JSON tool calls
        let content = if has_python_tag {
            let before = cleaned.split("<|python_tag|>").next().unwrap_or("").trim();
            if before.is_empty() {
                None
            } else {
                Some(before.to_string())
            }
        } else if tool_calls.is_empty() {
            Some(cleaned.trim().to_string())
        } else {
            None
        };

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.buffer.push_str(delta);
        let text = strip_think_tags(curr);

        if text.contains("<|python_tag|>") || text.contains("\"name\"") {
            self.in_tool_call = true;
        }

        if !self.in_tool_call {
            return StreamingParseResult {
                content: Some(delta.to_string()),
                tool_calls: vec![],
                finished: false,
            };
        }

        let all_calls = Self::extract_tool_calls(&text);
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

        // Heuristic: finished if we have at least one tool call and text ends
        // with a closing brace (possibly with whitespace) or an end tag.
        let finished = !all_calls.is_empty() && text.trim().ends_with('}');

        StreamingParseResult {
            content: None,
            tool_calls: new_deltas,
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
    fn test_python_tag_single() {
        let parser = LlamaToolParser::new();
        let text = r#"<|python_tag|>{"name": "get_weather", "parameters": {"city": "London"}}"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_python_tag_array() {
        let parser = LlamaToolParser::new();
        let text = r#"<|python_tag|>[{"name": "get_weather", "parameters": {"city": "London"}}, {"name": "get_time", "parameters": {"tz": "UTC"}}]"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 2);
    }

    #[test]
    fn test_direct_json() {
        let parser = LlamaToolParser::new();
        let text = r#"{"name": "search", "parameters": {"query": "rust lang"}}"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn test_no_tool_calls() {
        let parser = LlamaToolParser::new();
        let text = "Just a regular response with no tools.";
        let result = parser.parse(text);
        assert!(result.tool_calls.is_empty());
        assert!(result.content.is_some());
    }

    #[test]
    fn test_with_content_before_tag() {
        let parser = LlamaToolParser::new();
        let text = r#"Let me help you.
<|python_tag|>{"name": "get_weather", "parameters": {"city": "London"}}"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.is_some());
        assert!(result.content.unwrap().contains("Let me help"));
    }
}
