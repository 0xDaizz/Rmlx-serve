//! Qwen-style tool call parser.
//!
//! Detects multiple formats:
//! 1. `✿FUNCTION✿: name\n✿ARGS✿: {json}\n✿RESULT✿`
//! 2. `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
//! 3. `<|tool_call_start|>{"name": ..., "arguments": ...}<|tool_call_end|>`

use regex::Regex;
use std::sync::LazyLock;
use tracing::debug;

use crate::parsers::utils::{generate_tool_call_id, parse_json_tool_call, strip_think_tags};
use crate::tool_parser::ToolCallParser;
use crate::types::{DeltaToolCall, ParsedToolCall, StreamingParseResult, ToolCallParseResult};

static FLOWER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)✿FUNCTION✿:\s*(\S+)\s*\n✿ARGS✿:\s*(.*?)\s*\n✿RESULT✿").unwrap()
});

static TOOL_CALL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").unwrap());

static QWEN_TOOL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>").unwrap()
});

/// Parser for Qwen-style tool calls.
pub struct QwenToolParser {
    buffer: String,
    current_tool_count: usize,
    in_tool_call: bool,
}

impl QwenToolParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            current_tool_count: 0,
            in_tool_call: false,
        }
    }

    fn parse_flower_format(text: &str) -> Vec<ParsedToolCall> {
        let mut calls = Vec::new();
        for cap in FLOWER_RE.captures_iter(text) {
            let name = cap.get(1).unwrap().as_str().to_string();
            let args_str = cap.get(2).unwrap().as_str();

            // Try to parse as JSON, or use raw string
            let arguments = if let Ok(v) = serde_json::from_str::<serde_json::Value>(args_str) {
                serde_json::to_string(&v).unwrap_or_else(|_| args_str.to_string())
            } else {
                args_str.to_string()
            };

            calls.push(ParsedToolCall {
                id: generate_tool_call_id(),
                name,
                arguments,
            });
        }
        calls
    }

    fn parse_xml_format(text: &str, re: &Regex) -> Vec<ParsedToolCall> {
        let mut calls = Vec::new();
        for cap in re.captures_iter(text) {
            let json_str = cap.get(1).unwrap().as_str();
            if let Some(tc) = parse_json_tool_call(json_str) {
                calls.push(tc);
            }
        }
        calls
    }
}

impl Default for QwenToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for QwenToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        let cleaned = strip_think_tags(text);

        // Try flower format first (✿FUNCTION✿)
        let mut tool_calls = Self::parse_flower_format(&cleaned);

        // If no flower format found, try <tool_call> format
        if tool_calls.is_empty() {
            tool_calls = Self::parse_xml_format(&cleaned, &TOOL_CALL_RE);
        }

        // If still nothing, try <|tool_call_start|> format
        if tool_calls.is_empty() {
            tool_calls = Self::parse_xml_format(&cleaned, &QWEN_TOOL_RE);
        }

        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "parsed Qwen tool calls");
        }

        // Extract content: everything not inside tool call markers
        let content = extract_non_tool_content(&cleaned);

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.buffer.push_str(delta);
        let text = strip_think_tags(curr);

        // Detect tool call region
        if text.contains("✿FUNCTION✿")
            || text.contains("<tool_call>")
            || text.contains("<|tool_call_start|>")
        {
            self.in_tool_call = true;
        }

        if !self.in_tool_call {
            return StreamingParseResult {
                content: Some(delta.to_string()),
                tool_calls: vec![],
                finished: false,
            };
        }

        // Parse completed tool calls from full accumulated text
        let mut all_calls = Self::parse_flower_format(&text);
        if all_calls.is_empty() {
            all_calls = Self::parse_xml_format(&text, &TOOL_CALL_RE);
        }
        if all_calls.is_empty() {
            all_calls = Self::parse_xml_format(&text, &QWEN_TOOL_RE);
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

        let finished = !all_calls.is_empty()
            && (text.contains("✿RESULT✿")
                || (text.contains("</tool_call>") && !text.ends_with("<tool_call>"))
                || text.contains("<|tool_call_end|>"));

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

    fn supports_native_tool_format(&self) -> bool {
        true
    }
}

/// Extract non-tool-call content from text.
fn extract_non_tool_content(text: &str) -> Option<String> {
    let mut content = text.to_string();

    // Remove flower-format tool calls
    content = FLOWER_RE.replace_all(&content, "").to_string();
    // Remove XML-format tool calls
    content = TOOL_CALL_RE.replace_all(&content, "").to_string();
    content = QWEN_TOOL_RE.replace_all(&content, "").to_string();

    let trimmed = content.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flower_format() {
        let parser = QwenToolParser::new();
        let text = "✿FUNCTION✿: get_weather\n✿ARGS✿: {\"city\": \"London\"}\n✿RESULT✿";
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_xml_tool_call_format() {
        let parser = QwenToolParser::new();
        let text =
            r#"<tool_call>{"name": "get_weather", "arguments": {"city": "London"}}</tool_call>"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_qwen_special_format() {
        let parser = QwenToolParser::new();
        let text =
            r#"<|tool_call_start|>{"name": "search", "arguments": {"q": "rust"}}<|tool_call_end|>"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn test_no_tool_calls() {
        let parser = QwenToolParser::new();
        let text = "Hello, I can help you!";
        let result = parser.parse(text);
        assert!(result.tool_calls.is_empty());
        assert!(result.content.is_some());
    }
}
