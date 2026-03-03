//! DeepSeek-style tool call parser.
//!
//! Detects `<｜tool▁call▁begin｜>` ... `<｜tool▁call▁end｜>` blocks
//! containing function name and arguments.
//!
//! Format:
//! ```text
//! <｜tool▁call▁begin｜>function_name
//! {"arg1": "val1"}
//! <｜tool▁call▁end｜>
//! ```

use regex::Regex;
use std::sync::LazyLock;
use tracing::debug;

use crate::parsers::utils::{generate_tool_call_id, strip_think_tags};
use crate::tool_parser::ToolCallParser;
use crate::types::{DeltaToolCall, ParsedToolCall, StreamingParseResult, ToolCallParseResult};

// DeepSeek uses fullwidth characters in its special tokens.
// ｜ = U+FF5C (fullwidth vertical line)
// ▁ = U+2581 (lower one eighth block, used as word separator)
static DEEPSEEK_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<\u{ff5c}tool\u{2581}call\u{2581}begin\u{ff5c}>\s*(.*?)\s*<\u{ff5c}tool\u{2581}call\u{2581}end\u{ff5c}>").unwrap()
});

/// Marker for the beginning of a DeepSeek tool call block.
const TOOL_CALL_BEGIN: &str = "<\u{ff5c}tool\u{2581}call\u{2581}begin\u{ff5c}>";
/// Marker for the end of a DeepSeek tool call block.
const TOOL_CALL_END: &str = "<\u{ff5c}tool\u{2581}call\u{2581}end\u{ff5c}>";

/// Parser for DeepSeek-style tool calls.
pub struct DeepSeekToolParser {
    buffer: String,
    current_tool_count: usize,
    in_tool_call: bool,
}

impl DeepSeekToolParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            current_tool_count: 0,
            in_tool_call: false,
        }
    }

    fn parse_block(block_content: &str) -> Option<ParsedToolCall> {
        let trimmed = block_content.trim();

        // The block format is:
        //   function_name\n{"arg": "value"}
        // or sometimes:
        //   function_name\n```json\n{"arg": "value"}\n```
        // or just:
        //   {"name": "fn", "arguments": {...}}

        // First, try parsing the entire block as a JSON tool call object
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(obj) = v.as_object() {
                if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
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
                    return Some(ParsedToolCall {
                        id: generate_tool_call_id(),
                        name: name.to_string(),
                        arguments,
                    });
                }
            }
        }

        // Try the "function_name\njson_args" format
        if let Some(newline_pos) = trimmed.find('\n') {
            let name = trimmed[..newline_pos].trim().to_string();
            let rest = trimmed[newline_pos + 1..].trim();

            if name.is_empty() {
                return None;
            }

            // Strip optional markdown code fences
            let json_str = rest
                .strip_prefix("```json")
                .or_else(|| rest.strip_prefix("```"))
                .unwrap_or(rest);
            let json_str = json_str.strip_suffix("```").unwrap_or(json_str).trim();

            let arguments = if json_str.is_empty() {
                "{}".to_string()
            } else if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                serde_json::to_string(&v).unwrap_or_else(|_| json_str.to_string())
            } else {
                json_str.to_string()
            };

            return Some(ParsedToolCall {
                id: generate_tool_call_id(),
                name,
                arguments,
            });
        }

        // Single line with just a function name, no args
        if !trimmed.is_empty() && !trimmed.starts_with('{') {
            return Some(ParsedToolCall {
                id: generate_tool_call_id(),
                name: trimmed.to_string(),
                arguments: "{}".to_string(),
            });
        }

        None
    }
}

impl Default for DeepSeekToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for DeepSeekToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        let cleaned = strip_think_tags(text);

        let mut tool_calls = Vec::new();
        let mut content_parts = Vec::new();
        let mut last_end = 0;

        for cap in DEEPSEEK_BLOCK_RE.captures_iter(&cleaned) {
            let full_match = cap.get(0).unwrap();
            let block_content = cap.get(1).unwrap().as_str();

            // Collect text before this block
            let before = &cleaned[last_end..full_match.start()];
            let trimmed = before.trim();
            if !trimmed.is_empty() {
                content_parts.push(trimmed.to_string());
            }
            last_end = full_match.end();

            if let Some(tc) = Self::parse_block(block_content) {
                debug!(name = %tc.name, "parsed DeepSeek tool call");
                tool_calls.push(tc);
            }
        }

        // Trailing text
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

        if text.contains(TOOL_CALL_BEGIN) {
            self.in_tool_call = true;
        }

        if !self.in_tool_call {
            return StreamingParseResult {
                content: Some(delta.to_string()),
                tool_calls: vec![],
                finished: false,
            };
        }

        // Parse completed blocks
        let mut all_calls = Vec::new();
        for cap in DEEPSEEK_BLOCK_RE.captures_iter(&text) {
            let block_content = cap.get(1).unwrap().as_str();
            if let Some(tc) = Self::parse_block(block_content) {
                all_calls.push(tc);
            }
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

        // Finished when the last open block is closed
        let finished = if let Some(last_begin) = text.rfind(TOOL_CALL_BEGIN) {
            text[last_begin..].contains(TOOL_CALL_END)
        } else {
            false
        };

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
    fn test_deepseek_function_format() {
        let parser = DeepSeekToolParser::new();
        let text = format!(
            "{}get_weather\n{{\"city\": \"London\"}}{}",
            TOOL_CALL_BEGIN, TOOL_CALL_END
        );
        let result = parser.parse(&text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("London"));
    }

    #[test]
    fn test_deepseek_json_format() {
        let parser = DeepSeekToolParser::new();
        let text = format!(
            "{}{{\"name\": \"search\", \"arguments\": {{\"q\": \"rust\"}}}}{}",
            TOOL_CALL_BEGIN, TOOL_CALL_END
        );
        let result = parser.parse(&text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn test_deepseek_multiple_calls() {
        let parser = DeepSeekToolParser::new();
        let text = format!(
            "{}get_weather\n{{\"city\": \"London\"}}{}{}get_time\n{{\"tz\": \"UTC\"}}{}",
            TOOL_CALL_BEGIN, TOOL_CALL_END, TOOL_CALL_BEGIN, TOOL_CALL_END
        );
        let result = parser.parse(&text);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[1].name, "get_time");
    }

    #[test]
    fn test_deepseek_with_content() {
        let parser = DeepSeekToolParser::new();
        let text = format!(
            "I'll check the weather. {}get_weather\n{{\"city\": \"London\"}}{}",
            TOOL_CALL_BEGIN, TOOL_CALL_END
        );
        let result = parser.parse(&text);
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.is_some());
        assert!(result.content.unwrap().contains("check the weather"));
    }

    #[test]
    fn test_no_tool_calls() {
        let parser = DeepSeekToolParser::new();
        let text = "Just a regular message.";
        let result = parser.parse(text);
        assert!(result.tool_calls.is_empty());
        assert!(result.content.is_some());
    }
}
