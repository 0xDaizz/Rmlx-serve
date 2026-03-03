//! DeepSeek-style tool call parser.
//!
//! Supports two formats:
//!
//! ## Format 1: ASCII special tokens (DeepSeek-V3 / R1)
//! ```text
//! <|tool_calls_begin|><|tool_call_begin|>function<|tool_sep|>name
//! {"arg": "val"}<|tool_call_end|><|tool_calls_end|>
//! ```
//!
//! Multiple calls separated by `<|tool_call_end|><|tool_call_begin|>`.
//!
//! ## Format 2: Fullwidth-character special tokens (legacy)
//! ```text
//! <｜tool▁call▁begin｜>function_name
//! {"arg1": "val1"}
//! <｜tool▁call▁end｜>
//! ```
//!
//! ｜ = U+FF5C (fullwidth vertical line)
//! ▁ = U+2581 (lower one eighth block)

use regex::Regex;
use std::sync::LazyLock;
use tracing::debug;

use crate::parsers::utils::{generate_tool_call_id, strip_think_tags};
use crate::tool_parser::ToolCallParser;
use crate::types::{DeltaToolCall, ParsedToolCall, StreamingParseResult, ToolCallParseResult};

// ---------------------------------------------------------------------------
// Format 1: ASCII tokens  <|tool_call_begin|>function<|tool_sep|>name\n{...}<|tool_call_end|>
// ---------------------------------------------------------------------------
const TOOL_CALLS_BEGIN: &str = "<|tool_calls_begin|>";
const TOOL_CALLS_END: &str = "<|tool_calls_end|>";
const TOOL_CALL_BEGIN: &str = "<|tool_call_begin|>";
const TOOL_CALL_END: &str = "<|tool_call_end|>";
const TOOL_SEP: &str = "<|tool_sep|>";

/// Regex for ASCII-token individual call blocks:
/// `<|tool_call_begin|>function<|tool_sep|>name\n{...}<|tool_call_end|>`
static ASCII_CALL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?s)<\|tool_call_begin\|>\s*(?:function\s*)?<\|tool_sep\|>\s*(.*?)\s*<\|tool_call_end\|>",
    )
    .unwrap()
});

// ---------------------------------------------------------------------------
// Format 2: Fullwidth-character tokens (legacy DeepSeek)
// ---------------------------------------------------------------------------
/// Marker for the beginning of a legacy DeepSeek tool call block.
const LEGACY_TOOL_CALL_BEGIN: &str = "<\u{ff5c}tool\u{2581}call\u{2581}begin\u{ff5c}>";
/// Marker for the end of a legacy DeepSeek tool call block.
const LEGACY_TOOL_CALL_END: &str = "<\u{ff5c}tool\u{2581}call\u{2581}end\u{ff5c}>";

static LEGACY_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?s)<\u{ff5c}tool\u{2581}call\u{2581}begin\u{ff5c}>\s*(.*?)\s*<\u{ff5c}tool\u{2581}call\u{2581}end\u{ff5c}>"
    )
    .unwrap()
});

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

    /// Parse a block that has already been extracted from between begin/end markers.
    ///
    /// For the ASCII format, the block content is everything between `<|tool_sep|>` and
    /// `<|tool_call_end|>`, which looks like:  `name\n{"arg": "val"}`
    ///
    /// For the legacy format, the block content is:  `function_name\n{"arg": "val"}`
    fn parse_block(block_content: &str) -> Option<ParsedToolCall> {
        let trimmed = block_content.trim();

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

    /// Extract tool calls from the ASCII token format.
    fn extract_ascii_calls(text: &str) -> Vec<ParsedToolCall> {
        let mut calls = Vec::new();
        for cap in ASCII_CALL_RE.captures_iter(text) {
            let block = cap.get(1).unwrap().as_str();
            if let Some(tc) = Self::parse_block(block) {
                calls.push(tc);
            }
        }
        calls
    }

    /// Extract tool calls from the legacy fullwidth token format.
    fn extract_legacy_calls(text: &str) -> Vec<ParsedToolCall> {
        let mut calls = Vec::new();
        for cap in LEGACY_BLOCK_RE.captures_iter(text) {
            let block = cap.get(1).unwrap().as_str();
            if let Some(tc) = Self::parse_block(block) {
                calls.push(tc);
            }
        }
        calls
    }

    /// Detect whether the text uses ASCII-format or legacy-format markers.
    fn has_ascii_markers(text: &str) -> bool {
        text.contains(TOOL_CALL_BEGIN) || text.contains(TOOL_CALLS_BEGIN)
    }

    fn has_legacy_markers(text: &str) -> bool {
        text.contains(LEGACY_TOOL_CALL_BEGIN)
    }

    /// Extract content (non-tool-call text) from the text.
    fn extract_content(text: &str) -> Option<String> {
        let mut content = text.to_string();

        // Remove ASCII-format wrapper
        if let Some(begin_pos) = content.find(TOOL_CALLS_BEGIN) {
            if let Some(end_pos) = content.find(TOOL_CALLS_END) {
                let before = content[..begin_pos].trim();
                let after = content[end_pos + TOOL_CALLS_END.len()..].trim();
                let mut parts = Vec::new();
                if !before.is_empty() {
                    parts.push(before.to_string());
                }
                if !after.is_empty() {
                    parts.push(after.to_string());
                }
                return if parts.is_empty() {
                    None
                } else {
                    Some(parts.join("\n"))
                };
            }
        }

        // Remove individual ASCII call blocks
        content = ASCII_CALL_RE.replace_all(&content, "").to_string();
        // Remove legacy blocks
        content = LEGACY_BLOCK_RE.replace_all(&content, "").to_string();
        // Remove remaining markers
        content = content.replace(TOOL_CALLS_BEGIN, "");
        content = content.replace(TOOL_CALLS_END, "");
        content = content.replace(TOOL_CALL_BEGIN, "");
        content = content.replace(TOOL_CALL_END, "");
        content = content.replace(TOOL_SEP, "");

        let trimmed = content.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
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

        let tool_calls = if Self::has_ascii_markers(&cleaned) {
            let calls = Self::extract_ascii_calls(&cleaned);
            if !calls.is_empty() {
                calls
            } else if Self::has_legacy_markers(&cleaned) {
                Self::extract_legacy_calls(&cleaned)
            } else {
                vec![]
            }
        } else if Self::has_legacy_markers(&cleaned) {
            Self::extract_legacy_calls(&cleaned)
        } else {
            vec![]
        };

        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "parsed DeepSeek tool calls");
        }

        let content = if tool_calls.is_empty() {
            let trimmed = cleaned.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        } else {
            Self::extract_content(&cleaned)
        };

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.buffer.push_str(delta);
        let text = strip_think_tags(curr);

        if text.contains(TOOL_CALL_BEGIN)
            || text.contains(TOOL_CALLS_BEGIN)
            || text.contains(LEGACY_TOOL_CALL_BEGIN)
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

        // Parse completed blocks
        let all_calls = if Self::has_ascii_markers(&text) {
            Self::extract_ascii_calls(&text)
        } else {
            Self::extract_legacy_calls(&text)
        };

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
        let finished = if Self::has_ascii_markers(&text) {
            // Finished when we see tool_calls_end or the last tool_call_begin has a matching end
            text.contains(TOOL_CALLS_END)
                || (if let Some(last_begin) = text.rfind(TOOL_CALL_BEGIN) {
                    text[last_begin..].contains(TOOL_CALL_END)
                } else {
                    false
                })
        } else if let Some(last_begin) = text.rfind(LEGACY_TOOL_CALL_BEGIN) {
            text[last_begin..].contains(LEGACY_TOOL_CALL_END)
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

    fn supports_native_tool_format(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ASCII format tests (DeepSeek-V3 / R1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ascii_format_single_call() {
        let parser = DeepSeekToolParser::new();
        let text = format!(
            "{}{}function{}get_weather\n{{\"city\": \"London\"}}{}{}",
            TOOL_CALLS_BEGIN, TOOL_CALL_BEGIN, TOOL_SEP, TOOL_CALL_END, TOOL_CALLS_END
        );
        let result = parser.parse(&text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("London"));
    }

    #[test]
    fn test_ascii_format_multiple_calls() {
        let parser = DeepSeekToolParser::new();
        let text = format!(
            "{}{}function{}get_weather\n{{\"city\": \"London\"}}{}{}function{}get_time\n{{\"tz\": \"UTC\"}}{}{}",
            TOOL_CALLS_BEGIN,
            TOOL_CALL_BEGIN, TOOL_SEP, TOOL_CALL_END,
            TOOL_CALL_BEGIN, TOOL_SEP, TOOL_CALL_END,
            TOOL_CALLS_END
        );
        let result = parser.parse(&text);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[1].name, "get_time");
    }

    #[test]
    fn test_ascii_format_with_content() {
        let parser = DeepSeekToolParser::new();
        let text = format!(
            "I'll check the weather. {}{}function{}get_weather\n{{\"city\": \"London\"}}{}{}",
            TOOL_CALLS_BEGIN, TOOL_CALL_BEGIN, TOOL_SEP, TOOL_CALL_END, TOOL_CALLS_END
        );
        let result = parser.parse(&text);
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.is_some());
        assert!(result.content.unwrap().contains("check the weather"));
    }

    // -----------------------------------------------------------------------
    // Legacy fullwidth format tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_legacy_function_format() {
        let parser = DeepSeekToolParser::new();
        let text = format!(
            "{}get_weather\n{{\"city\": \"London\"}}{}",
            LEGACY_TOOL_CALL_BEGIN, LEGACY_TOOL_CALL_END
        );
        let result = parser.parse(&text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("London"));
    }

    #[test]
    fn test_legacy_json_format() {
        let parser = DeepSeekToolParser::new();
        let text = format!(
            "{}{{\"name\": \"search\", \"arguments\": {{\"q\": \"rust\"}}}}{}",
            LEGACY_TOOL_CALL_BEGIN, LEGACY_TOOL_CALL_END
        );
        let result = parser.parse(&text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn test_legacy_multiple_calls() {
        let parser = DeepSeekToolParser::new();
        let text = format!(
            "{}get_weather\n{{\"city\": \"London\"}}{}{}get_time\n{{\"tz\": \"UTC\"}}{}",
            LEGACY_TOOL_CALL_BEGIN,
            LEGACY_TOOL_CALL_END,
            LEGACY_TOOL_CALL_BEGIN,
            LEGACY_TOOL_CALL_END
        );
        let result = parser.parse(&text);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[1].name, "get_time");
    }

    #[test]
    fn test_legacy_with_content() {
        let parser = DeepSeekToolParser::new();
        let text = format!(
            "I'll check the weather. {}get_weather\n{{\"city\": \"London\"}}{}",
            LEGACY_TOOL_CALL_BEGIN, LEGACY_TOOL_CALL_END
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
