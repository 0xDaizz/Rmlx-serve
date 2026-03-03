//! Mistral-style tool call parser.
//!
//! Detects `[TOOL_CALLS]` followed by a JSON array of tool calls.
//! Format: `[TOOL_CALLS] [{"name": "func", "arguments": {...}}]`
//!
//! Tool call IDs are 9-character alphanumeric strings.

use regex::Regex;
use std::sync::LazyLock;
use tracing::debug;

use crate::parsers::utils::{extract_json_array, strip_think_tags};
use crate::tool_parser::ToolCallParser;
use crate::types::{DeltaToolCall, ParsedToolCall, StreamingParseResult, ToolCallParseResult};

static TOOL_CALLS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)\[TOOL_CALLS\]\s*(.*)").unwrap());

/// Generate a Mistral-style 9-character alphanumeric tool call ID.
fn generate_mistral_tool_call_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let mixed = count
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(nanos);

    // Generate a 9-character alphanumeric string
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut result = String::with_capacity(9);
    let mut val = mixed;
    for _ in 0..9 {
        result.push(CHARS[(val % CHARS.len() as u64) as usize] as char);
        val /= CHARS.len() as u64;
    }
    result
}

/// Parser for Mistral-style `[TOOL_CALLS]` blocks.
pub struct MistralToolParser {
    buffer: String,
    current_tool_count: usize,
    in_tool_call: bool,
}

impl MistralToolParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            current_tool_count: 0,
            in_tool_call: false,
        }
    }

    fn parse_tool_calls(text: &str) -> (Vec<ParsedToolCall>, Option<String>) {
        if let Some(cap) = TOOL_CALLS_RE.captures(text) {
            let full_match = cap.get(0).unwrap();
            let json_region = cap.get(1).unwrap().as_str().trim();

            // Content is everything before [TOOL_CALLS]
            let before = text[..full_match.start()].trim();
            let content = if before.is_empty() {
                None
            } else {
                Some(before.to_string())
            };

            // Try to parse as JSON array
            let mut calls = Vec::new();

            // Extract the JSON array from the region
            let json_str = if let Some(arr) = extract_json_array(json_region) {
                arr
            } else {
                // Maybe the entire region is the array
                json_region.to_string()
            };

            if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(&json_str) {
                for item in arr {
                    if let Some(obj) = item.as_object() {
                        let name = obj
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_string();

                        if name.is_empty() {
                            continue;
                        }

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

                        // Use the id from the JSON if present, otherwise generate
                        // a Mistral-style 9-character alphanumeric ID
                        let id = obj
                            .get("id")
                            .and_then(|id| id.as_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(generate_mistral_tool_call_id);

                        calls.push(ParsedToolCall {
                            id,
                            name,
                            arguments,
                        });
                    }
                }
            }

            (calls, content)
        } else {
            (vec![], Some(text.trim().to_string()))
        }
    }
}

impl Default for MistralToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for MistralToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        let cleaned = strip_think_tags(text);
        let (tool_calls, content) = Self::parse_tool_calls(&cleaned);

        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "parsed Mistral tool calls");
        }

        // If no tool calls found and content is the full text, return it
        let content = if tool_calls.is_empty() {
            let trimmed = cleaned.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        } else {
            content
        };

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.buffer.push_str(delta);
        let text = strip_think_tags(curr);

        if text.contains("[TOOL_CALLS]") {
            self.in_tool_call = true;
        }

        if !self.in_tool_call {
            return StreamingParseResult {
                content: Some(delta.to_string()),
                tool_calls: vec![],
                finished: false,
            };
        }

        let (all_calls, _) = Self::parse_tool_calls(&text);
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

        // Finished when we've parsed the full JSON array (ends with ])
        let finished = !all_calls.is_empty()
            && text
                .split("[TOOL_CALLS]")
                .last()
                .map(|s| s.trim().ends_with(']'))
                .unwrap_or(false);

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

    #[test]
    fn test_single_tool_call() {
        let parser = MistralToolParser::new();
        let text = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "London"}}]"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_multiple_tool_calls() {
        let parser = MistralToolParser::new();
        let text = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "London"}}, {"name": "get_time", "arguments": {"tz": "UTC"}}]"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 2);
    }

    #[test]
    fn test_with_content() {
        let parser = MistralToolParser::new();
        let text = r#"Let me check that for you.
[TOOL_CALLS] [{"name": "search", "arguments": {"q": "rust"}}]"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.is_some());
        assert!(result.content.unwrap().contains("Let me check"));
    }

    #[test]
    fn test_no_tool_calls() {
        let parser = MistralToolParser::new();
        let text = "Hello there!";
        let result = parser.parse(text);
        assert!(result.tool_calls.is_empty());
        assert!(result.content.is_some());
    }

    #[test]
    fn test_with_id() {
        let parser = MistralToolParser::new();
        let text = r#"[TOOL_CALLS] [{"id": "call_123", "name": "get_weather", "arguments": {"city": "London"}}]"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].id, "call_123");
    }

    #[test]
    fn test_generated_id_is_9_chars_alphanumeric() {
        let id = generate_mistral_tool_call_id();
        assert_eq!(id.len(), 9);
        assert!(id.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn test_auto_generated_id_format() {
        let parser = MistralToolParser::new();
        let text = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "London"}}]"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        // Auto-generated ID should be 9 chars alphanumeric
        let id = &result.tool_calls[0].id;
        assert_eq!(id.len(), 9);
        assert!(id.chars().all(|c| c.is_ascii_alphanumeric()));
    }
}
