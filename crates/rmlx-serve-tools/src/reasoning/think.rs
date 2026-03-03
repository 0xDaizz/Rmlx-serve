//! Generic `<think>...</think>` reasoning parser.
//!
//! Separates content inside `<think>` blocks from the visible response.

use regex::Regex;
use std::sync::LazyLock;

use crate::reasoning_parser::ReasoningParser;
use crate::types::ReasoningParseResult;

static THINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>(.*?)</think>").unwrap());

/// Generic parser for `<think>...</think>` blocks.
pub struct ThinkParser {
    /// Streaming state: whether we are currently inside a `<think>` block.
    in_think: bool,
    /// Accumulated thinking content during streaming.
    think_buffer: String,
    /// Accumulated output content during streaming.
    content_buffer: String,
    /// Raw buffer for partial tag detection.
    raw_buffer: String,
}

impl ThinkParser {
    pub fn new() -> Self {
        Self {
            in_think: false,
            think_buffer: String::new(),
            content_buffer: String::new(),
            raw_buffer: String::new(),
        }
    }
}

impl Default for ThinkParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for ThinkParser {
    fn parse(&self, text: &str) -> ReasoningParseResult {
        let mut thinking_parts = Vec::new();

        for cap in THINK_RE.captures_iter(text) {
            let think_content = cap.get(1).unwrap().as_str().trim();
            if !think_content.is_empty() {
                thinking_parts.push(think_content.to_string());
            }
        }

        let content = THINK_RE.replace_all(text, "").to_string();
        let content = content.trim().to_string();

        let thinking = if thinking_parts.is_empty() {
            None
        } else {
            Some(thinking_parts.join("\n\n"))
        };

        ReasoningParseResult { thinking, content }
    }

    fn parse_streaming(&mut self, delta: &str) -> Option<ReasoningParseResult> {
        self.raw_buffer.push_str(delta);

        // Process the raw buffer looking for <think> and </think> tags
        loop {
            if self.in_think {
                // Look for closing </think> tag
                if let Some(end_pos) = self.raw_buffer.find("</think>") {
                    // Everything before </think> is thinking content
                    let think_part = &self.raw_buffer[..end_pos];
                    self.think_buffer.push_str(think_part);
                    self.in_think = false;
                    // Remove processed part from raw buffer
                    self.raw_buffer = self.raw_buffer[end_pos + "</think>".len()..].to_string();
                } else {
                    // Check if we might have a partial </think> at the end
                    let potential_tag = "</think>";
                    let mut partial = false;
                    for i in 1..potential_tag.len() {
                        if self.raw_buffer.ends_with(&potential_tag[..i]) {
                            partial = true;
                            break;
                        }
                    }
                    if partial {
                        // Wait for more data
                        return None;
                    }
                    // All of raw_buffer is thinking content
                    self.think_buffer.push_str(&self.raw_buffer);
                    self.raw_buffer.clear();
                    return None;
                }
            } else {
                // Look for opening <think> tag
                if let Some(start_pos) = self.raw_buffer.find("<think>") {
                    // Everything before <think> is content
                    let content_part = &self.raw_buffer[..start_pos];
                    self.content_buffer.push_str(content_part);
                    self.in_think = true;
                    self.raw_buffer = self.raw_buffer[start_pos + "<think>".len()..].to_string();
                } else {
                    // Check for partial <think> at the end
                    let potential_tag = "<think>";
                    let mut partial_len = 0;
                    for i in 1..potential_tag.len() {
                        if self.raw_buffer.ends_with(&potential_tag[..i]) {
                            partial_len = i;
                            break;
                        }
                    }
                    if partial_len > 0 {
                        // Move everything except the potential partial tag to content
                        let safe_end = self.raw_buffer.len() - partial_len;
                        self.content_buffer.push_str(&self.raw_buffer[..safe_end]);
                        self.raw_buffer = self.raw_buffer[safe_end..].to_string();
                        break;
                    }
                    // All of raw_buffer is content
                    self.content_buffer.push_str(&self.raw_buffer);
                    self.raw_buffer.clear();
                    break;
                }
            }
        }

        let thinking = if self.think_buffer.is_empty() {
            None
        } else {
            Some(self.think_buffer.clone())
        };

        Some(ReasoningParseResult {
            thinking,
            content: self.content_buffer.clone(),
        })
    }

    fn reset(&mut self) {
        self.in_think = false;
        self.think_buffer.clear();
        self.content_buffer.clear();
        self.raw_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_think_block() {
        let parser = ThinkParser::new();
        let text = "<think>I need to figure this out</think>The answer is 42.";
        let result = parser.parse(text);
        assert_eq!(result.thinking.unwrap(), "I need to figure this out");
        assert_eq!(result.content, "The answer is 42.");
    }

    #[test]
    fn test_multiple_think_blocks() {
        let parser = ThinkParser::new();
        let text = "<think>step 1</think>Result 1\n<think>step 2</think>Result 2";
        let result = parser.parse(text);
        assert!(result.thinking.is_some());
        let thinking = result.thinking.unwrap();
        assert!(thinking.contains("step 1"));
        assert!(thinking.contains("step 2"));
    }

    #[test]
    fn test_no_think_blocks() {
        let parser = ThinkParser::new();
        let text = "Just a regular response.";
        let result = parser.parse(text);
        assert!(result.thinking.is_none());
        assert_eq!(result.content, "Just a regular response.");
    }

    #[test]
    fn test_multiline_think() {
        let parser = ThinkParser::new();
        let text = "<think>\nLine 1\nLine 2\nLine 3\n</think>\nAnswer here.";
        let result = parser.parse(text);
        assert!(result.thinking.is_some());
        let thinking = result.thinking.unwrap();
        assert!(thinking.contains("Line 1"));
        assert!(thinking.contains("Line 3"));
        assert_eq!(result.content, "Answer here.");
    }

    #[test]
    fn test_streaming_basic() {
        let mut parser = ThinkParser::new();

        let r1 = parser.parse_streaming("<think>");
        // May or may not emit yet
        let r2 = parser.parse_streaming("thinking here");
        let r3 = parser.parse_streaming("</think>");
        let r4 = parser.parse_streaming("visible content");

        // After all deltas, the last result should have both
        let final_result = r4.or(r3).or(r2).or(r1).unwrap();
        assert!(final_result.thinking.is_some());
        assert!(final_result.content.contains("visible"));
    }
}
