//! Qwen3 reasoning parser.
//!
//! Handles `<think>...</think>` blocks with Qwen3-specific behavior:
//! - Qwen3 always starts with `<think>` when reasoning is enabled
//! - The model may produce multiple think blocks
//! - Think blocks are always well-formed (unlike DeepSeek)

use regex::Regex;
use std::sync::LazyLock;

use crate::reasoning_parser::ReasoningParser;
use crate::types::ReasoningParseResult;

static THINK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<think>(.*?)</think>").unwrap()
});

/// Parser for Qwen3-style reasoning blocks.
pub struct Qwen3Parser {
    in_think: bool,
    think_buffer: String,
    content_buffer: String,
    raw_buffer: String,
}

impl Qwen3Parser {
    pub fn new() -> Self {
        Self {
            in_think: false,
            think_buffer: String::new(),
            content_buffer: String::new(),
            raw_buffer: String::new(),
        }
    }
}

impl Default for Qwen3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for Qwen3Parser {
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

        loop {
            if self.in_think {
                if let Some(end_pos) = self.raw_buffer.find("</think>") {
                    let think_part = &self.raw_buffer[..end_pos];
                    self.think_buffer.push_str(think_part);
                    self.in_think = false;
                    self.raw_buffer = self.raw_buffer[end_pos + "</think>".len()..].to_string();
                } else {
                    let potential_tag = "</think>";
                    let mut partial = false;
                    for i in 1..potential_tag.len() {
                        if self.raw_buffer.ends_with(&potential_tag[..i]) {
                            partial = true;
                            break;
                        }
                    }
                    if partial {
                        return None;
                    }
                    self.think_buffer.push_str(&self.raw_buffer);
                    self.raw_buffer.clear();
                    return None;
                }
            } else if let Some(start_pos) = self.raw_buffer.find("<think>") {
                let content_part = &self.raw_buffer[..start_pos];
                self.content_buffer.push_str(content_part);
                self.in_think = true;
                self.raw_buffer = self.raw_buffer[start_pos + "<think>".len()..].to_string();
            } else {
                let potential_tag = "<think>";
                let mut partial_len = 0;
                for i in 1..potential_tag.len() {
                    if self.raw_buffer.ends_with(&potential_tag[..i]) {
                        partial_len = i;
                        break;
                    }
                }
                if partial_len > 0 {
                    let safe_end = self.raw_buffer.len() - partial_len;
                    self.content_buffer
                        .push_str(&self.raw_buffer[..safe_end]);
                    self.raw_buffer = self.raw_buffer[safe_end..].to_string();
                    break;
                }
                self.content_buffer.push_str(&self.raw_buffer);
                self.raw_buffer.clear();
                break;
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
    fn test_qwen3_standard() {
        let parser = Qwen3Parser::new();
        let text = "<think>\nI need to consider this carefully.\n\nStep 1: analyze.\nStep 2: conclude.\n</think>\n\nThe answer is 42.";
        let result = parser.parse(text);
        assert!(result.thinking.is_some());
        let thinking = result.thinking.unwrap();
        assert!(thinking.contains("Step 1"));
        assert!(thinking.contains("Step 2"));
        assert_eq!(result.content, "The answer is 42.");
    }

    #[test]
    fn test_qwen3_no_thinking() {
        let parser = Qwen3Parser::new();
        let text = "Direct answer without thinking.";
        let result = parser.parse(text);
        assert!(result.thinking.is_none());
        assert_eq!(result.content, "Direct answer without thinking.");
    }

    #[test]
    fn test_qwen3_empty_think() {
        let parser = Qwen3Parser::new();
        let text = "<think>\n\n</think>\nQuick answer.";
        let result = parser.parse(text);
        assert!(result.thinking.is_none());
        assert_eq!(result.content, "Quick answer.");
    }
}
