//! GPT-OSS reasoning parser.
//!
//! Handles `<reasoning>...</reasoning>` tags used by some open-source GPT-style models.

use regex::Regex;
use std::sync::LazyLock;

use crate::reasoning_parser::ReasoningParser;
use crate::types::ReasoningParseResult;

static REASONING_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<reasoning>(.*?)</reasoning>").unwrap()
});

const OPEN_TAG: &str = "<reasoning>";
const CLOSE_TAG: &str = "</reasoning>";

/// Parser for GPT-OSS style `<reasoning>` blocks.
pub struct GptOssParser {
    in_reasoning: bool,
    reasoning_buffer: String,
    content_buffer: String,
    raw_buffer: String,
}

impl GptOssParser {
    pub fn new() -> Self {
        Self {
            in_reasoning: false,
            reasoning_buffer: String::new(),
            content_buffer: String::new(),
            raw_buffer: String::new(),
        }
    }
}

impl Default for GptOssParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for GptOssParser {
    fn parse(&self, text: &str) -> ReasoningParseResult {
        let mut reasoning_parts = Vec::new();

        for cap in REASONING_RE.captures_iter(text) {
            let reasoning_content = cap.get(1).unwrap().as_str().trim();
            if !reasoning_content.is_empty() {
                reasoning_parts.push(reasoning_content.to_string());
            }
        }

        let content = REASONING_RE.replace_all(text, "").to_string();
        let content = content.trim().to_string();

        let thinking = if reasoning_parts.is_empty() {
            None
        } else {
            Some(reasoning_parts.join("\n\n"))
        };

        ReasoningParseResult { thinking, content }
    }

    fn parse_streaming(&mut self, delta: &str) -> Option<ReasoningParseResult> {
        self.raw_buffer.push_str(delta);

        loop {
            if self.in_reasoning {
                if let Some(end_pos) = self.raw_buffer.find(CLOSE_TAG) {
                    let reasoning_part = &self.raw_buffer[..end_pos];
                    self.reasoning_buffer.push_str(reasoning_part);
                    self.in_reasoning = false;
                    self.raw_buffer = self.raw_buffer[end_pos + CLOSE_TAG.len()..].to_string();
                } else {
                    let mut partial = false;
                    for i in 1..CLOSE_TAG.len() {
                        if self.raw_buffer.ends_with(&CLOSE_TAG[..i]) {
                            partial = true;
                            break;
                        }
                    }
                    if partial {
                        return None;
                    }
                    self.reasoning_buffer.push_str(&self.raw_buffer);
                    self.raw_buffer.clear();
                    return None;
                }
            } else {
                if let Some(start_pos) = self.raw_buffer.find(OPEN_TAG) {
                    let content_part = &self.raw_buffer[..start_pos];
                    self.content_buffer.push_str(content_part);
                    self.in_reasoning = true;
                    self.raw_buffer = self.raw_buffer[start_pos + OPEN_TAG.len()..].to_string();
                } else {
                    let mut partial_len = 0;
                    for i in 1..OPEN_TAG.len() {
                        if self.raw_buffer.ends_with(&OPEN_TAG[..i]) {
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
        }

        let thinking = if self.reasoning_buffer.is_empty() {
            None
        } else {
            Some(self.reasoning_buffer.clone())
        };

        Some(ReasoningParseResult {
            thinking,
            content: self.content_buffer.clone(),
        })
    }

    fn reset(&mut self) {
        self.in_reasoning = false;
        self.reasoning_buffer.clear();
        self.content_buffer.clear();
        self.raw_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_block() {
        let parser = GptOssParser::new();
        let text = "<reasoning>Step 1: analyze\nStep 2: conclude</reasoning>\nThe answer is 42.";
        let result = parser.parse(text);
        assert!(result.thinking.is_some());
        let thinking = result.thinking.unwrap();
        assert!(thinking.contains("Step 1"));
        assert_eq!(result.content, "The answer is 42.");
    }

    #[test]
    fn test_no_reasoning() {
        let parser = GptOssParser::new();
        let text = "Direct response.";
        let result = parser.parse(text);
        assert!(result.thinking.is_none());
        assert_eq!(result.content, "Direct response.");
    }

    #[test]
    fn test_multiple_reasoning_blocks() {
        let parser = GptOssParser::new();
        let text = "<reasoning>first</reasoning>middle<reasoning>second</reasoning>end";
        let result = parser.parse(text);
        assert!(result.thinking.is_some());
        let thinking = result.thinking.unwrap();
        assert!(thinking.contains("first"));
        assert!(thinking.contains("second"));
        assert!(result.content.contains("middle"));
        assert!(result.content.contains("end"));
    }
}
