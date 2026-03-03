//! Harmony reasoning parser.
//!
//! Handles `<reflection>...</reflection>` tags used in harmony-style reasoning.

use regex::Regex;
use std::sync::LazyLock;

use crate::reasoning_parser::ReasoningParser;
use crate::types::ReasoningParseResult;

static REFLECTION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<reflection>(.*?)</reflection>").unwrap()
});

const OPEN_TAG: &str = "<reflection>";
const CLOSE_TAG: &str = "</reflection>";

/// Parser for harmony-style `<reflection>` reasoning blocks.
pub struct HarmonyParser {
    in_reflection: bool,
    reflection_buffer: String,
    content_buffer: String,
    raw_buffer: String,
}

impl HarmonyParser {
    pub fn new() -> Self {
        Self {
            in_reflection: false,
            reflection_buffer: String::new(),
            content_buffer: String::new(),
            raw_buffer: String::new(),
        }
    }
}

impl Default for HarmonyParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for HarmonyParser {
    fn parse(&self, text: &str) -> ReasoningParseResult {
        let mut reflection_parts = Vec::new();

        for cap in REFLECTION_RE.captures_iter(text) {
            let reflection_content = cap.get(1).unwrap().as_str().trim();
            if !reflection_content.is_empty() {
                reflection_parts.push(reflection_content.to_string());
            }
        }

        let content = REFLECTION_RE.replace_all(text, "").to_string();
        let content = content.trim().to_string();

        let thinking = if reflection_parts.is_empty() {
            None
        } else {
            Some(reflection_parts.join("\n\n"))
        };

        ReasoningParseResult { thinking, content }
    }

    fn parse_streaming(&mut self, delta: &str) -> Option<ReasoningParseResult> {
        self.raw_buffer.push_str(delta);

        loop {
            if self.in_reflection {
                if let Some(end_pos) = self.raw_buffer.find(CLOSE_TAG) {
                    let reflection_part = &self.raw_buffer[..end_pos];
                    self.reflection_buffer.push_str(reflection_part);
                    self.in_reflection = false;
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
                    self.reflection_buffer.push_str(&self.raw_buffer);
                    self.raw_buffer.clear();
                    return None;
                }
            } else {
                if let Some(start_pos) = self.raw_buffer.find(OPEN_TAG) {
                    let content_part = &self.raw_buffer[..start_pos];
                    self.content_buffer.push_str(content_part);
                    self.in_reflection = true;
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

        let thinking = if self.reflection_buffer.is_empty() {
            None
        } else {
            Some(self.reflection_buffer.clone())
        };

        Some(ReasoningParseResult {
            thinking,
            content: self.content_buffer.clone(),
        })
    }

    fn reset(&mut self) {
        self.in_reflection = false;
        self.reflection_buffer.clear();
        self.content_buffer.clear();
        self.raw_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflection_block() {
        let parser = HarmonyParser::new();
        let text =
            "<reflection>Let me reconsider...</reflection>\nActually, the answer is 7.";
        let result = parser.parse(text);
        assert!(result.thinking.is_some());
        assert!(result.thinking.unwrap().contains("reconsider"));
        assert_eq!(result.content, "Actually, the answer is 7.");
    }

    #[test]
    fn test_no_reflection() {
        let parser = HarmonyParser::new();
        let text = "No reflection here.";
        let result = parser.parse(text);
        assert!(result.thinking.is_none());
        assert_eq!(result.content, "No reflection here.");
    }

    #[test]
    fn test_empty_reflection() {
        let parser = HarmonyParser::new();
        let text = "<reflection></reflection>Answer.";
        let result = parser.parse(text);
        assert!(result.thinking.is_none());
        assert_eq!(result.content, "Answer.");
    }
}
