//! The [`ReasoningParser`] trait and [`ReasoningParserRegistry`].

use std::collections::HashMap;

use crate::types::ReasoningParseResult;

/// Trait implemented by reasoning/thinking block parsers.
///
/// These parsers detect model "thinking" sections (e.g. `<think>...</think>`)
/// and separate them from the visible output content.
pub trait ReasoningParser: Send + Sync {
    /// Parse a complete model output, separating thinking from content.
    fn parse(&self, text: &str) -> ReasoningParseResult;

    /// Incrementally parse a streaming delta.
    ///
    /// Returns `Some` when there is a meaningful update, `None` when the delta
    /// is still being buffered (e.g. accumulating inside a thinking block).
    fn parse_streaming(&mut self, delta: &str) -> Option<ReasoningParseResult>;

    /// Reset internal streaming state.
    fn reset(&mut self);
}

/// A factory function that creates a fresh reasoning parser instance.
pub type ReasoningParserFactory = fn() -> Box<dyn ReasoningParser>;

/// Registry of reasoning parser backends, keyed by name.
pub struct ReasoningParserRegistry {
    factories: HashMap<String, ReasoningParserFactory>,
}

impl ReasoningParserRegistry {
    /// Create a registry pre-populated with all built-in reasoning parsers.
    pub fn new() -> Self {
        let mut reg = Self {
            factories: HashMap::new(),
        };
        reg.register("think", || {
            Box::new(crate::reasoning::think::ThinkParser::new())
        });
        reg.register("deepseek_r1", || {
            Box::new(crate::reasoning::deepseek_r1::DeepSeekR1Parser::new())
        });
        reg.register("qwen3", || {
            Box::new(crate::reasoning::qwen3::Qwen3Parser::new())
        });
        reg.register("gpt_oss", || {
            Box::new(crate::reasoning::gpt_oss::GptOssParser::new())
        });
        reg.register("harmony", || {
            Box::new(crate::reasoning::harmony::HarmonyParser::new())
        });
        reg
    }

    /// Retrieve a fresh reasoning parser instance by name.
    pub fn get(&self, name: &str) -> Option<Box<dyn ReasoningParser>> {
        self.factories.get(name).map(|f| f())
    }

    /// Register a reasoning parser factory under the given name.
    pub fn register(&mut self, name: &str, factory: ReasoningParserFactory) {
        self.factories.insert(name.to_string(), factory);
    }

    /// List all registered parser names.
    pub fn names(&self) -> Vec<&str> {
        self.factories.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ReasoningParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}
