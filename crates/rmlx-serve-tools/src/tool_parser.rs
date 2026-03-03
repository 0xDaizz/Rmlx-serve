//! The [`ToolCallParser`] trait and [`ToolParserRegistry`].

use std::collections::HashMap;

use crate::types::{StreamingParseResult, ToolCallParseResult};

/// Trait implemented by every tool-call parser backend.
///
/// Parsers detect tool-call patterns in generated text and extract structured
/// [`ParsedToolCall`](crate::types::ParsedToolCall) values.
pub trait ToolCallParser: Send + Sync {
    /// Parse a complete model output and extract tool calls.
    fn parse(&self, text: &str) -> ToolCallParseResult;

    /// Incrementally parse a streaming delta.
    ///
    /// * `prev` - all text generated so far *before* this step.
    /// * `curr` - all text generated so far *including* this step.
    /// * `delta` - the new text added in this step (`curr[prev.len()..]`).
    fn parse_streaming(&mut self, prev: &str, curr: &str, delta: &str) -> StreamingParseResult;

    /// Reset any internal streaming state (call between requests).
    fn reset(&mut self);

    /// Whether this model family supports native tool-format tokens in its
    /// tokenizer / chat template.  Models that return `true` (e.g. Hermes,
    /// Llama, Mistral, DeepSeek) can be prompted with structured tool
    /// definitions; models that return `false` rely on free-form text parsing.
    ///
    /// The default implementation returns `false`.
    fn supports_native_tool_format(&self) -> bool {
        false
    }
}

/// A factory function that creates a fresh parser instance.
pub type ToolParserFactory = fn() -> Box<dyn ToolCallParser>;

/// Registry of tool-call parser backends, keyed by name.
pub struct ToolParserRegistry {
    factories: HashMap<String, ToolParserFactory>,
}

impl ToolParserRegistry {
    /// Create a registry pre-populated with all built-in parsers.
    pub fn new() -> Self {
        let mut reg = Self {
            factories: HashMap::new(),
        };
        // Register built-in parsers
        reg.register("hermes", || {
            Box::new(crate::parsers::hermes::HermesToolParser::new())
        });
        reg.register("auto", || {
            Box::new(crate::parsers::auto::AutoToolParser::new())
        });
        reg.register("llama", || {
            Box::new(crate::parsers::llama::LlamaToolParser::new())
        });
        // Aliases: llama3 and llama4 map to the Llama parser
        reg.register("llama3", || {
            Box::new(crate::parsers::llama::LlamaToolParser::new())
        });
        reg.register("llama4", || {
            Box::new(crate::parsers::llama::LlamaToolParser::new())
        });
        reg.register("mistral", || {
            Box::new(crate::parsers::mistral::MistralToolParser::new())
        });
        reg.register("qwen", || {
            Box::new(crate::parsers::qwen::QwenToolParser::new())
        });
        // Alias: qwen3 maps to the Qwen parser (which supports Hermes-style <tool_call> tags too)
        reg.register("qwen3", || {
            Box::new(crate::parsers::qwen::QwenToolParser::new())
        });
        reg.register("deepseek", || {
            Box::new(crate::parsers::deepseek::DeepSeekToolParser::new())
        });
        // Aliases: deepseek_v3 and deepseek_r1 map to the DeepSeek parser
        reg.register("deepseek_v3", || {
            Box::new(crate::parsers::deepseek::DeepSeekToolParser::new())
        });
        reg.register("deepseek_r1", || {
            Box::new(crate::parsers::deepseek::DeepSeekToolParser::new())
        });
        reg.register("glm4_7", || {
            Box::new(crate::parsers::generic::GLM47ToolParser::new())
        });
        reg.register("granite", || {
            Box::new(crate::parsers::generic::GraniteToolParser::new())
        });
        reg.register("xlam", || {
            Box::new(crate::parsers::generic::XLamToolParser::new())
        });
        reg.register("kimi", || {
            Box::new(crate::parsers::generic::KimiToolParser::new())
        });
        reg.register("nemotron", || {
            Box::new(crate::parsers::generic::NemotronToolParser::new())
        });
        reg.register("functionary", || {
            Box::new(crate::parsers::generic::FunctionaryToolParser::new())
        });
        reg.register("harmony", || {
            Box::new(crate::parsers::generic::HarmonyToolParser::new())
        });
        reg
    }

    /// Retrieve a fresh parser instance by name.
    pub fn get(&self, name: &str) -> Option<Box<dyn ToolCallParser>> {
        self.factories.get(name).map(|f| f())
    }

    /// Register a parser factory under the given name.
    pub fn register(&mut self, name: &str, factory: ToolParserFactory) {
        self.factories.insert(name.to_string(), factory);
    }

    /// List all registered parser names.
    pub fn names(&self) -> Vec<&str> {
        self.factories.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ToolParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}
