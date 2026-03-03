//! `rmlx-serve-tokenizer` -- tokenization, chat-template rendering, and
//! streaming detokenization for the rmlx-serve inference engine.
//!
//! This crate wraps the HuggingFace `tokenizers` crate with higher-level
//! helpers that are specific to serving LLMs:
//!
//! - **[`Tokenizer`]** -- loads `tokenizer.json` + `tokenizer_config.json`,
//!   resolves special-token IDs, and supports configurable stop tokens.
//! - **[`ChatTemplate`]** -- Jinja2-based chat-template rendering (with ChatML
//!   fallback).
//! - **[`StreamingDetokenizer`]** -- incremental text decoding with
//!   auto-selected SPM / BPE / Naive backends.
//!
//! # Dependency policy
//!
//! This crate has **no RMLX dependencies** -- only `rmlx-serve-types`,
//! `tokenizers`, `minijinja`, `serde`, `serde_json`, `thiserror`, and `tracing`.

pub mod chat_template;
pub mod detokenizer;
pub mod error;
pub mod tokenizer;

// Re-export the main public types at the crate root for convenience.
pub use chat_template::{ChatTemplate, TemplateMessage};
pub use detokenizer::{
    create_detokenizer, detect_type, BPEStreamingDetokenizer, DetokenizerType,
    NaiveStreamingDetokenizer, SPMStreamingDetokenizer, StreamingDetokenizer,
};
pub use error::{Result, TokenizerError};
pub use tokenizer::Tokenizer;
