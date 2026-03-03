//! Shared application state for all HTTP handlers.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use rmlx_serve_engine::Engine;
use rmlx_serve_tools::{ReasoningParserRegistry, ToolParserRegistry};
use rmlx_serve_types::config::ServerConfig;

/// Shared application state passed to every request handler via Axum's
/// `State` extractor.
pub struct AppState {
    /// The inference engine backing all generation requests.
    pub engine: Arc<dyn Engine>,

    /// Server configuration (host, port, auth, feature flags, etc.).
    pub config: ServerConfig,

    /// Registry of tool-call parsers (Hermes, Llama, Mistral, ...).
    pub tool_parser_registry: ToolParserRegistry,

    /// Registry of reasoning/thinking parsers (think, deepseek_r1, ...).
    pub reasoning_parser_registry: ReasoningParserRegistry,

    /// Monotonically increasing request counter (for metrics / request IDs).
    pub request_count: AtomicU64,

    /// The instant the server was started (for uptime calculation).
    pub start_time: std::time::Instant,
}

impl AppState {
    /// Create a new `AppState` wrapped in an `Arc`, ready for injection into
    /// the Axum router.
    pub fn new(engine: Arc<dyn Engine>, config: ServerConfig) -> Arc<Self> {
        Arc::new(Self {
            engine,
            config,
            tool_parser_registry: ToolParserRegistry::new(),
            reasoning_parser_registry: ReasoningParserRegistry::new(),
            request_count: AtomicU64::new(0),
            start_time: std::time::Instant::now(),
        })
    }
}
