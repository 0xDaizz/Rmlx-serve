//! Shared application state for all HTTP handlers.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use rmlx_serve_engine::Engine;
use rmlx_serve_tools::{ReasoningParserRegistry, ToolParserRegistry};
use rmlx_serve_types::config::ServerConfig;

use crate::mcp::manager::McpClientManager;
use crate::middleware::RateLimiter;

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

    /// Name of the active tool-call parser (e.g. `"hermes"`, `"llama"`).
    /// `None` means tool-call detection is disabled.
    pub tool_parser_name: Option<String>,

    /// Name of the active reasoning parser (e.g. `"think"`, `"deepseek_r1"`).
    /// `None` means reasoning extraction is disabled.
    pub reasoning_parser_name: Option<String>,

    /// Monotonically increasing request counter (for metrics / request IDs).
    pub request_count: AtomicU64,

    /// The instant the server was started (for uptime calculation).
    pub start_time: std::time::Instant,

    /// Per-IP sliding-window rate limiter.
    pub rate_limiter: RateLimiter,

    /// MCP client manager for connected tool servers.
    /// `None` when MCP is not configured.
    pub mcp_manager: Option<Arc<McpClientManager>>,
}

impl AppState {
    /// Create a new `AppState` wrapped in an `Arc`, ready for injection into
    /// the Axum router.
    pub fn new(
        engine: Arc<dyn Engine>,
        config: ServerConfig,
        mcp_manager: Option<Arc<McpClientManager>>,
    ) -> Arc<Self> {
        let tool_parser_name = if config.enable_auto_tool_choice {
            Some(
                config
                    .tool_call_parser
                    .clone()
                    .unwrap_or_else(|| "auto".to_string()),
            )
        } else {
            None
        };

        let reasoning_parser_name = if config.enable_thinking {
            Some(
                config
                    .reasoning_parser
                    .clone()
                    .unwrap_or_else(|| "think".to_string()),
            )
        } else {
            None
        };

        let rate_limiter = RateLimiter::new(config.rate_limit);
        Arc::new(Self {
            engine,
            config,
            tool_parser_registry: ToolParserRegistry::new(),
            reasoning_parser_registry: ReasoningParserRegistry::new(),
            tool_parser_name,
            reasoning_parser_name,
            request_count: AtomicU64::new(0),
            start_time: std::time::Instant::now(),
            rate_limiter,
            mcp_manager,
        })
    }
}
