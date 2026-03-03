//! MCP client manager.
//!
//! Manages connections to MCP tool servers and routes tool execution requests.

use rmlx_serve_types::mcp::MCPTool;

/// Manages multiple MCP server connections and provides a unified interface
/// for listing tools and executing tool calls.
pub struct McpClientManager;

impl McpClientManager {
    /// Return all tools available across all connected MCP servers.
    pub async fn all_tools(&self) -> Vec<MCPTool> {
        Vec::new()
    }

    /// Execute a tool on a specific MCP server.
    pub async fn execute(
        &self,
        _server: &str,
        _tool: &str,
        _arguments: serde_json::Value,
    ) -> Result<serde_json::Value, McpError> {
        Err(McpError("MCP execution not yet implemented".to_string()))
    }
}

/// Error type for MCP operations.
#[derive(Debug)]
pub struct McpError(pub String);

impl std::fmt::Display for McpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for McpError {}
