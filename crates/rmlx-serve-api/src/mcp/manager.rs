//! Multi-server MCP client manager.
//!
//! Aggregates multiple [`McpClient`] instances, provides unified tool
//! discovery with conflict-free naming, and routes tool execution requests
//! to the appropriate server.

use std::collections::HashMap;

use serde_json::Value;

use rmlx_serve_types::mcp::MCPTool;

use super::client::{McpClient, McpClientError};
use super::config::McpConfig;
use super::security::McpSecurity;

/// Manages multiple MCP server connections.
pub struct McpClientManager {
    /// Map of server name -> connected client.
    clients: HashMap<String, McpClient>,
}

impl McpClientManager {
    /// Initialize all configured MCP servers.
    ///
    /// Servers that fail to start are logged as warnings but do not prevent
    /// the remaining servers from starting.
    pub async fn new(config: &McpConfig) -> Result<Self, McpClientError> {
        let mut clients = HashMap::new();

        for (name, server_config) in &config.servers {
            match McpClient::new(name, server_config).await {
                Ok(client) => {
                    tracing::info!("MCP server {:?} connected", name);
                    clients.insert(name.clone(), client);
                }
                Err(e) => {
                    tracing::warn!("Failed to start MCP server {:?}: {}", name, e);
                }
            }
        }

        Ok(Self { clients })
    }

    /// Aggregate tools from all connected servers.
    ///
    /// When multiple servers provide a tool with the same name, the tool name
    /// is prefixed with the server name (e.g. `server_name__tool_name`) to
    /// avoid conflicts.
    pub async fn all_tools(&self) -> Vec<MCPTool> {
        let mut server_tools: Vec<(String, Vec<MCPTool>)> = Vec::new();

        for (name, client) in &self.clients {
            match client.discover_tools().await {
                Ok(tools) => {
                    server_tools.push((name.clone(), tools));
                }
                Err(e) => {
                    tracing::warn!("Failed to discover tools from MCP server {:?}: {}", name, e);
                }
            }
        }

        // Build a frequency map of tool names to detect conflicts.
        let mut name_counts: HashMap<String, usize> = HashMap::new();
        for (_server_name, tools) in &server_tools {
            for tool in tools {
                *name_counts.entry(tool.name.clone()).or_insert(0) += 1;
            }
        }

        // Flatten, prefixing duplicates with the server name.
        let mut all_tools = Vec::new();
        for (server_name, tools) in server_tools {
            for mut tool in tools {
                if name_counts.get(&tool.name).copied().unwrap_or(0) > 1 {
                    tool.name = format!("{}__{}", server_name, tool.name);
                }
                tool.server_name = Some(server_name.clone());
                all_tools.push(tool);
            }
        }

        all_tools
    }

    /// Execute a tool on a specific server.
    pub async fn execute(
        &self,
        server: &str,
        tool: &str,
        args: Value,
    ) -> Result<Value, McpClientError> {
        let client = self.clients.get(server).ok_or_else(|| {
            McpClientError::ServerError(format!("MCP server {:?} not found", server))
        })?;

        // Health check and restart if needed.
        client.restart_if_needed().await?;

        let result = client.execute_tool(tool, args).await;

        // Audit log.
        McpSecurity::audit_log(server, tool, result.is_ok());

        result
    }

    /// Gracefully shut down all connected MCP servers.
    pub async fn shutdown(&self) {
        for (name, client) in &self.clients {
            tracing::info!("Shutting down MCP server {:?}", name);
            client.shutdown().await;
        }
    }

    /// Return the names of all connected servers.
    pub fn server_names(&self) -> Vec<String> {
        self.clients.keys().cloned().collect()
    }

    /// Check whether a specific server is connected and alive.
    pub async fn is_server_alive(&self, name: &str) -> bool {
        if let Some(client) = self.clients.get(name) {
            client.is_alive().await
        } else {
            false
        }
    }
}
