//! MCP client implementation (stdio transport).
//!
//! Communicates with an MCP server process via JSON-RPC 2.0 over stdin/stdout.
//! Manages the process lifecycle including spawning, health checks, and
//! restart on crash.

use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use rmlx_serve_types::mcp::MCPTool;

use super::config::McpServerConfig;
use super::security::McpSecurity;

// ---------------------------------------------------------------------------
// JSON-RPC types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Deserialize, Debug)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    #[allow(dead_code)]
    id: Option<u64>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize, Debug)]
struct JsonRpcError {
    #[allow(dead_code)]
    code: i64,
    message: String,
    #[allow(dead_code)]
    data: Option<Value>,
}

// ---------------------------------------------------------------------------
// McpClient
// ---------------------------------------------------------------------------

/// A client connected to a single MCP server process via stdio.
pub struct McpClient {
    /// The server name (for logging / namespacing).
    server_name: String,

    /// Original configuration, kept for restart.
    config: McpServerConfig,

    /// The child process handle, protected by a mutex for exclusive write
    /// access to stdin.
    child: Arc<Mutex<Option<Child>>>,

    /// Buffered reader over the child's stdout.
    stdout_reader: Arc<Mutex<Option<BufReader<tokio::process::ChildStdout>>>>,

    /// Monotonically increasing JSON-RPC request id.
    next_id: AtomicU64,
}

impl McpClient {
    /// Spawn a new MCP server process and return a client connected to it.
    pub async fn new(
        server_name: &str,
        config: &McpServerConfig,
    ) -> Result<Self, McpClientError> {
        // Validate command against security rules.
        let security = McpSecurity::permissive();
        if !security.is_command_allowed(&config.command) {
            return Err(McpClientError::SecurityViolation(format!(
                "command {:?} is not in the whitelist",
                config.command
            )));
        }
        if McpSecurity::has_dangerous_patterns(&config.args) {
            return Err(McpClientError::SecurityViolation(
                "dangerous patterns detected in server arguments".to_string(),
            ));
        }

        let client = Self {
            server_name: server_name.to_string(),
            config: config.clone(),
            child: Arc::new(Mutex::new(None)),
            stdout_reader: Arc::new(Mutex::new(None)),
            next_id: AtomicU64::new(1),
        };

        client.spawn_process().await?;
        client.initialize().await?;

        Ok(client)
    }

    /// Spawn (or re-spawn) the server process.
    async fn spawn_process(&self) -> Result<(), McpClientError> {
        tracing::info!(
            "Spawning MCP server {:?}: {} {}",
            self.server_name,
            self.config.command,
            self.config.args.join(" "),
        );

        let mut cmd = Command::new(&self.config.command);
        cmd.args(&self.config.args)
            .envs(&self.config.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            McpClientError::SpawnFailed(format!(
                "failed to spawn {:?}: {}",
                self.config.command, e
            ))
        })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpClientError::Protocol("failed to capture stdout".to_string()))?;

        let reader = BufReader::new(stdout);

        *self.child.lock().await = Some(child);
        *self.stdout_reader.lock().await = Some(reader);

        Ok(())
    }

    /// Send the MCP `initialize` handshake to the server.
    async fn initialize(&self) -> Result<(), McpClientError> {
        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "rmlx-serve",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let response = self.send_request("initialize", Some(params)).await?;
        tracing::info!(
            "MCP server {:?} initialized: {:?}",
            self.server_name,
            response,
        );

        // Send initialized notification (no id, no response expected).
        self.send_notification("notifications/initialized", None)
            .await?;

        Ok(())
    }

    /// Discover the tools advertised by this MCP server.
    pub async fn discover_tools(&self) -> Result<Vec<MCPTool>, McpClientError> {
        let response = self.send_request("tools/list", None).await?;

        let tools_value = response
            .as_ref()
            .and_then(|v| v.get("tools"))
            .cloned()
            .unwrap_or(Value::Array(vec![]));

        let raw_tools: Vec<RawMcpTool> =
            serde_json::from_value(tools_value).map_err(|e| {
                McpClientError::Protocol(format!("failed to parse tools/list response: {}", e))
            })?;

        let tools = raw_tools
            .into_iter()
            .map(|t| MCPTool {
                name: t.name,
                description: t.description,
                input_schema: t.input_schema,
                server_name: Some(self.server_name.clone()),
            })
            .collect();

        Ok(tools)
    }

    /// Execute a tool on this MCP server.
    pub async fn execute_tool(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<Value, McpClientError> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });

        let response = self.send_request("tools/call", Some(params)).await?;

        response.ok_or_else(|| {
            McpClientError::Protocol("tools/call returned no result".to_string())
        })
    }

    /// Check whether the server process is still alive.
    pub async fn is_alive(&self) -> bool {
        let mut guard = self.child.lock().await;
        if let Some(ref mut child) = *guard {
            match child.try_wait() {
                Ok(None) => true,  // still running
                Ok(Some(_)) => false, // exited
                Err(_) => false,
            }
        } else {
            false
        }
    }

    /// Restart the server process if it has crashed.
    pub async fn restart_if_needed(&self) -> Result<(), McpClientError> {
        if !self.is_alive().await {
            tracing::warn!(
                "MCP server {:?} is not alive, restarting...",
                self.server_name
            );
            self.spawn_process().await?;
            self.initialize().await?;
        }
        Ok(())
    }

    /// Gracefully shut down the server process.
    pub async fn shutdown(&self) {
        let mut guard = self.child.lock().await;
        if let Some(ref mut child) = *guard {
            // Try to kill the process gracefully.
            let _ = child.kill().await;
            tracing::info!("MCP server {:?} stopped", self.server_name);
        }
        *guard = None;
        *self.stdout_reader.lock().await = None;
    }

    // -----------------------------------------------------------------------
    // Internal transport helpers
    // -----------------------------------------------------------------------

    /// Send a JSON-RPC request and wait for the response.
    async fn send_request(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Option<Value>, McpClientError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method: method.to_string(),
            params,
        };

        let mut payload =
            serde_json::to_string(&request).map_err(|e| {
                McpClientError::Protocol(format!("failed to serialize request: {}", e))
            })?;
        payload.push('\n');

        // Write to stdin.
        {
            let mut guard = self.child.lock().await;
            let child = guard
                .as_mut()
                .ok_or_else(|| McpClientError::Protocol("server process not running".to_string()))?;
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| McpClientError::Protocol("stdin not available".to_string()))?;
            stdin
                .write_all(payload.as_bytes())
                .await
                .map_err(|e| McpClientError::Protocol(format!("failed to write to stdin: {}", e)))?;
            stdin
                .flush()
                .await
                .map_err(|e| McpClientError::Protocol(format!("failed to flush stdin: {}", e)))?;
        }

        // Read response from stdout.
        let response = self.read_response().await?;

        if let Some(err) = response.error {
            return Err(McpClientError::ServerError(err.message));
        }

        Ok(response.result)
    }

    /// Send a JSON-RPC notification (no id, no response expected).
    async fn send_notification(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<(), McpClientError> {
        #[derive(Serialize)]
        struct JsonRpcNotification {
            jsonrpc: &'static str,
            method: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            params: Option<Value>,
        }

        let notification = JsonRpcNotification {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
        };

        let mut payload = serde_json::to_string(&notification).map_err(|e| {
            McpClientError::Protocol(format!("failed to serialize notification: {}", e))
        })?;
        payload.push('\n');

        let mut guard = self.child.lock().await;
        let child = guard
            .as_mut()
            .ok_or_else(|| McpClientError::Protocol("server process not running".to_string()))?;
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| McpClientError::Protocol("stdin not available".to_string()))?;
        stdin
            .write_all(payload.as_bytes())
            .await
            .map_err(|e| McpClientError::Protocol(format!("failed to write notification: {}", e)))?;
        stdin
            .flush()
            .await
            .map_err(|e| McpClientError::Protocol(format!("failed to flush notification: {}", e)))?;

        Ok(())
    }

    /// Read a single JSON-RPC response line from the server's stdout.
    async fn read_response(&self) -> Result<JsonRpcResponse, McpClientError> {
        let mut guard = self.stdout_reader.lock().await;
        let reader = guard.as_mut().ok_or_else(|| {
            McpClientError::Protocol("stdout reader not available".to_string())
        })?;

        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line).await.map_err(|e| {
                McpClientError::Protocol(format!("failed to read from stdout: {}", e))
            })?;

            if bytes_read == 0 {
                return Err(McpClientError::Protocol(
                    "server closed stdout (EOF)".to_string(),
                ));
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue; // skip blank lines
            }

            // Attempt to parse as JSON-RPC response.  Some servers emit
            // notifications or log lines that are not responses -- skip them.
            match serde_json::from_str::<JsonRpcResponse>(trimmed) {
                Ok(resp) => return Ok(resp),
                Err(_) => {
                    // Not a valid JSON-RPC response; could be a notification
                    // or log line. Skip and try next line.
                    tracing::trace!(
                        "Skipping non-response line from MCP server {:?}: {}",
                        self.server_name,
                        trimmed
                    );
                    continue;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

/// Raw tool definition as returned by tools/list (before conversion to MCPTool).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawMcpTool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default = "default_input_schema")]
    input_schema: serde_json::Value,
}

fn default_input_schema() -> serde_json::Value {
    serde_json::json!({"type": "object"})
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur in the MCP client.
#[derive(Debug, thiserror::Error)]
pub enum McpClientError {
    /// Failed to spawn the server process.
    #[error("spawn failed: {0}")]
    SpawnFailed(String),

    /// JSON-RPC protocol error.
    #[error("protocol error: {0}")]
    Protocol(String),

    /// The server returned an error response.
    #[error("server error: {0}")]
    ServerError(String),

    /// A security violation was detected.
    #[error("security violation: {0}")]
    SecurityViolation(String),
}
