//! MCP (Model Context Protocol) types.
//!
//! These types model the configuration and runtime state of MCP tool servers
//! that can be connected to the inference engine.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===========================================================================
// Transport
// ===========================================================================

/// Transport mechanism used to communicate with an MCP server.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MCPTransport {
    /// Standard I/O (stdin/stdout) transport -- the MCP server is launched as
    /// a child process.
    Stdio,
    /// Server-Sent Events over HTTP(S).
    Sse,
}

// ===========================================================================
// Server config
// ===========================================================================

/// Configuration for connecting to a single MCP tool server.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MCPServerConfig {
    /// Human-readable name for this server.
    pub name: String,

    /// Transport mechanism.
    pub transport: MCPTransport,

    /// For `Stdio` transport: the command to launch the server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,

    /// For `Stdio` transport: command-line arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,

    /// For `Stdio` transport: extra environment variables.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<HashMap<String, String>>,

    /// For `Sse` transport: the server URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// For `Sse` transport: optional bearer token or API key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Timeout in seconds for tool calls to this server.
    #[serde(default = "default_tool_timeout")]
    pub timeout_secs: u64,

    /// Maximum number of retries for transient failures.
    #[serde(default)]
    pub max_retries: u32,
}

fn default_tool_timeout() -> u64 {
    30
}

impl Default for MCPServerConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            transport: MCPTransport::Stdio,
            command: None,
            args: None,
            env: None,
            url: None,
            api_key: None,
            timeout_secs: default_tool_timeout(),
            max_retries: 2,
        }
    }
}

// ===========================================================================
// Tool definition
// ===========================================================================

/// An MCP tool definition as advertised by a connected server.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MCPTool {
    /// The tool name (must be unique within a server).
    pub name: String,

    /// Human-readable description shown to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema describing the tool's input parameters.
    pub input_schema: serde_json::Value,

    /// The name of the MCP server that provides this tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_name: Option<String>,
}

// ===========================================================================
// Tool result
// ===========================================================================

/// The result of invoking an MCP tool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MCPToolResult {
    /// The content returned by the tool.
    pub content: Vec<MCPToolResultContent>,

    /// Whether the tool invocation was an error.
    #[serde(default)]
    pub is_error: bool,
}

/// A single content block in an MCP tool result.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MCPToolResultContent {
    /// Plain text result.
    Text { text: String },

    /// An image result.
    Image {
        /// Base64-encoded image data.
        data: String,
        /// MIME type, e.g. `"image/png"`.
        mime_type: String,
    },

    /// An embedded resource.
    Resource {
        resource: MCPResource,
    },
}

/// An embedded resource in an MCP tool result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MCPResource {
    /// URI identifying the resource.
    pub uri: String,
    /// MIME type of the resource.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Optional text content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Optional base64-encoded binary content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

// ===========================================================================
// Server state
// ===========================================================================

/// Runtime state of a connected MCP server.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MCPServerState {
    /// The server has not been started yet.
    NotStarted,
    /// The server is currently being initialised (handshake in progress).
    Connecting,
    /// The server is connected and ready to serve tool calls.
    Ready,
    /// The server has encountered an error and is unavailable.
    Error,
    /// The server has been intentionally stopped.
    Stopped,
}

impl Default for MCPServerState {
    fn default() -> Self {
        Self::NotStarted
    }
}

/// Runtime information about a connected MCP server.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MCPServerStatus {
    /// Server configuration.
    pub config: MCPServerConfig,

    /// Current state.
    pub state: MCPServerState,

    /// Tools advertised by the server (populated after initialisation).
    #[serde(default)]
    pub tools: Vec<MCPTool>,

    /// Last error message, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}
