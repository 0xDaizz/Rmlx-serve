//! Chat-template rendering using Jinja2 (via `minijinja`).
//!
//! Most modern chat models ship a `chat_template` string inside their
//! `tokenizer_config.json`.  This module loads that template and renders
//! an array of messages into the prompt string the model expects.

use std::path::Path;

use minijinja::Environment;
use serde::Serialize;
use tracing::{debug, warn};

use crate::error::{Result, TokenizerError};

// ---------------------------------------------------------------------------
// Message type used for template rendering
// ---------------------------------------------------------------------------

/// A chat message suitable for Jinja2 rendering.
///
/// We intentionally keep this separate from any protocol-specific message type
/// so that callers can convert from OpenAI-style, Anthropic-style, or any
/// other format.
#[derive(Debug, Clone, Serialize)]
pub struct TemplateMessage {
    pub role: String,
    pub content: String,
}

// ---------------------------------------------------------------------------
// Default ChatML template
// ---------------------------------------------------------------------------

/// The classic ChatML format used as a fallback when no chat_template is
/// provided by the model.
const DEFAULT_CHATML_TEMPLATE: &str = "\
{% for message in messages %}\
<|im_start|>{{ message.role }}\n\
{{ message.content }}<|im_end|>\n\
{% endfor %}\
{% if add_generation_prompt %}\
<|im_start|>assistant\n\
{% endif %}";

// ---------------------------------------------------------------------------
// ChatTemplate
// ---------------------------------------------------------------------------

/// Wraps a Jinja2 chat template and provides rendering for message arrays.
pub struct ChatTemplate {
    /// We store the raw template string so we can clone / inspect it.
    template_source: String,
}

impl ChatTemplate {
    /// Create a new `ChatTemplate` from a raw Jinja2 template string.
    pub fn new(template: String) -> Self {
        Self {
            template_source: template,
        }
    }

    /// Return a reference to the raw template source.
    pub fn source(&self) -> &str {
        &self.template_source
    }

    /// Render a list of messages using the template.
    ///
    /// The template receives the following variables:
    /// - `messages` -- the message array
    /// - `add_generation_prompt` -- always `true` (we want the model to start
    ///   generating after the last message)
    /// - `bos_token`, `eos_token` -- set to empty strings unless overridden
    ///   by the caller via [`apply_with_tokens`].
    pub fn apply(&self, messages: &[TemplateMessage]) -> Result<String> {
        self.apply_with_tokens(messages, "", "")
    }

    /// Same as [`apply`] but allows injecting `bos_token` / `eos_token`
    /// strings that some templates reference.
    pub fn apply_with_tokens(
        &self,
        messages: &[TemplateMessage],
        bos_token: &str,
        eos_token: &str,
    ) -> Result<String> {
        let mut env = Environment::new();

        // Register a `raise_exception` function that many HF templates use.
        env.add_function("raise_exception", raise_exception);

        env.add_template("chat", &self.template_source)
            .map_err(|e| {
                TokenizerError::TemplateFailed(format!("failed to compile chat template: {e}"))
            })?;

        let tmpl = env.get_template("chat").map_err(|e| {
            TokenizerError::TemplateFailed(format!("failed to retrieve compiled template: {e}"))
        })?;

        let ctx = minijinja::context! {
            messages => messages,
            add_generation_prompt => true,
            bos_token => bos_token,
            eos_token => eos_token,
        };

        tmpl.render(ctx).map_err(|e| {
            TokenizerError::TemplateFailed(format!("template render error: {e}"))
        })
    }

    /// Try to load a `ChatTemplate` from `tokenizer_config.json` located in
    /// the given directory.  Returns `Ok(None)` if the file exists but does
    /// not contain a `chat_template` key.
    pub fn from_config(path: impl AsRef<Path>) -> Result<Option<Self>> {
        let dir = path.as_ref();
        let config_path = dir.join("tokenizer_config.json");

        if !config_path.exists() {
            debug!("no tokenizer_config.json at {}, using default ChatML", dir.display());
            return Ok(Some(Self::new(DEFAULT_CHATML_TEMPLATE.to_string())));
        }

        let raw = std::fs::read_to_string(&config_path).map_err(|e| {
            TokenizerError::LoadFailed {
                path: config_path.clone(),
                reason: e.to_string(),
            }
        })?;

        let parsed: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| TokenizerError::InvalidConfig(format!(
                "failed to parse {}: {e}",
                config_path.display()
            )))?;

        match parsed.get("chat_template") {
            Some(serde_json::Value::String(t)) => {
                debug!("loaded chat_template from {}", config_path.display());
                Ok(Some(Self::new(t.clone())))
            }
            Some(serde_json::Value::Array(arr)) => {
                // Some models provide an array of {name, template} objects.
                // We pick the one named "default", or failing that the first entry.
                let chosen = arr
                    .iter()
                    .find(|v| {
                        v.get("name")
                            .and_then(|n| n.as_str())
                            .map_or(false, |n| n == "default")
                    })
                    .or_else(|| arr.first());

                if let Some(obj) = chosen {
                    if let Some(t) = obj.get("template").and_then(|t| t.as_str()) {
                        debug!(
                            "loaded chat_template (array entry) from {}",
                            config_path.display()
                        );
                        return Ok(Some(Self::new(t.to_string())));
                    }
                }
                warn!(
                    "chat_template array in {} contained no usable entry, using ChatML default",
                    config_path.display()
                );
                Ok(Some(Self::new(DEFAULT_CHATML_TEMPLATE.to_string())))
            }
            _ => {
                debug!(
                    "no chat_template key in {}, returning None",
                    config_path.display()
                );
                Ok(None)
            }
        }
    }

    /// Return a `ChatTemplate` using the built-in ChatML default.
    pub fn default_chatml() -> Self {
        Self::new(DEFAULT_CHATML_TEMPLATE.to_string())
    }
}

// ---------------------------------------------------------------------------
// Template helpers
// ---------------------------------------------------------------------------

/// A `raise_exception` function that many HuggingFace Jinja templates use to
/// signal invalid input.  We surface it as a minijinja error.
fn raise_exception(msg: String) -> std::result::Result<String, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        format!("template raised exception: {msg}"),
    ))
}
