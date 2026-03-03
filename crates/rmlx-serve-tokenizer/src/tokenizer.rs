//! Core tokenizer wrapper around the HuggingFace `tokenizers` crate.

use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;
use tracing::{debug, warn};

use crate::error::{Result, TokenizerError};

/// A thin wrapper around [`tokenizers::Tokenizer`] that also tracks special token
/// IDs (BOS, EOS) and allows registering additional stop tokens for generation.
pub struct Tokenizer {
    /// The underlying HuggingFace tokenizer.
    inner: tokenizers::Tokenizer,
    /// All token IDs that should be considered end-of-sequence.
    eos_token_ids: Vec<u32>,
    /// Beginning-of-sequence token ID (if the model defines one).
    bos_token_id: Option<u32>,
    /// Vocabulary size as reported by the tokenizer.
    vocab_size: usize,
    /// User-registered additional stop tokens beyond the model's native EOS set.
    additional_stop_tokens: HashSet<u32>,
}

// ---------------------------------------------------------------------------
// Helpers for parsing `tokenizer_config.json`
// ---------------------------------------------------------------------------

/// In HuggingFace configs the special token field can be either a plain string
/// or an object with a `"content"` key (the `AddedToken` representation).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum TokenOrString {
    Plain(String),
    Object { content: String },
}

impl TokenOrString {
    fn as_str(&self) -> &str {
        match self {
            Self::Plain(s) => s.as_str(),
            Self::Object { content } => content.as_str(),
        }
    }
}

/// Subset of `tokenizer_config.json` that we care about.
#[derive(Debug, Default, Deserialize)]
struct TokenizerConfig {
    eos_token: Option<EosToken>,
    bos_token: Option<TokenOrString>,
    #[allow(dead_code)]
    chat_template: Option<serde_json::Value>,
}

/// EOS can be a single token specification or a list of them.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum EosToken {
    Single(TokenOrString),
    Multiple(Vec<TokenOrString>),
}

// ---------------------------------------------------------------------------
// Tokenizer implementation
// ---------------------------------------------------------------------------

impl Tokenizer {
    /// Load a tokenizer from a directory that contains `tokenizer.json` and
    /// (optionally) `tokenizer_config.json`.
    ///
    /// The config file is used to resolve `eos_token` and `bos_token` to their
    /// numeric IDs via the vocabulary.
    pub fn from_pretrained(path: impl AsRef<Path>) -> Result<Self> {
        let dir = path.as_ref();
        let tokenizer_json = dir.join("tokenizer.json");

        let inner = tokenizers::Tokenizer::from_file(&tokenizer_json).map_err(|e| {
            TokenizerError::LoadFailed {
                path: tokenizer_json.clone(),
                reason: e.to_string(),
            }
        })?;

        let vocab_size = inner.get_vocab_size(true);

        // Try to load the config; if absent we just proceed with defaults.
        let config_path = dir.join("tokenizer_config.json");
        let config: TokenizerConfig = if config_path.exists() {
            let raw = std::fs::read_to_string(&config_path).map_err(|e| {
                TokenizerError::LoadFailed {
                    path: config_path.clone(),
                    reason: e.to_string(),
                }
            })?;
            serde_json::from_str(&raw).map_err(|e| TokenizerError::InvalidConfig(format!(
                "failed to parse {}: {e}",
                config_path.display()
            )))?
        } else {
            debug!("no tokenizer_config.json found at {}, using defaults", dir.display());
            TokenizerConfig::default()
        };

        // Resolve EOS token(s) to IDs.
        let eos_token_ids = Self::resolve_eos_tokens(&inner, &config);
        let bos_token_id = Self::resolve_token_id(&inner, config.bos_token.as_ref());

        debug!(
            vocab_size,
            eos_ids = ?eos_token_ids,
            bos_id = ?bos_token_id,
            "tokenizer loaded from {}",
            dir.display()
        );

        Ok(Self {
            inner,
            eos_token_ids,
            bos_token_id,
            vocab_size,
            additional_stop_tokens: HashSet::new(),
        })
    }

    // -- public API ----------------------------------------------------------

    /// Encode a string into a sequence of token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::EncodeFailed(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a sequence of token IDs back into text.
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| TokenizerError::DecodeFailed(e.to_string()))
    }

    /// The total number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The primary EOS token ID, if one was found.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_ids.first().copied()
    }

    /// All EOS token IDs (some models define more than one).
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// The BOS token ID, if one was found.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Returns `true` if `token_id` is any of the EOS token IDs **or** one of
    /// the additional stop tokens registered via [`add_stop_token`].
    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id) || self.additional_stop_tokens.contains(&token_id)
    }

    /// Register an extra token ID that should be treated as a stop/EOS token
    /// during generation.
    pub fn add_stop_token(&mut self, token_id: u32) {
        self.additional_stop_tokens.insert(token_id);
    }

    /// Access the underlying HuggingFace tokenizer, e.g. for streaming
    /// detokenization that needs vocabulary inspection.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }

    // -- helpers -------------------------------------------------------------

    /// Resolve EOS tokens from the config into a list of IDs.
    fn resolve_eos_tokens(inner: &tokenizers::Tokenizer, config: &TokenizerConfig) -> Vec<u32> {
        let strings: Vec<&str> = match &config.eos_token {
            Some(EosToken::Single(tok)) => vec![tok.as_str()],
            Some(EosToken::Multiple(toks)) => toks.iter().map(|t| t.as_str()).collect(),
            None => return Vec::new(),
        };

        let mut ids = Vec::with_capacity(strings.len());
        for s in strings {
            if let Some(id) = inner.token_to_id(s) {
                ids.push(id);
            } else {
                warn!("eos_token {:?} not found in vocabulary, skipping", s);
            }
        }
        ids
    }

    /// Resolve a single optional special-token string into its token ID.
    fn resolve_token_id(
        inner: &tokenizers::Tokenizer,
        tok: Option<&TokenOrString>,
    ) -> Option<u32> {
        let tok = tok?;
        let s = tok.as_str();
        let id = inner.token_to_id(s);
        if id.is_none() {
            warn!("special token {:?} not found in vocabulary", s);
        }
        id
    }
}
