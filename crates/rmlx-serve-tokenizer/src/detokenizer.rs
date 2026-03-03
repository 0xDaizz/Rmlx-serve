//! Streaming detokenizers -- emit decoded text incrementally as tokens arrive.
//!
//! This module is a Rust port of the approach used in
//! [mlx-lm](https://github.com/ml-explore/mlx-examples) `tokenizer_utils.py`.
//!
//! Three implementations are provided:
//!
//! | Type    | Complexity per token | Notes                              |
//! |---------|---------------------|------------------------------------|
//! | Naive   | O(T)                | Decodes the full token buffer each step. Works for any tokenizer. |
//! | SPM     | O(1) amortised      | Exploits SentencePiece `▁` token boundaries. |
//! | BPE     | O(1) amortised      | Exploits BPE `Ġ` (leading-space) token boundaries. |
//!
//! Use [`create_detokenizer`] to automatically pick the best variant based on
//! the vocabulary of a loaded [`Tokenizer`](crate::Tokenizer).

use crate::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Trait for incremental (streaming) detokenisation.
///
/// The typical usage loop is:
///
/// ```ignore
/// for id in generated_ids {
///     detok.add_token(id);
///     let text = detok.last_segment();
///     if !text.is_empty() {
///         send_to_client(text);
///     }
/// }
/// let final_text = detok.finalize();
/// ```
pub trait StreamingDetokenizer: Send {
    /// Feed the next generated token into the detokenizer.
    fn add_token(&mut self, token_id: u32);

    /// Return the newly decoded text since the previous call to
    /// `last_segment` (or since creation / reset).
    fn last_segment(&self) -> &str;

    /// Flush any remaining buffered bytes and return the final decoded text.
    fn finalize(&mut self) -> String;

    /// Reset all internal state so the detokenizer can be reused.
    fn reset(&mut self);
}

// ===========================================================================
// NaiveStreamingDetokenizer
// ===========================================================================

/// The simplest streaming detokenizer: keeps the full list of token IDs and
/// re-decodes the entire sequence every time a new token arrives.
///
/// This is O(T^2) over the whole generation but works correctly for any
/// tokenizer without needing to understand its internal encoding scheme.
pub struct NaiveStreamingDetokenizer {
    tokenizer: tokenizers::Tokenizer,
    token_ids: Vec<u32>,
    /// The decoded text up to (but not including) the current step.
    prev_text: String,
    /// The delta produced by the most recent `add_token` call.
    current_segment: String,
}

impl NaiveStreamingDetokenizer {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        Self {
            tokenizer: tokenizer.inner().clone(),
            token_ids: Vec::new(),
            prev_text: String::new(),
            current_segment: String::new(),
        }
    }
}

impl StreamingDetokenizer for NaiveStreamingDetokenizer {
    fn add_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);

        // Decode the full buffer.
        let full = self
            .tokenizer
            .decode(&self.token_ids, true)
            .unwrap_or_default();

        // The new segment is whatever is beyond the previously emitted text.
        if full.len() > self.prev_text.len() {
            self.current_segment = full[self.prev_text.len()..].to_string();
            self.prev_text = full;
        } else {
            // Tokens that merge into the previous character (e.g. combining
            // marks) can cause the new decode to be the same length or even
            // shorter -- in that case emit nothing.
            self.current_segment.clear();
            self.prev_text = full;
        }
    }

    fn last_segment(&self) -> &str {
        &self.current_segment
    }

    fn finalize(&mut self) -> String {
        let full = self
            .tokenizer
            .decode(&self.token_ids, true)
            .unwrap_or_default();
        let remaining = if full.len() > self.prev_text.len() {
            full[self.prev_text.len()..].to_string()
        } else {
            String::new()
        };
        self.prev_text = full;
        self.current_segment.clear();
        remaining
    }

    fn reset(&mut self) {
        self.token_ids.clear();
        self.prev_text.clear();
        self.current_segment.clear();
    }
}

// ===========================================================================
// SPMStreamingDetokenizer
// ===========================================================================

/// A streaming detokenizer optimised for SentencePiece models.
///
/// SentencePiece represents word boundaries with the Unicode character `▁`
/// (U+2581). When we see a new token whose decoded form starts with `▁` (or a
/// space -- after the tokenizer internally replaces it), we know the previous
/// buffered text forms a complete word boundary and can be flushed.
///
/// Between word boundaries we accumulate tokens; only when we reach the next
/// boundary do we emit the previous chunk. This avoids the O(T^2) cost of the
/// naive approach.
pub struct SPMStreamingDetokenizer {
    tokenizer: tokenizers::Tokenizer,
    /// Token IDs in the current pending chunk (not yet emitted).
    pending_ids: Vec<u32>,
    /// Decoded text of the fully-flushed segment from the previous boundary.
    current_segment: String,
    /// Whether we have ever emitted a segment (used to handle the very first token).
    first_token: bool,
    /// Replacement character used by this tokenizer (typically `▁`).
    replacement: char,
}

impl SPMStreamingDetokenizer {
    pub fn new(tokenizer: &Tokenizer, replacement: char) -> Self {
        Self {
            tokenizer: tokenizer.inner().clone(),
            pending_ids: Vec::new(),
            current_segment: String::new(),
            first_token: true,
            replacement,
        }
    }

    /// Decode a slice of token IDs with `skip_special_tokens = true`.
    fn decode_ids(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, true).unwrap_or_default()
    }

    /// Check whether a token's decoded text starts with the replacement char
    /// or a space (which indicates a word boundary for SPM tokenizers).
    fn is_boundary_token(&self, token_id: u32) -> bool {
        let text = self.decode_ids(&[token_id]);
        text.starts_with(self.replacement) || text.starts_with(' ')
    }
}

impl StreamingDetokenizer for SPMStreamingDetokenizer {
    fn add_token(&mut self, token_id: u32) {
        // On word boundaries, flush the pending buffer into `current_segment`.
        if !self.first_token && self.is_boundary_token(token_id) {
            // The pending buffer now forms a complete word -- emit it.
            self.current_segment = self.decode_ids(&self.pending_ids);
            self.pending_ids.clear();
        } else {
            // Not a boundary -- whatever was in current_segment has already
            // been consumed by the caller, so clear it.
            self.current_segment.clear();
        }

        self.pending_ids.push(token_id);
        self.first_token = false;
    }

    fn last_segment(&self) -> &str {
        &self.current_segment
    }

    fn finalize(&mut self) -> String {
        let remaining = self.decode_ids(&self.pending_ids);
        self.pending_ids.clear();
        self.current_segment.clear();
        self.first_token = true;
        remaining
    }

    fn reset(&mut self) {
        self.pending_ids.clear();
        self.current_segment.clear();
        self.first_token = true;
    }
}

// ===========================================================================
// BPEStreamingDetokenizer
// ===========================================================================

/// A streaming detokenizer optimised for BPE models (GPT-2 / GPT-NeoX style).
///
/// These tokenizers prepend `Ġ` (U+0120) to tokens that follow a space in the
/// original text. Like the SPM detokenizer, we use this marker as a word
/// boundary signal.
pub struct BPEStreamingDetokenizer {
    tokenizer: tokenizers::Tokenizer,
    /// Token IDs in the current pending chunk.
    pending_ids: Vec<u32>,
    /// The segment that was flushed at the last boundary.
    current_segment: String,
    /// Whether we have received the very first token.
    first_token: bool,
}

impl BPEStreamingDetokenizer {
    /// The Ġ character that GPT-2 / BPE tokenizers use.
    const BPE_SPACE: char = '\u{0120}';

    pub fn new(tokenizer: &Tokenizer) -> Self {
        Self {
            tokenizer: tokenizer.inner().clone(),
            pending_ids: Vec::new(),
            current_segment: String::new(),
            first_token: true,
        }
    }

    fn decode_ids(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, true).unwrap_or_default()
    }

    /// Check whether the raw (pre-decode) token string starts with `Ġ`.
    fn is_boundary_token(&self, token_id: u32) -> bool {
        // We look at the raw vocabulary entry (before byte-level decoding)
        // because `Ġ` is converted to a space in the final decoded string.
        if let Some(token_str) = self.tokenizer.id_to_token(token_id) {
            token_str.starts_with(Self::BPE_SPACE) || token_str.starts_with(' ')
        } else {
            false
        }
    }
}

impl StreamingDetokenizer for BPEStreamingDetokenizer {
    fn add_token(&mut self, token_id: u32) {
        if !self.first_token && self.is_boundary_token(token_id) {
            self.current_segment = self.decode_ids(&self.pending_ids);
            self.pending_ids.clear();
        } else {
            self.current_segment.clear();
        }

        self.pending_ids.push(token_id);
        self.first_token = false;
    }

    fn last_segment(&self) -> &str {
        &self.current_segment
    }

    fn finalize(&mut self) -> String {
        let remaining = self.decode_ids(&self.pending_ids);
        self.pending_ids.clear();
        self.current_segment.clear();
        self.first_token = true;
        remaining
    }

    fn reset(&mut self) {
        self.pending_ids.clear();
        self.current_segment.clear();
        self.first_token = true;
    }
}

// ===========================================================================
// Auto-detection
// ===========================================================================

/// Describes which streaming detokenizer variant was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetokenizerType {
    /// SentencePiece-style (vocab contains `▁`).
    Spm,
    /// BPE-style (vocab contains `Ġ`).
    Bpe,
    /// Fallback / unknown tokenizer.
    Naive,
}

impl std::fmt::Display for DetokenizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spm => write!(f, "SPM"),
            Self::Bpe => write!(f, "BPE"),
            Self::Naive => write!(f, "Naive"),
        }
    }
}

/// The standard SentencePiece replacement character.
const SPM_UNDERSCORE: char = '\u{2581}'; // ▁

/// Inspect the tokenizer's vocabulary to decide which streaming detokenizer
/// implementation is most appropriate, then construct and return it.
pub fn create_detokenizer(tokenizer: &Tokenizer) -> Box<dyn StreamingDetokenizer> {
    let dt = detect_type(tokenizer);
    tracing::debug!("selected streaming detokenizer: {dt}");

    match dt {
        DetokenizerType::Spm => Box::new(SPMStreamingDetokenizer::new(tokenizer, SPM_UNDERSCORE)),
        DetokenizerType::Bpe => Box::new(BPEStreamingDetokenizer::new(tokenizer)),
        DetokenizerType::Naive => Box::new(NaiveStreamingDetokenizer::new(tokenizer)),
    }
}

/// Detect the tokenizer type by sampling the vocabulary for characteristic
/// marker characters.
pub fn detect_type(tokenizer: &Tokenizer) -> DetokenizerType {
    let vocab = tokenizer.inner().get_vocab(true);

    let mut has_spm = false;
    let mut has_bpe = false;

    for token_str in vocab.keys() {
        if !has_spm && token_str.contains(SPM_UNDERSCORE) {
            has_spm = true;
        }
        if !has_bpe && token_str.contains('\u{0120}') {
            has_bpe = true;
        }
        if has_spm || has_bpe {
            break;
        }
    }

    // SPM takes precedence because some models have both characters in their
    // vocabulary (e.g., Llama uses SPM but may have stray BPE-like tokens).
    if has_spm {
        DetokenizerType::Spm
    } else if has_bpe {
        DetokenizerType::Bpe
    } else {
        DetokenizerType::Naive
    }
}
