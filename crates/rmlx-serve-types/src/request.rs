//! Core request and response types for the inference engine.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// RequestId
// ---------------------------------------------------------------------------

/// A thin wrapper around [`Uuid`] used to uniquely identify every inference
/// request flowing through the system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(pub Uuid);

impl RequestId {
    /// Generate a new random (v4) request id.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Return the inner [`Uuid`].
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for RequestId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

// ---------------------------------------------------------------------------
// RequestStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of a request as it moves through the scheduler and engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    /// Waiting in the queue, not yet scheduled.
    Pending,
    /// Actively being processed (prefill or decode).
    Running,
    /// Generation complete.
    Finished,
    /// Preempted by the scheduler (e.g. to free KV-cache for higher-priority
    /// requests). The request may be rescheduled later.
    Preempted,
    /// Finished because a stop token or stop string was matched.
    FinishedStopped,
    /// Finished because the maximum token limit was reached.
    FinishedLengthCapped,
    /// Finished because the client disconnected or an unrecoverable error
    /// occurred.
    FinishedAborted,
}

// ---------------------------------------------------------------------------
// FinishReason
// ---------------------------------------------------------------------------

/// Why a generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model emitted an EOS / stop token or matched a stop string.
    Stop,
    /// `max_tokens` was reached.
    Length,
    /// The model produced a tool-call and is waiting for the result.
    ToolCall,
    /// An error occurred during generation.
    Error,
}

// ---------------------------------------------------------------------------
// SamplingParams
// ---------------------------------------------------------------------------

/// Parameters controlling the sampling / decoding strategy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Temperature for softmax scaling. 0.0 means greedy.
    pub temperature: f32,

    /// Nucleus (top-p) sampling threshold.
    pub top_p: f32,

    /// Top-k filtering. 0 disables it.
    pub top_k: u32,

    /// Min-p filtering threshold. 0.0 disables it.
    pub min_p: f32,

    /// Maximum number of tokens to generate.
    pub max_tokens: usize,

    /// Stop strings -- generation halts when any of these appear in the output.
    #[serde(default)]
    pub stop: Vec<String>,

    /// Stop token ids -- generation halts when any of these token ids are sampled.
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,

    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,

    /// Frequency penalty applied proportionally to token count in the output.
    pub frequency_penalty: f32,

    /// Presence penalty applied once per unique token in the output.
    pub presence_penalty: f32,

    /// Per-token logit bias. Keys are token ids, values are additive bias.
    #[serde(default)]
    pub logit_bias: HashMap<u32, f32>,

    /// Optional seed for reproducible sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Number of independent completions to generate.
    pub n: usize,

    /// If `Some(k)`, return the top-k log-probabilities for each generated token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<usize>,

    /// XTC (eXclusive Top Choice) probability -- probability of excluding
    /// the top token to encourage diversity. 0.0 disables.
    pub xtc_probability: f32,

    /// XTC threshold -- only tokens above this probability are candidates for
    /// exclusion. 0.0 disables.
    pub xtc_threshold: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            max_tokens: 256,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: HashMap::new(),
            seed: None,
            n: 1,
            logprobs: None,
            xtc_probability: 0.0,
            xtc_threshold: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// An inference request as tracked by the scheduler and engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Request {
    /// Unique request identifier.
    pub id: RequestId,

    /// Token ids of the prompt (already encoded by the tokenizer).
    pub prompt_token_ids: Vec<u32>,

    /// Sampling parameters for this request.
    pub sampling_params: SamplingParams,

    /// Monotonic timestamp (seconds) when the request arrived.
    pub arrival_time: f64,

    /// Current lifecycle status.
    pub status: RequestStatus,

    /// Whether the client expects a streaming (SSE) response.
    pub stream: bool,

    /// Whether the prompt contains non-text modalities (images, audio, video).
    pub is_multimodal: bool,
}

impl Request {
    /// Create a new pending request with the given prompt tokens and sampling
    /// parameters. `arrival_time` should be the current wall-clock time in
    /// fractional seconds.
    pub fn new(
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
        arrival_time: f64,
    ) -> Self {
        Self {
            id: RequestId::new(),
            prompt_token_ids,
            sampling_params,
            arrival_time,
            status: RequestStatus::Pending,
            stream: false,
            is_multimodal: false,
        }
    }
}

// ---------------------------------------------------------------------------
// RequestOutput
// ---------------------------------------------------------------------------

/// The output produced for a single request, potentially containing multiple
/// completions (when `n > 1`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestOutput {
    /// The id of the originating request.
    pub request_id: RequestId,

    /// One [`CompletionOutput`] per completion sequence.
    pub outputs: Vec<CompletionOutput>,

    /// `true` once all sequences have finished.
    pub finished: bool,

    /// Timing and token-count metrics (populated when finished).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<RequestMetrics>,
}

// ---------------------------------------------------------------------------
// CompletionOutput
// ---------------------------------------------------------------------------

/// Output for a single completion sequence within a request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionOutput {
    /// Zero-based index among the `n` completions.
    pub index: usize,

    /// The generated text (decoded from `token_ids`).
    pub text: String,

    /// The generated token ids.
    pub token_ids: Vec<u32>,

    /// Why this sequence stopped (absent while still generating).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Per-token log-probability information (empty unless `logprobs` was set).
    #[serde(default)]
    pub logprobs: Vec<TokenLogprob>,
}

// ---------------------------------------------------------------------------
// RequestMetrics
// ---------------------------------------------------------------------------

/// Timing and usage metrics for a completed request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestMetrics {
    /// Wall-clock arrival time in fractional seconds.
    pub arrival_time: f64,

    /// Wall-clock time when the first token was emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_token_time: Option<f64>,

    /// Wall-clock time when generation finished.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_time: Option<f64>,

    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,

    /// Number of tokens generated across all sequences.
    pub completion_tokens: usize,
}

impl RequestMetrics {
    /// Time-to-first-token in seconds, if available.
    pub fn ttft(&self) -> Option<f64> {
        self.first_token_time
            .map(|ft| ft - self.arrival_time)
    }

    /// Total generation latency in seconds, if available.
    pub fn total_latency(&self) -> Option<f64> {
        self.finish_time
            .map(|ft| ft - self.arrival_time)
    }

    /// Tokens per second (completion tokens / generation time after first token).
    pub fn tokens_per_second(&self) -> Option<f64> {
        match (self.first_token_time, self.finish_time) {
            (Some(first), Some(finish)) if finish > first && self.completion_tokens > 0 => {
                // Subtract 1 because the first token time is when the first token
                // was generated, so remaining tokens = completion_tokens - 1.
                let remaining = self.completion_tokens.saturating_sub(1) as f64;
                let decode_time = finish - first;
                if decode_time > 0.0 {
                    Some(remaining / decode_time)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// TokenLogprob
// ---------------------------------------------------------------------------

/// Log-probability information for a single generated token.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenLogprob {
    /// The sampled token id.
    pub token_id: u32,

    /// Log-probability of the sampled token.
    pub logprob: f32,

    /// Top-k (token_id, logprob) pairs at this position.
    #[serde(default)]
    pub top_logprobs: Vec<(u32, f32)>,
}
