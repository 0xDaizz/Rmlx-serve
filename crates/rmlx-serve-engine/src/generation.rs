//! High-level generation helpers for CLI and testing use.
//!
//! The [`generate_text`] function provides a one-shot, self-contained
//! entry point that loads a model, tokenizes a prompt, generates a
//! response, and returns it as a [`GenerationResponse`]. This is the
//! Rust equivalent of `mlx-lm generate.py`'s top-level `generate()`.

use std::time::Instant;

use tracing::info;

use rmlx_serve_tokenizer::Tokenizer;
use rmlx_serve_types::{EngineConfig, FinishReason, Request, SamplingParams};

use crate::simple::SimpleEngine;
use crate::{Engine, EngineError};

// ---------------------------------------------------------------------------
// GenerationResponse
// ---------------------------------------------------------------------------

/// Result of a one-shot text generation.
///
/// Contains the generated text along with throughput metrics and
/// resource usage information. Intended for CLI output and benchmarking.
#[derive(Debug, Clone)]
pub struct GenerationResponse {
    /// The generated text (decoded from the output tokens).
    pub text: String,

    /// The generated token IDs.
    pub tokens: Vec<u32>,

    /// Number of tokens in the input prompt.
    pub prompt_tokens: usize,

    /// Number of tokens generated.
    pub generation_tokens: usize,

    /// Prompt processing throughput (tokens per second).
    pub prompt_tps: f64,

    /// Generation throughput (tokens per second).
    pub generation_tps: f64,

    /// Peak GPU memory usage in megabytes (approximate).
    pub peak_memory_mb: f64,

    /// Why generation stopped.
    pub finish_reason: FinishReason,
}

// ---------------------------------------------------------------------------
// generate_text
// ---------------------------------------------------------------------------

/// Simple one-shot text generation function.
///
/// Loads the model and tokenizer, encodes the prompt, runs generation
/// via [`SimpleEngine`], and returns a [`GenerationResponse`] with
/// timing metrics.
///
/// This is a convenience wrapper for CLI and testing use. For server
/// workloads, prefer constructing an engine directly and reusing it
/// across requests.
///
/// # Arguments
/// * `model_path` - Path to the model directory (HuggingFace layout).
/// * `prompt` - The text prompt to complete.
/// * `params` - Sampling parameters controlling generation behavior.
///
/// # Example
/// ```ignore
/// use rmlx_serve_types::SamplingParams;
/// use rmlx_serve_engine::generate_text;
///
/// let response = generate_text(
///     "/path/to/llama-3-8b",
///     "Once upon a time",
///     SamplingParams { max_tokens: 100, ..Default::default() },
/// ).await?;
///
/// println!("{}", response.text);
/// println!("Prompt: {:.1} tok/s", response.prompt_tps);
/// println!("Generation: {:.1} tok/s", response.generation_tps);
/// ```
pub async fn generate_text(
    model_path: &str,
    prompt: &str,
    params: SamplingParams,
) -> Result<GenerationResponse, EngineError> {
    let overall_start = Instant::now();

    info!(
        model = model_path,
        prompt_len = prompt.len(),
        "generate_text starting"
    );

    // Build engine config.
    let config = EngineConfig {
        model: model_path.to_string(),
        ..Default::default()
    };

    // Create the engine (loads model, tokenizer, GPU kernels).
    let engine = SimpleEngine::new(config).await?;

    // Encode the prompt.
    let tokenizer = Tokenizer::from_pretrained(model_path)?;
    let prompt_token_ids = tokenizer.encode(prompt, true)?;
    let prompt_tokens = prompt_token_ids.len();

    info!(prompt_tokens, "prompt encoded");

    // Build the request.
    let arrival_time = overall_start.elapsed().as_secs_f64();
    let request = Request::new(prompt_token_ids, params, arrival_time);

    // Run generation.
    let gen_start = Instant::now();
    let output = engine.generate(request).await?;
    let gen_elapsed = gen_start.elapsed();

    // Extract results.
    let completion = output
        .outputs
        .first()
        .ok_or_else(|| EngineError::Internal("no completion output".into()))?;

    let generation_tokens = completion.token_ids.len();
    let finish_reason = completion.finish_reason.unwrap_or(FinishReason::Length);

    // Compute throughput from metrics if available, otherwise estimate.
    let (prompt_tps, generation_tps) = if let Some(metrics) = &output.metrics {
        let p_tps = if let Some(ttft) = metrics.ttft() {
            if ttft > 0.0 {
                prompt_tokens as f64 / ttft
            } else {
                0.0
            }
        } else {
            0.0
        };

        let g_tps = metrics.tokens_per_second().unwrap_or(0.0);
        (p_tps, g_tps)
    } else {
        let total_secs = gen_elapsed.as_secs_f64();
        let tps = if total_secs > 0.0 {
            generation_tokens as f64 / total_secs
        } else {
            0.0
        };
        (0.0, tps)
    };

    info!(
        generation_tokens,
        prompt_tps = format!("{:.1}", prompt_tps),
        generation_tps = format!("{:.1}", generation_tps),
        total_ms = overall_start.elapsed().as_millis(),
        "generate_text complete"
    );

    Ok(GenerationResponse {
        text: completion.text.clone(),
        tokens: completion.token_ids.clone(),
        prompt_tokens,
        generation_tokens,
        prompt_tps,
        generation_tps,
        peak_memory_mb: 0.0, // TODO: query Metal device for peak memory
        finish_reason,
    })
}
