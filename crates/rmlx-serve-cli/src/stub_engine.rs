//! Stub engine implementation for development and testing.
//!
//! This module provides a [`StubEngine`] that implements the [`Engine`] trait
//! without loading a real model.  It returns synthetic outputs so that the CLI
//! binary can be exercised end-to-end before the real MLX-backed engine is
//! ready.
//!
//! **This module will be removed** once `rmlx_serve_engine` ships concrete
//! `SimpleEngine` / `BatchedEngine` implementations.

use async_trait::async_trait;
use rmlx_serve_engine::{Engine, EngineError, EngineHealth, EngineStats};
use rmlx_serve_types::config::EngineConfig;
use rmlx_serve_types::{
    CompletionOutput, FinishReason, Request, RequestMetrics, RequestOutput,
};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc;

/// A no-op engine that returns dummy completions.
pub struct StubEngine {
    config: EngineConfig,
    request_counter: AtomicU64,
}

impl StubEngine {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            config,
            request_counter: AtomicU64::new(0),
        }
    }
}

#[async_trait]
impl Engine for StubEngine {
    fn model_name(&self) -> &str {
        &self.config.model
    }

    async fn generate(&self, request: Request) -> Result<RequestOutput, EngineError> {
        let _id = self.request_counter.fetch_add(1, Ordering::Relaxed);

        let prompt_len = request.prompt_token_ids.len();
        let max_tokens = request.sampling_params.max_tokens;

        // Simulate a small delay proportional to token count.
        let prefill_ms = (prompt_len as u64).min(50);
        let decode_ms = (max_tokens as u64).min(200);
        tokio::time::sleep(tokio::time::Duration::from_millis(prefill_ms + decode_ms)).await;

        let arrival = request.arrival_time;
        let first_token_time = arrival + (prefill_ms as f64 / 1000.0);
        let finish_time = first_token_time + (decode_ms as f64 / 1000.0);

        // Generate dummy token ids (1, 2, 3, ...).
        let gen_token_ids: Vec<u32> = (1..=max_tokens as u32).collect();
        let gen_text = format!(
            "[stub output: {} tokens generated for prompt of {} tokens]",
            max_tokens, prompt_len
        );

        Ok(RequestOutput {
            request_id: request.id,
            outputs: vec![CompletionOutput {
                index: 0,
                text: gen_text,
                token_ids: gen_token_ids,
                finish_reason: Some(FinishReason::Length),
                logprobs: Vec::new(),
            }],
            finished: true,
            metrics: Some(RequestMetrics {
                arrival_time: arrival,
                first_token_time: Some(first_token_time),
                finish_time: Some(finish_time),
                prompt_tokens: prompt_len,
                completion_tokens: max_tokens,
            }),
        })
    }

    async fn generate_stream(
        &self,
        request: Request,
    ) -> Result<mpsc::UnboundedReceiver<RequestOutput>, EngineError> {
        let (tx, rx) = mpsc::unbounded_channel();

        let output = self.generate(request).await?;
        let _ = tx.send(output);

        Ok(rx)
    }

    async fn health(&self) -> EngineHealth {
        EngineHealth {
            is_ready: true,
            status: "ok (stub)".to_string(),
            model: self.config.model.clone(),
            active_requests: 0,
        }
    }

    fn get_stats(&self) -> EngineStats {
        EngineStats {
            total_requests: self.request_counter.load(Ordering::Relaxed),
            ..EngineStats::default()
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        // Crude stub: one token per whitespace-delimited word, starting at id 1.
        let ids: Vec<u32> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| (i + 1) as u32)
            .collect();
        Ok(ids)
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String, EngineError> {
        // Crude stub: render each token id as "<id>".
        let text = token_ids
            .iter()
            .map(|id| format!("<{}>", id))
            .collect::<Vec<_>>()
            .join(" ");
        Ok(text)
    }
}
