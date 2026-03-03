//! SimpleEngine: single-request, non-batched inference engine.
//!
//! This engine processes one request at a time, serializing all GPU work
//! behind a `tokio::sync::Mutex`. It is the simplest engine implementation
//! and is well-suited for CLI use, testing, and low-concurrency scenarios.
//!
//! Ported from the core generation loop in mlx-lm `generate.py`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use rmlx_core::{ops, DType, KernelRegistry};
use rmlx_metal::metal;
use rmlx_metal::GpuDevice;
use rmlx_serve_models::{load_model, LlmModel};
use rmlx_serve_sampling::{make_logits_processors, make_sampler, top_logprobs};
use rmlx_serve_tokenizer::{create_detokenizer, Tokenizer};
use rmlx_serve_types::{
    CompletionOutput, EngineConfig, FinishReason, Request, RequestMetrics,
    RequestOutput, TokenLogprob,
};

use crate::{Engine, EngineError, EngineHealth, EngineStats};

// ---------------------------------------------------------------------------
// SimpleEngine
// ---------------------------------------------------------------------------

/// A single-request inference engine with no batching.
///
/// Metal GPU operations are serialized behind `tokio::sync::Mutex` because
/// Metal command buffers must be submitted serially on Apple Silicon. The
/// engine holds the model, tokenizer, and GPU resources for the lifetime of
/// the process.
pub struct SimpleEngine {
    /// The loaded LLM, behind a mutex for serial GPU access.
    model: Arc<tokio::sync::Mutex<Box<dyn LlmModel>>>,

    /// The tokenizer for encoding prompts and decoding output.
    tokenizer: Arc<Tokenizer>,

    /// Kernel registry for dispatching Metal compute pipelines.
    registry: Arc<KernelRegistry>,

    /// Metal command queue for GPU command submission.
    /// Retained for resource ownership; generation tasks create their own queues.
    #[allow(dead_code)]
    queue: metal::CommandQueue,

    /// GPU device handle (for cache allocation).
    /// Retained for resource ownership; generation tasks acquire their own handles.
    #[allow(dead_code)]
    device: GpuDevice,

    /// Name or path of the loaded model.
    model_name: String,

    /// Aggregate statistics, behind a mutex for interior mutability.
    stats: Arc<tokio::sync::Mutex<EngineStats>>,

    /// When the engine was created, for uptime calculation.
    start_time: Instant,

    /// Number of tokens to batch before emitting a streaming response.
    stream_interval: usize,

    /// Whether the engine is running (for graceful shutdown).
    running: Arc<AtomicBool>,
}

impl SimpleEngine {
    /// Create a new `SimpleEngine` from an [`EngineConfig`].
    ///
    /// This loads the model and tokenizer from disk, initializes the GPU
    /// device and kernel registry, and registers all built-in Metal kernels.
    pub async fn new(config: EngineConfig) -> Result<Self, EngineError> {
        let model_path = config.model.clone();

        info!(model = %model_path, "loading model for SimpleEngine");

        // 1. Load model via the model registry.
        let (model, model_config) = load_model(&model_path)?;

        info!(
            model_type = model_config.model_type.as_str(),
            num_layers = model.num_layers(),
            vocab_size = model.vocab_size(),
            "model loaded"
        );

        // 2. Load tokenizer from the model directory (or separate path).
        let tokenizer_path = config.tokenizer.as_deref().unwrap_or(&model_path);
        let tokenizer = Tokenizer::from_pretrained(tokenizer_path)?;

        info!(
            vocab_size = tokenizer.vocab_size(),
            "tokenizer loaded"
        );

        // 3. Create GPU device and kernel registry.
        let device = GpuDevice::system_default().map_err(|e| {
            EngineError::Internal(format!("failed to acquire Metal device: {e}"))
        })?;
        let queue = device.new_command_queue();
        let registry = KernelRegistry::new(device);

        // 4. Register all built-in GPU kernels.
        ops::register_all(&registry)?;

        info!(
            device = registry.device().name(),
            aot = registry.has_aot(),
            "GPU kernels registered"
        );

        // Re-acquire device (the first was moved into KernelRegistry::new).
        let device = GpuDevice::system_default().map_err(|e| {
            EngineError::Internal(format!("failed to acquire Metal device: {e}"))
        })?;

        let stream_interval = config.scheduler.stream_interval.max(1);

        Ok(Self {
            model: Arc::new(tokio::sync::Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            registry: Arc::new(registry),
            queue,
            device,
            model_name: model_path,
            stats: Arc::new(tokio::sync::Mutex::new(EngineStats::default())),
            start_time: Instant::now(),
            stream_interval,
            running: Arc::new(AtomicBool::new(true)),
        })
    }

    /// Access the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Start the engine. For SimpleEngine this marks the engine as ready
    /// to serve requests. No background threads are needed.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        info!("SimpleEngine started");
    }

    /// Gracefully stop the engine. Marks the engine as not-ready so new
    /// requests are rejected. In-flight requests will complete naturally
    /// since they hold the model lock.
    pub fn stop(&self) {
        info!("SimpleEngine stopping");
        self.running.store(false, Ordering::SeqCst);
        info!("SimpleEngine stopped");
    }

    /// Whether the engine is currently running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Extract logits from a model output Array as an f32 Vec on CPU.
    ///
    /// Handles Float32, Float16, and Bfloat16 model outputs.
    fn extract_logits(logits_array: &rmlx_core::Array) -> Result<Vec<f32>, EngineError> {
        let dtype = logits_array.dtype();
        match dtype {
            DType::Float32 => Ok(logits_array.to_vec_checked::<f32>()),
            DType::Float16 => {
                let bytes = logits_array.to_bytes();
                let numel = logits_array.numel();
                let mut result = Vec::with_capacity(numel);
                for i in 0..numel {
                    let offset = i * 2;
                    if offset + 2 <= bytes.len() {
                        let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                        result.push(f16_to_f32(bits));
                    }
                }
                Ok(result)
            }
            DType::Bfloat16 => {
                let bytes = logits_array.to_bytes();
                let numel = logits_array.numel();
                let mut result = Vec::with_capacity(numel);
                for i in 0..numel {
                    let offset = i * 2;
                    if offset + 2 <= bytes.len() {
                        let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                        result.push(bf16_to_f32(bits));
                    }
                }
                Ok(result)
            }
            other => Err(EngineError::Internal(format!(
                "unsupported logits dtype: {:?}",
                other
            ))),
        }
    }
}

#[async_trait]
impl Engine for SimpleEngine {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn generate(&self, request: Request) -> Result<RequestOutput, EngineError> {
        let request_id = request.id;
        let prompt_tokens = request.prompt_token_ids.clone();
        let sampling_params = request.sampling_params.clone();
        let max_tokens = sampling_params.max_tokens;
        let logprobs_k = sampling_params.logprobs;

        debug!(
            request_id = %request_id,
            prompt_len = prompt_tokens.len(),
            max_tokens,
            "SimpleEngine::generate starting"
        );

        let arrival_time = request.arrival_time;

        // Clone Arcs for the blocking task.
        let model = Arc::clone(&self.model);
        let registry = Arc::clone(&self.registry);
        let tokenizer = Arc::clone(&self.tokenizer);

        // We need a device handle inside spawn_blocking.
        let device = GpuDevice::system_default().map_err(|e| {
            EngineError::Internal(format!("failed to acquire Metal device: {e}"))
        })?;

        // Run the entire prefill + decode loop in spawn_blocking to avoid
        // blocking the tokio runtime with CPU/GPU-intensive forward passes.
        let generation_result = tokio::task::spawn_blocking(move || {
            let start = Instant::now();

            // Build sampler and logits processors from sampling params.
            let sampler = make_sampler(&sampling_params);
            let logits_processors = make_logits_processors(&sampling_params);

            // Collect stop token ids: model EOS + user-specified.
            let mut stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
            stop_token_ids.extend_from_slice(&sampling_params.stop_token_ids);

            let queue = device.new_command_queue();

            // Lock the model for serial GPU access (blocking_lock for non-async context).
            let model_guard = model.blocking_lock();

            // Create KV cache for this request.
            let mut cache = model_guard.make_cache(device.raw());

            // ── Prefill ──
            let prefill_start = Instant::now();
            let logits_array = model_guard
                .forward(&prompt_tokens, Some(&mut cache), &registry, &queue)
                .map_err(|e| EngineError::Model(format!("prefill failed: {e}")))?;

            let prefill_elapsed = prefill_start.elapsed();
            let prompt_tps = if prefill_elapsed.as_secs_f64() > 0.0 {
                prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64()
            } else {
                0.0
            };

            debug!(
                prompt_tokens = prompt_tokens.len(),
                prefill_ms = prefill_elapsed.as_millis(),
                prompt_tps = format!("{:.1}", prompt_tps),
                "prefill complete"
            );

            // Extract logits and sample first token.
            let mut logits_f32 = Self::extract_logits(&logits_array)?;
            let mut context_tokens: Vec<u32> = prompt_tokens.clone();

            for proc in &logits_processors {
                proc.process(&mut logits_f32, &context_tokens);
            }

            let first_token = sampler(&logits_f32);
            let first_token_time = start.elapsed().as_secs_f64() + arrival_time;

            // Build logprob info for the first token if requested.
            let first_logprob = logprobs_k.map(|k| {
                let top = top_logprobs(&logits_f32, k);
                TokenLogprob {
                    token_id: first_token,
                    logprob: rmlx_serve_sampling::log_softmax(&logits_f32)
                        .get(first_token as usize)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY),
                    top_logprobs: top,
                }
            });

            let mut generated_tokens: Vec<u32> = vec![first_token];
            let mut token_logprobs: Vec<TokenLogprob> = Vec::new();
            if let Some(lp) = first_logprob {
                token_logprobs.push(lp);
            }
            context_tokens.push(first_token);

            // Check if first token is a stop token.
            let mut finish_reason: Option<FinishReason> = None;
            if stop_token_ids.contains(&first_token) {
                finish_reason = Some(FinishReason::Stop);
            }
            if generated_tokens.len() >= max_tokens {
                finish_reason = Some(FinishReason::Length);
            }

            // ── Decode loop ──
            let decode_start = Instant::now();
            let mut steps_since_memory_check: usize = 0;

            while finish_reason.is_none() {
                let last_token = *generated_tokens.last().unwrap();

                // Memory pressure monitoring every 64 decode steps.
                steps_since_memory_check += 1;
                if steps_since_memory_check >= 64 {
                    steps_since_memory_check = 0;
                    check_and_clear_memory_pressure(device.raw());
                }

                let logits_array = model_guard
                    .forward(&[last_token], Some(&mut cache), &registry, &queue)
                    .map_err(|e| EngineError::Model(format!("decode failed: {e}")))?;

                let mut logits_f32 = Self::extract_logits(&logits_array)?;

                for proc in &logits_processors {
                    proc.process(&mut logits_f32, &context_tokens);
                }

                let next_token = sampler(&logits_f32);

                // Build logprob info if requested.
                if let Some(k) = logprobs_k {
                    let top = top_logprobs(&logits_f32, k);
                    token_logprobs.push(TokenLogprob {
                        token_id: next_token,
                        logprob: rmlx_serve_sampling::log_softmax(&logits_f32)
                            .get(next_token as usize)
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY),
                        top_logprobs: top,
                    });
                }

                generated_tokens.push(next_token);
                context_tokens.push(next_token);

                // Check stop conditions.
                if stop_token_ids.contains(&next_token) {
                    finish_reason = Some(FinishReason::Stop);
                } else if generated_tokens.len() >= max_tokens {
                    finish_reason = Some(FinishReason::Length);
                }
            }

            let decode_elapsed = decode_start.elapsed();
            let generation_tokens = generated_tokens.len();
            let decode_tokens = generation_tokens.saturating_sub(1);
            let generation_tps = if decode_elapsed.as_secs_f64() > 0.0 && decode_tokens > 0 {
                decode_tokens as f64 / decode_elapsed.as_secs_f64()
            } else {
                0.0
            };

            // Drop the model lock.
            drop(model_guard);

            // Decode generated tokens to text.
            let generated_text = tokenizer
                .decode(&generated_tokens, true)
                .unwrap_or_default();

            let finish_time = start.elapsed().as_secs_f64() + arrival_time;

            debug!(
                request_id = %request_id,
                generation_tokens,
                generation_tps = format!("{:.1}", generation_tps),
                total_ms = start.elapsed().as_millis(),
                "generation complete"
            );

            Ok::<_, EngineError>((
                request_id,
                generated_text,
                generated_tokens,
                finish_reason,
                token_logprobs,
                arrival_time,
                first_token_time,
                finish_time,
                prompt_tokens,
                generation_tokens,
                generation_tps,
            ))
        })
        .await
        .map_err(|e| EngineError::Internal(format!("spawn_blocking join error: {e}")))?;

        let (
            request_id,
            generated_text,
            generated_tokens,
            finish_reason,
            token_logprobs,
            arrival_time,
            first_token_time,
            finish_time,
            prompt_tokens,
            generation_tokens,
            generation_tps,
        ) = generation_result?;

        // Update stats.
        {
            let mut stats = self.stats.lock().await;
            stats.total_requests += 1;
            stats.total_prompt_tokens += prompt_tokens.len() as u64;
            stats.total_completion_tokens += generation_tokens as u64;
            stats.uptime_secs = self.start_time.elapsed().as_secs_f64();

            let ttft_ms = (first_token_time - arrival_time) * 1000.0;
            let n = stats.total_requests as f64;
            stats.avg_ttft_ms = stats.avg_ttft_ms * ((n - 1.0) / n) + ttft_ms / n;
            if generation_tps > 0.0 {
                stats.avg_tps = stats.avg_tps * ((n - 1.0) / n) + generation_tps / n;
            }
        }

        Ok(RequestOutput {
            request_id,
            outputs: vec![CompletionOutput {
                index: 0,
                text: generated_text,
                token_ids: generated_tokens,
                finish_reason,
                logprobs: token_logprobs,
            }],
            finished: true,
            metrics: Some(RequestMetrics {
                arrival_time,
                first_token_time: Some(first_token_time),
                finish_time: Some(finish_time),
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: generation_tokens,
            }),
        })
    }

    async fn generate_stream(
        &self,
        request: Request,
    ) -> Result<mpsc::UnboundedReceiver<RequestOutput>, EngineError> {
        let (tx, rx) = mpsc::unbounded_channel::<RequestOutput>();

        let request_id = request.id;
        let prompt_tokens = request.prompt_token_ids.clone();
        let sampling_params = request.sampling_params.clone();
        let max_tokens = sampling_params.max_tokens;
        let logprobs_k = sampling_params.logprobs;
        let arrival_time = request.arrival_time;
        let stream_interval = self.stream_interval;

        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let registry = Arc::clone(&self.registry);
        let stats = Arc::clone(&self.stats);
        let engine_start_time = self.start_time;

        // We need a second device handle for cache allocation.
        let device = GpuDevice::system_default().map_err(|e| {
            EngineError::Internal(format!("failed to acquire Metal device: {e}"))
        })?;

        // Spawn the generation loop in a blocking task to avoid
        // blocking the tokio runtime with CPU/GPU-intensive forward passes.
        tokio::task::spawn_blocking(move || {
            let start = Instant::now();
            let queue = device.new_command_queue();

            let sampler = make_sampler(&sampling_params);
            let logits_processors = make_logits_processors(&sampling_params);

            let mut stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
            stop_token_ids.extend_from_slice(&sampling_params.stop_token_ids);

            // Create streaming detokenizer.
            let mut detokenizer = create_detokenizer(&tokenizer);

            let model_guard = model.blocking_lock();
            let mut cache = model_guard.make_cache(device.raw());

            // ── Prefill ──
            let logits_array = match model_guard.forward(
                &prompt_tokens,
                Some(&mut cache),
                &registry,
                &queue,
            ) {
                Ok(arr) => arr,
                Err(e) => {
                    warn!("prefill failed: {e}");
                    return;
                }
            };

            let mut logits_f32 = match Self::extract_logits(&logits_array) {
                Ok(l) => l,
                Err(e) => {
                    warn!("logit extraction failed: {e}");
                    return;
                }
            };

            let mut context_tokens = prompt_tokens.clone();

            for proc in &logits_processors {
                proc.process(&mut logits_f32, &context_tokens);
            }

            let first_token = sampler(&logits_f32);
            let first_token_time = start.elapsed().as_secs_f64() + arrival_time;

            let first_logprob = logprobs_k.map(|k| {
                let top = top_logprobs(&logits_f32, k);
                TokenLogprob {
                    token_id: first_token,
                    logprob: rmlx_serve_sampling::log_softmax(&logits_f32)
                        .get(first_token as usize)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY),
                    top_logprobs: top,
                }
            });

            let mut generated_tokens: Vec<u32> = vec![first_token];
            let mut token_logprobs: Vec<TokenLogprob> = Vec::new();
            if let Some(lp) = first_logprob {
                token_logprobs.push(lp);
            }
            context_tokens.push(first_token);

            // Stream the first token.
            detokenizer.add_token(first_token);
            let text_so_far = detokenizer.last_segment().to_string();

            let mut finish_reason: Option<FinishReason> = None;
            if stop_token_ids.contains(&first_token) {
                finish_reason = Some(FinishReason::Stop);
            }
            if generated_tokens.len() >= max_tokens {
                finish_reason = Some(FinishReason::Length);
            }

            // Always emit first token immediately.
            let output = RequestOutput {
                request_id,
                outputs: vec![CompletionOutput {
                    index: 0,
                    text: text_so_far,
                    token_ids: generated_tokens.clone(),
                    finish_reason,
                    logprobs: token_logprobs.clone(),
                }],
                finished: finish_reason.is_some(),
                metrics: None,
            };

            if tx.send(output).is_err() {
                // Receiver dropped -- client disconnected.
                drop(model_guard);
                return;
            }

            // ── Decode loop ──
            let mut tokens_since_emit: usize = 0;
            let mut steps_since_memory_check: usize = 0;

            while finish_reason.is_none() {
                let last_token = *generated_tokens.last().unwrap();

                // Memory pressure monitoring every 64 decode steps.
                steps_since_memory_check += 1;
                if steps_since_memory_check >= 64 {
                    steps_since_memory_check = 0;
                    check_and_clear_memory_pressure(device.raw());
                }

                let logits_array = match model_guard.forward(
                    &[last_token],
                    Some(&mut cache),
                    &registry,
                    &queue,
                ) {
                    Ok(arr) => arr,
                    Err(e) => {
                        warn!("decode failed: {e}");
                        break;
                    }
                };

                let mut logits_f32 = match Self::extract_logits(&logits_array) {
                    Ok(l) => l,
                    Err(e) => {
                        warn!("logit extraction failed: {e}");
                        break;
                    }
                };

                for proc in &logits_processors {
                    proc.process(&mut logits_f32, &context_tokens);
                }

                let next_token = sampler(&logits_f32);

                if let Some(k) = logprobs_k {
                    let top = top_logprobs(&logits_f32, k);
                    token_logprobs.push(TokenLogprob {
                        token_id: next_token,
                        logprob: rmlx_serve_sampling::log_softmax(&logits_f32)
                            .get(next_token as usize)
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY),
                        top_logprobs: top,
                    });
                }

                generated_tokens.push(next_token);
                context_tokens.push(next_token);

                // Update streaming detokenizer.
                detokenizer.add_token(next_token);

                // Check stop conditions.
                if stop_token_ids.contains(&next_token) {
                    finish_reason = Some(FinishReason::Stop);
                } else if generated_tokens.len() >= max_tokens {
                    finish_reason = Some(FinishReason::Length);
                }

                let is_finished = finish_reason.is_some();
                tokens_since_emit += 1;

                // Emit a streaming response every `stream_interval` tokens,
                // or always on the final token.
                let should_emit = is_finished || tokens_since_emit >= stream_interval;
                if !should_emit {
                    continue;
                }
                tokens_since_emit = 0;

                let text_segment = detokenizer.last_segment().to_string();

                // On finish, finalize the detokenizer to flush buffered text.
                let final_text = if is_finished {
                    let remaining = detokenizer.finalize();
                    if remaining.is_empty() {
                        text_segment
                    } else {
                        format!("{}{}", text_segment, remaining)
                    }
                } else {
                    text_segment
                };

                let metrics = if is_finished {
                    let finish_time = start.elapsed().as_secs_f64() + arrival_time;
                    Some(RequestMetrics {
                        arrival_time,
                        first_token_time: Some(first_token_time),
                        finish_time: Some(finish_time),
                        prompt_tokens: prompt_tokens.len(),
                        completion_tokens: generated_tokens.len(),
                    })
                } else {
                    None
                };

                let output = RequestOutput {
                    request_id,
                    outputs: vec![CompletionOutput {
                        index: 0,
                        text: final_text,
                        token_ids: generated_tokens.clone(),
                        finish_reason,
                        logprobs: token_logprobs.clone(),
                    }],
                    finished: is_finished,
                    metrics,
                };

                if tx.send(output).is_err() {
                    // Receiver dropped.
                    break;
                }
            }

            // Update stats (blocking_lock is safe in spawn_blocking context).
            drop(model_guard);
            let mut s = stats.blocking_lock();
            s.total_requests += 1;
            s.total_prompt_tokens += prompt_tokens.len() as u64;
            s.total_completion_tokens += generated_tokens.len() as u64;
            s.uptime_secs = engine_start_time.elapsed().as_secs_f64();
        });

        Ok(rx)
    }

    async fn health(&self) -> EngineHealth {
        EngineHealth {
            is_ready: true,
            status: "ok".to_string(),
            model: self.model_name.clone(),
            active_requests: 0,
        }
    }

    fn get_stats(&self) -> EngineStats {
        // Best-effort: try_lock avoids blocking the caller.
        match self.stats.try_lock() {
            Ok(mut guard) => {
                guard.uptime_secs = self.start_time.elapsed().as_secs_f64();
                guard.clone()
            }
            Err(_) => EngineStats::default(),
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        self.tokenizer
            .encode(text, true)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String, EngineError> {
        self.tokenizer
            .decode(token_ids, true)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Float conversion helpers
// ---------------------------------------------------------------------------

/// Convert an IEEE 754 half-precision float (f16) bit pattern to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut m = mantissa;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let f32_exp = ((e + 127) as u32) & 0xff;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exponent == 31 {
        if mantissa == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (mantissa << 13))
        }
    } else {
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
    }
}

/// Convert a bfloat16 bit pattern to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ---------------------------------------------------------------------------
// Memory pressure monitoring
// ---------------------------------------------------------------------------

/// Check Metal GPU memory pressure and log a warning if critically high.
///
/// Uses the Metal device's `current_allocated_size` and
/// `recommended_max_working_set_size` to compute a pressure ratio.
/// When pressure exceeds 90%, a warning is emitted.
fn check_and_clear_memory_pressure(device: &metal::Device) {
    let allocated = device.current_allocated_size() as f64;
    let recommended = device.recommended_max_working_set_size() as f64;
    if recommended <= 0.0 {
        return;
    }
    let pressure = allocated / recommended;
    if pressure > 0.9 {
        warn!(
            "GPU memory pressure high ({:.1}%), allocated={:.0} MB / recommended={:.0} MB",
            pressure * 100.0,
            allocated / (1024.0 * 1024.0),
            recommended / (1024.0 * 1024.0),
        );
        // On Apple Silicon, Metal manages memory automatically and there is
        // no explicit cache-clear API at the device level.  The warning
        // gives operators visibility; sequence-level caches are freed when
        // the request completes.
    }
}
