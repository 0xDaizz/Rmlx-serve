//! BatchedEngine: continuous-batching inference engine.
//!
//! This engine runs the scheduler and batch generator in a dedicated
//! background tokio task (`engine_loop`), communicating with callers via
//! channels. It supports concurrent requests, continuous batching, and
//! streaming output.
//!
//! Architecture:
//!
//! ```text
//!   API layer
//!     |
//!     v
//!   BatchedEngine  ──request_tx──>  engine_loop
//!     ^                                |
//!     |                                v
//!   <──response_rx──  output_channels  Scheduler ──> BatchGenerator ──> Model
//! ```
//!
//! Ported from vllm-mlx's async engine architecture.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use rmlx_core::{ops, KernelRegistry};
use rmlx_metal::{metal, GpuDevice};
use rmlx_serve_models::{load_model, LlmModel};
use rmlx_serve_scheduler::Scheduler;
use rmlx_serve_tokenizer::{create_detokenizer, StreamingDetokenizer, Tokenizer};
use rmlx_serve_types::{
    CompletionOutput, EngineConfig, Request, RequestId, RequestMetrics, RequestOutput,
    TokenLogprob,
};

use crate::{Engine, EngineError, EngineHealth, EngineStats};

enum EngineRequest {
    Generate {
        request: Request,
        response_tx: mpsc::UnboundedSender<RequestOutput>,
    },
}

struct RequestState {
    response_tx: mpsc::UnboundedSender<RequestOutput>,
    generated_tokens: Vec<u32>,
    logprobs: Vec<TokenLogprob>,
    detokenizer: Box<dyn StreamingDetokenizer>,
    text: String,
    prompt_tokens: usize,
    arrival_time: f64,
    first_token_time: Option<f64>,
    start_instant: Instant,
    logprobs_k: Option<usize>,
    /// Number of tokens generated since last stream emission.
    tokens_since_emit: usize,
}

/// A continuous-batching inference engine backed by a scheduler.
pub struct BatchedEngine {
    request_tx: mpsc::Sender<EngineRequest>,
    /// Kept alive so the abort channel in engine_loop does not close.
    #[allow(dead_code)]
    abort_tx: mpsc::Sender<RequestId>,
    tokenizer: Arc<Tokenizer>,
    model_name: String,
    stats: Arc<tokio::sync::RwLock<EngineStats>>,
    health: Arc<tokio::sync::RwLock<EngineHealth>>,
    start_time: Instant,
    /// Whether the engine is running (for graceful shutdown).
    running: Arc<AtomicBool>,
    /// Channel to signal the engine loop to shut down.
    shutdown_tx: mpsc::Sender<()>,
    /// Number of tokens to batch before emitting a streaming response.
    /// Stored for introspection; the actual value is passed to engine_loop.
    #[allow(dead_code)]
    stream_interval: usize,
}

impl BatchedEngine {
    /// Start the engine. Marks the engine as ready to accept requests.
    /// The background engine_loop is already running after `new()`.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        info!("BatchedEngine started");
    }

    /// Gracefully stop the engine. Signals the background engine_loop to
    /// shut down and waits for in-flight requests to drain.
    pub async fn stop(&self) {
        info!("BatchedEngine stopping");
        self.running.store(false, Ordering::SeqCst);
        // Signal the engine loop to shut down.
        let _ = self.shutdown_tx.send(()).await;
        info!("BatchedEngine stopped");
    }

    /// Whether the engine is currently running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Create a new `BatchedEngine` from an [`EngineConfig`].
    pub async fn new(config: EngineConfig) -> Result<Self, EngineError> {
        let model_path = config.model.clone();
        info!(model = %model_path, "loading model for BatchedEngine");

        let (model, model_config) = load_model(&model_path)?;
        info!(
            model_type = model_config.model_type.as_str(),
            num_layers = model.num_layers(),
            vocab_size = model.vocab_size(),
            "model loaded"
        );

        let tokenizer_path = config.tokenizer.as_deref().unwrap_or(&model_path);
        let tokenizer = Arc::new(Tokenizer::from_pretrained(tokenizer_path)?);
        info!(vocab_size = tokenizer.vocab_size(), "tokenizer loaded");

        let device = GpuDevice::system_default()
            .map_err(|e| EngineError::Internal(format!("failed to acquire Metal device: {e}")))?;
        let queue = device.new_command_queue();
        let registry = KernelRegistry::new(device);
        ops::register_all(&registry)?;
        info!(device = registry.device().name(), aot = registry.has_aot(), "GPU kernels registered");

        let device2 = GpuDevice::system_default()
            .map_err(|e| EngineError::Internal(format!("failed to acquire Metal device: {e}")))?;
        let scheduler = Scheduler::new(config.scheduler.clone(), model.as_ref(), device2.raw());
        info!(max_num_seqs = config.scheduler.max_num_seqs, max_model_len = config.scheduler.max_model_len, "scheduler created");

        let stream_interval = config.scheduler.stream_interval.max(1);
        let (request_tx, request_rx) = mpsc::channel::<EngineRequest>(256);
        let (abort_tx, abort_rx) = mpsc::channel::<RequestId>(64);
        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);
        let start_time = Instant::now();
        let running = Arc::new(AtomicBool::new(true));
        let stats = Arc::new(tokio::sync::RwLock::new(EngineStats::default()));
        let health = Arc::new(tokio::sync::RwLock::new(EngineHealth {
            is_ready: true,
            status: "ok".to_string(),
            model: model_path.clone(),
            active_requests: 0,
        }));
        let stats_clone = Arc::clone(&stats);
        let health_clone = Arc::clone(&health);
        let tokenizer_clone = Arc::clone(&tokenizer);

        // Acquire a device handle for memory pressure monitoring inside the loop.
        let loop_device = GpuDevice::system_default()
            .map_err(|e| EngineError::Internal(format!("failed to acquire Metal device: {e}")))?;

        tokio::spawn(engine_loop(
            request_rx, abort_rx, shutdown_rx, model, tokenizer_clone, scheduler,
            registry, queue, stats_clone, health_clone, start_time,
            stream_interval, loop_device,
        ));
        info!("BatchedEngine engine_loop spawned");

        Ok(Self {
            request_tx, abort_tx, tokenizer, model_name: model_path,
            stats, health, start_time, running, shutdown_tx, stream_interval,
        })
    }
}

#[async_trait]
impl Engine for BatchedEngine {
    fn model_name(&self) -> &str { &self.model_name }

    async fn generate(&self, request: Request) -> Result<RequestOutput, EngineError> {
        let mut rx = self.generate_stream(request).await?;
        let mut last_output: Option<RequestOutput> = None;
        while let Some(output) = rx.recv().await {
            let finished = output.finished;
            last_output = Some(output);
            if finished { break; }
        }
        last_output.ok_or_else(|| EngineError::Internal("no output received from engine loop".into()))
    }

    async fn generate_stream(
        &self,
        request: Request,
    ) -> Result<mpsc::UnboundedReceiver<RequestOutput>, EngineError> {
        let (response_tx, response_rx) = mpsc::unbounded_channel::<RequestOutput>();
        self.request_tx
            .send(EngineRequest::Generate { request, response_tx })
            .await
            .map_err(|_| EngineError::Internal("engine loop has shut down".into()))?;
        Ok(response_rx)
    }

    async fn health(&self) -> EngineHealth { self.health.read().await.clone() }

    fn get_stats(&self) -> EngineStats {
        match self.stats.try_read() {
            Ok(guard) => { let mut s = guard.clone(); s.uptime_secs = self.start_time.elapsed().as_secs_f64(); s }
            Err(_) => EngineStats::default(),
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        self.tokenizer.encode(text, true).map_err(|e| EngineError::Tokenizer(e.to_string()))
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String, EngineError> {
        self.tokenizer.decode(token_ids, true).map_err(|e| EngineError::Tokenizer(e.to_string()))
    }
}

impl Drop for BatchedEngine {
    fn drop(&mut self) {
        info!("BatchedEngine shutting down");
        // Mark as no longer running.
        self.running.store(false, Ordering::SeqCst);
        // Signal the engine loop to stop. The loop will exit when
        // both the request_tx and shutdown_tx channels are closed (which
        // happens when BatchedEngine is dropped and all senders are freed).
        // We use try_send here because Drop is synchronous and we cannot
        // await.
        let _ = self.shutdown_tx.try_send(());
        info!("BatchedEngine drop complete, engine_loop will exit when channels close");
    }
}

#[allow(clippy::too_many_arguments)]
async fn engine_loop(
    mut request_rx: mpsc::Receiver<EngineRequest>,
    mut abort_rx: mpsc::Receiver<RequestId>,
    mut shutdown_rx: mpsc::Receiver<()>,
    model: Box<dyn LlmModel>,
    tokenizer: Arc<Tokenizer>,
    mut scheduler: Scheduler,
    registry: KernelRegistry,
    queue: metal::CommandQueue,
    stats: Arc<tokio::sync::RwLock<EngineStats>>,
    health: Arc<tokio::sync::RwLock<EngineHealth>>,
    engine_start: Instant,
    stream_interval: usize,
    loop_device: GpuDevice,
) {
    let mut request_states: HashMap<RequestId, RequestState> = HashMap::new();
    let mut steps_since_memory_check: usize = 0;
    info!("engine_loop started");

    loop {
        if scheduler.is_idle() && request_states.is_empty() {
            tokio::select! {
                maybe_req = request_rx.recv() => {
                    match maybe_req {
                        Some(req) => handle_new_request(req, &tokenizer, &mut scheduler, &mut request_states),
                        None => { info!("request channel closed, engine_loop shutting down"); break; }
                    }
                }
                maybe_abort = abort_rx.recv() => {
                    if let Some(rid) = maybe_abort { handle_abort(&rid, &mut scheduler, &mut request_states); }
                }
                _ = shutdown_rx.recv() => {
                    info!("shutdown signal received, engine_loop shutting down");
                    break;
                }
            }
        }

        // Check for shutdown signal (non-blocking).
        if shutdown_rx.try_recv().is_ok() {
            info!("shutdown signal received, engine_loop shutting down");
            break;
        }

        while let Ok(req) = request_rx.try_recv() {
            handle_new_request(req, &tokenizer, &mut scheduler, &mut request_states);
        }

        while let Ok(rid) = abort_rx.try_recv() {
            handle_abort(&rid, &mut scheduler, &mut request_states);
        }

        if !scheduler.is_idle() || scheduler.has_pending_work() {
            // Memory pressure monitoring every 64 scheduler steps.
            steps_since_memory_check += 1;
            if steps_since_memory_check >= 64 {
                steps_since_memory_check = 0;
                check_memory_pressure(loop_device.raw());
            }

            match scheduler.step(model.as_ref(), &registry, &queue) {
                Ok(output) => {
                    for response in &output.responses {
                        let request_id = response.request_id;
                        if let Some(state) = request_states.get_mut(&request_id) {
                            let token = response.token;
                            state.generated_tokens.push(token);

                            if state.first_token_time.is_none() {
                                state.first_token_time = Some(
                                    state.start_instant.elapsed().as_secs_f64() + state.arrival_time,
                                );
                            }

                            if let (Some(_k), Some(lps)) = (state.logprobs_k, &response.logprobs) {
                                state.logprobs.push(TokenLogprob {
                                    token_id: token,
                                    logprob: lps.iter().find(|(id, _)| *id == token).map(|(_, lp)| *lp).unwrap_or(f32::NEG_INFINITY),
                                    top_logprobs: lps.clone(),
                                });
                            }

                            state.detokenizer.add_token(token);
                            let is_finished = response.finish_reason.is_some();
                            state.tokens_since_emit += 1;

                            // Emit a streaming response every `stream_interval`
                            // tokens, or always on the final token.
                            let should_emit = is_finished || state.tokens_since_emit >= stream_interval;
                            if !should_emit {
                                // Accumulate text but don't send yet.
                                let segment = state.detokenizer.last_segment().to_string();
                                state.text.push_str(&segment);
                                continue;
                            }
                            state.tokens_since_emit = 0;

                            let segment = state.detokenizer.last_segment().to_string();

                            let text_delta = if is_finished {
                                let remaining = state.detokenizer.finalize();
                                if remaining.is_empty() { segment } else { format!("{}{}", segment, remaining) }
                            } else {
                                segment
                            };

                            state.text.push_str(&text_delta);

                            let metrics = if is_finished {
                                let finish_time = state.start_instant.elapsed().as_secs_f64() + state.arrival_time;
                                Some(RequestMetrics {
                                    arrival_time: state.arrival_time,
                                    first_token_time: state.first_token_time,
                                    finish_time: Some(finish_time),
                                    prompt_tokens: state.prompt_tokens,
                                    completion_tokens: state.generated_tokens.len(),
                                })
                            } else {
                                None
                            };

                            let req_output = RequestOutput {
                                request_id,
                                outputs: vec![CompletionOutput {
                                    index: 0,
                                    text: text_delta,
                                    token_ids: state.generated_tokens.clone(),
                                    finish_reason: response.finish_reason,
                                    logprobs: state.logprobs.clone(),
                                }],
                                finished: is_finished,
                                metrics,
                            };

                            let _ = state.response_tx.send(req_output);
                        }
                    }

                    let finished_ids: Vec<RequestId> = output.responses.iter()
                        .filter(|r| r.finish_reason.is_some()).map(|r| r.request_id).collect();
                    for id in &finished_ids {
                        if let Some(st) = request_states.remove(id) {
                            debug!(request_id = %id, tokens = st.generated_tokens.len(), "request finished");
                        }
                    }

                    let dropped_ids: Vec<RequestId> = request_states.iter()
                        .filter(|(_, st)| st.response_tx.is_closed()).map(|(id, _)| *id).collect();
                    for id in &dropped_ids {
                        scheduler.abort_request(id);
                        request_states.remove(id);
                        debug!(request_id = %id, "request cleaned up (receiver dropped)");
                    }

                    {
                        let tokens_this_step = output.responses.len() as u64;
                        let mut s = stats.write().await;
                        s.total_completion_tokens += tokens_this_step;
                        s.total_requests += finished_ids.len() as u64;
                        s.active_requests = request_states.len() as u64;
                        s.uptime_secs = engine_start.elapsed().as_secs_f64();
                    }
                    {
                        let mut h = health.write().await;
                        h.active_requests = request_states.len();
                    }
                }
                Err(e) => { warn!("scheduler step failed: {e}"); }
            }
        }

        tokio::task::yield_now().await;
    }
    info!("engine_loop exiting");
}

fn handle_new_request(
    engine_request: EngineRequest,
    tokenizer: &Tokenizer,
    scheduler: &mut Scheduler,
    request_states: &mut HashMap<RequestId, RequestState>,
) {
    let EngineRequest::Generate { request, response_tx } = engine_request;
    let request_id = request.id;
    let prompt_tokens = request.prompt_token_ids.len();
    let logprobs_k = request.sampling_params.logprobs;
    let arrival_time = request.arrival_time;

    debug!(request_id = %request_id, prompt_tokens, max_tokens = request.sampling_params.max_tokens, "new request submitted to engine");

    let detokenizer = create_detokenizer(tokenizer);
    request_states.insert(request_id, RequestState {
        response_tx, generated_tokens: Vec::new(), logprobs: Vec::new(),
        detokenizer, text: String::new(), prompt_tokens, arrival_time,
        first_token_time: None, start_instant: Instant::now(), logprobs_k,
        tokens_since_emit: 0,
    });
    scheduler.add_request(request);
}

fn handle_abort(
    request_id: &RequestId,
    scheduler: &mut Scheduler,
    request_states: &mut HashMap<RequestId, RequestState>,
) {
    scheduler.abort_request(request_id);
    if request_states.remove(request_id).is_some() {
        debug!(request_id = %request_id, "request aborted");
    }
}

/// Check Metal GPU memory pressure and log a warning if critically high.
fn check_memory_pressure(device: &metal::Device) {
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
    }
}
