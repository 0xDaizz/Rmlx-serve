//! `rmlx-serve bench` subcommand -- benchmark inference throughput and latency.

use rmlx_serve_engine::{BatchedEngine, Engine, SimpleEngine};
use rmlx_serve_types::config::EngineConfig;
use rmlx_serve_types::SamplingParams;
use std::sync::Arc;
use tokio::time::Instant;
use tracing::info;

/// Arguments for the `bench` subcommand.
#[derive(clap::Args, Debug)]
pub struct BenchArgs {
    /// Path or HuggingFace repo id for the model.
    #[arg(long, required = true)]
    pub model: String,

    /// Number of requests to run.
    #[arg(long, default_value = "10")]
    pub num_requests: usize,

    /// Target prompt length in tokens (dummy prompt will be generated).
    #[arg(long, default_value = "128")]
    pub prompt_length: usize,

    /// Maximum number of tokens to generate per request.
    #[arg(long, default_value = "128")]
    pub max_tokens: usize,

    /// Number of concurrent requests to issue in parallel.
    #[arg(long, default_value = "1")]
    pub concurrency: usize,

    /// Enable continuous batching mode for the engine.
    #[arg(long, default_value = "false")]
    pub continuous_batching: bool,
}

/// Timing results for a single benchmark request.
#[derive(Debug, Clone)]
struct RequestResult {
    /// Time to first token, in seconds.
    ttft_secs: f64,
    /// End-to-end latency, in seconds.
    e2e_latency_secs: f64,
    /// Number of tokens generated.
    gen_tokens: usize,
    /// Generation throughput (tokens / second) for this request.
    tps: f64,
}

/// Run the `bench` subcommand.
pub async fn run_bench(args: BenchArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("=========================================================");
    info!("  rmlx-serve bench");
    info!("=========================================================");
    info!("model          : {}", args.model);
    info!("num_requests   : {}", args.num_requests);
    info!("prompt_length  : {} tokens", args.prompt_length);
    info!("max_tokens     : {}", args.max_tokens);
    info!("concurrency    : {}", args.concurrency);
    info!(
        "batching       : {}",
        if args.continuous_batching {
            "continuous"
        } else {
            "simple"
        }
    );
    info!("---------------------------------------------------------");

    // -----------------------------------------------------------------------
    // 1. Create engine
    // -----------------------------------------------------------------------
    let engine_config = EngineConfig {
        model: args.model.clone(),
        scheduler: rmlx_serve_types::config::SchedulerConfig {
            max_num_seqs: args.concurrency.max(256),
            ..Default::default()
        },
        ..Default::default()
    };

    let engine: Arc<dyn Engine> = if args.continuous_batching {
        info!("initialising BatchedEngine ...");
        Arc::new(BatchedEngine::new(engine_config).await?)
    } else {
        info!("initialising SimpleEngine ...");
        Arc::new(SimpleEngine::new(engine_config).await?)
    };

    info!("engine initialised, model loaded");

    // -----------------------------------------------------------------------
    // 2. Generate dummy prompts
    // -----------------------------------------------------------------------
    // Create a repeating token sequence of the requested length.
    // Uses token ID 1 repeated -- a real benchmark would use varied text.
    let dummy_prompt_tokens: Vec<u32> = (0..args.prompt_length)
        .map(|i| ((i % 100) + 1) as u32)
        .collect();

    info!(
        "dummy prompt: {} tokens (ids 1..100 repeating)",
        dummy_prompt_tokens.len()
    );

    // -----------------------------------------------------------------------
    // 3. Run benchmark requests
    // -----------------------------------------------------------------------
    let bench_start = Instant::now();
    let mut results: Vec<RequestResult> = Vec::with_capacity(args.num_requests);

    if args.concurrency <= 1 {
        // Sequential execution
        for i in 0..args.num_requests {
            info!("request {}/{}", i + 1, args.num_requests);
            let result =
                run_single_request(engine.clone(), dummy_prompt_tokens.clone(), args.max_tokens)
                    .await?;
            results.push(result);
        }
    } else {
        // Concurrent execution in batches of `concurrency`
        let mut remaining = args.num_requests;
        let mut batch_idx = 0;
        while remaining > 0 {
            let batch_size = remaining.min(args.concurrency);
            info!(
                "batch {}: launching {} concurrent request(s) ({} remaining)",
                batch_idx + 1,
                batch_size,
                remaining
            );

            let mut handles = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let eng = engine.clone();
                let prompt = dummy_prompt_tokens.clone();
                let max_tok = args.max_tokens;
                handles.push(tokio::spawn(async move {
                    run_single_request(eng, prompt, max_tok).await
                }));
            }

            for handle in handles {
                let result = handle.await??;
                results.push(result);
            }

            remaining -= batch_size;
            batch_idx += 1;
        }
    }

    let bench_elapsed = bench_start.elapsed();

    // -----------------------------------------------------------------------
    // 4. Compute and display statistics
    // -----------------------------------------------------------------------
    if results.is_empty() {
        eprintln!("no results collected");
        return Ok(());
    }

    let total_gen_tokens: usize = results.iter().map(|r| r.gen_tokens).sum();

    let mut ttft_values: Vec<f64> = results.iter().map(|r| r.ttft_secs).collect();
    let mut latency_values: Vec<f64> = results.iter().map(|r| r.e2e_latency_secs).collect();
    let mut tps_values: Vec<f64> = results.iter().map(|r| r.tps).collect();

    ttft_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    latency_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tps_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let total_secs = bench_elapsed.as_secs_f64();
    let overall_tps = if total_secs > 0.0 {
        total_gen_tokens as f64 / total_secs
    } else {
        0.0
    };

    println!();
    println!("=========================================================");
    println!("  Benchmark Results");
    println!("=========================================================");
    println!();
    print_table_header();
    print_metric_row("TTFT (s)", &ttft_values);
    print_metric_row("E2E Latency (s)", &latency_values);
    print_metric_row("Throughput (tok/s)", &tps_values);
    print_table_footer();
    println!();
    println!("  Total requests       : {}", args.num_requests);
    println!("  Total tokens gen     : {}", total_gen_tokens);
    println!("  Total time           : {:.3}s", total_secs);
    println!("  Overall throughput   : {:.1} tok/s", overall_tps);
    println!("  Concurrency          : {}", args.concurrency);
    println!();

    Ok(())
}

/// Execute a single inference request and return timing data.
async fn run_single_request(
    engine: Arc<dyn Engine>,
    prompt_token_ids: Vec<u32>,
    max_tokens: usize,
) -> Result<RequestResult, Box<dyn std::error::Error + Send + Sync>> {
    let params = SamplingParams {
        temperature: 0.0,
        max_tokens,
        ..SamplingParams::default()
    };

    let request_start = Instant::now();

    let request = rmlx_serve_types::Request::new(
        prompt_token_ids,
        params,
        request_start.elapsed().as_secs_f64(),
    );

    let output = engine.generate(request).await.map_err(
        |e| -> Box<dyn std::error::Error + Send + Sync> {
            Box::new(std::io::Error::other(e.to_string()))
        },
    )?;

    let e2e_elapsed = request_start.elapsed();

    let gen_tokens = output
        .outputs
        .first()
        .map(|o| o.token_ids.len())
        .unwrap_or(0);

    // Extract TTFT from metrics if available, otherwise estimate as a fraction
    // of the total time.
    let ttft_secs = output.metrics.as_ref().and_then(|m| m.ttft()).unwrap_or({
        // Heuristic fallback: assume prefill is ~10% of total time.
        e2e_elapsed.as_secs_f64() * 0.1
    });

    let e2e_secs = e2e_elapsed.as_secs_f64();
    let tps = if e2e_secs > 0.0 && gen_tokens > 0 {
        gen_tokens as f64 / e2e_secs
    } else {
        0.0
    };

    Ok(RequestResult {
        ttft_secs,
        e2e_latency_secs: e2e_secs,
        gen_tokens,
        tps,
    })
}

// ---------------------------------------------------------------------------
// Percentile helpers
// ---------------------------------------------------------------------------

/// Compute a percentile from a *sorted* slice using linear interpolation.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    assert!(!sorted.is_empty());
    assert!((0.0..=100.0).contains(&p));

    if sorted.len() == 1 {
        return sorted[0];
    }

    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

// ---------------------------------------------------------------------------
// Table formatting
// ---------------------------------------------------------------------------

fn print_table_header() {
    println!(
        "  {:<20} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Metric", "Mean", "P50", "P90", "P95", "P99"
    );
    println!("  {}", "-".repeat(72));
}

fn print_metric_row(label: &str, sorted_values: &[f64]) {
    let avg = mean(sorted_values);
    let p50 = percentile(sorted_values, 50.0);
    let p90 = percentile(sorted_values, 90.0);
    let p95 = percentile(sorted_values, 95.0);
    let p99 = percentile(sorted_values, 99.0);

    println!(
        "  {:<20} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
        label, avg, p50, p90, p95, p99
    );
}

fn print_table_footer() {
    println!("  {}", "-".repeat(72));
}
