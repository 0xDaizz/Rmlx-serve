//! `rmlx-serve generate` subcommand -- one-shot text generation.

use rmlx_serve_types::SamplingParams;
use tracing::info;

/// Arguments for the `generate` subcommand.
#[derive(clap::Args, Debug)]
pub struct GenerateArgs {
    /// Path or HuggingFace repo id for the model.
    #[arg(long, required = true)]
    pub model: String,

    /// The prompt to complete.
    #[arg(long, required = true)]
    pub prompt: String,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value = "256")]
    pub max_tokens: usize,

    /// Sampling temperature. 0.0 = greedy decoding.
    #[arg(long, default_value = "0.0")]
    pub temperature: f32,

    /// Nucleus (top-p) sampling threshold.
    #[arg(long, default_value = "1.0")]
    pub top_p: f32,

    /// Optional seed for reproducible sampling.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Print timing statistics after generation.
    #[arg(long, default_value = "false")]
    pub verbose: bool,
}

/// Run the `generate` subcommand.
pub async fn run_generate(args: GenerateArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // -----------------------------------------------------------------------
    // 1. Build SamplingParams
    // -----------------------------------------------------------------------
    let sampling_params = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        max_tokens: args.max_tokens,
        seed: args.seed,
        ..SamplingParams::default()
    };

    info!("model          : {}", args.model);
    info!("prompt         : {:?}", args.prompt);
    info!("max_tokens     : {}", args.max_tokens);
    info!("temperature    : {}", args.temperature);
    info!("top_p          : {}", args.top_p);
    if let Some(seed) = args.seed {
        info!("seed           : {}", seed);
    }

    // -----------------------------------------------------------------------
    // 2. Create engine and run generation
    // -----------------------------------------------------------------------
    //
    // When `rmlx_serve_engine::generate_text` is implemented:
    //
    //   let response = rmlx_serve_engine::generate_text(
    //       &args.model,
    //       &args.prompt,
    //       sampling_params,
    //   ).await?;
    //
    // For now we run through the Engine trait via StubEngine.

    let engine_config = rmlx_serve_types::config::EngineConfig {
        model: args.model.clone(),
        ..rmlx_serve_types::config::EngineConfig::default()
    };

    let engine = crate::stub_engine::StubEngine::new(engine_config);

    let start = tokio::time::Instant::now();

    // Build a Request from the prompt.
    let prompt_token_ids = rmlx_serve_engine::Engine::encode(&engine, &args.prompt)?;
    let prompt_len = prompt_token_ids.len();

    let request = rmlx_serve_types::Request::new(
        prompt_token_ids,
        sampling_params,
        start.elapsed().as_secs_f64(),
    );

    let output = rmlx_serve_engine::Engine::generate(&engine, request).await?;

    let elapsed = start.elapsed();

    // -----------------------------------------------------------------------
    // 3. Print result
    // -----------------------------------------------------------------------
    if let Some(completion) = output.outputs.first() {
        println!("{}", completion.text);

        if args.verbose {
            let gen_tokens = completion.token_ids.len();
            let total_secs = elapsed.as_secs_f64();
            let tps = if total_secs > 0.0 {
                gen_tokens as f64 / total_secs
            } else {
                0.0
            };

            eprintln!();
            eprintln!("--- statistics ---");
            eprintln!("prompt tokens  : {}", prompt_len);
            eprintln!("gen tokens     : {}", gen_tokens);
            eprintln!("total time     : {:.3}s", total_secs);
            eprintln!("throughput     : {:.1} tok/s", tps);
            if let Some(ref metrics) = output.metrics {
                if let Some(ttft) = metrics.ttft() {
                    eprintln!("ttft           : {:.3}s", ttft);
                }
                if let Some(lat) = metrics.total_latency() {
                    eprintln!("e2e latency    : {:.3}s", lat);
                }
                if let Some(rate) = metrics.tokens_per_second() {
                    eprintln!("decode tok/s   : {:.1}", rate);
                }
            }
        }
    } else {
        eprintln!("warning: engine returned no completions");
    }

    Ok(())
}
