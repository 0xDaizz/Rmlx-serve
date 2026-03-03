//! `rmlx-serve generate` subcommand -- one-shot text generation.

use rmlx_serve_engine::generate_text;
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
pub async fn run_generate(
    args: GenerateArgs,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
    // 2. Run generation via the convenience function
    // -----------------------------------------------------------------------
    let start = tokio::time::Instant::now();

    let response = generate_text(&args.model, &args.prompt, sampling_params).await?;

    let elapsed = start.elapsed();

    // -----------------------------------------------------------------------
    // 3. Print result
    // -----------------------------------------------------------------------
    println!("{}", response.text);

    if args.verbose {
        let total_secs = elapsed.as_secs_f64();

        eprintln!();
        eprintln!("--- statistics ---");
        eprintln!("prompt tokens  : {}", response.prompt_tokens);
        eprintln!("gen tokens     : {}", response.generation_tokens);
        eprintln!("total time     : {:.3}s", total_secs);
        eprintln!("prompt tok/s   : {:.1}", response.prompt_tps);
        eprintln!("gen tok/s      : {:.1}", response.generation_tps);
        eprintln!("finish reason  : {:?}", response.finish_reason);
        if response.peak_memory_mb > 0.0 {
            eprintln!("peak memory    : {:.1} MB", response.peak_memory_mb);
        }
    }

    Ok(())
}
