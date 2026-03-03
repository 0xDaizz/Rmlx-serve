//! `rmlx-serve` -- high-performance LLM serving on Apple Silicon.
//!
//! This is the CLI binary entry point.  It exposes three subcommands:
//!
//! - **`serve`** -- start the HTTP API server (OpenAI/Anthropic-compatible).
//! - **`generate`** -- one-shot text generation from the command line.
//! - **`bench`** -- benchmark inference throughput and latency.

mod bench;
mod generate;
mod serve;
pub(crate) mod stub_engine;

use clap::Parser;
use tracing_subscriber::EnvFilter;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// High-performance LLM serving on Apple Silicon, powered by RMLX.
#[derive(Parser, Debug)]
#[command(
    name = "rmlx-serve",
    version,
    about = "High-performance LLM serving on Apple Silicon"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Start the HTTP API server (OpenAI + Anthropic compatible).
    Serve(serve::ServeArgs),

    /// Run one-shot text generation from the command line.
    Generate(generate::GenerateArgs),

    /// Benchmark inference throughput and latency.
    Bench(bench::BenchArgs),
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // -----------------------------------------------------------------------
    // 1. Initialise tracing
    // -----------------------------------------------------------------------
    // Respect the RUST_LOG environment variable.  If unset, default to `info`
    // for our own crates and `warn` for everything else.
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("rmlx_serve=info,rmlx_serve_cli=info,rmlx_serve_engine=info,rmlx_serve_api=info,warn")
    });

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();

    // -----------------------------------------------------------------------
    // 2. Parse CLI args and dispatch
    // -----------------------------------------------------------------------
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve(args) => serve::run_serve(args).await,
        Commands::Generate(args) => generate::run_generate(args).await,
        Commands::Bench(args) => bench::run_bench(args).await,
    }
}
