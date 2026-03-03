# rmlx-serve

High-performance LLM inference serving for Apple Silicon, built in Rust.

**rmlx-serve** integrates the serving architecture of [vLLM-MLX](https://github.com/waybarrios/vllm-mlx) with the model inference capabilities of [mlx-lm](https://github.com/ml-explore/mlx-lm), reimplemented in Rust on top of the [RMLX](https://github.com/rmlx) Metal GPU framework.

## Current Status

rmlx-serve has completed a major five-phase overhaul. The system compiles, passes **228 tests** across the workspace, and the core inference pipeline is connected end-to-end: CLI talks to real engines, speculative decoding is wired to the scheduler, tool/reasoning parsers feed into API handlers, and prefix caching integrates with request lifecycle.

Key areas that remain incomplete:
- Multimodal (vision-language) model support
- Audio (TTS/STT) serving
- Embedding engine
- Multi-rank distributed transport (stub interfaces only)
- MTP head computation (interface only, no real model heads)
- Real batched prefill/decode (requires model API changes)

## Features

- **Metal GPU acceleration** via RMLX (native Apple Silicon, no Python runtime)
- **Hardware auto-detection** -- Apple Silicon M1-M4 detection with 5-tier performance auto-tuning
- **Continuous batching** with integrated BatchGenerator + Scheduler
- **Paged KV cache** with prefix caching (wired to scheduler), LRU eviction, CacheOps trait, and causal mask generation
- **Speculative decoding** -- N-gram (incremental indexing), draft model, and MTP proposer (interface), wired to scheduler with epsilon floor and probe reactivation
- **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
- **Anthropic-compatible API** (`/v1/messages`)
- **MCP Runtime** -- client, manager, security, config, and API endpoints for Model Context Protocol
- **Tool calling** -- 13 parser formats (Hermes, Qwen, Llama, Mistral, DeepSeek, etc.) with 81 tests, wired to streaming and non-streaming handlers
- **Reasoning support** -- 5 parser formats (think tags, DeepSeek-R1, Qwen3, etc.)
- **Streaming** -- SSE-based token-by-token streaming with streaming detokenizer, completions streaming, `stream_options.include_usage` support
- **Sampling pipeline** -- correct filter order (top_p -> min_p -> XTC -> top_k), temperature applied after filters, full token context for logits processors
- **16 model types** across 10 architecture families -- LLaMA, Qwen, Mixtral, DeepSeek, Gemma, Phi, Mistral, StarCoder, Cohere/Command-R, InternLM
- **RoPE scaling** -- Linear, NTK, and YaRN modes
- **Quantization** -- AWQ/GPTQ (with correct interleaved bit extraction, unsigned values, transpose, 2D scale indexing) and native RMLX formats (Q4_0, Q4_1, Q8_0)
- **Weights** -- sharded loading, tie_word_embeddings check, dtype mapping
- **API hardening** -- constant-time API key comparison (subtle crate), rate limiting, disconnect detection, structured output, default parameter resolution
- **Engine** -- spawn_blocking for forward passes, memory pressure monitoring, Drop impl, lifecycle management, stream_interval
- **Tokenizer** -- UTF-8 multi-byte handling, thinking/tool token detection, chat template improvements
- **Expert parallelism** -- EPAdapter with dispatch/combine cycle
- **Distributed** -- communicator, worker, launcher (stub interfaces for multi-rank transport)
- **LoRA adapters** -- hot-loadable fine-tuned adapters
- **50+ CLI flags** via comprehensive args.rs, connecting to real SimpleEngine/BatchedEngine

## Architecture

```
rmlx-serve (12 Rust crates)
├── rmlx-serve-types        Shared types, API models, errors, MCP types
├── rmlx-serve-tokenizer    HF tokenizer + streaming detokenizer
├── rmlx-serve-weights      Safetensors loading, config parsing, quantization
├── rmlx-serve-sampling     Full sampling pipeline (top-k/p, min-p, XTC, repetition penalty)
├── rmlx-serve-models       Model abstraction, dynamic architecture registry (16 types)
├── rmlx-serve-cache        KV cache (standard, rotating, quantized, paged, prefix), CacheOps trait
├── rmlx-serve-scheduler    BatchGenerator + Scheduler (continuous batching, prefix cache integration)
├── rmlx-serve-engine       SimpleEngine, BatchedEngine, EP adapter, distributed, hardware detection
├── rmlx-serve-speculative  N-gram, draft model, MTP proposers, rejection sampling
├── rmlx-serve-tools        Tool call parsers (13) + reasoning parsers (5) — 81 tests
├── rmlx-serve-api          axum HTTP server (OpenAI + Anthropic + MCP endpoints)
└── rmlx-serve-cli          CLI binary (serve, generate, bench) — 50+ flags
```

Built on RMLX (6 crates):
- `rmlx-core` -- Array, GPU ops, kernel registry
- `rmlx-nn` -- Neural network layers, transformer blocks
- `rmlx-metal` -- Metal GPU device abstraction
- `rmlx-alloc` -- GPU memory allocator
- `rmlx-distributed` -- Distributed computing (tensor parallelism)
- `rmlx-rdma` -- Thunderbolt 5 RDMA transport

## Quick Start

### Serve a model

```bash
cargo run --release --bin rmlx-serve -- serve \
    --model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct \
    --port 8000 \
    --continuous-batching
```

### Generate text

```bash
cargo run --release --bin rmlx-serve -- generate \
    --model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct \
    --prompt "Explain quantum computing in simple terms" \
    --max-tokens 512 \
    --temperature 0.7
```

### Query the API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "stream": true
  }'
```

### Benchmark

```bash
cargo run --release --bin rmlx-serve -- bench \
    --model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct \
    --num-requests 100 \
    --concurrency 8 \
    --continuous-batching
```

## Configuration

### Serving Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to HuggingFace model directory |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |
| `--continuous-batching` | `false` | Enable continuous batching (BatchedEngine) |
| `--max-num-seqs` | `256` | Maximum concurrent sequences |
| `--api-key` | none | API key for authentication (constant-time comparison) |
| `--dtype` | auto | Model dtype (float16, bfloat16, float32) |

### Cache Options

| Flag | Default | Description |
|------|---------|-------------|
| `--no-prefix-cache` | `false` | Disable prefix caching |
| `--cache-memory-mb` | auto | KV cache memory budget |
| `--kv-cache-quantization` | none | KV cache quantization (4bit, 8bit) |
| `--cache-type` | auto | Cache strategy selection (standard, rotating, paged) |

### Speculative Decoding

| Flag | Default | Description |
|------|---------|-------------|
| `--speculative-method` | none | Method: ngram, draft, mtp |
| `--num-speculative-tokens` | `5` | Number of speculative tokens |

### Tool Calling & Reasoning

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-auto-tool-choice` | `false` | Enable automatic tool call detection |
| `--tool-call-parser` | auto | Parser: hermes, qwen, llama, mistral, deepseek, auto |
| `--enable-thinking` | `false` | Enable reasoning/thinking extraction |
| `--reasoning-parser` | think | Parser: think, deepseek_r1, qwen3, gpt_oss, harmony |

## Supported Models

16 registered model types across 10 architecture families:

| Family | Model Types | Examples |
|--------|-------------|----------|
| **LLaMA** | `llama` | LLaMA 2, LLaMA 3, LLaMA 3.1, LLaMA 3.2 |
| **Qwen** | `qwen2`, `qwen3` | Qwen2, Qwen2.5, Qwen3 |
| **MoE** | `mixtral`, `deepseek_v2`, `deepseek_v3` | Mixtral 8x7B/8x22B, DeepSeek-V2/V3 |
| **Mistral** | `mistral` | Mistral 7B, Mistral-Nemo |
| **Gemma** | `gemma`, `gemma2`, `gemma3` | Gemma, Gemma 2, Gemma 3 |
| **Phi** | `phi`, `phi3` | Phi-2, Phi-3, Phi-4 |
| **StarCoder** | `starcoder2` | StarCoder2 |
| **Cohere** | `cohere`, `command-r` | Command-R, Command-R+ |
| **InternLM** | `internlm2` | InternLM2 |

All architectures use a unified TransformerLlm abstraction with RoPE scaling (Linear/NTK/YaRN) and prefill chunking support.

## Building

```bash
# Requirements: Rust 1.80+, macOS with Apple Silicon
git clone https://github.com/0xDaizz/Rmlx-serve
cd Rmlx-serve
cargo build --release
cargo test --workspace
```

## Acknowledgments

- [RMLX](https://github.com/rmlx) -- Rust ML framework for Apple Silicon (Metal GPU backend)
- [vLLM-MLX](https://github.com/waybarrios/vllm-mlx) -- Apple Silicon MLX backend for vLLM
- [MLX](https://github.com/ml-explore/mlx) -- Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) -- LLM inference library
- [vLLM](https://github.com/vllm-project/vllm) -- High-throughput LLM serving

## License

MIT OR Apache-2.0
