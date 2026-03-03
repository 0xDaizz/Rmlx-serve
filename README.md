# rmlx-serve

High-performance LLM inference serving for Apple Silicon, built in Rust.

**rmlx-serve** integrates the serving architecture of [vLLM-MLX](https://github.com/waybarrios/vllm-mlx) with the model inference capabilities of [mlx-lm](https://github.com/ml-explore/mlx-lm), reimplemented in Rust on top of the [RMLX](https://github.com/rmlx) Metal GPU framework.

Supports LLM, VLM, Audio (TTS/STT), and Embedding model serving through OpenAI and Anthropic-compatible APIs.

## Features

- **Metal GPU acceleration** via RMLX (native Apple Silicon, no Python runtime)
- **Continuous batching** with integrated BatchGenerator + Scheduler
- **Paged KV cache** with prefix caching, LRU eviction, and quantized KV storage
- **Speculative decoding** — N-gram, draft model, and MTP (Multi-Token Prediction)
- **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
- **Anthropic-compatible API** (`/v1/messages`)
- **Tool calling** — 13 parser formats (Hermes, Qwen, Llama, Mistral, DeepSeek, etc.)
- **Reasoning support** — 5 parser formats (think tags, DeepSeek-R1, Qwen3, etc.)
- **Streaming** — SSE-based token-by-token streaming with streaming detokenizer
- **100+ model architectures** — LLaMA, Qwen, Mixtral, DeepSeek, Gemma, Phi, and more
- **Quantization** — AWQ, GPTQ, and native RMLX quantized formats (Q4_0, Q4_1, Q8_0)
- **Distributed inference** — tensor parallelism via RDMA/Thunderbolt 5
- **LoRA adapters** — hot-loadable fine-tuned adapters

## Architecture

```
rmlx-serve (12 Rust crates)
├── rmlx-serve-types        Shared types, API models, errors
├── rmlx-serve-tokenizer    HF tokenizer + streaming detokenizer
├── rmlx-serve-weights      Safetensors loading, config parsing, quantization
├── rmlx-serve-sampling     Full sampling pipeline (top-k/p, min-p, XTC, repetition penalty)
├── rmlx-serve-models       Model abstraction, dynamic architecture selection
├── rmlx-serve-cache        KV cache (standard, rotating, quantized, paged, prefix)
├── rmlx-serve-scheduler    BatchGenerator + Scheduler (continuous batching)
├── rmlx-serve-engine       SimpleEngine, BatchedEngine, generation loop
├── rmlx-serve-speculative  N-gram, draft model, MTP, rejection sampling
├── rmlx-serve-tools        Tool call parsers (13) + reasoning parsers (5)
├── rmlx-serve-api          axum HTTP server (OpenAI + Anthropic API)
└── rmlx-serve-cli          CLI binary (serve, generate, bench)
```

Built on RMLX (6 crates):
- `rmlx-core` — Array, GPU ops, kernel registry
- `rmlx-nn` — Neural network layers, transformer blocks
- `rmlx-metal` — Metal GPU device abstraction
- `rmlx-alloc` — GPU memory allocator
- `rmlx-distributed` — Distributed computing (tensor parallelism)
- `rmlx-rdma` — Thunderbolt 5 RDMA transport

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
| `--api-key` | none | API key for authentication |
| `--dtype` | auto | Model dtype (float16, bfloat16, float32) |

### Cache Options

| Flag | Default | Description |
|------|---------|-------------|
| `--no-prefix-cache` | `false` | Disable prefix caching |
| `--cache-memory-mb` | auto | KV cache memory budget |
| `--kv-cache-quantization` | none | KV cache quantization (4bit, 8bit) |

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

Initial support (4 architectures):
- **LLaMA** — LLaMA 2, LLaMA 3, LLaMA 3.1, LLaMA 3.2
- **Qwen** — Qwen2, Qwen2.5, Qwen3
- **Mixtral** — Mixtral 8x7B, Mixtral 8x22B
- **DeepSeek** — DeepSeek-V2, DeepSeek-V3

Planned (expanding to 100+ via mlx-lm parity):
- Gemma, Phi, GLM, Mistral, Nemotron, Command-R, StarCoder, and more

## Building

```bash
# Requirements: Rust 1.80+, macOS with Apple Silicon
git clone https://github.com/0xDaizz/Rmlx-serve
cd Rmlx-serve
cargo build --release
cargo test --workspace
```

## Acknowledgments

- [RMLX](https://github.com/rmlx) — Rust ML framework for Apple Silicon (Metal GPU backend)
- [vLLM-MLX](https://github.com/waybarrios/vllm-mlx) — Apple Silicon MLX backend for vLLM
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLM inference library
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Vision-language models
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — Text-to-Speech and Speech-to-Text
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) — Text embeddings
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput LLM serving

## License

MIT OR Apache-2.0
