# rmlx-serve Roadmap

## Phase 1: Foundation (Current)
- [x] Workspace structure (12 crates)
- [x] `rmlx-serve-types` — Shared types, OpenAI/Anthropic API models, error types
- [x] `rmlx-serve-tokenizer` — HF tokenizer, chat templates, streaming detokenizer (SPM/BPE/Naive)
- [x] `rmlx-serve-weights` — Safetensors loading, config.json parsing, AWQ/GPTQ quantization
- [x] `rmlx-serve-sampling` — Full sampling pipeline (temperature, top-k/p, min-p, XTC, repetition penalty)
- [x] `rmlx-serve-models` — LlmModel trait, TransformerLlm, dynamic architecture registry (4 models)
- [x] `rmlx-serve-cache` — KVCache, RotatingKVCache, QuantizedKVCache, BatchKVCache, PagedCacheManager, PrefixCacheManager
- [x] `rmlx-serve-scheduler` — BatchGenerator + Scheduler (continuous batching)
- [x] `rmlx-serve-engine` — SimpleEngine, BatchedEngine, generation loop
- [x] `rmlx-serve-speculative` — N-gram, draft model, MTP proposers, rejection sampler
- [x] `rmlx-serve-tools` — 13 tool call parsers + 5 reasoning parsers
- [x] `rmlx-serve-api` — axum HTTP server (OpenAI + Anthropic API)
- [x] `rmlx-serve-cli` — CLI binary (serve, generate, bench)

## Phase 2: Model Expansion
- [ ] Add Gemma/Gemma2 architecture
- [ ] Add Phi-3/Phi-4 architecture
- [ ] Add Mistral/Mistral-Nemo architecture
- [ ] Add GLM-4 architecture
- [ ] Add Command-R architecture
- [ ] Add StarCoder2 architecture
- [ ] Add Nemotron architecture
- [ ] Add Cohere architecture
- [ ] Target: 20+ architectures

## Phase 3: Performance Optimization
- [ ] Flash Attention 2 integration (when RMLX adds tiled Metal kernel)
- [ ] Paged attention kernel optimization
- [ ] Chunked prefill optimization (pipeline prefill + decode)
- [ ] Speculative decoding auto-tuning
- [ ] Memory pool optimization for KV cache
- [ ] Batch scheduling fairness improvements

## Phase 4: Multimodal
- [ ] Vision-Language Model support (LLaVA, Qwen-VL, InternVL)
- [ ] Image preprocessing pipeline
- [ ] Video frame extraction
- [ ] VisionEmbeddingCache for multimodal prefix caching
- [ ] MLLMScheduler for multimodal batching
- [ ] Conv1d/Conv2d support in RMLX (required for vision encoders)

## Phase 5: Audio
- [ ] Text-to-Speech (Kokoro, Chatterbox, F5-TTS, MaskGCT, Dia, Spark)
- [ ] Speech-to-Text (Whisper, Parakeet)
- [ ] `/v1/audio/speech` and `/v1/audio/transcriptions` endpoints
- [ ] Audio streaming support

## Phase 6: Advanced Serving
- [ ] MCP (Model Context Protocol) server integration
- [ ] Dynamic model loading/unloading
- [ ] Multi-model serving
- [ ] Request routing and load balancing
- [ ] Disk-based KV cache persistence
- [ ] GGUF format support

## Phase 7: Distributed
- [ ] Multi-GPU tensor parallelism via RMLX distributed
- [ ] Pipeline parallelism
- [ ] Expert parallelism for MoE models
- [ ] RDMA/Thunderbolt 5 transport optimization
- [ ] SSH-based distributed launcher

## Phase 8: Production
- [ ] Prometheus metrics export
- [ ] Structured logging (JSON)
- [ ] Graceful shutdown with request draining
- [ ] Health check improvements
- [ ] Rate limiting
- [ ] Request timeout configuration
- [ ] SSL/TLS support

## RMLX Dependencies Roadmap

Features needed from RMLX for full rmlx-serve capabilities:

| Feature | rmlx-serve Workaround | RMLX Target |
|---------|----------------------|-------------|
| allreduce/allgather | all_sum pattern in engine | RMLX collective ops |
| GELU activation | CPU fallback | Metal GELU kernel |
| Flash Attention 2 | SDPA (existing) | Metal tiled kernel |
| GGUF format | rmlx-serve-weights impl | RMLX native support |
| AWQ/GPTQ quantization | rmlx-serve-weights unpacking | RMLX native support |
| FP8 support | N/A | DType extension |
| RotatingKVCache | Pure Rust in rmlx-serve-cache | LayerKvCache circular mode |
| QuantizedKVCache (4/8-bit) | Pure Rust in rmlx-serve-cache | Quantized KV support |
| BatchKVCache | Vec<LayerKvCache> wrapper | Batch-aware cache |
| Dynamic shapes | Max seq_len pre-allocation | Dynamic dispatch |
| Conv1d/Conv2d | Not implemented | Multimodal support |

## Components to Migrate from RMLX

As RMLX repositions as a general-purpose ML framework, LLM-specific code migrates to rmlx-serve:

| Component | RMLX Current Location | rmlx-serve Target |
|-----------|----------------------|-------------------|
| Model configs (LLaMA, Qwen, etc.) | `rmlx-nn/src/models/` | `rmlx-serve-models` |
| TransformerModel/Block | `rmlx-nn/src/transformer.rs` | `rmlx-serve-models` |
| LoRA fine-tuning (transformer) | `rmlx-core/src/lora.rs` | `rmlx-serve-models` |

**Remains in RMLX** (general-purpose NN building blocks):
- Linear, Embedding, Attention, MoE, SDPA, RoPE
