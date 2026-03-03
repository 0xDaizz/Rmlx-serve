# rmlx-serve Roadmap

## Implemented

### Phase 1 -- P0 Bug Fixes
- [x] AWQ/GPTQ interleaved bit extraction, unsigned values, transpose, 2D scale indexing
- [x] Sampling pipeline: correct filter order (top_p -> min_p -> XTC -> top_k), temperature after filters
- [x] Scheduler: HashMap-based cache slot mapping, full token context for logits processors
- [x] Speculative: epsilon floor, probe reactivation logic
- [x] API: constant-time API key comparison (subtle crate)

### Phase 2 -- StubEngine Removed
- [x] CLI connects to real SimpleEngine/BatchedEngine (StubEngine removed)
- [x] 50+ CLI flags via comprehensive args.rs
- [x] Config types extended: SchedulerConfig, CacheConfig, EngineConfig, ServerConfig (33+ new fields)
- [x] RequestStatus: 4 new variants

### Phase 3 -- Orphan Code Connected
- [x] Tool/reasoning parsers wired to API handlers (streaming + non-streaming)
- [x] Speculative decoding wired to scheduler (ngram/mtp/draft)
- [x] Prefix cache wired to scheduler (lookup on admission, insert on completion)
- [x] CacheType enum for cache strategy selection from config

### Phase 4 -- Quality Fixes
- [x] Tokenizer: UTF-8 multi-byte handling, thinking/tool token detection, chat template improvements
- [x] Engine: spawn_blocking for forward passes, memory pressure monitoring, Drop impl, lifecycle, stream_interval
- [x] Tools: 7 parser format fixes (Hermes fallback, Llama function tags, Mistral IDs, DeepSeek tokens), aliases, 81 tests
- [x] API: completions streaming, rate limiting, disconnect detection, structured output, stream_options.include_usage, default parameter resolution
- [x] Models: sharded_load, RoPE scaling (Linear/NTK/YaRN), 10 new architectures (total 16 model types), prefill chunking
- [x] Weights: tie_word_embeddings check, dtype mapping
- [x] Cache: CacheOps trait, make_causal_mask
- [x] Speculative: incremental n-gram indexing, MTP proposer interface fleshed out

### Phase 5 -- New Subsystems
- [x] MCP Runtime: client, manager, security, config, API endpoints
- [x] Expert Parallelism: EPAdapter with dispatch/combine cycle
- [x] Distributed: communicator, worker, launcher (stub interfaces for multi-rank transport)
- [x] Hardware Detection: Apple Silicon M1-M4 detection, 5-tier auto-tuning

### Foundation (12 crates)
- [x] `rmlx-serve-types` -- Shared types, OpenAI/Anthropic API models, error types, MCP types
- [x] `rmlx-serve-tokenizer` -- HF tokenizer, chat templates, streaming detokenizer (SPM/BPE/Naive)
- [x] `rmlx-serve-weights` -- Safetensors loading, config.json parsing, AWQ/GPTQ quantization, sharded load
- [x] `rmlx-serve-sampling` -- Full sampling pipeline (temperature, top-k/p, min-p, XTC, repetition penalty)
- [x] `rmlx-serve-models` -- LlmModel trait, TransformerLlm, dynamic architecture registry (16 model types)
- [x] `rmlx-serve-cache` -- KVCache, RotatingKVCache, QuantizedKVCache, BatchKVCache, PagedCacheManager, PrefixCacheManager, CacheOps trait
- [x] `rmlx-serve-scheduler` -- BatchGenerator + Scheduler (continuous batching, prefix cache integration)
- [x] `rmlx-serve-engine` -- SimpleEngine, BatchedEngine, EPAdapter, distributed stubs, hardware detection
- [x] `rmlx-serve-speculative` -- N-gram (incremental indexing), draft model, MTP proposer (interface), rejection sampler
- [x] `rmlx-serve-tools` -- 13 tool call parsers + 5 reasoning parsers (81 tests)
- [x] `rmlx-serve-api` -- axum HTTP server (OpenAI + Anthropic + MCP endpoints), rate limiting, structured output
- [x] `rmlx-serve-cli` -- CLI binary (serve, generate, bench), 50+ flags

---

## In Progress (Stubs / Partial)

These subsystems have interfaces and scaffolding but are not yet fully functional:

- **Multi-rank distributed transport** -- communicator, worker, and launcher exist but the actual transport layer (allreduce/allgather over network) is stubbed out
- **MTP head computation** -- MTP proposer interface is defined and wired to the scheduler, but no real MTP model heads compute speculative tokens yet
- **Batched prefill/decode** -- BatchedEngine exists but real batched prefill and decode require model API changes that are not yet implemented (TODO)
- **Expert parallelism** -- EPAdapter dispatch/combine cycle is implemented, but cross-device expert routing is not yet tested end-to-end with real MoE models

---

## Planned

### Multimodal (VLM)
- [ ] Vision-Language Model support (LLaVA, Qwen-VL, InternVL)
- [ ] Image preprocessing pipeline
- [ ] Video frame extraction
- [ ] VisionEmbeddingCache for multimodal prefix caching
- [ ] MLLMScheduler for multimodal batching
- [ ] Conv1d/Conv2d support in RMLX (required for vision encoders)

### Audio
- [ ] Text-to-Speech (Kokoro, Chatterbox, F5-TTS, MaskGCT, Dia, Spark)
- [ ] Speech-to-Text (Whisper, Parakeet)
- [ ] `/v1/audio/speech` and `/v1/audio/transcriptions` endpoints
- [ ] Audio streaming support

### Embedding Engine
- [ ] Embedding model support (separate from generative LLMs)
- [ ] `/v1/embeddings` endpoint
- [ ] Batch embedding with pooling strategies

### Performance Optimization
- [ ] Flash Attention 2 integration (when RMLX adds tiled Metal kernel)
- [ ] Paged attention kernel optimization
- [ ] Chunked prefill optimization (pipeline prefill + decode)
- [ ] Speculative decoding auto-tuning
- [ ] Memory pool optimization for KV cache
- [ ] Batch scheduling fairness improvements

### Additional Model Architectures
- [ ] GLM-4
- [ ] Nemotron
- [ ] OLMo
- [ ] GPT-NeoX / GPT-J
- [ ] Target: 30+ model types

### Advanced Serving
- [ ] Dynamic model loading/unloading
- [ ] Multi-model serving
- [ ] Request routing and load balancing
- [ ] Disk-based KV cache persistence
- [ ] GGUF format support

### Distributed (Full)
- [ ] Real multi-rank tensor parallelism via network transport
- [ ] Pipeline parallelism
- [ ] RDMA/Thunderbolt 5 transport optimization
- [ ] SSH-based distributed launcher (functional)

### Production Hardening
- [ ] Prometheus metrics export
- [ ] Structured logging (JSON)
- [ ] Graceful shutdown with request draining
- [ ] Health check improvements
- [ ] Request timeout configuration
- [ ] SSL/TLS support

---

## RMLX Dependencies Roadmap

Features needed from RMLX for full rmlx-serve capabilities:

| Feature | rmlx-serve Workaround | RMLX Target |
|---------|----------------------|-------------|
| allreduce/allgather | all_sum pattern in engine | RMLX collective ops |
| GELU activation | CPU fallback | Metal GELU kernel |
| Flash Attention 2 | SDPA (existing) | Metal tiled kernel |
| GGUF format | rmlx-serve-weights impl | RMLX native support |
| AWQ/GPTQ quantization | rmlx-serve-weights unpacking (implemented) | RMLX native support |
| FP8 support | N/A | DType extension |
| RotatingKVCache | Pure Rust in rmlx-serve-cache (implemented) | LayerKvCache circular mode |
| QuantizedKVCache (4/8-bit) | Pure Rust in rmlx-serve-cache (implemented) | Quantized KV support |
| BatchKVCache | Vec<LayerKvCache> wrapper (implemented) | Batch-aware cache |
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
