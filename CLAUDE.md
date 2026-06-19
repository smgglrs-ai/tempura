# tempura-serve

Rust-native LLM inference server. Uses cuda-oxide (https://github.com/NVlabs/cuda-oxide)
for GPU kernels once stabilized; currently runs CPU reference implementations.

## Architecture

Three-crate workspace:

- **kernels/** (`tempura-kernels`) — Kernel traits + CPU reference impls
  - `Attention`, `LayerNorm`, `RoPE`, `Sampler` traits in separate modules
  - CPU implementations behind `cpu` feature (default on)
  - `cuda` feature flag reserved for cuda-oxide backends
  - Tests use `approx` crate for floating-point comparison

- **engine/** (`tempura-engine`) — Inference engine
  - `ModelConfig` — architecture parameters (has presets: `llama2_7b()`, `tiny_test()`)
  - `KvCache` — pre-allocated key-value cache with slot management
  - `WeightLoader` trait + `SafetensorsLoader` implementation
  - `TransformerModel<A, L, R, S>` — generic over kernel implementations
  - `Tokenizer` — wrapper around HF `tokenizers` crate

- **server/** (`tempura-server`) — HTTP serving layer
  - Axum-based, OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
  - `BatchScheduler` — continuous batching (stub implementation)
  - SSE streaming types for token-by-token output
  - CLI via clap: `--host`, `--port`, `--model`, `--tokenizer`, `--max-batch-size`, `--max-seq-len`

## Key Design Decisions

1. **Kernel traits as the verification contract.** Engine code is generic over kernel
   implementations. The SAME trait contract is implemented by CPU (for verification)
   and GPU via cuda-oxide (for production). Kani proves both adhere to the contract.

2. **NVIDIA GPUs via cuda-oxide are the production target.** CPU reference
   implementations exist for testing, Kani verification, and DualPipeline
   comparison — not as the production backend.

3. **No cuda-oxide dependency yet.** It requires a custom rustc (`cargo oxide build`).
   The crate structure is ready behind a feature flag. Phase 2 builds the verified
   foundation; Phase 3 delivers GPU implementations.

4. **Three-layer verification architecture:**
   - **Kani** proves safety (no panics, no OOB access) on Rust code — CPU and GPU
   - **DualPipeline** validates numerical equivalence between kernel implementations
   - **Property tests** cover mathematical properties Kani can't handle (f32 exp/trig)
   - cuda-oxide and NVIDIA hardware are trusted infrastructure (like rustc/CPU)

5. **Inference traces are first-class.** Every forward pass produces a structured
   `InferenceTrace` at configurable granularity. The API exposes traces as queryable,
   diffable, replayable objects — not debug artifacts.

6. **Deterministic by design.** Seeded RNG for sampling, controlled reduction order
   in kernels. Same input + seed = bit-identical output. On GPU, cuda-oxide's
   Rust-level control enables deterministic reduction without post-hoc hacks.

## Conventions

- All commits must be signed off (`git commit -s`)
- Rust edition 2021, resolver v2
- No comments unless the WHY is non-obvious
- Tests go in `#[cfg(test)] mod tests` at bottom of each file
- Feature flags: `cpu` (default), `cuda` (future)
- NEVER RUN MULTIPLE INSTANCES AT THE SAME TIME. CHECK AVAILABLE MEMORY.

## What's Implemented

- [x] Workspace structure and all crate boundaries
- [x] Kernel traits with full signatures
- [x] CPU reference: attention (multi-head, GQA-ready)
- [x] CPU reference: layer normalization
- [x] CPU reference: rotary position embeddings
- [x] CPU reference: top-k / top-p / temperature sampling
- [x] KV-cache with append/advance/reset
- [x] Safetensors weight loader
- [x] Tokenizer wrapper
- [x] TransformerModel struct (forward pass is a stub)
- [x] OpenAI-compatible API endpoints (return stub responses)
- [x] Batch scheduler (FIFO, no preemption yet)
- [x] SSE streaming types
- [x] 7 passing tests

## What's Done (Phases 2–8)

- [x] All kernel traits + CPU reference implementations (Attention, RMSNorm, RoPE, MatMul, Sampler)
- [x] Kani formal verification harnesses for all kernels and KvCache
- [x] InferenceTrace type system with configurable granularity
- [x] Forward pass (LLaMA architecture) with trace observability
- [x] Token generation loop (autoregressive, deterministic)
- [x] cuda-oxide GPU kernels (Attention, RMSNorm, RoPE, MatMul) behind `cuda` feature
- [x] TurboQuant/PQO KV-cache compression kernel
- [x] Continuous batching with preemption, KV cache pool
- [x] Speculative decoding, tensor parallelism, quantization (GPTQ/AWQ/GGUF)
- [x] OpenAI-compatible API: chat completions (streaming + non-streaming), completions, models
- [x] Logprobs, stop sequences, N completions, presence/frequency penalties
- [x] Chat templates (Jinja2/HF), response_format (json_object/json_schema), tool/function calling
- [x] BatchScheduler wired into API handlers with backpressure
- [x] Chunked prefill, prefix caching (hash-based KV reuse)
- [x] Per-request timeouts, graceful shutdown, readiness probes
- [x] Prometheus metrics (TTFT, ITL, queue depth, KV usage), OpenTelemetry export
- [x] HuggingFace config.json auto-loading, Dockerfile
- [x] Inference traces, deterministic replay, DualPipeline comparison, batch-invariant verification
- [x] 148 passing tests

## What's Next

### Phase 9: API Completeness
- [ ] `stream_options.include_usage` in streaming responses
- [ ] Expose `seed` parameter on API requests
- [ ] Support `max_completion_tokens` alias
- [ ] Model-specific tool call parsers (hermes, mistral, llama3)
- [ ] Wire guided decoding into tool_choice modes

### Phase 10: Production Hardening
- [ ] Priority-based request scheduling (SLA differentiation)
- [ ] Black-box health check with priority inference (`/health/inference`)
- [ ] Prefix cache tenant isolation (`cache_salt`)

### Phase 11: Forward-Looking
- [ ] OpenAI Responses API (`/v1/responses`)
- [ ] Disaggregated prefill/decode serving

## Build & Test

```sh
cargo build              # build all crates
cargo test               # run all tests (148 currently)
cargo run -- --help      # see CLI options
cargo run -- --port 8080 # start server (stub responses)
```

## Formal Verification

```sh
cargo kani --package tempura-kernels  # verify kernel properties
```
