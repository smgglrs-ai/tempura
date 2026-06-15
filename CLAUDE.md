# oxide-serve

Rust-native LLM inference server. Uses cuda-oxide (https://github.com/NVlabs/cuda-oxide)
for GPU kernels once stabilized; currently runs CPU reference implementations.

## Architecture

Three-crate workspace:

- **kernels/** (`oxide-kernels`) — Kernel traits + CPU reference impls
  - `Attention`, `LayerNorm`, `RoPE`, `Sampler` traits in separate modules
  - CPU implementations behind `cpu` feature (default on)
  - `cuda` feature flag reserved for cuda-oxide backends
  - Tests use `approx` crate for floating-point comparison

- **engine/** (`oxide-engine`) — Inference engine
  - `ModelConfig` — architecture parameters (has presets: `llama2_7b()`, `tiny_test()`)
  - `KvCache` — pre-allocated key-value cache with slot management
  - `WeightLoader` trait + `SafetensorsLoader` implementation
  - `TransformerModel<A, L, R, S>` — generic over kernel implementations
  - `Tokenizer` — wrapper around HF `tokenizers` crate

- **server/** (`oxide-server`) — HTTP serving layer
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

## What's Next

### Phase 2: Verified, Observable, Composable Inference Foundation
#### 2A — Foundation (parallel)
- [ ] MatMul kernel trait + CPU reference
- [ ] RMSNorm kernel trait + CPU reference
- [ ] InferenceTrace type system (TraceLevel: Off/Metadata/Summary/Full)
- [ ] ModelWeights storage (replace `_weights_loaded` placeholder)
- [ ] Token selection in Sampler (seeded RNG for deterministic reproducibility)
- [ ] Kani harnesses for all CPU kernels (no-panic, bounds safety)
- [ ] Kani proofs for KvCache (append/slice bounds)

#### 2B — Forward pass (traced from birth)
- [ ] InferenceSession extraction (per-request state: cache, trace, RNG)
- [ ] KernelPipeline abstraction (5-trait bundle, trace-aware)
- [ ] Forward pass implementation (LLaMA arch, populates LayerSnapshot)
- [ ] Token generation loop (autoregressive, deterministic)
- [ ] Wire API to real inference
- [ ] SSE streaming

#### 2C — Differentiating API surface
- [ ] Trace retrieval API (`/oxide/v1/traces`)
- [ ] Verification status API (`/oxide/v1/verified`)
- [ ] Deterministic replay API (`/oxide/v1/replay`)
- [ ] DualPipeline comparison endpoint (`/oxide/v1/compare`)
- [ ] Kani in CI (`cargo kani` on every PR)

### Phase 3: cuda-oxide GPU Kernels (core product target)
- [ ] cuda-oxide toolchain setup
- [ ] GPU kernel implementations (CudaAttention, CudaRMSNorm, CudaRoPE, CudaMatMul)
  - Same Kani harnesses verify against same trait contract as CPU
  - Deterministic GPU reduction order by design
- [ ] DualPipeline CPU vs GPU validation
- [ ] TurboQuant/PQO KV-cache compression kernel (verified)

### Phase 4: Continuous Batching + Scheduling
- [ ] Multi-sequence KV cache (slot-based pool)
- [ ] Verified continuous batching with preemption (Kani proofs for fairness)

### Phase 5: Production hardening
- [ ] Speculative decoding
- [ ] Tensor parallelism
- [ ] Quantization support (GPTQ, AWQ, GGUF)
- [ ] Prometheus metrics
- [ ] Docker/container support

## Build & Test

```sh
cargo build              # build all crates
cargo test               # run all tests (7 currently)
cargo run -- --help      # see CLI options
cargo run -- --port 8080 # start server (stub responses)
```

## Formal Verification (future)

```sh
cargo kani --package oxide-kernels  # verify kernel properties
```
