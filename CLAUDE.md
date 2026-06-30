# tempura-serve

Rust-native LLM inference server. Uses cuda-oxide (https://github.com/NVlabs/cuda-oxide)
for GPU kernels once stabilized; currently runs CPU reference implementations.

## Architecture

Three-crate workspace + standalone verification crate:

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

- **verus-verified/** (standalone, NOT in workspace) — Verus deductive proofs
  - Own `rust-toolchain.toml` (stable 1.96.0, Verus requirement)
  - Re-declares critical structures with `verus!{}` blocks and formal specs
  - Proves properties for ALL inputs (unbounded), complementing Kani's bounded checks
  - 14 indexing modules (mirror Kani): kv_cache, attention, rmsnorm, layernorm,
    rope, matmul, sampling, turboquant, quantized, model, speculative,
    disaggregated, weights, gguf
  - 36 invariant modules (beyond Kani): kv_cache_invariant, page_allocator,
    block_table, chunked_forward, tick_monotonicity, session_consistency,
    speculative_invariant, prefix_cache, batch_scheduler, model_config,
    radix_tree, token_generation, forward_pass, gqa, fused_kernels, medusa,
    projection_shapes, kv_append_advance, sampling_penalties, trace_invariant,
    gpu_tiles, paged_kv_cache, causal_mask, flash_tiles, fp8, weight_shapes,
    lora, trace_diff, guided_decoding, config_validation, rope_frequency,
    tensor_parallel, fp8_kv_cache, quantized_forward, gguf_alignment,
    position_encoding
  - 59 modules total, 757 verification conditions

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

4. **Four-layer verification architecture:**
   - **Kani** proves safety (no panics, no OOB access) on Rust code — bounded instances
   - **Verus** proves properties for ALL inputs (deductive, unbounded) via SMT
   - **DualPipeline** validates numerical equivalence between kernel implementations
   - **Property tests** cover mathematical properties Kani can't handle (f32 exp/trig)
   - **Shared trait contracts** (`Verified*` traits in kernel-api) bind real impls to
     Verus specs — `debug_assert!(well_formed())` in methods, drift detection in CI
   - **Inline `verus!{}` blocks** in engine source (kv_cache.rs, session.rs) — compiled
     to nothing by normal `cargo build` (vstd ghost erasure), canonical verification
     runs against `verus-verified/` crate
   - cuda-oxide and NVIDIA hardware are trusted infrastructure (like rustc/CPU)

5. **Inference traces are first-class.** Every forward pass produces a structured
   `InferenceTrace` at configurable granularity. The API exposes traces as queryable,
   diffable, replayable objects — not debug artifacts.

6. **Deterministic by design.** Seeded RNG for sampling, controlled reduction order
   in kernels. Same input + seed = bit-identical output. On GPU, cuda-oxide's
   Rust-level control enables deterministic reduction without post-hoc hacks.

7. **BitNet ternary models as a verification showcase.** Ternary weight operations
   ({-1,0,1} → add/sub/skip) are fully decidable by Kani — no floating-point
   approximation in the weight path. The bit-packing roundtrip is a textbook
   formal verification target. TQ2_0 format first (GPU-friendly 2-bit packing),
   TQ1_0 later (CPU-only, base-3 packing is SIMD-unfriendly).

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

### Phase 12: BitNet Ternary Model Support
- [ ] TQ2_0 block struct + decode in kernel-api
- [ ] BitLinear CPU kernel (per-tensor INT8 activation quant, integer add/sub)
- [ ] Squared ReLU activation function
- [ ] GGUF loader: detect `model_type: "bitnet"`, route to BitLinear layers
- [ ] TQ2_0 cuda-oxide GPU kernel (dp4a path, shared-memory tile unpack)
- [ ] TQ1_0 CPU-only decode
- [ ] Validation against microsoft/bitnet-b1.58-2B-4T-gguf

## Build & Test

```sh
cargo build              # build all crates
cargo test               # run all tests (148 currently)
cargo run -- --help      # see CLI options
cargo run -- --port 8080 # start server (stub responses)
```

## GPU Kernels (cuda-oxide)

We use a fork of cuda-oxide at `~/Code/github.com/NVLabs/cuda-oxide` with
mma.sync intrinsics and invariant.load codegen. **`cargo oxide build` uses a
cached backend `.so` from `~/.cargo/cuda-oxide/`, built from upstream — NOT
the fork.** You MUST point it at the local build:

```sh
# Build the codegen backend from our fork (once, or after fork changes)
cd ~/Code/github.com/NVLabs/cuda-oxide/crates/rustc-codegen-cuda
cargo build --release

# Set this in every shell that runs cargo oxide build
export CUDA_OXIDE_BACKEND=~/Code/github.com/NVLabs/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
```

Build and benchmark GPU kernels:

```sh
cd kernels && cargo oxide build                       # standalone kernel compile
cd tests-gpu && cargo oxide build bench-q4k-e2e       # Q4K e2e benchmark
cd tests-gpu && cargo oxide run --bin bench-q4k-e2e   # run benchmark
cd tests-gpu && cargo oxide run --bin tempura-tests-gpu  # full GPU test suite
```

If you see `Symbol cuda_device__mma_sync__... not found`, the backend is stale.
Rebuild it from the fork and set `CUDA_OXIDE_BACKEND`.

## Formal Verification

```sh
cargo kani --package tempura-kernels  # Kani: bounded model checking (57 harnesses)
./scripts/run-verus.sh               # Verus: deductive unbounded proofs (757 verification conditions)
```

Verus requires a pre-built binary at `~/.local/verus/verus-x86-linux/verus`
(or set `VERUS_PATH`). Download from https://github.com/verus-lang/verus/releases.
The `verus-verified/` crate is standalone (own toolchain, own workspace) — do NOT
add it to the main workspace.

Verus covers 56 of 57 Kani harnesses unboundedly (54 directly, 2 restated as
integer arithmetic: `q4k_monotonicity_within_group`, `q4k_accumulator_no_overflow`).
The 3 excluded require f32/f64 reasoning which Verus does not support; these are
covered by proptest instead: `head_dim_scale_never_zero_division` (engine),
`int8_dequant_bounded_error`, `q4k_dequant_value_bounded` (kernels).
The remaining `cpu_allreduce_is_identity` is a runtime Vec execution test — Kani-only.
