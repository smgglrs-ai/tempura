# tempura-serve

Rust-native LLM inference server, starting as a translation of llama.cpp, to
which we add formal proof via Kani, TLA+ and Verus. It is hardware agnostic,
but leverages Rust-native libraries. To ground formal proof, it builds and
proves the kernel with CPU reference implementions. Then, these kernels are
optimized using hardware-specific Rust-native features to get more performance:

  - CPU SIMD for Intel, AMD and ARM processors,
  - Neural Processing Unit (NPU) libraries,
  - NVIDIA cuda-oxide (https://github.com/NVlabs/cuda-oxide)

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
  - 62 modules total, 828 verification conditions

## Key Design Decisions

1. **Kernel traits as the verification contract.** Engine code is generic over kernel
   implementations. The SAME trait contract is implemented by CPU (for verification)
   and GPU via cuda-oxide (for production). Kani proves both adhere to the contract.

2. **NVIDIA GPUs via cuda-oxide are the production target.** CPU reference
   implementations exist for testing, Kani verification, and DualPipeline
   comparison — not as the production backend.

3. **cuda-oxide is the GPU production backend.** Two complementary paths:
   - **cuTile JIT** — cuTile tile-DSL kernels compiled by tileiras at runtime.
     Used for RMSNorm, RoPE, flash attention, elementwise ops, residual adds.
   - **rustc-codegen-cuda (oxide)** — hand-optimized GEMV kernels compiled with
     `cargo oxide build`, using warp-shuffle reductions and `shfl.sync.idx.b32`
     scale broadcasts matching llama.cpp's `vec_dot_q4_K_q8_1` design.
     Used for Q2_0/Q4K/Q6K decode GEMV (hot path). Feature flag: `oxide-gemv`.
     **Two backends**: fork (`crates/rustc-codegen-cuda`, nightly-2026-04-03, has mma.sync)
     and upstream (`cuda-oxide/crates/rustc-codegen-cuda`, nightly-2026-10-11, has dp4a).
     dp4a kernels require upstream backend: `RUSTUP_TOOLCHAIN=nightly-2026-10-11`.
   Both coexist: oxide kernels auto-loaded from PTX when present, cuTile fallback.

4. **Verification-first architecture — every property gets the strongest proof
   available.** New code without verification is incomplete code.
   - **Kani** proves safety (no panics, no OOB) on Rust code — bounded instances
   - **Verus** proves properties for ALL inputs (deductive, unbounded) via SMT —
     struct invariants, loop termination, state machines, GQA, causal masking,
     tensor parallel sharding, BitNet packing roundtrips, end-to-end composition
   - **Property tests** cover f32/f64 properties Kani/Verus can't handle
   - **DualPipeline** validates numerical equivalence between kernel implementations
   - **Shared trait contracts** (`Verified*` traits in kernel-api) bind real impls
     to Verus specs — `debug_assert!(well_formed())` in methods
   - **Inline `verus!{}` blocks** in engine source — compiled to nothing by normal
     `cargo build` (vstd ghost erasure), canonical proofs in `verus-verified/`
   - **Drift detection** in `run-verus.sh` — compares struct fields and function
     names between source and Verus re-declarations
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
- Feature flags: `cpu` (default), `cutile` (cuTile GPU kernels), `oxide-gemv` (hand-optimized oxide GEMV)
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
- [x] 540+ passing tests (CPU + GPU with cutile + oxide features)

## What's Done (Phase 13: GPU Inference)

- [x] cuTile GPU forward pass (CutileGpuModel) — all projection weights uploaded to GPU
- [x] GPU-resident KV cache with D→D append (no PCIe per layer per token)
- [x] GPU flash attention for decode (cutile_decode_attn — one warp per head)
- [x] Q2_0, Q4K, Q6K quantized weights GPU-resident (packed u32 format, dequant on-the-fly)
- [x] Full async pipeline (single stream.synchronize() per token)
- [x] Pre-computed RoPE frequency table (no cos/sin H2D per call)
- [x] Pre-allocated RMSNorm + RoPE scratch pools (no zeros alloc per layer)
- [x] CUDA graph infrastructure (DecodeGraph, cuGraphExecUpdate_v2)
- [x] Fused GPU RMSNorm (no D→H→D for scale computation)
- [x] On-device RoPE interleave kernel (cutile_interleave via MAP_SHAPE [1,BN])
- [x] oxide GEMV kernels (rustc-codegen-cuda): Q2_0, Q4K v2, Q6K v2
  - Q2_0: two-output-per-warp (half-warp XOR reduce), SMEM activation staging (Opt 2)
  - Q4K v2: multi-warp K-parallelism (4 warps per neuron, smem cross-warp reduce) (Opt 8)
  - Q4K q8: Q8_1 activation quantization + mini-warp dp4a (Opt 7, needs upstream backend)
  - Q6K v2: d broadcast via shfl, 8-step unrolled q_full
- [x] Kani harnesses + Verus proofs for Q2_0/Q4K/Q6K block dequant bounds
- [x] cuTile 0.4.0 upgrade (path deps to local fork); vec_width hint infrastructure
- [x] Benchmark: Q2_0 31 tok/s, Q4K 23-24 tok/s (gap: 6.7×/8.7× vs llama.cpp 209 tok/s)
  - Root cause of Q4K gap: compute path (llama.cpp uses dp4a = 4× throughput)
  - dp4a kernel infrastructure in place; next: add K-parallelism to dp4a kernel

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
- [x] TQ2_0 block struct + decode in kernel-api
- [x] BitLinear CPU kernel (per-tensor INT8 activation quant, integer add/sub)
- [x] Squared ReLU activation function
- [x] GGUF loader: detect `model_type: "bitnet"`, route to BitLinear layers
- [ ] TQ2_0 cuda-oxide GPU kernel (dp4a path, shared-memory tile unpack)
- [x] TQ1_0 CPU-only decode (I2S format via bitnet.cpp translation)
- [x] Validation against microsoft/bitnet-b1.58-2B-4T-gguf (decode: ~42 tok/s, prefill pp42: 362 tok/s)
- [x] AVX2/AVX512 maddubs with i16 batched accumulation (bitnet.cpp pattern)
- [x] Pre-quantized activation sharing across q/k/v and gate/up projections
- [x] Layer-level rayon::join parallelism for independent projections
- [x] LUT-based dot product infrastructure (scalar + AVX2 shuffle, verified correct)
- [x] Persistent thread pool (condvar + per-worker generation tracking)
- [x] forward_bitnet_decode: dedicated decode path with pre-allocated buffers
- [x] i2s_quantize_into: zero-allocation activation quantization
- [x] Persistent thread model for decode (run_indexed, no thread spawn per token)
- [x] Pre-allocated TL2 LUT buffers (zero heap allocs in layer loop)
- [x] Fused serial ops: RMSNorm+q8snap, residual+RMSNorm+q8snap (12→8 barriers/layer)
- [x] SIMD fused kernels: squared ReLU (AVX2), residual add (AVX2)
- [x] Zero-alloc i2s_quantize_into AVX2/AVX512 path (no Vec+memcpy per call)
- [x] Parallel attention across heads in decode (all threads, pre-allocated scores)
- [x] target-cpu=native via .cargo/config.toml (enables full AVX2/AVX512 codegen)
- [x] Const-specialized three_tbl_impl_bk96 (fully unrolled inner loops)
- [x] target_feature(enable="avx2") on tl2_gemv_tiles_cfg (enables inlining)
- [x] BM=320 for (640,2560) K/V projections (matches bitnet.cpp)
- [x] Work-stealing tile dispatch for all TL2 matmuls (atomic counter)
- [x] forward_bitnet_prefill: persistent-thread prefill with parallel serial work
- [x] Batched TL2 GEMM tiles (tl2_gemm_tiles_batched, zero-alloc per-thread)
- [x] Parallel per-token LUT preprocessing (tl2_preprocess_token_range)
- [x] attention_prefill_items: inline multi-token attention for persistent threads
- [x] Parallel per-token RMSNorm/q8snap/sqrelu/sub_norms/residual_add
- [x] Parallel RoPE + parallel KV cache copy (token-range distribution)
- [x] Fused attention + sub_norm + LUT (eliminate B3 barrier)
- [x] Token-parallel attention (all heads per token range, better cache locality)
- [x] Interleaved gate+up work-stealing (54 tiles vs 27 doubled)
- [x] Hybrid tile dispatch (work-stealing for QKV, static for O/gate+up/down)
- [x] SIMD vectorized rescale in tl2_gemm_tiles_batched
- [x] generate_bitnet_decode: multi-token persistent-thread decode loop (one run_indexed for all tokens)
- [x] AVX-VNNI I2S dot product (_mm256_dpbusd_epi32) for non-AVX-512 CPUs
- [x] Multi-row I2S GEMV with macro-unrolled 32-iteration inner loop
- [x] mmap GGUF loading with zero-copy I2S/F16 tensor access
- [x] f16 embedding lookup (no f32 dequant copy)
- [x] f16 KV cache with F16C SIMD conversion (vcvtps2ph/vcvtph_ps)
- [x] SIMD f16 KV attention kernel (dot_f32_f16, weighted_add_f16)
- [x] Lock-free gen_atomic thread dispatch counter
- [x] Prefill L1 cache tiling BS=24 (fits 40KB L1d)
- [x] TEMPURA_FORCE_I2S / TEMPURA_NO_TL2 runtime kernel selection
- [x] Parity with bitnet.cpp: 36 tok/s decode, 1.24GB memory (Lunar Lake laptop)

## Build & Test

```sh
cargo build              # build all crates
cargo test               # run all tests (337 currently)
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
# cuTile kernels (standard cargo build — JIT compiled at runtime)
cargo build -p tempura-bench-gpu --release --bin bench-real-model

# oxide hand-optimized GEMV kernels (require cargo oxide build)
# Build all three oxide GEMV crates:
for crate in oxide-gemv oxide-gemv-q4k oxide-gemv-q6k; do
  cd tools/$crate && cargo oxide build && cd ../..
done

# Copy PTX to build output directories:
for d in target/release/build/tempura-kernels-*/out; do
  cp tools/oxide-gemv/oxide_gemv.ptx       $d/oxide_q2_gemv.ptx  2>/dev/null
  cp tools/oxide-gemv-q4k/oxide_gemv_q4k.ptx $d/oxide_q4k_gemv.ptx 2>/dev/null
  cp tools/oxide-gemv-q6k/oxide_gemv_q6k.ptx $d/oxide_q6k_gemv.ptx 2>/dev/null
done

# Benchmark (oxide-gemv feature already in tools/bench-gpu/Cargo.toml):
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release \
  --bin bench-real-model -- /path/to/model.gguf
```

Current results (RTX 5090, see `bench-gpu-rtx5090.json`):
- Bonsai-1.7B Q2_0: **30 tok/s decode** (7× gap vs llama.cpp 209 tok/s)
- Llama-3.1-8B Q4K: **24 tok/s decode** (8.7× gap vs llama.cpp)

If you see `Symbol cuda_device__mma_sync__... not found`, the backend is stale.
Rebuild it from the fork and set `CUDA_OXIDE_BACKEND`.

## Formal Verification

**Every new feature, struct, or algorithm MUST be verified to the maximum extent
each tool supports.** This is not optional — formal proof coverage is a first-class
deliverable, not a follow-up task. When adding or modifying code:

1. **Add Kani harnesses** for bounded safety (no panics, no OOB) — any function
   with indexing arithmetic, buffer slicing, or integer overflow risk.
2. **Add Verus proofs** for unbounded deductive properties — struct invariants,
   loop termination, state machine transitions, resource conservation, and all
   Kani harnesses that can be generalized (pure integer/bit, no f32/f64).
3. **Add property tests** (proptest) for floating-point properties Kani/Verus
   can't handle — round-trip error bounds, monotonicity, NaN/Inf avoidance.
4. **Implement `Verified*` traits** for any new struct with a well-formedness
   invariant — add `debug_assert!(well_formed())` in mutating methods.
5. **Add inline `verus!{}` blocks** next to the real code when the proof mirrors
   the implementation closely — serves as documentation + compile-time syntax check.
6. **Update `run-verus.sh`** with any new Verus module.

### Running Verification

```sh
cargo test                           # unit + property tests (315 tests)
cargo kani --package tempura-kernels  # Kani: bounded model checking (61 harnesses)
./scripts/run-verus.sh               # Verus: deductive unbounded proofs (828 verification conditions)
```

### Tools

| Tool | What it proves | Scope | When to use |
|------|---------------|-------|-------------|
| **Kani** | Safety (no panics, no OOB, no overflow) | Bounded instances (e.g. max_seq_len≤8) | Any function with indexing or arithmetic |
| **Verus** | Properties for ALL inputs (unbounded) via SMT | Integer/bit arithmetic, struct invariants, state machines | Everything Kani proves + struct lifecycle + composition |
| **Property tests** | Probabilistic confidence for f32/f64 properties | Randomized (millions of inputs) | Float round-trip, monotonicity, NaN avoidance |
| **`Verified*` traits** | Impl-spec sync at compile time | Trait method signatures | Any struct with well-formedness invariant |
| **Inline `verus!{}`** | Documentation + syntax check | Same source file | Struct invariants, key algorithms |
| **Drift detection** | Struct field / function name sync | `run-verus.sh` script | Automatic on every Verus run |

### Coverage Summary

**Kani:** 61 bounded harnesses across kernels/ and engine/ — indexing arithmetic
for attention, RMSNorm, RoPE, matmul, sampling, turboquant, quantized (Q4K/Q6K/TQ2/I2S),
KV cache, model, speculative, disaggregated, weights, GGUF.

**Verus:** 62 modules, 828 verification conditions — 56 of 57 Kani harnesses
proven unboundedly (54 directly + 2 restated as integer arithmetic), plus 48
modules of struct invariants, loop proofs, state machines, GPU kernel
decomposition, GQA head mapping, tensor parallel sharding, causal masking,
flash attention tiles, FP8/f16 encoding, BitNet TQ2/I2S packing, end-to-end
pipeline composition, and weight shape validation.

**Property tests:** 3 float-specific harnesses (`head_dim_scale_never_zero_division`,
`int8_dequant_bounded_error`, `q4k_dequant_value_bounded`) that require IEEE 754
reasoning beyond Verus/Kani capability.

**Kani-only:** 1 runtime execution test (`cpu_allreduce_is_identity`) — requires
Vec allocation, not expressible in Verus.

**Shared trait contracts:** 5 `Verified*` traits in kernel-api — `VerifiedKvCache`,
`VerifiedPageAllocator`, `VerifiedBlockTable`, `VerifiedSession`, `VerifiedPrefixCache`.
Implemented on real structs with `debug_assert!(well_formed())`.

### Setup

Verus requires a pre-built binary at `~/.local/verus/verus-x86-linux/verus`
(or set `VERUS_PATH`). Download from https://github.com/verus-lang/verus/releases.
The `verus-verified/` crate is standalone (own toolchain, own workspace) — do NOT
add it to the main workspace.

Verus covers 60 of 61 Kani harnesses unboundedly. The 6 excluded from Verus:
- 2 restated as integer proofs: `q4k_monotonicity_within_group`, `q4k_accumulator_no_overflow`
- 3 covered by proptest (f32/f64): `head_dim_scale_never_zero_division`,
  `int8_dequant_bounded_error`, `q4k_dequant_value_bounded`
- 1 Kani-only (runtime Vec): `cpu_allreduce_is_identity`
