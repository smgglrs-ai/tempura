# tempura-serve

Rust-native LLM inference server, started as a translation of llama.cpp and
now an independent engine. **Research project today; the ambition is a real
competitor to llama.cpp (local/CPU/edge) and vllm (served/GPU).** The
differentiator is verification-first architecture: every layer — from
quantization bit-packing to the batch scheduler — carries machine-checked
proofs (Kani, Verus, TLA+) alongside the production code.

Hardware-agnostic core, Rust-native acceleration:

  - CPU SIMD for Intel, AMD and ARM processors (x86 AVX2/AVX-512/F16C/VNNI
    optimized today; ARM NEON/SVE pending),
  - NPU libraries (stubs today: `npu-intel`, `npu-qualcomm` features),
  - NVIDIA via cuda-oxide (https://github.com/NVlabs/cuda-oxide).

Ternary models (BitNet-style, PrismML Bonsai) are a first-class target: the
weight path {-1,0,1} → add/sub/skip is fully decidable by Kani, making ternary
inference the flagship verification showcase.

**This file is the source of truth for agents. README.md and docs/ may lag —
when you add a crate, module, or capability, update this file in the same
commit.**

## Architecture

Twelve workspace crates (see root `Cargo.toml`), one standalone verification
crate, five standalone `cargo oxide` tool crates, plus TLA+ specs, C
cross-validation vectors, and a Zola docs site.

```
kernel-api ──── traits only, zero impls (the verification contract)
    │
kernels ─────── all kernel implementations (CPU ref + optimized + GPU)
    │             └─ crates/staticpar: static-partition thread pool
    │
kernel-verify ─ DualPipeline numerical comparison of kernel impls
    │
engine ──────── models, KV caches, loaders, traces, decoding strategies
    │
    ├─ model ─── HuggingFace hub download/inspect (CLI: pull/list/inspect)
    └─ server ── Axum HTTP: OpenAI-compatible API + tempura extensions

tools/ (workspace): trace-cli (TUI trace viewer), bench (HTTP throughput),
  bench-gpu (7 GPU benchmarks), compare (server-vs-server), integration
tools/ (standalone, cargo oxide): oxide-gemv (Q2_0), oxide-gemv-q4k,
  oxide-gemv-q6k, oxide-decode-attn, mxf4-gemv (cute-rs SM120a test)

verus-verified/ (standalone, own toolchain — NOT in workspace)
tla/ (TLC model checking), tests-ggml/ (C-generated reference vectors),
docs/ (Zola site, deployed by .github/workflows/pages.yml)
```

- **kernel-api/** (`tempura-kernel-api`, ~800 lines) — Kernel traits:
  `Attention`, `MatMul`, `MatMulQ`, `RMSNorm`, `LayerNorm`, `RoPE`, `Sampler`,
  `AllReduce`, `KvCacheCompressor`, `GuidedDecoder`, `DeltaNet`, graph-level
  `FusedTransformerLayer`/`CompiledPrefill` (NPU-shaped). Plus the shared
  verification contracts: `VerifiedKvCache`, `VerifiedPageAllocator`,
  `VerifiedBlockTable`, `VerifiedSession`, `VerifiedPrefixCache`,
  `VerifiedDeltaNetState` (well-formedness specs implemented by both real
  structs and the Verus crate) and `VerifiedEquivalent<Ref>` (dual-channel
  numerical equivalence: optimized kernel must cite the proof bounding its
  error vs. the reference).

- **kernels/** (`tempura-kernels`, ~94k lines) —
  - CPU reference: `attention.rs`, `rmsnorm.rs`, `rope.rs`, `matmul.rs`,
    `sampling.rs`, `layernorm.rs` — clean, provable versions.
  - `ggml/` (~50k lines): direct Rust translation of llama.cpp's CPU backend,
    file-by-file with C source line references (`ops_llama*.rs` ↔ ops.cpp,
    `repack_llama*.rs` ↔ repack.cpp, `sgemm_llama*.rs` ↔ llamafile tinyBLAS,
    `cpu_llama*.rs` ↔ ggml threadpool, `flash_attn_llama.rs`, `iq_tables.rs`).
    All ggml quant formats incl. IQ* importance-matrix quants.
  - `tl2.rs` (~3.2k lines): bitnet.cpp TL2 LUT-GEMM translation (ternary
    triplets, AVX2 shuffle lookup, i16 accumulation); I2S packing in
    `quantized.rs` (~10.3k lines, 95 public fns).
  - `mxfp4.rs`: native MXFP4 (E2M1 + UE8M0 power-of-2 block scales) quant.
  - GPU: `cutile_*.rs` (cuTile JIT kernels), `oxide_*_host.rs` (rustc-codegen
    -cuda PTX loaders), `q4k/q6k_*_ptx.rs` (hand-written sm_120a PTX),
    `gpu_capabilities.rs` (runtime feature dispatch), `kv_scatter_write.rs`.
  - `deltanet.rs`: gated delta rule (linear attention) recurrent + chunked.
  - `turboquant.rs`: KV-cache compression. `moe.rs`, `guided.rs`,
    `sampler_chain.rs`, `thread_pool.rs`, `inference_pool.rs`.

- **crates/staticpar/** (`staticpar`) — static-partition thread pool:
  spin-then-condvar hybrid (llama.cpp `ggml_graph_compute_poll_for_work`
  pattern), epoch GraphBarrier, raw-pointer graph dispatch. Used by kernels
  for decode GEMV parallelism.

- **kernel-verify/** (`tempura-kernel-verify`, ~2.3k lines) —
  `DualPipelineRunner`: runs two kernel implementations on identical inputs,
  grades error vs. tolerance into `Compiled`/`Tested`/`Proven` tiers, emits
  JSON `VerificationReport`s. Includes the `verified_equivalent_registry`
  test — the checklist that fails when a new optimized `MatMulQ` lacks a
  `VerifiedEquivalent` impl. GPU-vs-CPU comparisons under `cutile` feature.

- **engine/** (`tempura-engine`, ~31k lines) —
  - `config.rs`: `ModelConfig` maps 29 HF `model_type`s onto 6 architecture
    families (Llama — incl. qwen*/mistral/yi/bitnet/deepseek — GPT-2, Phi,
    Gemma, Falcon, MPT) with per-family quirks: Gemma offset-RMSNorm +
    embedding scale + logit softcapping, Phi parallel attn+FFN, MPT ALiBi,
    QK-norm, partial RoPE, sliding window, YaRN/NTK/LLaMA-3 rope scaling,
    MoE (n_experts), and hybrid `LayerKind::Attention|DeltaNet` layers.
  - `model.rs` (~11k lines): `TransformerModel<A,N,R,M,S>` forward paths —
    f32, quantized GGUF, BitNet persistent-thread decode/prefill (zero heap
    allocs in the layer loop). `CutileInferenceWrapper` for GPU.
  - `kv_cache.rs`: F32/F16 KV (F16C SIMD), `PagedKvCache` + `BlockTable` +
    `PageAllocator`, `Fp8PagedKvCache` (e4m3), `SharedPrefixCache` (hash
    prefix reuse), `KvCacheTransfer` for disaggregation.
  - `gguf_loader.rs`: full GGUF v3 parser, all ggml types + bitnet.cpp
    TQ1_0/TQ2_0/I2S + PrismML Q1_0/Q2_0 custom ternary (see below).
  - `speculative.rs` (target+draft), `multi_token.rs` (Medusa heads),
    `disaggregated.rs` (`PrefillWorker`/`DecodeWorker` traits +
    `SessionTransfer`), `dual_pipeline.rs`, `prefix_cache.rs`, `lora.rs`,
    `tool_call.rs` (hermes/mistral/llama3 parsers + template auto-detect),
    `tokenizer.rs` (HF tokenizers + GGUF-metadata extraction + chat
    templates), `trace.rs`/`trace_diff.rs` (first-class traces),
    `cutile_gpu_forward.rs` (~6.7k lines, CUDA-graph decode).

- **model/** (`tempura-model`) — `tempura-serve pull <repo-id>` HuggingFace
  download, `list`, `inspect` (format/arch detection).

- **server/** (`tempura-server`, ~4.2k lines) — Axum. OpenAI-compatible:
  `/v1/chat/completions` (SSE streaming, `stream_options.include_usage`,
  `seed`, `max_completion_tokens` alias, logprobs, n, penalties, tools,
  `response_format` json_object/json_schema, guided decoding wired into
  `tool_choice`), `/v1/completions`, `/v1/models`, `/v1/responses`,
  `/v1/embeddings`, `/tokenize`, `/detokenize`. tempura extensions:
  `/tempura/v1/traces` (query/delete), `/tempura/v1/replay` (deterministic
  replay), `/tempura/v1/compare` (DualPipeline), `/tempura/v1/verified`
  (verification status), `/tempura/v1/schema`, `/tempura/v1/debug/logits`.
  Ops: `/metrics` (Prometheus: TTFT, ITL, queue depth, KV usage), `/health`,
  `/health/inference` (black-box priority inference), `/ready`, `/props`,
  API-key auth (`--api-key` / `TEMPURA_API_KEY`), per-request timeouts,
  graceful shutdown. Inference runs on a dedicated engine thread
  (`engine_task.rs`, mpsc-fed) with 512-token chunked prefill;
  `batch.rs` `BatchScheduler` does slot allocation, priority preemption,
  preemption-requeue, prefix matching.

- **verus-verified/** (standalone, own `rust-toolchain.toml`, stable 1.96.0) —
  deductive proofs for ALL inputs. 80 modules, 588 proof functions. Re-declares
  critical structures in `verus!{}` blocks with specs; proves indexing safety,
  struct invariants, state machines, GQA mapping, causal masking, tensor
  parallel sharding, FP8/f16 encodings, BitNet TQ2/I2S packing roundtrips,
  MXFP4 bounds, end-to-end composition. Drift detection vs. source structs in
  `run-verus.sh`.

- **tla/** — TLA+ specs checked with TLC: `BatchScheduler.tla` (KV slot
  conservation, no double-schedule, no starvation liveness) and
  `SpeculativeDecoding.tla`.

- **tests-ggml/** — `gen_test_vectors.c` runs against real llama.cpp to
  produce reference binaries (Q4_0/Q4K/Q6K/IQ4XS/IQ4NL quantize/dequant/dot,
  RoPE cos/sin) consumed by Rust cross-validation tests via
  `run_crossval.sh`.

- **docs/** — Zola site with a full curriculum (math → hardware → kernel math
  → verification → serving); deployed by CI (`pages.yml`).

Scale: 173 first-party Rust files, ~142k lines (workspace crates + tools;
excludes all `target/` dirs with their generated cuda-bindings, and
`.claude/worktrees/` copies), plus 80 Verus proof modules (~13.6k lines).
853 commits on main. CI: `kani.yml` (Kani 0.67.0 on
every PR/push), `pages.yml`.

## Key Design Decisions

1. **Kernel traits as the verification contract.** Engine code is generic over
   kernel implementations. The SAME trait contract is implemented by CPU (for
   verification) and GPU via cuda-oxide (for production). Kani proves both
   adhere to the contract.

2. **NVIDIA GPUs via cuda-oxide are the production target.** CPU reference
   implementations exist for testing, Kani verification, and DualPipeline
   comparison — not as the production backend. Two complementary GPU paths:
   - **cuTile JIT** — tile-DSL kernels compiled at runtime: RMSNorm, RoPE,
     flash attention, elementwise, quantized GEMM, MXFP4 tensor-core GEMM.
   - **rustc-codegen-cuda (oxide)** — hand-optimized kernels compiled with
     `cargo oxide build`: Q2_0/Q4K/Q6K decode GEMV (warp-shuffle reductions,
     dp4a), decode attention. Feature flag: `oxide-gemv`.
     **Two backends**: fork (`crates/rustc-codegen-cuda`, nightly-2026-04-03,
     has mma.sync) and upstream (`cuda-oxide/crates/rustc-codegen-cuda`,
     nightly-2026-10-11, has dp4a). dp4a kernels require the upstream
     backend: `RUSTUP_TOOLCHAIN=nightly-2026-10-11`.
   Both coexist: oxide kernels auto-loaded from PTX when present, cuTile
   fallback. A third path: hand-written **sm_120a PTX** modules for Blackwell
   (dp4a/mma.sync/cp.async mbar kernels, stream-K batch GEMM).

3. **Verification-first architecture — every property gets the strongest proof
   available.** New code without verification is incomplete code.
   - **Kani** proves safety (no panics, no OOB) on Rust code — bounded instances
   - **Verus** proves properties for ALL inputs (deductive, unbounded) via SMT
   - **TLA+** model-checks state machines (scheduler, speculative decoding)
   - **Property tests** cover f32/f64 properties Kani/Verus can't handle
   - **DualPipeline** validates numerical equivalence between implementations
   - **Shared trait contracts** (`Verified*` in kernel-api) bind real impls
     to Verus specs — `debug_assert!(well_formed())` in mutating methods
   - **Inline `verus!{}` blocks** in engine source — compiled to nothing by
     normal `cargo build` (vstd ghost erasure), canonical proofs in
     `verus-verified/`
   - **Drift detection** in `run-verus.sh` — compares struct fields and
     function names between source and Verus re-declarations
   - **Cross-validation** against llama.cpp itself via C-generated vectors
     (`tests-ggml/`)
   - cuda-oxide and NVIDIA hardware are trusted infrastructure (like rustc/CPU)

4. **Inference traces are first-class.** Every forward pass produces a
   structured `InferenceTrace` at configurable granularity. The API exposes
   traces as queryable, diffable, replayable objects — not debug artifacts.
   `tools/trace-cli` is a TUI viewer/differ/exporter for them.

5. **Deterministic by design.** Seeded RNG for sampling, controlled reduction
   order in kernels. Same input + seed = bit-identical output. On GPU,
   cuda-oxide's Rust-level control enables deterministic reduction without
   post-hoc hacks.

6. **Ternary models are both a verification showcase and a deployment
   target.** Ternary weight ops ({-1,0,1} → add/sub/skip) are fully decidable
   by Kani — no floating-point approximation in the weight path. The bit-packing
   roundtrip is a textbook formal verification target. Formats: TQ2_0
   (GPU-friendly 2-bit), TQ1_0 (CPU, base-3 SIMD-unfriendly), I2S/TL2
   (bitnet.cpp LUT formats), PrismML Q2_0/Q1_0 (see below).

7. **The llama.cpp translation is the reference, not the product.** `ggml/`
   mirrors upstream file-by-file (doc comments cite C line numbers) so
   behavior can be diffed against upstream and cross-validated with
   C-generated vectors. Optimized paths then diverge deliberately (staticpar
   pool, persistent-thread BitNet decode, fused kernels) with DualPipeline
   proofs covering the divergence.

## PrismML Bonsai support

PrismML (https://prismml.com) publishes Bonsai ternary models — 27B-class
multimodal models at 1.76 effective bits/weight (~5.9GB), ternary {-1,0,+1}
weights with FP16 group-wise scaling (Bonsai 2 27B, Sept 2026, based on
Qwen3.8 27B, Apache-2.0). Bonsai runs on phones/laptops where **KV cache is
the memory bottleneck** — this motivated the hybrid DeltaNet work.

tempura supports the Bonsai GGUF format via custom type ids in
`engine/src/gguf_loader.rs`:
  - `Q2_0 = 42` — PrismML ternary, QK=128, 34 bytes/block (2-byte f16 scale +
    32 bytes packed 2-bit). **NOTE: upstream GGUF uses 39 for MXFP4; we use
    39 for bitnet.cpp I2S. Do not "fix" these ids without checking loader
    + producers.**
  - `Q1_0 = 41` — companion 1-bit format.
CPU vec-dot in `kernels/src/quantized.rs` (`vec_dot_q2_0_q8_0`, PrismML fork
lineage), GPU GEMV in `cutile_quantized_gemm.rs` + `oxide_q2_gemv_host.rs`.

The hybrid attention+DeltaNet pipeline (Qwen3.6-style: in_proj → conv1d →
SiLU → L2-norm → beta/alpha → recurrence → ssm_norm → z-gating → out_proj)
was built for these models. **Not yet validated against real Bonsai 2 /
Qwen3.6-27B weights** — see open work.

## Conventions

- All commits must be signed off (`git commit -s`)
- Rust edition 2021, resolver v2 (standalone oxide tool crates may use 2024)
- No comments unless the WHY is non-obvious
- Tests go in `#[cfg(test)] mod tests` at bottom of each file
- Feature flags:
  - kernels: `cpu` (default), `cuda` (alias of `cutile`), `cutile`,
    `oxide-gemv`, `npu-intel`, `npu-qualcomm`, `openmp`
  - engine: `cuda`, `cutile`, `oxide-gemv`, `edge-metrics`, `openmp`
  - server: `cuda`, `cutile`, `otlp`, `openmp`
- NEVER RUN MULTIPLE INSTANCES AT THE SAME TIME. CHECK AVAILABLE MEMORY.
- Update this file when adding crates/modules/capabilities — CLAUDE.md is a
  symlink to it, so one edit covers both.

## Implemented capabilities

**Models & loading**
- [x] GGUF v3 (all ggml quant types incl. IQ2*/IQ3*/IQ4*, TQ1_0/TQ2_0, I2S,
      PrismML Q1_0/Q2_0, BF16; mmap zero-copy I2S/F16 access)
- [x] Safetensors (F32/BF16→MXFP4 tiling), HF config.json auto-detection
- [x] Tokenizer: HF tokenizer.json, GGUF-metadata extraction, chat templates
      (Jinja2/HF), special-token registration
- [x] 6 architecture families, 29 model_types mapped (see config.rs)
- [x] Hybrid Attention/DeltaNet layers (gated delta rule, recurrent + chunk)
- [x] MoE (top-k routing, Qwen-style weight naming)
- [x] BitNet b1.58 validated end-to-end vs microsoft/bitnet-b1.58-2B-4T-gguf

**Serving**
- [x] OpenAI-compatible chat/completions/completions/models/responses/
      embeddings/tokenize/detokenize; SSE streaming with include_usage
- [x] Logprobs, top-logprobs, stop sequences, n completions, presence/
      frequency penalties, seed, max_completion_tokens alias
- [x] Tool/function calling (hermes/mistral/llama3 parsers + auto-detect),
      response_format (json_object/json_schema), guided decoding wired into
      tool_choice
- [x] Continuous batching with priority preemption + preemption-requeue,
      chunked prefill (512-token), prefix caching with cache_salt tenant
      isolation
- [x] Dedicated engine thread (async HTTP ↔ sync SIMD isolation)
- [x] Prometheus metrics, OpenTelemetry export (`otlp` feature), readiness,
      black-box /health/inference, API-key auth, timeouts, graceful shutdown
- [x] Deterministic replay via traces

**CPU inference**
- [x] llama.cpp ggml CPU backend translation (ops, repack, tinyBLAS sgemm,
      threadpool, flash attention, all quants) with C-vector cross-validation
- [x] BitNet decode/prefill persistent-thread paths: TL2 LUT GEMM (AVX2
      shuffle + AVX-VNNI dpbusd), I2S, fused RMSNorm+q8snap / residual+
      RMSNorm+q8snap, squared-ReLU, f16 KV cache (F16C), work-stealing tiles,
      L1-sized chunking; parity with bitnet.cpp (36 tok/s decode, 1.24GB on
      Lunar Lake; pp42 up to 2× bitnet.cpp)
- [x] staticpar hybrid spin/condvar pool, OpenMP GEMV variant (`openmp`)
- [x] BOLT post-link optimization (scripts/build-bolt.sh): +4% pp128
      (203→211 tok/s); Rust PIE forces BOLT lite mode — that is the ceiling
- [x] PGO workflow (scripts/build-pgo.sh); thin LTO only (fat LTO SIGSEGVs
      unsafe SIMD — see root Cargo.toml)

**GPU inference (RTX 5090, Llama-3.1-8B Q4_K_M)**
- [x] cuTile forward pass, GPU-resident KV (D→D append), fused RMSNorm,
      on-device RoPE, batched flash attention
- [x] oxide GEMV: Q2_0, Q4K (dp4a), Q6K (dp4a kpar8; the sm120 mbar variant
      is default-off pending the race fix, see open work); oxide decode attn
- [x] sm_120a native PTX: q4k/q6k GEMV, fused gate+up SiLU, mma.sync batch
      GEMM, stream-K batch GEMM (persistent blocks, grid=SM_COUNT)
- [x] CUDA graphs for single + batch decode (M=2..16+, dynamic
      max_batch_graph from VRAM), KV scatter-write (offset read on device —
      no per-step graph updates)
- [x] Native MXFP4 pipeline: E2M1+UE8M0 quant (CPU), on-GPU activation
      quant, tensor-core GEMM, all 7 projections, BF16 loader; 55.6 TFLOPS
      at M=64
- [x] Prefill fast path (2026-10-27): m>1 GEMMs run on-GPU -- tiled Q4K
      projections via MXFP4 quant + mxf4_batch_gemm_tiled in 16-row chunks,
      Q6K tensors via the sm120 M-row batch GEMV chunked over rows
      (capped_batch_chunks); host-dequant fallback remains for incompatible
      strides. 394 tok/s pp128 (was 7 host-bound), TTFT P50 324 ms, gap vs
      llama.cpp 7335 down to 18.6x. Verified: NaN-free, fresh-session logits
      bitwise-replay (bench-real-model fingerprints + TEMPURA_BENCH_PREFILL_AB
      numerics delta vs exact-dequant); batch-decode fingerprints unchanged.
      TEMPURA_NO_GPU_PREFILL=1 bisects back to the host path. Remaining
      prefill levers: attention still host-staged (Q download + host-mirror KV
      + re-upload per layer), GEMM chunked at m=16 pending the cutlass-mlir
      codegen fix (see open work).
- [x] Benchmarks (2026-10-27, post-fix, RTX 5090, bench-batch all-M
      verified with NO bisect switches — NaN-free + fresh-cache replay
      bitwise): single-seq 134 tok/s decode, ITL 7.44 ms; prefill 394 tok/s
      pp128, TTFT P50 324 ms, logits fingerprint det=OK (bench-real-model);
      batch M=1 135, M=2 93, M=4 164, M=5 190, M=8 259, M=16 374 tok/s
      (llama.cpp single-seq reference: 209; pp128 reference: 7335).
      All fingerprints are post-fix values: pre-fix outputs were computed
      with frozen-position RoPE and the wrong FFN (see open work) — old
      numbers retracted, these are the first fully-verified batch numbers.
      Batch decode uses the generalized M-row q6k batch kernel (M>=2) and
      MXFP4 tensor-core Q/K/O/gate/up with m_valid-masked activation quant.
- [x] Decode correctness (2026-10-27): fixed sm120 fused-silu format mismatch
      (fed native 36-u32 weights to a repacked-42-u32 kernel -> all-NaN logits
      since introduction; every prior bench computed garbage at full speed) and
      the cp.async cross-lane visibility race in the q6k sm120 kernels
      (wait_group is per-thread; needs bar.sync 0 after it, mbar variant now
      gated on k_blocks % 8 == 0). bench-real-model prints a NaN-tolerant
      logits fingerprint; TEMPURA_DEBUG_NAN=1 gives per-op NaN bisection;
      TEMPURA_NO_SM120_KPAR / TEMPURA_NO_SM120_Q6K / TEMPURA_NO_OXIDE_ATTN /
      TEMPURA_NO_DUAL_BLOCK / TEMPURA_NO_MXFP4 / TEMPURA_NO_FUSED_SILU are
      kernel-path bisect switches (TEMPURA_ALLOW_SM120_Q6K=1 opts back into
      the disabled-by-default mbar GEMV). Divergence-hunting tooling added
      2026-10-27: per-op bits_sum in the nan-trace, TEMPURA_DEBUG_SLICES
      (per-1k-element slice sums), bench-batch TEMPURA_BENCH_STEP_FP /
      TEMPURA_BENCH_ONLY_M / TEMPURA_BENCH_DUMP (element-wise logit diff at
      the first diverging step) — this is how the silu stream race, the
      gate/up aliasing, and the mbar drift were localized.
      Second fix round (batch re-verification, same day): the M=2-hardcoded
      q6k sm120 batch kernel was rewritten as a generalized M-row two-phase
      kernel; MXFP4 activation quant got m_valid masking (stale m_pad rows
      poisoned the m16k32 tile scale); m=1 batch decode routes to the
      verified single-seq path; graph-replayed host launch scalars (kv_len,
      rope pos) were replaced by device-scalar indirection (d2d_gather_row
      + oxide_decode_attn reading kv_len from device) — frozen-position
      RoPE, a bug class determinism+NaN checks CANNOT catch; silu_mul runs
      on the caller's stream (private-stream race); kpar GEMVs write
      per-slot outputs (gate/up aliased the shared q8_out_scratch ->
      wrong FFN). Post-fix fishtank re-verification, NO bisect switches,
      run twice (incl. after the avx512 rustflags fix): every bench-batch M
      (1,2,4,5,8,16) NaN-free with bitwise-identical fresh-cache replay;
      single-seq 134 tok/s.

**Decoding strategies**: speculative (target+draft), Medusa multi-token,
guided (json/json_schema), sampler chain (temperature/top-k/top-p/min-p/
typical-p/repeat/presence/frequency)

## Open work (verified gaps — do not assume these are done)

- [ ] q6k_gemv_sm120_mbar_kpar8 (single-seq Q6K GEMV): outputs are
      allocation/kernel-history dependent. CONFIRMED 2026-10-27 (was
      "suspected" in the original batch re-measure): with every other fix
      in place, the M=1 fresh-cache replay diverges at L0 down-proj with
      BITWISE-IDENTICAL inputs (gate-silu matched) -- ULP-level output
      drift only. TEMPURA_NO_SM120_Q6K=1 (dp4a kpar8) flips the entire
      batch matrix to verified. The mbar variant is now DEFAULT OFF
      (opt in with TEMPURA_ALLOW_SM120_Q6K=1); the sm120 BATCH kernel is
      unaffected (different code path, replay-verified) and loads
      independently. Suspect: async-proxy -> generic-proxy ordering of
      the double-buffer swap on sm_120 (bar.sync after wait_group is the
      canonical pattern but insufficient here); needs a memory-model
      audit or an mbarrier-based rewrite.
- [ ] Close the 1.56× single-sequence GPU decode gap vs llama.cpp
      (134 vs 209 tok/s, post-fix, verified). Known levers: port the
      sm120 fused gate+up+SiLU kernel to native 36-u32 format (currently
      bypassed by the format gate), GEMV still ~40% DRAM (dmon: sm
      99-100%, mem 36-46%), host-side per-token overhead
- [ ] Prefill on GPU: 394 tok/s pp128, TTFT P50 324 ms (2026-10-27 fast
      path, verified) vs llama.cpp 7335 (18.6x gap). Remaining levers:
      attention is host-staged (per layer: Q D->H download, K/V read from the
      host mirror, attn output H->D re-upload -- forward_batched host API),
      KV host-mirror updates sync the stream twice per layer, and the MXFP4
      GEMMs are chunked at m=16 per launch pending the codegen fix below
- [ ] cutlass-mlir codegen bug (cuda-oxide, upstream): mxf4_batch_gemm_tiled
      miscompiles at larger grids. Two manifestations found 2026-10-27 via the
      prefill det=MISMATCH hunt: (a) the original persistent tile-walk
      (flat index + loop carry) left ~half the output tiles unwritten and
      stored garbage from mis-carried registers once a warp iterated more
      than once (m >= 32); (b) even a trivial one-tile-per-warp 2D-grid
      rewrite makes ILLEGAL memory accesses at m_tiles >= 8 (standalone shape
      bisection: (16,4096,4096) and (32,4096,4096) pass, (128,4096,64)
      faults with CUDA error 700; all source-level indexing is exactly
      in-bounds). Workaround shipped: MxfpGemm::forward_offset chunks prefill
      GEMMs at m=16 (the bitwise-verified batch-decode configuration; the
      standalone binary in tools/mxf4-gemv carries the shape-bisection
      regression tests). Needs an upstream issue + fix; a hand-written-PTX
      GEMM (like the quant kernel, which is shape-safe) is the local escape.
- [ ] Validate hybrid DeltaNet pipeline against real Qwen3.6 / Bonsai 2
      weights (last attempt: "architecture incompatible" — diagnostics added,
      commit 9c6362d)
- [ ] ARM NEON/SVE optimized CPU kernels (optimized paths are x86-only today)
- [ ] NPU backends: `npu-intel`/`npu-qualcomm` are todo!() stubs (OpenVINO /
      QNN SDK required)
- [ ] Disaggregated serving over network — traits + local worker impl only
- [ ] NVFP4: enum + block size only, no dequant/kernels
- [ ] Multimodal input (Bonsai 2 is text+image; no vision encoder path)
- [ ] TQ1_0 GPU kernel (CPU-only today)

## Build & Test

```sh
cargo build              # build all workspace crates
                         # (.cargo/config.toml uses target-cpu=native only --
                         # never force +avx512* features: hosts without
                         # AVX-512 SIGILL, hosts with it get the full set
                         # from native anyway)
cargo test               # 907 test functions (26 ignored)
cargo kani --package tempura-kernels   # 240 Kani harnesses (kernels+engine)
./scripts/run-verus.sh   # 588 Verus proof functions, 80 modules
cargo run -- --help      # CLI: serve, pull, list, inspect
cargo run -- serve --model /path/to/model.gguf --port 8080
```

Note: `cargo test` compiles `tempura-kernels` with the `oxide-gemv` feature
(feature unification via tools/bench-gpu), so the oxide GEMV PTX must be
present in the kernels build out dirs — `scripts/gpu-remote.sh oxide` or the
manual copy loop in the GPU section provides them.

Benchmarks: `tools/bench` (HTTP throughput, llama.cpp-bench-style pp/tg),
`tools/bench-gpu` (bench-real-model, bench-batch, bench-mxfp4, bench-e2e,
bench-attn, bench-kernels, bench-speculative), `tools/compare` (A/B two
servers), `engine/benches` (custom `harness = false` timing benches, not
criterion). Results in `bench-*.json` at root.

Two-machine workflow: the RTX 5090 lives on `fishtank.home.local`
(passwordless SSH); development happens on the laptop. `scripts/gpu-remote.sh`
drives the remote GPU host (any host: `GPU_REMOTE_HOST`): `status`
(pre-flight — enforces the never-run-multiple-instances rule), `sync [--wip]`
(source moves through git origin only; never `target/`, `.oxide-artifacts/`
or models; `--wip` pushes a snapshot of uncommitted work as `wip/<hostname>`),
`bench [--no-sync] BIN` (sync + tmux job + PTX re-copy after build), `run`,
`oxide`, `watch`/`log`/`results`/`stop`. Remote run scripts and logs live in
`~/.cache/tempura/gpu-remote/`, models in `~/.cache/tempura/models/` (a bare
model filename passed to `bench` resolves there). Env overrides:
`GPU_REMOTE_HOST`, `GPU_REMOTE_SESSION`, `GPU_REMOTE_BUSY_MB`/`_BUSY_UTIL`
(pre-flight thresholds), `GPU_REMOTE_ALLOW_PROCS=1` (skip the
no-other-instances check when idle CPU-mode servers live on the host). Full
usage: run the script with no arguments.

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

The UPSTREAM backend (`cuda-oxide/cuda-oxide/crates/rustc-codegen-cuda`) is
required for oxide-decode-attn (ex2_approx_f32) and the dp4a crates. Its
current build links the **1.99.1** toolchain's rustc driver (the upstream
tree pins channel "1.99.1"), and `cargo oxide` passes `-Z codegen-backend`,
which stable rejects — so:

```sh
export CUDA_OXIDE_BACKEND=~/Code/github.com/NVLabs/cuda-oxide/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
export LD_LIBRARY_PATH=~/.rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu/lib:$LD_LIBRARY_PATH
RUSTC_BOOTSTRAP=1 RUSTUP_TOOLCHAIN=1.99.1 cargo oxide build
# If the backend .so is rebuilt from a different toolchain, match
# LD_LIBRARY_PATH/RUSTUP_TOOLCHAIN to that toolchain (the .so dlopens a
# versioned librustc_driver-*.so).
```

Build and benchmark GPU kernels:

```sh
# cuTile kernels (standard cargo build — JIT compiled at runtime)
cargo build -p tempura-bench-gpu --release --bin bench-real-model

# oxide hand-optimized GEMV kernels (require cargo oxide build)
# Build all oxide GEMV crates:
for crate in oxide-gemv oxide-gemv-q4k oxide-gemv-q6k oxide-decode-attn; do
  cd tools/$crate && cargo oxide build && cd ../..
done

# Copy PTX to build output directories:
for d in target/release/build/tempura-kernels-*/out; do
  cp tools/oxide-gemv/oxide_gemv.ptx       $d/oxide_q2_gemv.ptx  2>/dev/null
  cp tools/oxide-gemv-q4k/oxide_gemv_q4k.ptx $d/oxide_q4k_gemv.ptx 2>/dev/null
  cp tools/oxide-gemv-q6k/oxide_gemv_q6k.ptx $d/oxide_q6k_gemv.ptx 2>/dev/null
  cp tools/oxide-decode-attn/oxide_decode_attn.ptx $d/oxide_decode_attn.ptx 2>/dev/null
done

# From the laptop, scripts/gpu-remote.sh runs all of this on the GPU host:
#   scripts/gpu-remote.sh oxide
#   scripts/gpu-remote.sh bench real-model Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Benchmark (oxide-gemv feature already in tools/bench-gpu/Cargo.toml):
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release \
  --bin bench-real-model -- /path/to/model.gguf
```

Current results (RTX 5090, see `bench-gpu-rtx5090.json`):
- Llama-3.1-8B Q4K single-sequence: **134 tok/s decode**, ITL 7.44 ms
      (1.56x gap vs llama.cpp 209 tok/s; outputs verified NaN-free +
      deterministic, 2026-10-27 post-fix)
- Llama-3.1-8B Q4K batch decode (2026-10-27 post-fix, NO bisect switches;
      verified = NaN scan every step + final-step fingerprints + fresh-cache
      replay bitwise identical; measured twice, incl. after the avx512
      rustflags fix):
      **M=1 134-135, M=2 92-93, M=4 164, M=5 189-190, M=8 259-260,
      M=16 372-374 tok/s** — every bench-batch M verified.
      M>=2 uses the generalized M-row q6k sm120 batch kernel + MXFP4
      tensor-core Q/K/O/gate/up (m_valid-masked activation quant); M=1
      routes to the single-seq CUDA graph path.
      All pre-2026-10-27 batch numbers are retracted (frozen-position RoPE,
      wrong FFN, all-NaN fused silu — see Decode correctness above).
- MXFP4 M=64: 55.6 TFLOPS

If you see `Symbol cuda_device__mma_sync__... not found`, the backend is stale.
Rebuild it from the fork and set `CUDA_OXIDE_BACKEND`.

## Formal Verification

**Every new feature, struct, or algorithm MUST be verified to the maximum
extent each tool supports.** This is not optional — formal proof coverage is
a first-class deliverable, not a follow-up task. When adding or modifying
code:

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
6. **Add a `VerifiedEquivalent` impl** for any new optimized kernel and a row
   in the `verified_equivalent_registry` test (kernel-verify).
7. **Update `run-verus.sh`** with any new Verus module.

### Running Verification

```sh
cargo test                            # unit + property tests (907 tests)
cargo kani --package tempura-kernels  # Kani: bounded model checking (240 harnesses)
./scripts/run-verus.sh               # Verus: deductive unbounded proofs (588 proof fns, 80 modules)
```

Kani 0.67.0 runs in CI on every PR and push (`.github/workflows/kani.yml`).

### Tools

| Tool | What it proves | Scope | When to use |
|------|---------------|-------|-------------|
| **Kani** | Safety (no panics, no OOB, no overflow) | Bounded instances (e.g. max_seq_len≤8) | Any function with indexing or arithmetic |
| **Verus** | Properties for ALL inputs (unbounded) via SMT | Integer/bit arithmetic, struct invariants, state machines | Everything Kani proves + struct lifecycle + composition |
| **TLA+** | State machine invariants + liveness (TLC) | BatchScheduler, speculative decoding | Multi-step protocol/state logic |
| **Property tests** | Probabilistic confidence for f32/f64 properties | Randomized (millions of inputs) | Float round-trip, monotonicity, NaN avoidance |
| **DualPipeline** | Numerical equivalence of two implementations | Kernel pairs, layer-by-layer, GPU vs CPU | Any optimized kernel vs its reference |
| **`Verified*` traits** | Impl-spec sync at compile time | Trait method signatures | Any struct with well-formedness invariant |
| **Inline `verus!{}`** | Documentation + syntax check | Same source file | Struct invariants, key algorithms |
| **Drift detection** | Struct field / function name sync | `run-verus.sh` script | Automatic on every Verus run |
| **C cross-validation** | Bit-exact match vs llama.cpp output | `tests-ggml/` vectors | ggml translation fidelity |

### Coverage Summary (2026-10-26, machine-counted)

**Kani:** 240 `#[kani::proof]` harnesses across kernels/ and engine/ —
attention, RMSNorm, RoPE, matmul, sampling, turboquant, quantized
(Q4K/Q5K/Q6K/Q2_0/TQ2/I2S/MXFP4 E2M1/UE8M0), KV cache (plain/paged/fp8),
model, speculative, disaggregated, weights, GGUF, GPU dispatch (grid dims,
buffer sizes, batch graph eligibility), tl2 LUT, deltanet, MoE, fused,
guided, ggml ops/repack/vec_dot/dequantize/quantize.

**Verus:** 80 modules, 588 proof functions — indexing safety for all Kani
harnesses that generalize (pure integer/bit), plus struct invariants, loop
proofs, state machines, GPU kernel decomposition, GQA head mapping, tensor
parallel sharding, causal masking, flash attention tiles, FP8/f16 encoding,
BitNet TQ2/I2S packing, MXFP4 bounds, end-to-end pipeline composition, and
weight shape validation.

**Property tests:** float-specific harnesses (e.g. `head_dim_scale_never_zero_division`,
`int8_dequant_bounded_error`, `q4k_dequant_value_bounded`) that require IEEE 754
reasoning beyond Verus/Kani capability.

**Shared trait contracts:** 6 `Verified*` traits in kernel-api — `VerifiedKvCache`,
`VerifiedPageAllocator`, `VerifiedBlockTable`, `VerifiedSession`,
`VerifiedPrefixCache`, `VerifiedDeltaNetState`. Implemented on real structs
with `debug_assert!(well_formed())`.

**Kani-only:** 1 runtime execution test (`cpu_allreduce_is_identity`) — requires
Vec allocation, not expressible in Verus. Float-dependent harnesses are covered
by proptest instead.

### Setup

Verus requires a pre-built binary at `~/.local/verus/verus-x86-linux/verus`
(or set `VERUS_PATH`). Download from https://github.com/verus-lang/verus/releases.
The `verus-verified/` crate is standalone (own toolchain, own workspace) — do NOT
add it to the main workspace.
