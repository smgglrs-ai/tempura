# tempura

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

Thirteen workspace crates (see root `Cargo.toml`), one standalone verification
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
    ├─ model ─── HuggingFace hub download/inspect (library)
    ├─ server ── Axum HTTP library: OpenAI-compatible API + tempura
    │             extensions
    └─ cli ───── the `tempura` binary: serve, pull, list, inspect

tools/ (workspace): trace-cli (TUI trace viewer), bench (HTTP throughput),
  bench-gpu (9 GPU benchmarks: bench-real-model, bench-batch, bench-mxfp4,
  bench-e2e, bench-attn, bench-kernels, bench-speculative, bench-quality,
  bench-decode-ops), compare (server-vs-server), integration
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

- **model/** (`tempura-model`) — HuggingFace hub download/inspect library
  (`pull`, `list`, `inspect` format/arch detection; consumed by the CLI).
  Model store at `~/.cache/tempura/models` (XDG cache via `dirs::cache_dir()`,
  `XDG_CACHE_HOME` respected — same root the gpu-remote workflow uses).

- **server/** (`tempura-server`, ~4.2k lines, library) — Axum. OpenAI-compatible:
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

- **cli/** (`tempura`) — the `tempura` binary: `serve` (delegates to
  tempura-server's `run_server`), `pull`/`list`/`inspect` (tempura-model).
  Forwards `cuda`/`cutile`/`otlp`/`openmp` features to tempura-server; builds
  with `cargo build -p tempura`. API-visible identity strings
  (`system_fingerprint`, `owned_by`, OTel `service.name`) are `tempura`.

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

Scale: 174 first-party Rust files, ~142k lines (workspace crates + tools;
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
  - server (lib): `cuda`, `cutile`, `otlp`, `openmp`
  - cli: same four, forwarded to tempura-server (the `tempura` binary lives
    here — `cargo build -p tempura --features otlp` etc.)
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
- [x] oxide GEMV: Q2_0, Q4K (dp4a), Q6K (dp4a kpar8 fallback + sm120
      register-exchange kpar8, default on sm_120 since 2026-10-29); oxide decode attn
- [x] sm_120a native PTX: q4k/q6k GEMV, fused gate+up SiLU, mma.sync batch
      GEMM, stream-K batch GEMM (persistent blocks, grid=SM_COUNT)
- [x] CUDA graphs for single + batch decode (M=2..16+, dynamic
      max_batch_graph from VRAM), KV scatter-write (offset read on device —
      no per-step graph updates)
- [x] Native MXFP4 pipeline: E2M1+UE8M0 quant (CPU), on-GPU activation
      quant, tensor-core GEMM, all 7 projections, BF16 loader; 55.6 TFLOPS
      at M=64
- [x] Prefill fast path (2026-10-27): m>1 GEMMs run on-GPU -- tiled Q4K
      projections via MXFP4 quant + mxf4_batch_gemm_tiled in a single launch
      at full m_pad, Q6K tensors via the sm120 M-row batch GEMV chunked over
      rows (capped_batch_chunks); host-dequant fallback remains for
      incompatible strides. 408 tok/s pp128 (was 7 host-bound), TTFT P50
      314 ms, gap vs llama.cpp 7335 down to 18x. Verified: NaN-free,
      fresh-session logits bitwise-replay, A/B numerics vs exact-dequant
      max |logit| delta rel 0.10 (TEMPURA_BENCH_PREFILL_AB=1).
      TEMPURA_NO_GPU_PREFILL=1 bisects back to the host path.
- [x] Prefill attention on-device (2026-10-28): see open work for the
      current state (5816 tok/s pp128, TTFT P50 22.1 ms, 1.26x gap) --
      gather/flash/scatter kernel chain + scratch RoPE with cos/sin cache
      + pooled prefill norms/adds/SiLU + Q6K down_proj via MXFP4 (nsys-
      driven), TEMPURA_NO_GPU_ATTN bisect, latent stale-host-mirror fix.
- [x] Benchmarks (2026-10-27, post-store-mapping-fix, RTX 5090, bench-batch
      all-M verified with NO bisect switches — NaN-free + fresh-cache replay
      bitwise): single-seq 134 tok/s decode, ITL 7.44 ms; prefill 408 tok/s
      pp128, TTFT P50 314 ms, logits fingerprint det=OK (bench-real-model);
      batch M=1 135, M=2 133, M=4 224, M=5 253, M=8 332, M=16 444 tok/s
      (llama.cpp single-seq reference: 209; pp128 reference: 7335).
      All M>=2 batch fingerprints were RE-BASELINED 2026-10-27: every earlier
      MXFP4 batch output was computed with the broken accumulator store
      mapping (see Decode correctness) and is retracted. Earlier same-day
      retraction (frozen-position RoPE, wrong FFN) still applies to anything
      measured before that.
- [x] Decode correctness (2026-10-27): fixed sm120 fused-silu format mismatch
      (fed native 36-u32 weights to a repacked-42-u32 kernel -> all-NaN logits
      since introduction; every prior bench computed garbage at full speed) and
      the cp.async cross-lane visibility race in the q6k sm120 kernels
      (wait_group is per-thread; needs bar.sync 0 after it, mbar variant now
      gated on k_blocks % 8 == 0). Third fix round (same day, found via the
      prefill det=MISMATCH hunt + compute-sanitizer): the mxf4 GEMM's
      hand-derived m16n8 accumulator store mapping was wrong (row =
      lane%16 + 8*(j/2) wrote up to 7 rows past every m16 tile into the
      cudaMallocAsync pool -- silent neighbor corruption at small grids,
      CUDA error 700 at ~896+ blocks; columns 4-7 never written). Every
      MXFP4 batch-decode output M>=2 was computed with it and is retracted.
      The canonical mapping (mma_frag::acc_coords: row = lane/4 + 8*(j>>1),
      col = 2*(lane%4) + (j&1)) is validated by a random-data CPU reference
      in tools/mxf4-gemv -- uniform-data tests cannot catch fragment-mapping
      bugs (any store permutation "passes", NaN comparisons are false).
      bench-real-model prints a NaN-tolerant logits fingerprint;
      TEMPURA_DEBUG_NAN=1 gives per-op NaN bisection;
      TEMPURA_NO_SM120_KPAR / TEMPURA_NO_SM120_Q6K / TEMPURA_NO_OXIDE_ATTN /
      TEMPURA_NO_DUAL_BLOCK / TEMPURA_NO_MXFP4 / TEMPURA_NO_FUSED_SILU /
      TEMPURA_NO_GPU_PREFILL are kernel-path bisect switches
      (TEMPURA_NO_SM120_Q6K=1 opts the sm120 q6k kpar8 GEMV back out to the
      dp4a baseline; it is default-ON since 2026-10-29). Divergence-hunting tooling added 2026-10-27: per-op bits_sum in
      the nan-trace, TEMPURA_DEBUG_SLICES (per-1k-element slice sums),
      bench-batch TEMPURA_BENCH_STEP_FP / TEMPURA_BENCH_ONLY_M /
      TEMPURA_BENCH_DUMP, and compute-sanitizer on the standalone repro —
      this is how the silu stream race, the gate/up aliasing, and the store
      mapping were localized.
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

- [x] q6k_gemv_sm120_kpar8 (single-seq Q6K GEMV, renamed from
      ..._mbar_kpar8): FIXED and DEFAULT ON since 2026-10-29
      (register-exchange rewrite; TEMPURA_NO_SM120_Q6K=1 opts out).
      The kb=16 residual race is closed: 0/81920 at EVERY %8 shape
      including kb=16, kb=56 real weights 5e-6 vs CPU, 8/8
      DETERMINISTIC under churn, bench-quality teacher-forced PASS
      90.6%, 169 tok/s decode (ITL 5.92 ms). The fix, and the two
      hardware truths found on the way (SASS + a synthetic-weights
      micro harness that drives the real kernel, a permanent
      bench-decode-ops section):
        1. The kb=16 residual was CROSS-LANE smem reads: lanes read
           words other lanes copied via cp.async, and no CTA barrier
           fully ordered that publication. The fix removes the
           dependency class: each lane reads ONLY the words it copied
           itself (round 1: lane l copies word l; round 2: lanes 0..20
           copy words 32+l), and cross-lane values (d, scales, ql/qh
           words) arrive via shfl.sync.idx REGISTER exchange (registers
           carry no proxy semantics; the shuffle's data dependence on
           the source lane's ld.shared forces that load to complete).
           12 ld.shared -> 2 predicated ld.shared + 12 shuffles.
        2. The CTA bar.sync x3 at the loop tail is LOAD-BEARING, not
           belt-and-suspenders: with only [cp.async.wait_group 0;
           fence.proxy.async] (per-thread), iteration >= 2 read
           UNWRITTEN slots at kb=13/16/40 with BOTH wait placements
           (tail and rotated loop-head), SASS audited correct in both --
           MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S does not publish LDGSTS
           data to subsequent same-thread LDS on this hw/ptxas without
           a CTA barrier. (Earlier findings stand: ptxas ELIDES warp-
           uniform mbarrier arrive+try_wait pairs; bar.sync lowers to
           DEFER_BLOCKING -- the 2nd bar forces the 1st to complete,
           the 3rd pins it; the cross-warp reduce uses [st.shared;
           membar.cta; red.release.cta counter bump / ld.acquire poll;
           sequential 8-partial sum on tid 0] -- no shuffles.)
       Consequence of (2): the k_blocks % 8 == 0 host gate STAYS (the
       CTA loop barrier requires uniform warp trip counts). Run the
       micro harness: `bench-decode-ops` (section "sm120_kpar8 micro
       (REAL kernel, synthetic weights)").
- [x] **lm_head was dequantized to f32 at load (2.1GB) although the GGUF
      stores it Q6_K (435MB) -- FIXED 2026-10-29** (nsys
      `--cuda-graph-trace=node` per-kernel profile of one decode step: the
      f32 lm_head GEMV alone was 1.24ms = 21% of the 5.92ms ITL). lm_head
      now goes through the same Q4_K -> Q6_K -> Q2_0 -> f32 upload priority
      as the layer projections (try_proj); decode/prefill m=1 dispatch the
      exact Q6K kpar8 GEMV (result mirrored into SLOT_LOGITS with a D2D
      copy so the graph callers keep finding it there); m>1 batch paths get
      an MXFP4 tiled copy (435MB -> 109MB). Decode 169 -> 196 tok/s
      (ITL 5.92 -> 5.11 ms, gap 1.07x), prefill 5866 -> 6033, batch
      M=2 136 -> 244 / M=16 602 -> 796 (the batch lm_head GEMM re-read
      2.1GB f32 per step), +1.3GB VRAM. Verified: probe/prefill argmax
      IDENTICAL to the f32 path, bits_sum within ULP noise; bench-quality
      teacher-forced PASS 90.6% UNCHANGED (same mismatch list); bench-batch
      all M NaN-free + fresh-cache replay bitwise (fingerprints
      re-baselined for the MXFP4 lm_head).
- [x] **q6k sm120 kpar8 bandwidth gap FIXED (dual-block direct-load
      variant, 2026-10-29)**: the cp.async kernel ran down_proj
      [4096,14336] at ~0.86 TB/s while the q4k dual_block kernel does the
      same shape at ~1.25 TB/s -- the cp.async design's load-bearing CTA
      bar.sync x3 per iteration (21 per CTA) locksteps all 8 warps: every
      warp waits for the slowest warp's prefetch+compute before issuing
      its next, exposing DRAM latency with no way to skew.
      `q6k_gemv_sm120_kpar8_db` (second entry in the same module) adopts
      the q4k recipe with the register-exchange consume: weights load
      DIRECTLY into registers via ld.global.nc (2 warp-instructions per
      212-byte block -- no smem, no cp.async, no proxy fences, NO barriers
      in the loop), dual-block (blocks A and A+8 per iteration, 2x MLP),
      prefetch.global.L2 16 blocks ahead, running x pointers.
      Accumulation order unchanged -> bitwise-equal output (probe
      fingerprint identical; micro harness 0/81920 at every kb including
      non-multiples of 8 -- the _db variant needs no k_blocks % 8 gate).
      Decode 196 -> 204 tok/s (ITL 5.11 -> 4.89 ms, gap to llama.cpp
      1.025x). TEMPURA_NO_Q6K_DB=1 bisects back to the cp.async variant.
- [x] **Single-sequence GPU decode gap vs llama.cpp CLOSED at parity:
      210 vs 209 tok/s (2026-10-29).** Final change: cuGraphUpload after
      graph instantiation at all three capture sites (single decode +
      both batch graphs) -- the ~930-node decode graph's cuGraphLaunch
      was 122-139us of SERIALIZED host time per step (GPU idle; nsys
      runtime trace); pre-staging the graph on the device took ITL
      4.86 -> 4.77 ms (206 -> 210 tok/s), logits bitwise-unchanged.
      Cumulative decode timeline on this item: 140 (BK=512 lm_head) ->
      169 (q6k sm120 register exchange, default ON) -> 196 (lm_head via
      the GGUF's native Q6_K instead of 2.1GB dequantized f32) -> 204
      (q6k dual-block direct-load GEMV) -> 206 (Q6K v_proj via kpar) ->
      210 (cuGraphUpload). bench-quality teacher-forced PASS 89.1%
      throughout. Remaining micro-optimizations (not a gap anymore):
      logits DtoH 39-49us/step and the CPU argmax are GONE for greedy
      decoding since 2026-10-29: `forward_greedy` (InferenceModel trait,
      CutileInferenceWrapper override) runs the decode graph + a GPU
      argmax kernel (kernels/src/greedy_argmax.rs -- packed
      (ordered-key<<32)|index u64 atomicMax reduction whose tie-break
      exactly matches Rust max_by's last-max-wins; validated by a
      permanent bench-quality section: generate vs free-run manual
      continuation, 64/64 MATCH) and downloads 4 bytes instead of 513KB.
      The same change fixed a pre-existing bug: CutileInferenceWrapper::
      generate delegated to the CPU model's generate, which runs the CPU
      forward and PANICS on GPU-loaded models whose CPU projection
      weights were freed -- it is now a GPU-native loop. Remaining: ~40us
      residual launch cost (node-count reduction: rmsnorm is 3 kernels x
      65, the q/k rope chain 6 kernels x 32).
      NOTE: this GGUF (bartowski Llama-3.1-8B Q4_K_M) has a mixed layout
      -- 16 layers with Q6K v+down_proj, 16 with Q4K -- verified from the
      GGUF tensor-type census (193 Q4K + 33 Q6K) and the trace.
- [x] **Prefill on GPU: 7304 tok/s pp128, TTFT P50 17.5 ms -- llama.cpp
      PARITY (ref 7335, 1.004x; 2026-10-29).** Lever-by-lever history
      below (nsys-
      driven: on-device attention + scratch RoPE + pooled norms/adds/SiLU +
      Q6K down_proj via MXFP4 + flash BR=16 + lm_head BK=512/last-row d2d)
      vs llama.cpp 7335 (1.01x gap -- parity; 17.8x the 408 tok/s host-staged
      baseline). Six changes landed 2026-10-28:
      (a) The prefill attention is fully on-device: 3
      `cutile_gather_heads` (de-interleave Q/K/V into head-major padded
      buffers) + ONE `cutile_flash_attn_heads` launch for all heads (the
      audited `cutile_flash_attn_head` math with multi-head tiling) +
      `cutile_scatter_heads` back to interleaved -- 5 async kernels per
      layer, zero PCIe, zero host syncs, scratch pooled
      (`PrefillAttnScratch`, RopeScratch-style take/try_unwrap). 408 ->
      815 tok/s. Prefill KV append uses `append_device_gpu_only` (the
      staleness guard before the layer loop now syncs the host mirror
      whenever THIS call needs host attention -- this also fixes a latent
      stale-mirror bug where a prefill chunk after GPU-only decode read a
      mirror missing the GPU-only region). TEMPURA_NO_GPU_ATTN=1 bisects
      back to the host-staged path.
      (b) Prefill RoPE left the `forward_device` fallback (which per layer
      recomputed cos/sin on the CPU, did 2 blocking H2D uploads, 5 blocking
      allocs and a FULL stream synchronize) for dedicated prefill rope
      scratches: `RopeScratch::ensure_capacity` grows the buffers to the
      chunk size once, and a (positions, theta)-keyed cos/sin cache makes
      layers 2..N upload-free -- 815 -> 1305 tok/s, logits fingerprint
      BITWISE-identical (pure overhead removal). The prefill scratches are
      separate fields because ensure_capacity reallocates and the
      decode/graph scratches are pointer-baked into captured CUDA graphs.
      (c) Prefill norms/residuals/SiLU left the allocating paths for
      dedicated prefill scratch pools: forward_device_fused_scratch for both
      RMSNorms (sum_t/scale_t gained missing size checks -- a stale smaller
      buffer would silently norm only the first rows), add_device_scratch
      for both residual adds (gained the same size check; add_device
      removed), silu_mul_device_scratch for the FFN. Eliminates ~288
      blocking api::zeros.sync_on allocs per 32-layer chunk (3 per norm x2,
      1 per add x2, 1 per SiLU); 1305 -> 1365 tok/s, logits fingerprint
      still bitwise-identical.
      (d) nsys showed q6k_gemv_sm120_batch at 77% of prefill GPU time: the
      kernel caps at 16 rows per launch (smem reduction area), so an m=128
      chunk launches 8x per layer, each re-reading the full 47MB down_proj
      weight matrix at ~40% DRAM efficiency (~2.3ms/layer, ~74ms of the
      94ms TTFT). Fix: Q6K down_proj is now converted to MXFP4 tiled at
      load (maybe_upload_mxfp4_tiled_from_q6k -- the same double-
      quantization the Q4K projections already use, but LOSSIER: Q6K is
      6.5-bit) and prefill dispatch tries gemm_mxfp4_prefill before the
      q6k batch GEMV. +~1GB VRAM for the 32 MXFP4 down_proj copies.
      1365 -> 5126 tok/s, TTFT 94 -> 25 ms. PRECISION TRADEOFF (accepted):
      prefill A/B rel 0.1015 -> 0.1331; bench-quality teacher-forced PASS
      89.1% (57/64, all 7 mismatches cpu_margin <= 0.65 near-ties), prefill
      last-position argmax MISMATCH vs CPU (margin 0.264 near-tie; was
      MATCH). Escape hatches: decode (m=1) keeps the exact Q6K kpar GEMV;
      TEMPURA_NO_MXFP4=1 skips the conversion (down_proj falls back to the
      q6k batch path, prefill ~1365 tok/s, exact); TEMPURA_NO_GPU_PREFILL=1
      keeps the host-dequant prefill. Batch decode M>=2 also switched
      down_proj to MXFP4 (the batch path already preferred it when present)
      -- fingerprints RE-BASELINED 2026-10-28 and batch throughput ROSE:
      M=2 133->136, M=4 224->243, M=5 253->290, M=8 332->407,
      M=16 444->602 tok/s (all M re-verified NaN-free + fresh-cache replay
      bitwise). The capture-pre-size blocks in both batch graph captures
      now pre-size sum_t/scale_t and the add scratches to exact-m as well
      (the new size checks in forward_device_fused_scratch /
      add_device_scratch realloc on mismatch, which is illegal inside
      stream capture -- found by bench-batch, DriverError 900).
      (e) Flash-attn tile tuning: the multi-head kernel was CTA-count-bound
      at prefill sizes (grid 128 CTAs on 170 SMs at seq=128, running at
      ~2.7 TFLOPS / 13% of prefill GPU time). A (BR, BC) sweep (now a
      permanent bench-decode-ops section -- BR/BC are launch-time generics,
      one build sweeps all tuples) found BR=16 ~1.9x faster at both seq=128
      (48 vs 91us) and seq=512 (400 vs 754us); BR=8 overcorrects, larger
      BC strictly worse. BR 32 -> 16; BITWISE-neutral (per-row math
      unchanged -- identical logits fingerprint and identical bench-quality
      verdict). 5126 -> 5485 tok/s.
      (f) lm_head GEMV BK=512 + final-norm last-row d2d: the f32 lm_head
      projection ran cutile_gemv_coalesced at BK=128 = 709 GB/s (2.6ms per
      prefill, 11% of TTFT, and ~1.5ms of EVERY decode token). A BK sweep
      on the real weights (permanent bench-decode-ops section) found BK=512
      at 1475 GB/s (82% of peak, 2.08x); the d2d wrappers pick 512 when
      k_pad allows, 128 otherwise. The prefill final-norm also stopped
      downloading the whole [seq, h_pad] hidden state to extract the last
      row (blocking sync + PCIe) -- one D->D row copy now. 5485 -> 5816
      tok/s prefill, TTFT 23.4 -> 22.1 ms; DECODE 139 -> 171 tok/s
      (ITL 7.15 -> 5.83 ms -- the decode gap item below). Not bitwise
      (reduction order), but det=OK, NaN-free, bench-quality PASS 89.1%
      unchanged, argmax identical.
      (g) MXFP4 GEMM register-blocked variant (2026-10-29): the m16n8
      kernel issues ~10 global loads per MMA (load-issue-bound at
      111-127 TFLOPS on the m=128 prefill shapes; GEMMs = 75% of prefill
      GPU time, nsys). mxf4_batch_gemm_tiled_rb4 gives each warp an
      m16 x n32 tile -- 4 MMAs share each A-fragment load (per-MMA load
      count ~4.5, A re-read traffic /4). BITWISE-equal output (same MMA,
      same k-order; validated by the tool's random-data CPU reference AND
      a cross-kernel bit comparison, plus the in-engine prefill logits
      fingerprint staying identical). Wins only on large-n grids (it
      quarters the block count): gate/up [128,14336,4096] 122.4 -> 106.2 us
      (141.6 TFLOPS) but loses at n<=4096 -- the host dispatches rb4 only
      when ceil(n_tiles/16) * m_tiles >= 512 (RB4_MIN_BLOCKS; m=16 batch
      decode never qualifies). 6084 -> 6255 tok/s, TTFT 21.0 -> 20.4 ms.
      A first cut of rb4 dropped the `+ lane/4` from the B row index -- the
      cross-kernel bit comparison caught it instantly (uniform-data tests
      could not have).
      (h) m32 register blocking (2026-10-29): after (g), the large-n GEMMs
      were B-re-read-bound across m-tiles (gate/up re-read the 29MB B
      matrix 8x = 235MB per GEMM of L2/DRAM traffic). rb4m32 gives each
      warp an m32 x n32 tile (2 m16 halves x 4 n8 tiles = 8 MMAs per
      k-step; B fragments shared across the m-halves -> B re-reads /2).
      BITWISE-equal (same validation chain as rb4). Isolated (L2-hot)
      benchmark: gate/up 123.3 -> 57.8 us (259.9 TFLOPS); in-pipeline
      ~67us (B streams from DRAM per layer). Dispatched when m % 32 == 0
      and its grid >= RB4M32_MIN_BLOCKS (128): gate/up/down/q/o at m=128
      qualify, k/v (n=1024) stays m16n8, m=16 batch decode stays m16n8.
      6255 -> 6481 tok/s, TTFT 20.4 -> 19.7 ms (1.13x gap).
      (i) rb2m32 (m32n16/warp, 2026-10-29): the down_proj GEMM
      [128,4096,14336] ran 162.7us IN-PIPELINE vs 114.6 isolated -- its
      rb4m32 grid (128 blocks = 512 warps = 3/SM) cannot hide the DRAM
      latency of the streamed B matrix. rb2m32 doubles the warp count for
      the same m32 tiling (2 n-tiles per warp instead of 4): down
      114.6 -> 97.5 us (154.1 TFLOPS) and q/o 35.0 -> 30.4 us isolated;
      gate/up keeps rb4m32 (57.5 vs 65.3), k/v keeps m16n8. Dispatch
      tiers: rb4m32 (grid >= 256) > rb2m32 (grid >= 128) > rb4 (m%32!=0
      large-n) > m16n8. 6481 -> 6865 tok/s, TTFT 19.7 -> 18.6 ms
      (1.07x gap). Remaining GEMM structure: ~12ms of the 18.6ms;
      non-GEMM overhead ~5ms (flash 1.5, silu 0.9, adds/rope/gather/
      scatter/rmsnorm/quant ~2.0, host ~0.6).
      (j) Elementwise tile width (2026-10-29): the residual adds and the
      SiLU-mul ran on 32-element cutile tiles -- 384B moved per program
      (2x128B reads + 128B store), 16K tiny blocks per [128,4096] add at
      ~690 GB/s = half of DRAM peak (nsys). B 32 -> 256 (1KB tiles;
      elementwise -> bitwise-identical for any tile size; the q8_snap
      kernel keeps B=32 -- one Q8 block). 6865 -> 7257 tok/s, TTFT
      18.6 -> 17.6 ms (1.01x gap).
      (k) GEMM-epilogue fusion experiment (2026-10-29): rb4m32_silu (UP
      projection: C = silu(gate) * up, gate read in the store epilogue)
      and rb2m32_add (DOWN: C = acc + residual) -- both validated (tool
      test 4: silu within 1e-3 of the f64 reference, add exact) but
      MEASURED AS A REGRESSION in-pipeline: prefill 7257 -> 6994 tok/s.
      The epilogue reads gate/res at the m16n8 accumulator's scattered
      store addresses (~32B transactions) -- that costs more than the
      eliminated silu_mul/vector_add kernels. Dispatch is default-OFF
      (TEMPURA_FUSED_EPILOGUE=1 opts in); the kernels + validation stay
      in tools/mxf4-gemv for a future attempt that stages the epilogue
      read through coalesced smem. Post-revert: 7304 tok/s, TTFT
      17.5 ms, 1.004x gap -- prefill parity.
      Verified 2026-10-28: bench-decode-ops "prefill attn dev" 8/8, "rope
      prefill scratch" 4/4 and "rmsnorm prefill scratch" (grow/reuse/
      shrink/odd-seq) 4/4 all
      bitwise-exact vs their references; bench-real-model NaN-free, det=OK,
      A/B vs host-dequant rel 0.1015 (the documented MXFP4 band); decode
      unchanged at 139-140 tok/s. Remaining levers: gemm dispatch host
      overhead in the chunk loop, the gather/scatter copies (~3 extra
      passes over Q/K/V activations -- fusible into the flash kernel or the
      RoPE/GEMM epilogues), BR/BC tile tuning for long prompts
- [ ] Validate hybrid DeltaNet pipeline against real Qwen3.6 / Bonsai 2
      weights (last attempt: "architecture incompatible" — diagnostics added,
      commit 9c6362d)
- [x] **m=1 GPU decode path produced wrong logits -- FOUND and FIXED
      (2026-10-28, bench-quality + bench-decode-ops)**. Root cause: the
      hand-written sm_120 PTX kernels used `shfl.sync.bfly/idx` with
      c=0x71F for their 8-lane mini-warp reductions; the correct width-8
      encoding is c=0x181F (verified via `nvcc --ptx` on sm_120). With
      0x71F, mini-warp 3 (lanes 24-31) collapsed onto lane 24: the
      reduction returned 8x lane-24's partial and dropped sub_l 1-7, so
      j-group 3 of every Q4_K GEMV contributed garbage whenever its qs
      bytes were non-uniform (~110% relative logit error). Fixed in
      q4k_gemv_v4_ptx.rs, q4k_batch_sm120_ptx.rs, q4k_fused_silu_v4_ptx.rs
      (18 shuffle sites). Uniform-data tests cannot catch this class --
      the one-hot element sweep in bench-decode-ops is the regression
      test. Post-fix: bench-quality shows 100% argmax agreement (64/64
      tokens, free-running greedy, identical continuations) vs the CPU
      reference; 134 tok/s decode unchanged. Every decode op now has a
      real-weights CPU-math validation in bench-decode-ops: RoPE table +
      graph-gather paths, KV scatter-write, q6k kpar GEMV, q4k sm120 kpar
      GEMV, oxide decode attention (splitk + kpar, incl. dirty
      4096-row cache), rmsnorm fused scratch, silu_mul.
- [x] q4k_gemv_dp4a_q8_raw (sm_80 Rust-codegen fallback GEMV): WRONG --
      FOUND and FIXED (2026-10-28). Root cause: the crate's device-side
      f16_to_f32 returned f32-DENORMAL bits for f16-subnormal inputs
      (exp==0 path: sign | mant<<13), so every Q4_K weight block whose f16
      super-scale d fell just below 2^-14 (extremely common -- quantizers
      emit d right around the subnormal boundary) lost its scale term and
      only the -dmin*m contribution survived: 414/4096 output rows off by
      ~0.1-0.4 (additive error, invisible on large outputs -- "correct on
      large, wrong on small"). The sm_120 PTX kernels dequant d/dmin via the
      hardware cvt.f32.f16 and were never affected. Also fixed
      f32_to_f16_bits (used by the SHARED quantize_q8_1 kernel): it flushed
      subnormal f16 Q8_1 activation scales to zero, silently dropping any
      32-element activation block with amax < ~0.0078 -- now rounds to
      nearest subnormal. Post-fix: bench-decode-ops q8_raw rel 0.0003
      (was 0.333), 0/4096 bad rows, one-hot sweeps (tail + main loop,
      incl. negative d) clean; 134 tok/s decode unchanged. Regression
      tests: the q8_raw real-weights census + crafted one-hot sweeps +
      per-block isolation in bench-decode-ops. NOTE: the PTX is a
      gitignored build artifact -- kernel-source changes to
      tools/oxide-gemv-q4k require `cargo oxide build` + the out-dir copy
      loop (see the GPU Kernels section) on every clone.
- [x] **Short-prompt GPU prefill wrong -- FOUND and FIXED (2026-10-28, three
      distinct bugs, found by bench-quality + bench-decode-ops)**:
      1. The embedded mxf4_gemv.cubin was STALE: the 2026-10-27 accumulator
         store-mapping fix (11d2d7c) changed tools/mxf4-gemv/src/main.rs but
         the cubin was never rebuilt/committed -- the engine ran the pre-fix
         kernel (rows written past the m16 tile into the cudaMallocAsync
         pool; outputs allocation-history dependent) while the standalone
         tool compiled the fixed source and passed. Symlink-style trap: the
         tool validates SOURCE, the engine runs the CHECKED-IN BINARY.
         Lesson recorded below in the mxf4 build recipe. Post-rebuild the
         bench-decode-ops quant+GEMM composition test is bit-exact at every
         m (the old cubin failed at ALL m including 128; uniform-scale and
         permutation-cancelling random-nibble tests cannot catch it).
      2. cutile_flash_attn_head computed num_kv_tiles = kv_len / BC
         (truncating): kv_len=12 gave 0 tiles (all-NaN logits from 0/0) and
         kv_len=85 silently dropped keys 64..84 for the last rows. Fixed to
         ceil_div -- this also cut the m=86 host-dequant prefill delta from
         1.77 to 0.34 and the teacher-forced decode delta from 1.87 to 0.37.
      3. The GGUF tokenizer defaulted add_bos=false when the
         tokenizer.ggml.add_bos_token key is absent -- Llama-3.1 GGUFs ship
         bos_token_id without that key, so prompts lost their BOS
         (generation degraded, worst on short prompts). from_gguf_metadata
         now defaults BOS-on for the llama/mistral/gemma families (llama.cpp
         LLM_ARCH semantics); llama.cpp-arbiter verified.
      Post-fix: fast prefill argmax-MATCH at every prompt length (m=13: 3.43,
      m=86: 1.91, both vs the CPU reference -- the intrinsic MXFP4 4-bit
      noise band, matching the documented pp128 A/B rel 0.10); short-prompt
      CPU and GPU continuations are identical text.
- [x] Generation-quality validation of the MXFP4/decode paths (2026-10-28):
      bench-quality (greedy GPU-vs-CPU continuation, per-position argmax
      agreement, teacher forcing, NaN-aware diffs, GPU decode-vs-prefill
      self-test) + bench-decode-ops (per-op real-weights CPU-math checks +
      one-hot element sweeps) + a llama.cpp arbiter run. Result after the
      shuffle fix: 100% argmax agreement, 64/64 identical tokens
      free-running (85-token prompt, Llama-3.1-8B Q4_K_M, logits max_abs
      4.08 / avg 2.17 -- same band as the prefill's own CPU delta).
      MODE CHANGE (2026-10-28, later): teacher forcing is now the DEFAULT
      (TEMPURA_QUALITY_FREE=1 restores free-running). The free-running
      verdict is a coin flip -- post-BOS-fix prompts land the model in
      near-tie territory (first divergence at a margin-0.014 top1/top2 gap)
      and every position after the flip "disagrees" with logit deltas of
      10-20 that measure different contexts, not kernel health. Same code,
      same day: free-running 15.6% FAIL while teacher-forced PASSes at
      92.2% with max_abs 4.27 / avg 2.44 (the documented band). Use
      teacher-forced agreement + the logit-delta band as the quality gate;
      use free-running only to eyeball continuation text.
      Cross-impl note: the CPU reference diverges from llama.cpp after ~2
      greedy tokens (near-tie argmax flips, quantized-impl accumulation
      differences) -- llama.cpp text output remains the ground truth for
      "is the model thinking correctly".
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
cargo run -p tempura -- --help   # CLI: serve, pull, list, inspect
cargo run -p tempura -- serve --model /path/to/model.gguf --port 8080
```

Note: `cargo test` compiles `tempura-kernels` with the `oxide-gemv` feature
(feature unification via tools/bench-gpu), so the oxide GEMV PTX must be
present in the kernels build out dirs — `scripts/gpu-remote.sh oxide` or the
manual copy loop in the GPU section provides them. `oxide` now selects the
backend per crate (fork for oxide-gemv, upstream+1.99.1 for the dp4a crates)
-- the fork backend cannot compile the f16-subnormal fix in oxide-gemv-q4k
and silently leaves a stale PTX behind.

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
mma.sync intrinsics and invariant.load codegen. (A 2026-10-27 bug-hunt
branch here was deleted after the "cutlass-mlir miscompile" resolved as our
own kernel's wrong accumulator store mapping — see Decode correctness in
Implemented capabilities; the record lives in tempura's history.)

**`cargo oxide build` uses a
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

# mxf4-gemv (MXFP4 tensor-core GEMM cubin): DIFFERENT recipe from the crates
# above — upstream backend (nested tree), cutlass-mlir device backend, and the
# managed CUTLASS compiler. The cubin is include_bytes!()d by tempura-kernels,
# so a normal cargo build picks it up after this step.
#
# WARNING (2026-10-28 stale-cubin incident): any change to
# tools/mxf4-gemv/src/main.rs MUST be followed by this rebuild AND a commit
# of the regenerated mxf4_gemv.cubin in the SAME commit. The standalone
# `mxf4-gemv` binary compiles from source and will pass while the engine
# runs the checked-in binary -- a stale cubin shipped the pre-fix
# accumulator store mapping for a day and produced allocation-history-
# dependent garbage prefill outputs. Regression test: bench-decode-ops
# "mxfp4 quant+gemm composition" (bit-exact, real quant scales, padded m).
cd tools/mxf4-gemv
export CUDA_OXIDE_BACKEND=~/Code/github.com/NVLabs/cuda-oxide/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
export CUDA_OXIDE_CUTLASS_COMPILER=~/.cargo/cuda-oxide/toolchains/cutlass/4.7.0/x86_64-cu13/lib/libCutlassCompiler.so
export LD_LIBRARY_PATH=~/.rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu/lib:$LD_LIBRARY_PATH
RUSTC_BOOTSTRAP=1 RUSTUP_TOOLCHAIN=1.99.1 CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide build --arch sm_120a
# The standalone binary validates the kernel against a random-data CPU
# reference (uniform-data tests cannot catch fragment-mapping bugs) across
# all shapes incl. the engine's (128, 4096, 4096):
LD_LIBRARY_PATH=~/.rustup/toolchains/nightly-2026-10-11-x86_64-unknown-linux-gnu/lib:$LD_LIBRARY_PATH \
  ./target/release/mxf4-gemv

# From the laptop, scripts/gpu-remote.sh runs all of this on the GPU host:
#   scripts/gpu-remote.sh oxide
#   scripts/gpu-remote.sh bench real-model Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Benchmark (oxide-gemv feature already in tools/bench-gpu/Cargo.toml):
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release \
  --bin bench-real-model -- /path/to/model.gguf
```

Current results (RTX 5090, see `bench-gpu-rtx5090.json`):
- Llama-3.1-8B Q4K single-sequence: **171 tok/s decode**, ITL 5.83 ms
      (1.22x gap vs llama.cpp 209 tok/s; outputs verified NaN-free +
      deterministic, 2026-10-27 post-fix. A 2026-10-28 retraction (outputs
      numerically wrong vs CPU/llama.cpp) was resolved the same day: the
      sm_120 PTX shuffle-encoding bug -- see open work's fixed item.
      bench-quality now shows 100% argmax agreement vs the CPU reference,
      64/64 identical tokens free-running. 2026-10-28 fused-silu raw-format
      port: 134 -> 140 tok/s; lm_head GEMV BK=128 -> 512 (709 -> 1475 GB/s):
      140 -> 171 tok/s; 2026-10-29 the register-exchange rewrite of the
      sm120 q6k kernel fixed the kb=16 residual and made it default-ON
      (169 tok/s); 2026-10-29 the lm_head upload switched from dequantized
      f32 (2.1GB read per token) to the GGUF's native Q6_K via the kpar
      GEMV: 204 tok/s, ITL 4.89 ms, bench-quality teacher-forced PASS 90.6%
      unchanged, +1.3GB VRAM; 2026-10-29 the dual-block direct-load q6k
      variant (no cp.async/barriers, 2x MLP, L2 prefetch) closed the q6k
      down_proj bandwidth gap: 196 -> 204 tok/s, bitwise-identical logits;
      the Q6K v_proj moved to the kpar path (206) and cuGraphUpload after
      instantiation removed the 122-139us serialized launch cost:
      **210 tok/s -- llama.cpp parity reached (ref 209)**, ITL 4.77 ms,
      teacher-forced PASS 89.1%)
- Llama-3.1-8B Q4K prefill: **7304 tok/s pp128**, TTFT P50 17.5 ms
      (1.004x gap vs llama.cpp 7335 -- parity; logits det=OK, NaN-free; A/B numerics vs
      exact-dequant max |logit| rel 0.1331 -- the MXFP4 band including the
      Q6K->MXFP4 down_proj double-quantization (lossier than the Q4K path's
      0.1015; bench-quality teacher-forced PASS 89.1%, all mismatches
      near-ties). 2026-10-28 stack: on-device attention (408 -> 815) +
      scratch RoPE with a cos/sin cache (815 -> 1305) + pooled prefill
      norms/adds/SiLU (1305 -> 1365) + Q6K down_proj via MXFP4
      (1365 -> 5126, nsys-driven) + flash BR=16 tile tuning (5126 -> 5485,
      bitwise-neutral) + lm_head BK=512 + last-row d2d (5485 -> 5816).
      Kernel-level verification bitwise-exact
      vs the CPU causal reference for the attention/RoPE/norm paths, see
      open work's prefill item. Decode at 171 tok/s)
- Llama-3.1-8B Q4K batch decode (2026-10-27, post-store-mapping-fix, NO
      bisect switches; verified = NaN scan every step + final-step
      fingerprints + fresh-cache replay bitwise identical; every M>=2
      fingerprint re-baselined this day — earlier MXFP4 batch outputs were
      computed with the broken accumulator store mapping and retracted):
      **M=1 198, M=2 244, M=4 407, M=5 466, M=8 606, M=16 796 tok/s**
      (re-baselined 2026-10-29 after the lm_head upload switched from
      dequantized f32 to the GGUF's native Q6_K + an MXFP4 tiled copy for
      the M-row batch GEMM -- the f32 lm_head re-read 2.1GB per batch step;
      was M=1 174, M=2 136, M=4 243, M=5 290, M=8 407, M=16 602 on
      2026-10-28; all M re-verified NaN-free + fresh-cache replay bitwise).
      (re-baselined 2026-10-28 after down_proj moved to MXFP4 -- was
      M=1 135, M=2 133, M=4 224, M=5 253, M=8 332, M=16 444 on Q6K batch
      GEMV; M=1 rose again 141 -> 174 with the lm_head BK=512 fix; all M
      re-verified NaN-free + fresh-cache replay bitwise).
      M>=2 uses the generalized M-row q6k batch kernel + MXFP4 tensor-core
      Q/K/O/gate/up (m_valid-masked activation quant); M=1 routes to the
      single-seq CUDA graph path.
- MXFP4 M=64: 55.6 TFLOPS; register-blocked rb4 variant (m16n32/warp,
      large-n grids only): 141.6 TFLOPS at (128, 14336, 4096); rb4m32
      (m32n32/warp, B re-reads halved): 259.9 TFLOPS isolated /
      ~67us in-pipeline on the same shape; rb2m32 (m32n16/warp, finer
      grid for small-n/long-k): 154.1 TFLOPS at (128, 4096, 14336)

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
