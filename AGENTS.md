# tempura

Rust-native LLM inference server, started as a translation of llama.cpp and
now an independent engine. **Research project today; the ambition is a real
competitor to llama.cpp (local/CPU/edge) and vllm (served/GPU).** The
differentiator is verification-first architecture: every layer — from
quantization bit-packing to the batch scheduler — carries machine-checked
proofs (Kani, Verus, TLA+) alongside the production code.


**STATUS lives in `docs/status/current.md` (the `status` reference) — read it
before planning work. It lists verified gaps and open items; do not assume they
are done. Record session progress there, not in this file; archive completed
items to `docs/history/`.**

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
  bench-gpu (11 GPU benchmarks: bench-real-model, bench-batch, bench-mxfp4,
  bench-e2e, bench-attn, bench-kernels, bench-speculative, bench-quality,
  bench-decode-ops, bench-perplexity, bench-mmlu), compare (server-vs-server),
  integration, fp8-sim (CPU-side precision-class simulation harness: runs the
  MMLU/logit-band comparison with e4m3 round-trips injected into the real
  quantized forward — 2026-11-18 negative result recorded in docs/status:
  e4m3 activations fail the MMLU gate at 90.8% agreement; use it to test any
  future activation/weight precision class BEFORE building kernels)
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
    `eval.rs` (llama.cpp-compatible corpus perplexity), `tool_call.rs` (hermes/mistral/llama3 parsers + template auto-detect),
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
  deductive proofs for ALL inputs. 81 modules, 600 proof functions. Re-declares
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
every PR/push), `pages.yml`. HISTORY (2026-11-10/11): kani.yml had
NEVER passed before -- 0/210 green. Two root causes, both fixed:
(1) absolute /home/... path deps on cuda-oxide made every CI cargo
invocation fail in 0s (cargo resolves optional path-dep manifests
even with features off) -- fixed with relative sibling paths +
a cuda-oxide checkout step in the workflow; (2) three engine
harnesses had symbolic loop bounds (gguf_dequant_q4_0/q8_0_block_size,
dequant_i8_group_indices_in_bounds -- >116 min EACH locally,
#[kani::unwind] bounds took them to 2-5 s). First green run: 219
(2026-11-11), all 25 engine harnesses verified. Workflow shape:
dorny/paths-filter (docs-only pushes skip everything; kernels changes
run both kernels+engine; pure engine changes skip kernels), kernels
as one ~100-min job, engine as singleton-shard matrix jobs enumerated
from #[kani::proof] greps by a plan job (self-maintaining: new
harnesses cannot be silently skipped; empty enumeration hard-fails).
The 2026-11-05 --ignore-global-asm fix was only ever validated
locally.

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
     **Upstream backend** (`cuda-oxide/crates/rustc-codegen-cuda`,
     toolchain 1.99.1, has dp4a + libm-to-libdevice). The old fork
     (mma.sync, invariant.load) was dropped 2026-10-29: no tempura
     kernel used the fork-only features. DIVERGENCE-FREE POLICY
     (2026-11-10): tempura builds exclusively against upstream main --
     the last fork-only dependency (the `vec_width` optimization hint
     on the three cutile GEMV kernels, from the feat/mma-sync lineage)
     was removed; gpu-remote.sh builds ALL oxide crates with the
     upstream backend (oxide-gemv/Q2_0 validated: PTX byte-identical to
     the fork-built one, self-tests exact). Fork branches stay archived
     on the fork remote; the untracked fork backend at the cuda-oxide
     repo root (crates/rustc-codegen-cuda) is dead weight.
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
- [x] ragged no-graph batch decode (the multi-client serving path):
      ~23.5 ms/step at m=6 with ~440-token contexts (~255 tok/s aggregate;
      was 199 ms pre-2026-11-15, 47 ms after the sync-free + oxide-kernel
      fixes, 31 ms after the batch decode-attention + GPU argmax kernels,
      27 ms after the Q4K batch GEMV; 2026-11-16: Q4K AND Q6K projections
      (v/down + quantized lm_head) batched through bit-identical sm120
      m-loop batch kernels -- weights read once per batch, per-row
      numerics bit-exact vs the per-row GEMVs, bench-probe-q4k-batch /
      bench-probe-q6k-batch are the regression gates; the earlier
      stream-K/MMA batch kernels were DISABLED 2026-11-16 (numerically
      wrong -- see the 2026-11-16 status entry), then FIXED 2026-11-17
      (true m16n8k32 fragment mapping + Q4K nibble/dequant conventions,
      probe-gated at 1e-5, measured 3e-8 sum-order noise; re-enabled for
      the equal-position graph paths and evaluable on the no-graph path
      via TEMPURA_Q4K_BATCH_KERNEL=mma|streamk -- uniform across all m,
      composition-invariant + ppl-clean, but m=1 pays 47-88% so the
      DEFAULT stays the bit-identical kpar8 batch; see the 2026-11-17
      status entry for the measured trade-off) --
      sync-free scratch pool + kpar attention + the oxide BATCH
      decode-attention kernel (one launch per layer, grid (n_heads, m),
      bit-identical to per-request kpar) + GPU greedy argmax for
      pure-greedy sessions (sample+emit 11 ms -> ~8 us: the token is
      computed on-device, logits rows never downloaded) + the fused
      top-k-heap CPU sampler for sampled sessions (sample+emit 10.4 ms
      -> 0.87 ms at m=6/temp=0.8: O(n log k) selection instead of the
      full-vocab sort; GPU-side top-k kernel measured NOT worth building
      -- see the 2026-11-16 status entry); 2026-11-16c: m=1 ticks route
      through the single-seq CUDA graph on f16/e4m3 pools (bit-identity-
      pinned capture: kpar attention twins + per-row GEMVs + dtype
      kv_scatter_write twins + table RoPE -- the same kernels the
      generic path dispatches at m=1, so solo and batched decode are
      bit-identical; single-client 10.4 -> 5.2 ms/step short-ctx,
      9.0 -> 7.6 ms at 400-step ctx; TEMPURA_NO_M1_GRAPH=1 bisects; see
      the 2026-11-16c status entry); 2026-11-16d: m>=2 ticks route
      through position-synced RAGGED batch CUDA graphs on f16/e4m3
      pools, keyed (m, dtype), with EVERY per-step value device-read
      (per-row positions via in-graph RoPE row gathers, per-row KV write
      offsets + kv_lens + per-layer run base pointers re-uploaded per
      replay) -- batch-composition-INDEPENDENT (session churn never
      re-captures), new kv_scatter_write_rows{,_f16,_e4m3} per-row-
      offset batch scatter kernels (probe-gated bit-exact vs the twins),
      body bit-identity-pinned to the generic path (graph-routed and
      generic-routed checksums byte-identical, bench-ab-repro);
      m=6 21.6 -> 21.3 ms short-ctx, 23.8 -> 23.5 long -- a modest win,
      the m>=2 step is kernel-bound (see the 2026-11-16d status entry
      for the honest breakdown); TEMPURA_NO_M2P_GRAPH=1 bisects;
      2026-11-16e: the per-request RoPE chain REPLACED by ONE
      batched per-row rope launch per tensor (cutile_rope_rows_batch:
      strided source read, per-row gathered cos/sin rows, direct
      contiguous q_z/k_z writes, probe-gated bit-exact vs the
      per-request table rope) + the per-row-offset scatter append, in
      BOTH the generic and m>=2 graph paths on SHARED per-m buffers
      (op-for-op identical); m=6 21.4 -> 18.9 short / 23.9 -> 21.2 long
      = 317/283 tok/s; m=1 unchanged; TEMPURA_NO_BATCH_ROPE=1 bisects;
      see the 2026-11-16e status entry
- [x] Logprobs, top-logprobs, stop sequences, n completions, presence/
      frequency penalties, seed, max_completion_tokens alias
- [x] Tool/function calling (hermes/mistral/llama3 parsers + auto-detect),
      response_format (json_object/json_schema), guided decoding wired into
      tool_choice
- [x] Continuous batching with priority preemption + preemption-requeue,
      chunked prefill (512-token), prefix caching with cache_salt tenant
      isolation
- [x] Serving KV pool stores f16 by default (128 KiB/token GQA; ppl-gated
      2026-11-15: f16/f32 ratio 1.0003 on WikiText-2; TEMPURA_KV_F32=1
      restores f32).  e4m3 (FP8, 64 KiB/token, vLLM's production density)
      opt-in via TEMPURA_KV_DTYPE=e4m3 (ppl-gated: e4m3/f16 ratio 0.995;
      shared e4m3 bit-math helpers in tempura-kernels used by the cutile
      cast/prefill kernels, the oxide decode twins, and the mirror syncs)
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
      GEMM, stream-K batch GEMM (persistent blocks, grid=SM_COUNT;
      NOTE: the stream-K PTX was DEAD ON ARRIVAL 2026-10-22..2026-11-12
      -- an invalid `mov.b16 %h, %r32` (b32 source into a b16 dest)
      made cuModuleLoadData fail rc=218 on every run, silently falling
      back to mma_batch; ALL batch numbers before 2026-11-12 were
      measured on the mma_batch fallback.  Fixed with the vector
      unpack `mov.b32 {%h0, %h1}, %r55`; ptxas-clean; first-ever
      streamk run 2026-11-12: batch M=1 223, M=2 245, M=4 409, M=5 471,
      M=8 609, M=16 811 tok/s -- all NaN-free + deterministic, at or
      above the mma_batch baselines, M=16 +2%.  RETRACTED 2026-11-16:
      bench-probe-q4k-batch showed BOTH the stream-K and mma_batch
      kernels misparse the Q4K sub-block nibble/scale layout (12-31% rel
      error vs the per-row kpar8 reference on random data) -- those
      numbers were internally consistent garbage; both kernels were
      disabled at the has_* level. FIXED 2026-11-17: the misparse was
      threefold (invented m16n8k32 fragment mapping -- the real one is
      tid=lane&3/gid=lane>>2 with D = mxf4 acc_coords; byte group mw%4
      with hi-select mw>=4 -> mw>>1 with odd-mw hi-select; nibble-8 with
      sc/32 -> raw nibble, raw 6-bit scales), both kernels now pass the
      probe at 2.8e-9..7.5e-8 (sum-order noise) and the mma variant
      measured m=6 19.2->14.3 ms short / 21.2->16.4 ms long on the
      no-graph path -- but uniform routing (composition invariance)
      costs m=1 47-88%, so the DEFAULT stays q4k_gemv_sm120_kpar8_batch
      (bit-identical per row to the per-row kpar8 GEMV; the no-graph
      serving path uses it since 2026-11-16; the fixed tensor-core
      kernels are selectable via TEMPURA_Q4K_BATCH_KERNEL and serve the
      equal-position graph paths)
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
- [x] Prefill precision quality mode (2026-11-17; 2026-11-19 exact route
      restructured to the mmq-class kernel q4k_mmq_batch_sm120 -- 128
      neurons x 32 tokens/CTA, cp.async smem staging, f16 scale
      pre-extraction, exact 2732 -> 4101 tok/s pp128, MMLU 62.8%/96.8%,
      ppl 8.0863; the exact-route default kernel, TEMPURA_Q4K_PREFILL_KERNEL
      =mma128 restores the bit-identical class): TEMPURA_PREFILL_PRECISION
      =exact opts prefill projections out of the MXFP4 tensor-core GEMM
      (E2M1 activation quant + Q6K->MXFP4 double-quantization concentrated
      a rel~0.1 logit band on single-prefill workloads: MMLU GPU 48.8% vs
      CPU 61.6%, agreement 58.0%). Exact routes Q4K through the
      bit-identical-per-row kpar8 batch GEMM + Q6K sm120 batch GEMV:
      MMLU GPU 62.8%, agreement 96.8%, PASS -- at 355 tok/s pp128 (vs 7790
      MXFP4, llama.cpp 7335), TTFT P50 360 ms. Opt-in only (20.6x prefill
      cost); default stays MXFP4. The old TEMPURA_NO_MXFP4=1 prefill
      capture-panic is GONE (prefill graph off by default since cuda-oxide
      v0.5.0; that switch now runs, on the 9 tok/s host-dequant path).
      ACCELERATED (2026-11-17d): both exact-mode sides route through NEW
      LARGE-m INT8 tensor-core kernels -- q4k_mma128_batch_sm120 (per-row
      numerics bitwise-equal to q4k_mma_batch_sm120) and
      q6k_mma128_batch_sm120 (Q8_1 activations = llama.cpp's Q6_K x Q8_K
      class; 2-mma k16 split for Q6K's per-16-element scales) -- up to 128
      rows per launch with the weight matrix read ONCE per launch (the
      16-row chunking re-read weights m/16 times, which dominated the cost).
      Exact mode: 2508 tok/s pp128, TTFT P50 51 ms (7.1x the 355 tok/s
      kpar8 route; 2.9x from llama.cpp's 7335); decode unchanged. Kernel
      choice under exact: TEMPURA_Q4K_PREFILL_KERNEL=kpar8|mma|streamk
      (default mma128 for m>16), TEMPURA_Q6K_PREFILL_KERNEL=sm120 (default
      mma128). Probe-gated: bench-probe-q4k-batch (mma128 arm: <= 1e-5 vs
      kpar per-row AND bitwise vs chunked mma16), bench-probe-q6k-batch
      (mma128 arm: random-data CPU f64 reference from the same quantized
      activations, cancellation-aware gate 1e-5 * sum|terms| -- a naive
      rel-to-result gate false-positives on cancellative sums).
      2026-11-20 (q6k mmq port, kernels/src/q6k_mmq_batch_sm120_ptx.rs):
      the q4k-mmq treatment ported to the Q6K side -- 128 neurons x 32
      tokens per CTA, single cp.async staging buffer, borrow-free SWAR s8
      unpack, f16 (d*sc) pre-extraction (one product rounding); the exact
      route's Q6K DEFAULT (TEMPURA_Q6K_PREFILL_KERNEL=mma128 bisects).
      Probe arm 4: bit-identical to the CPU mirror + 2e-3*abssum class
      band vs mma128. q6k stack 6.9 -> 4.64 ms/step, exact mode
      4101 -> 4381 tok/s pp128 (llama.cpp gap 1.7x); q4k_mmq now dominates
      the step (20.3 ms, 71%) -- f16x2 scale pipes / GLU fusion / k-split
      remain the levers to the 6000-tok/s default-flip bar.
      2026-11-21 (q4k mmq grid.z k-split, same module as q4k_mmq): CTA z
      owns a k-block range, partials combined by the fixed-order
      q4k_mmq_split_reduce entry (deterministic; S=1 = the unchanged
      unsplit path; S>1 changes only the f32 sum order -- probe-gated by
      the SPLIT arms + split CPU mirror). Engine heuristic is n-ONLY
      (composition invariance): k_proj S=8 (0.059 -> 0.019 ms), q/o S=5
      (0.060 -> 0.049), gate/up S=3 (0.157 -> 0.146); exact mode 4381 ->
      5189 tok/s pp128 (gap 1.4x); TEMPURA_Q4K_MMQ_SPLIT=0 bisects.
      The 6000 bar is NOT met -- exact stays opt-in; remaining levers
      (f16x2 epilogue, GLU fusion, the q6k-side split) recorded in the
      2026-11-21 status entry.
      2026-11-18 (kernel-efficiency pass, profile + llama.cpp mmq study in
      docs/status): quantize_q8_1 precomputes the Q8_1 block rowsum into
      the ds word's high bits (llama.cpp's Q8_1 ds layout; all other
      consumers read only the low f16) and q4k_mma128 reads it for the
      min term -- bit-identical numerics, ~40 fewer ops/tile; exact mode
      2508 -> 2732 tok/s pp128 (llama.cpp 7335 = 2.7x). A WIDE-neuron
      variant (q4k_mma128w, 64 neurons/block warp-owned, grid.y row
      split; opt-in via TEMPURA_Q4K_PREFILL_KERNEL=wide; probe arm
      mma128w gated <= 1e-5, sequential (kb,mw) accumulation so NOT
      bitwise vs mma16) measured a WASH vs mma128 at every shape with
      grid_y swept 1/2/4/8 -- the L2-activation-re-reading hypothesis is
      refuted; the mma128 class is bound by the per-mma f32 scale chain
      (~7 fp ops/output element vs llama.cpp mmq's ~4 with load-time
      (d*sc)/(dmin*mv) pre-extraction) + A-fragment global loads (mmq:
      smem staging + ldmatrix) + no fused GLU writeback. The
      default-flip bar (>= ~6000 tok/s at reference quality) is NOT met:
      exact stays opt-in, MXFP4 stays default (7771 tok/s, unchanged).
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
typical-p/repeat/presence/frequency; decode path uses the fused top-k-heap
sampler -- O(n log k), token-identical to the full chain)

## Build & Test

```sh
cargo build              # build all workspace crates
                         # (.cargo/config.toml uses target-cpu=native only --
                         # never force +avx512* features: hosts without
                         # AVX-512 SIGILL, hosts with it get the full set
                         # from native anyway)
cargo test               # 907 test functions (26 ignored)
                         # NOTE: `cargo test --workspace` needs the CUDA
                         # toolkit on this host (bench-gpu, a workspace
                         # member, defaults to the cutile feature); on
                         # CPU-only hosts use --exclude tempura-bench-gpu
cargo kani --package tempura-kernels -Z unstable-options --ignore-global-asm
                         # 248 Kani harnesses (kernels+engine). The flag is
                         # REQUIRED: the gemm->pulp dep contains global ASM
                         # that Kani 0.67 cannot compile (without it the
                         # job fails -- CI was silently red on this from
                         # 2026-06-15 until fixed 2026-11-05). Kani 0.67.0
                         # (matches CI) + rustup must be installed: the
                         # Fedora system rustc has no rustup, and cargo kani
                         # setup needs one (rustup default = 1.99.1 to match
                         # the system toolchain; the repo's
                         # rust-toolchain.toml nightly-2026-04-03 is for
                         # cargo oxide only)
./scripts/run-verus.sh   # 600 Verus proof functions, 81 modules
cargo run -p tempura -- --help   # CLI: serve, pull, list, inspect
cargo run -p tempura -- serve --model /path/to/model.gguf --port 8080
```

Note: `cargo test` compiles `tempura-kernels` with the `oxide-gemv` feature
(feature unification via tools/bench-gpu), so the oxide GEMV PTX must be
present in the kernels build out dirs — `scripts/gpu-remote.sh oxide` or the
manual copy loop in the GPU section provides them. All oxide crates use
the upstream backend (toolchain 1.99.1).

Benchmarks: `tools/bench` (HTTP throughput, llama.cpp-bench-style pp/tg),
`tools/bench-gpu` (bench-real-model, bench-batch, bench-mxfp4, bench-e2e,
bench-attn, bench-kernels, bench-speculative), `tools/compare` (A/B two
servers), `engine/benches` (custom `harness = false` timing benches, not
criterion). Results in `bench-*.json` at root.
Sanitizer gate (GPU host): `bash scripts/sanitizer-gate.sh` (on fishtank,
repo root) runs compute-sanitizer memcheck over bench-probe-q4k-batch,
bench-probe-q6k-batch and bench-decode-ops, plus racecheck/initcheck on
the probes -- the OOB-read regression gate for the serving-path GEMV
kernels (2026-11-17: the kpar8 dual_block OOB reads passed every
numerics gate for weeks). Run it when touching any hand-written PTX
kernel or oxide host wrapper.

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
- STORAGE HYGIENE (2026-11-10 cleanup reclaimed ~160 GB across both hosts;
  these regrow silently): `.claude/worktrees/` accumulates one worktree +
  `worktree-agent-*` branch per agent session -- prune with
  `git worktree remove --force` + branch -D, but CHECK
  `git log --oneline main..<branch>` first (pre-landing versions of
  squash-merged work show as unmerged; compare by content --
  `git cat-file -e <tip>` + `git merge-base --is-ancestor <tip> main`
  audits a batch of tips). `target/debug` + `target/kani` (~55 GB on
  the laptop) and `tools/*/target` (the PTX lives at the crate ROOTS,
  tracked or regenerable -- cleaning target/ is safe) are fully
  regenerable; keep `target/release` (bench binaries) and the cuda-oxide
  backend .so (cuda-oxide/crates/rustc-codegen-cuda/target/release/
  librustc_codegen_cuda.so); its target/debug +
  target/x86_64-unknown-linux-gnu, the root cutile-rs/target and
  cuda-oxide/target are regenerable.

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

### Coverage Summary (2026-11-05, machine-counted)

**Kani:** 248 `#[kani::proof]` harnesses across kernels/ and engine/ —
attention, RMSNorm, RoPE, matmul, sampling, turboquant, quantized
(Q4K/Q5K/Q6K/Q2_0/TQ2/I2S/MXFP4 E2M1/UE8M0/NVFP4 UE4M3), KV cache (plain/paged/fp8),
model, speculative, disaggregated, weights, GGUF, GPU dispatch (grid dims,
buffer sizes, batch graph eligibility), tl2 LUT, deltanet, MoE, fused,
guided, ggml ops/repack/vec_dot/dequantize/quantize.

**Verus:** 81 modules, 600 proof functions — indexing safety for all Kani
harnesses that generalize (pure integer/bit), plus struct invariants, loop
proofs, state machines, GPU kernel decomposition, GQA head mapping, tensor
parallel sharding, causal masking, flash attention tiles, FP8/f16 encoding,
BitNet TQ2/I2S packing, MXFP4 bounds, NVFP4 UE4M3 encode/decode + split
layout, end-to-end pipeline composition, and
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

## Where things live

- `docs/status/current.md` — open work, verified gaps, latest session status
  (also advertised as the `status` reference). This is where session status
  updates go.
- `docs/history/` — archived status entries for completed work.
- `kernels/AGENTS.md` — cuda-oxide build recipes, backend `.so` caching, PTX
  copy loops, the mxf4-gemv cubin rebuild procedure and the stale-cubin
  warning. Loaded automatically when working under `kernels/`.
