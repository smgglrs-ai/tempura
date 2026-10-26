+++
title = "Architecture"
description = "Three-crate workspace, kernel traits, and the verification contract."
weight = 20
template = "docs/section.html"

[extra]
toc = true
+++

## Design principles

1. **Kernel traits as the verification contract.** Engine code is generic over kernel implementations. The same trait is implemented by CPU (for verification) and GPU (for production). Kani proves both adhere to the contract.

2. **Tiered verification.** Kani proves safety (no panics, no OOB) on bounded instances. Verus proves properties for all inputs via SMT (struct invariants, loop termination, state machines). DualPipeline validates numerical equivalence. Property tests cover floating-point properties neither tool can handle.

3. **Inference traces are first-class.** Every forward pass produces a structured `InferenceTrace` at configurable granularity. Traces are queryable, diffable, and replayable.

4. **Deterministic by design.** Seeded RNG, controlled reduction order. Same input + seed = bit-identical output.

## Crate ecosystem

Four crates in the main workspace, plus a standalone verification crate:

### tempura-kernel-api

The stable contract third parties code against. Contains all kernel traits and shared types, zero implementations.

**Traits:** `Attention`, `RMSNorm`, `RoPE`, `MatMul`, `Sampler`, `LayerNorm`, `AllReduce`, `KvCacheCompressor`, `GuidedDecoder`

### tempura-kernels

First-party implementations: CPU reference (always available) and CUDA via cuda-oxide (behind `cuda` feature flag).

### tempura-engine

Inference engine generic over kernel traits. Contains `TransformerModel`, `KernelPipeline`, `InferenceSession`, `InferenceTrace`, `DualPipeline`, weight loading, tokenizer integration.

### tempura

HTTP serving layer. Axum-based, OpenAI-compatible API. Continuous batching, speculative decoding, prefix caching, Prometheus metrics, OpenTelemetry export.

## KernelPipeline

The `KernelPipeline<A, N, R, M, S>` bundles kernel implementations into a composable unit:

```rust
pub struct KernelPipeline<A, N, R, M, S> {
    pub attention: A,
    pub rmsnorm: N,
    pub rope: R,
    pub matmul: M,
    pub sampler: S,
}
```

Factory methods: `KernelPipeline::cpu_verified()` for testing, `KernelPipeline::cuda()` for GPU inference. Third parties compose their own pipelines by mixing implementations.

## GPU Inference Architecture

### GpuCapabilities and feature-based dispatch

Runtime kernel selection is driven by `GpuCapabilities`, which probes the device
and returns a set of `GpuFeature` bitflags:

- `DP4A` -- cc >= 610 (Volta and later): enables Q8_1 activation quantization + dp4a integer MACs
- `SM120_ISA` -- cc >= 1200 (Blackwell): enables cp.async mbar, sm_120 PTX kernels
- `MXFP4_MMA` -- sm_120a only (RTX 5090 consumer Blackwell): enables MXFP4 tensor core ops

The `KernelVariant` table maps `(GpuFeature, quantization)` to the correct PTX
function. Kernel names encode their requirements: `q4k_gemv_sm120_dp4a_kpar8`
requires SM120_ISA + DP4A and uses 8-warp K-parallelism. The sm_80 fallback
`q4k_gemv_dp4a_kpar8` runs on any DP4A-capable GPU.

### CUDA graph decode

Single-sequence decode captures a CUDA graph once per generation. Each step
calls `cuGraphLaunch` once instead of 68+ individual CUDA driver calls. Key
design points:

- `kv_len` is a device pointer (`*const i32`) read at kernel execution time;
  no graph update is needed per step
- KV scatter-write kernel reads `kv_write_offset` from device, eliminating
  64 `cuGraphExecMemcpyNodeSetParams` calls per step
- The graph is invalidated only when `kv_len` exceeds the captured length

### Batch decode with CUDA graph

`BatchDecodeGraph` captures a graph for M sequences simultaneously. Design:

- `max_batch_graph` is computed from available VRAM at startup (13+ sequences
  for Llama-3.1-8B Q4K on RTX 5090 with 32 GB VRAM)
- QKV projections launch on separate CUDA streams for concurrency
- Stream-K GEMM (`q4k_streamk_batch_sm120`): persistent grid of 170 blocks
  (= SM_COUNT on RTX 5090), cp.async double-buffer, mma.sync tensor cores

Performance (RTX 5090, Llama-3.1-8B Q4_K_M):

| Batch size | Aggregate tok/s |
|-----------|----------------|
| M=1 | 130 |
| M=2 | 134 |
| M=4 | 243 |
| M=8 | 417 |
| M=16 | ~600 |

### MXFP4 weight format

MXFP4 uses the `E2M1` format for weights and `UE8M0` for per-block scales,
matching the CUTLASS MLIR sm_120a MXFP4 MMA convention. All 7 projections
(Q, K, V, O, gate, up, down) can be stored and computed in MXFP4.

The `cute-rs` library drives the sm_120a MMA tile operations. At M=64,
the MXFP4 pipeline reaches 55.6 TFLOPS. MXFP4 is only useful at M>=16 due
to padding waste in smaller tiles -- batch decode at M<16 uses stream-K Q4K.

BF16 safetensors weights are tiled into MXFP4 at load time by the native
safetensors loader, with zero re-quantization overhead at inference time.
