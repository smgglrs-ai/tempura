---
title: "Kernel naming and dispatch"
description: "GpuCapabilities, GpuFeature bitflags, and the KernelVariant dispatch table"
weight: 30
---

# Kernel naming and dispatch

tempura names kernels by the GPU features they require, not by an internal
version number. This makes the dispatch logic self-documenting and prevents
the "v4 is better than v2" confusion when names don't reflect what changed.

## GpuCapabilities

`GpuCapabilities` is probed once at model load time from the CUDA device
properties. It returns a set of `GpuFeature` bitflags:

| Feature | Condition | Meaning |
|---------|-----------|---------|
| `DP4A` | cc >= 610 | INT8 dot-product-4-accumulate instruction |
| `SM120_ISA` | cc >= 1200 | sm_120 PTX ISA (Blackwell, cp.async mbar, etc.) |
| `MXFP4_MMA` | sm_120a only | MXFP4 tensor core MMA (consumer Blackwell) |

RTX 5090 has all three. An A100 (sm_80) has only DP4A.

## KernelVariant table

The `KernelVariant` table maps `(quantization, GpuFeature set)` to the PTX
function name to load. The table is evaluated at model init; each kernel slot
selects the best available variant:

```
Q4K decode GEMV:
  SM120_ISA + DP4A -> q4k_gemv_sm120_dp4a_kpar8  (best on RTX 5090)
  DP4A only        -> q4k_gemv_dp4a_kpar8          (sm_80 fallback)
  neither          -> q4k_gemv_f32                  (reference)

Q4K FFN fused gate+up:
  SM120_ISA + DP4A -> q4k_fused_silu_sm120_dp4a

Q6K decode GEMV:
  SM120_ISA        -> q6k_gemv_sm120_mbar_kpar8
  DP4A only        -> q6k_gemv_dp4a_kpar8

Q4K batch GEMM (M>1):
  SM120_ISA        -> q4k_streamk_batch_sm120

MXFP4 MMA (M>=16):
  MXFP4_MMA        -> mxfp4_mma_sm120a (cute-rs)
```

## Kernel name conventions

A kernel name encodes:
- **quantization format**: `q4k`, `q6k`, `q2_0`, `mxfp4`
- **GPU feature requirement**: `sm120` (SM120_ISA), `dp4a` (DP4A), `mbar` (cp.async mbar)
- **parallelism strategy**: `kpar8` (8-warp K-parallelism), `streamk` (stream-K)

Examples:
- `q4k_gemv_sm120_dp4a_kpar8` -- Q4K GEMV, requires SM120_ISA + DP4A, 8-warp K-par
- `q4k_gemv_dp4a_kpar8` -- same but sm_80 compatible (no sm_120 ISA)
- `q4k_fused_silu_sm120_dp4a` -- Q4K with gate+up fused SiLU, sm_120 PTX
- `q6k_gemv_sm120_mbar_kpar8` -- Q6K with cp.async mbar prefetch, 8-warp K-par
- `q4k_streamk_batch_sm120` -- Q4K stream-K batch GEMM for M>1

## Adding a new kernel

1. Write the kernel in `tools/oxide-gemv-<fmt>/src/main.rs`.
2. Name it following the convention above.
3. Add an entry to the `KernelVariant` table in `kernels/src/gpu_dispatch.rs`.
4. Add a `GpuFeature` flag if the kernel requires a new GPU capability.
5. Add Kani harnesses for grid dim bounds and buffer size non-overflow.
