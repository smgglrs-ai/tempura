---
title: "MXFP4 pipeline"
description: "E2M1 weights, UE8M0 scales, cute-rs MMA, BF16 loader, and when to use MXFP4"
weight: 40
---

# MXFP4 pipeline

MXFP4 is a microscaling floating-point format: 4-bit weights (E2M1) with shared
8-bit scale factors (UE8M0) per 16-element block. On sm_120a (RTX 5090 consumer
Blackwell), CUDA MMA natively supports MXFP4 tiles, achieving 55.6 TFLOPS at
M=64 on Llama-3.1-8B.

## Format

**E2M1** (4 bits): 2 exponent bits, 1 mantissa bit, 1 sign bit. Values span
roughly [-6, 6] with 16 representable levels.

**UE8M0** (8 bits): unsigned 8-bit exponent. One scale per 16 weights (a block).
The per-block scale is a power of two, making dequantization a bit-shift.

A weight matrix tile looks like:

```
weights:  [e2m1 e2m1 e2m1 ... e2m1]  (N x K/2 bytes, packed 2 weights/byte)
scales:   [ue8m0 ... ue8m0]           (N x K/16 bytes, one per 16-element block)
```

## cute-rs and sm_120a MMA

`cute-rs` provides Rust bindings to CUTLASS's cute tile abstraction for sm_120a.
The MMA instruction `mma.sync.aligned.m16n8k32.row.col.f32.e2m1x2.e2m1x2.f32`
operates on 16x8x32 tiles of packed MXFP4 weights. The UE8M0 scale is applied
in the epilogue.

The cute-rs tile layout matches the CUTLASS MLIR sm_120a convention. Tempura
uses this layout for all 7 projection matrices (Q, K, V, O, gate, up, down).

## BF16 safetensors loader

BF16 safetensors weights are tiled directly into MXFP4 format during loading:

1. Load a BF16 tile of shape (N, K) from the safetensors file.
2. For each 16-element block, compute `max_abs = max(|w_i|)` and
   `scale = ue8m0_encode(max_abs / 6.0)` (the UE8M0 scale).
3. Quantize each weight: `e2m1_encode(w_i * pow2(-ue8m0_decode(scale)))`.
4. Pack two E2M1 nibbles per byte.

This runs on CPU during model load with no separate quantization step at
inference time. The Kani harnesses verify the UE8M0 encode/decode roundtrip
and E2M1 nibble bounds for all inputs.

## When to use MXFP4

MXFP4 is efficient when M (batch size) is large enough to fill the MMA tiles:

| Batch M | Recommended kernel |
|---------|-------------------|
| 1 | Q4K GEMV (stream-K at M=1 has padding waste) |
| 2..15 | Stream-K Q4K batch GEMM |
| 16..63 | Stream-K Q4K or MXFP4 depending on kv_len |
| 64+ | MXFP4 MMA (55 TFLOPS) |

The crossover point where MXFP4 beats Q4K stream-K depends on the model's
hidden dimension and head count. For Llama-3.1-8B (hidden=4096), MXFP4 wins
at M>=32 in most projection shapes.

## Performance

| Config | Throughput |
|--------|-----------|
| M=64, all 7 proj | 55.6 TFLOPS |
| M=32 | ~28 TFLOPS |
| M=16 | ~14 TFLOPS (approaching Q4K stream-K) |

RTX 5090 peak FP4 MMA: ~167 TFLOPS. The 55.6 TFLOPS at M=64 is 33% of peak,
limited by memory bandwidth for KV cache and activation loads.

## Formal verification

Kani harnesses cover:
- `e2m1_nibble_bounds`: for any BF16 input, the E2M1 nibble fits in [0, 15]
- `ue8m0_encode_decode`: roundtrip is identity within one ULP
- `bf16_upcast_correctness`: upcast E2M1+UE8M0 -> BF16 stays within error bound
- `gpu_dispatch_grid_dims`: MXFP4 grid dimensions fit CUDA limits for any valid M, N, K
- `batch_graph_eligibility`: determinism proof -- same config always yields same max_batch
