+++
title = "DualPipeline"
description = "What formal proofs miss, how runtime comparison catches it, epsilon thresholds."
weight = 90
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Why DualPipeline exists

Kani proves safety (no panics, no OOB). It does **not** prove numerical correctness — that softmax outputs sum to 1.0, that attention produces a convex combination of values, or that a GPU kernel matches the CPU reference within acceptable error.

DualPipeline fills this gap: run the same computation through two kernel implementations, compare outputs element-by-element, report divergence.

## How it works

Given two implementations of the same trait (e.g., `CpuAttention` and `CudaAttention`):

1. Generate random inputs at configurable dimensions
2. Run both implementations on the same input
3. Compute max absolute difference and L2 distance
4. Report pass/fail against an epsilon threshold

```rust
let runner = DualPipelineRunner::new(1e-4);
let report = runner.compare_attention(
    "CPU vs GPU",
    &CpuAttention,
    &CudaAttention::new()?,
    &configs,
);
assert!(report.pass);
```

## Epsilon thresholds

Different comparison scenarios need different tolerances:

| Comparison | Typical epsilon | Why |
|-----------|----------------|-----|
| CPU vs CPU (same code) | 0.0 | Identical algorithms → identical output |
| CPU vs GPU (f32) | 1e-4 | Different reduction order → different rounding |
| f32 vs f16 KV cache | 1e-3 | Half-precision loses mantissa bits |
| Float vs quantized (Q4K) | 1e-2 | 4-bit quantization introduces systematic error |

The epsilon reflects the expected numerical divergence, not a quality threshold. A pass means "the implementations agree within the precision of the lower-precision one."

## What DualPipeline catches

- **Algorithmic bugs**: Wrong index, swapped dimensions, incorrect reduction — these produce large divergence immediately.
- **Precision drift**: Accumulated floating-point error across layers — shows up as gradually increasing max_abs_diff per layer.
- **Format errors**: Wrong endianness, incorrect dequantization — produces garbage output with max_abs_diff >> epsilon.

## What DualPipeline does NOT catch

- **Both implementations wrong the same way**: If the reference has a bug, DualPipeline won't catch it. The CPU reference is validated by proptest against f64 reference implementations.
- **Input-dependent failures**: DualPipeline uses random inputs. A bug triggered only by specific value patterns may be missed. Kani covers the input-independent safety properties.

## Batch-invariant verification

A specialized DualPipeline variant (`BatchInvariantVerifier`) runs the same prompt solo and inside different batch compositions, asserting bit-identical output. This proves that batching doesn't leak state between sequences.

## Source code

- [`kernel-verify/src/lib.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernel-verify/src/lib.rs) — `DualPipelineRunner` with compare methods per kernel trait
- [`engine/src/dual_pipeline.rs`](https://github.com/smgglrs-ai/tempura/blob/main/engine/src/dual_pipeline.rs) — server-level pipeline comparison
- [`engine/src/batch_invariant.rs`](https://github.com/smgglrs-ai/tempura/blob/main/engine/src/batch_invariant.rs) — batch isolation verification

## See also

- [Bounded model checking](../bounded-model-checking/) — proves safety properties Kani handles
- [Verification tiers](../verification-tiers/) — how DualPipeline results determine tier assignment
