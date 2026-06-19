+++
title = "DualPipeline"
description = "Empirical equivalence testing — what formal proofs miss and how runtime comparison catches it."
weight = 90
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## The gap formal proofs leave

Kani proves *safety*: no panics, no out-of-bounds access. It does not prove *numerical accuracy*: that the GPU attention kernel produces the same output as the CPU reference. Two implementations can both be safe yet produce different numbers — different reduction orders, different precision, different rounding.

DualPipeline fills this gap by running two implementations on identical inputs and comparing outputs.

## How it works

```
Input tokens + seed
       │
       ├──→ Implementation A (e.g., CPU reference)  ──→ Output A
       │
       └──→ Implementation B (e.g., CUDA kernel)    ──→ Output B
       
       Compare: per-layer norm stats, logits, sampled tokens
```

The comparison reports:
- **Per-layer divergence**: max absolute difference and L2 distance of pre-norm and post-norm statistics
- **Logits divergence**: how far the final probability distributions differ
- **Token match**: whether the same tokens were sampled (deterministic sampling with identical seeds)

## Epsilon thresholds

<!-- TODO: Why exact equality is wrong for floating-point (reduction order).
     How to choose epsilon: f32 precision (~1e-7 relative), GPU vs CPU
     expected divergence (~1e-6 for matmul, ~1e-7 for elementwise).
     When divergence exceeds epsilon: is it a bug or a precision difference?
     The role of the verification tier in communicating this. -->

## What DualPipeline catches that Kani cannot

<!-- TODO: Precision loss from GPU parallelism (different reduction order).
     Quantization accuracy (is int4 matmul "close enough"?).
     Implementation bugs that are safe but wrong (e.g., applying RoPE
     to the wrong dimension — no crash, wrong output).
     Regression detection across code changes. -->

## Relationship to InferenceTrace

<!-- TODO: DualPipeline uses InferenceTrace internally to capture per-layer
     statistics. The trace diff engine (Phase 14) generalizes this to
     arbitrary trace pairs, not just A-vs-B comparisons. -->
