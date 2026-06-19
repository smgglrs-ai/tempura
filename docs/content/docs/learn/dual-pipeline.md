+++
title = "23. DualPipeline"
description = "Empirical equivalence testing — what formal proofs miss and how runtime comparison catches it."
weight = 230
template = "docs/page.html"

[extra]
part = "verification"
toc = true
math = true
+++

## The gap formal proofs leave

[Kani](../bounded-model-checking/) proves *safety*: no panics, no out-of-bounds access, no integer overflow. It does not prove *numerical accuracy*: that the GPU attention kernel produces the same output as the CPU reference.

Two implementations can both be safe yet produce different numbers. This happens routinely:

- **Different reduction order.** CPU adds left-to-right; GPU adds in a tree pattern. With f32, $(a + b) + c \neq a + (b + c)$ at the limits of precision.
- **Different precision.** f16 matmul accumulating in f32 vs f32 throughout.
- **Implementation bugs.** A kernel that applies RoPE to the wrong dimension pair — no crash, wrong output.

DualPipeline fills this gap by running two implementations on identical inputs and comparing outputs.

## How it works

```
Token IDs + seed
       │
       ├──→ Pipeline A (e.g., CPU reference)  ──→ InferenceTrace A
       │
       └──→ Pipeline B (e.g., CUDA kernels)   ──→ InferenceTrace B
       
Compare traces layer-by-layer, token-by-token
```

Both pipelines receive the same token IDs, the same model weights, and the same random seed. They produce identical [InferenceTraces](../../trace-tooling/) (at Summary or Full level). The comparison examines three levels:

### Per-layer comparison

For each transformer layer, compare the pre-normalization and post-normalization statistics:

$$
\Delta_{\text{layer}} = \max_i |a_i - b_i|
$$

where $a_i$ and $b_i$ are the statistics (mean, variance, max absolute value) of layer $i$'s activations in pipelines A and B.

### Logits comparison

After the final layer, compare the output logit distributions:

$$
\Delta_{\text{logits}} = \sqrt{\sum_j (z^A_j - z^B_j)^2}
$$

This L2 distance measures how far the two implementations diverge in their token predictions.

### Token comparison

With deterministic sampling (same seed, same RNG), the same logits produce the same token. A token mismatch means the logit distributions diverged enough to cross a sampling boundary — a significant discrepancy.

## Epsilon thresholds

Exact equality is the wrong standard for floating-point comparison. Two correct implementations of the same math will differ at the precision limit. The question is: how much difference is acceptable?

**Elementwise operations** (RoPE, RMSNorm normalization step): differences up to $\sim 10^{-7}$ (f32 machine epsilon). These perform one or two floating-point operations per element.

**Reductions** (RMSNorm sum-of-squares, softmax): differences up to $\sim 10^{-6}$. Summing $d$ values accumulates roundoff error proportional to $\sqrt{d} \cdot \epsilon_{\text{machine}}$.

**Matrix multiplication**: differences up to $\sim 10^{-5}$ for large dimensions. A matmul with inner dimension $k$ accumulates $k$ multiply-add operations, each with roundoff.

**Full forward pass** (32 layers of attention + FFN): differences up to $\sim 10^{-4}$ in per-layer statistics. Errors compound across layers, though normalization between layers prevents unbounded growth.

tempura's DualPipeline comparison uses configurable epsilon thresholds per metric. The default thresholds are conservative — passing means the implementations agree to the limits of f32 precision.

## What DualPipeline catches

**Precision loss from GPU parallelism.** GPU kernels often reduce in a tree pattern for parallelism. The tree order differs from the sequential order, producing different f32 results. DualPipeline detects when this divergence exceeds the expected precision loss.

**Quantization accuracy.** Is int4 matmul "close enough" to f32? DualPipeline gives a precise answer: the max absolute difference and L2 distance between the quantized and full-precision outputs.

**Implementation bugs that are safe but wrong.** A kernel that swaps the Q and K inputs to attention will not crash (both are the same shape) but will produce wrong outputs. DualPipeline catches this immediately.

**Regression detection.** After a kernel optimization, run DualPipeline to verify the output hasn't changed beyond the expected precision. This is the floating-point equivalent of a regression test.

## What DualPipeline does not catch

**Bugs present in both implementations.** If the CPU reference has a bug and the GPU implementation faithfully reproduces it, DualPipeline sees no divergence. This is why Kani proofs and property tests exist — they check properties of individual implementations, not just agreement between two.

**Performance regressions.** DualPipeline compares outputs, not execution time. A kernel that produces correct results but takes 10x longer will pass DualPipeline.

## Relationship to InferenceTrace

DualPipeline uses [InferenceTrace](../../trace-tooling/) internally. Each pipeline produces a trace at Summary level (per-layer norm stats, per-token sampling decisions), and the comparison operates on these structured traces.

The trace diff engine (Phase 14) generalizes this to arbitrary trace pairs — not just A-vs-B implementation comparisons, but also before-vs-after comparisons across code changes, or CPU-vs-GPU-vs-NPU three-way comparisons.

## Using DualPipeline

Via the API:

```
POST /tempura/v1/compare
{
  "token_ids": [1, 2, 3, 4, 5],
  "seed": 42,
  "trace_level": "summary"
}
```

Returns:

```json
{
  "layers": [
    {"index": 0, "max_abs_diff": 2.4e-7, "l2_distance": 1.1e-6},
    {"index": 1, "max_abs_diff": 3.1e-7, "l2_distance": 1.4e-6}
  ],
  "logits_l2": 4.2e-5,
  "tokens_match": true,
  "verdict": "equivalent"
}
```

A `"verdict": "divergent"` result includes the first layer where divergence exceeded the threshold — the starting point for debugging.
