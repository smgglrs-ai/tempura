+++
title = "Quantization"
description = "What rounding preserves and destroys — group quantization, error bounds, and provable properties."
weight = 50
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Why quantize

A 7B parameter model in float32 requires 28 GB of memory. Quantization reduces precision — typically to 8-bit or 4-bit integers — shrinking memory by 4-8x and enabling inference on smaller hardware. The tradeoff: some information is lost in the rounding.

## Symmetric quantization

The simplest scheme maps a floating-point range to integer levels:

$$
q = \text{round}\left(\frac{x}{s}\right), \quad \hat{x} = q \cdot s
$$

where $s = \frac{\max(|x|)}{2^{b-1} - 1}$ is the scale factor for $b$-bit quantization. The maximum quantization error is bounded:

$$
|x - \hat{x}| \leq \frac{s}{2}
$$

This bound is what makes quantization analyzable — the error is deterministic and bounded, not random.

## Group quantization

<!-- TODO: Per-tensor vs per-channel vs per-group scaling.
     Why groups (typically 32-128 elements) balance accuracy and overhead.
     The group boundary invariant: index arithmetic must never read across groups.
     GPTQ, AWQ, GGUF approaches. -->

## Monotonicity

<!-- TODO: For most quantization schemes, x > y implies quant(x) >= quant(y)
     within the same group. Why this matters: softmax ordering is preserved,
     so the model's token preferences are maintained even after quantization.
     When monotonicity breaks (asymmetric quantization edge cases). -->

## Accumulator overflow

<!-- TODO: Quantized matmul uses int32 accumulators. For group_size g and
     hidden_dim d, the maximum accumulator value is bounded by
     g * max_int4^2. Prove this doesn't overflow int32. -->

## What we prove

<!-- TODO (p18-foundations-connection): Kani proofs for bounded error,
     monotonicity, accumulator safety, group boundary indexing. -->

## What we do NOT prove

<!-- TODO (p18-foundations-connection): Accuracy impact on model quality (empirical),
     optimal group size selection (hyperparameter tuning). -->

## Implementation

Quantization types are in [`engine/src/weights.rs`](https://github.com/smgglrs-ai/tempura/tree/main/engine/src/weights.rs). The `MatMulQ` and `Dequantize` traits will be defined in tempura-kernel-api (Phase 15).
