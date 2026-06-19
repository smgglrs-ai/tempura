+++
title = "Normalization"
description = "RMS normalization — the variance invariant, numerical stability, and why LLMs skip mean subtraction."
weight = 20
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## The problem normalization solves

Neural network layers compose: each layer's output becomes the next layer's input. Without normalization, activations can drift — growing unboundedly large or collapsing toward zero — making training unstable and inference unpredictable.

Normalization constrains the scale of activations at each layer boundary, ensuring that the signal stays in a numerically well-behaved range regardless of depth.

## RMS Normalization

tempura implements RMS (Root Mean Square) normalization, the variant used by LLaMA, Mistral, and most modern LLMs. Unlike Layer Normalization, RMS Norm skips the mean subtraction step:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$

where $x \in \mathbb{R}^d$ is the input vector, $\gamma \in \mathbb{R}^d$ is a learned scale parameter, and $\epsilon$ is a small constant (typically $10^{-5}$) for numerical stability.

## Why no mean subtraction?

<!-- TODO: The empirical finding that re-centering is unnecessary for transformer
     training stability. The computational saving. The invariance properties
     that RMS Norm preserves vs LayerNorm. Zhang & Sennrich (2019). -->

## The variance invariant

<!-- TODO: After RMSNorm, the RMS of the output is approximately 1 (scaled by gamma).
     Why this matters for downstream operations. The relationship to weight
     initialization schemes. -->

## Numerical stability

<!-- TODO: The epsilon term, why it matters for small activations, f32 precision
     considerations, the interaction with quantization. -->

## What we prove

<!-- TODO (p18-foundations-connection): Link to Kani harness for RMSNorm.
     What index arithmetic is proven safe. -->

## What we do NOT prove

<!-- TODO (p18-foundations-connection): Numerical accuracy of the sqrt,
     gamma scaling correctness (DualPipeline). -->

## Implementation

The CPU reference is in [`kernels/src/rmsnorm.rs`](https://github.com/smgglrs-ai/tempura/tree/main/kernels/src/rmsnorm.rs). The `RMSNorm` trait is defined in the kernel API crate.
