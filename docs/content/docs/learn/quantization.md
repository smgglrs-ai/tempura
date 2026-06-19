+++
title = "19. Quantization"
description = "What rounding preserves and destroys — group quantization, error bounds, and provable properties."
weight = 190
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Why quantize

A 7B parameter model in float32 requires 28 GB of memory — more than most GPUs can hold. Quantization reduces precision, trading some accuracy for dramatic savings:

| Precision | Bits per parameter | Memory (7B model) | Relative quality |
|-----------|-------------------|-------------------|-----------------|
| f32 | 32 | 28 GB | Baseline |
| f16/bf16 | 16 | 14 GB | ~Equal |
| int8 | 8 | 7 GB | Minimal loss |
| int4 | 4 | 3.5 GB | Noticeable on benchmarks, often acceptable in practice |

The goal: find the lowest precision where the model's behavior is "close enough" to the full-precision version. "Close enough" is precisely what DualPipeline measures.

## Symmetric quantization

The simplest scheme maps a floating-point value to an integer level and back:

**Quantize:**
$$
q = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right), -2^{b-1}, 2^{b-1} - 1\right)
$$

**Dequantize:**
$$
\hat{x} = q \cdot s
$$

where $s$ is the scale factor and $b$ is the bit width. For a tensor with maximum absolute value $|x|_{\max}$:

$$
s = \frac{|x|_{\max}}{2^{b-1} - 1}
$$

## Bounded error

The maximum quantization error for any single value is:

$$
|x - \hat{x}| \leq \frac{s}{2} = \frac{|x|_{\max}}{2(2^{b-1} - 1)}
$$

This is a hard bound — not a probabilistic estimate. For int8 ($b = 8$, $2^{b-1} - 1 = 127$) with $|x|_{\max} = 1.0$:

$$
|x - \hat{x}| \leq \frac{1.0}{254} \approx 0.0039
$$

For int4 ($b = 4$, $2^{b-1} - 1 = 7$):

$$
|x - \hat{x}| \leq \frac{1.0}{14} \approx 0.071
$$

The per-value error is bounded, but errors accumulate through matrix multiplication. A quantized matmul with hidden dimension $d$ accumulates $d$ quantization errors — this is where group quantization helps.

## Group quantization

Per-tensor quantization uses a single scale for the entire tensor. If the tensor has outliers (a few values much larger than the rest), the scale is dominated by the outliers and most values are quantized to a small range of integers — wasting precision.

Group quantization splits the tensor into groups of $g$ consecutive elements (typically 32–128), each with its own scale:

$$
s_j = \frac{\max_{i \in \text{group}_j} |x_i|}{2^{b-1} - 1}
$$

This adapts the quantization range to local statistics, dramatically improving accuracy at the cost of storing one scale per group.

**The group boundary invariant.** Index arithmetic must never read across groups — group $j$'s scale must only be applied to elements $[jg, (j+1)g)$. This is a safety property that Kani can prove.

## Monotonicity

For symmetric quantization within a single group, the mapping preserves order:

$$
x > y \implies \text{quant}(x) \geq \text{quant}(y)
$$

(with equality when $x$ and $y$ round to the same integer level).

**Why this matters:** softmax converts logits into a probability distribution, and the model's token preferences are encoded in the relative ordering of logits. If quantization reversed the order of two logits, the model would prefer the wrong token. Monotonicity guarantees that the largest logit remains the largest after quantization — greedy decoding produces the same token.

For top-k and top-p sampling, monotonicity ensures the top-k set after quantization is a subset of the top-k set before quantization (with possible ties at the boundary). The sampling distribution may shift, but the set of "plausible" tokens is preserved.

## Accumulator overflow

Quantized matrix multiplication replaces f32 multiply-accumulate with integer operations. For int4 × int4 with int32 accumulation and group size $g$:

$$
\text{acc}_{\max} = g \cdot (2^{b-1} - 1)^2 = g \cdot 7^2 = 49g
$$

For $g = 128$: $\text{acc}_{\max} = 6272$, well within int32 range ($2^{31} - 1 \approx 2.1 \times 10^9$).

For int8 × int8: $\text{acc}_{\max} = g \cdot 127^2 = 16129g$. At $g = 128$: $\text{acc}_{\max} = 2{,}064{,}512$ — still far from overflow.

This is provable: for any group size up to the model's hidden dimension, the accumulator cannot overflow. Kani can verify this bound.

## GPTQ, AWQ, and GGUF

These are quantization *methods* — algorithms for choosing which values to round and how to minimize the impact:

**GPTQ** (Frantar et al., 2023): quantizes weights column-by-column, using second-order information (the Hessian) to adjust remaining weights after each column is quantized. Achieves near-lossless int4 on many models.

**AWQ** (Lin et al., 2024): identifies "salient" weight channels (those that matter most for accuracy) and protects them by scaling before quantization. Simpler than GPTQ, competitive quality.

**GGUF**: a file format for quantized models, used by llama.cpp and compatible tools. Supports multiple quantization types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, etc.) with per-group scales and optional zero points.

tempura can load and dequantize all three formats. The quantized kernel traits (Phase 15) will enable native quantized inference without dequantization.

## What we prove

Kani verifies (planned, Phase 15):

- **Bounded error**: $|\hat{x} - x| \leq s/2$ for symmetric int8 dequantization
- **Monotonicity**: $x > y \implies \text{quant}(x) \geq \text{quant}(y)$ within the same group
- **Accumulator safety**: int32 accumulator does not overflow for group sizes up to 128
- **Group boundary safety**: index arithmetic never reads across group boundaries

## What we do NOT prove

**Accuracy impact on model quality.** Whether int4 quantization maintains acceptable perplexity is a property of the model and the quantization method, not the dequantization kernel. This requires empirical evaluation.

**Optimal group size.** The best group size balances accuracy (smaller groups = better) against overhead (more scales to store and process). This is a hyperparameter, not a verifiable property.

## Implementation

Quantization types are in [`engine/src/weights.rs`](https://github.com/smgglrs-ai/tempura/tree/main/engine/src/weights.rs). The `MatMulQ` and `Dequantize` traits will be defined in tempura-kernel-api as part of Phase 15 (Quantized Verification).
