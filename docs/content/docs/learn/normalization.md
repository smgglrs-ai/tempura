+++
title = "16. Normalization"
description = "RMS normalization — the variance invariant, numerical stability, and why LLMs skip mean subtraction."
weight = 160
template = "docs/page.html"

[extra]
part = "kernel-math"
toc = true
math = true
+++

## The problem normalization solves

Neural network layers compose: each layer's output becomes the next layer's input. Without normalization, activations can drift — growing unboundedly large or collapsing toward zero — making training unstable and inference unpredictable.

Consider a transformer with 32 layers. If each layer scales activations by even 1.01x on average, the final output is $1.01^{32} \approx 1.37$ times the input. At 1.1x, it's $1.1^{32} \approx 21$x. At 0.9x, activations shrink to $0.9^{32} \approx 0.03$x — effectively vanishing. Normalization constrains the scale at each layer boundary, keeping the signal in a well-behaved range regardless of depth.

## Layer Normalization vs RMS Normalization

**Layer Normalization** (Ba et al., 2016) centers and scales:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

where $\mu = \frac{1}{d}\sum_i x_i$ is the mean, $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$ is the variance, $\gamma$ and $\beta$ are learned parameters, and $\epsilon$ prevents division by zero.

**RMS Normalization** (Zhang & Sennrich, 2019) drops the mean subtraction and the bias:

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x) + \epsilon} \cdot \gamma
$$

where:

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}
$$

## Why no mean subtraction?

Zhang & Sennrich showed empirically that re-centering (subtracting the mean) provides negligible benefit in transformers while adding computational cost. The intuition:

**The scale matters more than the center.** What destabilizes deep networks is activations growing or shrinking multiplicatively. The mean can shift without causing the same instability — a vector $[10, 11, 12]$ and $[0, 1, 2]$ have the same variance but different means, and both are equally well-conditioned for the next layer.

**Computational saving.** RMSNorm requires one reduction (sum of squares) instead of two (sum for mean, then sum of squared deviations). This is a meaningful difference on hardware where memory bandwidth limits reduction throughput.

**Fewer parameters.** No bias term $\beta$ — just the scale $\gamma$. Fewer parameters, fewer opportunities for overfitting.

LLaMA, Mistral, Gemma, Qwen, and most modern LLMs use RMSNorm. tempura implements it as the default normalization.

## The variance invariant

After RMSNorm, the RMS of the output (ignoring $\gamma$) is approximately 1:

$$
\text{RMS}\left(\frac{x}{\text{RMS}(x)}\right) = \frac{\text{RMS}(x)}{\text{RMS}(x)} = 1
$$

This means every layer receives inputs with unit RMS, regardless of what the previous layer produced. The learned $\gamma$ parameter then re-scales each dimension independently — the model controls the scale of each feature, rather than the scale being an accident of depth.

**Why this matters for verification:** the invariant is a mathematical property of the normalization formula, not something that needs to be learned or tested. The implementation needs to compute $\text{RMS}(x)$ accurately — Kani proves the index arithmetic is safe, and DualPipeline verifies the numerical result.

## Numerical stability

The $\epsilon$ term (typically $10^{-5}$ or $10^{-6}$) prevents division by zero when the input is all zeros. In practice, it also prevents numerical instability when the RMS is very small — dividing by a near-zero value amplifies any noise in the input.

**Sum of squares overflow.** For long hidden dimensions ($d = 8192$ in 70B models), the sum $\sum x_i^2$ can be large. With f32, the maximum representable value is ~$3.4 \times 10^{38}$. If each $x_i \approx 200$ (already unusually large for normalized activations), $8192 \times 200^2 = 3.3 \times 10^{8}$ — well within range. Overflow is not a practical concern, but it's the kind of property that Kani can bound.

**Interaction with quantization.** When activations are quantized (e.g., f16 or int8), the sum of squares loses precision. RMSNorm should be computed in f32 even when activations are stored at lower precision — tempura's implementation always accumulates in f32.

## Pre-norm vs post-norm

Transformers place normalization either before each sublayer (pre-norm) or after (post-norm):

- **Pre-norm** (used by LLaMA, GPT-2+): $x + \text{Sublayer}(\text{RMSNorm}(x))$
- **Post-norm** (original Transformer): $\text{RMSNorm}(x + \text{Sublayer}(x))$

Pre-norm is more stable for training deep networks because the residual connection carries the unnormalized signal. tempura implements pre-norm following the LLaMA architecture.

## What we prove

Kani verifies the index arithmetic in the RMSNorm implementation: for any valid `hidden_size` and `num_tokens` within bounded ranges, the sum-of-squares loop and the normalization loop never access memory beyond the input/output/weight buffer lengths.

## What we do NOT prove

**Numerical accuracy of the square root.** Kani reasons about integers and simple arithmetic. The `f32::sqrt()` call is trusted — its accuracy is a property of the hardware/compiler, validated by DualPipeline comparison.

**Epsilon sufficiency.** Whether $\epsilon = 10^{-5}$ is large enough for all practical inputs is an empirical question. Adversarial inputs could produce near-zero RMS values where a larger epsilon would be needed.

## Implementation

The CPU reference is in [`kernels/src/rmsnorm.rs`](https://github.com/smgglrs-ai/tempura/tree/main/kernels/src/rmsnorm.rs). The `RMSNorm` trait is defined in the kernel API crate. The implementation uses a two-pass approach: first compute the sum of squares, then normalize each element.
