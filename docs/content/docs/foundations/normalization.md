+++
title = "Normalization"
description = "RMS normalization — the variance invariant, why no mean subtraction, numerical stability."
weight = 20
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## RMS normalization

LLaMA and most modern LLMs use RMSNorm instead of LayerNorm. It normalizes by the root mean square without subtracting the mean:

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma
$$

where:

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

The learnable weight $\gamma$ (one per dimension) scales the normalized output. The small constant $\epsilon$ (typically $10^{-5}$ or $10^{-6}$) prevents division by zero.

## Why not LayerNorm?

LayerNorm subtracts the mean before normalizing:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta
$$

RMSNorm drops the mean subtraction and the bias $\beta$. Empirically this works equally well for LLMs while being ~7% faster (one fewer reduction pass). The key insight: what matters is stabilizing the scale, not centering at zero.

## Numerical stability

The sum $\sum x_i^2$ can overflow for large $d$ if values are large. In practice, post-embedding and post-attention activations are bounded (typically $|x_i| < 100$), so $d \cdot 10000 < 10^7$ is well within f32 range. The $\epsilon$ prevents the pathological case where all $x_i = 0$.

## Worked example

$x = [3, 4]$, $\gamma = [1, 1]$, $\epsilon = 0$:

$$
\text{RMS} = \sqrt{\frac{9 + 16}{2}} = \sqrt{12.5} \approx 3.536
$$

$$
\text{output} = \left[\frac{3}{3.536}, \frac{4}{3.536}\right] \approx [0.849, 1.131]
$$

Verify: the output's RMS is $\sqrt{(0.849^2 + 1.131^2)/2} = \sqrt{(0.72 + 1.28)/2} = 1.0$. RMSNorm produces unit RMS — this is the invariant.

## What we prove

- **`rmsnorm_indices_in_bounds`**: Row indexing `x[row * hidden_size + i]` is within buffer bounds for any valid (tokens, hidden_size, i).
- **`rmsnorm_full_execution_no_panic`**: Complete `CpuRMSNorm::forward` execution does not panic for any valid bounded input (tokens 1–2, hidden 1–8).

## What we do NOT prove

- **Output has unit RMS**: Validated by proptest (`prop_output_has_unit_rms`), not formally proven — Kani cannot reason about `sqrt()` and division precision.
- **Numerical match to f64 reference**: Validated by `test_rmsnorm_matches_f64_reference`, which checks the f32 implementation matches an f64 computation within 1e-5.
- **Epsilon prevents division by zero**: The $\epsilon$ guard is structurally present in the code, but Kani doesn't prove the resulting `1.0 / rms` is finite — that's a floating-point property.

## Source code

- **Trait**: [`kernel-api/src/lib.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernel-api/src/lib.rs) — `RMSNorm` trait
- **CPU implementation**: [`kernels/src/rmsnorm.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/rmsnorm.rs) — `CpuRMSNorm`, in-place normalization
- **Kani proofs**: Same file — proves index bounds and full execution no-panic

## See also

- [Attention](../attention/) — RMSNorm is applied before each attention block
- [Quantization](../quantization/) — normalization affects activation magnitude, which impacts quantization range
