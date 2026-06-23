+++
title = "Sampling"
description = "Temperature as energy scaling, top-k/top-p as distribution truncation, the probability space."
weight = 40
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## From logits to tokens

The model's final layer produces a logit vector $z \in \mathbb{R}^V$ (one score per vocabulary token). Sampling converts this to a probability distribution and selects a token.

## Softmax

Softmax maps logits to probabilities:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

Properties: $p_i > 0$ for all $i$, and $\sum p_i = 1$. Larger logits get exponentially more probability mass.

**Numerical stability**: Computing $e^{z_i}$ directly overflows for large $z$. The standard trick: subtract $\max(z)$ before exponentiation. This doesn't change the probabilities (the constant cancels) but keeps values in $[0, 1]$.

## Temperature

Temperature $\tau$ scales logits before softmax:

$$
p_i = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}
$$

- $\tau = 1$: standard softmax
- $\tau \to 0$: concentrates all mass on the argmax (greedy/deterministic)
- $\tau > 1$: flattens the distribution (more random)

Temperature is energy scaling from statistical mechanics: lower temperature → lower entropy → more peaked distribution.

## Top-k sampling

Keep only the $k$ highest-probability tokens, zero out the rest, renormalize:

$$
p_i' = \begin{cases} p_i / \sum_{j \in \text{top-}k} p_j & \text{if } i \in \text{top-}k \\ 0 & \text{otherwise} \end{cases}
$$

This prevents sampling rare, low-quality tokens while preserving diversity among the top candidates.

## Top-p (nucleus) sampling

Keep the smallest set of tokens whose cumulative probability exceeds $p$:

$$
\text{top-}p = \min \left\{ S \subseteq V : \sum_{i \in S} p_i \geq p \right\}
$$

where tokens are considered in decreasing probability order.

Top-p adapts to the shape of the distribution: when the model is confident (peaked), fewer tokens are included. When uncertain (flat), more are kept.

## Worked example

Logits $z = [2.0, 1.0, 0.5, 0.1]$, temperature $\tau = 1.0$:

$$
e^z = [7.389, 2.718, 1.649, 1.105], \quad \sum = 12.861
$$

$$
p = [0.574, 0.211, 0.128, 0.086]
$$

With top-k=2: keep tokens 0 and 1, renormalize: $p' = [0.731, 0.269, 0, 0]$.

With top-p=0.8: cumulative $[0.574, 0.786, 0.914, ...]$ — need tokens 0, 1, 2 to exceed 0.8.

## Greedy decoding

Temperature $\tau = 0$ (or equivalently, top-k=1) always selects the highest-probability token. Deterministic — same input always produces the same output. Used for benchmarking and reproducibility.

## Source code

- **Trait**: [`kernel-api/src/lib.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernel-api/src/lib.rs) — `Sampler` trait with `apply` (logits → probs) and `sample_token` (probs → token)
- **CPU implementation**: [`kernels/src/sampling.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/sampling.rs) — `CpuSampler` with softmax, top-k, top-p, temperature
- **Kani proofs**: Same file — proves top-k index bounds and fallback token validity
- **Penalties**: [`kernel-api/src/lib.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernel-api/src/lib.rs) — `SamplingParams::apply_penalties` for presence/frequency penalties

## See also

- [Attention](../attention/) — produces the logits that sampling operates on
- [Quantization](../quantization/) — quantization can shift logit distributions, affecting sampling behavior
