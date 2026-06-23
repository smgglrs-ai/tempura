+++
title = "Positional Encoding"
description = "Rotary Position Embeddings (RoPE) — the rotation matrix, frequency bands, and extrapolation."
weight = 30
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Why position matters

Attention is permutation-invariant: $\text{softmax}(QK^T)V$ produces the same output regardless of token order. Positional encoding breaks this symmetry by encoding where each token sits in the sequence.

## Rotary Position Embeddings (RoPE)

RoPE encodes position by rotating query and key vectors. For position $m$ and dimension pair $(2i, 2i+1)$:

$$
\begin{bmatrix} q_{2i}' \\ q_{2i+1}' \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
$$

where the frequency for dimension pair $i$ is:

$$
\theta_i = \theta_\text{base}^{-2i/d}
$$

with $\theta_\text{base} = 10000$ (LLaMA) or $1000000$ (LLaMA 3, Qwen3).

## Key properties

**Relative position encoding**: The dot product $q_m \cdot k_n$ depends only on $m - n$, not on absolute positions. This is because rotation is a group action: $R_m^T R_n = R_{n-m}$.

**Norm preservation**: Rotation preserves L2 norm ($\|R_m q\| = \|q\|$), so RoPE doesn't change the magnitude of query/key vectors — it only changes their direction.

**Position zero is identity**: At $m = 0$, $\cos(0) = 1$ and $\sin(0) = 0$, so the rotation matrix is the identity. The first token's vectors are unchanged.

## Frequency bands

Low-frequency dimensions (small $i$, large $\theta_i$) capture long-range position differences. High-frequency dimensions (large $i$, small $\theta_i$) capture fine-grained local position. With $\theta_\text{base} = 10000$ and $d = 128$:
- Dimension pair 0: period $= 2\pi / \theta_0 = 2\pi$ (changes every token)
- Dimension pair 63: period $= 2\pi \cdot 10000^{63/64} \approx 56000$ tokens

## Worked example

Position $m = 1$, $d = 4$, $\theta_\text{base} = 10000$:

Frequencies: $\theta_0 = 10000^0 = 1$, $\theta_1 = 10000^{-0.5} = 0.01$

$$
q = [1, 0, 1, 0] \xrightarrow{RoPE}
[\cos(1), -\sin(1), \cos(0.01), -\sin(0.01)]
\approx [0.540, -0.841, 1.000, -0.010]
$$

The first pair rotates significantly ($\theta_0 = 1$ rad), while the second barely moves ($\theta_1 = 0.01$ rad).

## What we prove

- **`rope_indices_in_bounds`**: Position lookup and rotation pair indexing are within buffer bounds for any valid (batch, seq_len, num_heads, head_dim, position).

## What we do NOT prove

- **Norm preservation**: $\|R_m q\| = \|q\|$ is validated by proptest (`prop_rotation_preserves_l2_norm`), not formally proven — Kani cannot reason about `cos²(θ) + sin²(θ) = 1`.
- **Position-zero identity**: $R_0 = I$ validated by proptest (`prop_position_zero_is_identity`), not Kani-proven.
- **Relative position encoding**: The property that $q_m \cdot k_n$ depends only on $m - n$ is a mathematical consequence of the rotation group structure — not testable per se, but follows from the correct implementation of the rotation matrix.

## Source code

- **Trait**: [`kernel-api/src/lib.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernel-api/src/lib.rs) — `RoPE` trait with `positions` array
- **CPU implementation**: [`kernels/src/rope.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/rope.rs) — `CpuRoPE`, in-place rotation
- **Kani proofs**: Same file — proves index bounds for position lookup and rotation pairs

## See also

- [Attention](../attention/) — RoPE is applied to Q and K before computing attention scores
- [Normalization](../normalization/) — applied before the QKV projection that produces the vectors RoPE rotates
