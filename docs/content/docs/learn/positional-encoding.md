+++
title = "17. Positional Encoding"
description = "Rotary Position Embeddings (RoPE) — the rotation matrix, frequency bands, and length extrapolation."
weight = 170
template = "docs/page.html"

[extra]
part = "kernel-math"
toc = true
math = true
+++

## Why position matters

Attention is permutation-invariant by construction — $\text{softmax}(QK^T)V$ produces the same output regardless of token order. But language is sequential: "the cat sat on the mat" means something different from "the mat sat on the cat." Position encodings inject ordering information into the representation.

Early transformers added a fixed position vector to the input embedding. RoPE takes a different approach: instead of adding position to the representation, it encodes position in the *rotation* of query and key vectors, so that their dot product naturally depends on relative position.

## The rotation idea

Consider two dimensions $(x_1, x_2)$ of a vector. RoPE rotates this pair by an angle proportional to the token's position $m$:

$$
\begin{pmatrix} x_1' \\\\ x_2' \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\\\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_1 \\\\ x_2 \end{pmatrix}
$$

The key insight: when we compute the dot product between a query at position $m$ and a key at position $n$, the rotation matrices combine:

$$
\langle R_m q, R_n k \rangle = \langle R_{m-n} q, k \rangle
$$

The dot product depends only on the *relative position* $m - n$, not the absolute positions. Token 5 attending to token 3 computes the same relative attention as token 105 attending to token 103.

## Frequency bands

RoPE applies different rotation speeds to different dimension pairs. For dimension pair $i$, the rotation frequency is:

$$
\theta_i = \frac{1}{10000^{2i/d}}
$$

This creates a spectrum of frequencies:

- **Low-frequency pairs** (large $\theta$, small $i$): rotate slowly. They encode coarse position — distinguishing nearby vs distant tokens.
- **High-frequency pairs** (small $\theta$, large $i$): rotate quickly. They encode fine position — distinguishing adjacent tokens.

The analogy to Fourier analysis is deliberate: just as a Fourier transform represents a signal at multiple frequency scales, RoPE represents position at multiple scales simultaneously. The base frequency (10000) determines the longest "wavelength" — positions beyond this distance are less distinguishable.

## The full RoPE formula

The mathematical definition pairs adjacent dimensions $(x_{2i}, x_{2i+1})$:

$$
f(x, m)_{2i} = x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i)
$$
$$
f(x, m)_{2i+1} = x_{2i} \sin(m\theta_i) + x_{2i+1} \cos(m\theta_i)
$$

This is equivalent to applying $d/2$ independent 2D rotation matrices, one per dimension pair.

## Split-half layout

tempura (like LLaMA and most modern implementations) uses a **split-half** layout instead of the interleaved layout above. Instead of pairing $(x_0, x_1)$, $(x_2, x_3)$, ..., it pairs $(x_0, x_{d/2})$, $(x_1, x_{d/2+1})$, ...:

$$
f(x, m)_i = x_i \cos(m\theta_i) - x_{i + d/2} \sin(m\theta_i)
$$
$$
f(x, m)_{i + d/2} = x_i \sin(m\theta_i) + x_{i + d/2} \cos(m\theta_i)
$$

for $i \in [0, d/2)$. The mathematical properties are identical — both are 2D rotations of the same pairs — but the memory access pattern differs. The split-half layout is more cache-friendly on modern hardware because the two halves of each pair are separated by exactly `head_dim / 2` elements, enabling vectorized loads.

RoPE is applied to Q and K vectors *after* projection and *before* the dot product. The `RoPE::forward` signature takes position indices per token, and the base frequency $\theta$ as parameters.

## Position at zero

At position $m = 0$, all rotation angles are zero: $\cos(0) = 1$, $\sin(0) = 0$. The rotation matrix is the identity — the vector is unchanged. This means the first token's Q and K vectors are the raw projections, with no positional information added.

This is a useful property for verification: the identity at position zero provides a simple test case, and tempura's tests verify it.

## Length extrapolation

RoPE generalizes to unseen sequence lengths better than learned positional embeddings because the rotation operation is defined for any position $m$ — there's no embedding table that stops at a maximum length.

However, extrapolation is not free. At positions much longer than the training length, the high-frequency dimensions have rotated many times, and the model hasn't learned to handle these rotation angles. Several techniques extend the effective context:

**NTK-aware scaling** adjusts the base frequency to spread rotations more evenly across the extended range. Instead of $\theta_i = 10000^{-2i/d}$, use $\theta_i = (\alpha \cdot 10000)^{-2i/d}$ where $\alpha > 1$ stretches the wavelength.

**YaRN** (Yet another RoPE extensioN) combines NTK scaling with attention scaling and a temperature adjustment, achieving better extrapolation with minimal fine-tuning.

tempura's `RoPE` trait accepts the base frequency as a parameter, making these scaling schemes configurable without code changes.

## What we prove

Kani verifies two properties of the RoPE implementation:

**Index safety for the split-half pattern.** tempura's RoPE uses the split-half layout: for a `head_dim`-sized vector, dimension $d$ is paired with dimension $d + \text{head\_dim}/2$. The harness proves that for any valid `batch`, `seq_len`, `num_heads`, and `head_dim` (constrained to even values) within bounded ranges, both `offset + d` and `offset + d + half_dim` stay within the buffer length.

**Position array bounds.** The harness also proves that the position lookup `positions[b * seq_len + s]` never exceeds the positions array length.

## What we do NOT prove

**Trigonometric accuracy.** The correctness of `f32::cos()` and `f32::sin()` is a property of the CPU/compiler, not something Kani can reason about. Property-based tests verify that position-zero produces the identity (within f32 epsilon).

**Extrapolation quality.** Whether NTK-scaled RoPE maintains attention quality at extended lengths is an empirical question, dependent on the trained model.

## Implementation

The CPU reference is in [`kernels/src/rope.rs`](https://github.com/smgglrs-ai/tempura/tree/main/kernels/src/rope.rs). The `RoPE` trait takes a `positions: &[u32]` array (one position per token per batch element) and the base frequency `theta: f32`. The implementation uses the split-half layout, processing the first half and second half of each head vector simultaneously. The CUDA kernel launches one thread per element for single-batch inputs.
