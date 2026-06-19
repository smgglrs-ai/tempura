+++
title = "Positional Encoding"
description = "Rotary Position Embeddings (RoPE) — the rotation matrix, frequency bands, and length extrapolation."
weight = 30
template = "docs/page.html"

[extra]
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

For a $d$-dimensional vector at position $m$, RoPE processes pairs of dimensions:

$$
f(x, m)_{2i} = x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i)
$$
$$
f(x, m)_{2i+1} = x_{2i} \sin(m\theta_i) + x_{2i+1} \cos(m\theta_i)
$$

This is equivalent to applying $d/2$ independent 2D rotation matrices, one per dimension pair.

**In tempura**, RoPE is applied to Q and K vectors *after* projection and *before* the dot product. The `RoPE::forward` signature takes the head dimension and position offset, applying rotations to each pair of elements.

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

Kani verifies the index arithmetic in the RoPE implementation: for any valid `head_dim` (must be even) and `seq_len` within bounded ranges, the paired-element rotation loop never accesses memory beyond the Q/K buffer lengths. The proof covers the `2i` and `2i+1` indexing pattern that processes dimension pairs.

## What we do NOT prove

**Trigonometric accuracy.** The correctness of `f32::cos()` and `f32::sin()` is a property of the CPU/compiler, not something Kani can reason about. Property-based tests verify that position-zero produces the identity (within f32 epsilon).

**Extrapolation quality.** Whether NTK-scaled RoPE maintains attention quality at extended lengths is an empirical question, dependent on the trained model.

## Implementation

The CPU reference is in [`kernels/src/rope.rs`](https://github.com/smgglrs-ai/tempura/tree/main/kernels/src/rope.rs). The `RoPE` trait is defined in the kernel API crate. The implementation processes dimension pairs sequentially, computing $\cos(m\theta_i)$ and $\sin(m\theta_i)$ for each pair and applying the 2D rotation.
