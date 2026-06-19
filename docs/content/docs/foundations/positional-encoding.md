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

## Rotary Position Embeddings

RoPE encodes position by rotating the query and key vectors in 2D subspaces. For a vector at position $m$, RoPE applies:

$$
f(x, m) = \begin{pmatrix} x_1 \\\\ x_2 \\\\ x_3 \\\\ x_4 \\\\ \vdots \end{pmatrix} \odot \begin{pmatrix} \cos m\theta_1 \\\\ \cos m\theta_1 \\\\ \cos m\theta_2 \\\\ \cos m\theta_2 \\\\ \vdots \end{pmatrix} + \begin{pmatrix} -x_2 \\\\ x_1 \\\\ -x_4 \\\\ x_3 \\\\ \vdots \end{pmatrix} \odot \begin{pmatrix} \sin m\theta_1 \\\\ \sin m\theta_1 \\\\ \sin m\theta_2 \\\\ \sin m\theta_2 \\\\ \vdots \end{pmatrix}
$$

where $\theta_i = 10000^{-2i/d}$ defines the frequency bands.

## The rotation intuition

<!-- TODO: Each pair of dimensions forms a 2D plane. RoPE rotates the vector
     in that plane by an angle proportional to position. The dot product
     between two rotated vectors depends only on their relative position
     (m - n), not absolute position. This is the key property. -->

## Frequency bands

<!-- TODO: Low-frequency bands (large theta) encode coarse position (nearby vs far).
     High-frequency bands (small theta) encode fine position (adjacent tokens).
     The base frequency (10000) and how it relates to context length.
     NTK-aware scaling for length extrapolation. -->

## Length extrapolation

<!-- TODO: Why RoPE generalizes to unseen lengths better than learned positional
     embeddings. The limitations. YaRN, dynamic NTK scaling. -->

## What we prove

<!-- TODO (p18-foundations-connection): Link to Kani harness for RoPE.
     Index arithmetic safety in the paired-element rotation. -->

## What we do NOT prove

<!-- TODO (p18-foundations-connection): Trigonometric accuracy (property tests),
     extrapolation quality (empirical). -->

## Implementation

The CPU reference is in [`kernels/src/rope.rs`](https://github.com/smgglrs-ai/tempura/tree/main/kernels/src/rope.rs). The `RoPE` trait is defined in the kernel API crate.
