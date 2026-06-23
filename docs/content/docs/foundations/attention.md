+++
title = "Attention"
description = "Scaled dot-product attention — QKV projections, softmax, multi-head and grouped-query attention."
weight = 10
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Scaled dot-product attention

Given queries $Q$, keys $K$, and values $V$, attention computes a weighted average of values where the weights come from query-key similarity:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

The scale factor $\frac{1}{\sqrt{d_k}}$ prevents dot products from growing with dimension, which would push softmax into saturation (near-zero gradients).

## Multi-head attention

Instead of one large attention, split $Q$, $K$, $V$ into $h$ heads of dimension $d_k = d_\text{model} / h$:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.

Each head learns different attention patterns. The output projection $W^O$ combines them.

## Grouped-query attention (GQA)

LLaMA 2 70B, Mistral, and Qwen use GQA: fewer KV heads than query heads. If `num_heads = 32` and `num_kv_heads = 8`, each KV head serves 4 query heads. This reduces KV cache size by $\frac{\text{num\_heads}}{\text{num\_kv\_heads}}$ with minimal quality loss.

In the implementation, the KV head index for query head $h$ is:

$$
\text{kv\_h} = \lfloor h / (\text{num\_heads} / \text{num\_kv\_heads}) \rfloor
$$

## Causal masking

For autoregressive generation, position $i$ can only attend to positions $\leq i$. Positions $> i$ get score $-\infty$ before softmax, zeroing their attention weight.

## Worked example

2 tokens, 1 head, $d_k = 2$:

$$
Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
V = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

Scores (with causal mask, scale $1/\sqrt{2} \approx 0.707$):

$$
S = \begin{bmatrix} 0.707 & -\infty \\ 0 & 0.707 \end{bmatrix}
$$

After softmax: token 0 attends 100% to itself → output $[1, 2]$. Token 1 attends ~73% to itself, ~27% to token 0 → output $\approx [2.46, 3.46]$.

## Source code

- **Trait**: [`kernel-api/src/lib.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernel-api/src/lib.rs) — `Attention` trait with `forward` and `forward_paged`
- **CPU implementation**: [`kernels/src/attention.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/attention.rs) — `CpuAttention`, nested loops over batch/head/position
- **Kani proofs**: Same file, `mod kani_proofs` — proves Q/KV offset bounds and batch isolation

## See also

- [Normalization](../normalization/) — applied before and after attention in each layer
- [Positional encoding](../positional-encoding/) — RoPE applied to Q and K before attention
- [Sampling](../sampling/) — operates on the logits produced after all attention layers
