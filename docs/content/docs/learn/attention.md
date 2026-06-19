+++
title = "15. Attention"
description = "Scaled dot-product attention — the linear algebra, softmax, multi-head, and GQA."
weight = 150
template = "docs/page.html"

[extra]
part = "kernel-math"
toc = true
math = true
+++

## The core idea

Attention answers a question: given a query, which parts of the input are relevant, and how should they contribute to the output?

The mechanism is a weighted average. The weights are not learned directly — they are computed from the similarity between queries and keys, normalized by softmax into a probability distribution, then used to weight the values.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

where $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$, and $\frac{1}{\sqrt{d_k}}$ is a scaling factor that prevents dot products from growing large enough to push softmax into saturation.

**Step by step:**

1. **Score**: compute $S = QK^T \in \mathbb{R}^{n \times m}$. Each entry $S_{ij}$ is the dot product between query $i$ and key $j$ — a measure of how relevant position $j$ is to query $i$.

2. **Scale**: divide by $\sqrt{d_k}$. Without this, dot products grow proportionally to $d_k$, pushing softmax into regions where gradients vanish. The scaling keeps the variance of the scores approximately 1, regardless of dimension.

3. **Normalize**: apply softmax row-wise. Each row of $\text{softmax}(S / \sqrt{d_k})$ becomes a probability distribution over positions — the attention weights $\alpha_{ij}$.

4. **Aggregate**: multiply by $V$. The output for query $i$ is $\sum_j \alpha_{ij} v_j$ — a weighted combination of value vectors, where the weights are the attention scores.

## Why softmax?

Softmax converts arbitrary real-valued scores into a probability distribution:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Three properties make it the right choice:

**Positivity and normalization.** All outputs are positive and sum to 1. This means the attention output is a convex combination of the values — it stays within the convex hull of the value vectors, which is a useful inductive bias.

**Differentiability.** Unlike argmax (which is not differentiable), softmax provides smooth gradients everywhere. This is essential for training, though for inference it only matters that the forward computation is well-defined.

**Sharpness control.** Softmax with a temperature parameter $T$ gives:

$$
\text{softmax}(z / T)_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

As $T \to 0$, this approaches a one-hot vector (hard attention). As $T \to \infty$, it approaches a uniform distribution. The scaling factor $\frac{1}{\sqrt{d_k}}$ in the attention formula acts as a fixed temperature — see [Sampling](../sampling/) for how temperature is used at generation time.

**Numerical stability.** In practice, we compute $\text{softmax}(z - \max(z))$ to prevent overflow. Since $\text{softmax}(z + c) = \text{softmax}(z)$ for any constant $c$, subtracting the maximum value doesn't change the result but keeps the exponentials in a representable range.

## Multi-head attention

A single attention head has limited expressive power — it can only compute one set of attention weights per position. Multi-head attention runs multiple attention heads in parallel, each projecting the input into a different subspace:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

where each head is:

$$
\text{head}_i = \text{Attention}(XW^Q_i, XW^K_i, XW^V_i)
$$

with learned projection matrices $W^Q_i, W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$, and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$.

Typically $d_k = d_v = d_{\text{model}} / h$, so the total computation cost is similar to a single head with full dimensionality. The benefit: different heads can attend to different types of relationships (syntactic, semantic, positional) simultaneously.

**In tempura**, the `Attention` trait receives pre-projected Q, K, V tensors. The projections are matrix multiplications handled by `MatMul`, keeping the attention kernel focused on the score-normalize-aggregate computation.

## Grouped-Query Attention (GQA)

The KV cache stores key and value tensors for all previous tokens. With standard multi-head attention (MHA), each head has its own K and V — for a model with 32 heads, that's 32 copies of K and V in the cache.

GQA reduces this by sharing K and V across groups of heads:

| Variant | KV heads | Q heads | Ratio | KV cache size |
|---------|----------|---------|-------|--------------|
| MHA | 32 | 32 | 1:1 | Full |
| GQA-4 | 8 | 32 | 1:4 | 25% |
| GQA-8 | 4 | 32 | 1:8 | 12.5% |
| MQA | 1 | 32 | 1:32 | 3.1% |

The implementation maps each query head to its KV group: head $i$ uses KV group $\lfloor i \cdot n_{\text{kv}} / n_{\text{heads}} \rfloor$. This index arithmetic is one of the things Kani proves safe — see [What we prove](#what-we-prove) below.

**In tempura**, the `Attention::forward` signature takes both `num_heads` and `num_kv_heads` as parameters, making GQA a runtime configuration rather than a code variant.

## Numerical considerations

**Accumulation order.** The dot product $\sum_{i=1}^{d_k} q_i k_i$ is mathematically associative, but floating-point addition is not. Different summation orders produce different results at the limits of f32 precision (~$10^{-7}$ relative error per operation). tempura controls reduction order to ensure deterministic output — the same input always produces bit-identical results, even at the cost of foregoing optimizations that would reorder accumulations.

**Softmax precision.** For long sequences, the attention score matrix can be large ($n \times m$), and the softmax normalization must sum many exponentials. The max-subtraction trick prevents overflow, but the sum can still lose precision when many small values are added to large ones. Flash Attention addresses this by computing softmax in tiles with online normalization — tempura's CUDA attention kernel uses this approach.

**KV cache precision.** Values stored in the KV cache are typically f32 or f16. Quantized KV caches (see [Quantization](../quantization/)) reduce precision further but bound the error per the quantization scheme's guarantees.

## What we prove

Kani verifies two properties of the attention implementation:

**Q/KV offset safety.** The attention kernel indexes into flat arrays using arithmetic that combines batch index, head index, sequence position, and head dimension. The Kani harness proves that for any valid combination of `batch_size`, `seq_len`, `kv_len`, `num_heads`, `num_kv_heads`, and `head_dim` within bounded ranges, the computed offsets never exceed the buffer lengths.

This matters because out-of-bounds access in attention would silently read another sequence's data — a correctness and security issue, not just a crash.

**Batch isolation.** A separate harness proves that attention computation for sequence $i$ in a batch never reads memory belonging to sequence $j$. This is the multi-tenant safety guarantee: when multiple requests share a batch, their data cannot leak across sequences.

## What we do NOT prove

**Numerical accuracy.** Kani proves that array accesses are safe, not that the computed attention scores are numerically correct. The DualPipeline validates this by comparing CPU and GPU implementations on identical inputs.

**Softmax stability.** The max-subtraction trick is essential for numerical stability but involves floating-point operations that Kani cannot reason about precisely. Property-based tests verify softmax output sums to approximately 1.0 and all values are non-negative.

**Attention pattern quality.** Whether the model attends to the "right" things is a property of the trained weights and the data, not the kernel implementation. This is outside the scope of formal verification.

## Implementation

The CPU reference implementation is in [`kernels/src/attention.rs`](https://github.com/smgglrs-ai/tempura/tree/main/kernels/src/attention.rs). The `Attention` trait is defined in the kernel API crate. The implementation follows the score → scale → softmax → aggregate pipeline described above, with explicit support for GQA via the `num_kv_heads` parameter.
