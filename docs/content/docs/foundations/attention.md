+++
title = "Attention"
description = "Scaled dot-product attention — the linear algebra, softmax, multi-head, and GQA."
weight = 10
template = "docs/page.html"

[extra]
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

## Why softmax?

<!-- TODO: Softmax as a smooth argmax, the Boltzmann distribution connection,
     temperature relationship (links to Sampling page), numerical stability
     via the max-subtraction trick. -->

## Multi-head attention

<!-- TODO: The projection idea — multiple independent attention "heads" attending
     to different representation subspaces. The split/project/attend/concat/project
     pipeline. Why this works better than a single large attention. -->

## Grouped-Query Attention (GQA)

<!-- TODO: The KV sharing observation — not every head needs its own K and V.
     GQA as the continuum between MHA (every head has its own KV) and MQA
     (all heads share one KV). The memory/compute tradeoff. -->

## Numerical considerations

<!-- TODO: f32 precision in dot products, accumulation order,
     why tempura controls reduction order for determinism. -->

## What we prove

<!-- TODO (p18-foundations-connection): Map to the actual Kani harnesses.
     Link to the Q/KV offset proof, the batch isolation proof.
     Explain what each proof guarantees about the math above. -->

## What we do NOT prove

<!-- TODO (p18-foundations-connection): Numerical accuracy (DualPipeline),
     softmax stability (property tests), attention pattern quality (empirical). -->

## Implementation

The CPU reference implementation is in [`kernels/src/attention.rs`](https://github.com/smgglrs-ai/tempura/tree/main/kernels/src/attention.rs). The `Attention` trait is defined in the kernel API crate.
