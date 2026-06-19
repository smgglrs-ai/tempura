+++
title = "Learn"
description = "From high-school math to understanding how language models work — a progressive course."
sort_by = "weight"
weight = 5
template = "docs/section.html"

[extra]
toc = true
+++

This section builds your understanding from the ground up. No machine learning background required — just high-school algebra and curiosity.

Each page introduces one concept, uses it to build the next, and includes worked examples with small, concrete numbers you can verify by hand. By the end, you'll understand what a language model is, how it generates text, and what tempura does to make that process verifiable and observable.

## The path

| # | Topic | You'll learn | You'll need |
|---|-------|-------------|-------------|
| 1 | [Vectors](vectors/) | How to represent things as lists of numbers | Basic arithmetic |
| 2 | [Matrices](matrices/) | How to transform vectors, matrix multiplication | Vectors |
| 3 | [Functions](functions/) | Linear vs non-linear transformations, activation functions | Matrices |
| 4 | [Probability](probability/) | Distributions, softmax, sampling from distributions | Functions |
| 5 | [Learning](learning/) | How a model learns from data — loss, gradients, optimization | Probability |
| 6 | [Neural Networks](neural-networks/) | Composing layers into a network, the forward pass | Learning |
| 7 | [Words as Numbers](words-as-numbers/) | Tokenization, embeddings, meaning in vector space | Neural networks |
| 8 | [The Transformer](the-transformer/) | Attention, the architecture that powers modern AI | Everything above |
| 9 | [Inference](inference/) | What happens when you use a model — and what tempura does | The transformer |

## How to read this

**Go in order.** Each page builds on the previous one. Skipping ahead will leave gaps.

**Do the examples.** When we show $[3, 4] \cdot [1, 2] = 3 \times 1 + 4 \times 2 = 11$, verify it. The numbers are small on purpose — understanding comes from computing, not just reading.

**Then read [Foundations](../foundations/).** Once you've completed this track, the Foundations section goes deeper into the specific operations tempura implements — the same concepts, but at the level of detail needed to understand the code and the proofs.
