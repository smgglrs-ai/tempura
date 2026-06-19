+++
title = "Learn"
description = "From zero to understanding how AI language models work — no prerequisites."
sort_by = "weight"
weight = 5
template = "docs/section.html"

[extra]
toc = true
+++

You don't need a computer science degree. You don't need to know what a neural network is. You don't even need to remember much math from school — we'll rebuild what we need as we go.

This section takes you from "I've heard of AI but I don't know how it works" to "I understand what happens inside a language model when it generates text." Each page builds on the previous one, with worked examples using small numbers you can verify with a calculator or by hand.

## The path

| # | Topic | What you'll understand | What you need first |
|---|-------|----------------------|-------------------|
| 0 | [What is a Model?](what-is-a-model/) | What AI models actually are (and aren't) | Nothing — start here |
| 1 | [Vectors](vectors/) | How to represent things as lists of numbers | Chapter 0 |
| 2 | [Matrices](matrices/) | How to transform vectors systematically | Vectors |
| 3 | [Functions](functions/) | Why neural networks need non-linearity | Matrices |
| 4 | [Probability](probability/) | How models express uncertainty and make choices | Functions |
| 5 | [Learning](learning/) | How a model learns from examples | Probability |
| 6 | [Neural Networks](neural-networks/) | How layers stack into a thinking machine | Learning |
| 7 | [Words as Numbers](words-as-numbers/) | How text becomes math | Neural networks |
| 8 | [The Transformer](the-transformer/) | The architecture behind ChatGPT, Claude, and LLaMA | Everything above |
| 9 | [Inference](inference/) | What happens when you use a model — and what tempura does | The transformer |

## How to read this

**Go in order.** Each page builds on the previous one. If something doesn't make sense, go back one page — you may have missed a building block.

**Do the examples.** When we compute $3 \times 1 + 4 \times 2 = 11$, do it yourself. When we show a table of numbers, trace through the calculation. Understanding comes from doing, not reading.

**Don't worry about remembering everything.** You can always come back. The goal is to build intuition, not memorize formulas.

**Then read [Foundations](../foundations/).** Once you've completed this track, the Foundations section covers the same concepts at implementation depth — with the rigor needed to understand tempura's code and proofs.
