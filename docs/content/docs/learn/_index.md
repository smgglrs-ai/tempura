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

This section takes you from "I've heard of AI but I don't know how it works" to "I understand what happens inside a language model when it generates text, and how GPUs execute the math." Each page builds on the previous one, with worked examples using small numbers you can verify with a calculator or by hand.

## Part I: The Math

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

## Part II: The Hardware

| # | Topic | What you'll understand | What you need first |
|---|-------|----------------------|-------------------|
| 10 | [GPU Computing](gpu-computing/) | Why GPUs, not CPUs, run AI — and how they think differently | Part I complete |
| 11 | [Memory Hierarchy](memory-hierarchy/) | Why memory speed matters more than compute speed | GPU Computing |
| 12 | [Writing GPU Kernels](writing-gpu-kernels/) | Threads, blocks, shared memory — the CUDA programming model | Memory Hierarchy |
| 13 | [cuda-oxide](cuda-oxide/) | GPU kernels in Rust — what cuda-oxide changes | Writing GPU Kernels |
| 14 | [From Math to Metal](from-math-to-metal/) | How tempura maps attention and matmul to GPU hardware | cuda-oxide |

## How to read this

**Part I** stands alone — you can stop after Chapter 9 and understand how language models work. **Part II** goes deeper into how the math runs on hardware, which is essential if you want to understand tempura's GPU kernels and formal verification of GPU code.

**Go in order.** Each page builds on the previous one. If something doesn't make sense, go back one page.

**Do the examples.** Understanding comes from computing, not reading.

**Then read [Foundations](../foundations/).** The Foundations section covers the same concepts at implementation depth — with the rigor needed to understand tempura's code and proofs.
