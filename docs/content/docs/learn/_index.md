+++
title = "Learn"
description = "From zero to understanding AI — math, transformers, GPU hardware, and formal verification."
sort_by = "weight"
weight = 5
template = "docs/section.html"

[extra]
toc = true
no_card_list = true

[[extra.sidebar_parts]]
title = "Part I: The Math"
id = "math"

[[extra.sidebar_parts]]
title = "Part II: The Hardware"
id = "hardware"

[[extra.sidebar_parts]]
title = "Part III: Kernel Math"
id = "kernel-math"

[[extra.sidebar_parts]]
title = "Part IV: Verification"
id = "verification"
+++

You don't need a computer science degree. You don't need to know what a neural network is. You do need basic arithmetic and simple algebra — multiplication, fractions, and expressions like $f(x) = 2x + 3$. Everything beyond that, we build from scratch.

This section takes you from "I've heard of AI but I don't know how it works" all the way to understanding how GPU kernels are formally verified.

## Part I: The Math

| # | Topic | What you'll understand |
|---|-------|----------------------|
| 0 | [What is a Model?](what-is-a-model/) | What AI models actually are (and aren't) |
| 0.5 | [Reading Math Notation](reading-math/) | A guide to every symbol we use — $\sum$, $\in$, $\mathbb{R}$, Greek letters |
| 1 | [Vectors](vectors/) | How to represent things as lists of numbers |
| 2 | [Matrices](matrices/) | How to transform vectors systematically |
| 3 | [Functions](functions/) | Why neural networks need non-linearity |
| 4 | [Probability](probability/) | How models express uncertainty and make choices |
| 5 | [Learning](learning/) | How a model learns from examples |
| 6 | [Neural Networks](neural-networks/) | How layers stack into a thinking machine |
| 7 | [Words as Numbers](words-as-numbers/) | How text becomes math |
| 8 | [The Transformer](the-transformer/) | The architecture behind ChatGPT, Claude, and LLaMA |
| 9 | [Inference](inference/) | What happens when you use a model — and what tempura does |

## Part II: The Hardware

| # | Topic | What you'll understand |
|---|-------|----------------------|
| 10 | [GPU Computing](gpu-computing/) | Why GPUs, not CPUs, run AI — and how they think differently |
| 11 | [Memory Hierarchy](memory-hierarchy/) | Why memory speed matters more than compute speed |
| 12 | [Writing GPU Kernels](writing-gpu-kernels/) | Threads, blocks, shared memory — the CUDA programming model |
| 13 | [cuda-oxide](cuda-oxide/) | GPU kernels in Rust — what cuda-oxide changes |
| 14 | [From Math to Metal](from-math-to-metal/) | How tempura maps attention and matmul to GPU hardware |

## Part III: Foundations

### Kernel math

| # | Topic | What you'll understand |
|---|-------|----------------------|
| 15 | [Attention](attention/) | Scaled dot-product attention, softmax stability, GQA, Kani proofs |
| 16 | [Normalization](normalization/) | RMS norm, the variance invariant, pre-norm vs post-norm |
| 17 | [Positional Encoding](positional-encoding/) | RoPE rotation matrix, frequency bands, length extrapolation |
| 18 | [Sampling](sampling/) | Temperature as Boltzmann, top-k/top-p, deterministic RNG |
| 19 | [Quantization](quantization/) | Bounded error, group quantization, monotonicity, GPTQ/AWQ/GGUF |

## Part IV: Verification Theory

| # | Topic | What you'll understand |
|---|-------|----------------------|
| 20 | [Bounded Model Checking](bounded-model-checking/) | What Kani proves, SAT/SMT, what "all inputs" means within bounds |
| 21 | [Reading Kani Harnesses](reading-kani-harnesses/) | Annotated walkthrough of a real proof from tempura |
| 22 | [Compositionality](compositionality/) | Why proving individual operations proves the graph |
| 23 | [DualPipeline](dual-pipeline/) | Empirical equivalence testing — what formal proofs miss |
| 24 | [Verification Tiers](verification-tiers/) | Proven, Tested, Compiled — what each tier guarantees |

## How to read this

**Part I** stands alone — stop after Chapter 9 and you'll understand how language models work, end to end. No hardware or verification knowledge needed.

**Part II** explains how the math runs on GPUs. Read this if you want to understand why inference is fast, what cuda-oxide changes, and how tempura's kernels map to hardware.

**Part III** covers each kernel's math at implementation depth — equations, numerical considerations, and links to the source code.

**Part IV** covers the verification system — what Kani proves, how to read proof harnesses, compositionality, DualPipeline, and verification tiers. Read this if you want to understand or contribute to tempura's proofs.

**Go in order** within each part. Parts II, III, and IV can be read in any order after Part I.

**Do the examples.** The numbers are small on purpose — understanding comes from computing, not reading.
