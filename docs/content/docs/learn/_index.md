+++
title = "Learn"
description = "From zero to understanding AI — math, transformers, GPU hardware, and formal verification."
sort_by = "weight"
weight = 5
template = "docs/section.html"

[extra]
toc = true
+++

You don't need a computer science degree. You don't need to know what a neural network is. You don't even need to remember much math from school — we'll rebuild what we need as we go.

This section takes you from "I've heard of AI but I don't know how it works" all the way to understanding how GPU kernels are formally verified. Each page builds on the previous one, with worked examples you can verify by hand.

## Part I: The Math

| # | Topic | What you'll understand |
|---|-------|----------------------|
| 0 | [What is a Model?](what-is-a-model/) | What AI models actually are (and aren't) |
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

The same operations from Part I, now at implementation depth — with equations, Kani proof walkthroughs, and links to the source code.

### Kernel math

| # | Topic | What you'll understand |
|---|-------|----------------------|
| 15 | [Attention](attention/) | Scaled dot-product attention, softmax stability, GQA, Kani proofs |
| 16 | [Normalization](normalization/) | RMS norm, the variance invariant, pre-norm vs post-norm |
| 17 | [Positional Encoding](positional-encoding/) | RoPE rotation matrix, frequency bands, length extrapolation |
| 18 | [Sampling](sampling/) | Temperature as Boltzmann, top-k/top-p, deterministic RNG |
| 19 | [Quantization](quantization/) | Bounded error, group quantization, monotonicity, GPTQ/AWQ/GGUF |

### Verification theory

| # | Topic | What you'll understand |
|---|-------|----------------------|
| 20 | [Bounded Model Checking](bounded-model-checking/) | What Kani proves, SAT/SMT, what "all inputs" means within bounds |
| 21 | [Reading Kani Harnesses](reading-kani-harnesses/) | Annotated walkthrough of a real proof from tempura |
| 22 | [Compositionality](compositionality/) | Why proving individual operations proves the graph |
| 23 | [DualPipeline](dual-pipeline/) | Empirical equivalence testing — what formal proofs miss |
| 24 | [Verification Tiers](verification-tiers/) | Proven, Tested, Compiled — what each tier guarantees |

## How to read this

**Part I** stands alone — you can stop after Chapter 9 and understand how language models work. **Part II** explains how the math runs on GPU hardware. **Part III** goes to implementation depth on each kernel and the verification system.

**Go in order** within each part. Parts II and III can be read in either order after Part I.

**Do the examples.** Understanding comes from computing, not reading.
