+++
title = "Learn"
description = "From zero to operating a production inference server — math, hardware, serving, and verification."
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

[[extra.sidebar_parts]]
title = "Part V: Serving"
id = "serving"

[[extra.sidebar_parts]]
title = "Part VI: Production"
id = "production"
+++

You don't need a computer science degree. You don't need to know what a neural network is. You do need basic arithmetic and simple algebra — multiplication, fractions, and expressions like $f(x) = 2x + 3$. Everything beyond that, we build from scratch.

This section takes you from "I've heard of AI but I don't know how it works" all the way to understanding how to build, verify, and operate a production inference server.

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

## Part III: Kernel Math

| # | Topic | What you'll understand |
|---|-------|----------------------|
| 15 | [Attention](attention/) | Scaled dot-product attention, softmax stability, GQA, Kani proofs |
| 16 | [Normalization](normalization/) | RMS norm, the variance invariant, pre-norm vs post-norm |
| 17 | [Positional Encoding](positional-encoding/) | RoPE rotation matrix, frequency bands, length extrapolation |
| 18 | [Sampling](sampling/) | Temperature as Boltzmann, top-k/top-p, deterministic RNG |
| 19 | [Quantization](quantization/) | Bounded error, group quantization, monotonicity, GPTQ/AWQ/GGUF |
| 20 | [Quantized MatMul on GPUs](quantized-matmul-gpu/) | Q4_K block layout, decode vs prefill parallelism, warp-cooperative kernels |
| 21 | [Ternary Weights](ternary-weights/) | When weights are just -1, 0, or +1 — BitNet, BitLinear, and why verification loves ternary |
| 22 | [Ternary Bit-Packing](ternary-packing/) | TQ2_0, TQ1_0, STQ1_0 — three ways to pack {-1,0,1} and why the choice shapes GPU kernels |
| 23 | [Ternary GPU Kernels](ternary-gpu-kernels/) | dp4a integer paths, LUT approaches, and why ternary is even more memory-bound than Q4_K |

## Part IV: Verification Theory

| # | Topic | What you'll understand |
|---|-------|----------------------|
| 24 | [Bounded Model Checking](bounded-model-checking/) | What Kani proves, SAT/SMT, what "all inputs" means within bounds |
| 25 | [Reading Kani Harnesses](reading-kani-harnesses/) | Annotated walkthrough of a real proof from tempura |
| 26 | [Compositionality](compositionality/) | Why proving individual operations proves the graph |
| 27 | [DualPipeline](dual-pipeline/) | Empirical equivalence testing — what formal proofs miss |
| 28 | [Verification Tiers](verification-tiers/) | Proven, Tested, Compiled — what each tier guarantees |

## Part V: Serving

| # | Topic | What you'll understand |
|---|-------|----------------------|
| 25 | HTTP and APIs | What REST is, request/response, why OpenAI's API became the standard |
| 26 | Chat Templates | How chat messages become token sequences — Jinja2, ChatML, role markers |
| 27 | Streaming | SSE, token-by-token delivery, why users see text appear word by word |
| 28 | Continuous Batching | Why serving one request at a time is wasteful, how requests enter/exit mid-step |
| 29 | PagedAttention | Virtual memory for KV cache — pages, block tables, copy-on-write prefix sharing |
| 30 | Tool Calling and Structured Output | Function calling, JSON mode, guided decoding, grammar-based constraints |
| 31 | Speculative Decoding | Draft-verify loop, why a small fast model + a large slow model beats one medium model |

## Part VI: Production

| # | Topic | What you'll understand |
|---|-------|----------------------|
| 32 | Observability | Prometheus metrics (TTFT, ITL, queue depth), OpenTelemetry, what to alert on |
| 33 | Kernel Fusion and CUDA Graphs | Fusing operations, capturing the forward pass, eliminating launch overhead |
| 34 | Performance Tuning | Prefill vs decode bottlenecks, FP8 KV cache, FlashDecoding, batch size tuning |
| 35 | Deployment | Containers, health/readiness probes, graceful shutdown, resource sizing |
| 36 | Multi-GPU and Scaling | Tensor parallelism, disaggregated prefill/decode, when to scale out vs up |
| 37 | Security and Isolation | Batch isolation (and how Kani proves it), cache_salt, request timeouts |

## How to read this

**Part I** stands alone — stop after Chapter 9 and you'll understand how language models work, end to end.

**Part II** explains how the math runs on GPUs — threads, memory, kernels, cuda-oxide.

**Part III** covers each kernel's math at implementation depth — equations, numerical stability, and links to the source code.

**Part IV** covers the verification system — what Kani proves, how to read proofs, compositionality, DualPipeline.

**Part V** explains what makes an inference *server* — APIs, batching, memory management, tool calling. This is where PagedAttention, streaming, and speculative decoding live.

**Part VI** covers operating a production server — observability, performance tuning, deployment, multi-GPU, and security.

**Go in order** within each part. After Part I, the other parts can be read in any order based on interest.

**Do the examples.** The numbers are small on purpose — understanding comes from computing, not reading.
