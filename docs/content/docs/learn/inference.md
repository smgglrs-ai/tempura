+++
title = "9. Inference"
description = "What happens when you use a model — the generation loop, KV caching, and where tempura fits."
weight = 90
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Training is over. Now what?

A model has been trained — billions of weight values are frozen in files on disk. Someone types a prompt: "Explain quantum computing in simple terms."

What happens next is **inference**: running the trained model to generate a response, one token at a time. This is what tempura does.

## The generation loop

Language models generate text **autoregressively** — one token at a time, feeding each generated token back as input:

```
Prompt: "Explain quantum"
                    ↓
    Forward pass → logits → sample → "computing"
                    ↓
    Forward pass → logits → sample → " in"
                    ↓
    Forward pass → logits → sample → " simple"
                    ↓
    Forward pass → logits → sample → " terms"
                    ↓
    ... (continue until stop condition)
```

Each step:

1. Run the entire transformer forward pass on the current sequence
2. Take the logits at the last position
3. Apply [softmax and sampling](../probability/) to choose the next token
4. Append the new token to the sequence
5. Repeat

## The KV cache: why inference is different from training

Here's a problem: at step 100 of generation, we'd need to run the full transformer on all 100 tokens. At step 1000, on all 1000 tokens. Attention is $O(n^2)$ — this gets slow fast.

The insight: when generating token 101, tokens 1–100 haven't changed. The key and value vectors we computed for them in the previous step are still valid. We only need to compute Q, K, V for the *new* token.

The **KV cache** stores all previously computed key and value vectors:

| Step | What we compute | What we reuse from cache |
|------|----------------|------------------------|
| 1 (prompt, 5 tokens) | Q, K, V for all 5 tokens | Nothing |
| 2 (generate token 6) | Q, K, V for token 6 only | K, V for tokens 1–5 |
| 3 (generate token 7) | Q, K, V for token 7 only | K, V for tokens 1–6 |
| 100 | Q, K, V for token 104 | K, V for tokens 1–103 |

Each generation step is a constant-cost operation (plus the attention computation, which scales linearly with sequence length). Without the KV cache, cost would grow quadratically.

**In tempura**, the `KvCache` struct manages this storage. Kani proves the cache's slot management never goes out of bounds — see [Reading Kani Harnesses](../../foundations/reading-kani-harnesses/).

## Two phases of inference

### Prefill

The first forward pass processes the entire prompt at once. For a prompt of 100 tokens, this runs 100 tokens through all transformer layers, computing and caching all K and V vectors.

This is **compute-bound** — lots of matrix multiplications, and the GPU is busy.

### Decode

Each subsequent step generates one token. The forward pass runs on just one token, but the attention must look at all cached K, V vectors.

This is **memory-bound** — the GPU spends most of its time loading KV cache data from memory, not computing.

The prefill/decode distinction is why inference servers use different optimization strategies for each phase. tempura supports chunked prefill (processing the prompt in pieces to share GPU resources with decode) and disaggregated serving (running prefill and decode on separate hardware).

## Batching: serving multiple requests

A single request uses a fraction of the GPU's compute capacity during decode (the forward pass for one token is small). **Continuous batching** serves multiple requests simultaneously:

```
GPU step 1: Process token for Request A, Request B, Request C
GPU step 2: Process token for Request A, Request B, Request D (C finished, D started)
GPU step 3: Process token for Request A, Request D, Request E (B finished, E started)
```

New requests enter and completed requests leave *between steps*, not in fixed batches. This keeps the GPU busy and reduces latency.

tempura's `BatchScheduler` manages this — allocating KV cache slots, ordering requests by priority, and preempting low-priority requests when resources are tight.

## What makes inference hard

The math is straightforward — it's the same [matrix multiplications](../matrices/), [attention](../the-transformer/), and [softmax](../probability/) from the previous pages. What makes inference engineering challenging:

**Memory.** A 7B model's weights are 14 GB in float16. The KV cache for 100 concurrent requests at 2048 tokens each can be another 10+ GB. It all has to fit in GPU memory.

**Throughput.** Users expect multiple tokens per second. Every token requires a full forward pass — 32+ layers of attention and FFN. Optimized kernels, batching, and caching are essential.

**Correctness.** A bug in attention indexing doesn't produce an error message — it silently reads the wrong data and generates plausible but wrong text. This is why tempura uses [Kani proofs](../../foundations/bounded-model-checking/) for index safety and [DualPipeline](../../foundations/dual-pipeline/) for numerical equivalence.

**Determinism.** Given the same prompt and seed, the same output should be produced every time. This requires controlling floating-point reduction order, RNG seeding, and cache state — things that most inference servers treat as optional.

## Where tempura fits

tempura is an **inference engine and framework**. It:

| Does | Doesn't |
|------|---------|
| Loads trained model weights from files | Train models |
| Runs the forward pass (all the math from this course) | Compute gradients or update weights |
| Manages KV cache for efficient generation | Store training data |
| Serves an OpenAI-compatible API | Fine-tune or align models |
| Proves kernel safety with Kani | Prove the model produces good outputs |
| Compares implementations with DualPipeline | Judge output quality |
| Records structured InferenceTraces | Make the model smarter |

The unique contribution: verification and observability. Most inference servers are fast black boxes. tempura is a fast box you can look inside — every forward pass produces a [trace](../../trace-tooling/) you can query, diff, and replay.

## What to read next

You've completed the learning track. You now understand every operation a language model performs — from [vectors](../vectors/) to inference. Here's where to go deeper:

- **[Foundations](../../foundations/)** — the same concepts at implementation depth, with equations, Kani proof walkthroughs, and links to the source code
- **[Architecture](../../architecture/)** — how tempura's crates are organized
- **[Getting Started](../../getting-started/)** — build and run tempura
- **[API Reference](../../api/)** — the endpoints you can call

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| Autoregressive generation | Generate one token at a time, feed back as input | How language models produce text |
| KV cache | Store previously computed keys and values | Makes generation efficient |
| Prefill vs decode | First pass (all tokens) vs subsequent passes (one token) | Different bottlenecks, different optimizations |
| Continuous batching | Serve multiple requests simultaneously | Maximizes GPU utilization |
| Inference vs training | Using a trained model vs teaching a model | tempura does inference |
