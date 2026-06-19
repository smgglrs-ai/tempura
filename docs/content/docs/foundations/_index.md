+++
title = "Foundations"
description = "The math and verification theory behind tempura — from linear algebra to formal proofs."
sort_by = "weight"
weight = 15
template = "docs/section.html"

[extra]
toc = true
+++

tempura is an educational project as much as an engineering one. Every kernel implements a mathematical operation. Every Kani harness proves a property of that operation. This section teaches both — the math and the proofs — so that contributors and reviewers can trace any code change through its mathematical meaning to its verification guarantee.

## How to read this section

**If you're new to transformers**, start with [Attention](attention/) — it's the core operation, and every other concept connects back to it.

**If you're new to formal verification**, start with [Bounded Model Checking](bounded-model-checking/) — it explains what Kani actually does under the hood, without assuming prior verification experience.

**If you're reviewing a PR**, find the kernel being modified, read its foundation page, and follow the "What we prove" section to understand which properties the Kani harnesses guarantee. The "What we do NOT prove" section tells you what to check manually or with DualPipeline.

## The three layers

| Layer | Question it answers | Example |
|-------|-------------------|---------|
| **Math** | What does this operation compute, and why? | Attention computes a weighted average of values, where weights come from query-key similarity via softmax |
| **Verification** | What properties does Kani prove about the implementation? | Q and KV offset arithmetic never produces an out-of-bounds index for any valid input within the bounded range |
| **Connection** | How does the proof relate to the math? | The offset proof ensures that the attention computation reads exactly the memory it should — no adjacent sequence's data leaks in |

## Pages

### Math behind the kernels
- [Attention](attention/) — scaled dot-product attention, multi-head, GQA
- [Normalization](normalization/) — RMS norm, the variance invariant
- [Positional Encoding](positional-encoding/) — rotary position embeddings (RoPE)
- [Sampling](sampling/) — temperature, top-k, top-p
- [Quantization](quantization/) — rounding, group quantization, error bounds

### Verification theory
- [Bounded Model Checking](bounded-model-checking/) — what Kani proves and how
- [Reading Kani Harnesses](reading-kani-harnesses/) — annotated walkthrough of a real proof
- [Compositionality](compositionality/) — proving graphs from proven operations
- [DualPipeline](dual-pipeline/) — empirical equivalence testing
- [Verification Tiers](verification-tiers/) — Proven, Tested, Compiled
