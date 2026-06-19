+++
title = "Sampling"
description = "Temperature, top-k, top-p — the probability space of token generation."
weight = 40
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## From logits to tokens

The transformer's forward pass produces a vector of logits $z \in \mathbb{R}^V$ where $V$ is the vocabulary size. Each logit $z_i$ represents the model's unnormalized preference for token $i$. Sampling converts these logits into a probability distribution and selects the next token.

## Temperature

Temperature scales the logits before softmax, controlling the "sharpness" of the distribution:

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

- $T = 1$: the model's native distribution
- $T \to 0$: approaches argmax (greedy, deterministic)
- $T > 1$: flattens the distribution (more random, more diverse)

The connection to statistical mechanics is direct: this is the Boltzmann distribution with $T$ as temperature and $z_i$ as energy.

## Top-k sampling

<!-- TODO: Truncate to the k highest-probability tokens, renormalize.
     Why this prevents sampling from the long tail of implausible tokens.
     The tradeoff: too small k → repetitive, too large k → incoherent. -->

## Top-p (nucleus) sampling

<!-- TODO: Truncate to the smallest set of tokens whose cumulative probability
     exceeds p. Adaptive: more tokens when the model is uncertain, fewer
     when it's confident. Holtzman et al. (2020). The cumulative sum
     computation and its numerical properties. -->

## Deterministic sampling in tempura

<!-- TODO: Seeded RNG, why reproducibility matters for verification,
     the relationship between sampling determinism and inference traces,
     how DualPipeline uses identical seeds to compare implementations. -->

## What we prove

<!-- TODO (p18-foundations-connection): Kani harness for top-k index bounds,
     cumulative sum fallback safety. -->

## What we do NOT prove

<!-- TODO (p18-foundations-connection): Distribution correctness (property tests),
     sampling quality (empirical). -->

## Implementation

The CPU reference is in [`kernels/src/sampling.rs`](https://github.com/smgglrs-ai/tempura/tree/main/kernels/src/sampling.rs). The `Sampler` trait is defined in the kernel API crate.
