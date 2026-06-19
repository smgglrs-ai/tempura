+++
title = "18. Sampling (Deep Dive)"
description = "Temperature, top-k, top-p — the probability space of token generation."
weight = 180
template = "docs/page.html"

[extra]
part = "kernel-math"
toc = true
math = true
+++

## From logits to tokens

The transformer's forward pass produces a vector of logits $z \in \mathbb{R}^V$ where $V$ is the vocabulary size (typically 32,000–128,000). Each logit $z_i$ represents the model's unnormalized preference for token $i$. Sampling converts these logits into a probability distribution and selects the next token.

This is the only stochastic step in inference. Everything before it — embeddings, attention, normalization, FFN — is deterministic arithmetic. The sampling strategy controls the tradeoff between predictability and diversity in the generated text.

## Temperature

Temperature scales the logits before softmax, controlling the entropy of the distribution:

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

The connection to statistical mechanics is direct. In the Boltzmann distribution, $T$ is the physical temperature, $z_i$ is the energy of state $i$, and $p_i$ is the probability of the system being in state $i$. Higher temperature = more disorder = more uniform distribution.

| Temperature | Effect | Use case |
|-------------|--------|----------|
| $T \to 0$ | Approaches argmax — always picks the most likely token | Code generation, factual answers |
| $T = 1$ | The model's native distribution as trained | General-purpose generation |
| $T > 1$ | Flatter distribution — more diverse, more creative | Creative writing, brainstorming |

**Greedy decoding** ($T = 0$) is the deterministic special case: always select $\arg\max_i z_i$. tempura implements this as a separate code path (no softmax needed — just find the maximum).

## Top-k sampling

After applying temperature, top-k truncates the distribution to the $k$ most probable tokens and renormalizes:

$$
p_i' = \begin{cases} \frac{p_i}{\sum_{j \in \text{top-}k} p_j} & \text{if } i \in \text{top-}k \\\\ 0 & \text{otherwise} \end{cases}
$$

**Why truncate?** Language models assign small but non-zero probability to every token in the vocabulary. Without truncation, sampling occasionally draws from the extreme tail — producing incoherent text. Top-k removes these implausible tokens entirely.

**The tradeoff:** too small $k$ makes output repetitive (only a few tokens are ever sampled); too large $k$ approaches untruncated sampling. Typical values are $k \in [10, 100]$.

**Implementation:** tempura sorts the logits, takes the top $k$, and samples from the truncated distribution. The sort is $O(V \log V)$ — for a typical vocabulary this is fast enough, but specialized partial-sort algorithms could reduce this to $O(V + k \log k)$.

## Top-p (nucleus) sampling

Top-p (Holtzman et al., 2020) adapts the truncation to the model's confidence. Instead of a fixed number of tokens, it includes the smallest set whose cumulative probability exceeds $p$:

$$
\text{top-}p = \min \\{ S \subseteq V : \sum_{i \in S} p_i \geq p \\}
$$

where tokens are included in order of decreasing probability.

**Why adaptive?** When the model is confident (one token has 90% probability), top-p with $p = 0.9$ includes just that one token. When the model is uncertain (many tokens share probability), the same $p$ includes many tokens. Top-k always includes exactly $k$ regardless of confidence.

**The computation:** sort by probability, compute the cumulative sum, find the cutoff. The cumulative sum is:

$$
C_i = \sum_{j=1}^{i} p_{\sigma(j)}
$$

where $\sigma$ orders tokens by decreasing probability. The nucleus is $\\{j : C_j \leq p\\}$ plus the first token that pushes the sum past $p$.

## Combined sampling

In practice, all three techniques are applied together: temperature first (modifies the distribution shape), then top-k (hard cutoff), then top-p (adaptive cutoff within the top-k), then sample from the surviving distribution.

tempura's `SamplingParams` struct configures all three:

```rust
SamplingParams {
    temperature: 0.7,
    top_k: 50,
    top_p: 0.9,
    seed: Some(42),
    // ...
}
```

## Deterministic sampling in tempura

Sampling requires a random number generator. tempura uses a seeded `StdRng` (Rust's standard PRNG) — given the same seed and the same logits, the same token is always selected.

This is critical for verification:

- **DualPipeline** uses identical seeds to compare implementations. If sampling were non-deterministic, you couldn't tell whether a token mismatch was caused by a kernel bug or random chance.
- **InferenceTrace** records the seed per token, enabling exact replay.
- **Reproducibility** is a user-facing guarantee: `seed: 42` in the API request produces the same output every time.

## What we prove

Kani verifies two properties of the sampling implementation:

**Top-k index safety.** After sorting, the top-k selection indexes into the sorted array. The harness proves that for any valid `k` and vocabulary size, the index never exceeds the array length.

**Cumulative sum fallback.** The top-p implementation has a fallback path: if floating-point imprecision causes the cumulative sum to never quite reach $p$, the last token is selected. The harness proves that this fallback always returns a valid token index — the sampler never fails to produce output.

## What we do NOT prove

**Distribution correctness.** That the softmax output actually forms a valid probability distribution (sums to 1.0, all values non-negative) is verified by property-based tests, not Kani. Floating-point softmax can produce values that sum to 0.99999... or 1.00001..., which is acceptable but not formally provable.

**Sampling quality.** Whether the generated text is "good" for a given temperature/top-k/top-p combination is an empirical question about the model and the task.

## Implementation

The CPU reference is in [`kernels/src/sampling.rs`](https://github.com/smgglrs-ai/tempura/tree/main/kernels/src/sampling.rs). The `Sampler` trait is defined in the kernel API crate. The implementation supports greedy (argmax), top-k, top-p, and temperature-scaled sampling with a seeded RNG.
