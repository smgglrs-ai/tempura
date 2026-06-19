+++
title = "4. Probability"
description = "Distributions, softmax, and sampling — how models make decisions."
weight = 40
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## From scores to choices

A language model needs to choose the next word. After processing the input, it produces a list of **scores** — one for each word in its vocabulary. A vocabulary of 32,000 words means 32,000 scores.

But scores aren't decisions. A score of 8.3 for "cat" and 6.1 for "dog" tells us "cat" is preferred, but by how much? And how do we choose when we don't always want the top answer?

We need to convert scores into **probabilities** — numbers between 0 and 1 that sum to 1. That's what this page is about.

## What is a probability distribution?

A **probability distribution** assigns a probability to each possible outcome such that:

1. Every probability is between 0 and 1: $0 \leq p_i \leq 1$
2. All probabilities sum to 1: $\sum_i p_i = 1$

A fair die: $[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]$. A loaded die: $[0.3, 0.2, 0.1, 0.1, 0.1, 0.2]$.

The probabilities tell you how likely each outcome is. To **sample** from a distribution, you pick an outcome at random, weighted by the probabilities. With the loaded die, you'll roll 1 about 30% of the time.

## Softmax: from scores to probabilities

**Softmax** is the function that converts a vector of arbitrary scores into a probability distribution:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Let's compute it step by step for scores $z = [2, 1, 0]$:

**Step 1:** Exponentiate each score:

$$
e^2 \approx 7.39, \quad e^1 \approx 2.72, \quad e^0 = 1.00
$$

**Step 2:** Sum the exponentials:

$$
7.39 + 2.72 + 1.00 = 11.11
$$

**Step 3:** Divide each by the sum:

$$
\text{softmax}([2, 1, 0]) = \left[\frac{7.39}{11.11}, \frac{2.72}{11.11}, \frac{1.00}{11.11}\right] = [0.665, 0.245, 0.090]
$$

**Check:** $0.665 + 0.245 + 0.090 = 1.000$ ✓

The highest score (2) got 66.5% of the probability. The lowest score (0) got 9%. Softmax preserves the ordering but creates a proper probability distribution.

## Temperature

What if you want the probabilities to be more or less spread out? **Temperature** $T$ scales the scores before softmax:

$$
\text{softmax}(z / T)_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

Using our example $z = [2, 1, 0]$:

| Temperature $T$ | Probabilities | Effect |
|-----------------|--------------|--------|
| $0.5$ (cold) | $[0.844, 0.114, 0.042]$ | Sharper — the winner dominates |
| $1.0$ (normal) | $[0.665, 0.245, 0.090]$ | The model's natural distribution |
| $2.0$ (hot) | $[0.506, 0.307, 0.186]$ | Flatter — more diversity |

Low temperature → more confident, less varied output. High temperature → more random, more creative output. At $T \to 0$, softmax approaches argmax (always pick the highest). At $T \to \infty$, it approaches uniform (every word equally likely).

This is exactly the temperature parameter in ChatGPT, Claude, and tempura's API.

## Sampling from a distribution

Given probabilities $[0.665, 0.245, 0.090]$, how do we pick a word?

**Greedy (argmax):** always pick the highest probability. Output is deterministic — always "cat" in this case. Simple but repetitive.

**Random sampling:** imagine a number line from 0 to 1, divided into segments proportional to the probabilities:

```
|-------- 0.665 --------|---- 0.245 ---|-- 0.090 --|
0                      0.665         0.910        1.0
         word A               word B        word C
```

Generate a random number $r \in [0, 1)$. If $r < 0.665$, pick word A. If $0.665 \leq r < 0.910$, pick word B. Otherwise, pick word C.

**Top-k sampling:** only consider the $k$ most probable words. If $k = 2$, ignore word C entirely and renormalize:

$$
[0.665, 0.245] \to \left[\frac{0.665}{0.910}, \frac{0.245}{0.910}\right] = [0.731, 0.269]
$$

This prevents the model from occasionally picking very unlikely words.

**Top-p (nucleus) sampling:** instead of a fixed $k$, include words until their cumulative probability exceeds $p$. With $p = 0.9$ and our probabilities $[0.665, 0.245, 0.090]$:

- Word A: cumulative $= 0.665$ (below 0.9 — include)
- Word B: cumulative $= 0.910$ (above 0.9 — include this one, stop)

Result: consider words A and B, ignore C. The advantage over top-k: when the model is very confident (one word has 95% probability), only that word is considered. When it's uncertain, more words are included.

tempura implements all of these in its [Sampler kernel](../../foundations/sampling/).

## Conditional probability

One more concept you'll need: the probability of event A **given** event B:

$$
P(A | B) = \frac{P(A \text{ and } B)}{P(B)}
$$

In language modeling, this is the core question: "given the words so far, what's the probability of the next word?"

$$
P(\text{next word} | \text{previous words})
$$

A language model is essentially a machine that computes this conditional probability for all 32,000 words in its vocabulary, at every position in the sequence. The output of the model (the scores/logits) are the raw values that softmax converts into these conditional probabilities.

## Why determinism matters

In tempura, sampling uses a **seeded random number generator**. Given the same seed and the same probabilities, the same word is always chosen.

Why does this matter?

- **Reproducibility:** the same prompt with the same seed produces the same response, every time.
- **Verification:** [DualPipeline](../../foundations/dual-pipeline/) compares two implementations. If sampling were truly random, you couldn't tell whether a different output was caused by a bug or by different random numbers.
- **Debugging:** when something goes wrong, you can replay the exact inference with the same seed and investigate.

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| Probability distribution | Numbers in $[0,1]$ that sum to 1 | How models express uncertainty |
| Softmax | Converts scores → probabilities | The bridge from logits to decisions |
| Temperature | Scales scores before softmax | Controls randomness vs determinism |
| Sampling | Picking from a distribution | How the model generates each token |

**Next:** [Learning](../learning/) — how a model acquires its knowledge.
