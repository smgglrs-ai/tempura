+++
title = "5. Learning"
description = "How a model learns from data — loss functions, gradients, and the idea of optimization."
weight = 50
template = "docs/page.html"

[extra]
part = "math"
toc = true
math = true
+++

## The question

We have a model that takes input words and produces probabilities for the next word (using matrices, activation functions, and softmax from the previous pages). But where do the numbers in those matrices come from?

They are **learned** from data. This page explains how.

## Prediction and error

Imagine a very simple model trying to predict the next word after "the cat sat on the":

| Word | Model's probability | Correct answer |
|------|-------------------|----------------|
| mat | 0.30 | ← This one (it's in the training data) |
| dog | 0.25 | |
| floor | 0.20 | |
| chair | 0.15 | |
| ... | 0.10 | |

The model gave "mat" 30% probability. We want it closer to 100%. The gap between the model's prediction and the right answer is the **error** (or **loss**).

## Loss functions

A **loss function** measures how wrong the model is. Lower loss = better predictions.

For language models, the standard loss is **cross-entropy**:

$$
\mathcal{L} = -\log(p_{\text{correct}})
$$

where $p_{\text{correct}}$ is the probability the model assigned to the right answer.

**Worked example:** if the model gives "mat" probability 0.30:

$$
\mathcal{L} = -\log(0.30) \approx 1.20
$$

If the model gives "mat" probability 0.90:

$$
\mathcal{L} = -\log(0.90) \approx 0.11
$$

If the model gives "mat" probability 1.00:

$$
\mathcal{L} = -\log(1.00) = 0
$$

The loss is 0 when the model is perfectly confident in the right answer, and grows toward infinity as the probability approaches 0. The $-\log$ function penalizes low probabilities harshly — assigning 1% to the correct answer is much worse than 10%, which is much worse than 50%.

## Gradients: which way to adjust

The loss tells us *how wrong* the model is, but not *how to fix it*. For that, we need **gradients**.

A gradient tells you: "if I increase this weight by a tiny amount, how does the loss change?"

$$
\frac{\partial \mathcal{L}}{\partial w} \approx \frac{\mathcal{L}(w + \epsilon) - \mathcal{L}(w)}{\epsilon}
$$

- If the gradient is positive: increasing $w$ increases the loss (bad) → decrease $w$
- If the gradient is negative: increasing $w$ decreases the loss (good) → increase $w$
- If the gradient is zero: $w$ is at a sweet spot (for now)

**The analogy:** imagine you're blindfolded on a hilly landscape, trying to find the lowest point. You can feel the slope under your feet. The gradient is the slope — it tells you which direction is downhill. You take a step downhill, feel the slope again, and repeat.

## Gradient descent

**Gradient descent** is the algorithm that adjusts all weights simultaneously:

$$
w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}
$$

where $\eta$ (eta) is the **learning rate** — how big a step to take. Too large and you overshoot; too small and learning is slow.

For a model with 7 billion weights, this means computing 7 billion gradients and making 7 billion tiny adjustments. On every training step. For millions of steps.

## How training works

Training is a loop:

1. **Forward pass:** feed a batch of text through the model, get predictions
2. **Compute loss:** compare predictions to the actual next words
3. **Backward pass:** compute the gradient of the loss with respect to every weight
4. **Update:** adjust each weight by a small step in the direction that reduces the loss
5. Repeat

After millions of iterations over billions of words, the weight matrices encode patterns of language — grammar, facts, reasoning strategies — all as numbers.

## What tempura does (and doesn't do)

tempura is an **inference** engine, not a training engine. It runs the forward pass (step 1) — taking a trained model's weight matrices and computing predictions for new input.

tempura does NOT do steps 2–4. It doesn't compute gradients, doesn't update weights, doesn't learn. The learning happened on GPU clusters weeks or months before tempura sees the model.

This distinction matters: inference is simpler than training (no backward pass, no gradient computation), but it still involves all the matrix multiplications, activations, and softmax computations from the previous pages. Making these fast, correct, and verifiable is tempura's purpose.

## The scale of training

To give you a sense of what training involves:

| Aspect | Typical values (7B model) |
|--------|--------------------------|
| Training data | 1–2 trillion tokens of text |
| Training steps | ~1 million gradient updates |
| Training time | 100,000+ GPU-hours |
| Compute cost | $1–5 million |
| Power consumption | ~1 GWh (a small town for a month) |

After training, the model's weights are saved to files. These files — typically 14 GB for a 7B model in float16 — are what tempura loads and runs.

**The asymmetry:** training is enormously expensive but happens once. Inference is cheap per request but happens billions of times. A model trained for $2 million might serve 100 million users over its lifetime. This is why inference efficiency matters — and why tempura focuses on making inference fast, correct, and verifiable.

## Check your understanding

1. If a model assigns probability 0.5 to the correct next word, what is the cross-entropy loss?
2. What happens if the learning rate $\eta$ is too large?
3. Why doesn't tempura need to compute gradients?
4. A model has 7 billion parameters. If each gradient update adjusts every parameter, how many multiplications does one training step require at minimum?

Answers: (1) $-\log(0.5) \approx 0.69$. (2) The weight updates overshoot — the loss oscillates or increases instead of decreasing. (3) tempura does inference (using frozen weights), not training — no backward pass needed. (4) At least 7 billion — one per parameter — plus the forward pass computation.

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| Loss function | Measures how wrong the model is | The signal that drives learning |
| Cross-entropy | $-\log(p_{\text{correct}})$ | The standard loss for language models |
| Gradient | Slope of the loss with respect to a weight | Tells which direction to adjust |
| Gradient descent | $w \leftarrow w - \eta \cdot \nabla \mathcal{L}$ | The algorithm that adjusts weights |
| Training vs inference | Training learns weights; inference uses them | tempura does inference |

**Next:** [Neural Networks](../neural-networks/) — how layers compose into a complete network.
