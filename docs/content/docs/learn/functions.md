+++
title = "3. Functions"
description = "Linear vs non-linear transformations — why neural networks need both."
weight = 30
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## The linearity problem

In [Matrices](../matrices/), you learned that a matrix transforms a vector: $y = Wx$. This is a **linear transformation** — it can scale, rotate, and project, but it cannot do anything more complex.

Here's the problem: stacking two linear transformations is still linear.

$$
y = W_2 (W_1 x) = (W_2 W_1) x = W' x
$$

No matter how many matrices you multiply, the result is equivalent to a single matrix $W' = W_2 W_1$. A 100-layer network of pure matrix multiplications has the same expressive power as one layer.

To build networks that can learn complex patterns, we need **non-linear functions** between the matrix multiplications.

## What is a function?

A function takes an input and produces an output. You already know many:

- $f(x) = 2x + 3$ (linear — straight line)
- $f(x) = x^2$ (quadratic — parabola)
- $f(x) = \sin(x)$ (oscillating)

A **linear function** has the form $f(x) = ax + b$. Its graph is a straight line. It preserves proportions: doubling the input doubles the change in output.

A **non-linear function** is everything else. Its graph curves. This curvature is what gives neural networks their power — it lets them represent boundaries, thresholds, and complex patterns that straight lines cannot.

## Activation functions

In neural networks, the non-linear function applied after each matrix multiplication is called an **activation function**. It operates on each component of the vector independently:

$$
\text{activation}([z_1, z_2, z_3]) = [\text{activation}(z_1), \text{activation}(z_2), \text{activation}(z_3)]
$$

### ReLU — the simplest

**ReLU** (Rectified Linear Unit) is the most widely used activation:

$$
\text{ReLU}(x) = \max(0, x)
$$

- If $x$ is positive, output $x$
- If $x$ is negative, output $0$

**Worked example:**

$$
\text{ReLU}([-2, 3, -0.5, 7]) = [0, 3, 0, 7]
$$

ReLU is almost linear (it's linear for positive values), but the kink at zero — where negatives become zero — is the non-linearity that matters.

### SiLU — what modern LLMs use

LLaMA, Mistral, and most modern language models use **SiLU** (Sigmoid Linear Unit), also called Swish:

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

**Worked examples:**

| $x$ | $\sigma(x)$ | $\text{SiLU}(x) = x \cdot \sigma(x)$ |
|-----|------------|--------------------------------------|
| $-2$ | $0.12$ | $-0.24$ |
| $0$ | $0.5$ | $0$ |
| $2$ | $0.88$ | $1.76$ |
| $5$ | $0.99$ | $4.95$ |

SiLU is smooth everywhere (no kink like ReLU) and allows small negative values through (unlike ReLU which clips to zero). This helps with gradient flow during training.

**In tempura**, the forward pass applies SiLU in the FFN (feed-forward network) layers between matrix multiplications.

## The sigmoid function

The **sigmoid** $\sigma(x) = \frac{1}{1 + e^{-x}}$ deserves its own mention because it appears everywhere:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

It squashes any real number into the range $(0, 1)$:

| $x$ | $\sigma(x)$ |
|-----|------------|
| $-10$ | $\approx 0.00$ |
| $-2$ | $0.12$ |
| $0$ | $0.50$ |
| $2$ | $0.88$ |
| $10$ | $\approx 1.00$ |

Large positive inputs → output near 1. Large negative inputs → output near 0. The transition happens around $x = 0$.

Sigmoid is closely related to [softmax](../probability/) (which generalizes it to multiple outputs) and appears inside SiLU.

## Putting it together: one layer

A single neural network layer is:

$$
\text{output} = \text{activation}(Wx + b)
$$

1. Multiply the input vector by a weight matrix: $Wx$
2. Add a bias vector: $Wx + b$
3. Apply the activation function to each component

**Worked example** with $W = \begin{pmatrix} 1 & -1 \\\\ 2 & 0 \end{pmatrix}$, $b = \begin{pmatrix} 0 \\\\ -1 \end{pmatrix}$, $x = \begin{pmatrix} 3 \\\\ 2 \end{pmatrix}$, and ReLU:

Step 1: $Wx = \begin{pmatrix} 1 \times 3 + (-1) \times 2 \\\\ 2 \times 3 + 0 \times 2 \end{pmatrix} = \begin{pmatrix} 1 \\\\ 6 \end{pmatrix}$

Step 2: $Wx + b = \begin{pmatrix} 1 + 0 \\\\ 6 + (-1) \end{pmatrix} = \begin{pmatrix} 1 \\\\ 5 \end{pmatrix}$

Step 3: $\text{ReLU}\begin{pmatrix} 1 \\\\ 5 \end{pmatrix} = \begin{pmatrix} 1 \\\\ 5 \end{pmatrix}$ (both positive, so unchanged)

Now stack two of these layers and the composition is non-linear — it can represent curved boundaries that a single matrix cannot. This is the foundation of [Neural Networks](../neural-networks/).

## The exponential function

One function appears constantly in what follows: the **exponential** $e^x$, where $e \approx 2.718$.

$$
e^0 = 1, \quad e^1 \approx 2.72, \quad e^2 \approx 7.39, \quad e^{-1} \approx 0.37, \quad e^{-2} \approx 0.14
$$

Key property: $e^x$ is always positive, and it grows very fast for positive $x$. This makes it useful for converting arbitrary numbers into positive values — which is exactly what [softmax](../probability/) needs to create probability distributions.

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| Linear function | $f(x) = ax + b$ | Matrices are linear — stacking them is still linear |
| Non-linear function | Everything else ($x^2$, ReLU, sigmoid, ...) | Breaks linearity — enables complex patterns |
| Activation function | Non-linearity applied after each matrix multiply | Makes deep networks more powerful than shallow ones |
| Sigmoid | Squashes to $(0, 1)$ | Foundation of softmax and SiLU |

**Next:** [Probability](../probability/) — how to turn numbers into decisions.
