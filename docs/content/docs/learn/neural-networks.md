+++
title = "6. Neural Networks"
description = "Composing layers into a network — the forward pass through a stack of transformations."
weight = 60
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Stacking layers

In [Functions](../functions/), you saw that one layer is:

$$
\text{output} = \text{activation}(Wx + b)
$$

A **neural network** is a sequence of these layers, where each layer's output becomes the next layer's input:

$$
h_1 = \text{activation}(W_1 x + b_1)
$$
$$
h_2 = \text{activation}(W_2 h_1 + b_2)
$$
$$
y = W_3 h_2 + b_3
$$

The intermediate values $h_1, h_2$ are called **hidden states** — they're the network's internal representation of the input, neither visible to the user nor directly interpretable.

## A concrete example

Let's trace an input through a tiny 2-layer network:

**Input:** $x = [1, 2]$

**Layer 1:** $W_1 = \begin{pmatrix} 0.5 & -0.3 \\\\ 0.2 & 0.8 \\\\ -0.1 & 0.4 \end{pmatrix}$, $b_1 = [0.1, 0, -0.2]$, ReLU activation

$$
W_1 x = \begin{pmatrix} 0.5 - 0.6 \\\\ 0.2 + 1.6 \\\\ -0.1 + 0.8 \end{pmatrix} = \begin{pmatrix} -0.1 \\\\ 1.8 \\\\ 0.7 \end{pmatrix}
$$

$$
W_1 x + b_1 = [-0.1 + 0.1, \; 1.8 + 0, \; 0.7 - 0.2] = [0, \; 1.8, \; 0.5]
$$

$$
h_1 = \text{ReLU}([0, 1.8, 0.5]) = [0, \; 1.8, \; 0.5]
$$

The input started as a 2D vector and is now a 3D hidden state. The network has *expanded* the representation.

**Layer 2:** $W_2 = \begin{pmatrix} 0.3 & 0.6 & -0.2 \\\\ -0.5 & 0.1 & 0.9 \end{pmatrix}$, $b_2 = [0, 0]$

$$
W_2 h_1 = \begin{pmatrix} 0 + 1.08 - 0.1 \\\\ 0 + 0.18 + 0.45 \end{pmatrix} = \begin{pmatrix} 0.98 \\\\ 0.63 \end{pmatrix}
$$

$$
y = [0.98, \; 0.63]
$$

The network transformed $[1, 2]$ into $[0.98, 0.63]$ through a sequence of linear and non-linear operations.

## The forward pass

Running an input through all layers from start to finish is called the **forward pass**. In tempura, the forward pass is the core computation:

1. Look up the word's embedding vector
2. Pass it through 32 (or more) transformer layers
3. Project to vocabulary size
4. Apply softmax to get probabilities

Each of those "transformer layers" is itself a mini-network — but the principle is the same: matrix multiply, activate, repeat.

## Width and depth

**Width** is the dimension of the hidden states. Wider networks can represent more features simultaneously. A 7B model typically uses width $d = 4096$.

**Depth** is the number of layers. Deeper networks can compose more complex transformations. A 7B model typically has 32 layers.

| Model | Parameters | Width ($d$) | Depth (layers) |
|-------|-----------|-------------|----------------|
| GPT-2 Small | 117M | 768 | 12 |
| LLaMA 7B | 7B | 4096 | 32 |
| LLaMA 70B | 70B | 8192 | 80 |

The parameters are mostly in the weight matrices: each layer has several $d \times d$ matrices, and there are many layers.

## Residual connections

Deep networks have a problem: the signal from the input can get lost or distorted as it passes through many layers. **Residual connections** (He et al., 2016) add the input back to the output of each layer:

$$
h_{i+1} = h_i + \text{Layer}(h_i)
$$

Instead of replacing $h_i$ with a completely new vector, the layer only needs to learn the *difference* (the residual). If a layer has nothing useful to add, it can learn to output near-zero, and the signal passes through unchanged.

In modern transformers, every attention block and every FFN block uses a residual connection. The hidden state flows through the entire network as a "residual stream," with each layer adding its contribution.

## Why normalization appears between layers

As you stack more layers, the hidden state values can grow. After 32 layers of matrix multiplications and additions, even small growth per layer compounds. [RMS Normalization](../../foundations/normalization/) constrains the scale of the hidden state at each layer boundary, keeping the values in a numerically well-behaved range.

In a transformer, the pattern is:

$$
h' = h + \text{Attention}(\text{RMSNorm}(h))
$$
$$
h'' = h' + \text{FFN}(\text{RMSNorm}(h'))
$$

Normalize before each sublayer, apply the sublayer, add the residual.

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| Hidden state | Intermediate vector between layers | The network's internal representation |
| Forward pass | Running input through all layers | The computation tempura performs |
| Width | Dimension of hidden states | More features per layer |
| Depth | Number of layers | More complex transformations |
| Residual connection | $h + \text{Layer}(h)$ | Prevents signal loss in deep networks |

**Next:** [Words as Numbers](../words-as-numbers/) — how text becomes vectors.
