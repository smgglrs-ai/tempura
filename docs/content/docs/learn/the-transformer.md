+++
title = "8. The Transformer"
description = "Attention as the core mechanism — how a model understands context."
weight = 80
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## The problem attention solves

Consider the sentence: "The animal didn't cross the street because **it** was too tired."

What does "it" refer to? The animal. But how does a neural network know this?

The hidden state at "it" is just a vector — it doesn't contain the word "animal" unless the network explicitly looks back and incorporates information from that earlier position. **Attention** is the mechanism that does this looking-back.

## Queries, keys, and values

Every token in the sequence gets three vectors, computed by [matrix multiplication](../matrices/) from its hidden state $h$:

$$
q = h W^Q \qquad k = h W^K \qquad v = h W^V
$$

Think of it like a search engine:

- **Query** ($q$): "what am I looking for?" — what information does this token need?
- **Key** ($k$): "what do I contain?" — what information does this token offer?
- **Value** ($v$): "what do I provide?" — the actual content to retrieve if matched

The attention mechanism computes how well each query matches each key (using [dot products](../vectors/)), then retrieves a weighted combination of the values.

## The attention computation

For a sequence of $n$ tokens:

**Step 1: Compute scores.** Each query's dot product with every key:

$$
S_{ij} = q_i \cdot k_j
$$

Token $i$ asks: "how relevant is token $j$ to me?"

**Step 2: Scale.** Divide by $\sqrt{d_k}$ where $d_k$ is the dimension of the key vectors:

$$
S_{ij} \leftarrow \frac{S_{ij}}{\sqrt{d_k}}
$$

This prevents the scores from getting too large (which would push [softmax](../probability/) into saturation).

**Step 3: Mask.** For language generation, token $i$ should only attend to tokens $j \leq i$ — it can't see the future. Set $S_{ij} = -\infty$ for $j > i$ so softmax gives them zero probability.

**Step 4: Normalize.** Apply softmax across each row:

$$
\alpha_{ij} = \text{softmax}(S_i)_j
$$

Now each row sums to 1 — the attention weights form a probability distribution over positions.

**Step 5: Aggregate.** Weighted sum of value vectors:

$$
\text{output}_i = \sum_j \alpha_{ij} \cdot v_j
$$

Each token's output is a blend of all values, weighted by how much its query matched each key.

## A worked example

Three tokens with 2-dimensional Q, K, V (unrealistically small, for clarity):

$$
Q = \begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\\\ 1 & 1 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\\\ 0.5 & 0.5 \end{pmatrix}, \quad V = \begin{pmatrix} 10 & 0 \\\\ 0 & 10 \\\\ 5 & 5 \end{pmatrix}
$$

**Scores** ($QK^T$, then scale by $1/\sqrt{2} \approx 0.707$):

$$
QK^T = \begin{pmatrix} 1 & 0 & 0.5 \\\\ 0 & 1 & 0.5 \\\\ 1 & 1 & 1 \end{pmatrix} \xrightarrow{\div \sqrt{2}} \begin{pmatrix} 0.71 & 0 & 0.35 \\\\ 0 & 0.71 & 0.35 \\\\ 0.71 & 0.71 & 0.71 \end{pmatrix}
$$

**Causal mask** (token can only see itself and earlier tokens — zero out upper triangle):

$$
\begin{pmatrix} 0.71 & -\infty & -\infty \\\\ 0 & 0.71 & -\infty \\\\ 0.71 & 0.71 & 0.71 \end{pmatrix}
$$

**Softmax** (per row):

Row 1: $\text{softmax}([0.71]) = [1.0]$ → only sees itself

Row 2: $\text{softmax}([0, 0.71]) = [0.33, 0.67]$

Row 3: $\text{softmax}([0.71, 0.71, 0.71]) = [0.33, 0.33, 0.33]$

**Output** (weighted sum of V):

Row 3: $0.33 \times [10, 0] + 0.33 \times [0, 10] + 0.33 \times [5, 5] = [5, 5]$

Token 3's query matched all keys equally, so its output is the average of all values. A real model would learn Q, K, V projections that create more interesting attention patterns.

## Multi-head attention

One set of Q, K, V captures one type of relationship. **Multi-head attention** runs multiple independent attention computations ("heads") in parallel:

- Head 1 might attend to syntactic structure (subject → verb)
- Head 2 might attend to nearby words (local context)
- Head 3 might attend to semantic similarity (related concepts)

The results are concatenated and projected back to the model dimension. See [Attention](../../foundations/attention/) for the full details and GQA.

## One transformer layer

A complete transformer layer combines attention with a feed-forward network:

```
Input hidden state (h)
    │
    ├── RMSNorm(h)
    │       │
    │       ├── Q, K, V = MatMul(norm_h, W_Q/K/V)
    │       ├── Apply RoPE to Q, K (position encoding)
    │       ├── Attention(Q, K, V)
    │       └── MatMul(attn_output, W_O)
    │
    ├── + (residual connection: h + attention_output)
    │
    ├── RMSNorm(h')
    │       │
    │       ├── gate = MatMul(norm_h', W_gate)
    │       ├── up = MatMul(norm_h', W_up)
    │       ├── SiLU(gate) × up
    │       └── MatMul(result, W_down)
    │
    └── + (residual connection: h' + ffn_output)
    
Output hidden state (h'')
```

Every operation here — RMSNorm, MatMul, RoPE, Attention — is a kernel in tempura, with [Kani proofs](../../foundations/reading-kani-harnesses/) for index safety.

## The full model

A language model is:

1. **Embedding:** token IDs → vectors (from [Words as Numbers](../words-as-numbers/))
2. **N transformer layers** (typically 32–80): each refining the hidden states
3. **Final RMSNorm:** normalize the output
4. **Output projection:** hidden state → vocabulary logits (one per word)
5. **Softmax + sampling:** logits → probabilities → next token (from [Probability](../probability/))

The "intelligence" is distributed across all those transformer layers. Each layer adds a bit of understanding — early layers handle syntax and word relationships, later layers handle higher-level reasoning and knowledge.

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| Query, Key, Value | Three vectors per token | The mechanism for looking at other tokens |
| Attention scores | Dot product of Q and K | Measures relevance between positions |
| Causal mask | Prevents seeing future tokens | Essential for left-to-right generation |
| Multi-head | Multiple independent attention computations | Captures different types of relationships |
| Transformer layer | Attention + FFN + residuals + normalization | The repeating building block |

**Next:** [Inference](../inference/) — what happens when you use a model, and what tempura does.
