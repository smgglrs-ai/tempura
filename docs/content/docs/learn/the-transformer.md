+++
title = "8. The Transformer"
description = "Attention as the core mechanism — how a model understands context."
weight = 80
template = "docs/page.html"

[extra]
part = "math"
toc = true
math = true
+++

## Before the transformer: what was missing

In [Neural Networks](../neural-networks/), you learned that layers process vectors sequentially. In [Words as Numbers](../words-as-numbers/), you learned that words become vectors. But there's a problem we haven't solved yet.

Consider the sentence: "The bank by the river was steep." Now consider: "I went to the bank to deposit money."

The word "bank" appears in both sentences, but means completely different things. If each word is just a fixed vector (from the embedding table), "bank" has the same vector in both sentences. The model can't tell the difference.

The model needs a way for each word to *look at the other words in the sentence* and adjust its meaning based on context. The word "bank" next to "river" should become a riverbank-vector. The word "bank" next to "deposit" should become a financial-institution-vector.

This is what **attention** does.

## The library analogy

Imagine a library with a reference desk. You walk in with a question (a **query**). Each book has a label on its spine describing its contents (a **key**). Inside each book is the actual information (a **value**).

The librarian compares your question to every book's label, figures out which books are most relevant, and gives you a weighted blend of information from those books. Books whose labels match your question well contribute more; irrelevant books contribute almost nothing.

Replace "you" with a word in a sentence, "books" with all other words in the sentence, and "the librarian" with a matrix multiplication + softmax, and you have the attention mechanism.

## The problem attention solves

Consider the sentence: "The animal didn't cross the street because **it** was too tired."

What does "it" refer to? The animal. But how does a neural network know this?

The hidden state at "it" is just a vector — it doesn't contain the word "animal" unless the network explicitly looks back and incorporates information from that earlier position. **Attention** is the mechanism that does this looking-back. After attention, the vector for "it" has been updated to include information from "animal" — because "it" queried for a noun subject, and "animal" was the best match.

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

The results are concatenated and projected back to the model dimension. See [Attention](attention/) for the full details and GQA.

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

Every operation here — RMSNorm, MatMul, RoPE, Attention — is a kernel in tempura, with [Kani proofs](reading-kani-harnesses/) for index safety.

## The feed-forward network (FFN)

After attention, each token has gathered information from the other tokens. But it hasn't *processed* that information yet — it's just a blend. The **feed-forward network** (FFN) within each transformer layer does the processing.

The FFN is two [matrix multiplications](../matrices/) with a [non-linear activation](../functions/) in between:

1. **Expand:** multiply by a matrix that increases the dimension (typically 4x). A 4096-dimensional hidden state becomes 16,384-dimensional.
2. **Activate:** apply SiLU (from [Functions](../functions/)) — the non-linear function that lets the network learn complex patterns.
3. **Compress:** multiply by another matrix that brings the dimension back down to 4096.

Why expand and then compress? The higher-dimensional intermediate space gives the network more room to compute. It's like doing scratch work on a large whiteboard and then writing the final answer on a small card.

In modern LLMs (LLaMA, Mistral), the FFN uses a **gated** variant:

$$
\text{FFN}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}) W_{\text{down}}
$$

where $\odot$ means element-wise multiplication. The gate controls how much of the expanded representation passes through — another learned non-linearity.

## The full model: end to end

Let's trace what happens when you type "The cat sat on the" into a language model:

**Step 1: Tokenize.** Split the text into tokens and look up their IDs:

```
"The" → 450   "cat" → 4208   "sat" → 3290   "on" → 373   "the" → 279
```

**Step 2: Embed.** Look up each token ID in the embedding table (a matrix with one row per vocabulary entry):

```
450  → [0.12, -0.34, 0.56, ..., 0.08]   (4096 numbers)
4208 → [-0.23, 0.71, 0.03, ..., -0.15]  (4096 numbers)
3290 → [0.45, 0.11, -0.28, ..., 0.33]   (4096 numbers)
373  → [-0.08, 0.52, 0.19, ..., -0.41]  (4096 numbers)
279  → [0.12, -0.34, 0.56, ..., 0.08]   (4096 numbers)
```

Now we have a matrix of shape $5 \times 4096$ — five token vectors, each with 4096 dimensions.

**Step 3: 32 transformer layers.** Each layer refines these vectors:

- Attention lets each word look at the other words. After layer 1, "sat" knows it's related to "cat." After deeper layers, "the" (the second one) knows it's part of "on the ___" and can anticipate what comes next.
- The FFN processes the gathered information, computing more abstract features.
- Normalization and residual connections keep the numbers well-behaved.

After all 32 layers, the last token's vector has been transformed from a simple embedding of "the" into a rich representation that encodes the entire sentence's meaning and context.

**Step 4: Project to vocabulary.** Multiply the last token's 4096-dimensional hidden state by the output matrix (vocabulary size × 4096) to get one score per word in the vocabulary:

```
"mat"   → 8.3
"floor" → 6.1
"dog"   → 3.2
"table" → 5.8
...     → (32,000 scores total)
```

**Step 5: Sample.** Apply [softmax](../probability/) to convert scores into probabilities, then sample:

```
"mat"   → 0.42
"floor" → 0.18
"table" → 0.12
...
```

With temperature 0 (greedy): output "mat." With temperature 0.7: probably "mat," but sometimes "floor" or "table."

**Step 6: Repeat.** Append "mat" to the input and run steps 2–5 again to generate the next word. Continue until the model outputs a stop token or reaches the length limit.

The "intelligence" is distributed across all 32 transformer layers. Each layer adds a bit of understanding — early layers handle syntax and word relationships, later layers handle higher-level reasoning and knowledge retrieval.

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| Query, Key, Value | Three vectors per token | The mechanism for looking at other tokens |
| Attention scores | Dot product of Q and K | Measures relevance between positions |
| Causal mask | Prevents seeing future tokens | Essential for left-to-right generation |
| Multi-head | Multiple independent attention computations | Captures different types of relationships |
| Transformer layer | Attention + FFN + residuals + normalization | The repeating building block |

**Next:** [Inference](../inference/) — what happens when you use a model, and what tempura does.
