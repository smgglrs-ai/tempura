+++
title = "7. Words as Numbers"
description = "Tokenization, embeddings, and meaning in vector space."
weight = 70
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## The bridge problem

Neural networks operate on numbers — vectors and matrices. But language is made of words. How do you turn "the cat sat on the mat" into vectors that a network can process?

Two steps: **tokenization** (splitting text into pieces) and **embedding** (mapping each piece to a vector).

## Tokenization

A **tokenizer** splits text into **tokens** — usually subword pieces, not full words:

```
"unfortunately" → ["un", "fortunate", "ly"]
"ChatGPT"       → ["Chat", "G", "PT"]
"hello world"   → ["hello", " world"]
```

Each token gets a unique integer ID from the **vocabulary** — a fixed list of 32,000–128,000 tokens that the model knows:

```
"hello"  → 15339
" world" → 1917
"the"    → 279
"cat"    → 4208
```

**Why subwords, not full words?** A word-level vocabulary would need millions of entries to cover every possible word, misspelling, and technical term. Subword tokenization handles unseen words by composing known pieces: "unforgettable" → ["un", "forget", "table"]. The vocabulary stays manageable while covering any text.

**In tempura**, the `Tokenizer` wraps HuggingFace's tokenizers library, which handles the text → token ID conversion.

## Embeddings

Each token ID maps to a vector through an **embedding table** — a matrix $E \in \mathbb{R}^{V \times d}$ where $V$ is the vocabulary size and $d$ is the embedding dimension:

$$
\text{embedding}(\text{token\_id}) = E[\text{token\_id}]
$$

This is just a table lookup: token 15339 → row 15339 of the matrix. The result is a $d$-dimensional vector (typically $d = 4096$).

**Worked example** with a tiny vocabulary of 4 tokens and $d = 3$:

$$
E = \begin{pmatrix} 0.2 & -0.1 & 0.5 \\\\ 0.8 & 0.3 & -0.4 \\\\ -0.3 & 0.7 & 0.1 \\\\ 0.5 & 0.5 & 0.5 \end{pmatrix}
$$

Token 0 ("the") → $[0.2, -0.1, 0.5]$

Token 2 ("cat") → $[-0.3, 0.7, 0.1]$

These vectors are **learned during training**. The model adjusts the embedding values so that tokens used in similar contexts end up with similar vectors.

## Meaning in vector space

After training, embeddings capture semantic relationships as geometric properties:

**Similar words are nearby.** "cat" and "dog" will have similar embedding vectors — their dot product (from [Vectors](../vectors/)) is high. "cat" and "democracy" will be far apart.

**Relationships are directions.** The famous example: the vector from "king" to "queen" is approximately the same as the vector from "man" to "woman":

$$
\vec{\text{queen}} - \vec{\text{king}} \approx \vec{\text{woman}} - \vec{\text{man}}
$$

This means the embedding space has a "gender" direction, a "tense" direction, a "formality" direction — all learned from data, not programmed.

**Analogies are arithmetic.** $\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$. This works because the embedding space is organized such that semantic relationships correspond to geometric translations.

## From text to a sequence of vectors

Putting it together, a sentence becomes a matrix — one row per token:

```
"the cat sat" → [279, 4208, 3290]
```

$$
\begin{pmatrix} E[279] \\\\ E[4208] \\\\ E[3290] \end{pmatrix} = \begin{pmatrix} 0.2 & -0.1 & 0.5 & \cdots \\\\ -0.3 & 0.7 & 0.1 & \cdots \\\\ 0.1 & 0.4 & -0.2 & \cdots \end{pmatrix}
$$

This matrix (shape: $\text{seq\_len} \times d$) is the input to the transformer. Each row is a token's representation, and the transformer's job is to transform these rows so that each one "knows about" the other tokens through [attention](../the-transformer/).

## The output layer: vectors back to words

At the end of the network, the model produces a hidden state vector for each position. The last position's vector needs to become a probability distribution over the vocabulary — "what's the next token?"

This is done by multiplying the hidden state by the embedding table (or a similar matrix) transposed:

$$
\text{logits} = h \cdot E^T \in \mathbb{R}^V
$$

Each entry in the logits vector is a dot product between the hidden state and one token's embedding — measuring how "similar" the model's output is to each word. Then [softmax](../probability/) converts these logits into probabilities.

The circle is complete: words → token IDs → embedding vectors → transformer → hidden state → dot product with embeddings → logits → softmax → probabilities → next word.

## Why subword tokenization is clever

Subword tokenization is a compromise between word-level and character-level:

| Approach | Vocabulary size | Handles new words? | Sequence length |
|----------|----------------|-------------------|----------------|
| Character-level | ~256 | Yes (any character) | Very long (many tokens per word) |
| Word-level | ~1 million+ | No (unknown words → `[UNK]`) | Short (one token per word) |
| Subword (BPE/SentencePiece) | 32K–128K | Yes (compose from pieces) | Medium |

The algorithm (**Byte Pair Encoding** or **SentencePiece**) learns which character sequences appear most often in the training data and merges them into single tokens. Common words like "the" become single tokens; rare words like "unforgettable" are split into known pieces.

This means the tokenizer has its own kind of learned knowledge — it "knows" that "un-" is a prefix, "-ing" is a suffix, and "quantum" appears often enough to be a single token.

## Check your understanding

1. Why does the model use subword tokens instead of whole words?
2. If the embedding dimension is 4096 and the vocabulary is 32,000 tokens, how large is the embedding table (in number of parameters)?
3. Why does the output projection compute a dot product between the hidden state and each token's embedding?
4. If "king" - "man" + "woman" ≈ "queen" in embedding space, what does this suggest about how the model represents gender?

Answers: (1) Whole words would require a vocabulary too large to handle, and couldn't represent novel words. Subwords can compose any word from known pieces. (2) $32{,}000 \times 4096 = 131$ million parameters. (3) The dot product measures similarity (from [Vectors](../vectors/)) — the hidden state should be most similar to the embedding of the correct next token. (4) Gender is represented as a direction in the embedding space — you can move along that direction to change gender while preserving other meaning.

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| Tokenization | Text → token IDs | Makes text processable as numbers |
| Embedding table | Token ID → vector ($V \times d$) | Gives each token a learnable representation |
| Subword tokenization | BPE/SentencePiece — learned vocabulary | Handles any text with a fixed vocabulary |
| Semantic similarity | Nearby vectors = similar meaning | Emergent property of training |
| Output projection | Hidden state → vocabulary scores via dot product | Converts network output back to word probabilities |

**Next:** [The Transformer](../the-transformer/) — the architecture that puts it all together.
