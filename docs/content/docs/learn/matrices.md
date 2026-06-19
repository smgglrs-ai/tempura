+++
title = "2. Matrices"
description = "How to transform vectors — matrix multiplication as the building block of neural networks."
weight = 20
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## From one vector to another

In [Vectors](../vectors/), we represented things as lists of numbers. Now we need to *transform* them — convert a vector in one space into a vector in another space.

Suppose a student's raw scores are $[80, 90, 70]$ (math, science, English) and you want to compute a weighted GPA where math counts double:

$$
\text{GPA} = 2 \times 80 + 1 \times 90 + 1 \times 70 = 160 + 90 + 70 = 320
$$

That's a dot product: $[2, 1, 1] \cdot [80, 90, 70] = 320$. One dot product converts a 3D vector into a single number.

What if you want *multiple* outputs — say, a "STEM score" and a "humanities score"?

$$
\text{STEM} = 1.5 \times 80 + 1.5 \times 90 + 0 \times 70 = 255
$$
$$
\text{humanities} = 0 \times 80 + 0.5 \times 90 + 1.5 \times 70 = 150
$$

Two dot products, each producing one output. We can write this more compactly as a **matrix**:

$$
\begin{pmatrix} 1.5 & 1.5 & 0 \\\\ 0 & 0.5 & 1.5 \end{pmatrix} \begin{pmatrix} 80 \\\\ 90 \\\\ 70 \end{pmatrix} = \begin{pmatrix} 255 \\\\ 150 \end{pmatrix}
$$

## What is a matrix?

A **matrix** is a rectangular grid of numbers. An $m \times n$ matrix has $m$ rows and $n$ columns:

$$
W = \begin{pmatrix} 1.5 & 1.5 & 0 \\\\ 0 & 0.5 & 1.5 \end{pmatrix}
$$

This is a $2 \times 3$ matrix — 2 rows, 3 columns. Each row defines one output: the first row computes the STEM score, the second computes the humanities score.

## Matrix-vector multiplication

Multiplying a matrix $W$ (size $m \times n$) by a vector $x$ (size $n$) produces a vector of size $m$:

$$
(Wx)_i = \sum_{j=1}^{n} W_{ij} \times x_j
$$

Each output is a dot product of one row of $W$ with $x$. The matrix is a **bundle of dot products**.

**Worked example:**

$$
\begin{pmatrix} 2 & 0 \\\\ 1 & 3 \\\\ -1 & 1 \end{pmatrix} \begin{pmatrix} 4 \\\\ 5 \end{pmatrix} = \begin{pmatrix} 2 \times 4 + 0 \times 5 \\\\ 1 \times 4 + 3 \times 5 \\\\ -1 \times 4 + 1 \times 5 \end{pmatrix} = \begin{pmatrix} 8 \\\\ 19 \\\\ 1 \end{pmatrix}
$$

A $3 \times 2$ matrix transforms a 2D vector into a 3D vector.

**Try it:** compute $\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \end{pmatrix} \begin{pmatrix} 5 \\\\ 6 \end{pmatrix}$.

Answer: $\begin{pmatrix} 1 \times 5 + 2 \times 6 \\\\ 3 \times 5 + 4 \times 6 \end{pmatrix} = \begin{pmatrix} 17 \\\\ 39 \end{pmatrix}$

## Matrix-matrix multiplication

What if you want to transform *multiple vectors at once*? Stack them as columns (or rows) and multiply two matrices:

$$
(AB)_{ij} = \sum_{k=1}^{p} A_{ik} \times B_{kj}
$$

For $A$ ($m \times p$) and $B$ ($p \times n$), the result is $m \times n$. The inner dimensions must match — $A$'s columns must equal $B$'s rows.

**Worked example:**

$$
\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \end{pmatrix} \begin{pmatrix} 5 & 6 \\\\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 1 \times 5 + 2 \times 7 & 1 \times 6 + 2 \times 8 \\\\ 3 \times 5 + 4 \times 7 & 3 \times 6 + 4 \times 8 \end{pmatrix} = \begin{pmatrix} 19 & 22 \\\\ 43 & 50 \end{pmatrix}
$$

**This is the most important operation in deep learning.** When a language model processes your text, it performs millions of matrix multiplications per token. tempura's `MatMul` kernel implements exactly this operation — and [Kani proves](../../foundations/reading-kani-harnesses/) the index arithmetic is safe.

## Dimensions rule

The dimension rule for matrix multiplication is simple but strict:

$$
(m \times \colorbox{#fff3cd}{p}) \times (\colorbox{#fff3cd}{p} \times n) = (m \times n)
$$

The highlighted dimensions must match. If they don't, the multiplication is undefined. In code, this is a runtime error; in tempura, it returns a `DimensionMismatch` error.

**A useful mental model:** the matrix eats the inner dimension and produces the outer dimensions.

## Matrices as transformations

A matrix isn't just a grid of numbers — it's a *function* that transforms vectors. Different matrices do different things:

**Scaling:** $\begin{pmatrix} 2 & 0 \\\\ 0 & 3 \end{pmatrix}$ stretches $x$ by 2 and $y$ by 3.

**Rotation:** $\begin{pmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{pmatrix}$ rotates a 2D vector by angle $\theta$. This is exactly what [RoPE](../../foundations/positional-encoding/) does to encode position — it rotates query and key vectors by an angle proportional to their position in the sequence.

**Projection:** a $2 \times 3$ matrix projects a 3D vector into 2D — it reduces dimensions. A $3 \times 2$ matrix does the opposite — it increases dimensions. In transformers, projection matrices convert word vectors into queries, keys, and values for [attention](../../foundations/attention/).

## Why this matters for AI

In a language model, almost every computation is a matrix multiplication:

| Operation | What the matrix does | Size |
|-----------|---------------------|------|
| Embedding lookup | Map token ID → vector | $V \times d$ (vocabulary × dimension) |
| QKV projection | Word vector → query, key, value | $d \times 3d$ |
| Attention output | Combine attention results | $d \times d$ |
| FFN layers | Non-linear transformation | $d \times 4d$, then $4d \times d$ |
| Final projection | Hidden state → vocabulary scores | $d \times V$ |

A 7B parameter model stores 7 billion numbers in these matrices. The matrices *are* the model — they encode everything the model has learned. When we say "the model knows X," we mean "the model's weight matrices produce the right outputs for inputs related to X."

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| Matrix | Grid of numbers ($m \times n$) | Defines a transformation |
| Matrix-vector multiply | Bundle of dot products | Transforms one vector |
| Matrix-matrix multiply | Bundle of matrix-vector products | Transforms many vectors at once |
| Dimension matching | Inner dimensions must agree | The basic correctness check |

**Next:** [Functions](../functions/) — what happens between matrix multiplications.
