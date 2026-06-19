+++
title = "2. Matrices"
description = "How to transform vectors — matrix multiplication as the building block of neural networks."
weight = 20
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## The problem: transforming information

In [Vectors](../vectors/), you learned to represent things as lists of numbers. But a model needs to *process* those numbers — transform them, combine them, extract new information from them.

Consider a simple task: a university wants to compute two scores from each student's grades:

- **STEM score** = 1.5 × math + 1.5 × science + 0 × English
- **Humanities score** = 0 × math + 0.5 × science + 1.5 × English

For a student with grades $[80, 90, 70]$:

$$
\text{STEM} = 1.5 \times 80 + 1.5 \times 90 + 0 \times 70 = 120 + 135 + 0 = 255
$$
$$
\text{Humanities} = 0 \times 80 + 0.5 \times 90 + 1.5 \times 70 = 0 + 45 + 105 = 150
$$

Each score is a dot product between the grades and a set of weights. We've turned a 3-number input (grades) into a 2-number output (scores). This is exactly what a matrix does — it bundles multiple dot products together.

## What is a matrix?

A **matrix** is a rectangular grid of numbers, arranged in rows and columns:

$$
W = \begin{pmatrix} 1.5 & 1.5 & 0 \\\\ 0 & 0.5 & 1.5 \end{pmatrix}
$$

This matrix has **2 rows** and **3 columns** — we call it a "$2 \times 3$ matrix." Each row defines one output computation:

- Row 1: $[1.5, 1.5, 0]$ → computes the STEM score
- Row 2: $[0, 0.5, 1.5]$ → computes the humanities score

## Matrix-vector multiplication

Multiplying the matrix $W$ by the grade vector $x$ produces the output:

$$
\begin{pmatrix} 1.5 & 1.5 & 0 \\\\ 0 & 0.5 & 1.5 \end{pmatrix} \begin{pmatrix} 80 \\\\ 90 \\\\ 70 \end{pmatrix} = \begin{pmatrix} 255 \\\\ 150 \end{pmatrix}
$$

**The rule:** each row of the matrix takes a dot product with the input vector. Row 1 · input = output 1. Row 2 · input = output 2.

Let's do another example step by step:

$$
\begin{pmatrix} 2 & 0 \\\\ 1 & 3 \\\\ -1 & 1 \end{pmatrix} \begin{pmatrix} 4 \\\\ 5 \end{pmatrix} = \begin{pmatrix} ? \\\\ ? \\\\ ? \end{pmatrix}
$$

Row 1: $2 \times 4 + 0 \times 5 = 8 + 0 = 8$

Row 2: $1 \times 4 + 3 \times 5 = 4 + 15 = 19$

Row 3: $(-1) \times 4 + 1 \times 5 = -4 + 5 = 1$

$$
\text{Result: } \begin{pmatrix} 8 \\\\ 19 \\\\ 1 \end{pmatrix}
$$

A $3 \times 2$ matrix transforms a 2-component vector into a 3-component vector. The matrix changes the **dimension** of the vector — from 2D to 3D.

**Try it yourself:** compute $\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \end{pmatrix} \begin{pmatrix} 5 \\\\ 6 \end{pmatrix}$

Answer: Row 1: $1 \times 5 + 2 \times 6 = 5 + 12 = 17$. Row 2: $3 \times 5 + 4 \times 6 = 15 + 24 = 39$. Result: $\begin{pmatrix} 17 \\\\ 39 \end{pmatrix}$

## The dimension rule

A matrix with $m$ rows and $n$ columns transforms a vector with $n$ components into a vector with $m$ components:

$$
\underbrace{W}_{m \times n} \times \underbrace{x}_{n} = \underbrace{y}_{m}
$$

The number of **columns** in the matrix must match the number of **components** in the vector. If they don't match, the multiplication is undefined — you'll get an error.

| Matrix size | Input size | Output size | What it does |
|------------|-----------|-------------|-------------|
| $2 \times 3$ | $3$ | $2$ | Compresses (3D → 2D) |
| $3 \times 3$ | $3$ | $3$ | Transforms (keeps dimension) |
| $5 \times 3$ | $3$ | $5$ | Expands (3D → 5D) |
| $1 \times 3$ | $3$ | $1$ | Reduces to a single number |

## Matrix-matrix multiplication

What if you want to transform multiple vectors at once? Or chain two transformations together? That's matrix-matrix multiplication.

Two matrices $A$ ($m \times p$) and $B$ ($p \times n$) multiply to produce a result $C$ ($m \times n$):

$$
C_{ij} = \sum_{k=1}^{p} A_{ik} \times B_{kj}
$$

That looks abstract. Here's what it means: entry $C_{ij}$ (row $i$, column $j$ of the result) is the dot product of row $i$ of $A$ with column $j$ of $B$.

**Worked example:**

$$
\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \end{pmatrix} \times \begin{pmatrix} 5 & 6 \\\\ 7 & 8 \end{pmatrix} = ?
$$

$C_{11}$: row 1 of $A$ · column 1 of $B$ = $1 \times 5 + 2 \times 7 = 5 + 14 = 19$

$C_{12}$: row 1 of $A$ · column 2 of $B$ = $1 \times 6 + 2 \times 8 = 6 + 16 = 22$

$C_{21}$: row 2 of $A$ · column 1 of $B$ = $3 \times 5 + 4 \times 7 = 15 + 28 = 43$

$C_{22}$: row 2 of $A$ · column 2 of $B$ = $3 \times 6 + 4 \times 8 = 18 + 32 = 50$

$$
\text{Result: } \begin{pmatrix} 19 & 22 \\\\ 43 & 50 \end{pmatrix}
$$

The dimension rule extends: the inner dimensions must match.

$$
\underbrace{A}_{m \times \colorbox{#fff3cd}{p}} \times \underbrace{B}_{\colorbox{#fff3cd}{p} \times n} = \underbrace{C}_{m \times n}
$$

If $A$ is $2 \times 3$ and $B$ is $3 \times 4$, the result is $2 \times 4$. If $A$ is $2 \times 3$ and $B$ is $5 \times 4$, the multiplication is impossible — 3 ≠ 5.

## Matrices as transformations

A matrix isn't just a grid of numbers — it defines a *function* that transforms space. Different matrices do different things:

**Scaling:** multiplies each dimension independently:

$$
\begin{pmatrix} 2 & 0 \\\\ 0 & 3 \end{pmatrix} \begin{pmatrix} x \\\\ y \end{pmatrix} = \begin{pmatrix} 2x \\\\ 3y \end{pmatrix}
$$

Stretches $x$ by 2 and $y$ by 3.

**Rotation:** rotates a vector by angle $\theta$:

$$
\begin{pmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{pmatrix}
$$

This is exactly what [RoPE](../../foundations/positional-encoding/) uses to encode word position in a sentence — rotating vectors by an angle proportional to their position.

**Projection:** a non-square matrix changes dimensions. A $2 \times 100$ matrix collapses a 100-dimensional vector down to 2 dimensions (like summarizing a complex thing into two numbers). A $100 \times 2$ matrix does the opposite — expands a simple representation into a rich one.

## Why this matters for AI

Here is the key fact: **almost everything a language model does is matrix multiplication.**

| Operation in a language model | What the matrix does |
|------------------------------|---------------------|
| Embedding lookup | Maps a token ID to a vector |
| Query/Key/Value projection | Converts hidden state into attention vectors |
| Attention output projection | Combines attention results |
| Feed-forward network (up) | Expands to higher dimension |
| Feed-forward network (down) | Compresses back to model dimension |
| Final output projection | Converts hidden state to vocabulary scores |

A 7B parameter model stores 7 billion numbers arranged in these matrices. The matrices *are* the model. When people say "the model learned X," they mean "the numbers in the model's matrices are arranged so that the math produces the right output for inputs related to X."

tempura's `MatMul` kernel implements matrix multiplication, and [Kani proves](../../foundations/reading-kani-harnesses/) that the index arithmetic — which determines where in the matrix to read each number — never goes out of bounds.

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| Matrix | Rectangular grid of numbers ($m \times n$) | Defines a transformation |
| Matrix-vector multiply | One dot product per row | Transforms one vector |
| Matrix-matrix multiply | Row-of-A · column-of-B for each entry | Transforms multiple vectors / chains transformations |
| Dimension rule | Inner dimensions must match | The fundamental correctness check |
| Transformation view | Matrices rotate, scale, project, compress, expand | Every model operation is a matrix multiply |

**Next:** [Functions](../functions/) — what happens between matrix multiplications, and why it matters.
