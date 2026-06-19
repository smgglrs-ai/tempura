+++
title = "0.5 Reading Math Notation"
description = "A guide to the symbols used in this course — so you never have to guess what ∑ or ∈ means."
weight = 7
template = "docs/page.html"

[extra]
part = "math"
toc = true
math = true
+++

## Why this page exists

Mathematics has its own shorthand. Mathematicians write $\sum_{i=1}^{n} a_i$ instead of "$a_1 + a_2 + \cdots + a_n$" because it's shorter. But if you haven't seen $\sum$ before, it's a wall.

This page explains every symbol we use in this course. You don't need to memorize them — come back and look them up whenever you see something unfamiliar. We also make a promise: **every time we introduce compact notation, we show the expanded version first.**

## Subscripts and superscripts

**Subscripts** (small text below) label things. $a_1$ means "the first element of $a$." $a_2$ means the second. $a_i$ means "the $i$-th element" — where $i$ could be any position.

$$
a = [5, 8, 3] \implies a_1 = 5, \quad a_2 = 8, \quad a_3 = 3
$$

**Superscripts** (small text above) usually mean exponents (powers). $x^2$ means $x \times x$. $x^3$ means $x \times x \times x$.

$$
3^2 = 9, \quad 2^4 = 16, \quad 5^1 = 5, \quad 7^0 = 1
$$

**Exception:** in this course, superscripts on matrices sometimes mean something else — $W^Q$ means "the matrix called $W$ that produces queries," not "$W$ raised to the power $Q$." Context makes it clear: if the superscript is a letter like $Q$, $K$, $V$, or $T$, it's a label, not an exponent.

**Transpose:** $A^T$ means "flip the matrix — swap rows and columns." A $3 \times 2$ matrix becomes $2 \times 3$.

## The sum symbol: $\sum$

$\sum$ (capital Greek sigma) means "add up." It always has a variable, a starting point, and an ending point:

$$
\sum_{i=1}^{4} i = 1 + 2 + 3 + 4 = 10
$$

Read this as: "sum $i$ from 1 to 4" — that is, let $i$ take the values 1, 2, 3, 4 and add them all up.

More examples:

$$
\sum_{i=1}^{3} i^2 = 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
$$

$$
\sum_{j=1}^{3} a_j b_j = a_1 b_1 + a_2 b_2 + a_3 b_3
$$

This last one is the [dot product](../vectors/) — the most important operation in this course.

**When you see $\sum$, just expand it.** Replace the variable with each value from the bottom number to the top number, and add.

## Greek letters

Mathematicians use Greek letters when they run out of Latin ones (or want to signal "this is a parameter, not a variable"). Here are the ones we use:

| Letter | Name | How we use it |
|--------|------|--------------|
| $\alpha$ | alpha | Attention weights (how much to attend to each word) |
| $\beta$ | beta | Bias parameter in normalization |
| $\gamma$ | gamma | Scale parameter in normalization |
| $\epsilon$ | epsilon | A tiny number to prevent division by zero (like $0.00001$) |
| $\eta$ | eta | Learning rate (how big a step to take during training) |
| $\theta$ | theta | An angle (used in [RoPE](../positional-encoding/)), or model parameters in general |
| $\sigma$ | sigma | The sigmoid function, or standard deviation |
| $\mu$ | mu | Mean (average) |

You don't need to memorize these. Each page introduces them with their meaning when they first appear. This table is just a reference to come back to.

## Set notation

Sets are collections of things. We use a few symbols:

| Symbol | Meaning | Example |
|--------|---------|---------|
| $\in$ | "belongs to" or "is an element of" | $3 \in \{1, 2, 3, 4\}$ — 3 is in the set |
| $\mathbb{R}$ | The set of all real numbers | $x \in \mathbb{R}$ means $x$ is any real number |
| $\mathbb{R}^n$ | The set of all vectors with $n$ real-number components | $v \in \mathbb{R}^3$ means $v$ is a 3D vector like $[1, 2, 3]$ |
| $\subseteq$ | "is a subset of" | $\{1, 2\} \subseteq \{1, 2, 3\}$ |
| $\forall$ | "for all" or "for every" | $\forall x > 0$ means "for every positive $x$" |

**The most common one you'll see:** $Q \in \mathbb{R}^{n \times d}$ means "$Q$ is a matrix with $n$ rows and $d$ columns of real numbers." Read it as: "Q is an $n$-by-$d$ matrix."

## Comparison and approximation

| Symbol | Meaning | Example |
|--------|---------|---------|
| $=$ | Equals (exactly) | $2 + 3 = 5$ |
| $\approx$ | Approximately equals | $\pi \approx 3.14$ |
| $\leq$ | Less than or equal to | $x \leq 10$ means $x$ is at most 10 |
| $\geq$ | Greater than or equal to | $p \geq 0$ means $p$ is not negative |
| $\neq$ | Not equal | $0.1 + 0.2 \neq 0.3$ (in floating-point!) |
| $\to$ | "approaches" or "maps to" | $T \to 0$ means "as $T$ gets closer to zero" |
| $\infty$ | Infinity | $-\infty$ means "negative infinity" |

## Functions

A function takes input and produces output. We write:

$$
f(x) = 2x + 3
$$

This means: "the function $f$ takes a number $x$ and returns $2x + 3$." So $f(5) = 2 \times 5 + 3 = 13$.

**Functions on vectors** work the same way, but the input is a vector:

$$
\text{ReLU}(x) = \max(0, x)
$$

When applied to a vector, it's applied to each element: $\text{ReLU}([3, -1, 0]) = [3, 0, 0]$.

**Named functions** you'll see in this course:

| Notation | Name | What it does |
|----------|------|-------------|
| $\max(a, b)$ | Maximum | The larger of $a$ and $b$ |
| $\min(a, b)$ | Minimum | The smaller of $a$ and $b$ |
| $\sqrt{x}$ | Square root | The number that, squared, gives $x$. $\sqrt{9} = 3$ |
| $\log(x)$ | Logarithm | The inverse of $e^x$. $\log(e^3) = 3$ |
| $e^x$ | Exponential | $e \approx 2.718$; $e^0 = 1$, $e^1 \approx 2.72$, $e^2 \approx 7.39$ |
| $\cos(\theta)$, $\sin(\theta)$ | Cosine, sine | Trigonometric functions (used in [RoPE](../positional-encoding/)) |

## Norms and absolute values

| Notation | Name | What it means |
|----------|------|--------------|
| $\|x\|$ | Absolute value (for numbers) | Distance from zero: $\|-5\| = 5$, $\|3\| = 3$ |
| $\|v\|$ | Norm (for vectors) | Length of the vector: $\|[3, 4]\| = \sqrt{9+16} = 5$ |

The double bars $\| \cdot \|$ always mean "how big is this thing" — for a number, it's the absolute value; for a vector, it's the length.

## Matrix notation

| Notation | What it means |
|----------|--------------|
| $A \in \mathbb{R}^{m \times n}$ | $A$ is a matrix with $m$ rows and $n$ columns |
| $A_{ij}$ | The entry in row $i$, column $j$ of matrix $A$ |
| $A^T$ | $A$ transposed — rows become columns |
| $AB$ | Matrix $A$ multiplied by matrix $B$ |
| $Wx$ | Matrix $W$ multiplied by vector $x$ |
| $\odot$ | Element-wise multiplication (multiply matching positions, not matrix multiply) |

## The product symbol: $\prod$

Less common than $\sum$, but you'll see it occasionally. $\prod$ means "multiply together" the same way $\sum$ means "add together":

$$
\prod_{i=1}^{4} i = 1 \times 2 \times 3 \times 4 = 24
$$

## Our promise

Throughout this course, when we introduce compact notation for the first time, we always show the expanded form:

> **The dot product** adds up the products of matching elements:
>
> $a_1 \times b_1 + a_2 \times b_2 + \cdots + a_n \times b_n$
>
> In compact notation: $\sum_{i=1}^{n} a_i b_i$

If you see a symbol you don't recognize, come back to this page. And if we ever use notation without explaining it first, that's a bug — [let us know](https://github.com/smgglrs-ai/tempura/issues).

**Next:** [Vectors](../vectors/) — the first building block.
