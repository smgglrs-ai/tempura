+++
title = "1. Vectors"
description = "How to represent things as lists of numbers — the foundation of everything that follows."
weight = 10
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Numbers describe things

Suppose you want to describe a city's weather: temperature is 22°C, humidity is 65%, wind speed is 12 km/h. You've just created a vector:

$$
\text{weather} = [22, 65, 12]
$$

A **vector** is an ordered list of numbers. Each number is a **component** (or dimension). This vector has 3 dimensions — it lives in 3-dimensional space.

Vectors can describe anything measurable:

- A pixel's color: $[red, green, blue] = [255, 128, 0]$ (orange)
- A student's grades: $[math, science, english] = [85, 92, 78]$
- A word's meaning: $[\ldots 300 \text{ numbers} \ldots]$ (we'll get to this in [Words as Numbers](../words-as-numbers/))

## Basic operations

**Addition** — component by component:

$$
[3, 1, 4] + [1, 5, 9] = [4, 6, 13]
$$

**Scalar multiplication** — multiply every component by a number:

$$
2 \times [3, 1, 4] = [6, 2, 8]
$$

**These are intuitive.** If today's weather is $[22, 65, 12]$ and tomorrow's change is $[+3, -10, +5]$, tomorrow's weather is $[22, 65, 12] + [3, -10, 5] = [25, 55, 17]$.

## The dot product

The **dot product** of two vectors multiplies corresponding components and sums the results:

$$
[3, 4] \cdot [1, 2] = 3 \times 1 + 4 \times 2 = 3 + 8 = 11
$$

More generally, for vectors $a$ and $b$ with $n$ components:

$$
a \cdot b = \sum_{i=1}^{n} a_i \times b_i
$$

**Try it:** compute $[1, 0, 2] \cdot [3, 5, 1]$.

Answer: $1 \times 3 + 0 \times 5 + 2 \times 1 = 3 + 0 + 2 = 5$

## Why the dot product matters: similarity

The dot product measures how much two vectors "agree." When both vectors point in the same direction, the dot product is large and positive. When they're perpendicular, it's zero. When they point in opposite directions, it's large and negative.

Consider two students' grade profiles:

$$
\text{Alice} = [90, 95, 70] \qquad \text{Bob} = [88, 92, 75] \qquad \text{Carol} = [70, 72, 95]
$$

Alice and Bob are similar — both strong in math and science, weaker in English. Carol is the opposite. The dot products reflect this:

$$
\text{Alice} \cdot \text{Bob} = 90 \times 88 + 95 \times 92 + 70 \times 75 = 7920 + 8740 + 5250 = 21910
$$

$$
\text{Alice} \cdot \text{Carol} = 90 \times 70 + 95 \times 72 + 70 \times 95 = 6300 + 6840 + 6650 = 19790
$$

Alice · Bob > Alice · Carol, reflecting that Alice and Bob are more similar.

**This is exactly how attention works in language models:** the model computes dot products between vectors to measure which words are "similar" or "relevant" to each other. We'll see this in [The Transformer](../the-transformer/).

## Vector length (norm)

The **length** (or norm) of a vector is:

$$
\|a\| = \sqrt{a_1^2 + a_2^2 + \cdots + a_n^2}
$$

For $a = [3, 4]$: $\|a\| = \sqrt{9 + 16} = \sqrt{25} = 5$.

This is the Pythagorean theorem extended to any number of dimensions.

**Normalization** divides a vector by its length, producing a unit vector (length 1) that preserves direction:

$$
\hat{a} = \frac{a}{\|a\|} = \frac{[3, 4]}{5} = [0.6, 0.8]
$$

Check: $\|\hat{a}\| = \sqrt{0.36 + 0.64} = \sqrt{1} = 1$ ✓

We'll see normalization again in [RMS Normalization](../../foundations/normalization/), where it keeps neural network activations from growing out of control.

## Why this matters for AI

In a language model:

- Every word is represented as a vector (typically with 1,000–8,000 dimensions)
- The model measures relationships between words using dot products
- It transforms word vectors using the operations you'll learn in the next pages

The entire process — from reading a sentence to generating the next word — is a sequence of vector operations. No magic, no understanding — just arithmetic on lists of numbers. The surprising thing is how well it works.

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| Vector | Ordered list of numbers | Represents anything as numbers |
| Dot product | Multiply-and-sum | Measures similarity between vectors |
| Norm | Length of a vector | Measures magnitude |
| Normalization | Divide by length | Controls scale |

**Next:** [Matrices](../matrices/) — how to transform vectors.
