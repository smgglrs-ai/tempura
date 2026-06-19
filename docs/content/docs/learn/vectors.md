+++
title = "1. Vectors"
description = "How to represent things as lists of numbers — the foundation of everything that follows."
weight = 10
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Why numbers?

Computers can't read words. They can't see images. They can't hear sounds. All they can do is add, multiply, and compare numbers — very, very fast. If we want a computer to work with language, images, or sound, we first need to convert everything into numbers.

This chapter introduces the tool for doing that: **vectors**.

## What is a vector?

A vector is simply an **ordered list of numbers**. That's it. No special symbol, no complicated definition — just numbers in a row, enclosed in brackets:

$$
[3, 7, 2]
$$

This vector has **3 components** (or dimensions). The first component is 3, the second is 7, the third is 2. Order matters: $[3, 7, 2]$ is different from $[7, 3, 2]$.

## Representing real things

The power of vectors is that they can describe anything measurable. You just decide what each position means and fill in the numbers:

**Weather in a city:**

| Position | Meaning | Value |
|----------|---------|-------|
| 1 | Temperature (°C) | 22 |
| 2 | Humidity (%) | 65 |
| 3 | Wind speed (km/h) | 12 |

$$
\text{weather} = [22, 65, 12]
$$

**A pixel's color:**

| Position | Meaning | Value (0–255) |
|----------|---------|-------|
| 1 | Red | 255 |
| 2 | Green | 128 |
| 3 | Blue | 0 |

$$
\text{pixel} = [255, 128, 0] \quad \text{(orange)}
$$

**A student's grades:**

$$
\text{grades} = [85, 92, 78, 91] \quad \text{(math, science, English, history)}
$$

**A word's meaning** (we'll get to this in [Words as Numbers](../words-as-numbers/)):

$$
\text{"king"} = [0.21, -0.43, 0.87, \ldots, -0.12] \quad \text{(hundreds or thousands of numbers)}
$$

The crucial idea: once something is a vector, we can do math on it. And math is what computers are good at.

## Operations on vectors

### Addition

Add vectors by adding their corresponding components:

$$
[3, 1, 4] + [1, 5, 9] = [3+1, \; 1+5, \; 4+9] = [4, 6, 13]
$$

Both vectors must have the same number of components. You can't add a 3-component vector to a 4-component vector — it wouldn't make sense.

**Real-world example:** Today's weather is $[22, 65, 12]$. Tomorrow's change is forecast as $[+3, -10, +5]$. Tomorrow's weather:

$$
[22, 65, 12] + [3, -10, 5] = [25, 55, 17]
$$

25°C, 55% humidity, 17 km/h wind. Addition combines information.

### Scalar multiplication

Multiply every component by the same number (a **scalar**):

$$
2 \times [3, 1, 4] = [6, 2, 8]
$$

$$
0.5 \times [10, 20] = [5, 10]
$$

This scales the vector — makes it longer or shorter while keeping the same proportions.

## The dot product: the most important operation

The **dot product** of two vectors multiplies corresponding components and adds up the results:

$$
[3, 4] \cdot [1, 2] = (3 \times 1) + (4 \times 2) = 3 + 8 = 11
$$

Let's be very explicit about what happens:

| Component | Vector $a$ | Vector $b$ | Product |
|-----------|-----------|-----------|---------|
| 1st | 3 | 1 | $3 \times 1 = 3$ |
| 2nd | 4 | 2 | $4 \times 2 = 8$ |
| | | **Sum:** | **11** |

The general formula for vectors with $n$ components:

$$
a \cdot b = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n = \sum_{i=1}^{n} a_i b_i
$$

**Try these yourself:**

1. $[1, 0, 2] \cdot [3, 5, 1] = ?$
2. $[2, 2] \cdot [3, 3] = ?$
3. $[1, -1] \cdot [1, 1] = ?$

Answers:
1. $1 \times 3 + 0 \times 5 + 2 \times 1 = 3 + 0 + 2 = 5$
2. $2 \times 3 + 2 \times 3 = 6 + 6 = 12$
3. $1 \times 1 + (-1) \times 1 = 1 - 1 = 0$

Notice that #3 gives exactly zero. We'll see why that matters in a moment.

## The dot product measures similarity

This is the key insight that makes language models possible. The dot product tells you how much two vectors "agree" — how similar they are in their overall pattern.

**When vectors point the same way** — both have large values in the same positions — the products are all positive, and the sum is large.

**When vectors are unrelated** — large values in one correspond to small or zero values in the other — products cancel out, and the sum is near zero.

**When vectors are opposite** — one has positive values where the other has negative — products are negative, and the sum is negative.

### A concrete example

Three students' grade profiles (math, science, English):

$$
\text{Alice} = [90, 95, 70] \qquad \text{Bob} = [88, 92, 75] \qquad \text{Carol} = [70, 72, 95]
$$

Alice and Bob are similar — both strong in math and science, weaker in English. Carol is the opposite — strong in English, weaker in STEM.

$$
\text{Alice} \cdot \text{Bob} = 90 \times 88 + 95 \times 92 + 70 \times 75
$$
$$
= 7920 + 8740 + 5250 = 21{,}910
$$

$$
\text{Alice} \cdot \text{Carol} = 90 \times 70 + 95 \times 72 + 70 \times 95
$$
$$
= 6300 + 6840 + 6650 = 19{,}790
$$

Alice · Bob (21,910) > Alice · Carol (19,790). The dot product is higher for more similar students.

### Why this matters for AI

In a language model, every word is represented as a vector. The model computes dot products between word vectors to determine which words are related to each other. When the model processes the sentence "The cat sat on the mat," it uses dot products to figure out that "sat" is related to "cat" (the cat is the one sitting) rather than to "mat."

This is the heart of the [attention mechanism](../the-transformer/) — and it's just the dot product you learned above, applied to word vectors with thousands of components instead of three.

## Vector length (norm)

How "big" is a vector? Its **length** (also called **norm** or **magnitude**) is computed using the Pythagorean theorem, extended to any number of dimensions:

$$
\|a\| = \sqrt{a_1^2 + a_2^2 + \cdots + a_n^2}
$$

**For 2D**, this is the familiar distance formula. A vector $[3, 4]$ has length:

$$
\|[3, 4]\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$

If you draw $[3, 4]$ as an arrow from the origin, it's exactly 5 units long — a 3-4-5 right triangle.

**For 3D:** $\|[1, 2, 2]\| = \sqrt{1 + 4 + 4} = \sqrt{9} = 3$

**Try it:** what's the length of $[1, 0, 0]$?

Answer: $\sqrt{1 + 0 + 0} = 1$. This is a **unit vector** — its length is exactly 1.

## Normalization

**Normalization** rescales a vector to length 1, preserving its direction:

$$
\hat{a} = \frac{a}{\|a\|}
$$

For $[3, 4]$ with $\|a\| = 5$:

$$
\hat{a} = \frac{[3, 4]}{5} = [0.6, 0.8]
$$

Verify: $\|[0.6, 0.8]\| = \sqrt{0.36 + 0.64} = \sqrt{1} = 1$ ✓

Why normalize? Because sometimes you care about *direction* (what kind of pattern) but not *magnitude* (how strong the pattern is). Normalization lets you compare patterns regardless of scale.

We'll see normalization again in [Neural Networks](../neural-networks/), where [RMS Normalization](normalization/) prevents numbers from growing out of control as they pass through many layers.

## Vectors in higher dimensions

The examples above use 2 or 3 dimensions, which you can visualize. But vectors can have any number of dimensions:

- A pixel: 3 dimensions (red, green, blue)
- A short audio clip: thousands of dimensions (one per time sample)
- A word in a language model: 4,096 or more dimensions

You can't visualize a 4,096-dimensional vector, but all the math works exactly the same — addition, dot product, and normalization are defined the same way regardless of dimension count. This is the power of the abstraction: the formulas scale without modification.

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| Vector | Ordered list of numbers | Represents anything as computable data |
| Addition | Component-wise sum | Combines information |
| Scalar multiplication | Multiply all components by one number | Scales magnitude |
| Dot product | Multiply-and-sum: $\sum a_i b_i$ | Measures similarity |
| Norm | $\sqrt{\sum a_i^2}$ — the vector's length | Measures magnitude |
| Normalization | Divide by the norm → unit length | Compares direction, ignoring scale |

**Next:** [Matrices](../matrices/) — how to systematically transform one vector into another.
