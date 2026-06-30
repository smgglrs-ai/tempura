+++
title = "21. Ternary Weights"
description = "When weights are just -1, 0, or +1 — how BitNet eliminates multiplication and what it means for verification."
weight = 210
template = "docs/page.html"

[extra]
part = "kernel-math"
toc = true
math = true
+++

## Why go below 4 bits

[Quantization](../quantization/) showed how to represent model weights with fewer bits. [Quantized matmul on GPUs](../quantized-matmul-gpu/) showed that even with Q4\_K compression, decode is memory-bandwidth bound — the GPU spends most of its time waiting for weight data to arrive from VRAM.

The logical response: use fewer bits. Q4 gives 16 distinct values. Q3 gives 8. Q2 gives 4. But at some point, fewer bits means fewer distinct values — and the quantization error per value grows.

What if we accept that tradeoff fully and go to the extreme: only three possible values?

## The ternary alphabet: {-1, 0, 1}

A ternary weight can be exactly one of three values: $-1$, $0$, or $+1$. No scales, no zero points, no dequantization formula. The weight *is* the value.

How many bits to represent 3 states? Information theory gives the answer:

$$
\log_2(3) \approx 1.585 \text{ bits}
$$

This is where the "1.58-bit" name comes from — it is the theoretical minimum number of bits needed to encode a ternary value. In practice, packing ternary values into binary storage uses slightly more (a common scheme packs 5 ternary values into 8 bits, which is 1.6 bits per value).

Compare the memory cost for a 7B parameter model:

| Precision | Bits per param | Memory (7B) |
|-----------|---------------|-------------|
| f16       | 16            | 14 GB       |
| int4 (Q4) | 4 + overhead  | ~3.5 GB     |
| ternary   | ~1.6          | ~1.4 GB     |

A 7B ternary model fits in under 1.5 GB — smaller than many images.

## Multiply becomes add/sub/skip

This is the key insight. In standard inference, the inner loop of [matrix multiplication](../matrices/) computes:

$$
\text{acc} = \sum_{j} w_j \cdot x_j
$$

Each step requires a floating-point multiply followed by an add. The multiply is the expensive part — it uses dedicated FPU circuitry and consumes energy.

With ternary weights, $w_j \in \{-1, 0, +1\}$, and the multiply disappears:

- $w_j = +1$: $\text{acc} \mathrel{+}= x_j$ (addition)
- $w_j = -1$: $\text{acc} \mathrel{-}= x_j$ (subtraction)
- $w_j = 0$: skip entirely (no operation)

No floating-point multiplier needed. The entire matmul reduces to additions, subtractions, and conditional skips.

### Worked example

Consider a small $4 \times 4$ weight matrix and input vector:

```
Weights W              Input x
┌────────────────┐     ┌────┐
│ +1  -1   0  +1 │     │ 2.0│
│  0  +1  -1   0 │     │ 1.5│
│ -1   0  +1  +1 │     │ 3.0│
│ +1  +1   0  -1 │     │ 0.5│
└────────────────┘     └────┘
```

Row 0: $(+1)(2.0) + (-1)(1.5) + (0)(3.0) + (+1)(0.5)$
$= 2.0 - 1.5 + 0 + 0.5 = 1.0$

Row 1: $(0)(2.0) + (+1)(1.5) + (-1)(3.0) + (0)(0.5)$
$= 0 + 1.5 - 3.0 + 0 = -1.5$

Row 2: $(-1)(2.0) + (0)(1.5) + (+1)(3.0) + (+1)(0.5)$
$= -2.0 + 0 + 3.0 + 0.5 = 1.5$

Row 3: $(+1)(2.0) + (+1)(1.5) + (0)(3.0) + (-1)(0.5)$
$= 2.0 + 1.5 + 0 - 0.5 = 3.0$

Every "multiply" was actually an add, subtract, or skip. The output is $[1.0, -1.5, 1.5, 3.0]$.

## The BitLinear layer

BitNet b1.58 (Ma et al., 2024) replaces every `nn.Linear` in the transformer with a BitLinear layer. The forward pass has four steps:

### Step 1: RMSNorm the input

The input activation is normalized using [RMSNorm](../normalization/) — the same normalization used in standard LLaMA. This centers the values and controls their magnitude before quantization.

### Step 2: Quantize activations to INT8

Activations are quantized per-token using absmax scaling:

$$
Q(x) = \text{clamp}\left(\text{round}\left(\frac{127 \cdot x}{\max(|x|)}\right),\ -128,\ 127\right)
$$

This maps the full activation range into 8-bit integers. The per-token scale $\gamma = \max(|x|) / 127$ is saved for rescaling.

### Step 3: Integer matmul (add/sub only)

The quantized INT8 activations are multiplied by the ternary weights. Since $w \in \{-1, 0, +1\}$, the "multiplication" is just addition, subtraction, or skip applied to integer values. The accumulator is INT32 — no floating-point arithmetic anywhere in this step.

### Step 4: Rescale

The INT32 result is converted back to floating-point and rescaled:

$$
\text{output} = \text{acc} \times \frac{\gamma \cdot \beta}{127}
$$

where $\gamma$ is the per-token activation scale and $\beta$ is a learned per-channel weight scale.

### Weight quantization

The weights themselves are quantized during training using absmean quantization:

$$
W_q = \text{RoundClip}\left(\frac{W}{\text{mean}(|W|)},\ -1,\ 1\right)
$$

where $\text{RoundClip}(x, a, b) = \max(a, \min(b, \text{round}(x)))$. Dividing by the mean absolute value (not the max) centers the distribution so that roughly equal numbers of weights map to each of the three values.

## Squared ReLU, not SwiGLU

Standard LLaMA uses SwiGLU as the FFN activation:

$$
\text{SwiGLU}(x) = (\text{Swish}(xW_1)) \odot (xW_3)
$$

The Swish function contains a sigmoid gate: $\text{Swish}(x) = x \cdot \sigma(x)$. This gate produces values across a continuous range, which fights with the discrete nature of ternary weights — the smooth gating signal gets destroyed by the coarse weight representation.

BitNet b1.58 replaces SwiGLU with Squared ReLU:

$$
\text{SquaredReLU}(x) = (\text{ReLU}(x))^2
$$

This works better with ternary weights for two reasons:

1. **Hard zero boundary.** ReLU produces exact zeros for all negative inputs, which aligns with the structural sparsity of ternary weights. SwiGLU's sigmoid never produces exact zeros.

2. **Sparsity amplification.** Squaring pushes small positive values toward zero while amplifying large ones, creating a sharper distinction between active and inactive neurons.

## ~42% zeros

When you divide each weight by $\text{mean}(|W|)$ and round to the nearest integer in $\{-1, 0, 1\}$, a large fraction of values land on zero. In practice, absmean quantization produces roughly 42--51% zeros across layers, depending on the weight distribution.

These zeros are structural sparsity that comes for free:

- No pruning algorithm needed
- No fine-tuning to recover accuracy
- Every zero is a skipped operation in the inner loop
- The sparsity pattern is fixed at quantization time — no runtime branching to discover it

Combined with add/sub instead of multiply, the effective compute per output element is roughly half as many integer additions as a full-precision model has floating-point multiply-adds.

## Why verification loves ternary

[Quantization](../quantization/) discussed proving properties of Q4\_K dequantization — bounded error, monotonicity, accumulator safety. Those proofs are complicated by the floating-point arithmetic in the dequantization formula: scales, minimums, and the conversion back to f32 all involve FP rounding.

Ternary weights eliminate that complexity. The entire weight path is integer:

- Weights are $\{-1, 0, +1\}$ — stored as packed integers
- Activations are INT8 after quantization
- The accumulator is INT32
- The only floating-point operation is the final rescale

This gives Kani much more to work with:

**Bit-packing roundtrip.** Five ternary values pack into one byte (3^5 = 243 < 256). Kani can exhaustively verify that $\text{unpack}(\text{pack}(v)) = v$ for all $3^5 = 243$ valid inputs — a complete proof, not a bounded check.

**Accumulator overflow.** The worst case for an INT32 accumulator summing INT8 values is:

$$
|\text{acc}|_{\max} = d \times 127
$$

where $d$ is the hidden dimension. For $d = 4096$: $|\text{acc}|_{\max} = 520{,}192$, far below the INT32 limit of $2^{31} - 1 \approx 2.1 \times 10^9$. Kani can prove no overflow for any hidden dimension up to $2^{31}/127 \approx 16.9$ million.

**Index safety.** The decode loop iterates over packed bytes and extracts ternary values. Kani can prove the unpacking logic never accesses out of bounds and never produces values outside $\{-1, 0, 1\}$.

Compare this to Q4\_K, where dequantization involves float16-to-float32 conversion, 6-bit scale unpacking, and a multiply-subtract formula. The ternary verification story is strictly stronger — more properties proven, with higher confidence, because the arithmetic is simpler.

In tempura, the BitLinear kernel implements this integer-only forward path, and the same Kani harnesses that verify the CPU reference apply to the GPU kernel via trait-based dispatch. The verification contract does not weaken when moving to hardware.

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|----------------|
| Ternary weights | Weights restricted to $\{-1, 0, +1\}$ | Eliminates floating-point multiplication entirely |
| 1.58-bit | $\log_2(3) \approx 1.585$ bits per value | Theoretical minimum encoding; 7B model fits in ~1.4 GB |
| BitLinear | Replacement for nn.Linear | RMSNorm, INT8 activations, add/sub matmul, rescale |
| Absmean quantization | $W_q = \text{RoundClip}(W/\text{mean}(\|W\|))$ | Centers distribution so ~42% of weights become zero |
| Squared ReLU | $(\text{ReLU}(x))^2$ replaces SwiGLU | Hard zeros align with ternary sparsity; sigmoid does not |
| Integer-only path | INT8 activations, ternary weights, INT32 accumulator | Kani can prove overflow safety, packing roundtrip, index bounds |
| Structural sparsity | ~42--51% of weights are zero | Free speedup: zeros are skipped, no pruning needed |

## What to read next

[Ternary Packing](../ternary-packing/) covers how these three-valued weights are stored in memory — the bit-packing schemes, alignment constraints, and decode loops that the kernel must implement efficiently.
