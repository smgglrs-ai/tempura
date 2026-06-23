+++
title = "Quantization"
description = "What rounding preserves and destroys — group quantization, error bounds, and provable properties."
weight = 50
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## What quantization does

Quantization maps f32 weights to lower-precision integers (4-bit, 8-bit) with scale factors. This reduces model size (7B params × 4 bytes = 28GB → 4-bit: ~4GB) and increases throughput (less memory bandwidth per weight).

## Symmetric int8

The simplest scheme. Given a weight vector $w$:

$$
\text{scale} = \frac{\max |w_i|}{127}, \quad q_i = \text{round}\left(\frac{w_i}{\text{scale}}\right)
$$

Dequantization: $\hat{w}_i = q_i \times \text{scale}$.

The maximum round-trip error per element is $\frac{\text{scale}}{2}$ — half the quantization step. This is proven by Kani (see `int8_dequant_bounded_error`).

## Group quantization (Q4_K)

GGUF's Q4_K format uses 4-bit quantization with per-group scale and minimum:

$$
\hat{w}_i = d \cdot \text{sc}_g \cdot q_i - d_\text{min} \cdot m_g
$$

where $d$ is the block-level scale (f16), $\text{sc}_g$ and $m_g$ are per-group (6-bit each, 8 groups per block of 256 elements), and $q_i$ is a 4-bit nibble (0-15).

Each Q4_K block is 144 bytes for 256 elements = 4.5 bits/element.

## Monotonicity

Within a group (fixed $d$, $\text{sc}_g$, $d_\text{min}$, $m_g$), the dequantized value is a linear function of the nibble:

$$
\hat{w}(q) = d \cdot \text{sc}_g \cdot q - d_\text{min} \cdot m_g
$$

Since $d \cdot \text{sc}_g \geq 0$ (both non-negative), $q_1 > q_2 \implies \hat{w}(q_1) \geq \hat{w}(q_2)$. This is proven by Kani (see `q4k_monotonicity_within_group`).

Monotonicity matters for softmax: if the float weight ordering is preserved through quantization, the attention pattern is preserved.

## Accumulator safety

In quantized matmul, partial sums accumulate in i32 or f32. For Q4_K with group_size=32:

$$
\text{max partial sum} = 127 \times 15 \times 128 = 243{,}840 \ll 2^{31} - 1
$$

This is proven by Kani (see `i32_accumulator_no_overflow_quantized_matmul`).

## Group boundary safety

The dequant loop over (j, l) must not cross group boundaries — reading a scale from group $g$ while indexing elements from group $g'$ produces wrong values. Kani proves element indices stay in $[64j, 64(j+1))$ and qs byte indices stay in $[32j, 32(j+1))$ for each group $j$ (see `q4k_group_boundary_safety`).

## Worked example

Q4_K block with $d = 0.01$, $d_\text{min} = 0.005$, group 0 scale $\text{sc}_0 = 10$, min $m_0 = 5$:

For nibble $q = 7$:
$$
\hat{w} = 0.01 \times 10 \times 7 - 0.005 \times 5 = 0.7 - 0.025 = 0.675
$$

For nibble $q = 15$ (max):
$$
\hat{w} = 0.01 \times 10 \times 15 - 0.005 \times 5 = 1.5 - 0.025 = 1.475
$$

The dequantized range for this group: $[-0.025, 1.475]$.

## Source code

- **Trait**: [`kernel-api/src/lib.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernel-api/src/lib.rs) — `MatMulQ` trait for quantized matmul, `Dequantize` trait
- **CPU implementation**: [`kernels/src/quantized.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/quantized.rs) — `CpuMatMulQ4K` (reference, Kani-proven), `CpuMatMulQ4KOpt` (SIMD-optimized)
- **Kani proofs**: Same file, `mod kani_proofs` — 16 proofs covering block access, dequant bounds, monotonicity, accumulator safety, group boundaries, ggml compatibility
- **GGUF loader**: [`engine/src/gguf_loader.rs`](https://github.com/smgglrs-ai/tempura/blob/main/engine/src/gguf_loader.rs) — parses Q4_K, Q6_K, Q8_0 blocks from GGUF files

## See also

- [Normalization](../normalization/) — activation magnitudes affect quantization range
- [Sampling](../sampling/) — quantization-induced logit shifts can change sampling outcomes
- [Attention](../attention/) — KV cache quantization (TurboQuant) compresses cached values
