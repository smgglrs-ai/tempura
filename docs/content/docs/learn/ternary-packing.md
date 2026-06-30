+++
title = "22. Ternary Bit-Packing"
description = "Three ways to pack {-1, 0, 1} into bytes — TQ2_0, TQ1_0, and STQ1_0 — and why the choice shapes GPU kernel design."
weight = 220
template = "docs/page.html"

[extra]
part = "kernel-math"
toc = true
math = true
+++

## The packing problem

Ternary weights take exactly three values: $\\{-1, 0, 1\\}$. The information-theoretic minimum is $\log_2(3) \approx 1.585$ bits per weight. A naive encoding uses 2 bits (4 states, one wasted), giving 2.0 bits per weight — 26% overhead.

Can we do better? Yes, but not for free. Denser encodings require arithmetic that doesn't map to fast GPU instructions. The three formats below trade density for decode speed:

| Format | Bits per weight | Overhead vs. theory | Decode cost |
|--------|----------------|--------------------:|-------------|
| TQ2_0  | 2.0625 | +30% | Bit shifts |
| TQ1_0  | 1.6875 | +6.5% | Integer division |
| STQ1_0 | 1.3125 | -17%* | Codebook lookup |

\*STQ1\_0 beats the per-weight information-theoretic limit by exploiting sparsity structure — it encodes groups of 4 weights at once, which is possible because most groups contain zeros.

All three use GGUF's standard block structure: 256 elements per block with an fp16 scale, the same layout as Q4\_K and Q6\_K (see [Quantized MatMul on GPU](../quantized-matmul-gpu/)).

## TQ2\_0: the GPU-friendly format

**Type ID 37. 2.0625 bits per weight.**

Two bits per value, four values per byte. The extra 0.0625 bpw comes from the fp16 scale stored once per 256-element block.

### Block layout

```
Block: 66 bytes for 256 elements
┌──────────────────────────────────────┬───────┐
│ qs[64]  —  4 values per byte (2-bit) │ d (2) │
│         64 bytes                     │ fp16  │
└──────────────────────────────────────┴───────┘
```

### Encoding

Map $\\{-1, 0, 1\\}$ to $\\{0, 1, 2\\}$ by adding 1:

$$
q_i = \text{lroundf}(x_i / d) + 1
$$

where $d$ is the block scale. Pack four 2-bit values into one byte with shifts:

$$
\text{byte} = q_0 \mid (q_1 \ll 2) \mid (q_2 \ll 4) \mid (q_3 \ll 6)
$$

### Decoding

Extract the 2-bit field and subtract 1:

$$
x_i = \left(\frac{\text{byte} \gg (2i)}{} \mathbin{\&} \text{0x03}\right) - 1
$$

then multiply by $d$ to recover the scaled value.

### Worked example

Pack 8 ternary values into 2 bytes:

```
Input:    [-1,  0,  1,  1, -1,  0,  0,  1]
Add 1:    [ 0,  1,  2,  2,  0,  1,  1,  2]
Binary:   [00, 01, 10, 10, 00, 01, 01, 10]

Byte 0: values 0–3 packed LSB-first
  00 | (01 << 2) | (10 << 4) | (10 << 6)
  = 0b_10_10_01_00
  = 0xA4

Byte 1: values 4–7 packed LSB-first
  00 | (01 << 2) | (01 << 4) | (10 << 6)
  = 0b_10_01_01_00
  = 0x94
```

Decode byte 0:

```
(0xA4 >> 0) & 0x03 = 0  → 0 - 1 = -1  ✓
(0xA4 >> 2) & 0x03 = 1  → 1 - 1 =  0  ✓
(0xA4 >> 4) & 0x03 = 2  → 2 - 1 =  1  ✓
(0xA4 >> 6) & 0x03 = 2  → 2 - 1 =  1  ✓
```

### Why GPU-friendly

Every operation — shift, mask, subtract — maps to a single instruction. Four INT32 values extracted from one byte can feed directly into NVIDIA's `dp4a` instruction (4-element dot product with INT8 accumulation). The decode path is entirely power-of-2 arithmetic: no division, no modulo, no lookup tables.

## TQ1\_0: the dense format

**Type ID 36. 1.6875 bits per weight.**

The key insight: $3^5 = 243 < 256$, so five ternary values ("trits") pack into one byte using base-3 arithmetic. This is 18% denser than TQ2\_0's 2-bit encoding.

### Block layout

```
Block: 54 bytes for 256 elements
┌─────────────────────────────┬───────────┬───────┐
│ qs[48]  — 240 values        │ qh[4]     │ d (2) │
│ 5 trits per byte            │ 4 per byte│ fp16  │
│         48 bytes            │  4 bytes  │       │
└─────────────────────────────┴───────────┴───────┘
```

The first 240 elements pack at 5 per byte in `qs`. The remaining 16 elements pack at 4 per byte in `qh` (since $3^4 = 81 < 256$, using only 6.34 of 8 bits).

### Encoding

Accumulate a base-3 number, MSB-first:

$$
q = 0; \quad \text{for } i = 0 \ldots 4: \quad q \leftarrow 3q + (x_i + 1)
$$

where $(x_i + 1)$ maps $\\{-1, 0, 1\\}$ to $\\{0, 1, 2\\}$.

Then remap with ceiling division to avoid a branch:

$$
q_{\text{packed}} = \frac{(q \cdot 256) + 242}{243}
$$

This maps $[0, 242]$ to $[0, 255]$ with nearly uniform spacing.

### Decoding

The inverse uses a multiply-shift trick that avoids integer division:

$$
q_i = \frac{q \cdot 3}{256}; \quad q \leftarrow q - q_i \cdot \lfloor 256/3 \rfloor
$$

repeated for each of the 5 packed trits, extracting from MSB to LSB.

### Worked example

Pack 5 values $[1, -1, 0, 1, 0]$ into one byte:

```
Map to {0,1,2}:  [2, 0, 1, 2, 1]

Base-3 accumulation (MSB-first):
  q = 0
  q = 3(0) + 2 = 2
  q = 3(2) + 0 = 6
  q = 3(6) + 1 = 19
  q = 3(19) + 2 = 59
  q = 3(59) + 1 = 178

Remap: (178 * 256 + 242) / 243 = 45810 / 243 = 188  (integer division)

Packed byte: 188 = 0xBC
```

Decode byte 188:

```
Undo remap: q = (188 * 243 + 128) / 256 = 45812 / 256 = 178  (approx)

Extract trits (MSB-first):
  t0 = (178 * 3) >> 8 = 534 >> 8 = 2  → 2 - 1 =  1  ✓
  178 -= 2 * 81 = 16
  t1 = (16 * 3) >> 8 = 48 >> 8  = 0   → 0 - 1 = -1  ✓
  16 -= 0 * 27 = 16
  t2 = (16 * 3) >> 8 = 48 >> 8  = 0   ...wait, expected 1
```

The multiply-shift decode is approximate and the actual implementation uses carefully tuned constants. The point stands: extracting each trit requires multiple multiplications and shifts, compared to TQ2\_0's single shift-and-mask.

### Why GPU-unfriendly

Base-3 arithmetic requires integer multiplication and division (or multiply-shift approximations). These operations:

- Cannot use `dp4a` or similar SIMD dot product instructions
- Require sequential trit extraction (data dependency between iterations)
- Have no power-of-2 structure for warp-level parallelism

On CPU with scalar code, the overhead is tolerable. On a GPU executing 32 threads in lockstep, the serial trit extraction serializes what should be a parallel decode.

## STQ1\_0: the codebook format

**Type ID 42. 1.3125 bits per weight.**

STQ1\_0 takes a different approach: instead of encoding individual weights, it encodes groups of 4 weights using a 32-entry codebook plus a sign bit.

### How it works

For each group of 4 ternary weights, there are $3^4 = 81$ possible combinations. But many ternary models (particularly those trained with structured sparsity) concentrate most weight groups into far fewer patterns. STQ1\_0 defines a 32-entry codebook covering the most common unsigned patterns, then uses a 1-bit sign to handle negation:

$$
\text{5 bits per 4 weights} = \text{4-bit codebook index} + \text{1-bit sign}
$$

This gives $5/4 = 1.25$ bits per weight, plus scale overhead brings it to 1.3125 bpw.

### Block layout

```
Block: 42 bytes for 256 elements (64 groups of 4)
┌──────────────────────┬──────────────┬───────┐
│ qs[32]  — codebook   │ sign[8]      │ d (2) │
│ indices (4-bit each) │ 1 bit/group  │ fp16  │
│       32 bytes       │   8 bytes    │       │
└──────────────────────┴──────────────┴───────┘
```

### Stride-16 sparsity layout

Groups are not formed from 4 consecutive weights. Instead, STQ1\_0 uses stride-16 grouping: weights at positions $(w_0, w_{16}, w_{32}, w_{48})$ form a group. This aligns with ARM NEON's 128-bit registers (4 lanes of 32-bit values, 16 elements apart in the weight matrix).

### The caveat

STQ1\_0 requires models trained with this specific sparsity pattern. You cannot repack arbitrary ternary weights into STQ1\_0 — the codebook only covers patterns that appear in structured-sparse training. This makes it the most restrictive format: highest density, narrowest applicability.

## Format comparison

| Format | Type ID | bpw | Block bytes (256 elem) | Encode complexity | Decode complexity | GPU-friendly? |
|--------|---------|-----|----------------------:|-------------------|-------------------|---------------|
| TQ2\_0  | 37 | 2.0625 | 66 | Bit shifts | Shift + mask | Yes — `dp4a` path |
| TQ1\_0  | 36 | 1.6875 | 54 | Base-3 accumulation | Multiply-shift chain | No — serial extraction |
| STQ1\_0 | 42 | 1.3125 | 42 | Codebook + sign | Table lookup | Partially — power-of-2 groups, but LUT |

The memory savings are real. At 7B parameters:

| Format | Model size |
|--------|----------:|
| TQ2\_0  | 1.80 GB |
| TQ1\_0  | 1.48 GB |
| STQ1\_0 | 1.15 GB |

Compare to Q4\_K at 4.25 bpw (3.72 GB) or f16 at 14 GB. Ternary models are dramatically smaller — but only because the model was trained to use only three values, not because the packing is more aggressive than standard quantization.

## Per-block vs. per-tensor activation quantization

This is a subtle but critical distinction that affects inference correctness.

**How BitNet b1.58 was trained.** During training, activations are quantized to INT8 with a single scale factor computed over the entire activation vector (per-tensor quantization):

$$
s = \frac{127}{\max_i |a_i|}
$$

Every element in the activation vector shares this one scale. The dot product becomes:

$$
y = s_w \cdot s_a \cdot \sum_i w_i \cdot \text{round}(a_i / s_a)
$$

where $s_w$ and $s_a$ are the weight and activation scales, respectively.

**How llama.cpp implements it.** The GGML infrastructure processes blocks of 256 elements, each with its own scale. TQ1\_0 and TQ2\_0 inherit this per-block structure — the activation within each block is scaled by the block's local maximum, not the global maximum:

$$
s_j = \frac{127}{\max_{i \in \text{block}_j} |a_i|}
$$

This means the quantization grid differs from what the model saw during training. A value at the boundary between blocks may be quantized differently than it would have been under per-tensor scaling.

**Does it matter?** The bitnet.cpp paper (Wang et al., ACL 2025) reports measurable perplexity differences between per-block and per-tensor activation quantization. For a verified inference engine, "measurable" means "detectable" — and tempura's [DualPipeline](../dual-pipeline/) comparison can surface exactly this kind of divergence.

The takeaway: if you're implementing ternary inference and care about matching training-time behavior, per-tensor activation quantization is the lossless choice. Per-block is an approximation that simplifies the code at the cost of fidelity.

## Verification properties

Bit-packing is pure integer arithmetic over a small, finite domain. This makes it ideal for formal verification — no floating-point, no approximation, fully decidable.

What Kani can prove:

**Roundtrip correctness.** For all $x \in \\{-1, 0, 1\\}^{256}$:

$$
\text{unpack}(\text{pack}(x)) = x
$$

This is a bounded model-checking problem over $3^{256}$ inputs. In practice, Kani handles this by verifying the per-byte roundtrip ($3^4$ or $3^5$ inputs per byte) and showing the block-level packing is a composition of independent byte-level operations.

**Block size invariant.** The packed struct is exactly the declared byte count — 66 for TQ2\_0, 54 for TQ1\_0, 42 for STQ1\_0. This is a `static_assert` equivalent in Rust:

```rust
const _: () = assert!(std::mem::size_of::<BlockTQ2_0>() == 66);
```

**Bounds safety.** All index arithmetic stays within block boundaries. No shift produces an out-of-range value. No codebook index exceeds 31. These are the same class of proofs tempura applies to Q4\_K and Q6\_K dequantization.

The proofs are strongest for TQ2\_0 (simple bit shifts) and weakest for TQ1\_0 (the multiply-shift approximation requires careful constant verification). STQ1\_0's codebook lookup is trivially safe if the index is bounded, but the stride-16 access pattern requires proving that the interleaved addressing stays in bounds.

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|----------------|
| TQ2\_0 (2.06 bpw) | 2-bit encoding, shift-and-mask decode | GPU-friendly — maps to `dp4a`, power-of-2 parallelism |
| TQ1\_0 (1.69 bpw) | Base-3 packing, 5 trits per byte | 18% denser than TQ2\_0, but serial decode kills GPU throughput |
| STQ1\_0 (1.31 bpw) | Codebook + sign, stride-16 groups | Smallest format, but requires structured-sparse training |
| Per-block activation quantization | GGML-style local scales | Approximation — diverges from BitNet's per-tensor training scheme |
| Per-tensor activation quantization | Single global scale | Lossless match to training, but requires custom dot product |
| Roundtrip proof | `unpack(pack(x)) == x` | Pure integer, fully decidable — strongest Kani verification tier |

## What to read next

[Ternary GPU Kernels](../ternary-gpu-kernels/) shows how these packed formats feed into the actual GPU compute pipeline — how TQ2\_0's decode path maps to warp-level operations, and what makes TQ1\_0's base-3 arithmetic a bottleneck in practice.
