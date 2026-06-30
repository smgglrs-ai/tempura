+++
title = "23. Ternary GPU Kernels"
description = "How ternary matmul runs on a GPU — dp4a integer paths, LUT-based approaches, and why ternary is even more memory-bound than Q4_K."
weight = 230
template = "docs/page.html"

[extra]
part = "kernel-math"
toc = true
math = true
+++

## The opportunity

[Ternary bit-packing](../ternary-packing/) showed how to compress $\{-1, 0, +1\}$
weights into 1.69--2.06 bits per weight. That is 2.2--2.7x smaller than Q4\_K's
4.5 bits per weight. Since [decode is memory-bandwidth bound](../quantized-matmul-gpu/),
reading less data per token means generating more tokens per second — if we can
unpack and compute fast enough.

The question is no longer "how do we multiply?" (ternary multiply is trivial:
negate, zero, or copy). The question is "how do we unpack billions of 2-bit
values fast enough to keep the memory bus saturated?"

## Arithmetic intensity comparison

Recall from [Quantized MatMul on GPUs](../quantized-matmul-gpu/) that Q4\_K
achieves an arithmetic intensity of 3.6 FLOPs/byte during decode. For TQ2\_0
(66 bytes per 256 elements):

$$\text{AI} = \frac{2 \times 256}{66} \approx 7.8 \text{ FLOPs/byte}$$

That looks better, but there is a catch: ternary "FLOPs" are not the same as
floating-point multiply-add operations. Multiplying by $\{-1, 0, +1\}$ is just
an integer add, subtract, or skip — much cheaper than a fused multiply-add. The
effective compute cost is lower, which means the ratio of "useful work per byte
read" is even more favorable.

| Format | Bytes / 256 elems | AI (FLOPs/byte) | Compute type |
|--------|-------------------|------------------|--------------|
| Q4\_K  | 144               | 3.6              | FP16 FMA     |
| TQ2\_0 | 66                | 7.8              | int add/sub  |
| TQ1\_0 | 54                | 9.5              | int add/sub  |

An RTX 5090 needs ~58 FLOPs/byte to saturate its compute units. At 7.8--9.5,
ternary decode is still firmly memory-bound. But that is exactly the point:
we read 2.2x less data than Q4\_K, so at the same bandwidth we generate
tokens 2.2x faster — if the unpack logic is not the bottleneck.

## The W2A8 GEMV approach

Microsoft's BitNet reference kernel (W2A8) shows how to do ternary matmul on
CUDA hardware. The strategy: map ternary weights to int8, then use the GPU's
native integer dot product instruction.

### Packing for HBM storage

Four ternary values are packed into one byte. Each value maps to a 2-bit
encoding: $-1 \to 0$, $0 \to 1$, $+1 \to 2$. Within a 32-bit word, 16
values are interleaved in a specific pattern:

```
32-bit word layout (16 ternary values):

Bit positions:  [31 ····················· 0]
Value indices:  [3,7,11,15, 2,6,10,14, 1,5,9,13, 0,4,8,12]
                  group 3    group 2    group 1    group 0
```

The interleaving pattern — `[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]`
— places values that belong to the same dp4a group in adjacent bit pairs.
This makes extracting a group of 4 values a single shift-and-mask operation
rather than bit gathering from scattered positions.

### Unpacking to shared memory

In shared memory, the packed 2-bit values are expanded back to full int8.
Each 2-bit code becomes the int8 value $\{0, 1, 2\}$, then a bias is
subtracted to recover $\{-1, 0, +1\}$. This costs one subtract per element
but produces data that dp4a can consume directly.

### The dp4a pipeline

```
HBM            Shared Memory     Registers        Output
┌─────────┐    ┌─────────────┐   ┌─────────┐     ┌───────┐
│ packed   │──▶│ unpacked    │──▶│  dp4a   │────▶│  int32 │
│ 2-bit/val│    │ int8/val    │   │ 4×int8  │     │  accum │
│ 66 B/256 │    │ 256 B/256   │   │ per call│     │        │
└─────────┘    └─────────────┘   └─────────┘     └───────┘
                                      ▲
                     ┌────────────────┘
                     │
              ┌──────┴──────┐
              │ activations │
              │ int8 (Q8_1) │
              └─────────────┘
```

## The dp4a instruction

`dp4a` (Dot Product of four int8 values, Accumulate in int32) is a single
hardware instruction available on NVIDIA GPUs since SM 6.1 (Pascal). It
computes:

$$c \mathrel{+}= a_0 b_0 + a_1 b_1 + a_2 b_2 + a_3 b_3$$

where $a_i$ and $b_i$ are signed 8-bit integers packed four to a 32-bit word,
and $c$ is a 32-bit integer accumulator.

### Worked example

Suppose we have 4 ternary weights $w = [+1, -1, 0, +1]$ and 4 activation
values $x = [3, 7, 2, 5]$ (int8-quantized).

The expected dot product:

$$
(+1)(3) + (-1)(7) + (0)(2) + (+1)(5) = 3 - 7 + 0 + 5 = 1
$$

In dp4a terms:

```
a = packed int8: [+1, -1,  0, +1]  →  0x01_FF_00_01
b = packed int8: [ 3,  7,  2,  5]  →  0x03_07_02_05
c = 0  (initial accumulator)

dp4a(a, b, c):
  c += (1)(3) + (-1)(7) + (0)(2) + (1)(5)
  c = 1
```

One instruction, four multiply-accumulates. For a 256-element block, we need
$256 / 4 = 64$ dp4a calls — each taking one clock cycle.

## The MMQ path

llama.cpp takes a different approach for ternary types: reuse the existing
MMQ (Matrix Multiply Quantized) infrastructure that already works for Q4\_K,
Q5\_K, Q6\_K, and Q8\_0.

The insight: after unpacking, a ternary tile looks like regular int8 data.
So the kernel only needs a new unpacking stage; the dot product, reduction,
and tiling logic stay the same.

### How it works

1. **Load**: Read TQ2\_0 blocks from global memory (66 bytes per 256 elements)
2. **Unpack**: In shared memory, expand 2-bit values to 8-bit int8 values
3. **Quantize activations**: The activation vector is quantized to Q8\_1 format
   (32 elements per scale factor, not 256 like TQ2\_0 — the asymmetry is fine
   because the dp4a dot product handles int8 $\times$ int8 regardless of how
   many elements share a scale)
4. **dp4a dot product**: Same kernel code as Q4\_K's MMQ path
5. **Dequantize**: Scale the int32 accumulator by the TQ2\_0 block scale and
   the Q8\_1 activation scale to recover the floating-point result

The advantage: less new code. The Q4\_K MMQ kernel has already been optimized
for coalesced memory access, warp shuffles, and shared memory bank conflicts.
Ternary gets all of those optimizations for free — only the unpack stage is new.

### The unpack logic

For TQ2\_0, unpacking 256 elements from 64 bytes of packed data:

```
For each 32-bit word (16 ternary values):
  val_0 = (word >> 0) & 0x03   // 2-bit extract
  val_1 = (word >> 2) & 0x03
  ...
  val_15 = (word >> 30) & 0x03

  // Map {0,1,2} → {-1,0,+1}
  int8_val = val - 1
```

With the interleaved layout, the extraction naturally groups values that will
be consumed by the same dp4a call. One shift, one mask, one subtract per value.

## LUT-based approaches

An alternative to dp4a: precompute all possible partial sums in a lookup table,
then "multiply" by table lookup instead of arithmetic.

### TL1: basic LUT (2.0 bpw)

Pack 2 ternary values into a 4-bit index. Since each value has 3 states,
2 values have $3^2 = 9$ combinations, which fits in 4 bits (16 slots, 9 used).

For each pair $(w_0, w_1)$ and activation values $(x_0, x_1)$, precompute:

```
LUT[0] = (-1)·x₀ + (-1)·x₁ = -(x₀ + x₁)
LUT[1] = (-1)·x₀ + ( 0)·x₁ = -x₀
LUT[2] = (-1)·x₀ + (+1)·x₁ = -x₀ + x₁
LUT[3] = ( 0)·x₀ + (-1)·x₁ = -x₁
LUT[4] = ( 0)·x₀ + ( 0)·x₁ = 0
LUT[5] = ( 0)·x₀ + (+1)·x₁ = x₁
LUT[6] = (+1)·x₀ + (-1)·x₁ = x₀ - x₁
LUT[7] = (+1)·x₀ + ( 0)·x₁ = x₀
LUT[8] = (+1)·x₀ + (+1)·x₁ = x₀ + x₁
```

The "multiply" becomes: extract 4-bit index from packed weights, look up
the result in a 9-entry table. No multiplication at all.

On CPU, SIMD shuffle instructions do 16--32 lookups per clock cycle:
`vpshufb` on AVX2 or `vqtbl1q` on ARM NEON. Each shuffle processes a
16-byte LUT and 16 indices simultaneously.

### TL2: mirror-consolidated LUT (1.67 bpw)

Pack 3 ternary values into a sign bit plus a 4-bit index. The key
observation: negating all three weights just negates the output. So the
$3^3 = 27$ combinations collapse to $\lceil 27/2 \rceil = 14$ entries
plus a sign bit.

The sign bit is stored separately; the 4-bit index addresses a 14-entry
LUT. This achieves $\lceil \log_2(3^3) \rceil / 3 + \text{sign overhead}
\approx 1.67$ bpw — 17% smaller than TL1.

### LUTs on GPU: an open question

On CPU, SIMD shuffles make LUT approaches competitive with or faster than
dp4a (which CPUs do not have natively). On GPU, the situation is less clear:

| Approach | Compute | Memory access | Proven on GPU? |
|----------|---------|---------------|----------------|
| dp4a     | 4 MAC/instruction | Sequential unpacking | Yes (SM 6.1+) |
| TL1 LUT  | 16 lookups/shuffle | Random access to LUT | Research stage |
| TL2 LUT  | 16 lookups/shuffle + sign | Random access + sign gather | Research stage |

GPU shared memory supports broadcast reads within a warp (when all lanes read
the same address), but LUT lookups are indexed by different weight values —
producing bank conflicts. Whether the reduced arithmetic offsets the bank
conflict penalty is an open research question.

The key advantage of LUTs: they scale linearly with bit width. As packing
density increases (more trits per index), the LUT grows but the lookup cost
stays constant. dp4a is fixed at 4 int8 products per instruction regardless
of how few bits the original data used.

## Why cuda-oxide fits ternary

The [cuda-oxide](../cuda-oxide/) toolchain gives us Rust-level control over
GPU kernel internals. For ternary, this matters more than for Q4\_K.

**Simpler kernel, easier to verify.** Q4\_K dequantization involves
floating-point scale/min pairs, 6-bit sub-group extraction, and FP16
multiply-add. Ternary dequantization is pure integer bit manipulation:
shift, mask, subtract. The entire unpack-and-accumulate path is integer-only,
which is exactly what Kani can reason about — no floating-point
undecidability.

**Reuse existing infrastructure.** tempura's Q4\_K GPU kernel already
implements dp4a-based dot products, warp shuffles, and shared memory tiling
([see the cooperative kernel](../quantized-matmul-gpu/)). The MMQ path for
ternary reuses all of that; only the unpack stage changes.

**End-to-end correctness chain.** The verification story for a ternary kernel:

```
Kani proves:  unpack logic has no OOB access, no panics
              dp4a accumulation does not overflow int32
              (256 elements × max value 127 × 3 < 2³¹)

DualPipeline: GPU kernel output matches CPU reference
              on identical inputs (bit-exact for integer path)

Property tests: scale factor application preserves relative
                ordering (floating-point boundary)
```

Because the core path is integer-only, DualPipeline comparison can be
**bit-exact** — not just within a tolerance like Q4\_K's $5 \times 10^{-7}$
relative error. This is a stronger correctness guarantee than any
floating-point kernel can provide.

## What to read next

For how Kani proves kernel correctness (including the integer-only invariants
that make ternary verification simpler), see Part IV on
[bounded model checking](../bounded-model-checking/) and
[reading Kani harnesses](../reading-kani-harnesses/).

For the full picture of how ternary models work in practice — training,
quantization-aware fine-tuning, and deployment considerations — see the
[BitNet Research Reference](../../framework/bitnet-research/).

## Summary

| Concept | What it is | Why it matters |
|---------|-----------|----------------|
| Ternary AI | 7.8--9.5 FLOPs/byte vs 3.6 for Q4\_K | Still memory-bound, but reads 2.2x less data |
| dp4a | 4 int8 multiply-accumulates in one instruction | Hardware-native path for ternary dot products |
| W2A8 | Pack 4 trits into 1 byte, unpack to int8, dp4a | Microsoft's reference approach for ternary GEMV |
| MMQ path | Unpack ternary to int8, reuse Q4\_K kernel infrastructure | Less new code, proven tiling and reduction |
| TL1 LUT | 9-entry table, 2 trits per lookup (2.0 bpw) | Eliminates multiplication entirely |
| TL2 LUT | 14-entry table + sign bit, 3 trits per lookup (1.67 bpw) | Denser packing via mirror consolidation |
| LUTs on GPU | Shared memory lookups vs dp4a | Open research question — bank conflicts vs reduced arithmetic |
| cuda-oxide fit | Integer-only path, Kani-verifiable, reuses dp4a infrastructure | Bit-exact DualPipeline verification (stronger than FP tolerance) |
