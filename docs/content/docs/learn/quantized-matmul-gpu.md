+++
title = "20. Quantized MatMul on GPUs"
description = "Q4_K block layout, decode vs prefill parallelism, warp-cooperative kernels, and why memory bandwidth is the wall."
weight = 200
template = "docs/page.html"

[extra]
part = "kernel-math"
toc = true
math = true
+++

## The problem

The dominant operation in LLM inference is matrix-vector multiplication during
decode: one token at a time, each requiring a dot product of the activation
vector ($k$ elements) against every row of the weight matrix ($n \times k$).
With quantized weights, we read packed integers and dequantize on the fly.

The arithmetic is cheap. The bottleneck is reading the weights from memory.

## Q4\_K block layout

GGML's Q4\_K format packs 256 `float32` values into 144 bytes — a 7.1x
compression ratio. Each block contains:

```
Offset  Size  Content
──────  ────  ───────────────────────────────
0       2     d      (float16) — super-block scale
2       2     dmin   (float16) — super-block minimum
4       12    scales (12 bytes) — 8 packed 6-bit scale/min pairs
16      128   qs     (128 bytes) — 256 4-bit quantized values
──────  ────
        144 bytes total
```

The 256 elements are organized as 4 groups of 64. Each group has two
sub-groups of 32 elements sharing a scale and minimum value. The
dequantization formula for element $i$ in sub-group $(j, \text{half})$:

$$x_i = d \cdot \text{sc}[j] \cdot q_i - d_{\min} \cdot m[j]$$

where $q_i$ is the 4-bit quantized value (0–15), $\text{sc}[j]$ is the
6-bit per-group scale, and $m[j]$ is the 6-bit per-group minimum.

## Decode is memory-bound

For decode ($m = 1$), the dot product reads the entire weight matrix once
and produces $n$ output values. The arithmetic intensity:

$$\text{AI} = \frac{2nk \text{ FLOPs}}{nk/256 \times 144 \text{ bytes}} = \frac{2 \times 256}{144} \approx 3.6 \text{ FLOPs/byte}$$

An RTX 5090 has ~105 TFLOPS compute and ~1.8 TB/s bandwidth. The
compute-to-bandwidth ratio:

$$\frac{105 \times 10^{12}}{1.8 \times 10^{12}} \approx 58 \text{ FLOPs/byte}$$

Since 3.6 < 58, decode is **firmly memory-bandwidth bound**. The GPU's
compute units are idle most of the time, waiting for data from VRAM.
No amount of compute optimization can help — only better memory access
patterns matter.

For prefill ($m = 128$), the arithmetic intensity rises by $128\times$ to
~460 FLOPs/byte, pushing the operation into the compute-bound regime.
This is why our GPU kernels show 1900+ GFLOPS on prefill but struggle
on decode.

## Three kernel strategies

### Strategy 1: One thread per output (naive)

The simplest approach: launch $m \times n$ threads, each computing one
complete dot product over all $k/256$ blocks.

```
Grid:  (ceil(m*n / 256), 1, 1)
Block: (256, 1, 1)
Each thread: reads ~74 blocks × 144 bytes = 10.6 KB sequentially
```

This works well when $n$ is large (gate\_proj: $n = 18944$) because
there are enough threads to saturate the memory controllers. It fails
when $n$ is small (down\_proj: $n = 3584$) because 3584 threads cannot
generate enough memory transactions to fill the pipeline.

**Measured on RTX 5090:**

| Layer | N | K | Time | GFLOPS | BW |
|-------|-------|--------|--------|--------|---------|
| gate\_proj | 18944 | 3584 | 0.099ms | 1372 | 386 GB/s |
| down\_proj | 3584 | 18944 | 0.504ms | 269 | 75 GB/s |

The gate\_proj achieves 21% of peak bandwidth. The down\_proj achieves
only 4% — a 5x inefficiency from insufficient parallelism.

### Strategy 2: Split-K

Partition the $k$ dimension across multiple thread groups. Each group
computes a partial dot product; a reduction kernel sums the partials.

This increases thread count from $n$ to $n \times S$ (where $S$ is the
number of splits), but adds a synchronization point and extra memory
traffic for the partial sums.

**Result:** No improvement on down\_proj. The extra threads don't help
because the bottleneck is per-thread memory access patterns, not thread
count. Each thread still reads non-adjacent 144-byte blocks.

### Strategy 3: Warp-cooperative (what works)

Assign one warp (32 threads) to each output element. The 32 lanes split
the $k/256$ blocks: lane $i$ processes blocks $i, i+32, i+64, \ldots$
Adjacent lanes read adjacent blocks, which coalesces into 32-wide
memory transactions.

After accumulation, a 5-step warp shuffle reduces the 32 partial sums:

```rust
sum += warp::shuffle_xor_f32(sum, 16);
sum += warp::shuffle_xor_f32(sum, 8);
sum += warp::shuffle_xor_f32(sum, 4);
sum += warp::shuffle_xor_f32(sum, 2);
sum += warp::shuffle_xor_f32(sum, 1);
```

Only lane 0 writes the final result.

**Measured on RTX 5090:**

| Layer | N | K | Naive | Warp | Speedup |
|-------|-------|--------|---------|---------|---------|
| qkv\_proj | 3584 | 3584 | 0.098ms | 0.041ms | **2.4x** |
| down\_proj | 3584 | 18944 | 0.504ms | 0.194ms | **2.6x** |
| gate\_proj | 18944 | 3584 | 0.098ms | 0.167ms | 0.59x |
| lm\_head | 151936 | 3584 | 0.612ms | 1.296ms | 0.47x |

The warp kernel wins when $n$ is small (the bottleneck case) and loses
when $n$ is large (the naive kernel already has enough parallelism).
The optimal strategy selects the kernel based on $n$.

## The bandwidth wall

With the best kernel per layer, projected Qwen3-8B decode:

| Component | Kernel | Per-layer | Total (32 layers) |
|-----------|--------|-----------|-------------------|
| qkv\_proj | warp | 0.041ms | 1.31ms |
| o\_proj | warp | 0.041ms | 1.31ms |
| gate\_proj | naive | 0.098ms | 3.14ms |
| up\_proj | naive | 0.098ms | 3.14ms |
| down\_proj | warp | 0.194ms | 6.21ms |
| lm\_head | naive | — | 0.61ms |
| **Total** | | | **15.7ms** |

That is **64 tokens/second** (matmul-only — no attention, norms, or KV
cache). For comparison, llama.cpp's **entire forward pass** (including all
operations) takes 4.78ms on the same hardware, achieving **209 tokens/second**
(measured with `llama-bench`, build 468bdec, RTX 5090, Qwen3-8B Q4\_K\_M,
`-ngl 99`). Our matmul kernels alone take 3.3x longer than llama.cpp's
complete token generation.

The remaining gap is bandwidth utilization. Our best kernel achieves
~200 GB/s; llama.cpp achieves ~750+ GB/s (inferred from 4.78ms total
for ~37 MB of Q4\_K weights plus activations). The difference comes from:

1. **Activation vector access**: each thread reads `a[blk * 256 + j]`
   which is strided across blocks. llama.cpp keeps activation tiles in
   registers across blocks, reducing redundant loads.

2. **Block header overhead**: reading 16 bytes of scales per 128 bytes
   of quantized data (12.5% overhead) fragments memory transactions.

3. **Occupancy**: 4 warps per CTA limits the number of in-flight memory
   requests per SM. Higher occupancy allows the memory controller to
   hide latency better.

These are engineering optimizations, not algorithmic barriers. The
theoretical minimum at 1.8 TB/s bandwidth is ~0.6ms per forward pass
(37 MB of Q4\_K weights). Our 15.7ms matmul-only total represents 4% of
theoretical peak. llama.cpp at 4.78ms total achieves ~12%. Both have room
to improve, but llama.cpp is 3x closer to the hardware limit.

## What tempura proves that llama.cpp does not

tempura's quantized matmul has formal verification properties that no
other inference engine provides:

- **CpuMatMulQ4K** has Kani proofs for bounds safety (no OOB access,
  no panics) on the dequantization arithmetic
- **DualPipeline** validates GPU output against the proven CPU reference
  (max relative error: $5 \times 10^{-7}$)
- **VerificationTier** tracks which kernel produced each inference result

The performance gap is real and documented. The correctness guarantees
are also real and unique. tempura prioritizes knowing the answer is
right over computing it fast — then closes the speed gap iteratively
with verified optimizations.

## Links to source

- CPU Q4K reference: [`kernels/src/quantized.rs` — `CpuMatMulQ4K`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/quantized.rs)
- GPU naive kernel: [`kernels/src/gpu/quantized_gemm_kernel.rs` — `gemm_q4k_bt`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/gpu/quantized_gemm_kernel.rs)
- GPU warp kernel: [`kernels/src/gpu/quantized_gemm_kernel.rs` — `gemv_q4k_warp`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/gpu/quantized_gemm_kernel.rs)
- Kani proofs: [`kernels/src/quantized.rs` — `mod kani_proofs`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/quantized.rs)
