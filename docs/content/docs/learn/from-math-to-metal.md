+++
title = "14. From Math to Metal"
description = "How tempura maps attention, matmul, and normalization to GPU hardware."
weight = 140
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## The full picture

You've now learned everything needed to understand tempura end-to-end:

- **Part I** taught you the math: vectors, matrices, attention, softmax, the transformer architecture
- **Part II** taught you the hardware: GPUs, memory hierarchy, kernels, cuda-oxide

This final page connects the two: how does each mathematical operation from Part I map to GPU hardware?

## MatMul: the workhorse

Matrix multiplication dominates inference compute. In a single transformer layer:

| Operation | Matrix sizes (7B model) | FLOPs |
|-----------|------------------------|-------|
| Q projection | $[1, 4096] \times [4096, 4096]$ | 33M |
| K projection | $[1, 4096] \times [4096, 4096]$ | 33M |
| V projection | $[1, 4096] \times [4096, 4096]$ | 33M |
| Output projection | $[1, 4096] \times [4096, 4096]$ | 33M |
| FFN gate | $[1, 4096] \times [4096, 11008]$ | 90M |
| FFN up | $[1, 4096] \times [4096, 11008]$ | 90M |
| FFN down | $[1, 11008] \times [11008, 4096]$ | 90M |

**Per layer total: ~400M FLOPs.** Across 32 layers: **~13 GFLOPs per token.**

### How tempura tiles it

tempura's `CudaMatMul` uses a tiled approach (from [Memory Hierarchy](../memory-hierarchy/)):

1. **Grid:** one block per $16 \times 16$ output tile
2. **Threads:** 256 threads per block ($16 \times 16$)
3. **Loop:** iterate over $k$ in tiles of 16, loading $A$ and $B$ tiles into shared memory
4. **Accumulate:** each thread accumulates its output element in registers
5. **Write:** store the final result to HBM

On Blackwell GPUs, step 4 uses MMA tensor core instructions — computing an entire $32 \times 32$ submatrix in hardware instead of individual multiply-adds.

**Kani proves:** the index decomposition $\text{row} = \text{block\_y} \times 16 + \text{ty}$, $\text{col} = \text{block\_x} \times 16 + \text{tx}$ never produces out-of-bounds indices for the output, A-tile, or B-tile accesses.

## Attention: the memory challenge

Attention is mathematically simple ($\text{softmax}(QK^T / \sqrt{d_k}) V$) but tricky on GPUs because the attention score matrix $QK^T$ can be enormous: for a 2048-token sequence, it's $2048 \times 2048 = 4M$ entries per head. With 32 heads, that's 128M entries — 512 MB in f32. Reading and writing this matrix to HBM is slow.

### Flash Attention

tempura's CUDA attention kernel uses **Flash Attention** — an algorithm that computes attention without ever materializing the full score matrix in HBM:

1. **Tile Q** into blocks of rows
2. **For each Q tile, iterate over K/V tiles:**
   - Compute partial scores $QK^T$ for this tile pair (in shared memory)
   - Apply softmax incrementally (using an online algorithm that tracks the running max and sum)
   - Accumulate the weighted V contribution
3. **Write the final output** to HBM

The key trick: softmax is computed *online* — you don't need to see all scores before normalizing. The running statistics ($\max$ and $\sum e^{x - \max}$) update as each K/V tile is processed.

**Memory savings:** the score matrix lives entirely in shared memory, never in HBM. For a 2048-token sequence, this reduces HBM traffic from ~512 MB to ~32 MB — a 16x reduction.

**Kani proves:** the Q offset and KV offset arithmetic in the tiled loops stays within the buffer bounds. The batch isolation proof ensures different sequences' data never overlaps.

## RMSNorm: two passes

[RMS Normalization](../../foundations/normalization/) requires a reduction (sum of squares) followed by an elementwise operation (divide by RMS). These can't be fused into one kernel because the division needs the completed sum.

**Pass 1 — Sum of squares:**
- Launch one warp (32 threads) per token (row of the hidden state)
- Each thread sums a stripe of the row: thread $t$ adds elements $t, t+32, t+64, \ldots$
- Warp-level reduction (`warp_shuffle_down`) combines 32 partial sums into one total
- Store $\text{sum}[t]$ to a small buffer (one value per token)

**Pass 2 — Normalize:**
- For each element: $x[i] = x[i] \times \text{weight}[i] / \sqrt{\text{sum}[\text{row}] / d + \epsilon}$
- This is elementwise — one thread per element, no synchronization needed

**Kani proves:** both passes' index arithmetic stays in bounds — the pass-1 stride pattern `start + i` never exceeds the buffer, and pass-2's decomposition of flat index into `(row, col)` correctly maps to the sum buffer and weight array.

## RoPE: elementwise rotation

[RoPE](../../foundations/positional-encoding/) applies a 2D rotation to pairs of dimensions. Each pair is independent — perfect for GPU parallelism.

**Mapping:** one thread per element. Thread $i$ computes its position in the tensor:

$$
s = i / (\text{num\_heads} \times \text{head\_dim}), \quad h = (i / \text{head\_dim}) \% \text{num\_heads}, \quad d = i \% \text{head\_dim}
$$

If $d < \text{half\_dim}$, this thread handles the "cosine" term. It reads its paired element at $\text{offset} + d + \text{half\_dim}$ and computes the rotation.

This is the simplest of tempura's GPU kernels — no shared memory, no synchronization, no reductions. The complexity is in the index arithmetic, which Kani proves safe.

**GPU vs CPU divergence:** on an RTX 5090, the RoPE kernel produces results within $2.4 \times 10^{-7}$ of the CPU reference — the limit of f32 precision for trigonometric operations.

## The forward pass on GPU

Putting it all together, one transformer layer on the GPU:

```
Input hidden state (in HBM)
    │
    ├── RMSNorm Pass 1: warp reduction → sum_of_squares (in HBM, small)
    ├── RMSNorm Pass 2: elementwise normalize → norm_h (in HBM)
    │
    ├── MatMul: norm_h × W_Q → Q (in HBM)
    ├── MatMul: norm_h × W_K → K (in HBM, also stored in KV cache)
    ├── MatMul: norm_h × W_V → V (in HBM, also stored in KV cache)
    │
    ├── RoPE: rotate Q and K in-place (in HBM)
    │
    ├── Attention: Flash Attention with cached K, V → attn_out (in HBM)
    │
    ├── MatMul: attn_out × W_O → projected (in HBM)
    │
    ├── Residual add: h + projected → h' (elementwise, in HBM)
    │
    ├── RMSNorm Pass 1 + 2: normalize h' → norm_h'
    │
    ├── MatMul: norm_h' × W_gate → gate (in HBM)
    ├── MatMul: norm_h' × W_up → up (in HBM)
    ├── SiLU(gate) × up → ffn_intermediate (elementwise, in HBM)
    ├── MatMul: ffn_intermediate × W_down → ffn_out (in HBM)
    │
    └── Residual add: h' + ffn_out → h'' (elementwise, in HBM)

Output hidden state h'' (in HBM)
```

That's **7 matrix multiplications, 2 RMSNorm passes, 1 RoPE, 1 attention, and 4 elementwise operations** per layer. For 32 layers: 224 matmuls, 64 norm passes, 32 attention kernels, and 128 elementwise operations — all to generate a single token.

## The verification guarantee

For every kernel in this pipeline:

1. **Kani proves** the index arithmetic never goes out of bounds — for any valid model configuration within bounded ranges
2. **DualPipeline** verifies the GPU output matches the CPU reference within f32 precision
3. **InferenceTrace** records per-layer statistics so you can inspect any step

This is what tempura means by "verified inference" — not just tested, but proven at the arithmetic level and compared at the numerical level.

## What you now understand

If you've followed from Chapter 0 to here, you can trace the entire path:

1. A user types a prompt → **tokenized** into integer IDs
2. IDs → **embedded** into vectors via a lookup table
3. Vectors → **transformed** through 32 transformer layers, each with attention and FFN
4. Attention uses **dot products** (Chapter 1) to measure word relevance
5. The math runs as **matrix multiplications** (Chapter 2) on **GPU hardware** (Chapter 10)
6. GPU kernels use **tiling and shared memory** (Chapter 11) for efficiency
7. Kernels are written in **Rust** (Chapter 13) and formally **verified with Kani**
8. The output logits → **softmax** (Chapter 4) → **sampling** → next token
9. Repeat until done

No magic anywhere. Just arithmetic — addition, multiplication, exponentiation — executed on parallel hardware, with proofs that it's done correctly.

## Where to go next

- **[Foundations](../../foundations/)** — each kernel's math at implementation depth, with Kani proof walkthroughs
- **[Architecture](../../architecture/)** — how tempura's crates are organized
- **[Getting Started](../../getting-started/)** — build and run tempura yourself
- **[Framework Guide](../../framework/)** — implement your own kernel backend

## Check your understanding

1. How many matrix multiplications does one transformer layer require?
2. Why does Flash Attention avoid materializing the full score matrix?
3. Why does RMSNorm need two GPU kernel launches?
4. What three verification mechanisms does tempura apply to each kernel?
5. Trace the path of the word "cat" from input text to its contribution to the next token's probability.
