+++
title = "12. Writing GPU Kernels"
description = "Threads, blocks, shared memory — the CUDA programming model in practice."
weight = 120
template = "docs/page.html"

[extra]
part = "hardware"
toc = true
math = true
+++

## What is a kernel?

In GPU programming, a **kernel** is a function that runs on the GPU. You write it once, and the GPU executes it across thousands of threads simultaneously. The word "kernel" here has nothing to do with operating system kernels — it's just the GPU programming term for "the function that does the parallel work."

In tempura, each mathematical operation has a kernel: attention, RMSNorm, RoPE, MatMul, sampling. Each kernel is a function that transforms input arrays into output arrays, with thousands of GPU threads working on different elements simultaneously.

## The simplest kernel: element-wise operations

The simplest GPU kernel processes each element independently. Consider adding two vectors:

```
// Pseudocode for a GPU kernel
kernel vector_add(a: &[f32], b: &[f32], output: &mut [f32]) {
    let i = global_thread_index();   // which element am I responsible for?
    output[i] = a[i] + b[i];        // do the work
}

// Launch: one thread per element
launch vector_add with n threads
```

Each thread computes its own index $i$ from its position in the grid, reads `a[i]` and `b[i]`, computes the sum, and writes `output[i]`. No thread needs to communicate with any other thread — this is called an **embarrassingly parallel** operation.

RoPE is close to this pattern: each thread rotates one pair of dimensions independently.

## Thread indexing: the fundamental skill

Every kernel starts with the same question: "which piece of data am I responsible for?"

The thread's identity comes from built-in variables:

| Variable | What it means |
|----------|--------------|
| `thread_idx` | This thread's index within its block (0 to block_size - 1) |
| `block_idx` | This block's index within the grid |
| `block_dim` | Number of threads per block |

The global index is:

$$
i = \text{block\_idx} \times \text{block\_dim} + \text{thread\_idx}
$$

**Example:** block size = 256, block 3, thread 42:

$$
i = 3 \times 256 + 42 = 810
$$

This thread processes element 810 of the input array.

**This is exactly the arithmetic that tempura's Kani proofs verify.** The [attention offset harness](reading-kani-harnesses/) proves that a more complex version of this indexing — `((b * seq_len + s) * num_heads + h) * head_dim` — never exceeds the buffer length for any valid thread assignment.

## A real example: vector scaling

Let's trace through a complete kernel that multiplies every element by a constant:

```
// Kernel function — runs on the GPU
kernel scale(x: &mut [f32], factor: f32, n: usize) {
    let i = block_idx * block_dim + thread_idx;
    if i < n {        // guard: don't process beyond the array
        x[i] *= factor;
    }
}

// Host code — runs on the CPU
let n = 10_000;
let block_size = 256;
let grid_size = (n + block_size - 1) / block_size;  // = 40 blocks (ceiling division)

launch scale<<<grid_size, block_size>>>(device_x, 2.0, n);
```

**Why the guard?** 40 blocks × 256 threads = 10,240 threads, but we only have 10,000 elements. Threads 10,000–10,239 have nothing to do. The `if i < n` check prevents them from writing past the end of the array. Without this check, you'd get an out-of-bounds access — exactly the kind of bug Kani catches.

## Reductions: the hard problem

Element-wise operations are easy — no threads need to talk to each other. But some operations require combining values from many elements. Computing the sum of an array, the maximum, or the mean — these are **reductions**.

Reductions are hard to parallelize because each step depends on the previous:

$$
\text{sum} = x_0 + x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7
$$

**Sequential**: add left to right. $O(n)$ operations, no parallelism.

**Parallel tree reduction**: add pairs simultaneously at each level:

```
Level 0:  x₀  x₁  x₂  x₃  x₄  x₅  x₆  x₇
           \  /    \  /    \  /    \  /
Level 1:   s₀₁    s₂₃    s₄₅    s₆₇      ← 4 additions in parallel
             \    /          \    /
Level 2:     s₀₁₂₃        s₄₅₆₇           ← 2 additions in parallel
                 \        /
Level 3:         total_sum                   ← 1 addition
```

This takes $\log_2(n)$ steps instead of $n$, but each step requires synchronization — all threads must finish their additions before the next level starts.

**Where this appears in tempura:**
- **RMSNorm**: sum of squares (a reduction), then divide (elementwise). Two kernels: one for the reduction, one for the normalization.
- **Softmax**: find the max (reduction), subtract and exponentiate (elementwise), sum the exponentials (reduction), divide (elementwise).
- **Attention scores**: dot products are reductions along the key dimension.

## Synchronization

When threads need to cooperate, they must synchronize:

**`sync_threads()`**: all threads in a block wait until every thread reaches this point. Used after loading shared memory tiles.

**Atomic operations**: when multiple threads need to update the same memory location. `atomic_add(&sum, my_value)` ensures additions don't collide.

**Warp-level primitives**: threads within a warp (32 threads) can share data directly through registers using `warp_shuffle`. This is faster than shared memory for small reductions. tempura's RMSNorm uses warp-level reduction for the sum of squares within each row.

## Occupancy: using the GPU fully

A kernel's **occupancy** is the fraction of the GPU's threads that are actually active. Low occupancy means the GPU is underutilized.

Occupancy depends on resources per thread:
- **Registers**: more registers per thread → fewer threads per SM
- **Shared memory**: more shared memory per block → fewer blocks per SM
- **Block size**: too small → not enough threads to hide memory latency

The art of GPU programming is finding the right balance: enough registers and shared memory to do useful work, but not so much that occupancy drops and the GPU sits idle.

## The lifecycle of a kernel call

From start to finish:

1. **CPU allocates GPU memory** (`malloc` on the device)
2. **CPU copies input data** from system RAM to GPU memory (HBM)
3. **CPU launches the kernel** with a grid/block configuration
4. **GPU executes** thousands of threads in parallel
5. **CPU copies results back** from GPU memory to system RAM
6. **CPU frees GPU memory**

Steps 2 and 5 are the **data transfer overhead**. For large models, the weights stay on the GPU permanently — they're loaded once at startup. Only the input tokens and output logits move back and forth.

tempura's GPU kernels keep weight matrices and KV cache on the GPU. Only token IDs and generated text cross the CPU-GPU boundary.

## Why kernel correctness is hard

GPU bugs are particularly nasty because:

**No helpful crashes.** Out-of-bounds reads on a GPU often return garbage data instead of crashing. The kernel "works" but produces wrong output. This is why tempura's Kani proofs for index arithmetic matter — they catch these bugs at compile time, not in production.

**Race conditions.** Two threads writing to the same location: one wins, one loses, and the result depends on timing. Hard to reproduce, hard to debug.

**Divergence.** Threads in a warp must execute the same instruction. If an `if` statement sends half the warp one way and half the other, both paths execute (with inactive threads masked out), halving performance.

**Floating-point non-determinism.** A tree reduction adds numbers in a different order than a sequential sum. With floating-point arithmetic, $(a + b) + c \neq a + (b + c)$ at the limits of precision. Different results on GPU vs CPU, even with no bugs. This is exactly what [DualPipeline](dual-pipeline/) detects and quantifies.

## Check your understanding

1. What does `let i = block_idx * block_dim + thread_idx` compute?
2. Why do we need `if i < n` in most kernels?
3. Why can't you just add all elements of an array in one parallel step?
4. What goes wrong if you skip `sync_threads()` after loading shared memory?

**Next:** [cuda-oxide](../cuda-oxide/) — writing GPU kernels in Rust instead of C.
