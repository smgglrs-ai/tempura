+++
title = "11. Memory Hierarchy"
description = "Why memory speed matters more than compute speed — the GPU memory system."
weight = 110
template = "docs/page.html"

[extra]
part = "hardware"
toc = true
math = true
+++

## The memory wall

Here's a paradox: a modern GPU can perform 1,000 trillion floating-point operations per second (1 PFLOP). But if you naively write a program that reads each number from main memory before multiplying it, the GPU spends most of its time *waiting for data*, not computing.

The reason: reading from GPU main memory (called **HBM** — High Bandwidth Memory) takes ~400 clock cycles. A multiply-add takes 1 cycle. If every operation reads two values and writes one, the GPU needs 400 cycles per operation just for memory access, even though the arithmetic itself takes 1 cycle. That's 99.75% of the time spent waiting.

The solution: a hierarchy of faster (but smaller) memories, closer to the compute units.

## The hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│  HBM (High Bandwidth Memory)                                │
│  Capacity: 24–80 GB        Speed: ~2 TB/s        Shared by  │
│  Holds: model weights, KV cache, activations    entire GPU  │
├─────────────────────────────────────────────────────────────┤
│  L2 Cache                                                    │
│  Capacity: 6–50 MB         Speed: ~6 TB/s        Shared by  │
│  Holds: recently accessed data (automatic)      entire GPU  │
├─────────────────────────────────────────────────────────────┤
│  Shared Memory / L1 Cache (per SM)                           │
│  Capacity: 128–228 KB      Speed: ~20 TB/s      Shared by   │
│  Holds: tiles of data for cooperative computation  one block │
├─────────────────────────────────────────────────────────────┤
│  Registers (per thread)                                      │
│  Capacity: 256 KB per SM   Speed: instantaneous   Private    │
│  Holds: loop counters, temporary values          per thread  │
└─────────────────────────────────────────────────────────────┘
```

Each level is faster but smaller than the level above:

| Level | Size | Speed | Analogy |
|-------|------|-------|---------|
| Registers | 256 per thread | Instant | Your hands — what you're holding right now |
| Shared memory | 128–228 KB per SM | ~20 TB/s | Your desk — fast to grab, limited space |
| L2 cache | 6–50 MB | ~6 TB/s | Filing cabinet in your office — close but slower |
| HBM | 24–80 GB | ~2 TB/s | Warehouse down the street — lots of space, takes time |

## How matrix multiplication uses the hierarchy

A naive matmul reads every element from HBM: each thread reads a row of A and a column of B, computes their dot product, and writes the result. For a $4096 \times 4096$ multiply, this means reading 4096 elements from HBM per output element.

A **tiled** matmul (which is what tempura and every serious GPU library uses) exploits the hierarchy:

**Step 1:** A block of threads cooperatively loads a tile (e.g., $16 \times 16$) of matrix A and a tile of matrix B from HBM into **shared memory**.

**Step 2:** Each thread computes partial dot products using the tile data from shared memory — which is 10x faster than HBM.

**Step 3:** Load the next tile and accumulate. Repeat until the full dot product is computed.

```
Matrix A                    Matrix B
┌──────────────────┐        ┌──────────────────┐
│ ████             │        │ ██               │
│ ████  tile A₁    │   ×    │ ██  tile B₁      │
│ ████             │        │ ██               │
│                  │        │                  │
│      ████        │        │    ██            │
│      ████ tile A₂│   ×    │    ██ tile B₂    │
│      ████        │        │    ██            │
│                  │        │                  │
└──────────────────┘        └──────────────────┘

Each tile is loaded into shared memory once,
then used by all threads in the block.
```

**Why this helps:** without tiling, each element of A is read from HBM once per output in its row — $n$ times total. With tiling, each element is read from HBM once into shared memory, then read from shared memory $n/\text{tile\_size}$ times. The HBM traffic drops by a factor of `tile_size`.

tempura's `CudaMatMul` uses $16 \times 16$ tiles. On Blackwell GPUs (sm\_100), it also uses **MMA (Matrix Multiply-Accumulate) instructions** that compute entire $16 \times 16$ or $32 \times 32$ submatrices in hardware.

## Why this matters for inference

During inference, the dominant cost is reading model weights from HBM. For a single-token decode step:

- **Weights read:** $\sim$400 MB for a 7B model (all attention + FFN matrices for all layers, in float16)
- **Computation:** $\sim$14 GFLOPs (multiply each weight by one activation)
- **At 2 TB/s HBM bandwidth:** reading takes $\sim$0.2 ms
- **At 400 TFLOPS compute:** arithmetic takes $\sim$0.04 ms

The read is 5x slower than the compute. The GPU is **memory-bound**: compute units sit idle waiting for data. This is why:

1. **Quantization** helps — int4 weights are 4x smaller than float16, so 4x faster to read.
2. **Batching** helps — reading the weights once and multiplying by multiple token vectors amortizes the memory cost.
3. **Caching** helps — the KV cache avoids recomputing keys/values, but it adds to memory usage.

## Shared memory in practice

Shared memory is explicitly managed by the programmer — unlike CPU caches (which are automatic), GPU shared memory requires you to declare what data goes where and when.

In tempura's CUDA kernels:

```rust
// Allocate a 16×16 tile in shared memory
let tile_a: SharedSlice<f32> = shared_memory.slice(0, TILE * TILE);
let tile_b: SharedSlice<f32> = shared_memory.slice(TILE * TILE, TILE * TILE);

// Cooperatively load the tile: each thread loads one element
tile_a[ty * TILE + tx] = a[row * k + (t * TILE + tx)];
tile_b[ty * TILE + tx] = b[(t * TILE + ty) * n + col];

// Synchronize: wait until all threads have finished loading
sync_threads();

// Compute using the tile data (from shared memory, not HBM)
for p in 0..TILE {
    sum += tile_a[ty * TILE + p] * tile_b[p * TILE + tx];
}
```

The `sync_threads()` is critical: thread 0 might need data that thread 15 is loading. Without synchronization, thread 0 could read stale data. This is the fundamental challenge of shared memory programming — coordination between threads.

## Registers and register pressure

Each thread has a small set of private **registers** — the fastest memory on the GPU. Loop counters, accumulator values, and intermediate results live here.

The catch: registers are limited (typically 255 per thread on modern GPUs). If a kernel uses too many registers per thread, the SM can run fewer threads simultaneously, reducing parallelism. This tradeoff is called **register pressure**.

For tempura's kernels, register usage is carefully managed — the Kani proofs verify index arithmetic using values that would live in registers, and cuda-oxide gives the Rust compiler control over register allocation.

## Check your understanding

1. Why is shared memory faster than HBM?
2. What is the purpose of `sync_threads()` in a tiled matmul?
3. Why does quantization (using smaller numbers) speed up inference even though the compute is the same?
4. If a GPU has 2 TB/s bandwidth and a model's weights are 14 GB, how long does one decode step take just for reading weights?

Answers: (1) Physically closer to compute units, on-chip vs off-chip. (2) Ensures all threads have loaded their tile elements before any thread tries to read from the tile. (3) Smaller numbers = less data to read from memory, and inference is memory-bound. (4) $14 \text{ GB} / 2 \text{ TB/s} = 7$ ms — this is why batch size > 1 matters.

**Next:** [Writing GPU Kernels](../writing-gpu-kernels/) — the CUDA programming model.
