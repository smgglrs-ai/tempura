+++
title = "10. GPU Computing"
description = "Why GPUs run AI — how thousands of simple cores outperform a few fast ones."
weight = 100
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Why not just use a CPU?

Your computer's CPU (Central Processing Unit) is incredibly versatile. It can run a web browser, a spreadsheet, a video game, and a compiler — often simultaneously. It's fast at any single task: a modern CPU core can execute billions of operations per second.

But language model inference has a specific shape: millions of independent multiply-add operations that follow the same pattern. A CPU has 8–16 cores. A GPU has *thousands*.

## The factory analogy

**CPU**: a team of 8 expert engineers. Each one can solve any problem — calculus, welding, writing, plumbing. They're fast and flexible, but there are only 8 of them. If you need to multiply 10 million pairs of numbers, the 8 engineers take turns.

**GPU**: a factory floor with 10,000 workers. Each worker can only do one thing — multiply two numbers and add the result. But they all work simultaneously. That same 10 million multiplications finishes almost instantly because 10,000 happen at once.

Language model inference is a factory job, not an engineering job. The operations are simple (multiply, add) and massively parallel (millions of independent operations). GPUs were designed for exactly this.

## SIMT: Single Instruction, Multiple Threads

GPUs execute the same instruction across many data elements simultaneously. NVIDIA calls this **SIMT** (Single Instruction, Multiple Threads):

```
Instruction: multiply element[i] by weight[i] and add to accumulator

Thread 0:  multiply element[0] by weight[0]   → result[0]
Thread 1:  multiply element[1] by weight[1]   → result[1]
Thread 2:  multiply element[2] by weight[2]   → result[2]
...
Thread 4095: multiply element[4095] by weight[4095] → result[4095]

All 4096 threads execute simultaneously.
```

Every thread runs the same code, but on different data. Thread 0 works on index 0, thread 1 on index 1, and so on. The GPU hardware ensures all threads execute in lockstep — they all perform the multiply at the same clock cycle, then all perform the add at the next cycle.

## How a matrix multiplication maps to a GPU

Recall from [Matrices](../matrices/) that multiplying a $m \times k$ matrix by a $k \times n$ matrix produces a $m \times n$ output. Each output element is a dot product:

$$
C_{ij} = \sum_{p=1}^{k} A_{ip} \times B_{pj}
$$

On a CPU, you'd compute this with three nested loops — one for $i$, one for $j$, one for $p$. With $m = n = k = 4096$ (typical for a 7B model), that's $4096^3 \approx 69$ billion multiply-adds.

On a GPU, you assign one thread (or a small group of threads) to each output element $(i, j)$. There are $m \times n = 16.7$ million output elements, and the GPU has enough threads to work on thousands of them simultaneously. What takes a CPU seconds takes a GPU milliseconds.

## GPU architecture: SMs, warps, and threads

An NVIDIA GPU is organized in a hierarchy:

```
GPU
 └── Streaming Multiprocessors (SMs)   ← 80–144 on modern GPUs
      └── Warps                         ← groups of 32 threads
           └── Threads                   ← the individual workers
```

**Thread**: the smallest unit of execution. Runs one copy of the program on one piece of data.

**Warp**: 32 threads that execute in lockstep. All 32 threads in a warp execute the same instruction at the same time. If thread 5 needs to branch one way and thread 17 needs to branch the other, the warp executes *both* paths (with some threads idle during each) — this is called **divergence** and is the main performance pitfall of GPU programming.

**Streaming Multiprocessor (SM)**: the "brain" of the GPU. Each SM can run multiple warps, switching between them to hide memory latency. A modern GPU has 80–144 SMs.

**The numbers**: an RTX 5090 has 170 SMs, each capable of running multiple warps. At 32 threads per warp, that's tens of thousands of simultaneous threads. For inference, this means processing thousands of matrix elements at once.

## Blocks and grids: how you organize work

When you launch a GPU computation, you organize threads into **blocks**, and blocks into a **grid**:

```
Grid (all the work)
 ├── Block (0,0)  ← 256 threads
 ├── Block (0,1)  ← 256 threads
 ├── Block (1,0)  ← 256 threads
 ├── Block (1,1)  ← 256 threads
 └── ...          ← as many blocks as needed
```

You choose:
- **Block size**: how many threads per block (typically 128 or 256)
- **Grid size**: how many blocks (enough to cover all the work)

For a matrix multiplication with 16.7 million output elements and 256 threads per block, you'd launch $16.7M / 256 \approx 65{,}000$ blocks. The GPU schedules these across its SMs automatically.

## Why GPUs are fast for AI — and what they're bad at

**Fast for AI because:**
- Matrix multiplication = millions of independent multiply-adds → maps perfectly to SIMT
- Attention score computation = many dot products → same pattern
- Softmax = elementwise exp + reduction → parallelizable
- The entire transformer forward pass is composed of these operations

**Bad for:**
- Sequential logic (if-then-else chains where each step depends on the previous)
- Small, irregular tasks (overhead of launching a GPU kernel exceeds the computation)
- Code with heavy branching (warp divergence wastes threads)

Sampling — choosing the next token — is an example of a task that's hard to parallelize. tempura's sampler runs on the CPU for this reason.

## The compute vs memory bottleneck

A surprising fact: for inference (not training), the GPU is usually waiting for data, not computing. A modern GPU can multiply numbers faster than it can read them from memory.

| GPU | Compute (TFLOPS) | Memory bandwidth (TB/s) |
|-----|-----------------|----------------------|
| A100 | 312 | 2.0 |
| H100 | 990 | 3.4 |
| RTX 5090 | 419 | 1.8 |

For inference with batch size 1, each transformer layer reads the entire weight matrix from memory (to multiply with a single token's vector). The weight data is large (megabytes per layer) and the computation is small (one vector × one matrix). Memory bandwidth — how fast data can be moved from GPU memory to the compute units — is the bottleneck.

This is why the [Memory Hierarchy](../memory-hierarchy/) matters so much, and why techniques like quantization (smaller weights = less data to read) help inference more than compute optimizations.

## Check your understanding

1. Why is a GPU faster than a CPU for matrix multiplication, even though each GPU thread is slower than a CPU core?
2. What happens when two threads in the same warp need to take different branches?
3. Why is inference typically memory-bound rather than compute-bound?

**Next:** [Memory Hierarchy](../memory-hierarchy/) — the GPU's memory system and why it determines performance.
