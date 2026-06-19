+++
title = "13. cuda-oxide"
description = "GPU kernels in Rust — what cuda-oxide changes and why it matters for verification."
weight = 130
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## The CUDA ecosystem

NVIDIA GPUs are programmed using **CUDA** (Compute Unified Device Architecture) — a parallel computing platform with its own compiler, runtime, and programming model. For 17 years (since 2007), CUDA kernels have been written in **CUDA C/C++**, a C-like language with extensions for GPU concepts.

CUDA C works. It's fast, well-documented, and has a massive ecosystem. But it inherits C's problems: manual memory management, pointer arithmetic, no bounds checking, and undefined behavior on errors. These problems are manageable for simple kernels, but language model kernels are complex — thousands of lines of index arithmetic, shared memory management, and synchronization.

## What cuda-oxide is

[cuda-oxide](https://github.com/NVlabs/cuda-oxide) is a research project from NVIDIA Research (NVLabs) that lets you write CUDA kernels in **Rust** instead of C. It provides:

1. **A custom Rust compiler backend** that emits NVIDIA PTX (the GPU's assembly language) instead of x86 machine code. You write normal Rust, and it compiles to GPU instructions.
2. **Rust's type system and safety guarantees** on the GPU: bounds checking, no null pointers, no uninitialized memory, no data races (at the language level).
3. **Access to GPU intrinsics**: shared memory, synchronization, warp operations, tensor core instructions — expressed as Rust APIs instead of C intrinsics.

The key insight: Rust's compiler catches entire classes of bugs that CUDA C allows. A bounds-check violation that would silently corrupt data in CUDA C becomes a compile-time error in cuda-oxide.

## Why Rust on the GPU?

### Memory safety

CUDA C:
```c
// CUDA C — compiles and runs, but out of bounds
__global__ void bad_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = 42.0f;  // no bounds check — what if i >= n?
}
```

cuda-oxide (Rust):
```rust
// Rust — the slice bounds are checked
fn kernel(data: &mut DeviceSlice<f32>) {
    let i = thread_index();
    if i < data.len() {
        data[i] = 42.0;  // safe — bounds checked at compile time via the slice type
    }
    // data[i] without the check → compile error or runtime panic, never silent corruption
}
```

This isn't just convenience — for tempura, it means that the same code Kani verifies is the same code that runs on the GPU. No translation gap between "the Rust code we proved safe" and "the C code we actually deployed."

### Ownership and lifetimes

Rust's ownership system prevents use-after-free and double-free bugs:

```rust
let buffer = device_alloc::<f32>(1024);   // allocate on GPU
launch_kernel(&buffer);                    // use the buffer
// buffer is automatically freed when it goes out of scope
// no manual free(), no memory leak, no use-after-free
```

In CUDA C, forgetting to call `cudaFree()` leaks memory. Calling it twice corrupts the allocator. Using the pointer after freeing is undefined behavior. Rust makes all three impossible.

### No undefined behavior

CUDA C has numerous sources of undefined behavior — using uninitialized shared memory, integer overflow, out-of-bounds pointer arithmetic. Each of these can produce silent wrong results that are almost impossible to debug.

Rust has **no undefined behavior** in safe code. The compiler either accepts your code (and it's safe) or rejects it (and you have to fix it). For GPU kernels doing complex index arithmetic over multi-dimensional tensors, this is a major advantage.

## How tempura uses cuda-oxide

tempura's GPU kernels are in `kernels/src/gpu/`. Each kernel implements the same Rust trait as the CPU reference:

```rust
// CPU and GPU implement the same trait
pub trait Attention: Send + Sync {
    fn forward(&self, q: &[f32], k: &[f32], v: &[f32],
               output: &mut [f32], /* ... */) -> Result<()>;
}

// CPU implementation — pure Rust, runs on CPU
pub struct CpuAttention;
impl Attention for CpuAttention { /* ... */ }

// GPU implementation — Rust compiled to PTX via cuda-oxide
pub struct CudaAttention { /* GPU context */ }
impl Attention for CudaAttention { /* launches GPU kernel */ }
```

The trait signature is identical. The CPU version is the reference; the GPU version is the optimized implementation. [DualPipeline](dual-pipeline/) compares them.

## Building with cuda-oxide

cuda-oxide uses a custom Rust compiler (`cargo oxide build` instead of `cargo build`). The workflow:

```sh
# Build the GPU kernels
cargo oxide build --release

# Run GPU tests — compares GPU output to CPU reference
cargo oxide run --bin tests-gpu

# Verify with Kani — proves index arithmetic for both CPU and GPU harnesses
cargo kani --package tempura-kernels
```

The critical point: **Kani verifies the same Rust code that cuda-oxide compiles to PTX.** There's no translation step where bugs could hide. The code you prove is the code you run.

## What cuda-oxide provides for kernels

| Capability | cuda-oxide API | CUDA C equivalent |
|-----------|---------------|-------------------|
| Thread index | `thread_idx()`, `block_idx()` | `threadIdx.x`, `blockIdx.x` |
| Shared memory | `SharedSlice<f32>` | `__shared__ float[]` |
| Synchronization | `sync_threads()` | `__syncthreads()` |
| Atomic operations | `atomic_add()` | `atomicAdd()` |
| Warp shuffle | `warp_shuffle_down()` | `__shfl_down_sync()` |
| MMA instructions | `sgemm_mma_32x32()` | Inline PTX / `wmma::mma_sync()` |
| Device memory | `DeviceBuffer<f32>` | `float*` + `cudaMalloc` |

The APIs map closely to CUDA C, but with Rust's type system ensuring safe usage.

## Tensor cores and MMA

Modern NVIDIA GPUs have specialized hardware for matrix multiplication called **Tensor Cores**. Instead of computing one multiply-add per cycle, a tensor core computes an entire small matrix multiply (e.g., $16 \times 16 \times 16$) in one instruction.

tempura's `CudaMatMul` uses MMA (Matrix Multiply-Accumulate) instructions on Blackwell GPUs (sm\_100+):

```rust
// One MMA instruction computes a 32×32 tile in hardware
sgemm_mma_32x32(tile_a, tile_b, &mut accumulator);
```

This is where the "bleeding edge" is — cuda-oxide gives Rust-level access to the newest GPU instructions (like `tcgen05` on Blackwell) that CUDA C requires inline PTX assembly for.

## The verification story

With cuda-oxide, tempura achieves something no other inference server can: **formal verification of GPU kernel safety.**

```
Rust source code
    │
    ├──→ Kani: proves index arithmetic safe for all valid inputs
    │
    └──→ cuda-oxide: compiles to GPU PTX → runs on NVIDIA hardware
```

The same source → two outputs → one is a proof, the other is the binary. There's no gap between "what we proved" and "what we run."

Compare this to the traditional approach:

```
C++ source code
    │
    ├──→ (no formal verification available for CUDA C)
    │
    └──→ nvcc: compiles to PTX → runs on NVIDIA hardware
    
Trust, but don't verify.
```

## Current limitations

cuda-oxide is a research project, not a production-ready toolkit. Current limitations:

- **Custom compiler.** You need the cuda-oxide fork of rustc, not the standard Rust compiler. This is why tempura uses `cargo oxide build` instead of `cargo build`.
- **Not all Rust features work.** The GPU backend supports a subset of Rust — no standard library, no heap allocation, no dynamic dispatch. Kernel code must be simple and predictable.
- **Limited debugging.** GPU debugging tools (like `cuda-gdb`) don't understand Rust DWARF info. `printf` debugging is the current fallback.
- **NVIDIA only.** cuda-oxide targets NVIDIA PTX. AMD (ROCm) and Intel GPUs are not supported.

These are engineering limitations, not fundamental ones. The verification story — Kani proofs on Rust code that runs on the GPU — is the unique contribution.

## Check your understanding

1. What advantage does Rust give over C for GPU kernel code?
2. Why is it significant that Kani and cuda-oxide both operate on the same Rust source?
3. What is an MMA instruction and why is it faster than regular multiply-add?
4. Why does tempura keep the CPU reference implementation alongside the GPU one?

**Next:** [From Math to Metal](../from-math-to-metal/) — how tempura maps the transformer's math to GPU hardware.
