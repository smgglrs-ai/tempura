+++
title = "25. Reading Kani Harnesses"
description = "Annotated walkthrough of a real Kani proof harness from the tempura codebase."
weight = 250
template = "docs/page.html"

[extra]
part = "verification"
toc = true
math = true
+++

## What is a harness?

A Kani harness is a Rust function annotated with `#[kani::proof]` that describes the property to verify. It has three parts:

1. **Declare** symbolic inputs — values that represent "any possible value"
2. **Constrain** them to valid ranges with `kani::assume()`
3. **Assert** the property that must hold for all valid inputs

If Kani finds any input that satisfies all assumptions but violates an assertion, it reports a counterexample. If no such input exists, the property is proven.

## Walkthrough: `attention_q_offset_in_bounds`

This harness from [`kernels/src/attention.rs`](https://github.com/smgglrs-ai/tempura/tree/main/kernels/src/attention.rs) proves that the query offset arithmetic in the attention kernel never produces an out-of-bounds index. Let's read it line by line.

### Part 1: Declare the shape parameters

```rust
#[kani::proof]
fn attention_q_offset_in_bounds() {
    let batch: usize = kani::any();
    let seq_len: usize = kani::any();
    let num_heads: usize = kani::any();
    let head_dim: usize = kani::any();
```

`kani::any()` creates a symbolic value — it represents *every* possible `usize` simultaneously. At this point, `batch` could be 0, 1, 1000, or $2^{64} - 1$. The SAT solver will consider all of them.

### Part 2: Constrain to valid ranges

```rust
    kani::assume(batch >= 1 && batch <= 2);
    kani::assume(seq_len >= 1 && seq_len <= 4);
    kani::assume(num_heads >= 1 && num_heads <= 4);
    kani::assume(head_dim >= 1 && head_dim <= 4);
```

`kani::assume()` restricts the search space. The solver only considers inputs satisfying these constraints. This means:

- $\text{batch} \in \\{1, 2\\}$
- $\text{seq\_len} \in \\{1, 2, 3, 4\\}$
- $\text{num\_heads} \in \\{1, 2, 3, 4\\}$
- $\text{head\_dim} \in \\{1, 2, 3, 4\\}$

That's $2 \times 4 \times 4 \times 4 = 128$ shape combinations. For each, the solver will try every possible index.

**Why these bounds?** They're small enough for the SAT solver to handle quickly, but large enough to exercise every code path. The offset arithmetic is linear — `((b * seq_len + s) * num_heads + h) * head_dim` — so if it's safe at 4, it's safe at 4096 (assuming no integer overflow, which is checked separately).

### Part 3: Compute the buffer size

```rust
    let q_len = batch * seq_len * num_heads * head_dim;
```

This is the total size of the Q buffer. In the actual attention implementation, the caller allocates a buffer of this size. The harness needs to know the buffer size to check if offsets stay within it.

### Part 4: Declare the loop indices

```rust
    let b: usize = kani::any();
    let s: usize = kani::any();
    let h: usize = kani::any();
    kani::assume(b < batch);
    kani::assume(s < seq_len);
    kani::assume(h < num_heads);
```

Three more symbolic values representing a specific position in the iteration space: batch element `b`, sequence position `s`, head `h`. The assumptions constrain them to valid loop bounds.

### Part 5: The property

```rust
    let q_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
    assert!(q_offset + head_dim <= q_len);
}
```

This is the actual offset arithmetic from the attention kernel. The assertion states: *the offset plus the number of elements we'll read (`head_dim`) never exceeds the buffer length*.

If this assertion can be violated for any combination of valid shape parameters and valid indices, Kani will find it. If it can't, the property is proven.

## What this proves in plain language

> For any batch size (1–2), sequence length (1–4), number of heads (1–4), and head dimension (1–4), and for any valid batch/sequence/head index within those ranges, the query offset arithmetic `((b * seq_len + s) * num_heads + h) * head_dim` produces an offset that, together with `head_dim` elements, stays within the allocated buffer.

This means the attention kernel's inner loop — which reads `head_dim` consecutive f32 values starting at `q_offset` — never reads past the end of the Q buffer.

## The companion harness: `attention_kv_offset_in_bounds`

A second harness proves the same property for the key/value buffers, which have different dimensions (they use `kv_len` and `num_kv_heads` instead of `seq_len` and `num_heads`). The GQA head-group mapping `h * num_kv_heads / num_heads` is also covered.

## How to read a harness in a PR

When reviewing a PR that modifies a Kani harness:

1. **Read the `assert!` first.** That's the property being proven. Did it change?
2. **Read the `kani::assume()` calls.** Those define what "valid input" means. Did the assumptions get tighter? Tighter assumptions make the proof easier to pass but weaken what it guarantees — this is the most common mistake.
3. **Compare the offset arithmetic** to the actual kernel code. Does the harness match the implementation? If the kernel changed its indexing formula, the harness must change too — otherwise it's proving a property of code that no longer exists.
4. **Check the bounds.** Are they large enough? If the PR introduces a new code path that only triggers at `num_heads > 4`, and the harness only covers `num_heads <= 4`, the new path is unverified.

## Common harness patterns in tempura

**Offset-bounds pattern** (attention, matmul): declare shape parameters, constrain loop indices, compute offset, assert offset + element count ≤ buffer length.

**Capacity-accounting pattern** (KvCache): declare cache capacity and operations (append, advance), assert slot count invariants (allocated ≤ capacity, released ≤ allocated).

**Index-isolation pattern** (batch invariant): declare two different batch indices, compute their offsets, assert the memory ranges don't overlap.

Recognizing the pattern helps you read new harnesses quickly — most tempura proofs follow one of these three.

## GPU thread decomposition proofs

tempura also verifies CUDA kernel index arithmetic. GPU kernels launch thousands of threads, each computing its position from a flat thread index. A GPU harness proves that this decomposition stays in bounds:

```rust
#[kani::proof]
fn gpu_attention_thread_decomposition_in_bounds() {
    // ... shape parameters with kani::any() and assume ...
    let total = batch * seq_len * num_heads * head_dim;
    let i: usize = kani::any();
    kani::assume(i < total);
    
    let d = i % head_dim;
    let h = (i / head_dim) % num_heads;
    let s = (i / (head_dim * num_heads)) % seq_len;
    let b = i / (head_dim * num_heads * seq_len);
    
    assert!(b < batch);
    assert!(s < seq_len);
    assert!(h < num_heads);
    let q_base = ((b * seq_len + s) * num_heads + h) * head_dim;
    assert!(q_base + d < q_len);
}
```

This proves that a flat thread index `i` decomposes into valid `(batch, seq, head, dim)` coordinates, and the reconstructed offset stays within the buffer. Every CUDA kernel in tempura has a companion decomposition proof.

## The full proof inventory

| Kernel | Proofs | What's proven |
|--------|--------|--------------|
| Attention | 5 | Q offset, KV offset, batch isolation, GPU thread decomposition (×2) |
| RMSNorm | 3 | Index bounds, full execution no-panic, GPU two-pass index safety |
| RoPE | 2 | Split-half index bounds + position array, GPU thread decomposition |
| Sampler | 3 | Top-k index bounds, full execution no-panic, cumsum termination |
| MatMul | 2 | Triple-loop index bounds, GPU tiled GEMM shared memory + output indices |

All harnesses run in CI via `cargo kani --package tempura-kernels`.
