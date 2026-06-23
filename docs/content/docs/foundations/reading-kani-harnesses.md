+++
title = "Reading Kani Harnesses"
description = "Annotated walkthrough of a real Kani harness, mapping code to property."
weight = 70
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Anatomy of a harness

Every Kani harness follows this pattern:

```rust
#[kani::proof]
fn property_name() {
    // 1. Declare symbolic inputs
    let x: usize = kani::any();
    
    // 2. Constrain inputs to realistic bounds
    kani::assume(x >= 1 && x <= 64);
    
    // 3. Compute derived values (same arithmetic as the real code)
    let offset = x * stride;
    
    // 4. Assert the safety property
    assert!(offset + stride <= buffer_size);
}
```

## Real example: attention Q offset bounds

From `kernels/src/attention.rs`:

```rust
#[kani::proof]
fn attention_q_offset_in_bounds() {
    let batch: usize = kani::any();
    let seq_len: usize = kani::any();
    let num_heads: usize = kani::any();
    let head_dim: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 2);
    kani::assume(seq_len >= 1 && seq_len <= 4);
    kani::assume(num_heads >= 1 && num_heads <= 4);
    kani::assume(head_dim >= 1 && head_dim <= 4);

    let q_len = batch * seq_len * num_heads * head_dim;

    let b: usize = kani::any();
    let s: usize = kani::any();
    let h: usize = kani::any();
    kani::assume(b < batch);
    kani::assume(s < seq_len);
    kani::assume(h < num_heads);

    let q_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
    assert!(q_offset + head_dim <= q_len);
}
```

**Line by line:**

- **Lines 2–5**: Four symbolic dimension parameters. Kani explores all combinations.
- **Lines 7–10**: Constrain to realistic ranges. The solver checks $2 \times 4 \times 4 \times 4 = 128$ dimension combinations, but within each, `b`, `s`, `h` are also symbolic.
- **Line 12**: The buffer size — how much memory was allocated.
- **Lines 14–18**: Symbolic loop indices with bounds derived from dimensions.
- **Line 20**: The offset formula — **exactly the same arithmetic** as `CpuAttention::forward`.
- **Line 21**: The safety property: accessing `q[q_offset..q_offset + head_dim]` never reads past `q_len`.

**What this proves**: For any valid (batch, seq_len, num_heads, head_dim) in the bounded range, and any valid loop index (b, s, h), the Q vector access is within bounds. No out-of-bounds panic is possible.

## Reading the quantization harness

From `kernels/src/quantized.rs`, proving ggml compatibility:

```rust
#[kani::proof]
fn q4k_scale_unpack_matches_ggml() {
    let packed: [u8; 12] = kani::any();
    let j: usize = kani::any();
    kani::assume(j < 8);

    // ggml's get_scale_min_k4
    let (ggml_sc, ggml_m) = if j < 4 { ... } else { ... };
    
    // Our implementation
    let (our_sc, our_m) = if j < 4 { ... } else { ... };

    assert!(ggml_sc == our_sc);
    assert!(ggml_m == our_m);
}
```

This proves that for **any** 12-byte packed scale array and **any** group index 0–7, our scale unpacking produces the same result as ggml's `get_scale_min_k4`. The solver checks all $2^{96}$ possible byte arrays (symbolically, not enumeratively).

## When a harness fails

If Kani finds a counterexample, it reports the concrete input values. For example:

```
VERIFICATION:- FAILED
Counterexample: batch=2, seq_len=3, h=2 → q_offset=26, q_len=24
```

This means the formula produces an offset (26) that exceeds the buffer (24). The fix is clear: the index arithmetic has a bug at these specific dimensions.

## Source code

All harnesses: grep for `#[kani::proof]` across `kernels/src/` and `engine/src/`.

Run all harnesses: `cargo kani --package tempura-kernels`

## See also

- [Bounded model checking](../bounded-model-checking/) — the theory behind Kani
- [Compositionality](../compositionality/) — proving graphs from proven operations
