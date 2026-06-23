+++
title = "Bounded Model Checking"
description = "What Kani does, SAT/SMT under the hood, what 'all inputs' means within bounds."
weight = 60
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## What Kani proves

[Kani](https://model-checking.github.io/kani/) is a bounded model checker for Rust. It proves that a function **cannot panic** for any input within specified bounds — no out-of-bounds access, no integer overflow, no unwrap on None, no assertion failure.

Unlike testing (which checks specific inputs), Kani checks **all** inputs within the bounded space. If it finds no violation, the property is proven for that space. If it finds one, it provides a concrete counterexample.

## How it works: SAT/SMT

Kani translates Rust code (via LLVM IR) into a logical formula and hands it to [CBMC](https://www.cmedsoftware.com/cbmc/), which uses SAT/SMT solving:

1. **Unroll** loops up to a bound (specified by `#[kani::unwind(N)]`)
2. **Symbolize** inputs: each `kani::any()` becomes a symbolic variable
3. **Encode** all paths through the function as a Boolean formula
4. **Ask the SAT solver**: is there an assignment to the symbolic variables that makes an assertion fail?

If the solver says "unsatisfiable" — no such assignment exists — the property holds for all inputs within the unwinding bound.

## The bounded nature

"All inputs" means all inputs within `kani::assume` constraints. A harness that assumes `num_heads >= 1 && num_heads <= 4` proves the property for heads 1 through 4, not for 1000 heads. The bounds are chosen to balance proof strength vs solver runtime.

Tempura harnesses use realistic bounds:
- Batch: 1–2, Seq length: 1–4, Heads: 1–4, Head dim: 1–8
- These are small but exercise all index arithmetic paths

The solver runtime grows exponentially with loop unwinds, so larger bounds are impractical. The key insight: **if the index arithmetic is wrong, it's wrong at small sizes too** — off-by-one errors don't appear only at size 1000.

## What Kani does NOT prove

- **Mathematical correctness**: Kani cannot verify that softmax outputs sum to 1.0. It over-approximates floating-point operations (exp, cos, sqrt). Mathematical properties are tested via proptest and f64 reference comparison.
- **Unbounded inputs**: Properties hold within the assumed bounds only.
- **Liveness**: Kani proves safety (no panic), not termination or progress.
- **Performance**: A proven kernel may be slow. Kani says nothing about efficiency.

## Source code

All Kani harnesses are co-located in `#[cfg(kani)] mod kani_proofs` blocks:

- [`kernels/src/attention.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/attention.rs) — Q/KV offset bounds, batch isolation
- [`kernels/src/rmsnorm.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/rmsnorm.rs) — row indexing, full execution no-panic
- [`kernels/src/sampling.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/sampling.rs) — top-k bounds, fallback token validity
- [`kernels/src/quantized.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/quantized.rs) — Q4K/Q6K block access, dequant bounds, monotonicity, accumulator overflow, ggml equivalence
- [`engine/src/kv_cache.rs`](https://github.com/smgglrs-ai/tempura/blob/main/engine/src/kv_cache.rs) — append/slice bounds, paged block table offsets

## See also

- [Reading Kani harnesses](../reading-kani-harnesses/) — annotated walkthrough of a real harness
- [Verification tiers](../verification-tiers/) — what each tier guarantees
