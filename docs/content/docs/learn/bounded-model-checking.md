+++
title = "20. Bounded Model Checking"
description = "What Kani proves, how SAT/SMT works under the hood, and what 'all inputs' means within bounds."
weight = 200
template = "docs/page.html"

[extra]
part = "verification"
toc = true
math = true
+++

## Testing vs proving

A test checks one input. A thousand tests check a thousand inputs. A fuzzer explores random inputs for hours and checks millions. But none of them can say "this property holds for *every possible input*."

Bounded model checking can. It encodes the program as a logical formula and asks a SAT solver whether any input — any at all — can violate the property. If the solver says no, the property holds for all inputs within the declared bounds.

This is not a probabilistic statement. It's a proof.

## How Kani works

[Kani](https://model-checking.github.io/kani/) is a model checker for Rust, built on [CBMC](https://www.cmu.edu/scs/s3d/mchk/cbmc-user-manual/). The pipeline:

1. **Compile** Rust code through the standard compiler frontend, producing an intermediate representation.
2. **Unroll** all loops up to a configurable bound. A `for i in 0..n` loop becomes $n$ sequential copies of the loop body.
3. **Encode** the entire unrolled program as a Boolean satisfiability (SAT) problem. Each variable becomes a set of Boolean variables (one per bit). Each operation becomes a Boolean circuit.
4. **Solve**. A SAT solver (typically MiniSat or CaDiCaL) searches for an assignment of inputs that makes an assertion false.
5. **Report** either "verification successful" (no counterexample exists) or a concrete counterexample with the exact input values that triggered the violation.

## Symbolic execution vs exhaustive testing

Kani does not run the program on every possible input. That would take $2^{32}$ executions for a single `u32` parameter — about 4 billion. Instead, it reasons about all inputs *simultaneously* by treating input variables as symbolic.

Consider a simple function:

```rust
fn index(data: &[f32], i: usize, len: usize) -> f32 {
    data[i * len]
}
```

A test would check specific values of `i` and `len`. Kani introduces symbolic variables `i` and `len` that represent *any* value, then asks: "is there any `i` and `len` such that `i * len >= data.len()`?" The SAT solver either proves this is impossible (given the constraints) or finds a counterexample.

## What "all inputs" means

"All inputs" means all inputs *within the declared bounds*. Kani cannot reason about unbounded programs — loops must terminate, arrays must have fixed maximum sizes.

In a tempura Kani harness, you'll see:

```rust
#[kani::proof]
fn verify_offset_safety() {
    let batch: usize = kani::any();
    let seq_len: usize = kani::any();
    kani::assume(batch <= 4);
    kani::assume(seq_len <= 8);
    // ...
}
```

`kani::any()` produces a symbolic value — it represents every possible `usize`. `kani::assume()` constrains it: "only consider values where `batch <= 4`." The proof then holds for all $(batch, seq\_len) \in [0, 4] \times [0, 8]$.

**Are the bounds large enough?** This is a design decision. Larger bounds give stronger proofs but take exponentially longer to solve. tempura's harnesses use bounds that cover the parameter space of practical models (typically `seq_len <= 8`, `head_dim <= 8`, `num_heads <= 4`) — small enough for the solver but large enough to exercise all code paths.

**What about values beyond the bounds?** The proof does not cover them. If a model has 128 attention heads and the proof only covers up to 4, a bug specific to head index 127 would not be found. However, the index arithmetic in tempura's kernels is *linear* in these parameters — the same multiplication and addition patterns that are safe at 4 heads are safe at 128. Non-linear edge cases (overflow, for example) are checked with bounds large enough to trigger them.

## What Kani can prove

- **No panics.** Array indexing, integer arithmetic, unwrap/expect — any operation that can panic is checked.
- **No out-of-bounds access.** Slice indexing is bounds-checked by Rust, and Kani verifies the bounds check always passes.
- **Arithmetic invariants.** Assert statements in the code are verified for all inputs.
- **No integer overflow.** Multiplication and addition that could overflow `usize` are checked.

## What Kani cannot prove

- **Floating-point properties.** Kani's reasoning about f32/f64 is limited. Properties like "softmax output sums to 1.0" involve transcendental functions (exp) that cannot be precisely encoded in SAT. Use property-based tests for these.
- **Performance.** Kani verifies functional properties, not timing.
- **Correctness.** Kani proves the implementation is *safe* (no crashes, no OOB), not that it's *correct* (computes the right mathematical function). The distinction between "this attention kernel never panics" and "this attention kernel computes attention" is fundamental. DualPipeline bridges this gap.
- **Behavior beyond bounds.** As discussed above — the proof covers the declared range.

## When to use Kani vs other verification tools

| Tool | What it checks | Strength | Weakness |
|------|---------------|----------|----------|
| **Kani** | All inputs within bounds | Exhaustive proof | Bounded, no f32 reasoning |
| **Property tests** (proptest) | Random inputs | Good at f32, covers edge cases | Probabilistic, not exhaustive |
| **DualPipeline** | Specific inputs, two implementations | Catches numerical bugs | Only compares specific inputs |
| **Type system** | Compile-time shape constraints | Zero runtime cost | Limited to what types can express |

tempura uses all four. They are complementary, not competing.

## SAT under the hood

For the curious: how does program verification reduce to Boolean satisfiability?

**Variables.** A `usize` on a 64-bit system is 64 bits. It becomes 64 Boolean variables: $x_0, x_1, \ldots, x_{63}$.

**Addition.** $a + b = c$ becomes a circuit of Boolean gates (XOR for sum, AND/OR for carry), producing 64 Boolean equations.

**Comparison.** $a < b$ becomes a lexicographic comparison circuit on the bit representations.

**An assertion** `assert!(i < len)` becomes: "the negation of $i < len$ is satisfiable?" If the SAT solver finds a satisfying assignment, that assignment is the counterexample — the specific values of `i` and `len` that violate the assertion.

The entire program (after loop unrolling) becomes one large Boolean formula. Modern SAT solvers handle formulas with millions of variables, which is why Kani works on real code — kernel functions with bounded loops and array operations produce formulas that solvers can handle in seconds to minutes.

## Further reading

- [Kani documentation](https://model-checking.github.io/kani/)
- [CBMC: Bounded Model Checking for ANSI-C](https://www.cmu.edu/scs/s3d/mchk/cbmc-user-manual/)
- Clarke, Biere, Raimi, Zhu (2001). "Bounded Model Checking Using Satisfiability Solving." *Formal Methods in System Design*.
- The [Reading Kani Harnesses](../reading-kani-harnesses/) page walks through a real tempura harness line by line.
