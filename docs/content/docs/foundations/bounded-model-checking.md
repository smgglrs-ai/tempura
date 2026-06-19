+++
title = "Bounded Model Checking"
description = "What Kani proves, how SAT/SMT works under the hood, and what 'all inputs' means within bounds."
weight = 60
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Testing vs proving

A test checks one input. A thousand tests check a thousand inputs. Bounded model checking checks *every possible input* within a declared range — not by running them all, but by encoding the program as a logical formula and asking a SAT solver whether any input can violate the property.

If the solver finds no counterexample, the property holds for all inputs in the range. This is a proof, not a sample.

## How Kani works

[Kani](https://model-checking.github.io/kani/) is a model checker for Rust. It:

1. **Compiles** Rust code to an intermediate representation (via CBMC)
2. **Unrolls** loops up to a configurable bound
3. **Encodes** the entire program as a Boolean satisfiability (SAT) problem
4. **Asks** a SAT solver: "is there any input that causes a panic, out-of-bounds access, or assertion violation?"
5. **Reports** either "no counterexample found" (proof) or a concrete counterexample

## What "all inputs" means

<!-- TODO: The bounded nature — loop unrolling depth, integer range limits,
     what kani::any() generates. Why bounds must be chosen carefully.
     The relationship between bound size and solver time.
     What happens when bounds are too small (missed bugs) or too large
     (solver timeout). -->

## What Kani can and cannot prove

<!-- TODO: CAN prove: no panics, no OOB, arithmetic invariants, assert! conditions.
     CANNOT prove: floating-point properties (exp, sin, etc.), performance,
     correctness of the math (only safety of the implementation),
     behavior beyond the unrolling bound. The complementary tools:
     property tests for f32, DualPipeline for numerical equivalence. -->

## SAT/SMT under the hood

<!-- TODO: High-level intuition of how program verification reduces to SAT.
     Why this is tractable for the kind of code kernels contain (bounded loops,
     array indexing, integer arithmetic) but not for arbitrary programs.
     No need to understand SAT solving to use Kani — this section is for
     readers who want to understand WHY it works, not just THAT it works. -->

## Further reading

- [Kani documentation](https://model-checking.github.io/kani/)
- [CBMC: Bounded Model Checking for ANSI-C](https://www.cs.cmu.edu/~modelcheck/cbmc/)
- Clarke, Kroening, Lerda (2004). "A Tool for Checking ANSI-C Programs"
