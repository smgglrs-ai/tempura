+++
title = "Reading Kani Harnesses"
description = "Annotated walkthrough of a real Kani proof harness from the tempura codebase."
weight = 70
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## What is a harness?

A Kani harness is a Rust function annotated with `#[kani::proof]` that describes the property to verify. It declares symbolic inputs (values that represent "any possible value"), calls the code under test, and asserts the property that should hold for all inputs.

## Anatomy of a harness

A typical tempura harness has three parts:

1. **Input declaration** — `kani::any()` generates symbolic values. `kani::assume()` constrains them to valid ranges.
2. **Operation** — call the function being verified with the symbolic inputs.
3. **Assertion** — `assert!()` the property that must hold for all valid inputs.

## Walkthrough: attention offset safety

<!-- TODO: Use the actual attention Kani harness from kernels/src/attention.rs.
     Annotate each line explaining:
     - What the symbolic inputs represent (batch, seq_len, num_heads, head_dim)
     - What the assume() constraints mean (why these bounds)
     - What the assert! proves (Q/KV offset never exceeds buffer length)
     - What this means for the math (attention reads exactly the right memory)
     Show the harness code with inline annotations. -->

## How to read a harness in a PR

<!-- TODO: Practical guide for reviewers:
     1. Read the assert! first — that's the property
     2. Read the assume() — those are the valid input constraints
     3. Check: does the property still hold after the code change?
     4. Check: are the assumptions still valid? (common mistake: tightening
        assumptions to make a proof pass, which weakens what's proven)
     5. If the harness changed, ask: was the old property wrong, or is the
        new code weaker? -->

## Common patterns

<!-- TODO: The offset-bounds pattern (attention, matmul).
     The capacity-accounting pattern (KvCache).
     The index-isolation pattern (batch invariant).
     How to recognize which pattern a harness uses. -->
