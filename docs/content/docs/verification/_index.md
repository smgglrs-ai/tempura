+++
title = "Verification"
description = "Tiered verification, DualPipeline, Kani proofs, and composition guarantees."
weight = 30
template = "docs/section.html"

[extra]
toc = true
+++

## Verification tiers

Every kernel implementation declares its verification tier. The tier is recorded in every `InferenceTrace`, so users see exactly what level of assurance produced each response.

| Tier | Requirements | What it means |
|------|-------------|---------------|
| **Proven** | Kani proofs pass + DualPipeline equivalence within epsilon | Formal safety + numerical fidelity |
| **Tested** | DualPipeline equivalence only | No formal proof, but numerically matches reference |
| **Compiled** | Type-checks against traits | No equivalence data; may diverge numerically |

## Kani proofs

[Kani](https://model-checking.github.io/kani/) is a model checker for Rust that proves properties hold for *all possible inputs* within bounded ranges. tempura uses Kani to prove:

- **No panics** in any kernel implementation for valid inputs
- **No out-of-bounds access** in slice indexing arithmetic
- **Slot accounting safety** in KV cache (no double-alloc, no OOB)
- **Batch isolation** — attention indexing for one sequence cannot read another sequence's memory

```sh
cargo kani --package tempura-kernels    # 11 proof harnesses
```

## DualPipeline

Run two kernel implementations on identical inputs and compare outputs:

```
POST /tempura/v1/compare
{
  "token_ids": [1, 2, 3],
  "seed": 42
}
```

Returns per-layer divergence metrics (max absolute diff, L2 distance) and whether the final sampled tokens match. Use this to validate a new kernel implementation against the CPU reference.

## Composition proofs

If each operation is individually proven safe, and the wiring between operations preserves dimensional invariants, the composed graph is proven safe. This is the theoretical foundation for NPU graph-level traits: prove the unfused operations individually, verify the fused output matches via DualPipeline.
