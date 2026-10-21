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
| **Proven** | Kani + Verus proofs pass + DualPipeline equivalence within epsilon | Formal safety + unbounded correctness + numerical fidelity |
| **Tested** | DualPipeline equivalence only | No formal proof, but numerically matches reference |
| **Compiled** | Type-checks against traits | No equivalence data; may diverge numerically |

## Kani proofs

[Kani](https://model-checking.github.io/kani/) is a model checker for Rust that proves properties hold for *all possible inputs* within bounded ranges. tempura uses Kani to prove:

- **No panics** in any kernel implementation for valid inputs
- **No out-of-bounds access** in slice indexing arithmetic
- **Slot accounting safety** in KV cache (no double-alloc, no OOB)
- **Batch isolation** — attention indexing for one sequence cannot read another sequence's memory

61 proof harnesses across `kernels/` and `engine/`, covering attention, RMSNorm, RoPE, matmul, sampling, quantization (Q4K/Q6K/TQ2/I2S), KV cache, speculative decoding, and more.

```sh
cargo kani --package tempura-kernels    # 61 proof harnesses
```

## Verus proofs

[Verus](https://github.com/verus-lang/verus) is a deductive verifier for Rust that proves properties hold for *all inputs* (unbounded) via SMT. Where Kani checks bounded instances, Verus gives mathematical certainty.

tempura uses Verus to prove:

- **Struct invariants** — `KvCache`, `PageAllocator`, `BlockTable`, `Session`, `PrefixCache`
- **Loop termination** — all iterative algorithms provably terminate
- **State machine transitions** — prefill → decode lifecycle, KV cache append/advance
- **GQA head mapping** — group query attention index arithmetic is in-bounds for all configs
- **Tensor parallel sharding** — slice assignments cover the full output with no gaps or overlaps
- **Causal masking** — attention mask is strictly lower-triangular for all sequence lengths
- **BitNet packing roundtrips** — TQ2_0 and I2S pack/unpack are mathematical inverses

828 verification conditions across 62 modules in the standalone `verus-verified/` crate.

```sh
./scripts/run-verus.sh    # 828 verification conditions across 62 modules
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
