+++
title = "Verification Tiers"
description = "What each tier guarantees, when to use which, proof strength vs development velocity."
weight = 100
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Three tiers

Every kernel implementation declares a `VerificationTier`:

| Tier | Meaning | Requirements |
|------|---------|-------------|
| **Proven** | Kani proofs pass + DualPipeline equivalence | Safety proofs + numerical match to reference |
| **Tested** | DualPipeline equivalence only | Numerical match to reference, no formal proofs |
| **Compiled** | Type-checks against trait | Compiles, no equivalence data |

The `KernelPipeline` computes `min(component tiers)` — the pipeline is only as strong as its weakest kernel. A pipeline with one `Compiled` kernel is tier `Compiled` regardless of how many `Proven` kernels it contains.

## What each tier guarantees

### Compiled

The implementation satisfies the Rust type system: correct function signatures, correct data types. This guarantees nothing about correctness — a function that returns all zeros is `Compiled`.

**When it's enough**: Stub implementations, work-in-progress kernels, NPU backends waiting for hardware.

### Tested

The implementation produces output numerically equivalent to the CPU reference (within epsilon) across randomized test configurations. This catches algorithmic bugs, precision issues, and format errors.

**When it's enough**: GPU kernels validated against the proven CPU reference. The CPU reference acts as the oracle — if the GPU matches it, the GPU is correct modulo the CPU's correctness (which is separately validated by proptest + f64 reference).

### Proven

Kani proves safety properties (no panics, no OOB) + DualPipeline confirms numerical equivalence. This is the strongest guarantee: the implementation cannot crash AND it produces correct output.

**When it's enough**: Production kernels that must not fail. The CPU reference implementations are all `Proven`.

## Tier in the API

Every inference response includes the pipeline's verification tier in tempura-specific metadata:

```json
{
  "choices": [...],
  "tempura": {
    "verification_tier": "proven",
    "trace_id": "abc123"
  }
}
```

Users can query `GET /tempura/v1/verified` for a per-kernel breakdown:

```json
{
  "kernels": [
    {"kernel": "CpuAttention", "status": "proven", "properties": ["Q indexing within bounds", ...]},
    {"kernel": "CudaMatMul", "status": "tested", "properties": [...]}
  ]
}
```

## The tradeoff

Proven tiers require writing Kani harnesses and ensuring they pass — this takes hours per kernel. Tested tier requires only DualPipeline comparison — minutes of developer time. Compiled is instant.

For a new backend (e.g., NPU), the practical path is:
1. Implement traits → **Compiled** (can test wiring)
2. Run DualPipeline against CPU reference → **Tested** (production-ready)
3. Write Kani harnesses → **Proven** (highest assurance)

Most users stop at Tested. Proven is for safety-critical deployments and the CPU reference itself.

## Source code

- [`kernel-api/src/lib.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernel-api/src/lib.rs) — `VerificationTier` enum
- [`engine/src/pipeline.rs`](https://github.com/smgglrs-ai/tempura/blob/main/engine/src/pipeline.rs) — `KernelPipeline` tier aggregation
- [`server/src/api.rs`](https://github.com/smgglrs-ai/tempura/blob/main/server/src/api.rs) — `GET /tempura/v1/verified` endpoint

## See also

- [Bounded model checking](../bounded-model-checking/) — what Proven tier's Kani proofs cover
- [DualPipeline](../dual-pipeline/) — how Tested tier is achieved
- [Compositionality](../compositionality/) — proving graphs from proven components
