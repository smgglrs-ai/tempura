+++
title = "28. Verification Tiers"
description = "Proven, Tested, Compiled — what each tier guarantees and when to use which."
weight = 280
template = "docs/page.html"

[extra]
part = "verification"
toc = true
math = true
+++

## Why tiers?

Not every kernel implementation needs the same level of assurance. A production GPU kernel running in a datacenter warrants formal proofs. A prototype NPU backend in early development may only need to type-check. Forcing full verification on every implementation blocks experimentation; skipping it entirely undermines the project's thesis.

Verification tiers solve this by making assurance *explicit*: every implementation declares its tier, and the tier is recorded in the [InferenceTrace](../../trace-tooling/). Users and operators see what level of verification produced each response.

## The three tiers

### Proven

**Requirements:** [Kani](../bounded-model-checking/) proofs pass + [DualPipeline](../dual-pipeline/) equivalence within epsilon.

**What it guarantees:**
- No panics, no out-of-bounds access for all valid inputs within bounded ranges (formal proof)
- Numerical output matches the CPU reference within epsilon (empirical equivalence)

**When to use:** production kernels — the CPU reference implementations and the CUDA kernels.

### Tested

**Requirements:** DualPipeline equivalence within epsilon (no Kani proofs).

**What it guarantees:**
- Numerical output matches the CPU reference (empirical equivalence)
- Safety is demonstrated by testing, not proven

**When to use:** new backends where Kani setup isn't complete yet, third-party kernel implementations, NPU backends operating through graph-level traits.

### Compiled

**Requirements:** type-checks against kernel traits (no Kani, no DualPipeline).

**What it guarantees:**
- The implementation exists and has the correct function signature
- Nothing about safety or numerical equivalence

**When to use:** early prototypes, stubs, backends under active development.

## The progression

Tiers represent a natural development progression:

$$
\text{Compiled} \xrightarrow{\text{add DualPipeline tests}} \text{Tested} \xrightarrow{\text{add Kani proofs}} \text{Proven}
$$

1. **Start Compiled.** Implement the trait, get it to compile. The type system catches shape mismatches.
2. **Advance to Tested.** Run DualPipeline against the CPU reference. Fix numerical bugs until the output matches within epsilon.
3. **Advance to Proven.** Write Kani harnesses for the safety properties. Fix any issues the solver finds. Publish with the strongest guarantee.

The verification toolkit (tempura-kernel-verify, Phase 12) makes each step easy: `cargo tempura verify --kernel my-attention` runs the full DualPipeline + Kani suite and reports the achieved tier.

## How tiers are recorded

Each kernel trait includes a method for declaring the tier:

```rust
pub trait Attention: Send + Sync {
    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled  // default — override if you've verified
    }
    
    fn forward(&self, /* ... */) -> Result<()>;
}
```

The `KernelPipeline` computes the composite tier as the minimum of its components:

$$
\text{tier}_{\text{pipeline}} = \min(\text{tier}_A, \text{tier}_N, \text{tier}_R, \text{tier}_M, \text{tier}_S)
$$

If four kernels are Proven and one is Tested, the pipeline is Tested. The weakest link determines the overall guarantee.

This composite tier is recorded in every `InferenceTrace` and exposed in the API response metadata.

## Rigor as measurement, not gate

A deliberate design choice: tiers don't block usage. You can run a Compiled kernel in production. But the InferenceTrace records it, so:

- **Operators** see "this response was produced by a Tested pipeline" and can route high-stakes requests to Proven instances.
- **Developers** see "this kernel is Compiled" and know what work remains before it can be promoted.
- **Auditors** see the verification history across deployments and can assess risk.

The alternative — requiring Proven for all deployments — would make tempura impractical for experimentation and early-stage backends. Making rigor visible instead of enforced gets adoption without compromising transparency.

## Tier comparison across implementations

DualPipeline enables a specific workflow: compare a new kernel implementation against the reference, and the result determines the tier:

| DualPipeline result | Kani result | Assigned tier |
|--------------------|-------------|--------------|
| Not run | Not run | Compiled |
| Passes within epsilon | Not run | Tested |
| Passes within epsilon | All harnesses pass | Proven |
| Fails (divergence > epsilon) | — | Cannot advance past Compiled |
| — | Harness failure (counterexample found) | Cannot advance past Tested |

If DualPipeline fails, the implementation has a numerical bug — fix it before claiming any tier above Compiled. If a Kani harness finds a counterexample, the implementation has a safety bug — fix it before claiming Proven.
