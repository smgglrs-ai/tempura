+++
title = "Verification Tiers"
description = "Proven, Tested, Compiled — what each tier guarantees and when to use which."
weight = 100
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Why tiers?

Not every kernel implementation needs the same level of assurance. A production GPU kernel running in a datacenter warrants formal proofs. A prototype NPU backend in early development may only need to type-check. Forcing full verification on every implementation would block experimentation; skipping it entirely would undermine the project's thesis.

Verification tiers solve this: every implementation declares its tier, and the tier is recorded in the InferenceTrace. Users see what level of assurance produced each response.

## The three tiers

| Tier | Requirements | What it guarantees |
|------|-------------|-------------------|
| **Proven** | Kani proofs pass + DualPipeline equivalence within $\epsilon$ | No panics for any valid input (formal) + numerical fidelity to the reference (empirical) |
| **Tested** | DualPipeline equivalence within $\epsilon$ | Numerical fidelity to the reference, but safety is not formally proven |
| **Compiled** | Type-checks against kernel traits | The implementation exists and has the right signature, but no equivalence or safety data |

## How tiers are recorded

<!-- TODO: The VerificationTier enum in tempura-kernel-api.
     How KernelPipeline computes min(component tiers).
     How InferenceTrace records the pipeline tier.
     How the API response exposes it. -->

## The tradeoff

<!-- TODO: Proven is the strongest but requires Kani setup and DualPipeline tests.
     Tested is practical for new backends — write a test, compare to reference.
     Compiled is the starting point — implement the trait, iterate until tested.
     The progression: Compiled → Tested → Proven as confidence grows. -->

## Rigor as measurement, not gate

<!-- TODO: The design philosophy. Tiers don't block usage — you can run a
     Compiled kernel in production. But the trace records it, so users
     and operators know what they're running. Rigor is transparent,
     not enforced. This is how you get adoption without compromising
     the verification thesis. -->
