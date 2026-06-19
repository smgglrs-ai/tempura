+++
title = "Compositionality"
description = "Why proving individual operations proves the graph — and the assumptions that make this work."
weight = 80
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## The composition question

tempura proves each kernel operation safe individually: attention, RMSNorm, RoPE, MatMul. But a transformer layer chains them: norm → QKV projection → RoPE → attention → output projection → residual → norm → FFN → residual. Does proving each operation individually prove the chain?

Yes — under specific assumptions. This page explains the argument and its boundaries.

## The dimensional compatibility invariant

The key property: if the output shape of operation $f$ matches the expected input shape of operation $g$, and both $f$ and $g$ are individually proven safe for their input shapes, then $g(f(x))$ is safe for any valid $x$.

$$
\forall x \in \text{valid}(f): \quad f(x) \in \text{valid}(g) \implies g(f(x)) \text{ is safe}
$$

This reduces graph-level safety to two things:
1. Each operation is individually safe (Kani proves this)
2. The wiring preserves dimensional compatibility (the engine's type system enforces this)

## The formal argument

<!-- TODO: Step through the actual transformer layer composition.
     Show how the output dimensions of RMSNorm (d_model) match the input
     dimensions of the QKV projection (d_model → 3 * d_model).
     Show how the engine's KernelPipeline enforces this at compile time
     via Rust's type system. The argument that compile-time + per-op Kani
     = graph-level safety. -->

## What this does NOT prove

<!-- TODO: Numerical accuracy of the composition (DualPipeline).
     Correctness of the math (the graph computes attention, not just
     "safe array operations"). The distinction between safety (no crashes)
     and correctness (right answer). Why DualPipeline fills the gap. -->

## The NPU extension

<!-- TODO: Graph-level traits (Phase 17) let NPU backends implement entire
     layers as compiled blobs. The composition argument still works:
     DualPipeline compares the fused output against the composition of
     individual operations. You lose per-operation observability inside
     the fused block, but the I/O contract is verified. -->
