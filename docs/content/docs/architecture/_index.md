+++
title = "Architecture"
description = "Three-crate workspace, kernel traits, and the verification contract."
weight = 20
template = "docs/section.html"

[extra]
toc = true
+++

## Design principles

1. **Kernel traits as the verification contract.** Engine code is generic over kernel implementations. The same trait is implemented by CPU (for verification) and GPU (for production). Kani proves both adhere to the contract.

2. **Three-layer verification.** Kani proves safety (no panics, no OOB). DualPipeline validates numerical equivalence. Property tests cover mathematical properties Kani can't handle.

3. **Inference traces are first-class.** Every forward pass produces a structured `InferenceTrace` at configurable granularity. Traces are queryable, diffable, and replayable.

4. **Deterministic by design.** Seeded RNG, controlled reduction order. Same input + seed = bit-identical output.

## Crate ecosystem

### tempura-kernel-api

The stable contract third parties code against. Contains all kernel traits and shared types, zero implementations.

**Traits:** `Attention`, `RMSNorm`, `RoPE`, `MatMul`, `Sampler`, `LayerNorm`, `AllReduce`, `KvCacheCompressor`, `GuidedDecoder`

### tempura-kernels

First-party implementations: CPU reference (always available) and CUDA via cuda-oxide (behind `cuda` feature flag).

### tempura-engine

Inference engine generic over kernel traits. Contains `TransformerModel`, `KernelPipeline`, `InferenceSession`, `InferenceTrace`, `DualPipeline`, weight loading, tokenizer integration.

### tempura-serve

HTTP serving layer. Axum-based, OpenAI-compatible API. Continuous batching, speculative decoding, prefix caching, Prometheus metrics, OpenTelemetry export.

## KernelPipeline

The `KernelPipeline<A, N, R, M, S>` bundles kernel implementations into a composable unit:

```rust
pub struct KernelPipeline<A, N, R, M, S> {
    pub attention: A,
    pub rmsnorm: N,
    pub rope: R,
    pub matmul: M,
    pub sampler: S,
}
```

Factory methods: `KernelPipeline::cpu_verified()` for testing, `KernelPipeline::cuda()` for GPU inference. Third parties compose their own pipelines by mixing implementations.
