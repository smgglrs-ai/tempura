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

tempura proves each kernel operation safe individually: [attention](../attention/), [RMSNorm](../normalization/), [RoPE](../positional-encoding/), MatMul. But a transformer layer chains them together:

$$
\text{RMSNorm} \to W_Q, W_K, W_V \to \text{RoPE} \to \text{Attention} \to W_O \to \text{Residual} \to \text{RMSNorm} \to \text{FFN} \to \text{Residual}
$$

Does proving each box individually prove the chain?

**Yes** — under a specific assumption that tempura's architecture enforces. This page explains the argument and its boundaries.

## The dimensional compatibility argument

The key property: if the output of operation $f$ has the right shape and bounds for the input of operation $g$, and both $f$ and $g$ are individually proven safe for their respective inputs, then $g \circ f$ is safe.

More precisely, define:

- $\text{Safe}(f)$: for all valid inputs $x$, $f(x)$ terminates without panics or out-of-bounds access
- $\text{Valid}(f)$: the set of inputs for which $f$ is safe
- $\text{Range}(f)$: the set of possible outputs of $f$

The composition is safe when:

$$
\text{Safe}(f) \land \text{Safe}(g) \land \text{Range}(f) \subseteq \text{Valid}(g) \implies \text{Safe}(g \circ f)
$$

The first two conditions are proven by Kani per-operation. The third — that the output of one operation is valid input for the next — is enforced by tempura's type system and the `KernelPipeline` wiring.

## How tempura enforces compatibility

The `KernelPipeline` and the forward pass in `TransformerModel` hardcode the wiring between operations. The key dimensional invariants:

**RMSNorm output shape = input shape.** RMSNorm normalizes each token's hidden vector in-place (conceptually — the implementation uses separate input/output buffers of the same size). If the input is $(\text{batch} \times \text{seq\_len}) \times d_{\text{model}}$, the output is the same.

**MatMul output = determined by weight shape.** $Q = X W_Q$ where $X \in \mathbb{R}^{n \times d_{\text{model}}}$ and $W_Q \in \mathbb{R}^{d_{\text{model}} \times d_{\text{out}}}$. The output shape $n \times d_{\text{out}}$ is determined by the weight matrix, which is loaded from the model file with known dimensions.

**RoPE preserves shape.** RoPE rotates pairs of elements in-place — the output has the same shape as the input.

**Attention output shape = determined by parameters.** Given Q, K, V with compatible dimensions (checked at the trait boundary), the output shape is $n \times d_v$.

Each connection is a matrix multiplication or an in-place operation — the type system ensures shapes are consistent, and the Kani proofs for each operation assume shapes that match the wiring.

## What the type system gives us

Rust's type system enforces some of these invariants at compile time:

```rust
pub struct KernelPipeline<A, N, R, M, S> {
    pub attention: A,    // A: Attention
    pub rmsnorm: N,      // N: RMSNorm
    pub rope: R,         // R: RoPE
    pub matmul: M,       // M: MatMul
    pub sampler: S,      // S: Sampler
}
```

The generic parameters ensure that only types implementing the correct trait can be used. The trait signatures define the input/output buffer relationships:

```rust
pub trait Attention: Send + Sync {
    fn forward(
        &self,
        q: &[f32], k: &[f32], v: &[f32],
        output: &mut [f32],
        batch_size: usize, seq_len: usize, kv_len: usize,
        num_heads: usize, num_kv_heads: usize,
        head_dim: usize, scale: f32,
    ) -> Result<()>;
}
```

The buffer sizes must match the dimensional parameters — this is what the Kani proofs verify.

## The argument, stated precisely

1. **Kani proves** each kernel operation is safe for all valid inputs within bounds.
2. **The forward pass** connects operations with buffers whose sizes are computed from the `ModelConfig` — a fixed set of dimensions loaded from the model.
3. **The output buffer of operation $i$** is allocated with the same dimension arithmetic used as the input buffer of operation $i+1$.
4. **Therefore**, the input to operation $i+1$ is always in the valid set that Kani proved safe.
5. **Therefore**, the composed forward pass is safe.

Step 3 is the critical assumption: the buffer allocation in the forward pass must use exactly the dimensional arithmetic that the Kani proofs bound. This is verified by code inspection and by running DualPipeline on complete forward passes (which exercises every connection).

## What this does NOT prove

**Numerical correctness.** The composition argument proves safety (no crashes), not correctness (right answer). Composing a safe-but-wrong attention with a safe-but-wrong RMSNorm produces a safe-but-wrong forward pass. DualPipeline catches this by comparing the composed output against a reference.

**Correctness of the math.** "The graph computes attention followed by normalization" is a statement about the mathematical semantics, not about array indexing. Compositionality proves "the arrays don't overflow," not "the computation implements the transformer architecture."

**Beyond the bounds.** If individual proofs cover `seq_len <= 8` and the composed forward pass runs with `seq_len = 2048`, the composition argument holds only if the index arithmetic is the same form at both sizes. In practice it is (linear indexing), but this is an assumption, not a proof.

## The NPU extension

[Graph-level traits](../../framework/) (Phase 17) let NPU backends implement entire layers as compiled blobs. The composition argument adapts:

- The NPU implements `FusedTransformerLayer`, which takes full layer inputs and produces full layer outputs.
- DualPipeline compares the fused output against the composition of individual CPU operations.
- You lose per-operation trace granularity inside the fused block, but the I/O contract is verified: the composed output of individually-proven CPU operations matches the NPU's fused output within epsilon.

This is the "compilation preserves semantics" argument from programming language theory — applied to kernel fusion rather than code optimization.
