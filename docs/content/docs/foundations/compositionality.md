+++
title = "Compositionality"
description = "If each op is safe and the wiring preserves dimensions, the graph is safe."
weight = 80
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## The compositionality argument

A transformer layer is a sequence of operations: RMSNorm → QKV projection → RoPE → Attention → Output projection → Residual → RMSNorm → Gate/Up projection → SiLU×Gate → Down projection → Residual.

If each operation is individually proven safe (no panics, no OOB access), and the output buffer of operation $N$ has the correct size for the input of operation $N+1$, then the composed sequence is safe.

Formally: given operations $f_1, f_2, \ldots, f_n$ where each $f_i: \mathbb{R}^{d_\text{in}^i} \to \mathbb{R}^{d_\text{out}^i}$ is proven panic-free, and $d_\text{out}^i = d_\text{in}^{i+1}$ for all $i$, then $f_n \circ \cdots \circ f_1$ is panic-free.

## Dimensional compatibility

The `CompositionVerifier` in `kernel-verify/src/composition.rs` checks that output dimensions match input dimensions at every boundary. For a LLaMA-7B layer:

| Operation | Input elements | Output elements |
|-----------|---------------|-----------------|
| RMSNorm(input) | $s \times 4096$ | $s \times 4096$ |
| MatMul(Q proj) | $s \times 4096$ | $s \times 4096$ |
| RoPE(Q) | $s \times 4096$ | $s \times 4096$ |
| Attention | $s \times 4096$ | $s \times 4096$ |
| MatMul(O proj) | $s \times 4096$ | $s \times 4096$ |
| Residual | $s \times 4096$ | $s \times 4096$ |
| RMSNorm(post) | $s \times 4096$ | $s \times 4096$ |
| MatMul(gate) | $s \times 4096$ | $s \times 11008$ |
| SiLU×gate | $s \times 11008$ | $s \times 11008$ |
| MatMul(down) | $s \times 11008$ | $s \times 4096$ |
| Residual(FFN) | $s \times 4096$ | $s \times 4096$ |

Every boundary is dimensionally compatible. The verifier checks this programmatically for any model configuration.

## What composition does NOT prove

- **Numerical adequacy**: Operations compose safely, but accumulated floating-point error may produce wrong answers. DualPipeline catches this empirically.
- **Cross-operation invariants**: Composition verifies buffer sizes, not value ranges. A kernel that outputs values outside the expected range won't trigger a dimension error but may cause downstream issues (e.g., softmax overflow).

## The bridge to NPU verification

NPU backends implement graph-level traits (`FusedTransformerLayer`) that fuse multiple operations. The compositionality framework proves the unfused version safe; DualPipeline verifies the fused version matches numerically. This two-level approach avoids re-proving each fused implementation while catching regressions.

## Source code

- [`kernel-verify/src/composition.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernel-verify/src/composition.rs) — `CompositionVerifier` with `verify_transformer_layer`
- [`kernels/src/verification_manifest.rs`](https://github.com/smgglrs-ai/tempura/blob/main/kernels/src/verification_manifest.rs) — composition proof entry

## See also

- [Bounded model checking](../bounded-model-checking/) — how individual operations are proven
- [DualPipeline](../dual-pipeline/) — how numerical equivalence is verified
