+++
title = "Framework Guide"
description = "Implement custom kernels and use the verification toolkit."
weight = 50
template = "docs/section.html"

[extra]
toc = true
+++

## Implementing a custom kernel

Third-party kernel implementations depend on `tempura-kernel-api` only:

```toml
[dependencies]
tempura-kernel-api = "1"
```

Implement the trait for your hardware:

```rust
use tempura_kernel_api::{Attention, Result};

pub struct MyNpuAttention { /* ... */ }

impl Attention for MyNpuAttention {
    fn forward(
        &self,
        q: &[f32], k: &[f32], v: &[f32],
        output: &mut [f32],
        batch: usize, seq_len: usize, kv_len: usize,
        num_heads: usize, num_kv_heads: usize,
        head_dim: usize, scale: f32,
    ) -> Result<()> {
        // Your implementation here
        todo!()
    }
}
```

## Verifying your implementation

Use `tempura-kernel-verify` to test against the CPU reference:

```toml
[dev-dependencies]
tempura-kernel-verify = "1"
```

```rust
use tempura_kernel_verify::DualPipelineRunner;

#[test]
fn verify_attention() {
    let runner = DualPipelineRunner::new();
    let report = runner.compare_attention::<MyNpuAttention>(
        MyNpuAttention::new(),
        1e-4,  // epsilon threshold
    );
    assert!(report.passed());
}
```

## Verification tiers

Your implementation automatically receives a verification tier:

- **Compiled** — it type-checks against the trait (you get this for free)
- **Tested** — DualPipeline equivalence confirmed within epsilon
- **Proven** — Kani proofs pass + DualPipeline equivalence

Override the default tier by implementing `verification_tier()`:

```rust
impl Attention for MyNpuAttention {
    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Tested
    }
    // ...
}
```

## Composing a pipeline

Mix implementations from different crates:

```rust
let pipeline = KernelPipeline {
    attention: MyNpuAttention::new(),
    rmsnorm: CpuRMSNorm,
    rope: CpuRoPE,
    matmul: MyNpuMatMul::new(),
    sampler: CpuSampler,
};
```
