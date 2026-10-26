+++
title = "GPU Inference"
description = "CUDA graph decode, stream-K batch GEMM, MXFP4 pipeline, and kernel naming."
weight = 50
template = "docs/section.html"

[extra]
toc = true
+++

tempura's GPU path targets NVIDIA GPUs via cuda-oxide. All kernels are written
in Rust with the oxide backend compiling to PTX. Formal verification (Kani + Verus)
covers every kernel's arithmetic bounds.

## Pages

- [oxide GEMV optimizations](oxide-gemv-optimizations/) -- full optimization history, roofline analysis, status table
- [Batch decode](batch-decode/) -- CUDA graph batch decode, stream-K GEMM, performance numbers
- [Kernel naming](kernel-naming/) -- GpuCapabilities, GpuFeature bitflags, KernelVariant dispatch table
- [MXFP4 pipeline](mxfp4/) -- E2M1+UE8M0 format, cute-rs, BF16 loader, when to use

## Quick reference

| Kernel | Format | GPU | Notes |
|--------|--------|-----|-------|
| q4k_gemv_sm120_dp4a_kpar8 | Q4K | sm_120+ | primary single decode |
| q4k_gemv_dp4a_kpar8 | Q4K | sm_80+ (DP4A) | fallback |
| q4k_fused_silu_sm120_dp4a | Q4K | sm_120+ | fused gate+up FFN |
| q6k_gemv_sm120_mbar_kpar8 | Q6K | sm_120+ | down_proj |
| q4k_streamk_batch_sm120 | Q4K | sm_120+ | batch M=2..16, stream-K |
| q6k_gemv_sm120_batch | Q6K | sm_120+ | batch Q6K GEMM |
| MXFP4 MMA (cute-rs) | E2M1+UE8M0 | sm_120a | batch M>=16, 55 TFLOPS |
