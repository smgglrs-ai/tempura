---
title: "Batch decode"
description: "CUDA graph batch decode: stream-K GEMM, KV scatter-write, performance numbers"
weight: 20
---

# Batch decode

Batch decode runs M sequences simultaneously in a single CUDA graph launch.
At M=8, RTX 5090, Llama-3.1-8B Q4K achieves 417 tok/s aggregate (3.5ms ITL/seq),
compared to 130 tok/s single-sequence.

## Architecture

### BatchDecodeGraph

`BatchDecodeGraph` captures a CUDA graph for M sequences. The graph is captured
once at the first step and replayed for all subsequent steps. Key properties:

- `max_batch_graph` is computed from available VRAM at startup. On RTX 5090 with
  32 GB VRAM and Llama-3.1-8B Q4K, this is 13+ sequences.
- QKV projections launch on separate CUDA streams to expose concurrency.
- KV cache writes use the scatter-write kernel (no graph updates per step).

### Stream-K Q4K batch GEMM

For M>1, the GEMV (M=1) is replaced by `q4k_streamk_batch_sm120`, a stream-K
GEMM:

- Persistent grid: 170 blocks (= SM_COUNT on RTX 5090), each block covers a
  variable number of output tiles determined by the stream-K partitioning scheme.
- cp.async double-buffer: two pipeline stages hide global memory latency.
- mma.sync tensor cores: 16x8x16 MMA tiles on int8 data.
- Grid shape: `(SM_COUNT, 1, 1)` -- all SMs busy regardless of M.

This fills all 170 SMs at M=2, which is the main advantage over running M
independent GEMVs (which would leave most SMs idle).

### KV scatter-write kernel

The KV scatter-write kernel reads `kv_write_offset` from a device buffer at
kernel execution time. This eliminates the need to update the CUDA graph's
memcpy nodes (64 `cuGraphExecMemcpyNodeSetParams` calls per step in the old
design). The device offset buffer is updated with a single tiny H->D memcpy
before `cuGraphLaunch`.

### Dynamic max_batch_graph

At startup, tempura queries free VRAM and computes the maximum batch size that
fits in the CUDA graph. For Llama-3.1-8B Q4K on RTX 5090 (32 GB):

```
VRAM breakdown per sequence:
  KV cache (32 layers x 32 heads x 128 dim x 2 (K+V) x max_seq_len x 2 bytes): ~2 GB
  Activations (intermediate buffers): ~200 MB
  Available for N sequences: ~28 GB -> 13 sequences
```

## Performance (RTX 5090, Llama-3.1-8B Q4_K_M)

| Batch M | Aggregate tok/s | ITL P50/seq | TFLOPS (Q4K GEMM) |
|---------|----------------|------------|-------------------|
| 1 | 130 | 7.7ms | -- (GEMV) |
| 2 | 134 | -- | -- |
| 4 | 243 | -- | -- |
| 8 | 417 | 3.5ms | -- |
| 16 | ~600 | -- | -- |
| 64 (MXFP4) | -- | -- | 55.6 |

llama.cpp single-sequence reference: 209 tok/s.

## Building and running

```sh
export CUDA_OXIDE_BACKEND=~/Code/github.com/NVLabs/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so

# Build oxide kernels
for crate in oxide-gemv oxide-gemv-q4k oxide-gemv-q6k oxide-decode-attn; do
  cd tools/$crate && cargo oxide build && cd ../..
done

# Batch decode benchmark (M=1..16 in one run)
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release \
  --bin bench-batch -- \
  ~/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

## Formal verification

The batch graph eligibility check (`is_batch_eligible`) is Kani-verified:
- Determinism: same model config + VRAM always produces the same max_batch_graph
- No buffer size overflow: grid dims fit within CUDA limits for any eligible M
