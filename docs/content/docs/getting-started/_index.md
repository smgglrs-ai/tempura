+++
title = "Getting Started"
description = "Build, test, and run tempura in under five minutes."
weight = 10
template = "docs/section.html"

[extra]
toc = true
+++

## Prerequisites

- Rust 1.80+ (edition 2021)
- For GPU: NVIDIA GPU with cuda-oxide toolchain

## Build

```sh
cargo build              # build all crates
cargo test               # run all tests (907 total, 26 ignored)
```

> **Note:** `cargo test` compiles `tempura-kernels` with the `oxide-gemv`
> feature (feature unification via `tools/bench-gpu`), so the oxide GEMV PTX
> must exist in the kernels build out dirs — build them with
> `cargo oxide build` in `tools/oxide-gemv*/` (or, from the laptop,
> `scripts/gpu-remote.sh oxide`). Without them `tempura-kernels` fails to
> compile; tests from the other packages still run.

## Run

```sh
cargo run -p tempura -- --help              # see CLI options
cargo run -p tempura -- serve --port 8080   # start server (CPU inference)
cargo run -p tempura -- serve --model /path/to/model --port 8080  # load a real model
```

The server exposes an OpenAI-compatible API:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

## GPU inference (cuda-oxide)

The GPU path uses cuda-oxide for hand-optimized PTX kernels. You need the
`CUDA_OXIDE_BACKEND` environment variable pointing at the local fork build.

```sh
# Build the backend from the fork (once)
cd ~/Code/github.com/NVLabs/cuda-oxide/crates/rustc-codegen-cuda
cargo build --release

# Export in every shell that runs GPU benchmarks
export CUDA_OXIDE_BACKEND=~/Code/github.com/NVLabs/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so

# Build oxide GEMV kernels
cd /path/to/tempura
for crate in oxide-gemv oxide-gemv-q4k oxide-gemv-q6k oxide-decode-attn; do
  cd tools/$crate && cargo oxide build && cd ../..
done

# Single-sequence benchmark (Llama-3.1-8B Q4K, RTX 5090: 130 tok/s)
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release \
  --bin bench-real-model -- ~/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Batch decode benchmark (M=1..16 in one run; RTX 5090: 417 tok/s aggregate at M=8)
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release \
  --bin bench-batch -- ~/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

## MXFP4 pipeline (sm_120a / RTX 5090)

MXFP4 requires sm_120a (RTX 5090 consumer Blackwell). Load from BF16 safetensors:

```sh
# Benchmark MXFP4 at M=64 (55.6 TFLOPS on RTX 5090)
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release \
  --bin bench-mxfp4 -- /path/to/model-bf16/model.safetensors
```

Weights are tiled from BF16 to E2M1+UE8M0 at load time. MXFP4 is efficient
at M>=16 (batch). For M<16 batch decode, the stream-K Q4K path is faster.

## Formal verification

```sh
cargo kani --package tempura-kernels   # 78 bounded harnesses
./scripts/run-verus.sh                 # 828 Verus verification conditions
```

## Project structure

```
tempura/
  kernel-api/    # Trait definitions (the verification contract)
  kernels/       # CPU and CUDA implementations
  engine/        # Inference engine, model loading, traces
  server/        # HTTP serving, OpenAI API, batch scheduling
  tools/         # Benchmarks, oxide GEMV crates, bench-gpu
  verus-verified/ # Standalone Verus deductive proofs (own toolchain)
```
