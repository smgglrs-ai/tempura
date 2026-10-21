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
cargo test               # run all tests (~580 total)
```

> **Note:** `cargo test` requires the oxide PTX files to exist. These are built separately via `cargo oxide build` in `tools/oxide-gemv*/`. Without them the `tempura-kernels` crate will fail to compile. ~270 tests from the other packages (`engine`, `server`, `model`, etc.) run without PTX.

## Run

```sh
cargo run -- --help              # see CLI options
cargo run -- --port 8080         # start server (CPU inference)
cargo run -- --model /path/to/model --port 8080  # load a real model
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

## Formal verification

```sh
cargo kani --package tempura-kernels   # verify kernel safety properties
```

## Project structure

```
tempura/
  kernel-api/    # Trait definitions (the verification contract)
  kernels/       # CPU and CUDA implementations
  engine/        # Inference engine, model loading, traces
  server/        # HTTP serving, OpenAI API, batch scheduling
```
