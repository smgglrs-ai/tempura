# tempura

Rust-native LLM inference server targeting NVIDIA GPUs via cuda-oxide, with
formal verification via Kani and Verus on every kernel.

## Performance (RTX 5090, Llama-3.1-8B Q4_K_M)

| Mode | tok/s | vs llama.cpp |
|------|-------|-------------|
| Single-sequence decode | 130 | 1.60x gap |
| Batch M=2 decode | 134 aggregate | -- |
| Batch M=4 decode | 243 aggregate | -- |
| Batch M=8 decode | 417 aggregate | -- |
| Batch M=16 decode | ~600 aggregate | -- |
| MXFP4 at M=64 | 55.6 TFLOPS | -- |

llama.cpp reference: 209 tok/s single-sequence decode, 7335 tok/s pp128.

## Design

- **Kernel traits as verification contract.** Engine is generic over kernel implementations. CPU reference for proof; CUDA via cuda-oxide for production. Kani proves both adhere to the contract.
- **Formal verification first.** Every kernel has Kani harnesses (bounded safety) and Verus proofs (unbounded deductive). 78 Kani harnesses, 828 Verus verification conditions.
- **Inference traces as first-class objects.** Every forward pass produces a structured `InferenceTrace`. Queryable, diffable, replayable.
- **Deterministic by design.** Seeded RNG, controlled reduction order. Same input + seed = bit-identical output.
- **CUDA graph for decode.** Single `cuGraphLaunch` per step (was 68+ CUDA driver calls). Batch graphs capture M sequences per launch.

## Workspace

- **kernel-api/** -- Kernel traits (the verification contract). Zero implementations.
- **kernels/** -- CPU reference implementations + CUDA kernels via cuda-oxide.
- **engine/** -- Inference engine: model loading, KV cache, transformer forward pass, CUDA graph execution.
- **server/** -- OpenAI-compatible HTTP API, continuous batching, SSE streaming, Prometheus metrics.
- **verus-verified/** -- Standalone Verus deductive proofs (own toolchain, 62 modules).

## Quick Start

```sh
cargo build
cargo test
cargo run -- --help
cargo run -- --model /path/to/model.gguf --port 8080
```

## GPU (cuda-oxide)

```sh
# Build oxide GEMV kernels
export CUDA_OXIDE_BACKEND=~/Code/github.com/NVLabs/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
for crate in oxide-gemv oxide-gemv-q4k oxide-gemv-q6k; do
  cd tools/$crate && cargo oxide build && cd ../..
done

# Benchmark single-sequence decode
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release \
  --bin bench-real-model -- ~/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Benchmark batch decode (M=8)
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release \
  --bin bench-batch-decode -- ~/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --batch 8
```

## Formal Verification

```sh
cargo kani --package tempura-kernels       # 78 bounded harnesses
./scripts/run-verus.sh                     # 828 Verus verification conditions
```

## License

Apache-2.0
