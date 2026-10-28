# tempura

Rust-native LLM inference server targeting NVIDIA GPUs via cuda-oxide and
x86 CPUs via AVX2/AVX-512, with formal verification (Kani, Verus, TLA+) on
every layer — from quantization bit-packing to the batch scheduler.

Research project today; the ambition is a real competitor to llama.cpp
(local/CPU/edge) and vllm (served/GPU).

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

CPU (Lunar Lake laptop, BitNet b1.58-2B ternary): 36 tok/s decode at 1.24GB —
parity with bitnet.cpp; short-prompt prefill up to 2x faster.

## Design

- **Kernel traits as verification contract.** Engine is generic over kernel implementations. CPU reference for proof; CUDA via cuda-oxide for production. Kani proves both adhere to the contract.
- **Formal verification first.** 240 Kani harnesses (bounded safety), 588 Verus proof functions in 80 modules (unbounded deductive), 2 TLA+ specs (TLC model checking), plus proptest for float properties and C cross-validation of the ggml translation against llama.cpp itself.
- **Inference traces as first-class objects.** Every forward pass produces a structured `InferenceTrace`. Queryable, diffable, replayable.
- **Deterministic by design.** Seeded RNG, controlled reduction order. Same input + seed = bit-identical output.
- **CUDA graph for decode.** Single `cuGraphLaunch` per step (was 68+ CUDA driver calls). Batch graphs capture M sequences per launch.
- **The llama.cpp translation is the reference, not the product.** `kernels/src/ggml/` mirrors upstream file-by-file (doc comments cite C line numbers); optimized paths diverge deliberately with DualPipeline proofs covering the divergence.
- **Ternary models are a first-class target.** BitNet b1.58 (TQ2_0/TQ1_0/I2S/TL2) and PrismML Bonsai (custom Q2_0/Q1_0 GGUF types) — the {-1,0,1} weight path is fully decidable by Kani.

## Workspace

- **kernel-api/** -- Kernel traits + shared verification contracts (`Verified*`, `VerifiedEquivalent`). Zero implementations.
- **kernels/** -- CPU reference + optimized CPU (llama.cpp ggml translation, BitNet TL2 LUT kernels), cuTile JIT + oxide PTX + sm_120a PTX GPU kernels, MXFP4, TurboQuant, DeltaNet.
- **crates/staticpar/** -- Static-partition thread pool (spin/condvar hybrid, epoch barrier).
- **kernel-verify/** -- DualPipeline numerical comparison of kernel implementations.
- **engine/** -- Inference engine: 6 architecture families (29 model_types), hybrid attention/DeltaNet layers, MoE, KV caches (F16/paged/FP8/prefix), GGUF + safetensors loaders, speculative decoding, Medusa, disaggregated prefill/decode, CUDA-graph execution.
- **model/** -- HuggingFace hub download/inspect library (model store at `~/.cache/tempura/models`).
- **server/** -- OpenAI-compatible HTTP API (chat/completions/responses/embeddings), continuous batching with priority preemption, SSE streaming, Prometheus metrics. Library crate.
- **cli/** -- The `tempura` binary: `serve`, `pull`, `list`, `inspect`.
- **tools/** -- trace-cli (TUI trace viewer), bench, bench-gpu, compare, integration (workspace); oxide-gemv{,-q4k,-q6k}, oxide-decode-attn, mxf4-gemv (standalone `cargo oxide` crates).
- **verus-verified/** -- Standalone Verus deductive proofs (own toolchain, 80 modules).
- **tla/** -- TLA+ specs: BatchScheduler, SpeculativeDecoding (TLC-checked).
- **tests-ggml/** -- C-generated reference vectors from real llama.cpp.
- **docs/** -- Zola documentation site (math → hardware → kernel math → verification → serving).

## Quick Start

```sh
cargo build
cargo test
cargo run -p tempura -- --help
cargo run -p tempura -- pull TinyLlama/TinyLlama-1.1B-Chat-v1.0   # download from HuggingFace
cargo run -p tempura -- serve --model /path/to/model.gguf --port 8080
```

## GPU (cuda-oxide)

```sh
# Build oxide GEMV kernels
export CUDA_OXIDE_BACKEND=~/Code/github.com/NVLabs/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
for crate in oxide-gemv oxide-gemv-q4k oxide-gemv-q6k oxide-decode-attn; do
  cd tools/$crate && cargo oxide build && cd ../..
done

# Benchmark single-sequence decode
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release \
  --bin bench-real-model -- ~/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Benchmark batch decode (benches M=1,2,4,5,8,16 in one run)
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release \
  --bin bench-batch -- ~/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

When the GPU lives on another machine, `scripts/gpu-remote.sh` drives the whole
loop over SSH (sync → oxide build → bench → pull results) — see AGENTS.md.

## Formal Verification

```sh
cargo test                            # 907 test functions
cargo kani --package tempura-kernels  # 240 bounded harnesses
./scripts/run-verus.sh               # 588 Verus proof functions, 80 modules
```

## License

Apache-2.0
