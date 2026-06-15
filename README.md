# oxide-serve

Rust-native LLM inference server designed to use [cuda-oxide](https://github.com/NVlabs/cuda-oxide) for GPU kernels. Currently runs on CPU with reference kernel implementations; cuda-oxide integration is planned behind a feature flag once the compiler stabilizes.

## Workspace

- **kernels** — kernel traits (attention, layernorm, RoPE, sampling) with CPU reference implementations
- **engine** — model loading, KV-cache, transformer forward pass, tokenizer
- **server** — OpenAI-compatible HTTP API with continuous batching and SSE streaming

## Quick Start

```sh
cargo build
cargo test
cargo run -- --help
```

## License

Apache-2.0
