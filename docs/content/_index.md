+++
title = "tempura"

[extra]
lead = 'A composable framework where every kernel is formally proven, every forward pass is observable, and every output is deterministic.'
url = "/docs/getting-started/"
url_button = "Get started"
repo_version = "GitHub"
repo_license = "Open-source Apache-2.0 License."
repo_url = "https://github.com/smgglrs-ai/tempura"

[[extra.menu.main]]
name = "Docs"
section = "docs"
url = "/docs/getting-started/"
weight = 10

# ── Feature cards (grid) ──

[[extra.list]]
title = "Formally verified"
content = "Every kernel implementation is proven safe with <a href='https://model-checking.github.io/kani/'>Kani</a> — no panics, no out-of-bounds access. DualPipeline validates numerical equivalence between implementations."

[[extra.list]]
title = "Observable by design"
content = "Every forward pass produces a structured InferenceTrace at configurable granularity. Layer-by-layer statistics, per-token sampling decisions, timing breakdowns."

[[extra.list]]
title = "Deterministic"
content = "Same input + same seed = bit-identical output. Seeded RNG, controlled reduction order, reproducible sampling. On GPU, cuda-oxide enables deterministic reduction without post-hoc hacks."

[[extra.list]]
title = "Composable kernel ecosystem"
content = "Implement <code>Attention</code>, <code>MatMul</code>, <code>RoPE</code> for your hardware — CPU, GPU, NPU — and the framework verifies your implementation against the reference. Publish as a crate."

[[extra.list]]
title = "Production-ready serving"
content = "OpenAI-compatible API with continuous batching, speculative decoding, prefix caching, tensor parallelism, guided decoding, tool calling. Prometheus metrics, OpenTelemetry traces."

[[extra.list]]
title = "Rust-native, zero Python"
content = "No Python runtime, no GIL, no PyTorch dependency. Single static binary. Compiles to a scratch container image. GPU kernels in Rust via cuda-oxide."

# ── Deep-dive sections ──

[[extra.showcase]]
title = "Three layers of verification"
subtitle = "Trust, but prove."
description = "Other inference servers test their kernels. tempura proves them. Three verification layers work together — each catching what the others can't."
items = [
  { title = "Kani model checking", text = "Proves safety for <strong>all possible inputs</strong> within bounded ranges. No panics, no out-of-bounds access, no undefined behavior. Runs on every PR in CI." },
  { title = "DualPipeline comparison", text = "Runs two kernel implementations on identical inputs and compares outputs layer-by-layer. Catches numerical drift, precision loss, and implementation bugs." },
  { title = "Property-based testing", text = "Covers mathematical properties Kani can't handle — floating-point exp/trig, distribution equivalence, statistical invariants. Complements formal proofs with empirical coverage." },
]

[[extra.showcase]]
title = "The wireshark of inference"
subtitle = "InferenceTrace"
description = "Every forward pass is a structured, queryable object — not a debug artifact. Capture what happened inside the transformer, diff two runs, replay deterministically."
items = [
  { title = "Capture", text = "Per-layer norm statistics, per-token logit distributions, sampling decisions, timing breakdowns. Summary traces are ~16 KB — small enough to store for every production request." },
  { title = "Diff", text = "Compare two traces to find the exact layer and token where they diverge. Bisect numerical drift between kernel implementations, quantization effects, or cross-device behavior." },
  { title = "Replay", text = "Given a trace, re-run with the same inputs and seed. Verify bit-identical output. Debug production issues by replaying the exact inference that produced them." },
]

[[extra.showcase]]
title = "Composable crate ecosystem"
subtitle = "Build for your hardware"
description = "tempura is not a monolithic server — it's a framework. Kernel traits are the contract. Implement them for your hardware, verify against the reference, publish as a crate."
items = [
  { title = "tempura-kernel-api", text = "The stable contract. All 9 kernel traits, zero implementations. Depend on this to build a backend for any device — CPU, GPU, NPU, FPGA." },
  { title = "tempura-kernel-verify", text = "The verification toolkit. DualPipeline runner, Kani harness templates, numerical equivalence test generators. One command verifies your implementation." },
  { title = "tempura-kernels", text = "First-party CPU and CUDA implementations. The reference that third-party backends are verified against. Proven with Kani, tested in production." },
]

+++
