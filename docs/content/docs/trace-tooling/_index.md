+++
title = "Trace Tooling"
description = "InferenceTrace, diff engine, replay — the wireshark of inference."
weight = 60
template = "docs/section.html"

[extra]
toc = true
+++

## InferenceTrace

Every forward pass produces a structured trace at configurable granularity:

| Level | What's captured | Size (32-layer, 100 tokens) |
|-------|----------------|----------------------------|
| **Off** | Nothing | 0 |
| **Metadata** | Config, token counts, timing | ~100 bytes |
| **Summary** | Per-layer norm stats, per-token top logits, sampling choices | ~16 KB |
| **Full** | Attention score matrices per layer per head | ~512 MB/layer |

Summary is the production default — small enough to store for every request.

## Requesting traces

Add `trace_level` to any inference request:

```sh
curl http://localhost:8080/v1/chat/completions \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "trace_level": "summary"
  }'
```

The response includes a `trace_id`. Retrieve the full trace:

```sh
curl http://localhost:8080/tempura/v1/traces/{trace_id}
```

## Trace diffing

Compare two traces to find where they diverge:

```
Divergence at layer 12, token 37:
  pre_norm mean shifted by 0.0034
  post_norm max_abs_diff = 0.008
  Classification: Numerical (same tokens selected)
```

Use this to debug differences between kernel implementations, quantization effects, or cross-device behavior.

## Deterministic replay

Replay a previous inference and verify bit-identical output:

```sh
curl -X POST http://localhost:8080/tempura/v1/replay \
  -d '{"trace_id": "abc123"}'
```

Returns `match: true` if the replay produces identical tokens.

## The vision: wireshark for inference

Wireshark captures packets at every protocol layer. tempura captures statistics at every transformer layer. Filter by layer, operation, token position. Navigate from overview to per-head attention scores. Bisect the exact point where two inference runs diverge.
