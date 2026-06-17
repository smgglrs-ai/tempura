+++
title = "API Reference"
description = "OpenAI-compatible endpoints and tempura-specific verification APIs."
weight = 40
template = "docs/section.html"

[extra]
toc = true
+++

## OpenAI-compatible endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |
| `/v1/completions` | POST | Text completions |
| `/v1/models` | GET | List loaded models |
| `/v1/responses` | POST | Responses API (stateless) |

All endpoints support: `logprobs`, `top_logprobs`, `stop` sequences, `n` completions, `presence_penalty`, `frequency_penalty`, `response_format` (JSON mode / JSON Schema), `tools` / `tool_choice`, `seed`, `max_completion_tokens`, `stream_options.include_usage`.

## tempura-specific endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tempura/v1/traces` | GET | List recent inference traces |
| `/tempura/v1/traces/{id}` | GET | Retrieve a full InferenceTrace |
| `/tempura/v1/traces/{id}` | DELETE | Remove a stored trace |
| `/tempura/v1/replay` | POST | Deterministic replay of a previous inference |
| `/tempura/v1/compare` | POST | DualPipeline comparison between kernel configs |
| `/tempura/v1/verified` | GET | Verification manifest (Kani proof status per kernel) |

## Operational endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe (always 200) |
| `/ready` | GET | Readiness probe (200 when model loaded) |
| `/health/inference` | GET | Black-box probe (runs 1-token inference) |
| `/metrics` | GET | Prometheus metrics |
