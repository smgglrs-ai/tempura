# tempura server tests

Pytest suite covering the tempura inference server API. Organised into five levels matching the testing plan.

## Install

```sh
pip install -r requirements.txt
```

## Run

### Level 5 — API surface (only needs tempura running)

```sh
TEMPURA_URL=http://localhost:8080 pytest test_api_surface.py test_streaming.py -v
```

### Level 2 — Tokenizer parity (fixture-based, no llama.cpp needed)

```sh
# Generate fixtures first (one-time, against a running server with a real model)
python generate_fixtures.py http://localhost:8080

# Run parity tests
TEMPURA_URL=http://localhost:8080 pytest test_tokenizer_parity.py -v
```

### Level 4 — Generation parity (against llama.cpp)

```sh
# Start llama.cpp server on 8081 with the same model
TEMPURA_URL=http://localhost:8080 \
LLAMACPP_URL=http://localhost:8081 \
pytest test_generation_parity.py -m compare -v
```

### Authentication tests

```sh
TEMPURA_URL=http://localhost:8080 \
TEMPURA_API_KEY=your-key \
pytest test_auth.py -v
```

### All non-compare tests

```sh
TEMPURA_URL=http://localhost:8080 pytest -v --ignore=fixtures
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `TEMPURA_URL` | `http://localhost:8080` | Tempura server address |
| `LLAMACPP_URL` | _(empty)_ | llama.cpp server for compare tests; tests are skipped if unset |
| `TEMPURA_API_KEY` | _(empty)_ | Required for auth tests; enables auth header on all requests |

## Markers

- `compare` — requires both `TEMPURA_URL` and `LLAMACPP_URL`; auto-skipped in CI unless both are set
- `slow` — long-running tests; pass `--run-slow` to include

## Fixture generation

```sh
# Tokenizer fixtures only
python generate_fixtures.py http://localhost:8080

# Tokenizer + greedy generation fixtures
python generate_fixtures.py http://localhost:8080 --greedy
```

Fixtures are committed to `fixtures/` once generated against a trusted reference run. Re-generate when the model or tokenizer changes.
