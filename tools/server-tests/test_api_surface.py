"""Level 5: API surface tests. Only requires a running tempura server."""

import json

import httpx
import pytest


def test_health_returns_200(http):
    r = http.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data


def test_ready_endpoint(http):
    r = http.get("/ready")
    assert r.status_code in (200, 503)


def test_metrics_prometheus_format(http):
    r = http.get("/metrics")
    assert r.status_code == 200
    assert "tempura_" in r.text or "# HELP" in r.text


def test_list_models(openai_client):
    models = openai_client.models.list()
    assert len(models.data) >= 1


def test_chat_completion_basic(openai_client):
    resp = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Say 'hello'"}],
        max_tokens=10,
    )
    assert resp.choices[0].message.content is not None
    assert len(resp.choices[0].message.content) > 0


def test_chat_completion_stream(openai_client):
    chunks = []
    with openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Count: 1 2 3"}],
        max_tokens=10,
        stream=True,
    ) as stream:
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
    assert len(chunks) > 0


def test_stream_equals_non_stream(openai_client):
    prompt = [{"role": "user", "content": "The capital of France is"}]

    non_stream = openai_client.chat.completions.create(
        model="default", messages=prompt, max_tokens=5, temperature=0.0, seed=42
    )

    streamed = []
    with openai_client.chat.completions.create(
        model="default",
        messages=prompt,
        max_tokens=5,
        temperature=0.0,
        seed=42,
        stream=True,
    ) as stream:
        for chunk in stream:
            if chunk.choices[0].delta.content:
                streamed.append(chunk.choices[0].delta.content)

    assert non_stream.choices[0].message.content == "".join(streamed)


def test_stop_sequence_respected(openai_client):
    resp = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Count from 1 to 10:"}],
        max_tokens=50,
        stop=["5"],
    )
    assert "6" not in resp.choices[0].message.content


def test_response_format_json(openai_client):
    resp = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": 'Return JSON: {"x": 1}'}],
        max_tokens=20,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content
    json.loads(text)


def test_logprobs_returned(openai_client):
    resp = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=3,
        logprobs=True,
        top_logprobs=3,
    )
    assert resp.choices[0].logprobs is not None


def test_seed_is_deterministic(openai_client):
    kwargs = dict(
        model="default",
        messages=[{"role": "user", "content": "Random word:"}],
        max_tokens=5,
        temperature=0.8,
        seed=12345,
    )
    r1 = openai_client.chat.completions.create(**kwargs)
    r2 = openai_client.chat.completions.create(**kwargs)
    assert r1.choices[0].message.content == r2.choices[0].message.content


def test_usage_field_present(openai_client):
    resp = openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5,
    )
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0


def test_stream_options_include_usage(openai_client):
    chunks = []
    with openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5,
        stream=True,
        stream_options={"include_usage": True},
    ) as stream:
        for chunk in stream:
            chunks.append(chunk)
    assert len(chunks) > 0
    last = chunks[-1]
    # Final chunk should carry usage when include_usage=True
    assert last.usage is not None or any(c.usage is not None for c in chunks)


def test_completions_basic(openai_client):
    resp = openai_client.completions.create(
        model="default", prompt="The sky is", max_tokens=5
    )
    assert resp.choices[0].text is not None


def test_tokenize_returns_token_ids(http):
    r = http.post("/tokenize", json={"model": "default", "content": "Hello world"})
    assert r.status_code == 200
    data = r.json()
    assert "tokens" in data
    assert isinstance(data["tokens"], list)
    assert len(data["tokens"]) > 0


def test_detokenize_roundtrip(http):
    r1 = http.post("/tokenize", json={"model": "default", "content": "Hello world"})
    tokens = r1.json()["tokens"]
    r2 = http.post("/detokenize", json={"model": "default", "tokens": tokens})
    assert r2.status_code == 200
    text = r2.json()["content"]
    assert "hello" in text.lower() or "Hello" in text


def test_tokenize_empty_string(http):
    r = http.post("/tokenize", json={"model": "default", "content": ""})
    assert r.status_code == 200


def test_embeddings_returns_error_without_embed_model(openai_client):
    try:
        openai_client.embeddings.create(model="default", input="test")
        pytest.fail("Expected an error for non-embedding model")
    except Exception as e:
        err = str(e).lower()
        assert "not_supported" in err or "400" in err or "not supported" in err


def test_props_endpoint(http):
    r = http.get("/props")
    assert r.status_code == 200
    data = r.json()
    assert "total_slots" in data or "build_info" in data


def test_malformed_request_returns_400(http):
    r = http.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code in (400, 422)


def test_missing_messages_returns_422(http):
    r = http.post("/v1/chat/completions", json={"model": "default"})
    assert r.status_code in (400, 422)


def test_debug_logits_endpoint(http):
    r = http.post("/tempura/v1/debug/logits", json={"tokens": [1, 2, 3]})
    assert r.status_code in (200, 404, 501)
    if r.status_code == 200:
        data = r.json()
        assert "logits" in data
        assert len(data["logits"]) > 0


def test_verified_status_endpoint(http):
    r = http.get("/tempura/v1/verified")
    assert r.status_code == 200


def test_traces_endpoint(http):
    r = http.get("/tempura/v1/traces")
    assert r.status_code == 200
