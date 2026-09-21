"""Streaming correctness: chunk structure, delta accumulation, SSE framing."""

import os

import httpx
import pytest
from openai import OpenAI

TEMPURA_URL = os.environ.get("TEMPURA_URL", "http://localhost:8080")
TEMPURA_API_KEY = os.environ.get("TEMPURA_API_KEY", "")


def _client():
    return OpenAI(base_url=f"{TEMPURA_URL}/v1", api_key=TEMPURA_API_KEY or "test")


def test_streaming_chunks_have_id(openai_client):
    ids = set()
    with openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
        stream=True,
    ) as stream:
        for chunk in stream:
            ids.add(chunk.id)
    assert len(ids) == 1, "All chunks should share the same request ID"


def test_streaming_role_in_first_chunk(openai_client):
    first = None
    with openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
        stream=True,
    ) as stream:
        for chunk in stream:
            first = chunk
            break
    assert first is not None
    assert first.choices[0].delta.role == "assistant"


def test_streaming_finish_reason_in_last_content_chunk(openai_client):
    last = None
    with openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
        stream=True,
    ) as stream:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].finish_reason:
                last = chunk
    assert last is not None
    assert last.choices[0].finish_reason in ("stop", "length")


def test_streaming_concatenated_equals_full(openai_client):
    prompt = [{"role": "user", "content": "Name three colors"}]
    full = openai_client.chat.completions.create(
        model="default", messages=prompt, max_tokens=20, temperature=0.0, seed=7
    )
    parts = []
    with openai_client.chat.completions.create(
        model="default",
        messages=prompt,
        max_tokens=20,
        temperature=0.0,
        seed=7,
        stream=True,
    ) as stream:
        for chunk in stream:
            if chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
    assert full.choices[0].message.content == "".join(parts)


def test_streaming_no_empty_content_chunks(openai_client):
    with openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
        stream=True,
    ) as stream:
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                assert content != "", "Content deltas should not be empty strings"


def test_stream_options_include_usage_appends_usage_chunk(openai_client):
    usage_chunks = []
    with openai_client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
        stream=True,
        stream_options={"include_usage": True},
    ) as stream:
        for chunk in stream:
            if chunk.usage is not None:
                usage_chunks.append(chunk)
    assert len(usage_chunks) >= 1
    u = usage_chunks[-1].usage
    assert u.prompt_tokens > 0
    assert u.completion_tokens > 0


def test_raw_sse_format(http):
    """Verify raw SSE wire format has data: prefix and data: [DONE] terminator."""
    with http.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        lines = []
        for line in r.iter_lines():
            lines.append(line)
    data_lines = [l for l in lines if l.startswith("data:")]
    assert len(data_lines) >= 2
    assert data_lines[-1].strip() == "data: [DONE]"
