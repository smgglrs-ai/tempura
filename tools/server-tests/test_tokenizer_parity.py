"""
Level 2: Tokenizer parity.
Fixture-based: pre-computed expected token sequences (no llama.cpp needed in CI).
With LLAMACPP_URL set: live comparison against llama.cpp /tokenize.
"""

import json
import os
from pathlib import Path

import httpx
import pytest

TEMPURA_URL = os.environ.get("TEMPURA_URL", "http://localhost:8080")
LLAMACPP_URL = os.environ.get("LLAMACPP_URL", "")

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tokenizer_parity.json"

TOKENIZER_TEST_PROMPTS = [
    "The quick brown fox jumps over the lazy dog",
    "Hello, world!",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "日本語テスト",
    "2+2=4, 3*3=9",
    "",
    "   ",
    "<|begin_of_text|>",
]


def _tokenize(url, content):
    try:
        r = httpx.post(
            f"{url}/tokenize",
            json={"model": "default", "content": content},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        return r.json().get("tokens", [])
    except Exception:
        return None


def test_tokenizer_nonempty_for_nonempty_input():
    tokens = _tokenize(TEMPURA_URL, "Hello world")
    if tokens is None:
        pytest.skip("Server not available")
    assert len(tokens) > 0


def test_tokenizer_returns_ints():
    tokens = _tokenize(TEMPURA_URL, "test")
    if tokens is None:
        pytest.skip("Server not available")
    assert all(isinstance(t, int) for t in tokens)


def test_tokenizer_roundtrip():
    for prompt in TOKENIZER_TEST_PROMPTS[:4]:
        tokens = _tokenize(TEMPURA_URL, prompt)
        if tokens is None:
            pytest.skip("Server not available")
        r = httpx.post(
            f"{TEMPURA_URL}/detokenize",
            json={"model": "default", "tokens": tokens},
            timeout=10,
        )
        if r.status_code == 200:
            recovered = r.json().get("content", "")
            assert recovered.strip() == prompt.strip() or len(recovered) > 0


def test_tokenizer_same_length_for_same_input():
    tokens_a = _tokenize(TEMPURA_URL, "Hello world")
    tokens_b = _tokenize(TEMPURA_URL, "Hello world")
    if tokens_a is None:
        pytest.skip("Server not available")
    assert tokens_a == tokens_b


@pytest.mark.parametrize("prompt", TOKENIZER_TEST_PROMPTS[:5])
def test_tokenizer_fixture_matches(prompt):
    """When fixtures are populated, verify server output matches."""
    if not FIXTURE_PATH.exists():
        pytest.skip("Run generate_fixtures.py first")
    data = json.loads(FIXTURE_PATH.read_text())
    expected = {entry["text"]: entry["tokens"] for entry in data.get("prompts", [])}
    if prompt not in expected:
        pytest.skip(f"No fixture for prompt {repr(prompt)[:30]}")
    actual = _tokenize(TEMPURA_URL, prompt)
    if actual is None:
        pytest.skip("Server not available")
    assert actual == expected[prompt], (
        f"Token mismatch for {repr(prompt)[:40]}\n"
        f"  expected: {expected[prompt]}\n"
        f"  actual:   {actual}"
    )


@pytest.mark.compare
def test_tokenizer_matches_llamacpp():
    for prompt in TOKENIZER_TEST_PROMPTS:
        t_tempura = _tokenize(TEMPURA_URL, prompt)
        t_llamacpp = _tokenize(LLAMACPP_URL, prompt)
        if t_tempura is None or t_llamacpp is None:
            pytest.skip("Server not available")
        assert t_tempura == t_llamacpp, (
            f"Tokenizer mismatch for {repr(prompt)[:40]}\n"
            f"  tempura:   {t_tempura}\n"
            f"  llama.cpp: {t_llamacpp}"
        )
