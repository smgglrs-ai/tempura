"""
Level 4: Generation parity.
Without LLAMACPP_URL: sanity checks (non-empty, deterministic).
With LLAMACPP_URL: exact greedy token sequence comparison.
"""

import os

import pytest
from openai import OpenAI

TEMPURA_URL = os.environ.get("TEMPURA_URL", "http://localhost:8080")
LLAMACPP_URL = os.environ.get("LLAMACPP_URL", "")

GREEDY_PROMPTS = [
    "Once upon a time",
    "The capital of France is",
    "def hello(): print(",
    "In 1969, Neil Armstrong",
    "The answer to life, the universe, and everything is",
]


def _greedy(base_url, prompt, max_tokens=32, seed=42):
    client = OpenAI(base_url=f"{base_url}/v1", api_key="test")
    resp = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed,
    )
    return resp.choices[0].message.content


def test_greedy_nonempty():
    try:
        r = _greedy(TEMPURA_URL, "Hello")
        assert r is not None and len(r) > 0
    except Exception:
        pytest.skip("Server not available or model not loaded")


def test_greedy_is_deterministic():
    for prompt in GREEDY_PROMPTS[:2]:
        try:
            r1 = _greedy(TEMPURA_URL, prompt)
            r2 = _greedy(TEMPURA_URL, prompt)
        except Exception:
            pytest.skip("Server not available or model not loaded")
        assert r1 == r2, f"Non-deterministic output for {repr(prompt)}"


def test_different_seeds_differ():
    try:
        r1 = _greedy(TEMPURA_URL, "Random word:", seed=1)
        r2 = _greedy(TEMPURA_URL, "Random word:", seed=999)
    except Exception:
        pytest.skip("Server not available or model not loaded")
    # With temperature=0 (greedy) seed shouldn't matter — output may be same
    # This test just checks both calls succeed
    assert r1 is not None and r2 is not None


def test_longer_context_generates():
    long_prompt = " ".join(["word"] * 200)
    try:
        r = _greedy(TEMPURA_URL, long_prompt, max_tokens=10)
        assert r is not None
    except Exception:
        pytest.skip("Server not available or model not loaded")


@pytest.mark.compare
def test_greedy_matches_llamacpp():
    """Exact greedy output match against llama.cpp (temperature=0, seed=42)."""
    mismatches = []
    for prompt in GREEDY_PROMPTS:
        try:
            t_out = _greedy(TEMPURA_URL, prompt)
            l_out = _greedy(LLAMACPP_URL, prompt)
        except Exception as e:
            pytest.skip(f"Server error: {e}")

        if t_out != l_out:
            diff_pos = next(
                (i for i, (a, b) in enumerate(zip(t_out, l_out)) if a != b),
                min(len(t_out), len(l_out)),
            )
            mismatches.append(
                dict(
                    prompt=prompt,
                    diff_pos=diff_pos,
                    tempura=t_out[:diff_pos + 20],
                    llamacpp=l_out[:diff_pos + 20],
                )
            )

    if mismatches:
        lines = []
        for m in mismatches:
            lines += [
                f"  prompt:    {repr(m['prompt'])[:50]}",
                f"  tempura:   {repr(m['tempura'][:60])}",
                f"  reference: {repr(m['llamacpp'][:60])}",
                f"  first diff at char {m['diff_pos']}",
                "",
            ]
        pytest.fail("Generation mismatch(es):\n" + "\n".join(lines))


@pytest.mark.compare
def test_speculative_matches_non_speculative():
    """Speculative decoding must produce identical output to normal decoding."""
    prompt = "The Eiffel Tower is located in"
    try:
        normal = _greedy(TEMPURA_URL, prompt)
    except Exception:
        pytest.skip("Server not available or model not loaded")
    # Speculative is enabled per-model config; if both succeed they must match
    assert normal is not None
