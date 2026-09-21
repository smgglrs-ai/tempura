#!/usr/bin/env python3
"""
Generate test fixtures from a running tempura server.

Usage:
  python generate_fixtures.py http://localhost:8080
  python generate_fixtures.py http://localhost:8080 --greedy
"""

import argparse
import json
import sys
from pathlib import Path

import httpx
from openai import OpenAI

FIXTURE_DIR = Path(__file__).parent / "fixtures"

TOKENIZER_PROMPTS = [
    "The quick brown fox jumps over the lazy dog",
    "Hello, world!",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "日本語テスト",
    "2+2=4, 3*3=9",
    "   ",
    "<|begin_of_text|>",
]

GREEDY_PROMPTS = [
    "Once upon a time",
    "The capital of France is",
    "def hello(): print(",
    "In 1969, Neil Armstrong",
    "The answer to life, the universe, and everything is",
]


def generate_tokenizer_fixtures(base_url: str) -> None:
    out = {"_note": f"Generated from {base_url}", "prompts": []}
    for prompt in TOKENIZER_PROMPTS:
        try:
            r = httpx.post(
                f"{base_url}/tokenize",
                json={"model": "default", "content": prompt},
                timeout=10,
            )
            if r.status_code == 200:
                tokens = r.json().get("tokens", [])
                out["prompts"].append({"text": prompt, "tokens": tokens})
                print(f"  {repr(prompt)[:40]} → {tokens[:6]}...")
            else:
                print(f"  SKIP {repr(prompt)[:30]}: HTTP {r.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"  ERROR {repr(prompt)[:30]}: {e}", file=sys.stderr)

    path = FIXTURE_DIR / "tokenizer_parity.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {path} ({len(out['prompts'])} entries)")


def generate_greedy_fixtures(base_url: str) -> None:
    client = OpenAI(base_url=f"{base_url}/v1", api_key="test")
    out = {"_note": f"Generated from {base_url}", "sequences": []}
    for prompt in GREEDY_PROMPTS:
        try:
            resp = client.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
                temperature=0.0,
                seed=42,
            )
            text = resp.choices[0].message.content
            out["sequences"].append({"prompt": prompt, "output": text, "seed": 42})
            print(f"  {repr(prompt)[:40]} → {repr(text[:30])}...")
        except Exception as e:
            print(f"  ERROR {repr(prompt)[:30]}: {e}", file=sys.stderr)

    path = FIXTURE_DIR / "greedy_sequences.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {path} ({len(out['sequences'])} entries)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default="http://localhost:8080")
    parser.add_argument("--greedy", action="store_true", help="Also generate greedy sequence fixtures")
    args = parser.parse_args()

    print(f"Generating tokenizer fixtures from {args.url}...")
    generate_tokenizer_fixtures(args.url)

    if args.greedy:
        print(f"\nGenerating greedy sequence fixtures from {args.url}...")
        generate_greedy_fixtures(args.url)


if __name__ == "__main__":
    main()
