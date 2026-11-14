#!/usr/bin/env python3
"""Streaming chat client for the p24 kv-swap A/B validation.

Usage: kv-swap-ab-client.py URL PROMPT OUTFILE MAX_TOKENS

Writes the decoded completion text (concatenated delta.content fields) to
OUTFILE so run A and run B can be compared byte-for-byte with cmp.

IMPORTANT: the SSE terminator is the EXACT line `data: [DONE]`. The
completion text itself may legally contain the substring "[DONE]", so no
substring search is ever performed on the stream -- each line is parsed,
and only a line whose entire data payload equals "[DONE]" ends the read.
"""
import json
import sys
import time
import urllib.request


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: kv-swap-ab-client.py URL PROMPT OUTFILE MAX_TOKENS", file=sys.stderr)
        return 2
    url, prompt, outfile, max_tokens = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
    )

    body = json.dumps(
        {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": 1234,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )

    chunks = []
    saw_done = False
    finish_reason = None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            # Exact-match sentinel: a completion containing "[DONE]" as text
            # arrives JSON-encoded inside delta.content, never on this path.
            if payload == "[DONE]":
                saw_done = True
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    chunks.append(delta["content"])
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    text = "".join(chunks)
    with open(outfile, "w") as f:
        f.write(text)
    elapsed = time.time() - t0
    print(
        f"{outfile}: {len(text)} chars, finish={finish_reason}, "
        f"done_sentinel={saw_done}, {elapsed:.1f}s",
        file=sys.stderr,
    )
    if not text:
        print(f"{outfile}: ERROR -- empty completion", file=sys.stderr)
        return 1
    if not saw_done:
        print(f"{outfile}: ERROR -- stream ended without [DONE] sentinel", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
