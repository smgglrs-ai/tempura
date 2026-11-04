#!/usr/bin/env bash
# Download GSM8K test set and build the deterministic sample consumed by
# bench-selfconsistency (~/.cache/tempura/eval/gsm8k/gsm8k_sample.jsonl).
# Source: openai/grade-school-math (GSM8K is MIT-licensed).
set -euo pipefail

CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/tempura/eval/gsm8k"
mkdir -p "$CACHE"
RAW="$CACHE/test.jsonl"

if [ ! -s "$RAW" ]; then
  echo "Downloading GSM8K test set..."
  curl -sL -o "$RAW" \
    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
fi

# Deterministic stratified sample: every 13th problem (1319 -> 101 problems),
# gold = the number after the "####" marker, commas stripped.
python3 - "$RAW" "$CACHE/gsm8k_sample.jsonl" <<'EOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
out = []
with open(src) as f:
    for i, line in enumerate(f):
        if i % 13:
            continue
        d = json.loads(line)
        gold = d["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"question": d["question"].strip(), "gold": gold})
with open(dst, "w") as f:
    for d in out:
        f.write(json.dumps(d) + "\n")
print(f"wrote {len(out)} problems to {dst}")
EOF
