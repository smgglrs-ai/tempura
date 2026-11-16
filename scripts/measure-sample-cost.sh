#!/bin/bash
# measure-sample-cost.sh -- ITEM 2 measurement: how much of the engine step
# the CPU sampler chain costs for SAMPLED (temperature > 0) sessions at m=6,
# under TEMPURA_ENGINE_DEBUG_TICKS. Decides build-vs-skip for a GPU-side
# temperature+top-k+top-p path (skip if the win is < 2 ms/step).
set -euo pipefail
cd "$HOME/Code/github.com/smgglrs-ai/tempura"
RUN="$HOME/.cache/tempura/gpu-remote"
LOG="$RUN/sample-cost.log"
MODEL="$HOME/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PORT=8127

# kill leftovers BY PID (never multiple instances)
for p in $(pgrep -f "target/release/tempura serve" 2>/dev/null); do kill "$p" 2>/dev/null || true; done
sleep 2

cargo build --release -p tempura --features cutile >/dev/null 2>&1
for d in target/release/build/tempura-kernels-*/out; do
    [ -d "$d" ] || continue
    cp tools/oxide-gemv/oxide_gemv.ptx               "$d/oxide_q2_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-gemv-q4k/oxide_gemv_q4k.ptx       "$d/oxide_q4k_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-gemv-q6k/oxide_gemv_q6k.ptx       "$d/oxide_q6k_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-decode-attn/oxide_decode_attn.ptx "$d/oxide_decode_attn.ptx" 2>/dev/null || true
done

TEMPURA_KV_POOL_TOKENS=20000 \
TEMPURA_ENGINE_DEBUG_TICKS=1 \
    ./target/release/tempura serve \
    --model "$MODEL" --port "$PORT" --device cuda \
    --max-seq-len 24576 --request-timeout-secs 600 \
    > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

for i in $(seq 60); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 2
done

# 6 concurrent streaming clients, temperature=0.8 (sampled, llama.cpp-style
# default-ish), 300 tokens each.
python3 - <<'EOF'
import json, threading, urllib.request

def client(i):
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": (
            f"Explain topic number {i} in depth: how paged attention, KV caches and "
            "continuous batching interact in an LLM inference server. Write 400 words."
        )}],
        "stream": True, "temperature": 0.8, "max_tokens": 300, "seed": 1234,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:8127/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.strip() == b"data: [DONE]":
                break

ts = [threading.Thread(target=client, args=(i,)) for i in range(6)]
for t in ts: t.start()
for t in ts: t.join()
EOF

kill $SRV 2>/dev/null || true
trap - EXIT
sleep 2

echo "=== sample+emit timing percentiles (sampled sessions, temp=0.8) ==="
grep -o 'sample+emit=[0-9.]*ms' "$LOG" | sed 's/sample+emit=//;s/ms//' | sort -n | awk '
    { a[NR]=$1 } END {
        printf "n=%d  min=%.3f  p50=%.3f  p90=%.3f  max=%.3f ms (all ticks)\n",
            NR, a[1], a[int(NR*0.5)], a[int(NR*0.9)], a[NR]
    }'
echo "=== m=6 ticks only ==="
grep 'm=6 ' "$LOG" | grep -o 'sample+emit=[0-9.]*ms' | sed 's/sample+emit=//;s/ms//' | sort -n | awk '
    { a[NR]=$1 } END {
        printf "n=%d  min=%.3f  p50=%.3f  p90=%.3f  max=%.3f ms\n",
            NR, a[1], a[int(NR*0.5)+1], a[int(NR*0.9)+1], a[NR]
    }'
echo "=== forward timing percentiles (all ticks, for context) ==="
grep -o 'forward=[0-9.]*ms' "$LOG" | sed 's/forward=//;s/ms//' | sort -n | awk '
    { a[NR]=$1 } END {
        printf "n=%d  p50=%.3f  p90=%.3f  max=%.3f ms\n", NR, a[int(NR*0.5)+1], a[int(NR*0.9)+1], a[NR]
    }'
