#!/usr/bin/env bash
# EAGLE acceptance/throughput sweep (fishtank): runs the solo 400-token
# request under each config and prints the telemetry + wall time.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=$HOME/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
EAGLE=$HOME/.cache/tempura/models/eagle3-llama3.1-8b
PORT=8197
PROMPT="Explain, step by step, how the softmax function works and why subtracting the max improves numerical stability. Be precise about the math."

run_one() {
    local tag="$1"; shift
    pkill -f "release/tempura" 2>/dev/null || true
    sleep 3
    env TEMPURA_ENGINE_DEBUG_TICKS=1 "$@" \
        ./target/release/tempura serve --model "$MODEL" --port $PORT --device cuda \
        --max-seq-len 24576 --eagle-head "$EAGLE" > /tmp/eagle-sweep.log 2>&1 &
    local srv=$!
    for _ in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1 && break
        sleep 1
    done
    local t0 t1
    t0=$(date +%s.%N)
    timeout 240 python3 scripts/kv-swap-ab-client.py \
        "http://127.0.0.1:$PORT/v1/chat/completions" "$PROMPT" /tmp/eagle-sweep.txt 400 >/dev/null 2>&1
    t1=$(date +%s.%N)
    local secs=$(echo "$t1 - $t0" | bc)
    echo "=== $tag: 400 tok in ${secs}s ($(awk -v s="$secs" 'BEGIN{printf "%.1f", 400/s}') tok/s client-side)"
    grep -o "spec: cycles=.*hist=\[[0-9, ]*\]" /tmp/eagle-sweep.log | tail -1 || echo "(no spec line)"
    kill $srv 2>/dev/null || true
    wait $srv 2>/dev/null || true
}

run_one "default K=4 shift=1 Q6K"
run_one "K=6" TEMPURA_SPEC_K=6
run_one "K=3" TEMPURA_SPEC_K=3
run_one "pos_shift=0" TEMPURA_EAGLE_POS_SHIFT=0
run_one "f32 head" TEMPURA_EAGLE_F32=1
pkill -f "release/tempura" 2>/dev/null || true
echo SWEEP-DONE
