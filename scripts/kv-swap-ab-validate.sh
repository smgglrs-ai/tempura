#!/usr/bin/env bash
# p24 kv-swap FINAL VALIDATION GATE (runs ON fishtank, not the laptop).
#
# Streaming A/B:
#   run A: TEMPURA_KV_POOL_TOKENS=20000  (no eviction expected)
#   run B: TEMPURA_KV_POOL_TOKENS=3072   (forced eviction via swap_out)
#   6 concurrent "stream": true clients, greedy temp=0.
# PASS = per-client decoded text byte-identical A vs B
#        AND "swapped out" lines present in B's serve log.
#
# Usage: bash scripts/kv-swap-ab-validate.sh
# (scp this file + scripts/kv-swap-ab-client.py to the remote repo first;
#  the remote clone must be on the wip branch with the p24 code.)
#
# NOTE: client termination uses the EXACT SSE line `data: [DONE]`, never a
# substring search -- completions may legally contain "[DONE]" as text.
set -uo pipefail

cd "$(dirname "$0")/.."
REPO=$(pwd)
MODEL="$HOME/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PORT=8199
CLIENTS=6
MAX_TOKENS=512
RUN=$(mktemp -d /tmp/kv-swap-ab.XXXXXX)
SRV_PID=""

log() { echo "[validate] $*"; }

cleanup() {
    if [ -n "$SRV_PID" ] && kill -0 "$SRV_PID" 2>/dev/null; then
        log "stopping server pid $SRV_PID"
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

kill_leftovers() {
    # Kill BY PID, never pkill by name (gpu-remote preflight rule).
    for p in $(pgrep -f "target/release/tempura serve" 2>/dev/null); do
        log "killing leftover server pid $p"
        kill "$p" 2>/dev/null || true
    done
    sleep 2
    for p in $(pgrep -f "target/release/tempura serve" 2>/dev/null); do
        log "force-killing pid $p"
        kill -9 "$p" 2>/dev/null || true
    done
}

wait_ready() {
    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

run_clients() {
    local tag="$1" i=0 p
    mkdir -p "$RUN/$tag"
    local pids=()
    for p in "${PROMPTS[@]}"; do
        i=$((i + 1))
        python3 scripts/kv-swap-ab-client.py \
            "http://127.0.0.1:$PORT/v1/chat/completions" \
            "$p" "$RUN/$tag/c$i.txt" "$MAX_TOKENS" \
            > "$RUN/$tag/c$i.stderr" 2>&1 &
        pids+=($!)
        # 1s stagger so admission order is stable-ish; all still overlap.
        sleep 1
    done
    # Wait ONLY for the clients: a bare `wait` also waits on the server
    # job and blocks forever (found the hard way -- the script sat 11h
    # at this line while the server hung on a dead engine thread).
    wait "${pids[@]}" || true
    cat "$RUN/$tag"/*.stderr
}

PROMPTS=(
    "Explain in detail how paged attention works in an LLM inference server, covering page tables, block sizes, and why it helps memory utilization. Write at least 400 words."
    "Write a long technical explanation of the differences between prefill and decode phases in transformer inference, including KV cache behavior and batching strategies. Write at least 400 words."
    "Describe the gated delta rule (linear attention) architecture in depth: how the recurrent and chunked forms work, and when hybrid attention/delta layers are preferable. Write at least 400 words."
    "Explain ternary quantization ({-1, 0, +1} weights) for LLMs: packing formats, SIMD evaluation, and the accuracy/memory tradeoffs compared to Q4/Q8. Write at least 400 words."
    "Write a detailed explanation of speculative decoding: draft and target models, acceptance sampling, and expected speedup. Include the math. Write at least 400 words."
    "Explain how CUDA graphs reduce per-step launch overhead in LLM decode loops, and what constraints they impose (fixed shapes, stable pointers). Write at least 400 words."
)

[ -f "$MODEL" ] || { log "ERROR: model not found: $MODEL"; exit 1; }

kill_leftovers

log "building server (cutile)..."
cargo build --release -p tempura --features cutile || { log "ERROR: build failed"; exit 1; }
# WHY: a rebuild lands tempura-kernels in a new fingerprinted out dir; the
# oxide PTX must be copied there again or decode silently falls back.
for d in target/release/build/tempura-kernels-*/out; do
    [ -d "$d" ] || continue
    cp tools/oxide-gemv/oxide_gemv.ptx               "$d/oxide_q2_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-gemv-q4k/oxide_gemv_q4k.ptx       "$d/oxide_q4k_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-gemv-q6k/oxide_gemv_q6k.ptx       "$d/oxide_q6k_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-decode-attn/oxide_decode_attn.ptx "$d/oxide_decode_attn.ptx" 2>/dev/null || true
done

run_server() {
    local tag="$1" pool_tokens="$2"
    log "starting server $tag: TEMPURA_KV_POOL_TOKENS=$pool_tokens"
    TEMPURA_KV_POOL_TOKENS="$pool_tokens" \
    TEMPURA_ENGINE_DEBUG_TICKS=1 \
        ./target/release/tempura serve \
        --model "$MODEL" --port "$PORT" --device cuda \
        --max-seq-len 24576 \
        --request-timeout-secs 2400 \
        > "$RUN/serve-$tag.log" 2>&1 &
    SRV_PID=$!
    if ! wait_ready; then
        log "ERROR: server $tag never became ready; log tail:"
        tail -50 "$RUN/serve-$tag.log"
        return 1
    fi
    log "server $tag ready (pid $SRV_PID)"
    return 0
}

stop_server() {
    local tag="$1"
    kill "$SRV_PID" 2>/dev/null || true
    # A panicked engine thread can hang graceful shutdown (it waits on the
    # dead thread forever) -- escalate to SIGKILL after a grace period.
    for _ in $(seq 1 15); do
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$SRV_PID" 2>/dev/null; then
        log "server $tag did not exit in 15s -- SIGKILL"
        kill -9 "$SRV_PID" 2>/dev/null || true
    fi
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
    sleep 2
    # Sanity: nothing left listening.
    if pgrep -f "target/release/tempura serve" >/dev/null 2>&1; then
        kill_leftovers
    fi
}

# ---- run A: large pool, no eviction ----
if ! run_server A 20000; then exit 1; fi
run_clients A
stop_server A

# ---- run B: small pool, forced eviction ----
if ! run_server B 3072; then exit 1; fi
run_clients B
stop_server B

# ---- compare ----
PASS=1
log "=== text comparison (A vs B, byte-identical required) ==="
for i in $(seq 1 "$CLIENTS"); do
    a="$RUN/A/c$i.txt"; b="$RUN/B/c$i.txt"
    if [ ! -s "$a" ]; then
        log "FAIL: A/c$i.txt empty"; PASS=0
    elif [ ! -s "$b" ]; then
        log "FAIL: B/c$i.txt empty"; PASS=0
    elif cmp -s "$a" "$b"; then
        log "c$i: MATCH ($(wc -c < "$a") bytes)"
    else
        log "FAIL: c$i MISMATCH ($(wc -c < "$a") vs $(wc -c < "$b") bytes)"
        PASS=0
    fi
done

log "=== swap evidence ==="
SWAP_B=$(grep -c "swapped out" "$RUN/serve-B.log" || true)
SWAP_A=$(grep -c "swapped out" "$RUN/serve-A.log" || true)
log "run A 'swapped out' lines: $SWAP_A (expected 0)"
log "run B 'swapped out' lines: $SWAP_B (required >= 1)"
if [ "$SWAP_B" -lt 1 ]; then
    log "FAIL: run B shows no swaps -- eviction was not exercised"
    PASS=0
fi

echo
if [ "$PASS" -eq 1 ]; then
    log "RESULT: PASS -- streamed text byte-identical, eviction exercised"
    log "artifacts: $RUN"
    exit 0
else
    log "RESULT: FAIL"
    log "artifacts: $RUN (serve-A.log, serve-B.log, A/, B/)"
    exit 1
fi
