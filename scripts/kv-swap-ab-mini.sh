#!/usr/bin/env bash
# FAST mini discriminator for the 6-client kv-swap A/B (runs ON fishtank).
#
# 3 concurrent streaming clients, max_tokens ~200:
#   run A1: TEMPURA_KV_POOL_TOKENS=20000 (no eviction)
#   run A2: TEMPURA_KV_POOL_TOKENS=20000 again (fresh server restart)
#   run B : TEMPURA_KV_POOL_TOKENS=700   (forced eviction mid-run)
#
# PASS = A1 == A2 == B byte-identical per client, and B's serve log shows
#        at least one "swapped out" line.
# The A1-vs-A2 comparison isolates cross-restart nondeterminism (the deepest
# clue from the full-gate failure) from eviction effects.
#
# Usage: bash scripts/kv-swap-ab-mini.sh
set -uo pipefail

cd "$(dirname "$0")/.."
MODEL="$HOME/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PORT=8197
CLIENTS=3
MAX_TOKENS=200
RUN=$(mktemp -d /tmp/kv-swap-mini.XXXXXX)
SRV_PID=""

log() { echo "[mini] $*"; }

cleanup() {
    if [ -n "$SRV_PID" ] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

kill_leftovers() {
    for p in $(pgrep -f "target/release/tempura serve" 2>/dev/null); do
        log "killing leftover server pid $p"
        kill "$p" 2>/dev/null || true
    done
    sleep 2
    for p in $(pgrep -f "target/release/tempura serve" 2>/dev/null); do
        kill -9 "$p" 2>/dev/null || true
    done
}

wait_ready() {
    for _ in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1 && return 0
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
        sleep 1
    done
    wait "${pids[@]}" || true
    cat "$RUN/$tag"/*.stderr
}

PROMPTS=(
    "Explain in detail how paged attention works in an LLM inference server, covering page tables, block sizes, and why it helps memory utilization. Write at least 400 words."
    "Write a long technical explanation of the differences between prefill and decode phases in transformer inference, including KV cache behavior and batching strategies. Write at least 400 words."
    "Describe the gated delta rule (linear attention) architecture in depth: how the recurrent and chunked forms work, and when hybrid attention/delta layers are preferable. Write at least 400 words."
)

[ -f "$MODEL" ] || { log "ERROR: model not found: $MODEL"; exit 1; }

kill_leftovers

log "building server (cutile)..."
cargo build --release -p tempura --features cutile || { log "ERROR: build failed"; exit 1; }
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
    kill "$SRV_PID" 2>/dev/null || true
    for _ in $(seq 1 15); do
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$SRV_PID" 2>/dev/null; then
        kill -9 "$SRV_PID" 2>/dev/null || true
    fi
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
    sleep 2
    if pgrep -f "target/release/tempura serve" >/dev/null 2>&1; then
        kill_leftovers
    fi
}

cmp_tag() {
    # cmp_tag BASE OTHER -> compares per-client files, prints verdict lines
    local base="$1" other="$2" i pass=1
    for i in $(seq 1 "$CLIENTS"); do
        local a="$RUN/$base/c$i.txt" b="$RUN/$other/c$i.txt"
        if [ ! -s "$a" ]; then
            log "FAIL: $base/c$i.txt empty"; pass=0
        elif [ ! -s "$b" ]; then
            log "FAIL: $other/c$i.txt empty"; pass=0
        elif cmp -s "$a" "$b"; then
            log "$base vs $other c$i: MATCH ($(wc -c < "$a") bytes)"
        else
            log "FAIL: $base vs $other c$i MISMATCH ($(wc -c < "$a") vs $(wc -c < "$b") bytes)"
            pass=0
        fi
    done
    return $pass
}

if ! run_server A1 20000; then exit 1; fi
run_clients A1
stop_server A1

if ! run_server A2 20000; then exit 1; fi
run_clients A2
stop_server A2

if ! run_server B 700; then exit 1; fi
run_clients B
stop_server B

PASS=1
log "=== A1 vs A2 (cross-restart determinism) ==="
cmp_tag A1 A2 || PASS=0
log "=== A1 vs B (eviction invariance) ==="
cmp_tag A1 B || PASS=0

log "=== swap evidence ==="
SWAP_B=$(grep -c "swapped out" "$RUN/serve-B.log" || true)
log "run B 'swapped out' lines: $SWAP_B (required >= 1)"
[ "$SWAP_B" -lt 1 ] && { log "FAIL: run B shows no swaps"; PASS=0; }

echo
if [ "$PASS" -eq 1 ]; then
    log "RESULT: PASS"
    log "artifacts: $RUN"
    exit 0
else
    log "RESULT: FAIL"
    log "artifacts: $RUN (serve-*.log, A1/, A2/, B/)"
    exit 1
fi
