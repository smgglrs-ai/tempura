#!/usr/bin/env bash
# THE EAGLE-3 speculative-decoding equivalence + throughput gate (SOLO
# scope -- the structural accounting says speculation cannot win at
# serving batch sizes, so EAGLE runs only when exactly one greedy
# decode-ready session exists; multi-client batches keep plain decode).
#
# The equivalence property (same shape as spec-equivalence.sh): greedy
# EAGLE-3 draft+verify emits EXACTLY the token sequence of plain greedy
# decode -- by construction the verify is the target's own argmax over
# spec_row_views rows (the same v1 machinery), and the head's numerics
# only influence acceptance length.  This script makes it executable:
#
#   run OFF: server WITHOUT --eagle-head, 1 solo client (greedy)
#   run ON : the same server + --eagle-head (RedHatAI speculators
#            EAGLE-3 head for Llama-3.1-8B-Instruct)
#
# PASS = the solo client's ON output is byte-identical to OFF.
# Reports solo tok/s OFF vs ON and the acceptance telemetry.
#
# Usage: bash scripts/eagle-equivalence.sh          (run ON fishtank)
#        EAGLE=... MODEL=... bash scripts/eagle-equivalence.sh
set -uo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-$HOME/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
EAGLE="${EAGLE:-$HOME/.cache/tempura/models/eagle3-llama3.1-8b}"
PORT="${PORT:-8198}"
SOLO_TOKENS="${SOLO_TOKENS:-400}"
RUN=$(mktemp -d /tmp/eagle-equiv.XXXXXX)
SRV_PID=""

log() { echo "[eagle-equiv] $*"; }

cleanup() {
    if [ -n "$SRV_PID" ] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

kill_leftovers() {
    for p in $(pgrep -f "target/release/tempura serve" 2>/dev/null); do
        kill "$p" 2>/dev/null || true
    done
    sleep 2
    for p in $(pgrep -f "target/release/tempura serve" 2>/dev/null); do
        kill -9 "$p" 2>/dev/null || true
    done
}

wait_ready() {
    for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

SOLO_PROMPT="Explain, step by step, how the softmax function works and why subtracting the max improves numerical stability. Be precise about the math."

[ -f "$MODEL" ] || { log "ERROR: target model not found: $MODEL"; exit 1; }
[ -f "$EAGLE/model.safetensors" ] || { log "ERROR: EAGLE head not found: $EAGLE (config.json + model.safetensors from RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3)"; exit 1; }

run_solo() {
    local tag="$1" t0 t1
    mkdir -p "$RUN/$tag"
    t0=$(date +%s.%N)
    python3 scripts/kv-swap-ab-client.py \
        "http://127.0.0.1:$PORT/v1/chat/completions" \
        "$SOLO_PROMPT" "$RUN/$tag/solo.txt" "$SOLO_TOKENS" \
        > "$RUN/$tag/solo.stderr" 2>&1
    t1=$(date +%s.%N)
    echo "$SOLO_TOKENS $(echo "$t1 - t0" | bc)" > "$RUN/$tag/solo.time"
    cat "$RUN/$tag/solo.stderr"
}

run_server() {
    local tag="$1"; shift
    log "starting server $tag: $*"
    TEMPURA_ENGINE_DEBUG_TICKS=1 \
        ./target/release/tempura serve \
        --model "$MODEL" --port "$PORT" --device cuda \
        --max-seq-len 24576 \
        --request-timeout-secs 2400 \
        "$@" \
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
}

tokrate() {
    read -r toks secs < "$1"
    awk -v t="$toks" -v s="$secs" 'BEGIN { if (s <= 0) { print "n/a" } else { printf "%.1f tok/s (%.1fs)", t / s, s } }'
}

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

# -- run OFF (plain greedy): the reference output + solo throughput -------
if ! run_server off; then exit 1; fi
run_solo off
stop_server

# -- run ON (--eagle-head): must reproduce OFF byte-for-byte --------------
if ! run_server on --eagle-head "$EAGLE"; then exit 1; fi
run_solo on
stop_server

# -- compare --------------------------------------------------------------
pass=1
a="$RUN/off/solo.txt" b="$RUN/on/solo.txt"
if [ ! -s "$a" ]; then
    log "FAIL: off/solo.txt empty"; pass=0
elif [ ! -s "$b" ]; then
    log "FAIL: on/solo.txt empty"; pass=0
elif cmp -s "$a" "$b"; then
    log "OFF vs ON solo: MATCH ($(wc -c < "$a") bytes)"
else
    log "FAIL: OFF vs ON solo MISMATCH ($(wc -c < "$a") vs $(wc -c < "$b") bytes)"
    pass=0
fi

log "solo OFF: $(tokrate "$RUN/off/solo.time")"
log "solo ON : $(tokrate "$RUN/on/solo.time")"
log "spec telemetry (run ON):"
grep -o "spec: cycles=.*" "$RUN/serve-on.log" | tail -1 || \
    log "  (no spec tick line found -- check the serve log)"
log "[eagle] head lines (run ON):"
grep -e "\[eagle\]" "$RUN/serve-on.log" | tail -3 || true

if [ "$pass" -eq 1 ]; then
    log "PASS: EAGLE-3 solo speculation is output-equivalent to plain greedy decode"
    exit 0
else
    log "FAIL: outputs diverged"
    exit 1
fi
