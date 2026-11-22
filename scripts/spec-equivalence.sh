#!/usr/bin/env bash
# THE speculative-decoding equivalence gate (greedy v1).
#
# The equivalence property: greedy speculative decoding emits EXACTLY the
# token sequence of plain greedy decode -- provably, by construction (the
# verify forward runs the same decode-class kernels, per-row bit-identical
# to plain decode; the draft only influences acceptance length). This
# script makes the proof executable:
#
#   run OFF: server WITHOUT --draft-model, 1 solo client + 6 concurrent
#            streaming clients, greedy, outputs captured per client
#   run ON : the same server invocation + --draft-model (a Llama-3.2-1B
#            drafting for the Llama-3.1-8B target)
#
# PASS = every client's ON output is byte-identical to its OFF output.
# The script also reports the acceptance-length telemetry (mean emitted
# per verify cycle, from TEMPURA_ENGINE_DEBUG_TICKS) and the wall-time
# per phase so the spec-on/off throughput ratio is visible.
#
# Usage: bash scripts/spec-equivalence.sh          (run ON fishtank)
#        DRAFT=... MODEL=... bash scripts/spec-equivalence.sh
set -uo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-$HOME/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
DRAFT="${DRAFT:-$HOME/.cache/tempura/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
PORT="${PORT:-8199}"
CLIENTS=6
MAX_TOKENS="${MAX_TOKENS:-256}"
SOLO_TOKENS="${SOLO_TOKENS:-400}"
RUN=$(mktemp -d /tmp/spec-equiv.XXXXXX)
SRV_PID=""

log() { echo "[spec-equiv] $*"; }

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
    for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

# One prompt for the solo client, six for the concurrent phase.
SOLO_PROMPT="Explain, step by step, how the softmax function works and why subtracting the max improves numerical stability. Be precise about the math."
PROMPTS=(
    "Explain in detail how paged attention works in an LLM inference server, covering page tables, block sizes, and why it helps memory utilization. Write at least 400 words."
    "Write a long technical explanation of the differences between prefill and decode phases in transformer inference, including KV cache behavior and batching strategies. Write at least 400 words."
    "Describe the gated delta rule (linear attention) architecture in depth: how the recurrent and chunked forms work, and when hybrid attention/delta layers are preferable. Write at least 400 words."
    "Explain speculative decoding in LLM inference: how draft-model proposal and target verification work, why greedy speculative decoding is output-equivalent to plain greedy decoding, and where the speedup comes from. Write at least 400 words."
    "Write a detailed explanation of CUDA graphs: what problem launch overhead solves, how capture and replay work, and the constraints captured kernels must satisfy. Write at least 400 words."
    "Explain quantization for LLM weights: Q4_K_M block layout, importance matrices, and the accuracy/memory trade-off. Write at least 400 words."
)

[ -f "$MODEL" ] || { log "ERROR: target model not found: $MODEL"; exit 1; }

run_clients() {
    local tag="$1" i=0 p t0 t1
    mkdir -p "$RUN/$tag"
    # Solo client (timed).
    t0=$(date +%s.%N)
    python3 scripts/kv-swap-ab-client.py \
        "http://127.0.0.1:$PORT/v1/chat/completions" \
        "$SOLO_PROMPT" "$RUN/$tag/solo.txt" "$SOLO_TOKENS" \
        > "$RUN/$tag/solo.stderr" 2>&1
    t1=$(date +%s.%N)
    echo "$SOLO_TOKENS $(echo "$t1 - $t0" | bc)" > "$RUN/$tag/solo.time"
    # Concurrent clients.
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
    t0=$(date +%s.%N)
    wait "${pids[@]}" || true
    t1=$(date +%s.%N)
    echo "$((MAX_TOKENS * CLIENTS)) $(echo "$t1 - $t0" | bc)" > "$RUN/$tag/multi.time"
    cat "$RUN/$tag"/*.stderr
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
    if pgrep -f "target/release/tempura serve" >/dev/null 2>&1; then
        kill_leftovers
    fi
}

tokrate() { # file with "tokens seconds" -> "tok/s (seconds s)"
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

# -- run OFF (no draft): the reference outputs + plain throughput --------
if ! run_server off; then exit 1; fi
run_clients off
stop_server

# -- run ON (--draft-model): must reproduce OFF byte-for-byte ------------
if [ ! -f "$DRAFT" ]; then
    log "ERROR: draft model not found: $DRAFT (pull Llama-3.2-1B-Instruct-Q4_K_M.gguf first)"
    exit 1
fi
if ! run_server on --draft-model "$DRAFT"; then exit 1; fi
run_clients on
stop_server

# -- compare --------------------------------------------------------------
pass=1
for f in solo $(seq 1 $CLIENTS | sed 's/^/c/'); do
    a="$RUN/off/$f.txt" b="$RUN/on/$f.txt"
    if [ ! -s "$a" ]; then
        log "FAIL: off/$f.txt empty"; pass=0
    elif [ ! -s "$b" ]; then
        log "FAIL: on/$f.txt empty"; pass=0
    elif cmp -s "$a" "$b"; then
        log "OFF vs ON $f: MATCH ($(wc -c < "$a") bytes)"
    else
        log "FAIL: OFF vs ON $f MISMATCH ($(wc -c < "$a") vs $(wc -c < "$b") bytes)"
        pass=0
    fi
done

# -- throughput + acceptance telemetry ------------------------------------
log "solo  OFF: $(tokrate "$RUN/off/solo.time")"
log "solo  ON : $(tokrate "$RUN/on/solo.time")"
log "multi OFF: $(tokrate "$RUN/off/multi.time")"
log "multi ON : $(tokrate "$RUN/on/multi.time")"
log "spec telemetry (run ON):"
grep -o "spec: cycles=.*" "$RUN/serve-on.log" | tail -1 || \
    log "  (no spec tick line found -- check the serve log)"

if [ "$pass" -eq 1 ]; then
    log "PASS: speculative decoding is output-equivalent to plain greedy decode"
    exit 0
else
    log "FAIL: outputs diverged"
    exit 1
fi
