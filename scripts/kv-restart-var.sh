#!/usr/bin/env bash
# Restart-variance bisect: run the SAME 3-client streaming load against TWO
# fresh server starts and compare per-client text + per-step logits
# checksums.  Env vars set when invoking this script are passed through to
# BOTH servers -- so kernel-path bisect switches (TEMPURA_NO_*) can be A/B'd
# for their effect on cross-restart determinism.
#
# Usage: [ENV=val ...] bash scripts/kv-restart-var.sh
set -uo pipefail

cd "$(dirname "$0")/.."
MODEL="$HOME/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PORT=8196
MAX_TOKENS="${TEMPURA_KV_RV_TOKENS:-200}"
CLIENTS="${TEMPURA_KV_RV_CLIENTS:-3}"
RUN=$(mktemp -d /tmp/kv-restart-var.XXXXXX)
SRV_PID=""

log() { echo "[restart-var] $*"; }

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
    for _ in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

PROMPTS=(
    "Explain in detail how paged attention works in an LLM inference server, covering page tables, block sizes, and why it helps memory utilization. Write at least 400 words."
    "Write a long technical explanation of the differences between prefill and decode phases in transformer inference, including KV cache behavior and batching strategies. Write at least 400 words."
    "Describe the gated delta rule (linear attention) architecture in depth: how the recurrent and chunked forms work, and when hybrid attention/delta layers are preferable. Write at least 400 words."
)

run_server() {
    local tag="$1"
    log "starting server $tag (env: $(env | grep -o '^TEMPURA_[A-Z0-9_]*=[^ ]*' | tr '\n' ' '))"
    TEMPURA_KV_POOL_TOKENS=20000 \
    TEMPURA_ENGINE_DEBUG_TICKS=1 \
        ./target/release/tempura serve \
        --model "$MODEL" --port "$PORT" --device cuda \
        --max-seq-len 24576 \
        --request-timeout-secs 600 \
        > "$RUN/serve-$tag.log" 2>&1 &
    SRV_PID=$!
    wait_ready || { log "server $tag never ready"; tail -30 "$RUN/serve-$tag.log"; exit 1; }
    log "server $tag ready (pid $SRV_PID)"
}

run_clients() {
    local tag="$1" i=0 p
    mkdir -p "$RUN/$tag"
    local pids=()
    for p in "${PROMPTS[@]:0:$CLIENTS}"; do
        i=$((i + 1))
        python3 scripts/kv-swap-ab-client.py \
            "http://127.0.0.1:$PORT/v1/chat/completions" \
            "$p" "$RUN/$tag/c$i.txt" "$MAX_TOKENS" \
            > "$RUN/$tag/c$i.stderr" 2>&1 &
        pids+=($!)
        sleep 1
    done
    wait "${pids[@]}" || true
}

stop_server() {
    kill "$SRV_PID" 2>/dev/null || true
    for _ in $(seq 1 15); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
    sleep 2
    pgrep -f "target/release/tempura serve" >/dev/null 2>&1 && kill_leftovers
    return 0
}

[ -f "$MODEL" ] || { log "ERROR: model not found: $MODEL"; exit 1; }
kill_leftovers

log "building server (cutile)..."
cargo build --release -p tempura --features cutile || { log "build failed"; exit 1; }
for d in target/release/build/tempura-kernels-*/out; do
    [ -d "$d" ] || continue
    cp tools/oxide-gemv/oxide_gemv.ptx               "$d/oxide_q2_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-gemv-q4k/oxide_gemv_q4k.ptx       "$d/oxide_q4k_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-gemv-q6k/oxide_gemv_q6k.ptx       "$d/oxide_q6k_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-decode-attn/oxide_decode_attn.ptx "$d/oxide_decode_attn.ptx" 2>/dev/null || true
done

run_server R1
run_clients R1
stop_server R1
run_server R2
run_clients R2
stop_server R2

PASS=1
log "=== text comparison R1 vs R2 ==="
for i in $(seq 1 "$CLIENTS"); do
    if cmp -s "$RUN/R1/c$i.txt" "$RUN/R2/c$i.txt"; then
        log "c$i: MATCH"
    else
        log "c$i: MISMATCH"
        PASS=0
    fi
done
log "=== first divergent decode step per session (logits-cs) ==="
for f in "$RUN"/serve-R1.log; do :; done
python3 - "$RUN" <<'EOF'
import re, sys, glob
run = sys.argv[1]
def sessions(path):
    order = []
    per = {}
    for line in open(path, errors="replace"):
        m = re.match(r"\[logits-cs\] (\S+) step=(\d+) cs=([0-9a-f]+)", line)
        if not m:
            continue
        s, step, cs = m.groups()
        if s not in per:
            order.append(s)
            per[s] = {}
        per[s][int(step)] = cs
    return order, per
o1, p1 = sessions(glob.glob(run + "/serve-R1.log")[0])
o2, p2 = sessions(glob.glob(run + "/serve-R2.log")[0])
ok = True
for idx in range(min(len(o1), len(o2))):
    s1, s2 = o1[idx], o2[idx]
    d1, d2 = p1[s1], p2[s2]
    first = None
    for step in sorted(d1):
        if d2.get(step) != d1[step]:
            first = step
            break
    if first is None and len(d1) == len(d2):
        print(f"  session #{idx+1}: IDENTICAL ({len(d1)} steps)")
    else:
        ok = False
        print(f"  session #{idx+1}: first diff at step {first} (of {len(d1)}/{len(d2)} steps)")
sys.exit(0 if ok else 1)
EOF
CS_RC=$?
log "artifacts: $RUN"
if [ "$PASS" -eq 1 ] && [ "$CS_RC" -eq 0 ]; then
    log "RESULT: PASS -- no cross-restart variance"
    exit 0
else
    log "RESULT: FAIL -- cross-restart variance present"
    exit 1
fi
