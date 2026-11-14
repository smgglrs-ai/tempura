#!/usr/bin/env bash
# Smoke test: the non-streaming chat route now goes through the ENGINE
# (same scheduler/pool/swap path as streaming).  With composition-
# invariant decode, the same greedy prompt must produce IDENTICAL text
# via "stream": true and via the buffered non-streaming response.
#
# Usage: bash scripts/nonstream-route-smoke.sh   (on fishtank, repo root)
set -uo pipefail

cd "$(dirname "$0")/.."
MODEL="$HOME/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PORT=8198
RUN=$(mktemp -d /tmp/nonstream-smoke.XXXXXX)
SRV_PID=""

log() { echo "[smoke] $*"; }

cleanup() {
    if [ -n "$SRV_PID" ] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        for _ in $(seq 1 10); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for p in $(pgrep -f "target/release/tempura serve" 2>/dev/null); do
    kill -9 "$p" 2>/dev/null || true
done
sleep 2

log "building..."
cargo build --release -p tempura --features cutile || { log "build failed"; exit 1; }
for d in target/release/build/tempura-kernels-*/out; do
    [ -d "$d" ] || continue
    cp tools/oxide-gemv/oxide_gemv.ptx               "$d/oxide_q2_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-gemv-q4k/oxide_gemv_q4k.ptx       "$d/oxide_q4k_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-gemv-q6k/oxide_gemv_q6k.ptx       "$d/oxide_q6k_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-decode-attn/oxide_decode_attn.ptx "$d/oxide_decode_attn.ptx" 2>/dev/null || true
done

log "starting server"
TEMPURA_KV_POOL_TOKENS=20000 \
    ./target/release/tempura serve \
    --model "$MODEL" --port "$PORT" --device cuda \
    --max-seq-len 8192 --request-timeout-secs 600 \
    > "$RUN/serve.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1 && break
    sleep 1
done
curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1 || { log "server never ready"; tail -20 "$RUN/serve.log"; exit 1; }
log "server ready (pid $SRV_PID)"

PROMPT="Explain in two sentences what a KV cache is in an LLM inference server."
BODY=$(printf '{"model":"default","messages":[{"role":"user","content":"%s"}],"temperature":0.0,"max_tokens":80,"seed":7}' "$PROMPT")

log "non-streaming request (must route through the engine now)"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" -d "$BODY" > "$RUN/nonstream.json"
python3 - "$RUN/nonstream.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
if "choices" not in r:
    print("NON-STREAM FAIL:", json.dumps(r)[:300]); sys.exit(1)
c = r["choices"][0]
print("finish:", c["finish_reason"], "| usage:", r.get("usage"))
open(sys.argv[1] + ".text", "w").write(c["message"]["content"] or "")
EOF
[ -f "$RUN/nonstream.json.text" ] || { log "FAIL: non-streaming response unusable"; cat "$RUN/nonstream.json"; exit 1; }

log "streaming request (same prompt)"
curl -sN "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(printf '{"model":"default","messages":[{"role":"user","content":"%s"}],"temperature":0.0,"max_tokens":80,"seed":7,"stream":true}' "$PROMPT")" \
    > "$RUN/stream.sse"
python3 - "$RUN/stream.sse" <<'EOF'
import json, sys
out = []
for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if payload == "[DONE]":
        break
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        continue
    for ch in obj.get("choices", []):
        d = ch.get("delta") or {}
        if d.get("content"):
            out.append(d["content"])
open(sys.argv[1] + ".text", "w").write("".join(out))
EOF

log "comparison"
if cmp -s "$RUN/nonstream.json.text" "$RUN/stream.sse.text"; then
    log "PASS -- non-streaming and streaming outputs identical ($(wc -c < "$RUN/nonstream.json.text") bytes)"
    log "artifacts: $RUN"
    exit 0
else
    log "FAIL -- outputs differ"
    log "non-stream: $(head -c 200 "$RUN/nonstream.json.text")"
    log "stream:     $(head -c 200 "$RUN/stream.sse.text")"
    exit 1
fi
