#!/usr/bin/env bash
# PGO build script for tempura.
# Requires: rustup component add llvm-tools-preview (for matching llvm-profdata)
set -euo pipefail

MODEL=${1:-/home/fdupont/.local/share/navra/models/qwen3-8b.gguf}
PORT=8099
THREADS=16
PGO_DIR=/tmp/pgo-data
PROFDATA=$(find ~/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu \
  -name "llvm-profdata" 2>/dev/null | head -1)

echo "=== Step 1: Instrumented build (profile_gen cfg enabled) ==="
mkdir -p "$PGO_DIR" && rm -f "$PGO_DIR"/*.profraw
touch server/src/lib.rs cli/src/main.rs  # force rebuild
RUSTFLAGS="-Cprofile-generate=$PGO_DIR --cfg profile_gen" \
  cargo build --release -p tempura

echo "=== Step 2: Collect profile (prefill + decode coverage) ==="
# Note: instrumented binary is ~10x slower -- requests may take 5-10 minutes each.
# We use curl with a 30-minute timeout and send a mix of prefill + decode requests.
fuser -k $PORT/tcp 2>/dev/null || true; sleep 1
LLVM_PROFILE_FILE="$PGO_DIR/tempura-%p.profraw" TEMPURA_THREADS=$THREADS \
  ./target/release/tempura serve --model "$MODEL" \
  --port $PORT --max-seq-len 512 > /tmp/pgo-srv.log 2>&1 &
SRV_PID=$!

# Wait for model load (repacking takes ~3s uninstrumented, ~30s instrumented)
echo "Waiting for model load (may take up to 60s under instrumentation)..."
for i in $(seq 1 60); do
  curl -sf --max-time 5 http://localhost:$PORT/v1/models >/dev/null 2>&1 && break
  sleep 2
done

echo "Server ready. Sending profile workload (this will be slow)..."
CURL="curl -s -X POST http://localhost:$PORT/v1/completions \
  -H Content-Type:application/json \
  --max-time 1800"

# Prefill coverage: 5 requests (fast even instrumented)
for prompt in "The capital of France" "In the beginning" "Machine learning is" "Rust programming language" "Hello world"; do
  $CURL -d "{\"model\":\"default\",\"prompt\":\"$prompt\",\"max_tokens\":1}" | python3 -c "import sys,json; d=json.load(sys.stdin); print('pp ok:', d['choices'][0]['text'][:10])" 2>/dev/null || true
done

# Decode coverage: 2 requests with 16 tokens each (slow under instrumentation)
echo "Running decode requests (expect 5-15 minutes each under instrumentation)..."
for prompt in "Paris is the capital of France and" "The quick brown fox jumps over"; do
  $CURL -d "{\"model\":\"default\",\"prompt\":\"$prompt\",\"max_tokens\":16}" | python3 -c "import sys,json; d=json.load(sys.stdin); print('tg ok:', d['choices'][0]['text'][:20])" 2>/dev/null || true
done

kill "$SRV_PID"; sleep 5

echo "=== Step 3: Merge profiles ==="
"$PROFDATA" merge -output=/tmp/pgo-merged.profdata "$PGO_DIR"/*.profraw
ls -lh /tmp/pgo-merged.profdata

echo "=== Step 4: PGO-optimized build ==="
RUSTFLAGS="-Cprofile-use=/tmp/pgo-merged.profdata" \
  cargo build --release -p tempura
echo "Done -- run the benchmark to verify."
