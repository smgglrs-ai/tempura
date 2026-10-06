#!/usr/bin/env bash
# PGO build script for tempura-serve.
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
touch server/src/main.rs  # force rebuild
RUSTFLAGS="-Cprofile-generate=$PGO_DIR --cfg profile_gen" \
  cargo build --release -p tempura-serve

echo "=== Step 2: Collect profile (prefill + decode coverage) ==="
fuser -k $PORT/tcp 2>/dev/null || true; sleep 1
LLVM_PROFILE_FILE="$PGO_DIR/tempura-%p.profraw" TEMPURA_THREADS=$THREADS \
  ./target/release/tempura-serve serve --model "$MODEL" \
  --port $PORT --max-seq-len 512 > /tmp/pgo-srv.log 2>&1 &
SRV_PID=$!
sleep 14
./target/release/tempura-bench --host "http://localhost:$PORT" \
  --pp 128 --pp 512 --tg 32 --runs 3 --warmup 1
kill "$SRV_PID"; sleep 5

echo "=== Step 3: Merge profiles ==="
"$PROFDATA" merge -output=/tmp/pgo-merged.profdata "$PGO_DIR"/*.profraw
ls -lh /tmp/pgo-merged.profdata

echo "=== Step 4: PGO-optimized build ==="
RUSTFLAGS="-Cprofile-use=/tmp/pgo-merged.profdata" \
  cargo build --release -p tempura-serve
echo "Done — run the benchmark to verify."
