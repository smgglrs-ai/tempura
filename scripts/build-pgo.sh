#!/usr/bin/env bash
# Build tempura-serve with Profile-Guided Optimization (PGO).
#
# PGO requires three passes:
#   1. Instrument: compile with profiling instrumentation
#   2. Profile:    run a representative workload to collect .profraw data
#   3. Optimise:   recompile using the merged profile data
#
# Usage:
#   ./scripts/build-pgo.sh [--model /path/to/model.gguf] [--port PORT]
#
# The optimised binary lands at target/release/tempura-serve (same as a
# normal --release build).  Intermediate artefacts are written under
# /tmp/pgo-tempura/ and can be deleted afterwards.
#
# Requirements:
#   - llvm-profdata from the active rustup toolchain (bundled with nightly)
#   - curl (for sending test requests)
#   - A GGUF model file reachable from this machine

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${SCRIPT_DIR}/.."
PGO_DIR="/tmp/pgo-tempura"
PROFDATA="${PGO_DIR}/merged.profdata"

MODEL="${1:-}"
PORT="${2:-18765}"

# Locate the llvm-profdata bundled with the active rustup toolchain so it
# matches the LLVM version used by rustc (currently LLVM 22).
RUSTUP_TC=$(rustup show active-toolchain 2>/dev/null | awk '{print $1}')
LLVM_TOOLS="${HOME}/.rustup/toolchains/${RUSTUP_TC}/lib/rustlib/x86_64-unknown-linux-gnu/bin"
LLVM_PROFDATA="${LLVM_TOOLS}/llvm-profdata"

if [[ ! -x "${LLVM_PROFDATA}" ]]; then
    echo "ERROR: llvm-profdata not found at ${LLVM_PROFDATA}" >&2
    echo "       Install the llvm-tools-preview component:" >&2
    echo "       rustup component add llvm-tools-preview" >&2
    exit 1
fi

if [[ -z "${MODEL}" ]]; then
    # Try the default model path used in development
    MODEL="${HOME}/.local/share/navra/models/qwen3-8b.gguf"
fi

if [[ ! -f "${MODEL}" ]]; then
    echo "ERROR: model not found: ${MODEL}" >&2
    echo "       Pass the path as the first argument: $0 /path/to/model.gguf" >&2
    exit 1
fi

echo "=== PGO build for tempura-serve ==="
echo "Model:   ${MODEL}"
echo "Port:    ${PORT}"
echo "PGO dir: ${PGO_DIR}"
echo ""

mkdir -p "${PGO_DIR}"
rm -f "${PGO_DIR}"/*.profraw

# ──────────────────────────────────────────────────────────────────────────────
# Pass 1: Instrumented build
# ──────────────────────────────────────────────────────────────────────────────
echo "[1/3] Building instrumented binary..."
cd "${WORKSPACE}"

RUSTFLAGS="-Cprofile-generate=${PGO_DIR}" \
    cargo build --release -p tempura-serve

echo "      done."

# ──────────────────────────────────────────────────────────────────────────────
# Pass 2: Collect profiles
# ──────────────────────────────────────────────────────────────────────────────
echo "[2/3] Collecting profile data..."

# Start server in background; store PID for clean shutdown
SERVER_PID_FILE="${PGO_DIR}/server.pid"
./target/release/tempura-serve \
    --model "${MODEL}" \
    --port "${PORT}" \
    &>/tmp/tempura-pgo-server.log &
echo $! > "${SERVER_PID_FILE}"
SERVER_PID=$(cat "${SERVER_PID_FILE}")

echo "      Server started (PID ${SERVER_PID}), waiting for readiness..."

# Wait up to 30 s for the server to become ready
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "      Server ready."
        break
    fi
    sleep 1
    if [[ "${i}" -eq 30 ]]; then
        echo "ERROR: server did not become ready within 30 s" >&2
        kill "${SERVER_PID}" 2>/dev/null || true
        exit 1
    fi
done

# Run a representative decode workload (128 tokens × 5 requests)
echo "      Sending decode workload..."
for i in $(seq 1 5); do
    curl -sf "http://localhost:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "default",
            "prompt": "The quick brown fox jumps over the lazy dog. Explain the historical context of this sentence in detail.",
            "max_tokens": 128,
            "temperature": 0.0
        }' >/dev/null
done

# Run a representative prefill workload (long prompt × 3 requests)
echo "      Sending prefill workload..."
LONG_PROMPT=$(python3 -c "print('word ' * 512)")
for i in $(seq 1 3); do
    curl -sf "http://localhost:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"default\",
            \"prompt\": \"${LONG_PROMPT} Summarise:\",
            \"max_tokens\": 32,
            \"temperature\": 0.0
        }" >/dev/null
done

echo "      Workload complete. Stopping server..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

# ──────────────────────────────────────────────────────────────────────────────
# Merge .profraw files
# ──────────────────────────────────────────────────────────────────────────────
echo "      Merging profiles..."
"${LLVM_PROFDATA}" merge \
    --output="${PROFDATA}" \
    "${PGO_DIR}"/*.profraw

PROFRAW_COUNT=$(ls "${PGO_DIR}"/*.profraw 2>/dev/null | wc -l)
echo "      Merged ${PROFRAW_COUNT} .profraw file(s) into ${PROFDATA}."

# ──────────────────────────────────────────────────────────────────────────────
# Pass 3: Optimised build
# ──────────────────────────────────────────────────────────────────────────────
echo "[3/3] Building PGO-optimised binary..."

RUSTFLAGS="-Cprofile-use=${PROFDATA} -Cllvm-args=-pgo-warn-missing-function" \
    cargo build --release -p tempura-serve

echo "      done."
echo ""
echo "=== PGO build complete ==="
echo "Binary: ${WORKSPACE}/target/release/tempura-serve"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Optional: measure GEMV instruction count
# ──────────────────────────────────────────────────────────────────────────────
BINARY="${WORKSPACE}/target/release/tempura-serve"
GEMV_SYM=$(nm "${BINARY}" 2>/dev/null | grep "ggml_gemv_q4_k_8x8_q8_k" | awk '{print $1}')
GEMV_NEXT=$(nm "${BINARY}" 2>/dev/null | sort | awk "/^${GEMV_SYM}/{found=1;next} found{print \$1;exit}")
if [[ -n "${GEMV_SYM}" && -n "${GEMV_NEXT}" ]]; then
    COUNT=$(objdump -d \
        --start-address="0x${GEMV_SYM}" \
        --stop-address="0x${GEMV_NEXT}" \
        "${BINARY}" | grep -c "^\s\+[0-9a-f]\+:")
    echo "ggml_gemv_q4_k_8x8_q8_k: ${COUNT} instructions (GCC -O3 baseline: 344)"
fi
