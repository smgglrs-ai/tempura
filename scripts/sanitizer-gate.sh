#!/usr/bin/env bash
# Sanitizer gate: compute-sanitizer over the serving-path GEMV kernels.
#
# Runs memcheck on bench-probe-q4k-batch, bench-probe-q6k-batch and
# bench-decode-ops (the probes drive the sm120 kpar8 / kpar8_db / batch
# GEMV kernels in isolation on random native-GGUF blocks; bench-decode-ops
# covers the wider decode kernel set on real weights), plus racecheck and
# initcheck on the two probes. FAILS on any sanitizer error OR any
# application-level FAIL -- a kernel that passes its numerics probe while
# reading out of bounds is not green (the 2026-11-17 kpar8 dual_block
# OOB reads passed every output gate for weeks; benign OOB reads can fault
# at allocation boundaries -- the historical "error 700 at ~896 blocks"
# class -- or read retired memory under cuda-oxide's allocator).
#
# Usage: bash scripts/sanitizer-gate.sh   (on fishtank, repo root)
#   COMPUTE_SANITIZER=/path/to/compute-sanitizer  (default: first found in
#   PATH or /usr/local/cuda/bin)
set -uo pipefail

cd "$(dirname "$0")/.."

SAN="${COMPUTE_SANITIZER:-$(command -v compute-sanitizer || true)}"
if [ -z "$SAN" ] && [ -x /usr/local/cuda/bin/compute-sanitizer ]; then
    SAN=/usr/local/cuda/bin/compute-sanitizer
fi
if [ -z "$SAN" ]; then
    echo "[sanitizer-gate] compute-sanitizer not found" >&2
    exit 1
fi

RUN=$(mktemp -d /tmp/sanitizer-gate.XXXXXX)
trap 'rm -rf "$RUN"' EXIT

log() { echo "[sanitizer-gate] $*"; }

log "building bench binaries..."
cargo build --release -p tempura-bench-gpu \
    --bin bench-probe-q4k-batch --bin bench-probe-q6k-batch --bin bench-decode-ops \
    || { log "build failed"; exit 1; }

fail=0

# run TOOL BIN LOGNAME -- fails on app failure or any sanitizer error.
# memcheck/initcheck report "ERROR SUMMARY: N errors"; racecheck reports
# "RACECHECK SUMMARY: N hazards displayed".
run() {
    local tool="$1"
    local bin="$2"
    local name="$3"
    local log="$RUN/$name.log"
    log "$tool: $bin -> $log"
    "$SAN" --tool "$tool" --print-limit 100000 \
        "./target/release/$bin" > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FAIL: $bin exited $rc under $tool"
        fail=1
    fi
    local pattern summary
    if [ "$tool" = racecheck ]; then
        pattern="RACECHECK SUMMARY: 0 hazards"
    else
        pattern="ERROR SUMMARY: 0 errors"
    fi
    summary=$(grep -h "SUMMARY" "$log" | tail -1)
    if ! grep -q "$pattern" "$log"; then
        echo "  FAIL: $bin sanitizer errors under $tool ($summary)"
        fail=1
    else
        echo "  PASS ($summary)"
    fi
}

# memcheck: everything (OOB reads/writes).
run memcheck bench-probe-q4k-batch memcheck-q4k
run memcheck bench-probe-q6k-batch memcheck-q6k
run memcheck bench-decode-ops memcheck-decode-ops

# racecheck on the q6k probe with the documented handoff allowlist (below).
run_racecheck_q6k_with_handoff_allowlist() {
    local log="$RUN/racecheck-q6k.log"
    log "racecheck: bench-probe-q6k-batch -> $log"
    "$SAN" --tool racecheck --print-limit 100000 \
        ./target/release/bench-probe-q6k-batch > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FAIL: bench-probe-q6k-batch exited $rc under racecheck"
        fail=1
    fi
    local other
    other=$(grep "Race reported between" "$log" \
        | grep -vF "q6k_gemv_sm120_kpar8_db_batch+0x36c0" \
        | grep -vF "q6k_gemv_sm120_kpar8_db+0x17a0" || true)
    if [ -n "$other" ]; then
        echo "  FAIL: racecheck hazards outside the documented db-handoff pattern:"
        echo "$other" | head -5
        fail=1
    else
        echo "  PASS (only the documented kpar8_db release/acquire handoff hazards)"
    fi
}

# racecheck + initcheck: the probes are cheap; skip via SAN_GATE_FAST=1.
if [ -z "${SAN_GATE_FAST:-}" ]; then
    for tool in racecheck initcheck; do
        run "$tool" bench-probe-q4k-batch "$tool-q4k"
        if [ "$tool" = racecheck ]; then
            # q6k racecheck KNOWN FALSE POSITIVE: the kpar8_db / kpar8_db_batch
            # cross-warp reduction uses a release/acquire shared-memory counter
            # handoff (st.shared partials -> red.release.cta counter bump ->
            # ld.acquire.cta poll -> thread-0 reads). racecheck does not model
            # scoped-atomics ordering and reports the handoff as Write/Read
            # hazards between exactly those two kernels. The pattern is the
            # FIX for the real 2026-10-29 cross-warp race (plain store +
            # membar and mbarrier variants raced on this hw -- see the PTX
            # header in kernels/src/q6k_gemv_v4_ptx.rs) and is determinism-
            # gated (bench-decode-ops "q6k sm120_kpar8 determinism: 0/8").
            # Any hazard in ANY other kernel is a real failure.
            run_racecheck_q6k_with_handoff_allowlist
        else
            run "$tool" bench-probe-q6k-batch "$tool-q6k"
        fi
    done
fi

if [ $fail -ne 0 ]; then
    echo "=== sanitizer-gate: FAIL (logs in $RUN) ==="
    exit 1
fi
echo "=== sanitizer-gate: PASS ==="
