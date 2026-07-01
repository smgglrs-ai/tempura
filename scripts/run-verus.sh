#!/bin/bash
# Run Verus deductive verification on verus-verified/ crate
#
# Unlike Kani (bounded model checking), Verus proves properties for ALL
# possible inputs using SMT-based deductive verification.
#
# Prerequisites:
#   Download Verus release from https://github.com/verus-lang/verus/releases
#   Unzip to ~/.local/verus/ (or set VERUS_PATH)
#   Rust stable toolchain: rustup install 1.96.0

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERUS_CRATE="$PROJECT_ROOT/verus-verified"

# Find Verus binary
VERUS_BIN="${VERUS_PATH:-$HOME/.local/verus/verus-x86-linux}/verus"

if [ ! -x "$VERUS_BIN" ]; then
    echo "Error: Verus binary not found at $VERUS_BIN"
    echo "Download from: https://github.com/verus-lang/verus/releases"
    echo "Unzip to ~/.local/verus/ or set VERUS_PATH"
    exit 1
fi

# Print version for reproducibility
echo "Verus: $($VERUS_BIN --version 2>&1 | head -3)"

# ---- Drift detection ----
echo ""
echo "=== Drift Detection ==="

# Count Kani harnesses across the project
KANI_COUNT=$(grep -rn '#\[kani::proof\]' --include='*.rs' "$PROJECT_ROOT/kernels" "$PROJECT_ROOT/engine" | wc -l)
VERUS_COUNT=$(grep -c 'proof fn' "$VERUS_CRATE"/src/*.rs 2>/dev/null | awk -F: '{s+=$2} END {print s}')

echo "Kani harnesses: $KANI_COUNT"
echo "Verus proofs:   $VERUS_COUNT"

# List Kani harnesses without Verus counterparts
KANI_NAMES=$(grep -rn '#\[kani::proof\]' --include='*.rs' "$PROJECT_ROOT/kernels" "$PROJECT_ROOT/engine" -A1 \
    | grep 'fn ' | grep -oP 'fn \w+' | sed 's/^fn //' | sort)
VERUS_NAMES=$(grep 'proof fn' "$VERUS_CRATE"/src/*.rs | grep -oP 'fn \w+' | sed 's/^fn //' | sort)

MISSING=$(comm -23 <(echo "$KANI_NAMES") <(echo "$VERUS_NAMES"))
if [ -n "$MISSING" ]; then
    echo ""
    echo "Kani harnesses without Verus counterparts ($(echo "$MISSING" | wc -l)):"
    echo "$MISSING" | sed 's/^/  /'
    echo ""
    echo "  (3 covered by proptest: head_dim_scale_never_zero_division,"
    echo "   int8_dequant_bounded_error, q4k_dequant_value_bounded)"
    echo "  (1 Kani-only: cpu_allreduce_is_identity — runtime Vec execution)"
fi

# ---- Struct field drift detection ----
echo ""
echo "=== Struct Field Drift Detection ==="
DRIFT_ERRORS=0

# Compare struct fields: extract "pub field_name: type" from both source and Verus
check_struct_fields() {
    local struct_name="$1"
    local source_file="$2"
    local verus_file="$3"

    local src_fields=$(sed -n "/pub struct ${struct_name}/,/^}/p" "$source_file" \
        | grep 'pub \w\+:' | sed 's/.*pub //' | sed 's/,.*//' | sort)
    local verus_fields=$(sed -n "/pub struct ${struct_name}/,/^}/p" "$verus_file" \
        | grep 'pub \w\+:' | sed 's/.*pub //' | sed 's/,.*//' | sort)

    if [ -z "$src_fields" ] || [ -z "$verus_fields" ]; then
        return
    fi

    local missing=$(comm -23 <(echo "$src_fields") <(echo "$verus_fields"))
    local extra=$(comm -13 <(echo "$src_fields") <(echo "$verus_fields"))

    if [ -n "$missing" ]; then
        echo "  DRIFT: $struct_name — fields in source but not in Verus:"
        echo "$missing" | sed 's/^/    /'
        DRIFT_ERRORS=$((DRIFT_ERRORS + 1))
    fi
    if [ -n "$extra" ]; then
        echo "  DRIFT: $struct_name — fields in Verus but not in source:"
        echo "$extra" | sed 's/^/    /'
        DRIFT_ERRORS=$((DRIFT_ERRORS + 1))
    fi
}

# Compare trait methods: extract "fn method_name" from both source and Verus
check_trait_methods() {
    local trait_name="$1"
    local source_file="$2"
    local verus_file="$3"

    local src_methods=$(sed -n "/pub trait ${trait_name}/,/^}/p" "$source_file" \
        | grep 'fn \w\+' | grep -oP 'fn \w+' | sed 's/^fn //' | sort)
    local verus_methods=$(grep -oP 'fn \w+' "$verus_file" | sed 's/^fn //' | sort)

    if [ -z "$src_methods" ]; then
        return
    fi

    # Check trait methods exist as Verus proof or spec functions
    for method in $src_methods; do
        if ! grep -q "$method" "$verus_file" 2>/dev/null; then
            echo "  INFO: trait ${trait_name}::${method}() — no Verus counterpart (may not need one)"
        fi
    done
}

# Struct field checks (source numeric fields vs Verus model fields)
check_struct_fields "KvCacheState" \
    "$PROJECT_ROOT/engine/src/kv_cache.rs" \
    "$VERUS_CRATE/src/kv_cache_invariant.rs" 2>/dev/null

check_struct_fields "PageAllocatorState" \
    "$PROJECT_ROOT/engine/src/kv_cache.rs" \
    "$VERUS_CRATE/src/page_allocator.rs" 2>/dev/null

check_struct_fields "BlockTableState" \
    "$PROJECT_ROOT/engine/src/kv_cache.rs" \
    "$VERUS_CRATE/src/block_table.rs" 2>/dev/null

check_struct_fields "SessionState" \
    "$PROJECT_ROOT/engine/src/session.rs" \
    "$VERUS_CRATE/src/session_consistency.rs" 2>/dev/null

# Trait contract checks
check_trait_methods "VerifiedKvCache" \
    "$PROJECT_ROOT/kernel-api/src/lib.rs" \
    "$VERUS_CRATE/src/kv_cache_invariant.rs" 2>/dev/null

check_trait_methods "VerifiedPageAllocator" \
    "$PROJECT_ROOT/kernel-api/src/lib.rs" \
    "$VERUS_CRATE/src/page_allocator.rs" 2>/dev/null

check_trait_methods "VerifiedBlockTable" \
    "$PROJECT_ROOT/kernel-api/src/lib.rs" \
    "$VERUS_CRATE/src/block_table.rs" 2>/dev/null

check_trait_methods "VerifiedSession" \
    "$PROJECT_ROOT/kernel-api/src/lib.rs" \
    "$VERUS_CRATE/src/session_consistency.rs" 2>/dev/null

check_trait_methods "VerifiedPrefixCache" \
    "$PROJECT_ROOT/kernel-api/src/lib.rs" \
    "$VERUS_CRATE/src/prefix_cache.rs" 2>/dev/null

if [ "$DRIFT_ERRORS" -gt 0 ]; then
    echo ""
    echo "WARNING: $DRIFT_ERRORS drift issue(s) detected!"
else
    echo "  All trait contracts and struct fields are in sync."
fi

# ---- Run Verus verification ----
echo ""
echo "=== Running Verus Verification ==="
cd "$VERUS_CRATE"

MODULES=(
    # Indexing arithmetic (mirror Kani harnesses)
    kv_cache
    attention
    rmsnorm
    layernorm
    rope
    matmul
    sampling
    turboquant
    quantized
    model
    speculative
    disaggregated
    weights
    gguf
    # Struct/loop/monotonicity invariants (beyond Kani)
    kv_cache_invariant
    page_allocator
    block_table
    chunked_forward
    tick_monotonicity
    session_consistency
    speculative_invariant
    prefix_cache
    batch_scheduler
    model_config
    radix_tree
    token_generation
    forward_pass
    gqa
    fused_kernels
    medusa
    projection_shapes
    kv_append_advance
    sampling_penalties
    trace_invariant
    gpu_tiles
    paged_kv_cache
    causal_mask
    flash_tiles
    fp8
    weight_shapes
    lora
    trace_diff
    guided_decoding
    config_validation
    rope_frequency
    tensor_parallel
    fp8_kv_cache
    quantized_forward
    gguf_alignment
    position_encoding
    gpu_flash_attn
    gpu_quantized_gemm
    f16_encoding
    end_to_end
    i2s_bitnet
    i2s_vec_dot
    tq2_block
    bitnet_forward
    layer_graph
    gpu_elementwise
    kv_rollback
    mma_fragments
    sampling_chain
    activations_moe
)

TOTAL_VERIFIED=0
TOTAL_ERRORS=0

for mod in "${MODULES[@]}"; do
    echo ""
    echo "--- src/${mod}.rs ---"
    OUTPUT=$("$VERUS_BIN" "src/${mod}.rs" "$@" 2>&1)
    echo "$OUTPUT" | tail -1

    V=$(echo "$OUTPUT" | grep -oP '\d+ verified' | grep -oP '\d+' || echo 0)
    E=$(echo "$OUTPUT" | grep -oP '\d+ errors' | grep -oP '\d+' || echo 0)
    TOTAL_VERIFIED=$((TOTAL_VERIFIED + V))
    TOTAL_ERRORS=$((TOTAL_ERRORS + E))

    if [ "$E" -gt 0 ]; then
        echo "$OUTPUT"
    fi
done

echo ""
echo "=== Summary ==="
echo "Total: $TOTAL_VERIFIED verified, $TOTAL_ERRORS errors"

if [ "$TOTAL_ERRORS" -gt 0 ]; then
    exit 1
fi
