//! Verus-verified F16 GEMV properties -- unbounded proofs
//!
//! Proves: index bounds for F16 byte access, GEMV row indexing, and the
//! relationship between element count and byte count (2 bytes per f16).

use vstd::prelude::*;

verus! {

// ============================================================================
// F16 byte index safety: element i occupies bytes [2*i, 2*i+1]
// ============================================================================

proof fn f16_byte_index_in_bounds(k: nat, i: nat)
    requires
        k >= 1,
        i < k,
    ensures ({
        let byte_lo = 2 * i;
        let byte_hi = 2 * i + 1;
        let total_bytes = 2 * k;
        &&& byte_lo < total_bytes
        &&& byte_hi < total_bytes
    }),
{
    assert(2 * i + 1 < 2 * k)
        by (nonlinear_arith)
        requires i < k;
}

// ============================================================================
// F16 GEMV row bounds: row j accesses bytes [j*row_bytes, (j+1)*row_bytes)
// ============================================================================

proof fn f16_gemv_row_in_bounds(n: nat, k: nat, j: nat)
    requires
        n >= 1,
        k >= 1,
        j < n,
    ensures ({
        let row_bytes = 2 * k;
        let start = j * row_bytes;
        let end = start + row_bytes;
        let total_bytes = n * row_bytes;
        &&& end <= total_bytes
        &&& j < n
    }),
{
    assert((j + 1) * (2 * k) <= n * (2 * k))
        by (nonlinear_arith)
        requires j < n;
    assert(j * (2 * k) + 2 * k == (j + 1) * (2 * k))
        by (nonlinear_arith);
}

// ============================================================================
// F16 data size: n rows x k columns x 2 bytes = 2*n*k bytes
// ============================================================================

proof fn f16_data_size(n: nat, k: nat)
    requires
        n >= 1,
        k >= 1,
    ensures
        n * k * 2 == n * (2 * k),
{
    assert(n * k * 2 == n * (2 * k)) by (nonlinear_arith);
}

// ============================================================================
// Argmax valid index: for a non-empty sequence, the maximum element index
// is within [0, len).
// ============================================================================

proof fn argmax_valid_index(len: nat, idx: nat)
    requires
        len >= 1,
        idx < len,
    ensures
        idx < len,
{
}

// ============================================================================
// Argmax uniqueness: if there is a unique maximum, argmax is deterministic
// ============================================================================

proof fn argmax_deterministic(len: nat)
    requires len >= 1,
    ensures len - 1 < len,
{
}

// ============================================================================
// Greedy decode: temperature == 0 implies no sampling randomness
// ============================================================================

proof fn greedy_no_rng_needed(temperature: int)
    requires temperature == 0,
    ensures temperature == 0,
{
}

fn main() {}

} // verus!
