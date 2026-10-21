//! Verus-verified multi-row I2S GEMV indexing -- unbounded proofs
//!
//! Proves: macro-unrolled inner loop indexing for the multi-row I2S
//! GEMV kernel. Weight and activation addresses stay within bounds
//! for all valid (row, group, k) combinations.

use vstd::prelude::*;

verus! {

// Multi-row I2S GEMV: processes one weight row at a time.
// Inner loop iterates g = 0..nb where nb = k/128.
// Weight: row * (k/4) + g * 32 .. + 32
// Activation: g * 128 .. + 128

proof fn multi_row_weight_index(
    k: nat,
    n: nat,
    row: nat,
    g: nat,
)
    requires
        k >= 128,
        k % 128 == 0,
        n >= 1,
        row < n,
        g < k / 128,
    ensures ({
        let row_packed = k / 4;
        let px = row * row_packed + g * 32;
        &&& px + 32 <= n * row_packed
        &&& g * 32 + 32 <= row_packed
    }),
{
    let nb = k / 128;
    let row_packed = k / 4;
    // k % 128 == 0 ??? k == (k/128)*128
    assert(k % 128 == 0);
    assert(k / 128 * 128 + k % 128 == k) by (nonlinear_arith);
    // ??? nb * 128 == k
    assert(nb * 32 == k / 4) by (nonlinear_arith)
        requires nb * 128 == k;
    assert(g * 32 + 32 <= nb * 32) by (nonlinear_arith)
        requires g < nb;
    assert(row * row_packed + row_packed <= n * row_packed) by (nonlinear_arith)
        requires row < n;
}

proof fn multi_row_activation_index(
    k: nat,
    g: nat,
)
    requires
        k >= 128,
        k % 128 == 0,
        g < k / 128,
    ensures ({
        let py = g * 128;
        &&& py + 96 + 32 <= k
        &&& py + 64 + 32 <= k
        &&& py + 32 + 32 <= k
        &&& py + 32 <= k
    }),
{
    let nb = k / 128;
    assert((g + 1) * 128 <= nb * 128) by (nonlinear_arith)
        requires g < nb;
    assert(nb * 128 == k) by (nonlinear_arith)
        requires k >= 128, k % 128 == 0, nb == k / 128;
}

// group32_num decomposition: nb = group32_num * 32 + remainder
proof fn group32_decomposition(k: nat)
    requires
        k >= 128,
        k % 128 == 0,
    ensures ({
        let nb = k / 128;
        let g32 = nb / 32;
        let rem = nb % 32;
        &&& g32 * 32 + rem == nb
        &&& rem < 32
        &&& g32 * 32 <= nb
    }),
{
    let nb = k / 128;
    assert(nb / 32 * 32 + nb % 32 == nb) by (nonlinear_arith);
}

// i16 accumulator overflow safety: 32 groups x 4 maddubs x max ??254 = ??32512
// fits i16 range [-32768, 32767]
proof fn i16_accumulator_no_overflow()
    ensures ({
        let max_per_maddubs: int = 254; // max abs from maddubs(u8, i8)
        let groups_per_batch: int = 32;
        let adds_per_group: int = 4;  // 4 shift positions
        let max_accu = groups_per_batch * adds_per_group * max_per_maddubs;
        &&& max_accu <= 32767
        &&& -max_accu >= -32768
    }),
{
}

}
