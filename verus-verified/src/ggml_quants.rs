//! Verus-verified ggml quantization proofs -- unbounded
//!
//! Proves bit-manipulation, integer arithmetic, and indexing properties
//! of ggml quantized formats: Q4_0, Q5_0, Q6_K, Q2_K, Q3_K, TQ2_0,
//! TQ1_0, Q8_0, Q8_K, and mul_mat/vec_dot index safety.
//!
//! Generalizes 34 Kani bounded proofs to unbounded deductive proofs.
//!
//! Excluded from Verus (require f32/f64 or runtime alloc):
//!   - dequant_*_no_panic (runtime slice ops)
//!   - q8_0_scale_nonneg, q8_0_quant_in_range, q8_0_output_in_range (f32)
//!   - f16_to_f32_*, f32_to_f16_*, bf16_to_f32_* (f32)
//!   - nearest_int_no_panic (f32)

use vstd::prelude::*;

verus! {

// ============================================================================
// Block size constants (matching kernels/src/ggml/blocks.rs)
// ============================================================================

pub spec const GGML_QK: nat = 32;
pub spec const GGML_QK_K: nat = 256;

pub spec const BLOCK_Q4_0_SIZE: nat = 18;
pub spec const BLOCK_Q4_1_SIZE: nat = 20;
pub spec const BLOCK_Q5_0_SIZE: nat = 22;
pub spec const BLOCK_Q5_1_SIZE: nat = 24;
pub spec const BLOCK_Q8_0_SIZE: nat = 34;
pub spec const BLOCK_Q8_1_SIZE: nat = 36;

pub spec const BLOCK_Q2K_SIZE: nat = 84;
pub spec const BLOCK_Q3K_SIZE: nat = 110;
pub spec const BLOCK_Q4K_SIZE: nat = 144;
pub spec const BLOCK_Q5K_SIZE: nat = 176;
pub spec const BLOCK_Q6K_SIZE: nat = 210;
pub spec const BLOCK_Q8K_SIZE: nat = 292;

pub spec const BLOCK_TQ1_0_SIZE: nat = 54;
pub spec const BLOCK_TQ2_0_SIZE: nat = 66;

// ============================================================================
// 1. Q4_0 nibble range
// ============================================================================

proof fn q4_0_nibble_range(byte: u8)
    ensures ({
        let lo: u8 = byte & 0x0Fu8;
        let hi: u8 = byte >> 4u8;
        &&& lo <= 15
        &&& hi <= 15
        &&& (lo as int - 8) >= -8int
        &&& (lo as int - 8) <= 7int
        &&& (hi as int - 8) >= -8int
        &&& (hi as int - 8) <= 7int
    }),
{
    assert(byte & 0x0Fu8 <= 15) by (bit_vector);
    assert(byte >> 4u8 <= 15) by (bit_vector);
}

// ============================================================================
// 2. Q5_0 5-bit reconstruction range
// ============================================================================

proof fn q5_0_5bit_reconstruction_range(qs_byte: u8, qh_bit: u8)
    requires qh_bit <= 1,
    ensures ({
        let lo_nibble: u8 = qs_byte & 0x0Fu8;
        let hi_nibble: u8 = qs_byte >> 4u8;
        let v0: u8 = lo_nibble | (qh_bit << 4u8);
        let v1: u8 = hi_nibble | (qh_bit << 4u8);
        &&& v0 <= 31u8
        &&& v1 <= 31u8
        &&& (v0 as int - 16) >= -16int
        &&& (v0 as int - 16) <= 15int
        &&& (v1 as int - 16) >= -16int
        &&& (v1 as int - 16) <= 15int
    }),
{
    assert(qs_byte & 0x0Fu8 <= 15u8) by (bit_vector);
    assert(qs_byte >> 4u8 <= 15u8) by (bit_vector);
    assert(forall |lo: u8, hi: u8| lo <= 15u8 && hi <= 1u8 ==>
        #[trigger] (lo | (hi << 4u8)) <= 31u8) by (bit_vector);
}

// Full Q5_0 reconstruction as done in the Kani harness: extract from u32 qh
proof fn q5_0_5bit_range_from_qh(qs_byte: u8, qh: u32, j: nat)
    requires j < 16,
    ensures ({
        let xh_0: u8 = (((qh >> (j as u32)) & 1u32) as u8);
        let xh_1: u8 = (((qh >> ((j + 16) as u32)) & 1u32) as u8);
        let v0: u8 = (qs_byte & 0x0Fu8) | (xh_0 << 4u8);
        let v1: u8 = (qs_byte >> 4u8) | (xh_1 << 4u8);
        &&& v0 <= 31u8
        &&& v1 <= 31u8
    }),
{
    assert(qs_byte & 0x0Fu8 <= 15u8) by (bit_vector);
    assert(qs_byte >> 4u8 <= 15u8) by (bit_vector);
    assert(forall |x: u32, s: u32| s < 32 ==>
        #[trigger] ((x >> s) & 1u32) <= 1u32) by (bit_vector);
    assert(forall |lo: u8, hi: u8| lo <= 15u8 && hi <= 1u8 ==>
        #[trigger] (lo | (hi << 4u8)) <= 31u8) by (bit_vector);
}

// ============================================================================
// 3. Q6_K 6-bit reconstruction
// ============================================================================

proof fn q6k_6bit_reconstruction(ql_byte: u8, qh_2bits: u8)
    requires qh_2bits <= 3,
    ensures ({
        let low: u8 = ql_byte & 0x0Fu8;
        let q6: u8 = low | (qh_2bits << 4u8);
        q6 <= 63u8
    }),
{
    assert(ql_byte & 0x0Fu8 <= 15u8) by (bit_vector);
    assert(forall |lo: u8, hi: u8| lo <= 15u8 && hi <= 3u8 ==>
        #[trigger] (lo | (hi << 4u8)) <= 63u8) by (bit_vector);
}

// With shift extraction from qh byte (mirrors Kani dequant_q6k_6bit_range)
proof fn q6k_6bit_range_with_shift(ql_byte: u8, qh_byte: u8, shift: u32)
    requires shift < 4,
    ensures ({
        let low: u8 = ql_byte & 0x0Fu8;
        let shift_amount: u8 = (shift * 2) as u8;
        let high: u8 = (qh_byte >> shift_amount) & 0x03u8;
        let q6: u8 = low | (high << 4u8);
        q6 <= 63u8
    }),
{
    assert(ql_byte & 0x0Fu8 <= 15u8) by (bit_vector);
    assert(forall |b: u8, s: u8| s < 8 ==>
        #[trigger] ((b >> s) & 0x03u8) <= 3u8) by (bit_vector);
    assert(forall |lo: u8, hi: u8| lo <= 15u8 && hi <= 3u8 ==>
        #[trigger] (lo | (hi << 4u8)) <= 63u8) by (bit_vector);
}

// ============================================================================
// 4. Q2_K 2-bit extraction
// ============================================================================

proof fn q2k_2bit_extraction(byte: u8, shift: nat)
    requires shift < 4,
    ensures ({
        let shift_amount: u8 = (shift * 2) as u8;
        let q: u8 = (byte >> shift_amount) & 3u8;
        q <= 3
    }),
{
    assert(forall |b: u8, s: u8| s < 8 ==>
        #[trigger] ((b >> s) & 3u8) <= 3u8) by (bit_vector);
}

// ============================================================================
// 5. Q3_K 3-bit reconstruction
// ============================================================================

proof fn q3k_3bit_reconstruction(low2: u8, high_bit: u8)
    requires
        low2 <= 3,
        high_bit <= 1,
    ensures ({
        let q3: u8 = low2 | (high_bit << 2u8);
        q3 <= 7u8
    }),
{
    assert(forall |lo: u8, hi: u8| lo <= 3u8 && hi <= 1u8 ==>
        #[trigger] (lo | (hi << 2u8)) <= 7u8) by (bit_vector);
}

// ============================================================================
// 6. TQ2_0 trit range
// ============================================================================

proof fn tq2_0_trit_range(byte: u8, shift: nat)
    requires shift < 4,
    ensures ({
        let shift_amount: u8 = (shift * 2) as u8;
        let bits: u8 = (byte >> shift_amount) & 0x03u8;
        let value: int = (bits as int) - 1;
        &&& bits <= 3
        &&& value >= -1int
        &&& value <= 2int
    }),
{
    assert(forall |b: u8, s: u8| s < 8 ==>
        #[trigger] ((b >> s) & 0x03u8) <= 3u8) by (bit_vector);
}

// ============================================================================
// 7. TQ1_0 base-3 decomposition
// ============================================================================

// Each step of the base-3 decomposition produces a trit in [0, 2]
proof fn tq1_0_single_trit(val: nat)
    ensures (val % 3) <= 2,
{
}

// Full 5-trit decomposition: for any byte < 243, all 5 extracted trits are in [0, 2]
proof fn tq1_0_base3_decomposition(byte: nat)
    requires byte < 243,
    ensures ({
        let t0 = byte % 3;
        let v1 = byte / 3;
        let t1 = v1 % 3;
        let v2 = v1 / 3;
        let t2 = v2 % 3;
        let v3 = v2 / 3;
        let t3 = v3 % 3;
        let v4 = v3 / 3;
        let t4 = v4 % 3;
        &&& t0 <= 2
        &&& t1 <= 2
        &&& t2 <= 2
        &&& t3 <= 2
        &&& t4 <= 2
        // 3^5 = 243, so v4/3 < 3 means t4 < 3
        &&& v4 < 81
    }),
{
    // byte < 243 = 3^5
    // v1 = byte / 3 < 81 = 3^4
    // v2 = v1 / 3 < 27 = 3^3
    // v3 = v2 / 3 < 9 = 3^2
    // v4 = v3 / 3 < 3 = 3^1
    assert(byte / 3 < 81) by (nonlinear_arith) requires byte < 243;
    assert(byte / 3 / 3 < 27) by (nonlinear_arith) requires byte < 243;
    assert(byte / 3 / 3 / 3 < 9) by (nonlinear_arith) requires byte < 243;
    assert(byte / 3 / 3 / 3 / 3 < 3) by (nonlinear_arith) requires byte < 243;
}

// ============================================================================
// 8. get_scale_min_k4 output range
// ============================================================================

proof fn get_scale_min_k4_low_range(scales: Seq<u8>, j: nat)
    requires
        scales.len() == 12,
        j < 4,
    ensures ({
        let sc: u8 = scales[j as int] & 63u8;
        let mn: u8 = scales[(j + 4) as int] & 63u8;
        &&& sc <= 63
        &&& mn <= 63
    }),
{
    assert(forall |x: u8| (x & 63u8) <= 63u8) by (bit_vector);
}

proof fn get_scale_min_k4_high_range(scales: Seq<u8>, j: nat)
    requires
        scales.len() == 12,
        j >= 4,
        j < 8,
    ensures ({
        let i = j - 4;
        let sc_raw: u8 = (scales[(j + 4) as int] & 0x0Fu8) | ((scales[(j - 4) as int] >> 6u8) << 4u8);
        let mn_raw: u8 = (scales[(j + 4) as int] >> 4u8) | ((scales[j as int] >> 6u8) << 4u8);
        &&& sc_raw <= 63
        &&& mn_raw <= 63
    }),
{
    // For j in [4,8): scales[j+4] in [8,12), scales[j-4] in [0,4), scales[j] in [4,8)
    let a = scales[(j + 4) as int];
    let b = scales[(j - 4) as int];
    let c = scales[j as int];
    assert((a & 0x0Fu8) | ((b >> 6u8) << 4u8) <= 63u8) by (bit_vector);
    assert((a >> 4u8) | ((c >> 6u8) << 4u8) <= 63u8) by (bit_vector);
}

// ============================================================================
// 9. Q8_0 quantization output range -- clamped i32 fits in i8
// ============================================================================

proof fn q8_0_clamp_fits_i8(val: int)
    ensures ({
        let clamped: int = if val < -128 { -128int } else if val > 127 { 127int } else { val };
        &&& clamped >= -128int
        &&& clamped <= 127int
    }),
{
}

// ============================================================================
// 10. Q8_K bsums consistency -- sum of 16 i8 values fits in i16
// ============================================================================

proof fn q8k_bsums_range(sum: int)
    requires
        sum >= -128int * 16,
        sum <= 127int * 16,
    ensures
        // -2048 <= sum <= 2032
        sum >= -2048int,
        sum <= 2032int,
        // fits in i16 [-32768, 32767]
        sum >= -32768int,
        sum <= 32767int,
{
    assert(-128 * 16 == -2048int) by (nonlinear_arith);
    assert(127 * 16 == 2032int) by (nonlinear_arith);
}

// Stronger: prove the bound inductively for n i8 values (n <= 16)
proof fn q8k_bsums_inductive(n: nat, partial_sum: int)
    requires
        n <= 16,
        partial_sum >= -128int * (n as int),
        partial_sum <= 127int * (n as int),
    ensures
        // adding one more i8 extends the range correctly
        forall |x: int| (-128 <= x && x <= 127) ==>
            #[trigger] (partial_sum + x) >= -128int * ((n + 1) as int)
            && (partial_sum + x) <= 127int * ((n + 1) as int),
{
    assert forall |x: int| -128 <= x && x <= 127 implies
        #[trigger] (partial_sum + x) >= -128int * ((n + 1) as int)
        && (partial_sum + x) <= 127int * ((n + 1) as int)
    by {
        assert(-128 * (n as int) + (-128) == -128 * ((n + 1) as int))
            by (nonlinear_arith);
        assert(127 * (n as int) + 127 == 127 * ((n + 1) as int))
            by (nonlinear_arith);
    }
}

// ============================================================================
// 11. Block size constants match ggml
// ============================================================================

proof fn block_size_constants()
    ensures
        // Q4_0: block_size=32, type_size=2+16=18
        GGML_QK == 32,
        BLOCK_Q4_0_SIZE == 18,
        // Q4K: block_size=256, type_size=2+2+12+128=144
        GGML_QK_K == 256,
        BLOCK_Q4K_SIZE == 144,
        // Q6K: block_size=256, type_size=128+64+16+2=210
        BLOCK_Q6K_SIZE == 210,
        // Q8K: block_size=256, type_size=4+256+32=292
        BLOCK_Q8K_SIZE == 292,
        // Q8_0: block_size=32, type_size=2+32=34
        BLOCK_Q8_0_SIZE == 34,
        // TQ2_0: block_size=256, type_size=64+2=66
        BLOCK_TQ2_0_SIZE == 66,
        // TQ1_0: block_size=256, type_size=48+4+2=54
        BLOCK_TQ1_0_SIZE == 54,
        // Q2K: 16+64+2+2=84
        BLOCK_Q2K_SIZE == 84,
        // Q3K: 32+64+12+2=110
        BLOCK_Q3K_SIZE == 110,
        // Q5K: 2+2+12+32+128=176
        BLOCK_Q5K_SIZE == 176,
{
}

// ============================================================================
// 12. vec_dot aux8 index bounds (QK=256, 4 chunks of 64)
// ============================================================================

proof fn vec_dot_aux8_index_bounds(j: nat)
    requires j < GGML_QK_K,
    ensures ({
        let chunk: nat = j / 64;
        let within: nat = j % 64;
        let q4_off: nat = chunk * 32;
        &&& chunk < 4
        &&& within < 64
        &&& j < 256
        &&& (if within < 32 { q4_off + within < 128 } else { q4_off + (within - 32) < 128 })
    }),
{
    assert(j / 64 < 4) by (nonlinear_arith) requires j < 256;
}

// ============================================================================
// 13. mul_mat dst index -- row * n + col < m * n
// ============================================================================

proof fn mul_mat_dst_index(m: nat, n: nat, row: nat, col: nat)
    requires
        m >= 1,
        n >= 1,
        row < m,
        col < n,
    ensures
        row * n + col < m * n,
{
    assert(row * n + col < m * n)
        by (nonlinear_arith) requires row < m, col < n, n >= 1;
}

// Weight row bytes: nb * type_size doesn't overflow
proof fn mul_mat_weight_row_bytes(k: nat, block_size: nat, type_size: nat)
    requires
        k >= 1,
        block_size >= 1,
        k % block_size == 0,
        type_size <= block_size,
    ensures ({
        let nb = k / block_size;
        nb * type_size <= k
    }),
{
    assert((k / block_size) * block_size == k) by (nonlinear_arith)
        requires k % block_size == 0, block_size >= 1;
    assert((k / block_size) * type_size <= (k / block_size) * block_size) by (nonlinear_arith)
        requires type_size <= block_size;
}

// ============================================================================
// 14. f16 mantissa rounding
// ============================================================================

proof fn f16_mantissa_rounding(mantissa: u32)
    requires mantissa <= 0x7FFFFFu32,
    ensures ({
        let truncated: u32 = (mantissa >> 13u32) & 0x3FFu32;
        let remainder: u32 = mantissa & 0x1FFFu32;
        &&& truncated <= 0x3FF
        &&& remainder < 0x2000
    }),
{
    assert((mantissa >> 13u32) & 0x3FFu32 <= 0x3FFu32)
        by (bit_vector);
    assert(mantissa & 0x1FFFu32 < 0x2000u32)
        by (bit_vector);
}

// ============================================================================
// Additional proofs derived from Kani harnesses
// ============================================================================

// Q4_0 vec_dot accumulator no i32 overflow
// 16 terms, each v in [-8,7], y in [-128,127], max product = 8*128 = 1024
// 16 * 1024 = 16384 per half, total 32768, within i32 range
proof fn q4_0_accumulator_no_i32_overflow(n: nat)
    requires n <= 16,
    ensures ({
        let max_abs_product: int = 8int * 128int;
        let max_half_sum: int = max_abs_product * (n as int);
        &&& max_half_sum <= 16384
        &&& max_half_sum + max_half_sum <= 32768
        &&& max_half_sum + max_half_sum <= 2147483647 // i32::MAX
    }),
{
    assert(8 * 128 * 16 == 16384int) by (nonlinear_arith);
}

// Q4K scale unpacking: each output byte <= 63
proof fn q4k_scale_unpack_byte_range(val: u32, mask: u32)
    requires mask == 0x3F3F3F3Fu32,
    ensures ({
        let masked: u32 = val & mask;
        let b0: int = (masked & 0xFFu32) as int;
        let b1: int = ((masked >> 8u32) & 0xFFu32) as int;
        let b2: int = ((masked >> 16u32) & 0xFFu32) as int;
        let b3: int = ((masked >> 24u32) & 0xFFu32) as int;
        &&& b0 <= 63
        &&& b1 <= 63
        &&& b2 <= 63
        &&& b3 <= 63
    }),
{
    assert((val & 0x3F3F3F3Fu32) & 0xFFu32 <= 63u32) by (bit_vector);
    assert(((val & 0x3F3F3F3Fu32) >> 8u32) & 0xFFu32 <= 63u32) by (bit_vector);
    assert(((val & 0x3F3F3F3Fu32) >> 16u32) & 0xFFu32 <= 63u32) by (bit_vector);
    assert(((val & 0x3F3F3F3Fu32) >> 24u32) & 0xFFu32 <= 63u32) by (bit_vector);
}

// Q6K dequant index bounds (mirrors Kani q6k_dequant_indices)
proof fn q6k_dequant_indices(half: nat, l: nat)
    requires half < 2, l < 32,
    ensures ({
        let ql_off = half * 64;
        let qh_off = half * 32;
        let y_off = half * 128;
        &&& ql_off + l < 128
        &&& ql_off + 32 + l < 128
        &&& qh_off + l < 64
        &&& y_off + l < 256
        &&& y_off + 32 + l < 256
        &&& y_off + 64 + l < 256
        &&& y_off + 96 + l < 256
    }),
{
}

// Q1_0 single-bit extraction
proof fn q1_0_bit_value(byte: u8, bit_idx: nat)
    requires bit_idx < 8,
    ensures ({
        let bit: u8 = (byte >> (bit_idx as u8)) & 1u8;
        bit <= 1
    }),
{
    assert(forall |b: u8, s: u8| s < 8 ==> ((b >> s) & 1u8) <= 1u8) by (bit_vector);
}

// Q8_K partition: k elements divide into groups of 16
proof fn q8k_bsums_partition(k: nat, j: nat)
    requires
        k > 0,
        k % 16 == 0,
        j < k / 16,
    ensures
        j * 16 < k,
        (j + 1) * 16 <= k,
{
    assert((j + 1) * 16 <= k) by (nonlinear_arith)
        requires j < k / 16, k % 16 == 0;
}

// ============================================================================
// 15. IQ grid lookup index bounds
// ============================================================================

// IQ2XXS: any u8 value indexes into a 256-entry grid
proof fn iq2xxs_grid_index_in_bounds(value: u8)
    ensures (value as nat) < 256,
{
}

// IQ2XS: (q2 & 511) indexes into a 512-entry grid
proof fn iq2xs_grid_index_from_q2(q2: u16)
    ensures ((q2 & 511u16) as nat) < 512,
{
    assert(q2 & 511u16 < 512u16) by (bit_vector);
}

// IQ2XS signs: (q2 >> 9) indexes into a 128-entry table
proof fn iq2xs_signs_index_in_bounds(q2: u16)
    ensures ((q2 >> 9u16) as nat) < 128,
{
    assert(q2 >> 9u16 <= 127u16) by (bit_vector);
}

// IQ2S: (qs | (qh_2bits << 8)) indexes into a 1024-entry grid
proof fn iq2s_grid_index_in_bounds(qs: u8, qh_2bits: u8)
    requires qh_2bits <= 3,
    ensures ({
        let idx: u16 = (qs as u16) | ((qh_2bits as u16) << 8u16);
        (idx as nat) < 1024
    }),
{
    // qs <= 255, qh_2bits <= 3, so (qh_2bits << 8) <= 768
    // 255 | 768 = 1023 < 1024
    assert(forall |a: u8, b: u8| b <= 3 ==>
        #[trigger] ((a as u16) | ((b as u16) << 8u16)) < 1024u16) by (bit_vector);
}

// IQ3: (qs | (bit << 8)) indexes into a 512-entry grid
proof fn iq3_grid_index_in_bounds(qs: u8, qh_bit: u8)
    requires qh_bit <= 1,
    ensures ({
        let idx: u16 = (qs as u16) | ((qh_bit as u16) << 8u16);
        (idx as nat) < 512
    }),
{
    assert(forall |a: u8, b: u8| b <= 1 ==>
        #[trigger] ((a as u16) | ((b as u16) << 8u16)) < 512u16) by (bit_vector);
}

// IQ1S: (qs | (qh_3bits << 8)) indexes into a 2048-entry grid
proof fn iq1s_grid_index_in_bounds(qs: u8, qh_3bits: u8)
    requires qh_3bits <= 7,
    ensures ({
        let idx: u16 = (qs as u16) | ((qh_3bits as u16) << 8u16);
        (idx as nat) < 2048
    }),
{
    // qs <= 255, qh_3bits <= 7, (7 << 8) = 1792, 255 | 1792 = 2047 < 2048
    assert(forall |a: u8, b: u8| b <= 7 ==>
        #[trigger] ((a as u16) | ((b as u16) << 8u16)) < 2048u16) by (bit_vector);
}

// KSIGNS: ((val >> (7*l)) & 127) indexes into a 128-entry table
proof fn ksigns_7bit_mask_in_bounds(val: u32, l: u32)
    requires l < 4,
    ensures
        ((val >> (7u32 * l)) & 127u32) < 128u32,
{
    assert(forall |v: u32, s: u32| s < 28 ==>
        #[trigger] ((v >> s) & 127u32) < 128u32) by (bit_vector);
}

// KMASK_IQ2XS: j < 8 means index is safe (table has 8 entries)
proof fn kmask_iq2xs_index_in_bounds(j: nat)
    requires j < 8,
    ensures j < 8,
{
}

// KVALUES_IQ4NL: (nibble & 0xF) indexes into a 16-entry table
proof fn kvalues_iq4nl_index_in_bounds(nibble: u8)
    ensures ((nibble & 0x0Fu8) as nat) < 16,
{
    assert(nibble & 0x0Fu8 < 16u8) by (bit_vector);
}

// IQ4XS scales: ib/2 < 4 for ib in [0,4)
proof fn iq4xs_scales_index(ib: nat)
    requires ib < 4,
    ensures ib / 2 < 4,
{
}

fn main() {}

} // verus!
