//! Direct Rust translation of llama.cpp ggml/src/ggml-cpu/iqp.cpp
//! IQP: 8-row interleaved panel decode + integer GEMV/GEMM against Q8_K rows.
//! Pure translation — no optimization. Algorithm identical to llama.cpp.
//! GGML_IQP_USE_BIAS = 0 path (AVX2 without VNNI).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::blocks::*;
use super::helpers::f16_to_f32;
use super::iq_tables::*;

// ── Constants ─────────────────────────────────────────────────────────────────

pub const QK_K: usize = 256;
pub const IQP_NB_ROWS: usize = 8;
pub const IQP_SB_SIZE: usize = 16;
pub const IQP_NSB: usize = QK_K / IQP_SB_SIZE; // 16
pub const IQP_MIN_BATCH: usize = 8;
pub const IQP_MIN_BATCH_ID: usize = 8;

// ── BlockIqpX8 ────────────────────────────────────────────────────────────────

/// One super-block of a grid-based IQ type decoded to int8, 8 rows interleaved.
///
/// dfac[row] * iscales[sb*8 + row] * qs is bit-identical to dequantize_row_iq*.
/// Layout: qs[sb*128 + g*32 + row*4 + k] = column sb*16 + g*4 + k.
#[repr(C)]
pub struct BlockIqpX8 {
    pub dfac: [f32; IQP_NB_ROWS],
    pub bias: [i32; IQP_NB_ROWS],
    pub iscales: [i8; IQP_NSB * IQP_NB_ROWS],
    pub qs: [i8; QK_K * IQP_NB_ROWS],
}

pub const BLOCK_IQP_X8_SIZE: usize =
    IQP_NB_ROWS * 4 + IQP_NB_ROWS * 4 + IQP_NSB * IQP_NB_ROWS + QK_K * IQP_NB_ROWS;
// = 32 + 32 + 128 + 2048 = 2240

const _: () = {
    assert!(
        std::mem::size_of::<BlockIqpX8>() == BLOCK_IQP_X8_SIZE,
    );
};

// ── IqType enum for decode dispatch ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IqType {
    IQ1S,
    IQ1M,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ4XS,
}

impl IqType {
    pub fn block_size(self) -> usize {
        match self {
            IqType::IQ1S => BLOCK_IQ1S_SIZE,
            IqType::IQ1M => BLOCK_IQ1M_SIZE,
            IqType::IQ2XXS => BLOCK_IQ2XXS_SIZE,
            IqType::IQ2XS => BLOCK_IQ2XS_SIZE,
            IqType::IQ2S => BLOCK_IQ2S_SIZE,
            IqType::IQ3XXS => BLOCK_IQ3XXS_SIZE,
            IqType::IQ3S => BLOCK_IQ3S_SIZE,
            IqType::IQ4XS => BLOCK_IQ4XS_SIZE,
        }
    }
}

// ── Scalar helpers ────────────────────────────────────────────────────────────

/// Low 7 bits of v are the first 7 signs and the 8th is their parity
/// (cf. unpack_ksigns in the CUDA backend).
#[inline(always)]
pub fn iqp_unpack_ksigns(v: u32) -> u8 {
    let mut p = v ^ (v >> 4);
    p ^= p >> 2;
    p ^= p >> 1;
    (v ^ ((p & 1) << 7)) as u8
}

// ── AVX2 helpers ──────────────────────────────────────────────────────────────

/// 0xFF in every byte whose sign bit is set; sv holds each sign byte broadcast
/// over the 8 bytes it governs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iqp_sign_mask(sv: __m256i) -> __m256i {
    let sel = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
    _mm256_cmpeq_epi8(_mm256_and_si256(sv, sel), sel)
}

/// signs holds four sign bytes, byte l governing values 8*l..8*l+7 — spread
/// each over its 8 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iqp_sign_bytes(signs: u32) -> __m256i {
    let bcast = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    );
    _mm256_shuffle_epi8(_mm256_set1_epi32(signs as i32), bcast)
}

/// x ^ m - m negates the lanes where m is 0xFF.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iqp_apply_signs(x: __m256i, m: __m256i) -> __m256i {
    _mm256_sub_epi8(_mm256_xor_si256(x, m), m)
}

// ── Store helpers ─────────────────────────────────────────────────────────────

/// 32 values from four 8-byte grid entries, sign byte l of signs applied to
/// group l.
unsafe fn iqp_store_signed_x8(
    dst: *mut i8,
    g0: u64,
    g1: u64,
    g2: u64,
    g3: u64,
    signs: u32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return iqp_store_signed_x8_avx2(dst, g0, g1, g2, g3, signs);
        }
    }
    iqp_store_signed_x8_scalar(dst, g0, g1, g2, g3, signs);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iqp_store_signed_x8_avx2(
    dst: *mut i8,
    g0: u64,
    g1: u64,
    g2: u64,
    g3: u64,
    signs: u32,
) {
    let g = _mm256_set_epi64x(g3 as i64, g2 as i64, g1 as i64, g0 as i64);
    let m = iqp_sign_mask(iqp_sign_bytes(signs));
    _mm256_storeu_si256(dst as *mut __m256i, iqp_apply_signs(g, m));
}

unsafe fn iqp_store_signed_x8_scalar(
    dst: *mut i8,
    g0: u64,
    g1: u64,
    g2: u64,
    g3: u64,
    signs: u32,
) {
    let g = [g0, g1, g2, g3];
    for l in 0..4 {
        let grid = g[l].to_le_bytes();
        let s = (signs >> (8 * l)) as u8;
        for j in 0..8 {
            let gv = grid[j] as i8;
            *dst.add(8 * l + j) = if s & KMASK_IQ2XS[j] != 0 {
                gv.wrapping_neg()
            } else {
                gv
            };
        }
    }
}

/// Same as iqp_store_signed_x8 but the eight values of group l come from two
/// 4-byte grid entries.
unsafe fn iqp_store_signed_x4(
    dst: *mut i8,
    g0a: u32,
    g0b: u32,
    g1a: u32,
    g1b: u32,
    g2a: u32,
    g2b: u32,
    g3a: u32,
    g3b: u32,
    signs: u32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return iqp_store_signed_x4_avx2(dst, g0a, g0b, g1a, g1b, g2a, g2b, g3a, g3b, signs);
        }
    }
    iqp_store_signed_x4_scalar(dst, g0a, g0b, g1a, g1b, g2a, g2b, g3a, g3b, signs);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn iqp_store_signed_x4_avx2(
    dst: *mut i8,
    g0a: u32,
    g0b: u32,
    g1a: u32,
    g1b: u32,
    g2a: u32,
    g2b: u32,
    g3a: u32,
    g3b: u32,
    signs: u32,
) {
    let g = _mm256_setr_epi32(
        g0a as i32, g0b as i32, g1a as i32, g1b as i32,
        g2a as i32, g2b as i32, g3a as i32, g3b as i32,
    );
    let m = iqp_sign_mask(iqp_sign_bytes(signs));
    _mm256_storeu_si256(dst as *mut __m256i, iqp_apply_signs(g, m));
}

#[allow(clippy::too_many_arguments)]
unsafe fn iqp_store_signed_x4_scalar(
    dst: *mut i8,
    g0a: u32,
    g0b: u32,
    g1a: u32,
    g1b: u32,
    g2a: u32,
    g2b: u32,
    g3a: u32,
    g3b: u32,
    signs: u32,
) {
    let ga = [g0a, g1a, g2a, g3a];
    let gb = [g0b, g1b, g2b, g3b];
    for l in 0..4 {
        let grid1 = ga[l].to_le_bytes();
        let grid2 = gb[l].to_le_bytes();
        let s = (signs >> (8 * l)) as u8;
        for j in 0..4 {
            let v1 = grid1[j] as i8;
            let v2 = grid2[j] as i8;
            *dst.add(8 * l + j) = if s & KMASK_IQ2XS[j] != 0 { v1.wrapping_neg() } else { v1 };
            *dst.add(8 * l + j + 4) =
                if s & KMASK_IQ2XS[j + 4] != 0 { v2.wrapping_neg() } else { v2 };
        }
    }
}

/// 32 values of 8 * grid + delta from four 8-byte grid entries (grid bytes are
/// in {-1, 0, 1}), byte l of deltas applying to group l.
unsafe fn iqp_store_iq1_x8(dst: *mut i8, g0: u64, g1: u64, g2: u64, g3: u64, deltas: u32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return iqp_store_iq1_x8_avx2(dst, g0, g1, g2, g3, deltas);
        }
    }
    iqp_store_iq1_x8_scalar(dst, g0, g1, g2, g3, deltas);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iqp_store_iq1_x8_avx2(
    dst: *mut i8,
    g0: u64,
    g1: u64,
    g2: u64,
    g3: u64,
    deltas: u32,
) {
    let mut g = _mm256_set_epi64x(g3 as i64, g2 as i64, g1 as i64, g0 as i64);
    g = _mm256_add_epi8(g, g);
    g = _mm256_add_epi8(g, g);
    g = _mm256_add_epi8(g, g);
    _mm256_storeu_si256(dst as *mut __m256i, _mm256_add_epi8(g, iqp_sign_bytes(deltas)));
}

unsafe fn iqp_store_iq1_x8_scalar(
    dst: *mut i8,
    g0: u64,
    g1: u64,
    g2: u64,
    g3: u64,
    deltas: u32,
) {
    let g = [g0, g1, g2, g3];
    for l in 0..4 {
        let grid_bytes = g[l].to_le_bytes();
        let delta = (deltas >> (8 * l)) as i8;
        for j in 0..8 {
            let gv = grid_bytes[j] as i8;
            *dst.add(8 * l + j) = 8i8.wrapping_mul(gv).wrapping_add(delta);
        }
    }
}

/// 32 values from 16 packed nibbles through the KVALUES_IQ4NL lookup: low
/// nibbles first, then high.
unsafe fn iqp_store_iq4_x32(dst: *mut i8, qs: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return iqp_store_iq4_x32_avx2(dst, qs);
        }
    }
    for j in 0..16 {
        let b = *qs.add(j);
        *dst.add(j) = KVALUES_IQ4NL[(b & 0xf) as usize];
        *dst.add(j + 16) = KVALUES_IQ4NL[(b >> 4) as usize];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,ssse3")]
unsafe fn iqp_store_iq4_x32_avx2(dst: *mut i8, qs: *const u8) {
    let q = _mm_loadu_si128(qs as *const __m128i);
    let lut = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4 = _mm_set1_epi8(0xf);
    _mm_storeu_si128(dst as *mut __m128i, _mm_shuffle_epi8(lut, _mm_and_si128(q, m4)));
    _mm_storeu_si128(
        dst.add(16) as *mut __m128i,
        _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4)),
    );
}

// ── Decode functions ──────────────────────────────────────────────────────────

unsafe fn iqp_decode_iq2_xxs(
    vx: *const u8,
    vals: *mut i8,
    iscales: *mut i8,
    dfac: *mut f32,
) {
    let x = &*(vx as *const BlockIQ2XXS);
    *dfac = f16_to_f32(x.d) * 0.125;

    for ib32 in 0..(QK_K / 32) {
        let q0 = x.qs[4 * ib32] as u32;
        let q1 = x.qs[4 * ib32 + 1] as u32;
        let q2 = x.qs[4 * ib32 + 2] as u32;
        let q3 = x.qs[4 * ib32 + 3] as u32;
        let aux32_0 = q0 | (q1 << 16);
        let aux32_1 = q2 | (q3 << 16);

        let ls = (2 * (aux32_1 >> 28) + 1) as i8;
        *iscales.add(2 * ib32) = ls;
        *iscales.add(2 * ib32 + 1) = ls;

        let aux8 = [
            aux32_0 as u8,
            (aux32_0 >> 8) as u8,
            (aux32_0 >> 16) as u8,
            (aux32_0 >> 24) as u8,
        ];

        let signs = (iqp_unpack_ksigns((aux32_1 >> 0) & 127) as u32)
            | ((iqp_unpack_ksigns((aux32_1 >> 7) & 127) as u32) << 8)
            | ((iqp_unpack_ksigns((aux32_1 >> 14) & 127) as u32) << 16)
            | ((iqp_unpack_ksigns((aux32_1 >> 21) & 127) as u32) << 24);

        iqp_store_signed_x8(
            vals.add(32 * ib32),
            IQ2XXS_GRID[aux8[0] as usize],
            IQ2XXS_GRID[aux8[1] as usize],
            IQ2XXS_GRID[aux8[2] as usize],
            IQ2XXS_GRID[aux8[3] as usize],
            signs,
        );
    }
}

unsafe fn iqp_decode_iq2_xs(
    vx: *const u8,
    vals: *mut i8,
    iscales: *mut i8,
    dfac: *mut f32,
) {
    let x = &*(vx as *const BlockIQ2XS);
    *dfac = f16_to_f32(x.d) * 0.125;

    for ib32 in 0..(QK_K / 32) {
        *iscales.add(2 * ib32) = (2 * (x.scales[ib32] & 0xf) + 1) as i8;
        *iscales.add(2 * ib32 + 1) = (2 * (x.scales[ib32] >> 4) + 1) as i8;

        let q0 = x.qs[4 * ib32];
        let q1 = x.qs[4 * ib32 + 1];
        let q2 = x.qs[4 * ib32 + 2];
        let q3 = x.qs[4 * ib32 + 3];

        let signs = (iqp_unpack_ksigns((q0 >> 9) as u32) as u32)
            | ((iqp_unpack_ksigns((q1 >> 9) as u32) as u32) << 8)
            | ((iqp_unpack_ksigns((q2 >> 9) as u32) as u32) << 16)
            | ((iqp_unpack_ksigns((q3 >> 9) as u32) as u32) << 24);

        iqp_store_signed_x8(
            vals.add(32 * ib32),
            IQ2XS_GRID[(q0 & 511) as usize],
            IQ2XS_GRID[(q1 & 511) as usize],
            IQ2XS_GRID[(q2 & 511) as usize],
            IQ2XS_GRID[(q3 & 511) as usize],
            signs,
        );
    }
}

unsafe fn iqp_decode_iq2_s(
    vx: *const u8,
    vals: *mut i8,
    iscales: *mut i8,
    dfac: *mut f32,
) {
    let x = &*(vx as *const BlockIQ2S);
    *dfac = f16_to_f32(x.d) * 0.125;

    for ib32 in 0..(QK_K / 32) {
        *iscales.add(2 * ib32) = (2 * (x.scales[ib32] & 0xf) + 1) as i8;
        *iscales.add(2 * ib32 + 1) = (2 * (x.scales[ib32] >> 4) + 1) as i8;

        // x.qs[0..32] = grid indices, x.qs[32..64] = sign bytes
        let qs_off = 4 * ib32;
        let signs_off = 32 + 4 * ib32;

        let sbits = (x.qs[signs_off] as u32)
            | ((x.qs[signs_off + 1] as u32) << 8)
            | ((x.qs[signs_off + 2] as u32) << 16)
            | ((x.qs[signs_off + 3] as u32) << 24);

        let qh = x.qh[ib32] as u32;
        let idx0 = (x.qs[qs_off] as u32) | ((qh << 8) & 0x300);
        let idx1 = (x.qs[qs_off + 1] as u32) | ((qh << 6) & 0x300);
        let idx2 = (x.qs[qs_off + 2] as u32) | ((qh << 4) & 0x300);
        let idx3 = (x.qs[qs_off + 3] as u32) | ((qh << 2) & 0x300);

        iqp_store_signed_x8(
            vals.add(32 * ib32),
            IQ2S_GRID[idx0 as usize],
            IQ2S_GRID[idx1 as usize],
            IQ2S_GRID[idx2 as usize],
            IQ2S_GRID[idx3 as usize],
            sbits,
        );
    }
}

unsafe fn iqp_decode_iq3_xxs(
    vx: *const u8,
    vals: *mut i8,
    iscales: *mut i8,
    dfac: *mut f32,
) {
    let x = &*(vx as *const BlockIQ3XXS);
    *dfac = f16_to_f32(x.d) * 0.25;

    // x.qs[0..64] = grid indices (8 per group), x.qs[64..96] = scales+signs
    for ib32 in 0..(QK_K / 32) {
        let sas_off = 64 + 4 * ib32;
        let aux32 = (x.qs[sas_off] as u32)
            | ((x.qs[sas_off + 1] as u32) << 8)
            | ((x.qs[sas_off + 2] as u32) << 16)
            | ((x.qs[sas_off + 3] as u32) << 24);

        let ls = (2 * (aux32 >> 28) + 1) as i8;
        *iscales.add(2 * ib32) = ls;
        *iscales.add(2 * ib32 + 1) = ls;

        let signs = (iqp_unpack_ksigns((aux32 >> 0) & 127) as u32)
            | ((iqp_unpack_ksigns((aux32 >> 7) & 127) as u32) << 8)
            | ((iqp_unpack_ksigns((aux32 >> 14) & 127) as u32) << 16)
            | ((iqp_unpack_ksigns((aux32 >> 21) & 127) as u32) << 24);

        let qs_off = 8 * ib32;
        iqp_store_signed_x4(
            vals.add(32 * ib32),
            IQ3XXS_GRID[x.qs[qs_off] as usize],
            IQ3XXS_GRID[x.qs[qs_off + 1] as usize],
            IQ3XXS_GRID[x.qs[qs_off + 2] as usize],
            IQ3XXS_GRID[x.qs[qs_off + 3] as usize],
            IQ3XXS_GRID[x.qs[qs_off + 4] as usize],
            IQ3XXS_GRID[x.qs[qs_off + 5] as usize],
            IQ3XXS_GRID[x.qs[qs_off + 6] as usize],
            IQ3XXS_GRID[x.qs[qs_off + 7] as usize],
            signs,
        );
    }
}

unsafe fn iqp_decode_iq3_s(
    vx: *const u8,
    vals: *mut i8,
    iscales: *mut i8,
    dfac: *mut f32,
) {
    let x = &*(vx as *const BlockIQ3S);
    *dfac = f16_to_f32(x.d);

    let mut k: usize = 0;
    let mut qs_off: usize = 0;
    let mut signs_off: usize = 0;
    let mut qh_off: usize = 0;

    let mut ib32 = 0usize;
    while ib32 < QK_K / 32 {
        let db1 = (1 + 2 * (x.scales[ib32 / 2] & 0xf)) as i8;
        let db2 = (1 + 2 * (x.scales[ib32 / 2] >> 4)) as i8;

        *iscales.add(2 * ib32) = db1;
        *iscales.add(2 * ib32 + 1) = db1;
        *iscales.add(2 * ib32 + 2) = db2;
        *iscales.add(2 * ib32 + 3) = db2;

        for h in 0..2 {
            let qh_val = x.qh[qh_off + h] as u32;
            let sbits = (x.signs[signs_off] as u32)
                | ((x.signs[signs_off + 1] as u32) << 8)
                | ((x.signs[signs_off + 2] as u32) << 16)
                | ((x.signs[signs_off + 3] as u32) << 24);

            let idx0 = (x.qs[qs_off] as u32) | ((qh_val << 8) & 256);
            let idx1 = (x.qs[qs_off + 1] as u32) | ((qh_val << 7) & 256);
            let idx2 = (x.qs[qs_off + 2] as u32) | ((qh_val << 6) & 256);
            let idx3 = (x.qs[qs_off + 3] as u32) | ((qh_val << 5) & 256);
            let idx4 = (x.qs[qs_off + 4] as u32) | ((qh_val << 4) & 256);
            let idx5 = (x.qs[qs_off + 5] as u32) | ((qh_val << 3) & 256);
            let idx6 = (x.qs[qs_off + 6] as u32) | ((qh_val << 2) & 256);
            let idx7 = (x.qs[qs_off + 7] as u32) | ((qh_val << 1) & 256);

            iqp_store_signed_x4(
                vals.add(k),
                IQ3S_GRID[idx0 as usize],
                IQ3S_GRID[idx1 as usize],
                IQ3S_GRID[idx2 as usize],
                IQ3S_GRID[idx3 as usize],
                IQ3S_GRID[idx4 as usize],
                IQ3S_GRID[idx5 as usize],
                IQ3S_GRID[idx6 as usize],
                IQ3S_GRID[idx7 as usize],
                sbits,
            );

            k += 32;
            qs_off += 8;
            signs_off += 4;
        }
        qh_off += 2;
        ib32 += 2;
    }
}

/// dequantize_row_iq1_* computes y = dl * (grid[j] + delta) with delta =
/// +-1/8, so the panel stores 8 * grid[j] +- 1 and folds the /8 into dfac.
unsafe fn iqp_decode_iq1_s(
    vx: *const u8,
    vals: *mut i8,
    iscales: *mut i8,
    dfac: *mut f32,
) {
    let x = &*(vx as *const BlockIQ1S);
    *dfac = f16_to_f32(x.d) * 0.125;

    let mut qs_off = 0usize;
    for ib in 0..(QK_K / 32) {
        let qh = x.qh[ib];
        let dl = (2 * ((qh >> 12) & 7) + 1) as i8;
        let delta: i8 = if qh & 0x8000 != 0 { -1 } else { 1 };

        *iscales.add(2 * ib) = dl;
        *iscales.add(2 * ib + 1) = dl;

        let idx0 = (x.qs[qs_off] as u32) | (((qh as u32 >> 0) & 7) << 8);
        let idx1 = (x.qs[qs_off + 1] as u32) | (((qh as u32 >> 3) & 7) << 8);
        let idx2 = (x.qs[qs_off + 2] as u32) | (((qh as u32 >> 6) & 7) << 8);
        let idx3 = (x.qs[qs_off + 3] as u32) | (((qh as u32 >> 9) & 7) << 8);

        let udelta = delta as u8;
        let deltas = (udelta as u32)
            | ((udelta as u32) << 8)
            | ((udelta as u32) << 16)
            | ((udelta as u32) << 24);

        iqp_store_iq1_x8(
            vals.add(32 * ib),
            IQ1S_GRID[idx0 as usize],
            IQ1S_GRID[idx1 as usize],
            IQ1S_GRID[idx2 as usize],
            IQ1S_GRID[idx3 as usize],
            deltas,
        );
        qs_off += 4;
    }
}

unsafe fn iqp_decode_iq1_m(
    vx: *const u8,
    vals: *mut i8,
    iscales: *mut i8,
    dfac: *mut f32,
) {
    let x = &*(vx as *const BlockIQ1M);

    // block_iq1_m has no d field — the fp16 super-block scale is spread over
    // the top nibbles of the four scale words.
    let sc = [
        u16::from_le_bytes([x.scales[0], x.scales[1]]),
        u16::from_le_bytes([x.scales[2], x.scales[3]]),
        u16::from_le_bytes([x.scales[4], x.scales[5]]),
        u16::from_le_bytes([x.scales[6], x.scales[7]]),
    ];
    let scale_u16 =
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    *dfac = f16_to_f32(scale_u16) * 0.125;

    let mut qs_off = 0usize;
    let mut qh_off = 0usize;

    for ib in 0..(QK_K / 32) {
        let sc_ib = sc[ib / 2];
        *iscales.add(2 * ib) =
            (2 * ((sc_ib >> (6 * (ib % 2) as u32 + 0)) & 0x7) + 1) as i8;
        *iscales.add(2 * ib + 1) =
            (2 * ((sc_ib >> (6 * (ib % 2) as u32 + 3)) & 0x7) + 1) as i8;

        let qh0 = x.qh[qh_off];
        let qh1 = x.qh[qh_off + 1];

        let idx0 = (x.qs[qs_off] as u32) | (((qh0 as u32) << 8) & 0x700);
        let idx1 = (x.qs[qs_off + 1] as u32) | (((qh0 as u32) << 4) & 0x700);
        let idx2 = (x.qs[qs_off + 2] as u32) | (((qh1 as u32) << 8) & 0x700);
        let idx3 = (x.qs[qs_off + 3] as u32) | (((qh1 as u32) << 4) & 0x700);

        let d0 = if qh0 & 0x08 != 0 { 0xffu32 } else { 0x01u32 };
        let d1 = if qh0 & 0x80 != 0 { 0xffu32 } else { 0x01u32 };
        let d2 = if qh1 & 0x08 != 0 { 0xffu32 } else { 0x01u32 };
        let d3 = if qh1 & 0x80 != 0 { 0xffu32 } else { 0x01u32 };
        let deltas = d0 | (d1 << 8) | (d2 << 16) | (d3 << 24);

        iqp_store_iq1_x8(
            vals.add(32 * ib),
            IQ1S_GRID[idx0 as usize],
            IQ1S_GRID[idx1 as usize],
            IQ1S_GRID[idx2 as usize],
            IQ1S_GRID[idx3 as usize],
            deltas,
        );
        qs_off += 4;
        qh_off += 2;
    }
}

unsafe fn iqp_decode_iq4_xs(
    vx: *const u8,
    vals: *mut i8,
    iscales: *mut i8,
    dfac: *mut f32,
) {
    let x = &*(vx as *const BlockIQ4XS);
    *dfac = f16_to_f32(x.d);

    let mut qs_ptr = x.qs.as_ptr();
    for ib in 0..(QK_K / 32) {
        let ls = ((x.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32
            | ((((x.scales_h >> (2 * ib as u32)) & 3) as i32) << 4);
        let dl = (ls - 32) as i8;

        *iscales.add(2 * ib) = dl;
        *iscales.add(2 * ib + 1) = dl;

        iqp_store_iq4_x32(vals.add(32 * ib), qs_ptr);
        qs_ptr = qs_ptr.add(16);
    }
}

/// Decode one super-block of the given type. Returns true if the type is
/// supported.
pub unsafe fn iqp_decode_superblock(
    ty: IqType,
    vx: *const u8,
    vals: *mut i8,
    iscales: *mut i8,
    dfac: *mut f32,
) {
    match ty {
        IqType::IQ2XXS => iqp_decode_iq2_xxs(vx, vals, iscales, dfac),
        IqType::IQ2XS => iqp_decode_iq2_xs(vx, vals, iscales, dfac),
        IqType::IQ2S => iqp_decode_iq2_s(vx, vals, iscales, dfac),
        IqType::IQ3XXS => iqp_decode_iq3_xxs(vx, vals, iscales, dfac),
        IqType::IQ3S => iqp_decode_iq3_s(vx, vals, iscales, dfac),
        IqType::IQ1S => iqp_decode_iq1_s(vx, vals, iscales, dfac),
        IqType::IQ1M => iqp_decode_iq1_m(vx, vals, iscales, dfac),
        IqType::IQ4XS => iqp_decode_iq4_xs(vx, vals, iscales, dfac),
    }
}

// ── AVX2 interleave ───────────────────────────────────────────────────────────

/// 8x8 int32 transpose of the 32-column group starting at column `off`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iqp_interleave_x8(
    dst: *mut i8,
    vals: &[[i8; QK_K]; IQP_NB_ROWS],
    off: usize,
) {
    let mut v = [_mm256_setzero_si256(); IQP_NB_ROWS];
    for r in 0..IQP_NB_ROWS {
        v[r] = _mm256_loadu_si256(vals[r][off..].as_ptr() as *const __m256i);
    }

    let a0 = _mm256_unpacklo_epi32(v[0], v[1]);
    let a1 = _mm256_unpackhi_epi32(v[0], v[1]);
    let a2 = _mm256_unpacklo_epi32(v[2], v[3]);
    let a3 = _mm256_unpackhi_epi32(v[2], v[3]);
    let a4 = _mm256_unpacklo_epi32(v[4], v[5]);
    let a5 = _mm256_unpackhi_epi32(v[4], v[5]);
    let a6 = _mm256_unpacklo_epi32(v[6], v[7]);
    let a7 = _mm256_unpackhi_epi32(v[6], v[7]);

    let b0 = _mm256_unpacklo_epi64(a0, a2);
    let b1 = _mm256_unpackhi_epi64(a0, a2);
    let b2 = _mm256_unpacklo_epi64(a1, a3);
    let b3 = _mm256_unpackhi_epi64(a1, a3);
    let b4 = _mm256_unpacklo_epi64(a4, a6);
    let b5 = _mm256_unpackhi_epi64(a4, a6);
    let b6 = _mm256_unpacklo_epi64(a5, a7);
    let b7 = _mm256_unpackhi_epi64(a5, a7);

    _mm256_storeu_si256(dst.add(0 * 32) as *mut __m256i, _mm256_permute2x128_si256(b0, b4, 0x20));
    _mm256_storeu_si256(dst.add(1 * 32) as *mut __m256i, _mm256_permute2x128_si256(b1, b5, 0x20));
    _mm256_storeu_si256(dst.add(2 * 32) as *mut __m256i, _mm256_permute2x128_si256(b2, b6, 0x20));
    _mm256_storeu_si256(dst.add(3 * 32) as *mut __m256i, _mm256_permute2x128_si256(b3, b7, 0x20));
    _mm256_storeu_si256(dst.add(4 * 32) as *mut __m256i, _mm256_permute2x128_si256(b0, b4, 0x31));
    _mm256_storeu_si256(dst.add(5 * 32) as *mut __m256i, _mm256_permute2x128_si256(b1, b5, 0x31));
    _mm256_storeu_si256(dst.add(6 * 32) as *mut __m256i, _mm256_permute2x128_si256(b2, b6, 0x31));
    _mm256_storeu_si256(dst.add(7 * 32) as *mut __m256i, _mm256_permute2x128_si256(b3, b7, 0x31));
}

// ── Panel decode ──────────────────────────────────────────────────────────────

/// Decode IQP_NB_ROWS consecutive source rows (starting at src, row stride
/// nb01) into a panel of nblocks BlockIqpX8 structs.
///
/// # Safety
/// src must point to nblocks * nb01 * IQP_NB_ROWS valid bytes of the given
/// IQ type. dst must point to nblocks initialized BlockIqpX8 structs.
pub unsafe fn iqp_decode_panel_8(
    ty: IqType,
    src: *const u8,
    nb01: usize,
    nblocks: usize,
    dst: *mut BlockIqpX8,
) {
    let bsize = ty.block_size();

    for x in 0..nblocks {
        let panel = &mut *dst.add(x);

        let mut vals = [[0i8; QK_K]; IQP_NB_ROWS];
        let mut row_iscales = [[0i8; IQP_NSB]; IQP_NB_ROWS];
        let mut dfac = [0f32; IQP_NB_ROWS];

        for r in 0..IQP_NB_ROWS {
            let blk = src.add(r * nb01 + x * bsize);
            iqp_decode_superblock(
                ty,
                blk,
                vals[r].as_mut_ptr(),
                row_iscales[r].as_mut_ptr(),
                &mut dfac[r],
            );
        }

        for r in 0..IQP_NB_ROWS {
            panel.dfac[r] = dfac[r];
            for sb in 0..IQP_NSB {
                panel.iscales[sb * IQP_NB_ROWS + r] = row_iscales[r][sb];
            }
            // bias = 0 (GGML_IQP_USE_BIAS = 0)
            panel.bias[r] = 0;
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            for grp in 0..(QK_K / 32) {
                iqp_interleave_x8(panel.qs.as_mut_ptr().add(grp * 256), &vals, grp * 32);
            }
            continue;
        }

        for r in 0..IQP_NB_ROWS {
            for sb in 0..IQP_NSB {
                for g in 0..(IQP_SB_SIZE / 4) {
                    let src_off = sb * IQP_SB_SIZE + g * 4;
                    let dst_off = sb * 128 + g * 32 + r * 4;
                    std::ptr::copy_nonoverlapping(
                        vals[r][src_off..].as_ptr(),
                        panel.qs[dst_off..].as_mut_ptr(),
                        4,
                    );
                }
            }
        }
    }
}

// ── AVX2 GEMM helpers ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_i16_pairs_acc_int32x8(acc: __m256i, x: __m256i) -> __m256i {
    let ones = _mm256_set1_epi16(1);
    _mm256_add_epi32(acc, _mm256_madd_epi16(ones, x))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_sum_us8_pairs_acc_int32x8(acc: __m256i, ax: __m256i, sy: __m256i) -> __m256i {
    let dot = _mm256_maddubs_epi16(ax, sy);
    sum_i16_pairs_acc_int32x8(acc, dot)
}

/// Integer multiply int8_t, add pairwise twice, accumulate into i32x8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_sum_i8_pairs_acc_int32x8(acc: __m256i, x: __m256i, y: __m256i) -> __m256i {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    mul_sum_us8_pairs_acc_int32x8(acc, ax, sy)
}

/// Load 16 activations of one sub-block (GGML_IQP_USE_BIAS = 0: no offset).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iqp_load_y(qs: *const i8) -> __m256i {
    let y = _mm_loadu_si128(qs as *const __m128i);
    _mm256_broadcastsi128_si256(y)
}

/// xv: 8 rows x 4 signed weights, yb: the matching 4 activation bytes
/// broadcast to all 8 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iqp_dot4(acc: __m256i, xv: __m256i, yb: __m256i) -> __m256i {
    mul_sum_i8_pairs_acc_int32x8(acc, xv, yb)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iqp_load_iscales(iscales: *const i8) -> __m256i {
    _mm256_cvtepi8_epi32(_mm_loadl_epi64(iscales as *const __m128i))
}

/// Accumulate one super-block of 8 interleaved rows against one Q8_K row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iqp_acc_block(b: &BlockIqpX8, a: &BlockQ8K) -> __m256i {
    let mut sumi = _mm256_setzero_si256();

    for sb in 0..IQP_NSB {
        let qs = b.qs.as_ptr().add(sb * 128);
        let yv = iqp_load_y(a.qs.as_ptr().add(sb * 16));
        let mut isum = _mm256_setzero_si256();

        isum = iqp_dot4(isum, _mm256_loadu_si256(qs.add(0) as *const __m256i), _mm256_shuffle_epi32(yv, 0x00));
        isum = iqp_dot4(isum, _mm256_loadu_si256(qs.add(32) as *const __m256i), _mm256_shuffle_epi32(yv, 0x55));
        isum = iqp_dot4(isum, _mm256_loadu_si256(qs.add(64) as *const __m256i), _mm256_shuffle_epi32(yv, 0xAA));
        isum = iqp_dot4(isum, _mm256_loadu_si256(qs.add(96) as *const __m256i), _mm256_shuffle_epi32(yv, 0xFF));

        sumi = _mm256_add_epi32(
            sumi,
            _mm256_mullo_epi32(isum, iqp_load_iscales(b.iscales.as_ptr().add(sb * 8))),
        );
    }
    // no bias subtraction (GGML_IQP_USE_BIAS = 0)
    sumi
}

/// One 4-row × nc-column tile; s points at the first of the four output rows,
/// bs floats apart.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn iqp_gemm_tile_4_avx2(
    nb: usize,
    s: *mut f32,
    bs: usize,
    b_ptr_start: *const BlockIqpX8,
    a_ptr: [*const BlockQ8K; 4],
    nc: usize,
) {
    const NCOLS: usize = 8;

    for x in 0..(nc / NCOLS) {
        let b_ptr = b_ptr_start.add(x * nb);

        let mut sumf = [_mm256_setzero_ps(); 4];

        for l in 0..nb {
            let bl = &*b_ptr.add(l);
            let mut sumi = [_mm256_setzero_si256(); 4];

            for sb in 0..IQP_NSB {
                let qs = bl.qs.as_ptr().add(sb * 128);
                let xv0 = _mm256_loadu_si256(qs.add(0) as *const __m256i);
                let xv1 = _mm256_loadu_si256(qs.add(32) as *const __m256i);
                let xv2 = _mm256_loadu_si256(qs.add(64) as *const __m256i);
                let xv3 = _mm256_loadu_si256(qs.add(96) as *const __m256i);

                let isc = iqp_load_iscales(bl.iscales.as_ptr().add(sb * 8));

                for m in 0..4 {
                    let yv = iqp_load_y((*a_ptr[m].add(l)).qs.as_ptr().add(sb * 16));
                    let mut isum = _mm256_setzero_si256();

                    isum = iqp_dot4(isum, xv0, _mm256_shuffle_epi32(yv, 0x00));
                    isum = iqp_dot4(isum, xv1, _mm256_shuffle_epi32(yv, 0x55));
                    isum = iqp_dot4(isum, xv2, _mm256_shuffle_epi32(yv, 0xAA));
                    isum = iqp_dot4(isum, xv3, _mm256_shuffle_epi32(yv, 0xFF));

                    sumi[m] = _mm256_add_epi32(sumi[m], _mm256_mullo_epi32(isum, isc));
                }
            }
            // no bias (GGML_IQP_USE_BIAS = 0)

            let dfac = _mm256_loadu_ps(bl.dfac.as_ptr());
            for m in 0..4 {
                let ad = (*a_ptr[m].add(l)).d;
                sumf[m] = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(sumi[m]),
                    _mm256_mul_ps(dfac, _mm256_set1_ps(ad)),
                    sumf[m],
                );
            }
        }

        for m in 0..4 {
            _mm256_storeu_ps(s.add(m * bs + x * NCOLS), sumf[m]);
        }
    }
}

// ── Generic (scalar) kernels ──────────────────────────────────────────────────

/// Generic GEMV: one activation row against nc weight columns (nc % 8 == 0).
pub unsafe fn iqp_gemv_8x8_q8_k_generic(
    n: usize,
    s: *mut f32,
    _bs: usize,
    vx: *const BlockIqpX8,
    vy: *const BlockQ8K,
    _nr: usize,
    nc: usize,
) {
    let nb = n / QK_K;
    const NCOLS: usize = 8;

    debug_assert_eq!(n % QK_K, 0);
    debug_assert_eq!(nc % NCOLS, 0);

    for x in 0..(nc / NCOLS) {
        let b_ptr = vx.add(x * nb);
        let mut sumf = [0f32; NCOLS];

        for l in 0..nb {
            let bl = &*b_ptr.add(l);
            let al = &*vy.add(l);
            let mut sumi = [0i32; NCOLS];

            for sb in 0..IQP_NSB {
                let mut isum = [0i32; NCOLS];
                for g in 0..4 {
                    for j in 0..NCOLS {
                        for k in 0..4 {
                            isum[j] = isum[j].wrapping_add(
                                (bl.qs[sb * 128 + g * 32 + j * 4 + k] as i32)
                                    * (al.qs[sb * 16 + g * 4 + k] as i32),
                            );
                        }
                    }
                }
                for j in 0..NCOLS {
                    sumi[j] = sumi[j]
                        .wrapping_add(isum[j].wrapping_mul(bl.iscales[sb * 8 + j] as i32));
                }
            }

            for j in 0..NCOLS {
                sumf[j] += sumi[j] as f32 * (bl.dfac[j] * al.d);
            }
        }

        for j in 0..NCOLS {
            *s.add(x * NCOLS + j) = sumf[j];
        }
    }
}

/// Generic tile: 4 activation rows × nc weight columns.
unsafe fn iqp_gemm_tile_4_generic(
    nb: usize,
    s: *mut f32,
    bs: usize,
    b_ptr_start: *const BlockIqpX8,
    a_ptr: [*const BlockQ8K; 4],
    nc: usize,
) {
    const NCOLS: usize = 8;

    for x in 0..(nc / NCOLS) {
        let b_ptr = b_ptr_start.add(x * nb);
        let mut sumf = [[0f32; NCOLS]; 4];

        for l in 0..nb {
            let bl = &*b_ptr.add(l);

            for m in 0..4 {
                let al = &*a_ptr[m].add(l);
                let mut sumi = [0i32; NCOLS];

                for sb in 0..IQP_NSB {
                    let mut isum = [0i32; NCOLS];
                    for g in 0..4 {
                        for j in 0..NCOLS {
                            for k in 0..4 {
                                isum[j] = isum[j].wrapping_add(
                                    (bl.qs[sb * 128 + g * 32 + j * 4 + k] as i32)
                                        * (al.qs[sb * 16 + g * 4 + k] as i32),
                                );
                            }
                        }
                    }
                    for j in 0..NCOLS {
                        sumi[j] = sumi[j].wrapping_add(
                            isum[j].wrapping_mul(bl.iscales[sb * 8 + j] as i32),
                        );
                    }
                }

                for j in 0..NCOLS {
                    sumf[m][j] += sumi[j] as f32 * (bl.dfac[j] * al.d);
                }
            }
        }

        for m in 0..4 {
            for j in 0..NCOLS {
                *s.add(m * bs + x * NCOLS + j) = sumf[m][j];
            }
        }
    }
}

pub unsafe fn iqp_gemm_8x8_q8_k_generic(
    n: usize,
    s: *mut f32,
    bs: usize,
    vx: *const BlockIqpX8,
    vy: *const BlockQ8K,
    nr: usize,
    nc: usize,
) {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    debug_assert_eq!(nr % 4, 0);
    debug_assert_eq!(nc % 8, 0);

    for y in 0..(nr / 4) {
        let a_ptr: [*const BlockQ8K; 4] = [
            vy.add((y * 4 + 0) * nb),
            vy.add((y * 4 + 1) * nb),
            vy.add((y * 4 + 2) * nb),
            vy.add((y * 4 + 3) * nb),
        ];
        iqp_gemm_tile_4_generic(nb, s.add(y * 4 * bs), bs, vx, a_ptr, nc);
    }
}

pub unsafe fn iqp_gemm_8x8_q8_k_p4_generic(
    n: usize,
    s: *mut f32,
    bs: usize,
    vx: *const BlockIqpX8,
    vy: [*const BlockQ8K; 4],
    nc: usize,
) {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    debug_assert_eq!(nc % 8, 0);
    iqp_gemm_tile_4_generic(nb, s, bs, vx, vy, nc);
}

// ── Public kernel entry points ────────────────────────────────────────────────

/// GEMV: one activation row against nc weight columns.
/// vx: *BlockIqpX8 panel, vy: *BlockQ8K activation row.
pub unsafe fn iqp_gemv_8x8_q8_k(
    n: usize,
    s: *mut f32,
    bs: usize,
    vx: *const BlockIqpX8,
    vy: *const BlockQ8K,
    nr: usize,
    nc: usize,
) {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    debug_assert_eq!(nc % 8, 0);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        const NCOLS: usize = 8;
        for x in 0..(nc / NCOLS) {
            let b_ptr = vx.add(x * nb);
            let mut sumf = _mm256_setzero_ps();

            for l in 0..nb {
                let bl = &*b_ptr.add(l);
                let al = &*vy.add(l);
                let dv = {
                    let dfac = _mm256_loadu_ps(bl.dfac.as_ptr());
                    _mm256_mul_ps(dfac, _mm256_set1_ps(al.d))
                };
                let sumi = iqp_acc_block(bl, al);
                sumf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi), dv, sumf);
            }

            _mm256_storeu_ps(s.add(x * NCOLS), sumf);
        }
        return;
    }

    iqp_gemv_8x8_q8_k_generic(n, s, bs, vx, vy, nr, nc);
}

/// GEMM: nr activation rows × nc weight columns. nr must be a multiple of 4.
/// vx: *BlockIqpX8 panel, vy: *BlockQ8K activation matrix (nr × nb blocks).
pub unsafe fn iqp_gemm_8x8_q8_k(
    n: usize,
    s: *mut f32,
    bs: usize,
    vx: *const BlockIqpX8,
    vy: *const BlockQ8K,
    nr: usize,
    nc: usize,
) {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    debug_assert_eq!(nr % 4, 0);
    debug_assert_eq!(nc % 8, 0);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        for y in 0..(nr / 4) {
            let a_ptr: [*const BlockQ8K; 4] = [
                vy.add((y * 4 + 0) * nb),
                vy.add((y * 4 + 1) * nb),
                vy.add((y * 4 + 2) * nb),
                vy.add((y * 4 + 3) * nb),
            ];
            iqp_gemm_tile_4_avx2(nb, s.add(y * 4 * bs), bs, vx, a_ptr, nc);
        }
        return;
    }

    iqp_gemm_8x8_q8_k_generic(n, s, bs, vx, vy, nr, nc);
}

/// GEMM with 4 activation rows passed as separate pointers (for scattered rows
/// in MUL_MAT_ID).
pub unsafe fn iqp_gemm_8x8_q8_k_p4(
    n: usize,
    s: *mut f32,
    bs: usize,
    vx: *const BlockIqpX8,
    vy: [*const BlockQ8K; 4],
    nc: usize,
) {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    debug_assert_eq!(nc % 8, 0);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        iqp_gemm_tile_4_avx2(nb, s, bs, vx, vy, nc);
        return;
    }

    iqp_gemm_8x8_q8_k_p4_generic(n, s, bs, vx, vy, nc);
}

// ── Dispatch helpers (ggml-free equivalents) ──────────────────────────────────

/// Scratch buffer size for one thread's panel, padded to 64 bytes.
pub fn iqp_scratch_size(ne00: usize) -> usize {
    let raw = (ne00 / QK_K) * BLOCK_IQP_X8_SIZE;
    (raw + 63) & !63
}

/// Whether cne1 rows of src1 justify the decode cost for MUL_MAT_ID.
pub fn iqp_mul_mat_id_min_batch(cne1: i64) -> bool {
    cne1 >= IQP_MIN_BATCH_ID as i64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_iqp_x8_size() {
        assert_eq!(std::mem::size_of::<BlockIqpX8>(), BLOCK_IQP_X8_SIZE);
        assert_eq!(BLOCK_IQP_X8_SIZE, 2240);
    }

    #[test]
    fn iqp_unpack_ksigns_known_values() {
        // 0b0000000 → all positive, parity even → parity bit 0
        assert_eq!(iqp_unpack_ksigns(0), 0);
        // 0b0000001 → sign[0]=1, parity=1 → bit7=1 → 0b10000001 = 0x81
        assert_eq!(iqp_unpack_ksigns(1), 0x81);
        // 0b0000011 → sign[0]=sign[1]=1, parity=0 → bit7=0 → 0b00000011 = 0x03
        assert_eq!(iqp_unpack_ksigns(3), 0x03);
        // 0b1111111 → all 7 set, parity(7 ones)=1 → bit7=1 → 0x7f ^ 0x80 = 0xff
        assert_eq!(iqp_unpack_ksigns(0x7f), 0xff);
    }

    #[test]
    fn iqp_decode_iq4_xs_known_block() {
        // Construct a minimal IQ4_XS block: d=1.0 f16, scales all = 32 (dl=0),
        // qs all 0x00 → every nibble 0 → KVALUES_IQ4NL[0] = -127.
        let mut block = BlockIQ4XS {
            d: 0x3c00u16, // f16 1.0
            scales_h: 0x0000,
            scales_l: [0x00; 4], // ls = 0 for all, dl = 0 - 32 = -32
            qs: [0u8; 128],
        };
        // Set scales_l so ls = 32, dl = 0
        // ls = (scales_l[ib/2] >> 4*(ib%2)) & 0xf | (scales_h >> 2*ib & 3) << 4
        // To get ls=32: low nibble = 0, scales_h bits = 2 → ls = 0 | (2<<4) = 32
        block.scales_h = 0xAAAA; // bits [1:0] per pair = 10b = 2 for all 8 groups

        let mut vals = [0i8; QK_K];
        let mut iscales = [0i8; IQP_NSB];
        let mut dfac = 0f32;

        unsafe {
            iqp_decode_iq4_xs(
                &block as *const BlockIQ4XS as *const u8,
                vals.as_mut_ptr(),
                iscales.as_mut_ptr(),
                &mut dfac,
            );
        }

        assert_eq!(dfac, 1.0f32);
        // dl = 32 - 32 = 0 for all sub-blocks
        for &s in &iscales {
            assert_eq!(s, 0);
        }
        // All qs bytes = 0x00, so all nibbles = 0 → KVALUES_IQ4NL[0] = -127
        for &v in &vals {
            assert_eq!(v, KVALUES_IQ4NL[0]);
        }
    }

    #[test]
    fn iqp_decode_iq4_xs_scale_check() {
        // Test that the scale decomposition is correct.
        // Set scales_l[0] = 0x11 (both nibbles = 1), scales_h = 0 → ls = 1, dl = -31
        let block = BlockIQ4XS {
            d: 0x3c00u16, // 1.0 f16
            scales_h: 0x0000,
            scales_l: [0x11, 0x11, 0x11, 0x11],
            qs: [0u8; 128],
        };

        let mut vals = [0i8; QK_K];
        let mut iscales = [0i8; IQP_NSB];
        let mut dfac = 0f32;

        unsafe {
            iqp_decode_iq4_xs(
                &block as *const BlockIQ4XS as *const u8,
                vals.as_mut_ptr(),
                iscales.as_mut_ptr(),
                &mut dfac,
            );
        }

        assert_eq!(dfac, 1.0f32);
        // ls = 1, dl = 1 - 32 = -31
        for &s in &iscales {
            assert_eq!(s, -31);
        }
    }

    #[test]
    fn iqp_decode_panel_8_iq4_xs_runs() {
        // Smoke test: decode a panel of 1 block of IQ4_XS.
        // Build 8 identical blocks (one per row).
        let block = BlockIQ4XS {
            d: 0x3c00u16,
            scales_h: 0xAAAAu16,
            scales_l: [0x00; 4],
            qs: [0x12u8; 128],
        };

        // 8 copies of the block (one per row), row stride = block size
        let mut data = vec![0u8; BLOCK_IQ4XS_SIZE * IQP_NB_ROWS];
        for r in 0..IQP_NB_ROWS {
            let dst = &mut data[r * BLOCK_IQ4XS_SIZE..(r + 1) * BLOCK_IQ4XS_SIZE];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &block as *const BlockIQ4XS as *const u8,
                    dst.as_mut_ptr(),
                    BLOCK_IQ4XS_SIZE,
                );
            }
        }

        let mut panel = std::mem::MaybeUninit::<BlockIqpX8>::uninit();

        unsafe {
            iqp_decode_panel_8(
                IqType::IQ4XS,
                data.as_ptr(),
                BLOCK_IQ4XS_SIZE,
                1,
                panel.as_mut_ptr(),
            );
        }

        let panel = unsafe { panel.assume_init() };

        // All rows should have dfac = 1.0
        for &d in &panel.dfac {
            assert_eq!(d, 1.0f32);
        }

        // All iscales should be 0 (dl=0, scales_h=0xAAAA→ls=32→dl=0)
        for &s in &panel.iscales {
            assert_eq!(s, 0);
        }
    }

    #[test]
    fn iqp_store_signed_x8_scalar_positive() {
        // g0 = bytes [1,2,3,4,5,6,7,8], signs = 0 → all positive
        let g0 = u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]);
        let g1 = u64::from_le_bytes([9, 10, 11, 12, 13, 14, 15, 16]);
        let g2 = u64::from_le_bytes([17, 18, 19, 20, 21, 22, 23, 24]);
        let g3 = u64::from_le_bytes([25, 26, 27, 28, 29, 30, 31, 32]);

        let mut dst = [0i8; 32];
        unsafe {
            iqp_store_signed_x8_scalar(dst.as_mut_ptr(), g0, g1, g2, g3, 0);
        }

        for i in 0..32 {
            assert_eq!(dst[i], (i + 1) as i8, "index {i}");
        }
    }

    #[test]
    fn iqp_store_signed_x8_scalar_negate_first_byte() {
        // signs = 0x00_00_00_01: bit 0 of byte 0 set → negate dst[0]
        let g0 = u64::from_le_bytes([5, 0, 0, 0, 0, 0, 0, 0]);
        let mut dst = [0i8; 32];
        unsafe {
            iqp_store_signed_x8_scalar(dst.as_mut_ptr(), g0, 0, 0, 0, 0x01);
        }
        assert_eq!(dst[0], -5);
        for i in 1..32 {
            assert_eq!(dst[i], 0);
        }
    }

    #[test]
    fn scratch_size_alignment() {
        // ne00 = 256 → 1 block → BLOCK_IQP_X8_SIZE = 2240 → already 64-aligned
        assert_eq!(iqp_scratch_size(256), 2240);
        // ne00 = 512 → 2 blocks → 4480, padded to next 64 → 4480
        assert_eq!(iqp_scratch_size(512), 4480);
    }
}
