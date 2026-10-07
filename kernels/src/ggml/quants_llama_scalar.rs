//! Direct Rust translation of the scalar (non-AVX) fallback paths from
//! llama.cpp ggml/src/ggml-cpu/arch/x86/quants.c.
//!
//! These are the `#else` sections that execute when none of __AVX2__, __AVX__,
//! or __SSSE3__ are defined, plus the tail scalar loops that run after the
//! SIMD body to handle remaining elements.

use super::blocks::*;
use super::helpers::f16_to_f32;
use super::quants_llama_generic::*;
use super::quantize::{quantize_row_q8_0 as quantize_row_q8_0_ref, quantize_row_q8_1 as quantize_row_q8_1_ref};

const QK8_0: usize = 32;
const QK8_1: usize = 32;
const QK_NVFP4_SUB: usize = 16;

// ── quantize_row_q8_0 scalar path ────────────────────────────────────────────
// C: #else quantize_row_q8_0_ref(x, y, k);
// Called when no AVX/AVX2 target feature is available.
pub fn quantize_row_q8_0_scalar(x: &[f32], y: &mut [BlockQ8_0]) {
    quantize_row_q8_0_ref(x, y);
}

// ── quantize_row_q8_1 scalar path ────────────────────────────────────────────
// C: #else quantize_row_q8_1_ref(x, y, k);
pub fn quantize_row_q8_1_scalar(x: &[f32], y: &mut [BlockQ8_1]) {
    quantize_row_q8_1_ref(x, y);
}

// ── ggml_vec_dot_q4_0_q8_0 scalar tail loop ─────────────────────────────────
// This loop runs unconditionally in C after the #endif of the SIMD block to
// process any remaining blocks (SIMD may leave ib < nb).
// In the pure-scalar path ib starts at 0 and processes all nb blocks.
pub fn ggml_vec_dot_q4_0_q8_0_scalar_tail(
    x: &[BlockQ4_0],
    y: &[BlockQ8_0],
    ib_start: usize,
    nb: usize,
) -> f32 {
    let qk = QK8_0;
    let mut sumf = 0.0f32;
    for ib in ib_start..nb {
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;
        for j in 0..(qk / 2) {
            let v0 = ((x[ib].qs[j] & 0x0F) as i32) - 8;
            let v1 = ((x[ib].qs[j] >> 4) as i32) - 8;
            sumi0 += v0 * (y[ib].qs[j] as i32);
            sumi1 += v1 * (y[ib].qs[j + qk / 2] as i32);
        }
        let sumi = sumi0 + sumi1;
        sumf += sumi as f32 * f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d);
    }
    sumf
}

// ── ggml_vec_dot_q8_0_q8_0 scalar tail loop ─────────────────────────────────
// C: for (; ib < nb; ++ib) { int sumi = 0; ... *s = sumf; }
pub fn ggml_vec_dot_q8_0_q8_0_scalar_tail(
    x: &[BlockQ8_0],
    y: &[BlockQ8_0],
    ib_start: usize,
    nb: usize,
) -> f32 {
    let qk = QK8_0;
    let mut sumf = 0.0f32;
    for ib in ib_start..nb {
        let mut sumi = 0i32;
        for j in 0..qk {
            sumi += (x[ib].qs[j] as i32) * (y[ib].qs[j] as i32);
        }
        sumf += sumi as f32 * (f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d));
    }
    sumf
}

// ── ggml_vec_dot_mxfp4_q8_0 scalar tail loop ────────────────────────────────
// C: for (;ib < nb; ++ib) { const float d = ...; ... sumf += dy * d * (...); }
// Called after #endif for any remaining blocks.
pub fn ggml_vec_dot_mxfp4_q8_0_scalar_tail(
    x: &[BlockMxfp4],
    y: &[BlockQ8_0],
    ib_start: usize,
    nb: usize,
    kvalues_fp4: &[i8; 16],
) -> f32 {
    let mut sumf = 0.0f32;
    for ib in ib_start..nb {
        // C: GGML_CPU_E8M0_TO_FP32_HALF(x[ib].e) * GGML_CPU_FP16_TO_FP32(y[ib].d)
        let d = e8m0_to_f32_half(x[ib].e) * f16_to_f32(y[ib].d);
        let mut sumi = 0i32;
        for j in 0..(QK8_0 / 2) {
            let qv = x[ib].qs[j];
            sumi += (y[ib].qs[j] as i32) * (kvalues_fp4[(qv & 0xf) as usize] as i32);
            sumi += (y[ib].qs[j + QK8_0 / 2] as i32) * (kvalues_fp4[(qv >> 4) as usize] as i32);
        }
        sumf += d * sumi as f32;
    }
    sumf
}

// ── ggml_vec_dot_nvfp4_q8_0 scalar tail loop ────────────────────────────────
// C: for (;ib < nb; ++ib) { for (int s_idx = 0; s_idx < 4; ++s_idx) { ... } }
pub fn ggml_vec_dot_nvfp4_q8_0_scalar_tail(
    x: &[BlockNvfp4],
    y: &[BlockQ8_0],
    ib_start: usize,
    nb: usize,
    kvalues_fp4: &[i8; 16],
) -> f32 {
    let mut sumf = 0.0f32;
    for ib in ib_start..nb {
        for s_idx in 0..4usize {
            let d = ue4m3_to_f32(x[ib].d[s_idx]);
            let q8_block = s_idx / 2;
            let q8_off = (s_idx % 2) * QK_NVFP4_SUB;
            let dy = f16_to_f32(y[2 * ib + q8_block].d);
            let mut sumi_lo = 0i32;
            let mut sumi_hi = 0i32;
            for j in 0..(QK_NVFP4_SUB / 2) {
                let qv = x[ib].qs[s_idx * (QK_NVFP4_SUB / 2) + j];
                sumi_lo += (y[2 * ib + q8_block].qs[q8_off + j] as i32)
                    * (kvalues_fp4[(qv & 0xf) as usize] as i32);
                sumi_hi += (y[2 * ib + q8_block].qs[q8_off + j + QK_NVFP4_SUB / 2] as i32)
                    * (kvalues_fp4[(qv >> 4) as usize] as i32);
            }
            sumf += dy * d * (sumi_lo + sumi_hi) as f32;
        }
    }
    sumf
}

// ── Scalar dispatch wrappers for all vec_dot functions ───────────────────────
// These mirror the #else branches in arch/x86/quants.c:
//   #else
//     ggml_vec_dot_*_generic(n, s, bs, vx, bx, vy, by, nrc);
//   #endif

pub fn ggml_vec_dot_q1_0_q8_0_scalar(n: usize, x: &[BlockQ1_0], y: &[BlockQ8_0]) -> f32 {
    ggml_vec_dot_q1_0_q8_0_generic(n, x, y)
}

pub fn ggml_vec_dot_q4_1_q8_1_scalar(n: usize, x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    ggml_vec_dot_q4_1_q8_1_generic(n, x, y)
}

pub fn ggml_vec_dot_q5_0_q8_0_scalar(n: usize, x: &[BlockQ5_0], y: &[BlockQ8_0]) -> f32 {
    ggml_vec_dot_q5_0_q8_0_generic(n, x, y)
}

pub fn ggml_vec_dot_q5_1_q8_1_scalar(n: usize, x: &[BlockQ5_1], y: &[BlockQ8_1]) -> f32 {
    ggml_vec_dot_q5_1_q8_1_generic(n, x, y)
}

pub fn ggml_vec_dot_tq1_0_q8_k_scalar(n: usize, x: &[BlockTQ1_0], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_tq1_0_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_tq2_0_q8_k_scalar(n: usize, x: &[BlockTQ2_0], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_tq2_0_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_q2_k_q8_k_scalar(n: usize, x: &[BlockQ2K], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_q2_k_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_q3_k_q8_k_scalar(n: usize, x: &[BlockQ3K], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_q3_k_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_q4_k_q8_k_scalar(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_q4_k_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_q5_k_q8_k_scalar(n: usize, x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_q5_k_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_q6_k_q8_k_scalar(n: usize, x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_q6_k_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_iq2_xxs_q8_k_scalar(n: usize, x: &[BlockIQ2XXS], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_iq2_xxs_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_iq2_xs_q8_k_scalar(n: usize, x: &[BlockIQ2XS], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_iq2_xs_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_iq2_s_q8_k_scalar(n: usize, x: &[BlockIQ2S], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_iq2_s_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_iq3_xxs_q8_k_scalar(n: usize, x: &[BlockIQ3XXS], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_iq3_xxs_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_iq3_s_q8_k_scalar(n: usize, x: &[BlockIQ3S], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_iq3_s_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_iq1_s_q8_k_scalar(n: usize, x: &[BlockIQ1S], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_iq1_s_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_iq1_m_q8_k_scalar(n: usize, x: &[BlockIQ1M], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_iq1_m_q8_k_generic(n, x, y)
}

pub fn ggml_vec_dot_iq4_xs_q8_k_scalar(n: usize, x: &[BlockIQ4XS], y: &[BlockQ8K]) -> f32 {
    ggml_vec_dot_iq4_xs_q8_k_generic(n, x, y)
}

// ── Internal helpers ─────────────────────────────────────────────────────────

// GGML_CPU_E8M0_TO_FP32_HALF: decode E8M0 biased exponent, divide by 2.
#[inline(always)]
fn e8m0_to_f32_half(x: u8) -> f32 {
    let bits: u32 = if x < 2 {
        0x00200000u32 << x
    } else {
        (x as u32 - 1) << 23
    };
    f32::from_bits(bits)
}

// UE4M3 (NV fp4 exponent/mantissa encoding) to f32.
// Matches GGML_CPU_UE4M3_TO_FP32 in ggml-impl.h.
#[inline(always)]
fn ue4m3_to_f32(x: u8) -> f32 {
    // Table-based decode: E[3:0] M[2:0] with bias 7.
    // Values: 0=0.0, then 2^(e-7) * (1 + m/8) for e>0, m in 0..7.
    let e = (x >> 3) & 0xf;
    let m = x & 0x7;
    if e == 0 {
        if m == 0 {
            return 0.0f32;
        }
        // subnormal: 2^(-6) * m/8
        return (m as f32) * (1.0f32 / 64.0f32) * (1.0f32 / 8.0f32);
    }
    let scale = f32::from_bits(((e as u32 + 127 - 7) << 23) | ((m as u32) << 20));
    scale
}
