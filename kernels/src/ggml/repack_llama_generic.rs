//! Direct translation of ggml-cpu/repack.cpp lines 1-2681 — generic scalar
//! GEMV/GEMM kernels for Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q2_K, IQ4_NL, mxfp4,
//! Q1_0; and quantize-mat routines for Q8_0 / Q8_K.
//!
//! Skips RISC-V-only (#if defined __riscv_zvfh) blocks.
//! Functions already translated in repack_llama.rs (Q2K/Q5K/Q6K 8x8 GEMV,
//! Q5K/Q6K 8x4 GEMM) are not duplicated here.

use super::blocks::{
    BlockQ2Kx8, BlockQ4Kx8, BlockQ5Kx8, BlockQ6Kx8, BlockQ8K, BlockQ8Kx4,
};
use super::repack_llama2::{BlockQ4_0x8, BlockQ8_0x4};

// ── Constants ─────────────────────────────────────────────────────────────────

pub const QK8_0: usize = 32;
pub const QK_K: usize = 256;
pub const QK4_0: usize = 32;
pub const QK4_NL: usize = 32;
pub const QK_MXFP4: usize = 32;
pub const QK1_0: usize = 128;

// ── Lookup tables ─────────────────────────────────────────────────────────────

pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

// e2m1 values (doubled), shared by MXFP4 and NVFP4
pub const KVALUES_MXFP4: [i8; 16] = [
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
];

// ── New interleaved block structs ─────────────────────────────────────────────

/// 4 Q4_0 blocks interleaved (block<4,4>): d[4] f16, qs[64] nibble-packed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ4_0x4 {
    pub d:  [u16; 4],
    pub qs: [i8;  64],
}

/// 4 Q1_0 blocks interleaved (block<1,4>): d[4] f16, qs[64] bit-packed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ1_0x4 {
    pub d:  [u16; 4],
    pub qs: [i8;  64],
}

/// 4 IQ4_NL blocks interleaved: d[4] f16, qs[64] nibble-packed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockIQ4NLx4 {
    pub d:  [u16; 4],
    pub qs: [u8;  64],
}

/// 8 IQ4_NL blocks interleaved: d[8] f16, qs[128] nibble-packed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockIQ4NLx8 {
    pub d:  [u16; 8],
    pub qs: [u8; 128],
}

/// 4 MXFP4 blocks interleaved: e[4] E8M0 exponents, qs[64] nibble-packed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockMxfp4x4 {
    pub e:  [u8;  4],
    pub qs: [u8;  64],
}

/// 8 MXFP4 blocks interleaved: e[8] E8M0 exponents, qs[128] nibble-packed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockMxfp4x8 {
    pub e:  [u8;   8],
    pub qs: [u8; 128],
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Software IEEE-754 f16 → f32.
#[inline(always)]
fn f16_to_f32(v: u16) -> f32 {
    let s = (v as u32 & 0x8000) << 16;
    let e = (v as u32 >> 10) & 0x1f;
    let m = v as u32 & 0x3ff;
    let bits = if e == 0 {
        if m == 0 { s } else {
            let mut exp = 127u32 - 14;
            let mut man = m;
            while man & 0x400 == 0 { man <<= 1; exp -= 1; }
            s | ((exp) << 23) | ((man & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f800000 | (m << 13)
    } else {
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Software IEEE-754 f32 → f16 (round-to-nearest-even).
#[inline(always)]
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let s = ((bits >> 16) & 0x8000) as u16;
    let e = ((bits >> 23) & 0xff) as i32;
    let m = bits & 0x007fffff;
    if e == 255 {
        return s | 0x7c00 | ((m != 0) as u16 * 0x0200);
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 { return s | 0x7c00; }
    if e16 <= 0 {
        if e16 < -10 { return s; }
        let man = (m | 0x800000) >> (14 - e16);
        return s | (man >> 13) as u16;
    }
    s | ((e16 as u16) << 10) | (m >> 13) as u16
}

/// Fast round-to-nearest integer (mirrors nearest_int in repack.cpp).
#[inline(always)]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4194303.0);
    let val = fval + 12582912.0f32;
    let i = val.to_bits() as i32;
    (i & 0x007fffff) - 0x00400000
}

/// E8M0 exponent byte → f32: 2^(e − 127).
#[inline(always)]
fn e8m0_to_f32(e: u8) -> f32 {
    2.0f32.powi(e as i32 - 127)
}

// ── Quantize-mat functions (generic scalar) ───────────────────────────────────

/// ggml_quantize_mat_q8_0_4x4_generic — 4-row Q8_0 with blck_size_interleave=4.
pub unsafe fn quantize_mat_q8_0_4x4_generic(
    x: *const f32,
    vy: *mut u8,
    k: usize,
) {
    let nb = k / QK8_0;
    let y = vy as *mut BlockQ8_0x4;
    const BSI: usize = 4;

    let mut srcv = [[0f32; 32]; 4];
    let mut id   = [0f32; 4];

    for i in 0..nb {
        for row in 0..4usize {
            let mut amax = 0f32;
            for j in 0..QK8_0 {
                srcv[row][j] = *x.add(row * k + i * QK8_0 + j);
                if srcv[row][j].abs() > amax { amax = srcv[row][j].abs(); }
            }
            let d = amax / 127.0;
            id[row] = if d != 0.0 { 1.0 / d } else { 0.0 };
            (*y.add(i)).d[row] = f32_to_f16(d);
        }
        for j in 0..(QK8_0 * 4) {
            let src_off = (j / (4 * BSI)) * BSI + (j % BSI);
            let src_id  = (j % (4 * BSI)) / BSI;
            let x0 = srcv[src_id][src_off] * id[src_id];
            (*y.add(i)).qs[j] = x0.round() as i8;
        }
    }
}

/// ggml_quantize_mat_q8_0_4x8_generic — 4-row Q8_0 with blck_size_interleave=8.
pub unsafe fn quantize_mat_q8_0_4x8_generic(
    x: *const f32,
    vy: *mut u8,
    k: usize,
) {
    let nb = k / QK8_0;
    let y = vy as *mut BlockQ8_0x4;
    const BSI: usize = 8;

    let mut srcv = [[0f32; 32]; 4];
    let mut id   = [0f32; 4];

    for i in 0..nb {
        for row in 0..4usize {
            let mut amax = 0f32;
            for j in 0..QK8_0 {
                srcv[row][j] = *x.add(row * k + i * QK8_0 + j);
                if srcv[row][j].abs() > amax { amax = srcv[row][j].abs(); }
            }
            let d = amax / 127.0;
            id[row] = if d != 0.0 { 1.0 / d } else { 0.0 };
            (*y.add(i)).d[row] = f32_to_f16(d);
        }
        for j in 0..(QK8_0 * 4) {
            let src_off = (j / (4 * BSI)) * BSI + (j % BSI);
            let src_id  = (j % (4 * BSI)) / BSI;
            let x0 = srcv[src_id][src_off] * id[src_id];
            (*y.add(i)).qs[j] = x0.round() as i8;
        }
    }
}

/// ggml_quantize_mat_q8_K_4x4_generic — 4-row Q8_K with blck_size_interleave=4.
pub unsafe fn quantize_mat_q8_k_4x4_generic(
    x: *const f32,
    vy: *mut u8,
    k: usize,
) {
    let nb = k / QK_K;
    let y = vy as *mut BlockQ8Kx4;
    const BSI: usize = 4;

    let mut srcv  = [[0f32; 256]; 4];
    let mut iscale = [0f32; 4];

    for i in 0..nb {
        for row in 0..4usize {
            let mut amax = 0f32;
            let mut max  = 0f32;
            for j in 0..QK_K {
                srcv[row][j] = *x.add(row * k + i * QK_K + j);
                let a = srcv[row][j].abs();
                if a > amax { amax = a; max = srcv[row][j]; }
            }
            iscale[row] = if amax != 0.0 { -127.0 / max } else { 0.0 };
            (*y.add(i)).d[row] = if amax != 0.0 { 1.0 / iscale[row] } else { 0.0 };
        }
        for j in 0..(QK_K / 4) { (*y.add(i)).bsums[j] = 0; }

        for j in 0..(QK_K * 4) {
            let src_off = (j / (4 * BSI)) * BSI + (j % BSI);
            let src_id  = (j % (4 * BSI)) / BSI;
            let index = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);
            let x0 = srcv[src_id][src_off] * iscale[src_id];
            let q = nearest_int(x0);
            (*y.add(i)).qs[j] = q as i8;
            (*y.add(i)).bsums[index] += q as i16;
        }
    }
}

/// ggml_quantize_mat_q8_K_4x8_generic — 4-row Q8_K with blck_size_interleave=8.
pub unsafe fn quantize_mat_q8_k_4x8_generic(
    x: *const f32,
    vy: *mut u8,
    k: usize,
) {
    let nb = k / QK_K;
    let y = vy as *mut BlockQ8Kx4;
    const BSI: usize = 8;

    let mut srcv  = [[0f32; 256]; 4];
    let mut iscale = [0f32; 4];

    for i in 0..nb {
        for row in 0..4usize {
            let mut amax = 0f32;
            let mut max  = 0f32;
            for j in 0..QK_K {
                srcv[row][j] = *x.add(row * k + i * QK_K + j);
                let a = srcv[row][j].abs();
                if a > amax { amax = a; max = srcv[row][j]; }
            }
            iscale[row] = if amax != 0.0 { -127.0 / max } else { 0.0 };
            (*y.add(i)).d[row] = if amax != 0.0 { 1.0 / iscale[row] } else { 0.0 };
        }
        for j in 0..(QK_K / 4) { (*y.add(i)).bsums[j] = 0; }

        for j in 0..(QK_K * 4) {
            let src_off = (j / (4 * BSI)) * BSI + (j % BSI);
            let src_id  = (j % (4 * BSI)) / BSI;
            let index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);
            let x0 = srcv[src_id][src_off] * iscale[src_id];
            let q = nearest_int(x0);
            (*y.add(i)).qs[j] = q as i8;
            (*y.add(i)).bsums[index] += q as i16;
        }
    }
}

// ── Helper: decode Q4K / Q5K scale bytes ──────────────────────────────────────

/// Decode one 12-byte Q4K/Q5K scale group into utmp[4] (mirrors memcpy+kmask logic).
#[inline(always)]
fn decode_q4k_scales(scales_src: &[u8], utmp: &mut [u32; 4]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    // copy 12 bytes as three u32
    utmp[0] = u32::from_le_bytes(scales_src[0..4].try_into().unwrap());
    utmp[1] = u32::from_le_bytes(scales_src[4..8].try_into().unwrap());
    utmp[2] = u32::from_le_bytes(scales_src[8..12].try_into().unwrap());
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1]  = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2]  = uaux;
    utmp[0] &= KMASK1;
}

// ── GEMV Q4_0 ─────────────────────────────────────────────────────────────────

/// ggml_gemv_q4_0_4x4_q8_0_generic — scalar, blocklen=4, ncols=4.
pub unsafe fn gemv_q4_0_4x4_q8_0_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 4;
    let nb = n / QK8_0;
    let a_ptr = vy as *const super::blocks::BlockQ8_0;
    let b_base = vx as *const BlockQ4_0x4;
    let mut sumf = [0f32; NCOLS];
    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        for j in 0..NCOLS { sumf[j] = 0.0; }
        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);
            for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                for j in 0..NCOLS {
                    let mut sumi = 0i32;
                    for i in 0..BLOCKLEN {
                        let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i] as i32;
                        let v0 = ((byte << 4) as i8) as i32;
                        let v1 = ((byte & 0xF0i32) as i8) as i32;
                        sumi += (v0 * a.qs[k * BLOCKLEN + i] as i32
                               + v1 * a.qs[k * BLOCKLEN + i + QK8_0 / 2] as i32) >> 4;
                    }
                    sumf[j] += sumi as f32 * f16_to_f32(b.d[j]) * f16_to_f32(a.d);
                }
            }
        }
        for j in 0..NCOLS { *s.add(x * NCOLS + j) = sumf[j]; }
    }
}

/// ggml_gemv_q4_0_4x8_q8_0_generic — scalar, blocklen=8, ncols=4.
pub unsafe fn gemv_q4_0_4x8_q8_0_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 8;
    let nb = n / QK8_0;
    let a_ptr = vy as *const super::blocks::BlockQ8_0;
    let b_base = vx as *const BlockQ4_0x4;
    let mut sumf = [0f32; NCOLS];
    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        for j in 0..NCOLS { sumf[j] = 0.0; }
        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);
            for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                for j in 0..NCOLS {
                    let mut sumi = 0i32;
                    for i in 0..BLOCKLEN {
                        let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i] as i32;
                        let v0 = ((byte << 4) as i8) as i32;
                        let v1 = ((byte & 0xF0i32) as i8) as i32;
                        sumi += (v0 * a.qs[k * BLOCKLEN + i] as i32
                               + v1 * a.qs[k * BLOCKLEN + i + QK8_0 / 2] as i32) >> 4;
                    }
                    sumf[j] += sumi as f32 * f16_to_f32(b.d[j]) * f16_to_f32(a.d);
                }
            }
        }
        for j in 0..NCOLS { *s.add(x * NCOLS + j) = sumf[j]; }
    }
}

/// ggml_gemv_q4_0_8x8_q8_0_generic — scalar, blocklen=8, ncols=8.
pub unsafe fn gemv_q4_0_8x8_q8_0_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const NCOLS: usize = 8;
    const BLOCKLEN: usize = 8;
    let nb = n / QK8_0;
    let a_ptr = vy as *const super::blocks::BlockQ8_0;
    let b_base = vx as *const BlockQ4_0x8;
    let mut sumf = [0f32; NCOLS];
    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        for j in 0..NCOLS { sumf[j] = 0.0; }
        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);
            for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                for j in 0..NCOLS {
                    let mut sumi = 0i32;
                    for i in 0..BLOCKLEN {
                        let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i] as i32;
                        let v0 = ((byte << 4) as i8) as i32;
                        let v1 = ((byte & 0xF0i32) as i8) as i32;
                        sumi += (v0 * a.qs[k * BLOCKLEN + i] as i32
                               + v1 * a.qs[k * BLOCKLEN + i + QK8_0 / 2] as i32) >> 4;
                    }
                    sumf[j] += sumi as f32 * f16_to_f32(b.d[j]) * f16_to_f32(a.d);
                }
            }
        }
        for j in 0..NCOLS { *s.add(x * NCOLS + j) = sumf[j]; }
    }
}

// ── GEMV Q4_K ─────────────────────────────────────────────────────────────────

/// Inner Q4K GEMV kernel parameterised by BLOCKLEN and NCOLS.
/// Mirrors ggml_gemv_q4_K_8x{4,8}_q8_K_generic in repack.cpp.
unsafe fn gemv_q4_k_q8_k_impl<const BLOCKLEN: usize, const NCOLS: usize>(
    n: usize, s: *mut f32, vx: *const u8, vy: *const u8, nc: usize,
) {
    let nb = n / QK_K;
    let a_ptr = vy as *const BlockQ8K;
    let b_base = vx as *const BlockQ4Kx8;
    let mut utmp_all = [0u32; 32];

    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        let mut sumf     = [0f32; NCOLS];
        let mut sum_minf = [0f32; NCOLS];

        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);

            // decode all 8 scale groups of 12 bytes each
            for sb in 0..8usize {
                let mut tmp = [0u32; 4];
                decode_q4k_scales(&b.scales[sb * 12..], &mut tmp);
                utmp_all[sb * 4]     = tmp[0];
                utmp_all[sb * 4 + 1] = tmp[1];
                utmp_all[sb * 4 + 2] = tmp[2];
                utmp_all[sb * 4 + 3] = tmp[3];
            }
            let utmp_bytes = {
                let ptr = utmp_all.as_ptr() as *const u8;
                std::slice::from_raw_parts(ptr, 128)
            };

            for k in 0..(QK_K / (2 * BLOCKLEN)) {
                let scale_grp = k / (8 / BLOCKLEN);
                let scales_0 = &utmp_bytes[scale_grp * 32..];
                let scales_1 = &utmp_bytes[scale_grp * 32 + 16..];

                for j in 0..NCOLS {
                    let mut sumi1 = 0i32;
                    let mut sumi2 = 0i32;
                    for i in 0..BLOCKLEN {
                        let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i];
                        let v0 = (byte & 0xF) as i32;
                        let v1 = (byte >> 4) as i32;
                        // Q8 index depends on BLOCKLEN
                        let q8_idx = if BLOCKLEN == 4 {
                            (k / 8) * 64 + (k % 8) * BLOCKLEN + i
                        } else {
                            (k >> 2) * 64 + (k % 4) * BLOCKLEN + i
                        };
                        sumi1 = v0 * a.qs[q8_idx] as i32 * scales_0[j] as i32;
                        sumi2 = v1 * a.qs[q8_idx + 32] as i32 * scales_1[j] as i32;
                        sumf[j] += (sumi1 + sumi2) as f32
                            * f16_to_f32(b.d[j])
                            * a.d;
                    }
                    let _ = (sumi1, sumi2); // suppress unused
                }
            }

            for sb in 0..8usize {
                let mins_ptr = utmp_bytes.as_ptr().add(8 + sb * 16);
                let mins = std::slice::from_raw_parts(mins_ptr, NCOLS);
                for j in 0..NCOLS {
                    sum_minf[j] += mins[j] as f32
                        * (a.bsums[sb * 2] as i32 + a.bsums[sb * 2 + 1] as i32) as f32
                        * f16_to_f32(b.dmin[j])
                        * a.d;
                }
            }
        }

        for j in 0..NCOLS {
            *s.add(x * NCOLS + j) = sumf[j] - sum_minf[j];
        }
    }
}

/// ggml_gemv_q4_K_8x4_q8_K_generic — blocklen=4, ncols=8.
pub unsafe fn gemv_q4_k_8x4_q8_k_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    gemv_q4_k_q8_k_impl::<4, 8>(n, s, vx, vy, nc);
}

/// ggml_gemv_q4_K_8x8_q8_K_generic — blocklen=8, ncols=8.
pub unsafe fn gemv_q4_k_8x8_q8_k_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    gemv_q4_k_q8_k_impl::<8, 8>(n, s, vx, vy, nc);
}

// ── GEMV Q5_K (blocklen=4 variant — the 8x8 is in repack_llama.rs) ───────────

/// ggml_gemv_q5_K_8x4_q8_K_generic — blocklen=4, ncols=8.
pub unsafe fn gemv_q5_k_8x4_q8_k_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const BLOCKLEN: usize = 4;
    const NCOLS: usize = 8;
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let nb = n / QK_K;
    let a_ptr = vy as *const BlockQ8K;
    let b_base = vx as *const BlockQ5Kx8;
    let mut utmp = [0u32; 32];

    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        let mut sumf     = [0f32; NCOLS];
        let mut sum_minf = [0f32; NCOLS];

        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);

            for sb in 0..8usize {
                let src = &b.scales[sb * 12..];
                let u0 = u32::from_le_bytes(src[0..4].try_into().unwrap());
                let u1 = u32::from_le_bytes(src[4..8].try_into().unwrap());
                let u2 = u32::from_le_bytes(src[8..12].try_into().unwrap());
                utmp[sb * 4 + 3] = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
                let uaux = u1 & KMASK1;
                utmp[sb * 4 + 1] = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
                utmp[sb * 4 + 2] = uaux;
                utmp[sb * 4 + 0] = u0 & KMASK1;
            }
            let utmp_bytes = std::slice::from_raw_parts(utmp.as_ptr() as *const u8, 128);

            for k in 0..(QK_K / (2 * BLOCKLEN)) {
                const SCALE_STRIDE: usize = 32;
                let sg = k / (32 / BLOCKLEN);
                let scales_0 = &utmp_bytes[sg * SCALE_STRIDE..];
                let scales_1 = &utmp_bytes[sg * SCALE_STRIDE + 16..];
                let qh_shift = sg * 2;

                for j in 0..NCOLS {
                    let mut sumi1 = 0i32;
                    let mut sumi2 = 0i32;
                    let mut sumi  = 0i32;
                    for i in 0..BLOCKLEN {
                        let b_qs_off = k * NCOLS * BLOCKLEN + j * BLOCKLEN + i;
                        let qh_idx   = (k * BLOCKLEN + i) % 32;
                        let qh_chunk = qh_idx / BLOCKLEN;
                        let qh_pos   = qh_idx % BLOCKLEN;
                        let b_qh_off = qh_chunk * (BLOCKLEN * NCOLS) + j * BLOCKLEN + qh_pos;
                        let qh_val   = b.qh[b_qh_off];
                        let h0 = ((qh_val as u32 >> qh_shift) & 1) as i32;
                        let h1 = ((qh_val as u32 >> (qh_shift + 1)) & 1) as i32;
                        let v0 = ((b.qs[b_qs_off] & 0xF) as i32 | (h0 << 4)) as i8 as i32;
                        let v1 = (((b.qs[b_qs_off] >> 4) & 0xF) as i32 | (h1 << 4)) as i8 as i32;
                        let q8_off = sg * 64 + (k % (32 / BLOCKLEN)) * BLOCKLEN + i;
                        sumi1 = v0 * a.qs[q8_off] as i32 * scales_0[j] as i32;
                        sumi2 = v1 * a.qs[q8_off + 32] as i32 * scales_1[j] as i32;
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi as f32 * f16_to_f32(b.d[j]) * a.d;
                }
            }

            for sb in 0..8usize {
                let mins = &utmp_bytes[8 + sb * 16..];
                for j in 0..NCOLS {
                    sum_minf[j] += (mins[j] as i32
                        * (a.bsums[sb * 2] as i32 + a.bsums[sb * 2 + 1] as i32)) as f32
                        * f16_to_f32(b.dmin[j]) * a.d;
                }
            }
        }
        for j in 0..NCOLS {
            *s.add(x * NCOLS + j) = sumf[j] - sum_minf[j];
        }
    }
}

// ── GEMV Q6_K (blocklen=4 variant — the 8x8 is in repack_llama.rs) ───────────

/// ggml_gemv_q6_K_8x4_q8_K_generic — blocklen=4, ncols=8.
pub unsafe fn gemv_q6_k_8x4_q8_k_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const BLOCKLEN: usize = 4;
    const NCOLS: usize = 8;

    let nb = n / QK_K;
    let blocks_per_half = 64 / BLOCKLEN;
    let a_ptr = vy as *const BlockQ8K;
    let b_base = vx as *const BlockQ6Kx8;
    let mut sumf = [0f32; NCOLS];

    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        for j in 0..NCOLS { sumf[j] = 0.0; }

        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);

            for k in 0..(QK_K / (2 * BLOCKLEN)) {
                let base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * BLOCKLEN;
                let base_h = base_l + 64;
                let scale_idx_l = base_l / 16;
                let scale_idx_h = base_h / 16;
                let qh_shift_l  = ((base_l % 128) / 32) * 2;
                let qh_shift_h  = ((base_h % 128) / 32) * 2;
                let qh_half_l   = (base_l / 128) * 32;
                let qh_half_h   = (base_h / 128) * 32;

                for j in 0..NCOLS {
                    let scale_l = b.scales[scale_idx_l * NCOLS + j] as i32;
                    let scale_h = b.scales[scale_idx_h * NCOLS + j] as i32;
                    let mut sumi_l = 0i32;
                    let mut sumi_h = 0i32;

                    for i in 0..BLOCKLEN {
                        let ql_pos = k * NCOLS * BLOCKLEN + j * BLOCKLEN + i;
                        let l_4  = (b.ql[ql_pos] & 0xF) as i32;
                        let hi_4 = ((b.ql[ql_pos] >> 4) & 0xF) as i32;

                        let qh_idx_l   = qh_half_l + ((base_l + i) % 32);
                        let qh_chunk_l = qh_idx_l / BLOCKLEN;
                        let qh_pos_l   = qh_idx_l % BLOCKLEN;
                        let qh_off_l   = qh_chunk_l * (BLOCKLEN * NCOLS) + j * BLOCKLEN + qh_pos_l;
                        let hi_2_l     = ((b.qh[qh_off_l] as u32 >> qh_shift_l) & 3) as i32;

                        let qh_idx_h   = qh_half_h + ((base_h + i) % 32);
                        let qh_chunk_h = qh_idx_h / BLOCKLEN;
                        let qh_pos_h   = qh_idx_h % BLOCKLEN;
                        let qh_off_h   = qh_chunk_h * (BLOCKLEN * NCOLS) + j * BLOCKLEN + qh_pos_h;
                        let hi_2_h     = ((b.qh[qh_off_h] as u32 >> qh_shift_h) & 3) as i32;

                        let q_l = ((hi_2_l << 4) | l_4) - 32;
                        let q_h = ((hi_2_h << 4) | hi_4) - 32;

                        sumi_l += q_l * a.qs[base_l + i] as i32;
                        sumi_h += q_h * a.qs[base_h + i] as i32;
                    }
                    sumf[j] += (sumi_l * scale_l + sumi_h * scale_h) as f32
                        * f16_to_f32(b.d[j]) * a.d;
                }
            }
        }
        for j in 0..NCOLS { *s.add(x * NCOLS + j) = sumf[j]; }
    }
}

// ── GEMV IQ4_NL ──────────────────────────────────────────────────────────────

/// ggml_gemv_iq4_nl_4x4_q8_0_generic — blocklen=4, ncols=4.
pub unsafe fn gemv_iq4_nl_4x4_q8_0_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 4;
    let nb = n / QK8_0;
    let a_ptr = vy as *const super::blocks::BlockQ8_0;
    let b_base = vx as *const BlockIQ4NLx4;
    let mut sumf = [0f32; NCOLS];
    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        for j in 0..NCOLS { sumf[j] = 0.0; }
        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);
            for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                for j in 0..NCOLS {
                    let mut sumi = 0i32;
                    for i in 0..BLOCKLEN {
                        let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i];
                        let v0 = KVALUES_IQ4NL[(byte & 0x0F) as usize] as i32;
                        let v1 = KVALUES_IQ4NL[(byte >> 4)  as usize] as i32;
                        sumi += v0 * a.qs[k * BLOCKLEN + i] as i32
                              + v1 * a.qs[k * BLOCKLEN + i + QK8_0 / 2] as i32;
                    }
                    sumf[j] += sumi as f32 * f16_to_f32(b.d[j]) * f16_to_f32(a.d);
                }
            }
        }
        for j in 0..NCOLS { *s.add(x * NCOLS + j) = sumf[j]; }
    }
}

/// ggml_gemv_iq4_nl_8x8_q8_0_generic — blocklen=8, ncols=8.
pub unsafe fn gemv_iq4_nl_8x8_q8_0_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const NCOLS: usize = 8;
    const BLOCKLEN: usize = 8;
    let nb = n / QK8_0;
    let a_ptr = vy as *const super::blocks::BlockQ8_0;
    let b_base = vx as *const BlockIQ4NLx8;
    let mut sumf = [0f32; NCOLS];
    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        for j in 0..NCOLS { sumf[j] = 0.0; }
        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);
            for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                for j in 0..NCOLS {
                    let mut sumi = 0i32;
                    for i in 0..BLOCKLEN {
                        let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i];
                        let v0 = KVALUES_IQ4NL[(byte & 0x0F) as usize] as i32;
                        let v1 = KVALUES_IQ4NL[(byte >> 4)  as usize] as i32;
                        sumi += v0 * a.qs[k * BLOCKLEN + i] as i32
                              + v1 * a.qs[k * BLOCKLEN + i + QK8_0 / 2] as i32;
                    }
                    sumf[j] += sumi as f32 * f16_to_f32(b.d[j]) * f16_to_f32(a.d);
                }
            }
        }
        for j in 0..NCOLS { *s.add(x * NCOLS + j) = sumf[j]; }
    }
}

// ── GEMV mxfp4 ───────────────────────────────────────────────────────────────

/// ggml_gemv_mxfp4_4x4_q8_0_generic — blocklen=4, ncols=4.
pub unsafe fn gemv_mxfp4_4x4_q8_0_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 4;
    let nb = n / QK8_0;
    let a_ptr = vy as *const super::blocks::BlockQ8_0;
    let b_base = vx as *const BlockMxfp4x4;
    let mut sumf = [0f32; NCOLS];
    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        for j in 0..NCOLS { sumf[j] = 0.0; }
        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);
            for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                for j in 0..NCOLS {
                    let mut sumi = 0i32;
                    for i in 0..BLOCKLEN {
                        let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i];
                        let v0 = KVALUES_MXFP4[(byte & 0x0F) as usize] as i32;
                        let v1 = KVALUES_MXFP4[(byte >> 4)  as usize] as i32;
                        sumi += v0 * a.qs[k * BLOCKLEN + i] as i32
                              + v1 * a.qs[k * BLOCKLEN + i + QK8_0 / 2] as i32;
                    }
                    sumf[j] += sumi as f32 * e8m0_to_f32(b.e[j]) * f16_to_f32(a.d);
                }
            }
        }
        for j in 0..NCOLS { *s.add(x * NCOLS + j) = sumf[j]; }
    }
}

/// ggml_gemv_mxfp4_8x8_q8_0_generic — blocklen=8, ncols=8.
pub unsafe fn gemv_mxfp4_8x8_q8_0_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const NCOLS: usize = 8;
    const BLOCKLEN: usize = 8;
    let nb = n / QK8_0;
    let a_ptr = vy as *const super::blocks::BlockQ8_0;
    let b_base = vx as *const BlockMxfp4x8;
    let mut sumf = [0f32; NCOLS];
    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        for j in 0..NCOLS { sumf[j] = 0.0; }
        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);
            for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                for j in 0..NCOLS {
                    let mut sumi = 0i32;
                    for i in 0..BLOCKLEN {
                        let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i];
                        let v0 = KVALUES_MXFP4[(byte & 0x0F) as usize] as i32;
                        let v1 = KVALUES_MXFP4[(byte >> 4)  as usize] as i32;
                        sumi += v0 * a.qs[k * BLOCKLEN + i] as i32
                              + v1 * a.qs[k * BLOCKLEN + i + QK8_0 / 2] as i32;
                    }
                    sumf[j] += sumi as f32 * e8m0_to_f32(b.e[j]) * f16_to_f32(a.d);
                }
            }
        }
        for j in 0..NCOLS { *s.add(x * NCOLS + j) = sumf[j]; }
    }
}

// ── GEMV Q8_0 ─────────────────────────────────────────────────────────────────

/// ggml_gemv_q8_0_4x4_q8_0_generic — blocklen=4, ncols=4.
pub unsafe fn gemv_q8_0_4x4_q8_0_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 4;
    let nb = n / QK8_0;
    let a_ptr = vy as *const super::blocks::BlockQ8_0;
    let b_base = vx as *const BlockQ8_0x4;
    let mut sumf = [0f32; NCOLS];
    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        for j in 0..NCOLS { sumf[j] = 0.0; }
        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);
            for k in 0..(QK8_0 / BLOCKLEN) {
                for j in 0..NCOLS {
                    let mut sumi = 0i32;
                    for i in 0..BLOCKLEN {
                        sumi += b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i] as i32
                              * a.qs[k * BLOCKLEN + i] as i32;
                    }
                    sumf[j] += sumi as f32 * f16_to_f32(b.d[j]) * f16_to_f32(a.d);
                }
            }
        }
        for j in 0..NCOLS { *s.add(x * NCOLS + j) = sumf[j]; }
    }
}

/// ggml_gemv_q8_0_4x8_q8_0_generic — blocklen=8, ncols=4.
pub unsafe fn gemv_q8_0_4x8_q8_0_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 8;
    let nb = n / QK8_0;
    let a_ptr = vy as *const super::blocks::BlockQ8_0;
    let b_base = vx as *const BlockQ8_0x4;
    let mut sumf = [0f32; NCOLS];
    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        for j in 0..NCOLS { sumf[j] = 0.0; }
        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);
            for k in 0..(QK8_0 / BLOCKLEN) {
                for j in 0..NCOLS {
                    let mut sumi = 0i32;
                    for i in 0..BLOCKLEN {
                        sumi += b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i] as i32
                              * a.qs[k * BLOCKLEN + i] as i32;
                    }
                    sumf[j] += sumi as f32 * f16_to_f32(b.d[j]) * f16_to_f32(a.d);
                }
            }
        }
        for j in 0..NCOLS { *s.add(x * NCOLS + j) = sumf[j]; }
    }
}

// ── GEMV Q1_0 ─────────────────────────────────────────────────────────────────

/// ggml_gemv_q1_0_4x4_q8_0_generic — blocklen=QK8_0 (4-wide SIMD tiles), ncols=4.
pub unsafe fn gemv_q1_0_4x4_q8_0_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    let nb = n / QK1_0;
    let a_ptr = vy as *const super::blocks::BlockQ8_0;
    let b_base = vx as *const BlockQ1_0x4;
    let mut sumf = [0f32; NCOLS];

    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        for j in 0..NCOLS { sumf[j] = 0.0; }

        for l in 0..nb {
            let b = &*b_ptr.add(l);
            let d0 = [
                f16_to_f32(b.d[0]), f16_to_f32(b.d[1]),
                f16_to_f32(b.d[2]), f16_to_f32(b.d[3]),
            ];
            for k in 0..(QK1_0 / QK8_0) {
                let a_blk = &*a_ptr.add(l * (QK1_0 / QK8_0) + k);
                let d1 = f16_to_f32(a_blk.d);
                let scale = [d0[0]*d1, d0[1]*d1, d0[2]*d1, d0[3]*d1];

                for tile in 0..(QK8_0 / 4) {
                    let bits_lo = b.qs[k * 16 + 2 * tile + 0] as u8;
                    let bits_hi = b.qs[k * 16 + 2 * tile + 1] as u8;
                    for p in 0..4usize {
                        let q = a_blk.qs[tile * 4 + p] as f32;
                        sumf[0] += (if bits_lo & (1 << p) != 0 { scale[0] } else { -scale[0] }) * q;
                        sumf[1] += (if bits_lo & (1 << (4+p)) != 0 { scale[1] } else { -scale[1] }) * q;
                        sumf[2] += (if bits_hi & (1 << p) != 0 { scale[2] } else { -scale[2] }) * q;
                        sumf[3] += (if bits_hi & (1 << (4+p)) != 0 { scale[3] } else { -scale[3] }) * q;
                    }
                }
            }
        }
        for j in 0..NCOLS { *s.add(x * NCOLS + j) = sumf[j]; }
    }
}

/// ggml_gemv_q1_0_4x8_q8_0_generic — blocklen=8, ncols=4.
pub unsafe fn gemv_q1_0_4x8_q8_0_generic(
    n: usize, s: *mut f32, _bs: usize,
    vx: *const u8, vy: *const u8, _nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 8;
    let nb = n / QK1_0;
    let a_ptr = vy as *const super::blocks::BlockQ8_0;
    let b_base = vx as *const BlockQ1_0x4;
    let mut sumf = [0f32; NCOLS];

    for x in 0..(nc / NCOLS) {
        let b_ptr = b_base.add(x * nb);
        for j in 0..NCOLS { sumf[j] = 0.0; }

        for l in 0..nb {
            let b = &*b_ptr.add(l);
            let d0 = [
                f16_to_f32(b.d[0]), f16_to_f32(b.d[1]),
                f16_to_f32(b.d[2]), f16_to_f32(b.d[3]),
            ];
            for k in 0..(QK1_0 / BLOCKLEN) {
                let a_blk = &*a_ptr.add(l * (QK1_0 / QK8_0) + k / (QK8_0 / BLOCKLEN));
                let d1 = f16_to_f32(a_blk.d);
                let scale = [d0[0]*d1, d0[1]*d1, d0[2]*d1, d0[3]*d1];
                let bits = [
                    b.qs[k * NCOLS + 0] as u8,
                    b.qs[k * NCOLS + 1] as u8,
                    b.qs[k * NCOLS + 2] as u8,
                    b.qs[k * NCOLS + 3] as u8,
                ];
                let q_off = (k % (QK8_0 / BLOCKLEN)) * BLOCKLEN;
                for p in 0..BLOCKLEN {
                    let q = a_blk.qs[q_off + p] as f32;
                    sumf[0] += (if bits[0] & (1 << p) != 0 { scale[0] } else { -scale[0] }) * q;
                    sumf[1] += (if bits[1] & (1 << p) != 0 { scale[1] } else { -scale[1] }) * q;
                    sumf[2] += (if bits[2] & (1 << p) != 0 { scale[2] } else { -scale[2] }) * q;
                    sumf[3] += (if bits[3] & (1 << p) != 0 { scale[3] } else { -scale[3] }) * q;
                }
            }
        }
        for j in 0..NCOLS { *s.add(x * NCOLS + j) = sumf[j]; }
    }
}

// ── GEMM Q4_0 ─────────────────────────────────────────────────────────────────

/// ggml_gemm_q4_0_4x4_q8_0_generic — blocklen=4, ncols=4, nrows=4.
pub unsafe fn gemm_q4_0_4x4_q8_0_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 4;
    let nb = n / QK8_0;
    let b_base = vx as *const BlockQ4_0x4;
    let a_base = vy as *const BlockQ8_0x4;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];
            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);
                for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi = 0i32;
                            for i in 0..BLOCKLEN {
                                let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i] as i32;
                                let v0 = ((byte << 4) as i8) as i32;
                                let v1 = ((byte & 0xF0i32) as i8) as i32;
                                sumi += (v0 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i] as i32
                                       + v1 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i + QK8_0 / 2 * 4] as i32) >> 4;
                            }
                            sumf[m][j] += sumi as f32
                                * f16_to_f32(b.d[j]) * f16_to_f32(a.d[m]);
                        }
                    }
                }
            }
            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
}

/// ggml_gemm_q4_0_4x8_q8_0_generic — blocklen=8, ncols=4.
pub unsafe fn gemm_q4_0_4x8_q8_0_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 8;
    let nb = n / QK8_0;
    let b_base = vx as *const BlockQ4_0x4;
    let a_base = vy as *const BlockQ8_0x4;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];
            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);
                for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi = 0i32;
                            for i in 0..BLOCKLEN {
                                let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i] as i32;
                                let v0 = ((byte << 4) as i8) as i32;
                                let v1 = ((byte & 0xF0i32) as i8) as i32;
                                sumi += (v0 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i] as i32
                                       + v1 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i + QK8_0 / 2 * 4] as i32) >> 4;
                            }
                            sumf[m][j] += sumi as f32
                                * f16_to_f32(b.d[j]) * f16_to_f32(a.d[m]);
                        }
                    }
                }
            }
            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
}

/// ggml_gemm_q4_0_8x8_q8_0_generic — blocklen=8, ncols=8.
pub unsafe fn gemm_q4_0_8x8_q8_0_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const NCOLS: usize = 8;
    const BLOCKLEN: usize = 8;
    let nb = n / QK8_0;
    let b_base = vx as *const BlockQ4_0x8;
    let a_base = vy as *const BlockQ8_0x4;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];
            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);
                for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi = 0i32;
                            for i in 0..BLOCKLEN {
                                let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i] as i32;
                                let v0 = ((byte << 4) as i8) as i32;
                                let v1 = ((byte & 0xF0i32) as i8) as i32;
                                sumi += (v0 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i] as i32
                                       + v1 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i + QK8_0 / 2 * 4] as i32) >> 4;
                            }
                            sumf[m][j] += sumi as f32
                                * f16_to_f32(b.d[j]) * f16_to_f32(a.d[m]);
                        }
                    }
                }
            }
            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
}

// ── GEMM Q4_K ─────────────────────────────────────────────────────────────────

unsafe fn gemm_q4_k_q8_k_impl<const BLOCKLEN: usize, const NCOLS: usize>(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    let nb = n / QK_K;
    let b_base = vx as *const BlockQ4Kx8;
    let a_base = vy as *const BlockQ8Kx4;
    let mut utmp_all = [0u32; 32];

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf     = [[0f32; NCOLS]; 4];
            let mut sum_minf = [[0f32; NCOLS]; 4];

            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);

                for sb in 0..8usize {
                    let mut tmp = [0u32; 4];
                    decode_q4k_scales(&b.scales[sb * 12..], &mut tmp);
                    utmp_all[sb*4] = tmp[0]; utmp_all[sb*4+1] = tmp[1];
                    utmp_all[sb*4+2] = tmp[2]; utmp_all[sb*4+3] = tmp[3];
                }
                let utmp_bytes = std::slice::from_raw_parts(utmp_all.as_ptr() as *const u8, 128);

                for k in 0..(QK_K / (2 * BLOCKLEN)) {
                    let scale_grp = if BLOCKLEN == 4 { k / 8 } else { k >> 2 };
                    let scales_0 = &utmp_bytes[scale_grp * 32..];
                    let scales_1 = &utmp_bytes[scale_grp * 32 + 16..];

                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi1 = 0i32;
                            let mut sumi2 = 0i32;
                            for i in 0..BLOCKLEN {
                                let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i];
                                let v0 = (byte & 0xF) as i32;
                                let v1 = (byte >> 4) as i32;
                                let q8_idx = if BLOCKLEN == 4 {
                                    (k / 8) * 256 + (k % 8) * 4 * BLOCKLEN + m * BLOCKLEN + i
                                } else {
                                    (k >> 2) * 256 + (k % 4) * 4 * BLOCKLEN + m * BLOCKLEN + i
                                };
                                sumi1 = v0 * a.qs[q8_idx] as i32 * scales_0[j] as i32;
                                sumi2 = v1 * a.qs[q8_idx + 128] as i32 * scales_1[j] as i32;
                                sumf[m][j] += (sumi1 + sumi2) as f32
                                    * f16_to_f32(b.d[j]) * a.d[m];
                            }
                            let _ = (sumi1, sumi2);
                        }
                    }
                }

                for sb in 0..8usize {
                    let mins_ptr = utmp_bytes.as_ptr().add(8 + sb * 16);
                    let mins = std::slice::from_raw_parts(mins_ptr, NCOLS);
                    for m in 0..4usize {
                        // bsums index mirrors C: (sb * 8) + (m * 4) - ((sb % 2) * 6)
                        let bsums_base = (sb * 8 + m * 4).wrapping_sub((sb % 2) * 6);
                        let bs0 = a.bsums[bsums_base] as i32;
                        let bs1 = a.bsums[bsums_base + 1] as i32;
                        for j in 0..NCOLS {
                            sum_minf[m][j] += mins[j] as f32 * (bs0 + bs1) as f32
                                * f16_to_f32(b.dmin[j]) * a.d[m];
                        }
                    }
                }
            }

            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

/// ggml_gemm_q4_K_8x4_q8_K_generic — blocklen=4, ncols=8.
pub unsafe fn gemm_q4_k_8x4_q8_k_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    gemm_q4_k_q8_k_impl::<4, 8>(n, s, bs, vx, vy, nr, nc);
}

/// ggml_gemm_q4_K_8x8_q8_K_generic — blocklen=8, ncols=8.
pub unsafe fn gemm_q4_k_8x8_q8_k_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    gemm_q4_k_q8_k_impl::<8, 8>(n, s, bs, vx, vy, nr, nc);
}

// ── GEMM Q2_K ─────────────────────────────────────────────────────────────────

/// ggml_gemm_q2_K_8x8_q8_K_generic — blocklen=8, ncols=8.
pub unsafe fn gemm_q2_k_8x8_q8_k_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const NCOLS: usize = 8;
    const BLOCKLEN: usize = 8;
    let nb = n / QK_K;
    let b_base = vx as *const BlockQ2Kx8;
    let a_base = vy as *const BlockQ8Kx4;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf     = [[0f32; NCOLS]; 4];
            let mut sum_minf = [[0f32; NCOLS]; 4];

            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);

                for k in 0..(QK_K / (4 * BLOCKLEN)) {
                    let scales_0 = &b.scales[(k / 4) * 64..];
                    let scales_1 = &b.scales[(k / 4) * 64 + 16..];
                    let scales_2 = &b.scales[(k / 4) * 64 + 32..];
                    let scales_3 = &b.scales[(k / 4) * 64 + 48..];

                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi = 0i32;
                            let offset = ((k / 2) % 2) + j * 2;
                            for i in 0..BLOCKLEN {
                                let qs = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i];
                                let v0 = (qs & 3) as i32;
                                let v1 = ((qs >> 2) & 3) as i32;
                                let v2 = ((qs >> 4) & 3) as i32;
                                let v3 = (qs >> 6) as i32;
                                let base = (k >> 2) * 512 + (k % 4) * 4 * BLOCKLEN + m * BLOCKLEN + i;
                                let s1 = v0 * a.qs[base] as i32 * (scales_0[offset] & 0xF) as i32;
                                let s2 = v1 * a.qs[base + 128] as i32 * (scales_1[offset] & 0xF) as i32;
                                let s3 = v2 * a.qs[base + 256] as i32 * (scales_2[offset] & 0xF) as i32;
                                let s4 = v3 * a.qs[base + 384] as i32 * (scales_3[offset] & 0xF) as i32;
                                sumi += s1 + s2 + s3 + s4;
                            }
                            sumf[m][j] += sumi as f32
                                * f16_to_f32(b.d[j]) * a.d[m];
                        }
                    }
                }

                for sb in 0..8usize {
                    let mins = &b.scales[sb * 16..];
                    for m in 0..4usize {
                        let bsums_base = (sb * 8 + m * 4).wrapping_sub((sb % 2) * 6);
                        let bs0 = a.bsums[bsums_base] as i32;
                        let bs1 = a.bsums[bsums_base + 1] as i32;
                        for j in 0..NCOLS {
                            let mins_prod = (mins[j * 2] >> 4) as i32 * bs0
                                          + (mins[j * 2 + 1] >> 4) as i32 * bs1;
                            sum_minf[m][j] += mins_prod as f32
                                * f16_to_f32(b.dmin[j]) * a.d[m];
                        }
                    }
                }
            }

            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

// ── GEMM Q5_K ─────────────────────────────────────────────────────────────────

/// ggml_gemm_q5_K_8x8_q8_K_generic — blocklen=8, ncols=8 (8x4 is in repack_llama.rs).
pub unsafe fn gemm_q5_k_8x8_q8_k_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const BLOCKLEN: usize = 8;
    const NCOLS: usize = 8;
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let nb = n / QK_K;
    let b_base = vx as *const BlockQ5Kx8;
    let a_base = vy as *const BlockQ8Kx4;
    let mut utmp = [0u32; 32];

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf     = [[0f32; NCOLS]; 4];
            let mut sum_minf = [[0f32; NCOLS]; 4];

            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);

                for sb in 0..8usize {
                    let src = &b.scales[sb * 12..];
                    let u0 = u32::from_le_bytes(src[0..4].try_into().unwrap());
                    let u1 = u32::from_le_bytes(src[4..8].try_into().unwrap());
                    let u2 = u32::from_le_bytes(src[8..12].try_into().unwrap());
                    utmp[sb*4+3] = ((u2>>4)&KMASK2)|(((u1>>6)&KMASK3)<<4);
                    let uaux = u1 & KMASK1;
                    utmp[sb*4+1] = (u2&KMASK2)|(((u0>>6)&KMASK3)<<4);
                    utmp[sb*4+2] = uaux;
                    utmp[sb*4+0] = u0 & KMASK1;
                }
                let utmp_bytes = std::slice::from_raw_parts(utmp.as_ptr() as *const u8, 128);

                for k in 0..(QK_K / (2 * BLOCKLEN)) {
                    const SCALE_STRIDE: usize = 32;
                    let sg = k / (32 / BLOCKLEN);
                    let scales_0 = &utmp_bytes[sg * SCALE_STRIDE..];
                    let scales_1 = &utmp_bytes[sg * SCALE_STRIDE + 16..];
                    let qh_shift = sg * 2;

                    // Q8 in q8_Kx4: stride per block = 256, within block stride = 4*BLOCKLEN
                    // q8_half_stride=512, q8_low_high_step=256 (from the GEMM template)
                    let q8_base = (k / (32 / BLOCKLEN)) * 512 + (k % (32 / BLOCKLEN)) * (BLOCKLEN * 4);

                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi1 = 0i32;
                            let mut sumi2 = 0i32;
                            for i in 0..BLOCKLEN {
                                let b_qs_off = k * NCOLS * BLOCKLEN + j * BLOCKLEN + i;
                                let qh_idx   = (k * BLOCKLEN + i) % 32;
                                let qh_chunk = qh_idx / BLOCKLEN;
                                let qh_pos   = qh_idx % BLOCKLEN;
                                let b_qh_off = qh_chunk * (BLOCKLEN * NCOLS) + j * BLOCKLEN + qh_pos;
                                let qh_val   = b.qh[b_qh_off];
                                let h0 = ((qh_val as u32 >> qh_shift) & 1) as i32;
                                let h1 = ((qh_val as u32 >> (qh_shift+1)) & 1) as i32;
                                let v0 = ((b.qs[b_qs_off] & 0xF) as i32 | (h0 << 4)) as i8 as i32;
                                let v1 = (((b.qs[b_qs_off] >> 4) & 0xF) as i32 | (h1 << 4)) as i8 as i32;
                                let q8_l = a.qs[q8_base + m * BLOCKLEN + i] as i32;
                                let q8_h = a.qs[q8_base + m * BLOCKLEN + i + 256] as i32;
                                sumi1 = v0 * q8_l * scales_0[j] as i32;
                                sumi2 = v1 * q8_h * scales_1[j] as i32;
                                sumf[m][j] += (sumi1 + sumi2) as f32
                                    * f16_to_f32(b.d[j]) * a.d[m];
                            }
                            let _ = (sumi1, sumi2);
                        }
                    }
                }

                for sb in 0..8usize {
                    let mins = &utmp_bytes[8 + sb * 16..];
                    for m in 0..4usize {
                        let bsums_base = (sb * 8 + m * 4).wrapping_sub((sb % 2) * 6);
                        let bs0 = a.bsums[bsums_base] as i32;
                        let bs1 = a.bsums[bsums_base + 1] as i32;
                        for j in 0..NCOLS {
                            sum_minf[m][j] += mins[j] as f32 * (bs0 + bs1) as f32
                                * f16_to_f32(b.dmin[j]) * a.d[m];
                        }
                    }
                }
            }

            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

// ── GEMM Q6_K ─────────────────────────────────────────────────────────────────

/// ggml_gemm_q6_K_8x8_q8_K_generic — blocklen=8, ncols=8 (8x4 is in repack_llama.rs).
pub unsafe fn gemm_q6_k_8x8_q8_k_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const BLOCKLEN: usize = 8;
    const NCOLS: usize = 8;

    let nb = n / QK_K;
    let blocks_per_half = 64 / BLOCKLEN;
    let b_base = vx as *const BlockQ6Kx8;
    let a_base = vy as *const BlockQ8Kx4;
    let q8_half_stride   = 512usize;
    let q8_low_high_step = 256usize;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];

            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);

                for k in 0..(QK_K / (2 * BLOCKLEN)) {
                    let base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * BLOCKLEN;
                    let base_h = base_l + 64;
                    let scale_idx_l = base_l / 16;
                    let scale_idx_h = base_h / 16;
                    let qh_shift_l  = ((base_l % 128) / 32) * 2;
                    let qh_shift_h  = ((base_h % 128) / 32) * 2;
                    let qh_half_l   = (base_l / 128) * 32;
                    let qh_half_h   = (base_h / 128) * 32;
                    let q8_base_k = (k / blocks_per_half) * q8_half_stride
                        + (k % blocks_per_half) * (BLOCKLEN * 4);

                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let scale_l = b.scales[scale_idx_l * NCOLS + j] as i32;
                            let scale_h = b.scales[scale_idx_h * NCOLS + j] as i32;
                            let mut sumi_l = 0i32;
                            let mut sumi_h = 0i32;

                            for i in 0..BLOCKLEN {
                                let ql_pos = k * NCOLS * BLOCKLEN + j * BLOCKLEN + i;
                                let l_4  = (b.ql[ql_pos] & 0xF) as i32;
                                let hi_4 = ((b.ql[ql_pos] >> 4) & 0xF) as i32;

                                let qh_idx_l   = qh_half_l + ((base_l + i) % 32);
                                let qh_chunk_l = qh_idx_l / BLOCKLEN;
                                let qh_pos_l   = qh_idx_l % BLOCKLEN;
                                let qh_off_l   = qh_chunk_l * (BLOCKLEN*NCOLS) + j*BLOCKLEN + qh_pos_l;
                                let hi_2_l     = ((b.qh[qh_off_l] as u32 >> qh_shift_l) & 3) as i32;

                                let qh_idx_h   = qh_half_h + ((base_h + i) % 32);
                                let qh_chunk_h = qh_idx_h / BLOCKLEN;
                                let qh_pos_h   = qh_idx_h % BLOCKLEN;
                                let qh_off_h   = qh_chunk_h * (BLOCKLEN*NCOLS) + j*BLOCKLEN + qh_pos_h;
                                let hi_2_h     = ((b.qh[qh_off_h] as u32 >> qh_shift_h) & 3) as i32;

                                let q_l = ((hi_2_l << 4) | l_4) - 32;
                                let q_h = ((hi_2_h << 4) | hi_4) - 32;

                                let q8_l = a.qs[q8_base_k + m * BLOCKLEN + i] as i32;
                                let q8_h = a.qs[q8_base_k + m * BLOCKLEN + i + q8_low_high_step] as i32;

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }
                            sumf[m][j] += (sumi_l * scale_l + sumi_h * scale_h) as f32
                                * f16_to_f32(b.d[j]) * a.d[m];
                        }
                    }
                }
            }

            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
}

// ── GEMM IQ4_NL ───────────────────────────────────────────────────────────────

/// ggml_gemm_iq4_nl_4x4_q8_0_generic — blocklen=4, ncols=4.
pub unsafe fn gemm_iq4_nl_4x4_q8_0_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 4;
    let nb = n / QK8_0;
    let b_base = vx as *const BlockIQ4NLx4;
    let a_base = vy as *const BlockQ8_0x4;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];
            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);
                for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi = 0i32;
                            for i in 0..BLOCKLEN {
                                let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i];
                                let v0 = KVALUES_IQ4NL[(byte & 0x0F) as usize] as i32;
                                let v1 = KVALUES_IQ4NL[(byte >> 4) as usize] as i32;
                                sumi += v0 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i] as i32
                                      + v1 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i + QK8_0 / 2 * 4] as i32;
                            }
                            sumf[m][j] += sumi as f32
                                * f16_to_f32(b.d[j]) * f16_to_f32(a.d[m]);
                        }
                    }
                }
            }
            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
}

/// ggml_gemm_iq4_nl_8x8_q8_0_generic — blocklen=8, ncols=8.
pub unsafe fn gemm_iq4_nl_8x8_q8_0_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const NCOLS: usize = 8;
    const BLOCKLEN: usize = 8;
    let nb = n / QK8_0;
    let b_base = vx as *const BlockIQ4NLx8;
    let a_base = vy as *const BlockQ8_0x4;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];
            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);
                for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi = 0i32;
                            for i in 0..BLOCKLEN {
                                let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i];
                                let v0 = KVALUES_IQ4NL[(byte & 0x0F) as usize] as i32;
                                let v1 = KVALUES_IQ4NL[(byte >> 4) as usize] as i32;
                                sumi += v0 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i] as i32
                                      + v1 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i + QK8_0 / 2 * 4] as i32;
                            }
                            sumf[m][j] += sumi as f32
                                * f16_to_f32(b.d[j]) * f16_to_f32(a.d[m]);
                        }
                    }
                }
            }
            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
}

// ── GEMM mxfp4 ───────────────────────────────────────────────────────────────

/// ggml_gemm_mxfp4_4x4_q8_0_generic — blocklen=4, ncols=4.
pub unsafe fn gemm_mxfp4_4x4_q8_0_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 4;
    let nb = n / QK8_0;
    let b_base = vx as *const BlockMxfp4x4;
    let a_base = vy as *const BlockQ8_0x4;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];
            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);
                for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi = 0i32;
                            for i in 0..BLOCKLEN {
                                let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i];
                                let v0 = KVALUES_MXFP4[(byte & 0x0F) as usize] as i32;
                                let v1 = KVALUES_MXFP4[(byte >> 4) as usize] as i32;
                                sumi += v0 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i] as i32
                                      + v1 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i + QK8_0 / 2 * 4] as i32;
                            }
                            sumf[m][j] += sumi as f32
                                * e8m0_to_f32(b.e[j]) * f16_to_f32(a.d[m]);
                        }
                    }
                }
            }
            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
}

/// ggml_gemm_mxfp4_8x8_q8_0_generic — blocklen=8, ncols=8.
pub unsafe fn gemm_mxfp4_8x8_q8_0_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const NCOLS: usize = 8;
    const BLOCKLEN: usize = 8;
    let nb = n / QK8_0;
    let b_base = vx as *const BlockMxfp4x8;
    let a_base = vy as *const BlockQ8_0x4;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];
            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);
                for k in 0..(QK8_0 / (2 * BLOCKLEN)) {
                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi = 0i32;
                            for i in 0..BLOCKLEN {
                                let byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i];
                                let v0 = KVALUES_MXFP4[(byte & 0x0F) as usize] as i32;
                                let v1 = KVALUES_MXFP4[(byte >> 4) as usize] as i32;
                                sumi += v0 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i] as i32
                                      + v1 * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i + QK8_0 / 2 * 4] as i32;
                            }
                            sumf[m][j] += sumi as f32
                                * e8m0_to_f32(b.e[j]) * f16_to_f32(a.d[m]);
                        }
                    }
                }
            }
            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
}

// ── GEMM Q8_0 ─────────────────────────────────────────────────────────────────

/// ggml_gemm_q8_0_4x4_q8_0_generic — blocklen=4, ncols=4.
pub unsafe fn gemm_q8_0_4x4_q8_0_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 4;
    let nb = n / QK8_0;
    let b_base = vx as *const BlockQ8_0x4;
    let a_base = vy as *const BlockQ8_0x4;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];
            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);
                for k in 0..(QK8_0 / BLOCKLEN) {
                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi = 0i32;
                            for i in 0..BLOCKLEN {
                                sumi += b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i] as i32
                                      * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i] as i32;
                            }
                            sumf[m][j] += sumi as f32
                                * f16_to_f32(b.d[j]) * f16_to_f32(a.d[m]);
                        }
                    }
                }
            }
            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
}

/// ggml_gemm_q8_0_4x8_q8_0_generic — blocklen=8, ncols=4.
pub unsafe fn gemm_q8_0_4x8_q8_0_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 8;
    let nb = n / QK8_0;
    let b_base = vx as *const BlockQ8_0x4;
    let a_base = vy as *const BlockQ8_0x4;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];
            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);
                for k in 0..(QK8_0 / BLOCKLEN) {
                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi = 0i32;
                            for i in 0..BLOCKLEN {
                                sumi += b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i] as i32
                                      * a.qs[k * 4 * BLOCKLEN + m * BLOCKLEN + i] as i32;
                            }
                            sumf[m][j] += sumi as f32
                                * f16_to_f32(b.d[j]) * f16_to_f32(a.d[m]);
                        }
                    }
                }
            }
            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
}

// ── GEMM Q1_0 ─────────────────────────────────────────────────────────────────

/// ggml_gemm_q1_0_4x4_q8_0_generic — ncols=4, nrows=4.
pub unsafe fn gemm_q1_0_4x4_q8_0_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    let nb = n / QK1_0;
    let b_base = vx as *const BlockQ1_0x4;
    let a_base = vy as *const BlockQ8_0x4;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(4 * y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];

            for l in 0..nb {
                let b = &*b_ptr.add(l);
                let d0 = [
                    f16_to_f32(b.d[0]), f16_to_f32(b.d[1]),
                    f16_to_f32(b.d[2]), f16_to_f32(b.d[3]),
                ];

                for k in 0..(QK1_0 / QK8_0) {
                    let a_blk = &*a_ptr.add(4 * l + k);
                    let a_d = [
                        f16_to_f32(a_blk.d[0]), f16_to_f32(a_blk.d[1]),
                        f16_to_f32(a_blk.d[2]), f16_to_f32(a_blk.d[3]),
                    ];

                    for tile in 0..(QK8_0 / 4) {
                        let bits_lo = b.qs[k * 16 + 2 * tile + 0] as u8;
                        let bits_hi = b.qs[k * 16 + 2 * tile + 1] as u8;
                        let tile_offset = tile * 16;

                        for p in 0..4usize {
                            let q_row = [
                                a_blk.qs[tile_offset + 0 * 4 + p] as f32,
                                a_blk.qs[tile_offset + 1 * 4 + p] as f32,
                                a_blk.qs[tile_offset + 2 * 4 + p] as f32,
                                a_blk.qs[tile_offset + 3 * 4 + p] as f32,
                            ];
                            let sign = [
                                if bits_lo & (1 << p) != 0 { 1.0f32 } else { -1.0 },
                                if bits_lo & (1 << (4+p)) != 0 { 1.0 } else { -1.0 },
                                if bits_hi & (1 << p) != 0 { 1.0 } else { -1.0 },
                                if bits_hi & (1 << (4+p)) != 0 { 1.0 } else { -1.0 },
                            ];
                            for m in 0..4usize {
                                sumf[m][0] += sign[0] * q_row[m] * d0[0] * a_d[m];
                                sumf[m][1] += sign[1] * q_row[m] * d0[1] * a_d[m];
                                sumf[m][2] += sign[2] * q_row[m] * d0[2] * a_d[m];
                                sumf[m][3] += sign[3] * q_row[m] * d0[3] * a_d[m];
                            }
                        }
                    }
                }
            }

            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
}

/// ggml_gemm_q1_0_4x8_q8_0_generic — blocklen=8, ncols=4, nrows=4.
pub unsafe fn gemm_q1_0_4x8_q8_0_generic(
    n: usize, s: *mut f32, bs: usize,
    vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    const NCOLS: usize = 4;
    const BLOCKLEN: usize = 8;
    let nb = n / QK1_0;
    let b_base = vx as *const BlockQ1_0x4;
    let a_base = vy as *const BlockQ8_0x4;

    for y in 0..(nr / 4) {
        let a_ptr = a_base.add(4 * y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];

            for l in 0..nb {
                let b = &*b_ptr.add(l);
                let d0 = [
                    f16_to_f32(b.d[0]), f16_to_f32(b.d[1]),
                    f16_to_f32(b.d[2]), f16_to_f32(b.d[3]),
                ];

                for k in 0..(QK1_0 / BLOCKLEN) {
                    let a_blk = &*a_ptr.add(4 * l + k / (QK8_0 / BLOCKLEN));
                    let a_d = [
                        f16_to_f32(a_blk.d[0]), f16_to_f32(a_blk.d[1]),
                        f16_to_f32(a_blk.d[2]), f16_to_f32(a_blk.d[3]),
                    ];
                    let bits = [
                        b.qs[k * NCOLS + 0] as u8,
                        b.qs[k * NCOLS + 1] as u8,
                        b.qs[k * NCOLS + 2] as u8,
                        b.qs[k * NCOLS + 3] as u8,
                    ];
                    let q_offset = (k % (QK8_0 / BLOCKLEN)) * 4 * BLOCKLEN;

                    for p in 0..BLOCKLEN {
                        let q_row = [
                            a_blk.qs[q_offset + 0 * BLOCKLEN + p] as f32,
                            a_blk.qs[q_offset + 1 * BLOCKLEN + p] as f32,
                            a_blk.qs[q_offset + 2 * BLOCKLEN + p] as f32,
                            a_blk.qs[q_offset + 3 * BLOCKLEN + p] as f32,
                        ];
                        let sign = [
                            if bits[0] & (1 << p) != 0 { 1.0f32 } else { -1.0 },
                            if bits[1] & (1 << p) != 0 { 1.0 } else { -1.0 },
                            if bits[2] & (1 << p) != 0 { 1.0 } else { -1.0 },
                            if bits[3] & (1 << p) != 0 { 1.0 } else { -1.0 },
                        ];
                        for m in 0..4usize {
                            sumf[m][0] += sign[0] * q_row[m] * d0[0] * a_d[m];
                            sumf[m][1] += sign[1] * q_row[m] * d0[1] * a_d[m];
                            sumf[m][2] += sign[2] * q_row[m] * d0[2] * a_d[m];
                            sumf[m][3] += sign[3] * q_row[m] * d0[3] * a_d[m];
                        }
                    }
                }
            }

            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
}
