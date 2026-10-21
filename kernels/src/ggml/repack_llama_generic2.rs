//! Direct translation of llama.cpp ggml-cpu/repack.cpp lines 2598-5253.
//!
//! Covers:
//! - ggml_gemm_q1_0_4x8_q8_0_generic       (lines 2598-2681)
//! - RISC-V generic GEMM stubs              (lines 2683-3018, cfg-gated)
//! - make_block_q1_0x4                      (lines 3039-3081)
//! - make_block_q4_0x4                      (lines 3083-3122)
//! - make_block_q4_0x16                     (lines 3152-3175)
//! - make_block_q4_kx16                     (lines 3254-3304)
//! - make_block_q2_kx16                     (lines 3479-3539)
//! - make_block_q8_0x16                     (lines 3885-3906)
//! - make_block_iq4_nlx4/x8/x16             (lines 3939-4086)
//! - make_block_mxfp4x4/x8                  (lines 4121-4200)
//! - repack_q4_0_to_q4_0_4_bl               (lines 3541-3570)
//! - repack_q4_k_to_q4_k_16_bl              (lines 3603-3631)
//! - repack_q2_k_to_q2_k_16_bl              (lines 3664-3697)
//! - repack_q4_0_to_q4_0_16_bl              (lines 3699-3727)
//! - repack_q8_0_to_q8_0_4_bl               (lines 3821-3851)
//! - repack_q1_0_to_q1_0_4_bl               (lines 3853-3883)
//! - repack_q8_0_to_q8_0_16_bl              (lines 3908-3937)
//! - repack_iq4_nl_to_iq4_nl_{4,8,16}_bl    (lines 3974-4119)
//! - repack_mxfp4_to_mxfp4_{4,8}_bl         (lines 4145-4233)
//! - get_optimal_repack_type                 (lines 4925-5141)

use super::blocks::{
    BlockIQ4NL, BlockMxfp4, BlockQ1_0, BlockQ2K, BlockQ4K, BlockQ4_0, BlockQ8_0, BlockQ8Kx4,
};
use super::repack_llama2::{make_block_q8_0x4, BlockQ8_0x4};

// -- Constants -----------------------------------------------------------------

const QK1_0: usize = 128;
const QK4_0: usize = 32;
const QK8_0: usize = 32;
const QK_K: usize = 256;
const QK4_NL: usize = 32;
const QK_MXFP4: usize = 32;

// -- f16 helper ----------------------------------------------------------------

#[inline(always)]
fn f16_to_f32(v: u16) -> f32 {
    half::f16::from_bits(v).to_f32()
}

// -- New interleaved block structs ---------------------------------------------

/// 4 Q1_0 blocks interleaved (block_q1_0x4 from repack.h).
/// d[4] f16 scales + qs[64] packed bits.
/// Total: 8 + 64 = 72 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ1_0x4 {
    pub d:  [u16; 4],
    pub qs: [u8; 64],
}

/// 4 Q4_0 blocks interleaved (block_q4_0x4 from repack.h).
/// d[4] f16 scales + qs[64] nibbles XOR'd.
/// Total: 8 + 64 = 72 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ4_0x4 {
    pub d:  [u16; 4],
    pub qs: [u8; 64],
}

/// 16 Q4_0 blocks interleaved (RISC-V 16x1, block_q4_0x16).
/// d[16] f16 scales + qs[256] nibbles.
/// Total: 32 + 256 = 288 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ4_0x16 {
    pub d:  [u16; 16],
    pub qs: [u8; 256],
}

/// 16 Q4K blocks interleaved (RISC-V 16x1, block_q4_Kx16).
/// d[16] + dmin[16] + scales[192] + qs[2048].
/// Total: 32 + 32 + 192 + 2048 = 2304 bytes.
#[repr(C)]
pub struct BlockQ4Kx16 {
    pub d:      [u16; 16],
    pub dmin:   [u16; 16],
    pub scales: [u8; 192],
    pub qs:     [u8; 2048],
}

/// 16 Q2K blocks interleaved (RISC-V 16x1, block_q2_Kx16).
/// d[16] + dmin[16] + scales[256] + qs[1024].
/// Total: 32 + 32 + 256 + 1024 = 1344 bytes.
#[repr(C)]
pub struct BlockQ2Kx16 {
    pub d:      [u16; 16],
    pub dmin:   [u16; 16],
    pub scales: [u8; 256],
    pub qs:     [u8; 1024],
}

/// 16 Q8_0 blocks interleaved (RISC-V 16x1, block_q8_0x16).
/// d[16] f16 scales + qs[512].
/// Total: 32 + 512 = 544 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8_0x16 {
    pub d:  [u16; 16],
    pub qs: [i8; 512],
}

/// 4 IQ4_NL blocks interleaved (block_iq4_nlx4).
/// d[4] f16 + qs[64].  Total: 72 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockIQ4NLx4 {
    pub d:  [u16; 4],
    pub qs: [u8; 64],
}

/// 8 IQ4_NL blocks interleaved (block_iq4_nlx8).
/// d[8] f16 + qs[128].  Total: 144 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockIQ4NLx8 {
    pub d:  [u16; 8],
    pub qs: [u8; 128],
}

/// 16 IQ4_NL blocks interleaved (RISC-V, block_iq4_nlx16).
/// d[16] f16 + qs[256].  Total: 288 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockIQ4NLx16 {
    pub d:  [u16; 16],
    pub qs: [u8; 256],
}

/// 4 MXFP4 blocks interleaved (block_mxfp4x4).
/// e[4] E8M0 scales + qs[64].  Total: 68 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockMxfp4x4 {
    pub e:  [u8; 4],
    pub qs: [u8; 64],
}

/// 8 MXFP4 blocks interleaved (block_mxfp4x8).
/// e[8] E8M0 scales + qs[128].  Total: 136 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockMxfp4x8 {
    pub e:  [u8; 8],
    pub qs: [u8; 128],
}

// -- Size assertions ------------------------------------------------------------

const _: () = assert!(std::mem::size_of::<BlockQ1_0x4>() == 72);
const _: () = assert!(std::mem::size_of::<BlockQ4_0x4>() == 72);
const _: () = assert!(std::mem::size_of::<BlockQ4_0x16>() == 288);
const _: () = assert!(std::mem::size_of::<BlockQ4Kx16>() == 2304);
const _: () = assert!(std::mem::size_of::<BlockQ2Kx16>() == 1344);
const _: () = assert!(std::mem::size_of::<BlockQ8_0x16>() == 544);
const _: () = assert!(std::mem::size_of::<BlockIQ4NLx4>() == 72);
const _: () = assert!(std::mem::size_of::<BlockIQ4NLx8>() == 144);
const _: () = assert!(std::mem::size_of::<BlockIQ4NLx16>() == 288);
const _: () = assert!(std::mem::size_of::<BlockMxfp4x4>() == 68);
const _: () = assert!(std::mem::size_of::<BlockMxfp4x8>() == 136);

// -- IQ4_NL lookup table -------------------------------------------------------

/// Non-linear 4-bit lookup table from ggml-quants.h.
#[rustfmt::skip]
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

// -- GEMM generic implementation ------------------------------------------------

/// Direct translation of ggml_gemm_q1_0_4x8_q8_0_generic (repack.cpp lines 2598-2681).
///
/// Computes C += A * B where:
///   A (lhs) = Q8_0 activations packed as BlockQ8_0x4 (4-row interleaved)
///   B (rhs) = Q1_0 weights packed as BlockQ1_0x4 (4-col interleaved, blocklen=8)
///
/// # Safety
/// `vx` and `vy` must point to valid packed blocks; `s` must have room for nr*bs floats.
pub unsafe fn ggml_gemm_q1_0_4x8_q8_0_generic(
    n:   i32,
    s:   *mut f32,
    bs:  usize,
    vx:  *const core::ffi::c_void,
    vy:  *const core::ffi::c_void,
    nr:  i32,
    nc:  i32,
) {
    let qk: usize = QK1_0;
    let nb: usize = n as usize / qk;
    let ncols_interleaved: usize = 4;
    let blocklen: usize = 8;

    debug_assert_eq!(n as usize % qk, 0);
    debug_assert_eq!(nr as usize % 4, 0);
    debug_assert_eq!(nc as usize % ncols_interleaved, 0);

    let a_base = vy as *const BlockQ8_0x4;
    let b_base = vx as *const BlockQ1_0x4;

    for y in 0..(nr as usize / 4) {
        let a_ptr = a_base.add(4 * y * nb);
        for x in 0..(nc as usize / ncols_interleaved) {
            let b_ptr = b_base.add(x * nb);

            let mut sumf = [[0.0f32; 4]; 4];

            for l in 0..nb {
                let b = &*b_ptr.add(l);
                let d0 = [
                    f16_to_f32(b.d[0]),
                    f16_to_f32(b.d[1]),
                    f16_to_f32(b.d[2]),
                    f16_to_f32(b.d[3]),
                ];

                for k in 0..(qk / blocklen) {
                    // a_blk index: 4*l + k / (QK8_0 / blocklen)
                    let a_blk = &*a_ptr.add(4 * l + k / (QK8_0 / blocklen));
                    let a_d = [
                        f16_to_f32(a_blk.d[0]),
                        f16_to_f32(a_blk.d[1]),
                        f16_to_f32(a_blk.d[2]),
                        f16_to_f32(a_blk.d[3]),
                    ];
                    let bits0 = b.qs[k * ncols_interleaved + 0];
                    let bits1 = b.qs[k * ncols_interleaved + 1];
                    let bits2 = b.qs[k * ncols_interleaved + 2];
                    let bits3 = b.qs[k * ncols_interleaved + 3];

                    let q_offset = (k % (QK8_0 / blocklen)) * 4 * blocklen;

                    for p in 0..blocklen {
                        let q_row = [
                            a_blk.qs[q_offset + 0 * blocklen + p],
                            a_blk.qs[q_offset + 1 * blocklen + p],
                            a_blk.qs[q_offset + 2 * blocklen + p],
                            a_blk.qs[q_offset + 3 * blocklen + p],
                        ];
                        let sign = [
                            if (bits0 & (1u8 << p)) != 0 { 1i32 } else { -1i32 },
                            if (bits1 & (1u8 << p)) != 0 { 1i32 } else { -1i32 },
                            if (bits2 & (1u8 << p)) != 0 { 1i32 } else { -1i32 },
                            if (bits3 & (1u8 << p)) != 0 { 1i32 } else { -1i32 },
                        ];
                        for m in 0..4 {
                            let row_scale = a_d[m];
                            sumf[m][0] += sign[0] as f32 * q_row[m] as f32 * d0[0] * row_scale;
                            sumf[m][1] += sign[1] as f32 * q_row[m] as f32 * d0[1] * row_scale;
                            sumf[m][2] += sign[2] as f32 * q_row[m] as f32 * d0[2] * row_scale;
                            sumf[m][3] += sign[3] as f32 * q_row[m] as f32 * d0[3] * row_scale;
                        }
                    }
                }
            }

            for m in 0..4 {
                for j in 0..ncols_interleaved {
                    *s.add((y * 4 + m) * bs + x * ncols_interleaved + j) = sumf[m][j];
                }
            }
        }
    }
}

// -- RISC-V specific generic GEMMs ---------------------------------------------
// Lines 2683-3018: only compiled on riscv targets with zvfh extension.

/// Direct translation of ggml_gemm_q4_0_16x1_q8_0_generic (repack.cpp lines 2685-2737).
/// 16-column outer-product GEMM for Q4_0 x Q8_0. Only meaningful on RISC-V zvfh.
#[cfg(target_arch = "riscv64")]
pub unsafe fn ggml_gemm_q4_0_16x1_q8_0_generic(
    n:   i32,
    s:   *mut f32,
    bs:  usize,
    vx:  *const core::ffi::c_void,
    vy:  *const core::ffi::c_void,
    nr:  i32,
    nc:  i32,
) {
    let qk: usize = QK8_0;
    let nb: usize = n as usize / qk;
    let ncols_interleaved: usize = 16;
    let blocklen: usize = 1;

    debug_assert_eq!(n as usize % qk, 0);
    debug_assert_eq!(nr as usize % 4, 0);
    debug_assert_eq!(nc as usize % ncols_interleaved, 0);

    let a_base = vy as *const BlockQ8_0x4;
    let b_base = vx as *const BlockQ4_0x16;

    for y in 0..(nr as usize / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc as usize / ncols_interleaved) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0.0f32; 16]; 4];

            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);
                for k in 0..(qk / (2 * blocklen)) {
                    for m in 0..4usize {
                        for j in 0..ncols_interleaved {
                            let mut sumi = 0i32;
                            for i in 0..blocklen {
                                let v0 = ((b.qs[k * ncols_interleaved * blocklen + j * blocklen + i] as i8) << 4) as i32;
                                let v1 = (b.qs[k * ncols_interleaved * blocklen + j * blocklen + i] as i8 & -16i8) as i32;
                                sumi += ((v0 * a.qs[k * 4 * blocklen + m * blocklen + i] as i32)
                                    + (v1 * a.qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4] as i32))
                                    >> 4;
                            }
                            sumf[m][j] += sumi as f32
                                * f16_to_f32(b.d[j])
                                * f16_to_f32(a.d[m]);
                        }
                    }
                }
            }

            for m in 0..4usize {
                for j in 0..ncols_interleaved {
                    *s.add((y * 4 + m) * bs + x * ncols_interleaved + j) = sumf[m][j];
                }
            }
        }
    }
}

/// Direct translation of ggml_gemm_q4_K_16x1_q8_K_generic (repack.cpp lines 2739-2831).
#[cfg(target_arch = "riscv64")]
pub unsafe fn ggml_gemm_q4_k_16x1_q8_k_generic(
    n:   i32,
    s:   *mut f32,
    bs:  usize,
    vx:  *const core::ffi::c_void,
    vy:  *const core::ffi::c_void,
    nr:  i32,
    nc:  i32,
) {
    let nb: usize = n as usize / QK_K;
    let ncols_interleaved: usize = 16;

    debug_assert_eq!(n as usize % QK_K, 0);
    debug_assert_eq!(nr as usize % 4, 0);
    debug_assert_eq!(nc as usize % ncols_interleaved, 0);

    let a_base = vy as *const BlockQ8Kx4;
    let b_base = vx as *const BlockQ4Kx16;

    for y in 0..(nr as usize / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc as usize / ncols_interleaved) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0.0f32; 16]; 4];
            let mut sum_minf = [[0.0f32; 16]; 4];

            for l in 0..nb {
                let b = &*b_ptr.add(l);
                let a = &*a_ptr.add(l);

                // Unpack scales and mins from 192-byte packed format.
                let mut scales = [0u8; 128];
                let mut mins   = [0u8; 128];
                for i in 0..128usize {
                    scales[i] = b.scales[i] & 0x0F;
                    mins[i]   = b.scales[i] >> 4;
                }
                for i in 0..64usize {
                    scales[i]      |= (b.scales[128 + i] & 0x03) << 4;
                    mins[i]        |= (b.scales[128 + i] & 0x0C) << 2;
                    scales[i + 64] |= b.scales[128 + i] & 0x30;
                    mins[i + 64]   |= (b.scales[128 + i] & 0xC0) >> 2;
                }

                // Correction terms using bsums
                for sb in 0..8usize {
                    let min_base = sb * 16;
                    for m in 0..4usize {
                        let bsums = a.bsums[sb * 8 + m] as i32 + a.bsums[sb * 8 + m + 4] as i32;
                        for j in 0..ncols_interleaved {
                            sum_minf[m][j] += mins[min_base + j] as f32
                                * bsums as f32
                                * f16_to_f32(b.dmin[j])
                                * a.d[m];
                        }
                    }
                }

                // Main dot products
                for sb in (0..8usize).step_by(2) {
                    let scales_0_base = sb * 16;
                    let scales_1_base = (sb + 1) * 16;

                    for i in 0..32usize {
                        for m in 0..4usize {
                            for j in 0..ncols_interleaved {
                                let v0 = (b.qs[sb * 256 + i * 16 + j] & 0xF) as i32;
                                let v1 = (b.qs[sb * 256 + i * 16 + j] >> 4) as i32;
                                let sumi1 = v0 * a.qs[sb * 4 * 32 + i * 4 + m] as i32
                                    * scales[scales_0_base + j] as i32;
                                let sumi2 = v1 * a.qs[sb * 4 * 32 + 32 * 4 + i * 4 + m] as i32
                                    * scales[scales_1_base + j] as i32;
                                sumf[m][j] += (sumi1 + sumi2) as f32
                                    * f16_to_f32(b.d[j])
                                    * a.d[m];
                            }
                        }
                    }
                }
            }

            for m in 0..4usize {
                for j in 0..ncols_interleaved {
                    *s.add((y * 4 + m) * bs + x * ncols_interleaved + j) =
                        sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

/// Direct translation of ggml_gemm_iq4_nl_16x1_q8_0_generic (repack.cpp lines 2833-2875).
#[cfg(target_arch = "riscv64")]
pub unsafe fn ggml_gemm_iq4_nl_16x1_q8_0_generic(
    n:   i32,
    s:   *mut f32,
    bs:  usize,
    vx:  *const core::ffi::c_void,
    vy:  *const core::ffi::c_void,
    nr:  i32,
    nc:  i32,
) {
    let qk: usize = QK8_0;
    let nb: usize = n as usize / qk;
    let ncols_interleaved: usize = 16;
    let blocklen: usize = 1;

    debug_assert_eq!(n as usize % qk, 0);
    debug_assert_eq!(nr as usize % 4, 0);
    debug_assert_eq!(nc as usize % ncols_interleaved, 0);

    let a_base = vy as *const BlockQ8_0x4;
    let b_base = vx as *const BlockIQ4NLx16;

    for y in 0..(nr as usize / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc as usize / ncols_interleaved) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0.0f32; 16]; 4];

            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);
                for k in 0..(qk / (2 * blocklen)) {
                    for m in 0..4usize {
                        for j in 0..ncols_interleaved {
                            let mut sumi = 0i32;
                            for i in 0..blocklen {
                                let byte = b.qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                let v0 = KVALUES_IQ4NL[(byte & 0x0F) as usize] as i32;
                                let v1 = KVALUES_IQ4NL[(byte >> 4) as usize] as i32;
                                sumi += v0 * a.qs[k * 4 * blocklen + m * blocklen + i] as i32
                                    + v1 * a.qs[k * 4 * blocklen + m * blocklen + i + (qk / 2) * 4] as i32;
                            }
                            sumf[m][j] += sumi as f32
                                * f16_to_f32(b.d[j])
                                * f16_to_f32(a.d[m]);
                        }
                    }
                }
            }

            for m in 0..4usize {
                for j in 0..ncols_interleaved {
                    *s.add((y * 4 + m) * bs + x * ncols_interleaved + j) = sumf[m][j];
                }
            }
        }
    }
}

/// Direct translation of ggml_gemm_q8_0_16x1_q8_0_generic (repack.cpp lines 2877-2921).
#[cfg(target_arch = "riscv64")]
pub unsafe fn ggml_gemm_q8_0_16x1_q8_0_generic(
    n:   i32,
    s:   *mut f32,
    bs:  usize,
    vx:  *const core::ffi::c_void,
    vy:  *const core::ffi::c_void,
    nr:  i32,
    nc:  i32,
) {
    let qk: usize = QK8_0;
    let nb: usize = n as usize / qk;
    let ncols_interleaved: usize = 16;
    let blocklen: usize = 1;

    debug_assert_eq!(n as usize % qk, 0);
    debug_assert_eq!(nr as usize % 4, 0);
    debug_assert_eq!(nc as usize % ncols_interleaved, 0);

    let a_base = vy as *const BlockQ8_0x4;
    let b_base = vx as *const BlockQ8_0x16;

    for y in 0..(nr as usize / 4) {
        let a_ptr = a_base.add(y * nb);
        for x in 0..(nc as usize / ncols_interleaved) {
            let b_ptr = b_base.add(x * nb);
            let mut sumf = [[0.0f32; 16]; 4];

            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);
                for k in 0..(qk / blocklen) {
                    for m in 0..4usize {
                        for j in 0..ncols_interleaved {
                            let mut sumi = 0i32;
                            for i in 0..blocklen {
                                let v0 = b.qs[k * ncols_interleaved * blocklen + j * blocklen + i] as i32;
                                sumi += v0 * a.qs[k * 4 * blocklen + m * blocklen + i] as i32;
                            }
                            sumf[m][j] += sumi as f32
                                * f16_to_f32(b.d[j])
                                * f16_to_f32(a.d[m]);
                        }
                    }
                }
            }

            for m in 0..4usize {
                for j in 0..ncols_interleaved {
                    *s.add((y * 4 + m) * bs + x * ncols_interleaved + j) = sumf[m][j];
                }
            }
        }
    }
}

/// Direct translation of ggml_gemm_q2_K_16x1_q8_K_generic (repack.cpp lines 2924-3017).
#[cfg(target_arch = "riscv64")]
pub unsafe fn ggml_gemm_q2_k_16x1_q8_k_generic(
    n:   i32,
    s:   *mut f32,
    bs:  usize,
    vx:  *const core::ffi::c_void,
    vy:  *const core::ffi::c_void,
    nr:  i32,
    nc:  i32,
) {
    debug_assert_eq!(n as usize % QK_K, 0);
    debug_assert_eq!(nr as usize % 4, 0);
    debug_assert_eq!(nc as usize % 16, 0);

    let nb: usize = n as usize / QK_K;
    let x_base = vx as *const BlockQ2Kx16;
    let y_base = vy as *const BlockQ8Kx4;

    let sb_perm: [usize; 16] = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15];

    for row_tile in (0..nr as usize).step_by(4) {
        for col_tile in (0..nc as usize).step_by(16) {
            let x_ptr = x_base.add((col_tile / 16) * nb);
            let y_ptr = y_base.add((row_tile / 4) * nb);

            let mut sumf = [[0.0f32; 16]; 4];

            for k_block in 0..nb {
                let xb = &*x_ptr.add(k_block);
                let yb = &*y_ptr.add(k_block);

                let mut isum  = [[0i32; 16]; 4];
                let mut summs = [[0i32; 16]; 4];

                for sb in 0..16usize {
                    let scale_offset = sb_perm[sb] * 16;
                    let byte_base = if sb < 8 {
                        if sb % 2 == 0 { 0usize } else { 16usize }
                    } else {
                        if sb % 2 == 0 { 32usize } else { 48usize }
                    };
                    let shift = ((sb / 2) % 4) * 2;

                    for col in 0..16usize {
                        let sc_val = xb.scales[scale_offset + col];
                        let d_sb = (sc_val & 0xF) as i32;
                        let m_sb = (sc_val >> 4) as i32;

                        for r in 0..4usize {
                            let bsum_idx = (sb / 4) * 16 + r * 4 + (sb % 4);
                            summs[r][col] += yb.bsums[bsum_idx] as i32 * m_sb;
                        }

                        for l in 0..16usize {
                            let qs_idx = (byte_base + l) * 16 + col;
                            let q2_val = ((xb.qs[qs_idx] >> shift) & 3) as i32;
                            let k = sb * 16 + l;
                            let q8_idx = (k / 4) * 16 + (k % 4);
                            for r in 0..4usize {
                                let q8_val = yb.qs[q8_idx + r * 4] as i32;
                                isum[r][col] += q8_val * q2_val * d_sb;
                            }
                        }
                    }
                }

                for col in 0..16usize {
                    let d_rhs  = f16_to_f32(xb.d[col]);
                    let dm_rhs = f16_to_f32(xb.dmin[col]);
                    for r in 0..4usize {
                        let d_lhs = yb.d[r];
                        sumf[r][col] += isum[r][col] as f32 * d_lhs * d_rhs
                            - summs[r][col] as f32 * d_lhs * dm_rhs;
                    }
                }
            }

            for r in 0..4usize {
                for col in 0..16usize {
                    *s.add((row_tile + r) * bs + (col_tile + col)) = sumf[r][col];
                }
            }
        }
    }
}

// -- make_block helpers --------------------------------------------------------

/// Direct translation of make_block_q1_0x4 (repack.cpp lines 3039-3081).
///
/// Interleaves 4 Q1_0 blocks. blck_size_interleave must be 4 or 8.
pub fn make_block_q1_0x4(in_blocks: &[BlockQ1_0; 4], blck_size_interleave: usize) -> BlockQ1_0x4 {
    assert!(
        blck_size_interleave == 4 || blck_size_interleave == 8,
        "blck_size_interleave must be 4 or 8"
    );

    let mut out = BlockQ1_0x4 {
        d:  [0u16; 4],
        qs: [0u8; 64],
    };
    for i in 0..4 {
        out.d[i] = in_blocks[i].d;
    }

    if blck_size_interleave == 4 {
        for k in 0..(QK1_0 / QK8_0) {
            for tile in 0..(QK8_0 / 4) {
                let mut packed_lo: u8 = 0;
                let mut packed_hi: u8 = 0;
                let weight_base = k * QK8_0 + tile * 4;
                for pos in 0..4usize {
                    let weight_idx = weight_base + pos;
                    let byte_idx  = weight_idx / 8;
                    let bit_idx   = weight_idx % 8;
                    packed_lo |= ((in_blocks[0].qs[byte_idx] >> bit_idx) & 1u8) << pos;
                    packed_lo |= ((in_blocks[1].qs[byte_idx] >> bit_idx) & 1u8) << (4 + pos);
                    packed_hi |= ((in_blocks[2].qs[byte_idx] >> bit_idx) & 1u8) << pos;
                    packed_hi |= ((in_blocks[3].qs[byte_idx] >> bit_idx) & 1u8) << (4 + pos);
                }
                out.qs[k * 16 + 2 * tile + 0] = packed_lo;
                out.qs[k * 16 + 2 * tile + 1] = packed_hi;
            }
        }
    } else {
        // blck_size_interleave == 8: simple byte interleave
        for byte_idx in 0..(QK1_0 / 8) {
            out.qs[byte_idx * 4 + 0] = in_blocks[0].qs[byte_idx];
            out.qs[byte_idx * 4 + 1] = in_blocks[1].qs[byte_idx];
            out.qs[byte_idx * 4 + 2] = in_blocks[2].qs[byte_idx];
            out.qs[byte_idx * 4 + 3] = in_blocks[3].qs[byte_idx];
        }
    }
    out
}

/// Direct translation of make_block_q4_0x4 (repack.cpp lines 3083-3122).
///
/// Interleaves 4 Q4_0 blocks. blck_size_interleave must be 4 or 8.
pub fn make_block_q4_0x4(in_blocks: &[BlockQ4_0; 4], blck_size_interleave: usize) -> BlockQ4_0x4 {
    let mut out = BlockQ4_0x4 {
        d:  [0u16; 4],
        qs: [0u8; 64],
    };
    for i in 0..4 {
        out.d[i] = in_blocks[i].d;
    }

    let end = QK4_0 * 2 / blck_size_interleave;

    if blck_size_interleave == 8 {
        let xor_mask: u64 = 0x8888888888888888;
        for i in 0..end {
            let src_id     = i % 4;
            let src_offset = (i / 4) * blck_size_interleave;
            let dst_offset = i * blck_size_interleave;
            let mut elems: u64 = 0;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    in_blocks[src_id].qs.as_ptr().add(src_offset),
                    (&mut elems) as *mut u64 as *mut u8,
                    8,
                );
            }
            elems ^= xor_mask;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&elems) as *const u64 as *const u8,
                    out.qs.as_mut_ptr().add(dst_offset),
                    8,
                );
            }
        }
    } else if blck_size_interleave == 4 {
        let xor_mask: u32 = 0x88888888;
        for i in 0..end {
            let src_id     = i % 4;
            let src_offset = (i / 4) * blck_size_interleave;
            let dst_offset = i * blck_size_interleave;
            let mut elems: u32 = 0;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    in_blocks[src_id].qs.as_ptr().add(src_offset),
                    (&mut elems) as *mut u32 as *mut u8,
                    4,
                );
            }
            elems ^= xor_mask;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&elems) as *const u32 as *const u8,
                    out.qs.as_mut_ptr().add(dst_offset),
                    4,
                );
            }
        }
    } else {
        panic!("blck_size_interleave must be 4 or 8");
    }
    out
}

/// Direct translation of make_block_q4_0x16 (repack.cpp lines 3152-3175).
///
/// Interleaves 16 Q4_0 blocks (RISC-V outer-product layout). blck_size_interleave must be 1.
pub fn make_block_q4_0x16(
    in_blocks: &[BlockQ4_0; 16],
    blck_size_interleave: usize,
) -> BlockQ4_0x16 {
    assert_eq!(blck_size_interleave, 1, "make_block_q4_0x16 requires blck_size_interleave==1");

    let mut out = BlockQ4_0x16 {
        d:  [0u16; 16],
        qs: [0u8; 256],
    };
    for i in 0..16 {
        out.d[i] = in_blocks[i].d;
    }

    let xor_mask: u8 = 0x88;
    let end = QK4_0 * 8; // QK4_0*8/1 = 256
    for i in 0..end {
        let src_id     = i % 16;
        let src_offset = i / 16;
        out.qs[i] = in_blocks[src_id].qs[src_offset] ^ xor_mask;
    }
    out
}

/// Direct translation of make_block_q4_Kx16 (repack.cpp lines 3254-3304).
///
/// Interleaves 16 Q4K blocks (RISC-V outer-product layout). blck_size_interleave must be 1.
pub fn make_block_q4_kx16(in_blocks: &[BlockQ4K; 16], blck_size_interleave: usize) -> BlockQ4Kx16 {
    assert_eq!(blck_size_interleave, 1, "make_block_q4_kx16 requires blck_size_interleave==1");

    let mut out = BlockQ4Kx16 {
        d:      [0u16; 16],
        dmin:   [0u16; 16],
        scales: [0u8; 192],
        qs:     [0u8; 2048],
    };
    for i in 0..16 {
        out.d[i]    = in_blocks[i].d;
        out.dmin[i] = in_blocks[i].dmin;
    }

    // Interleave qs: 1 byte at a time, 16-way interleave
    let end = QK_K * 8; // QK_K/2 * 16 / 1
    for i in 0..end {
        let src_id     = i % 16;
        let src_offset = i / 16;
        out.qs[i] = in_blocks[src_id].qs[src_offset];
    }

    // Unpack and repack scales/mins from 6-bit packed format.
    // s[128], m[128]: first 64 from low bits, next 64 from high+extra bits.
    let mut s = [0u8; 128];
    let mut m = [0u8; 128];
    for i in 0..4usize {
        for j in 0..16usize {
            s[i * 16 + j] = in_blocks[j].scales[i] & 63;
            m[i * 16 + j] = in_blocks[j].scales[i + 4] & 63;
        }
    }
    for i in 0..4usize {
        for j in 0..16usize {
            s[64 + i * 16 + j] =
                ((in_blocks[j].scales[i] & 192) >> 2) | (in_blocks[j].scales[i + 8] & 15);
            m[64 + i * 16 + j] =
                ((in_blocks[j].scales[i + 4] & 192) >> 2)
                | ((in_blocks[j].scales[i + 8] & 240) >> 4);
        }
    }

    // Pack into out.scales[0..128]: lower nibble of s, lower nibble of m in upper nibble
    for i in 0..128usize {
        out.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
    }
    // Pack high bits into out.scales[128..192]
    for i in 0..64usize {
        out.scales[128 + i] =
            ((s[i] & 48) >> 4)
            | ((m[i] & 48) >> 2)
            | (s[64 + i] & 48)
            | ((m[64 + i] & 48) << 2);
    }
    out
}

/// Direct translation of make_block_q2_Kx16 (repack.cpp lines 3479-3539).
///
/// Interleaves 16 Q2K blocks (RISC-V outer-product layout).
pub fn make_block_q2_kx16(in_blocks: &[BlockQ2K; 16], blck_size_interleave: usize) -> BlockQ2Kx16 {
    let n_cols: usize = 16;

    let mut out = BlockQ2Kx16 {
        d:      [0u16; 16],
        dmin:   [0u16; 16],
        scales: [0u8; 256],
        qs:     [0u8; 1024],
    };
    for i in 0..n_cols {
        out.d[i]    = in_blocks[i].d;
        out.dmin[i] = in_blocks[i].dmin;
    }

    // Interleave qs
    let bytes_per_col = 64usize;
    let total_bytes   = n_cols * bytes_per_col;
    let end           = total_bytes / blck_size_interleave;
    for i in 0..end {
        let src_col_id = i % n_cols;
        let src_offset = (i / n_cols) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        unsafe {
            std::ptr::copy_nonoverlapping(
                in_blocks[src_col_id].qs.as_ptr().add(src_offset),
                out.qs.as_mut_ptr().add(dst_offset),
                blck_size_interleave,
            );
        }
    }

    // Repack scales in "sequential-parallel" layout.
    let mut out_idx: usize = 0;
    let groups: [[usize; 4]; 4] = [
        [0, 2, 4, 6],    // even_low
        [1, 3, 5, 7],    // odd_low
        [8, 10, 12, 14], // even_high
        [9, 11, 13, 15], // odd_high
    ];
    for group in &groups {
        for &sb in group {
            for col in 0..n_cols {
                out.scales[out_idx] = in_blocks[col].scales[sb];
                out_idx += 1;
            }
        }
    }
    out
}

/// Direct translation of make_block_q8_0x16 (repack.cpp lines 3885-3906).
///
/// Interleaves 16 Q8_0 blocks (RISC-V outer-product layout). blck_size_interleave must be 1.
pub fn make_block_q8_0x16(
    in_blocks: &[BlockQ8_0; 16],
    blck_size_interleave: usize,
) -> BlockQ8_0x16 {
    assert_eq!(blck_size_interleave, 1, "make_block_q8_0x16 requires blck_size_interleave==1");

    let mut out = BlockQ8_0x16 {
        d:  [0u16; 16],
        qs: [0i8; 512],
    };
    for i in 0..16 {
        out.d[i] = in_blocks[i].d;
    }

    let end = QK8_0 * 16;
    for i in 0..end {
        let src_id     = i % 16;
        let src_offset = i / 16;
        out.qs[i] = in_blocks[src_id].qs[src_offset];
    }
    out
}

/// Direct translation of make_block_iq4_nlx4 (repack.cpp lines 3939-3972).
///
/// Interleaves 4 IQ4_NL blocks. blck_size_interleave must be 4.
pub fn make_block_iq4_nlx4(in_blocks: &[BlockIQ4NL; 4], blck_size_interleave: usize) -> BlockIQ4NLx4 {
    assert_eq!(blck_size_interleave, 4, "make_block_iq4_nlx4 requires blck_size_interleave==4");

    let mut out = BlockIQ4NLx4 {
        d:  [0u16; 4],
        qs: [0u8; 64],
    };
    for i in 0..4 {
        out.d[i] = in_blocks[i].d;
    }

    let end = QK4_NL * 2 / blck_size_interleave;
    for i in 0..end {
        let src_id     = i % 4;
        let src_offset = (i / 4) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        unsafe {
            std::ptr::copy_nonoverlapping(
                in_blocks[src_id].qs.as_ptr().add(src_offset),
                out.qs.as_mut_ptr().add(dst_offset),
                4,
            );
        }
    }
    out
}

/// Direct translation of make_block_iq4_nlx8 (repack.cpp lines 4007-4029).
///
/// Interleaves 8 IQ4_NL blocks. blck_size_interleave must be 8.
pub fn make_block_iq4_nlx8(in_blocks: &[BlockIQ4NL; 8], blck_size_interleave: usize) -> BlockIQ4NLx8 {
    assert_eq!(blck_size_interleave, 8, "make_block_iq4_nlx8 requires blck_size_interleave==8");

    let mut out = BlockIQ4NLx8 {
        d:  [0u16; 8],
        qs: [0u8; 128],
    };
    for i in 0..8 {
        out.d[i] = in_blocks[i].d;
    }

    let end = QK4_NL * 4 / blck_size_interleave;
    for i in 0..end {
        let src_id     = i % 8;
        let src_offset = (i / 8) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        unsafe {
            std::ptr::copy_nonoverlapping(
                in_blocks[src_id].qs.as_ptr().add(src_offset),
                out.qs.as_mut_ptr().add(dst_offset),
                8,
            );
        }
    }
    out
}

/// Direct translation of make_block_iq4_nlx16 (repack.cpp lines 4064-4086).
///
/// Interleaves 16 IQ4_NL blocks (RISC-V). blck_size_interleave must be 1.
pub fn make_block_iq4_nlx16(
    in_blocks: &[BlockIQ4NL; 16],
    blck_size_interleave: usize,
) -> BlockIQ4NLx16 {
    assert_eq!(blck_size_interleave, 1, "make_block_iq4_nlx16 requires blck_size_interleave==1");

    let mut out = BlockIQ4NLx16 {
        d:  [0u16; 16],
        qs: [0u8; 256],
    };
    for i in 0..16 {
        out.d[i] = in_blocks[i].d;
    }

    let end = QK4_NL * 8;
    for i in 0..end {
        let src_id     = i % 16;
        let src_offset = i / 16;
        out.qs[i] = in_blocks[src_id].qs[src_offset];
    }
    out
}

/// Direct translation of make_block_mxfp4x4 (repack.cpp lines 4121-4143).
///
/// Interleaves 4 MXFP4 blocks. blck_size_interleave must be 4.
pub fn make_block_mxfp4x4(in_blocks: &[BlockMxfp4; 4], blck_size_interleave: usize) -> BlockMxfp4x4 {
    assert_eq!(blck_size_interleave, 4, "make_block_mxfp4x4 requires blck_size_interleave==4");

    let mut out = BlockMxfp4x4 {
        e:  [0u8; 4],
        qs: [0u8; 64],
    };
    for i in 0..4 {
        out.e[i] = in_blocks[i].e;
    }

    let end = QK_MXFP4 * 2 / blck_size_interleave;
    for i in 0..end {
        let src_id     = i % 4;
        let src_offset = (i / 4) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        unsafe {
            std::ptr::copy_nonoverlapping(
                in_blocks[src_id].qs.as_ptr().add(src_offset),
                out.qs.as_mut_ptr().add(dst_offset),
                4,
            );
        }
    }
    out
}

/// Direct translation of make_block_mxfp4x8 (repack.cpp lines 4178-4200).
///
/// Interleaves 8 MXFP4 blocks. blck_size_interleave must be 8.
pub fn make_block_mxfp4x8(in_blocks: &[BlockMxfp4; 8], blck_size_interleave: usize) -> BlockMxfp4x8 {
    assert_eq!(blck_size_interleave, 8, "make_block_mxfp4x8 requires blck_size_interleave==8");

    let mut out = BlockMxfp4x8 {
        e:  [0u8; 8],
        qs: [0u8; 128],
    };
    for i in 0..8 {
        out.e[i] = in_blocks[i].e;
    }

    let end = QK_MXFP4 * 4 / blck_size_interleave;
    for i in 0..end {
        let src_id     = i % 8;
        let src_offset = (i / 8) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        unsafe {
            std::ptr::copy_nonoverlapping(
                in_blocks[src_id].qs.as_ptr().add(src_offset),
                out.qs.as_mut_ptr().add(dst_offset),
                8,
            );
        }
    }
    out
}

// -- Bulk repack functions -----------------------------------------------------

/// Tensor shape passed to repack functions (mirrors ggml_tensor fields used here).
pub struct TensorShape {
    /// ne[0]: columns (weight dimension)
    pub ne0: i64,
    /// ne[1]: rows (n_rows in the weight matrix)
    pub ne1: i64,
}

/// Direct translation of repack_q4_0_to_q4_0_4_bl (repack.cpp lines 3541-3570).
///
/// Repacks nrow x nblocks Q4_0 blocks into 4-row-interleaved BlockQ4_0x4 layout.
/// Returns 0 on success, -1 if alignment constraints are not met.
pub fn repack_q4_0_to_q4_0_4_bl(
    shape: &TensorShape,
    interleave_block: usize,
    src: &[BlockQ4_0],
    dst: &mut [BlockQ4_0x4],
) -> i32 {
    debug_assert!(interleave_block == 4 || interleave_block == 8);
    const NROWS: usize = 4;

    let nrow    = shape.ne1 as usize;
    let nblocks = shape.ne0 as usize / QK4_0;

    if nrow % NROWS != 0 || shape.ne0 % 8 != 0 {
        return -1;
    }

    let mut dst_idx = 0usize;
    let mut src_off = 0usize;
    for _b in (0..nrow).step_by(NROWS) {
        for x in 0..nblocks {
            let tmp = [
                src[x + 0 * nblocks + src_off],
                src[x + 1 * nblocks + src_off],
                src[x + 2 * nblocks + src_off],
                src[x + 3 * nblocks + src_off],
            ];
            dst[dst_idx] = make_block_q4_0x4(&tmp, interleave_block);
            dst_idx += 1;
        }
        src_off += NROWS * nblocks;
    }
    0
}

/// Direct translation of repack_q4_K_to_q4_K_16_bl (repack.cpp lines 3603-3631).
///
/// Repacks Q4K blocks into 16-row-interleaved BlockQ4Kx16 (RISC-V layout).
pub fn repack_q4_k_to_q4_k_16_bl(
    shape: &TensorShape,
    interleave_block: usize,
    src: &[BlockQ4K],
    dst: &mut [BlockQ4Kx16],
) -> i32 {
    const NROWS: usize = 16;

    let nrow    = shape.ne1 as usize;
    let nblocks = shape.ne0 as usize / QK_K;

    if nrow % NROWS != 0 || shape.ne0 % 8 != 0 {
        return -1;
    }

    let mut dst_idx = 0usize;
    let mut src_off = 0usize;
    for _b in (0..nrow).step_by(NROWS) {
        for x in 0..nblocks {
            let mut tmp = [BlockQ4K::default(); 16];
            for i in 0..NROWS {
                tmp[i] = src[x + i * nblocks + src_off];
            }
            dst[dst_idx] = make_block_q4_kx16(&tmp, interleave_block);
            dst_idx += 1;
        }
        src_off += NROWS * nblocks;
    }
    0
}

/// Direct translation of repack_q2_K_to_q2_K_16_bl (repack.cpp lines 3664-3697).
///
/// Repacks Q2K blocks into 16-row-interleaved BlockQ2Kx16 (RISC-V layout).
pub fn repack_q2_k_to_q2_k_16_bl(
    shape: &TensorShape,
    interleave_block: usize,
    src: &[BlockQ2K],
    dst: &mut [BlockQ2Kx16],
) -> i32 {
    const NROWS: usize = 16;

    let nrow    = shape.ne1 as usize;
    let nblocks = shape.ne0 as usize / QK_K;

    if nrow % NROWS != 0 || shape.ne0 % 8 != 0 {
        return -1;
    }

    let mut dst_idx = 0usize;
    let mut src_off = 0usize;
    for _b in (0..nrow).step_by(NROWS) {
        for x in 0..nblocks {
            let mut tmp = [BlockQ2K::default(); 16];
            for i in 0..NROWS {
                tmp[i] = src[x + i * nblocks + src_off];
            }
            dst[dst_idx] = make_block_q2_kx16(&tmp, interleave_block);
            dst_idx += 1;
        }
        src_off += NROWS * nblocks;
    }
    0
}

/// Direct translation of repack_q4_0_to_q4_0_16_bl (repack.cpp lines 3699-3727).
///
/// Repacks Q4_0 blocks into 16-row-interleaved BlockQ4_0x16 (RISC-V layout).
pub fn repack_q4_0_to_q4_0_16_bl(
    shape: &TensorShape,
    interleave_block: usize,
    src: &[BlockQ4_0],
    dst: &mut [BlockQ4_0x16],
) -> i32 {
    const NROWS: usize = 16;

    let nrow    = shape.ne1 as usize;
    let nblocks = shape.ne0 as usize / QK4_0;

    if nrow % NROWS != 0 || shape.ne0 % 8 != 0 {
        return -1;
    }

    let mut dst_idx = 0usize;
    let mut src_off = 0usize;
    for _b in (0..nrow).step_by(NROWS) {
        for x in 0..nblocks {
            let mut tmp = [BlockQ4_0::default(); 16];
            for i in 0..NROWS {
                tmp[i] = src[x + i * nblocks + src_off];
            }
            dst[dst_idx] = make_block_q4_0x16(&tmp, interleave_block);
            dst_idx += 1;
        }
        src_off += NROWS * nblocks;
    }
    0
}

/// Direct translation of repack_q8_0_to_q8_0_4_bl (repack.cpp lines 3821-3851).
///
/// Repacks Q8_0 blocks into 4-row-interleaved BlockQ8_0x4 layout.
pub fn repack_q8_0_to_q8_0_4_bl(
    shape: &TensorShape,
    interleave_block: usize,
    src: &[BlockQ8_0],
    dst: &mut [BlockQ8_0x4],
) -> i32 {
    debug_assert!(interleave_block == 4 || interleave_block == 8);
    const NROWS: usize = 4;

    let nrow    = shape.ne1 as usize;
    let nblocks = shape.ne0 as usize / QK8_0;

    if nrow % NROWS != 0 || shape.ne0 % 8 != 0 {
        return -1;
    }

    let mut dst_idx = 0usize;
    let mut src_off = 0usize;
    for _b in (0..nrow).step_by(NROWS) {
        for x in 0..nblocks {
            let tmp = [
                src[x + 0 * nblocks + src_off],
                src[x + 1 * nblocks + src_off],
                src[x + 2 * nblocks + src_off],
                src[x + 3 * nblocks + src_off],
            ];
            dst[dst_idx] = make_block_q8_0x4(&tmp, interleave_block);
            dst_idx += 1;
        }
        src_off += NROWS * nblocks;
    }
    0
}

/// Direct translation of repack_q1_0_to_q1_0_4_bl (repack.cpp lines 3853-3883).
///
/// Repacks Q1_0 blocks into 4-row-interleaved BlockQ1_0x4 layout.
pub fn repack_q1_0_to_q1_0_4_bl(
    shape: &TensorShape,
    interleave_block: usize,
    src: &[BlockQ1_0],
    dst: &mut [BlockQ1_0x4],
) -> i32 {
    debug_assert!(interleave_block == 4 || interleave_block == 8);
    const NROWS: usize = 4;

    let nrow    = shape.ne1 as usize;
    let nblocks = shape.ne0 as usize / QK1_0;

    if nrow % NROWS != 0 {
        return -1;
    }

    let mut dst_idx = 0usize;
    let mut src_off = 0usize;
    for _b in (0..nrow).step_by(NROWS) {
        for x in 0..nblocks {
            let tmp = [
                src[x + 0 * nblocks + src_off],
                src[x + 1 * nblocks + src_off],
                src[x + 2 * nblocks + src_off],
                src[x + 3 * nblocks + src_off],
            ];
            dst[dst_idx] = make_block_q1_0x4(&tmp, interleave_block);
            dst_idx += 1;
        }
        src_off += NROWS * nblocks;
    }
    0
}

/// Direct translation of repack_q8_0_to_q8_0_16_bl (repack.cpp lines 3908-3937).
///
/// Repacks Q8_0 blocks into 16-row-interleaved BlockQ8_0x16 (RISC-V layout).
pub fn repack_q8_0_to_q8_0_16_bl(
    shape: &TensorShape,
    interleave_block: usize,
    src: &[BlockQ8_0],
    dst: &mut [BlockQ8_0x16],
) -> i32 {
    const NROWS: usize = 16;

    let nrow    = shape.ne1 as usize;
    let nblocks = shape.ne0 as usize / QK8_0;

    if nrow % NROWS != 0 || shape.ne0 % 8 != 0 {
        return -1;
    }

    let mut dst_idx = 0usize;
    let mut src_off = 0usize;
    for _b in (0..nrow).step_by(NROWS) {
        for x in 0..nblocks {
            let mut tmp = [BlockQ8_0::default(); 16];
            for i in 0..NROWS {
                tmp[i] = src[x + i * nblocks + src_off];
            }
            dst[dst_idx] = make_block_q8_0x16(&tmp, interleave_block);
            dst_idx += 1;
        }
        src_off += NROWS * nblocks;
    }
    0
}

/// Direct translation of repack_iq4_nl_to_iq4_nl_4_bl (repack.cpp lines 3974-4005).
///
/// Repacks IQ4_NL blocks into 4-row-interleaved BlockIQ4NLx4 layout.
pub fn repack_iq4_nl_to_iq4_nl_4_bl(
    shape: &TensorShape,
    interleave_block: usize,
    src: &[BlockIQ4NL],
    dst: &mut [BlockIQ4NLx4],
) -> i32 {
    debug_assert_eq!(interleave_block, 4);
    const NROWS: usize = 4;

    let nrow    = shape.ne1 as usize;
    let nblocks = shape.ne0 as usize / QK4_NL;

    if nrow % NROWS != 0 || shape.ne0 % 8 != 0 {
        return -1;
    }

    let mut dst_idx = 0usize;
    let mut src_off = 0usize;
    for _b in (0..nrow).step_by(NROWS) {
        for x in 0..nblocks {
            let tmp = [
                src[x + 0 * nblocks + src_off],
                src[x + 1 * nblocks + src_off],
                src[x + 2 * nblocks + src_off],
                src[x + 3 * nblocks + src_off],
            ];
            dst[dst_idx] = make_block_iq4_nlx4(&tmp, interleave_block);
            dst_idx += 1;
        }
        src_off += NROWS * nblocks;
    }
    0
}

/// Direct translation of repack_iq4_nl_to_iq4_nl_8_bl (repack.cpp lines 4031-4062).
///
/// Repacks IQ4_NL blocks into 8-row-interleaved BlockIQ4NLx8 layout.
pub fn repack_iq4_nl_to_iq4_nl_8_bl(
    shape: &TensorShape,
    interleave_block: usize,
    src: &[BlockIQ4NL],
    dst: &mut [BlockIQ4NLx8],
) -> i32 {
    debug_assert_eq!(interleave_block, 8);
    const NROWS: usize = 8;

    let nrow    = shape.ne1 as usize;
    let nblocks = shape.ne0 as usize / QK4_NL;

    if nrow % NROWS != 0 {
        return -1;
    }

    let mut dst_idx = 0usize;
    let mut src_off = 0usize;
    for _b in (0..nrow).step_by(NROWS) {
        for x in 0..nblocks {
            let mut tmp = [BlockIQ4NL::default(); 8];
            for i in 0..NROWS {
                tmp[i] = src[x + i * nblocks + src_off];
            }
            dst[dst_idx] = make_block_iq4_nlx8(&tmp, interleave_block);
            dst_idx += 1;
        }
        src_off += NROWS * nblocks;
    }
    0
}

/// Direct translation of repack_iq4_nl_to_iq4_nl_16_bl (repack.cpp lines 4088-4119).
///
/// Repacks IQ4_NL blocks into 16-row-interleaved BlockIQ4NLx16 (RISC-V layout).
pub fn repack_iq4_nl_to_iq4_nl_16_bl(
    shape: &TensorShape,
    interleave_block: usize,
    src: &[BlockIQ4NL],
    dst: &mut [BlockIQ4NLx16],
) -> i32 {
    debug_assert_eq!(interleave_block, 1);
    const NROWS: usize = 16;

    let nrow    = shape.ne1 as usize;
    let nblocks = shape.ne0 as usize / QK4_NL;

    if nrow % NROWS != 0 {
        return -1;
    }

    let mut dst_idx = 0usize;
    let mut src_off = 0usize;
    for _b in (0..nrow).step_by(NROWS) {
        for x in 0..nblocks {
            let mut tmp = [BlockIQ4NL::default(); 16];
            for i in 0..NROWS {
                tmp[i] = src[x + i * nblocks + src_off];
            }
            dst[dst_idx] = make_block_iq4_nlx16(&tmp, interleave_block);
            dst_idx += 1;
        }
        src_off += NROWS * nblocks;
    }
    0
}

/// Direct translation of repack_mxfp4_to_mxfp4_4_bl (repack.cpp lines 4145-4176).
///
/// Repacks MXFP4 blocks into 4-row-interleaved BlockMxfp4x4 layout.
pub fn repack_mxfp4_to_mxfp4_4_bl(
    shape: &TensorShape,
    interleave_block: usize,
    src: &[BlockMxfp4],
    dst: &mut [BlockMxfp4x4],
) -> i32 {
    debug_assert_eq!(interleave_block, 4);
    const NROWS: usize = 4;

    let nrow    = shape.ne1 as usize;
    let nblocks = shape.ne0 as usize / QK_MXFP4;

    if nrow % NROWS != 0 || shape.ne0 % 8 != 0 {
        return -1;
    }

    let mut dst_idx = 0usize;
    let mut src_off = 0usize;
    for _b in (0..nrow).step_by(NROWS) {
        for x in 0..nblocks {
            let tmp = [
                src[x + 0 * nblocks + src_off],
                src[x + 1 * nblocks + src_off],
                src[x + 2 * nblocks + src_off],
                src[x + 3 * nblocks + src_off],
            ];
            dst[dst_idx] = make_block_mxfp4x4(&tmp, interleave_block);
            dst_idx += 1;
        }
        src_off += NROWS * nblocks;
    }
    0
}

/// Direct translation of repack_mxfp4_to_mxfp4_8_bl (repack.cpp lines 4202-4233).
///
/// Repacks MXFP4 blocks into 8-row-interleaved BlockMxfp4x8 layout.
pub fn repack_mxfp4_to_mxfp4_8_bl(
    shape: &TensorShape,
    interleave_block: usize,
    src: &[BlockMxfp4],
    dst: &mut [BlockMxfp4x8],
) -> i32 {
    debug_assert_eq!(interleave_block, 8);
    const NROWS: usize = 8;

    let nrow    = shape.ne1 as usize;
    let nblocks = shape.ne0 as usize / QK_MXFP4;

    if nrow % NROWS != 0 {
        return -1;
    }

    let mut dst_idx = 0usize;
    let mut src_off = 0usize;
    for _b in (0..nrow).step_by(NROWS) {
        for x in 0..nblocks {
            let mut tmp = [BlockMxfp4::default(); 8];
            for i in 0..NROWS {
                tmp[i] = src[x + i * nblocks + src_off];
            }
            dst[dst_idx] = make_block_mxfp4x8(&tmp, interleave_block);
            dst_idx += 1;
        }
        src_off += NROWS * nblocks;
    }
    0
}

// -- Repack type selector ------------------------------------------------------

/// Which interleaved layout and kernel should be used for a weight tensor.
///
/// Direct translation of ggml_repack_get_optimal_repack_type (repack.cpp lines 4925-5141).
/// The returned value identifies the (BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE) specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepackKind {
    /// Q4_0 weights, 8-byte interleave, 4 cols -- ARM dotprod
    Q4_0_4x4_Q8_0,
    /// Q4_0 weights, 8-byte interleave, 4 rows, 8 interleave -- ARM i8mm
    Q4_0_4x8_Q8_0,
    /// Q4_0 weights, 8-byte interleave, 8 cols -- AVX2 / SVE
    Q4_0_8x8_Q8_0,
    /// Q4_0 weights, 1-byte interleave, 16 cols -- RISC-V zvfh
    Q4_0_16x1_Q8_0,

    /// Q4K weights, 4-byte interleave, 8 cols -- ARM dotprod
    Q4K_8x4_Q8K,
    /// Q4K weights, 8-byte interleave, 8 cols -- AVX2 / ARM i8mm
    Q4K_8x8_Q8K,
    /// Q4K weights, 1-byte interleave, 16 cols -- RISC-V zvfh
    Q4K_16x1_Q8K,

    /// Q5K weights, 4-byte interleave, 8 cols -- ARM dotprod
    Q5K_8x4_Q8K,
    /// Q5K weights, 8-byte interleave, 8 cols -- ARM i8mm
    Q5K_8x8_Q8K,

    /// Q6K weights, 4-byte interleave, 8 cols -- ARM dotprod
    Q6K_8x4_Q8K,
    /// Q6K weights, 8-byte interleave, 8 cols -- ARM i8mm
    Q6K_8x8_Q8K,

    /// Q2K weights, 8-byte interleave, 8 cols -- AVX-512
    Q2K_8x8_Q8K,
    /// Q2K weights, 1-byte interleave, 16 cols -- RISC-V zvfh
    Q2K_16x1_Q8K,

    /// IQ4_NL weights, 4-byte interleave, 4 cols -- ARM dotprod
    IQ4NL_4x4_Q8_0,
    /// IQ4_NL weights, 8-byte interleave, 8 cols -- AVX2
    IQ4NL_8x8_Q8_0,
    /// IQ4_NL weights, 1-byte interleave, 16 cols -- RISC-V zvfh
    IQ4NL_16x1_Q8_0,

    /// MXFP4 weights, 4-byte interleave, 4 cols -- ARM dotprod
    MXFP4_4x4_Q8_0,
    /// MXFP4 weights, 8-byte interleave, 8 cols -- AVX2
    MXFP4_8x8_Q8_0,

    /// Q8_0 weights, 4-byte interleave, 4 cols -- ARM dotprod
    Q8_0_4x4_Q8_0,
    /// Q8_0 weights, 8-byte interleave, 4 rows -- ARM i8mm
    Q8_0_4x8_Q8_0,
    /// Q8_0 weights, 1-byte interleave, 16 cols -- RISC-V zvfh
    Q8_0_16x1_Q8_0,

    /// Q1_0 weights, 4-byte interleave, 4 cols -- ARM dotprod
    Q1_0_4x4_Q8_0,
    /// Q1_0 weights, 8-byte interleave, 4 cols -- ARM i8mm
    Q1_0_4x8_Q8_0,
}

/// Quant type tag (mirrors ggml_type for the weight tensor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    Q1_0,
    Q4_0,
    Q4K,
    Q5K,
    Q6K,
    Q2K,
    IQ4NL,
    Mxfp4,
    Q8_0,
}

/// Direct translation of ggml_repack_get_optimal_repack_type (repack.cpp lines 4925-5141).
///
/// Given a weight tensor's quant type and dimensions, returns the best
/// `RepackKind` for the current CPU, or `None` if no interleaved kernel applies.
pub fn get_optimal_repack_type(quant: QuantType, _ne0: i64, ne1: i64) -> Option<RepackKind> {
    #[allow(unused_variables)]
    let has_avx2: bool = {
        #[cfg(target_arch = "x86_64")]
        { is_x86_feature_detected!("avx2") }
        #[cfg(not(target_arch = "x86_64"))]
        { false }
    };
    #[allow(unused_variables)]
    let has_avx512: bool = {
        #[cfg(target_arch = "x86_64")]
        { is_x86_feature_detected!("avx512f") }
        #[cfg(not(target_arch = "x86_64"))]
        { false }
    };
    // NEON + dotprod / matmul_int8 -- ARM only; always false on x86
    let has_neon_dotprod: bool = false;
    let has_neon_i8mm: bool    = false;

    match quant {
        QuantType::Q4_0 => {
            if has_avx2 && ne1 % 8 == 0 {
                return Some(RepackKind::Q4_0_8x8_Q8_0);
            }
            if has_neon_i8mm && ne1 % 4 == 0 {
                return Some(RepackKind::Q4_0_4x8_Q8_0);
            }
            if has_neon_dotprod && ne1 % 4 == 0 {
                return Some(RepackKind::Q4_0_4x4_Q8_0);
            }
            None
        }
        QuantType::Q4K => {
            if has_avx2 && ne1 % 8 == 0 {
                return Some(RepackKind::Q4K_8x8_Q8K);
            }
            if has_neon_i8mm && ne1 % 8 == 0 {
                return Some(RepackKind::Q4K_8x8_Q8K);
            }
            if has_neon_dotprod && ne1 % 8 == 0 {
                return Some(RepackKind::Q4K_8x4_Q8K);
            }
            None
        }
        QuantType::Q2K => {
            if has_avx512 && ne1 % 8 == 0 {
                return Some(RepackKind::Q2K_8x8_Q8K);
            }
            None
        }
        QuantType::Q5K => {
            if has_neon_i8mm && ne1 % 8 == 0 {
                return Some(RepackKind::Q5K_8x8_Q8K);
            }
            if has_neon_dotprod && ne1 % 8 == 0 {
                return Some(RepackKind::Q5K_8x4_Q8K);
            }
            None
        }
        QuantType::Q6K => {
            if has_neon_i8mm && ne1 % 8 == 0 {
                return Some(RepackKind::Q6K_8x8_Q8K);
            }
            if has_neon_dotprod && ne1 % 8 == 0 {
                return Some(RepackKind::Q6K_8x4_Q8K);
            }
            None
        }
        QuantType::IQ4NL => {
            if has_avx2 && ne1 % 8 == 0 {
                return Some(RepackKind::IQ4NL_8x8_Q8_0);
            }
            if has_neon_dotprod && ne1 % 4 == 0 {
                return Some(RepackKind::IQ4NL_4x4_Q8_0);
            }
            None
        }
        QuantType::Mxfp4 => {
            if has_avx2 && ne1 % 8 == 0 {
                return Some(RepackKind::MXFP4_8x8_Q8_0);
            }
            if has_neon_dotprod && ne1 % 4 == 0 {
                return Some(RepackKind::MXFP4_4x4_Q8_0);
            }
            None
        }
        QuantType::Q8_0 => {
            if has_neon_i8mm && ne1 % 4 == 0 {
                return Some(RepackKind::Q8_0_4x8_Q8_0);
            }
            if has_neon_dotprod && ne1 % 4 == 0 {
                return Some(RepackKind::Q8_0_4x4_Q8_0);
            }
            None
        }
        QuantType::Q1_0 => {
            if has_neon_i8mm && ne1 % 4 == 0 {
                return Some(RepackKind::Q1_0_4x8_Q8_0);
            }
            if has_neon_dotprod && ne1 % 4 == 0 {
                return Some(RepackKind::Q1_0_4x4_Q8_0);
            }
            None
        }
    }
}

// -- Default impls needed by repack functions ----------------------------------

impl Default for BlockQ4_0 {
    fn default() -> Self {
        Self { d: 0, qs: [0; 16] }
    }
}

impl Default for BlockQ8_0 {
    fn default() -> Self {
        Self { d: 0, qs: [0; 32] }
    }
}

impl Default for BlockQ4K {
    fn default() -> Self {
        Self {
            d:      0,
            dmin:   0,
            scales: [0; 12],
            qs:     [0; 128],
        }
    }
}

impl Default for BlockQ2K {
    fn default() -> Self {
        Self {
            scales: [0; 16],
            qs:     [0; 64],
            d:      0,
            dmin:   0,
        }
    }
}

impl Default for BlockIQ4NL {
    fn default() -> Self {
        Self { d: 0, qs: [0; 16] }
    }
}

impl Default for BlockMxfp4 {
    fn default() -> Self {
        Self { e: 0, qs: [0; 16] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_q1_0() -> BlockQ1_0 {
        BlockQ1_0 { d: 0, qs: [0; 16] }
    }

    fn zero_q4_0() -> BlockQ4_0 {
        BlockQ4_0::default()
    }

    fn zero_iq4_nl() -> BlockIQ4NL {
        BlockIQ4NL::default()
    }

    fn zero_mxfp4() -> BlockMxfp4 {
        BlockMxfp4::default()
    }

    #[test]
    fn test_make_block_q1_0x4_blk8_round_trip_scale() {
        let mut blocks = [zero_q1_0(); 4];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.d = (i + 1) as u16;
        }
        let out = make_block_q1_0x4(&blocks, 8);
        for i in 0..4 {
            assert_eq!(out.d[i], blocks[i].d);
        }
    }

    #[test]
    fn test_make_block_q1_0x4_blk4_scale() {
        let mut blocks = [zero_q1_0(); 4];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.d = (i + 10) as u16;
        }
        let out = make_block_q1_0x4(&blocks, 4);
        for i in 0..4 {
            assert_eq!(out.d[i], blocks[i].d, "d[{i}] mismatch");
        }
    }

    #[test]
    fn test_make_block_q4_0x4_blk4_scales() {
        let mut blocks = [zero_q4_0(); 4];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.d = (i + 1) as u16;
        }
        let out = make_block_q4_0x4(&blocks, 4);
        for i in 0..4 {
            assert_eq!(out.d[i], (i + 1) as u16);
        }
    }

    #[test]
    fn test_make_block_q4_0x4_blk8_scales() {
        let mut blocks = [zero_q4_0(); 4];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.d = (i + 5) as u16;
        }
        let out = make_block_q4_0x4(&blocks, 8);
        for i in 0..4 {
            assert_eq!(out.d[i], (i + 5) as u16);
        }
    }

    #[test]
    fn test_make_block_q4_0x16_xor_mask() {
        let mut blocks = [zero_q4_0(); 16];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.qs.fill(0x00);
            b.d = i as u16;
        }
        let out = make_block_q4_0x16(&blocks, 1);
        // All source nibbles are 0, XOR with 0x88 -> output should be 0x88
        for byte in &out.qs {
            assert_eq!(*byte, 0x88, "expected 0x88 from XOR mask on zero input");
        }
    }

    #[test]
    fn test_make_block_iq4_nlx4_scales() {
        let mut blocks = [zero_iq4_nl(); 4];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.d = (i + 1) as u16;
        }
        let out = make_block_iq4_nlx4(&blocks, 4);
        for i in 0..4 {
            assert_eq!(out.d[i], (i + 1) as u16);
        }
    }

    #[test]
    fn test_make_block_iq4_nlx8_scales() {
        let mut blocks = [zero_iq4_nl(); 8];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.d = (i + 2) as u16;
        }
        let out = make_block_iq4_nlx8(&blocks, 8);
        for i in 0..8 {
            assert_eq!(out.d[i], (i + 2) as u16);
        }
    }

    #[test]
    fn test_make_block_mxfp4x4_scales() {
        let mut blocks = [zero_mxfp4(); 4];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.e = (i + 3) as u8;
        }
        let out = make_block_mxfp4x4(&blocks, 4);
        for i in 0..4 {
            assert_eq!(out.e[i], (i + 3) as u8);
        }
    }

    #[test]
    fn test_make_block_mxfp4x8_scales() {
        let mut blocks = [zero_mxfp4(); 8];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.e = (i + 1) as u8;
        }
        let out = make_block_mxfp4x8(&blocks, 8);
        for i in 0..8 {
            assert_eq!(out.e[i], (i + 1) as u8);
        }
    }

    #[test]
    fn test_repack_q1_0_bad_alignment() {
        // ne1 = 3 is not a multiple of 4 -> should return -1
        let shape = TensorShape { ne0: 128, ne1: 3 };
        let src  = vec![zero_q1_0(); 3 * 1];
        let mut dst = vec![BlockQ1_0x4 { d: [0; 4], qs: [0; 64] }; 10];
        let rc = repack_q1_0_to_q1_0_4_bl(&shape, 4, &src, &mut dst);
        assert_eq!(rc, -1);
    }

    #[test]
    fn test_repack_q8_0_bad_alignment() {
        let shape = TensorShape { ne0: 32, ne1: 3 };
        let src  = vec![BlockQ8_0::default(); 3];
        let mut dst = vec![BlockQ8_0x4 { d: [0; 4], qs: [0; 128] }; 10];
        let rc = repack_q8_0_to_q8_0_4_bl(&shape, 4, &src, &mut dst);
        assert_eq!(rc, -1);
    }
}
