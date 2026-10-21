//! Direct Rust translation of llama.cpp ggml/src/ggml-cpu/ggml-cpu-impl.h (539 lines).
//!
//! Covers:
//! - GgmlComputeParams struct (ggml_compute_params)
//! - ARM NEON 32-bit compat shims (aarch32 fallbacks for AArch64 intrinsics)
//! - ARM NEON ggml_vdotq_s32 fallback when __ARM_FEATURE_DOTPROD absent
//! - ARM NEON ggml_nvfp4_dot8 helper
//! - WASM SIMD128 (via feature gate; wasm_simd128.h has no extra helpers here)
//! - POWER9/AltiVec (via feature gate; altivec.h has no extra helpers here)
//! - s390x VXE / VXE2 -- vector types, operator macros, and helper functions
//! - LoongArch LSX/LASX scalar broadcast helpers

// -- GgmlComputeParams --------------------------------------------------------
// Mirrors: struct ggml_compute_params { int ith, nth; size_t wsize; void* wdata; ... }

#[derive(Debug, Clone)]
pub struct GgmlComputeParams {
    pub ith: i32,
    pub nth: i32,
    pub wsize: usize,
    pub wdata: *mut std::ffi::c_void,
    pub use_ref: bool,
}

impl Default for GgmlComputeParams {
    fn default() -> Self {
        GgmlComputeParams {
            ith: 0,
            nth: 1,
            wsize: 0,
            wdata: std::ptr::null_mut(),
            use_ref: false,
        }
    }
}

// -- ARM NEON -----------------------------------------------------------------
//
// ggml_vld1q_u32(w,x,y,z) macro: constructs a uint32x4_t from 4 u32 values.
// On MSVC: packed as two u64; on all other compilers: direct array.
// In Rust this is a no-op pattern since we use array literals directly, but we
// provide a helper for source-level parity.

#[cfg(target_arch = "arm")]
pub mod neon_compat {
    //! 32-bit ARM (AArch32) compatibility shims for AArch64 NEON intrinsics.
    //! Mirrors the ggml-cpu-impl.h block guarded by `!defined(__aarch64__)`.
    //!
    //! These implement the AArch64-only intrinsics in terms of the AArch32
    //! NEON instructions that are available on both.

    use std::arch::arm::*;

    // vaddlvq_s16: sum all 8 int16 lanes into int32.
    // C: int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    //    return vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
    #[inline(always)]
    pub unsafe fn vaddlvq_s16_compat(v: int16x8_t) -> i32 {
        let v0: int32x4_t = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
        vgetq_lane_s32::<0>(v0) + vgetq_lane_s32::<2>(v0)
    }

    // vpaddq_s16: pairwise add, combining two int16x8_t into one int16x8_t.
    // C: int16x4_t a0 = vpadd_s16(...); int16x4_t b0 = vpadd_s16(...);
    //    return vcombine_s16(a0, b0);
    #[inline(always)]
    pub unsafe fn vpaddq_s16_compat(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        let a0 = vpadd_s16(vget_low_s16(a), vget_high_s16(a));
        let b0 = vpadd_s16(vget_low_s16(b), vget_high_s16(b));
        vcombine_s16(a0, b0)
    }

    // vpaddq_s32: pairwise add of two int32x4_t.
    #[inline(always)]
    pub unsafe fn vpaddq_s32_compat(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        let a0 = vpadd_s32(vget_low_s32(a), vget_high_s32(a));
        let b0 = vpadd_s32(vget_low_s32(b), vget_high_s32(b));
        vcombine_s32(a0, b0)
    }

    // vaddvq_s32: horizontal add of all 4 int32 lanes.
    #[inline(always)]
    pub unsafe fn vaddvq_s32_compat(v: int32x4_t) -> i32 {
        vgetq_lane_s32::<0>(v) + vgetq_lane_s32::<1>(v)
            + vgetq_lane_s32::<2>(v) + vgetq_lane_s32::<3>(v)
    }

    // vaddvq_f32: horizontal add of all 4 float32 lanes.
    #[inline(always)]
    pub unsafe fn vaddvq_f32_compat(v: float32x4_t) -> f32 {
        vgetq_lane_f32::<0>(v) + vgetq_lane_f32::<1>(v)
            + vgetq_lane_f32::<2>(v) + vgetq_lane_f32::<3>(v)
    }

    // vmaxvq_f32: horizontal max of all 4 float32 lanes.
    #[inline(always)]
    pub unsafe fn vmaxvq_f32_compat(v: float32x4_t) -> f32 {
        let a = vgetq_lane_f32::<0>(v).max(vgetq_lane_f32::<1>(v));
        let b = vgetq_lane_f32::<2>(v).max(vgetq_lane_f32::<3>(v));
        a.max(b)
    }

    // vcvtnq_s32_f32: round-to-nearest float32x4 -> int32x4.
    // C: res[i] = roundf(vgetq_lane_f32(v, i));
    #[inline(always)]
    pub unsafe fn vcvtnq_s32_f32_compat(v: float32x4_t) -> int32x4_t {
        let mut res = [0i32; 4];
        res[0] = vgetq_lane_f32::<0>(v).round() as i32;
        res[1] = vgetq_lane_f32::<1>(v).round() as i32;
        res[2] = vgetq_lane_f32::<2>(v).round() as i32;
        res[3] = vgetq_lane_f32::<3>(v).round() as i32;
        vld1q_s32(res.as_ptr())
    }

    // vzip1_u8: interleave lower halves of two uint8x8_t.
    // C: res[2i]=a[i], res[2i+1]=b[i] for i in 0..4.
    #[inline(always)]
    pub unsafe fn vzip1_u8_compat(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
        let av: [u8; 8] = std::mem::transmute(a);
        let bv: [u8; 8] = std::mem::transmute(b);
        let res: [u8; 8] = [av[0], bv[0], av[1], bv[1], av[2], bv[2], av[3], bv[3]];
        std::mem::transmute(res)
    }

    // vzip2_u8: interleave upper halves of two uint8x8_t.
    // C: res[2i]=a[i+4], res[2i+1]=b[i+4] for i in 0..4.
    #[inline(always)]
    pub unsafe fn vzip2_u8_compat(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
        let av: [u8; 8] = std::mem::transmute(a);
        let bv: [u8; 8] = std::mem::transmute(b);
        let res: [u8; 8] = [av[4], bv[4], av[5], bv[5], av[6], bv[6], av[7], bv[7]];
        std::mem::transmute(res)
    }

    // vld1q_s16_x2 compat: load 16 int16 values as a pair of int16x8.
    #[derive(Copy, Clone)]
    pub struct Int16x8x2(pub int16x8_t, pub int16x8_t);

    #[inline(always)]
    pub unsafe fn ggml_vld1q_s16_x2(ptr: *const i16) -> Int16x8x2 {
        Int16x8x2(vld1q_s16(ptr), vld1q_s16(ptr.add(8)))
    }

    // vld1q_u8_x2 compat.
    #[derive(Copy, Clone)]
    pub struct Uint8x16x2(pub uint8x16_t, pub uint8x16_t);

    #[inline(always)]
    pub unsafe fn ggml_vld1q_u8_x2(ptr: *const u8) -> Uint8x16x2 {
        Uint8x16x2(vld1q_u8(ptr), vld1q_u8(ptr.add(16)))
    }

    // vld1q_u8_x4 compat.
    #[derive(Copy, Clone)]
    pub struct Uint8x16x4(pub uint8x16_t, pub uint8x16_t, pub uint8x16_t, pub uint8x16_t);

    #[inline(always)]
    pub unsafe fn ggml_vld1q_u8_x4(ptr: *const u8) -> Uint8x16x4 {
        Uint8x16x4(
            vld1q_u8(ptr),
            vld1q_u8(ptr.add(16)),
            vld1q_u8(ptr.add(32)),
            vld1q_u8(ptr.add(48)),
        )
    }

    // vld1q_s8_x2 compat.
    #[derive(Copy, Clone)]
    pub struct Int8x16x2(pub int8x16_t, pub int8x16_t);

    #[inline(always)]
    pub unsafe fn ggml_vld1q_s8_x2(ptr: *const i8) -> Int8x16x2 {
        Int8x16x2(vld1q_s8(ptr), vld1q_s8(ptr.add(16)))
    }

    // vld1q_s8_x4 compat.
    #[derive(Copy, Clone)]
    pub struct Int8x16x4(pub int8x16_t, pub int8x16_t, pub int8x16_t, pub int8x16_t);

    #[inline(always)]
    pub unsafe fn ggml_vld1q_s8_x4(ptr: *const i8) -> Int8x16x4 {
        Int8x16x4(
            vld1q_s8(ptr),
            vld1q_s8(ptr.add(16)),
            vld1q_s8(ptr.add(32)),
            vld1q_s8(ptr.add(48)),
        )
    }

    // ggml_vqtbl1q_s8: table-lookup (indexed byte permute), scalar fallback.
    // NOTE: not tested (matches the C comment).
    #[inline(always)]
    pub unsafe fn ggml_vqtbl1q_s8_compat(a: int8x16_t, b: uint8x16_t) -> int8x16_t {
        let av: [i8; 16] = std::mem::transmute(a);
        let bv: [u8; 16] = std::mem::transmute(b);
        let mut res = [0i8; 16];
        for i in 0..16usize {
            res[i] = av[(bv[i] & 0xf) as usize];
        }
        std::mem::transmute(res)
    }

    // ggml_vqtbl1q_u8: table-lookup, unsigned variant.
    #[inline(always)]
    pub unsafe fn ggml_vqtbl1q_u8_compat(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        let av: [u8; 16] = std::mem::transmute(a);
        let bv: [u8; 16] = std::mem::transmute(b);
        let mut res = [0u8; 16];
        for i in 0..16usize {
            res[i] = av[(bv[i] & 0xf) as usize];
        }
        std::mem::transmute(res)
    }
}

// -- ARM NEON (both AArch32 and AArch64) -------------------------------------
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub mod neon_dot {
    //! ggml_vdotq_s32: dot product fallback when __ARM_FEATURE_DOTPROD is absent.
    //! Also: ggml_nvfp4_dot8 helper.

    use std::arch::aarch64::*;

    // ggml_vdotq_s32 -- fallback when vdotq_s32 unavailable.
    // NOTE: per the original comment, this produces the same total sum but with
    // different per-lane grouping vs native vdotq_s32.
    #[cfg(not(target_feature = "dotprod"))]
    #[inline(always)]
    pub unsafe fn ggml_vdotq_s32(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
        vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)))
    }

    // When dotprod is available, ggml_vdotq_s32 is just vdotq_s32.
    #[cfg(target_feature = "dotprod")]
    #[inline(always)]
    pub unsafe fn ggml_vdotq_s32(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        vdotq_s32(acc, a, b)
    }

    // ggml_nvfp4_dot8: dot product of two pairs of int8x8 -- used for NVFP4.
    // C: const int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    //    const int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    //    const int32x4_t sum_lo = vpaddlq_s16(p_lo);
    //    const int32x4_t sum_hi = vpaddlq_s16(p_hi);
    //    return vaddq_s32(sum_lo, sum_hi);
    #[inline(always)]
    pub unsafe fn ggml_nvfp4_dot8(
        q4_lo: int8x8_t,
        q8_lo: int8x8_t,
        q4_hi: int8x8_t,
        q8_hi: int8x8_t,
    ) -> int32x4_t {
        let p_lo = vmull_s8(q4_lo, q8_lo);
        let p_hi = vmull_s8(q4_hi, q8_hi);
        let sum_lo = vpaddlq_s16(p_lo);
        let sum_hi = vpaddlq_s16(p_hi);
        vaddq_s32(sum_lo, sum_hi)
    }
}

// -- AArch64 (on AArch64 all the compat aliases are simple type aliases) ------
// On AArch64, the C code defines:
//   #define ggml_int16x8x2_t  int16x8x2_t
//   #define ggml_vld1q_s16_x2 vld1q_s16_x2
//   ... etc.
// In Rust, these re-export from std::arch::aarch64 directly; no extra code needed.

// -- s390x VXE / VXE2 ---------------------------------------------------------
#[cfg(target_arch = "s390x")]
pub mod vxe {
    //! Translation of the __VXE__ / __VXE2__ block in ggml-cpu-impl.h.
    //!
    //! The C code defines:
    //!   - Operator macros: vec_neg, vec_add, vec_sub, vec_mul, vec_div, vec_sl,
    //!     vec_sra, vec_sr, vec_slo, vec_sro, vec_and, vec_or, vec_xor
    //!   - Vector typedefs using GCC __attribute__((vector_size(16)))
    //!   - ggml_uint8x16x2_t / x4, ggml_int8x16x4_t, ggml_int16x8x2_t struct types
    //!   - Helper functions: ggml_vec_xl_u8x2, x4, s8x4, s16x2
    //!   - ggml_vec_tbl: scalar table-lookup
    //!   - vec_padd_s16: pairwise add for int16x8
    //!   - vec_hsum_f32x4, vec_hsum_i32x4: horizontal sums
    //!   - ggml_vec_dot: int8 dot product accumulate
    //!
    //! Rust uses repr(simd) via std::simd or raw [T; N] arrays since stable Rust
    //! does not yet expose s390x vector intrinsics. We use byte arrays of the
    //! correct size so the types have the right ABI layout.

    // Vector types (16-byte, matching GCC vector_size(16)).
    // We represent them as arrays for stable Rust portability.
    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug)]
    pub struct Int8x16(pub [i8; 16]);

    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug)]
    pub struct Uint8x16(pub [u8; 16]);

    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug)]
    pub struct Int16x8(pub [i16; 8]);

    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug)]
    pub struct Int32x4(pub [i32; 4]);

    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug)]
    pub struct Uint16x8(pub [u16; 8]);

    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug)]
    pub struct Uint32x4(pub [u32; 4]);

    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug)]
    pub struct Float32x4(pub [f32; 4]);

    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug)]
    pub struct Double64x2(pub [f64; 2]);

    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug)]
    pub struct Long64x2(pub [i64; 2]);

    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug)]
    pub struct Ulong64x2(pub [u64; 2]);

    // Struct types mirroring ggml_uint8x16x2_t etc.
    #[derive(Copy, Clone, Debug)]
    pub struct GgmlUint8x16x2(pub Uint8x16, pub Uint8x16);

    #[derive(Copy, Clone, Debug)]
    pub struct GgmlUint8x16x4(pub Uint8x16, pub Uint8x16, pub Uint8x16, pub Uint8x16);

    #[derive(Copy, Clone, Debug)]
    pub struct GgmlInt8x16x4(pub Int8x16, pub Int8x16, pub Int8x16, pub Int8x16);

    #[derive(Copy, Clone, Debug)]
    pub struct GgmlInt16x8x2(pub Int16x8, pub Int16x8);

    // ggml_vec_xl_u8x2: load two 16-byte uint8 vectors from ptr.
    // C: res.val[0] = vec_xl(0, ptr); res.val[1] = vec_xl(16, ptr);
    pub fn ggml_vec_xl_u8x2(ptr: *const u8) -> GgmlUint8x16x2 {
        unsafe {
            let mut v0 = Uint8x16([0u8; 16]);
            let mut v1 = Uint8x16([0u8; 16]);
            std::ptr::copy_nonoverlapping(ptr, v0.0.as_mut_ptr(), 16);
            std::ptr::copy_nonoverlapping(ptr.add(16), v1.0.as_mut_ptr(), 16);
            GgmlUint8x16x2(v0, v1)
        }
    }

    // ggml_vec_xl_u8x4: load four 16-byte uint8 vectors.
    pub fn ggml_vec_xl_u8x4(ptr: *const u8) -> GgmlUint8x16x4 {
        unsafe {
            let mut vals = [Uint8x16([0u8; 16]); 4];
            for i in 0..4 {
                std::ptr::copy_nonoverlapping(ptr.add(i * 16), vals[i].0.as_mut_ptr(), 16);
            }
            GgmlUint8x16x4(vals[0], vals[1], vals[2], vals[3])
        }
    }

    // ggml_vec_xl_s8x4: load four 16-byte int8 vectors.
    pub fn ggml_vec_xl_s8x4(ptr: *const i8) -> GgmlInt8x16x4 {
        unsafe {
            let mut vals = [Int8x16([0i8; 16]); 4];
            for i in 0..4 {
                std::ptr::copy_nonoverlapping(ptr.add(i * 16), vals[i].0.as_mut_ptr(), 16);
            }
            GgmlInt8x16x4(vals[0], vals[1], vals[2], vals[3])
        }
    }

    // ggml_vec_xl_s16x2: load two 8-element int16 vectors.
    pub fn ggml_vec_xl_s16x2(ptr: *const i16) -> GgmlInt16x8x2 {
        unsafe {
            let mut v0 = Int16x8([0i16; 8]);
            let mut v1 = Int16x8([0i16; 8]);
            std::ptr::copy_nonoverlapping(ptr as *const u8, v0.0.as_mut_ptr() as *mut u8, 16);
            std::ptr::copy_nonoverlapping(ptr.add(8) as *const u8, v1.0.as_mut_ptr() as *mut u8, 16);
            GgmlInt16x8x2(v0, v1)
        }
    }

    // ggml_vec_tbl: scalar table-lookup (WARNING: very slow per original comment).
    // C: res[i] = a[b[i]]; for i in 0..16
    pub fn ggml_vec_tbl(a: Int8x16, b: Uint8x16) -> Int8x16 {
        let mut res = [0i8; 16];
        for i in 0..16 {
            res[i] = a.0[(b.0[i] & 0xf) as usize];
        }
        Int8x16(res)
    }

    // vec_padd_s16: pairwise horizontal add of two int16x8 -> int16x8.
    // C: const uchar8x16_t v_maske = {0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29};
    //    v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);  // pack high 16 of each 32
    //    v_abe = vec_perm(a, b, v_maske);               // interleave evens
    //    return v_abo + v_abe;
    pub fn vec_padd_s16(a: Int16x8, b: Int16x8) -> Int16x8 {
        let mut res = [0i16; 8];
        // Even lanes: a[0]+a[1], a[2]+a[3], a[4]+a[5], a[6]+a[7]
        // then b[0]+b[1], b[2]+b[3], b[4]+b[5], b[6]+b[7]
        for i in 0..4 {
            res[i] = a.0[2 * i].wrapping_add(a.0[2 * i + 1]);
        }
        for i in 0..4 {
            res[4 + i] = b.0[2 * i].wrapping_add(b.0[2 * i + 1]);
        }
        Int16x8(res)
    }

    // vec_hsum_f32x4: horizontal sum of 4 float32 lanes.
    // C: float32x4_t v_temp = v + vec_reve(v); return v_temp[0] + v_temp[1];
    pub fn vec_hsum_f32x4(v: Float32x4) -> f32 {
        // vec_reve reverses element order: [3,2,1,0]
        let rev = [v.0[3], v.0[2], v.0[1], v.0[0]];
        let temp = [v.0[0] + rev[0], v.0[1] + rev[1], v.0[2] + rev[2], v.0[3] + rev[3]];
        temp[0] + temp[1]
    }

    // vec_hsum_i32x4: horizontal sum of 4 int32 lanes.
    pub fn vec_hsum_i32x4(v: Int32x4) -> i32 {
        let rev = [v.0[3], v.0[2], v.0[1], v.0[0]];
        let temp = [v.0[0].wrapping_add(rev[0]), v.0[1].wrapping_add(rev[1]),
                    v.0[2].wrapping_add(rev[2]), v.0[3].wrapping_add(rev[3])];
        temp[0].wrapping_add(temp[1])
    }

    // ggml_vec_dot: accumulate int8 dot product into int32x4.
    // C: const int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    //    return acc + (vec_unpackh(p) + vec_unpackl(p));
    // vec_mule/vec_mulo: even/odd element multiply -> int16x8_t
    // vec_unpackh/l: sign-extend int16 to int32
    pub fn ggml_vec_dot(acc: Int32x4, a: Int8x16, b: Int8x16) -> Int32x4 {
        // mule: multiply even-indexed bytes (0,2,4,...14)
        let mut p_even = [0i16; 8];
        for i in 0..8 {
            p_even[i] = (a.0[2 * i] as i16) * (b.0[2 * i] as i16);
        }
        // mulo: multiply odd-indexed bytes (1,3,5,...15)
        let mut p_odd = [0i16; 8];
        for i in 0..8 {
            p_odd[i] = (a.0[2 * i + 1] as i16) * (b.0[2 * i + 1] as i16);
        }
        // p = p_even + p_odd
        let mut p = [0i16; 8];
        for i in 0..8 {
            p[i] = p_even[i].wrapping_add(p_odd[i]);
        }
        // unpackh: sign-extend upper 4 int16 to int32
        // unpackl: sign-extend lower 4 int16 to int32
        // sum: (unpackh + unpackl) gives element-wise 8->4 sum, then acc+
        let mut res = [0i32; 4];
        for i in 0..4 {
            res[i] = acc.0[i]
                .wrapping_add((p[i] as i32).wrapping_add(p[i + 4] as i32));
        }
        Int32x4(res)
    }
}

// -- LoongArch LSX/LASX --------------------------------------------------------
#[cfg(target_arch = "loongarch64")]
pub mod loongarch {
    //! Translation of LoongArch scalar broadcast helpers from ggml-cpu-impl.h.
    //!
    //! The C code uses GCC vector extension types (__m128/__m256) for LoongArch
    //! LSX/LASX. Rust exposes LoongArch intrinsics through nightly only, so we
    //! provide scalar-equivalent functions here for completeness.

    // __lsx_vreplfr2vr_s: broadcast f32 scalar into a 4-lane float32 LSX vector.
    // C: v4f32 res = {val, val, val, val}; return (__m128)res;
    pub fn lsx_vreplfr2vr_s(val: f32) -> [f32; 4] {
        [val, val, val, val]
    }

    // __lasx_xvreplfr2vr_s: broadcast f32 scalar into an 8-lane float32 LASX vector.
    // C: v8f32 res = {val, val, val, val, val, val, val, val}; return (__m256)res;
    pub fn lasx_xvreplfr2vr_s(val: f32) -> [f32; 8] {
        [val, val, val, val, val, val, val, val]
    }
}

// -- Threadpool interface ------------------------------------------------------
// These are declared in ggml-cpu-impl.h with external linkage.
// In the Rust codebase they are implemented elsewhere; we declare them as
// extern "C" so that any future FFI call matches the original ABI.
//
// NOTE: ggml_barrier, ggml_threadpool_chunk_set, ggml_threadpool_chunk_add
// are declared but not defined here -- definition is in ggml-threading.
// Commented out to avoid unresolved symbol errors until FFI is wired up.
//
// extern "C" {
//     pub fn ggml_barrier(tp: *mut std::ffi::c_void);
//     pub fn ggml_threadpool_chunk_set(tp: *mut std::ffi::c_void, value: i32);
//     pub fn ggml_threadpool_chunk_add(tp: *mut std::ffi::c_void, value: i32) -> i32;
// }
