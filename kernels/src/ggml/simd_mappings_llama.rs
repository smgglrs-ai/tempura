// Translation of llama.cpp ggml/src/ggml-cpu/simd-mappings.h
//
// Defines platform-specific SIMD type wrappers and element-per-register
// constants that mirror the GGML_F32_EPR / GGML_F32_VEC_* macro family.
// Only the AVX2, AVX512F, SSE3, and scalar paths are translated here;
// ARM NEON/SVE, RISC-V, POWER9, LoongArch, WASM, and s390x are omitted
// because this crate targets x86-64 inference workloads.

// ── FP16 ↔ FP32 conversion ────────────────────────────────────────────────
//
// On x86: GGML_CPU_COMPUTE_FP16_TO_FP32(x) = _cvtsh_ss(x)  (F16C)
//         GGML_CPU_COMPUTE_FP32_TO_FP16(x) = _cvtss_sh(x, 0) (F16C)
// Fallback: table lookup / IEEE 754 bit manipulation via `half` crate.

/// Convert a raw IEEE 754 half-precision bit pattern to f32.
///
/// With F16C hardware uses `_cvtsh_ss`; otherwise falls back to the `half`
/// crate's software decode, matching `ggml_lookup_fp16_to_fp32` semantics.
#[inline(always)]
pub fn fp16_to_fp32(h: u16) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
    unsafe {
        use std::arch::x86_64::*;
        // GGML_CPU_COMPUTE_FP16_TO_FP32: _cvtsh_ss(x)
        let v = _mm_cvtsi32_si128(h as i32);
        _mm_cvtss_f32(_mm_cvtph_ps(v))
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "f16c")))]
    {
        half::f16::from_bits(h).to_f32()
    }
}

/// Convert an f32 to a raw IEEE 754 half-precision bit pattern.
///
/// With F16C hardware uses `_cvtss_sh`; otherwise falls back to the `half`
/// crate, matching `GGML_CPU_FP32_TO_FP16` semantics.
#[inline(always)]
pub fn fp32_to_fp16(f: f32) -> u16 {
    #[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
    unsafe {
        use std::arch::x86_64::*;
        // GGML_CPU_COMPUTE_FP32_TO_FP16: _cvtss_sh(x, 0)
        let v = _mm_set_ss(f);
        let r = _mm_cvtps_ph(v, 0);
        _mm_extract_epi16(r, 0) as u16
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "f16c")))]
    {
        half::f16::from_f32(f).to_bits()
    }
}

// ── SIMD constants ─────────────────────────────────────────────────────────
//
// GGML_F32_EPR: elements per register (number of f32 lanes in one SIMD vec).
// GGML_F32_STEP = 4 * GGML_F32_EPR (llama.cpp unrolls 4 vectors per step).
// GGML_F32_ARR  = GGML_F32_STEP / GGML_F32_EPR = 4 (accumulator array size).
//
// AVX512F: __m512  → 16 lanes  STEP=64  ARR=4
// AVX2:    __m256  →  8 lanes  STEP=32  ARR=4
// SSE3:    __m128  →  4 lanes  STEP=32  ARR=8  (llama.cpp uses STEP=32/EPR=4)
// Scalar:           →  1 lane  (no SIMD)

#[cfg(target_feature = "avx512f")]
pub const F32_EPR: usize = 16;
#[cfg(target_feature = "avx512f")]
pub const F32_STEP: usize = 64;
#[cfg(target_feature = "avx512f")]
pub const F32_ARR: usize = 4;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub const F32_EPR: usize = 8;
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub const F32_STEP: usize = 32;
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub const F32_ARR: usize = 4;

#[cfg(all(
    target_feature = "sse3",
    not(target_feature = "avx2"),
    not(target_feature = "avx512f")
))]
pub const F32_EPR: usize = 4;
#[cfg(all(
    target_feature = "sse3",
    not(target_feature = "avx2"),
    not(target_feature = "avx512f")
))]
pub const F32_STEP: usize = 32;
#[cfg(all(
    target_feature = "sse3",
    not(target_feature = "avx2"),
    not(target_feature = "avx512f")
))]
pub const F32_ARR: usize = 8;

#[cfg(not(any(target_feature = "sse3", target_feature = "avx2", target_feature = "avx512f")))]
pub const F32_EPR: usize = 1;
#[cfg(not(any(target_feature = "sse3", target_feature = "avx2", target_feature = "avx512f")))]
pub const F32_STEP: usize = 1;
#[cfg(not(any(target_feature = "sse3", target_feature = "avx2", target_feature = "avx512f")))]
pub const F32_ARR: usize = 1;

// ── GEMM register tile constants ───────────────────────────────────────────
//
// Matching llama.cpp simd-gemm.h GEMM_RM / GEMM_RN.
// These are in units of F32_EPR (one tile column = RN * EPR floats).
//
// AVX512F / ARM NEON: RM=4  RN=4  → 4*4+4+1 = 21/32 registers
// AVX2    / AVX:      RM=6  RN=2  → 6*2+2+1 = 15/16 registers
// Other SIMD:         RM=2  RN=2

#[cfg(target_feature = "avx512f")]
pub const GEMM_RM: usize = 4;
#[cfg(target_feature = "avx512f")]
pub const GEMM_RN: usize = 4;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub const GEMM_RM: usize = 6;
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub const GEMM_RN: usize = 2;

#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
pub const GEMM_RM: usize = 2;
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
pub const GEMM_RN: usize = 2;

// ── AVX2 F32x8 operations ──────────────────────────────────────────────────
//
// Maps GGML_F32x8_* macros from simd-mappings.h § "F32 AVX" to Rust.
// All functions are `#[inline(always)] unsafe` to match the macro semantics.

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use std::arch::x86_64::*;

    // GGML_F32x8_ZERO = _mm256_setzero_ps()
    #[inline(always)]
    pub unsafe fn f32x8_zero() -> __m256 {
        _mm256_setzero_ps()
    }

    // GGML_F32x8_SET1(x) = _mm256_set1_ps(x)
    #[inline(always)]
    pub unsafe fn f32x8_set1(x: f32) -> __m256 {
        _mm256_set1_ps(x)
    }

    // GGML_F32x8_LOAD = _mm256_loadu_ps
    #[inline(always)]
    pub unsafe fn f32x8_load(p: *const f32) -> __m256 {
        _mm256_loadu_ps(p)
    }

    // GGML_F32x8_STORE = _mm256_storeu_ps
    #[inline(always)]
    pub unsafe fn f32x8_store(p: *mut f32, v: __m256) {
        _mm256_storeu_ps(p, v)
    }

    // GGML_F32x8_FMA(a, b, c) = _mm256_fmadd_ps(b, c, a)   [with FMA]
    //                         = _mm256_add_ps(_mm256_mul_ps(b, c), a)  [without]
    #[inline(always)]
    pub unsafe fn f32x8_fma(a: __m256, b: __m256, c: __m256) -> __m256 {
        #[cfg(target_feature = "fma")]
        {
            _mm256_fmadd_ps(b, c, a)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            _mm256_add_ps(_mm256_mul_ps(b, c), a)
        }
    }

    // GGML_F32x8_ADD = _mm256_add_ps
    #[inline(always)]
    pub unsafe fn f32x8_add(a: __m256, b: __m256) -> __m256 {
        _mm256_add_ps(a, b)
    }

    // GGML_F32x8_MUL = _mm256_mul_ps
    #[inline(always)]
    pub unsafe fn f32x8_mul(a: __m256, b: __m256) -> __m256 {
        _mm256_mul_ps(a, b)
    }

    // GGML_F32x8_REDUCE(res, x): horizontal sum of one __m256.
    // Equivalent to the scalar reduce of a single accumulator.
    //
    // Uses the same hadd pattern as llama.cpp's GGML_F32x8_REDUCE macro:
    //   t0 = _mm_add_ps(castps256_ps128(x), extractf128_ps(x, 1))
    //   t1 = _mm_hadd_ps(t0, t0)
    //   res = _mm_cvtss_f32(_mm_hadd_ps(t1, t1))
    #[inline(always)]
    pub unsafe fn f32x8_reduce_one(x: __m256) -> f32 {
        let t0 = _mm_add_ps(
            _mm256_castps256_ps128(x),
            _mm256_extractf128_ps(x, 1),
        );
        let t1 = _mm_hadd_ps(t0, t0);
        _mm_cvtss_f32(_mm_hadd_ps(t1, t1))
    }

    // GGML_F32x8_REDUCE over an array: pairwise-reduce ARR accumulators, then
    // call f32x8_reduce_one on the result.  Used in dot-product loops.
    #[inline(always)]
    pub unsafe fn f32x8_reduce_arr<const ARR: usize>(x: &mut [__m256; ARR]) -> f32 {
        let mut offset = ARR >> 1;
        while offset > 0 {
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            offset >>= 1;
        }
        f32x8_reduce_one(x[0])
    }
}

// ── AVX512F F32x16 operations ──────────────────────────────────────────────
//
// Maps GGML_F32x16_* macros from simd-mappings.h § "F32 AVX512".

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub mod avx512 {
    use std::arch::x86_64::*;

    // GGML_F32x16_ZERO = _mm512_setzero_ps()
    #[inline(always)]
    pub unsafe fn f32x16_zero() -> __m512 {
        _mm512_setzero_ps()
    }

    // GGML_F32x16_SET1(x) = _mm512_set1_ps(x)
    #[inline(always)]
    pub unsafe fn f32x16_set1(x: f32) -> __m512 {
        _mm512_set1_ps(x)
    }

    // GGML_F32x16_LOAD = _mm512_loadu_ps
    #[inline(always)]
    pub unsafe fn f32x16_load(p: *const f32) -> __m512 {
        _mm512_loadu_ps(p)
    }

    // GGML_F32x16_STORE = _mm512_storeu_ps
    #[inline(always)]
    pub unsafe fn f32x16_store(p: *mut f32, v: __m512) {
        _mm512_storeu_ps(p, v)
    }

    // GGML_F32x16_FMA(a, b, c) = _mm512_fmadd_ps(b, c, a)
    // AVX512F implies FMA support; no fallback needed.
    #[inline(always)]
    pub unsafe fn f32x16_fma(a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmadd_ps(b, c, a)
    }

    // GGML_F32x16_ADD = _mm512_add_ps
    #[inline(always)]
    pub unsafe fn f32x16_add(a: __m512, b: __m512) -> __m512 {
        _mm512_add_ps(a, b)
    }

    // GGML_F32x16_MUL = _mm512_mul_ps
    #[inline(always)]
    pub unsafe fn f32x16_mul(a: __m512, b: __m512) -> __m512 {
        _mm512_mul_ps(a, b)
    }

    // GGML_F32x16_REDUCE: uses _mm512_reduce_add_ps for the final scalar.
    #[inline(always)]
    pub unsafe fn f32x16_reduce_one(x: __m512) -> f32 {
        _mm512_reduce_add_ps(x)
    }

    #[inline(always)]
    pub unsafe fn f32x16_reduce_arr<const ARR: usize>(x: &mut [__m512; ARR]) -> f32 {
        let mut offset = ARR >> 1;
        while offset > 0 {
            for i in 0..offset {
                x[i] = _mm512_add_ps(x[i], x[offset + i]);
            }
            offset >>= 1;
        }
        f32x16_reduce_one(x[0])
    }
}

// ── SSE3 F32x4 operations ─────────────────────────────────────────────────
//
// Maps GGML_F32x4_* macros from simd-mappings.h § "F32 SSE".

#[cfg(target_arch = "x86_64")]
pub mod sse3 {
    use std::arch::x86_64::*;

    // GGML_F32x4_ZERO = _mm_setzero_ps()
    #[inline(always)]
    pub unsafe fn f32x4_zero() -> __m128 {
        _mm_setzero_ps()
    }

    // GGML_F32x4_SET1(x) = _mm_set1_ps(x)
    #[inline(always)]
    pub unsafe fn f32x4_set1(x: f32) -> __m128 {
        _mm_set1_ps(x)
    }

    // GGML_F32x4_LOAD = _mm_loadu_ps
    #[inline(always)]
    pub unsafe fn f32x4_load(p: *const f32) -> __m128 {
        _mm_loadu_ps(p)
    }

    // GGML_F32x4_STORE = _mm_storeu_ps
    #[inline(always)]
    pub unsafe fn f32x4_store(p: *mut f32, v: __m128) {
        _mm_storeu_ps(p, v)
    }

    // GGML_F32x4_FMA(a, b, c) = _mm_fmadd_ps(b, c, a)  [with FMA]
    //                          = _mm_add_ps(_mm_mul_ps(b, c), a) [without]
    #[inline(always)]
    pub unsafe fn f32x4_fma(a: __m128, b: __m128, c: __m128) -> __m128 {
        #[cfg(target_feature = "fma")]
        {
            _mm_fmadd_ps(b, c, a)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            _mm_add_ps(_mm_mul_ps(b, c), a)
        }
    }

    // GGML_F32x4_ADD = _mm_add_ps
    #[inline(always)]
    pub unsafe fn f32x4_add(a: __m128, b: __m128) -> __m128 {
        _mm_add_ps(a, b)
    }

    // GGML_F32x4_MUL = _mm_mul_ps
    #[inline(always)]
    pub unsafe fn f32x4_mul(a: __m128, b: __m128) -> __m128 {
        _mm_mul_ps(a, b)
    }

    // GGML_F32x4_REDUCE: SSE3 hadd path.
    #[inline(always)]
    pub unsafe fn f32x4_reduce_one(x: __m128) -> f32 {
        let t0 = _mm_hadd_ps(x, x);
        _mm_cvtss_f32(_mm_hadd_ps(t0, t0))
    }

    #[inline(always)]
    pub unsafe fn f32x4_reduce_arr<const ARR: usize>(x: &mut [__m128; ARR]) -> f32 {
        let mut offset = ARR >> 1;
        while offset > 0 {
            for i in 0..offset {
                x[i] = _mm_add_ps(x[i], x[offset + i]);
            }
            offset >>= 1;
        }
        f32x4_reduce_one(x[0])
    }
}
