// Translation of llama.cpp ggml/src/ggml-cpu/simd-gemm.h
//
// Computes C[M × N] += A[M × K] × B[K × N] (accumulate into C).
//
// Three implementations are provided, selected at compile time:
//
//   1. avx2_gemm   — AVX2 path, RM=6 × RN=2 × 8 tile  (15/16 ymm registers)
//   2. avx512_gemm — AVX-512F path, RM=4 × RN=4 × 16 tile (21/32 zmm registers)
//   3. scalar_gemm — portable fallback with no intrinsics
//
// The public `simd_gemm` entry point dispatches at compile time to the
// fastest available path, matching the #ifdef chain in the original header.

// ── AVX2 path ─────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
mod avx2_impl {
    use std::arch::x86_64::*;
    use super::super::simd_mappings_llama::avx2;

    const EPR: usize = 8; // GGML_F32_EPR for AVX2: 8 f32 per __m256

    // simd_gemm_ukernel<RM, RN>:
    //   C[ RM × (RN*EPR) ] += A[ RM × K ] × B[ K × (RN*EPR) ]
    //
    // Template parameters become Rust const generics.
    // Each accumulator covers one (row, col-block) pair:
    //   acc[i][r] accumulates row i, columns [r*EPR .. (r+1)*EPR).
    //
    // Register pressure for AVX2 RM=6, RN=2:
    //   12 acc + 2 Bv + 1 p = 15 ymm registers (fits in 16).
    #[target_feature(enable = "avx2")]
    pub unsafe fn ukernel<const RM: usize, const RN: usize>(
        c: *mut f32,
        a: *const f32,
        b: *const f32,
        k: usize,
        n: usize,
    ) {
        // Load RM*RN accumulators from C.
        // GGML_F32_VEC acc[RM][RN]; for(...) acc[i][r] = GGML_F32_VEC_LOAD(C + i*N + r*KN);
        let mut acc: [[__m256; RN]; RM] = std::mem::zeroed();
        for i in 0..RM {
            for r in 0..RN {
                acc[i][r] = avx2::f32x8_load(c.add(i * n + r * EPR));
            }
        }

        // Inner K loop.
        // for(kk=0; kk<K; kk++) {
        //   Bv[r] = GGML_F32_VEC_LOAD(B + kk*N + r*KN);
        //   p     = GGML_F32_VEC_SET1(A[i*K + kk]);
        //   acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
        // }
        for kk in 0..k {
            let mut bv: [__m256; RN] = std::mem::zeroed();
            for r in 0..RN {
                bv[r] = avx2::f32x8_load(b.add(kk * n + r * EPR));
            }
            for i in 0..RM {
                let p = avx2::f32x8_set1(*a.add(i * k + kk));
                for r in 0..RN {
                    acc[i][r] = avx2::f32x8_fma(acc[i][r], bv[r], p);
                }
            }
        }

        // Store RM*RN accumulators back to C.
        // for(...) GGML_F32_VEC_STORE(C + i*N + r*KN, acc[i][r]);
        for i in 0..RM {
            for r in 0..RN {
                avx2::f32x8_store(c.add(i * n + r * EPR), acc[i][r]);
            }
        }
    }

    // simd_gemm (AVX2): C[M×N] += A[M×K] × B[K×N]
    //
    // Outer tiling loop from llama.cpp simd-gemm.h `simd_gemm` function.
    // RM=6, RN=2 for AVX2 as per GEMM_RM/GEMM_RN constants.
    #[target_feature(enable = "avx2")]
    pub unsafe fn gemm(
        c: *mut f32,
        a: *const f32,
        b: *const f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        const RM: usize = 6;
        const RN: usize = 2;

        let mut a = a;
        let mut c = c;
        let mut ii = 0usize;

        // Full-RM row blocks.
        // for (; ii + GEMM_RM <= M; ii += GEMM_RM) { ... }
        while ii + RM <= m {
            let mut jj = 0usize;

            // Full RN*EPR column blocks.
            // for (; jj + GEMM_RN*KN <= N; jj += GEMM_RN*KN)
            //   simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C+jj, A, B+jj, K, N);
            while jj + RN * EPR <= n {
                ukernel::<RM, RN>(c.add(jj), a, b.add(jj), k, n);
                jj += RN * EPR;
            }

            // Remaining single-EPR column blocks.
            // for (; jj + KN <= N; jj += KN)
            //   simd_gemm_ukernel<GEMM_RM, 1>(C+jj, A, B+jj, K, N);
            while jj + EPR <= n {
                ukernel::<RM, 1>(c.add(jj), a, b.add(jj), k, n);
                jj += EPR;
            }

            // Scalar tail for leftover columns.
            // for (; jj < N; jj++) { for i in 0..GEMM_RM { scalar fma } }
            while jj < n {
                for i in 0..RM {
                    let mut acc = *c.add(i * n + jj);
                    for kk in 0..k {
                        acc += *a.add(i * k + kk) * *b.add(kk * n + jj);
                    }
                    *c.add(i * n + jj) = acc;
                }
                jj += 1;
            }

            a = a.add(RM * k);
            c = c.add(RM * n);
            ii += RM;
        }

        // Tail rows: one at a time.
        // for (; ii < M; ii++) { ... }
        while ii < m {
            let mut jj = 0usize;

            while jj + RN * EPR <= n {
                ukernel::<1, RN>(c.add(jj), a, b.add(jj), k, n);
                jj += RN * EPR;
            }
            while jj + EPR <= n {
                ukernel::<1, 1>(c.add(jj), a, b.add(jj), k, n);
                jj += EPR;
            }
            while jj < n {
                let mut acc = *c.add(jj);
                for kk in 0..k {
                    acc += *a.add(kk) * *b.add(kk * n + jj);
                }
                *c.add(jj) = acc;
                jj += 1;
            }

            a = a.add(k);
            c = c.add(n);
            ii += 1;
        }
    }
}

// ── AVX-512F path ─────────────────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[allow(dead_code)]
mod avx512_impl {
    use std::arch::x86_64::*;
    use super::super::simd_mappings_llama::avx512;

    const EPR: usize = 16; // GGML_F32_EPR for AVX512: 16 f32 per __m512

    // simd_gemm_ukernel<RM, RN> for AVX-512.
    // RM=4, RN=4: 4*4+4+1 = 21/32 zmm registers.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn ukernel<const RM: usize, const RN: usize>(
        c: *mut f32,
        a: *const f32,
        b: *const f32,
        k: usize,
        n: usize,
    ) {
        let mut acc: [[__m512; RN]; RM] = std::mem::zeroed();
        for i in 0..RM {
            for r in 0..RN {
                acc[i][r] = avx512::f32x16_load(c.add(i * n + r * EPR));
            }
        }
        for kk in 0..k {
            let mut bv: [__m512; RN] = std::mem::zeroed();
            for r in 0..RN {
                bv[r] = avx512::f32x16_load(b.add(kk * n + r * EPR));
            }
            for i in 0..RM {
                let p = avx512::f32x16_set1(*a.add(i * k + kk));
                for r in 0..RN {
                    acc[i][r] = avx512::f32x16_fma(acc[i][r], bv[r], p);
                }
            }
        }
        for i in 0..RM {
            for r in 0..RN {
                avx512::f32x16_store(c.add(i * n + r * EPR), acc[i][r]);
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn gemm(
        c: *mut f32,
        a: *const f32,
        b: *const f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        const RM: usize = 4;
        const RN: usize = 4;

        let mut a = a;
        let mut c = c;
        let mut ii = 0usize;

        while ii + RM <= m {
            let mut jj = 0usize;
            while jj + RN * EPR <= n {
                ukernel::<RM, RN>(c.add(jj), a, b.add(jj), k, n);
                jj += RN * EPR;
            }
            while jj + EPR <= n {
                ukernel::<RM, 1>(c.add(jj), a, b.add(jj), k, n);
                jj += EPR;
            }
            while jj < n {
                for i in 0..RM {
                    let mut acc = *c.add(i * n + jj);
                    for kk in 0..k {
                        acc += *a.add(i * k + kk) * *b.add(kk * n + jj);
                    }
                    *c.add(i * n + jj) = acc;
                }
                jj += 1;
            }
            a = a.add(RM * k);
            c = c.add(RM * n);
            ii += RM;
        }

        while ii < m {
            let mut jj = 0usize;
            while jj + RN * EPR <= n {
                ukernel::<1, RN>(c.add(jj), a, b.add(jj), k, n);
                jj += RN * EPR;
            }
            while jj + EPR <= n {
                ukernel::<1, 1>(c.add(jj), a, b.add(jj), k, n);
                jj += EPR;
            }
            while jj < n {
                let mut acc = *c.add(jj);
                for kk in 0..k {
                    acc += *a.add(kk) * *b.add(kk * n + jj);
                }
                *c.add(jj) = acc;
                jj += 1;
            }
            a = a.add(k);
            c = c.add(n);
            ii += 1;
        }
    }
}

// ── Scalar fallback path ───────────────────────────────────────────────────
//
// Matches the `#else // scalar path` branch in simd-gemm.h.
// Used when no SIMD feature is detected at compile time.

fn scalar_gemm(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    m: usize,
    k: usize,
    n: usize,
) {
    // for (i=0; i<M; i++) for (j=0; j<N; j++) {
    //   sum = C[i*N+j];
    //   for (kk=0; kk<K; kk++) sum += A[i*K+kk] * B[kk*N+j];
    //   C[i*N+j] = sum;
    // }
    for i in 0..m {
        for j in 0..n {
            let mut sum = unsafe { *c.add(i * n + j) };
            for kk in 0..k {
                sum += unsafe { *a.add(i * k + kk) } * unsafe { *b.add(kk * n + j) };
            }
            unsafe { *c.add(i * n + j) = sum; }
        }
    }
}

// ── Public entry point ─────────────────────────────────────────────────────
//
// `simd_gemm(C, A, B, M, K, N)` — C[M×N] += A[M×K] × B[K×N]
//
// Dispatches at compile time to the best available path:
//   avx512f  → avx512_impl::gemm
//   avx2     → avx2_impl::gemm
//   otherwise → scalar_gemm
//
// Safety: C, A, B must point to valid, non-overlapping f32 arrays of the
//         indicated sizes (M*N, M*K, K*N respectively).  No bounds checking
//         is performed.

/// C\[M×N\] += A\[M×K\] × B\[K×N\].  All three matrices are row-major,
/// C is accumulated into (not overwritten).
///
/// # Safety
/// Pointers must be valid for the indicated dimensions and must not alias.
pub unsafe fn simd_gemm(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    m: usize,
    k: usize,
    n: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        avx512_impl::gemm(c, a, b, m, k, n);
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
    {
        avx2_impl::gemm(c, a, b, m, k, n);
        return;
    }
    #[allow(unreachable_code)]
    scalar_gemm(c, a, b, m, k, n);
}

/// Safe wrapper around `simd_gemm` that accepts slices and validates dims.
///
/// Panics if slice lengths do not match M*K, K*N, and M*N respectively.
pub fn gemm_slices(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) {
    assert_eq!(a.len(), m * k, "A must have M*K elements");
    assert_eq!(b.len(), k * n, "B must have K*N elements");
    assert_eq!(c.len(), m * n, "C must have M*N elements");
    unsafe { simd_gemm(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), m, k, n) }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_gemm(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut s = c[i * n + j];
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
    }

    fn run_test(m: usize, k: usize, n: usize) {
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.01).collect();
        let mut c_ref: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.001).collect();
        let mut c_simd = c_ref.clone();

        ref_gemm(&mut c_ref, &a, &b, m, k, n);
        gemm_slices(&mut c_simd, &a, &b, m, k, n);

        for (i, (r, s)) in c_ref.iter().zip(c_simd.iter()).enumerate() {
            let diff = (r - s).abs();
            assert!(
                diff < 1e-3,
                "mismatch at [{i}]: ref={r} simd={s} diff={diff} (m={m} k={k} n={n})"
            );
        }
    }

    #[test]
    fn gemm_1x1x1() {
        run_test(1, 1, 1);
    }

    #[test]
    fn gemm_small() {
        run_test(4, 4, 4);
    }

    #[test]
    fn gemm_aligned_to_rm_rn() {
        // 6 rows, 16 cols (= 2*8 for AVX2 RN=2): hits the main tile path
        run_test(6, 8, 16);
    }

    #[test]
    fn gemm_tall() {
        // more rows than GEMM_RM to exercise both main and tail loops
        run_test(13, 32, 16);
    }

    #[test]
    fn gemm_wide() {
        // more cols than RN*EPR to exercise inner column tiling
        run_test(4, 8, 33);
    }

    #[test]
    fn gemm_odd_dims() {
        run_test(7, 5, 11);
    }
}
