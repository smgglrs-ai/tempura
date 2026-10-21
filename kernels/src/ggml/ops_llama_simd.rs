//! Direct Rust translation of the SIMD fast-paths from llama.cpp
//! ggml/src/ggml-cpu/ops.cpp that were skipped in the initial scalar
//! translation.
//!
//! Translated functions (all x86_64):
//!   rwkv_wkv6_f32_avx      -- ops.cpp line 10450 (#if AVX && !AVX512F)
//!   rwkv_wkv6_f32_avx512   -- ops.cpp line 10458 (#elif AVX512F)
//!   gla_f32_avx            -- ops.cpp line 10662 (#if AVX && !AVX512F)
//!   gla_f32_avx512         -- ops.cpp line 10670 (#elif AVX512F)
//!   rwkv_wkv7_f32_avx2     -- ops.cpp line 11496 (#else branch of GGML_SIMD
//!                             with GGML_F32_VEC=__m256, ARR=4, EPR=8, STEP=32)
//!   rwkv_wkv7_f32_avx512   -- ops.cpp line 11496, GGML_F32_VEC=__m512,
//!                             ARR=4, EPR=16, STEP=64
//!
//! Scalar dispatch wrappers at the bottom select the appropriate path at
//! runtime via `is_x86_feature_detected!`.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

// --- rwkv_wkv6 SIMD paths ----------------------------------------------------
//
// ops.cpp lines 10450-10601.
//
// The #if-AVX / #elif-AVX512F blocks share identical structure; only the
// vector width (8 vs 16 floats per register) differs.
//
// GGML_F32X_FMA(a, b, c) = _mm256_fmadd_ps(b, c, a) = a + b*c
// GGML_F32X_MUL(a, b)    = a * b
// GGML_F32X_SET1(x)       = broadcast x
// GGML_F32X_LOAD(p)       = unaligned load
// GGML_F32X_STORE(p, v)   = unaligned store

/// AVX (8-wide) inner-loop for `ggml_compute_forward_rwkv_wkv6_f32`.
/// ops.cpp line 10450.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub unsafe fn rwkv_wkv6_f32_avx(
    k_data:          *const f32,
    v_data:          *const f32,
    r_data:          *const f32,
    time_faaaa_data: *const f32,
    time_decay_data: *const f32,
    state_in_data:   *const f32,
    dst_data:        *mut f32,
    state_out:       *mut f32,
    t_cap: i64,
    c: i64,
    heads: i64,
    n_seqs: i64,
    h_start: usize,
    h_end: usize,
) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const VECTOR_SIZE: usize = 8;

    let t_stride    = c as usize;
    let h_stride    = (c / heads) as usize;
    let head_size   = h_stride;
    let h_stride_2d = head_size * head_size;
    let vec_count   = head_size / VECTOR_SIZE;
    let seq_len     = t_cap as usize / n_seqs as usize;

    for t in 0..t_cap as usize {
        let t_offset     = t * t_stride;
        let state_offset = head_size * c as usize * (t / seq_len);
        let state_cur    = state_out.add(state_offset);
        let state_prev: *const f32 = if t % seq_len != 0 {
            state_cur
        } else {
            state_in_data.add(state_offset)
        };

        for h in h_start..h_end {
            let h_offset    = h * h_stride;
            let t_h_offset  = t_offset + h_offset;
            let h_2d_offset = h * h_stride_2d;

            for i in 0..head_size {
                let t_h_i_offset  = t_h_offset + i;
                let h_i_offset    = h_offset + i;
                let h_2d_i_offset = h_2d_offset + i * h_stride;

                let k_val          = *k_data.add(t_h_i_offset);
                let r_val          = *r_data.add(t_h_i_offset);
                let time_faaaa_val = *time_faaaa_data.add(h_i_offset);
                let time_decay_val = *time_decay_data.add(t_h_i_offset);

                let k_vec          = _mm256_set1_ps(k_val);
                let r_vec          = _mm256_set1_ps(r_val);
                let time_faaaa_vec = _mm256_set1_ps(time_faaaa_val);
                let time_decay_vec = _mm256_set1_ps(time_decay_val);

                for j in 0..vec_count {
                    let base_j        = j * VECTOR_SIZE;
                    let t_h_j         = t_h_offset  + base_j;
                    let h_2d_i_j      = h_2d_i_offset + base_j;

                    let v_vec         = _mm256_loadu_ps(v_data.add(t_h_j));
                    let prev_state    = _mm256_loadu_ps(state_prev.add(h_2d_i_j));
                    let dst_vec       = _mm256_loadu_ps(dst_data.add(t_h_j));

                    // kv = v * k
                    let kv_vec  = _mm256_mul_ps(v_vec, k_vec);

                    // temp = kv * time_faaaa + prev_state  (FMA: prev_state + kv*time_faaaa)
                    let temp    = _mm256_fmadd_ps(kv_vec, time_faaaa_vec, prev_state);

                    // dst += temp * r  (FMA: dst + temp*r)
                    let dst_new = _mm256_fmadd_ps(temp, r_vec, dst_vec);
                    _mm256_storeu_ps(dst_data.add(t_h_j), dst_new);

                    // new_state = prev_state * time_decay + kv  (FMA: kv + prev_state*time_decay)
                    let new_state = _mm256_fmadd_ps(prev_state, time_decay_vec, kv_vec);
                    _mm256_storeu_ps(state_cur.add(h_2d_i_j), new_state);
                }

                // tail (head_size not a multiple of 8)
                for j in (vec_count * VECTOR_SIZE)..head_size {
                    let t_h_j    = t_h_offset  + j;
                    let h_2d_i_j = h_2d_i_offset + j;
                    let v_val    = *v_data.add(t_h_j);
                    let kv_val   = v_val * k_val;
                    let prev     = *state_prev.add(h_2d_i_j);
                    let temp     = kv_val * time_faaaa_val + prev;
                    *dst_data.add(t_h_j)   += temp * r_val;
                    *state_cur.add(h_2d_i_j) = prev * time_decay_val + kv_val;
                }
            }
        }
    }
}

/// AVX-512F (16-wide) inner-loop for `ggml_compute_forward_rwkv_wkv6_f32`.
/// ops.cpp line 10458.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn rwkv_wkv6_f32_avx512(
    k_data:          *const f32,
    v_data:          *const f32,
    r_data:          *const f32,
    time_faaaa_data: *const f32,
    time_decay_data: *const f32,
    state_in_data:   *const f32,
    dst_data:        *mut f32,
    state_out:       *mut f32,
    t_cap: i64,
    c: i64,
    heads: i64,
    n_seqs: i64,
    h_start: usize,
    h_end: usize,
) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const VECTOR_SIZE: usize = 16;

    let t_stride    = c as usize;
    let h_stride    = (c / heads) as usize;
    let head_size   = h_stride;
    let h_stride_2d = head_size * head_size;
    let vec_count   = head_size / VECTOR_SIZE;
    let seq_len     = t_cap as usize / n_seqs as usize;

    for t in 0..t_cap as usize {
        let t_offset     = t * t_stride;
        let state_offset = head_size * c as usize * (t / seq_len);
        let state_cur    = state_out.add(state_offset);
        let state_prev: *const f32 = if t % seq_len != 0 {
            state_cur
        } else {
            state_in_data.add(state_offset)
        };

        for h in h_start..h_end {
            let h_offset    = h * h_stride;
            let t_h_offset  = t_offset + h_offset;
            let h_2d_offset = h * h_stride_2d;

            for i in 0..head_size {
                let t_h_i_offset  = t_h_offset + i;
                let h_i_offset    = h_offset + i;
                let h_2d_i_offset = h_2d_offset + i * h_stride;

                let k_val          = *k_data.add(t_h_i_offset);
                let r_val          = *r_data.add(t_h_i_offset);
                let time_faaaa_val = *time_faaaa_data.add(h_i_offset);
                let time_decay_val = *time_decay_data.add(t_h_i_offset);

                let k_vec          = _mm512_set1_ps(k_val);
                let r_vec          = _mm512_set1_ps(r_val);
                let time_faaaa_vec = _mm512_set1_ps(time_faaaa_val);
                let time_decay_vec = _mm512_set1_ps(time_decay_val);

                for j in 0..vec_count {
                    let base_j   = j * VECTOR_SIZE;
                    let t_h_j    = t_h_offset  + base_j;
                    let h_2d_i_j = h_2d_i_offset + base_j;

                    let v_vec      = _mm512_loadu_ps(v_data.add(t_h_j));
                    let prev_state = _mm512_loadu_ps(state_prev.add(h_2d_i_j));
                    let dst_vec    = _mm512_loadu_ps(dst_data.add(t_h_j));

                    let kv_vec    = _mm512_mul_ps(v_vec, k_vec);
                    let temp      = _mm512_fmadd_ps(kv_vec, time_faaaa_vec, prev_state);
                    let dst_new   = _mm512_fmadd_ps(temp, r_vec, dst_vec);
                    _mm512_storeu_ps(dst_data.add(t_h_j), dst_new);

                    let new_state = _mm512_fmadd_ps(prev_state, time_decay_vec, kv_vec);
                    _mm512_storeu_ps(state_cur.add(h_2d_i_j), new_state);
                }

                for j in (vec_count * VECTOR_SIZE)..head_size {
                    let t_h_j    = t_h_offset  + j;
                    let h_2d_i_j = h_2d_i_offset + j;
                    let v_val    = *v_data.add(t_h_j);
                    let kv_val   = v_val * k_val;
                    let prev     = *state_prev.add(h_2d_i_j);
                    let temp     = kv_val * time_faaaa_val + prev;
                    *dst_data.add(t_h_j)    += temp * r_val;
                    *state_cur.add(h_2d_i_j) = prev * time_decay_val + kv_val;
                }
            }
        }
    }
}

// --- gla SIMD paths ----------------------------------------------------------
//
// ops.cpp lines 10662-10802.
//
// The GLA kernel is structurally identical to wkv6 except:
//   - inputs are k, v, q, g  (no time_faaaa / time_decay)
//   - gating:   temp = kv + prev_state * g  (not kv*time_faaaa + prev_state)
//   - output:   dst += temp * q             (not temp * r)
//   - next_state = temp                     (not prev_state*decay + kv)
//
// GGML_F32X_FMA(a,b,c) = a + b*c:
//   temp_vec = FMA(kv_vec, prev_state_vec, g_vec)
//            = kv + prev_state * g
//   dst_vec  = FMA(dst_vec, temp_vec, q_vec)
//            = dst + temp * q
// (state_cur = temp_vec, stored directly)

/// AVX (8-wide) inner-loop for `ggml_compute_forward_gla_f32`.
/// ops.cpp line 10662.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub unsafe fn gla_f32_avx(
    k_data:        *const f32,
    v_data:        *const f32,
    q_data:        *const f32,
    g_data:        *const f32,
    state_in_data: *const f32,
    dst_data:      *mut f32,
    state_out:     *mut f32,
    t_cap: i64,
    c: i64,
    heads: i64,
    n_seqs: i64,
    scale: f32,
    h_start: usize,
    h_end: usize,
) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const VECTOR_SIZE: usize = 8;

    let t_stride    = c as usize;
    let h_stride    = (c / heads) as usize;
    let head_size   = h_stride;
    let h_stride_2d = head_size * head_size;
    let vec_count   = head_size / VECTOR_SIZE;
    let seq_len     = t_cap as usize / n_seqs as usize;

    for t in 0..t_cap as usize {
        let t_offset     = t * t_stride;
        let state_offset = head_size * c as usize * (t / seq_len);
        let state_cur    = state_out.add(state_offset);
        let state_prev: *const f32 = if t % seq_len != 0 {
            state_cur
        } else {
            state_in_data.add(state_offset)
        };

        for h in h_start..h_end {
            let h_offset    = h * h_stride;
            let t_h_offset  = t_offset + h_offset;
            let h_2d_offset = h * h_stride_2d;

            for i in 0..head_size {
                let t_h_i_offset  = t_h_offset + i;
                let h_2d_i_offset = h_2d_offset + i * h_stride;

                let k_val = *k_data.add(t_h_i_offset);
                let q_val = *q_data.add(t_h_i_offset) * scale;
                let g_val = *g_data.add(t_h_i_offset);

                let k_vec = _mm256_set1_ps(k_val);
                let q_vec = _mm256_set1_ps(q_val);
                let g_vec = _mm256_set1_ps(g_val);

                for j in 0..vec_count {
                    let base_j   = j * VECTOR_SIZE;
                    let t_h_j    = t_h_offset  + base_j;
                    let h_2d_i_j = h_2d_i_offset + base_j;

                    let v_vec      = _mm256_loadu_ps(v_data.add(t_h_j));
                    let prev_state = _mm256_loadu_ps(state_prev.add(h_2d_i_j));
                    let dst_vec    = _mm256_loadu_ps(dst_data.add(t_h_j));

                    // kv = v * k
                    let kv_vec = _mm256_mul_ps(v_vec, k_vec);

                    // temp = kv + prev_state * g  (FMA: kv + prev_state*g)
                    let temp   = _mm256_fmadd_ps(prev_state, g_vec, kv_vec);

                    // dst += temp * q  (FMA: dst + temp*q)
                    let dst_new = _mm256_fmadd_ps(temp, q_vec, dst_vec);
                    _mm256_storeu_ps(dst_data.add(t_h_j), dst_new);

                    // state_cur = temp
                    _mm256_storeu_ps(state_cur.add(h_2d_i_j), temp);
                }

                for j in (vec_count * VECTOR_SIZE)..head_size {
                    let t_h_j    = t_h_offset  + j;
                    let h_2d_i_j = h_2d_i_offset + j;
                    let v_val    = *v_data.add(t_h_j);
                    let kv_val   = v_val * k_val;
                    let prev     = *state_prev.add(h_2d_i_j);
                    let temp     = kv_val + prev * g_val;
                    *dst_data.add(t_h_j)    += temp * q_val;
                    *state_cur.add(h_2d_i_j) = temp;
                }
            }
        }
    }
}

/// AVX-512F (16-wide) inner-loop for `ggml_compute_forward_gla_f32`.
/// ops.cpp line 10670.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gla_f32_avx512(
    k_data:        *const f32,
    v_data:        *const f32,
    q_data:        *const f32,
    g_data:        *const f32,
    state_in_data: *const f32,
    dst_data:      *mut f32,
    state_out:     *mut f32,
    t_cap: i64,
    c: i64,
    heads: i64,
    n_seqs: i64,
    scale: f32,
    h_start: usize,
    h_end: usize,
) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const VECTOR_SIZE: usize = 16;

    let t_stride    = c as usize;
    let h_stride    = (c / heads) as usize;
    let head_size   = h_stride;
    let h_stride_2d = head_size * head_size;
    let vec_count   = head_size / VECTOR_SIZE;
    let seq_len     = t_cap as usize / n_seqs as usize;

    for t in 0..t_cap as usize {
        let t_offset     = t * t_stride;
        let state_offset = head_size * c as usize * (t / seq_len);
        let state_cur    = state_out.add(state_offset);
        let state_prev: *const f32 = if t % seq_len != 0 {
            state_cur
        } else {
            state_in_data.add(state_offset)
        };

        for h in h_start..h_end {
            let h_offset    = h * h_stride;
            let t_h_offset  = t_offset + h_offset;
            let h_2d_offset = h * h_stride_2d;

            for i in 0..head_size {
                let t_h_i_offset  = t_h_offset + i;
                let h_2d_i_offset = h_2d_offset + i * h_stride;

                let k_val = *k_data.add(t_h_i_offset);
                let q_val = *q_data.add(t_h_i_offset) * scale;
                let g_val = *g_data.add(t_h_i_offset);

                let k_vec = _mm512_set1_ps(k_val);
                let q_vec = _mm512_set1_ps(q_val);
                let g_vec = _mm512_set1_ps(g_val);

                for j in 0..vec_count {
                    let base_j   = j * VECTOR_SIZE;
                    let t_h_j    = t_h_offset  + base_j;
                    let h_2d_i_j = h_2d_i_offset + base_j;

                    let v_vec      = _mm512_loadu_ps(v_data.add(t_h_j));
                    let prev_state = _mm512_loadu_ps(state_prev.add(h_2d_i_j));
                    let dst_vec    = _mm512_loadu_ps(dst_data.add(t_h_j));

                    let kv_vec  = _mm512_mul_ps(v_vec, k_vec);
                    let temp    = _mm512_fmadd_ps(prev_state, g_vec, kv_vec);
                    let dst_new = _mm512_fmadd_ps(temp, q_vec, dst_vec);
                    _mm512_storeu_ps(dst_data.add(t_h_j), dst_new);
                    _mm512_storeu_ps(state_cur.add(h_2d_i_j), temp);
                }

                for j in (vec_count * VECTOR_SIZE)..head_size {
                    let t_h_j    = t_h_offset  + j;
                    let h_2d_i_j = h_2d_i_offset + j;
                    let v_val    = *v_data.add(t_h_j);
                    let kv_val   = v_val * k_val;
                    let prev     = *state_prev.add(h_2d_i_j);
                    let temp     = kv_val + prev * g_val;
                    *dst_data.add(t_h_j)    += temp * q_val;
                    *state_cur.add(h_2d_i_j) = temp;
                }
            }
        }
    }
}

// --- rwkv_wkv7 SIMD paths ----------------------------------------------------
//
// ops.cpp lines 11496-11573  (#else branch when GGML_SIMD is defined and not
// SVE/RVV).
//
// The C++ code uses the generic GGML_F32_VEC abstraction.  We translate two
// concrete instantiations:
//
//   AVX2:    GGML_F32_VEC = __m256  (EPR=8, ARR=4, STEP=32)
//   AVX-512: GGML_F32_VEC = __m512  (EPR=16, ARR=4, STEP=64)
//
// GGML_F32x8_REDUCE(res, x[4]):
//   x[0] += x[2]; x[1] += x[3]; x[0] += x[1]
//   t0 = _mm_add_ps(lo128(x[0]), hi128(x[0]))
//   t1 = _mm_hadd_ps(t0, t0)
//   res = _mm_cvtss_f32(_mm_hadd_ps(t1, t1))
//
// GGML_F32_VEC_FMA(a, b, c) = a + b*c  (fmadd(b,c,a))

/// Horizontal sum of a single `__m256` register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn hsum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let t0 = _mm_add_ps(lo, hi);
    let t1 = _mm_hadd_ps(t0, t0);
    _mm_cvtss_f32(_mm_hadd_ps(t1, t1))
}

/// AVX2 (4x8-wide) path for `ggml_compute_forward_rwkv_wkv7_f32`.
/// ops.cpp line 11496.  Instantiates GGML_F32_VEC = __m256 with ARR=4,
/// EPR=8, STEP=32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub unsafe fn rwkv_wkv7_f32_avx2(
    r_data:          *const f32,
    w_data:          *const f32,
    k_data:          *const f32,
    v_data:          *const f32,
    a_data:          *const f32,
    b_data:          *const f32,
    state_in_data:   *const f32,
    dst_data:        *mut f32,
    state_out:       *mut f32,
    t_cap: i64,
    c: i64,
    heads: i64,
    n_seqs: i64,
    h_start: usize,
    h_end: usize,
) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // GGML_F32_ARR = 4, GGML_F32_EPR = 8, GGML_F32_STEP = 32
    const EPR: usize  = 8;
    const ARR: usize  = 4;
    const STEP: usize = ARR * EPR; // 32

    let t_stride    = c as usize;
    let h_stride    = (c / heads) as usize;
    let head_size   = h_stride;
    let h_stride_2d = head_size * head_size;
    let seq_len     = t_cap as usize / n_seqs as usize;

    for t in 0..t_cap as usize {
        let t_offset     = t * t_stride;
        let state_offset = head_size * c as usize * (t / seq_len);
        let state_cur    = state_out.add(state_offset);
        let state_prev: *const f32 = if t % seq_len != 0 {
            state_cur
        } else {
            state_in_data.add(state_offset)
        };

        for h in h_start..h_end {
            let h_offset    = h * h_stride;
            let t_h_offset  = t_offset + h_offset;
            let h_2d_offset = h * h_stride_2d;

            for ii in 0..head_size {
                let t_h_i_offset  = t_h_offset + ii;
                let h_2d_i_offset = h_2d_offset + ii * h_stride;

                let v_vec = _mm256_set1_ps(*v_data.add(t_h_i_offset));

                // Compute sa = dot(a[t_h..], state_prev[h_2d_i..])
                let mut sum: [__m256; ARR] = [_mm256_setzero_ps(); ARR];
                let mut j = 0usize;
                while j + STEP <= head_size {
                    for kk in 0..ARR {
                        let off = t_h_offset + j + kk * EPR;
                        let ax  = _mm256_loadu_ps(a_data.add(off));
                        let ay  = _mm256_loadu_ps(state_prev.add(h_2d_i_offset + j + kk * EPR));
                        sum[kk] = _mm256_fmadd_ps(ax, ay, sum[kk]);
                    }
                    j += STEP;
                }
                // GGML_F32x8_REDUCE(sa, sum)
                sum[0] = _mm256_add_ps(sum[0], sum[2]);
                sum[1] = _mm256_add_ps(sum[1], sum[3]);
                sum[0] = _mm256_add_ps(sum[0], sum[1]);
                let mut sa = hsum_avx(sum[0]);

                // scalar tail for sa
                for jj in j..head_size {
                    sa += *a_data.add(t_h_offset + jj) * *state_prev.add(h_2d_i_offset + jj);
                }

                let sa_vec = _mm256_set1_ps(sa);

                // Main update loop
                let mut result: [__m256; ARR] = [_mm256_setzero_ps(); ARR];
                j = 0;
                while j + STEP <= head_size {
                    for kk in 0..ARR {
                        let t_h_j    = t_h_offset    + j + kk * EPR;
                        let h_2d_i_j = h_2d_i_offset + j + kk * EPR;

                        let r_vec = _mm256_loadu_ps(r_data.add(t_h_j));
                        let w_vec = _mm256_loadu_ps(w_data.add(t_h_j));
                        // k_vec reused for kv
                        let kv    = _mm256_mul_ps(v_vec, _mm256_loadu_ps(k_data.add(t_h_j)));
                        let b_vec = _mm256_loadu_ps(b_data.add(t_h_j));

                        let s_prev = _mm256_loadu_ps(state_prev.add(h_2d_i_j));
                        // state = kv + s_prev * w
                        let s = _mm256_fmadd_ps(s_prev, w_vec, kv);
                        // state += sa * b
                        let s = _mm256_fmadd_ps(sa_vec, b_vec, s);
                        _mm256_storeu_ps(state_cur.add(h_2d_i_j), s);

                        // result += state * r
                        result[kk] = _mm256_fmadd_ps(s, r_vec, result[kk]);
                    }
                    j += STEP;
                }
                // GGML_F32x8_REDUCE(dst_data[t_h_i], result)
                result[0] = _mm256_add_ps(result[0], result[2]);
                result[1] = _mm256_add_ps(result[1], result[3]);
                result[0] = _mm256_add_ps(result[0], result[1]);
                let mut dst_val = hsum_avx(result[0]);

                // scalar tail
                for jj in j..head_size {
                    let t_h_j    = t_h_offset    + jj;
                    let h_2d_i_j = h_2d_i_offset + jj;
                    let kv_val   = *v_data.add(t_h_i_offset) * *k_data.add(t_h_j);
                    let prev     = *state_prev.add(h_2d_i_j);
                    let s        = prev * *w_data.add(t_h_j) + kv_val + sa * *b_data.add(t_h_j);
                    *state_cur.add(h_2d_i_j) = s;
                    dst_val += s * *r_data.add(t_h_j);
                }

                *dst_data.add(t_h_i_offset) = dst_val;
            }
        }
    }
}

/// Horizontal sum of a single `__m512` register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn hsum_avx512(v: std::arch::x86_64::__m512) -> f32 {
    use std::arch::x86_64::*;
    _mm512_reduce_add_ps(v)
}

/// AVX-512F (4x16-wide) path for `ggml_compute_forward_rwkv_wkv7_f32`.
/// Same structure as `rwkv_wkv7_f32_avx2` but with 512-bit registers.
/// ARR=4, EPR=16, STEP=64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn rwkv_wkv7_f32_avx512(
    r_data:          *const f32,
    w_data:          *const f32,
    k_data:          *const f32,
    v_data:          *const f32,
    a_data:          *const f32,
    b_data:          *const f32,
    state_in_data:   *const f32,
    dst_data:        *mut f32,
    state_out:       *mut f32,
    t_cap: i64,
    c: i64,
    heads: i64,
    n_seqs: i64,
    h_start: usize,
    h_end: usize,
) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const EPR: usize  = 16;
    const ARR: usize  = 4;
    const STEP: usize = ARR * EPR; // 64

    let t_stride    = c as usize;
    let h_stride    = (c / heads) as usize;
    let head_size   = h_stride;
    let h_stride_2d = head_size * head_size;
    let seq_len     = t_cap as usize / n_seqs as usize;

    for t in 0..t_cap as usize {
        let t_offset     = t * t_stride;
        let state_offset = head_size * c as usize * (t / seq_len);
        let state_cur    = state_out.add(state_offset);
        let state_prev: *const f32 = if t % seq_len != 0 {
            state_cur
        } else {
            state_in_data.add(state_offset)
        };

        for h in h_start..h_end {
            let h_offset    = h * h_stride;
            let t_h_offset  = t_offset + h_offset;
            let h_2d_offset = h * h_stride_2d;

            for ii in 0..head_size {
                let t_h_i_offset  = t_h_offset + ii;
                let h_2d_i_offset = h_2d_offset + ii * h_stride;

                let v_vec = _mm512_set1_ps(*v_data.add(t_h_i_offset));

                let mut sum: [__m512; ARR] = [_mm512_setzero_ps(); ARR];
                let mut j = 0usize;
                while j + STEP <= head_size {
                    for kk in 0..ARR {
                        let off = t_h_offset + j + kk * EPR;
                        let ax  = _mm512_loadu_ps(a_data.add(off));
                        let ay  = _mm512_loadu_ps(state_prev.add(h_2d_i_offset + j + kk * EPR));
                        sum[kk] = _mm512_fmadd_ps(ax, ay, sum[kk]);
                    }
                    j += STEP;
                }
                sum[0] = _mm512_add_ps(sum[0], sum[2]);
                sum[1] = _mm512_add_ps(sum[1], sum[3]);
                sum[0] = _mm512_add_ps(sum[0], sum[1]);
                let mut sa = hsum_avx512(sum[0]);

                for jj in j..head_size {
                    sa += *a_data.add(t_h_offset + jj) * *state_prev.add(h_2d_i_offset + jj);
                }

                let sa_vec = _mm512_set1_ps(sa);

                let mut result: [__m512; ARR] = [_mm512_setzero_ps(); ARR];
                j = 0;
                while j + STEP <= head_size {
                    for kk in 0..ARR {
                        let t_h_j    = t_h_offset    + j + kk * EPR;
                        let h_2d_i_j = h_2d_i_offset + j + kk * EPR;

                        let r_vec  = _mm512_loadu_ps(r_data.add(t_h_j));
                        let w_vec  = _mm512_loadu_ps(w_data.add(t_h_j));
                        let kv     = _mm512_mul_ps(v_vec, _mm512_loadu_ps(k_data.add(t_h_j)));
                        let b_vec  = _mm512_loadu_ps(b_data.add(t_h_j));
                        let s_prev = _mm512_loadu_ps(state_prev.add(h_2d_i_j));

                        let s = _mm512_fmadd_ps(s_prev, w_vec, kv);
                        let s = _mm512_fmadd_ps(sa_vec, b_vec, s);
                        _mm512_storeu_ps(state_cur.add(h_2d_i_j), s);

                        result[kk] = _mm512_fmadd_ps(s, r_vec, result[kk]);
                    }
                    j += STEP;
                }
                result[0] = _mm512_add_ps(result[0], result[2]);
                result[1] = _mm512_add_ps(result[1], result[3]);
                result[0] = _mm512_add_ps(result[0], result[1]);
                let mut dst_val = hsum_avx512(result[0]);

                for jj in j..head_size {
                    let t_h_j    = t_h_offset    + jj;
                    let h_2d_i_j = h_2d_i_offset + jj;
                    let kv_val   = *v_data.add(t_h_i_offset) * *k_data.add(t_h_j);
                    let prev     = *state_prev.add(h_2d_i_j);
                    let s        = prev * *w_data.add(t_h_j) + kv_val + sa * *b_data.add(t_h_j);
                    *state_cur.add(h_2d_i_j) = s;
                    dst_val += s * *r_data.add(t_h_j);
                }

                *dst_data.add(t_h_i_offset) = dst_val;
            }
        }
    }
}

// --- Runtime dispatch --------------------------------------------------------
//
// These wrappers are called from the scalar dispatch in ops_llama4.rs.  They
// select the widest SIMD path available at runtime and fall back to the scalar
// implementation (already in ops_llama4.rs) if no SIMD is present.

/// Runtime-dispatching wrapper for the rwkv_wkv6 inner loop.
///
/// Callers pass the same arguments as the scalar path in `ops_llama4.rs`.
/// The dispatch is done here rather than in the tensor-level function because
/// the `params`/`dst` ggml-tensor types are not yet mirrored in Rust.
#[cfg(target_arch = "x86_64")]
pub unsafe fn rwkv_wkv6_f32_dispatch(
    k_data: *const f32, v_data: *const f32, r_data: *const f32,
    time_faaaa_data: *const f32, time_decay_data: *const f32,
    state_in_data: *const f32, dst_data: *mut f32, state_out: *mut f32,
    t_cap: i64, c: i64, heads: i64, n_seqs: i64,
    h_start: usize, h_end: usize,
) {
    if is_x86_feature_detected!("avx512f") {
        rwkv_wkv6_f32_avx512(
            k_data, v_data, r_data, time_faaaa_data, time_decay_data,
            state_in_data, dst_data, state_out,
            t_cap, c, heads, n_seqs, h_start, h_end,
        );
    } else if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        rwkv_wkv6_f32_avx(
            k_data, v_data, r_data, time_faaaa_data, time_decay_data,
            state_in_data, dst_data, state_out,
            t_cap, c, heads, n_seqs, h_start, h_end,
        );
    } else {
        super::ops_llama4::compute_forward_rwkv_wkv6_f32(
            k_data, v_data, r_data, time_faaaa_data, time_decay_data,
            state_in_data, dst_data, state_out,
            t_cap, c, heads, n_seqs, h_start, h_end,
        );
    }
}

/// Runtime-dispatching wrapper for the gla inner loop.
#[cfg(target_arch = "x86_64")]
pub unsafe fn gla_f32_dispatch(
    k_data: *const f32, v_data: *const f32, q_data: *const f32, g_data: *const f32,
    state_in_data: *const f32, dst_data: *mut f32, state_out: *mut f32,
    t_cap: i64, c: i64, heads: i64, n_seqs: i64, scale: f32,
    h_start: usize, h_end: usize,
) {
    if is_x86_feature_detected!("avx512f") {
        gla_f32_avx512(
            k_data, v_data, q_data, g_data, state_in_data, dst_data, state_out,
            t_cap, c, heads, n_seqs, scale, h_start, h_end,
        );
    } else if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        gla_f32_avx(
            k_data, v_data, q_data, g_data, state_in_data, dst_data, state_out,
            t_cap, c, heads, n_seqs, scale, h_start, h_end,
        );
    } else {
        super::ops_llama4::compute_forward_gla_f32(
            k_data, v_data, q_data, g_data, state_in_data, dst_data, state_out,
            t_cap, c, heads, n_seqs, scale, h_start, h_end,
        );
    }
}

/// Runtime-dispatching wrapper for the rwkv_wkv7 inner loop.
#[cfg(target_arch = "x86_64")]
pub unsafe fn rwkv_wkv7_f32_dispatch(
    r_data: *const f32, w_data: *const f32, k_data: *const f32, v_data: *const f32,
    a_data: *const f32, b_data: *const f32,
    state_in_data: *const f32, dst_data: *mut f32, state_out: *mut f32,
    t_cap: i64, c: i64, heads: i64, n_seqs: i64,
    h_start: usize, h_end: usize,
) {
    if is_x86_feature_detected!("avx512f") {
        rwkv_wkv7_f32_avx512(
            r_data, w_data, k_data, v_data, a_data, b_data,
            state_in_data, dst_data, state_out,
            t_cap, c, heads, n_seqs, h_start, h_end,
        );
    } else if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        rwkv_wkv7_f32_avx2(
            r_data, w_data, k_data, v_data, a_data, b_data,
            state_in_data, dst_data, state_out,
            t_cap, c, heads, n_seqs, h_start, h_end,
        );
    } else {
        super::ops_llama4::compute_forward_rwkv_wkv7_f32(
            r_data, w_data, k_data, v_data, a_data, b_data,
            state_in_data, dst_data, state_out,
            t_cap, c, heads, n_seqs, h_start, h_end,
        );
    }
}
