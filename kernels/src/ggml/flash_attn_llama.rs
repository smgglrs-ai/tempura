//! Faithful translation of llama.cpp `ggml_compute_forward_flash_attn_ext_f16`
//! and helpers from `ggml/src/ggml-cpu/ops.cpp`.
//!
//! Source functions:
//!   - `ggml_compute_forward_flash_attn_ext_f16_one_chunk` (~line 8614)
//!   - `ggml_compute_forward_flash_attn_ext_tiled`         (~line 8852)
//!   - `ggml_flash_attn_ext_reduce_partials`               (~line 9142)
//!   - `ggml_compute_forward_flash_attn_ext_f16`           (~line 9212)
//!
//! All tensors are passed as flat slices with explicit dimension params.
//! q is always f32; k/v are f16 (as &[u16]); mask is f16 (as &[u16]).
//! Scratch memory that the C code pulls from `params->wdata` is allocated
//! as local Vecs per call -- correct behaviour, not as fast as the pooled
//! wdata approach but faithfully equivalent.

use crate::ggml::helpers::f16_to_f32;

// Tile sizes matching GGML_FA_TILE_Q / GGML_FA_TILE_KV in common.h
const Q_TILE_SZ: usize = 64;
const KV_TILE_SZ: usize = 64;

// -- helpers ------------------------------------------------------------------

#[inline(always)]
fn vec_scale_f32(v: &mut [f32], s: f32) {
    for x in v { *x *= s; }
}

#[inline(always)]
fn vec_mad_f32(acc: &mut [f32], src: &[f32], vs: f32) {
    for (a, &b) in acc.iter_mut().zip(src) { *a += b * vs; }
}

/// Scalar GEMM: C[mxn] += A[mxk] x B[kxn]
/// B is stored transposed as B_T[kxn] (i.e. B[dk][kv] in the K packing).
fn scalar_gemm_acc(c: &mut [f32], a: &[f32], b_t: &[f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b_t[p * n + j];
            }
        }
    }
}

fn vec_max_f32(v: &[f32]) -> f32 {
    v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

/// Computes softmax in-place: v[i] = exp(v[i] - max_val), returns sum.
fn vec_soft_max_f32(v: &mut [f32], max_val: f32) -> f32 {
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    sum
}

// -- slope computation (ALiBi / max_bias) -------------------------------------

fn head_slope(h: usize, n_head: usize, max_bias: f32) -> f32 {
    if max_bias <= 0.0 {
        return 1.0;
    }
    let n_head_log2 = 1usize << (n_head as f32).log2().floor() as u32;
    let m0 = f32::powf(2.0, -max_bias / n_head_log2 as f32);
    let m1 = f32::powf(2.0, -(max_bias / 2.0) / n_head_log2 as f32);
    if h < n_head_log2 {
        f32::powf(m0, (h + 1) as f32)
    } else {
        f32::powf(m1, (2 * (h - n_head_log2) + 1) as f32)
    }
}

// -- ggml_compute_forward_flash_attn_ext_f16_one_chunk ------------------------
//
// Translated from ops.cpp ~8614. Processes query rows ir0..ir1, KV positions
// ic_start..ic_end.
//
// Shape conventions (all contiguous):
//   q:    [neq3, neq2, neq1, dk]  -- neq3=batch, neq2=n_head, neq1=n_tokens
//   k:    [nek3, nek2, nek1, dk]  (f16 as u16)
//   v:    [nev3, nev2, nek1, dv]  (f16 as u16)
//   mask: [neq1, nek1]            (f16 as u16, optional; neq1 rows)
//   dst:  [neq3, neq2, neq1, dv]  -- output (contiguous, f32)
//
// ir = flat index over neq3 x neq2 x neq1.
//
// partials: if Some, write [M, S, VKQ[dv]] per ir into that buffer with the
// given stride (in f32 elements); otherwise write normalized output to dst.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_ext_f16_one_chunk(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    mask: Option<&[u16]>,
    sinks: Option<&[f32]>,
    dst: &mut [f32],
    // dimensions
    neq1: usize, // n_tokens (N)
    neq2: usize, // n_head (query heads)
    neq3: usize, // batch (usually 1)
    nek1: usize, // n_kv
    nek2: usize, // n_kv_head
    nek3: usize, // kv batch
    dv: usize,   // value head dim
    dk: usize,   // key/query head dim
    // op params
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
    // row range
    ir0: usize,
    ir1: usize,
    // kv range
    ic_start: usize,
    ic_end: usize,
    // partial output (None -> write directly to dst)
    partials: Option<&mut [f32]>,
    partial_stride: usize,
) {
    let write_partials = partials.is_some();
    let rk2 = neq2 / nek2;
    let rk3 = neq3 / nek3;
    let rv2 = neq2 / (neq2 / nek2).max(1); // nev2 == nek2 assumed
    let rv3 = neq3 / nek3;

    let effective_scale = if logit_softcap != 0.0 { scale / logit_softcap } else { scale };

    // Scratch buffers (C: from wdata; here: stack/heap per call)
    let mut vkq32 = vec![0.0f32; dv];
    let mut partials_mut = partials;

    for ir in ir0..ir1 {
        // q indices
        let iq3 = ir / (neq2 * neq1);
        let iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        let iq1 = ir - iq3 * neq2 * neq1 - iq2 * neq1;

        let h = iq2;
        let slope = head_slope(h, neq2, max_bias);

        let mut s_sum = 0.0f32;
        let mut m_max = f32::NEG_INFINITY;

        vkq32.fill(0.0);

        // q row for (iq1, iq2, iq3)
        let q_off = (iq3 * neq2 * neq1 + iq2 * neq1 + iq1) * dk;
        let pq = &q[q_off..q_off + dk];

        // k/v broadcast heads
        let ik2 = iq2 / rk2;
        let ik3 = iq3 / rk3;
        let iv2 = iq2 / rv2;
        let iv3 = iq3 / rv3;

        for ic in ic_start..ic_end {
            // mask value
            let mv = if let Some(mp) = mask {
                let mask_off = iq1 * nek1 + ic;
                let raw = mp[mask_off];
                slope * f16_to_f32(raw)
            } else {
                0.0
            };
            if mv == f32::NEG_INFINITY {
                continue;
            }

            // KQ = dot(Q_q, K[ic])
            let k_off = (ik3 * nek2 * nek1 + ik2 * nek1 + ic) * dk;
            let mut s = 0.0f32;
            for d in 0..dk {
                s += pq[d] * f16_to_f32(k[k_off + d]);
            }
            s *= effective_scale;

            if logit_softcap != 0.0 {
                s = logit_softcap * s.tanh();
            }
            s += mv;

            let m_old = m_max;
            let ms;
            let vs;

            if s > m_max {
                m_max = s;
                ms = (m_old - m_max).exp();
                vs = 1.0;
                vec_scale_f32(&mut vkq32, ms);
            } else {
                ms = 1.0;
                vs = (s - m_max).exp();
            }

            // VKQ += V[ic] * vs
            let v_off = (iv3 * nek2 * nek1 + iv2 * nek1 + ic) * dv;
            for d in 0..dv {
                vkq32[d] += f16_to_f32(v[v_off + d]) * vs;
            }

            s_sum = s_sum * ms + vs;
        }

        // sinks (apply only on first kv-chunk)
        if let Some(sinks_data) = sinks {
            if ic_start == 0 {
                let s = sinks_data[h];
                if s > m_max {
                    let ms = (m_max - s).exp();
                    m_max = s;
                    vec_scale_f32(&mut vkq32, ms);
                    s_sum = s_sum * ms + 1.0;
                } else {
                    let vs = (s - m_max).exp();
                    s_sum += vs;
                }
            }
        }

        if write_partials {
            if let Some(ref mut p) = partials_mut {
                let base = ir * partial_stride;
                p[base] = m_max;
                p[base + 1] = s_sum;
                p[base + 2..base + 2 + dv].copy_from_slice(&vkq32);
            }
        } else {
            // normalize
            let s_inv = if s_sum == 0.0 { 0.0 } else { 1.0 / s_sum };
            vec_scale_f32(&mut vkq32, s_inv);

            // dst layout: permute(0,2,1,3) -> [iq3, iq2, iq1] -> offset = (iq3*neq2*neq1 + iq2 + iq1*neq1)*dv
            // (matching nb1 = dv*sizeof(f32), row = i2 + i1*ne1 where ne1 = neq2)
            let dst_off = (iq3 * neq2 * neq1 + iq2 + iq1 * neq2) * dv;
            dst[dst_off..dst_off + dv].copy_from_slice(&vkq32);
        }
    }
}

// -- ggml_compute_forward_flash_attn_ext_tiled --------------------------------
//
// Translated from ops.cpp ~8852. Tiled version using Q_TILE_SZ x KV_TILE_SZ
// blocks and a scalar GEMM kernel (replaces simd_gemm). Operates on all KV
// positions (no ic_start/ic_end split -- the caller provides ir0..ir1).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_ext_tiled(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    mask: Option<&[u16]>,
    sinks: Option<&[f32]>,
    dst: &mut [f32],
    neq1: usize,
    neq2: usize,
    neq3: usize,
    nek1: usize,
    nek2: usize,
    nek3: usize,
    dv: usize,
    dk: usize,
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
    ir0: usize,
    ir1: usize,
) {
    let rk2 = neq2 / nek2;
    let rk3 = neq3 / nek3;
    let rv2 = rk2;
    let rv3 = rk3;

    let effective_scale = if logit_softcap != 0.0 { scale / logit_softcap } else { scale };

    // Scratch buffers (C: from params->wdata)
    let mut q_tile = vec![0.0f32; Q_TILE_SZ * dk];
    let mut kq    = vec![0.0f32; Q_TILE_SZ * KV_TILE_SZ];
    let mut mask32 = vec![0.0f32; Q_TILE_SZ * KV_TILE_SZ];
    let mut vkq32 = vec![0.0f32; Q_TILE_SZ * dv];
    let mut v32   = vec![0.0f32; KV_TILE_SZ * dv];
    let mut k_f32 = vec![0.0f32; KV_TILE_SZ * dk];

    let mut ir = ir0;
    while ir < ir1 {
        let iq3 = ir / (neq2 * neq1);
        let iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        let iq1 = ir - iq3 * neq2 * neq1 - iq2 * neq1;

        let tile_rows = Q_TILE_SZ.min((ir1 - ir).min(neq1 - iq1));
        assert!(tile_rows > 0);

        let h = iq2;
        let slope = head_slope(h, neq2, max_bias);

        let mut s_arr = [0.0f32; Q_TILE_SZ];
        let mut m_arr = [f32::NEG_INFINITY; Q_TILE_SZ];

        // Load Q tile
        for tq in 0..Q_TILE_SZ {
            if tq < tile_rows {
                let q_off = (iq3 * neq2 * neq1 + iq2 * neq1 + iq1 + tq) * dk;
                q_tile[tq * dk..(tq + 1) * dk].copy_from_slice(&q[q_off..q_off + dk]);
            } else {
                q_tile[tq * dk..(tq + 1) * dk].fill(0.0);
            }
        }

        vkq32.fill(0.0);

        let ik2 = iq2 / rk2;
        let ik3 = iq3 / rk3;
        let iv2 = iq2 / rv2;
        let iv3 = iq3 / rv3;

        let mut ic = 0usize;
        while ic < nek1 {
            let kv_tile = KV_TILE_SZ.min(nek1 - ic);

            // load and check mask
            if let Some(mp) = mask {
                mask32.fill(f32::NEG_INFINITY);
                let mut can_skip = true;
                for tq in 0..tile_rows {
                    for tk in 0..kv_tile {
                        let raw = mp[(iq1 + tq) * nek1 + ic + tk];
                        let mv = slope * f16_to_f32(raw);
                        mask32[tq * KV_TILE_SZ + tk] = mv;
                        if mv != f32::NEG_INFINITY { can_skip = false; }
                    }
                }
                if can_skip { ic += KV_TILE_SZ; continue; }
            }

            // Pack K tile transposed: K_f32[dk][kv_tile]
            k_f32.fill(0.0);
            for tk in 0..kv_tile {
                let k_off = (ik3 * nek2 * nek1 + ik2 * nek1 + ic + tk) * dk;
                for dk_i in 0..dk {
                    k_f32[dk_i * KV_TILE_SZ + tk] = f16_to_f32(k[k_off + dk_i]);
                }
            }

            // KQ[Q_TILE_SZ][KV_TILE_SZ] = Q_tile[Q_TILE_SZ][dk] x K_f32[dk][KV_TILE_SZ]
            kq.fill(0.0);
            scalar_gemm_acc(&mut kq, &q_tile, &k_f32, Q_TILE_SZ, dk, KV_TILE_SZ);
            vec_scale_f32(&mut kq[..Q_TILE_SZ * KV_TILE_SZ], effective_scale);

            // Pad KQ beyond kv_tile with -inf
            if kv_tile < KV_TILE_SZ {
                for tq in 0..Q_TILE_SZ {
                    for tk in kv_tile..KV_TILE_SZ {
                        kq[tq * KV_TILE_SZ + tk] = f32::NEG_INFINITY;
                    }
                }
            }

            if logit_softcap != 0.0 {
                for x in kq.iter_mut() { *x = logit_softcap * x.tanh(); }
            }

            if mask.is_some() {
                for i in 0..tile_rows * KV_TILE_SZ {
                    kq[i] += mask32[i];
                }
            }

            let mut skip = [false; Q_TILE_SZ];
            for tq in 0..Q_TILE_SZ {
                let kq_row = &mut kq[tq * KV_TILE_SZ..(tq + 1) * KV_TILE_SZ];
                let tile_max = vec_max_f32(kq_row);
                if tile_max == f32::NEG_INFINITY {
                    skip[tq] = true;
                    continue;
                }
                let m_old = m_arr[tq];
                let m_new = m_old.max(tile_max);
                if m_new > m_old {
                    let ms = (m_old - m_new).exp();
                    let base = tq * dv;
                    vec_scale_f32(&mut vkq32[base..base + dv], ms);
                    s_arr[tq] *= ms;
                }
                m_arr[tq] = m_new;
                s_arr[tq] += vec_soft_max_f32(kq_row, m_new);
            }

            // Pack V tile
            v32.fill(0.0);
            for tk in 0..kv_tile {
                let v_off = (iv3 * nek2 * nek1 + iv2 * nek1 + ic + tk) * dv;
                for d in 0..dv {
                    v32[tk * dv + d] = f16_to_f32(v[v_off + d]);
                }
            }

            // Zero out skipped KQ rows so they don't contribute
            for tq in 0..Q_TILE_SZ {
                if skip[tq] {
                    kq[tq * KV_TILE_SZ..(tq + 1) * KV_TILE_SZ].fill(0.0);
                }
            }

            // VKQ32 += KQ[Q_TILE_SZ][KV_TILE_SZ] x V32[KV_TILE_SZ][dv]
            // V32 stored as [KV_TILE_SZ][dv] (row-major), need to transpose K dim
            // scalar_gemm_acc expects b_t[k][n], here k=KV_TILE_SZ, n=dv -> V32 already [tk][dv]
            // but scalar_gemm_acc uses b_t[p][j] -> b_t[KV_TILE_SZ][dv], which matches V32
            scalar_gemm_acc(&mut vkq32, &kq, &v32, Q_TILE_SZ, KV_TILE_SZ, dv);

            ic += KV_TILE_SZ;
        }

        // sinks
        if let Some(sinks_data) = sinks {
            let s = sinks_data[h];
            for tq in 0..tile_rows {
                let base = tq * dv;
                if s > m_arr[tq] {
                    let ms = (m_arr[tq] - s).exp();
                    vec_scale_f32(&mut vkq32[base..base + dv], ms);
                    s_arr[tq] = s_arr[tq] * ms + 1.0;
                } else {
                    let vs = (s - m_arr[tq]).exp();
                    s_arr[tq] += vs;
                }
            }
        }

        for tq in 0..tile_rows {
            let s_inv = if s_arr[tq] == 0.0 { 0.0 } else { 1.0 / s_arr[tq] };
            let base = tq * dv;
            vec_scale_f32(&mut vkq32[base..base + dv], s_inv);

            let i1 = iq1 + tq;
            let dst_off = (iq3 * neq2 * neq1 + iq2 + i1 * neq2) * dv;
            dst[dst_off..dst_off + dv].copy_from_slice(&vkq32[base..base + dv]);
        }

        ir += tile_rows;
    }
}

// -- ggml_flash_attn_ext_reduce_partials --------------------------------------
//
// Translated from ops.cpp ~9142.
// Combines partial [M, S, VKQ[dv]] results written by different KV-chunk
// threads into a final normalized output.
//
// partials layout: [n_q_heads * n_chunks][2 + dv]  (row = q_head * n_chunks + chunk_idx)
// Each thread (ith/nth) owns a subset of q_heads.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_ext_reduce_partials(
    partials: &[f32],   // [n_q_heads * n_chunks, 2 + dv]
    dst: &mut [f32],
    n_q_heads: usize,
    n_chunks: usize,
    n_kv: usize,
    chunk_size: usize,
    dv: usize,
    neq1: usize,  // n_tokens (N) -- for dst layout
    neq2: usize,  // n_head
    ith: usize,
    nth: usize,
) {
    let partial_size = 2 + dv;
    let mut vkq_final = vec![0.0f32; dv];

    let mut q_head = ith;
    while q_head < n_q_heads {
        let mut m_final = f32::NEG_INFINITY;
        let mut s_final = 0.0f32;
        vkq_final.fill(0.0);

        for chunk_idx in 0..n_chunks {
            let ic_start = chunk_idx * chunk_size;
            if ic_start >= n_kv { continue; }

            let base = (q_head * n_chunks + chunk_idx) * partial_size;
            let m_chunk = partials[base];
            let s_chunk = partials[base + 1];
            let vkq_chunk = &partials[base + 2..base + 2 + dv];

            if s_chunk == 0.0 { continue; }

            let m_new = m_final.max(m_chunk);
            let scale_old = (m_final - m_new).exp();
            let scale_new = (m_chunk - m_new).exp();

            for d in 0..dv {
                vkq_final[d] = vkq_final[d] * scale_old + vkq_chunk[d] * scale_new;
            }
            s_final = s_final * scale_old + s_chunk * scale_new;
            m_final = m_new;
        }

        if s_final != 0.0 {
            let s_inv = 1.0 / s_final;
            vec_scale_f32(&mut vkq_final, s_inv);
        }

        // iq1=0, iq3=0 for decode (seqlen=1)
        let dst_off = (q_head) * dv; // ne1=neq2 used in permuted layout: (0*neq2*neq1 + q_head + 0*neq2)*dv
        dst[dst_off..dst_off + dv].copy_from_slice(&vkq_final);

        q_head += nth;
    }
}

// -- ggml_compute_forward_flash_attn_ext_f16 (main dispatch) ------------------
//
// Translated from ops.cpp ~9212.
//
// For decode (neq1==1, neq3==1, nek1>=512, k/v same f16 type):
//   Split KV across threads -> each thread processes one chunk -> barrier -> reduce.
// Otherwise:
//   Work-steal query rows, use tiled or one-chunk path per row range.
//
// In the Rust translation the caller is responsible for parallelism.
// This function processes the slice assigned to thread `ith` of `nth`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_ext_f16(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    mask: Option<&[u16]>,
    sinks: Option<&[f32]>,
    dst: &mut [f32],
    neq1: usize,
    neq2: usize,
    neq3: usize,
    nek1: usize,
    nek2: usize,
    nek3: usize,
    dv: usize,
    dk: usize,
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
    partials: Option<&mut [f32]>,
    partial_stride: usize,
    ir0: usize,
    ir1: usize,
    ic_start: usize,
    ic_end: usize,
) {
    // use_tiled: q is F32, k==v type (both f16), neq1 >= Q_TILE_SZ, no KV chunking
    let use_tiled = partials.is_none() && neq1 >= Q_TILE_SZ && ic_start == 0 && ic_end == nek1;

    if use_tiled {
        flash_attn_ext_tiled(
            q, k, v, mask, sinks, dst,
            neq1, neq2, neq3, nek1, nek2, nek3,
            dv, dk, scale, max_bias, logit_softcap,
            ir0, ir1,
        );
    } else {
        flash_attn_ext_f16_one_chunk(
            q, k, v, mask, sinks, dst,
            neq1, neq2, neq3, nek1, nek2, nek3,
            dv, dk, scale, max_bias, logit_softcap,
            ir0, ir1, ic_start, ic_end,
            partials, partial_stride,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Smoke-test: single head, single token, 4-dim k/v, 2 kv positions.
    // Verified numerically against reference flash attention formula.
    #[test]
    fn flash_attn_one_chunk_smoke() {
        let dk = 4usize;
        let dv = 4usize;
        let neq1 = 1; // n_tokens
        let neq2 = 1; // n_head
        let neq3 = 1;
        let nek1 = 2; // n_kv
        let nek2 = 1;
        let nek3 = 1;

        // Q = [1, 0, 0, 0]
        let q = vec![1.0f32, 0.0, 0.0, 0.0];

        // K row 0 = [1,0,0,0] (f16), K row 1 = [0,1,0,0]
        let k: Vec<u16> = {
            let mut v = vec![0u16; nek1 * dk];
            fn f32_to_f16_raw(x: f32) -> u16 {
                half::f16::from_f32(x).to_bits()
            }
            v[0] = f32_to_f16_raw(1.0);
            v[4 + 1] = f32_to_f16_raw(1.0);
            v
        };

        // V row 0 = [1,0,0,0] (f16), V row 1 = [0,1,0,0]
        let v: Vec<u16> = k.clone();

        let mut dst = vec![0.0f32; neq2 * neq1 * dv];
        let scale = 1.0 / (dk as f32).sqrt();

        flash_attn_ext_f16_one_chunk(
            &q, &k, &v, None, None, &mut dst,
            neq1, neq2, neq3, nek1, nek2, nek3,
            dv, dk, scale, 0.0, 0.0,
            0, neq1 * neq2 * neq3, 0, nek1,
            None, 0,
        );

        // Q*K[0] = 1.0 * scale, Q*K[1] = 0.0 * scale
        // After softmax: weight[0] ~= 0.731, weight[1] ~= 0.269
        // Output ~= weight[0] * V[0] + weight[1] * V[1]
        let scale_v = scale;
        let s0 = 1.0f32 * scale_v;
        let s1 = 0.0f32;
        let m = s0.max(s1);
        let w0 = (s0 - m).exp();
        let w1 = (s1 - m).exp();
        let s = w0 + w1;
        let expected_0 = w0 / s; // output[0]
        let expected_1 = w1 / s; // output[1]

        let eps = 1e-4;
        assert!((dst[0] - expected_0).abs() < eps, "dst[0]={} expected={}", dst[0], expected_0);
        assert!((dst[1] - expected_1).abs() < eps, "dst[1]={} expected={}", dst[1], expected_1);
        assert!(dst[2].abs() < eps);
        assert!(dst[3].abs() < eps);
    }
}
