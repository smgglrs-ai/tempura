pub use crate::Attention;
use crate::Result;

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
use cuda_core::DeviceBuffer;

// ── Dot product strategies ─────────────────────────────────────────────────

#[cfg(feature = "cpu")]
#[inline(always)]
fn dot_f32(q: &[f32], k: &[f32]) -> f32 {
    q.iter().zip(k).map(|(a, b)| a * b).sum()
}

#[cfg(feature = "cpu")]
#[inline(always)]
fn dot_f16(q: &[f32], k: &[f32]) -> f32 {
    q.iter()
        .zip(k)
        .map(|(a, b)| half::f16::from_f32(*a).to_f32() * half::f16::from_f32(*b).to_f32())
        .sum()
}

// ── Naive attention (parameterized by dot product) ─────────────────────────

#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch: usize,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    window: Option<usize>,
    dot_fn: fn(&[f32], &[f32]) -> f32,
) -> Result<()> {
    let heads_per_group = num_heads / num_kv_heads;

    for b in 0..batch {
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            for s in 0..seq_len {
                let q_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                let cache_offset = kv_len - seq_len;
                let causal_limit = cache_offset + s;
                let window_start = match window {
                    Some(w) if causal_limit >= w => causal_limit - w + 1,
                    _ => 0,
                };

                let mut scores = vec![0.0f32; kv_len];
                for (kv_s, score) in scores.iter_mut().enumerate() {
                    if kv_s > causal_limit || kv_s < window_start {
                        *score = f32::NEG_INFINITY;
                        continue;
                    }
                    let k_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim;
                    let k_vec = &k[k_offset..k_offset + head_dim];
                    *score = dot_fn(q_vec, k_vec) * scale;
                }

                let scores_in = scores.clone();
                crate::ggml::ops_llama::soft_max_rows_f32(&scores_in, &mut scores, None, kv_len, 1.0);

                let o_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for (kv_s, &score) in scores.iter().enumerate() {
                        let v_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim + d;
                        val += score * v[v_offset];
                    }
                    output[o_offset + d] = val;
                }
            }
        }
    }

    Ok(())
}

/// Optimized multi-token attention: parallel over heads, pre-allocated scores,
/// SIMD dot products. Matches ggml semantics (f64 softmax sum, ggml_exp).
fn optimized_attention_ggml(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    _batch: usize,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    window: Option<usize>,
) -> Result<()> {
    let heads_per_group = num_heads / num_kv_heads;
    let out_ptr = output.as_mut_ptr() as usize;
    let q_ptr = q.as_ptr() as usize;
    let k_ptr = k.as_ptr() as usize;
    let v_ptr = v.as_ptr() as usize;

    crate::thread_pool::global_pool().parallel_chunks(num_heads * seq_len, 1, &|start, end| {
        let q_sl = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, q.len()) };
        let k_sl = unsafe { std::slice::from_raw_parts(k_ptr as *const f32, k.len()) };
        let v_sl = unsafe { std::slice::from_raw_parts(v_ptr as *const f32, v.len()) };

        let mut scores = vec![0.0f32; kv_len];

        for idx in start..end {
            let h = idx / seq_len;
            let s = idx % seq_len;
            let kv_h = h / heads_per_group;

            let q_offset = (s * num_heads + h) * head_dim;
            let q_vec = &q_sl[q_offset..q_offset + head_dim];

            let cache_offset = kv_len - seq_len;
            let causal_limit = cache_offset + s;
            let window_start = match window {
                Some(w) if causal_limit >= w => causal_limit - w + 1,
                _ => 0,
            };

            for (kv_s, score) in scores.iter_mut().enumerate() {
                if kv_s > causal_limit || kv_s < window_start {
                    *score = f32::NEG_INFINITY;
                    continue;
                }
                let k_offset = (kv_s * num_kv_heads + kv_h) * head_dim;
                *score = dot_f32(q_vec, &k_sl[k_offset..k_offset + head_dim]) * scale;
            }

            let mut max_score = f32::NEG_INFINITY;
            for &sc in &scores[..kv_len] {
                if sc > max_score {
                    max_score = sc;
                }
            }

            let sum = crate::ggml::vec_llama::vec_soft_max_f32_inplace(&mut scores[..kv_len], max_score);
            let inv_sum = (1.0 / sum) as f32;
            for sc in &mut scores[..kv_len] {
                *sc *= inv_sum;
            }

            let o_offset = (s * num_heads + h) * head_dim;
            let out_sl = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(o_offset), head_dim)
            };
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for (kv_s, &score) in scores[..kv_len].iter().enumerate() {
                    let v_offset = (kv_s * num_kv_heads + kv_h) * head_dim + d;
                    val += score * v_sl[v_offset];
                }
                out_sl[d] = val;
            }
        }
    });

    Ok(())
}

/// GGML-compatible naive attention: f64 softmax sum, standard expf.
/// Translated from ggml.c:14576 (softmax) + ggml.c:16660 (mul_mat for QKV).
#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
fn naive_attention_ggml(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch: usize,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    window: Option<usize>,
) -> Result<()> {
    let heads_per_group = num_heads / num_kv_heads;

    for b in 0..batch {
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            for s in 0..seq_len {
                let q_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                let cache_offset = kv_len - seq_len;
                let causal_limit = cache_offset + s;
                let window_start = match window {
                    Some(w) if causal_limit >= w => causal_limit - w + 1,
                    _ => 0,
                };

                // C: QK dot product with scale
                let mut scores = vec![0.0f32; kv_len];
                for (kv_s, score) in scores.iter_mut().enumerate() {
                    if kv_s > causal_limit || kv_s < window_start {
                        *score = f32::NEG_INFINITY;
                        continue;
                    }
                    let k_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim;
                    let k_vec = &k[k_offset..k_offset + head_dim];
                    // C: ggml_vec_dot_f32 then scale
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_vec[d] * k_vec[d];
                    }
                    *score = dot * scale;
                }

                // C: ggml_vec_max_f32
                let mut max_score = f32::NEG_INFINITY;
                for &s in &scores {
                    if s > max_score {
                        max_score = s;
                    }
                }

                // C: ggml_vec_soft_max_f32 — AVX2 polynomial exp path
                let sum = crate::ggml::vec_llama::vec_soft_max_f32_inplace(&mut scores, max_score);

                // C: sum = 1.0/sum; ggml_vec_scale_f32(nc, dp, sum);
                let inv_sum = (1.0 / sum) as f32;
                for s in &mut scores {
                    *s *= inv_sum;
                }

                // C: V weighted sum
                let o_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for (kv_s, &score) in scores.iter().enumerate() {
                        let v_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim + d;
                        val += score * v[v_offset];
                    }
                    output[o_offset + d] = val;
                }
            }
        }
    }

    Ok(())
}

/// GGML-compatible attention with logit softcapping.
/// After computing score = dot(q, k) * scale, applies tanh(score/cap) * cap
/// before softmax. Used by Gemma 2 and similar architectures.
#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
fn naive_attention_ggml_softcap(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch: usize,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    window: Option<usize>,
    cap: f32,
) -> Result<()> {
    let heads_per_group = num_heads / num_kv_heads;

    for b in 0..batch {
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            for s in 0..seq_len {
                let q_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                let cache_offset = kv_len - seq_len;
                let causal_limit = cache_offset + s;
                let window_start = match window {
                    Some(w) if causal_limit >= w => causal_limit - w + 1,
                    _ => 0,
                };

                let mut scores = vec![0.0f32; kv_len];
                for (kv_s, score) in scores.iter_mut().enumerate() {
                    if kv_s > causal_limit || kv_s < window_start {
                        *score = f32::NEG_INFINITY;
                        continue;
                    }
                    let k_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim;
                    let k_vec = &k[k_offset..k_offset + head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_vec[d] * k_vec[d];
                    }
                    let raw = dot * scale;
                    *score = (raw / cap).tanh() * cap;
                }

                let mut max_score = f32::NEG_INFINITY;
                for &s in &scores {
                    if s > max_score {
                        max_score = s;
                    }
                }

                let sum = crate::ggml::vec_llama::vec_soft_max_f32_inplace(&mut scores, max_score);

                let inv_sum = (1.0 / sum) as f32;
                for s in &mut scores {
                    *s *= inv_sum;
                }

                let o_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for (kv_s, &score) in scores.iter().enumerate() {
                        let v_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim + d;
                        val += score * v[v_offset];
                    }
                    output[o_offset + d] = val;
                }
            }
        }
    }

    Ok(())
}

// ── Flash attention (tiled online softmax, parameterized by dot product) ───

#[cfg(feature = "cpu")]
const FLASH_BC: usize = 32;

#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
fn flash_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch: usize,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    dot_fn: fn(&[f32], &[f32]) -> f32,
) -> Result<()> {
    let heads_per_group = num_heads / num_kv_heads;

    for b in 0..batch {
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            for s in 0..seq_len {
                let q_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                let cache_offset = kv_len - seq_len;
                let causal_limit = cache_offset + s;

                let mut o_acc = vec![0.0f32; head_dim];
                let mut row_max = f32::NEG_INFINITY;
                let mut row_sum = 0.0f32;

                let num_tiles = kv_len.div_ceil(FLASH_BC);
                for tile in 0..num_tiles {
                    let tile_start = tile * FLASH_BC;
                    let tile_end = (tile_start + FLASH_BC).min(kv_len);

                    let mut tile_max = f32::NEG_INFINITY;
                    let mut tile_scores = vec![0.0f32; tile_end - tile_start];

                    for (i, kv_s) in (tile_start..tile_end).enumerate() {
                        if kv_s > causal_limit {
                            tile_scores[i] = f32::NEG_INFINITY;
                            continue;
                        }
                        let k_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim;
                        let k_vec = &k[k_offset..k_offset + head_dim];
                        tile_scores[i] = dot_fn(q_vec, k_vec) * scale;
                        if tile_scores[i] > tile_max {
                            tile_max = tile_scores[i];
                        }
                    }

                    if tile_max == f32::NEG_INFINITY {
                        continue;
                    }

                    let new_max = row_max.max(tile_max);
                    let correction = if row_max > f32::NEG_INFINITY {
                        crate::portable_math::portable_exp(row_max - new_max)
                    } else {
                        0.0
                    };

                    for val in &mut o_acc {
                        *val *= correction;
                    }
                    row_sum *= correction;

                    let mut tile_sum = 0.0f32;
                    for score in &mut tile_scores {
                        if *score > f32::NEG_INFINITY {
                            *score = crate::portable_math::portable_exp(*score - new_max);
                        } else {
                            *score = 0.0;
                        }
                        tile_sum += *score;
                    }

                    for (i, kv_s) in (tile_start..tile_end).enumerate() {
                        let w = tile_scores[i];
                        if w == 0.0 {
                            continue;
                        }
                        let v_base = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim;
                        for d in 0..head_dim {
                            o_acc[d] += w * v[v_base + d];
                        }
                    }

                    row_sum += tile_sum;
                    row_max = new_max;
                }

                let o_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                let inv_sum = if row_sum > 0.0 { 1.0 / row_sum } else { 0.0 };
                for d in 0..head_dim {
                    output[o_offset + d] = o_acc[d] * inv_sum;
                }
            }
        }
    }

    Ok(())
}

// ── ALiBi slope computation ───────────────────────────────────────────────
//
// Translated from ggml's ggml_compute_forward_soft_max_f32 (ops.cpp:5348-5370).
// For n_heads heads with max_bias > 0:
//   n_head_log2 = 1 << floor(log2(n_heads))
//   m0 = 2^(-max_bias / n_head_log2)
//   m1 = 2^(-max_bias / 2 / n_head_log2)
//   slope(h) = m0^(h+1)                    if h < n_head_log2
//            = m1^(2*(h-n_head_log2)+1)     otherwise

#[cfg(feature = "cpu")]
pub fn alibi_slopes(n_heads: usize, max_bias: f32) -> Vec<f32> {
    if max_bias <= 0.0 || n_heads == 0 {
        return vec![1.0; n_heads];
    }
    let n_head_log2 = 1u32 << (n_heads as f32).log2().floor() as u32;
    let m0 = 2.0f32.powf(-max_bias / n_head_log2 as f32);
    let m1 = 2.0f32.powf(-max_bias / 2.0 / n_head_log2 as f32);

    (0..n_heads)
        .map(|h| {
            if (h as u32) < n_head_log2 {
                m0.powf(h as f32 + 1.0)
            } else {
                m1.powf(2.0 * (h as u32 - n_head_log2) as f32 + 1.0)
            }
        })
        .collect()
}

// ── Masked attention with ALiBi ──────────────────────────────────────────
//
// Supports:
// - Additive mask: mask[q_pos * kv_len + k_pos] added to score before softmax
// - ALiBi slopes: slope[h] * (k_pos - q_pos) added to score
// - Both: mask + alibi combined
// - Neither: falls back to causal masking
//
// Uses ggml-compatible softmax: f64 sum accumulation, ggml_exp.

#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
fn naive_attention_masked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch: usize,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    mask: Option<&[f32]>,
    slopes: Option<&[f32]>,
    window: Option<usize>,
) -> Result<()> {
    let heads_per_group = num_heads / num_kv_heads;
    let has_mask = mask.is_some();

    for b in 0..batch {
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            let slope = slopes.map_or(0.0, |s| s[h]);

            for s in 0..seq_len {
                let q_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                let cache_offset = kv_len - seq_len;
                let q_pos = cache_offset + s;
                let causal_limit = q_pos;
                let window_start = match window {
                    Some(w) if causal_limit >= w => causal_limit - w + 1,
                    _ => 0,
                };

                let mut scores = vec![0.0f32; kv_len];
                for (kv_s, score) in scores.iter_mut().enumerate() {
                    if has_mask {
                        let mask_val = mask.unwrap()[s * kv_len + kv_s];
                        if mask_val == f32::NEG_INFINITY {
                            *score = f32::NEG_INFINITY;
                            continue;
                        }
                    } else if kv_s > causal_limit || kv_s < window_start {
                        *score = f32::NEG_INFINITY;
                        continue;
                    }

                    let k_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim;
                    let k_vec = &k[k_offset..k_offset + head_dim];

                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_vec[d] * k_vec[d];
                    }
                    *score = dot * scale;

                    if has_mask {
                        *score += mask.unwrap()[s * kv_len + kv_s];
                    }

                    if slopes.is_some() {
                        *score += slope * (kv_s as f32 - q_pos as f32);
                    }
                }

                let mut max_score = f32::NEG_INFINITY;
                for &sc in &scores {
                    if sc > max_score {
                        max_score = sc;
                    }
                }

                let sum = crate::ggml::vec_llama::vec_soft_max_f32_inplace(&mut scores, max_score);

                let inv_sum = (1.0 / sum) as f32;
                for sc in &mut scores {
                    *sc *= inv_sum;
                }

                let o_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for (kv_s, &score) in scores.iter().enumerate() {
                        let v_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim + d;
                        val += score * v[v_offset];
                    }
                    output[o_offset + d] = val;
                }
            }
        }
    }

    Ok(())
}

/// Decode-only attention for a range of heads [head_start, head_end).
/// Pre-allocated scores buffer avoids per-head allocation.
/// seq_len MUST be 1 (decode). For parallelize-across-heads from the decode loop.
#[cfg(feature = "cpu")]
pub fn attention_decode_heads(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    scores: &mut [f32],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    head_start: usize,
    head_end: usize,
) {
    let heads_per_group = num_heads / num_kv_heads;

    for h in head_start..head_end {
        let kv_h = h / heads_per_group;
        let q_offset = h * head_dim;
        let q_vec = &q[q_offset..q_offset + head_dim];

        let scores = &mut scores[..kv_len];

        // QK: AVX2 vec_dot_f32 replaces scalar d-loop (4×8-wide FMA for head_dim=128)
        for kv_s in 0..kv_len {
            let k_offset = (kv_s * num_kv_heads + kv_h) * head_dim;
            let k_vec = &k[k_offset..k_offset + head_dim];
            scores[kv_s] = crate::ggml::vec_llama::vec_dot_f32(q_vec, k_vec) * scale;
        }

        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum = crate::ggml::vec_llama::vec_soft_max_f32_inplace(scores, max_score);
        let inv_sum = (1.0 / sum) as f32;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }

        // AV: AXPY pattern — outer loop over kv_s (contiguous v_row), inner AVX2 fmadd.
        // Matches llama.cpp's ggml_vec_mad_f32 (4×8-wide fmadd per iteration).
        let o_offset = h * head_dim;
        let out_row = &mut output[o_offset..o_offset + head_dim];
        for x in out_row.iter_mut() {
            *x = 0.0;
        }

        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        for (kv_s, &score) in scores.iter().enumerate() {
            let v_offset = (kv_s * num_kv_heads + kv_h) * head_dim;
            let v_row = &v[v_offset..v_offset + head_dim];
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe { axpy_avx2(out_row, v_row, score); }
            } else {
                for d in 0..head_dim {
                    out_row[d] += score * v_row[d];
                }
            }
        }
    }
}

/// AVX2 AXPY: out[d] += scale * src[d] for n elements (matching llama.cpp's ggml_vec_mad_f32).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(out: &mut [f32], src: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let n = out.len();
    debug_assert_eq!(n, src.len());
    let sv = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 32 <= n {
        let o0 = _mm256_loadu_ps(out.as_ptr().add(i));
        let o1 = _mm256_loadu_ps(out.as_ptr().add(i + 8));
        let o2 = _mm256_loadu_ps(out.as_ptr().add(i + 16));
        let o3 = _mm256_loadu_ps(out.as_ptr().add(i + 24));
        let r0 = _mm256_fmadd_ps(sv, _mm256_loadu_ps(src.as_ptr().add(i)), o0);
        let r1 = _mm256_fmadd_ps(sv, _mm256_loadu_ps(src.as_ptr().add(i + 8)), o1);
        let r2 = _mm256_fmadd_ps(sv, _mm256_loadu_ps(src.as_ptr().add(i + 16)), o2);
        let r3 = _mm256_fmadd_ps(sv, _mm256_loadu_ps(src.as_ptr().add(i + 24)), o3);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), r0);
        _mm256_storeu_ps(out.as_mut_ptr().add(i + 8), r1);
        _mm256_storeu_ps(out.as_mut_ptr().add(i + 16), r2);
        _mm256_storeu_ps(out.as_mut_ptr().add(i + 24), r3);
        i += 32;
    }
    while i < n {
        out[i] += scale * src[i];
        i += 1;
    }
}

/// SIMD f16→f32 dot product: q[f32] · k[f16], head_dim elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn dot_f32_f16(q: &[f32], k_f16: *const u16, head_dim: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut d = 0;
    while d + 8 <= head_dim {
        let h = _mm_loadu_si128(k_f16.add(d) as *const __m128i);
        let kv = _mm256_cvtph_ps(h);
        let qv = _mm256_loadu_ps(q.as_ptr().add(d));
        acc = _mm256_fmadd_ps(qv, kv, acc);
        d += 8;
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// SIMD weighted sum: output[d] += score * v_f16[offset+d] for head_dim elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn weighted_add_f16(output: &mut [f32], v_f16: *const u16, score: f32, head_dim: usize) {
    use std::arch::x86_64::*;
    let sv = _mm256_set1_ps(score);
    let mut d = 0;
    while d + 8 <= head_dim {
        let h = _mm_loadu_si128(v_f16.add(d) as *const __m128i);
        let vv = _mm256_cvtph_ps(h);
        let ov = _mm256_loadu_ps(output.as_ptr().add(d));
        _mm256_storeu_ps(output.as_mut_ptr().add(d), _mm256_fmadd_ps(sv, vv, ov));
        d += 8;
    }
}

pub fn attention_decode_heads_f16kv(
    q: &[f32],
    k_f16: &[u16],
    v_f16: &[u16],
    output: &mut [f32],
    scores: &mut [f32],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    head_start: usize,
    head_end: usize,
) {
    let heads_per_group = num_heads / num_kv_heads;

    for h in head_start..head_end {
        let kv_h = h / heads_per_group;
        let q_offset = h * head_dim;
        let q_vec = &q[q_offset..q_offset + head_dim];

        let scores = &mut scores[..kv_len];
        for kv_s in 0..kv_len {
            let k_offset = (kv_s * num_kv_heads + kv_h) * head_dim;
            scores[kv_s] =
                unsafe { dot_f32_f16(q_vec, k_f16.as_ptr().add(k_offset), head_dim) } * scale;
        }

        let mut max_score = f32::NEG_INFINITY;
        for &s in scores.iter() {
            if s > max_score {
                max_score = s;
            }
        }

        let sum = crate::ggml::vec_llama::vec_soft_max_f32_inplace(scores, max_score);
        let inv_sum = (1.0 / sum) as f32;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }

        let o_offset = h * head_dim;
        let out_slice = &mut output[o_offset..o_offset + head_dim];
        for d in out_slice.iter_mut() {
            *d = 0.0;
        }
        for (kv_s, &score) in scores.iter().enumerate() {
            let v_offset = (kv_s * num_kv_heads + kv_h) * head_dim;
            unsafe { weighted_add_f16(out_slice, v_f16.as_ptr().add(v_offset), score, head_dim) };
        }
    }
}

/// Process a range of (head, position) work items for prefill attention.
/// `item_start..item_end` indexes into `num_heads * seq_len` items where
/// item = head * seq_len + position. Uses pre-allocated `scores` buffer.
#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_items(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output_addr: usize,
    scores: &mut [f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    item_start: usize,
    item_end: usize,
) {
    let heads_per_group = num_heads / num_kv_heads;

    for idx in item_start..item_end {
        let h = idx / seq_len;
        let s = idx % seq_len;
        let kv_h = h / heads_per_group;

        let q_offset = (s * num_heads + h) * head_dim;
        let q_vec = &q[q_offset..q_offset + head_dim];

        let cache_offset = kv_len - seq_len;
        let causal_limit = cache_offset + s;

        let scores = &mut scores[..kv_len];
        for (kv_s, score) in scores.iter_mut().enumerate() {
            if kv_s > causal_limit {
                *score = f32::NEG_INFINITY;
                continue;
            }
            let k_offset = (kv_s * num_kv_heads + kv_h) * head_dim;
            *score = dot_f32(q_vec, &k[k_offset..k_offset + head_dim]) * scale;
        }

        let mut max_score = f32::NEG_INFINITY;
        for &sc in &scores[..kv_len] {
            if sc > max_score {
                max_score = sc;
            }
        }

        let sum = crate::ggml::vec_llama::vec_soft_max_f32_inplace(&mut scores[..kv_len], max_score);
        let inv_sum = (1.0 / sum) as f32;
        for sc in &mut scores[..kv_len] {
            *sc *= inv_sum;
        }

        let o_offset = (s * num_heads + h) * head_dim;
        let out_sl = unsafe {
            std::slice::from_raw_parts_mut((output_addr as *mut f32).add(o_offset), head_dim)
        };
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for (kv_s, &score) in scores[..kv_len].iter().enumerate() {
                let v_offset = (kv_s * num_kv_heads + kv_h) * head_dim + d;
                val += score * v[v_offset];
            }
            out_sl[d] = val;
        }
    }
}

pub fn attention_prefill_items_f16kv(
    q: &[f32],
    k_f16: &[u16],
    v_f16: &[u16],
    output_addr: usize,
    scores: &mut [f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    item_start: usize,
    item_end: usize,
) {
    let heads_per_group = num_heads / num_kv_heads;

    for idx in item_start..item_end {
        let h = idx / seq_len;
        let s = idx % seq_len;
        let kv_h = h / heads_per_group;

        let q_offset = (s * num_heads + h) * head_dim;
        let q_vec = &q[q_offset..q_offset + head_dim];

        let cache_offset = kv_len - seq_len;
        let causal_limit = cache_offset + s;

        let scores = &mut scores[..kv_len];
        for (kv_s, score) in scores.iter_mut().enumerate() {
            if kv_s > causal_limit {
                *score = f32::NEG_INFINITY;
                continue;
            }
            let k_offset = (kv_s * num_kv_heads + kv_h) * head_dim;
            *score = unsafe { dot_f32_f16(q_vec, k_f16.as_ptr().add(k_offset), head_dim) } * scale;
        }

        let mut max_score = f32::NEG_INFINITY;
        for &sc in &scores[..kv_len] {
            if sc > max_score {
                max_score = sc;
            }
        }

        let sum = crate::ggml::vec_llama::vec_soft_max_f32_inplace(&mut scores[..kv_len], max_score);
        let inv_sum = (1.0 / sum) as f32;
        for sc in &mut scores[..kv_len] {
            *sc *= inv_sum;
        }

        let o_offset = (s * num_heads + h) * head_dim;
        let out_sl = unsafe {
            std::slice::from_raw_parts_mut((output_addr as *mut f32).add(o_offset), head_dim)
        };
        for d in out_sl.iter_mut() {
            *d = 0.0;
        }
        for (kv_s, &score) in scores[..kv_len].iter().enumerate() {
            let v_offset = (kv_s * num_kv_heads + kv_h) * head_dim;
            unsafe { weighted_add_f16(out_sl, v_f16.as_ptr().add(v_offset), score, head_dim) };
        }
    }
}

// ── Concrete CPU attention structs ─────────────────────────────────────────

#[cfg(feature = "cpu")]
pub struct CpuAttentionGgml;

#[cfg(feature = "cpu")]
impl Attention for CpuAttentionGgml {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Tested
    }

    fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        if seq_len > 1 && batch == 1 {
            return optimized_attention_ggml(
                q,
                k,
                v,
                output,
                batch,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                None,
            );
        }
        naive_attention_ggml(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            None,
        )
    }

    fn forward_windowed(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        window_size: usize,
    ) -> Result<()> {
        naive_attention_ggml(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            Some(window_size),
        )
    }
}

/// GGML attention with logit softcapping: tanh(score/cap)*cap before softmax.
/// Used by Gemma 2 and similar models.
#[cfg(feature = "cpu")]
pub struct CpuAttentionSoftcap {
    pub cap: f32,
}

#[cfg(feature = "cpu")]
impl CpuAttentionSoftcap {
    pub fn new(cap: f32) -> Self {
        Self { cap }
    }
}

#[cfg(feature = "cpu")]
impl Attention for CpuAttentionSoftcap {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Tested
    }

    fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        naive_attention_ggml_softcap(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            None,
            self.cap,
        )
    }

    fn forward_windowed(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        window_size: usize,
    ) -> Result<()> {
        naive_attention_ggml_softcap(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            Some(window_size),
            self.cap,
        )
    }
}

/// ALiBi-aware attention with optional additive mask support.
///
/// Computes per-head ALiBi slopes from `max_bias` (ggml convention).
/// When `max_bias > 0`, adds `slope[h] * (k_pos - q_pos)` to each score.
/// Accepts an optional additive mask that replaces built-in causal masking.
#[cfg(feature = "cpu")]
pub struct CpuAttentionAlibi {
    slopes: Vec<f32>,
}

#[cfg(feature = "cpu")]
impl CpuAttentionAlibi {
    pub fn new(n_heads: usize, max_bias: f32) -> Self {
        Self {
            slopes: alibi_slopes(n_heads, max_bias),
        }
    }

    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_masked(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        mask: Option<&[f32]>,
    ) -> Result<()> {
        naive_attention_masked(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            mask,
            Some(&self.slopes),
            None,
        )
    }
}

#[cfg(feature = "cpu")]
impl Attention for CpuAttentionAlibi {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Tested
    }

    fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        naive_attention_masked(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            None,
            Some(&self.slopes),
            None,
        )
    }

    fn forward_windowed(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        window_size: usize,
    ) -> Result<()> {
        naive_attention_masked(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            None,
            Some(&self.slopes),
            Some(window_size),
        )
    }
}

/// Attention with additive mask support (no ALiBi).
/// Mask shape: [seq_len, kv_len], additive (use -inf for masked positions).
/// When mask is None, applies causal masking as usual.
#[cfg(feature = "cpu")]
pub struct CpuAttentionMasked;

#[cfg(feature = "cpu")]
impl CpuAttentionMasked {
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_mask(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        mask: &[f32],
    ) -> Result<()> {
        naive_attention_masked(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            Some(mask),
            None,
            None,
        )
    }
}

#[cfg(feature = "cpu")]
impl Attention for CpuAttentionMasked {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Tested
    }

    fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        naive_attention_masked(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            None,
            None,
            None,
        )
    }
}

#[cfg(feature = "cpu")]
pub struct CpuAttention;

#[cfg(feature = "cpu")]
pub struct CpuAttentionF16;

#[cfg(feature = "cpu")]
pub struct CpuFlashAttention;

#[cfg(feature = "cpu")]
pub struct CpuFlashAttentionF16;

#[cfg(feature = "cpu")]
macro_rules! impl_cpu_attention {
    ($name:ident, $algo:ident, $dot:ident $(, $extra:expr)?) => {
        #[cfg(feature = "cpu")]
        impl Attention for $name {
            fn verification_tier(&self) -> crate::VerificationTier {
                crate::VerificationTier::Tested
            }

            fn forward(
                &self,
                q: &[f32], k: &[f32], v: &[f32], output: &mut [f32],
                batch: usize, seq_len: usize, kv_len: usize,
                num_heads: usize, num_kv_heads: usize, head_dim: usize, scale: f32,
            ) -> Result<()> {
                $algo(q, k, v, output, batch, seq_len, kv_len,
                      num_heads, num_kv_heads, head_dim, scale, $($extra,)? $dot)
            }
        }
    };
}

impl_cpu_attention!(CpuAttentionF16, naive_attention, dot_f16, None);
impl_cpu_attention!(CpuFlashAttention, flash_attention, dot_f32);
impl_cpu_attention!(CpuFlashAttentionF16, flash_attention, dot_f16);

#[cfg(feature = "cpu")]
impl Attention for CpuAttention {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Proven
    }

    fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        naive_attention(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            None,
            dot_f32,
        )
    }

    fn forward_windowed(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        window_size: usize,
    ) -> Result<()> {
        naive_attention(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            Some(window_size),
            dot_f32,
        )
    }
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
pub struct CudaAttention {
    gpu: crate::gpu::GpuContext,
    module: crate::gpu::attention_kernel::kernels::LoadedModule,
    flash_module: Option<crate::gpu::flash_attention_kernel::kernels::LoadedModule>,
    decode_module: Option<crate::gpu::flash_decode_kernel::kernels::LoadedModule>,
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
unsafe impl Send for CudaAttention {}
#[cfg(all(feature = "cuda", not(feature = "cutile")))]
unsafe impl Sync for CudaAttention {}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
impl CudaAttention {
    pub fn gpu(&self) -> &crate::gpu::GpuContext {
        &self.gpu
    }

    pub fn new() -> Result<Self> {
        let gpu = crate::gpu::GpuContext::new(0)?;
        let module = crate::gpu::attention_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        let flash_module = crate::gpu::flash_attention_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))
            .ok();
        let decode_module = crate::gpu::flash_decode_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))
            .ok();
        Ok(Self {
            gpu,
            module,
            flash_module,
            decode_module,
        })
    }

    pub fn with_context(ctx: std::sync::Arc<cuda_core::CudaContext>) -> Result<Self> {
        let gpu = crate::gpu::GpuContext::from_context(ctx);
        let module = crate::gpu::attention_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        let flash_module = crate::gpu::flash_attention_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))
            .ok();
        let decode_module = crate::gpu::flash_decode_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))
            .ok();
        Ok(Self {
            gpu,
            module,
            flash_module,
            decode_module,
        })
    }
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
#[allow(clippy::too_many_arguments)]
impl CudaAttention {
    pub fn forward_device(
        &self,
        q: &DeviceBuffer<f32>,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();

        if head_dim == 128 && seq_len == 1 {
            if kv_len >= 512 {
                if let Some(ref decode) = self.decode_module {
                    let kv_per_split = 256usize;
                    let num_splits = kv_len.div_ceil(kv_per_split);
                    let partials_size = batch * num_heads * num_splits * (head_dim + 2);
                    let mut partials = crate::gpu::alloc_device(&stream, partials_size)?;

                    let partial_config = LaunchConfig {
                        grid_dim: (num_heads as u32, num_splits as u32, batch as u32),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    decode
                        .flash_decode_partial(
                            &stream,
                            partial_config,
                            q,
                            k,
                            v,
                            &mut partials,
                            batch as u32,
                            kv_len as u32,
                            num_heads as u32,
                            num_kv_heads as u32,
                            head_dim as u32,
                            num_splits as u32,
                            kv_per_split as u32,
                            scale,
                        )
                        .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

                    let reduce_config = LaunchConfig {
                        grid_dim: (num_heads as u32, 1, batch as u32),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    decode
                        .flash_decode_reduce(
                            &stream,
                            reduce_config,
                            &partials,
                            output,
                            batch as u32,
                            num_heads as u32,
                            head_dim as u32,
                            num_splits as u32,
                        )
                        .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

                    return Ok(());
                }
            }

            if let Some(ref flash) = self.flash_module {
                let config = LaunchConfig {
                    grid_dim: (
                        num_heads as u32,
                        (seq_len as u32).div_ceil(32),
                        batch as u32,
                    ),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                };
                flash
                    .flash_attention_forward(
                        &stream,
                        config,
                        q,
                        k,
                        v,
                        output,
                        batch as u32,
                        seq_len as u32,
                        kv_len as u32,
                        num_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        scale,
                    )
                    .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
                return Ok(());
            }
        }

        let total = batch * seq_len * num_heads * head_dim;
        self.module
            .attention_forward(
                &stream,
                LaunchConfig::for_num_elems(total as u32),
                q,
                k,
                v,
                output,
                batch as u32,
                seq_len as u32,
                kv_len as u32,
                num_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                scale,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn forward_device_with_mode(
        &self,
        q: &DeviceBuffer<f32>,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        mode: crate::AttentionMode,
    ) -> crate::Result<()> {
        use crate::AttentionMode::*;
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();

        match mode {
            NaiveF32 => {
                let total = batch * seq_len * num_heads * head_dim;
                self.module
                    .attention_forward(
                        &stream,
                        LaunchConfig::for_num_elems(total as u32),
                        q,
                        k,
                        v,
                        output,
                        batch as u32,
                        seq_len as u32,
                        kv_len as u32,
                        num_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        scale,
                    )
                    .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
            }
            NaiveF16 => {
                let total = batch * seq_len * num_heads * head_dim;
                self.module
                    .attention_forward_f16(
                        &stream,
                        LaunchConfig::for_num_elems(total as u32),
                        q,
                        k,
                        v,
                        output,
                        batch as u32,
                        seq_len as u32,
                        kv_len as u32,
                        num_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        scale,
                    )
                    .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
            }
            FlashF16 => {
                let flash = self.flash_module.as_ref().ok_or_else(|| {
                    crate::KernelError::InvalidConfig("flash module not loaded".into())
                })?;
                let config = LaunchConfig {
                    grid_dim: (
                        num_heads as u32,
                        (seq_len as u32).div_ceil(32),
                        batch as u32,
                    ),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                };
                flash
                    .flash_attention_forward(
                        &stream,
                        config,
                        q,
                        k,
                        v,
                        output,
                        batch as u32,
                        seq_len as u32,
                        kv_len as u32,
                        num_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        scale,
                    )
                    .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
            }
            FlashF32 => {
                let flash = self.flash_module.as_ref().ok_or_else(|| {
                    crate::KernelError::InvalidConfig("flash module not loaded".into())
                })?;
                let total = batch * seq_len * num_heads;
                let config = LaunchConfig::for_num_elems(total as u32);
                flash
                    .flash_attention_forward_f32(
                        &stream,
                        config,
                        q,
                        k,
                        v,
                        output,
                        batch as u32,
                        seq_len as u32,
                        kv_len as u32,
                        num_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        scale,
                    )
                    .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
            }
            Auto => {
                return self.forward_device(
                    q,
                    k,
                    v,
                    output,
                    batch,
                    seq_len,
                    kv_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                );
            }
        }
        Ok(())
    }

    pub fn forward_device_naive(
        &self,
        q: &DeviceBuffer<f32>,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();
        let total = batch * seq_len * num_heads * head_dim;
        self.module
            .attention_forward(
                &stream,
                LaunchConfig::for_num_elems(total as u32),
                q,
                k,
                v,
                output,
                batch as u32,
                seq_len as u32,
                kv_len as u32,
                num_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                scale,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn forward_device_f16kv(
        &self,
        q: &DeviceBuffer<f32>,
        k: &DeviceBuffer<u16>,
        v: &DeviceBuffer<u16>,
        output: &mut DeviceBuffer<f32>,
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();

        if head_dim == 128 && seq_len > 0 {
            if let Some(ref flash) = self.flash_module {
                let config = LaunchConfig {
                    grid_dim: (
                        num_heads as u32,
                        (seq_len as u32).div_ceil(32),
                        batch as u32,
                    ),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                };
                flash
                    .flash_attention_forward_f16(
                        &stream,
                        config,
                        q,
                        k,
                        v,
                        output,
                        batch as u32,
                        seq_len as u32,
                        kv_len as u32,
                        num_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        scale,
                    )
                    .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
                return Ok(());
            }
        }

        Err(crate::KernelError::InvalidConfig(
            "FP16 KV attention requires head_dim=128 and flash module".to_string(),
        ))
    }
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
impl Attention for CudaAttention {
    fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let stream = self.gpu.stream();
        let total = batch * seq_len * num_heads * head_dim;

        let q_dev = crate::gpu::to_device(&stream, q)?;
        let k_dev = crate::gpu::to_device(&stream, k)?;
        let v_dev = crate::gpu::to_device(&stream, v)?;
        let mut out_dev = crate::gpu::to_device(&stream, &vec![0.0f32; total])?;

        self.forward_device(
            &q_dev,
            &k_dev,
            &v_dev,
            &mut out_dev,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )?;

        crate::gpu::copy_back(&stream, &out_dev, output)?;
        Ok(())
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn attention_q_offset_in_bounds() {
        let batch: usize = kani::any();
        let seq_len: usize = kani::any();
        let num_heads: usize = kani::any();
        let head_dim: usize = kani::any();

        kani::assume(batch >= 1 && batch <= 2);
        kani::assume(seq_len >= 1 && seq_len <= 4);
        kani::assume(num_heads >= 1 && num_heads <= 4);
        kani::assume(head_dim >= 1 && head_dim <= 4);

        let q_len = batch * seq_len * num_heads * head_dim;

        let b: usize = kani::any();
        let s: usize = kani::any();
        let h: usize = kani::any();
        kani::assume(b < batch);
        kani::assume(s < seq_len);
        kani::assume(h < num_heads);

        let q_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
        assert!(q_offset + head_dim <= q_len);
    }

    #[kani::proof]
    fn attention_kv_offset_in_bounds() {
        let batch: usize = kani::any();
        let kv_len: usize = kani::any();
        let num_kv_heads: usize = kani::any();
        let head_dim: usize = kani::any();

        kani::assume(batch >= 1 && batch <= 2);
        kani::assume(kv_len >= 1 && kv_len <= 4);
        kani::assume(num_kv_heads >= 1 && num_kv_heads <= 4);
        kani::assume(head_dim >= 1 && head_dim <= 4);

        let k_len = batch * kv_len * num_kv_heads * head_dim;

        let b: usize = kani::any();
        let kv_s: usize = kani::any();
        let kv_h: usize = kani::any();
        kani::assume(b < batch);
        kani::assume(kv_s < kv_len);
        kani::assume(kv_h < num_kv_heads);

        let k_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim;
        assert!(k_offset + head_dim <= k_len);
    }

    #[kani::proof]
    fn attention_batch_isolation_no_overlap() {
        let batch: usize = kani::any();
        let seq_len: usize = kani::any();
        let num_heads: usize = kani::any();
        let head_dim: usize = kani::any();

        kani::assume(batch >= 2 && batch <= 4);
        kani::assume(seq_len >= 1 && seq_len <= 2);
        kani::assume(num_heads >= 1 && num_heads <= 2);
        kani::assume(head_dim >= 1 && head_dim <= 4);

        let b1: usize = kani::any();
        let b2: usize = kani::any();
        kani::assume(b1 < batch);
        kani::assume(b2 < batch);
        kani::assume(b1 != b2);

        let stride = seq_len * num_heads * head_dim;
        let start1 = b1 * stride;
        let end1 = start1 + stride;
        let start2 = b2 * stride;
        let end2 = start2 + stride;

        assert!(end1 <= start2 || end2 <= start1);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn alibi_slopes_no_panic() {
        let n_heads: usize = kani::any();
        kani::assume(n_heads >= 1 && n_heads <= 8);
        let max_bias: f32 = kani::any();
        kani::assume(max_bias.is_finite() && max_bias >= 0.0 && max_bias <= 16.0);
        let slopes = super::alibi_slopes(n_heads, max_bias);
        assert!(slopes.len() == n_heads);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn alibi_slopes_decreasing() {
        let n_heads: usize = kani::any();
        kani::assume(n_heads >= 2 && n_heads <= 8);
        // Use power-of-two heads so all slopes come from the same geometric series
        kani::assume(n_heads == 2 || n_heads == 4 || n_heads == 8);
        let max_bias: f32 = kani::any();
        kani::assume(max_bias.is_finite() && max_bias > 0.0 && max_bias <= 16.0);
        let slopes = super::alibi_slopes(n_heads, max_bias);
        for i in 1..slopes.len() {
            assert!(slopes[i] <= slopes[i - 1]);
        }
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn alibi_slopes_positive() {
        let n_heads: usize = kani::any();
        kani::assume(n_heads >= 1 && n_heads <= 8);
        let max_bias: f32 = kani::any();
        kani::assume(max_bias.is_finite() && max_bias > 0.0 && max_bias <= 16.0);
        let slopes = super::alibi_slopes(n_heads, max_bias);
        for &s in &slopes {
            assert!(s > 0.0);
        }
    }

    #[kani::proof]
    fn gpu_attention_thread_decomposition_in_bounds() {
        let batch: usize = kani::any();
        let seq_len: usize = kani::any();
        let num_heads: usize = kani::any();
        let head_dim: usize = kani::any();

        kani::assume(batch >= 1 && batch <= 2);
        kani::assume(seq_len >= 1 && seq_len <= 2);
        kani::assume(num_heads >= 1 && num_heads <= 2);
        kani::assume(head_dim >= 1 && head_dim <= 4);

        let total = batch * seq_len * num_heads * head_dim;
        let q_len = total;

        let i: usize = kani::any();
        kani::assume(i < total);

        let hd = head_dim;
        let nh = num_heads;
        let sl = seq_len;

        let d = i % hd;
        let h = (i / hd) % nh;
        let s = (i / hd / nh) % sl;
        let b = i / hd / nh / sl;

        assert!(b < batch);
        assert!(s < seq_len);
        assert!(h < num_heads);
        assert!(d < head_dim);

        let q_base = ((b * sl + s) * nh + h) * hd;
        assert!(q_base + d < q_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_head_self_attention() {
        let attn = CpuAttention;
        let head_dim = 4;
        let seq_len = 2;
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = q.clone();
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; seq_len * head_dim];

        attn.forward(
            &q,
            &k,
            &v,
            &mut output,
            1,
            seq_len,
            seq_len,
            1,
            1,
            head_dim,
            1.0 / (head_dim as f32).sqrt(),
        )
        .unwrap();

        // output should be weighted combination of v rows
        assert_eq!(output.len(), 8);
        // first token attends more to itself (q[0] dot k[0] > q[0] dot k[1])
        assert!(output[0] < 3.0); // closer to v[0]=1 than v[4]=5
    }

    #[test]
    fn test_sliding_window_attention() {
        let attn = CpuAttention;
        let head_dim = 4;
        let seq_len = 4;
        let num_heads = 1;
        let num_kv_heads = 1;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        let mut out_full = vec![0.0f32; seq_len * head_dim];
        attn.forward(
            &q,
            &k,
            &v,
            &mut out_full,
            1,
            seq_len,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();

        let mut out_windowed = vec![0.0f32; seq_len * head_dim];
        attn.forward_windowed(
            &q,
            &k,
            &v,
            &mut out_windowed,
            1,
            seq_len,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            2,
        )
        .unwrap();

        // Token 0: window=2 but only 1 token visible → same as full
        for d in 0..head_dim {
            assert!(
                (out_full[d] - out_windowed[d]).abs() < 1e-5,
                "token 0 dim {d}: full={} windowed={}",
                out_full[d],
                out_windowed[d]
            );
        }

        // Token 3: full attention sees all 4 tokens, windowed sees only tokens 2-3
        // Outputs should differ because windowed ignores tokens 0-1
        let tok3_full = &out_full[3 * head_dim..4 * head_dim];
        let tok3_wind = &out_windowed[3 * head_dim..4 * head_dim];
        let max_diff: f32 = tok3_full
            .iter()
            .zip(tok3_wind)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff > 1e-3,
            "token 3 should differ between full and windowed attention (max_diff={max_diff})"
        );
    }

    #[test]
    fn test_sliding_window_large_window_equals_full() {
        let attn = CpuAttention;
        let head_dim = 4;
        let seq_len = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();
        let k = q.clone();
        let v = q.clone();

        let mut out_full = vec![0.0f32; seq_len * head_dim];
        attn.forward(
            &q,
            &k,
            &v,
            &mut out_full,
            1,
            seq_len,
            seq_len,
            1,
            1,
            head_dim,
            scale,
        )
        .unwrap();

        let mut out_windowed = vec![0.0f32; seq_len * head_dim];
        attn.forward_windowed(
            &q,
            &k,
            &v,
            &mut out_windowed,
            1,
            seq_len,
            seq_len,
            1,
            1,
            head_dim,
            scale,
            1000,
        )
        .unwrap();

        for i in 0..out_full.len() {
            assert!(
                (out_full[i] - out_windowed[i]).abs() < 1e-6,
                "large window should equal full attention at {i}"
            );
        }
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_attention_output_bounded(
                head_dim in 2usize..8usize,
            ) {
                let attn = CpuAttention;
                let seq_len = 2;
                let q = vec![0.1f32; seq_len * head_dim];
                let k = vec![0.1f32; seq_len * head_dim];
                let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();
                let mut output = vec![0.0f32; seq_len * head_dim];
                let scale = 1.0 / (head_dim as f32).sqrt();
                attn.forward(&q, &k, &v, &mut output, 1, seq_len, seq_len, 1, 1, head_dim, scale).unwrap();
                let v_max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let v_min = v.iter().cloned().fold(f32::INFINITY, f32::min);
                for (i, &o) in output.iter().enumerate() {
                    prop_assert!(o >= v_min - 1e-4 && o <= v_max + 1e-4,
                        "output[{i}]={o} outside value range [{v_min}, {v_max}]");
                }
            }
        }
    }

    #[test]
    fn test_causal_masking_prefill_matches_sequential() {
        let attn = CpuAttention;
        let head_dim = 4;
        let num_heads = 1;
        let num_kv_heads = 1;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_data = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, 0.0, // token 1
            0.0, 0.0, 1.0, 0.0, // token 2
        ];
        let k_data = q_data.clone();
        let v_data = vec![
            1.0, 2.0, 3.0, 4.0, // token 0
            5.0, 6.0, 7.0, 8.0, // token 1
            9.0, 10.0, 11.0, 12.0, // token 2
        ];

        // Prefill: all 3 tokens at once (causal masked)
        let mut prefill_out = vec![0.0f32; 3 * head_dim];
        attn.forward(
            &q_data,
            &k_data,
            &v_data,
            &mut prefill_out,
            1,
            3,
            3,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();

        // Sequential: process one token at a time, accumulating KV
        // Token 0: attends only to itself
        let mut seq_out_0 = vec![0.0f32; head_dim];
        attn.forward(
            &q_data[0..head_dim],
            &k_data[0..head_dim],
            &v_data[0..head_dim],
            &mut seq_out_0,
            1,
            1,
            1,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();

        // Token 1: attends to tokens 0-1
        let mut seq_out_1 = vec![0.0f32; head_dim];
        attn.forward(
            &q_data[head_dim..2 * head_dim],
            &k_data[0..2 * head_dim],
            &v_data[0..2 * head_dim],
            &mut seq_out_1,
            1,
            1,
            2,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();

        // Token 2: attends to tokens 0-2
        let mut seq_out_2 = vec![0.0f32; head_dim];
        attn.forward(
            &q_data[2 * head_dim..3 * head_dim],
            &k_data[0..3 * head_dim],
            &v_data[0..3 * head_dim],
            &mut seq_out_2,
            1,
            1,
            3,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();

        // Prefill output for each position must match sequential output
        for d in 0..head_dim {
            assert!(
                (prefill_out[d] - seq_out_0[d]).abs() < 1e-5,
                "pos 0, dim {d}: prefill={} seq={}",
                prefill_out[d],
                seq_out_0[d]
            );
            assert!(
                (prefill_out[head_dim + d] - seq_out_1[d]).abs() < 1e-5,
                "pos 1, dim {d}: prefill={} seq={}",
                prefill_out[head_dim + d],
                seq_out_1[d]
            );
            assert!(
                (prefill_out[2 * head_dim + d] - seq_out_2[d]).abs() < 1e-5,
                "pos 2, dim {d}: prefill={} seq={}",
                prefill_out[2 * head_dim + d],
                seq_out_2[d]
            );
        }
    }

    #[test]
    fn test_cpu_attention_precision_matrix() {
        let head_dim = 128;
        let seq_len = 1;
        let kv_len = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.3)
            .collect();
        let k: Vec<f32> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
            .collect();
        let v: Vec<f32> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.1)
            .collect();

        let run = |attn: &dyn Attention| -> Vec<f32> {
            let mut out = vec![0.0f32; seq_len * num_heads * head_dim];
            attn.forward(
                &q,
                &k,
                &v,
                &mut out,
                1,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
            .unwrap();
            out
        };

        let ref_out = run(&CpuAttention);
        let f16_naive = run(&CpuAttentionF16);
        let f32_flash = run(&CpuFlashAttention);
        let f16_flash = run(&CpuFlashAttentionF16);

        let max_diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        let d_f16_naive = max_diff(&ref_out, &f16_naive);
        let d_f32_flash = max_diff(&ref_out, &f32_flash);
        let d_f16_flash = max_diff(&ref_out, &f16_flash);
        let d_f16_cross = max_diff(&f16_naive, &f16_flash);

        eprintln!("  naive-f32 vs naive-f16:  {d_f16_naive:.6}");
        eprintln!("  naive-f32 vs flash-f32:  {d_f32_flash:.6}");
        eprintln!("  naive-f32 vs flash-f16:  {d_f16_flash:.6}");
        eprintln!("  naive-f16 vs flash-f16:  {d_f16_cross:.6}");

        // f32 flash must match f32 naive exactly (same precision, different algorithm)
        assert!(
            d_f32_flash < 1e-5,
            "flash-f32 diverges from naive-f32: {d_f32_flash}"
        );
        // f16 variants must match each other (same precision, different algorithm)
        assert!(
            d_f16_cross < 1e-5,
            "naive-f16 diverges from flash-f16: {d_f16_cross}"
        );
        // f16 vs f32 should differ (proving f16 truncation has an effect) but not wildly
        assert!(
            d_f16_naive > 1e-6,
            "f16 truncation should have measurable effect"
        );
        assert!(d_f16_naive < 0.1, "f16 divergence too large: {d_f16_naive}");
    }

    #[test]
    fn test_alibi_slopes_power_of_two_heads() {
        // 8 heads, max_bias=8 → standard ALiBi slopes
        // n_head_log2=8, m0 = 2^(-8/8) = 0.5
        // slopes: 0.5^1, 0.5^2, ..., 0.5^8
        let slopes = alibi_slopes(8, 8.0);
        assert_eq!(slopes.len(), 8);
        let expected = [
            0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
        ];
        for (i, (&got, &want)) in slopes.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "slope[{i}]: got={got}, want={want}"
            );
        }
    }

    #[test]
    fn test_alibi_slopes_non_power_of_two_heads() {
        // 6 heads, max_bias=8 → n_head_log2=4
        // m0 = 2^(-8/4) = 0.25
        // m1 = 2^(-8/2/4) = 2^(-1) = 0.5
        // heads 0..3: m0^(h+1) = 0.25, 0.0625, 0.015625, 0.00390625
        // heads 4..5: m1^(2*(h-4)+1) = 0.5^1, 0.5^3
        let slopes = alibi_slopes(6, 8.0);
        assert_eq!(slopes.len(), 6);
        assert!((slopes[0] - 0.25).abs() < 1e-6);
        assert!((slopes[1] - 0.0625).abs() < 1e-6);
        assert!((slopes[4] - 0.5).abs() < 1e-6);
        assert!((slopes[5] - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_slopes_no_bias() {
        let slopes = alibi_slopes(4, 0.0);
        assert!(slopes.iter().all(|&s| s == 1.0));
    }

    #[test]
    fn test_alibi_attention_biases_scores() {
        let head_dim = 4;
        let seq_len = 1;
        let kv_len = 4;
        let num_heads = 2;
        let num_kv_heads = 2;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Uniform Q and K so all dot products are equal → without ALiBi,
        // attention weights would be uniform over visible positions.
        let q = vec![1.0f32; seq_len * num_heads * head_dim];
        let k = vec![1.0f32; kv_len * num_kv_heads * head_dim];
        let v: Vec<f32> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| i as f32)
            .collect();

        // Without ALiBi
        let baseline = CpuAttentionGgml;
        let mut out_baseline = vec![0.0f32; seq_len * num_heads * head_dim];
        baseline
            .forward(
                &q,
                &k,
                &v,
                &mut out_baseline,
                1,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
            .unwrap();

        // With ALiBi (max_bias=8)
        let alibi = CpuAttentionAlibi::new(num_heads, 8.0);
        let mut out_alibi = vec![0.0f32; seq_len * num_heads * head_dim];
        alibi
            .forward(
                &q,
                &k,
                &v,
                &mut out_alibi,
                1,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
            .unwrap();

        // ALiBi should shift attention toward more recent tokens (higher positions)
        // so output should differ from uniform baseline
        let max_diff: f32 = out_baseline
            .iter()
            .zip(&out_alibi)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff > 0.01,
            "ALiBi should change attention output (max_diff={max_diff})"
        );
    }

    #[test]
    fn test_alibi_recency_bias() {
        // ALiBi adds slope*(k_pos - q_pos) which is negative for past tokens,
        // more negative the further away. This biases attention toward recent tokens.
        let head_dim = 4;
        let seq_len = 1;
        let kv_len = 8;
        let num_heads = 1;
        let num_kv_heads = 1;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = vec![0.5f32; seq_len * num_heads * head_dim];
        let k = vec![0.5f32; kv_len * num_kv_heads * head_dim];
        // V = [0,0,0,0, 1,1,1,1, ...] — later tokens have higher values
        let v: Vec<f32> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| (i / head_dim) as f32)
            .collect();

        // Strong ALiBi slope
        let alibi = CpuAttentionAlibi::new(num_heads, 8.0);
        let mut out_alibi = vec![0.0f32; seq_len * num_heads * head_dim];
        alibi
            .forward(
                &q,
                &k,
                &v,
                &mut out_alibi,
                1,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
            .unwrap();

        // Without ALiBi, uniform weights → mean value ≈ 3.5
        // With ALiBi, recency bias → output should be > 3.5 (biased toward later tokens)
        let mean_val = out_alibi.iter().sum::<f32>() / out_alibi.len() as f32;
        assert!(
            mean_val > 3.5,
            "ALiBi should bias toward recent tokens, got mean={mean_val}"
        );
    }

    #[test]
    fn test_masked_attention_matches_causal() {
        let head_dim = 4;
        let seq_len = 3;
        let kv_len = 3;
        let num_heads = 1;
        let num_kv_heads = 1;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| (i as f32) * 0.1 + 0.1)
            .collect();
        let k = q.clone();
        let v: Vec<f32> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.2)
            .collect();

        // Causal output from CpuAttentionGgml
        let causal = CpuAttentionGgml;
        let mut out_causal = vec![0.0f32; seq_len * num_heads * head_dim];
        causal
            .forward(
                &q,
                &k,
                &v,
                &mut out_causal,
                1,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
            .unwrap();

        // Build equivalent causal mask: lower-triangular = 0, upper = -inf
        let mut mask = vec![0.0f32; seq_len * kv_len];
        for s in 0..seq_len {
            for kv_s in 0..kv_len {
                if kv_s > s {
                    mask[s * kv_len + kv_s] = f32::NEG_INFINITY;
                }
            }
        }

        let masked = CpuAttentionMasked;
        let mut out_masked = vec![0.0f32; seq_len * num_heads * head_dim];
        masked
            .forward_with_mask(
                &q,
                &k,
                &v,
                &mut out_masked,
                1,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                &mask,
            )
            .unwrap();

        for i in 0..out_causal.len() {
            assert!(
                (out_causal[i] - out_masked[i]).abs() < 1e-5,
                "causal vs masked differ at {i}: causal={}, masked={}",
                out_causal[i],
                out_masked[i]
            );
        }
    }

    #[test]
    fn test_masked_attention_bidirectional() {
        // Mask with all zeros = bidirectional (no masking)
        let head_dim = 4;
        let seq_len = 3;
        let kv_len = 3;
        let num_heads = 1;
        let num_kv_heads = 1;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let k = q.clone();
        let v: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        // All-zero mask = attend everywhere
        let mask = vec![0.0f32; seq_len * kv_len];
        let masked = CpuAttentionMasked;
        let mut out = vec![0.0f32; seq_len * num_heads * head_dim];
        masked
            .forward_with_mask(
                &q,
                &k,
                &v,
                &mut out,
                1,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                &mask,
            )
            .unwrap();

        // Token 0 with bidirectional mask can see all 3 tokens,
        // causal would only see token 0. So token 0's output should reflect all V rows.
        // With causal, out[0..4] = v[0..4] = [1,2,3,4] exactly.
        // With bidirectional, it's a weighted mix of all V rows.
        // Use CpuAttention (sequential, not parallel) to avoid thread-pool non-determinism.
        let causal = CpuAttention;
        let mut out_causal = vec![0.0f32; seq_len * num_heads * head_dim];
        causal
            .forward(
                &q,
                &k,
                &v,
                &mut out_causal,
                1,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            )
            .unwrap();

        // Token 0 should differ (bidirectional sees more tokens)
        let diff_tok0: f32 = (0..head_dim)
            .map(|d| (out[d] - out_causal[d]).abs())
            .fold(0.0, f32::max);
        assert!(
            diff_tok0 > 0.1,
            "bidirectional mask should differ from causal for token 0 (diff={diff_tok0})"
        );

        // Token 2 (last) should match exactly: causal and bidirectional both see all 3 tokens.
        let off = 2 * head_dim;
        let diff_tok2: f32 = (0..head_dim)
            .map(|d| (out[off + d] - out_causal[off + d]).abs())
            .fold(0.0, f32::max);
        assert!(
            diff_tok2 < 1e-4,
            "last token should match causal (diff={diff_tok2})"
        );
    }

    #[test]
    fn test_softcapping_reduces_extreme_scores() {
        let cap = 5.0f32;
        let head_dim = 4;
        let seq_len = 1;
        let kv_len = 2;

        // Large Q and K values to produce extreme scores
        let q = vec![20.0, 20.0, 20.0, 20.0];
        let k = vec![20.0, 20.0, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        let scale = 1.0 / (head_dim as f32).sqrt();
        // Normal score0 = dot(q,k0)*scale = 1600*0.5 = 800
        // Normal score1 = dot(q,k1)*scale = 80*0.5 = 40
        // Huge gap => normal attention puts ~100% on key0
        // Softcap: tanh(800/5)*5 ≈ 5.0, tanh(40/5)*5 ≈ 5.0 (both saturated)
        // So softcapped attention is ~50/50 between the two keys

        let mut out_normal = vec![0.0f32; head_dim];
        CpuAttentionGgml
            .forward(
                &q,
                &k,
                &v,
                &mut out_normal,
                1,
                seq_len,
                kv_len,
                1,
                1,
                head_dim,
                scale,
            )
            .unwrap();

        let mut out_softcap = vec![0.0f32; head_dim];
        CpuAttentionSoftcap::new(cap)
            .forward(
                &q,
                &k,
                &v,
                &mut out_softcap,
                1,
                seq_len,
                kv_len,
                1,
                1,
                head_dim,
                scale,
            )
            .unwrap();

        let diff: f32 = (0..head_dim)
            .map(|d| (out_normal[d] - out_softcap[d]).abs())
            .fold(0.0, f32::max);
        assert!(
            diff > 0.01,
            "softcapping should change attention output (diff={diff})"
        );

        // Softcapped output should be closer to the average of the two V rows
        // than the normal output (which heavily favors V[0])
        let avg_v0 = 0.5;
        let normal_dist = (out_normal[0] - avg_v0).abs();
        let softcap_dist = (out_softcap[0] - avg_v0).abs();
        assert!(softcap_dist < normal_dist,
            "softcapped output should be more uniform: normal_dist={normal_dist}, softcap_dist={softcap_dist}");
    }

    #[test]
    fn test_softcapping_tanh_clamps_extreme_values() {
        let cap = 10.0f32;
        let large_val = 1000.0f32;
        let result = (large_val / cap).tanh() * cap;
        assert!(
            (result - cap).abs() < 0.01,
            "tanh(x/cap)*cap should approach cap for large x: got {result}"
        );

        let small_val = 0.1f32;
        let result_small = (small_val / cap).tanh() * cap;
        assert!((result_small - small_val).abs() < 0.01,
            "tanh(x/cap)*cap should approximate x for small x: got {result_small}, expected ~{small_val}");
    }
}

#[cfg(feature = "cuda")]
pub type CudaAttention = crate::cutile_flash_attention::CutileFlashAttention;
