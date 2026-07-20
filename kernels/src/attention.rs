pub use crate::Attention;
use crate::Result;

#[cfg(feature = "cuda")]
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

                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = crate::portable_math::portable_exp(*s - max_score);
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }

                let o_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for (kv_s, &score) in scores.iter().enumerate() {
                        let v_offset =
                            ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim + d;
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
    q: &[f32], k: &[f32], v: &[f32], output: &mut [f32],
    _batch: usize, seq_len: usize, kv_len: usize,
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    scale: f32, window: Option<usize>,
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
            for &sc in &scores[..kv_len] { if sc > max_score { max_score = sc; } }

            let mut sum: f64 = 0.0;
            for sc in &mut scores[..kv_len] {
                let val = crate::portable_math::ggml_exp(*sc - max_score);
                sum += val as f64;
                *sc = val;
            }
            let inv_sum = (1.0 / sum) as f32;
            for sc in &mut scores[..kv_len] { *sc *= inv_sum; }

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
                    if s > max_score { max_score = s; }
                }

                // C: ggml_vec_soft_max_f32 — scalar path: expf + f64 sum
                let mut sum: f64 = 0.0;
                for s in &mut scores {
                    // C: float val = expf(x[i] - max);
                    let val = crate::portable_math::ggml_exp(*s - max_score);
                    // C: sum += (ggml_float)val;
                    sum += val as f64;
                    *s = val;
                }

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
                        let v_offset =
                            ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim + d;
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
                        let k_offset =
                            ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim;
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
                        let v_base =
                            ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim;
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
        for kv_s in 0..kv_len {
            let k_offset = (kv_s * num_kv_heads + kv_h) * head_dim;
            let k_vec = &k[k_offset..k_offset + head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_vec[d] * k_vec[d];
            }
            scores[kv_s] = dot * scale;
        }

        let mut max_score = f32::NEG_INFINITY;
        for &s in scores.iter() {
            if s > max_score { max_score = s; }
        }

        let mut sum: f64 = 0.0;
        for s in scores.iter_mut() {
            let val = crate::portable_math::ggml_exp(*s - max_score);
            sum += val as f64;
            *s = val;
        }

        let inv_sum = (1.0 / sum) as f32;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }

        let o_offset = h * head_dim;
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for (kv_s, &score) in scores.iter().enumerate() {
                let v_offset = (kv_s * num_kv_heads + kv_h) * head_dim + d;
                val += score * v[v_offset];
            }
            output[o_offset + d] = val;
        }
    }
}

#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") {
            return unsafe {
                use std::arch::x86_64::*;
                let h = _mm_set1_epi16(bits as i16);
                _mm_cvtss_f32(_mm_cvtph_ps(h))
            };
        }
    }
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    if exp == 0 { f32::from_bits(sign << 31) }
    else if exp == 31 { f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13)) }
    else { f32::from_bits((sign << 31) | ((exp + 112) << 23) | (frac << 13)) }
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
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_vec[d] * f16_to_f32(k_f16[k_offset + d]);
            }
            scores[kv_s] = dot * scale;
        }

        let mut max_score = f32::NEG_INFINITY;
        for &s in scores.iter() {
            if s > max_score { max_score = s; }
        }

        let mut sum: f64 = 0.0;
        for s in scores.iter_mut() {
            let val = crate::portable_math::ggml_exp(*s - max_score);
            sum += val as f64;
            *s = val;
        }

        let inv_sum = (1.0 / sum) as f32;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }

        let o_offset = h * head_dim;
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for (kv_s, &score) in scores.iter().enumerate() {
                let v_offset = (kv_s * num_kv_heads + kv_h) * head_dim + d;
                val += score * f16_to_f32(v_f16[v_offset]);
            }
            output[o_offset + d] = val;
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
        for &sc in &scores[..kv_len] { if sc > max_score { max_score = sc; } }

        let mut sum: f64 = 0.0;
        for sc in &mut scores[..kv_len] {
            let val = crate::portable_math::ggml_exp(*sc - max_score);
            sum += val as f64;
            *sc = val;
        }
        let inv_sum = (1.0 / sum) as f32;
        for sc in &mut scores[..kv_len] { *sc *= inv_sum; }

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
        q: &[f32], k: &[f32], v: &[f32], output: &mut [f32],
        batch: usize, seq_len: usize, kv_len: usize,
        num_heads: usize, num_kv_heads: usize, head_dim: usize, scale: f32,
    ) -> Result<()> {
        if seq_len > 1 && batch == 1 {
            return optimized_attention_ggml(q, k, v, output, batch, seq_len, kv_len,
                                           num_heads, num_kv_heads, head_dim, scale, None);
        }
        naive_attention_ggml(q, k, v, output, batch, seq_len, kv_len,
                            num_heads, num_kv_heads, head_dim, scale, None)
    }

    fn forward_windowed(
        &self,
        q: &[f32], k: &[f32], v: &[f32], output: &mut [f32],
        batch: usize, seq_len: usize, kv_len: usize,
        num_heads: usize, num_kv_heads: usize, head_dim: usize, scale: f32,
        window_size: usize,
    ) -> Result<()> {
        naive_attention_ggml(q, k, v, output, batch, seq_len, kv_len,
                            num_heads, num_kv_heads, head_dim, scale, Some(window_size))
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
        q: &[f32], k: &[f32], v: &[f32], output: &mut [f32],
        batch: usize, seq_len: usize, kv_len: usize,
        num_heads: usize, num_kv_heads: usize, head_dim: usize, scale: f32,
    ) -> Result<()> {
        naive_attention(q, k, v, output, batch, seq_len, kv_len,
                       num_heads, num_kv_heads, head_dim, scale, None, dot_f32)
    }

    fn forward_windowed(
        &self,
        q: &[f32], k: &[f32], v: &[f32], output: &mut [f32],
        batch: usize, seq_len: usize, kv_len: usize,
        num_heads: usize, num_kv_heads: usize, head_dim: usize,
        scale: f32, window_size: usize,
    ) -> Result<()> {
        naive_attention(q, k, v, output, batch, seq_len, kv_len,
                       num_heads, num_kv_heads, head_dim, scale,
                       Some(window_size), dot_f32)
    }
}

#[cfg(feature = "cuda")]
pub struct CudaAttention {
    gpu: crate::gpu::GpuContext,
    module: crate::gpu::attention_kernel::kernels::LoadedModule,
    flash_module: Option<crate::gpu::flash_attention_kernel::kernels::LoadedModule>,
    decode_module: Option<crate::gpu::flash_decode_kernel::kernels::LoadedModule>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaAttention {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaAttention {}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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
                        &stream, LaunchConfig::for_num_elems(total as u32),
                        q, k, v, output,
                        batch as u32, seq_len as u32, kv_len as u32,
                        num_heads as u32, num_kv_heads as u32, head_dim as u32, scale,
                    )
                    .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
            }
            NaiveF16 => {
                let total = batch * seq_len * num_heads * head_dim;
                self.module
                    .attention_forward_f16(
                        &stream, LaunchConfig::for_num_elems(total as u32),
                        q, k, v, output,
                        batch as u32, seq_len as u32, kv_len as u32,
                        num_heads as u32, num_kv_heads as u32, head_dim as u32, scale,
                    )
                    .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
            }
            FlashF16 => {
                let flash = self.flash_module.as_ref().ok_or_else(|| {
                    crate::KernelError::InvalidConfig("flash module not loaded".into())
                })?;
                let config = LaunchConfig {
                    grid_dim: (num_heads as u32, (seq_len as u32).div_ceil(32), batch as u32),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                };
                flash
                    .flash_attention_forward(
                        &stream, config, q, k, v, output,
                        batch as u32, seq_len as u32, kv_len as u32,
                        num_heads as u32, num_kv_heads as u32, head_dim as u32, scale,
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
                        &stream, config, q, k, v, output,
                        batch as u32, seq_len as u32, kv_len as u32,
                        num_heads as u32, num_kv_heads as u32, head_dim as u32, scale,
                    )
                    .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
            }
            Auto => {
                return self.forward_device(
                    q, k, v, output, batch, seq_len, kv_len,
                    num_heads, num_kv_heads, head_dim, scale,
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

#[cfg(feature = "cuda")]
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
            attn.forward(&q, &k, &v, &mut out, 1, seq_len, kv_len,
                         num_heads, num_kv_heads, head_dim, scale).unwrap();
            out
        };

        let ref_out = run(&CpuAttention);
        let f16_naive = run(&CpuAttentionF16);
        let f32_flash = run(&CpuFlashAttention);
        let f16_flash = run(&CpuFlashAttentionF16);

        let max_diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
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
        assert!(d_f32_flash < 1e-5, "flash-f32 diverges from naive-f32: {d_f32_flash}");
        // f16 variants must match each other (same precision, different algorithm)
        assert!(d_f16_cross < 1e-5, "naive-f16 diverges from flash-f16: {d_f16_cross}");
        // f16 vs f32 should differ (proving f16 truncation has an effect) but not wildly
        assert!(d_f16_naive > 1e-6, "f16 truncation should have measurable effect");
        assert!(d_f16_naive < 0.1, "f16 divergence too large: {d_f16_naive}");
    }
}
