//! Direct Rust translation of llama.cpp ggml/src/ggml-cpu/ops.cpp lines 4001–8000.
//!
//! Every function carries a doc-comment referencing the original C source line.
//! Signatures use raw f32/i8 slices with explicit size parameters instead of
//! ggml_tensor; the GGML threading (ith/nth) and wdata workspace are stripped —
//! callers manage parallelism externally.

// ── RMS norm backward ────────────────────────────────────────────────────────

/// Backward pass for RMS normalisation.
/// Direct translation of `ggml_compute_forward_rms_norm_back_f32` (ops.cpp line 4031).
///
/// `dz`  — gradient from the forward pass output (ne0 elements)
/// `x`   — original input to the forward RMS-norm (ne0 elements)
/// `dx`  — output gradient w.r.t. the input (ne0 elements)
/// `eps` — epsilon used in the forward pass
pub fn rms_norm_back_f32(dz: &[f32], x: &[f32], dx: &mut [f32], eps: f32) {
    let n = dz.len();
    assert_eq!(x.len(), n);
    assert_eq!(dx.len(), n);

    let mut sum_xx: f64 = 0.0;
    let mut sum_xdz: f64 = 0.0;
    for i in 0..n {
        sum_xx += (x[i] * x[i]) as f64;
        sum_xdz += (x[i] * dz[i]) as f64;
    }

    let mean_eps = sum_xx as f32 / n as f32 + eps;
    let sum_eps = sum_xx as f32 + eps * n as f32;
    let rrms = 1.0_f32 / mean_eps.sqrt();

    // dx[i] = (dz[i] + x[i] * (-sum_xdz / sum_eps)) * rrms
    let scale_x = -sum_xdz as f32 / sum_eps;
    for i in 0..n {
        dx[i] = (dz[i] + x[i] * scale_x) * rrms;
    }
}

// ── Group norm ────────────────────────────────────────────────────────────────

/// Group normalisation for a single 3-D slice.
/// Direct translation of `ggml_compute_forward_group_norm_f32` inner logic
/// (ops.cpp line 4208).
///
/// Layout: `data[i02][i01][i00]` stored row-major.
/// `ne00` — inner-most dimension (vector length)
/// `ne01` — second dimension (rows)
/// `ne02` — channel count (third dimension)
/// `n_groups` — number of normalisation groups (must divide ne02 evenly or the
///              last group is smaller)
/// `eps` — epsilon for numerical stability
pub fn group_norm_f32(
    src: &[f32],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
    ne02: usize,
    n_groups: usize,
    eps: f32,
) {
    assert_eq!(src.len(), ne00 * ne01 * ne02);
    assert_eq!(dst.len(), src.len());

    let n_channels_per_group = (ne02 + n_groups - 1) / n_groups;

    for g in 0..n_groups {
        let start = g * n_channels_per_group;
        let end = (start + n_channels_per_group).min(ne02);
        let step = end - start;

        // compute mean
        let mut sum: f64 = 0.0;
        for i02 in start..end {
            for i01 in 0..ne01 {
                let base = i02 * ne01 * ne00 + i01 * ne00;
                for i00 in 0..ne00 {
                    sum += src[base + i00] as f64;
                }
            }
        }
        let mean = sum as f32 / (ne00 * ne01 * step) as f32;

        // subtract mean → dst, accumulate variance
        let mut sum2: f64 = 0.0;
        for i02 in start..end {
            for i01 in 0..ne01 {
                let base = i02 * ne01 * ne00 + i01 * ne00;
                for i00 in 0..ne00 {
                    let v = src[base + i00] - mean;
                    dst[base + i00] = v;
                    sum2 += (v * v) as f64;
                }
            }
        }
        let variance = sum2 as f32 / (ne00 * ne01 * step) as f32;
        let scale = 1.0_f32 / (variance + eps).sqrt();

        // scale dst
        for i02 in start..end {
            for i01 in 0..ne01 {
                let base = i02 * ne01 * ne00 + i01 * ne00;
                for i00 in 0..ne00 {
                    dst[base + i00] *= scale;
                }
            }
        }
    }
}

// ── L2 norm ───────────────────────────────────────────────────────────────────

/// L2 normalisation of a flat row vector.
/// Direct translation of `ggml_compute_forward_l2_norm_f32` inner kernel
/// (ops.cpp line 4303).
///
/// `x`   — input slice (ne00 elements, contiguous)
/// `y`   — output slice (ne00 elements)
/// `eps` — minimum norm (max(||x||, eps) is used as divisor)
pub fn l2_norm_row_f32(x: &[f32], y: &mut [f32], eps: f32) {
    assert_eq!(x.len(), y.len());
    let mut sum: f64 = 0.0;
    for &xi in x.iter() {
        sum += (xi * xi) as f64;
    }
    let scale = 1.0_f32 / (sum as f32).sqrt().max(eps);
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = xi * scale;
    }
}

/// L2 normalisation over multiple rows.
/// Direct translation of the outer loop in `ggml_compute_forward_l2_norm_f32`
/// (ops.cpp line 4322).
pub fn l2_norm_f32(src: &[f32], dst: &mut [f32], ne00: usize, ne01: usize, eps: f32) {
    assert!(eps >= 0.0);
    assert_eq!(src.len(), ne00 * ne01);
    assert_eq!(dst.len(), src.len());
    for i01 in 0..ne01 {
        let s = &src[i01 * ne00..(i01 + 1) * ne00];
        let d = &mut dst[i01 * ne00..(i01 + 1) * ne00];
        l2_norm_row_f32(s, d, eps);
    }
}

// ── Outer product ─────────────────────────────────────────────────────────────

/// Outer product accumulation: `dst[i0] += src0[i0, i01] * src1[i1, i01]`.
/// Direct translation of `ggml_compute_forward_out_prod_f32` inner loop
/// (ops.cpp line 4371).
///
/// `ne0`  — number of rows in src0 / columns in dst row
/// `ne01` — shared contraction dimension
/// `ne1`  — number of rows in src1
/// `src0` — row-major [ne01, ne0]
/// `src1` — row-major [ne01, ne1]
/// `dst`  — row-major [ne1, ne0], **accumulated** (caller must zero before use)
pub fn out_prod_f32(src0: &[f32], src1: &[f32], dst: &mut [f32], ne0: usize, ne01: usize, ne1: usize) {
    assert_eq!(src0.len(), ne01 * ne0);
    assert_eq!(src1.len(), ne01 * ne1);
    assert_eq!(dst.len(), ne1 * ne0);

    for i01 in 0..ne01 {
        let s0 = &src0[i01 * ne0..(i01 + 1) * ne0];
        for i1 in 0..ne1 {
            let sv = src1[i1 * ne01 + i01]; // src1[i1, i01]
            let d = &mut dst[i1 * ne0..(i1 + 1) * ne0];
            for (di, s0i) in d.iter_mut().zip(s0.iter()) {
                *di += s0i * sv;
            }
        }
    }
}

/// Outer product of f16 src0 (converted to f32) and f32 src1.
/// Direct translation of `ggml_compute_forward_out_prod_f16_f32` (ops.cpp line 4581).
pub fn out_prod_f16_f32(
    src0_f16: &[u16], // raw f16 bits
    src1: &[f32],
    dst: &mut [f32],
    ne0: usize,
    ne01: usize,
    ne1: usize,
) {
    assert_eq!(src0_f16.len(), ne01 * ne0);
    assert_eq!(src1.len(), ne01 * ne1);
    assert_eq!(dst.len(), ne1 * ne0);

    for i01 in 0..ne01 {
        // convert f16 row to f32
        let row_f32: Vec<f32> = src0_f16[i01 * ne0..(i01 + 1) * ne0]
            .iter()
            .map(|&h| half::f16::from_bits(h).to_f32())
            .collect();
        for i1 in 0..ne1 {
            let sv = src1[i1 * ne01 + i01];
            let d = &mut dst[i1 * ne0..(i1 + 1) * ne0];
            for (di, s0i) in d.iter_mut().zip(row_f32.iter()) {
                *di += s0i * sv;
            }
        }
    }
}

// ── Scale ─────────────────────────────────────────────────────────────────────

/// Scale each element: `dst[i] = src[i] * s + b`.
/// Direct translation of `ggml_compute_forward_scale_f32` (ops.cpp line 4697).
///
/// When `b == 0` this is a pure multiply. When `b != 0` it is a fused
/// scale+bias (ggml `vec_mad1`). The function handles both in one pass.
pub fn scale_f32(src: &[f32], dst: &mut [f32], s: f32, b: f32) {
    assert_eq!(src.len(), dst.len());
    if b == 0.0 {
        for (d, x) in dst.iter_mut().zip(src.iter()) {
            *d = x * s;
        }
    } else {
        for (d, x) in dst.iter_mut().zip(src.iter()) {
            *d = x * s + b;
        }
    }
}

// ── Get-rows ──────────────────────────────────────────────────────────────────

/// Gather rows from a float32 embedding table.
/// Direct translation of `ggml_compute_forward_get_rows_f32` kernel body
/// (ops.cpp line 5105).
///
/// `table`  — row-major [n_vocab, nc], all rows contiguous
/// `indices`— row indices to gather (values in [0, n_vocab))
/// `dst`    — output [indices.len(), nc]
/// `nc`     — number of columns (embedding dimension)
/// `n_vocab`— number of rows in table
pub fn get_rows_f32(table: &[f32], indices: &[i32], dst: &mut [f32], nc: usize, n_vocab: usize) {
    assert_eq!(table.len(), n_vocab * nc);
    assert_eq!(dst.len(), indices.len() * nc);
    for (i, &row_idx) in indices.iter().enumerate() {
        assert!(row_idx >= 0 && (row_idx as usize) < n_vocab);
        let src_base = row_idx as usize * nc;
        let dst_base = i * nc;
        dst[dst_base..dst_base + nc].copy_from_slice(&table[src_base..src_base + nc]);
    }
}

/// Gather rows from an f16 embedding table, converting to f32.
/// Direct translation of `ggml_compute_forward_get_rows_f16` (ops.cpp line 5023).
pub fn get_rows_f16(
    table_f16: &[u16], // raw f16 bits, [n_vocab, nc]
    indices: &[i32],
    dst: &mut [f32],
    nc: usize,
    n_vocab: usize,
) {
    assert_eq!(table_f16.len(), n_vocab * nc);
    assert_eq!(dst.len(), indices.len() * nc);
    for (i, &row_idx) in indices.iter().enumerate() {
        assert!(row_idx >= 0 && (row_idx as usize) < n_vocab);
        let src_base = row_idx as usize * nc;
        let dst_base = i * nc;
        for j in 0..nc {
            dst[dst_base + j] = half::f16::from_bits(table_f16[src_base + j]).to_f32();
        }
    }
}

// ── Get-rows backward ─────────────────────────────────────────────────────────

/// Backward pass for get_rows (f32 src0).
/// Direct translation of `ggml_compute_forward_get_rows_back_f32` (ops.cpp line 5361).
///
/// Scatter-accumulates gradients: `dst[indices[i]] += src0[i]`.
/// `src0`    — upstream gradient rows, [nr, nc]
/// `indices` — which dst row each src0 row maps to (length nr)
/// `dst`     — accumulation buffer, pre-zeroed, [n_vocab, nc]
pub fn get_rows_back_f32(src0: &[f32], indices: &[i32], dst: &mut [f32], nc: usize) {
    let nr = indices.len();
    assert_eq!(src0.len(), nr * nc);
    for (i, &r) in indices.iter().enumerate() {
        let r = r as usize;
        let dst_base = r * nc;
        let src_base = i * nc;
        for j in 0..nc {
            dst[dst_base + j] += src0[src_base + j];
        }
    }
}

/// Backward pass for get_rows (f16 src0 → f32 accumulation).
/// Direct translation of `ggml_compute_forward_get_rows_back_f32_f16` (ops.cpp line 5328).
pub fn get_rows_back_f16(
    src0_f16: &[u16], // raw f16 bits, [nr, nc]
    indices: &[i32],
    dst: &mut [f32],
    nc: usize,
) {
    let nr = indices.len();
    assert_eq!(src0_f16.len(), nr * nc);
    for (i, &r) in indices.iter().enumerate() {
        let r = r as usize;
        let dst_base = r * nc;
        let src_base = i * nc;
        for j in 0..nc {
            dst[dst_base + j] += half::f16::from_bits(src0_f16[src_base + j]).to_f32();
        }
    }
}

// ── Diagonal ─────────────────────────────────────────────────────────────────

/// Build a diagonal matrix from a 1-D vector.
/// Direct translation of `ggml_compute_forward_diag_f32` (ops.cpp line 5436).
///
/// `src`   — input vector length `n`
/// `dst`   — output [n, n] row-major (must be zeroed by caller)
/// `n`     — size
pub fn diag_f32(src: &[f32], dst: &mut [f32], n: usize) {
    assert_eq!(src.len(), n);
    assert_eq!(dst.len(), n * n);
    dst.fill(0.0);
    for i in 0..n {
        dst[i * n + i] = src[i];
    }
}

// ── Diagonal mask ─────────────────────────────────────────────────────────────

/// Set above-diagonal entries to `value` (in-place or out-of-place).
/// Direct translation of `ggml_compute_forward_diag_mask_f32` inner loop
/// (ops.cpp line 5496).
///
/// `data`   — row-major [nz, nr, nc] in row-major order
/// `nr`     — number of rows per "z-slice"
/// `nc`     — number of columns (inner-most)
/// `nz`     — number of z-slices
/// `n_past` — columns up to `n_past` (inclusive) are left intact; columns
///            `n_past+1..j` (i.e. those above the diagonal + n_past offset)
///            are set to `value`
pub fn diag_mask_inplace_f32(data: &mut [f32], nr: usize, nc: usize, nz: usize, n_past: usize, value: f32) {
    assert_eq!(data.len(), nz * nr * nc);
    for k in 0..nz {
        for j in 0..nr {
            for i in n_past..nc {
                if i > n_past + j {
                    data[k * nr * nc + j * nc + i] = value;
                }
            }
        }
    }
}

/// Apply `diag_mask_inplace_f32` with value `-f32::INFINITY` (causal mask).
/// (ops.cpp line 5546)
pub fn diag_mask_inf_f32(data: &mut [f32], nr: usize, nc: usize, nz: usize, n_past: usize) {
    diag_mask_inplace_f32(data, nr, nc, nz, n_past, f32::NEG_INFINITY);
}

/// Apply `diag_mask_inplace_f32` with value `0` (zero mask).
/// (ops.cpp line 5564)
pub fn diag_mask_zero_f32(data: &mut [f32], nr: usize, nc: usize, nz: usize, n_past: usize) {
    diag_mask_inplace_f32(data, nr, nc, nz, n_past, 0.0);
}

// ── Softmax ───────────────────────────────────────────────────────────────────

/// Numerically-stable softmax over a single row.
/// Direct translation of the inner body of `ggml_compute_forward_soft_max_f32`
/// (ops.cpp line 5584).
///
/// `scale`     — multiply inputs by `scale` before softmax
/// `max_bias`  — ALiBi head-slope bias (0 = no ALiBi)
/// `n_head`    — total number of attention heads (for ALiBi computation)
/// `head_idx`  — zero-based head index (for ALiBi)
/// `x`         — logits input (ne00 elements)
/// `mask`      — optional additive mask (same length as `x`); may be `None`
/// `y`         — softmax output
pub fn softmax_row_f32(
    x: &[f32],
    mask: Option<&[f32]>,
    y: &mut [f32],
    scale: f32,
    max_bias: f32,
    n_head: u32,
    head_idx: u32,
) {
    let n = x.len();
    assert_eq!(y.len(), n);
    if let Some(m) = mask {
        assert_eq!(m.len(), n);
    }

    // ALiBi slope
    let n_head_log2 = 1u32 << (n_head as f32).log2().floor() as u32;
    let m0 = (2.0_f32).powf(-(max_bias) / n_head_log2 as f32);
    let m1 = (2.0_f32).powf(-(max_bias / 2.0) / n_head_log2 as f32);
    let slope = if max_bias > 0.0 {
        if head_idx < n_head_log2 {
            m0.powf(head_idx as f32 + 1.0)
        } else {
            m1.powf(2.0 * (head_idx - n_head_log2) as f32 + 1.0)
        }
    } else {
        1.0
    };

    // scaled + masked copy into y
    for i in 0..n {
        y[i] = x[i] * scale;
        if let Some(m) = mask {
            y[i] += slope * m[i];
        }
    }

    // find max for numerical stability
    let max = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // exp(y - max) and sum
    let mut sum: f64 = 0.0;
    for yi in y.iter_mut() {
        *yi = (*yi - max).exp();
        sum += *yi as f64;
    }

    // normalise
    let inv_sum = 1.0 / sum as f32;
    for yi in y.iter_mut() {
        *yi *= inv_sum;
    }
}

/// Backward pass for softmax.
/// Direct translation of `ggml_compute_forward_soft_max_ext_back_f32` kernel
/// (ops.cpp line 5717).
///
/// `dy` — upstream gradient (softmax output gradient)
/// `y`  — softmax forward-pass output
/// `dx` — gradient w.r.t. softmax input
/// `scale` — scale that was applied in the forward pass
pub fn softmax_back_row_f32(dy: &[f32], y: &[f32], dx: &mut [f32], scale: f32) {
    let n = dy.len();
    assert_eq!(y.len(), n);
    assert_eq!(dx.len(), n);

    // dot_y_dy = y · dy
    let dot_y_dy: f32 = y.iter().zip(dy.iter()).map(|(&yi, &di)| yi * di).sum();

    // dx = y * (dy - dot_y_dy) * scale
    for i in 0..n {
        dx[i] = y[i] * (dy[i] - dot_y_dy) * scale;
    }
}

// ── Clamp ─────────────────────────────────────────────────────────────────────

/// Element-wise clamp.
/// Direct translation of `ggml_compute_forward_clamp_f32` (ops.cpp line 5821).
pub fn clamp_f32(src: &[f32], dst: &mut [f32], min: f32, max: f32) {
    assert_eq!(src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = s.max(min).min(max);
    }
}

/// Element-wise clamp for f16 (converted through f32).
/// Direct translation of `ggml_compute_forward_clamp_f16` (ops.cpp line 5857).
pub fn clamp_f16(src_f16: &[u16], dst_f16: &mut [u16], min: f32, max: f32) {
    assert_eq!(src_f16.len(), dst_f16.len());
    for (d, &s) in dst_f16.iter_mut().zip(src_f16.iter()) {
        let v = half::f16::from_bits(s).to_f32().max(min).min(max);
        *d = half::f16::from_f32(v).to_bits();
    }
}

// ── RoPE helpers ──────────────────────────────────────────────────────────────

/// YaRN ramp function.
/// Direct translation of `rope_yarn_ramp` (ops.cpp line 5951).
#[inline]
pub fn rope_yarn_ramp(low: f32, high: f32, i0: i64) -> f32 {
    let y = (i0 as f32 / 2.0 - low) / (high - low).max(0.001);
    1.0 - y.min(1.0).max(0.0)
}

/// YaRN rotation angle computation.
/// Direct translation of `rope_yarn` (ops.cpp line 5958).
///
/// Returns `(cos_theta, sin_theta)`.
pub fn rope_yarn(
    theta_extrap: f32,
    freq_scale: f32,
    corr_dims: [f32; 2],
    i0: i64,
    ext_factor: f32,
    mscale: f32,
) -> (f32, f32) {
    let theta_interp = freq_scale * theta_extrap;
    let (theta, mscale) = if ext_factor != 0.0 {
        let ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        let theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        let mscale = mscale * (1.0 + 0.1 * (1.0 / freq_scale).ln());
        (theta, mscale)
    } else {
        (theta_interp, mscale)
    };
    (theta.cos() * mscale, theta.sin() * mscale)
}

/// Build the RoPE sin/cos cache for a single position.
/// Direct translation of `ggml_rope_cache_init` (ops.cpp line 5975).
///
/// `cache` must have length `ne0` (even). Elements at 2k and 2k+1 receive
/// cos and sin respectively.
pub fn rope_cache_init(
    theta_base: f32,
    freq_scale: f32,
    freq_factors: Option<&[f32]>,
    corr_dims: [f32; 2],
    ne0: usize,
    ext_factor: f32,
    mscale: f32,
    cache: &mut [f32],
    sin_sign: f32,
    theta_scale: f32,
) {
    assert_eq!(cache.len(), ne0);
    let mut theta = theta_base;
    let mut i0 = 0i64;
    while i0 < ne0 as i64 {
        let ff = freq_factors.map_or(1.0, |ff| ff[(i0 / 2) as usize]);
        let (cos_t, sin_t) = rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale);
        cache[i0 as usize] = cos_t;
        cache[i0 as usize + 1] = sin_t * sin_sign;
        theta *= theta_scale;
        i0 += 2;
    }
}

/// Apply rotary pairs using a precomputed cache.
/// Direct translation of `rotate_pairs<float>` (ops.cpp line 6063).
///
/// `src` and `dst` have length `n`; `n_offset` is the stride between the two
/// rotated elements (1 for NORMAL, n_dims/2 for NEoX/MRoPE).
/// `scale` controls the index into cache (1 for NORMAL, 2 for others).
pub fn rotate_pairs_f32(
    src: &[f32],
    dst: &mut [f32],
    cache: &[f32],
    n: usize,
    n_offset: usize,
    scale: usize,
) {
    assert!(dst.len() >= n_offset + n);
    let mut i0 = 0usize;
    while i0 < n {
        let ic = i0 / scale;
        let cos_theta = cache[i0];
        let sin_theta = cache[i0 + 1];
        let x0 = src[ic];
        let x1 = src[ic + n_offset];
        dst[ic] = x0 * cos_theta - x1 * sin_theta;
        dst[ic + n_offset] = x0 * sin_theta + x1 * cos_theta;
        i0 += 2;
    }
}

/// Apply rotary pairs for f16 values using a precomputed cache.
/// Direct translation of `rotate_pairs<ggml_fp16_t>` (ops.cpp line 6063).
pub fn rotate_pairs_f16(
    src_f16: &[u16],
    dst_f16: &mut [u16],
    cache: &[f32],
    n: usize,
    n_offset: usize,
    scale: usize,
) {
    let mut i0 = 0usize;
    while i0 < n {
        let ic = i0 / scale;
        let cos_theta = cache[i0];
        let sin_theta = cache[i0 + 1];
        let x0 = half::f16::from_bits(src_f16[ic]).to_f32();
        let x1 = half::f16::from_bits(src_f16[ic + n_offset]).to_f32();
        dst_f16[ic] = half::f16::from_f32(x0 * cos_theta - x1 * sin_theta).to_bits();
        dst_f16[ic + n_offset] = half::f16::from_f32(x0 * sin_theta + x1 * cos_theta).to_bits();
        i0 += 2;
    }
}

// ── RoPE mode constants (mirrors GGML_ROPE_TYPE_*) ───────────────────────────

/// RoPE rotation modes (mirror of `GGML_ROPE_TYPE_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeMode {
    Normal,  // GGML_ROPE_TYPE_NORMAL = 0
    NeoX,    // GGML_ROPE_TYPE_NEOX   = 2
    MRope,   // GGML_ROPE_TYPE_MROPE  = 8
    Vision,  // GGML_ROPE_TYPE_VISION = 24
    IMRope,  // GGML_ROPE_TYPE_IMROPE = 25 (qwen3vl)
}

// ── 1-D transposed convolution ────────────────────────────────────────────────

/// 1-D transposed convolution accumulation (single output row, f32 kernel).
/// Direct translation of the inner kernel in
/// `ggml_compute_forward_conv_transpose_1d_f32` (ops.cpp line 6376).
///
/// `kernel`     — permuted kernel: row `i1` has shape [ne02, ne00], stored as
///                `wdata[i1 * ne02 * ne00 ..]`
/// `src`        — permuted source: shape [ne10, ne11], stored as
///                `wdata_src[..]` (ne11 = channels, ne10 = time-steps)
/// `dst_row`    — output row (length at least `ne10 * s0 + ne00`)
/// `ne00`       — kernel width
/// `ne02`       — number of input channels (Cin)
/// `ne10`       — source time-steps
/// `ne11`       — source channels (must equal ne02)
/// `s0`         — output stride
pub fn conv_transpose_1d_row_f32(
    kernel: &[f32],  // shape [ne00, ne02] (row i1 slice)
    src: &[f32],     // shape [ne10, ne11]
    dst_row: &mut [f32],
    ne00: usize,
    ne02: usize,
    ne10: usize,
    ne11: usize,
    s0: usize,
) {
    assert_eq!(ne11, ne02);
    assert_eq!(kernel.len(), ne00 * ne02);
    assert_eq!(src.len(), ne10 * ne11);

    for i10 in 0..ne10 {
        let src_slice = &src[i10 * ne11..(i10 + 1) * ne11];
        for i00 in 0..ne00 {
            // dot product over ne02 channels
            let mut v = 0.0_f32;
            let kernel_row = &kernel[i00 * ne02..(i00 + 1) * ne02];
            for (ks, ss) in kernel_row.iter().zip(src_slice.iter()) {
                v += ks * ss;
            }
            dst_row[i10 * s0 + i00] += v;
        }
    }
}

// ── im2col ────────────────────────────────────────────────────────────────────

/// im2col for a 2-D convolution (f32 output).
/// Direct translation of `ggml_compute_forward_im2col_f32` (ops.cpp line 6490).
///
/// Output layout: `[N, OH, OW, IC*KH*KW]`.
pub fn im2col_f32(
    src: &[f32],
    dst: &mut [f32],
    n: usize,    // batch
    ic: usize,   // input channels
    ih: usize,   // input height
    iw: usize,   // input width
    kh: usize,   // kernel height
    kw: usize,   // kernel width
    oh: usize,   // output height
    ow: usize,   // output width
    s0: i32,     // stride width
    s1: i32,     // stride height
    p0: i32,     // padding width
    p1: i32,     // padding height
    d0: i32,     // dilation width
    d1: i32,     // dilation height
) {
    assert_eq!(src.len(), n * ic * ih * iw);
    assert_eq!(dst.len(), n * oh * ow * (ic * kh * kw));

    for in_ in 0..n {
        for ioh in 0..oh {
            for iow in 0..ow {
                let dst_base = (in_ * oh * ow + ioh * ow + iow) * (ic * kh * kw);
                for iic in 0..ic {
                    let src_plane_base = in_ * ic * ih * iw + iic * ih * iw;
                    for ikh in 0..kh {
                        for ikw in 0..kw {
                            let iiw = iow as i32 * s0 + ikw as i32 * d0 - p0;
                            let iih = ioh as i32 * s1 + ikh as i32 * d1 - p1;
                            let dst_idx = dst_base + iic * (kh * kw) + ikh * kw + ikw;
                            if iih < 0 || iih >= ih as i32 || iiw < 0 || iiw >= iw as i32 {
                                dst[dst_idx] = 0.0;
                            } else {
                                dst[dst_idx] = src[src_plane_base + iih as usize * iw + iiw as usize];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// im2col for a 2-D convolution (f16 output, f32 input).
/// Direct translation of `ggml_compute_forward_im2col_f16` (ops.cpp line 6566).
pub fn im2col_f16_from_f32(
    src: &[f32],
    dst_f16: &mut [u16],
    n: usize,
    ic: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    d0: i32,
    d1: i32,
) {
    assert_eq!(src.len(), n * ic * ih * iw);
    assert_eq!(dst_f16.len(), n * oh * ow * (ic * kh * kw));

    for in_ in 0..n {
        for ioh in 0..oh {
            for iow in 0..ow {
                let dst_base = (in_ * oh * ow + ioh * ow + iow) * (ic * kh * kw);
                for iic in 0..ic {
                    let src_plane_base = in_ * ic * ih * iw + iic * ih * iw;
                    for ikh in 0..kh {
                        for ikw in 0..kw {
                            let iiw = iow as i32 * s0 + ikw as i32 * d0 - p0;
                            let iih = ioh as i32 * s1 + ikh as i32 * d1 - p1;
                            let dst_idx = dst_base + iic * (kh * kw) + ikh * kw + ikw;
                            if iih < 0 || iih >= ih as i32 || iiw < 0 || iiw >= iw as i32 {
                                dst_f16[dst_idx] = half::f16::from_f32(0.0).to_bits();
                            } else {
                                let v = src[src_plane_base + iih as usize * iw + iiw as usize];
                                dst_f16[dst_idx] = half::f16::from_f32(v).to_bits();
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── im2col backward ───────────────────────────────────────────────────────────

/// Backward pass of im2col (col2im-style gradient accumulation).
/// Direct translation of `ggml_compute_forward_im2col_back_f32` (ops.cpp line 6665).
///
/// `grad_out` — gradient of im2col output, [N, OH, OW, IC*KH*KW]
/// `grad_in`  — gradient w.r.t. im2col input, [N, IC, IH, IW]
pub fn im2col_back_f32(
    grad_out: &[f32],
    grad_in: &mut [f32],
    n: usize,
    ic: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    d0: i32,
    d1: i32,
) {
    assert_eq!(grad_out.len(), n * oh * ow * (ic * kh * kw));
    assert_eq!(grad_in.len(), n * ic * ih * iw);

    for in_ in 0..n {
        for iic in 0..ic {
            for iih in 0..ih {
                for iiw in 0..iw {
                    let mut grad = 0.0_f32;
                    for ikh in 0..kh {
                        for ikw in 0..kw {
                            let tmpw = iiw as i32 + p0 - ikw as i32 * d0;
                            if tmpw % s0 != 0 {
                                continue;
                            }
                            let iow = tmpw / s0;

                            let tmph = iih as i32 + p1 - ikh as i32 * d1;
                            if tmph % s1 != 0 {
                                continue;
                            }
                            let ioh = tmph / s1;

                            if iow < 0 || iow >= ow as i32 || ioh < 0 || ioh >= oh as i32 {
                                continue;
                            }
                            let grad_in_base =
                                (in_ * oh * ow + ioh as usize * ow + iow as usize) * (ic * kh * kw);
                            grad += grad_out[grad_in_base + iic * (kh * kw) + ikh * kw + ikw];
                        }
                    }
                    grad_in[in_ * ic * ih * iw + iic * ih * iw + iih * iw + iiw] = grad;
                }
            }
        }
    }
}

// ── im2col 3D ─────────────────────────────────────────────────────────────────

/// im2col for 3-D convolution (f32 output, f32 input).
/// Direct translation of `ggml_compute_forward_im2col_3d_f32` (ops.cpp line 6851).
///
/// src layout: [N*IC, ID, IH, IW]
/// dst layout: [N*OD, OH, OW, IC * KD * KH * KW]
#[allow(clippy::too_many_arguments)]
pub fn im2col_3d_f32(
    src: &[f32],
    dst: &mut [f32],
    n: usize,
    ic: usize,
    id: usize,
    ih: usize,
    iw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    od: usize,
    oh: usize,
    ow: usize,
    s0: i32,
    s1: i32,
    s2: i32,
    p0: i32,
    p1: i32,
    p2: i32,
    d0: i32,
    d1: i32,
    d2: i32,
) {
    let ic_kd_kh_kw = ic * kd * kh * kw;
    let kd_kh_kw = kd * kh * kw;
    let kh_kw = kh * kw;
    let oh_ow = oh * ow;

    assert_eq!(src.len(), n * ic * id * ih * iw);
    assert_eq!(dst.len(), n * od * oh_ow * ic_kd_kh_kw);

    for in_ in 0..n {
        for iod in 0..od {
            for ioh in 0..oh {
                for iow in 0..ow {
                    let dst_base = (in_ * od * oh_ow + iod * oh_ow + ioh * ow + iow) * ic_kd_kh_kw;
                    for iic in 0..ic {
                        let src_plane_base = (in_ * ic + iic) * id * ih * iw;
                        for ikd in 0..kd {
                            for ikh in 0..kh {
                                for ikw in 0..kw {
                                    let siw = iow as i32 * s0 + ikw as i32 * d0 - p0;
                                    let sih = ioh as i32 * s1 + ikh as i32 * d1 - p1;
                                    let sid = iod as i32 * s2 + ikd as i32 * d2 - p2;
                                    let dst_idx = dst_base + iic * kd_kh_kw + ikd * kh_kw + ikh * kw + ikw;
                                    if sid < 0
                                        || sid >= id as i32
                                        || sih < 0
                                        || sih >= ih as i32
                                        || siw < 0
                                        || siw >= iw as i32
                                    {
                                        dst[dst_idx] = 0.0;
                                    } else {
                                        dst[dst_idx] = src[src_plane_base
                                            + sid as usize * ih * iw
                                            + sih as usize * iw
                                            + siw as usize];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── col2im 1D ─────────────────────────────────────────────────────────────────

/// col2im scatter-accumulation for 1-D signals.
/// Direct translation of `ggml_compute_forward_col2im_1d_impl<float>` (ops.cpp line 7017).
///
/// `col`    — column data layout [K*OC, T_in]
/// `dst`    — output signal layout [T_out, OC]  (must be pre-zeroed)
/// `k_oc`  — K * OC (number of rows in col)
/// `t_in`  — number of input time-steps
/// `t_out` — number of output time-steps
/// `oc`    — number of output channels
/// `k`     — kernel size
/// `s0`    — stride
/// `p0`    — padding
pub fn col2im_1d_f32(
    col: &[f32],
    dst: &mut [f32],
    k_oc: usize,
    t_in: usize,
    t_out: usize,
    oc: usize,
    k: usize,
    s0: usize,
    p0: usize,
) {
    assert_eq!(col.len(), k_oc * t_in);
    assert_eq!(dst.len(), t_out * oc);

    for oc_i in 0..oc {
        for t_out_i in 0..t_out {
            let t_abs = t_out_i + p0;

            let t_in_min = if t_abs + 1 > k {
                (t_abs - k + 1 + s0 - 1) / s0
            } else {
                0
            };
            let t_in_max = (t_abs / s0).min(t_in.saturating_sub(1));

            let mut sum = 0.0_f32;
            for t_in_i in t_in_min..=t_in_max {
                if t_abs < t_in_i * s0 {
                    continue;
                }
                let k_idx = t_abs - t_in_i * s0;
                if k_idx < k {
                    // col layout: [K*OC, T_in], element (oc_i*K + k_idx, t_in_i)
                    sum += col[(oc_i * k + k_idx) + t_in_i * k_oc];
                }
            }
            // dst layout: [T_out, OC], element (t_out_i, oc_i)
            dst[t_out_i + oc_i * t_out] = sum;
        }
    }
}

// ── Depthwise 2-D convolution ─────────────────────────────────────────────────

/// Depthwise 2-D convolution, WHCN layout (channels-last in channel axis).
/// Direct translation of `ggml_compute_forward_conv_2d_dw_whcn` (ops.cpp line 7603).
///
/// `src`    — [N, C, IH, IW] stored as `src[batch * c * ih * iw + c_i * ih * iw + ..]`
/// `kernel` — [C, KH, KW] stored as `kernel[c_i * kh * kw + ..]`
/// `dst`    — [N, C, OH, OW] same layout
pub fn conv_2d_dw_whcn(
    src: &[f32],
    kernel: &[f32],
    dst: &mut [f32],
    channels: usize,
    batch: usize,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    knl_w: usize,
    knl_h: usize,
    stride_x: i32,
    stride_y: i32,
    pad_x: i32,
    pad_y: i32,
    dilation_x: i32,
    dilation_y: i32,
) {
    assert_eq!(src.len(), batch * channels * src_h * src_w);
    assert_eq!(kernel.len(), channels * knl_h * knl_w);
    assert_eq!(dst.len(), batch * channels * dst_h * dst_w);

    let n = channels * batch;
    for i in 0..n {
        let c_i = i % channels;
        let knl_offset = c_i * knl_w * knl_h;
        let src_base = i * src_w * src_h;
        let dst_base = i * dst_w * dst_h;

        for dst_y in 0..dst_h {
            for dst_x in 0..dst_w {
                let mut sum = 0.0_f32;
                for knl_y in 0..knl_h {
                    let src_y = dst_y as i32 * stride_y + knl_y as i32 * dilation_y - pad_y;
                    if src_y < 0 || src_y >= src_h as i32 {
                        continue;
                    }
                    for knl_x in 0..knl_w {
                        let src_x = dst_x as i32 * stride_x + knl_x as i32 * dilation_x - pad_x;
                        if src_x < 0 || src_x >= src_w as i32 {
                            continue;
                        }
                        sum += kernel[knl_offset + knl_y * knl_w + knl_x]
                            * src[src_base + src_y as usize * src_w + src_x as usize];
                    }
                }
                dst[dst_base + dst_y * dst_w + dst_x] = sum;
            }
        }
    }
}

/// Depthwise 2-D convolution, CWHN layout (channels-first in the outer axis).
/// Direct translation of `ggml_compute_forward_conv_2d_dw_cwhn` (ops.cpp line 7520).
///
/// `src`    — [N, OH, OW, C] where the batch and spatial dims come first
/// `kernel` — [KH, KW, C] stored as `kernel[(knl_y * knl_w + knl_x) * channels + c_i]`
/// `dst`    — same shape as src
pub fn conv_2d_dw_cwhn(
    src: &[f32],
    kernel: &[f32],
    dst: &mut [f32],
    channels: usize,
    batch: usize,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    knl_w: usize,
    knl_h: usize,
    stride_x: i32,
    stride_y: i32,
    pad_x: i32,
    pad_y: i32,
    dilation_x: i32,
    dilation_y: i32,
) {
    assert_eq!(src.len(), batch * dst_h * dst_w * channels);
    assert_eq!(kernel.len(), knl_h * knl_w * channels);
    assert_eq!(dst.len(), src.len());

    let rows_total = dst_h * batch;
    for row in 0..rows_total {
        let dst_y = row % dst_h;
        let src_data_batch = (row / dst_h) * src_w * src_h * channels;
        for dst_x in 0..dst_w {
            let dst_data = &mut dst[(row * dst_w + dst_x) * channels..][..channels];
            let src_y_base = dst_y as i32 * stride_y - pad_y;
            let src_x_base = dst_x as i32 * stride_x - pad_x;

            for c_i in 0..channels {
                let mut sum = 0.0_f32;
                for knl_y in 0..knl_h {
                    let src_y = src_y_base + knl_y as i32 * dilation_y;
                    if src_y < 0 || src_y >= src_h as i32 {
                        continue;
                    }
                    for knl_x in 0..knl_w {
                        let src_x = src_x_base + knl_x as i32 * dilation_x;
                        if src_x < 0 || src_x >= src_w as i32 {
                            continue;
                        }
                        let k = kernel[(knl_y * knl_w + knl_x) * channels + c_i];
                        let s = src[src_data_batch + (src_y as usize * src_w + src_x as usize) * channels + c_i];
                        sum += k * s;
                    }
                }
                dst_data[c_i] = sum;
            }
        }
    }
}

// ── 1-D pooling ──────────────────────────────────────────────────────────────

/// Pool operation selector (mirrors `ggml_op_pool`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolOp {
    Avg,
    Max,
}

/// 1-D average / max pooling over a single row.
/// Direct translation of `ggml_compute_forward_pool_1d_ksp` inner loop
/// (ops.cpp line 7683).
///
/// `src_row` — input row length `iw`
/// `dst_row` — output row length `ow`; `ow = (iw + 2*p - k) / s + 1`
pub fn pool_1d_row_f32(
    src_row: &[f32],
    dst_row: &mut [f32],
    k: usize,
    s: usize,
    p: i32,
    op: PoolOp,
) {
    let iw = src_row.len() as i32;
    for ow in 0..dst_row.len() {
        let base = ow as i32 * s as i32 - p;
        let mut res = match op {
            PoolOp::Avg => 0.0_f32,
            PoolOp::Max => f32::NEG_INFINITY,
        };
        let mut count = 0i32;
        for ki in 0..k as i32 {
            let j = base + ki;
            if j < 0 || j >= iw {
                continue;
            }
            let v = src_row[j as usize];
            match op {
                PoolOp::Avg => {
                    res += v;
                    count += 1;
                }
                PoolOp::Max => {
                    if v > res {
                        res = v;
                    }
                }
            }
        }
        dst_row[ow] = match op {
            PoolOp::Avg => {
                if count > 0 {
                    res / count as f32
                } else {
                    0.0
                }
            }
            PoolOp::Max => res,
        };
    }
}

/// 2-D average / max pooling over a single plane.
/// Direct translation of `ggml_compute_forward_pool_2d` inner loops
/// (ops.cpp line 7769).
///
/// `src_plane` — [IH, IW] row-major
/// `dst_plane` — [OH, OW] row-major
pub fn pool_2d_plane_f32(
    src_plane: &[f32],
    dst_plane: &mut [f32],
    iw: usize,
    ih: usize,
    pw: usize,  // output width
    ph: usize,  // output height
    k0: usize,
    k1: usize,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    op: PoolOp,
) {
    let ka = (k0 * k1) as f32;
    for oy in 0..ph {
        for ox in 0..pw {
            let ix = ox as i32 * s0 - p0;
            let iy = oy as i32 * s1 - p1;
            let mut res = match op {
                PoolOp::Avg => 0.0_f32,
                PoolOp::Max => f32::NEG_INFINITY,
            };
            for ky in 0..k1 as i32 {
                if iy + ky < 0 || iy + ky >= ih as i32 {
                    continue;
                }
                let srow = &src_plane[(iy + ky) as usize * iw..];
                for kx in 0..k0 as i32 {
                    let j = ix + kx;
                    if j < 0 || j >= iw as i32 {
                        continue;
                    }
                    let v = srow[j as usize];
                    match op {
                        PoolOp::Avg => res += v,
                        PoolOp::Max => {
                            if v > res {
                                res = v;
                            }
                        }
                    }
                }
            }
            dst_plane[oy * pw + ox] = match op {
                PoolOp::Avg => res / ka,
                PoolOp::Max => res,
            };
        }
    }
}

/// 2-D max-pool backward pass.
/// Direct translation of `ggml_compute_forward_pool_2d_back` (ops.cpp line 7856).
///
/// For max pooling: scatters gradient to the location of the argmax.
/// For avg pooling: distributes gradient evenly over the pooling window.
///
/// `grad_out`   — upstream gradient [OH, OW]
/// `fwd_dst`    — forward-pass output (needed for max argmax) [IH, IW]
/// `grad_in`    — output gradient [IH, IW] (pre-zeroed)
pub fn pool_2d_back_plane_f32(
    grad_out: &[f32],
    fwd_dst: &[f32],
    grad_in: &mut [f32],
    iw: usize,
    ih: usize,
    pw: usize,
    ph: usize,
    k0: usize,
    k1: usize,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    op: PoolOp,
) {
    let ka = (k0 * k1) as f32;
    for oy in 0..ph {
        for ox in 0..pw {
            let grad0 = grad_out[oy * pw + ox];
            let ix = ox as i32 * s0 - p0;
            let iy = oy as i32 * s1 - p1;

            match op {
                PoolOp::Max => {
                    // find argmax in window
                    let mut maxval = f32::NEG_INFINITY;
                    let mut kxmax = -1i32;
                    let mut kymax = -1i32;
                    for ky in 0..k1 as i32 {
                        if iy + ky < 0 || iy + ky >= ih as i32 {
                            continue;
                        }
                        for kx in 0..k0 as i32 {
                            let j = ix + kx;
                            if j < 0 || j >= iw as i32 {
                                continue;
                            }
                            let val = fwd_dst[(iy + ky) as usize * iw + j as usize];
                            if val > maxval {
                                maxval = val;
                                kxmax = kx;
                                kymax = ky;
                            }
                        }
                    }
                    if kxmax >= 0 && kymax >= 0 {
                        grad_in[(iy + kymax) as usize * iw + (ix + kxmax) as usize] += grad0;
                    }
                }
                PoolOp::Avg => {
                    let grad = grad0 / ka;
                    for ky in 0..k1 as i32 {
                        if iy + ky < 0 || iy + ky >= ih as i32 {
                            continue;
                        }
                        for kx in 0..k0 as i32 {
                            let j = ix + kx;
                            if j < 0 || j >= iw as i32 {
                                continue;
                            }
                            grad_in[(iy + ky) as usize * iw + j as usize] += grad;
                        }
                    }
                }
            }
        }
    }
}

// ── Upscale ───────────────────────────────────────────────────────────────────

/// Nearest-neighbour upscale for a 4-D tensor.
/// Direct translation of `ggml_compute_forward_upscale_f32` nearest-neighbour
/// branch (ops.cpp line 7978).
///
/// `src`  — [ne00, ne01, ne02, ne03]
/// `dst`  — [ne0, ne1, ne2, ne3]
/// Scale factors: `sf_i = ne_i / ne0i`.
pub fn upscale_nearest_f32(
    src: &[f32],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
    ne02: usize,
    ne03: usize,
    ne0: usize,
    ne1: usize,
    ne2: usize,
    ne3: usize,
) {
    assert_eq!(src.len(), ne00 * ne01 * ne02 * ne03);
    assert_eq!(dst.len(), ne0 * ne1 * ne2 * ne3);

    for i3 in 0..ne3 {
        let i03 = (i3 * ne03 / ne3).min(ne03 - 1);
        for i2 in 0..ne2 {
            let i02 = (i2 * ne02 / ne2).min(ne02 - 1);
            for i1 in 0..ne1 {
                let i01 = (i1 * ne01 / ne1).min(ne01 - 1);
                for i0 in 0..ne0 {
                    let i00 = (i0 * ne00 / ne0).min(ne00 - 1);
                    dst[i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0 + i0] =
                        src[i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00 + i00];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_back_identity() {
        // When x = [1, 0, 0] and dz = [1, 0, 0], the backward should be consistent.
        let x = vec![1.0_f32, 0.0, 0.0];
        let dz = vec![1.0_f32, 0.0, 0.0];
        let mut dx = vec![0.0_f32; 3];
        rms_norm_back_f32(&dz, &x, &mut dx, 1e-5);
        // dx should be finite
        for &v in &dx {
            assert!(v.is_finite(), "dx should be finite, got {v}");
        }
    }

    #[test]
    fn test_l2_norm_unit_vector() {
        let x = vec![3.0_f32, 4.0, 0.0];
        let mut y = vec![0.0_f32; 3];
        l2_norm_row_f32(&x, &mut y, 1e-12);
        let norm: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm should be 1, got {norm}");
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut y = vec![0.0_f32; 4];
        softmax_row_f32(&x, None, &mut y, 1.0, 0.0, 1, 0);
        let sum: f32 = y.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1, got {sum}");
    }

    #[test]
    fn test_clamp_bounds() {
        let src = vec![-2.0_f32, 0.5, 3.0];
        let mut dst = vec![0.0_f32; 3];
        clamp_f32(&src, &mut dst, -1.0, 1.0);
        assert_eq!(dst, vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn test_diag_f32() {
        let src = vec![1.0_f32, 2.0, 3.0];
        let mut dst = vec![0.0_f32; 9];
        diag_f32(&src, &mut dst, 3);
        assert_eq!(dst[0], 1.0);
        assert_eq!(dst[4], 2.0);
        assert_eq!(dst[8], 3.0);
        assert_eq!(dst[1], 0.0);
    }

    #[test]
    fn test_get_rows_f32_basic() {
        // 3 vocab, 4 dims
        let table = vec![
            1.0_f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        let indices = vec![2_i32, 0];
        let mut dst = vec![0.0_f32; 8];
        get_rows_f32(&table, &indices, &mut dst, 4, 3);
        assert_eq!(&dst[0..4], &[9.0, 10.0, 11.0, 12.0]);
        assert_eq!(&dst[4..8], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pool_1d_avg() {
        let src = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut dst = vec![0.0_f32; 2];
        pool_1d_row_f32(&src, &mut dst, 2, 2, 0, PoolOp::Avg);
        assert!((dst[0] - 1.5).abs() < 1e-6);
        assert!((dst[1] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_pool_1d_max() {
        let src = vec![1.0_f32, 3.0, 2.0, 4.0];
        let mut dst = vec![0.0_f32; 2];
        pool_1d_row_f32(&src, &mut dst, 2, 2, 0, PoolOp::Max);
        assert!((dst[0] - 3.0).abs() < 1e-6);
        assert!((dst[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_f32_no_bias() {
        let src = vec![1.0_f32, 2.0, 3.0];
        let mut dst = vec![0.0_f32; 3];
        scale_f32(&src, &mut dst, 2.0, 0.0);
        assert_eq!(dst, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_scale_f32_with_bias() {
        let src = vec![1.0_f32, 2.0, 3.0];
        let mut dst = vec![0.0_f32; 3];
        scale_f32(&src, &mut dst, 2.0, 1.0);
        assert_eq!(dst, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_rope_yarn_ramp() {
        // at low boundary: ramp = 1 - 0 = 1
        let r = rope_yarn_ramp(2.0, 4.0, 4); // i0/2=2 → y=(2-2)/(4-2)=0 → 1-0=1
        assert!((r - 1.0).abs() < 1e-6);
        // at high boundary: ramp = 0
        let r2 = rope_yarn_ramp(2.0, 4.0, 8); // i0/2=4 → y=(4-2)/2=1 → 1-1=0
        assert!((r2 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_im2col_f32_identity() {
        // 1x1 kernel, stride=1, no pad → output equals input
        let src = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut dst = vec![0.0_f32; 4];
        im2col_f32(&src, &mut dst, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1);
        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_upscale_nearest_2x() {
        let src = vec![1.0_f32, 2.0, 3.0, 4.0]; // 2x2
        let mut dst = vec![0.0_f32; 16]; // 4x4
        upscale_nearest_f32(&src, &mut dst, 2, 2, 1, 1, 4, 4, 1, 1);
        // top-left 2x2 should be 1.0
        assert_eq!(dst[0], 1.0);
        assert_eq!(dst[1], 1.0);
        assert_eq!(dst[4], 1.0);
        assert_eq!(dst[5], 1.0);
        // top-right 2x2 should be 2.0
        assert_eq!(dst[2], 2.0);
    }
}
