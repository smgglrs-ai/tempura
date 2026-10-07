//! Direct Rust translation of llama.cpp ggml/src/ggml-cpu/ops.cpp and
//! ggml/src/ggml-cpu/binary-ops.cpp.
//!
//! Every function carries a doc-comment referencing the original C source line.
//! Signatures use raw f32/i8 slices with explicit size parameters instead of
//! ggml_tensor; the GGML threading (ith/nth) and wdata workspace are stripped —
//! callers manage parallelism externally.

// ── Constants (vec.h) ────────────────────────────────────────────────────────

const GELU_COEF_A: f32 = 0.044715;
const GELU_QUICK_COEF: f32 = -1.702;
const SQRT_2_OVER_PI: f32 = 0.797_884_56;
const SQRT_2_INV: f32 = 0.707_106_78;

// ── Scalar activation helpers (vec.h) ────────────────────────────────────────

/// Direct translation of `ggml_gelu_f32` (vec.h line 968).
/// Uses the f16 lookup table — zero tanh/exp at inference time.
#[inline]
pub fn gelu_f32(x: f32) -> f32 {
    super::helpers::gelu_f32_lut(x)
}

/// Direct translation of `ggml_gelu_quick_f32` (vec.h line 1017).
/// Uses the f16 lookup table — zero exp at inference time.
#[inline]
pub fn gelu_quick_f32(x: f32) -> f32 {
    super::helpers::gelu_quick_f32_lut(x)
}

/// GELU with erf — `0.5*x*(1 + erf(x / sqrt(2)))`.
/// Derived from `ggml_vec_gelu_erf_f32` (vec.h line 1010).
#[inline]
pub fn gelu_erf_f32(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * SQRT_2_INV))
}

/// Direct translation of `ggml_silu_f32` (vec.h line 1046).
#[inline]
pub fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Direct translation of `ggml_vec_leaky_relu_f32` (vec.h line 929).
#[inline]
pub fn leaky_relu_f32(x: f32, ns: f32) -> f32 {
    (if x > 0.0 { x } else { 0.0 }) + ns * (if x < 0.0 { x } else { 0.0 })
}

// ── Element-wise vector ops ───────────────────────────────────────────────────

/// Apply GELU to every element.
/// Direct translation of `ggml_vec_gelu_f32` (vec.h line 1003).
///
/// # Panics
/// Panics if `x` and `y` have different lengths.
pub fn vec_gelu_f32(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = gelu_f32(*xi);
    }
}

/// Apply GELU-quick to every element.
/// Direct translation of `ggml_vec_gelu_quick_f32` (vec.h line 1038).
pub fn vec_gelu_quick_f32(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = gelu_quick_f32(*xi);
    }
}

/// Apply GELU-erf to every element.
/// Direct translation of `ggml_vec_gelu_erf_f32` (vec.h line 1010).
pub fn vec_gelu_erf_f32(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = gelu_erf_f32(*xi);
    }
}

/// Apply SiLU to every element.
/// Direct translation of `ggml_vec_silu_f32` (ops.cpp line 2611 callsite).
pub fn vec_silu_f32(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = silu_f32(*xi);
    }
}

/// Apply leaky ReLU to every element.
/// Direct translation of `ggml_vec_leaky_relu_f32` (vec.h line 929).
pub fn vec_leaky_relu_f32(y: &mut [f32], x: &[f32], ns: f32) {
    assert_eq!(y.len(), x.len());
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = leaky_relu_f32(*xi, ns);
    }
}

/// Element-wise addition: `z[i] = x[i] + y[i]`.
/// Direct translation of `op_add` / `vec_binary_op_contiguous` (binary-ops.cpp line 9, 26).
pub fn vec_add_f32(z: &mut [f32], x: &[f32], y: &[f32]) {
    assert_eq!(z.len(), x.len());
    assert_eq!(z.len(), y.len());
    for i in 0..z.len() {
        z[i] = x[i] + y[i];
    }
}

/// Element-wise subtraction: `z[i] = x[i] - y[i]`.
/// Direct translation of `op_sub` / `vec_binary_op_contiguous` (binary-ops.cpp line 13, 26).
pub fn vec_sub_f32(z: &mut [f32], x: &[f32], y: &[f32]) {
    assert_eq!(z.len(), x.len());
    assert_eq!(z.len(), y.len());
    for i in 0..z.len() {
        z[i] = x[i] - y[i];
    }
}

/// Element-wise multiplication: `z[i] = x[i] * y[i]`.
/// Direct translation of `op_mul` / `vec_binary_op_contiguous` (binary-ops.cpp line 17, 26).
pub fn vec_mul_f32(z: &mut [f32], x: &[f32], y: &[f32]) {
    assert_eq!(z.len(), x.len());
    assert_eq!(z.len(), y.len());
    for i in 0..z.len() {
        z[i] = x[i] * y[i];
    }
}

/// Element-wise division: `z[i] = x[i] / y[i]`.
/// Direct translation of `op_div` / `vec_binary_op_contiguous` (binary-ops.cpp line 21, 26).
pub fn vec_div_f32(z: &mut [f32], x: &[f32], y: &[f32]) {
    assert_eq!(z.len(), x.len());
    assert_eq!(z.len(), y.len());
    for i in 0..z.len() {
        z[i] = x[i] / y[i];
    }
}

// ── Normalization ─────────────────────────────────────────────────────────────

/// Layer normalization over a single contiguous row of length `n`.
///
/// Computes `(x - mean) / sqrt(var + eps)`.  Rows are contiguous; strides are
/// ignored (caller flattens).
///
/// Direct translation of `ggml_compute_forward_norm_f32` (ops.cpp line 3827).
///
/// # Panics
/// Panics if `x` and `out` have different lengths or `n` does not divide evenly.
pub fn layer_norm_rows_f32(x: &[f32], out: &mut [f32], n: usize, eps: f32) {
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % n, 0);
    let n_rows = x.len() / n;
    for row in 0..n_rows {
        let xr = &x[row * n..(row + 1) * n];
        let yr = &mut out[row * n..(row + 1) * n];
        let mean: f32 = xr.iter().sum::<f32>() / n as f32;
        let mut variance = 0.0f32;
        for i in 0..n {
            let v = xr[i] - mean;
            yr[i] = v;
            variance += v * v;
        }
        variance /= n as f32;
        let scale = 1.0 / (variance + eps).sqrt();
        for yi in yr.iter_mut() {
            *yi *= scale;
        }
    }
}

/// RMS normalization over a single contiguous row of length `n`.
///
/// Computes `x / sqrt(mean(x^2) + eps)`.
///
/// Direct translation of `ggml_compute_forward_rms_norm_f32<NONE>` (ops.cpp line 3924).
pub fn rms_norm_rows_f32(x: &[f32], out: &mut [f32], n: usize, eps: f32) {
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % n, 0);
    let n_rows = x.len() / n;
    for row in 0..n_rows {
        let xr = &x[row * n..(row + 1) * n];
        let yr = &mut out[row * n..(row + 1) * n];
        let sum_sq: f64 = xr.iter().map(|&v| (v * v) as f64).sum();
        let mean = sum_sq as f32 / n as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        for i in 0..n {
            yr[i] = xr[i] * scale;
        }
    }
}

/// Fused RMS norm + weight multiply: `out[i] = rms_norm(x)[i] * w[i]`.
///
/// Weight `w` has the same length as a row (`n`), and is broadcast across all
/// rows in `x`.
///
/// Direct translation of `ggml_compute_forward_rms_norm_f32<MUL>` (ops.cpp line 3971).
pub fn rms_norm_mul_rows_f32(x: &[f32], w: &[f32], out: &mut [f32], n: usize, eps: f32) {
    assert_eq!(w.len(), n);
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % n, 0);
    let n_rows = x.len() / n;
    for row in 0..n_rows {
        let xr = &x[row * n..(row + 1) * n];
        let yr = &mut out[row * n..(row + 1) * n];
        let sum_sq: f64 = xr.iter().map(|&v| (v * v) as f64).sum();
        let mean = sum_sq as f32 / n as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        for i in 0..n {
            yr[i] = xr[i] * scale * w[i];
        }
    }
}

// ── Softmax ───────────────────────────────────────────────────────────────────

/// Softmax over rows of a [n_rows × n] matrix.
///
/// `scale` multiplies each logit before softmax.  `mask` is an optional matrix
/// of the same shape (additive, already in the same dtype as x).  When `mask`
/// is `Some`, it must have the same shape as `x`.
///
/// Direct translation of `ggml_compute_forward_soft_max_f32` (ops.cpp line 5584),
/// simplified: no ALiBi bias, no sinks, no f16 mask.
pub fn soft_max_rows_f32(
    x: &[f32],
    out: &mut [f32],
    mask: Option<&[f32]>,
    n: usize,
    scale: f32,
) {
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % n, 0);
    if let Some(m) = mask {
        assert_eq!(m.len(), x.len());
    }
    let n_rows = x.len() / n;
    for row in 0..n_rows {
        let xr = &x[row * n..(row + 1) * n];
        let yr = &mut out[row * n..(row + 1) * n];

        // apply scale and mask into output buffer
        for i in 0..n {
            yr[i] = xr[i] * scale;
            if let Some(m) = mask {
                yr[i] += m[row * n + i];
            }
        }

        // find max for numerical stability
        let max = yr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // exp(x - max) and sum — AVX2 polynomial fast path
        let sum = crate::ggml::vec_llama::vec_soft_max_f32_inplace(yr, max);

        // normalise
        let inv_sum = (1.0 / sum) as f32;
        for yi in yr.iter_mut() {
            *yi *= inv_sum;
        }
    }
}

// ── RoPE (Rotary Position Embeddings) ────────────────────────────────────────

/// YaRN ramp function.
/// Direct translation of `rope_yarn_ramp` (ops.cpp line 5951).
#[inline]
fn rope_yarn_ramp(low: f32, high: f32, i0: i64) -> f32 {
    let y = (i0 as f32 / 2.0 - low) / (high - low).max(0.001);
    1.0 - y.max(0.0).min(1.0)
}

/// YaRN-corrected rotation angle.
/// Direct translation of `rope_yarn` (ops.cpp line 5958).
#[inline]
fn rope_yarn(
    theta_extrap: f32,
    freq_scale: f32,
    corr_dims: [f32; 2],
    i0: i64,
    ext_factor: f32,
    mscale: f32,
) -> (f32, f32) {
    let theta_interp = freq_scale * theta_extrap;
    let theta = if ext_factor != 0.0 {
        let ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        let th = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        let ms = mscale * (1.0 + 0.1 * (1.0 / freq_scale).ln());
        (th, ms)
    } else {
        (theta_interp, mscale)
    };
    (theta.1 * theta.0.cos(), theta.1 * theta.0.sin())
}

/// Compute YaRN correction dimensions.
/// Direct translation of `ggml_rope_yarn_corr_dims` (ggml.c / llama.cpp internals).
/// Formula: low = n_dims * log(n_ctx_orig / (beta_fast * 2π)) / (2 * log(freq_base))
///          high = n_dims * log(n_ctx_orig / (beta_slow * 2π)) / (2 * log(freq_base))
pub fn rope_yarn_corr_dims(
    n_dims: usize,
    n_ctx_orig: usize,
    freq_base: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> [f32; 2] {
    let denom = 2.0 * freq_base.ln();
    let low = (n_dims as f32 * (n_ctx_orig as f32 / (beta_fast * 2.0 * std::f32::consts::PI)).ln())
        / denom;
    let high = (n_dims as f32 * (n_ctx_orig as f32 / (beta_slow * 2.0 * std::f32::consts::PI)).ln())
        / denom;
    [low, high]
}

/// Build the per-position (cos, sin) cache used by `rope_f32`.
///
/// `cache` has length `ne0` (head_dim); entries come in pairs `[cos, sin]`.
/// `sin_sign` is `1.0` for forward RoPE, `-1.0` for the backward pass.
///
/// Direct translation of `ggml_rope_cache_init` (ops.cpp line 5975).
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
    while (i0 as usize) < ne0 {
        let ff = freq_factors.map_or(1.0, |f| f[i0 as usize / 2]);
        let (cos_t, sin_t) = rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale);
        cache[i0 as usize] = cos_t;
        cache[i0 as usize + 1] = sin_t * sin_sign;
        theta *= theta_scale;
        i0 += 2;
    }
}

/// Build the multi-rope (mRoPE) per-position cache.
/// Direct translation of `ggml_mrope_cache_init` (ops.cpp line 5991).
#[allow(clippy::too_many_arguments)]
pub fn mrope_cache_init(
    theta_base_t: f32,
    theta_base_h: f32,
    theta_base_w: f32,
    theta_base_e: f32,
    sections: [i32; 4],
    is_imrope: bool,
    indep_sects: bool,
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
    let mut theta_t = theta_base_t;
    let mut theta_h = theta_base_h;
    let mut theta_w = theta_base_w;
    let mut theta_e = theta_base_e;
    let sect_dims = (sections[0] + sections[1] + sections[2] + sections[3]) as usize;
    let sec_w = (sections[1] + sections[0]) as usize;
    let sec_e = (sections[2] as usize) + sec_w;

    let mut i0: usize = 0;
    while i0 < ne0 {
        let ff = freq_factors.map_or(1.0, |f| f[i0 / 2]);
        let sector = (i0 / 2) % sect_dims;

        if indep_sects {
            if sector == 0 {
                theta_t = theta_base_t;
            } else if sector == sections[0] as usize {
                theta_h = theta_base_h;
            } else if sector == sec_w {
                theta_w = theta_base_w;
            } else if sector == sec_e {
                theta_e = theta_base_e;
            }
        }

        let theta = if is_imrope {
            if sector % 3 == 1 && sector < 3 * sections[1] as usize {
                theta_h
            } else if sector % 3 == 2 && sector < 3 * sections[2] as usize {
                theta_w
            } else if sector % 3 == 0 && sector < 3 * sections[0] as usize {
                theta_t
            } else {
                theta_e
            }
        } else if sector >= sections[0] as usize && sector < sec_w {
            theta_h
        } else if sector >= sec_w && sector < sec_w + sections[2] as usize {
            theta_w
        } else if sector >= sec_w + sections[2] as usize {
            theta_e
        } else {
            theta_t
        };

        let (cos_t, sin_t) = rope_yarn(theta / ff, freq_scale, corr_dims, i0 as i64, ext_factor, mscale);
        cache[i0] = cos_t;
        cache[i0 + 1] = sin_t * sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
        i0 += 2;
    }
}

/// Apply a pre-computed (cos, sin) cache to a head row of `n` elements.
///
/// Rotates pairs `(src[ic], src[ic + n_offset])` using the cached values.
/// `n_offset` is `n/2` for NeoX-style, or `1` for normal (GPT-NeoX vs GPT-J).
///
/// Direct translation of `rotate_pairs<float>` (ops.cpp line 6063).
pub fn rotate_pairs_f32(
    n: usize,
    n_offset: usize,
    cache: &[f32],
    src: &[f32],
    dst: &mut [f32],
    scale: usize,
) {
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

/// RoPE (Rotary Position Embedding) forward pass — f32, NeoX mode.
///
/// Applies rotary embeddings in-place to `x`, which is a flat array of shape
/// `[n_tokens × n_heads × head_dim]`.  `positions` contains one integer token
/// position per token (length `n_tokens`).
///
/// Parameters match llama.cpp defaults:
/// - `mode = GGML_ROPE_TYPE_NEOX` (split-half, used by LLaMA/Mistral)
/// - no mRoPE, no YaRN extension by default (`ext_factor = 0`, `freq_scale = 1`)
///
/// Direct translation of `ggml_compute_forward_rope_flt<float>` (ops.cpp line 6082)
/// restricted to the NEOX case.
///
/// # Panics
/// Panics if `x.len() != n_tokens * n_heads * head_dim`.
#[allow(clippy::too_many_arguments)]
pub fn rope_neox_f32(
    x: &mut [f32],
    positions: &[i32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    n_dims: usize,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    n_ctx_orig: usize,
    freq_factors: Option<&[f32]>,
) {
    assert_eq!(x.len(), n_tokens * n_heads * head_dim);
    assert_eq!(positions.len(), n_tokens);
    assert!(n_dims <= head_dim);
    assert_eq!(n_dims % 2, 0);

    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    // cache: head_dim floats, [cos0, sin0, cos2, sin2, ...]
    let mut cache = vec![0.0f32; head_dim];

    for tok in 0..n_tokens {
        let p = positions[tok] as f32;
        rope_cache_init(
            p,
            freq_scale,
            freq_factors,
            corr_dims,
            head_dim,
            ext_factor,
            attn_factor,
            &mut cache,
            1.0,
            theta_scale,
        );

        for head in 0..n_heads {
            let base = tok * n_heads * head_dim + head * head_dim;
            // NeoX: rotate pairs (x[i], x[i + n_dims/2]) for i in 0..n_dims/2
            // We need a temp copy to avoid aliasing
            let src: Vec<f32> = x[base..base + head_dim].to_vec();
            let dst = &mut x[base..base + head_dim];
            rotate_pairs_f32(n_dims, n_dims / 2, &cache, &src, dst, 2);
            // copy non-rotated part unchanged (already in place since src == dst for those)
            // positions n_dims..head_dim are already correct (they came from src copy)
            for i in n_dims..head_dim {
                dst[i] = src[i];
            }
        }
    }
}

/// RoPE forward pass — f32, normal (GPT-J) mode.
///
/// Like `rope_neox_f32` but rotates `(x[i], x[i+1])` pairs (interleaved).
/// Direct translation of the `GGML_ROPE_TYPE_NORMAL` branch in `rotate_pairs`
/// (ops.cpp line 6207).
#[allow(clippy::too_many_arguments)]
pub fn rope_normal_f32(
    x: &mut [f32],
    positions: &[i32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    n_dims: usize,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    n_ctx_orig: usize,
    freq_factors: Option<&[f32]>,
) {
    assert_eq!(x.len(), n_tokens * n_heads * head_dim);
    assert_eq!(positions.len(), n_tokens);

    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);
    let mut cache = vec![0.0f32; head_dim];

    for tok in 0..n_tokens {
        let p = positions[tok] as f32;
        rope_cache_init(
            p,
            freq_scale,
            freq_factors,
            corr_dims,
            head_dim,
            ext_factor,
            attn_factor,
            &mut cache,
            1.0,
            theta_scale,
        );

        for head in 0..n_heads {
            let base = tok * n_heads * head_dim + head * head_dim;
            let src: Vec<f32> = x[base..base + head_dim].to_vec();
            let dst = &mut x[base..base + head_dim];
            // scale=1 ⟹ ic = i0 (interleaved pairs)
            rotate_pairs_f32(n_dims, 1, &cache, &src, dst, 1);
            for i in n_dims..head_dim {
                dst[i] = src[i];
            }
        }
    }
}

// ── Flash Attention ───────────────────────────────────────────────────────────

/// Online-softmax flash attention — pure f32.
///
/// Computes `out[iq, :] = softmax(Q[iq] K^T * scale) V` using the
/// numerically-stable online algorithm (Dao et al., 2022).
///
/// Shapes:
/// - `q`:  `[n_q × d_k]` — query matrix (row-major)
/// - `k`:  `[n_kv × d_k]` — key matrix (row-major)
/// - `v`:  `[n_kv × d_v]` — value matrix (row-major)
/// - `out`: `[n_q × d_v]` — output (row-major)
/// - `mask`: optional `[n_q × n_kv]` additive attention mask (e.g. causal)
///
/// Direct translation of `ggml_compute_forward_flash_attn_ext_f16_one_chunk`
/// (ops.cpp line 8614), simplified to f32 throughout and a single-threaded,
/// single-chunk execution.
pub fn flash_attn_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    mask: Option<&[f32]>,
    n_q: usize,
    n_kv: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
) {
    assert_eq!(q.len(), n_q * d_k);
    assert_eq!(k.len(), n_kv * d_k);
    assert_eq!(v.len(), n_kv * d_v);
    assert_eq!(out.len(), n_q * d_v);
    if let Some(m) = mask {
        assert_eq!(m.len(), n_q * n_kv);
    }

    for iq in 0..n_q {
        let q_row = &q[iq * d_k..(iq + 1) * d_k];
        let out_row = &mut out[iq * d_v..(iq + 1) * d_v];

        // online softmax accumulators
        let mut s_sum = 0.0f32; // running sum (denominator)
        let mut m_max = f32::NEG_INFINITY; // running max

        // accumulator for weighted values
        let mut vkq = vec![0.0f32; d_v];

        for ic in 0..n_kv {
            let k_row = &k[ic * d_k..(ic + 1) * d_k];

            // KQ dot product
            let mut s: f32 = q_row.iter().zip(k_row.iter()).map(|(&qi, &ki)| qi * ki).sum();
            s *= scale;

            // additive mask
            if let Some(m) = mask {
                let mv = m[iq * n_kv + ic];
                if mv == f32::NEG_INFINITY {
                    continue;
                }
                s += mv;
            }

            // online max update
            let m_old = m_max;
            let ms; // rescale factor for existing accumulator
            let vs; // softmax weight for this key

            if s > m_max {
                m_max = s;
                ms = (m_old - m_max).exp();
                vs = 1.0; // exp(s - s) == 1
            } else {
                ms = 1.0;
                vs = (s - m_max).exp();
            }

            // rescale accumulator
            for vi in vkq.iter_mut() {
                *vi *= ms;
            }
            s_sum = s_sum * ms + vs;

            // accumulate V * vs
            let v_row = &v[ic * d_v..(ic + 1) * d_v];
            for di in 0..d_v {
                vkq[di] += v_row[di] * vs;
            }
        }

        // normalise
        let inv_s = if s_sum == 0.0 { 0.0 } else { 1.0 / s_sum };
        for di in 0..d_v {
            out_row[di] = vkq[di] * inv_s;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ── RMS norm ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rms_norm_unit_vector() {
        // A unit vector should remain a unit vector after RMS norm (scale == 1).
        let x = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut out = vec![0.0f32; 4];
        rms_norm_rows_f32(&x, &mut out, 4, 1e-5);
        // rms(x) = sqrt(1/4) = 0.5  =>  scale = 2  =>  out = [2, 0, 0, 0]
        assert!(approx_eq(out[0], 2.0, 1e-4));
        for &v in &out[1..] {
            assert!(approx_eq(v, 0.0, 1e-6));
        }
    }

    #[test]
    fn test_rms_norm_all_ones() {
        let x = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rms_norm_rows_f32(&x, &mut out, 4, 1e-5);
        // rms(ones) = 1  =>  scale = 1  =>  out == x
        for &v in &out {
            assert!(approx_eq(v, 1.0, 1e-5));
        }
    }

    #[test]
    fn test_rms_norm_two_rows() {
        let x = vec![2.0f32, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0f32; 8];
        rms_norm_rows_f32(&x, &mut out, 4, 1e-8);
        // row 0: rms = sqrt((4+4)/4) = sqrt(2) => scale = 1/sqrt(2)
        let expected0 = 2.0f32 / 2.0f32.sqrt();
        assert!(approx_eq(out[0], expected0, 1e-5));
        assert!(approx_eq(out[1], expected0, 1e-5));
        // row 1: rms = 1 => out = [1,1,1,1]
        for &v in &out[4..8] {
            assert!(approx_eq(v, 1.0, 1e-5));
        }
    }

    // ── RMS norm + mul fused ──────────────────────────────────────────────────

    #[test]
    fn test_rms_norm_mul_scales_by_weight() {
        let x = vec![1.0f32; 4];
        let w = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut out = vec![0.0f32; 4];
        rms_norm_mul_rows_f32(&x, &w, &mut out, 4, 1e-8);
        // rms=1, scale=1, so out = w
        for (i, &v) in out.iter().enumerate() {
            assert!(approx_eq(v, w[i], 1e-5));
        }
    }

    // ── Layer norm ────────────────────────────────────────────────────────────

    #[test]
    fn test_layer_norm_zero_mean() {
        let x = vec![-1.0f32, 0.0, 1.0, 0.0];
        let mut out = vec![0.0f32; 4];
        layer_norm_rows_f32(&x, &mut out, 4, 1e-8);
        // mean = 0, var = (1+0+1+0)/4 = 0.5, scale = sqrt(2)
        let scale = 2.0f32.sqrt();
        assert!(approx_eq(out[0], -1.0 * scale, 1e-5));
        assert!(approx_eq(out[1], 0.0, 1e-6));
        assert!(approx_eq(out[2], 1.0 * scale, 1e-5));
        assert!(approx_eq(out[3], 0.0, 1e-6));
    }

    // ── Softmax ───────────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_uniform() {
        let x = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 4];
        soft_max_rows_f32(&x, &mut out, None, 4, 1.0);
        for &v in &out {
            assert!(approx_eq(v, 0.25, 1e-6));
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        soft_max_rows_f32(&x, &mut out, None, 4, 1.0);
        let sum: f32 = out.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-6));
        // monotone: larger input → larger probability
        assert!(out[3] > out[2]);
        assert!(out[2] > out[1]);
        assert!(out[1] > out[0]);
    }

    #[test]
    fn test_softmax_with_causal_mask() {
        // Two rows of length 2; second element is masked to -inf on row 0.
        let x = vec![1.0f32, 1.0, 1.0, 1.0];
        let mask = vec![0.0f32, f32::NEG_INFINITY, 0.0, 0.0];
        let mut out = vec![0.0f32; 4];
        soft_max_rows_f32(&x, &mut out, Some(&mask), 2, 1.0);
        // row 0: only first element survives  → [1, 0]
        assert!(approx_eq(out[0], 1.0, 1e-5));
        assert!(approx_eq(out[1], 0.0, 1e-5));
        // row 1: both equal → [0.5, 0.5]
        assert!(approx_eq(out[2], 0.5, 1e-5));
        assert!(approx_eq(out[3], 0.5, 1e-5));
    }

    // ── RoPE ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_rope_cache_identity_at_zero() {
        // position 0 → all θ = 0 → cos=1, sin=0 → rotation is identity
        let mut cache = vec![0.0f32; 4];
        rope_cache_init(0.0, 1.0, None, [0.0, 0.0], 4, 0.0, 1.0, &mut cache, 1.0, 0.5);
        // cache[0] = cos(0) = 1, cache[1] = sin(0) = 0
        assert!(approx_eq(cache[0], 1.0, 1e-6));
        assert!(approx_eq(cache[1], 0.0, 1e-6));
    }

    #[test]
    fn test_rope_neox_preserves_norm() {
        // Rotating a vector should not change its L2 norm.
        let n_tokens = 1;
        let n_heads = 1;
        let head_dim = 4;
        let n_dims = 4;
        let mut x = vec![1.0f32, 0.0, 1.0, 0.0];
        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let positions = vec![1i32];
        rope_neox_f32(
            &mut x,
            &positions,
            n_tokens,
            n_heads,
            head_dim,
            n_dims,
            10000.0,
            1.0,
            0.0,
            1.0,
            32.0,
            1.0,
            8192,
            None,
        );
        let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(approx_eq(norm_before, norm_after, 1e-5));
    }

    #[test]
    fn test_rope_neox_position_zero_is_identity() {
        // At position 0, θ = 0 everywhere, so RoPE is the identity transform.
        let n_tokens = 1;
        let n_heads = 2;
        let head_dim = 4;
        let n_dims = 4;
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = original.clone();
        let positions = vec![0i32];
        rope_neox_f32(
            &mut x,
            &positions,
            n_tokens,
            n_heads,
            head_dim,
            n_dims,
            10000.0,
            1.0,
            0.0,
            1.0,
            32.0,
            1.0,
            8192,
            None,
        );
        for i in 0..x.len() {
            assert!(approx_eq(x[i], original[i], 1e-5), "element {i} changed");
        }
    }

    // ── Flash attention ───────────────────────────────────────────────────────

    #[test]
    fn test_flash_attn_single_query_uniform_keys() {
        // With uniform keys, the output should equal the average of V rows.
        let n_q = 1;
        let n_kv = 3;
        let d_k = 2;
        let d_v = 2;
        let q = vec![1.0f32, 0.0];
        let k = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0]; // all same
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0f32; n_q * d_v];
        flash_attn_f32(&q, &k, &v, &mut out, None, n_q, n_kv, d_k, d_v, 1.0);
        // uniform weights → output = mean of V rows = [3, 4]
        assert!(approx_eq(out[0], 3.0, 1e-4));
        assert!(approx_eq(out[1], 4.0, 1e-4));
    }

    #[test]
    fn test_flash_attn_causal_mask() {
        // With a lower-triangular mask, query 0 should only attend to key 0.
        let n_q = 2;
        let n_kv = 2;
        let d_k = 1;
        let d_v = 1;
        let q = vec![1.0f32, 1.0];
        let k = vec![1.0f32, 1.0];
        let v = vec![10.0f32, 20.0];
        // mask: [[0, -inf], [0, 0]]
        let mask = vec![0.0f32, f32::NEG_INFINITY, 0.0, 0.0];
        let mut out = vec![0.0f32; n_q * d_v];
        flash_attn_f32(&q, &k, &v, &mut out, Some(&mask), n_q, n_kv, d_k, d_v, 1.0);
        // q0 → only attends to k0 → v0 = 10
        assert!(approx_eq(out[0], 10.0, 1e-4));
        // q1 → attends to both k0, k1 equally → (10+20)/2 = 15
        assert!(approx_eq(out[1], 15.0, 1e-4));
    }

    // ── Activations ───────────────────────────────────────────────────────────

    #[test]
    fn test_silu_positive() {
        // silu(x) = x * sigmoid(x); for large x, silu ≈ x
        let y = silu_f32(10.0);
        assert!(approx_eq(y, 10.0, 0.01));
    }

    #[test]
    fn test_silu_zero() {
        assert!(approx_eq(silu_f32(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_gelu_zero() {
        assert!(approx_eq(gelu_f32(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_gelu_large_positive() {
        // gelu(x) ≈ x for large x
        let y = gelu_f32(10.0);
        assert!(approx_eq(y, 10.0, 0.01));
    }

    #[test]
    fn test_gelu_quick_zero() {
        assert!(approx_eq(gelu_quick_f32(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_leaky_relu_positive() {
        assert!(approx_eq(leaky_relu_f32(2.0, 0.1), 2.0, 1e-6));
    }

    #[test]
    fn test_leaky_relu_negative() {
        assert!(approx_eq(leaky_relu_f32(-2.0, 0.1), -0.2, 1e-6));
    }

    // ── Element-wise ops ──────────────────────────────────────────────────────

    #[test]
    fn test_vec_add_f32() {
        let x = vec![1.0f32, 2.0, 3.0];
        let y = vec![4.0f32, 5.0, 6.0];
        let mut z = vec![0.0f32; 3];
        vec_add_f32(&mut z, &x, &y);
        assert_eq!(z, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vec_mul_f32() {
        let x = vec![2.0f32, 3.0];
        let y = vec![4.0f32, 5.0];
        let mut z = vec![0.0f32; 2];
        vec_mul_f32(&mut z, &x, &y);
        assert_eq!(z, vec![8.0, 15.0]);
    }
}
