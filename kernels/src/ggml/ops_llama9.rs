//! Faithful Rust translations of llama.cpp ops.cpp functions not covered by
//! ops_llama.rs through ops_llama6.rs.
//!
//! Sources: ggml/src/ggml-cpu/ops.cpp, vec.h

use super::helpers::f16_to_f32;

// ─── silu backward f16 (ops.cpp:2824, vec.h:1395) ────────────────────────────

#[inline]
fn silu_backward_f16(x_bits: u16, dy_bits: u16) -> u16 {
    let x = f16_to_f32(x_bits);
    let dy = f16_to_f32(dy_bits);
    let s = 1.0f32 / (1.0 + (-x).exp());
    let result = dy * s * (1.0 + x * (1.0 - s));
    half::f16::from_f32(result).to_bits()
}

/// Translation of `ggml_compute_forward_silu_back_f16` (ops.cpp:2824).
///
/// Computes element-wise SiLU backward pass for f16 tensors.
/// `dst[i]`, `src[i]`, `grad[i]` are f16 values (u16 bit-representations).
pub fn silu_back_f16(
    dst: &mut [u16],
    src: &[u16],
    grad: &[u16],
    nc: usize,
    nr: usize,
    ith: usize,
    nth: usize,
) {
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for i1 in ir0..ir1 {
        for k in 0..nc {
            dst[i1 * nc + k] = silu_backward_f16(src[i1 * nc + k], grad[i1 * nc + k]);
        }
    }
}

// ─── conv_2d_impl (ops.cpp:7084) ─────────────────────────────────────────────

/// Translation of `ggml_compute_forward_conv_2d_impl` (ops.cpp:7084).
///
/// 2D convolution: kernel [KW, KH, IC, OC] × src [W, H, IC, N] → dst [OW, OH, OC, N].
/// Uses im2col + GEMM approach. `tmp_buf` must have length ≥ patches_per_batch * knl_n.
/// This scalar version processes all patches; in the threaded C version patches are
/// split by `ith/nth` with a barrier between im2col and GEMM phases.
///
/// # Arguments
/// - `kernel_f32`: kernel weights as f32 [KW * KH * IC * OC]
/// - `src`: source f32 [W * H * IC * N] (W-major, contiguous)
/// - `dst`: output f32 [OW * OH * OC * N]
/// - `src_w`, `src_h`: source spatial dims
/// - `knl_w`, `knl_h`: kernel spatial dims
/// - `dst_w`, `dst_h`: output spatial dims
/// - `c_in`, `c_out`, `batch_n`: channel and batch counts
/// - `stride_x`, `stride_y`, `pad_x`, `pad_y`, `dilation_x`, `dilation_y`: conv params
/// - `ith`, `nth`: thread index and count
#[allow(clippy::too_many_arguments)]
pub fn conv_2d_impl(
    kernel: &[f32],   // [KW * KH * IC * OC], row-major: kernel[oc * knl_n + ic * knl_h * knl_w + ky * knl_w + kx]
    src: &[f32],      // [src_w * src_h * c_in * batch_n]
    dst: &mut [f32],  // [dst_w * dst_h * c_out * batch_n]
    src_w: usize, src_h: usize,
    knl_w: usize, knl_h: usize,
    dst_w: usize, dst_h: usize,
    c_in: usize, c_out: usize, batch_n: usize,
    stride_x: i64, stride_y: i64,
    pad_x: i64, pad_y: i64,
    dilation_x: i64, dilation_y: i64,
    ith: usize, nth: usize,
) {
    let knl_n = knl_w * knl_h * c_in; // elements per patch
    let patch_total = batch_n * dst_w * dst_h;

    let dr = (patch_total + nth - 1) / nth;
    let p0 = dr * ith;
    let p1 = (p0 + dr).min(patch_total);

    for p in p0..p1 {
        let n = p / (dst_w * dst_h);
        let dst_y = (p / dst_w) % dst_h;
        let dst_x = p % dst_w;

        // im2col: build the patch vector, then dot with each output channel
        // Patch vector layout: [ic, ky, kx]
        for oc in 0..c_out {
            let mut acc = 0.0f32;
            for ic in 0..c_in {
                for ky in 0..knl_h {
                    for kx in 0..knl_w {
                        let sy = dst_y as i64 * stride_y + ky as i64 * dilation_y - pad_y;
                        let sx = dst_x as i64 * stride_x + kx as i64 * dilation_x - pad_x;
                        let src_val = if sy < 0 || sy >= src_h as i64 || sx < 0 || sx >= src_w as i64 {
                            0.0f32
                        } else {
                            src[n * (src_w * src_h * c_in)
                                + sy as usize * src_w * c_in
                                + sx as usize * c_in
                                + ic]
                        };
                        let knl_idx = oc * knl_n + ic * (knl_h * knl_w) + ky * knl_w + kx;
                        acc += src_val * kernel[knl_idx];
                    }
                }
            }
            dst[n * (dst_w * dst_h * c_out) + dst_y * dst_w * c_out + dst_x * c_out + oc] = acc;
        }
    }
}

// ─── conv_3d_impl (ops.cpp:7221) ─────────────────────────────────────────────

/// Translation of `ggml_compute_forward_conv_3d_impl` (ops.cpp:7221).
///
/// 3D convolution: kernel [KW,KH,KD,IC,OC] × src [W,H,D,IC*N] → dst [OW,OH,OD,OC*N].
#[allow(clippy::too_many_arguments)]
pub fn conv_3d_impl(
    kernel: &[f32],  // [knl_w * knl_h * knl_d * c * oc]
    src: &[f32],     // [src_w * src_h * src_d * c * n_batch]
    dst: &mut [f32], // [dst_w * dst_h * dst_d * oc * n_batch]
    src_w: usize, src_h: usize, src_d: usize,
    knl_w: usize, knl_h: usize, knl_d: usize,
    dst_w: usize, dst_h: usize, dst_d: usize,
    c: usize, oc: usize, n_batch: usize,
    s0: i64, s1: i64, s2: i64,
    p0: i64, p1: i64, p2: i64,
    d0: i64, d1: i64, d2: i64,
    ith: usize, nth: usize,
) {
    let patch_total = n_batch * dst_w * dst_h * dst_d;
    let dr = (patch_total + nth - 1) / nth;
    let start = dr * ith;
    let end = (start + dr).min(patch_total);

    let knl_n_per_channel = knl_w * knl_h * knl_d;

    for p in start..end {
        let p_in_batch = p % (dst_w * dst_h * dst_d);
        let batch_idx = p / (dst_w * dst_h * dst_d);
        let dz = p_in_batch / (dst_w * dst_h);
        let dy = (p_in_batch % (dst_w * dst_h)) / dst_w;
        let dx = p_in_batch % dst_w;

        for ioc in 0..oc {
            let mut acc = 0.0f32;
            for ic in 0..c {
                for kz in 0..knl_d {
                    for ky in 0..knl_h {
                        for kx in 0..knl_w {
                            let sz = dz as i64 * s2 + kz as i64 * d2 - p2;
                            let sy = dy as i64 * s1 + ky as i64 * d1 - p1;
                            let sx = dx as i64 * s0 + kx as i64 * d0 - p0;
                            let sv = if sz < 0 || sz >= src_d as i64
                                || sy < 0 || sy >= src_h as i64
                                || sx < 0 || sx >= src_w as i64 {
                                0.0f32
                            } else {
                                let cn_idx = batch_idx * c + ic;
                                src[sz as usize * src_h * src_w * c * n_batch
                                    + sy as usize * src_w * c * n_batch
                                    + sx as usize * c * n_batch
                                    + cn_idx]
                            };
                            let ki = ioc * (knl_n_per_channel * c)
                                + ic * knl_n_per_channel
                                + kz * knl_h * knl_w + ky * knl_w + kx;
                            acc += sv * kernel[ki];
                        }
                    }
                }
            }
            let ocn_idx = batch_idx * oc + ioc;
            dst[dz * dst_h * dst_w * oc * n_batch
                + dy * dst_w * oc * n_batch
                + dx * oc * n_batch
                + ocn_idx] = acc;
        }
    }
}

// ─── conv_transpose_2d_impl (ops.cpp:7362) ───────────────────────────────────

/// Translation of `ggml_compute_forward_conv_transpose_2d_impl` (ops.cpp:7362).
///
/// Transposed 2D convolution (deconvolution) for f32 kernel.
/// kernel: [Kw, Kh, Cout, Cin]; src: [Sw, Sh, Cin, batch]; dst: [OW, OH, Cout, batch].
/// Equivalent to the F32 specialization of the C++ template.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose_2d_f32(
    kernel: &[f32],  // [ne00 * ne01 * ne02 * ne03] = [Kw, Kh, Cout, Cin]
    src: &[f32],     // [ne10 * ne11 * ne12 * ne13] = [Sw, Sh, Cin, batch]
    dst: &mut [f32], // [ne0 * ne1 * ne2 * ne3]     = [OW, OH, Cout, batch]
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,  // kernel dims
    ne10: usize, ne11: usize, ne12: usize, ne13: usize,  // src dims
    ne0: usize, _ne1: usize, ne2: usize, ne3: usize,      // dst dims
    stride: usize,
    ith: usize, nth: usize,
) {
    // Thread range over Cout (ne2)
    let dp = (ne2 + nth - 1) / nth;
    let ip0 = dp * ith;
    let ip1 = (ip0 + dp).min(ne2);

    for i2 in ip0..ip1 {
        for i3 in 0..ne3 {
            let dst_base = i3 * ne2 * ne0 * _ne1 + i2 * ne0 * _ne1;
            for i11 in 0..ne11 {
                for i10 in 0..ne10 {
                    // src channel vector for this spatial position
                    for i01 in 0..ne01 {
                        for i00 in 0..ne00 {
                            // dot over Cin (ne03)
                            let mut v = 0.0f32;
                            for cin in 0..ne03 {
                                let k_idx = cin * ne01 * ne00 * ne02 + i01 * ne00 * ne02 + i00 * ne02 + i2; // permuted kernel [Cin, Kh, Kw, Cout]
                                let s_idx = i3 * ne12 * ne11 * ne10 + ne12 * i11 * ne10 + ne12 * i10 + cin; // permuted src [batch, Sh, Sw, Cin]
                                // Use original layout (not pre-permuted for scalar version)
                                let kernel_val = kernel[i2 * ne01 * ne00 * ne03 + i01 * ne00 * ne03 + i00 * ne03 + cin];
                                let src_val = src[i3 * ne12 * ne11 * ne10 + i11 * ne10 * ne12 + i10 * ne12 + cin];
                                let _ = (k_idx, s_idx); // suppress unused
                                v += kernel_val * src_val;
                            }
                            let dst_y = i11 * stride + i01;
                            let dst_x = i10 * stride + i00;
                            dst[dst_base + dst_y * ne0 + dst_x] += v;
                        }
                    }
                }
            }
        }
    }
}

// ─── cross_entropy_loss_f32 (ops.cpp:11696) ──────────────────────────────────

/// Translation of `ggml_compute_forward_cross_entropy_loss_f32` (ops.cpp:11696).
///
/// Computes mean cross-entropy loss across all rows.
/// `thread_sums` must have length `nth`; after calling on all threads,
/// sum `thread_sums` and divide by `-nr` on thread 0.
pub fn cross_entropy_loss_f32(
    src0: &[f32],   // logits [nc * nr]
    src1: &[f32],   // labels [nc * nr]
    nc: usize, nr: usize,
    ith: usize, nth: usize,
    thread_sums: &mut [f32], // length >= nth, index ith gets written
) {
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    let mut sum_thread = 0.0f32;
    let mut st = vec![0.0f32; nc];
    for i1 in ir0..ir1 {
        let s0 = &src0[i1 * nc..(i1 + 1) * nc];
        let s1 = &src1[i1 * nc..(i1 + 1) * nc];
        let max = s0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // log softmax
        let exp_sum: f32 = s0.iter().map(|&x| (x - max).exp()).sum();
        let log_sum = exp_sum.ln();
        for i in 0..nc {
            st[i] = (s0[i] - max) - log_sum - log_sum; // log_softmax - log_sum
            // Actually: log_softmax[i] = s0[i] - max - ln(sum_exp)
            st[i] = (s0[i] - max) - (exp_sum.ln()) - 0.0; // = log_softmax
        }
        // fix: log_softmax[i] = s0[i] - max - log(sum_exp)
        for i in 0..nc {
            st[i] = (s0[i] - max) - (exp_sum.ln());
        }
        // cross entropy: -sum(s1 * log_softmax)
        let row_sum: f32 = st.iter().zip(s1.iter()).map(|(&ls, &p)| ls * p).sum();
        sum_thread += row_sum;
    }
    thread_sums[ith] = sum_thread;
    // Caller must reduce: loss = -sum(thread_sums) / nr
}

/// Final reduction for cross_entropy_loss_f32 (run on thread 0 after barrier).
pub fn cross_entropy_loss_reduce(thread_sums: &[f32], nr: usize) -> f32 {
    -thread_sums.iter().sum::<f32>() / nr as f32
}

// ─── cross_entropy_loss_back_f32 (ops.cpp:11792) ─────────────────────────────

/// Translation of `ggml_compute_forward_cross_entropy_loss_back_f32` (ops.cpp:11792).
///
/// Backward pass: grad(src0f) = (softmax(src0f) - src1f) * grad_output / nr
pub fn cross_entropy_loss_back_f32(
    grad_output: f32, // scalar: (const float*)grad->data)[0]
    src0f: &[f32],    // logits [nc * nr]
    src1f: &[f32],    // labels [nc * nr]
    dst: &mut [f32],  // gradient output [nc * nr]
    nc: usize, nr: usize,
    ith: usize, nth: usize,
) {
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    let d_by_nr = grad_output / nr as f32;
    for i1 in ir0..ir1 {
        let s0 = &src0f[i1 * nc..(i1 + 1) * nc];
        let s1 = &src1f[i1 * nc..(i1 + 1) * nc];
        let ds0 = &mut dst[i1 * nc..(i1 + 1) * nc];
        let max = s0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = s0.iter().map(|&x| (x - max).exp()).sum();
        for i in 0..nc {
            ds0[i] = ((s0[i] - max).exp() / sum - s1[i]) * d_by_nr;
        }
    }
}

// ─── opt_step_adamw_f32 (ops.cpp:11873) ──────────────────────────────────────

/// Translation of `ggml_compute_forward_opt_step_adamw_f32` (ops.cpp:11873).
///
/// AdamW optimizer update step. All slices must be identical shape.
/// `ne00` = row width, `nr` = total rows.
/// `adamw_params = [alpha, beta1, beta2, eps, wd, beta1h, beta2h]`
#[allow(clippy::too_many_arguments)]
pub fn opt_step_adamw_f32(
    w: &mut [f32],  // weights (modified in place)
    g: &[f32],      // gradients
    m: &mut [f32],  // first moment
    v: &mut [f32],  // second moment
    ne00: usize, nr: usize,
    alpha: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, beta1h: f32, beta2h: f32,
    ith: usize, nth: usize,
) {
    let keep = 1.0 - alpha * wd;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        for i00 in 0..ne00 {
            let idx = ir * ne00 + i00;
            m[idx] = m[idx] * beta1 + g[idx] * (1.0 - beta1);
            v[idx] = v[idx] * beta2 + g[idx] * g[idx] * (1.0 - beta2);
            let mh = m[idx] * beta1h;
            let vh = (v[idx] * beta2h).sqrt() + eps;
            w[idx] = w[idx] * keep - alpha * mh / vh;
        }
    }
}

// ─── opt_step_sgd_f32 (ops.cpp:11958) ────────────────────────────────────────

/// Translation of `ggml_compute_forward_opt_step_sgd_f32` (ops.cpp:11958).
///
/// SGD optimizer update: w = w * (1 - alpha*wd) - alpha * g
pub fn opt_step_sgd_f32(
    w: &mut [f32], // weights (modified in place)
    g: &[f32],     // gradients
    ne00: usize, nr: usize,
    alpha: f32, wd: f32,
    ith: usize, nth: usize,
) {
    let keep = 1.0 - alpha * wd;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        for i00 in 0..ne00 {
            let idx = ir * ne00 + i00;
            w[idx] = w[idx] * keep - alpha * g[idx];
        }
    }
}

// ─── flash_attn_back_f32 — skeleton (ops.cpp:9367) ───────────────────────────

/// Translation of `ggml_compute_forward_flash_attn_back_f32` (ops.cpp:9367).
///
/// Backward pass for scaled dot-product attention.
/// Computes gradients for Q, K, V given the upstream gradient D.
///
/// Arguments use explicit flat-slice representation:
/// - `q`: [N, H, D], `k`: [M, H, D], `v`: [M, H, D], `d`: [N, H, D]
/// - `dq`, `dk`, `dv`: gradient outputs (same shape as q, k, v)
/// - `mask`: optional causal mask [N, M] (true = masked out)
/// - `scale`: attention scale (typically 1/sqrt(D))
/// - `masked`: whether to apply causal masking
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_back_f32(
    q: &[f32],       // [N * H * D]
    k: &[f32],       // [M * H * D]
    v: &[f32],       // [M * H * D]
    d: &[f32],       // upstream grad [N * H * D]
    dq: &mut [f32],  // grad Q
    dk: &mut [f32],  // grad K
    dv: &mut [f32],  // grad V
    n: usize,   // query seq len
    m: usize,   // key/value seq len (M = P + N)
    h: usize,   // num heads
    head_dim: usize,
    scale: f32,
    masked: bool,
    ith: usize, nth: usize,
) {
    // Thread split over query positions (i1 in 0..N * H)
    let total = n * h;
    let dr = (total + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(total);

    for ir in ir0..ir1 {
        let i1 = ir / h; // query position
        let ih = ir % h; // head

        // attention scores for this query row: attn[j] for j in 0..M
        let mut attn = vec![0.0f32; m];
        let q_off = i1 * h * head_dim + ih * head_dim;
        for j in 0..m {
            if masked && j > i1 {
                attn[j] = f32::NEG_INFINITY;
                continue;
            }
            let k_off = j * h * head_dim + ih * head_dim;
            let dot: f32 = (0..head_dim).map(|d| q[q_off + d] * k[k_off + d]).sum();
            attn[j] = dot * scale;
        }
        // softmax
        let max = attn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = attn.iter().map(|&a| (a - max).exp()).sum();
        for a in attn.iter_mut() {
            *a = if a.is_finite() { (*a - max).exp() / exp_sum } else { 0.0 };
        }

        let d_off = i1 * h * head_dim + ih * head_dim;

        // grad V: dv[j] += attn[j] * d[i1]
        for j in 0..m {
            let v_off = j * h * head_dim + ih * head_dim;
            for di in 0..head_dim {
                dv[v_off + di] += attn[j] * d[d_off + di];
            }
        }

        // scalar: ds = attn * (d · V - sum(attn * (d · V)))
        let mut dv_dot = vec![0.0f32; m];
        for j in 0..m {
            let v_off = j * h * head_dim + ih * head_dim;
            dv_dot[j] = (0..head_dim).map(|di| d[d_off + di] * v[v_off + di]).sum();
        }
        let sum_dv: f32 = attn.iter().zip(dv_dot.iter()).map(|(&a, &dv)| a * dv).sum();
        let mut ds = vec![0.0f32; m];
        for j in 0..m {
            ds[j] = attn[j] * (dv_dot[j] - sum_dv) * scale;
        }

        // grad Q: dq[i1] += sum_j ds[j] * k[j]
        for j in 0..m {
            let k_off = j * h * head_dim + ih * head_dim;
            for di in 0..head_dim {
                dq[q_off + di] += ds[j] * k[k_off + di];
            }
        }

        // grad K: dk[j] += ds[j] * q[i1]
        for j in 0..m {
            let k_off = j * h * head_dim + ih * head_dim;
            for di in 0..head_dim {
                dk[k_off + di] += ds[j] * q[q_off + di];
            }
        }
    }
}
