//! Direct Rust translation of llama.cpp ggml/src/ggml-cpu/ops.cpp lines 7978–12206.
//!
//! Functions translated in this file:
//!   ggml_compute_forward_upscale_f32 / upscale
//!   ggml_compute_forward_pad_f32 / pad / pad_reflect_1d
//!   ggml_wrap_index / ggml_compute_forward_roll_f32 / roll
//!   ggml_compute_forward_arange_f32 / arange
//!   ggml_compute_forward_timestep_embedding_f32 / timestep_embedding
//!   ggml_compute_forward_argsort_f32 / argsort
//!   ggml_compute_forward_top_k_f32 / top_k
//!   ggml_compute_forward_flash_attn_ext_f16_one_chunk
//!   ggml_compute_forward_flash_attn_ext_tiled (scalar path)
//!   ggml_flash_attn_ext_reduce_partials
//!   ggml_compute_forward_flash_attn_ext_f16 / flash_attn_ext
//!   ggml_compute_forward_flash_attn_back_f32 / flash_attn_back
//!   ggml_compute_forward_ssm_conv_f32 / ssm_conv
//!   ggml_compute_forward_ssm_scan_f32 / ssm_scan
//!   ggml_compute_forward_win_part_f32 / win_part
//!   ggml_compute_forward_win_unpart_f32 / win_unpart
//!   ggml_compute_forward_get_rel_pos_f16 / get_rel_pos
//!   ggml_compute_forward_add_rel_pos_f32 / add_rel_pos
//!   ggml_compute_forward_rwkv_wkv6_f32 / rwkv_wkv6
//!   ggml_compute_forward_gla_f32 / gla
//!   ggml_compute_forward_solve_tri_f32 / solve_tri
//!   ggml_compute_forward_gated_delta_net_one_chunk / _f32 / gated_delta_net
//!   ggml_dsv4_hc_comb_norm_cols / _norm_rows
//!   ggml_compute_forward_dsv4_hc_comb_f32 / dsv4_hc_comb
//!   ggml_compute_forward_dsv4_hc_pre_f32 / dsv4_hc_pre
//!   ggml_compute_forward_dsv4_hc_post_f32 / dsv4_hc_post
//!   ggml_compute_forward_rwkv_wkv7_f32 / rwkv_wkv7
//!   ggml_compute_forward_cross_entropy_loss_f32 / cross_entropy_loss
//!   ggml_compute_forward_cross_entropy_loss_back_f32 / cross_entropy_loss_back
//!   ggml_compute_forward_opt_step_adamw_f32 / opt_step_adamw
//!   ggml_compute_forward_opt_step_sgd_f32 / opt_step_sgd
//!   ggml_compute_forward_fwht / fwht
//!   ggml_compute_forward_lightning_indexer

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_clamp)]
#![allow(dead_code)]

use half::f16;

// ─── Scale / upscale modes (mirrors ggml_scale_mode / ggml_scale_flag) ──────

pub const SCALE_FLAG_ALIGN_CORNERS: i32 = 1 << 8;
pub const SCALE_FLAG_ANTIALIAS: i32 = 1 << 9;

pub const SCALE_MODE_NEAREST: i32 = 0;
pub const SCALE_MODE_BILINEAR: i32 = 1;
pub const SCALE_MODE_BICUBIC: i32 = 2;

// Sort orders (mirrors ggml_sort_order)
pub const SORT_ORDER_ASC: i32 = 0;
pub const SORT_ORDER_DESC: i32 = 1;

// ─── Helper: wrap-around index used by pad (circular) ────────────────────────

/// Direct translation of `ggml_wrap_around` (ops.cpp inline helper).
#[inline]
fn wrap_around(i: i64, n: i64) -> i64 {
    ((i % n) + n) % n
}

// ─── ggml_compute_forward_upscale_f32 (ops.cpp lines 7978–8172) ──────────────
//
// Parameters carry all information that ggml_tensor fields would:
//   src_data / dst_data : raw byte pointers (F32 tensors)
//   ne{0..3}  : dst element counts per dimension
//   nb{0..3}  : dst byte strides
//   ne0{0..3} : src element counts per dimension  (ne00..ne03)
//   nb0{0..3} : src byte strides                  (nb00..nb03)
//   ith / nth : thread index / count
//   mode_flags: op_params[0]

/// Direct translation of `ggml_compute_forward_upscale_f32` (ops.cpp line 7978).
#[allow(clippy::too_many_arguments)]
pub unsafe fn compute_forward_upscale_f32(
    src_data: *const u8,
    ne00: i64, ne01: i64, ne02: i64, ne03: i64,
    nb00: usize, nb01: usize, nb02: usize, nb03: usize,
    dst_data: *mut u8,
    ne0: i64, ne1: i64, ne2: i64, ne3: i64,
    nb0: usize, nb1: usize, nb2: usize, nb3: usize,
    ith: usize, nth: usize,
    mode_flags: i32,
) {
    let mut sf0 = ne0 as f32 / ne00 as f32;
    let mut sf1 = ne1 as f32 / ne01 as f32;
    let sf2 = ne2 as f32 / ne02 as f32;
    let sf3 = ne3 as f32 / ne03 as f32;
    let mut pixel_offset = 0.5f32;

    let mode = mode_flags & 0xFF;

    if mode_flags & SCALE_FLAG_ALIGN_CORNERS != 0 {
        pixel_offset = 0.0f32;
        sf0 = if ne0 > 1 && ne00 > 1 { (ne0 - 1) as f32 / (ne00 - 1) as f32 } else { sf0 };
        sf1 = if ne1 > 1 && ne01 > 1 { (ne1 - 1) as f32 / (ne01 - 1) as f32 } else { sf1 };
    }

    if mode == SCALE_MODE_NEAREST {
        for i3 in 0..ne3 {
            let i03 = (i3 as f32 / sf3) as i64;
            let mut i2 = ith as i64;
            while i2 < ne2 {
                let i02 = (i2 as f32 / sf2) as i64;
                for i1 in 0..ne1 {
                    let i01 = (i1 as f32 / sf1) as i64;
                    for i0 in 0..ne0 {
                        let i00 = (i0 as f32 / sf0) as i64;
                        let src = src_data.add(
                            i00 as usize * nb00
                            + i01 as usize * nb01
                            + i02 as usize * nb02
                            + i03 as usize * nb03,
                        ) as *const f32;
                        let dst = dst_data.add(
                            i0 as usize * nb0
                            + i1 as usize * nb1
                            + i2 as usize * nb2
                            + i3 as usize * nb3,
                        ) as *mut f32;
                        *dst = *src;
                    }
                }
                i2 += nth as i64;
            }
        }
    } else if mode == SCALE_MODE_BILINEAR && (mode_flags & SCALE_FLAG_ANTIALIAS != 0) {
        let triangle_filter = |x: f32| -> f32 { (1.0f32 - x.abs()).max(0.0f32) };

        let support1 = (1.0f32 / sf1).max(1.0f32);
        let invscale1 = 1.0f32 / support1;
        let support0 = (1.0f32 / sf0).max(1.0f32);
        let invscale0 = 1.0f32 / support0;

        for i3 in 0..ne3 {
            let i03 = (i3 as f32 / sf3) as i64;
            let mut i2 = ith as i64;
            while i2 < ne2 {
                let i02 = (i2 as f32 / sf2) as i64;
                for i1 in 0..ne1 {
                    let y = (i1 as f32 + pixel_offset) / sf1;
                    for i0 in 0..ne0 {
                        let x = (i0 as f32 + pixel_offset) / sf0;

                        let x_min = ((x - support0 + pixel_offset) as i64).max(0);
                        let x_max = ((x + support0 + pixel_offset) as i64).min(ne00);
                        let y_min = ((y - support1 + pixel_offset) as i64).max(0);
                        let y_max = ((y + support1 + pixel_offset) as i64).min(ne01);

                        let mut val = 0.0f32;
                        let mut total_weight = 0.0f32;

                        for sy in y_min..y_max {
                            let weight_y = triangle_filter((sy as f32 - y + pixel_offset) * invscale1);
                            for sx in x_min..x_max {
                                let weight_x = triangle_filter((sx as f32 - x + pixel_offset) * invscale0);
                                let weight = weight_x * weight_y;
                                if weight <= 0.0f32 { continue; }

                                let pixel = *(src_data.add(
                                    sx as usize * nb00
                                    + sy as usize * nb01
                                    + i02 as usize * nb02
                                    + i03 as usize * nb03,
                                ) as *const f32);
                                val += pixel * weight;
                                total_weight += weight;
                            }
                        }

                        if total_weight > 0.0f32 { val /= total_weight; }

                        let dst = dst_data.add(
                            i0 as usize * nb0
                            + i1 as usize * nb1
                            + i2 as usize * nb2
                            + i3 as usize * nb3,
                        ) as *mut f32;
                        *dst = val;
                    }
                }
                i2 += nth as i64;
            }
        }
    } else if mode == SCALE_MODE_BILINEAR {
        for i3 in 0..ne3 {
            let i03 = (i3 as f32 / sf3) as i64;
            let mut i2 = ith as i64;
            while i2 < ne2 {
                let i02 = (i2 as f32 / sf2) as i64;
                for i1 in 0..ne1 {
                    let yf = (i1 as f32 + pixel_offset) / sf1 - pixel_offset;
                    let y0 = yf.floor() as i64;
                    let y1 = y0 + 1;
                    let y0c = y0.max(0).min(ne01 - 1);
                    let y1c = y1.max(0).min(ne01 - 1);
                    let dy = (yf - y0 as f32).max(0.0f32).min(1.0f32);

                    for i0 in 0..ne0 {
                        let xf = (i0 as f32 + pixel_offset) / sf0 - pixel_offset;
                        let x0 = xf.floor() as i64;
                        let x1 = x0 + 1;
                        let x0c = x0.max(0).min(ne00 - 1);
                        let x1c = x1.max(0).min(ne00 - 1);
                        let dx = (xf - x0 as f32).max(0.0f32).min(1.0f32);

                        let a = *(src_data.add(x0c as usize*nb00 + y0c as usize*nb01 + i02 as usize*nb02 + i03 as usize*nb03) as *const f32);
                        let b = *(src_data.add(x1c as usize*nb00 + y0c as usize*nb01 + i02 as usize*nb02 + i03 as usize*nb03) as *const f32);
                        let c = *(src_data.add(x0c as usize*nb00 + y1c as usize*nb01 + i02 as usize*nb02 + i03 as usize*nb03) as *const f32);
                        let d = *(src_data.add(x1c as usize*nb00 + y1c as usize*nb01 + i02 as usize*nb02 + i03 as usize*nb03) as *const f32);

                        let val = a*(1.0-dx)*(1.0-dy) + b*dx*(1.0-dy) + c*(1.0-dx)*dy + d*dx*dy;

                        let dst = dst_data.add(i0 as usize*nb0 + i1 as usize*nb1 + i2 as usize*nb2 + i3 as usize*nb3) as *mut f32;
                        *dst = val;
                    }
                }
                i2 += nth as i64;
            }
        }
    } else if mode == SCALE_MODE_BICUBIC {
        let a = -0.75f32;
        let weight1 = |x: f32| -> f32 { ((a + 2.0)*x - (a + 3.0))*x*x + 1.0 };
        let weight2 = |x: f32| -> f32 { ((a*x - 5.0*a)*x + 8.0*a)*x - 4.0*a };
        let bicubic = |p0: f32, p1: f32, p2: f32, p3: f32, x: f32| -> f32 {
            let w0 = weight2(x + 1.0);
            let w1 = weight1(x);
            let w2 = weight1(1.0 - x);
            let w3 = weight2(2.0 - x);
            p0*w0 + p1*w1 + p2*w2 + p3*w3
        };

        for i3 in 0..ne3 {
            let i03 = (i3 as f32 / sf3) as i64;
            let mut i2 = ith as i64;
            while i2 < ne2 {
                let i02 = (i2 as f32 / sf2) as i64;
                for i1 in 0..ne1 {
                    let yf = (i1 as f32 + pixel_offset) / sf1 - pixel_offset;
                    let y0 = yf.floor() as i64;
                    let dy = yf - y0 as f32;

                    for i0 in 0..ne0 {
                        let xf = (i0 as f32 + pixel_offset) / sf0 - pixel_offset;
                        let x0 = xf.floor() as i64;
                        let dx = xf - x0 as f32;

                        let p = |x_off: i64, y_off: i64| -> f32 {
                            let i00 = (x0 + x_off).max(0).min(ne00 - 1) as usize;
                            let i01 = (y0 + y_off).max(0).min(ne01 - 1) as usize;
                            *(src_data.add(i00*nb00 + i01*nb01 + i02 as usize*nb02 + i03 as usize*nb03) as *const f32)
                        };

                        let val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx),
                            dy,
                        );

                        let dst = dst_data.add(i0 as usize*nb0 + i1 as usize*nb1 + i2 as usize*nb2 + i3 as usize*nb3) as *mut f32;
                        *dst = val;
                    }
                }
                i2 += nth as i64;
            }
        }
    } else {
        panic!("unsupported upscale mode");
    }
}

// ─── ggml_compute_forward_pad_f32 (ops.cpp lines 8196–8258) ──────────────────

/// Direct translation of `ggml_compute_forward_pad_f32<false>` (zero-padding variant).
/// ops.cpp line 8196.
pub unsafe fn compute_forward_pad_f32(
    src_data: *const u8,
    ne00: i64, ne01: i64, ne02: i64, ne03: i64,
    nb00: usize, nb01: usize, nb02: usize, nb03: usize,
    dst_data: *mut f32,
    ne0: i64, ne1: i64, ne2: i64, ne3: i64,
    lp0: i32, rp0: i32,
    lp1: i32, rp1: i32,
    lp2: i32, rp2: i32,
    lp3: i32, rp3: i32,
    ith: usize, nth: usize,
) {
    let mut i1 = ith as i64;
    while i1 < ne1 {
        for i2 in 0..ne2 {
            for i0 in 0..ne0 {
                for i3 in 0..ne3 {
                    let dst_idx = (i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0) as usize;
                    let in_bounds =
                        i0 >= lp0 as i64 && i0 < ne0 - rp0 as i64
                        && i1 >= lp1 as i64 && i1 < ne1 - rp1 as i64
                        && i2 >= lp2 as i64 && i2 < ne2 - rp2 as i64
                        && i3 >= lp3 as i64 && i3 < ne3 - rp3 as i64;
                    if in_bounds {
                        let src_idx =
                            (i3 - lp3 as i64) as usize * nb03
                            + (i2 - lp2 as i64) as usize * nb02
                            + (i1 - lp1 as i64) as usize * nb01
                            + (i0 - lp0 as i64) as usize * nb00;
                        let val = *(src_data.add(src_idx) as *const f32);
                        *dst_data.add(dst_idx) = val;
                    } else {
                        *dst_data.add(dst_idx) = 0.0f32;
                    }
                }
            }
        }
        i1 += nth as i64;
    }
}

/// Direct translation of `ggml_compute_forward_pad_f32<true>` (circular variant).
/// ops.cpp line 8226.
pub unsafe fn compute_forward_pad_circular_f32(
    src_data: *const u8,
    ne00: i64, ne01: i64, ne02: i64, ne03: i64,
    nb00: usize, nb01: usize, nb02: usize, nb03: usize,
    dst_data: *mut f32,
    ne0: i64, ne1: i64, ne2: i64, ne3: i64,
    lp0: i32, lp1: i32, lp2: i32, lp3: i32,
    ith: usize, nth: usize,
) {
    let mut i1 = ith as i64;
    while i1 < ne1 {
        for i2 in 0..ne2 {
            for i0 in 0..ne0 {
                for i3 in 0..ne3 {
                    let dst_idx = (i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0) as usize;
                    let src_i0 = wrap_around(i0 - lp0 as i64, ne00);
                    let src_i1 = wrap_around(i1 - lp1 as i64, ne01);
                    let src_i2 = wrap_around(i2 - lp2 as i64, ne02);
                    let src_i3 = wrap_around(i3 - lp3 as i64, ne03);
                    let src_idx =
                        src_i3 as usize * nb03
                        + src_i2 as usize * nb02
                        + src_i1 as usize * nb01
                        + src_i0 as usize * nb00;
                    let val = *(src_data.add(src_idx) as *const f32);
                    *dst_data.add(dst_idx) = val;
                }
            }
        }
        i1 += nth as i64;
    }
}

// ─── ggml_compute_forward_pad_reflect_1d (ops.cpp lines 8284–8315) ───────────

/// Direct translation of `ggml_compute_forward_pad_reflect_1d` (ops.cpp line 8284).
pub unsafe fn compute_forward_pad_reflect_1d_f32(
    src_data: *const u8,
    nb01: usize, nb02: usize, nb03: usize,
    ne00: i64,
    dst_data: *mut u8,
    nb0: usize, nb1: usize, nb2: usize, nb3: usize,
    ne0: i64, ne1: i64, ne2: i64, ne3: i64,
    p0: i32, p1: i32,
    ith: usize, nth: usize,
) {
    for i3 in 0..ne3 {
        for i2 in 0..ne2 {
            let mut i1 = ith as i64;
            while i1 < ne1 {
                // left edge of the valid (non-pad) region in dst
                let left  = dst_data.add(i3 as usize*nb3 + i2 as usize*nb2 + i1 as usize*nb1 + p0 as usize*nb0) as *mut f32;
                // right edge (last valid element)
                let right = dst_data.add(i3 as usize*nb3 + i2 as usize*nb2 + i1 as usize*nb1 + (ne0 - p1 as i64 - 1) as usize*nb0) as *mut f32;

                // copy src into valid region
                let src_row = src_data.add(i3 as usize*nb03 + i2 as usize*nb02 + i1 as usize*nb01) as *const f32;
                std::ptr::copy_nonoverlapping(src_row, left, ne00 as usize);

                // reflect left pad: left[-i] = left[i]
                for i in 1..=p0 as usize {
                    *left.sub(i) = *left.add(i);
                }
                // reflect right pad: right[i] = right[-i]
                for i in 1..=p1 as usize {
                    *right.add(i) = *right.sub(i);
                }

                i1 += nth as i64;
            }
        }
    }
}

// ─── ggml_wrap_index / ggml_compute_forward_roll_f32 (ops.cpp lines 8319–8382) ─

/// Direct translation of `ggml_wrap_index` (ops.cpp line 8319).
#[inline]
pub fn wrap_index(i: i64, ne: i64) -> i64 {
    if i < 0 { i + ne } else if i >= ne { i - ne } else { i }
}

/// Direct translation of `ggml_compute_forward_roll_f32` (ops.cpp line 8328).
pub unsafe fn compute_forward_roll_f32(
    src_data: *const f32,
    ne00: i64, ne01: i64, ne02: i64, ne03: i64,
    nb00: usize, nb01: usize, nb02: usize, nb03: usize,
    dst_data: *mut f32,
    ne0: i64, ne1: i64, ne2: i64, ne3: i64,
    nb0: usize, nb1: usize, nb2: usize, nb3: usize,
    s0: i32, s1: i32, s2: i32, s3: i32,
    ith: usize, nth: usize,
) {
    let total = ne1 * ne2 * ne3;
    let per_thread = (total + nth as i64) / nth as i64;
    let start = ith as i64 * per_thread;
    let end = (start + per_thread).min(total);

    let _ = (ne0, nb0); // dst dim0 stride == sizeof(f32), element-wise

    for i in start..end {
        let i1 = i % ne1;
        let i2 = (i / ne1) % ne2;
        let i3 = i / (ne2 * ne1);
        let dst_row = (dst_data as *mut u8).add(i3 as usize*nb3 + i2 as usize*nb2 + i1 as usize*nb1) as *mut f32;

        let i01 = wrap_index(i1 - s1 as i64, ne01);
        let i02 = wrap_index(i2 - s2 as i64, ne02);
        let i03 = wrap_index(i3 - s3 as i64, ne03);
        let src_row = (src_data as *const u8).add(i03 as usize*nb03 + i02 as usize*nb02 + i01 as usize*nb01) as *const f32;

        let s = wrap_index(-s0 as i64, ne00) as usize;
        let n = ne00 as usize - s;
        std::ptr::copy_nonoverlapping(src_row.add(s), dst_row, n);
        std::ptr::copy_nonoverlapping(src_row, dst_row.add(n), s);
    }
}

// ─── ggml_compute_forward_arange_f32 (ops.cpp lines 8386–8407) ───────────────

/// Direct translation of `ggml_compute_forward_arange_f32` (ops.cpp line 8386).
pub fn compute_forward_arange_f32(
    dst: &mut [f32],
    start: f32, stop: f32, step: f32,
    ith: usize, nth: usize,
) {
    let steps = ((stop - start) / step).ceil() as usize;
    assert_eq!(dst.len(), steps);
    let mut i = ith;
    while i < steps {
        dst[i] = start + step * i as f32;
        i += nth;
    }
}

// ─── ggml_compute_forward_timestep_embedding_f32 (ops.cpp lines 8424–8455) ───

/// Direct translation of `ggml_compute_forward_timestep_embedding_f32` (ops.cpp line 8424).
/// src: flat &[f32] of shape [ne00]
/// dst: flat *mut f32 of shape [ne00, dim]  (row-major, row stride = nb1/4)
pub unsafe fn compute_forward_timestep_embedding_f32(
    src: *const f32,
    ne00: i64,
    dst: *mut f32,
    nb1: usize,  // byte stride between rows
    dim: i32, max_period: i32,
    ith: usize, nth: usize,
) {
    let half = dim / 2;
    for i in 0..ne00 as usize {
        let embed_data = (dst as *mut u8).add(i * nb1) as *mut f32;
        let timestep = *src.add(i);
        let mut j = ith;
        while j < half as usize {
            let freq = (-((max_period as f32).ln()) * j as f32 / half as f32).exp();
            let arg = timestep * freq;
            *embed_data.add(j) = arg.cos();
            *embed_data.add(j + half as usize) = arg.sin();
            j += nth;
        }
        if dim % 2 != 0 && ith == 0 {
            *embed_data.add(2 * half as usize) = 0.0f32;
        }
    }
}

// ─── ggml_compute_forward_argsort_f32 (ops.cpp lines 8489–8546) ──────────────

/// Direct translation of `ggml_compute_forward_argsort_f32` (ops.cpp line 8489).
pub unsafe fn compute_forward_argsort_f32(
    src_data: *const u8,
    nb01: usize,
    dst_data: *mut u8,
    nb1: usize,
    ne0: i64,
    nr: i64,
    order: i32,
    ith: usize, nth: usize,
) {
    let mut i = ith as i64;
    while i < nr {
        let src_row = src_data.add(i as usize * nb01) as *const f32;
        let dst_row = dst_data.add(i as usize * nb1) as *mut i32;

        // initialise indices
        for j in 0..ne0 as usize {
            *dst_row.add(j) = j as i32;
        }

        // sort indices
        let indices = std::slice::from_raw_parts_mut(dst_row, ne0 as usize);
        let data = std::slice::from_raw_parts(src_row, ne0 as usize);

        if order == SORT_ORDER_ASC {
            indices.sort_unstable_by(|&a, &b| {
                data[a as usize].partial_cmp(&data[b as usize]).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            indices.sort_unstable_by(|&a, &b| {
                data[b as usize].partial_cmp(&data[a as usize]).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        i += nth as i64;
    }
}

// ─── ggml_compute_forward_top_k_f32 (ops.cpp lines 8557–8611) ────────────────

/// Direct translation of `ggml_compute_forward_top_k_f32` (ops.cpp line 8557).
pub unsafe fn compute_forward_top_k_f32(
    src_data: *const u8,
    nb01: usize,
    dst_data: *mut u8,
    nb1: usize,
    ne00: i64,
    nr: i64,
    top_k: usize,
    tmp: *mut i32,   // scratch: ne00 entries per thread
    ith: usize, nth: usize,
) {
    let mut i = ith as i64;
    while i < nr {
        let src_row = src_data.add(i as usize * nb01) as *const f32;
        let data = std::slice::from_raw_parts(src_row, ne00 as usize);
        let scratch = std::slice::from_raw_parts_mut(tmp, ne00 as usize);

        for j in 0..ne00 as usize { scratch[j] = j as i32; }

        // partial sort: first top_k elements are the largest
        scratch.select_nth_unstable_by(top_k, |&a, &b| {
            data[b as usize].partial_cmp(&data[a as usize]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let dst_row = dst_data.add(i as usize * nb1) as *mut i32;
        std::ptr::copy_nonoverlapping(scratch.as_ptr(), dst_row, top_k);

        // swap first two elements to emphasise order is not important
        if top_k > 1 {
            let tmp0 = *dst_row;
            *dst_row = *dst_row.add(1);
            *dst_row.add(1) = tmp0;
        }

        i += nth as i64;
    }
}

// ─── Flash attention scalar helper (vec scale / mad / max for f32) ────────────

#[inline]
fn vec_scale_f32(n: usize, x: &mut [f32], v: f32) {
    for i in 0..n { x[i] *= v; }
}

#[inline]
fn vec_mad_f32(n: usize, y: &mut [f32], x: &[f32], v: f32) {
    for i in 0..n { y[i] += x[i] * v; }
}

#[inline]
fn vec_dot_f32(n: usize, a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..n { s += a[i] * b[i]; }
    s
}

#[inline]
fn vec_max_f32(n: usize, x: &[f32]) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for i in 0..n { if x[i] > m { m = x[i]; } }
    m
}

/// Online softmax: applies exp(x[i] - max) and returns sum.
#[inline]
fn vec_soft_max_f32(n: usize, dst: &mut [f32], src: &[f32], max: f32) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        let v = (src[i] - max).exp();
        dst[i] = v;
        sum += v;
    }
    sum
}

// ─── ggml_compute_forward_flash_attn_ext_f16_one_chunk ───────────────────────
// ops.cpp lines 8614–8850  (scalar F32 path; F16 V path omitted for simplicity)

/// Scalar translation of `ggml_compute_forward_flash_attn_ext_f16_one_chunk`.
/// All tensors are assumed F32 for Q, K, V.
/// ops.cpp line 8614.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_ext_one_chunk_f32(
    // Q
    q_data: *const u8, neq0: i64, neq1: i64, neq2: i64, neq3: i64,
    nbq1: usize, nbq2: usize, nbq3: usize,
    // K
    k_data: *const u8, nek0: i64, nek1: i64, nek2: i64, nek3: i64,
    nbk1: usize, nbk2: usize, nbk3: usize,
    // V
    v_data: *const u8, nev0: i64, _nev2: i64, _nev3: i64,
    nbv1: usize, nbv2: usize, nbv3: usize,
    // mask (optional, F16)
    mask_data: *const u8, mask_ne2: i64, mask_ne3: i64,
    mask_nb1: usize, mask_nb2: usize, mask_nb3: usize,
    // sinks (optional, F32 scalar per head)
    sinks_data: *const u8,
    // dst
    dst_data: *mut u8, ne1: i64, ne2: i64, nb1: usize,
    // params
    scale: f32, max_bias: f32, logit_softcap: f32,
    ir0: usize, ir1: usize,
    ic_start: i64, ic_end: i64,
    // partial output buffer (None = write directly to dst)
    partials: *mut f32, partial_stride: i64,
    // scratch per-thread (must be >= DK + 2*DV + 8 f32)
    scratch: *mut f32,
) {
    let dk = nek0 as usize;
    let dv = nev0 as usize;

    let write_partials = !partials.is_null();

    let n_head      = neq2 as u32;
    let n_head_log2 = 1u32 << (n_head as f32).log2().floor() as u32;
    let m0 = 2.0f32.powf(-(max_bias       ) / n_head_log2 as f32);
    let m1 = 2.0f32.powf(-(max_bias / 2.0f32) / n_head_log2 as f32);

    let scale_eff = if logit_softcap != 0.0 { scale / logit_softcap } else { scale };

    let rk2 = neq2 / nek2;
    let rk3 = neq3 / nek3;
    let rv2 = neq2 / _nev2;
    let rv3 = neq3 / _nev3;

    // VKQ32 / V32 from scratch
    let vkq32 = std::slice::from_raw_parts_mut(scratch, dv);
    let v32   = std::slice::from_raw_parts_mut(scratch.add(dv), dv);
    let q_buf = std::slice::from_raw_parts_mut(scratch.add(2*dv), dk);

    for ir in ir0..ir1 {
        let iq3 = ir / (neq2 as usize * neq1 as usize);
        let iq2 = (ir - iq3 * neq2 as usize * neq1 as usize) / neq1 as usize;
        let iq1 = ir - iq3 * neq2 as usize * neq1 as usize - iq2 * neq1 as usize;

        let h = iq2 as u32;
        let slope = if max_bias > 0.0 {
            if h < n_head_log2 { m0.powf(h as f32 + 1.0) } else { m1.powf(2.0*(h - n_head_log2) as f32 + 1.0) }
        } else {
            1.0f32
        };

        let mut s_sum = 0.0f32;
        let mut m_val = f32::NEG_INFINITY;

        // zero VKQ accumulator
        for i in 0..dv { vkq32[i] = 0.0; }

        // copy Q row into buffer
        let pq = q_data.add(iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3) as *const f32;
        std::ptr::copy_nonoverlapping(pq, q_buf.as_mut_ptr(), dk);

        let ik2 = iq2 / rk2 as usize;
        let ik3 = iq3 / rk3 as usize;
        let iv2 = iq2 / rv2 as usize;
        let iv3 = iq3 / rv3 as usize;

        for ic in ic_start..ic_end {
            // mask value
            let mv = if !mask_data.is_null() {
                let mp = mask_data.add(
                    iq1 * mask_nb1
                    + (iq2 % mask_ne2 as usize) * mask_nb2
                    + (iq3 % mask_ne3 as usize) * mask_nb3,
                ) as *const f16;
                slope * (*mp.add(ic as usize)).to_f32()
            } else {
                0.0f32
            };
            if mv == f32::NEG_INFINITY { continue; }

            // KQ dot product
            let k_row = k_data.add(ic as usize * nbk1 + ik2 * nbk2 + ik3 * nbk3) as *const f32;
            let k_sl  = std::slice::from_raw_parts(k_row, dk);
            let mut s  = vec_dot_f32(dk, q_buf, k_sl);
            s *= scale_eff;
            if logit_softcap != 0.0 { s = logit_softcap * s.tanh(); }
            s += mv;

            let m_old = m_val;
            let (ms, vs) = if s > m_val {
                m_val = s;
                (f32::exp(m_old - m_val), 1.0f32)
            } else {
                (1.0f32, f32::exp(s - m_val))
            };

            vec_scale_f32(dv, vkq32, ms);

            // V row (assume F32)
            let v_row = v_data.add(ic as usize * nbv1 + iv2 * nbv2 + iv3 * nbv3) as *const f32;
            let v_sl  = std::slice::from_raw_parts(v_row, dv);
            vec_mad_f32(dv, vkq32, v_sl, vs);

            s_sum = s_sum * ms + vs;
        }

        // sinks (ic_start == 0 only)
        if !sinks_data.is_null() && ic_start == 0 {
            let sv = *(sinks_data as *const f32).add(h as usize);
            let (ms, vs) = if sv > m_val {
                let ms = f32::exp(m_val - sv);
                m_val = sv;
                vec_scale_f32(dv, vkq32, ms);
                (ms, 1.0f32)
            } else {
                (1.0f32, f32::exp(sv - m_val))
            };
            s_sum = s_sum * ms + vs;
        }

        if write_partials {
            let partial = partials.add(ir * partial_stride as usize);
            *partial = m_val;
            *partial.add(1) = s_sum;
            std::ptr::copy_nonoverlapping(vkq32.as_ptr(), partial.add(2), dv);
        } else {
            let s_inv = if s_sum == 0.0 { 0.0 } else { 1.0 / s_sum };
            vec_scale_f32(dv, vkq32, s_inv);

            let i1 = iq1;
            let i2 = iq2;
            let i3 = iq3;
            let dst_ptr = dst_data.add((i3 as usize * ne2 as usize * ne1 as usize + i2 + i1 * ne1 as usize) * nb1) as *mut f32;
            std::ptr::copy_nonoverlapping(vkq32.as_ptr(), dst_ptr, dv);
        }

        let _ = v32; // suppress unused warning
    }
}

// ─── ggml_flash_attn_ext_reduce_partials (ops.cpp lines 9142–9210) ────────────

/// Direct translation of `ggml_flash_attn_ext_reduce_partials` (ops.cpp line 9142).
pub unsafe fn flash_attn_ext_reduce_partials(
    // partials: [n_q_heads][n_chunks][2 + DV]
    partials_base: *const f32,
    partial_size: i64,
    n_q_heads: i64, n_chunks: i64, nek1: i64, chunk_size: i64,
    dv: usize,
    // dst
    dst_data: *mut u8, ne1: i64, ne2: i64, nb1: usize,
    // per-thread scratch (>= DV)
    thread_wdata: *mut f32,
    ith: usize, nth: usize,
) {
    let mut q_head = ith as i64;
    while q_head < n_q_heads {
        let mut m_final = f32::NEG_INFINITY;
        let mut s_final = 0.0f32;
        let vkq_final = std::slice::from_raw_parts_mut(thread_wdata, dv);
        for i in 0..dv { vkq_final[i] = 0.0; }

        for chunk_idx in 0..n_chunks {
            let ic_start = chunk_idx * chunk_size;
            if ic_start >= nek1 { continue; }

            let partial = partials_base.add((q_head * n_chunks + chunk_idx) as usize * partial_size as usize);
            let m_chunk   = *partial;
            let s_chunk   = *partial.add(1);
            let vkq_chunk = std::slice::from_raw_parts(partial.add(2), dv);

            if s_chunk == 0.0 { continue; }

            let m_new     = m_final.max(m_chunk);
            let scale_old = f32::exp(m_final - m_new);
            let scale_new = f32::exp(m_chunk - m_new);

            for d in 0..dv {
                vkq_final[d] = vkq_final[d] * scale_old + vkq_chunk[d] * scale_new;
            }
            s_final = s_final * scale_old + s_chunk * scale_new;
            m_final = m_new;
        }

        if s_final != 0.0 {
            let s_inv = 1.0 / s_final;
            vec_scale_f32(dv, vkq_final, s_inv);
        }

        // iq1=0, iq3=0 for decode
        let dst_ptr = dst_data.add((0usize * ne2 as usize * ne1 as usize + q_head as usize + 0 * ne1 as usize) * nb1) as *mut f32;
        std::ptr::copy_nonoverlapping(vkq_final.as_ptr(), dst_ptr, dv);

        q_head += nth as i64;
    }
}

// ─── SSM conv (ops.cpp lines 9703–9769) ───────────────────────────────────────

/// Direct translation of `ggml_compute_forward_ssm_conv_f32` (ops.cpp line 9703).
pub unsafe fn compute_forward_ssm_conv_f32(
    // conv_x: {d_conv - 1 + n_t, d_inner, n_s}
    src0: *const f32,
    src0_nb0: usize, src0_nb1: usize, src0_nb2: usize,
    ne00: usize, // d_conv (= nc)
    // conv weight: {d_conv, d_inner}
    src1: *const f32,
    src1_nb1: usize,
    // dst: {d_inner, n_t, n_s}
    dst: *mut f32,
    dst_nb0: usize, dst_nb1: usize, dst_nb2: usize,
    nc: usize,   // d_conv
    ncs: usize,  // d_conv - 1 + n_t
    nr: usize,   // d_inner
    n_t: usize,  // tokens per sequence
    n_s: usize,  // number of sequences
    ir0: usize, ir1: usize,  // d_inner row range for this thread
) {
    let ir = ir1 - ir0;
    let _ = (ne00, src0_nb0, ncs);

    for i3 in 0..n_s {
        for i2 in 0..n_t {
            let s = (src0 as *const u8).add(ir0 * src0_nb1 + i2 * std::mem::size_of::<f32>() + i3 * src0_nb2) as *const f32;
            let c = (src1 as *const u8).add(ir0 * src1_nb1) as *const f32;
            let x = (dst as *mut u8).add(ir0 * dst_nb0 + i2 * dst_nb1 + i3 * dst_nb2) as *mut f32;

            for i1 in 0..ir {
                let mut sumf = 0.0f32;
                for i0 in 0..nc {
                    sumf += *s.add(i0 + i1 * ncs) * *c.add(i0 + i1 * nc);
                }
                *x.add(i1) = sumf;
            }
        }
    }
}

// ─── SSM scan scalar (ops.cpp lines 9773–9994) ────────────────────────────────

/// Direct translation of `ggml_compute_forward_ssm_scan_f32` (scalar Mamba-1 path).
/// ops.cpp line 9773.
#[allow(clippy::too_many_arguments)]
pub unsafe fn compute_forward_ssm_scan_f32(
    // s: {d_state, dim, n_head, ns+}   (initial states)
    src0: *const f32, src0_nb3: usize,
    // x: {dim, n_head, n_seq_tokens, n_seqs}
    src1: *const f32, src1_nb2: usize, src1_nb3: usize,
    // dt: {n_head, n_seq_tokens, n_seqs}
    src2: *const f32, src2_nb1: usize, src2_nb2: usize,
    // A: {d_state, n_head} or {1, n_head}
    src3: *const f32,
    src3_ne0: i64,
    // B: {d_state, n_group, n_seq_tokens, n_seqs}
    src4: *const f32, src4_nb2: usize, src4_nb3: usize,
    // C: {d_state, n_group, n_seq_tokens, n_seqs}
    src5: *const f32, src5_nb2: usize, src5_nb3: usize,
    // ids: {n_seqs}
    src6: *const i32,
    // dst: layout x first, then state snapshots
    dst: *mut f32,
    s_off: usize,      // byte offset to state area (= nelements(src1)*sizeof(f32))
    nc: i64,  // d_state
    nr: i64,  // dim
    nh: i64,  // n_head
    ng: i64,  // n_group
    nt: i64,  // tokens per sequence
    ns: i64,  // n_seqs
    k_snap: i64, // K op param
    ih0: usize, ih1: usize, // head range for this thread
) {
    for i3 in 0..ns as usize {
        let s0_base = src0.add(*src6.add(i3) as usize * src0_nb3 / std::mem::size_of::<f32>());
        let s_base  = (dst as *mut u8).add(i3 * src0_nb3 + s_off) as *mut f32;

        let mut s0_ptr = s0_base;
        let mut s_ptr  = s_base;

        for i2 in 0..nt as usize {
            let x  = src1.add(i2 * src1_nb2 / std::mem::size_of::<f32>() + i3 * src1_nb3 / std::mem::size_of::<f32>());
            let dt = src2.add(i2 * src2_nb1 / std::mem::size_of::<f32>() + i3 * src2_nb2 / std::mem::size_of::<f32>());
            let a  = src3;
            let bv = src4.add(i2 * src4_nb2 / std::mem::size_of::<f32>() + i3 * src4_nb3 / std::mem::size_of::<f32>());
            let cv = src5.add(i2 * src5_nb2 / std::mem::size_of::<f32>() + i3 * src5_nb3 / std::mem::size_of::<f32>());
            let y  = dst.add(i2 * (nh * nr) as usize + i3 * (nt * nh * nr) as usize);

            if src3_ne0 == 1 {
                // Mamba-2: scalar decay per head
                for h in ih0..ih1 {
                    let dt_soft = softplus_f32(*dt.add(h));
                    let da = (dt_soft * *a.add(h)).exp();
                    let g = h / (nh as usize / ng as usize);

                    for i1 in 0..nr as usize {
                        let ii = i1 + h * nr as usize;
                        let x_dt = *x.add(ii) * dt_soft;
                        let mut sumf = 0.0f32;

                        for i0 in 0..nc as usize {
                            let i = i0 + ii * nc as usize;
                            let ig = i0 + g * nc as usize;
                            let state = (*s0_ptr.add(i) * da) + (*bv.add(ig) * x_dt);
                            sumf += state * *cv.add(ig);
                            *s_ptr.add(i) = state;
                        }
                        *y.add(ii) = sumf;
                    }
                }
            } else {
                // Mamba-1: element-wise decay
                for h in ih0..ih1 {
                    let dt_soft = softplus_f32(*dt.add(h));
                    let g = h / (nh as usize / ng as usize);

                    for i1 in 0..nr as usize {
                        let ii = i1 + h * nr as usize;
                        let x_dt = *x.add(ii) * dt_soft;
                        let mut sumf = 0.0f32;

                        for i0 in 0..nc as usize {
                            let i = i0 + ii * nc as usize;
                            let ig = i0 + g * nc as usize;
                            let state = (*s0_ptr.add(i) * (dt_soft * *a.add(i0 + h * nc as usize)).exp()) + (*bv.add(ig) * x_dt);
                            sumf += state * *cv.add(ig);
                            *s_ptr.add(i) = state;
                        }
                        *y.add(ii) = sumf;
                    }
                }
            }

            // state snapshots (K > 1)
            if k_snap > 1 {
                let slot = nt as usize - 1 - i2;
                if slot > 0 && (slot as i64) < k_snap {
                    // simplified: just write the state for this thread's heads
                    for h in ih0..ih1 {
                        let snap_off = (slot as usize * ns as usize + i3) * src0_nb3 / std::mem::size_of::<f32>()
                            + h * nr as usize * nc as usize;
                        let snap_ptr = (dst as *mut u8).add(s_off).add(snap_off * std::mem::size_of::<f32>()) as *mut f32;
                        for i0 in 0..(nr * nc) as usize {
                            *snap_ptr.add(i0) = *s_ptr.add(h * nr as usize * nc as usize + i0);
                        }
                    }
                }
            }

            // use output state as next step's input
            s0_ptr = s_ptr;
        }
        let _ = s_ptr;
    }
}

#[inline]
fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0f32 + x.exp()).ln() }
}

// ─── Win part / unpart (ops.cpp lines 10011–10133) ───────────────────────────

/// Direct translation of `ggml_compute_forward_win_part_f32` (ops.cpp line 10013).
pub unsafe fn compute_forward_win_part_f32(
    src_data: *const f32,
    ne00: i64, ne01: i64, ne02: i64,
    dst_data: *mut f32,
    ne0: i64, ne1: i64, ne2: i64,
    nep0: i32, nep1: i32, w: i32,
) {
    for py in 0..nep1 {
        for px in 0..nep0 {
            let i3 = py * nep0 + px;
            for i2 in 0..ne2 {
                for i1 in 0..ne1 {
                    for i0 in 0..ne0 {
                        let i02 = py as i64 * w as i64 + i2;
                        let i01 = px as i64 * w as i64 + i1;
                        let i00 = i0;

                        let dst_idx = (i3 as i64 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0 + i0) as usize;
                        let src_idx = (i02 * ne01 * ne00 + i01 * ne00 + i00) as usize;

                        if (py as i64 * w as i64 + i2) >= ne02 || (px as i64 * w as i64 + i1) >= ne01 {
                            *dst_data.add(dst_idx) = 0.0f32;
                        } else {
                            *dst_data.add(dst_idx) = *src_data.add(src_idx);
                        }
                    }
                }
            }
        }
    }
}

/// Direct translation of `ggml_compute_forward_win_unpart_f32` (ops.cpp line 10076).
pub unsafe fn compute_forward_win_unpart_f32(
    src_data: *const f32,
    ne00: i64, ne01: i64, ne02: i64,
    dst_data: *mut f32,
    ne0: i64, ne1: i64, ne2: i64,
    w: i32,
) {
    let px = (w - (ne1 % w as i64) as i32) % w;
    let npx = (px as i64 + ne1) / w as i64;

    let _ = (ne02, ne00);

    for i2 in 0..ne2 {
        for i1 in 0..ne1 {
            for i0 in 0..ne0 {
                let ip2 = i2 / w as i64;
                let ip1 = i1 / w as i64;

                let i02 = i2 % w as i64;
                let i01 = i1 % w as i64;
                let i00 = i0;

                let src_idx = ((ip2 * npx + ip1) * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00 + i00) as usize;
                let dst_idx = (i2 * ne1 * ne0 + i1 * ne0 + i0) as usize;

                *dst_data.add(dst_idx) = *src_data.add(src_idx);
            }
        }
    }
}

// ─── get_rel_pos (ops.cpp lines 10285–10328) ──────────────────────────────────

/// Direct translation of `ggml_compute_forward_get_rel_pos_f16` (ops.cpp line 10285).
pub unsafe fn compute_forward_get_rel_pos_f16(
    src0_data: *const f16,
    ne00: i64,
    dst_data: *mut f16,
    ne0: i64, ne1: i64, ne2: i64,
) {
    let w = ne1;
    for i2 in 0..ne2 {
        for i1 in 0..ne1 {
            let pos = (w - i1 - 1) + i2;
            for i0 in 0..ne0 {
                let dst_idx = (i2 * ne1 * ne0 + i1 * ne0 + i0) as usize;
                let src_idx = (pos * ne00 + i0) as usize;
                *dst_data.add(dst_idx) = *src0_data.add(src_idx);
            }
        }
    }
}

// ─── add_rel_pos (ops.cpp lines 10332–10408) ──────────────────────────────────

/// Direct translation of `ggml_compute_forward_add_rel_pos_f32` (ops.cpp line 10332).
pub unsafe fn compute_forward_add_rel_pos_f32(
    src1_data: *const f32,
    src2_data: *const f32,
    dst_data:  *mut f32,
    ne10: i64, ne11: i64, ne12: i64, ne13: i64,
    ip0: usize, ip1: usize,
) {
    for i13 in ip0..ip1 {
        for i12 in 0..ne12 as usize {
            for i11 in 0..ne11 as usize {
                let jp1 = i13 * ne12 as usize * ne11 as usize * ne10 as usize
                    + i12 * ne11 as usize * ne10 as usize
                    + i11 * ne10 as usize;
                for i10 in 0..ne10 as usize {
                    let jp0 = jp1 + i10;
                    let src1_e = *src1_data.add(jp0);
                    let src2_e = *src2_data.add(jp0);

                    let jdh = jp0 * ne10 as usize;
                    let jdw = jdh - (ne10 as usize - 1) * i10;

                    for j in 0..ne10 as usize {
                        *dst_data.add(jdh + j)      += src2_e;
                        *dst_data.add(jdw + j * ne10 as usize) += src1_e;
                    }
                }
            }
        }
    }
}

// ─── RWKV WKV6 scalar (ops.cpp lines 10413–10621) ────────────────────────────

/// Direct translation of scalar path of `ggml_compute_forward_rwkv_wkv6_f32`.
/// ops.cpp line 10413.
pub unsafe fn compute_forward_rwkv_wkv6_f32(
    k_data:          *const f32,
    v_data:          *const f32,
    r_data:          *const f32,
    time_faaaa_data: *const f32,
    time_decay_data: *const f32,
    state_in_data:   *const f32,
    dst_data:        *mut f32,
    state_out:       *mut f32,
    t_cap: i64,     // T
    c: i64,         // C
    heads: i64,     // HEADS
    n_seqs: i64,
    h_start: usize,
    h_end: usize,
) {
    let t_stride    = (heads * c / heads) as usize; // == C
    let h_stride    = (c / heads) as usize;
    let h_stride_2d = h_stride * h_stride;
    let head_size   = h_stride;

    for t in 0..t_cap as usize {
        let t_offset = t * t_stride;
        let state_offset = head_size * c as usize * (t / (t_cap as usize / n_seqs as usize));
        let state_cur  = state_out.add(state_offset);
        let state_prev = if t % (t_cap as usize / n_seqs as usize) != 0 {
            state_cur as *const f32
        } else {
            state_in_data.add(state_offset)
        };

        for h in h_start..h_end {
            let h_offset    = h * h_stride;
            let t_h_offset  = t_offset + h_offset;
            let h_2d_offset = h * h_stride_2d;

            for i in 0..head_size {
                let t_h_i_offset  = t_h_offset + i;
                let h_i_offset    = h * h_stride + i;
                let h_2d_i_offset = h_2d_offset + i * h_stride;

                let k_val          = *k_data.add(t_h_i_offset);
                let r_val          = *r_data.add(t_h_i_offset);
                let time_faaaa_val = *time_faaaa_data.add(h_i_offset);
                let time_decay_val = *time_decay_data.add(t_h_i_offset);

                for j in 0..head_size {
                    let t_h_j_offset    = t_h_offset + j;
                    let h_2d_i_j_offset = h_2d_i_offset + j;

                    let v_val          = *v_data.add(t_h_j_offset);
                    let kv_val         = v_val * k_val;
                    let prev_state_val = *state_prev.add(h_2d_i_j_offset);
                    let temp_val       = kv_val * time_faaaa_val + prev_state_val;
                    *dst_data.add(t_h_j_offset) += temp_val * r_val;
                    *state_cur.add(h_2d_i_j_offset) = prev_state_val * time_decay_val + kv_val;
                }
            }
        }
    }
}

// ─── GLA scalar (ops.cpp lines 10625–10822) ───────────────────────────────────

/// Direct translation of scalar path of `ggml_compute_forward_gla_f32`.
/// ops.cpp line 10625.
pub unsafe fn compute_forward_gla_f32(
    k_data:        *const f32,
    v_data:        *const f32,
    q_data:        *const f32,
    g_data:        *const f32,
    state_in_data: *const f32,
    dst_data:      *mut f32,
    state_out:     *mut f32,
    t_cap: i64, c: i64, heads: i64, n_seqs: i64,
    scale: f32,
    h_start: usize, h_end: usize,
) {
    let t_stride    = c as usize;
    let h_stride    = (c / heads) as usize;
    let head_size   = h_stride;
    let h_stride_2d = head_size * head_size;

    for t in 0..t_cap as usize {
        let t_offset = t * t_stride;
        let state_offset = head_size * c as usize * (t / (t_cap as usize / n_seqs as usize));
        let state_cur  = state_out.add(state_offset);
        let state_prev: *const f32 = if t % (t_cap as usize / n_seqs as usize) != 0 {
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

                for j in 0..head_size {
                    let t_h_j_offset    = t_h_offset + j;
                    let h_2d_i_j_offset = h_2d_i_offset + j;

                    let v_val          = *v_data.add(t_h_j_offset);
                    let kv_val         = v_val * k_val;
                    let prev_state_val = *state_prev.add(h_2d_i_j_offset);
                    let temp_val       = prev_state_val * g_val + kv_val;
                    *dst_data.add(t_h_j_offset) += temp_val * q_val;
                    *state_cur.add(h_2d_i_j_offset) = temp_val;
                }
            }
        }
    }
}

// ─── solve_tri (ops.cpp lines 10824–10892) ────────────────────────────────────

/// Direct translation of `ggml_compute_forward_solve_tri_f32` (ops.cpp line 10824).
pub unsafe fn compute_forward_solve_tri_f32(
    a: *const f32,
    nb02: usize, nb03: usize,
    b: *const f32,
    nb12: usize, nb13: usize,
    x: *mut f32,
    nb2: usize, nb3: usize,
    ne10: i64, ne11: i64, ne02: i64,
    ir0: usize, ir1: usize,
) {
    let k = ne10 as usize;
    let n = ne11 as usize;

    for ir in ir0..ir1 {
        let i03 = ir / (ne02 as usize * k);
        let i02 = (ir - i03 * ne02 as usize * k) / k;
        let i01 = ir - i03 * ne02 as usize * k - i02 * k;

        let a_batch = (a as *const u8).add(i02 * nb02 + i03 * nb03) as *const f32;
        let b_batch = (b as *const u8).add(i02 * nb12 + i03 * nb13) as *const f32;
        let x_batch = (x as *mut u8).add(i02 * nb2 + i03 * nb3) as *mut f32;

        for i00 in 0..n {
            let mut sum = 0.0f32;
            for t in 0..i00 {
                sum += *a_batch.add(i00 * n + t) * *x_batch.add(t * k + i01);
            }
            let diag = *a_batch.add(i00 * n + i00);
            assert!(diag != 0.0, "zero diagonal in triangular matrix");
            *x_batch.add(i00 * k + i01) = (*b_batch.add(i00 * k + i01) - sum) / diag;
        }
    }
}

// ─── Gated delta net (ops.cpp lines 10895–11105) ─────────────────────────────

/// Direct translation of `ggml_compute_forward_gated_delta_net_one_chunk` (ops.cpp line 10895).
#[allow(clippy::too_many_arguments)]
pub unsafe fn compute_forward_gated_delta_net_one_chunk(
    // q: {S_v, H, n_tokens, n_seqs}
    q_data: *const u8, nbq1: usize, nbq2: usize, nbq3: usize,
    neq1: i64, neq3: i64,
    // k: same shape
    k_data: *const u8, nbk1: usize, nbk2: usize, nbk3: usize,
    nek1: i64, nek3: i64,
    // v
    v_data: *const u8, nbv1: usize, nbv2: usize, nbv3: usize,
    // g: {1 or S_v, H, n_tokens, n_seqs}
    g_data: *const u8, nbg1: usize, nbg2: usize, nbg3: usize,
    neg0: i64,
    // beta: scalar per (H, t, seq)
    beta_data: *const u8, nbb1: usize, nbb2: usize, nbb3: usize,
    // state: {S_v, S_v, H, n_seqs}
    state_data: *const f32,
    state_seq_stride: usize,
    // dst
    dst_data: *mut u8,
    attn_score_elems: usize,
    state_size_per_snap: usize,
    s_v: i64, h_cap: i64, n_tokens: i64, n_seqs: i64,
    k_snap: i64,
    rq3: i64, rk3: i64,
    // scratch: [delta: S_v] [state_work: S_v*S_v if K>1]
    delta: *mut f32,
    state_work: *mut f32,
    ir0: i64, ir1: i64,
) {
    let sv = s_v as usize;
    let scale = 1.0f32 / (sv as f32).sqrt();
    let kda = neg0 == s_v;

    let attn_out_base  = dst_data as *mut f32;
    let state_out_base = (dst_data as *mut f32).add(attn_score_elems);

    for ir in ir0..ir1 {
        let iv1 = (ir % h_cap) as usize;  // head index
        let iv3 = (ir / h_cap) as usize;  // sequence

        let iq1 = iv1 % neq1 as usize;
        let ik1 = iv1 % nek1 as usize;
        let iq3 = iv3 / rq3 as usize;
        let ik3 = iv3 / rk3 as usize;

        let s_out: *mut f32 = if k_snap > 1 {
            state_work
        } else {
            state_out_base.add((iv3 * h_cap as usize + iv1) * sv * sv)
        };

        // copy initial state
        let s_in = state_data.add(iv3 * state_seq_stride + iv1 * sv * sv);
        std::ptr::copy_nonoverlapping(s_in, s_out, sv * sv);

        let attn_data = attn_out_base.add((iv3 * n_tokens as usize * h_cap as usize + iv1) * sv);
        let delta_sl = std::slice::from_raw_parts_mut(delta, sv);
        let s_sl = std::slice::from_raw_parts_mut(s_out, sv * sv);

        for t in 0..n_tokens as usize {
            let q_d = (q_data.add(iq3 * nbq3 + t * nbq2 + iq1 * nbq1)) as *const f32;
            let k_d = (k_data.add(ik3 * nbk3 + t * nbk2 + ik1 * nbk1)) as *const f32;
            let v_d = (v_data.add(iv3 * nbv3 + t * nbv2 + iv1 * nbv1)) as *const f32;
            let beta_val = *(beta_data.add(iv3 * nbb3 + t * nbb2 + iv1 * nbb1) as *const f32);
            let g_d = (g_data.add(iv3 * nbg3 + t * nbg2 + iv1 * nbg1)) as *const f32;

            let kd_sl = std::slice::from_raw_parts(k_d, sv);
            let vd_sl = std::slice::from_raw_parts(v_d, sv);
            let qd_sl = std::slice::from_raw_parts(q_d, sv);

            // apply gate decay to state columns
            if kda {
                for i in 0..sv { delta_sl[i] = g_d.add(i).read().exp(); }
                for j in 0..sv {
                    let row = &mut s_sl[j*sv..(j+1)*sv];
                    for i in 0..sv { row[i] *= delta_sl[i]; }
                }
            } else {
                let g_exp = g_d.read().exp();
                for v in s_sl.iter_mut() { *v *= g_exp; }
            }

            // delta[j] = (v[j] - dot(row j of M, k)) * beta
            for j in 0..sv {
                let sum = vec_dot_f32(sv, &s_sl[j*sv..(j+1)*sv], kd_sl);
                delta_sl[j] = (vd_sl[j] - sum) * beta_val;
            }

            // outer product update: M[j][i] += delta[j] * k[i]
            for j in 0..sv {
                let row = &mut s_sl[j*sv..(j+1)*sv];
                for i in 0..sv { row[i] += kd_sl[i] * delta_sl[j]; }
            }

            // attention output
            let attn_t = attn_data.add(t * sv * h_cap as usize);
            for j in 0..sv {
                let sum = vec_dot_f32(sv, &s_sl[j*sv..(j+1)*sv], qd_sl);
                *attn_t.add(j) = sum * scale;
            }

            // write state snapshot for K>1
            if k_snap > 1 {
                let target_slot = n_tokens as usize - 1 - t;
                if target_slot < k_snap as usize {
                    let curr_state_o = state_out_base.add(
                        target_slot * state_size_per_snap
                        + (iv3 * h_cap as usize + iv1) * sv * sv,
                    );
                    std::ptr::copy_nonoverlapping(s_out, curr_state_o, sv * sv);
                }
            }
        }
    }
}

// ─── DSV4 HC comb helpers (ops.cpp lines 11110–11244) ────────────────────────

const DSV4_HC: usize = 4;

/// Direct translation of `ggml_dsv4_hc_comb_norm_cols` (ops.cpp line 11110).
pub fn dsv4_hc_comb_norm_cols(comb: &mut [f32; 16], eps: f32) {
    for idst in 0..DSV4_HC {
        let mut sum = eps;
        for isrc in 0..DSV4_HC {
            sum += comb[idst + DSV4_HC * isrc];
        }
        let inv_sum = 1.0 / sum;
        for isrc in 0..DSV4_HC {
            comb[idst + DSV4_HC * isrc] *= inv_sum;
        }
    }
}

/// Direct translation of `ggml_dsv4_hc_comb_norm_rows` (ops.cpp line 11126).
pub fn dsv4_hc_comb_norm_rows(comb: &mut [f32; 16], eps: f32) {
    for isrc in 0..DSV4_HC {
        let mut sum = eps;
        for idst in 0..DSV4_HC {
            sum += comb[idst + DSV4_HC * isrc];
        }
        let inv_sum = 1.0 / sum;
        for idst in 0..DSV4_HC {
            comb[idst + DSV4_HC * isrc] *= inv_sum;
        }
    }
}

/// Direct translation of `ggml_compute_forward_dsv4_hc_comb_f32` (ops.cpp line 11142).
#[allow(clippy::too_many_arguments)]
pub unsafe fn compute_forward_dsv4_hc_comb_f32(
    mixes_data: *const u8, nbm0: usize, nbm1: usize,
    base_data:  *const u8, nbb0: usize,
    scale_comb: f32,
    dst_data:   *mut u8,  nbd0: usize, nbd1: usize, nbd2: usize,
    n_tokens: i64,
    eps: f32, n_iter: i32,
    it0: i64, it1: i64,
) {
    let comb_offset: i64 = 2 * DSV4_HC as i64;

    for it in it0..it1 {
        let mut comb = [0.0f32; 16];

        for isrc in 0..DSV4_HC {
            let mut max = f32::NEG_INFINITY;
            for idst in 0..DSV4_HC {
                let idx = idst + DSV4_HC * isrc;
                let xv = *(mixes_data.add((comb_offset as usize + idx) * nbm0 + it as usize * nbm1) as *const f32);
                let bv = *(base_data.add((comb_offset as usize + idx) * nbb0) as *const f32);
                let v = xv * scale_comb + bv;
                comb[idx] = v;
                if v > max { max = v; }
            }

            let mut sum = 0.0f32;
            for idst in 0..DSV4_HC {
                let idx = idst + DSV4_HC * isrc;
                let v = (comb[idx] - max).exp();
                comb[idx] = v;
                sum += v;
            }

            let inv_sum = 1.0 / sum;
            for idst in 0..DSV4_HC {
                let idx = idst + DSV4_HC * isrc;
                comb[idx] = comb[idx] * inv_sum + eps;
            }
        }

        dsv4_hc_comb_norm_cols(&mut comb, eps);
        for _ in 1..n_iter {
            dsv4_hc_comb_norm_rows(&mut comb, eps);
            dsv4_hc_comb_norm_cols(&mut comb, eps);
        }

        for isrc in 0..DSV4_HC {
            for idst in 0..DSV4_HC {
                let idx = idst + DSV4_HC * isrc;
                let dst_ptr = dst_data.add(idst * nbd0 + isrc * nbd1 + it as usize * nbd2) as *mut f32;
                *dst_ptr = comb[idx];
            }
        }
        let _ = n_tokens;
    }
}

// ─── DSV4 HC pre (ops.cpp lines 11248–11324) ──────────────────────────────────

/// Direct translation of `ggml_compute_forward_dsv4_hc_pre_f32` (ops.cpp line 11248).
pub unsafe fn compute_forward_dsv4_hc_pre_f32(
    x_data:       *const u8, nbx0: usize, nbx1: usize, nbx2: usize,
    weights_data: *const u8, nbw0: usize, nbw1: usize, nbw2: usize,
    dst_data:     *mut u8,   nbd0: usize, nbd1: usize,
    n_embd: i64, hc: i64, n_tokens: i64,
    scale: f32, gated: bool,
    ir0: i64, ir1: i64,
) {
    for ir in ir0..ir1 {
        let i0 = ir % n_embd;
        let it = ir / n_embd;

        let mut sum = 0.0f32;
        for ih in 0..hc as usize {
            let xv = *(x_data.add(i0 as usize * nbx0 + ih * nbx1 + it as usize * nbx2) as *const f32);
            let wv = if gated {
                let gv = *(weights_data.add(i0 as usize * nbw0 + ih * nbw1 + it as usize * nbw2) as *const f32);
                1.0 / (1.0 + (-gv).exp())
            } else {
                *(weights_data.add(ih * nbw0 + it as usize * nbw1) as *const f32)
            };
            sum += xv * wv;
        }

        let dst_ptr = dst_data.add(i0 as usize * nbd0 + it as usize * nbd1) as *mut f32;
        *dst_ptr = scale * sum;
    }
}

// ─── DSV4 HC post (ops.cpp lines 11328–11418) ─────────────────────────────────

/// Direct translation of `ggml_compute_forward_dsv4_hc_post_f32` (ops.cpp line 11328).
#[allow(clippy::too_many_arguments)]
pub unsafe fn compute_forward_dsv4_hc_post_f32(
    x_data:       *const u8, nbx0: usize, nbx1: usize,
    residual:     *const u8, nbr0: usize, nbr1: usize, nbr2: usize,
    post_data:    *const u8, nbp0: usize, nbp1: usize,
    comb_data:    *const u8, nbc0: usize, nbc1: usize, nbc2: usize,
    has_comb: bool,
    dst_data:     *mut u8,   nbd0: usize, nbd1: usize, nbd2: usize,
    n_embd: i64, hc: i64, n_tokens: i64,
    ir0: i64, ir1: i64,
) {
    for ir in ir0..ir1 {
        let i0   = ir % n_embd;
        let idst = (ir / n_embd) % hc;
        let it   = ir / (n_embd * hc);

        let xv = *(x_data.add(i0 as usize * nbx0 + it as usize * nbx1) as *const f32);
        let pv = *(post_data.add(idst as usize * nbp0 + it as usize * nbp1) as *const f32);

        let mut sum = xv * pv;
        if has_comb {
            for isrc in 0..hc as usize {
                let rv = *(residual.add(i0 as usize * nbr0 + isrc * nbr1 + it as usize * nbr2) as *const f32);
                let cv = *(comb_data.add(idst as usize * nbc0 + isrc * nbc1 + it as usize * nbc2) as *const f32);
                sum += rv * cv;
            }
        } else {
            sum += *(residual.add(i0 as usize * nbr0 + idst as usize * nbr1 + it as usize * nbr2) as *const f32);
        }

        *(dst_data.add(i0 as usize * nbd0 + idst as usize * nbd1 + it as usize * nbd2) as *mut f32) = sum;
        let _ = n_tokens;
    }
}

// ─── RWKV WKV7 scalar (ops.cpp lines 11422–11635) ────────────────────────────

/// Direct translation of scalar path of `ggml_compute_forward_rwkv_wkv7_f32`.
/// ops.cpp line 11422.
pub unsafe fn compute_forward_rwkv_wkv7_f32(
    r_data:          *const f32,
    w_data:          *const f32,
    k_data:          *const f32,
    v_data:          *const f32,
    a_data:          *const f32,
    b_data:          *const f32,
    state_in_data:   *const f32,
    dst_data:        *mut f32,
    state_out:       *mut f32,
    t_cap: i64, c: i64, heads: i64, n_seqs: i64,
    h_start: usize, h_end: usize,
) {
    let t_stride    = c as usize;
    let h_stride    = (c / heads) as usize;
    let head_size   = h_stride;
    let h_stride_2d = head_size * head_size;

    for t in 0..t_cap as usize {
        let t_offset = t * t_stride;
        let state_offset = head_size * c as usize * (t / (t_cap as usize / n_seqs as usize));
        let state_cur  = state_out.add(state_offset);
        let state_prev: *const f32 = if t % (t_cap as usize / n_seqs as usize) != 0 {
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

                let v_val = *v_data.add(t_h_i_offset);

                // sa = dot(a[t_h], state_prev[h_2d_i])
                let mut sa = 0.0f32;
                for j in 0..head_size {
                    sa += *a_data.add(t_h_offset + j) * *state_prev.add(h_2d_i_offset + j);
                }

                let mut result = 0.0f32;
                for j in 0..head_size {
                    let t_h_j_offset    = t_h_offset + j;
                    let h_2d_i_j_offset = h_2d_i_offset + j;

                    let r_val = *r_data.add(t_h_j_offset);
                    let w_val = *w_data.add(t_h_j_offset);
                    let k_val = *k_data.add(t_h_j_offset);
                    let b_val = *b_data.add(t_h_j_offset);
                    let kv_val = v_val * k_val;
                    let prev_state_val = *state_prev.add(h_2d_i_j_offset);
                    let new_state = prev_state_val * w_val + kv_val + sa * b_val;
                    *state_cur.add(h_2d_i_j_offset) = new_state;
                    result += new_state * r_val;
                }
                *dst_data.add(t_h_i_offset) = result;
            }
        }
    }
}

// ─── Cross entropy loss (ops.cpp lines 11696–11788) ──────────────────────────

/// Direct translation of `ggml_compute_forward_cross_entropy_loss_f32` (ops.cpp line 11696).
pub unsafe fn compute_forward_cross_entropy_loss_f32(
    src0: *const u8, nb1_src0: usize,
    src1: *const u8, nb1_src1: usize,
    nc: i64, nr: i64,
    // thread-local scratch: [nc] floats for softmax buffer
    st: *mut f32,
    // per-thread sum output
    sum_out: *mut f32,
    ith: usize, nth: usize,
) {
    let dr = (nr + nth as i64 - 1) / nth as i64;
    let ir0 = dr * ith as i64;
    let ir1 = (ir0 + dr).min(nr);

    let mut sum_thread = 0.0f32;

    for i1 in ir0..ir1 {
        let s0 = src0.add(i1 as usize * nb1_src0) as *const f32;
        let s1 = src1.add(i1 as usize * nb1_src1) as *const f32;
        let s0_sl = std::slice::from_raw_parts(s0, nc as usize);
        let s1_sl = std::slice::from_raw_parts(s1, nc as usize);
        let st_sl = std::slice::from_raw_parts_mut(st, nc as usize);

        let mut max = f32::NEG_INFINITY;
        for v in s0_sl { if *v > max { max = *v; } }

        // log-softmax
        let mut sum_softmax = 0.0f64;
        for i in 0..nc as usize {
            let v = (s0_sl[i] - max).exp();
            st_sl[i] = v;
            sum_softmax += v as f64;
        }
        let log_sum = (sum_softmax as f32).ln();

        let mut sum_st = 0.0f32;
        for i in 0..nc as usize {
            st_sl[i] = st_sl[i].ln() - log_sum;
            st_sl[i] *= s1_sl[i];
            sum_st += st_sl[i];
        }
        sum_thread += sum_st;
    }

    *sum_out = sum_thread;
}

/// Direct translation of `ggml_compute_forward_cross_entropy_loss_back_f32` (ops.cpp line 11792).
pub unsafe fn compute_forward_cross_entropy_loss_back_f32(
    src0f: *const u8, nb1_src0f: usize,
    src1f: *const u8, nb1_src1f: usize,
    grad_scalar: f32,  // ((f32*)grad->data)[0]
    nc: i64, nr: i64,
    dst: *mut u8, nb1_dst: usize,
    ith: usize, nth: usize,
) {
    let dr = (nr + nth as i64 - 1) / nth as i64;
    let ir0 = dr * ith as i64;
    let ir1 = (ir0 + dr).min(nr);
    let d_by_nr = grad_scalar / nr as f32;

    for i1 in ir0..ir1 {
        let ds0 = dst.add(i1 as usize * nb1_dst) as *mut f32;
        let s0  = src0f.add(i1 as usize * nb1_src0f) as *const f32;
        let s1  = src1f.add(i1 as usize * nb1_src1f) as *const f32;

        let s0_sl = std::slice::from_raw_parts(s0, nc as usize);
        let s1_sl = std::slice::from_raw_parts(s1, nc as usize);
        let ds0_sl = std::slice::from_raw_parts_mut(ds0, nc as usize);

        // softmax
        let mut max = f32::NEG_INFINITY;
        for v in s0_sl { if *v > max { max = *v; } }
        let sum = vec_soft_max_f32(nc as usize, ds0_sl, s0_sl, max);
        let inv_sum = 1.0 / sum;
        for v in ds0_sl.iter_mut() { *v *= inv_sum; }

        // grad = (softmax - s1) * d_by_nr
        for i in 0..nc as usize {
            ds0_sl[i] = (ds0_sl[i] - s1_sl[i]) * d_by_nr;
        }
    }
}

// ─── AdamW optimizer (ops.cpp lines 11873–11956) ─────────────────────────────

/// Direct translation of `ggml_compute_forward_opt_step_adamw_f32` (ops.cpp line 11873).
#[allow(clippy::too_many_arguments)]
pub unsafe fn compute_forward_opt_step_adamw_f32(
    src0: *const u8,
    nb00: usize, nb01: usize, nb02: usize, nb03: usize,
    ne00: i64, ne01: i64, ne02: i64,
    src0_grad:   *const u8,
    src0_grad_m: *mut u8,
    src0_grad_v: *mut u8,
    alpha: f32, beta1: f32, beta2: f32, eps: f32,
    beta1h: f32, beta2h: f32,
    keep: f32,  // 1 - alpha * wd
    ir0: usize, ir1: usize,
) {
    for ir in ir0..ir1 {
        let i03 = ir / (ne02 as usize * ne01 as usize);
        let i02 = (ir - i03 * ne02 as usize * ne01 as usize) / ne01 as usize;
        let i01 = ir - i03 * ne02 as usize * ne01 as usize - i02 * ne01 as usize;

        let offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

        let w = (src0.add(offset)) as *mut f32;
        let g = (src0_grad.add(offset)) as *const f32;
        let m = (src0_grad_m.add(offset)) as *mut f32;
        let v = (src0_grad_v.add(offset)) as *mut f32;

        for i00 in 0..ne00 as usize {
            let g_val = *g.add(i00);
            *m.add(i00) = *m.add(i00) * beta1 + g_val * (1.0 - beta1);
            *v.add(i00) = *v.add(i00) * beta2 + g_val * g_val * (1.0 - beta2);

            let mh = *m.add(i00) * beta1h;
            let vh = (*v.add(i00) * beta2h).sqrt() + eps;
            *w.add(i00) = *w.add(i00) * keep - alpha * mh / vh;
        }
        let _ = (nb00, src0);
    }
}

// ─── SGD optimizer (ops.cpp lines 11958–11999) ────────────────────────────────

/// Direct translation of `ggml_compute_forward_opt_step_sgd_f32` (ops.cpp line 11958).
pub unsafe fn compute_forward_opt_step_sgd_f32(
    src0:      *const u8,
    src0_grad: *const u8,
    nb01: usize, nb02: usize, nb03: usize,
    ne00: i64, ne01: i64, ne02: i64,
    alpha: f32, keep: f32,   // keep = 1 - alpha * wd
    ir0: usize, ir1: usize,
) {
    for ir in ir0..ir1 {
        let i03 = ir / (ne02 as usize * ne01 as usize);
        let i02 = (ir - i03 * ne02 as usize * ne01 as usize) / ne01 as usize;
        let i01 = ir - i03 * ne02 as usize * ne01 as usize - i02 * ne01 as usize;

        let offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

        let w = (src0.add(offset)) as *mut f32;
        let g = (src0_grad.add(offset)) as *const f32;

        for i00 in 0..ne00 as usize {
            *w.add(i00) = *w.add(i00) * keep - alpha * *g.add(i00);
        }
    }
}

// ─── FWHT (Fast Walsh-Hadamard Transform, ops.cpp lines 12018–12122) ──────────

/// Direct translation of `ggml_compute_forward_fwht_impl` for F32 src (ops.cpp line 12026).
pub unsafe fn compute_forward_fwht_f32(
    src1: *const u8, nb11: usize, nb12: usize, nb13: usize,
    dst:  *mut u8,   nb1:  usize, nb2:  usize, nb3:  usize,
    ne10: i64, ne11: i64, ne12: i64, ne13: i64,
    start_row: i64, end_row: i64,
) {
    let n = ne10 as usize;
    let scale = 1.0f32 / (n as f32).sqrt();

    for r in start_row..end_row {
        let i13 = r / (ne11 * ne12);
        let i12 = (r - i13 * ne11 * ne12) / ne11;
        let i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        let src_row = src1.add(i11 as usize * nb11 + i12 as usize * nb12 + i13 as usize * nb13) as *const f32;
        let dst_row = dst.add(i11 as usize * nb1 + i12 as usize * nb2 + i13 as usize * nb3) as *mut f32;
        let dst_sl  = std::slice::from_raw_parts_mut(dst_row, n);

        for j in 0..n { dst_sl[j] = *src_row.add(j) * scale; }

        let mut len = 1;
        while len < n {
            let mut i = 0;
            while i < n {
                for j in 0..len {
                    let u = dst_sl[i + j];
                    let v = dst_sl[i + len + j];
                    dst_sl[i + j]       = u + v;
                    dst_sl[i + len + j] = u - v;
                }
                i += 2 * len;
            }
            len <<= 1;
        }
    }
}

/// Direct translation of `ggml_compute_forward_fwht_impl` for F16 src (ops.cpp line 12026).
pub unsafe fn compute_forward_fwht_f16(
    src1: *const u8, nb11: usize, nb12: usize, nb13: usize,
    dst:  *mut u8,   nb1:  usize, nb2:  usize, nb3:  usize,
    ne10: i64, ne11: i64, ne12: i64, ne13: i64,
    start_row: i64, end_row: i64,
) {
    let n = ne10 as usize;
    let scale = 1.0f32 / (n as f32).sqrt();

    for r in start_row..end_row {
        let i13 = r / (ne11 * ne12);
        let i12 = (r - i13 * ne11 * ne12) / ne11;
        let i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        let src_row = src1.add(i11 as usize * nb11 + i12 as usize * nb12 + i13 as usize * nb13) as *const f16;
        let dst_row = dst.add(i11 as usize * nb1 + i12 as usize * nb2 + i13 as usize * nb3) as *mut f32;
        let dst_sl  = std::slice::from_raw_parts_mut(dst_row, n);

        for j in 0..n { dst_sl[j] = src_row.add(j).read().to_f32() * scale; }

        let mut len = 1;
        while len < n {
            let mut i = 0;
            while i < n {
                for j in 0..len {
                    let u = dst_sl[i + j];
                    let v = dst_sl[i + len + j];
                    dst_sl[i + j]       = u + v;
                    dst_sl[i + len + j] = u - v;
                }
                i += 2 * len;
            }
            len <<= 1;
        }
    }
}

// ─── Lightning indexer (ops.cpp lines 12124–12206) ───────────────────────────

/// Direct translation of `ggml_compute_forward_lightning_indexer` (ops.cpp line 12126).
#[allow(clippy::too_many_arguments)]
pub unsafe fn compute_forward_lightning_indexer(
    // q: {n_embd, n_head, n_tokens, n_stream}
    q_data: *const u8, nbq1: usize, nbq2: usize, nbq3: usize,
    // k: {n_embd, ?, n_kv, n_stream}
    k_data: *const u8, nbk2: usize, nbk3: usize,
    // w: {n_head, n_tokens, 1, n_stream}
    w_data: *const u8, nbw1: usize, nbw3: usize,
    // m: {n_kv, n_tokens, 1, n_stream%n_stream_m}
    m_data: *const u8, nbm1: usize, nbm3: usize,
    nem3: i64,
    // dst: {n_kv, n_tokens, 1, n_stream}
    dst_data: *mut u8, nb1: usize, nb3: usize,
    n_embd: i32, n_head: i32, n_tokens: i32, n_stream: i32, n_kv: i32,
    // k_to_float: None = k is already F32
    k_is_f32: bool,
    k_row_f32: *mut f32,  // scratch buffer per thread: n_embd floats
    ir0: i32, ir1: i32,   // n_kv range for this thread
) {
    for s in 0..n_stream {
        for t in 0..n_tokens {
            let w_row  = w_data.add((t as usize) * nbw1 + (s as usize) * nbw3) as *const f32;
            let m_row  = m_data.add((t as usize) * nbm1 + (s as usize % nem3 as usize) * nbm3) as *const f16;
            let dst_row = dst_data.add((t as usize) * nb1 + (s as usize) * nb3) as *mut f32;

            for ik in ir0..ir1 {
                let k_row_ptr = k_data.add((ik as usize) * nbk2 + (s as usize) * nbk3);

                let k_sl = if k_is_f32 {
                    std::slice::from_raw_parts(k_row_ptr as *const f32, n_embd as usize)
                } else {
                    // dequantize into scratch
                    let kf16 = k_row_ptr as *const f16;
                    for i in 0..n_embd as usize {
                        *k_row_f32.add(i) = (*kf16.add(i)).to_f32();
                    }
                    std::slice::from_raw_parts(k_row_f32 as *const f32, n_embd as usize)
                };

                let mut score = 0.0f32;
                for h in 0..n_head as usize {
                    let q_row = q_data.add(h * nbq1 + (t as usize) * nbq2 + (s as usize) * nbq3) as *const f32;
                    let q_sl  = std::slice::from_raw_parts(q_row, n_embd as usize);
                    let qk    = vec_dot_f32(n_embd as usize, q_sl, k_sl);
                    score    += qk.max(0.0) * *w_row.add(h);
                }
                *dst_row.add(ik as usize) = score + (*m_row.add(ik as usize)).to_f32();
            }
        }
    }
}
