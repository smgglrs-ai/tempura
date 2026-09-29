//! Faithful Rust translation of ops.cpp:
//! ssm_conv_f32, ssm_scan_f32, win_part_f32, win_unpart_f32,
//! rwkv_wkv6_f32, rwkv_wkv7_f32, gla_f32,
//! gated_delta_net_one_chunk, gated_delta_net_f32,
//! dsv4_hc_comb_f32, dsv4_hc_pre_f32, dsv4_hc_post_f32,
//! fwht_f32.

// ── ssm_conv_f32 (ops.cpp:9703) ──────────────────────────────────────────────

/// Translation of ggml_compute_forward_ssm_conv_f32.
/// src0: conv_x {d_conv - 1 + n_t, d_inner, n_seqs}
/// src1: conv1d.weight {d_conv, d_inner}
/// dst:  {d_inner, n_t, n_seqs}
pub fn ssm_conv_f32(
    src0: &[f32], // {ncs, nr, n_s} row-major
    src1: &[f32], // {nc, nr} row-major (weights)
    dst:  &mut [f32], // {nr, n_t, n_s}
    nc: usize,  // d_conv
    ncs: usize, // d_conv - 1 + n_t
    nr: usize,  // d_inner
    n_t: usize, // tokens per sequence
    n_s: usize, // number of sequences
    ith: usize, nth: usize,
) {
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = ir0.min(nr) + dr.min(nr.saturating_sub(ir0));
    let ir = ir1 - ir0.min(ir1);

    for i3 in 0..n_s {
        for i2 in 0..n_t {
            for i1 in 0..ir {
                let row = ir0 + i1;
                let mut sumf = 0.0f32;
                for i0 in 0..nc {
                    let s_idx = i0 + i2 * 1 + row * ncs + i3 * (ncs * nr);
                    let c_idx = i0 + row * nc;
                    // src0[i0 + i1*ncs + i3*ncs*nr] (sliding window at offset i2)
                    let sv = src0[(i0 + i2) + row * ncs + i3 * (ncs * nr)];
                    let cv = src1[i0 + row * nc];
                    let _ = s_idx; // suppress unused
                    sumf += sv * cv;
                }
                dst[row + i2 * nr + i3 * (nr * n_t)] = sumf;
            }
        }
    }
}

// ── ssm_scan_f32 (ops.cpp:9773) ──────────────────────────────────────────────

/// Softplus: log(1 + exp(x)), numerically stable version.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Translation of ggml_compute_forward_ssm_scan_f32 (scalar branch, Mamba-1 and Mamba-2).
///
/// Mamba-2 (scalar A per head): `a_is_scalar = true`
/// Mamba-1 (element-wise A):    `a_is_scalar = false`
///
/// All tensors are provided as flat contiguous f32 slices with explicit strides.
/// Layout mirrors the C struct member access:
///   s0:  [d_state, dim, n_head, n_seqs]  — previous state
///   s:   [d_state, dim, n_head, n_seqs]  — new state (output portion)
///   x:   [dim, n_head, n_seq_tokens, n_seqs]
///   dt:  [n_head, n_seq_tokens, n_seqs]
///   A:   [d_state, n_head] (element-wise) or [1, n_head] (scalar)
///   B:   [d_state, n_group, n_seq_tokens, n_seqs]
///   C:   [d_state, n_group, n_seq_tokens, n_seqs]
///   ids: [n_seqs]  — sequence-slot mapping
///   y:   output attention [dim, n_head, n_seq_tokens, n_seqs]
#[allow(clippy::too_many_arguments)]
pub fn ssm_scan_f32(
    s0_data: &[f32], s_data: &mut [f32], y_data: &mut [f32],
    x_data:  &[f32], dt_data: &[f32],  a_data: &[f32],
    b_data:  &[f32], c_data:  &[f32],  ids:    &[i32],
    nc: i64, nr: i64, nh: i64, ng: i64, nt: i64, ns: i64,
    k_param: i64, a_is_scalar: bool,
    ith: usize, nth: usize,
) {
    let dh = ((nh as usize) + nth - 1) / nth;
    let ih0 = (dh * ith) as i64;
    let ih1 = ih0 + dh as i64;
    let ih1 = ih1.min(nh);

    let s_off = (x_data.len()) as i64; // offset in floats into s_data
    let nh = nh as usize; let nr = nr as usize; let nc = nc as usize;
    let ng = ng as usize; let nt = nt as usize; let ns = ns as usize;
    let k_param = k_param as usize;
    let ih0 = ih0 as usize; let ih1 = ih1 as usize;

    // stride in floats: nc * nr * nh per sequence slot
    let seq_stride = nc * nr * nh;

    for i3 in 0..ns {
        let s0_seq_base = ids[i3] as usize * seq_stride;
        let s_seq_base  = i3 * seq_stride + s_off as usize;

        let s0_seq = &s0_data[s0_seq_base..s0_seq_base + seq_stride];
        // copy s0 → s for this sequence slot (will be updated in-place below)
        s_data[s_seq_base..s_seq_base + seq_stride].copy_from_slice(s0_seq);

        for i2 in 0..nt {
            // pointers into x, dt, B, C for token i2, seq i3
            let x_base  = i2 * nr * nh + i3 * nt * nr * nh;
            let dt_base = i2 * nh + i3 * nt * nh;
            let b_base  = i2 * nc * ng + i3 * nt * nc * ng;
            let c_base  = i2 * nc * ng + i3 * nt * nc * ng;
            let y_base  = i2 * nh * nr + i3 * nt * nh * nr;

            if a_is_scalar {
                for h in ih0..ih1 {
                    let dt_soft_plus = softplus(dt_data[dt_base + h]);
                    let da = (dt_soft_plus * a_data[h]).exp();
                    let g = h / (nh / ng);
                    for i1 in 0..nr {
                        let ii = i1 + h * nr;
                        let x_dt = x_data[x_base + ii] * dt_soft_plus;
                        let mut sumf = 0.0f32;
                        for i0 in 0..nc {
                            let si = s_seq_base + i0 + ii * nc;
                            let ig = i0 + g * nc;
                            let state = s_data[si] * da + b_data[b_base + ig] * x_dt;
                            sumf += state * c_data[c_base + ig];
                            s_data[si] = state;
                        }
                        y_data[y_base + ii] = sumf;
                    }
                }
            } else {
                for h in ih0..ih1 {
                    let dt_soft_plus = softplus(dt_data[dt_base + h]);
                    let g = h / (nh / ng);
                    for i1 in 0..nr {
                        let ii = i1 + h * nr;
                        let x_dt = x_data[x_base + ii] * dt_soft_plus;
                        let mut sumf = 0.0f32;
                        for i0 in 0..nc {
                            let si = s_seq_base + i0 + ii * nc;
                            let ig = i0 + g * nc;
                            let state = (s_data[si] * (dt_soft_plus * a_data[i0 + h * nc]).exp())
                                + b_data[b_base + ig] * x_dt;
                            sumf += state * c_data[c_base + ig];
                            s_data[si] = state;
                        }
                        y_data[y_base + ii] = sumf;
                    }
                }
            }

            // snapshot slots (K > 1)
            if k_param > 1 {
                let slot = nt - 1 - i2;
                if slot > 0 && slot < k_param {
                    let snap_base = (slot * ns + i3) * seq_stride + s_off as usize;
                    for h in ih0..ih1 {
                        let h_off = h * nr * nc;
                        let h_bytes = nr * nc;
                        for b in 0..h_bytes {
                            s_data[snap_base + h_off + b] = s_data[s_seq_base + h_off + b];
                        }
                    }
                }
            }
        }
    }
}

// ── win_part_f32 / win_unpart_f32 (ops.cpp:10013, 10076) ─────────────────────

/// Translation of ggml_compute_forward_win_part_f32.
/// Partitions src0 into windows and zero-pads out-of-bounds.
///   ne00, ne01, ne02: src0 shape (d_embd, width, height)
///   ne0, ne1, ne2, ne3: dst shape (d_embd, w, w, n_windows)
pub fn win_part_f32(
    src0: &[f32], dst: &mut [f32],
    ne00: i64, ne01: i64, ne02: i64,
    ne0: i64, ne1: i64, ne2: i64, ne3: i64,
    nep0: i32, nep1: i32, w: i32,
) {
    assert_eq!(ne00, ne0);
    assert_eq!(ne3, (nep0 * nep1) as i64);
    for py in 0..nep1 {
        for px in 0..nep0 {
            let i3 = (py * nep0 + px) as i64;
            for i2 in 0..ne2 {
                for i1 in 0..ne1 {
                    for i0 in 0..ne0 {
                        let i02 = py as i64 * w as i64 + i2;
                        let i01 = px as i64 * w as i64 + i1;
                        let dst_idx = (i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0 + i0) as usize;
                        if i02 >= ne02 || i01 >= ne01 {
                            dst[dst_idx] = 0.0;
                        } else {
                            let src_idx = (i02 * ne01 * ne00 + i01 * ne00 + i0) as usize;
                            dst[dst_idx] = src0[src_idx];
                        }
                    }
                }
            }
        }
    }
}

/// Translation of ggml_compute_forward_win_unpart_f32.
pub fn win_unpart_f32(
    src0: &[f32], dst: &mut [f32],
    ne00: i64,
    ne0: i64, ne1: i64, ne2: i64,
    ne01: i64, ne02: i64,
    w: i32,
) {
    assert_eq!(ne0, ne00);
    let px = ((w as i64 - (ne1 % w as i64)) % w as i64);
    let npx = (px + ne1) / w as i64;
    for i2 in 0..ne2 {
        for i1 in 0..ne1 {
            for i0 in 0..ne0 {
                let ip2 = i2 / w as i64;
                let ip1 = i1 / w as i64;
                let i02 = i2 % w as i64;
                let i01 = i1 % w as i64;
                let src_idx = ((ip2 * npx + ip1) * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00 + i0) as usize;
                let dst_idx = (i2 * ne1 * ne0 + i1 * ne0 + i0) as usize;
                dst[dst_idx] = src0[src_idx];
            }
        }
    }
}

// ── rwkv_wkv6_f32 (ops.cpp:10413) ────────────────────────────────────────────

/// Translation of ggml_compute_forward_rwkv_wkv6_f32 (scalar path, #else branch).
/// dst_data: [C * T] output + [head_size * C * n_seqs] state trailing
/// k, v, r: [T * C], time_faaaa: [C], time_decay: [T * C]
/// state_init: initial state from src[5]
#[allow(clippy::too_many_arguments)]
pub fn rwkv_wkv6_f32(
    k: &[f32], v: &[f32], r: &[f32],
    time_faaaa: &[f32], time_decay: &[f32],
    state_init: &[f32],
    dst_data: &mut [f32],
    // state is written after the dst_data block
    t_val: usize, c_val: usize, heads: usize, n_seqs: usize,
    ith: usize, nth: usize,
) {
    let head_size = c_val / heads;
    let t_stride = heads * head_size; // == c_val
    let h_stride = c_val / heads;
    let h_stride_2d = head_size * head_size;

    let h_start = heads * ith / nth;
    let h_end = ((heads * (ith + 1)) / nth).min(heads);

    // zero dst_data for token outputs
    if ith == 0 {
        for x in dst_data[..t_val * c_val].iter_mut() { *x = 0.0; }
    }

    let state_base = t_val * c_val;

    for t in 0..t_val {
        let t_offset = t * t_stride;
        let state_offset = head_size * c_val * (t / (t_val / n_seqs));
        let is_first = t % (t_val / n_seqs) == 0;

        for h in h_start..h_end {
            let h_offset = h * h_stride;
            let t_h_offset = t_offset + h_offset;
            let h_2d_offset = h * h_stride_2d;

            for i in 0..head_size {
                let t_h_i_offset = t_h_offset + i;
                let h_i_offset = h_offset + i;
                let h_2d_i_offset = h_2d_offset + i * h_stride;

                let k_val_s = k[t_h_i_offset];
                let r_val_s = r[t_h_i_offset];
                let time_faaaa_val = time_faaaa[h_i_offset];
                let time_decay_val = time_decay[t_h_i_offset];

                for j in 0..head_size {
                    let t_h_j_offset = t_h_offset + j;
                    let h_2d_i_j_offset = h_2d_i_offset + j;

                    let v_val_s = v[t_h_j_offset];
                    let kv_val = v_val_s * k_val_s;
                    let prev_state_val = if is_first {
                        state_init[state_offset + h_2d_i_j_offset]
                    } else {
                        dst_data[state_base + state_offset + h_2d_i_j_offset]
                    };
                    let temp_val = kv_val * time_faaaa_val + prev_state_val;
                    dst_data[t_h_j_offset] += temp_val * r_val_s;
                    dst_data[state_base + state_offset + h_2d_i_j_offset] =
                        prev_state_val * time_decay_val + kv_val;
                }
            }
        }
    }
}

// ── rwkv_wkv7_f32 (ops.cpp:11422) ────────────────────────────────────────────

/// Translation of ggml_compute_forward_rwkv_wkv7_f32 (scalar path).
#[allow(clippy::too_many_arguments)]
pub fn rwkv_wkv7_f32(
    r: &[f32], w: &[f32], k: &[f32], v: &[f32], a: &[f32], b: &[f32],
    state_init: &[f32],
    dst_data: &mut [f32],
    t_val: usize, c_val: usize, heads: usize, n_seqs: usize,
    ith: usize, nth: usize,
) {
    let head_size = c_val / heads;
    let t_stride = heads * head_size;
    let h_stride = c_val / heads;
    let h_stride_2d = head_size * head_size;

    let h_start = heads * ith / nth;
    let h_end = ((heads * (ith + 1)) / nth).min(heads);

    let state_base = t_val * c_val;

    for t in 0..t_val {
        let t_offset = t * t_stride;
        let state_offset = head_size * c_val * (t / (t_val / n_seqs));
        let is_first = t % (t_val / n_seqs) == 0;

        for h in h_start..h_end {
            let h_offset = h * h_stride;
            let t_h_offset = t_offset + h_offset;
            let h_2d_offset = h * h_stride_2d;

            for ii in 0..head_size {
                let t_h_i_offset = t_h_offset + ii;
                let h_2d_i_offset = h_2d_offset + ii * h_stride;

                let v_val_s = v[t_h_i_offset];

                // sa = dot(a[t_h_offset..], state_prev[h_2d_i_offset..])
                let mut sa = 0.0f32;
                for j in 0..head_size {
                    let h_2d_i_j_offset = h_2d_i_offset + j;
                    let prev = if is_first {
                        state_init[state_offset + h_2d_i_j_offset]
                    } else {
                        dst_data[state_base + state_offset + h_2d_i_j_offset]
                    };
                    sa += a[t_h_offset + j] * prev;
                }

                let mut result = 0.0f32;
                for j in 0..head_size {
                    let t_h_j_offset = t_h_offset + j;
                    let h_2d_i_j_offset = h_2d_i_offset + j;
                    let prev = if is_first {
                        state_init[state_offset + h_2d_i_j_offset]
                    } else {
                        dst_data[state_base + state_offset + h_2d_i_j_offset]
                    };
                    let new_state = prev * w[t_h_j_offset] + v_val_s * k[t_h_j_offset] + sa * b[t_h_j_offset];
                    dst_data[state_base + state_offset + h_2d_i_j_offset] = new_state;
                    result += new_state * r[t_h_j_offset];
                }
                dst_data[t_h_i_offset] = result;
            }
        }
    }
}

// ── gla_f32 (ops.cpp:10625) ───────────────────────────────────────────────────

/// Translation of ggml_compute_forward_gla_f32 (scalar #else path).
#[allow(clippy::too_many_arguments)]
pub fn gla_f32(
    k: &[f32], v: &[f32], q: &[f32], g: &[f32],
    state_init: &[f32],
    dst_data: &mut [f32],
    scale: f32,
    t_val: usize, c_val: usize, heads: usize, n_seqs: usize,
    ith: usize, nth: usize,
) {
    let head_size = c_val / heads;
    let t_stride = heads * head_size;
    let h_stride = c_val / heads;
    let h_stride_2d = head_size * head_size;

    let h_start = heads * ith / nth;
    let h_end = ((heads * (ith + 1)) / nth).min(heads);

    if ith == 0 {
        for x in dst_data[..t_val * c_val].iter_mut() { *x = 0.0; }
    }

    let state_base = t_val * c_val;

    for t in 0..t_val {
        let t_offset = t * t_stride;
        let state_offset = head_size * c_val * (t / (t_val / n_seqs));
        let is_first = t % (t_val / n_seqs) == 0;

        for h in h_start..h_end {
            let h_offset = h * h_stride;
            let t_h_offset = t_offset + h_offset;
            let h_2d_offset = h * h_stride_2d;

            for i in 0..head_size {
                let t_h_i_offset = t_h_offset + i;
                let h_2d_i_offset = h_2d_offset + i * h_stride;

                let k_val_s = k[t_h_i_offset];
                let q_val_s = q[t_h_i_offset] * scale;
                let g_val_s = g[t_h_i_offset];

                for j in 0..head_size {
                    let t_h_j_offset = t_h_offset + j;
                    let h_2d_i_j_offset = h_2d_i_offset + j;

                    let v_val_s = v[t_h_j_offset];
                    let kv_val = v_val_s * k_val_s;
                    let prev = if is_first {
                        state_init[state_offset + h_2d_i_j_offset]
                    } else {
                        dst_data[state_base + state_offset + h_2d_i_j_offset]
                    };
                    let temp_val = prev * g_val_s + kv_val;
                    dst_data[t_h_j_offset] += temp_val * q_val_s;
                    dst_data[state_base + state_offset + h_2d_i_j_offset] = temp_val;
                }
            }
        }
    }
}

// ── gated_delta_net (ops.cpp:10895) ──────────────────────────────────────────

/// Translation of ggml_compute_forward_gated_delta_net_one_chunk.
/// All inputs are contiguous flat slices. Index math follows the C source.
/// `delta` and `state_work` are scratch buffers (per-thread).
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_net_one_chunk(
    q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], state_in: &[f32],
    attn_out: &mut [f32], state_out: &mut [f32],
    s_v: usize, h: usize, n_tokens: usize, n_seqs: usize,
    k_param: usize, kda: bool, // kda = g.len() == S_v per token
    ir0: usize, ir1: usize,
    scale: f32,
) {
    let h_sv = s_v; // dimension per head

    let state_size_per_head = h_sv * h_sv;
    let attn_score_elems = h_sv * h * n_tokens * n_seqs;

    let mut delta = vec![0.0f32; h_sv];
    let mut state_work = vec![0.0f32; h_sv * h_sv];

    for ir in ir0..ir1 {
        let iv1 = ir % h; // head index
        let iv3 = ir / h; // sequence

        // always work in state_work; copy to state_out at end / per slot
        let s_in_base = iv3 * h * state_size_per_head + iv1 * state_size_per_head;
        state_work.copy_from_slice(&state_in[s_in_base..s_in_base + state_size_per_head]);
        let s_out_slice = &mut state_work[..];

        for t in 0..n_tokens {
            let q_base   = iv3 * n_tokens * h * h_sv + t * h * h_sv + iv1 * h_sv;
            let k_base   = iv3 * n_tokens * h * h_sv + t * h * h_sv + iv1 * h_sv;
            let v_base   = iv3 * n_tokens * h * h_sv + t * h * h_sv + iv1 * h_sv;
            let beta_val = beta[iv3 * n_tokens * h + t * h + iv1];

            if kda {
                // exp(g) element-wise into delta
                let g_base = iv3 * n_tokens * h * h_sv + t * h * h_sv + iv1 * h_sv;
                for i in 0..h_sv { delta[i] = g[g_base + i].exp(); }
                // S[i][:] *= exp(g[i]) — for each col j of M: M[j][i] *= delta[i]
                for j in 0..h_sv {
                    for i in 0..h_sv {
                        s_out_slice[j * h_sv + i] *= delta[i];
                    }
                }
            } else {
                let g_base = iv3 * n_tokens * h + t * h + iv1;
                let eg = g[g_base].exp();
                for x in s_out_slice.iter_mut() { *x *= eg; }
            }

            // delta[j] = dot(row j of M, k) — then delta[j] = (v[j] - delta[j]) * beta
            for j in 0..h_sv {
                let mut sum = 0.0f32;
                for i in 0..h_sv { sum += s_out_slice[j * h_sv + i] * k[k_base + i]; }
                delta[j] = (v[v_base + j] - sum) * beta_val;
            }

            // outer product: M[j][i] += delta[j] * k[i]
            for j in 0..h_sv {
                for i in 0..h_sv {
                    s_out_slice[j * h_sv + i] += k[k_base + i] * delta[j];
                }
            }

            // attn_out[j] = dot(row j of M, q) * scale
            let attn_base = attn_score_elems + (iv3 * n_tokens * h + t * h + iv1) * h_sv;
            let attn_base = attn_base - attn_score_elems; // relative to attn_out start
            let _ = attn_base;
            let ao_base = (iv3 * n_tokens * h + t * h + iv1) * h_sv;
            for j in 0..h_sv {
                let mut sum = 0.0f32;
                for i in 0..h_sv { sum += s_out_slice[j * h_sv + i] * q[q_base + i]; }
                attn_out[ao_base + j] = sum * scale;
            }

            // snapshot slots
            if k_param > 1 {
                let target_slot = n_tokens - 1 - t;
                if target_slot < k_param {
                    let snap_base = target_slot * h_sv * h_sv * h * n_seqs
                        + (iv3 * h + iv1) * state_size_per_head;
                    state_out[snap_base..snap_base + state_size_per_head]
                        .copy_from_slice(s_out_slice);
                }
            }
        }
        // write final state to slot 0 (k_param==1) or already written per slot above
        if k_param == 1 {
            let dst_base = (iv3 * h + iv1) * state_size_per_head;
            state_out[dst_base..dst_base + state_size_per_head].copy_from_slice(s_out_slice);
        }
    }
}

// ── dsv4_hc_comb helpers (ops.cpp:11110) ─────────────────────────────────────

fn dsv4_hc_comb_norm_cols(comb: &mut [f32], eps: f32) {
    const HC: usize = 4;
    for idst in 0..HC {
        let mut sum = eps;
        for isrc in 0..HC { sum += comb[idst + HC * isrc]; }
        let inv = 1.0 / sum;
        for isrc in 0..HC { comb[idst + HC * isrc] *= inv; }
    }
}

fn dsv4_hc_comb_norm_rows(comb: &mut [f32], eps: f32) {
    const HC: usize = 4;
    for isrc in 0..HC {
        let mut sum = eps;
        for idst in 0..HC { sum += comb[idst + HC * isrc]; }
        let inv = 1.0 / sum;
        for idst in 0..HC { comb[idst + HC * isrc] *= inv; }
    }
}

// ── dsv4_hc_comb_f32 (ops.cpp:11142) ─────────────────────────────────────────

/// Translation of ggml_compute_forward_dsv4_hc_comb_f32.
/// mixes: [hc_mix_dim, n_tokens], scale: [>=3], base: [hc_mix_dim]
/// dst: [hc, hc, n_tokens]
#[allow(clippy::too_many_arguments)]
pub fn dsv4_hc_comb_f32(
    mixes: &[f32], scale: &[f32], base: &[f32], dst: &mut [f32],
    n_tokens: usize, eps: f32, n_iter: i32,
    ith: usize, nth: usize,
) {
    const HC: usize = 4;
    const COMB_OFFSET: usize = 2 * HC;

    let dr = (n_tokens + nth - 1) / nth;
    let it0 = dr * ith;
    let it1 = (it0 + dr).min(n_tokens);

    let scale_comb = scale[2];
    let hc_mix_dim = (2 + HC) * HC;

    for it in it0..it1 {
        let mut comb = [0.0f32; HC * HC];
        for isrc in 0..HC {
            let mut mx = f32::NEG_INFINITY;
            for idst in 0..HC {
                let idx = idst + HC * isrc;
                let xv = mixes[(COMB_OFFSET + idx) + it * hc_mix_dim];
                let bv = base[COMB_OFFSET + idx];
                let v = xv * scale_comb + bv;
                comb[idx] = v;
                if v > mx { mx = v; }
            }
            let mut sum = 0.0f32;
            for idst in 0..HC {
                let idx = idst + HC * isrc;
                let v = (comb[idx] - mx).exp();
                comb[idx] = v;
                sum += v;
            }
            let inv = 1.0 / sum;
            for idst in 0..HC {
                let idx = idst + HC * isrc;
                comb[idx] = comb[idx] * inv + eps;
            }
        }

        dsv4_hc_comb_norm_cols(&mut comb, eps);
        for _ in 1..n_iter {
            dsv4_hc_comb_norm_rows(&mut comb, eps);
            dsv4_hc_comb_norm_cols(&mut comb, eps);
        }

        for isrc in 0..HC {
            for idst in 0..HC {
                let idx = idst + HC * isrc;
                dst[idst + isrc * HC + it * HC * HC] = comb[idx];
            }
        }
    }
}

// ── dsv4_hc_pre_f32 (ops.cpp:11248) ──────────────────────────────────────────

/// Translation of ggml_compute_forward_dsv4_hc_pre_f32.
/// x: [n_embd, hc, n_tokens], weights: [n_embd, hc, n_tokens] (gated) or [hc, n_tokens]
/// dst: [n_embd, n_tokens]
#[allow(clippy::too_many_arguments)]
pub fn dsv4_hc_pre_f32(
    x: &[f32], weights: &[f32], dst: &mut [f32],
    n_embd: usize, hc: usize, n_tokens: usize,
    scale: f32, gated: bool,
    ith: usize, nth: usize,
) {
    let nr = n_embd * n_tokens;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);

    for ir in ir0..ir1 {
        let i0 = ir % n_embd;
        let it = ir / n_embd;

        let mut sum = 0.0f32;
        for ih in 0..hc {
            let xv = x[i0 + ih * n_embd + it * n_embd * hc];
            let wv = if gated {
                let gv = weights[i0 + ih * n_embd + it * n_embd * hc];
                1.0 / (1.0 + (-gv).exp())
            } else {
                weights[ih + it * hc]
            };
            sum += xv * wv;
        }
        dst[i0 + it * n_embd] = scale * sum;
    }
}

// ── dsv4_hc_post_f32 (ops.cpp:11328) ─────────────────────────────────────────

/// Translation of ggml_compute_forward_dsv4_hc_post_f32.
/// x: [n_embd, n_tokens], residual: [n_embd, hc, n_tokens]
/// post: [hc, n_tokens], comb: [hc, hc, n_tokens] or None
/// dst: [n_embd, hc, n_tokens]
#[allow(clippy::too_many_arguments)]
pub fn dsv4_hc_post_f32(
    x: &[f32], residual: &[f32], post: &[f32], comb: Option<&[f32]>,
    dst: &mut [f32],
    n_embd: usize, hc: usize, n_tokens: usize,
    ith: usize, nth: usize,
) {
    let nr = n_embd * hc * n_tokens;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);

    for ir in ir0..ir1 {
        let i0   = ir % n_embd;
        let idst = (ir / n_embd) % hc;
        let it   = ir / (n_embd * hc);

        let xv = x[i0 + it * n_embd];
        let pv = post[idst + it * hc];
        let mut sum = xv * pv;

        if let Some(comb) = comb {
            for isrc in 0..hc {
                let rv = residual[i0 + isrc * n_embd + it * n_embd * hc];
                let cv = comb[idst + isrc * hc + it * hc * hc];
                sum += rv * cv;
            }
        } else {
            sum += residual[i0 + idst * n_embd + it * n_embd * hc];
        }

        dst[i0 + idst * n_embd + it * n_embd * hc] = sum;
    }
}

// ── fwht_f32 (ops.cpp:12027) ──────────────────────────────────────────────────

/// Translation of ggml_compute_forward_fwht_impl (f32 specialization, scalar path).
/// src: flat contiguous [ne13 * ne12 * ne11 * n], dst: same shape
pub fn fwht_f32(
    src: &[f32], dst: &mut [f32],
    n: usize, // must be power of 2
    nr: usize, // ne11 * ne12 * ne13
    ith: usize, nth: usize,
) {
    assert!(n.is_power_of_two());
    let scale = 1.0 / (n as f32).sqrt();
    let rows_per_thread = (nr + nth - 1) / nth;
    let start_row = ith * rows_per_thread;
    let end_row = (start_row + rows_per_thread).min(nr);

    for r in start_row..end_row {
        let row_base = r * n;
        // load and scale
        for j in 0..n {
            dst[row_base + j] = src[row_base + j] * scale;
        }
        // butterfly passes
        let mut len = 1;
        while len < n {
            let mut i = 0;
            while i < n {
                for j in 0..len {
                    let u = dst[row_base + i + j];
                    let v = dst[row_base + i + len + j];
                    dst[row_base + i + j]       = u + v;
                    dst[row_base + i + len + j] = u - v;
                }
                i += 2 * len;
            }
            len <<= 1;
        }
    }
}
