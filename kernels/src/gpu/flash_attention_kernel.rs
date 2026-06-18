use cuda_device::{kernel, ptx_asm, thread, warp, DisjointSlice, SharedArray};
use cuda_host::cuda_module;

const BR: usize = 64;
const BC: usize = 64;
const HD: usize = 128;
const MMA_K: usize = 16;

const SMEM_QKV: usize = BR * HD; // 8192 floats per tile

#[inline(always)]
fn pack_f16x2(lo: f32, hi: f32) -> u32 {
    let result: u32;
    unsafe {
        ptx_asm!(
            "cvt.rn.f16x2.f32 %0, %2, %1;",
            out("=r") result,
            in("f") lo,
            in("f") hi,
        );
    }
    result
}

#[inline(always)]
fn exp_f32(x: f32) -> f32 {
    let result: f32;
    unsafe {
        ptx_asm!(
            "ex2.approx.ftz.f32 %0, %1;",
            out("=f") result,
            in("f") x * 1.442695041, // log2(e)
        );
    }
    result
}

#[inline(always)]
fn max_f32(a: f32, b: f32) -> f32 {
    let result: f32;
    unsafe {
        ptx_asm!(
            "max.f32 %0, %1, %2;",
            out("=f") result,
            in("f") a,
            in("f") b,
        );
    }
    result
}

#[inline(always)]
fn shuffle_xor_max(val: f32, delta: u32) -> f32 {
    max_f32(val, warp::shuffle_xor_f32(val, delta))
}

#[inline(always)]
fn shuffle_xor_sum(val: f32, delta: u32) -> f32 {
    val + warp::shuffle_xor_f32(val, delta)
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Flash Attention v2 with MMA tensor cores (m16n8k16).
    ///
    /// Grid:  (num_heads, ceil(seq_len/BR), batch)
    /// Block: (128, 1, 1) — 4 warps
    /// SMEM:  98304 bytes (3 × 64×128 f32 tiles)
    ///
    /// Each CTA processes one (batch, head, q_tile) group.
    /// Warp w owns Q rows [w*16, w*16+16) throughout the kernel.
    #[kernel]
    pub fn flash_attention_forward(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut output: DisjointSlice<f32>,
        _batch: u32,
        seq_len: u32,
        kv_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
    ) {
        static mut Q_SMEM: SharedArray<f32, SMEM_QKV> = SharedArray::UNINIT;
        static mut K_SMEM: SharedArray<f32, SMEM_QKV> = SharedArray::UNINIT;
        static mut V_SMEM: SharedArray<f32, SMEM_QKV> = SharedArray::UNINIT;

        let tid_x = thread::threadIdx_x() as usize;
        let head_idx = thread::blockIdx_x() as usize;
        let q_tile_idx = thread::blockIdx_y() as usize;
        let batch_idx = thread::blockIdx_z() as usize;

        let warp_id = tid_x / 32;
        let lane = tid_x % 32;
        let group_id = lane >> 2;
        let tid = lane & 3;

        let sl = seq_len as usize;
        let kvl = kv_len as usize;
        let nh = num_heads as usize;
        let nkv = num_kv_heads as usize;
        let hd = head_dim as usize;
        let heads_per_group = nh / nkv;
        let kv_head_idx = head_idx / heads_per_group;

        let q_row_start = q_tile_idx * BR;
        let warp_row_base = warp_id * 16;

        // --- Phase 1: Load Q tile into SMEM ---
        let mut li = 0usize;
        while li < 64 {
            let elem = tid_x * 64 + li;
            let row = elem / HD;
            let col = elem % HD;
            let global_row = q_row_start + row;
            unsafe {
                if global_row < sl && col < hd {
                    let q_idx = ((batch_idx * sl + global_row) * nh + head_idx) * hd + col;
                    Q_SMEM[elem] = q[q_idx];
                } else {
                    Q_SMEM[elem] = 0.0;
                }
            }
            li += 1;
        }
        thread::sync_threads();

        // --- Phase 2: Init online softmax state ---
        // Output accumulator: flat [f32; 64] = 16 col-groups × 4 MMA elements.
        // o_acc[ci*4+j] corresponds to MMA output for col-group ci, element j.
        let mut o_acc = [0.0f32; 64];
        let mut row_max_0 = f32::NEG_INFINITY;
        let mut row_max_1 = f32::NEG_INFINITY;
        let mut row_sum_0 = 0.0f32;
        let mut row_sum_1 = 0.0f32;

        // --- Phase 3: Loop over KV tiles ---
        let cache_offset = kvl - sl;
        let last_q_row = q_row_start + BR - 1;
        let causal_kv_max = if last_q_row < sl {
            cache_offset + last_q_row
        } else {
            cache_offset + sl - 1
        };
        let num_kv_tiles = kvl.div_ceil(BC);
        let max_kv_tile = if num_kv_tiles < (causal_kv_max + BC) / BC {
            num_kv_tiles
        } else {
            (causal_kv_max + BC) / BC
        };

        let mut kv_tile = 0usize;
        while kv_tile < max_kv_tile {
            let kv_col_start = kv_tile * BC;

            // --- 3a: Load K tile into K_SMEM ---
            li = 0;
            while li < 64 {
                let elem = tid_x * 64 + li;
                let row = elem / HD;
                let col = elem % HD;
                let global_kv_pos = kv_col_start + row;
                unsafe {
                    if global_kv_pos < kvl && col < hd {
                        let k_idx =
                            ((batch_idx * kvl + global_kv_pos) * nkv + kv_head_idx) * hd + col;
                        K_SMEM[elem] = k[k_idx];
                    } else {
                        K_SMEM[elem] = 0.0;
                    }
                }
                li += 1;
            }
            thread::sync_threads();

            // --- 3b: Compute S = Q · K^T via MMA ---
            // Score accumulators: flat [f32; 32] = 8 col-groups × 4.
            let mut s_acc = [0.0f32; 32];

            let mut kt = 0usize;
            while kt < HD / MMA_K {
                let k_offset = kt * MMA_K;

                let at_r0 = warp_row_base + group_id;
                let at_r1 = warp_row_base + group_id + 8;

                let a_frag: [u32; 4];
                unsafe {
                    a_frag = [
                        pack_f16x2(
                            Q_SMEM[at_r0 * HD + k_offset + tid * 2 + 1],
                            Q_SMEM[at_r0 * HD + k_offset + tid * 2],
                        ),
                        pack_f16x2(
                            Q_SMEM[at_r1 * HD + k_offset + tid * 2 + 1],
                            Q_SMEM[at_r1 * HD + k_offset + tid * 2],
                        ),
                        pack_f16x2(
                            Q_SMEM[at_r0 * HD + k_offset + tid * 2 + 9],
                            Q_SMEM[at_r0 * HD + k_offset + tid * 2 + 8],
                        ),
                        pack_f16x2(
                            Q_SMEM[at_r1 * HD + k_offset + tid * 2 + 9],
                            Q_SMEM[at_r1 * HD + k_offset + tid * 2 + 8],
                        ),
                    ];
                }

                let mut ni = 0usize;
                while ni < 8 {
                    let kn = ni * 8 + group_id;
                    let b_frag: [u32; 2];
                    unsafe {
                        b_frag = [
                            pack_f16x2(
                                K_SMEM[kn * HD + k_offset + tid * 2 + 1],
                                K_SMEM[kn * HD + k_offset + tid * 2],
                            ),
                            pack_f16x2(
                                K_SMEM[kn * HD + k_offset + tid * 2 + 9],
                                K_SMEM[kn * HD + k_offset + tid * 2 + 8],
                            ),
                        ];
                    }

                    // MMA into s_acc[ni*4..ni*4+4] — use temp to avoid 2-level projection
                    let mut tmp = [
                        s_acc[ni * 4],
                        s_acc[ni * 4 + 1],
                        s_acc[ni * 4 + 2],
                        s_acc[ni * 4 + 3],
                    ];
                    unsafe {
                        cuda_device::mma_sync::mma_sync_m16n8k16_f32_f16(
                            &mut tmp, &a_frag, &b_frag,
                        );
                    }
                    s_acc[ni * 4] = tmp[0];
                    s_acc[ni * 4 + 1] = tmp[1];
                    s_acc[ni * 4 + 2] = tmp[2];
                    s_acc[ni * 4 + 3] = tmp[3];

                    ni += 1;
                }

                kt += 1;
            }

            // --- 3c: Scale S and apply causal mask ---
            let global_q_row_0 = q_row_start + warp_row_base + group_id;
            let global_q_row_1 = q_row_start + warp_row_base + group_id + 8;
            let limit_0 = if global_q_row_0 < sl {
                cache_offset + global_q_row_0
            } else {
                0
            };
            let limit_1 = if global_q_row_1 < sl {
                cache_offset + global_q_row_1
            } else {
                0
            };

            let mut ni = 0usize;
            while ni < 8 {
                let base = ni * 4;
                s_acc[base] *= scale;
                s_acc[base + 1] *= scale;
                s_acc[base + 2] *= scale;
                s_acc[base + 3] *= scale;

                let global_kv_col_0 = kv_col_start + ni * 8 + tid * 2;
                let global_kv_col_1 = kv_col_start + ni * 8 + tid * 2 + 1;

                if global_q_row_0 >= sl || global_kv_col_0 > limit_0 || global_kv_col_0 >= kvl {
                    s_acc[base] = f32::NEG_INFINITY;
                }
                if global_q_row_0 >= sl || global_kv_col_1 > limit_0 || global_kv_col_1 >= kvl {
                    s_acc[base + 1] = f32::NEG_INFINITY;
                }
                if global_q_row_1 >= sl || global_kv_col_0 > limit_1 || global_kv_col_0 >= kvl {
                    s_acc[base + 2] = f32::NEG_INFINITY;
                }
                if global_q_row_1 >= sl || global_kv_col_1 > limit_1 || global_kv_col_1 >= kvl {
                    s_acc[base + 3] = f32::NEG_INFINITY;
                }

                ni += 1;
            }

            // --- 3d: Online softmax update ---
            // Step 1: Row-wise max
            let mut local_max_0 = f32::NEG_INFINITY;
            let mut local_max_1 = f32::NEG_INFINITY;
            ni = 0;
            while ni < 8 {
                let base = ni * 4;
                local_max_0 = max_f32(local_max_0, max_f32(s_acc[base], s_acc[base + 1]));
                local_max_1 = max_f32(local_max_1, max_f32(s_acc[base + 2], s_acc[base + 3]));
                ni += 1;
            }

            local_max_0 = shuffle_xor_max(local_max_0, 1);
            local_max_0 = shuffle_xor_max(local_max_0, 2);
            local_max_1 = shuffle_xor_max(local_max_1, 1);
            local_max_1 = shuffle_xor_max(local_max_1, 2);

            // Step 2: Correction factor
            let new_max_0 = max_f32(row_max_0, local_max_0);
            let new_max_1 = max_f32(row_max_1, local_max_1);

            let correction_0 = if row_max_0 > f32::NEG_INFINITY {
                exp_f32(row_max_0 - new_max_0)
            } else {
                0.0
            };
            let correction_1 = if row_max_1 > f32::NEG_INFINITY {
                exp_f32(row_max_1 - new_max_1)
            } else {
                0.0
            };

            // Rescale output accumulators
            let mut ci = 0usize;
            while ci < 16 {
                let base = ci * 4;
                o_acc[base] *= correction_0;
                o_acc[base + 1] *= correction_0;
                o_acc[base + 2] *= correction_1;
                o_acc[base + 3] *= correction_1;
                ci += 1;
            }

            row_sum_0 *= correction_0;
            row_sum_1 *= correction_1;

            // Step 3: Compute P = exp(S - new_max), accumulate row_sum
            let mut local_sum_0 = 0.0f32;
            let mut local_sum_1 = 0.0f32;
            ni = 0;
            while ni < 8 {
                let base = ni * 4;
                s_acc[base] = if s_acc[base] > f32::NEG_INFINITY {
                    exp_f32(s_acc[base] - new_max_0)
                } else {
                    0.0
                };
                s_acc[base + 1] = if s_acc[base + 1] > f32::NEG_INFINITY {
                    exp_f32(s_acc[base + 1] - new_max_0)
                } else {
                    0.0
                };
                s_acc[base + 2] = if s_acc[base + 2] > f32::NEG_INFINITY {
                    exp_f32(s_acc[base + 2] - new_max_1)
                } else {
                    0.0
                };
                s_acc[base + 3] = if s_acc[base + 3] > f32::NEG_INFINITY {
                    exp_f32(s_acc[base + 3] - new_max_1)
                } else {
                    0.0
                };

                local_sum_0 += s_acc[base] + s_acc[base + 1];
                local_sum_1 += s_acc[base + 2] + s_acc[base + 3];

                ni += 1;
            }

            local_sum_0 = shuffle_xor_sum(local_sum_0, 1);
            local_sum_0 = shuffle_xor_sum(local_sum_0, 2);
            local_sum_1 = shuffle_xor_sum(local_sum_1, 1);
            local_sum_1 = shuffle_xor_sum(local_sum_1, 2);

            row_sum_0 += local_sum_0;
            row_sum_1 += local_sum_1;

            row_max_0 = new_max_0;
            row_max_1 = new_max_1;

            // --- 3e: Write P to K_SMEM (reuse as P_SMEM) ---
            ni = 0;
            while ni < 8 {
                let base = ni * 4;
                let pr0 = warp_row_base + group_id;
                let pr1 = warp_row_base + group_id + 8;
                let pc0 = ni * 8 + tid * 2;
                let pc1 = ni * 8 + tid * 2 + 1;

                unsafe {
                    K_SMEM[pr0 * BC + pc0] = s_acc[base];
                    K_SMEM[pr0 * BC + pc1] = s_acc[base + 1];
                    K_SMEM[pr1 * BC + pc0] = s_acc[base + 2];
                    K_SMEM[pr1 * BC + pc1] = s_acc[base + 3];
                }

                ni += 1;
            }

            // --- 3f: Load V tile into V_SMEM ---
            thread::sync_threads();

            li = 0;
            while li < 64 {
                let elem = tid_x * 64 + li;
                let row = elem / HD;
                let col = elem % HD;
                let global_kv_pos = kv_col_start + row;
                unsafe {
                    if global_kv_pos < kvl && col < hd {
                        let v_idx =
                            ((batch_idx * kvl + global_kv_pos) * nkv + kv_head_idx) * hd + col;
                        V_SMEM[elem] = v[v_idx];
                    } else {
                        V_SMEM[elem] = 0.0;
                    }
                }
                li += 1;
            }
            thread::sync_threads();

            // --- 3g: Accumulate O += P · V via MMA ---
            // P is [64×64] in K_SMEM, V is [64×128] in V_SMEM.
            // Each warp handles 16 rows × 128 cols of output.
            // K-dim = BC = 64, tiled in 16s = 4 k-tiles.
            // Per k-tile: 16 col-groups × 1 MMA = 16 MMA ops.

            let mut pv_kt = 0usize;
            while pv_kt < BC / MMA_K {
                let pk_offset = pv_kt * MMA_K;

                let pa_r0 = warp_row_base + group_id;
                let pa_r1 = warp_row_base + group_id + 8;

                let p_frag: [u32; 4];
                unsafe {
                    p_frag = [
                        pack_f16x2(
                            K_SMEM[pa_r0 * BC + pk_offset + tid * 2 + 1],
                            K_SMEM[pa_r0 * BC + pk_offset + tid * 2],
                        ),
                        pack_f16x2(
                            K_SMEM[pa_r1 * BC + pk_offset + tid * 2 + 1],
                            K_SMEM[pa_r1 * BC + pk_offset + tid * 2],
                        ),
                        pack_f16x2(
                            K_SMEM[pa_r0 * BC + pk_offset + tid * 2 + 9],
                            K_SMEM[pa_r0 * BC + pk_offset + tid * 2 + 8],
                        ),
                        pack_f16x2(
                            K_SMEM[pa_r1 * BC + pk_offset + tid * 2 + 9],
                            K_SMEM[pa_r1 * BC + pk_offset + tid * 2 + 8],
                        ),
                    ];
                }

                ci = 0;
                while ci < 16 {
                    let vn = ci * 8 + group_id;
                    let v_frag: [u32; 2];
                    unsafe {
                        v_frag = [
                            pack_f16x2(
                                V_SMEM[(pk_offset + tid * 2 + 1) * HD + vn],
                                V_SMEM[(pk_offset + tid * 2) * HD + vn],
                            ),
                            pack_f16x2(
                                V_SMEM[(pk_offset + tid * 2 + 9) * HD + vn],
                                V_SMEM[(pk_offset + tid * 2 + 8) * HD + vn],
                            ),
                        ];
                    }

                    let base = ci * 4;
                    let mut tmp = [o_acc[base], o_acc[base + 1], o_acc[base + 2], o_acc[base + 3]];
                    unsafe {
                        cuda_device::mma_sync::mma_sync_m16n8k16_f32_f16(
                            &mut tmp, &p_frag, &v_frag,
                        );
                    }
                    o_acc[base] = tmp[0];
                    o_acc[base + 1] = tmp[1];
                    o_acc[base + 2] = tmp[2];
                    o_acc[base + 3] = tmp[3];

                    ci += 1;
                }

                pv_kt += 1;
            }

            thread::sync_threads();
            kv_tile += 1;
        }

        // --- Phase 4: Finalize output ---
        let inv_sum_0 = if row_sum_0 > 0.0 {
            1.0 / row_sum_0
        } else {
            0.0
        };
        let inv_sum_1 = if row_sum_1 > 0.0 {
            1.0 / row_sum_1
        } else {
            0.0
        };

        let mut ci = 0usize;
        while ci < 16 {
            let base = ci * 4;
            o_acc[base] *= inv_sum_0;
            o_acc[base + 1] *= inv_sum_0;
            o_acc[base + 2] *= inv_sum_1;
            o_acc[base + 3] *= inv_sum_1;
            ci += 1;
        }

        // --- Phase 5: Write output to global memory ---
        ci = 0;
        while ci < 16 {
            let base = ci * 4;
            let r0 = q_row_start + warp_id * 16 + group_id;
            let r1 = q_row_start + warp_id * 16 + group_id + 8;
            let c0 = ci * 8 + tid * 2;
            let c1 = ci * 8 + tid * 2 + 1;

            if r0 < sl && c0 < hd {
                let out_idx = ((batch_idx * sl + r0) * nh + head_idx) * hd + c0;
                unsafe {
                    *output.get_unchecked_mut(out_idx) = o_acc[base];
                }
            }
            if r0 < sl && c1 < hd {
                let out_idx = ((batch_idx * sl + r0) * nh + head_idx) * hd + c1;
                unsafe {
                    *output.get_unchecked_mut(out_idx) = o_acc[base + 1];
                }
            }
            if r1 < sl && c0 < hd {
                let out_idx = ((batch_idx * sl + r1) * nh + head_idx) * hd + c0;
                unsafe {
                    *output.get_unchecked_mut(out_idx) = o_acc[base + 2];
                }
            }
            if r1 < sl && c1 < hd {
                let out_idx = ((batch_idx * sl + r1) * nh + head_idx) * hd + c1;
                unsafe {
                    *output.get_unchecked_mut(out_idx) = o_acc[base + 3];
                }
            }

            ci += 1;
        }
    }
}
