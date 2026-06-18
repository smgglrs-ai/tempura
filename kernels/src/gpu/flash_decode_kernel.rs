use cuda_device::{kernel, ptx_asm, thread, warp, DisjointSlice, SharedArray};
use cuda_host::cuda_module;

const HD: usize = 128;
const BC: usize = 32;
const SMEM_KV: usize = BC * HD; // 4096 floats per tile

#[inline(always)]
fn exp_f32(x: f32) -> f32 {
    let result: f32;
    unsafe {
        ptx_asm!(
            "ex2.approx.ftz.f32 %0, %1;",
            out("=f") result,
            in("f") x * 1.442695041,
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

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Split-K decode attention: each CTA computes partial attention for one
    /// Q token over a chunk of KV positions.
    ///
    /// Grid:  (num_heads, num_splits, batch)
    /// Block: (32, 1, 1) — 1 warp
    ///
    /// Outputs per CTA (written to partials buffer):
    ///   partials[batch][head][split][0..head_dim]  — unnormalized weighted V sum
    ///   partials[batch][head][split][head_dim]      — row_max
    ///   partials[batch][head][split][head_dim+1]    — row_sum
    ///
    /// Stride: (head_dim + 2) per split entry.
    #[kernel]
    pub fn flash_decode_partial(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut partials: DisjointSlice<f32>,
        _batch: u32,
        kv_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        num_splits: u32,
        kv_per_split: u32,
        scale: f32,
    ) {
        static mut K_SMEM: SharedArray<f32, SMEM_KV> = SharedArray::UNINIT;
        static mut V_SMEM: SharedArray<f32, SMEM_KV> = SharedArray::UNINIT;

        let lane = thread::threadIdx_x() as usize;
        let head_idx = thread::blockIdx_x() as usize;
        let split_idx = thread::blockIdx_y() as usize;
        let batch_idx = thread::blockIdx_z() as usize;

        let kvl = kv_len as usize;
        let nh = num_heads as usize;
        let nkv = num_kv_heads as usize;
        let hd = head_dim as usize;
        let n_splits = num_splits as usize;
        let kv_split = kv_per_split as usize;

        let heads_per_group = nh / nkv;
        let kv_head_idx = head_idx / heads_per_group;

        let kv_start = split_idx * kv_split;
        let kv_end = if kv_start + kv_split < kvl {
            kv_start + kv_split
        } else {
            kvl
        };

        // Load Q vector into registers: 128 floats / 32 threads = 4 per thread
        let q_base = (batch_idx * nh + head_idx) * hd;
        let mut q_reg = [0.0f32; 4];
        let mut qi = 0usize;
        while qi < 4 {
            let dim = lane * 4 + qi;
            if dim < hd {
                q_reg[qi] = q[q_base + dim];
            }
            qi += 1;
        }

        // Online softmax state
        let mut row_max = f32::NEG_INFINITY;
        let mut row_sum = 0.0f32;

        // Output accumulator: each thread holds 4 dims of the head_dim=128 output
        let mut o_acc = [0.0f32; 4];

        // Process KV tiles within this split
        let mut kv_pos = kv_start;
        while kv_pos < kv_end {
            let tile_end = if kv_pos + BC < kv_end {
                kv_pos + BC
            } else {
                kv_end
            };
            let tile_size = tile_end - kv_pos;

            // Load K tile into SMEM: [tile_size × head_dim]
            // 32 threads cooperatively load tile_size * 128 floats
            let total_elems = tile_size * HD;
            let mut li = lane;
            while li < total_elems {
                let row = li / HD;
                let col = li % HD;
                let global_kv = kv_pos + row;
                unsafe {
                    let k_idx = ((batch_idx * kvl + global_kv) * nkv + kv_head_idx) * hd + col;
                    K_SMEM[li] = k[k_idx];
                }
                li += 32;
            }
            thread::sync_threads();

            // Compute scores: each thread computes one score per KV position
            // 32 threads, up to 32 KV positions per tile — 1 per thread
            let mut scores = [0.0f32; 1];
            let kv_idx_in_tile = lane;

            if kv_idx_in_tile < tile_size {
                // Dot product q · k[kv_idx_in_tile] across head_dim
                // Each thread has 4 dims of Q in q_reg
                // Need full dot product — each thread computes partial, then reduce
                let mut dot = 0.0f32;
                let mut d = 0usize;
                while d < hd {
                    unsafe {
                        dot += q[q_base + d] * K_SMEM[kv_idx_in_tile * HD + d];
                    }
                    d += 1;
                }
                scores[0] = dot * scale;
            } else {
                scores[0] = f32::NEG_INFINITY;
            }

            // Find tile max via warp reduction
            let mut tile_max = scores[0];
            tile_max = max_f32(tile_max, warp::shuffle_xor_f32(tile_max, 16));
            tile_max = max_f32(tile_max, warp::shuffle_xor_f32(tile_max, 8));
            tile_max = max_f32(tile_max, warp::shuffle_xor_f32(tile_max, 4));
            tile_max = max_f32(tile_max, warp::shuffle_xor_f32(tile_max, 2));
            tile_max = max_f32(tile_max, warp::shuffle_xor_f32(tile_max, 1));

            // Online softmax correction
            let new_max = max_f32(row_max, tile_max);
            let correction = if row_max > f32::NEG_INFINITY {
                exp_f32(row_max - new_max)
            } else {
                0.0
            };

            // Rescale existing accumulator
            qi = 0;
            while qi < 4 {
                o_acc[qi] *= correction;
                qi += 1;
            }
            row_sum *= correction;

            // Compute exp(score - new_max) and tile sum
            let p = if scores[0] > f32::NEG_INFINITY {
                exp_f32(scores[0] - new_max)
            } else {
                0.0
            };

            let mut tile_sum = p;
            tile_sum += warp::shuffle_xor_f32(tile_sum, 16);
            tile_sum += warp::shuffle_xor_f32(tile_sum, 8);
            tile_sum += warp::shuffle_xor_f32(tile_sum, 4);
            tile_sum += warp::shuffle_xor_f32(tile_sum, 2);
            tile_sum += warp::shuffle_xor_f32(tile_sum, 1);

            row_sum += tile_sum;
            row_max = new_max;

            // Load V tile into SMEM
            thread::sync_threads();
            li = lane;
            while li < total_elems {
                let row = li / HD;
                let col = li % HD;
                let global_kv = kv_pos + row;
                unsafe {
                    let v_idx = ((batch_idx * kvl + global_kv) * nkv + kv_head_idx) * hd + col;
                    V_SMEM[li] = v[v_idx];
                }
                li += 32;
            }
            thread::sync_threads();

            // Accumulate O += p * V for this thread's KV position
            // Thread `lane` has weight `p` for KV position `kv_idx_in_tile`
            // Each thread reads V[kv_idx_in_tile, dim] for its 4 dims, multiplies by p,
            // then warp-reduces across threads to accumulate into the output dims.
            //
            // Strategy: each thread broadcasts its (p, kv_idx) and all threads
            // accumulate the weighted V contribution for their owned dims.
            // This is more efficient: iterate over valid KV positions, each thread
            // reads its own dims from V_SMEM.
            let mut ki = 0usize;
            while ki < tile_size {
                // Get weight for this KV position from the thread that owns it
                let w = warp::shuffle_f32(p, ki as u32);

                // Each thread accumulates w * V[ki, own_dims]
                qi = 0;
                while qi < 4 {
                    let dim = lane * 4 + qi;
                    if dim < hd {
                        unsafe {
                            o_acc[qi] += w * V_SMEM[ki * HD + dim];
                        }
                    }
                    qi += 1;
                }
                ki += 1;
            }

            thread::sync_threads();
            kv_pos += BC;
        }

        // Write partial results to global memory
        // Layout: partials[batch][head][split][0..head_dim+2]
        let stride = hd + 2;
        let partial_base =
            ((batch_idx * nh + head_idx) * n_splits + split_idx) * stride;

        // Each thread writes its 4 dims of output
        qi = 0;
        while qi < 4 {
            let dim = lane * 4 + qi;
            if dim < hd {
                unsafe {
                    *partials.get_unchecked_mut(partial_base + dim) = o_acc[qi];
                }
            }
            qi += 1;
        }

        // Thread 0 writes softmax stats
        if lane == 0 {
            unsafe {
                *partials.get_unchecked_mut(partial_base + hd) = row_max;
                *partials.get_unchecked_mut(partial_base + hd + 1) = row_sum;
            }
        }
    }

    /// Reduction kernel: combine partial results from split-K decode.
    ///
    /// Grid:  (num_heads, 1, batch)
    /// Block: (32, 1, 1) — 1 warp
    ///
    /// Reads partials[batch][head][0..num_splits][head_dim+2]
    /// Writes final output[batch][head][head_dim]
    #[kernel]
    pub fn flash_decode_reduce(
        partials: &[f32],
        mut output: DisjointSlice<f32>,
        _batch: u32,
        num_heads: u32,
        head_dim: u32,
        num_splits: u32,
    ) {
        let lane = thread::threadIdx_x() as usize;
        let head_idx = thread::blockIdx_x() as usize;
        let batch_idx = thread::blockIdx_z() as usize;

        let nh = num_heads as usize;
        let hd = head_dim as usize;
        let n_splits = num_splits as usize;
        let stride = hd + 2;

        let base = ((batch_idx * nh + head_idx) * n_splits) * stride;

        // First pass: find global max across all splits
        let mut global_max = f32::NEG_INFINITY;
        let mut si = 0usize;
        while si < n_splits {
            let partial_max = partials[base + si * stride + hd];
            global_max = max_f32(global_max, partial_max);
            si += 1;
        }

        // Second pass: accumulate corrected outputs and sums
        let mut final_out = [0.0f32; 4];
        let mut final_sum = 0.0f32;

        si = 0;
        while si < n_splits {
            let split_base = base + si * stride;
            let partial_max = partials[split_base + hd];
            let partial_sum = partials[split_base + hd + 1];

            let correction = if partial_max > f32::NEG_INFINITY {
                exp_f32(partial_max - global_max)
            } else {
                0.0
            };

            // Accumulate corrected output (each thread handles its 4 dims)
            let mut qi = 0usize;
            while qi < 4 {
                let dim = lane * 4 + qi;
                if dim < hd {
                    final_out[qi] += partials[split_base + dim] * correction;
                }
                qi += 1;
            }

            final_sum += partial_sum * correction;
            si += 1;
        }

        // Normalize by total sum
        let inv_sum = if final_sum > 0.0 {
            1.0 / final_sum
        } else {
            0.0
        };

        // Write final output
        let out_base = (batch_idx * nh + head_idx) * hd;
        let mut qi = 0usize;
        while qi < 4 {
            let dim = lane * 4 + qi;
            if dim < hd {
                let out_idx = out_base + dim;
                unsafe {
                    *output.get_unchecked_mut(out_idx) = final_out[qi] * inv_sum;
                }
            }
            qi += 1;
        }
    }
}
