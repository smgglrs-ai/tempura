/*
 * oxide-decode-attn: GPU decode flash-attention kernel via rustc-codegen-cuda.
 *
 * Unlike the cuTile version (cutile_decode_attn), this kernel accepts kv_len
 * as a *const i32 device pointer. The pointer address is stable (baked into
 * the CUDA graph at capture); the value at that address is updated H->D each
 * decode step. This eliminates the stale-kv_len problem in the single-capture
 * CUDA graph.
 *
 * Build:
 *   export CUDA_OXIDE_BACKEND=~/Code/github.com/NVLabs/cuda-oxide/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
 *   RUSTUP_TOOLCHAIN=nightly-2026-10-11 cargo oxide build
 *
 * Layout (matching cutile_decode_attn and CutileKvCache):
 *   q:       [n_heads * head_dim] f32
 *   k_cache: [gpu_max_seq_len * n_kv_heads * head_dim] f32, layout [token, kv_head, head_dim]
 *   v_cache: same layout as k_cache
 *   out:     [n_heads * head_dim] f32
 *
 * Grid:  (n_heads, 1, 1) -- one block per attention head
 * Block: (32, 1, 1) -- one warp per head (32 lanes share the work)
 * Smem:  2 * head_dim * 4 bytes = Q stage + accumulator
 *
 * Online softmax (exp2 trick, matching cutile_decode_attn):
 *   score_exp2 = Q*K * (scale / ln2)
 *   exp2(score_exp2) = exp(Q*K * scale)
 */

use cuda_device::{DynamicSharedArray, kernel, thread, warp, float};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_core::simt::LaunchConfig;

const WARPS_PER_BLOCK_KPAR: u32 = 8;
const BLOCK_DIM_KPAR: u32 = WARPS_PER_BLOCK_KPAR * 32;

/// Split-K variant: 4 warps per block, grid (n_heads, n_splits).
const WARPS_PER_BLOCK_SPLITK: u32 = 4;
const BLOCK_DIM_SPLITK: u32 = WARPS_PER_BLOCK_SPLITK * 32;

/// Base f32 index of the (head, split) scratch slot:
/// acc at [base..base+head_dim), m at [base+head_dim], l at [base+head_dim+1].
/// Mirrors `tempura_kernels::decode_attn_splitk::slot_base` (Kani-proven).
#[inline(always)]
fn split_slot_base(h: u32, n_splits: u32, dstride: usize, s: u32) -> usize {
    (h as usize * n_splits as usize + s as usize) * dstride
}

#[cuda_module]
mod kernels {
    use super::*;

    /// K-parallel decode attention: WARPS_PER_BLOCK_KPAR warps per head.
    ///
    /// Smem layout (W = WARPS_PER_BLOCK_KPAR, D = head_dim):
    ///   [0..D]              Q staging (shared)
    ///   [D..D+W]            m_local per warp
    ///   [D+W..D+2W]         l_local per warp
    ///   [D+2W..D+2W+W*D]    acc_local per warp (each warp owns D floats)
    #[kernel]
    pub unsafe fn oxide_decode_attn_kpar(
        out: *mut f32,          // [n_heads * head_dim] -- raw ptr avoids DisjointSlice 2-param ABI
        q:   *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        kv_len_ptr: *const i32,
        scale: f32,
        n_kv_heads: u32,
        head_dim: u32,
        heads_per_group: u32,
    ) {
        let h       = thread::blockIdx_x();
        let tid     = thread::threadIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;

        let kv_len = *kv_len_ptr;
        if kv_len <= 0 { return; }

        let kv_h      = h / heads_per_group;
        let kv_stride = n_kv_heads * head_dim;
        let ln2: f32  = 0.693_147_18_f32;
        let scale_log2 = scale / ln2;

        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        let w = WARPS_PER_BLOCK_KPAR;

        // Q staging: smem[0..head_dim]
        let q_base = h * head_dim;
        let mut i = tid;
        while i < head_dim {
            *smem.add(i as usize) = *q.add((q_base + i) as usize);
            i += BLOCK_DIM_KPAR;
        }
        // Acc region for this warp: smem[head_dim + 2*W + warp_id*head_dim .. ]
        let acc_base = (head_dim + 2 * w + warp_id * head_dim) as usize;
        let mut ai = lane;
        while ai < head_dim {
            *smem.add(acc_base + ai as usize) = 0.0_f32;
            ai += 32;
        }
        thread::sync_threads();

        // Per-warp online softmax over warp's KV slice
        let mut m_local: f32 = f32::NEG_INFINITY;
        let mut l_local: f32 = 0.0_f32;

        let mut t = warp_id as i32;
        while t < kv_len {
            let kv_base = (t as u32 * kv_stride + kv_h * head_dim) as usize;

            let mut qk: f32 = 0.0_f32;
            let mut di = lane;
            while di < head_dim {
                qk += *smem.add(di as usize) * *k_cache.add(kv_base + di as usize);
                di += 32;
            }
            qk = warp::reduce_sum_f32(qk);

            let score = qk * scale_log2;
            let m_new  = if score > m_local { score } else { m_local };
            let alpha: f32 = float::ex2_approx_f32(m_local - m_new);
            let p:     f32 = float::ex2_approx_f32(score - m_new);

            l_local = l_local * alpha + p;

            let mut vi = lane;
            while vi < head_dim {
                let v_val = *v_cache.add(kv_base + vi as usize);
                let old   = *smem.add(acc_base + vi as usize);
                *smem.add(acc_base + vi as usize) = old * alpha + v_val * p;
                vi += 32;
            }
            m_local = m_new;
            t += WARPS_PER_BLOCK_KPAR as i32;
        }

        // Write per-warp m and l to smem -- ALL warps participate
        if lane == 0 {
            *smem.add((head_dim + warp_id) as usize) = m_local;
            *smem.add((head_dim + w + warp_id) as usize) = l_local;
        }
        thread::sync_threads();   // barrier 1: m/l visible to all

        // All threads compute m_global (tiny loop, W=8, trivial)
        let mut m_global: f32 = f32::NEG_INFINITY;
        let mut ww = 0u32;
        while ww < w {
            let mw = *smem.add((head_dim + ww) as usize);
            if mw > m_global { m_global = mw; }
            ww += 1;
        }

        // All warps rescale their own acc region: acc_w *= exp2(m_local - m_global)
        let alpha_w: f32 = float::ex2_approx_f32(m_local - m_global);
        let mut di = lane;
        while di < head_dim {
            let a = *smem.add(acc_base + di as usize);
            *smem.add(acc_base + di as usize) = a * alpha_w;
            di += 32;
        }
        thread::sync_threads();   // barrier 2: all acc rescaled

        // Warp 0 computes l_global and final output (lanes cover head_dim)
        if warp_id == 0 {
            // l_global: sum of exp2(m_w - m_global) * l_w across warps
            let mut l_global: f32 = 0.0_f32;
            let mut ww = 0u32;
            while ww < w {
                let mw = *smem.add((head_dim + ww) as usize);
                let lw = *smem.add((head_dim + w + ww) as usize);
                l_global += float::ex2_approx_f32(mw - m_global) * lw;
                ww += 1;
            }

            // Sum acc across warps, normalize, write out
            let out_base = h * head_dim;
            let mut oi = lane;
            while oi < head_dim {
                let mut total: f32 = 0.0_f32;
                let mut ww = 0u32;
                while ww < w {
                    total += *smem.add((head_dim + 2 * w + ww * head_dim + oi) as usize);
                    ww += 1;
                }
                *out.add((out_base + oi) as usize) = total / l_global;
                oi += 32;
            }
        }
    }

    /// Split-K decode attention: grid (n_heads, n_splits), 4 warps per block.
    ///
    /// Why: one-block-per-head (kpar/single-warp) launches only n_heads blocks
    /// (e.g. 32 on Llama-3.1-8B) on a 170-SM GPU -- 19% occupancy regardless of
    /// warp count. Splitting the KV sequence across n_splits blocks per head
    /// fills the machine. Each block produces an online-softmax partial
    /// (m, l, unnormalised acc) in `scratch`; `oxide_decode_attn_combine`
    /// merges them.
    ///
    /// Chunking is kv_len-derived at execution time (graph-stable): with
    /// chunk = ceil(kv_len / n_splits), block (h, s) covers tokens
    /// [s*chunk, min((s+1)*chunk, kv_len)). Blocks whose chunk is empty write
    /// m = -inf, l = 0, acc = 0 so the combine pass can skip them.
    ///
    /// Scratch layout, per (head, split) pair, base = (h*n_splits + s)*(head_dim+2):
    ///   [0..head_dim)        acc, rescaled to the split's m but NOT normalised
    ///   [head_dim]           m (split max score, or -inf when empty)
    ///   [head_dim + 1]       l (split sum of exp)
    ///
    /// Smem layout (W = 4 warps, D = head_dim), identical to kpar:
    ///   [0..D)               Q staging (shared)
    ///   [D..D+W)             m_local per warp
    ///   [D+W..D+2W)          l_local per warp
    ///   [D+2W..D+2W+W*D)     acc_local per warp
    #[kernel]
    pub unsafe fn oxide_decode_attn_splitk(
        scratch:    *mut f32,        // [n_heads * n_splits * (head_dim + 2)]
        q:          *const f32,
        k_cache:    *const f32,
        v_cache:    *const f32,
        kv_len_ptr: *const i32,
        scale:      f32,
        n_kv_heads: u32,
        head_dim:   u32,
        heads_per_group: u32,
        n_splits:   u32,
    ) {
        let h       = thread::blockIdx_x();
        let s       = thread::blockIdx_y();
        let tid     = thread::threadIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;
        let w       = WARPS_PER_BLOCK_SPLITK;

        let kv_len = *kv_len_ptr;

        let kv_h      = h / heads_per_group;
        let kv_stride = n_kv_heads * head_dim;
        let ln2: f32  = 0.693_147_18_f32;
        let scale_log2 = scale / ln2;

        // Split bounds derived from the CURRENT kv_len (device-read).
        let n_splits_i = n_splits as i32;
        let chunk = if kv_len > 0 { (kv_len + n_splits_i - 1) / n_splits_i } else { 0 };
        let t0    = s as i32 * chunk;
        let t1    = if t0 + chunk < kv_len { t0 + chunk } else { kv_len };

        let dstride = (head_dim + 2) as usize;
        let base    = (h as usize * n_splits as usize + s as usize) * dstride;

        let smem: *mut f32 = DynamicSharedArray::<f32>::get();

        if t0 >= t1 {
            // Empty chunk: emit an identity partial (combine skips m == -inf).
            let mut i = tid;
            while i < head_dim {
                *scratch.add(base + i as usize) = 0.0_f32;
                i += BLOCK_DIM_SPLITK;
            }
            if tid == 0 {
                *scratch.add(base + head_dim as usize) = f32::NEG_INFINITY;
                *scratch.add(base + head_dim as usize + 1) = 0.0_f32;
            }
            return;
        }

        // Q staging: smem[0..head_dim]
        let q_base = h * head_dim;
        let mut i = tid;
        while i < head_dim {
            *smem.add(i as usize) = *q.add((q_base + i) as usize);
            i += BLOCK_DIM_SPLITK;
        }
        // Acc region for this warp: smem[head_dim + 2*W + warp_id*head_dim .. ]
        let acc_base = (head_dim + 2 * w + warp_id * head_dim) as usize;
        let mut ai = lane;
        while ai < head_dim {
            *smem.add(acc_base + ai as usize) = 0.0_f32;
            ai += 32;
        }
        thread::sync_threads();

        // Per-warp online softmax over the warp's stripe of [t0, t1).
        let mut m_local: f32 = f32::NEG_INFINITY;
        let mut l_local: f32 = 0.0_f32;

        let mut t = t0 + warp_id as i32;
        while t < t1 {
            let kv_base = (t as u32 * kv_stride + kv_h * head_dim) as usize;

            let mut qk: f32 = 0.0_f32;
            let mut di = lane;
            while di < head_dim {
                qk += *smem.add(di as usize) * *k_cache.add(kv_base + di as usize);
                di += 32;
            }
            qk = warp::reduce_sum_f32(qk);

            let score = qk * scale_log2;
            let m_new  = if score > m_local { score } else { m_local };
            let alpha: f32 = float::ex2_approx_f32(m_local - m_new);
            let p:     f32 = float::ex2_approx_f32(score - m_new);

            l_local = l_local * alpha + p;

            let mut vi = lane;
            while vi < head_dim {
                let v_val = *v_cache.add(kv_base + vi as usize);
                let old   = *smem.add(acc_base + vi as usize);
                *smem.add(acc_base + vi as usize) = old * alpha + v_val * p;
                vi += 32;
            }
            m_local = m_new;
            t += w as i32;
        }

        // Write per-warp m and l to smem -- ALL warps participate
        if lane == 0 {
            *smem.add((head_dim + warp_id) as usize) = m_local;
            *smem.add((head_dim + w + warp_id) as usize) = l_local;
        }
        thread::sync_threads();   // barrier 1: m/l visible to all

        // All threads compute m_global (tiny loop, W=4, trivial)
        let mut m_global: f32 = f32::NEG_INFINITY;
        let mut ww = 0u32;
        while ww < w {
            let mw = *smem.add((head_dim + ww) as usize);
            if mw > m_global { m_global = mw; }
            ww += 1;
        }

        // All warps rescale their own acc region: acc_w *= exp2(m_local - m_global)
        let alpha_w: f32 = float::ex2_approx_f32(m_local - m_global);
        let mut di = lane;
        while di < head_dim {
            let a = *smem.add(acc_base + di as usize);
            *smem.add(acc_base + di as usize) = a * alpha_w;
            di += 32;
        }
        thread::sync_threads();   // barrier 2: all acc rescaled

        // Warp 0 merges the per-warp partials into this split's scratch slot.
        if warp_id == 0 {
            let mut l_global: f32 = 0.0_f32;
            let mut ww = 0u32;
            while ww < w {
                let mw = *smem.add((head_dim + ww) as usize);
                let lw = *smem.add((head_dim + w + ww) as usize);
                l_global += float::ex2_approx_f32(mw - m_global) * lw;
                ww += 1;
            }

            // Sum acc across warps into scratch (unnormalised).
            let mut oi = lane;
            while oi < head_dim {
                let mut total: f32 = 0.0_f32;
                let mut ww = 0u32;
                while ww < w {
                    total += *smem.add((head_dim + 2 * w + ww * head_dim + oi) as usize);
                    ww += 1;
                }
                *scratch.add(base + oi as usize) = total;
                oi += 32;
            }
            if lane == 0 {
                *scratch.add(base + head_dim as usize) = m_global;
                *scratch.add(base + head_dim as usize + 1) = l_global;
            }
        }
    }

    /// Merge split-K partials and normalise: grid (n_heads), 128 threads.
    ///
    /// out[h,d] = (sum_s exp2(m_s - m_g) * acc_s[d]) / (sum_s exp2(m_s - m_g) * l_s)
    ///
    /// Splits with m_s == -inf (empty chunk) are skipped: exp2(-inf - m_g)
    /// would be 0 anyway, but skipping avoids the NaN from -inf - -inf when all
    /// splits are empty (then l_global == 0 and the output is written as 0).
    #[kernel]
    pub unsafe fn oxide_decode_attn_combine(
        out:     *mut f32,        // [n_heads * head_dim]
        scratch: *const f32,      // [n_heads * n_splits * (head_dim + 2)]
        n_splits: u32,
        head_dim: u32,
    ) {
        let h   = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let dstride = (head_dim + 2) as usize;

        let mut m_global: f32 = f32::NEG_INFINITY;
        let mut s = 0u32;
        while s < n_splits {
            let b = split_slot_base(h, n_splits, dstride, s);
            let m_s = *scratch.add(b + head_dim as usize);
            if m_s > m_global { m_global = m_s; }
            s += 1;
        }

        let mut l_global: f32 = 0.0_f32;
        let mut s2 = 0u32;
        while s2 < n_splits {
            let b = split_slot_base(h, n_splits, dstride, s2);
            let m_s = *scratch.add(b + head_dim as usize);
            if m_s != f32::NEG_INFINITY {
                l_global += float::ex2_approx_f32(m_s - m_global) * *scratch.add(b + head_dim as usize + 1);
            }
            s2 += 1;
        }

        let mut d = tid;
        while d < head_dim {
            let mut total: f32 = 0.0_f32;
            let mut s3 = 0u32;
            while s3 < n_splits {
                let b = split_slot_base(h, n_splits, dstride, s3);
                let m_s = *scratch.add(b + head_dim as usize);
                if m_s != f32::NEG_INFINITY {
                    total += float::ex2_approx_f32(m_s - m_global) * *scratch.add(b + d as usize);
                }
                s3 += 1;
            }
            *out.add((h * head_dim + d) as usize) =
                if l_global > 0.0_f32 { total / l_global } else { 0.0_f32 };
            d += BLOCK_DIM_SPLITK;
        }
    }

    /// Decode attention: one block per attention head, one warp per block.
    ///
    /// kv_len_ptr is a stable device pointer updated H->D before each CUDA
    /// graph replay -- the kernel reads the CURRENT kv_len at execution time.
    #[kernel]
    pub unsafe fn oxide_decode_attn(
        out: *mut f32,                // [n_heads * head_dim] -- raw ptr, 1 PTX param
        q:       *const f32,          // [n_heads * head_dim]
        k_cache: *const f32,          // [gpu_max_seq_len * n_kv_heads * head_dim]
        v_cache: *const f32,          // same layout as k_cache
        kv_len_ptr: *const i32,       // device pointer -- read at execution time
        scale: f32,
        n_kv_heads: u32,
        head_dim: u32,
        heads_per_group: u32,         // n_heads / n_kv_heads
    ) {
        let h    = thread::blockIdx_x();         // attention head index
        let lane = thread::threadIdx_x();         // 0..31

        // Read kv_len from device memory at execution time.
        // The pointer address is baked into the graph; only the value changes.
        let kv_len = *kv_len_ptr;
        if kv_len <= 0 {
            return;
        }

        let kv_h = h / heads_per_group;          // GQA: which KV head
        let kv_stride = n_kv_heads * head_dim;   // stride between KV tokens

        // Shared memory: smem[0..head_dim] = Q staging; smem[head_dim..2*head_dim] = acc
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();

        // Stage Q into shared memory cooperatively.
        let q_base = h * head_dim;
        let mut i = lane;
        while i < head_dim {
            *smem.add(i as usize) = *q.add((q_base + i) as usize);
            i += 32;
        }

        // Zero out accumulator.
        let acc_smem: *mut f32 = smem.add(head_dim as usize);
        let mut ai = lane;
        while ai < head_dim {
            *acc_smem.add(ai as usize) = 0.0_f32;
            ai += 32;
        }
        thread::sync_threads();

        // Online softmax constants (exp2 trick).
        let ln2: f32 = 0.693_147_18_f32;
        let scale_log2 = scale / ln2;

        // Running statistics.
        let mut m_val: f32 = f32::NEG_INFINITY;
        let mut l_val: f32 = 0.0_f32;

        // Sequential loop over KV tokens.
        let mut t: i32 = 0;
        while t < kv_len {
            let kv_base = (t as u32 * kv_stride + kv_h * head_dim) as usize;

            // Q*K dot product: each lane accumulates a stripe, then warp-reduce.
            let mut qk: f32 = 0.0_f32;
            let mut di = lane;
            while di < head_dim {
                qk += *smem.add(di as usize) * *k_cache.add(kv_base + di as usize);
                di += 32;
            }
            qk = warp::reduce_sum_f32(qk);  // all lanes now hold full dot product

            let score = qk * scale_log2;

            // Online softmax update.
            let m_new = if score > m_val { score } else { m_val };
            let alpha: f32 = float::ex2_approx_f32(m_val - m_new);
            let p:     f32 = float::ex2_approx_f32(score - m_new);

            l_val = l_val * alpha + p;

            // Accumulate weighted V.
            let mut vi = lane;
            while vi < head_dim {
                let v_val  = *v_cache.add(kv_base + vi as usize);
                let acc_old = *acc_smem.add(vi as usize);
                *acc_smem.add(vi as usize) = acc_old * alpha + v_val * p;
                vi += 32;
            }
            thread::sync_threads();

            m_val = m_new;
            t += 1;
        }

        // Normalise and write output.
        let out_base = h * head_dim;
        let mut oi = lane;
        while oi < head_dim {
            let val = *acc_smem.add(oi as usize) / l_val;
            *out.add((out_base + oi) as usize) = val;
            oi += 32;
        }
    }

}

// -- Host-side correctness test ------------------------------------------------

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();

    // Small test: n_heads=4, n_kv_heads=2, head_dim=8, kv_len=3
    let n_heads: usize = 4;
    let n_kv_heads: usize = 2;
    let head_dim: usize = 8;
    let kv_len: i32 = 3;
    let heads_per_group = (n_heads / n_kv_heads) as u32;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Q: all ones.
    let q_host: Vec<f32> = vec![1.0_f32; n_heads * head_dim];

    // KV cache: all ones (simplified, so Q*K = head_dim for all t, kv_h).
    // shape: [kv_len_max=8, n_kv_heads, head_dim]
    let kv_max = 8usize;
    let k_host: Vec<f32> = vec![1.0_f32; kv_max * n_kv_heads * head_dim];
    let v_host: Vec<f32> = vec![0.5_f32; kv_max * n_kv_heads * head_dim];

    let kv_len_host: i32 = kv_len;

    let q_dev   = DeviceBuffer::from_host(&stream, &q_host).expect("q upload");
    let k_dev   = DeviceBuffer::from_host(&stream, &k_host).expect("k upload");
    let v_dev   = DeviceBuffer::from_host(&stream, &v_host).expect("v upload");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, n_heads * head_dim).expect("out alloc");
    let kv_len_dev  = DeviceBuffer::from_host(&stream, &[kv_len_host]).expect("kv_len upload");

    let module = kernels::load(&ctx).expect("load oxide decode-attn module");

    // -- Test 1: original single-warp kernel ------------------------------------
    let shared_bytes = (2 * head_dim * std::mem::size_of::<f32>()) as u32;
    let cfg = LaunchConfig {
        grid_dim:  (n_heads as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    unsafe {
        module.oxide_decode_attn(
            stream.as_ref(),
            cfg,
            out_dev.cu_deviceptr() as *mut f32,
            q_dev.cu_deviceptr() as *const f32,
            k_dev.cu_deviceptr() as *const f32,
            v_dev.cu_deviceptr() as *const f32,
            kv_len_dev.cu_deviceptr() as *const i32,
            scale,
            n_kv_heads as u32,
            head_dim as u32,
            heads_per_group,
        )
    }.expect("single-warp kernel launch");

    let result = out_dev.to_host_vec(&stream).expect("download");
    let expected = 0.5_f32;
    let max_err = result.iter().map(|v| (v - expected).abs()).fold(0.0_f32, f32::max);
    if max_err < 0.01 {
        println!("oxide-decode-attn single-warp: PASSED (expected~={expected:.3}, max_err={max_err:.4})");
    } else {
        eprintln!("oxide-decode-attn single-warp: FAILED (max_err={max_err:.4})");
        std::process::exit(1);
    }

    // -- Test 2: K-parallel kernel (WARPS_PER_BLOCK_KPAR warps) ----------------
    let w = WARPS_PER_BLOCK_KPAR as usize;
    // smem: (head_dim*(W+1) + 2*W) * 4 bytes
    let kpar_smem = ((head_dim * (w + 1) + 2 * w) * std::mem::size_of::<f32>()) as u32;
    let kpar_cfg = LaunchConfig {
        grid_dim:  (n_heads as u32, 1, 1),
        block_dim: (BLOCK_DIM_KPAR, 1, 1),
        shared_mem_bytes: kpar_smem,
    };

    let mut out_kpar = DeviceBuffer::<f32>::zeroed(&stream, n_heads * head_dim).expect("out_kpar alloc");
    unsafe {
        module.oxide_decode_attn_kpar(
            stream.as_ref(),
            kpar_cfg,
            out_kpar.cu_deviceptr() as *mut f32,
            q_dev.cu_deviceptr() as *const f32,
            k_dev.cu_deviceptr() as *const f32,
            v_dev.cu_deviceptr() as *const f32,
            kv_len_dev.cu_deviceptr() as *const i32,
            scale,
            n_kv_heads as u32,
            head_dim as u32,
            heads_per_group,
        )
    }.expect("kpar kernel launch");

    let result_kpar = out_kpar.to_host_vec(&stream).expect("kpar download");
    let max_err_kpar = result.iter().zip(result_kpar.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    if max_err_kpar < 0.01 {
        println!("oxide-decode-attn kpar (W={w}): PASSED vs single-warp (max_diff={max_err_kpar:.4})");
    } else {
        eprintln!("oxide-decode-attn kpar (W={w}): FAILED vs single-warp (max_diff={max_err_kpar:.4})");
        eprintln!("  kpar[0..8] = {:?}", &result_kpar[..8.min(result_kpar.len())]);
        eprintln!("  ref [0..8] = {:?}", &result[..8.min(result.len())]);
        std::process::exit(1);
    }

    // -- Test 3: split-K kernel (grid n_heads x n_splits) + combine ----------
    //
    // kv_len values 3, 13, 100 with n_splits=16 exercise: empty trailing
    // splits (chunk < n_splits), non-divisible chunking, and warps within a
    // block seeing zero tokens (chunk=1 < WARPS_PER_BLOCK_SPLITK=4).
    let splits: u32 = 16;
    let kv_max3 = 128usize;
    // Varied K/V: all-ones data would give every split an identical partial,
    // hiding chunking/indexing bugs.
    let k3: Vec<f32> = (0..kv_max3 * n_kv_heads * head_dim)
        .map(|i| ((((i * 1103515245 + 12345) >> 16) as i32 % 13) - 6) as f32 * 0.1)
        .collect();
    let v3: Vec<f32> = (0..kv_max3 * n_kv_heads * head_dim)
        .map(|i| (((i * 2654435761) >> 13) & 0xFF) as f32 / 255.0)
        .collect();
    let k3_dev = DeviceBuffer::from_host(&stream, &k3).expect("k3 upload");
    let v3_dev = DeviceBuffer::from_host(&stream, &v3).expect("v3 upload");
    let scratch_len = n_heads * splits as usize * (head_dim + 2);
    let scratch_dev = DeviceBuffer::<f32>::zeroed(&stream, scratch_len).expect("scratch alloc");

    let splitk_smem = ((head_dim * (WARPS_PER_BLOCK_SPLITK as usize + 1)
        + 2 * WARPS_PER_BLOCK_SPLITK as usize) * std::mem::size_of::<f32>()) as u32;
    let splitk_cfg = LaunchConfig {
        grid_dim:  (n_heads as u32, splits, 1),
        block_dim: (BLOCK_DIM_SPLITK, 1, 1),
        shared_mem_bytes: splitk_smem,
    };
    let combine_cfg = LaunchConfig {
        grid_dim:  (n_heads as u32, 1, 1),
        block_dim: (BLOCK_DIM_SPLITK, 1, 1),
        shared_mem_bytes: 0,
    };

    for &kv3 in &[3usize, 13, 100] {
        // Reference: single-warp kernel on the same varied data.
        let kv_dev = DeviceBuffer::from_host(&stream, &[kv3 as i32]).expect("kv upload");
        let mut ref_dev = DeviceBuffer::<f32>::zeroed(&stream, n_heads * head_dim).expect("ref alloc");
        unsafe {
            module.oxide_decode_attn(
                stream.as_ref(),
                cfg,
                ref_dev.cu_deviceptr() as *mut f32,
                q_dev.cu_deviceptr() as *const f32,
                k3_dev.cu_deviceptr() as *const f32,
                v3_dev.cu_deviceptr() as *const f32,
                kv_dev.cu_deviceptr() as *const i32,
                scale,
                n_kv_heads as u32,
                head_dim as u32,
                heads_per_group,
            )
        }.expect("single-warp reference launch");
        let reference = ref_dev.to_host_vec(&stream).expect("ref download");

        unsafe {
            module.oxide_decode_attn_splitk(
                stream.as_ref(),
                splitk_cfg,
                scratch_dev.cu_deviceptr() as *mut f32,
                q_dev.cu_deviceptr() as *const f32,
                k3_dev.cu_deviceptr() as *const f32,
                v3_dev.cu_deviceptr() as *const f32,
                kv_dev.cu_deviceptr() as *const i32,
                scale,
                n_kv_heads as u32,
                head_dim as u32,
                heads_per_group,
                splits,
            )
        }.expect("splitk launch");
        let mut out3_dev = DeviceBuffer::<f32>::zeroed(&stream, n_heads * head_dim).expect("out3 alloc");
        unsafe {
            module.oxide_decode_attn_combine(
                stream.as_ref(),
                combine_cfg,
                out3_dev.cu_deviceptr() as *mut f32,
                scratch_dev.cu_deviceptr() as *const f32,
                splits,
                head_dim as u32,
            )
        }.expect("combine launch");
        let result3 = out3_dev.to_host_vec(&stream).expect("out3 download");

        let max_err3 = reference.iter().zip(result3.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        if max_err3 < 1e-3 {
            println!("oxide-decode-attn splitk (S={splits}, kv_len={kv3}): PASSED vs single-warp (max_diff={max_err3:.6})");
        } else {
            eprintln!("oxide-decode-attn splitk (S={splits}, kv_len={kv3}): FAILED vs single-warp (max_diff={max_err3:.6})");
            eprintln!("  ref    [0..8] = {:?}", &reference[..8.min(reference.len())]);
            eprintln!("  splitk [0..8] = {:?}", &result3[..8.min(result3.len())]);
            std::process::exit(1);
        }
    }

    println!("oxide-decode-attn: all tests PASSED");
}
