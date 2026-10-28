use std::sync::Arc;

use cuda_core::{Device, Stream};
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Reshape, Tensor};

use crate::{Attention, KernelError, Result, VerificationTier};

/// Tile dimensions (compile-time constants for the JIT).
/// BR=16: the flash kernels are CTA-count-bound at prefill sizes -- at
/// seq=128, BR=16 gives 256 CTAs on 170 SMs vs 128 at BR=32, ~1.9x faster
/// (tile sweep in bench-decode-ops, 2026-10-28; BR=8 overcorrects to
/// 57.8us).  BC=32: larger KV tiles are strictly slower at these sizes.
const BR: usize = 16;
const BC: usize = 32;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// Flash Attention v2 kernel (Dao et al., 2022).
    ///
    /// Computes attention for a SINGLE head: Q [SEQ, D], K [KVL, D], V [KVL, D].
    /// Each program handles one Q tile of BR rows.
    ///
    /// Algorithm (online softmax, exp2-based for stability):
    ///   For each Q tile q_tile (BR x D):
    ///     For each KV tile (j): k_tile (BC x D), v_tile (BC x D):
    ///       S = q_tile @ k_tile^T                  [BR x BC]
    ///       Apply causal mask (set future positions to -inf)
    ///       S *= scale / log2                       (for exp2 trick)
    ///       qk_max = rowmax(S)                      [BR x 1]
    ///       m_new = max(m_old, qk_max)
    ///       alpha = exp2(m_old - m_new)
    ///       P = exp2(S - m_new)                     [BR x BC]
    ///       l_new = alpha * l_old + rowsum(P)       [BR x 1]
    ///       acc = alpha * acc + P @ v_tile          [BR x D]
    ///     out_tile = acc / l_new
    ///
    /// Grid: (ceil(seq_len / BR), 1, 1)
    /// Generics: BR (Q tile rows), BC (KV tile rows), D (head dim), KVL (kv_len padded)
    /// KVL is now dynamic ([-1, D]) so the kernel compiles once per (BR, BC, D)
    /// tuple instead of once per kv_len_pad value.  The iteration count is driven
    /// by the runtime `kv_len` parameter; the tensor rank annotation just tells
    /// cuTile the expected shape for bounds checking.
    #[cutile::entry()]
    pub fn cutile_flash_attn_head<const BR: i32, const BC: i32, const D: i32>(
        out: &mut Tensor<f32, { [BR, D] }>,
        q: &Tensor<f32, { [-1, D] }>,
        k: &Tensor<f32, { [-1, D] }>,
        v: &Tensor<f32, { [-1, D] }>,
        scale: f32,
        cache_offset: i32,
        seq_len: i32,
        kv_len: i32,
    ) {
        let q_tile_idx = program_id(0);

        let part_q = q.partition(shape![BR, D]);
        let tq: Tile<f32, { [BR, D] }> = part_q.load([q_tile_idx, 0i32]);

        // exp2 trick: use scale / log2 so exp2(qk * scale/log2) = exp(qk * scale).
        let two: Tile<f32, { [] }> = constant(2.0f32, shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale: Tile<f32, { [BR, BC] }> = broadcast_scalar(scale / log2, shape![BR, BC]);

        // Running statistics for online softmax.
        let mut m_i: Tile<f32, { [BR, 1] }> = constant(f32::NEG_INFINITY, shape![BR, 1]);
        let mut l_i: Tile<f32, { [BR, 1] }> = constant(0.0f32, shape![BR, 1]);
        let mut acc: Tile<f32, { [BR, D] }> = constant(0.0f32, shape![BR, D]);

        let part_k = k.partition(shape![BC, D]);
        let part_v = v.partition(shape![BC, D]);
        // Ceil, NOT truncating division: K/V buffers are padded to a multiple
        // of BC and the bounds mask (offs_n < kv_len) excludes the pad, but a
        // truncated count DROPS the final partial tile entirely -- kv_len=12
        // gave 0 tiles (all-NaN output from 0/0) and kv_len=85 silently
        // skipped keys 64..84 for the last rows (found by bench-quality
        // small-m prefill NaN + nan-trace, 2026-10-28).
        let num_kv_tiles: i32 = ceil_div(kv_len, BC);

        // Causal boundary: only process KV tiles reachable from this Q tile.
        // cache_offset = kv_len - seq_len (positions already in KV cache before current tokens).
        let last_q_row: i32 = q_tile_idx * BR + BR - 1;
        let causal_limit_row: i32 = if last_q_row < seq_len {
            last_q_row
        } else {
            seq_len - 1
        };
        let causal_kv_max: i32 = cache_offset + causal_limit_row;
        let max_kv_tile: i32 = ceil_div(causal_kv_max + 1, BC);
        let tile_count: i32 = min(num_kv_tiles, max_kv_tile);

        // Precompute per-lane Q row indices (absolute, including cache offset).
        let offs_m_iota: Tile<i32, { [BR] }> = iota(shape![BR]);
        let offs_m_base: Tile<i32, { [BR, 1] }> =
            broadcast_scalar(q_tile_idx * BR + cache_offset, shape![BR, 1])
                + offs_m_iota.reshape(shape![BR, 1]);
        let offs_m: Tile<i32, { [BR, BC] }> = offs_m_base.broadcast(shape![BR, BC]);

        // KV column iota (reused across tiles; each tile adds j*BC).
        let offs_n_iota: Tile<i32, { [BC] }> = iota(shape![BC]);
        let offs_n_base: Tile<i32, { [BR, BC] }> =
            offs_n_iota.reshape(shape![1, BC]).broadcast(shape![BR, BC]);

        let kv_len_tile: Tile<i32, { [BR, BC] }> = broadcast_scalar(kv_len, shape![BR, BC]);
        let mask_add_true: Tile<f32, { [BR, BC] }> = constant(0.0f32, shape![BR, BC]);
        let mask_add_false: Tile<f32, { [BR, BC] }> = constant(f32::NEG_INFINITY, shape![BR, BC]);

        for j in 0i32..tile_count {
            let tile_k: Tile<f32, { [BC, D] }> = part_k.load_pipelined::<4>([j, 0i32]);

            // S = Q @ K^T  [BR x BC]
            let mut qk: Tile<f32, { [BR, BC] }> = constant(0.0f32, shape![BR, BC]);
            qk = mma(tq, tile_k.transpose(), qk);
            qk = qk * qk_scale;

            // Causal mask: allow kv_col <= q_row (causal) and kv_col < kv_len (padding).
            let offs_n: Tile<i32, { [BR, BC] }> =
                offs_n_base + broadcast_scalar(j * BC, shape![BR, BC]);
            let causal_mask: Tile<bool, { [BR, BC] }> = ge_tile(offs_m, offs_n);
            let bounds_mask: Tile<bool, { [BR, BC] }> = lt_tile(offs_n, kv_len_tile);
            let combined_mask: Tile<bool, { [BR, BC] }> = causal_mask & bounds_mask;
            qk = qk + select(combined_mask, mask_add_true, mask_add_false);

            // Online softmax update.
            let qk_max_1d: Tile<f32, { [BR] }> = reduce_max(qk, 1);
            let qk_max: Tile<f32, { [BR, 1] }> = qk_max_1d.reshape(shape![BR, 1]);
            let m_ij: Tile<f32, { [BR, 1] }> = max_tile(m_i, qk_max);

            // Correction factor for previously accumulated rows.
            let alpha: Tile<f32, { [BR, 1] }> = exp2(m_i - m_ij, ftz::Disabled);

            // P = exp2(S - m_new)  [BR x BC]
            let qk_shifted: Tile<f32, { [BR, BC] }> = qk - m_ij.broadcast(shape![BR, BC]);
            let p: Tile<f32, { [BR, BC] }> = exp2(qk_shifted, ftz::Disabled);

            // Update running sum and output accumulator.
            let l_ij_1d: Tile<f32, { [BR] }> = reduce_sum(p, 1);
            let l_ij: Tile<f32, { [BR, 1] }> = l_ij_1d.reshape(shape![BR, 1]);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(shape![BR, D]);

            let tile_v: Tile<f32, { [BC, D] }> = part_v.load_pipelined::<4>([j, 0i32]);
            acc = mma(p, tile_v, acc);
            m_i = m_ij;
        }

        // Normalize: divide each row by its softmax denominator.
        out.store(true_div(acc, l_i.broadcast(shape![BR, D])));
    }

    /// Multi-head prefill flash attention over head-major (gathered) buffers.
    ///
    /// Identical math to `cutile_flash_attn_head` (same online-softmax loop,
    /// same causal masking), but one launch covers ALL attention heads:
    ///
    ///   q: flat [n_heads * seq_pad * D], head-major -- head h owns rows
    ///      [h * seq_pad, (h+1) * seq_pad).  Rank-2 view [-1, D].
    ///   k, v: flat [n_kv_heads * kv_pad * D], head-major, same layout.
    ///   out: same layout/shape as q.
    ///
    /// Grid: (n_heads * seq_pad / BR, 1, 1) -- inferred from the out
    /// partition.  program_id(0) is the GLOBAL Q-tile index; the head and
    /// the tile-within-head are recovered by division.  GQA maps head h to
    /// kv-head h / heads_per_group; that head's KV tiles start at block
    /// kv_head * kv_tiles_per_head.
    ///
    /// q rows in [seq_len, seq_pad) and kv rows in [kv_len, kv_pad) are
    /// zero-padding written by `cutile_gather_heads`; the causal + bounds
    /// masks make them irrelevant, and the scatter never reads them back.
    #[cutile::entry()]
    pub fn cutile_flash_attn_heads<const BR: i32, const BC: i32, const D: i32>(
        out: &mut Tensor<f32, { [BR, D] }>,
        q: &Tensor<f32, { [-1, D] }>,
        k: &Tensor<f32, { [-1, D] }>,
        v: &Tensor<f32, { [-1, D] }>,
        scale: f32,
        cache_offset: i32,
        seq_len: i32,
        kv_len: i32,
        q_tiles_per_head: i32,
        kv_tiles_per_head: i32,
        heads_per_group: i32,
    ) {
        let q_tile_global = program_id(0);
        let head = q_tile_global / q_tiles_per_head;
        let q_tile_idx = q_tile_global % q_tiles_per_head;
        let kv_head = head / heads_per_group;

        let part_q = q.partition(shape![BR, D]);
        let tq: Tile<f32, { [BR, D] }> = part_q.load([q_tile_global, 0i32]);

        // exp2 trick: use scale / log2 so exp2(qk * scale/log2) = exp(qk * scale).
        let two: Tile<f32, { [] }> = constant(2.0f32, shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale: Tile<f32, { [BR, BC] }> = broadcast_scalar(scale / log2, shape![BR, BC]);

        // Running statistics for online softmax.
        let mut m_i: Tile<f32, { [BR, 1] }> = constant(f32::NEG_INFINITY, shape![BR, 1]);
        let mut l_i: Tile<f32, { [BR, 1] }> = constant(0.0f32, shape![BR, 1]);
        let mut acc: Tile<f32, { [BR, D] }> = constant(0.0f32, shape![BR, D]);

        let part_k = k.partition(shape![BC, D]);
        let part_v = v.partition(shape![BC, D]);
        // Ceil, NOT truncating division (see cutile_flash_attn_head).
        let num_kv_tiles: i32 = ceil_div(kv_len, BC);

        // Causal boundary: only process KV tiles reachable from this Q tile.
        // cache_offset = kv_len - seq_len (positions already in KV cache before current tokens).
        let last_q_row: i32 = q_tile_idx * BR + BR - 1;
        let causal_limit_row: i32 = if last_q_row < seq_len {
            last_q_row
        } else {
            seq_len - 1
        };
        let causal_kv_max: i32 = cache_offset + causal_limit_row;
        let max_kv_tile: i32 = ceil_div(causal_kv_max + 1, BC);
        let tile_count: i32 = min(num_kv_tiles, max_kv_tile);

        // Precompute per-lane Q row indices (absolute, including cache offset).
        let offs_m_iota: Tile<i32, { [BR] }> = iota(shape![BR]);
        let offs_m_base: Tile<i32, { [BR, 1] }> =
            broadcast_scalar(q_tile_idx * BR + cache_offset, shape![BR, 1])
                + offs_m_iota.reshape(shape![BR, 1]);
        let offs_m: Tile<i32, { [BR, BC] }> = offs_m_base.broadcast(shape![BR, BC]);

        // KV column iota (reused across tiles; each tile adds j*BC).
        let offs_n_iota: Tile<i32, { [BC] }> = iota(shape![BC]);
        let offs_n_base: Tile<i32, { [BR, BC] }> =
            offs_n_iota.reshape(shape![1, BC]).broadcast(shape![BR, BC]);

        let kv_len_tile: Tile<i32, { [BR, BC] }> = broadcast_scalar(kv_len, shape![BR, BC]);
        let mask_add_true: Tile<f32, { [BR, BC] }> = constant(0.0f32, shape![BR, BC]);
        let mask_add_false: Tile<f32, { [BR, BC] }> = constant(f32::NEG_INFINITY, shape![BR, BC]);

        let kv_block_base: i32 = kv_head * kv_tiles_per_head;
        for j in 0i32..tile_count {
            let tile_k: Tile<f32, { [BC, D] }> =
                part_k.load_pipelined::<4>([kv_block_base + j, 0i32]);

            // S = Q @ K^T  [BR x BC]
            let mut qk: Tile<f32, { [BR, BC] }> = constant(0.0f32, shape![BR, BC]);
            qk = mma(tq, tile_k.transpose(), qk);
            qk = qk * qk_scale;

            // Causal mask: allow kv_col <= q_row (causal) and kv_col < kv_len (padding).
            let offs_n: Tile<i32, { [BR, BC] }> =
                offs_n_base + broadcast_scalar(j * BC, shape![BR, BC]);
            let causal_mask: Tile<bool, { [BR, BC] }> = ge_tile(offs_m, offs_n);
            let bounds_mask: Tile<bool, { [BR, BC] }> = lt_tile(offs_n, kv_len_tile);
            let combined_mask: Tile<bool, { [BR, BC] }> = causal_mask & bounds_mask;
            qk = qk + select(combined_mask, mask_add_true, mask_add_false);

            // Online softmax update.
            let qk_max_1d: Tile<f32, { [BR] }> = reduce_max(qk, 1);
            let qk_max: Tile<f32, { [BR, 1] }> = qk_max_1d.reshape(shape![BR, 1]);
            let m_ij: Tile<f32, { [BR, 1] }> = max_tile(m_i, qk_max);

            // Correction factor for previously accumulated rows.
            let alpha: Tile<f32, { [BR, 1] }> = exp2(m_i - m_ij, ftz::Disabled);

            // P = exp2(S - m_new)  [BR x BC]
            let qk_shifted: Tile<f32, { [BR, BC] }> = qk - m_ij.broadcast(shape![BR, BC]);
            let p: Tile<f32, { [BR, BC] }> = exp2(qk_shifted, ftz::Disabled);

            // Update running sum and output accumulator.
            let l_ij_1d: Tile<f32, { [BR] }> = reduce_sum(p, 1);
            let l_ij: Tile<f32, { [BR, 1] }> = l_ij_1d.reshape(shape![BR, 1]);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(shape![BR, D]);

            let tile_v: Tile<f32, { [BC, D] }> =
                part_v.load_pipelined::<4>([kv_block_base + j, 0i32]);
            acc = mma(p, tile_v, acc);
            m_i = m_ij;
        }

        // Normalize: divide each row by its softmax denominator.
        out.store(true_div(acc, l_i.broadcast(shape![BR, D])));
    }

    /// De-interleave head rows for one tensor: [len, heads, D] -> [heads, rows_per_head, D].
    ///
    /// `out` is the head-major destination, rank-2 [heads * rows_per_head, D],
    /// partitioned [1, D] -> grid heads * rows_per_head.  `src` is the flat
    /// interleaved source, rank-1 [len * heads * D].  Program blk maps to
    /// (h = blk / rows_per_head, i = blk % rows_per_head) and copies src row
    /// (i, h) -- block i * heads + h of the [D]-partition -- to out row blk.
    /// Rows i >= len are zero-filled (Q padding for the flash kernel; the
    /// causal/bounds masks make K/V padding inert either way).
    ///
    /// Same indexing scheme as `cutile_decode_attn`'s cache reads (computed
    /// block indices into a rank-1 partition), plus the conditional-store
    /// pattern of `cutile_interleave`.
    #[cutile::entry()]
    pub fn cutile_gather_heads<const D: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        src: &Tensor<f32, { [-1] }>,
        src_len: i32,
        src_heads: i32,
        rows_per_head: i32,
    ) {
        let blk = program_id(0);
        let h = blk / rows_per_head;
        let i = blk % rows_per_head;
        let part_src = src.partition(shape![D]);
        if i < src_len {
            let t: Tile<f32, { [D] }> = part_src.load([i * src_heads + h]);
            out.store(t.reshape(shape![1, D]));
        } else {
            let z: Tile<f32, { [1, D] }> = constant(0.0f32, shape![1, D]);
            out.store(z);
        }
    }

    /// Re-interleave head rows: [heads, rows_per_head, D] -> [len, heads, D].
    ///
    /// `out` is the interleaved destination, rank-2 [len, heads * D],
    /// partitioned [1, D] -> grid (len, heads): program_id(0) = token row i,
    /// program_id(1) = head h.  `src` is the head-major flash-output buffer,
    /// rank-2 [heads * rows_per_head, D]; block (h * rows_per_head + i, 0)
    /// holds the attention output for (i, h).  Pad rows (i >= len) are never
    /// launched, so no bounds masking is needed.
    #[cutile::entry()]
    pub fn cutile_scatter_heads<const D: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        src: &Tensor<f32, { [-1, D] }>,
        rows_per_head: i32,
    ) {
        let i = program_id(0);
        let h = program_id(1);
        let part_src = src.partition(shape![1, D]);
        let t: Tile<f32, { [1, D] }> = part_src.load([h * rows_per_head + i, 0i32]);
        out.store(t);
    }

    /// Decode (seq_len=1) GQA attention -- GPU-resident interleaved KV cache.
    ///
    /// One program per attention head.  Reads Q, K, and V entirely on device
    /// from the flat interleaved GPU KV cache; no PCIe traffic for K or V.
    ///
    /// Cache layout (both K and V):
    ///   flat [gpu_max_seq_len * kv_dim], interleaved [token, kv_head, head_dim]
    ///   Partition by HEAD_DIM -> block index (t * N_KV_HEADS + kv_h) for token t, kv-head kv_h.
    ///
    /// Online softmax uses the exp2 trick (same as `cutile_flash_attn_head`):
    ///   score_exp2 = Q*K * (scale / log2)
    ///   exp2(score_exp2) = exp(Q*K * scale)
    ///
    /// Grid: `n_heads` programs (one per attention head, launched by partition size).
    #[cutile::entry()]
    pub fn cutile_decode_attn<const N_KV_HEADS: i32, const HEAD_DIM: i32>(
        out: &mut Tensor<f32, { [HEAD_DIM] }>,
        q: &Tensor<f32, { [-1] }>,
        k_cache: &Tensor<f32, { [-1] }>,
        v_cache: &Tensor<f32, { [-1] }>,
        kv_len: i32,
        scale: f32,
        heads_per_group: i32,
    ) {
        let h = program_id(0);
        let kv_h = h / heads_per_group;

        // Load Q for this attention head: q[h * HEAD_DIM .. (h+1) * HEAD_DIM]
        let q_tile: Tile<f32, { [HEAD_DIM] }> = q.partition(shape![HEAD_DIM]).load([h]);

        // Online softmax running statistics (exp2 trick, matching cutile_flash_attn_head).
        let mut m: Tile<f32, { [1] }> = constant(f32::NEG_INFINITY, shape![1]);
        let mut l: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);
        let mut acc: Tile<f32, { [HEAD_DIM] }> = constant(0.0f32, shape![HEAD_DIM]);

        // Precompute scale / log2 once (broadcast_scalar accepts runtime f32 values).
        let two: Tile<f32, { [] }> = constant(2.0f32, shape![]);
        let log2_val: f32 = tile_to_scalar(log(two));
        let scale_log2: Tile<f32, { [1] }> = broadcast_scalar(scale / log2_val, shape![1]);

        let part_k = k_cache.partition(shape![HEAD_DIM]);
        let part_v = v_cache.partition(shape![HEAD_DIM]);

        for t in 0i32..kv_len {
            // Cache block index for (token t, kv-head kv_h): t * N_KV_HEADS + kv_h
            let cache_blk: i32 = t * N_KV_HEADS + kv_h;

            // Load K for this token/kv-head and compute dot product with Q.
            let k_tile: Tile<f32, { [HEAD_DIM] }> = part_k.load([cache_blk]);
            let qk_sum: Tile<f32, { [] }> = reduce_sum(q_tile * k_tile, 0i32);
            // Score in exp2 space: Q*K * (scale / log2)
            let score: Tile<f32, { [1] }> = qk_sum.reshape(shape![1]) * scale_log2;

            // Online softmax update (same algebra as cutile_flash_attn_head).
            let m_new: Tile<f32, { [1] }> = max_tile(m, score);
            let alpha: Tile<f32, { [1] }> = exp2(m - m_new, ftz::Disabled);
            let p: Tile<f32, { [1] }> = exp2(score - m_new, ftz::Disabled);

            // Load V and accumulate weighted sum.
            let v_tile: Tile<f32, { [HEAD_DIM] }> = part_v.load([cache_blk]);
            l = l * alpha + p;
            acc = acc * alpha.broadcast(shape![HEAD_DIM])
                + v_tile * p.broadcast(shape![HEAD_DIM]);
            m = m_new;
        }

        // Normalise and write attention output for head h.
        out.store(true_div(acc, l.broadcast(shape![HEAD_DIM])));
    }
}

/// GPU-accelerated Flash Attention using NVIDIA cuTile Rust (`cutile-rs`).
///
/// Implements the Flash Attention v2 tiled online-softmax algorithm. Runs on
/// the stable Rust toolchain without a custom compiler backend.
pub struct CutileFlashAttention {
    pub(crate) device: Arc<Device>,
    pub(crate) stream: Arc<Stream>,
}

unsafe impl Send for CutileFlashAttention {}
unsafe impl Sync for CutileFlashAttention {}

impl CutileFlashAttention {
    /// Initialize with the default CUDA device (ordinal 0).
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    /// Initialize with a specific CUDA device ordinal.
    pub fn with_device(ordinal: usize) -> Result<Self> {
        let device = Device::new(ordinal).map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let stream = device
            .new_stream()
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Self { device, stream })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }
}

/// Gather one head's Q rows into a contiguous padded buffer.
///
/// Input Q layout: [batch, seq_len, num_heads, head_dim]
/// Output:         [seq_len_pad, head_dim_pad]   (zero-padded)
fn gather_q_head(
    q: &[f32],
    batch_idx: usize,
    head_idx: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    seq_len_pad: usize,
    head_dim_pad: usize,
) -> Vec<f32> {
    let mut buf = vec![0.0f32; seq_len_pad * head_dim_pad];
    for i in 0..seq_len {
        for d in 0..head_dim {
            let src = (batch_idx * seq_len + i) * num_heads * head_dim + head_idx * head_dim + d;
            buf[i * head_dim_pad + d] = q[src];
        }
    }
    buf
}

/// Gather one KV head's rows into a contiguous padded buffer.
///
/// Input K/V layout: [batch, kv_len, num_kv_heads, head_dim]
/// Output:           [kv_len_pad, head_dim_pad]   (zero-padded)
fn gather_kv_head(
    kv: &[f32],
    batch_idx: usize,
    kv_head_idx: usize,
    kv_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len_pad: usize,
    head_dim_pad: usize,
) -> Vec<f32> {
    let mut buf = vec![0.0f32; kv_len_pad * head_dim_pad];
    for i in 0..kv_len {
        for d in 0..head_dim {
            let src =
                (batch_idx * kv_len + i) * num_kv_heads * head_dim + kv_head_idx * head_dim + d;
            buf[i * head_dim_pad + d] = kv[src];
        }
    }
    buf
}

/// Scatter a computed head result back into the output tensor.
///
/// Output layout: [batch, seq_len, num_heads, head_dim]
fn scatter_output_head(
    result: &[f32],
    output: &mut [f32],
    batch_idx: usize,
    head_idx: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    head_dim_pad: usize,
) {
    for i in 0..seq_len {
        for d in 0..head_dim {
            let dst = (batch_idx * seq_len + i) * num_heads * head_dim + head_idx * head_dim + d;
            output[dst] = result[i * head_dim_pad + d];
        }
    }
}

impl CutileFlashAttention {
    /// Q, K, V already on device. Output stays on device.
    ///
    /// Q: [seq_len, n_heads, head_dim] on device (heads interleaved)
    /// K: [kv_len, n_kv_heads, head_dim] on device
    /// V: [kv_len, n_kv_heads, head_dim] on device
    /// Returns [seq_len, n_heads * head_dim] on device.
    ///
    /// Flash attention over dynamic activations still requires per-head host
    /// gather today (cuTile cannot slice interleaved head rows device-side).
    /// The return type lets callers chain device operations without a separate
    /// download round-trip; the primary zero-copy benefit is in
    /// `CutileMatMul::forward_bt_device_to_device` for static weight matrices.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_device(
        &self,
        q: &Arc<Tensor<f32>>,
        k: &Arc<Tensor<f32>>,
        v: &Arc<Tensor<f32>>,
        seq_len: usize,
        kv_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Arc<Tensor<f32>>> {
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(KernelError::InvalidConfig(format!(
                "n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )));
        }

        let stream = &self.stream;

        // Download activations from device (they were uploaded by the caller's
        // projection path and are already on device, but the kernel needs
        // per-head contiguous slices which require a host-side gather).
        let q_host: Vec<f32> = q
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let k_host: Vec<f32> = k
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let v_host: Vec<f32> = v
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // Run kernel via the existing host-based forward (handles gather/scatter
        // and per-head dispatch internally).
        let mut output_host = vec![0.0f32; seq_len * n_heads * head_dim];
        self.forward(
            &q_host,
            &k_host,
            &v_host,
            &mut output_host,
            1,
            seq_len,
            kv_len,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
        )?;

        // Upload result back to device so the caller avoids a redundant download.
        let out_tensor: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(output_host))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[seq_len, n_heads * head_dim])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        Ok(out_tensor)
    }
}

impl CutileFlashAttention {
    /// Batched flash attention: phases all GPU operations across heads to minimise
    /// CPU-GPU synchronisation bubbles.
    ///
    /// Instead of syncing five times per head (upload Q + upload K + upload V +
    /// kernel + download), this method:
    ///   1. Gathers all per-head host buffers on the CPU (no CUDA cost).
    ///   2. Uploads Q/K/V for all heads with `sync_on` (tensor handles required
    ///      before kernel dispatch; H2D copies are fast).
    ///   3. Allocates all output tensors with `sync_on`.
    ///   4. Submits all N head kernels with `async_on` -- NO sync between heads.
    ///   5. A single `stream.synchronize()` covers all N kernels.
    ///   6. Downloads and scatters all outputs.
    ///
    /// Reduces per-layer attention kernel syncs from N (one per head) to 1.
    /// For 20 heads x 30 layers = 600 kernel syncs down to 30.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batched(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let expected_q = batch * seq_len * n_heads * head_dim;
        let expected_kv = batch * kv_len * n_kv_heads * head_dim;
        let expected_out = batch * seq_len * n_heads * head_dim;

        if q.len() != expected_q {
            return Err(KernelError::DimensionMismatch {
                expected: expected_q,
                got: q.len(),
            });
        }
        if k.len() != expected_kv {
            return Err(KernelError::DimensionMismatch {
                expected: expected_kv,
                got: k.len(),
            });
        }
        if v.len() != expected_kv {
            return Err(KernelError::DimensionMismatch {
                expected: expected_kv,
                got: v.len(),
            });
        }
        if output.len() != expected_out {
            return Err(KernelError::DimensionMismatch {
                expected: expected_out,
                got: output.len(),
            });
        }
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(KernelError::InvalidConfig(format!(
                "n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )));
        }

        let seq_len_pad = seq_len.div_ceil(BR) * BR;
        let kv_len_pad = kv_len.div_ceil(BC) * BC;
        let head_dim_pad = head_dim;
        let heads_per_group = n_heads / n_kv_heads;
        let cache_offset = (kv_len as i32) - (seq_len as i32);
        let stream = &self.stream;

        // KVL removed: K/V tensors now use dynamic [-1, D] so the kernel compiles
        // once per (BR, BC, D) instead of once per kv_len_pad increment.
        let generics = vec![BR.to_string(), BC.to_string(), head_dim_pad.to_string()];

        for batch_idx in 0..batch {
            // -- Phase 1: host-side gather (pure CPU, zero CUDA cost) ----------
            let q_bufs: Vec<Vec<f32>> = (0..n_heads)
                .map(|h| {
                    gather_q_head(
                        q,
                        batch_idx,
                        h,
                        seq_len,
                        n_heads,
                        head_dim,
                        seq_len_pad,
                        head_dim_pad,
                    )
                })
                .collect();
            let k_bufs: Vec<Vec<f32>> = (0..n_kv_heads)
                .map(|h| {
                    gather_kv_head(
                        k,
                        batch_idx,
                        h,
                        kv_len,
                        n_kv_heads,
                        head_dim,
                        kv_len_pad,
                        head_dim_pad,
                    )
                })
                .collect();
            let v_bufs: Vec<Vec<f32>> = (0..n_kv_heads)
                .map(|h| {
                    gather_kv_head(
                        v,
                        batch_idx,
                        h,
                        kv_len,
                        n_kv_heads,
                        head_dim,
                        kv_len_pad,
                        head_dim_pad,
                    )
                })
                .collect();

            // -- Phase 2: upload all Q head tensors ----------------------------
            // sync_on is required here to obtain the Arc<Tensor<f32>> handle for
            // each head before launching the corresponding kernel.  H2D copies
            // are fast (PCIe bandwidth, not compute-bound).
            let mut q_devs: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(n_heads);
            for buf in &q_bufs {
                let dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(buf.clone()))
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .reshape(&[seq_len_pad, head_dim_pad])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .into();
                q_devs.push(dev);
            }

            // -- Phase 3: upload all K kv-head tensors -------------------------
            let mut k_devs: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(n_kv_heads);
            for buf in &k_bufs {
                let dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(buf.clone()))
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .reshape(&[kv_len_pad, head_dim_pad])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .into();
                k_devs.push(dev);
            }

            // -- Phase 4: upload all V kv-head tensors -------------------------
            let mut v_devs: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(n_kv_heads);
            for buf in &v_bufs {
                let dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(buf.clone()))
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .reshape(&[kv_len_pad, head_dim_pad])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .into();
                v_devs.push(dev);
            }

            // -- Phase 5: allocate output tensors for all heads -----------------
            // -- Phase 6: submit all head kernels -- NO sync between heads --------
            // All N kernel launches go onto the stream without blocking the CPU.
            // The GPU executes them serially (single stream); the CPU immediately
            // proceeds to the next head's launch.
            //
            // SAFETY: stream.synchronize() in Phase 7 ensures all GPU writes
            // complete before any CPU read of out_parts in Phase 8.
            let mut out_parts = Vec::with_capacity(n_heads);
            for head in 0..n_heads {
                let kv_head = head / heads_per_group;

                let out_part = api::zeros::<f32>(&[seq_len_pad, head_dim_pad])
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .partition([BR, head_dim_pad]);

                let (out_part, ..) = unsafe {
                    kernels::cutile_flash_attn_head(
                        out_part,
                        q_devs[head].clone(),
                        k_devs[kv_head].clone(),
                        v_devs[kv_head].clone(),
                        scale,
                        cache_offset,
                        seq_len as i32,
                        kv_len as i32,
                    )
                    .generics(generics.clone())
                    .async_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                };

                out_parts.push(out_part);
            }

            // -- Phase 7: single synchronisation for all N head kernels ---------
            // SAFETY: all async_on calls above queued on this stream; synchronize
            // ensures all GPU writes to out_parts are visible before CPU reads.
            unsafe {
                stream
                    .synchronize()
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            }

            // -- Phase 8: download and scatter all head outputs -----------------
            // Stream is idle; sync_on here is a fast D2H copy with no GPU wait.
            for (head, out_part) in out_parts.into_iter().enumerate() {
                let host_result: Vec<f32> = out_part
                    .unpartition()
                    .to_host_vec()
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

                scatter_output_head(
                    &host_result,
                    output,
                    batch_idx,
                    head,
                    seq_len,
                    n_heads,
                    head_dim,
                    head_dim_pad,
                );
            }
        }

        Ok(())
    }
}

impl CutileFlashAttention {
    /// Flash attention where K and V are already resident on device (per-head format).
    ///
    /// `k_devs[kv_h]` is a flat GPU tensor of `gpu_max_seq_len * head_dim` elements
    /// holding the K history for KV-head `kv_h` (written by `CutileKvCache::append_device`).
    /// `v_devs[kv_h]` is the corresponding V history.
    ///
    /// Only Q is supplied as a host slice (downloaded once from device by the caller).
    /// This eliminates the `n_kv_heads x 2` H2D uploads that `forward_batched` performs
    /// when the KV cache lives on the host, saving `2 * n_kv_heads` CUDA sync calls
    /// per layer and eliminating the O(kv_len) PCIe traffic for K/V.
    ///
    /// The method follows the same phased structure as `forward_batched`:
    ///   1. Gather Q per-head on the CPU (no CUDA cost).
    ///   2. Upload Q per-head with `sync_on` (H2D; K/V skipped -- already on device).
    ///   3. Submit all head kernels with `async_on` (no sync between heads).
    ///   4. Single `stream.synchronize()`.
    ///   5. Download and scatter outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batched_device_kv(
        &self,
        q: &[f32],
        k_devs: &[Arc<Tensor<f32>>],
        v_devs: &[Arc<Tensor<f32>>],
        output: &mut [f32],
        seq_len: usize,
        kv_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        if k_devs.len() != n_kv_heads || v_devs.len() != n_kv_heads {
            return Err(KernelError::InvalidConfig(format!(
                "k_devs/v_devs length ({}/{}) != n_kv_heads ({n_kv_heads})",
                k_devs.len(),
                v_devs.len()
            )));
        }
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(KernelError::InvalidConfig(format!(
                "n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )));
        }

        let seq_len_pad = seq_len.div_ceil(BR) * BR;
        let kv_len_pad = kv_len.div_ceil(BC) * BC;
        let head_dim_pad = head_dim;
        let heads_per_group = n_heads / n_kv_heads;
        let cache_offset = (kv_len as i32) - (seq_len as i32);
        let stream = &self.stream;
        let generics = vec![BR.to_string(), BC.to_string(), head_dim_pad.to_string()];

        // -- Phase 1: Q host-side gather (CPU only) ----------------------------
        let q_bufs: Vec<Vec<f32>> = (0..n_heads)
            .map(|h| {
                gather_q_head(
                    q,
                    0, // single batch
                    h,
                    seq_len,
                    n_heads,
                    head_dim,
                    seq_len_pad,
                    head_dim_pad,
                )
            })
            .collect();

        // -- Phase 2: reshape K/V device tensors to [kv_len_pad, head_dim] -----
        // The GPU KV cache tensors hold [gpu_max_seq * head_dim] contiguous
        // elements.  We reshape to [gpu_max_seq, head_dim] and rely on the flash
        // attention kernel's runtime `kv_len` parameter to iterate only the valid
        // prefix. The reshape is a zero-copy metadata operation.
        //
        // We cannot reshape to [kv_len_pad, head_dim] because kv_len_pad may
        // exceed the tensor's actual element count (gpu_max_seq_len * head_dim
        // > kv_len_pad is not guaranteed). Using the raw tensor and letting the
        // kernel's `kv_len` bound the loop is safe and avoids the mismatch.
        //
        // No H2D for K or V -- they are already on device.

        // -- Phase 3: upload Q per-head with sync_on ---------------------------
        let mut q_devs: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(n_heads);
        for buf in &q_bufs {
            let dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(buf.clone()))
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .reshape(&[seq_len_pad, head_dim_pad])
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .into();
            q_devs.push(dev);
        }

        // -- Phase 4: allocate output tensors; submit kernels async -------------
        // Pre-reshape K/V per-head once outside the loop.
        // Each GPU KV cache tensor holds [gpu_max_seq_len * head_dim] elements.
        // Reshape to [gpu_max_seq_len, head_dim] so the kernel's dynamic [-1, D]
        // shape accepts it. The runtime `kv_len` parameter bounds tile iteration
        // so only the first `kv_len` tokens are read regardless of gpu_max_seq_len.
        // DO NOT reshape to [kv_len_pad, head_dim] -- that would fail when
        // gpu_max_seq_len * head_dim ??? kv_len_pad * head_dim.
        // Reshape K/V per-KV-head from flat [gpu_max_seq * head_dim] to 2-D
        // [gpu_max_seq, head_dim] once outside the head loop. reshape on
        // Arc<Tensor<f32>> returns Arc<Tensor<f32>> (zero-copy metadata).
        let gpu_max_seq_len = k_devs[0].size() / head_dim_pad;
        let mut k_devs_2d: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(n_kv_heads);
        for t in k_devs.iter() {
            let r = t
                .clone()
                .reshape(&[gpu_max_seq_len, head_dim_pad])
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            k_devs_2d.push(r);
        }
        let mut v_devs_2d: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(n_kv_heads);
        for t in v_devs.iter() {
            let r = t
                .clone()
                .reshape(&[gpu_max_seq_len, head_dim_pad])
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            v_devs_2d.push(r);
        }

        let mut out_parts = Vec::with_capacity(n_heads);
        for head in 0..n_heads {
            let kv_head = head / heads_per_group;

            let k_dev = k_devs_2d[kv_head].clone();
            let v_dev = v_devs_2d[kv_head].clone();

            let out_part = api::zeros::<f32>(&[seq_len_pad, head_dim_pad])
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .partition([BR, head_dim_pad]);

            let (out_part, ..) = unsafe {
                kernels::cutile_flash_attn_head(
                    out_part,
                    q_devs[head].clone(),
                    k_dev,
                    v_dev,
                    scale,
                    cache_offset,
                    seq_len as i32,
                    kv_len as i32,
                )
                .generics(generics.clone())
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            };

            out_parts.push(out_part);
        }

        // -- Phase 5: single sync for all head kernels -------------------------
        unsafe {
            stream
                .synchronize()
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }

        // -- Phase 6: download and scatter outputs -----------------------------
        for (head, out_part) in out_parts.into_iter().enumerate() {
            let host_result: Vec<f32> = out_part
                .unpartition()
                .to_host_vec()
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

            scatter_output_head(
                &host_result,
                output,
                0, // batch 0
                head,
                seq_len,
                n_heads,
                head_dim,
                head_dim_pad,
            );
        }

        Ok(())
    }
}

impl CutileFlashAttention {
    /// Decode (seq_len=1) GQA attention entirely on device.
    ///
    /// Reads Q from `q_dev` (already on device, flat `[n_heads * head_dim]`) and
    /// K/V from the GPU-resident interleaved cache tensors `k_cache` / `v_cache`
    /// (flat `[gpu_max_seq_len * kv_dim]`).  No PCIe traffic for K or V.
    ///
    /// Returns a flat `[n_heads * head_dim]` device tensor.  Callers that need
    /// `[1, q_pad]` for the O-projection must pad via D->D copy.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `n_heads` is not divisible by `n_kv_heads` or if any
    /// cuTile / CUDA call fails.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_attention_device(
        &self,
        q_dev: &Arc<Tensor<f32>>,
        k_cache: &Arc<Tensor<f32>>,
        v_cache: &Arc<Tensor<f32>>,
        kv_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        stream: &Arc<Stream>,
    ) -> Result<Arc<Tensor<f32>>> {
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(KernelError::InvalidConfig(format!(
                "n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )));
        }
        let q_dim = n_heads * head_dim;
        let heads_per_group = (n_heads / n_kv_heads) as i32;
        let generics = vec![n_kv_heads.to_string(), head_dim.to_string()];

        // Allocate output: exactly n_heads blocks of HEAD_DIM -- no padding.
        let out_part = api::zeros::<f32>(&[q_dim])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([head_dim]);

        let (out_part, ..) = unsafe {
            kernels::cutile_decode_attn(
                out_part,
                q_dev.clone(),
                k_cache.clone(),
                v_cache.clone(),
                kv_len as i32,
                scale,
                heads_per_group,
            )
            .generics(generics)
            .async_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };

        // Kernel is queued async on the stream.  GPU-side consumers on the same
        // stream execute in order.  Caller is responsible for synchronising before
        // any CPU read of the output tensor.
        Ok(Arc::new(out_part.unpartition()))
    }

    /// Decode attention reusing a pre-allocated output buffer.
    ///
    /// Identical to `decode_attention_device` but eliminates the `api::zeros.sync_on`
    /// allocation (which is a full stream synchronize + cudaMallocAsync) by reusing
    /// a caller-supplied `scratch` buffer.  Tries `Arc::try_unwrap` to reclaim the
    /// previous result; falls back to a fresh allocation on a pool miss.
    ///
    /// # Safety
    ///
    /// The returned tensor is being written asynchronously.  Caller MUST synchronise
    /// the stream (or rely on a downstream D->H `sync_on`) before reading on the CPU.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_attention_device_scratch(
        &self,
        q_dev: &Arc<Tensor<f32>>,
        k_cache: &Arc<Tensor<f32>>,
        v_cache: &Arc<Tensor<f32>>,
        kv_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        stream: &Arc<Stream>,
        scratch: &mut Option<Arc<Tensor<f32>>>,
    ) -> Result<Arc<Tensor<f32>>> {
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(KernelError::InvalidConfig(format!(
                "n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )));
        }
        let q_dim = n_heads * head_dim;
        let heads_per_group = (n_heads / n_kv_heads) as i32;
        let generics = vec![n_kv_heads.to_string(), head_dim.to_string()];

        // Reclaim or allocate output buffer.
        let out_tensor = match scratch.take() {
            Some(arc) => match Arc::try_unwrap(arc) {
                Ok(tensor) => tensor
                    .reshape(&[q_dim])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
                Err(arc_back) => {
                    *scratch = Some(arc_back);
                    api::zeros::<f32>(&[q_dim])
                        .sync_on(stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                }
            },
            None => api::zeros::<f32>(&[q_dim])
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        };

        let out_part = out_tensor.partition([head_dim]);

        let (out_part, ..) = unsafe {
            kernels::cutile_decode_attn(
                out_part,
                q_dev.clone(),
                k_cache.clone(),
                v_cache.clone(),
                kv_len as i32,
                scale,
                heads_per_group,
            )
            .generics(generics)
            .async_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };

        let result = Arc::new(out_part.unpartition());
        *scratch = Some(result.clone());
        Ok(result)
    }

    /// On-device prefill (seq_len > 1) attention over the GPU-resident KV cache.
    ///
    /// Replaces the host-staged prefill path (Q D->H download, K/V reads from
    /// the host mirror, per-head H2D uploads, per-head D2H downloads, output
    /// H->D re-upload) with five async device kernels per layer:
    ///
    ///   1-3. `cutile_gather_heads` x3 -- de-interleave Q (from `q_dev`) and
    ///        K/V (from the flat GPU cache) into head-major padded buffers.
    ///   4.   `cutile_flash_attn_heads` -- ONE launch for all heads (the
    ///        audited `cutile_flash_attn_head` math, multi-head tiling).
    ///   5.   `cutile_scatter_heads` -- re-interleave into [seq_len, q_dim].
    ///
    /// Zero PCIe traffic, zero host synchronisations, zero steady-state
    /// allocations (all buffers pooled in `scratch`, RopeScratch-style:
    /// take -> try_unwrap -> partition, restore after the launch).
    ///
    /// `q_dev`: flat rank-1 [seq_len * n_heads * head_dim], interleaved
    /// [token, head, dim] (RoPE output).  `k_cache`/`v_cache`: flat rank-1
    /// GPU-cache tensors (any length >= kv_len * n_kv_heads * head_dim;
    /// `CutileKvCache::kv_device_tensors` returns the full-capacity buffers).
    ///
    /// Returns rank-2 [seq_len, n_heads * head_dim] on device -- feed to
    /// `attn_out_to_oproj_device_m`.
    ///
    /// # Safety
    ///
    /// The returned tensor is written asynchronously; callers must rely on
    /// stream ordering (or sync) before any CPU read.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_attention_device(
        &self,
        q_dev: &Arc<Tensor<f32>>,
        k_cache: &Arc<Tensor<f32>>,
        v_cache: &Arc<Tensor<f32>>,
        seq_len: usize,
        kv_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        stream: &Arc<Stream>,
        scratch: &mut PrefillAttnScratch,
    ) -> Result<Arc<Tensor<f32>>> {
        if seq_len == 0 || kv_len == 0 {
            return Err(KernelError::InvalidConfig(format!(
                "prefill_attention_device: empty seq_len/kv_len ({seq_len}/{kv_len})"
            )));
        }
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(KernelError::InvalidConfig(format!(
                "n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )));
        }
        let expected_q = seq_len * n_heads * head_dim;
        let got_q = q_dev.shape().iter().map(|&d| d as usize).product::<usize>();
        if got_q < expected_q {
            return Err(KernelError::DimensionMismatch {
                expected: expected_q,
                got: got_q,
            });
        }

        let seq_pad = seq_len.div_ceil(BR) * BR;
        let kv_pad = kv_len.div_ceil(BC) * BC;
        let q_rows = n_heads * seq_pad;
        let kv_rows = n_kv_heads * kv_pad;
        let q_dim = n_heads * head_dim;
        let heads_per_group = n_heads / n_kv_heads;

        // Head-major scratch buffers, exact logical size.  Pool hits (same
        // shape as the previous layer/chunk) avoid api::zeros.sync_on; only
        // k_g/v_g resize as kv_len grows across prefill chunks.
        let q_g = take_or_alloc(&mut scratch.q_gathered, &[q_rows, head_dim], stream)?;
        let k_g = take_or_alloc(&mut scratch.k_gathered, &[kv_rows, head_dim], stream)?;
        let v_g = take_or_alloc(&mut scratch.v_gathered, &[kv_rows, head_dim], stream)?;
        let out_g = take_or_alloc(&mut scratch.out_gathered, &[q_rows, head_dim], stream)?;
        let out_flat = take_or_alloc(&mut scratch.out_flat, &[seq_len, q_dim], stream)?;

        // -- 1-3. gathers: interleaved -> head-major (async, stream-ordered) --
        let gather = |out_t: Tensor<f32>,
                      src: &Arc<Tensor<f32>>,
                      src_len: usize,
                      src_heads: usize,
                      rows_per_head: usize|
         -> Result<Arc<Tensor<f32>>> {
            let (part, ..) = unsafe {
                kernels::cutile_gather_heads(
                    out_t.partition([1, head_dim]),
                    src.clone(),
                    src_len as i32,
                    src_heads as i32,
                    rows_per_head as i32,
                )
                .generics(vec![head_dim.to_string()])
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            };
            Ok(Arc::new(part.unpartition()))
        };
        let q_g = gather(q_g, q_dev, seq_len, n_heads, seq_pad)?;
        let k_g = gather(k_g, k_cache, kv_len, n_kv_heads, kv_pad)?;
        let v_g = gather(v_g, v_cache, kv_len, n_kv_heads, kv_pad)?;
        scratch.q_gathered = Some(q_g.clone());
        scratch.k_gathered = Some(k_g.clone());
        scratch.v_gathered = Some(v_g.clone());

        // -- 4. flash attention: one launch for all heads ---------------------
        // cache_offset = kv_len - seq_len (tokens already cached pre-prompt).
        let cache_offset = (kv_len - seq_len) as i32;
        let (out_part, ..) = unsafe {
            kernels::cutile_flash_attn_heads(
                out_g.partition([BR, head_dim]),
                q_g.clone(),
                k_g.clone(),
                v_g.clone(),
                scale,
                cache_offset,
                seq_len as i32,
                kv_len as i32,
                (seq_pad / BR) as i32,
                (kv_pad / BC) as i32,
                heads_per_group as i32,
            )
            .generics(vec![BR.to_string(), BC.to_string(), head_dim.to_string()])
            .async_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };
        let out_g = Arc::new(out_part.unpartition());
        scratch.out_gathered = Some(out_g.clone());

        // -- 5. scatter: head-major -> interleaved [seq_len, q_dim] -----------
        let (out_part, ..) = unsafe {
            kernels::cutile_scatter_heads(
                out_flat.partition([1, head_dim]),
                out_g.clone(),
                seq_pad as i32,
            )
            .generics(vec![head_dim.to_string()])
            .async_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };
        let result = Arc::new(out_part.unpartition());
        scratch.out_flat = Some(result.clone());
        Ok(result)
    }
}

/// Pooled device buffers for `CutileFlashAttention::prefill_attention_device`.
///
/// RopeScratch-style pooling: each buffer is `take`n, `Arc::try_unwrap`ped,
/// partitioned as the kernel output and restored after the launch.  q_gathered
/// / k_gathered / v_gathered / out_gathered never leave the method, so their
/// Arcs are always unique at the next call.  out_flat is shared with the
/// caller (returned); the caller drops it before the next layer, restoring
/// uniqueness -- same contract as the decode `attn_out_scratch`.
///
/// All buffers are exact-logical-size (per-call seq_pad/kv_pad); k_gathered
/// and v_gathered grow with kv_len across prefill chunks, reallocating only
/// on the (rare) size change.
#[derive(Default)]
pub struct PrefillAttnScratch {
    q_gathered: Option<Arc<Tensor<f32>>>,
    k_gathered: Option<Arc<Tensor<f32>>>,
    v_gathered: Option<Arc<Tensor<f32>>>,
    out_gathered: Option<Arc<Tensor<f32>>>,
    out_flat: Option<Arc<Tensor<f32>>>,
}

unsafe impl Send for PrefillAttnScratch {}
unsafe impl Sync for PrefillAttnScratch {}

/// Reclaim a pooled buffer with the requested shape, or allocate a fresh one.
///
/// Pool miss (wrong shape, extra Arc owners, first call) falls back to
/// `api::zeros(...).sync_on(...)` -- a blocking alloc that only happens on
/// size changes, never in the steady state.
fn take_or_alloc(
    pool: &mut Option<Arc<Tensor<f32>>>,
    shape: &[usize],
    stream: &Arc<Stream>,
) -> Result<Tensor<f32>> {
    let want: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    if let Some(t) = pool
        .take()
        .and_then(|arc| Arc::try_unwrap(arc).ok())
        .filter(|t| t.shape() == want.as_slice())
    {
        return Ok(t);
    }
    Ok(api::zeros::<f32>(shape)
        .sync_on(stream)
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?)
}

impl Attention for CutileFlashAttention {
    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Tested
    }

    #[allow(clippy::too_many_arguments)]
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
        // Validate dimensions.
        let expected_q = batch * seq_len * num_heads * head_dim;
        let expected_kv = batch * kv_len * num_kv_heads * head_dim;
        let expected_out = batch * seq_len * num_heads * head_dim;

        if q.len() != expected_q {
            return Err(KernelError::DimensionMismatch {
                expected: expected_q,
                got: q.len(),
            });
        }
        if k.len() != expected_kv {
            return Err(KernelError::DimensionMismatch {
                expected: expected_kv,
                got: k.len(),
            });
        }
        if v.len() != expected_kv {
            return Err(KernelError::DimensionMismatch {
                expected: expected_kv,
                got: v.len(),
            });
        }
        if output.len() != expected_out {
            return Err(KernelError::DimensionMismatch {
                expected: expected_out,
                got: output.len(),
            });
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(KernelError::InvalidConfig(format!(
                "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )));
        }

        // Pad dimensions to tile multiples.
        let seq_len_pad = seq_len.div_ceil(BR) * BR;
        let kv_len_pad = kv_len.div_ceil(BC) * BC;
        // head_dim is a JIT const so it doesn't need padding beyond what the kernel handles.
        let head_dim_pad = head_dim; // passed as const D; caller must ensure divisibility if needed

        let heads_per_group = num_heads / num_kv_heads;
        let cache_offset = (kv_len as i32) - (seq_len as i32);
        let stream = &self.stream;

        // Build generics string for the JIT: BR, BC, D, KVL.
        // KVL removed: K/V tensors now use dynamic [-1, D] so the kernel compiles
        // once per (BR, BC, D) instead of once per kv_len_pad increment.
        let generics = vec![BR.to_string(), BC.to_string(), head_dim_pad.to_string()];

        // Process each (batch, head) independently.
        for batch_idx in 0..batch {
            for head_idx in 0..num_heads {
                let kv_head_idx = head_idx / heads_per_group;

                // Gather host-side contiguous slices.
                let q_buf = gather_q_head(
                    q,
                    batch_idx,
                    head_idx,
                    seq_len,
                    num_heads,
                    head_dim,
                    seq_len_pad,
                    head_dim_pad,
                );
                let k_buf = gather_kv_head(
                    k,
                    batch_idx,
                    kv_head_idx,
                    kv_len,
                    num_kv_heads,
                    head_dim,
                    kv_len_pad,
                    head_dim_pad,
                );
                let v_buf = gather_kv_head(
                    v,
                    batch_idx,
                    kv_head_idx,
                    kv_len,
                    num_kv_heads,
                    head_dim,
                    kv_len_pad,
                    head_dim_pad,
                );

                // Upload tensors.
                let q_arc: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(q_buf))
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .reshape(&[seq_len_pad, head_dim_pad])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .into();

                let k_arc: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(k_buf))
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .reshape(&[kv_len_pad, head_dim_pad])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .into();

                let v_arc: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(v_buf))
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .reshape(&[kv_len_pad, head_dim_pad])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .into();

                // Allocate and partition output tile (one program per Q tile row-block).
                let out_part = api::zeros::<f32>(&[seq_len_pad, head_dim_pad])
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .partition([BR, head_dim_pad]);

                let (out_part, _q, _k, _v, _scale, _co, _sl, _kl) =
                    kernels::cutile_flash_attn_head(
                        out_part,
                        q_arc,
                        k_arc,
                        v_arc,
                        scale,
                        cache_offset,
                        seq_len as i32,
                        kv_len as i32,
                    )
                    .generics(generics.clone())
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

                // Download result.
                let host_result: Vec<f32> = out_part
                    .unpartition()
                    .to_host_vec()
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

                // Scatter back to output tensor.
                scatter_output_head(
                    &host_result,
                    output,
                    batch_idx,
                    head_idx,
                    seq_len,
                    num_heads,
                    head_dim,
                    head_dim_pad,
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::CpuAttention;

    fn run_flash_attn_vs_cpu(
        batch: usize,
        seq: usize,
        kv: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) {
        let scale = 1.0 / (hd as f32).sqrt();
        let n = batch * seq * nh * hd;
        let nkv_total = batch * kv * nkv * hd;

        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let k: Vec<f32> = (0..nkv_total).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let v: Vec<f32> = (0..nkv_total).map(|i| (i as f32) * 0.005 - 0.2).collect();

        let mut out_cpu = vec![0.0f32; n];
        let cpu = CpuAttention;
        cpu.forward(&q, &k, &v, &mut out_cpu, batch, seq, kv, nh, nkv, hd, scale)
            .expect("CPU attention failed");

        let mut out_gpu = vec![0.0f32; n];
        let gpu = CutileFlashAttention::new().expect("CutileFlashAttention::new failed");
        gpu.forward(&q, &k, &v, &mut out_gpu, batch, seq, kv, nh, nkv, hd, scale)
            .expect("GPU attention failed");

        for (i, (a, b)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "Mismatch at index {i}: cpu={a:.6}, gpu={b:.6}"
            );
        }
    }

    #[test]
    fn test_cutile_flash_attn_basic() {
        run_flash_attn_vs_cpu(1, 32, 32, 2, 2, 64);
    }

    #[test]
    fn test_cutile_flash_attn_gqa() {
        run_flash_attn_vs_cpu(1, 32, 32, 4, 2, 64);
    }

    #[test]
    fn test_cutile_flash_attn_kv_cache() {
        run_flash_attn_vs_cpu(1, 32, 64, 2, 2, 64);
    }

    #[test]
    fn test_cutile_flash_attn_batch() {
        run_flash_attn_vs_cpu(2, 32, 32, 2, 2, 64);
    }

    /// Verify `forward_batched_device_kv` matches `forward` (CPU reference).
    ///
    /// Builds per-KV-head GPU tensors from the same host K/V data used by the
    /// CPU reference, runs `forward_batched_device_kv`, and checks that every
    /// output element is within 1e-3 of the CPU result.
    fn run_device_kv_vs_cpu(seq: usize, kv: usize, nh: usize, nkv: usize, hd: usize) {
        use cutile::api;
        use cutile::prelude::DeviceOp;

        let scale = 1.0 / (hd as f32).sqrt();
        let nq = seq * nh * hd;
        let nkv_total = kv * nkv * hd;

        let q: Vec<f32> = (0..nq).map(|i| (i as f32) * 0.01 - 0.5).collect();
        // K/V in interleaved [kv, nkv, hd] layout.
        let k: Vec<f32> = (0..nkv_total).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let v: Vec<f32> = (0..nkv_total).map(|i| (i as f32) * 0.005 - 0.2).collect();

        let mut out_cpu = vec![0.0f32; nq];
        let cpu = crate::attention::CpuAttention;
        cpu.forward(&q, &k, &v, &mut out_cpu, 1, seq, kv, nh, nkv, hd, scale)
            .expect("CPU attention failed");

        let gpu = CutileFlashAttention::new().expect("CutileFlashAttention::new failed");
        let stream = gpu.stream();

        // Build per-KV-head device tensors in contiguous [kv, hd] layout.
        let mut k_devs: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(nkv);
        let mut v_devs: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(nkv);
        for kv_h in 0..nkv {
            // Gather head kv_h: stride through [kv, nkv, hd].
            let mut k_head = vec![0.0f32; kv * hd];
            let mut v_head = vec![0.0f32; kv * hd];
            for t in 0..kv {
                let src = t * nkv * hd + kv_h * hd;
                k_head[t * hd..(t + 1) * hd].copy_from_slice(&k[src..src + hd]);
                v_head[t * hd..(t + 1) * hd].copy_from_slice(&v[src..src + hd]);
            }
            let k_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(k_head))
                .sync_on(stream)
                .expect("k_t upload")
                .into();
            let v_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(v_head))
                .sync_on(stream)
                .expect("v_t upload")
                .into();
            k_devs.push(k_t);
            v_devs.push(v_t);
        }

        let mut out_gpu_dkv = vec![0.0f32; nq];
        gpu.forward_batched_device_kv(
            &q,
            &k_devs,
            &v_devs,
            &mut out_gpu_dkv,
            seq,
            kv,
            nh,
            nkv,
            hd,
            scale,
        )
        .expect("forward_batched_device_kv failed");

        for (i, (a, b)) in out_cpu.iter().zip(out_gpu_dkv.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "device_kv mismatch at {i}: cpu={a:.6}, gpu={b:.6}"
            );
        }
    }

    #[test]
    fn test_forward_batched_device_kv_basic() {
        run_device_kv_vs_cpu(1, 32, 2, 2, 64);
    }

    #[test]
    fn test_forward_batched_device_kv_gqa() {
        run_device_kv_vs_cpu(1, 64, 4, 2, 64);
    }

    #[test]
    fn test_forward_batched_device_kv_decode_long() {
        run_device_kv_vs_cpu(1, 128, 4, 2, 64);
    }
}
