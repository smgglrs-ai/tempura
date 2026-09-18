use std::sync::Arc;

use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Reshape, Tensor};
use cuda_core::{Device, Stream};

use crate::{Attention, KernelError, Result, VerificationTier};

/// Tile dimensions (compile-time constants for the JIT).
const BR: usize = 32;
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
    ///   For each Q tile q_tile (BR × D):
    ///     For each KV tile (j): k_tile (BC × D), v_tile (BC × D):
    ///       S = q_tile @ k_tile^T                  [BR × BC]
    ///       Apply causal mask (set future positions to -∞)
    ///       S *= scale / log2                       (for exp2 trick)
    ///       qk_max = rowmax(S)                      [BR × 1]
    ///       m_new = max(m_old, qk_max)
    ///       alpha = exp2(m_old - m_new)
    ///       P = exp2(S - m_new)                     [BR × BC]
    ///       l_new = alpha * l_old + rowsum(P)       [BR × 1]
    ///       acc = alpha * acc + P @ v_tile          [BR × D]
    ///     out_tile = acc / l_new
    ///
    /// Grid: (ceil(seq_len / BR), 1, 1)
    /// Generics: BR (Q tile rows), BC (KV tile rows), D (head dim), KVL (kv_len padded)
    #[cutile::entry()]
    pub fn cutile_flash_attn_head<const BR: i32, const BC: i32, const D: i32, const KVL: i32>(
        out: &mut Tensor<f32, { [BR, D] }>,
        q: &Tensor<f32, { [-1, D] }>,
        k: &Tensor<f32, { [KVL, D] }>,
        v: &Tensor<f32, { [KVL, D] }>,
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
        let num_kv_tiles: i32 = kv_len / BC;

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

            // S = Q @ K^T  [BR × BC]
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

            // P = exp2(S - m_new)  [BR × BC]
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
        let device = Device::new(ordinal)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
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
            let src = (batch_idx * kv_len + i) * num_kv_heads * head_dim
                + kv_head_idx * head_dim
                + d;
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
            return Err(KernelError::DimensionMismatch { expected: expected_q, got: q.len() });
        }
        if k.len() != expected_kv {
            return Err(KernelError::DimensionMismatch { expected: expected_kv, got: k.len() });
        }
        if v.len() != expected_kv {
            return Err(KernelError::DimensionMismatch { expected: expected_kv, got: v.len() });
        }
        if output.len() != expected_out {
            return Err(KernelError::DimensionMismatch {
                expected: expected_out,
                got: output.len(),
            });
        }
        if num_heads % num_kv_heads != 0 {
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
        let generics = vec![
            BR.to_string(),
            BC.to_string(),
            head_dim_pad.to_string(),
            kv_len_pad.to_string(),
        ];

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

    fn run_flash_attn_vs_cpu(batch: usize, seq: usize, kv: usize, nh: usize, nkv: usize, hd: usize) {
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
}
