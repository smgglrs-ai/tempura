use crate::{Attention, KernelError, Result, VerificationTier};
use cuda_core::{Device, Stream};
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Reshape, Tensor};
use std::sync::Arc;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// Flash multi-head attention with online softmax.
    ///
    /// Processes one head at a time: Q has shape [1, 1, seq_pad, D],
    /// K/V have shape [1, 1, kv_pad, D] (already GQA-selected on host).
    ///
    /// Grid: (1, ceil(seq_pad / BM), 1) -- one program per query tile.
    ///
    /// Template parameters:
    ///   BM -- query tile height (rows of Q processed per program)
    ///   BN -- key-value tile height (KV positions per inner tile)
    ///   D  -- head dimension (must equal actual head_dim)
    ///   CAUSAL -- 1 to apply causal masking, 0 for full attention
    #[cutile::entry(
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 1, max_divisibility = 16,),
        )
    )]
    pub fn attention_fwd<const BM: i32, const BN: i32, const D: i32, const CAUSAL: i32>(
        out: &mut Tensor<f32, { [1, BM, D] }>,
        q: &Tensor<f32, { [-1, -1, -1, -1] }>,
        k: &Tensor<f32, { [-1, -1, -1, -1] }>,
        v: &Tensor<f32, { [-1, -1, -1, -1] }>,
        qk_scale: f32,
        input_pos: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        // batch_idx = 0, head_idx = 0 -- host already selected the right head
        let q_m_idx = pid.1;

        // Convert scale to log2 space so we can use exp2 (GPU fast path)
        let two: Tile<f32, { [] }> = constant(2.0f32, shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale_tile: Tile<f32, { [BM, BN] }> =
            broadcast_scalar(qk_scale / log2, shape![BM, BN]);

        // Online softmax accumulators
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, shape![BM, 1]);
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);

        // Load the Q tile for this program: slice [0, 0, q_m_idx, 0]
        let q_part = q.partition(shape![1, 1, BM, D]);
        let tq: Tile<f32, { [BM, D] }> = q_part
            .load([0i32, 0i32, q_m_idx, 0i32])
            .reshape(shape![BM, D]);

        // KV sequence length from the runtime tensor shape
        let kv_seqlen: i32 = get_shape_dim(k.shape(), 2i32);

        // How many KV tiles to visit (causal -> only up to the last visible KV position)
        let m_end: i32 = input_pos + (q_m_idx + 1i32) * BM;
        let mut mask_start: i32 = kv_seqlen / BN;
        let mut tc: i32 = ceil_div(kv_seqlen, BN);
        if CAUSAL == 1i32 {
            mask_start = (input_pos + q_m_idx * BM) / BN;
            let kv_seqlen_tiles = kv_seqlen / BN;
            mask_start = min(mask_start, kv_seqlen_tiles);
            tc = ceil_div(min(m_end, kv_seqlen), BN);
        }

        let k_part = k.partition(shape![1, 1, BN, D]);
        let v_part = v.partition(shape![1, 1, BN, D]);

        // Tile indices for causal masking: rows (query positions) and cols (KV positions)
        let offs_n_iota: Tile<i32, { [BN] }> = iota(shape![BN]);
        let offs_n_base: Tile<i32, { [BM, BN] }> =
            offs_n_iota.reshape(shape![1, BN]).broadcast(shape![BM, BN]);

        let offs_m_iota: Tile<i32, { [BM] }> = iota(shape![BM]);
        let offs_m_col: Tile<i32, { [BM, 1] }> =
            broadcast_scalar(q_m_idx * BM + input_pos, shape![BM, 1])
                + offs_m_iota.reshape(shape![BM, 1]);
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m_col.broadcast(shape![BM, BN]);
        let kv_seqlen_tile: Tile<i32, { [BM, BN] }> = kv_seqlen.broadcast(shape![BM, BN]);
        let mask_keep: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let mask_drop: Tile<f32, { [BM, BN] }> = constant(f32::NEG_INFINITY, shape![BM, BN]);

        for j in 0i32..tc {
            let k_tile: Tile<f32, { [BN, D] }> =
                k_part.load([0i32, 0i32, j, 0i32]).reshape(shape![BN, D]);
            let k_tile_t: Tile<f32, { [D, BN] }> = k_tile.transpose();

            // QK^T for this tile: [BM, BN]
            let mut qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            qk = mma(tq, k_tile_t, qk);

            // Apply causal / boundary masks when needed
            if CAUSAL == 1i32 && j >= mask_start {
                let offs_n: Tile<i32, { [BM, BN] }> =
                    broadcast_scalar(j * BN, shape![BM, BN]) + offs_n_base;
                // mask: keep (q_pos >= kv_pos) -- standard causal constraint
                let causal_ok: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
                // mask: keep (kv_pos < kv_seqlen) -- guard for padding
                let in_bounds: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_seqlen_tile);
                let mask: Tile<bool, { [BM, BN] }> = causal_ok & in_bounds;
                qk = qk + select(mask, mask_keep, mask_drop);
            }

            // Scale in log2 space
            qk = qk * qk_scale_tile;

            // Online softmax update
            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max_2d: Tile<f32, { [BM, 1] }> = qk_max.reshape(shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max_2d);

            // exp2(score - new_max): probabilities for this tile
            let p: Tile<f32, { [BM, BN] }> =
                exp2(qk - m_ij.broadcast(shape![BM, BN]), ftz::Disabled);

            // Correction: rescale previous sum and accumulator for new max
            let alpha: Tile<f32, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij_2d: Tile<f32, { [BM, 1] }> = l_ij.reshape(shape![BM, 1]);

            l_i = l_i * alpha + l_ij_2d;
            acc = acc * alpha.broadcast(shape![BM, D]);

            // Accumulate weighted values
            let v_tile: Tile<f32, { [BN, D] }> =
                v_part.load([0i32, 0i32, j, 0i32]).reshape(shape![BN, D]);
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }

        // Normalize and write output tile
        out.store(true_div(acc, l_i.broadcast(shape![BM, D])).reshape(shape![1, BM, D]));
    }
}

/// GPU-accelerated multi-head attention using NVIDIA cuTile Rust (`cutile-rs`).
///
/// Uses flash attention with online softmax, tiling over both query and
/// key-value positions. Handles grouped-query attention (GQA) on the host
/// by selecting the correct KV head before each kernel invocation.
pub struct CutileAttention {
    device: Arc<Device>,
    stream: Arc<Stream>,
}

unsafe impl Send for CutileAttention {}
unsafe impl Sync for CutileAttention {}

impl CutileAttention {
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

// Tile dimensions. BM and BN must be powers-of-two >= 1 and divisors of the
// padded sequence lengths. D is passed as a runtime generic to the JIT.
const BM: usize = 16;
const BN: usize = 16;

impl Attention for CutileAttention {
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
        // Validate dimensions
        if q.len() != batch * seq_len * num_heads * head_dim {
            return Err(KernelError::DimensionMismatch {
                expected: batch * seq_len * num_heads * head_dim,
                got: q.len(),
            });
        }
        if k.len() != batch * kv_len * num_kv_heads * head_dim {
            return Err(KernelError::DimensionMismatch {
                expected: batch * kv_len * num_kv_heads * head_dim,
                got: k.len(),
            });
        }
        if output.len() != batch * seq_len * num_heads * head_dim {
            return Err(KernelError::DimensionMismatch {
                expected: batch * seq_len * num_heads * head_dim,
                got: output.len(),
            });
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(KernelError::InvalidConfig(format!(
                "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )));
        }

        let heads_per_group = num_heads / num_kv_heads;
        // Cache offset: how many KV positions precede the first query token.
        let input_pos = (kv_len - seq_len) as i32;

        // Pad seq_len and kv_len to tile multiples
        let seq_pad = seq_len.div_ceil(BM) * BM;
        let kv_pad = kv_len.div_ceil(BN) * BN;

        let stream = &self.stream;

        // Process one (batch, head) pair at a time to keep kernel complexity low.
        // GQA is handled here: Q head `h` reads from KV head `h / heads_per_group`.
        for b in 0..batch {
            for h in 0..num_heads {
                let kv_h = h / heads_per_group;

                // Extract and pad the Q slice for this (batch, head): [seq_len, head_dim]
                let mut q_slice = vec![0.0f32; seq_pad * head_dim];
                for s in 0..seq_len {
                    let src = ((b * seq_len + s) * num_heads + h) * head_dim;
                    let dst = s * head_dim;
                    q_slice[dst..dst + head_dim].copy_from_slice(&q[src..src + head_dim]);
                }

                // Extract and pad K slice: [kv_len, head_dim]
                let mut k_slice = vec![0.0f32; kv_pad * head_dim];
                for s in 0..kv_len {
                    let src = ((b * kv_len + s) * num_kv_heads + kv_h) * head_dim;
                    let dst = s * head_dim;
                    k_slice[dst..dst + head_dim].copy_from_slice(&k[src..src + head_dim]);
                }

                // Extract and pad V slice: [kv_len, head_dim]
                let mut v_slice = vec![0.0f32; kv_pad * head_dim];
                for s in 0..kv_len {
                    let src = ((b * kv_len + s) * num_kv_heads + kv_h) * head_dim;
                    let dst = s * head_dim;
                    v_slice[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
                }

                // Upload tensors to device.
                // Kernel expects 4D: [1, 1, seq_or_kv, head_dim]
                let q_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(q_slice))
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .reshape(&[1, 1, seq_pad, head_dim])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .into();

                let k_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(k_slice))
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .reshape(&[1, 1, kv_pad, head_dim])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .into();

                let v_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(v_slice))
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .reshape(&[1, 1, kv_pad, head_dim])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .into();

                // Allocate output: [1, seq_pad, head_dim], partitioned as [1, BM, head_dim]
                let out_part = api::zeros::<f32>(&[1, seq_pad, head_dim])
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .partition([1, BM, head_dim]);

                // Generics: BM, BN, D (head_dim), CAUSAL=1
                let generics = vec![
                    BM.to_string(),
                    BN.to_string(),
                    head_dim.to_string(),
                    "1".to_string(), // CAUSAL
                ];

                let (out_part_result, _, _, _, _, _) =
                    kernels::attention_fwd(out_part, q_dev, k_dev, v_dev, scale, input_pos)
                        .generics(generics)
                        .sync_on(stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

                // Download result: [1, seq_pad, head_dim] -> first seq_len rows
                let host: Vec<f32> = out_part_result
                    .unpartition()
                    .to_host_vec()
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

                // Scatter back into the caller's output buffer (layout: [b, s, h, d])
                for s in 0..seq_len {
                    let src = s * head_dim;
                    let dst = ((b * seq_len + s) * num_heads + h) * head_dim;
                    output[dst..dst + head_dim].copy_from_slice(&host[src..src + head_dim]);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::CpuAttention;

    fn reference_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * seq_len * num_heads * head_dim];
        let cpu = CpuAttention;
        cpu.forward(
            q,
            k,
            v,
            &mut out,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .expect("cpu attention");
        out
    }

    #[test]
    fn cutile_attention_matches_cpu() {
        let gpu = CutileAttention::new()
            .expect("Failed to create GPU context -- is an NVIDIA GPU available?");

        let (batch, seq_len, kv_len) = (1, 4, 4);
        let (num_heads, num_kv_heads, head_dim) = (2, 2, 16);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n = batch * seq_len * num_heads * head_dim;
        let nkv = batch * kv_len * num_kv_heads * head_dim;

        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..nkv).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> = (0..nkv).map(|i| (i as f32) * 0.005).collect();

        let expected = reference_attention(
            &q,
            &k,
            &v,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        );

        let mut got = vec![0.0f32; n];
        gpu.forward(
            &q,
            &k,
            &v,
            &mut got,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .expect("cutile attention");

        for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (e - g).abs() < 1e-3,
                "mismatch at {i}: expected {e}, got {g}"
            );
        }
    }

    #[test]
    fn cutile_attention_gqa() {
        let gpu = CutileAttention::new()
            .expect("Failed to create GPU context -- is an NVIDIA GPU available?");

        let (batch, seq_len, kv_len) = (1, 2, 4);
        let (num_heads, num_kv_heads, head_dim) = (4, 2, 16);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n = batch * seq_len * num_heads * head_dim;
        let nkv = batch * kv_len * num_kv_heads * head_dim;

        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..nkv).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> = (0..nkv).map(|i| (i as f32) * 0.005).collect();

        let expected = reference_attention(
            &q,
            &k,
            &v,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        );

        let mut got = vec![0.0f32; n];
        gpu.forward(
            &q,
            &k,
            &v,
            &mut got,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .expect("cutile attention GQA");

        for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (e - g).abs() < 1e-3,
                "mismatch at {i}: expected {e}, got {g}"
            );
        }
    }
}
