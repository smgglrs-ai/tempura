use crate::{KernelError, Result, RoPE, VerificationTier};
use cuda_core::{Device, Stream};
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Tensor};
use std::sync::Arc;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// Core RoPE rotation kernel.
    ///
    /// Each program instance handles one (seq × head) pair, represented by
    /// `pid = program_id(0)`.  The input is pre-split into first and second
    /// halves on the host so the kernel receives two separate 1D tensors.
    /// Cosine and sine values are also pre-expanded per (seq × head) pair
    /// by the host, enabling a single unified kernel for both on-the-fly and
    /// table-based variants.
    ///
    /// Rotation:
    ///   out_first[d]  =  x_first[d] * cos[d] - x_second[d] * sin[d]
    ///   out_second[d] =  x_first[d] * sin[d] + x_second[d] * cos[d]
    ///
    /// All inputs are flattened to `[num_sh * HALF_DIM]`; each program loads
    /// its tile at `[pid]` in the partition grid.
    #[cutile::entry()]
    pub fn cutile_rope_kernel<const HALF_DIM: i32>(
        out_first: &mut Tensor<f32, { [HALF_DIM] }>,
        out_second: &mut Tensor<f32, { [HALF_DIM] }>,
        x_first: &Tensor<f32, { [-1] }>,
        x_second: &Tensor<f32, { [-1] }>,
        cos_table: &Tensor<f32, { [-1] }>,
        sin_table: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile_x0 = x_first.partition(shape![HALF_DIM]).load([pid]);
        let tile_x1 = x_second.partition(shape![HALF_DIM]).load([pid]);
        let tile_cos = cos_table.partition(shape![HALF_DIM]).load([pid]);
        let tile_sin = sin_table.partition(shape![HALF_DIM]).load([pid]);
        out_first.store(tile_x0 * tile_cos - tile_x1 * tile_sin);
        out_second.store(tile_x0 * tile_sin + tile_x1 * tile_cos);
    }

    /// Deinterleave kernel: split interleaved [num_sh × head_dim] into two half-tensors.
    ///
    /// `x` is a flat [num_sh * head_dim] tensor in interleaved layout:
    ///   x[sh * head_dim .. sh * head_dim + half_dim]       = first  half of head sh
    ///   x[sh * head_dim + half_dim .. (sh+1) * head_dim]   = second half of head sh
    ///
    /// With HALF_DIM-element tiles, head sh occupies tiles 2*sh and 2*sh+1.
    /// Each program (pid = sh) loads both tiles using the indexed load API.
    #[cutile::entry()]
    pub fn cutile_deinterleave<const HALF_DIM: i32>(
        first: &mut Tensor<f32, { [HALF_DIM] }>,
        second: &mut Tensor<f32, { [HALF_DIM] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        first.store(x.partition(shape![HALF_DIM]).load([2 * pid]));
        second.store(x.partition(shape![HALF_DIM]).load([2 * pid + 1]));
    }

    /// Interleave kernel: merge two half-tensors back into flat [num_sh * head_dim].
    ///
    /// Inverse of `cutile_deinterleave`. Launches `num_sh * 2` programs so each
    /// program writes exactly one HALF_DIM-element tile — no indexed store needed.
    ///
    ///   program 2*sh   → z tile 2*sh   = first tile sh
    ///   program 2*sh+1 → z tile 2*sh+1 = second tile sh
    ///
    /// `z` is partitioned with HALF_DIM elements per tile (total = num_sh * 2 tiles).
    /// Both `first` and `second` carry num_sh HALF_DIM-element tiles accessed by
    /// indexed load at position pid/2.
    #[cutile::entry()]
    pub fn cutile_interleave<const HALF_DIM: i32>(
        z: &mut Tensor<f32, { [HALF_DIM] }>,
        first: &Tensor<f32, { [-1] }>,
        second: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let sh = pid / 2;
        let half_idx = pid % 2;
        if half_idx == 0 {
            z.store(first.partition(shape![HALF_DIM]).load([sh]));
        } else {
            z.store(second.partition(shape![HALF_DIM]).load([sh]));
        }
    }

}

/// GPU-accelerated Rotary Position Embeddings using NVIDIA cuTile Rust (`cutile-rs`).
///
/// The host precomputes cosine and sine values per (batch × seq × head) pair
/// so the GPU kernel receives only pre-expanded tables.  This keeps the kernel
/// simple and avoids the need for per-program scalar parameters.
pub struct CutileRoPE {
    device: Arc<Device>,
    stream: Arc<Stream>,
}

unsafe impl Send for CutileRoPE {}
unsafe impl Sync for CutileRoPE {}

impl CutileRoPE {
    /// Initialise with the default CUDA device (ordinal 0).
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    /// Initialise with a specific CUDA device ordinal.
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

    /// Apply rotary embeddings via a pre-built cos/sin table.
    ///
    /// `cos_sin_table` is flat `[max_pos * half * 2]` where
    ///   `table[(pos * half + d) * 2 + 0]` = cos,
    ///   `table[(pos * half + d) * 2 + 1]` = sin.
    pub fn forward_table(
        &self,
        x: &mut [f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        positions: &[u32],
        cos_sin_table: &[f32],
    ) -> Result<()> {
        if !head_dim.is_multiple_of(2) {
            return Err(KernelError::InvalidConfig(format!(
                "head_dim must be even, got {head_dim}"
            )));
        }
        let half = head_dim / 2;
        let num_sh = batch * seq_len * num_heads;
        if x.len() != num_sh * head_dim {
            return Err(KernelError::DimensionMismatch {
                expected: num_sh * head_dim,
                got: x.len(),
            });
        }

        // Pre-expand cos/sin: [num_sh, half] — replicate same position across all heads.
        let mut cos_exp = vec![0.0f32; num_sh * half];
        let mut sin_exp = vec![0.0f32; num_sh * half];
        for b in 0..batch {
            for s in 0..seq_len {
                let pos = positions[b * seq_len + s] as usize;
                for h in 0..num_heads {
                    let sh = (b * seq_len + s) * num_heads + h;
                    for d in 0..half {
                        let ti = (pos * half + d) * 2;
                        cos_exp[sh * half + d] = cos_sin_table[ti];
                        sin_exp[sh * half + d] = cos_sin_table[ti + 1];
                    }
                }
            }
        }

        self.apply_rotation(x, num_sh, head_dim, half, cos_exp, sin_exp)
    }
}

impl RoPE for CutileRoPE {
    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Tested
    }

    /// Apply rotary position embeddings in-place (on-the-fly frequency computation).
    ///
    /// `x` must have length `batch * seq_len * num_heads * head_dim`.
    /// Frequencies are computed on the host as `cos/sin(pos / theta^(2d/head_dim))`
    /// then uploaded as pre-expanded tables for the GPU kernel.
    fn forward(
        &self,
        x: &mut [f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        positions: &[u32],
        theta: f32,
    ) -> Result<()> {
        if !head_dim.is_multiple_of(2) {
            return Err(KernelError::InvalidConfig(format!(
                "head_dim must be even, got {head_dim}"
            )));
        }
        let half = head_dim / 2;
        let num_sh = batch * seq_len * num_heads;
        if x.len() != num_sh * head_dim {
            return Err(KernelError::DimensionMismatch {
                expected: num_sh * head_dim,
                got: x.len(),
            });
        }

        // Pre-expand cos/sin: compute on host, then pass to kernel.
        let mut cos_exp = vec![0.0f32; num_sh * half];
        let mut sin_exp = vec![0.0f32; num_sh * half];
        for b in 0..batch {
            for s in 0..seq_len {
                let pos = positions[b * seq_len + s] as f32;
                for h in 0..num_heads {
                    let sh = (b * seq_len + s) * num_heads + h;
                    for d in 0..half {
                        let freq = pos / theta.powf(2.0 * d as f32 / head_dim as f32);
                        cos_exp[sh * half + d] = freq.cos();
                        sin_exp[sh * half + d] = freq.sin();
                    }
                }
            }
        }

        self.apply_rotation(x, num_sh, head_dim, half, cos_exp, sin_exp)
    }
}

impl CutileRoPE {
    /// Apply RoPE entirely on device: deinterleave x on the GPU, rotate, and
    /// return a rotated device tensor — no download of `x_dev` required.
    ///
    /// The only host-to-device transfers are the cos/sin tables, which are
    /// `seq_len × n_heads × head_dim/2` floats — tiny even for large batches.
    /// The rotation itself runs on the GPU via the existing split-half kernel.
    /// Output halves are downloaded and reconstructed on the CPU before a
    /// final upload; this intermediate step will be eliminated once cutile
    /// supports indexed device-side stores.
    ///
    /// PCIe transfers per call: 2 H→D (cos/sin) + 2 D→H (output halves) + 1 H→D (result)
    /// vs. the host-side path: 1 D→H (download x) + 4 H→D + 2 D→H + 1 H→D = 8 transfers.
    pub fn forward_device(
        &self,
        x_dev: &Arc<Tensor<f32>>,
        positions: &[u32],
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<Arc<Tensor<f32>>> {
        if !head_dim.is_multiple_of(2) {
            return Err(KernelError::InvalidConfig(format!(
                "head_dim must be even, got {head_dim}"
            )));
        }
        let half = head_dim / 2;
        let num_sh = seq_len * n_heads;

        // ── 1. Compute cos/sin tables on CPU — tiny: num_sh * half floats each ──
        let mut cos_exp = vec![0.0f32; num_sh * half];
        let mut sin_exp = vec![0.0f32; num_sh * half];
        for s in 0..seq_len {
            let pos = positions[s] as f32;
            for h in 0..n_heads {
                let sh = s * n_heads + h;
                for d in 0..half {
                    let freq = pos / theta.powf(2.0 * d as f32 / head_dim as f32);
                    cos_exp[sh * half + d] = freq.cos();
                    sin_exp[sh * half + d] = freq.sin();
                }
            }
        }

        let stream = &self.stream;

        let upload_vec = |v: Vec<f32>| -> Result<Arc<Tensor<f32>>> {
            Ok(api::copy_host_vec_to_device(&Arc::new(v))
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .into())
        };

        // ── 2. Upload cos/sin to device (2 H→D, tiny) ─────────────────────────
        let cos_t = upload_vec(cos_exp)?;
        let sin_t = upload_vec(sin_exp)?;

        // ── 3–5. Deinterleave → RoPE → Interleave: all async, one final sync ──
        // Allocations (api::zeros) block only for cudaMalloc (fast); the kernel
        // launches that fill them use async_on so no CPU sync between them.
        // A single stream.synchronize() at the end covers all three kernels.
        let x_first_t = api::zeros::<f32>(&[num_sh * half])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([half]);
        let x_second_t = api::zeros::<f32>(&[num_sh * half])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([half]);

        let generics = vec![half.to_string()];

        // SAFETY: tensors are read/written only after the final stream.synchronize().
        let (x_first_dev, x_second_dev, _x) = unsafe {
            kernels::cutile_deinterleave(x_first_t, x_second_t, x_dev.clone())
                .generics(generics.clone())
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };
        let x_first_arc: Arc<Tensor<f32>> = Arc::new(x_first_dev.unpartition());
        let x_second_arc: Arc<Tensor<f32>> = Arc::new(x_second_dev.unpartition());

        let out_first_t = api::zeros::<f32>(&[num_sh * half])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([half]);
        let out_second_t = api::zeros::<f32>(&[num_sh * half])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([half]);

        let (out_f, out_s, _xf, _xs, _c, _s) = unsafe {
            kernels::cutile_rope_kernel(
                out_first_t,
                out_second_t,
                x_first_arc,
                x_second_arc,
                cos_t,
                sin_t,
            )
            .generics(generics)
            .async_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };
        let first_arc: Arc<Tensor<f32>> = Arc::new(out_f.unpartition());
        let second_arc: Arc<Tensor<f32>> = Arc::new(out_s.unpartition());

        let z_t = api::zeros::<f32>(&[num_sh * head_dim])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([half]);

        let (z_t, _of, _os) = unsafe {
            kernels::cutile_interleave(z_t, first_arc, second_arc)
                .generics(vec![half.to_string()])
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };

        // Single sync covers deinterleave + rope + interleave.
        // SAFETY: synchronize waits for all stream ops; no concurrent access.
        unsafe {
            stream
                .synchronize()
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }

        Ok(Arc::new(z_t.unpartition()))
    }

    /// Shared rotation apply: splits `x` into halves, runs the GPU kernel,
    /// and reconstructs `x` in-place.
    fn apply_rotation(
        &self,
        x: &mut [f32],
        num_sh: usize,
        head_dim: usize,
        half: usize,
        cos_exp: Vec<f32>,
        sin_exp: Vec<f32>,
    ) -> Result<()> {
        // Split x into first-half and second-half flat vectors.
        let mut x_first = vec![0.0f32; num_sh * half];
        let mut x_second = vec![0.0f32; num_sh * half];
        for sh in 0..num_sh {
            let base = sh * head_dim;
            x_first[sh * half..(sh + 1) * half].copy_from_slice(&x[base..base + half]);
            x_second[sh * half..(sh + 1) * half].copy_from_slice(&x[base + half..base + head_dim]);
        }

        let stream = &self.stream;
        let n = num_sh * half;

        let out_first_t = api::zeros::<f32>(&[n])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([half]);

        let out_second_t = api::zeros::<f32>(&[n])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([half]);

        let upload = |v: Vec<f32>| -> Result<Arc<Tensor<f32>>> {
            Ok(api::copy_host_vec_to_device(&Arc::new(v))
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .into())
        };

        let x_first_t = upload(x_first)?;
        let x_second_t = upload(x_second)?;
        let cos_t = upload(cos_exp)?;
        let sin_t = upload(sin_exp)?;

        let generics = vec![half.to_string()];

        let (out_f, out_s, _xf, _xs, _c, _s) = kernels::cutile_rope_kernel(
            out_first_t,
            out_second_t,
            x_first_t,
            x_second_t,
            cos_t,
            sin_t,
        )
        .generics(generics)
        .sync_on(stream)
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let host_first: Vec<f32> = out_f
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let host_second: Vec<f32> = out_s
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // Reconstruct x from first and second halves.
        for sh in 0..num_sh {
            let base = sh * head_dim;
            x[base..base + half].copy_from_slice(&host_first[sh * half..(sh + 1) * half]);
            x[base + half..base + head_dim]
                .copy_from_slice(&host_second[sh * half..(sh + 1) * half]);
        }
        Ok(())
    }
}
