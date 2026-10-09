use std::sync::Arc;

fn next_pow2(x: usize) -> usize {
    if x <= 1 {
        return 1;
    }
    1usize << (usize::BITS - (x - 1).leading_zeros()) as usize
}

use crate::{KernelError, RMSNorm, Result, VerificationTier};
use cuda_core::{Device, Stream};
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Reshape, Tensor};

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// RMSNorm pass 1: compute sum(x²) per token row into a 1-element tile.
    ///
    /// One program instance per token. H = hidden_size.
    /// Avoids all f32 scalar arithmetic in the JIT — sum is stored as a tile
    /// element; the CPU computes scale = 1/sqrt(mean+eps) between the two passes.
    #[cutile::entry()]
    pub fn rmsnorm_sum_sq<const H: i32>(
        sum_out: &mut Tensor<f32, { [1] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile_x: Tile<f32, { [H] }> = x.partition(shape![H]).load([pid]);
        let sq: Tile<f32, { [H] }> = tile_x * tile_x;
        let sum: Tile<f32, { [] }> = reduce_sum(sq, 0i32);
        sum_out.store(sum.reshape(shape![1]));
    }

    /// RMSNorm scale kernel: compute per-token scale = 1/sqrt(sum_sq/hidden_size + eps).
    ///
    /// One program instance per token. Runs entirely on device between pass 1
    /// (sum_sq) and pass 2 (apply), eliminating the CPU round-trip.
    /// `inv_h` = 1.0 / hidden_size is pre-computed on the host and passed as
    /// a scalar parameter to avoid integer-to-f32 casting inside the JIT.
    #[cutile::entry()]
    pub fn cutile_rmsnorm_scale(
        scale: &mut Tensor<f32, { [1] }>,
        sum_sq: &Tensor<f32, { [-1] }>,
        inv_h: f32,
        eps: f32,
    ) {
        let pid = program_id(0);
        let s: Tile<f32, { [1] }> = sum_sq.partition(shape![1]).load([pid]);
        let mean_eps = s * broadcast_scalar(inv_h, shape![1]) + broadcast_scalar(eps, shape![1]);
        scale.store(rsqrt(mean_eps, ftz::Disabled));
    }

    /// RMSNorm pass 2: apply GPU-computed per-token scale and weight.
    ///
    /// One program instance per token. H = hidden_size.
    /// `scale` carries one float per token: 1/sqrt(mean(x²)+eps).
    /// Broadcasting is done via tile_to_scalar + broadcast_scalar, keeping
    /// all operations within cuTile tile ops — no f32 scalar arithmetic.
    #[cutile::entry()]
    pub fn rmsnorm_apply<const H: i32>(
        out: &mut Tensor<f32, { [H] }>,
        x: &Tensor<f32, { [-1] }>,
        weight: &Tensor<f32, { [-1] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile_x: Tile<f32, { [H] }> = x.partition(shape![H]).load([pid]);
        let tile_w: Tile<f32, { [H] }> = weight.partition(shape![H]).load([0i32]);
        let tile_s: Tile<f32, { [1] }> = scale.partition(shape![1]).load([pid]);
        let s_broad: Tile<f32, { [H] }> =
            broadcast_scalar(tile_to_scalar(reduce_sum(tile_s, 0i32)), shape![H]);
        out.store(tile_x * s_broad * tile_w);
    }
}

/// GPU-accelerated RMSNorm using NVIDIA cuTile Rust (`cutile-rs`).
///
/// Two-pass approach: pass 1 computes sum(x²) per token on GPU, pass 2 applies
/// the scale (computed on CPU from the downloaded sums) and weight on GPU.
/// All f32 scalar arithmetic (division, sqrt) runs on CPU between the two passes,
/// avoiding the cuTile JIT's restriction on f32 scalar arithmetic in kernels.
pub struct CutileRmsNorm {
    device: Arc<Device>,
    stream: Arc<Stream>,
}

unsafe impl Send for CutileRmsNorm {}
unsafe impl Sync for CutileRmsNorm {}

impl CutileRmsNorm {
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

    /// RMSNorm entirely on device.
    ///
    /// `x_dev` is a flat 1-D device tensor of `seq_len * h_pad` elements (any
    /// shape is accepted; data is treated as row-major `[seq_len, h_pad]`).
    /// `weight_dev` is a pre-uploaded 1-D tensor of `h_pad` elements.
    ///
    /// Pass 1 computes `sum(x²)` per token on device.  Only `seq_len` f32 sums
    /// cross the PCIe bus (D→H).  The CPU computes `scale[t] = 1/sqrt(mean+eps)`
    /// and uploads `seq_len` f32 scales (H→D).  Pass 2 applies scale and weight
    /// entirely on device.
    ///
    /// Returns a `[seq_len, h_pad]` device tensor ready for projection GEMMs.
    pub fn forward_device(
        &self,
        stream: &Arc<Stream>,
        x_dev: Arc<Tensor<f32>>,
        weight_dev: &Arc<Tensor<f32>>,
        seq_len: usize,
        hidden_size: usize,
        h_pad: usize,
        eps: f32,
    ) -> Result<Arc<Tensor<f32>>> {
        // Pass 1: sum(x²) per token — one scalar per token on device.
        let sum_t = api::zeros::<f32>(&[seq_len])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1]);

        let (sum_t, x_t) = kernels::rmsnorm_sum_sq(sum_t, x_dev)
            .generics(vec![h_pad.to_string()])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // Download seq_len floats — negligible PCIe traffic.
        let sum_sq_vec: Vec<f32> = sum_t
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // CPU: scale[t] = 1/sqrt(mean(x²) + eps).
        let scale_vec: Vec<f32> = sum_sq_vec
            .iter()
            .map(|&s| 1.0f32 / (s / hidden_size as f32 + eps).sqrt())
            .collect();

        let scale_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(scale_vec))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        // Pass 2: apply scale and weight on device.
        let out_t = api::zeros::<f32>(&[seq_len * h_pad])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([h_pad]);

        let (out_t, _x, _w, _s) = kernels::rmsnorm_apply(out_t, x_t, weight_dev.clone(), scale_t)
            .generics(vec![h_pad.to_string()])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // Reshape flat output to [seq_len, h_pad] for direct use by projection GEMMs.
        let out_2d: Arc<Tensor<f32>> = out_t
            .unpartition()
            .reshape(&[seq_len, h_pad])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();
        Ok(out_2d)
    }

    /// Fully-fused RMSNorm entirely on device — no CPU round-trip between passes.
    ///
    /// Same interface as `forward_device` but all three GPU kernels are submitted
    /// asynchronously (pass 1 → scale → pass 2) and a single stream synchronize
    /// covers them all, eliminating the two blocking `sync_on` calls that
    /// previously forced the CPU to wait for each pass before the next began.
    ///
    /// 112 blocking round-trips per decode token (2 RMSNorm × 28+ layers × 2 sync_on)
    /// become 3 async kernel launches + 1 synchronize.
    pub fn forward_device_fused(
        &self,
        stream: &Arc<Stream>,
        x_dev: Arc<Tensor<f32>>,
        weight_dev: &Arc<Tensor<f32>>,
        seq_len: usize,
        hidden_size: usize,
        h_pad: usize,
        eps: f32,
    ) -> Result<Arc<Tensor<f32>>> {
        let inv_h = 1.0f32 / hidden_size as f32;

        // Pass 1: sum(x²) per token — async.
        let sum_t = api::zeros::<f32>(&[seq_len])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1]);

        // SAFETY: tensors are accessed only after the final stream.synchronize().
        let (sum_t, x_t) = unsafe {
            kernels::rmsnorm_sum_sq(sum_t, x_dev)
                .generics(vec![h_pad.to_string()])
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };
        let sum_arc: Arc<Tensor<f32>> = Arc::new(sum_t.unpartition());

        // Scale kernel: scale[t] = 1/sqrt(sum_sq[t]/hidden_size + eps) — async.
        let scale_t = api::zeros::<f32>(&[seq_len])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1]);

        let (scale_t, _sum, _inv_h, _eps) = unsafe {
            kernels::cutile_rmsnorm_scale(scale_t, sum_arc, inv_h, eps)
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };
        let scale_arc: Arc<Tensor<f32>> = Arc::new(scale_t.unpartition());

        // Pass 2: apply scale and weight on device — async.
        let out_t = api::zeros::<f32>(&[seq_len * h_pad])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([h_pad]);

        let (out_t, _x, _w, _s) = unsafe {
            kernels::rmsnorm_apply(out_t, x_t, weight_dev.clone(), scale_arc)
                .generics(vec![h_pad.to_string()])
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };

        // Single sync covers sum_sq + scale + apply.
        // SAFETY: synchronize waits for all stream ops; no concurrent access.
        unsafe {
            stream
                .synchronize()
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }

        let out_2d: Arc<Tensor<f32>> = out_t
            .unpartition()
            .reshape(&[seq_len, h_pad])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();
        Ok(out_2d)
    }
}

impl RMSNorm for CutileRmsNorm {
    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Tested
    }

    fn forward(&self, x: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) -> Result<()> {
        if weight.len() != hidden_size {
            return Err(KernelError::DimensionMismatch {
                expected: hidden_size,
                got: weight.len(),
            });
        }

        let num_tokens = x.len() / hidden_size;
        if num_tokens == 0 {
            return Ok(());
        }

        // cuTile JIT on sm_120 requires tile/reduction dims to be power-of-2.
        // Pad hidden_size to next power-of-2; padded elements are zero so they
        // don't affect sum(x²) or the weighted output.
        let h_pad = next_pow2(hidden_size);

        let stream = &self.stream;

        // Upload x (zero-padded to h_pad per token) and weight (zero-padded).
        let x_padded: Vec<f32> = if h_pad == hidden_size {
            x.to_vec()
        } else {
            let mut v = vec![0.0f32; num_tokens * h_pad];
            for t in 0..num_tokens {
                v[t * h_pad..t * h_pad + hidden_size]
                    .copy_from_slice(&x[t * hidden_size..(t + 1) * hidden_size]);
            }
            v
        };
        let w_padded: Vec<f32> = if h_pad == hidden_size {
            weight.to_vec()
        } else {
            let mut v = vec![0.0f32; h_pad];
            v[..hidden_size].copy_from_slice(weight);
            v
        };

        let x_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x_padded))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let w_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(w_padded))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        // Pass 1: compute sum(x²) per token — one element per token in sum_t.
        let sum_t = api::zeros::<f32>(&[num_tokens])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1]);

        let (sum_t, x_t) = kernels::rmsnorm_sum_sq(sum_t, x_t)
            .generics(vec![h_pad.to_string()])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let sum_sq_vec: Vec<f32> = sum_t
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // CPU: scale[i] = 1 / sqrt(mean(x²) + eps) — no JIT constraint here.
        let scale_vec: Vec<f32> = sum_sq_vec
            .iter()
            .map(|&s| 1.0f32 / (s / hidden_size as f32 + eps).sqrt())
            .collect();

        // Pass 2: apply scale and weight.
        let scale_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(scale_vec))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let out_t = api::zeros::<f32>(&[num_tokens * h_pad])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([h_pad]);

        let (out_t, _x, _w, _s) = kernels::rmsnorm_apply(out_t, x_t, w_t, scale_t)
            .generics(vec![h_pad.to_string()])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let host: Vec<f32> = out_t
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // Trim padded output back to hidden_size per token.
        if h_pad == hidden_size {
            x.copy_from_slice(&host);
        } else {
            for t in 0..num_tokens {
                x[t * hidden_size..(t + 1) * hidden_size]
                    .copy_from_slice(&host[t * h_pad..t * h_pad + hidden_size]);
            }
        }
        Ok(())
    }
}
