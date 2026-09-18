use std::sync::Arc;
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Tensor};
use cuda_core::{Device, Stream};
use crate::{KernelError, RMSNorm, Result, VerificationTier};

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// RMSNorm forward: out[i] = x[i] * weight[i] / sqrt(mean(x²) + eps)
    ///
    /// One kernel instance per row (token). H = hidden_size.
    /// Single pass: reduce to sum-of-squares, compute scale, apply.
    #[cutile::entry()]
    pub fn rmsnorm_forward<const H: i32>(
        out: &mut Tensor<f32, { [H] }>,
        x: &Tensor<f32, { [-1] }>,
        weight: &Tensor<f32, { [-1] }>,
        eps: f32,
    ) {
        let pid = program_id(0);
        let tile_x: Tile<f32, { [H] }> = x.partition(shape![H]).load([pid]);
        let tile_w: Tile<f32, { [H] }> = weight.partition(shape![H]).load([0i32]);

        let tile_sq: Tile<f32, { [H] }> = tile_x * tile_x;
        let sum_sq_0d: Tile<f32, { [] }> = reduce_sum(tile_sq, 0i32);
        let sum_sq: f32 = tile_to_scalar(sum_sq_0d);
        let rms_scale: f32 = 1.0f32 / (sum_sq / H as f32 + eps).sqrt();

        let scale_tile: Tile<f32, { [H] }> = broadcast_scalar(rms_scale, shape![H]);
        let result: Tile<f32, { [H] }> = tile_x * scale_tile * tile_w;

        out.store(result);
    }
}

/// GPU-accelerated RMSNorm using NVIDIA cuTile Rust (`cutile-rs`).
///
/// Single-pass kernel: each program instance handles one token row,
/// reducing to sum-of-squares and applying the inverse RMS scale in a
/// single tile-level pass without a temporary intermediate buffer.
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

        let stream = &self.stream;

        // Partition output into one tile of H elements per token row.
        let out_t = api::zeros::<f32>(&[num_tokens * hidden_size])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([hidden_size]);

        let x_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x.to_vec()))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let w_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(weight.to_vec()))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let (out_t, _x, _w, _eps) = kernels::rmsnorm_forward(out_t, x_t, w_t, eps)
            .generics(vec![hidden_size.to_string()])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let host: Vec<f32> = out_t
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        x.copy_from_slice(&host);
        Ok(())
    }
}
