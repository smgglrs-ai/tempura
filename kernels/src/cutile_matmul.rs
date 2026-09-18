use std::sync::Arc;
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Reshape, Tensor};
use cuda_core::{Device, Stream};
use crate::{KernelError, MatMul, Result, VerificationTier};

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// GEMM: z = x @ y
    /// Shapes: z: [M, N], x: [M, K], y: [K, N]
    /// Tile dimensions: BM, BN, BK; K must be divisible by BK.
    #[cutile::entry()]
    pub fn cutile_gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid_m = program_id(0);
        let pid_n = program_id(1);
        let mut tile_z = load_tile_mut(z);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid_m, i]);
            let tile_y = part_y.load([i, pid_n]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }

    /// GEMM with B transposed: z = x @ y^T
    /// Shapes: z: [M, N], x: [M, K], y: [N, K]
    /// Tile dimensions: BM, BN, BK; K must be divisible by BK.
    #[cutile::entry()]
    pub fn cutile_gemm_b_transposed<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [-1, K] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BN, BK]);
        let pid_m = program_id(0);
        let pid_n = program_id(1);
        let mut tile_z = load_tile_mut(z);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid_m, i]);
            let tile_y = part_y.load([pid_n, i]);
            tile_z = mma(tile_x, tile_y.transpose(), tile_z);
        }
        z.store(tile_z);
    }
}

/// GPU-accelerated Matrix Multiplication using NVIDIA cuTile Rust (`cutile-rs`).
///
/// Executes on standard stable Rust toolchain without custom compiler backends.
pub struct CutileMatMul {
    device: Arc<Device>,
    stream: Arc<Stream>,
}

unsafe impl Send for CutileMatMul {}
unsafe impl Sync for CutileMatMul {}

impl CutileMatMul {
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

const BM: usize = 16;
const BN: usize = 16;
const BK: usize = 8;

impl MatMul for CutileMatMul {
    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Tested
    }

    fn forward(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k {
            return Err(KernelError::DimensionMismatch {
                expected: m * k,
                got: a.len(),
            });
        }
        if b.len() != k * n {
            return Err(KernelError::DimensionMismatch {
                expected: k * n,
                got: b.len(),
            });
        }
        if output.len() != m * n {
            return Err(KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }

        // Pad K to multiple of BK (8) if necessary
        let k_pad = k.div_ceil(BK) * BK;
        let (a_vec, b_vec) = if k_pad == k {
            (Arc::new(a.to_vec()), Arc::new(b.to_vec()))
        } else {
            let mut a_padded = vec![0.0f32; m * k_pad];
            for r in 0..m {
                a_padded[r * k_pad..r * k_pad + k].copy_from_slice(&a[r * k..(r + 1) * k]);
            }
            let mut b_padded = vec![0.0f32; k_pad * n];
            for r in 0..k {
                b_padded[r * n..(r + 1) * n].copy_from_slice(&b[r * n..(r + 1) * n]);
            }
            (Arc::new(a_padded), Arc::new(b_padded))
        };

        let stream = &self.stream;

        // Allocate and partition output tensor
        let z = api::zeros::<f32>(&[m, n])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([BM, BN]);

        // Upload and reshape input tensors
        let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&a_vec)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[m, k_pad])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let y: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&b_vec)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[k_pad, n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let generics = vec![
            BM.to_string(),
            BN.to_string(),
            BK.to_string(),
            k_pad.to_string(),
        ];

        let (z, _x, _y) = kernels::cutile_gemm(z, x, y)
            .generics(generics)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let host: Vec<f32> = z
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        output.copy_from_slice(&host);
        Ok(())
    }

    fn forward_b_transposed(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k {
            return Err(KernelError::DimensionMismatch {
                expected: m * k,
                got: a.len(),
            });
        }
        if b.len() != n * k {
            return Err(KernelError::DimensionMismatch {
                expected: n * k,
                got: b.len(),
            });
        }
        if output.len() != m * n {
            return Err(KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }

        // Pad K to multiple of BK (8) if necessary
        let k_pad = k.div_ceil(BK) * BK;
        let (a_vec, b_vec) = if k_pad == k {
            (Arc::new(a.to_vec()), Arc::new(b.to_vec()))
        } else {
            let mut a_padded = vec![0.0f32; m * k_pad];
            for r in 0..m {
                a_padded[r * k_pad..r * k_pad + k].copy_from_slice(&a[r * k..(r + 1) * k]);
            }
            let mut b_padded = vec![0.0f32; n * k_pad];
            for r in 0..n {
                b_padded[r * k_pad..r * k_pad + k].copy_from_slice(&b[r * k..(r + 1) * k]);
            }
            (Arc::new(a_padded), Arc::new(b_padded))
        };

        let stream = &self.stream;

        let z = api::zeros::<f32>(&[m, n])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([BM, BN]);

        let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&a_vec)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[m, k_pad])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let y: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&b_vec)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[n, k_pad])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let generics = vec![
            BM.to_string(),
            BN.to_string(),
            BK.to_string(),
            k_pad.to_string(),
        ];

        let (z, _x, _y) = kernels::cutile_gemm_b_transposed(z, x, y)
            .generics(generics)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let host: Vec<f32> = z
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        output.copy_from_slice(&host);
        Ok(())
    }
}
