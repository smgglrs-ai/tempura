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
    /// GEMM with B transposed: z = x @ y^T
    /// Accumulator starts at 0 in registers — caller need NOT pre-zero z.
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
        // Zero accumulator in registers — no memory load, no pre-zero of z required.
        let mut tile_z: Tile<f32, { [BM, BN] }> = broadcast_scalar(0.0f32, shape![BM, BN]);
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

const BM: usize = 64;
const BN: usize = 64;
const BK: usize = 16;

/// Round up to the next power-of-2 ≥ x. Required: cuTile JIT on sm_120
/// rejects non-power-of-2 K const generics in MMA-accumulation loops.
fn next_pow2(x: usize) -> usize {
    if x <= 1 { return 1; }
    1usize << (usize::BITS - (x - 1).leading_zeros()) as usize
}

// Decode/GEMV path (m == 1): wide N tiles amortise memory latency better than
// a square BM×BN block when there is only one output row.
const BM_GEMV: usize = 1;
const BN_GEMV: usize = 64;
const BK_GEMV: usize = 16;

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

        // Pad K to multiple of BK if necessary
        let k_pad = next_pow2(k.div_ceil(BK) * BK);
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

        if m == 1 {
            return self.forward_gemv(a, b, output, n, k);
        }

        // Pad K to multiple of BK if necessary
        let k_pad = next_pow2(k.div_ceil(BK) * BK);
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

impl CutileMatMul {
    /// GEMM with B pre-uploaded to the GPU: z = a @ b^T.
    ///
    /// Shapes: z[M, N] = a[M, K_raw] @ b[N, K_pad]^T.
    /// `a` is supplied from the host each call; `b_device` is resident on the GPU.
    /// `k_raw` is the logical K size; `k_pad` is the padded K the tensor was uploaded with
    /// (must be a multiple of BK = 8 and match the tensor's actual column count).
    pub fn forward_bt_device_b(
        &self,
        a: &[f32],
        b_device: &Arc<Tensor<f32>>,
        output: &mut [f32],
        m: usize,
        n: usize,
        k_raw: usize,
        k_pad: usize,
    ) -> Result<()> {
        if a.len() != m * k_raw {
            return Err(KernelError::DimensionMismatch {
                expected: m * k_raw,
                got: a.len(),
            });
        }
        if output.len() != m * n {
            return Err(KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }

        // Pad A to k_pad if necessary so the kernel sees [M, K_pad].
        let a_vec: Arc<Vec<f32>> = if k_pad == k_raw {
            Arc::new(a.to_vec())
        } else {
            let mut padded = vec![0.0f32; m * k_pad];
            for r in 0..m {
                padded[r * k_pad..r * k_pad + k_raw]
                    .copy_from_slice(&a[r * k_raw..(r + 1) * k_raw]);
            }
            Arc::new(padded)
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

        // Clone the Arc so b_device keeps its reference after the kernel takes ownership.
        let y = Arc::clone(b_device);

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

impl CutileMatMul {
    /// Both A and B are already on device — no host-to-device upload needed.
    ///
    /// A: [M, K] on device, B: [N, K] on device (B-transposed layout).
    /// Writes [M, N] result to `output` on the host.
    pub fn forward_bt_device(
        &self,
        a: &Arc<Tensor<f32>>,
        b: &Arc<Tensor<f32>>,
        output: &mut Vec<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let k_pad = next_pow2(k.div_ceil(BK) * BK);
        let stream = &self.stream;

        let z = api::zeros::<f32>(&[m, n])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([BM, BN]);

        let generics = vec![
            BM.to_string(),
            BN.to_string(),
            BK.to_string(),
            k_pad.to_string(),
        ];

        let (z, _a, _b) = kernels::cutile_gemm_b_transposed(z, Arc::clone(a), Arc::clone(b))
            .generics(generics)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let host: Vec<f32> = z
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        *output = host;
        Ok(())
    }

    /// A on host, B already on device — only A is uploaded.
    ///
    /// A: [M, K] on host, B: [N, K] on device (B-transposed layout).
    /// Writes [M, N] result to `output` on the host.
    pub fn forward_bt_a_host_b_device(
        &self,
        a: &[f32],
        b: &Arc<Tensor<f32>>,
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k {
            return Err(KernelError::DimensionMismatch { expected: m * k, got: a.len() });
        }
        if output.len() != m * n {
            return Err(KernelError::DimensionMismatch { expected: m * n, got: output.len() });
        }

        let k_pad = next_pow2(k.div_ceil(BK) * BK);
        let stream = &self.stream;

        let a_vec = if k_pad == k {
            Arc::new(a.to_vec())
        } else {
            let mut padded = vec![0.0f32; m * k_pad];
            for r in 0..m {
                padded[r * k_pad..r * k_pad + k].copy_from_slice(&a[r * k..(r + 1) * k]);
            }
            Arc::new(padded)
        };

        let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&a_vec)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[m, k_pad])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let z = api::zeros::<f32>(&[m, n])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([BM, BN]);

        let generics = vec![
            BM.to_string(),
            BN.to_string(),
            BK.to_string(),
            k_pad.to_string(),
        ];

        let (z, _x, _b) = kernels::cutile_gemm_b_transposed(z, x, Arc::clone(b))
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

    /// GEMV path for `forward_b_transposed` when m == 1.
    ///
    /// Uses smaller BM_GEMV×BN_GEMV tiles that better amortise memory latency
    /// for a single-row output on Blackwell hardware.
    fn forward_gemv(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        n: usize,
        k: usize,
    ) -> Result<()> {
        let m = 1usize;
        let k_pad = next_pow2(k.div_ceil(BK_GEMV) * BK_GEMV);
        let (a_vec, b_vec) = if k_pad == k {
            (Arc::new(a.to_vec()), Arc::new(b.to_vec()))
        } else {
            let mut a_padded = vec![0.0f32; k_pad];
            a_padded[..k].copy_from_slice(a);
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
            .partition([BM_GEMV, BN_GEMV]);

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
            BM_GEMV.to_string(),
            BN_GEMV.to_string(),
            BK_GEMV.to_string(),
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

    /// Device-to-device GEMM reusing a caller-supplied pre-allocated output buffer.
    ///
    /// `out` must contain exactly m×n elements (typically pre-allocated as a 1-D
    /// tensor via `api::zeros::<f32>(&[n])`). It is zeroed in-place then reshaped
    /// to `[m, n]` — no new device allocation is made. Returns the filled tensor
    /// as an `Arc` so the caller can store a clone in a scratch pool and later
    /// reclaim the underlying `Tensor<f32>` via `Arc::try_unwrap` once its
    /// reference count drops to one.
    pub fn forward_bt_d2d_into(
        &self,
        a: &Arc<Tensor<f32>>,
        b: &Arc<Tensor<f32>>,
        out: Tensor<f32>,
        m: usize,
        n: usize,
        k_pad: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let stream = &self.stream;

        // Kernel initializes accumulator to 0 in registers — no fill needed.
        let z = out
            .reshape(&[m, n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([BM, BN]);

        let generics = vec![
            BM.to_string(),
            BN.to_string(),
            BK.to_string(),
            k_pad.to_string(),
        ];

        let (z, _a, _b) = kernels::cutile_gemm_b_transposed(z, Arc::clone(a), Arc::clone(b))
            .generics(generics)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        Ok(Arc::new(z.unpartition()))
    }

    /// Full device-to-device: A and B on device, output stays on device.
    ///
    /// A: [M, K] on device, B: [N, K] on device (B-transposed layout).
    /// Returns [M, N] tensor on device — no host round-trip.
    pub fn forward_bt_device_to_device(
        &self,
        a: &Arc<Tensor<f32>>,
        b: &Arc<Tensor<f32>>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let k_pad = next_pow2(k.div_ceil(BK) * BK);
        let stream = &self.stream;

        let z = api::zeros::<f32>(&[m, n])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([BM, BN]);

        let generics = vec![
            BM.to_string(),
            BN.to_string(),
            BK.to_string(),
            k_pad.to_string(),
        ];

        let (z, _a, _b) = kernels::cutile_gemm_b_transposed(z, Arc::clone(a), Arc::clone(b))
            .generics(generics)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        Ok(Arc::new(z.unpartition()))
    }
}
