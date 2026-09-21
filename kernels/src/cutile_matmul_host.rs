use crate::cutile_matmul::CutileMatMul;
use crate::{MatMul, Result, VerificationTier};

/// Host-side adapter over [`CutileMatMul`] that mirrors the calling conventions
/// of [`crate::matmul::CudaMatMul`].
///
/// Both standard tiled GEMM and MMA-style GEMM route through `CutileMatMul`.
/// cuTile's Tile IR compiles the `mma` instruction to hardware MMA ops, so there
/// is no need for a separate MMA implementation.
pub struct CutileMatMulHost {
    inner: CutileMatMul,
}

unsafe impl Send for CutileMatMulHost {}
unsafe impl Sync for CutileMatMulHost {}

impl CutileMatMulHost {
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: CutileMatMul::new()?,
        })
    }

    pub fn with_device(ordinal: usize) -> Result<Self> {
        Ok(Self {
            inner: CutileMatMul::with_device(ordinal)?,
        })
    }

    pub fn inner(&self) -> &CutileMatMul {
        &self.inner
    }
}

impl MatMul for CutileMatMulHost {
    fn verification_tier(&self) -> VerificationTier {
        self.inner.verification_tier()
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
        self.inner.forward(a, b, output, m, n, k)
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
        self.inner.forward_b_transposed(a, b, output, m, n, k)
    }
}

/// Device-buffer API for call sites that mirror [`crate::matmul::CudaMatMul`]'s
/// `forward_device` / `forward_bt_device` methods.
///
/// Because `CutileMatMul` manages its own device memory via cutile's API, these
/// methods round-trip through host memory. They are intended for correctness /
/// integration; performance-critical paths should use the host-slice `MatMul`
/// trait directly.
#[cfg(all(feature = "cuda", not(feature = "cutile")))]
impl CutileMatMulHost {
    pub fn forward_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b: &cuda_core::DeviceBuffer<f32>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let stream = self.inner.stream();
        let a_host = a
            .to_host_vec(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let b_host = b
            .to_host_vec(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let mut out_host = vec![0.0f32; m * n];
        self.inner
            .forward(&a_host, &b_host, &mut out_host, m, n, k)?;
        output
            .copy_from_host(stream, &out_host)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn forward_bt_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b: &cuda_core::DeviceBuffer<f32>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let stream = self.inner.stream();
        let a_host = a
            .to_host_vec(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let b_host = b
            .to_host_vec(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let mut out_host = vec![0.0f32; m * n];
        self.inner
            .forward_b_transposed(&a_host, &b_host, &mut out_host, m, n, k)?;
        output
            .copy_from_host(stream, &out_host)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }
}
