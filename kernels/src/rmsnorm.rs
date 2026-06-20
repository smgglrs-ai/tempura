pub use crate::RMSNorm;
use crate::Result;

#[cfg(feature = "cuda")]
use cuda_core::DeviceBuffer;

#[cfg(feature = "cpu")]
pub struct CpuRMSNorm;

#[cfg(feature = "cpu")]
impl RMSNorm for CpuRMSNorm {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Proven
    }

    fn forward(&self, x: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) -> Result<()> {
        if weight.len() != hidden_size {
            return Err(crate::KernelError::DimensionMismatch {
                expected: hidden_size,
                got: weight.len(),
            });
        }

        let num_tokens = x.len() / hidden_size;
        for t in 0..num_tokens {
            let start = t * hidden_size;
            let row = &mut x[start..start + hidden_size];

            let rms: f32 =
                (row.iter().map(|v| v * v).sum::<f32>() / hidden_size as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;

            for (i, val) in row.iter_mut().enumerate() {
                *val = *val * inv_rms * weight[i];
            }
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub struct CudaRMSNorm {
    gpu: crate::gpu::GpuContext,
    module: crate::gpu::rmsnorm_kernel::kernels::LoadedModule,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaRMSNorm {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaRMSNorm {}

#[cfg(feature = "cuda")]
impl CudaRMSNorm {
    pub fn gpu(&self) -> &crate::gpu::GpuContext {
        &self.gpu
    }

    pub fn new() -> Result<Self> {
        let gpu = crate::gpu::GpuContext::new(0)?;
        let module = crate::gpu::rmsnorm_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(Self { gpu, module })
    }
}

#[cfg(feature = "cuda")]
impl CudaRMSNorm {
    pub fn forward_device(
        &self,
        x: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        hidden_size: usize,
        eps: f32,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();
        let num_tokens = x.len() / hidden_size;
        let warp_size = 32u32;

        // Padded buffer: one warp (32 lanes) per row, all lanes write
        let mut ss_dev = DeviceBuffer::zeroed(&stream, num_tokens * warp_size as usize)
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

        let module = &self.module;

        // Pass 1: warp-parallel sum of squares (one warp per row)
        let sum_sq_config = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (warp_size, 1, 1),
            shared_mem_bytes: 0,
        };
        module
            .rmsnorm_sum_sq(&stream, sum_sq_config, x, &mut ss_dev, hidden_size as u32)
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

        // Pass 2: apply normalization (ss read at stride 32)
        module
            .rmsnorm_apply(
                &stream,
                LaunchConfig::for_num_elems((num_tokens * hidden_size) as u32),
                x,
                weight,
                &ss_dev,
                output,
                hidden_size as u32,
                warp_size,
                eps,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl RMSNorm for CudaRMSNorm {
    fn forward(&self, x: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) -> Result<()> {
        use cuda_core::LaunchConfig;

        if weight.len() != hidden_size {
            return Err(crate::KernelError::DimensionMismatch {
                expected: hidden_size,
                got: weight.len(),
            });
        }

        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();
        let num_tokens = x.len() / hidden_size;
        let total = num_tokens * hidden_size;
        let warp_size = 32u32;

        let x_dev = crate::gpu::to_device(&stream, x)?;
        let w_dev = crate::gpu::to_device(&stream, weight)?;
        let mut ss_dev =
            crate::gpu::to_device(&stream, &vec![0.0f32; num_tokens * warp_size as usize])?;
        let mut out_dev = crate::gpu::to_device(&stream, &vec![0.0f32; total])?;

        let module = &self.module;

        let sum_sq_config = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (warp_size, 1, 1),
            shared_mem_bytes: 0,
        };
        module
            .rmsnorm_sum_sq(
                &stream,
                sum_sq_config,
                &x_dev,
                &mut ss_dev,
                hidden_size as u32,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

        module
            .rmsnorm_apply(
                &stream,
                LaunchConfig::for_num_elems(total as u32),
                &x_dev,
                &w_dev,
                &ss_dev,
                &mut out_dev,
                hidden_size as u32,
                warp_size,
                eps,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

        crate::gpu::copy_back(&stream, &out_dev, x)?;
        Ok(())
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::RMSNorm;

    #[kani::proof]
    fn rmsnorm_indices_in_bounds() {
        let hidden_size: usize = kani::any();
        let num_tokens: usize = kani::any();

        kani::assume(hidden_size >= 1 && hidden_size <= 8);
        kani::assume(num_tokens >= 1 && num_tokens <= 4);

        let x_len = num_tokens * hidden_size;

        let t: usize = kani::any();
        let i: usize = kani::any();
        kani::assume(t < num_tokens);
        kani::assume(i < hidden_size);

        let start = t * hidden_size;
        assert!(start + i < x_len);
        assert!(start + hidden_size <= x_len);
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn rmsnorm_full_execution_no_panic() {
        let hidden_size: usize = kani::any();
        kani::assume(hidden_size >= 1 && hidden_size <= 4);

        let mut x = vec![1.0f32; hidden_size];
        let weight = vec![1.0f32; hidden_size];
        let eps = 1e-6f32;

        let rn = super::CpuRMSNorm;
        let result = rn.forward(&mut x, &weight, hidden_size, eps);
        assert!(result.is_ok());
    }

    #[kani::proof]
    fn gpu_rmsnorm_pass_indices_in_bounds() {
        let h: usize = kani::any();
        let num_rows: usize = kani::any();

        kani::assume(h >= 1 && h <= 8);
        kani::assume(num_rows >= 1 && num_rows <= 4);

        let x_len = num_rows * h;
        let ss_len = num_rows;
        let weight_len = h;

        // Pass 1: start = row * h, access x[start + i] for i in 0..h
        let row1: usize = kani::any();
        kani::assume(row1 < num_rows);
        let start = row1 * h;
        let i1: usize = kani::any();
        kani::assume(i1 < h);
        assert!(start + i1 < x_len);

        // Pass 2: i decomposes into (row, col)
        let i2: usize = kani::any();
        kani::assume(i2 < x_len);
        let row2 = i2 / h;
        let col = i2 % h;
        assert!(row2 < ss_len);
        assert!(col < weight_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_unit_rms_with_identity_weights() {
        let rn = CpuRMSNorm;
        let hidden = 4;
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; hidden];

        rn.forward(&mut x, &weight, hidden, 1e-6).unwrap();

        let rms: f32 = (x.iter().map(|v| v * v).sum::<f32>() / hidden as f32).sqrt();
        assert_relative_eq!(rms, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_scaling_with_weights() {
        let rn = CpuRMSNorm;
        let hidden = 2;
        let mut x = vec![3.0, 4.0];
        let weight = vec![2.0, 0.5];

        rn.forward(&mut x, &weight, hidden, 1e-6).unwrap();

        // RMS of [3,4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        let rms_orig = (12.5f32).sqrt();
        assert_relative_eq!(x[0], 3.0 / rms_orig * 2.0, epsilon = 1e-4);
        assert_relative_eq!(x[1], 4.0 / rms_orig * 0.5, epsilon = 1e-4);
    }

    #[test]
    fn test_dimension_mismatch() {
        let rn = CpuRMSNorm;
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0]; // wrong size

        let result = rn.forward(&mut x, &weight, 4, 1e-6);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_tokens() {
        let rn = CpuRMSNorm;
        let hidden = 2;
        // Two tokens: [1,1] and [3,4]
        let mut x = vec![1.0, 1.0, 3.0, 4.0];
        let weight = vec![1.0; hidden];

        rn.forward(&mut x, &weight, hidden, 1e-6).unwrap();

        // Each row independently normalized to unit RMS
        let rms0: f32 = (x[0] * x[0] + x[1] * x[1]).sqrt() / (hidden as f32).sqrt();
        let rms1: f32 = (x[2] * x[2] + x[3] * x[3]).sqrt() / (hidden as f32).sqrt();
        assert_relative_eq!(rms0, 1.0, epsilon = 1e-4);
        assert_relative_eq!(rms1, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_rmsnorm_matches_f64_reference() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, -1.0, 0.001, 7.0];
        let h = input.len();
        let weight = vec![1.0f32; h];
        let eps = 1e-6f32;

        let rms_f64 = (input.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / h as f64
            + eps as f64)
            .sqrt();
        let ref_out: Vec<f64> = input
            .iter()
            .zip(weight.iter())
            .map(|(&x, &w)| (x as f64) / rms_f64 * (w as f64))
            .collect();

        let rn = CpuRMSNorm;
        let mut x = input.clone();
        rn.forward(&mut x, &weight, h, eps).unwrap();

        for (i, (&f32_val, &f64_val)) in x.iter().zip(ref_out.iter()).enumerate() {
            let diff = (f32_val as f64 - f64_val).abs();
            assert!(
                diff < 1e-5,
                "rmsnorm mismatch at {i}: f32={f32_val}, f64_ref={f64_val}, diff={diff}"
            );
        }
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_output_has_unit_rms(
                input in prop::collection::vec(0.1f32..10.0f32, 4..32)
            ) {
                let rn = CpuRMSNorm;
                let h = input.len();
                let weight = vec![1.0f32; h];
                let mut x = input.clone();
                rn.forward(&mut x, &weight, h, 1e-6).unwrap();
                let rms = (x.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
                prop_assert!((rms - 1.0).abs() < 1e-3, "rms={rms}");
            }
        }
    }
}
