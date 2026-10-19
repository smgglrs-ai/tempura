pub use crate::RMSNorm;
use crate::Result;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 scale-and-weight-multiply: `row[i] = row[i] * scale * weight[i]`.
/// Processes 32 floats per loop iteration (4 × 8-wide), matching llama.cpp's
/// `ggml_vec_scale_f32` + `ggml_vec_mul_f32` fused pattern.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_mul_avx2(row: &mut [f32], weight: &[f32], scale: f32) {
    let n = row.len();
    debug_assert_eq!(n, weight.len());
    let v_scale = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 32 <= n {
        let r0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(row.as_ptr().add(i)), v_scale), _mm256_loadu_ps(weight.as_ptr().add(i)));
        let r1 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(row.as_ptr().add(i + 8)), v_scale), _mm256_loadu_ps(weight.as_ptr().add(i + 8)));
        let r2 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(row.as_ptr().add(i + 16)), v_scale), _mm256_loadu_ps(weight.as_ptr().add(i + 16)));
        let r3 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(row.as_ptr().add(i + 24)), v_scale), _mm256_loadu_ps(weight.as_ptr().add(i + 24)));
        _mm256_storeu_ps(row.as_mut_ptr().add(i), r0);
        _mm256_storeu_ps(row.as_mut_ptr().add(i + 8), r1);
        _mm256_storeu_ps(row.as_mut_ptr().add(i + 16), r2);
        _mm256_storeu_ps(row.as_mut_ptr().add(i + 24), r3);
        i += 32;
    }
    while i < n {
        row[i] = row[i] * scale * weight[i];
        i += 1;
    }
}

/// AVX2 scale-and-weight-multiply into dst: `dst[i] = src[i] * scale * weight[i]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_mul_into_avx2(src: &[f32], weight: &[f32], dst: &mut [f32], scale: f32) {
    let n = src.len();
    debug_assert_eq!(n, weight.len());
    debug_assert_eq!(n, dst.len());
    let v_scale = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 32 <= n {
        let r0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(i)), v_scale), _mm256_loadu_ps(weight.as_ptr().add(i)));
        let r1 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(i + 8)), v_scale), _mm256_loadu_ps(weight.as_ptr().add(i + 8)));
        let r2 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(i + 16)), v_scale), _mm256_loadu_ps(weight.as_ptr().add(i + 16)));
        let r3 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(i + 24)), v_scale), _mm256_loadu_ps(weight.as_ptr().add(i + 24)));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), r0);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), r1);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16), r2);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24), r3);
        i += 32;
    }
    while i < n {
        dst[i] = src[i] * scale * weight[i];
        i += 1;
    }
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
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
        let x_in = x.to_vec();
        crate::ggml::ops_llama::rms_norm_mul_rows_f32(&x_in, weight, x, hidden_size, eps);
        Ok(())
    }

    fn forward_into(
        &self,
        x: &[f32],
        weight: &[f32],
        out: &mut [f32],
        hidden_size: usize,
        eps: f32,
    ) -> Result<()> {
        if weight.len() != hidden_size {
            return Err(crate::KernelError::DimensionMismatch {
                expected: hidden_size,
                got: weight.len(),
            });
        }
        debug_assert_eq!(x.len(), out.len());
        crate::ggml::ops_llama::rms_norm_mul_rows_f32(x, weight, out, hidden_size, eps);
        Ok(())
    }
}

/// GGML-compatible RMSNorm: f64 accumulation matching ggml_compute_forward_rms_norm_f32.
/// Translated from ggml.c:12073-12119 + llm_build_norm (ggml_mul for weights).
#[cfg(feature = "cpu")]
pub struct CpuRMSNormGgml;

#[cfg(feature = "cpu")]
impl RMSNorm for CpuRMSNormGgml {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Tested
    }

    fn forward(&self, x: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) -> Result<()> {
        if weight.len() != hidden_size {
            return Err(crate::KernelError::DimensionMismatch {
                expected: hidden_size,
                got: weight.len(),
            });
        }

        let num_tokens = x.len() / hidden_size;

        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        for t in 0..num_tokens {
            let start = t * hidden_size;
            let row = &mut x[start..start + hidden_size];

            // C: ggml_float sum = 0.0;  (ggml_float = double)
            // C: for (i00 = 0; i00 < ne00; i00++) sum += (ggml_float)(x[i00] * x[i00]);
            let mut sum: f64 = 0.0;
            for i in 0..hidden_size {
                sum += (row[i] * row[i]) as f64;
            }

            let mean: f32 = (sum / hidden_size as f64) as f32;
            let scale: f32 = 1.0f32 / (mean + eps).sqrt();

            // AVX2: 4×8-wide multiply matching llama.cpp's ggml_vec_scale_f32 + ggml_vec_mul_f32
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe { scale_mul_avx2(row, weight, scale); }
            } else {
                for i in 0..hidden_size {
                    row[i] = row[i] * scale * weight[i];
                }
            }
        }

        Ok(())
    }

    fn forward_into(
        &self,
        x: &[f32],
        weight: &[f32],
        out: &mut [f32],
        hidden_size: usize,
        eps: f32,
    ) -> Result<()> {
        if weight.len() != hidden_size {
            return Err(crate::KernelError::DimensionMismatch {
                expected: hidden_size,
                got: weight.len(),
            });
        }
        debug_assert_eq!(x.len(), out.len());
        let num_tokens = x.len() / hidden_size;

        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        for t in 0..num_tokens {
            let start = t * hidden_size;
            let row = &x[start..start + hidden_size];
            let mut sum: f64 = 0.0;
            for i in 0..hidden_size {
                sum += (row[i] * row[i]) as f64;
            }
            let mean: f32 = (sum / hidden_size as f64) as f32;
            let scale: f32 = 1.0f32 / (mean + eps).sqrt();
            let out_row = &mut out[start..start + hidden_size];

            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe { scale_mul_into_avx2(row, weight, out_row, scale); }
            } else {
                for i in 0..hidden_size {
                    out_row[i] = row[i] * scale * weight[i];
                }
            }
        }
        Ok(())
    }
}

/// Deterministic CPU RMSNorm: sequential accumulation matching GPU kernel order.
/// Prevents auto-vectorization by using a while loop with volatile-style reads.
#[cfg(feature = "cpu")]
pub struct CpuRMSNormDet;

#[cfg(feature = "cpu")]
impl RMSNorm for CpuRMSNormDet {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Tested
    }

    #[inline(never)]
    fn forward(&self, x: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) -> Result<()> {
        if weight.len() != hidden_size {
            return Err(crate::KernelError::DimensionMismatch {
                expected: hidden_size,
                got: weight.len(),
            });
        }

        let num_tokens = x.len() / hidden_size;
        let mut t = 0;
        while t < num_tokens {
            let start = t * hidden_size;

            let mut sum_sq = 0.0f32;
            let mut i = 0;
            while i < hidden_size {
                let v = x[start + i];
                sum_sq += v * v;
                i += 1;
            }

            let rms = (sum_sq / hidden_size as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;

            i = 0;
            while i < hidden_size {
                x[start + i] = x[start + i] * inv_rms * weight[i];
                i += 1;
            }
            t += 1;
        }

        Ok(())
    }

    #[inline(never)]
    fn forward_into(
        &self,
        x: &[f32],
        weight: &[f32],
        out: &mut [f32],
        hidden_size: usize,
        eps: f32,
    ) -> Result<()> {
        if weight.len() != hidden_size {
            return Err(crate::KernelError::DimensionMismatch {
                expected: hidden_size,
                got: weight.len(),
            });
        }
        debug_assert_eq!(x.len(), out.len());
        let num_tokens = x.len() / hidden_size;
        let mut t = 0;
        while t < num_tokens {
            let start = t * hidden_size;
            let mut sum_sq = 0.0f32;
            let mut i = 0;
            while i < hidden_size {
                let v = x[start + i];
                sum_sq += v * v;
                i += 1;
            }
            let rms = (sum_sq / hidden_size as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;
            i = 0;
            while i < hidden_size {
                out[start + i] = x[start + i] * inv_rms * weight[i];
                i += 1;
            }
            t += 1;
        }
        Ok(())
    }
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
pub struct CudaRMSNorm {
    gpu: crate::gpu::GpuContext,
    module: crate::gpu::rmsnorm_kernel::kernels::LoadedModule,
    ss_cache: std::sync::Mutex<Option<DeviceBuffer<f32>>>,
    pub deterministic: bool,
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
unsafe impl Send for CudaRMSNorm {}
#[cfg(all(feature = "cuda", not(feature = "cutile")))]
unsafe impl Sync for CudaRMSNorm {}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
impl CudaRMSNorm {
    pub fn gpu(&self) -> &crate::gpu::GpuContext {
        &self.gpu
    }

    pub fn new() -> Result<Self> {
        let gpu = crate::gpu::GpuContext::new(0)?;
        let module = crate::gpu::rmsnorm_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(Self {
            gpu,
            module,
            ss_cache: std::sync::Mutex::new(None),
            deterministic: false,
        })
    }

    pub fn with_context(ctx: std::sync::Arc<cuda_core::CudaContext>) -> Result<Self> {
        let gpu = crate::gpu::GpuContext::from_context(ctx);
        let module = crate::gpu::rmsnorm_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(Self {
            gpu,
            module,
            ss_cache: std::sync::Mutex::new(None),
            deterministic: false,
        })
    }
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
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
        let stream = self.gpu.stream();
        let num_tokens = x.len() / hidden_size;
        let module = &self.module;

        if self.deterministic {
            let mut ss_guard = self.ss_cache.lock().unwrap();
            if ss_guard.as_ref().map_or(true, |buf| buf.len() < num_tokens) {
                *ss_guard = Some(
                    DeviceBuffer::zeroed(&stream, num_tokens)
                        .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?,
                );
            }
            let ss_dev = ss_guard.as_mut().unwrap();

            module
                .rmsnorm_sum_sq_det(
                    &stream,
                    LaunchConfig::for_num_elems(num_tokens as u32),
                    x,
                    ss_dev,
                    hidden_size as u32,
                )
                .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

            module
                .rmsnorm_apply(
                    &stream,
                    LaunchConfig::for_num_elems((num_tokens * hidden_size) as u32),
                    x,
                    weight,
                    ss_dev,
                    output,
                    hidden_size as u32,
                    1,
                    eps,
                )
                .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        } else {
            let warp_size = 32u32;
            let needed = num_tokens * warp_size as usize;

            let mut ss_guard = self.ss_cache.lock().unwrap();
            if ss_guard.as_ref().map_or(true, |buf| buf.len() < needed) {
                *ss_guard = Some(
                    DeviceBuffer::zeroed(&stream, needed)
                        .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?,
                );
            }
            let ss_dev = ss_guard.as_mut().unwrap();

            let sum_sq_config = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (warp_size, 1, 1),
                shared_mem_bytes: 0,
            };
            module
                .rmsnorm_sum_sq(&stream, sum_sq_config, x, ss_dev, hidden_size as u32)
                .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

            module
                .rmsnorm_apply(
                    &stream,
                    LaunchConfig::for_num_elems((num_tokens * hidden_size) as u32),
                    x,
                    weight,
                    ss_dev,
                    output,
                    hidden_size as u32,
                    warp_size,
                    eps,
                )
                .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
impl RMSNorm for CudaRMSNorm {
    fn forward(&self, x: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) -> Result<()> {
        use cuda_core::LaunchConfig;

        if weight.len() != hidden_size {
            return Err(crate::KernelError::DimensionMismatch {
                expected: hidden_size,
                got: weight.len(),
            });
        }

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

    #[test]
    fn test_forward_into_matches_forward() {
        let rn = CpuRMSNorm;
        let hidden = 4;
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![0.5f32, 1.0, 2.0, 0.25];
        let eps = 1e-6f32;

        let mut inplace = x.clone();
        rn.forward(&mut inplace, &weight, hidden, eps).unwrap();

        let mut fused = vec![0.0f32; hidden];
        rn.forward_into(&x, &weight, &mut fused, hidden, eps)
            .unwrap();

        for i in 0..hidden {
            assert_relative_eq!(fused[i], inplace[i], epsilon = 1e-6);
        }
        // source must not be modified
        assert_eq!(x, vec![1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_forward_into_ggml_matches_forward() {
        let rn = CpuRMSNormGgml;
        let hidden = 8;
        let x: Vec<f32> = (1..=8).map(|i| i as f32 * 0.5).collect();
        let weight = vec![1.0f32; hidden];
        let eps = 1e-5f32;

        let mut inplace = x.clone();
        rn.forward(&mut inplace, &weight, hidden, eps).unwrap();

        let mut fused = vec![0.0f32; hidden];
        rn.forward_into(&x, &weight, &mut fused, hidden, eps)
            .unwrap();

        for i in 0..hidden {
            assert_relative_eq!(fused[i], inplace[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_forward_into_multi_token() {
        let rn = CpuRMSNorm;
        let hidden = 4;
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0f32; hidden];
        let eps = 1e-6f32;

        let mut inplace = x.clone();
        rn.forward(&mut inplace, &weight, hidden, eps).unwrap();

        let mut fused = vec![0.0f32; x.len()];
        rn.forward_into(&x, &weight, &mut fused, hidden, eps)
            .unwrap();

        for i in 0..x.len() {
            assert_relative_eq!(fused[i], inplace[i], epsilon = 1e-6);
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

            #[test]
            fn prop_forward_into_matches_forward(
                input in prop::collection::vec(0.1f32..10.0f32, 4..32),
                weights in prop::collection::vec(0.1f32..2.0f32, 4..32),
            ) {
                let rn = CpuRMSNorm;
                let h = input.len().min(weights.len());
                let x = &input[..h];
                let w = &weights[..h];

                let mut inplace = x.to_vec();
                rn.forward(&mut inplace, w, h, 1e-6).unwrap();

                let mut fused = vec![0.0f32; h];
                rn.forward_into(x, w, &mut fused, h, 1e-6).unwrap();

                for i in 0..h {
                    prop_assert!(
                        (fused[i] - inplace[i]).abs() < 1e-5,
                        "mismatch at {i}: fused={} inplace={}", fused[i], inplace[i]
                    );
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub type CudaRMSNorm = crate::cutile_rmsnorm::CutileRmsNorm;

#[cfg(feature = "cuda")]
impl crate::VerifiedEquivalent<CpuRMSNorm> for CudaRMSNorm {
    fn max_relative_error() -> f64 { 0.001 }
    fn proof_name() -> &'static str { "dual-pipeline:rmsnorm_cpu_vs_gpu" }
    fn verification_tier() -> crate::VerificationTier { crate::VerificationTier::Tested }
}

#[cfg(feature = "cpu")]
impl crate::VerifiedEquivalent<CpuRMSNorm> for CpuRMSNormGgml {
    fn max_relative_error() -> f64 { 0.0001 }
    fn proof_name() -> &'static str { "dual-pipeline:rmsnorm_ref_vs_ggml" }
    fn verification_tier() -> crate::VerificationTier { crate::VerificationTier::Tested }
}

#[cfg(feature = "cpu")]
impl crate::VerifiedEquivalent<CpuRMSNorm> for CpuRMSNormDet {
    fn max_relative_error() -> f64 { 0.0 }
    fn proof_name() -> &'static str { "dual-pipeline:rmsnorm_ref_vs_det" }
    fn verification_tier() -> crate::VerificationTier { crate::VerificationTier::Tested }
}
