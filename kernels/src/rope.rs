use crate::Result;
pub use crate::RoPE;

#[cfg(feature = "cuda")]
use cuda_core::DeviceBuffer;

#[cfg(feature = "cpu")]
pub struct CpuRoPE;

#[cfg(feature = "cpu")]
impl RoPE for CpuRoPE {
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
        let half_dim = head_dim / 2;

        for b in 0..batch {
            for s in 0..seq_len {
                let pos = positions[b * seq_len + s] as f32;
                for h in 0..num_heads {
                    let offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                    for d in 0..half_dim {
                        let freq = pos / theta.powf(2.0 * d as f32 / head_dim as f32);
                        let cos_val = freq.cos();
                        let sin_val = freq.sin();

                        let x0 = x[offset + d];
                        let x1 = x[offset + d + half_dim];
                        x[offset + d] = x0 * cos_val - x1 * sin_val;
                        x[offset + d + half_dim] = x0 * sin_val + x1 * cos_val;
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub struct CudaRoPE {
    gpu: crate::gpu::GpuContext,
    module: crate::gpu::rope_kernel::kernels::LoadedModule,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaRoPE {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaRoPE {}

#[cfg(feature = "cuda")]
impl CudaRoPE {
    pub fn new() -> Result<Self> {
        let gpu = crate::gpu::GpuContext::new(0)?;
        let module = crate::gpu::rope_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(Self { gpu, module })
    }
}

#[cfg(feature = "cuda")]
impl CudaRoPE {
    pub fn forward_device(
        &self,
        x_in: &DeviceBuffer<f32>,
        x_out: &mut DeviceBuffer<f32>,
        positions: &DeviceBuffer<u32>,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        theta: f32,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();
        let total = seq_len * num_heads * head_dim;
        let module = &self.module;
        module
            .rope_forward(
                &stream,
                LaunchConfig::for_num_elems(total as u32),
                x_in,
                positions,
                x_out,
                seq_len as u32,
                num_heads as u32,
                head_dim as u32,
                theta,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl RoPE for CudaRoPE {
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
        use cuda_core::LaunchConfig;

        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();
        let total_elems = batch * seq_len * num_heads * head_dim;

        let x_dev = crate::gpu::to_device(&stream, x)?;
        let pos_dev = crate::gpu::to_device_u32(&stream, positions)?;
        let mut out_dev = crate::gpu::to_device(&stream, &vec![0.0f32; total_elems])?;

        let module = &self.module;

        for b in 0..batch {
            let b_offset = b * seq_len * num_heads * head_dim;
            let b_pos_offset = b * seq_len;
            let b_elems = seq_len * num_heads * head_dim;

            if batch == 1 {
                module
                    .rope_forward(
                        &stream,
                        LaunchConfig::for_num_elems(b_elems as u32),
                        &x_dev,
                        &pos_dev,
                        &mut out_dev,
                        seq_len as u32,
                        num_heads as u32,
                        head_dim as u32,
                        theta,
                    )
                    .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
            } else {
                let b_x = &mut x[b_offset..b_offset + b_elems];
                let b_pos = &positions[b_pos_offset..b_pos_offset + seq_len];
                let cpu = CpuRoPE;
                cpu.forward(b_x, 1, seq_len, num_heads, head_dim, b_pos, theta)?;
            }
        }

        if batch == 1 {
            crate::gpu::copy_back(&stream, &out_dev, x)?;
        }

        Ok(())
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn rope_indices_in_bounds() {
        let batch: usize = kani::any();
        let seq_len: usize = kani::any();
        let num_heads: usize = kani::any();
        let head_dim: usize = kani::any();

        kani::assume(batch >= 1 && batch <= 2);
        kani::assume(seq_len >= 1 && seq_len <= 4);
        kani::assume(num_heads >= 1 && num_heads <= 4);
        kani::assume(head_dim >= 2 && head_dim <= 8);
        kani::assume(head_dim % 2 == 0);

        let x_len = batch * seq_len * num_heads * head_dim;
        let pos_len = batch * seq_len;
        let half_dim = head_dim / 2;

        let b: usize = kani::any();
        let s: usize = kani::any();
        let h: usize = kani::any();
        let d: usize = kani::any();
        kani::assume(b < batch);
        kani::assume(s < seq_len);
        kani::assume(h < num_heads);
        kani::assume(d < half_dim);

        assert!(b * seq_len + s < pos_len);
        let offset = ((b * seq_len + s) * num_heads + h) * head_dim;
        assert!(offset + d < x_len);
        assert!(offset + d + half_dim < x_len);
    }

    #[kani::proof]
    fn gpu_rope_thread_decomposition_in_bounds() {
        let seq_len: usize = kani::any();
        let num_heads: usize = kani::any();
        let head_dim: usize = kani::any();

        kani::assume(seq_len >= 1 && seq_len <= 4);
        kani::assume(num_heads >= 1 && num_heads <= 4);
        kani::assume(head_dim >= 2 && head_dim <= 4);
        kani::assume(head_dim % 2 == 0);

        let total = seq_len * num_heads * head_dim;
        let x_len = total;

        let i: usize = kani::any();
        kani::assume(i < total);

        let hd = head_dim;
        let d_local = i % hd;
        let h = (i / hd) % num_heads;
        let s = i / hd / num_heads;

        assert!(s < seq_len);
        assert!(h < num_heads);
        assert!(d_local < head_dim);

        let offset = (s * num_heads + h) * hd;
        assert!(offset + d_local < x_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rope_matches_cpu() {
        let cpu = CpuRoPE;
        let gpu = CudaRoPE::new().unwrap();

        let head_dim = 16;
        let seq_len = 4;
        let num_heads = 4;
        let theta = 10000.0f32;
        let n = seq_len * num_heads * head_dim;

        let x_orig: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let mut x_cpu = x_orig.clone();
        cpu.forward(
            &mut x_cpu, 1, seq_len, num_heads, head_dim, &positions, theta,
        )
        .unwrap();

        let mut x_gpu = x_orig.clone();
        gpu.forward(
            &mut x_gpu, 1, seq_len, num_heads, head_dim, &positions, theta,
        )
        .unwrap();

        for i in 0..n {
            let diff = (x_cpu[i] - x_gpu[i]).abs();
            assert!(
                diff < 1e-4,
                "CPU vs GPU mismatch at index {i}: cpu={}, gpu={}, diff={diff}",
                x_cpu[i],
                x_gpu[i]
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rope_position_zero_identity() {
        let gpu = CudaRoPE::new().unwrap();
        let head_dim = 8;
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = original.clone();
        let positions = vec![0u32];

        gpu.forward(&mut x, 1, 1, 1, head_dim, &positions, 10000.0)
            .unwrap();

        for (i, (a, b)) in x.iter().zip(original.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "Position-zero should be identity, but index {i}: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn test_rope_position_zero_is_identity() {
        let rope = CpuRoPE;
        let head_dim = 4;
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = original.clone();
        let positions = vec![0u32];

        rope.forward(&mut x, 1, 1, 1, head_dim, &positions, 10000.0)
            .unwrap();

        // at position 0, cos=1 sin=0, so output should equal input
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_rotation_preserves_l2_norm(
                head_dim in (2usize..8usize).prop_filter("must be even", |d| d % 2 == 0),
                position in 0u32..128u32,
            ) {
                let rope = CpuRoPE;
                let mut data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
                let l2_before: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
                rope.forward(&mut data, 1, 1, 1, head_dim, &[position], 10000.0).unwrap();
                let l2_after: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
                prop_assert!((l2_before - l2_after).abs() < 1e-4,
                    "L2 changed: {l2_before} -> {l2_after}");
            }

            #[test]
            fn prop_position_zero_is_identity(
                head_dim in (2usize..8usize).prop_filter("must be even", |d| d % 2 == 0),
            ) {
                let rope = CpuRoPE;
                let original: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
                let mut data = original.clone();
                rope.forward(&mut data, 1, 1, 1, head_dim, &[0], 10000.0).unwrap();
                for (i, (a, b)) in original.iter().zip(data.iter()).enumerate() {
                    prop_assert!((a - b).abs() < 1e-5,
                        "position 0 should be identity, diff at {i}: {a} vs {b}");
                }
            }
        }
    }
}
