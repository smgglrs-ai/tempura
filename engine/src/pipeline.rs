use oxide_kernels::{AllReduce, Attention, MatMul, RMSNorm, RoPE, Sampler};

use crate::trace::{LayerSnapshot, NormStats, TraceLevel};

pub struct KernelPipeline<A, N, R, M, S, AR = oxide_kernels::allreduce::CpuAllReduce> {
    pub attention: A,
    pub rmsnorm: N,
    pub rope: R,
    pub matmul: M,
    pub sampler: S,
    pub allreduce: AR,
}

impl
    KernelPipeline<
        oxide_kernels::attention::CpuAttention,
        oxide_kernels::rmsnorm::CpuRMSNorm,
        oxide_kernels::rope::CpuRoPE,
        oxide_kernels::matmul::CpuMatMul,
        oxide_kernels::sampling::CpuSampler,
    >
{
    pub fn cpu_verified() -> Self {
        use oxide_kernels::allreduce::CpuAllReduce;
        use oxide_kernels::attention::CpuAttention;
        use oxide_kernels::matmul::CpuMatMul;
        use oxide_kernels::rmsnorm::CpuRMSNorm;
        use oxide_kernels::rope::CpuRoPE;
        use oxide_kernels::sampling::CpuSampler;

        Self {
            attention: CpuAttention,
            rmsnorm: CpuRMSNorm,
            rope: CpuRoPE,
            matmul: CpuMatMul,
            sampler: CpuSampler,
            allreduce: CpuAllReduce,
        }
    }
}

#[cfg(feature = "cuda")]
impl
    KernelPipeline<
        oxide_kernels::attention::CudaAttention,
        oxide_kernels::rmsnorm::CudaRMSNorm,
        oxide_kernels::rope::CudaRoPE,
        oxide_kernels::matmul::CudaMatMul,
        oxide_kernels::sampling::CudaSampler,
        oxide_kernels::allreduce::CpuAllReduce,
    >
{
    pub fn cuda() -> Self {
        use oxide_kernels::allreduce::CpuAllReduce;
        use oxide_kernels::attention::CudaAttention;
        use oxide_kernels::matmul::CudaMatMul;
        use oxide_kernels::rmsnorm::CudaRMSNorm;
        use oxide_kernels::rope::CudaRoPE;
        use oxide_kernels::sampling::CudaSampler;

        Self {
            attention: CudaAttention::new().expect("Failed to initialize CudaAttention"),
            rmsnorm: CudaRMSNorm::new().expect("Failed to initialize CudaRMSNorm"),
            rope: CudaRoPE::new().expect("Failed to initialize CudaRoPE"),
            matmul: CudaMatMul::new().expect("Failed to initialize CudaMatMul"),
            sampler: CudaSampler,
            allreduce: CpuAllReduce,
        }
    }
}

impl<A, N, R, M, S, AR> KernelPipeline<A, N, R, M, S, AR>
where
    A: Attention,
    N: RMSNorm,
    R: RoPE,
    M: MatMul,
    S: Sampler,
    AR: AllReduce,
{
    pub fn allreduce_sum(&self, data: &mut [f32]) -> oxide_kernels::Result<()> {
        self.allreduce.sum_in_place(data)
    }

    pub fn rmsnorm(
        &self,
        x: &mut [f32],
        weight: &[f32],
        hidden_size: usize,
        eps: f32,
        trace_level: TraceLevel,
        snapshot: &mut Option<&mut LayerSnapshot>,
    ) -> oxide_kernels::Result<()> {
        if trace_level >= TraceLevel::Summary {
            if let Some(ref mut snap) = snapshot {
                let mean = x.iter().sum::<f32>() / x.len() as f32;
                let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
                let max_abs = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                snap.pre_norm = Some(NormStats {
                    mean,
                    variance: var,
                    max_abs,
                });
            }
        }

        self.rmsnorm.forward(x, weight, hidden_size, eps)?;

        if trace_level >= TraceLevel::Summary {
            if let Some(ref mut snap) = snapshot {
                let mean = x.iter().sum::<f32>() / x.len() as f32;
                let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
                let max_abs = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                snap.post_norm = Some(NormStats {
                    mean,
                    variance: var,
                    max_abs,
                });
            }
        }

        Ok(())
    }

    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> oxide_kernels::Result<()> {
        self.matmul.forward(a, b, output, m, n, k)
    }

    /// Matrix multiply with B transposed: output = a × bᵀ
    /// b is [n, k] row-major (PyTorch Linear weight convention)
    pub fn matmul_bt(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> oxide_kernels::Result<()> {
        self.matmul.forward_b_transposed(a, b, output, m, n, k)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &mut [f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        positions: &[u32],
        theta: f32,
    ) -> oxide_kernels::Result<()> {
        self.rope
            .forward(x, batch, seq_len, num_heads, head_dim, positions, theta)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
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
    ) -> oxide_kernels::Result<()> {
        self.attention.forward(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_verified_factory() {
        let pipeline = KernelPipeline::cpu_verified();
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        pipeline
            .rmsnorm(&mut x, &weight, 4, 1e-6, TraceLevel::Off, &mut None)
            .unwrap();

        let rms: f32 = (x.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_rmsnorm_with_trace() {
        let pipeline = KernelPipeline::cpu_verified();
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let mut snapshot = LayerSnapshot {
            layer_index: 0,
            pre_norm: None,
            post_norm: None,
            attention_scores: None,
            duration: None,
        };

        pipeline
            .rmsnorm(
                &mut x,
                &weight,
                4,
                1e-6,
                TraceLevel::Summary,
                &mut Some(&mut snapshot),
            )
            .unwrap();

        assert!(snapshot.pre_norm.is_some());
        assert!(snapshot.post_norm.is_some());
    }

    #[test]
    fn test_matmul_delegation() {
        let pipeline = KernelPipeline::cpu_verified();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 4];

        pipeline.matmul(&a, &b, &mut output, 2, 2, 2).unwrap();
        assert_eq!(output, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
