use tempura_kernels::{AllReduce, Attention, MatMul, RMSNorm, RoPE, Sampler, VerificationTier};

use crate::trace::{LayerSnapshot, NormStats, TraceLevel};

pub struct KernelPipeline<A, N, R, M, S, AR = tempura_kernels::allreduce::CpuAllReduce> {
    pub attention: A,
    pub rmsnorm: N,
    pub rope: R,
    pub matmul: M,
    pub sampler: S,
    pub allreduce: AR,
}

impl
    KernelPipeline<
        tempura_kernels::attention::CpuAttention,
        tempura_kernels::rmsnorm::CpuRMSNorm,
        tempura_kernels::rope::CpuRoPE,
        tempura_kernels::matmul::CpuMatMul,
        tempura_kernels::sampling::CpuSampler,
    >
{
    pub fn cpu_verified() -> Self {
        use tempura_kernels::allreduce::CpuAllReduce;
        use tempura_kernels::attention::CpuAttention;
        use tempura_kernels::matmul::CpuMatMul;
        use tempura_kernels::rmsnorm::CpuRMSNorm;
        use tempura_kernels::rope::CpuRoPE;
        use tempura_kernels::sampling::CpuSampler;

        Self {
            attention: CpuAttention,
            rmsnorm: CpuRMSNorm,
            rope: CpuRoPE,
            matmul: CpuMatMul,
            sampler: CpuSampler,
            allreduce: CpuAllReduce,
        }
    }

    /// Production CPU pipeline -- same kernels as cpu_verified() today; diverges as
    /// ARM NEON/SVE and other native backends are added. Quantized matmul dispatch
    /// (Q4K, Q6K) uses AVX2/VNNI via CpuMatMulQ4KOpt, wired in forward_quantized.
    pub fn cpu_native() -> Self {
        Self::cpu_verified()
    }
}

#[cfg(feature = "cutile")]
impl
    KernelPipeline<
        tempura_kernels::cutile_flash_attention::CutileFlashAttention,
        tempura_kernels::cutile_rmsnorm::CutileRmsNorm,
        tempura_kernels::cutile_rope::CutileRoPE,
        tempura_kernels::cutile_matmul::CutileMatMul,
        tempura_kernels::sampling::CpuSampler,
        tempura_kernels::allreduce::CpuAllReduce,
    >
{
    pub fn cutile() -> KernelPipeline<
        tempura_kernels::cutile_flash_attention::CutileFlashAttention,
        tempura_kernels::cutile_rmsnorm::CutileRmsNorm,
        tempura_kernels::cutile_rope::CutileRoPE,
        tempura_kernels::cutile_matmul::CutileMatMul,
        tempura_kernels::sampling::CpuSampler,
    > {
        use tempura_kernels::allreduce::CpuAllReduce;
        use tempura_kernels::cutile_flash_attention::CutileFlashAttention;
        use tempura_kernels::cutile_matmul::CutileMatMul;
        use tempura_kernels::cutile_rmsnorm::CutileRmsNorm;
        use tempura_kernels::cutile_rope::CutileRoPE;
        use tempura_kernels::sampling::CpuSampler;

        KernelPipeline {
            attention: CutileFlashAttention::new()
                .expect("Failed to initialize CutileFlashAttention"),
            rmsnorm: CutileRmsNorm::new().expect("Failed to initialize CutileRmsNorm"),
            rope: CutileRoPE::new().expect("Failed to initialize CutileRoPE"),
            matmul: CutileMatMul::new().expect("Failed to initialize CutileMatMul"),
            sampler: CpuSampler,
            allreduce: CpuAllReduce,
        }
    }
}

#[cfg(feature = "cuda")]
impl
    KernelPipeline<
        tempura_kernels::attention::CudaAttention,
        tempura_kernels::rmsnorm::CudaRMSNorm,
        tempura_kernels::rope::CudaRoPE,
        tempura_kernels::matmul::CudaMatMul,
        tempura_kernels::sampling::CudaSampler,
        tempura_kernels::allreduce::CpuAllReduce,
    >
{
    pub fn cuda() -> KernelPipeline<
        tempura_kernels::attention::CudaAttention,
        tempura_kernels::rmsnorm::CudaRMSNorm,
        tempura_kernels::rope::CudaRoPE,
        tempura_kernels::matmul::CudaMatMul,
        tempura_kernels::sampling::CpuSampler,
    > {
        use tempura_kernels::allreduce::CpuAllReduce;
        use tempura_kernels::attention::CudaAttention;
        use tempura_kernels::matmul::CudaMatMul;
        use tempura_kernels::rmsnorm::CudaRMSNorm;
        use tempura_kernels::rope::CudaRoPE;
        use tempura_kernels::sampling::CpuSampler;

        KernelPipeline {
            attention: CudaAttention::new().expect("Failed to initialize CudaAttention"),
            rmsnorm: CudaRMSNorm::new().expect("Failed to initialize CudaRMSNorm"),
            rope: CudaRoPE::new().expect("Failed to initialize CudaRoPE"),
            matmul: CudaMatMul::new().expect("Failed to initialize CudaMatMul"),
            sampler: CpuSampler,
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
    pub fn verification_tier(&self) -> VerificationTier {
        [
            self.attention.verification_tier(),
            self.rmsnorm.verification_tier(),
            self.rope.verification_tier(),
            self.matmul.verification_tier(),
            self.sampler.verification_tier(),
            self.allreduce.verification_tier(),
        ]
        .into_iter()
        .min()
        .unwrap_or(VerificationTier::Compiled)
    }

    pub fn allreduce_sum(&self, data: &mut [f32]) -> tempura_kernels::Result<()> {
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
    ) -> tempura_kernels::Result<()> {
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

    /// Fused RMSNorm + elementwise multiply, source -> output.
    /// Reads from `x`, writes to `out` without an intermediate copy.
    pub fn rmsnorm_into(
        &self,
        x: &[f32],
        weight: &[f32],
        out: &mut [f32],
        hidden_size: usize,
        eps: f32,
        trace_level: TraceLevel,
        snapshot: &mut Option<&mut LayerSnapshot>,
    ) -> tempura_kernels::Result<()> {
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

        self.rmsnorm
            .forward_into(x, weight, out, hidden_size, eps)?;

        if trace_level >= TraceLevel::Summary {
            if let Some(ref mut snap) = snapshot {
                let mean = out.iter().sum::<f32>() / out.len() as f32;
                let var =
                    out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / out.len() as f32;
                let max_abs = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
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
    ) -> tempura_kernels::Result<()> {
        self.matmul.forward(a, b, output, m, n, k)
    }

    /// Matrix multiply with B transposed: output = a x b???
    /// b is [n, k] row-major (PyTorch Linear weight convention)
    pub fn matmul_bt(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> tempura_kernels::Result<()> {
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
    ) -> tempura_kernels::Result<()> {
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
    ) -> tempura_kernels::Result<()> {
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

/// Split `total` rows into batch chunks within `[min, cap]`.
///
/// For kernels whose single launch accepts at most `cap` rows and rejects
/// fewer than `min` -- e.g. the q6k sm120 batch GEMV (2..=16 rows; its
/// shared-memory reduction area is sized by the row count), fed with
/// `capped_batch_chunks(m, 2, 16)` by the GPU prefill path. Full `cap`-row
/// chunks first; whenever the remaining tail would drop below `min`, take
/// `rem - min` instead so the final chunk lands exactly on `min` and every
/// chunk stays in range while the sizes sum to `total`.
///
/// # Panics
/// Panics when `total < min` (fewer rows than the smallest legal chunk --
/// callers must route those to a different kernel) or `cap < 2 * min - 1`
/// (the split-early step needs both halves >= min).
pub fn capped_batch_chunks(total: usize, min: usize, cap: usize) -> Vec<usize> {
    assert!(min >= 1, "min chunk size must be >= 1");
    assert!(cap >= min, "cap must be >= min");
    assert!(cap >= 2 * min - 1, "cap too small for the split-early tail");
    assert!(total >= min, "total < min: route to a smaller-row kernel instead");
    let mut chunks = Vec::new();
    let mut rem = total;
    while rem > cap {
        // rem - cap < min means a plain cap-sized chunk would strand a tail
        // below min; take rem - min now so the tail lands exactly on min.
        let take = if rem - cap < min { rem - min } else { cap };
        chunks.push(take);
        rem -= take;
    }
    chunks.push(rem);
    chunks
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

    /// Every chunk must be in `[min, cap]` and the chunks must sum to `total`.
    fn chunks_valid(total: usize, min: usize, cap: usize) -> bool {
        let chunks = capped_batch_chunks(total, min, cap);
        !chunks.is_empty()
            && chunks.iter().all(|&c| c >= min && c <= cap)
            && chunks.iter().sum::<usize>() == total
    }

    #[test]
    fn capped_batch_chunks_single_chunk_within_kernel_range() {
        for m in 2..=16 {
            assert_eq!(capped_batch_chunks(m, 2, 16), vec![m]);
        }
    }

    #[test]
    fn capped_batch_chunks_split_early_tail() {
        // 17 = 16 + 1 would strand a 1-row tail; splits as 15 + 2 instead.
        assert_eq!(capped_batch_chunks(17, 2, 16), vec![15, 2]);
        assert_eq!(capped_batch_chunks(33, 2, 16), vec![16, 15, 2]);
        assert_eq!(capped_batch_chunks(512, 2, 16), vec![16; 32]);
    }

    #[test]
    fn capped_batch_chunks_sum_to_total_for_all_relevant_sizes() {
        // Engine prefill chunks are 512 tokens; bench uses 128; arbitrary
        // prompt tails can be any length. Also covers larger min values.
        for m in 2..=1024 {
            assert!(chunks_valid(m, 2, 16), "chunk scheme invalid at m={m}");
        }
        for m in 3..=256 {
            assert!(chunks_valid(m, 3, 8), "chunk scheme invalid at m={m} (3,8)");
        }
    }

    #[test]
    #[should_panic(expected = "total < min")]
    fn capped_batch_chunks_rejects_below_min() {
        capped_batch_chunks(1, 2, 16);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::capped_batch_chunks;

    /// For any total/min/cap satisfying the preconditions, every chunk lands
    /// in [min, cap] and the chunk sizes sum to exactly `total` -- no stranded
    /// below-min tail (out-of-range kernel launch) and no lost rows.
    #[kani::proof]
    #[kani::unwind(14)]
    fn capped_batch_chunks_valid_for_all_totals() {
        let min: usize = kani::any();
        let cap: usize = kani::any();
        let total: usize = kani::any();
        kani::assume(min >= 2 && min <= 3);
        kani::assume(cap >= 2 * min - 1 && cap <= 8);
        kani::assume(total >= min && total <= 20);

        let chunks = capped_batch_chunks(total, min, cap);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|&c| c >= min && c <= cap));
        assert_eq!(chunks.iter().sum::<usize>(), total);
    }
}
