use crate::Result;

#[allow(clippy::too_many_arguments)]
pub trait Attention: Send + Sync {
    /// Compute scaled dot-product attention.
    ///
    /// - `q`: [batch, seq_len, num_heads, head_dim]
    /// - `k`: [batch, kv_len, num_kv_heads, head_dim]
    /// - `v`: [batch, kv_len, num_kv_heads, head_dim]
    /// - `output`: [batch, seq_len, num_heads, head_dim]
    /// - `scale`: typically 1/sqrt(head_dim)
    fn forward(
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
    ) -> Result<()>;
}

#[cfg(feature = "cpu")]
pub struct CpuAttention;

#[cfg(feature = "cpu")]
impl Attention for CpuAttention {
    fn forward(
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
    ) -> Result<()> {
        let heads_per_group = num_heads / num_kv_heads;

        for b in 0..batch {
            for h in 0..num_heads {
                let kv_h = h / heads_per_group;
                for s in 0..seq_len {
                    let q_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                    let q_vec = &q[q_offset..q_offset + head_dim];

                    let mut scores = vec![0.0f32; kv_len];
                    for (kv_s, score) in scores.iter_mut().enumerate() {
                        let k_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim;
                        let k_vec = &k[k_offset..k_offset + head_dim];
                        let dot: f32 = q_vec.iter().zip(k_vec).map(|(a, b)| a * b).sum();
                        *score = dot * scale;
                    }

                    // softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_score).exp();
                        sum += *s;
                    }
                    for s in &mut scores {
                        *s /= sum;
                    }

                    // weighted sum of values
                    let o_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                    for d in 0..head_dim {
                        let mut val = 0.0f32;
                        for (kv_s, &score) in scores.iter().enumerate() {
                            let v_offset =
                                ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim + d;
                            val += score * v[v_offset];
                        }
                        output[o_offset + d] = val;
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub struct CudaAttention {
    gpu: crate::gpu::GpuContext,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaAttention {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaAttention {}

#[cfg(feature = "cuda")]
impl CudaAttention {
    pub fn new() -> Result<Self> {
        Ok(Self {
            gpu: crate::gpu::GpuContext::new(0)?,
        })
    }
}

#[cfg(feature = "cuda")]
impl Attention for CudaAttention {
    fn forward(
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
    ) -> Result<()> {
        use cuda_core::LaunchConfig;

        let stream = self.gpu.stream();
        let total = batch * seq_len * num_heads * head_dim;

        let q_dev = crate::gpu::to_device(&stream, q)?;
        let k_dev = crate::gpu::to_device(&stream, k)?;
        let v_dev = crate::gpu::to_device(&stream, v)?;
        let mut out_dev = crate::gpu::to_device(&stream, &vec![0.0f32; total])?;

        let module = crate::gpu::attention_kernel::kernels::load(self.gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

        module
            .attention_forward(
                &stream,
                LaunchConfig::for_num_elems(total as u32),
                &q_dev,
                &k_dev,
                &v_dev,
                &mut out_dev,
                batch as u32,
                seq_len as u32,
                kv_len as u32,
                num_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                scale,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

        crate::gpu::copy_back(&stream, &out_dev, output)?;
        Ok(())
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn attention_q_offset_in_bounds() {
        let batch: usize = kani::any();
        let seq_len: usize = kani::any();
        let num_heads: usize = kani::any();
        let head_dim: usize = kani::any();

        kani::assume(batch >= 1 && batch <= 2);
        kani::assume(seq_len >= 1 && seq_len <= 4);
        kani::assume(num_heads >= 1 && num_heads <= 4);
        kani::assume(head_dim >= 1 && head_dim <= 4);

        let q_len = batch * seq_len * num_heads * head_dim;

        let b: usize = kani::any();
        let s: usize = kani::any();
        let h: usize = kani::any();
        kani::assume(b < batch);
        kani::assume(s < seq_len);
        kani::assume(h < num_heads);

        let q_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
        assert!(q_offset + head_dim <= q_len);
    }

    #[kani::proof]
    fn attention_kv_offset_in_bounds() {
        let batch: usize = kani::any();
        let kv_len: usize = kani::any();
        let num_kv_heads: usize = kani::any();
        let head_dim: usize = kani::any();

        kani::assume(batch >= 1 && batch <= 2);
        kani::assume(kv_len >= 1 && kv_len <= 4);
        kani::assume(num_kv_heads >= 1 && num_kv_heads <= 4);
        kani::assume(head_dim >= 1 && head_dim <= 4);

        let k_len = batch * kv_len * num_kv_heads * head_dim;

        let b: usize = kani::any();
        let kv_s: usize = kani::any();
        let kv_h: usize = kani::any();
        kani::assume(b < batch);
        kani::assume(kv_s < kv_len);
        kani::assume(kv_h < num_kv_heads);

        let k_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim;
        assert!(k_offset + head_dim <= k_len);
    }

    #[kani::proof]
    fn attention_batch_isolation_no_overlap() {
        let batch: usize = kani::any();
        let seq_len: usize = kani::any();
        let num_heads: usize = kani::any();
        let head_dim: usize = kani::any();

        kani::assume(batch >= 2 && batch <= 4);
        kani::assume(seq_len >= 1 && seq_len <= 2);
        kani::assume(num_heads >= 1 && num_heads <= 2);
        kani::assume(head_dim >= 1 && head_dim <= 4);

        let b1: usize = kani::any();
        let b2: usize = kani::any();
        kani::assume(b1 < batch);
        kani::assume(b2 < batch);
        kani::assume(b1 != b2);

        let stride = seq_len * num_heads * head_dim;
        let start1 = b1 * stride;
        let end1 = start1 + stride;
        let start2 = b2 * stride;
        let end2 = start2 + stride;

        assert!(end1 <= start2 || end2 <= start1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_head_self_attention() {
        let attn = CpuAttention;
        let head_dim = 4;
        let seq_len = 2;
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = q.clone();
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; seq_len * head_dim];

        attn.forward(
            &q,
            &k,
            &v,
            &mut output,
            1,
            seq_len,
            seq_len,
            1,
            1,
            head_dim,
            1.0 / (head_dim as f32).sqrt(),
        )
        .unwrap();

        // output should be weighted combination of v rows
        assert_eq!(output.len(), 8);
        // first token attends more to itself (q[0] dot k[0] > q[0] dot k[1])
        assert!(output[0] < 3.0); // closer to v[0]=1 than v[4]=5
    }
}
