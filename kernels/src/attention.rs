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
