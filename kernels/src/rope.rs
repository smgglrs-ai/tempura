use crate::Result;

#[allow(clippy::too_many_arguments)]
pub trait RoPE: Send + Sync {
    /// Apply rotary position embeddings in-place.
    ///
    /// - `x`: [batch, seq_len, num_heads, head_dim]
    /// - `positions`: [batch, seq_len] — position indices
    /// - `theta`: base frequency (typically 10000.0)
    fn forward(
        &self,
        x: &mut [f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        positions: &[u32],
        theta: f32,
    ) -> Result<()>;
}

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
