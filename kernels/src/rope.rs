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
