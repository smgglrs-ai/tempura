use crate::Result;

pub trait LayerNorm: Send + Sync {
    /// Apply layer normalization in-place.
    ///
    /// - `x`: [batch * seq_len, hidden_size] (flattened)
    /// - `weight`: [hidden_size] (gamma)
    /// - `bias`: [hidden_size] (beta), optional
    /// - `eps`: small constant for numerical stability
    fn forward(
        &self,
        x: &mut [f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        hidden_size: usize,
        eps: f32,
    ) -> Result<()>;
}

#[cfg(feature = "cpu")]
pub struct CpuLayerNorm;

#[cfg(feature = "cpu")]
impl LayerNorm for CpuLayerNorm {
    fn forward(
        &self,
        x: &mut [f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        hidden_size: usize,
        eps: f32,
    ) -> Result<()> {
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

            let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;
            let var: f32 =
                row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden_size as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            for (i, val) in row.iter_mut().enumerate() {
                let normed = (*val - mean) * inv_std;
                *val = normed * weight[i] + bias.map_or(0.0, |b| b[i]);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_identity_weights() {
        let ln = CpuLayerNorm;
        let hidden = 4;
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; hidden];

        ln.forward(&mut x, &weight, None, hidden, 1e-5).unwrap();

        let sum: f32 = x.iter().sum();
        assert!((sum).abs() < 1e-5, "mean should be ~0 after layernorm");
    }
}
