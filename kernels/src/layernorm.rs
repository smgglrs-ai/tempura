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

#[cfg(feature = "cuda")]
pub struct CudaLayerNorm;

#[cfg(feature = "cuda")]
impl LayerNorm for CudaLayerNorm {
    fn forward(
        &self,
        _x: &mut [f32],
        _weight: &[f32],
        _bias: Option<&[f32]>,
        _hidden_size: usize,
        _eps: f32,
    ) -> Result<()> {
        todo!("CudaLayerNorm: requires cargo oxide build")
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn layernorm_indices_in_bounds() {
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
    fn layernorm_full_execution_no_panic() {
        let hidden_size: usize = kani::any();
        kani::assume(hidden_size >= 1 && hidden_size <= 4);

        let mut x = vec![1.0f32; hidden_size];
        let weight = vec![1.0f32; hidden_size];
        let bias = vec![0.0f32; hidden_size];
        let eps = 1e-6f32;

        let ln = super::CpuLayerNorm;
        let result = ln.forward(&mut x, &weight, Some(&bias), hidden_size, eps);
        assert!(result.is_ok());
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
