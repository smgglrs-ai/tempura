use crate::Result;

pub trait RMSNorm: Send + Sync {
    /// Apply RMS normalization in-place.
    ///
    /// - `x`: [batch * seq_len, hidden_size] (flattened)
    /// - `weight`: [hidden_size] (gamma)
    /// - `eps`: small constant for numerical stability
    fn forward(&self, x: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) -> Result<()>;
}

#[cfg(feature = "cpu")]
pub struct CpuRMSNorm;

#[cfg(feature = "cpu")]
impl RMSNorm for CpuRMSNorm {
    fn forward(&self, x: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) -> Result<()> {
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

            let rms: f32 =
                (row.iter().map(|v| v * v).sum::<f32>() / hidden_size as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;

            for (i, val) in row.iter_mut().enumerate() {
                *val = *val * inv_rms * weight[i];
            }
        }

        Ok(())
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
}
