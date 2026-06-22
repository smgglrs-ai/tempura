use crate::Result;

pub trait FusedRMSNormResidual: Send + Sync {
    fn forward(
        &self,
        x: &mut [f32],
        residual: &[f32],
        weight: &[f32],
        hidden_size: usize,
        eps: f32,
    ) -> Result<()>;
}

pub trait FusedSiluGate: Send + Sync {
    fn forward(&self, gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<()>;
}

#[cfg(feature = "cpu")]
pub struct CpuFusedRMSNormResidual;

#[cfg(feature = "cpu")]
impl FusedRMSNormResidual for CpuFusedRMSNormResidual {
    fn forward(
        &self,
        x: &mut [f32],
        residual: &[f32],
        weight: &[f32],
        hidden_size: usize,
        eps: f32,
    ) -> Result<()> {
        if weight.len() != hidden_size {
            return Err(crate::KernelError::DimensionMismatch {
                expected: hidden_size,
                got: weight.len(),
            });
        }
        if x.len() != residual.len() {
            return Err(crate::KernelError::DimensionMismatch {
                expected: x.len(),
                got: residual.len(),
            });
        }

        let num_tokens = x.len() / hidden_size;
        for t in 0..num_tokens {
            let start = t * hidden_size;
            let end = start + hidden_size;

            for i in start..end {
                x[i] += residual[i];
            }

            let mut ss = 0.0f32;
            for i in start..end {
                ss += x[i] * x[i];
            }
            let rms = (ss / hidden_size as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;
            for i in 0..hidden_size {
                x[start + i] = x[start + i] * inv_rms * weight[i];
            }
        }
        Ok(())
    }
}

#[cfg(feature = "cpu")]
pub struct CpuFusedSiluGate;

#[cfg(feature = "cpu")]
impl FusedSiluGate for CpuFusedSiluGate {
    fn forward(&self, gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<()> {
        if gate.len() != up.len() || gate.len() != output.len() {
            return Err(crate::KernelError::DimensionMismatch {
                expected: gate.len(),
                got: up.len(),
            });
        }
        for i in 0..gate.len() {
            let silu = gate[i] / (1.0 + (-gate[i]).exp());
            output[i] = silu * up[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_rmsnorm_residual_matches_sequential() {
        let hidden = 8;
        let x_orig = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let residual = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let weight = vec![1.0f32; hidden];
        let eps = 1e-5;

        let mut x_seq = x_orig.clone();
        for i in 0..hidden {
            x_seq[i] += residual[i];
        }
        let mut ss = 0.0f32;
        for &v in &x_seq {
            ss += v * v;
        }
        let rms = (ss / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            x_seq[i] = x_seq[i] / rms * weight[i];
        }

        let mut x_fused = x_orig.clone();
        let fused = CpuFusedRMSNormResidual;
        fused
            .forward(&mut x_fused, &residual, &weight, hidden, eps)
            .unwrap();

        for i in 0..hidden {
            assert!(
                (x_seq[i] - x_fused[i]).abs() < 1e-6,
                "mismatch at {i}: seq={}, fused={}",
                x_seq[i],
                x_fused[i]
            );
        }
    }

    #[test]
    fn test_fused_silu_gate_matches_sequential() {
        let gate = vec![1.0f32, -1.0, 0.5, 2.0, -0.5, 3.0, 0.0, -2.0];
        let up = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut expected = vec![0.0f32; 8];
        for i in 0..8 {
            let silu = gate[i] / (1.0 + (-gate[i]).exp());
            expected[i] = silu * up[i];
        }

        let mut output = vec![0.0f32; 8];
        let fused = CpuFusedSiluGate;
        fused.forward(&gate, &up, &mut output).unwrap();

        for i in 0..8 {
            assert!(
                (expected[i] - output[i]).abs() < 1e-6,
                "mismatch at {i}: expected={}, got={}",
                expected[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_fused_silu_gate_zero_input() {
        let gate = vec![0.0f32; 4];
        let up = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        let fused = CpuFusedSiluGate;
        fused.forward(&gate, &up, &mut output).unwrap();
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_fused_rmsnorm_residual_dimension_check() {
        let fused = CpuFusedRMSNormResidual;
        let mut x = vec![1.0f32; 8];
        let residual = vec![1.0f32; 4];
        let weight = vec![1.0f32; 8];
        assert!(fused.forward(&mut x, &residual, &weight, 8, 1e-5).is_err());
    }

    #[test]
    fn test_fused_silu_gate_dimension_check() {
        let fused = CpuFusedSiluGate;
        let gate = vec![1.0f32; 8];
        let up = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 8];
        assert!(fused.forward(&gate, &up, &mut output).is_err());
    }
}
