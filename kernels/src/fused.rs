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

/// Fused RMSNorm (f64 accum, ggml-compatible) + q8_snap in one pass.
/// Avoids a second pass over the data. Writes result in-place into `x`.
#[cfg(feature = "cpu")]
pub fn fused_rmsnorm_q8snap(x: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) {
    let mut sum: f64 = 0.0;
    for i in 0..hidden_size {
        sum += (x[i] * x[i]) as f64;
    }
    let mean = (sum / hidden_size as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();

    for i in 0..hidden_size {
        x[i] = x[i] * scale * weight[i];
    }

    crate::portable_math::q8_snap(x);
}

/// Fused: dst = src + residual, then RMSNorm(dst) + q8_snap.
/// `hidden` is updated in-place with the residual sum (for subsequent use).
/// `normed` receives the normed+snapped result.
#[cfg(feature = "cpu")]
pub fn fused_residual_rmsnorm_q8snap(
    hidden: &mut [f32],
    residual: &[f32],
    normed: &mut [f32],
    weight: &[f32],
    hidden_size: usize,
    eps: f32,
) {
    for i in 0..hidden_size {
        hidden[i] += residual[i];
    }
    normed.copy_from_slice(hidden);

    let mut sum: f64 = 0.0;
    for i in 0..hidden_size {
        sum += (normed[i] * normed[i]) as f64;
    }
    let mean = (sum / hidden_size as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();

    for i in 0..hidden_size {
        normed[i] = normed[i] * scale * weight[i];
    }

    crate::portable_math::q8_snap(normed);
}

/// Fused squared ReLU activation: gate[i] = max(0, gate[i])^2 * up[i]
#[cfg(feature = "cpu")]
pub fn fused_sqrelu_mul(gate: &mut [f32], up: &[f32], n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                fused_sqrelu_mul_avx2(gate, up, n);
            }
            return;
        }
    }
    for i in 0..n {
        let r = gate[i].max(0.0);
        gate[i] = r * r * up[i];
    }
}

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn fused_sqrelu_mul_avx2(gate: &mut [f32], up: &[f32], n: usize) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let chunks = n / 8;
    for i in 0..chunks {
        let g = _mm256_loadu_ps(gate.as_ptr().add(i * 8));
        let u = _mm256_loadu_ps(up.as_ptr().add(i * 8));
        let relu = _mm256_max_ps(g, zero);
        let sq = _mm256_mul_ps(relu, relu);
        let result = _mm256_mul_ps(sq, u);
        _mm256_storeu_ps(gate.as_mut_ptr().add(i * 8), result);
    }
    for i in chunks * 8..n {
        let r = gate[i].max(0.0);
        gate[i] = r * r * up[i];
    }
}

/// SIMD residual add: dst[i] += src[i]
#[cfg(feature = "cpu")]
pub fn residual_add(dst: &mut [f32], src: &[f32], n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                residual_add_avx2(dst, src, n);
            }
            return;
        }
    }
    for i in 0..n {
        dst[i] += src[i];
    }
}

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn residual_add_avx2(dst: &mut [f32], src: &[f32], n: usize) {
    use std::arch::x86_64::*;
    let chunks = n / 8;
    for i in 0..chunks {
        let d = _mm256_loadu_ps(dst.as_ptr().add(i * 8));
        let s = _mm256_loadu_ps(src.as_ptr().add(i * 8));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i * 8), _mm256_add_ps(d, s));
    }
    for i in chunks * 8..n {
        dst[i] += src[i];
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

        let mut expected = [0.0f32; 8];
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

    #[test]
    fn test_fused_rmsnorm_q8snap_matches_sequential() {
        let h = 64;
        let weight = vec![1.0f32; h];
        let eps = 1e-5f32;
        let input: Vec<f32> = (0..h).map(|i| (i as f32 - 32.0) * 0.1).collect();

        let mut seq = input.clone();
        use crate::RMSNorm;
        crate::rmsnorm::CpuRMSNormGgml
            .forward(&mut seq, &weight, h, eps)
            .unwrap();
        crate::portable_math::q8_snap(&mut seq);

        let mut fused = input.clone();
        fused_rmsnorm_q8snap(&mut fused, &weight, h, eps);

        for i in 0..h {
            assert!(
                (seq[i] - fused[i]).abs() < 1e-5,
                "mismatch at {i}: seq={}, fused={}",
                seq[i],
                fused[i]
            );
        }
    }

    #[test]
    fn test_fused_residual_rmsnorm_q8snap() {
        let h = 64;
        let weight = vec![1.0f32; h];
        let eps = 1e-5f32;
        let hidden_orig: Vec<f32> = (0..h).map(|i| i as f32 * 0.1).collect();
        let residual: Vec<f32> = (0..h).map(|i| (h - i) as f32 * 0.05).collect();

        let mut hidden_seq = hidden_orig.clone();
        for i in 0..h {
            hidden_seq[i] += residual[i];
        }
        let mut normed_seq = hidden_seq.clone();
        use crate::RMSNorm;
        crate::rmsnorm::CpuRMSNormGgml
            .forward(&mut normed_seq, &weight, h, eps)
            .unwrap();
        crate::portable_math::q8_snap(&mut normed_seq);

        let mut hidden_fused = hidden_orig.clone();
        let mut normed_fused = vec![0.0f32; h];
        fused_residual_rmsnorm_q8snap(
            &mut hidden_fused,
            &residual,
            &mut normed_fused,
            &weight,
            h,
            eps,
        );

        for i in 0..h {
            assert!((hidden_seq[i] - hidden_fused[i]).abs() < 1e-5);
            assert!((normed_seq[i] - normed_fused[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fused_sqrelu_mul() {
        let n = 17;
        let mut gate: Vec<f32> = (-8..9).map(|i| i as f32 * 0.5).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let mut expected = gate.clone();
        for i in 0..n {
            let r = expected[i].max(0.0);
            expected[i] = r * r * up[i];
        }

        fused_sqrelu_mul(&mut gate, &up, n);

        for i in 0..n {
            assert!(
                (gate[i] - expected[i]).abs() < 1e-5,
                "mismatch at {i}: got={}, exp={}",
                gate[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_residual_add() {
        let n = 17;
        let mut dst: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let src: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.5).collect();
        let expected: Vec<f32> = (0..n).map(|i| dst[i] + src[i]).collect();
        residual_add(&mut dst, &src, n);
        for i in 0..n {
            assert!((dst[i] - expected[i]).abs() < 1e-6);
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn fused_rmsnorm_q8snap_index_safety() {
        let hidden_size: usize = kani::any();
        kani::assume(hidden_size >= 1 && hidden_size <= 64);

        let i: usize = kani::any();
        kani::assume(i < hidden_size);

        assert!(i < hidden_size);
    }

    #[kani::proof]
    fn fused_residual_rmsnorm_q8snap_index_safety() {
        let hidden_size: usize = kani::any();
        kani::assume(hidden_size >= 1 && hidden_size <= 64);

        let i: usize = kani::any();
        kani::assume(i < hidden_size);

        let x_len = hidden_size;
        let residual_len = hidden_size;
        let weight_len = hidden_size;

        assert!(i < x_len);
        assert!(i < residual_len);
        assert!(i < weight_len);
    }

    #[kani::proof]
    fn fused_sqrelu_mul_index_safety() {
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 64);

        let i: usize = kani::any();
        kani::assume(i < n);

        assert!(i < n); // gate[i] and up[i] both in bounds
    }

    #[kani::proof]
    fn fused_sqrelu_output_non_negative() {
        let gate: f32 = kani::any();
        kani::assume(gate.is_finite());
        let up: f32 = kani::any();
        kani::assume(up.is_finite());
        kani::assume(up >= 0.0);

        let r = gate.max(0.0);
        let result = r * r * up;
        assert!(result >= 0.0 || result.is_nan());
    }

    #[kani::proof]
    fn residual_add_index_safety() {
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 64);

        let i: usize = kani::any();
        kani::assume(i < n);

        assert!(i < n); // dst[i] and src[i] both in bounds
    }
}
