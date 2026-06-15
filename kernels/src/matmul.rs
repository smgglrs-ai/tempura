use crate::Result;

pub trait MatMul: Send + Sync {
    /// Matrix multiplication: output = a × b
    ///
    /// - `a`: [m, k] row-major
    /// - `b`: [k, n] row-major
    /// - `output`: [m, n] row-major
    fn forward(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;
}

#[cfg(feature = "cpu")]
pub struct CpuMatMul;

#[cfg(feature = "cpu")]
impl MatMul for CpuMatMul {
    fn forward(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k {
            return Err(crate::KernelError::DimensionMismatch {
                expected: m * k,
                got: a.len(),
            });
        }
        if b.len() != k * n {
            return Err(crate::KernelError::DimensionMismatch {
                expected: k * n,
                got: b.len(),
            });
        }
        if output.len() != m * n {
            return Err(crate::KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                output[i * n + j] = sum;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matrix() {
        let mm = CpuMatMul;
        // 2x2 identity × [1,2; 3,4] = [1,2; 3,4]
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        mm.forward(&identity, &b, &mut output, 2, 2, 2).unwrap();
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_known_2x2() {
        let mm = CpuMatMul;
        // [1,2; 3,4] × [5,6; 7,8] = [19,22; 43,50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 4];

        mm.forward(&a, &b, &mut output, 2, 2, 2).unwrap();
        assert_eq!(output, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_known_3x3() {
        let mm = CpuMatMul;
        // [1,0,0; 0,1,0; 0,0,1] × [1,2,3; 4,5,6; 7,8,9]
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut output = vec![0.0; 9];

        mm.forward(&a, &b, &mut output, 3, 3, 3).unwrap();
        assert_eq!(output, b);
    }

    #[test]
    fn test_non_square() {
        let mm = CpuMatMul;
        // [1,2,3] (1x3) × [4; 5; 6] (3x1) = [32] (1x1)
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut output = vec![0.0; 1];

        mm.forward(&a, &b, &mut output, 1, 1, 3).unwrap();
        assert_eq!(output, vec![32.0]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mm = CpuMatMul;
        let a = vec![1.0, 2.0]; // 1x2
        let b = vec![1.0, 2.0, 3.0]; // expects 2x? but len=3 doesn't match any valid k*n
        let mut output = vec![0.0; 2];

        let result = mm.forward(&a, &b, &mut output, 1, 2, 2);
        assert!(result.is_err());
    }
}
