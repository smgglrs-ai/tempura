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

    /// Matrix multiplication with transposed B: output = a × bᵀ
    ///
    /// - `a`: [m, k] row-major
    /// - `b`: [n, k] row-major (stored transposed, as in PyTorch Linear weights)
    /// - `output`: [m, n] row-major
    fn forward_b_transposed(
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

        // Safety: dimension checks above guarantee all pointer arithmetic
        // stays within allocated slices. Kani proof `matmul_indices_in_bounds`
        // verifies the indexing arithmetic.
        // Safety: dimension checks above guarantee all pointer arithmetic
        // stays within allocated slices.
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                output.as_mut_ptr(),
                1,          // dst_cs (column stride = 1 for row-major)
                n as isize, // dst_rs (row stride = n for row-major)
                false,      // read_dst: overwrite, don't accumulate
                a.as_ptr(),
                1,          // lhs_cs
                k as isize, // lhs_rs
                b.as_ptr(),
                1,          // rhs_cs
                n as isize, // rhs_rs
                0.0f32,     // alpha: not reading dst
                1.0f32,     // beta: dst = 1.0 * lhs × rhs
                false,      // conj_dst
                false,      // conj_lhs
                false,      // conj_rhs
                gemm::Parallelism::Rayon(0),
            );
        }

        Ok(())
    }

    fn forward_b_transposed(
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
        if b.len() != n * k {
            return Err(crate::KernelError::DimensionMismatch {
                expected: n * k,
                got: b.len(),
            });
        }
        if output.len() != m * n {
            return Err(crate::KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }

        // b is [n, k] row-major — swap rhs strides to read it as [k, n]
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                output.as_mut_ptr(),
                1,          // dst_cs
                n as isize, // dst_rs
                false,
                a.as_ptr(),
                1,          // lhs_cs
                k as isize, // lhs_rs
                b.as_ptr(),
                k as isize, // rhs_cs: stride between columns = k (transposed)
                1,          // rhs_rs: stride between rows = 1 (transposed)
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub struct CudaMatMul;

#[cfg(feature = "cuda")]
impl MatMul for CudaMatMul {
    fn forward(
        &self,
        _a: &[f32],
        _b: &[f32],
        _output: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        todo!("CudaMatMul: requires cargo oxide build")
    }

    fn forward_b_transposed(
        &self,
        _a: &[f32],
        _b: &[f32],
        _output: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        todo!("CudaMatMul: requires cargo oxide build")
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn matmul_indices_in_bounds() {
        let m: usize = kani::any();
        let n: usize = kani::any();
        let k: usize = kani::any();

        kani::assume(m >= 1 && m <= 4);
        kani::assume(n >= 1 && n <= 4);
        kani::assume(k >= 1 && k <= 4);

        let a_len = m * k;
        let b_len = k * n;
        let o_len = m * n;

        let i: usize = kani::any();
        let j: usize = kani::any();
        let p: usize = kani::any();
        kani::assume(i < m);
        kani::assume(j < n);
        kani::assume(p < k);

        assert!(i * k + p < a_len);
        assert!(p * n + j < b_len);
        assert!(i * n + j < o_len);
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
