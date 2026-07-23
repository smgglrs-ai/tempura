pub use crate::DeltaNet;
use crate::Result;

#[inline(always)]
fn softplus(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(feature = "cpu")]
pub struct CpuDeltaNet;

#[cfg(feature = "cpu")]
impl CpuDeltaNet {
    fn recurrent_step(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha: &[f32],
        beta: &[f32],
        state: &mut [f32],
        output: &mut [f32],
        num_heads: usize,
        head_dim: usize,
        ssm_a: f32,
    ) {
        let d2 = head_dim * head_dim;

        for h in 0..num_heads {
            let s_off = h * d2;
            let qkv_off = h * head_dim;

            // alpha and beta are PRE-COMPUTED:
            // alpha[h] = gate value (negative scalar, e.g. softplus(raw+dt_bias)*(-exp(A_log)))
            // beta[h] = sigmoid(beta_raw) in [0,1]
            // When called from old API: alpha = raw, ssm_a = scalar → gate = softplus(alpha)*ssm_a
            let gate = if ssm_a == 1.0 { alpha[h] } else { softplus(alpha[h]) * ssm_a };
            let decay = gate.exp();
            for i in 0..d2 {
                state[s_off + i] *= decay;
            }

            let beta_val = if ssm_a == 1.0 { beta[h] } else { sigmoid(beta[h]) };

            let mut retrieved = vec![0.0f32; head_dim];
            for row in 0..head_dim {
                for col in 0..head_dim {
                    retrieved[col] += state[s_off + row * head_dim + col] * k[qkv_off + row];
                }
            }

            let mut delta = vec![0.0f32; head_dim];
            for i in 0..head_dim {
                delta[i] = beta_val * (v[qkv_off + i] - retrieved[i]);
            }

            for row in 0..head_dim {
                for col in 0..head_dim {
                    state[s_off + row * head_dim + col] += k[qkv_off + row] * delta[col];
                }
            }

            for row in 0..head_dim {
                let mut sum = 0.0f32;
                for col in 0..head_dim {
                    sum += state[s_off + row * head_dim + col] * q[qkv_off + col];
                }
                output[qkv_off + row] = sum;
            }
        }
    }
}

#[cfg(feature = "cpu")]
impl DeltaNet for CpuDeltaNet {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Tested
    }

    fn forward_recurrent(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha: &[f32],
        beta: &[f32],
        state: &mut [f32],
        output: &mut [f32],
        num_heads: usize,
        head_dim: usize,
        ssm_a: f32,
    ) -> Result<()> {
        let vec_len = num_heads * head_dim;
        if q.len() != vec_len || k.len() != vec_len || v.len() != vec_len {
            return Err(crate::KernelError::DimensionMismatch {
                expected: vec_len,
                got: q.len(),
            });
        }
        if alpha.len() != num_heads || beta.len() != num_heads {
            return Err(crate::KernelError::DimensionMismatch {
                expected: num_heads,
                got: alpha.len(),
            });
        }
        let state_len = num_heads * head_dim * head_dim;
        if state.len() != state_len {
            return Err(crate::KernelError::DimensionMismatch {
                expected: state_len,
                got: state.len(),
            });
        }
        if output.len() != vec_len {
            return Err(crate::KernelError::DimensionMismatch {
                expected: vec_len,
                got: output.len(),
            });
        }

        Self::recurrent_step(q, k, v, alpha, beta, state, output, num_heads, head_dim, ssm_a);
        Ok(())
    }

    fn forward_chunk(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha: &[f32],
        beta: &[f32],
        state: &mut [f32],
        output: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        ssm_a: f32,
    ) -> Result<()> {
        let vec_len = num_heads * head_dim;
        let total_qkv = seq_len * vec_len;
        if q.len() != total_qkv || k.len() != total_qkv || v.len() != total_qkv {
            return Err(crate::KernelError::DimensionMismatch {
                expected: total_qkv,
                got: q.len(),
            });
        }
        let total_gates = seq_len * num_heads;
        if alpha.len() != total_gates || beta.len() != total_gates {
            return Err(crate::KernelError::DimensionMismatch {
                expected: total_gates,
                got: alpha.len(),
            });
        }
        let state_len = num_heads * head_dim * head_dim;
        if state.len() != state_len {
            return Err(crate::KernelError::DimensionMismatch {
                expected: state_len,
                got: state.len(),
            });
        }
        if output.len() != total_qkv {
            return Err(crate::KernelError::DimensionMismatch {
                expected: total_qkv,
                got: output.len(),
            });
        }

        for t in 0..seq_len {
            let qkv_start = t * vec_len;
            let gate_start = t * num_heads;
            Self::recurrent_step(
                &q[qkv_start..qkv_start + vec_len],
                &k[qkv_start..qkv_start + vec_len],
                &v[qkv_start..qkv_start + vec_len],
                &alpha[gate_start..gate_start + num_heads],
                &beta[gate_start..gate_start + num_heads],
                state,
                &mut output[qkv_start..qkv_start + vec_len],
                num_heads,
                head_dim,
                ssm_a,
            );
        }

        Ok(())
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn deltanet_state_indices_in_bounds() {
        let num_heads: usize = kani::any();
        let head_dim: usize = kani::any();

        kani::assume(num_heads >= 1 && num_heads <= 4);
        kani::assume(head_dim >= 1 && head_dim <= 8);

        let state_len = num_heads * head_dim * head_dim;
        let vec_len = num_heads * head_dim;

        let h: usize = kani::any();
        let row: usize = kani::any();
        let col: usize = kani::any();
        kani::assume(h < num_heads);
        kani::assume(row < head_dim);
        kani::assume(col < head_dim);

        let state_idx = h * head_dim * head_dim + row * head_dim + col;
        assert!(state_idx < state_len);

        let i: usize = kani::any();
        kani::assume(i < head_dim);
        let qkv_idx = h * head_dim + i;
        assert!(qkv_idx < vec_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deltanet_single_step() {
        let dn = CpuDeltaNet;
        let num_heads = 1;
        let head_dim = 2;

        let q = vec![1.0, 0.0];
        let k = vec![0.0, 1.0];
        let v = vec![3.0, 4.0];
        let alpha = vec![-100.0]; // softplus(-100)*ssm_a ~ 0, so exp(gate) ~ 1
        let beta = vec![0.0]; // sigmoid(0) = 0.5
        let ssm_a = -1.0;
        let mut state = vec![0.0; head_dim * head_dim]; // 2x2 = 4
        let mut output = vec![0.0; num_heads * head_dim];

        dn.forward_recurrent(
            &q, &k, &v, &alpha, &beta, &mut state, &mut output,
            num_heads, head_dim, ssm_a,
        )
        .unwrap();

        // With state=0:
        //   gate = softplus(-100) * (-1) ~ 0, decay = exp(0) = 1
        //   state stays 0 after decay
        //   retrieved = S^T @ k = [0,0]^T @ [0,1] = [0,0]
        //   delta = 0.5 * ([3,4] - [0,0]) = [1.5, 2.0]
        //   S += outer([0,1], [1.5,2.0]) = [[0,0],[1.5,2.0]]
        //   output = S @ q = [[0,0],[1.5,2.0]] @ [1,0] = [0, 1.5]
        assert!((output[0] - 0.0).abs() < 1e-4, "output[0]={}", output[0]);
        assert!((output[1] - 1.5).abs() < 1e-4, "output[1]={}", output[1]);

        // Verify state
        assert!((state[0] - 0.0).abs() < 1e-4); // [0][0]
        assert!((state[1] - 0.0).abs() < 1e-4); // [0][1]
        assert!((state[2] - 1.5).abs() < 1e-4); // [1][0]
        assert!((state[3] - 2.0).abs() < 1e-4); // [1][1]
    }

    #[test]
    fn test_deltanet_chunk_matches_sequential() {
        let dn = CpuDeltaNet;
        let num_heads = 2;
        let head_dim = 3;
        let seq_len = 3;
        let vec_len = num_heads * head_dim;
        let d2 = head_dim * head_dim;
        let ssm_a = -0.5;

        let q: Vec<f32> = (0..seq_len * vec_len)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * vec_len)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.2)
            .collect();
        let v: Vec<f32> = (0..seq_len * vec_len)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.15)
            .collect();
        let alpha: Vec<f32> = (0..seq_len * num_heads)
            .map(|i| ((i % 3) as f32 - 1.0) * 0.5)
            .collect();
        let beta: Vec<f32> = (0..seq_len * num_heads)
            .map(|i| ((i % 4) as f32 - 1.5) * 0.3)
            .collect();

        // Chunk path
        let mut state_chunk = vec![0.0f32; num_heads * d2];
        let mut output_chunk = vec![0.0f32; seq_len * vec_len];
        dn.forward_chunk(
            &q, &k, &v, &alpha, &beta, &mut state_chunk, &mut output_chunk,
            seq_len, num_heads, head_dim, ssm_a,
        )
        .unwrap();

        // Sequential path
        let mut state_seq = vec![0.0f32; num_heads * d2];
        let mut output_seq = vec![0.0f32; seq_len * vec_len];
        for t in 0..seq_len {
            let qkv_start = t * vec_len;
            let gate_start = t * num_heads;
            dn.forward_recurrent(
                &q[qkv_start..qkv_start + vec_len],
                &k[qkv_start..qkv_start + vec_len],
                &v[qkv_start..qkv_start + vec_len],
                &alpha[gate_start..gate_start + num_heads],
                &beta[gate_start..gate_start + num_heads],
                &mut state_seq,
                &mut output_seq[qkv_start..qkv_start + vec_len],
                num_heads,
                head_dim,
                ssm_a,
            )
            .unwrap();
        }

        for i in 0..state_chunk.len() {
            assert!(
                (state_chunk[i] - state_seq[i]).abs() < 1e-5,
                "state mismatch at {i}: chunk={} seq={}",
                state_chunk[i],
                state_seq[i],
            );
        }
        for i in 0..output_chunk.len() {
            assert!(
                (output_chunk[i] - output_seq[i]).abs() < 1e-5,
                "output mismatch at {i}: chunk={} seq={}",
                output_chunk[i],
                output_seq[i],
            );
        }
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_zero_state_identity(
                head_dim in 2usize..6usize,
                num_heads in 1usize..3usize,
            ) {
                let dn = CpuDeltaNet;
                let ssm_a = -1.0;

                let q: Vec<f32> = (0..num_heads * head_dim)
                    .map(|i| (i as f32 + 1.0) * 0.1)
                    .collect();
                let k: Vec<f32> = (0..num_heads * head_dim)
                    .map(|i| ((i % 3) as f32 + 1.0) * 0.2)
                    .collect();
                let v: Vec<f32> = (0..num_heads * head_dim)
                    .map(|i| (i as f32 + 0.5) * 0.3)
                    .collect();
                // Very negative alpha => softplus(alpha) ~ 0 => gate ~ 0 => decay = exp(0) = 1
                let alpha = vec![-100.0f32; num_heads];
                let beta = vec![0.0f32; num_heads]; // sigmoid(0) = 0.5

                let mut state = vec![0.0f32; num_heads * head_dim * head_dim];
                let mut output = vec![0.0f32; num_heads * head_dim];

                dn.forward_recurrent(
                    &q, &k, &v, &alpha, &beta, &mut state, &mut output,
                    num_heads, head_dim, ssm_a,
                ).unwrap();

                // With S=0 initially, decay=1:
                //   retrieved = 0^T @ k = 0
                //   delta = 0.5 * (v - 0) = 0.5 * v
                //   S = outer(k, 0.5*v)
                //   output = S @ q = k * (0.5*v . q)   where . is dot product per head
                for h in 0..num_heads {
                    let off = h * head_dim;
                    let k_h = &k[off..off + head_dim];
                    let v_h = &v[off..off + head_dim];
                    let q_h = &q[off..off + head_dim];

                    let half_v_dot_q: f32 = v_h.iter().zip(q_h)
                        .map(|(vi, qi)| 0.5 * vi * qi)
                        .sum();

                    for d in 0..head_dim {
                        let expected = k_h[d] * half_v_dot_q;
                        prop_assert!(
                            (output[off + d] - expected).abs() < 1e-4,
                            "head {h} dim {d}: output={} expected={}",
                            output[off + d], expected,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(kani)]
mod kani_conv1d {
    #[kani::proof]
    fn conv1d_index_bounds() {
        let seq_len: usize = kani::any();
        let conv_dim: usize = kani::any();
        let kernel_size: usize = kani::any();
        kani::assume(seq_len >= 1 && seq_len <= 4);
        kani::assume(conv_dim >= 1 && conv_dim <= 8);
        kani::assume(kernel_size >= 1 && kernel_size <= 4);

        let conv_input_len = seq_len * conv_dim;
        let conv_w_len = conv_dim * kernel_size;
        let conv_out_len = seq_len * conv_dim;

        let ch: usize = kani::any();
        let t: usize = kani::any();
        let ki: usize = kani::any();
        kani::assume(ch < conv_dim);
        kani::assume(t < seq_len);
        kani::assume(ki < kernel_size);

        let src_t = t as isize - (kernel_size as isize - 1 - ki as isize);
        if src_t >= 0 && (src_t as usize) < seq_len {
            let input_idx = src_t as usize * conv_dim + ch;
            assert!(input_idx < conv_input_len);
        }
        let w_idx = ch * kernel_size + ki;
        assert!(w_idx < conv_w_len);
        let out_idx = t * conv_dim + ch;
        assert!(out_idx < conv_out_len);
    }
}
