use super::quantize::{quantize_activations, quantized_row_size};
use super::GgmlType;

fn vec_dot_best(n: usize, result: &mut f32, vx: &[u8], vy: &[u8], weight_type: GgmlType) {
    use super::blocks::*;

    #[cfg(target_arch = "x86_64")]
    {
        // AVX2 + FMA (benchmarked faster than VNNI for Q4K due to dequant amortization)
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            match weight_type {
                GgmlType::Q4_0 => {
                    let x = unsafe {
                        std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ4_0, n / 32)
                    };
                    let y = unsafe {
                        std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8_0, n / 32)
                    };
                    *result = unsafe { super::vec_dot_avx2::vec_dot_q4_0_q8_0_avx2(n, x, y) };
                    return;
                }
                GgmlType::Q8_0 => {
                    let x = unsafe {
                        std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ8_0, n / 32)
                    };
                    let y = unsafe {
                        std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8_0, n / 32)
                    };
                    *result = unsafe { super::vec_dot_avx2::vec_dot_q8_0_q8_0_avx2(n, x, y) };
                    return;
                }
                GgmlType::Q4K => {
                    let x = unsafe {
                        std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ4K, n / 256)
                    };
                    let y = unsafe {
                        std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256)
                    };
                    *result = unsafe { super::vec_dot_avx2::vec_dot_q4_k_q8_k_avx2(n, x, y) };
                    return;
                }
                GgmlType::Q6K => {
                    let x = unsafe {
                        std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ6K, n / 256)
                    };
                    let y = unsafe {
                        std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256)
                    };
                    *result = unsafe { super::vec_dot_avx2::vec_dot_q6_k_q8_k_avx2(n, x, y) };
                    return;
                }
                _ => {}
            }
        }
    }

    super::vec_dot::vec_dot(n, result, vx, vy, weight_type);
}

/// Registry-driven quantized mat-mul, matching ggml's `ggml_compute_forward_mul_mat` pattern.
///
/// src0 (weights) is quantized with type `src0_type`, stored row-major.
/// src1 (activations) is f32, stored row-major as m rows × k cols.
/// dst is f32, stored row-major as m rows × n cols.
///
/// Each weight row is `k / block_size * type_size` bytes.
/// n = number of weight rows = number of output columns.
///
/// Algorithm:
/// 1. Quantize each src1 row to vec_dot_type (Q8_0 or Q8_K)
/// 2. For each (src1_row, weight_row), call vec_dot → one output element
pub fn mul_mat_q(
    src0_type: GgmlType,
    src0: &[u8],
    src1: &[f32],
    dst: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let vdt = src0_type.vec_dot_type();
    let weight_row_bytes = quantized_row_size(k, src0_type);
    let act_row_bytes = quantized_row_size(k, vdt);

    assert_eq!(src0.len(), n * weight_row_bytes, "src0 size mismatch");
    assert_eq!(src1.len(), m * k, "src1 size mismatch");
    assert_eq!(dst.len(), m * n, "dst size mismatch");

    // quantize all activation rows
    let mut act_quantized = vec![0u8; m * act_row_bytes];
    for row in 0..m {
        let src_row = &src1[row * k..(row + 1) * k];
        let dst_row = &mut act_quantized[row * act_row_bytes..(row + 1) * act_row_bytes];
        quantize_activations(src_row, dst_row, vdt);
    }

    // For Q4K with AVX2, call the optimized path directly to avoid per-call dispatch
    use rayon::prelude::*;

    if m == 1 {
        // Decode path: parallelize over output columns (not rows — there's only 1)
        let act_row = &act_quantized[..act_row_bytes];
        dst.par_iter_mut().enumerate().for_each(|(col, out)| {
            let weight_row = &src0[col * weight_row_bytes..(col + 1) * weight_row_bytes];
            vec_dot_best(k, out, weight_row, act_row, src0_type);
        });
    } else {
        // Batch path: parallelize over rows
        dst.par_chunks_mut(n)
            .enumerate()
            .for_each(|(row, dst_row)| {
                let act_row = &act_quantized[row * act_row_bytes..(row + 1) * act_row_bytes];
                for col in 0..n {
                    let weight_row = &src0[col * weight_row_bytes..(col + 1) * weight_row_bytes];
                    vec_dot_best(k, &mut dst_row[col], weight_row, act_row, src0_type);
                }
            });
    }
}

/// Pre-quantized activations for reuse across multiple mul_mat_q calls.
pub struct PreQuantizedQ8 {
    pub data: Vec<u8>,
    pub vec_dot_type: GgmlType,
    pub k: usize,
    pub m: usize,
}

impl PreQuantizedQ8 {
    pub fn from_f32(src: &[f32], k: usize, weight_type: GgmlType) -> Self {
        let vdt = weight_type.vec_dot_type();
        let m = src.len() / k;
        let act_row_bytes = quantized_row_size(k, vdt);
        let mut data = vec![0u8; m * act_row_bytes];
        for row in 0..m {
            let src_row = &src[row * k..(row + 1) * k];
            let dst_row = &mut data[row * act_row_bytes..(row + 1) * act_row_bytes];
            quantize_activations(src_row, dst_row, vdt);
        }
        Self {
            data,
            vec_dot_type: vdt,
            k,
            m,
        }
    }

    pub fn row_bytes(&self) -> usize {
        quantized_row_size(self.k, self.vec_dot_type)
    }
}

/// mul_mat with pre-quantized activations — avoids re-quantizing when
/// the same activations are used across multiple weight tensors.
pub fn mul_mat_q_preq(
    src0_type: GgmlType,
    src0: &[u8],
    preq: &PreQuantizedQ8,
    dst: &mut [f32],
    n: usize,
) {
    let k = preq.k;
    let m = preq.m;
    let weight_row_bytes = quantized_row_size(k, src0_type);
    let act_row_bytes = preq.row_bytes();

    assert_eq!(src0.len(), n * weight_row_bytes, "src0 size mismatch");
    assert_eq!(dst.len(), m * n, "dst size mismatch");

    use rayon::prelude::*;
    dst.par_chunks_mut(n)
        .enumerate()
        .for_each(|(row, dst_row)| {
            let act_row = &preq.data[row * act_row_bytes..(row + 1) * act_row_bytes];
            for col in 0..n {
                let weight_row = &src0[col * weight_row_bytes..(col + 1) * weight_row_bytes];
                vec_dot_best(k, &mut dst_row[col], weight_row, act_row, src0_type);
            }
        });
}

/// Scalar-only mul_mat (no AVX2). Used for DualPipeline validation.
pub fn mul_mat_q_scalar(
    src0_type: GgmlType,
    src0: &[u8],
    src1: &[f32],
    dst: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let vdt = src0_type.vec_dot_type();
    let weight_row_bytes = quantized_row_size(k, src0_type);
    let act_row_bytes = quantized_row_size(k, vdt);

    assert_eq!(src0.len(), n * weight_row_bytes, "src0 size mismatch");
    assert_eq!(src1.len(), m * k, "src1 size mismatch");
    assert_eq!(dst.len(), m * n, "dst size mismatch");

    let mut act_quantized = vec![0u8; m * act_row_bytes];
    for row in 0..m {
        let src_row = &src1[row * k..(row + 1) * k];
        let dst_row = &mut act_quantized[row * act_row_bytes..(row + 1) * act_row_bytes];
        quantize_activations(src_row, dst_row, vdt);
    }

    for row in 0..m {
        let act_row = &act_quantized[row * act_row_bytes..(row + 1) * act_row_bytes];
        for col in 0..n {
            let weight_row = &src0[col * weight_row_bytes..(col + 1) * weight_row_bytes];
            super::vec_dot::vec_dot(k, &mut dst[row * n + col], weight_row, act_row, src0_type);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::blocks::*;
    use super::super::dequantize::dequantize_row_q4_k;
    use super::super::helpers::f32_to_f16;
    use super::*;

    fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[j * k + p]; // b is transposed (row-major weight rows)
                }
                c[i * n + j] = sum;
            }
        }
    }

    #[test]
    fn mul_mat_q4k_matches_dequant_matmul() {
        let n = 2; // 2 weight rows (output cols)
        let k = 256; // must be multiple of 256 for Q4K

        // create Q4K weight blocks (2 rows × 1 block each)
        let mut weights = Vec::new();
        for row_idx in 0..n {
            let blk = BlockQ4K {
                d: f32_to_f16(1.0),
                dmin: f32_to_f16(0.0),
                scales: [((row_idx + 1) & 63) as u8; 12],
                qs: [(row_idx as u8 + 1) | ((row_idx as u8 + 2) << 4); 128],
            };
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &blk as *const BlockQ4K as *const u8,
                    std::mem::size_of::<BlockQ4K>(),
                )
            };
            weights.extend_from_slice(bytes);
        }

        // dequantize for reference
        let mut w_f32 = vec![0.0f32; n * k];
        for row_idx in 0..n {
            let blk_bytes = &weights[row_idx * BLOCK_Q4K_SIZE..(row_idx + 1) * BLOCK_Q4K_SIZE];
            let blk = unsafe { &*(blk_bytes.as_ptr() as *const BlockQ4K) };
            dequantize_row_q4_k(
                std::slice::from_ref(blk),
                &mut w_f32[row_idx * k..(row_idx + 1) * k],
            );
        }

        let m = 1;
        let activations: Vec<f32> = (0..k).map(|i| (i as f32 - 128.0) / 128.0).collect();

        // reference: f32 matmul
        let mut ref_out = vec![0.0f32; m * n];
        naive_matmul(&activations, &w_f32, &mut ref_out, m, n, k);

        // quantized: mul_mat_q
        let mut q_out = vec![0.0f32; m * n];
        mul_mat_q(GgmlType::Q4K, &weights, &activations, &mut q_out, m, n, k);

        for j in 0..n {
            let rel_err = if ref_out[j].abs() > 0.01 {
                ((q_out[j] - ref_out[j]) / ref_out[j]).abs()
            } else {
                (q_out[j] - ref_out[j]).abs()
            };
            assert!(
                rel_err < 0.15,
                "col {j}: q_out={}, ref_out={}, rel_err={rel_err}",
                q_out[j],
                ref_out[j]
            );
        }
    }

    #[test]
    fn mul_mat_q4_0_small() {
        let k = 32;
        let n = 1;

        // create a Q4_0 block: all nibbles = 8 (value = 0 after -8 offset)
        let blk = BlockQ4_0 {
            d: f32_to_f16(1.0),
            qs: [0x88; 16],
        };
        let weights = unsafe {
            std::slice::from_raw_parts(
                &blk as *const BlockQ4_0 as *const u8,
                std::mem::size_of::<BlockQ4_0>(),
            )
        };

        let activations = vec![1.0f32; k];
        let mut out = vec![99.0f32; 1];
        mul_mat_q(GgmlType::Q4_0, weights, &activations, &mut out, 1, n, k);
        // all weight values are 0 (nibble 8 - 8 = 0), so dot should be 0
        assert!(out[0].abs() < 0.1, "expected ~0, got {}", out[0]);
    }

    // ── Determinism: same input → bit-identical output ──

    #[test]
    fn determinism_q4k_avx2() {
        let n = 16;
        let k = 1024;
        let m = 1;
        let weights = make_q4k_weights(n, k, 42);
        let activations = make_activations(m, k);

        let mut out1 = vec![0.0f32; m * n];
        let mut out2 = vec![0.0f32; m * n];

        mul_mat_q(GgmlType::Q4K, &weights, &activations, &mut out1, m, n, k);
        mul_mat_q(GgmlType::Q4K, &weights, &activations, &mut out2, m, n, k);

        for j in 0..m * n {
            assert_eq!(
                out1[j].to_bits(),
                out2[j].to_bits(),
                "Q4K determinism failure at [{j}]: run1={} run2={}",
                out1[j],
                out2[j]
            );
        }
    }

    #[test]
    fn determinism_q6k_avx2() {
        let n = 16;
        let k = 1024;
        let m = 1;
        let weights = make_q6k_weights(n, k, 17);
        let activations = make_activations(m, k);

        let mut out1 = vec![0.0f32; m * n];
        let mut out2 = vec![0.0f32; m * n];

        mul_mat_q(GgmlType::Q6K, &weights, &activations, &mut out1, m, n, k);
        mul_mat_q(GgmlType::Q6K, &weights, &activations, &mut out2, m, n, k);

        for j in 0..m * n {
            assert_eq!(
                out1[j].to_bits(),
                out2[j].to_bits(),
                "Q6K determinism failure at [{j}]: run1={} run2={}",
                out1[j],
                out2[j]
            );
        }
    }

    // ── Scalar vs AVX2 equivalence (not bit-identical due to f32 associativity) ──
    // madd_epi16(scale, maddubs(q4, q8)) computes scale*(a0*b0+a1*b1)
    // scalar computes scale*a0*b0 + scale*a1*b1 — different rounding.
    // Tolerance: < 0.01 relative error (acceptable for quantized inference).

    fn make_q4k_weights(n: usize, k: usize, seed: u8) -> Vec<u8> {
        let blocks_per_row = k / 256;
        let mut weights = Vec::with_capacity(n * blocks_per_row * BLOCK_Q4K_SIZE);
        for row in 0..n {
            for blk_idx in 0..blocks_per_row {
                let s = seed.wrapping_add(row as u8).wrapping_add(blk_idx as u8);
                let blk = BlockQ4K {
                    d: f32_to_f16(0.1 + 0.01 * s as f32),
                    dmin: f32_to_f16(0.005 * s as f32),
                    scales: [
                        (s.wrapping_mul(3)) & 63,
                        (s.wrapping_mul(7)) & 63,
                        (s.wrapping_mul(11)) & 63,
                        (s.wrapping_mul(13)) & 63,
                        (s.wrapping_mul(17)) & 63,
                        (s.wrapping_mul(19)) & 63,
                        (s.wrapping_mul(23)) & 63,
                        (s.wrapping_mul(29)) & 63,
                        s & 0x0F,
                        s.wrapping_add(1) & 0x0F,
                        s.wrapping_add(2) & 0x0F,
                        s.wrapping_add(3) & 0x0F,
                    ],
                    qs: std::array::from_fn(|i| {
                        let lo = (s.wrapping_add(i as u8)) & 0x0F;
                        let hi = (s.wrapping_add(i as u8).wrapping_mul(3)) & 0x0F;
                        lo | (hi << 4)
                    }),
                };
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &blk as *const BlockQ4K as *const u8,
                        std::mem::size_of::<BlockQ4K>(),
                    )
                };
                weights.extend_from_slice(bytes);
            }
        }
        weights
    }

    fn make_q6k_weights(n: usize, k: usize, seed: u8) -> Vec<u8> {
        let blocks_per_row = k / 256;
        let mut weights = Vec::with_capacity(n * blocks_per_row * BLOCK_Q6K_SIZE);
        for row in 0..n {
            for blk_idx in 0..blocks_per_row {
                let s = seed.wrapping_add(row as u8).wrapping_add(blk_idx as u8);
                let blk = BlockQ6K {
                    ql: std::array::from_fn(|i| s.wrapping_add(i as u8).wrapping_mul(7)),
                    qh: std::array::from_fn(|i| s.wrapping_add(i as u8).wrapping_mul(3) & 0x3F),
                    scales: std::array::from_fn(|i| (s.wrapping_add(i as u8) % 16) as i8),
                    d: f32_to_f16(0.05 + 0.01 * s as f32),
                };
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &blk as *const BlockQ6K as *const u8,
                        std::mem::size_of::<BlockQ6K>(),
                    )
                };
                weights.extend_from_slice(bytes);
            }
        }
        weights
    }

    fn make_q4_0_weights(n: usize, k: usize, seed: u8) -> Vec<u8> {
        let blocks_per_row = k / 32;
        let mut weights = Vec::with_capacity(n * blocks_per_row * BLOCK_Q4_0_SIZE);
        for row in 0..n {
            for blk_idx in 0..blocks_per_row {
                let s = seed.wrapping_add(row as u8).wrapping_add(blk_idx as u8);
                let blk = BlockQ4_0 {
                    d: f32_to_f16(0.1 + 0.02 * s as f32),
                    qs: std::array::from_fn(|i| {
                        let lo = (s.wrapping_add(i as u8)) & 0x0F;
                        let hi = (s.wrapping_add(i as u8).wrapping_mul(5)) & 0x0F;
                        lo | (hi << 4)
                    }),
                };
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &blk as *const BlockQ4_0 as *const u8,
                        std::mem::size_of::<BlockQ4_0>(),
                    )
                };
                weights.extend_from_slice(bytes);
            }
        }
        weights
    }

    fn make_activations(m: usize, k: usize) -> Vec<f32> {
        (0..m * k)
            .map(|i| {
                let phase = i as f32 * 0.1;
                (phase.sin() * 0.5 + 0.1) / (1.0 + (i % k) as f32 * 0.001)
            })
            .collect()
    }

    #[test]
    fn dual_pipeline_q4k_scalar_vs_avx2() {
        let n = 16;
        let k = 1024; // 4 Q4K blocks per row
        let m = 1;
        let weights = make_q4k_weights(n, k, 42);
        let activations = make_activations(m, k);

        let mut out_avx2 = vec![0.0f32; m * n];
        let mut out_scalar = vec![0.0f32; m * n];

        mul_mat_q(
            GgmlType::Q4K,
            &weights,
            &activations,
            &mut out_avx2,
            m,
            n,
            k,
        );
        mul_mat_q_scalar(
            GgmlType::Q4K,
            &weights,
            &activations,
            &mut out_scalar,
            m,
            n,
            k,
        );

        let mut max_diff = 0.0f32;
        for j in 0..m * n {
            let diff = (out_avx2[j] - out_scalar[j]).abs();
            max_diff = max_diff.max(diff);
            let rel = if out_scalar[j].abs() > 0.01 {
                diff / out_scalar[j].abs()
            } else {
                diff
            };
            assert!(
                rel < 0.001,
                "Q4K dual_pipeline mismatch at [{j}]: scalar={}, avx2={}, rel_err={rel}",
                out_scalar[j],
                out_avx2[j]
            );
        }
        eprintln!("Q4K dual_pipeline: max_abs_diff={max_diff:.6e} across {n} outputs (k={k})");
    }

    #[test]
    fn dual_pipeline_q6k_scalar_vs_avx2() {
        let n = 16;
        let k = 1024;
        let m = 1;
        let weights = make_q6k_weights(n, k, 17);
        let activations = make_activations(m, k);

        let mut out_avx2 = vec![0.0f32; m * n];
        let mut out_scalar = vec![0.0f32; m * n];

        mul_mat_q(
            GgmlType::Q6K,
            &weights,
            &activations,
            &mut out_avx2,
            m,
            n,
            k,
        );
        mul_mat_q_scalar(
            GgmlType::Q6K,
            &weights,
            &activations,
            &mut out_scalar,
            m,
            n,
            k,
        );

        let mut max_diff = 0.0f32;
        for j in 0..m * n {
            let diff = (out_avx2[j] - out_scalar[j]).abs();
            max_diff = max_diff.max(diff);
            let rel = if out_scalar[j].abs() > 0.01 {
                diff / out_scalar[j].abs()
            } else {
                diff
            };
            assert!(
                rel < 0.001,
                "Q6K dual_pipeline mismatch at [{j}]: scalar={}, avx2={}, rel_err={rel}",
                out_scalar[j],
                out_avx2[j]
            );
        }
        eprintln!("Q6K dual_pipeline: max_abs_diff={max_diff:.6e} across {n} outputs (k={k})");
    }

    #[test]
    fn dual_pipeline_q4_0_scalar_vs_avx2() {
        let n = 32;
        let k = 512; // 16 Q4_0 blocks per row
        let m = 2;
        let weights = make_q4_0_weights(n, k, 7);
        let activations = make_activations(m, k);

        let mut out_avx2 = vec![0.0f32; m * n];
        let mut out_scalar = vec![0.0f32; m * n];

        mul_mat_q(
            GgmlType::Q4_0,
            &weights,
            &activations,
            &mut out_avx2,
            m,
            n,
            k,
        );
        mul_mat_q_scalar(
            GgmlType::Q4_0,
            &weights,
            &activations,
            &mut out_scalar,
            m,
            n,
            k,
        );

        let mut max_diff = 0.0f32;
        for j in 0..m * n {
            let diff = (out_avx2[j] - out_scalar[j]).abs();
            max_diff = max_diff.max(diff);
            let rel = if out_scalar[j].abs() > 0.01 {
                diff / out_scalar[j].abs()
            } else {
                diff
            };
            assert!(
                rel < 0.001,
                "Q4_0 dual_pipeline mismatch at [{j}]: scalar={}, avx2={}, rel_err={rel}",
                out_scalar[j],
                out_avx2[j]
            );
        }
        eprintln!(
            "Q4_0 dual_pipeline: max_abs_diff={max_diff:.6e} across {} outputs (k={k})",
            m * n
        );
    }

    #[test]
    fn dual_pipeline_q4k_batch_scalar_vs_avx2() {
        let n = 8;
        let k = 512;
        let m = 4; // batch of 4 — exercises batched activation quantization
        let weights = make_q4k_weights(n, k, 99);
        let activations = make_activations(m, k);

        let mut out_avx2 = vec![0.0f32; m * n];
        let mut out_scalar = vec![0.0f32; m * n];

        mul_mat_q(
            GgmlType::Q4K,
            &weights,
            &activations,
            &mut out_avx2,
            m,
            n,
            k,
        );
        mul_mat_q_scalar(
            GgmlType::Q4K,
            &weights,
            &activations,
            &mut out_scalar,
            m,
            n,
            k,
        );

        let mut max_diff = 0.0f32;
        for j in 0..m * n {
            let diff = (out_avx2[j] - out_scalar[j]).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff < 0.01,
            "Q4K batch dual_pipeline: max_abs_diff={max_diff:.6e} exceeds tolerance"
        );
        eprintln!("Q4K batch dual_pipeline: max_abs_diff={max_diff:.6e} across {} outputs (m={m}, n={n}, k={k})", m * n);
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn weight_row_bytes_no_overflow() {
        let k: usize = kani::any();
        kani::assume(k > 0 && k <= 4096);
        kani::assume(k % 256 == 0);
        // Q4K: block_size=256, type_size=144
        let block_size = 256usize;
        let type_size = 144usize;
        let nb = k / block_size;
        let row_bytes = nb.checked_mul(type_size);
        assert!(row_bytes.is_some());
        let row_bytes = row_bytes.unwrap();
        assert!(row_bytes <= k);
    }

    #[kani::proof]
    fn dst_index_in_bounds() {
        let m: usize = kani::any();
        let n: usize = kani::any();
        kani::assume(m > 0 && m <= 8);
        kani::assume(n > 0 && n <= 8);
        let total = m * n;
        let row: usize = kani::any();
        let col: usize = kani::any();
        kani::assume(row < m && col < n);
        let idx = row * n + col;
        assert!(idx < total);
    }
}
