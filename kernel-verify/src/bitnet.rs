use rand::{Rng, SeedableRng};
use crate::KernelTestResult;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

fn random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
}

fn random_i8(len: usize, seed: u64) -> Vec<i8> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random_range(-127i8..=127i8)).collect()
}

fn random_i2s_weights(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let packed_bytes = n * k / 4;
    let mut data = vec![0u8; packed_bytes + 4]; // +4 for scale
    let row_bytes = k / 4;
    for r in 0..n {
        for c in 0..k {
            let trit: u8 = rng.random_range(0u8..=2u8); // 0=-1, 1=0, 2=+1
            let group = c / 128;
            let within = c % 128;
            let group_idx = within / 32;
            let group_pos = within % 32;
            let byte_off = r * row_bytes + group * 32 + group_pos;
            data[byte_off] |= trit << (6 - 2 * group_idx);
        }
    }
    data[packed_bytes..packed_bytes + 4].copy_from_slice(&1.0f32.to_ne_bytes());
    data
}

fn random_f16_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut bytes = vec![0u8; len * 2];
    for i in 0..len {
        let val = rng.random::<f32>() * 2.0 - 1.0;
        let h = half::f16::from_f32(val);
        let bits = h.to_bits().to_le_bytes();
        bytes[i * 2] = bits[0];
        bytes[i * 2 + 1] = bits[1];
    }
    bytes
}

// ── 1. I2S dot product: scalar vs AVX2 ────────────────────────────────────────

pub fn compare_i2s_dot(k: usize) -> KernelTestResult {
    use tempura_kernels::quantized::{vec_dot_i2_i8_s, vec_dot_i2_i8_s_avx2};

    let packed = random_i2s_weights(1, k, 100);
    let a_q = random_i8(k, 101);

    let scalar = vec_dot_i2_i8_s(&packed, &a_q, k);
    let avx2 = unsafe { vec_dot_i2_i8_s_avx2(&packed, &a_q, k) };

    let pass = scalar == avx2;
    KernelTestResult {
        test_name: format!("i2s_dot k={k}"),
        pass,
        max_abs_diff: (scalar - avx2).abs() as f32,
        l2_distance: (scalar - avx2).abs() as f32,
        detail: if pass {
            "exact i32 match".into()
        } else {
            format!("scalar={scalar} avx2={avx2}")
        },
    }
}

// ── 2. I2S quantization: scalar vs AVX2 ───────────────────────────────────────

pub fn compare_i2s_quant(k: usize) -> KernelTestResult {
    use tempura_kernels::quantized::{quantize_row_i8_s, quantize_row_i8_s_avx2_into};

    let x = random_f32(k, 200);

    let mut dst_scalar = vec![0i8; k];
    let (scale_s, sum_s) = quantize_row_i8_s(&x, &mut dst_scalar);

    let mut dst_avx2 = vec![0i8; k];
    let (scale_a, sum_a) = unsafe { quantize_row_i8_s_avx2_into(&x, &mut dst_avx2) };

    let scale_match = (scale_s - scale_a).abs() < 1e-6;
    let sum_match = sum_s == sum_a;
    let vals_match = dst_scalar == dst_avx2;
    let pass = scale_match && sum_match && vals_match;

    let detail = if pass {
        "exact match".into()
    } else {
        let mismatches: Vec<_> = dst_scalar.iter().zip(dst_avx2.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .take(5)
            .map(|(i, (a, b))| format!("[{i}]: scalar={a} avx2={b}"))
            .collect();
        format!("scale_diff={:.1e} sum_diff={} val_mismatches={}", (scale_s - scale_a).abs(), (sum_s - sum_a).abs(), mismatches.join(", "))
    };

    KernelTestResult {
        test_name: format!("i2s_quant k={k}"),
        pass,
        max_abs_diff: (scale_s - scale_a).abs(),
        l2_distance: 0.0,
        detail,
    }
}

// ── 3. TL2 end-to-end: AVX2 LUT vs scalar direct oracle ──────────────────────

pub fn compare_tl2_vs_direct(m: usize, k: usize, epsilon: f32) -> KernelTestResult {
    use tempura_kernels::tl2::{tl2_forward, tl2_gemv_direct_scalar, repack_i2s_to_tl2};
    use tempura_kernels::quantized::i2s_weight_scale;

    let i2s = random_i2s_weights(m, k, 300);
    let act = random_f32(k, 301);
    let w_scale = i2s_weight_scale(&i2s, m, k);

    // Scalar direct oracle
    let mut scalar_out = vec![0.0f32; m];
    tl2_gemv_direct_scalar(&act, &i2s, w_scale, &mut scalar_out, m, k);

    // AVX2 TL2 path
    let mut avx2_out = vec![0.0f32; m];
    let ok = if let Some((tl2_data, tl2_scale)) = repack_i2s_to_tl2(&i2s, m, k) {
        tl2_forward(&act, &tl2_data, tl2_scale, &mut avx2_out, m, k)
    } else {
        false
    };

    if !ok {
        return KernelTestResult {
            test_name: format!("tl2_vs_direct m={m} k={k}"),
            pass: false,
            max_abs_diff: f32::NAN,
            l2_distance: f32::NAN,
            detail: "TL2 repack or forward failed".into(),
        };
    }

    let mad = max_abs_diff(&scalar_out, &avx2_out);
    let l2 = l2_distance(&scalar_out, &avx2_out);
    let pass = mad <= epsilon;

    KernelTestResult {
        test_name: format!("tl2_vs_direct m={m} k={k}"),
        pass,
        max_abs_diff: mad,
        l2_distance: l2,
        detail: if pass {
            format!("max_abs_diff={mad:.2e}")
        } else {
            format!("max_abs_diff={mad:.2e} exceeds epsilon {epsilon:.2e}")
        },
    }
}

// ── 4. F16 dot product: scalar vs AVX2 ────────────────────────────────────────

pub fn compare_f16_dot(k: usize, epsilon: f32) -> KernelTestResult {
    use tempura_kernels::f16_gemv::{f16_dot_scalar, f16_dot_avx2};

    let a = random_f32(k, 400);
    let b = random_f16_bytes(k, 401);

    let scalar = f16_dot_scalar(&a, &b, k);
    let avx2 = unsafe { f16_dot_avx2(&a, &b, k) };

    let diff = (scalar - avx2).abs();
    let pass = diff <= epsilon;

    KernelTestResult {
        test_name: format!("f16_dot k={k}"),
        pass,
        max_abs_diff: diff,
        l2_distance: diff,
        detail: if pass {
            format!("diff={diff:.2e}")
        } else {
            format!("scalar={scalar} avx2={avx2} diff={diff:.2e}")
        },
    }
}

// ── 5. Per-tensor quant: scalar vs AVX2 ───────────────────────────────────────

pub fn compare_per_tensor_quant(k: usize) -> KernelTestResult {
    use tempura_kernels::tl2::{per_tensor_quant, per_tensor_quant_scalar};

    let b = random_f32(k, 500);
    let scalar = per_tensor_quant_scalar(&b);
    let avx2 = unsafe { per_tensor_quant(k, &b) };

    let diff = (scalar - avx2).abs();
    let pass = diff < 1e-6;

    KernelTestResult {
        test_name: format!("per_tensor_quant k={k}"),
        pass,
        max_abs_diff: diff,
        l2_distance: diff,
        detail: if pass {
            "exact match".into()
        } else {
            format!("scalar={scalar} avx2={avx2}")
        },
    }
}

// ── 6. Fused SqReLU*Mul: scalar vs AVX2 ──────────────────────────────────────

pub fn compare_fused_sqrelu(n: usize) -> KernelTestResult {
    use tempura_kernels::fused::fused_sqrelu_mul_avx2;

    let gate_orig = random_f32(n, 600);
    let up = random_f32(n, 601);

    // Scalar path
    let mut gate_scalar = gate_orig.clone();
    for i in 0..n {
        let r = gate_scalar[i].max(0.0);
        gate_scalar[i] = r * r * up[i];
    }

    // AVX2 path
    let mut gate_avx2 = gate_orig.clone();
    unsafe { fused_sqrelu_mul_avx2(&mut gate_avx2, &up, n) };

    let mad = max_abs_diff(&gate_scalar, &gate_avx2);
    let l2 = l2_distance(&gate_scalar, &gate_avx2);
    let pass = mad < 1e-6;

    KernelTestResult {
        test_name: format!("fused_sqrelu n={n}"),
        pass,
        max_abs_diff: mad,
        l2_distance: l2,
        detail: if pass { "exact match".into() } else { format!("mad={mad:.2e}") },
    }
}

// ── 7. Fused residual_add: scalar vs AVX2 ─────────────────────────────────────

pub fn compare_fused_residual_add(n: usize) -> KernelTestResult {
    use tempura_kernels::fused::residual_add_avx2;

    let dst_orig = random_f32(n, 700);
    let src = random_f32(n, 701);

    // Scalar path
    let mut dst_scalar = dst_orig.clone();
    for i in 0..n {
        dst_scalar[i] += src[i];
    }

    // AVX2 path
    let mut dst_avx2 = dst_orig.clone();
    unsafe { residual_add_avx2(&mut dst_avx2, &src, n) };

    let mad = max_abs_diff(&dst_scalar, &dst_avx2);
    let l2 = l2_distance(&dst_scalar, &dst_avx2);
    let pass = mad < 1e-6;

    KernelTestResult {
        test_name: format!("fused_residual_add n={n}"),
        pass,
        max_abs_diff: mad,
        l2_distance: l2,
        detail: if pass { "exact match".into() } else { format!("mad={mad:.2e}") },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i2s_dot_scalar_vs_avx2() {
        for k in [128, 256, 512, 1024, 2560] {
            let r = compare_i2s_dot(k);
            assert!(r.pass, "{}: {}", r.test_name, r.detail);
        }
    }

    #[test]
    fn test_i2s_quant_scalar_vs_avx2() {
        for k in [32, 64, 128, 256, 512, 2560] {
            let r = compare_i2s_quant(k);
            assert!(r.pass, "{}: {}", r.test_name, r.detail);
        }
    }

    #[test]
    fn test_tl2_vs_direct_scalar() {
        for (m, k) in [(128, 128), (256, 384), (128, 256), (256, 256)] {
            let r = compare_tl2_vs_direct(m, k, 0.0);
            assert!(r.pass, "{}: max_abs_diff={:.2e} {}", r.test_name, r.max_abs_diff, r.detail);
        }
    }

    #[test]
    fn test_f16_dot_bit_exact() {
        for k in [8, 16, 32, 64, 128, 256, 512, 2560] {
            let r = compare_f16_dot(k, 0.0);
            assert_eq!(r.max_abs_diff, 0.0, "{}: not bit-exact, diff={:.2e}", r.test_name, r.max_abs_diff);
        }
    }

    #[test]
    fn test_per_tensor_quant_bit_exact() {
        for k in [8, 16, 32, 64, 128, 256, 2560] {
            let r = compare_per_tensor_quant(k);
            assert_eq!(r.max_abs_diff, 0.0, "{}: not bit-exact, diff={:.2e}", r.test_name, r.max_abs_diff);
        }
    }

    #[test]
    fn test_fused_sqrelu_bit_exact() {
        for n in [1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 2560] {
            let r = compare_fused_sqrelu(n);
            assert_eq!(r.max_abs_diff, 0.0, "{}: not bit-exact, diff={:.2e}", r.test_name, r.max_abs_diff);
        }
    }

    #[test]
    fn test_fused_residual_add_bit_exact() {
        for n in [1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 2560] {
            let r = compare_fused_residual_add(n);
            assert_eq!(r.max_abs_diff, 0.0, "{}: not bit-exact, diff={:.2e}", r.test_name, r.max_abs_diff);
        }
    }
}
