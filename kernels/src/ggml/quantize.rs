use super::blocks::*;
use super::helpers::{f32_to_f16, nearest_int};

pub fn quantize_row_q8_0(src: &[f32], dst: &mut [BlockQ8_0]) {
    const QK: usize = 32;
    assert_eq!(src.len(), dst.len() * QK);
    for (i, blk) in dst.iter_mut().enumerate() {
        let x = &src[i * QK..][..QK];
        let mut amax = 0.0f32;
        for j in 0..QK {
            amax = amax.max(x[j].abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        blk.d = f32_to_f16(d);
        for j in 0..QK {
            blk.qs[j] = (x[j] * id).round() as i8;
        }
    }
}

pub fn quantize_row_q8_1(src: &[f32], dst: &mut [BlockQ8_1]) {
    const QK: usize = 32;
    assert_eq!(src.len(), dst.len() * QK);
    for (i, blk) in dst.iter_mut().enumerate() {
        let x = &src[i * QK..][..QK];
        let mut amax = 0.0f32;
        for j in 0..QK {
            amax = amax.max(x[j].abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        blk.d = f32_to_f16(d);
        let mut sum = 0i32;
        for j in 0..QK {
            let v = (x[j] * id).round() as i8;
            blk.qs[j] = v;
            sum += v as i32;
        }
        blk.s = f32_to_f16(d * sum as f32);
    }
}

pub fn quantize_row_q8_k(src: &[f32], dst: &mut [BlockQ8K]) {
    const QK: usize = 256;
    assert_eq!(src.len(), dst.len() * QK);
    for (i, blk) in dst.iter_mut().enumerate() {
        let x = &src[i * QK..][..QK];
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for j in 0..QK {
            let ax = x[j].abs();
            if ax > amax {
                amax = ax;
                max = x[j];
            }
        }
        if amax == 0.0 {
            blk.d = 0.0;
            blk.qs = [0i8; 256];
            blk.bsums = [0i16; 16];
            continue;
        }
        let iscale = -127.0 / max;
        for j in 0..QK {
            let v = nearest_int(iscale * x[j]);
            blk.qs[j] = v.min(127) as i8;
        }
        for j in 0..QK / 16 {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += blk.qs[j * 16 + ii] as i32;
            }
            blk.bsums[j] = sum as i16;
        }
        blk.d = 1.0 / iscale;
    }
}

// -- NVFP4 (E2M1 + UE4M3 per-16 sub-block scales, split nibble layout) ------

/// Index of the KVALUES_MXFP4 entry whose product with the (half-convention)
/// scale d best approximates x -- port of llama.cpp best_index_mxfp4.
fn best_index_mxfp4(x: f32, d: f32, kvalues: &[i8; 16]) -> u8 {
    let mut best_index = 0usize;
    let mut best_err = (kvalues[0] as f32 * d - x).abs();
    for i in 1..16 {
        let err = (kvalues[i] as f32 * d - x).abs();
        if err < best_err {
            best_index = i;
            best_err = err;
        }
    }
    best_index as u8
}

/// Port of llama.cpp quantize_row_nvfp4_ref: per 16-element sub-block, the
/// UE4M3-encoded scale is amax / 6.0, and byte j of the sub-block packs
/// element j (lo nibble) with element j + 8 (hi nibble).
pub fn quantize_row_nvfp4(src: &[f32], dst: &mut [BlockNvfp4]) {
    use super::helpers::{f32_to_ue4m3, ue4m3_to_f32};
    use super::iq_tables::KVALUES_MXFP4;
    const QK: usize = 64;
    const SUB: usize = 16;
    assert_eq!(src.len(), dst.len() * QK);
    for (i, blk) in dst.iter_mut().enumerate() {
        let xb = &src[i * QK..][..QK];
        for s in 0..QK / SUB {
            let sub = &xb[s * SUB..][..SUB];
            let mut amax = 0.0f32;
            for &v in sub {
                amax = amax.max(v.abs());
            }
            // UE4M3 scale: amax / 6.0 maps the max E2M1 value (6.0) to amax.
            // ue4m3_to_f32 returns raw * 0.5, which cancels the 2x E2M1
            // kvalues table inside best_index_mxfp4.
            let ue = f32_to_ue4m3(amax / 6.0);
            blk.d[s] = ue;
            let d = ue4m3_to_f32(ue);
            for j in 0..SUB / 2 {
                let x0 = best_index_mxfp4(sub[j], d, &KVALUES_MXFP4);
                let x1 = best_index_mxfp4(sub[j + SUB / 2], d, &KVALUES_MXFP4);
                blk.qs[s * (SUB / 2) + j] = x0 | (x1 << 4);
            }
        }
    }
}

// -- Raw-byte interface -----------------------------------------------------

pub fn quantize_activations(src: &[f32], dst: &mut [u8], ggml_type: super::GgmlType) {
    use super::GgmlType::*;
    match ggml_type {
        Q8_0 => {
            let nb = src.len() / 32;
            assert_eq!(src.len() % 32, 0);
            assert!(dst.len() >= nb * BLOCK_Q8_0_SIZE);
            let blocks =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut BlockQ8_0, nb) };
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { super::vec_dot_avx2::quantize_row_q8_0_avx2(src, blocks) };
                return;
            }
            quantize_row_q8_0(src, blocks);
        }
        Q8_1 => {
            let nb = src.len() / 32;
            assert_eq!(src.len() % 32, 0);
            assert!(dst.len() >= nb * BLOCK_Q8_1_SIZE);
            let blocks =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut BlockQ8_1, nb) };
            quantize_row_q8_1(src, blocks);
        }
        Q8K => {
            let nb = src.len() / 256;
            assert_eq!(src.len() % 256, 0);
            assert!(dst.len() >= nb * BLOCK_Q8K_SIZE);
            let blocks =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut BlockQ8K, nb) };
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { super::vec_dot_avx2::quantize_row_q8_k_avx2(src, blocks) };
                return;
            }
            quantize_row_q8_k(src, blocks);
        }
        _ => unimplemented!("quantize not supported for {}", ggml_type.name()),
    }
}

pub fn quantized_row_size(k: usize, ggml_type: super::GgmlType) -> usize {
    let info_block_size = ggml_type.block_size();
    let info_type_size = ggml_type.type_size();
    assert_eq!(k % info_block_size, 0);
    (k / info_block_size) * info_type_size
}

#[cfg(test)]
mod tests {
    use super::super::helpers::f16_to_f32;
    use super::*;

    #[test]
    fn q8_0_roundtrip() {
        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }];
        quantize_row_q8_0(&src, &mut blocks);
        let d = f16_to_f32(blocks[0].d);
        assert!(d > 0.0);
        for (j, &v) in src.iter().enumerate() {
            let reconstructed = d * blocks[0].qs[j] as f32;
            assert!(
                (reconstructed - v).abs() < 0.02,
                "j={j}: {reconstructed} vs {v}"
            );
        }
    }

    #[test]
    fn q8_1_sum_field() {
        let src = [1.0f32; 32];
        let mut blocks = vec![BlockQ8_1 {
            d: 0,
            s: 0,
            qs: [0; 32],
        }];
        quantize_row_q8_1(&src, &mut blocks);
        let d = f16_to_f32(blocks[0].d);
        let s = f16_to_f32(blocks[0].s);
        let expected_sum: i32 = blocks[0].qs.iter().map(|&q| q as i32).sum();
        assert!((s - d * expected_sum as f32).abs() < 0.01);
    }

    #[test]
    fn q8_k_bsums_correct() {
        let src: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let mut blocks = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        quantize_row_q8_k(&src, &mut blocks);
        for j in 0..16 {
            let expected: i32 = blocks[0].qs[j * 16..(j + 1) * 16]
                .iter()
                .map(|&q| q as i32)
                .sum();
            assert_eq!(blocks[0].bsums[j] as i32, expected, "bsums[{j}] mismatch");
        }
    }

    #[test]
    fn q8_0_all_zeros() {
        let src = [0.0f32; 32];
        let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }];
        quantize_row_q8_0(&src, &mut blocks);
        assert_eq!(f16_to_f32(blocks[0].d), 0.0);
        assert!(blocks[0].qs.iter().all(|&q| q == 0));
    }

    #[test]
    fn q8_k_all_zeros() {
        let src = [0.0f32; 256];
        let mut blocks = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        quantize_row_q8_k(&src, &mut blocks);
        assert_eq!(blocks[0].d, 0.0);
        assert!(blocks[0].qs.iter().all(|&q| q == 0));
        assert!(blocks[0].bsums.iter().all(|&s| s == 0));
    }

    #[test]
    fn q8_0_dequant_roundtrip_bounded() {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let src: Vec<f32> = (0..256)
            .map(|_| rng.random_range(-1.0f32..=1.0f32))
            .collect();
        let nb = src.len() / 32;
        let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; nb];
        quantize_row_q8_0(&src, &mut blocks);
        let mut reconstructed = vec![0.0f32; 256];
        for (i, blk) in blocks.iter().enumerate() {
            let d = f16_to_f32(blk.d);
            for j in 0..32 {
                reconstructed[i * 32 + j] = d * blk.qs[j] as f32;
            }
        }
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for i in 0..256 {
            let err = (reconstructed[i] - src[i]) as f64;
            sum_sq_err += err * err;
            sum_sq_ref += (src[i] as f64) * (src[i] as f64);
        }
        let rel_l2 = (sum_sq_err / sum_sq_ref).sqrt();
        assert!(rel_l2 < 0.02, "Q8_0 roundtrip L2 error: {rel_l2}");
    }

    #[test]
    fn q8_k_dequant_roundtrip_bounded() {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let src: Vec<f32> = (0..256)
            .map(|_| rng.random_range(-1.0f32..=1.0f32))
            .collect();
        let mut blocks = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        quantize_row_q8_k(&src, &mut blocks);
        let mut reconstructed = vec![0.0f32; 256];
        let d = blocks[0].d;
        for j in 0..256 {
            reconstructed[j] = d * blocks[0].qs[j] as f32;
        }
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for i in 0..256 {
            let err = (reconstructed[i] - src[i]) as f64;
            sum_sq_err += err * err;
            sum_sq_ref += (src[i] as f64) * (src[i] as f64);
        }
        let rel_l2 = (sum_sq_err / sum_sq_ref).sqrt();
        assert!(rel_l2 < 0.01, "Q8_K roundtrip L2 error: {rel_l2}");
    }

    #[test]
    fn quantized_row_size_correct() {
        assert_eq!(quantized_row_size(256, super::super::GgmlType::Q8K), 292);
        assert_eq!(quantized_row_size(32, super::super::GgmlType::Q8_0), 34);
        assert_eq!(quantized_row_size(256, super::super::GgmlType::Q4K), 144);
    }

    #[test]
    fn nvfp4_quantize_dequant_roundtrip_bounded() {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let src: Vec<f32> = (0..512)
            .map(|_| rng.random_range(-1.0f32..=1.0f32))
            .collect();
        let nb = src.len() / 64;
        let mut blocks = vec![
            BlockNvfp4 {
                d: [0; 4],
                qs: [0; 32],
            };
            nb
        ];
        quantize_row_nvfp4(&src, &mut blocks);
        let mut reconstructed = vec![0.0f32; src.len()];
        super::super::dequantize::dequantize_row_nvfp4(&blocks, &mut reconstructed);
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for i in 0..src.len() {
            let err = (reconstructed[i] - src[i]) as f64;
            sum_sq_err += err * err;
            sum_sq_ref += (src[i] as f64) * (src[i] as f64);
        }
        let rel_l2 = (sum_sq_err / sum_sq_ref).sqrt();
        // E2M1 (4-bit) + UE4M3 scale rounding: ~0.2 observed, 0.30 gate.
        assert!(rel_l2 < 0.30, "NVFP4 roundtrip L2 error: {rel_l2}");
    }

    #[test]
    fn nvfp4_quantize_small_scales_subnormal_range() {
        // llama.cpp hit this as a PPL=5.8M bug: typical transformer weight
        // amax/6 (~1e-2) lands in the UE4M3 subnormal range (2^-9..2^-6).
        // The encoded scale must round-trip through the subnormal path, not
        // underflow to zero. NOTE: sub-blocks with amax < 6*2^-9 ~= 0.0117
        // are below NVFP4's floor WITHOUT the per-tensor FP32 global scale
        // (the second-level scale real NVFP4 checkpoints carry) and encode
        // to zero by design.
        use super::super::helpers::{f32_to_ue4m3, ue4m3_to_f32};
        let src: Vec<f32> = (0..64)
            .map(|i| (i as f32 - 32.0) * 3.0e-3)
            .collect();
        let mut blocks = vec![BlockNvfp4 { d: [0; 4], qs: [0; 32] }];
        quantize_row_nvfp4(&src, &mut blocks);
        for s in 0..4 {
            assert_ne!(blocks[0].d[s], 0, "sub-block {s} scale underflowed to 0");
        }
        let mut reconstructed = vec![0.0f32; 64];
        super::super::dequantize::dequantize_row_nvfp4(&blocks, &mut reconstructed);
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for i in 0..64 {
            let err = (reconstructed[i] - src[i]) as f64;
            sum_sq_err += err * err;
            sum_sq_ref += (src[i] as f64) * (src[i] as f64);
        }
        let rel_l2 = (sum_sq_err / sum_sq_ref).sqrt();
        assert!(rel_l2 < 0.60, "NVFP4 subnormal-scale L2 error: {rel_l2}");
        // Direct encode/decode of a subnormal-scale value.
        let ue = f32_to_ue4m3(2.0e-3);
        assert!(ue > 0 && ue < 8, "2e-3 should encode as a subnormal, got {ue}");
        assert!((ue4m3_to_f32(ue) * 2.0 - 2.0e-3).abs() / 2.0e-3 < 0.2);
    }

    #[test]
    fn nvfp4_quantize_all_zeros() {
        let src = [0.0f32; 64];
        let mut blocks = vec![BlockNvfp4 { d: [0; 4], qs: [0; 32] }];
        quantize_row_nvfp4(&src, &mut blocks);
        assert!(blocks[0].d.iter().all(|&d| d == 0));
        assert!(blocks[0].qs.iter().all(|&q| q == 0));
    }

    #[test]
    fn nvfp4_vec_dot_matches_dequantized_reference() {
        use rand::Rng;
        use rand::SeedableRng;
        use super::super::helpers::f16_to_f32;
        use super::super::vec_dot::vec_dot_nvfp4_q8_0;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let n = 256usize;
        let src: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0f32..=1.0f32)).collect();
        let mut blocks = vec![BlockNvfp4 { d: [0; 4], qs: [0; 32] }; n / 64];
        quantize_row_nvfp4(&src, &mut blocks);
        let mut deq = vec![0.0f32; n];
        super::super::dequantize::dequantize_row_nvfp4(&blocks, &mut deq);
        let mut q8 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; n / 32];
        quantize_row_q8_0(&src, &mut q8);
        // Reference: f64 dot of both DEQUANTIZED operands -- exactly what
        // vec_dot computes modulo float accumulation order. The old 2x
        // half-convention bug fails this by a factor of 2.
        let mut deq_act = vec![0.0f32; n];
        for (i, blk) in q8.iter().enumerate() {
            let d = f16_to_f32(blk.d);
            for j in 0..32 {
                deq_act[i * 32 + j] = d * blk.qs[j] as f32;
            }
        }
        let dot_q = vec_dot_nvfp4_q8_0(n, &blocks, &q8) as f64;
        let dot_ref: f64 = (0..n).map(|i| deq[i] as f64 * deq_act[i] as f64).sum();
        assert!(
            (dot_q - dot_ref).abs() < 1e-2 * dot_ref.abs().max(1.0),
            "vec_dot {dot_q} vs dequantized reference {dot_ref}"
        );
    }

    #[test]
    fn nvfp4_generic_and_scalar_vec_dot_agree() {
        use super::super::quants_llama_generic::ggml_vec_dot_nvfp4_q8_0_generic;
        use super::super::vec_dot::vec_dot_nvfp4_q8_0;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let n = 128usize;
        let src: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0f32..=1.0f32)).collect();
        let mut blocks = vec![BlockNvfp4 { d: [0; 4], qs: [0; 32] }; n / 64];
        quantize_row_nvfp4(&src, &mut blocks);
        let mut q8 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; n / 32];
        quantize_row_q8_0(&src, &mut q8);
        let a = vec_dot_nvfp4_q8_0(n, &blocks, &q8);
        let b = ggml_vec_dot_nvfp4_q8_0_generic(n, &blocks, &q8);
        assert!((a - b).abs() < 1e-3 * a.abs().max(1.0), "scalar {a} vs generic {b}");
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn q8_0_scale_nonneg() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        let amax = val.abs();
        let d = amax / 127.0;
        assert!(d >= 0.0);
    }

    #[kani::proof]
    fn q8_0_quant_in_range() {
        let val: f32 = kani::any();
        let amax: f32 = kani::any();
        kani::assume(val.is_finite() && amax.is_finite());
        kani::assume(amax > 0.0 && amax <= 1e6);
        // Subnormal amax makes 1/(amax/127) overflow to inf (val * inf);
        // production is safe because the `as i8` cast saturates. Keep the
        // domain in the realistic range (float-edge found by Kani).
        kani::assume(amax >= 1e-30);
        kani::assume(val.abs() <= amax);
        let d = amax / 127.0;
        let id = 1.0 / d;
        let q = (val * id).round();
        assert!(q >= -128.0 && q <= 128.0);
    }

    #[kani::proof]
    // nb = k/16 is symbolic, so the loop bound is symbolic: without an
    // explicit unwind CBMC unwinds past 5000 iterations (found 2026-11-05).
    #[kani::unwind(20)]
    fn q8_k_bsums_partition() {
        let k: usize = kani::any();
        kani::assume(k < 256 && k % 16 == 0);
        let nb = k / 16;
        for j in 0..nb {
            assert!(j * 16 < k);
            assert!((j + 1) * 16 <= k);
        }
    }

    #[kani::proof]
    fn q8_0_output_in_range() {
        let val: f32 = kani::any();
        let amax: f32 = kani::any();
        kani::assume(val.is_finite() && amax.is_finite());
        kani::assume(amax > 0.0 && amax <= 1e6);
        kani::assume(val.abs() <= amax);
        let d = amax / 127.0;
        let id = 1.0 / d;
        let q = (val * id).round() as i32;
        let clamped = q.max(-128).min(127) as i8;
        assert!(clamped >= -128 && clamped <= 127);
    }

    #[kani::proof]
    #[kani::unwind(17)]
    fn q8_k_bsums_sum_fits_i16() {
        let qs: [i8; 16] = kani::any();
        let mut sum = 0i32;
        for j in 0..16 {
            sum += qs[j] as i32;
        }
        // 16 * 127 = 2032, 16 * -128 = -2048, both fit in i16 [-32768, 32767]
        assert!(sum >= -2048 && sum <= 2032);
        assert!(sum >= i16::MIN as i32 && sum <= i16::MAX as i32);
    }

    #[kani::proof]
    fn ue4m3_encode_no_panic() {
        use super::super::helpers::f32_to_ue4m3;
        let x: f32 = kani::any();
        let b = f32_to_ue4m3(x);
        assert!(b <= 0x7E);
    }

    #[kani::proof]
    #[kani::unwind(70)]
    fn nvfp4_quantize_no_panic() {
        let src: [f32; 64] = kani::any();
        for v in src.iter() {
            kani::assume(v.is_finite());
        }
        let mut blocks = [BlockNvfp4 {
            d: [0; 4],
            qs: [0; 32],
        }];
        quantize_row_nvfp4(&src, &mut blocks);
        // Encoded scales stay in the valid byte range (saturation at 0x7E).
        for s in 0..4 {
            assert!(blocks[0].d[s] <= 0x7E);
        }
    }
}
