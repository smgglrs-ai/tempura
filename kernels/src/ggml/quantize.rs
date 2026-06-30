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

// ── Raw-byte interface ─────────────────────────────────────────────────────

pub fn quantize_activations(src: &[f32], dst: &mut [u8], ggml_type: super::GgmlType) {
    use super::GgmlType::*;
    match ggml_type {
        Q8_0 => {
            let nb = src.len() / 32;
            assert_eq!(src.len() % 32, 0);
            assert!(dst.len() >= nb * BLOCK_Q8_0_SIZE);
            let blocks = unsafe {
                std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut BlockQ8_0, nb)
            };
            quantize_row_q8_0(src, blocks);
        }
        Q8_1 => {
            let nb = src.len() / 32;
            assert_eq!(src.len() % 32, 0);
            assert!(dst.len() >= nb * BLOCK_Q8_1_SIZE);
            let blocks = unsafe {
                std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut BlockQ8_1, nb)
            };
            quantize_row_q8_1(src, blocks);
        }
        Q8K => {
            let nb = src.len() / 256;
            assert_eq!(src.len() % 256, 0);
            assert!(dst.len() >= nb * BLOCK_Q8K_SIZE);
            let blocks = unsafe {
                std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut BlockQ8K, nb)
            };
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
    use super::*;
    use super::super::helpers::f16_to_f32;

    #[test]
    fn q8_0_roundtrip() {
        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }];
        quantize_row_q8_0(&src, &mut blocks);
        let d = f16_to_f32(blocks[0].d);
        assert!(d > 0.0);
        for (j, &v) in src.iter().enumerate() {
            let reconstructed = d * blocks[0].qs[j] as f32;
            assert!((reconstructed - v).abs() < 0.02, "j={j}: {reconstructed} vs {v}");
        }
    }

    #[test]
    fn q8_1_sum_field() {
        let src = [1.0f32; 32];
        let mut blocks = vec![BlockQ8_1 { d: 0, s: 0, qs: [0; 32] }];
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
    fn quantized_row_size_correct() {
        assert_eq!(quantized_row_size(256, super::super::GgmlType::Q8K), 292);
        assert_eq!(quantized_row_size(32, super::super::GgmlType::Q8_0), 34);
        assert_eq!(quantized_row_size(256, super::super::GgmlType::Q4K), 144);
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
        kani::assume(val.abs() <= amax);
        let d = amax / 127.0;
        let id = 1.0 / d;
        let q = (val * id).round();
        assert!(q >= -128.0 && q <= 128.0);
    }

    #[kani::proof]
    fn q8_k_bsums_partition() {
        let k: usize = kani::any();
        kani::assume(k < 256 && k % 16 == 0);
        let nb = k / 16;
        for j in 0..nb {
            assert!(j * 16 < k);
            assert!((j + 1) * 16 <= k);
        }
    }
}
