/// Convert f16 bits to f32. Uses F16C hardware instruction when compiled with
/// target-feature=+f16c (enabled via .cargo/config.toml target-cpu=native),
/// which matches llama.cpp's vcvtph2ps — 1 instruction vs 6+ for software.
#[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    unsafe {
        use std::arch::x86_64::{_mm_cvtph_ps, _mm_cvtss_f32, _mm_set1_epi16};
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(bits as i16)))
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "f16c")))]
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp == 0xff {
        return (sign << 15) | (0x1f << 10) | ((mant >> 13) as u16 & 0x3ff);
    }

    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        return (sign << 15) | (0x1f << 10);
    }

    if new_exp <= 0 {
        if new_exp < -10 {
            return sign << 15;
        }
        let m = (mant | 0x800000) >> (1 - new_exp + 13);
        return (sign << 15) | (m as u16 & 0x3ff);
    }

    // round-to-nearest-even (IEEE 754)
    let truncated = (mant >> 13) as u16 & 0x3ff;
    let remainder = mant & 0x1fff;
    let round_up = remainder > 0x1000 || (remainder == 0x1000 && (truncated & 1) != 0);
    let mut h = (sign << 15) | ((new_exp as u16) << 10) | truncated;
    if round_up {
        h += 1;
    }
    h
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

pub fn nearest_int(val: f32) -> i32 {
    assert!(val <= 4194303.0);
    let magic = 12582912.0f32; // 0x4B400000
    let val_plus_magic = val + magic;
    let bits = val_plus_magic.to_bits() as i32;
    bits - magic.to_bits() as i32
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn f16_to_f32_no_panic() {
        let bits: u16 = kani::any();
        let _ = f16_to_f32(bits);
    }

    #[kani::proof]
    fn f16_to_f32_finite_for_normal() {
        let bits: u16 = kani::any();
        let exp = (bits >> 10) & 0x1f;
        kani::assume(exp > 0 && exp < 31);
        let result = f16_to_f32(bits);
        assert!(result.is_finite());
    }

    #[kani::proof]
    fn f32_to_f16_no_panic() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        let _ = f32_to_f16(val);
    }

    #[kani::proof]
    fn get_scale_min_k4_no_oob() {
        let j: usize = kani::any();
        kani::assume(j < 8);
        let scales: [u8; 12] = kani::any();
        let (s, m) = get_scale_min_k4(j, &scales);
        assert!(s <= 63);
        assert!(m <= 63);
    }

    #[kani::proof]
    fn nearest_int_no_panic() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val.abs() < 4194303.0);
        kani::assume(val <= 4194303.0);
        let _ = nearest_int(val);
    }

    #[kani::proof]
    fn bf16_to_f32_no_panic() {
        let bits: u16 = kani::any();
        let _ = bf16_to_f32(bits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 65504.0, 6.1e-5] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            let rel = if v.abs() > 0.0 {
                ((back - v) / v).abs()
            } else {
                (back - v).abs()
            };
            assert!(
                rel < 0.002,
                "f16 roundtrip failed for {v}: got {back}, rel_err={rel}"
            );
        }
    }

    #[test]
    fn bf16_known_values() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
    }

    #[test]
    fn get_scale_min_k4_low_indices() {
        let scales = [10, 20, 30, 40, 50, 60, 70, 80, 0, 0, 0, 0];
        let (s, m) = get_scale_min_k4(0, &scales);
        assert_eq!(s, 10 & 63);
        assert_eq!(m, 50 & 63);
    }

    #[test]
    fn f16_f32_roundtrip_error_bounded() {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..10000 {
            let x: f32 = rng.random_range(-65504.0f32..=65504.0f32);
            let h = f32_to_f16(x);
            let back = f16_to_f32(h);
            if x.abs() > 1e-4 {
                let rel = ((back - x) / x).abs();
                assert!(
                    rel < 0.001,
                    "f16 roundtrip: x={x}, back={back}, rel_err={rel}"
                );
            }
        }
    }

    #[test]
    fn f16_preserves_sign() {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..10000 {
            let x: f32 = rng.random_range(-65504.0f32..=65504.0f32);
            if x == 0.0 {
                continue;
            }
            let h = f32_to_f16(x);
            let back = f16_to_f32(h);
            if back != 0.0 {
                assert_eq!(
                    x.is_sign_positive(),
                    back.is_sign_positive(),
                    "sign mismatch: x={x}, back={back}"
                );
            }
        }
    }

    #[test]
    fn f16_monotonic() {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..10000 {
            let a: f32 = rng.random_range(-65504.0f32..65504.0f32);
            let b: f32 = rng.random_range(a..=65504.0f32);
            if a >= b {
                continue;
            }
            let ha = f32_to_f16(a);
            let hb = f32_to_f16(b);
            let fa = f16_to_f32(ha);
            let fb = f16_to_f32(hb);
            assert!(
                fa <= fb,
                "monotonicity: a={a} -> f16 -> {fa}, b={b} -> f16 -> {fb}"
            );
        }
    }

    #[test]
    fn nearest_int_rounds_correctly() {
        assert_eq!(nearest_int(0.5), 0);
        assert_eq!(nearest_int(1.5), 2);
        assert_eq!(nearest_int(2.5), 2);
        assert_eq!(nearest_int(0.7), 1);
        assert_eq!(nearest_int(-0.7), -1);
    }
}
