pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // denormalized
        let mut e = 0u32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let f32_exp = (127u32 - 15 + 1).wrapping_sub(e);
        let f32_mant = (m & 0x3ff) << 13;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant);
    }

    if exp == 31 {
        let f32_mant = mant << 13;
        return f32::from_bits((sign << 31) | (0xff << 23) | f32_mant);
    }

    let f32_exp = exp + (127 - 15);
    let f32_mant = mant << 13;
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
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

    (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16 & 0x3ff)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 65504.0, 6.1e-5] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            let rel = if v.abs() > 0.0 { ((back - v) / v).abs() } else { (back - v).abs() };
            assert!(rel < 0.002, "f16 roundtrip failed for {v}: got {back}, rel_err={rel}");
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
    fn nearest_int_rounds_correctly() {
        assert_eq!(nearest_int(0.5), 0);
        assert_eq!(nearest_int(1.5), 2);
        assert_eq!(nearest_int(2.5), 2);
        assert_eq!(nearest_int(0.7), 1);
        assert_eq!(nearest_int(-0.7), -1);
    }
}
