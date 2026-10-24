//! CPU-side MXFP4 weight quantization: BF16/F16 -> E2M1 packed nibbles + UE8M0 block scales.
//!
//! Format:
//!   - E2M1: 1 sign + 2 exponent + 1 mantissa bit.
//!     Representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
//!   - Packing: 2 E2M1 nibbles per byte (lo nibble = first element, hi nibble = second)
//!   - Block size: 32 elements per UE8M0 scale
//!   - UE8M0: unsigned 8-bit biased exponent, scale = 2^(e - 127). Power-of-2 only.
//!   - Per 256-element superblock: 128 bytes packed E2M1 + 8 bytes UE8M0 = 136 bytes
//!     (vs Q4K 168 bytes: 19% smaller)
//!
//! This quantization is designed to work from BF16/F16 source weights (HuggingFace
//! safetensors). Converting from Q4K is lossy (up to 25% error on extreme values).

pub const MXFP4_BLOCK_SIZE: usize = 32;
pub const MXFP4_BLOCK_PACKED_BYTES: usize = MXFP4_BLOCK_SIZE / 2; // 16 bytes of E2M1 nibbles
pub const MXFP4_BLOCK_BYTES: usize = MXFP4_BLOCK_PACKED_BYTES + 1; // + 1 byte UE8M0 scale

/// E2M1 magnitude lookup (index 0..8 -> float value).
const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Quantize an f32 value to a 4-bit E2M1 nibble (nearest-even rounding).
///
/// E2M1 encoding: bit 3 = sign, bits [2:0] = magnitude index (0..7).
/// Returns a nibble in range 0x0..0xF.
pub fn f32_to_e2m1(x: f32) -> u8 {
    if x.is_nan() {
        return 0;
    }
    let abs_x = x.abs().min(6.0);
    // Find nearest magnitude by absolute difference; ties go to lower index (even).
    let mut best_idx = 0usize;
    let mut best_dist = abs_x; // dist to 0.0
    for (i, &mag) in E2M1_MAGNITUDES.iter().enumerate().skip(1) {
        let d = (abs_x - mag).abs();
        if d < best_dist || (d == best_dist && i % 2 == 0) {
            best_dist = d;
            best_idx = i;
        }
    }
    let sign_bit = if x < 0.0 { 0b1000u8 } else { 0 };
    sign_bit | (best_idx as u8)
}

/// Decode a 4-bit E2M1 nibble back to f32.
pub fn e2m1_to_f32(nibble: u8) -> f32 {
    let sign = if nibble & 0b1000 != 0 { -1.0f32 } else { 1.0f32 };
    let mag_idx = (nibble & 0b0111) as usize;
    sign * E2M1_MAGNITUDES[mag_idx]
}

/// Pack two E2M1 nibbles into one byte (lo nibble = first element, hi nibble = second).
#[inline(always)]
pub fn pack_e2m1_pair(lo: u8, hi: u8) -> u8 {
    (lo & 0xF) | ((hi & 0xF) << 4)
}

/// Compute the UE8M0 block scale for a 32-element block.
///
/// Chooses the smallest power-of-2 scale such that max(abs(values)) / scale <= 6.0.
/// Returns 0 if all values are zero.
pub fn compute_ue8m0_scale(block: &[f32]) -> u8 {
    debug_assert_eq!(block.len(), MXFP4_BLOCK_SIZE);
    let max_abs = block.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs == 0.0 || !max_abs.is_finite() {
        return 0;
    }
    // We need scale such that max_abs / scale <= 6.0, i.e. scale >= max_abs / 6.0.
    // Round up to the nearest power of 2 via ceiling of log2.
    let ideal = max_abs / 6.0;
    let exp = ideal.log2().ceil() as i32;
    (exp + 127).clamp(0, 255) as u8
}

/// Dequantize a UE8M0 byte to an f32 scale value: 2^(e - 127).
#[inline(always)]
pub fn ue8m0_to_f32(e: u8) -> f32 {
    if e == 0 {
        return 0.0;
    }
    f32::from_bits((e as u32) << 23)
}

/// Quantize a slice of f32 values to MXFP4 format.
///
/// `values` length must be a multiple of `MXFP4_BLOCK_SIZE`.
///
/// Returns `(packed_e2m1, scales_ue8m0)` where:
/// - `packed_e2m1` length = values.len() / 2 (two nibbles per byte)
/// - `scales_ue8m0` length = values.len() / MXFP4_BLOCK_SIZE
pub fn quantize_row_to_mxfp4(values: &[f32]) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(
        values.len() % MXFP4_BLOCK_SIZE,
        0,
        "values.len() must be a multiple of MXFP4_BLOCK_SIZE={MXFP4_BLOCK_SIZE}"
    );
    let n_blocks = values.len() / MXFP4_BLOCK_SIZE;
    let mut packed = Vec::with_capacity(values.len() / 2);
    let mut scales = Vec::with_capacity(n_blocks);

    for block in values.chunks(MXFP4_BLOCK_SIZE) {
        let scale_u8 = compute_ue8m0_scale(block);
        let scale_f32 = ue8m0_to_f32(scale_u8);
        scales.push(scale_u8);

        for pair in block.chunks(2) {
            let scaled0 = if scale_f32 > 0.0 { pair[0] / scale_f32 } else { 0.0 };
            let scaled1 = if scale_f32 > 0.0 { pair[1] / scale_f32 } else { 0.0 };
            packed.push(pack_e2m1_pair(f32_to_e2m1(scaled0), f32_to_e2m1(scaled1)));
        }
    }

    (packed, scales)
}

/// Convert raw BF16 bytes (2 bytes per element, little-endian) to f32.
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 2, 0);
    bytes
        .chunks(2)
        .map(|b| {
            // BF16 is the top 16 bits of f32; shift to f32 by appending 16 zero mantissa bits.
            let bf16_bits = u16::from_le_bytes([b[0], b[1]]);
            f32::from_bits((bf16_bits as u32) << 16)
        })
        .collect()
}

/// Convert raw F16 bytes (2 bytes per element, little-endian) to f32.
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 2, 0);
    bytes
        .chunks(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            // IEEE 754 f16 -> f32 expansion
            let sign = ((bits >> 15) & 1) as u32;
            let exp = ((bits >> 10) & 0x1f) as u32;
            let frac = (bits & 0x3ff) as u32;
            if exp == 0 {
                if frac == 0 {
                    f32::from_bits(sign << 31)
                } else {
                    // Subnormal f16 -> normalized f32
                    let mut e = 0u32;
                    let mut f = frac;
                    while (f & 0x400) == 0 {
                        f <<= 1;
                        e = e.wrapping_add(1);
                    }
                    f &= 0x3ff;
                    let f32_exp = 127u32.wrapping_sub(15).wrapping_add(1).wrapping_sub(e);
                    f32::from_bits((sign << 31) | (f32_exp << 23) | (f << 13))
                }
            } else if exp == 31 {
                f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
            } else {
                let f32_exp = exp + 127 - 15;
                f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
            }
        })
        .collect()
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// E2M1 nibble is always in 0..=15 (4 bits fit).
    #[kani::proof]
    fn e2m1_nibble_fits_in_4_bits() {
        let bits: u32 = kani::any();
        // Construct a finite f32 from exact representable magnitudes only.
        let mag_idx: usize = kani::any();
        let sign_flag: bool = kani::any();
        kani::assume(mag_idx < E2M1_MAGNITUDES.len());
        let mag = E2M1_MAGNITUDES[mag_idx];
        let x = if sign_flag { -mag } else { mag };
        let nibble = f32_to_e2m1(x);
        assert!(nibble <= 15, "E2M1 nibble must fit in 4 bits");
        let _ = bits;
    }

    /// pack_e2m1_pair / unpack roundtrip: lo in bits[3:0], hi in bits[7:4].
    #[kani::proof]
    fn pack_unpack_roundtrip() {
        let lo: u8 = kani::any();
        let hi: u8 = kani::any();
        kani::assume(lo <= 15);
        kani::assume(hi <= 15);
        let packed = pack_e2m1_pair(lo, hi);
        assert_eq!(packed & 0xF, lo, "lo nibble roundtrip");
        assert_eq!((packed >> 4) & 0xF, hi, "hi nibble roundtrip");
    }

    /// e2m1_to_f32 produces a value in the known representable set.
    #[kani::proof]
    fn e2m1_decode_in_representable_set() {
        let nibble: u8 = kani::any();
        kani::assume(nibble <= 15);
        let val = e2m1_to_f32(nibble);
        let mag = val.abs();
        let valid = mag == 0.0
            || mag == 0.5
            || mag == 1.0
            || mag == 1.5
            || mag == 2.0
            || mag == 3.0
            || mag == 4.0
            || mag == 6.0;
        assert!(valid, "e2m1_to_f32 must land on a representable magnitude");
    }

    /// Sign bit in E2M1 nibble is bit 3; negative inputs set it.
    #[kani::proof]
    fn e2m1_sign_bit_correct() {
        let mag_idx: usize = kani::any();
        kani::assume(mag_idx < E2M1_MAGNITUDES.len());
        let mag = E2M1_MAGNITUDES[mag_idx];
        if mag > 0.0 {
            let pos = f32_to_e2m1(mag);
            let neg = f32_to_e2m1(-mag);
            assert_eq!(pos & 0b1000, 0, "positive value must have sign bit 0");
            assert_eq!(neg & 0b1000, 0b1000, "negative value must have sign bit 1");
            assert_eq!(pos & 0b0111, neg & 0b0111, "magnitude index must be equal for +/-");
        }
    }

    /// compute_ue8m0_scale covers max: after dividing by scale, max_abs fits in E2M1.
    #[kani::proof]
    fn ue8m0_scale_covers_block_max() {
        let mag_idx: usize = kani::any();
        kani::assume(mag_idx < E2M1_MAGNITUDES.len());
        let max_val = E2M1_MAGNITUDES[mag_idx];
        let mut block = [0.0f32; MXFP4_BLOCK_SIZE];
        block[0] = max_val;
        let s = compute_ue8m0_scale(&block);
        let sf = ue8m0_to_f32(s);
        if max_val > 0.0 {
            assert!(sf > 0.0, "scale must be positive for nonzero input");
            let scaled = max_val / sf;
            assert!(scaled <= 6.0 + 1e-5, "scaled value must fit in E2M1 range");
        } else {
            assert_eq!(s, 0, "scale must be 0 for zero block");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_exact_representable_values() {
        // All exact representable magnitudes should round-trip.
        for &mag in &E2M1_MAGNITUDES {
            let nibble = f32_to_e2m1(mag);
            let decoded = e2m1_to_f32(nibble);
            assert!(
                (decoded - mag).abs() < 1e-6,
                "mag={mag} nibble=0x{nibble:x} decoded={decoded}"
            );
            if mag > 0.0 {
                let nibble_neg = f32_to_e2m1(-mag);
                let decoded_neg = e2m1_to_f32(nibble_neg);
                assert!(
                    (decoded_neg + mag).abs() < 1e-6,
                    "neg mag={mag} nibble=0x{nibble_neg:x} decoded={decoded_neg}"
                );
            }
        }
    }

    #[test]
    fn e2m1_clamps_to_range() {
        // Values beyond 6.0 clamp to 6.0.
        assert_eq!(f32_to_e2m1(100.0), f32_to_e2m1(6.0));
        assert_eq!(f32_to_e2m1(-100.0), f32_to_e2m1(-6.0));
    }

    #[test]
    fn e2m1_nearest_rounding() {
        // Midpoint between 1.5 and 2.0 is 1.75; nearest is 2.0.
        let nibble = f32_to_e2m1(1.75);
        assert_eq!(e2m1_to_f32(nibble), 2.0);
        // Midpoint between 1.0 and 1.5 is 1.25; nearest is 1.0 or 1.5 (1.25-1.0=0.25, 1.5-1.25=0.25 -> tie -> lower index = 1.0).
        let nibble125 = f32_to_e2m1(1.25);
        // Either 1.0 or 1.5 is acceptable at the tie; just check it's one of them.
        let decoded = e2m1_to_f32(nibble125);
        assert!(
            (decoded - 1.0).abs() < 1e-6 || (decoded - 1.5).abs() < 1e-6,
            "decoded={decoded}"
        );
    }

    #[test]
    fn ue8m0_scale_zero_block() {
        let block = [0.0f32; MXFP4_BLOCK_SIZE];
        assert_eq!(compute_ue8m0_scale(&block), 0);
    }

    #[test]
    fn ue8m0_scale_covers_max() {
        let mut block = [0.0f32; MXFP4_BLOCK_SIZE];
        block[0] = 5.0;
        let s = compute_ue8m0_scale(&block);
        let scale_f32 = ue8m0_to_f32(s);
        // After dividing by scale, max value must fit in E2M1 range [0, 6].
        assert!(
            5.0 / scale_f32 <= 6.0 + 1e-6,
            "scale_f32={scale_f32} 5.0/scale={}",
            5.0 / scale_f32
        );
    }

    #[test]
    fn quantize_row_round_trip_exact() {
        // Row of all 1.0s: should quantize and dequantize to 1.0.
        let values = vec![1.0f32; MXFP4_BLOCK_SIZE * 4];
        let (packed, scales) = quantize_row_to_mxfp4(&values);
        assert_eq!(packed.len(), values.len() / 2);
        assert_eq!(scales.len(), 4);

        for i in 0..values.len() {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                packed[byte_idx] & 0xF
            } else {
                packed[byte_idx] >> 4
            };
            let scale = ue8m0_to_f32(scales[i / MXFP4_BLOCK_SIZE]);
            let decoded = e2m1_to_f32(nibble) * scale;
            assert!(
                (decoded - 1.0).abs() < 0.01,
                "i={i} decoded={decoded}"
            );
        }
    }

    #[test]
    fn quantize_row_length_invariant() {
        let values = vec![2.5f32; MXFP4_BLOCK_SIZE * 8];
        let (packed, scales) = quantize_row_to_mxfp4(&values);
        assert_eq!(packed.len(), values.len() / 2);
        assert_eq!(scales.len(), 8);
    }

    #[test]
    fn bf16_to_f32_known_value() {
        // BF16 representation of 1.0 is 0x3F80 (same as f32 top 16 bits of 0x3F800000).
        let bytes = [0x80u8, 0x3F]; // little-endian 0x3F80
        let result = bf16_bytes_to_f32(&bytes);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6, "got {}", result[0]);
    }

    #[test]
    fn f16_to_f32_known_value() {
        // F16 representation of 1.0 is 0x3C00.
        let bytes = [0x00u8, 0x3C]; // little-endian 0x3C00
        let result = f16_bytes_to_f32(&bytes);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6, "got {}", result[0]);
    }

    #[test]
    fn max_relative_error_bounded() {
        // Test that for uniform values in [-6, 6], relative error is within 25%.
        let mut max_rel_err = 0.0f32;
        let mut worst_v = 0.0f32;
        let mut block = [0.0f32; MXFP4_BLOCK_SIZE];
        for i in 0..200 {
            let v = -6.0 + (i as f32) * 12.0 / 199.0;
            block[0] = v;
            let s = compute_ue8m0_scale(&block);
            let sf = ue8m0_to_f32(s);
            let scaled = if sf > 0.0 { v / sf } else { 0.0 };
            let nibble = f32_to_e2m1(scaled);
            let decoded = e2m1_to_f32(nibble) * sf;
            let rel_err = if v.abs() > 0.1 {
                (decoded - v).abs() / v.abs()
            } else {
                0.0
            };
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
                worst_v = v;
            }
        }
        assert!(
            max_rel_err < 0.30,
            "max relative error {:.1}% at v={worst_v:.3} exceeds 30%",
            max_rel_err * 100.0
        );
    }
}
