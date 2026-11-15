// FP8 e4m3 (OCP E4M3FN) bit conversions — ONE shared implementation.
//
// Used by three compilation contexts: the host (engine DRAM-mirror syncs
// and checksums in `cutile_kv_cache.rs`), the cutile cast/prefill
// kernels' documented semantics, and the oxide decode-attention twins
// (`tools/oxide-decode-attn` `include!`s this file and compiles it to
// PTX — keep it `#[inline(always)]`, pure integer/bit math, no crate
// dependencies, no `std`).
//
// Layout: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
// E4M3FN has NO infinity; 0x7F/0xFF are NaN; max finite is +-448
// (S=0x7E); subnormals are m/8 * 2^-6 down to 2^-9.
//
// PRECISION DECISION (both directions, symmetric so round-trips and
// cross-path checksums stay EXACT): e4m3 subnormals are FLUSHED to
// signed zero. Their absolute value is < 2^-6, negligible in attention
// dot products; the ppl quality gate arbitrates. Consequence for the
// writer: |x| below the RNE midpoint of the smallest normal (2^-6) also
// quantizes to signed zero.

/// e4m3 bits -> f32, EXACT for every normal encoding (subnormals flush
/// to signed zero). exp==15 (the NaN encodings 0x7F/0xFF) maps through
/// the normal path to a large finite f32 -- unreachable from our
/// saturating writer ([`f32_to_e4m3_bits`] never emits it); documented
/// rather than special-cased so device and host readers agree by
/// construction.
#[inline(always)]
pub fn e4m3_bits_to_f32(b: u8) -> f32 {
    let b = b as u32;
    let sign = (b & 0x80) << 24;
    let exp = (b >> 3) & 0x0F;
    let man = b & 0x07;
    if exp == 0 {
        // Subnormal flush (see module docs): read as signed zero.
        f32::from_bits(sign)
    } else {
        // f32 exponent = e4m3 exponent + (127 - 7) = e + 120;
        // the 3 mantissa bits land in the f32 mantissa's top 3 bits.
        f32::from_bits(sign | ((exp + 120) << 23) | (man << 20))
    }
}

/// f32 -> e4m3 bits: round-to-nearest-EVEN on the 20 dropped mantissa
/// bits, saturation to +-448 (E4M3FN has no infinity), subnormal flush
/// (see module docs), NaN -> 0x7F.
///
/// Exact inverse of [`e4m3_bits_to_f32`] on every normal e4m3 value:
/// representable inputs have zero dropped bits, so no rounding occurs.
#[inline(always)]
pub fn f32_to_e4m3_bits(x: f32) -> u8 {
    let bits = x.to_bits();
    let sign = (bits >> 24) & 0x80;
    let abs = bits & 0x7FFF_FFFF;
    // NaN (abs > inf bits) -> the e4m3fn NaN encoding.
    if abs > 0x7F80_0000 {
        return 0x7F;
    }
    // Saturate to +-448 (max finite; E4M3FN has no inf). Values >= 448
    // clamp to 448 exactly -- matching cvt.rn.satfinite semantics.
    let x = if x > 448.0 {
        448.0
    } else if x < -448.0 {
        -448.0
    } else {
        x
    };
    let bits = x.to_bits();
    let abs = bits & 0x7FFF_FFFF;
    let exp = (abs >> 23) & 0xFF;
    // f32 exponent 121 == 2^-6 == the smallest e4m3 normal. Below that
    // (incl. f32 subnormals and +-0) -> signed zero (subnormal flush).
    if exp < 121 {
        return sign as u8;
    }
    let man = abs & 0x7F_FFFF;
    let e2 = exp - 120; // target e4m3 exponent field, 1..=15
    let kept = man >> 20; // top 3 mantissa bits
    let dropped = man & 0xF_FFFF;
    // RNE: round up when dropped > half, or == half and the kept LSB is 1.
    let round_up = dropped > 0x8_0000 || (dropped == 0x8_0000 && (kept & 1) == 1);
    let (e2, kept) = if round_up {
        if kept == 7 {
            (e2 + 1, 0)
        } else {
            (e2, kept + 1)
        }
    } else {
        (e2, kept)
    };
    // The carry can only push e2 to 16 from (15, kept=7) -- i.e. values
    // >= 464 -- which the 448 clamp made unreachable (448 is exactly
    // e2=15, kept=6, dropped=0; nothing in [432, 448] rounds past it).
    (sign | (e2 << 3) | kept) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every normal e4m3 value round-trips exactly through
    /// f32 -> bits -> f32.
    #[test]
    fn roundtrip_exact_all_normals() {
        for b in 0u16..=0xFF {
            let b = b as u8;
            let exp = (b >> 3) & 0x0F;
            let v = e4m3_bits_to_f32(b);
            if exp == 0 {
                // Subnormal encodings read as signed zero; the writer
                // flushes anything below 2^-6 to signed zero too.
                assert_eq!(v, if b & 0x80 != 0 { -0.0 } else { 0.0 });
                continue;
            }
            if exp == 15 && (b & 0x07) == 7 {
                continue; // NaN encodings -- writer never emits
            }
            let back = f32_to_e4m3_bits(v);
            assert_eq!(back, b, "roundtrip mismatch for bits {b:#04x} (v={v})");
        }
    }

    #[test]
    fn known_values() {
        assert_eq!(f32_to_e4m3_bits(0.0), 0x00);
        assert_eq!(f32_to_e4m3_bits(-0.0), 0x80);
        assert_eq!(e4m3_bits_to_f32(0x00), 0.0);
        assert_eq!(e4m3_bits_to_f32(0x38), 1.0); // 1.0 = 0x38
        assert_eq!(f32_to_e4m3_bits(1.0), 0x38);
        assert_eq!(f32_to_e4m3_bits(-1.0), 0xB8);
        assert_eq!(e4m3_bits_to_f32(0x7E), 448.0); // max finite
        assert_eq!(f32_to_e4m3_bits(448.0), 0x7E);
        assert_eq!(f32_to_e4m3_bits(-448.0), 0xFE);
        // Saturation (no inf in E4M3FN).
        assert_eq!(f32_to_e4m3_bits(1e30), 0x7E);
        assert_eq!(f32_to_e4m3_bits(f32::INFINITY), 0x7E);
        assert_eq!(f32_to_e4m3_bits(-f32::INFINITY), 0xFE);
        assert_eq!(f32_to_e4m3_bits(f32::NAN), 0x7F);
        // Smallest normal: 2^-6 = 0.015625.
        assert_eq!(f32_to_e4m3_bits(0.015625), 0x08);
        assert_eq!(e4m3_bits_to_f32(0x08), 0.015625);
        // Subnormal flush: anything < 2^-6 -> signed zero.
        assert_eq!(f32_to_e4m3_bits(0.01), 0x00);
        assert_eq!(f32_to_e4m3_bits(0.001), 0x00);
        assert_eq!(f32_to_e4m3_bits(-0.001), 0x80);
        // e4m3 subnormal encodings read as signed zero.
        assert_eq!(e4m3_bits_to_f32(0x01), 0.0);
        assert_eq!(e4m3_bits_to_f32(0x81), -0.0);
    }

    /// RNE behavior at the tie point of the 1.0 binade (spacing 0.125):
    /// 1.0625 is exactly between 1.0 and 1.125 -- rounds to EVEN (1.0).
    #[test]
    fn rne_tie_to_even() {
        assert_eq!(f32_to_e4m3_bits(1.0), 0x38);
        assert_eq!(f32_to_e4m3_bits(1.0625), 0x38); // tie -> even (kept 0)
        assert_eq!(f32_to_e4m3_bits(1.0626), 0x39); // just past tie -> up
        assert_eq!(f32_to_e4m3_bits(1.1875), 0x3A); // tie -> even (kept 2)
        assert_eq!(f32_to_e4m3_bits(1.125), 0x39);
    }

    /// Mantissa carry into the exponent: 15.5 ties between 15.0 (kept 7)
    /// and 16.0 (next exponent); tie-with-odd-LSB rounds up, carrying
    /// kept 7 -> exp+1.
    #[test]
    fn rne_carry_into_exponent() {
        assert_eq!(e4m3_bits_to_f32(0x53), 11.0);
        // 15.0 = (1+7/8)*8 = exp 10, kept 7; the next value up is 16.0.
        assert_eq!(f32_to_e4m3_bits(15.0), 0x57);
        assert_eq!(f32_to_e4m3_bits(15.4375), 0x57); // below the tie
        assert_eq!(f32_to_e4m3_bits(15.5), 0x58); // tie, kept 7 odd -> carry
        assert_eq!(f32_to_e4m3_bits(15.9375), 0x58);
        assert_eq!(f32_to_e4m3_bits(16.0), 0x58);
        assert_eq!(e4m3_bits_to_f32(0x58), 16.0);
    }
}
