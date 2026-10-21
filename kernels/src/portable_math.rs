/// Portable exp(x) producing bit-identical results on CPU and GPU.
/// Cody-Waite reduction + degree-6 minimax polynomial.
/// No FMA dependency: all operations are explicit add/multiply.
#[inline(always)]
pub fn portable_exp(x: f32) -> f32 {
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const P1: f32 = 1.0;
    const P2: f32 = 0.5;
    const P3: f32 = 0.166_666_67;
    const P4: f32 = 0.041666653;
    const P5: f32 = 0.008_333_36;
    const P6: f32 = 0.001388888;

    if x > 88.7 {
        return f32::INFINITY;
    }
    if x < -87.3 {
        return 0.0;
    }

    let n = (x * LOG2E + 0.5).floor();
    let ni = n as i32;
    let r = (x - n * LN2_HI) - n * LN2_LO;

    let mut p = P6;
    p = p * r + P5;
    p = p * r + P4;
    p = p * r + P3;
    p = p * r + P2;
    p = p * r + P1;
    p = p * r + 1.0;

    let scale = f32::from_bits(((ni + 127) as u32) << 23);
    p * scale
}

/// GGML-compatible exp approximation.
/// Translated from ggml.c:2746 (ggml_v_expf AVX2 path), scalar equivalent.
/// Uses the same constants and polynomial as the SIMD version.
#[inline(always)]
pub fn ggml_exp(x: f32) -> f32 {
    // C: const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const R: f32 = 12582912.0; // 0x1.8p23 = 1.5 * 2^23

    // C: const __m256 z = _mm256_fmadd_ps(x, 0x1.715476p+0f, r);
    // 0x1.715476p+0 = log2(e) ~= 1.4426950216293335
    const LOG2E: f32 = f32::from_bits(0x3fb8aa3b); // 0x1.715476p+0
    let z: f32 = x.mul_add(LOG2E, R);

    let n: f32 = z - R;

    const LN2_HI: f32 = f32::from_bits(0x3f317200); // 0x1.62e4p-1
    const LN2_LO: f32 = f32::from_bits(0x35bfbe8e); // 0x1.7f7d1cp-20
    let b: f32 = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, x));

    // C: e = slli_epi32(castps_si256(z), 23)
    let z_bits = z.to_bits();
    let e: u32 = z_bits << 23;

    // C: k = castsi256_ps(add_epi32(e, castps_si256(set1_ps(1))))
    let k: f32 = f32::from_bits(e.wrapping_add(1.0f32.to_bits()));

    // C: c = cmp(andnot(-0.f, n), 126, _CMP_GT_OQ)  -- overflow check
    let n_abs = n.abs();
    let overflow = n_abs > 126.0;

    // C: u = mul(b, b)
    let u: f32 = b * b;

    // Polynomial evaluation:
    // C: j = fmadd(fmadd(fmadd(0x1.0e4020p-7, b, 0x1.573e2ep-5), u,
    //              fmadd(0x1.555e66p-3, b, 0x1.fffdb6p-2)), u,
    //              mul(0x1.ffffecp-1, b))
    const P0: f32 = f32::from_bits(0x3f7ffff6); // 0x1.ffffecp-1
    const P1: f32 = f32::from_bits(0x3efffedb); // 0x1.fffdb6p-2
    const P2: f32 = f32::from_bits(0x3e2aaf33); // 0x1.555e66p-3
    const P3: f32 = f32::from_bits(0x3d2b9f17); // 0x1.573e2ep-5
    const P4: f32 = f32::from_bits(0x3c072010); // 0x1.0e4020p-7

    let j: f32 = P4
        .mul_add(b, P3)
        .mul_add(u, P2.mul_add(b, P1))
        .mul_add(u, P0 * b);

    if !overflow {
        // C: return fmadd(j, k, k)
        return j.mul_add(k, k);
    }

    // Slow path: handle large |n|
    // C: g = and(cmp_le(n, 0), 0x82000000)
    let g: u32 = if n <= 0.0 { 0x82000000u32 } else { 0 };

    // C: s1 = castsi256_ps(add(g, 0x7f000000))
    let s1 = f32::from_bits(g.wrapping_add(0x7f000000));

    // C: s2 = castsi256_ps(sub(e, g))
    let s2 = f32::from_bits(e.wrapping_sub(g));

    // C: d = cmp(abs(n), 192, _CMP_GT_OQ) -- extreme overflow
    let extreme = n_abs > 192.0;

    if extreme {
        // C: return s1 * s1 (flush to 0 or inf)
        return s1 * s1;
    }

    // C: return fmadd(s2, j, s2) * s1
    s2.mul_add(j, s2) * s1
}

/// Portable SiLU: x / (1 + exp(-x))
#[inline(always)]
pub fn portable_silu(x: f32) -> f32 {
    x / (1.0 + portable_exp(-x))
}

/// Q8 round-trip: quantize to int8 then dequantize back to f32.
/// Snaps each 32-element block to a 255-level grid, absorbing sub-ULP differences.
pub fn q8_snap(x: &mut [f32]) {
    const BLOCK: usize = 32;
    let n_blocks = x.len().div_ceil(BLOCK);
    for blk in 0..n_blocks {
        let start = blk * BLOCK;
        let end = (start + BLOCK).min(x.len());

        let mut amax = 0.0f32;
        for i in start..end {
            let a = x[i].abs();
            if a > amax {
                amax = a;
            }
        }

        let scale = amax / 127.0;
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };

        for i in start..end {
            let q = (x[i] * inv).round().clamp(-128.0, 127.0) as i8;
            x[i] = q as f32 * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portable_exp_accuracy() {
        let test_vals: [f32; 11] = [
            0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 10.0, -10.0, 80.0, -80.0,
        ];
        for &x in &test_vals {
            let reference = x.exp();
            let portable = portable_exp(x);
            let rel_err = if reference != 0.0 {
                ((portable - reference) / reference).abs()
            } else {
                (portable - reference).abs()
            };
            assert!(
                rel_err < 1e-6,
                "exp({x}): portable={portable}, ref={reference}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn test_ggml_exp_accuracy() {
        let test_vals: [f32; 13] = [
            0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 10.0, -10.0, 50.0, -50.0, 80.0, -80.0,
        ];
        for &x in &test_vals {
            let reference = x.exp();
            let ggml = ggml_exp(x);
            let rel_err = if reference > 1e-30 {
                ((ggml - reference) / reference).abs()
            } else {
                (ggml - reference).abs()
            };
            assert!(
                rel_err < 1e-3,
                "ggml_exp({x}): got={ggml}, ref={reference}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn test_ggml_exp_edge_cases() {
        assert!(ggml_exp(200.0).is_infinite());
        assert!(ggml_exp(-200.0) < 1e-30);
        let one = ggml_exp(0.0);
        assert!((one - 1.0).abs() < 1e-6, "ggml_exp(0) = {one}");
    }

    #[test]
    fn test_portable_exp_edge_cases() {
        assert_eq!(portable_exp(89.0), f32::INFINITY);
        assert_eq!(portable_exp(-88.0), 0.0);
        assert_eq!(portable_exp(0.0), 1.0);
    }
}
