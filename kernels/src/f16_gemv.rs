// -- F16 GEMV (lm_head) --------------------------------------------------------

/// F16xF16 dot product matching llama.cpp's ggml_vec_dot_f16 AVX-512 path exactly.
/// Both inputs are raw F16 bytes. Uses 4 x __m512 accumulators, 64 elements/iter,
/// tree reduction, _mm512_reduce_add_ps -- same structure as GGML_F32Cx16_REDUCE.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,f16c,fma")]
pub unsafe fn f16_dot_ggml(x_f16: &[u8], y_f16: &[u8], k: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = [_mm512_setzero_ps(); 4];
    let x_ptr = x_f16.as_ptr() as *const __m256i;
    let y_ptr = y_f16.as_ptr() as *const __m256i;

    let k64 = k & !63;
    let mut i = 0usize;
    while i < k64 {
        for j in 0..4usize {
            let ax = _mm512_cvtph_ps(_mm256_loadu_si256(x_ptr.add((i + j * 16) / 16)));
            let ay = _mm512_cvtph_ps(_mm256_loadu_si256(y_ptr.add((i + j * 16) / 16)));
            sum[j] = _mm512_fmadd_ps(ax, ay, sum[j]);
        }
        i += 64;
    }

    sum[0] = _mm512_add_ps(sum[0], sum[2]);
    sum[1] = _mm512_add_ps(sum[1], sum[3]);
    sum[0] = _mm512_add_ps(sum[0], sum[1]);
    let mut result = _mm512_reduce_add_ps(sum[0]);

    while i < k {
        let xv =
            half::f16::from_bits(u16::from_le_bytes([x_f16[i * 2], x_f16[i * 2 + 1]])).to_f32();
        let yv =
            half::f16::from_bits(u16::from_le_bytes([y_f16[i * 2], y_f16[i * 2 + 1]])).to_f32();
        result += xv * yv;
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn f16_dot_avx2(a: &[f32], b_f16: &[u8], k: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b_f16.as_ptr() as *const __m128i;

    let k32 = k & !31;
    let mut i = 0usize;
    while i < k32 {
        let b0 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(i / 8)));
        let a0 = _mm256_loadu_ps(a_ptr.add(i));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);

        let b1 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(i / 8 + 1)));
        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);

        let b2 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(i / 8 + 2)));
        let a2 = _mm256_loadu_ps(a_ptr.add(i + 16));
        acc2 = _mm256_fmadd_ps(a2, b2, acc2);

        let b3 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(i / 8 + 3)));
        let a3 = _mm256_loadu_ps(a_ptr.add(i + 24));
        acc3 = _mm256_fmadd_ps(a3, b3, acc3);

        i += 32;
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    while i + 8 <= k {
        let b0 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(i / 8)));
        let a0 = _mm256_loadu_ps(a_ptr.add(i));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        i += 8;
    }

    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    let mut result = _mm_cvtss_f32(sums2);

    while i < k {
        let b_val =
            half::f16::from_bits(u16::from_le_bytes([b_f16[i * 2], b_f16[i * 2 + 1]])).to_f32();
        result += a[i] * b_val;
        i += 1;
    }

    result
}

/// F16xF16 scalar ref matching ggml_vec_dot_f16 AVX-512 accumulation order.
pub fn f16_dot_ggml_scalar(x_f16: &[u8], y_f16: &[u8], k: usize) -> f32 {
    let mut acc = [0.0f32; 64];
    let k64 = k & !63;
    let mut i = 0usize;

    while i < k64 {
        for j in 0..4 {
            for lane in 0..16 {
                let idx = i + j * 16 + lane;
                let xv =
                    half::f16::from_bits(u16::from_le_bytes([x_f16[idx * 2], x_f16[idx * 2 + 1]]))
                        .to_f32();
                let yv =
                    half::f16::from_bits(u16::from_le_bytes([y_f16[idx * 2], y_f16[idx * 2 + 1]]))
                        .to_f32();
                acc[j * 16 + lane] = xv.mul_add(yv, acc[j * 16 + lane]);
            }
        }
        i += 64;
    }

    for lane in 0..16 {
        acc[lane] += acc[32 + lane];
        acc[16 + lane] += acc[48 + lane];
    }
    for lane in 0..16 {
        acc[lane] += acc[16 + lane];
    }

    let mut h = [0.0f32; 8];
    for i in 0..8 {
        h[i] = acc[i] + acc[i + 8];
    }
    let mut q = [0.0f32; 4];
    for i in 0..4 {
        q[i] = h[i] + h[i + 4];
    }
    let p0 = q[1] + q[0];
    let p1 = q[3] + q[2];
    let mut s = p0 + p1;

    while i < k {
        let xv =
            half::f16::from_bits(u16::from_le_bytes([x_f16[i * 2], x_f16[i * 2 + 1]])).to_f32();
        let yv =
            half::f16::from_bits(u16::from_le_bytes([y_f16[i * 2], y_f16[i * 2 + 1]])).to_f32();
        s += xv * yv;
        i += 1;
    }

    s
}

/// AVX512 F32xF16 dot: 16 floats/iter with _mm512_cvtph_ps -> _mm512_fmadd_ps.
/// ~2x throughput vs AVX2 on 512-bit-wide hardware (9950X3D).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,f16c,fma")]
pub unsafe fn f16_dot_avx512(a: &[f32], b_f16: &[u8], k: usize) -> f32 {
    use std::arch::x86_64::*;
    let a_ptr = a.as_ptr();
    let b_ptr = b_f16.as_ptr() as *const __m256i;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let k32 = k & !31;
    let mut i = 0usize;
    while i < k32 {
        let b0 = _mm512_cvtph_ps(_mm256_loadu_si256(b_ptr.add(i / 16)));
        let b1 = _mm512_cvtph_ps(_mm256_loadu_si256(b_ptr.add(i / 16 + 1)));
        let a0 = _mm512_loadu_ps(a_ptr.add(i));
        let a1 = _mm512_loadu_ps(a_ptr.add(i + 16));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        i += 32;
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    // 16-element tail with AVX2
    if i + 16 <= k {
        let b0 = _mm512_cvtph_ps(_mm256_loadu_si256(b_ptr.add(i / 16)));
        let a0 = _mm512_loadu_ps(a_ptr.add(i));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        i += 16;
    }
    let mut result = _mm512_reduce_add_ps(acc0);
    while i < k {
        let bv = half::f16::from_bits(u16::from_le_bytes([b_f16[i*2], b_f16[i*2+1]])).to_f32();
        result += a[i] * bv;
        i += 1;
    }
    result
}

/// Runtime-dispatch F32xF16 dot: AVX512 -> AVX2 -> scalar.
#[inline]
pub fn f16_dot_dispatch(a: &[f32], b_f16: &[u8], k: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
        return unsafe { f16_dot_avx512(a, b_f16, k) };
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
        return unsafe { f16_dot_avx2(a, b_f16, k) };
    }
    f16_dot_scalar(a, b_f16, k)
}

pub fn f16_dot_scalar(a: &[f32], b_f16: &[u8], k: usize) -> f32 {
    let mut acc = [0.0f32; 32];
    let k32 = k & !31;
    let mut i = 0usize;

    while i < k32 {
        for lane in 0..8 {
            let b_val = half::f16::from_bits(u16::from_le_bytes([
                b_f16[(i + lane) * 2],
                b_f16[(i + lane) * 2 + 1],
            ]))
            .to_f32();
            acc[lane] = a[i + lane].mul_add(b_val, acc[lane]);
        }
        for lane in 0..8 {
            let b_val = half::f16::from_bits(u16::from_le_bytes([
                b_f16[(i + 8 + lane) * 2],
                b_f16[(i + 8 + lane) * 2 + 1],
            ]))
            .to_f32();
            acc[8 + lane] = a[i + 8 + lane].mul_add(b_val, acc[8 + lane]);
        }
        for lane in 0..8 {
            let b_val = half::f16::from_bits(u16::from_le_bytes([
                b_f16[(i + 16 + lane) * 2],
                b_f16[(i + 16 + lane) * 2 + 1],
            ]))
            .to_f32();
            acc[16 + lane] = a[i + 16 + lane].mul_add(b_val, acc[16 + lane]);
        }
        for lane in 0..8 {
            let b_val = half::f16::from_bits(u16::from_le_bytes([
                b_f16[(i + 24 + lane) * 2],
                b_f16[(i + 24 + lane) * 2 + 1],
            ]))
            .to_f32();
            acc[24 + lane] = a[i + 24 + lane].mul_add(b_val, acc[24 + lane]);
        }
        i += 32;
    }

    for lane in 0..8 {
        acc[lane] += acc[8 + lane];
        acc[16 + lane] += acc[24 + lane];
        acc[lane] += acc[16 + lane];
    }

    while i + 8 <= k {
        for lane in 0..8 {
            let b_val = half::f16::from_bits(u16::from_le_bytes([
                b_f16[(i + lane) * 2],
                b_f16[(i + lane) * 2 + 1],
            ]))
            .to_f32();
            acc[lane] = a[i + lane].mul_add(b_val, acc[lane]);
        }
        i += 8;
    }

    let mut s = [0.0f32; 4];
    for j in 0..4 {
        s[j] = acc[j] + acc[j + 4];
    }
    let p0 = s[0] + s[1];
    let p1 = s[2] + s[3];
    let mut result = p0 + p1;

    while i < k {
        let b_val =
            half::f16::from_bits(u16::from_le_bytes([b_f16[i * 2], b_f16[i * 2 + 1]])).to_f32();
        result += a[i] * b_val;
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn f16_gemv_avx2(
    a_ptr: usize,
    b_ptr: usize,
    out_ptr: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    let row_bytes = k * 2;
    let a_slice = std::slice::from_raw_parts(a_ptr as *const f32, k);
    for j in row_start..row_end {
        let b_row = std::slice::from_raw_parts((b_ptr as *const u8).add(j * row_bytes), row_bytes);
        let val = f16_dot_avx2(a_slice, b_row, k);
        *(out_ptr as *mut f32).add(j) = val;
    }
}

pub fn f16_gemv(a: &[f32], b_f16: &[u8], output: &mut [f32], n: usize, k: usize) {
    assert_eq!(a.len(), k);
    assert_eq!(b_f16.len(), n * k * 2);
    assert_eq!(output.len(), n);

    let a_ptr = a.as_ptr() as usize;
    let b_ptr = b_f16.as_ptr() as usize;
    let out_ptr = output.as_mut_ptr() as usize;

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("f16c")
        && is_x86_feature_detected!("fma");

    crate::thread_pool::global_pool().parallel_chunks(n, 1, &|row_start, row_end| {
        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            unsafe { f16_gemv_avx2(a_ptr, b_ptr, out_ptr, k, row_start, row_end) };
            return;
        }
        let row_bytes = k * 2;
        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, k) };
        for j in row_start..row_end {
            let b_row = unsafe {
                std::slice::from_raw_parts((b_ptr as *const u8).add(j * row_bytes), row_bytes)
            };
            unsafe {
                *(out_ptr as *mut f32).add(j) = f16_dot_scalar(a_slice, b_row, k);
            }
        }
    });
}
