//! Direct Rust translations of llama.cpp ops.cpp functions (batch 2).
//!
//! Sources: ggml/src/ggml-cpu/ops.cpp

use crate::ggml::helpers::f16_to_f32;

// ─── gelu_f16 ────────────────────────────────────────────────────────────────
// ops.cpp lines 2158-2204

/// Translation of `ggml_compute_forward_gelu_f16`.
/// Applies GELU element-wise to f16 rows, parallelised by row.
pub fn gelu_rows_f16(
    src: &[u16],  // f16 bits, [nc] per row
    dst: &mut [u16],
    nc: usize, nr: usize,
    ith: usize, nth: usize,
) {
    let dr = nr.div_ceil(nth);
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        let row_src = &src[ir * nc..(ir + 1) * nc];
        let row_dst = &mut dst[ir * nc..(ir + 1) * nc];
        for (s, d) in row_src.iter().zip(row_dst.iter_mut()) {
            let x = f16_to_f32(*s);
            let g = x * 0.5 * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2));
            *d = half::f16::from_f32(g).to_bits();
        }
    }
}

// ─── gelu_erf_f16 ────────────────────────────────────────────────────────────
// ops.cpp lines 2388-2434

/// Translation of `ggml_compute_forward_gelu_erf_f16`.
pub fn gelu_erf_rows_f16(
    src: &[u16],
    dst: &mut [u16],
    nc: usize, nr: usize,
    ith: usize, nth: usize,
) {
    let dr = nr.div_ceil(nth);
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        let row_src = &src[ir * nc..(ir + 1) * nc];
        let row_dst = &mut dst[ir * nc..(ir + 1) * nc];
        for (s, d) in row_src.iter().zip(row_dst.iter_mut()) {
            let x = f16_to_f32(*s);
            let g = x * 0.5 * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2));
            *d = half::f16::from_f32(g).to_bits();
        }
    }
}

// ─── gelu_quick_f16 ──────────────────────────────────────────────────────────
// ops.cpp lines 2507-2553

/// Translation of `ggml_compute_forward_gelu_quick_f16`.
/// gelu_quick(x) = x * sigmoid(1.702 * x)
pub fn gelu_quick_rows_f16(
    src: &[u16],
    dst: &mut [u16],
    nc: usize, nr: usize,
    ith: usize, nth: usize,
) {
    let dr = nr.div_ceil(nth);
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        let row_src = &src[ir * nc..(ir + 1) * nc];
        let row_dst = &mut dst[ir * nc..(ir + 1) * nc];
        for (s, d) in row_src.iter().zip(row_dst.iter_mut()) {
            let x = f16_to_f32(*s);
            let g = x * (1.0 / (1.0 + (-1.702_f32 * x).exp()));
            *d = half::f16::from_f32(g).to_bits();
        }
    }
}

// ─── leaky_relu_f16 ──────────────────────────────────────────────────────────
// ops.cpp lines 2727-2755

/// Translation of `ggml_compute_forward_leaky_relu_f16`.
/// Thread 0 only (as in original).
pub fn leaky_relu_rows_f16(
    src: &[u16],
    dst: &mut [u16],
    n: usize, nc: usize,
    negative_slope: f32,
    ith: usize,
) {
    if ith != 0 {
        return;
    }
    for i in 0..n {
        let row_src = &src[i * nc..(i + 1) * nc];
        let row_dst = &mut dst[i * nc..(i + 1) * nc];
        for (s, d) in row_src.iter().zip(row_dst.iter_mut()) {
            let x = f16_to_f32(*s);
            let y = if x >= 0.0 { x } else { negative_slope * x };
            *d = half::f16::from_f32(y).to_bits();
        }
    }
}

// ─── geglu_f16 ───────────────────────────────────────────────────────────────
// ops.cpp lines 3094-3152

/// Translation of `ggml_compute_forward_geglu_f16`.
/// gate(x) = x * GELU(gate_x). Optional src1 or interleaved.
pub fn geglu_rows_f16(
    src0: &[u16], src1: Option<&[u16]>,
    dst: &mut [u16],
    nc: usize, nr: usize,
    src0_row_stride: usize, src1_row_stride: usize,
    swapped: bool,
    ith: usize, nth: usize,
) {
    let dr = nr.div_ceil(nth);
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for i1 in ir0..ir1 {
        let (p0_off, p1_off) = if src1.is_some() {
            (i1 * src0_row_stride, i1 * src1_row_stride)
        } else if swapped {
            (i1 * src0_row_stride + nc, i1 * src0_row_stride)
        } else {
            (i1 * src0_row_stride, i1 * src0_row_stride + nc)
        };
        let src1_slice = src1.unwrap_or(src0);
        let dst_row = &mut dst[i1 * nc..(i1 + 1) * nc];
        for i in 0..nc {
            let x = f16_to_f32(src0[p0_off + i]);
            let gate = f16_to_f32(src1_slice[p1_off + i]);
            // GELU of gate, then multiply by x
            let g = gate * 0.5 * (1.0 + libm::erff(gate * std::f32::consts::FRAC_1_SQRT_2));
            dst_row[i] = half::f16::from_f32(x * g).to_bits();
        }
    }
}

// ─── geglu_erf_f16 ───────────────────────────────────────────────────────────
// ops.cpp lines 3600-3657 (identical algorithm to geglu_f16 for f16 path)

/// Translation of `ggml_compute_forward_geglu_erf_f16`.
pub fn geglu_erf_rows_f16(
    src0: &[u16], src1: Option<&[u16]>,
    dst: &mut [u16],
    nc: usize, nr: usize,
    src0_row_stride: usize, src1_row_stride: usize,
    swapped: bool,
    ith: usize, nth: usize,
) {
    // geglu and geglu_erf are identical in the f16 path (both use erf-GELU)
    geglu_rows_f16(src0, src1, dst, nc, nr, src0_row_stride, src1_row_stride, swapped, ith, nth);
}

// ─── geglu_quick_f16 ─────────────────────────────────────────────────────────
// ops.cpp lines 3743-3800

/// Translation of `ggml_compute_forward_geglu_quick_f16`.
pub fn geglu_quick_rows_f16(
    src0: &[u16], src1: Option<&[u16]>,
    dst: &mut [u16],
    nc: usize, nr: usize,
    src0_row_stride: usize, src1_row_stride: usize,
    swapped: bool,
    ith: usize, nth: usize,
) {
    let dr = nr.div_ceil(nth);
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for i1 in ir0..ir1 {
        let (p0_off, p1_off) = if src1.is_some() {
            (i1 * src0_row_stride, i1 * src1_row_stride)
        } else if swapped {
            (i1 * src0_row_stride + nc, i1 * src0_row_stride)
        } else {
            (i1 * src0_row_stride, i1 * src0_row_stride + nc)
        };
        let src1_slice = src1.unwrap_or(src0);
        let dst_row = &mut dst[i1 * nc..(i1 + 1) * nc];
        for i in 0..nc {
            let x = f16_to_f32(src0[p0_off + i]);
            let gate = f16_to_f32(src1_slice[p1_off + i]);
            // gelu_quick(gate)
            let g = gate * (1.0 / (1.0 + (-1.702_f32 * gate).exp()));
            dst_row[i] = half::f16::from_f32(x * g).to_bits();
        }
    }
}

// ─── fill_f16 ────────────────────────────────────────────────────────────────
// ops.cpp lines 2249-2265

/// Translation of `ggml_compute_forward_fill_f16`.
pub fn fill_rows_f16(
    dst: &mut [u16],
    ne0: usize, ne1: usize, ne2: usize,
    c: f32,
    ith: usize, nth: usize,
) {
    let nr = ne1 * ne2;
    let dr = nr.div_ceil(nth);
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    let cv = half::f16::from_f32(c).to_bits();
    for ir in ir0..ir1 {
        let row = &mut dst[ir * ne0..(ir + 1) * ne0];
        row.fill(cv);
    }
}

// ─── add1_f16_f32 / add1_f16_f16 / add1_bf16_f32 / add1_bf16_bf16 ───────────
// ops.cpp lines 825-1080

/// Translation of `ggml_compute_forward_add1_f16_f32`.
/// Adds scalar f32 `v` to each f16 element.
pub fn add1_f16_f32(
    src: &[u16], dst: &mut [u16],
    ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    v: f32,
    ith: usize, nth: usize,
) {
    let nr = ne1 * ne2 * ne3;
    let dr = nr.div_ceil(nth);
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        for i in 0..ne0 {
            let idx = ir * ne0 + i;
            let x = f16_to_f32(src[idx]);
            dst[idx] = half::f16::from_f32(x + v).to_bits();
        }
    }
}

/// Translation of `ggml_compute_forward_add1_f16_f16`.
/// Adds scalar f16 `v` (as f32) to each f16 element.
pub fn add1_f16_f16(
    src: &[u16], dst: &mut [u16],
    ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    v_f16: u16,
    ith: usize, nth: usize,
) {
    let v = f16_to_f32(v_f16);
    add1_f16_f32(src, dst, ne0, ne1, ne2, ne3, v, ith, nth);
}

/// Translation of `ggml_compute_forward_add1_bf16_f32`.
/// Adds scalar f32 `v` to each bf16 element.
pub fn add1_bf16_f32(
    src: &[u16], dst: &mut [u16],
    ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    v: f32,
    ith: usize, nth: usize,
) {
    let nr = ne1 * ne2 * ne3;
    let dr = nr.div_ceil(nth);
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        for i in 0..ne0 {
            let idx = ir * ne0 + i;
            let x = half::bf16::from_bits(src[idx]).to_f32();
            dst[idx] = half::bf16::from_f32(x + v).to_bits();
        }
    }
}

/// Translation of `ggml_compute_forward_add1_bf16_bf16`.
/// Adds scalar bf16 `v` (as f32) to each bf16 element.
pub fn add1_bf16_bf16(
    src: &[u16], dst: &mut [u16],
    ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    v_bf16: u16,
    ith: usize, nth: usize,
) {
    let v = half::bf16::from_bits(v_bf16).to_f32();
    add1_bf16_f32(src, dst, ne0, ne1, ne2, ne3, v, ith, nth);
}

/// Translation of `ggml_compute_forward_add1_f32`.
/// Adds scalar f32 `v` to each f32 element.
pub fn add1_f32(
    src: &[f32], dst: &mut [f32],
    ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    v: f32,
    ith: usize, nth: usize,
) {
    let nr = ne1 * ne2 * ne3;
    let dr = nr.div_ceil(nth);
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        for i in 0..ne0 {
            let idx = ir * ne0 + i;
            dst[idx] = src[idx] + v;
        }
    }
}

// ─── add_id_f32 ──────────────────────────────────────────────────────────────
// ops.cpp lines 704-753

/// Translation of `ggml_compute_forward_add_id_f32`.
/// dst[i0,i1,i2,i3] = src0[i0,i1,i2,i3] + src1[i0, ids[i1,i2], i2, i3]
pub fn add_id_f32(
    src0: &[f32],
    src1: &[f32],
    ids: &[i32],
    dst: &mut [f32],
    ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    ne10: usize, ne11: usize,
    ith: usize, nth: usize,
) {
    let nr = ne1 * ne2 * ne3;
    let dr = nr.div_ceil(nth);
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        let i3 = ir / (ne2 * ne1);
        let i2 = (ir - i3 * ne2 * ne1) / ne1;
        let i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
        // id index into src1 rows
        let i11 = ids[i1 + i2 * ne1] as usize;
        assert!(i11 < ne11);
        let src0_row = &src0[(i3 * ne2 * ne1 + i2 * ne1 + i1) * ne0..(i3 * ne2 * ne1 + i2 * ne1 + i1) * ne0 + ne0];
        let src1_row = &src1[(i3 * ne2 * ne10 + i2 * ne10 + i11) * ne0..(i3 * ne2 * ne10 + i2 * ne10 + i11) * ne0 + ne0];
        let dst_row = &mut dst[(i3 * ne2 * ne1 + i2 * ne1 + i1) * ne0..(i3 * ne2 * ne1 + i2 * ne1 + i1) * ne0 + ne0];
        for i in 0..ne0 {
            dst_row[i] = src0_row[i] + src1_row[i];
        }
        let _ = ne10;
    }
}

// ─── add_rel_pos_f32 ─────────────────────────────────────────────────────────
// ops.cpp lines 10332-10391

/// Translation of `ggml_compute_forward_add_rel_pos_f32`.
/// Adds relative position embeddings to an attention bias tensor.
/// dst must already be initialised (inplace=true) or zeroed (inplace=false).
pub fn add_rel_pos_f32(
    src1: &[f32],
    src2: &[f32],
    dst: &mut [f32],
    ne10: usize, ne11: usize, ne12: usize, ne13: usize,
    ith: usize, nth: usize,
) {
    let np = ne13;
    let dp = np.div_ceil(nth);
    let ip0 = dp * ith;
    let ip1 = (ip0 + dp).min(np);

    for i13 in ip0..ip1 {
        for i12 in 0..ne12 {
            for i11 in 0..ne11 {
                let jp1 = i13 * ne12 * ne11 * ne10 + i12 * ne11 * ne10 + i11 * ne10;
                for i10 in 0..ne10 {
                    let jp0 = jp1 + i10;
                    let src1_e = src1[jp0];
                    let src2_e = src2[jp0];

                    let jdh = jp0 * ne10;
                    let jdw = jdh - (ne10 - 1) * i10;

                    for j in 0..ne10 {
                        dst[jdh + j] += src2_e;
                        dst[jdw + j * ne10] += src1_e;
                    }
                }
            }
        }
    }
}

// ─── concat_any ──────────────────────────────────────────────────────────────
// ops.cpp lines 1899-1949

/// Translation of `ggml_compute_forward_concat_any`.
/// Concatenates src0 and src1 along `dim` by raw byte copy.
/// `row_size_src0` and `row_size_src1` are the byte sizes of one row.
pub fn concat_any(
    src0_bytes: &[u8], src1_bytes: &[u8],
    dst_bytes: &mut [u8],
    // src0 shape
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    // src1 shape
    ne10: usize, ne11: usize, ne12: usize, ne13: usize,
    // dst stride in bytes
    nb1: usize, nb2: usize, nb3: usize,
    // src0 stride in bytes
    nb01: usize, nb02: usize, nb03: usize,
    // src1 stride in bytes
    nb11: usize, nb12: usize, nb13: usize,
    dim: usize,
    // byte size of one element
    elem_size: usize,
    ith: usize, nth: usize,
) {
    // offset[dim] of src1 in dst
    let o = [
        if dim == 0 { ne00 } else { 0 },
        if dim == 1 { ne01 } else { 0 },
        if dim == 2 { ne02 } else { 0 },
        if dim == 3 { ne03 } else { 0 },
    ];
    let row_bytes_src0 = ne00 * elem_size;
    let row_bytes_src1 = ne10 * elem_size;

    // Region 1: src0 → dst, parallelise over i2
    for i3 in 0..ne03 {
        let mut i2 = ith;
        while i2 < ne02 {
            for i1 in 0..ne01 {
                let src_off = i1 * nb01 + i2 * nb02 + i3 * nb03;
                let dst_off = i1 * nb1 + i2 * nb2 + i3 * nb3;
                dst_bytes[dst_off..dst_off + row_bytes_src0]
                    .copy_from_slice(&src0_bytes[src_off..src_off + row_bytes_src0]);
            }
            i2 += nth;
        }
    }

    // Region 2: src1 → dst with offset
    for i3 in 0..ne13 {
        let mut i2 = ith;
        while i2 < ne12 {
            for i1 in 0..ne11 {
                let src_off = i1 * nb11 + i2 * nb12 + i3 * nb13;
                let dst_off = (i1 + o[1]) * nb1 + (i2 + o[2]) * nb2 + (i3 + o[3]) * nb3 + o[0] * elem_size;
                dst_bytes[dst_off..dst_off + row_bytes_src1]
                    .copy_from_slice(&src1_bytes[src_off..src_off + row_bytes_src1]);
            }
            i2 += nth;
        }
    }
}

// ─── out_prod_q_f32 ──────────────────────────────────────────────────────────
// ops.cpp lines 4492-4579

/// Translation of `ggml_compute_forward_out_prod_q_f32`.
/// Outer product: dst[i0,i1] += dequant(src0[i0,i01]) * src1[i1,i01]
/// `dequantize_row` converts one quantised row to f32.
pub fn out_prod_q_f32<F>(
    src1: &[f32],
    dst: &mut [f32],
    ne0: usize, ne1: usize, ne01: usize, ne2: usize, ne3: usize,
    dequantize_row: F,
    wdata: &mut [f32],
    ith: usize, nth: usize,
) where F: Fn(usize, &mut [f32]) // (row_idx, output_buf)
{
    // thread 0 zeros dst before barrier; caller must call ith==0 first and barrier
    let nr = ne1 * ne2 * ne3;
    let dr = nr.div_ceil(nth);
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);

    for ir in ir0..ir1 {
        let i3 = ir / (ne2 * ne1);
        let i2 = (ir - i3 * ne2 * ne1) / ne1;
        let i1 = ir - i3 * ne2 * ne1 - i2 * ne1;

        let d = &mut dst[(i3 * ne2 * ne1 + i2 * ne1 + i1) * ne0..(i3 * ne2 * ne1 + i2 * ne1 + i1) * ne0 + ne0];

        for i01 in 0..ne01 {
            let i11 = i01;
            // dequantize src0 row (i01 + i2*ne01 + i3*ne01*ne2) into wdata
            let row_idx = i3 * ne2 * ne01 + i2 * ne01 + i01;
            dequantize_row(row_idx, &mut wdata[..ne0]);
            let s1_val = src1[(i3 * ne2 * ne1 * ne01 + i2 * ne1 * ne01 + i11 * ne1 + i1)];
            for i in 0..ne0 {
                d[i] += wdata[i] * s1_val;
            }
        }
    }
}

// ── From ops.cpp:17 ──────────────────────────────────────────────────────────
/// ggml_compute_forward_dup_same_cont — parallel memcpy for contiguous same-type tensors.
/// Parallelizes by blocks. ith/nth = thread index / count.
pub fn dup_same_cont(src: &[u8], dst: &mut [u8], type_size: usize, ith: usize, nth: usize) {
    let nk = src.len() / type_size;
    let dr = (nk + nth - 1) / nth;
    let k0 = dr * ith;
    let k1 = (k0 + dr).min(nk);
    if k0 < k1 {
        let byte_start = k0 * type_size;
        let byte_end = k1 * type_size;
        dst[byte_start..byte_end].copy_from_slice(&src[byte_start..byte_end]);
    }
}

// ── From ops.cpp:47 (f32→f32 contiguous row copy variant) ────────────────────
/// ggml_compute_forward_dup_flt — element-wise copy with type conversion via f32 intermediary.
/// Handles f32↔f16 conversion. Parallel by rows (ith/nth).
/// ne00: src cols, ne01: src rows, ne02: batch2, ne03: batch3
/// nb00/nb01/nb02/nb03: src strides in bytes
/// nb0/nb1/nb2/nb3: dst strides in bytes
#[allow(clippy::too_many_arguments)]
pub fn dup_flt_f32_f32(
    src: *const u8, ne00: i64, ne01: i64, ne02: i64, ne03: i64,
    nb00: usize, nb01: usize, nb02: usize, nb03: usize,
    dst: *mut u8, nb0: usize, nb1: usize, nb2: usize, nb3: usize,
    ith: usize, nth: usize,
) {
    let nr = ne01 as usize;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    // contiguous same-type fast path: copy by rows
    if nb00 == 4 && nb0 == 4 && ne00 as usize * 4 == nb1.min(nb01) {
        let rs = ne00 as usize * 4;
        unsafe {
            for i03 in 0..ne03 as usize {
                for i02 in 0..ne02 as usize {
                    for i01 in ir0..ir1 {
                        let s = src.add(i01*nb01 + i02*nb02 + i03*nb03);
                        let d = dst.add(i01*nb1  + i02*nb2  + i03*nb3);
                        std::ptr::copy_nonoverlapping(s, d, rs);
                    }
                }
            }
        }
        return;
    }
    // generic strided copy (f32→f32 element-wise)
    unsafe {
        for i03 in 0..ne03 as usize {
            for i02 in 0..ne02 as usize {
                for i01 in ir0..ir1 {
                    for i00 in 0..ne00 as usize {
                        let sv = src.add(i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03) as *const f32;
                        let dv = dst.add(i00*nb0  + i01*nb1  + i02*nb2  + i03*nb3)  as *mut f32;
                        *dv = *sv;
                    }
                }
            }
        }
    }
}

/// ggml_compute_forward_dup_flt — f32→f16 variant. Converts via f32.
#[allow(clippy::too_many_arguments)]
pub fn dup_flt_f32_f16(
    src: *const u8, ne00: i64, ne01: i64, ne02: i64, ne03: i64,
    nb00: usize, nb01: usize, nb02: usize, nb03: usize,
    dst: *mut u8, nb0: usize, nb1: usize, nb2: usize, nb3: usize,
    ith: usize, nth: usize,
) {
    let nr = ne01 as usize;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    unsafe {
        for i03 in 0..ne03 as usize {
            for i02 in 0..ne02 as usize {
                for i01 in ir0..ir1 {
                    for i00 in 0..ne00 as usize {
                        let sv = *(src.add(i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03) as *const f32);
                        let dv = dst.add(i00*nb0 + i01*nb1 + i02*nb2 + i03*nb3) as *mut u16;
                        *dv = half::f16::from_f32(sv).to_bits();
                    }
                }
            }
        }
    }
}

/// ggml_compute_forward_dup_flt — f16→f32 variant.
#[allow(clippy::too_many_arguments)]
pub fn dup_flt_f16_f32(
    src: *const u8, ne00: i64, ne01: i64, ne02: i64, ne03: i64,
    nb00: usize, nb01: usize, nb02: usize, nb03: usize,
    dst: *mut u8, nb0: usize, nb1: usize, nb2: usize, nb3: usize,
    ith: usize, nth: usize,
) {
    let nr = ne01 as usize;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    unsafe {
        for i03 in 0..ne03 as usize {
            for i02 in 0..ne02 as usize {
                for i01 in ir0..ir1 {
                    for i00 in 0..ne00 as usize {
                        let sv_bits = *(src.add(i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03) as *const u16);
                        let sv = f16_to_f32(sv_bits);
                        let dv = dst.add(i00*nb0 + i01*nb1 + i02*nb2 + i03*nb3) as *mut f32;
                        *dv = sv;
                    }
                }
            }
        }
    }
}

// ── From ops.cpp:475 ─────────────────────────────────────────────────────────
/// ggml_compute_forward_dup_from_q — dequantize quantized src to float dst.
/// `dequantize_row_q`: fn(&[u8], &mut [f32]) for one quantized block (qk elements).
/// Parallel by quantized blocks (ir0..ir1 each thread).
pub fn dup_from_q(
    src: &[u8], dst: &mut [f32],
    ne00: i64, ne01: i64, ne02: i64, ne03: i64,
    nb00: usize, nb01: usize, nb02: usize, nb03: usize,
    ne10: i64, ne11: i64, ne12: i64, ne13: i64,
    nb10: usize, nb11: usize, nb12: usize, nb13: usize,
    qk: usize,
    dequantize_row_q: fn(&[u8], &mut [f32]),
    ith: usize, nth: usize,
) {
    let nr = (ne10 * ne11 * ne12 * ne13) as usize / qk;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    let mut tmp = vec![0.0f32; qk];
    for ir in ir0..ir1 {
        let i = ir * qk;
        let i03 = i / (ne00 as usize * ne01 as usize * ne02 as usize);
        let i02 = (i - i03*ne00 as usize*ne01 as usize*ne02 as usize) / (ne00 as usize*ne01 as usize);
        let i01 = (i - i03*ne00 as usize*ne01 as usize*ne02 as usize - i02*ne01 as usize*ne00 as usize) / ne00 as usize;
        let i00 = i - i03*ne00 as usize*ne01 as usize*ne02 as usize - i02*ne01 as usize*ne00 as usize - i01*ne00 as usize;
        let x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03*nb03;

        let i13 = i / (ne10 as usize * ne11 as usize * ne12 as usize);
        let i12 = (i - i13*ne10 as usize*ne11 as usize*ne12 as usize) / (ne10 as usize*ne11 as usize);
        let i11 = (i - i13*ne10 as usize*ne11 as usize*ne12 as usize - i12*ne10 as usize*ne11 as usize) / ne10 as usize;
        let i10 = i - i13*ne10 as usize*ne11 as usize*ne12 as usize - i12*ne10 as usize*ne11 as usize - i11*ne10 as usize;
        let dst_offset = (i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13) / 4;

        dequantize_row_q(&src[x_offset..], &mut tmp);
        dst[dst_offset..dst_offset+qk].copy_from_slice(&tmp);
    }
}

// ── From ops.cpp:578 ─────────────────────────────────────────────────────────
/// ggml_compute_forward_add_q_f32 — dequantize row, add f32 src1, requantize.
/// `dequantize_row_q`: fn(&[u8], &mut [f32]) for one quantized row (ne00 elements).
/// `quantize_row_q`: optional fn(&[f32], &mut [u8]).
pub fn add_q_f32(
    src0: &[u8], src1: &[f32], dst: &mut [u8],
    ne00: usize, nr: usize,
    nb01: usize, nb02: usize, nb03: usize,
    ne01: usize, ne02: usize,
    nb11: usize, nb12: usize, nb13: usize,
    nb1: usize, nb2: usize, nb3: usize,
    nb0: usize,
    dequantize_row_q: fn(&[u8], &mut [f32]),
    quantize_row_q: Option<fn(&[f32], &mut [u8])>,
    ith: usize, nth: usize,
) {
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    let mut wdata = vec![0.0f32; ne00];
    for ir in ir0..ir1 {
        let i03 = ir / (ne02 * ne01);
        let i02 = (ir - i03*ne02*ne01) / ne01;
        let i01 = ir - i03*ne02*ne01 - i02*ne01;
        let i13 = i03; let i12 = i02; let i11 = i01;
        let i3 = i03; let i2 = i02; let i1 = i01;
        let src0_off = i01*nb01 + i02*nb02 + i03*nb03;
        let src1_off = (i11*nb11 + i12*nb12 + i13*nb13) / 4;
        let dst_off  = i1*nb1 + i2*nb2 + i3*nb3;
        dequantize_row_q(&src0[src0_off..], &mut wdata);
        let s1 = &src1[src1_off..src1_off+ne00];
        for i in 0..ne00 { wdata[i] += s1[i]; }
        if let Some(q) = quantize_row_q {
            q(&wdata, &mut dst[dst_off..]);
        } else {
            let dwords = &mut dst[dst_off..dst_off+ne00*nb0];
            // copy f32 wdata as bytes
            for i in 0..ne00 {
                let b = wdata[i].to_le_bytes();
                dwords[i*nb0..i*nb0+4].copy_from_slice(&b);
            }
        }
    }
}

// ── From ops.cpp:1513 ────────────────────────────────────────────────────────
/// ggml_compute_forward_mean_f32 — per-row mean (single-threaded, ith==0 only).
pub fn mean_f32(
    src: &[f32], dst: &mut [f32],
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    ith: usize,
) {
    if ith != 0 { return; }
    for i03 in 0..ne03 {
        for i02 in 0..ne02 {
            for i01 in 0..ne01 {
                let src_row = &src[(i01 + i02*ne01 + i03*ne02*ne01)*ne00..][..ne00];
                let dst_idx = i01 + i02*ne01 + i03*ne02*ne01;
                let sum: f32 = src_row.iter().sum();
                dst[dst_idx] = sum / ne00 as f32;
            }
        }
    }
}

// ── From ops.cpp:1410 ────────────────────────────────────────────────────────
/// ggml_compute_forward_cumsum_f32 — prefix sum per row, parallel by rows.
pub fn cumsum_f32(
    src: &[f32], dst: &mut [f32],
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    ith: usize, nth: usize,
) {
    let nr = ne01 * ne02 * ne03;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        let i03 = ir / (ne02 * ne01);
        let i02 = (ir - i03*ne02*ne01) / ne01;
        let i01 = ir - i03*ne02*ne01 - i02*ne01;
        let src_off = (i01 + i02*ne01 + i03*ne02*ne01) * ne00;
        let dst_off = src_off;
        let mut acc = 0.0f32;
        for i in 0..ne00 {
            acc += src[src_off + i];
            dst[dst_off + i] = acc;
        }
    }
}

// ── From ops.cpp:3827 ────────────────────────────────────────────────────────
/// ggml_compute_forward_norm_f32 — layer normalization. Parallel by rows (ith step nth).
pub fn norm_f32(
    src: &[f32], dst: &mut [f32],
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    eps: f32, ith: usize, nth: usize,
) {
    for i03 in 0..ne03 {
        for i02 in 0..ne02 {
            let mut i01 = ith;
            while i01 < ne01 {
                let off = (i01 + i02*ne01 + i03*ne02*ne01) * ne00;
                let src_row = &src[off..off+ne00];
                let dst_row = &mut dst[off..off+ne00];
                let sum: f32 = src_row.iter().sum();
                let mean = sum / ne00 as f32;
                let variance: f32 = src_row.iter().map(|&x| { let v = x - mean; v*v }).sum::<f32>() / ne00 as f32;
                let scale = 1.0 / (variance + eps).sqrt();
                for i in 0..ne00 {
                    dst_row[i] = (src_row[i] - mean) * scale;
                }
                i01 += nth;
            }
        }
    }
}

// ── From ops.cpp:5496 ────────────────────────────────────────────────────────
/// ggml_compute_forward_diag_mask_f32 — mask upper triangle with `value`.
/// n_past: how many past tokens to keep unmasked. Parallel by rows (j step nth).
pub fn diag_mask_f32(
    data: &mut [f32],
    nc: usize, nr: usize, nz: usize,
    n_past: usize, value: f32,
    ith: usize, nth: usize,
) {
    for k in 0..nz {
        let mut j = ith;
        while j < nr {
            for i in n_past..nc {
                if i > n_past + j {
                    data[k*nr*nc + j*nc + i] = value;
                }
            }
            j += nth;
        }
    }
}

// ── From ops.cpp:7683 ────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
pub enum PoolOp { Avg, Max }

/// ggml_compute_forward_pool_1d_ksp — 1D pooling (avg or max). Single-threaded (ith==0).
pub fn pool_1d_ksp(
    src: &[f32], iw: usize,
    dst: &mut [f32], ow: usize,
    nr: usize,
    op: PoolOp, k: usize, s: usize, p: usize,
    ith: usize,
) {
    if ith != 0 { return; }
    for ir in 0..nr {
        let srow = &src[ir*iw..(ir+1)*iw];
        let drow = &mut dst[ir*ow..(ir+1)*ow];
        for out_w in 0..ow {
            let mut res = match op { PoolOp::Avg => 0.0f32, PoolOp::Max => f32::NEG_INFINITY };
            let mut count = 0usize;
            let base = (out_w * s) as i64 - p as i64;
            for ki in 0..k as i64 {
                let j = base + ki;
                if j < 0 || j >= iw as i64 { continue; }
                let v = srow[j as usize];
                match op {
                    PoolOp::Avg => { res += v; count += 1; }
                    PoolOp::Max => { if v > res { res = v; } }
                }
            }
            drow[out_w] = match op {
                PoolOp::Avg => if count > 0 { res / count as f32 } else { 0.0 },
                PoolOp::Max => res,
            };
        }
    }
}

// ── From ops.cpp:7978 ────────────────────────────────────────────────────────
/// ggml_compute_forward_upscale_f32 — nearest-neighbor upscaling.
/// sf0/sf1/sf2/sf3: scale factors per dimension. Parallel by i2 (ith step nth).
#[allow(clippy::too_many_arguments)]
pub fn upscale_f32_nearest(
    src: &[f32], ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    dst: &mut [f32], ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    sf0: f32, sf1: f32, sf2: f32, sf3: f32,
    ith: usize, nth: usize,
) {
    for i3 in 0..ne3 {
        let i03 = (i3 as f32 / sf3) as usize;
        let mut i2 = ith;
        while i2 < ne2 {
            let i02 = (i2 as f32 / sf2) as usize;
            for i1 in 0..ne1 {
                let i01 = (i1 as f32 / sf1) as usize;
                for i0 in 0..ne0 {
                    let i00 = (i0 as f32 / sf0) as usize;
                    let sv = src[i00 + i01*ne00 + i02*ne00*ne01 + i03*ne00*ne01*ne02];
                    dst[i0 + i1*ne0 + i2*ne0*ne1 + i3*ne0*ne1*ne2] = sv;
                }
            }
            i2 += nth;
        }
    }
}

// ── From ops.cpp:8424 ────────────────────────────────────────────────────────
/// ggml_compute_forward_timestep_embedding_f32 — sinusoidal timestep embeddings.
/// Parallel by output half-dim j (ith step nth).
pub fn timestep_embedding_f32(
    timesteps: &[f32], dst: &mut [f32],
    ne00: usize, dim: usize, max_period: usize,
    ith: usize, nth: usize,
) {
    let half = dim / 2;
    for i in 0..ne00 {
        let embed_off = i * dim;
        let mut j = ith;
        while j < half {
            let freq = f32::exp(-(max_period as f32).ln() * j as f32 / half as f32);
            let arg = timesteps[i] * freq;
            dst[embed_off + j] = arg.cos();
            dst[embed_off + j + half] = arg.sin();
            j += nth;
        }
        if dim % 2 != 0 && ith == 0 {
            dst[embed_off + 2*half] = 0.0;
        }
    }
}

// ── From ops.cpp:10824 ───────────────────────────────────────────────────────
/// ggml_compute_forward_solve_tri_f32 — lower-triangular solve Ax=B.
/// A: [n×n], B: [k×n], X: [k×n] output. Parallel by (batch×k) columns.
#[allow(clippy::too_many_arguments)]
pub fn solve_tri_f32(
    a: &[f32], b: &[f32], x: &mut [f32],
    n: usize, k: usize, batches: usize,
    ith: usize, nth: usize,
) {
    let nr = batches * k;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        let batch = ir / k;
        let col = ir % k;
        let a_off = batch * n * n;
        let b_off = batch * n * k;
        let x_off = batch * n * k;
        for row in 0..n {
            let mut sum = 0.0f32;
            for t in 0..row {
                sum += a[a_off + row*n + t] * x[x_off + t*k + col];
            }
            let diag = a[a_off + row*n + row];
            x[x_off + row*k + col] = (b[b_off + row*k + col] - sum) / diag;
        }
    }
}
