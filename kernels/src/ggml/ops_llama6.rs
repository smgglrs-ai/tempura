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
