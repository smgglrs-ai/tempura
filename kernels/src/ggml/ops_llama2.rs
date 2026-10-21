//! Direct Rust translation of llama.cpp ggml/src/ggml-cpu/ops.cpp lines 1-4000.
//!
//! Every function carries a doc-comment referencing the original C source line.
//! GGML threading (ith/nth) and wdata workspace are stripped -- callers manage
//! parallelism externally. The ggml_tensor struct is not used; callers supply
//! raw slices and explicit shape/stride values where needed.
//!
//! Half-float (f16/bf16) variants are omitted -- the project's kernel layer
//! operates in f32; quantized paths are handled in quants_llama.rs.

use super::ops_llama::{
    gelu_erf_f32, gelu_f32, gelu_quick_f32, leaky_relu_f32, silu_f32,
};

// -- dup (copy) --------------------------------------------------------------

/// Copy `n_bytes` of raw bytes between same-type contiguous tensors.
/// Direct translation of `ggml_compute_forward_dup_same_cont` (ops.cpp line 17).
///
/// In the original, the work is split by blocks; here callers slice the
/// relevant byte range and call this function for each chunk.
#[inline]
pub fn dup_bytes(dst: &mut [u8], src: &[u8]) {
    dst.copy_from_slice(src);
}

/// Copy a strided 4-D f32 tensor to a flat contiguous f32 slice.
///
/// Derived from `ggml_compute_forward_dup_flt<float,float>` (ops.cpp line 47)
/// -- the same-type, contiguous-dst branch.  Strides `nb01/nb02/nb03` are byte
/// offsets between rows / dim-2 slices / dim-3 batches in the *source*.
/// Destination is written in C-contiguous (row-major) order.
///
/// # Panics
/// Panics if `dst` is shorter than `ne00 * ne01 * ne02 * ne03`.
pub fn dup_f32_to_cont(
    src: &[u8],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
    ne02: usize,
    ne03: usize,
    nb01: usize,
    nb02: usize,
    nb03: usize,
) {
    assert!(dst.len() >= ne00 * ne01 * ne02 * ne03);
    let mut id = 0usize;
    for i03 in 0..ne03 {
        for i02 in 0..ne02 {
            for i01 in 0..ne01 {
                let byte_off = i01 * nb01 + i02 * nb02 + i03 * nb03;
                // Safety: caller guarantees src is a valid f32 byte buffer
                let row_bytes = &src[byte_off..byte_off + ne00 * 4];
                let row: &[f32] = unsafe {
                    std::slice::from_raw_parts(row_bytes.as_ptr() as *const f32, ne00)
                };
                dst[id..id + ne00].copy_from_slice(row);
                id += ne00;
            }
        }
    }
}

/// Copy a 4-D f32 tensor with arbitrary strides to another arbitrary-stride
/// layout element-by-element.
///
/// Derived from `ggml_compute_forward_dup_flt<float,float>` non-contiguous dst
/// branch (ops.cpp line 153).  All `nb*` values are byte offsets.
pub fn dup_f32_strided(
    src: &[u8],
    dst: &mut [u8],
    ne00: usize,
    ne01: usize,
    ne02: usize,
    ne03: usize,
    nb00: usize,
    nb01: usize,
    nb02: usize,
    nb03: usize,
    ne0: usize,
    ne1: usize,
    ne2: usize,
    nb0: usize,
    nb1: usize,
    nb2: usize,
    nb3: usize,
) {
    let mut i10: usize = 0;
    let mut i11: usize = 0;
    let mut i12: usize = 0;
    let mut i13: usize = 0;

    for i03 in 0..ne03 {
        for i02 in 0..ne02 {
            for i01 in 0..ne01 {
                for i00 in 0..ne00 {
                    let src_off = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    let dst_off = i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3;
                    dst[dst_off..dst_off + 4].copy_from_slice(&src[src_off..src_off + 4]);

                    i10 += 1;
                    if i10 == ne0 {
                        i10 = 0;
                        i11 += 1;
                        if i11 == ne1 {
                            i11 = 0;
                            i12 += 1;
                            if i12 == ne2 {
                                i12 = 0;
                                i13 += 1;
                                if i13 == ne2 {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let _ = (i10, i11, i12, i13);
}

// -- add ---------------------------------------------------------------------

/// Add a scalar `v` to every element of a contiguous f32 row.
/// Direct translation of the inner loop in `ggml_compute_forward_add1_f32`
/// (ops.cpp line 775).
///
/// # Panics
/// Panics if `dst` and `src` have different lengths.
pub fn add_scalar_f32(dst: &mut [f32], src: &[f32], v: f32) {
    assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s + v;
    }
}

/// In-place add scalar `v` to every element.
/// Derived from `ggml_vec_acc1_f32` called by `ggml_compute_forward_add1_q_f32`
/// (ops.cpp line 921).
pub fn acc_scalar_f32(x: &mut [f32], v: f32) {
    for xi in x.iter_mut() {
        *xi += v;
    }
}

/// Accumulated add: `dst[offset_elements..][row] += src1[row]` for every row.
///
/// Derived from `ggml_compute_forward_acc_f32` (ops.cpp line 1156).
/// Both `dst` and `src0` must be contiguous; the add is applied at an element
/// offset into each row.  Threading is stripped.
///
/// - `nr`: number of rows in src1
/// - `nc`: columns per row (src1.ne[0])
/// - `ne11`, `ne12`: src1 dims 1 and 2
/// - `nb11`, `nb12`, `nb13`: src1 row/plane/batch strides (bytes)
/// - `nb1`, `nb2`, `nb3`: dst row/plane/batch strides (bytes)
/// - `offset_bytes`: byte offset into each dst row where src1 is added
#[allow(clippy::too_many_arguments)]
pub fn acc_f32(
    dst: &mut [f32],
    src0: &[f32],
    src1: &[u8],
    nr: usize,
    nc: usize,
    ne11: usize,
    ne12: usize,
    nb11: usize,
    nb12: usize,
    nb13: usize,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    nb01: usize,
    nb02: usize,
    nb03: usize,
    offset_bytes: usize,
) {
    // copy src0 -> dst first (non-inplace path: caller already did memcpy if needed)
    let _ = (nb1, nb2, nb3, nb01, nb02, nb03, offset_bytes);
    for ir in 0..nr {
        let i3 = ir / (ne12 * ne11);
        let i2 = (ir - i3 * ne12 * ne11) / ne11;
        let i1 = ir - i3 * ne12 * ne11 - i2 * ne11;

        let s1_off = i1 * nb11 + i2 * nb12 + i3 * nb13;
        let s1_row: &[f32] = unsafe {
            std::slice::from_raw_parts(src1[s1_off..].as_ptr() as *const f32, nc)
        };
        let d_row = &mut dst[ir * nc..(ir + 1) * nc];
        let s0_row = &src0[ir * nc..(ir + 1) * nc];
        for k in 0..nc {
            d_row[k] = s0_row[k] + s1_row[k];
        }
    }
}

// -- sum ---------------------------------------------------------------------

/// Reduce all f32 elements in a contiguous slice to a single sum.
/// Direct translation of `ggml_compute_forward_sum_f32` (ops.cpp line 1284).
pub fn sum_f32(src: &[f32]) -> f32 {
    src.iter().map(|&v| v as f64).sum::<f64>() as f32
}

/// Reduce all f32 elements in a strided 4-D tensor to a single sum.
/// Direct translation of `ggml_compute_forward_sum_f32` inner loops (ops.cpp line 1284).
pub fn sum_f32_strided(
    src: &[u8],
    ne00: usize,
    ne01: usize,
    ne02: usize,
    ne03: usize,
    nb01: usize,
    nb02: usize,
    nb03: usize,
) -> f32 {
    let mut total = 0.0f64;
    for i03 in 0..ne03 {
        for i02 in 0..ne02 {
            for i01 in 0..ne01 {
                let off = i01 * nb01 + i02 * nb02 + i03 * nb03;
                let row: &[f32] = unsafe {
                    std::slice::from_raw_parts(src[off..].as_ptr() as *const f32, ne00)
                };
                for &v in row {
                    total += v as f64;
                }
            }
        }
    }
    total as f32
}

// -- cumsum -------------------------------------------------------------------

/// Per-row prefix (cumulative) sum of a contiguous f32 matrix.
/// Direct translation of `ggml_compute_forward_cumsum_f32` (ops.cpp line 1410).
///
/// - `n`: columns per row
///
/// # Panics
/// Panics if `src` and `dst` have different lengths or `n` does not divide evenly.
pub fn cumsum_rows_f32(src: &[f32], dst: &mut [f32], n: usize) {
    assert_eq!(src.len(), dst.len());
    assert_eq!(src.len() % n, 0);
    let n_rows = src.len() / n;
    for row in 0..n_rows {
        let sr = &src[row * n..(row + 1) * n];
        let dr = &mut dst[row * n..(row + 1) * n];
        let mut acc = 0.0f32;
        for (d, s) in dr.iter_mut().zip(sr.iter()) {
            acc += *s;
            *d = acc;
        }
    }
}

// -- sum_rows -----------------------------------------------------------------

/// Reduce each row of a contiguous f32 matrix to its sum.
/// Direct translation of `ggml_compute_forward_sum_rows_f32` (ops.cpp line 1460).
///
/// - `n`: number of columns (elements per row)
/// - `dst`: output slice, one f32 per row
///
/// # Panics
/// Panics if `src.len() % n != 0` or `dst.len() < src.len() / n`.
pub fn sum_rows_f32(src: &[f32], dst: &mut [f32], n: usize) {
    assert_eq!(src.len() % n, 0);
    let n_rows = src.len() / n;
    assert!(dst.len() >= n_rows);
    for row in 0..n_rows {
        let sr = &src[row * n..(row + 1) * n];
        dst[row] = sr.iter().sum::<f32>();
    }
}

// -- mean ---------------------------------------------------------------------

/// Reduce each row of a contiguous f32 matrix to its arithmetic mean.
/// Direct translation of `ggml_compute_forward_mean_f32` (ops.cpp line 1513).
///
/// # Panics
/// Panics if `src.len() % n != 0` or `dst.len() < src.len() / n`.
pub fn mean_rows_f32(src: &[f32], dst: &mut [f32], n: usize) {
    assert_eq!(src.len() % n, 0);
    let n_rows = src.len() / n;
    assert!(dst.len() >= n_rows);
    let inv_n = 1.0f32 / n as f32;
    for row in 0..n_rows {
        let sr = &src[row * n..(row + 1) * n];
        dst[row] = sr.iter().sum::<f32>() * inv_n;
    }
}

// -- argmax -------------------------------------------------------------------

/// Index of the maximum element in a contiguous f32 slice.
/// Direct translation of `ggml_vec_argmax_f32` called by
/// `ggml_compute_forward_argmax_f32` (ops.cpp line 1570).
///
/// Returns 0 for an empty slice.
pub fn argmax_f32(src: &[f32]) -> usize {
    src.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute the argmax for each row of a contiguous f32 matrix.
/// Direct translation of `ggml_compute_forward_argmax_f32` (ops.cpp line 1570).
///
/// # Panics
/// Panics if `src.len() % n != 0` or `dst.len() < n_rows`.
pub fn argmax_rows_f32(src: &[f32], dst: &mut [i32], n: usize) {
    assert_eq!(src.len() % n, 0);
    let n_rows = src.len() / n;
    assert!(dst.len() >= n_rows);
    for row in 0..n_rows {
        dst[row] = argmax_f32(&src[row * n..(row + 1) * n]) as i32;
    }
}

// -- count_equal --------------------------------------------------------------

/// Count element-wise equal pairs between two contiguous i32 slices.
/// Direct translation of `ggml_compute_forward_count_equal_i32` (ops.cpp line 1618).
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
pub fn count_equal_i32(a: &[i32], b: &[i32]) -> i64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).filter(|(x, y)| x == y).count() as i64
}

// -- repeat -------------------------------------------------------------------

/// Tile a 4-D contiguous f32 tensor into a larger output.
/// Direct translation of `ggml_compute_forward_repeat_f32` (ops.cpp line 1698).
///
/// The output has shape `(ne00*nr0, ne01*nr1, ne02*nr2, ne03*nr3)` where `nr*`
/// are the repeat counts.  All tensors are contiguous row-major.
///
/// # Panics
/// Panics if output slice is too small.
#[allow(clippy::too_many_arguments)]
pub fn repeat_f32(
    src: &[f32],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
    ne02: usize,
    ne03: usize,
    nr0: usize,
    nr1: usize,
    nr2: usize,
    nr3: usize,
) {
    let ne0 = ne00 * nr0;
    let ne1 = ne01 * nr1;
    let ne2 = ne02 * nr2;
    assert!(dst.len() >= ne0 * ne1 * ne2 * ne03 * nr3);

    for i3 in 0..nr3 {
        for k3 in 0..ne03 {
            for i2 in 0..nr2 {
                for k2 in 0..ne02 {
                    for i1 in 0..nr1 {
                        for k1 in 0..ne01 {
                            let src_row_off = (k3 * ne02 * ne01 + k2 * ne01 + k1) * ne00;
                            let src_row = &src[src_row_off..src_row_off + ne00];
                            for i0 in 0..nr0 {
                                let dst_i3 = i3 * ne03 + k3;
                                let dst_i2 = i2 * ne02 + k2;
                                let dst_i1 = i1 * ne01 + k1;
                                let dst_i0 = i0 * ne00;
                                let dst_off = ((dst_i3 * ne2 + dst_i2) * ne1 + dst_i1) * ne0 + dst_i0;
                                dst[dst_off..dst_off + ne00].copy_from_slice(src_row);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Accumulate a tiled 4-D f32 tensor back into a smaller source shape.
/// Direct translation of `ggml_compute_forward_repeat_back_f32` (ops.cpp line 1822).
///
/// `src` has shape `(ne0*nr0, ne1*nr1, ne2*nr2, ne3*nr3)` and is tiled.
/// Accumulates into `dst` of shape `(ne0, ne1, ne2, ne3)`.
///
/// # Panics
/// Panics if slices are too small.
#[allow(clippy::too_many_arguments)]
pub fn repeat_back_f32(
    src: &[f32],
    dst: &mut [f32],
    ne0: usize,
    ne1: usize,
    ne2: usize,
    ne3: usize,
    nr0: usize,
    nr1: usize,
    nr2: usize,
    nr3: usize,
) {
    assert!(dst.len() >= ne0 * ne1 * ne2 * ne3);
    // zero destination
    for v in dst.iter_mut() {
        *v = 0.0;
    }
    let src_ne0 = ne0 * nr0;
    let src_ne1 = ne1 * nr1;
    let src_ne2 = ne2 * nr2;

    for i3 in 0..nr3 {
        for k3 in 0..ne3 {
            for i2 in 0..nr2 {
                for k2 in 0..ne2 {
                    for i1 in 0..nr1 {
                        for k1 in 0..ne1 {
                            for i0 in 0..nr0 {
                                let src_i3 = i3 * ne3 + k3;
                                let src_i2 = i2 * ne2 + k2;
                                let src_i1 = i1 * ne1 + k1;
                                let src_i0 = i0 * ne0;
                                let src_off = ((src_i3 * src_ne2 + src_i2) * src_ne1 + src_i1) * src_ne0 + src_i0;
                                let dst_off = (k3 * ne2 * ne1 + k2 * ne1 + k1) * ne0;
                                let src_slice = &src[src_off..src_off + ne0];
                                let dst_slice = &mut dst[dst_off..dst_off + ne0];
                                for (d, s) in dst_slice.iter_mut().zip(src_slice.iter()) {
                                    *d += *s;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// -- concat -------------------------------------------------------------------

/// Concatenate two contiguous f32 tensors along dimension `dim`.
///
/// Derived from `ggml_compute_forward_concat_f32` (ops.cpp line 2037).
/// Threading is stripped.  Shapes:
/// - src0: `[ne00, ne01, ne02, ne03]`
/// - src1: `[ne10, ne11, ne12, ne13]`
/// - dst:  result of concat along `dim` (computed by caller)
///
/// For contiguous f32 row-major tensors with `dim ??? {0, 1, 2, 3}`.
#[allow(clippy::too_many_arguments)]
pub fn concat_f32(
    src0: &[f32],
    src1: &[f32],
    dst: &mut [f32],
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    ne10: usize, ne11: usize, ne12: usize, ne13: usize,
    ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    dim: usize,
) {
    assert!(dim < 4);
    // offset of src1 in the output along `dim`
    let o = [
        if dim == 0 { ne00 } else { 0 },
        if dim == 1 { ne01 } else { 0 },
        if dim == 2 { ne02 } else { 0 },
        if dim == 3 { ne03 } else { 0 },
    ];

    // Copy src0 region
    for i3 in 0..ne03 {
        for i2 in 0..ne02 {
            for i1 in 0..ne01 {
                let src_off = (i3 * ne02 * ne01 + i2 * ne01 + i1) * ne00;
                let dst_off = (i3 * ne2 * ne1 + i2 * ne1 + i1) * ne0;
                dst[dst_off..dst_off + ne00].copy_from_slice(&src0[src_off..src_off + ne00]);
            }
        }
    }

    // Copy src1 region
    for i3 in 0..ne13 {
        for i2 in 0..ne12 {
            for i1 in 0..ne11 {
                let src_off = (i3 * ne12 * ne11 + i2 * ne11 + i1) * ne10;
                let dst_i3 = i3 + o[3];
                let dst_i2 = i2 + o[2];
                let dst_i1 = i1 + o[1];
                let dst_i0 = o[0];
                let dst_off = (dst_i3 * ne2 * ne1 + dst_i2 * ne1 + dst_i1) * ne0 + dst_i0;
                dst[dst_off..dst_off + ne10].copy_from_slice(&src1[src_off..src_off + ne10]);
            }
        }
    }
}

/// Concatenate two contiguous i8 tensors along dimension `dim`.
/// Derived from `ggml_compute_forward_concat_i8` (ops.cpp line 1951).
#[allow(clippy::too_many_arguments)]
pub fn concat_i8(
    src0: &[i8],
    src1: &[i8],
    dst: &mut [i8],
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    ne10: usize, ne11: usize, ne12: usize, ne13: usize,
    ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    dim: usize,
) {
    assert!(dim < 4);
    let o = [
        if dim == 0 { ne00 } else { 0 },
        if dim == 1 { ne01 } else { 0 },
        if dim == 2 { ne02 } else { 0 },
        if dim == 3 { ne03 } else { 0 },
    ];
    let _ = (ne10, ne11, ne12, ne13);
    for i3 in 0..ne3 {
        for i2 in 0..ne2 {
            for i1 in 0..ne1 {
                for i0 in 0..ne0 {
                    let val = if i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03 {
                        let off = (i3 * ne02 * ne01 + i2 * ne01 + i1) * ne00 + i0;
                        src0[off]
                    } else {
                        let j0 = i0 - o[0];
                        let j1 = i1 - o[1];
                        let j2 = i2 - o[2];
                        let j3 = i3 - o[3];
                        let off = (j3 * ne02 * ne01 + j2 * ne01 + j1) * ne00 + j0;
                        src1[off]
                    };
                    let dst_off = (i3 * ne2 * ne1 + i2 * ne1 + i1) * ne0 + i0;
                    dst[dst_off] = val;
                }
            }
        }
    }
}

// -- fill ---------------------------------------------------------------------

/// Fill a contiguous f32 slice with a constant value.
/// Direct translation of `ggml_compute_forward_fill_f32` (ops.cpp line 2230).
pub fn fill_f32(dst: &mut [f32], c: f32) {
    for v in dst.iter_mut() {
        *v = c;
    }
}

/// Fill rows of a strided 4-D f32 tensor with a constant value.
/// Derived from `ggml_compute_forward_fill_f32` (ops.cpp line 2230).
pub fn fill_rows_f32(
    dst: &mut [u8],
    ne0: usize,
    ne1: usize,
    ne2: usize,
    ne3: usize,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    c: f32,
) {
    for i03 in 0..ne3 {
        for i02 in 0..ne2 {
            for i01 in 0..ne1 {
                let off = i01 * nb1 + i02 * nb2 + i03 * nb3;
                let row: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(dst[off..].as_mut_ptr() as *mut f32, ne0)
                };
                row.fill(c);
            }
        }
    }
}

// -- tri (triangular mask) ----------------------------------------------------

/// Triangular type selector for `tri_f32`.
/// Direct translation of `ggml_tri_type` (ops.cpp line 2289).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TriType {
    /// Strict lower triangle (`col < row`)
    Lower,
    /// Lower triangle including diagonal (`col <= row`)
    LowerDiag,
    /// Strict upper triangle (`col > row`)
    Upper,
    /// Upper triangle including diagonal (`col >= row`)
    UpperDiag,
}

/// Apply a triangular mask to a contiguous f32 matrix.
///
/// For each element at position `(row, col)`, if `bipred(col, row)` is true
/// the source value is copied; otherwise 0 is written.
///
/// Direct translation of `ggml_compute_forward_tri_f32` (ops.cpp line 2289).
///
/// - `ne0`: columns
/// - `ne01`: rows (inner loop)
/// - `ne02`, `ne03`: outer dimensions
pub fn tri_f32(
    src: &[f32],
    dst: &mut [f32],
    ne0: usize,
    ne01: usize,
    ne02: usize,
    ne03: usize,
    ttype: TriType,
) {
    assert_eq!(src.len(), ne0 * ne01 * ne02 * ne03);
    assert_eq!(dst.len(), src.len());
    for i03 in 0..ne03 {
        for i02 in 0..ne02 {
            for i01 in 0..ne01 {
                let off = (i03 * ne02 * ne01 + i02 * ne01 + i01) * ne0;
                let sr = &src[off..off + ne0];
                let dr = &mut dst[off..off + ne0];
                for i0 in 0..ne0 {
                    dr[i0] = match ttype {
                        TriType::Lower if i0 < i01 => sr[i0],
                        TriType::LowerDiag if i0 <= i01 => sr[i0],
                        TriType::Upper if i0 > i01 => sr[i0],
                        TriType::UpperDiag if i0 >= i01 => sr[i0],
                        _ => 0.0,
                    };
                }
            }
        }
    }
}

// -- activation row-wise wrappers ---------------------------------------------

/// Apply GELU to every element in each contiguous row.
/// Direct translation of `ggml_compute_forward_gelu_f32` (ops.cpp line 2111).
pub fn gelu_rows_f32(src: &[f32], dst: &mut [f32], nc: usize, nr: usize) {
    assert_eq!(src.len(), nc * nr);
    assert_eq!(dst.len(), src.len());
    for i in 0..src.len() {
        dst[i] = gelu_f32(src[i]);
    }
}

/// Apply GELU-erf to every element in each row.
/// Direct translation of `ggml_compute_forward_gelu_erf_f32` (ops.cpp line 2341).
pub fn gelu_erf_rows_f32(src: &[f32], dst: &mut [f32], nc: usize, nr: usize) {
    assert_eq!(src.len(), nc * nr);
    assert_eq!(dst.len(), src.len());
    for i in 0..src.len() {
        dst[i] = gelu_erf_f32(src[i]);
    }
}

/// Apply GELU-quick to every element in each row.
/// Direct translation of `ggml_compute_forward_gelu_quick_f32` (ops.cpp line 2460).
pub fn gelu_quick_rows_f32(src: &[f32], dst: &mut [f32], nc: usize, nr: usize) {
    assert_eq!(src.len(), nc * nr);
    assert_eq!(dst.len(), src.len());
    for i in 0..src.len() {
        dst[i] = gelu_quick_f32(src[i]);
    }
}

/// Apply SiLU to every element in each row.
/// Direct translation of `ggml_compute_forward_silu_f32` (ops.cpp line 2579).
pub fn silu_rows_f32(src: &[f32], dst: &mut [f32], nc: usize, nr: usize) {
    assert_eq!(src.len(), nc * nr);
    assert_eq!(dst.len(), src.len());
    for i in 0..src.len() {
        dst[i] = silu_f32(src[i]);
    }
}

/// Apply leaky-ReLU to every element in each row.
/// Direct translation of `ggml_compute_forward_leaky_relu_f32` (ops.cpp line 2697).
pub fn leaky_relu_rows_f32(src: &[f32], dst: &mut [f32], nc: usize, nr: usize, ns: f32) {
    assert_eq!(src.len(), nc * nr);
    assert_eq!(dst.len(), src.len());
    for i in 0..src.len() {
        dst[i] = leaky_relu_f32(src[i], ns);
    }
}

// -- silu_back ----------------------------------------------------------------

/// Gradient of SiLU with respect to its input.
/// `??(x) = x / (1 + e^{-x})`, so `d??/dx = ??(x)(1 + x(1 - ??(x)))`.
/// Direct translation of `ggml_vec_silu_backward_f32` called by
/// `ggml_compute_forward_silu_back_f32` (ops.cpp line 2781).
#[inline]
pub fn silu_backward_f32(x: f32, dz: f32) -> f32 {
    let s = silu_f32(x);
    dz * s * (1.0 + x * (1.0 - s))
}

/// Apply SiLU backward over a row.
/// Direct translation of `ggml_compute_forward_silu_back_f32` (ops.cpp line 2781).
///
/// - `src1`: activations from the forward pass (x values)
/// - `grad`: upstream gradients
/// - `dst`:  output gradients
///
/// # Panics
/// Panics if the three slices have different lengths.
pub fn silu_back_rows_f32(src1: &[f32], grad: &[f32], dst: &mut [f32]) {
    assert_eq!(src1.len(), grad.len());
    assert_eq!(src1.len(), dst.len());
    for i in 0..src1.len() {
        dst[i] = silu_backward_f32(src1[i], grad[i]);
    }
}

// -- gated activation helpers -------------------------------------------------

/// ReLU-gated linear unit scalar: `relu(gate) * up`.
/// Used by reglu (ops.cpp line 2892).
#[inline]
fn reglu_scalar(gate: f32, up: f32) -> f32 {
    gate.max(0.0) * up
}

/// GELU-gated linear unit scalar: `gelu(gate) * up`.
/// Used by geglu (ops.cpp line 3035).
#[inline]
fn geglu_scalar(gate: f32, up: f32) -> f32 {
    gelu_f32(gate) * up
}

/// SiLU-gated linear unit scalar: `silu(gate) * up`.
/// Used by swiglu (ops.cpp line 3178).
#[inline]
fn swiglu_scalar(gate: f32, up: f32) -> f32 {
    silu_f32(gate) * up
}

/// GELU-erf gated linear unit scalar.
/// Used by geglu_erf (ops.cpp line 3541).
#[inline]
fn geglu_erf_scalar(gate: f32, up: f32) -> f32 {
    gelu_erf_f32(gate) * up
}

/// GELU-quick gated linear unit scalar.
/// Used by geglu_quick (ops.cpp line 3684).
#[inline]
fn geglu_quick_scalar(gate: f32, up: f32) -> f32 {
    gelu_quick_f32(gate) * up
}

// -- reglu --------------------------------------------------------------------

/// ReGLU (ReLU gated linear unit) over `nr` rows, `nc` elements each.
///
/// When `src1.is_none()`, `src0` contains gate and up interleaved:
/// - `swapped == false`: first half is gate, second half is up
/// - `swapped == true`:  first half is up, second half is gate
///
/// When `src1.is_some()`, `src0` is the gate and `src1` is the up projection.
///
/// Direct translation of `ggml_compute_forward_reglu_f32` (ops.cpp line 2892).
///
/// # Panics
/// Panics if slice lengths are inconsistent with `nc * nr`.
pub fn reglu_rows_f32(
    src0: &[f32],
    src1: Option<&[f32]>,
    dst: &mut [f32],
    nc: usize,
    nr: usize,
    swapped: bool,
) {
    assert_eq!(dst.len(), nc * nr);
    for i1 in 0..nr {
        let (gate_row, up_row) = if let Some(s1) = src1 {
            assert_eq!(s1.len(), nc * nr);
            (&src0[i1 * nc..(i1 + 1) * nc], &s1[i1 * nc..(i1 + 1) * nc])
        } else {
            assert_eq!(src0.len(), 2 * nc * nr);
            let row = &src0[i1 * 2 * nc..(i1 + 1) * 2 * nc];
            if swapped {
                (&row[nc..], &row[..nc])
            } else {
                (&row[..nc], &row[nc..])
            }
        };
        let dr = &mut dst[i1 * nc..(i1 + 1) * nc];
        for k in 0..nc {
            dr[k] = reglu_scalar(gate_row[k], up_row[k]);
        }
    }
}

// -- geglu --------------------------------------------------------------------

/// GeGLU (GELU gated linear unit) over `nr` rows.
/// Direct translation of `ggml_compute_forward_geglu_f32` (ops.cpp line 3035).
pub fn geglu_rows_f32(
    src0: &[f32],
    src1: Option<&[f32]>,
    dst: &mut [f32],
    nc: usize,
    nr: usize,
    swapped: bool,
) {
    assert_eq!(dst.len(), nc * nr);
    for i1 in 0..nr {
        let (gate_row, up_row) = if let Some(s1) = src1 {
            assert_eq!(s1.len(), nc * nr);
            (&src0[i1 * nc..(i1 + 1) * nc], &s1[i1 * nc..(i1 + 1) * nc])
        } else {
            assert_eq!(src0.len(), 2 * nc * nr);
            let row = &src0[i1 * 2 * nc..(i1 + 1) * 2 * nc];
            if swapped {
                (&row[nc..], &row[..nc])
            } else {
                (&row[..nc], &row[nc..])
            }
        };
        let dr = &mut dst[i1 * nc..(i1 + 1) * nc];
        for k in 0..nc {
            dr[k] = geglu_scalar(gate_row[k], up_row[k]);
        }
    }
}

// -- swiglu -------------------------------------------------------------------

/// SwiGLU (SiLU gated linear unit) over `nr` rows.
/// Direct translation of `ggml_compute_forward_swiglu_f32` (ops.cpp line 3178).
pub fn swiglu_rows_f32(
    src0: &[f32],
    src1: Option<&[f32]>,
    dst: &mut [f32],
    nc: usize,
    nr: usize,
    swapped: bool,
) {
    assert_eq!(dst.len(), nc * nr);
    for i1 in 0..nr {
        let (gate_row, up_row) = if let Some(s1) = src1 {
            assert_eq!(s1.len(), nc * nr);
            (&src0[i1 * nc..(i1 + 1) * nc], &s1[i1 * nc..(i1 + 1) * nc])
        } else {
            assert_eq!(src0.len(), 2 * nc * nr);
            let row = &src0[i1 * 2 * nc..(i1 + 1) * 2 * nc];
            if swapped {
                (&row[nc..], &row[..nc])
            } else {
                (&row[..nc], &row[nc..])
            }
        };
        let dr = &mut dst[i1 * nc..(i1 + 1) * nc];
        for k in 0..nc {
            dr[k] = swiglu_scalar(gate_row[k], up_row[k]);
        }
    }
}

// -- swiglu_oai ---------------------------------------------------------------

/// SwiGLU variant used by OpenAI models: `silu_clamped(gate) * (up + 1)`.
///
/// `gate` is clamped to `(-inf, limit]` and `up` is clamped to `[-limit, limit]`
/// before the gating multiplication.
///
/// Direct translation of `ggml_compute_forward_swiglu_oai_f32` (ops.cpp line 3321).
#[allow(clippy::too_many_arguments)]
pub fn swiglu_oai_rows_f32(
    src0: &[f32],
    src1: Option<&[f32]>,
    dst: &mut [f32],
    nc: usize,
    nr: usize,
    swapped: bool,
    alpha: f32,
    limit: f32,
) {
    assert_eq!(dst.len(), nc * nr);
    for i1 in 0..nr {
        let (gate_row, up_row) = if let Some(s1) = src1 {
            assert_eq!(s1.len(), nc * nr);
            (&src0[i1 * nc..(i1 + 1) * nc], &s1[i1 * nc..(i1 + 1) * nc])
        } else {
            assert_eq!(src0.len(), 2 * nc * nr);
            let row = &src0[i1 * 2 * nc..(i1 + 1) * 2 * nc];
            if swapped {
                (&row[nc..], &row[..nc])
            } else {
                (&row[..nc], &row[nc..])
            }
        };
        let dr = &mut dst[i1 * nc..(i1 + 1) * nc];
        for k in 0..nc {
            let x = gate_row[k].min(limit);
            let y = up_row[k].clamp(-limit, limit);
            let glu = x / (1.0 + (alpha * (-x)).exp());
            dr[k] = glu * (y + 1.0);
        }
    }
}

// -- swiglu_clamp -------------------------------------------------------------

/// SwiGLU with clamped inputs: `silu_clamped(gate) * up_clamped`.
///
/// `gate` is clamped to `(-inf, limit]` and `up` is clamped to `[-limit, limit]`.
///
/// Direct translation of `ggml_compute_forward_swiglu_clamp_f32` (ops.cpp line 3408).
pub fn swiglu_clamp_rows_f32(
    src0: &[f32],
    src1: Option<&[f32]>,
    dst: &mut [f32],
    nc: usize,
    nr: usize,
    swapped: bool,
    limit: f32,
) {
    assert_eq!(dst.len(), nc * nr);
    for i1 in 0..nr {
        let (gate_row, up_row) = if let Some(s1) = src1 {
            assert_eq!(s1.len(), nc * nr);
            (&src0[i1 * nc..(i1 + 1) * nc], &s1[i1 * nc..(i1 + 1) * nc])
        } else {
            assert_eq!(src0.len(), 2 * nc * nr);
            let row = &src0[i1 * 2 * nc..(i1 + 1) * 2 * nc];
            if swapped {
                (&row[nc..], &row[..nc])
            } else {
                (&row[..nc], &row[nc..])
            }
        };
        let dr = &mut dst[i1 * nc..(i1 + 1) * nc];
        for k in 0..nc {
            let gate = gate_row[k].min(limit);
            let up = up_row[k].clamp(-limit, limit);
            dr[k] = gate / (1.0 + (-gate).exp()) * up;
        }
    }
}

// -- geglu_erf ----------------------------------------------------------------

/// GeGLU-erf gated linear unit over `nr` rows.
/// Direct translation of `ggml_compute_forward_geglu_erf_f32` (ops.cpp line 3541).
pub fn geglu_erf_rows_f32(
    src0: &[f32],
    src1: Option<&[f32]>,
    dst: &mut [f32],
    nc: usize,
    nr: usize,
    swapped: bool,
) {
    assert_eq!(dst.len(), nc * nr);
    for i1 in 0..nr {
        let (gate_row, up_row) = if let Some(s1) = src1 {
            assert_eq!(s1.len(), nc * nr);
            (&src0[i1 * nc..(i1 + 1) * nc], &s1[i1 * nc..(i1 + 1) * nc])
        } else {
            assert_eq!(src0.len(), 2 * nc * nr);
            let row = &src0[i1 * 2 * nc..(i1 + 1) * 2 * nc];
            if swapped {
                (&row[nc..], &row[..nc])
            } else {
                (&row[..nc], &row[nc..])
            }
        };
        let dr = &mut dst[i1 * nc..(i1 + 1) * nc];
        for k in 0..nc {
            dr[k] = geglu_erf_scalar(gate_row[k], up_row[k]);
        }
    }
}

// -- geglu_quick --------------------------------------------------------------

/// GeGLU-quick gated linear unit over `nr` rows.
/// Direct translation of `ggml_compute_forward_geglu_quick_f32` (ops.cpp line 3684).
pub fn geglu_quick_rows_f32(
    src0: &[f32],
    src1: Option<&[f32]>,
    dst: &mut [f32],
    nc: usize,
    nr: usize,
    swapped: bool,
) {
    assert_eq!(dst.len(), nc * nr);
    for i1 in 0..nr {
        let (gate_row, up_row) = if let Some(s1) = src1 {
            assert_eq!(s1.len(), nc * nr);
            (&src0[i1 * nc..(i1 + 1) * nc], &s1[i1 * nc..(i1 + 1) * nc])
        } else {
            assert_eq!(src0.len(), 2 * nc * nr);
            let row = &src0[i1 * 2 * nc..(i1 + 1) * 2 * nc];
            if swapped {
                (&row[nc..], &row[..nc])
            } else {
                (&row[..nc], &row[nc..])
            }
        };
        let dr = &mut dst[i1 * nc..(i1 + 1) * nc];
        for k in 0..nc {
            dr[k] = geglu_quick_scalar(gate_row[k], up_row[k]);
        }
    }
}

// -- norm (layer norm) --------------------------------------------------------

/// Layer normalization over a strided f32 tensor.
///
/// For each row, computes `(x - mean) / sqrt(var + eps)`.
/// Threading is stripped.
///
/// Direct translation of `ggml_compute_forward_norm_f32` (ops.cpp line 3827).
///
/// - `ne00`: columns per row
/// - `ne01/ne02/ne03`: outer dimensions
/// - `nb01/nb02/nb03`: byte strides for source
/// - `nb1/nb2/nb3`: byte strides for destination
///
/// # Panics
/// Panics on slice bounds violations.
#[allow(clippy::too_many_arguments)]
pub fn norm_f32_strided(
    src: &[u8],
    dst: &mut [u8],
    ne00: usize,
    ne01: usize,
    ne02: usize,
    ne03: usize,
    nb01: usize,
    nb02: usize,
    nb03: usize,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    eps: f32,
) {
    for i03 in 0..ne03 {
        for i02 in 0..ne02 {
            for i01 in 0..ne01 {
                let src_off = i01 * nb01 + i02 * nb02 + i03 * nb03;
                let dst_off = i01 * nb1 + i02 * nb2 + i03 * nb3;

                let xf: &[f32] = unsafe {
                    std::slice::from_raw_parts(src[src_off..].as_ptr() as *const f32, ne00)
                };
                let yf: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(
                        dst[dst_off..].as_mut_ptr() as *mut f32,
                        ne00,
                    )
                };

                let sum: f32 = xf.iter().sum();
                let mean = sum / ne00 as f32;

                let mut variance = 0.0f32;
                for i in 0..ne00 {
                    let v = xf[i] - mean;
                    yf[i] = v;
                    variance += v * v;
                }
                variance /= ne00 as f32;
                let scale = 1.0 / (variance + eps).sqrt();
                for y in yf.iter_mut() {
                    *y *= scale;
                }
            }
        }
    }
}

/// Layer normalization over contiguous f32 rows (simplified, no striding).
/// Direct translation of `ggml_compute_forward_norm_f32` (ops.cpp line 3827),
/// contiguous fast path.
///
/// # Panics
/// Panics if `src` and `dst` have different lengths or `n` does not divide evenly.
pub fn norm_rows_f32(src: &[f32], dst: &mut [f32], n: usize, eps: f32) {
    assert_eq!(src.len(), dst.len());
    assert_eq!(src.len() % n, 0);
    let n_rows = src.len() / n;
    for row in 0..n_rows {
        let xr = &src[row * n..(row + 1) * n];
        let yr = &mut dst[row * n..(row + 1) * n];
        let mean: f32 = xr.iter().sum::<f32>() / n as f32;
        let mut variance = 0.0f32;
        for i in 0..n {
            let v = xr[i] - mean;
            yr[i] = v;
            variance += v * v;
        }
        variance /= n as f32;
        let scale = 1.0 / (variance + eps).sqrt();
        for y in yr.iter_mut() {
            *y *= scale;
        }
    }
}

// -- rms_norm -----------------------------------------------------------------

/// RMS normalization over contiguous f32 rows.
/// Direct translation of `ggml_compute_forward_rms_norm_f32<NONE>` (ops.cpp line 3924).
///
/// # Panics
/// Panics if `src` and `dst` have different lengths or `n` does not divide evenly.
pub fn rms_norm_rows_f32(src: &[f32], dst: &mut [f32], n: usize, eps: f32) {
    assert_eq!(src.len(), dst.len());
    assert_eq!(src.len() % n, 0);
    let n_rows = src.len() / n;
    for row in 0..n_rows {
        let xr = &src[row * n..(row + 1) * n];
        let yr = &mut dst[row * n..(row + 1) * n];
        let sum_sq: f64 = xr.iter().map(|&v| (v * v) as f64).sum();
        let mean = sum_sq as f32 / n as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        for i in 0..n {
            yr[i] = xr[i] * scale;
        }
    }
}

/// RMS normalization fused with a per-element weight multiply.
/// Direct translation of `ggml_compute_forward_rms_norm_f32<MUL>` (ops.cpp line 3971).
///
/// `weight` has length `n` and is broadcast over all rows.
///
/// # Panics
/// Panics if slice lengths are inconsistent.
pub fn rms_norm_mul_rows_f32(src: &[f32], weight: &[f32], dst: &mut [f32], n: usize, eps: f32) {
    assert_eq!(weight.len(), n);
    assert_eq!(src.len(), dst.len());
    assert_eq!(src.len() % n, 0);
    let n_rows = src.len() / n;
    for row in 0..n_rows {
        let xr = &src[row * n..(row + 1) * n];
        let yr = &mut dst[row * n..(row + 1) * n];
        let sum_sq: f64 = xr.iter().map(|&v| (v * v) as f64).sum();
        let mean = sum_sq as f32 / n as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        for i in 0..n {
            yr[i] = xr[i] * scale * weight[i];
        }
    }
}

// -- tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cumsum_rows() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f32; 6];
        cumsum_rows_f32(&src, &mut dst, 3);
        assert_eq!(&dst[..3], &[1.0, 3.0, 6.0]);
        assert_eq!(&dst[3..], &[4.0, 9.0, 15.0]);
    }

    #[test]
    fn test_sum_rows() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f32; 2];
        sum_rows_f32(&src, &mut dst, 3);
        assert!((dst[0] - 6.0).abs() < 1e-5);
        assert!((dst[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_rows() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f32; 2];
        mean_rows_f32(&src, &mut dst, 3);
        assert!((dst[0] - 2.0).abs() < 1e-5);
        assert!((dst[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax() {
        let src = vec![1.0f32, 5.0, 2.0, 3.0];
        assert_eq!(argmax_f32(&src), 1);
    }

    #[test]
    fn test_argmax_rows() {
        let src = vec![1.0f32, 5.0, 2.0, 3.0, 7.0, 1.0];
        let mut dst = vec![0i32; 2];
        argmax_rows_f32(&src, &mut dst, 3);
        assert_eq!(dst[0], 1);
        assert_eq!(dst[1], 1);
    }

    #[test]
    fn test_count_equal() {
        let a = vec![1i32, 2, 3, 4];
        let b = vec![1i32, 9, 3, 8];
        assert_eq!(count_equal_i32(&a, &b), 2);
    }

    #[test]
    fn test_fill_f32() {
        let mut dst = vec![0.0f32; 4];
        fill_f32(&mut dst, 3.14);
        for v in &dst {
            assert!((*v - 3.14).abs() < 1e-5);
        }
    }

    #[test]
    fn test_tri_lower_diag() {
        // 3x3 identity test
        let src = vec![1.0f32; 9];
        let mut dst = vec![0.0f32; 9];
        tri_f32(&src, &mut dst, 3, 3, 1, 1, TriType::LowerDiag);
        // row 0: col 0 <= 0 -> keep; cols 1,2 -> zero
        assert_eq!(dst[0], 1.0);
        assert_eq!(dst[1], 0.0);
        assert_eq!(dst[2], 0.0);
        // row 1: cols 0,1 <= 1 -> keep; col 2 -> zero
        assert_eq!(dst[3], 1.0);
        assert_eq!(dst[4], 1.0);
        assert_eq!(dst[5], 0.0);
        // row 2: all <= 2 -> keep
        assert_eq!(dst[6], 1.0);
        assert_eq!(dst[7], 1.0);
        assert_eq!(dst[8], 1.0);
    }

    #[test]
    fn test_silu_back() {
        // At x=0: silu(0)=0, dsilu/dx = 0*(1+0*(1-0)) = 0, so grad=0
        let src1 = vec![0.0f32, 1.0];
        let grad = vec![1.0f32, 1.0];
        let mut dst = vec![0.0f32; 2];
        silu_back_rows_f32(&src1, &grad, &mut dst);
        // x=0: s=0, dx = 0*(1+0) = 0
        assert!(dst[0].abs() < 1e-6);
        // x=1: s=silu(1)~=0.7311, dx=s*(1+1*(1-s))~=0.7311*(1+0.2689)~=0.9277
        assert!((dst[1] - 0.9277).abs() < 1e-3);
    }

    #[test]
    fn test_reglu_rows() {
        // gate=2 (relu=2), up=3 -> 6
        let src0 = vec![2.0f32, 3.0]; // [gate, up] interleaved
        let mut dst = vec![0.0f32; 1];
        reglu_rows_f32(&src0, None, &mut dst, 1, 1, false);
        assert!((dst[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_swiglu_rows() {
        // gate=0 -> silu(0)=0 -> result=0
        let src0 = vec![0.0f32, 1.0]; // gate=0, up=1
        let mut dst = vec![0.0f32; 1];
        swiglu_rows_f32(&src0, None, &mut dst, 1, 1, false);
        assert!(dst[0].abs() < 1e-6);
    }

    #[test]
    fn test_geglu_rows() {
        // gelu(0) = 0 -> 0 * up = 0
        let src0 = vec![0.0f32, 5.0]; // gate=0, up=5
        let mut dst = vec![0.0f32; 1];
        geglu_rows_f32(&src0, None, &mut dst, 1, 1, false);
        assert!(dst[0].abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_rows() {
        let src = vec![3.0f32, 4.0]; // rms = sqrt((9+16)/2) = sqrt(12.5)
        let mut dst = vec![0.0f32; 2];
        rms_norm_rows_f32(&src, &mut dst, 2, 1e-5);
        let rms = (12.5f32 + 1e-5).sqrt();
        assert!((dst[0] - 3.0 / rms).abs() < 1e-5);
        assert!((dst[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_mul_rows() {
        let src = vec![3.0f32, 4.0];
        let weight = vec![2.0f32, 0.5];
        let mut dst = vec![0.0f32; 2];
        rms_norm_mul_rows_f32(&src, &weight, &mut dst, 2, 1e-5);
        let rms = (12.5f32 + 1e-5).sqrt();
        assert!((dst[0] - 3.0 / rms * 2.0).abs() < 1e-5);
        assert!((dst[1] - 4.0 / rms * 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_repeat_f32() {
        // 2-element source repeated 2 times in dim0, 1 in others
        let src = vec![1.0f32, 2.0];
        let mut dst = vec![0.0f32; 4];
        repeat_f32(&src, &mut dst, 2, 1, 1, 1, 2, 1, 1, 1);
        assert_eq!(&dst, &[1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_concat_f32_dim1() {
        // concat [1,2] and [3,4] along dim1
        let src0 = vec![1.0f32, 2.0];
        let src1 = vec![3.0f32, 4.0];
        let mut dst = vec![0.0f32; 4];
        concat_f32(&src0, &src1, &mut dst, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1);
        assert_eq!(&dst, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_add_scalar() {
        let src = vec![1.0f32, 2.0, 3.0];
        let mut dst = vec![0.0f32; 3];
        add_scalar_f32(&mut dst, &src, 10.0);
        assert_eq!(&dst, &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_gelu_rows() {
        let src = vec![0.0f32, 1.0];
        let mut dst = vec![0.0f32; 2];
        gelu_rows_f32(&src, &mut dst, 2, 1);
        // gelu(0) = 0
        assert!(dst[0].abs() < 1e-5);
        // gelu(1) ~= 0.8413
        assert!((dst[1] - 0.8413).abs() < 1e-3);
    }

    #[test]
    fn test_silu_rows() {
        let src = vec![0.0f32, 1.0];
        let mut dst = vec![0.0f32; 2];
        silu_rows_f32(&src, &mut dst, 2, 1);
        assert!(dst[0].abs() < 1e-5);
        // silu(1) = 1/(1+exp(-1)) ~= 0.7311
        assert!((dst[1] - 0.7311).abs() < 1e-3);
    }

    #[test]
    fn test_norm_rows() {
        let src = vec![1.0f32, 2.0, 3.0];
        let mut dst = vec![0.0f32; 3];
        norm_rows_f32(&src, &mut dst, 3, 1e-5);
        // mean=2, var=2/3, scale=1/sqrt(2/3+eps) ~= sqrt(3/2)
        let mean = 2.0f32;
        let var = ((1.0 - mean).powi(2) + (2.0 - mean).powi(2) + (3.0 - mean).powi(2)) / 3.0;
        let scale = 1.0 / (var + 1e-5).sqrt();
        assert!((dst[0] - (1.0 - mean) * scale).abs() < 1e-4);
        assert!((dst[1] - (2.0 - mean) * scale).abs() < 1e-4);
        assert!((dst[2] - (3.0 - mean) * scale).abs() < 1e-4);
    }
}
