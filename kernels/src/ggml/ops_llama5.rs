//! Direct Rust translations of llama.cpp ops.cpp functions not yet covered
//! by ops_llama.rs through ops_llama4.rs.
//!
//! Sources: ggml/src/ggml-cpu/ops.cpp

// --- sum_f16 / sum_bf16 ------------------------------------------------------
// ops.cpp lines 1316-1380

/// Translation of `ggml_compute_forward_sum_f16`.
/// Sums all f16 elements across a 4-D tensor (thread 0 only).
pub fn sum_f16(
    src: &[u16], // f16 bits, row-major
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    // strides in bytes; nb00 == 2 is assumed
    nb01: usize, nb02: usize, nb03: usize,
    ith: usize,
) -> f32 {
    if ith != 0 { return 0.0; }
    let mut sum = 0.0f32;
    for i03 in 0..ne03 {
        for i02 in 0..ne02 {
            for i01 in 0..ne01 {
                let base = (i01 * nb01 + i02 * nb02 + i03 * nb03) / 2;
                for i00 in 0..ne00 {
                    sum += half::f16::from_bits(src[base + i00]).to_f32();
                }
            }
        }
    }
    sum
}

/// Translation of `ggml_compute_forward_sum_bf16`.
/// Same as sum_f16 but for bf16 (bfloat16) represented as u16 bits.
pub fn sum_bf16(
    src: &[u16], // bf16 bits, row-major
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    nb01: usize, nb02: usize, nb03: usize,
    ith: usize,
) -> f32 {
    if ith != 0 { return 0.0; }
    let mut sum = 0.0f32;
    for i03 in 0..ne03 {
        for i02 in 0..ne02 {
            for i01 in 0..ne01 {
                let base = (i01 * nb01 + i02 * nb02 + i03 * nb03) / 2;
                for i00 in 0..ne00 {
                    // bf16 -> f32: shift bits left 16
                    let bits = (src[base + i00] as u32) << 16;
                    sum += f32::from_bits(bits);
                }
            }
        }
    }
    sum
}

// --- repeat_f16 --------------------------------------------------------------
// ops.cpp lines 1742-1787

/// Translation of `ggml_compute_forward_repeat_f16`.
/// Tiles src (ne00xne01xne02xne03) into dst (ne0xne1xne2xne3).
/// Both are contiguous with element size 2 bytes.
/// Thread 0 only; not parallelised in the original either.
pub fn repeat_f16(
    src: &[u16],    // f16/bf16/i16 bits -- treated as opaque 16-bit
    dst: &mut [u16],
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    ne0:  usize, ne1:  usize, ne2:  usize, ne3:  usize,
    ith: usize,
) {
    if ith != 0 { return; }
    let nr0 = ne0 / ne00;
    let nr1 = ne1 / ne01;
    let nr2 = ne2 / ne02;
    let nr3 = ne3 / ne03;
    for i3 in 0..nr3 {
        for k3 in 0..ne03 {
            for i2 in 0..nr2 {
                for k2 in 0..ne02 {
                    for i1 in 0..nr1 {
                        for k1 in 0..ne01 {
                            for i0 in 0..nr0 {
                                let dst_base = (i3*ne03 + k3)*ne2*ne1*ne0
                                    + (i2*ne02 + k2)*ne1*ne0
                                    + (i1*ne01 + k1)*ne0
                                    + i0*ne00;
                                let src_base = k3*ne02*ne01*ne00 + k2*ne01*ne00 + k1*ne00;
                                dst[dst_base..dst_base+ne00]
                                    .copy_from_slice(&src[src_base..src_base+ne00]);
                            }
                        }
                    }
                }
            }
        }
    }
}

// --- concat_f16 / concat_any -------------------------------------------------
// ops.cpp lines 1899-2035

/// Translation of `ggml_compute_forward_concat_f16`.
/// Concatenates src0 and src1 along `dim` into dst (element size 2 bytes).
/// Parallelised over i2 (dim-2 axis).
pub fn concat_f16(
    src0: &[u16], src1: &[u16], dst: &mut [u16],
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    ne10: usize, ne11: usize, ne12: usize, ne13: usize,
    ne0:  usize, ne1:  usize, ne2:  usize, ne3:  usize,
    dim: usize,
    ith: usize, nth: usize,
) {
    assert!(dim < 4);
    let mut o = [0usize; 4];
    o[dim] = match dim { 0 => ne00, 1 => ne01, 2 => ne02, _ => ne03 };

    // Copy from src0
    for i3 in 0..ne03 {
        let mut i2 = ith;
        while i2 < ne02 {
            for i1 in 0..ne01 {
                for i0 in 0..ne00 {
                    let si = i3*ne02*ne01*ne00 + i2*ne01*ne00 + i1*ne00 + i0;
                    let di = i3*ne3*ne2*ne1*ne0 / ne3 + i2*ne1*ne0 + i1*ne0 + i0;
                    // simple contiguous index: dst[i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0 + i0]
                    let di = i3*(ne2*ne1*ne0) + i2*(ne1*ne0) + i1*ne0 + i0;
                    dst[di] = src0[si];
                }
            }
            i2 += nth;
        }
    }

    // Copy from src1, offset by o[]
    for i3 in 0..ne13 {
        let mut i2 = ith;
        while i2 < ne12 {
            for i1 in 0..ne11 {
                for i0 in 0..ne10 {
                    let si = i3*(ne12*ne11*ne10) + i2*(ne11*ne10) + i1*ne10 + i0;
                    let di = (i3 + o[3])*(ne2*ne1*ne0)
                        + (i2 + o[2])*(ne1*ne0)
                        + (i1 + o[1])*ne0
                        + (i0 + o[0]);
                    dst[di] = src1[si];
                }
            }
            i2 += nth;
        }
    }
    let _ = (ne3, ne1, ne0);
}

// --- set_f32 / set_i32 -------------------------------------------------------
// ops.cpp lines 4769-4909

/// Translation of `ggml_compute_forward_set_f32`.
/// Copies rows from src1 into dst starting at byte offset, with given strides.
/// dst has been pre-memcpy'd from src0 by thread 0 before this call.
pub fn set_f32(
    src1: &[f32],
    dst: &mut [f32],
    ne10: usize, ne11: usize, ne12: usize, ne13: usize,
    // strides for src1 in bytes (nb10 == 4 assumed)
    nb11: usize, nb12: usize, nb13: usize,
    // strides for dst view in bytes (nb0 == 4 assumed)
    dst_nb1: usize, dst_nb2: usize, dst_nb3: usize,
    offset_bytes: usize,
    ith: usize, nth: usize,
) {
    let nr = ne13 * ne12 * ne11;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        let i3 = ir / (ne12 * ne11);
        let i2 = (ir - i3 * ne12 * ne11) / ne11;
        let i1 = ir - i3 * ne12 * ne11 - i2 * ne11;
        // src1 offset in f32 elements
        let src_off = (i3 * nb13 + i2 * nb12 + i1 * nb11) / 4;
        // dst offset in f32 elements (offset_bytes + view strides)
        let dst_off = (offset_bytes + i3 * dst_nb3 + i2 * dst_nb2 + i1 * dst_nb1) / 4;
        dst[dst_off..dst_off + ne10].copy_from_slice(&src1[src_off..src_off + ne10]);
    }
}

/// Translation of `ggml_compute_forward_set_i32`.
/// Same as set_f32 but for i32 elements (nb10 == 4 assumed).
pub fn set_i32(
    src1: &[i32],
    dst: &mut [i32],
    ne10: usize, ne11: usize, ne12: usize, ne13: usize,
    nb11: usize, nb12: usize, nb13: usize,
    dst_nb1: usize, dst_nb2: usize, dst_nb3: usize,
    offset_bytes: usize,
    ith: usize, nth: usize,
) {
    let nr = ne13 * ne12 * ne11;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for ir in ir0..ir1 {
        let i3 = ir / (ne12 * ne11);
        let i2 = (ir - i3 * ne12 * ne11) / ne11;
        let i1 = ir - i3 * ne12 * ne11 - i2 * ne11;
        let src_off = (i3 * nb13 + i2 * nb12 + i1 * nb11) / 4;
        let dst_off = (offset_bytes + i3 * dst_nb3 + i2 * dst_nb2 + i1 * dst_nb1) / 4;
        dst[dst_off..dst_off + ne10].copy_from_slice(&src1[src_off..src_off + ne10]);
    }
}

// --- get_rows_f16 / get_rows_bf16 / get_rows_q -------------------------------
// ops.cpp lines 4979-5120

/// Translation of `ggml_compute_forward_get_rows_f16`.
/// For each index in src1, copies the corresponding f16 row from src0 -> dst as f32.
pub fn get_rows_f16(
    src0: &[u16], // f16 weights [ne01 x ne00] (+batch dims)
    src1: &[i32], // row indices [ne10 x ne11 x ne12]
    dst: &mut [f32],
    ne00: usize, ne01: usize,
    ne10: usize, ne11: usize, ne12: usize,
    // strides for src0 in bytes (nb00==2 assumed)
    nb01: usize, nb02: usize, nb03: usize,
    // strides for dst in bytes (nb0==4 assumed)
    nb1: usize, nb2: usize, nb3: usize,
    ith: usize, nth: usize,
) {
    let nr = ne12 * ne11 * ne10;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for i in ir0..ir1 {
        let i12 = i / (ne11 * ne10);
        let i11 = (i - i12 * ne11 * ne10) / ne10;
        let i10 = i - i12 * ne11 * ne10 - i11 * ne10;
        let i01 = src1[i12 * ne11 * ne10 + i11 * ne10 + i10] as usize;
        assert!(i01 < ne01);
        let src_off = (i01 * nb01 + i11 * nb02 + i12 * nb03) / 2;
        let dst_off = (i10 * nb1 + i11 * nb2 + i12 * nb3) / 4;
        for i00 in 0..ne00 {
            dst[dst_off + i00] = half::f16::from_bits(src0[src_off + i00]).to_f32();
        }
    }
    let _ = ne00;
}

/// Translation of `ggml_compute_forward_get_rows_bf16`.
/// Same as get_rows_f16 but for bf16 source.
pub fn get_rows_bf16(
    src0: &[u16], // bf16 weights
    src1: &[i32],
    dst: &mut [f32],
    ne00: usize, ne01: usize,
    ne10: usize, ne11: usize, ne12: usize,
    nb01: usize, nb02: usize, nb03: usize,
    nb1: usize, nb2: usize, nb3: usize,
    ith: usize, nth: usize,
) {
    let nr = ne12 * ne11 * ne10;
    let dr = (nr + nth - 1) / nth;
    let ir0 = dr * ith;
    let ir1 = (ir0 + dr).min(nr);
    for i in ir0..ir1 {
        let i12 = i / (ne11 * ne10);
        let i11 = (i - i12 * ne11 * ne10) / ne10;
        let i10 = i - i12 * ne11 * ne10 - i11 * ne10;
        let i01 = src1[i12 * ne11 * ne10 + i11 * ne10 + i10] as usize;
        assert!(i01 < ne01);
        let src_off = (i01 * nb01 + i11 * nb02 + i12 * nb03) / 2;
        let dst_off = (i10 * nb1 + i11 * nb2 + i12 * nb3) / 4;
        for i00 in 0..ne00 {
            let bits = (src0[src_off + i00] as u32) << 16;
            dst[dst_off + i00] = f32::from_bits(bits);
        }
    }
    let _ = ne00;
}

// --- pad_f32 -----------------------------------------------------------------
// ops.cpp lines 8196-8258 (non-circular variant)

/// Translation of `ggml_compute_forward_pad_f32<false>` (zero-padding).
/// Pads src with zeros: lp0/rp0 on dim 0, lp1/rp1 on dim 1, etc.
pub fn pad_f32(
    src: &[f32],
    dst: &mut [f32],
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    lp0: i64, _rp0: i64,
    lp1: i64, _rp1: i64,
    lp2: i64, _rp2: i64,
    lp3: i64, _rp3: i64,
    ith: usize, nth: usize,
) {
    for i2 in 0..ne2 {
        let mut i1 = ith;
        while i1 < ne1 {
            for i0 in 0..ne0 {
                for i3 in 0..ne3 {
                    let dst_idx = i3 * (ne0 * ne1 * ne2) + i2 * (ne0 * ne1) + i1 * ne0 + i0;
                    let s0 = i0 as i64 - lp0;
                    let s1 = i1 as i64 - lp1;
                    let s2 = i2 as i64 - lp2;
                    let s3 = i3 as i64 - lp3;
                    if s0 >= 0 && s0 < ne00 as i64
                        && s1 >= 0 && s1 < ne01 as i64
                        && s2 >= 0 && s2 < ne02 as i64
                        && s3 >= 0 && s3 < ne03 as i64
                    {
                        let src_idx = s3 as usize * (ne02 * ne01 * ne00)
                            + s2 as usize * (ne01 * ne00)
                            + s1 as usize * ne00
                            + s0 as usize;
                        dst[dst_idx] = src[src_idx];
                    } else {
                        dst[dst_idx] = 0.0;
                    }
                }
            }
            i1 += nth;
        }
    }
}

/// Translation of `ggml_compute_forward_pad_f32<true>` (circular/wrap padding).
pub fn pad_circular_f32(
    src: &[f32],
    dst: &mut [f32],
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    lp0: i64, _rp0: i64,
    lp1: i64, _rp1: i64,
    lp2: i64, _rp2: i64,
    lp3: i64, _rp3: i64,
    ith: usize, nth: usize,
) {
    fn wrap(i: i64, n: usize) -> usize {
        let n = n as i64;
        (((i % n) + n) % n) as usize
    }
    for i2 in 0..ne2 {
        let mut i1 = ith;
        while i1 < ne1 {
            for i0 in 0..ne0 {
                for i3 in 0..ne3 {
                    let dst_idx = i3 * (ne0 * ne1 * ne2) + i2 * (ne0 * ne1) + i1 * ne0 + i0;
                    let s0 = wrap(i0 as i64 - lp0, ne00);
                    let s1 = wrap(i1 as i64 - lp1, ne01);
                    let s2 = wrap(i2 as i64 - lp2, ne02);
                    let s3 = wrap(i3 as i64 - lp3, ne03);
                    let src_idx = s3*(ne02*ne01*ne00) + s2*(ne01*ne00) + s1*ne00 + s0;
                    dst[dst_idx] = src[src_idx];
                }
            }
            i1 += nth;
        }
    }
}

// --- roll_f32 ----------------------------------------------------------------
// ops.cpp lines 8328-8363

/// Translation of `ggml_compute_forward_roll_f32`.
/// Cyclically shifts src along each axis by s0/s1/s2/s3.
pub fn roll_f32(
    src: &[f32],
    dst: &mut [f32],
    ne00: usize, ne01: usize, ne02: usize, ne03: usize,
    // strides in bytes for src (assumed contiguous: nb00=4, nb01=4*ne00, ...)
    // For simplicity we assume contiguous layout matching the strides.
    ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    s0: i64, s1: i64, s2: i64, s3: i64,
    ith: usize, nth: usize,
) {
    fn wrap_idx(i: i64, n: usize) -> usize {
        let n = n as i64;
        (((i % n) + n) % n) as usize
    }
    let total = (ne1 * ne2 * ne3) as i64;
    let per_thread = (total + nth as i64) / nth as i64;
    let start = ith as i64 * per_thread;
    let end = (start + per_thread).min(total);
    for idx in start..end {
        let i1 = (idx % ne1 as i64) as usize;
        let i2 = ((idx / ne1 as i64) % ne2 as i64) as usize;
        let i3 = (idx / (ne2 * ne1) as i64) as usize;
        let dst_row_off = i3 * (ne2 * ne1 * ne0) + i2 * (ne1 * ne0) + i1 * ne0;
        let i01 = wrap_idx(i1 as i64 - s1, ne01);
        let i02 = wrap_idx(i2 as i64 - s2, ne02);
        let i03 = wrap_idx(i3 as i64 - s3, ne03);
        let src_row_off = i03 * (ne02 * ne01 * ne00) + i02 * (ne01 * ne00) + i01 * ne00;
        let s_wrap = wrap_idx(-s0, ne00);
        let n = ne00 - s_wrap;
        // copy src[s_wrap..] to dst[0..n]
        dst[dst_row_off..dst_row_off + n]
            .copy_from_slice(&src[src_row_off + s_wrap..src_row_off + s_wrap + n]);
        // copy src[0..s_wrap] to dst[n..]
        dst[dst_row_off + n..dst_row_off + ne00]
            .copy_from_slice(&src[src_row_off..src_row_off + s_wrap]);
    }
    let _ = (ne0, ne3, s3);
}

// --- argsort_f32 -------------------------------------------------------------
// ops.cpp lines 8489-8545

/// Translation of `ggml_compute_forward_argsort_f32`.
/// For each row of src, fills dst with indices sorted by src values.
/// ascending=true for ASC order, false for DESC.
pub fn argsort_f32(
    src: &[f32],
    dst: &mut [i32],
    ne0: usize, ne01: usize, // ne0=output cols, ne01=src cols (must equal ne0)
    nr: usize,  // number of rows
    ascending: bool,
    ith: usize, nth: usize,
) {
    assert_eq!(ne0, ne01);
    let mut i = ith;
    while i < nr {
        let src_row = &src[i * ne01..(i + 1) * ne01];
        let dst_row = &mut dst[i * ne0..(i + 1) * ne0];
        for j in 0..ne0 {
            dst_row[j] = j as i32;
        }
        if ascending {
            dst_row.sort_by(|&a, &b| {
                src_row[a as usize]
                    .partial_cmp(&src_row[b as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            dst_row.sort_by(|&a, &b| {
                src_row[b as usize]
                    .partial_cmp(&src_row[a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        i += nth;
    }
}

// --- top_k_f32 ---------------------------------------------------------------
// ops.cpp lines 8557-8594

/// Translation of `ggml_compute_forward_top_k_f32`.
/// For each row of src (ne00 elements), writes the top_k indices (by value) to dst.
/// The first two output indices may be swapped (matching the original's behavior).
pub fn top_k_f32(
    src: &[f32],
    dst: &mut [i32],
    ne00: usize, // source row length
    top_k: usize, // output row length = top_k
    nr: usize,   // number of rows
    ith: usize, nth: usize,
) {
    let mut tmp = vec![0i32; ne00];
    let mut i = ith;
    while i < nr {
        let src_row = &src[i * ne00..(i + 1) * ne00];
        for j in 0..ne00 { tmp[j] = j as i32; }
        // partial sort: top_k largest first (desc by src value)
        tmp[..ne00].sort_by(|&a, &b| {
            src_row[b as usize]
                .partial_cmp(&src_row[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let dst_row = &mut dst[i * top_k..(i + 1) * top_k];
        dst_row.copy_from_slice(&tmp[..top_k]);
        // swap first two if top_k > 1 (matches original)
        if top_k > 1 {
            dst_row.swap(0, 1);
        }
        i += nth;
    }
}

// --- get_rel_pos_f16 ---------------------------------------------------------
// ops.cpp lines 10285-10308

/// Translation of `ggml_compute_forward_get_rel_pos_f16`.
/// Used by SAM image encoder for relative positional embeddings.
/// src0: [ne00, ne_src0] -- the position embedding table
/// dst:  [ne0, ne1, ne2] -- relative position output
pub fn get_rel_pos_f16(
    src0: &[u16], // f16 bits
    dst: &mut [u16],
    ne00: usize,
    ne0: usize, ne1: usize, ne2: usize,
) {
    let w = ne1;
    for i2 in 0..ne2 {
        for i1 in 0..ne1 {
            let pos = (w - i1 - 1) + i2;
            for i0 in 0..ne0 {
                dst[i2 * ne1 * ne0 + i1 * ne0 + i0] = src0[pos * ne00 + i0];
            }
        }
    }
}

// --- pad_reflect_1d_f32 ------------------------------------------------------
// ops.cpp lines 8284-8315

/// Translation of `ggml_compute_forward_pad_reflect_1d`.
/// Copies src row into center of dst row, reflects p0 samples left and p1 right.
pub fn pad_reflect_1d_f32(
    src: &[f32],
    dst: &mut [f32],
    ne00: usize, // source row length
    ne0: usize, ne1: usize, ne2: usize, ne3: usize,
    p0: usize, p1: usize,
    ith: usize, nth: usize,
) {
    for i3 in 0..ne3 {
        for i2 in 0..ne2 {
            let mut i1 = ith;
            while i1 < ne1 {
                let dst_row_off = i3 * (ne2 * ne1 * ne0) + i2 * (ne1 * ne0) + i1 * ne0;
                let src_row_off = i3 * (ne2 * ne1 * ne00) + i2 * (ne1 * ne00) + i1 * ne00;
                // copy src into center
                dst[dst_row_off + p0..dst_row_off + p0 + ne00]
                    .copy_from_slice(&src[src_row_off..src_row_off + ne00]);
                // reflect left
                for i in 1..=p0 {
                    dst[dst_row_off + p0 - i] = dst[dst_row_off + p0 + i];
                }
                // reflect right
                for i in 1..=p1 {
                    dst[dst_row_off + p0 + ne00 - 1 + i] = dst[dst_row_off + p0 + ne00 - 1 - i];
                }
                i1 += nth;
            }
        }
    }
    let _ = ne0;
}
