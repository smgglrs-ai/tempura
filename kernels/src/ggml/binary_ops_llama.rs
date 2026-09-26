// Rust translation of llama.cpp ggml/src/ggml-cpu/binary-ops.cpp
// Element-wise binary ops (add, sub, mul, div) with broadcasting.

#[inline(always)]
fn op_add(a: f32, b: f32) -> f32 {
    a + b
}

#[inline(always)]
fn op_sub(a: f32, b: f32) -> f32 {
    a - b
}

#[inline(always)]
fn op_mul(a: f32, b: f32) -> f32 {
    a * b
}

#[inline(always)]
fn op_div(a: f32, b: f32) -> f32 {
    a / b
}

// vec_binary_op_contiguous<op, float, float, float>
// Both src0 and src1 are contiguous; n elements.
#[inline(always)]
pub fn vec_binary_op_contiguous(
    n: i64,
    z: &mut [f32],
    x: &[f32],
    y: &[f32],
    op: fn(f32, f32) -> f32,
) {
    for i in 0..n as usize {
        z[i] = op(x[i], y[i]);
    }
}

// vec_binary_op_non_contiguous<op, float, float, float>
// src1 is accessed with byte stride nb10, modulo ne10.
// `y` is the raw base pointer of src1 (byte-addressed).
pub unsafe fn vec_binary_op_non_contiguous(
    n: i64,
    ne10: i64,
    nb10: usize,
    z: &mut [f32],
    x: &[f32],
    y: *const u8,
    op: fn(f32, f32) -> f32,
) {
    for i in 0..n as usize {
        let i10 = (i as i64 % ne10) as usize;
        let y_ptr = y.add(i10 * nb10) as *const f32;
        z[i] = op(x[i], *y_ptr);
    }
}

// apply_binary_op<op, float, float, float>
// Full tensor-geometry version with 4-D strides and thread row range.
//
// Parameters mirror GGML_TENSOR_BINARY_OP_LOCALS:
//   ne0..ne2   — dst / src0 shape (same shape asserted by caller)
//   nb1..nb3   — dst byte strides (nb0 == sizeof(f32) assumed)
//   nb01..nb03 — src0 byte strides
//   ne10..ne13 — src1 shape (may be smaller: broadcasting)
//   nb10..nb13 — src1 byte strides
//   is_src1_contiguous_rows — ggml_is_contiguous_rows(src1)
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_binary_op_f32(
    op: fn(f32, f32) -> f32,
    ir0: i64,
    ir1: i64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    dst: *mut u8,
    nb01: usize,
    nb02: usize,
    nb03: usize,
    src0: *const u8,
    ne10: i64,
    ne11: i64,
    ne12: i64,
    ne13: i64,
    nb10: usize,
    nb11: usize,
    nb12: usize,
    nb13: usize,
    src1: *const u8,
    is_src1_contiguous_rows: bool,
) {
    let _ = (nb11, nb12); // only nb10 and nb13/nb12/nb11 used for row ptr, nb10 for element stride
    for ir in ir0..ir1 {
        let i03 = ir / (ne2 * ne1);
        let i02 = (ir - i03 * ne2 * ne1) / ne1;
        let i01 = ir - i03 * ne2 * ne1 - i02 * ne1;

        let i13 = i03 % ne13;
        let i12 = i02 % ne12;
        let i11 = i01 % ne11;

        let dst_row =
            dst.add(i03 as usize * nb3 + i02 as usize * nb2 + i01 as usize * nb1) as *mut f32;
        let src0_row =
            src0.add(i03 as usize * nb03 + i02 as usize * nb02 + i01 as usize * nb01)
                as *const f32;
        let src1_row =
            src1.add(i13 as usize * nb13 + i12 as usize * nb12 + i11 as usize * nb11);

        if is_src1_contiguous_rows {
            // src1 is broadcastable: tile src1 row across src0 row
            let nr0 = (ne0 / ne10) as usize;
            let src1_ptr = src1_row as *const f32;
            for r in 0..nr0 {
                let offset = r * ne10 as usize;
                let z = std::slice::from_raw_parts_mut(dst_row.add(offset), ne10 as usize);
                let x = std::slice::from_raw_parts(src0_row.add(offset), ne10 as usize);
                let y = std::slice::from_raw_parts(src1_ptr, ne10 as usize);
                vec_binary_op_contiguous(ne10, z, x, y, op);
            }
        } else {
            let z = std::slice::from_raw_parts_mut(dst_row, ne0 as usize);
            let x = std::slice::from_raw_parts(src0_row, ne0 as usize);
            vec_binary_op_non_contiguous(ne0, ne10, nb10, z, x, src1_row, op);
        }
    }
}

// ── Public entry points (f32 contiguous slices, single-threaded) ─────────────
// These correspond to ggml_compute_forward_{add,sub,mul,div} for the common
// all-f32 contiguous case.

pub fn compute_forward_add(n: usize, dst: &mut [f32], src0: &[f32], src1: &[f32]) {
    vec_binary_op_contiguous(n as i64, dst, src0, src1, op_add);
}

pub fn compute_forward_sub(n: usize, dst: &mut [f32], src0: &[f32], src1: &[f32]) {
    vec_binary_op_contiguous(n as i64, dst, src0, src1, op_sub);
}

pub fn compute_forward_mul(n: usize, dst: &mut [f32], src0: &[f32], src1: &[f32]) {
    vec_binary_op_contiguous(n as i64, dst, src0, src1, op_mul);
}

pub fn compute_forward_div(n: usize, dst: &mut [f32], src0: &[f32], src1: &[f32]) {
    vec_binary_op_contiguous(n as i64, dst, src0, src1, op_div);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_scalar() {
        assert_eq!(op_add(1.0, 2.0), 3.0);
        assert_eq!(op_add(-1.0, 1.0), 0.0);
    }

    #[test]
    fn test_sub_scalar() {
        assert_eq!(op_sub(5.0, 3.0), 2.0);
        assert_eq!(op_sub(0.0, 1.0), -1.0);
    }

    #[test]
    fn test_mul_scalar() {
        assert_eq!(op_mul(3.0, 4.0), 12.0);
        assert_eq!(op_mul(-2.0, 3.0), -6.0);
    }

    #[test]
    fn test_div_scalar() {
        assert_eq!(op_div(10.0, 2.0), 5.0);
        assert_eq!(op_div(1.0, 4.0), 0.25);
    }

    #[test]
    fn test_compute_forward_add() {
        let src0 = [1.0f32, 2.0, 3.0, 4.0];
        let src1 = [10.0f32, 20.0, 30.0, 40.0];
        let mut dst = [0.0f32; 4];
        compute_forward_add(4, &mut dst, &src0, &src1);
        assert_eq!(dst, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_compute_forward_sub() {
        let src0 = [10.0f32, 20.0, 30.0, 40.0];
        let src1 = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst = [0.0f32; 4];
        compute_forward_sub(4, &mut dst, &src0, &src1);
        assert_eq!(dst, [9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn test_compute_forward_mul() {
        let src0 = [1.0f32, 2.0, 3.0, 4.0];
        let src1 = [2.0f32, 3.0, 4.0, 5.0];
        let mut dst = [0.0f32; 4];
        compute_forward_mul(4, &mut dst, &src0, &src1);
        assert_eq!(dst, [2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_compute_forward_div() {
        let src0 = [4.0f32, 9.0, 16.0, 25.0];
        let src1 = [2.0f32, 3.0, 4.0, 5.0];
        let mut dst = [0.0f32; 4];
        compute_forward_div(4, &mut dst, &src0, &src1);
        assert_eq!(dst, [2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_vec_binary_op_contiguous_vs_scalar() {
        let n = 8usize;
        let src0: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let src1: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let mut dst_vec = vec![0.0f32; n];
        let mut dst_scalar = vec![0.0f32; n];

        vec_binary_op_contiguous(n as i64, &mut dst_vec, &src0, &src1, op_add);
        for i in 0..n {
            dst_scalar[i] = op_add(src0[i], src1[i]);
        }
        assert_eq!(dst_vec, dst_scalar);

        vec_binary_op_contiguous(n as i64, &mut dst_vec, &src0, &src1, op_mul);
        for i in 0..n {
            dst_scalar[i] = op_mul(src0[i], src1[i]);
        }
        assert_eq!(dst_vec, dst_scalar);
    }

    #[test]
    fn test_vec_binary_op_non_contiguous_broadcast() {
        // src1 has 2 elements, src0 has 4 — non-contiguous with modular access
        let src0 = [1.0f32, 2.0, 3.0, 4.0];
        let src1 = [10.0f32, 20.0];
        let mut dst = [0.0f32; 4];
        // ne10=2, nb10=sizeof(f32)=4
        unsafe {
            vec_binary_op_non_contiguous(
                4,
                2,
                std::mem::size_of::<f32>(),
                &mut dst,
                &src0,
                src1.as_ptr() as *const u8,
                op_add,
            );
        }
        // i=0: i10=0 → src1[0]=10 → 1+10=11
        // i=1: i10=1 → src1[1]=20 → 2+20=22
        // i=2: i10=0 → src1[0]=10 → 3+10=13
        // i=3: i10=1 → src1[1]=20 → 4+20=24
        assert_eq!(dst, [11.0, 22.0, 13.0, 24.0]);
    }
}
