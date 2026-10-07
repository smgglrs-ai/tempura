// Rust translation of llama.cpp ggml/src/ggml-cpu/unary-ops.cpp
// Element-wise unary ops: neg, abs, sgn, step, tanh, elu, relu, sigmoid,
// hardsigmoid, exp, hardswish, sqr, sqrt, sin, cos, log, expm1, softplus,
// floor, ceil, round, trunc, xielu.

// ── Scalar ops ───────────────────────────────────────────────────────────────

#[inline(always)]
fn op_abs(x: f32) -> f32 {
    x.abs()
}

#[inline(always)]
fn op_sgn(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[inline(always)]
fn op_neg(x: f32) -> f32 {
    -x
}

#[inline(always)]
fn op_step(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

#[inline(always)]
fn op_tanh(x: f32) -> f32 {
    x.tanh()
}

#[inline(always)]
fn op_elu(x: f32) -> f32 {
    if x > 0.0 { x } else { x.exp_m1() }
}

#[inline(always)]
fn op_relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[inline(always)]
fn op_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn op_hardsigmoid(x: f32) -> f32 {
    ((x + 3.0) / 6.0).clamp(0.0, 1.0)
}

#[inline(always)]
fn op_exp(x: f32) -> f32 {
    x.exp()
}

#[inline(always)]
fn op_hardswish(x: f32) -> f32 {
    x * ((x + 3.0) / 6.0).clamp(0.0, 1.0)
}

#[inline(always)]
fn op_sqr(x: f32) -> f32 {
    x * x
}

#[inline(always)]
fn op_sqrt(x: f32) -> f32 {
    x.sqrt()
}

// xielu: parameterised exponential linear unit with quadratic positive branch
// alpha_n, alpha_p: negative/positive curvature; beta: linear slope; eps: clamp
#[inline(always)]
fn op_xielu(x: f32, alpha_n: f32, alpha_p: f32, beta: f32, eps: f32) -> f32 {
    if x > 0.0 {
        alpha_p * x * x + beta * x
    } else {
        let min_x_eps = x.min(eps);
        (min_x_eps.exp_m1() - x) * alpha_n + beta * x
    }
}

#[inline(always)]
fn op_sin(x: f32) -> f32 {
    x.sin()
}

#[inline(always)]
fn op_cos(x: f32) -> f32 {
    x.cos()
}

#[inline(always)]
fn op_log(x: f32) -> f32 {
    x.ln()
}

// Note: llama.cpp implements op_expm1 as expf(x)-1.0, not expm1f(x).
#[inline(always)]
fn op_expm1(x: f32) -> f32 {
    x.exp() - 1.0
}

#[inline(always)]
fn op_softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

#[inline(always)]
fn op_floor(x: f32) -> f32 {
    x.floor()
}

#[inline(always)]
fn op_ceil(x: f32) -> f32 {
    x.ceil()
}

#[inline(always)]
fn op_round(x: f32) -> f32 {
    x.round()
}

#[inline(always)]
fn op_trunc(x: f32) -> f32 {
    x.trunc()
}

// ── Vector loops ─────────────────────────────────────────────────────────────

// vec_unary_op<op, float, float>
#[inline(always)]
pub fn vec_unary_op(n: i64, y: &mut [f32], x: &[f32], op: fn(f32) -> f32) {
    for i in 0..n as usize {
        y[i] = op(x[i]);
    }
}

// vec_unary_op_functor — functor / closure variant (for parameterised ops like xielu)
#[inline(always)]
pub fn vec_unary_op_functor<F: Fn(f32) -> f32>(n: i64, y: &mut [f32], x: &[f32], op: F) {
    for i in 0..n as usize {
        y[i] = op(x[i]);
    }
}

// ── Full tensor-geometry apply (f32 contiguous rows) ─────────────────────────

// apply_unary_op<op, float, float>
// Mirrors the C++ loop over linearised rows (ir0..ir1).
//
// Parameters:
//   ne0        — elements per row (ne0 == ne00, same shape asserted)
//   ne1        — rows per matrix (ne01)
//   ne2        — matrices per batch (ne02)
//   nb1..nb3   — dst byte strides
//   nb01..nb03 — src0 byte strides
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_unary_op_f32(
    op: fn(f32) -> f32,
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
) {
    for ir in ir0..ir1 {
        let i03 = ir / (ne2 * ne1);
        let i02 = (ir - i03 * ne2 * ne1) / ne1;
        let i01 = ir - i03 * ne2 * ne1 - i02 * ne1;

        let dst_ptr =
            dst.add(i03 as usize * nb3 + i02 as usize * nb2 + i01 as usize * nb1) as *mut f32;
        let src0_ptr =
            src0.add(i03 as usize * nb03 + i02 as usize * nb02 + i01 as usize * nb01)
                as *const f32;

        let y = std::slice::from_raw_parts_mut(dst_ptr, ne0 as usize);
        let x = std::slice::from_raw_parts(src0_ptr, ne0 as usize);
        vec_unary_op(ne0, y, x, op);
    }
}

// apply_unary_op_functor — functor variant for parameterised ops
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_unary_op_functor_f32<F: Fn(f32) -> f32>(
    op: F,
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
) {
    for ir in ir0..ir1 {
        let i03 = ir / (ne2 * ne1);
        let i02 = (ir - i03 * ne2 * ne1) / ne1;
        let i01 = ir - i03 * ne2 * ne1 - i02 * ne1;

        let dst_ptr =
            dst.add(i03 as usize * nb3 + i02 as usize * nb2 + i01 as usize * nb1) as *mut f32;
        let src0_ptr =
            src0.add(i03 as usize * nb03 + i02 as usize * nb02 + i01 as usize * nb01)
                as *const f32;

        let y = std::slice::from_raw_parts_mut(dst_ptr, ne0 as usize);
        let x = std::slice::from_raw_parts(src0_ptr, ne0 as usize);
        vec_unary_op_functor(ne0, y, x, &op);
    }
}

// ── Public entry points (f32 contiguous slices, single-threaded) ─────────────

pub fn compute_forward_abs(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_abs);
}

pub fn compute_forward_sgn(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_sgn);
}

pub fn compute_forward_neg(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_neg);
}

pub fn compute_forward_step(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_step);
}

pub fn compute_forward_tanh(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_tanh);
}

pub fn compute_forward_elu(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_elu);
}

pub fn compute_forward_relu(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_relu);
}

pub fn compute_forward_sigmoid(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_sigmoid);
}

pub fn compute_forward_hardsigmoid(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_hardsigmoid);
}

pub fn compute_forward_exp(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_exp);
}

pub fn compute_forward_hardswish(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_hardswish);
}

pub fn compute_forward_sqr(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_sqr);
}

pub fn compute_forward_sqrt(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_sqrt);
}

pub fn compute_forward_sin(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_sin);
}

pub fn compute_forward_cos(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_cos);
}

pub fn compute_forward_log(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_log);
}

pub fn compute_forward_expm1(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_expm1);
}

pub fn compute_forward_softplus(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_softplus);
}

pub fn compute_forward_floor(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_floor);
}

pub fn compute_forward_ceil(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_ceil);
}

pub fn compute_forward_round(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_round);
}

pub fn compute_forward_trunc(n: usize, y: &mut [f32], x: &[f32]) {
    vec_unary_op(n as i64, y, x, op_trunc);
}

// xielu uses a closure to capture the four parameters
pub fn compute_forward_xielu(
    n: usize,
    y: &mut [f32],
    x: &[f32],
    alpha_n: f32,
    alpha_p: f32,
    beta: f32,
    eps: f32,
) {
    vec_unary_op_functor(n as i64, y, x, |f| op_xielu(f, alpha_n, alpha_p, beta, eps));
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Scalar reference tests ────────────────────────────────────────────────

    #[test]
    fn test_op_abs() {
        assert_eq!(op_abs(-3.0), 3.0);
        assert_eq!(op_abs(2.5), 2.5);
        assert_eq!(op_abs(0.0), 0.0);
    }

    #[test]
    fn test_op_sgn() {
        assert_eq!(op_sgn(5.0), 1.0);
        assert_eq!(op_sgn(-5.0), -1.0);
        assert_eq!(op_sgn(0.0), 0.0);
    }

    #[test]
    fn test_op_neg() {
        assert_eq!(op_neg(3.0), -3.0);
        assert_eq!(op_neg(-1.0), 1.0);
    }

    #[test]
    fn test_op_step() {
        assert_eq!(op_step(1.0), 1.0);
        assert_eq!(op_step(0.0), 0.0);
        assert_eq!(op_step(-1.0), 0.0);
    }

    #[test]
    fn test_op_relu() {
        assert_eq!(op_relu(2.0), 2.0);
        assert_eq!(op_relu(0.0), 0.0);
        assert_eq!(op_relu(-1.0), 0.0);
    }

    #[test]
    fn test_op_elu() {
        assert_eq!(op_elu(1.0), 1.0);
        // elu(0) = expm1(0) = 0
        assert_eq!(op_elu(0.0), 0.0);
        // elu(-1) = expm1(-1) = e^-1 - 1 ≈ -0.6321
        let v = op_elu(-1.0);
        assert!((v - ((-1.0f32).exp() - 1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_op_sigmoid() {
        let v = op_sigmoid(0.0);
        assert!((v - 0.5).abs() < 1e-7);
        assert!(op_sigmoid(100.0) > 0.999);
        assert!(op_sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn test_op_hardsigmoid() {
        assert_eq!(op_hardsigmoid(-10.0), 0.0);
        assert_eq!(op_hardsigmoid(10.0), 1.0);
        let v = op_hardsigmoid(0.0);
        assert!((v - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_op_hardswish() {
        assert_eq!(op_hardswish(-10.0), 0.0);
        // hardswish(3.0) = 3.0 * 1.0 = 3.0
        assert_eq!(op_hardswish(3.0), 3.0);
        // hardswish(0.0) = 0.0 * 0.5 = 0.0
        assert_eq!(op_hardswish(0.0), 0.0);
    }

    #[test]
    fn test_op_sqr() {
        assert_eq!(op_sqr(3.0), 9.0);
        assert_eq!(op_sqr(-2.0), 4.0);
    }

    #[test]
    fn test_op_sqrt() {
        assert!((op_sqrt(4.0) - 2.0).abs() < 1e-7);
        assert!((op_sqrt(9.0) - 3.0).abs() < 1e-7);
    }

    #[test]
    fn test_op_softplus() {
        // softplus(21.0) = 21.0 (threshold branch)
        assert_eq!(op_softplus(21.0), 21.0);
        // softplus(0.0) = ln(2) ≈ 0.6931
        let v = op_softplus(0.0);
        assert!((v - 2.0f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_op_expm1_matches_scalar() {
        // llama.cpp implements as expf(x)-1, not expm1f(x)
        for x in [-1.0f32, 0.0, 1.0, 2.0] {
            assert_eq!(op_expm1(x), x.exp() - 1.0);
        }
    }

    #[test]
    fn test_op_xielu_positive() {
        // For x>0: alpha_p*x^2 + beta*x
        let v = op_xielu(2.0, 0.5, 1.0, 1.0, -0.1);
        assert!((v - (1.0 * 4.0 + 1.0 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_op_xielu_negative() {
        // For x<=0, x<eps: (expm1(x) - x)*alpha_n + beta*x
        // alpha_n=1, alpha_p=1, beta=1, eps=-0.1, x=-0.5
        let v = op_xielu(-0.5, 1.0, 1.0, 1.0, -0.1);
        // min_x_eps = min(-0.5, -0.1) = -0.5
        let expected = ((-0.5f32).exp_m1() - (-0.5)) * 1.0 + 1.0 * (-0.5);
        assert!((v - expected).abs() < 1e-6);
    }

    // ── vec_unary_op vs scalar ────────────────────────────────────────────────

    fn check_vec_vs_scalar(vals: &[f32], op: fn(f32) -> f32) {
        let mut y = vec![0.0f32; vals.len()];
        vec_unary_op(vals.len() as i64, &mut y, vals, op);
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(y[i], op(v), "mismatch at i={i} for x={v}");
        }
    }

    #[test]
    fn test_vec_unary_op_all_ops() {
        let vals = [-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        check_vec_vs_scalar(&vals, op_abs);
        check_vec_vs_scalar(&vals, op_sgn);
        check_vec_vs_scalar(&vals, op_neg);
        check_vec_vs_scalar(&vals, op_step);
        check_vec_vs_scalar(&vals, op_tanh);
        check_vec_vs_scalar(&vals, op_elu);
        check_vec_vs_scalar(&vals, op_relu);
        check_vec_vs_scalar(&vals, op_sigmoid);
        check_vec_vs_scalar(&vals, op_hardsigmoid);
        check_vec_vs_scalar(&vals, op_exp);
        check_vec_vs_scalar(&vals, op_hardswish);
        check_vec_vs_scalar(&vals, op_sqr);
        check_vec_vs_scalar(&vals, op_sin);
        check_vec_vs_scalar(&vals, op_cos);
        check_vec_vs_scalar(&vals, op_expm1);
        check_vec_vs_scalar(&vals, op_softplus);
        check_vec_vs_scalar(&vals, op_floor);
        check_vec_vs_scalar(&vals, op_ceil);
        check_vec_vs_scalar(&vals, op_round);
        check_vec_vs_scalar(&vals, op_trunc);
    }

    #[test]
    fn test_vec_unary_op_positive_only() {
        let vals = [0.25f32, 1.0, 4.0, 9.0];
        check_vec_vs_scalar(&vals, op_sqrt);
        check_vec_vs_scalar(&vals, op_log);
    }

    #[test]
    fn test_compute_forward_abs() {
        let x = [-1.0f32, -2.0, 3.0, -4.0];
        let mut y = [0.0f32; 4];
        compute_forward_abs(4, &mut y, &x);
        assert_eq!(y, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_compute_forward_relu() {
        let x = [-1.0f32, 0.0, 1.0, 2.0];
        let mut y = [0.0f32; 4];
        compute_forward_relu(4, &mut y, &x);
        assert_eq!(y, [0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_compute_forward_sigmoid() {
        let x = [0.0f32];
        let mut y = [0.0f32; 1];
        compute_forward_sigmoid(1, &mut y, &x);
        assert!((y[0] - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_compute_forward_xielu() {
        let x = [1.0f32, -1.0];
        let mut y = [0.0f32; 2];
        // alpha_n=0.5, alpha_p=1.0, beta=1.0, eps=-0.1
        compute_forward_xielu(2, &mut y, &x, 0.5, 1.0, 1.0, -0.1);
        let expected_pos = op_xielu(1.0, 0.5, 1.0, 1.0, -0.1);
        let expected_neg = op_xielu(-1.0, 0.5, 1.0, 1.0, -0.1);
        assert!((y[0] - expected_pos).abs() < 1e-6);
        assert!((y[1] - expected_neg).abs() < 1e-6);
    }
}
