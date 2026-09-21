use crate::portable_math::portable_exp;

const GELU_COEF_A: f32 = 0.044715;
const SQRT_2_OVER_PI: f32 = 0.79788456080286535587989211986876;
const GELU_QUICK_COEF: f32 = -1.702;

/// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
/// Used by GPT-2, Phi, BERT. Formula matches ggml_gelu_f32.
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + portable_tanh(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)))
}

/// Fast GELU approximation: x * sigmoid(1.702 * x)
/// Matches ggml_gelu_quick_f32.
#[inline(always)]
pub fn gelu_quick(x: f32) -> f32 {
    x * (1.0 / (1.0 + portable_exp(GELU_QUICK_COEF * x)))
}

/// ReLU activation: max(0, x)
#[inline(always)]
pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Squared ReLU: relu(x)^2
/// Used by BitNet models.
#[inline(always)]
pub fn squared_relu(x: f32) -> f32 {
    let r = relu(x);
    r * r
}

/// GeGLU: gelu(gate) * up
/// Used by Gemma.
#[inline(always)]
pub fn geglu(gate: f32, up: f32) -> f32 {
    gelu(gate) * up
}

/// ReGLU: relu(gate) * up
#[inline(always)]
pub fn reglu(gate: f32, up: f32) -> f32 {
    relu(gate) * up
}

/// Portable tanh using portable_exp for cross-platform determinism.
/// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
#[inline(always)]
fn portable_tanh(x: f32) -> f32 {
    if x > 10.0 {
        return 1.0;
    }
    if x < -10.0 {
        return -1.0;
    }
    let e2x = portable_exp(2.0 * x);
    (e2x - 1.0) / (e2x + 1.0)
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn gelu_no_panic() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite());
        kani::assume(x >= -100.0 && x <= 100.0);
        let _y = gelu(x);
    }

    #[kani::proof]
    fn gelu_finite_output() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite());
        kani::assume(x >= -100.0 && x <= 100.0);
        let y = gelu(x);
        assert!(y.is_finite(), "gelu({x}) produced non-finite {y}");
    }

    #[kani::proof]
    fn gelu_quick_finite_for_finite_input() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite());
        kani::assume(x >= -100.0 && x <= 100.0);
        let y = gelu_quick(x);
        assert!(y.is_finite(), "gelu_quick({x}) produced non-finite {y}");
    }

    #[kani::proof]
    fn relu_non_negative() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite());
        let y = relu(x);
        assert!(y >= 0.0);
    }

    #[kani::proof]
    fn squared_relu_non_negative() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite());
        let y = squared_relu(x);
        assert!(y >= 0.0);
    }

    #[kani::proof]
    fn geglu_no_panic() {
        let gate: f32 = kani::any();
        let up: f32 = kani::any();
        kani::assume(gate.is_finite());
        kani::assume(up.is_finite());
        kani::assume(gate >= -100.0 && gate <= 100.0);
        kani::assume(up >= -100.0 && up <= 100.0);
        let _y = geglu(gate, up);
    }

    #[kani::proof]
    fn reglu_no_panic() {
        let gate: f32 = kani::any();
        let up: f32 = kani::any();
        kani::assume(gate.is_finite());
        kani::assume(up.is_finite());
        let _y = reglu(gate, up);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_known_values() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        // gelu(1.0) ≈ 0.8412
        assert!(
            (gelu(1.0) - 0.8412).abs() < 1e-3,
            "gelu(1.0) = {}",
            gelu(1.0)
        );
        // gelu(-1.0) ≈ -0.1588
        assert!(
            (gelu(-1.0) - (-0.1588)).abs() < 1e-3,
            "gelu(-1.0) = {}",
            gelu(-1.0)
        );
        // large positive → ~x
        assert!((gelu(10.0) - 10.0).abs() < 1e-3);
        // large negative → ~0
        assert!(gelu(-10.0).abs() < 1e-3);
    }

    #[test]
    fn test_gelu_matches_ggml() {
        // Compare against the standard formula: 0.5*x*(1+tanh(sqrt(2/pi)*x*(1+0.044715*x*x)))
        let test_vals = [0.0f32, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        for &x in &test_vals {
            let reference =
                0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh());
            let ours = gelu(x);
            let err = (ours - reference).abs();
            assert!(
                err < 1e-5,
                "gelu({x}): ours={ours}, ref={reference}, err={err}"
            );
        }
    }

    #[test]
    fn test_gelu_quick_known_values() {
        assert!((gelu_quick(0.0) - 0.0).abs() < 1e-6);
        // gelu_quick(x) = x * sigmoid(1.702 * x)
        let x = 1.0f32;
        let reference = x * (1.0 / (1.0 + (-1.702f32 * x).exp()));
        assert!(
            (gelu_quick(x) - reference).abs() < 1e-4,
            "gelu_quick(1.0) = {}",
            gelu_quick(x)
        );
    }

    #[test]
    fn test_gelu_quick_matches_ggml() {
        let test_vals = [0.0f32, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0];
        for &x in &test_vals {
            let reference = x * (1.0 / (1.0 + (GELU_QUICK_COEF * x).exp()));
            let ours = gelu_quick(x);
            let err = (ours - reference).abs();
            assert!(
                err < 1e-4,
                "gelu_quick({x}): ours={ours}, ref={reference}, err={err}"
            );
        }
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(1.0), 1.0);
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(5.5), 5.5);
        assert_eq!(relu(-100.0), 0.0);
    }

    #[test]
    fn test_squared_relu() {
        assert_eq!(squared_relu(0.0), 0.0);
        assert_eq!(squared_relu(2.0), 4.0);
        assert_eq!(squared_relu(-1.0), 0.0);
        assert_eq!(squared_relu(3.0), 9.0);
    }

    #[test]
    fn test_geglu() {
        assert!((geglu(0.0, 1.0) - 0.0).abs() < 1e-6);
        // geglu(gate, up) = gelu(gate) * up
        let g = gelu(1.0);
        assert!((geglu(1.0, 2.0) - g * 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_reglu() {
        assert_eq!(reglu(0.0, 1.0), 0.0);
        assert_eq!(reglu(2.0, 3.0), 6.0);
        assert_eq!(reglu(-1.0, 5.0), 0.0);
    }

    #[test]
    fn test_gelu_edge_cases() {
        assert!(gelu(f32::NAN).is_nan());
        // Very large positive: gelu(x) ≈ x
        assert!((gelu(50.0) - 50.0).abs() < 1e-3);
        // Very large negative: gelu(x) ≈ 0
        assert!(gelu(-50.0).abs() < 1e-3);
    }

    #[test]
    fn test_portable_tanh_accuracy() {
        let test_vals = [0.0f32, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0];
        for &x in &test_vals {
            let reference = x.tanh();
            let ours = portable_tanh(x);
            let err = (ours - reference).abs();
            assert!(
                err < 1e-5,
                "tanh({x}): ours={ours}, ref={reference}, err={err}"
            );
        }
    }

    #[test]
    fn test_portable_tanh_saturation() {
        assert!((portable_tanh(20.0) - 1.0).abs() < 1e-6);
        assert!((portable_tanh(-20.0) - (-1.0)).abs() < 1e-6);
    }
}
