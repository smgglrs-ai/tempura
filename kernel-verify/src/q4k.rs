//! Q4K dual-pipeline: ref (CpuMatMulQ4K) vs SIMD-opt (CpuMatMulQ4KOpt) and GPU.
//!
//! The Q4K optimised path quantises activations to Q8 (int8) before the dot
//! product, introducing <= max_abs/127 rounding per activation.  The relative
//! error across a realistic LLM projection is empirically < 2% -- confirmed
//! here at the kernel level.  For GPU, FP16 accumulation adds another rounding
//! tier; 5% covers the combined error.

use crate::{DualPipelineRunner, Q4kConfig, VerificationReport};
use tempura_kernels::quantized::{CpuMatMulQ4K, CpuMatMulQ4KOpt};

/// Compare CpuMatMulQ4K (scalar reference) vs CpuMatMulQ4KOpt (AVX2/VNNI) on
/// realistic LLM projection dimensions.  Relative tolerance 2%.
pub fn compare_ref_vs_opt() -> VerificationReport {
    let runner = DualPipelineRunner::default();
    let configs = vec![
        Q4kConfig { m: 1, n: 4096,  k: 4096 },
        Q4kConfig { m: 1, n: 11008, k: 4096 },
        Q4kConfig { m: 8, n: 4096,  k: 4096 },
    ];
    runner.compare_q4k(
        "CpuMatMulQ4K-ref-vs-CpuMatMulQ4KOpt",
        &CpuMatMulQ4K,
        &CpuMatMulQ4KOpt,
        &configs,
        0.02,
    )
}

/// Compare CpuMatMulQ4K (scalar reference) vs the GPU kernel (CutileQuantizedGemm)
/// on a single LLM projection.  Relative tolerance 5% -- GPU uses FP16 accumulation.
///
/// Requires the `cutile` feature and an NVIDIA GPU.  Run with:
///   cargo test --package tempura-kernel-verify --features cutile
#[cfg(feature = "cutile")]
pub fn compare_ref_vs_gpu() -> VerificationReport {
    use tempura_kernels::cutile_quantized_gemm::CutileQuantizedGemm;
    let runner = DualPipelineRunner::default();
    let gpu = CutileQuantizedGemm::new().expect("CUDA device required");
    let configs = vec![Q4kConfig { m: 1, n: 4096, k: 4096 }];
    runner.compare_q4k(
        "CpuMatMulQ4K-ref-vs-CutileQuantizedGemm-GPU",
        &CpuMatMulQ4K,
        &gpu,
        &configs,
        0.05,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4k_ref_vs_opt_dual_pipeline() {
        let report = compare_ref_vs_opt();
        for r in &report.results {
            assert!(
                r.pass,
                "Q4K ref???opt mismatch on {}: {}",
                r.test_name, r.detail
            );
        }
        assert!(report.pass);
    }

    #[test]
    #[cfg(feature = "cutile")]
    fn test_q4k_ref_vs_gpu_dual_pipeline() {
        let report = compare_ref_vs_gpu();
        for r in &report.results {
            assert!(
                r.pass,
                "Q4K ref???GPU mismatch on {}: {}",
                r.test_name, r.detail
            );
        }
        assert!(report.pass);
    }
}
