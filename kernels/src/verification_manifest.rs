use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct KernelProof {
    pub kernel: &'static str,
    pub harness: &'static str,
    pub properties: Vec<&'static str>,
    pub status: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct MathProperty {
    pub kernel: &'static str,
    pub property: &'static str,
    pub method: &'static str,
    pub status: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerificationManifest {
    pub kernels: Vec<KernelProof>,
    pub math_properties: Vec<MathProperty>,
    pub caveats: Vec<&'static str>,
}

pub fn manifest() -> VerificationManifest {
    VerificationManifest {
        kernels: vec![
            KernelProof {
                kernel: "CpuAttention",
                harness: "attention_q_offset_in_bounds",
                properties: vec!["Q indexing within allocated bounds"],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuAttention",
                harness: "attention_kv_offset_in_bounds",
                properties: vec!["K/V indexing within allocated bounds"],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuLayerNorm",
                harness: "layernorm_indices_in_bounds",
                properties: vec!["Row indexing within allocated bounds"],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuRMSNorm",
                harness: "rmsnorm_indices_in_bounds",
                properties: vec!["Row indexing within allocated bounds"],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuRoPE",
                harness: "rope_indices_in_bounds",
                properties: vec![
                    "Position lookup within bounds",
                    "Rotation pair indexing within bounds",
                ],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuMatMul",
                harness: "matmul_indices_in_bounds",
                properties: vec!["Triple-loop indexing within allocated bounds"],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuSampler",
                harness: "sampler_top_k_index_in_bounds",
                properties: vec!["top-k index access within vocab bounds"],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuSampler",
                harness: "sample_token_cumsum_terminates",
                properties: vec!["Fallback index is valid"],
                status: "proven",
            },
            KernelProof {
                kernel: "KvCache",
                harness: "append_kv_offsets_in_bounds",
                properties: vec!["Append destination within allocated buffer"],
                status: "proven",
            },
            KernelProof {
                kernel: "KvCache",
                harness: "k_v_slice_in_bounds",
                properties: vec!["Slice length within allocated buffer"],
                status: "proven",
            },
            KernelProof {
                kernel: "KvCache",
                harness: "remaining_capacity_consistent",
                properties: vec!["seq_len + remaining = max_seq_len"],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuPolarQuant",
                harness: "compress_indices_in_bounds",
                properties: vec![
                    "Vector start offset within data bounds",
                    "Angle index within quantized_angles bounds",
                ],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuPolarQuant",
                harness: "decompress_indices_in_bounds",
                properties: vec![
                    "Angle lookup within compressed buffer bounds",
                    "Output vector indexing within allocated bounds",
                ],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuAttention",
                harness: "attention_batch_isolation_no_overlap",
                properties: vec![
                    "Q/K/V offset ranges for different batch indices never overlap",
                    "Batch-invariance: each sequence's memory region is isolated",
                ],
                status: "proven",
            },
            KernelProof {
                kernel: "GpuAttention",
                harness: "gpu_attention_thread_decomposition_in_bounds",
                properties: vec![
                    "Thread index decomposition (b,s,h,d) stays within dimension bounds",
                    "Reconstructed q_base offset is within buffer bounds",
                ],
                status: "proven",
            },
            KernelProof {
                kernel: "GpuRoPE",
                harness: "gpu_rope_thread_decomposition_in_bounds",
                properties: vec![
                    "Thread index decomposition (s,h,d) stays within dimension bounds",
                    "Reconstructed offset is within buffer bounds",
                ],
                status: "proven",
            },
            KernelProof {
                kernel: "GpuRMSNorm",
                harness: "gpu_rmsnorm_pass_indices_in_bounds",
                properties: vec![
                    "Pass 1 row*h + i indexing within x bounds",
                    "Pass 2 (row, col) decomposition within ss and weight bounds",
                ],
                status: "proven",
            },
            KernelProof {
                kernel: "GpuMatMul",
                harness: "gpu_tiled_gemm_smem_index_in_bounds",
                properties: vec![
                    "Shared memory index ty*TILE+tx < TILE*TILE",
                    "Guarded global memory accesses within matrix bounds",
                ],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuRMSNorm",
                harness: "rmsnorm_full_execution_no_panic",
                properties: vec!["Full kernel execution with bounded inputs does not panic"],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuSampler",
                harness: "sampler_full_execution_no_panic",
                properties: vec!["Full kernel execution with bounded inputs does not panic"],
                status: "proven",
            },
            KernelProof {
                kernel: "CpuLayerNorm",
                harness: "layernorm_full_execution_no_panic",
                properties: vec!["Full kernel execution with bounded inputs does not panic"],
                status: "proven",
            },
        ],
        math_properties: vec![
            MathProperty {
                kernel: "CpuSampler",
                property: "Softmax output sums to 1.0",
                method: "proptest",
                status: "tested",
            },
            MathProperty {
                kernel: "CpuSampler",
                property: "All probabilities non-negative",
                method: "proptest",
                status: "tested",
            },
            MathProperty {
                kernel: "CpuSampler",
                property: "Top-k limits nonzero probability count",
                method: "proptest",
                status: "tested",
            },
            MathProperty {
                kernel: "CpuSampler",
                property: "Near-greedy temperature selects argmax",
                method: "proptest",
                status: "tested",
            },
            MathProperty {
                kernel: "CpuRMSNorm",
                property: "Output has unit RMS with identity weights",
                method: "proptest",
                status: "tested",
            },
            MathProperty {
                kernel: "CpuRMSNorm",
                property: "Weight scaling applied correctly",
                method: "unit_test",
                status: "tested",
            },
            MathProperty {
                kernel: "CpuRoPE",
                property: "Rotation preserves L2 norm",
                method: "proptest",
                status: "tested",
            },
            MathProperty {
                kernel: "CpuRoPE",
                property: "Position zero is identity transform",
                method: "proptest",
                status: "tested",
            },
            MathProperty {
                kernel: "CpuAttention",
                property: "Output bounded by value range (convex combination)",
                method: "proptest",
                status: "tested",
            },
            MathProperty {
                kernel: "CpuSampler",
                property: "Softmax f32 output matches f64 reference within 1e-6",
                method: "f64_reference",
                status: "tested",
            },
            MathProperty {
                kernel: "CpuRMSNorm",
                property: "RMSNorm f32 output matches f64 reference within 1e-5",
                method: "f64_reference",
                status: "tested",
            },
        ],
        caveats: vec![
            "Proofs cover memory safety (no OOB, no panics) — NOT mathematical correctness",
            "Floating-point operations (exp, sqrt, cos, sin) are over-approximated by CBMC; f64 reference comparison tests validate numerical adequacy of f32 implementations",
            "Full-execution harnesses prove no-panic on CpuRMSNorm, CpuSampler, and CpuLayerNorm with bounded inputs",
            "GPU kernel offset arithmetic is verified via CPU-side Kani harnesses that prove the same decomposition formulas",
            "Mathematical properties verified via proptest (see math_properties); Kani cannot handle f32 transcendentals",
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_non_empty() {
        let m = manifest();
        assert!(!m.kernels.is_empty());
        assert!(!m.caveats.is_empty());
    }

    #[test]
    fn test_all_proofs_have_proven_status() {
        let m = manifest();
        for proof in &m.kernels {
            assert_eq!(
                proof.status, "proven",
                "kernel {} harness {} is not proven",
                proof.kernel, proof.harness
            );
        }
    }

    #[test]
    fn test_all_proofs_have_properties() {
        let m = manifest();
        for proof in &m.kernels {
            assert!(
                !proof.properties.is_empty(),
                "kernel {} harness {} has no properties",
                proof.kernel,
                proof.harness
            );
        }
    }

    #[test]
    fn test_manifest_serializes_to_json() {
        let m = manifest();
        let json = serde_json::to_string(&m).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["kernels"].is_array());
        assert!(v["caveats"].is_array());
    }
}
