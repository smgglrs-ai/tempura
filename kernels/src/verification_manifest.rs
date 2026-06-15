use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct KernelProof {
    pub kernel: &'static str,
    pub harness: &'static str,
    pub properties: Vec<&'static str>,
    pub status: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerificationManifest {
    pub kernels: Vec<KernelProof>,
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
        ],
        caveats: vec![
            "Proofs cover memory safety (no OOB, no panics) — NOT mathematical correctness",
            "Floating-point operations (exp, sqrt, cos, sin) are over-approximated by CBMC",
            "Proofs verify indexing arithmetic, not full kernel execution (avoids CBMC memory explosion)",
            "GPU kernels (cuda-oxide) will be verified with the same trait contract in Phase 3",
        ],
    }
}
