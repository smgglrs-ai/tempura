use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct OpShape {
    pub name: &'static str,
    pub input_elements: usize,
    pub output_elements: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BoundaryCheck {
    pub from_op: &'static str,
    pub to_op: &'static str,
    pub from_output: usize,
    pub to_input: usize,
    pub compatible: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompositionReport {
    pub operations: Vec<OpShape>,
    pub boundaries: Vec<BoundaryCheck>,
    pub composed_safe: bool,
    pub detail: String,
}

pub struct CompositionVerifier;

impl CompositionVerifier {
    pub fn verify_sequence(ops: &[OpShape]) -> CompositionReport {
        if ops.is_empty() {
            return CompositionReport {
                operations: Vec::new(),
                boundaries: Vec::new(),
                composed_safe: true,
                detail: "empty sequence is trivially safe".into(),
            };
        }

        let mut boundaries = Vec::new();
        let mut all_compatible = true;

        for i in 0..ops.len() - 1 {
            let from = &ops[i];
            let to = &ops[i + 1];
            let compatible = from.output_elements == to.input_elements;
            if !compatible {
                all_compatible = false;
            }
            boundaries.push(BoundaryCheck {
                from_op: from.name,
                to_op: to.name,
                from_output: from.output_elements,
                to_input: to.input_elements,
                compatible,
            });
        }

        let detail = if all_compatible {
            format!(
                "{} operations compose safely: all {} boundaries dimensionally compatible",
                ops.len(),
                boundaries.len()
            )
        } else {
            let mismatches: Vec<String> = boundaries
                .iter()
                .filter(|b| !b.compatible)
                .map(|b| {
                    format!(
                        "{} → {}: output {} ≠ input {}",
                        b.from_op, b.to_op, b.from_output, b.to_input
                    )
                })
                .collect();
            format!("dimensional mismatch at: {}", mismatches.join(", "))
        };

        CompositionReport {
            operations: ops.to_vec(),
            boundaries,
            composed_safe: all_compatible,
            detail,
        }
    }

    pub fn transformer_layer_ops(
        seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
    ) -> Vec<OpShape> {
        let h = hidden_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let inter = intermediate_size;
        let s = seq_len;

        vec![
            OpShape {
                name: "RMSNorm(input)",
                input_elements: s * h,
                output_elements: s * h,
            },
            OpShape {
                name: "MatMul(Q proj)",
                input_elements: s * h,
                output_elements: s * q_dim,
            },
            OpShape {
                name: "MatMul(K proj)",
                input_elements: s * h,
                output_elements: s * kv_dim,
            },
            OpShape {
                name: "MatMul(V proj)",
                input_elements: s * h,
                output_elements: s * kv_dim,
            },
            OpShape {
                name: "RoPE(Q)",
                input_elements: s * q_dim,
                output_elements: s * q_dim,
            },
            OpShape {
                name: "RoPE(K)",
                input_elements: s * kv_dim,
                output_elements: s * kv_dim,
            },
            OpShape {
                name: "Attention",
                input_elements: s * q_dim,
                output_elements: s * q_dim,
            },
            OpShape {
                name: "MatMul(O proj)",
                input_elements: s * q_dim,
                output_elements: s * h,
            },
            OpShape {
                name: "Residual add",
                input_elements: s * h,
                output_elements: s * h,
            },
            OpShape {
                name: "RMSNorm(post-attn)",
                input_elements: s * h,
                output_elements: s * h,
            },
            OpShape {
                name: "MatMul(gate proj)",
                input_elements: s * h,
                output_elements: s * inter,
            },
            OpShape {
                name: "MatMul(up proj)",
                input_elements: s * h,
                output_elements: s * inter,
            },
            OpShape {
                name: "SiLU×gate",
                input_elements: s * inter,
                output_elements: s * inter,
            },
            OpShape {
                name: "MatMul(down proj)",
                input_elements: s * inter,
                output_elements: s * h,
            },
            OpShape {
                name: "Residual add(FFN)",
                input_elements: s * h,
                output_elements: s * h,
            },
        ]
    }

    pub fn verify_transformer_layer(
        seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
    ) -> CompositionReport {
        let ops = Self::transformer_layer_ops(
            seq_len,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
        );

        let attn_chain = vec![
            ops[0].clone(), // RMSNorm
            ops[1].clone(), // Q proj
            ops[4].clone(), // RoPE(Q)
            ops[6].clone(), // Attention
            ops[7].clone(), // O proj
            ops[8].clone(), // Residual
        ];
        let attn_report = Self::verify_sequence(&attn_chain);

        let ffn_chain = vec![
            ops[9].clone(),  // RMSNorm(post-attn)
            ops[10].clone(), // gate proj
            ops[12].clone(), // SiLU×gate
            ops[13].clone(), // down proj
            ops[14].clone(), // Residual(FFN)
        ];
        let ffn_report = Self::verify_sequence(&ffn_chain);

        let composed_safe = attn_report.composed_safe && ffn_report.composed_safe;
        let mut all_boundaries = attn_report.boundaries;
        all_boundaries.extend(ffn_report.boundaries);

        let detail = if composed_safe {
            format!(
                "transformer layer ({} ops) verified: attention chain + FFN chain both dimensionally safe",
                ops.len()
            )
        } else {
            format!(
                "transformer layer verification failed: attn={}, ffn={}",
                attn_report.detail, ffn_report.detail
            )
        };

        CompositionReport {
            operations: ops,
            boundaries: all_boundaries,
            composed_safe,
            detail,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compatible_sequence() {
        let ops = vec![
            OpShape {
                name: "A",
                input_elements: 100,
                output_elements: 200,
            },
            OpShape {
                name: "B",
                input_elements: 200,
                output_elements: 200,
            },
            OpShape {
                name: "C",
                input_elements: 200,
                output_elements: 50,
            },
        ];
        let report = CompositionVerifier::verify_sequence(&ops);
        assert!(report.composed_safe);
        assert_eq!(report.boundaries.len(), 2);
        assert!(report.boundaries.iter().all(|b| b.compatible));
    }

    #[test]
    fn test_incompatible_sequence() {
        let ops = vec![
            OpShape {
                name: "A",
                input_elements: 100,
                output_elements: 200,
            },
            OpShape {
                name: "B",
                input_elements: 300,
                output_elements: 50,
            },
        ];
        let report = CompositionVerifier::verify_sequence(&ops);
        assert!(!report.composed_safe);
        assert!(!report.boundaries[0].compatible);
    }

    #[test]
    fn test_empty_sequence() {
        let report = CompositionVerifier::verify_sequence(&[]);
        assert!(report.composed_safe);
    }

    #[test]
    fn test_single_op() {
        let ops = vec![OpShape {
            name: "A",
            input_elements: 10,
            output_elements: 20,
        }];
        let report = CompositionVerifier::verify_sequence(&ops);
        assert!(report.composed_safe);
        assert!(report.boundaries.is_empty());
    }

    #[test]
    fn test_transformer_layer_llama_7b() {
        let report = CompositionVerifier::verify_transformer_layer(1, 4096, 32, 32, 128, 11008);
        assert!(
            report.composed_safe,
            "LLaMA-7B layer composition failed: {}",
            report.detail
        );
        assert_eq!(report.operations.len(), 15);
    }

    #[test]
    fn test_transformer_layer_tiny_test() {
        let report = CompositionVerifier::verify_transformer_layer(4, 64, 4, 4, 16, 256);
        assert!(
            report.composed_safe,
            "tiny_test composition failed: {}",
            report.detail
        );
    }

    #[test]
    fn test_transformer_layer_gqa() {
        let report = CompositionVerifier::verify_transformer_layer(1, 4096, 32, 8, 128, 14336);
        assert!(
            report.composed_safe,
            "GQA composition failed: {}",
            report.detail
        );
    }

    #[test]
    fn test_report_serializes() {
        let report = CompositionVerifier::verify_transformer_layer(1, 64, 4, 4, 16, 256);
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("composed_safe"));
        assert!(json.contains("boundaries"));
    }
}
