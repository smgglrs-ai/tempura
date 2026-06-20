use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct KernelSchema {
    pub name: String,
    pub description: String,
    pub methods: Vec<MethodSchema>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MethodSchema {
    pub name: String,
    pub params: Vec<ParamSchema>,
    pub returns: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParamSchema {
    pub name: String,
    pub r#type: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SchemaRegistry {
    pub traits: Vec<KernelSchema>,
}

pub fn registry() -> SchemaRegistry {
    SchemaRegistry {
        traits: vec![
            attention_schema(),
            matmul_schema(),
            rmsnorm_schema(),
            rope_schema(),
            sampler_schema(),
            layernorm_schema(),
            allreduce_schema(),
            kv_cache_compressor_schema(),
            matmul_q_schema(),
            guided_decoder_schema(),
        ],
    }
}

pub fn get_schema(name: &str) -> Option<KernelSchema> {
    registry().traits.into_iter().find(|s| s.name == name)
}

fn attention_schema() -> KernelSchema {
    KernelSchema {
        name: "Attention".into(),
        description: "Scaled dot-product attention with GQA support".into(),
        methods: vec![MethodSchema {
            name: "forward".into(),
            params: vec![
                p("q", "&[f32]", "[batch, seq_len, num_heads, head_dim]"),
                p("k", "&[f32]", "[batch, kv_len, num_kv_heads, head_dim]"),
                p("v", "&[f32]", "[batch, kv_len, num_kv_heads, head_dim]"),
                p("output", "&mut [f32]", "[batch, seq_len, num_heads, head_dim]"),
                p("batch", "usize", "batch size"),
                p("seq_len", "usize", "query sequence length"),
                p("kv_len", "usize", "key/value sequence length"),
                p("num_heads", "usize", "number of query heads"),
                p("num_kv_heads", "usize", "number of KV heads (GQA: num_heads / num_kv_heads = groups)"),
                p("head_dim", "usize", "dimension per head"),
                p("scale", "f32", "typically 1/sqrt(head_dim)"),
            ],
            returns: "Result<()>".into(),
        }],
    }
}

fn matmul_schema() -> KernelSchema {
    KernelSchema {
        name: "MatMul".into(),
        description: "Matrix multiplication (f32)".into(),
        methods: vec![
            MethodSchema {
                name: "forward".into(),
                params: vec![
                    p("a", "&[f32]", "[m, k] row-major"),
                    p("b", "&[f32]", "[k, n] row-major"),
                    p("output", "&mut [f32]", "[m, n] row-major"),
                    p("m", "usize", "rows of A / output"),
                    p("n", "usize", "columns of B / output"),
                    p("k", "usize", "inner dimension"),
                ],
                returns: "Result<()>".into(),
            },
            MethodSchema {
                name: "forward_b_transposed".into(),
                params: vec![
                    p("a", "&[f32]", "[m, k] row-major"),
                    p("b", "&[f32]", "[n, k] row-major (transposed)"),
                    p("output", "&mut [f32]", "[m, n] row-major"),
                    p("m", "usize", "rows"),
                    p("n", "usize", "columns"),
                    p("k", "usize", "inner dimension"),
                ],
                returns: "Result<()>".into(),
            },
        ],
    }
}

fn rmsnorm_schema() -> KernelSchema {
    KernelSchema {
        name: "RMSNorm".into(),
        description: "RMS normalization (in-place)".into(),
        methods: vec![MethodSchema {
            name: "forward".into(),
            params: vec![
                p("x", "&mut [f32]", "[tokens, hidden_size] modified in-place"),
                p("weight", "&[f32]", "[hidden_size] gamma"),
                p("hidden_size", "usize", "hidden dimension"),
                p("eps", "f32", "numerical stability epsilon"),
            ],
            returns: "Result<()>".into(),
        }],
    }
}

fn rope_schema() -> KernelSchema {
    KernelSchema {
        name: "RoPE".into(),
        description: "Rotary position embeddings (in-place)".into(),
        methods: vec![MethodSchema {
            name: "forward".into(),
            params: vec![
                p("x", "&mut [f32]", "[batch, seq_len, num_heads, head_dim]"),
                p("batch", "usize", "batch size"),
                p("seq_len", "usize", "sequence length"),
                p("num_heads", "usize", "number of heads"),
                p("head_dim", "usize", "dimension per head (must be even)"),
                p("positions", "&[u32]", "[batch, seq_len] position indices"),
                p("theta", "f32", "base frequency (typically 10000.0)"),
            ],
            returns: "Result<()>".into(),
        }],
    }
}

fn sampler_schema() -> KernelSchema {
    KernelSchema {
        name: "Sampler".into(),
        description: "Token sampling with temperature, top-k, top-p".into(),
        methods: vec![
            MethodSchema {
                name: "apply".into(),
                params: vec![
                    p("logits", "&[f32]", "[vocab_size] raw logits"),
                    p("output", "&mut [f32]", "[vocab_size] filtered probabilities"),
                    p("params", "&SamplingParams", "temperature, top_k, top_p"),
                ],
                returns: "Result<()>".into(),
            },
            MethodSchema {
                name: "sample_token".into(),
                params: vec![
                    p("probs", "&[f32]", "[vocab_size] probability distribution"),
                    p("rng", "Option<&mut dyn RngCore>", "None for greedy argmax"),
                ],
                returns: "u32".into(),
            },
        ],
    }
}

fn layernorm_schema() -> KernelSchema {
    KernelSchema {
        name: "LayerNorm".into(),
        description: "Layer normalization (in-place)".into(),
        methods: vec![MethodSchema {
            name: "forward".into(),
            params: vec![
                p("x", "&mut [f32]", "[tokens, hidden_size]"),
                p("weight", "&[f32]", "[hidden_size] gamma"),
                p("bias", "Option<&[f32]>", "[hidden_size] beta"),
                p("hidden_size", "usize", "hidden dimension"),
                p("eps", "f32", "epsilon"),
            ],
            returns: "Result<()>".into(),
        }],
    }
}

fn allreduce_schema() -> KernelSchema {
    KernelSchema {
        name: "AllReduce".into(),
        description: "Distributed all-reduce sum (in-place)".into(),
        methods: vec![MethodSchema {
            name: "sum_in_place".into(),
            params: vec![p("data", "&mut [f32]", "data to reduce across ranks")],
            returns: "Result<()>".into(),
        }],
    }
}

fn kv_cache_compressor_schema() -> KernelSchema {
    KernelSchema {
        name: "KvCacheCompressor".into(),
        description: "KV cache quantization/compression".into(),
        methods: vec![
            MethodSchema {
                name: "compress".into(),
                params: vec![
                    p("data", "&[f32]", "KV vectors to compress"),
                    p("head_dim", "usize", "dimension per head"),
                ],
                returns: "Result<CompressedKV>".into(),
            },
            MethodSchema {
                name: "decompress".into(),
                params: vec![p("compressed", "&CompressedKV", "compressed KV data")],
                returns: "Result<Vec<f32>>".into(),
            },
        ],
    }
}

fn matmul_q_schema() -> KernelSchema {
    KernelSchema {
        name: "MatMulQ".into(),
        description: "Quantized matrix multiplication (Q4K/Q6K)".into(),
        methods: vec![
            MethodSchema {
                name: "forward_b_transposed_q4k".into(),
                params: vec![
                    p("a", "&[f32]", "[m, k] activations"),
                    p("b_quantized", "&[u8]", "Q4_K packed weights"),
                    p("output", "&mut [f32]", "[m, n] output"),
                    p("m", "usize", "rows"),
                    p("n", "usize", "columns"),
                    p("k", "usize", "inner dimension (multiple of 256)"),
                ],
                returns: "Result<()>".into(),
            },
            MethodSchema {
                name: "forward_b_transposed_q6k".into(),
                params: vec![
                    p("a", "&[f32]", "[m, k] activations"),
                    p("b_quantized", "&[u8]", "Q6_K packed weights"),
                    p("output", "&mut [f32]", "[m, n] output"),
                    p("m", "usize", "rows"),
                    p("n", "usize", "columns"),
                    p("k", "usize", "inner dimension (multiple of 256)"),
                ],
                returns: "Result<()>".into(),
            },
        ],
    }
}

fn guided_decoder_schema() -> KernelSchema {
    KernelSchema {
        name: "GuidedDecoder".into(),
        description: "Grammar-guided logit masking for constrained decoding".into(),
        methods: vec![
            MethodSchema {
                name: "mask_logits".into(),
                params: vec![
                    p("token_ids_so_far", "&[u32]", "tokens generated so far"),
                    p("logits", "&mut [f32]", "[vocab_size] logits to mask"),
                ],
                returns: "()".into(),
            },
            MethodSchema {
                name: "is_complete".into(),
                params: vec![],
                returns: "bool".into(),
            },
            MethodSchema {
                name: "tokens_masked_last_step".into(),
                params: vec![],
                returns: "usize".into(),
            },
        ],
    }
}

fn p(name: &str, ty: &str, desc: &str) -> ParamSchema {
    ParamSchema {
        name: name.into(),
        r#type: ty.into(),
        description: desc.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_all_traits() {
        let reg = registry();
        assert_eq!(reg.traits.len(), 10);
        let names: Vec<&str> = reg.traits.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"Attention"));
        assert!(names.contains(&"MatMul"));
        assert!(names.contains(&"RMSNorm"));
        assert!(names.contains(&"GuidedDecoder"));
    }

    #[test]
    fn test_get_schema_found() {
        let schema = get_schema("Attention").unwrap();
        assert_eq!(schema.methods.len(), 1);
        assert_eq!(schema.methods[0].params.len(), 11);
    }

    #[test]
    fn test_get_schema_not_found() {
        assert!(get_schema("NonExistent").is_none());
    }

    #[test]
    fn test_registry_serializes() {
        let reg = registry();
        let json = serde_json::to_string(&reg).unwrap();
        assert!(json.contains("Attention"));
        assert!(json.contains("head_dim"));
    }
}
