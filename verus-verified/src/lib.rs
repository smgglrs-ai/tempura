//! Verus-verified components for tempura
//!
//! This crate re-declares critical data structures from engine/ and kernels/
//! inside verus!{} blocks with formal specifications. Verus proves properties
//! for ALL possible inputs (deductive verification), complementing Kani's
//! bounded model checking which verifies for finite instances.
//!
//! Verification hierarchy:
//!   - Kani: implementation properties (bounded model checking, e.g. max_seq_len<=8)
//!   - Verus: implementation properties (deductive, unbounded — all N)
//!
//! Proof categories:
//!   - Indexing arithmetic (mirrors Kani harnesses): kv_cache, attention, rmsnorm,
//!     layernorm, rope, matmul, sampling, turboquant, quantized, model, speculative,
//!     disaggregated, weights, gguf
//!   - Struct invariants: kv_cache_invariant, page_allocator, block_table,
//!     session_consistency, prefix_cache
//!   - Loop invariants: chunked_forward, speculative_invariant
//!   - Monotonicity: tick_monotonicity

#![no_std]

use vstd::prelude::*;

// Indexing arithmetic proofs (mirror Kani harnesses)
pub mod attention;
pub mod disaggregated;
pub mod ggml_quants;
pub mod gguf;
pub mod kv_cache;
pub mod layernorm;
pub mod matmul;
pub mod model;
pub mod quantized;
pub mod rmsnorm;
pub mod rope;
pub mod sampling;
pub mod speculative;
pub mod turboquant;
pub mod weights;

// Struct invariant proofs (beyond Kani)
pub mod batch_scheduler;
pub mod block_table;
pub mod chunked_forward;
pub mod forward_pass;
pub mod fused_kernels;
pub mod gqa;
pub mod kv_append_advance;
pub mod kv_cache_invariant;
pub mod medusa;
pub mod model_config;
pub mod page_allocator;
pub mod prefix_cache;
pub mod projection_shapes;
pub mod radix_tree;
pub mod sampling_penalties;
pub mod session_consistency;
pub mod speculative_invariant;
pub mod tick_monotonicity;
pub mod token_generation;
pub mod trace_invariant;

// GPU, attention, and weight proofs
pub mod causal_mask;
pub mod end_to_end;
pub mod f16_encoding;
pub mod gpu_elementwise;
pub mod gpu_flash_attn;
pub mod gpu_quantized_gemm;
pub mod bitnet_forward;
pub mod i2s_bitnet;
pub mod i2s_vec_dot;
pub mod kv_rollback;
pub mod tq2_block;
pub mod layer_graph;
pub mod mma_fragments;
pub mod config_validation;
pub mod flash_tiles;
pub mod fp8;
pub mod fp8_kv_cache;
pub mod gguf_alignment;
pub mod gpu_tiles;
pub mod guided_decoding;
pub mod lora;
pub mod paged_kv_cache;
pub mod position_encoding;
pub mod quantized_forward;
pub mod rope_frequency;
pub mod tensor_parallel;
pub mod trace_diff;
pub mod activations_moe;
pub mod deltanet;
pub mod sampling_chain;
pub mod weight_shapes;
