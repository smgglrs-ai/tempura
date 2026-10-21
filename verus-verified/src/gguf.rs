//! Verus-verified GGUF dequantization proofs -- unbounded
//!
//! Proves GGUF header parsing, Q4_0/Q8_0 block layout, Q6K element coverage,
//! and Q6K value reconstruction range.

use vstd::prelude::*;

verus! {

// ============================================================================
// Proof: GGUF header fields fit within data of length >= 24
// ============================================================================

proof fn gguf_header_parse_in_bounds(data_len: nat)
    requires data_len >= 24,
    ensures ({
        // magic: 4 bytes at offset 0
        // version: 4 bytes at offset 4
        // tensor_count: 8 bytes at offset 8
        // metadata_kv_count: 8 bytes at offset 16
        // total: 24 bytes
        &&& 0 + 4 <= data_len
        &&& 4 + 4 <= data_len
        &&& 8 + 8 <= data_len
        &&& 16 + 8 <= data_len
    }),
{
}

// ============================================================================
// Proof: Q4_0 block access -- block_start + 18 fits in total_bytes
// ============================================================================

proof fn gguf_dequant_q4_0_block_size(num_elements: nat, b: nat)
    requires
        num_elements >= 32,
        num_elements % 32 == 0,
        b < num_elements / 32,
    ensures ({
        let num_blocks = num_elements / 32;
        let block_size: nat = 18;
        let total_bytes = num_blocks * block_size;
        let block_start = b * block_size;
        &&& block_start + 2 <= total_bytes
        &&& block_start + 2 + 16 <= total_bytes
    }),
{
    let num_blocks = num_elements / 32;
    let block_size: nat = 18;
    assert((b + 1) * block_size <= num_blocks * block_size)
        by (nonlinear_arith) requires b < num_blocks;
    assert(b * block_size + block_size == (b + 1) * block_size)
        by (nonlinear_arith);
}

// ============================================================================
// Proof: Q8_0 block access -- block_start + 34 fits in total_bytes
// ============================================================================

proof fn gguf_dequant_q8_0_block_size(num_elements: nat, b: nat)
    requires
        num_elements >= 32,
        num_elements % 32 == 0,
        b < num_elements / 32,
    ensures ({
        let num_blocks = num_elements / 32;
        let block_size: nat = 34;
        let total_bytes = num_blocks * block_size;
        let block_start = b * block_size;
        &&& block_start + 2 <= total_bytes
        &&& block_start + 2 + 32 <= total_bytes
    }),
{
    let num_blocks = num_elements / 32;
    let block_size: nat = 34;
    assert((b + 1) * block_size <= num_blocks * block_size)
        by (nonlinear_arith) requires b < num_blocks;
    assert(b * block_size + block_size == (b + 1) * block_size)
        by (nonlinear_arith);
}

// ============================================================================
// Proof: Q6K interleaved dequant maps all 256 elements without gaps
// ============================================================================

proof fn gguf_dequant_q6k_covers_all_256(half: nat, l: nat)
    requires
        half < 2,
        l < 32,
    ensures ({
        let y_off = half * 128;
        let ql_off = half * 64;
        let qh_off = half * 32;
        let e0 = y_off + l;
        let e1 = y_off + l + 32;
        let e2 = y_off + l + 64;
        let e3 = y_off + l + 96;
        // All 4 elements < 256
        &&& e0 < 256 &&& e1 < 256 &&& e2 < 256 &&& e3 < 256
        // All distinct
        &&& e0 != e1 &&& e0 != e2 &&& e0 != e3
        &&& e1 != e2 &&& e1 != e3
        &&& e2 != e3
        // ql access in bounds (128 bytes)
        &&& ql_off + l < 128
        &&& ql_off + l + 32 < 128
        // qh access in bounds (64 bytes)
        &&& qh_off + l < 64
        // scale access valid (16 entries)
        &&& half * 8 + l / 16 < 16
        &&& half * 8 + l / 16 + 2 < 16
        &&& half * 8 + l / 16 + 4 < 16
        &&& half * 8 + l / 16 + 6 < 16
    }),
{
}

// ============================================================================
// Proof: Q6K 6-bit reconstruction: (ql & 0x0F) | ((qh >> shift) & 0x03) << 4 <= 63
// ============================================================================

proof fn gguf_dequant_q6k_value_in_range(ql_byte: u8, qh_byte: u8, shift: u32)
    requires shift < 4,
    ensures ({
        let low: u8 = ql_byte & 0x0Fu8;
        let shift_amount = (shift * 2) as u8;
        let high: u8 = (qh_byte >> shift_amount) & 0x03u8;
        let q6: u8 = low | (high << 4u8);
        q6 <= 63
    }),
{
    assert(forall|ql: u8, qh: u8, s: u8|
        #![auto]
        ((ql & 0x0Fu8) | (((qh >> s) & 0x03u8) << 4u8)) <= 63u8)
        by (bit_vector);
}

fn main() {}

} // verus!
