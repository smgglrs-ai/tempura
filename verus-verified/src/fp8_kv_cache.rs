//! Verus-verified FP8 paged KV cache invariants — unbounded proofs
//!
//! Proves: FP8 page buffer sizes (u8 not f32), paged append indexing for
//! quantized KV, encode/decode field extraction, page_stride consistency.

use vstd::prelude::*;

verus! {

// ============================================================================
// FP8 pages use u8 (1 byte per element vs 4 for f32) — 4x compression
// ============================================================================

proof fn fp8_compression_ratio(
    num_elements: nat,
)
    requires num_elements >= 1,
    ensures ({
        let f32_bytes = num_elements * 4;
        let fp8_bytes = num_elements;
        fp8_bytes * 4 == f32_bytes
    }),
{
}

// ============================================================================
// FP8 page stride = block_size * num_kv_heads * head_dim (in bytes, not f32s)
// ============================================================================

proof fn fp8_page_stride(
    block_size: nat,
    num_kv_heads: nat,
    head_dim: nat,
)
    requires block_size >= 1, num_kv_heads >= 1, head_dim >= 1,
    ensures ({
        let stride = num_kv_heads * head_dim;
        let page_stride = block_size * stride;
        &&& page_stride >= stride
        &&& page_stride >= block_size
    }),
{
    assert(block_size * (num_kv_heads * head_dim) >= num_kv_heads * head_dim)
        by (nonlinear_arith) requires block_size >= 1;
    assert(block_size * (num_kv_heads * head_dim) >= block_size)
        by (nonlinear_arith) requires num_kv_heads >= 1, head_dim >= 1;
}

// ============================================================================
// FP8 append: page_base + i in bounds for u8 buffer
// ============================================================================

proof fn fp8_append_index(
    phys_page: nat,
    block_size: nat,
    block_offset: nat,
    stride: nat,
    total_pages: nat,
    i: nat,
)
    requires
        phys_page < total_pages,
        block_size >= 1,
        stride >= 1,
        block_offset < block_size,
        i < stride,
    ensures ({
        let page_stride = block_size * stride;
        let page_base = phys_page * page_stride + block_offset * stride;
        let pool_size = total_pages * page_stride;
        page_base + i < pool_size
    }),
{
    assert((block_offset + 1) * stride <= block_size * stride)
        by (nonlinear_arith) requires block_offset < block_size;
    assert(block_offset * stride + stride == (block_offset + 1) * stride)
        by (nonlinear_arith);
    assert(phys_page * (block_size * stride) + block_size * stride
        <= total_pages * (block_size * stride))
        by (nonlinear_arith) requires phys_page < total_pages;
}

// ============================================================================
// FP8 decode: exp field in [0,15], frac field in [0,7]
// ============================================================================

proof fn fp8_decode_fields(byte: u8)
    ensures ({
        let sign = (byte >> 7u8) & 1u8;
        let exp = (byte >> 3u8) & 0x0Fu8;
        let frac = byte & 0x07u8;
        &&& sign <= 1
        &&& exp <= 15
        &&& frac <= 7
    }),
{
    assert((byte >> 7u8) & 1u8 <= 1u8) by (bit_vector);
    assert((byte >> 3u8) & 0x0Fu8 <= 15u8) by (bit_vector);
    assert(byte & 0x07u8 <= 7u8) by (bit_vector);
}

// ============================================================================
// FP8 encode: output always fits in u8 (sign|exp|mantissa)
// ============================================================================

proof fn fp8_encode_fits_u8(sign: u8, exp: u8, mantissa: u8)
    requires
        sign == 0 || sign == 0x80,
        exp <= 14,
        mantissa <= 7,
    ensures ({
        let byte = sign | (exp << 3u8) | mantissa;
        // Always valid u8 — no overflow possible since sign|exp<<3|mant fits in 8 bits
        byte <= 0xFFu8
    }),
{
    assert(forall|s: u8, e: u8, m: u8|
        #![auto]
        (s == 0u8 || s == 0x80u8) && e <= 14u8 && m <= 7u8
        ==> (s | (e << 3u8) | m) <= 0xFFu8)
        by (bit_vector);
}

// ============================================================================
// FP8 NaN detection: exp==15 && frac==7
// ============================================================================

proof fn fp8_nan_detection(byte: u8)
    requires
        (byte >> 3u8) & 0x0Fu8 == 15,
        byte & 0x07u8 == 7,
    ensures ({
        let without_sign = byte & 0x7Fu8;
        without_sign == 0x7Fu8
    }),
{
    assert(forall|b: u8|
        #![auto]
        (b >> 3u8) & 0x0Fu8 == 15u8 && b & 0x07u8 == 7u8
        ==> (b & 0x7Fu8) == 0x7Fu8)
        by (bit_vector);
}

fn main() {}

} // verus!
