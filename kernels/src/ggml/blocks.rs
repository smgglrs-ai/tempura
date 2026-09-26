// Block structs matching ggml/src/ggml-common.h exactly.
// Every struct is #[repr(C)] so raw GGUF bytes can be reinterpreted safely.
// Size constants are validated by const_assert! at the bottom.

// ── Basic quants (QK = 32) ─────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ1_0 {
    pub d: u16,       // f16 scale
    pub qs: [u8; 16], // 1-bit packed (QK=128, 128/8=16 bytes)
}
pub const BLOCK_Q1_0_SIZE: usize = 18;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub d: u16,       // f16 scale
    pub qs: [u8; 16], // 4-bit quantized (32 values, 2 per byte)
}
pub const BLOCK_Q4_0_SIZE: usize = 18;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_1 {
    pub d: u16,       // f16 scale
    pub m: u16,       // f16 min
    pub qs: [u8; 16], // 4-bit quantized
}
pub const BLOCK_Q4_1_SIZE: usize = 20;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5_0 {
    pub d: u16,       // f16 scale
    pub qh: [u8; 4],  // 5th bit of 32 values (32/8=4 bytes)
    pub qs: [u8; 16], // low 4 bits
}
pub const BLOCK_Q5_0_SIZE: usize = 22;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5_1 {
    pub d: u16,       // f16 scale
    pub m: u16,       // f16 min
    pub qh: [u8; 4],  // 5th bit
    pub qs: [u8; 16], // low 4 bits
}
pub const BLOCK_Q5_1_SIZE: usize = 24;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    pub d: u16,       // f16 scale
    pub qs: [i8; 32], // 8-bit quantized
}
pub const BLOCK_Q8_0_SIZE: usize = 34;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_1 {
    pub d: u16,       // f16 delta
    pub s: u16,       // f16 d * sum(qs)
    pub qs: [i8; 32], // 8-bit quantized
}
pub const BLOCK_Q8_1_SIZE: usize = 36;

// ── K-quants (QK_K = 256) ──────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ2K {
    pub scales: [u8; 16], // 4-bit scales and mins (QK_K/16)
    pub qs: [u8; 64],     // 2-bit quantized (256/4=64 bytes)
    pub d: u16,           // f16 super-block scale
    pub dmin: u16,        // f16 super-block min
}
pub const BLOCK_Q2K_SIZE: usize = 84;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ3K {
    pub hmask: [u8; 32],  // high bit masks (256/8=32 bytes)
    pub qs: [u8; 64],     // low 2 bits (256/4=64 bytes)
    pub scales: [u8; 12], // 6-bit scales, packed
    pub d: u16,           // f16 super-block scale
}
pub const BLOCK_Q3K_SIZE: usize = 110;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4K {
    pub d: u16,           // f16 super-block scale
    pub dmin: u16,        // f16 super-block min
    pub scales: [u8; 12], // 6-bit packed scales/mins (K_SCALE_SIZE)
    pub qs: [u8; 128],    // 4-bit quantized (256/2=128 bytes)
}
pub const BLOCK_Q4K_SIZE: usize = 144;

/// 8 Q4K blocks interleaved for 8-wide SIMD GEMV (matches llama.cpp block_q4_Kx8).
///
/// Layout (1152 bytes total):
/// - d[8]:       f16 super-block scales, one per weight row
/// - dmin[8]:    f16 super-block mins, one per weight row
/// - scales[96]: 8×12 scale bytes interleaved by sub-block pair (24 bytes per sb pair)
/// - qs[1024]:   8×128 nibble bytes interleaved in 8-byte chunks
///               [B0[0:8], B1[0:8], ..., B7[0:8], B0[8:16], ..., B7[24:32],
///                B0[32:40], ..., B7[56:64], ...] (one complete 256-byte sb, then next)
#[repr(C)]
pub struct BlockQ4Kx8 {
    pub d: [u16; 8],
    pub dmin: [u16; 8],
    pub scales: [u8; 96],
    pub qs: [u8; 1024],
}
pub const BLOCK_Q4KX8_SIZE: usize = 1152;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5K {
    pub d: u16,           // f16 super-block scale
    pub dmin: u16,        // f16 super-block min
    pub scales: [u8; 12], // 6-bit packed scales/mins
    pub qh: [u8; 32],     // 5th bit (256/8=32 bytes)
    pub qs: [u8; 128],    // low 4 bits
}
pub const BLOCK_Q5K_SIZE: usize = 176;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ6K {
    pub ql: [u8; 128],    // lower 4 bits of 6-bit values (256/2=128 bytes)
    pub qh: [u8; 64],     // upper 2 bits (256/4=64 bytes)
    pub scales: [i8; 16], // 8-bit scales per group of 16
    pub d: u16,           // f16 super-block scale
}
pub const BLOCK_Q6K_SIZE: usize = 210;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8K {
    pub d: f32,           // f32 scale (NOT f16!)
    pub qs: [i8; 256],    // 8-bit quantized
    pub bsums: [i16; 16], // per-16-element sums (precomputed)
}
pub const BLOCK_Q8K_SIZE: usize = 292;

/// 4 Q8K activation rows interleaved for 4×8 GEMM with Q4Kx8 weights.
///
/// Layout (1168 bytes total, matches llama.cpp block_q8_Kx4):
/// - d[4]:       f32 scales, one per activation row
/// - qs[1024]:   4×256 int8 quants interleaved with blck_size_interleave=8:
///               groups of 32 bytes: 8 from row 0, 8 from row 1, 8 from row 2, 8 from row 3
/// - bsums[64]:  per-16-element sums, index = row*4 + 16*(elem/64) + (elem/16)%4
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8Kx4 {
    pub d: [f32; 4],
    pub qs: [i8; 1024],
    pub bsums: [i16; 64],
}
pub const BLOCK_Q8KX4_SIZE: usize = 1168;

impl std::fmt::Debug for BlockQ8K {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockQ8K")
            .field("d", &self.d)
            .field("qs", &&self.qs[..8])
            .field("bsums", &&self.bsums[..4])
            .finish()
    }
}

// ── Ternary quants (QK_K = 256) ────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockTQ1_0 {
    pub qs: [u8; 48], // base-3 packed: 5 trits per byte (256/5≈52, but 48+4=52 trits)
    pub qh: [u8; 4],  // 4 trits per byte (remaining 8 trits)
    pub d: u16,       // f16 scale
}
pub const BLOCK_TQ1_0_SIZE: usize = 54;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockTQ2_0 {
    pub qs: [u8; 64], // 2-bit packed (256/4=64 bytes)
    pub d: u16,       // f16 scale
}
pub const BLOCK_TQ2_0_SIZE: usize = 66;

// ── Micro-FP quants ────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockMxfp4 {
    pub e: u8,        // E8M0 shared scale
    pub qs: [u8; 16], // E2M1 values (32/2=16 bytes)
}
pub const BLOCK_MXFP4_SIZE: usize = 17;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockNvfp4 {
    pub d: [u8; 4],   // UE4M3 per-16-element scales (64/16=4 sub-blocks)
    pub qs: [u8; 32], // E2M1 values (64/2=32 bytes)
}
pub const BLOCK_NVFP4_SIZE: usize = 36;

// ── IQ (importance-based) quants ───────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ1S {
    pub d: u16,       // f16 scale
    pub qs: [u8; 32], // grid indices
    pub qh: [u16; 8], // high bits + signs
}
pub const BLOCK_IQ1S_SIZE: usize = 50;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ1M {
    pub qs: [u8; 32],    // grid indices
    pub qh: [u8; 16],    // high bits
    pub scales: [u8; 8], // scales (no d field — uses iq1m_scale_t union)
}
pub const BLOCK_IQ1M_SIZE: usize = 56;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ2XXS {
    pub d: u16,        // f16 scale
    pub qs: [u16; 32], // grid indices + signs
}
pub const BLOCK_IQ2XXS_SIZE: usize = 66;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ2XS {
    pub d: u16,          // f16 scale
    pub qs: [u16; 32],   // grid indices + signs
    pub scales: [u8; 8], // per-32-element scales
}
pub const BLOCK_IQ2XS_SIZE: usize = 74;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ2S {
    pub d: u16,          // f16 scale
    pub qs: [u8; 64],    // grid indices
    pub qh: [u8; 8],     // sign bits
    pub scales: [u8; 8], // scales
}
pub const BLOCK_IQ2S_SIZE: usize = 82;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ3XXS {
    pub d: u16,       // f16 scale
    pub qs: [u8; 96], // grid indices (3 bytes per 8 values: 256/8*3=96)
}
pub const BLOCK_IQ3XXS_SIZE: usize = 98;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ3S {
    pub d: u16,          // f16 scale
    pub qs: [u8; 64],    // grid indices
    pub qh: [u8; 8],     // high bits
    pub signs: [u8; 32], // sign bits
    pub scales: [u8; 4], // scales
}
pub const BLOCK_IQ3S_SIZE: usize = 110;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ4NL {
    pub d: u16,       // f16 scale
    pub qs: [u8; 16], // 4-bit non-linear indices (QK=32, 32/2=16)
}
pub const BLOCK_IQ4NL_SIZE: usize = 18;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ4XS {
    pub d: u16,            // f16 scale
    pub scales_h: u16,     // high bits of scales
    pub scales_l: [u8; 4], // low bits of scales
    pub qs: [u8; 128],     // 4-bit non-linear indices (QK_K=256)
}
pub const BLOCK_IQ4XS_SIZE: usize = 136;

// ── Size assertions ────────────────────────────────────────────────────────

macro_rules! assert_block_size {
    ($ty:ty, $expected:expr) => {
        const _: () = {
            assert!(
                std::mem::size_of::<$ty>() == $expected,
                // can't format in const context, but the type name in the macro invocation is enough
            );
        };
    };
}

assert_block_size!(BlockQ1_0, BLOCK_Q1_0_SIZE);
assert_block_size!(BlockQ4_0, BLOCK_Q4_0_SIZE);
assert_block_size!(BlockQ4_1, BLOCK_Q4_1_SIZE);
assert_block_size!(BlockQ5_0, BLOCK_Q5_0_SIZE);
assert_block_size!(BlockQ5_1, BLOCK_Q5_1_SIZE);
assert_block_size!(BlockQ8_0, BLOCK_Q8_0_SIZE);
assert_block_size!(BlockQ8_1, BLOCK_Q8_1_SIZE);
assert_block_size!(BlockQ2K, BLOCK_Q2K_SIZE);
assert_block_size!(BlockQ3K, BLOCK_Q3K_SIZE);
assert_block_size!(BlockQ4K, BLOCK_Q4K_SIZE);
assert_block_size!(BlockQ5K, BLOCK_Q5K_SIZE);
assert_block_size!(BlockQ6K, BLOCK_Q6K_SIZE);
assert_block_size!(BlockQ8K, BLOCK_Q8K_SIZE);
assert_block_size!(BlockTQ1_0, BLOCK_TQ1_0_SIZE);
assert_block_size!(BlockTQ2_0, BLOCK_TQ2_0_SIZE);
assert_block_size!(BlockMxfp4, BLOCK_MXFP4_SIZE);
assert_block_size!(BlockNvfp4, BLOCK_NVFP4_SIZE);
assert_block_size!(BlockIQ1S, BLOCK_IQ1S_SIZE);
assert_block_size!(BlockIQ1M, BLOCK_IQ1M_SIZE);
assert_block_size!(BlockIQ2XXS, BLOCK_IQ2XXS_SIZE);
assert_block_size!(BlockIQ2XS, BLOCK_IQ2XS_SIZE);
assert_block_size!(BlockIQ2S, BLOCK_IQ2S_SIZE);
assert_block_size!(BlockIQ3XXS, BLOCK_IQ3XXS_SIZE);
assert_block_size!(BlockIQ3S, BLOCK_IQ3S_SIZE);
assert_block_size!(BlockIQ4NL, BLOCK_IQ4NL_SIZE);
assert_block_size!(BlockIQ4XS, BLOCK_IQ4XS_SIZE);

// ── Block reinterpretation helpers ─────────────────────────────────────────

pub fn cast_blocks<T: Copy>(data: &[u8], type_size: usize) -> &[T] {
    assert_eq!(std::mem::size_of::<T>(), type_size);
    assert_eq!(data.len() % type_size, 0);
    assert_eq!(data.as_ptr() as usize % std::mem::align_of::<T>().min(1), 0);
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, data.len() / type_size) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_ggml() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ4_1>(), 20);
        assert_eq!(std::mem::size_of::<BlockQ5_0>(), 22);
        assert_eq!(std::mem::size_of::<BlockQ5_1>(), 24);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
        assert_eq!(std::mem::size_of::<BlockQ8_1>(), 36);
        assert_eq!(std::mem::size_of::<BlockQ2K>(), 84);
        assert_eq!(std::mem::size_of::<BlockQ3K>(), 110);
        assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
        assert_eq!(std::mem::size_of::<BlockQ5K>(), 176);
        assert_eq!(std::mem::size_of::<BlockQ6K>(), 210);
        assert_eq!(std::mem::size_of::<BlockQ8K>(), 292);
        assert_eq!(std::mem::size_of::<BlockTQ1_0>(), 54);
        assert_eq!(std::mem::size_of::<BlockTQ2_0>(), 66);
        assert_eq!(std::mem::size_of::<BlockMxfp4>(), 17);
        assert_eq!(std::mem::size_of::<BlockNvfp4>(), 36);
        assert_eq!(std::mem::size_of::<BlockQ1_0>(), 18);
    }

    #[test]
    fn iq_block_sizes_match_ggml() {
        assert_eq!(std::mem::size_of::<BlockIQ1S>(), 50);
        assert_eq!(std::mem::size_of::<BlockIQ1M>(), 56);
        assert_eq!(std::mem::size_of::<BlockIQ2XXS>(), 66);
        assert_eq!(std::mem::size_of::<BlockIQ2XS>(), 74);
        assert_eq!(std::mem::size_of::<BlockIQ2S>(), 82);
        assert_eq!(std::mem::size_of::<BlockIQ3XXS>(), 98);
        assert_eq!(std::mem::size_of::<BlockIQ3S>(), 110);
        assert_eq!(std::mem::size_of::<BlockIQ4NL>(), 18);
        assert_eq!(std::mem::size_of::<BlockIQ4XS>(), 136);
    }
}
