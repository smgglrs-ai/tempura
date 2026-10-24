//! sm_120 PTX kernel for GPU-side MXFP4 activation quantization.
//!
//! Converts f32[M][K] activations to E2M1 + UE8M0 tiled layout required by
//! mxf4_batch_gemm_tiled, entirely on-GPU with no PCIe round-trips.
//!
//! Tile: 16 rows x 32 cols = 512 elements, one UE8M0 scale per tile.
//! Grid:  (m16_tiles * k32_blocks, 1, 1) -- one block per tile
//! Block: (256, 1, 1) -- 8 warps, each thread quantizes 2 elements
//!
//! Output layout matches quantize_activation_to_mxfp4_tiled (CPU reference):
//!   packed:  [M][K/2] bytes, E2M1 nibbles, row-major
//!   scales:  [M/16][K/32] bytes, UE8M0 tile scales
//!
//! Targets sm_120 (not sm_120a -- no tensor core instructions needed).

pub const ENTRY: &str = "mxfp4_quantize_activation_tiled";
pub const MIN_CC: u32 = 1200;

pub const PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// mxfp4_quantize_activation_tiled
// f32[M][K] -> (packed_e2m1[M][K/2], scales_ue8m0[M/16][K/32])
//
// Tile: 16 rows x 32 cols = 512 elements, one UE8M0 block scale per tile.
// Grid:  (m16_tiles * k32_blocks, 1, 1)  -- one block per [16xK/32] tile
// Block: (256, 1, 1)                      -- 8 warps, 2 elements per thread
//
// Thread tid in [0,256):
//   row_in_tile = tid / 16      (which of 16 rows in this tile)
//   col_in_tile = (tid%16) * 2  (always even -- thread handles col pair)
//
// Phase 1: warp-level shfl.sync max-abs reduce, inter-warp reduce in smem.
// Phase 2: all threads compute UE8M0 scale from shared max_abs.
// Phase 3: each thread quantizes its 2 elements to E2M1 nibbles and packs.
//
// E2M1 thresholds (nearest-neighbor, ignoring tie-breaking at exact midpoints):
//   [0,    0.25) -> idx 0  (0.0)
//   [0.25, 0.75) -> idx 1  (0.5)
//   [0.75, 1.25) -> idx 2  (1.0)
//   [1.25, 1.75) -> idx 3  (1.5)
//   [1.75, 2.5 ) -> idx 4  (2.0)
//   [2.5,  3.5 ) -> idx 5  (3.0)
//   [3.5,  5.0 ) -> idx 6  (4.0)
//   [5.0,  inf ) -> idx 7  (6.0)
//
// ASCII only, no unicode.
.visible .entry mxfp4_quantize_activation_tiled(
    .param .u64 param_src,       // *const f32 -- input [M * K]
    .param .u64 param_packed,    // *mut u8    -- output [M * K / 2]
    .param .u64 param_scales,    // *mut u8    -- output [M/16 * K/32]
    .param .u32 param_m,         // M (multiple of 16)
    .param .u32 param_k,         // K (multiple of 32)
    .param .u32 param_k32_blocks // K / 32
)
{
    .reg .u32  %r<40>;
    .reg .u64  %rd<8>;
    .reg .f32  %f<12>;
    .reg .b32  %b<8>;
    .reg .pred %p<6>;

    // smem: 8 f32 slots (one per warp) for inter-warp max-abs reduce.
    .shared .align 4 .b8 smem[32];

    // Load parameters.
    ld.param.u64 %rd0, [param_src];
    ld.param.u64 %rd1, [param_packed];
    ld.param.u64 %rd2, [param_scales];
    ld.param.u32 %r0,  [param_m];
    ld.param.u32 %r1,  [param_k];
    ld.param.u32 %r2,  [param_k32_blocks];

    mov.u32 %r3, %ctaid.x;    // block_idx
    mov.u32 %r4, %tid.x;      // threadIdx.x [0,256)

    // m16_tile = block_idx / k32_blocks
    // k32_block = block_idx % k32_blocks
    div.u32 %r5, %r3, %r2;   // m16_tile
    rem.u32 %r6, %r3, %r2;   // k32_block

    // m_base = m16_tile * 16
    shl.b32 %r7, %r5, 4;

    // k_base = k32_block * 32
    shl.b32 %r8, %r6, 5;

    // row_in_tile = tid / 16  (0..15)
    shr.u32 %r9, %r4, 4;

    // col_in_tile = (tid % 16) * 2  (0,2,4,...,30) -- always even
    and.b32 %r10, %r4, 15;
    shl.b32 %r10, %r10, 1;

    // global_row = m_base + row_in_tile
    add.u32 %r11, %r7, %r9;

    // global_col0 = k_base + col_in_tile
    add.u32 %r12, %r8, %r10;

    // offset_bytes = (global_row * K + global_col0) * 4
    mul.lo.u32 %r13, %r11, %r1;
    add.u32    %r13, %r13, %r12;
    mul.lo.u32 %r14, %r13, 4;

    // Load val0 = src[global_row * K + global_col0]
    cvt.u64.u32 %rd3, %r14;
    add.u64     %rd3, %rd0, %rd3;
    ld.global.f32 %f0, [%rd3];

    // Load val1 = src[global_row * K + global_col0 + 1]  (+4 bytes)
    add.u64     %rd4, %rd3, 4;
    ld.global.f32 %f1, [%rd4];

    // max_abs = max(abs(val0), abs(val1))
    abs.f32 %f2, %f0;
    abs.f32 %f3, %f1;
    max.f32 %f4, %f2, %f3;

    // Warp-level reduce: shfl.sync.down 5 rounds (32 -> 16 -> 8 -> 4 -> 2 -> 1).
    // shfl.sync.down.b32 requires b32 operands; use explicit bit-copy via mov.b32.
    mov.b32 %b0, %f4;
    shfl.sync.down.b32 %b1, %b0, 16, 31, 0xFFFFFFFF;
    mov.b32 %f5, %b1;
    max.f32 %f4, %f4, %f5;

    mov.b32 %b0, %f4;
    shfl.sync.down.b32 %b1, %b0, 8, 31, 0xFFFFFFFF;
    mov.b32 %f5, %b1;
    max.f32 %f4, %f4, %f5;

    mov.b32 %b0, %f4;
    shfl.sync.down.b32 %b1, %b0, 4, 31, 0xFFFFFFFF;
    mov.b32 %f5, %b1;
    max.f32 %f4, %f4, %f5;

    mov.b32 %b0, %f4;
    shfl.sync.down.b32 %b1, %b0, 2, 31, 0xFFFFFFFF;
    mov.b32 %f5, %b1;
    max.f32 %f4, %f4, %f5;

    mov.b32 %b0, %f4;
    shfl.sync.down.b32 %b1, %b0, 1, 31, 0xFFFFFFFF;
    mov.b32 %f5, %b1;
    max.f32 %f4, %f4, %f5;

    // Lane 0 of each warp writes warp-max to smem[warp_id * 4].
    and.b32 %r15, %r4, 31;     // lane_id = tid % 32
    shr.u32 %r16, %r4, 5;      // warp_id = tid / 32
    setp.eq.u32 %p0, %r15, 0;  // lane_id == 0

    mov.u32 %r17, smem;         // smem base address (u32, shared space)
    shl.b32 %r18, %r16, 2;     // warp_id * 4 bytes
    add.u32 %r19, %r17, %r18;  // smem + warp_id * 4
    @%p0 st.shared.f32 [%r19], %f4;

    bar.sync 0;

    // Thread 0: sequential reduce of 8 warp-maxima from smem.
    setp.eq.u32 %p1, %r4, 0;
    @%p1 ld.shared.f32 %f6, [smem + 0];
    @%p1 ld.shared.f32 %f7, [smem + 4];
    @%p1 max.f32 %f6, %f6, %f7;
    @%p1 ld.shared.f32 %f7, [smem + 8];
    @%p1 max.f32 %f6, %f6, %f7;
    @%p1 ld.shared.f32 %f7, [smem + 12];
    @%p1 max.f32 %f6, %f6, %f7;
    @%p1 ld.shared.f32 %f7, [smem + 16];
    @%p1 max.f32 %f6, %f6, %f7;
    @%p1 ld.shared.f32 %f7, [smem + 20];
    @%p1 max.f32 %f6, %f6, %f7;
    @%p1 ld.shared.f32 %f7, [smem + 24];
    @%p1 max.f32 %f6, %f6, %f7;
    @%p1 ld.shared.f32 %f7, [smem + 28];
    @%p1 max.f32 %f6, %f6, %f7;
    @%p1 st.shared.f32 [smem], %f6;  // broadcast block max_abs

    bar.sync 0;

    // All threads read block-max from smem[0].
    ld.shared.f32 %f4, [smem];

    // Compute UE8M0 scale from max_abs:
    //   ideal = max_abs / 6.0
    //   ue8m0 = biased_exp(ideal) + (mantissa != 0 ? 1 : 0)   // ceiling
    //   (matches compute_ue8m0_scale: exp = ceil(log2(ideal)) + 127)
    //
    // Special case: max_abs <= 0 or NaN => ue8m0 = 0 (zero block).
    setp.leu.f32 %p2, %f4, 0f00000000;  // max_abs <= 0.0 or NaN

    // ideal = max_abs / 6.0  (0f40C00000 = 6.0f)
    mov.f32 %f8, 0f40C00000;
    div.approx.f32 %f9, %f4, %f8;

    // Extract biased exponent of ideal.
    mov.b32 %b2, %f9;
    and.b32 %r20, %b2, 0x7F800000;
    shr.u32 %r20, %r20, 23;     // biased_exp(ideal)

    // Mantissa nonzero => need ceiling (increment exponent).
    and.b32 %r21, %b2, 0x007FFFFF;
    setp.ne.u32 %p4, %r21, 0;
    mov.u32 %r22, %r20;
    @%p4 add.u32 %r22, %r20, 1;

    // Clamp to [0, 255].
    min.u32 %r22, %r22, 255;

    // Override with 0 for zero/NaN block.
    @%p2 mov.u32 %r22, 0;

    // Thread 0 writes ue8m0 to scales[m16_tile * k32_blocks + k32_block].
    mul.lo.u32 %r23, %r5, %r2;
    add.u32    %r23, %r23, %r6;
    cvt.u64.u32 %rd5, %r23;
    add.u64    %rd5, %rd2, %rd5;
    @%p1 st.global.u8 [%rd5], %r22;

    // Reconstruct scale_f32 = 2^(ue8m0 - 127) from IEEE 754 bit pattern.
    // ue8m0 == 0 => scale = 0.0 (zero block, all elements map to 0).
    setp.eq.u32 %p5, %r22, 0;
    shl.b32 %b3, %r22, 23;     // ue8m0 << 23 (zero mantissa = exact power of 2)
    mov.b32 %f10, %b3;
    @%p5 mov.f32 %f10, 0f00000000;

    // Divide val0 by scale (0 if scale == 0).
    @%p5 mov.f32 %f0, 0f00000000;
    setp.ne.f32 %p3, %f10, 0f00000000;
    @%p3 div.approx.f32 %f0, %f0, %f10;  // scaled0

    // Divide val1 by scale.
    @%p5 mov.f32 %f1, 0f00000000;
    @%p3 div.approx.f32 %f1, %f1, %f10;  // scaled1

    // Quantize scaled0 to E2M1 nibble in %r25:
    //   nibble = sign_bit(3) | magnitude_idx(2:0)
    mov.b32 %b4, %f0;
    and.b32 %r24, %b4, 0x80000000;     // sign of f0 (0 or 0x80000000)
    abs.f32 %f11, %f0;
    mov.u32 %r25, 0;
    setp.ge.f32 %p3, %f11, 0f3E800000; @%p3 mov.u32 %r25, 1;  // >= 0.25
    setp.ge.f32 %p3, %f11, 0f3F400000; @%p3 mov.u32 %r25, 2;  // >= 0.75
    setp.ge.f32 %p3, %f11, 0f3FA00000; @%p3 mov.u32 %r25, 3;  // >= 1.25
    setp.ge.f32 %p3, %f11, 0f3FE00000; @%p3 mov.u32 %r25, 4;  // >= 1.75
    setp.ge.f32 %p3, %f11, 0f40200000; @%p3 mov.u32 %r25, 5;  // >= 2.5
    setp.ge.f32 %p3, %f11, 0f40600000; @%p3 mov.u32 %r25, 6;  // >= 3.5
    setp.ge.f32 %p3, %f11, 0f40A00000; @%p3 mov.u32 %r25, 7;  // >= 5.0
    setp.ne.u32 %p3, %r24, 0;
    @%p3 or.b32 %r25, %r25, 8;         // set sign bit (bit 3)

    // Quantize scaled1 to E2M1 nibble in %r27.
    mov.b32 %b5, %f1;
    and.b32 %r26, %b5, 0x80000000;
    abs.f32 %f11, %f1;
    mov.u32 %r27, 0;
    setp.ge.f32 %p3, %f11, 0f3E800000; @%p3 mov.u32 %r27, 1;
    setp.ge.f32 %p3, %f11, 0f3F400000; @%p3 mov.u32 %r27, 2;
    setp.ge.f32 %p3, %f11, 0f3FA00000; @%p3 mov.u32 %r27, 3;
    setp.ge.f32 %p3, %f11, 0f3FE00000; @%p3 mov.u32 %r27, 4;
    setp.ge.f32 %p3, %f11, 0f40200000; @%p3 mov.u32 %r27, 5;
    setp.ge.f32 %p3, %f11, 0f40600000; @%p3 mov.u32 %r27, 6;
    setp.ge.f32 %p3, %f11, 0f40A00000; @%p3 mov.u32 %r27, 7;
    setp.ne.u32 %p3, %r26, 0;
    @%p3 or.b32 %r27, %r27, 8;

    // Pack: byte = (nibble0 & 0xF) | ((nibble1 & 0xF) << 4)
    // Lo nibble = first element (col_in_tile), hi nibble = second (col_in_tile+1).
    and.b32 %r28, %r25, 15;
    and.b32 %r29, %r27, 15;
    shl.b32 %r29, %r29, 4;
    or.b32  %r28, %r28, %r29;

    // Write packed byte to packed[global_row * (K/2) + k_base/2 + col_in_tile/2].
    shr.u32 %r30, %r1, 1;           // K/2
    mul.lo.u32 %r31, %r11, %r30;    // global_row * (K/2)
    shr.u32 %r32, %r8, 1;           // k_base / 2
    add.u32    %r31, %r31, %r32;
    shr.u32 %r33, %r10, 1;          // col_in_tile / 2
    add.u32    %r31, %r31, %r33;
    cvt.u64.u32 %rd6, %r31;
    add.u64    %rd6, %rd1, %rd6;
    st.global.u8 [%rd6], %r28;

    ret;
}
"#;
