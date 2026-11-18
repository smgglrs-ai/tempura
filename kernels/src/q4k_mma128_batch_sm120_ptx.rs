/// Hand-tuned PTX kernel for Q4K x Q8_1 batch GEMM using INT8 tensor cores
/// (sm_120), LARGE-m variant: up to 128 activation rows (8 m16 tiles) in ONE
/// launch so the weight matrix is read ONCE per launch instead of once per
/// 16-row chunk (the prefill shape). The per-tile math is byte-identical to
/// q4k_mma_batch_sm120 -- same fragment mapping, same warp stripe, same
/// per-row f32 accumulation order -- only restructured with the m16 tile as
/// an inner loop over statically-unrolled register accumulators.
///
/// # Design: 8-warp K-parallel with mma.sync.aligned.m16n8k32, m-tile unroll
///
/// Grid:  (ceil(N/8), 1, 1) -- one block per n8-column output tile
/// Block: (256, 1, 1) -- 8 warps
///
/// Same warp stripe / fragment mapping / Q4K parsing as
/// `q4k_mma_batch_sm120` (see that file's docs; the 2026-11-16f fix). The
/// ONLY structural change: the per-(kb, mw) body loops over 8 statically
/// unrolled m16 tiles with per-tile REGISTER accumulators (f32), so:
/// - weights (B nibbles + header) are loaded once per (kb, mw) for ALL
///   128 rows (8x fewer weight reads than 16-row chunking at m=128);
/// - the per-row f32 accumulation ORDER is unchanged (same (kb, mw)
///   sequence per warp, same 8-partial final reduction), so each output row
///   is bitwise-equal to `q4k_mma_batch_sm120` run on the same rows --
///   probe-gatable at bit level;
/// - tiles with T*16 >= m are skipped by a CTA-uniform branch.
///
/// # Shared memory (33152 bytes)
/// [0..383]:     header (d8/dmin8/sc8/mv8 per neuron), as the 16-row kernel
/// [384..33151]: smem_partial[8 warps][8 tiles][16 rows x 8 cols] f32
///               slot = 384 + warp*4096 + t*512 + (row*8 + col)*4
/// Written ONCE at the end (register accumulators), reduced per tile by
/// warp 0 with two bar.sync per tile (store / reduce+write).
pub const MIN_CC: u32 = 1200;
pub const ENTRY: &str = "q4k_mma128_batch_sm120";

pub const PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// q4k_mma128_batch_sm120
//
// Large-m (<= 128 rows = 8 m16 tiles) INT8 tensor-core batch GEMM, Q4K x Q8_1.
// Weights read once per launch. Grid: (ceil(N/8),1,1), Block: (256,1,1).
//
// Parameters:
//   param_weights:    *const u32  -- native GGUF Q4K [N * k_blocks * 36 u32]
//   param_act_qs:     *const u32  -- Q8_1 INT8 activations [M * k_blocks * 64 u32]
//   param_act_ds:     *const u32  -- Q8_1 block scales [M * k_blocks * 8 u32]
//   param_out:        *mut f32    -- output [M * N]
//   param_n:          u32         -- N (output neurons, multiple of 8)
//   param_k_blocks:   u32         -- k_blocks (K = k_blocks * 256)
//   param_m:          u32         -- M (actual activation rows, <= 128)
.visible .entry q4k_mma128_batch_sm120(
    .param .u64 param_weights,
    .param .u64 param_act_qs,
    .param .u64 param_act_ds,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .u32 param_k_blocks,
    .param .u32 param_m
)
{
    .reg .u32  %r<200>;
    .reg .u64  %rd<40>;
    .reg .f32  %f<140>;
    .reg .pred %p<32>;
    .reg .b16  %h<8>;

    // [0..127]:   smem_d8[8]   f32
    // [128..255]: smem_dmin8[8] f32
    // [256..319]: smem_sc8[8 neurons][8 sub-blocks] u8
    // [320..383]: smem_mv8[8 neurons][8 sub-blocks] u8
    // [384..33151]: smem_partial[8 warps][8 tiles][16 rows][8 cols] f32
    .shared .align 4 .b8 smem_buf[33152];

    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_act_qs];
    ld.param.u64 %rd2, [param_act_ds];
    ld.param.u64 %rd3, [param_out];
    ld.param.u32 %r0,  [param_n];
    ld.param.u32 %r1,  [param_k_blocks];
    ld.param.u32 %r2,  [param_m];

    // ---- Validate block: n_tile_base < N ------------------------------------
    mov.u32 %r3, %ctaid.x;
    shl.b32 %r4, %r3, 3;         // n_tile_base = blockIdx.x * 8
    setp.ge.u32 %p0, %r4, %r0;
    @%p0 bra DONE;

    // ---- Thread decomposition (TRUE m16n8k32 fragment layout) ----------------
    mov.u32 %r5, %tid.x;
    and.b32 %r6, %r5, 31;         // lane
    shr.u32 %r7, %r5, 5;          // warp_id (0..7)
    and.b32 %r8, %r6, 3;          // tid = lane & 3
    shr.u32 %r9, %r6, 2;          // gid = lane >> 2 (0..7)
    add.u32 %r140, %r9, 8;        // gid + 8
    shl.b32 %r141, %r8, 1;        // col0 = 2*tid

    setp.lt.u32 %p1, %r5, 8;      // header loader
    setp.eq.u32 %p3, %r7, 0;      // warp 0 reduces

    // smem partial slot base (tile 0): 384 + warp*4096 + gid*32 + col0*4
    shl.b32 %r15, %r7, 12;
    shl.b32 %r142, %r9, 5;
    shl.b32 %r143, %r8, 3;
    add.u32  %r144, %r142, %r143;
    add.u32  %r144, %r144, %r15;
    add.u32  %r144, %r144, 384;
    mov.u32 %r18, smem_buf;
    add.u32 %r17, %r18, %r144;    // s0 base (tile offsets added as immediates)

    shl.b32 %r145, %r8, 2;        // tid*4 (k byte offset in 16-byte half)

    // B column (neuron) for this thread: col = n_tile_base + gid
    add.u32 %r12, %r4, %r9;

    // ---- Zero the 32 register accumulators (8 tiles x 4 D) -------------------
    mov.f32 %f100, 0f00000000;
    mov.f32 %f101, 0f00000000;
    mov.f32 %f102, 0f00000000;
    mov.f32 %f103, 0f00000000;
    mov.f32 %f104, 0f00000000;
    mov.f32 %f105, 0f00000000;
    mov.f32 %f106, 0f00000000;
    mov.f32 %f107, 0f00000000;
    mov.f32 %f108, 0f00000000;
    mov.f32 %f109, 0f00000000;
    mov.f32 %f110, 0f00000000;
    mov.f32 %f111, 0f00000000;
    mov.f32 %f112, 0f00000000;
    mov.f32 %f113, 0f00000000;
    mov.f32 %f114, 0f00000000;
    mov.f32 %f115, 0f00000000;
    mov.f32 %f116, 0f00000000;
    mov.f32 %f117, 0f00000000;
    mov.f32 %f118, 0f00000000;
    mov.f32 %f119, 0f00000000;
    mov.f32 %f120, 0f00000000;
    mov.f32 %f121, 0f00000000;
    mov.f32 %f122, 0f00000000;
    mov.f32 %f123, 0f00000000;
    mov.f32 %f124, 0f00000000;
    mov.f32 %f125, 0f00000000;
    mov.f32 %f126, 0f00000000;
    mov.f32 %f127, 0f00000000;
    mov.f32 %f128, 0f00000000;
    mov.f32 %f129, 0f00000000;
    mov.f32 %f130, 0f00000000;
    mov.f32 %f131, 0f00000000;

    // ---- Outer loop: k_blocks Q4K blocks ------------------------------------
    mov.u32 %r21, 0;               // kb = 0

KB_LOOP:
    setp.ge.u32 %p5, %r21, %r1;
    @%p5 bra KB_DONE;

    // ---- Header load by first 8 threads --------------------------------------
    @!%p1 bra AFTER_HEADER;

    mul.lo.u32 %r22, %r5, %r1;
    mul.lo.u32 %r22, %r22, 144;
    mad.lo.u32 %r22, %r21, 144, %r22;
    mul.lo.u32 %r23, %r4, %r1;
    mul.lo.u32 %r23, %r23, 144;
    add.u32    %r22, %r22, %r23;
    cvt.u64.u32 %rd6, %r22;
    add.u64 %rd6, %rd0, %rd6;

    ld.global.nc.b16 %h0, [%rd6 + 0];
    ld.global.nc.b16 %h1, [%rd6 + 2];
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;

    ld.global.nc.b32 %r24, [%rd6 + 4];
    ld.global.nc.b32 %r25, [%rd6 + 8];
    ld.global.nc.b32 %r26, [%rd6 + 12];

    and.b32 %r27, %r24, 63;
    shr.u32 %r28, %r24, 8;  and.b32 %r28, %r28, 63;
    shr.u32 %r29, %r24, 16; and.b32 %r29, %r29, 63;
    shr.u32 %r30, %r24, 24; and.b32 %r30, %r30, 63;
    and.b32 %r31, %r25, 63;
    shr.u32 %r32, %r25, 8;  and.b32 %r32, %r32, 63;
    shr.u32 %r33, %r25, 16; and.b32 %r33, %r33, 63;
    shr.u32 %r34, %r25, 24; and.b32 %r34, %r34, 63;
    and.b32 %r35, %r26, 255;
    shr.u32 %r36, %r26, 8;  and.b32 %r36, %r36, 255;
    and.b32 %r37, %r24, 192; shr.u32 %r37, %r37, 2;
    shr.u32 %r38, %r24, 8;  and.b32 %r38, %r38, 192; shr.u32 %r38, %r38, 2;
    and.b32 %r39, %r35, 15;  or.b32 %r39, %r39, %r37;
    and.b32 %r40, %r36, 15;  or.b32 %r40, %r40, %r38;
    shr.u32 %r41, %r35, 4;  and.b32 %r41, %r41, 15;
    and.b32 %r42, %r25, 192; shr.u32 %r42, %r42, 2;
    or.b32  %r41, %r41, %r42;
    shr.u32 %r43, %r36, 4;  and.b32 %r43, %r43, 15;
    shr.u32 %r44, %r25, 8;  and.b32 %r44, %r44, 192; shr.u32 %r44, %r44, 2;
    or.b32  %r43, %r43, %r44;
    shr.u32 %r45, %r26, 16; and.b32 %r45, %r45, 255;
    shr.u32 %r46, %r26, 24; and.b32 %r46, %r46, 255;
    shr.u32 %r47, %r24, 16; and.b32 %r47, %r47, 192; shr.u32 %r47, %r47, 2;
    shr.u32 %r48, %r24, 24; and.b32 %r48, %r48, 192; shr.u32 %r48, %r48, 2;
    and.b32 %r49, %r45, 15; or.b32 %r49, %r49, %r47;
    and.b32 %r50, %r46, 15; or.b32 %r50, %r50, %r48;
    shr.u32 %r51, %r45, 4;  and.b32 %r51, %r51, 15;
    shr.u32 %r52, %r25, 16; and.b32 %r52, %r52, 192; shr.u32 %r52, %r52, 2;
    or.b32  %r51, %r51, %r52;
    shr.u32 %r53, %r46, 4;  and.b32 %r53, %r53, 15;
    shr.u32 %r54, %r25, 24; and.b32 %r54, %r54, 192; shr.u32 %r54, %r54, 2;
    or.b32  %r53, %r53, %r54;

    shl.b32 %r55, %r5, 2;
    mov.u32 %r56, smem_buf;
    add.u32 %r57, %r56, %r55;
    st.shared.f32 [%r57 + 0],   %f0;
    st.shared.f32 [%r57 + 128], %f1;

    shl.b32 %r58, %r5, 3;
    add.u32 %r58, %r58, 256;
    add.u32 %r58, %r56, %r58;
    st.shared.u8 [%r58 + 0], %r27;
    st.shared.u8 [%r58 + 1], %r28;
    st.shared.u8 [%r58 + 2], %r29;
    st.shared.u8 [%r58 + 3], %r30;
    st.shared.u8 [%r58 + 4], %r39;
    st.shared.u8 [%r58 + 5], %r40;
    st.shared.u8 [%r58 + 6], %r49;
    st.shared.u8 [%r58 + 7], %r50;
    add.u32 %r59, %r58, 64;
    st.shared.u8 [%r59 + 0], %r31;
    st.shared.u8 [%r59 + 1], %r32;
    st.shared.u8 [%r59 + 2], %r33;
    st.shared.u8 [%r59 + 3], %r34;
    st.shared.u8 [%r59 + 4], %r41;
    st.shared.u8 [%r59 + 5], %r43;
    st.shared.u8 [%r59 + 6], %r51;
    st.shared.u8 [%r59 + 7], %r53;

AFTER_HEADER:
    bar.sync 0;

    // ---- Inner loop: sub-blocks, warp-strided --------------------------------
    mov.u32 %r60, 0;               // mw = 0

MW_LOOP:
    setp.ge.u32 %p6, %r60, 8;
    @%p6 bra MW_DONE;

    mul.lo.u32 %r61, %r21, 8;
    add.u32    %r61, %r61, %r60;   // global_sub = kb*8 + mw
    and.b32    %r62, %r61, 7;
    setp.eq.u32 %p8, %r62, %r7;
    @!%p8 bra MW_NEXT;

    // (kb*8+mw) byte offsets: qs block = *32 bytes, ds = *4 bytes
    shl.b32    %r63, %r61, 5;
    cvt.u64.u32 %rd14, %r63;       // qs block byte offset (u64)
    shl.b32    %r68, %r61, 2;
    cvt.u64.u32 %rd25, %r68;       // ds block byte offset (u64)
    cvt.u64.u32 %rd17, %r145;      // tid*4 (u64)

    // ---- Load Q4K nibbles for B (col = gid) -- ONCE for all 8 tiles ----------
    shr.u32 %r72, %r60, 1;         // byte group j = mw>>1
    shl.b32 %r72, %r72, 5;

    mul.lo.u32 %r74, %r12, %r1;    // neuron * k_blocks
    mul.lo.u32 %r74, %r74, 144;
    mad.lo.u32 %r74, %r21, 144, %r74;
    add.u32    %r74, %r74, 16;
    add.u32    %r74, %r74, %r72;
    add.u32    %r75, %r74, %r145;
    add.u32    %r76, %r75, 16;

    cvt.u64.u32 %rd9,  %r75; add.u64 %rd9,  %rd0, %rd9;
    cvt.u64.u32 %rd10, %r76; add.u64 %rd10, %rd0, %rd10;
    ld.global.nc.b32 %r77, [%rd9];
    ld.global.nc.b32 %r78, [%rd10];

    and.b32 %r130, %r60, 1;
    setp.ne.u32 %p7, %r130, 0;
    mov.u32 %r79, 0x0F0F0F0F;
    and.b32 %r81, %r77, %r79;
    shr.u32 %r82, %r77, 4; and.b32 %r82, %r82, %r79;
    selp.u32 %r83, %r82, %r81, %p7;
    and.b32 %r84, %r78, %r79;
    shr.u32 %r85, %r78, 4; and.b32 %r85, %r85, %r79;
    selp.u32 %r86, %r85, %r84, %p7;

    // ---- Per-column Q4K scales (cols col0, col0+1) -- ONCE for all tiles -----
    shl.b32 %r146, %r141, 2;
    add.u32 %r147, %r18, %r146;
    ld.shared.f32 %f3,  [%r147];
    ld.shared.f32 %f4,  [%r147 + 4];
    ld.shared.f32 %f7,  [%r147 + 128];
    ld.shared.f32 %f8,  [%r147 + 132];

    shl.b32 %r148, %r141, 3;
    add.u32 %r149, %r18, %r148;
    add.u32 %r149, %r149, 256;
    add.u32 %r149, %r149, %r60;
    ld.shared.u8 %r93,  [%r149];
    ld.shared.u8 %r94,  [%r149 + 8];
    add.u32 %r150, %r149, 64;
    ld.shared.u8 %r98,  [%r150];
    ld.shared.u8 %r99,  [%r150 + 8];

    cvt.rn.f32.u32 %f11, %r93;
    cvt.rn.f32.u32 %f12, %r94;
    cvt.rn.f32.u32 %f15, %r98;
    cvt.rn.f32.u32 %f16, %r99;

    // Tile-invariant products: (d*sc) and (dmin*mv) per column.
    mul.f32 %f70, %f3, %f11;       // (d*sc)[col0]
    mul.f32 %f71, %f4, %f12;       // (d*sc)[col0+1]
    mul.f32 %f72, %f7, %f15;       // (dmin*mv)[col0]
    mul.f32 %f73, %f8, %f16;       // (dmin*mv)[col0+1]

    // ---- tile 0: rows 0..15 ------------------------------------
    setp.ge.u32 %p20, 0, %r2;
    @%p20 bra TILE0_END;

    add.u32 %r150, %r9, 0;        // row_lo = gid + 0*16
    add.u32 %r151, %r140, 0;      // row_hi = gid+8 + 0*16
    setp.lt.u32 %p21, %r150, %r2;     // row_lo valid
    setp.lt.u32 %p22, %r151, %r2;     // row_hi valid

    // A qs addresses: x_qs + row*k_blocks*256 + (kb*8+mw)*32 + tid*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd20, %r152;  add.u64 %rd20, %rd20, %rd1;  add.u64 %rd20, %rd20, %rd14;
    add.u64 %rd21, %rd20, %rd17;      // a0 addr (row_lo, k half 0)
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd22, %r152;  add.u64 %rd22, %rd22, %rd1;  add.u64 %rd22, %rd22, %rd14;
    add.u64 %rd23, %rd22, %rd17;      // a1 addr (row_hi, k half 0)

    mov.u32 %r153, 0; mov.u32 %r154, 0; mov.u32 %r155, 0; mov.u32 %r156, 0;
    @%p21 ld.global.nc.b32 %r153, [%rd21];
    @%p22 ld.global.nc.b32 %r154, [%rd23];
    @%p21 ld.global.nc.b32 %r155, [%rd21 + 16];
    @%p22 ld.global.nc.b32 %r156, [%rd23 + 16];

    // d8 scales: x_ds + row*k_blocks*32 + (kb*8+mw)*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd24, %r152;  add.u64 %rd24, %rd24, %rd2;  add.u64 %rd24, %rd24, %rd25;
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd26, %r152;  add.u64 %rd26, %rd26, %rd2;  add.u64 %rd26, %rd26, %rd25;
    mov.b16 %h2, 0;
    mov.b16 %h3, 0;
    @%p21 ld.global.nc.b16 %h2, [%rd24];
    @%p22 ld.global.nc.b16 %h3, [%rd26];
    cvt.f32.f16 %f60, %h2;            // d8 row_lo
    cvt.f32.f16 %f61, %h3;            // d8 row_hi

    // Row sums (min term): bytesum(a0)+bytesum(a2) for row_lo, a1/a3 for row_hi.
    mov.u32 %r157, 0;
    shl.b32 %r158, %r153, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r153, 24;                               add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r155, 24;                               add.s32 %r157, %r157, %r158;
    mov.u32 %r159, 0;
    shl.b32 %r158, %r154, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r154, 24;                               add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r156, 24;                               add.s32 %r159, %r159, %r158;

    shfl.sync.bfly.b32 %r160, %r157, 1, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r157, 2, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 1, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 2, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;

    // Per-tile scale factors: pos/min = (d*sc | dmin*mv) * d8[row].
    mul.f32 %f62, %f70, %f60;   // pos row-lo col0 = (d*sc)[col0] * d8_lo
    mul.f32 %f63, %f71, %f60;   // pos row-lo col1
    mul.f32 %f64, %f70, %f61;   // pos row-hi col0
    mul.f32 %f65, %f71, %f61;   // pos row-hi col1
    mul.f32 %f66, %f72, %f60;   // min row-lo col0 = (dmin*mv)[col0] * d8_lo
    mul.f32 %f67, %f73, %f60;   // min row-lo col1
    mul.f32 %f68, %f72, %f61;   // min row-hi col0
    mul.f32 %f69, %f73, %f61;   // min row-hi col1

    cvt.rn.f32.s32 %f74, %r157;      // rowsum(row_lo)
    cvt.rn.f32.s32 %f75, %r159;      // rowsum(row_hi)

    // MMA: D = A * B (fresh accumulators)
    mov.u32 %r161, 0; mov.u32 %r162, 0;
    mov.u32 %r163, 0; mov.u32 %r164, 0;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r161, %r162, %r163, %r164},
        {%r153, %r154, %r155, %r156},
        {%r83,  %r86},
        {%r161, %r162, %r163, %r164};

    // acc[T] += D * pos - rowsum * min  (same op order as the 16-row kernel)
    cvt.rn.f32.s32 %f76, %r161;
    mul.f32 %f76, %f76, %f62;   mul.f32 %f77, %f74, %f66;   sub.f32 %f76, %f76, %f77;
    add.f32 %f100, %f100, %f76;

    cvt.rn.f32.s32 %f76, %r162;
    mul.f32 %f76, %f76, %f63;   mul.f32 %f77, %f74, %f67;   sub.f32 %f76, %f76, %f77;
    add.f32 %f101, %f101, %f76;

    cvt.rn.f32.s32 %f76, %r163;
    mul.f32 %f76, %f76, %f64;   mul.f32 %f77, %f75, %f68;   sub.f32 %f76, %f76, %f77;
    add.f32 %f102, %f102, %f76;

    cvt.rn.f32.s32 %f76, %r164;
    mul.f32 %f76, %f76, %f65;   mul.f32 %f77, %f75, %f69;   sub.f32 %f76, %f76, %f77;
    add.f32 %f103, %f103, %f76;

TILE0_END:


    // ---- tile 1: rows 16..31 ------------------------------------
    setp.ge.u32 %p20, 16, %r2;
    @%p20 bra TILE1_END;

    add.u32 %r150, %r9, 16;        // row_lo = gid + 1*16
    add.u32 %r151, %r140, 16;      // row_hi = gid+8 + 1*16
    setp.lt.u32 %p21, %r150, %r2;     // row_lo valid
    setp.lt.u32 %p22, %r151, %r2;     // row_hi valid

    // A qs addresses: x_qs + row*k_blocks*256 + (kb*8+mw)*32 + tid*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd20, %r152;  add.u64 %rd20, %rd20, %rd1;  add.u64 %rd20, %rd20, %rd14;
    add.u64 %rd21, %rd20, %rd17;      // a0 addr (row_lo, k half 0)
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd22, %r152;  add.u64 %rd22, %rd22, %rd1;  add.u64 %rd22, %rd22, %rd14;
    add.u64 %rd23, %rd22, %rd17;      // a1 addr (row_hi, k half 0)

    mov.u32 %r153, 0; mov.u32 %r154, 0; mov.u32 %r155, 0; mov.u32 %r156, 0;
    @%p21 ld.global.nc.b32 %r153, [%rd21];
    @%p22 ld.global.nc.b32 %r154, [%rd23];
    @%p21 ld.global.nc.b32 %r155, [%rd21 + 16];
    @%p22 ld.global.nc.b32 %r156, [%rd23 + 16];

    // d8 scales: x_ds + row*k_blocks*32 + (kb*8+mw)*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd24, %r152;  add.u64 %rd24, %rd24, %rd2;  add.u64 %rd24, %rd24, %rd25;
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd26, %r152;  add.u64 %rd26, %rd26, %rd2;  add.u64 %rd26, %rd26, %rd25;
    mov.b16 %h2, 0;
    mov.b16 %h3, 0;
    @%p21 ld.global.nc.b16 %h2, [%rd24];
    @%p22 ld.global.nc.b16 %h3, [%rd26];
    cvt.f32.f16 %f60, %h2;            // d8 row_lo
    cvt.f32.f16 %f61, %h3;            // d8 row_hi

    // Row sums (min term): bytesum(a0)+bytesum(a2) for row_lo, a1/a3 for row_hi.
    mov.u32 %r157, 0;
    shl.b32 %r158, %r153, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r153, 24;                               add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r155, 24;                               add.s32 %r157, %r157, %r158;
    mov.u32 %r159, 0;
    shl.b32 %r158, %r154, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r154, 24;                               add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r156, 24;                               add.s32 %r159, %r159, %r158;

    shfl.sync.bfly.b32 %r160, %r157, 1, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r157, 2, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 1, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 2, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;

    // Per-tile scale factors: pos/min = (d*sc | dmin*mv) * d8[row].
    mul.f32 %f62, %f70, %f60;   // pos row-lo col0 = (d*sc)[col0] * d8_lo
    mul.f32 %f63, %f71, %f60;   // pos row-lo col1
    mul.f32 %f64, %f70, %f61;   // pos row-hi col0
    mul.f32 %f65, %f71, %f61;   // pos row-hi col1
    mul.f32 %f66, %f72, %f60;   // min row-lo col0 = (dmin*mv)[col0] * d8_lo
    mul.f32 %f67, %f73, %f60;   // min row-lo col1
    mul.f32 %f68, %f72, %f61;   // min row-hi col0
    mul.f32 %f69, %f73, %f61;   // min row-hi col1

    cvt.rn.f32.s32 %f74, %r157;      // rowsum(row_lo)
    cvt.rn.f32.s32 %f75, %r159;      // rowsum(row_hi)

    // MMA: D = A * B (fresh accumulators)
    mov.u32 %r161, 0; mov.u32 %r162, 0;
    mov.u32 %r163, 0; mov.u32 %r164, 0;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r161, %r162, %r163, %r164},
        {%r153, %r154, %r155, %r156},
        {%r83,  %r86},
        {%r161, %r162, %r163, %r164};

    // acc[T] += D * pos - rowsum * min  (same op order as the 16-row kernel)
    cvt.rn.f32.s32 %f76, %r161;
    mul.f32 %f76, %f76, %f62;   mul.f32 %f77, %f74, %f66;   sub.f32 %f76, %f76, %f77;
    add.f32 %f104, %f104, %f76;

    cvt.rn.f32.s32 %f76, %r162;
    mul.f32 %f76, %f76, %f63;   mul.f32 %f77, %f74, %f67;   sub.f32 %f76, %f76, %f77;
    add.f32 %f105, %f105, %f76;

    cvt.rn.f32.s32 %f76, %r163;
    mul.f32 %f76, %f76, %f64;   mul.f32 %f77, %f75, %f68;   sub.f32 %f76, %f76, %f77;
    add.f32 %f106, %f106, %f76;

    cvt.rn.f32.s32 %f76, %r164;
    mul.f32 %f76, %f76, %f65;   mul.f32 %f77, %f75, %f69;   sub.f32 %f76, %f76, %f77;
    add.f32 %f107, %f107, %f76;

TILE1_END:


    // ---- tile 2: rows 32..47 ------------------------------------
    setp.ge.u32 %p20, 32, %r2;
    @%p20 bra TILE2_END;

    add.u32 %r150, %r9, 32;        // row_lo = gid + 2*16
    add.u32 %r151, %r140, 32;      // row_hi = gid+8 + 2*16
    setp.lt.u32 %p21, %r150, %r2;     // row_lo valid
    setp.lt.u32 %p22, %r151, %r2;     // row_hi valid

    // A qs addresses: x_qs + row*k_blocks*256 + (kb*8+mw)*32 + tid*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd20, %r152;  add.u64 %rd20, %rd20, %rd1;  add.u64 %rd20, %rd20, %rd14;
    add.u64 %rd21, %rd20, %rd17;      // a0 addr (row_lo, k half 0)
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd22, %r152;  add.u64 %rd22, %rd22, %rd1;  add.u64 %rd22, %rd22, %rd14;
    add.u64 %rd23, %rd22, %rd17;      // a1 addr (row_hi, k half 0)

    mov.u32 %r153, 0; mov.u32 %r154, 0; mov.u32 %r155, 0; mov.u32 %r156, 0;
    @%p21 ld.global.nc.b32 %r153, [%rd21];
    @%p22 ld.global.nc.b32 %r154, [%rd23];
    @%p21 ld.global.nc.b32 %r155, [%rd21 + 16];
    @%p22 ld.global.nc.b32 %r156, [%rd23 + 16];

    // d8 scales: x_ds + row*k_blocks*32 + (kb*8+mw)*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd24, %r152;  add.u64 %rd24, %rd24, %rd2;  add.u64 %rd24, %rd24, %rd25;
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd26, %r152;  add.u64 %rd26, %rd26, %rd2;  add.u64 %rd26, %rd26, %rd25;
    mov.b16 %h2, 0;
    mov.b16 %h3, 0;
    @%p21 ld.global.nc.b16 %h2, [%rd24];
    @%p22 ld.global.nc.b16 %h3, [%rd26];
    cvt.f32.f16 %f60, %h2;            // d8 row_lo
    cvt.f32.f16 %f61, %h3;            // d8 row_hi

    // Row sums (min term): bytesum(a0)+bytesum(a2) for row_lo, a1/a3 for row_hi.
    mov.u32 %r157, 0;
    shl.b32 %r158, %r153, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r153, 24;                               add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r155, 24;                               add.s32 %r157, %r157, %r158;
    mov.u32 %r159, 0;
    shl.b32 %r158, %r154, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r154, 24;                               add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r156, 24;                               add.s32 %r159, %r159, %r158;

    shfl.sync.bfly.b32 %r160, %r157, 1, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r157, 2, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 1, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 2, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;

    // Per-tile scale factors: pos/min = (d*sc | dmin*mv) * d8[row].
    mul.f32 %f62, %f70, %f60;   // pos row-lo col0 = (d*sc)[col0] * d8_lo
    mul.f32 %f63, %f71, %f60;   // pos row-lo col1
    mul.f32 %f64, %f70, %f61;   // pos row-hi col0
    mul.f32 %f65, %f71, %f61;   // pos row-hi col1
    mul.f32 %f66, %f72, %f60;   // min row-lo col0 = (dmin*mv)[col0] * d8_lo
    mul.f32 %f67, %f73, %f60;   // min row-lo col1
    mul.f32 %f68, %f72, %f61;   // min row-hi col0
    mul.f32 %f69, %f73, %f61;   // min row-hi col1

    cvt.rn.f32.s32 %f74, %r157;      // rowsum(row_lo)
    cvt.rn.f32.s32 %f75, %r159;      // rowsum(row_hi)

    // MMA: D = A * B (fresh accumulators)
    mov.u32 %r161, 0; mov.u32 %r162, 0;
    mov.u32 %r163, 0; mov.u32 %r164, 0;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r161, %r162, %r163, %r164},
        {%r153, %r154, %r155, %r156},
        {%r83,  %r86},
        {%r161, %r162, %r163, %r164};

    // acc[T] += D * pos - rowsum * min  (same op order as the 16-row kernel)
    cvt.rn.f32.s32 %f76, %r161;
    mul.f32 %f76, %f76, %f62;   mul.f32 %f77, %f74, %f66;   sub.f32 %f76, %f76, %f77;
    add.f32 %f108, %f108, %f76;

    cvt.rn.f32.s32 %f76, %r162;
    mul.f32 %f76, %f76, %f63;   mul.f32 %f77, %f74, %f67;   sub.f32 %f76, %f76, %f77;
    add.f32 %f109, %f109, %f76;

    cvt.rn.f32.s32 %f76, %r163;
    mul.f32 %f76, %f76, %f64;   mul.f32 %f77, %f75, %f68;   sub.f32 %f76, %f76, %f77;
    add.f32 %f110, %f110, %f76;

    cvt.rn.f32.s32 %f76, %r164;
    mul.f32 %f76, %f76, %f65;   mul.f32 %f77, %f75, %f69;   sub.f32 %f76, %f76, %f77;
    add.f32 %f111, %f111, %f76;

TILE2_END:


    // ---- tile 3: rows 48..63 ------------------------------------
    setp.ge.u32 %p20, 48, %r2;
    @%p20 bra TILE3_END;

    add.u32 %r150, %r9, 48;        // row_lo = gid + 3*16
    add.u32 %r151, %r140, 48;      // row_hi = gid+8 + 3*16
    setp.lt.u32 %p21, %r150, %r2;     // row_lo valid
    setp.lt.u32 %p22, %r151, %r2;     // row_hi valid

    // A qs addresses: x_qs + row*k_blocks*256 + (kb*8+mw)*32 + tid*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd20, %r152;  add.u64 %rd20, %rd20, %rd1;  add.u64 %rd20, %rd20, %rd14;
    add.u64 %rd21, %rd20, %rd17;      // a0 addr (row_lo, k half 0)
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd22, %r152;  add.u64 %rd22, %rd22, %rd1;  add.u64 %rd22, %rd22, %rd14;
    add.u64 %rd23, %rd22, %rd17;      // a1 addr (row_hi, k half 0)

    mov.u32 %r153, 0; mov.u32 %r154, 0; mov.u32 %r155, 0; mov.u32 %r156, 0;
    @%p21 ld.global.nc.b32 %r153, [%rd21];
    @%p22 ld.global.nc.b32 %r154, [%rd23];
    @%p21 ld.global.nc.b32 %r155, [%rd21 + 16];
    @%p22 ld.global.nc.b32 %r156, [%rd23 + 16];

    // d8 scales: x_ds + row*k_blocks*32 + (kb*8+mw)*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd24, %r152;  add.u64 %rd24, %rd24, %rd2;  add.u64 %rd24, %rd24, %rd25;
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd26, %r152;  add.u64 %rd26, %rd26, %rd2;  add.u64 %rd26, %rd26, %rd25;
    mov.b16 %h2, 0;
    mov.b16 %h3, 0;
    @%p21 ld.global.nc.b16 %h2, [%rd24];
    @%p22 ld.global.nc.b16 %h3, [%rd26];
    cvt.f32.f16 %f60, %h2;            // d8 row_lo
    cvt.f32.f16 %f61, %h3;            // d8 row_hi

    // Row sums (min term): bytesum(a0)+bytesum(a2) for row_lo, a1/a3 for row_hi.
    mov.u32 %r157, 0;
    shl.b32 %r158, %r153, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r153, 24;                               add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r155, 24;                               add.s32 %r157, %r157, %r158;
    mov.u32 %r159, 0;
    shl.b32 %r158, %r154, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r154, 24;                               add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r156, 24;                               add.s32 %r159, %r159, %r158;

    shfl.sync.bfly.b32 %r160, %r157, 1, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r157, 2, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 1, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 2, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;

    // Per-tile scale factors: pos/min = (d*sc | dmin*mv) * d8[row].
    mul.f32 %f62, %f70, %f60;   // pos row-lo col0 = (d*sc)[col0] * d8_lo
    mul.f32 %f63, %f71, %f60;   // pos row-lo col1
    mul.f32 %f64, %f70, %f61;   // pos row-hi col0
    mul.f32 %f65, %f71, %f61;   // pos row-hi col1
    mul.f32 %f66, %f72, %f60;   // min row-lo col0 = (dmin*mv)[col0] * d8_lo
    mul.f32 %f67, %f73, %f60;   // min row-lo col1
    mul.f32 %f68, %f72, %f61;   // min row-hi col0
    mul.f32 %f69, %f73, %f61;   // min row-hi col1

    cvt.rn.f32.s32 %f74, %r157;      // rowsum(row_lo)
    cvt.rn.f32.s32 %f75, %r159;      // rowsum(row_hi)

    // MMA: D = A * B (fresh accumulators)
    mov.u32 %r161, 0; mov.u32 %r162, 0;
    mov.u32 %r163, 0; mov.u32 %r164, 0;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r161, %r162, %r163, %r164},
        {%r153, %r154, %r155, %r156},
        {%r83,  %r86},
        {%r161, %r162, %r163, %r164};

    // acc[T] += D * pos - rowsum * min  (same op order as the 16-row kernel)
    cvt.rn.f32.s32 %f76, %r161;
    mul.f32 %f76, %f76, %f62;   mul.f32 %f77, %f74, %f66;   sub.f32 %f76, %f76, %f77;
    add.f32 %f112, %f112, %f76;

    cvt.rn.f32.s32 %f76, %r162;
    mul.f32 %f76, %f76, %f63;   mul.f32 %f77, %f74, %f67;   sub.f32 %f76, %f76, %f77;
    add.f32 %f113, %f113, %f76;

    cvt.rn.f32.s32 %f76, %r163;
    mul.f32 %f76, %f76, %f64;   mul.f32 %f77, %f75, %f68;   sub.f32 %f76, %f76, %f77;
    add.f32 %f114, %f114, %f76;

    cvt.rn.f32.s32 %f76, %r164;
    mul.f32 %f76, %f76, %f65;   mul.f32 %f77, %f75, %f69;   sub.f32 %f76, %f76, %f77;
    add.f32 %f115, %f115, %f76;

TILE3_END:


    // ---- tile 4: rows 64..79 ------------------------------------
    setp.ge.u32 %p20, 64, %r2;
    @%p20 bra TILE4_END;

    add.u32 %r150, %r9, 64;        // row_lo = gid + 4*16
    add.u32 %r151, %r140, 64;      // row_hi = gid+8 + 4*16
    setp.lt.u32 %p21, %r150, %r2;     // row_lo valid
    setp.lt.u32 %p22, %r151, %r2;     // row_hi valid

    // A qs addresses: x_qs + row*k_blocks*256 + (kb*8+mw)*32 + tid*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd20, %r152;  add.u64 %rd20, %rd20, %rd1;  add.u64 %rd20, %rd20, %rd14;
    add.u64 %rd21, %rd20, %rd17;      // a0 addr (row_lo, k half 0)
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd22, %r152;  add.u64 %rd22, %rd22, %rd1;  add.u64 %rd22, %rd22, %rd14;
    add.u64 %rd23, %rd22, %rd17;      // a1 addr (row_hi, k half 0)

    mov.u32 %r153, 0; mov.u32 %r154, 0; mov.u32 %r155, 0; mov.u32 %r156, 0;
    @%p21 ld.global.nc.b32 %r153, [%rd21];
    @%p22 ld.global.nc.b32 %r154, [%rd23];
    @%p21 ld.global.nc.b32 %r155, [%rd21 + 16];
    @%p22 ld.global.nc.b32 %r156, [%rd23 + 16];

    // d8 scales: x_ds + row*k_blocks*32 + (kb*8+mw)*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd24, %r152;  add.u64 %rd24, %rd24, %rd2;  add.u64 %rd24, %rd24, %rd25;
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd26, %r152;  add.u64 %rd26, %rd26, %rd2;  add.u64 %rd26, %rd26, %rd25;
    mov.b16 %h2, 0;
    mov.b16 %h3, 0;
    @%p21 ld.global.nc.b16 %h2, [%rd24];
    @%p22 ld.global.nc.b16 %h3, [%rd26];
    cvt.f32.f16 %f60, %h2;            // d8 row_lo
    cvt.f32.f16 %f61, %h3;            // d8 row_hi

    // Row sums (min term): bytesum(a0)+bytesum(a2) for row_lo, a1/a3 for row_hi.
    mov.u32 %r157, 0;
    shl.b32 %r158, %r153, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r153, 24;                               add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r155, 24;                               add.s32 %r157, %r157, %r158;
    mov.u32 %r159, 0;
    shl.b32 %r158, %r154, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r154, 24;                               add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r156, 24;                               add.s32 %r159, %r159, %r158;

    shfl.sync.bfly.b32 %r160, %r157, 1, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r157, 2, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 1, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 2, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;

    // Per-tile scale factors: pos/min = (d*sc | dmin*mv) * d8[row].
    mul.f32 %f62, %f70, %f60;   // pos row-lo col0 = (d*sc)[col0] * d8_lo
    mul.f32 %f63, %f71, %f60;   // pos row-lo col1
    mul.f32 %f64, %f70, %f61;   // pos row-hi col0
    mul.f32 %f65, %f71, %f61;   // pos row-hi col1
    mul.f32 %f66, %f72, %f60;   // min row-lo col0 = (dmin*mv)[col0] * d8_lo
    mul.f32 %f67, %f73, %f60;   // min row-lo col1
    mul.f32 %f68, %f72, %f61;   // min row-hi col0
    mul.f32 %f69, %f73, %f61;   // min row-hi col1

    cvt.rn.f32.s32 %f74, %r157;      // rowsum(row_lo)
    cvt.rn.f32.s32 %f75, %r159;      // rowsum(row_hi)

    // MMA: D = A * B (fresh accumulators)
    mov.u32 %r161, 0; mov.u32 %r162, 0;
    mov.u32 %r163, 0; mov.u32 %r164, 0;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r161, %r162, %r163, %r164},
        {%r153, %r154, %r155, %r156},
        {%r83,  %r86},
        {%r161, %r162, %r163, %r164};

    // acc[T] += D * pos - rowsum * min  (same op order as the 16-row kernel)
    cvt.rn.f32.s32 %f76, %r161;
    mul.f32 %f76, %f76, %f62;   mul.f32 %f77, %f74, %f66;   sub.f32 %f76, %f76, %f77;
    add.f32 %f116, %f116, %f76;

    cvt.rn.f32.s32 %f76, %r162;
    mul.f32 %f76, %f76, %f63;   mul.f32 %f77, %f74, %f67;   sub.f32 %f76, %f76, %f77;
    add.f32 %f117, %f117, %f76;

    cvt.rn.f32.s32 %f76, %r163;
    mul.f32 %f76, %f76, %f64;   mul.f32 %f77, %f75, %f68;   sub.f32 %f76, %f76, %f77;
    add.f32 %f118, %f118, %f76;

    cvt.rn.f32.s32 %f76, %r164;
    mul.f32 %f76, %f76, %f65;   mul.f32 %f77, %f75, %f69;   sub.f32 %f76, %f76, %f77;
    add.f32 %f119, %f119, %f76;

TILE4_END:


    // ---- tile 5: rows 80..95 ------------------------------------
    setp.ge.u32 %p20, 80, %r2;
    @%p20 bra TILE5_END;

    add.u32 %r150, %r9, 80;        // row_lo = gid + 5*16
    add.u32 %r151, %r140, 80;      // row_hi = gid+8 + 5*16
    setp.lt.u32 %p21, %r150, %r2;     // row_lo valid
    setp.lt.u32 %p22, %r151, %r2;     // row_hi valid

    // A qs addresses: x_qs + row*k_blocks*256 + (kb*8+mw)*32 + tid*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd20, %r152;  add.u64 %rd20, %rd20, %rd1;  add.u64 %rd20, %rd20, %rd14;
    add.u64 %rd21, %rd20, %rd17;      // a0 addr (row_lo, k half 0)
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd22, %r152;  add.u64 %rd22, %rd22, %rd1;  add.u64 %rd22, %rd22, %rd14;
    add.u64 %rd23, %rd22, %rd17;      // a1 addr (row_hi, k half 0)

    mov.u32 %r153, 0; mov.u32 %r154, 0; mov.u32 %r155, 0; mov.u32 %r156, 0;
    @%p21 ld.global.nc.b32 %r153, [%rd21];
    @%p22 ld.global.nc.b32 %r154, [%rd23];
    @%p21 ld.global.nc.b32 %r155, [%rd21 + 16];
    @%p22 ld.global.nc.b32 %r156, [%rd23 + 16];

    // d8 scales: x_ds + row*k_blocks*32 + (kb*8+mw)*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd24, %r152;  add.u64 %rd24, %rd24, %rd2;  add.u64 %rd24, %rd24, %rd25;
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd26, %r152;  add.u64 %rd26, %rd26, %rd2;  add.u64 %rd26, %rd26, %rd25;
    mov.b16 %h2, 0;
    mov.b16 %h3, 0;
    @%p21 ld.global.nc.b16 %h2, [%rd24];
    @%p22 ld.global.nc.b16 %h3, [%rd26];
    cvt.f32.f16 %f60, %h2;            // d8 row_lo
    cvt.f32.f16 %f61, %h3;            // d8 row_hi

    // Row sums (min term): bytesum(a0)+bytesum(a2) for row_lo, a1/a3 for row_hi.
    mov.u32 %r157, 0;
    shl.b32 %r158, %r153, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r153, 24;                               add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r155, 24;                               add.s32 %r157, %r157, %r158;
    mov.u32 %r159, 0;
    shl.b32 %r158, %r154, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r154, 24;                               add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r156, 24;                               add.s32 %r159, %r159, %r158;

    shfl.sync.bfly.b32 %r160, %r157, 1, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r157, 2, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 1, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 2, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;

    // Per-tile scale factors: pos/min = (d*sc | dmin*mv) * d8[row].
    mul.f32 %f62, %f70, %f60;   // pos row-lo col0 = (d*sc)[col0] * d8_lo
    mul.f32 %f63, %f71, %f60;   // pos row-lo col1
    mul.f32 %f64, %f70, %f61;   // pos row-hi col0
    mul.f32 %f65, %f71, %f61;   // pos row-hi col1
    mul.f32 %f66, %f72, %f60;   // min row-lo col0 = (dmin*mv)[col0] * d8_lo
    mul.f32 %f67, %f73, %f60;   // min row-lo col1
    mul.f32 %f68, %f72, %f61;   // min row-hi col0
    mul.f32 %f69, %f73, %f61;   // min row-hi col1

    cvt.rn.f32.s32 %f74, %r157;      // rowsum(row_lo)
    cvt.rn.f32.s32 %f75, %r159;      // rowsum(row_hi)

    // MMA: D = A * B (fresh accumulators)
    mov.u32 %r161, 0; mov.u32 %r162, 0;
    mov.u32 %r163, 0; mov.u32 %r164, 0;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r161, %r162, %r163, %r164},
        {%r153, %r154, %r155, %r156},
        {%r83,  %r86},
        {%r161, %r162, %r163, %r164};

    // acc[T] += D * pos - rowsum * min  (same op order as the 16-row kernel)
    cvt.rn.f32.s32 %f76, %r161;
    mul.f32 %f76, %f76, %f62;   mul.f32 %f77, %f74, %f66;   sub.f32 %f76, %f76, %f77;
    add.f32 %f120, %f120, %f76;

    cvt.rn.f32.s32 %f76, %r162;
    mul.f32 %f76, %f76, %f63;   mul.f32 %f77, %f74, %f67;   sub.f32 %f76, %f76, %f77;
    add.f32 %f121, %f121, %f76;

    cvt.rn.f32.s32 %f76, %r163;
    mul.f32 %f76, %f76, %f64;   mul.f32 %f77, %f75, %f68;   sub.f32 %f76, %f76, %f77;
    add.f32 %f122, %f122, %f76;

    cvt.rn.f32.s32 %f76, %r164;
    mul.f32 %f76, %f76, %f65;   mul.f32 %f77, %f75, %f69;   sub.f32 %f76, %f76, %f77;
    add.f32 %f123, %f123, %f76;

TILE5_END:


    // ---- tile 6: rows 96..111 ------------------------------------
    setp.ge.u32 %p20, 96, %r2;
    @%p20 bra TILE6_END;

    add.u32 %r150, %r9, 96;        // row_lo = gid + 6*16
    add.u32 %r151, %r140, 96;      // row_hi = gid+8 + 6*16
    setp.lt.u32 %p21, %r150, %r2;     // row_lo valid
    setp.lt.u32 %p22, %r151, %r2;     // row_hi valid

    // A qs addresses: x_qs + row*k_blocks*256 + (kb*8+mw)*32 + tid*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd20, %r152;  add.u64 %rd20, %rd20, %rd1;  add.u64 %rd20, %rd20, %rd14;
    add.u64 %rd21, %rd20, %rd17;      // a0 addr (row_lo, k half 0)
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd22, %r152;  add.u64 %rd22, %rd22, %rd1;  add.u64 %rd22, %rd22, %rd14;
    add.u64 %rd23, %rd22, %rd17;      // a1 addr (row_hi, k half 0)

    mov.u32 %r153, 0; mov.u32 %r154, 0; mov.u32 %r155, 0; mov.u32 %r156, 0;
    @%p21 ld.global.nc.b32 %r153, [%rd21];
    @%p22 ld.global.nc.b32 %r154, [%rd23];
    @%p21 ld.global.nc.b32 %r155, [%rd21 + 16];
    @%p22 ld.global.nc.b32 %r156, [%rd23 + 16];

    // d8 scales: x_ds + row*k_blocks*32 + (kb*8+mw)*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd24, %r152;  add.u64 %rd24, %rd24, %rd2;  add.u64 %rd24, %rd24, %rd25;
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd26, %r152;  add.u64 %rd26, %rd26, %rd2;  add.u64 %rd26, %rd26, %rd25;
    mov.b16 %h2, 0;
    mov.b16 %h3, 0;
    @%p21 ld.global.nc.b16 %h2, [%rd24];
    @%p22 ld.global.nc.b16 %h3, [%rd26];
    cvt.f32.f16 %f60, %h2;            // d8 row_lo
    cvt.f32.f16 %f61, %h3;            // d8 row_hi

    // Row sums (min term): bytesum(a0)+bytesum(a2) for row_lo, a1/a3 for row_hi.
    mov.u32 %r157, 0;
    shl.b32 %r158, %r153, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r153, 24;                               add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r155, 24;                               add.s32 %r157, %r157, %r158;
    mov.u32 %r159, 0;
    shl.b32 %r158, %r154, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r154, 24;                               add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r156, 24;                               add.s32 %r159, %r159, %r158;

    shfl.sync.bfly.b32 %r160, %r157, 1, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r157, 2, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 1, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 2, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;

    // Per-tile scale factors: pos/min = (d*sc | dmin*mv) * d8[row].
    mul.f32 %f62, %f70, %f60;   // pos row-lo col0 = (d*sc)[col0] * d8_lo
    mul.f32 %f63, %f71, %f60;   // pos row-lo col1
    mul.f32 %f64, %f70, %f61;   // pos row-hi col0
    mul.f32 %f65, %f71, %f61;   // pos row-hi col1
    mul.f32 %f66, %f72, %f60;   // min row-lo col0 = (dmin*mv)[col0] * d8_lo
    mul.f32 %f67, %f73, %f60;   // min row-lo col1
    mul.f32 %f68, %f72, %f61;   // min row-hi col0
    mul.f32 %f69, %f73, %f61;   // min row-hi col1

    cvt.rn.f32.s32 %f74, %r157;      // rowsum(row_lo)
    cvt.rn.f32.s32 %f75, %r159;      // rowsum(row_hi)

    // MMA: D = A * B (fresh accumulators)
    mov.u32 %r161, 0; mov.u32 %r162, 0;
    mov.u32 %r163, 0; mov.u32 %r164, 0;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r161, %r162, %r163, %r164},
        {%r153, %r154, %r155, %r156},
        {%r83,  %r86},
        {%r161, %r162, %r163, %r164};

    // acc[T] += D * pos - rowsum * min  (same op order as the 16-row kernel)
    cvt.rn.f32.s32 %f76, %r161;
    mul.f32 %f76, %f76, %f62;   mul.f32 %f77, %f74, %f66;   sub.f32 %f76, %f76, %f77;
    add.f32 %f124, %f124, %f76;

    cvt.rn.f32.s32 %f76, %r162;
    mul.f32 %f76, %f76, %f63;   mul.f32 %f77, %f74, %f67;   sub.f32 %f76, %f76, %f77;
    add.f32 %f125, %f125, %f76;

    cvt.rn.f32.s32 %f76, %r163;
    mul.f32 %f76, %f76, %f64;   mul.f32 %f77, %f75, %f68;   sub.f32 %f76, %f76, %f77;
    add.f32 %f126, %f126, %f76;

    cvt.rn.f32.s32 %f76, %r164;
    mul.f32 %f76, %f76, %f65;   mul.f32 %f77, %f75, %f69;   sub.f32 %f76, %f76, %f77;
    add.f32 %f127, %f127, %f76;

TILE6_END:


    // ---- tile 7: rows 112..127 ------------------------------------
    setp.ge.u32 %p20, 112, %r2;
    @%p20 bra TILE7_END;

    add.u32 %r150, %r9, 112;        // row_lo = gid + 7*16
    add.u32 %r151, %r140, 112;      // row_hi = gid+8 + 7*16
    setp.lt.u32 %p21, %r150, %r2;     // row_lo valid
    setp.lt.u32 %p22, %r151, %r2;     // row_hi valid

    // A qs addresses: x_qs + row*k_blocks*256 + (kb*8+mw)*32 + tid*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd20, %r152;  add.u64 %rd20, %rd20, %rd1;  add.u64 %rd20, %rd20, %rd14;
    add.u64 %rd21, %rd20, %rd17;      // a0 addr (row_lo, k half 0)
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd22, %r152;  add.u64 %rd22, %rd22, %rd1;  add.u64 %rd22, %rd22, %rd14;
    add.u64 %rd23, %rd22, %rd17;      // a1 addr (row_hi, k half 0)

    mov.u32 %r153, 0; mov.u32 %r154, 0; mov.u32 %r155, 0; mov.u32 %r156, 0;
    @%p21 ld.global.nc.b32 %r153, [%rd21];
    @%p22 ld.global.nc.b32 %r154, [%rd23];
    @%p21 ld.global.nc.b32 %r155, [%rd21 + 16];
    @%p22 ld.global.nc.b32 %r156, [%rd23 + 16];

    // d8 scales: x_ds + row*k_blocks*32 + (kb*8+mw)*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd24, %r152;  add.u64 %rd24, %rd24, %rd2;  add.u64 %rd24, %rd24, %rd25;
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd26, %r152;  add.u64 %rd26, %rd26, %rd2;  add.u64 %rd26, %rd26, %rd25;
    mov.b16 %h2, 0;
    mov.b16 %h3, 0;
    @%p21 ld.global.nc.b16 %h2, [%rd24];
    @%p22 ld.global.nc.b16 %h3, [%rd26];
    cvt.f32.f16 %f60, %h2;            // d8 row_lo
    cvt.f32.f16 %f61, %h3;            // d8 row_hi

    // Row sums (min term): bytesum(a0)+bytesum(a2) for row_lo, a1/a3 for row_hi.
    mov.u32 %r157, 0;
    shl.b32 %r158, %r153, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r153, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r153, 24;                               add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 24; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 16; shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shl.b32 %r158, %r155, 8;  shr.s32 %r158, %r158, 24; add.s32 %r157, %r157, %r158;
    shr.s32 %r158, %r155, 24;                               add.s32 %r157, %r157, %r158;
    mov.u32 %r159, 0;
    shl.b32 %r158, %r154, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r154, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r154, 24;                               add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 24; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 16; shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shl.b32 %r158, %r156, 8;  shr.s32 %r158, %r158, 24; add.s32 %r159, %r159, %r158;
    shr.s32 %r158, %r156, 24;                               add.s32 %r159, %r159, %r158;

    shfl.sync.bfly.b32 %r160, %r157, 1, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r157, 2, 31, 0xFFFFFFFF;
    add.s32 %r157, %r157, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 1, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;
    shfl.sync.bfly.b32 %r160, %r159, 2, 31, 0xFFFFFFFF;
    add.s32 %r159, %r159, %r160;

    // Per-tile scale factors: pos/min = (d*sc | dmin*mv) * d8[row].
    mul.f32 %f62, %f70, %f60;   // pos row-lo col0 = (d*sc)[col0] * d8_lo
    mul.f32 %f63, %f71, %f60;   // pos row-lo col1
    mul.f32 %f64, %f70, %f61;   // pos row-hi col0
    mul.f32 %f65, %f71, %f61;   // pos row-hi col1
    mul.f32 %f66, %f72, %f60;   // min row-lo col0 = (dmin*mv)[col0] * d8_lo
    mul.f32 %f67, %f73, %f60;   // min row-lo col1
    mul.f32 %f68, %f72, %f61;   // min row-hi col0
    mul.f32 %f69, %f73, %f61;   // min row-hi col1

    cvt.rn.f32.s32 %f74, %r157;      // rowsum(row_lo)
    cvt.rn.f32.s32 %f75, %r159;      // rowsum(row_hi)

    // MMA: D = A * B (fresh accumulators)
    mov.u32 %r161, 0; mov.u32 %r162, 0;
    mov.u32 %r163, 0; mov.u32 %r164, 0;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r161, %r162, %r163, %r164},
        {%r153, %r154, %r155, %r156},
        {%r83,  %r86},
        {%r161, %r162, %r163, %r164};

    // acc[T] += D * pos - rowsum * min  (same op order as the 16-row kernel)
    cvt.rn.f32.s32 %f76, %r161;
    mul.f32 %f76, %f76, %f62;   mul.f32 %f77, %f74, %f66;   sub.f32 %f76, %f76, %f77;
    add.f32 %f128, %f128, %f76;

    cvt.rn.f32.s32 %f76, %r162;
    mul.f32 %f76, %f76, %f63;   mul.f32 %f77, %f74, %f67;   sub.f32 %f76, %f76, %f77;
    add.f32 %f129, %f129, %f76;

    cvt.rn.f32.s32 %f76, %r163;
    mul.f32 %f76, %f76, %f64;   mul.f32 %f77, %f75, %f68;   sub.f32 %f76, %f76, %f77;
    add.f32 %f130, %f130, %f76;

    cvt.rn.f32.s32 %f76, %r164;
    mul.f32 %f76, %f76, %f65;   mul.f32 %f77, %f75, %f69;   sub.f32 %f76, %f76, %f77;
    add.f32 %f131, %f131, %f76;

TILE7_END:

MW_NEXT:
    add.u32 %r60, %r60, 1;
    bra MW_LOOP;

MW_DONE:
    bar.sync 0;
    add.u32 %r21, %r21, 1;
    bra KB_LOOP;

KB_DONE:
    bar.sync 0;

    // ---- tile 0 reduction ----
    setp.ge.u32 %p20, 0, %r2;
    @%p20 bra TILE0_RDONE;
    st.shared.f32 [%r17 + 0 + 0],   %f100;
    st.shared.f32 [%r17 + 0 + 4],   %f101;
    st.shared.f32 [%r17 + 0 + 256], %f102;
    st.shared.f32 [%r17 + 0 + 260], %f103;
    bar.sync 0;
    @!%p3 bra TILE0_RBYPASS;
    ld.shared.f32 %f50, [%r17 + 0 + 0];
    ld.shared.f32 %f51, [%r17 + 0 + 4096];
    ld.shared.f32 %f52, [%r17 + 0 + 8192];
    ld.shared.f32 %f53, [%r17 + 0 + 12288];
    ld.shared.f32 %f54, [%r17 + 0 + 16384];
    ld.shared.f32 %f55, [%r17 + 0 + 20480];
    ld.shared.f32 %f56, [%r17 + 0 + 24576];
    ld.shared.f32 %f57, [%r17 + 0 + 28672];
    add.f32 %f50, %f50, %f51; add.f32 %f50, %f50, %f52; add.f32 %f50, %f50, %f53;
    add.f32 %f50, %f50, %f54; add.f32 %f50, %f50, %f55; add.f32 %f50, %f50, %f56;
    add.f32 %f50, %f50, %f57;

    ld.shared.f32 %f58, [%r17 + 0 + 4];
    ld.shared.f32 %f59, [%r17 + 0 + 4100];
    ld.shared.f32 %f60, [%r17 + 0 + 8196];
    ld.shared.f32 %f61, [%r17 + 0 + 12292];
    ld.shared.f32 %f62, [%r17 + 0 + 16388];
    ld.shared.f32 %f63, [%r17 + 0 + 20484];
    ld.shared.f32 %f64, [%r17 + 0 + 24580];
    ld.shared.f32 %f65, [%r17 + 0 + 28676];
    add.f32 %f58, %f58, %f59; add.f32 %f58, %f58, %f60; add.f32 %f58, %f58, %f61;
    add.f32 %f58, %f58, %f62; add.f32 %f58, %f58, %f63; add.f32 %f58, %f58, %f64;
    add.f32 %f58, %f58, %f65;

    ld.shared.f32 %f62, [%r17 + 0 + 256];
    ld.shared.f32 %f63, [%r17 + 0 + 4352];
    ld.shared.f32 %f64, [%r17 + 0 + 8448];
    ld.shared.f32 %f65, [%r17 + 0 + 12544];
    ld.shared.f32 %f66, [%r17 + 0 + 16640];
    ld.shared.f32 %f67, [%r17 + 0 + 20736];
    ld.shared.f32 %f68, [%r17 + 0 + 24832];
    ld.shared.f32 %f69, [%r17 + 0 + 28928];
    add.f32 %f62, %f62, %f63; add.f32 %f62, %f62, %f64; add.f32 %f62, %f62, %f65;
    add.f32 %f62, %f62, %f66; add.f32 %f62, %f62, %f67; add.f32 %f62, %f62, %f68;
    add.f32 %f62, %f62, %f69;

    ld.shared.f32 %f66, [%r17 + 0 + 260];
    ld.shared.f32 %f67, [%r17 + 0 + 4356];
    ld.shared.f32 %f68, [%r17 + 0 + 8452];
    ld.shared.f32 %f69, [%r17 + 0 + 12548];
    ld.shared.f32 %f70, [%r17 + 0 + 16644];
    ld.shared.f32 %f71, [%r17 + 0 + 20740];
    ld.shared.f32 %f72, [%r17 + 0 + 24836];
    ld.shared.f32 %f73, [%r17 + 0 + 28932];
    add.f32 %f66, %f66, %f67; add.f32 %f66, %f66, %f68; add.f32 %f66, %f66, %f69;
    add.f32 %f66, %f66, %f70; add.f32 %f66, %f66, %f71; add.f32 %f66, %f66, %f72;
    add.f32 %f66, %f66, %f73;

    add.u32 %r150, %r9, 0;
    add.u32 %r151, %r140, 0;
    setp.lt.u32 %p21, %r150, %r2;
    setp.lt.u32 %p22, %r151, %r2;
    mad.lo.u32 %r106, %r150, %r0, %r4;
    add.u32    %r106, %r106, %r141;
    shl.b32    %r106, %r106, 2;
    cvt.u64.u32 %rd11, %r106;
    add.u64 %rd11, %rd3, %rd11;
    @%p21 st.global.f32 [%rd11 + 0], %f50;
    @%p21 st.global.f32 [%rd11 + 4], %f58;

    mad.lo.u32 %r107, %r151, %r0, %r4;
    add.u32    %r107, %r107, %r141;
    shl.b32    %r107, %r107, 2;
    cvt.u64.u32 %rd16, %r107;
    add.u64 %rd16, %rd3, %rd16;
    @%p22 st.global.f32 [%rd16 + 0], %f62;
    @%p22 st.global.f32 [%rd16 + 4], %f66;

TILE0_RBYPASS:
    bar.sync 0;
TILE0_RDONE:


    // ---- tile 1 reduction ----
    setp.ge.u32 %p20, 16, %r2;
    @%p20 bra TILE1_RDONE;
    st.shared.f32 [%r17 + 512 + 0],   %f104;
    st.shared.f32 [%r17 + 512 + 4],   %f105;
    st.shared.f32 [%r17 + 512 + 256], %f106;
    st.shared.f32 [%r17 + 512 + 260], %f107;
    bar.sync 0;
    @!%p3 bra TILE1_RBYPASS;
    ld.shared.f32 %f50, [%r17 + 512 + 0];
    ld.shared.f32 %f51, [%r17 + 512 + 4096];
    ld.shared.f32 %f52, [%r17 + 512 + 8192];
    ld.shared.f32 %f53, [%r17 + 512 + 12288];
    ld.shared.f32 %f54, [%r17 + 512 + 16384];
    ld.shared.f32 %f55, [%r17 + 512 + 20480];
    ld.shared.f32 %f56, [%r17 + 512 + 24576];
    ld.shared.f32 %f57, [%r17 + 512 + 28672];
    add.f32 %f50, %f50, %f51; add.f32 %f50, %f50, %f52; add.f32 %f50, %f50, %f53;
    add.f32 %f50, %f50, %f54; add.f32 %f50, %f50, %f55; add.f32 %f50, %f50, %f56;
    add.f32 %f50, %f50, %f57;

    ld.shared.f32 %f58, [%r17 + 512 + 4];
    ld.shared.f32 %f59, [%r17 + 512 + 4100];
    ld.shared.f32 %f60, [%r17 + 512 + 8196];
    ld.shared.f32 %f61, [%r17 + 512 + 12292];
    ld.shared.f32 %f62, [%r17 + 512 + 16388];
    ld.shared.f32 %f63, [%r17 + 512 + 20484];
    ld.shared.f32 %f64, [%r17 + 512 + 24580];
    ld.shared.f32 %f65, [%r17 + 512 + 28676];
    add.f32 %f58, %f58, %f59; add.f32 %f58, %f58, %f60; add.f32 %f58, %f58, %f61;
    add.f32 %f58, %f58, %f62; add.f32 %f58, %f58, %f63; add.f32 %f58, %f58, %f64;
    add.f32 %f58, %f58, %f65;

    ld.shared.f32 %f62, [%r17 + 512 + 256];
    ld.shared.f32 %f63, [%r17 + 512 + 4352];
    ld.shared.f32 %f64, [%r17 + 512 + 8448];
    ld.shared.f32 %f65, [%r17 + 512 + 12544];
    ld.shared.f32 %f66, [%r17 + 512 + 16640];
    ld.shared.f32 %f67, [%r17 + 512 + 20736];
    ld.shared.f32 %f68, [%r17 + 512 + 24832];
    ld.shared.f32 %f69, [%r17 + 512 + 28928];
    add.f32 %f62, %f62, %f63; add.f32 %f62, %f62, %f64; add.f32 %f62, %f62, %f65;
    add.f32 %f62, %f62, %f66; add.f32 %f62, %f62, %f67; add.f32 %f62, %f62, %f68;
    add.f32 %f62, %f62, %f69;

    ld.shared.f32 %f66, [%r17 + 512 + 260];
    ld.shared.f32 %f67, [%r17 + 512 + 4356];
    ld.shared.f32 %f68, [%r17 + 512 + 8452];
    ld.shared.f32 %f69, [%r17 + 512 + 12548];
    ld.shared.f32 %f70, [%r17 + 512 + 16644];
    ld.shared.f32 %f71, [%r17 + 512 + 20740];
    ld.shared.f32 %f72, [%r17 + 512 + 24836];
    ld.shared.f32 %f73, [%r17 + 512 + 28932];
    add.f32 %f66, %f66, %f67; add.f32 %f66, %f66, %f68; add.f32 %f66, %f66, %f69;
    add.f32 %f66, %f66, %f70; add.f32 %f66, %f66, %f71; add.f32 %f66, %f66, %f72;
    add.f32 %f66, %f66, %f73;

    add.u32 %r150, %r9, 16;
    add.u32 %r151, %r140, 16;
    setp.lt.u32 %p21, %r150, %r2;
    setp.lt.u32 %p22, %r151, %r2;
    mad.lo.u32 %r106, %r150, %r0, %r4;
    add.u32    %r106, %r106, %r141;
    shl.b32    %r106, %r106, 2;
    cvt.u64.u32 %rd11, %r106;
    add.u64 %rd11, %rd3, %rd11;
    @%p21 st.global.f32 [%rd11 + 0], %f50;
    @%p21 st.global.f32 [%rd11 + 4], %f58;

    mad.lo.u32 %r107, %r151, %r0, %r4;
    add.u32    %r107, %r107, %r141;
    shl.b32    %r107, %r107, 2;
    cvt.u64.u32 %rd16, %r107;
    add.u64 %rd16, %rd3, %rd16;
    @%p22 st.global.f32 [%rd16 + 0], %f62;
    @%p22 st.global.f32 [%rd16 + 4], %f66;

TILE1_RBYPASS:
    bar.sync 0;
TILE1_RDONE:


    // ---- tile 2 reduction ----
    setp.ge.u32 %p20, 32, %r2;
    @%p20 bra TILE2_RDONE;
    st.shared.f32 [%r17 + 1024 + 0],   %f108;
    st.shared.f32 [%r17 + 1024 + 4],   %f109;
    st.shared.f32 [%r17 + 1024 + 256], %f110;
    st.shared.f32 [%r17 + 1024 + 260], %f111;
    bar.sync 0;
    @!%p3 bra TILE2_RBYPASS;
    ld.shared.f32 %f50, [%r17 + 1024 + 0];
    ld.shared.f32 %f51, [%r17 + 1024 + 4096];
    ld.shared.f32 %f52, [%r17 + 1024 + 8192];
    ld.shared.f32 %f53, [%r17 + 1024 + 12288];
    ld.shared.f32 %f54, [%r17 + 1024 + 16384];
    ld.shared.f32 %f55, [%r17 + 1024 + 20480];
    ld.shared.f32 %f56, [%r17 + 1024 + 24576];
    ld.shared.f32 %f57, [%r17 + 1024 + 28672];
    add.f32 %f50, %f50, %f51; add.f32 %f50, %f50, %f52; add.f32 %f50, %f50, %f53;
    add.f32 %f50, %f50, %f54; add.f32 %f50, %f50, %f55; add.f32 %f50, %f50, %f56;
    add.f32 %f50, %f50, %f57;

    ld.shared.f32 %f58, [%r17 + 1024 + 4];
    ld.shared.f32 %f59, [%r17 + 1024 + 4100];
    ld.shared.f32 %f60, [%r17 + 1024 + 8196];
    ld.shared.f32 %f61, [%r17 + 1024 + 12292];
    ld.shared.f32 %f62, [%r17 + 1024 + 16388];
    ld.shared.f32 %f63, [%r17 + 1024 + 20484];
    ld.shared.f32 %f64, [%r17 + 1024 + 24580];
    ld.shared.f32 %f65, [%r17 + 1024 + 28676];
    add.f32 %f58, %f58, %f59; add.f32 %f58, %f58, %f60; add.f32 %f58, %f58, %f61;
    add.f32 %f58, %f58, %f62; add.f32 %f58, %f58, %f63; add.f32 %f58, %f58, %f64;
    add.f32 %f58, %f58, %f65;

    ld.shared.f32 %f62, [%r17 + 1024 + 256];
    ld.shared.f32 %f63, [%r17 + 1024 + 4352];
    ld.shared.f32 %f64, [%r17 + 1024 + 8448];
    ld.shared.f32 %f65, [%r17 + 1024 + 12544];
    ld.shared.f32 %f66, [%r17 + 1024 + 16640];
    ld.shared.f32 %f67, [%r17 + 1024 + 20736];
    ld.shared.f32 %f68, [%r17 + 1024 + 24832];
    ld.shared.f32 %f69, [%r17 + 1024 + 28928];
    add.f32 %f62, %f62, %f63; add.f32 %f62, %f62, %f64; add.f32 %f62, %f62, %f65;
    add.f32 %f62, %f62, %f66; add.f32 %f62, %f62, %f67; add.f32 %f62, %f62, %f68;
    add.f32 %f62, %f62, %f69;

    ld.shared.f32 %f66, [%r17 + 1024 + 260];
    ld.shared.f32 %f67, [%r17 + 1024 + 4356];
    ld.shared.f32 %f68, [%r17 + 1024 + 8452];
    ld.shared.f32 %f69, [%r17 + 1024 + 12548];
    ld.shared.f32 %f70, [%r17 + 1024 + 16644];
    ld.shared.f32 %f71, [%r17 + 1024 + 20740];
    ld.shared.f32 %f72, [%r17 + 1024 + 24836];
    ld.shared.f32 %f73, [%r17 + 1024 + 28932];
    add.f32 %f66, %f66, %f67; add.f32 %f66, %f66, %f68; add.f32 %f66, %f66, %f69;
    add.f32 %f66, %f66, %f70; add.f32 %f66, %f66, %f71; add.f32 %f66, %f66, %f72;
    add.f32 %f66, %f66, %f73;

    add.u32 %r150, %r9, 32;
    add.u32 %r151, %r140, 32;
    setp.lt.u32 %p21, %r150, %r2;
    setp.lt.u32 %p22, %r151, %r2;
    mad.lo.u32 %r106, %r150, %r0, %r4;
    add.u32    %r106, %r106, %r141;
    shl.b32    %r106, %r106, 2;
    cvt.u64.u32 %rd11, %r106;
    add.u64 %rd11, %rd3, %rd11;
    @%p21 st.global.f32 [%rd11 + 0], %f50;
    @%p21 st.global.f32 [%rd11 + 4], %f58;

    mad.lo.u32 %r107, %r151, %r0, %r4;
    add.u32    %r107, %r107, %r141;
    shl.b32    %r107, %r107, 2;
    cvt.u64.u32 %rd16, %r107;
    add.u64 %rd16, %rd3, %rd16;
    @%p22 st.global.f32 [%rd16 + 0], %f62;
    @%p22 st.global.f32 [%rd16 + 4], %f66;

TILE2_RBYPASS:
    bar.sync 0;
TILE2_RDONE:


    // ---- tile 3 reduction ----
    setp.ge.u32 %p20, 48, %r2;
    @%p20 bra TILE3_RDONE;
    st.shared.f32 [%r17 + 1536 + 0],   %f112;
    st.shared.f32 [%r17 + 1536 + 4],   %f113;
    st.shared.f32 [%r17 + 1536 + 256], %f114;
    st.shared.f32 [%r17 + 1536 + 260], %f115;
    bar.sync 0;
    @!%p3 bra TILE3_RBYPASS;
    ld.shared.f32 %f50, [%r17 + 1536 + 0];
    ld.shared.f32 %f51, [%r17 + 1536 + 4096];
    ld.shared.f32 %f52, [%r17 + 1536 + 8192];
    ld.shared.f32 %f53, [%r17 + 1536 + 12288];
    ld.shared.f32 %f54, [%r17 + 1536 + 16384];
    ld.shared.f32 %f55, [%r17 + 1536 + 20480];
    ld.shared.f32 %f56, [%r17 + 1536 + 24576];
    ld.shared.f32 %f57, [%r17 + 1536 + 28672];
    add.f32 %f50, %f50, %f51; add.f32 %f50, %f50, %f52; add.f32 %f50, %f50, %f53;
    add.f32 %f50, %f50, %f54; add.f32 %f50, %f50, %f55; add.f32 %f50, %f50, %f56;
    add.f32 %f50, %f50, %f57;

    ld.shared.f32 %f58, [%r17 + 1536 + 4];
    ld.shared.f32 %f59, [%r17 + 1536 + 4100];
    ld.shared.f32 %f60, [%r17 + 1536 + 8196];
    ld.shared.f32 %f61, [%r17 + 1536 + 12292];
    ld.shared.f32 %f62, [%r17 + 1536 + 16388];
    ld.shared.f32 %f63, [%r17 + 1536 + 20484];
    ld.shared.f32 %f64, [%r17 + 1536 + 24580];
    ld.shared.f32 %f65, [%r17 + 1536 + 28676];
    add.f32 %f58, %f58, %f59; add.f32 %f58, %f58, %f60; add.f32 %f58, %f58, %f61;
    add.f32 %f58, %f58, %f62; add.f32 %f58, %f58, %f63; add.f32 %f58, %f58, %f64;
    add.f32 %f58, %f58, %f65;

    ld.shared.f32 %f62, [%r17 + 1536 + 256];
    ld.shared.f32 %f63, [%r17 + 1536 + 4352];
    ld.shared.f32 %f64, [%r17 + 1536 + 8448];
    ld.shared.f32 %f65, [%r17 + 1536 + 12544];
    ld.shared.f32 %f66, [%r17 + 1536 + 16640];
    ld.shared.f32 %f67, [%r17 + 1536 + 20736];
    ld.shared.f32 %f68, [%r17 + 1536 + 24832];
    ld.shared.f32 %f69, [%r17 + 1536 + 28928];
    add.f32 %f62, %f62, %f63; add.f32 %f62, %f62, %f64; add.f32 %f62, %f62, %f65;
    add.f32 %f62, %f62, %f66; add.f32 %f62, %f62, %f67; add.f32 %f62, %f62, %f68;
    add.f32 %f62, %f62, %f69;

    ld.shared.f32 %f66, [%r17 + 1536 + 260];
    ld.shared.f32 %f67, [%r17 + 1536 + 4356];
    ld.shared.f32 %f68, [%r17 + 1536 + 8452];
    ld.shared.f32 %f69, [%r17 + 1536 + 12548];
    ld.shared.f32 %f70, [%r17 + 1536 + 16644];
    ld.shared.f32 %f71, [%r17 + 1536 + 20740];
    ld.shared.f32 %f72, [%r17 + 1536 + 24836];
    ld.shared.f32 %f73, [%r17 + 1536 + 28932];
    add.f32 %f66, %f66, %f67; add.f32 %f66, %f66, %f68; add.f32 %f66, %f66, %f69;
    add.f32 %f66, %f66, %f70; add.f32 %f66, %f66, %f71; add.f32 %f66, %f66, %f72;
    add.f32 %f66, %f66, %f73;

    add.u32 %r150, %r9, 48;
    add.u32 %r151, %r140, 48;
    setp.lt.u32 %p21, %r150, %r2;
    setp.lt.u32 %p22, %r151, %r2;
    mad.lo.u32 %r106, %r150, %r0, %r4;
    add.u32    %r106, %r106, %r141;
    shl.b32    %r106, %r106, 2;
    cvt.u64.u32 %rd11, %r106;
    add.u64 %rd11, %rd3, %rd11;
    @%p21 st.global.f32 [%rd11 + 0], %f50;
    @%p21 st.global.f32 [%rd11 + 4], %f58;

    mad.lo.u32 %r107, %r151, %r0, %r4;
    add.u32    %r107, %r107, %r141;
    shl.b32    %r107, %r107, 2;
    cvt.u64.u32 %rd16, %r107;
    add.u64 %rd16, %rd3, %rd16;
    @%p22 st.global.f32 [%rd16 + 0], %f62;
    @%p22 st.global.f32 [%rd16 + 4], %f66;

TILE3_RBYPASS:
    bar.sync 0;
TILE3_RDONE:


    // ---- tile 4 reduction ----
    setp.ge.u32 %p20, 64, %r2;
    @%p20 bra TILE4_RDONE;
    st.shared.f32 [%r17 + 2048 + 0],   %f116;
    st.shared.f32 [%r17 + 2048 + 4],   %f117;
    st.shared.f32 [%r17 + 2048 + 256], %f118;
    st.shared.f32 [%r17 + 2048 + 260], %f119;
    bar.sync 0;
    @!%p3 bra TILE4_RBYPASS;
    ld.shared.f32 %f50, [%r17 + 2048 + 0];
    ld.shared.f32 %f51, [%r17 + 2048 + 4096];
    ld.shared.f32 %f52, [%r17 + 2048 + 8192];
    ld.shared.f32 %f53, [%r17 + 2048 + 12288];
    ld.shared.f32 %f54, [%r17 + 2048 + 16384];
    ld.shared.f32 %f55, [%r17 + 2048 + 20480];
    ld.shared.f32 %f56, [%r17 + 2048 + 24576];
    ld.shared.f32 %f57, [%r17 + 2048 + 28672];
    add.f32 %f50, %f50, %f51; add.f32 %f50, %f50, %f52; add.f32 %f50, %f50, %f53;
    add.f32 %f50, %f50, %f54; add.f32 %f50, %f50, %f55; add.f32 %f50, %f50, %f56;
    add.f32 %f50, %f50, %f57;

    ld.shared.f32 %f58, [%r17 + 2048 + 4];
    ld.shared.f32 %f59, [%r17 + 2048 + 4100];
    ld.shared.f32 %f60, [%r17 + 2048 + 8196];
    ld.shared.f32 %f61, [%r17 + 2048 + 12292];
    ld.shared.f32 %f62, [%r17 + 2048 + 16388];
    ld.shared.f32 %f63, [%r17 + 2048 + 20484];
    ld.shared.f32 %f64, [%r17 + 2048 + 24580];
    ld.shared.f32 %f65, [%r17 + 2048 + 28676];
    add.f32 %f58, %f58, %f59; add.f32 %f58, %f58, %f60; add.f32 %f58, %f58, %f61;
    add.f32 %f58, %f58, %f62; add.f32 %f58, %f58, %f63; add.f32 %f58, %f58, %f64;
    add.f32 %f58, %f58, %f65;

    ld.shared.f32 %f62, [%r17 + 2048 + 256];
    ld.shared.f32 %f63, [%r17 + 2048 + 4352];
    ld.shared.f32 %f64, [%r17 + 2048 + 8448];
    ld.shared.f32 %f65, [%r17 + 2048 + 12544];
    ld.shared.f32 %f66, [%r17 + 2048 + 16640];
    ld.shared.f32 %f67, [%r17 + 2048 + 20736];
    ld.shared.f32 %f68, [%r17 + 2048 + 24832];
    ld.shared.f32 %f69, [%r17 + 2048 + 28928];
    add.f32 %f62, %f62, %f63; add.f32 %f62, %f62, %f64; add.f32 %f62, %f62, %f65;
    add.f32 %f62, %f62, %f66; add.f32 %f62, %f62, %f67; add.f32 %f62, %f62, %f68;
    add.f32 %f62, %f62, %f69;

    ld.shared.f32 %f66, [%r17 + 2048 + 260];
    ld.shared.f32 %f67, [%r17 + 2048 + 4356];
    ld.shared.f32 %f68, [%r17 + 2048 + 8452];
    ld.shared.f32 %f69, [%r17 + 2048 + 12548];
    ld.shared.f32 %f70, [%r17 + 2048 + 16644];
    ld.shared.f32 %f71, [%r17 + 2048 + 20740];
    ld.shared.f32 %f72, [%r17 + 2048 + 24836];
    ld.shared.f32 %f73, [%r17 + 2048 + 28932];
    add.f32 %f66, %f66, %f67; add.f32 %f66, %f66, %f68; add.f32 %f66, %f66, %f69;
    add.f32 %f66, %f66, %f70; add.f32 %f66, %f66, %f71; add.f32 %f66, %f66, %f72;
    add.f32 %f66, %f66, %f73;

    add.u32 %r150, %r9, 64;
    add.u32 %r151, %r140, 64;
    setp.lt.u32 %p21, %r150, %r2;
    setp.lt.u32 %p22, %r151, %r2;
    mad.lo.u32 %r106, %r150, %r0, %r4;
    add.u32    %r106, %r106, %r141;
    shl.b32    %r106, %r106, 2;
    cvt.u64.u32 %rd11, %r106;
    add.u64 %rd11, %rd3, %rd11;
    @%p21 st.global.f32 [%rd11 + 0], %f50;
    @%p21 st.global.f32 [%rd11 + 4], %f58;

    mad.lo.u32 %r107, %r151, %r0, %r4;
    add.u32    %r107, %r107, %r141;
    shl.b32    %r107, %r107, 2;
    cvt.u64.u32 %rd16, %r107;
    add.u64 %rd16, %rd3, %rd16;
    @%p22 st.global.f32 [%rd16 + 0], %f62;
    @%p22 st.global.f32 [%rd16 + 4], %f66;

TILE4_RBYPASS:
    bar.sync 0;
TILE4_RDONE:


    // ---- tile 5 reduction ----
    setp.ge.u32 %p20, 80, %r2;
    @%p20 bra TILE5_RDONE;
    st.shared.f32 [%r17 + 2560 + 0],   %f120;
    st.shared.f32 [%r17 + 2560 + 4],   %f121;
    st.shared.f32 [%r17 + 2560 + 256], %f122;
    st.shared.f32 [%r17 + 2560 + 260], %f123;
    bar.sync 0;
    @!%p3 bra TILE5_RBYPASS;
    ld.shared.f32 %f50, [%r17 + 2560 + 0];
    ld.shared.f32 %f51, [%r17 + 2560 + 4096];
    ld.shared.f32 %f52, [%r17 + 2560 + 8192];
    ld.shared.f32 %f53, [%r17 + 2560 + 12288];
    ld.shared.f32 %f54, [%r17 + 2560 + 16384];
    ld.shared.f32 %f55, [%r17 + 2560 + 20480];
    ld.shared.f32 %f56, [%r17 + 2560 + 24576];
    ld.shared.f32 %f57, [%r17 + 2560 + 28672];
    add.f32 %f50, %f50, %f51; add.f32 %f50, %f50, %f52; add.f32 %f50, %f50, %f53;
    add.f32 %f50, %f50, %f54; add.f32 %f50, %f50, %f55; add.f32 %f50, %f50, %f56;
    add.f32 %f50, %f50, %f57;

    ld.shared.f32 %f58, [%r17 + 2560 + 4];
    ld.shared.f32 %f59, [%r17 + 2560 + 4100];
    ld.shared.f32 %f60, [%r17 + 2560 + 8196];
    ld.shared.f32 %f61, [%r17 + 2560 + 12292];
    ld.shared.f32 %f62, [%r17 + 2560 + 16388];
    ld.shared.f32 %f63, [%r17 + 2560 + 20484];
    ld.shared.f32 %f64, [%r17 + 2560 + 24580];
    ld.shared.f32 %f65, [%r17 + 2560 + 28676];
    add.f32 %f58, %f58, %f59; add.f32 %f58, %f58, %f60; add.f32 %f58, %f58, %f61;
    add.f32 %f58, %f58, %f62; add.f32 %f58, %f58, %f63; add.f32 %f58, %f58, %f64;
    add.f32 %f58, %f58, %f65;

    ld.shared.f32 %f62, [%r17 + 2560 + 256];
    ld.shared.f32 %f63, [%r17 + 2560 + 4352];
    ld.shared.f32 %f64, [%r17 + 2560 + 8448];
    ld.shared.f32 %f65, [%r17 + 2560 + 12544];
    ld.shared.f32 %f66, [%r17 + 2560 + 16640];
    ld.shared.f32 %f67, [%r17 + 2560 + 20736];
    ld.shared.f32 %f68, [%r17 + 2560 + 24832];
    ld.shared.f32 %f69, [%r17 + 2560 + 28928];
    add.f32 %f62, %f62, %f63; add.f32 %f62, %f62, %f64; add.f32 %f62, %f62, %f65;
    add.f32 %f62, %f62, %f66; add.f32 %f62, %f62, %f67; add.f32 %f62, %f62, %f68;
    add.f32 %f62, %f62, %f69;

    ld.shared.f32 %f66, [%r17 + 2560 + 260];
    ld.shared.f32 %f67, [%r17 + 2560 + 4356];
    ld.shared.f32 %f68, [%r17 + 2560 + 8452];
    ld.shared.f32 %f69, [%r17 + 2560 + 12548];
    ld.shared.f32 %f70, [%r17 + 2560 + 16644];
    ld.shared.f32 %f71, [%r17 + 2560 + 20740];
    ld.shared.f32 %f72, [%r17 + 2560 + 24836];
    ld.shared.f32 %f73, [%r17 + 2560 + 28932];
    add.f32 %f66, %f66, %f67; add.f32 %f66, %f66, %f68; add.f32 %f66, %f66, %f69;
    add.f32 %f66, %f66, %f70; add.f32 %f66, %f66, %f71; add.f32 %f66, %f66, %f72;
    add.f32 %f66, %f66, %f73;

    add.u32 %r150, %r9, 80;
    add.u32 %r151, %r140, 80;
    setp.lt.u32 %p21, %r150, %r2;
    setp.lt.u32 %p22, %r151, %r2;
    mad.lo.u32 %r106, %r150, %r0, %r4;
    add.u32    %r106, %r106, %r141;
    shl.b32    %r106, %r106, 2;
    cvt.u64.u32 %rd11, %r106;
    add.u64 %rd11, %rd3, %rd11;
    @%p21 st.global.f32 [%rd11 + 0], %f50;
    @%p21 st.global.f32 [%rd11 + 4], %f58;

    mad.lo.u32 %r107, %r151, %r0, %r4;
    add.u32    %r107, %r107, %r141;
    shl.b32    %r107, %r107, 2;
    cvt.u64.u32 %rd16, %r107;
    add.u64 %rd16, %rd3, %rd16;
    @%p22 st.global.f32 [%rd16 + 0], %f62;
    @%p22 st.global.f32 [%rd16 + 4], %f66;

TILE5_RBYPASS:
    bar.sync 0;
TILE5_RDONE:


    // ---- tile 6 reduction ----
    setp.ge.u32 %p20, 96, %r2;
    @%p20 bra TILE6_RDONE;
    st.shared.f32 [%r17 + 3072 + 0],   %f124;
    st.shared.f32 [%r17 + 3072 + 4],   %f125;
    st.shared.f32 [%r17 + 3072 + 256], %f126;
    st.shared.f32 [%r17 + 3072 + 260], %f127;
    bar.sync 0;
    @!%p3 bra TILE6_RBYPASS;
    ld.shared.f32 %f50, [%r17 + 3072 + 0];
    ld.shared.f32 %f51, [%r17 + 3072 + 4096];
    ld.shared.f32 %f52, [%r17 + 3072 + 8192];
    ld.shared.f32 %f53, [%r17 + 3072 + 12288];
    ld.shared.f32 %f54, [%r17 + 3072 + 16384];
    ld.shared.f32 %f55, [%r17 + 3072 + 20480];
    ld.shared.f32 %f56, [%r17 + 3072 + 24576];
    ld.shared.f32 %f57, [%r17 + 3072 + 28672];
    add.f32 %f50, %f50, %f51; add.f32 %f50, %f50, %f52; add.f32 %f50, %f50, %f53;
    add.f32 %f50, %f50, %f54; add.f32 %f50, %f50, %f55; add.f32 %f50, %f50, %f56;
    add.f32 %f50, %f50, %f57;

    ld.shared.f32 %f58, [%r17 + 3072 + 4];
    ld.shared.f32 %f59, [%r17 + 3072 + 4100];
    ld.shared.f32 %f60, [%r17 + 3072 + 8196];
    ld.shared.f32 %f61, [%r17 + 3072 + 12292];
    ld.shared.f32 %f62, [%r17 + 3072 + 16388];
    ld.shared.f32 %f63, [%r17 + 3072 + 20484];
    ld.shared.f32 %f64, [%r17 + 3072 + 24580];
    ld.shared.f32 %f65, [%r17 + 3072 + 28676];
    add.f32 %f58, %f58, %f59; add.f32 %f58, %f58, %f60; add.f32 %f58, %f58, %f61;
    add.f32 %f58, %f58, %f62; add.f32 %f58, %f58, %f63; add.f32 %f58, %f58, %f64;
    add.f32 %f58, %f58, %f65;

    ld.shared.f32 %f62, [%r17 + 3072 + 256];
    ld.shared.f32 %f63, [%r17 + 3072 + 4352];
    ld.shared.f32 %f64, [%r17 + 3072 + 8448];
    ld.shared.f32 %f65, [%r17 + 3072 + 12544];
    ld.shared.f32 %f66, [%r17 + 3072 + 16640];
    ld.shared.f32 %f67, [%r17 + 3072 + 20736];
    ld.shared.f32 %f68, [%r17 + 3072 + 24832];
    ld.shared.f32 %f69, [%r17 + 3072 + 28928];
    add.f32 %f62, %f62, %f63; add.f32 %f62, %f62, %f64; add.f32 %f62, %f62, %f65;
    add.f32 %f62, %f62, %f66; add.f32 %f62, %f62, %f67; add.f32 %f62, %f62, %f68;
    add.f32 %f62, %f62, %f69;

    ld.shared.f32 %f66, [%r17 + 3072 + 260];
    ld.shared.f32 %f67, [%r17 + 3072 + 4356];
    ld.shared.f32 %f68, [%r17 + 3072 + 8452];
    ld.shared.f32 %f69, [%r17 + 3072 + 12548];
    ld.shared.f32 %f70, [%r17 + 3072 + 16644];
    ld.shared.f32 %f71, [%r17 + 3072 + 20740];
    ld.shared.f32 %f72, [%r17 + 3072 + 24836];
    ld.shared.f32 %f73, [%r17 + 3072 + 28932];
    add.f32 %f66, %f66, %f67; add.f32 %f66, %f66, %f68; add.f32 %f66, %f66, %f69;
    add.f32 %f66, %f66, %f70; add.f32 %f66, %f66, %f71; add.f32 %f66, %f66, %f72;
    add.f32 %f66, %f66, %f73;

    add.u32 %r150, %r9, 96;
    add.u32 %r151, %r140, 96;
    setp.lt.u32 %p21, %r150, %r2;
    setp.lt.u32 %p22, %r151, %r2;
    mad.lo.u32 %r106, %r150, %r0, %r4;
    add.u32    %r106, %r106, %r141;
    shl.b32    %r106, %r106, 2;
    cvt.u64.u32 %rd11, %r106;
    add.u64 %rd11, %rd3, %rd11;
    @%p21 st.global.f32 [%rd11 + 0], %f50;
    @%p21 st.global.f32 [%rd11 + 4], %f58;

    mad.lo.u32 %r107, %r151, %r0, %r4;
    add.u32    %r107, %r107, %r141;
    shl.b32    %r107, %r107, 2;
    cvt.u64.u32 %rd16, %r107;
    add.u64 %rd16, %rd3, %rd16;
    @%p22 st.global.f32 [%rd16 + 0], %f62;
    @%p22 st.global.f32 [%rd16 + 4], %f66;

TILE6_RBYPASS:
    bar.sync 0;
TILE6_RDONE:


    // ---- tile 7 reduction ----
    setp.ge.u32 %p20, 112, %r2;
    @%p20 bra TILE7_RDONE;
    st.shared.f32 [%r17 + 3584 + 0],   %f128;
    st.shared.f32 [%r17 + 3584 + 4],   %f129;
    st.shared.f32 [%r17 + 3584 + 256], %f130;
    st.shared.f32 [%r17 + 3584 + 260], %f131;
    bar.sync 0;
    @!%p3 bra TILE7_RBYPASS;
    ld.shared.f32 %f50, [%r17 + 3584 + 0];
    ld.shared.f32 %f51, [%r17 + 3584 + 4096];
    ld.shared.f32 %f52, [%r17 + 3584 + 8192];
    ld.shared.f32 %f53, [%r17 + 3584 + 12288];
    ld.shared.f32 %f54, [%r17 + 3584 + 16384];
    ld.shared.f32 %f55, [%r17 + 3584 + 20480];
    ld.shared.f32 %f56, [%r17 + 3584 + 24576];
    ld.shared.f32 %f57, [%r17 + 3584 + 28672];
    add.f32 %f50, %f50, %f51; add.f32 %f50, %f50, %f52; add.f32 %f50, %f50, %f53;
    add.f32 %f50, %f50, %f54; add.f32 %f50, %f50, %f55; add.f32 %f50, %f50, %f56;
    add.f32 %f50, %f50, %f57;

    ld.shared.f32 %f58, [%r17 + 3584 + 4];
    ld.shared.f32 %f59, [%r17 + 3584 + 4100];
    ld.shared.f32 %f60, [%r17 + 3584 + 8196];
    ld.shared.f32 %f61, [%r17 + 3584 + 12292];
    ld.shared.f32 %f62, [%r17 + 3584 + 16388];
    ld.shared.f32 %f63, [%r17 + 3584 + 20484];
    ld.shared.f32 %f64, [%r17 + 3584 + 24580];
    ld.shared.f32 %f65, [%r17 + 3584 + 28676];
    add.f32 %f58, %f58, %f59; add.f32 %f58, %f58, %f60; add.f32 %f58, %f58, %f61;
    add.f32 %f58, %f58, %f62; add.f32 %f58, %f58, %f63; add.f32 %f58, %f58, %f64;
    add.f32 %f58, %f58, %f65;

    ld.shared.f32 %f62, [%r17 + 3584 + 256];
    ld.shared.f32 %f63, [%r17 + 3584 + 4352];
    ld.shared.f32 %f64, [%r17 + 3584 + 8448];
    ld.shared.f32 %f65, [%r17 + 3584 + 12544];
    ld.shared.f32 %f66, [%r17 + 3584 + 16640];
    ld.shared.f32 %f67, [%r17 + 3584 + 20736];
    ld.shared.f32 %f68, [%r17 + 3584 + 24832];
    ld.shared.f32 %f69, [%r17 + 3584 + 28928];
    add.f32 %f62, %f62, %f63; add.f32 %f62, %f62, %f64; add.f32 %f62, %f62, %f65;
    add.f32 %f62, %f62, %f66; add.f32 %f62, %f62, %f67; add.f32 %f62, %f62, %f68;
    add.f32 %f62, %f62, %f69;

    ld.shared.f32 %f66, [%r17 + 3584 + 260];
    ld.shared.f32 %f67, [%r17 + 3584 + 4356];
    ld.shared.f32 %f68, [%r17 + 3584 + 8452];
    ld.shared.f32 %f69, [%r17 + 3584 + 12548];
    ld.shared.f32 %f70, [%r17 + 3584 + 16644];
    ld.shared.f32 %f71, [%r17 + 3584 + 20740];
    ld.shared.f32 %f72, [%r17 + 3584 + 24836];
    ld.shared.f32 %f73, [%r17 + 3584 + 28932];
    add.f32 %f66, %f66, %f67; add.f32 %f66, %f66, %f68; add.f32 %f66, %f66, %f69;
    add.f32 %f66, %f66, %f70; add.f32 %f66, %f66, %f71; add.f32 %f66, %f66, %f72;
    add.f32 %f66, %f66, %f73;

    add.u32 %r150, %r9, 112;
    add.u32 %r151, %r140, 112;
    setp.lt.u32 %p21, %r150, %r2;
    setp.lt.u32 %p22, %r151, %r2;
    mad.lo.u32 %r106, %r150, %r0, %r4;
    add.u32    %r106, %r106, %r141;
    shl.b32    %r106, %r106, 2;
    cvt.u64.u32 %rd11, %r106;
    add.u64 %rd11, %rd3, %rd11;
    @%p21 st.global.f32 [%rd11 + 0], %f50;
    @%p21 st.global.f32 [%rd11 + 4], %f58;

    mad.lo.u32 %r107, %r151, %r0, %r4;
    add.u32    %r107, %r107, %r141;
    shl.b32    %r107, %r107, 2;
    cvt.u64.u32 %rd16, %r107;
    add.u64 %rd16, %rd3, %rd16;
    @%p22 st.global.f32 [%rd16 + 0], %f62;
    @%p22 st.global.f32 [%rd16 + 4], %f66;

TILE7_RBYPASS:
    bar.sync 0;
TILE7_RDONE:

DONE:
    ret;
}
"#;
