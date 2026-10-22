/// Stream-K persistent-block PTX kernel for Q4K x Q8_1 batch GEMM using INT8 tensor cores (sm_120).
///
/// # Design: grid=(SM_COUNT,1,1) persistent blocks
///
/// Grid:  (SM_COUNT, 1, 1)  -- one block per SM; all SMs active simultaneously
/// Block: (128, 1, 1)       -- 4 warps, K-parallel within each tile
///
/// Each block is assigned ceil(N_col_tiles / SM_COUNT) output column tiles
/// (n8-column tiles) and processes them sequentially, computing the FULL
/// K-dimension reduction for each tile before writing to output.
///
/// # Key difference from mma_batch (N/8, 1, 1)
///
/// mma_batch has 512 blocks for N=4096 -> 3 waves on 170 SMs -> wave fragmentation.
/// stream-K has exactly 170 blocks -> all SMs busy the entire time with no wave tail.
/// Each block processes 3 or 4 tiles sequentially; smem_partial is reset per tile.
///
/// # Shared memory layout (same as mma_batch)
///
/// [0..127]:   smem_d8[8]    f32 super-scale d for 8 neurons
/// [128..255]: smem_dmin8[8] f32 super-min dmin for 8 neurons
/// [256..319]: smem_sc8[8 neurons][8 sub-blocks] u8 = 64 bytes
/// [320..383]: smem_mv8[8 neurons][8 sub-blocks] u8 = 64 bytes
/// [384..2431]: smem_partial[4 warps][16 rows][8 cols] f32 = 2048 bytes
/// Total: 2432 bytes (same as mma_batch)
///
/// # Synchronisation per tile
///
/// 1. bar.sync 0 -- after AFTER_HEADER: all 8 header loads committed
/// 2. bar.sync 0 -- at KB_DONE: all K-loop smem_partial writes committed
/// 3. bar.sync 1 -- at SKIP_WRITE: warp-0 reads of smem_partial committed;
///                  safe to zero smem_partial for the next tile

pub const MIN_CC: u32 = 1200;
pub const ENTRY: &str  = "q4k_streamk_batch_sm120";

pub const PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// q4k_streamk_batch_sm120
//
// Stream-K persistent Q4K batch GEMM for sm_120 (RTX 5090).
// Grid:  (SM_COUNT, 1, 1)
// Block: (128, 1, 1) -- 4 warps
//
// Parameters:
//   param_weights:    *const u32 -- native GGUF Q4K [N * k_blocks * 36 u32]
//   param_act_qs:     *const u32 -- Q8_1 INT8 activations [M * k_blocks * 64 u32]
//   param_act_ds:     *const u32 -- Q8_1 block scales [M * k_blocks * 8 u32]
//   param_out:        *mut f32   -- output [M * N]
//   param_n:          u32        -- N (output neurons, multiple of 8)
//   param_k_blocks:   u32        -- k_blocks (K = k_blocks * 256)
//   param_m:          u32        -- M (batch size, <= 16)
//   param_sm_count:   u32        -- SM count (CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
.visible .entry q4k_streamk_batch_sm120(
    .param .u64 param_weights,
    .param .u64 param_act_qs,
    .param .u64 param_act_ds,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .u32 param_k_blocks,
    .param .u32 param_m,
    .param .u32 param_sm_count
)
{
    // r0   = N
    // r1   = k_blocks
    // r2   = M
    // r3   = n_tile_base (cur_tile * 8) -- updated per tile
    // r5   = tid.x (constant)
    // r6   = lane  (constant)
    // r7   = warp_id (constant)
    // r8   = d_row = lane%16  (constant)
    // r9   = lane&16          (constant)
    // r10  = lane/16          (constant)
    // r11  = d_col_base (0 or 4) (constant)
    // r12  = lane%8 = local neuron (constant)
    // r13  = lane/8 = k-group (constant)
    // r14  = global neuron = n_tile_base + lane%8 (updated per tile)
    // r15  = warp_id * 512
    // r16  = (lane%16)*32
    // r17  = smem_partial slot abs addr (constant after setup)
    // r18  = smem_buf base addr (constant)
    // r19  = d_row * k_blocks * 256 (x_qs row offset)
    // r20  = d_row * k_blocks * 32  (x_ds row offset)
    // r21  = kb (k_blocks loop counter, reset per tile)
    // r60  = mw (sub-block counter, 0..7, reset per kb)
    // r130 = sm_count
    // r131 = n_col_tiles
    // r132 = tiles_per_block
    // r133 = blockIdx.x
    // r134 = my_tile_start
    // r135 = my_tile_end
    // r136 = cur_tile (outer tile loop counter)
    .reg .u32  %r<140>;
    .reg .u64  %rd<30>;
    .reg .f32  %f<70>;
    .reg .pred %p<15>;
    .reg .b16  %h<6>;

    .shared .align 4 .b8 smem_buf[2432];

    // ---- Load kernel parameters ------------------------------------------
    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_act_qs];
    ld.param.u64 %rd2, [param_act_ds];
    ld.param.u64 %rd3, [param_out];
    ld.param.u32 %r0,   [param_n];
    ld.param.u32 %r1,   [param_k_blocks];
    ld.param.u32 %r2,   [param_m];
    ld.param.u32 %r130, [param_sm_count];

    // ---- Thread decomposition (tile-independent) -------------------------
    mov.u32 %r5,  %tid.x;
    and.b32 %r6,  %r5, 31;          // lane
    shr.u32 %r7,  %r5, 5;           // warp_id (0..3)
    and.b32 %r8,  %r6, 15;          // d_row = lane%16
    and.b32 %r9,  %r6, 16;          // lane&16
    shr.u32 %r10, %r6, 4;           // lane/16
    shl.b32 %r11, %r10, 2;          // d_col_base: 0 or 4
    and.b32 %r12, %r6, 7;           // lane%8 = local neuron within n8 tile
    shr.u32 %r13, %r6, 3;           // lane/8 = k-group index (0..3)

    // ---- Thread predicates (tile-independent) ----------------------------
    setp.lt.u32 %p1, %r5, 8;        // tid < 8 (header-loading threads)
    setp.lt.u32 %p2, %r8, %r2;      // d_row < m (valid activation row)
    setp.eq.u32 %p3, %r7, 0;        // warp_id == 0 (reduction warp)

    // ---- smem_partial slot address (tile-independent layout) ------------
    // slot = smem_buf + 384 + warp_id*512 + (lane%16)*32 + (lane&16)
    shl.b32 %r15, %r7, 9;           // warp_id * 512
    shl.b32 %r16, %r8, 5;           // (lane%16) * 32
    or.b32  %r17, %r16, %r9;        // + lane&16
    add.u32 %r17, %r15, %r17;
    add.u32 %r17, %r17, 384;        // + smem_partial_offset
    mov.u32 %r18, smem_buf;
    add.u32 %r17, %r18, %r17;       // absolute smem_partial slot address

    // ---- Activation row base pointers (tile-independent) ----------------
    // x_qs row stride: k_blocks * 64 u32 = k_blocks * 256 bytes
    mul.lo.u32 %r19, %r8, %r1;      // d_row * k_blocks
    shl.b32    %r19, %r19, 8;       // * 256 bytes
    cvt.u64.u32 %rd4, %r19;
    add.u64    %rd4, %rd1, %rd4;    // x_qs row base for d_row

    // x_ds row stride: k_blocks * 8 u32 = k_blocks * 32 bytes
    mul.lo.u32 %r20, %r8, %r1;
    shl.b32    %r20, %r20, 5;
    cvt.u64.u32 %rd5, %r20;
    add.u64    %rd5, %rd2, %rd5;    // x_ds row base for d_row

    // ---- Compute tile assignment for this block -------------------------
    // n_col_tiles = ceil(N / 8)
    add.u32 %r131, %r0, 7;
    shr.u32 %r131, %r131, 3;        // n_col_tiles

    // tiles_per_block = ceil(n_col_tiles / sm_count)
    add.u32 %r132, %r131, %r130;
    sub.u32 %r132, %r132, 1;
    div.u32 %r132, %r132, %r130;    // tiles_per_block

    // my_tile_start = blockIdx.x * tiles_per_block
    mov.u32 %r133, %ctaid.x;
    mul.lo.u32 %r134, %r133, %r132; // my_tile_start

    // my_tile_end = min(my_tile_start + tiles_per_block, n_col_tiles)
    add.u32 %r135, %r134, %r132;
    min.u32 %r135, %r135, %r131;    // my_tile_end

    // cur_tile = my_tile_start
    mov.u32 %r136, %r134;

// =========================================================================
// OUTER TILE LOOP: process assigned column tiles sequentially
// =========================================================================
TILE_LOOP:
    setp.ge.u32 %p14, %r136, %r135;
    @%p14 bra DONE;

    // n_tile_base = cur_tile * 8
    shl.b32 %r3, %r136, 3;

    // Global neuron for B loads: n_tile_base + lane%8
    add.u32 %r14, %r3, %r12;

    // Zero smem_partial for this tile (each thread zeroes its own slot).
    // No preceding bar.sync needed -- bar.sync 1 at SKIP_WRITE of previous
    // tile ensures warp-0 has finished reading smem_partial before we zero.
    st.shared.f32 [%r17 + 0],  0f00000000;
    st.shared.f32 [%r17 + 4],  0f00000000;
    st.shared.f32 [%r17 + 8],  0f00000000;
    st.shared.f32 [%r17 + 12], 0f00000000;

    // ---- Outer K-block loop ---------------------------------------------
    mov.u32 %r21, 0;                 // kb = 0

KB_LOOP:
    setp.ge.u32 %p4, %r21, %r1;
    @%p4 bra KB_DONE;

    // ---- Header load by first 8 threads (tid < 8) -----------------------
    // Thread tid (0..7) loads Q4K block header for neuron n_tile_base+tid.
    @!%p1 bra AFTER_HEADER;

    // Q4K block byte offset:
    //   (n_tile_base + tid) * k_blocks * 144 + kb * 144
    //   = n_tile_base * k_blocks * 144 + tid * k_blocks * 144 + kb * 144
    mul.lo.u32 %r22, %r5, %r1;      // tid * k_blocks
    mul.lo.u32 %r22, %r22, 144;
    mad.lo.u32 %r22, %r21, 144, %r22;  // + kb*144
    mul.lo.u32 %r23, %r3, %r1;      // n_tile_base * k_blocks
    mul.lo.u32 %r23, %r23, 144;
    add.u32    %r22, %r22, %r23;
    cvt.u64.u32 %rd6, %r22;
    add.u64 %rd6, %rd0, %rd6;       // ptr to Q4K block header

    // Load d, dmin (f16 -> f32)
    ld.global.nc.b16 %h0, [%rd6 + 0];
    ld.global.nc.b16 %h1, [%rd6 + 2];
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;

    // Load three scale words (bytes 4..15)
    ld.global.nc.b32 %r24, [%rd6 + 4];
    ld.global.nc.b32 %r25, [%rd6 + 8];
    ld.global.nc.b32 %r26, [%rd6 + 12];

    // Extract sc[0..7] (6-bit) from scales field
    and.b32 %r27, %r24, 63;
    shr.u32 %r28, %r24, 8;  and.b32 %r28, %r28, 63;
    shr.u32 %r29, %r24, 16; and.b32 %r29, %r29, 63;
    shr.u32 %r30, %r24, 24; and.b32 %r30, %r30, 63;
    // m_val[0..3] from ps1
    and.b32 %r31, %r25, 63;
    shr.u32 %r32, %r25, 8;  and.b32 %r32, %r32, 63;
    shr.u32 %r33, %r25, 16; and.b32 %r33, %r33, 63;
    shr.u32 %r34, %r25, 24; and.b32 %r34, %r34, 63;
    // sc[4..7] and m_val[4..7]: cross-byte from all 3 words
    and.b32 %r35, %r26, 255;
    shr.u32 %r36, %r26, 8;  and.b32 %r36, %r36, 255;
    and.b32 %r37, %r24, 192; shr.u32 %r37, %r37, 2;
    shr.u32 %r38, %r24, 8;  and.b32 %r38, %r38, 192; shr.u32 %r38, %r38, 2;
    and.b32 %r39, %r35, 15;  or.b32 %r39, %r39, %r37;   // sc[4]
    and.b32 %r40, %r36, 15;  or.b32 %r40, %r40, %r38;   // sc[5]
    shr.u32 %r41, %r35, 4;  and.b32 %r41, %r41, 15;
    and.b32 %r42, %r25, 192; shr.u32 %r42, %r42, 2;
    or.b32  %r41, %r41, %r42;                             // m_val[4]
    shr.u32 %r43, %r36, 4;  and.b32 %r43, %r43, 15;
    shr.u32 %r44, %r25, 8;  and.b32 %r44, %r44, 192; shr.u32 %r44, %r44, 2;
    or.b32  %r43, %r43, %r44;                             // m_val[5]
    shr.u32 %r45, %r26, 16; and.b32 %r45, %r45, 255;
    shr.u32 %r46, %r26, 24; and.b32 %r46, %r46, 255;
    shr.u32 %r47, %r24, 16; and.b32 %r47, %r47, 192; shr.u32 %r47, %r47, 2;
    shr.u32 %r48, %r24, 24; and.b32 %r48, %r48, 192; shr.u32 %r48, %r48, 2;
    and.b32 %r49, %r45, 15; or.b32 %r49, %r49, %r47;   // sc[6]
    and.b32 %r50, %r46, 15; or.b32 %r50, %r50, %r48;   // sc[7]
    shr.u32 %r51, %r45, 4;  and.b32 %r51, %r51, 15;
    shr.u32 %r52, %r25, 16; and.b32 %r52, %r52, 192; shr.u32 %r52, %r52, 2;
    or.b32  %r51, %r51, %r52;                             // m_val[6]
    shr.u32 %r53, %r46, 4;  and.b32 %r53, %r53, 15;
    shr.u32 %r54, %r25, 24; and.b32 %r54, %r54, 192; shr.u32 %r54, %r54, 2;
    or.b32  %r53, %r53, %r54;                             // m_val[7]

    // Store to smem: neuron index = tid (0..7)
    shl.b32 %r55, %r5, 2;           // tid * 4 bytes
    mov.u32 %r56, smem_buf;
    add.u32 %r57, %r56, %r55;
    st.shared.f32 [%r57 + 0],   %f0;  // smem_d8[tid]
    st.shared.f32 [%r57 + 128], %f1;  // smem_dmin8[tid]

    // smem_sc8[tid][8] at smem_buf + 256 + tid*8
    shl.b32 %r58, %r5, 3;           // tid * 8
    add.u32 %r58, %r58, 256;
    add.u32 %r58, %r56, %r58;
    st.shared.u8 [%r58 + 0], %r27;  // sc[0]
    st.shared.u8 [%r58 + 1], %r28;  // sc[1]
    st.shared.u8 [%r58 + 2], %r29;  // sc[2]
    st.shared.u8 [%r58 + 3], %r30;  // sc[3]
    st.shared.u8 [%r58 + 4], %r39;  // sc[4]
    st.shared.u8 [%r58 + 5], %r40;  // sc[5]
    st.shared.u8 [%r58 + 6], %r49;  // sc[6]
    st.shared.u8 [%r58 + 7], %r50;  // sc[7]
    // smem_mv8[tid][8] at smem_buf + 320 + tid*8
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
    bar.sync 0;   // all threads wait for header stores

    // ---- Inner sub-block loop (warp-striped over k_blocks*8 sub-blocks) --
    // global_sub_idx = kb*8 + mw; warp w handles sub-blocks where global_sub%4 == warp_id.
    mov.u32 %r60, 0;                 // mw = 0

MW_LOOP:
    setp.ge.u32 %p5, %r60, 8;
    @%p5 bra MW_DONE;

    // Check warp assignment: (kb*8 + mw) % 4 == warp_id?
    mul.lo.u32 %r61, %r21, 8;
    add.u32    %r61, %r61, %r60;    // global_sub = kb*8 + mw
    and.b32    %r62, %r61, 3;       // global_sub % 4
    setp.eq.u32 %p6, %r62, %r7;
    @!%p6 bra MW_NEXT;

    // ---- Load Q8_1 activation INT8 for this sub-block -------------------
    mul.lo.u32 %r63, %r21, 8;
    add.u32    %r63, %r63, %r60;    // kb*8 + mw
    shl.b32    %r63, %r63, 5;       // * 32 bytes
    add.u32    %r63, %r63, %r9;     // + lane&16 (k-half selector)
    cvt.u64.u32 %rd7, %r63;
    add.u64    %rd7, %rd4, %rd7;   // x_qs row + sub-block offset

    mov.u32 %r64, 0; mov.u32 %r65, 0;
    mov.u32 %r66, 0; mov.u32 %r67, 0;
    @%p2 ld.global.nc.b32 %r64, [%rd7 + 0];
    @%p2 ld.global.nc.b32 %r65, [%rd7 + 4];
    @%p2 ld.global.nc.b32 %r66, [%rd7 + 8];
    @%p2 ld.global.nc.b32 %r67, [%rd7 + 12];

    // ---- Load Q8_1 scale for this sub-block -----------------------------
    mul.lo.u32 %r68, %r21, 8;
    add.u32    %r68, %r68, %r60;
    shl.b32    %r68, %r68, 2;       // * 4 bytes (one u32 per block)
    cvt.u64.u32 %rd8, %r68;
    add.u64    %rd8, %rd5, %rd8;
    mov.b16    %h2, 0;
    @%p2 ld.global.nc.b16 %h2, [%rd8];  // f16 scale from low 16 bits
    cvt.f32.f16 %f2, %h2;

    // ---- Activation partial sum for min-scale correction ----------------
    // Sum 16 signed INT8 bytes from a0..a3 for this k-half.
    // Both k-halves combined via warp butterfly.
    mov.u32 %r69, 0;
    shl.b32 %r70, %r64, 24; shr.s32 %r70, %r70, 24; add.s32 %r69, %r69, %r70;
    shl.b32 %r70, %r64, 16; shr.s32 %r70, %r70, 24; add.s32 %r69, %r69, %r70;
    shl.b32 %r70, %r64, 8;  shr.s32 %r70, %r70, 24; add.s32 %r69, %r69, %r70;
    shr.s32 %r70, %r64, 24;                           add.s32 %r69, %r69, %r70;
    shl.b32 %r70, %r65, 24; shr.s32 %r70, %r70, 24; add.s32 %r69, %r69, %r70;
    shl.b32 %r70, %r65, 16; shr.s32 %r70, %r70, 24; add.s32 %r69, %r69, %r70;
    shl.b32 %r70, %r65, 8;  shr.s32 %r70, %r70, 24; add.s32 %r69, %r69, %r70;
    shr.s32 %r70, %r65, 24;                           add.s32 %r69, %r69, %r70;
    shl.b32 %r70, %r66, 24; shr.s32 %r70, %r70, 24; add.s32 %r69, %r69, %r70;
    shl.b32 %r70, %r66, 16; shr.s32 %r70, %r70, 24; add.s32 %r69, %r69, %r70;
    shl.b32 %r70, %r66, 8;  shr.s32 %r70, %r70, 24; add.s32 %r69, %r69, %r70;
    shr.s32 %r70, %r66, 24;                           add.s32 %r69, %r69, %r70;
    shl.b32 %r70, %r67, 24; shr.s32 %r70, %r70, 24; add.s32 %r69, %r69, %r70;
    shl.b32 %r70, %r67, 16; shr.s32 %r70, %r70, 24; add.s32 %r69, %r69, %r70;
    shl.b32 %r70, %r67, 8;  shr.s32 %r70, %r70, 24; add.s32 %r69, %r69, %r70;
    shr.s32 %r70, %r67, 24;                           add.s32 %r69, %r69, %r70;
    // Combine k-halves: butterfly XOR 16 => sum all 32 elements
    shfl.sync.bfly.b32 %r71, %r69, 16, 31, 0xFFFFFFFF;
    add.s32 %r69, %r69, %r71;       // full row_sum over 32 elements

    // ---- Load Q4K nibbles and dequant to INT8 for B matrix --------------
    // B layout: neuron=lane%8 (r12), k-group=lane/8 (r13), mw=%r60
    and.b32 %r72, %r60, 3;          // mw % 4
    shl.b32 %r72, %r72, 5;          // * 32 (byte offset within qs area)
    shl.b32 %r73, %r13, 2;          // (lane/8) * 4 (4-byte group in the 32)

    // Weight byte offset: neuron * k_blocks * 144 + kb * 144 + 16 + mw_offset + k_offset
    mul.lo.u32 %r74, %r14, %r1;     // global_neuron * k_blocks
    mul.lo.u32 %r74, %r74, 144;
    mad.lo.u32 %r74, %r21, 144, %r74; // + kb * 144
    add.u32    %r74, %r74, 16;      // + qs_area_start
    add.u32    %r74, %r74, %r72;    // + mw_byte_offset
    add.u32    %r75, %r74, %r73;    // + k_byte_offset (b0: k-half 0)
    add.u32    %r76, %r75, 16;      // + 16 (b1: k-half 1)

    cvt.u64.u32 %rd9,  %r75; add.u64 %rd9,  %rd0, %rd9;
    cvt.u64.u32 %rd10, %r76; add.u64 %rd10, %rd0, %rd10;
    ld.global.nc.b32 %r77, [%rd9];   // raw nibbles for b0
    ld.global.nc.b32 %r78, [%rd10];  // raw nibbles for b1

    // Dequant nibbles to signed INT8: use hi nibbles when mw >= 4
    setp.ge.u32 %p7, %r60, 4;
    mov.u32 %r79, 0x0F0F0F0F;
    mov.u32 %r80, 0x08080808;
    and.b32 %r81, %r77, %r79;
    shr.u32 %r82, %r77, 4; and.b32 %r82, %r82, %r79;
    selp.u32 %r83, %r82, %r81, %p7; // select hi vs lo nibbles
    sub.u32  %r83, %r83, %r80;       // subtract 8: range -8..7
    and.b32 %r84, %r78, %r79;
    shr.u32 %r85, %r78, 4; and.b32 %r85, %r85, %r79;
    selp.u32 %r86, %r85, %r84, %p7;
    sub.u32  %r86, %r86, %r80;

    // ---- Load per-neuron Q4K scales from smem for this sub-block --------
    // d_col_base = %r11 (0 or 4), loads 4 consecutive neurons' scales.
    mov.u32 %r87, smem_buf;
    shl.b32 %r88, %r11, 2;          // d_col_base * 4 bytes
    add.u32 %r89, %r87, %r88;
    ld.shared.f32 %f3,  [%r89 + 0];   // d[n+0]
    ld.shared.f32 %f4,  [%r89 + 4];   // d[n+1]
    ld.shared.f32 %f5,  [%r89 + 8];   // d[n+2]
    ld.shared.f32 %f6,  [%r89 + 12];  // d[n+3]
    ld.shared.f32 %f7,  [%r89 + 128]; // dmin[n+0]
    ld.shared.f32 %f8,  [%r89 + 132]; // dmin[n+1]
    ld.shared.f32 %f9,  [%r89 + 136]; // dmin[n+2]
    ld.shared.f32 %f10, [%r89 + 140]; // dmin[n+3]

    // smem_sc8[n_local][mw]: base at smem_buf + 256 + n_local*8, index by mw
    shl.b32 %r90, %r11, 3;          // d_col_base * 8
    add.u32 %r91, %r87, %r90;
    add.u32 %r91, %r91, 256;        // smem_sc8 base for n=d_col_base
    add.u32 %r92, %r91, %r60;       // + mw
    ld.shared.u8 %r93,  [%r92 + 0];   // sc[n+0][mw]
    ld.shared.u8 %r94,  [%r92 + 8];   // sc[n+1][mw]
    ld.shared.u8 %r95,  [%r92 + 16];  // sc[n+2][mw]
    ld.shared.u8 %r96,  [%r92 + 24];  // sc[n+3][mw]
    add.u32 %r97, %r92, 64;           // smem_mv8 offset
    ld.shared.u8 %r98,  [%r97 + 0];   // mv[n+0][mw]
    ld.shared.u8 %r99,  [%r97 + 8];   // mv[n+1][mw]
    ld.shared.u8 %r100, [%r97 + 16];  // mv[n+2][mw]
    ld.shared.u8 %r101, [%r97 + 24];  // mv[n+3][mw]

    // scale_pos[0..3] = d[n+i] * (sc[n+i][mw] / 32) * d_q8
    // 1/32 = 0f3D000000
    cvt.rn.f32.u32 %f11, %r93; mul.f32 %f11, %f11, 0f3D000000; mul.f32 %f11, %f3, %f11; mul.f32 %f11, %f11, %f2;
    cvt.rn.f32.u32 %f12, %r94; mul.f32 %f12, %f12, 0f3D000000; mul.f32 %f12, %f4, %f12; mul.f32 %f12, %f12, %f2;
    cvt.rn.f32.u32 %f13, %r95; mul.f32 %f13, %f13, 0f3D000000; mul.f32 %f13, %f5, %f13; mul.f32 %f13, %f13, %f2;
    cvt.rn.f32.u32 %f14, %r96; mul.f32 %f14, %f14, 0f3D000000; mul.f32 %f14, %f6, %f14; mul.f32 %f14, %f14, %f2;
    // scale_min[0..3] = dmin[n+i] * (mv[n+i][mw] / 32) * d_q8
    cvt.rn.f32.u32 %f15, %r98;  mul.f32 %f15, %f15, 0f3D000000; mul.f32 %f15, %f7,  %f15; mul.f32 %f15, %f15, %f2;
    cvt.rn.f32.u32 %f16, %r99;  mul.f32 %f16, %f16, 0f3D000000; mul.f32 %f16, %f8,  %f16; mul.f32 %f16, %f16, %f2;
    cvt.rn.f32.u32 %f17, %r100; mul.f32 %f17, %f17, 0f3D000000; mul.f32 %f17, %f9,  %f17; mul.f32 %f17, %f17, %f2;
    cvt.rn.f32.u32 %f18, %r101; mul.f32 %f18, %f18, 0f3D000000; mul.f32 %f18, %f10, %f18; mul.f32 %f18, %f18, %f2;

    // min correction: row_sum_f * scale_min[0..3]
    cvt.rn.f32.s32 %f19, %r69;
    mul.f32 %f20, %f19, %f15;
    mul.f32 %f21, %f19, %f16;
    mul.f32 %f22, %f19, %f17;
    mul.f32 %f23, %f19, %f18;

    // ---- MMA: D = A * B (accumulate from zero each sub-block) -----------
    // mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
    // A: m16 x k32 (s8), 4 regs: %r64..%r67
    // B: k32 x n8 (s8, col-major), 2 regs: %r83, %r86
    // D: m16 x n8 (s32), 4 regs: accumulated into smem_partial
    mov.u32 %r102, 0; mov.u32 %r103, 0;
    mov.u32 %r104, 0; mov.u32 %r105, 0;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r102, %r103, %r104, %r105},
        {%r64,  %r65,  %r66,  %r67},
        {%r83,  %r86},
        {%r102, %r103, %r104, %r105};

    // Scale and accumulate into smem_partial
    cvt.rn.f32.s32 %f24, %r102; mul.f32 %f24, %f24, %f11; sub.f32 %f24, %f24, %f20;
    cvt.rn.f32.s32 %f25, %r103; mul.f32 %f25, %f25, %f12; sub.f32 %f25, %f25, %f21;
    cvt.rn.f32.s32 %f26, %r104; mul.f32 %f26, %f26, %f13; sub.f32 %f26, %f26, %f22;
    cvt.rn.f32.s32 %f27, %r105; mul.f32 %f27, %f27, %f14; sub.f32 %f27, %f27, %f23;

    ld.shared.f32 %f28, [%r17 + 0];  add.f32 %f28, %f28, %f24;  st.shared.f32 [%r17 + 0],  %f28;
    ld.shared.f32 %f29, [%r17 + 4];  add.f32 %f29, %f29, %f25;  st.shared.f32 [%r17 + 4],  %f29;
    ld.shared.f32 %f30, [%r17 + 8];  add.f32 %f30, %f30, %f26;  st.shared.f32 [%r17 + 8],  %f30;
    ld.shared.f32 %f31, [%r17 + 12]; add.f32 %f31, %f31, %f27;  st.shared.f32 [%r17 + 12], %f31;

MW_NEXT:
    add.u32 %r60, %r60, 1;
    bra MW_LOOP;

MW_DONE:
    add.u32 %r21, %r21, 1;
    bra KB_LOOP;

// =========================================================================
// REDUCTION: warp 0 sums smem_partial[0..3] and writes to global out
// =========================================================================
KB_DONE:
    bar.sync 0;   // ensure all K-loop smem_partial writes are visible

    // Only warp 0, valid rows do the reduction and global store.
    // All threads (including those that branch here) reconverge at SKIP_WRITE.
    @!%p3 bra SKIP_WRITE;  // non-warp-0: skip to barrier
    @!%p2 bra SKIP_WRITE;  // warp-0, invalid row: skip to barrier

    // Warp 0: read partial sums from all 4 warps.
    // %r17 = smem_partial slot for warp 0 (warp_id=0 -> %r15=0).
    // Warp w is at offset w*512 from warp-0 base.
    ld.shared.f32 %f32, [%r17 + 0];
    ld.shared.f32 %f33, [%r17 + 4];
    ld.shared.f32 %f34, [%r17 + 8];
    ld.shared.f32 %f35, [%r17 + 12];
    ld.shared.f32 %f36, [%r17 + 512];
    ld.shared.f32 %f37, [%r17 + 516];
    ld.shared.f32 %f38, [%r17 + 520];
    ld.shared.f32 %f39, [%r17 + 524];
    ld.shared.f32 %f40, [%r17 + 1024];
    ld.shared.f32 %f41, [%r17 + 1028];
    ld.shared.f32 %f42, [%r17 + 1032];
    ld.shared.f32 %f43, [%r17 + 1036];
    ld.shared.f32 %f44, [%r17 + 1536];
    ld.shared.f32 %f45, [%r17 + 1540];
    ld.shared.f32 %f46, [%r17 + 1544];
    ld.shared.f32 %f47, [%r17 + 1548];

    add.f32 %f32, %f32, %f36; add.f32 %f32, %f32, %f40; add.f32 %f32, %f32, %f44;
    add.f32 %f33, %f33, %f37; add.f32 %f33, %f33, %f41; add.f32 %f33, %f33, %f45;
    add.f32 %f34, %f34, %f38; add.f32 %f34, %f34, %f42; add.f32 %f34, %f34, %f46;
    add.f32 %f35, %f35, %f39; add.f32 %f35, %f35, %f43; add.f32 %f35, %f35, %f47;

    // Write to global out[d_row * N + n_tile_base + d_col_base + 0..3]
    add.u32 %r106, %r3, %r11;           // n_tile_base + d_col_base
    mad.lo.u32 %r107, %r8, %r0, %r106;  // d_row * N + n_tile_base + d_col_base
    shl.b32    %r107, %r107, 2;          // * 4 bytes
    cvt.u64.u32 %rd11, %r107;
    add.u64    %rd11, %rd3, %rd11;
    st.global.f32 [%rd11 + 0],  %f32;
    st.global.f32 [%rd11 + 4],  %f33;
    st.global.f32 [%rd11 + 8],  %f34;
    st.global.f32 [%rd11 + 12], %f35;

SKIP_WRITE:
    // All 128 threads (warp-0 after write, others from early branch) meet here.
    // bar.sync 1 ensures warp-0 has finished all smem_partial reads before
    // any thread zeros smem_partial for the next tile.
    bar.sync 1;

    add.u32 %r136, %r136, 1;
    bra TILE_LOOP;

DONE:
    ret;
}
"#;
