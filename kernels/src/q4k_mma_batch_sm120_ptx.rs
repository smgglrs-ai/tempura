/// Hand-tuned PTX kernel for Q4K x Q8_1 batch GEMM using INT8 tensor cores (sm_120).
///
/// # Design: 8-warp K-parallel with mma.sync.aligned.m16n8k32
///
/// Grid:  (ceil(N/8), 1, 1) -- one block per n8-column output tile
/// Block: (256, 1, 1) -- 8 warps
///
/// Each block processes all M activation rows against one n8-column weight tile.
/// The 8 sub-blocks of each k_block are striped across the 8 warps:
///   warp w handles sub-blocks with (kb*8 + mw) % 8 == w
/// (originally 4 warps with a %4 stripe; converted to the streamk-proven
/// 8-warp geometry 2026-11-17 while fixing the fragment mapping -- the 4-warp
/// stripe produced deterministic odd-column corruption the probe caught).
///
/// # Fragment layout (m16n8k32.row.col.s8.s8.s32 -- PTX ISA)
///
/// FIXED 2026-11-16f: the original version invented a lane%16/lane/16-based
/// mapping ("confirmed via nvcc PTX output" -- it was not); every (reg, byte)
/// landed on the wrong (row, k) / (k, col) / (row, col), which no amount of
/// nibble-layout fixing could repair (the mxf4-gemv fragment-mapping lesson:
/// any store permutation "passes" uniform data, and random data shows
/// 15-40% rel error). The TRUE layout, with
///   tid = lane & 3 (threadID_in_group), gid = lane >> 2 (groupID):
///
/// A (m16 x k32, s8, 4 regs of 4 bytes; byte b of reg = k+b):
///   a0: row = gid,     k = tid*4 + 0..3
///   a1: row = gid + 8, k = tid*4 + 0..3
///   a2: row = gid,     k = 16 + tid*4 + 0..3
///   a3: row = gid + 8, k = 16 + tid*4 + 0..3
///
/// B (k32 x n8, s8, 2 regs of 4 bytes):
///   b0: col = gid, k = tid*4 + 0..3
///   b1: col = gid, k = 16 + tid*4 + 0..3
///
/// D (m16 x n8, s32, 4 regs):
///   d_j: row = gid + 8*(j>>1), col = 2*tid + (j&1)
///   (the mxf4-gemv mma_frag::acc_coords canonical mapping)
///
/// NOTE: A and B use the same (reg -> k-half) assignment, so the k-pairing
/// of products is layout-consistent; the load-bearing mappings are the
/// A rows, the B column, and the D (row, col) -- all fixed above.
///
/// # Q4K weight format (native GGUF 36-u32 = 144 bytes per block)
///
/// bytes  0-1:  d    (f16 super-scale)
/// bytes  2-3:  dmin (f16 super-min)
/// bytes 4-15:  scales[12] (8 x sc + 8 x m_val, 6-bit each)
/// bytes 16-143: qs[128] (256 nibbles, 2 per byte)
///   Byte group j (0..3) = qs[32j..32j+31] holds sub-blocks 2j and 2j+1:
///     sub-block 2j:   elements 32j..32j+31    = LO nibbles of qs[32j+b]
///     sub-block 2j+1: elements 32j+32..32j+63 = HI nibbles of qs[32j+b]
///   (the native-GGUF layout the kpar8 reference parses; the pre-2026-11-16f
///   version read byte group mw%4 with hi-select mw>=4, pairing nibbles with
///   the wrong activation elements.)
///
/// Dequant convention (kpar8-identical, fixed 2026-11-16f): the nibble is the
/// RAW unsigned value 0..15 (no -8 offset) and sc/mv are RAW 6-bit integers
/// (no /32): value = d*sc*nibble - dmin*mv, applied per 32-element sub-block
/// against the matching Q8_1 block (scale d8, min term dmin*mv*sum(q8)).
///
/// # Q8_1 activation format
///
/// x_qs: [M rows][k_blocks * 8 Q8_1-blocks][8 u32 = 32 INT8]
/// x_ds: [M rows][k_blocks * 8 Q8_1-blocks][1 u32 with f16 in low bits]
///
/// # K-parallel reduction
///
/// Each warp accumulates partial f32 results for the m16 x n8 tile in
/// smem_partial[8][m16*n8]; warp 0 sums the 8 partials and writes the global
/// output. The per-row min term (dmin*mv*sum of the row's 32 Q8_1 bytes) is
/// computed per thread via a 2-step XOR butterfly over the 4-lane group
/// (each row's 32 bytes are spread across the quad's 4 threads).

pub const MIN_CC: u32 = 1200;
pub const ENTRY: &str  = "q4k_mma_batch_sm120";

pub const PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// q4k_mma_batch_sm120
//
// 4-warp K-parallel INT8 tensor-core batch GEMM for Q4K decode.
// Grid:  (ceil(N/8), 1, 1) -- one block per n8 output tile
// Block: (256,      1, 1) -- 8 warps, K-parallel
//
// Output: out[m][n] for m=0..M-1, n in [blockIdx.x*8 .. blockIdx.x*8+7]
//
// Parameters:
//   param_weights:    *const u32  -- native GGUF Q4K [N * k_blocks * 36 u32]
//   param_act_qs:     *const u32  -- Q8_1 INT8 activations [M * k_blocks * 64 u32]
//   param_act_ds:     *const u32  -- Q8_1 block scales [M * k_blocks * 8 u32]
//   param_out:        *mut f32    -- output [M * N]
//   param_n:          u32         -- N (output neurons, multiple of 8)
//   param_k_blocks:   u32         -- k_blocks (K = k_blocks * 256)
//   param_m:          u32         -- M (actual activation rows, <= 16)
.visible .entry q4k_mma_batch_sm120(
    .param .u64 param_weights,
    .param .u64 param_act_qs,
    .param .u64 param_act_ds,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .u32 param_k_blocks,
    .param .u32 param_m
)
{
    // ---- Register declarations -----------------------------------------------
    .reg .u32  %r<160>;
    .reg .u64  %rd<40>;
    .reg .f32  %f<80>;
    .reg .pred %p<16>;
    .reg .b16  %h<6>;

    // ---- Shared memory layout ------------------------------------------------
    // [0..127]:   smem_d8[8]   f32 -- d values for 8 neurons in n8 tile
    // [128..255]: smem_dmin8[8] f32 -- dmin values for 8 neurons
    // [256..319]: smem_sc8[8 neurons][8 sub-blocks]  u8 = 64 bytes
    // [320..383]: smem_mv8[8 neurons][8 sub-blocks]  u8 = 64 bytes
    // [384..4479]: smem_partial[8 warps][16 rows][8 cols] f32 = 4096 bytes
    //   slot(row, col) = 384 + warp_id*512 + (row*8 + col)*4
    .shared .align 4 .b8 smem_buf[4480];

    // ---- Load kernel parameters ----------------------------------------------
    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_act_qs];
    ld.param.u64 %rd2, [param_act_ds];
    ld.param.u64 %rd3, [param_out];
    ld.param.u32 %r0,  [param_n];
    ld.param.u32 %r1,  [param_k_blocks];
    ld.param.u32 %r2,  [param_m];

    // ---- Validate block: n_tile_base < N ------------------------------------
    mov.u32 %r3, %ctaid.x;       // blockIdx.x = n8 tile index
    shl.b32 %r4, %r3, 3;         // n_tile_base = blockIdx.x * 8
    setp.ge.u32 %p0, %r4, %r0;
    @%p0 bra DONE;

    // ---- Thread decomposition (TRUE m16n8k32 fragment layout) ----------------
    mov.u32 %r5, %tid.x;          // flat thread (0..127)
    and.b32 %r6, %r5, 31;         // lane within warp (0..31)
    shr.u32 %r7, %r5, 5;          // warp_id (0..7)
    and.b32 %r8, %r6, 3;          // tid = lane & 3 (threadID_in_group)
    shr.u32 %r9, %r6, 2;          // gid = lane >> 2 (groupID, 0..7)
    add.u32 %r140, %r9, 8;        // gid + 8 (A rows a1/a3, D rows d2/d3)
    shl.b32 %r141, %r8, 1;        // col0 = 2*tid (D cols d0/d2)

    // Predicates
    // p1: tid.x < 8 (this thread loads the Q4K header for one of 8 neurons)
    setp.lt.u32 %p1, %r5, 8;
    // p2: gid < m (row gid valid)
    setp.lt.u32 %p2, %r9, %r2;
    // p3: warp_id == 0 (only warp 0 does the final reduction + global write)
    setp.eq.u32 %p3, %r7, 0;
    // p4: gid + 8 < m (row gid+8 valid)
    setp.lt.u32 %p4, %r140, %r2;

    // ---- smem_partial slots for this thread's 4 D elements --------------------
    // D elements: rows {gid, gid+8} x cols {2*tid, 2*tid+1}.
    // slot(row, col) = smem_buf + 384 + warp_id*512 + (row*8 + col)*4
    //   s0 = slot(gid, col0); s1 = s0 + 4; s2 = s0 + 256; s3 = s2 + 4
    shl.b32 %r15, %r7, 9;         // warp_id * 512
    shl.b32 %r142, %r9, 5;        // gid * 32
    shl.b32 %r143, %r8, 3;        // (2*tid) * 4
    add.u32  %r144, %r142, %r143;
    add.u32  %r144, %r144, %r15;
    add.u32  %r144, %r144, 384;
    mov.u32 %r18, smem_buf;
    add.u32 %r17, %r18, %r144;    // s0 (slot(gid, col0))

    st.shared.f32 [%r17 + 0],   0f00000000;
    st.shared.f32 [%r17 + 4],   0f00000000;
    st.shared.f32 [%r17 + 256], 0f00000000;
    st.shared.f32 [%r17 + 260], 0f00000000;

    // ---- Activation row base pointers (rows gid and gid+8) --------------------
    // x_qs row stride: k_blocks * 64 u32 = k_blocks * 256 bytes
    mul.lo.u32 %r19, %r9, %r1;    // gid * k_blocks
    shl.b32    %r19, %r19, 8;     // * 256 bytes
    mul.lo.u32 %r20, %r140, %r1;  // (gid+8) * k_blocks
    shl.b32    %r20, %r20, 8;
    cvt.u64.u32 %rd4, %r19;
    add.u64    %rd4, %rd1, %rd4;  // x_qs row gid
    cvt.u64.u32 %rd12, %r20;
    add.u64    %rd12, %rd1, %rd12; // x_qs row gid+8
    // x_ds row stride: k_blocks * 8 u32 = k_blocks * 32 bytes
    mul.lo.u32 %r19, %r9, %r1;
    shl.b32    %r19, %r19, 5;
    mul.lo.u32 %r20, %r140, %r1;
    shl.b32    %r20, %r20, 5;
    cvt.u64.u32 %rd5, %r19;
    add.u64    %rd5, %rd2, %rd5;  // x_ds row gid
    cvt.u64.u32 %rd13, %r20;
    add.u64    %rd13, %rd2, %rd13; // x_ds row gid+8

    // k byte offset within a 16-byte half: tid * 4
    shl.b32 %r145, %r8, 2;

    // B column (neuron) for this thread: col = gid
    add.u32 %r12, %r4, %r9;       // n_tile_base + gid

    // ---- Outer loop: k_blocks Q4K blocks ------------------------------------
    mov.u32 %r21, 0;               // kb = 0

KB_LOOP:
    setp.ge.u32 %p5, %r21, %r1;
    @%p5 bra KB_DONE;

    // ---- Header load by first 8 threads (tid.x < 8) --------------------------
    // Thread tid loads the Q4K block header for neuron n_tile_base+tid.
    @!%p1 bra AFTER_HEADER;

    // Q4K block byte offset: (n_tile_base + tid) * k_blocks * 144 + kb * 144
    mul.lo.u32 %r22, %r5, %r1;    // tid * k_blocks
    mul.lo.u32 %r22, %r22, 144;   // * 144
    mad.lo.u32 %r22, %r21, 144, %r22; // + kb*144
    mul.lo.u32 %r23, %r4, %r1;    // n_tile_base * k_blocks
    mul.lo.u32 %r23, %r23, 144;
    add.u32    %r22, %r22, %r23;
    cvt.u64.u32 %rd6, %r22;
    add.u64 %rd6, %rd0, %rd6;     // Q4K block base for (n_tile_base+tid, kb)

    // Load d, dmin
    ld.global.nc.b16 %h0, [%rd6 + 0];
    ld.global.nc.b16 %h1, [%rd6 + 2];
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;

    // Load scale words: ps0 bytes 4-7, ps1 bytes 8-11, ps2 bytes 12-15
    ld.global.nc.b32 %r24, [%rd6 + 4];
    ld.global.nc.b32 %r25, [%rd6 + 8];
    ld.global.nc.b32 %r26, [%rd6 + 12];

    // Extract sc[0..7] and m_val[0..7] (6-bit values from 12-byte scales field)
    // sc[0] = ps0 & 63, sc[1] = (ps0>>8) & 63
    // sc[2] = (ps0>>16)&63, sc[3] = (ps0>>24)&63
    and.b32 %r27, %r24, 63;
    shr.u32 %r28, %r24, 8;  and.b32 %r28, %r28, 63;
    shr.u32 %r29, %r24, 16; and.b32 %r29, %r29, 63;
    shr.u32 %r30, %r24, 24; and.b32 %r30, %r30, 63;
    // m_val[0..3] from ps1:
    and.b32 %r31, %r25, 63;
    shr.u32 %r32, %r25, 8;  and.b32 %r32, %r32, 63;
    shr.u32 %r33, %r25, 16; and.b32 %r33, %r33, 63;
    shr.u32 %r34, %r25, 24; and.b32 %r34, %r34, 63;

    // sc[4..7] and m_val[4..7]: cross-byte from all 3 words
    // For sub 4,5 (shift=0): b8=(ps2>>0)&0xFF, b9=(ps2>>8)&0xFF
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
    // For sub 6,7 (shift=16):
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

    // Store to smem: neuron index = tid (0..7 = r5)
    shl.b32 %r55, %r5, 2;         // tid * 4 bytes
    mov.u32 %r56, smem_buf;
    add.u32 %r57, %r56, %r55;
    st.shared.f32 [%r57 + 0],   %f0;   // smem_d8[tid]
    st.shared.f32 [%r57 + 128], %f1;   // smem_dmin8[tid]

    // smem_sc8[tid][8]: at smem_buf + 256 + tid*8
    shl.b32 %r58, %r5, 3;         // tid * 8
    add.u32 %r58, %r58, 256;
    add.u32 %r58, %r56, %r58;
    st.shared.u8 [%r58 + 0], %r27;   // sc[0]
    st.shared.u8 [%r58 + 1], %r28;   // sc[1]
    st.shared.u8 [%r58 + 2], %r29;   // sc[2]
    st.shared.u8 [%r58 + 3], %r30;   // sc[3]
    st.shared.u8 [%r58 + 4], %r39;   // sc[4]
    st.shared.u8 [%r58 + 5], %r40;   // sc[5]
    st.shared.u8 [%r58 + 6], %r49;   // sc[6]
    st.shared.u8 [%r58 + 7], %r50;   // sc[7]
    // smem_mv8[tid][8]: at smem_buf + 320 + tid*8
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
    bar.sync 0;    // all threads wait for headers

    // ---- Inner loop: sub-blocks, warp-strided --------------------------------
    // warp w handles sub-blocks with (kb*8 + mw) % 8 == warp_id.

    mov.u32 %r60, 0;               // mw = 0 (sub-block index 0..7)

MW_LOOP:
    setp.ge.u32 %p6, %r60, 8;
    @%p6 bra MW_DONE;

    // Check if this (kb, mw) belongs to warp_id
    mul.lo.u32 %r61, %r21, 8;
    add.u32    %r61, %r61, %r60;   // global_sub = kb*8 + mw
    and.b32    %r62, %r61, 7;      // global_sub % 8
    setp.eq.u32 %p8, %r62, %r7;   // %p8 = (this sub belongs to warp_id)
    @!%p8 bra MW_NEXT;

    // ---- Load Q8_1 activation fragments (A) ----------------------------------
    // a0: row gid,     k half 0 (bytes tid*4..+3)
    // a1: row gid+8,   k half 0
    // a2: row gid,     k half 1 (bytes 16+tid*4..+3)
    // a3: row gid+8,   k half 1
    mul.lo.u32 %r63, %r21, 8;
    add.u32    %r63, %r63, %r60;   // kb*8+mw
    shl.b32    %r63, %r63, 5;      // * 32 bytes (q8 block stride)
    cvt.u64.u32 %rd14, %r63;      // q8 block byte offset
    add.u64 %rd7,  %rd4,  %rd14;  // row gid block base
    add.u64 %rd14, %rd12, %rd14;  // row gid+8 block base

    mov.u32 %r64, 0; mov.u32 %r65, 0;
    mov.u32 %r66, 0; mov.u32 %r67, 0;
    cvt.u64.u32 %rd17, %r145;
    add.u64 %rd18, %rd7,  %rd17;    // a0 address (row gid, k half 0)
    add.u64 %rd19, %rd14, %rd17;    // a1 address (row gid+8, k half 0)
    @%p2 ld.global.nc.b32 %r64, [%rd18];
    @%p4 ld.global.nc.b32 %r65, [%rd19];
    @%p2 ld.global.nc.b32 %r66, [%rd18 + 16];
    @%p4 ld.global.nc.b32 %r67, [%rd19 + 16];

    // ---- Load Q8_1 scales for rows gid (f2) and gid+8 (f40) ------------------
    mul.lo.u32 %r68, %r21, 8;
    add.u32    %r68, %r68, %r60;   // kb*8+mw
    shl.b32    %r68, %r68, 2;      // * 4 bytes (one u32 per q8 block)
    cvt.u64.u32 %rd8, %r68;
    add.u64 %rd15, %rd13, %rd8;    // row gid+8 ds address
    add.u64 %rd8,  %rd5,  %rd8;    // row gid ds address
    mov.b16    %h2, 0;
    mov.b16    %h3, 0;
    @%p2 ld.global.nc.b16 %h2, [%rd8];   // f16 scale, row gid
    @%p4 ld.global.nc.b16 %h3, [%rd15];  // f16 scale, row gid+8
    cvt.f32.f16 %f2,  %h2;
    cvt.f32.f16 %f40, %h3;

    // ---- Row sums for the min term -------------------------------------------
    // s_lo (r69): row gid partial = bytesum(a0) + bytesum(a2)
    // s_hi (r70): row gid+8 partial = bytesum(a1) + bytesum(a3)
    // Full 32-byte row sums via a 2-step XOR butterfly over the 4-lane group
    // (the row's 32 bytes are spread across the quad's 4 threads).
    mov.u32 %r69, 0;
    shl.b32 %r71, %r64, 24; shr.s32 %r71, %r71, 24; add.s32 %r69, %r69, %r71;
    shl.b32 %r71, %r64, 16; shr.s32 %r71, %r71, 24; add.s32 %r69, %r69, %r71;
    shl.b32 %r71, %r64, 8;  shr.s32 %r71, %r71, 24; add.s32 %r69, %r69, %r71;
    shr.s32 %r71, %r64, 24;                            add.s32 %r69, %r69, %r71;
    shl.b32 %r71, %r66, 24; shr.s32 %r71, %r71, 24; add.s32 %r69, %r69, %r71;
    shl.b32 %r71, %r66, 16; shr.s32 %r71, %r71, 24; add.s32 %r69, %r69, %r71;
    shl.b32 %r71, %r66, 8;  shr.s32 %r71, %r71, 24; add.s32 %r69, %r69, %r71;
    shr.s32 %r71, %r66, 24;                            add.s32 %r69, %r69, %r71;
    mov.u32 %r70, 0;
    shl.b32 %r71, %r65, 24; shr.s32 %r71, %r71, 24; add.s32 %r70, %r70, %r71;
    shl.b32 %r71, %r65, 16; shr.s32 %r71, %r71, 24; add.s32 %r70, %r70, %r71;
    shl.b32 %r71, %r65, 8;  shr.s32 %r71, %r71, 24; add.s32 %r70, %r70, %r71;
    shr.s32 %r71, %r65, 24;                            add.s32 %r70, %r70, %r71;
    shl.b32 %r71, %r67, 24; shr.s32 %r71, %r71, 24; add.s32 %r70, %r70, %r71;
    shl.b32 %r71, %r67, 16; shr.s32 %r71, %r71, 24; add.s32 %r70, %r70, %r71;
    shl.b32 %r71, %r67, 8;  shr.s32 %r71, %r71, 24; add.s32 %r70, %r70, %r71;
    shr.s32 %r71, %r67, 24;                            add.s32 %r70, %r70, %r71;

    shfl.sync.bfly.b32 %r72, %r69, 1, 31, 0xFFFFFFFF;
    add.s32 %r69, %r69, %r72;
    shfl.sync.bfly.b32 %r72, %r69, 2, 31, 0xFFFFFFFF;
    add.s32 %r69, %r69, %r72;     // rowsum(row gid), all 32 q8 bytes
    shfl.sync.bfly.b32 %r72, %r70, 1, 31, 0xFFFFFFFF;
    add.s32 %r70, %r70, %r72;
    shfl.sync.bfly.b32 %r72, %r70, 2, 31, 0xFFFFFFFF;
    add.s32 %r70, %r70, %r72;     // rowsum(row gid+8)

    // ---- Load Q4K nibbles for B (col = gid, k halves tid*4 / 16+tid*4) -------
    // Byte group j = mw>>1 (sub-blocks 2j, 2j+1 share qs[32j..32j+31]);
    // hi nibbles selected when mw is ODD (sub-block 2j+1).
    shr.u32 %r72, %r60, 1;        // byte group j = mw>>1
    shl.b32 %r72, %r72, 5;        // * 32

    // Weight byte offset: neuron * k_blocks * 144 + kb * 144 + 16 + j*32
    mul.lo.u32 %r74, %r12, %r1;   // neuron * k_blocks
    mul.lo.u32 %r74, %r74, 144;
    mad.lo.u32 %r74, %r21, 144, %r74; // + kb * 144
    add.u32    %r74, %r74, 16;    // + qs area start
    add.u32    %r74, %r74, %r72;  // + byte group offset
    add.u32    %r75, %r74, %r145; // + tid*4 (b0: k half 0)
    add.u32    %r76, %r75, 16;    // + 16 (b1: k half 1)

    cvt.u64.u32 %rd9,  %r75; add.u64 %rd9,  %rd0, %rd9;
    cvt.u64.u32 %rd10, %r76; add.u64 %rd10, %rd0, %rd10;
    ld.global.nc.b32 %r77, [%rd9];   // raw nibbles for b0
    ld.global.nc.b32 %r78, [%rd10];  // raw nibbles for b1

    // use_hi = (mw is odd): sub-block 2j+1 lives in the hi nibbles of group j.
    // The nibble IS the value (raw unsigned 0..15, Q4K convention -- no -8
    // offset; the min term carries it via dmin*mv*sum(q8)).
    and.b32 %r130, %r60, 1;
    setp.ne.u32 %p7, %r130, 0;
    mov.u32 %r79, 0x0F0F0F0F;
    and.b32 %r81, %r77, %r79;           // lo nibbles of b0
    shr.u32 %r82, %r77, 4; and.b32 %r82, %r82, %r79; // hi nibbles of b0
    selp.u32 %r83, %r82, %r81, %p7;    // select by mw parity
    and.b32 %r84, %r78, %r79;
    shr.u32 %r85, %r78, 4; and.b32 %r85, %r85, %r79;
    selp.u32 %r86, %r85, %r84, %p7;

    // ---- Per-column Q4K scales (cols col0 = 2*tid and col0+1) ----------------
    shl.b32 %r146, %r141, 2;      // col0 * 4 bytes (f32 tables)
    add.u32 %r147, %r18, %r146;
    ld.shared.f32 %f3,  [%r147];      // d[col0]
    ld.shared.f32 %f4,  [%r147 + 4];  // d[col0+1]
    ld.shared.f32 %f7,  [%r147 + 128]; // dmin[col0]
    ld.shared.f32 %f8,  [%r147 + 132]; // dmin[col0+1]

    // smem_sc8[c][mw] at smem_buf + 256 + c*8 + mw; mv8 at +64
    shl.b32 %r148, %r141, 3;      // col0 * 8
    add.u32 %r149, %r18, %r148;
    add.u32 %r149, %r149, 256;
    add.u32 %r149, %r149, %r60;   // + mw
    ld.shared.u8 %r93,  [%r149];      // sc[col0][mw]
    ld.shared.u8 %r94,  [%r149 + 8];  // sc[col0+1][mw]
    add.u32 %r150, %r149, 64;
    ld.shared.u8 %r98,  [%r150];      // mv[col0][mw]
    ld.shared.u8 %r99,  [%r150 + 8];  // mv[col0+1][mw]

    // pos[row][col] = d[col] * sc[col][mw] * d8[row]   (sc is the RAW 6-bit value)
    // min[row][col] = dmin[col] * mv[col][mw] * d8[row]
    cvt.rn.f32.u32 %f11, %r93;
    cvt.rn.f32.u32 %f12, %r94;
    cvt.rn.f32.u32 %f15, %r98;
    cvt.rn.f32.u32 %f16, %r99;
    mul.f32 %f17, %f3, %f11;  mul.f32 %f17, %f17, %f2;    // pos row-lo col0
    mul.f32 %f18, %f4, %f12;  mul.f32 %f18, %f18, %f2;    // pos row-lo col1
    mul.f32 %f41, %f3, %f11;  mul.f32 %f41, %f41, %f40;   // pos row-hi col0
    mul.f32 %f42, %f4, %f12;  mul.f32 %f42, %f42, %f40;   // pos row-hi col1
    mul.f32 %f19, %f7, %f15;  mul.f32 %f19, %f19, %f2;    // min row-lo col0
    mul.f32 %f20, %f8, %f16;  mul.f32 %f20, %f20, %f2;    // min row-lo col1
    mul.f32 %f43, %f7, %f15;  mul.f32 %f43, %f43, %f40;   // min row-hi col0
    mul.f32 %f44, %f8, %f16;  mul.f32 %f44, %f44, %f40;   // min row-hi col1

    cvt.rn.f32.s32 %f21, %r69;   // rowsum(row gid)
    cvt.rn.f32.s32 %f45, %r70;   // rowsum(row gid+8)

    // ---- MMA: D = A * B + 0 --------------------------------------------------
    mov.u32 %r102, 0; mov.u32 %r103, 0;
    mov.u32 %r104, 0; mov.u32 %r105, 0;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r102, %r103, %r104, %r105},
        {%r64,  %r65,  %r66,  %r67},
        {%r83,  %r86},
        {%r102, %r103, %r104, %r105};

    // ---- Scale + min term, accumulate into smem_partial ----------------------
    // d0: (gid, col0); d1: (gid, col0+1); d2: (gid+8, col0); d3: (gid+8, col0+1)
    cvt.rn.f32.s32 %f24, %r102; mul.f32 %f24, %f24, %f17;
    mul.f32 %f25, %f21, %f19;   sub.f32 %f24, %f24, %f25;
    ld.shared.f32 %f28, [%r17 + 0];   add.f32 %f28, %f28, %f24;  st.shared.f32 [%r17 + 0],   %f28;

    cvt.rn.f32.s32 %f24, %r103; mul.f32 %f24, %f24, %f18;
    mul.f32 %f25, %f21, %f20;   sub.f32 %f24, %f24, %f25;
    ld.shared.f32 %f28, [%r17 + 4];   add.f32 %f28, %f28, %f24;  st.shared.f32 [%r17 + 4],   %f28;

    cvt.rn.f32.s32 %f24, %r104; mul.f32 %f24, %f24, %f41;
    mul.f32 %f25, %f45, %f43;   sub.f32 %f24, %f24, %f25;
    ld.shared.f32 %f28, [%r17 + 256]; add.f32 %f28, %f28, %f24;  st.shared.f32 [%r17 + 256], %f28;

    cvt.rn.f32.s32 %f24, %r105; mul.f32 %f24, %f24, %f42;
    mul.f32 %f25, %f45, %f44;   sub.f32 %f24, %f24, %f25;
    ld.shared.f32 %f28, [%r17 + 260]; add.f32 %f28, %f28, %f24;  st.shared.f32 [%r17 + 260], %f28;

MW_NEXT:
    add.u32 %r60, %r60, 1;
    bra MW_LOOP;

MW_DONE:
    // All warps must finish READING this kb's header tables before any
    // tid<8 thread overwrites them with the kb+1 header at the top of the
    // next iteration (streamk avoids this with its double-buffer + bar in
    // the drain; without this barrier the header load races slower warps'
    // scale lookups -- 9-20% rel error, timing-dependent).
    bar.sync 0;
    add.u32 %r21, %r21, 1;
    bra KB_LOOP;

KB_DONE:
    // ========================================================================
    // REDUCTION: warp 0 sums smem_partial[0..7] and writes to global out
    // ========================================================================
    bar.sync 0;    // all warps done accumulating

    @!%p3 bra DONE;

    // This thread's D elements: rows {gid, gid+8} x cols {col0, col0+1}.
    // Partials from the 8 warps sit at slot + w*512.
    ld.shared.f32 %f50, [%r17 + 0];
    ld.shared.f32 %f51, [%r17 + 512];
    ld.shared.f32 %f52, [%r17 + 1024];
    ld.shared.f32 %f53, [%r17 + 1536];
    ld.shared.f32 %f54, [%r17 + 2048];
    ld.shared.f32 %f55, [%r17 + 2560];
    ld.shared.f32 %f56, [%r17 + 3072];
    ld.shared.f32 %f57, [%r17 + 3584];
    add.f32 %f50, %f50, %f51; add.f32 %f50, %f50, %f52; add.f32 %f50, %f50, %f53;
    add.f32 %f50, %f50, %f54; add.f32 %f50, %f50, %f55; add.f32 %f50, %f50, %f56;
    add.f32 %f50, %f50, %f57;

    ld.shared.f32 %f58, [%r17 + 4];
    ld.shared.f32 %f59, [%r17 + 516];
    ld.shared.f32 %f60, [%r17 + 1028];
    ld.shared.f32 %f61, [%r17 + 1540];
    ld.shared.f32 %f62, [%r17 + 2052];
    ld.shared.f32 %f63, [%r17 + 2564];
    ld.shared.f32 %f64, [%r17 + 3076];
    ld.shared.f32 %f65, [%r17 + 3588];
    add.f32 %f58, %f58, %f59; add.f32 %f58, %f58, %f60; add.f32 %f58, %f58, %f61;
    add.f32 %f58, %f58, %f62; add.f32 %f58, %f58, %f63; add.f32 %f58, %f58, %f64;
    add.f32 %f58, %f58, %f65;

    ld.shared.f32 %f62, [%r17 + 256];
    ld.shared.f32 %f63, [%r17 + 768];
    ld.shared.f32 %f64, [%r17 + 1280];
    ld.shared.f32 %f65, [%r17 + 1792];
    ld.shared.f32 %f66, [%r17 + 2304];
    ld.shared.f32 %f67, [%r17 + 2816];
    ld.shared.f32 %f68, [%r17 + 3328];
    ld.shared.f32 %f69, [%r17 + 3840];
    add.f32 %f62, %f62, %f63; add.f32 %f62, %f62, %f64; add.f32 %f62, %f62, %f65;
    add.f32 %f62, %f62, %f66; add.f32 %f62, %f62, %f67; add.f32 %f62, %f62, %f68;
    add.f32 %f62, %f62, %f69;

    ld.shared.f32 %f66, [%r17 + 260];
    ld.shared.f32 %f67, [%r17 + 772];
    ld.shared.f32 %f68, [%r17 + 1284];
    ld.shared.f32 %f69, [%r17 + 1796];
    ld.shared.f32 %f70, [%r17 + 2308];
    ld.shared.f32 %f71, [%r17 + 2820];
    ld.shared.f32 %f72, [%r17 + 3332];
    ld.shared.f32 %f73, [%r17 + 3844];
    add.f32 %f66, %f66, %f67; add.f32 %f66, %f66, %f68; add.f32 %f66, %f66, %f69;
    add.f32 %f66, %f66, %f70; add.f32 %f66, %f66, %f71; add.f32 %f66, %f66, %f72;
    add.f32 %f66, %f66, %f73;

    // Write out[row * N + n_tile_base + col]:
    //   d0 -> (gid, col0); d1 -> (gid, col0+1); d2 -> (gid+8, col0); d3 -> (gid+8, col0+1)
    mad.lo.u32 %r106, %r9, %r0, %r4;    // gid * N + n_tile_base
    add.u32    %r106, %r106, %r141;     // + col0
    shl.b32    %r106, %r106, 2;
    cvt.u64.u32 %rd11, %r106;
    add.u64    %rd11, %rd3, %rd11;
    @%p2 st.global.f32 [%rd11 + 0], %f50;
    @%p2 st.global.f32 [%rd11 + 4], %f58;

    mad.lo.u32 %r107, %r140, %r0, %r4;  // (gid+8) * N + n_tile_base
    add.u32    %r107, %r107, %r141;     // + col0
    shl.b32    %r107, %r107, 2;
    cvt.u64.u32 %rd16, %r107;
    add.u64    %rd16, %rd3, %rd16;
    @%p4 st.global.f32 [%rd16 + 0], %f62;
    @%p4 st.global.f32 [%rd16 + 4], %f66;

DONE:
    ret;
}
"#;
