/// Hand-tuned PTX kernel for Q4K x Q8_1 GEMV decode, targeting sm_120 (Blackwell RTX 5090).
///
/// # Implementation: native 36-u32 GGUF format
///
/// This kernel reads Q4K weights in the NATIVE 36-u32 GGUF format (144 bytes/block),
/// giving 14% fewer DRAM reads vs the repacked 42-u32 format (168 bytes/block).
///
/// The engine's `dispatch_kv_on_stream` now uses `submit_v4_kpar_kv_into_slot` for K/V
/// projections (separate per-KV-stream Q8_1 scratch avoids racing with Q on stream_main).
///
/// # Native Q4K block layout (36 u32 = 144 bytes per block of 256 elements)
///
/// ```text
/// bytes  0-1:  d    (f16 super-scale)
/// bytes  2-3:  dmin (f16 super-min)
/// bytes  4-15: scales[12] (6-bit packed: 8 x (sc, m) pairs)
/// bytes 16-143: qs[128] (packed nibbles)
/// ```
///
/// Scale layout in scales[12]:
///   ps0 = u32 from bytes 4-7, ps1 from bytes 8-11, ps2 from bytes 12-15
///
///   For mw=0,1 (shift = (mw&1)*16):
///     sc_lo = (ps0 >> shift) & 63
///     sc_hi = (ps0 >> (shift+8)) & 63
///     m_lo  = (ps1 >> shift) & 63
///     m_hi  = (ps1 >> (shift+8)) & 63
///
///   For mw=2,3 (shift = (mw&1)*16, cross-byte encoding):
///     b0 = (ps0>>shift)&0xFF, b1 = (ps0>>(shift+8))&0xFF
///     b4 = (ps1>>shift)&0xFF, b5 = (ps1>>(shift+8))&0xFF
///     b8 = (ps2>>shift)&0xFF, b9 = (ps2>>(shift+8))&0xFF
///     sc_lo = (b8&0x0F) | ((b0&0xC0)>>2)
///     sc_hi = (b9&0x0F) | ((b1&0xC0)>>2)
///     m_lo  = ((b8>>4)&0x0F) | ((b4&0xC0)>>2)
///     m_hi  = ((b9>>4)&0x0F) | ((b5&0xC0)>>2)
///
///   Scale extraction is BRANCHLESS (selp) -- ITS (Independent Thread Scheduling)
///   on Volta+ does not auto-reconverge warp divergence in hand-written PTX.
///
/// # Q8_1 activation layout (9 u32 = 36 bytes per 32-element block)
///
/// ```text
/// u32[0..8]: 32 packed i8 activations (4 per u32)
/// u32[8]:    block scale d (f16 in low 16 bits)
/// ```
///
/// # Launch geometry
///
/// Grid:  (n_rows, 1, 1)  -- one block per output row
/// Block: (256, 1, 1)     -- 8 warps x 32 threads
/// Smem:  128 bytes       -- 32 f32 slots (8 warps x 4 j-groups)
///
/// # K-parallelism
///
/// 8 warps stride by 8 blocks per iteration (VDR=1).
/// With k_blocks=16 (dim=4096), each warp handles exactly 2 blocks.
///
/// # Kernel parameters
///
/// ```text
/// param_weights:  *const u32  -- native Q4K blocks [n_rows * k_blocks * 36]
/// param_act_qs:   *const u32  -- Q8_1 quantized activations [k/4 packed i8s]
/// param_act_ds:   *const u32  -- Q8_1 block scales [k/32 f16s, low 16 bits of u32]
/// param_out:      *mut f32    -- output [n_rows]
/// param_n_rows:   u32
/// param_k_blocks: u32
/// ```
pub const PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// q4k_gemv_decode_v4_kpar
// Native GGUF Q4K (36 u32/block = 144 bytes) x Q8_1 GEMV decode.
// sm_120 (Blackwell) hand-tuned PTX -- direct codegen, no JIT.
//
// Block: 256 threads = 8 warps. K-parallel: warp w handles blk w, w+8, w+16, ...
// Scale extraction: BRANCHLESS (selp) -- avoids ITS warp divergence.
// shfl.sync.bfly c=0x181F: nvcc's encoding for __shfl_xor_sync(..., width=8)
// (verified via nvcc --ptx on sm_120). The previous c=0x71F silently collapsed
// mini-warp 3 (lanes 24-31) onto lane 24: the reduction returned 8x lane-24's
// partial and dropped sub_l 1-7 -- j-group 3 contributed garbage whenever its
// qs bytes were non-uniform. Found by the bench-decode-ops one-hot sweep
// (2026-10-28); uniform-data tests cannot catch it.
.visible .entry q4k_gemv_sm120_dp4a_kpar8(
    .param .u64 param_weights,
    .param .u64 param_act_qs,
    .param .u64 param_act_ds,
    .param .u64 param_out,
    .param .u32 param_n_rows,
    .param .u32 param_k_blocks
)
{
    .reg .u32  %r<57>;
    .reg .u64  %rd<11>;
    .reg .f32  %f<20>;
    .reg .pred %p<11>;
    .reg .b16  %h<4>;

    // 32 f32 slots: smem[warp_id * 4 + mw] holds the mini-warp partial sum.
    .shared .align 4 .b8 smem_buf[128];

    // -- Load kernel parameters -----------------------------------------------
    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_act_qs];
    ld.param.u64 %rd2, [param_act_ds];
    ld.param.u64 %rd3, [param_out];
    ld.param.u32 %r0,  [param_n_rows];
    ld.param.u32 %r1,  [param_k_blocks];

    // -- Thread index decomposition --------------------------------------------
    mov.u32 %r2, %ctaid.x;       // n_out = blockIdx.x
    setp.ge.u32 %p0, %r2, %r0;   // guard: n_out >= n_rows -> exit
    @%p0 bra DONE;

    mov.u32 %r3, %tid.x;          // flat thread index (0..255)
    and.b32 %r4, %r3, 31;         // lane = tid & 31
    shr.u32 %r5, %r3, 5;          // warp_id = tid >> 5  (0..7)
    shr.u32 %r6, %r4, 3;          // mw = lane >> 3  (0..3, j-group)
    and.b32 %r7, %r4, 7;          // sub_l = lane & 7

    // shift = (mw & 1) << 4  [0 for mw=0,2; 16 for mw=1,3]
    and.b32 %r8, %r6, 1;
    shl.b32 %r8, %r8, 4;

    // is_upper = mw >> 1  [1 for mw=2,3; 0 for mw=0,1]
    shr.u32 %r9, %r6, 1;
    setp.ne.u32 %p1, %r9, 0;      // %p1 = is_upper

    // sub_l == 0 predicate (for smem write in reduction)
    setp.eq.u32 %p2, %r7, 0;      // %p2 = (sub_l == 0)

    // acc = 0.0
    mov.f32 %f0, 0f00000000;

    // K-loop: warp_id strides by 8 (WARPS_PER_BLOCK_KPAR = 8)
    mov.u32 %r12, %r5;            // blk = warp_id

    // -- Main K-loop -----------------------------------------------------------
KLOOP:
    setp.ge.u32 %p3, %r12, %r1;   // blk >= k_blocks -> exit loop
    @%p3 bra REDUCE;

    // base_ptr = weights + (n_out * k_blocks + blk) * 144 bytes (36 u32 * 4)
    mad.lo.u32 %r14, %r2, %r1, %r12;  // n_out * k_blocks + blk
    mul.lo.u32 %r15, %r14, 144;        // * 144 bytes/block
    cvt.u64.u32 %rd4, %r15;
    add.u64 %rd4, %rd0, %rd4;          // %rd4 = base_ptr

    // --- Prefetch next block into L2 ---
    // next_blk = blk + 8 (warp stride = WARPS_PER_BLOCK_KPAR = 8).
    // Only prefetch if next_blk < k_blocks (i.e. a second iteration exists).
    // For k_blocks=16 with 8 warps, this fires for the 1st of 2 iterations per warp.
    // 8 blocks * 144 bytes/block = 1152 bytes stride.
    add.u32 %r56, %r12, 8;
    setp.lt.u32 %p10, %r56, %r1;
    add.u64 %rd10, %rd4, 1152;
    @%p10 prefetch.global.L2 [%rd10];

    // -- Load d and dmin as f16 (bytes 0-1, 2-3) -- .nc: weights are scan-once
    ld.global.nc.b16 %h0, [%rd4];         // d as f16 (bytes 0-1)
    ld.global.nc.b16 %h1, [%rd4 + 2];    // dmin as f16 (bytes 2-3)
    cvt.f32.f16 %f1, %h0;              // d as f32 (exact, no rounding modifier)
    cvt.f32.f16 %f2, %h1;              // dmin as f32

    // -- Load scale words (bytes 4-15) -- .nc: weights are scan-once, free L1 for activations
    ld.global.nc.b32 %r16, [%rd4 + 4];   // ps0 = scales[0..3]
    ld.global.nc.b32 %r17, [%rd4 + 8];   // ps1 = scales[4..7]
    ld.global.nc.b32 %r18, [%rd4 + 12];  // ps2 = scales[8..11]

    // -- Branchless scale extraction (no branch divergence, avoids ITS issues) -
    // Compute shift+8 once for both lower and upper formulas.
    add.u32 %r19, %r8, 8;              // shift + 8

    // Lower formula (mw=0,1): 6-bit direct extraction
    shr.u32 %r20, %r16, %r8;           // ps0 >> shift
    and.b32 %r20, %r20, 63;            // lo_sc_lo = & 0x3F

    shr.u32 %r21, %r16, %r19;          // ps0 >> (shift+8)
    and.b32 %r21, %r21, 63;            // lo_sc_hi

    shr.u32 %r22, %r17, %r8;           // ps1 >> shift
    and.b32 %r22, %r22, 63;            // lo_m_lo

    shr.u32 %r23, %r17, %r19;          // ps1 >> (shift+8)
    and.b32 %r23, %r23, 63;            // lo_m_hi

    // Upper formula (mw=2,3): cross-byte 6-bit extraction
    // Extract bytes from ps0,ps1,ps2 at positions shift and shift+8
    shr.u32 %r40, %r16, %r8;  and.b32 %r40, %r40, 255;   // b0
    shr.u32 %r41, %r16, %r19; and.b32 %r41, %r41, 255;   // b1
    shr.u32 %r42, %r17, %r8;  and.b32 %r42, %r42, 255;   // b4
    shr.u32 %r43, %r17, %r19; and.b32 %r43, %r43, 255;   // b5
    shr.u32 %r44, %r18, %r8;  and.b32 %r44, %r44, 255;   // b8
    shr.u32 %r45, %r18, %r19; and.b32 %r45, %r45, 255;   // b9

    // up_sc_lo = (b8 & 0x0F) | ((b0 & 0xC0) >> 2)
    and.b32 %r47, %r44, 15;
    and.b32 %r46, %r40, 192; shr.u32 %r46, %r46, 2;
    or.b32  %r47, %r47, %r46;

    // up_sc_hi = (b9 & 0x0F) | ((b1 & 0xC0) >> 2)
    and.b32 %r48, %r45, 15;
    and.b32 %r46, %r41, 192; shr.u32 %r46, %r46, 2;
    or.b32  %r48, %r48, %r46;

    // up_m_lo = ((b8 >> 4) & 0x0F) | ((b4 & 0xC0) >> 2)
    shr.u32 %r49, %r44, 4; and.b32 %r49, %r49, 15;
    and.b32 %r46, %r42, 192; shr.u32 %r46, %r46, 2;
    or.b32  %r49, %r49, %r46;

    // up_m_hi = ((b9 >> 4) & 0x0F) | ((b5 & 0xC0) >> 2)
    shr.u32 %r50, %r45, 4; and.b32 %r50, %r50, 15;
    and.b32 %r46, %r43, 192; shr.u32 %r46, %r46, 2;
    or.b32  %r50, %r50, %r46;

    // selp: if is_upper (%p1), use upper formula; else use lower
    selp.u32 %r20, %r47, %r20, %p1;    // sc_lo
    selp.u32 %r21, %r48, %r21, %p1;    // sc_hi
    selp.u32 %r22, %r49, %r22, %p1;    // m_lo
    selp.u32 %r23, %r50, %r23, %p1;    // m_hi

    // Convert 6-bit integer scales to f32
    cvt.rn.f32.u32 %f3, %r20;         // sc_lo as f32
    cvt.rn.f32.u32 %f4, %r21;         // sc_hi as f32
    cvt.rn.f32.u32 %f5, %r22;         // m_lo as f32
    cvt.rn.f32.u32 %f6, %r23;         // m_hi as f32

    // -- Load qs word (lane-specific): u32[4 + mw*8 + sub_l] ------------------
    // Byte offset = (4 + mw*8 + sub_l) * 4  (skips 16-byte header)
    mul.lo.u32 %r24, %r6, 8;           // mw * 8
    add.u32    %r24, %r24, 4;           // + 4 (skip header: u32[0..3])
    add.u32    %r24, %r24, %r7;         // + sub_l
    shl.b32    %r24, %r24, 2;           // * 4 bytes
    cvt.u64.u32 %rd5, %r24;
    add.u64    %rd5, %rd4, %rd5;
    ld.global.nc.b32 %r25, [%rd5];     // qs u32 (.nc: weight, scan-once)

    // -- Load Q8_1 activation data --------------------------------------------
    // q8_blk_lo = blk * 8 + mw * 2
    mul.lo.u32 %r26, %r12, 8;          // blk * 8
    mul.lo.u32 %r27, %r6, 2;           // mw * 2
    add.u32    %r26, %r26, %r27;        // q8_blk_lo

    // u_lo = act_qs[q8_blk_lo * 8 + sub_l]
    mul.lo.u32 %r27, %r26, 8;          // q8_blk_lo * 8
    add.u32    %r27, %r27, %r7;         // + sub_l
    shl.b32    %r28, %r27, 2;           // * 4 bytes
    cvt.u64.u32 %rd6, %r28;
    add.u64    %rd6, %rd1, %rd6;
    ld.global.b32 %r28, [%rd6];        // u_lo

    // u_hi = act_qs[q8_blk_lo*8 + sub_l + 8]
    add.u32    %r29, %r27, 8;
    shl.b32    %r30, %r29, 2;
    cvt.u64.u32 %rd7, %r30;
    add.u64    %rd7, %rd1, %rd7;
    ld.global.b32 %r29, [%rd7];        // u_hi

    // d8_lo = act_ds[q8_blk_lo]  (f16 in low 16 bits)
    shl.b32    %r30, %r26, 2;           // q8_blk_lo * 4 bytes
    cvt.u64.u32 %rd8, %r30;
    add.u64    %rd8, %rd2, %rd8;
    ld.global.b16 %h2, [%rd8];         // f16 d8_lo
    cvt.f32.f16 %f7, %h2;

    // d8_hi = act_ds[q8_blk_lo + 1]
    add.u64    %rd9, %rd8, 4;
    ld.global.b16 %h3, [%rd9];         // f16 d8_hi
    cvt.f32.f16 %f8, %h3;

    // -- dp4a: Q4K nibbles x Q8_1 bytes ----------------------------------------
    and.b32 %r32, %r25, 0x0F0F0F0F;   // v_lo = lo nibbles (0-15 as s8)
    shr.u32 %r33, %r25, 4;
    and.b32 %r33, %r33, 0x0F0F0F0F;   // v_hi = hi nibbles

    dp4a.s32.s32 %r34, %r32, %r28, 0; // dot_lo
    dp4a.s32.s32 %r35, %r33, %r29, 0; // dot_hi
    dp4a.s32.s32 %r36, 0x01010101, %r28, 0; // dot_min_lo = sum(u_lo)
    dp4a.s32.s32 %r37, 0x01010101, %r29, 0; // dot_min_hi = sum(u_hi)

    // -- Accumulate ------------------------------------------------------------
    cvt.rn.f32.s32 %f9,  %r34;
    cvt.rn.f32.s32 %f10, %r35;
    cvt.rn.f32.s32 %f11, %r36;
    cvt.rn.f32.s32 %f12, %r37;

    // term_lo = d*sc_lo*dot_lo - dmin*m_lo*dot_min_lo
    mul.f32 %f13, %f1, %f3; mul.f32 %f13, %f13, %f9;
    mul.f32 %f14, %f2, %f5; mul.f32 %f14, %f14, %f11;
    sub.f32 %f13, %f13, %f14; mul.f32 %f13, %f13, %f7;
    add.f32 %f0, %f0, %f13;

    // term_hi = d*sc_hi*dot_hi - dmin*m_hi*dot_min_hi
    mul.f32 %f15, %f1, %f4; mul.f32 %f15, %f15, %f10;
    mul.f32 %f16, %f2, %f6; mul.f32 %f16, %f16, %f12;
    sub.f32 %f15, %f15, %f16; mul.f32 %f15, %f15, %f8;
    add.f32 %f0, %f0, %f15;

    // -- Advance blk and loop --------------------------------------------------
    add.u32 %r12, %r12, 8;
    bra KLOOP;

    // -- Reduction -------------------------------------------------------------
REDUCE:
    // 3-step XOR butterfly within 8-lane mini-warp.
    // c=0x181F = nvcc width-8 XOR-bfly encoding (see file header note).
    // member_mask=0xFFFFFFFF: all 32 lanes execute (required by shfl.sync).
    // Note: redux.sync.add.f32 does NOT exist in PTX ISA (only integer types).
    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 4, 0x181F, 0xFFFFFFFF;
    mov.b32 %f17, %r51; add.f32 %f0, %f0, %f17;

    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 2, 0x181F, 0xFFFFFFFF;
    mov.b32 %f17, %r51; add.f32 %f0, %f0, %f17;

    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 1, 0x181F, 0xFFFFFFFF;
    mov.b32 %f17, %r51; add.f32 %f0, %f0, %f17;
    // After 3 steps: ALL lanes in each mini-warp hold the mini-warp sum.

    // Step 2: sub_l==0 writes to smem[warp_id*4 + mw]
    mul.lo.u32 %r52, %r5, 4;
    add.u32    %r52, %r52, %r6;
    shl.b32    %r53, %r52, 2;
    mov.u32    %r54, smem_buf;
    add.u32    %r54, %r54, %r53;
    @%p2 st.shared.f32 [%r54], %f0;

    bar.sync 0;

    // Step 3: warp 0 lanes 0-3 cross-warp reduce per j-group
    setp.eq.u32 %p6, %r5, 0;
    setp.lt.u32 %p7, %r4, 4;
    and.pred    %p8, %p6, %p7;
    mul.lo.u32 %r55, %r4, 4;
    mov.u32    %r54, smem_buf;
    add.u32    %r55, %r54, %r55;    // smem_buf + j*4

    mov.f32    %f18, 0f00000000;
    @%p8 ld.shared.f32 %f19, [%r55 +   0]; @%p8 add.f32 %f18, %f18, %f19; // w=0
    @%p8 ld.shared.f32 %f19, [%r55 +  16]; @%p8 add.f32 %f18, %f18, %f19; // w=1
    @%p8 ld.shared.f32 %f19, [%r55 +  32]; @%p8 add.f32 %f18, %f18, %f19; // w=2
    @%p8 ld.shared.f32 %f19, [%r55 +  48]; @%p8 add.f32 %f18, %f18, %f19; // w=3
    @%p8 ld.shared.f32 %f19, [%r55 +  64]; @%p8 add.f32 %f18, %f18, %f19; // w=4
    @%p8 ld.shared.f32 %f19, [%r55 +  80]; @%p8 add.f32 %f18, %f18, %f19; // w=5
    @%p8 ld.shared.f32 %f19, [%r55 +  96]; @%p8 add.f32 %f18, %f18, %f19; // w=6
    @%p8 ld.shared.f32 %f19, [%r55 + 112]; @%p8 add.f32 %f18, %f18, %f19; // w=7

    @%p8 st.shared.f32 [%r55], %f18;

    bar.sync 0;

    // Step 4: thread 0 sums smem[0..3] and writes output
    setp.eq.u32 %p9, %r3, 0;
    mov.u32    %r53, smem_buf;
    mov.f32    %f18, 0f00000000;
    @%p9 ld.shared.f32 %f18, [%r53 +  0];
    @%p9 ld.shared.f32 %f19, [%r53 +  4]; @%p9 add.f32 %f18, %f18, %f19;
    @%p9 ld.shared.f32 %f19, [%r53 +  8]; @%p9 add.f32 %f18, %f18, %f19;
    @%p9 ld.shared.f32 %f19, [%r53 + 12]; @%p9 add.f32 %f18, %f18, %f19;

    shl.b32    %r50, %r2, 2;
    cvt.u64.u32 %rd4, %r50;
    add.u64    %rd4, %rd3, %rd4;
    @%p9 st.global.f32 [%rd4], %f18;

DONE:
    ret;
}

// q4k_gemv_sm120_dp4a_kpar8_dual_block: two-block-per-iteration unroll of
// kpar8.
//
// kpar8 processes ONE 144-byte block per warp-iteration. With k_blocks=16
// (h=4096) each warp runs only 2 iterations -- the loop is latency-bound:
// 7 dependent weight loads per iteration, little to overlap.
//
// This variant processes blocks blkA and blkB=blkA+8 per iteration: both
// blocks' weight loads issue back-to-back (2x memory-level parallelism per
// warp), then compute A, load B activations, compute B. For k_blocks=16 the
// loop collapses to a single iteration with 14 loads in flight.
//
// Accumulation order per warp is IDENTICAL to kpar8 (blkA, blkB, blkA+16,
// ...), so results are bitwise-equal by construction.
//
// When blkB >= k_blocks (ragged tail, e.g. down_proj k_blocks=56), baseB and
// B's activation indices clamp to block A's (safe addresses -- Blackwell
// traps on OOB even under predication) and B's accumulate is predicated off.
.visible .entry q4k_gemv_sm120_dp4a_kpar8_dual_block(
    .param .u64 param_weights,
    .param .u64 param_act_qs,
    .param .u64 param_act_ds,
    .param .u64 param_out,
    .param .u32 param_n_rows,
    .param .u32 param_k_blocks
)
{
    .reg .u32  %r<80>;
    .reg .u64  %rd<16>;
    .reg .f32  %f<40>;
    .reg .pred %p<16>;
    .reg .b16  %h<8>;

    // 32 f32 slots: smem[warp_id * 4 + mw] holds the mini-warp partial sum.
    .shared .align 4 .b8 smem_buf[128];

    // -- Load kernel parameters -----------------------------------------------
    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_act_qs];
    ld.param.u64 %rd2, [param_act_ds];
    ld.param.u64 %rd3, [param_out];
    ld.param.u32 %r0,  [param_n_rows];
    ld.param.u32 %r1,  [param_k_blocks];

    // -- Thread index decomposition --------------------------------------------
    mov.u32 %r2, %ctaid.x;       // n_out = blockIdx.x
    setp.ge.u32 %p0, %r2, %r0;   // guard: n_out >= n_rows -> exit
    @%p0 bra DONE2;

    mov.u32 %r3, %tid.x;          // flat thread index (0..255)
    and.b32 %r4, %r3, 31;         // lane = tid & 31
    shr.u32 %r5, %r3, 5;          // warp_id = tid >> 5  (0..7)
    shr.u32 %r6, %r4, 3;          // mw = lane >> 3  (0..3, j-group)
    and.b32 %r7, %r4, 7;          // sub_l = lane & 7

    // shift = (mw & 1) << 4  [0 for mw=0,2; 16 for mw=1,3]
    and.b32 %r8, %r6, 1;
    shl.b32 %r8, %r8, 4;
    add.u32 %r19, %r8, 8;         // shift + 8

    // is_upper = mw >> 1  [1 for mw=2,3; 0 for mw=0,1]
    shr.u32 %r9, %r6, 1;
    setp.ne.u32 %p1, %r9, 0;      // %p1 = is_upper

    // sub_l == 0 predicate (for smem write in reduction)
    setp.eq.u32 %p2, %r7, 0;      // %p2 = (sub_l == 0)

    // acc = 0.0
    mov.f32 %f0, 0f00000000;

    // Per-lane qs word byte offset within a block: (4 + mw*8 + sub_l) * 4.
    mul.lo.u32 %r24, %r6, 8;
    add.u32    %r24, %r24, 4;
    add.u32    %r24, %r24, %r7;
    shl.b32    %r24, %r24, 2;

    // K-loop: warp strides by 16 (two 144-byte blocks per iteration).
    mov.u32 %r12, %r5;            // blkA = warp_id

KLOOP2:
    setp.ge.u32 %p3, %r12, %r1;   // blkA >= k_blocks -> exit loop
    @%p3 bra REDUCE2;

    // -- Addresses --------------------------------------------------------------
    // baseA = weights + (n_out * k_blocks + blkA) * 144
    mad.lo.u32 %r14, %r2, %r1, %r12;
    mul.lo.u32 %r15, %r14, 144;
    cvt.u64.u32 %rd4, %r15;
    add.u64 %rd4, %rd0, %rd4;

    // blkB = blkA + 8; clamp baseB to baseA when invalid (safe load address).
    add.u32 %r60, %r12, 8;
    setp.lt.u32 %p10, %r60, %r1;  // %p10 = B valid
    mad.lo.u32 %r61, %r2, %r1, %r60;
    mul.lo.u32 %r62, %r61, 144;
    cvt.u64.u32 %rd5, %r62;
    add.u64 %rd5, %rd0, %rd5;
    selp.u64 %rd5, %rd5, %rd4, %p10;

    // -- Prefetch next pair into L2 (16 blocks ahead = 2304 bytes) -------------
    add.u32 %r63, %r12, 16;
    setp.lt.u32 %p11, %r63, %r1;
    add.u64 %rd10, %rd4, 2304;
    @%p11 prefetch.global.L2 [%rd10];

    // ---- Load A weights (d, dmin, 3 scale words, qs) --------------------------
    ld.global.nc.b16 %h0, [%rd4];
    ld.global.nc.b16 %h1, [%rd4 + 2];
    cvt.f32.f16 %f1, %h0;              // d
    cvt.f32.f16 %f2, %h1;              // dmin
    ld.global.nc.b32 %r16, [%rd4 + 4];   // ps0
    ld.global.nc.b32 %r17, [%rd4 + 8];   // ps1
    ld.global.nc.b32 %r18, [%rd4 + 12];  // ps2
    cvt.u64.u32 %rd13, %r24;             // qs_off (per-lane byte offset)
    add.u64 %rd6, %rd4, %rd13;
    ld.global.nc.b32 %r25, [%rd6];       // qs word A

    // ---- Load B weights (clamped-safe when invalid) ---------------------------
    ld.global.nc.b16 %h4, [%rd5];
    ld.global.nc.b16 %h5, [%rd5 + 2];
    cvt.f32.f16 %f20, %h4;             // d_B
    cvt.f32.f16 %f21, %h5;             // dmin_B
    ld.global.nc.b32 %r64, [%rd5 + 4];
    ld.global.nc.b32 %r65, [%rd5 + 8];
    ld.global.nc.b32 %r66, [%rd5 + 12];
    add.u64 %rd7, %rd5, %rd13;
    ld.global.nc.b32 %r67, [%rd7];       // qs word B

    // =============== COMPUTE BLOCK A ==========================================
    // Branchless scale extraction (identical to kpar8).
    shr.u32 %r20, %r16, %r8;           and.b32 %r20, %r20, 63;
    shr.u32 %r21, %r16, %r19;          and.b32 %r21, %r21, 63;
    shr.u32 %r22, %r17, %r8;           and.b32 %r22, %r22, 63;
    shr.u32 %r23, %r17, %r19;          and.b32 %r23, %r23, 63;

    shr.u32 %r40, %r16, %r8;  and.b32 %r40, %r40, 255;   // b0
    shr.u32 %r41, %r16, %r19; and.b32 %r41, %r41, 255;   // b1
    shr.u32 %r42, %r17, %r8;  and.b32 %r42, %r42, 255;   // b4
    shr.u32 %r43, %r17, %r19; and.b32 %r43, %r43, 255;   // b5
    shr.u32 %r44, %r18, %r8;  and.b32 %r44, %r44, 255;   // b8
    shr.u32 %r45, %r18, %r19; and.b32 %r45, %r45, 255;   // b9

    and.b32 %r47, %r44, 15;
    and.b32 %r46, %r40, 192; shr.u32 %r46, %r46, 2;
    or.b32  %r47, %r47, %r46;
    and.b32 %r48, %r45, 15;
    and.b32 %r46, %r41, 192; shr.u32 %r46, %r46, 2;
    or.b32  %r48, %r48, %r46;
    shr.u32 %r49, %r44, 4; and.b32 %r49, %r49, 15;
    and.b32 %r46, %r42, 192; shr.u32 %r46, %r46, 2;
    or.b32  %r49, %r49, %r46;
    shr.u32 %r50, %r45, 4; and.b32 %r50, %r50, 15;
    and.b32 %r46, %r43, 192; shr.u32 %r46, %r46, 2;
    or.b32  %r50, %r50, %r46;

    selp.u32 %r20, %r47, %r20, %p1;    // sc_lo
    selp.u32 %r21, %r48, %r21, %p1;    // sc_hi
    selp.u32 %r22, %r49, %r22, %p1;    // m_lo
    selp.u32 %r23, %r50, %r23, %p1;    // m_hi

    cvt.rn.f32.u32 %f3, %r20;
    cvt.rn.f32.u32 %f4, %r21;
    cvt.rn.f32.u32 %f5, %r22;
    cvt.rn.f32.u32 %f6, %r23;

    // -- A activations: q8_blk_lo = blkA * 8 + mw * 2 ---------------------------
    mul.lo.u32 %r26, %r12, 8;
    mul.lo.u32 %r27, %r6, 2;
    add.u32    %r26, %r26, %r27;

    mul.lo.u32 %r27, %r26, 8;
    add.u32    %r27, %r27, %r7;
    shl.b32    %r28, %r27, 2;
    cvt.u64.u32 %rd8, %r28;
    add.u64 %rd8, %rd1, %rd8;
    ld.global.b32 %r28, [%rd8];        // u_lo A

    add.u32 %r29, %r27, 8;
    shl.b32 %r30, %r29, 2;
    cvt.u64.u32 %rd9, %r30;
    add.u64 %rd9, %rd1, %rd9;
    ld.global.b32 %r29, [%rd9];        // u_hi A

    shl.b32 %r30, %r26, 2;
    cvt.u64.u32 %rd11, %r30;
    add.u64 %rd11, %rd2, %rd11;
    ld.global.b16 %h2, [%rd11];
    cvt.f32.f16 %f7, %h2;              // d8_lo A
    add.u64 %rd12, %rd11, 4;
    ld.global.b16 %h3, [%rd12];
    cvt.f32.f16 %f8, %h3;              // d8_hi A

    // -- A dp4a -----------------------------------------------------------------
    and.b32 %r32, %r25, 0x0F0F0F0F;
    shr.u32 %r33, %r25, 4;
    and.b32 %r33, %r33, 0x0F0F0F0F;

    dp4a.s32.s32 %r34, %r32, %r28, 0;
    dp4a.s32.s32 %r35, %r33, %r29, 0;
    dp4a.s32.s32 %r36, 0x01010101, %r28, 0;
    dp4a.s32.s32 %r37, 0x01010101, %r29, 0;

    cvt.rn.f32.s32 %f9,  %r34;
    cvt.rn.f32.s32 %f10, %r35;
    cvt.rn.f32.s32 %f11, %r36;
    cvt.rn.f32.s32 %f12, %r37;

    mul.f32 %f13, %f1, %f3; mul.f32 %f13, %f13, %f9;
    mul.f32 %f14, %f2, %f5; mul.f32 %f14, %f14, %f11;
    sub.f32 %f13, %f13, %f14; mul.f32 %f13, %f13, %f7;
    add.f32 %f0, %f0, %f13;

    mul.f32 %f15, %f1, %f4; mul.f32 %f15, %f15, %f10;
    mul.f32 %f16, %f2, %f6; mul.f32 %f16, %f16, %f12;
    sub.f32 %f15, %f15, %f16; mul.f32 %f15, %f15, %f8;
    add.f32 %f0, %f0, %f15;

    // =============== COMPUTE BLOCK B ==========================================
    // Scale extraction from B's scale words (identical shape to A).
    shr.u32 %r20, %r64, %r8;           and.b32 %r20, %r20, 63;
    shr.u32 %r21, %r64, %r19;          and.b32 %r21, %r21, 63;
    shr.u32 %r22, %r65, %r8;           and.b32 %r22, %r22, 63;
    shr.u32 %r23, %r65, %r19;          and.b32 %r23, %r23, 63;

    shr.u32 %r40, %r64, %r8;  and.b32 %r40, %r40, 255;
    shr.u32 %r41, %r64, %r19; and.b32 %r41, %r41, 255;
    shr.u32 %r42, %r65, %r8;  and.b32 %r42, %r42, 255;
    shr.u32 %r43, %r65, %r19; and.b32 %r43, %r43, 255;
    shr.u32 %r44, %r66, %r8;  and.b32 %r44, %r44, 255;
    shr.u32 %r45, %r66, %r19; and.b32 %r45, %r45, 255;

    and.b32 %r47, %r44, 15;
    and.b32 %r46, %r40, 192; shr.u32 %r46, %r46, 2;
    or.b32  %r47, %r47, %r46;
    and.b32 %r48, %r45, 15;
    and.b32 %r46, %r41, 192; shr.u32 %r46, %r46, 2;
    or.b32  %r48, %r48, %r46;
    shr.u32 %r49, %r44, 4; and.b32 %r49, %r49, 15;
    and.b32 %r46, %r42, 192; shr.u32 %r46, %r46, 2;
    or.b32  %r49, %r49, %r46;
    shr.u32 %r50, %r45, 4; and.b32 %r50, %r50, 15;
    and.b32 %r46, %r43, 192; shr.u32 %r46, %r46, 2;
    or.b32  %r50, %r50, %r46;

    selp.u32 %r20, %r47, %r20, %p1;
    selp.u32 %r21, %r48, %r21, %p1;
    selp.u32 %r22, %r49, %r22, %p1;
    selp.u32 %r23, %r50, %r23, %p1;

    cvt.rn.f32.u32 %f3, %r20;
    cvt.rn.f32.u32 %f4, %r21;
    cvt.rn.f32.u32 %f5, %r22;
    cvt.rn.f32.u32 %f6, %r23;

    // -- B activations: q8_blk_lo = blkB * 8 + mw * 2 (clamped to A when invalid)
    mul.lo.u32 %r68, %r60, 8;
    selp.u32 %r68, %r68, %r26, %p10;
    mul.lo.u32 %r27, %r6, 2;
    add.u32    %r68, %r68, %r27;

    mul.lo.u32 %r27, %r68, 8;
    add.u32    %r27, %r27, %r7;
    shl.b32    %r28, %r27, 2;
    cvt.u64.u32 %rd8, %r28;
    add.u64 %rd8, %rd1, %rd8;
    ld.global.b32 %r28, [%rd8];        // u_lo B

    add.u32 %r29, %r27, 8;
    shl.b32 %r30, %r29, 2;
    cvt.u64.u32 %rd9, %r30;
    add.u64 %rd9, %rd1, %rd9;
    ld.global.b32 %r29, [%rd9];        // u_hi B

    shl.b32 %r30, %r68, 2;
    cvt.u64.u32 %rd11, %r30;
    add.u64 %rd11, %rd2, %rd11;
    ld.global.b16 %h2, [%rd11];
    cvt.f32.f16 %f7, %h2;              // d8_lo B
    add.u64 %rd12, %rd11, 4;
    ld.global.b16 %h3, [%rd12];
    cvt.f32.f16 %f8, %h3;              // d8_hi B

    // -- B dp4a (accumulate guarded: B may be the clamped duplicate of A) --------
    and.b32 %r32, %r67, 0x0F0F0F0F;
    shr.u32 %r33, %r67, 4;
    and.b32 %r33, %r33, 0x0F0F0F0F;

    dp4a.s32.s32 %r34, %r32, %r28, 0;
    dp4a.s32.s32 %r35, %r33, %r29, 0;
    dp4a.s32.s32 %r36, 0x01010101, %r28, 0;
    dp4a.s32.s32 %r37, 0x01010101, %r29, 0;

    cvt.rn.f32.s32 %f9,  %r34;
    cvt.rn.f32.s32 %f10, %r35;
    cvt.rn.f32.s32 %f11, %r36;
    cvt.rn.f32.s32 %f12, %r37;

    mul.f32 %f13, %f20, %f3; mul.f32 %f13, %f13, %f9;
    mul.f32 %f14, %f21, %f5; mul.f32 %f14, %f14, %f11;
    sub.f32 %f13, %f13, %f14; mul.f32 %f13, %f13, %f7;
    @%p10 add.f32 %f0, %f0, %f13;

    mul.f32 %f15, %f20, %f4; mul.f32 %f15, %f15, %f10;
    mul.f32 %f16, %f21, %f6; mul.f32 %f16, %f16, %f12;
    sub.f32 %f15, %f15, %f16; mul.f32 %f15, %f15, %f8;
    @%p10 add.f32 %f0, %f0, %f15;

    // -- Advance: next pair ------------------------------------------------------
    add.u32 %r12, %r12, 16;
    bra KLOOP2;

    // -- Reduction (identical to kpar8) ------------------------------------------
REDUCE2:
    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 4, 0x181F, 0xFFFFFFFF;
    mov.b32 %f17, %r51; add.f32 %f0, %f0, %f17;

    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 2, 0x181F, 0xFFFFFFFF;
    mov.b32 %f17, %r51; add.f32 %f0, %f0, %f17;

    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 1, 0x181F, 0xFFFFFFFF;
    mov.b32 %f17, %r51; add.f32 %f0, %f0, %f17;

    mul.lo.u32 %r52, %r5, 4;
    add.u32    %r52, %r52, %r6;
    shl.b32    %r53, %r52, 2;
    mov.u32    %r54, smem_buf;
    add.u32    %r54, %r54, %r53;
    @%p2 st.shared.f32 [%r54], %f0;

    bar.sync 0;

    setp.eq.u32 %p6, %r5, 0;
    setp.lt.u32 %p7, %r4, 4;
    and.pred    %p8, %p6, %p7;
    mul.lo.u32 %r55, %r4, 4;
    mov.u32    %r54, smem_buf;
    add.u32    %r55, %r54, %r55;

    mov.f32 %f18, 0f00000000;
    @%p8 ld.shared.f32 %f19, [%r55 +   0]; @%p8 add.f32 %f18, %f18, %f19;
    @%p8 ld.shared.f32 %f19, [%r55 +  16]; @%p8 add.f32 %f18, %f18, %f19;
    @%p8 ld.shared.f32 %f19, [%r55 +  32]; @%p8 add.f32 %f18, %f18, %f19;
    @%p8 ld.shared.f32 %f19, [%r55 +  48]; @%p8 add.f32 %f18, %f18, %f19;
    @%p8 ld.shared.f32 %f19, [%r55 +  64]; @%p8 add.f32 %f18, %f18, %f19;
    @%p8 ld.shared.f32 %f19, [%r55 +  80]; @%p8 add.f32 %f18, %f18, %f19;
    @%p8 ld.shared.f32 %f19, [%r55 +  96]; @%p8 add.f32 %f18, %f18, %f19;
    @%p8 ld.shared.f32 %f19, [%r55 + 112]; @%p8 add.f32 %f18, %f18, %f19;

    @%p8 st.shared.f32 [%r55], %f18;

    bar.sync 0;

    setp.eq.u32 %p9, %r3, 0;
    mov.u32    %r53, smem_buf;
    mov.f32 %f18, 0f00000000;
    @%p9 ld.shared.f32 %f18, [%r53 +  0];
    @%p9 ld.shared.f32 %f19, [%r53 +  4]; @%p9 add.f32 %f18, %f18, %f19;
    @%p9 ld.shared.f32 %f19, [%r53 +  8]; @%p9 add.f32 %f18, %f18, %f19;
    @%p9 ld.shared.f32 %f19, [%r53 + 12]; @%p9 add.f32 %f18, %f18, %f19;

    shl.b32    %r50, %r2, 2;
    cvt.u64.u32 %rd4, %r50;
    add.u64 %rd4, %rd3, %rd4;
    @%p9 st.global.f32 [%rd4], %f18;

DONE2:
    ret;
}
"#;

/// Kernel entry name -- used by `cuModuleGetFunction`.
pub const ENTRY: &str = "q4k_gemv_sm120_dp4a_kpar8";

/// Two-block-per-iteration unroll of [`ENTRY`] (same ABI and launch config).
/// Selected on sm_120+ when present; bitwise-equal output by construction.
pub const ENTRY_DUAL_BLOCK: &str = "q4k_gemv_sm120_dp4a_kpar8_dual_block";

/// Minimum compute capability required (sm_120 = Blackwell).
pub const MIN_CC: u32 = 1200;

/// Block format: native 36-u32 GGUF (144 bytes/block).
/// Used by engine to choose upload path and dispatch_kv_on_stream variant.
pub const BLOCK_STRIDE_BYTES: u32 = 144;
