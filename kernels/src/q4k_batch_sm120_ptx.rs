/// Hand-tuned PTX batch kernel for Q4K x Q8_1 GEMV decode, targeting sm_120 (Blackwell).
///
/// # Purpose
///
/// The oxide sm_80 PTX kernels that use `DynamicSharedArray` (`q4k_gemv_dp4a_base_batch`,
/// `q4k_gemm_colocated`) crash with CUDA error 719 on RTX 5090 (sm_120 Blackwell) due
/// to JIT recompilation incompatibility of dynamic shared memory. This kernel avoids the
/// crash by using exclusively STATIC shared memory.
///
/// # Design (v2 -- true weight amortization)
///
/// One block per output neuron; each block processes ALL M=2 activation rows, reading
/// each weight block ONCE from DRAM and computing M dot products per block.
///
///   - `n_out = blockIdx.x`  -- output neuron (0..n_rows)
///   - blockIdx.y is always 0 (grid is (n_rows, 1, 1))
///
/// Grid:  (n_rows, 1, 1)
/// Block: (256, 1, 1) = 8 warps
/// Smem:  256 bytes -- 2 x 32 f32 slots (row0 at [0..127], row1 at [128..255])
///
/// Weight reads: ONCE per (n_out, k_block). Each weight block is loaded into
/// registers and reused for both activation rows -- halving DRAM weight bandwidth
/// compared to grid (n_rows, m_rows, 1).
///
/// Activation layout (row-major, M=2 rows):
///   act_qs[m_rows * k_blocks * 64 * 4 bytes]:  packed i8 activations
///   act_ds[m_rows * k_blocks * 8 * 4 bytes]:   Q8_1 block scales (f16)
/// Row stride: k_blocks * 256 bytes (qs), k_blocks * 32 bytes (ds).
///
/// Output layout: out[row * n_rows + n_out] -- row-major, M rows.
///
/// # Format compatibility
///
/// Weight format: native 36-u32 GGUF (144 bytes/block) -- same as kpar8.
/// Activation format: Q8_1 -- same as kpar8.
/// Requires `uses_raw_gguf=true` (i.e. sm_120 kpar is selected).
/// param_m_rows must equal 2.
pub const PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// q4k_gemv_sm120_batch (v2)
// Q4K x Q8_1 GEMV, M=2 rows, weight amortization via single block per neuron.
// Grid:  (n_rows, 1, 1)
// Block: (256, 1, 1) = 8 warps. K-parallel: warp w handles blocks w, w+8, ...
// Static smem: 256 bytes -- 2x32 f32 slots.
// Weights loaded ONCE per (n_out, k_block); used for both activation rows.
.visible .entry q4k_gemv_sm120_batch(
    .param .u64 param_weights,
    .param .u64 param_act_qs,
    .param .u64 param_act_ds,
    .param .u64 param_out,
    .param .u32 param_n_rows,
    .param .u32 param_k_blocks,
    .param .u32 param_m_rows
)
{
    .reg .u32  %r<66>;
    .reg .u64  %rd<16>;
    .reg .f32  %f<22>;
    .reg .pred %p<11>;
    .reg .b16  %h<6>;

    // Static smem:
    //   [0..127]:   row 0 partial sums (8 warps x 4 j-groups x 4 bytes)
    //   [128..255]: row 1 partial sums
    .shared .align 4 .b8 smem_buf[256];

    // -- Load kernel parameters -----------------------------------------------
    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_act_qs];
    ld.param.u64 %rd2, [param_act_ds];
    ld.param.u64 %rd3, [param_out];
    ld.param.u32 %r0,  [param_n_rows];
    ld.param.u32 %r1,  [param_k_blocks];
    // param_m_rows not read (M=2 hardcoded)

    // -- Guard: n_out < n_rows ------------------------------------------------
    mov.u32 %r2, %ctaid.x;       // n_out = blockIdx.x
    setp.ge.u32 %p0, %r2, %r0;
    @%p0 bra DONE;

    // -- Compute row 1 activation base pointers --------------------------------
    // act_qs row stride: k_blocks * 64 u32 * 4 bytes = k_blocks * 256 bytes
    shl.b32 %r59, %r1, 8;        // k_blocks * 256
    cvt.u64.u32 %rd13, %r59;
    add.u64 %rd13, %rd1, %rd13;  // %rd13 = act_qs + k_blocks * 256 (row 1 qs base)

    // act_ds row stride: k_blocks * 8 u32 * 4 bytes = k_blocks * 32 bytes
    shl.b32 %r60, %r1, 5;        // k_blocks * 32
    cvt.u64.u32 %rd14, %r60;
    add.u64 %rd14, %rd2, %rd14;  // %rd14 = act_ds + k_blocks * 32 (row 1 ds base)

    // -- Thread index decomposition -------------------------------------------
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

    // sub_l == 0 (for smem write in reduction)
    setp.eq.u32 %p2, %r7, 0;      // %p2 = (sub_l == 0)

    // row 0 accumulator = 0.0
    mov.f32 %f0,  0f00000000;
    // row 1 accumulator = 0.0
    mov.f32 %f19, 0f00000000;

    // K-loop: warp_id strides by 8 (WARPS_PER_BLOCK_KPAR = 8)
    mov.u32 %r12, %r5;            // blk = warp_id

    // -- Main K-loop -----------------------------------------------------------
KLOOP:
    setp.ge.u32 %p3, %r12, %r1;   // blk >= k_blocks -> exit
    @%p3 bra REDUCE;

    // weight base_ptr = weights + (n_out * k_blocks + blk) * 144 bytes
    mad.lo.u32 %r14, %r2, %r1, %r12;  // n_out * k_blocks + blk
    mul.lo.u32 %r15, %r14, 144;        // * 144 bytes/block
    cvt.u64.u32 %rd4, %r15;
    add.u64 %rd4, %rd0, %rd4;          // %rd4 = weight block base

    // prefetch next weight block into L2 (stride = 8 blocks = 1152 bytes)
    add.u32 %r56, %r12, 8;
    setp.lt.u32 %p10, %r56, %r1;
    add.u64 %rd10, %rd4, 1152;
    @%p10 prefetch.global.L2 [%rd10];

    // load d, dmin (f16 at bytes 0-1, 2-3)
    ld.global.nc.b16 %h0, [%rd4];
    ld.global.nc.b16 %h1, [%rd4 + 2];
    cvt.f32.f16 %f1, %h0;             // d as f32
    cvt.f32.f16 %f2, %h1;             // dmin as f32

    // load scale words (bytes 4-15)
    ld.global.nc.b32 %r16, [%rd4 + 4];
    ld.global.nc.b32 %r17, [%rd4 + 8];
    ld.global.nc.b32 %r18, [%rd4 + 12];

    // branchless scale extraction (identical to kpar8)
    add.u32 %r19, %r8, 8;

    shr.u32 %r20, %r16, %r8;  and.b32 %r20, %r20, 63;
    shr.u32 %r21, %r16, %r19; and.b32 %r21, %r21, 63;
    shr.u32 %r22, %r17, %r8;  and.b32 %r22, %r22, 63;
    shr.u32 %r23, %r17, %r19; and.b32 %r23, %r23, 63;

    shr.u32 %r40, %r16, %r8;  and.b32 %r40, %r40, 255;
    shr.u32 %r41, %r16, %r19; and.b32 %r41, %r41, 255;
    shr.u32 %r42, %r17, %r8;  and.b32 %r42, %r42, 255;
    shr.u32 %r43, %r17, %r19; and.b32 %r43, %r43, 255;
    shr.u32 %r44, %r18, %r8;  and.b32 %r44, %r44, 255;
    shr.u32 %r45, %r18, %r19; and.b32 %r45, %r45, 255;

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

    selp.u32 %r20, %r47, %r20, %p1;   // sc_lo
    selp.u32 %r21, %r48, %r21, %p1;   // sc_hi
    selp.u32 %r22, %r49, %r22, %p1;   // m_lo
    selp.u32 %r23, %r50, %r23, %p1;   // m_hi

    cvt.rn.f32.u32 %f3, %r20;
    cvt.rn.f32.u32 %f4, %r21;
    cvt.rn.f32.u32 %f5, %r22;
    cvt.rn.f32.u32 %f6, %r23;

    // load weight qs word for this (mw, sub_l) lane: u32[4 + mw*8 + sub_l]
    // %r25 = weight nibbles -- shared across both activation rows
    mul.lo.u32 %r24, %r6, 8;
    add.u32    %r24, %r24, 4;
    add.u32    %r24, %r24, %r7;
    shl.b32    %r24, %r24, 2;
    cvt.u64.u32 %rd5, %r24;
    add.u64    %rd5, %rd4, %rd5;
    ld.global.nc.b32 %r25, [%rd5];    // %r25 = weight qs (REUSED for both rows)

    // Extract nibbles once -- valid for both row 0 and row 1
    and.b32 %r32, %r25, 0x0F0F0F0F;   // v_lo = lo nibbles
    shr.u32 %r33, %r25, 4;
    and.b32 %r33, %r33, 0x0F0F0F0F;   // v_hi = hi nibbles

    // -- Row 0 activations (base pointers %rd1/%rd2) ---------------------------
    // q8_blk_lo = blk * 8 + mw * 2
    mul.lo.u32 %r26, %r12, 8;
    mul.lo.u32 %r27, %r6, 2;
    add.u32    %r26, %r26, %r27;      // %r26 = q8_blk_lo

    // u_lo_r0 = act_qs_r0[(q8_blk_lo * 8 + sub_l) * 4]
    mul.lo.u32 %r27, %r26, 8;
    add.u32    %r27, %r27, %r7;       // %r27 = q8_blk_lo * 8 + sub_l
    shl.b32    %r28, %r27, 2;
    cvt.u64.u32 %rd6, %r28;
    add.u64    %rd6, %rd1, %rd6;
    ld.global.b32 %r28, [%rd6];

    // u_hi_r0 = act_qs_r0[(q8_blk_lo * 8 + sub_l + 8) * 4]
    add.u32    %r29, %r27, 8;
    shl.b32    %r30, %r29, 2;
    cvt.u64.u32 %rd7, %r30;
    add.u64    %rd7, %rd1, %rd7;
    ld.global.b32 %r29, [%rd7];

    // d8_lo_r0 = act_ds_r0[q8_blk_lo * 4]  (f16)
    shl.b32    %r30, %r26, 2;
    cvt.u64.u32 %rd8, %r30;
    add.u64    %rd8, %rd2, %rd8;
    ld.global.b16 %h2, [%rd8];
    cvt.f32.f16 %f7, %h2;

    // d8_hi_r0 = act_ds_r0[(q8_blk_lo + 1) * 4]  (f16)
    add.u64    %rd9, %rd8, 4;
    ld.global.b16 %h3, [%rd9];
    cvt.f32.f16 %f8, %h3;

    // -- Row 0 dp4a: weight nibbles x Q8_1 bytes ------------------------------
    dp4a.s32.s32 %r34, %r32, %r28, 0; // dot_lo_r0
    dp4a.s32.s32 %r35, %r33, %r29, 0; // dot_hi_r0
    dp4a.s32.s32 %r36, 0x01010101, %r28, 0; // dot_min_lo_r0
    dp4a.s32.s32 %r37, 0x01010101, %r29, 0; // dot_min_hi_r0

    cvt.rn.f32.s32 %f9,  %r34;
    cvt.rn.f32.s32 %f10, %r35;
    cvt.rn.f32.s32 %f11, %r36;
    cvt.rn.f32.s32 %f12, %r37;

    // acc0 += (d*sc_lo*dot_lo - dmin*m_lo*dot_min_lo) * d8_lo
    mul.f32 %f13, %f1, %f3; mul.f32 %f13, %f13, %f9;
    mul.f32 %f14, %f2, %f5; mul.f32 %f14, %f14, %f11;
    sub.f32 %f13, %f13, %f14; mul.f32 %f13, %f13, %f7;
    add.f32 %f0, %f0, %f13;

    // acc0 += (d*sc_hi*dot_hi - dmin*m_hi*dot_min_hi) * d8_hi
    mul.f32 %f15, %f1, %f4; mul.f32 %f15, %f15, %f10;
    mul.f32 %f16, %f2, %f6; mul.f32 %f16, %f16, %f12;
    sub.f32 %f15, %f15, %f16; mul.f32 %f15, %f15, %f8;
    add.f32 %f0, %f0, %f15;

    // -- Row 1 activations (base pointers %rd13/%rd14) ------------------------
    // %r26 = q8_blk_lo (same value as row 0, already computed)
    // %r27 = q8_blk_lo * 8 + sub_l  (same offset index)

    // u_lo_r1 = act_qs_r1[(q8_blk_lo * 8 + sub_l) * 4]
    shl.b32    %r28, %r27, 2;         // REUSE %r28 for row 1
    cvt.u64.u32 %rd15, %r28;
    add.u64    %rd15, %rd13, %rd15;
    ld.global.b32 %r28, [%rd15];

    // u_hi_r1 = act_qs_r1[(q8_blk_lo * 8 + sub_l + 8) * 4]
    add.u32    %r29, %r27, 8;
    shl.b32    %r30, %r29, 2;
    cvt.u64.u32 %rd15, %r30;
    add.u64    %rd15, %rd13, %rd15;
    ld.global.b32 %r29, [%rd15];

    // d8_lo_r1 = act_ds_r1[q8_blk_lo * 4]  (f16)
    shl.b32    %r30, %r26, 2;
    cvt.u64.u32 %rd15, %r30;
    add.u64    %rd15, %rd14, %rd15;
    ld.global.b16 %h4, [%rd15];
    cvt.f32.f16 %f20, %h4;

    // d8_hi_r1 = act_ds_r1[(q8_blk_lo + 1) * 4]  (f16)
    add.u64    %rd15, %rd15, 4;
    ld.global.b16 %h5, [%rd15];
    cvt.f32.f16 %f21, %h5;

    // -- Row 1 dp4a: SAME weight nibbles (%r32/%r33 still in registers) -------
    dp4a.s32.s32 %r34, %r32, %r28, 0; // dot_lo_r1
    dp4a.s32.s32 %r35, %r33, %r29, 0; // dot_hi_r1
    dp4a.s32.s32 %r36, 0x01010101, %r28, 0; // dot_min_lo_r1
    dp4a.s32.s32 %r37, 0x01010101, %r29, 0; // dot_min_hi_r1

    cvt.rn.f32.s32 %f9,  %r34;
    cvt.rn.f32.s32 %f10, %r35;
    cvt.rn.f32.s32 %f11, %r36;
    cvt.rn.f32.s32 %f12, %r37;

    // acc1 += (d*sc_lo*dot_lo - dmin*m_lo*dot_min_lo) * d8_lo_r1
    // %f1..%f6 still valid in registers (weight data, unchanged)
    mul.f32 %f13, %f1, %f3; mul.f32 %f13, %f13, %f9;
    mul.f32 %f14, %f2, %f5; mul.f32 %f14, %f14, %f11;
    sub.f32 %f13, %f13, %f14; mul.f32 %f13, %f13, %f20;
    add.f32 %f19, %f19, %f13;

    // acc1 += (d*sc_hi*dot_hi - dmin*m_hi*dot_min_hi) * d8_hi_r1
    mul.f32 %f15, %f1, %f4; mul.f32 %f15, %f15, %f10;
    mul.f32 %f16, %f2, %f6; mul.f32 %f16, %f16, %f12;
    sub.f32 %f15, %f15, %f16; mul.f32 %f15, %f15, %f21;
    add.f32 %f19, %f19, %f15;

    add.u32 %r12, %r12, 8;
    bra KLOOP;

    // -- Reduction -------------------------------------------------------------
REDUCE:
    // Row 0: 3-step XOR butterfly within 8-lane mini-warp
    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 4, 0x181F, 0xFFFFFFFF;
    mov.b32 %f17, %r51; add.f32 %f0, %f0, %f17;
    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 2, 0x181F, 0xFFFFFFFF;
    mov.b32 %f17, %r51; add.f32 %f0, %f0, %f17;
    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 1, 0x181F, 0xFFFFFFFF;
    mov.b32 %f17, %r51; add.f32 %f0, %f0, %f17;

    // sub_l==0 writes row 0 mini-warp sum to smem[0..127]
    mul.lo.u32 %r52, %r5, 4;
    add.u32    %r52, %r52, %r6;
    shl.b32    %r53, %r52, 2;
    mov.u32    %r54, smem_buf;
    add.u32    %r54, %r54, %r53;
    @%p2 st.shared.f32 [%r54], %f0;

    // Row 1: 3-step XOR butterfly within 8-lane mini-warp
    mov.b32 %r50, %f19;
    shfl.sync.bfly.b32 %r51, %r50, 4, 0x181F, 0xFFFFFFFF;
    mov.b32 %f18, %r51; add.f32 %f19, %f19, %f18;
    mov.b32 %r50, %f19;
    shfl.sync.bfly.b32 %r51, %r50, 2, 0x181F, 0xFFFFFFFF;
    mov.b32 %f18, %r51; add.f32 %f19, %f19, %f18;
    mov.b32 %r50, %f19;
    shfl.sync.bfly.b32 %r51, %r50, 1, 0x181F, 0xFFFFFFFF;
    mov.b32 %f18, %r51; add.f32 %f19, %f19, %f18;

    // sub_l==0 writes row 1 mini-warp sum to smem[128..255]
    mov.u32    %r64, smem_buf;
    add.u32    %r64, %r64, 128;       // row 1 smem base
    add.u32    %r65, %r64, %r53;      // smem_buf + 128 + (warp_id*4 + mw)*4
    @%p2 st.shared.f32 [%r65], %f19;

    bar.sync 0;

    // -- Cross-warp reduce for row 0 ------------------------------------------
    // warp 0, lanes 0-3: sum smem[j*4 + warp*16] for warp = 0..7
    setp.eq.u32 %p6, %r5, 0;
    setp.lt.u32 %p7, %r4, 4;
    and.pred    %p8, %p6, %p7;

    mul.lo.u32 %r55, %r4, 4;
    mov.u32    %r54, smem_buf;
    add.u32    %r55, %r54, %r55;      // smem_buf + lane*4

    mov.f32    %f17, 0f00000000;
    @%p8 ld.shared.f32 %f18, [%r55 +   0]; @%p8 add.f32 %f17, %f17, %f18;
    @%p8 ld.shared.f32 %f18, [%r55 +  16]; @%p8 add.f32 %f17, %f17, %f18;
    @%p8 ld.shared.f32 %f18, [%r55 +  32]; @%p8 add.f32 %f17, %f17, %f18;
    @%p8 ld.shared.f32 %f18, [%r55 +  48]; @%p8 add.f32 %f17, %f17, %f18;
    @%p8 ld.shared.f32 %f18, [%r55 +  64]; @%p8 add.f32 %f17, %f17, %f18;
    @%p8 ld.shared.f32 %f18, [%r55 +  80]; @%p8 add.f32 %f17, %f17, %f18;
    @%p8 ld.shared.f32 %f18, [%r55 +  96]; @%p8 add.f32 %f17, %f17, %f18;
    @%p8 ld.shared.f32 %f18, [%r55 + 112]; @%p8 add.f32 %f17, %f17, %f18;
    @%p8 st.shared.f32 [%r55], %f17;

    // -- Cross-warp reduce for row 1 (smem[128..255]) -------------------------
    mul.lo.u32 %r62, %r4, 4;
    mov.u32    %r63, smem_buf;
    add.u32    %r63, %r63, 128;
    add.u32    %r62, %r63, %r62;      // smem_buf + 128 + lane*4

    mov.f32    %f20, 0f00000000;
    @%p8 ld.shared.f32 %f18, [%r62 +   0]; @%p8 add.f32 %f20, %f20, %f18;
    @%p8 ld.shared.f32 %f18, [%r62 +  16]; @%p8 add.f32 %f20, %f20, %f18;
    @%p8 ld.shared.f32 %f18, [%r62 +  32]; @%p8 add.f32 %f20, %f20, %f18;
    @%p8 ld.shared.f32 %f18, [%r62 +  48]; @%p8 add.f32 %f20, %f20, %f18;
    @%p8 ld.shared.f32 %f18, [%r62 +  64]; @%p8 add.f32 %f20, %f20, %f18;
    @%p8 ld.shared.f32 %f18, [%r62 +  80]; @%p8 add.f32 %f20, %f20, %f18;
    @%p8 ld.shared.f32 %f18, [%r62 +  96]; @%p8 add.f32 %f20, %f20, %f18;
    @%p8 ld.shared.f32 %f18, [%r62 + 112]; @%p8 add.f32 %f20, %f20, %f18;
    @%p8 st.shared.f32 [%r62], %f20;

    bar.sync 0;

    // -- Thread 0 writes both outputs ------------------------------------------
    setp.eq.u32 %p9, %r3, 0;
    mov.u32    %r53, smem_buf;

    // Row 0 final: sum smem[0..3] (j-group sums from warp 0)
    mov.f32    %f17, 0f00000000;
    @%p9 ld.shared.f32 %f17, [%r53 +  0];
    @%p9 ld.shared.f32 %f18, [%r53 +  4]; @%p9 add.f32 %f17, %f17, %f18;
    @%p9 ld.shared.f32 %f18, [%r53 +  8]; @%p9 add.f32 %f17, %f17, %f18;
    @%p9 ld.shared.f32 %f18, [%r53 + 12]; @%p9 add.f32 %f17, %f17, %f18;

    // Write row 0: out[0 * n_rows + n_out] = out[n_out]
    shl.b32    %r61, %r2, 2;           // n_out * 4
    cvt.u64.u32 %rd4, %r61;
    add.u64    %rd4, %rd3, %rd4;
    @%p9 st.global.f32 [%rd4], %f17;

    // Row 1 final: sum smem[128..131] (j-group sums from warp 0)
    add.u32    %r53, %r53, 128;
    mov.f32    %f20, 0f00000000;
    @%p9 ld.shared.f32 %f20, [%r53 +  0];
    @%p9 ld.shared.f32 %f18, [%r53 +  4]; @%p9 add.f32 %f20, %f20, %f18;
    @%p9 ld.shared.f32 %f18, [%r53 +  8]; @%p9 add.f32 %f20, %f20, %f18;
    @%p9 ld.shared.f32 %f18, [%r53 + 12]; @%p9 add.f32 %f20, %f20, %f18;

    // Write row 1: out[1 * n_rows + n_out] = out[n_rows + n_out]
    add.u32    %r62, %r0, %r2;         // n_rows + n_out
    shl.b32    %r62, %r62, 2;          // * 4 bytes
    cvt.u64.u32 %rd4, %r62;
    add.u64    %rd4, %rd3, %rd4;
    @%p9 st.global.f32 [%rd4], %f20;

DONE:
    ret;
}
"#;

/// Kernel entry name for `cuModuleGetFunction`.
pub const ENTRY: &str = "q4k_gemv_sm120_batch";

/// Minimum compute capability (sm_120 = Blackwell).
pub const MIN_CC: u32 = 1200;
