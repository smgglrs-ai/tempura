/// Hand-tuned sm_120 PTX for fused gate+up GEMV with inline SiLU.
///
/// Kernel: q4k_fused_silu_sm120_dp4a
/// Computes: out[i] = silu(gate_dot[i]) * up_dot[i]
///   where gate_dot[i] and up_dot[i] are Q4K x Q8_1 dot products.
///
/// # Weight format: repacked 42-u32 (168 bytes/block of 256 elements)
///
/// ```text
/// u32[0]      : f32 bits of d    (f16->f32 at upload time)
/// u32[1]      : f32 bits of dmin (f16->f32 at upload time)
/// u32[2+mw]   : scale word for mini-warp mw (mw=0..3):
///                 bits[7:0]  = sc_lo (scale for lo Q8_1 sub-block)
///                 bits[15:8] = sc_hi (scale for hi Q8_1 sub-block)
/// u32[6+mw]   : min word for mini-warp mw (same encoding as scale)
/// u32[10+mw*8+sub_l]: packed nibble weights (32 u32s, one per lane)
/// ```
///
/// # Q8_1 activation format (9 u32 = 36 bytes per 32-element block)
///
/// ```text
/// u32[0..8]: 32 packed i8 activations (4 per u32, sub_l indexes which u32)
/// u32[8]:    block scale d (f16 in low 16 bits) -- but stored as u32 here
/// ```
/// Activation ds array: u32[blk] = f16 d as u32 (low 16 bits = f16 bits).
///
/// # Launch geometry
///
/// Grid:  (N, 1, 1)   -- one block per output element
/// Block: (128, 1, 1) -- 4 warps x 32 threads
/// Smem:  128 bytes   -- 32 f32 slots (16 gate + 16 up: [warp*4+mw] layout)
///
/// # K-loop
///
/// VDR=1, stride = WARPS_PER_BLOCK = 4.
/// warp w owns blocks w, w+4, w+8, ...
/// Each block: dual deferred-D accumulators (gate_acc, up_acc).
/// Accumulation: gate_acc += gd * (d8_lo*sc_lo*dot_lo + d8_hi*sc_hi*dot_hi)
///                         - gdmin * (d8_lo*m_lo*min_lo + d8_hi*m_hi*min_hi)
/// Same for up_acc.
///
/// # Reduction
///
/// Mini-warp butterfly (3-step XOR within 8-lane mini-warp, c=0x181F).
/// Cross-warp via smem: warp 0 lanes 0..3 sum across warps per mw group.
/// Thread 0 sums mw groups, applies SiLU, writes out[blockIdx.x].
///
/// # SiLU via ex2.approx
///
/// sigmoid(x) = 1 / (1 + e^{-x}) = 1 / (1 + ex2(-x * 1.44269504))
/// silu(x) = x * sigmoid(x)
pub const PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// q4k_fused_silu_sm120_dp4a
// Fused gate+up Q4K GEMV with inline SiLU. sm_120 hand-tuned PTX.
// Block: 128 threads = 4 warps. Grid: (N,1,1). Smem: 128 bytes (32 f32 slots).
// Weights: repacked 42-u32 format. Activations: Q8_1 (pre-quantized).
.visible .entry q4k_fused_silu_sm120_dp4a(
    .param .u64 param_gate,
    .param .u64 param_up,
    .param .u64 param_act_qs,
    .param .u64 param_act_ds,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .u32 param_k_blocks
)
{
    .reg .u32  %r<64>;
    .reg .u64  %rd<9>;
    .reg .f32  %f<26>;
    .reg .pred %p<8>;
    .reg .b16  %h<2>;

    // smem[0..15] = gate mini-warp sums (slot warp*4+mw)
    // smem[16..31] = up mini-warp sums  (slot 16+warp*4+mw)
    // 32 f32 slots = 128 bytes
    .shared .align 4 .b8 smem_buf[128];

    // -- Load kernel parameters -----------------------------------------------
    ld.param.u64 %rd0, [param_gate];
    ld.param.u64 %rd1, [param_up];
    ld.param.u64 %rd2, [param_act_qs];
    ld.param.u64 %rd3, [param_act_ds];
    ld.param.u64 %rd4, [param_out];
    ld.param.u32 %r0,  [param_n];
    ld.param.u32 %r1,  [param_k_blocks];

    // -- Thread decomposition -------------------------------------------------
    mov.u32 %r2, %ctaid.x;          // n_out = blockIdx.x
    setp.ge.u32 %p0, %r2, %r0;      // guard: n_out >= n -> exit
    @%p0 bra DONE;

    mov.u32 %r3, %tid.x;             // flat tid (0..127)
    and.b32 %r4, %r3, 31;            // lane = tid & 31
    shr.u32 %r5, %r3, 5;             // warp_id = tid >> 5  (0..3)
    shr.u32 %r6, %r4, 3;             // mw = lane >> 3       (0..3, mini-warp)
    and.b32 %r7, %r4, 7;             // sub_l = lane & 7

    // base_lane = mw * 8  (first lane of this mini-warp)
    shl.b32 %r8, %r6, 3;

    // mini_mask = 0xFF << (mw * 8): bitmask for this mini-warp's 8 lanes
    mov.u32 %r9, 255;
    shl.b32 %r9, %r9, %r8;

    // sub_l == 0 predicate: one lane per mini-warp writes to smem
    setp.eq.u32 %p2, %r7, 0;

    // Dual accumulators
    mov.f32 %f0, 0f00000000;         // gate_acc
    mov.f32 %f1, 0f00000000;         // up_acc

    // K-loop: warp w handles blocks w, w+4, w+8, ... (stride = WARPS_PER_BLOCK = 4)
    mov.u32 %r12, %r5;               // blk = warp_id

    // -- Main K-loop ----------------------------------------------------------
KLOOP:
    setp.ge.u32 %p3, %r12, %r1;      // blk >= k_blocks -> reduce
    @%p3 bra REDUCE;

    // byte_offset = (n_out * k_blocks + blk) * 168  (42 u32s * 4 bytes)
    mad.lo.u32 %r14, %r2, %r1, %r12; // block_idx = n_out * k_blocks + blk
    mul.lo.u32 %r15, %r14, 168;       // * 168 bytes/block
    cvt.u64.u32 %rd7, %r15;           // u64 byte offset
    add.u64 %rd5, %rd0, %rd7;         // gate block base
    add.u64 %rd6, %rd1, %rd7;         // up block base

    // -- Load gate d, dmin (stored as f32 bits at u32[0], u32[1]; broadcast from lane 0)
    ld.global.nc.b32 %r16, [%rd5];
    ld.global.nc.b32 %r17, [%rd5 + 4];
    shfl.sync.idx.b32 %r16, %r16, 0, 31, 0xFFFFFFFF;  // broadcast d from lane 0
    shfl.sync.idx.b32 %r17, %r17, 0, 31, 0xFFFFFFFF;  // broadcast dmin
    mov.b32 %f2, %r16;                 // gd
    mov.b32 %f3, %r17;                 // gdmin

    // -- Load up d, dmin
    ld.global.nc.b32 %r18, [%rd6];
    ld.global.nc.b32 %r19, [%rd6 + 4];
    shfl.sync.idx.b32 %r18, %r18, 0, 31, 0xFFFFFFFF;
    shfl.sync.idx.b32 %r19, %r19, 0, 31, 0xFFFFFFFF;
    mov.b32 %f4, %r18;                 // ud
    mov.b32 %f5, %r19;                 // udmin

    // -- Load gate scale word for this mini-warp: u32[2 + mw]
    //    Only sub_l==0 lane loads; then broadcast within mini-warp.
    add.u32 %r20, %r6, 2;             // 2 + mw
    shl.b32 %r20, %r20, 2;            // * 4 bytes
    cvt.u64.u32 %rd8, %r20;
    add.u64 %rd8, %rd5, %rd8;
    mov.u32 %r21, 0;
    @%p2 ld.global.nc.b32 %r21, [%rd8];
    shfl.sync.idx.b32 %r21, %r21, %r8, 0x181F, 0xFFFFFFFF;  // broadcast within mini-warp

    // -- Load gate min word: u32[6 + mw]
    add.u32 %r22, %r6, 6;
    shl.b32 %r22, %r22, 2;
    cvt.u64.u32 %rd8, %r22;
    add.u64 %rd8, %rd5, %rd8;
    mov.u32 %r23, 0;
    @%p2 ld.global.nc.b32 %r23, [%rd8];
    shfl.sync.idx.b32 %r23, %r23, %r8, 0x181F, 0xFFFFFFFF;

    // -- Load up scale word: u32[2 + mw]
    cvt.u64.u32 %rd8, %r20;
    add.u64 %rd8, %rd6, %rd8;
    mov.u32 %r24, 0;
    @%p2 ld.global.nc.b32 %r24, [%rd8];
    shfl.sync.idx.b32 %r24, %r24, %r8, 0x181F, 0xFFFFFFFF;

    // -- Load up min word: u32[6 + mw]
    cvt.u64.u32 %rd8, %r22;
    add.u64 %rd8, %rd6, %rd8;
    mov.u32 %r25, 0;
    @%p2 ld.global.nc.b32 %r25, [%rd8];
    shfl.sync.idx.b32 %r25, %r25, %r8, 0x181F, 0xFFFFFFFF;

    // -- Extract scale/min byte fields
    // sc_lo = word & 0xFF, sc_hi = (word >> 8) & 0xFF
    and.b32 %r26, %r21, 255;          // gate sc_lo
    shr.u32 %r27, %r21, 8;
    and.b32 %r27, %r27, 255;          // gate sc_hi
    and.b32 %r28, %r23, 255;          // gate m_lo
    shr.u32 %r29, %r23, 8;
    and.b32 %r29, %r29, 255;          // gate m_hi

    and.b32 %r30, %r24, 255;          // up sc_lo
    shr.u32 %r31, %r24, 8;
    and.b32 %r31, %r31, 255;          // up sc_hi
    and.b32 %r32, %r25, 255;          // up m_lo
    shr.u32 %r33, %r25, 8;
    and.b32 %r33, %r33, 255;          // up m_hi

    cvt.rn.f32.u32 %f6,  %r26;        // gsc_lo f32
    cvt.rn.f32.u32 %f7,  %r27;        // gsc_hi f32
    cvt.rn.f32.u32 %f8,  %r28;        // gm_lo f32
    cvt.rn.f32.u32 %f9,  %r29;        // gm_hi f32
    cvt.rn.f32.u32 %f10, %r30;        // usc_lo f32
    cvt.rn.f32.u32 %f11, %r31;        // usc_hi f32
    cvt.rn.f32.u32 %f12, %r32;        // um_lo f32
    cvt.rn.f32.u32 %f13, %r33;        // um_hi f32

    // -- Load gate qs: u32[10 + mw*8 + sub_l]
    mul.lo.u32 %r34, %r6, 8;          // mw * 8
    add.u32    %r34, %r34, 10;         // + 10 (skip header)
    add.u32    %r34, %r34, %r7;        // + sub_l
    shl.b32    %r34, %r34, 2;          // * 4 bytes
    cvt.u64.u32 %rd8, %r34;
    add.u64    %rd8, %rd5, %rd8;
    ld.global.nc.b32 %r35, [%rd8];    // gate qs u32

    // -- Load up qs (same offset structure)
    cvt.u64.u32 %rd8, %r34;
    add.u64    %rd8, %rd6, %rd8;
    ld.global.nc.b32 %r36, [%rd8];    // up qs u32

    // -- Load Q8_1 activations ------------------------------------------------
    // q8_blk_lo = blk * 8 + mw * 2
    //   Each Q4K super-block (256 elements) = 8 Q8_1 blocks (32 elements each).
    //   Mini-warp mw handles 64 elements -> 2 Q8_1 blocks (lo and hi).
    mul.lo.u32 %r37, %r12, 8;         // blk * 8
    mul.lo.u32 %r38, %r6,  2;         // mw * 2
    add.u32    %r37, %r37, %r38;      // q8_blk_lo

    // u_lo = act_qs[q8_blk_lo * 8 + sub_l]
    mul.lo.u32 %r39, %r37, 8;
    add.u32    %r39, %r39, %r7;       // index into qs array
    shl.b32    %r40, %r39, 2;         // * 4 bytes
    cvt.u64.u32 %rd8, %r40;
    add.u64    %rd8, %rd2, %rd8;
    ld.global.b32 %r41, [%rd8];       // u_lo (4 packed i8)

    // u_hi = act_qs[q8_blk_lo * 8 + sub_l + 8]  (hi Q8_1 sub-block)
    add.u32    %r42, %r39, 8;
    shl.b32    %r43, %r42, 2;
    cvt.u64.u32 %rd8, %r43;
    add.u64    %rd8, %rd2, %rd8;
    ld.global.b32 %r44, [%rd8];       // u_hi

    // d8_lo = act_ds[q8_blk_lo] (f16 in low 16 bits)
    shl.b32    %r45, %r37, 2;         // q8_blk_lo * 4 bytes
    cvt.u64.u32 %rd8, %r45;
    add.u64    %rd8, %rd3, %rd8;
    ld.global.b16 %h0, [%rd8];
    cvt.f32.f16 %f14, %h0;            // d8_lo f32

    // d8_hi = act_ds[q8_blk_lo + 1]
    add.u64    %rd8, %rd8, 4;
    ld.global.b16 %h1, [%rd8];
    cvt.f32.f16 %f15, %h1;            // d8_hi f32

    // -- dp4a: gate nibbles x Q8_1 bytes -------------------------------------
    // Nibble extraction: v_lo = lo nibbles (bits 3:0 of each byte), v_hi = hi nibbles
    and.b32    %r46, %r35, 0x0F0F0F0F;    // gate v_lo
    shr.u32    %r47, %r35, 4;
    and.b32    %r47, %r47, 0x0F0F0F0F;    // gate v_hi

    dp4a.s32.s32 %r48, %r46, %r41, 0;     // gate dot_lo = sum(v_lo * u_lo)
    dp4a.s32.s32 %r49, %r47, %r44, 0;     // gate dot_hi = sum(v_hi * u_hi)
    dp4a.s32.s32 %r50, 0x01010101, %r41, 0; // gate dot_min_lo = sum(u_lo)
    dp4a.s32.s32 %r51, 0x01010101, %r44, 0; // gate dot_min_hi = sum(u_hi)

    // -- dp4a: up nibbles x Q8_1 bytes
    and.b32    %r52, %r36, 0x0F0F0F0F;    // up v_lo
    shr.u32    %r53, %r36, 4;
    and.b32    %r53, %r53, 0x0F0F0F0F;    // up v_hi

    dp4a.s32.s32 %r54, %r52, %r41, 0;     // up dot_lo
    dp4a.s32.s32 %r55, %r53, %r44, 0;     // up dot_hi
    dp4a.s32.s32 %r56, 0x01010101, %r41, 0; // up dot_min_lo
    dp4a.s32.s32 %r57, 0x01010101, %r44, 0; // up dot_min_hi

    // -- Accumulate gate: gate_acc += gd*(d8_lo*sc_lo*dot_lo + d8_hi*sc_hi*dot_hi)
    //                               - gdmin*(d8_lo*m_lo*min_lo + d8_hi*m_hi*min_hi)
    cvt.rn.f32.s32 %f16, %r48;            // dot_lo f32
    cvt.rn.f32.s32 %f17, %r49;            // dot_hi f32
    cvt.rn.f32.s32 %f18, %r50;            // dot_min_lo f32
    cvt.rn.f32.s32 %f19, %r51;            // dot_min_hi f32

    // term_lo = gd * d8_lo * sc_lo * dot_lo - gdmin * d8_lo * m_lo * dot_min_lo
    mul.f32 %f20, %f2,  %f14;             // gd * d8_lo
    mul.f32 %f20, %f20, %f6;              // * sc_lo
    mul.f32 %f20, %f20, %f16;             // * dot_lo

    mul.f32 %f21, %f3,  %f14;             // gdmin * d8_lo
    mul.f32 %f21, %f21, %f8;              // * m_lo
    mul.f32 %f21, %f21, %f18;             // * dot_min_lo

    sub.f32 %f20, %f20, %f21;             // term_lo
    add.f32 %f0,  %f0,  %f20;             // gate_acc += term_lo

    // term_hi = gd * d8_hi * sc_hi * dot_hi - gdmin * d8_hi * m_hi * dot_min_hi
    mul.f32 %f22, %f2,  %f15;             // gd * d8_hi
    mul.f32 %f22, %f22, %f7;              // * sc_hi
    mul.f32 %f22, %f22, %f17;             // * dot_hi

    mul.f32 %f23, %f3,  %f15;             // gdmin * d8_hi
    mul.f32 %f23, %f23, %f9;              // * m_hi
    mul.f32 %f23, %f23, %f19;             // * dot_min_hi

    sub.f32 %f22, %f22, %f23;
    add.f32 %f0,  %f0,  %f22;             // gate_acc += term_hi

    // -- Accumulate up
    cvt.rn.f32.s32 %f16, %r54;
    cvt.rn.f32.s32 %f17, %r55;
    cvt.rn.f32.s32 %f18, %r56;
    cvt.rn.f32.s32 %f19, %r57;

    mul.f32 %f20, %f4,  %f14;             // ud * d8_lo
    mul.f32 %f20, %f20, %f10;             // * usc_lo
    mul.f32 %f20, %f20, %f16;

    mul.f32 %f21, %f5,  %f14;             // udmin * d8_lo
    mul.f32 %f21, %f21, %f12;             // * um_lo
    mul.f32 %f21, %f21, %f18;

    sub.f32 %f20, %f20, %f21;
    add.f32 %f1,  %f1,  %f20;             // up_acc += term_lo

    mul.f32 %f22, %f4,  %f15;
    mul.f32 %f22, %f22, %f11;             // * usc_hi
    mul.f32 %f22, %f22, %f17;

    mul.f32 %f23, %f5,  %f15;
    mul.f32 %f23, %f23, %f13;             // * um_hi
    mul.f32 %f23, %f23, %f19;

    sub.f32 %f22, %f22, %f23;
    add.f32 %f1,  %f1,  %f22;             // up_acc += term_hi

    // -- Advance K-loop
    add.u32 %r12, %r12, 4;                // blk += WARPS_PER_BLOCK (= 4)
    bra KLOOP;

    // -- Mini-warp butterfly reduction ----------------------------------------
    // 3-step XOR within 8-lane mini-warp. c=0x181F: nvcc width-8 encoding
    // (the old 0x71F broke mini-warp 3 -- see q4k_gemv_v4_ptx.rs header).
    // After 3 steps all lanes in each mini-warp hold the mini-warp partial sum.
REDUCE:
    mov.b32 %r58, %f0;
    shfl.sync.bfly.b32 %r59, %r58, 4, 0x181F, 0xFFFFFFFF;
    mov.b32 %f24, %r59; add.f32 %f0, %f0, %f24;
    mov.b32 %r58, %f0;
    shfl.sync.bfly.b32 %r59, %r58, 2, 0x181F, 0xFFFFFFFF;
    mov.b32 %f24, %r59; add.f32 %f0, %f0, %f24;
    mov.b32 %r58, %f0;
    shfl.sync.bfly.b32 %r59, %r58, 1, 0x181F, 0xFFFFFFFF;
    mov.b32 %f24, %r59; add.f32 %f0, %f0, %f24;

    mov.b32 %r58, %f1;
    shfl.sync.bfly.b32 %r59, %r58, 4, 0x181F, 0xFFFFFFFF;
    mov.b32 %f24, %r59; add.f32 %f1, %f1, %f24;
    mov.b32 %r58, %f1;
    shfl.sync.bfly.b32 %r59, %r58, 2, 0x181F, 0xFFFFFFFF;
    mov.b32 %f24, %r59; add.f32 %f1, %f1, %f24;
    mov.b32 %r58, %f1;
    shfl.sync.bfly.b32 %r59, %r58, 1, 0x181F, 0xFFFFFFFF;
    mov.b32 %f24, %r59; add.f32 %f1, %f1, %f24;

    // -- Write mini-warp sums to smem (sub_l == 0 only) -----------------------
    // Gate: smem slot (warp_id*4+mw), byte offset (warp_id*4+mw)*4
    // Up:   smem slot (16+warp_id*4+mw), byte offset 64 + (warp_id*4+mw)*4
    mul.lo.u32 %r60, %r5, 4;              // warp_id * 4
    add.u32    %r60, %r60, %r6;           // warp_id*4 + mw
    shl.b32    %r61, %r60, 2;             // * 4 bytes
    mov.u32    %r62, smem_buf;
    add.u32    %r62, %r62, %r61;          // &smem[warp_id*4+mw]
    @%p2 st.shared.f32 [%r62],      %f0;  // gate sum
    @%p2 st.shared.f32 [%r62 + 64], %f1;  // up sum (64 = 16*4 bytes offset)

    bar.sync 0;

    // -- Cross-warp reduction: warp 0 lanes 0..3 sum across 4 warps -----------
    // Lane j reads gate slots j, 4+j, 8+j, 12+j and up slots 16+j, 20+j, 24+j, 28+j.
    setp.eq.u32 %p4, %r5, 0;             // warp_id == 0
    setp.lt.u32 %p5, %r4, 4;             // lane < 4
    and.pred    %p6, %p4, %p5;

    // Lane j -> byte base = smem_buf + lane*4
    mul.lo.u32 %r60, %r4, 4;             // j * 4
    mov.u32    %r62, smem_buf;
    add.u32    %r63, %r62, %r60;          // smem_buf + j*4

    // Gate sum: slots j, 4+j, 8+j, 12+j -> bytes j*4, j*4+16, j*4+32, j*4+48
    mov.f32 %f24, 0f00000000;
    @%p6 ld.shared.f32 %f20, [%r63 +  0]; @%p6 add.f32 %f24, %f24, %f20; // warp 0
    @%p6 ld.shared.f32 %f20, [%r63 + 16]; @%p6 add.f32 %f24, %f24, %f20; // warp 1
    @%p6 ld.shared.f32 %f20, [%r63 + 32]; @%p6 add.f32 %f24, %f24, %f20; // warp 2
    @%p6 ld.shared.f32 %f20, [%r63 + 48]; @%p6 add.f32 %f24, %f24, %f20; // warp 3
    @%p6 st.shared.f32 [%r63], %f24;      // write gate j-sum back to smem[j]

    // Up sum: same but +64 bytes offset (slots 16..31)
    mov.f32 %f25, 0f00000000;
    @%p6 ld.shared.f32 %f20, [%r63 + 64];  @%p6 add.f32 %f25, %f25, %f20;
    @%p6 ld.shared.f32 %f20, [%r63 + 80];  @%p6 add.f32 %f25, %f25, %f20;
    @%p6 ld.shared.f32 %f20, [%r63 + 96];  @%p6 add.f32 %f25, %f25, %f20;
    @%p6 ld.shared.f32 %f20, [%r63 + 112]; @%p6 add.f32 %f25, %f25, %f20;
    @%p6 st.shared.f32 [%r63 + 64], %f25;  // write up j-sum back to smem[16+j]

    bar.sync 0;

    // -- Thread 0: sum mw groups, apply SiLU, write output --------------------
    setp.eq.u32 %p7, %r3, 0;              // tid == 0
    mov.u32     %r62, smem_buf;

    // gate_total = smem[0] + smem[4] + smem[8] + smem[12]  (bytes 0,4,8,12)
    mov.f32 %f24, 0f00000000;
    @%p7 ld.shared.f32 %f20, [%r62 +  0]; @%p7 add.f32 %f24, %f24, %f20;
    @%p7 ld.shared.f32 %f20, [%r62 +  4]; @%p7 add.f32 %f24, %f24, %f20;
    @%p7 ld.shared.f32 %f20, [%r62 +  8]; @%p7 add.f32 %f24, %f24, %f20;
    @%p7 ld.shared.f32 %f20, [%r62 + 12]; @%p7 add.f32 %f24, %f24, %f20;

    // up_total = smem[64] + smem[68] + smem[72] + smem[76]  (bytes 64,68,72,76)
    mov.f32 %f25, 0f00000000;
    @%p7 ld.shared.f32 %f20, [%r62 + 64]; @%p7 add.f32 %f25, %f25, %f20;
    @%p7 ld.shared.f32 %f20, [%r62 + 68]; @%p7 add.f32 %f25, %f25, %f20;
    @%p7 ld.shared.f32 %f20, [%r62 + 72]; @%p7 add.f32 %f25, %f25, %f20;
    @%p7 ld.shared.f32 %f20, [%r62 + 76]; @%p7 add.f32 %f25, %f25, %f20;

    // SiLU: silu(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
    // sigmoid(x) via ex2.approx: exp(-x) = ex2(-x * log2(e)) = ex2(-x * 1.44269504)
    // 0fBFB8AA3B = -1.44269504f (= -log2(e) = -1/ln2)
    // 0f3F800000 = 1.0f
    @%p7 mul.f32 %f20, %f24, 0fBFB8AA3B;   // -gate * log2(e)
    @%p7 ex2.approx.f32 %f21, %f20;         // exp(-gate) ~ ex2(-gate*log2e)
    @%p7 add.f32 %f21, %f21, 0f3F800000;    // 1 + exp(-gate)
    @%p7 rcp.approx.f32 %f21, %f21;         // sigmoid(gate)
    @%p7 mul.f32 %f22, %f24, %f21;          // silu(gate) = gate * sigmoid(gate)
    @%p7 mul.f32 %f22, %f22, %f25;          // out = silu(gate) * up

    // Write output: out[n_out]
    shl.b32    %r60, %r2, 2;               // n_out * 4 bytes
    cvt.u64.u32 %rd8, %r60;
    add.u64    %rd8, %rd4, %rd8;
    @%p7 st.global.f32 [%rd8], %f22;

DONE:
    ret;
}
"#;

/// Kernel entry point name (must match `.visible .entry` in PTX string above).
pub const ENTRY: &str = "q4k_fused_silu_sm120_dp4a";

/// Minimum compute capability for this PTX (Blackwell sm_120).
pub const MIN_CC: u32 = 1200;
