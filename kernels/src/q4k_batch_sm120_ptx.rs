/// Hand-tuned PTX batch kernel for Q4K x Q8_1 GEMV decode, targeting sm_120 (Blackwell).
///
/// # Purpose
///
/// The oxide sm_80 PTX kernels that use `DynamicSharedArray` (`q4k_gemv_dp4a_base_batch`,
/// `q4k_gemm_colocated`) crash with CUDA error 719 on RTX 5090 (sm_120 Blackwell) due
/// to JIT recompilation incompatibility of dynamic shared memory. This kernel avoids the
/// crash by using exclusively STATIC shared memory (`.shared .align 4 .b8 smem_buf[128]`).
///
/// # Design
///
/// Extension of `q4k_gemv_sm120_dp4a_kpar8` with an added batch dimension.
/// Each block handles one (n_out, row_idx) pair:
///   - `n_out   = blockIdx.x`  -- output neuron (0..n_rows)
///   - `row_idx = blockIdx.y`  -- batch row     (0..m_rows)
///
/// Grid:  (n_rows, m_rows, 1)
/// Block: (256, 1, 1) = 8 warps
/// Smem:  128 bytes = 32 f32 slots (8 warps x 4 j-groups) -- STATIC, no dynamic smem
///
/// Activation layout (row-major, quantized by a single `quantize_q8_1` call over M*K):
///   act_qs[m_rows * k_blocks * 64]:  packed i8 activations (8 u32 per Q8_1 block of 32)
///   act_ds[m_rows * k_blocks * 8]:   Q8_1 block scales (f16 in low 16 bits of u32)
///
/// The kernel offsets into row_idx via byte arithmetic before entering the K-loop,
/// so the inner loop body is identical to the single-row kpar8 kernel.
///
/// # Format compatibility
///
/// Weight format: native 36-u32 GGUF (144 bytes/block) -- same as kpar8.
/// Activation format: Q8_1 -- same as kpar8.
/// Requires `uses_raw_gguf=true` (i.e. sm_120 kpar is selected).
pub const PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// q4k_gemv_sm120_batch
// Native Q4K (36 u32/block) x Q8_1 GEMV, batch M rows.
// Grid:  (n_rows, m_rows, 1)
// Block: (256, 1, 1) = 8 warps. K-parallel: warp w handles block w, w+8, ...
// Static smem: 128 bytes -- 32 f32 slots (8 warps x 4 j-groups).
// Activation row stride: k_blocks * 256 bytes (qs) / k_blocks * 32 bytes (ds).
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
    .reg .u32  %r<62>;
    .reg .u64  %rd<13>;
    .reg .f32  %f<20>;
    .reg .pred %p<11>;
    .reg .b16  %h<4>;

    // Static smem: warp_id * 4 + mw -> f32 mini-warp partial sum.
    .shared .align 4 .b8 smem_buf[128];

    // -- Load kernel parameters -----------------------------------------------
    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_act_qs];
    ld.param.u64 %rd2, [param_act_ds];
    ld.param.u64 %rd3, [param_out];
    ld.param.u32 %r0,  [param_n_rows];
    ld.param.u32 %r1,  [param_k_blocks];
    ld.param.u32 %r57, [param_m_rows];

    // -- Guard: n_out < n_rows, row_idx < m_rows ------------------------------
    mov.u32 %r2, %ctaid.x;       // n_out = blockIdx.x
    setp.ge.u32 %p0, %r2, %r0;
    @%p0 bra DONE;

    mov.u32 %r58, %ctaid.y;      // row_idx = blockIdx.y
    setp.ge.u32 %p0, %r58, %r57;
    @%p0 bra DONE;

    // -- Adjust activation pointers by row_idx --------------------------------
    // act_qs row stride: k_blocks * 64 u32 * 4 bytes = k_blocks * 256 bytes
    mul.lo.u32 %r59, %r58, %r1;  // row_idx * k_blocks
    shl.b32    %r59, %r59, 8;    // * 256 bytes
    cvt.u64.u32 %rd11, %r59;
    add.u64    %rd1, %rd1, %rd11; // act_qs += row_idx * k_blocks * 256

    // act_ds row stride: k_blocks * 8 u32 * 4 bytes = k_blocks * 32 bytes
    mul.lo.u32 %r60, %r58, %r1;  // row_idx * k_blocks
    shl.b32    %r60, %r60, 5;    // * 32 bytes
    cvt.u64.u32 %rd12, %r60;
    add.u64    %rd2, %rd2, %rd12; // act_ds += row_idx * k_blocks * 32

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

    // acc = 0.0
    mov.f32 %f0, 0f00000000;

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

    // load qs word for this (mw, sub_l) lane: u32[4 + mw*8 + sub_l]
    mul.lo.u32 %r24, %r6, 8;
    add.u32    %r24, %r24, 4;
    add.u32    %r24, %r24, %r7;
    shl.b32    %r24, %r24, 2;
    cvt.u64.u32 %rd5, %r24;
    add.u64    %rd5, %rd4, %rd5;
    ld.global.nc.b32 %r25, [%rd5];

    // load Q8_1 activation data (act_qs and act_ds already row-adjusted above)
    // q8_blk_lo = blk * 8 + mw * 2
    mul.lo.u32 %r26, %r12, 8;
    mul.lo.u32 %r27, %r6, 2;
    add.u32    %r26, %r26, %r27;

    // u_lo = act_qs[(q8_blk_lo * 8 + sub_l) * 4]
    mul.lo.u32 %r27, %r26, 8;
    add.u32    %r27, %r27, %r7;
    shl.b32    %r28, %r27, 2;
    cvt.u64.u32 %rd6, %r28;
    add.u64    %rd6, %rd1, %rd6;
    ld.global.b32 %r28, [%rd6];

    // u_hi = act_qs[(q8_blk_lo * 8 + sub_l + 8) * 4]
    add.u32    %r29, %r27, 8;
    shl.b32    %r30, %r29, 2;
    cvt.u64.u32 %rd7, %r30;
    add.u64    %rd7, %rd1, %rd7;
    ld.global.b32 %r29, [%rd7];

    // d8_lo = act_ds[q8_blk_lo * 4]  (f16)
    shl.b32    %r30, %r26, 2;
    cvt.u64.u32 %rd8, %r30;
    add.u64    %rd8, %rd2, %rd8;
    ld.global.b16 %h2, [%rd8];
    cvt.f32.f16 %f7, %h2;

    // d8_hi = act_ds[(q8_blk_lo + 1) * 4]  (f16)
    add.u64    %rd9, %rd8, 4;
    ld.global.b16 %h3, [%rd9];
    cvt.f32.f16 %f8, %h3;

    // dp4a: Q4K nibbles x Q8_1 bytes
    and.b32 %r32, %r25, 0x0F0F0F0F;   // v_lo = lo nibbles
    shr.u32 %r33, %r25, 4;
    and.b32 %r33, %r33, 0x0F0F0F0F;   // v_hi = hi nibbles

    dp4a.s32.s32 %r34, %r32, %r28, 0; // dot_lo
    dp4a.s32.s32 %r35, %r33, %r29, 0; // dot_hi
    dp4a.s32.s32 %r36, 0x01010101, %r28, 0; // dot_min_lo
    dp4a.s32.s32 %r37, 0x01010101, %r29, 0; // dot_min_hi

    cvt.rn.f32.s32 %f9,  %r34;
    cvt.rn.f32.s32 %f10, %r35;
    cvt.rn.f32.s32 %f11, %r36;
    cvt.rn.f32.s32 %f12, %r37;

    // accumulate: term_lo = (d*sc_lo*dot_lo - dmin*m_lo*dot_min_lo) * d8_lo
    mul.f32 %f13, %f1, %f3; mul.f32 %f13, %f13, %f9;
    mul.f32 %f14, %f2, %f5; mul.f32 %f14, %f14, %f11;
    sub.f32 %f13, %f13, %f14; mul.f32 %f13, %f13, %f7;
    add.f32 %f0, %f0, %f13;

    // term_hi = (d*sc_hi*dot_hi - dmin*m_hi*dot_min_hi) * d8_hi
    mul.f32 %f15, %f1, %f4; mul.f32 %f15, %f15, %f10;
    mul.f32 %f16, %f2, %f6; mul.f32 %f16, %f16, %f12;
    sub.f32 %f15, %f15, %f16; mul.f32 %f15, %f15, %f8;
    add.f32 %f0, %f0, %f15;

    add.u32 %r12, %r12, 8;
    bra KLOOP;

    // -- Reduction (identical to kpar8) ----------------------------------------
REDUCE:
    // 3-step XOR butterfly within 8-lane mini-warp.
    // c=0x71F: segMask=7 (XOR stays in 8-lane group), maxLane=31.
    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 4, 0x71F, 0xFFFFFFFF;
    mov.b32 %f17, %r51; add.f32 %f0, %f0, %f17;

    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 2, 0x71F, 0xFFFFFFFF;
    mov.b32 %f17, %r51; add.f32 %f0, %f0, %f17;

    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 1, 0x71F, 0xFFFFFFFF;
    mov.b32 %f17, %r51; add.f32 %f0, %f0, %f17;

    // sub_l==0 writes mini-warp sum to smem[warp_id*4 + mw]
    mul.lo.u32 %r52, %r5, 4;
    add.u32    %r52, %r52, %r6;
    shl.b32    %r53, %r52, 2;
    mov.u32    %r54, smem_buf;
    add.u32    %r54, %r54, %r53;
    @%p2 st.shared.f32 [%r54], %f0;

    bar.sync 0;

    // warp 0, lanes 0-3: cross-warp sum per j-group
    setp.eq.u32 %p6, %r5, 0;
    setp.lt.u32 %p7, %r4, 4;
    and.pred    %p8, %p6, %p7;
    mul.lo.u32 %r55, %r4, 4;
    mov.u32    %r54, smem_buf;
    add.u32    %r55, %r54, %r55;    // smem_buf + j*4

    mov.f32    %f18, 0f00000000;
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

    // thread 0 sums smem[0..3] and writes out[row_idx * n_rows + n_out]
    setp.eq.u32 %p9, %r3, 0;
    mov.u32    %r53, smem_buf;
    mov.f32    %f18, 0f00000000;
    @%p9 ld.shared.f32 %f18, [%r53 +  0];
    @%p9 ld.shared.f32 %f19, [%r53 +  4]; @%p9 add.f32 %f18, %f18, %f19;
    @%p9 ld.shared.f32 %f19, [%r53 +  8]; @%p9 add.f32 %f18, %f18, %f19;
    @%p9 ld.shared.f32 %f19, [%r53 + 12]; @%p9 add.f32 %f18, %f18, %f19;

    // out_byte_offset = (row_idx * n_rows + n_out) * 4
    mad.lo.u32 %r61, %r58, %r0, %r2;  // row_idx * n_rows + n_out
    shl.b32    %r61, %r61, 2;          // * 4 bytes
    cvt.u64.u32 %rd4, %r61;
    add.u64    %rd4, %rd3, %rd4;
    @%p9 st.global.f32 [%rd4], %f18;

DONE:
    ret;
}
"#;

/// Kernel entry name for `cuModuleGetFunction`.
pub const ENTRY: &str = "q4k_gemv_sm120_batch";

/// Minimum compute capability (sm_120 = Blackwell).
pub const MIN_CC: u32 = 1200;
