/// Hand-tuned PTX kernel for Q4K x Q8_1 GEMV decode, targeting sm_120 (Blackwell RTX 5090).
///
/// # Current implementation: repacked 42-u32 format (same layout as v3_kpar)
///
/// This kernel reads Q4K weights in the REPACKED 42-u32 format (168 bytes/block)
/// which is identical to what v3_kpar reads. The key benefit over v3_kpar is direct
/// sm_120 codegen: optimal register allocation, instruction scheduling, and no JIT
/// overhead (v3_kpar is sm_80 PTX JIT'd to sm_120 by the driver).
///
/// # Why not native 36-u32 format?
///
/// The production engine dispatches K/V projections via `dispatch_kv_on_stream` which
/// calls the basic f32 GEMV kernel. That kernel reads repacked 42-u32. If we used
/// native 36-u32 format, K/V GEMV would read the wrong memory layout, producing NaN
/// values that crash the graph replay (CUDA error 719). Proper native-format support
/// requires per-slot output tensors for K/V — a larger engine refactor tracked as
/// a future optimization.
///
/// # Repacked Q4K block layout (42 u32 = 168 bytes per block of 256 elements)
///
/// ```text
/// u32[0]:   d    (f32 bits, was f16 in GGUF, converted at upload)
/// u32[1]:   dmin (f32 bits)
/// u32[2+j]: sc[2j] | sc[2j+1]<<8  for j=0..3  (8-bit sub-block scale pairs)
/// u32[6+j]: m[2j] | m[2j+1]<<8   for j=0..3  (8-bit sub-block min pairs)
/// u32[10..42]: qs data (128 bytes = 256 nibbles, 4-bit each)
/// ```
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
/// Grid:  (n_rows, 1, 1) — one block per output row
/// Block: (256, 1, 1)    — 8 warps x 32 threads
/// Smem:  128 bytes      — 32 f32 slots (8 warps x 4 j-groups)
///
/// # Mini-warp structure (same as v3_kpar)
///
/// mw = lane/8 (0..3, j-group index), sub_l = lane%8 (0..7)
/// All 32 lanes execute the same instructions (no divergence), using
/// mw-dependent index computations for scale/qs access.
///
/// # K-parallelism
///
/// 8 warps stride by 8 blocks per iteration (VDR=1 for simplicity).
/// With k_blocks=16 (dim=4096), each warp handles exactly 2 blocks.
///
/// # Kernel parameters
///
/// ```text
/// param_weights:  *const u32  -- repacked Q4K blocks [n_rows * k_blocks * 42]
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
// Repacked Q4K (42 u32/block) x Q8_1 GEMV decode.
// sm_120 (Blackwell) hand-tuned PTX -- direct codegen, no JIT.
//
// Reads the same repacked 42-u32 block format as v3_kpar (oxide sm_80 kernel).
// Advantage: direct sm_120 codegen avoids driver JIT overhead and enables
// optimal register allocation and instruction scheduling for Blackwell.
//
// Scale extraction is SIMPLER than native format: scales are already 8-bit
// expanded (not 6-bit packed), with NO branch divergence in this kernel.
//
// Block geometry: 256 threads = 8 warps.
//   lane    = tid & 31          (0..31)
//   warp_id = tid >> 5          (0..7)
//   mw      = lane >> 3         (0..3, j-group index)
//   sub_l   = lane & 7          (0..7)
//
// Shared memory: 32 x 4 = 128 bytes (warp_id * 4 + mw -> partial sum).
// K-parallelism: warp w processes k-blocks w, w+8, w+16, ... (stride 8).
.visible .entry q4k_gemv_decode_v4_kpar(
    .param .u64 param_weights,
    .param .u64 param_act_qs,
    .param .u64 param_act_ds,
    .param .u64 param_out,
    .param .u32 param_n_rows,
    .param .u32 param_k_blocks
)
{
    .reg .u32  %r<56>;
    .reg .u64  %rd<10>;
    .reg .f32  %f<20>;
    .reg .pred %p<10>;
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

    // base_ptr = weights + (n_out * k_blocks + blk) * 168 bytes (42 u32 * 4)
    mad.lo.u32 %r14, %r2, %r1, %r12;  // n_out * k_blocks + blk
    mul.lo.u32 %r15, %r14, 168;        // * 168 bytes/block
    cvt.u64.u32 %rd4, %r15;
    add.u64 %rd4, %rd0, %rd4;          // %rd4 = base_ptr

    // -- Load d and dmin (f32 bits stored directly in repacked format) ----------
    // u32[0] = d as f32 bits, u32[1] = dmin as f32 bits. All lanes load same
    // address (L1 broadcast within warp).
    ld.global.b32 %r16, [%rd4];        // d bits (f32)
    ld.global.b32 %r17, [%rd4 + 4];   // dmin bits (f32)
    mov.b32 %f1, %r16;                  // d as f32
    mov.b32 %f2, %r17;                  // dmin as f32

    // -- Load scale pair for this j-group (mw) ---------------------------------
    // sc_pair = u32[2 + mw] = sc_lo | sc_hi<<8  (8-bit each)
    // m_pair  = u32[6 + mw] = m_lo  | m_hi<<8
    // Offset for sc_pair: (2 + mw) * 4 = 8 + mw*4 bytes
    shl.b32    %r18, %r6, 2;           // mw * 4 bytes
    add.u32    %r18, %r18, 8;          // + 8 bytes (skip u32[0,1] = d,dmin)
    cvt.u64.u32 %rd5, %r18;
    add.u64    %rd5, %rd4, %rd5;
    ld.global.b32 %r19, [%rd5];        // sc_pair: sc_lo | sc_hi<<8

    add.u32    %r18, %r18, 16;         // + 16 bytes (from sc to m section: 4*4 bytes)
    cvt.u64.u32 %rd6, %r18;
    add.u64    %rd6, %rd4, %rd6;
    ld.global.b32 %r20, [%rd6];        // m_pair: m_lo | m_hi<<8

    // Extract 8-bit scales and convert to f32 (no 6-bit unpacking needed!)
    and.b32 %r21, %r19, 0xFF;          // sc_lo (u32)
    shr.u32 %r22, %r19, 8;
    and.b32 %r22, %r22, 0xFF;          // sc_hi (u32)
    and.b32 %r23, %r20, 0xFF;          // m_lo (u32)
    shr.u32 %r24, %r20, 8;
    and.b32 %r24, %r24, 0xFF;          // m_hi (u32)

    cvt.rn.f32.u32 %f3, %r21;         // sc_lo as f32
    cvt.rn.f32.u32 %f4, %r22;         // sc_hi as f32
    cvt.rn.f32.u32 %f5, %r23;         // m_lo as f32
    cvt.rn.f32.u32 %f6, %r24;         // m_hi as f32

    // -- Load qs word (lane-specific) ------------------------------------------
    // qs for j-group mw, sub_l: u32[10 + mw*8 + sub_l]
    // Byte offset = (10 + mw*8 + sub_l) * 4 = 40 + mw*32 + sub_l*4
    mul.lo.u32 %r25, %r6, 8;           // mw * 8
    add.u32    %r25, %r25, 10;          // + 10 (skip header: d,dmin,sc*4,m*4)
    add.u32    %r25, %r25, %r7;         // + sub_l
    shl.b32    %r25, %r25, 2;           // * 4 bytes
    cvt.u64.u32 %rd7, %r25;
    add.u64    %rd7, %rd4, %rd7;
    ld.global.b32 %r26, [%rd7];        // qs word (4 nibble pairs)

    // -- Load Q8_1 activation data ---------------------------------------------
    // q8_blk_lo = blk * 8 + mw * 2   (Q8_1 block index for lo sub-group)
    mul.lo.u32 %r27, %r12, 8;          // blk * 8
    mul.lo.u32 %r28, %r6, 2;           // mw * 2
    add.u32    %r27, %r27, %r28;        // q8_blk_lo

    // u_lo = act_qs[q8_blk_lo * 8 + sub_l]
    mul.lo.u32 %r28, %r27, 8;          // q8_blk_lo * 8
    add.u32    %r28, %r28, %r7;         // + sub_l
    shl.b32    %r29, %r28, 2;           // * 4 bytes
    cvt.u64.u32 %rd8, %r29;
    add.u64    %rd8, %rd1, %rd8;
    ld.global.b32 %r29, [%rd8];        // u_lo (4 packed i8)

    // u_hi = act_qs[(q8_blk_lo+1) * 8 + sub_l]
    add.u32    %r30, %r28, 8;           // +8 to the qs index
    shl.b32    %r31, %r30, 2;           // * 4 bytes
    cvt.u64.u32 %rd9, %r31;
    add.u64    %rd9, %rd1, %rd9;
    ld.global.b32 %r30, [%rd9];        // u_hi (4 packed i8)

    // d8_lo = act_ds[q8_blk_lo]  (f16 in low 16 bits of u32)
    shl.b32    %r31, %r27, 2;           // q8_blk_lo * 4 bytes
    cvt.u64.u32 %rd5, %r31;
    add.u64    %rd5, %rd2, %rd5;
    ld.global.b16 %h2, [%rd5];         // f16 d8_lo
    cvt.f32.f16 %f7, %h2;              // d8_lo as f32

    // d8_hi = act_ds[q8_blk_lo + 1]
    add.u64    %rd6, %rd5, 4;           // +4 bytes
    ld.global.b16 %h3, [%rd6];         // f16 d8_hi
    cvt.f32.f16 %f8, %h3;              // d8_hi as f32

    // -- dp4a: Q4K nibbles x Q8_1 bytes ----------------------------------------
    // v_lo = qs & 0x0F0F0F0F  (lo nibbles, treated as s8 since 0-15 >= 0)
    and.b32 %r32, %r26, 0x0F0F0F0F;
    // v_hi = (qs >> 4) & 0x0F0F0F0F  (hi nibbles)
    shr.u32 %r33, %r26, 4;
    and.b32 %r33, %r33, 0x0F0F0F0F;

    // dot_lo  = sum(v_lo[i] * u_lo[i]) for i=0..3
    dp4a.s32.s32 %r34, %r32, %r29, 0;
    // dot_hi  = sum(v_hi[i] * u_hi[i])
    dp4a.s32.s32 %r35, %r33, %r30, 0;
    // dot_min_lo = sum(1 * u_lo[i]) = sum(u_lo[i])
    dp4a.s32.s32 %r36, 0x01010101, %r29, 0;
    // dot_min_hi = sum(1 * u_hi[i])
    dp4a.s32.s32 %r37, 0x01010101, %r30, 0;

    // -- Accumulate -----------------------------------------------------------
    // acc += d8_lo * (d * sc_lo * dot_lo - dmin * m_lo * dot_min_lo)
    //      + d8_hi * (d * sc_hi * dot_hi - dmin * m_hi * dot_min_hi)
    cvt.rn.f32.s32 %f9,  %r34;         // dot_lo as f32
    cvt.rn.f32.s32 %f10, %r35;         // dot_hi as f32
    cvt.rn.f32.s32 %f11, %r36;         // dot_min_lo as f32
    cvt.rn.f32.s32 %f12, %r37;         // dot_min_hi as f32

    // term_lo = d * sc_lo * dot_lo - dmin * m_lo * dot_min_lo
    mul.f32 %f13, %f1, %f3;            // d * sc_lo
    mul.f32 %f13, %f13, %f9;           // * dot_lo
    mul.f32 %f14, %f2, %f5;            // dmin * m_lo
    mul.f32 %f14, %f14, %f11;          // * dot_min_lo
    sub.f32 %f13, %f13, %f14;          // term_lo
    mul.f32 %f13, %f13, %f7;           // * d8_lo
    add.f32 %f0, %f0, %f13;            // acc += term_lo * d8_lo

    // term_hi = d * sc_hi * dot_hi - dmin * m_hi * dot_min_hi
    mul.f32 %f15, %f1, %f4;            // d * sc_hi
    mul.f32 %f15, %f15, %f10;          // * dot_hi
    mul.f32 %f16, %f2, %f6;            // dmin * m_hi
    mul.f32 %f16, %f16, %f12;          // * dot_min_hi
    sub.f32 %f15, %f15, %f16;          // term_hi
    mul.f32 %f15, %f15, %f8;           // * d8_hi
    add.f32 %f0, %f0, %f15;            // acc += term_hi * d8_hi

    // -- Advance and loop ------------------------------------------------------
    add.u32 %r12, %r12, 8;             // blk += 8 (stride = WARPS_PER_BLOCK_KPAR)
    bra KLOOP;

    // -- Reduction -------------------------------------------------------------
REDUCE:
    // Step 1: 3-step XOR butterfly within 8-lane mini-warp (sub_l 0..7).
    // c = 0x71F: segMask=7 (bits[12:8]), maxLane=31 (bits[4:0]).
    //   segMask=7: XOR stays within each 8-lane group (0-7, 8-15, 16-23, 24-31).
    //   maxLane=31: no clamping (all warp lanes valid as sources).
    //   member_mask=0xFFFFFFFF: all 32 lanes participate (same mask required).
    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 4, 0x71F, 0xFFFFFFFF;
    mov.b32 %f17, %r51;
    add.f32 %f0, %f0, %f17;

    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 2, 0x71F, 0xFFFFFFFF;
    mov.b32 %f17, %r51;
    add.f32 %f0, %f0, %f17;

    mov.b32 %r50, %f0;
    shfl.sync.bfly.b32 %r51, %r50, 1, 0x71F, 0xFFFFFFFF;
    mov.b32 %f17, %r51;
    add.f32 %f0, %f0, %f17;
    // After 3 steps: ALL lanes in each mini-warp hold the mini-warp sum.

    // Step 2: sub_l==0 writes to smem[warp_id*4 + mw]
    mul.lo.u32 %r52, %r5, 4;           // warp_id * 4
    add.u32    %r52, %r52, %r6;         // + mw  ->  slot (0..31)
    shl.b32    %r53, %r52, 2;           // * 4 bytes
    mov.u32    %r54, smem_buf;          // smem base (32-bit shared mem ptr)
    add.u32    %r54, %r54, %r53;        // smem slot address
    @%p2 st.shared.f32 [%r54], %f0;    // only sub_l==0 writes

    bar.sync 0;  // all 256 threads sync -- smem is now fully written

    // Step 3: warp 0 lanes 0-3 sum all 8 warps' contributions per j-group.
    setp.eq.u32 %p6, %r5, 0;           // warp_id == 0
    setp.lt.u32 %p7, %r4, 4;           // lane < 4
    and.pred    %p8, %p6, %p7;         // warp_id==0 && lane<4

    // j-base: smem_buf + j*4 where j = lane (0..3)
    mul.lo.u32 %r55, %r4, 4;           // lane * 4 bytes (= j * 4)
    mov.u32    %r54, smem_buf;
    add.u32    %r55, %r54, %r55;        // smem_buf + j*4

    // Unrolled 8-warp sum (stride = 16 bytes = 4 f32 slots per warp).
    mov.f32    %f18, 0f00000000;
    @%p8 ld.shared.f32 %f19, [%r55 +   0]; @%p8 add.f32 %f18, %f18, %f19; // w=0
    @%p8 ld.shared.f32 %f19, [%r55 +  16]; @%p8 add.f32 %f18, %f18, %f19; // w=1
    @%p8 ld.shared.f32 %f19, [%r55 +  32]; @%p8 add.f32 %f18, %f18, %f19; // w=2
    @%p8 ld.shared.f32 %f19, [%r55 +  48]; @%p8 add.f32 %f18, %f18, %f19; // w=3
    @%p8 ld.shared.f32 %f19, [%r55 +  64]; @%p8 add.f32 %f18, %f18, %f19; // w=4
    @%p8 ld.shared.f32 %f19, [%r55 +  80]; @%p8 add.f32 %f18, %f18, %f19; // w=5
    @%p8 ld.shared.f32 %f19, [%r55 +  96]; @%p8 add.f32 %f18, %f18, %f19; // w=6
    @%p8 ld.shared.f32 %f19, [%r55 + 112]; @%p8 add.f32 %f18, %f18, %f19; // w=7

    // Write j-group total back to smem[j] (first 4 slots, stride=4).
    @%p8 st.shared.f32 [%r55], %f18;

    bar.sync 0;  // second sync -- smem[0..3] now hold j-group totals

    // Step 4: thread 0 sums smem[0..3] and writes output.
    setp.eq.u32 %p9, %r3, 0;           // tid == 0
    mov.u32    %r53, smem_buf;          // smem base
    mov.f32    %f18, 0f00000000;
    @%p9 ld.shared.f32 %f18, [%r53 +  0]; // j=0
    @%p9 ld.shared.f32 %f19, [%r53 +  4]; // j=1
    @%p9 add.f32 %f18, %f18, %f19;
    @%p9 ld.shared.f32 %f19, [%r53 +  8]; // j=2
    @%p9 add.f32 %f18, %f18, %f19;
    @%p9 ld.shared.f32 %f19, [%r53 + 12]; // j=3
    @%p9 add.f32 %f18, %f18, %f19;

    // out[n_out] = final_sum
    shl.b32    %r50, %r2, 2;            // n_out * 4 bytes
    cvt.u64.u32 %rd4, %r50;
    add.u64    %rd4, %rd3, %rd4;        // &out[n_out]
    @%p9 st.global.f32 [%rd4], %f18;

DONE:
    ret;
}
"#;

/// Kernel entry name -- used by `cuModuleGetFunction`.
pub const ENTRY: &str = "q4k_gemv_decode_v4_kpar";

/// Minimum compute capability required (sm_120 = Blackwell).
pub const MIN_CC: u32 = 1200;
