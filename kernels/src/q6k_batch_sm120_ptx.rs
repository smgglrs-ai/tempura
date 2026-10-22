/// sm_120 PTX batch kernel for Q6K x f32 GEMV decode: M rows share one weight read.
///
/// # Purpose
///
/// Q6K down_proj [4096x14336] runs M times sequentially in the current batch decode,
/// reading weights M times. This kernel reads weights ONCE and computes M dot products.
///
/// # Q6K block format (53 u32 = 212 bytes/block -- same as q6k_gemv_v4_ptx)
///
/// ```text
/// u32[0]:      d (f32 bits) -- super-scale
/// u32[1..5]:   16 group scales as bias-128 u8 (4 per u32)
/// u32[5..37]:  ql[0..128] -- lower 4 bits of each weight
/// u32[37..53]: qh[0..64]  -- upper 2 bits of each weight
/// ```
///
/// # Launch geometry (M=2 hardcoded)
///
/// Grid:  (n_rows, 1, 1) -- one block per output neuron
/// Block: (256, 1, 1)    -- 8 warps x 32 threads
/// Smem:  3456 bytes (static):
///   [0..31]:    row 0 -- 8 f32 partial sums for cross-warp reduction
///   [32..63]:   row 1 -- 8 f32 partial sums for cross-warp reduction
///   [64..3455]: 8 warps x (buf_A[212] + buf_B[212]) = 3392 bytes
///
/// # Weight amortization
///
/// Each block loads Q6K weight block ONCE via cp.async double-buffering.
/// For each of 8 dequant steps:
///   1. Compute weight_contribution = d * sc * (q6 - 32)
///   2. Load x_row0[i], FMA into acc0
///   3. Load x_row1[i] (same index, row 1 base = x_ptr + k_blocks*256*4), FMA into acc1
///
/// # Activation layout
///
/// param_x: [M * k_blocks * 256] f32, row 0 then row 1 (contiguous).
///
/// # Output layout
///
/// param_out: [M * n_rows] f32 -- row 0 at [0..n-1], row 1 at [n..2n-1].
pub const PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// q6k_gemv_sm120_batch (M=2)
// Q6K x f32 GEMV decode, two activation rows sharing one weight read per block.
// Grid: (n_rows, 1, 1). Block: (256, 1, 1) = 8 warps.
// cp.async double-buffering: same as q6k_gemv_sm120_mbar_kpar8.
// Two accumulators (acc0, acc1); weight_contribution computed once per step.
// Smem layout (3456 bytes static):
//   [0..31]:    row 0 partial sums (8 warps x 4 bytes)
//   [32..63]:   row 1 partial sums (8 warps x 4 bytes)
//   [64..3455]: 8 warps x (buf_A[212] + buf_B[212]) = 3392 bytes
.visible .entry q6k_gemv_sm120_batch(
    .param .u64 param_out,
    .param .u64 param_out_len,
    .param .u64 param_x,
    .param .u64 param_x_len,
    .param .u64 param_packed,
    .param .u64 param_packed_len,
    .param .u32 param_n,
    .param .u32 param_k_blocks
)
{
    .reg .u32  %r<82>;
    .reg .u64  %rd<13>;
    .reg .f32  %f<16>;
    .reg .pred %p<8>;

    // smem: 64 bytes reduction (2 rows x 8 warps x 4 bytes)
    //     + 8 warps x (buf_A[212] + buf_B[212]) = 3392 bytes
    // Total: 3456 bytes.
    .shared .align 4 .b8 smem_buf[3456];

    // -- Load parameters --------------------------------------------------------
    ld.param.u64 %rd0,  [param_out];
    ld.param.u64 %rd1,  [param_x];
    ld.param.u64 %rd2,  [param_packed];
    ld.param.u32 %r0,   [param_n];
    ld.param.u32 %r1,   [param_k_blocks];

    // -- Thread decomposition ---------------------------------------------------
    mov.u32 %r2, %ctaid.x;          // n_out = blockIdx.x
    setp.ge.u32 %p0, %r2, %r0;
    @%p0 bra DONE;

    mov.u32 %r3, %tid.x;            // tid  (0..255)
    and.b32 %r4, %r3, 31;           // lane = tid & 31
    shr.u32 %r5, %r3, 5;            // warp_id = tid >> 5  (0..7)

    // byte_shift = (lane % 4) * 8
    and.b32 %r6, %r4, 3;
    shl.b32 %r6, %r6, 3;

    // lane_div4 = lane >> 2
    shr.u32 %r7, %r4, 2;

    // sc_shift_even = (lane>>4)*8; sc_shift_odd = sc_shift_even + 16
    shr.u32 %r8, %r4, 4;
    shl.b32 %r8, %r8, 3;
    add.u32 %r9, %r8, 16;

    // Predicate for round-2 of cp.async: lanes 0..20 issue 2 copies, 21..31 issue 1.
    setp.lt.u32 %p1, %r4, 21;

    // -- Accumulators -----------------------------------------------------------
    mov.f32 %f0, 0f00000000;        // acc0 = 0.0 (row 0)
    mov.f32 %f12, 0f00000000;       // acc1 = 0.0 (row 1)

    // -- Constants --------------------------------------------------------------
    mov.f32 %f10, 0f43000000;       // 128.0
    mov.f32 %f11, 0f42000000;       //  32.0

    // -- Row 1 x stride: k_blocks * 256 * 4 bytes (rd12 = byte offset to row 1 x)
    mul.lo.u32 %r81, %r1, 1024;     // k_blocks * 1024
    cvt.u64.u32 %rd12, %r81;        // rd12 = row1 x stride in bytes

    // -- Per-warp smem buffer base (double buffers start at offset 64) ----------
    mul.lo.u32 %r10, %r5, 424;
    add.u32    %r10, %r10, 64;      // curr smem base (buf_A for this warp)
    add.u32    %r11, %r10, 212;     // next smem base (buf_B for this warp)
    mov.u32    %r20, %r10;
    mov.u32    %r21, %r11;

    // -- Prologue: cp.async block 0 into curr (buf_A) --------------------------
    setp.ge.u32 %p2, %r5, %r1;
    @%p2 bra REDUCE;

    mad.lo.u32 %r22, %r2, %r1, %r5; // n_out*k_blocks + warp_id
    mul.lo.u32 %r23, %r22, 212;      // byte offset to block 0
    cvt.u64.u32 %rd3, %r23;
    add.u64 %rd3, %rd2, %rd3;        // rd3 = ptr to block 0

    // Round 1: all 32 lanes load one u32 each.
    shl.b32 %r24, %r4, 2;            // lane*4
    add.u32 %r25, %r20, %r24;
    cvt.u64.u32 %rd4, %r24;
    add.u64 %rd4, %rd3, %rd4;
    cp.async.ca.shared.global [%r25], [%rd4], 4;

    // Round 2: lanes 0..20 load one more u32 each.
    add.u32 %r26, %r4, 32;
    shl.b32 %r26, %r26, 2;
    add.u32 %r27, %r20, %r26;
    cvt.u64.u32 %rd4, %r26;
    add.u64 %rd4, %rd3, %rd4;
    @%p1 cp.async.ca.shared.global [%r27], [%rd4], 4;

    cp.async.commit_group;
    cp.async.wait_group 0;

    // -- K-loop: blk = warp_id, warp_id+8, ...  --------------------------------
    mov.u32 %r30, %r5;               // blk = warp_id

KLOOP:
    setp.ge.u32 %p0, %r30, %r1;     // blk >= k_blocks?
    @%p0 bra REDUCE;

    // Issue async-load of block (blk+8) into next smem buffer.
    add.u32 %r31, %r30, 8;           // next_blk = blk + 8
    setp.lt.u32 %p3, %r31, %r1;     // next_blk < k_blocks?
    @!%p3 bra SKIP_LOAD;

    mad.lo.u32 %r32, %r2, %r1, %r31; // n_out*k_blocks + next_blk
    mul.lo.u32 %r33, %r32, 212;
    cvt.u64.u32 %rd5, %r33;
    add.u64 %rd5, %rd2, %rd5;

    // Round 1: all 32 lanes.
    shl.b32 %r34, %r4, 2;
    add.u32 %r35, %r21, %r34;
    cvt.u64.u32 %rd6, %r34;
    add.u64 %rd6, %rd5, %rd6;
    cp.async.ca.shared.global [%r35], [%rd6], 4;

    // Round 2: lanes 0..20.
    add.u32 %r36, %r4, 32;
    shl.b32 %r36, %r36, 2;
    add.u32 %r37, %r21, %r36;
    cvt.u64.u32 %rd6, %r36;
    add.u64 %rd6, %rd5, %rd6;
    @%p1 cp.async.ca.shared.global [%r37], [%rd6], 4;

    cp.async.commit_group;

SKIP_LOAD:
    // -- Read Q6K weight block from curr smem -----------------------------------
    ld.shared.b32 %r40, [%r20 + 0];
    mov.b32 %f1, %r40;               // d (super-scale, f32 bits)

    ld.shared.b32 %r41, [%r20 +  4]; // sc_01
    ld.shared.b32 %r42, [%r20 +  8]; // sc_23
    ld.shared.b32 %r43, [%r20 + 12]; // sc_45
    ld.shared.b32 %r44, [%r20 + 16]; // sc_67

    // ql_a = smem_u32[5 + lane_div4] >> byte_shift & 0xFF  (steps 0,2)
    add.u32 %r45, %r7, 5;
    shl.b32 %r45, %r45, 2;
    add.u32 %r45, %r20, %r45;
    ld.shared.b32 %r45, [%r45];
    shr.u32 %r45, %r45, %r6;
    and.b32 %r45, %r45, 255;

    // ql_b = smem_u32[13 + lane_div4] >> byte_shift & 0xFF  (steps 1,3)
    add.u32 %r46, %r7, 13;
    shl.b32 %r46, %r46, 2;
    add.u32 %r46, %r20, %r46;
    ld.shared.b32 %r46, [%r46];
    shr.u32 %r46, %r46, %r6;
    and.b32 %r46, %r46, 255;

    // ql_c = smem_u32[21 + lane_div4] >> byte_shift & 0xFF  (steps 4,6)
    add.u32 %r47, %r7, 21;
    shl.b32 %r47, %r47, 2;
    add.u32 %r47, %r20, %r47;
    ld.shared.b32 %r47, [%r47];
    shr.u32 %r47, %r47, %r6;
    and.b32 %r47, %r47, 255;

    // ql_d = smem_u32[29 + lane_div4] >> byte_shift & 0xFF  (steps 5,7)
    add.u32 %r48, %r7, 29;
    shl.b32 %r48, %r48, 2;
    add.u32 %r48, %r20, %r48;
    ld.shared.b32 %r48, [%r48];
    shr.u32 %r48, %r48, %r6;
    and.b32 %r48, %r48, 255;

    // qh_0 = smem_u32[37 + lane_div4] >> byte_shift & 0xFF  (steps 0-3)
    add.u32 %r49, %r7, 37;
    shl.b32 %r49, %r49, 2;
    add.u32 %r49, %r20, %r49;
    ld.shared.b32 %r49, [%r49];
    shr.u32 %r49, %r49, %r6;
    and.b32 %r49, %r49, 255;

    // qh_1 = smem_u32[45 + lane_div4] >> byte_shift & 0xFF  (steps 4-7)
    add.u32 %r50, %r7, 45;
    shl.b32 %r50, %r50, 2;
    add.u32 %r50, %r20, %r50;
    ld.shared.b32 %r50, [%r50];
    shr.u32 %r50, %r50, %r6;
    and.b32 %r50, %r50, 255;

    // -- 8-step dequant + FMA for both rows ------------------------------------
    // For each step: compute f5 = d*sc*(q6-32) once, then FMA into acc0 and acc1.
    // Row 0 x addr: rd1 + offset; Row 1 x addr: rd7 + rd12 (= rd1 + offset + stride).
    mul.lo.u32 %r60, %r30, 256;     // blk * 256

    // Step 0: within=lane, sc=sc_01>>sc_shift_even, lo=ql_a&0xF, hi=qh_0&0x3
    shr.u32 %r61, %r41, %r8;
    and.b32 %r61, %r61, 255;
    cvt.rn.f32.u32 %f2, %r61;
    sub.f32 %f2, %f2, %f10;
    and.b32 %r62, %r45, 15;
    and.b32 %r63, %r49, 3;
    shl.b32 %r63, %r63, 4;
    or.b32  %r62, %r62, %r63;
    cvt.rn.f32.u32 %f3, %r62;
    sub.f32 %f3, %f3, %f11;
    add.u32 %r64, %r60, %r4;
    shl.b32 %r64, %r64, 2;
    cvt.u64.u32 %rd7, %r64;
    add.u64 %rd7, %rd1, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd12;
    ld.global.nc.f32 %f13, [%rd9];
    fma.rn.f32 %f12, %f5, %f13, %f12;

    // Step 1: within=32+lane, sc=sc_01>>sc_shift_odd, lo=ql_b&0xF, hi=(qh_0>>2)&0x3
    shr.u32 %r61, %r41, %r9;
    and.b32 %r61, %r61, 255;
    cvt.rn.f32.u32 %f2, %r61;
    sub.f32 %f2, %f2, %f10;
    and.b32 %r62, %r46, 15;
    shr.u32 %r63, %r49, 2;
    and.b32 %r63, %r63, 3;
    shl.b32 %r63, %r63, 4;
    or.b32  %r62, %r62, %r63;
    cvt.rn.f32.u32 %f3, %r62;
    sub.f32 %f3, %f3, %f11;
    add.u32 %r64, %r60, 32;
    add.u32 %r64, %r64, %r4;
    shl.b32 %r64, %r64, 2;
    cvt.u64.u32 %rd7, %r64;
    add.u64 %rd7, %rd1, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd12;
    ld.global.nc.f32 %f13, [%rd9];
    fma.rn.f32 %f12, %f5, %f13, %f12;

    // Step 2: within=64+lane, sc=sc_23>>sc_shift_even, lo=(ql_a>>4)&0xF, hi=(qh_0>>4)&0x3
    shr.u32 %r61, %r42, %r8;
    and.b32 %r61, %r61, 255;
    cvt.rn.f32.u32 %f2, %r61;
    sub.f32 %f2, %f2, %f10;
    shr.u32 %r62, %r45, 4;
    and.b32 %r62, %r62, 15;
    shr.u32 %r63, %r49, 4;
    and.b32 %r63, %r63, 3;
    shl.b32 %r63, %r63, 4;
    or.b32  %r62, %r62, %r63;
    cvt.rn.f32.u32 %f3, %r62;
    sub.f32 %f3, %f3, %f11;
    add.u32 %r64, %r60, 64;
    add.u32 %r64, %r64, %r4;
    shl.b32 %r64, %r64, 2;
    cvt.u64.u32 %rd7, %r64;
    add.u64 %rd7, %rd1, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd12;
    ld.global.nc.f32 %f13, [%rd9];
    fma.rn.f32 %f12, %f5, %f13, %f12;

    // Step 3: within=96+lane, sc=sc_23>>sc_shift_odd, lo=(ql_b>>4)&0xF, hi=(qh_0>>6)&0x3
    shr.u32 %r61, %r42, %r9;
    and.b32 %r61, %r61, 255;
    cvt.rn.f32.u32 %f2, %r61;
    sub.f32 %f2, %f2, %f10;
    shr.u32 %r62, %r46, 4;
    and.b32 %r62, %r62, 15;
    shr.u32 %r63, %r49, 6;
    and.b32 %r63, %r63, 3;
    shl.b32 %r63, %r63, 4;
    or.b32  %r62, %r62, %r63;
    cvt.rn.f32.u32 %f3, %r62;
    sub.f32 %f3, %f3, %f11;
    add.u32 %r64, %r60, 96;
    add.u32 %r64, %r64, %r4;
    shl.b32 %r64, %r64, 2;
    cvt.u64.u32 %rd7, %r64;
    add.u64 %rd7, %rd1, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd12;
    ld.global.nc.f32 %f13, [%rd9];
    fma.rn.f32 %f12, %f5, %f13, %f12;

    // Step 4: within=128+lane, sc=sc_45>>sc_shift_even, lo=ql_c&0xF, hi=qh_1&0x3
    shr.u32 %r61, %r43, %r8;
    and.b32 %r61, %r61, 255;
    cvt.rn.f32.u32 %f2, %r61;
    sub.f32 %f2, %f2, %f10;
    and.b32 %r62, %r47, 15;
    and.b32 %r63, %r50, 3;
    shl.b32 %r63, %r63, 4;
    or.b32  %r62, %r62, %r63;
    cvt.rn.f32.u32 %f3, %r62;
    sub.f32 %f3, %f3, %f11;
    add.u32 %r64, %r60, 128;
    add.u32 %r64, %r64, %r4;
    shl.b32 %r64, %r64, 2;
    cvt.u64.u32 %rd7, %r64;
    add.u64 %rd7, %rd1, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd12;
    ld.global.nc.f32 %f13, [%rd9];
    fma.rn.f32 %f12, %f5, %f13, %f12;

    // Step 5: within=160+lane, sc=sc_45>>sc_shift_odd, lo=ql_d&0xF, hi=(qh_1>>2)&0x3
    shr.u32 %r61, %r43, %r9;
    and.b32 %r61, %r61, 255;
    cvt.rn.f32.u32 %f2, %r61;
    sub.f32 %f2, %f2, %f10;
    and.b32 %r62, %r48, 15;
    shr.u32 %r63, %r50, 2;
    and.b32 %r63, %r63, 3;
    shl.b32 %r63, %r63, 4;
    or.b32  %r62, %r62, %r63;
    cvt.rn.f32.u32 %f3, %r62;
    sub.f32 %f3, %f3, %f11;
    add.u32 %r64, %r60, 160;
    add.u32 %r64, %r64, %r4;
    shl.b32 %r64, %r64, 2;
    cvt.u64.u32 %rd7, %r64;
    add.u64 %rd7, %rd1, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd12;
    ld.global.nc.f32 %f13, [%rd9];
    fma.rn.f32 %f12, %f5, %f13, %f12;

    // Step 6: within=192+lane, sc=sc_67>>sc_shift_even, lo=(ql_c>>4)&0xF, hi=(qh_1>>4)&0x3
    shr.u32 %r61, %r44, %r8;
    and.b32 %r61, %r61, 255;
    cvt.rn.f32.u32 %f2, %r61;
    sub.f32 %f2, %f2, %f10;
    shr.u32 %r62, %r47, 4;
    and.b32 %r62, %r62, 15;
    shr.u32 %r63, %r50, 4;
    and.b32 %r63, %r63, 3;
    shl.b32 %r63, %r63, 4;
    or.b32  %r62, %r62, %r63;
    cvt.rn.f32.u32 %f3, %r62;
    sub.f32 %f3, %f3, %f11;
    add.u32 %r64, %r60, 192;
    add.u32 %r64, %r64, %r4;
    shl.b32 %r64, %r64, 2;
    cvt.u64.u32 %rd7, %r64;
    add.u64 %rd7, %rd1, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd12;
    ld.global.nc.f32 %f13, [%rd9];
    fma.rn.f32 %f12, %f5, %f13, %f12;

    // Step 7: within=224+lane, sc=sc_67>>sc_shift_odd, lo=(ql_d>>4)&0xF, hi=(qh_1>>6)&0x3
    shr.u32 %r61, %r44, %r9;
    and.b32 %r61, %r61, 255;
    cvt.rn.f32.u32 %f2, %r61;
    sub.f32 %f2, %f2, %f10;
    shr.u32 %r62, %r48, 4;
    and.b32 %r62, %r62, 15;
    shr.u32 %r63, %r50, 6;
    and.b32 %r63, %r63, 3;
    shl.b32 %r63, %r63, 4;
    or.b32  %r62, %r62, %r63;
    cvt.rn.f32.u32 %f3, %r62;
    sub.f32 %f3, %f3, %f11;
    add.u32 %r64, %r60, 224;
    add.u32 %r64, %r64, %r4;
    shl.b32 %r64, %r64, 2;
    cvt.u64.u32 %rd7, %r64;
    add.u64 %rd7, %rd1, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd12;
    ld.global.nc.f32 %f13, [%rd9];
    fma.rn.f32 %f12, %f5, %f13, %f12;

    // -- Wait for next block (if prefetch was issued) ---------------------------
    @!%p3 bra NO_WAIT;
    cp.async.wait_group 0;
NO_WAIT:

    // -- Swap curr/next smem buffer pointers -----------------------------------
    mov.u32 %r70, %r20;
    mov.u32 %r20, %r21;
    mov.u32 %r21, %r70;

    add.u32 %r30, %r30, 8;
    bra KLOOP;

    // -- Intra-warp butterfly reduction: acc0 (5 steps, all 32 lanes) ----------
REDUCE:
    mov.b32 %r71, %f0;
    shfl.sync.bfly.b32 %r72, %r71, 16, 0x1F, 0xFFFFFFFF;
    mov.b32 %f6, %r72; add.f32 %f0, %f0, %f6;
    mov.b32 %r71, %f0;
    shfl.sync.bfly.b32 %r72, %r71, 8, 0x1F, 0xFFFFFFFF;
    mov.b32 %f6, %r72; add.f32 %f0, %f0, %f6;
    mov.b32 %r71, %f0;
    shfl.sync.bfly.b32 %r72, %r71, 4, 0x1F, 0xFFFFFFFF;
    mov.b32 %f6, %r72; add.f32 %f0, %f0, %f6;
    mov.b32 %r71, %f0;
    shfl.sync.bfly.b32 %r72, %r71, 2, 0x1F, 0xFFFFFFFF;
    mov.b32 %f6, %r72; add.f32 %f0, %f0, %f6;
    mov.b32 %r71, %f0;
    shfl.sync.bfly.b32 %r72, %r71, 1, 0x1F, 0xFFFFFFFF;
    mov.b32 %f6, %r72; add.f32 %f0, %f0, %f6;

    // -- Intra-warp butterfly reduction: acc1 (5 steps, all 32 lanes) ----------
    mov.b32 %r71, %f12;
    shfl.sync.bfly.b32 %r72, %r71, 16, 0x1F, 0xFFFFFFFF;
    mov.b32 %f6, %r72; add.f32 %f12, %f12, %f6;
    mov.b32 %r71, %f12;
    shfl.sync.bfly.b32 %r72, %r71, 8, 0x1F, 0xFFFFFFFF;
    mov.b32 %f6, %r72; add.f32 %f12, %f12, %f6;
    mov.b32 %r71, %f12;
    shfl.sync.bfly.b32 %r72, %r71, 4, 0x1F, 0xFFFFFFFF;
    mov.b32 %f6, %r72; add.f32 %f12, %f12, %f6;
    mov.b32 %r71, %f12;
    shfl.sync.bfly.b32 %r72, %r71, 2, 0x1F, 0xFFFFFFFF;
    mov.b32 %f6, %r72; add.f32 %f12, %f12, %f6;
    mov.b32 %r71, %f12;
    shfl.sync.bfly.b32 %r72, %r71, 1, 0x1F, 0xFFFFFFFF;
    mov.b32 %f6, %r72; add.f32 %f12, %f12, %f6;
    // lane 0 of each warp holds its partial row0 sum in %f0, row1 sum in %f12.

    // Lane 0 writes both partial sums to smem.
    // Row 0: smem_buf[warp_id*4]     (bytes 0..31)
    // Row 1: smem_buf[32+warp_id*4]  (bytes 32..63)
    setp.eq.u32 %p4, %r4, 0;
    mov.u32 %r73, smem_buf;
    mul.lo.u32 %r74, %r5, 4;          // warp_id * 4
    add.u32 %r74, %r73, %r74;         // smem_buf + warp_id*4
    @%p4 st.shared.f32 [%r74], %f0;
    add.u32 %r78, %r74, 32;           // smem_buf + 32 + warp_id*4
    @%p4 st.shared.f32 [%r78], %f12;
    bar.sync 0;

    // -- Cross-warp reduce: warp 0 lanes 0..7 load 8 partial sums and reduce --
    setp.eq.u32 %p5, %r5, 0;
    setp.lt.u32 %p6, %r4, 8;
    and.pred    %p7, %p5, %p6;

    // Row 0: load smem_buf + lane*4, reduce with 3-step butterfly (4,2,1).
    mov.f32 %f7, 0f00000000;
    mul.lo.u32 %r75, %r4, 4;          // lane * 4
    add.u32 %r75, %r73, %r75;         // smem_buf + lane*4
    @%p7 ld.shared.f32 %f7, [%r75];

    mov.b32 %r71, %f7;
    shfl.sync.bfly.b32 %r72, %r71, 4, 0x1F, 0xFFFFFFFF;
    mov.b32 %f8, %r72; add.f32 %f7, %f7, %f8;
    mov.b32 %r71, %f7;
    shfl.sync.bfly.b32 %r72, %r71, 2, 0x1F, 0xFFFFFFFF;
    mov.b32 %f8, %r72; add.f32 %f7, %f7, %f8;
    mov.b32 %r71, %f7;
    shfl.sync.bfly.b32 %r72, %r71, 1, 0x1F, 0xFFFFFFFF;
    mov.b32 %f8, %r72; add.f32 %f7, %f7, %f8;
    // lane 0 of warp 0 holds the total row 0 sum in %f7.

    // Row 1: load smem_buf + 32 + lane*4, reduce with 3-step butterfly (4,2,1).
    mov.f32 %f14, 0f00000000;
    add.u32 %r79, %r75, 32;           // smem_buf + lane*4 + 32 = smem_buf + 32 + lane*4
    @%p7 ld.shared.f32 %f14, [%r79];

    mov.b32 %r71, %f14;
    shfl.sync.bfly.b32 %r72, %r71, 4, 0x1F, 0xFFFFFFFF;
    mov.b32 %f15, %r72; add.f32 %f14, %f14, %f15;
    mov.b32 %r71, %f14;
    shfl.sync.bfly.b32 %r72, %r71, 2, 0x1F, 0xFFFFFFFF;
    mov.b32 %f15, %r72; add.f32 %f14, %f14, %f15;
    mov.b32 %r71, %f14;
    shfl.sync.bfly.b32 %r72, %r71, 1, 0x1F, 0xFFFFFFFF;
    mov.b32 %f15, %r72; add.f32 %f14, %f14, %f15;
    // lane 0 of warp 0 holds the total row 1 sum in %f14.

    // Thread 0 writes both outputs.
    setp.eq.u32 %p4, %r3, 0;

    // Row 0: out[n_out]
    shl.b32 %r76, %r2, 2;             // n_out * 4
    cvt.u64.u32 %rd8, %r76;
    add.u64 %rd8, %rd0, %rd8;
    @%p4 st.global.f32 [%rd8], %f7;

    // Row 1: out[n + n_out] (n = param_n = n_rows)
    add.u32 %r80, %r0, %r2;           // n_rows + n_out
    shl.b32 %r80, %r80, 2;            // * 4
    cvt.u64.u32 %rd9, %r80;
    add.u64 %rd9, %rd0, %rd9;
    @%p4 st.global.f32 [%rd9], %f14;

DONE:
    ret;
}
"#;

/// Kernel entry-point name -- used by `cuModuleGetFunction`.
pub const ENTRY: &str = "q6k_gemv_sm120_batch";

/// Minimum compute capability (sm_120 = Blackwell consumer).
pub const MIN_CC: u32 = 1200;
