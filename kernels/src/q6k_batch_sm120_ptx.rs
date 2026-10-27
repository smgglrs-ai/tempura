/// sm_120 PTX batch kernel for Q6K x f32 GEMV decode: M rows share one weight read.
///
/// # Purpose
///
/// Q6K down_proj [4096x14336] runs M times sequentially in per-row GEMV mode,
/// reading weights M times. This kernel reads weights ONCE (into shared memory)
/// and computes all M dot products. It replaces the earlier M=2-hardcoded
/// variant that silently left output rows 2..M unwritten for M > 2 (the
/// "nondeterministic at M>=5" batch bug: rows 2..M held stale memory).
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
/// # Launch geometry
///
/// Grid:  (n_rows, 1, 1) -- one block per output neuron
/// Block: (256, 1, 1)    -- 8 warps x 32 threads
/// Dynamic smem: `k_blocks * 212` bytes weight area + 512 bytes reduction area.
///
/// # Two-phase structure (why)
///
/// Phase 1: all 8 warps cooperatively cp.async EVERY weight block of this
/// neuron into its persistent smem slot, then one commit + wait_group 0 +
/// bar.sync. Because the barrier is outside every loop, warps may run
/// different k-loop trip counts safely -- the old kernel's `k_blocks % 8 == 0`
/// restriction (uniform loop-barrier count) is gone.
///
/// Phase 2: for each row pair p (rows 2p, 2p+1), each warp k-splits over the
/// smem-resident blocks (blk = warp_id, warp_id+8, ...), dequantizes and FMAs
/// into two register accumulators, butterfly-reduces across the warp, and
/// lane 0 writes per-warp partials to smem.
///
/// Phase 3 (after one bar.sync): warp 0 reduces the per-warp partials for each
/// pair and thread 0 writes out[row * n + n_out].
///
/// Weights are read from DRAM/L2 exactly once per neuron regardless of M.
///
/// # Activation layout
///
/// param_x: [M * k_blocks * 256] f32, row-major (row r at r * k_blocks * 256).
///
/// # Output layout
///
/// param_out: [M * n_rows] f32 -- row r at [r*n .. (r+1)*n).
///
/// # Odd M
///
/// The last pair's second row index is clamped to m-1 (its accumulator is
/// computed but discarded); the output write for row >= m is predicated off.
pub const PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// q6k_gemv_sm120_batch (M rows, 2..=16)
// Q6K x f32 GEMV decode, M activation rows sharing one weight read per block.
// Grid: (n_rows, 1, 1). Block: (256, 1, 1) = 8 warps.
// Dynamic smem: k_blocks*212 bytes weights + 512 bytes cross-warp partials.
// Phase 1: cooperative cp.async of all weight blocks; single CTA barrier.
// Phase 2: per row-pair k-split compute from smem (no barriers in loops).
// Phase 3: warp 0 cross-warp reduction + output writes.
.visible .entry q6k_gemv_sm120_batch(
    .param .u64 param_out,
    .param .u64 param_out_len,
    .param .u64 param_x,
    .param .u64 param_x_len,
    .param .u64 param_packed,
    .param .u64 param_packed_len,
    .param .u32 param_n,
    .param .u32 param_k_blocks,
    .param .u32 param_m
)
{
    .reg .u32  %r<84>;
    .reg .u64  %rd<13>;
    .reg .f32  %f<16>;
    .reg .pred %p<9>;

    // Dynamic shared memory: weights then cross-warp partials.
    .extern .shared .align 4 .b8 smem[];

    // -- Load parameters --------------------------------------------------------
    ld.param.u64 %rd0,  [param_out];
    ld.param.u64 %rd1,  [param_x];
    ld.param.u64 %rd2,  [param_packed];
    ld.param.u32 %r0,   [param_n];
    ld.param.u32 %r1,   [param_k_blocks];
    ld.param.u32 %r10,  [param_m];

    // -- Thread decomposition ---------------------------------------------------
    mov.u32 %r2, %ctaid.x;          // n_out = blockIdx.x
    setp.ge.u32 %p0, %r2, %r0;
    @%p0 bra DONE;

    mov.u32 %r3, %tid.x;            // tid  (0..255)
    and.b32 %r4, %r3, 31;           // lane = tid & 31
    shr.u32 %r5, %r3, 5;            // warp_id = tid >> 5  (0..7)

    // byte_shift = (lane % 4) * 8  - used for ql/qh byte extraction from u32
    and.b32 %r6, %r4, 3;
    shl.b32 %r6, %r6, 3;            // %r6 = byte_shift

    // lane_div4 = lane >> 2  - used for ql/qh word index
    shr.u32 %r7, %r4, 2;            // %r7 = lane_div4

    // sc_shift_even = (lane >> 4) * 8; sc_shift_odd = sc_shift_even + 16
    shr.u32 %r8, %r4, 4;
    shl.b32 %r8, %r8, 3;            // sc_shift_even  (0 or 8)
    add.u32 %r9, %r8, 16;           // sc_shift_odd   (16 or 24)

    // Predicate for round-2 of cp.async: lanes 0..20 issue 2 copies, 21..31 issue 1.
    setp.lt.u32 %p1, %r4, 21;

    // -- Shared memory layout ---------------------------------------------------
    mov.u32 %r12, smem;             // dynamic smem base
    mul.lo.u32 %r13, %r1, 212;      // partials byte offset = k_blocks * 212

    // Row pair count = (m + 1) / 2; row byte stride = k_blocks * 1024.
    add.u32 %r11, %r10, 1;
    shr.u32 %r11, %r11, 1;          // pairs
    mul.lo.u32 %r15, %r1, 1024;     // row stride in bytes (fits u32)
    cvt.u64.u32 %rd12, %r15;        // %rd12 = row stride bytes

    // -- Constants --------------------------------------------------------------
    mov.f32 %f10, 0f43000000;       // 128.0
    mov.f32 %f11, 0f42000000;       //  32.0

    // ===========================================================================
    // Phase 1: cooperatively load ALL k_blocks weight blocks into smem.
    // Warp w fetches blocks w, w+8, w+16, ... Every thread issues all of its
    // cp.asyncs, then ONE commit_group + wait_group 0 + bar.sync makes every
    // lane's copies CTA-visible (wait_group is per-thread).
    // ===========================================================================
    mov.u32 %r17, %r5;              // b = warp_id

P1LOOP:
    setp.ge.u32 %p2, %r17, %r1;     // b >= k_blocks?
    @%p2 bra P1DONE;

    mad.lo.u32 %r22, %r2, %r1, %r17; // n_out*k_blocks + b
    mul.lo.u32 %r23, %r22, 212;
    cvt.u64.u32 %rd3, %r23;
    add.u64 %rd3, %rd2, %rd3;        // %rd3 = global ptr to block b

    mul.lo.u32 %r24, %r17, 212;
    add.u32 %r24, %r12, %r24;        // %r24 = smem slot base for block b

    // Round 1: all 32 lanes copy one u32 each (words 0..31).
    shl.b32 %r25, %r4, 2;            // lane*4
    add.u32 %r26, %r24, %r25;
    cvt.u64.u32 %rd4, %r25;
    add.u64 %rd4, %rd3, %rd4;
    cp.async.ca.shared.global [%r26], [%rd4], 4;

    // Round 2: lanes 0..20 copy one more u32 each (words 32..52).
    add.u32 %r27, %r4, 32;
    shl.b32 %r27, %r27, 2;
    add.u32 %r28, %r24, %r27;
    cvt.u64.u32 %rd4, %r27;
    add.u64 %rd4, %rd3, %rd4;
    @%p1 cp.async.ca.shared.global [%r28], [%rd4], 4;

    add.u32 %r17, %r17, 8;
    bra P1LOOP;

P1DONE:
    cp.async.commit_group;
    cp.async.wait_group 0;
    bar.sync 0;

    // ===========================================================================
    // Phase 2: per row-pair k-split compute. No barriers in any loop: warps run
    // independent blk ranges and the butterflies are warp-local shfl.sync.
    // ===========================================================================
    mov.u32 %r16, 0;                 // p = 0

PAIRLOOP:
    setp.ge.u32 %p2, %r16, %r11;     // p >= pairs?
    @%p2 bra PAIRDONE;

    // row0 = 2p; row1 = min(2p+1, m-1). For odd m the last pair's second
    // accumulator duplicates row m-1; its output write is predicated off.
    shl.b32 %r29, %r16, 1;           // row0 = 2p
    add.u32 %r30, %r29, 1;           // row1 = 2p+1
    setp.ge.u32 %p8, %r30, %r10;
    @%p8 sub.u32 %r30, %r10, 1;      // clamp row1 to m-1

    // %rd10 = x + row0 * stride_bytes;  %rd11 = (row1 - row0) * stride_bytes
    cvt.u64.u32 %rd10, %r29;
    mul.lo.u64 %rd10, %rd10, %rd12;
    add.u64 %rd10, %rd1, %rd10;
    sub.u32 %r31, %r30, %r29;        // 0 or 1
    cvt.u64.u32 %rd11, %r31;
    mul.lo.u64 %rd11, %rd11, %rd12;

    mov.f32 %f0, 0f00000000;         // acc0 = 0.0 (row 2p)
    mov.f32 %f12, 0f00000000;        // acc1 = 0.0 (row 2p+1)

    mov.u32 %r18, %r5;               // blk = warp_id

KLOOP:
    setp.ge.u32 %p3, %r18, %r1;      // blk >= k_blocks?
    @%p3 bra KDONE;

    mul.lo.u32 %r32, %r18, 212;
    add.u32 %r32, %r12, %r32;        // %r32 = smem slot base for block blk

    // -- Read Q6K weight block header from smem --------------------------------
    ld.shared.b32 %r40, [%r32 + 0];
    mov.b32 %f1, %r40;               // d (super-scale, f32 bits)

    ld.shared.b32 %r41, [%r32 +  4]; // sc_01
    ld.shared.b32 %r42, [%r32 +  8]; // sc_23
    ld.shared.b32 %r43, [%r32 + 12]; // sc_45
    ld.shared.b32 %r44, [%r32 + 16]; // sc_67

    // ql_a = smem_u32[5 + lane_div4] >> byte_shift & 0xFF  (steps 0,2)
    add.u32 %r45, %r7, 5;
    shl.b32 %r45, %r45, 2;
    add.u32 %r45, %r32, %r45;
    ld.shared.b32 %r45, [%r45];
    shr.u32 %r45, %r45, %r6;
    and.b32 %r45, %r45, 255;

    // ql_b = smem_u32[13 + lane_div4] >> byte_shift & 0xFF  (steps 1,3)
    add.u32 %r46, %r7, 13;
    shl.b32 %r46, %r46, 2;
    add.u32 %r46, %r32, %r46;
    ld.shared.b32 %r46, [%r46];
    shr.u32 %r46, %r46, %r6;
    and.b32 %r46, %r46, 255;

    // ql_c = smem_u32[21 + lane_div4] >> byte_shift & 0xFF  (steps 4,6)
    add.u32 %r47, %r7, 21;
    shl.b32 %r47, %r47, 2;
    add.u32 %r47, %r32, %r47;
    ld.shared.b32 %r47, [%r47];
    shr.u32 %r47, %r47, %r6;
    and.b32 %r47, %r47, 255;

    // ql_d = smem_u32[29 + lane_div4] >> byte_shift & 0xFF  (steps 5,7)
    add.u32 %r48, %r7, 29;
    shl.b32 %r48, %r48, 2;
    add.u32 %r48, %r32, %r48;
    ld.shared.b32 %r48, [%r48];
    shr.u32 %r48, %r48, %r6;
    and.b32 %r48, %r48, 255;

    // qh_0 = smem_u32[37 + lane_div4] >> byte_shift & 0xFF  (steps 0-3)
    add.u32 %r49, %r7, 37;
    shl.b32 %r49, %r49, 2;
    add.u32 %r49, %r32, %r49;
    ld.shared.b32 %r49, [%r49];
    shr.u32 %r49, %r49, %r6;
    and.b32 %r49, %r49, 255;

    // qh_1 = smem_u32[45 + lane_div4] >> byte_shift & 0xFF  (steps 4-7)
    add.u32 %r50, %r7, 45;
    shl.b32 %r50, %r50, 2;
    add.u32 %r50, %r32, %r50;
    ld.shared.b32 %r50, [%r50];
    shr.u32 %r50, %r50, %r6;
    and.b32 %r50, %r50, 255;

    // -- 8-step dequant + FMA for both rows ------------------------------------
    // For each step: compute f5 = d*sc*(q6-32) once, then FMA into acc0 (row
    // 2p) and acc1 (row 2p+1). Row bases: %rd10 and %rd10 + %rd11.
    mul.lo.u32 %r60, %r18, 256;     // blk * 256

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
    add.u64 %rd7, %rd10, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd11;
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
    add.u64 %rd7, %rd10, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd11;
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
    add.u64 %rd7, %rd10, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd11;
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
    add.u64 %rd7, %rd10, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd11;
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
    add.u64 %rd7, %rd10, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd11;
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
    add.u64 %rd7, %rd10, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd11;
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
    add.u64 %rd7, %rd10, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd11;
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
    add.u64 %rd7, %rd10, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;
    add.u64 %rd9, %rd7, %rd11;
    ld.global.nc.f32 %f13, [%rd9];
    fma.rn.f32 %f12, %f5, %f13, %f12;

    add.u32 %r18, %r18, 8;
    bra KLOOP;

KDONE:
    // -- Warp butterfly reduction: acc0 (5 steps, all 32 lanes) ----------------
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

    // -- Warp butterfly reduction: acc1 (5 steps, all 32 lanes) ----------------
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
    // lane 0 of each warp holds its partial row sums.

    // -- Lane 0 writes both partials to smem ------------------------------------
    // row0 partials: smem[k_blocks*212 + (p*8 + warp_id)*4]
    // row1 partials: smem[k_blocks*212 + 256 + (p*8 + warp_id)*4]
    setp.eq.u32 %p4, %r4, 0;
    mad.lo.u32 %r33, %r16, 8, %r5;   // p*8 + warp_id
    shl.b32 %r33, %r33, 2;
    add.u32 %r33, %r13, %r33;
    add.u32 %r33, %r12, %r33;
    @%p4 st.shared.f32 [%r33], %f0;
    add.u32 %r34, %r33, 256;
    @%p4 st.shared.f32 [%r34], %f12;

    add.u32 %r16, %r16, 1;
    bra PAIRLOOP;

PAIRDONE:
    // One barrier: all pairs' partials are CTA-visible before warp 0 reduces.
    bar.sync 0;

    // ===========================================================================
    // Phase 3: warp 0 reduces the 8 per-warp partials per row and writes output.
    // ===========================================================================
    setp.eq.u32 %p5, %r5, 0;         // warp 0
    setp.lt.u32 %p6, %r4, 8;         // lanes 0..7
    and.pred    %p7, %p5, %p6;

    mov.u32 %r16, 0;                 // p = 0

P3LOOP:
    setp.ge.u32 %p2, %r16, %r11;     // p >= pairs?
    @%p2 bra DONE;

    // Row 0 (2p): load smem[k_blocks*212 + (p*8 + lane)*4], 3-step butterfly.
    mov.f32 %f7, 0f00000000;
    mad.lo.u32 %r35, %r16, 8, %r4;   // p*8 + lane
    shl.b32 %r35, %r35, 2;
    add.u32 %r35, %r13, %r35;
    add.u32 %r35, %r12, %r35;
    @%p7 ld.shared.f32 %f7, [%r35];

    mov.b32 %r71, %f7;
    shfl.sync.bfly.b32 %r72, %r71, 4, 0x1F, 0xFFFFFFFF;
    mov.b32 %f8, %r72; add.f32 %f7, %f7, %f8;
    mov.b32 %r71, %f7;
    shfl.sync.bfly.b32 %r72, %r71, 2, 0x1F, 0xFFFFFFFF;
    mov.b32 %f8, %r72; add.f32 %f7, %f7, %f8;
    mov.b32 %r71, %f7;
    shfl.sync.bfly.b32 %r72, %r71, 1, 0x1F, 0xFFFFFFFF;
    mov.b32 %f8, %r72; add.f32 %f7, %f7, %f8;
    // lane 0 of warp 0 holds the total row 2p sum in %f7.

    // Row 1 (2p+1): load smem[k_blocks*212 + 256 + (p*8 + lane)*4].
    mov.f32 %f14, 0f00000000;
    add.u32 %r38, %r35, 256;
    @%p7 ld.shared.f32 %f14, [%r38];

    mov.b32 %r71, %f14;
    shfl.sync.bfly.b32 %r72, %r71, 4, 0x1F, 0xFFFFFFFF;
    mov.b32 %f15, %r72; add.f32 %f14, %f14, %f15;
    mov.b32 %r71, %f14;
    shfl.sync.bfly.b32 %r72, %r71, 2, 0x1F, 0xFFFFFFFF;
    mov.b32 %f15, %r72; add.f32 %f14, %f14, %f15;
    mov.b32 %r71, %f14;
    shfl.sync.bfly.b32 %r72, %r71, 1, 0x1F, 0xFFFFFFFF;
    mov.b32 %f15, %r72; add.f32 %f14, %f14, %f15;
    // lane 0 of warp 0 holds the total row 2p+1 sum in %f14.

    // Thread 0 writes out[(2p)*n + n_out].
    setp.eq.u32 %p4, %r3, 0;
    shl.b32 %r36, %r16, 1;           // 2p
    mad.lo.u32 %r37, %r36, %r0, %r2; // 2p*n + n_out
    shl.b32 %r37, %r37, 2;
    cvt.u64.u32 %rd8, %r37;
    add.u64 %rd8, %rd0, %rd8;
    @%p4 st.global.f32 [%rd8], %f7;

    // Thread 0 writes out[(2p+1)*n + n_out] -- only when 2p+1 < m (odd M).
    add.u32 %r39, %r36, 1;           // 2p+1
    setp.lt.u32 %p8, %r39, %r10;
    and.pred    %p2, %p4, %p8;       // tid == 0 AND row live
    mad.lo.u32 %r37, %r39, %r0, %r2; // (2p+1)*n + n_out
    shl.b32 %r37, %r37, 2;
    cvt.u64.u32 %rd9, %r37;
    add.u64 %rd9, %rd0, %rd9;
    @%p2 st.global.f32 [%rd9], %f14;

    add.u32 %r16, %r16, 1;
    bra P3LOOP;

DONE:
    ret;
}
"#;

/// Kernel entry-point name -- used by `cuModuleGetFunction`.
pub const ENTRY: &str = "q6k_gemv_sm120_batch";

/// Minimum compute capability (sm_120 = Blackwell consumer).
pub const MIN_CC: u32 = 1200;

/// Cross-warp reduction area size in bytes (2 row areas x 8 pairs x 8 warps x 4B).
pub const REDUCTION_SMEM: usize = 512;

/// Dynamic shared memory bytes the host must request for `k_blocks` blocks.
pub fn dynamic_smem_bytes(k_blocks: usize) -> usize {
    k_blocks * 212 + REDUCTION_SMEM
}
