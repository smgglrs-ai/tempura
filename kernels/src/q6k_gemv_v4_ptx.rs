/// Hand-tuned sm_120 PTX kernel for Q6K x f32 GEMV decode with cp.async double-buffering.
///
/// # Why this kernel
///
/// Q6K down_proj [4096x14336] has k_blocks=56 with 8 warps -> 7 k_blocks/warp.
/// ncu showed 32% DRAM utilization on the oxide kpar8 baseline. With 7 iterations,
/// a 2-stage cp.async pipeline hides latency for all but the first block: the next
/// block's 212-byte load overlaps with the current block's 8-step dequant + f32 FMA.
///
/// # Q6K packed format (53 u32 = 212 bytes/block, native GGUF upload format)
///
/// ```text
/// u32[0]:      d (f32 bits) -- super-scale, f16 -> f32 at upload
/// u32[1..5]:   16 group scales as bias-128 u8 (4 per u32)
///              stored_byte = (scale_i8 + 128) as u8; decode: byte as f32 - 128.0
/// u32[5..37]:  ql[0..128] -- lower 4 bits of each weight (4 bytes per u32)
/// u32[37..53]: qh[0..64]  -- upper 2 bits of each weight (4 bytes per u32)
/// ```
///
/// # Launch geometry
///
/// Grid:  (n_rows, 1, 1) -- one block per output row
/// Block: (256, 1, 1)    -- 8 warps x 32 threads
/// Smem:  3424 bytes (static): 32 bytes reduction + 8 warps x 2 buffers x 212 bytes
///
/// # K-parallelism with double-buffering
///
/// Warp w handles blocks: w, w+8, w+16, ... (stride=8 warps).
/// For k_blocks=56: each warp handles exactly 7 blocks.
///
/// Prologue: cp.async block 0 -> buf_A, wait, then enter loop.
/// Loop iteration i:
///   1. Issue cp.async of block i+1 -> buf_next (overlaps with step 2)
///   2. Compute block i from buf_curr (8-step dequant + FMA, same as oxide kpar8)
///   3. cp.async.wait_group 0 (wait for block i+1 to land)
///   4. swap(curr, next)
///
/// Hides ~=6/7 of per-block DRAM latency (blocks 0-5 overlap with next prefetch).
///
/// # Parameters (same layout as oxide q6k_gemv_dp4a_kpar8)
///
/// ```text
/// param_out:        .u64 -- *mut f32, output [n_rows]
/// param_out_len:    .u64 -- n_rows (bounds guard)
/// param_x:          .u64 -- *const f32, activation [k_blocks * 256]
/// param_x_len:      .u64 -- activation length
/// param_packed:     .u64 -- *const u32, Q6K blocks [n_rows * k_blocks * 53]
/// param_packed_len: .u64 -- packed length
/// param_n:          .u32 -- n_rows
/// param_k_blocks:   .u32 -- k_blocks
/// ```
pub const PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// q6k_gemv_sm120_mbar_kpar8
// Q6K (53 u32/block = 212 bytes) x f32 GEMV decode, sm_120.
// cp.async double-buffering: next block loads while current block is computed.
// Same 8-step dequant as oxide q6k_gemv_dp4a_kpar8; reads from per-warp smem buffers.
//
// Grid: (n_rows, 1, 1). Block: (256, 1, 1) = 8 warps.
// Smem layout (3424 bytes static):
//   [0..31]:     8 f32 partial sums for cross-warp reduction
//   [32..3423]:  8 warps x (buf_A[212] + buf_B[212]) = 3392 bytes
.visible .entry q6k_gemv_sm120_mbar_kpar8(
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
    .reg .u64  %rd<12>;
    .reg .f32  %f<12>;
    .reg .pred %p<8>;

    // smem: reduction region (32 bytes) + per-warp double buffers (3392 bytes).
    .shared .align 4 .b8 smem_buf[3424];

    // -- Load parameters --------------------------------------------------------
    ld.param.u64 %rd0,  [param_out];
    ld.param.u64 %rd1,  [param_x];
    ld.param.u64 %rd2,  [param_packed];
    ld.param.u32 %r0,   [param_n];
    ld.param.u32 %r1,   [param_k_blocks];

    // -- Thread decomposition ---------------------------------------------------
    mov.u32 %r2, %ctaid.x;          // row = blockIdx.x
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

    // sc_shift_even = (lane >> 4) * 8  - scale bit-shift for even steps (0,2,4,6)
    // sc_shift_odd  = 16 + sc_shift_even - scale bit-shift for odd steps (1,3,5,7)
    shr.u32 %r8, %r4, 4;            // lane >> 4  (0 or 1)
    shl.b32 %r8, %r8, 3;            // sc_shift_even = (lane>>4)*8  (0 or 8)
    add.u32 %r9, %r8, 16;           // sc_shift_odd  (16 or 24)

    // Predicate for round-2 of cp.async: lanes 0..20 issue 2 copies, 21..31 issue 1.
    setp.lt.u32 %p1, %r4, 21;

    // -- Accumulator ------------------------------------------------------------
    mov.f32 %f0, 0f00000000;        // acc = 0.0

    // -- Constants --------------------------------------------------------------
    mov.f32 %f10, 0f43000000;       // 128.0f (bias for scale decoding)
    mov.f32 %f11, 0f42000000;       //  32.0f (bias for q6 centering)

    // -- Per-warp smem buffer base ----------------------------------------------
    // buf_A for warp w: smem_buf + 32 + w*424
    // buf_B for warp w: buf_A + 212
    mul.lo.u32 %r10, %r5, 424;
    add.u32    %r10, %r10, 32;       // %r10 = smem base for this warp (== address of buf_A)
    add.u32    %r11, %r10, 212;      // address of buf_B

    // curr = %r10 (buf_A), next = %r11 (buf_B)
    mov.u32    %r20, %r10;           // %r20 = curr smem address
    mov.u32    %r21, %r11;           // %r21 = next smem address

    // -- Prologue: cp.async block 0 - curr (buf_A) -----------------------------
    // Block 0 for warp w is at global index warp_id; all 8 blocks are contiguous
    // within the row.  Global byte offset = (row*k_blocks + warp_id) * 212.
    // Guard: if warp_id >= k_blocks, skip prologue and jump to REDUCE.
    setp.ge.u32 %p2, %r5, %r1;
    @%p2 bra REDUCE;

    mad.lo.u32 %r22, %r2, %r1, %r5;  // row*k_blocks + warp_id
    mul.lo.u32 %r23, %r22, 212;       // byte offset to block 0
    cvt.u64.u32 %rd3, %r23;
    add.u64 %rd3, %rd2, %rd3;          // %rd3 = ptr to block 0 for this warp

    // 53 u32s = 212 bytes.  Round 1 (all 32 lanes): lane l - smem[curr + l*4].
    shl.b32 %r24, %r4, 2;             // lane*4
    add.u32 %r25, %r20, %r24;         // smem dst (curr + lane*4)
    cvt.u64.u32 %rd4, %r24;
    add.u64 %rd4, %rd3, %rd4;          // global src (block0 + lane*4)
    cp.async.ca.shared.global [%r25], [%rd4], 4;

    // Round 2 (lanes 0..20): lane l - smem[curr + (32+l)*4].
    add.u32 %r26, %r4, 32;            // 32+lane
    shl.b32 %r26, %r26, 2;            // (32+lane)*4
    add.u32 %r27, %r20, %r26;
    cvt.u64.u32 %rd4, %r26;
    add.u64 %rd4, %rd3, %rd4;
    @%p1 cp.async.ca.shared.global [%r27], [%rd4], 4;

    cp.async.commit_group;
    cp.async.wait_group 0;
    // (SIMT lockstep: all 32 threads waited - smem[curr] is fully populated.)

    // -- K-loop: blk = warp_id, warp_id+8, ...  --------------------------------
    mov.u32 %r30, %r5;               // blk = warp_id

KLOOP:
    setp.ge.u32 %p0, %r30, %r1;     // blk >= k_blocks?
    @%p0 bra REDUCE;

    // Issue async-load of block (blk+8) into next smem buffer.
    add.u32 %r31, %r30, 8;           // next_blk = blk + 8
    setp.lt.u32 %p3, %r31, %r1;     // next_blk < k_blocks?
    @!%p3 bra SKIP_LOAD;

    mad.lo.u32 %r32, %r2, %r1, %r31; // row*k_blocks + next_blk
    mul.lo.u32 %r33, %r32, 212;
    cvt.u64.u32 %rd5, %r33;
    add.u64 %rd5, %rd2, %rd5;         // global ptr to next_blk

    // Round 1 (all 32 lanes)
    shl.b32 %r34, %r4, 2;
    add.u32 %r35, %r21, %r34;
    cvt.u64.u32 %rd6, %r34;
    add.u64 %rd6, %rd5, %rd6;
    cp.async.ca.shared.global [%r35], [%rd6], 4;

    // Round 2 (lanes 0..20)
    add.u32 %r36, %r4, 32;
    shl.b32 %r36, %r36, 2;
    add.u32 %r37, %r21, %r36;
    cvt.u64.u32 %rd6, %r36;
    add.u64 %rd6, %rd5, %rd6;
    @%p1 cp.async.ca.shared.global [%r37], [%rd6], 4;

    cp.async.commit_group;

SKIP_LOAD:
    // -- Read cached data from curr smem buffer ---------------------------------
    // u32[0]: d (f32 bits)
    ld.shared.b32 %r40, [%r20 + 0];
    mov.b32 %f1, %r40;               // d

    // u32[1..5]: scale words sc_01, sc_23, sc_45, sc_67
    ld.shared.b32 %r41, [%r20 +  4]; // sc_01 (groups 0-3)
    ld.shared.b32 %r42, [%r20 +  8]; // sc_23 (groups 4-7)
    ld.shared.b32 %r43, [%r20 + 12]; // sc_45 (groups 8-11)
    ld.shared.b32 %r44, [%r20 + 16]; // sc_67 (groups 12-15)

    // ql_a = smem_u32[(5  + lane_div4)] >> byte_shift & 0xFF   (steps 0,2)
    add.u32 %r45, %r7, 5;
    shl.b32 %r45, %r45, 2;
    add.u32 %r45, %r20, %r45;
    ld.shared.b32 %r45, [%r45];
    shr.u32 %r45, %r45, %r6;
    and.b32 %r45, %r45, 255;          // ql_a

    // ql_b = smem_u32[(13 + lane_div4)] >> byte_shift & 0xFF   (steps 1,3)
    add.u32 %r46, %r7, 13;
    shl.b32 %r46, %r46, 2;
    add.u32 %r46, %r20, %r46;
    ld.shared.b32 %r46, [%r46];
    shr.u32 %r46, %r46, %r6;
    and.b32 %r46, %r46, 255;          // ql_b

    // ql_c = smem_u32[(21 + lane_div4)] >> byte_shift & 0xFF   (steps 4,6)
    add.u32 %r47, %r7, 21;
    shl.b32 %r47, %r47, 2;
    add.u32 %r47, %r20, %r47;
    ld.shared.b32 %r47, [%r47];
    shr.u32 %r47, %r47, %r6;
    and.b32 %r47, %r47, 255;          // ql_c

    // ql_d = smem_u32[(29 + lane_div4)] >> byte_shift & 0xFF   (steps 5,7)
    add.u32 %r48, %r7, 29;
    shl.b32 %r48, %r48, 2;
    add.u32 %r48, %r20, %r48;
    ld.shared.b32 %r48, [%r48];
    shr.u32 %r48, %r48, %r6;
    and.b32 %r48, %r48, 255;          // ql_d

    // qh_0 = smem_u32[(37 + lane_div4)] >> byte_shift & 0xFF   (steps 0-3)
    add.u32 %r49, %r7, 37;
    shl.b32 %r49, %r49, 2;
    add.u32 %r49, %r20, %r49;
    ld.shared.b32 %r49, [%r49];
    shr.u32 %r49, %r49, %r6;
    and.b32 %r49, %r49, 255;          // qh_0

    // qh_1 = smem_u32[(45 + lane_div4)] >> byte_shift & 0xFF   (steps 4-7)
    add.u32 %r50, %r7, 45;
    shl.b32 %r50, %r50, 2;
    add.u32 %r50, %r20, %r50;
    ld.shared.b32 %r50, [%r50];
    shr.u32 %r50, %r50, %r6;
    and.b32 %r50, %r50, 255;          // qh_1

    // -- 8-step dequant + FMA (mirrors oxide q6k_gemv_dp4a_kpar8 v2 logic) ------
    // x base for this block: x_ptr + blk*256*4 bytes
    mul.lo.u32 %r60, %r30, 256;       // blk*256

    // Macro: for each step, compute sc, q6, load x, FMA into acc.
    // sc = (sc_word >> sc_shift) & 0xFF, then - 128.0
    // q6 = (lo_nibble) | (hi_bits << 4),  acc += d*sc*(q6-32.0)*x[i]

    // Step 0: within=lane, sc=sc_01>>sc_shift_even, lo=ql_a&0xF, hi=qh_0&0x3
    shr.u32 %r61, %r41, %r8;
    and.b32 %r61, %r61, 255;
    cvt.rn.f32.u32 %f2, %r61;
    sub.f32 %f2, %f2, %f10;           // sc
    and.b32 %r62, %r45, 15;           // lo
    and.b32 %r63, %r49, 3;            // hi
    shl.b32 %r63, %r63, 4;
    or.b32  %r62, %r62, %r63;         // q6
    cvt.rn.f32.u32 %f3, %r62;
    sub.f32 %f3, %f3, %f11;           // q6-32
    add.u32 %r64, %r60, %r4;          // blk*256+lane
    shl.b32 %r64, %r64, 2;
    cvt.u64.u32 %rd7, %r64;
    add.u64 %rd7, %rd1, %rd7;
    ld.global.nc.f32 %f4, [%rd7];
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

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

    // -- Wait for next block (if prefetch was issued) ---------------------------
    @!%p3 bra NO_WAIT;
    cp.async.wait_group 0;
    // SIMT lockstep: all 32 threads return here together - next smem buffer is ready.
NO_WAIT:

    // -- Swap curr/next smem buffer pointers -----------------------------------
    mov.u32 %r70, %r20;
    mov.u32 %r20, %r21;
    mov.u32 %r21, %r70;

    // -- Advance blk and loop --------------------------------------------------
    add.u32 %r30, %r30, 8;
    bra KLOOP;

    // -- Warp butterfly reduction (5 steps, full 32 lanes) --------------------
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
    // lane 0 of each warp holds partial row sum.

    // Lane 0 writes to smem[warp_id * 4].
    setp.eq.u32 %p4, %r4, 0;
    mov.u32 %r73, smem_buf;
    mul.lo.u32 %r74, %r5, 4;
    add.u32 %r74, %r73, %r74;
    @%p4 st.shared.f32 [%r74], %f0;
    bar.sync 0;

    // Warp 0 lanes 0..7 load partial sums, do 3-step XOR butterfly to sum 8 warps.
    setp.eq.u32 %p5, %r5, 0;
    setp.lt.u32 %p6, %r4, 8;
    and.pred    %p7, %p5, %p6;

    mov.f32 %f7, 0f00000000;
    mul.lo.u32 %r75, %r4, 4;
    add.u32 %r75, %r73, %r75;
    @%p7 ld.shared.f32 %f7, [%r75];

    // 3-step XOR butterfly: delta = 4, 2, 1.
    mov.b32 %r71, %f7;
    shfl.sync.bfly.b32 %r72, %r71, 4, 0x1F, 0xFFFFFFFF;
    mov.b32 %f8, %r72; add.f32 %f7, %f7, %f8;

    mov.b32 %r71, %f7;
    shfl.sync.bfly.b32 %r72, %r71, 2, 0x1F, 0xFFFFFFFF;
    mov.b32 %f8, %r72; add.f32 %f7, %f7, %f8;

    mov.b32 %r71, %f7;
    shfl.sync.bfly.b32 %r72, %r71, 1, 0x1F, 0xFFFFFFFF;
    mov.b32 %f8, %r72; add.f32 %f7, %f7, %f8;
    // lane 0 of warp 0 holds the full row sum.

    // Thread 0 writes output.
    setp.eq.u32 %p4, %r3, 0;
    shl.b32 %r76, %r2, 2;
    cvt.u64.u32 %rd8, %r76;
    add.u64 %rd8, %rd0, %rd8;
    @%p4 st.global.f32 [%rd8], %f7;

DONE:
    ret;
}
"#;

/// Kernel entry-point name -- used by `cuModuleGetFunction`.
pub const ENTRY: &str = "q6k_gemv_sm120_mbar_kpar8";

/// Minimum compute capability (sm_120 = Blackwell consumer).
pub const MIN_CC: u32 = 1200;
