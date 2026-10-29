/// Hand-tuned sm_120 PTX kernel for Q6K x f32 GEMV decode with cp.async
/// triple-buffering and REGISTER-EXCHANGE consumption.
///
/// # Why this kernel
///
/// Q6K down_proj [4096x14336] has k_blocks=56 with 8 warps -> 7 k_blocks/warp.
/// ncu showed 32% DRAM utilization on the oxide kpar8 baseline. With 7
/// iterations, a cp.async pipeline hides latency for all but the first block:
/// the next block's 212-byte load overlaps with the current block's 8-step
/// dequant + f32 FMA.
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
/// Smem:  5384 bytes (static): 32 bytes reduction + 8 warps x 3 buffers
///        x 212 bytes + u32 reduce counter (region 5120..5375 is dead,
///        reserved from the removed mbarrier design)
///
/// # K-parallelism with triple-buffering
///
/// Warp w handles blocks: w, w+8, w+16, ... (stride=8 warps).
/// For k_blocks=56: each warp handles exactly 7 blocks.
///
/// Loop iteration i:
///   1. Issue cp.async of block i+1 -> buf_next + commit_group (FIRST, so
///      the copy overlaps this iteration's consume+compute)
///   2. Each lane loads ONLY the words it copied itself (see "The register
///      exchange" below) -- the copies landing in `curr` were completed and
///      published by iteration i-1's tail (or the prologue for i=0)
///   3. Register broadcast via shfl.sync.idx -- every lane obtains d, the 4
///      scale words and its 6 ql/qh words
///   4. Compute block i (8-step dequant + FMA) from registers
///   5. cp.async.wait_group 0 + fence.proxy.async + bar.sync 0 x3 (the
///      publication point -- see below)
///   6. swap(curr, next) -- the slot written at i was last read at i-2
///      (triple buffers)
///
/// # The register exchange -- what fixes the kb=16 residual
///
/// The 2026-10-29 rewrite eliminated CROSS-LANE smem reads: each lane reads
/// exclusively the words it copied ITSELF (round 1: lane l copies word l;
/// round 2: lanes 0..20 copy words 32+l). The values a lane needs from
/// other lanes' words (d, the scale words, the ql/qh words addressed by
/// lane>>2) are exchanged in REGISTERS via shfl.sync.idx: registers carry
/// no proxy semantics, and the shuffle's data dependence on the source
/// lane's ld.shared result forces that load to complete first. This closed
/// the last residual race of the barrier-based design (~3/81920 wrong at
/// kb=16 only): cross-lane publication of cp.async data was the one thing
/// the CTA barriers could not fully order.
///
/// # Why the CTA barriers stay (2026-10-29 bisection)
///
/// Removing the loop's bar.sync x3 (keeping wait_group + the per-thread
/// fence.proxy.async, SASS audited correct with the DEPBAR lexically after
/// the commit AND at the rotated loop head) made iteration >= 2 read
/// UNWRITTEN slots at kb=13/16/40: MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S
/// does not publish LDGSTS data to subsequent same-thread LDS on this
/// hw/ptxas without a CTA barrier. The bars are load-bearing. Consequently:
///   - bar.sync lowers to BAR.SYNC.DEFER_BLOCKING on Blackwell (warps run
///     past it until an instruction requires completion); the second bar
///     forces the first to complete, the third pins it
///   - the barrier is CTA-wide, so all 8 warps must run the same number of
///     loop iterations -- the HOST GATES this kernel to k_blocks % 8 == 0
///     (other shapes fall back to the dp4a kpar kernel)
///
/// # Cross-warp reduction
///
/// The final 8 partial sums are exchanged via a smem store ->
/// membar.cta -> red.release.cta counter bump / ld.acquire.cta poll ->
/// sequential sum on thread 0 (generic proxy only; the mbarrier and
/// shuffle-butterfly variants of this handoff raced on this hw, bisected
/// 2026-10-29).
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

// q6k_gemv_sm120_kpar8
// Q6K (53 u32/block = 212 bytes) x f32 GEMV decode, sm_120.
// cp.async triple-buffering: next block loads while current block is computed.
// Same 8-step dequant as oxide q6k_gemv_dp4a_kpar8; consumption is per-lane
// (own copies only) + shfl.sync.idx register broadcast -- see the Rust doc
// comment above for why the loop has no barriers.
//
// Grid: (n_rows, 1, 1). Block: (256, 1, 1) = 8 warps.
// Smem layout (5384 bytes static):
//   [0..31]:     8 f32 partial sums for cross-warp reduction
//   [32..5119]:  8 warps x 3 slots x 212 bytes = 5088 bytes
//   [5120..5375]: dead (reserved from the removed mbarrier design)
//   [5376..5379]: u32 cross-warp reduce handoff counter
.visible .entry q6k_gemv_sm120_kpar8(
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
    .reg .u32  %r<110>;
    .reg .u64  %rd<12>;
    .reg .f32  %f<12>;
    .reg .pred %p<10>;

    .shared .align 8 .b8 smem_buf[5384];

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
    // Round 1: lane l copies word l. Round 2: lane l (<21) copies word 32+l.
    setp.lt.u32 %p1, %r4, 21;

    // -- Accumulator ------------------------------------------------------------
    mov.f32 %f0, 0f00000000;        // acc = 0.0

    // -- Constants --------------------------------------------------------------
    mov.f32 %f10, 0f43000000;       // 128.0f (bias for scale decoding)
    mov.f32 %f11, 0f42000000;       //  32.0f (bias for q6 centering)

    // -- Per-warp smem slot pointers --------------------------------------------
    // slot_0 for warp w: smem_buf + 32 + w*636; slots at +0, +212, +424.
    mul.lo.u32 %r10, %r5, 636;
    add.u32    %r10, %r10, 32;
    mov.u32    %r20, %r10;            // %r20 = curr smem address
    add.u32    %r21, %r10, 212;       // %r21 = next smem address
    add.u32    %r98, %r10, 424;       // %r98 = spare smem address

    // -- CTA-wide reduce HANDOFF COUNTER at smem_buf+5376 (u32, zeroed by
    // thread 0): the cross-warp partial exchange uses the classic
    // store -> membar.cta -> atomic counter bump / volatile poll ->
    // membar.cta -> load pattern.  The mbarrier variant of this handoff
    // raced on this hw/ptxas (warp 0's try_wait passed before the
    // partials were visible; bisected 2026-10-29 with a synthetic-weights
    // harness: buffer machinery 0/81920 wrong with the reduce bypassed,
    // corruption only with the mbarrier cross-warp handoff active).
    mov.u32 %r99, smem_buf;
    add.u32 %r99, %r99, 5376;
    setp.eq.u32 %p9, %r3, 0;
    @%p9 st.shared.u32 [%r99], 0;
    // Double barrier: on Blackwell bar.sync lowers to BAR.SYNC.DEFER_BLOCKING
    // -- warps continue until an instruction requires completion, and
    // cp.async/mbarrier ops do NOT require it, so a single barrier can let
    // an arrive-on hit an mbarrier whose init has not landed yet.  A
    // SECOND bar.sync forces the first to complete -- barrier instructions
    // are completion-requiring -- so the init writes are CTA-visible
    // before any arrive.
    bar.sync 0;
    bar.sync 0;

    // -- Prologue: issue cp.async of block warp_id -> curr (slot_0) -------------
    // Guard: if warp_id >= k_blocks, skip to REDUCE. (Dead under the host's
    // k_blocks % 8 == 0 gate -- kb >= 8 covers all 8 warps -- but kept for
    // safety.)
    setp.ge.u32 %p2, %r5, %r1;
    @%p2 bra REDUCE;

    mad.lo.u32 %r22, %r2, %r1, %r5;  // row*k_blocks + warp_id
    mul.lo.u32 %r23, %r22, 212;       // byte offset to block 0
    cvt.u64.u32 %rd3, %r23;
    add.u64 %rd3, %rd2, %rd3;          // %rd3 = ptr to block 0 for this warp

    // 53 u32s = 212 bytes.  Round 1 (all 32 lanes): lane l -> smem[curr + l*4].
    shl.b32 %r24, %r4, 2;             // lane*4
    add.u32 %r25, %r20, %r24;         // smem dst (curr + lane*4)
    cvt.u64.u32 %rd4, %r24;
    add.u64 %rd4, %rd3, %rd4;          // global src (block0 + lane*4)
    cp.async.ca.shared.global [%r25], [%rd4], 4;

    // Round 2 (lanes 0..20): lane l -> smem[curr + (32+l)*4].
    add.u32 %r26, %r4, 32;            // 32+lane
    shl.b32 %r26, %r26, 2;            // (32+lane)*4
    add.u32 %r27, %r20, %r26;
    cvt.u64.u32 %rd4, %r26;
    add.u64 %rd4, %rd3, %rd4;
    @%p1 cp.async.ca.shared.global [%r27], [%rd4], 4;

    cp.async.commit_group;
    // Publication point for iteration 1's consume: completion + proxy fence
    // + CTA bars (the bars are load-bearing on this hw -- see the tail-sync
    // comment in the KLOOP; the same recipe as the old kernel's prologue).
    // Uniform: with the host's k_blocks % 8 == 0 gate, every warp enters
    // the prologue.
    cp.async.wait_group 0;
    fence.proxy.async.shared::cta;
    bar.sync 0;
    bar.sync 0;

    // -- K-loop: blk = warp_id, warp_id+8, ...  --------------------------------
    mov.u32 %r30, %r5;               // blk = warp_id

KLOOP:
    setp.ge.u32 %p0, %r30, %r1;     // blk >= k_blocks?
    @%p0 bra REDUCE;

    // -- Issue async-load of block (blk+8) into next smem slot FIRST -----
    // (before the consume+compute of this iteration, so the copy overlaps
    // them; the tail's wait_group then rarely stalls).
    // TRIPLE BUFFERING: `next` was last read two iterations ago, and the
    // tail's fences + CTA bars order that read before this overwrite.
    // The tail barrier is CTA-wide, so every warp must run the same number
    // of loop iterations -- the host gates this kernel to k_blocks % 8 == 0.
    add.u32 %r31, %r30, 8;           // next_blk = blk + 8
    setp.lt.u32 %p3, %r31, %r1;     // next_blk < k_blocks?
    @!%p3 bra SKIP_LOAD;

    mad.lo.u32 %r32, %r2, %r1, %r31; // row*k_blocks + next_blk
    mul.lo.u32 %r33, %r32, 212;
    cvt.u64.u32 %rd5, %r33;
    add.u64 %rd5, %rd2, %rd5;         // global ptr to next_blk

    // Round 1 (all 32 lanes): lane l -> word l
    shl.b32 %r34, %r4, 2;
    add.u32 %r35, %r21, %r34;
    cvt.u64.u32 %rd6, %r34;
    add.u64 %rd6, %rd5, %rd6;
    cp.async.ca.shared.global [%r35], [%rd6], 4;

    // Round 2 (lanes 0..20): lane l -> word 32+l
    add.u32 %r36, %r4, 32;
    shl.b32 %r36, %r36, 2;
    add.u32 %r37, %r21, %r36;
    cvt.u64.u32 %rd6, %r36;
    add.u64 %rd6, %rd5, %rd6;
    @%p1 cp.async.ca.shared.global [%r37], [%rd6], 4;

    cp.async.commit_group;

SKIP_LOAD:
    // -- Consume curr -----------------------------------------------------------
    // Data landing in `curr` was completed + published by the PREVIOUS
    // iteration's tail (or the prologue for iteration 1).
    // Every lane reads exclusively the words IT copied (round 1: word
    // `lane`; round 2: word `32+lane`) -- no cross-lane smem reads exist,
    // which is what fixes the old kb=16 residual: cross-lane publication
    // of cp.async data was the one thing the CTA bars could not fully
    // order (~3/81920 wrong, 2026-10-29).  Values held in OTHER lanes'
    // words arrive via shfl.sync.idx REGISTER exchange below (registers
    // carry no proxy semantics; the shuffle's data dependence on the
    // source lane's ld.shared forces that load to complete first).
    // Own words -> registers.
    shl.b32 %r52, %r4, 2;             // lane*4
    add.u32 %r52, %r20, %r52;         // &curr[word lane]
    ld.shared.b32 %r40, [%r52];       // %r40 = Ra (round-1 word)
    add.u32 %r53, %r52, 128;          // &curr[word 32+lane]
    @%p1 ld.shared.b32 %r51, [%r53];  // %r51 = Pa (round-2 word, lanes<21)

    // Register broadcast.  Read-set per lane (q = lane>>2, byte = (lane%4)*8):
    //   d: word 0                (lane 0's Ra)
    //   sc_01..sc_67: words 1..4 (lanes 1..4's Ra)
    //   ql_a/ql_b/ql_c: words 5+q / 13+q / 21+q  (round-1, lanes 5+q/13+q/21+q)
    //   ql_d: word 29+q -- round-1 when q<3, round-2 (copied by lane q-3) else
    //   qh_0/qh_1: words 37+q / 45+q (round-2, lanes 5+q / 13+q)
    // All shfl.sync execute on all 32 lanes (they must be warp-uniform;
    // predication would break the sync); inapplicable sources yield garbage
    // that the selp discards.

    // d (f32 bits) from lane 0
    shfl.sync.idx.b32 %r57, %r40, 0, 0x1F, 0xFFFFFFFF;
    mov.b32 %f1, %r57;                // d

    // scale words 1..4 from lanes 1..4
    shfl.sync.idx.b32 %r41, %r40, 1, 0x1F, 0xFFFFFFFF;  // sc_01
    shfl.sync.idx.b32 %r42, %r40, 2, 0x1F, 0xFFFFFFFF;  // sc_23
    shfl.sync.idx.b32 %r43, %r40, 3, 0x1F, 0xFFFFFFFF;  // sc_45
    shfl.sync.idx.b32 %r44, %r40, 4, 0x1F, 0xFFFFFFFF;  // sc_67

    // ql_a = byte of word 5+q
    add.u32 %r54, %r7, 5;
    shfl.sync.idx.b32 %r53, %r40, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r45, %r53, %r6;
    and.b32 %r45, %r45, 255;          // ql_a

    // ql_b = byte of word 13+q
    add.u32 %r54, %r7, 13;
    shfl.sync.idx.b32 %r53, %r40, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r46, %r53, %r6;
    and.b32 %r46, %r46, 255;          // ql_b

    // ql_c = byte of word 21+q
    add.u32 %r54, %r7, 21;
    shfl.sync.idx.b32 %r53, %r40, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r47, %r53, %r6;
    and.b32 %r47, %r47, 255;          // ql_c

    // ql_d = byte of word 29+q: round-1 (q<3, src lane 29+q) or round-2
    // (q>=3, word 29+q = 32+(q-3), src lane q-3's Pa).  (29+q)&31 gives
    // both source lanes.  The q<3 lanes' Pa shuffle reads lanes 29..31,
    // which never loaded Pa -- garbage, discarded by the selp.
    add.u32 %r54, %r7, 29;
    and.b32 %r54, %r54, 31;
    shfl.sync.idx.b32 %r53, %r40, %r54, 0x1F, 0xFFFFFFFF;  // q<3 value
    shfl.sync.idx.b32 %r55, %r51, %r54, 0x1F, 0xFFFFFFFF;  // q>=3 value
    setp.lt.u32 %p8, %r7, 3;
    selp.b32 %r56, %r53, %r55, %p8;
    shr.u32 %r48, %r56, %r6;
    and.b32 %r48, %r48, 255;          // ql_d

    // qh_0 = byte of word 37+q (round-2, src lane 5+q)
    add.u32 %r54, %r7, 37;
    and.b32 %r54, %r54, 31;
    shfl.sync.idx.b32 %r53, %r51, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r49, %r53, %r6;
    and.b32 %r49, %r49, 255;          // qh_0

    // qh_1 = byte of word 45+q (round-2, src lane 13+q)
    add.u32 %r54, %r7, 45;
    and.b32 %r54, %r54, 31;
    shfl.sync.idx.b32 %r53, %r51, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r50, %r53, %r6;
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

    // -- Tail sync: publish the prefetch issued at the top of THIS iteration --
    // Skipped on the last iteration (no prefetch in flight, nothing to
    // publish) -- uniform across warps because k_blocks % 8 == 0.
    // The bar.sync x3 is REQUIRED, not belt-and-suspenders: with only
    // wait_group + fence.proxy.async (per-thread), iteration >= 2 read
    // UNWRITTEN slots at kb=13/16/40 on this hw with BOTH wait placements,
    // SASS audited correct in both (2026-10-29) -- MEMBAR.ALL.CTA +
    // FENCE.VIEW.ASYNC.S does not publish LDGSTS data to subsequent LDS
    // without a CTA barrier.  (bar.sync lowers to DEFER_BLOCKING on
    // Blackwell: warps continue until an instruction requires completion;
    // the second bar forces the first to complete, the third pins it --
    // SASS-audited 2026-10-29.)
    @!%p3 bra NO_WAIT;
    cp.async.wait_group 0;
    fence.proxy.async.shared::cta;
    bar.sync 0;
    bar.sync 0;
    bar.sync 0;
NO_WAIT:

    // -- Rotate the triple buffers --------------------------------------------
    // curr -> spare -> next -> curr: the slot prefetched this iteration
    // becomes curr; the slot read this iteration becomes spare; the slot
    // read last iteration becomes next (the overwrite target two
    // iterations from its last read).
    mov.u32 %r70, %r20;
    mov.u32 %r20, %r21;
    mov.u32 %r21, %r98;
    mov.u32 %r98, %r70;

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

    // Lane 0 writes to smem[warp_id * 4] with an explicit RELEASE (the
    // arrive's own release semantics proved insufficient for the prior
    // plain store on this hw/ptxas -- the cross-warp handoff read stale
    // partials nondeterministically; bisected 2026-10-29 via a synthetic-
    // weights harness that bypassed the reduce and came back 0/81920).
    setp.eq.u32 %p4, %r4, 0;
    mov.u32 %r73, smem_buf;
    mul.lo.u32 %r74, %r5, 4;
    add.u32 %r74, %r73, %r74;
    @%p4 st.shared.f32 [%r74], %f0;
    @%p4 membar.cta;
    @%p4 red.release.cta.shared.add.u32 [%r99], 1;
    setp.eq.u32 %p5, %r5, 0;
    @!%p5 bra DONE;
REDUCE_WAIT:
    ld.acquire.cta.shared.u32 %r78, [%r99];
    setp.lt.u32 %p9, %r78, 8;
    @%p9 bra REDUCE_WAIT;
    membar.cta;

    // Thread 0 sums the 8 partials sequentially (no shuffles: the
    // 3-step cross-warp butterfly produced nondeterministically wrong
    // sums on this hw -- bisected 2026-10-29).
    setp.eq.u32 %p4, %r3, 0;
    shl.b32 %r76, %r2, 2;
    cvt.u64.u32 %rd8, %r76;
    add.u64 %rd8, %rd0, %rd8;
    mov.f32 %f7, 0f00000000;
    mov.u32 %r77, smem_buf;
    @%p4 ld.shared.f32 %f8, [%r77];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+4];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+8];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+12];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+16];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+20];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+24];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+28];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 st.global.f32 [%rd8], %f7;

DONE:
    ret;
}

// q6k_gemv_sm120_kpar8_db: dual-block DIRECT-LOAD variant (2026-10-29).
//
// The cp.async variant above is barrier-bound: its loop-tail publication
// point needs CTA-wide bar.sync x3 (load-bearing on this hw), which
// locksteps all 8 warps -- every warp waits for the slowest warp's
// prefetch+compute before issuing its next, and the DRAM latency beyond
// compute is exposed. Measured: down_proj [4096,14336] at ~0.86 TB/s while
// the q4k dual_block kernel (no cp.async, no smem, no barriers, L2
// prefetch, free-running warps) does the same shape at ~1.25 TB/s.
//
// This variant adopts that recipe with the register-exchange consume:
//   - weights load DIRECTLY into registers via ld.global.nc (round 1:
//     lane l -> word l; round 2: lanes 0..20 -> words 32+l; 2 warp-
//     instructions per 212-byte block) -- no smem, no cp.async, no proxy
//     fences, and NO barriers in the loop at all
//   - dual-block: blocks (blkA, blkA+8) per iteration -- both blocks'
//     loads issue back-to-back (2x memory-level parallelism), compute A,
//     then compute B (branch-skipped when B is past k_blocks)
//   - prefetch.global.L2 16 blocks ahead (2 lines cover the 212-byte block)
//   - running x pointers (rd7 += 128 per step) instead of per-step address
//     recomputation
// Accumulation order per warp is IDENTICAL to kpar8 (blocks w, w+8,
// w+16, ... in sequence, same 8-step FMA order, same reduce), so results
// are bitwise-equal by construction (validated by the exact-arithmetic
// micro harness). Side benefit: no CTA loop barriers means k_blocks does
// NOT have to be a multiple of 8 (blkB is simply invalid on the ragged
// tail).
.visible .entry q6k_gemv_sm120_kpar8_db(
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
    .reg .u32  %r<110>;
    .reg .u64  %rd<16>;
    .reg .f32  %f<12>;
    .reg .pred %p<16>;

    // [0..31]: 8 f32 partials; [5376..5379]: reduce handoff counter.
    .shared .align 8 .b8 smem_buf[5380];

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

    // byte_shift = (lane % 4) * 8  - for ql/qh byte extraction from u32
    and.b32 %r6, %r4, 3;
    shl.b32 %r6, %r6, 3;            // %r6 = byte_shift

    // lane_div4 = lane >> 2  - for ql/qh word index
    shr.u32 %r7, %r4, 2;            // %r7 = lane_div4

    // sc_shift_even / sc_shift_odd
    shr.u32 %r8, %r4, 4;
    shl.b32 %r8, %r8, 3;            // (lane>>4)*8  (0 or 8)
    add.u32 %r9, %r8, 16;           // (16 or 24)

    // Round-2 predicate: lanes 0..20 load words 32+l.
    setp.lt.u32 %p1, %r4, 21;

    // lane*4 (byte offset of this lane's round-1 word)
    shl.b32 %r34, %r4, 2;

    // -- Accumulator ------------------------------------------------------------
    mov.f32 %f0, 0f00000000;        // acc = 0.0

    // -- Constants --------------------------------------------------------------
    mov.f32 %f10, 0f43000000;       // 128.0f
    mov.f32 %f11, 0f42000000;       //  32.0f

    // -- CTA-wide reduce HANDOFF COUNTER (zeroed by thread 0) -------------------
    mov.u32 %r99, smem_buf;
    add.u32 %r99, %r99, 5376;
    setp.eq.u32 %p9, %r3, 0;
    @%p9 st.shared.u32 [%r99], 0;
    // Double barrier (DEFER_BLOCKING quirk -- see kpar8's comment). These are
    // the ONLY CTA barriers in the kernel; the K-loop is barrier-free.
    bar.sync 0;
    bar.sync 0;

    // -- K-loop: blkA = warp_id, warp_id+16, ...  (blocks A and A+8 per pass) --
    mov.u32 %r30, %r5;               // blkA = warp_id

KLOOP_DB:
    setp.ge.u32 %p0, %r30, %r1;     // blkA >= k_blocks?
    @%p0 bra REDUCE;

    // -- Addresses ---------------------------------------------------------------
    // baseA = packed + (row*k_blocks + blkA) * 212
    mad.lo.u32 %r32, %r2, %r1, %r30;
    mul.lo.u32 %r33, %r32, 212;
    cvt.u64.u32 %rd4, %r33;
    add.u64 %rd4, %rd2, %rd4;

    // blkB = blkA + 8; clamp baseB to baseA when invalid (safe load address:
    // Blackwell traps on OOB even under predication; the loaded garbage is
    // never used -- the whole B section is branch-skipped).
    add.u32 %r31, %r30, 8;
    setp.lt.u32 %p10, %r31, %r1;    // %p10 = B valid
    mad.lo.u32 %r61, %r2, %r1, %r31;
    mul.lo.u32 %r62, %r61, 212;
    cvt.u64.u32 %rd5, %r62;
    add.u64 %rd5, %rd2, %rd5;
    selp.u64 %rd5, %rd5, %rd4, %p10;

    // -- Prefetch the block pair 16 blocks ahead into L2 (2 lines cover the
    // 212-byte block). Clamped to baseA when out of range.
    add.u32 %r63, %r30, 16;
    setp.lt.u32 %p11, %r63, %r1;
    add.u64 %rd6, %rd4, 3392;       // 16 * 212
    selp.u64 %rd6, %rd6, %rd4, %p11;
    prefetch.global.L2 [%rd6];
    add.u64 %rd6, %rd6, 128;
    selp.u64 %rd6, %rd6, %rd4, %p11;
    prefetch.global.L2 [%rd6];

    // -- Load block A + block B words straight into registers -------------------
    // Round 1: lane l -> word l; round 2 (lanes < 21): word 32+l.
    // 4 independent warp-instructions in flight -- the memory-level
    // parallelism the cp.async variant could not reach through its barriers.
    cvt.u64.u32 %rd7, %r34;
    add.u64 %rd7, %rd4, %rd7;
    ld.global.nc.b32 %r40, [%rd7];              // Ra (word lane)
    add.u64 %rd7, %rd7, 128;
    @%p1 ld.global.nc.b32 %r51, [%rd7];         // Pa (word 32+lane)
    cvt.u64.u32 %rd7, %r34;
    add.u64 %rd7, %rd5, %rd7;
    ld.global.nc.b32 %r58, [%rd7];              // Rb (word lane)
    add.u64 %rd7, %rd7, 128;
    @%p1 ld.global.nc.b32 %r59, [%rd7];         // Pb (word 32+lane)

    // -- Register-exchange consume + 8-step dequant/FMA for block A -------------
    // (identical math and order to kpar8's consume; sources are registers.)
    // d (f32 bits) from lane 0's Ra
    shfl.sync.idx.b32 %r57, %r40, 0, 0x1F, 0xFFFFFFFF;
    mov.b32 %f1, %r57;                // d

    // scale words 1..4 from lanes 1..4
    shfl.sync.idx.b32 %r41, %r40, 1, 0x1F, 0xFFFFFFFF;  // sc_01
    shfl.sync.idx.b32 %r42, %r40, 2, 0x1F, 0xFFFFFFFF;  // sc_23
    shfl.sync.idx.b32 %r43, %r40, 3, 0x1F, 0xFFFFFFFF;  // sc_45
    shfl.sync.idx.b32 %r44, %r40, 4, 0x1F, 0xFFFFFFFF;  // sc_67

    // ql_a/ql_b/ql_c bytes (words 5+q / 13+q / 21+q of block A)
    add.u32 %r54, %r7, 5;
    shfl.sync.idx.b32 %r53, %r40, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r45, %r53, %r6;
    and.b32 %r45, %r45, 255;

    add.u32 %r54, %r7, 13;
    shfl.sync.idx.b32 %r53, %r40, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r46, %r53, %r6;
    and.b32 %r46, %r46, 255;

    add.u32 %r54, %r7, 21;
    shfl.sync.idx.b32 %r53, %r40, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r47, %r53, %r6;
    and.b32 %r47, %r47, 255;

    // ql_d: word 29+q -- round-1 (q<3) or round-2 (q>=3, lane q-3's Pa)
    add.u32 %r54, %r7, 29;
    and.b32 %r54, %r54, 31;
    shfl.sync.idx.b32 %r53, %r40, %r54, 0x1F, 0xFFFFFFFF;
    shfl.sync.idx.b32 %r55, %r51, %r54, 0x1F, 0xFFFFFFFF;
    setp.lt.u32 %p8, %r7, 3;
    selp.b32 %r56, %r53, %r55, %p8;
    shr.u32 %r48, %r56, %r6;
    and.b32 %r48, %r48, 255;

    // qh_0 (word 37+q, lane 5+q's Pa) / qh_1 (word 45+q, lane 13+q's Pa)
    add.u32 %r54, %r7, 37;
    and.b32 %r54, %r54, 31;
    shfl.sync.idx.b32 %r53, %r51, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r49, %r53, %r6;
    and.b32 %r49, %r49, 255;

    add.u32 %r54, %r7, 45;
    and.b32 %r54, %r54, 31;
    shfl.sync.idx.b32 %r53, %r51, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r50, %r53, %r6;
    and.b32 %r50, %r50, 255;

    // Running x pointer for block A: x + (blkA*256 + lane)*4; += 128 per step.
    mul.lo.u32 %r60, %r30, 256;
    add.u32 %r60, %r60, %r4;
    shl.b32 %r60, %r60, 2;
    cvt.u64.u32 %rd7, %r60;
    add.u64 %rd7, %rd1, %rd7;

    // Step 0
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 1
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 2
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 3
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 4
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 5
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 6
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 7
    ld.global.nc.f32 %f4, [%rd7];
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
    mul.f32 %f5, %f1, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // -- Block B (skipped entirely when blkB >= k_blocks; uniform per warp) -----
    @!%p10 bra SKIP_B;

    shfl.sync.idx.b32 %r57, %r58, 0, 0x1F, 0xFFFFFFFF;
    mov.b32 %f9, %r57;                // dB

    shfl.sync.idx.b32 %r41, %r58, 1, 0x1F, 0xFFFFFFFF;
    shfl.sync.idx.b32 %r42, %r58, 2, 0x1F, 0xFFFFFFFF;
    shfl.sync.idx.b32 %r43, %r58, 3, 0x1F, 0xFFFFFFFF;
    shfl.sync.idx.b32 %r44, %r58, 4, 0x1F, 0xFFFFFFFF;

    add.u32 %r54, %r7, 5;
    shfl.sync.idx.b32 %r53, %r58, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r45, %r53, %r6;
    and.b32 %r45, %r45, 255;

    add.u32 %r54, %r7, 13;
    shfl.sync.idx.b32 %r53, %r58, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r46, %r53, %r6;
    and.b32 %r46, %r46, 255;

    add.u32 %r54, %r7, 21;
    shfl.sync.idx.b32 %r53, %r58, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r47, %r53, %r6;
    and.b32 %r47, %r47, 255;

    add.u32 %r54, %r7, 29;
    and.b32 %r54, %r54, 31;
    shfl.sync.idx.b32 %r53, %r58, %r54, 0x1F, 0xFFFFFFFF;
    shfl.sync.idx.b32 %r55, %r59, %r54, 0x1F, 0xFFFFFFFF;
    selp.b32 %r56, %r53, %r55, %p8;
    shr.u32 %r48, %r56, %r6;
    and.b32 %r48, %r48, 255;

    add.u32 %r54, %r7, 37;
    and.b32 %r54, %r54, 31;
    shfl.sync.idx.b32 %r53, %r59, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r49, %r53, %r6;
    and.b32 %r49, %r49, 255;

    add.u32 %r54, %r7, 45;
    and.b32 %r54, %r54, 31;
    shfl.sync.idx.b32 %r53, %r59, %r54, 0x1F, 0xFFFFFFFF;
    shr.u32 %r50, %r53, %r6;
    and.b32 %r50, %r50, 255;

    // Running x pointer for block B.
    mul.lo.u32 %r60, %r31, 256;
    add.u32 %r60, %r60, %r4;
    shl.b32 %r60, %r60, 2;
    cvt.u64.u32 %rd7, %r60;
    add.u64 %rd7, %rd1, %rd7;

    // Step 0
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f9, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 1
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f9, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 2
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f9, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 3
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f9, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 4
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f9, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 5
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f9, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 6
    ld.global.nc.f32 %f4, [%rd7];
    add.u64 %rd7, %rd7, 128;
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
    mul.f32 %f5, %f9, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

    // Step 7
    ld.global.nc.f32 %f4, [%rd7];
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
    mul.f32 %f5, %f9, %f2;
    mul.f32 %f5, %f5, %f3;
    fma.rn.f32 %f0, %f5, %f4, %f0;

SKIP_B:
    // -- Advance blkA by 16 (two blocks) and loop --------------------------------
    add.u32 %r30, %r30, 16;
    bra KLOOP_DB;

    // -- Warp butterfly reduction (identical to kpar8) --------------------------
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

    // Cross-warp handoff: st.shared + membar.cta + red.release counter bump /
    // ld.acquire poll; thread 0 sums the 8 partials sequentially.
    setp.eq.u32 %p4, %r4, 0;
    mov.u32 %r73, smem_buf;
    mul.lo.u32 %r74, %r5, 4;
    add.u32 %r74, %r73, %r74;
    @%p4 st.shared.f32 [%r74], %f0;
    @%p4 membar.cta;
    @%p4 red.release.cta.shared.add.u32 [%r99], 1;
    setp.eq.u32 %p5, %r5, 0;
    @!%p5 bra DONE;
REDUCE_WAIT_DB:
    ld.acquire.cta.shared.u32 %r78, [%r99];
    setp.lt.u32 %p9, %r78, 8;
    @%p9 bra REDUCE_WAIT_DB;
    membar.cta;

    setp.eq.u32 %p4, %r3, 0;
    shl.b32 %r76, %r2, 2;
    cvt.u64.u32 %rd8, %r76;
    add.u64 %rd8, %rd0, %rd8;
    mov.f32 %f7, 0f00000000;
    mov.u32 %r77, smem_buf;
    @%p4 ld.shared.f32 %f8, [%r77];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+4];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+8];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+12];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+16];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+20];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+24];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 ld.shared.f32 %f8, [%r77+28];
    @%p4 add.f32 %f7, %f7, %f8;
    @%p4 st.global.f32 [%rd8], %f7;

DONE:
    ret;
}
"#;

/// Kernel entry-point name -- used by `cuModuleGetFunction`.
pub const ENTRY: &str = "q6k_gemv_sm120_kpar8";

/// Dual-block direct-load variant entry (same module, second entry).
pub const ENTRY_DB: &str = "q6k_gemv_sm120_kpar8_db";

/// Minimum compute capability (sm_120 = Blackwell consumer).
pub const MIN_CC: u32 = 1200;
