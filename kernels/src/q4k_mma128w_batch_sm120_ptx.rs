/// Hand-tuned PTX kernel for Q4K x Q8_1 batch GEMM using INT8 tensor cores
/// (sm_120), WIDE-neuron LARGE-m variant of `q4k_mma128_batch_sm120`.
///
/// # What changed vs q4k_mma128_batch_sm120 (and why)
///
/// The 2026-11-18 profile of the exact-mode prefill showed q4k_mma128 at 77%
/// of the 128-token step. Its block covers only 8 output neurons, so the Q8_1
/// activation A-fragments are re-read from L2 once per 8 neurons -- at m=128
/// that is ~14x the weight bytes in L2 traffic (gate/up: ~917 MB L2 vs 66 MB
/// weights), and every k-block pays a CTA-wide bar.sync for the header stage.
/// llama.cpp's mmq (the existence proof at 7335 tok/s) tiles I=128 neurons
/// per block and stages the activation tile in smem ONCE for all of them.
///
/// This variant keeps the m16n8k32 fragment mapping, Q4K parsing, and
/// per-element fp op sequence of the mma128 kernel, but assigns each WARP
/// ownership of 8 neurons (block = 8 warps = 64 neurons):
/// - grid.x = N/64 (8x fewer blocks, 8x less A re-reading),
/// - the 8 sub-blocks (mw) per k-block are processed sequentially by the
///   owning warp instead of striped across 8 warps,
/// - headers are staged in WARP-PRIVATE smem (named barrier per warp, no
///   CTA-wide bar.sync),
/// - the cross-warp 8-partial smem reduction is GONE (each warp's register
///   accumulators are final) -- the epilogue stores directly to global.
///
/// # Numerics (IMPORTANT -- differs from mma128)
///
/// The per-output-element accumulation order is sequential over (kb, mw) by
/// ONE warp, not striped-then-8-way-reduced. The per-element fp op sequence
/// is identical, but the summation tree differs, so outputs are NOT
/// bitwise-equal to `q4k_mma_batch_sm120` / `q4k_mma128_batch_sm120`; they
/// match the per-row kpar8 reference to <= 1e-5 max rel err (the same gate
/// standard as q4k_mma_batch / q4k_streamk_batch -- pure f32 sum-order
/// noise, measured 1e-8-class on those kernels). Probe arm: `mma128w`.
///
/// # Row splitting (grid.y)
///
/// m <= 128 = 8 m16 tiles. For small N the N/64 grid would underfill the
/// GPU, so the host may split the row range: grid.y blocks each compute
/// `row_span` rows starting at ctaid.y * row_span (row_span = 128/grid.y,
/// a multiple of 16). Tiles outside the block's span or past m are skipped
/// by uniform branch. Total activation traffic is independent of grid.y
/// (each block reads only its row_span of A).
pub const MIN_CC: u32 = 1200;
pub const ENTRY: &str = "q4k_mma128w_batch_sm120";

use std::sync::OnceLock;

pub fn ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut tiles = String::new();
        for t in 0..8u32 {
            tiles.push_str(&tile_body(t));
        }
        let mut epilogue = String::new();
        for t in 0..8u32 {
            epilogue.push_str(&epilogue_tile(t));
        }
        format!(
            r#".version 8.7
.target sm_120
.address_size 64

// q4k_mma128w_batch_sm120  (wide-neuron variant; see module docs)
.visible .entry q4k_mma128w_batch_sm120(
    .param .u64 param_weights,
    .param .u64 param_act_qs,
    .param .u64 param_act_ds,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .u32 param_k_blocks,
    .param .u32 param_m,
    .param .u32 param_row_span
)
{{
    .reg .u32  %r<200>;
    .reg .u64  %rd<40>;
    .reg .f32  %f<140>;
    .reg .pred %p<32>;
    .reg .b16  %h<8>;

    // Warp-private header regions, 256 B each:
    //   +0..31   d8[8]     f32
    //   +32..63  dmin8[8]  f32
    //   +64..127 sc8[8 neurons][8 sub-blocks] u8
    //   +128..191 mv8[8 neurons][8 sub-blocks] u8
    .shared .align 4 .b8 smem_buf[2048];

    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_act_qs];
    ld.param.u64 %rd2, [param_act_ds];
    ld.param.u64 %rd3, [param_out];
    ld.param.u32 %r0,  [param_n];
    ld.param.u32 %r1,  [param_k_blocks];
    ld.param.u32 %r2,  [param_m];
    ld.param.u32 %r13, [param_row_span];

    mov.u32 %r3, %ctaid.x;
    shl.b32 %r4, %r3, 6;          // n_tile_base = blockIdx.x * 64
    mov.u32 %r5, %ctaid.y;
    mul.lo.u32 %r12, %r5, %r13;   // row0 = blockIdx.y * row_span

    mov.u32 %r5, %tid.x;
    and.b32 %r6, %r5, 31;         // lane
    shr.u32 %r7, %r5, 5;          // warp_id (0..7)
    and.b32 %r8, %r6, 3;          // tid = lane & 3
    shr.u32 %r9, %r6, 2;          // gid = lane >> 2 (0..7)
    shl.b32 %r141, %r8, 1;        // col0 = 2*tid (D column within warp's 8)

    // This warp's neuron octet base (global neuron index).
    shl.b32 %r14, %r7, 3;
    add.u32 %r10, %r4, %r14;      // warp_neuron_base

    add.u32 %r11, %r7, 1;         // named barrier id (1..8), warp-private

    mov.u32 %r18, smem_buf;
    shl.b32 %r15, %r7, 8;
    add.u32 %r19, %r18, %r15;     // warp smem region base

    setp.eq.u32 %p1, %r8, 0;      // header loader lanes (tid==0, 8 lanes)

    shl.b32 %r145, %r8, 2;        // tid*4 (k byte offset in 16-byte half)
    cvt.u64.u32 %rd17, %r145;

    // ---- Zero the 32 register accumulators (8 tiles x 4 D) -------------------
    mov.f32 %f100, 0f00000000;
    mov.f32 %f101, 0f00000000;
    mov.f32 %f102, 0f00000000;
    mov.f32 %f103, 0f00000000;
    mov.f32 %f104, 0f00000000;
    mov.f32 %f105, 0f00000000;
    mov.f32 %f106, 0f00000000;
    mov.f32 %f107, 0f00000000;
    mov.f32 %f108, 0f00000000;
    mov.f32 %f109, 0f00000000;
    mov.f32 %f110, 0f00000000;
    mov.f32 %f111, 0f00000000;
    mov.f32 %f112, 0f00000000;
    mov.f32 %f113, 0f00000000;
    mov.f32 %f114, 0f00000000;
    mov.f32 %f115, 0f00000000;
    mov.f32 %f116, 0f00000000;
    mov.f32 %f117, 0f00000000;
    mov.f32 %f118, 0f00000000;
    mov.f32 %f119, 0f00000000;
    mov.f32 %f120, 0f00000000;
    mov.f32 %f121, 0f00000000;
    mov.f32 %f122, 0f00000000;
    mov.f32 %f123, 0f00000000;
    mov.f32 %f124, 0f00000000;
    mov.f32 %f125, 0f00000000;
    mov.f32 %f126, 0f00000000;
    mov.f32 %f127, 0f00000000;
    mov.f32 %f128, 0f00000000;
    mov.f32 %f129, 0f00000000;
    mov.f32 %f130, 0f00000000;
    mov.f32 %f131, 0f00000000;

    // ---- Outer loop: k_blocks Q4K blocks ------------------------------------
    mov.u32 %r21, 0;               // kb = 0

KB_LOOP:
    setp.ge.u32 %p5, %r21, %r1;
    @%p5 bra KB_DONE;

    // ---- Header load: lane group tid==0, neuron = warp_neuron_base + gid ----
    @!%p1 bra AFTER_HEADER;

    add.u32 %r22, %r10, %r9;       // neuron
    mul.lo.u32 %r22, %r22, %r1;
    mul.lo.u32 %r22, %r22, 144;
    mad.lo.u32 %r22, %r21, 144, %r22;
    cvt.u64.u32 %rd6, %r22;
    add.u64 %rd6, %rd0, %rd6;

    ld.global.nc.b16 %h0, [%rd6 + 0];
    ld.global.nc.b16 %h1, [%rd6 + 2];
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;

    ld.global.nc.b32 %r24, [%rd6 + 4];
    ld.global.nc.b32 %r25, [%rd6 + 8];
    ld.global.nc.b32 %r26, [%rd6 + 12];

    and.b32 %r27, %r24, 63;
    shr.u32 %r28, %r24, 8;  and.b32 %r28, %r28, 63;
    shr.u32 %r29, %r24, 16; and.b32 %r29, %r29, 63;
    shr.u32 %r30, %r24, 24; and.b32 %r30, %r30, 63;
    and.b32 %r31, %r25, 63;
    shr.u32 %r32, %r25, 8;  and.b32 %r32, %r32, 63;
    shr.u32 %r33, %r25, 16; and.b32 %r33, %r33, 63;
    shr.u32 %r34, %r25, 24; and.b32 %r34, %r34, 63;
    and.b32 %r35, %r26, 255;
    shr.u32 %r36, %r26, 8;  and.b32 %r36, %r36, 255;
    and.b32 %r37, %r24, 192; shr.u32 %r37, %r37, 2;
    shr.u32 %r38, %r24, 8;  and.b32 %r38, %r38, 192; shr.u32 %r38, %r38, 2;
    and.b32 %r39, %r35, 15;  or.b32 %r39, %r39, %r37;
    and.b32 %r40, %r36, 15;  or.b32 %r40, %r40, %r38;
    shr.u32 %r41, %r35, 4;  and.b32 %r41, %r41, 15;
    and.b32 %r42, %r25, 192; shr.u32 %r42, %r42, 2;
    or.b32  %r41, %r41, %r42;
    shr.u32 %r43, %r36, 4;  and.b32 %r43, %r43, 15;
    shr.u32 %r44, %r25, 8;  and.b32 %r44, %r44, 192; shr.u32 %r44, %r44, 2;
    or.b32  %r43, %r43, %r44;
    shr.u32 %r45, %r26, 16; and.b32 %r45, %r45, 255;
    shr.u32 %r46, %r26, 24; and.b32 %r46, %r46, 255;
    shr.u32 %r47, %r24, 16; and.b32 %r47, %r47, 192; shr.u32 %r47, %r47, 2;
    shr.u32 %r48, %r24, 24; and.b32 %r48, %r48, 192; shr.u32 %r48, %r48, 2;
    and.b32 %r49, %r45, 15; or.b32 %r49, %r49, %r47;
    and.b32 %r50, %r46, 15; or.b32 %r50, %r50, %r48;
    shr.u32 %r51, %r45, 4;  and.b32 %r51, %r51, 15;
    shr.u32 %r52, %r25, 16; and.b32 %r52, %r52, 192; shr.u32 %r52, %r52, 2;
    or.b32  %r51, %r51, %r52;
    shr.u32 %r53, %r46, 4;  and.b32 %r53, %r53, 15;
    shr.u32 %r54, %r25, 24; and.b32 %r54, %r54, 192; shr.u32 %r54, %r54, 2;
    or.b32  %r53, %r53, %r54;

    // warp-private smem: d8[gid], dmin8[gid], sc8[gid*8+0..7], mv8[gid*8+0..7]
    shl.b32 %r55, %r9, 2;
    add.u32 %r57, %r19, %r55;
    st.shared.f32 [%r57 + 0],  %f0;
    st.shared.f32 [%r57 + 32], %f1;

    shl.b32 %r58, %r9, 3;
    add.u32 %r58, %r58, 64;
    add.u32 %r58, %r19, %r58;
    st.shared.u8 [%r58 + 0], %r27;
    st.shared.u8 [%r58 + 1], %r28;
    st.shared.u8 [%r58 + 2], %r29;
    st.shared.u8 [%r58 + 3], %r30;
    st.shared.u8 [%r58 + 4], %r39;
    st.shared.u8 [%r58 + 5], %r40;
    st.shared.u8 [%r58 + 6], %r49;
    st.shared.u8 [%r58 + 7], %r50;
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
    // CTA-scope bar (NOT bar.warp.sync): racecheck cannot model warp-scope
    // barriers and flags the header smem handoff as hazards; the CTA bar is
    // what q4k_mma128 uses and measured cost-neutral (2026-11-18 wash).
    bar.sync 0;

    // ---- Inner loop: all 8 sub-blocks of this k-block (warp owns them) ------
    mov.u32 %r60, 0;               // mw = 0

MW_LOOP:
    setp.ge.u32 %p6, %r60, 8;
    @%p6 bra MW_DONE;

    mul.lo.u32 %r61, %r21, 8;
    add.u32    %r61, %r61, %r60;   // global_sub = kb*8 + mw
    shl.b32    %r63, %r61, 5;
    cvt.u64.u32 %rd14, %r63;       // qs block byte offset (u64)
    shl.b32    %r68, %r61, 2;
    cvt.u64.u32 %rd25, %r68;       // ds block byte offset (u64)

    // ---- Load Q4K nibbles for B (neuron = warp_neuron_base + gid) -----------
    shr.u32 %r72, %r60, 1;         // byte group j = mw>>1
    shl.b32 %r72, %r72, 5;

    add.u32 %r74, %r10, %r9;       // neuron
    mul.lo.u32 %r74, %r74, %r1;
    mul.lo.u32 %r74, %r74, 144;
    mad.lo.u32 %r74, %r21, 144, %r74;
    add.u32    %r74, %r74, 16;
    add.u32    %r74, %r74, %r72;
    add.u32    %r75, %r74, %r145;
    add.u32    %r76, %r75, 16;

    cvt.u64.u32 %rd9,  %r75; add.u64 %rd9,  %rd0, %rd9;
    cvt.u64.u32 %rd10, %r76; add.u64 %rd10, %rd0, %rd10;
    ld.global.nc.b32 %r77, [%rd9];
    ld.global.nc.b32 %r78, [%rd10];

    and.b32 %r130, %r60, 1;
    setp.ne.u32 %p7, %r130, 0;
    mov.u32 %r79, 0x0F0F0F0F;
    and.b32 %r81, %r77, %r79;
    shr.u32 %r82, %r77, 4; and.b32 %r82, %r82, %r79;
    selp.u32 %r83, %r82, %r81, %p7;
    and.b32 %r84, %r78, %r79;
    shr.u32 %r85, %r78, 4; and.b32 %r85, %r85, %r79;
    selp.u32 %r86, %r85, %r84, %p7;

    // ---- Per-column Q4K scales (cols col0, col0+1) from warp smem -----------
    shl.b32 %r146, %r141, 2;
    add.u32 %r147, %r19, %r146;
    ld.shared.f32 %f3,  [%r147];
    ld.shared.f32 %f4,  [%r147 + 4];
    ld.shared.f32 %f7,  [%r147 + 32];
    ld.shared.f32 %f8,  [%r147 + 36];

    shl.b32 %r148, %r141, 3;
    add.u32 %r149, %r19, %r148;
    add.u32 %r149, %r149, 64;
    add.u32 %r149, %r149, %r60;
    ld.shared.u8 %r93,  [%r149];
    ld.shared.u8 %r94,  [%r149 + 8];
    add.u32 %r150, %r149, 64;
    ld.shared.u8 %r98,  [%r150];
    ld.shared.u8 %r99,  [%r150 + 8];

    cvt.rn.f32.u32 %f11, %r93;
    cvt.rn.f32.u32 %f12, %r94;
    cvt.rn.f32.u32 %f15, %r98;
    cvt.rn.f32.u32 %f16, %r99;

    // Tile-invariant products: (d*sc) and (dmin*mv) per column.
    mul.f32 %f70, %f3, %f11;       // (d*sc)[col0]
    mul.f32 %f71, %f4, %f12;       // (d*sc)[col0+1]
    mul.f32 %f72, %f7, %f15;       // (dmin*mv)[col0]
    mul.f32 %f73, %f8, %f16;       // (dmin*mv)[col0+1]

{tiles}
MW_NEXT:
    add.u32 %r60, %r60, 1;
    bra MW_LOOP;

MW_DONE:
    // Required (REAL race without it, found by racecheck): k-block kb+1's
    // header store must not overwrite the warp smem tables while other
    // warps still read kb's (same double-bar-per-kb as q4k_mma128).
    bar.sync 0;
    add.u32 %r21, %r21, 1;
    bra KB_LOOP;

KB_DONE:

{epilogue}

DONE:
    ret;
}}
"#
        )
    })
}

/// Per-(kb, mw) m16 tile body: A-fragment loads + ds (rowsum) + mma + f32
/// accumulate. Same op sequence as q4k_mma128_batch_sm120's tile bodies.
fn tile_body(t: u32) -> String {
    let base = t * 16;
    let f = 100 + 4 * t;
    format!(
        r#"    // ---- tile {t}: rows {base}..{e} ------------------------------------
    setp.ge.u32 %p30, {base}, %r13;          // tile outside this block's row span
    add.u32  %r152, %r12, {base};
    setp.ge.u32 %p31, %r152, %r2;            // tile start past m
    or.pred  %p20, %p30, %p31;
    @%p20 bra TILE{t}_END;

    add.u32 %r150, %r12, {fl};
    add.u32 %r150, %r150, %r9;     // row_lo = row0 + {fl} + gid
    add.u32 %r151, %r150, 8;       // row_hi = row_lo + 8
    setp.lt.u32 %p21, %r150, %r2;     // row_lo valid
    setp.lt.u32 %p22, %r151, %r2;     // row_hi valid

    // A qs addresses: x_qs + row*k_blocks*256 + (kb*8+mw)*32 + tid*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd20, %r152;  add.u64 %rd20, %rd20, %rd1;  add.u64 %rd20, %rd20, %rd14;
    add.u64 %rd21, %rd20, %rd17;      // a0 addr (row_lo, k half 0)
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 8;
    cvt.u64.u32 %rd22, %r152;  add.u64 %rd22, %rd22, %rd1;  add.u64 %rd22, %rd22, %rd14;
    add.u64 %rd23, %rd22, %rd17;      // a1 addr (row_hi, k half 0)

    mov.u32 %r153, 0; mov.u32 %r154, 0; mov.u32 %r155, 0; mov.u32 %r156, 0;
    @%p21 ld.global.nc.b32 %r153, [%rd21];
    @%p22 ld.global.nc.b32 %r154, [%rd23];
    @%p21 ld.global.nc.b32 %r155, [%rd21 + 16];
    @%p22 ld.global.nc.b32 %r156, [%rd23 + 16];

    // d8 scales: x_ds + row*k_blocks*32 + (kb*8+mw)*4
    mul.lo.u32 %r152, %r150, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd24, %r152;  add.u64 %rd24, %rd24, %rd2;  add.u64 %rd24, %rd24, %rd25;
    mul.lo.u32 %r152, %r151, %r1;  shl.b32 %r152, %r152, 5;
    cvt.u64.u32 %rd26, %r152;  add.u64 %rd26, %rd26, %rd2;  add.u64 %rd26, %rd26, %rd25;
    // Q8_1 ds word: low f16 = d8, high s16 = the block's exact integer
    // q8 rowsum (precomputed by quantize_q8_1; same integer the inline
    // bytesum computed -> bit-identical numerics, ~40 fewer ops/tile).
    mov.u32 %r165, 0;
    mov.u32 %r166, 0;
    @%p21 ld.global.nc.b32 %r165, [%rd24];
    @%p22 ld.global.nc.b32 %r166, [%rd26];
    mov.b32 {{%h2, %h3}}, %r165;        // h2 = d8 row_lo, h3 = rowsum row_lo (s16)
    mov.b32 {{%h4, %h5}}, %r166;        // h4 = d8 row_hi, h5 = rowsum row_hi (s16)
    cvt.f32.f16 %f60, %h2;
    cvt.f32.f16 %f61, %h4;
    cvt.s32.s16 %r157, %h3;
    cvt.s32.s16 %r159, %h5;

    // Per-tile scale factors: pos/min = (d*sc | dmin*mv) * d8[row].
    mul.f32 %f62, %f70, %f60;   // pos row-lo col0 = (d*sc)[col0] * d8_lo
    mul.f32 %f63, %f71, %f60;   // pos row-lo col1
    mul.f32 %f64, %f70, %f61;   // pos row-hi col0
    mul.f32 %f65, %f71, %f61;   // pos row-hi col1
    mul.f32 %f66, %f72, %f60;   // min row-lo col0 = (dmin*mv)[col0] * d8_lo
    mul.f32 %f67, %f73, %f60;   // min row-lo col1
    mul.f32 %f68, %f72, %f61;   // min row-hi col0
    mul.f32 %f69, %f73, %f61;   // min row-hi col1

    cvt.rn.f32.s32 %f74, %r157;      // rowsum(row_lo)
    cvt.rn.f32.s32 %f75, %r159;      // rowsum(row_hi)

    // MMA: D = A * B (fresh accumulators)
    mov.u32 %r161, 0; mov.u32 %r162, 0;
    mov.u32 %r163, 0; mov.u32 %r164, 0;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {{%r161, %r162, %r163, %r164}},
        {{%r153, %r154, %r155, %r156}},
        {{%r83,  %r86}},
        {{%r161, %r162, %r163, %r164}};

    // acc[T] += D * pos - rowsum * min  (same op order as the mma128 kernel)
    cvt.rn.f32.s32 %f76, %r161;
    mul.f32 %f76, %f76, %f62;   mul.f32 %f77, %f74, %f66;   sub.f32 %f76, %f76, %f77;
    add.f32 %f{f0}, %f{f0}, %f76;

    cvt.rn.f32.s32 %f76, %r162;
    mul.f32 %f76, %f76, %f63;   mul.f32 %f77, %f74, %f67;   sub.f32 %f76, %f76, %f77;
    add.f32 %f{f1}, %f{f1}, %f76;

    cvt.rn.f32.s32 %f76, %r163;
    mul.f32 %f76, %f76, %f64;   mul.f32 %f77, %f75, %f68;   sub.f32 %f76, %f76, %f77;
    add.f32 %f{f2}, %f{f2}, %f76;

    cvt.rn.f32.s32 %f76, %r164;
    mul.f32 %f76, %f76, %f65;   mul.f32 %f77, %f75, %f69;   sub.f32 %f76, %f76, %f77;
    add.f32 %f{f3}, %f{f3}, %f76;

TILE{t}_END:

"#,
        t = t,
        base = base,
        e = base + 15,
        fl = base,
        f0 = f,
        f1 = f + 1,
        f2 = f + 2,
        f3 = f + 3,
    )
}

/// Epilogue: direct global stores (no cross-warp reduction -- each warp's
/// register accumulators are final for its 8 neurons x row_span rows).
fn epilogue_tile(t: u32) -> String {
    let base = t * 16;
    let f = 100 + 4 * t;
    format!(
        r#"    // ---- tile {t} stores ----
    setp.ge.u32 %p30, {base}, %r13;
    add.u32  %r152, %r12, {base};
    setp.ge.u32 %p31, %r152, %r2;
    or.pred  %p20, %p30, %p31;
    @%p20 bra TILE{t}_SDONE;

    add.u32 %r150, %r12, {fl};
    add.u32 %r150, %r150, %r9;     // row_lo = row0 + {fl} + gid
    add.u32 %r151, %r150, 8;       // row_hi = row_lo + 8
    setp.lt.u32 %p21, %r150, %r2;
    setp.lt.u32 %p22, %r151, %r2;

    mad.lo.u32 %r106, %r150, %r0, %r10;
    add.u32    %r106, %r106, %r141;
    shl.b32    %r106, %r106, 2;
    cvt.u64.u32 %rd11, %r106;
    add.u64 %rd11, %rd3, %rd11;
    @%p21 st.global.f32 [%rd11 + 0], %f{f0};
    @%p21 st.global.f32 [%rd11 + 4], %f{f1};

    mad.lo.u32 %r107, %r151, %r0, %r10;
    add.u32    %r107, %r107, %r141;
    shl.b32    %r107, %r107, 2;
    cvt.u64.u32 %rd16, %r107;
    add.u64 %rd16, %rd3, %rd16;
    @%p22 st.global.f32 [%rd16 + 0], %f{f2};
    @%p22 st.global.f32 [%rd16 + 4], %f{f3};

TILE{t}_SDONE:

"#,
        t = t,
        base = base,
        fl = base,
        f0 = f,
        f1 = f + 1,
        f2 = f + 2,
        f3 = f + 3,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn ptx_generates() {
        let ptx = super::ptx();
        assert!(ptx.contains(".entry q4k_mma128w_batch_sm120"));
        assert!(ptx.contains("TILE0_END"));
        assert!(ptx.contains("TILE7_SDONE"));
        assert!(ptx.matches("mma.sync.aligned.m16n8k32").count() == 8);
        assert!(ptx.matches("bar.sync 0").count() == 2);
        // 32 accumulator zeroing movs + 32 store refs
        assert!(ptx.matches("%f131, 0f00000000").count() == 1);
    }
}
