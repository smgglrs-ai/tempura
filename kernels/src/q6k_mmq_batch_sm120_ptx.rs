/// Hand-written PTX kernel for Q6K x Q8_1 batch GEMM using INT8 tensor cores
/// (sm_120), the llama.cpp-mmq-STRUCTURED variant (2026-11-19/20, ported from
/// the q4k_mmq_batch_sm120 winner): one CTA covers 128 NEURONS x 32 TOKENS,
/// both operands staged through shared memory per k-block via cp.async
/// (SINGLE 37888-byte dynamic-smem buffer -> 2 CTAs/SM co-reside and cover
/// each other's staging latency; see the q4k iteration history in the
/// 2026-11-19 status entry -- double buffering measured slower at 1 CTA/SM).
///
/// # Q6K specifics vs the q4k mmq skeleton
///
/// - Packed GPU format: 53 u32 per 256-element block (u32[0] = f32 super-
///   scale d, converted from the GGUF's f16 so cvt.rn.f16.f32 is EXACT;
///   u32[1..5] = 16 per-16-element scales, bias-128 u8; u32[5..37] = ql
///   (lo 4 bits); u32[37..53] = qh (hi 2 bits)).
/// - PER-16-ELEMENT SCALES vs the k32 mma: each 32-element group gw spans
///   sub-blocks 2g' (k half 0) and 2g'+1 (k half 1) with DIFFERENT scales.
///   Like q6k_mma128, the kernel runs TWO mmas per (gw, token tile) with B
///   zeroed on the other k16 half -- each D is the exact s32 partial sum of
///   ONE sub-block, scaled by its own (d*sc).
/// - Weights staged RAW (ql+qh, 48 words = 192 B per neuron per k-block) and
///   unpacked to s8 A-fragment words at load time with a borrow-free SWAR:
///   for element x = nibble | (qhbits << 4) in [0,63], the s8 byte (x - 32)
///   is computed WORD-wise as
///     v = (((ql >> sh) & 0x0F0F0F0F) | ((((qh >> qsh) & 0x03030303)) << 4))
///         + 0x60606060 ^ 0x80808080
///   (per-byte x + 96 never carries -- x <= 63 -- and the bit-7 toggle maps
///   x + 96 to the two's-complement encoding of x - 32 for both halves of
///   the range). 8 ops per 4-element word instead of ~28 scalar ones.
/// - f16 (d*sc) scale pre-extraction at staging (the product computed in
///   f32 and rounded ONCE to f16 via cvt.rn.f16.f32 -- one RNE rounding
///   per (neuron, sub-block), the same single-rounding class as the q4k
///   mmq's mul.f16 on its f16-exact operands, with no pre-rounding of d
///   so arbitrary f32 d values match the CPU mirror exactly: raw s8 in
///   the mma, exact int32 D, f32 epilogue
///   acc = fma(cvt(D), ax_sub * d8, acc) in (kb, gw, sub) order).
///
/// # Numerics class (DIFFERENT from mma128 -- probe-gated, not bit-identical)
///
/// The only approximation vs the exact-int mma128 class is the single f16
/// rounding of each (d*sc) weight-scale product (2^-11 relative), the class
/// already validated end-to-end on the q4k side (MMLU 62.8%/96.8% agreement,
/// ppl 8.0863). The CPU probe mirror models this op sequence (ulp-level
/// agreement, gate 1e-5); the class band vs the f64-from-quantized reference
/// is gated at 2e-3 (cancellation-aware form), and the end-to-end quality
/// gates (MMLU / ppl) are the real arbiter.
///
/// # Geometry
///
/// Grid:  (n/128, ceil(m/32), 1) -- requires n % 128 == 0 (down_proj 4096,
///        lm_head 128256), m <= 128 (the prefill chunk).
/// Block: (256, 1, 1) -- 8 warps; warp w owns neurons [16w, 16w+16).
/// Per k-block: 8 warps x 8 groups(32k) x 4 token-tiles(8 tokens) x 2 k16
///            halves = 512 mma.sync.m16n8k32 per CTA, 16 f32 accumulators.
///
/// # Shared-memory buffer (37888 bytes, dynamic)
///
///   w_raw  128x48 words = 24576 (ql [0..32) + qh [32..48) per neuron)
///   w_dsc  128x8  half2 = 4096  ((d*sc) f16 per (neuron, sub-block pair))
///   a_qs   32x64 words  = 8192  (Q8_1 quants, group-major per token)
///   a_ds   8x32 words   = 1024  ({d8 f16, rowsum i16} per (group, token))
///
/// # Per-k-block schedule
///
///   bar.sync                 -- all warps done computing on the buffer
///   cp.async staging(kb)     -- raw weights + activations + ds words
///   w_dsc(kb)                -- f16 scale products (register path)
///   cp.async.wait_group 0    -- copies arrived
///   bar.sync                 -- staging visible to all warps
///   compute(kb)              -- 512 mmas + epilogues
pub const MIN_CC: u32 = 1200;
pub const ENTRY: &str = "q6k_mmq_batch_sm120";
/// Dynamic shared memory: one 37888-byte staging buffer.
pub const DYNAMIC_SMEM: usize = 37888;
/// Buffer layout offsets (bytes).
pub const BUF_W_RAW: u32 = 0;
pub const BUF_W_DSC: u32 = 24576;
pub const BUF_A_QS: u32 = 28672;
pub const BUF_A_DS: u32 = 36864;
pub const BUF_BYTES: u32 = 37888;

/// One token-tile mma pair body (jt = 0..4, unrolled by the generator).
/// Uses: %f30 fal0, %f31 fal1, %f32 fah0, %f33 fah1 (sub_lo/sub_hi scale
/// products for neurons gid/gid+8), %r70..73 A fragments, %r99 zero.
const JT_BODY: &str = r#"
        // B fragment (8 tokens x 32k): col = gid -> token jt*8+gid
        add.u32  %r201, %r150, %r60;
        add.u32  %r201, %r201, JT2048;
        ld.shared.b32 %r202, [%r201];
        ld.shared.b32 %r203, [%r201 + 16];
        // ds pair (tokens 2*tid4, 2*tid4+1 of the tile), group-major;
        // q6k needs only the low f16 (d8) -- no min/rowsum term.
        shl.b32  %r204, %r25, 7;           // gw*128 (32 tokens per group)
        add.u32  %r205, %r151, %r204;
        add.u32  %r205, %r205, JT32;
        ld.shared.v2.b32 {%r206, %r207}, [%r205];
        mov.b32 {%h10, %h11}, %r206;        // d8[c0], rowsum[c0]
        mov.b32 {%h12, %h13}, %r207;        // d8[c1], rowsum[c1]
        cvt.f32.f16 %f200, %h10;
        cvt.f32.f16 %f201, %h12;
        // scale products (f32; fal/fah are the f16-pre-extracted values)
        mul.f32 %f210, %f30, %f200;         // P[lo][c0] = fal0 * d8_c0
        mul.f32 %f211, %f30, %f201;         // P[lo][c1]
        mul.f32 %f212, %f31, %f200;         // P[hi][c0] = fal1 * d8_c0
        mul.f32 %f213, %f31, %f201;         // P[hi][c1]
        mul.f32 %f214, %f32, %f200;         // Q[lo][c0] = fah0 * d8_c0
        mul.f32 %f215, %f32, %f201;         // Q[lo][c1]
        mul.f32 %f216, %f33, %f200;         // Q[hi][c0]
        mul.f32 %f217, %f33, %f201;         // Q[hi][c1]
        // MMA #1: k half 0 (sub-block 2g'). B = {b0, 0} -> exact k16 sum.
        mov.u32 %r220, 0; mov.u32 %r221, 0;
        mov.u32 %r222, 0; mov.u32 %r223, 0;
        mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
            {%r220, %r221, %r222, %r223},
            {%r70, %r71, %r72, %r73},
            {%r202, %r99},
            {%r220, %r221, %r222, %r223};
        // epilogue: acc += cvt(D) * P (f32 fma, kb/gw/sub order)
        cvt.rn.f32.s32 %f220, %r220;
        fma.rn.f32 %f{ACC0}, %f220, %f210, %f{ACC0};
        cvt.rn.f32.s32 %f221, %r221;
        fma.rn.f32 %f{ACC1}, %f221, %f211, %f{ACC1};
        cvt.rn.f32.s32 %f222, %r222;
        fma.rn.f32 %f{ACC2}, %f222, %f212, %f{ACC2};
        cvt.rn.f32.s32 %f223, %r223;
        fma.rn.f32 %f{ACC3}, %f223, %f213, %f{ACC3};
        // MMA #2: k half 1 (sub-block 2g'+1). B = {0, b1}.
        mov.u32 %r220, 0; mov.u32 %r221, 0;
        mov.u32 %r222, 0; mov.u32 %r223, 0;
        mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
            {%r220, %r221, %r222, %r223},
            {%r70, %r71, %r72, %r73},
            {%r99, %r203},
            {%r220, %r221, %r222, %r223};
        cvt.rn.f32.s32 %f220, %r220;
        fma.rn.f32 %f{ACC0}, %f220, %f214, %f{ACC0};
        cvt.rn.f32.s32 %f221, %r221;
        fma.rn.f32 %f{ACC1}, %f221, %f215, %f{ACC1};
        cvt.rn.f32.s32 %f222, %r222;
        fma.rn.f32 %f{ACC2}, %f222, %f216, %f{ACC2};
        cvt.rn.f32.s32 %f223, %r223;
        fma.rn.f32 %f{ACC3}, %f223, %f217, %f{ACC3};
"#;

/// cp.async staging of k-block `%r21` (raw Q6K words + Q8_1 activations).
const STAGE_BODY: &str = r#"
        // ---- w_raw words: thread t handles words t + i*256 (24 each;
        //      128 neurons x 48 words = 6144). Word w -> neuron w/48,
        //      word-in-block wi = w%48; global word index
        //      ((n_base + neuron) * k_blocks + kb) * 53 + 5 + wi.
        mov.u32 %r95, 0;
STAGE_W_LOOP:
        setp.ge.u32 %p10, %r95, 24;
        @%p10 bra STAGE_W_DONE;
        mad.lo.u32 %r53, %r95, 256, %r5;   // w
        div.u32  %r54, %r53, 48;           // neuron (local)
        mul.lo.u32 %r55, %r54, 48;
        sub.u32  %r55, %r53, %r55;         // wi
        add.u32 %r57, %r4, %r54;
        mul.lo.u32 %r58, %r57, %r1;
        add.u32 %r58, %r58, %r21;
        mul.lo.u32 %r58, %r58, 53;
        add.u32 %r58, %r58, %r55;
        add.u32 %r58, %r58, 5;
        shl.b32 %r58, %r58, 2;             // word -> byte offset
        cvt.u64.u32 %rd10, %r58;
        add.u64 %rd10, %rd10, %rd0;
        // smem dst: BUF_W_RAW + neuron*192 + wi*4
        mul.lo.u32 %r59, %r54, 192;
        shl.b32  %r230, %r55, 2;
        add.u32 %r59, %r59, %r230;
        add.u32 %r59, %r59, %r28;
        add.u32 %r59, %r59, BUF_W_RAW;
        cp.async.ca.shared.global [%r59], [%rd10], 4;
        add.u32 %r95, %r95, 1;
        bra STAGE_W_LOOP;
STAGE_W_DONE:

        // ---- a_qs words: thread t handles words t + i*256 (8 each)
        mov.u32 %r95, 0;
STAGE_A_LOOP:
        setp.ge.u32 %p13, %r95, 8;
        @%p13 bra STAGE_A_DONE;
        mad.lo.u32 %r101, %r95, 256, %r5;  // w
        shr.u32 %r102, %r101, 6;           // token (local)
        and.b32 %r103, %r101, 63;          // word
        add.u32 %r104, %r40, %r102;        // global row
        shl.b32 %r108, %r102, 8;           // token*256
        shl.b32 %r109, %r103, 2;
        add.u32 %r108, %r108, %r109;
        add.u32 %r108, %r108, %r28;
        add.u32 %r108, %r108, BUF_A_QS;
        setp.ge.u32 %p14, %r104, %r6;      // row >= m -> zero-fill
        mov.u32 %r105, 0;
        @%p14 bra STAGE_A_ZFILL;
        mul.lo.u32 %r106, %r104, %r1;
        shl.b32 %r106, %r106, 6;           // row * k_blocks * 64 words
        shl.b32 %r107, %r21, 6;            // kb*64
        add.u32 %r106, %r106, %r107;
        add.u32 %r106, %r106, %r103;
        shl.b32 %r106, %r106, 2;           // word -> byte
        cvt.u64.u32 %rd12, %r106;
        add.u64 %rd12, %rd12, %rd1;
        cp.async.ca.shared.global [%r108], [%rd12], 4;
        bra STAGE_A_NEXT;
STAGE_A_ZFILL:
        st.shared.b32 [%r108], %r105;
STAGE_A_NEXT:
        add.u32 %r95, %r95, 1;
        bra STAGE_A_LOOP;
STAGE_A_DONE:

        // ---- a_ds word: thread t: token = t>>3, g = t&7
        shr.u32 %r110, %r5, 3;
        and.b32 %r111, %r5, 7;
        add.u32 %r112, %r40, %r110;        // global row
        shl.b32 %r116, %r111, 7;           // g*128 (32 tokens per group)
        shl.b32 %r117, %r110, 2;           // token*4
        add.u32 %r116, %r116, %r117;
        add.u32 %r116, %r116, %r28;
        add.u32 %r116, %r116, BUF_A_DS;
        setp.ge.u32 %p15, %r112, %r6;
        mov.u32 %r113, 0;
        @%p15 bra STAGE_DS_ZFILL;
        mul.lo.u32 %r114, %r112, %r1;
        shl.b32 %r114, %r114, 3;           // row * k_blocks * 8 words
        shl.b32 %r231, %r21, 3;            // kb*8
        add.u32 %r114, %r114, %r231;
        add.u32 %r114, %r114, %r111;
        shl.b32 %r114, %r114, 2;           // word -> byte
        cvt.u64.u32 %rd13, %r114;
        add.u64 %rd13, %rd13, %rd2;
        cp.async.ca.shared.global [%r116], [%rd13], 4;
        bra STAGE_DS_DONE;
STAGE_DS_ZFILL:
        st.shared.b32 [%r116], %r113;
STAGE_DS_DONE:
"#;

/// w_dsc computation + store into the buffer at base `%r28`, from the header
/// loaded by DSC_LOAD into %f0 (d) and %r160..%r163 (4 scale words).
/// Threads 0..127 each own one neuron; word g in 0..8 packs the f16 pair
/// (d*(sc[2g]-128), d*(sc[2g+1]-128)) = sub-blocks (2g, 2g+1).
const DSC_BODY: &str = r#"
    setp.ge.u32 %p11, %r5, 128;
    @%p11 bra DSC_DONE;
    mov.u32 %r170, 0;                       // word g (0..8)
DSC_LOOP:
    setp.ge.u32 %p13, %r170, 8;
    @%p13 bra DSC_DONE;
    // header scale word for word g: subs 2g, 2g+1 live in word (1 + (2g)>>2)
    // = r160 + (g>>1); byte of sub 2g = (2g)&3 = 2*(g&1) (adjacent pair).
    shr.u32 %r171, %r170, 1;                // g>>1
    mov.u32 %r172, %r160;
    setp.eq.u32 %p14, %r171, 1;
    @%p14 mov.u32 %r172, %r161;
    setp.eq.u32 %p15, %r171, 2;
    @%p15 mov.u32 %r172, %r162;
    setp.eq.u32 %p16, %r171, 3;
    @%p16 mov.u32 %r172, %r163;
    and.b32 %r174, %r170, 1;
    shl.b32 %r174, %r174, 4;                // byte bit base: 0 or 16
    shr.u32 %r175, %r172, %r174;
    and.b32 %r176, %r175, 0xFF;  sub.u32 %r176, %r176, 128;  cvt.rn.f32.s32 %f1, %r176;
    mul.f32 %f1, %f0, %f1;      cvt.rn.f16.f32 %h1, %f1;     // ONE f16 rounding
    shr.u32 %r177, %r175, 8;    and.b32 %r177, %r177, 0xFF;  sub.u32 %r177, %r177, 128;
    cvt.rn.f32.s32 %f2, %r177;  mul.f32 %f2, %f0, %f2;      cvt.rn.f16.f32 %h2, %f2;
    mov.b32 %r178, {%h1, %h2};
    shl.b32 %r179, %r5, 5;                  // neuron*32
    shl.b32 %r180, %r170, 2;                // word*4
    add.u32 %r179, %r179, %r180;
    add.u32 %r179, %r179, %r28;
    add.u32 %r179, %r179, BUF_W_DSC;
    st.shared.b32 [%r179], %r178;
    add.u32 %r170, %r170, 1;
    bra DSC_LOOP;
DSC_DONE:
"#;

/// Load the Q6K header words for k-block `%r21` (neuron = tid) into
/// %f0 (d) and %r160..%r163 (scale words). Threads >= 128 skip.
const DSC_LOAD: &str = r#"
    setp.ge.u32 %p12, %r5, 128;
    @%p12 bra DSC_LOAD_DONE;
    add.u32 %r165, %r4, %r5;
    mul.lo.u32 %r166, %r165, %r1;
    add.u32 %r166, %r166, %r21;
    mul.lo.u32 %r166, %r166, 53;
    cvt.u64.u32 %rd11, %r166;
    shl.b64  %rd11, %rd11, 2;
    add.u64 %rd11, %rd11, %rd0;
    ld.global.nc.f32 %f0, [%rd11];
    ld.global.nc.b32 %r160, [%rd11 + 4];
    ld.global.nc.b32 %r161, [%rd11 + 8];
    ld.global.nc.b32 %r162, [%rd11 + 12];
    ld.global.nc.b32 %r163, [%rd11 + 16];
DSC_LOAD_DONE:
"#;

pub fn ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut jt_bodies = String::new();
        for jt in 0..4usize {
            let acc0 = 40 + jt * 4;
            let body = JT_BODY
                .replace("JT2048", &format!("{}", jt * 2048))
                .replace("JT32", &format!("{}", jt * 32))
                .replace("{ACC0}", &format!("{acc0}"))
                .replace("{ACC1}", &format!("{}", acc0 + 1))
                .replace("{ACC2}", &format!("{}", acc0 + 2))
                .replace("{ACC3}", &format!("{}", acc0 + 3));
            jt_bodies.push_str(&body);
        }
        let mut acc_zero = String::new();
        for i in 0..16 {
            acc_zero.push_str(&format!("    mov.f32 %f{}, 0f00000000;\n", 40 + i));
        }
        // writeback: token = token_base + jt*8 + 2*tid4 + (i&1);
        // neuron = n_base + neuron_rel + 8*(i>>1); out[token*n + neuron]
        let mut wb = String::new();
        for jt in 0..4usize {
            for i in 0..4usize {
                let acc = 40 + jt * 4 + i;
                wb.push_str(&format!(
                    "        add.u32 %r300, %r40, {};\n        add.u32 %r300, %r300, %r41;\n        setp.ge.u32 %p30, %r300, %r6;\n        @%p30 bra WB_SKIP_{jt}_{i};\n        add.u32 %r302, %r281, {};\n        mad.lo.u32 %r301, %r300, %r2, %r302;\n        shl.b32 %r301, %r301, 2;\n        cvt.u64.u32 %rd30, %r301;\n        add.u64 %rd30, %rd30, %rd3;\n        st.global.f32 [%rd30], %f{acc};\nWB_SKIP_{jt}_{i}:\n",
                    jt * 8 + (i & 1),
                    8 * (i >> 1),
                    acc = acc,
                    jt = jt,
                    i = i
                ));
            }
        }
        format!(
            r#".version 8.7
.target sm_120
.address_size 64

// q6k_mmq_batch_sm120 -- llama.cpp-mmq-structured Q6K x Q8_1 batch GEMM.
// 128 neurons x 32 tokens per CTA, cp.async smem staging (single 37888-B
// dynamic-smem buffer, 2 CTAs/SM), raw ql/qh staging with a borrow-free
// SWAR s8 unpack at A-fragment load, f16 (d*sc) scale pre-extraction,
// 2-mma k16 split for the per-16-element scales.
// Grid (n/128, ceil(m/32)), block (256,1,1), dynamic smem 37888 bytes.
//
// Parameters:
//   param_weights:  *const u32 -- packed Q6K [N * k_blocks * 53 u32]
//   param_act_qs:   *const u32 -- Q8_1 activations [M * k_blocks * 64 u32]
//   param_act_ds:   *const u32 -- Q8_1 ds words    [M * k_blocks * 8 u32]
//   param_out:      *mut f32   -- output [M * N]
//   param_n:        u32        -- N (multiple of 128)
//   param_k_blocks: u32        -- K = k_blocks * 256
//   param_m:        u32        -- M (actual rows, <= 128)
.extern .shared .align 16 .b8 dyn_smem[];

.visible .entry q6k_mmq_batch_sm120(
    .param .u64 param_weights,
    .param .u64 param_act_qs,
    .param .u64 param_act_ds,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .u32 param_k_blocks,
    .param .u32 param_m
)
{{
    .reg .u32  %r<320>;
    .reg .u64  %rd<40>;
    .reg .f32  %f<240>;
    .reg .pred %p<32>;
    .reg .b16  %h<24>;

    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_act_qs];
    ld.param.u64 %rd2, [param_act_ds];
    ld.param.u64 %rd3, [param_out];
    ld.param.u32 %r2,  [param_n];
    ld.param.u32 %r1,  [param_k_blocks];
    ld.param.u32 %r6,  [param_m];

    mov.u32 %r18, dyn_smem;

    // ---- thread decomposition ---------------------------------------------
    mov.u32 %r5, %tid.x;
    and.b32 %r19, %r5, 31;         // lane
    shr.u32 %r7, %r5, 5;           // warp_id w (0..7)
    and.b32 %r8, %r19, 3;          // tid4 = lane & 3
    shr.u32 %r9, %r19, 2;          // gid = lane >> 2 (0..7)

    // n_base = blockIdx.x * 128; token_base = blockIdx.y * 32
    mov.u32 %r3, %ctaid.x;
    shl.b32 %r4, %r3, 7;           // n_base
    mov.u32 %r39, %ctaid.y;
    shl.b32 %r40, %r39, 5;         // token_base = blockIdx.y * 32

    // per-thread constants:
    // neuron_rel = w*16 + gid
    shl.b32 %r10, %r7, 4;
    add.u32 %r10, %r10, %r9;
    // w_raw row byte offset within the buffer: neuron_rel*192
    mul.lo.u32 %r11, %r10, 192;
    // w_dsc row byte offset: neuron_rel*32
    shl.b32 %r13, %r10, 5;
    // 2*tid4 (writeback col base)
    shl.b32 %r41, %r8, 1;
    // n_base + neuron_rel (writeback neuron base)
    add.u32 %r281, %r4, %r10;
    // zero register (mma B zero halves)
    mov.u32 %r99, 0;
    // SWAR constants
    mov.u32 %r96, 0x0F0F0F0F;
    mov.u32 %r97, 0x03030303;
    mov.u32 %r98, 0x60606060;
    mov.u32 %r94, 0x80808080;

    // ---- zero the 16 f32 accumulators --------------------------------------
{acc_zero}

    // ---- outer loop: k_blocks ----------------------------------------------
    mov.u32 %r21, 0;                       // kb

KB_LOOP:
    setp.ge.u32 %p5, %r21, %r1;
    @%p5 bra KB_DONE;

    // all warps done computing on the buffer before restaging it
    bar.sync 0;

    mov.u32 %r28, 0;                       // single buffer
{stage_loop}
    cp.async.commit_group;
{dsc_load}
{dsc_body}

    cp.async.wait_group 0;
    bar.sync 0;

    // =====================================================================
    // COMPUTE(kb)
    // =====================================================================
    shl.b32 %r60, %r8, 2;                  // goff = tid4*4
    mov.u32 %r25, 0;                       // gw
G_LOOP:
    setp.ge.u32 %p16, %r25, 8;
    @%p16 bra G_DONE;

    // ---- Q6K geometry for this gw (uniform per gw) --------------------------
    // half = gw>>2; g' = gw&3; nibble shift sh = (g'>>1)*4; qh bit shift 2g'.
    shr.u32 %r26, %r25, 2;                 // half
    and.b32 %r27, %r25, 3;                 // g'
    // ql word index (within the neuron's 32 ql words):
    //   half*16 + (g'&1)*8 + tid4   (k half 0); +4 for k half 1.
    shl.b32 %r232, %r26, 4;                // half*16
    and.b32 %r233, %r27, 1;
    shl.b32 %r233, %r233, 3;               // (g'&1)*8
    add.u32 %r234, %r232, %r233;
    add.u32 %r234, %r234, %r8;             // ql word idx
    shl.b32 %r234, %r234, 2;               // -> bytes
    // qh word index (within the neuron's 16 qh words): half*8 + tid4; +4.
    shl.b32 %r235, %r26, 3;                // half*8
    add.u32 %r235, %r235, %r8;
    add.u32 %r235, %r235, 32;              // qh region base (word 32)
    shl.b32 %r235, %r235, 2;               // -> bytes
    // nibble / qh bit shifts
    shr.u32 %r236, %r27, 1;
    shl.b32 %r237, %r236, 2;               // sh = (g'>>1)*4
    shl.b32 %r238, %r27, 1;                // qsh = 2*g'
    // w_dsc word index: half*4 + g'
    shl.b32 %r239, %r26, 2;
    add.u32 %r239, %r239, %r27;
    shl.b32 %r239, %r239, 2;               // -> bytes

    // ---- raw word loads (neurons gid and gid+8; +1536 = 8 neurons*192 B) ----
    add.u32 %r240, %r11, %r234;            // n0 ql byte offset
    add.u32 %r241, %r240, 1536;            // n1 ql byte offset
    add.u32 %r242, %r11, %r235;            // n0 qh byte offset
    add.u32 %r243, %r242, 1536;            // n1 qh byte offset
    ld.shared.b32 %r50, [%r240];
    ld.shared.b32 %r51, [%r241];
    ld.shared.b32 %r52, [%r240 + 16];
    ld.shared.b32 %r53, [%r241 + 16];
    ld.shared.b32 %r54, [%r242];
    ld.shared.b32 %r55, [%r243];
    ld.shared.b32 %r56, [%r242 + 16];
    ld.shared.b32 %r57, [%r243 + 16];

    // ---- SWAR s8 unpack: v = ((ql>>sh)&0F0F.. | (((qh>>qsh)&0303..)<<4))
    //      + 0x60606060 ^ 0x80808080  -> per-byte (x-32) as s8 ---------------
    shr.u32 %r58, %r50, %r237;  and.b32 %r58, %r58, %r96;
    shr.u32 %r59, %r54, %r238;  and.b32 %r59, %r59, %r97;
    shl.b32 %r59, %r59, 4;      or.b32  %r58, %r58, %r59;
    add.u32 %r58, %r58, %r98;   xor.b32 %r70, %r58, %r94;
    shr.u32 %r58, %r51, %r237;  and.b32 %r58, %r58, %r96;
    shr.u32 %r59, %r55, %r238;  and.b32 %r59, %r59, %r97;
    shl.b32 %r59, %r59, 4;      or.b32  %r58, %r58, %r59;
    add.u32 %r58, %r58, %r98;   xor.b32 %r71, %r58, %r94;
    shr.u32 %r58, %r52, %r237;  and.b32 %r58, %r58, %r96;
    shr.u32 %r59, %r56, %r238;  and.b32 %r59, %r59, %r97;
    shl.b32 %r59, %r59, 4;      or.b32  %r58, %r58, %r59;
    add.u32 %r58, %r58, %r98;   xor.b32 %r72, %r58, %r94;
    shr.u32 %r58, %r53, %r237;  and.b32 %r58, %r58, %r96;
    shr.u32 %r59, %r57, %r238;  and.b32 %r59, %r59, %r97;
    shl.b32 %r59, %r59, 4;      or.b32  %r58, %r58, %r59;
    add.u32 %r58, %r58, %r98;   xor.b32 %r73, %r58, %r94;

    // ---- dsc words -> f32 scale values: fal/fah for neurons (gid, gid+8) ---
    add.u32 %r121, %r13, %r239;
    add.u32 %r124, %r121, 256;             // neuron gid+8 row (8*32 B)
    ld.shared.b32 %r122, [%r121 + BUF_W_DSC];        // (fal0, fah0)
    ld.shared.b32 %r123, [%r124 + BUF_W_DSC];        // (fal1, fah1)
    mov.b32 {{%h7, %h8}}, %r122;
    mov.b32 {{%h9, %h14}}, %r123;
    cvt.f32.f16 %f30, %h7;                 // fal0 (sub_lo, neuron gid)
    cvt.f32.f16 %f32, %h8;                 // fah0 (sub_hi, neuron gid)
    cvt.f32.f16 %f31, %h9;                 // fal1 (sub_lo, neuron gid+8)
    cvt.f32.f16 %f33, %h14;                // fah1 (sub_hi, neuron gid+8)

    // per-thread a_qs / a_ds row bases
    shl.b32 %r152, %r9, 8;
    add.u32 %r150, %r152, BUF_A_QS;
    shl.b32 %r153, %r8, 3;                 // 2*tid4*4
    add.u32 %r151, %r153, BUF_A_DS;

{jt_bodies}

    // next gw
    add.u32 %r60, %r60, 32;
    add.u32 %r25, %r25, 1;
    bra G_LOOP;
G_DONE:

    add.u32 %r21, %r21, 1;
    bra KB_LOOP;
KB_DONE:

    // ---- writeback ---------------------------------------------------------
    // token = token_base + jt*8 + 2*tid4 + (i&1);
    // neuron = n_base + neuron_rel + 8*(i>>1); out[token*n + neuron]
{wb}

    ret;
}}"#,
            acc_zero = acc_zero,
            stage_loop = STAGE_BODY,
            dsc_load = DSC_LOAD,
            dsc_body = DSC_BODY,
            jt_bodies = jt_bodies,
            wb = wb,
        )
        .replace("BUF_W_RAW", &BUF_W_RAW.to_string())
        .replace("BUF_W_DSC", &BUF_W_DSC.to_string())
        .replace("BUF_A_QS", &BUF_A_QS.to_string())
        .replace("BUF_A_DS", &BUF_A_DS.to_string())
    })
}

use std::sync::OnceLock;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_generates() {
        let p = ptx();
        if let Ok(path) = std::env::var("TEMPURA_DUMP_PTX") {
            std::fs::write(&path, p).unwrap();
        }
        assert!(p.contains(".visible .entry q6k_mmq_batch_sm120"));
        // 4 jt bodies, each with the 2-mma k16 split (g loop is runtime)
        assert_eq!(p.matches("mma.sync.aligned.m16n8k32").count(), 8);
        assert_eq!(p.matches("st.global.f32").count(), 16);
        // single-buffered staging: 3 textual cp.async sites (runtime loops)
        assert_eq!(p.matches("cp.async.ca.shared.global").count(), 3);
        assert_eq!(p.matches("cp.async.commit_group").count(), 1);
        assert!(p.contains(".extern .shared .align 16 .b8 dyn_smem[]"));
        // the SWAR unpack constants appear once each (outside the loops)
        assert!(p.contains("0x60606060"));
        assert!(p.contains("0x80808080"));
    }
}
