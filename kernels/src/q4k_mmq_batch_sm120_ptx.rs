/// Hand-written PTX kernel for Q4K x Q8_1 batch GEMM using INT8 tensor cores
/// (sm_120), the llama.cpp-mmq-STRUCTURED variant (2026-11-19 restructure):
/// one CTA covers 128 NEURONS x 32 TOKENS, both operands staged through
/// shared memory per k-block via cp.async (SINGLE buffer: 29696 bytes, so
/// THREE CTAs co-reside per SM and cover each other's staging latency --
/// the earlier variants measured the kernel latency-bound: single-buffer
/// register staging 3635 tok/s pp128 at 2 CTAs/SM, double-buffered 59392 B
/// at 1 CTA/SM 3938 tok/s, J=64 3318 tok/s (grid tail); see the 2026-11-19
/// status entry for the full iteration history).
///
/// Weights are staged RAW (packed nibbles, 16 KiB per k-block) and the
/// nibble planes are unpacked at A-fragment load time (+~6 bit-ops per
/// (warp, group), amortized over 4 token-tile mmas). Buffer layout:
///   w_qs_raw 128x32 words = 16384 | w_dm 128x8 half2 = 4096 |
///   a_qs 32x64 words = 8192 | a_ds 8x32 words = 1024   -> 29696 bytes of
/// DYNAMIC shared memory (opt-in above the 48 KiB static limit is not
/// needed at this size, but the dynamic form matches the q6k batch
/// pattern and keeps the PTX free of a static buffer).
///
/// # Numerics class (DIFFERENT from mma128 -- probe-gated, not bit-identical)
///
/// The weight-side scale products (d*sc) and (dmin*mv) are computed in f16
/// at staging time (`mul.f16`, one RNE rounding each -- llama.cpp mmq's
/// load-time half2 pre-extraction class). Everything else matches the
/// mma128 class: raw nibbles in the mma (exact int32 C), exact integer q8
/// rowsum from the Q8_1 ds word, f32 scale application and f32 accumulation
/// across k-blocks in kb order. The CPU probe mirror models this op
/// sequence (ulp-level agreement, measured ~3e-8); the class band vs the
/// per-row kpar8 reference is ~5e-5 (the f16 scale rounding, 2^-11
/// relative), gated at 2e-3, and the end-to-end quality gates (MMLU / ppl)
/// are the real arbiter.
///
/// # Geometry
///
/// Grid:  (n/128, ceil(m/32), 1) -- requires n % 128 == 0 (all prefill
///        shapes: 4096, 14336), m <= 128 (the prefill chunk).
/// Block: (256, 1, 1) -- 8 warps; warp w owns neurons [16w, 16w+16).
/// Per k-block: 8 warps x 8 sub-blocks(32k) x 4 token-tiles(8 tokens)
///            = 256 mma.sync.m16n8k32 per CTA, 16 f32 accumulators/thread.
///
/// # Per-k-block schedule
///
///   bar.sync                 -- all warps done computing on the buffer
///   cp.async staging(kb)     -- raw weights + activations + ds words
///   w_dm(kb)                 -- f16 scale products (register path)
///   cp.async.wait_group 0    -- copies arrived
///   bar.sync                 -- staging visible to all warps
///   compute(kb)              -- 256 mmas + epilogues
pub const MIN_CC: u32 = 1200;
pub const ENTRY: &str = "q4k_mmq_batch_sm120";
/// Split-reduce entry (same module): out[i] = sum_s partials[s][i], s in
/// fixed order 0..splits-1 (deterministic combine for grid.z k-splits).
pub const ENTRY_REDUCE: &str = "q4k_mmq_split_reduce";
/// Dynamic shared memory: one 29696-byte staging buffer.
pub const DYNAMIC_SMEM: usize = 29696;
/// Buffer layout offsets (bytes).
pub const BUF_W_QS: u32 = 0;
pub const BUF_W_DM: u32 = 16384;
pub const BUF_A_QS: u32 = 20480;
pub const BUF_A_DS: u32 = 28672;
pub const BUF_BYTES: u32 = 29696;

/// One token-tile mma body (jt = 0..4, unrolled by the generator).
const JT_BODY: &str = r#"
        // B fragment (8 tokens x 32k): col = gid -> token jt*8+gid
        add.u32  %r201, %r150, %r60;
        add.u32  %r201, %r201, JT2048;
        ld.shared.b32 %r202, [%r201];
        ld.shared.b32 %r203, [%r201 + 16];
        // ds pair (tokens 2*tid4, 2*tid4+1 of the tile), group-major
        shl.b32  %r204, %r25, 7;           // g*128 (32 tokens per group)
        add.u32  %r205, %r151, %r204;
        add.u32  %r205, %r205, JT32;
        ld.shared.v2.b32 {%r206, %r207}, [%r205];
        mov.b32 {%h10, %h11}, %r206;        // d8[c0], rowsum[c0]
        mov.b32 {%h12, %h13}, %r207;        // d8[c1], rowsum[c1]
        cvt.f32.f16 %f200, %h10;
        cvt.f32.f16 %f201, %h12;
        cvt.s32.s16 %r208, %h11;
        cvt.rn.f32.s32 %f202, %r208;
        cvt.s32.s16 %r209, %h13;
        cvt.rn.f32.s32 %f203, %r209;
        // scale products (f32; ax/ay are the f16-pre-extracted values)
        mul.f32 %f210, %f30, %f200;         // pos[lo][c0] = ax_lo * d8_c0
        mul.f32 %f211, %f30, %f201;         // pos[lo][c1]
        mul.f32 %f212, %f31, %f200;         // pos[hi][c0] = ax_hi * d8_c0
        mul.f32 %f213, %f31, %f201;         // pos[hi][c1]
        mul.f32 %f214, %f200, %f202;        // tb[c0] = d8_c0 * rs_c0
        mul.f32 %f215, %f201, %f203;        // tb[c1]
        mul.f32 %f216, %f32, %f214;         // min[lo][c0] = ay_lo * tb_c0
        mul.f32 %f217, %f32, %f215;         // min[lo][c1]
        mul.f32 %f218, %f33, %f214;         // min[hi][c0]
        mul.f32 %f219, %f33, %f215;         // min[hi][c1]
        // MMA: D = A (16 neurons x 32k) * B (8 tokens x 32k)
        mov.u32 %r220, 0; mov.u32 %r221, 0;
        mov.u32 %r222, 0; mov.u32 %r223, 0;
        mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
            {%r220, %r221, %r222, %r223},
            {%r70, %r71, %r72, %r73},
            {%r202, %r203},
            {%r220, %r221, %r222, %r223};
        // epilogue: acc += D * pos + min (f32 fma + add, kb order)
        cvt.rn.f32.s32 %f220, %r220;
        fma.rn.f32 %f{ACC0}, %f220, %f210, %f{ACC0};
        add.f32 %f{ACC0}, %f{ACC0}, %f216;
        cvt.rn.f32.s32 %f221, %r221;
        fma.rn.f32 %f{ACC1}, %f221, %f211, %f{ACC1};
        add.f32 %f{ACC1}, %f{ACC1}, %f217;
        cvt.rn.f32.s32 %f222, %r222;
        fma.rn.f32 %f{ACC2}, %f222, %f212, %f{ACC2};
        add.f32 %f{ACC2}, %f{ACC2}, %f218;
        cvt.rn.f32.s32 %f223, %r223;
        fma.rn.f32 %f{ACC3}, %f223, %f213, %f{ACC3};
        add.f32 %f{ACC3}, %f{ACC3}, %f219;
"#;

/// cp.async staging of k-block `%r21` into the buffer at base `%r28`.
const STAGE_BODY: &str = r#"
        // ---- w_qs raw words: thread t handles words t + i*256 (16 each)
        mov.u32 %r95, 0;
STAGE_W_LOOP:
        setp.ge.u32 %p10, %r95, 16;
        @%p10 bra STAGE_W_DONE;
        mad.lo.u32 %r53, %r95, 256, %r5;   // w
        shr.u32 %r54, %r53, 5;             // neuron (local)
        and.b32 %r55, %r53, 31;            // wi
        // global word offset: ((n_base + neuron) * k_blocks + kb) * 36 + 4 + wi
        add.u32 %r57, %r4, %r54;
        mul.lo.u32 %r58, %r57, %r1;
        add.u32 %r58, %r58, %r21;
        mul.lo.u32 %r58, %r58, 36;
        add.u32 %r58, %r58, %r55;
        add.u32 %r58, %r58, 4;
        shl.b32 %r58, %r58, 2;             // word -> byte offset
        cvt.u64.u32 %rd10, %r58;
        add.u64 %rd10, %rd10, %rd0;
        // smem dst: %r28 + neuron*128 + wi*4
        shl.b32 %r59, %r54, 7;
        shl.b32 %r230, %r55, 2;
        add.u32 %r59, %r59, %r230;
        add.u32 %r59, %r59, %r28;
        add.u32 %r59, %r59, BUF_W_QS;
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

/// w_dm computation + store into the buffer at base `%r28`, from the
/// header words in %r160 (d|dmin), %r161 (s0), %r162 (s1), %r163 (s2).
/// Thread t covers neuron t>>1, groups [4*(t&1), +4).
const WDM_BODY: &str = r#"
    mov.b32 {%h0, %h1}, %r160;             // d, dmin
    xor.b16 %h2, %h1, 0x8000;              // -dmin
    shr.u32 %r168, %r5, 1;                 // neuron (local)
    and.b32 %r169, %r5, 1;
    shl.b32 %r170, %r169, 2;               // g0 = half*4
    mov.u32 %r171, 0;
WDM_LOOP:
    setp.ge.u32 %p11, %r171, 4;
    @%p11 bra WDM_DONE;
    add.u32 %r232, %r170, %r171;
    setp.ge.u32 %p12, %r232, 4;
    @!%p12 bra WDM_LOW;
    sub.u32 %r173, %r232, 4;
    shl.b32 %r174, %r173, 3;               // i*8
    shr.b32 %r175, %r163, %r174;
    and.b32 %r175, %r175, 0x0F;
    shr.b32 %r176, %r161, %r174;
    and.b32 %r176, %r176, 0xC0;
    shr.u32 %r176, %r176, 2;
    or.b32  %r177, %r175, %r176;           // sc_g
    shr.b32 %r178, %r163, %r174;
    shr.u32 %r178, %r178, 4;
    and.b32 %r178, %r178, 0x0F;
    shr.b32 %r179, %r162, %r174;
    and.b32 %r179, %r179, 0xC0;
    shr.u32 %r179, %r179, 2;
    or.b32  %r180, %r178, %r179;           // mv_g
    bra WDM_HAVE;
WDM_LOW:
    shl.b32 %r174, %r232, 3;               // g*8
    shr.b32 %r177, %r161, %r174;
    and.b32 %r177, %r177, 63;
    shr.b32 %r180, %r162, %r174;
    and.b32 %r180, %r180, 63;
WDM_HAVE:
    cvt.rn.f16.u32 %h3, %r177;
    cvt.rn.f16.u32 %h4, %r180;
    mul.f16 %h5, %h0, %h3;
    mul.f16 %h6, %h2, %h4;
    mov.b32 %r181, {%h5, %h6};
    shl.b32 %r182, %r168, 5;               // neuron*32
    shl.b32 %r183, %r232, 2;
    add.u32 %r182, %r182, %r183;
    add.u32 %r182, %r182, %r28;
    add.u32 %r182, %r182, BUF_W_DM;
    st.shared.b32 [%r182], %r181;
    add.u32 %r171, %r171, 1;
    bra WDM_LOOP;
WDM_DONE:
"#;

/// Load the w_dm header words for k-block `%r21` (neuron = tid>>1) into
/// %r160..%r163.
const WDM_LOAD: &str = r#"
    shr.u32 %r164, %r5, 1;                 // neuron (local)
    add.u32 %r165, %r4, %r164;
    mul.lo.u32 %r166, %r165, %r1;
    add.u32 %r166, %r166, %r21;
    mul.lo.u32 %r166, %r166, 144;
    cvt.u64.u32 %rd11, %r166;
    add.u64 %rd11, %rd11, %rd0;
    ld.global.nc.b32 %r160, [%rd11];       // d | dmin
    ld.global.nc.b32 %r161, [%rd11 + 4];   // s0
    ld.global.nc.b32 %r162, [%rd11 + 8];   // s1
    ld.global.nc.b32 %r163, [%rd11 + 12];  // s2
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
        // When splits > 1 (grid.z k-split), the partial goes to
        // partials[z*m*n + token*n + neuron]; a fixed-order reduce kernel
        // combines them. splits == 1 keeps the direct out store.
        let mut wb = String::new();
        for jt in 0..4usize {
            for i in 0..4usize {
                let acc = 40 + jt * 4 + i;
                wb.push_str(&format!(
                    "        add.u32 %r300, %r40, {tok};\n        add.u32 %r300, %r300, %r41;\n        setp.ge.u32 %p30, %r300, %r6;\n        @%p30 bra WB_SKIP_{jt}_{i};\n        add.u32 %r302, %r281, {neur};\n        mad.lo.u32 %r301, %r300, %r2, %r302;\n        shl.b32 %r301, %r301, 2;\n        cvt.u64.u32 %rd30, %r301;\n        setp.ne.u32 %p31, %r47, 1;\n        @%p31 bra WB_SPLIT_{jt}_{i};\n        add.u64 %rd30, %rd30, %rd3;\n        bra WB_STORE_{jt}_{i};\nWB_SPLIT_{jt}_{i}:\n        add.u64 %rd30, %rd30, %rd31;\n        add.u64 %rd30, %rd30, %rd4;\nWB_STORE_{jt}_{i}:\n        st.global.f32 [%rd30], %f{acc};\nWB_SKIP_{jt}_{i}:\n",
                    tok = jt * 8 + (i & 1),
                    neur = 8 * (i >> 1),
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

// q4k_mmq_batch_sm120 -- llama.cpp-mmq-structured Q4K x Q8_1 batch GEMM.
// 128 neurons x 32 tokens per CTA, cp.async smem staging (single buffer,
// 3 CTAs/SM), raw-weight staging with load-time nibble unpack, f16 scale
// pre-extraction. Grid (n/128, ceil(m/32)), block (256,1,1), dynamic smem
// 29696 bytes.
//
// Parameters:
//   param_weights:  *const u32 -- native GGUF Q4K [N * k_blocks * 36 u32]
//   param_act_qs:   *const u32 -- Q8_1 activations [M * k_blocks * 64 u32]
//   param_act_ds:   *const u32 -- Q8_1 ds words    [M * k_blocks * 8 u32]
//   param_out:      *mut f32   -- output [M * N]
//   param_n:        u32        -- N (multiple of 128)
//   param_k_blocks: u32        -- K = k_blocks * 256
//   param_m:        u32        -- M (actual rows, <= 128)
// Dynamic shared memory (the 29696-byte staging buffer); module-scope
// extern, sized at launch via cuLaunchKernel's sharedMemBytes.
.extern .shared .align 16 .b8 dyn_smem[];

.visible .entry q4k_mmq_batch_sm120(
    .param .u64 param_weights,
    .param .u64 param_act_qs,
    .param .u64 param_act_ds,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .u32 param_k_blocks,
    .param .u32 param_m,
    .param .u32 param_splits,
    .param .u64 param_partials
)
{{
    .reg .u32  %r<320>;
    .reg .u64  %rd<40>;
    .reg .f32  %f<240>;
    .reg .pred %p<32>;
    .reg .b16  %h<16>;

    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_act_qs];
    ld.param.u64 %rd2, [param_act_ds];
    ld.param.u64 %rd3, [param_out];
    ld.param.u32 %r2,  [param_n];
    ld.param.u32 %r1,  [param_k_blocks];
    ld.param.u32 %r6,  [param_m];
    ld.param.u32 %r47, [param_splits];
    ld.param.u64 %rd4, [param_partials];

    mov.u32 %r18, dyn_smem;

    // ---- k-split range (grid.z): CTA z owns k-blocks [z*kb/S, (z+1)*kb/S).
    // splits == 1 (z == 0) covers [0, kb) -- the unsplit path, numerically
    // identical to the pre-split kernel.
    mov.u32 %r42, %ctaid.z;
    mul.lo.u32 %r44, %r42, %r1;
    div.u32 %r45, %r44, %r47;             // kb_lo
    add.u32 %r46, %r42, 1;
    mul.lo.u32 %r46, %r46, %r1;
    div.u32 %r46, %r46, %r47;             // kb_hi
    // partials tile offset (bytes): z * m * n * 4
    mul.lo.u32 %r43, %r6, %r2;            // m*n (u32: <= 128*14336)
    cvt.u64.u32 %rd31, %r42;
    cvt.u64.u32 %rd32, %r43;
    mul.lo.u64 %rd31, %rd31, %rd32;
    shl.b64 %rd31, %rd31, 2;

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
    // dm word offset = neuron_rel*32 (within w_dm)
    shl.b32 %r13, %r10, 5;
    // raw weight row byte offset within w_qs: neuron_rel*128
    shl.b32 %r11, %r10, 7;
    // 2*tid4 (writeback col base)
    shl.b32 %r41, %r8, 1;
    // tid4*4 (A-fragment raw-word byte offset within a group pair)
    shl.b32 %r239, %r8, 2;
    // n_base + neuron_rel (writeback neuron base)
    add.u32 %r281, %r4, %r10;

    // ---- zero the 16 f32 accumulators --------------------------------------
{acc_zero}

    // ---- outer loop: k_blocks (this CTA's split range) ---------------------
    mov.u32 %r21, %r45;                   // kb = kb_lo

KB_LOOP:
    setp.ge.u32 %p5, %r21, %r46;
    @%p5 bra KB_DONE;

    // all warps done computing on the buffer before restaging it
    bar.sync 0;

    mov.u32 %r28, 0;                       // single buffer
{stage_loop}
    cp.async.commit_group;
{wdm_load}
{wdm_loop}

    cp.async.wait_group 0;
    bar.sync 0;

    // =====================================================================
    // COMPUTE(kb)
    // =====================================================================
    mov.u32 %r29, 0;                       // buffer base (dyn smem)
    shl.b32 %r60, %r8, 2;                  // goff = tid4*4
    mov.u32 %r25, 0;                       // g
G_LOOP:
    setp.ge.u32 %p16, %r25, 8;
    @%p16 bra G_DONE;

    // A fragments: raw weight words, nibble planes unpacked here.
    // word(k, row): w_qs[neuron][ (g>>1)*8 + tid4 (+4) ] -- the word index
    // depends on j = g>>1 only (both nibble planes share the word).
    shr.u32 %r26, %r25, 1;                 // j = g>>1
    shl.b32 %r27, %r26, 5;                 // j*32 bytes
    add.u32 %r233, %r29, %r11;             // buf + neuron_rel*128
    add.u32 %r233, %r233, %r27;
    add.u32 %r233, %r233, %r239;           // + tid4*4
    ld.shared.b32 %r235, [%r233];
    ld.shared.b32 %r236, [%r233 + 1024];   // neuron+8 row
    ld.shared.b32 %r237, [%r233 + 16];
    ld.shared.b32 %r238, [%r233 + 1040];
    and.b32 %r234, %r25, 1;
    setp.eq.u32 %p17, %r234, 0;
    mov.u32 %r99, 0x0F0F0F0F;
    @%p17 bra PLANE_LO;
    shr.u32 %r70, %r235, 4; and.b32 %r70, %r70, %r99;
    shr.u32 %r71, %r236, 4; and.b32 %r71, %r71, %r99;
    shr.u32 %r72, %r237, 4; and.b32 %r72, %r72, %r99;
    shr.u32 %r73, %r238, 4; and.b32 %r73, %r73, %r99;
    bra PLANE_DONE;
PLANE_LO:
    and.b32 %r70, %r235, %r99;
    and.b32 %r71, %r236, %r99;
    and.b32 %r72, %r237, %r99;
    and.b32 %r73, %r238, %r99;
PLANE_DONE:

    // dm words -> f32 scale values: ax/ay for neurons (gid, gid+8)
    add.u32 %r121, %r29, %r13;             // buf + neuron_rel*32
    shl.b32 %r120, %r25, 2;                // g*4
    add.u32 %r121, %r121, %r120;
    add.u32 %r121, %r121, BUF_W_DM;
    ld.shared.b32 %r122, [%r121];          // (ax_lo, ay_lo) half2
    ld.shared.b32 %r123, [%r121 + 256];    // (ax_hi, ay_hi)
    mov.b32 {{%h7, %h8}}, %r122;
    mov.b32 {{%h9, %h14}}, %r123;
    cvt.f32.f16 %f30, %h7;                 // ax_lo
    cvt.f32.f16 %f32, %h8;                 // ay_lo
    cvt.f32.f16 %f31, %h9;                 // ax_hi
    cvt.f32.f16 %f33, %h14;                // ay_hi

    // per-thread a_qs / a_ds row bases
    shl.b32 %r152, %r9, 8;
    add.u32 %r150, %r29, %r152;
    add.u32 %r150, %r150, BUF_A_QS;
    shl.b32 %r153, %r8, 3;                 // 2*tid4*4
    add.u32 %r151, %r29, %r153;
    add.u32 %r151, %r151, BUF_A_DS;

{jt_bodies}

    // next g
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
}}

// Deterministic split combine: out[i] = sum_s partials[s*total + i], s in
// fixed order 0..splits-1 (f32 adds, ascending s). Grid
// (ceil(total/256),1,1), block (256,1,1).
.visible .entry q4k_mmq_split_reduce(
    .param .u64 param_out,
    .param .u64 param_partials,
    .param .u32 param_total,
    .param .u32 param_splits
)
{{
    .reg .u32  %r<12>;
    .reg .u64  %rd<12>;
    .reg .f32  %f<4>;
    .reg .pred %p<4>;

    ld.param.u64 %rd0, [param_out];
    ld.param.u64 %rd1, [param_partials];
    ld.param.u32 %r2,  [param_total];
    ld.param.u32 %r3,  [param_splits];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r4, 256, %r5;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra RED_DONE;

    cvt.u64.u32 %rd2, %r6;
    shl.b64 %rd2, %rd2, 2;

    mov.u32 %r8, 0;
    mov.f32 %f0, 0f00000000;
RED_LOOP:
    setp.ge.u32 %p2, %r8, %r3;
    @%p2 bra RED_FIN;
    mul.wide.u32 %rd3, %r8, %r2;
    shl.b64 %rd3, %rd3, 2;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd4, %rd4, %rd2;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f0, %f0, %f1;
    add.u32 %r8, %r8, 1;
    bra RED_LOOP;
RED_FIN:
    add.u64 %rd5, %rd0, %rd2;
    st.global.f32 [%rd5], %f0;
RED_DONE:
    ret;
}}
"#,
            acc_zero = acc_zero,
            wdm_load = WDM_LOAD,
            stage_loop = STAGE_BODY,
            wdm_loop = WDM_BODY,
            jt_bodies = jt_bodies,
            wb = wb,
        )
        .replace("BUF_W_QS", &BUF_W_QS.to_string())
        .replace("BUF_W_DM", &BUF_W_DM.to_string())
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
        assert!(p.contains(".visible .entry q4k_mmq_batch_sm120"));
        // split-reduce entry in the same module
        assert!(p.contains(".visible .entry q4k_mmq_split_reduce"));
        // 4 jt bodies (the g loop is a runtime loop: 8 g x 4 jt at runtime)
        assert_eq!(p.matches("mma.sync.aligned.m16n8k32").count(), 4);
        assert_eq!(p.matches("st.global.f32").count(), 17); // 16 wb + 1 reduce
        // single-buffered staging: one stage body; 3 textual cp.async sites
        // (the word loops are runtime loops)
        assert_eq!(p.matches("cp.async.ca.shared.global").count(), 3);
        assert_eq!(p.matches("cp.async.commit_group").count(), 1);
        assert!(p.contains(".extern .shared .align 16 .b8 dyn_smem[]"));
    }
}
