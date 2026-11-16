/// Scatter-write kernel for the KV cache: copies `kv_dim` f32 elements from
/// a source tensor into the KV cache at the row given by a device-side offset.
///
/// The device pointer `write_off_ptr` is captured once in the CUDA graph and
/// never changes.  The *value* at that address is updated H->D each step (4
/// bytes).  Because the kernel reads the offset at execution time there are no
/// `cuGraphExecMemcpyNodeSetParams` calls -- the graph is topology-stable.
///
/// A batch variant (`kv_scatter_write_batch`) handles M simultaneous sequences
/// using `blockIdx.y` as the sequence index; it reads per-sequence source and
/// cache-base pointers from device-resident pointer arrays.
use crate::{KernelError, Result};
use cuda_core::sys;

// PTX module containing both the single-sequence and M-sequence scatter-write
// kernels.  Both entries share the same .version / .target / .address_size header.
//
// Single-sequence kernel (kv_scatter_write):
//   Grid:  ceil(kv_dim / BLOCK_SIZE) blocks.
//   Block: BLOCK_SIZE threads.
//   Params: src (u64), cache_base (u64), write_off_ptr (u64 -> *const i32), kv_dim (u32).
//
// Batch kernel (kv_scatter_write_batch):
//   Grid:  (ceil(kv_dim / BLOCK_SIZE), M, 1).
//   Block: BLOCK_SIZE threads.
//   Params: src_ptrs (*const u64 [M]), cache_bases (*const u64 [M]),
//           write_off_ptr (u64 -> *const i32 -- single offset shared by all seqs),
//           kv_dim (u32), m (u32).
//
// f16 twin (kv_scatter_write_f16): same addressing as the f32 entry, but
// the f32 source element is converted with cvt.rn.f16.f32 (IEEE
// round-to-nearest-even, subnormals preserved -- no .ftz) and stored as a
// b16 element.  This must stay BIT-IDENTICAL to the generic KV append
// path's cast kernel (cutile_cast_f32_f16, convert_tile f32->f16): a
// session decoded alone through the captured graph must produce the same
// pool bytes as the same session decoded inside a batch, which appends
// via cast + D2D.  Parity is probe-gated in bench-decode-ops ("kv
// scatter-write dtype twins") against both the cast kernel and the CPU
// half::f16::from_f32 reference on adversarial data.
//
// Params: identical to kv_scatter_write (src is f32, cache_base is a u16
// element array).
//
// e4m3 twin (kv_scatter_write_e4m3): scalar PTX port of the ONE shared
// implementation tempura_kernels::e4m3::f32_to_e4m3_bits (which the
// cutile_cast_f32_e4m3 tile kernel also mirrors bit-for-bit): RNE on the
// 20 dropped mantissa bits, saturation to +-448, subnormal flush (f32
// exponent < 121 -> signed zero), NaN -> 0x7F.  Stored as a u8 element.
// The min/max clamp runs on the ORIGINAL f32 (NaN lanes are overridden by
// the is_nan select at the end, matching the cutile kernel's select
// order); the round-up carry (kept == 7 -> mantissa 0, exponent +1) can
// only reach e2 = 16 from values >= 464, which the 448 clamp made
// unreachable.

// Row-gather kernel (d2d_gather_row):
//   Grid:  ceil(row_len / BLOCK_SIZE) blocks.
//   Block: BLOCK_SIZE threads.
//   Params: src (u64), dst (u64), pos_ptr (u64 -> *const i32), row_len (u32),
//           max_pos (u32).
//   Copies row `pos` of a row-major [max_pos x row_len] f32 table into dst.
//   `pos` is read from DEVICE memory at execution time, so a CUDA graph that
//   captures this kernel stays correct when the position changes per step
//   (same pattern as the write-offset read above). Used to refresh the RoPE
//   cos/sin row without re-capturing the decode graph.
//
// PTX ISA 8.7 / sm_120 (Blackwell consumer, RTX 5090).
const PTX_SOURCE: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// Single-sequence entry
.visible .entry kv_scatter_write(
    .param .u64 param_src,
    .param .u64 param_cache_base,
    .param .u64 param_write_off_ptr,
    .param .u32 param_kv_dim
)
{
    // %r0 = kv_dim  %r1 = ctaid.x  %r2 = ntid.x  %r3 = global tid  %r4 = write_off
    // %rd0 = src  %rd1 = cache_base  %rd2 = write_off_ptr
    // %rd3 = row  %rd4 = kv_dim64  %rd5 = tid64
    // %rd6 = dst_elem  %rd7 = dst_byte  %rd8 = src_byte
    .reg .u32   %r<8>;
    .reg .u64   %rd<10>;
    .reg .f32   %f1;
    .reg .pred  %p1;

    ld.param.u64 %rd0, [param_src];
    ld.param.u64 %rd1, [param_cache_base];
    ld.param.u64 %rd2, [param_write_off_ptr];
    ld.param.u32 %r0,  [param_kv_dim];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r3, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r3, %r0;
    @%p1 bra done_single;

    ld.global.s32 %r4,  [%rd2];
    cvt.u64.s32   %rd3, %r4;
    cvt.u64.u32   %rd4, %r0;
    cvt.u64.u32   %rd5, %r3;

    mul.lo.u64  %rd6, %rd3, %rd4;
    add.u64     %rd6, %rd6, %rd5;

    shl.b64  %rd7, %rd6, 2;
    add.u64  %rd7, %rd1, %rd7;

    shl.b64  %rd8, %rd5, 2;
    add.u64  %rd8, %rd0, %rd8;

    ld.global.f32 %f1, [%rd8];
    st.global.f32 [%rd7], %f1;

done_single:
    ret;
}

// f16 twin: dst element index = row * kv_dim + tid, byte address = 2 * elem.
.visible .entry kv_scatter_write_f16(
    .param .u64 param_src,
    .param .u64 param_cache_base,
    .param .u64 param_write_off_ptr,
    .param .u32 param_kv_dim
)
{
    .reg .u32   %r<8>;
    .reg .u64   %rd<10>;
    .reg .b16   %h1;
    .reg .f32   %f1;
    .reg .pred  %p1;

    ld.param.u64 %rd0, [param_src];
    ld.param.u64 %rd1, [param_cache_base];
    ld.param.u64 %rd2, [param_write_off_ptr];
    ld.param.u32 %r0,  [param_kv_dim];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r3, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r3, %r0;
    @%p1 bra done_f16;

    ld.global.s32 %r4,  [%rd2];
    cvt.u64.s32   %rd3, %r4;
    cvt.u64.u32   %rd4, %r0;
    cvt.u64.u32   %rd5, %r3;

    mul.lo.u64  %rd6, %rd3, %rd4;
    add.u64     %rd6, %rd6, %rd5;

    shl.b64  %rd7, %rd6, 1;
    add.u64  %rd7, %rd1, %rd7;

    shl.b64  %rd8, %rd5, 2;
    add.u64  %rd8, %rd0, %rd8;

    ld.global.f32  %f1, [%rd8];
    cvt.rn.f16.f32  %h1, %f1;
    st.global.b16  [%rd7], %h1;

done_f16:
    ret;
}

// e4m3 twin: scalar port of e4m3::f32_to_e4m3_bits; dst byte = elem index.
// %r9 = x bits  %r10 = sign8  %r11 = abs  %r12 = clamped bits  %r13 = exp
// %r14 = is_norm  %r15 = man/dropped  %r16 = kept  %r17 = round_up
// %r18 = eq_half/carry  %r19 = is_nan  %r6/%r7 = scratch  %f2 = clamped x
.visible .entry kv_scatter_write_e4m3(
    .param .u64 param_src,
    .param .u64 param_cache_base,
    .param .u64 param_write_off_ptr,
    .param .u32 param_kv_dim
)
{
    .reg .u32   %r<20>;
    .reg .u64   %rd<10>;
    .reg .f32   %f<3>;
    .reg .pred  %p1;

    ld.param.u64 %rd0, [param_src];
    ld.param.u64 %rd1, [param_cache_base];
    ld.param.u64 %rd2, [param_write_off_ptr];
    ld.param.u32 %r0,  [param_kv_dim];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r3, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r3, %r0;
    @%p1 bra done_e4m3;

    ld.global.s32 %r4,  [%rd2];
    cvt.u64.s32   %rd3, %r4;
    cvt.u64.u32   %rd4, %r0;
    cvt.u64.u32   %rd5, %r3;

    mul.lo.u64  %rd6, %rd3, %rd4;
    add.u64     %rd6, %rd6, %rd5;
    add.u64     %rd7, %rd1, %rd6;          // dst byte addr (1 byte/elem)

    shl.b64  %rd8, %rd5, 2;
    add.u64  %rd8, %rd0, %rd8;
    ld.global.f32 %f1, [%rd8];

    // NaN detect on the ORIGINAL bits: abs > inf-bits.
    mov.b32 %r9, %f1;
    and.b32 %r11, %r9, 0x7FFFFFFF;
    setp.gt.u32 %p1, %r11, 0x7F800000;
    selp.b32 %r19, 1, 0, %p1;              // is_nan

    // Clamp to [-448, 448] (E4M3FN has no infinity).  NaN lanes are
    // overridden by the is_nan select at the end (same order as the
    // cutile tile kernel's select chain).  448.0 = 0x43E00000.
    max.f32 %f2, %f1, 0fC3E00000;          // -448.0
    min.f32 %f2, %f2, 0f43E00000;          //  448.0

    mov.b32 %r12, %f2;
    and.b32 %r11, %r12, 0x7FFFFFFF;        // abs (clamped)
    shr.u32 %r13, %r11, 23;                // exp
    shr.u32 %r10, %r12, 24;
    and.b32 %r10, %r10, 0x80;              // sign8

    // Normal iff f32 exponent > 120 (2^-6 = smallest e4m3 normal);
    // below -> signed zero (subnormal flush).
    setp.gt.u32 %p1, %r13, 120;
    selp.b32 %r14, 1, 0, %p1;              // is_norm

    and.b32 %r15, %r11, 0x007FFFFF;        // man
    shr.u32 %r16, %r15, 20;                // kept
    and.b32 %r15, %r15, 0x000FFFFF;        // dropped

    // RNE: round up when dropped > half, or == half and kept LSB is 1.
    // gt_half and eq_half&odd are disjoint -> sum them.
    setp.gt.u32 %p1, %r15, 0x00080000;
    selp.b32 %r17, 1, 0, %p1;              // gt_half
    setp.eq.u32 %p1, %r15, 0x00080000;
    selp.b32 %r18, 1, 0, %p1;              // eq_half
    and.b32 %r6, %r16, 1;                  // kept LSB
    and.b32 %r6, %r6, %r18;                // eq_half && odd
    add.u32 %r17, %r17, %r6;               // round_up (0/1)

    add.u32 %r16, %r16, %r17;              // kept2
    // Carry: kept2 == 8 -> mantissa 0, exponent +1.  (e2 = 16 is
    // unreachable: the 448 clamp caps the top binade.)
    setp.eq.u32 %p1, %r16, 8;
    selp.b32 %r15, 0, %r16, %p1;           // man_f
    selp.b32 %r18, 1, 0, %p1;              // carry
    sub.u32 %r13, %r13, 120;               // e2 (wraps on flush lanes; discarded)
    add.u32 %r13, %r13, %r18;
    shl.b32 %r13, %r13, 3;
    or.b32  %r13, %r13, %r15;
    or.b32  %r13, %r13, %r10;              // body

    // res = is_norm ? body : sign8; res = is_nan ? 0x7F : res
    setp.ne.b32 %p1, %r14, 0;
    selp.b32 %r13, %r13, %r10, %p1;
    setp.ne.b32 %p1, %r19, 0;
    selp.b32 %r13, 0x7F, %r13, %p1;

    st.global.u8 [%rd7], %r13;

done_e4m3:
    ret;
}

// Batch entry: blockIdx.y = sequence index (0..M).
// src_ptrs[seq_idx] = device ptr to source K or V tensor for sequence seq_idx.
// cache_bases[seq_idx] = device ptr to KV-cache base for sequence seq_idx.
// write_off_ptr = ptr to single i32; all M seqs share the same write slot.
.visible .entry kv_scatter_write_batch(
    .param .u64 param_src_ptrs,
    .param .u64 param_cache_bases,
    .param .u64 param_write_off_ptr,
    .param .u32 param_kv_dim,
    .param .u32 param_m
)
{
    // %r0 = kv_dim  %r1 = ctaid.x  %r2 = ntid.x  %r3 = tid  %r4 = write_off
    // %r5 = seq_idx  %r6 = m
    // %rd0 = src_ptrs_array  %rd1 = cache_bases_array  %rd2 = write_off_ptr
    // %rd3 = src_ptr  %rd4 = cache_base  %rd5 = array byte offset
    // %rd6 = row  %rd7 = kv_dim64  %rd8 = tid64
    // %rd9 = dst_elem  %rd10 = dst_byte  %rd11 = src_byte
    .reg .u32   %r<8>;
    .reg .u64   %rd<12>;
    .reg .f32   %f1;
    .reg .pred  %p1;

    ld.param.u64 %rd0, [param_src_ptrs];
    ld.param.u64 %rd1, [param_cache_bases];
    ld.param.u64 %rd2, [param_write_off_ptr];
    ld.param.u32 %r0,  [param_kv_dim];
    ld.param.u32 %r6,  [param_m];

    // seq_idx = blockIdx.y; bounds-check
    mov.u32 %r5, %ctaid.y;
    setp.ge.u32 %p1, %r5, %r6;
    @%p1 bra done_batch;

    // global thread index within the kv_dim row
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r3, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r3, %r0;
    @%p1 bra done_batch;

    // byte offset into [M] u64 pointer arrays: seq_idx * 8
    cvt.u64.u32 %rd5, %r5;
    shl.b64     %rd5, %rd5, 3;

    // src_ptr = src_ptrs[seq_idx]
    add.u64       %rd3, %rd0, %rd5;
    ld.global.u64 %rd3, [%rd3];

    // cache_base = cache_bases[seq_idx]
    add.u64       %rd4, %rd1, %rd5;
    ld.global.u64 %rd4, [%rd4];

    // load write_offset from device (sign-extend to u64)
    ld.global.s32 %r4, [%rd2];
    cvt.u64.s32   %rd6, %r4;    // row
    cvt.u64.u32   %rd7, %r0;    // kv_dim64
    cvt.u64.u32   %rd8, %r3;    // tid64

    // dst element index = row * kv_dim + tid
    mul.lo.u64  %rd9, %rd6, %rd7;
    add.u64     %rd9, %rd9, %rd8;

    // dst byte address
    shl.b64  %rd10, %rd9, 2;
    add.u64  %rd10, %rd4, %rd10;

    // src byte address = src_ptr + tid * 4
    shl.b64  %rd11, %rd8, 2;
    add.u64  %rd11, %rd3, %rd11;

    ld.global.f32 %f1, [%rd11];
    st.global.f32 [%rd10], %f1;

done_batch:
    ret;
}

// Per-row-offset batch entries (kv_scatter_write_rows{,_f16,_e4m3}):
//   Grid:  (ceil(kv_dim / BLOCK_SIZE), M, 1).
//   Block: BLOCK_SIZE threads.
//   Params: src_ptrs (*const u64 [M]), cache_bases (*const u64 [M]),
//           write_offs (*const i32 [M] -- PER-ROW write slot), kv_dim (u32),
//           m (u32).
// The ragged (position-synced) batch CUDA-graph route: each sequence r has
// its OWN write offset (its session's kv_len), read from write_offs[r] at
// EXECUTION time, and its own source pointer and cache base, read from the
// device pointer arrays.  Everything per-step is device-read, so a captured
// graph is batch-COMPOSITION-independent (any M sessions of the same pool
// dtype share one graph; the arrays are re-uploaded H->D before each
// replay).  The cast bit-math is the twin kernels' verbatim (f16:
// cvt.rn.f16.f32; e4m3: the shared e4m3::f32_to_e4m3_bits port), so the
// pool bytes are bit-identical to both the per-row twins and the generic
// append path's cast + D2D -- probe-gated in bench-decode-ops alongside
// the twins.
//
// Per-row-offset batch entry (f32): every sequence reads its own write
// offset from write_offs[blockIdx.y].
.visible .entry kv_scatter_write_rows(
    .param .u64 param_src_ptrs,
    .param .u64 param_cache_bases,
    .param .u64 param_write_offs,
    .param .u32 param_kv_dim,
    .param .u32 param_m
)
{
    // %r0 = kv_dim  %r1 = ctaid.x  %r2 = ntid.x  %r3 = tid  %r4 = write_off
    // %r5 = seq_idx  %r6 = m
    // %rd0 = src_ptrs  %rd1 = cache_bases  %rd2 = write_offs
    // %rd3 = src_ptr  %rd4 = cache_base  %rd5 = seq64  %rd6 = scratch
    // %rd7 = kv_dim64  %rd8 = tid64  %rd9 = dst_elem  %rd10 = dst_byte
    // %rd11 = src_byte
    .reg .u32   %r<8>;
    .reg .u64   %rd<12>;
    .reg .f32   %f1;
    .reg .pred  %p1;

    ld.param.u64 %rd0, [param_src_ptrs];
    ld.param.u64 %rd1, [param_cache_bases];
    ld.param.u64 %rd2, [param_write_offs];
    ld.param.u32 %r0,  [param_kv_dim];
    ld.param.u32 %r6,  [param_m];

    mov.u32 %r5, %ctaid.y;
    setp.ge.u32 %p1, %r5, %r6;
    @%p1 bra done_rows;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r3, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r3, %r0;
    @%p1 bra done_rows;

    cvt.u64.u32 %rd5, %r5;
    shl.b64     %rd6, %rd5, 3;

    add.u64       %rd3, %rd0, %rd6;
    ld.global.u64 %rd3, [%rd3];

    add.u64       %rd4, %rd1, %rd6;
    ld.global.u64 %rd4, [%rd4];

    // per-row write offset: write_offs[seq_idx]
    shl.b64       %rd6, %rd5, 2;
    add.u64       %rd6, %rd2, %rd6;
    ld.global.s32 %r4,  [%rd6];
    cvt.u64.s32   %rd6, %r4;
    cvt.u64.u32   %rd7, %r0;
    cvt.u64.u32   %rd8, %r3;

    mul.lo.u64  %rd9, %rd6, %rd7;
    add.u64     %rd9, %rd9, %rd8;

    shl.b64  %rd10, %rd9, 2;
    add.u64  %rd10, %rd4, %rd10;

    shl.b64  %rd11, %rd8, 2;
    add.u64  %rd11, %rd3, %rd11;

    ld.global.f32 %f1, [%rd11];
    st.global.f32 [%rd10], %f1;

done_rows:
    ret;
}

// Per-row-offset batch f16 twin: same addressing, cvt.rn.f16.f32 + b16 store.
.visible .entry kv_scatter_write_rows_f16(
    .param .u64 param_src_ptrs,
    .param .u64 param_cache_bases,
    .param .u64 param_write_offs,
    .param .u32 param_kv_dim,
    .param .u32 param_m
)
{
    .reg .u32   %r<8>;
    .reg .u64   %rd<12>;
    .reg .b16   %h1;
    .reg .f32   %f1;
    .reg .pred  %p1;

    ld.param.u64 %rd0, [param_src_ptrs];
    ld.param.u64 %rd1, [param_cache_bases];
    ld.param.u64 %rd2, [param_write_offs];
    ld.param.u32 %r0,  [param_kv_dim];
    ld.param.u32 %r6,  [param_m];

    mov.u32 %r5, %ctaid.y;
    setp.ge.u32 %p1, %r5, %r6;
    @%p1 bra done_rows_f16;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r3, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r3, %r0;
    @%p1 bra done_rows_f16;

    cvt.u64.u32 %rd5, %r5;
    shl.b64     %rd6, %rd5, 3;

    add.u64       %rd3, %rd0, %rd6;
    ld.global.u64 %rd3, [%rd3];

    add.u64       %rd4, %rd1, %rd6;
    ld.global.u64 %rd4, [%rd4];

    shl.b64       %rd6, %rd5, 2;
    add.u64       %rd6, %rd2, %rd6;
    ld.global.s32 %r4,  [%rd6];
    cvt.u64.s32   %rd6, %r4;
    cvt.u64.u32   %rd7, %r0;
    cvt.u64.u32   %rd8, %r3;

    mul.lo.u64  %rd9, %rd6, %rd7;
    add.u64     %rd9, %rd9, %rd8;

    shl.b64  %rd10, %rd9, 1;
    add.u64  %rd10, %rd4, %rd10;

    shl.b64  %rd11, %rd8, 2;
    add.u64  %rd11, %rd3, %rd11;

    ld.global.f32  %f1, [%rd11];
    cvt.rn.f16.f32  %h1, %f1;
    st.global.b16  [%rd10], %h1;

done_rows_f16:
    ret;
}

// Per-row-offset batch e4m3 twin: same addressing, e4m3 bit-math + u8 store.
// Register use mirrors the single-seq twin; %r7 is the (single) scratch reg.
.visible .entry kv_scatter_write_rows_e4m3(
    .param .u64 param_src_ptrs,
    .param .u64 param_cache_bases,
    .param .u64 param_write_offs,
    .param .u32 param_kv_dim,
    .param .u32 param_m
)
{
    // %r9 = x bits  %r10 = sign8  %r11 = abs  %r12 = clamped bits  %r13 = exp
    // %r14 = is_norm  %r15 = man/dropped  %r16 = kept  %r17 = round_up
    // %r18 = eq_half/carry  %r19 = is_nan  %r7 = scratch
    .reg .u32   %r<20>;
    .reg .u64   %rd<12>;
    .reg .f32   %f<3>;
    .reg .pred  %p1;

    ld.param.u64 %rd0, [param_src_ptrs];
    ld.param.u64 %rd1, [param_cache_bases];
    ld.param.u64 %rd2, [param_write_offs];
    ld.param.u32 %r0,  [param_kv_dim];
    ld.param.u32 %r6,  [param_m];

    mov.u32 %r5, %ctaid.y;
    setp.ge.u32 %p1, %r5, %r6;
    @%p1 bra done_rows_e4m3;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r3, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r3, %r0;
    @%p1 bra done_rows_e4m3;

    cvt.u64.u32 %rd5, %r5;
    shl.b64     %rd6, %rd5, 3;

    add.u64       %rd3, %rd0, %rd6;
    ld.global.u64 %rd3, [%rd3];

    add.u64       %rd4, %rd1, %rd6;
    ld.global.u64 %rd4, [%rd4];

    shl.b64       %rd6, %rd5, 2;
    add.u64       %rd6, %rd2, %rd6;
    ld.global.s32 %r4,  [%rd6];
    cvt.u64.s32   %rd6, %r4;
    cvt.u64.u32   %rd7, %r0;
    cvt.u64.u32   %rd8, %r3;

    mul.lo.u64  %rd9, %rd6, %rd7;
    add.u64     %rd9, %rd9, %rd8;
    add.u64     %rd10, %rd4, %rd9;         // dst byte addr (1 byte/elem)

    shl.b64  %rd11, %rd8, 2;
    add.u64  %rd11, %rd3, %rd11;
    ld.global.f32 %f1, [%rd11];

    // NaN detect on the ORIGINAL bits: abs > inf-bits.
    mov.b32 %r9, %f1;
    and.b32 %r11, %r9, 0x7FFFFFFF;
    setp.gt.u32 %p1, %r11, 0x7F800000;
    selp.b32 %r19, 1, 0, %p1;              // is_nan

    max.f32 %f2, %f1, 0fC3E00000;          // -448.0
    min.f32 %f2, %f2, 0f43E00000;          //  448.0

    mov.b32 %r12, %f2;
    and.b32 %r11, %r12, 0x7FFFFFFF;        // abs (clamped)
    shr.u32 %r13, %r11, 23;                // exp
    shr.u32 %r10, %r12, 24;
    and.b32 %r10, %r10, 0x80;              // sign8

    setp.gt.u32 %p1, %r13, 120;
    selp.b32 %r14, 1, 0, %p1;              // is_norm

    and.b32 %r15, %r11, 0x007FFFFF;        // man
    shr.u32 %r16, %r15, 20;                // kept
    and.b32 %r15, %r15, 0x000FFFFF;        // dropped

    setp.gt.u32 %p1, %r15, 0x00080000;
    selp.b32 %r17, 1, 0, %p1;              // gt_half
    setp.eq.u32 %p1, %r15, 0x00080000;
    selp.b32 %r18, 1, 0, %p1;              // eq_half
    and.b32 %r7, %r16, 1;                  // kept LSB
    and.b32 %r7, %r7, %r18;                // eq_half && odd
    add.u32 %r17, %r17, %r7;               // round_up (0/1)

    add.u32 %r16, %r16, %r17;              // kept2
    setp.eq.u32 %p1, %r16, 8;
    selp.b32 %r15, 0, %r16, %p1;           // man_f
    selp.b32 %r18, 1, 0, %p1;              // carry
    sub.u32 %r13, %r13, 120;               // e2
    add.u32 %r13, %r13, %r18;
    shl.b32 %r13, %r13, 3;
    or.b32  %r13, %r13, %r15;
    or.b32  %r13, %r13, %r10;              // body

    setp.ne.b32 %p1, %r14, 0;
    selp.b32 %r13, %r13, %r10, %p1;
    setp.ne.b32 %p1, %r19, 0;
    selp.b32 %r13, 0x7F, %r13, %p1;

    st.global.u8 [%rd10], %r13;

done_rows_e4m3:
    ret;
}

// Row-gather entry: dst[tid] = src[pos * row_len + tid], pos from device memory.
// The position scalar is uploaded by the engine as an f32 bitcast, so it is
// read back with ld.global.s32 (the register is untyped; the bit pattern is
// the original i32 position).
.visible .entry d2d_gather_row(
    .param .u64 param_src,
    .param .u64 param_dst,
    .param .u64 param_pos_ptr,
    .param .u32 param_row_len,
    .param .u32 param_max_pos
)
{
    // %r0 = row_len  %r1 = max_pos  %r2 = ctaid.x  %r3 = ntid.x  %r4 = tid
    // %r5 = global tid  %r6 = pos
    // %rd0 = src  %rd1 = dst  %rd2 = pos_ptr
    // %rd3 = src elem addr  %rd4 = row_len64  %rd5 = tid64  %rd6 = dst addr
    .reg .u32   %r<8>;
    .reg .u64   %rd<8>;
    .reg .f32   %f1;
    .reg .pred  %p1;

    ld.param.u64 %rd0, [param_src];
    ld.param.u64 %rd1, [param_dst];
    ld.param.u64 %rd2, [param_pos_ptr];
    ld.param.u32 %r0,  [param_row_len];
    ld.param.u32 %r1,  [param_max_pos];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r5, %r0;
    @%p1 bra done_gather;

    // Position read at EXECUTION time (CUDA-graph stable).
    ld.global.s32 %r6, [%rd2];

    // Clamp to [0, max_pos): the engine routes out-of-range positions away
    // from graph replay, but never trust a scalar we do not control.
    setp.ge.s32 %p1, %r6, %r1;
    @%p1 sub.s32  %r6, %r1, 1;
    setp.lt.s32 %p1, %r6, 0;
    @%p1 mov.u32  %r6, 0;

    // src element address = src + (pos * row_len + tid) * 4
    cvt.u64.s32 %rd3, %r6;
    cvt.u64.u32 %rd4, %r0;
    mul.lo.u64  %rd3, %rd3, %rd4;
    cvt.u64.u32 %rd5, %r5;
    add.u64     %rd3, %rd3, %rd5;
    shl.b64     %rd3, %rd3, 2;
    add.u64     %rd3, %rd0, %rd3;

    // dst element address = dst + tid * 4
    shl.b64  %rd6, %rd5, 2;
    add.u64  %rd6, %rd1, %rd6;

    ld.global.f32 %f1, [%rd3];
    st.global.f32 [%rd6], %f1;

done_gather:
    ret;
}
"#;

const BLOCK_SIZE: u32 = 256;
const KERNEL_NAME: &str = "kv_scatter_write";
const KERNEL_NAME_F16: &str = "kv_scatter_write_f16";
const KERNEL_NAME_E4M3: &str = "kv_scatter_write_e4m3";
const KERNEL_NAME_BATCH: &str = "kv_scatter_write_batch";
const KERNEL_NAME_GATHER: &str = "d2d_gather_row";
const KERNEL_NAME_ROWS: &str = "kv_scatter_write_rows";
const KERNEL_NAME_ROWS_F16: &str = "kv_scatter_write_rows_f16";
const KERNEL_NAME_ROWS_E4M3: &str = "kv_scatter_write_rows_e4m3";

/// Scatter-write kernel loader and launcher.
///
/// Holds function handles for the single-sequence and batch scatter-write
/// variants plus the row-gather kernel; all are resolved from the same PTX
/// module so the module lifetime covers them.
pub struct KvScatterWrite {
    cu_module:          sys::CUmodule,
    cu_function:        sys::CUfunction,
    cu_function_f16:    sys::CUfunction,
    cu_function_e4m3:   sys::CUfunction,
    cu_function_batch:  sys::CUfunction,
    cu_function_gather: sys::CUfunction,
    cu_function_rows: sys::CUfunction,
    cu_function_rows_f16: sys::CUfunction,
    cu_function_rows_e4m3: sys::CUfunction,
}

unsafe impl Send for KvScatterWrite {}
unsafe impl Sync for KvScatterWrite {}

impl KvScatterWrite {
    /// Load the PTX module and resolve both kernel functions.
    ///
    /// Must be called after a CUDA device context is active (i.e. after any
    /// `CutileMatMul::new()` or similar CUDA initialisation).
    pub fn new() -> Result<Self> {
        let mut ptx_nul: Vec<u8> = PTX_SOURCE.bytes().collect();
        ptx_nul.push(0u8); // cuModuleLoadData requires null-terminated PTX

        let mut cu_module = std::ptr::null_mut::<sys::CUmod_st>();
        let rc = unsafe {
            sys::cuModuleLoadData(
                &mut cu_module,
                ptx_nul.as_ptr() as *const std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite: cuModuleLoadData failed: {rc}"
            )));
        }

        let name = std::ffi::CString::new(KERNEL_NAME).unwrap();
        let mut cu_function = std::ptr::null_mut::<sys::CUfunc_st>();
        let rc = unsafe {
            sys::cuModuleGetFunction(&mut cu_function, cu_module, name.as_ptr())
        };
        if rc != 0 {
            let _ = unsafe { sys::cuModuleUnload(cu_module) };
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite: cuModuleGetFunction({KERNEL_NAME}) failed: {rc}"
            )));
        }

        let name_f16 = std::ffi::CString::new(KERNEL_NAME_F16).unwrap();
        let mut cu_function_f16 = std::ptr::null_mut::<sys::CUfunc_st>();
        let rc = unsafe {
            sys::cuModuleGetFunction(&mut cu_function_f16, cu_module, name_f16.as_ptr())
        };
        if rc != 0 {
            let _ = unsafe { sys::cuModuleUnload(cu_module) };
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite: cuModuleGetFunction({KERNEL_NAME_F16}) failed: {rc}"
            )));
        }

        let name_e4m3 = std::ffi::CString::new(KERNEL_NAME_E4M3).unwrap();
        let mut cu_function_e4m3 = std::ptr::null_mut::<sys::CUfunc_st>();
        let rc = unsafe {
            sys::cuModuleGetFunction(&mut cu_function_e4m3, cu_module, name_e4m3.as_ptr())
        };
        if rc != 0 {
            let _ = unsafe { sys::cuModuleUnload(cu_module) };
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite: cuModuleGetFunction({KERNEL_NAME_E4M3}) failed: {rc}"
            )));
        }

        let name_batch = std::ffi::CString::new(KERNEL_NAME_BATCH).unwrap();
        let mut cu_function_batch = std::ptr::null_mut::<sys::CUfunc_st>();
        let rc = unsafe {
            sys::cuModuleGetFunction(&mut cu_function_batch, cu_module, name_batch.as_ptr())
        };
        if rc != 0 {
            let _ = unsafe { sys::cuModuleUnload(cu_module) };
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite: cuModuleGetFunction({KERNEL_NAME_BATCH}) failed: {rc}"
            )));
        }

        let name_gather = std::ffi::CString::new(KERNEL_NAME_GATHER).unwrap();
        let mut cu_function_gather = std::ptr::null_mut::<sys::CUfunc_st>();
        let rc = unsafe {
            sys::cuModuleGetFunction(&mut cu_function_gather, cu_module, name_gather.as_ptr())
        };
        if rc != 0 {
            let _ = unsafe { sys::cuModuleUnload(cu_module) };
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite: cuModuleGetFunction({KERNEL_NAME_GATHER}) failed: {rc}"
            )));
        }

        let resolve = |name: &str| -> Result<sys::CUfunction> {
            let cname = std::ffi::CString::new(name).unwrap();
            let mut f = std::ptr::null_mut::<sys::CUfunc_st>();
            let rc = unsafe { sys::cuModuleGetFunction(&mut f, cu_module, cname.as_ptr()) };
            if rc != 0 {
                return Err(KernelError::InvalidConfig(format!(
                    "KvScatterWrite: cuModuleGetFunction({name}) failed: {rc}"
                )));
            }
            Ok(f)
        };
        let cu_function_rows = resolve(KERNEL_NAME_ROWS)?;
        let cu_function_rows_f16 = resolve(KERNEL_NAME_ROWS_F16)?;
        let cu_function_rows_e4m3 = resolve(KERNEL_NAME_ROWS_E4M3)?;

        Ok(Self { cu_module, cu_function, cu_function_f16, cu_function_e4m3, cu_function_batch, cu_function_gather, cu_function_rows, cu_function_rows_f16, cu_function_rows_e4m3 })
    }

    /// Copy `kv_dim` f32 elements from `src_ptr` into
    /// `cache_base[*write_off_ptr * kv_dim .. (*write_off_ptr + 1) * kv_dim]`.
    ///
    /// All pointers are device addresses.  The call is asynchronous on `cu_stream`.
    pub fn copy_with_device_offset(
        &self,
        src_ptr:       sys::CUdeviceptr,
        cache_base:    sys::CUdeviceptr,
        write_off_ptr: sys::CUdeviceptr,
        kv_dim:        u32,
        cu_stream:     sys::CUstream,
    ) -> Result<()> {
        let grid = (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let mut p_src   = src_ptr;
        let mut p_cache = cache_base;
        let mut p_woff  = write_off_ptr;
        let mut p_dim   = kv_dim;

        let mut params: [*mut std::ffi::c_void; 4] = [
            &mut p_src   as *mut _ as *mut std::ffi::c_void,
            &mut p_cache as *mut _ as *mut std::ffi::c_void,
            &mut p_woff  as *mut _ as *mut std::ffi::c_void,
            &mut p_dim   as *mut _ as *mut std::ffi::c_void,
        ];

        let rc = unsafe {
            sys::cuLaunchKernel(
                self.cu_function,
                grid, 1, 1,
                BLOCK_SIZE, 1, 1,
                0,
                cu_stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite::copy_with_device_offset cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }

    /// [`Self::copy_with_device_offset`] for an f16 KV pool: the f32 source
    /// element is converted with IEEE RNE (cvt.rn.f16.f32, bit-identical to
    /// the generic append path's cutile_cast_f32_f16) and stored as u16 at
    /// `cache_base[write_off * kv_dim + i]` (u16 elements).
    pub fn copy_with_device_offset_f16(
        &self,
        src_ptr:       sys::CUdeviceptr,
        cache_base:    sys::CUdeviceptr,
        write_off_ptr: sys::CUdeviceptr,
        kv_dim:        u32,
        cu_stream:     sys::CUstream,
    ) -> Result<()> {
        let grid = (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let mut p_src   = src_ptr;
        let mut p_cache = cache_base;
        let mut p_woff  = write_off_ptr;
        let mut p_dim   = kv_dim;

        let mut params: [*mut std::ffi::c_void; 4] = [
            &mut p_src   as *mut _ as *mut std::ffi::c_void,
            &mut p_cache as *mut _ as *mut std::ffi::c_void,
            &mut p_woff  as *mut _ as *mut std::ffi::c_void,
            &mut p_dim   as *mut _ as *mut std::ffi::c_void,
        ];

        let rc = unsafe {
            sys::cuLaunchKernel(
                self.cu_function_f16,
                grid, 1, 1,
                BLOCK_SIZE, 1, 1,
                0,
                cu_stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite::copy_with_device_offset_f16 cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }

    /// [`Self::copy_with_device_offset`] for an e4m3 KV pool: the f32 source
    /// element is converted with the shared e4m3 bit-math (RNE, +-448
    /// saturation, subnormal flush, NaN -> 0x7F -- bit-identical to
    /// cutile_cast_f32_e4m3 and `e4m3::f32_to_e4m3_bits`) and stored as u8
    /// at `cache_base[write_off * kv_dim + i]` (u8 elements).
    pub fn copy_with_device_offset_e4m3(
        &self,
        src_ptr:       sys::CUdeviceptr,
        cache_base:    sys::CUdeviceptr,
        write_off_ptr: sys::CUdeviceptr,
        kv_dim:        u32,
        cu_stream:     sys::CUstream,
    ) -> Result<()> {
        let grid = (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let mut p_src   = src_ptr;
        let mut p_cache = cache_base;
        let mut p_woff  = write_off_ptr;
        let mut p_dim   = kv_dim;

        let mut params: [*mut std::ffi::c_void; 4] = [
            &mut p_src   as *mut _ as *mut std::ffi::c_void,
            &mut p_cache as *mut _ as *mut std::ffi::c_void,
            &mut p_woff  as *mut _ as *mut std::ffi::c_void,
            &mut p_dim   as *mut _ as *mut std::ffi::c_void,
        ];

        let rc = unsafe {
            sys::cuLaunchKernel(
                self.cu_function_e4m3,
                grid, 1, 1,
                BLOCK_SIZE, 1, 1,
                0,
                cu_stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite::copy_with_device_offset_e4m3 cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }

    /// Copy `kv_dim` f32 elements for M sequences in one kernel launch.
    ///
    /// - `src_ptrs_dev`:    device ptr to a `[M x u64]` array of source device pointers.
    /// - `cache_bases_dev`: device ptr to a `[M x u64]` array of KV-cache base device pointers.
    /// - `write_off_ptr`:   device ptr to a single `i32`; ALL M sequences share the same
    ///                      write slot (valid for synchronised batch decode where all seqs
    ///                      advance together).
    ///
    /// Grid is `(ceil(kv_dim / BLOCK_SIZE), m, 1)`.  The call is asynchronous on `cu_stream`.
    pub fn copy_m_with_device_offsets(
        &self,
        src_ptrs_dev:    sys::CUdeviceptr,
        cache_bases_dev: sys::CUdeviceptr,
        write_off_ptr:   sys::CUdeviceptr,
        kv_dim:          u32,
        m:               u32,
        cu_stream:       sys::CUstream,
    ) -> Result<()> {
        let grid_x = (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let mut p_srcs  = src_ptrs_dev;
        let mut p_bases = cache_bases_dev;
        let mut p_woff  = write_off_ptr;
        let mut p_dim   = kv_dim;
        let mut p_m     = m;

        let mut params: [*mut std::ffi::c_void; 5] = [
            &mut p_srcs  as *mut _ as *mut std::ffi::c_void,
            &mut p_bases as *mut _ as *mut std::ffi::c_void,
            &mut p_woff  as *mut _ as *mut std::ffi::c_void,
            &mut p_dim   as *mut _ as *mut std::ffi::c_void,
            &mut p_m     as *mut _ as *mut std::ffi::c_void,
        ];

        let rc = unsafe {
            sys::cuLaunchKernel(
                self.cu_function_batch,
                grid_x, m, 1,
                BLOCK_SIZE, 1, 1,
                0,
                cu_stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite::copy_m_with_device_offsets cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }

    /// Per-row-offset batch scatter for RAGGED (position-synced) batches:
    /// like [`Self::copy_m_with_device_offsets`] but every sequence r reads
    /// its OWN write slot from `write_offs_dev[r]` (an `[m] i32` device
    /// array) -- each session's kv_len, not a shared synchronised offset.
    ///
    /// `src_ptrs_dev` / `cache_bases_dev` are `[m] u64` device arrays read
    /// at execution time, so a CUDA graph capturing this kernel is
    /// batch-composition-independent (the arrays are re-uploaded H->D
    /// before each replay).  The f32 flavor stores raw f32 elements.
    pub fn copy_m_with_per_row_offsets(
        &self,
        src_ptrs_dev:    sys::CUdeviceptr,
        cache_bases_dev: sys::CUdeviceptr,
        write_offs_dev:  sys::CUdeviceptr,
        kv_dim:          u32,
        m:               u32,
        cu_stream:       sys::CUstream,
    ) -> Result<()> {
        self.launch_rows(
            self.cu_function_rows,
            src_ptrs_dev,
            cache_bases_dev,
            write_offs_dev,
            kv_dim,
            m,
            cu_stream,
        )
    }

    /// [`Self::copy_m_with_per_row_offsets`] for an f16 KV pool: the f32
    /// source is converted with cvt.rn.f16.f32 (bit-identical to both the
    /// single-seq f16 twin and the generic append path's cutile_cast_f32_f16)
    /// and stored as u16 elements.
    pub fn copy_m_with_per_row_offsets_f16(
        &self,
        src_ptrs_dev:    sys::CUdeviceptr,
        cache_bases_dev: sys::CUdeviceptr,
        write_offs_dev:  sys::CUdeviceptr,
        kv_dim:          u32,
        m:               u32,
        cu_stream:       sys::CUstream,
    ) -> Result<()> {
        self.launch_rows(
            self.cu_function_rows_f16,
            src_ptrs_dev,
            cache_bases_dev,
            write_offs_dev,
            kv_dim,
            m,
            cu_stream,
        )
    }

    /// [`Self::copy_m_with_per_row_offsets`] for an e4m3 KV pool: the f32
    /// source is converted with the shared e4m3 bit-math (bit-identical to
    /// both the single-seq e4m3 twin and cutile_cast_f32_e4m3) and stored
    /// as u8 elements.
    pub fn copy_m_with_per_row_offsets_e4m3(
        &self,
        src_ptrs_dev:    sys::CUdeviceptr,
        cache_bases_dev: sys::CUdeviceptr,
        write_offs_dev:  sys::CUdeviceptr,
        kv_dim:          u32,
        m:               u32,
        cu_stream:       sys::CUstream,
    ) -> Result<()> {
        self.launch_rows(
            self.cu_function_rows_e4m3,
            src_ptrs_dev,
            cache_bases_dev,
            write_offs_dev,
            kv_dim,
            m,
            cu_stream,
        )
    }

    fn launch_rows(
        &self,
        cu_function:     sys::CUfunction,
        src_ptrs_dev:    sys::CUdeviceptr,
        cache_bases_dev: sys::CUdeviceptr,
        write_offs_dev:  sys::CUdeviceptr,
        kv_dim:          u32,
        m:               u32,
        cu_stream:       sys::CUstream,
    ) -> Result<()> {
        let grid_x = kv_dim.div_ceil(BLOCK_SIZE);

        let mut p_srcs  = src_ptrs_dev;
        let mut p_bases = cache_bases_dev;
        let mut p_woffs = write_offs_dev;
        let mut p_dim   = kv_dim;
        let mut p_m     = m;

        let mut params: [*mut std::ffi::c_void; 5] = [
            &mut p_srcs  as *mut _ as *mut std::ffi::c_void,
            &mut p_bases as *mut _ as *mut std::ffi::c_void,
            &mut p_woffs as *mut _ as *mut std::ffi::c_void,
            &mut p_dim   as *mut _ as *mut std::ffi::c_void,
            &mut p_m     as *mut _ as *mut std::ffi::c_void,
        ];

        let rc = unsafe {
            sys::cuLaunchKernel(
                cu_function,
                grid_x, m, 1,
                BLOCK_SIZE, 1, 1,
                0,
                cu_stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite::copy_m_with_per_row_offsets cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }

    /// Copy row `pos` of a row-major f32 table `[max_pos * row_len]` into
    /// `dst` (`row_len` elements), where `pos` is read from device memory at
    /// execution time.
    ///
    /// `pos_ptr` points at an i32 scalar (the engine uploads the position as
    /// an f32 bitcast into an f32-typed buffer; the bit pattern is the i32
    /// position). CUDA-graph stable: the graph captures the pointers, only
    /// the value at `pos_ptr` changes per step. `pos` is clamped to
    /// `[0, max_pos)` on device.
    ///
    /// Used to refresh the RoPE cos/sin row each decode step without
    /// re-capturing the decode graph (the position is otherwise a baked
    /// kernel launch scalar and replays stale).
    pub fn gather_row_by_device_pos(
        &self,
        src:       sys::CUdeviceptr,
        dst:       sys::CUdeviceptr,
        pos_ptr:   sys::CUdeviceptr,
        row_len:   u32,
        max_pos:   u32,
        cu_stream: sys::CUstream,
    ) -> Result<()> {
        if row_len == 0 {
            return Ok(());
        }
        let grid = row_len.div_ceil(BLOCK_SIZE);

        let mut p_src = src;
        let mut p_dst = dst;
        let mut p_pos = pos_ptr;
        let mut p_len = row_len;
        let mut p_max = max_pos;

        let mut params: [*mut std::ffi::c_void; 5] = [
            &mut p_src as *mut _ as *mut std::ffi::c_void,
            &mut p_dst as *mut _ as *mut std::ffi::c_void,
            &mut p_pos as *mut _ as *mut std::ffi::c_void,
            &mut p_len as *mut _ as *mut std::ffi::c_void,
            &mut p_max as *mut _ as *mut std::ffi::c_void,
        ];

        let rc = unsafe {
            sys::cuLaunchKernel(
                self.cu_function_gather,
                grid, 1, 1,
                BLOCK_SIZE, 1, 1,
                0,
                cu_stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite::gather_row_by_device_pos cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }
}

impl Drop for KvScatterWrite {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = unsafe { sys::cuModuleUnload(self.cu_module) };
        }
    }
}
