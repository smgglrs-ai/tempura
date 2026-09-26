//! Direct Rust translation of llama.cpp `ggml_compute_forward_mul_mat`.
//!
//! Source: ggml/src/ggml-cpu/ggml-cpu.c (lines 215-1468)
//!
//! This module documents exactly how llama.cpp routes quantized matmul to
//! scalar vec_dot paths and how thread work is partitioned across chunks.
//! It strips the ggml tensor-graph scaffolding (nb00/nb01/ne00/etc.) and
//! expresses the dispatch logic in plain Rust with flat slice arguments.
//!
//! Notation that maps C locals to this file:
//!   ne00 = k    (inner dimension, elements per weight row)
//!   ne01 = n    (weight rows = output neurons)
//!   ne10 = k    (inner dimension, elements per activation row)
//!   ne11 = m    (activation rows — 1 for decode, >1 for prefill)
//!   ne0  = ne01 = n
//!   ne1  = ne11 = m
//!   ne2/ne3 = batch dims (1 for standard single-sequence inference)
//!
//! For a standard single-sequence forward pass:
//!   nr0 = n   (output neurons, indexed by weight row)
//!   nr1 = m   (tokens, indexed by activation row)

use std::sync::atomic::{AtomicI64, Ordering};

use super::quantize::{quantize_activations, quantized_row_size};
use super::GgmlType;

// ---------------------------------------------------------------------------
// Type-traits table (translated from type_traits_cpu[GGML_TYPE_COUNT])
// ---------------------------------------------------------------------------
//
// C definition (ggml-cpu.h):
//   struct ggml_type_traits_cpu {
//       ggml_from_float_t  from_float;    // f32[] → quantized[]
//       ggml_vec_dot_t     vec_dot;       // dot(weight_row, act_row) → f32
//       enum ggml_type     vec_dot_type;  // what format activations must be in
//       int64_t            nrows;         // weight rows processed per vec_dot call
//   };
//
// Key entries for the quant types we care about:
//
//   Q4_K:
//     from_float    = quantize_row_q4_K          (f32 → block_q4_K[])
//     vec_dot       = ggml_vec_dot_q4_K_q8_K     (block_q4_K[] · block_q8_K[])
//     vec_dot_type  = GGML_TYPE_Q8_K
//     nrows         = 2  (ARM MATMUL_INT8 mmla path)
//                   = 1  (x86, all other targets)
//
//   Q6_K:
//     from_float    = quantize_row_q6_K
//     vec_dot       = ggml_vec_dot_q6_K_q8_K
//     vec_dot_type  = GGML_TYPE_Q8_K
//     nrows         = 2 (ARM MATMUL_INT8) / 1 (x86)
//
//   Q4_0:
//     from_float    = quantize_row_q4_0
//     vec_dot       = ggml_vec_dot_q4_0_q8_0
//     vec_dot_type  = GGML_TYPE_Q8_0
//     nrows         = 2 (ARM MATMUL_INT8) / 1 (x86)
//
//   Q8_K (activations-only, used as vec_dot_type for *_K weights):
//     from_float    = quantize_row_q8_K
//     vec_dot       = (none — Q8_K is only used as the activation format)
//
//   F32:
//     from_float    = identity copy
//     vec_dot       = ggml_vec_dot_f32
//     vec_dot_type  = GGML_TYPE_F32
//     nrows         = 1
//
// In tempura terms: GgmlType::vec_dot_type() and quantized_row_size() already
// encode the vec_dot_type and per-row byte size from this table.

/// Trait-table nrows on x86 (no ARM mmla).
///
/// On ARM with MATMUL_INT8, Q4_0, Q4_1, Q8_0, Q4_K, Q6_K → nrows = 2, which
/// enables the mmla kernel that processes 2 weight rows × 2 activation rows per
/// call. On x86 this is always 1.
pub fn nrows_for_type(t: GgmlType) -> usize {
    // ARM nrows=2 path requires target_feature="neon" + ARM_FEATURE_MATMUL_INT8.
    // We only target x86_64 here.
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("i8mm") {
            match t {
                GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q8_0 | GgmlType::Q4K
                | GgmlType::Q6K => return 2,
                _ => {}
            }
        }
    }
    let _ = t;
    1
}

// ---------------------------------------------------------------------------
// Inner computation: ggml_compute_forward_mul_mat_one_chunk
// ---------------------------------------------------------------------------
//
// C signature (line 1165):
//   static void ggml_compute_forward_mul_mat_one_chunk(
//       const struct ggml_compute_params * params,
//       struct ggml_tensor * dst,
//       const enum ggml_type type,
//       const int64_t num_rows_per_vec_dot,
//       const int64_t ir0_start, const int64_t ir0_end,
//       const int64_t ir1_start, const int64_t ir1_end)
//
// Translated to flat-slice form below.  The ggml version navigates the tensor
// using nb01/nb11/nb1/nb0 strides; here we assume contiguous row-major storage
// so strides are just row_bytes.
//
// Block tiling:
//   blck_0 = 16  (weight-row tile — iterates over nr0, output neurons)
//   blck_1 = 16  (activation-row tile — iterates over nr1, tokens)
//
// The loop structure is:
//   for iir1 in (ir1_start..ir1_end).step_by(blck_1)
//     for iir0 in (ir0_start..ir0_end).step_by(blck_0)
//       for ir1 in (iir1..iir1+blck_1).step_by(num_rows_per_vec_dot) ← token loop
//         for ir0 in (iir0..iir0+blck_0).step_by(num_rows_per_vec_dot) ← neuron loop
//           vec_dot(k, &mut tmp[ir0-iir0], src0_row+ir0*row0, src1_col+ir1*row1, num_rows)
//         memcpy tmp → dst_col
//
// When num_rows_per_vec_dot == 1 (x86 path):
//   vec_dot processes one weight row × one activation row → one f32 output.
// When num_rows_per_vec_dot == 2 (ARM mmla path):
//   vec_dot processes 2 weight rows × 2 activation rows → 4 f32 outputs stored
//   in tmp at offsets [0..16) and [16..32) (16-float stride).

/// Translated inner-chunk kernel.
///
/// Arguments:
///   src0            weight bytes, row-major, n rows × weight_row_bytes
///   src1_q          activation bytes already quantized to vec_dot_type,
///                   row-major, m rows × act_row_bytes
///   dst             output f32, row-major, m rows × n cols
///   n               output neurons (weight rows)
///   m               tokens (activation rows)
///   k               inner dimension (elements)
///   weight_type     ggml weight quantization type
///   num_rows_per_vec_dot  1 on x86, 2 on ARM mmla
///   ir0_start/end   weight-row range this chunk owns [ir0_start, ir0_end)
///   ir1_start/end   activation-row range this chunk owns [ir1_start, ir1_end)
pub fn mul_mat_one_chunk(
    src0: &[u8],
    src1_q: &[u8],
    dst: &mut [f32],
    n: usize,
    _m: usize,
    k: usize,
    weight_type: GgmlType,
    num_rows_per_vec_dot: usize,
    ir0_start: usize,
    ir0_end: usize,
    ir1_start: usize,
    ir1_end: usize,
) {
    if ir0_start >= ir0_end || ir1_start >= ir1_end {
        return;
    }

    let weight_row_bytes = quantized_row_size(k, weight_type);
    let act_row_bytes = quantized_row_size(k, weight_type.vec_dot_type());

    // blck_0 / blck_1 match the C constants exactly.
    let blck_0: usize = 16;
    let blck_1: usize = 16;

    // tmp: 32 floats to accommodate num_rows_per_vec_dot == 2 (ARM mmla outputs
    // 2 rows interleaved with 16-float stride: tmp[0..16) = row0, tmp[16..32) = row1).
    // On x86 with num_rows_per_vec_dot == 1 only tmp[0..blck_0) is used.
    let mut tmp = [0.0f32; 32];

    // Outer tile: activation rows (ir1), then weight rows (ir0).
    let mut iir1 = ir1_start;
    while iir1 < ir1_end {
        let mut iir0 = ir0_start;
        while iir0 < ir0_end {
            // Inner loop over activation rows in this tile.
            let mut ir1 = iir1;
            while ir1 < (iir1 + blck_1).min(ir1_end) {
                // Pointer into quantized activation buffer for this token.
                let src1_col = &src1_q[ir1 * act_row_bytes..];

                // Pointer into dst for this token's output row.
                let dst_col = &mut dst[ir1 * n..];

                // Inner loop over weight rows in this tile, stepped by num_rows_per_vec_dot.
                let mut ir0 = iir0;
                while ir0 < (iir0 + blck_0).min(ir0_end) {
                    let src0_row = &src0[ir0 * weight_row_bytes..];

                    // C call (line 1244):
                    //   vec_dot(ne00,
                    //           &tmp[ir0 - iir0],
                    //           num_rows_per_vec_dot > 1 ? 16 : 0,      // dst stride (floats)
                    //           src0_row + ir0*nb01,
                    //           num_rows_per_vec_dot > 1 ? nb01 : 0,    // src0 row stride
                    //           src1_col,
                    //           num_rows_per_vec_dot > 1 ? src1_col_stride : 0,
                    //           num_rows_per_vec_dot);
                    //
                    // When num_rows_per_vec_dot == 1:
                    //   All stride args are 0 — scalar single-row dot product.
                    //   Result stored at tmp[ir0 - iir0].
                    //
                    // When num_rows_per_vec_dot == 2 (ARM mmla):
                    //   Processes weight rows ir0 and ir0+1 against activation
                    //   rows ir1 and ir1+1. Results written to:
                    //     tmp[ir0 - iir0]       = dot(weight[ir0], act[ir1])
                    //     tmp[ir0 - iir0 + 16]  = dot(weight[ir0+1], act[ir1])
                    //   (the +16 stride is fixed in the C implementation)

                    let tmp_idx = ir0 - iir0;

                    // In tempura we only implement the num_rows_per_vec_dot == 1 path.
                    // The ARM mmla (nrows == 2) path would require a different vec_dot
                    // ABI; see ggml-cpu-impl.h ggml_vec_dot_t signature for details.
                    super::vec_dot::vec_dot(
                        k,
                        &mut tmp[tmp_idx],
                        src0_row,
                        src1_col,
                        weight_type,
                    );

                    ir0 += num_rows_per_vec_dot;
                }

                // Writeback: copy computed dot products from tmp into dst.
                // C line 1247-1249:
                //   for cn in 0..num_rows_per_vec_dot:
                //     memcpy(&dst_col[iir0 + cn * nb1/nb0],
                //            tmp + cn * 16,
                //            MIN(iir0 + blck_0, ir0_end) - iir0)
                //
                // nb1/nb0 == n (stride between output rows in elements).
                // For num_rows_per_vec_dot == 1: one memcpy of (end-iir0) floats.
                // For num_rows_per_vec_dot == 2: two memcpys, second offset by n.
                let write_end = (iir0 + blck_0).min(ir0_end);
                let write_len = write_end - iir0;
                for cn in 0..num_rows_per_vec_dot {
                    let dst_start = iir0 + cn * n;
                    dst_col[dst_start..dst_start + write_len]
                        .copy_from_slice(&tmp[cn * 16..cn * 16 + write_len]);
                }

                ir1 += num_rows_per_vec_dot;
            }
            iir0 += blck_0;
        }
        iir1 += blck_1;
    }
}

// ---------------------------------------------------------------------------
// Outer dispatch: ggml_compute_forward_mul_mat routing
// ---------------------------------------------------------------------------
//
// C source: lines 1255-1468.
//
// High-level flow:
//
//  1. [llamafile sgemm fast path — F32/F16 only, skipped here]
//
//  2. Activation quantization (line 1323-1366):
//     If src1 is not already in vec_dot_type format (always true when src1 is
//     f32 and weights are Q4_K / Q6_K / etc.), quantize all activation rows
//     into params->wdata using from_float = type_traits_cpu[vec_dot_type].from_float.
//     This is parallelised: thread `ith` processes blocks [ith*ne10/bs/nth, (ith+1)*ne10/bs/nth)
//     of EACH activation row (not a row-per-thread split).
//
//  3. Barrier (ggml_barrier): all threads wait until quantization is complete.
//
//  4. [IQ panel gemm — for IQ2/IQ3 quants, skipped here]
//
//  5. Chunk setup (lines 1406-1437):
//     nr0 = ne01 = n   (output neurons — the "weight rows" dimension)
//     nr1 = ne11 * ne12 * ne13 = m   (tokens × batch dims)
//
//     chunk_size = 16  (default)
//     if nr0 == 1 or nr1 == 1: chunk_size = 64
//       → decode (m==1, nr1==1): chunk_size bumped to 64, so most chunks cover
//         weight-row tiles of 64 instead of 16, reducing per-chunk overhead.
//
//     nchunk0 = CEIL(nr0 / chunk_size)   (chunks along weight-row axis)
//     nchunk1 = CEIL(nr1 / chunk_size)   (chunks along activation-row axis)
//
//     NUMA / poor-chunk-count override (line 1429):
//       if nchunk0 * nchunk1 < nth*4 or NUMA system:
//         if nr0 > nr1: nchunk0 = nth, nchunk1 = 1  (parallelize weight rows)
//         else:         nchunk0 = 1, nchunk1 = nth  (parallelize activation rows)
//
//     dr0 = CEIL(nr0 / nchunk0)  (weight rows per chunk)
//     dr1 = CEIL(nr1 / nchunk1)  (activation rows per chunk)
//
//  6. Work stealing (lines 1440-1467):
//     Thread `ith` starts with chunk index `ith`.
//     After processing a chunk it atomically increments threadpool->current_chunk
//     and processes the returned index, continuing until >= nchunk0*nchunk1.
//     Thread 0 pre-initialises current_chunk = nth before the barrier so that
//     threads claim chunks nth..nchunk0*nchunk1 dynamically.
//
//  7. num_rows_per_vec_dot guard (line 1453-1459):
//     Start from vec_dot_num_rows (nrows from type_traits table: 1 on x86, 2 on ARM).
//     Fall back to 1 if:
//       nr0 % 2 != 0                         (total weight rows odd)
//       ne11 % 2 != 0                         (activation rows odd)
//       (ir0_end - ir0_start) % 2 != 0        (chunk weight rows odd)
//       (ir1_end - ir1_start) % 2 != 0        (chunk activation rows odd)
//     This prevents the mmla kernel from reading out-of-bounds on misaligned chunks.
//
//  8. Call mul_mat_one_chunk for the assigned range.
//
// Decode vs Prefill summary:
//   Decode (m == 1, nr1 == 1):
//     chunk_size = 64, nchunk1 = 1, all chunks span weight-row axis.
//     NUMA path kicks in when nth > nchunk0 * nchunk1 / 4, redistributing to
//     nchunk0 = nth (one chunk per thread, each covering n/nth weight rows).
//     Result: threads are load-balanced across output neurons.
//   Prefill (m > 1):
//     chunk_size = 16, both axes chunked.
//     Work-stealing distributes chunks dynamically across threads.

/// Translated outer dispatch without ggml tensor graph overhead.
///
/// Arguments match mul_mat_one_chunk; additionally:
///   nth   number of worker threads
///   ith   this thread's 0-based index
///   chunk_counter  shared atomic used for work-stealing (init to `nth` by thread 0)
///
/// Caller is responsible for:
///   - Quantizing src1 to src1_q before calling (barrier across threads).
///   - Initialising chunk_counter.store(nth) in thread 0 before all threads enter.
pub fn mul_mat_dispatch_chunk(
    src0: &[u8],
    src1_q: &[u8],
    dst: &mut [f32],
    n: usize,
    m: usize,
    k: usize,
    weight_type: GgmlType,
    nth: usize,
    ith: usize,
    chunk_counter: &AtomicI64,
) {
    let nr0 = n;
    let nr1 = m;

    // chunk_size: 64 for decode (one of the dims is 1), 16 for prefill.
    let chunk_size: usize = if nr0 == 1 || nr1 == 1 { 64 } else { 16 };

    let mut nchunk0 = nr0.div_ceil(chunk_size);
    let mut nchunk1 = nr1.div_ceil(chunk_size);

    // NUMA / poor-chunk override: redistribute by thread when chunks are sparse.
    if nchunk0 * nchunk1 < nth * 4 {
        if nr0 > nr1 {
            nchunk0 = nth;
            nchunk1 = 1;
        } else {
            nchunk0 = 1;
            nchunk1 = nth;
        }
    }

    let dr0 = nr0.div_ceil(nchunk0);
    let dr1 = nr1.div_ceil(nchunk1);

    let vec_dot_num_rows = nrows_for_type(weight_type);

    let total_chunks = (nchunk0 * nchunk1) as i64;
    let mut current_chunk = ith as i64;

    while current_chunk < total_chunks {
        let ith0 = (current_chunk % nchunk0 as i64) as usize;
        let ith1 = (current_chunk / nchunk0 as i64) as usize;

        let ir0_start = dr0 * ith0;
        let ir0_end = (ir0_start + dr0).min(nr0);
        let ir1_start = dr1 * ith1;
        let ir1_end = (ir1_start + dr1).min(nr1);

        // Guard: fall back to nrows=1 if any dimension is not 2-aligned.
        let num_rows_per_vec_dot = if vec_dot_num_rows == 2
            && nr0 % 2 == 0
            && nr1 % 2 == 0
            && (ir0_end - ir0_start) % 2 == 0
            && (ir1_end - ir1_start) % 2 == 0
        {
            2
        } else {
            1
        };

        mul_mat_one_chunk(
            src0,
            src1_q,
            dst,
            n,
            m,
            k,
            weight_type,
            num_rows_per_vec_dot,
            ir0_start,
            ir0_end,
            ir1_start,
            ir1_end,
        );

        if nth as i64 >= total_chunks {
            break;
        }
        current_chunk = chunk_counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// Single-threaded wrapper for testing / reference.
///
/// Translates the full ggml_compute_forward_mul_mat flow for a single thread:
///   1. Quantize activations
///   2. Dispatch chunks (all, sequentially)
pub fn mul_mat_llama(
    src0_type: GgmlType,
    src0: &[u8],
    src1: &[f32],
    dst: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let vdt = src0_type.vec_dot_type();
    let act_row_bytes = quantized_row_size(k, vdt);

    // Step 2: quantize activations (ggml line 1323-1366, single-thread version).
    // In ggml this is parallelised per-block across threads sharing the same
    // output activation row. Here we process all rows serially.
    let mut src1_q = vec![0u8; m * act_row_bytes];
    for row in 0..m {
        let src_row = &src1[row * k..(row + 1) * k];
        let dst_row = &mut src1_q[row * act_row_bytes..(row + 1) * act_row_bytes];
        quantize_activations(src_row, dst_row, vdt);
    }

    // Step 6: dispatch all chunks (nth=1, ith=0).
    let counter = AtomicI64::new(1); // init to nth (=1) as thread 0 would do
    mul_mat_dispatch_chunk(src0, &src1_q, dst, n, m, k, src0_type, 1, 0, &counter);
}

#[cfg(test)]
mod tests {
    use super::super::GgmlType;
    use super::super::quantize::quantized_row_size;
    use super::*;

    #[test]
    fn mul_mat_one_chunk_zero_weights() {
        // m=1 (decode), all-zero buffers → all-zero output.
        // Exercises the chunk dispatch path without a weight quantizer.
        let n = 8;
        let m = 1;
        let k = 256;

        let weight_type = GgmlType::Q4K;
        let weight_row_bytes = quantized_row_size(k, weight_type);
        let act_row_bytes = quantized_row_size(k, weight_type.vec_dot_type());

        let src0 = vec![0u8; n * weight_row_bytes];
        let src1_q = vec![0u8; m * act_row_bytes];
        let mut dst = vec![0.0f32; m * n];

        // chunk_size=64 for decode (nr1=1). With n=8 < 64, the single chunk
        // covers the entire weight-row axis.
        mul_mat_one_chunk(
            &src0, &src1_q, &mut dst, n, m, k, weight_type,
            1,    // num_rows_per_vec_dot
            0, n, // ir0
            0, m, // ir1
        );
        for v in &dst {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn mul_mat_llama_zero() {
        // Full dispatch with zero weights/activations. Result must be 0.
        let n = 16;
        let m = 4;
        let k = 256;
        let src0 = vec![0u8; n * quantized_row_size(k, GgmlType::Q4K)];
        let src1 = vec![0.0f32; m * k];
        let mut dst = vec![0.0f32; m * n];
        mul_mat_llama(GgmlType::Q4K, &src0, &src1, &mut dst, m, n, k);
        for v in &dst {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn mul_mat_llama_q8_0_roundtrip() {
        // Use Q8_0 (activations AND weights both Q8_0) so we can quantize
        // weights via quantize_row_q8_0 and verify the dispatch produces a
        // result close to the naive f32 matmul.
        use super::super::blocks::BlockQ8_0;
        use super::super::quantize::quantize_row_q8_0;

        let n = 4;
        let m = 2;
        let k = 32; // one Q8_0 block

        let weights_f32: Vec<f32> = (0..n * k)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let acts_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.2)
            .collect();

        // Quantize weight rows to Q8_0.
        let weight_row_bytes = quantized_row_size(k, GgmlType::Q8_0);
        let mut src0 = vec![0u8; n * weight_row_bytes];
        for row in 0..n {
            let src_row = &weights_f32[row * k..(row + 1) * k];
            let dst_bytes = &mut src0[row * weight_row_bytes..(row + 1) * weight_row_bytes];
            let dst_blocks = unsafe {
                std::slice::from_raw_parts_mut(
                    dst_bytes.as_mut_ptr() as *mut BlockQ8_0,
                    k / 32,
                )
            };
            quantize_row_q8_0(src_row, dst_blocks);
        }

        // Naive reference: f32 weights × f32 activations.
        let mut ref_dst = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += acts_f32[i * k + l] * weights_f32[j * k + l];
                }
                ref_dst[i * n + j] = sum;
            }
        }

        let mut got_dst = vec![0.0f32; m * n];
        mul_mat_llama(GgmlType::Q8_0, &src0, &acts_f32, &mut got_dst, m, n, k);

        for i in 0..m * n {
            let diff = (got_dst[i] - ref_dst[i]).abs();
            // Q8_0 introduces very small error (~0.4%).
            assert!(
                diff < ref_dst[i].abs() * 0.02 + 0.005,
                "output[{i}]: got={} ref={} diff={}",
                got_dst[i], ref_dst[i], diff
            );
        }
    }

    #[test]
    fn chunk_size_decode_vs_prefill() {
        // Documents the ggml chunk_size selection logic.
        let chunk_size_for = |nr0: usize, nr1: usize| -> usize {
            if nr0 == 1 || nr1 == 1 { 64 } else { 16 }
        };
        assert_eq!(chunk_size_for(4096, 1), 64);   // decode (single token)
        assert_eq!(chunk_size_for(1, 4096), 64);   // single-neuron (degenerate)
        assert_eq!(chunk_size_for(4096, 8), 16);   // prefill 8 tokens
        assert_eq!(chunk_size_for(4096, 128), 16); // prefill 128 tokens
    }
}
