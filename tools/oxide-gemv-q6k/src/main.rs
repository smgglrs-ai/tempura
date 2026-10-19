/*
 * oxide-gemv-q6k: Hand-optimized Q6_K decode GEMV kernel via rustc-codegen-cuda.
 *
 * Build:  cargo oxide build --manifest-path tools/oxide-gemv-q6k/Cargo.toml
 * The compiled binary embeds PTX in the `.oxart` ELF section.
 * The tempura-kernels build.rs extracts that PTX and writes it to OUT_DIR.
 *
 * Q6_K packed GPU format (53 u32 per block, 256 elements per block):
 *   u32[0]      = f32 bits of d (super-scale, f16 → f32 at upload)
 *   u32[1..5]   = 4 u32s: 16 group scales as bias-128 u8 (4 per u32)
 *                 stored_byte = (scale_i8 + 128) as u8; decode: byte as f32 - 128.0
 *   u32[5..37]  = 32 u32s: ql[0..128] in native GGUF order (4 u8 per u32)
 *   u32[37..53] = 16 u32s: qh[0..64] in native GGUF order (4 u8 per u32)
 *
 * Dequant for element `within` in a block:
 *   half = within/128, sub = (within%128)/32, l = within%32, group = within/16
 *   sc_byte = (packed[base+1+group/4] >> ((group%4)*8)) & 0xFF; sc = sc_byte - 128.0
 *   ql_byte_idx = half*64 + l + (sub%2)*32
 *   ql_byte = (packed[base+5+ql_byte_idx/4] >> ((ql_byte_idx%4)*8)) & 0xFF
 *   lo = (ql_byte >> ((sub/2)*4)) & 0xF
 *   qh_byte_idx = half*32 + l
 *   qh_byte = (packed[base+37+qh_byte_idx/4] >> ((qh_byte_idx%4)*8)) & 0xFF
 *   hi = (qh_byte >> (sub*2)) & 0x3
 *   q6 = lo | (hi<<4); weight = d * sc * (q6 as f32 - 32.0)
 *
 * v2 optimization: shfl d broadcast + 8 unrolled steps + cached ql/qh loads.
 *   For step s (0..7): half = s/4, sub = s%4, l = lane (always lane in a warp).
 *   Cached ql words per half: ql_a[lane] (steps 0,2), ql_b[lane+32] (steps 1,3),
 *                              ql_c[lane+64] (steps 4,6), ql_d[lane+96] (steps 5,7).
 *   Cached qh words: qh_0[lane] (steps 0-3), qh_1[lane+32] (steps 4-7).
 *   Scale words: sc_01 (groups 0-3, steps 0-1), sc_23 (4-7, steps 2-3),
 *                sc_45 (8-11, steps 4-5), sc_67 (12-15, steps 6-7).
 */

use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_core::simt::LaunchConfig;

const WARPS_PER_BLOCK: u32 = 4;

#[cuda_module]
mod kernels {
    use super::*;

    /// Q6_K GEMV for M=1 decode: out[N] = dequant(packed[N, K_BLOCKS, 53]) × x[K].
    ///
    /// Grid:  (ceil(N / WARPS_PER_BLOCK),) blocks — each warp owns one output neuron.
    /// Block: (WARPS_PER_BLOCK * 32,) threads.
    ///
    /// Fallback kernel: per-element stride-32 loop with native ql+qh dequant.
    #[kernel]
    pub fn q6k_gemv_decode(
        mut out: DisjointSlice<f32>,
        x: &[f32],
        packed: &[u32],
        n: u32,
        k_blocks: u32,
    ) {
        let tid     = thread::threadIdx_x();
        let bid     = thread::blockIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;
        let n_out   = bid * WARPS_PER_BLOCK + warp_id;

        if n_out >= n {
            return;
        }

        let mut acc = 0.0_f32;
        let k = k_blocks * 256;

        let mut elem = lane;
        while elem < k {
            let blk    = elem / 256;
            let within = elem % 256;

            let base = (n_out * k_blocks + blk) * 53;

            let d = f32::from_bits(packed[base as usize]);

            let half  = within / 128;
            let sub   = (within % 128) / 32;
            let l     = within % 32;
            let group = within / 16;

            let sc_byte = (packed[(base + 1 + group / 4) as usize] >> ((group % 4) * 8)) & 0xFF;
            let sc = sc_byte as f32 - 128.0;

            let ql_byte_idx = half * 64 + l + (sub % 2) * 32;
            let ql_u32 = packed[(base + 5 + ql_byte_idx / 4) as usize];
            let ql_byte = (ql_u32 >> ((ql_byte_idx % 4) * 8)) & 0xFF;
            let lo = (ql_byte >> ((sub / 2) * 4)) & 0xF;

            let qh_byte_idx = half * 32 + l;
            let qh_u32 = packed[(base + 37 + qh_byte_idx / 4) as usize];
            let qh_byte = (qh_u32 >> ((qh_byte_idx % 4) * 8)) & 0xFF;
            let hi = (qh_byte >> (sub * 2)) & 0x3;

            let q6 = lo | (hi << 4);
            let w = d * sc * (q6 as f32 - 32.0);

            acc += w * x[elem as usize];

            elem += 32;
        }

        acc = warp::reduce_sum_f32(acc);
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(n_out as usize) = acc;
            }
        }
    }

    /// Q6_K GEMV v2: shfl d broadcast + 8 unrolled steps with cached ql/qh loads.
    ///
    /// Optimization over v1:
    ///   - d: loaded once by lane 0 per block, broadcast via shfl.
    ///   - 8 steps per block fully unrolled (better ILP).
    ///   - ql/qh cached: 6 word loads per block (steps 0&2 share ql_a, 1&3 share ql_b,
    ///     0-3 share qh_0; same pattern for half=1 with ql_c, ql_d, qh_1).
    ///   - scale: 4 word loads per block (sc_01 covers groups 0-3, etc.)
    ///   - coalesced: 32 lanes access 8 consecutive u32s for each ql/qh section.
    #[kernel]
    pub fn q6k_gemv_decode_v2(
        mut out: DisjointSlice<f32>,
        x: &[f32],
        packed: &[u32],
        n: u32,
        k_blocks: u32,
    ) {
        let tid     = thread::threadIdx_x();
        let bid     = thread::blockIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;
        let n_out   = bid * WARPS_PER_BLOCK + warp_id;

        if n_out >= n {
            return;
        }

        let mut acc = 0.0_f32;

        let mut blk = 0u32;
        while blk < k_blocks {
            let base = (n_out * k_blocks + blk) * 53;

            // d: lane 0 loads once, broadcast to all 32 lanes via shfl.
            let d_raw = if lane == 0 { packed[base as usize] } else { 0 };
            let d = f32::from_bits(warp::shuffle(d_raw, 0));

            // Scale words (groups 0-3, 4-7, 8-11, 12-15).
            let sc_01 = packed[(base + 1) as usize];
            let sc_23 = packed[(base + 2) as usize];
            let sc_45 = packed[(base + 3) as usize];
            let sc_67 = packed[(base + 4) as usize];

            // Per-lane byte position: (lane%4)*8 selects the byte within each u32.
            let byte_shift = (lane % 4) * 8;

            // Cache ql and qh per lane (6 loads, shared across pairs of steps):
            // ql_a = ql[lane]:    u32[5 + lane/4], steps 0 (lo) and 2 (hi)
            // ql_b = ql[lane+32]: u32[13 + lane/4], steps 1 (lo) and 3 (hi)
            // qh_0 = qh[lane]:    u32[37 + lane/4], steps 0..3
            // ql_c = ql[lane+64]: u32[21 + lane/4], steps 4 (lo) and 6 (hi)
            // ql_d = ql[lane+96]: u32[29 + lane/4], steps 5 (lo) and 7 (hi)
            // qh_1 = qh[lane+32]: u32[45 + lane/4], steps 4..7
            let ql_a = (packed[(base + 5  + lane / 4) as usize] >> byte_shift) & 0xFF;
            let ql_b = (packed[(base + 13 + lane / 4) as usize] >> byte_shift) & 0xFF;
            let qh_0 = (packed[(base + 37 + lane / 4) as usize] >> byte_shift) & 0xFF;
            let ql_c = (packed[(base + 21 + lane / 4) as usize] >> byte_shift) & 0xFF;
            let ql_d = (packed[(base + 29 + lane / 4) as usize] >> byte_shift) & 0xFF;
            let qh_1 = (packed[(base + 45 + lane / 4) as usize] >> byte_shift) & 0xFF;

            // step 0: half=0, sub=0, within=lane, group=lane/16 (0 or 1)
            {
                let within = lane;
                let group = within / 16;
                let sc_byte = (sc_01 >> (group * 8)) & 0xFF;
                let sc = sc_byte as f32 - 128.0;
                let lo = ql_a & 0xF;
                let hi = qh_0 & 0x3;
                let q6 = lo | (hi << 4);
                acc += d * sc * (q6 as f32 - 32.0) * x[(blk * 256 + within) as usize];
            }
            // step 1: half=0, sub=1, within=32+lane, group=(32+lane)/16 (2 or 3)
            {
                let within = 32 + lane;
                let group = within / 16;
                let sc_byte = (sc_01 >> (group * 8)) & 0xFF;
                let sc = sc_byte as f32 - 128.0;
                let lo = ql_b & 0xF;
                let hi = (qh_0 >> 2) & 0x3;
                let q6 = lo | (hi << 4);
                acc += d * sc * (q6 as f32 - 32.0) * x[(blk * 256 + within) as usize];
            }
            // step 2: half=0, sub=2, within=64+lane, group=(64+lane)/16 (4 or 5)
            {
                let within = 64 + lane;
                let group = within / 16;
                let sc_byte = (sc_23 >> ((group - 4) * 8)) & 0xFF;
                let sc = sc_byte as f32 - 128.0;
                let lo = (ql_a >> 4) & 0xF;
                let hi = (qh_0 >> 4) & 0x3;
                let q6 = lo | (hi << 4);
                acc += d * sc * (q6 as f32 - 32.0) * x[(blk * 256 + within) as usize];
            }
            // step 3: half=0, sub=3, within=96+lane, group=(96+lane)/16 (6 or 7)
            {
                let within = 96 + lane;
                let group = within / 16;
                let sc_byte = (sc_23 >> ((group - 4) * 8)) & 0xFF;
                let sc = sc_byte as f32 - 128.0;
                let lo = (ql_b >> 4) & 0xF;
                let hi = (qh_0 >> 6) & 0x3;
                let q6 = lo | (hi << 4);
                acc += d * sc * (q6 as f32 - 32.0) * x[(blk * 256 + within) as usize];
            }
            // step 4: half=1, sub=0, within=128+lane, group=(128+lane)/16 (8 or 9)
            {
                let within = 128 + lane;
                let group = within / 16;
                let sc_byte = (sc_45 >> ((group - 8) * 8)) & 0xFF;
                let sc = sc_byte as f32 - 128.0;
                let lo = ql_c & 0xF;
                let hi = qh_1 & 0x3;
                let q6 = lo | (hi << 4);
                acc += d * sc * (q6 as f32 - 32.0) * x[(blk * 256 + within) as usize];
            }
            // step 5: half=1, sub=1, within=160+lane, group=(160+lane)/16 (10 or 11)
            {
                let within = 160 + lane;
                let group = within / 16;
                let sc_byte = (sc_45 >> ((group - 8) * 8)) & 0xFF;
                let sc = sc_byte as f32 - 128.0;
                let lo = ql_d & 0xF;
                let hi = (qh_1 >> 2) & 0x3;
                let q6 = lo | (hi << 4);
                acc += d * sc * (q6 as f32 - 32.0) * x[(blk * 256 + within) as usize];
            }
            // step 6: half=1, sub=2, within=192+lane, group=(192+lane)/16 (12 or 13)
            {
                let within = 192 + lane;
                let group = within / 16;
                let sc_byte = (sc_67 >> ((group - 12) * 8)) & 0xFF;
                let sc = sc_byte as f32 - 128.0;
                let lo = (ql_c >> 4) & 0xF;
                let hi = (qh_1 >> 4) & 0x3;
                let q6 = lo | (hi << 4);
                acc += d * sc * (q6 as f32 - 32.0) * x[(blk * 256 + within) as usize];
            }
            // step 7: half=1, sub=3, within=224+lane, group=(224+lane)/16 (14 or 15)
            {
                let within = 224 + lane;
                let group = within / 16;
                let sc_byte = (sc_67 >> ((group - 12) * 8)) & 0xFF;
                let sc = sc_byte as f32 - 128.0;
                let lo = (ql_d >> 4) & 0xF;
                let hi = (qh_1 >> 6) & 0x3;
                let q6 = lo | (hi << 4);
                acc += d * sc * (q6 as f32 - 32.0) * x[(blk * 256 + within) as usize];
            }

            blk += 1;
        }

        // Butterfly warp-reduce.
        acc = warp::reduce_sum_f32(acc);
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(n_out as usize) = acc;
            }
        }
    }
}

// ── Host-side correctness test ────────────────────────────────────────────────

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();

    let n: usize = 32;
    let k_blocks: usize = 1;
    let k = k_blocks * 256;

    let x_host: Vec<f32> = vec![1.0_f32; k];

    // Build packed weights (53 u32/block, native ql+qh format):
    //   d = 1.0, all group scales = 1 (stored as bias-128 u8 = 129 = 0x81)
    //   all q_full = 32 (→ weight = 0.0)
    //   q_full=32 = 0x20: lo nibble = 0, hi 2-bits = 2
    //   ql: all nibbles = 0 → all ql bytes = 0x00
    //   qh: all 4 bit-pairs = 2 (=0b10) → qh byte = 0b10_10_10_10 = 0xAA
    // Expected output: 0.0 for all neurons.
    let packed_u32s = n * k_blocks * 53;
    let mut packed_host = vec![0u32; packed_u32s];
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 53;
            // slot 0: d = 1.0
            packed_host[base] = 1.0_f32.to_bits();
            // slots 1..5: 16 scales = 1 as bias-128 u8 → 1+128=129=0x81
            for w in 0..4usize {
                packed_host[base + 1 + w] = 0x81818181_u32;
            }
            // slots 5..37: ql = all 0x00 (all nibbles = 0)
            for w in 0..32usize {
                packed_host[base + 5 + w] = 0x00000000_u32;
            }
            // slots 37..53: qh = 0xAA (each pair of bits = 2 = 0b10)
            for w in 0..16usize {
                packed_host[base + 37 + w] = 0xAAAAAAAA_u32;
            }
        }
    }

    let x_dev        = DeviceBuffer::from_host(&stream, &x_host).expect("upload x");
    let mut p_dev    = DeviceBuffer::from_host(&stream, &packed_host).expect("upload packed");
    let mut out_dev  = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc out");
    let mut out_dev2 = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc out2");

    let module = kernels::load(&ctx).expect("load oxide q6k module");

    let wpb = WARPS_PER_BLOCK as usize;
    let grid_x = (n + wpb - 1) / wpb;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x as u32, 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    // ── Test 1: q_full=32 → weight=0.0 ────────────────────────────────────────
    unsafe {
        module.q6k_gemv_decode(
            stream.as_ref(), cfg, &mut out_dev, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v1 launch");

    unsafe {
        module.q6k_gemv_decode_v2(
            stream.as_ref(), cfg, &mut out_dev2, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v2 launch");

    let result1 = out_dev.to_host_vec(&stream).expect("download v1");
    let result2 = out_dev2.to_host_vec(&stream).expect("download v2");

    let max_err1 = result1.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let max_err2 = result2.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);

    if max_err1 < 1e-4 {
        println!("oxide-gemv-q6k v1 test 1 (q_full=32→weight=0): PASSED (max_err={:.2e})", max_err1);
    } else {
        eprintln!("oxide-gemv-q6k v1 test 1 FAILED (max_err={:.4}, result[0]={})", max_err1, result1[0]);
        std::process::exit(1);
    }
    if max_err2 < 1e-4 {
        println!("oxide-gemv-q6k v2 test 1 (q_full=32→weight=0): PASSED (max_err={:.2e})", max_err2);
    } else {
        eprintln!("oxide-gemv-q6k v2 test 1 FAILED (max_err={:.4}, result[0]={})", max_err2, result2[0]);
        std::process::exit(1);
    }

    let diff12 = result1.iter().zip(result2.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f32, f32::max);
    if diff12 < 1e-4 {
        println!("oxide-gemv-q6k v1/v2 test 1 agreement: PASSED (max_diff={:.2e})", diff12);
    } else {
        eprintln!("oxide-gemv-q6k v1/v2 test 1 DISAGREEMENT (max_diff={:.4})", diff12);
        std::process::exit(1);
    }

    // ── Test 2: d=1.0, scale=2, q_full=33 → weight=2.0, sum=512.0 ──────────
    // q_full=33 = 0x21: lo nibble = 1, hi 2-bits = 2
    //   ql: all nibbles = 1 → ql byte = 0x11 (lo nibble = 1, hi nibble = 1)
    //   qh: all pairs = 2 → qh byte = 0xAA (same as test 1)
    //   scale=2: bias-128 → 2+128=130=0x82
    // weight = d * sc * (q6 - 32) = 1 * 2 * (33 - 32) = 2.0
    // Expected: 256 elements × 2.0 = 512.0
    let mut packed_host2 = vec![0u32; packed_u32s];
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 53;
            packed_host2[base] = 1.0_f32.to_bits();
            for w in 0..4usize {
                packed_host2[base + 1 + w] = 0x82828282_u32; // scale=2 → 2+128=130=0x82
            }
            for w in 0..32usize {
                packed_host2[base + 5 + w] = 0x11111111_u32; // ql: all nibbles = 1
            }
            for w in 0..16usize {
                packed_host2[base + 37 + w] = 0xAAAAAAAA_u32; // qh: all pairs = 2
            }
        }
    }
    p_dev.copy_from_host(&stream, &packed_host2).expect("re-upload");

    let zeros = vec![0.0_f32; n];
    out_dev.copy_from_host(&stream, &zeros).expect("zero out");
    out_dev2.copy_from_host(&stream, &zeros).expect("zero out2");

    unsafe {
        module.q6k_gemv_decode(
            stream.as_ref(), cfg, &mut out_dev, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v1 launch 2");

    unsafe {
        module.q6k_gemv_decode_v2(
            stream.as_ref(), cfg, &mut out_dev2, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v2 launch 2");

    let result1b = out_dev.to_host_vec(&stream).expect("download v1 t2");
    let result2b = out_dev2.to_host_vec(&stream).expect("download v2 t2");
    let expected2 = 512.0_f32;

    let max_err1b = result1b.iter().map(|v| (v - expected2).abs()).fold(0.0_f32, f32::max);
    let max_err2b = result2b.iter().map(|v| (v - expected2).abs()).fold(0.0_f32, f32::max);

    if max_err1b < 1.0 {
        println!(
            "oxide-gemv-q6k v1 test 2 (d=1,sc=2,q_full=33): PASSED (expected={}, max_err={:.2e})",
            expected2, max_err1b
        );
    } else {
        eprintln!(
            "oxide-gemv-q6k v1 test 2 FAILED (expected={}, max_err={:.4}, result[0]={})",
            expected2, max_err1b, result1b[0]
        );
        std::process::exit(1);
    }
    if max_err2b < 1.0 {
        println!(
            "oxide-gemv-q6k v2 test 2 (d=1,sc=2,q_full=33): PASSED (expected={}, max_err={:.2e})",
            expected2, max_err2b
        );
    } else {
        eprintln!(
            "oxide-gemv-q6k v2 test 2 FAILED (expected={}, max_err={:.4}, result[0]={})",
            expected2, max_err2b, result2b[0]
        );
        std::process::exit(1);
    }

    let diff12b = result1b.iter().zip(result2b.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f32, f32::max);
    if diff12b < 1.0 {
        println!("oxide-gemv-q6k v1/v2 test 2 agreement: PASSED (max_diff={:.2e})", diff12b);
    } else {
        eprintln!("oxide-gemv-q6k v1/v2 test 2 DISAGREEMENT (max_diff={:.4})", diff12b);
        std::process::exit(1);
    }

    println!("oxide-gemv-q6k: all tests PASSED");
}
