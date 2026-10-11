/*
 * oxide-gemv-q4k: Hand-optimized Q4_K decode GEMV kernel via rustc-codegen-cuda.
 *
 * Build:  cargo oxide build --manifest-path tools/oxide-gemv-q4k/Cargo.toml
 * The compiled binary embeds PTX in the `.oxart` ELF section.
 * The tempura-kernels build.rs extracts that PTX and writes it to OUT_DIR.
 *
 * Q4_K packed GPU format (42 u32 per block, 256 elements per block):
 *   u32[0]      = f32 bits of d (super-scale, f16 → f32 at upload)
 *   u32[1]      = f32 bits of dmin (super-min, f16 → f32 at upload)
 *   u32[2+j]    = sc[2*j] | (sc[2*j+1]<<8) for j=0..3 (group scale pairs, 8-bit each)
 *   u32[6+j]    = m[2*j]  | (m[2*j+1]<<8)  for j=0..3 (group min pairs, 8-bit each)
 *   u32[10..42] = 32 u32s of packed qs bytes (128 bytes = 256 nibbles, 4-bit each)
 *
 * Dequant for element `within` in a block:
 *   j       = within / 64                    — which j-pair (0..3)
 *   in_j    = within % 64
 *   is_hi   = in_j >= 32
 *   l       = in_j % 32
 *   sc_group = 2*j + is_hi  (0..7)
 *   sc = (packed[base + 2 + j] >> ((sc_group%2)*8)) & 0xFF  as f32
 *   m  = (packed[base + 6 + j] >> ((sc_group%2)*8)) & 0xFF  as f32
 *   nibble = lo nibble if !is_hi, hi nibble if is_hi  (from qs[j*32+l])
 *   weight = d * sc * nibble - dmin * m
 *
 * v2 optimization (shfl-based scale sharing):
 *   One warp (32 lanes) processes 256 elements of one block per iteration.
 *   Each lane handles elements: l, l+32, l+64, …, l+224 within the block.
 *   - d, dmin: lane 0 loads once; all lanes receive via shfl.
 *   - sc/m:    lanes 0..3 each load one u32 (2 scale/min values); broadcast via shfl.
 *   - qs:      each lane loads one u32 per j-group (contains lo+hi nibbles for that lane).
 *   The 4 j-groups are fully unrolled, each processing 2 elements per lane.
 */

use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_core::simt::LaunchConfig;

const WARPS_PER_BLOCK: u32 = 4;

#[cuda_module]
mod kernels {
    use super::*;

    /// Q4_K GEMV for M=1 decode: out[N] = dequant(packed[N, K_BLOCKS, 42]) × x[K].
    ///
    /// Grid:  (ceil(N / WARPS_PER_BLOCK),) blocks — each warp owns one output neuron.
    /// Block: (WARPS_PER_BLOCK * 32,) threads.
    ///
    /// Fallback kernel: per-element stride-32 loop, no shfl.
    #[kernel]
    pub fn q4k_gemv_decode(
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

            let base = (n_out * k_blocks + blk) * 42;

            let d    = f32::from_bits(packed[base as usize]);
            let dmin = f32::from_bits(packed[(base + 1) as usize]);

            let j     = within / 64;
            let in_j  = within % 64;
            let is_hi = in_j >= 32;
            let l     = in_j % 32;

            let qs_byte_idx = j * 32 + l;
            let qs_word_idx = qs_byte_idx / 4;
            let qs_byte_pos = qs_byte_idx % 4;
            let qs_word = packed[(base + 10 + qs_word_idx) as usize];
            let qs_byte = (qs_word >> (qs_byte_pos * 8)) & 0xFF;
            let nibble = if is_hi { qs_byte >> 4 } else { qs_byte & 0xF };

            let sc_group = 2 * j + is_hi as u32;
            let sc_pair_word = packed[(base + 2 + sc_group / 2) as usize];
            let m_pair_word  = packed[(base + 6 + sc_group / 2) as usize];
            let sc_shift = (sc_group % 2) * 8;
            let sc_u32 = (sc_pair_word >> sc_shift) & 0xFF;
            let m_u32  = (m_pair_word  >> sc_shift) & 0xFF;

            let w = d * (sc_u32 as f32) * (nibble as f32) - dmin * (m_u32 as f32);

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

    /// Q4_K GEMV v2: shfl-based scale/min sharing, block-at-a-time loop.
    ///
    /// Optimization over v1:
    ///   - d, dmin: loaded once by lane 0, broadcast via shfl (32x fewer global loads).
    ///   - sc, m:   lanes 0..3 load 4 u32s; each holds 2 scale/min values.
    ///              Each j-group broadcasts its pair via shfl(sc_raw, j).
    ///   - qs:      one u32 load per lane per j-group (serves both lo and hi nibbles).
    ///   - 4 j-groups fully unrolled; 2 accumulation steps per group per lane.
    #[kernel]
    pub fn q4k_gemv_decode_v2(
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
            let base = (n_out * k_blocks + blk) * 42;

            // d and dmin: lane 0 loads, all lanes receive via full-warp shfl.
            let d_raw    = if lane == 0 { packed[base as usize] } else { 0 };
            let dmin_raw = if lane == 0 { packed[(base + 1) as usize] } else { 0 };
            let d    = f32::from_bits(warp::shuffle(d_raw, 0));
            let dmin = f32::from_bits(warp::shuffle(dmin_raw, 0));

            // Lanes 0..3 load sc/m pair words; each u32 packs two 8-bit values.
            // Lane j: sc_raw = sc[2*j] | (sc[2*j+1] << 8)
            // Lane j: m_raw  = m[2*j]  | (m[2*j+1]  << 8)
            let sc_raw = if lane < 4 { packed[(base + 2 + lane) as usize] } else { 0 };
            let m_raw  = if lane < 4 { packed[(base + 6 + lane) as usize] } else { 0 };

            // j=0: sc_groups 0 (lo) and 1 (hi), elements within 0..31 and 32..63.
            {
                let sc_pair = warp::shuffle(sc_raw, 0);
                let m_pair  = warp::shuffle(m_raw, 0);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;

                // qs_byte_idx = lane (j=0, l=lane), word_idx = lane/4.
                let qs_word = packed[(base + 10 + lane / 4) as usize];
                let qs_shift = (lane % 4) * 8;
                let qs_byte = (qs_word >> qs_shift) & 0xFF;
                let nibble_lo = (qs_byte & 0xF) as f32;
                let nibble_hi = ((qs_byte >> 4) & 0xF) as f32;

                let elem0 = blk * 256 + lane;
                let elem1 = blk * 256 + 32 + lane;

                acc += (d * sc_lo * nibble_lo - dmin * m_lo) * x[elem0 as usize];
                acc += (d * sc_hi * nibble_hi - dmin * m_hi) * x[elem1 as usize];
            }

            // j=1: sc_groups 2 (lo) and 3 (hi), elements within 64..95 and 96..127.
            {
                let sc_pair = warp::shuffle(sc_raw, 1);
                let m_pair  = warp::shuffle(m_raw, 1);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;

                // qs_byte_idx = 32+lane, word_idx = 8+lane/4.
                let qs_word = packed[(base + 10 + 8 + lane / 4) as usize];
                let qs_shift = (lane % 4) * 8;
                let qs_byte = (qs_word >> qs_shift) & 0xFF;
                let nibble_lo = (qs_byte & 0xF) as f32;
                let nibble_hi = ((qs_byte >> 4) & 0xF) as f32;

                let elem2 = blk * 256 + 64 + lane;
                let elem3 = blk * 256 + 96 + lane;

                acc += (d * sc_lo * nibble_lo - dmin * m_lo) * x[elem2 as usize];
                acc += (d * sc_hi * nibble_hi - dmin * m_hi) * x[elem3 as usize];
            }

            // j=2: sc_groups 4 (lo) and 5 (hi), elements within 128..159 and 160..191.
            {
                let sc_pair = warp::shuffle(sc_raw, 2);
                let m_pair  = warp::shuffle(m_raw, 2);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;

                let qs_word = packed[(base + 10 + 16 + lane / 4) as usize];
                let qs_shift = (lane % 4) * 8;
                let qs_byte = (qs_word >> qs_shift) & 0xFF;
                let nibble_lo = (qs_byte & 0xF) as f32;
                let nibble_hi = ((qs_byte >> 4) & 0xF) as f32;

                let elem4 = blk * 256 + 128 + lane;
                let elem5 = blk * 256 + 160 + lane;

                acc += (d * sc_lo * nibble_lo - dmin * m_lo) * x[elem4 as usize];
                acc += (d * sc_hi * nibble_hi - dmin * m_hi) * x[elem5 as usize];
            }

            // j=3: sc_groups 6 (lo) and 7 (hi), elements within 192..223 and 224..255.
            {
                let sc_pair = warp::shuffle(sc_raw, 3);
                let m_pair  = warp::shuffle(m_raw, 3);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;

                let qs_word = packed[(base + 10 + 24 + lane / 4) as usize];
                let qs_shift = (lane % 4) * 8;
                let qs_byte = (qs_word >> qs_shift) & 0xFF;
                let nibble_lo = (qs_byte & 0xF) as f32;
                let nibble_hi = ((qs_byte >> 4) & 0xF) as f32;

                let elem6 = blk * 256 + 192 + lane;
                let elem7 = blk * 256 + 224 + lane;

                acc += (d * sc_lo * nibble_lo - dmin * m_lo) * x[elem6 as usize];
                acc += (d * sc_hi * nibble_hi - dmin * m_hi) * x[elem7 as usize];
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

    // Small problem: N=32 outputs, K_BLOCKS=1 (K=256).
    let n: usize = 32;
    let k_blocks: usize = 1;
    let k = k_blocks * 256;

    // Activation: x[i] = 1.0.
    let x_host: Vec<f32> = vec![1.0_f32; k];

    // Build packed weights:
    //   d = 1.0, dmin = 0.0
    //   scales: all groups sc[g] = 1 (u8), m[g] = 0 (u8)
    //   qs: all nibbles = 3 → packed byte = 0x33
    // Expected weight per element: 1.0 * 1.0 * 3 - 0.0 * 0.0 = 3.0
    // Expected output: sum over 256 elements of 3.0 * 1.0 = 768.0
    let packed_u32s = n * k_blocks * 42;
    let mut packed_host = vec![0u32; packed_u32s];
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 42;
            packed_host[base]     = 1.0_f32.to_bits();
            packed_host[base + 1] = 0.0_f32.to_bits();
            // sc[g]=1, m[g]=0 for all 8 groups (stored as pairs of bytes).
            // u32[2+j] = 1 | (1<<8) = 0x00000101
            // u32[6+j] = 0
            for j in 0..4usize {
                packed_host[base + 2 + j] = 0x00000101_u32;
                packed_host[base + 6 + j] = 0x00000000_u32;
            }
            // nibble=3: lo=3, hi=3, packed byte = 0x33 → u32 = 0x33333333
            for w in 0..32usize {
                packed_host[base + 10 + w] = 0x33333333_u32;
            }
        }
    }

    let x_dev        = DeviceBuffer::from_host(&stream, &x_host).expect("upload x");
    let mut p_dev    = DeviceBuffer::from_host(&stream, &packed_host).expect("upload packed");
    let mut out_dev  = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc out");
    let mut out_dev2 = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc out2");

    let module = kernels::load(&ctx).expect("load oxide q4k module");

    let wpb = WARPS_PER_BLOCK as usize;
    let grid_x = (n + wpb - 1) / wpb;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x as u32, 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    // ── Test 1: v1 and v2 both pass for nibble=3,sc=1,d=1,dmin=0 ──────────────
    unsafe {
        module.q4k_gemv_decode(
            stream.as_ref(), cfg, &mut out_dev, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v1 launch");

    unsafe {
        module.q4k_gemv_decode_v2(
            stream.as_ref(), cfg, &mut out_dev2, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v2 launch");

    let result1 = out_dev.to_host_vec(&stream).expect("download v1");
    let result2 = out_dev2.to_host_vec(&stream).expect("download v2");
    let expected = 768.0_f32;

    let max_err1 = result1.iter().map(|v| (v - expected).abs()).fold(0.0_f32, f32::max);
    let max_err2 = result2.iter().map(|v| (v - expected).abs()).fold(0.0_f32, f32::max);

    if max_err1 < 1.0 {
        println!(
            "oxide-gemv-q4k v1 test 1 (nibble=3,sc=1,d=1,dmin=0): PASSED (expected={}, max_err={:.2e})",
            expected, max_err1
        );
    } else {
        eprintln!(
            "oxide-gemv-q4k v1 test 1 FAILED (expected={}, max_err={:.4}, result[0]={})",
            expected, max_err1, result1[0]
        );
        std::process::exit(1);
    }

    if max_err2 < 1.0 {
        println!(
            "oxide-gemv-q4k v2 test 1 (nibble=3,sc=1,d=1,dmin=0): PASSED (expected={}, max_err={:.2e})",
            expected, max_err2
        );
    } else {
        eprintln!(
            "oxide-gemv-q4k v2 test 1 FAILED (expected={}, max_err={:.4}, result[0]={})",
            expected, max_err2, result2[0]
        );
        std::process::exit(1);
    }

    let diff12 = result1.iter().zip(result2.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f32, f32::max);
    if diff12 < 1.0 {
        println!("oxide-gemv-q4k v1/v2 test 1 agreement: PASSED (max_diff={:.2e})", diff12);
    } else {
        eprintln!("oxide-gemv-q4k v1/v2 test 1 DISAGREEMENT (max_diff={:.4})", diff12);
        std::process::exit(1);
    }

    // ── Test 2: d=2.0, sc=3, nibble=4, dmin=0.5, m=2 ─────────────────────────
    // weight = 2.0 * 3.0 * 4.0 - 0.5 * 2.0 = 24.0 - 1.0 = 23.0
    // sum over 256 elements = 256 * 23.0 = 5888.0
    let mut packed_host2 = vec![0u32; packed_u32s];
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 42;
            packed_host2[base]     = 2.0_f32.to_bits();
            packed_host2[base + 1] = 0.5_f32.to_bits();
            for j in 0..4usize {
                packed_host2[base + 2 + j] = 0x00000303_u32;
                packed_host2[base + 6 + j] = 0x00000202_u32;
            }
            for w in 0..32usize {
                packed_host2[base + 10 + w] = 0x44444444_u32;
            }
        }
    }
    p_dev.copy_from_host(&stream, &packed_host2).expect("re-upload");

    let zeros = vec![0.0_f32; n];
    out_dev.copy_from_host(&stream, &zeros).expect("zero out");
    out_dev2.copy_from_host(&stream, &zeros).expect("zero out2");

    unsafe {
        module.q4k_gemv_decode(
            stream.as_ref(), cfg, &mut out_dev, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v1 launch 2");

    unsafe {
        module.q4k_gemv_decode_v2(
            stream.as_ref(), cfg, &mut out_dev2, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v2 launch 2");

    let result1b = out_dev.to_host_vec(&stream).expect("download v1 t2");
    let result2b = out_dev2.to_host_vec(&stream).expect("download v2 t2");
    let expected2 = 5888.0_f32;

    let max_err1b = result1b.iter().map(|v| (v - expected2).abs()).fold(0.0_f32, f32::max);
    let max_err2b = result2b.iter().map(|v| (v - expected2).abs()).fold(0.0_f32, f32::max);

    if max_err1b < 1.0 {
        println!(
            "oxide-gemv-q4k v1 test 2 (d=2,sc=3,nibble=4,dmin=0.5,m=2): PASSED (expected={}, max_err={:.2e})",
            expected2, max_err1b
        );
    } else {
        eprintln!(
            "oxide-gemv-q4k v1 test 2 FAILED (expected={}, max_err={:.4}, result[0]={})",
            expected2, max_err1b, result1b[0]
        );
        std::process::exit(1);
    }

    if max_err2b < 1.0 {
        println!(
            "oxide-gemv-q4k v2 test 2 (d=2,sc=3,nibble=4,dmin=0.5,m=2): PASSED (expected={}, max_err={:.2e})",
            expected2, max_err2b
        );
    } else {
        eprintln!(
            "oxide-gemv-q4k v2 test 2 FAILED (expected={}, max_err={:.4}, result[0]={})",
            expected2, max_err2b, result2b[0]
        );
        std::process::exit(1);
    }

    let diff12b = result1b.iter().zip(result2b.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f32, f32::max);
    if diff12b < 1.0 {
        println!("oxide-gemv-q4k v1/v2 test 2 agreement: PASSED (max_diff={:.2e})", diff12b);
    } else {
        eprintln!("oxide-gemv-q4k v1/v2 test 2 DISAGREEMENT (max_diff={:.4})", diff12b);
        std::process::exit(1);
    }

    // ── Test 3: multi-block (K_BLOCKS=4) ──────────────────────────────────────
    // weight = 1.0 * 2.0 * 5.0 - 0.0 = 10.0, sum = 256*4 * 10.0 = 10240.0
    let k_blocks3: usize = 4;
    let k3 = k_blocks3 * 256;
    let x3: Vec<f32> = vec![1.0_f32; k3];
    let packed_u32s3 = n * k_blocks3 * 42;
    let mut ph3 = vec![0u32; packed_u32s3];
    for row in 0..n {
        for blk in 0..k_blocks3 {
            let base = (row * k_blocks3 + blk) * 42;
            ph3[base]     = 1.0_f32.to_bits();
            ph3[base + 1] = 0.0_f32.to_bits();
            for j in 0..4usize {
                ph3[base + 2 + j] = 0x00000202_u32;
                ph3[base + 6 + j] = 0x00000000_u32;
            }
            for w in 0..32usize {
                ph3[base + 10 + w] = 0x55555555_u32;
            }
        }
    }
    let expected3 = 10240.0_f32;

    let x_dev3    = DeviceBuffer::from_host(&stream, &x3).expect("upload x3");
    let p_dev3    = DeviceBuffer::from_host(&stream, &ph3).expect("upload p3");
    let mut od1_3 = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc od1_3");
    let mut od2_3 = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc od2_3");

    let grid3 = (n + wpb - 1) / wpb;
    let cfg3 = LaunchConfig {
        grid_dim:  (grid3 as u32, 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        module.q4k_gemv_decode(
            stream.as_ref(), cfg3, &mut od1_3, &x_dev3, &p_dev3, n as u32, k_blocks3 as u32,
        )
    }.expect("v1 launch 3");

    unsafe {
        module.q4k_gemv_decode_v2(
            stream.as_ref(), cfg3, &mut od2_3, &x_dev3, &p_dev3, n as u32, k_blocks3 as u32,
        )
    }.expect("v2 launch 3");

    let r1_3 = od1_3.to_host_vec(&stream).expect("download v1 t3");
    let r2_3 = od2_3.to_host_vec(&stream).expect("download v2 t3");

    let e1_3 = r1_3.iter().map(|v| (v - expected3).abs()).fold(0.0_f32, f32::max);
    let e2_3 = r2_3.iter().map(|v| (v - expected3).abs()).fold(0.0_f32, f32::max);
    let d3   = r1_3.iter().zip(r2_3.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f32, f32::max);

    if e1_3 < 1.0 {
        println!("oxide-gemv-q4k v1 test 3 (multi-block): PASSED (expected={}, max_err={:.2e})", expected3, e1_3);
    } else {
        eprintln!("oxide-gemv-q4k v1 test 3 FAILED (max_err={:.4}, result[0]={})", e1_3, r1_3[0]);
        std::process::exit(1);
    }
    if e2_3 < 1.0 {
        println!("oxide-gemv-q4k v2 test 3 (multi-block): PASSED (expected={}, max_err={:.2e})", expected3, e2_3);
    } else {
        eprintln!("oxide-gemv-q4k v2 test 3 FAILED (max_err={:.4}, result[0]={})", e2_3, r2_3[0]);
        std::process::exit(1);
    }
    if d3 < 1.0 {
        println!("oxide-gemv-q4k v1/v2 test 3 agreement: PASSED (max_diff={:.2e})", d3);
    } else {
        eprintln!("oxide-gemv-q4k v1/v2 test 3 DISAGREEMENT (max_diff={:.4})", d3);
        std::process::exit(1);
    }

    println!("oxide-gemv-q4k: all tests PASSED");
}
