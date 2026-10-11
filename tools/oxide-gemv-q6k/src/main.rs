/*
 * oxide-gemv-q6k: Hand-optimized Q6_K decode GEMV kernel via rustc-codegen-cuda.
 *
 * Build:  cargo oxide build --manifest-path tools/oxide-gemv-q6k/Cargo.toml
 * The compiled binary embeds PTX in the `.oxart` ELF section.
 * The tempura-kernels build.rs extracts that PTX and writes it to OUT_DIR.
 *
 * Q6_K packed GPU format (81 u32 per block, 256 elements per block):
 *   u32[0]      = f32 bits of d (super-scale, f16 → f32 at upload)
 *   u32[1..17]  = 16 f32 group scales (i8 → f32 preconverted at upload)
 *                 u32[1+g] = f32 bits of scale for group g (16 elements per group)
 *   u32[17..81] = 64 u32s of q_full bytes (4 bytes per u32, 4 elements per byte)
 *                 element i occupies byte (i%4) of u32[17 + i/4]
 *                 q_full ∈ [0, 63]; weight = d * scale[i/16] * (q_full - 32)
 *
 * Dequant for element `within` in a block:
 *   group  = within / 16
 *   scale  = f32::from_bits(packed[base + 1 + group])
 *   q_full = (packed[base + 17 + within/4] >> ((within%4)*8)) & 0xFF
 *   weight = d * scale * (q_full as f32 - 32.0)
 *
 * v2 optimization (shfl-based d sharing + block-at-a-time loop):
 *   One warp (32 lanes) processes all 256 elements of one block per outer iteration.
 *   - d: lane 0 loads once; all lanes receive via shfl.
 *   - 8 steps per block (256 / 32 = 8): within = step*32 + lane.
 *   - scale: loaded per-element (different per lane within each step), unavoidable.
 *   - q_full: perfectly coalesced (4 lanes share one u32).
 *   - 8 steps fully unrolled for better ILP and to expose scale-load independence.
 */

use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_core::simt::LaunchConfig;

const WARPS_PER_BLOCK: u32 = 4;

#[cuda_module]
mod kernels {
    use super::*;

    /// Q6_K GEMV for M=1 decode: out[N] = dequant(packed[N, K_BLOCKS, 81]) × x[K].
    ///
    /// Grid:  (ceil(N / WARPS_PER_BLOCK),) blocks — each warp owns one output neuron.
    /// Block: (WARPS_PER_BLOCK * 32,) threads.
    ///
    /// Fallback kernel: per-element stride-32 loop.
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

            let base = (n_out * k_blocks + blk) * 81;

            let d = f32::from_bits(packed[base as usize]);

            let group = within / 16;
            let scale = f32::from_bits(packed[(base + 1 + group) as usize]);

            let q_word     = packed[(base + 17 + within / 4) as usize];
            let q_byte_pos = within % 4;
            let q_full     = (q_word >> (q_byte_pos * 8)) & 0xFF;

            let w = d * scale * (q_full as f32 - 32.0);

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

    /// Q6_K GEMV v2: shfl d broadcast + block-at-a-time loop with 8 unrolled steps.
    ///
    /// Optimization over v1:
    ///   - d: loaded once by lane 0 per block, broadcast via shfl.
    ///   - 8 steps per block fully unrolled (better ILP, clearer to the compiler).
    ///   - scale loads and q_full loads remain per-lane (different per lane per step).
    ///   - q_full accesses are coalesced: 4 consecutive lanes share one u32.
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
            let base = (n_out * k_blocks + blk) * 81;

            // d: lane 0 loads once, broadcast to all 32 lanes via shfl.
            let d_raw = if lane == 0 { packed[base as usize] } else { 0 };
            let d = f32::from_bits(warp::shuffle(d_raw, 0));

            // 8 steps per block: within = step*32 + lane (for step = 0..7).
            // Each step: group = within/16, q_word = packed[base+17+within/4].
            // Fully unrolled to allow ILP across independent scale+q loads.

            // step 0: within = lane (0..31), group = lane/16
            {
                let within = lane;
                let group = within / 16;
                let scale = f32::from_bits(packed[(base + 1 + group) as usize]);
                let q_word = packed[(base + 17 + within / 4) as usize];
                let q_full = (q_word >> ((within % 4) * 8)) & 0xFF;
                let w = d * scale * (q_full as f32 - 32.0);
                acc += w * x[(blk * 256 + within) as usize];
            }
            // step 1: within = 32+lane, group = (32+lane)/16 = 2 + lane/16
            {
                let within = 32 + lane;
                let group = within / 16;
                let scale = f32::from_bits(packed[(base + 1 + group) as usize]);
                let q_word = packed[(base + 17 + within / 4) as usize];
                let q_full = (q_word >> ((within % 4) * 8)) & 0xFF;
                let w = d * scale * (q_full as f32 - 32.0);
                acc += w * x[(blk * 256 + within) as usize];
            }
            // step 2: within = 64+lane
            {
                let within = 64 + lane;
                let group = within / 16;
                let scale = f32::from_bits(packed[(base + 1 + group) as usize]);
                let q_word = packed[(base + 17 + within / 4) as usize];
                let q_full = (q_word >> ((within % 4) * 8)) & 0xFF;
                let w = d * scale * (q_full as f32 - 32.0);
                acc += w * x[(blk * 256 + within) as usize];
            }
            // step 3: within = 96+lane
            {
                let within = 96 + lane;
                let group = within / 16;
                let scale = f32::from_bits(packed[(base + 1 + group) as usize]);
                let q_word = packed[(base + 17 + within / 4) as usize];
                let q_full = (q_word >> ((within % 4) * 8)) & 0xFF;
                let w = d * scale * (q_full as f32 - 32.0);
                acc += w * x[(blk * 256 + within) as usize];
            }
            // step 4: within = 128+lane
            {
                let within = 128 + lane;
                let group = within / 16;
                let scale = f32::from_bits(packed[(base + 1 + group) as usize]);
                let q_word = packed[(base + 17 + within / 4) as usize];
                let q_full = (q_word >> ((within % 4) * 8)) & 0xFF;
                let w = d * scale * (q_full as f32 - 32.0);
                acc += w * x[(blk * 256 + within) as usize];
            }
            // step 5: within = 160+lane
            {
                let within = 160 + lane;
                let group = within / 16;
                let scale = f32::from_bits(packed[(base + 1 + group) as usize]);
                let q_word = packed[(base + 17 + within / 4) as usize];
                let q_full = (q_word >> ((within % 4) * 8)) & 0xFF;
                let w = d * scale * (q_full as f32 - 32.0);
                acc += w * x[(blk * 256 + within) as usize];
            }
            // step 6: within = 192+lane
            {
                let within = 192 + lane;
                let group = within / 16;
                let scale = f32::from_bits(packed[(base + 1 + group) as usize]);
                let q_word = packed[(base + 17 + within / 4) as usize];
                let q_full = (q_word >> ((within % 4) * 8)) & 0xFF;
                let w = d * scale * (q_full as f32 - 32.0);
                acc += w * x[(blk * 256 + within) as usize];
            }
            // step 7: within = 224+lane
            {
                let within = 224 + lane;
                let group = within / 16;
                let scale = f32::from_bits(packed[(base + 1 + group) as usize]);
                let q_word = packed[(base + 17 + within / 4) as usize];
                let q_full = (q_word >> ((within % 4) * 8)) & 0xFF;
                let w = d * scale * (q_full as f32 - 32.0);
                acc += w * x[(blk * 256 + within) as usize];
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

    // Build packed weights:
    //   d = 1.0, all group scales = 1.0, all q_full = 32 (→ weight = 0.0)
    // Expected output: 0.0 for all neurons.
    let packed_u32s = n * k_blocks * 81;
    let mut packed_host = vec![0u32; packed_u32s];
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 81;
            packed_host[base] = 1.0_f32.to_bits();
            for g in 0..16usize {
                packed_host[base + 1 + g] = 1.0_f32.to_bits();
            }
            for w in 0..64usize {
                packed_host[base + 17 + w] = 0x20202020_u32;
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

    // ── Test 2: d=1.0, scale=2.0, q_full=33 → weight=2.0, sum=512.0 ──────────
    let mut packed_host2 = vec![0u32; packed_u32s];
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 81;
            packed_host2[base] = 1.0_f32.to_bits();
            for g in 0..16usize {
                packed_host2[base + 1 + g] = 2.0_f32.to_bits();
            }
            for w in 0..64usize {
                packed_host2[base + 17 + w] = 0x21212121_u32;
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
