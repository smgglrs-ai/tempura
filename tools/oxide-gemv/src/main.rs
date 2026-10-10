/*
 * oxide-gemv: Hand-optimized Q2_0 decode GEMV kernel compiled with rustc-codegen-cuda.
 *
 * Build:  cargo oxide build --manifest-path tools/oxide-gemv/Cargo.toml
 * The compiled binary embeds PTX in the `.oxart` ELF section.
 * The tempura-kernels build.rs extracts that PTX and writes it to OUT_DIR.
 *
 * Q2_0 packed format (9 u32 per block, 128 elements per block):
 *   u32[0]    = f32 bits of block scale (f16 → f32 at upload time)
 *   u32[1..9] = 8 u32 words of 2-bit weights (16 elements per word)
 *   dequant:  q_raw = ((word >> (bit_pair*2)) & 3)
 *             weight = (q_raw as i32 - 1) as f32 * scale
 *             values: {-1, 0, 1, 2} × scale
 */

use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_core::simt::LaunchConfig;

const WARPS_PER_BLOCK: u32 = 4;

#[cuda_module]
mod kernels {
    use super::*;

    /// Q2_0 GEMV for M=1 decode: out[N] = dequant(packed[N, K_BLOCKS, 9]) × x[K].
    ///
    /// Grid:  (ceil(N / WARPS_PER_BLOCK),) blocks — each warp owns one output neuron.
    /// Block: (WARPS_PER_BLOCK * 32,) threads.
    ///
    /// Parameters:
    ///   out    — output f32 slice [N] (DisjointSlice — aliasing-free)
    ///   x      — activation f32 slice [K = K_BLOCKS * 128]
    ///   packed — Q2_0 packed u32 slice [N * K_BLOCKS * 9]
    ///   n      — number of output neurons
    ///   k_blocks — number of Q2_0 blocks per output row (K = k_blocks * 128)
    #[kernel]
    pub fn q2_0_gemv_decode(
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
        let k = k_blocks * 128;

        // Stride-32 loop: lane handles elem = lane, lane+32, lane+64, …
        let mut elem = lane;
        while elem < k {
            let blk      = elem / 128;
            let within   = elem % 128;
            let word_idx = within / 16;
            let bit_pair = within % 16;

            // Base of this block's 9-word packed region for output n_out.
            let base = (n_out * k_blocks + blk) * 9;

            // Scale: packed[base] stored as raw f32 bits.
            let scale = f32::from_bits(packed[base as usize]);

            // 2-bit weight: extract from the packed word.
            let w = packed[(base + 1 + word_idx) as usize];
            let q_raw = (w >> (bit_pair * 2)) & 3;

            // Dequantize: q ∈ {0,1,2,3} → weight ∈ {-1,0,1,2} × scale.
            let q_f = (q_raw as i32 - 1) as f32;

            let xv = x[elem as usize];
            acc += q_f * scale * xv;

            elem += 32;
        }

        // Butterfly warp-reduce: all 32 lanes contribute; lane 0 gets the sum.
        acc = warp::reduce_sum_f32(acc);
        if lane == 0 {
            // SAFETY: n_out < n (checked above); each warp has a unique n_out.
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

    // Small problem: N=64 outputs, K_BLOCKS=2 (K=256).
    let n: usize = 64;
    let k_blocks: usize = 2;
    let k = k_blocks * 128;

    // Activation: x[i] = 1.0.
    let x_host: Vec<f32> = vec![1.0_f32; k];

    // Packed weights: scale=1.0, all q=1 (q_raw=1, dequant=0).
    // q=1 → bit pattern 0b01 → word = 0x55555555 for all 16 elements.
    let packed_u32s = n * k_blocks * 9;
    let mut packed_host = vec![0u32; packed_u32s];
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 9;
            packed_host[base] = 1.0_f32.to_bits();
            for w in 0..8usize {
                packed_host[base + 1 + w] = 0x55555555_u32;
            }
        }
    }

    let x_dev     = DeviceBuffer::from_host(&stream, &x_host).expect("upload x");
    let mut p_dev = DeviceBuffer::from_host(&stream, &packed_host).expect("upload packed");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc out");

    // Load the module (PTX embedded via #[cuda_module]).
    let module = kernels::load(&ctx).expect("load oxide module");

    // Grid: ceil(N / WARPS_PER_BLOCK) blocks; block: WARPS_PER_BLOCK*32 threads.
    let wpb = WARPS_PER_BLOCK as usize;
    let grid_x = (n + wpb - 1) / wpb;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x as u32, 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Test 1: q=1 → dequant=0 → all outputs should be 0.
    unsafe {
        module.q2_0_gemv_decode(
            stream.as_ref(),
            cfg,
            &mut out_dev,
            &x_dev,
            &p_dev,
            n as u32,
            k_blocks as u32,
        )
    }
    .expect("kernel launch");

    let result = out_dev.to_host_vec(&stream).expect("download");
    let max_err = result.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_err < 1e-4 {
        println!("oxide-gemv test 1 (q=1→0): PASSED (max_err={:.2e})", max_err);
    } else {
        eprintln!("oxide-gemv test 1 FAILED (max_err={:.4})", max_err);
        std::process::exit(1);
    }

    // Test 2: q=2 → dequant = scale = 1.0; sum = k.
    // q=2 → bit pattern 0b10 → word = 0xAAAAAAAA.
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 9;
            packed_host[base] = 1.0_f32.to_bits();
            for w in 0..8usize {
                packed_host[base + 1 + w] = 0xAAAAAAAA_u32;
            }
        }
    }
    p_dev.copy_from_host(&stream, &packed_host).expect("re-upload");
    // Re-zero output: copy from a host zeroed buffer.
    let zeros = vec![0.0_f32; n];
    out_dev.copy_from_host(&stream, &zeros).expect("zero out");

    unsafe {
        module.q2_0_gemv_decode(
            stream.as_ref(),
            cfg,
            &mut out_dev,
            &x_dev,
            &p_dev,
            n as u32,
            k_blocks as u32,
        )
    }
    .expect("kernel launch 2");

    let result2 = out_dev.to_host_vec(&stream).expect("download 2");
    let expected = k as f32;
    let max_err2 = result2.iter().map(|v| (v - expected).abs()).fold(0.0_f32, f32::max);
    if max_err2 < 1e-2 {
        println!(
            "oxide-gemv test 2 (q=2→1.0): PASSED (expected={}, max_err={:.2e})",
            expected, max_err2
        );
    } else {
        eprintln!(
            "oxide-gemv test 2 FAILED (expected={}, max_err={:.4})",
            expected, max_err2
        );
        std::process::exit(1);
    }

    println!("oxide-gemv: all tests PASSED");
}
