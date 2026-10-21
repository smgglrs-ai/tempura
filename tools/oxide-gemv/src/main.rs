/*
 * oxide-gemv: Hand-optimized Q2_0 decode GEMV kernel compiled with rustc-codegen-cuda.
 *
 * Build:  cargo oxide build --manifest-path tools/oxide-gemv/Cargo.toml
 * The compiled binary embeds PTX in the `.oxart` ELF section.
 * The tempura-kernels build.rs extracts that PTX and writes it to OUT_DIR.
 *
 * Q2_0 packed format (9 u32 per block, 128 elements per block):
 *   u32[0]    = f32 bits of block scale (f16 -> f32 at upload time)
 *   u32[1..9] = 8 u32 words of 2-bit weights (16 elements per word)
 *   dequant:  q_raw = ((word >> (bit_pair*2)) & 3)
 *             weight = (q_raw as i32 - 1) as f32 * scale
 *             values: {-1, 0, 1, 2} x scale
 *
 * Optimizations over the scalar baseline:
 *   1. Shared memory staging for x: all WARPS_PER_BLOCK warps in a block
 *      compute different output neurons but read the SAME activation x.
 *      Staging x once into shared memory (K global reads total per block,
 *      vs K x WARPS_PER_BLOCK without smem) reduces global memory pressure.
 *   2. Float4-vectorized global loads: x is staged via F32x4 128-bit wide
 *      loads (LDG.E.128), coalescing 4 f32 per transaction per thread.
 *      k = k_blocks * 128 is always divisible by 4, so no tail handling needed.
 *   3. Two-output-per-warp: each 32-lane warp computes two adjacent output
 *      neurons. Lanes 0-15 handle n_out_0, lanes 16-31 handle n_out_1.
 *      Both halves share the same smem-staged x[], doubling arithmetic
 *      intensity for the same memory bandwidth. Grid halves to
 *      ceil(N / (2 * WARPS_PER_BLOCK)) blocks.
 */

use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, thread, warp};
use cuda_device::vector::F32x4;
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_core::simt::LaunchConfig;

const WARPS_PER_BLOCK: u32 = 4;
const BLOCK_DIM: u32 = WARPS_PER_BLOCK * 32; // 128 threads per block

#[cuda_module]
mod kernels {
    use super::*;

    /// Q2_0 GEMV for M=1 decode: out[N] = dequant(packed[N, K_BLOCKS, 9]) x x[K].
    ///
    /// Grid:  (ceil(N / (2 * WARPS_PER_BLOCK)),) blocks -- each warp owns TWO output neurons.
    /// Block: (WARPS_PER_BLOCK * 32,) threads.
    ///
    /// Dynamic shared memory: k * sizeof(f32) bytes per block.
    /// Set shared_mem_bytes = k_blocks * 128 * 4 in LaunchConfig.
    ///
    /// Parameters:
    ///   out    -- output f32 slice [N] (DisjointSlice -- aliasing-free)
    ///   x      -- activation f32 slice [K = K_BLOCKS * 128]
    ///   packed -- Q2_0 packed u32 slice [N * K_BLOCKS * 9]
    ///   n      -- number of output neurons
    ///   k_blocks -- number of Q2_0 blocks per output row (K = k_blocks * 128)
    /// Q2_0 GEMV for M=1 decode: two-output-per-warp variant.
    ///
    /// Each 32-lane warp computes TWO adjacent output neurons simultaneously:
    ///   lanes  0-15 -> n_out_0 = warp_slot * 2
    ///   lanes 16-31 -> n_out_1 = warp_slot * 2 + 1
    ///
    /// Both halves share the same smem-staged x[] read (Phase 1 unchanged),
    /// then each half iterates independently over all K blocks with a 16-lane
    /// inner loop (bit_pair = my_lane, word_idx 0..7), followed by a masked
    /// 16-lane butterfly reduce.  Arithmetic intensity doubles vs. single-output.
    ///
    /// Grid:  (ceil(N / (2 * WARPS_PER_BLOCK)),) blocks.
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
        let k = k_blocks * 128;

        // Phase 1: All BLOCK_DIM threads cooperatively stage x from global
        // memory into shared memory using 128-bit vectorized loads (F32x4).
        //
        // k is always divisible by 4 (k = k_blocks * 128, 128/4 = 32).
        // Each thread loads k / (4 * BLOCK_DIM) = k / 512 F32x4 vectors.
        // For k=2048: each thread loads 4 vectors (16 f32 elements).
        //
        // Using raw *const F32x4 and *mut F32x4 pointers ensures the load
        // and store are NOT decomposed into lanes before use, which is the
        // condition for the compiler to emit LDG.E.128 + STS.E.128 (128-bit
        // wide transactions) rather than four scalar accesses.
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        unsafe {
            let x_v4    = x.as_ptr() as *const F32x4;
            let smem_v4 = smem       as *mut   F32x4;
            let num_v4  = k / 4;
            let mut vi  = tid;
            while vi < num_v4 {
                *smem_v4.add(vi as usize) = *x_v4.add(vi as usize);
                vi += BLOCK_DIM;
            }
        }

        // Barrier: all threads must finish writing smem before any warp reads.
        thread::sync_threads();

        // Two-output-per-warp split.
        let my_half = lane / 16;  // 0 = lower half -> n_out_0, 1 = upper half -> n_out_1
        let my_lane = lane % 16;  // 0-15 within half (= bit_pair position within each word)

        let n_out_0 = (bid * WARPS_PER_BLOCK + warp_id) * 2;
        let n_out_1 = n_out_0 + 1;

        // Threads beyond the last warp slot participated in smem staging
        // but have no output to produce.
        if n_out_0 >= n {
            return;
        }

        let my_n_out = if my_half == 0 { n_out_0 } else { n_out_1 };
        // Masks and base-lane for masked shuffles within each 16-lane half.
        let half_mask      = if my_half == 0 { 0x0000_FFFF_u32 } else { 0xFFFF_0000_u32 };
        let half_base_lane = if my_half == 0 { 0_u32 } else { 16_u32 };

        // Phase 2: Each half-warp accumulates one output neuron.
        // Block-at-a-time loop; 16 lanes cover all 128 elements per block:
        //   my_lane selects bit_pair (0-15), word_idx iterates 0..7 for the
        //   8 packed u32 words -> 8 elements per lane per block.
        let mut acc = 0.0_f32;
        let mut blk = 0u32;
        while blk < k_blocks {
            let base = (my_n_out * k_blocks + blk) * 9;

            // Scale broadcast: my_lane==0 of each half loads, all half-lanes receive.
            let scale_raw = if my_lane == 0 { packed[base as usize] } else { 0 };
            let scale = f32::from_bits(warp::shuffle_sync(half_mask, scale_raw, half_base_lane));

            // Inner loop over 8 packed words; each lane extracts bit_pair = my_lane.
            let mut word_idx = 0u32;
            while word_idx < 8 {
                let w = packed[(base + 1 + word_idx) as usize];
                let q_raw = (w >> (my_lane * 2)) & 3;
                let q_f = (q_raw as i32 - 1) as f32;
                let elem = blk * 128 + word_idx * 16 + my_lane;
                let xv = unsafe { *smem.add(elem as usize) };
                acc += q_f * scale * xv;
                word_idx += 1;
            }
            blk += 1;
        }

        // Masked 16-lane butterfly reduce within each half.
        acc += warp::shuffle_xor_f32_sync(half_mask, acc, 8);
        acc += warp::shuffle_xor_f32_sync(half_mask, acc, 4);
        acc += warp::shuffle_xor_f32_sync(half_mask, acc, 2);
        acc += warp::shuffle_xor_f32_sync(half_mask, acc, 1);

        // my_lane==0 of each half writes; guard n_out_1 for odd N.
        if my_lane == 0 && my_n_out < n {
            unsafe {
                *out.get_unchecked_mut(my_n_out as usize) = acc;
            }
        }
    }
}

// -- Host-side correctness test ------------------------------------------------

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
    // q=1 -> bit pattern 0b01 -> word = 0x55555555 for all 16 elements.
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

    // Grid: ceil(N / (2 * WARPS_PER_BLOCK)) blocks (two outputs per warp);
    // block: WARPS_PER_BLOCK*32 threads.
    // Dynamic shared memory: k * 4 bytes to stage the activation tensor.
    let wpb = WARPS_PER_BLOCK as usize;
    let grid_x = (n + 2 * wpb - 1) / (2 * wpb);
    let shared_mem_bytes = (k * std::mem::size_of::<f32>()) as u32;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x as u32, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes,
    };

    // Test 1: q=1 -> dequant=0 -> all outputs should be 0.
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
        println!("oxide-gemv test 1 (q=1->0): PASSED (max_err={:.2e})", max_err);
    } else {
        eprintln!("oxide-gemv test 1 FAILED (max_err={:.4})", max_err);
        std::process::exit(1);
    }

    // Test 2: q=2 -> dequant = scale = 1.0; sum = k.
    // q=2 -> bit pattern 0b10 -> word = 0xAAAAAAAA.
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
            "oxide-gemv test 2 (q=2->1.0): PASSED (expected={}, max_err={:.2e})",
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
