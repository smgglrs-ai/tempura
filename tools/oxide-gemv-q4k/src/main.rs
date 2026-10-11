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
 *   u32[2+j]    = sc[2*j] | (sc[2*j+1]<<8) for j=0..4 (group scales, 8-bit slots)
 *   u32[6+j]    = m[2*j]  | (m[2*j+1]<<8)  for j=0..4 (group mins, 8-bit slots)
 *   u32[10..42] = 32 u32s of packed qs bytes (128 bytes = 256 nibbles, 4-bit each)
 *
 * Dequant for element `within` in a block:
 *   j       = within / 64                    — which j-pair (0..4)
 *   in_j    = within % 64
 *   is_hi   = in_j >= 32
 *   l       = in_j % 32
 *   sc_group = 2*j + is_hi  (0..8)
 *   sc = (packed[base + 2 + sc_group/2] >> ((sc_group%2)*8)) & 0xFF  as f32
 *   m  = (packed[base + 6 + sc_group/2] >> ((sc_group%2)*8)) & 0xFF  as f32
 *   nibble = lo nibble if !is_hi, hi nibble if is_hi  (from qs[j*32+l])
 *   weight = d * sc * nibble - dmin * m
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
    /// Parameters:
    ///   out      — output f32 slice [N]
    ///   x        — activation f32 slice [K = K_BLOCKS * 256]
    ///   packed   — Q4_K packed u32 slice [N * K_BLOCKS * 42]
    ///   n        — number of output neurons
    ///   k_blocks — number of Q4_K blocks per output row
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

        // Stride-32 loop: lane handles elem = lane, lane+32, lane+64, …
        let mut elem = lane;
        while elem < k {
            let blk    = elem / 256;
            let within = elem % 256;

            // Base of this block's 42-word packed region for output n_out.
            let base = (n_out * k_blocks + blk) * 42;

            // d and dmin stored as f32 bits in u32[0] and u32[1].
            let d    = f32::from_bits(packed[base as usize]);
            let dmin = f32::from_bits(packed[(base + 1) as usize]);

            // Decompose `within` into j, in_j, is_hi, l.
            let j     = within / 64;
            let in_j  = within % 64;
            let is_hi = in_j >= 32;
            let l     = in_j % 32;

            // qs byte index and location in packed u32[10..42].
            let qs_byte_idx = j * 32 + l;
            let qs_word_idx = qs_byte_idx / 4;
            let qs_byte_pos = qs_byte_idx % 4;
            let qs_word = packed[(base + 10 + qs_word_idx) as usize];
            let qs_byte = (qs_word >> (qs_byte_pos * 8)) & 0xFF;
            let nibble = if is_hi { qs_byte >> 4 } else { qs_byte & 0xF };

            // Scale group g = 2*j + is_hi (0..8); packed in pairs.
            let sc_group = 2 * j + is_hi as u32;
            let sc_pair_word = packed[(base + 2 + sc_group / 2) as usize];
            let m_pair_word  = packed[(base + 6 + sc_group / 2) as usize];
            let sc_shift = (sc_group % 2) * 8;
            let sc_u32 = (sc_pair_word >> sc_shift) & 0xFF;
            let m_u32  = (m_pair_word  >> sc_shift) & 0xFF;

            // Dequantize: w = d * sc * nibble - dmin * m
            let w = d * (sc_u32 as f32) * (nibble as f32) - dmin * (m_u32 as f32);

            acc += w * x[elem as usize];

            elem += 32;
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

    let x_dev      = DeviceBuffer::from_host(&stream, &x_host).expect("upload x");
    let mut p_dev  = DeviceBuffer::from_host(&stream, &packed_host).expect("upload packed");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc out");

    let module = kernels::load(&ctx).expect("load oxide q4k module");

    let wpb = WARPS_PER_BLOCK as usize;
    let grid_x = (n + wpb - 1) / wpb;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x as u32, 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Test 1: nibble=3, sc=1, d=1.0, dmin=0 → weight=3.0, sum=768.0
    unsafe {
        module.q4k_gemv_decode(
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
    let expected = 768.0_f32;
    let max_err = result.iter().map(|v| (v - expected).abs()).fold(0.0_f32, f32::max);
    if max_err < 1.0 {
        println!(
            "oxide-gemv-q4k test 1 (nibble=3,sc=1,d=1,dmin=0): PASSED (expected={}, max_err={:.2e})",
            expected, max_err
        );
    } else {
        eprintln!(
            "oxide-gemv-q4k test 1 FAILED (expected={}, max_err={:.4}, result[0]={})",
            expected, max_err, result[0]
        );
        std::process::exit(1);
    }

    // Test 2: d=2.0, sc=3, nibble=4, dmin=0.5, m=2
    // weight = 2.0 * 3.0 * 4.0 - 0.5 * 2.0 = 24.0 - 1.0 = 23.0
    // sum over 256 elements = 256 * 23.0 = 5888.0
    let mut packed_host2 = vec![0u32; packed_u32s];
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 42;
            packed_host2[base]     = 2.0_f32.to_bits();
            packed_host2[base + 1] = 0.5_f32.to_bits();
            // sc[g]=3, m[g]=2 → u32[2+j]=0x0303, u32[6+j]=0x0202
            for j in 0..4usize {
                packed_host2[base + 2 + j] = 0x00000303_u32;
                packed_host2[base + 6 + j] = 0x00000202_u32;
            }
            // nibble=4: lo=4, hi=4, packed byte = 0x44 → u32 = 0x44444444
            for w in 0..32usize {
                packed_host2[base + 10 + w] = 0x44444444_u32;
            }
        }
    }
    p_dev.copy_from_host(&stream, &packed_host2).expect("re-upload");
    let zeros = vec![0.0_f32; n];
    out_dev.copy_from_host(&stream, &zeros).expect("zero out");

    unsafe {
        module.q4k_gemv_decode(
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
    let expected2 = 5888.0_f32;
    let max_err2 = result2.iter().map(|v| (v - expected2).abs()).fold(0.0_f32, f32::max);
    if max_err2 < 1.0 {
        println!(
            "oxide-gemv-q4k test 2 (d=2,sc=3,nibble=4,dmin=0.5,m=2): PASSED (expected={}, max_err={:.2e})",
            expected2, max_err2
        );
    } else {
        eprintln!(
            "oxide-gemv-q4k test 2 FAILED (expected={}, max_err={:.4}, result[0]={})",
            expected2, max_err2, result2[0]
        );
        std::process::exit(1);
    }

    println!("oxide-gemv-q4k: all tests PASSED");
}
