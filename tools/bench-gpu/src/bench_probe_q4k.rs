// bench-probe-q4k -- standalone repro for the cuTile Q4K GEMV fallback
// crash (DriverError 719 "unspecified launch failure" under cuda-oxide
// a0a4dc78a, found 2026-11-12; blocks bench-real-model on main -- the
// engine constructor warmup is the only caller today).
//
// Drives CutileQ4Gemv::forward_d2d on SYNTHETIC packed Q4_K weights at
// several (n, k_blocks) shapes and checks the output against a CPU
// emulation of the kernel's exact dequant math.  Run directly to see the
// crash; run under compute-sanitizer to localize it:
//   compute-sanitizer --tool memcheck ./target/release/bench-probe-q4k
//
// Run: scripts/gpu-remote.sh bench probe-q4k

use std::sync::Arc;
use cuda_core::Stream;
use cutile::api;
use cutile::prelude::{DeviceOp, ToHostVec};
use cutile::tensor::Tensor;
use tempura_kernels::cutile_quantized_gemm::{CutileQ4Gemv, Q4K_PACKED_U32S};

/// CPU emulation of cutile_q4k_gemv's math on the same packed layout
/// (42 u32 per block: [0]=d f32 bits, [1]=dmin bits, [2..6]=sc pairs,
/// [6..10]=m pairs, [10..42]=128 qs bytes).
fn cpu_q4k_gemv(packed: &[u32], x: &[f32], n: usize, k_blocks: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for row in 0..n {
        let mut acc = 0.0f32;
        for blk in 0..k_blocks {
            let base = row * k_blocks * Q4K_PACKED_U32S + blk * Q4K_PACKED_U32S;
            let d = f32::from_bits(packed[base]);
            let dmin = f32::from_bits(packed[base + 1]);
            for j in 0..4usize {
                let sc_pair = packed[base + 2 + j];
                let m_pair = packed[base + 6 + j];
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo = (m_pair & 0xFF) as f32;
                let m_hi = ((m_pair >> 8) & 0xFF) as f32;
                let d1 = d * sc_lo;
                let m1 = dmin * m_lo;
                let d2 = d * sc_hi;
                let m2 = dmin * m_hi;
                for l in 0..32usize {
                    let qs_word_idx = (j * 32 + l) / 4;
                    let byte_shift = ((j * 32 + l) % 4) * 8;
                    let qs_word = packed[base + 10 + qs_word_idx];
                    let byte_val = (qs_word >> byte_shift) & 0xFF;
                    let lo_n = (byte_val & 0xF) as f32;
                    let hi_n = (byte_val >> 4) as f32;
                    let w_lo = d1 * lo_n - m1;
                    let w_hi = d2 * hi_n - m2;
                    let x_lo = x[blk * 256 + j * 64 + l];
                    let x_hi = x[blk * 256 + j * 64 + 32 + l];
                    acc += w_lo * x_lo;
                    acc += w_hi * x_hi;
                }
            }
        }
        out[row] = acc;
    }
    out
}

fn main() {
    let device = Arc::new(cuda_core::Device::new(0).expect("device"));
    let stream: Arc<Stream> = device
        .primary_context()
        .expect("ctx")
        .new_stream()
        .expect("stream");
    let gemv = CutileQ4Gemv::from_stream(stream.clone());

    // Deterministic LCG (no RNG dependency).
    let mut s = 0xC0FFEEu32;
    let mut nxt = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s
    };

    for (n, k_blocks) in [(8usize, 16usize), (64, 16), (128, 56), (4096, 16)] {
        let k = k_blocks * 256;
        let x: Vec<f32> = (0..k).map(|_| (nxt() & 0xFFFF) as f32 / 65535.0 - 0.5).collect();
        // Random-ish packed blocks: d in [0.001, 0.01], dmin in [0, 0.002],
        // sc/m bytes nonzero, qs bytes arbitrary.
        let total = n * k_blocks * Q4K_PACKED_U32S;
        let mut packed = vec![0u32; total];
        for row in 0..n {
            for blk in 0..k_blocks {
                let base = row * k_blocks * Q4K_PACKED_U32S + blk * Q4K_PACKED_U32S;
                packed[base] = (0.001 + (nxt() & 0xFF) as f32 / 255.0 * 0.009).to_bits();
                packed[base + 1] = ((nxt() & 0xFF) as f32 / 255.0 * 0.002).to_bits();
                for j in 0..4 {
                    packed[base + 2 + j] = (128 + (nxt() & 0x3F)) | ((64 + (nxt() & 0x3F)) << 8);
                    packed[base + 6 + j] = (64 + (nxt() & 0x3F)) | ((32 + (nxt() & 0x3F)) << 8);
                }
                for w in 0..32 {
                    packed[base + 10 + w] = nxt();
                }
            }
        }
        let cpu = cpu_q4k_gemv(&packed, &x, n, k_blocks);

        let x_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x))
            .sync_on(&stream)
            .expect("x upload")
            .into();
        let p_dev: Arc<Tensor<u32>> = api::copy_host_vec_to_device(&Arc::new(packed))
            .sync_on(&stream)
            .expect("packed upload")
            .into();

        println!("n={n} k_blocks={k_blocks}: launching...");
        let out = gemv
            .forward_d2d(&x_dev, &p_dev, n, k_blocks)
            .unwrap_or_else(|e| panic!("n={n} kb={k_blocks}: forward_d2d failed: {e}"));
        let gpu = out
            .clone()
            .to_host_vec()
            .sync_on(&stream)
            .expect("download");

        let mut max_abs = 0.0f32;
        let mut bad = 0usize;
        for (a, b) in gpu.iter().zip(&cpu) {
            let d = (a - b).abs();
            if d > 1e-4 {
                bad += 1;
            }
            max_abs = max_abs.max(d);
        }
        println!(
            "  n={n} k_blocks={k_blocks}: max_abs={max_abs:.6} bad={bad}/{n} -> {}",
            if bad == 0 { "PASS" } else { "FAIL" }
        );
    }
    println!("all done");
}
