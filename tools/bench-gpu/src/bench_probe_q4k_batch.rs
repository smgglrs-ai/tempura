// bench-probe-q4k-batch -- correctness cross-validation of the sm_120 Q4K
// batch GEMM kernels (q4k_mma_batch_sm120, q4k_streamk_batch_sm120, and the
// kpar8 batch) against the per-row kpar GEMV
// (q4k_gemv_sm120_dp4a_kpar8_dual_block), on RANDOM native-GGUF Q4K blocks.
//
// Why random data: uniform/simple data cannot catch scale-extraction or
// fragment-mapping bugs (the mxf4-gemv lesson) -- real weights exercise
// every bit of the 6-bit cross-byte scale packing. The kpar per-row GEMV
// is the ground truth here: it is the kernel the single-seq decode path
// ships with, validated end-to-end (bench-quality teacher-forced PASS vs
// the CPU reference).
//
// Acceptance:
//   - kpar8_batch: BIT-IDENTICAL to the per-row reference (it replicates
//     the kpar8 warp structure verbatim; this gates the no-graph path).
//   - mma_batch / streamk_batch: <= 1e-5 max rel error. The mma.sync
//     products are exact int8; the only permitted divergence from the
//     per-row reference is the f32 SUM ORDER (the mma K-parallel warp
//     partials + smem reduction vs the kpar butterfly), which is ~1e-6
//     rel on random data. Anything above 1e-5 is a layout bug, not sum
//     order (the 2026-11-16 retraction measured 12-31% on the broken
//     kernels; fixed 2026-11-16f: byte group mw>>1, hi-select mw&1, raw
//     nibble / raw 6-bit scales).
//
// Run: scripts/gpu-remote.sh bench probe-q4k-batch

use std::sync::Arc;
use cuda_core::Stream;
use cutile::api;
use cutile::prelude::{DeviceOp, ToHostVec};
use cutile::tensor::Tensor;
use tempura_kernels::oxide_q4k_gemv_host::OxideQ4kGemv;

/// Deterministic xorshift RNG (no external crates).
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x as u32
    }
    fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next_u32() as f64 / u32::MAX as f64) as f32 * (hi - lo)
    }
}

/// Snap to exactly-representable f16 values (m * 2^-11, integer m up to
/// 2047) so the GPU's cvt.f32.f16 round-trip is exact.
fn f16_snap(v: f32) -> u16 {
    let scaled = (v * 2048.0).round() as i32;
    let sign = if scaled < 0 { 1u16 << 15 } else { 0 };
    sign | scaled.unsigned_abs() as u16
}

/// Build one native-GGUF Q4K block (36 u32) from random parameters.
///
/// Layout: u32[0] = d | dmin<<16 (two f16), u32[1..4] = 12 scale bytes
/// (8x 6-bit sc + 8x 6-bit mv, llama.cpp k-quant cross-byte packing --
/// get_scale_min_k4), u32[4..36] = 128 qs bytes (lo nibble = element b,
/// hi nibble = element b+128).
fn build_block(rng: &mut Rng, out: &mut [u32]) {
    assert_eq!(out.len(), 36);
    let d = f16_snap(rng.f32_in(0.0005, 0.05));
    let dmin = f16_snap(rng.f32_in(0.0, 0.01));
    out[0] = d as u32 | ((dmin as u32) << 16);

    let mut sc = [0u8; 8];
    let mut mv = [0u8; 8];
    for i in 0..8 {
        sc[i] = (rng.next_u32() % 64) as u8;
        mv[i] = (rng.next_u32() % 64) as u8;
    }
    // Invert get_scale_min_k4 (q = the 12 scale bytes):
    //   j < 4:  sc[j] = q[j] & 63;           mv[j] = q[j+4] & 63
    //   j >= 4: sc[j] = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
    //           mv[j] = (q[j+4] >> 4) | ((q[j]   >> 6) << 4)
    let mut q = [0u8; 12];
    for j in 0..4 {
        q[j] = (sc[j] & 0x3F) | ((sc[4 + j] & 0x30) << 2);
        q[4 + j] = (mv[j] & 0x3F) | ((mv[4 + j] & 0x30) << 2);
        q[8 + j] = (sc[4 + j] & 0xF) | ((mv[4 + j] & 0xF) << 4);
    }
    out[1] = u32::from_le_bytes([q[0], q[1], q[2], q[3]]);
    out[2] = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
    out[3] = u32::from_le_bytes([q[8], q[9], q[10], q[11]]);

    // qs nibbles: 256 per block, 128 bytes. Element i (i<128): lo nibble
    // of qs byte i; element i+128: hi nibble of qs byte i.
    for w in 0..32usize {
        let mut word = 0u32;
        for j in 0..4usize {
            word |= ((rng.next_u32() & 0xFF) as u32) << (j * 8);
        }
        out[4 + w] = word;
    }
}

fn upload<T: cutile::DType + Default + Clone>(
    host: &[T],
    stream: &Arc<Stream>,
) -> Arc<Tensor<T>> {
    api::copy_host_vec_to_device(&Arc::new(host.to_vec()))
        .sync_on(stream)
        .expect("host->device upload")
        .into()
}

fn main() {
    let shapes: &[(usize, usize, &str)] = &[
        (64, 1, "tiny [n=64, k=256]"),
        (64, 2, "tiny2 [n=64, k=512]"),
        (4096, 16, "q_proj [n=4096, k=4096]"),
        (1024, 16, "k/v_proj [n=1024, k=4096]"),
        (14336, 16, "gate/up [n=14336, k=4096]"),
        (4096, 56, "down [n=4096, k=14336]"),
    ];
    let device = Arc::new(cuda_core::Device::new(0).expect("device"));
    let stream: Arc<Stream> = device
        .primary_context()
        .expect("ctx")
        .new_stream()
        .expect("stream");
    let oxide = OxideQ4kGemv::try_load(stream.clone()).expect("oxide q4k load");
    println!(
        "probe-q4k-batch: kpar_batch={} mma_batch={} streamk_batch={}",
        oxide.has_sm120_kpar_batch(),
        oxide.has_mma_batch(),
        oxide.has_streamk_batch(),
    );
    assert!(oxide.has_mma_batch(), "mma_batch PTX not loaded");
    assert!(oxide.has_streamk_batch(), "streamk_batch PTX not loaded");
    assert!(oxide.uses_raw_gguf, "expected native-GGUF upload format (sm120 active)");
    // The kpar per-row ground truth needs the pre-allocated KV-stream Q8_1
    // scratch (see submit_sm120_kpar_kv_into_slot).
    oxide
        .pre_allocate_q8_scratch(14336, 14336)
        .expect("q8 scratch pre-alloc");

    let mut any_fail = false;
    for &(n, k_blocks, label) in shapes {
        let k = k_blocks * 256;
        let mut rng = Rng(0x5EED_0000 + n as u64 + k_blocks as u64);

        // Random native-GGUF weights [n * k_blocks * 36].
        let mut packed_host = vec![0u32; n * k_blocks * 36];
        for row in 0..n {
            for blk in 0..k_blocks {
                let base = (row * k_blocks + blk) * 36;
                build_block(&mut rng, &mut packed_host[base..base + 36]);
            }
        }
        let packed = upload(&packed_host, &stream);

        for &m in &[1usize, 2, 4, 6, 16, 64, 128] {
            // Random activations [m * k].
            let mut x_host = vec![0f32; m * k];
            for v in x_host.iter_mut() {
                *v = rng.f32_in(-1.0, 1.0);
            }
            let x = upload(&x_host, &stream);

            // Ground truth: per-row sm120 kpar GEMV (the validated decode
            // kernel; needs its own out tensor per row).
            let mut ref_rows: Vec<Vec<f32>> = Vec::with_capacity(m);
            for r in 0..m {
                let row = upload(&x_host[r * k..(r + 1) * k], &stream);
                let out = Arc::new(
                    api::zeros::<f32>(&[n]).sync_on(&stream).expect("kpar out alloc"),
                );
                let res = unsafe {
                    oxide.submit_sm120_kpar_kv_into_slot(
                        &row,
                        &packed,
                        Arc::try_unwrap(out).expect("sole out ref"),
                        n,
                        k_blocks,
                        stream.cu_stream(),
                    )
                };
                let out = res.expect("kpar gemv");
                ref_rows.push(out.to_host_vec().sync_on(&stream).expect("kpar download"));
            }

            // Scratch for the batch kernels.
            let x_qs = Arc::new(
                api::zeros::<u32>(&[m * k / 4]).sync_on(&stream).expect("qs alloc"),
            );
            let x_ds = Arc::new(
                api::zeros::<u32>(&[m * k.div_ceil(32)]).sync_on(&stream).expect("ds alloc"),
            );

            // 1) The batch kpar8 kernel: MUST be bit-identical to the
            //    per-row reference (it replicates the kpar8 warp structure
            //    verbatim). This is the gate for the no-graph batch path.
            {
                let out = Arc::new(
                    api::zeros::<f32>(&[m * n]).sync_on(&stream).expect("out alloc"),
                );
                let mut row = 0usize;
                while row < m {
                    let rows = (m - row).min(16);
                    oxide
                        .forward_sm120_kpar_batch_d2d_rows(
                            &x, row, &packed, &out, row, rows, n, k_blocks, &x_qs, &x_ds,
                        )
                        .expect("kpar8 batch launch");
                    row += rows;
                }
                let got = out.to_host_vec().sync_on(&stream).expect("out download");
                let mut exact = true;
                for r in 0..m {
                    for c in 0..n {
                        if got[r * n + c].to_bits() != ref_rows[r][c].to_bits() {
                            exact = false;
                        }
                    }
                }
                if !exact {
                    any_fail = true;
                }
                println!(
                    "  {label} m={m} kpar8_batch: {} (bit-identical to per-row kpar8)",
                    if exact { "OK" } else { "MISMATCH" }
                );
            }

            // 2) mma_batch / streamk: tensor-core paths. The mma.sync int8
            //    products are exact; only the f32 sum ORDER may differ from
            //    the per-row reference (K-parallel warp partials vs the kpar
            //    butterfly) -> <= 1e-5 rel on random data. Anything more is
            //    a layout bug (the 2026-11-16 retraction was 12-31%).
            for (kernel, is_mma) in [("mma", true), ("streamk", false)] {
                let out = Arc::new(
                    api::zeros::<f32>(&[m * n]).sync_on(&stream).expect("out alloc"),
                );
                // Both kernels cap at 16 rows per launch: larger m runs as
                // chunked row windows (the engine's prefill chunking).
                let mut launch_err = None;
                let mut row = 0usize;
                while row < m {
                    let rows = (m - row).min(16);
                    let res = if is_mma {
                        oxide.forward_mma_batch_d2d_rows(
                            &x, row, &packed, &out, row, rows, n, k_blocks, &x_qs, &x_ds,
                        )
                    } else {
                        oxide.forward_streamk_batch_d2d_rows(
                            &x, row, &packed, &out, row, rows, n, k_blocks, &x_qs, &x_ds,
                        )
                    };
                    if let Err(e) = res {
                        launch_err = Some(e);
                        break;
                    }
                    row += rows;
                }
                if let Some(e) = launch_err {
                    // Not loaded is a hard failure too: the probe must cover
                    // every kernel the PTX ships (a silently-unloaded kernel
                    // is exactly how the 2026-11-16 retraction slipped).
                    any_fail = true;
                    println!("  {label} m={m} {kernel}: FAIL (launch error: {e})");
                    continue;
                }
                let got = out.to_host_vec().sync_on(&stream).expect("out download");
                let mut max_rel = 0f64;
                let mut bad_rows = 0;
                for r in 0..m {
                    let mut row_max = 0f64;
                    for c in 0..n {
                        let a = got[r * n + c];
                        let b = ref_rows[r][c];
                        let denom = b.abs().max(1.0);
                        let rel = ((a - b).abs() / denom) as f64;
                        row_max = row_max.max(rel);
                    }
                    if row_max > 1e-5 {
                        bad_rows += 1;
                    }
                    max_rel = max_rel.max(row_max);
                }
                if bad_rows > 0 {
                    any_fail = true;
                }
                println!(
                    "  {label} m={m} {kernel}: {} (max rel err {max_rel:.3e}, bad rows {bad_rows}/{m}, gate 1e-5)",
                    if bad_rows == 0 { "OK" } else { "MISMATCH" }
                );
            }

            // 3) mma128_batch: the LARGE-m tensor-core kernel (m <= 128 in
            //    ONE launch, weights once). Gates:
            //    a) <= 1e-5 rel vs the per-row kpar reference (same class
            //       as mma_batch: sum-order-only divergence);
            //    b) BITWISE-equal to the chunked mma_batch output (same
            //       fragment mapping + accumulation order by construction).
            if oxide.has_mma128_batch() {
                let out = Arc::new(
                    api::zeros::<f32>(&[m * n]).sync_on(&stream).expect("out alloc"),
                );
                let res = oxide.forward_mma128_batch_d2d_rows(
                    &x, 0, &packed, &out, 0, m, n, k_blocks, &x_qs, &x_ds,
                );
                match res {
                    Err(e) => {
                        any_fail = true;
                        println!("  {label} m={m} mma128: FAIL (launch error: {e})");
                    }
                    Ok(()) => {
                        let got = out.to_host_vec().sync_on(&stream).expect("out download");
                        let mut max_rel = 0f64;
                        let mut bad_rows = 0;
                        for r in 0..m {
                            let mut row_max = 0f64;
                            for c in 0..n {
                                let a = got[r * n + c];
                                let b = ref_rows[r][c];
                                let denom = b.abs().max(1.0);
                                let rel = ((a - b).abs() / denom) as f64;
                                row_max = row_max.max(rel);
                            }
                            if row_max > 1e-5 {
                                bad_rows += 1;
                            }
                            max_rel = max_rel.max(row_max);
                        }
                        let rel_ok = bad_rows == 0;
                        if !rel_ok {
                            any_fail = true;
                        }
                        // (b) bitwise vs chunked mma_batch
                        let out16 = Arc::new(
                            api::zeros::<f32>(&[m * n]).sync_on(&stream).expect("out alloc"),
                        );
                        let mut row = 0usize;
                        while row < m {
                            let rows = (m - row).min(16);
                            oxide
                                .forward_mma_batch_d2d_rows(
                                    &x, row, &packed, &out16, row, rows, n, k_blocks, &x_qs, &x_ds,
                                )
                                .expect("mma16 chunk launch");
                            row += rows;
                        }
                        let got16 = out16.to_host_vec().sync_on(&stream).expect("out download");
                        let bitwise = got == got16;
                        if !bitwise {
                            any_fail = true;
                        }
                        println!(
                            "  {label} m={m} mma128: {} (max rel err {max_rel:.3e}, bad rows {bad_rows}/{m}, gate 1e-5; vs chunked mma16: {})",
                            if rel_ok { "OK" } else { "MISMATCH" },
                            if bitwise { "bit-identical" } else { "BITWISE MISMATCH" }
                        );
                    }
                }
            } else {
                any_fail = true;
                println!("  {label} m={m} mma128: FAIL (kernel not loaded)");
            }

            // 4) mma128w_batch: the WIDE-neuron mma128 (64 neurons/block,
            //    warp-owned). Gate: <= 1e-5 rel vs the per-row kpar
            //    reference (sum-order-only divergence -- sequential (kb,mw)
            //    accumulation, NOT bitwise vs mma16; see kernel docs).
            if n % 64 == 0 {
                if oxide.has_mma128w_batch() {
                    let out = Arc::new(
                        api::zeros::<f32>(&[m * n]).sync_on(&stream).expect("out alloc"),
                    );
                    let res = oxide.forward_mma128w_batch_d2d_rows(
                        &x, 0, &packed, &out, 0, m, n, k_blocks, &x_qs, &x_ds,
                    );
                    match res {
                        Err(e) => {
                            any_fail = true;
                            println!("  {label} m={m} mma128w: FAIL (launch error: {e})");
                        }
                        Ok(()) => {
                            let got = out.to_host_vec().sync_on(&stream).expect("out download");
                            let mut max_rel = 0f64;
                            let mut bad_rows = 0;
                            for r in 0..m {
                                let mut row_max = 0f64;
                                for c in 0..n {
                                    let a = got[r * n + c];
                                    let b = ref_rows[r][c];
                                    let denom = b.abs().max(1.0);
                                    let rel = ((a - b).abs() / denom) as f64;
                                    row_max = row_max.max(rel);
                                }
                                if row_max > 1e-5 {
                                    bad_rows += 1;
                                }
                                max_rel = max_rel.max(row_max);
                            }
                            if bad_rows > 0 {
                                any_fail = true;
                            }
                            println!(
                                "  {label} m={m} mma128w: {} (max rel err {max_rel:.3e}, bad rows {bad_rows}/{m}, gate 1e-5)",
                                if bad_rows == 0 { "OK" } else { "MISMATCH" }
                            );
                        }
                    }
                } else {
                    any_fail = true;
                    println!("  {label} m={m} mma128w: FAIL (kernel not loaded)");
                }
            }
        }
    }
    println!(
        "\n=== probe-q4k-batch: {} ===",
        if any_fail { "FAIL" } else { "PASS" }
    );
    if any_fail {
        std::process::exit(1);
    }
}
