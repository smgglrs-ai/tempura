// bench-probe-prefill-tc -- LARGE-m timing of the fixed sm_120 Q4K batch
// GEMM kernels (kpar8 batch vs mma_batch vs streamk_batch) at prefill
// shapes, for the TEMPURA_PREFILL_PRECISION=exact acceleration decision.
//
// Correctness of all three kernels is gated by bench-probe-q4k-batch
// (kpar8 bit-identical, mma/streamk <= 1e-5 rel = f32 sum-order noise).
// This bench answers a DIFFERENT question: at prefill m (chunked into the
// kernels' 16-row windows), which variant is fastest, and what does the
// Q4K projection stack cost per 128-token prefill step?
//
// Method: random native-GGUF Q4K blocks at the real Llama-3.1-8B Q4_K_M
// projection shapes; each timed pass runs the full chunk loop over m_total
// rows (quantize + GEMM per 16-row chunk, exactly what the engine's
// gemm_q4k_prefill_batch does); host-timed around stream.synchronize().
//
// Run: scripts/gpu-remote.sh bench probe-prefill-tc

use std::sync::Arc;
use std::time::Instant;

use cuda_core::Stream;
use cutile::api;
use cutile::prelude::DeviceOp;
use cutile::tensor::Tensor;
use tempura_kernels::oxide_q4k_gemv_host::OxideQ4kGemv;

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

fn f16_snap(v: f32) -> u16 {
    let scaled = (v * 2048.0).round() as i32;
    let sign = if scaled < 0 { 1u16 << 15 } else { 0 };
    sign | scaled.unsigned_abs() as u16
}

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
    let mut q = [0u8; 12];
    for j in 0..4 {
        q[j] = (sc[j] & 0x3F) | ((sc[4 + j] & 0x30) << 2);
        q[4 + j] = (mv[j] & 0x3F) | ((mv[4 + j] & 0x30) << 2);
        q[8 + j] = (sc[4 + j] & 0xF) | ((mv[4 + j] & 0xF) << 4);
    }
    out[1] = u32::from_le_bytes([q[0], q[1], q[2], q[3]]);
    out[2] = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
    out[3] = u32::from_le_bytes([q[8], q[9], q[10], q[11]]);
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

const CHUNK: usize = 16;

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Kpar8,
    Mma,
    Mma128,
    Mma128w,
    Mmq,
    Streamk,
}

fn main() {
    // Real Llama-3.1-8B Q4_K_M Q4K projection shapes: (n, k_blocks, label).
    let shapes: &[(usize, usize, &str)] = &[
        (4096, 16, "q/o_proj [n=4096, k=4096]"),
        (1024, 16, "k_proj   [n=1024, k=4096]"),
        (14336, 16, "gate/up  [n=14336, k=4096]"),
        (4096, 56, "down*    [n=4096, k=14336] (Q6K in Q4_K_M; shape included for scale reference)"),
    ];
    let m_totals: &[usize] = &[16, 64, 128, 256, 512];

    let device = Arc::new(cuda_core::Device::new(0).expect("device"));
    let stream: Arc<Stream> = device
        .primary_context()
        .expect("ctx")
        .new_stream()
        .expect("stream");
    let oxide = OxideQ4kGemv::try_load(stream.clone()).expect("oxide q4k load");
    println!(
        "probe-prefill-tc: kpar_batch={} mma_batch={} streamk_batch={}",
        oxide.has_sm120_kpar_batch(),
        oxide.has_mma_batch(),
        oxide.has_streamk_batch(),
    );
    assert!(oxide.has_sm120_kpar_batch(), "kpar8_batch PTX not loaded");
    assert!(oxide.has_mma_batch(), "mma_batch PTX not loaded");
    assert!(oxide.has_mma128_batch(), "mma128_batch PTX not loaded");
    assert!(oxide.has_streamk_batch(), "streamk_batch PTX not loaded");
    assert!(oxide.uses_raw_gguf, "expected native-GGUF upload format");

    let variants = [
        (Variant::Kpar8, "kpar8_batch"),
        (Variant::Mma, "mma_batch"),
        (Variant::Mma128, "mma128_batch"),
        (Variant::Mma128w, "mma128w_batch"),
        (Variant::Mmq, "mmq_batch"),
        (Variant::Streamk, "streamk_batch"),
    ];

    // results[shape][variant][m_idx] = ms per full m_total pass
    let mut results = vec![vec![vec![0f64; m_totals.len()]; variants.len()]; shapes.len()];

    for (si, &(n, k_blocks, label)) in shapes.iter().enumerate() {
        let k = k_blocks * 256;
        let mut rng = Rng(0x5EED_0000 + n as u64 + k_blocks as u64);
        let mut packed_host = vec![0u32; n * k_blocks * 36];
        for row in 0..n {
            for blk in 0..k_blocks {
                let base = (row * k_blocks + blk) * 36;
                build_block(&mut rng, &mut packed_host[base..base + 36]);
            }
        }
        let packed = upload(&packed_host, &stream);
        println!("\n{label}");

        for (vi, (variant, vname)) in variants.iter().enumerate() {
            for (mi, &m_total) in m_totals.iter().enumerate() {
                let mut x_host = vec![0f32; m_total * k];
                for v in x_host.iter_mut() {
                    *v = rng.f32_in(-1.0, 1.0);
                }
                let x = upload(&x_host, &stream);
                let out = Arc::new(
                    api::zeros::<f32>(&[m_total * n])
                        .sync_on(&stream)
                        .expect("out alloc"),
                );
                // Q8_1 scratch for one chunk (16 rows for the 16-row-cap
                // kernels, 128 for mma128; same sizing as the engine's
                // gemm_q4k_prefill_batch).
                let scr_rows = if matches!(*variant, Variant::Mma128 | Variant::Mma128w | Variant::Mmq) { 128 } else { CHUNK };
                let x_qs = Arc::new(
                    api::zeros::<u32>(&[scr_rows * k / 4])
                        .sync_on(&stream)
                        .expect("qs alloc"),
                );
                let x_ds = Arc::new(
                    api::zeros::<u32>(&[scr_rows * k.div_ceil(32)])
                        .sync_on(&stream)
                        .expect("ds alloc"),
                );

                let mut run_pass = |sync: bool| {
                    let mut row = 0usize;
                    // The mma128 variant chunks at 128 rows (weights once
                    // per launch); the others at the kernels' 16-row cap.
                    let chunk = if matches!(*variant, Variant::Mma128 | Variant::Mma128w | Variant::Mmq) { 128 } else { CHUNK };
                    while row < m_total {
                        let rows = (m_total - row).min(chunk);
                        let res = match variant {
                            Variant::Kpar8 => oxide.forward_sm120_kpar_batch_d2d_rows(
                                &x, row, &packed, &out, row, rows, n, k_blocks, &x_qs, &x_ds,
                            ),
                            Variant::Mma => oxide.forward_mma_batch_d2d_rows(
                                &x, row, &packed, &out, row, rows, n, k_blocks, &x_qs, &x_ds,
                            ),
                            Variant::Mma128 => oxide.forward_mma128_batch_d2d_rows(
                                &x, row, &packed, &out, row, rows, n, k_blocks, &x_qs, &x_ds,
                            ),
                            Variant::Mma128w => oxide.forward_mma128w_batch_d2d_rows(
                                &x, row, &packed, &out, row, rows, n, k_blocks, &x_qs, &x_ds,
                            ),
                            Variant::Mmq => oxide.forward_mmq_batch_d2d_rows(
                                &x, row, &packed, &out, row, rows, n, k_blocks, &x_qs, &x_ds,
                            ),
                            Variant::Streamk => oxide.forward_streamk_batch_d2d_rows(
                                &x, row, &packed, &out, row, rows, n, k_blocks, &x_qs, &x_ds,
                            ),
                        };
                        res.unwrap_or_else(|e| panic!("{vname} m={m_total} row {row}: {e}"));
                        row += rows;
                    }
                    if sync {
                        stream.synchronize().expect("timed sync");
                    }
                };

                for _ in 0..3 {
                    run_pass(true);
                }
                let iters = if m_total >= 256 { 10 } else { 30 };
                let start = Instant::now();
                for _ in 0..iters {
                    run_pass(false);
                }
                stream.synchronize().expect("timed sync");
                let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
                results[si][vi][mi] = ms;
                // Effective weight bandwidth: the chunk loop re-reads the
                // full weight matrix ceil(m_total/16) times.
                let weight_bytes = (n * k_blocks * 144) as f64;
                let chunk = if matches!(*variant, Variant::Mma128 | Variant::Mma128w | Variant::Mmq) { 128 } else { CHUNK };
                let re_reads = m_total.div_ceil(chunk) as f64;
                let gbps = weight_bytes * re_reads / (ms / 1e3) / 1e9;
                println!(
                    "  {vname:>12} m={m_total:>3}: {ms:>9.3} ms  ({gbps:>7.0} GB/s effective weight read)"
                );
            }
        }
    }

    // Model-level projection: per 128-token prefill step, the Q4K projection
    // stack of one Llama-3.1-8B layer is q + k + o + gate + up (v_proj and
    // down_proj are Q6K in Q4_K_M and stay on their own path). 32 layers.
    println!("\n=== projected Q4K projection cost per 128-token prefill step (32 layers) ===");
    let q4k_shapes = [(4096usize, 16usize), (1024, 16), (4096, 16), (14336, 16), (14336, 16)];
    for (vi, (_, vname)) in variants.iter().enumerate() {
        let mut total_ms = 0f64;
        for &(n, kb) in &q4k_shapes {
            let si = shapes.iter().position(|&(sn, skb, _)| sn == n && skb == kb).unwrap();
            // interpolate at m=128 (exact entry exists)
            total_ms += results[si][vi][2]; // m_totals[2] == 128
        }
        let layer_ms = total_ms;
        let step_ms = layer_ms * 32.0;
        let tok_s = 128.0 / (step_ms / 1e3);
        println!(
            "  {vname:>12}: {layer_ms:.3} ms/layer -> {step_ms:.1} ms/step -> ~{tok_s:.0} tok/s pp128 (Q4K projections only)"
        );
    }
    // ---- Q6K side: the db_batch kernel (raw f32 activations, weights once
    // per 16-row chunk) at the two Q6K prefill shapes of Q4_K_M ----
    {
        use tempura_kernels::oxide_q6k_gemv_host::OxideQ6kGemv;
        let oxide6 = OxideQ6kGemv::try_load(stream.clone()).expect("oxide q6k load");
        // native 53-u32 Q6K blocks: [0]=f32(d), [1..5] bias-128 scales,
        // [5..37] ql, [37..53] qh.
        let q6k_shapes: &[(usize, usize, &str)] = &[
            (1024, 16, "v_proj   [n=1024, k=4096] Q6K"),
            (4096, 56, "down     [n=4096, k=14336] Q6K"),
        ];
        println!("\n=== Q6K sm120_batch timing (f32 activations, chunked 16) ===");
        for &(n, k_blocks, label) in q6k_shapes {
            let k = k_blocks * 256;
            let mut rng = Rng(0x6E11_0000 + n as u64);
            let mut packed_host = vec![0u32; n * k_blocks * 53];
            for row in 0..n {
                for blk in 0..k_blocks {
                    let base = (row * k_blocks + blk) * 53;
                    let d = rng.f32_in(0.0005, 0.05);
                    packed_host[base] = d.to_bits();
                    for w in 1..53 {
                        packed_host[base + w] = rng.next_u32();
                    }
                }
            }
            let packed = upload(&packed_host, &stream);
            println!("{label}");
            for &m_total in m_totals {
                let mut x_host = vec![0f32; m_total * k];
                for v in x_host.iter_mut() {
                    *v = rng.f32_in(-1.0, 1.0);
                }
                let x = upload(&x_host, &stream);
                let out = Arc::new(
                    api::zeros::<f32>(&[m_total * n])
                        .sync_on(&stream)
                        .expect("out alloc"),
                );
                let mut run_pass = |sync: bool| {
                    let mut row = 0usize;
                    while row < m_total {
                        let rows = (m_total - row).min(CHUNK);
                        oxide6
                            .forward_sm120_batch_rows(
                                &x,
                                row,
                                &packed,
                                &out,
                                row,
                                n,
                                k_blocks,
                                rows,
                                stream.cu_stream(),
                            )
                            .unwrap_or_else(|e| panic!("q6k batch m={m_total} row {row}: {e}"));
                        row += rows;
                    }
                    if sync {
                        stream.synchronize().expect("timed sync");
                    }
                };
                for _ in 0..3 {
                    run_pass(true);
                }
                let iters = if m_total >= 256 { 10 } else { 30 };
                let start = Instant::now();
                for _ in 0..iters {
                    run_pass(false);
                }
                stream.synchronize().expect("timed sync");
                let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
                let weight_bytes = (n * k_blocks * 212) as f64;
                let re_reads = m_total.div_ceil(CHUNK) as f64;
                let gbps = weight_bytes * re_reads / (ms / 1e3) / 1e9;
                println!(
                    "  {:>12} m={m_total:>3}: {ms:>9.3} ms  ({gbps:>7.0} GB/s effective weight read)"
                , "q6k_sm120_batch");
            }
        }
    }

    println!("\n=== probe-prefill-tc done ===");
}
