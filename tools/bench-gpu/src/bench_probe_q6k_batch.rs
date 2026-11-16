// bench-probe-q6k-batch -- correctness cross-validation of the sm_120 Q6K
// batch GEMV kernels against the per-row kpar GEMV on RANDOM native-GGUF
// Q6K blocks.
//
// Kernels probed:
//   1. q6k_gemv_sm120_kpar8_db_batch (the NEW no-graph batch kernel): MUST
//      be bit-identical per row to the per-row kpar dispatch (which selects
//      the dual-block direct-load kpar8_db variant by default -- the kernel
//      the no-graph serving path launches per row). This is the gate for
//      routing gemm_any_d2d_batch's Q6K arm through it.
//   2. q6k_gemv_sm120_batch (the EXISTING two-phase graph-path kernel):
//      informational. Its cross-warp reduction is a 3-step lane butterfly,
//      not the per-row kernel's sequential thread-0 sum, so bit-identity is
//      NOT expected -- printed for the record (it stays on the equal-
//      position graph paths where the captured composition is fixed).
//
// Why random data: uniform/simple data cannot catch layout bugs (the
// mxf4-gemv / q4k-streamk lessons). Includes an lm_head-shaped probe
// (n = 128256) -- the largest n the no-graph path dispatches.
//
// Run: scripts/gpu-remote.sh bench probe-q6k-batch

use std::sync::Arc;
use cuda_core::Stream;
use cutile::api;
use cutile::prelude::{DeviceOp, ToHostVec};
use cutile::tensor::Tensor;
use tempura_kernels::oxide_q6k_gemv_host::OxideQ6kGemv;

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

/// Build one native-GGUF Q6K block (53 u32) from random parameters.
///
/// Layout: u32[0] = d (f32 bits, super-scale), u32[1..5] = 16 group scales
/// as bias-128 u8, u32[5..37] = ql[0..128] (lower 4 bits of each weight),
/// u32[37..53] = qh[0..64] (upper 2 bits of each weight).
fn build_block(rng: &mut Rng, out: &mut [u32]) {
    assert_eq!(out.len(), 53);
    out[0] = rng.f32_in(0.0005, 0.05).to_bits();
    for w in 1..53usize {
        out[w] = rng.next_u32();
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
        (1024, 16, "v_proj [n=1024, k=4096]"),
        (4096, 16, "proj [n=4096, k=4096]"),
        (4096, 56, "down [n=4096, k=14336]"),
        (128256, 16, "lm_head [n=128256, k=4096]"),
    ];
    let device = Arc::new(cuda_core::Device::new(0).expect("device"));
    let stream: Arc<Stream> = device
        .primary_context()
        .expect("ctx")
        .new_stream()
        .expect("stream");
    let oxide = OxideQ6kGemv::try_load(stream.clone()).expect("oxide q6k load");
    println!(
        "probe-q6k-batch: kpar={} db={} db_batch={} two_phase_batch={}",
        oxide.has_kpar(),
        oxide.has_sm120_mbar(),
        oxide.has_sm120_db_batch(),
        oxide.has_sm120_batch(),
    );
    assert!(
        oxide.has_sm120_db_batch(),
        "q6k_gemv_sm120_kpar8_db_batch not loaded (PTX load failed?)"
    );

    let mut any_fail = false;
    for &(n, k_blocks, label) in shapes {
        let k = k_blocks * 256;
        let mut rng = Rng(0x6E6B_0000 + n as u64 + k_blocks as u64);

        // Random native-GGUF weights [n * k_blocks * 53].
        let mut packed_host = vec![0u32; n * k_blocks * 53];
        for row in 0..n {
            for blk in 0..k_blocks {
                let base = (row * k_blocks + blk) * 53;
                build_block(&mut rng, &mut packed_host[base..base + 53]);
            }
        }
        let packed = upload(&packed_host, &stream);

        for &m in &[1usize, 2, 4, 6] {
            // Random activations [m * k].
            let mut x_host = vec![0f32; m * k];
            for v in x_host.iter_mut() {
                *v = rng.f32_in(-1.0, 1.0);
            }
            let x = upload(&x_host, &stream);

            // Ground truth: the per-row kpar dispatch (selects kpar8_db by
            // default -- the exact kernel the no-graph path launches per
            // row via gemm_q6k_d2d_slot -> forward_kpar_d2d).
            let mut ref_rows: Vec<Vec<f32>> = Vec::with_capacity(m);
            for r in 0..m {
                let row = upload(&x_host[r * k..(r + 1) * k], &stream);
                let out = oxide
                    .forward_kpar_d2d(&row, &packed, n, k_blocks)
                    .expect("per-row kpar gemv");
                ref_rows.push(out.to_host_vec().sync_on(&stream).expect("kpar download"));
            }

            // 1) q6k_gemv_sm120_kpar8_db_batch: MUST be bit-identical to the
            //    per-row reference. This is the gate for the no-graph batch
            //    path's Q6K arm.
            {
                let out = Arc::new(
                    api::zeros::<f32>(&[m * n]).sync_on(&stream).expect("out alloc"),
                );
                oxide
                    .forward_sm120_db_batch_d2d_rows(
                        &x,
                        0,
                        &packed,
                        &out,
                        0,
                        m,
                        n,
                        k_blocks,
                        stream.cu_stream(),
                    )
                    .expect("db batch launch");
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
                    "  {label} m={m} db_batch: {} (bit-identical to per-row kpar8_db)",
                    if exact { "OK" } else { "MISMATCH" }
                );
            }

            // 2) The existing two-phase q6k_gemv_sm120_batch (graph path):
            //    informational -- bit-identity NOT expected (3-step lane
            //    butterfly cross-warp reduce vs the per-row sequential sum).
            //    Only defined for m in 2..=16.
            if m >= 2 {
                let out = Arc::new(
                    api::zeros::<f32>(&[m * n]).sync_on(&stream).expect("out alloc"),
                );
                let res = oxide.forward_sm120_batch_d2d_into(
                    &x, &packed, &out, n, k_blocks, m, stream.cu_stream(),
                );
                match res {
                    Ok(()) => {
                        let got = out.to_host_vec().sync_on(&stream).expect("out download");
                        let mut max_rel = 0f64;
                        let mut exact = true;
                        for r in 0..m {
                            for c in 0..n {
                                if got[r * n + c].to_bits() != ref_rows[r][c].to_bits() {
                                    exact = false;
                                }
                                let a = got[r * n + c];
                                let b = ref_rows[r][c];
                                let denom = b.abs().max(1.0);
                                max_rel = max_rel.max(((a - b).abs() / denom) as f64);
                            }
                        }
                        println!(
                            "  {label} m={m} two_phase_batch: {} (max rel err {max_rel:.3e}) \
                             [informational: graph-path kernel, reduce order differs]",
                            if exact { "bit-identical" } else { "not bit-identical" }
                        );
                    }
                    Err(e) => {
                        println!("  {label} m={m} two_phase_batch: SKIP ({e})");
                    }
                }
            }
        }
    }
    println!(
        "\n=== probe-q6k-batch: {} ===",
        if any_fail { "FAIL" } else { "PASS" }
    );
    if any_fail {
        std::process::exit(1);
    }
}
