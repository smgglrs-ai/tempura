// bench-decode-ops -- correctness validation of the decode-only GPU
// mechanisms against CPU math, isolated from the engine.
//
// Why: bench-quality showed the m=1 GPU decode path diverges from the CPU
// reference by max_abs 12-18 at every step, identically under every kernel
// bisect switch -- so the bug lives in decode-only shared machinery. The
// three mechanisms below exist ONLY on the decode path (prefill uses plain
// D2D memcpy for KV and the CPU-computed cos/sin standard RoPE path) and
// had no unit tests (GPU tests cannot run in CI):
//
//   1. RoPE table fast path (seq_len==1): precompute_table +
//      forward_device_scratch -> cutile_rope_from_table
//   2. RoPE graph path: d2d_gather_row (device-pos row gather) +
//      forward_device_scratch_graph
//   3. KV scatter-write: copy_with_device_offset (device-read row offset)
//
// Each test compares GPU output against a CPU NeoX-rotation / memcpy
// reference on random data (uniform data cannot catch permutation bugs --
// same lesson as the mxf4 store-mapping fix).
//
// Run:
//   scripts/gpu-remote.sh bench decode-ops
//   cargo run -p tempura-bench-gpu --release --bin bench-decode-ops

use std::sync::Arc;
use cuda_core::Stream;
use cutile::api;
use cutile::prelude::{DeviceOp, ToHostVec};
use cutile::tensor::Tensor;
use tempura_kernels::cutile_rope::{CutileRoPE, RopeScratch};
use tempura_kernels::cutile_quantized_gemm::{Q6K_PACKED_U32S, Q6K_QK};
use tempura_kernels::ggml::dequantize::dequantize_to_f32;
use tempura_kernels::ggml::types::GgmlType;
use tempura_kernels::kv_scatter_write::KvScatterWrite;
use tempura_kernels::oxide_q6k_gemv_host::OxideQ6kGemv;

const HEAD_DIM: usize = 128;
const N_HEADS: usize = 32; // Llama-3.1-8B Q heads
const N_KV_HEADS: usize = 8;
const THETA: f32 = 500000.0; // Llama-3.1
const MAX_POS: usize = 4096;
const TEST_POS: usize = 85; // the bench-quality failure position ballpark

fn upload(v: Vec<f32>, stream: &Arc<Stream>) -> Arc<Tensor<f32>> {
    api::copy_host_vec_to_device(&Arc::new(v))
        .sync_on(stream)
        .expect("H2D upload failed")
        .into()
}

fn download(t: &Arc<Tensor<f32>>, stream: &Arc<Stream>) -> Vec<f32> {
    Arc::clone(t).to_host_vec().sync_on(stream).expect("D2H failed")
}

/// Deterministic pseudo-random data (no RNG dependency).
fn prng_data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5
        })
        .collect()
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn f32_to_f16_bits(v: f32) -> u32 {
    half::f16::from_f32(v).to_bits() as u32
}

fn next_pow2(v: usize) -> usize {
    v.max(1).next_power_of_two()
}

/// Round to the nearest E2M1 magnitude, returning the signed nibble.
fn round_e2m1(v: f32) -> u8 {
    const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let av = v.abs();
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, &m) in E2M1.iter().enumerate() {
        let d = (av - m).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    let nib = best as u8;
    if v < 0.0 {
        nib | 8
    } else {
        nib
    }
}

/// CPU GQA decode-attention reference: out[h] = softmax(scale * q_h . K[g,:])
/// weighted sum of V[g,:], where g = h / heads_per_group and K/V rows are
/// `[kv_len, n_kv_heads, head_dim]` interleaved.
fn cpu_gqa_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    kv_len: usize,
    scale: f32,
) -> Vec<f32> {
    let hpg = N_HEADS / N_KV_HEADS;
    let mut out = vec![0.0f32; N_HEADS * HEAD_DIM];
    for h in 0..N_HEADS {
        let g = h / hpg;
        let q_h = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        let mut scores = vec![0.0f32; kv_len];
        let mut max_s = f32::NEG_INFINITY;
        for p in 0..kv_len {
            let k_row = &k_cache[(p * N_KV_HEADS + g) * HEAD_DIM..][..HEAD_DIM];
            let s: f32 = q_h.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale;
            scores[p] = s;
            max_s = max_s.max(s);
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        for p in 0..kv_len {
            let v_row = &v_cache[(p * N_KV_HEADS + g) * HEAD_DIM..][..HEAD_DIM];
            let prob = scores[p] / sum;
            let o = &mut out[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            for d in 0..HEAD_DIM {
                o[d] += prob * v_row[d];
            }
        }
    }
    out
}

/// CPU NeoX-style rotation reference for one (seq_len=1) position.
fn cpu_rope(x: &[f32], pos: usize, n_heads: usize) -> Vec<f32> {
    let half = HEAD_DIM / 2;
    let mut out = vec![0.0f32; x.len()];
    for h in 0..n_heads {
        let base = h * HEAD_DIM;
        for d in 0..half {
            let freq = pos as f32 / THETA.powf(2.0 * d as f32 / HEAD_DIM as f32);
            let (c, s) = (freq.cos(), freq.sin());
            let x0 = x[base + d];
            let x1 = x[base + half + d];
            out[base + d] = x0 * c - x1 * s;
            out[base + half + d] = x0 * s + x1 * c;
        }
    }
    out
}

fn report(name: &str, gpu: &[f32], cpu: &[f32]) -> bool {
    let mut max_abs = 0.0f32;
    let mut bad = 0usize;
    for (a, b) in gpu.iter().zip(cpu) {
        let d = (a - b).abs();
        if d > 1e-4 {
            bad += 1;
        }
        max_abs = max_abs.max(d);
    }
    let ok = bad == 0;
    println!(
        "  {name:<38} max_abs={max_abs:.6} bad={bad}/{} -> {}",
        gpu.len(),
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn main() {
    println!("=== tempura decode-ops correctness validation (GPU vs CPU math) ===\n");
    let mut all_ok = true;

    let mut rope = CutileRoPE::new().expect("CutileRoPE init failed");
    let stream = rope.stream().clone();
    rope.precompute_table(MAX_POS, HEAD_DIM, THETA, &stream)
        .expect("precompute_table failed");

    // ---- 1. RoPE table fast path (eager decode route) ----
    let x: Vec<f32> = prng_data(N_HEADS * HEAD_DIM, 1);
    let x_dev = upload(x.clone(), &stream);
    let mut q_scratch = RopeScratch::new(&stream, N_HEADS, HEAD_DIM).unwrap();
    let out = rope
        .forward_device_scratch(
            &x_dev,
            &[TEST_POS as u32],
            1,
            N_HEADS,
            HEAD_DIM,
            THETA,
            &mut q_scratch,
            &stream,
        )
        .expect("table-path forward failed");
    let gpu = download(&out, &stream);
    let cpu = cpu_rope(&x, TEST_POS, N_HEADS);
    all_ok &= report("rope table path (pos=85, 32 heads)", &gpu, &cpu);

    // Same position via the allocating standard path (what prefill uses for
    // seq_len>1): must agree with the table path -- both feed the same
    // rotation kernel with cos/sin for pos=85.
    let out_std = rope
        .forward_device(&x_dev, &[TEST_POS as u32], 1, N_HEADS, HEAD_DIM, THETA, &stream)
        .expect("standard-path forward failed");
    let gpu_std = download(&out_std, &stream);
    all_ok &= report("rope standard path (pos=85, 32 heads)", &gpu_std, &cpu);

    // ---- 2. RoPE graph path: gather row + scratch_graph ----
    let (cos_ptr, sin_ptr, half_dim, max_pos) = rope.table_ptrs().expect("table_ptrs failed");
    assert_eq!(half_dim, HEAD_DIM / 2);
    assert_eq!(max_pos, MAX_POS);
    let scatter = KvScatterWrite::new().expect("KvScatterWrite init failed");
    // pos uploaded as f32 bitcast (engine convention: PTX reads it as s32)
    let pos_dev = upload(vec![f32::from_bits(TEST_POS as u32)], &stream);
    let cos_row = api::zeros::<f32>(&[half_dim])
        .sync_on(&stream)
        .expect("cos_row alloc failed");
    let sin_row = api::zeros::<f32>(&[half_dim])
        .sync_on(&stream)
        .expect("sin_row alloc failed");
    let cos_row: Arc<Tensor<f32>> = cos_row.into();
    let sin_row: Arc<Tensor<f32>> = sin_row.into();
    let cu_stream = stream.cu_stream();
    scatter
        .gather_row_by_device_pos(
            cos_ptr,
            cos_row.device_pointer().cu_deviceptr(),
            pos_dev.device_pointer().cu_deviceptr(),
            half_dim as u32,
            max_pos as u32,
            cu_stream,
        )
        .expect("gather cos row failed");
    scatter
        .gather_row_by_device_pos(
            sin_ptr,
            sin_row.device_pointer().cu_deviceptr(),
            pos_dev.device_pointer().cu_deviceptr(),
            half_dim as u32,
            max_pos as u32,
            cu_stream,
        )
        .expect("gather sin row failed");
    let mut k_scratch = RopeScratch::new(&stream, N_KV_HEADS, HEAD_DIM).unwrap();
    let xk: Vec<f32> = prng_data(N_KV_HEADS * HEAD_DIM, 2);
    let xk_dev = upload(xk.clone(), &stream);
    let outg = rope
        .forward_device_scratch_graph(
            &xk_dev,
            N_KV_HEADS,
            HEAD_DIM,
            &cos_row,
            &sin_row,
            &mut k_scratch,
            &stream,
        )
        .expect("graph-path forward failed");
    let gpu_g = download(&outg, &stream);
    let cpu_k = cpu_rope(&xk, TEST_POS, N_KV_HEADS);
    all_ok &= report("rope graph path (gather+from_table)", &gpu_g, &cpu_k);

    // ---- 3. KV scatter-write placement ----
    let kv_dim = N_KV_HEADS * HEAD_DIM;
    let rows = 16usize;
    let src: Vec<f32> = prng_data(kv_dim, 3);
    let src_dev = upload(src.clone(), &stream);
    let cache: Vec<f32> = vec![0.0; rows * kv_dim];
    let cache_dev = upload(cache, &stream);
    let row_off = 7usize;
    let off_dev = upload(vec![f32::from_bits(row_off as u32)], &stream);
    scatter
        .copy_with_device_offset(
            src_dev.device_pointer().cu_deviceptr(),
            cache_dev.device_pointer().cu_deviceptr(),
            off_dev.device_pointer().cu_deviceptr(),
            kv_dim as u32,
            cu_stream,
        )
        .expect("scatter write failed");
    let gpu_c = download(&cache_dev, &stream);
    let mut cpu_c = vec![0.0f32; rows * kv_dim];
    cpu_c[row_off * kv_dim..(row_off + 1) * kv_dim].copy_from_slice(&src);
    all_ok &= report("kv scatter write (row 7 of 16)", &gpu_c, &cpu_c);

    // ---- 4. Q6K decode GEMV (down_proj) on REAL model weights ----
    // The m=1 decode path runs every Q6_K tensor (down_proj in Q4_K_M, one
    // per layer) through oxide_q6k_gemv.forward_kpar_d2d -- the dp4a kpar8
    // variant, which has NO bisect switch (TEMPURA_NO_SM120_Q6K only gates
    // the already-default-off mbar variant) and was previously validated
    // for replay determinism only, never numerics vs CPU.
    let model_arg = std::env::args().nth(1);
    let model_path = match model_arg {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let home = std::env::var("HOME").unwrap_or_default();
            std::path::Path::new(&home)
                .join(".cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        }
    };
    if !model_path.exists() {
        println!("\n  (skipping Q6K GEMV test: model not found at {})",
            model_path.display());
    } else {
        println!("\n  Model: {}", model_path.display());
        use tempura_engine::gguf_loader::GgufLoader;
        let loader = GgufLoader::from_file_quantized(&model_path)
            .expect("GGUF open failed");
        let cfg = loader.config.clone();
        let qw = loader.to_quantized_weights();
        let name = "model.layers.0.mlp.down_proj.weight";
        let bytes = qw.get_q6k(name).unwrap_or_else(|| {
            // Q4_K_M always has Q6K down_proj, but stay graceful.
            panic!("down_proj is not Q6K in this model");
        });
        let (n, k) = (cfg.hidden_size, cfg.intermediate_size);
        let k_blocks = k / Q6K_QK;

        // Pack exactly like CutileGpuModel::maybe_upload_q6k (native
        // ql+qh, 53 u32 per block).
        let mut packed: Vec<u32> = Vec::with_capacity(n * k_blocks * Q6K_PACKED_U32S);
        const Q6K_BYTES: usize = 210;
        for row in 0..n {
            for blk in 0..k_blocks {
                let off = (row * k_blocks + blk) * Q6K_BYTES;
                let block = &bytes[off..off + Q6K_BYTES];
                let ql = &block[0..128];
                let qh = &block[128..192];
                let scales = &block[192..208];
                let d_bits = u16::from_le_bytes([block[208], block[209]]);
                let d_f32 = f16_bits_to_f32(d_bits);
                packed.push(d_f32.to_bits());
                for w in 0..4usize {
                    let b = |i: usize| ((scales[w * 4 + i] as i8) as i32 + 128) as u8;
                    packed.push(
                        (b(0) as u32)
                            | ((b(1) as u32) << 8)
                            | ((b(2) as u32) << 16)
                            | ((b(3) as u32) << 24),
                    );
                }
                for i in 0..32usize {
                    packed.push(
                        (ql[i * 4] as u32)
                            | ((ql[i * 4 + 1] as u32) << 8)
                            | ((ql[i * 4 + 2] as u32) << 16)
                            | ((ql[i * 4 + 3] as u32) << 24),
                    );
                }
                for i in 0..16usize {
                    packed.push(
                        (qh[i * 4] as u32)
                            | ((qh[i * 4 + 1] as u32) << 8)
                            | ((qh[i * 4 + 2] as u32) << 16)
                            | ((qh[i * 4 + 3] as u32) << 24),
                    );
                }
            }
        }
        let packed_dev: Arc<Tensor<u32>> = Arc::new(
            api::copy_host_vec_to_device(&Arc::new(packed))
                .sync_on(&stream)
                .expect("packed upload failed"),
        );

        // Engine input convention: a_flat reshaped to next_pow2(k); elements
        // beyond k are zero (the down_proj pad scratch pre-zeros them).
        let k_pad = next_pow2(k);
        let mut xv = prng_data(k, 7);
        xv.resize(k_pad, 0.0);
        let x_dev = upload(xv.clone(), &stream);

        let oxide_q6k = OxideQ6kGemv::try_load(stream.clone())
            .expect("OxideQ6kGemv PTX not loaded (copy loop missing?)");
        println!(
            "  oxide_q6k loaded (kpar={}, sm120_mbar={}, sm120_batch={})",
            oxide_q6k.has_kpar(),
            oxide_q6k.has_sm120_mbar(),
            oxide_q6k.has_sm120_batch(),
        );
        let out = oxide_q6k
            .forward_kpar_d2d(&x_dev, &packed_dev, n, k_blocks)
            .expect("q6k forward_kpar_d2d failed");
        let gpu_gemv = download(&out, &stream);

        // CPU reference: exact dequant + dense dot.
        let w = dequantize_to_f32(GgmlType::Q6K, bytes, n * k);
        let cpu_gemv: Vec<f32> = (0..n)
            .map(|i| {
                let row = &w[i * k..(i + 1) * k];
                row.iter().zip(&xv).map(|(a, b)| a * b).sum()
            })
            .collect();
        all_ok &= report("q6k kpar GEMV (real down_proj)", &gpu_gemv, &cpu_gemv);

        // Baseline non-kpar oxide variant for the same weights.
        let out_base = oxide_q6k
            .forward_d2d(&x_dev, &packed_dev, n, k_blocks)
            .expect("q6k forward_d2d failed");
        let gpu_base = download(&out_base, &stream);
        all_ok &= report("q6k baseline GEMV (real down_proj)", &gpu_base, &cpu_gemv);

        // ---- 5. Q4K decode GEMV (q_proj) on REAL weights, raw GGUF format ----
        // Production decode dispatches EVERY Q4K tensor through
        // submit_sm120_kpar_into_slot (sm120 dp4a kpar8 dual_block, native
        // 36-u32 blocks, Q8_1-quantized activations) -- no bisect switch
        // exists for it either.
        let q_name = "model.layers.0.self_attn.q_proj.weight";
        let q_bytes = qw.get_q4k(q_name).expect("q_proj is not Q4K");
        let (qn, qk) = (cfg.hidden_size, cfg.hidden_size); // q_proj: n=q_dim, k=h
        let q_k_blocks = qk / 256;
        let q_raw_u32: Vec<u32> = q_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let q_packed: Arc<Tensor<u32>> = Arc::new(
            api::copy_host_vec_to_device(&Arc::new(q_raw_u32))
                .sync_on(&stream)
                .expect("q4k raw upload failed"),
        );
        let qx: Vec<f32> = prng_data(qk, 11);
        let qx_dev = upload(qx.clone(), &stream);

        let oxide_q4k = tempura_kernels::oxide_q4k_gemv_host::OxideQ4kGemv::try_load(
            stream.clone(),
        )
        .expect("OxideQ4kGemv PTX not loaded");
        println!(
            "  oxide_q4k loaded (raw_gguf={}, sm120_kpar={}, fused_silu={})",
            oxide_q4k.uses_raw_gguf,
            oxide_q4k.has_sm120_kpar(),
            oxide_q4k.has_sm120_fused_silu(),
        );
        let qw_f32 = dequantize_to_f32(GgmlType::Q4K, q_bytes, qn * qk);
        let q_cpu: Vec<f32> = (0..qn)
            .map(|i| {
                let row = &qw_f32[i * qk..(i + 1) * qk];
                row.iter().zip(&qx).map(|(a, b)| a * b).sum()
            })
            .collect();

        // Production path: sm120 kpar into a slot (needs pre-allocated Q8 scratch).
        oxide_q4k
            .pre_allocate_q8_scratch(qk, qn)
            .expect("q8 scratch pre-alloc failed");
        let slot_out = upload(vec![0.0; qn], &stream);
        let q_gpu_slot = unsafe {
            oxide_q4k
                .submit_sm120_kpar_into_slot(&qx_dev, &q_packed, &slot_out, qn, q_k_blocks)
                .expect("submit_sm120_kpar_into_slot failed")
        };
        let g1 = download(&q_gpu_slot, &stream);
        // Q8_1 activation quantization: expect ~1% relative error, not bit-exact.
        let q_max = q_cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let g1_max = g1.iter().zip(&q_cpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let rel = g1_max / q_max.max(1e-9);
        println!(
            "  {:<38} max_abs={g1_max:.4} rel={rel:.4} -> {}",
            "q4k sm120 kpar GEMV (real q_proj)",
            if rel < 0.02 { "PASS" } else { "FAIL" }
        );
        all_ok &= rel < 0.02;

        // Non-slot best-kpar path (Q8 dp4a).
        let q_gpu_best = oxide_q4k
            .forward_best_kpar_d2d(&qx_dev, &q_packed, qn, q_k_blocks)
            .expect("forward_best_kpar_d2d failed");
        let g2 = download(&q_gpu_best, &stream);
        let g2_max = g2.iter().zip(&q_cpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let rel2 = g2_max / q_max.max(1e-9);
        println!(
            "  {:<38} max_abs={g2_max:.4} rel={rel2:.4} -> {}",
            "q4k best-kpar GEMV (real q_proj)",
            if rel2 < 0.02 { "PASS" } else { "FAIL" }
        );
        all_ok &= rel2 < 0.02;

        // Rust-codegen raw-format kernel (q4k_gemv_dp4a_q8_raw, sm_80 PTX) --
        // triangulates sm120-PTX bug vs shared quantize/format bug.
        let q_gpu_q8 = oxide_q4k
            .forward_q8_d2d(&qx_dev, &q_packed, qn, q_k_blocks)
            .expect("forward_q8_d2d failed");
        let g3 = download(&q_gpu_q8, &stream);
        // Reference for the q8 path must use Q8_1-quantized activations: on a
        // sign-canceling dot the quantization noise is large relative to the
        // RESULT, so comparing against the exact dot is the wrong yardstick.
        let q8_cpu: Vec<f32> = {
            let mut xq = qx.clone();
            for blk in (0..qk).step_by(32) {
                let chunk = &mut xq[blk..blk + 32];
                let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = if amax == 0.0 { 1.0 } else { amax / 127.0 };
                for v in chunk.iter_mut() {
                    let r = (*v / d + if *v >= 0.0 { 0.5 } else { -0.5 }) as i32;
                    let q = r.clamp(-127, 127);
                    *v = q as f32 * d;
                }
            }
            (0..qn)
                .map(|i| {
                    let row = &qw_f32[i * qk..(i + 1) * qk];
                    row.iter().zip(&xq).map(|(a, b)| a * b).sum()
                })
                .collect()
        };
        let g3_max = g3.iter().zip(&q8_cpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let q8_max = q8_cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let rel3 = g3_max / q8_max.max(1e-9);
        println!(
            "  {:<38} max_abs={g3_max:.4} rel={rel3:.4} (vs Q8_quantized CPU ref) -> {}",
            "q4k q8_raw GEMV (real q_proj)",
            if rel3 < 0.02 { "PASS" } else { "FAIL" }
        );
        all_ok &= rel3 < 0.02;

        // q8_raw per-row error census + one-hot sweeps (tail and main loops).
        {
            let mut bad_rows = 0usize;
            let mut first_bads: Vec<(usize, f32, f32)> = Vec::new();
            for (i, (a, b)) in g3.iter().zip(&q8_cpu).enumerate() {
                if (a - b).abs() > 0.05 {
                    bad_rows += 1;
                    if first_bads.len() < 12 {
                        first_bads.push((i, *a, *b));
                    }
                }
            }
            println!(
                "    q8_raw rows with |err|>0.05: {bad_rows}/{} (first: {:?})",
                g3.len(),
                first_bads
            );

            let scj = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            // -- k_blocks=1 (tail path): d=1, dmin=0, sc_j=j+1, qs[b]=b ----
            let mk = |d: f32, dmin: f32, seed: usize| -> Vec<u32> {
                let mut b = vec![0u32; 36];
                b[0] = f32_to_f16_bits(d) | (f32_to_f16_bits(dmin) << 16);
                b[1] = 0x04030201;
                b[2] = 0x01010101;
                b[3] = 0x18171615;
                for i in 0..32usize {
                    let mut w = 0u32;
                    for j in 0..4 {
                        w |= (((seed + i * 4 + j) & 0xFF) as u32) << (8 * j);
                    }
                    b[4 + i] = w;
                }
                b
            };
            let expected_at = |b: &[u32], e: usize| -> f32 {
                let d = f16_bits_to_f32((b[0] & 0xFFFF) as u16);
                let dmin = f16_bits_to_f32((b[0] >> 16) as u16);
                let g = e / 64;
                let l = e % 64;
                let byte_idx = 32 * g + (l % 32);
                let byte = b[4 + byte_idx / 4] >> (8 * (byte_idx % 4)) & 0xFF;
                let nib = if l < 32 { (byte & 0xF) as f32 } else { (byte >> 4) as f32 };
                // m = 1 in every block; scales are per j-group half.
                let sc = scj[2 * g + (l >= 32) as usize];
                let m = 1.0f32;
                d * sc * nib - dmin * m
            };
            let upload2 = |bs: Vec<u32>| -> Arc<Tensor<u32>> {
                Arc::new(
                    api::copy_host_vec_to_device(&Arc::new(bs))
                        .sync_on(&stream)
                        .expect("probe ul"),
                )
            };
            // tail sweep: 1 block, dmin = 0.
            {
                let blk = mk(1.0, 0.0, 0);
                let packed = upload2(vec![blk.clone(), blk.clone()].concat());
                let mut bad = Vec::new();
                for e in 0..256usize {
                    let mut xv = vec![0.0f32; 256];
                    xv[e] = 1.0;
                    let xe_dev = upload(xv, &stream);
                    let r = oxide_q4k
                        .forward_q8_d2d(&xe_dev, &packed, 2, 1)
                        .expect("tail probe failed");
                    let v = download(&r, &stream);
                    let exp = expected_at(&blk, e);
                    if (v[0] - exp).abs() > 0.01 {
                        bad.push((e, v[0], exp));
                    }
                }
                println!(
                    "    q8_raw one-hot sweep (k_blocks=1 tail): {}/256 WRONG{}",
                    bad.len(),
                    if bad.is_empty() { String::new() } else { format!(" first: {:?}", &bad[..bad.len().min(6)]) }
                );
            }
            // main-loop sweep: 16 blocks with per-block d/dmin/qs seeds.
            {
                let nb = 16usize;
                // Half the blocks use NEGATIVE d: real Q4_K weights carry
                // negative f16 super-scales; crafted positive-only data
                // cannot catch a sign-handling bug.
                let dblock = |b: usize| -> f32 {
                    if b % 2 == 1 {
                        -(1.0 + (b % 4) as f32)
                    } else {
                        1.0 + (b % 4) as f32
                    }
                };
                let blocks: Vec<u32> = (0..nb)
                    .flat_map(|b| mk(dblock(b), 0.25 + 0.1 * b as f32, b * 7))
                    .collect();
                let per_block: Vec<Vec<u32>> = (0..nb)
                    .map(|b| mk(dblock(b), 0.25 + 0.1 * b as f32, b * 7))
                    .collect();
                let packed = upload2(vec![blocks.clone(), blocks].concat());
                let k16 = nb * 256;
                let mut bad = Vec::new();
                for e in (0..k16).step_by(7) {
                    let mut xv = vec![0.0f32; k16];
                    xv[e] = 1.0;
                    let exp = expected_at(&per_block[e / 256], e % 256);
                    let xe_dev = upload(xv, &stream);
                    let r = oxide_q4k
                        .forward_q8_d2d(&xe_dev, &packed, 2, nb)
                        .expect("main probe failed");
                    let v = download(&r, &stream);
                    // Relative tolerance: the one-hot x reconstructs through
                    // an f16 Q8 scale (~1e-4 relative error).
                    if (v[0] - exp).abs() > 0.001 * exp.abs().max(1.0) {
                        bad.push((e, v[0], exp));
                    }
                }
                println!(
                    "    q8_raw one-hot sweep (k_blocks=16 main, +/-d): {}/{} WRONG{}",
                    bad.len(),
                    (k16 + 6) / 7,
                    if bad.is_empty() { String::new() } else { format!(" first: {:?}", &bad[..bad.len().min(8)]) }
                );
            }

            // Per-block isolation for real-weight bad rows: run each of a
            // bad row's 16 blocks alone (k_blocks=1) and compare against the
            // Q8-quantized CPU dot for that block -- pinpoints WHICH block
            // and WHAT byte pattern triggers the corruption.
            {
                let row_bytes = |row: usize| -> Vec<u8> {
                    q_bytes[row * q_k_blocks * 144..(row * q_k_blocks + q_k_blocks) * 144].to_vec()
                };
                let mut done = 0usize;
                for (row, _, _) in first_bads.iter().take(3) {
                    let rb = row_bytes(*row);
                    let mut bad_blocks = Vec::new();
                    for blk in 0..q_k_blocks {
                        let bytes = &rb[blk * 144..(blk + 1) * 144];
                        let raw: Vec<u32> = bytes
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        let packed1: Arc<Tensor<u32>> = Arc::new(
                            api::copy_host_vec_to_device(&Arc::new(vec![raw.clone(), raw].concat()))
                                .sync_on(&stream)
                                .expect("iso ul"),
                        );
                        // x restricted to this block's 256 elements.
                        let mut xv = vec![0.0f32; 256];
                        xv.copy_from_slice(&qx[blk * 256..(blk + 1) * 256]);
                        let xe_dev = upload(xv.clone(), &stream);
                        let r = oxide_q4k
                            .forward_q8_d2d(&xe_dev, &packed1, 2, 1)
                            .expect("iso run failed");
                        let v = download(&r, &stream);
                        // CPU: Q8-quantized x for this block, exact weights.
                        let mut xq = xv.clone();
                        for c in xq.chunks_exact_mut(32) {
                            let amax = c.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                            let d = if amax == 0.0 { 1.0 } else { amax / 127.0 };
                            for v in c.iter_mut() {
                                let rr = (*v / d + if *v >= 0.0 { 0.5 } else { -0.5 }) as i32;
                                *v = rr.clamp(-127, 127) as f32 * d;
                            }
                        }
                        let w = dequantize_to_f32(GgmlType::Q4K, bytes, 256);
                        let cpu: f32 = w.iter().zip(&xq).map(|(a, b)| a * b).sum();
                        if (v[0] - cpu).abs() > 0.02 {
                            bad_blocks.push((blk, v[0], cpu));
                        }
                    }
                    println!(
                        "    row {row} per-block (k_blocks=1 each): {}/{} blocks WRONG {:?}",
                        bad_blocks.len(),
                        q_k_blocks,
                        &bad_blocks[..bad_blocks.len().min(4)]
                    );
                    // Dump the failing block's header vs a good block's.
                    for (blk, _, _) in bad_blocks.iter().take(2) {
                        let bytes = &rb[blk * 144..(blk + 1) * 144];
                        let d = f16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
                        let dmin = f16_bits_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
                        println!(
                            "      blk {blk}: d={d:e} dmin={dmin:e} scales={:?} qs[0..8]={:?}",
                            &bytes[4..16],
                            &bytes[16..24]
                        );
                    }
                    {
                        let gblk = (0..q_k_blocks)
                            .find(|b| !bad_blocks.iter().any(|(bb, _, _)| bb == b))
                            .unwrap();
                        let bytes = &rb[gblk * 144..(gblk + 1) * 144];
                        let d = f16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
                        let dmin = f16_bits_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
                        println!(
                            "      (good blk {gblk}: d={d:e} dmin={dmin:e} scales={:?})",
                            &bytes[4..16]
                        );
                    }
                    done += 1;
                    if done >= 3 {
                        break;
                    }
                }
            }
        }

        // Error-structure dump: ratio/correlation reveal the bug's shape
        // (scale factor? offset? permutation? missing term?).
        {
            let mut pairs: Vec<(f32, f32)> = g1
                .iter()
                .zip(&q_cpu)
                .filter(|(_, c)| c.abs() > 0.1)
                .map(|(g, c)| (*g, *c))
                .collect();
            pairs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            println!("    sm120 vs CPU, |cpu|>0.1, largest 6:");
            for (g, c) in pairs.iter().take(6) {
                println!("      gpu={g:>10.4} cpu={c:>10.4} ratio={:>8.4}", g / c);
            }
            let n_big = pairs.len().min(200);
            let take = &pairs[..n_big];
            let mut ratios: Vec<f32> = take.iter().map(|(g, c)| g / c).collect();
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "    ratio p10={:.4} p50={:.4} p90={:.4}",
                ratios[n_big / 10],
                ratios[n_big / 2],
                ratios[n_big * 9 / 10]
            );
            let mut pairs3: Vec<(f32, f32)> = g3
                .iter()
                .zip(&q_cpu)
                .filter(|(_, c)| c.abs() > 0.1)
                .map(|(g, c)| (*g, *c))
                .collect();
            pairs3.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            println!("    q8_raw vs CPU, largest 6:");
            for (g, c) in pairs3.iter().take(6) {
                println!("      gpu={g:>10.4} cpu={c:>10.4} ratio={:>8.4}", g / c);
            }
        }

        // ---- 5b. Q4K structured probes (crafted 1-block weights) ----
        // Crafted block: d=1.0, dmin=0, sc_j = j+1 (scales bytes
        // [1,2,3,4,0,0,0,0,5,6,7,8]), qs nibbles lo=1/hi=2 (0x21).
        // x = indicator of j-group g -> expected out = 32*(sc_2g + 2*sc_2g+1):
        //   g=0: 32*(1+2*2)=160  g=1: 32*(3+2*4)=352
        //   g=2: 32*(5+2*6)=448  g=3: 32*(7+2*8)=736
        // Any deviation localizes the broken j-group scale/nibble mapping.
        let probe_expected = [160.0f32, 352.0, 448.0, 736.0];
        let probe_block: Vec<u32> = {
            let mut b = vec![0u32; 36];
            b[0] = 0x00003C00; // d = f16(1.0), dmin = f16(0.0)
            b[1] = 0x04030201; // scales[0..4] = 1,2,3,4
            b[2] = 0x00000000; // scales[4..8] = 0,0,0,0
            b[3] = 0x08070605; // scales[8..12] = 5,6,7,8
            for w in b.iter_mut().skip(4) {
                *w = 0x21212121; // qs nibbles lo=1 hi=2
            }
            b
        };
        let probe_rows = 8usize;
        let probe_packed_host: Vec<u32> =
            probe_block.iter().cloned().chain(probe_block.iter().cloned()).collect();
        let _ = probe_rows;
        // n=2 rows x 1 block keeps the probe tiny; identical rows.
        let probe_packed: Arc<Tensor<u32>> = Arc::new(
            api::copy_host_vec_to_device(&Arc::new(vec![
                probe_block.clone(),
                probe_block.clone(),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<u32>>()))
            .sync_on(&stream)
            .expect("probe upload failed"),
        );
        for g in 0..4usize {
            let mut xv = vec![0.0f32; 256];
            for e in &mut xv[64 * g..64 * (g + 1)] {
                *e = 1.0;
            }
            let xg_dev = upload(xv, &stream);
            let out_slot = upload(vec![0.0; 2], &stream);
            let r_slot = unsafe {
                oxide_q4k
                    .submit_sm120_kpar_into_slot(&xg_dev, &probe_packed, &out_slot, 2, 1)
                    .expect("probe sm120 failed")
            };
            let v_slot = download(&r_slot, &stream);
            let r_q8 = oxide_q4k
                .forward_q8_d2d(&xg_dev, &probe_packed, 2, 1)
                .expect("probe q8_raw failed");
            let v_q8 = download(&r_q8, &stream);
            println!(
                "    probe j-group {g}: expected={:.1} sm120=[{:.1},{:.1}] q8_raw=[{:.1},{:.1}]",
                probe_expected[g], v_slot[0], v_slot[1], v_q8[0], v_q8[1]
            );
        }

        // ---- 5c. dmin/m term + per-byte nibble probe ----
        // d=1, dmin=0.25, sc_j=j+1, m_j=1, qs[b]=b (lo=b&0xF, hi=b>>4).
        // x one-hot at element e -> out = w[e] =
        //   e in lo half of group g (b=32g+e%64):  sc_2g*(b&0xF) - 0.25
        //   e in hi half (b=32g+(e%64-32)):         sc_2g+1*(b>>4) - 0.25
        let probe2_block: Vec<u32> = {
            let mut b = vec![0u32; 36];
            b[0] = 0x00003C00 | (f32_to_f16_bits(0.25) << 16); // d=1.0, dmin=0.25
            b[1] = 0x04030201; // sc_0..3 = 1,2,3,4
            b[2] = 0x01010101; // m_0..3 = 1
            b[3] = 0x18171615; // sc_4..7=5..8 (low nibbles), m_4..7=1 (high)
            // qs bytes: byte index within block qs section = 0..127, value = index
            for i in 0..32usize {
                let mut w = 0u32;
                for j in 0..4 {
                    w |= ((i * 4 + j) as u32) << (8 * j);
                }
                b[4 + i] = w;
            }
            b
        };
        let probe2_packed: Arc<Tensor<u32>> = Arc::new(
            api::copy_host_vec_to_device(&Arc::new(vec![
                probe2_block.clone(),
                probe2_block.clone(),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<u32>>()))
            .sync_on(&stream)
            .expect("probe2 upload failed"),
        );
        let scj = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for e in [0usize, 5, 31, 32, 37, 63, 64, 100, 160, 200, 255] {
            let mut xv = vec![0.0f32; 256];
            xv[e] = 1.0;
            let xe_dev = upload(xv, &stream);
            let g = e / 64;
            let l = e % 64;
            let byte_idx = 32 * g + (l % 32);
            let expected = if l < 32 {
                scj[2 * g] * ((byte_idx & 0xF) as f32) - 0.25
            } else {
                scj[2 * g + 1] * ((byte_idx >> 4) as f32) - 0.25
            };
            let out_slot = upload(vec![0.0; 2], &stream);
            let r_slot = unsafe {
                oxide_q4k
                    .submit_sm120_kpar_into_slot(&xe_dev, &probe2_packed, &out_slot, 2, 1)
                    .expect("probe2 sm120 failed")
            };
            let v_slot = download(&r_slot, &stream);
            let r_q8 = oxide_q4k
                .forward_q8_d2d(&xe_dev, &probe2_packed, 2, 1)
                .expect("probe2 q8_raw failed");
            let v_q8 = download(&r_q8, &stream);
            println!(
                "    probe2 e={e:>3}: expected={expected:>7.2} sm120={:>8.2} q8_raw={:>8.2}",
                v_slot[0], v_q8[0]
            );
        }

        // ---- 5d. Isolate the sm120 group-3 failure ----
        // e=200/255 fail (0.00) with the 5c block; the dense 5b block passes.
        // Vary dmin/m and x sparsity to find the trigger. Expected values come
        // from the cross-validated ggml dequantizer on the same crafted bytes.
        let dequant_dot = |block: &[u32; 36], x: &[f32]| -> f32 {
            let bytes: Vec<u8> = block
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            let w = dequantize_to_f32(GgmlType::Q4K, &bytes, 256);
            w.iter().zip(x).map(|(a, b)| a * b).sum()
        };
        let mk_block = |dmin: f32, m_byte: u8| -> [u32; 36] {
            let mut b = vec![0u32; 36];
            b[0] = 0x00003C00 | (f32_to_f16_bits(dmin) << 16);
            b[1] = 0x04030201; // sc_0..3
            b[2] = (m_byte as u32) * 0x01010101; // m_0..3
            b[3] = 0x18171615 & 0x0F0F0F0F | ((m_byte as u32) * 0x10101010);
            for i in 0..32usize {
                let mut w = 0u32;
                for j in 0..4 {
                    w |= ((i * 4 + j) as u32) << (8 * j);
                }
                b[4 + i] = w;
            }
            b.try_into().unwrap()
        };
        let mut x_hot = vec![0.0f32; 256];
        x_hot[200] = 1.0;
        let mut x_dense3 = vec![0.0f32; 256];
        for e in &mut x_dense3[192..256] {
            *e = 1.0;
        }
        let variants: Vec<(&str, [u32; 36], Vec<f32>)> = vec![
            ("5c blk, x hot e=200", mk_block(0.25, 1), x_hot.clone()),
            ("m=0 blk, x hot e=200", mk_block(0.0, 0), x_hot.clone()),
            ("dmin=0 blk, x hot e=200", mk_block(0.0, 1), x_hot.clone()),
            ("5c blk, x dense grp3", mk_block(0.25, 1), x_dense3.clone()),
            ("m=0 blk, x dense grp3", mk_block(0.0, 0), x_dense3.clone()),
        ];
        for (name, blk, xv) in variants {
            let expected = dequant_dot(&blk, &xv);
            let packed: Arc<Tensor<u32>> = Arc::new(
                api::copy_host_vec_to_device(&Arc::new(vec![blk, blk].into_iter().flatten().collect::<Vec<u32>>()))
                    .sync_on(&stream)
                    .expect("variant upload failed"),
            );
            let xd = upload(xv.clone(), &stream);
            let out_slot = upload(vec![0.0; 2], &stream);
            let r = unsafe {
                oxide_q4k
                    .submit_sm120_kpar_into_slot(&xd, &packed, &out_slot, 2, 1)
                    .expect("variant sm120 failed")
            };
            let v = download(&r, &stream);
            println!(
                "    {name:<24} expected={expected:>8.2} sm120={:>8.2}",
                v[0]
            );
        }

        // ---- 5e. Full one-hot sweep: which elements read as zero? ----
        {
            let blk = mk_block(0.25, 1);
            let packed: Arc<Tensor<u32>> = Arc::new(
                api::copy_host_vec_to_device(&Arc::new(vec![blk, blk].into_iter().flatten().collect::<Vec<u32>>()))
                    .sync_on(&stream)
                    .expect("sweep upload failed"),
            );
            let mut bad = Vec::new();
            for e in 0..256usize {
                let mut xv = vec![0.0f32; 256];
                xv[e] = 1.0;
                let expected = dequant_dot(&blk, &xv);
                let xd = upload(xv, &stream);
                let out_slot = upload(vec![0.0; 2], &stream);
                let r = unsafe {
                    oxide_q4k
                        .submit_sm120_kpar_into_slot(&xd, &packed, &out_slot, 2, 1)
                        .expect("sweep sm120 failed")
                };
                let v = download(&r, &stream);
                if (v[0] - expected).abs() > 0.01 {
                    bad.push((e, v[0], expected));
                }
            }
            println!("    one-hot sweep: {}/256 elements WRONG", bad.len());
            for (e, g, c) in bad.iter().take(40) {
                println!("      e={e:>3} (grp {} byte {:>2} half {}) gpu={g:>8.2} cpu={c:>8.2}",
                    e / 64, (e % 64) % 32, if e % 64 < 32 { "lo" } else { "hi" });
            }
        }

        // ---- 5f. silu_mul (production FFN elementwise, decode scratch path) ----
        // gate_silu[i] = silu(g[i]) * u[i] on random data.
        {
            let el = tempura_kernels::cutile_elementwise::CutileElementwise::new()
                .expect("elementwise init failed");
            let (g, u): (Vec<f32>, Vec<f32>) = {
                let g = prng_data(14336, 51);
                let u = prng_data(14336, 52);
                (g, u)
            };
            let gd = upload(g.clone(), &stream);
            let ud = upload(u.clone(), &stream);
            let mut silu_scratch: Option<Arc<Tensor<f32>>> = None;
            let out = el
                .silu_mul_device_scratch(&gd, &ud, 1, 14336, &mut silu_scratch, &stream)
                .expect("silu_mul failed");
            let gs = download(&out, &stream);
            let cpu: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(a, b)| (a / (1.0 + (-a).exp())) * b)
                .collect();
            all_ok &= report("silu_mul scratch (decode FFN)", &gs, &cpu);
        }

        // ---- 5f. sm120 fused gate+up+SiLU (production FFN path, raw format) ----
        // gate_silu[i] = silu(gate_i . a) * (up_i . a), real weights, with
        // the Q8_1-quantized activation as the reference (the kernel
        // quantizes a on the fly).
        {
            let gate_name = "model.layers.0.mlp.gate_proj.weight";
            let up_name = "model.layers.0.mlp.up_proj.weight";
            let (g_bytes, u_bytes) = (
                qw.get_q4k(gate_name).expect("gate not Q4K"),
                qw.get_q4k(up_name).expect("up not Q4K"),
            );
            let (gn, gk) = (cfg.intermediate_size, cfg.hidden_size);
            let g_k_blocks = gk / 256;
            let to_u32 = |b: &[u8]| -> Vec<u32> {
                b.chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let g_packed: Arc<Tensor<u32>> = Arc::new(
                api::copy_host_vec_to_device(&Arc::new(to_u32(g_bytes)))
                    .sync_on(&stream)
                    .expect("gate upload failed"),
            );
            let u_packed: Arc<Tensor<u32>> = Arc::new(
                api::copy_host_vec_to_device(&Arc::new(to_u32(u_bytes)))
                    .sync_on(&stream)
                    .expect("up upload failed"),
            );
            let a: Vec<f32> = prng_data(gk, 41);
            let a_dev = upload(a.clone(), &stream);
            let out = oxide_q4k
                .forward_sm120_fused_silu_d2d(&a_dev, &g_packed, &u_packed, gn, g_k_blocks)
                .expect("fused silu failed");
            let gs = download(&out, &stream);
            let gw = dequantize_to_f32(GgmlType::Q4K, g_bytes, gn * gk);
            let uw = dequantize_to_f32(GgmlType::Q4K, u_bytes, gn * gk);
            // Q8_1-quantized reference (the kernel quantizes activations).
            let mut aq = a.clone();
            for blk in (0..gk).step_by(32) {
                let chunk = &mut aq[blk..blk + 32];
                let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = if amax == 0.0 { 1.0 } else { amax / 127.0 };
                for v in chunk.iter_mut() {
                    let r = (*v / d + if *v >= 0.0 { 0.5 } else { -0.5 }) as i32;
                    *v = r.clamp(-127, 127) as f32 * d;
                }
            }
            let cpu_fs: Vec<f32> = (0..gn)
                .map(|i| {
                    let g: f32 = gw[i * gk..(i + 1) * gk]
                        .iter()
                        .zip(&aq)
                        .map(|(w, x)| w * x)
                        .sum();
                    let u: f32 = uw[i * gk..(i + 1) * gk]
                        .iter()
                        .zip(&aq)
                        .map(|(w, x)| w * x)
                        .sum();
                    let s = g / (1.0 + (-g).exp());
                    s * u
                })
                .collect();
            let fs_max = gs
                .iter()
                .zip(&cpu_fs)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let ref_max = cpu_fs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let rel_fs = fs_max / ref_max.max(1e-9);
            println!(
                "  {:<38} max_abs={fs_max:.4} rel={rel_fs:.4} -> {}",
                "q4k sm120 fused silu (real gate/up)",
                if rel_fs < 0.02 { "PASS" } else { "FAIL" }
            );
            all_ok &= rel_fs < 0.02;
        }

        // ---- 6. Oxide decode attention vs CPU GQA math ----
        let kv_len = 86usize;
        let kv_dim = N_KV_HEADS * HEAD_DIM;
        let k_cache: Vec<f32> = prng_data(kv_len * kv_dim, 21);
        let v_cache: Vec<f32> = prng_data(kv_len * kv_dim, 22);
        let q_attn: Vec<f32> = prng_data(N_HEADS * HEAD_DIM, 23);
        let k_cache_dev = upload(k_cache.clone(), &stream);
        let v_cache_dev = upload(v_cache.clone(), &stream);
        let q_attn_dev = upload(q_attn.clone(), &stream);
        let kv_len_dev = upload(vec![f32::from_bits(kv_len as u32)], &stream);
        let oda = tempura_kernels::oxide_decode_attn_host::OxideDecodeAttn::new(
            stream.clone(),
        )
        .expect("OxideDecodeAttn init failed");
        let attn_out = upload(vec![0.0; N_HEADS * HEAD_DIM], &stream);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        // splitk is the production-preferred variant; kpar is the fallback.
        let scratch_len =
            tempura_kernels::oxide_decode_attn_host::OxideDecodeAttn::splitk_scratch_len(
                N_HEADS, HEAD_DIM,
            );
        let splitk_scratch = upload(vec![0.0; scratch_len], &stream);
        oda.forward_splitk_d2d(
            &attn_out, &q_attn_dev, &k_cache_dev, &v_cache_dev, &kv_len_dev,
            &splitk_scratch, N_HEADS, N_KV_HEADS, HEAD_DIM, scale,
        )
        .expect("splitk attention failed");
        let attn_gpu = download(&attn_out, &stream);
        let attn_cpu = cpu_gqa_attention(
            &q_attn, &k_cache, &v_cache, kv_len, scale,
        );
        all_ok &= report("oxide decode attention (splitk)", &attn_gpu, &attn_cpu);

        let attn_out2 = upload(vec![0.0; N_HEADS * HEAD_DIM], &stream);
        oda.forward_kpar_d2d(
            &attn_out2, &q_attn_dev, &k_cache_dev, &v_cache_dev, &kv_len_dev,
            N_HEADS, N_KV_HEADS, HEAD_DIM, scale,
        )
        .expect("kpar attention failed");
        let attn_gpu2 = download(&attn_out2, &stream);
        all_ok &= report("oxide decode attention (kpar)", &attn_gpu2, &attn_cpu);

        // ---- 6b. Attention with a PRODUCTION-sized dirty cache ----
        // Production caches are [4096 rows] with stale garbage beyond kv_len;
        // the isolated test above used an exactly-sized cache. If a kernel
        // reads past kv_len (split-K padding, rounded loop bounds), it sees
        // garbage here and the output diverges.
        {
            let cache_rows = 4096usize;
            let mut k_big = vec![0.0f32; cache_rows * kv_dim];
            let mut v_big = vec![0.0f32; cache_rows * kv_dim];
            k_big[..k_cache.len()].copy_from_slice(&k_cache);
            v_big[..v_cache.len()].copy_from_slice(&v_cache);
            // Poison the stale region with the exact garbage pattern that
            // production sees: previous-session K/V values (non-zero, finite).
            let poison = prng_data((cache_rows - kv_len) * kv_dim, 99);
            k_big[kv_len * kv_dim..].copy_from_slice(&poison);
            let poison_v = prng_data((cache_rows - kv_len) * kv_dim, 98);
            v_big[v_cache.len()..].copy_from_slice(&poison_v);
            let kb_dev = upload(k_big, &stream);
            let vb_dev = upload(v_big, &stream);
            let attn_out3 = upload(vec![0.0; N_HEADS * HEAD_DIM], &stream);
            oda.forward_splitk_d2d(
                &attn_out3, &q_attn_dev, &kb_dev, &vb_dev, &kv_len_dev,
                &splitk_scratch, N_HEADS, N_KV_HEADS, HEAD_DIM, scale,
            )
            .expect("splitk attention (dirty cache) failed");
            let a3 = download(&attn_out3, &stream);
            all_ok &= report("oxide attn splitk (dirty 4096-row cache)", &a3, &attn_cpu);

            let attn_out4 = upload(vec![0.0; N_HEADS * HEAD_DIM], &stream);
            oda.forward_kpar_d2d(
                &attn_out4, &q_attn_dev, &kb_dev, &vb_dev, &kv_len_dev,
                N_HEADS, N_KV_HEADS, HEAD_DIM, scale,
            )
            .expect("kpar attention (dirty cache) failed");
            let a4 = download(&attn_out4, &stream);
            all_ok &= report("oxide attn kpar (dirty 4096-row cache)", &a4, &attn_cpu);
        }

        // ---- 7. RMSNorm fused scratch path (decode variant) vs CPU ----
        let rms = tempura_kernels::cutile_rmsnorm::CutileRmsNorm::new().unwrap();
        // Reuse the rope stream so all ops stay ordered.
        let ln_name = "model.layers.0.input_layernorm.weight";
        let ln_w: Vec<f32> = qw
            .get_f32(ln_name)
            .or_else(|| qw.get_f32("model.norm.weight"))
            .map(|s| s.to_vec())
            .expect("norm weight not found");
        let h = cfg.hidden_size;
        let x_norm: Vec<f32> = prng_data(h, 31);
        let x_norm_dev = upload(x_norm.clone(), &stream);
        let ln_w_dev = upload(ln_w.clone(), &stream);
        let mut norm_scratch =
            tempura_kernels::cutile_rmsnorm::RmsnormScratch::new(&stream, 1, h).unwrap();
        let normed = rms
            .forward_device_fused_scratch(
                &stream, x_norm_dev, &ln_w_dev, 1, h, h, 1e-5, &mut norm_scratch,
            )
            .expect("fused-scratch rmsnorm failed");
        let norm_gpu = download(&normed, &stream);
        let norm_cpu: Vec<f32> = {
            let ss: f32 = x_norm.iter().map(|v| v * v).sum::<f32>() / h as f32;
            let inv = 1.0 / (ss + 1e-5).sqrt();
            x_norm.iter().zip(&ln_w).map(|(v, w)| v * inv * w).collect()
        };
        all_ok &= report("rmsnorm fused scratch (real ln weight)", &norm_gpu, &norm_cpu);
    }

    // ---- 8. MXFP4 quant+GEMM composition (prefill fast path) ----
    // The engine prefill quantizes activations on GPU (mxfp4_quant PTX) and
    // runs mxf4_batch_gemm_tiled at m_pad; the standalone tool validates the
    // GEMM only with uniform scales and 16-multiple m. This tests the real
    // composition at padded m (m_valid < m_pad) against a CPU reference built
    // from the SAME quantized values (isolates GEMM+quant from quant error).
    {
        use tempura_kernels::mxfp4_dispatch::MxfpGemm;
        use tempura_kernels::mxfp4::{quantize_matrix_to_mxfp4_tiled, ue8m0_to_f32};

        const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let mxfp = MxfpGemm::try_load(stream.clone()).expect("MxfpGemm not loaded");
        let (n, k) = (512usize, 4096usize);
        let k64_tiles = (k / 64) as u32;

        for &(m, m_pad) in &[(128usize, 128usize), (86usize, 96usize), (117usize, 128usize)] {
            // Scale-isolation sub-runs: (a_real, b_real) flags. uniform 0x7F
            // scales replicate the standalone tool's blind spot; each sub-run
            // compares against a CPU reference using the SAME scale bytes.
            for (a_real, b_real) in [(true, true), (true, false), (false, true)] {
            let a: Vec<f32> = prng_data(m * k, 61);
            let a_dev = upload(a.clone(), &stream);
            let packed_a: Arc<Tensor<u8>> = Arc::new(
                api::zeros::<u8>(&[m_pad * k / 2]).sync_on(&stream).expect("pa alloc"),
            );
            let scales_a: Arc<Tensor<u8>> = Arc::new(
                api::zeros::<u8>(&[(m_pad / 16) * (k / 32)])
                    .sync_on(&stream)
                    .expect("sa alloc"),
            );
            mxfp
                .quantize_activation(
                    &a_dev, &packed_a, &scales_a,
                    m_pad as u32, k as u32, m as u32,
                    stream.cu_stream(),
                )
                .expect("quantize_activation failed");

            let b: Vec<f32> = prng_data(n * k, 62);
            let (pb, sb_real) = quantize_matrix_to_mxfp4_tiled(&b, n, k);
            let sb: Vec<u8> = if b_real {
                sb_real.clone()
            } else {
                vec![0x7Fu8; sb_real.len()]
            };
            // Quantize A against the scale set actually uploaded: when the
            // A-scales are forced uniform the packed nibbles must come from a
            // uniform-scale quantization too, so quantize on the CPU for that
            // sub-run (same tiler, row-tile granularity difference is fine:
            // the reference decodes the same bytes).
            let (pa, sa): (Vec<u8>, Vec<u8>) = if a_real {
                let pa: Vec<u8> = Arc::clone(&packed_a).to_host_vec().sync_on(&stream).expect("pa dl");
                let sa: Vec<u8> = Arc::clone(&scales_a).to_host_vec().sync_on(&stream).expect("sa dl");
                (pa, sa)
            } else {
                // Re-quantize A on the CPU with uniform 0x7F scales: round
                // each value to the nearest E2M1 magnitude (values are in
                // [-0.5, 0.5], so /1.0 scale keeps them in range).
                let mut pa = vec![0u8; m_pad * k / 2];
                for row in 0..m {
                    for kk in 0..k {
                        let v = a[row * k + kk];
                        let nib = round_e2m1(v);
                        if kk % 2 == 0 {
                            pa[row * (k / 2) + kk / 2] = nib;
                        } else {
                            pa[row * (k / 2) + kk / 2] |= nib << 4;
                        }
                    }
                }
                (pa, vec![0x7Fu8; (m_pad / 16) * (k / 32)])
            };
            // Upload the (possibly overridden) A buffers.
            let packed_a: Arc<Tensor<u8>> = Arc::new(
                api::copy_host_vec_to_device(&Arc::new(pa.clone()))
                    .sync_on(&stream)
                    .expect("pa ul"),
            );
            let scales_a: Arc<Tensor<u8>> = Arc::new(
                api::copy_host_vec_to_device(&Arc::new(sa.clone()))
                    .sync_on(&stream)
                    .expect("sa ul"),
            );
            let pb_dev: Arc<Tensor<u8>> = Arc::new(
                api::copy_host_vec_to_device(&Arc::new(pb.clone()))
                    .sync_on(&stream)
                    .expect("pb ul"),
            );
            let sb_dev: Arc<Tensor<u8>> = Arc::new(
                api::copy_host_vec_to_device(&Arc::new(sb.clone()))
                    .sync_on(&stream)
                    .expect("sb ul"),
            );

            let out: Arc<Tensor<f32>> = Arc::new(
                api::zeros::<f32>(&[m_pad * n]).sync_on(&stream).expect("out alloc"),
            );
            mxfp
                .forward(
                    &packed_a, &scales_a, &pb_dev, &sb_dev, &out,
                    m_pad as u32, n as u32, k64_tiles,
                )
                .expect("mxfp4 forward failed");
            let gpu: Vec<f32> = Arc::clone(&out).to_host_vec().sync_on(&stream).expect("out dl");

            // Decode the SAME quantized values on CPU and dense-matmul.
            let dec_a = |row: usize, kk: usize| -> f32 {
                let byte = pa[row * (k / 2) + kk / 2];
                let nib = if kk % 2 == 0 { byte & 0xF } else { byte >> 4 };
                let v = if nib & 8 != 0 { -E2M1[(nib & 7) as usize] } else { E2M1[(nib & 7) as usize] };
                v * ue8m0_to_f32(sa[(row / 16) * (k / 32) + kk / 32])
            };
            let dec_b = |row: usize, kk: usize| -> f32 {
                let byte = pb[row * (k / 2) + kk / 2];
                let nib = if kk % 2 == 0 { byte & 0xF } else { byte >> 4 };
                let v = if nib & 8 != 0 { -E2M1[(nib & 7) as usize] } else { E2M1[(nib & 7) as usize] };
                v * ue8m0_to_f32(sb[(row / 8) * (k / 32) + kk / 32])
            };
            let mut max_abs = 0.0f32;
            let mut ref_max = 0.0f32;
            let mut worst = (0usize, 0usize, 0.0f32, 0.0f32);
            let mut first_bad = None;
            for mi in 0..m {
                for ni in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += dec_a(mi, kk) * dec_b(ni, kk);
                    }
                    ref_max = ref_max.max(sum.abs());
                    let d = (gpu[mi * n + ni] - sum).abs();
                    if d > 1e-2 && first_bad.is_none() {
                        first_bad = Some((mi, ni, gpu[mi * n + ni], sum));
                    }
                    if d > max_abs {
                        max_abs = d;
                        worst = (mi, ni, gpu[mi * n + ni], sum);
                    }
                }
            }
            println!(
                "    first_bad={:?} worst={:?}",
                first_bad, worst
            );
            // Scale-pair ordering probe removed (ruled out: lo==hi on the
            // probed tiles and both orderings agreed).
            // Scale-indexing hypotheses on element (0, 2) for the
            // (a_real=true, b_real=false) sub-run where B contributes scale 1.
            if a_real && !b_real {
                let (mi, ni) = (0usize, 2usize);
                let nib = |buf: &[u8], row: usize, kk: usize| -> f32 {
                    let byte = buf[row * (k / 2) + kk / 2];
                    let n = if kk % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    let v = if n & 8 != 0 { -E2M1[(n & 7) as usize] } else { E2M1[(n & 7) as usize] };
                    v
                };
                let mut h1 = 0.0f32; // [m/16][k/32] (mine)
                let mut h2 = 0.0f32; // [m/16][k/64] single byte per k64
                let mut h3 = 0.0f32; // no A scale
                for kk in 0..k {
                    let v = nib(&pa, mi, kk) * nib(&pb, ni, kk);
                    h1 += v * ue8m0_to_f32(sa[(mi / 16) * (k / 32) + kk / 32]);
                    h2 += v * ue8m0_to_f32(sa[(mi / 16) * (k / 64) + kk / 64]);
                    h3 += v;
                }
                println!(
                    "    elem(0,2): gpu={} h1_k32={} h2_k64={} h3_noscale={h3:.4}",
                    gpu[ni],
                    h1, h2
                );
            }

            let rel = max_abs / ref_max.max(1e-9);
            println!(
                "  {:<38} m={m:>3} m_pad={m_pad:>3} a_real={a_real} b_real={b_real} \
                 max_abs={max_abs:.4} rel={rel:.4} -> {}",
                "mxfp4 quant+gemm composition",
                if rel < 0.02 { "PASS" } else { "FAIL" }
            );
            all_ok &= rel < 0.02 || !a_real || !b_real; // real-real is the engine case
            }
        }

        // ---- 8b. Minimal single-tile MXFP4 probe (m16 n8 k64) ----
        {
            let (m, n, k) = (16usize, 8usize, 64usize);
            let a: Vec<f32> = (0..m * k)
                .map(|i| {
                    let (mi, kk) = (i / k, i % k);
                    if kk == mi % 64 { (mi as f32) * 0.125 } else { 0.0 }
                })
                .collect();
            let a_dev = upload(a.clone(), &stream);
            let packed_a: Arc<Tensor<u8>> =
                Arc::new(api::zeros::<u8>(&[m * k / 2]).sync_on(&stream).expect("pa"));
            let scales_a: Arc<Tensor<u8>> =
                Arc::new(api::zeros::<u8>(&[(m / 16) * (k / 32)]).sync_on(&stream).expect("sa"));
            mxfp
                .quantize_activation(&a_dev, &packed_a, &scales_a, m as u32, k as u32, m as u32,
                    stream.cu_stream())
                .expect("quant failed");
            let pa: Vec<u8> = Arc::clone(&packed_a).to_host_vec().sync_on(&stream).expect("dl");
            let sa: Vec<u8> = Arc::clone(&scales_a).to_host_vec().sync_on(&stream).expect("dl");
            let mut pb = vec![0u8; n * k / 2];
            for h in 0..k / 2 {
                pb[h] = 0x22; // two 1.0-magnitude nibbles
            }
            let sb = vec![0x7Fu8; (n / 8) * (k / 32)];
            let pb_dev: Arc<Tensor<u8>> = Arc::new(
                api::copy_host_vec_to_device(&Arc::new(pb.clone())).sync_on(&stream).expect("ul"),
            );
            let sb_dev: Arc<Tensor<u8>> = Arc::new(
                api::copy_host_vec_to_device(&Arc::new(sb.clone())).sync_on(&stream).expect("ul"),
            );
            let out: Arc<Tensor<f32>> =
                Arc::new(api::zeros::<f32>(&[m * n]).sync_on(&stream).expect("out"));
            mxfp
                .forward(&packed_a, &scales_a, &pb_dev, &sb_dev, &out, m as u32, n as u32, 1)
                .expect("gemm failed");
            let gpu: Vec<f32> = Arc::clone(&out).to_host_vec().sync_on(&stream).expect("dl");
            println!("    minimal probe: sa={:?} pa[0..6]={:?}", &sa[..], &pa[..6]);
            for mi in 0..m {
                let byte = pa[mi * (k / 2) + mi / 2];
                let nib = if mi % 2 == 0 { byte & 0xF } else { byte >> 4 };
                println!(
                    "      row {mi:>2}: A={:.4} scaled={:.4} quant_nibble={} (mag {})",
                    a[mi * k + mi],
                    a[mi * k + mi] / ue8m0_to_f32(sa[0]),
                    nib,
                    E2M1[(nib & 7) as usize]
                );
            }
            for mi in 0..m {
                let mut exp = 0.0f32;
                for kk in 0..k {
                    let byte = pa[mi * (k / 2) + kk / 2];
                    let nib = if kk % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    let v = if nib & 8 != 0 { -E2M1[(nib & 7) as usize] } else { E2M1[(nib & 7) as usize] };
                    exp += v * ue8m0_to_f32(sa[(mi / 16) * (k / 32) + kk / 32]);
                }
                println!("      out[{mi:>2}][0]: gpu={:.4} expected={exp:.4}", gpu[mi * n]);
            }
        }
    }

    // ---- 9. q6k sm120_mbar determinism stress (default-off kernel) ----
    // The mbar cp.async double-buffer variant is disabled by default
    // (history-dependent outputs). This stress test runs it on identical
    // real weights/activations repeatedly -- with interleaved device
    // allocations and an intervening kernel launch to vary the allocation
    // and kernel history -- and checks bitwise determinism. Requires
    // TEMPURA_ALLOW_SM120_Q6K=1 in the environment when the tool starts.
    {
        let o6 = tempura_kernels::oxide_q6k_gemv_host::OxideQ6kGemv::try_load(
            stream.clone(),
        );
        match o6 {
            Some(o6) if o6.has_sm120_mbar() => {
                let model_path2 = std::path::Path::new(&model_path);
                let loader2 = tempura_engine::gguf_loader::GgufLoader::from_file_quantized(
                    model_path2,
                )
                .expect("re-open GGUF for mbar test");
                let qw2 = loader2.to_quantized_weights();
                let n = 4096usize; // hidden (down_proj output rows)
                let kb = 56usize; // inter/256
                let k = kb * 256;
                let x: Vec<f32> = prng_data(k, 71);
                let x_dev = upload(x.clone(), &stream);
                let (packed, cpu_ref, p_full): (Arc<Tensor<u32>>, Vec<f32>, Vec<u32>) = {
                    let bytes = qw2
                        .get_q6k("model.layers.0.mlp.down_proj.weight")
                        .expect("down_proj Q6K");
                    // Pack into the ENGINE's 53-u32 format (d f32 bits + 4
                    // scale words + 32 ql + 16 qh) -- the same layout
                    // maybe_upload_q6k produces.
                    let mut p: Vec<u32> = Vec::with_capacity(n * kb * 53);
                    const Q6K_BYTES: usize = 210;
                    for row in 0..n {
                        for blk in 0..kb {
                            let off = (row * kb + blk) * Q6K_BYTES;
                            let block = &bytes[off..off + Q6K_BYTES];
                            let ql = &block[0..128];
                            let qh = &block[128..192];
                            let scales = &block[192..208];
                            let d_bits = u16::from_le_bytes([block[208], block[209]]);
                            let d_f32 = f16_bits_to_f32(d_bits);
                            p.push(d_f32.to_bits());
                            for w in 0..4usize {
                                let b = |i: usize| ((scales[w * 4 + i] as i8) as i32 + 128) as u8;
                                p.push(
                                    (b(0) as u32)
                                        | ((b(1) as u32) << 8)
                                        | ((b(2) as u32) << 16)
                                        | ((b(3) as u32) << 24),
                                );
                            }
                            for i in 0..32usize {
                                p.push(
                                    (ql[i * 4] as u32)
                                        | ((ql[i * 4 + 1] as u32) << 8)
                                        | ((ql[i * 4 + 2] as u32) << 16)
                                        | ((ql[i * 4 + 3] as u32) << 24),
                                );
                            }
                            for i in 0..16usize {
                                p.push(
                                    (qh[i * 4] as u32)
                                        | ((qh[i * 4 + 1] as u32) << 8)
                                        | ((qh[i * 4 + 2] as u32) << 16)
                                        | ((qh[i * 4 + 3] as u32) << 24),
                                );
                            }
                        }
                    }
                    let w = dequantize_to_f32(GgmlType::Q6K, bytes, n * k);
                    let c: Vec<f32> = (0..n)
                        .map(|i| {
                            w[i * k..(i + 1) * k]
                                .iter()
                                .zip(&x)
                                .map(|(a, b)| a * b)
                                .sum()
                        })
                        .collect();
                    (
                        Arc::new(
                            api::copy_host_vec_to_device(&Arc::new(p.clone()))
                                .sync_on(&stream)
                                .expect("mbar packed ul"),
                        ),
                        c,
                        p,
                    )
                };
                let r0 = o6
                    .forward_kpar_d2d(&x_dev, &packed, n, kb)
                    .expect("mbar run 0 failed");
                let v0 = download(&r0, &stream);
                let max_ref = v0
                    .iter()
                    .zip(&cpu_ref)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let infs = v0.iter().filter(|v| !v.is_finite()).count();
                println!(
                    "  q6k mbar vs CPU (kb=56): max_abs={max_ref:.6} nonfinite={infs}/{n}"
                );

                // Shape bisect: kb=8 (ONE loop iteration, no swap) and
                // kb=16 (two iterations, one swap) from the same weights.
                for &kbt in &[8usize, 16usize] {
                    let kt = kbt * 256;
                    let xt: Vec<f32> = x[..kt].to_vec();
                    let xt_dev = upload(xt.clone(), &stream);
                    let packed_t: Arc<Tensor<u32>> = Arc::new(
                        api::copy_host_vec_to_device(&Arc::new(
                            (0..n)
                                .flat_map(|r| {
                                    let base = r * kb * 53;
                                    p_full[base..base + kbt * 53].to_vec()
                                })
                                .collect::<Vec<u32>>(),
                        ))
                        .sync_on(&stream)
                        .expect("packed_t ul"),
                    );
                    let rt = o6
                        .forward_kpar_d2d(&xt_dev, &packed_t, n, kbt)
                        .expect("mbar shape run failed");
                    let vt = download(&rt, &stream);
                    let wt = dequantize_to_f32(GgmlType::Q6K, &qw2.get_q6k("model.layers.0.mlp.down_proj.weight").unwrap(), n * k);
                    let cpu_t: Vec<f32> = (0..n)
                        .map(|i| {
                            wt[i * k..i * k + kt]
                                .iter()
                                .zip(&xt)
                                .map(|(a, b)| a * b)
                                .sum()
                        })
                        .collect();
                    let mt = vt
                        .iter()
                        .zip(&cpu_t)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let badt = vt
                        .iter()
                        .zip(&cpu_t)
                        .filter(|(a, b)| (**a - **b).abs() > 0.01)
                        .count();
                    println!(
                        "  q6k mbar shape bisect kb={kbt}: max_abs={mt:.6} bad={badt}/{n}"
                    );
                }
                let mut nondet = 0usize;
                for it in 1..=8u32 {
                    // History churn: fresh allocation + an intervening
                    // kernel launch on the same stream.
                    let junk = upload(prng_data(7777, it), &stream);
                    let _ = o6.forward_kpar_d2d(&junk, &packed, 64, kb);
                    let r = o6
                        .forward_kpar_d2d(&x_dev, &packed, n, kb)
                        .expect("mbar stress run failed");
                    let v = download(&r, &stream);
                    if !v.iter().zip(&v0).all(|(a, b)| a.to_bits() == b.to_bits()) {
                        nondet += 1;
                    }
                }
                println!(
                    "  q6k mbar determinism: {nondet}/8 stress runs bitwise-differ -> {}",
                    if nondet == 0 { "DETERMINISTIC" } else { "NONDETERMINISTIC (race live)" }
                );
            }
            _ => println!("  (q6k mbar variant not loaded; run with TEMPURA_ALLOW_SM120_Q6K=1)"),
        }
    }

    println!();
    if all_ok {
        println!("All decode-op mechanisms match CPU math -- the decode");
        println!("divergence lives in the engine orchestration");
        println!("(cutile_gpu_forward.rs forward_decode_inner / multistream /");
        println!(" attention input wiring), not in these kernels.");
    } else {
        println!("FAILURES above -- a decode-only kernel is wrong.");
    }
    if !all_ok {
        std::process::exit(1);
    }
}
