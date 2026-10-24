/// bench-mxfp4 -- measure MXFP4 tiled batch GEMM throughput vs Q4K sm120 stream-K
///
/// Loads a Q4_K_M GGUF model, quantizes projection weights to MXFP4 tiled format,
/// then benchmarks 7 GEMMs per layer (Q/K/V/O/gate/up/down projections) at various
/// batch sizes using the sm_120a mxf4_batch_gemm_tiled kernel.
///
/// Note: activation quantization (f32 -> MXFP4) is done once up-front with random
/// data; this measures raw GEMM throughput, not end-to-end tok/s with activation
/// quantization overhead.
///
/// Usage:
///   RUSTUP_TOOLCHAIN=nightly-2026-10-11 cargo run \
///     --manifest-path tools/bench-gpu/Cargo.toml --release \
///     --bin bench-mxfp4 -- /path/to/model.gguf

use std::path::Path;
use std::time::Instant;

use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::CutileInferenceWrapper;

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            let candidates = [
                "/data/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "/home/fdupont/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            ];
            candidates
                .iter()
                .find(|p| Path::new(p).exists())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    eprintln!("Usage: bench-mxfp4 <model.gguf>");
                    std::process::exit(1);
                })
        });

    eprintln!("Loading model from {model_path}");
    let loader = GgufLoader::from_file_quantized(Path::new(&model_path))
        .expect("GgufLoader failed");
    let config = loader.config.clone();
    let qw = loader.to_quantized_weights();
    let mut wrapper = CutileInferenceWrapper::new(config);
    wrapper.load_quantized(qw);

    if !wrapper.has_mxfp4() {
        eprintln!("MXFP4 not available: requires sm_120a (RTX 5090) + Q4K model");
        eprintln!("(Check that mxf4_gemv.cubin is present and compiled for sm_120a)");
        std::process::exit(1);
    }

    let cfg = &wrapper.model.config;
    eprintln!(
        "Model: hidden={} inter={} layers={} heads={}/kv_heads={}",
        cfg.hidden_size, cfg.intermediate_size, cfg.num_layers,
        cfg.num_attention_heads, cfg.num_kv_heads,
    );
    eprintln!("MXFP4 ready: {} tiled weights loaded", wrapper.mxfp4_weight_count());

    // Benchmark at batch sizes that are multiples of 16 (m16-tile requirement).
    for &batch in &[16usize, 32, 48, 64] {
        let result = wrapper.bench_mxfp4_one_layer(0, batch);
        match result {
            Some(dur) => {
                let ms = dur.as_secs_f64() * 1000.0;
                // Total flops for 7 GEMMs, one layer:
                // Q [batch, h, q_dim] + K [batch, h, kv_dim] + V [batch, h, kv_dim]
                // + O [batch, q_dim, h] + gate [batch, h, inter] + up [batch, h, inter]
                // + down [batch, inter, h]
                let h = cfg.hidden_size;
                let inter = cfg.intermediate_size;
                let q_dim = cfg.num_attention_heads * cfg.head_dim;
                let kv_dim = cfg.num_kv_heads * cfg.head_dim;
                let total_flops = 2 * batch * (
                    h * q_dim   // Q
                    + h * kv_dim // K
                    + h * kv_dim // V
                    + q_dim * h  // O
                    + h * inter  // gate
                    + h * inter  // up
                    + inter * h  // down
                );
                let tflops = total_flops as f64 / (dur.as_secs_f64() * 1e12);
                println!("M={batch:3}: {ms:.3} ms/layer  {tflops:.2} TFLOPS  MXFP4");
            }
            None => {
                println!("M={batch:3}: MXFP4 not available for this layer");
            }
        }
    }
}
