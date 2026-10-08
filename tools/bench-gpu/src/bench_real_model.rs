// bench-real-model — benchmark a real GGUF model end-to-end on GPU
//
// Measured on RTX 5090 (2026-09-18) with bitnet-2b-4t-tl2.gguf:
//   GGUF parse:   ~3.0s
//   GPU upload:   ~5.0s  (f32 dequant + cuTile weight upload for 30 layers)
//   Forward pass: NOT MEASURABLE — h=2560 is non-power-of-2; cuTile JIT
//                 requires power-of-2 tile dimensions (error: sm_120 tileiras).
//                 Use a model with power-of-2 hidden size for decode/prefill timing.
//   llama.cpp RTX 5090 reference: 209 tok/s decode, 7335 tok/s pp128
//
// Run:
//   cargo run -p tempura-bench-gpu --release --bin bench-real-model
//   cargo run -p tempura-bench-gpu --release --bin bench-real-model -- /path/to/model.gguf

use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::time::Instant;
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::{CutileInferenceWrapper, InferenceModel};
use tempura_engine::session::InferenceSession;
use tempura_engine::trace::TraceLevel;

const WARMUP_DECODE: usize = 20;
const BENCH_DECODE: usize = 50;
const PREFILL_LEN: usize = 128;

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    let idx = ((sorted.len() as f64 * pct / 100.0).ceil() as usize).saturating_sub(1);
    sorted[idx.min(sorted.len() - 1)]
}

/// Run a forward pass, catching any panic (e.g. cuTile JIT compile failure).
/// Returns None if the forward panicked.
fn try_forward(
    wrapper: &CutileInferenceWrapper,
    session: &mut InferenceSession,
    tokens: &[u32],
) -> Option<Vec<f32>> {
    std::panic::catch_unwind(AssertUnwindSafe(|| wrapper.forward(session, tokens))).ok()
}

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/fdupont/.cache/tempura/models/bitnet-2b-4t-tl2.gguf".to_string());

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  tempura cuTile Real-Model Benchmark                   ║");
    println!("╚══════════════════════════════════════════════════════╝\n");
    println!("  Model: {model_path}");
    println!("  Warmup decode: {WARMUP_DECODE} steps, Bench decode: {BENCH_DECODE} steps");
    println!("  Prefill length: {PREFILL_LEN} tokens\n");

    // --- Load GGUF ---
    println!("Loading GGUF (quantized path)…");
    let t0 = Instant::now();
    let loader =
        GgufLoader::from_file_quantized(Path::new(&model_path)).expect("failed to open GGUF file");
    let config = loader.config.clone();
    println!("  Parsed in {:.2}s", t0.elapsed().as_secs_f64());
    println!(
        "  Config: {} layers, h={}, heads={}/{}, d={}, vocab={}",
        config.num_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.vocab_size,
    );

    // --- Build model and load weights ---
    // Use load_quantized so that mmap-resident Q4K/Q6K tensors are wired
    // directly into QuantizedWeights without going through the WeightLoader
    // trait (which only sees pre-dequantized f32 tensors).
    println!("\nUploading weights to GPU…");
    let t1 = Instant::now();
    let qw = loader.to_quantized_weights();
    let mut wrapper = CutileInferenceWrapper::new(config.clone());
    wrapper.load_quantized(qw);
    let load_elapsed = t1.elapsed().as_secs_f64();
    let parse_elapsed = t0.elapsed().as_secs_f64() - load_elapsed;
    println!("  Loaded in {:.2}s", load_elapsed);

    // Probe: single forward to check if cuTile JIT succeeds for this model shape.
    println!("\nProbing forward pass compatibility…");
    let probe_tokens = vec![1u32];
    let mut probe_session = InferenceSession::new(&config, TraceLevel::Off, 0);
    let probe_result = try_forward(&wrapper, &mut probe_session, &probe_tokens);
    if probe_result.is_none() {
        println!("  WARNING: forward pass failed (likely cuTile JIT requires power-of-2 tile");
        println!(
            "  dimensions but h={} is not power-of-2).",
            config.hidden_size
        );
        println!("  Timing data unavailable for this model shape.");
        println!("\n─── Weight Load Timing ───");
        println!("  Parse time:  {parse_elapsed:.2}s");
        println!("  GPU upload:  {load_elapsed:.2}s");
        println!("  Total:       {:.2}s", parse_elapsed + load_elapsed);
        println!("\n  llama.cpp RTX 5090 reference: 209 tok/s decode, 7335 tok/s pp128");
        println!("\nDone (forward pass unsupported for this model shape).");
        return;
    }
    println!("  OK — forward pass works.");

    // --- Prefill benchmark ---
    println!("\n─── Prefill ({PREFILL_LEN} tokens) ───");
    let prompt: Vec<u32> = (0..PREFILL_LEN as u32)
        .map(|i| 1 + (i % (config.vocab_size as u32 - 1)))
        .collect();

    // Warmup prefill (lets JIT settle for all kernel variants)
    for _ in 0..5 {
        let mut session = InferenceSession::new(&config, TraceLevel::Off, 0);
        try_forward(&wrapper, &mut session, &prompt);
    }

    let mut prefill_times = Vec::with_capacity(20);
    for _ in 0..20 {
        let mut session = InferenceSession::new(&config, TraceLevel::Off, 0);
        let start = Instant::now();
        try_forward(&wrapper, &mut session, &prompt);
        prefill_times.push(start.elapsed().as_secs_f64() * 1e6); // µs
    }

    prefill_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_prefill = percentile(&prefill_times, 50.0);
    let p95_prefill = percentile(&prefill_times, 95.0);
    let avg_prefill = prefill_times.iter().sum::<f64>() / prefill_times.len() as f64;
    let pp_tok_s = PREFILL_LEN as f64 / (avg_prefill / 1e6);

    println!("  TTFT P50:        {:.2} ms", p50_prefill / 1000.0);
    println!("  TTFT P95:        {:.2} ms", p95_prefill / 1000.0);
    println!("  Prefill tok/s:   {pp_tok_s:.0}");
    println!("  llama.cpp ref:   7335 tok/s pp128 (RTX 5090)");

    // --- Decode benchmark ---
    println!("\n─── Decode (single-token autoregressive) ───");
    let mut session = InferenceSession::new(&config, TraceLevel::Off, 0);
    try_forward(&wrapper, &mut session, &prompt);

    let decode_token = [1u32];

    // Warmup decode
    println!("  Warming up {WARMUP_DECODE} decode steps…");
    for _ in 0..WARMUP_DECODE {
        try_forward(&wrapper, &mut session, &decode_token);
    }

    // Measure
    let mut decode_times: Vec<f64> = Vec::with_capacity(BENCH_DECODE);
    for _ in 0..BENCH_DECODE {
        let start = Instant::now();
        try_forward(&wrapper, &mut session, &decode_token);
        decode_times.push(start.elapsed().as_secs_f64() * 1e6); // µs
    }

    decode_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_itl = percentile(&decode_times, 50.0);
    let p95_itl = percentile(&decode_times, 95.0);
    let avg_itl = decode_times.iter().sum::<f64>() / decode_times.len() as f64;
    let decode_tok_s = 1e6 / avg_itl;

    println!("  ITL P50:         {:.2} ms", p50_itl / 1000.0);
    println!("  ITL P95:         {:.2} ms", p95_itl / 1000.0);
    println!("  Decode tok/s:    {decode_tok_s:.0}");
    println!("  llama.cpp ref:   209 tok/s decode (RTX 5090)");

    println!("\n─── Summary ───");
    println!(
        "  Prefill:  {pp_tok_s:.0} tok/s  (target: 7335 tok/s, gap: {:.1}×)",
        7335.0 / pp_tok_s.max(1.0)
    );
    println!(
        "  Decode:   {decode_tok_s:.0} tok/s  (target: 209 tok/s,  gap: {:.1}×)",
        209.0 / decode_tok_s.max(1.0)
    );
    println!("\nDone.");
}
