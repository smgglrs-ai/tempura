// spec-probe -- real-model gate for sampled (stochastic) speculative
// decoding on the CPU SpeculativeDecoder path:
//   1. DETERMINISM: two runs with the same seed must emit identical
//      tokens (the determinism contract).
//   2. LIKELIHOOD: the mean log p_target(emitted token) of spec-decoded
//      tokens must agree with plain sampling (the distributional
//      equivalence -- exact acceptance math is unit-gated in
//      engine/src/speculative.rs; this checks the real-model plumbing).
//   3. Acceptance telemetry from the trace.
//
// Run:
//   cargo run -p tempura-bench-gpu --release --bin spec-probe -- \
//     /path/to/target.gguf /path/to/draft.gguf
// (target and draft must share the tokenizer/vocab, e.g. Qwen3-0.6B +
// Qwen2.5-0.5B.)

use std::path::Path;
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::TransformerModel;
use tempura_engine::pipeline::KernelPipeline;
use tempura_engine::session::InferenceSession;
use tempura_engine::speculative::SpeculativeDecoder;
use tempura_engine::tokenizer::Tokenizer;
use tempura_engine::trace::TraceLevel;
use tempura_kernels::SamplingParams;

const PROMPT: &str = "The history of the Roman Empire begins with";
const N_TOKENS: usize = 48;
const SEEDS: u64 = 8;
const K: usize = 4;

fn build(path: &Path) -> (tempura_engine::config::ModelConfig, TransformerModel<
    tempura_kernels::attention::CpuAttention,
    tempura_kernels::rmsnorm::CpuRMSNorm,
    tempura_kernels::rope::CpuRoPE,
    tempura_kernels::matmul::CpuMatMul,
    tempura_kernels::sampling::CpuSampler,
>) {
    let loader = GgufLoader::from_file_quantized(path).expect("gguf open failed");
    let config = loader.config.clone();
    let qw = loader.to_quantized_weights();
    let mut model = TransformerModel::new(config.clone(), KernelPipeline::cpu_native());
    model.quantized_weights = Some(qw);
    (config, model)
}

/// Teacher-forced mean log p_target of a generated continuation.
fn score(
    model: &TransformerModel<
        tempura_kernels::attention::CpuAttention,
        tempura_kernels::rmsnorm::CpuRMSNorm,
        tempura_kernels::rope::CpuRoPE,
        tempura_kernels::matmul::CpuMatMul,
        tempura_kernels::sampling::CpuSampler,
    >,
    config: &tempura_engine::config::ModelConfig,
    prompt_tokens: &[u32],
    generated: &[u32],
) -> f64 {
    let mut sess = InferenceSession::new(config, TraceLevel::Off, 999);
    // forward() returns only the LAST position's logits: decode
    // incrementally, reading logp of generated[i] from the logits that
    // predicted it, then consuming the token.
    let mut logits = model.forward(&mut sess, prompt_tokens);
    let mut logp_sum = 0.0f64;
    for &tok in generated {
        let v = logits.len();
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse: f32 = logits.iter().map(|l| (l - max).exp()).sum::<f32>().ln();
        logp_sum += (logits[tok as usize] - max - lse) as f64;
        logits = model.forward(&mut sess, &[tok]);
    }
    logp_sum / generated.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target_path = Path::new(&args[1]);
    let draft_path = Path::new(&args[2]);

    let (tcfg, target) = build(target_path);
    let (dcfg, draft) = build(draft_path);
    assert_eq!(
        tcfg.vocab_size, dcfg.vocab_size,
        "target/draft vocab mismatch: {} vs {}",
        tcfg.vocab_size, dcfg.vocab_size
    );
    let loader = GgufLoader::from_file_quantized(target_path).unwrap();
    let tokenizer = Tokenizer::from_gguf_metadata(&loader.metadata).expect("tokenizer");
    let prompt_tokens = tokenizer.encode(PROMPT).expect("encode");
    println!(
        "target vocab={} layers={} | draft vocab={} layers={} | prompt {} tokens",
        tcfg.vocab_size, tcfg.num_layers, dcfg.vocab_size, dcfg.num_layers, prompt_tokens.len()
    );

    let params = SamplingParams {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.95,
        ..SamplingParams::default()
    };
    let decoder = SpeculativeDecoder::new(&target, &draft, K);

    // --- 1. Determinism ---
    let run_spec = |seed: u64| -> Vec<u32> {
        let mut ts = InferenceSession::new(&tcfg, TraceLevel::Summary, seed);
        let mut ds = InferenceSession::new(&dcfg, TraceLevel::Off, seed);
        decoder.generate(&mut ts, &mut ds, &prompt_tokens, N_TOKENS, &params, None)
    };
    let a = run_spec(7);
    let b = run_spec(7);
    assert_eq!(a, b, "same-seed spec runs diverged");
    println!("determinism: PASS ({} tokens identical)", a.len());
    println!("spec sample: {:?}", tokenizer.decode(&a[..16.min(a.len())]).unwrap_or_default());

    // --- 2. Likelihood comparison over seeds ---
    let mut plain_sum = 0.0f64;
    let mut spec_sum = 0.0f64;
    let mut accepted_total = 0u64;
    let mut drafted_total = 0u64;
    for seed in 0..SEEDS {
        let mut ts = InferenceSession::new(&tcfg, TraceLevel::Off, seed);
        let plain = target.generate(&mut ts, &prompt_tokens, N_TOKENS, &params, None);
        plain_sum += score(&target, &tcfg, &prompt_tokens, &plain);

        let mut ts = InferenceSession::new(&tcfg, TraceLevel::Summary, seed ^ 0xBEEF);
        let mut ds = InferenceSession::new(&dcfg, TraceLevel::Off, seed ^ 0xBEEF);
        let spec = decoder.generate(&mut ts, &mut ds, &prompt_tokens, N_TOKENS, &params, None);
        spec_sum += score(&target, &tcfg, &prompt_tokens, &spec);
        for snap in &ts.trace.speculative_steps {
            drafted_total += snap.draft_proposed.len() as u64;
            accepted_total += snap.accepted.len().min(snap.draft_proposed.len()) as u64;
        }
    }
    let plain_mean = plain_sum / SEEDS as f64;
    let spec_mean = spec_sum / SEEDS as f64;
    let accept_rate = accepted_total as f64 / drafted_total.max(1) as f64;
    println!(
        "mean logp: plain={plain_mean:.4} spec={spec_mean:.4} delta={:.4}",
        (spec_mean - plain_mean).abs()
    );
    println!("acceptance: {accepted_total}/{drafted_total} = {accept_rate:.3}");
    // SEEDS x N_TOKENS tokens: per-token logp std at temp 0.8 is ~1.5
    // nats; SE of a mean over 384 tokens ~ 0.08 per leg. A skewed
    // acceptance scheme shifts the mean by the acceptance skew (the old
    // rule's skew measured ~0.09 nats on the tiny-model unit gate).
    assert!(
        (spec_mean - plain_mean).abs() < 0.25,
        "spec likelihood diverges from plain sampling"
    );
    println!("likelihood gate: PASS");
}
