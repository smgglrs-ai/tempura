// greedy-probe -- debug tool: load a GGUF, run a prompt through the CPU
// quantized forward, print the greedy continuation. The poor-man's
// differential against `llama-cli --temp 0` (see the cross-implementation
// debugging playbook in docs/status/current.md).
//
// Run:
//   cargo run -p tempura-bench-gpu --release --bin greedy-probe -- \
//     /path/to/model.gguf "prompt text" [n_tokens]

use std::path::PathBuf;
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::TransformerModel;
use tempura_engine::pipeline::KernelPipeline;
use tempura_engine::session::InferenceSession;
use tempura_engine::trace::TraceLevel;

use tempura_engine::tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = PathBuf::from(&args[1]);
    let prompt = args.get(2).cloned().unwrap_or_else(|| "The capital of France is".into());
    let n_tokens: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    let loader = GgufLoader::from_file_quantized(&model_path).expect("gguf open failed");
    let config = loader.config.clone();
    let metadata = loader.metadata.clone();
    println!(
        "config: arch={:?} h={} heads={} kv={} hd={} layers={} theta={} theta_swa={:?} swa={:?} pattern={:?} qk_norm={} act={:?}",
        config.architecture,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.num_layers,
        config.rope_theta,
        config.rope_theta_swa,
        config.sliding_window,
        config.sliding_window_pattern,
        config.use_qk_norm,
        config.hidden_act,
    );

    let tokenizer = Tokenizer::from_gguf_metadata(&metadata).expect("tokenizer failed");
    let use_f32 = std::env::var("TEMPURA_PROBE_F32").is_ok();
    let mut model = if use_f32 {
        #[allow(unused_mut)]
        let mut m = TransformerModel::new(config.clone(), KernelPipeline::cpu_verified());
        m.load_weights(&loader).expect("f32 weight load failed");
        m
    } else {
        let qw = loader.to_quantized_weights();
        let mut m = TransformerModel::new(config.clone(), KernelPipeline::cpu_native());
        m.quantized_weights = Some(qw);
        m
    };

    let tokens = tokenizer.encode(&prompt).expect("encode failed");
    println!("prompt tokens ({}): {tokens:?}", tokens.len());

    let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
    let logits = model.forward(&mut session, &tokens);
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    println!(
        "logits[0..5]: {:?}  argmax={} piece={:?}",
        &logits[..5.min(logits.len())],
        argmax,
        tokenizer.decode(&[argmax as u32]).unwrap_or_default()
    );

    let mut generated = vec![argmax as u32];
    for _ in 1..n_tokens {
        let logits = model.forward(&mut session, &[*generated.last().unwrap()]);
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        generated.push(argmax as u32);
    }
    println!("greedy continuation: {}", tokenizer.decode(&generated).unwrap_or_default());
}
