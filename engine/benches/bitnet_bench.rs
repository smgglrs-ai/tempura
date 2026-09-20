use std::time::Instant;

use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::TransformerModel;
use tempura_engine::pipeline::KernelPipeline;
use tempura_engine::session::InferenceSession;
use tempura_engine::trace::TraceLevel;
use tempura_kernels::SamplingParams;

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: bitnet_bench <path-to-gguf> [max_tokens] [prompt] [temperature]");
        std::process::exit(1);
    });
    let max_tokens: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let prompt_text = std::env::args().nth(3);
    let temperature: f32 = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    eprintln!("Loading model from {model_path}...");
    let load_start = Instant::now();
    let gguf = GgufLoader::from_file_quantized(std::path::Path::new(&model_path)).unwrap();
    let config = gguf.config.clone();
    let tokenizer = tempura_engine::tokenizer::Tokenizer::from_gguf_metadata(&gguf.metadata).ok();
    let qweights = gguf.to_quantized_weights();
    let pipeline = KernelPipeline::cpu_verified();
    let mut model = TransformerModel::new(config.clone(), pipeline);
    model.quantized_weights = Some(qweights);
    eprintln!(
        "Model loaded in {:.1}s  (hidden={}, layers={}, heads={}, vocab={})",
        load_start.elapsed().as_secs_f64(),
        config.hidden_size,
        config.num_layers,
        config.num_attention_heads,
        config.vocab_size
    );

    let prompt_ids: Vec<u32> = if let Some(ref text) = prompt_text {
        if let Some(ref tok) = tokenizer {
            tok.encode(text).expect("encode failed")
        } else {
            vec![1]
        }
    } else {
        vec![1]
    };

    eprintln!("Prompt: {} tokens {:?}", prompt_ids.len(), &prompt_ids);

    let params = SamplingParams {
        temperature,
        top_k: if temperature == 0.0 { 1 } else { 40 },
        top_p: 0.95,
        ..SamplingParams::default()
    };
    if temperature > 0.0 {
        eprintln!("Sampling: temperature={temperature}, top_k={}, top_p={}", params.top_k, params.top_p);
    }

    // Dump top-5 logits for first 3 generated tokens
    if prompt_text.is_some() {
        eprintln!("=== Logit dump (first 3 tokens) ===");
        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits0 = model.forward(&mut session, &prompt_ids);
        let mut top: Vec<(usize, f32)> = logits0.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("  pos={}: top5={:?}", prompt_ids.len() - 1, &top[..5].iter().map(|(i,v)| format!("{}({:.4})", i, v)).collect::<Vec<_>>());
        let tok0 = top[0].0 as u32;

        let logits1 = model.forward(&mut session, &[tok0]);
        let mut top1: Vec<(usize, f32)> = logits1.iter().copied().enumerate().collect();
        top1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("  pos={}: top5={:?}", prompt_ids.len(), &top1[..5].iter().map(|(i,v)| format!("{}({:.4})", i, v)).collect::<Vec<_>>());
        let tok1 = top1[0].0 as u32;

        let logits2 = model.forward(&mut session, &[tok1]);
        let mut top2: Vec<(usize, f32)> = logits2.iter().copied().enumerate().collect();
        top2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("  pos={}: top5={:?}", prompt_ids.len() + 1, &top2[..5].iter().map(|(i,v)| format!("{}({:.4})", i, v)).collect::<Vec<_>>());
        eprintln!();
    }

    // Warmup
    eprintln!("Warmup...");
    {
        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let _ = model.generate(&mut session, &prompt_ids, 2, &params, None);
    }

    // Benchmark
    eprintln!("Generating {max_tokens} tokens...");
    let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
    let gen_start = Instant::now();
    let tokens = model.generate(&mut session, &prompt_ids, max_tokens, &params, None);
    let gen_elapsed = gen_start.elapsed();
    let gen_tokens = tokens.len();

    let timing = session.trace.timing.as_ref().unwrap();
    let tps = gen_tokens as f64 / gen_elapsed.as_secs_f64();

    eprintln!();
    println!("| model | tokens | time (s) | tok/s |");
    println!("| ----- | -----: | -------: | ----: |");
    println!(
        "| tempura bitnet tg{max_tokens} | {gen_tokens} | {:.2} | {:.2} |",
        gen_elapsed.as_secs_f64(),
        tps
    );
    eprintln!();
    eprintln!(
        "Engine timing: prompt={:.1}ms  generation={:.1}ms  engine_tok/s={:.2}",
        timing.prompt_ms, timing.generation_ms, timing.tokens_per_second
    );

    eprintln!("Output token IDs: {:?}", &tokens);
    if let Some(ref tok) = tokenizer {
        let decoded = tok.decode(&tokens).unwrap_or_else(|_| "(decode error)".into());
        eprintln!();
        eprintln!("=== Output ===");
        eprintln!("{decoded}");
    }
}
