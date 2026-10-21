use std::time::Instant;
use tempura_engine::config::ModelConfig;
use tempura_engine::loader::WeightLoader;
use tempura_engine::model::CutileInferenceWrapper;
use tempura_engine::model::InferenceModel;
use tempura_engine::session::InferenceSession;
use tempura_engine::trace::TraceLevel;

const WARMUP: usize = 30;
const ITERS: usize = 50;

struct RandomWeightLoader {
    tensors: std::collections::HashMap<String, Vec<f32>>,
}

impl RandomWeightLoader {
    fn new(config: &ModelConfig) -> Self {
        let mut tensors = std::collections::HashMap::new();
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let v = config.vocab_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        let gen = |size: usize| -> Vec<f32> {
            (0..size)
                .map(|i| ((i as f32) * 0.0001).sin() * 0.01)
                .collect()
        };

        tensors.insert("model.embed_tokens.weight".into(), gen(v * h));
        tensors.insert("model.norm.weight".into(), gen(h));
        tensors.insert("lm_head.weight".into(), gen(v * h));

        for l in 0..config.num_layers {
            let p = format!("model.layers.{l}");
            tensors.insert(format!("{p}.input_layernorm.weight"), gen(h));
            tensors.insert(format!("{p}.post_attention_layernorm.weight"), gen(h));
            tensors.insert(format!("{p}.self_attn.q_proj.weight"), gen(q_dim * h));
            tensors.insert(format!("{p}.self_attn.k_proj.weight"), gen(kv_dim * h));
            tensors.insert(format!("{p}.self_attn.v_proj.weight"), gen(kv_dim * h));
            tensors.insert(format!("{p}.self_attn.o_proj.weight"), gen(h * q_dim));
            tensors.insert(format!("{p}.mlp.gate_proj.weight"), gen(inter * h));
            tensors.insert(format!("{p}.mlp.up_proj.weight"), gen(inter * h));
            tensors.insert(format!("{p}.mlp.down_proj.weight"), gen(h * inter));
        }

        Self { tensors }
    }
}

impl WeightLoader for RandomWeightLoader {
    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, tempura_engine::loader::LoadError> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| tempura_engine::loader::LoadError::MissingTensor(name.to_string()))
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

fn main() {
    println!("????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????");
    println!("???  tempura cuTile End-to-End Benchmark                   ???");
    println!("????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????\n");

    let config = ModelConfig::tiny_test();
    println!(
        "  Model: {} layers, h={}, heads={}/{}, d={}, vocab={}",
        config.num_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.vocab_size,
    );
    println!("  Warmup: {WARMUP}, Iters: {ITERS}");
    println!(
        "  Note: first ~15 iters include JIT compilation -- stable numbers start around iter 15\n"
    );

    let loader = RandomWeightLoader::new(&config);
    let mut wrapper = CutileInferenceWrapper::new(config.clone());
    wrapper.model.load_weights(&loader).unwrap();

    // --- Prefill benchmark ---
    println!("--- Prefill (seq_len=64) ---");
    let prompt: Vec<u32> = (0..64).map(|i| (i % config.vocab_size) as u32).collect();

    for _ in 0..WARMUP {
        let mut session = InferenceSession::new(&config, TraceLevel::Off, 0);
        wrapper.forward(&mut session, &prompt);
    }

    let mut prefill_times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let mut session = InferenceSession::new(&config, TraceLevel::Off, 0);
        let start = Instant::now();
        wrapper.forward(&mut session, &prompt);
        prefill_times.push(start.elapsed().as_micros() as f64);
    }

    let avg_prefill = prefill_times.iter().sum::<f64>() / ITERS as f64;
    let min_prefill = prefill_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_prefill = prefill_times.iter().cloned().fold(0.0, f64::max);

    println!("  TTFT (avg):      {:.2} ms", avg_prefill / 1000.0);
    println!(
        "  TTFT (min/max):  {:.2} / {:.2} ms",
        min_prefill / 1000.0,
        max_prefill / 1000.0
    );
    println!(
        "  Prefill tok/s:   {:.0}",
        prompt.len() as f64 / (avg_prefill / 1e6)
    );
    println!();

    // --- Decode benchmark: single-token autoregressive steps ---
    // Use fresh sessions for each warmup/measure pass so position never overflows
    // the tiny_test KV cache (max_seq_len=128). GPU kernels remain JIT-compiled
    // after the first warmup session, so later measurements are kernel-stable.
    println!("--- Decode (single token, autoregressive) ---");
    let decode_token = [42u32];
    for _ in 0..WARMUP {
        let mut session = InferenceSession::new(&config, TraceLevel::Off, 0);
        wrapper.forward(&mut session, &prompt);
        wrapper.forward(&mut session, &decode_token);
    }

    let mut decode_times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let mut session = InferenceSession::new(&config, TraceLevel::Off, 0);
        wrapper.forward(&mut session, &prompt);
        let start = Instant::now();
        wrapper.forward(&mut session, &decode_token);
        decode_times.push(start.elapsed().as_micros() as f64);
    }

    let avg_decode = decode_times.iter().sum::<f64>() / ITERS as f64;
    let min_decode = decode_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_decode = decode_times.iter().cloned().fold(0.0, f64::max);
    let tok_per_s = 1e6 / avg_decode;

    println!("  ITL (avg):       {:.2} ms", avg_decode / 1000.0);
    println!(
        "  ITL (min/max):   {:.2} / {:.2} ms",
        min_decode / 1000.0,
        max_decode / 1000.0
    );
    println!("  Decode tok/s:    {:.0}", tok_per_s);
    println!();

    println!("Done.");
}
