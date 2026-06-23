use std::collections::HashMap;
use std::time::Instant;

use tempura_engine::config::ModelConfig;
use tempura_engine::kv_cache::{KvCache, PagedKvCache};
use tempura_engine::model::TransformerModel;
use tempura_engine::pipeline::KernelPipeline;
use tempura_engine::session::InferenceSession;
use tempura_engine::trace::TraceLevel;
use tempura_kernels::SamplingParams;

struct RandomWeightLoader {
    tensors: HashMap<String, Vec<f32>>,
}

impl RandomWeightLoader {
    fn from_config(config: &ModelConfig, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed);
        let h = config.hidden_size;
        let v = config.vocab_size;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let q_dim = config.num_attention_heads * config.head_dim;
        let inter = config.intermediate_size;

        let mut tensors = HashMap::new();
        let mut rand_vec = |size: usize| -> Vec<f32> {
            (0..size).map(|_| rng.random_range(-0.1..0.1)).collect()
        };

        tensors.insert("model.embed_tokens.weight".into(), rand_vec(v * h));
        tensors.insert("model.norm.weight".into(), vec![1.0; h]);
        tensors.insert("lm_head.weight".into(), rand_vec(v * h));

        for i in 0..config.num_layers {
            let prefix = format!("model.layers.{i}");
            tensors.insert(format!("{prefix}.input_layernorm.weight"), vec![1.0; h]);
            tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![1.0; h],
            );
            tensors.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                rand_vec(q_dim * h),
            );
            tensors.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                rand_vec(kv_dim * h),
            );
            tensors.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                rand_vec(kv_dim * h),
            );
            tensors.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                rand_vec(h * q_dim),
            );
            tensors.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                rand_vec(inter * h),
            );
            tensors.insert(format!("{prefix}.mlp.up_proj.weight"), rand_vec(inter * h));
            tensors.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                rand_vec(h * inter),
            );
        }

        Self { tensors }
    }
}

impl tempura_engine::loader::WeightLoader for RandomWeightLoader {
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

fn bench<F: FnMut()>(name: &str, warmup: usize, iters: usize, mut f: F) {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters as u32;
    println!(
        "  {name:<45} {:>8.2}ms/iter  ({iters} iters, {:.1}s total)",
        per_iter.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64()
    );
}

fn main() {
    let config = ModelConfig::tiny_test();
    let loader = RandomWeightLoader::from_config(&config, 42);
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 50,
        top_p: 1.0,
        ..SamplingParams::default()
    };
    let prompt = [1u32, 2, 3, 4, 5, 6, 7, 8];

    println!("tempura inference benchmark (tiny_test model)");
    println!("  hidden={} layers={} heads={} vocab={}",
        config.hidden_size, config.num_layers,
        config.num_attention_heads, config.vocab_size);
    println!();

    // --- Forward pass ---
    println!("Forward pass (prefill {} tokens):", prompt.len());
    {
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        model.load_weights(&loader).unwrap();

        bench("contiguous KV cache", 2, 50, || {
            let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
            let _ = model.forward(&mut session, &prompt);
        });

        bench("paged KV cache (block_size=4)", 2, 50, || {
            let mut session =
                InferenceSession::new_paged(&config, 64, 4, TraceLevel::Off, 42);
            let _ = model.forward(&mut session, &prompt);
        });

        bench("paged KV cache (block_size=16)", 2, 50, || {
            let mut session =
                InferenceSession::new_paged(&config, 64, 16, TraceLevel::Off, 42);
            let _ = model.forward(&mut session, &prompt);
        });
    }

    // --- Generate ---
    println!("\nGenerate (prompt={} tokens, generate 10 tokens):", prompt.len());
    {
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        model.load_weights(&loader).unwrap();

        bench("contiguous KV cache", 1, 20, || {
            let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
            let _ = model.generate(&mut session, &prompt, 10, &params, None);
        });

        bench("paged KV cache (block_size=16)", 1, 20, || {
            let mut session =
                InferenceSession::new_paged(&config, 64, 16, TraceLevel::Off, 42);
            let _ = model.generate(&mut session, &prompt, 10, &params, None);
        });
    }

    // --- KV cache operations ---
    println!("\nKV cache operations (4 layers, 8 kv_heads, 64 head_dim):");
    {
        let stride = config.num_kv_heads * config.head_dim;
        let k_data: Vec<f32> = (0..stride).map(|i| i as f32 * 0.01).collect();
        let v_data = k_data.clone();

        bench("contiguous append 1 token × 100", 2, 100, || {
            let mut cache = KvCache::new(&config);
            for _ in 0..100 {
                cache.append_kv(0, &k_data, &v_data, 1);
                cache.advance(1);
            }
        });

        bench("paged append 1 token × 100 (bs=16)", 2, 100, || {
            let mut cache = PagedKvCache::new(&config, 128, 16);
            for _ in 0..100 {
                cache.append_kv(0, &k_data, &v_data, 1);
                cache.advance(1);
            }
        });

        bench("paged k_contiguous(100 tokens)", 2, 500, || {
            let mut cache = PagedKvCache::new(&config, 128, 16);
            let k100: Vec<f32> = (0..100 * stride).map(|i| i as f32 * 0.01).collect();
            let v100 = k100.clone();
            cache.append_kv(0, &k100, &v100, 100);
            cache.advance(100);
            let _ = cache.k_contiguous(0, 100);
        });
    }

    // --- Page allocator ---
    println!("\nPage allocator (1024 pages):");
    {
        use tempura_engine::kv_cache::PageAllocator;

        bench("allocate+free 1024 pages", 5, 1000, || {
            let mut alloc = PageAllocator::new(1024);
            let mut pages = Vec::with_capacity(1024);
            for _ in 0..1024 {
                pages.push(alloc.allocate().unwrap());
            }
            for p in pages.drain(..) {
                alloc.free(p);
            }
        });
    }

    println!("\nDone.");
}
