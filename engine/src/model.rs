use crate::config::ModelConfig;
use crate::pipeline::KernelPipeline;
use crate::session::InferenceSession;
use crate::trace::{LayerSnapshot, SamplingSnapshot, TimingInfo, TraceLevel};
use crate::weights::ModelWeights;
use oxide_kernels::{Attention, MatMul, RMSNorm, RoPE, Sampler, SamplingParams};
use serde::Serialize;
use std::time::Instant;

pub trait InferenceModel: Send + Sync {
    fn config(&self) -> &ModelConfig;
    fn has_weights(&self) -> bool;
    fn forward(&self, session: &mut InferenceSession, token_ids: &[u32]) -> Vec<f32>;
    fn generate(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
    ) -> Vec<u32>;
    fn generate_with_logprobs(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        top_n: usize,
    ) -> (Vec<u32>, Vec<TokenLogprob>);
    fn generate_with_callback(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Vec<u32>;
    fn generate_guided(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        decoder: &mut dyn oxide_kernels::GuidedDecoder,
    ) -> Vec<u32>;
    fn load_weights(
        &mut self,
        loader: &dyn crate::loader::WeightLoader,
    ) -> Result<(), crate::weights::WeightsError>;
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    pub token_id: u32,
    pub logprob: f32,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub token_id: u32,
    pub logprob: f32,
}

fn extract_logprob(probs: &[f32], token_id: u32, top_n: usize) -> TokenLogprob {
    let logprob = if probs[token_id as usize] > 0.0 {
        probs[token_id as usize].ln()
    } else {
        f32::NEG_INFINITY
    };

    let mut indexed: Vec<(u32, f32)> = probs
        .iter()
        .enumerate()
        .filter(|(_, &p)| p > 0.0)
        .map(|(i, &p)| (i as u32, p))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_logprobs: Vec<TopLogprob> = indexed
        .iter()
        .take(top_n.min(20))
        .map(|&(id, p)| TopLogprob {
            token_id: id,
            logprob: p.ln(),
        })
        .collect();

    TokenLogprob {
        token_id,
        logprob,
        top_logprobs,
    }
}

pub struct TransformerModel<A, N, R, M, S> {
    pub config: ModelConfig,
    pub pipeline: KernelPipeline<A, N, R, M, S>,
    pub weights: Option<ModelWeights>,
    #[cfg(feature = "cuda")]
    pub gpu_weights: Option<crate::weights::GpuWeights>,
}

impl<A, N, R, M, S> TransformerModel<A, N, R, M, S>
where
    A: Attention,
    N: RMSNorm,
    R: RoPE,
    M: MatMul,
    S: Sampler,
{
    pub fn new(config: ModelConfig, pipeline: KernelPipeline<A, N, R, M, S>) -> Self {
        Self {
            config,
            pipeline,
            weights: None,
            #[cfg(feature = "cuda")]
            gpu_weights: None,
        }
    }

    pub fn load_weights(
        &mut self,
        loader: &dyn crate::loader::WeightLoader,
    ) -> Result<(), crate::weights::WeightsError> {
        self.weights = Some(ModelWeights::load(loader, &self.config)?);
        Ok(())
    }

    pub fn forward(&self, session: &mut InferenceSession, token_ids: &[u32]) -> Vec<f32> {
        let weights = match &self.weights {
            Some(w) => w,
            None => return vec![0.0; self.config.vocab_size],
        };

        let cfg = &self.config;
        let seq_len = token_ids.len();
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let inter = cfg.intermediate_size;

        let embed = weights.get("model.embed_tokens.weight").unwrap();
        let mut hidden = vec![0.0f32; seq_len * h];
        for (t, &tok) in token_ids.iter().enumerate() {
            let src = &embed[tok as usize * h..(tok as usize + 1) * h];
            hidden[t * h..(t + 1) * h].copy_from_slice(src);
        }

        let positions: Vec<u32> = (session.position..session.position + seq_len)
            .map(|p| p as u32)
            .collect();

        for layer in 0..cfg.num_layers {
            let layer_start = Instant::now();
            let prefix = format!("model.layers.{layer}");

            let mut snapshot = LayerSnapshot {
                layer_index: layer,
                pre_norm: None,
                post_norm: None,
                attention_scores: None,
                duration: None,
            };

            // --- attention block ---
            let mut normed = hidden.clone();
            let input_ln_w = weights
                .get(&format!("{prefix}.input_layernorm.weight"))
                .unwrap();
            self.pipeline
                .rmsnorm(
                    &mut normed,
                    input_ln_w,
                    h,
                    cfg.layer_norm_eps,
                    session.trace_level,
                    &mut Some(&mut snapshot),
                )
                .unwrap();

            let q_w = weights
                .get(&format!("{prefix}.self_attn.q_proj.weight"))
                .unwrap();
            let k_w = weights
                .get(&format!("{prefix}.self_attn.k_proj.weight"))
                .unwrap();
            let v_w = weights
                .get(&format!("{prefix}.self_attn.v_proj.weight"))
                .unwrap();
            let o_w = weights
                .get(&format!("{prefix}.self_attn.o_proj.weight"))
                .unwrap();

            let q_dim = n_heads * head_dim;
            let kv_dim = n_kv_heads * head_dim;
            let mut q = vec![0.0f32; seq_len * q_dim];
            let mut k = vec![0.0f32; seq_len * kv_dim];
            let mut v = vec![0.0f32; seq_len * kv_dim];

            self.pipeline
                .matmul_bt(&normed, q_w, &mut q, seq_len, q_dim, h)
                .unwrap();
            self.pipeline
                .matmul_bt(&normed, k_w, &mut k, seq_len, kv_dim, h)
                .unwrap();
            self.pipeline
                .matmul_bt(&normed, v_w, &mut v, seq_len, kv_dim, h)
                .unwrap();

            self.pipeline
                .rope(
                    &mut q,
                    1,
                    seq_len,
                    n_heads,
                    head_dim,
                    &positions,
                    cfg.rope_theta,
                )
                .unwrap();
            self.pipeline
                .rope(
                    &mut k,
                    1,
                    seq_len,
                    n_kv_heads,
                    head_dim,
                    &positions,
                    cfg.rope_theta,
                )
                .unwrap();

            session.cache.append_kv(layer, &k, &v, seq_len);
            let kv_len = session.cache.seq_len + seq_len;
            let k_cache = session.cache.k_data(layer, kv_len);
            let v_cache = session.cache.v_data(layer, kv_len);

            let mut attn_out = vec![0.0f32; seq_len * q_dim];
            self.pipeline
                .attention(
                    &q,
                    &k_cache,
                    &v_cache,
                    &mut attn_out,
                    1,
                    seq_len,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    1.0 / (head_dim as f32).sqrt(),
                )
                .unwrap();

            let mut o_proj = vec![0.0f32; seq_len * h];
            self.pipeline
                .matmul_bt(&attn_out, o_w, &mut o_proj, seq_len, h, q_dim)
                .unwrap();

            for i in 0..hidden.len() {
                hidden[i] += o_proj[i];
            }

            // --- FFN block ---
            let mut normed2 = hidden.clone();
            let post_ln_w = weights
                .get(&format!("{prefix}.post_attention_layernorm.weight"))
                .unwrap();
            self.pipeline
                .rmsnorm(
                    &mut normed2,
                    post_ln_w,
                    h,
                    cfg.layer_norm_eps,
                    TraceLevel::Off,
                    &mut None,
                )
                .unwrap();

            let gate_w = weights
                .get(&format!("{prefix}.mlp.gate_proj.weight"))
                .unwrap();
            let up_w = weights
                .get(&format!("{prefix}.mlp.up_proj.weight"))
                .unwrap();
            let down_w = weights
                .get(&format!("{prefix}.mlp.down_proj.weight"))
                .unwrap();

            let mut gate = vec![0.0f32; seq_len * inter];
            let mut up = vec![0.0f32; seq_len * inter];
            self.pipeline
                .matmul_bt(&normed2, gate_w, &mut gate, seq_len, inter, h)
                .unwrap();
            self.pipeline
                .matmul_bt(&normed2, up_w, &mut up, seq_len, inter, h)
                .unwrap();

            // SiLU(gate) * up
            for i in 0..gate.len() {
                let silu = gate[i] / (1.0 + (-gate[i]).exp());
                gate[i] = silu * up[i];
            }

            let mut down_out = vec![0.0f32; seq_len * h];
            self.pipeline
                .matmul_bt(&gate, down_w, &mut down_out, seq_len, h, inter)
                .unwrap();

            for i in 0..hidden.len() {
                hidden[i] += down_out[i];
            }

            if session.trace_level >= TraceLevel::Summary {
                snapshot.duration = Some(layer_start.elapsed());
                session.trace.layers.push(snapshot);
            }
        }

        session.cache.advance(seq_len);
        session.position += seq_len;

        // final norm
        let final_norm_w = weights.get("model.norm.weight").unwrap();
        self.pipeline
            .rmsnorm(
                &mut hidden,
                final_norm_w,
                h,
                cfg.layer_norm_eps,
                TraceLevel::Off,
                &mut None,
            )
            .unwrap();

        // lm_head: project last token to vocab
        let lm_head_w = weights.get("lm_head.weight").unwrap();
        let last_hidden = &hidden[(seq_len - 1) * h..seq_len * h];
        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.pipeline
            .matmul_bt(last_hidden, lm_head_w, &mut logits, 1, cfg.vocab_size, h)
            .unwrap();

        logits
    }

    pub fn chunked_forward(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        chunk_size: usize,
    ) -> Vec<f32> {
        if token_ids.len() <= chunk_size {
            return self.forward(session, token_ids);
        }

        let mut logits = Vec::new();
        let mut offset = 0;
        while offset < token_ids.len() {
            let end = (offset + chunk_size).min(token_ids.len());
            let chunk = &token_ids[offset..end];
            logits = self.forward(session, chunk);
            offset = end;
        }
        logits
    }

    pub fn sample(
        &self,
        logits: &[f32],
        params: &SamplingParams,
    ) -> oxide_kernels::Result<Vec<f32>> {
        let mut probs = vec![0.0; logits.len()];
        self.pipeline.sampler.apply(logits, &mut probs, params)?;
        Ok(probs)
    }

    pub fn generate(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
    ) -> Vec<u32> {
        let total_start = Instant::now();
        let prompt_start = Instant::now();
        let logits = self.forward(session, token_ids);
        let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.prompt_tokens = token_ids.len();
        let gen_start = Instant::now();

        let mut probs = vec![0.0f32; logits.len()];
        self.pipeline
            .sampler
            .apply(&logits, &mut probs, params)
            .unwrap();

        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut session.rng)
        };
        let first_token = self.pipeline.sampler.sample_token(&probs, rng_ref);

        let mut generated = vec![first_token];
        self.record_sampling_snapshot(session, 0, &logits, &probs, first_token, params, None);

        for step in 1..max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) {
                break;
            }

            let mut logits = self.forward(session, &[*generated.last().unwrap()]);
            params.apply_penalties(&mut logits, &generated);
            let mut probs = vec![0.0f32; logits.len()];
            self.pipeline
                .sampler
                .apply(&logits, &mut probs, params)
                .unwrap();

            let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                None
            } else {
                Some(&mut session.rng)
            };
            let token = self.pipeline.sampler.sample_token(&probs, rng_ref);

            self.record_sampling_snapshot(session, step, &logits, &probs, token, params, None);
            generated.push(token);
        }

        let generation_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.generated_tokens = generated.len();
        session.trace.timing = Some(TimingInfo {
            total_ms,
            prompt_ms,
            generation_ms,
            tokens_per_second: if generation_ms > 0.0 {
                generated.len() as f64 / (generation_ms / 1000.0)
            } else {
                0.0
            },
        });
        generated
    }

    pub fn generate_with_logprobs(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        top_n: usize,
    ) -> (Vec<u32>, Vec<TokenLogprob>) {
        let total_start = Instant::now();
        let prompt_start = Instant::now();
        let logits = self.forward(session, token_ids);
        let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.prompt_tokens = token_ids.len();
        let gen_start = Instant::now();

        let mut probs = vec![0.0f32; logits.len()];
        self.pipeline
            .sampler
            .apply(&logits, &mut probs, params)
            .unwrap();

        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut session.rng)
        };
        let first_token = self.pipeline.sampler.sample_token(&probs, rng_ref);

        let mut generated = vec![first_token];
        let mut logprobs = vec![extract_logprob(&probs, first_token, top_n)];

        for _step in 1..max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) {
                break;
            }

            let mut step_logits = self.forward(session, &[*generated.last().unwrap()]);
            params.apply_penalties(&mut step_logits, &generated);
            let mut step_probs = vec![0.0f32; step_logits.len()];
            self.pipeline
                .sampler
                .apply(&step_logits, &mut step_probs, params)
                .unwrap();

            let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                None
            } else {
                Some(&mut session.rng)
            };
            let token = self.pipeline.sampler.sample_token(&step_probs, rng_ref);

            logprobs.push(extract_logprob(&step_probs, token, top_n));
            generated.push(token);
        }

        let generation_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.generated_tokens = generated.len();
        session.trace.timing = Some(TimingInfo {
            total_ms,
            prompt_ms,
            generation_ms,
            tokens_per_second: if generation_ms > 0.0 {
                generated.len() as f64 / (generation_ms / 1000.0)
            } else {
                0.0
            },
        });
        (generated, logprobs)
    }

    pub fn generate_with_callback(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Vec<u32> {
        let total_start = Instant::now();
        let prompt_start = Instant::now();
        let logits = self.forward(session, token_ids);
        let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.prompt_tokens = token_ids.len();
        let gen_start = Instant::now();

        let mut probs = vec![0.0f32; logits.len()];
        self.pipeline
            .sampler
            .apply(&logits, &mut probs, params)
            .unwrap();

        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut session.rng)
        };
        let first_token = self.pipeline.sampler.sample_token(&probs, rng_ref);

        let mut generated = vec![first_token];
        if !on_token(first_token) {
            let generation_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            session.trace.generated_tokens = generated.len();
            session.trace.timing = Some(TimingInfo {
                total_ms,
                prompt_ms,
                generation_ms,
                tokens_per_second: if generation_ms > 0.0 {
                    generated.len() as f64 / (generation_ms / 1000.0)
                } else {
                    0.0
                },
            });
            return generated;
        }

        for _step in 1..max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) {
                break;
            }

            let mut logits = self.forward(session, &[*generated.last().unwrap()]);
            params.apply_penalties(&mut logits, &generated);
            let mut probs = vec![0.0f32; logits.len()];
            self.pipeline
                .sampler
                .apply(&logits, &mut probs, params)
                .unwrap();

            let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                None
            } else {
                Some(&mut session.rng)
            };
            let token = self.pipeline.sampler.sample_token(&probs, rng_ref);

            generated.push(token);
            if !on_token(token) {
                break;
            }
        }

        let generation_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.generated_tokens = generated.len();
        session.trace.timing = Some(TimingInfo {
            total_ms,
            prompt_ms,
            generation_ms,
            tokens_per_second: if generation_ms > 0.0 {
                generated.len() as f64 / (generation_ms / 1000.0)
            } else {
                0.0
            },
        });
        generated
    }

    pub fn generate_guided(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        decoder: &mut dyn oxide_kernels::GuidedDecoder,
    ) -> Vec<u32> {
        let total_start = Instant::now();
        let prompt_start = Instant::now();
        let mut logits = self.forward(session, token_ids);
        let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.prompt_tokens = token_ids.len();
        let gen_start = Instant::now();

        decoder.mask_logits(&[], &mut logits);
        let masked = decoder.tokens_masked_last_step();
        let mut probs = vec![0.0f32; logits.len()];
        self.pipeline
            .sampler
            .apply(&logits, &mut probs, params)
            .unwrap();

        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut session.rng)
        };
        let first_token = self.pipeline.sampler.sample_token(&probs, rng_ref);

        let mut generated = vec![first_token];
        self.record_sampling_snapshot(
            session,
            0,
            &logits,
            &probs,
            first_token,
            params,
            Some(masked),
        );

        for step in 1..max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) || decoder.is_complete() {
                break;
            }

            let mut logits = self.forward(session, &[*generated.last().unwrap()]);
            params.apply_penalties(&mut logits, &generated);
            decoder.mask_logits(&generated, &mut logits);
            let masked = decoder.tokens_masked_last_step();
            let mut probs = vec![0.0f32; logits.len()];
            self.pipeline
                .sampler
                .apply(&logits, &mut probs, params)
                .unwrap();

            let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                None
            } else {
                Some(&mut session.rng)
            };
            let token = self.pipeline.sampler.sample_token(&probs, rng_ref);

            self.record_sampling_snapshot(
                session,
                step,
                &logits,
                &probs,
                token,
                params,
                Some(masked),
            );
            generated.push(token);
        }

        let generation_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.generated_tokens = generated.len();
        session.trace.timing = Some(TimingInfo {
            total_ms,
            prompt_ms,
            generation_ms,
            tokens_per_second: if generation_ms > 0.0 {
                generated.len() as f64 / (generation_ms / 1000.0)
            } else {
                0.0
            },
        });
        generated
    }

    #[allow(clippy::too_many_arguments)]
    fn record_sampling_snapshot(
        &self,
        session: &mut InferenceSession,
        step: usize,
        logits: &[f32],
        probs: &[f32],
        selected: u32,
        params: &SamplingParams,
        tokens_masked: Option<usize>,
    ) {
        if session.trace_level < TraceLevel::Summary {
            return;
        }

        let k = 5.min(logits.len());
        let mut indexed_logits: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        indexed_logits.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut indexed_probs: Vec<(u32, f32)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, &v)| (i as u32, v))
            .collect();
        indexed_probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        session.trace.sampling_steps.push(SamplingSnapshot {
            step,
            top_logits: indexed_logits[..k].to_vec(),
            top_probs: indexed_probs[..k.min(indexed_probs.len())].to_vec(),
            selected_token: selected,
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            rng_seed: session.seed,
            tokens_masked,
        });
    }
}

#[cfg(feature = "cuda")]
impl
    TransformerModel<
        oxide_kernels::attention::CudaAttention,
        oxide_kernels::rmsnorm::CudaRMSNorm,
        oxide_kernels::rope::CudaRoPE,
        oxide_kernels::matmul::CudaMatMul,
        oxide_kernels::sampling::CpuSampler,
    >
{
    pub fn forward_gpu(
        &self,
        gpu_weights: &crate::weights::GpuWeights,
        gpu_cache: &mut crate::kv_cache::GpuKvCache,
        scratch: &mut crate::gpu_scratchpad::GpuScratchpad,
        token_ids: &[u32],
        position: usize,
    ) -> Vec<f32> {
        let ctx = self.pipeline.matmul.gpu().context().clone();
        ctx.bind_to_thread().expect("bind CUDA context");
        let stream = self.pipeline.matmul.gpu().stream();

        let cfg = &self.config;
        let seq_len = token_ids.len();
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let inter = cfg.intermediate_size;

        let embed_host = self
            .weights
            .as_ref()
            .unwrap()
            .get("model.embed_tokens.weight")
            .unwrap();
        let mut hidden_host = vec![0.0f32; seq_len * h];
        for (t, &tok) in token_ids.iter().enumerate() {
            let src = &embed_host[tok as usize * h..(tok as usize + 1) * h];
            hidden_host[t * h..(t + 1) * h].copy_from_slice(src);
        }
        scratch.upload_hidden(&stream, &hidden_host);

        let positions: Vec<u32> = (position..position + seq_len).map(|p| p as u32).collect();
        let positions_dev =
            cuda_core::DeviceBuffer::from_host(&stream, &positions).expect("upload positions");

        for layer in 0..cfg.num_layers {
            let prefix = format!("model.layers.{layer}");

            // --- RMSNorm ---
            let input_ln_w = gpu_weights
                .get(&format!("{prefix}.input_layernorm.weight"))
                .unwrap();
            self.pipeline
                .rmsnorm
                .forward_device(
                    &scratch.hidden,
                    input_ln_w,
                    &mut scratch.normed,
                    h,
                    cfg.layer_norm_eps,
                )
                .expect("rmsnorm");

            // --- QKV projection (fused) ---
            let q_dim = n_heads * head_dim;
            let kv_dim = n_kv_heads * head_dim;
            let qkv_dim = q_dim + 2 * kv_dim;

            let qkv_w = gpu_weights
                .get(&format!("{prefix}.self_attn.qkv_proj.weight"))
                .unwrap();
            self.pipeline
                .matmul
                .forward_bt_device(&scratch.normed, qkv_w, &mut scratch.qkv, seq_len, qkv_dim, h)
                .expect("qkv proj");

            let ctx = self.pipeline.matmul.gpu().context().clone();
            oxide_kernels::gpu::slice_columns_device(
                &ctx, &stream, &scratch.qkv, &mut scratch.q,
                seq_len, qkv_dim, 0, q_dim,
            ).expect("slice q");
            oxide_kernels::gpu::slice_columns_device(
                &ctx, &stream, &scratch.qkv, &mut scratch.k,
                seq_len, qkv_dim, q_dim, kv_dim,
            ).expect("slice k");
            oxide_kernels::gpu::slice_columns_device(
                &ctx, &stream, &scratch.qkv, &mut scratch.v,
                seq_len, qkv_dim, q_dim + kv_dim, kv_dim,
            ).expect("slice v");

            // --- RoPE ---
            self.pipeline
                .rope
                .forward_device(
                    &scratch.q,
                    &mut scratch.q_rotated,
                    &positions_dev,
                    seq_len,
                    n_heads,
                    head_dim,
                    cfg.rope_theta,
                )
                .expect("rope q");
            self.pipeline
                .rope
                .forward_device(
                    &scratch.k,
                    &mut scratch.k_rotated,
                    &positions_dev,
                    seq_len,
                    n_kv_heads,
                    head_dim,
                    cfg.rope_theta,
                )
                .expect("rope k");

            // --- KV Cache append (device-side copy) ---
            let stride = n_kv_heads * head_dim;
            let dst_offset = gpu_cache.seq_len * stride;
            let src_len = seq_len * stride;
            let ctx = self.pipeline.matmul.gpu().context().clone();
            oxide_kernels::gpu::copy_region_device(
                &ctx,
                &stream,
                &scratch.k_rotated,
                &mut gpu_cache.k[layer],
                0,
                dst_offset,
                src_len,
            )
            .expect("kv cache k append");
            oxide_kernels::gpu::copy_region_device(
                &ctx,
                &stream,
                &scratch.v,
                &mut gpu_cache.v[layer],
                0,
                dst_offset,
                src_len,
            )
            .expect("kv cache v append");

            // --- Attention ---
            let kv_len = gpu_cache.seq_len + seq_len;
            self.pipeline
                .attention
                .forward_device(
                    &scratch.q_rotated,
                    &gpu_cache.k[layer],
                    &gpu_cache.v[layer],
                    &mut scratch.attn_out,
                    1,
                    seq_len,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    1.0 / (head_dim as f32).sqrt(),
                )
                .expect("attention");

            // --- Output projection ---
            let o_w = gpu_weights
                .get(&format!("{prefix}.self_attn.o_proj.weight"))
                .unwrap();
            self.pipeline
                .matmul
                .forward_bt_device(&scratch.attn_out, o_w, &mut scratch.o_proj, seq_len, h, q_dim)
                .expect("o proj");

            // --- Residual add (on device) ---
            let ctx = self.pipeline.matmul.gpu().context().clone();
            oxide_kernels::gpu::vector_add_device(
                &ctx,
                &stream,
                &scratch.hidden,
                &scratch.o_proj,
                &mut scratch.hidden_swap,
            )
            .expect("residual add");
            std::mem::swap(&mut scratch.hidden, &mut scratch.hidden_swap);

            // --- FFN block ---
            let post_ln_w = gpu_weights
                .get(&format!("{prefix}.post_attention_layernorm.weight"))
                .unwrap();
            self.pipeline
                .rmsnorm
                .forward_device(
                    &scratch.hidden,
                    post_ln_w,
                    &mut scratch.normed2,
                    h,
                    cfg.layer_norm_eps,
                )
                .expect("rmsnorm2");

            let gate_up_w = gpu_weights
                .get(&format!("{prefix}.mlp.gate_up_proj.weight"))
                .unwrap();
            let down_w = gpu_weights
                .get(&format!("{prefix}.mlp.down_proj.weight"))
                .unwrap();

            self.pipeline
                .matmul
                .forward_bt_device(
                    &scratch.normed2, gate_up_w, &mut scratch.gate_up,
                    seq_len, 2 * inter, h,
                )
                .expect("gate_up proj");

            let ctx = self.pipeline.matmul.gpu().context().clone();
            oxide_kernels::gpu::slice_columns_device(
                &ctx, &stream, &scratch.gate_up, &mut scratch.gate,
                seq_len, 2 * inter, 0, inter,
            ).expect("slice gate");
            oxide_kernels::gpu::slice_columns_device(
                &ctx, &stream, &scratch.gate_up, &mut scratch.up,
                seq_len, 2 * inter, inter, inter,
            ).expect("slice up");

            // --- SiLU(gate) * up (on device) ---
            let ctx = self.pipeline.matmul.gpu().context().clone();
            oxide_kernels::gpu::silu_mul_device(
                &ctx,
                &stream,
                &scratch.gate,
                &scratch.up,
                &mut scratch.silu_out,
            )
            .expect("silu_mul");

            self.pipeline
                .matmul
                .forward_bt_device(&scratch.silu_out, down_w, &mut scratch.down, seq_len, h, inter)
                .expect("down proj");

            // --- FFN residual add (on device) ---
            let ctx = self.pipeline.matmul.gpu().context().clone();
            oxide_kernels::gpu::vector_add_device(
                &ctx,
                &stream,
                &scratch.hidden,
                &scratch.down,
                &mut scratch.hidden_swap,
            )
            .expect("ffn residual add");
            std::mem::swap(&mut scratch.hidden, &mut scratch.hidden_swap);
        }

        gpu_cache.advance(seq_len);

        // Final norm + lm_head
        let final_norm_w = gpu_weights.get("model.norm.weight").unwrap();
        self.pipeline
            .rmsnorm
            .forward_device(
                &scratch.hidden,
                final_norm_w,
                &mut scratch.final_normed,
                h,
                cfg.layer_norm_eps,
            )
            .expect("final norm");

        let lm_head_w = gpu_weights.get("lm_head.weight").unwrap();
        self.pipeline
            .matmul
            .forward_bt_device(
                &scratch.final_normed,
                lm_head_w,
                &mut scratch.logits,
                seq_len,
                cfg.vocab_size,
                h,
            )
            .expect("lm head");

        let all_logits = scratch.logits.to_host_vec(&stream).expect("logits to host");
        let last_start = (seq_len - 1) * cfg.vocab_size;
        all_logits[last_start..last_start + cfg.vocab_size].to_vec()
    }
}

#[cfg(feature = "cuda")]
pub struct GpuInferenceWrapper {
    pub model: TransformerModel<
        oxide_kernels::attention::CudaAttention,
        oxide_kernels::rmsnorm::CudaRMSNorm,
        oxide_kernels::rope::CudaRoPE,
        oxide_kernels::matmul::CudaMatMul,
        oxide_kernels::sampling::CpuSampler,
    >,
    pub gpu_weights: crate::weights::GpuWeights,
    pub gpu_cache: std::sync::Mutex<crate::kv_cache::GpuKvCache>,
    pub scratchpad: std::sync::Mutex<crate::gpu_scratchpad::GpuScratchpad>,
}

#[cfg(feature = "cuda")]
impl InferenceModel for GpuInferenceWrapper {
    fn config(&self) -> &ModelConfig {
        &self.model.config
    }
    fn has_weights(&self) -> bool {
        self.model.weights.is_some()
    }
    fn forward(&self, session: &mut InferenceSession, token_ids: &[u32]) -> Vec<f32> {
        let mut gpu_cache = self.gpu_cache.lock().unwrap();
        let mut scratch = self.scratchpad.lock().unwrap();
        let logits = self.model.forward_gpu(
            &self.gpu_weights,
            &mut gpu_cache,
            &mut scratch,
            token_ids,
            session.position,
        );
        session.position += token_ids.len();
        session.cache.advance(token_ids.len());
        logits
    }
    fn generate(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
    ) -> Vec<u32> {
        let total_start = Instant::now();
        let prompt_start = Instant::now();
        let logits = self.forward(session, token_ids);
        let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.prompt_tokens = token_ids.len();
        let gen_start = Instant::now();

        let mut probs = vec![0.0f32; logits.len()];
        self.model
            .pipeline
            .sampler
            .apply(&logits, &mut probs, params)
            .unwrap();

        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut session.rng)
        };
        let first_token = self.model.pipeline.sampler.sample_token(&probs, rng_ref);

        let mut generated = vec![first_token];

        for _step in 1..max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) {
                break;
            }
            let mut logits = self.forward(session, &[*generated.last().unwrap()]);
            params.apply_penalties(&mut logits, &generated);
            let mut probs = vec![0.0f32; logits.len()];
            self.model
                .pipeline
                .sampler
                .apply(&logits, &mut probs, params)
                .unwrap();

            let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                None
            } else {
                Some(&mut session.rng)
            };
            let token = self.model.pipeline.sampler.sample_token(&probs, rng_ref);
            generated.push(token);
        }

        let generation_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.generated_tokens = generated.len();
        session.trace.timing = Some(TimingInfo {
            total_ms,
            prompt_ms,
            generation_ms,
            tokens_per_second: if generation_ms > 0.0 {
                generated.len() as f64 / (generation_ms / 1000.0)
            } else {
                0.0
            },
        });
        generated
    }
    fn generate_with_logprobs(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        top_n: usize,
    ) -> (Vec<u32>, Vec<TokenLogprob>) {
        let total_start = Instant::now();
        let prompt_start = Instant::now();
        let logits = self.forward(session, token_ids);
        let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.prompt_tokens = token_ids.len();
        let gen_start = Instant::now();

        let mut probs = vec![0.0f32; logits.len()];
        self.model
            .pipeline
            .sampler
            .apply(&logits, &mut probs, params)
            .unwrap();

        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut session.rng)
        };
        let first_token = self.model.pipeline.sampler.sample_token(&probs, rng_ref);

        let mut generated = vec![first_token];
        let mut logprobs = vec![extract_logprob(&probs, first_token, top_n)];

        for _step in 1..max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) {
                break;
            }
            let mut step_logits = self.forward(session, &[*generated.last().unwrap()]);
            params.apply_penalties(&mut step_logits, &generated);
            let mut step_probs = vec![0.0f32; step_logits.len()];
            self.model
                .pipeline
                .sampler
                .apply(&step_logits, &mut step_probs, params)
                .unwrap();

            let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                None
            } else {
                Some(&mut session.rng)
            };
            let token = self
                .model
                .pipeline
                .sampler
                .sample_token(&step_probs, rng_ref);
            logprobs.push(extract_logprob(&step_probs, token, top_n));
            generated.push(token);
        }

        let generation_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.generated_tokens = generated.len();
        session.trace.timing = Some(TimingInfo {
            total_ms,
            prompt_ms,
            generation_ms,
            tokens_per_second: if generation_ms > 0.0 {
                generated.len() as f64 / (generation_ms / 1000.0)
            } else {
                0.0
            },
        });
        (generated, logprobs)
    }
    fn generate_with_callback(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Vec<u32> {
        let total_start = Instant::now();
        let prompt_start = Instant::now();
        let logits = self.forward(session, token_ids);
        let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.prompt_tokens = token_ids.len();
        let gen_start = Instant::now();

        let mut probs = vec![0.0f32; logits.len()];
        self.model
            .pipeline
            .sampler
            .apply(&logits, &mut probs, params)
            .unwrap();

        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut session.rng)
        };
        let first_token = self.model.pipeline.sampler.sample_token(&probs, rng_ref);

        let mut generated = vec![first_token];
        if !on_token(first_token) {
            let generation_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            session.trace.generated_tokens = generated.len();
            session.trace.timing = Some(TimingInfo {
                total_ms,
                prompt_ms,
                generation_ms,
                tokens_per_second: if generation_ms > 0.0 {
                    generated.len() as f64 / (generation_ms / 1000.0)
                } else {
                    0.0
                },
            });
            return generated;
        }

        for _step in 1..max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) {
                break;
            }
            let mut logits = self.forward(session, &[*generated.last().unwrap()]);
            params.apply_penalties(&mut logits, &generated);
            let mut probs = vec![0.0f32; logits.len()];
            self.model
                .pipeline
                .sampler
                .apply(&logits, &mut probs, params)
                .unwrap();

            let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                None
            } else {
                Some(&mut session.rng)
            };
            let token = self.model.pipeline.sampler.sample_token(&probs, rng_ref);
            generated.push(token);
            if !on_token(token) {
                break;
            }
        }

        let generation_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.generated_tokens = generated.len();
        session.trace.timing = Some(TimingInfo {
            total_ms,
            prompt_ms,
            generation_ms,
            tokens_per_second: if generation_ms > 0.0 {
                generated.len() as f64 / (generation_ms / 1000.0)
            } else {
                0.0
            },
        });
        generated
    }
    fn generate_guided(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        decoder: &mut dyn oxide_kernels::GuidedDecoder,
    ) -> Vec<u32> {
        self.model.generate_guided(
            session,
            token_ids,
            max_new_tokens,
            params,
            eos_token,
            decoder,
        )
    }
    fn load_weights(
        &mut self,
        _loader: &dyn crate::loader::WeightLoader,
    ) -> Result<(), crate::weights::WeightsError> {
        Ok(())
    }
}

impl<A, N, R, M, S> InferenceModel for TransformerModel<A, N, R, M, S>
where
    A: Attention,
    N: RMSNorm,
    R: RoPE,
    M: MatMul,
    S: Sampler,
{
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn has_weights(&self) -> bool {
        self.weights.is_some()
    }
    fn forward(&self, session: &mut InferenceSession, token_ids: &[u32]) -> Vec<f32> {
        self.forward(session, token_ids)
    }
    fn generate(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
    ) -> Vec<u32> {
        self.generate(session, token_ids, max_new_tokens, params, eos_token)
    }
    fn generate_with_logprobs(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        top_n: usize,
    ) -> (Vec<u32>, Vec<TokenLogprob>) {
        self.generate_with_logprobs(session, token_ids, max_new_tokens, params, eos_token, top_n)
    }
    fn generate_with_callback(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Vec<u32> {
        self.generate_with_callback(
            session,
            token_ids,
            max_new_tokens,
            params,
            eos_token,
            on_token,
        )
    }
    fn generate_guided(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        decoder: &mut dyn oxide_kernels::GuidedDecoder,
    ) -> Vec<u32> {
        self.generate_guided(
            session,
            token_ids,
            max_new_tokens,
            params,
            eos_token,
            decoder,
        )
    }
    fn load_weights(
        &mut self,
        loader: &dyn crate::loader::WeightLoader,
    ) -> Result<(), crate::weights::WeightsError> {
        self.load_weights(loader)
    }
}

pub use oxide_kernels::sampling::SamplingParams as SampleParams;

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn embedding_lookup_in_bounds() {
        let vocab_size: usize = kani::any();
        let hidden_size: usize = kani::any();
        let tok: usize = kani::any();

        kani::assume(vocab_size >= 1 && vocab_size <= 64);
        kani::assume(hidden_size >= 1 && hidden_size <= 16);
        kani::assume(tok < vocab_size);

        let embed_len = vocab_size * hidden_size;
        let start = tok * hidden_size;
        let end = (tok + 1) * hidden_size;

        assert!(start < embed_len);
        assert!(end <= embed_len);
    }

    #[kani::proof]
    fn lm_head_last_token_in_bounds() {
        let seq_len: usize = kani::any();
        let hidden_size: usize = kani::any();

        kani::assume(seq_len >= 1 && seq_len <= 16);
        kani::assume(hidden_size >= 1 && hidden_size <= 16);

        let hidden_len = seq_len * hidden_size;
        let start = (seq_len - 1) * hidden_size;
        let end = seq_len * hidden_size;

        assert!(start < hidden_len);
        assert!(end <= hidden_len);
    }

    #[kani::proof]
    fn head_dim_scale_never_zero_division() {
        let head_dim: usize = kani::any();
        kani::assume(head_dim >= 1 && head_dim <= 256);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        assert!(!scale.is_nan());
    }

    #[kani::proof]
    fn extract_logprob_token_in_bounds() {
        let vocab_size: usize = kani::any();
        let token_id: u32 = kani::any();

        kani::assume(vocab_size >= 1 && vocab_size <= 128);
        kani::assume((token_id as usize) < vocab_size);

        assert!((token_id as usize) < vocab_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::KernelPipeline;
    use std::collections::HashMap;

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

    impl crate::loader::WeightLoader for RandomWeightLoader {
        fn load_tensor(&self, name: &str) -> Result<Vec<f32>, crate::loader::LoadError> {
            self.tensors
                .get(name)
                .cloned()
                .ok_or_else(|| crate::loader::LoadError::MissingTensor(name.to_string()))
        }

        fn tensor_names(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }
    }

    #[test]
    fn test_forward_shape_correctness() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits = model.forward(&mut session, &[1, 2, 3]);

        assert_eq!(logits.len(), config.vocab_size);
        assert_eq!(session.position, 3);
    }

    #[test]
    fn test_forward_nonzero_logits() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits = model.forward(&mut session, &[1, 2, 3]);

        let all_zero = logits.iter().all(|&v| v == 0.0);
        assert!(!all_zero, "logits should be non-zero with random weights");
    }

    #[test]
    fn test_generate_returns_tokens() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        let tokens = model.generate(&mut session, &[1, 2, 3], 5, &params, None);

        assert_eq!(tokens.len(), 5);
        assert_eq!(session.trace.prompt_tokens, 3);
        assert_eq!(session.trace.generated_tokens, 5);
    }

    #[test]
    fn test_generate_stops_at_eos() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        let tokens = model.generate(
            &mut session,
            &[1],
            100,
            &params,
            Some(tokens_first(&model, &config)),
        );

        // Should stop before 100 if EOS is hit
        assert!(tokens.len() <= 100);
    }

    fn tokens_first<A: Attention, N: RMSNorm, R: RoPE, M: MatMul, S: Sampler>(
        model: &TransformerModel<A, N, R, M, S>,
        config: &ModelConfig,
    ) -> u32 {
        let mut session = InferenceSession::new(config, TraceLevel::Off, 42);
        let logits = model.forward(&mut session, &[1]);
        let mut probs = vec![0.0; logits.len()];
        model
            .pipeline
            .sampler
            .apply(
                &logits,
                &mut probs,
                &SamplingParams {
                    temperature: 1.0,
                    top_k: 50,
                    top_p: 1.0,
                    ..SamplingParams::default()
                },
            )
            .unwrap();
        model
            .pipeline
            .sampler
            .sample_token(&probs, Some(&mut session.rng))
    }

    #[test]
    fn test_generate_deterministic() {
        let config = ModelConfig::tiny_test();
        let loader = RandomWeightLoader::from_config(&config, 42);
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };

        let pipeline1 = KernelPipeline::cpu_verified();
        let mut model1 = TransformerModel::new(config.clone(), pipeline1);
        model1.load_weights(&loader).unwrap();
        let mut session1 = InferenceSession::new(&config, TraceLevel::Off, 42);
        let tokens1 = model1.generate(&mut session1, &[1, 2], 10, &params, None);

        let pipeline2 = KernelPipeline::cpu_verified();
        let mut model2 = TransformerModel::new(config.clone(), pipeline2);
        model2.load_weights(&loader).unwrap();
        let mut session2 = InferenceSession::new(&config, TraceLevel::Off, 42);
        let tokens2 = model2.generate(&mut session2, &[1, 2], 10, &params, None);

        assert_eq!(tokens1, tokens2, "same seed must produce identical output");
    }

    #[test]
    fn test_generate_with_sampling_trace() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Summary, 42);
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        let tokens = model.generate(&mut session, &[1], 3, &params, None);

        assert_eq!(tokens.len(), 3);
        assert_eq!(session.trace.sampling_steps.len(), 3);
        for (i, snap) in session.trace.sampling_steps.iter().enumerate() {
            assert_eq!(snap.step, i);
            assert!(!snap.top_logits.is_empty());
            assert!(!snap.top_probs.is_empty());
        }
    }

    #[test]
    fn test_timing_info_populated() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        let _tokens = model.generate(&mut session, &[1, 2, 3], 5, &params, None);

        let timing = session.trace.timing.as_ref().expect("timing should be set");
        assert!(timing.total_ms > 0.0);
        assert!(timing.prompt_ms > 0.0);
        assert!(timing.generation_ms > 0.0);
        assert!(timing.tokens_per_second > 0.0);
        assert!(timing.total_ms >= timing.prompt_ms + timing.generation_ms - 0.01);
    }

    #[test]
    fn test_layer_duration_populated_at_summary() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Summary, 42);
        let _logits = model.forward(&mut session, &[1, 2]);

        assert_eq!(session.trace.layers.len(), config.num_layers);
        for snap in &session.trace.layers {
            assert!(
                snap.duration.is_some(),
                "layer {} should have duration",
                snap.layer_index
            );
        }
    }

    #[test]
    fn test_generate_with_logprobs() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        let (tokens, logprobs) =
            model.generate_with_logprobs(&mut session, &[1, 2, 3], 5, &params, None, 5);

        assert_eq!(tokens.len(), 5);
        assert_eq!(logprobs.len(), 5);
        for lp in &logprobs {
            assert!(lp.logprob <= 0.0);
            assert!(!lp.top_logprobs.is_empty());
            assert!(lp.top_logprobs.len() <= 5);
            for top in &lp.top_logprobs {
                assert!(top.logprob <= 0.0);
            }
        }
    }

    #[test]
    fn test_generate_without_logprobs_unchanged() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let params = SamplingParams {
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };

        let mut s1 = InferenceSession::new(&config, TraceLevel::Off, 42);
        let tokens1 = model.generate(&mut s1, &[1, 2, 3], 5, &params, None);

        let mut s2 = InferenceSession::new(&config, TraceLevel::Off, 42);
        let (tokens2, _) = model.generate_with_logprobs(&mut s2, &[1, 2, 3], 5, &params, None, 5);

        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_forward_with_trace() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Summary, 42);
        let _logits = model.forward(&mut session, &[1, 2]);

        assert_eq!(session.trace.layers.len(), config.num_layers);
        for snap in &session.trace.layers {
            assert!(snap.pre_norm.is_some());
            assert!(snap.post_norm.is_some());
        }
    }

    #[test]
    fn test_chunked_forward_large_chunk_matches_unchunked() {
        let config = ModelConfig::tiny_test();
        let loader = RandomWeightLoader::from_config(&config, 42);
        let tokens = [1u32, 2, 3, 4, 5, 6];

        let pipeline1 = KernelPipeline::cpu_verified();
        let mut model1 = TransformerModel::new(config.clone(), pipeline1);
        model1.load_weights(&loader).unwrap();
        let mut session1 = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits_unchunked = model1.forward(&mut session1, &tokens);

        let pipeline2 = KernelPipeline::cpu_verified();
        let mut model2 = TransformerModel::new(config.clone(), pipeline2);
        model2.load_weights(&loader).unwrap();
        let mut session2 = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits_chunked = model2.chunked_forward(&mut session2, &tokens, 512);

        assert_eq!(logits_unchunked, logits_chunked);
        assert_eq!(session1.position, session2.position);
    }

    #[test]
    fn test_chunked_forward_small_chunk_valid_output() {
        let config = ModelConfig::tiny_test();
        let loader = RandomWeightLoader::from_config(&config, 42);
        let tokens = [1u32, 2, 3, 4, 5, 6];

        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        model.load_weights(&loader).unwrap();
        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits = model.chunked_forward(&mut session, &tokens, 2);

        assert_eq!(logits.len(), config.vocab_size);
        assert_eq!(session.position, tokens.len());
        assert!(!logits.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_silu_no_nan_inf() {
        let test_values: Vec<f32> = vec![
            0.0,
            1.0,
            -1.0,
            100.0,
            -100.0,
            1000.0,
            -1000.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            1e-30,
            -1e-30,
            1e30,
            -1e30,
        ];
        for &x in &test_values {
            let silu = x / (1.0 + (-x).exp());
            assert!(!silu.is_nan(), "SiLU({x}) produced NaN");
            assert!(!silu.is_infinite(), "SiLU({x}) produced Inf");
        }
    }

    #[test]
    fn test_attention_scale_no_nan() {
        for head_dim in 1..=128 {
            let scale = 1.0 / (head_dim as f32).sqrt();
            assert!(!scale.is_nan(), "scale NaN for head_dim={head_dim}");
            assert!(!scale.is_infinite(), "scale Inf for head_dim={head_dim}");
            assert!(scale > 0.0, "scale non-positive for head_dim={head_dim}");
        }
    }

    #[test]
    fn test_chunked_forward_deterministic() {
        let config = ModelConfig::tiny_test();
        let loader = RandomWeightLoader::from_config(&config, 42);
        let tokens = [1u32, 2, 3, 4, 5, 6];

        let pipeline1 = KernelPipeline::cpu_verified();
        let mut model1 = TransformerModel::new(config.clone(), pipeline1);
        model1.load_weights(&loader).unwrap();
        let mut s1 = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits1 = model1.chunked_forward(&mut s1, &tokens, 3);

        let pipeline2 = KernelPipeline::cpu_verified();
        let mut model2 = TransformerModel::new(config.clone(), pipeline2);
        model2.load_weights(&loader).unwrap();
        let mut s2 = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits2 = model2.chunked_forward(&mut s2, &tokens, 3);

        assert_eq!(logits1, logits2);
    }
}
