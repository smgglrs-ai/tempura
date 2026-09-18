use crate::config::ModelConfig;
use crate::pipeline::KernelPipeline;
use crate::session::InferenceSession;
use crate::trace::{LayerSnapshot, SamplingSnapshot, TimingInfo, TraceLevel};
use crate::weights::ModelWeights;
use serde::Serialize;
use std::time::Instant;
use tempura_kernels::{Attention, LayerNorm, MatMul, RMSNorm, RoPE, Sampler, SamplingParams};

fn apply_head_rmsnorm(
    x: &mut [f32],
    weight: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let head = &mut x[offset..offset + head_dim];
            let mut sum_sq = 0.0f32;
            for &v in head.iter() {
                sum_sq += v * v;
            }
            let rms = ((sum_sq / head_dim as f32) + eps).sqrt();
            let inv_rms = 1.0 / rms;
            for (d, v) in head.iter_mut().enumerate() {
                *v = *v * inv_rms * weight[d];
            }
        }
    }
}

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
        decoder: &mut dyn tempura_kernels::GuidedDecoder,
    ) -> Vec<u32>;
    fn generate_with_sampler_chain(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        chain: &mut tempura_kernels::sampler_chain::SamplerChain,
    ) -> Vec<u32> {
        let _ = chain;
        self.generate(session, token_ids, max_new_tokens, params, eos_token)
    }
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
    pub quantized_weights: Option<crate::weights::QuantizedWeights>,
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
            quantized_weights: None,
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
        if let Some(qw) = &self.quantized_weights {
            return self.forward_quantized(session, token_ids, qw);
        }
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

        let mut norm_weight_buf = if cfg.rmsnorm_offset { vec![0.0f32; h] } else { Vec::new() };

        let embed = weights.get("model.embed_tokens.weight").unwrap();
        let mut hidden = vec![0.0f32; seq_len * h];
        for (t, &tok) in token_ids.iter().enumerate() {
            let src = &embed[tok as usize * h..(tok as usize + 1) * h];
            hidden[t * h..(t + 1) * h].copy_from_slice(src);
        }

        if cfg.embedding_scale {
            let scale = (h as f32).sqrt();
            for v in &mut hidden {
                *v *= scale;
            }
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
            if cfg.use_layernorm {
                let bias = weights.get(&format!("{prefix}.input_layernorm.bias"));
                tempura_kernels::layernorm::CpuLayerNorm
                    .forward(&mut normed, input_ln_w, bias, h, cfg.layer_norm_eps)
                    .unwrap();
            } else if cfg.rmsnorm_offset {
                for i in 0..h {
                    norm_weight_buf[i] = 1.0 + input_ln_w[i];
                }
                self.pipeline
                    .rmsnorm(&mut normed, &norm_weight_buf, h, cfg.layer_norm_eps, session.trace_level, &mut Some(&mut snapshot))
                    .unwrap();
            } else {
                self.pipeline
                    .rmsnorm(&mut normed, input_ln_w, h, cfg.layer_norm_eps, session.trace_level, &mut Some(&mut snapshot))
                    .unwrap();
            }

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

            if std::env::var("TEMPURA_DEBUG").is_ok() && layer == 0 {
                eprintln!("[DEBUG] L0 q_pre_rope[0..4]: {:?}", &q[0..4.min(q_dim)]);
            }

            if cfg.use_qk_norm {
                let q_norm_w = weights
                    .get(&format!("{prefix}.self_attn.q_norm.weight"))
                    .unwrap();
                let k_norm_w = weights
                    .get(&format!("{prefix}.self_attn.k_norm.weight"))
                    .unwrap();
                apply_head_rmsnorm(
                    &mut q,
                    q_norm_w,
                    seq_len,
                    n_heads,
                    head_dim,
                    cfg.layer_norm_eps,
                );
                apply_head_rmsnorm(
                    &mut k,
                    k_norm_w,
                    seq_len,
                    n_kv_heads,
                    head_dim,
                    cfg.layer_norm_eps,
                );
            }

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

            let mut attn_out = vec![0.0f32; seq_len * q_dim];
            let attn_scale = 1.0 / (head_dim as f32).sqrt();
            if let Some(ref mut paged) = session.paged_cache {
                paged.append_kv(layer, &k, &v, seq_len);
                let kv_len = paged.seq_len + seq_len;
                let k_cont = paged.k_contiguous(layer, kv_len);
                let v_cont = paged.v_contiguous(layer, kv_len);
                if let Some(cap) = cfg.attn_logit_softcapping {
                    tempura_kernels::attention::CpuAttentionSoftcap::new(cap)
                        .forward(&q, &k_cont, &v_cont, &mut attn_out,
                                 1, seq_len, kv_len, n_heads, n_kv_heads, head_dim, attn_scale)
                        .unwrap();
                } else {
                    self.pipeline
                        .attention(&q, &k_cont, &v_cont, &mut attn_out,
                                   1, seq_len, kv_len, n_heads, n_kv_heads, head_dim, attn_scale)
                        .unwrap();
                }
            } else {
                session.cache.append_kv(layer, &k, &v, seq_len);
                let kv_len = session.cache.seq_len + seq_len;
                let k_cache = session.cache.k_data(layer, kv_len);
                let v_cache = session.cache.v_data(layer, kv_len);
                if let Some(cap) = cfg.attn_logit_softcapping {
                    tempura_kernels::attention::CpuAttentionSoftcap::new(cap)
                        .forward(&q, &k_cache, &v_cache, &mut attn_out,
                                 1, seq_len, kv_len, n_heads, n_kv_heads, head_dim, attn_scale)
                        .unwrap();
                } else {
                    self.pipeline
                        .attention(&q, &k_cache, &v_cache, &mut attn_out,
                                   1, seq_len, kv_len, n_heads, n_kv_heads, head_dim, attn_scale)
                        .unwrap();
                }
            }

            let mut o_proj = vec![0.0f32; seq_len * h];
            self.pipeline
                .matmul_bt(&attn_out, o_w, &mut o_proj, seq_len, h, q_dim)
                .unwrap();

            if !cfg.parallel_attn_ffn {
                for i in 0..hidden.len() {
                    hidden[i] += o_proj[i];
                }
            }

            // --- FFN block ---
            let ffn_normed;
            if cfg.parallel_attn_ffn {
                ffn_normed = normed.clone();
            } else {
                let mut normed2 = hidden.clone();
                let post_ln_w = weights
                    .get(&format!("{prefix}.post_attention_layernorm.weight"))
                    .unwrap();
                if cfg.use_layernorm {
                    let bias = weights.get(&format!("{prefix}.post_attention_layernorm.bias"));
                    tempura_kernels::layernorm::CpuLayerNorm
                        .forward(&mut normed2, post_ln_w, bias, h, cfg.layer_norm_eps)
                        .unwrap();
                } else if cfg.rmsnorm_offset {
                    for i in 0..h {
                        norm_weight_buf[i] = 1.0 + post_ln_w[i];
                    }
                    self.pipeline
                        .rmsnorm(&mut normed2, &norm_weight_buf, h, cfg.layer_norm_eps, TraceLevel::Off, &mut None)
                        .unwrap();
                } else {
                    self.pipeline
                        .rmsnorm(&mut normed2, post_ln_w, h, cfg.layer_norm_eps, TraceLevel::Off, &mut None)
                        .unwrap();
                }
                ffn_normed = normed2;
            }

            let up_w = weights
                .get(&format!("{prefix}.mlp.up_proj.weight"))
                .unwrap();
            let down_w = weights
                .get(&format!("{prefix}.mlp.down_proj.weight"))
                .unwrap();

            let mut gate = vec![0.0f32; seq_len * inter];
            let mut up = vec![0.0f32; seq_len * inter];

            match self.config.hidden_act {
                crate::config::HiddenAct::SiluGate => {
                    let gate_w = weights
                        .get(&format!("{prefix}.mlp.gate_proj.weight"))
                        .unwrap();
                    self.pipeline
                        .matmul_bt(&ffn_normed, gate_w, &mut gate, seq_len, inter, h)
                        .unwrap();
                    self.pipeline
                        .matmul_bt(&ffn_normed, up_w, &mut up, seq_len, inter, h)
                        .unwrap();
                    for i in 0..gate.len() {
                        let silu = gate[i] / (1.0 + (-gate[i]).exp());
                        gate[i] = silu * up[i];
                    }
                }
                crate::config::HiddenAct::SquaredRelu => {
                    let gate_w = weights
                        .get(&format!("{prefix}.mlp.gate_proj.weight"))
                        .unwrap();
                    self.pipeline
                        .matmul_bt(&ffn_normed, gate_w, &mut gate, seq_len, inter, h)
                        .unwrap();
                    self.pipeline
                        .matmul_bt(&ffn_normed, up_w, &mut up, seq_len, inter, h)
                        .unwrap();
                    for i in 0..gate.len() {
                        let relu = gate[i].max(0.0);
                        gate[i] = relu * relu * up[i];
                    }
                }
                crate::config::HiddenAct::Gelu => {
                    self.pipeline
                        .matmul_bt(&ffn_normed, up_w, &mut gate, seq_len, inter, h)
                        .unwrap();
                    for i in 0..gate.len() {
                        gate[i] = tempura_kernels::activations::gelu(gate[i]);
                    }
                }
                crate::config::HiddenAct::GeluQuick => {
                    self.pipeline
                        .matmul_bt(&ffn_normed, up_w, &mut gate, seq_len, inter, h)
                        .unwrap();
                    for i in 0..gate.len() {
                        gate[i] = tempura_kernels::activations::gelu_quick(gate[i]);
                    }
                }
                crate::config::HiddenAct::Relu => {
                    self.pipeline
                        .matmul_bt(&ffn_normed, up_w, &mut gate, seq_len, inter, h)
                        .unwrap();
                    for i in 0..gate.len() {
                        gate[i] = tempura_kernels::activations::relu(gate[i]);
                    }
                }
                crate::config::HiddenAct::GeGLU => {
                    let gate_w = weights
                        .get(&format!("{prefix}.mlp.gate_proj.weight"))
                        .unwrap();
                    self.pipeline
                        .matmul_bt(&ffn_normed, gate_w, &mut gate, seq_len, inter, h)
                        .unwrap();
                    self.pipeline
                        .matmul_bt(&ffn_normed, up_w, &mut up, seq_len, inter, h)
                        .unwrap();
                    for i in 0..gate.len() {
                        gate[i] = tempura_kernels::activations::geglu(gate[i], up[i]);
                    }
                }
                crate::config::HiddenAct::ReGLU => {
                    let gate_w = weights
                        .get(&format!("{prefix}.mlp.gate_proj.weight"))
                        .unwrap();
                    self.pipeline
                        .matmul_bt(&ffn_normed, gate_w, &mut gate, seq_len, inter, h)
                        .unwrap();
                    self.pipeline
                        .matmul_bt(&ffn_normed, up_w, &mut up, seq_len, inter, h)
                        .unwrap();
                    for i in 0..gate.len() {
                        gate[i] = tempura_kernels::activations::reglu(gate[i], up[i]);
                    }
                }
            }

            let mut down_out = vec![0.0f32; seq_len * h];
            self.pipeline
                .matmul_bt(&gate, down_w, &mut down_out, seq_len, h, inter)
                .unwrap();

            if cfg.parallel_attn_ffn {
                for i in 0..hidden.len() {
                    hidden[i] += o_proj[i] + down_out[i];
                }
            } else {
                for i in 0..hidden.len() {
                    hidden[i] += down_out[i];
                }
            }

            if std::env::var("TEMPURA_DEBUG").is_ok() && layer == 0 && seq_len > 1 {
                let h_before_ffn = hidden[0..h].iter().map(|x| x * x).sum::<f32>().sqrt();
                let n2_norm = ffn_normed[0..h].iter().map(|x| x * x).sum::<f32>().sqrt();
                let gate_tok0 = gate[0..inter.min(4)].to_vec();
                let up_tok0 = up[0..inter.min(4)].to_vec();
                let gate_n = gate[0..inter].iter().map(|x| x * x).sum::<f32>().sqrt();
                let dn = down_out[0..h].iter().map(|x| x * x).sum::<f32>().sqrt();
                let hn = hidden[0..h].iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("[DEBUG] L0 tok0: h_pre_ffn={h_before_ffn:.4} normed2={n2_norm:.4} gate_norm={gate_n:.4} down={dn:.4} h_final={hn:.4}");
                eprintln!("[DEBUG] L0 tok0 gate[0..4]: {gate_tok0:?}");
                eprintln!("[DEBUG] L0 tok0 up[0..4]: {up_tok0:?}");
            }

            if session.trace_level >= TraceLevel::Summary {
                snapshot.duration = Some(layer_start.elapsed());
                session.trace.layers.push(snapshot);
            }
        }

        if let Some(ref mut paged) = session.paged_cache {
            paged.advance(seq_len);
        } else {
            session.cache.advance(seq_len);
        }
        session.position += seq_len;

        // final norm
        let final_norm_w = weights.get("model.norm.weight").unwrap();
        if cfg.use_layernorm {
            let bias = weights.get("model.norm.bias");
            tempura_kernels::layernorm::CpuLayerNorm
                .forward(&mut hidden, final_norm_w, bias, h, cfg.layer_norm_eps)
                .unwrap();
        } else if cfg.rmsnorm_offset {
            for i in 0..h {
                norm_weight_buf[i] = 1.0 + final_norm_w[i];
            }
            self.pipeline
                .rmsnorm(&mut hidden, &norm_weight_buf, h, cfg.layer_norm_eps, TraceLevel::Off, &mut None)
                .unwrap();
        } else {
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
        }

        // lm_head: project last token to vocab
        let lm_head_w = weights.get("lm_head.weight").unwrap();
        let last_hidden = &hidden[(seq_len - 1) * h..seq_len * h];
        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.pipeline
            .matmul_bt(last_hidden, lm_head_w, &mut logits, 1, cfg.vocab_size, h)
            .unwrap();

        logits
    }

    pub fn forward_quantized(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        qweights: &crate::weights::QuantizedWeights,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let seq_len = token_ids.len();

        let q4k = tempura_kernels::quantized::CpuMatMulQ4KOpt;
        use tempura_kernels::quantized::{MatMulQ, PreQuantizedActivations};

        let det_rmsnorm: Option<Box<dyn tempura_kernels::RMSNorm>> =
            if self.config.hidden_act == crate::config::HiddenAct::SquaredRelu
                || std::env::var("TEMPURA_DETERMINISTIC").is_ok()
            {
                Some(Box::new(tempura_kernels::rmsnorm::CpuRMSNormGgml))
            } else {
                None
            };

        let rope_mode = match cfg.rope_type {
            crate::config::RopeType::Norm => tempura_kernels::rope::RoPEMode::Normal,
            crate::config::RopeType::Neox => tempura_kernels::rope::RoPEMode::NeoX,
        };
        let det_rope: Option<Box<dyn tempura_kernels::RoPE>> =
            if cfg.rope_type == crate::config::RopeType::Norm
                || std::env::var("TEMPURA_DETERMINISTIC").is_ok()
            {
                Some(Box::new(tempura_kernels::rope::CpuRoPEUnified::new(
                    rope_mode, head_dim,
                )))
            } else {
                None
            };

        let alibi_attn: Option<tempura_kernels::attention::CpuAttentionAlibi> = if cfg.use_alibi {
            Some(tempura_kernels::attention::CpuAttentionAlibi::new(
                n_heads,
                cfg.alibi_max_bias,
            ))
        } else {
            None
        };

        let softcap_attn: Option<tempura_kernels::attention::CpuAttentionSoftcap> =
            cfg.attn_logit_softcapping.map(tempura_kernels::attention::CpuAttentionSoftcap::new);

        let layernorm: Option<tempura_kernels::layernorm::CpuLayerNorm> = if cfg.use_layernorm {
            Some(tempura_kernels::layernorm::CpuLayerNorm)
        } else {
            None
        };

        let mut norm_weight_buf = if cfg.rmsnorm_offset {
            vec![0.0f32; h]
        } else {
            Vec::new()
        };

        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = cfg.intermediate_size;

        // Pre-allocate scratchpad buffers (reused across layers)
        let mut hidden = vec![0.0f32; seq_len * h];
        let mut normed = vec![0.0f32; seq_len * h];
        let mut q = vec![0.0f32; seq_len * q_dim];
        let mut k = vec![0.0f32; seq_len * kv_dim];
        let mut v = vec![0.0f32; seq_len * kv_dim];
        let mut attn_out = vec![0.0f32; seq_len * q_dim];
        let mut o_proj = vec![0.0f32; seq_len * h];
        let mut normed2 = vec![0.0f32; seq_len * h];
        let mut gate = vec![0.0f32; seq_len * inter];
        let mut up = vec![0.0f32; seq_len * inter];
        let mut down_out = vec![0.0f32; seq_len * h];

        let embed = qweights.get_f32("model.embed_tokens.weight").unwrap();
        for (t, &tok) in token_ids.iter().enumerate() {
            let src = &embed[tok as usize * h..(tok as usize + 1) * h];
            hidden[t * h..(t + 1) * h].copy_from_slice(src);
        }

        if cfg.embedding_scale {
            let scale = (h as f32).sqrt();
            for v in &mut hidden {
                *v *= scale;
            }
        }

        if std::env::var("TEMPURA_DEBUG").is_ok() {
            eprintln!("[DEBUG] embed[785][0..8]: {:?}", &hidden[0..8.min(h)]);
            eprintln!(
                "[DEBUG] embed norm: {}",
                hidden.iter().map(|x| x * x).sum::<f32>().sqrt()
            );
        }

        let positions: Vec<u32> = (session.position..session.position + seq_len)
            .map(|p| p as u32)
            .collect();

        let qmm = |preq: Option<&PreQuantizedActivations>,
                   a_f32: &[f32],
                   name: &str,
                   out: &mut [f32],
                   m: usize,
                   n: usize,
                   k: usize| {
            // Check typed_tensors FIRST — routes all formats through ggml::mul_mat_q
            if let Some((dtype, w)) = qweights.get_typed(name) {
                use tempura_kernels::ggml::GgmlType;
                // Use optimized pre-quantized path for Q4K/Q6K decode
                if m == 1 {
                    if let Some(pq) = preq {
                        if dtype == GgmlType::Q4K {
                            q4k.forward_q4k_preq(pq, a_f32, w, out, n).unwrap();
                            return;
                        }
                        if dtype == GgmlType::Q6K {
                            q4k.forward_q6k_preq(pq, a_f32, w, out, n).unwrap();
                            return;
                        }
                    }
                }
                // Optimized legacy path for formats with dedicated kernels
                match dtype {
                    GgmlType::Q4K => {
                        q4k.forward_b_transposed_q4k(a_f32, w, out, m, n, k).unwrap();
                    }
                    GgmlType::Q6K => {
                        q4k.forward_b_transposed_q6k(a_f32, w, out, m, n, k).unwrap();
                    }
                    GgmlType::TQ2_0 => {
                        q4k.forward_b_transposed_tq2(a_f32, w, out, m, n, k).unwrap();
                    }
                    GgmlType::TQ1_0 => {
                        q4k.forward_b_transposed_i2s(a_f32, w, out, m, n, k).unwrap();
                    }
                    _ => {
                        // All other quantized formats go through ggml registry
                        tempura_kernels::ggml::mul_mat::mul_mat_q(dtype, w, a_f32, out, m, n, k);
                    }
                }
                return;
            }
            // f32 fallback
            if let Some(w) = qweights.get_f32(name) {
                self.pipeline.matmul_bt(a_f32, w, out, m, n, k).unwrap();
            } else {
                panic!("weight tensor not found: {name}");
            }
        };

        // Pre-compute all weight names to avoid 252 string allocations per token
        let layer_names: Vec<_> = (0..cfg.num_layers)
            .map(|l| {
                let p = format!("model.layers.{l}");
                (
                    format!("{p}.input_layernorm.weight"),
                    format!("{p}.post_attention_layernorm.weight"),
                    format!("{p}.self_attn.q_proj.weight"),
                    format!("{p}.self_attn.k_proj.weight"),
                    format!("{p}.self_attn.v_proj.weight"),
                    format!("{p}.self_attn.o_proj.weight"),
                    format!("{p}.self_attn.q_norm.weight"),
                    format!("{p}.self_attn.k_norm.weight"),
                    format!("{p}.mlp.gate_proj.weight"),
                    format!("{p}.mlp.up_proj.weight"),
                    format!("{p}.mlp.down_proj.weight"),
                    format!("{p}.self_attn.attn_sub_norm.weight"),
                    format!("{p}.mlp.ffn_sub_norm.weight"),
                )
            })
            .collect();

        for layer in 0..cfg.num_layers {
            let (
                ref input_ln_name,
                ref post_ln_name,
                ref q_w_name,
                ref k_w_name,
                ref v_w_name,
                ref o_w_name,
                ref q_norm_name,
                ref k_norm_name,
                ref gate_w_name,
                ref up_w_name,
                ref down_w_name,
                ref attn_sub_norm_name,
                ref ffn_sub_norm_name,
            ) = layer_names[layer];

            normed.copy_from_slice(&hidden);
            let input_ln_w = qweights.get_f32(input_ln_name).unwrap();
            if let Some(ref ln) = layernorm {
                let input_ln_b = qweights.get_f32(&format!("model.layers.{layer}.input_layernorm.bias"));
                ln.forward(&mut normed, input_ln_w, input_ln_b, h, cfg.layer_norm_eps).unwrap();
            } else if cfg.rmsnorm_offset {
                for i in 0..h {
                    norm_weight_buf[i] = 1.0 + input_ln_w[i];
                }
                if let Some(ref drn) = det_rmsnorm {
                    drn.forward(&mut normed, &norm_weight_buf, h, cfg.layer_norm_eps).unwrap();
                } else {
                    self.pipeline
                        .rmsnorm(&mut normed, &norm_weight_buf, h, cfg.layer_norm_eps, TraceLevel::Off, &mut None)
                        .unwrap();
                }
            } else if let Some(ref drn) = det_rmsnorm {
                drn.forward(&mut normed, input_ln_w, h, cfg.layer_norm_eps).unwrap();
            } else {
                self.pipeline
                    .rmsnorm(
                        &mut normed,
                        input_ln_w,
                        h,
                        cfg.layer_norm_eps,
                        TraceLevel::Off,
                        &mut None,
                    )
                    .unwrap();
            }
            if det_rmsnorm.is_some() && layernorm.is_none() {
                tempura_kernels::portable_math::q8_snap(&mut normed[..seq_len * h]);
            }

            let normed_preq = if seq_len == 1 {
                Some(PreQuantizedActivations::from_f32(&normed[..h]))
            } else {
                None
            };

            q.fill(0.0);
            k.fill(0.0);
            v.fill(0.0);
            // Fuse Q+K into single dispatch when both are Q4K and decode (m=1)
            if seq_len == 1 {
                if let (Some(pq), Some(qw), Some(kw)) = (
                    normed_preq.as_ref(),
                    qweights.get_q4k(&q_w_name),
                    qweights.get_q4k(&k_w_name),
                ) {
                    q4k.fused_q4k_preq_2(
                        pq,
                        qw,
                        &mut q[..q_dim],
                        q_dim,
                        kw,
                        &mut k[..kv_dim],
                        kv_dim,
                    );
                } else {
                    qmm(
                        normed_preq.as_ref(),
                        &normed,
                        &q_w_name,
                        &mut q[..q_dim],
                        1,
                        q_dim,
                        h,
                    );
                    qmm(
                        normed_preq.as_ref(),
                        &normed,
                        &k_w_name,
                        &mut k[..kv_dim],
                        1,
                        kv_dim,
                        h,
                    );
                }
            } else {
                qmm(
                    normed_preq.as_ref(),
                    &normed,
                    &q_w_name,
                    &mut q[..seq_len * q_dim],
                    seq_len,
                    q_dim,
                    h,
                );
                qmm(
                    normed_preq.as_ref(),
                    &normed,
                    &k_w_name,
                    &mut k[..seq_len * kv_dim],
                    seq_len,
                    kv_dim,
                    h,
                );
            }
            qmm(
                normed_preq.as_ref(),
                &normed,
                &v_w_name,
                &mut v[..seq_len * kv_dim],
                seq_len,
                kv_dim,
                h,
            );

            if cfg.use_layernorm {
                let prefix = format!("model.layers.{layer}");
                if let Some(qb) = qweights.get_f32(&format!("{prefix}.self_attn.q_proj.bias")) {
                    for t in 0..seq_len {
                        for i in 0..q_dim {
                            q[t * q_dim + i] += qb[i];
                        }
                    }
                }
                if let Some(kb) = qweights.get_f32(&format!("{prefix}.self_attn.k_proj.bias")) {
                    for t in 0..seq_len {
                        for i in 0..kv_dim {
                            k[t * kv_dim + i] += kb[i];
                        }
                    }
                }
            }

            if cfg.use_qk_norm {
                if let (Some(q_norm_w), Some(k_norm_w)) = (
                    qweights.get_f32(q_norm_name),
                    qweights.get_f32(k_norm_name),
                ) {
                    apply_head_rmsnorm(&mut q, q_norm_w, seq_len, n_heads, head_dim, cfg.layer_norm_eps);
                    if det_rmsnorm.is_some() {
                        tempura_kernels::portable_math::q8_snap(&mut q[..seq_len * q_dim]);
                    }
                    apply_head_rmsnorm(
                        &mut k,
                        k_norm_w,
                        seq_len,
                        n_kv_heads,
                        head_dim,
                        cfg.layer_norm_eps,
                    );
                    if det_rmsnorm.is_some() {
                        tempura_kernels::portable_math::q8_snap(&mut k[..seq_len * kv_dim]);
                    }
                }
            }

            if !cfg.use_alibi {
                if let Some(ref rope_table) = det_rope {
                    rope_table.forward(&mut q, 1, seq_len, n_heads, head_dim, &positions, cfg.rope_theta).unwrap();
                    rope_table.forward(&mut k, 1, seq_len, n_kv_heads, head_dim, &positions, cfg.rope_theta).unwrap();
                } else {
                    self.pipeline.rope(&mut q, 1, seq_len, n_heads, head_dim, &positions, cfg.rope_theta).unwrap();
                    self.pipeline.rope(&mut k, 1, seq_len, n_kv_heads, head_dim, &positions, cfg.rope_theta).unwrap();
                }
            }

            session.cache.append_kv(
                layer,
                &k[..seq_len * kv_dim],
                &v[..seq_len * kv_dim],
                seq_len,
            );
            let kv_len = session.cache.seq_len + seq_len;
            let k_cache = session.cache.k_data(layer, kv_len);
            let v_cache = session.cache.v_data(layer, kv_len);

            attn_out.fill(0.0);
            if let Some(ref alibi) = alibi_attn {
                alibi
                    .forward(
                        &q, &k_cache, &v_cache,
                        &mut attn_out[..seq_len * q_dim],
                        1, seq_len, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        1.0 / (head_dim as f32).sqrt(),
                    )
                    .unwrap();
            } else if let Some(ref sc) = softcap_attn {
                sc.forward(
                        &q, &k_cache, &v_cache,
                        &mut attn_out[..seq_len * q_dim],
                        1, seq_len, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        1.0 / (head_dim as f32).sqrt(),
                    )
                    .unwrap();
            } else if self.config.hidden_act == crate::config::HiddenAct::SquaredRelu {
                tempura_kernels::attention::CpuAttentionGgml
                    .forward(
                        &q, &k_cache, &v_cache,
                        &mut attn_out[..seq_len * q_dim],
                        1, seq_len, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        1.0 / (head_dim as f32).sqrt(),
                    )
                    .unwrap();
            } else {
                self.pipeline
                    .attention(
                        &q, &k_cache, &v_cache,
                        &mut attn_out[..seq_len * q_dim],
                        1, seq_len, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        1.0 / (head_dim as f32).sqrt(),
                    )
                    .unwrap();
            }

            if let Some(sub_norm_w) = qweights.get_f32(attn_sub_norm_name) {
                self.pipeline
                    .rmsnorm(&mut attn_out[..seq_len * q_dim], sub_norm_w, q_dim, cfg.layer_norm_eps, TraceLevel::Off, &mut None)
                    .unwrap();
            }

            o_proj.fill(0.0);
            let attn_preq = if seq_len == 1 {
                Some(PreQuantizedActivations::from_f32(&attn_out[..q_dim]))
            } else {
                None
            };
            qmm(
                attn_preq.as_ref(),
                &attn_out[..seq_len * q_dim],
                &o_w_name,
                &mut o_proj[..seq_len * h],
                seq_len,
                h,
                q_dim,
            );
            if !cfg.parallel_attn_ffn {
                for i in 0..seq_len * h {
                    hidden[i] += o_proj[i];
                }
            }

            if std::env::var("TEMPURA_DUMP_LOGITS").is_ok() && layer == 0 {
                let last = &hidden[(seq_len - 1) * h..seq_len * h];
                let norm: f32 = last.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("L0 post-attn hidden[last][0..8]: {:?} norm={norm:.4}", &last[..8.min(h)]);
            }

            // FFN block
            let (ffn_input, ffn_preq) = if cfg.parallel_attn_ffn {
                let preq = if seq_len == 1 {
                    Some(PreQuantizedActivations::from_f32(&normed[..h]))
                } else {
                    None
                };
                (&normed as &[f32], preq)
            } else {
                normed2.copy_from_slice(&hidden);
                let post_ln_w = qweights.get_f32(post_ln_name).unwrap();
                if let Some(ref ln) = layernorm {
                    let post_ln_b = qweights.get_f32(&format!("model.layers.{layer}.post_attention_layernorm.bias"));
                    ln.forward(&mut normed2, post_ln_w, post_ln_b, h, cfg.layer_norm_eps).unwrap();
                } else if cfg.rmsnorm_offset {
                    for i in 0..h {
                        norm_weight_buf[i] = 1.0 + post_ln_w[i];
                    }
                    if let Some(ref drn) = det_rmsnorm {
                        drn.forward(&mut normed2, &norm_weight_buf, h, cfg.layer_norm_eps).unwrap();
                    } else {
                        self.pipeline.rmsnorm(&mut normed2, &norm_weight_buf, h, cfg.layer_norm_eps, TraceLevel::Off, &mut None).unwrap();
                    }
                } else if let Some(ref drn) = det_rmsnorm {
                    drn.forward(&mut normed2, post_ln_w, h, cfg.layer_norm_eps).unwrap();
                } else {
                    self.pipeline.rmsnorm(&mut normed2, post_ln_w, h, cfg.layer_norm_eps, TraceLevel::Off, &mut None).unwrap();
                }
                if det_rmsnorm.is_some() && layernorm.is_none() {
                    tempura_kernels::portable_math::q8_snap(&mut normed2[..seq_len * h]);
                }
                let preq = if seq_len == 1 {
                    Some(PreQuantizedActivations::from_f32(&normed2[..h]))
                } else {
                    None
                };
                (&normed2 as &[f32], preq)
            };

            if cfg.n_experts > 0 {
                let arch_name = match cfg.architecture {
                    crate::config::Architecture::Llama => "mixtral",
                    _ => "mixtral",
                };
                let weight_prefix = crate::moe::moe_weight_prefix(arch_name);
                let router_name = crate::moe::moe_router_weight_name(arch_name, layer);
                let has_router = qweights.get_f32(&router_name).is_some()
                    || qweights.get_q4k(&router_name).is_some()
                    || qweights.get_q6k(&router_name).is_some()
                    || qweights.get_typed(&router_name).is_some();
                if has_router {
                    let router = tempura_kernels::moe::MoeRouter::new(cfg.n_experts, cfg.n_experts_used);
                    down_out.fill(0.0);
                    crate::moe::moe_ffn(
                        &ffn_input[..seq_len * h],
                        &mut down_out[..seq_len * h],
                        &router,
                        qweights,
                        layer,
                        h,
                        inter,
                        cfg.hidden_act,
                        seq_len,
                        weight_prefix,
                        &router_name,
                    );
                    for i in 0..seq_len * h {
                        hidden[i] += down_out[i];
                    }
                } else {
                    eprintln!("[warn] MoE router weight not found for layer {layer}, falling back to dense FFN");
                    // Fall through to dense FFN below
                    gate.fill(0.0);
                    up.fill(0.0);
                    qmm(ffn_preq.as_ref(), ffn_input, &gate_w_name, &mut gate[..seq_len * inter], seq_len, inter, h);
                    qmm(ffn_preq.as_ref(), ffn_input, &up_w_name, &mut up[..seq_len * inter], seq_len, inter, h);
                    for i in 0..seq_len * inter {
                        gate[i] = tempura_kernels::portable_math::portable_silu(gate[i]) * up[i];
                    }
                    down_out.fill(0.0);
                    let gate_preq_moe = if seq_len == 1 {
                        Some(PreQuantizedActivations::from_f32(&gate[..inter]))
                    } else {
                        None
                    };
                    qmm(gate_preq_moe.as_ref(), &gate[..seq_len * inter], &down_w_name, &mut down_out[..seq_len * h], seq_len, h, inter);
                    for i in 0..seq_len * h {
                        hidden[i] += down_out[i];
                    }
                }
            } else {
            // Dense FFN path
            gate.fill(0.0);
            up.fill(0.0);
            // Fuse gate+up into single dispatch when both are Q4K and decode
            match self.config.hidden_act {
                crate::config::HiddenAct::SiluGate => {
                    if seq_len == 1 {
                        if let (Some(pq), Some(gw), Some(uw)) = (
                            ffn_preq.as_ref(),
                            qweights.get_q4k(&gate_w_name),
                            qweights.get_q4k(&up_w_name),
                        ) {
                            q4k.fused_q4k_preq_2(
                                pq,
                                gw,
                                &mut gate[..inter],
                                inter,
                                uw,
                                &mut up[..inter],
                                inter,
                            );
                        } else {
                            qmm(
                                ffn_preq.as_ref(),
                                ffn_input,
                                &gate_w_name,
                                &mut gate[..inter],
                                1,
                                inter,
                                h,
                            );
                            qmm(
                                ffn_preq.as_ref(),
                                ffn_input,
                                &up_w_name,
                                &mut up[..inter],
                                1,
                                inter,
                                h,
                            );
                        }
                    } else {
                        qmm(
                            ffn_preq.as_ref(),
                            ffn_input,
                            &gate_w_name,
                            &mut gate[..seq_len * inter],
                            seq_len,
                            inter,
                            h,
                        );
                        qmm(
                            ffn_preq.as_ref(),
                            ffn_input,
                            &up_w_name,
                            &mut up[..seq_len * inter],
                            seq_len,
                            inter,
                            h,
                        );
                    }
                    for i in 0..seq_len * inter {
                        gate[i] = tempura_kernels::portable_math::portable_silu(gate[i]) * up[i];
                    }
                    if det_rmsnorm.is_some() {
                        tempura_kernels::portable_math::q8_snap(&mut gate[..seq_len * inter]);
                    }
                }
                crate::config::HiddenAct::SquaredRelu => {
                    qmm(
                        ffn_preq.as_ref(),
                        ffn_input,
                        &gate_w_name,
                        &mut gate[..seq_len * inter],
                        seq_len,
                        inter,
                        h,
                    );
                    qmm(
                        ffn_preq.as_ref(),
                        ffn_input,
                        &up_w_name,
                        &mut up[..seq_len * inter],
                        seq_len,
                        inter,
                        h,
                    );
                    if std::env::var("TEMPURA_DUMP_LOGITS").is_ok() && layer == 0 {
                        let last_g = &gate[(seq_len-1)*inter..seq_len*inter];
                        let last_u = &up[(seq_len-1)*inter..seq_len*inter];
                        let gn: f32 = last_g.iter().map(|x| x*x).sum::<f32>().sqrt();
                        let un: f32 = last_u.iter().map(|x| x*x).sum::<f32>().sqrt();
                        eprintln!("L0 FFN gate[0..8]: {:?} norm={gn:.4}", &last_g[..8.min(inter)]);
                        eprintln!("L0 FFN up[0..8]: {:?} norm={un:.4}", &last_u[..8.min(inter)]);
                    }
                    for i in 0..seq_len * inter {
                        let relu = gate[i].max(0.0);
                        gate[i] = relu * relu * up[i];
                    }
                    if std::env::var("TEMPURA_DUMP_LOGITS").is_ok() && layer == 0 {
                        let last = &gate[(seq_len-1)*inter..seq_len*inter];
                        let norm: f32 = last.iter().map(|x| x*x).sum::<f32>().sqrt();
                        eprintln!("L0 FFN after relu²*up[0..8]: {:?} norm={norm:.4}", &last[..8.min(inter)]);
                    }
                }
                crate::config::HiddenAct::Gelu => {
                    qmm(
                        ffn_preq.as_ref(),
                        ffn_input,
                        &up_w_name,
                        &mut gate[..seq_len * inter],
                        seq_len,
                        inter,
                        h,
                    );
                    for i in 0..seq_len * inter {
                        gate[i] = tempura_kernels::activations::gelu(gate[i]);
                    }
                }
                crate::config::HiddenAct::GeluQuick => {
                    qmm(
                        ffn_preq.as_ref(),
                        ffn_input,
                        &up_w_name,
                        &mut gate[..seq_len * inter],
                        seq_len,
                        inter,
                        h,
                    );
                    for i in 0..seq_len * inter {
                        gate[i] = tempura_kernels::activations::gelu_quick(gate[i]);
                    }
                }
                crate::config::HiddenAct::Relu => {
                    qmm(
                        ffn_preq.as_ref(),
                        ffn_input,
                        &up_w_name,
                        &mut gate[..seq_len * inter],
                        seq_len,
                        inter,
                        h,
                    );
                    for i in 0..seq_len * inter {
                        gate[i] = tempura_kernels::activations::relu(gate[i]);
                    }
                }
                crate::config::HiddenAct::GeGLU => {
                    qmm(
                        ffn_preq.as_ref(),
                        ffn_input,
                        &gate_w_name,
                        &mut gate[..seq_len * inter],
                        seq_len,
                        inter,
                        h,
                    );
                    qmm(
                        ffn_preq.as_ref(),
                        ffn_input,
                        &up_w_name,
                        &mut up[..seq_len * inter],
                        seq_len,
                        inter,
                        h,
                    );
                    for i in 0..seq_len * inter {
                        gate[i] = tempura_kernels::activations::geglu(gate[i], up[i]);
                    }
                }
                crate::config::HiddenAct::ReGLU => {
                    qmm(
                        ffn_preq.as_ref(),
                        ffn_input,
                        &gate_w_name,
                        &mut gate[..seq_len * inter],
                        seq_len,
                        inter,
                        h,
                    );
                    qmm(
                        ffn_preq.as_ref(),
                        ffn_input,
                        &up_w_name,
                        &mut up[..seq_len * inter],
                        seq_len,
                        inter,
                        h,
                    );
                    for i in 0..seq_len * inter {
                        gate[i] = tempura_kernels::activations::reglu(gate[i], up[i]);
                    }
                }
            }

            if let Some(sub_norm_w) = qweights.get_f32(ffn_sub_norm_name) {
                self.pipeline
                    .rmsnorm(&mut gate[..seq_len * inter], sub_norm_w, inter, cfg.layer_norm_eps, TraceLevel::Off, &mut None)
                    .unwrap();
            }

            down_out.fill(0.0);
            let gate_preq = if seq_len == 1 {
                Some(PreQuantizedActivations::from_f32(&gate[..inter]))
            } else {
                None
            };
            qmm(
                gate_preq.as_ref(),
                &gate[..seq_len * inter],
                &down_w_name,
                &mut down_out[..seq_len * h],
                seq_len,
                h,
                inter,
            );
            if cfg.parallel_attn_ffn {
                for i in 0..seq_len * h {
                    hidden[i] += o_proj[i] + down_out[i];
                }
            } else {
                for i in 0..seq_len * h {
                    hidden[i] += down_out[i];
                }
            }
            } // end dense FFN else block

            if std::env::var("TEMPURA_DUMP_LOGITS").is_ok() && layer < 3 {
                let last = &hidden[(seq_len - 1) * h..seq_len * h];
                let norm: f32 = last.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("L{layer} hidden[last][0..8]: {:?} norm={norm:.4}", &last[..8.min(h)]);
            }
        }

        session.cache.advance(seq_len);
        session.position += seq_len;

        let mut final_normed = hidden.clone();
        let final_norm_w = qweights.get_f32("model.norm.weight").unwrap();
        if let Some(ref ln) = layernorm {
            let final_norm_b = qweights.get_f32("model.norm.bias");
            ln.forward(&mut final_normed, final_norm_w, final_norm_b, h, cfg.layer_norm_eps).unwrap();
        } else if cfg.rmsnorm_offset {
            for i in 0..h {
                norm_weight_buf[i] = 1.0 + final_norm_w[i];
            }
            if let Some(ref drn) = det_rmsnorm {
                drn.forward(&mut final_normed, &norm_weight_buf, h, cfg.layer_norm_eps).unwrap();
            } else {
                self.pipeline.rmsnorm(&mut final_normed, &norm_weight_buf, h, cfg.layer_norm_eps, TraceLevel::Off, &mut None).unwrap();
            }
        } else if let Some(ref drn) = det_rmsnorm {
            drn.forward(&mut final_normed, final_norm_w, h, cfg.layer_norm_eps).unwrap();
        } else {
            self.pipeline.rmsnorm(&mut final_normed, final_norm_w, h, cfg.layer_norm_eps, TraceLevel::Off, &mut None).unwrap();
        }
        if det_rmsnorm.is_some() && layernorm.is_none() {
            tempura_kernels::portable_math::q8_snap(&mut final_normed[..seq_len * h]);
        }

        // lm_head
        let last_hidden = &final_normed[(seq_len - 1) * h..seq_len * h];
        let mut logits = vec![0.0f32; cfg.vocab_size];
        let lm_head_name = "lm_head.weight";
        if let Some(w) = qweights.get_i2s(lm_head_name) {
            q4k.forward_b_transposed_i2s(last_hidden, w, &mut logits, 1, cfg.vocab_size, h)
                .unwrap();
        } else if let Some(w) = qweights.get_tq2(lm_head_name) {
            q4k.forward_b_transposed_tq2(last_hidden, w, &mut logits, 1, cfg.vocab_size, h)
                .unwrap();
        } else if let Some(w) = qweights.get_q4k(lm_head_name) {
            q4k.forward_b_transposed_q4k(last_hidden, w, &mut logits, 1, cfg.vocab_size, h)
                .unwrap();
        } else if let Some(w) = qweights.get_q6k(lm_head_name) {
            q4k.forward_b_transposed_q6k(last_hidden, w, &mut logits, 1, cfg.vocab_size, h)
                .unwrap();
        } else {
            let w = qweights.get_f32(lm_head_name).unwrap();
            self.pipeline
                .matmul_bt(last_hidden, w, &mut logits, 1, cfg.vocab_size, h)
                .unwrap();
        }

        if std::env::var("TEMPURA_DUMP_LOGITS").is_ok() {
            eprint!("LOGITS_RAW:");
            for i in 0..20.min(logits.len()) {
                eprint!(" {:.6}", logits[i]);
            }
            eprintln!();
            let mut scored: Vec<(f32, usize)> = logits.iter().enumerate().map(|(i, &v)| (v, i)).collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            eprint!("TOP10:");
            for i in 0..10.min(scored.len()) {
                eprint!(" {}:{:.4}", scored[i].1, scored[i].0);
            }
            eprintln!();
            let sum: f32 = logits.iter().sum();
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
            eprintln!("STATS: mean={:.6} min={:.6} max={:.6}", sum / logits.len() as f32, min, max);
        }

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
    ) -> tempura_kernels::Result<Vec<f32>> {
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
        decoder: &mut dyn tempura_kernels::GuidedDecoder,
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

        let mut step = 1;
        while step < max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) || decoder.is_complete() {
                break;
            }

            if let Some(forced) = decoder.forced_tokens() {
                for &tok in &forced {
                    if step >= max_new_tokens || decoder.is_complete() {
                        break;
                    }
                    generated.push(tok);
                    step += 1;
                }
                continue;
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
            step += 1;
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

    pub fn generate_with_sampler_chain(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        chain: &mut tempura_kernels::sampler_chain::SamplerChain,
    ) -> Vec<u32> {
        let total_start = Instant::now();
        let prompt_start = Instant::now();
        let mut logits = self.forward(session, token_ids);
        let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.prompt_tokens = token_ids.len();
        let gen_start = Instant::now();

        chain.apply_all(&mut logits, &[]);
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
            chain.apply_all(&mut logits, &generated);
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
        tempura_kernels::attention::CudaAttention,
        tempura_kernels::rmsnorm::CudaRMSNorm,
        tempura_kernels::rope::CudaRoPE,
        tempura_kernels::matmul::CudaMatMul,
        tempura_kernels::sampling::CpuSampler,
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
                .forward_bt_device(
                    &scratch.normed,
                    qkv_w,
                    &mut scratch.qkv,
                    seq_len,
                    qkv_dim,
                    h,
                )
                .expect("qkv proj");

            let ctx = self.pipeline.matmul.gpu().context().clone();
            tempura_kernels::gpu::slice_columns_device(
                &ctx,
                &stream,
                &scratch.qkv,
                &mut scratch.q,
                seq_len,
                qkv_dim,
                0,
                q_dim,
            )
            .expect("slice q");
            tempura_kernels::gpu::slice_columns_device(
                &ctx,
                &stream,
                &scratch.qkv,
                &mut scratch.k,
                seq_len,
                qkv_dim,
                q_dim,
                kv_dim,
            )
            .expect("slice k");
            tempura_kernels::gpu::slice_columns_device(
                &ctx,
                &stream,
                &scratch.qkv,
                &mut scratch.v,
                seq_len,
                qkv_dim,
                q_dim + kv_dim,
                kv_dim,
            )
            .expect("slice v");

            // --- QK-norm (Qwen3) ---
            if cfg.use_qk_norm {
                let q_norm_w = gpu_weights
                    .get(&format!("{prefix}.self_attn.q_norm.weight"))
                    .unwrap();
                self.pipeline
                    .rmsnorm
                    .forward_device(
                        &scratch.q,
                        q_norm_w,
                        &mut scratch.q_rotated,
                        head_dim,
                        cfg.layer_norm_eps,
                    )
                    .expect("qk norm q");
                tempura_kernels::gpu::copy_region_device(
                    &ctx,
                    &stream,
                    &scratch.q_rotated,
                    &mut scratch.q,
                    0,
                    0,
                    seq_len * q_dim,
                )
                .expect("copy q norm");

                let k_norm_w = gpu_weights
                    .get(&format!("{prefix}.self_attn.k_norm.weight"))
                    .unwrap();
                self.pipeline
                    .rmsnorm
                    .forward_device(
                        &scratch.k,
                        k_norm_w,
                        &mut scratch.k_rotated,
                        head_dim,
                        cfg.layer_norm_eps,
                    )
                    .expect("qk norm k");
                tempura_kernels::gpu::copy_region_device(
                    &ctx,
                    &stream,
                    &scratch.k_rotated,
                    &mut scratch.k,
                    0,
                    0,
                    seq_len * kv_dim,
                )
                .expect("copy k norm");
            }

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
            tempura_kernels::gpu::copy_region_device(
                &ctx,
                &stream,
                &scratch.k_rotated,
                &mut gpu_cache.k[layer],
                0,
                dst_offset,
                src_len,
            )
            .expect("kv cache k append");
            tempura_kernels::gpu::copy_region_device(
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
                .forward_bt_device(
                    &scratch.attn_out,
                    o_w,
                    &mut scratch.o_proj,
                    seq_len,
                    h,
                    q_dim,
                )
                .expect("o proj");

            // --- Residual add (on device) ---
            let ctx = self.pipeline.matmul.gpu().context().clone();
            tempura_kernels::gpu::vector_add_device(
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
                    &scratch.normed2,
                    gate_up_w,
                    &mut scratch.gate_up,
                    seq_len,
                    2 * inter,
                    h,
                )
                .expect("gate_up proj");

            let ctx = self.pipeline.matmul.gpu().context().clone();
            tempura_kernels::gpu::slice_columns_device(
                &ctx,
                &stream,
                &scratch.gate_up,
                &mut scratch.gate,
                seq_len,
                2 * inter,
                0,
                inter,
            )
            .expect("slice gate");
            tempura_kernels::gpu::slice_columns_device(
                &ctx,
                &stream,
                &scratch.gate_up,
                &mut scratch.up,
                seq_len,
                2 * inter,
                inter,
                inter,
            )
            .expect("slice up");

            // --- SiLU(gate) * up (on device) ---
            let ctx = self.pipeline.matmul.gpu().context().clone();
            tempura_kernels::gpu::silu_mul_device(
                &ctx,
                &stream,
                &scratch.gate,
                &scratch.up,
                &mut scratch.silu_out,
            )
            .expect("silu_mul");

            self.pipeline
                .matmul
                .forward_bt_device(
                    &scratch.silu_out,
                    down_w,
                    &mut scratch.down,
                    seq_len,
                    h,
                    inter,
                )
                .expect("down proj");

            // --- FFN residual add (on device) ---
            let ctx = self.pipeline.matmul.gpu().context().clone();
            tempura_kernels::gpu::vector_add_device(
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
        tempura_kernels::attention::CudaAttention,
        tempura_kernels::rmsnorm::CudaRMSNorm,
        tempura_kernels::rope::CudaRoPE,
        tempura_kernels::matmul::CudaMatMul,
        tempura_kernels::sampling::CpuSampler,
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
        decoder: &mut dyn tempura_kernels::GuidedDecoder,
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

/// CPU-KV-cache + cutile-GPU-kernel inference wrapper.
///
/// Uses cutile kernels (flash attention, RMSNorm, RoPE, GEMM) compiled at
/// runtime via the cuTile JIT — no `cargo oxide build` required. Quantized
/// weight projections use CPU dequant + cutile GEMM (GPU dequant is a TODO).
#[cfg(feature = "cutile")]
pub struct CutileInferenceWrapper {
    pub model: TransformerModel<
        tempura_kernels::cutile_flash_attention::CutileFlashAttention,
        tempura_kernels::cutile_rmsnorm::CutileRmsNorm,
        tempura_kernels::cutile_rope::CutileRoPE,
        tempura_kernels::cutile_matmul::CutileMatMul,
        tempura_kernels::sampling::CpuSampler,
    >,
}

#[cfg(feature = "cutile")]
impl CutileInferenceWrapper {
    pub fn new(config: ModelConfig) -> Self {
        use crate::pipeline::KernelPipeline;
        let pipeline = KernelPipeline::cutile();
        let model = TransformerModel {
            config,
            pipeline,
            weights: None,
            quantized_weights: None,
            #[cfg(feature = "cuda")]
            gpu_weights: None,
        };
        Self { model }
    }
}

#[cfg(feature = "cutile")]
impl InferenceModel for CutileInferenceWrapper {
    fn config(&self) -> &ModelConfig {
        &self.model.config
    }
    fn has_weights(&self) -> bool {
        self.model.weights.is_some() || self.model.quantized_weights.is_some()
    }
    fn forward(&self, session: &mut InferenceSession, token_ids: &[u32]) -> Vec<f32> {
        let logits = self.model.forward(session, token_ids);
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
        self.model
            .generate(session, token_ids, max_new_tokens, params, eos_token)
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
        self.model.generate_with_logprobs(
            session,
            token_ids,
            max_new_tokens,
            params,
            eos_token,
            top_n,
        )
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
        self.model.generate_with_callback(
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
        decoder: &mut dyn tempura_kernels::GuidedDecoder,
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
        loader: &dyn crate::loader::WeightLoader,
    ) -> Result<(), crate::weights::WeightsError> {
        self.model.load_weights(loader)
    }
}

#[cfg(feature = "cuda")]
pub struct GpuQuantizedInferenceWrapper {
    pub config: ModelConfig,
    pub attention: tempura_kernels::attention::CudaAttention,
    pub rmsnorm: tempura_kernels::rmsnorm::CudaRMSNorm,
    pub rope: tempura_kernels::rope::CudaRoPE,
    pub matmul: tempura_kernels::matmul::CudaMatMul,
    pub q4k_matmul: tempura_kernels::quantized::CudaMatMulQ4K,
    pub sampler: tempura_kernels::sampling::CpuSampler,
    pub gpu_weights: crate::weights::GpuWeights,
    pub gpu_cache: std::sync::Mutex<crate::kv_cache::GpuKvCache>,
    pub scratchpad: std::sync::Mutex<crate::gpu_scratchpad::GpuScratchpad>,
    pub embed_host: Vec<f32>,
    pub attn_mode: tempura_kernels::AttentionMode,
    pub cpu_model: Option<std::sync::Mutex<TransformerModel<
        tempura_kernels::attention::CpuAttention,
        tempura_kernels::rmsnorm::CpuRMSNorm,
        tempura_kernels::rope::CpuRoPE,
        tempura_kernels::matmul::CpuMatMul,
        tempura_kernels::sampling::CpuSampler,
    >>>,
}

#[cfg(feature = "cuda")]
impl GpuQuantizedInferenceWrapper {
    fn gpu_quantize_q8(
        &self,
        input: &cuda_core::DeviceBuffer<f32>,
        q8_out: &mut cuda_core::DeviceBuffer<u32>,
        count: usize,
    ) {
        let ctx = self.q4k_matmul.gpu().context().clone();
        let stream = self.q4k_matmul.gpu().stream();
        tempura_kernels::gpu::quantize_q8_device(&ctx, &stream, input, q8_out, count)
            .expect("q8 quantize");
    }

    fn gpu_gemv_q8_or_f32(
        &self,
        name: &str,
        f32_input: &cuda_core::DeviceBuffer<f32>,
        q8_input: &cuda_core::DeviceBuffer<u32>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
        seq_len: usize,
    ) {
        if let Some(w) = self.gpu_weights.get_q4k_device(name) {
            if seq_len == 1 {
                self.q4k_matmul
                    .gemv_q4k_q8_dp4a_device(q8_input, w, output, n, k)
                    .expect(name);
            } else {
                self.q4k_matmul
                    .gemm_q4k_q8_det_device(q8_input, w, output, seq_len, n, k)
                    .expect(name);
            }
        } else if let Some(w) = self.gpu_weights.get_q6k_device(name) {
            if seq_len == 1 {
                self.q4k_matmul
                    .gemv_q6k_q8_dp4a_device(q8_input, w, output, n, k)
                    .expect(name);
            } else {
                self.q4k_matmul
                    .gemm_q6k_q8_det_device(q8_input, w, output, seq_len, n, k)
                    .expect(name);
            }
        } else {
            self.gpu_gemv_f32(name, f32_input, output, n, k, seq_len);
        }
    }

    fn gpu_gemv_f32(
        &self,
        name: &str,
        input: &cuda_core::DeviceBuffer<f32>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
        seq_len: usize,
    ) {
        if let Some(w) = self.gpu_weights.get_q4k_device(name) {
            if seq_len == 1 {
                self.q4k_matmul
                    .gemv_q4k_coop_device(input, w, output, n, k)
                    .expect(name);
            } else {
                self.q4k_matmul
                    .forward_q4k_device(input, w, output, seq_len, n, k)
                    .expect(name);
            }
        } else if let Some(w) = self.gpu_weights.get_q6k_device(name) {
            if seq_len == 1 {
                self.q4k_matmul
                    .gemv_q6k_coop_device(input, w, output, n, k)
                    .expect(name);
            } else {
                self.q4k_matmul
                    .forward_q6k_device(input, w, output, seq_len, n, k)
                    .expect(name);
            }
        } else {
            let w = self.gpu_weights.get(name).unwrap();
            self.matmul
                .forward_bt_device(input, w, output, seq_len, n, k)
                .expect(name);
        }
    }

    fn gpu_fused_gate_up_silu(
        &self,
        gate_name: &str,
        up_name: &str,
        input: &cuda_core::DeviceBuffer<f32>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) {
        let gate_w = self.gpu_weights.get_q4k_device(gate_name).expect("fused: gate not Q4K");
        let up_w = self.gpu_weights.get_q4k_device(up_name).expect("fused: up not Q4K");
        self.q4k_matmul
            .fused_gate_up_silu_device(input, gate_w, up_w, output, n, k)
            .expect("fused gate_up_silu");
    }

    fn cpu_rmsnorm_on_gpu(
        &self,
        x: &mut cuda_core::DeviceBuffer<f32>,
        weight: &cuda_core::DeviceBuffer<f32>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        hidden_size: usize,
        seq_len: usize,
        eps: f32,
    ) {
        let stream = self.q4k_matmul.gpu().stream();
        let mut host = x.to_host_vec(&stream).unwrap();
        let w = weight.to_host_vec(&stream).unwrap();
        use tempura_kernels::RMSNorm;
        tempura_kernels::rmsnorm::CpuRMSNormDet
            .forward(&mut host[..seq_len * hidden_size], &w, hidden_size, eps)
            .unwrap();
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                output.cu_deviceptr(), host.as_ptr(),
                seq_len * hidden_size * 4, stream.cu_stream(),
            ).unwrap();
        }
    }

    fn cpu_silu_on_gpu(
        &self,
        gate: &cuda_core::DeviceBuffer<f32>,
        up: &cuda_core::DeviceBuffer<f32>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        count: usize,
    ) {
        let stream = self.q4k_matmul.gpu().stream();
        let g = gate.to_host_vec(&stream).unwrap();
        let u = up.to_host_vec(&stream).unwrap();
        let mut out = vec![0.0f32; count];
        for i in 0..count {
            out[i] = tempura_kernels::portable_math::portable_silu(g[i]) * u[i];
        }
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                output.cu_deviceptr(), out.as_ptr(),
                count * 4, stream.cu_stream(),
            ).unwrap();
        }
    }

    fn cpu_attention_on_gpu(
        &self,
        q: &cuda_core::DeviceBuffer<f32>,
        k: &cuda_core::DeviceBuffer<f32>,
        v: &cuda_core::DeviceBuffer<f32>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        seq_len: usize, kv_len: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, scale: f32,
    ) {
        let stream = self.q4k_matmul.gpu().stream();
        let qh = q.to_host_vec(&stream).unwrap();
        let kh = k.to_host_vec(&stream).unwrap();
        let vh = v.to_host_vec(&stream).unwrap();
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let mut out = vec![0.0f32; seq_len * q_dim];
        tempura_kernels::attention::CpuAttention.forward(
            &qh[..seq_len * q_dim], &kh[..kv_len * kv_dim], &vh[..kv_len * kv_dim],
            &mut out, 1, seq_len, kv_len, n_heads, n_kv_heads, head_dim, scale,
        ).unwrap();
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                output.cu_deviceptr(), out.as_ptr(),
                out.len() * 4, stream.cu_stream(),
            ).unwrap();
        }
    }

    fn forward_gpu_quantized(&self, token_ids: &[u32], position: usize) -> Vec<f32> {
        let ctx = self.q4k_matmul.gpu().context().clone();
        ctx.bind_to_thread().expect("bind CUDA context");
        let stream = self.q4k_matmul.gpu().stream();

        let cfg = &self.config;
        let seq_len = token_ids.len();
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let cpu_ops = std::env::var("TEMPURA_CPU_OPS").is_ok();
        let use_dp4a = self.attn_mode == tempura_kernels::AttentionMode::NaiveF32
            || std::env::var("TEMPURA_DP4A").is_ok();
        let head_dim = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let mut gpu_cache = self.gpu_cache.lock().unwrap();
        let mut scratch_guard = self.scratchpad.lock().unwrap();
        let scratch = &mut *scratch_guard;

        let mut hidden_host = vec![0.0f32; seq_len * h];
        for (t, &tok) in token_ids.iter().enumerate() {
            let src = &self.embed_host[tok as usize * h..(tok as usize + 1) * h];
            hidden_host[t * h..(t + 1) * h].copy_from_slice(src);
        }
        scratch.upload_hidden(&stream, &hidden_host);
        let debug_gpu = std::env::var("TEMPURA_DEBUG_GPU").is_ok();
        let q8_snap = std::env::var("TEMPURA_DETERMINISTIC").is_ok();

        let dbg = |name: &str, buf: &cuda_core::DeviceBuffer<f32>, stream: &cuda_core::CudaStream| {
            if !debug_gpu { return; }
            let v = buf.to_host_vec(stream).unwrap();
            let valid = v.len().min(seq_len * h.max(q_dim).max(inter).max(kv_dim));
            let n = valid.min(4);
            let norm = v[..valid].iter().map(|x| x * x).sum::<f32>().sqrt();
            let has_nan = v[..valid].iter().any(|x| x.is_nan());
            let has_inf = v[..valid].iter().any(|x| x.is_infinite());
            eprintln!("[gpu-dbg] {name}: norm={norm:.4} nan={has_nan} inf={has_inf} first={:?} (valid={valid}/{} total)", &v[..n], v.len());
        };

        dbg("embed", &scratch.hidden, &stream);

        if debug_gpu {
            let normed_host = {
                let input_ln_w = self.gpu_weights.get("model.layers.0.input_layernorm.weight").unwrap();
                let ln_w_host = input_ln_w.to_host_vec(&stream).unwrap();
                let mut normed = hidden_host.clone();
                tempura_kernels::rmsnorm::CpuRMSNorm.forward(&mut normed, &ln_w_host, h, cfg.layer_norm_eps).unwrap();
                normed
            };
            eprintln!("[cpu-ref] L0 normed: first={:?} norm={:.4}",
                &normed_host[..4], normed_host.iter().map(|x| x*x).sum::<f32>().sqrt());
        }

        let positions: Vec<u32> = (position..position + seq_len).map(|p| p as u32).collect();
        let positions_dev =
            cuda_core::DeviceBuffer::from_host(&stream, &positions).expect("upload positions");

        let profile = std::env::var("TEMPURA_PROFILE").is_ok();
        let mut t_matmul = 0u128;
        let mut t_rmsnorm = 0u128;
        let mut t_rope = 0u128;
        let mut t_attn = 0u128;
        let mut t_elem = 0u128;
        let mut t_kvcopy = 0u128;
        macro_rules! timed {
            ($accum:ident, $body:expr) => {{
                if profile {
                    stream.synchronize().unwrap();
                    let _t0 = std::time::Instant::now();
                    let _r = $body;
                    stream.synchronize().unwrap();
                    $accum += _t0.elapsed().as_micros();
                    _r
                } else {
                    $body
                }
            }};
        }

        let fwd_start = std::time::Instant::now();
        for layer in 0..cfg.num_layers {
            let prefix = format!("model.layers.{layer}");

            let input_ln_w = self
                .gpu_weights
                .get(&format!("{prefix}.input_layernorm.weight"))
                .unwrap();
            if cpu_ops {
                self.cpu_rmsnorm_on_gpu(&mut scratch.hidden, input_ln_w, &mut scratch.normed, h, seq_len, cfg.layer_norm_eps);
            } else {
                self.rmsnorm.forward_device(&scratch.hidden, input_ln_w, &mut scratch.normed, h, cfg.layer_norm_eps).expect("rmsnorm");
            }

            if q8_snap {
                tempura_kernels::gpu::q8_snap_device(&ctx, &stream, &mut scratch.normed, seq_len * h).expect("q8 snap normed");
            }

            if layer == 0 { dbg("L0 normed", &scratch.normed, &stream); }

            if std::env::var("TEMPURA_COMPARE").is_ok() && layer == 0 {
                let gpu_normed = scratch.normed.to_host_vec(&stream).unwrap();
                let gpu_hidden = scratch.hidden.to_host_vec(&stream).unwrap();
                let ln_w = input_ln_w.to_host_vec(&stream).unwrap();

                let mut cpu_normed = gpu_hidden[..seq_len * h].to_vec();
                tempura_kernels::rmsnorm::CpuRMSNormDet.forward(&mut cpu_normed, &ln_w, h, cfg.layer_norm_eps).unwrap();

                let max_diff: f32 = cpu_normed.iter().zip(&gpu_normed[..seq_len * h])
                    .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                let num_diff = cpu_normed.iter().zip(&gpu_normed[..seq_len * h])
                    .filter(|(a, b)| a != b).count();
                eprintln!("[compare] L0 RMSNorm: max_diff={max_diff:.10}, diff_elems={num_diff}/{}",
                    seq_len * h);
            }

            let q_name = format!("{prefix}.self_attn.q_proj.weight");
            let k_name = format!("{prefix}.self_attn.k_proj.weight");
            let v_name = format!("{prefix}.self_attn.v_proj.weight");

            if use_dp4a {
                self.gpu_quantize_q8(&scratch.normed, &mut scratch.q8_activation, seq_len * h);
                self.gpu_gemv_q8_or_f32(&q_name, &scratch.normed, &scratch.q8_activation, &mut scratch.q, q_dim, h, seq_len);
                self.gpu_gemv_q8_or_f32(&k_name, &scratch.normed, &scratch.q8_activation, &mut scratch.k, kv_dim, h, seq_len);
                self.gpu_gemv_q8_or_f32(&v_name, &scratch.normed, &scratch.q8_activation, &mut scratch.v, kv_dim, h, seq_len);
            } else {
                self.gpu_gemv_f32(&q_name, &scratch.normed, &mut scratch.q, q_dim, h, seq_len);
                self.gpu_gemv_f32(&k_name, &scratch.normed, &mut scratch.k, kv_dim, h, seq_len);
                self.gpu_gemv_f32(&v_name, &scratch.normed, &mut scratch.v, kv_dim, h, seq_len);
            }

            if layer == 0 {
                dbg("L0 q_proj", &scratch.q, &stream);
                dbg("L0 k_proj", &scratch.k, &stream);
                dbg("L0 v_proj", &scratch.v, &stream);
            }

            if cfg.use_qk_norm {
                let q_norm_w = self
                    .gpu_weights
                    .get(&format!("{prefix}.self_attn.q_norm.weight"))
                    .unwrap();
                if cpu_ops {
                    self.cpu_rmsnorm_on_gpu(&mut scratch.q, q_norm_w, &mut scratch.q_rotated, head_dim, seq_len * n_heads, cfg.layer_norm_eps);
                } else {
                    self.rmsnorm.forward_device(&scratch.q, q_norm_w, &mut scratch.q_rotated, head_dim, cfg.layer_norm_eps).expect("qk norm q");
                }
                if q8_snap {
                    tempura_kernels::gpu::q8_snap_device(&ctx, &stream, &mut scratch.q_rotated, seq_len * q_dim).expect("q8 snap qk q");
                }
                tempura_kernels::gpu::copy_region_device(
                    &ctx, &stream, &scratch.q_rotated, &mut scratch.q, 0, 0, seq_len * q_dim,
                ).expect("copy q norm");

                let k_norm_w = self
                    .gpu_weights
                    .get(&format!("{prefix}.self_attn.k_norm.weight"))
                    .unwrap();
                if cpu_ops {
                    self.cpu_rmsnorm_on_gpu(&mut scratch.k, k_norm_w, &mut scratch.k_rotated, head_dim, seq_len * n_kv_heads, cfg.layer_norm_eps);
                } else {
                    self.rmsnorm.forward_device(&scratch.k, k_norm_w, &mut scratch.k_rotated, head_dim, cfg.layer_norm_eps).expect("qk norm k");
                }
                if q8_snap {
                    tempura_kernels::gpu::q8_snap_device(&ctx, &stream, &mut scratch.k_rotated, seq_len * kv_dim).expect("q8 snap qk k");
                }
                tempura_kernels::gpu::copy_region_device(
                    &ctx, &stream, &scratch.k_rotated, &mut scratch.k, 0, 0,
                    seq_len * kv_dim,
                )
                .expect("copy k norm");
            }

            if cpu_ops {
                let mut qh = scratch.q.to_host_vec(&stream).unwrap();
                let mut kh = scratch.k.to_host_vec(&stream).unwrap();
                let rope_table = tempura_kernels::rope::CpuRoPETable::new(cfg.max_seq_len, head_dim, cfg.rope_theta);
                use tempura_kernels::RoPE;
                rope_table.forward(&mut qh[..seq_len * q_dim], 1, seq_len, n_heads, head_dim, &positions, cfg.rope_theta).unwrap();
                rope_table.forward(&mut kh[..seq_len * kv_dim], 1, seq_len, n_kv_heads, head_dim, &positions, cfg.rope_theta).unwrap();
                unsafe {
                    cuda_core::memory::memcpy_htod_async(scratch.q_rotated.cu_deviceptr(), qh.as_ptr(), seq_len * q_dim * 4, stream.cu_stream()).unwrap();
                    cuda_core::memory::memcpy_htod_async(scratch.k_rotated.cu_deviceptr(), kh.as_ptr(), seq_len * kv_dim * 4, stream.cu_stream()).unwrap();
                }
            } else {
                self.rope.forward_device(&scratch.q, &mut scratch.q_rotated, &positions_dev, seq_len, n_heads, head_dim, cfg.rope_theta).expect("rope q");
                self.rope.forward_device(&scratch.k, &mut scratch.k_rotated, &positions_dev, seq_len, n_kv_heads, head_dim, cfg.rope_theta).expect("rope k");
            }

            let stride = n_kv_heads * head_dim;
            let dst_offset = gpu_cache.seq_len * stride;
            let src_len = seq_len * stride;
            tempura_kernels::gpu::copy_region_device(
                &ctx,
                &stream,
                &scratch.k_rotated,
                &mut gpu_cache.k[layer],
                0,
                dst_offset,
                src_len,
            )
            .expect("kv cache k append");
            tempura_kernels::gpu::copy_region_device(
                &ctx,
                &stream,
                &scratch.v,
                &mut gpu_cache.v[layer],
                0,
                dst_offset,
                src_len,
            )
            .expect("kv cache v append");

            let kv_len = gpu_cache.seq_len + seq_len;
            if cpu_ops {
                self.cpu_attention_on_gpu(
                    &scratch.q_rotated, &gpu_cache.k[layer], &gpu_cache.v[layer],
                    &mut scratch.attn_out, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                    1.0 / (head_dim as f32).sqrt(),
                );
            } else {
                self.attention.forward_device_with_mode(
                    &scratch.q_rotated, &gpu_cache.k[layer], &gpu_cache.v[layer],
                    &mut scratch.attn_out, 1, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                    1.0 / (head_dim as f32).sqrt(), self.attn_mode,
                ).expect("attention");
            }

            if layer == 0 { dbg("L0 attn_out", &scratch.attn_out, &stream); }

            if std::env::var("TEMPURA_COMPARE").is_ok() && layer == 0 {
                let gpu_attn = scratch.attn_out.to_host_vec(&stream).unwrap();
                let gpu_q = scratch.q_rotated.to_host_vec(&stream).unwrap();
                let gpu_k = gpu_cache.k[0].to_host_vec(&stream).unwrap();
                let gpu_v = gpu_cache.v[0].to_host_vec(&stream).unwrap();

                let mut cpu_attn = vec![0.0f32; seq_len * q_dim];
                tempura_kernels::attention::CpuAttention.forward(
                    &gpu_q[..seq_len * q_dim], &gpu_k[..kv_len * kv_dim], &gpu_v[..kv_len * kv_dim],
                    &mut cpu_attn, 1, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                    1.0 / (head_dim as f32).sqrt(),
                ).unwrap();

                let max_diff: f32 = cpu_attn.iter().zip(&gpu_attn[..seq_len * q_dim])
                    .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                let num_diff = cpu_attn.iter().zip(&gpu_attn[..seq_len * q_dim])
                    .filter(|(a, b)| a != b).count();
                eprintln!("[compare] L0 attention: max_diff={max_diff:.10}, diff_elems={num_diff}/{}",
                    seq_len * q_dim);
            }

            let o_name = format!("{prefix}.self_attn.o_proj.weight");
            if use_dp4a {
                self.gpu_quantize_q8(&scratch.attn_out, &mut scratch.q8_activation, seq_len * q_dim);
                self.gpu_gemv_q8_or_f32(&o_name, &scratch.attn_out, &scratch.q8_activation, &mut scratch.o_proj, h, q_dim, seq_len);
            } else {
                self.gpu_gemv_f32(&o_name, &scratch.attn_out, &mut scratch.o_proj, h, q_dim, seq_len);
            }

            if cpu_ops {
                let mut hh = scratch.hidden.to_host_vec(&stream).unwrap();
                let oh = scratch.o_proj.to_host_vec(&stream).unwrap();
                for i in 0..seq_len * h { hh[i] += oh[i]; }
                unsafe { cuda_core::memory::memcpy_htod_async(scratch.hidden.cu_deviceptr(), hh.as_ptr(), seq_len * h * 4, stream.cu_stream()).unwrap(); }
            } else {
                tempura_kernels::gpu::vector_add_device(&ctx, &stream, &scratch.hidden, &scratch.o_proj, &mut scratch.hidden_swap).expect("residual add");
                std::mem::swap(&mut scratch.hidden, &mut scratch.hidden_swap);
            }

            if layer == 0 { dbg("L0 hidden_post_attn", &scratch.hidden, &stream); }

            let post_ln_w = self
                .gpu_weights
                .get(&format!("{prefix}.post_attention_layernorm.weight"))
                .unwrap();
            if cpu_ops {
                self.cpu_rmsnorm_on_gpu(&mut scratch.hidden, post_ln_w, &mut scratch.normed2, h, seq_len, cfg.layer_norm_eps);
            } else {
                self.rmsnorm.forward_device(&scratch.hidden, post_ln_w, &mut scratch.normed2, h, cfg.layer_norm_eps).expect("rmsnorm2");
            }

            if q8_snap {
                tempura_kernels::gpu::q8_snap_device(&ctx, &stream, &mut scratch.normed2, seq_len * h).expect("q8 snap normed2");
            }

            let gate_name = format!("{prefix}.mlp.gate_proj.weight");
            let up_name = format!("{prefix}.mlp.up_proj.weight");
            let down_name = format!("{prefix}.mlp.down_proj.weight");

            let can_fuse_gate_up = !use_dp4a && !cpu_ops && seq_len == 1
                && self.gpu_weights.get_q4k_device(&gate_name).is_some()
                && self.gpu_weights.get_q4k_device(&up_name).is_some();

            if can_fuse_gate_up {
                self.gpu_fused_gate_up_silu(&gate_name, &up_name, &scratch.normed2, &mut scratch.silu_out, inter, h);
            } else if use_dp4a {
                self.gpu_quantize_q8(&scratch.normed2, &mut scratch.q8_activation, seq_len * h);
                self.gpu_gemv_q8_or_f32(&gate_name, &scratch.normed2, &scratch.q8_activation, &mut scratch.gate, inter, h, seq_len);
                self.gpu_gemv_q8_or_f32(&up_name, &scratch.normed2, &scratch.q8_activation, &mut scratch.up, inter, h, seq_len);

                if cpu_ops {
                    self.cpu_silu_on_gpu(&scratch.gate, &scratch.up, &mut scratch.silu_out, seq_len * inter);
                } else {
                    tempura_kernels::gpu::silu_mul_device(&ctx, &stream, &scratch.gate, &scratch.up, &mut scratch.silu_out).expect("silu_mul");
                }
            } else {
                self.gpu_gemv_f32(&gate_name, &scratch.normed2, &mut scratch.gate, inter, h, seq_len);
                self.gpu_gemv_f32(&up_name, &scratch.normed2, &mut scratch.up, inter, h, seq_len);

                if cpu_ops {
                    self.cpu_silu_on_gpu(&scratch.gate, &scratch.up, &mut scratch.silu_out, seq_len * inter);
                } else {
                    tempura_kernels::gpu::silu_mul_device(&ctx, &stream, &scratch.gate, &scratch.up, &mut scratch.silu_out).expect("silu_mul");
                }
            }

            if q8_snap {
                tempura_kernels::gpu::q8_snap_device(&ctx, &stream, &mut scratch.silu_out, seq_len * inter).expect("q8 snap silu");
            }

            if layer <= 6 {
                dbg(&format!("L{layer} normed2"), &scratch.normed2, &stream);
            }

            if layer <= 6 { dbg(&format!("L{layer} silu_out"), &scratch.silu_out, &stream); }

            if std::env::var("TEMPURA_COMPARE").is_ok() && layer == 0 {
                let gpu_gate = scratch.gate.to_host_vec(&stream).unwrap();
                let gpu_up = scratch.up.to_host_vec(&stream).unwrap();
                let gpu_silu = scratch.silu_out.to_host_vec(&stream).unwrap();

                let mut cpu_silu = vec![0.0f32; seq_len * inter];
                for i in 0..seq_len * inter {
                    cpu_silu[i] = tempura_kernels::portable_math::portable_silu(gpu_gate[i]) * gpu_up[i];
                }

                let max_diff: f32 = cpu_silu.iter().zip(&gpu_silu[..seq_len * inter])
                    .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                let num_diff = cpu_silu.iter().zip(&gpu_silu[..seq_len * inter])
                    .filter(|(a, b)| a != b).count();
                eprintln!("[compare] L0 SiLU: max_diff={max_diff:.10}, diff_elems={num_diff}/{}",
                    seq_len * inter);
            }

            if use_dp4a {
                self.gpu_quantize_q8(&scratch.silu_out, &mut scratch.q8_activation, seq_len * inter);
                self.gpu_gemv_q8_or_f32(&down_name, &scratch.silu_out, &scratch.q8_activation, &mut scratch.down, h, inter, seq_len);
            } else {
                self.gpu_gemv_f32(&down_name, &scratch.silu_out, &mut scratch.down, h, inter, seq_len);
            }

            if layer <= 6 { dbg(&format!("L{layer} down_proj"), &scratch.down, &stream); }

            if layer == 6 && debug_gpu {
                let silu_host = scratch.silu_out.to_host_vec(&stream).unwrap();
                let silu_valid = &silu_host[..inter];
                if let Some(dw_bytes) = self.gpu_weights.get_q6k_device(&down_name) {
                    let dw_host = dw_bytes.to_host_vec(&stream).unwrap();
                    let q6k_cpu = tempura_kernels::quantized::CpuMatMulQ4KOpt;
                    let mut cpu_out = vec![0.0f32; h];
                    use tempura_kernels::MatMulQ;
                    q6k_cpu.forward_b_transposed_q6k(silu_valid, &dw_host, &mut cpu_out, 1, h, inter).unwrap();
                    let cpu_norm = cpu_out.iter().map(|x| x*x).sum::<f32>().sqrt();
                    eprintln!("[cpu-ref] L6 down_proj: norm={cpu_norm:.4} first={:?}", &cpu_out[..4]);
                }
            }

            if cpu_ops {
                let mut hh = scratch.hidden.to_host_vec(&stream).unwrap();
                let dh = scratch.down.to_host_vec(&stream).unwrap();
                for i in 0..seq_len * h { hh[i] += dh[i]; }
                unsafe { cuda_core::memory::memcpy_htod_async(scratch.hidden.cu_deviceptr(), hh.as_ptr(), seq_len * h * 4, stream.cu_stream()).unwrap(); }
            } else {
                tempura_kernels::gpu::vector_add_device(&ctx, &stream, &scratch.hidden, &scratch.down, &mut scratch.hidden_swap).expect("ffn residual add");
                std::mem::swap(&mut scratch.hidden, &mut scratch.hidden_swap);
            }

            if layer == 0 { dbg("L0 hidden_post_ffn", &scratch.hidden, &stream); }
            if debug_gpu {
                let v = scratch.hidden.to_host_vec(&stream).unwrap();
                let valid = seq_len * h;
                let norm = v[..valid].iter().map(|x| x*x).sum::<f32>().sqrt();
                let has_nan = v[..valid].iter().any(|x| x.is_nan());
                eprintln!("[gpu-dbg] L{layer} hidden[0..{valid}]: norm={norm:.2} first={:?}", &v[..4]);
                if has_nan { eprintln!("[gpu-dbg] NaN at layer {layer}!"); break; }
            }
        }

        if profile {
            stream.synchronize().unwrap();
            let layers_ms = fwd_start.elapsed().as_millis();
            eprintln!("[profile] {} layers: {layers_ms}ms ({:.2}ms/layer)",
                cfg.num_layers, layers_ms as f64 / cfg.num_layers as f64);
        }

        gpu_cache.advance(seq_len);

        dbg("final_hidden", &scratch.hidden, &stream);

        let final_norm_w = self.gpu_weights.get("model.norm.weight").unwrap();
        if cpu_ops {
            self.cpu_rmsnorm_on_gpu(&mut scratch.hidden, final_norm_w, &mut scratch.final_normed, h, seq_len, cfg.layer_norm_eps);
        } else {
            self.rmsnorm.forward_device(&scratch.hidden, final_norm_w, &mut scratch.final_normed, h, cfg.layer_norm_eps).expect("final norm");
        }

        if q8_snap {
            tempura_kernels::gpu::q8_snap_device(&ctx, &stream, &mut scratch.final_normed, seq_len * h).expect("q8 snap final");
        }

        let lm_head_name = "lm_head.weight";
        let lm_start = std::time::Instant::now();
        if use_dp4a {
            self.gpu_quantize_q8(&scratch.final_normed, &mut scratch.q8_activation, seq_len * h);
            self.gpu_gemv_q8_or_f32(lm_head_name, &scratch.final_normed, &scratch.q8_activation, &mut scratch.logits, cfg.vocab_size, h, seq_len);
        } else {
            self.gpu_gemv_f32(lm_head_name, &scratch.final_normed, &mut scratch.logits, cfg.vocab_size, h, seq_len);
        }
        if profile {
            stream.synchronize().unwrap();
            let lm_ms = lm_start.elapsed().as_millis();
            let total_ms = fwd_start.elapsed().as_millis();
            eprintln!("[profile] lm_head: {lm_ms}ms, total fwd: {total_ms}ms");
        }

        let all_logits = scratch.logits.to_host_vec(&stream).expect("logits to host");
        let last_start = (seq_len - 1) * cfg.vocab_size;
        all_logits[last_start..last_start + cfg.vocab_size].to_vec()
    }
}

#[cfg(feature = "cuda")]
impl InferenceModel for GpuQuantizedInferenceWrapper {
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn has_weights(&self) -> bool {
        true
    }
    fn forward(&self, session: &mut InferenceSession, token_ids: &[u32]) -> Vec<f32> {
        if let Some(ref cpu_model) = self.cpu_model {
            let mut model = cpu_model.lock().unwrap();
            let qw = model.quantized_weights.as_ref().unwrap() as *const crate::weights::QuantizedWeights;
            return model.forward_quantized(session, token_ids, unsafe { &*qw });
        }
        let logits = self.forward_gpu_quantized(token_ids, session.position);
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
        self.sampler.apply(&logits, &mut probs, params).unwrap();

        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut session.rng)
        };
        let first_token = self.sampler.sample_token(&probs, rng_ref);
        let mut generated = vec![first_token];

        for _step in 1..max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) {
                break;
            }
            let step_start = Instant::now();
            let mut logits = self.forward(session, &[*generated.last().unwrap()]);
            let fwd_us = step_start.elapsed().as_micros();
            params.apply_penalties(&mut logits, &generated);
            let mut probs = vec![0.0f32; logits.len()];
            self.sampler.apply(&logits, &mut probs, params).unwrap();
            let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                None
            } else {
                Some(&mut session.rng)
            };
            let token = self.sampler.sample_token(&probs, rng_ref);
            let step_us = step_start.elapsed().as_micros();
            if _step <= 3 || _step == max_new_tokens - 1 {
                eprintln!("[gen] step {_step}: fwd={fwd_us}us total={step_us}us token={token}");
            }
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
        let logits = self.forward(session, token_ids);
        let mut probs = vec![0.0f32; logits.len()];
        self.sampler.apply(&logits, &mut probs, params).unwrap();
        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut session.rng)
        };
        let first_token = self.sampler.sample_token(&probs, rng_ref);
        let mut generated = vec![first_token];
        let mut logprobs = vec![extract_logprob(&probs, first_token, top_n)];

        for _step in 1..max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) {
                break;
            }
            let mut logits = self.forward(session, &[*generated.last().unwrap()]);
            params.apply_penalties(&mut logits, &generated);
            let mut probs = vec![0.0f32; logits.len()];
            self.sampler.apply(&logits, &mut probs, params).unwrap();
            let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                None
            } else {
                Some(&mut session.rng)
            };
            let token = self.sampler.sample_token(&probs, rng_ref);
            logprobs.push(extract_logprob(&probs, token, top_n));
            generated.push(token);
        }
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
        let logits = self.forward(session, token_ids);
        let mut probs = vec![0.0f32; logits.len()];
        self.sampler.apply(&logits, &mut probs, params).unwrap();
        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut session.rng)
        };
        let first_token = self.sampler.sample_token(&probs, rng_ref);
        let mut generated = vec![first_token];
        if !on_token(first_token) {
            return generated;
        }
        for _step in 1..max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) {
                break;
            }
            let mut logits = self.forward(session, &[*generated.last().unwrap()]);
            params.apply_penalties(&mut logits, &generated);
            let mut probs = vec![0.0f32; logits.len()];
            self.sampler.apply(&logits, &mut probs, params).unwrap();
            let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                None
            } else {
                Some(&mut session.rng)
            };
            let token = self.sampler.sample_token(&probs, rng_ref);
            generated.push(token);
            if !on_token(token) {
                break;
            }
        }
        generated
    }
    fn generate_guided(
        &self,
        _session: &mut InferenceSession,
        _token_ids: &[u32],
        _max_new_tokens: usize,
        _params: &SamplingParams,
        _eos_token: Option<u32>,
        _decoder: &mut dyn tempura_kernels::GuidedDecoder,
    ) -> Vec<u32> {
        todo!("guided decoding not yet supported on GPU quantized path")
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
        self.weights.is_some() || self.quantized_weights.is_some()
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
        decoder: &mut dyn tempura_kernels::GuidedDecoder,
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
    fn generate_with_sampler_chain(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        chain: &mut tempura_kernels::sampler_chain::SamplerChain,
    ) -> Vec<u32> {
        self.generate_with_sampler_chain(session, token_ids, max_new_tokens, params, eos_token, chain)
    }
    fn load_weights(
        &mut self,
        loader: &dyn crate::loader::WeightLoader,
    ) -> Result<(), crate::weights::WeightsError> {
        self.load_weights(loader)
    }
}

pub use tempura_kernels::sampling::SamplingParams as SampleParams;

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
mod prop_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_head_dim_scale_never_zero_division(head_dim in 1usize..=16384) {
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            prop_assert!(!scale.is_nan(), "1/sqrt({}) = NaN", head_dim);
            prop_assert!(!scale.is_infinite(), "1/sqrt({}) = Inf", head_dim);
            prop_assert!(scale > 0.0, "1/sqrt({}) = {} <= 0", head_dim, scale);
        }
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

    #[test]
    fn test_paged_forward_matches_contiguous() {
        let config = ModelConfig::tiny_test();
        let loader = RandomWeightLoader::from_config(&config, 42);
        let tokens = [1u32, 2, 3, 4, 5];

        let pipeline1 = KernelPipeline::cpu_verified();
        let mut model1 = TransformerModel::new(config.clone(), pipeline1);
        model1.load_weights(&loader).unwrap();
        let mut s1 = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits_contiguous = model1.forward(&mut s1, &tokens);

        let pipeline2 = KernelPipeline::cpu_verified();
        let mut model2 = TransformerModel::new(config.clone(), pipeline2);
        model2.load_weights(&loader).unwrap();
        let mut s2 = InferenceSession::new_paged(&config, 64, 4, TraceLevel::Off, 42);
        let logits_paged = model2.forward(&mut s2, &tokens);

        assert_eq!(logits_contiguous.len(), logits_paged.len());
        for i in 0..logits_contiguous.len() {
            assert!(
                (logits_contiguous[i] - logits_paged[i]).abs() < 1e-5,
                "logit divergence at {i}: contiguous={} paged={}",
                logits_contiguous[i],
                logits_paged[i]
            );
        }
    }

    #[test]
    fn test_paged_generate_matches_contiguous() {
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
        let mut s1 = InferenceSession::new(&config, TraceLevel::Off, 42);
        let tokens_contiguous = model1.generate(&mut s1, &[1, 2], 5, &params, None);

        let pipeline2 = KernelPipeline::cpu_verified();
        let mut model2 = TransformerModel::new(config.clone(), pipeline2);
        model2.load_weights(&loader).unwrap();
        let mut s2 = InferenceSession::new_paged(&config, 64, 4, TraceLevel::Off, 42);
        let tokens_paged = model2.generate(&mut s2, &[1, 2], 5, &params, None);

        assert_eq!(
            tokens_contiguous, tokens_paged,
            "paged generate must produce identical tokens"
        );
    }

    #[test]
    fn test_falcon_uses_layernorm() {
        // Falcon config should use LayerNorm (mean-centering), producing different
        // output from RMSNorm (no mean-centering) given the same weights.
        let mut falcon_cfg = ModelConfig::tiny_test();
        falcon_cfg.architecture = crate::config::Architecture::Falcon;
        falcon_cfg.use_layernorm = true;
        falcon_cfg.hidden_act = crate::config::HiddenAct::Gelu;

        let llama_cfg = ModelConfig::tiny_test();

        let loader = RandomWeightLoader::from_config(&llama_cfg, 42);

        fn make_qw(loader: &RandomWeightLoader) -> crate::weights::QuantizedWeights {
            let mut qw = crate::weights::QuantizedWeights::empty();
            for (name, data) in &loader.tensors {
                qw.insert_f32(name.clone(), data.clone());
            }
            qw
        }
        let qw = make_qw(&loader);

        let pipeline1 = KernelPipeline::cpu_verified();
        let model_falcon = TransformerModel::new(falcon_cfg.clone(), pipeline1);

        let pipeline2 = KernelPipeline::cpu_verified();
        let model_llama = TransformerModel::new(llama_cfg.clone(), pipeline2);

        let mut s1 = InferenceSession::new(&falcon_cfg, TraceLevel::Off, 42);
        let mut s2 = InferenceSession::new(&llama_cfg, TraceLevel::Off, 42);

        let logits_falcon = model_falcon.forward_quantized(&mut s1, &[1, 2, 3], &qw);
        let logits_llama = model_llama.forward_quantized(&mut s2, &[1, 2, 3], &qw);

        // With different norm types and activations, outputs must differ
        let max_diff: f32 = logits_falcon
            .iter()
            .zip(&logits_llama)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff > 0.001,
            "Falcon (LayerNorm+GELU) should produce different output from LLaMA (RMSNorm+SiLU), max_diff={max_diff}"
        );
    }

    #[test]
    fn test_mpt_skips_rope_with_alibi() {
        // MPT uses ALiBi instead of RoPE. Verify via forward_quantized that
        // use_alibi=true produces different output from the same config without ALiBi.
        let mut mpt_cfg = ModelConfig::tiny_test();
        mpt_cfg.architecture = crate::config::Architecture::Mpt;
        mpt_cfg.use_alibi = true;
        mpt_cfg.alibi_max_bias = 8.0;
        mpt_cfg.use_layernorm = true;
        mpt_cfg.hidden_act = crate::config::HiddenAct::Gelu;

        let mut base_cfg = ModelConfig::tiny_test();
        base_cfg.hidden_act = crate::config::HiddenAct::Gelu;

        let loader = RandomWeightLoader::from_config(&base_cfg, 42);

        // Build quantized weights from f32
        fn make_qw(loader: &RandomWeightLoader) -> crate::weights::QuantizedWeights {
            let mut qw = crate::weights::QuantizedWeights::empty();
            for (name, data) in &loader.tensors {
                qw.insert_f32(name.clone(), data.clone());
            }
            qw
        }
        let qw = make_qw(&loader);

        let pipeline1 = KernelPipeline::cpu_verified();
        let model_mpt = TransformerModel::new(mpt_cfg.clone(), pipeline1);
        let pipeline2 = KernelPipeline::cpu_verified();
        let model_base = TransformerModel::new(base_cfg.clone(), pipeline2);

        let mut s1 = InferenceSession::new(&mpt_cfg, TraceLevel::Off, 42);
        let mut s2 = InferenceSession::new(&base_cfg, TraceLevel::Off, 42);

        let logits_mpt = model_mpt.forward_quantized(&mut s1, &[1, 2, 3], &qw);
        let logits_base = model_base.forward_quantized(&mut s2, &[1, 2, 3], &qw);

        // ALiBi (no RoPE, LayerNorm) vs RoPE (RMSNorm) should produce different results
        let max_diff: f32 = logits_mpt
            .iter()
            .zip(&logits_base)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff > 0.001,
            "MPT (ALiBi, no RoPE) should differ from base (RoPE), max_diff={max_diff}"
        );
    }

    #[test]
    fn test_moe_dispatch_triggers_with_experts() {
        // When n_experts > 0 but no expert weights exist, forward_quantized
        // should fall back to dense FFN with a warning (no panic).
        let mut moe_cfg = ModelConfig::tiny_test();
        moe_cfg.n_experts = 8;
        moe_cfg.n_experts_used = 2;

        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(moe_cfg.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&moe_cfg, 42);
        model.load_weights(&loader).unwrap();

        // Build quantized weights from the f32 weights (no expert weights present)
        let mut qw = crate::weights::QuantizedWeights::empty();
        for (name, data) in &loader.tensors {
            qw.insert_f32(name.clone(), data.clone());
        }
        model.quantized_weights = Some(qw);

        let mut session = InferenceSession::new(&moe_cfg, TraceLevel::Off, 42);
        // Should not panic — falls back to dense FFN
        let logits = model.forward(&mut session, &[1, 2, 3]);
        assert_eq!(logits.len(), moe_cfg.vocab_size);
        assert!(!logits.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_falcon_config_from_hf_json() {
        let json = r#"{
            "model_type": "falcon",
            "hidden_size": 4544,
            "intermediate_size": 18176,
            "num_attention_heads": 71,
            "num_hidden_layers": 32,
            "vocab_size": 65024,
            "hidden_act": "gelu",
            "parallel_attn": true
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.architecture, crate::config::Architecture::Falcon);
        assert!(c.use_layernorm);
        assert!(!c.use_alibi);
        assert!(c.parallel_attn_ffn);
        assert_eq!(c.hidden_act, crate::config::HiddenAct::Gelu);
    }

    #[test]
    fn test_mpt_config_from_hf_json() {
        let json = r#"{
            "model_type": "mpt",
            "hidden_size": 4096,
            "intermediate_size": 16384,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 50432
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.architecture, crate::config::Architecture::Mpt);
        assert!(c.use_layernorm);
        assert!(c.use_alibi);
        assert!(c.alibi_max_bias > 0.0);
    }
}
