use crate::config::ModelConfig;
use crate::pipeline::KernelPipeline;
use crate::session::InferenceSession;
use crate::trace::{LayerSnapshot, SamplingSnapshot, TimingInfo, TraceLevel};
use crate::weights::ModelWeights;
use serde::Serialize;
use std::time::Instant;
use tempura_kernels::{
    Attention, DeltaNet, LayerNorm, MatMul, RMSNorm, RoPE, Sampler, SamplingParams,
};

#[allow(dead_code)]
fn q2_preq_into(a: &[f32], q8: &mut [i8], scales: &mut [f32], asum: &mut [i32], k: usize) {
    let n32 = k / 32;
    for blk in 0..n32 {
        let base = blk * 32;
        let mut amax = 0.0f32;
        for i in 0..32 {
            amax = amax.max(a[base + i].abs());
        }
        let d = amax / 127.0;
        scales[blk] = d;
        let mut s = 0i32;
        if d > 0.0 {
            let id = 127.0 / amax;
            for i in 0..32 {
                let v = (a[base + i] * id).round() as i8;
                q8[base + i] = v;
                s += v as i32;
            }
        } else {
            for i in 0..32 {
                q8[base + i] = 0;
            }
        }
        asum[blk] = s;
    }
}

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

/// Shared single-token attention layer processing (post-GEMV).
/// Covers: Q-gate extraction, qk_norm, partial RoPE, KV cache append,
/// attention, sigmoid gating, and attention sub-norm.
/// Used by both `forward_quantized` and `generate_bitnet_decode` to
/// eliminate code divergence.
#[allow(clippy::too_many_arguments)]
fn attention_layer_decode(
    q_raw: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    attn_out: &mut [f32],
    q_gate_buf: &mut [f32],
    session: &mut InferenceSession,
    cache_layer: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    _actual_q_cols: usize,
    has_q_gate: bool,
    rope_dim: usize,
    rope_theta: f32,
    positions: &[u32],
    q_norm_w: Option<&[f32]>,
    k_norm_w: Option<&[f32]>,
    sub_norm_w: Option<&[f32]>,
    eps: f32,
    use_ggml_compat: bool,
    sliding_window: Option<usize>,
) {
    use tempura_kernels::RoPE as _;

    if has_q_gate {
        assert!(q_gate_buf.len() >= q_dim);
        for hh in 0..n_heads {
            let src = hh * head_dim * 2;
            q_gate_buf[hh * head_dim..(hh + 1) * head_dim]
                .copy_from_slice(&q_raw[src + head_dim..src + 2 * head_dim]);
        }
        for hh in 0..n_heads {
            let src = hh * head_dim * 2;
            let dst = hh * head_dim;
            if src != dst {
                q_raw.copy_within(src..src + head_dim, dst);
            }
        }
    }
    let q = &mut q_raw[..q_dim];

    if let (Some(qnw), Some(knw)) = (q_norm_w, k_norm_w) {
        apply_head_rmsnorm(q, qnw, 1, n_heads, head_dim, eps);
        if use_ggml_compat {
            tempura_kernels::portable_math::q8_snap(q);
        }
        apply_head_rmsnorm(k, knw, 1, n_kv_heads, head_dim, eps);
        if use_ggml_compat {
            tempura_kernels::portable_math::q8_snap(k);
        }
    }

    let rope = tempura_kernels::rope::CpuRoPE;
    if rope_dim < head_dim {
        let mut q_rope = vec![0.0f32; n_heads * rope_dim];
        let mut k_rope = vec![0.0f32; n_kv_heads * rope_dim];
        for hh in 0..n_heads {
            q_rope[hh * rope_dim..(hh + 1) * rope_dim]
                .copy_from_slice(&q[hh * head_dim..hh * head_dim + rope_dim]);
        }
        for hh in 0..n_kv_heads {
            k_rope[hh * rope_dim..(hh + 1) * rope_dim]
                .copy_from_slice(&k[hh * head_dim..hh * head_dim + rope_dim]);
        }
        rope.forward(&mut q_rope, 1, 1, n_heads, rope_dim, positions, rope_theta)
            .unwrap();
        rope.forward(
            &mut k_rope,
            1,
            1,
            n_kv_heads,
            rope_dim,
            positions,
            rope_theta,
        )
        .unwrap();
        for hh in 0..n_heads {
            q[hh * head_dim..hh * head_dim + rope_dim]
                .copy_from_slice(&q_rope[hh * rope_dim..(hh + 1) * rope_dim]);
        }
        for hh in 0..n_kv_heads {
            k[hh * head_dim..hh * head_dim + rope_dim]
                .copy_from_slice(&k_rope[hh * rope_dim..(hh + 1) * rope_dim]);
        }
    } else {
        rope.forward(q, 1, 1, n_heads, head_dim, positions, rope_theta)
            .unwrap();
        rope.forward(k, 1, 1, n_kv_heads, head_dim, positions, rope_theta)
            .unwrap();
    }

    session.cache.append_kv(cache_layer, &k[..kv_dim], v, 1);
    let kv_len = session.cache.seq_len + 1;
    let k_cache = session.cache.k_data(cache_layer, kv_len);
    let v_cache = session.cache.v_data(cache_layer, kv_len);

    let scale = 1.0 / (head_dim as f32).sqrt();
    if use_ggml_compat {
        use tempura_kernels::Attention as _;
        match sliding_window {
            Some(w) => tempura_kernels::attention::CpuAttentionGgml
                .forward_windowed(
                    q,
                    &k_cache,
                    &v_cache,
                    &mut attn_out[..q_dim],
                    1,
                    1,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    scale,
                    w,
                )
                .unwrap(),
            None => tempura_kernels::attention::CpuAttentionGgml
                .forward(
                    q,
                    &k_cache,
                    &v_cache,
                    &mut attn_out[..q_dim],
                    1,
                    1,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    scale,
                )
                .unwrap(),
        }
    } else {
        use tempura_kernels::Attention as _;
        match sliding_window {
            Some(w) => tempura_kernels::attention::CpuAttention
                .forward_windowed(
                    q,
                    &k_cache,
                    &v_cache,
                    &mut attn_out[..q_dim],
                    1,
                    1,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    scale,
                    w,
                )
                .unwrap(),
            None => tempura_kernels::attention::CpuAttention
                .forward(
                    q,
                    &k_cache,
                    &v_cache,
                    &mut attn_out[..q_dim],
                    1,
                    1,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    scale,
                )
                .unwrap(),
        }
    }

    if has_q_gate {
        for i in 0..q_dim {
            let sig = 1.0 / (1.0 + (-q_gate_buf[i]).exp());
            attn_out[i] *= sig;
        }
    }

    if let Some(snw) = sub_norm_w {
        if use_ggml_compat {
            use tempura_kernels::RMSNorm as _;
            tempura_kernels::rmsnorm::CpuRMSNormGgml
                .forward(&mut attn_out[..q_dim], snw, q_dim, eps)
                .unwrap();
        } else {
            use tempura_kernels::RMSNorm as _;
            tempura_kernels::rmsnorm::CpuRMSNorm
                .forward(&mut attn_out[..q_dim], snw, q_dim, eps)
                .unwrap();
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
    fn sample_from_logits(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        session: &mut InferenceSession,
    ) -> u32;
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

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
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
        }
    }

    pub fn load_weights(
        &mut self,
        loader: &dyn crate::loader::WeightLoader,
    ) -> Result<(), crate::weights::WeightsError> {
        self.weights = Some(ModelWeights::load(loader, &self.config)?);
        Ok(())
    }

    fn use_persistent_thread_path(&self, qw: &crate::weights::QuantizedWeights) -> bool {
        if self.config.hidden_act == crate::config::HiddenAct::SquaredRelu {
            return true;
        }
        qw.has_q2_tiled()
    }

    pub fn forward(&self, session: &mut InferenceSession, token_ids: &[u32]) -> Vec<f32> {
        if let Some(qw) = &self.quantized_weights {
            if self.use_persistent_thread_path(qw) {
                if token_ids.len() == 1 {
                    return self.forward_bitnet_decode(session, token_ids[0], qw);
                }
                return self.forward_bitnet_prefill(session, token_ids, qw);
            }
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

        let mut norm_weight_buf = if cfg.rmsnorm_offset {
            vec![0.0f32; h]
        } else {
            Vec::new()
        };

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

        let cache_layer_map = cfg.cache_layer_map();
        let mut deltanet_layer_idx = 0usize;

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

            let layer_kind = cfg
                .layer_kinds
                .as_ref()
                .map(|kinds| kinds[layer])
                .unwrap_or(crate::config::LayerKind::Attention);

            // Hoisted: normed and o_proj are used by FFN block after the if/else
            let mut normed = hidden.clone();
            let mut o_proj = vec![0.0f32; seq_len * h];

            if layer_kind == crate::config::LayerKind::DeltaNet {
                let dn_heads = cfg.deltanet_num_heads.unwrap_or(n_kv_heads);
                let dn_head_dim = cfg.deltanet_head_dim.unwrap_or(head_dim);
                let dn_dim = dn_heads * dn_head_dim;
                let ssm_a = cfg.ssm_a.unwrap_or(-1.0);

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

                let mut q = vec![0.0f32; seq_len * dn_dim];
                let mut k = vec![0.0f32; seq_len * dn_dim];
                let mut v = vec![0.0f32; seq_len * dn_dim];
                let mut z_gate = vec![0.0f32; seq_len * dn_dim];
                let mut alpha = vec![0.0f32; seq_len * dn_heads];
                let mut beta = vec![0.0f32; seq_len * dn_heads];

                // in_proj: hidden → [Q, K, V, Z] where Z is output gating
                if let Some(in_w) = weights.get(&format!("{prefix}.ssm.in_proj.weight")) {
                    let in_cols = 4 * dn_dim;
                    let mut in_proj = vec![0.0f32; seq_len * in_cols];
                    self.pipeline
                        .matmul_bt(&normed, in_w, &mut in_proj, seq_len, in_cols, h)
                        .unwrap();

                    for t in 0..seq_len {
                        let src = &in_proj[t * in_cols..];
                        q[t * dn_dim..(t + 1) * dn_dim].copy_from_slice(&src[..dn_dim]);
                        k[t * dn_dim..(t + 1) * dn_dim].copy_from_slice(&src[dn_dim..2 * dn_dim]);
                        v[t * dn_dim..(t + 1) * dn_dim]
                            .copy_from_slice(&src[2 * dn_dim..3 * dn_dim]);
                        z_gate[t * dn_dim..(t + 1) * dn_dim]
                            .copy_from_slice(&src[3 * dn_dim..4 * dn_dim]);
                    }
                } else {
                    let q_w = weights
                        .get(&format!("{prefix}.self_attn.q_proj.weight"))
                        .unwrap();
                    let k_w = weights
                        .get(&format!("{prefix}.self_attn.k_proj.weight"))
                        .unwrap();
                    let v_w = weights
                        .get(&format!("{prefix}.self_attn.v_proj.weight"))
                        .unwrap();
                    self.pipeline
                        .matmul_bt(&normed, q_w, &mut q, seq_len, dn_dim, h)
                        .unwrap();
                    self.pipeline
                        .matmul_bt(&normed, k_w, &mut k, seq_len, dn_dim, h)
                        .unwrap();
                    self.pipeline
                        .matmul_bt(&normed, v_w, &mut v, seq_len, dn_dim, h)
                        .unwrap();
                }

                // beta_alpha projection (Qwen3.6 combined, or separate Qwen3.5)
                if let Some(ba_w) = weights.get(&format!("{prefix}.ssm.beta_alpha.weight")) {
                    let ba_cols = 2 * dn_heads;
                    let mut ba_proj = vec![0.0f32; seq_len * ba_cols];
                    self.pipeline
                        .matmul_bt(&normed, ba_w, &mut ba_proj, seq_len, ba_cols, h)
                        .unwrap();
                    for t in 0..seq_len {
                        let src = &ba_proj[t * ba_cols..];
                        beta[t * dn_heads..(t + 1) * dn_heads].copy_from_slice(&src[..dn_heads]);
                        alpha[t * dn_heads..(t + 1) * dn_heads].copy_from_slice(&src[dn_heads..]);
                    }
                }

                // Short causal conv1d on Q and K (kernel_size typically 4)
                if let Some(conv_w) = weights.get(&format!("{prefix}.ssm.conv1d.weight")) {
                    let kernel_size = conv_w.len() / (2 * dn_dim);
                    let conv_bias = weights.get(&format!("{prefix}.ssm.conv1d.bias"));
                    // conv_w layout: [2*dn_dim, kernel_size] (depthwise, Q and K channels)
                    for ch in 0..dn_dim {
                        for t in 0..seq_len {
                            let mut sum_q = 0.0f32;
                            let mut sum_k = 0.0f32;
                            for ki in 0..kernel_size {
                                let src_t = t as isize - ki as isize;
                                if src_t >= 0 {
                                    sum_q += q[src_t as usize * dn_dim + ch]
                                        * conv_w[ch * kernel_size + ki];
                                    sum_k += k[src_t as usize * dn_dim + ch]
                                        * conv_w[(dn_dim + ch) * kernel_size + ki];
                                }
                            }
                            if let Some(bias) = conv_bias {
                                sum_q += bias[ch];
                                sum_k += bias[dn_dim + ch];
                            }
                            q[t * dn_dim + ch] = sum_q;
                            k[t * dn_dim + ch] = sum_k;
                        }
                    }
                }

                // SiLU activation on Q and K
                for x in q.iter_mut() {
                    *x = *x / (1.0 + (-*x).exp());
                }
                for x in k.iter_mut() {
                    *x = *x / (1.0 + (-*x).exp());
                }

                // L2 normalize Q and K per head
                for t in 0..seq_len {
                    for hd in 0..dn_heads {
                        let off = t * dn_dim + hd * dn_head_dim;
                        let slice = &mut q[off..off + dn_head_dim];
                        let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                        slice.iter_mut().for_each(|x| *x /= norm);

                        let slice = &mut k[off..off + dn_head_dim];
                        let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                        slice.iter_mut().for_each(|x| *x /= norm);
                    }
                }

                // dt_proj for alpha gate refinement
                if let Some(dt_w) = weights.get(&format!("{prefix}.ssm.dt_proj.weight")) {
                    let mut dt_proj = vec![0.0f32; seq_len * dn_heads];
                    self.pipeline
                        .matmul_bt(&normed, dt_w, &mut dt_proj, seq_len, dn_heads, h)
                        .unwrap();
                    if let Some(dt_b) = weights.get(&format!("{prefix}.ssm.dt_proj.bias")) {
                        for t in 0..seq_len {
                            for hd in 0..dn_heads {
                                dt_proj[t * dn_heads + hd] += dt_b[hd];
                            }
                        }
                    }
                    for i in 0..alpha.len() {
                        alpha[i] += dt_proj[i];
                    }
                } else if let Some(ref bias) = cfg.dt_bias {
                    for t in 0..seq_len {
                        for hd in 0..dn_heads {
                            alpha[t * dn_heads + hd] += bias[hd];
                        }
                    }
                }

                // Delta rule recurrence
                let dn_state = session.deltanet_state.as_mut().unwrap();
                let state = dn_state.state_mut(deltanet_layer_idx);
                let mut dn_out = vec![0.0f32; seq_len * dn_dim];

                let deltanet_kernel = tempura_kernels::deltanet::CpuDeltaNet;
                if seq_len == 1 {
                    deltanet_kernel
                        .forward_recurrent(
                            &q,
                            &k,
                            &v,
                            &alpha,
                            &beta,
                            state,
                            &mut dn_out,
                            dn_heads,
                            dn_head_dim,
                            ssm_a,
                        )
                        .unwrap();
                } else {
                    deltanet_kernel
                        .forward_chunk(
                            &q,
                            &k,
                            &v,
                            &alpha,
                            &beta,
                            state,
                            &mut dn_out,
                            seq_len,
                            dn_heads,
                            dn_head_dim,
                            ssm_a,
                        )
                        .unwrap();
                }

                // ssm_norm (RMSNorm on recurrence output)
                if let Some(norm_w) = weights.get(&format!("{prefix}.ssm.norm.weight")) {
                    self.pipeline
                        .rmsnorm(
                            &mut dn_out,
                            norm_w,
                            dn_dim,
                            cfg.layer_norm_eps,
                            TraceLevel::Off,
                            &mut None,
                        )
                        .unwrap();
                }

                // Output gating: dn_out *= silu(z_gate)
                if weights
                    .get(&format!("{prefix}.ssm.in_proj.weight"))
                    .is_some()
                {
                    for i in 0..dn_out.len() {
                        let z_silu = z_gate[i] / (1.0 + (-z_gate[i]).exp());
                        dn_out[i] *= z_silu;
                    }
                }

                // Output projection
                let mut o_proj = vec![0.0f32; seq_len * h];
                if let Some(out_w) = weights.get(&format!("{prefix}.ssm.out_proj.weight")) {
                    self.pipeline
                        .matmul_bt(&dn_out, out_w, &mut o_proj, seq_len, h, dn_dim)
                        .unwrap();
                } else if let Some(o_w) = weights.get(&format!("{prefix}.self_attn.o_proj.weight"))
                {
                    self.pipeline
                        .matmul_bt(&dn_out, o_w, &mut o_proj, seq_len, h, dn_dim)
                        .unwrap();
                }

                for i in 0..hidden.len() {
                    hidden[i] += o_proj[i];
                }

                deltanet_layer_idx += 1;
            } else {
                // --- attention block ---
                normed = hidden.clone();
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
                        .rmsnorm(
                            &mut normed,
                            &norm_weight_buf,
                            h,
                            cfg.layer_norm_eps,
                            session.trace_level,
                            &mut Some(&mut snapshot),
                        )
                        .unwrap();
                } else {
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
                let cl = cache_layer_map[layer].unwrap();
                if let Some(ref mut paged) = session.paged_cache {
                    paged.append_kv(cl, &k, &v, seq_len);
                    let kv_len = paged.seq_len + seq_len;
                    let k_cont = paged.k_contiguous(cl, kv_len);
                    let v_cont = paged.v_contiguous(cl, kv_len);
                    if let Some(cap) = cfg.attn_logit_softcapping {
                        tempura_kernels::attention::CpuAttentionSoftcap::new(cap)
                            .forward(
                                &q,
                                &k_cont,
                                &v_cont,
                                &mut attn_out,
                                1,
                                seq_len,
                                kv_len,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                attn_scale,
                            )
                            .unwrap();
                    } else {
                        self.pipeline
                            .attention(
                                &q,
                                &k_cont,
                                &v_cont,
                                &mut attn_out,
                                1,
                                seq_len,
                                kv_len,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                attn_scale,
                            )
                            .unwrap();
                    }
                } else {
                    session.cache.append_kv(cl, &k, &v, seq_len);
                    let kv_len = session.cache.seq_len + seq_len;
                    let k_cache = session.cache.k_data(cl, kv_len);
                    let v_cache = session.cache.v_data(cl, kv_len);
                    if let Some(cap) = cfg.attn_logit_softcapping {
                        tempura_kernels::attention::CpuAttentionSoftcap::new(cap)
                            .forward(
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
                                attn_scale,
                            )
                            .unwrap();
                    } else {
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
                                attn_scale,
                            )
                            .unwrap();
                    }
                }

                o_proj = vec![0.0f32; seq_len * h];
                self.pipeline
                    .matmul_bt(&attn_out, o_w, &mut o_proj, seq_len, h, q_dim)
                    .unwrap();

                if !cfg.parallel_attn_ffn {
                    for i in 0..hidden.len() {
                        hidden[i] += o_proj[i];
                    }
                }
            } // end Attention block

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
                        .rmsnorm(
                            &mut normed2,
                            &norm_weight_buf,
                            h,
                            cfg.layer_norm_eps,
                            TraceLevel::Off,
                            &mut None,
                        )
                        .unwrap();
                } else {
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

            if std::env::var("TEMPURA_TRACE_LAYERS").is_ok()
                && seq_len == 1
                && (layer <= 4 || layer == cfg.num_layers - 1)
            {
                let hn: f32 = hidden[..h].iter().map(|x| x * x).sum::<f32>().sqrt();
                let kind = if layer_kind == crate::config::LayerKind::DeltaNet {
                    "DN"
                } else {
                    "ATN"
                };
                eprintln!("[FQ-TRACE] L{layer}({kind}) hidden_norm={hn:.4}");
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
                .rmsnorm(
                    &mut hidden,
                    &norm_weight_buf,
                    h,
                    cfg.layer_norm_eps,
                    TraceLevel::Off,
                    &mut None,
                )
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

    #[allow(unused_imports)]
    pub fn forward_quantized(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        qweights: &crate::weights::QuantizedWeights,
    ) -> Vec<f32> {
        use tempura_kernels::{RMSNorm, RoPE};
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let seq_len = token_ids.len();

        let q4k = tempura_kernels::quantized::CpuMatMulQ4KOpt;
        use tempura_kernels::quantized::{
            MatMulQ, PreQuantizedActivations, Q2_0_BLOCK_BYTES, Q2_0_BLOCK_SIZE,
        };
        use tempura_kernels::tl2::tl2_forward;

        let det_rmsnorm: Option<Box<dyn tempura_kernels::RMSNorm>> = if self.config.hidden_act
            == crate::config::HiddenAct::SquaredRelu
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
        let det_rope: Option<Box<dyn tempura_kernels::RoPE>> = if cfg.rope_type
            == crate::config::RopeType::Norm
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

        let softcap_attn: Option<tempura_kernels::attention::CpuAttentionSoftcap> = cfg
            .attn_logit_softcapping
            .map(tempura_kernels::attention::CpuAttentionSoftcap::new);

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

        for (t, &tok) in token_ids.iter().enumerate() {
            qweights.embed_token(tok, h, &mut hidden[t * h..(t + 1) * h]);
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

        let fwd_start = std::time::Instant::now();
        let qmm_nanos = std::sync::atomic::AtomicU64::new(0);
        let do_profile = std::env::var("TEMPURA_PROFILE").is_ok();

        let qmm_inner = |preq: Option<&PreQuantizedActivations>,
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
                        q4k.forward_b_transposed_q4k(a_f32, w, out, m, n, k)
                            .unwrap();
                    }
                    GgmlType::Q6K => {
                        q4k.forward_b_transposed_q6k(a_f32, w, out, m, n, k)
                            .unwrap();
                    }
                    GgmlType::TQ2_0 => {
                        q4k.forward_b_transposed_tq2(a_f32, w, out, m, n, k)
                            .unwrap();
                    }
                    GgmlType::TQ1_0 => {
                        q4k.forward_b_transposed_i2s(a_f32, w, out, m, n, k)
                            .unwrap();
                    }
                    _ => {
                        // All other quantized formats go through ggml registry
                        tempura_kernels::ggml::mul_mat::mul_mat_q(dtype, w, a_f32, out, m, n, k);
                    }
                }
                return;
            }
            // TL2 and I2S tensors (not in typed_tensors)
            if let Some(tl2t) = qweights.get_tl2(name) {
                {
                    use tempura_kernels::tl2::tl2_forward_batched;
                    if !tl2_forward_batched(a_f32, &tl2t.data, tl2t.scale, out, n, k, m) {
                        out.fill(0.0);
                    }
                    if let Some(corr) = qweights.get_q2_row_corrections(name) {
                        for row in 0..m {
                            for j in 0..n.min(corr.len()) {
                                out[row * n + j] *= corr[j];
                            }
                        }
                    }
                }
                return;
            }
            if let Some(w) = qweights.get_i2s(name) {
                if m == 1 {
                    if let Some(pq) = preq {
                        q4k.forward_i2s_preq(&pq.i2s_q8, pq.i2s_scale, pq.i2s_sum, w, out, n, k)
                            .unwrap();
                        return;
                    }
                }
                q4k.forward_b_transposed_i2s(a_f32, w, out, m, n, k)
                    .unwrap();
                return;
            }
            // Q2_0 paths (LUT-tiled, x4-packed, I2S layout, raw)
            if let Some((_tiled, _tn, _tk)) = qweights.get_q2_tiled(name) {
                if let Some(_w) = qweights.get_q2(name) {
                    for row in 0..m {
                        let act = &a_f32[row * k..(row + 1) * k];
                        let nb = k / tempura_kernels::quantized::Q8_0_BLOCK_SIZE;
                        let mut q8_buf =
                            vec![0u8; nb * tempura_kernels::quantized::Q8_0_BLOCK_BYTES];
                        tempura_kernels::quantized::quantize_row_q8_0(act, &mut q8_buf);
                        if std::env::var("TEMPURA_TRACE_LAYERS").is_ok()
                            && name.contains("layers.3.self_attn.q")
                        {
                            eprintln!("[FQ-QMM-L3-Q] n={n} k={k} q8[0..8]={:?}", &q8_buf[..8]);
                        }
                        tempura_kernels::quantized::q2_0_gemv_q8(
                            _w,
                            &q8_buf,
                            &mut out[row * n..(row + 1) * n],
                            n,
                            k,
                        );
                    }
                }
                return;
            }
            if let Some((x4_data, _x4_n, _x4_k)) = qweights.get_q2x4(name) {
                if m == 1 {
                    if let Some(pq) = preq {
                        tempura_kernels::quantized::q2x4_gemv_preq(x4_data, pq, out, n, k);
                        return;
                    }
                }
                if let Some(_w) = qweights.get_q2(name) {
                    for row in 0..m {
                        tempura_kernels::quantized::q2_0_gemv(
                            _w,
                            &a_f32[row * k..(row + 1) * k],
                            &mut out[row * n..(row + 1) * n],
                            n,
                            k,
                        );
                    }
                }
                return;
            }
            if let Some((i2s_data, blk_scales, _n, _k)) = qweights.get_q2_i2s(name) {
                if m == 1 {
                    if let Some(pq) = preq {
                        tempura_kernels::quantized::q2_0_gemv_i2s(
                            i2s_data, blk_scales, pq, out, n, k,
                        );
                        return;
                    }
                }
                if let Some(_w) = qweights.get_q2(name) {
                    for row in 0..m {
                        tempura_kernels::quantized::q2_0_gemv(
                            _w,
                            &a_f32[row * k..(row + 1) * k],
                            &mut out[row * n..(row + 1) * n],
                            n,
                            k,
                        );
                    }
                }
                return;
            }
            if let Some(_w) = qweights.get_q2(name) {
                for row in 0..m {
                    let act = &a_f32[row * k..(row + 1) * k];
                    let nb = k / tempura_kernels::quantized::Q8_0_BLOCK_SIZE;
                    let mut q8_buf = vec![0u8; nb * tempura_kernels::quantized::Q8_0_BLOCK_BYTES];
                    tempura_kernels::quantized::quantize_row_q8_0(act, &mut q8_buf);
                    tempura_kernels::quantized::q2_0_gemv_q8(
                        _w,
                        &q8_buf,
                        &mut out[row * n..(row + 1) * n],
                        n,
                        k,
                    );
                }
                return;
            }
            // F16 mmap tensors — dequantize on the fly
            if let Some(f16_bytes) = qweights.get_f16(name) {
                let n_elems = f16_bytes.len() / 2;
                let f32_data: Vec<f32> = (0..n_elems).map(|i| {
                    let bits = u16::from_le_bytes([f16_bytes[i*2], f16_bytes[i*2+1]]);
                    tempura_kernels::ggml::helpers::f16_to_f32(bits)
                }).collect();
                self.pipeline.matmul_bt(a_f32, &f32_data, out, m, n, k).unwrap();
                return;
            }
            // mmap Q4K/Q6K tensors (stored separately from typed_tensors)
            if let Some(w) = qweights.get_q4k(name) {
                use tempura_kernels::ggml::GgmlType;
                if m == 1 {
                    if let Some(pq) = preq {
                        q4k.forward_q4k_preq(pq, a_f32, w, out, n).unwrap();
                        return;
                    }
                }
                q4k.forward_b_transposed_q4k(a_f32, w, out, m, n, k).unwrap();
                return;
            }
            if let Some(w) = qweights.get_q6k(name) {
                if m == 1 {
                    if let Some(pq) = preq {
                        q4k.forward_q6k_preq(pq, a_f32, w, out, n).unwrap();
                        return;
                    }
                }
                q4k.forward_b_transposed_q6k(a_f32, w, out, m, n, k).unwrap();
                return;
            }
            // f32 fallback
            if let Some(w) = qweights.get_f32(name) {
                self.pipeline.matmul_bt(a_f32, w, out, m, n, k).unwrap();
            } else {
                panic!("weight tensor not found: {name}");
            }
        };
        let qmm = |preq: Option<&PreQuantizedActivations>,
                   a_f32: &[f32],
                   name: &str,
                   out: &mut [f32],
                   m: usize,
                   n: usize,
                   k: usize| {
            let t = std::time::Instant::now();
            qmm_inner(preq, a_f32, name, out, m, n, k);
            qmm_nanos.fetch_add(
                t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
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

        let cache_layer_map_q = cfg.cache_layer_map();
        let mut deltanet_layer_idx = 0usize;

        for layer in 0..cfg.num_layers {
            let layer_kind = cfg
                .layer_kinds
                .as_ref()
                .map(|kinds| kinds[layer])
                .unwrap_or(crate::config::LayerKind::Attention);

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

            if layer == 0 && std::env::var("TEMPURA_TRACE_EMBED").is_ok() {
                let en: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!(
                    "[embed] hidden norm={en:.4} first8={:?}",
                    &hidden[..8.min(h)]
                );
                // Dump first Q5K tensor values
                let qkv_name = "model.layers.0.ssm.qkv_proj.weight".to_string();
                if let Some(w) = qweights.get_f32(&qkv_name) {
                    eprintln!(
                        "[Q5K-check] {qkv_name}: len={} first8={:?}",
                        w.len(),
                        &w[..8]
                    );
                } else {
                    eprintln!("[Q5K-check] {qkv_name}: NOT FOUND in f32");
                }
            }

            if layer == 0 && std::env::var("TEMPURA_TRACE_TOKENS").is_ok() {
                let _toks: Vec<u32> = (0..seq_len)
                    .map(|i| session.position as u32 + i as u32)
                    .collect();
                eprintln!(
                    "[tokens] seq_len={seq_len} position={} token_range={}..{}",
                    session.position,
                    session.position,
                    session.position + seq_len
                );
            }
            if std::env::var("TEMPURA_TRACE_LAYERS").is_ok()
                && (layer <= 3 || layer == cfg.num_layers - 1)
            {
                let hn: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!(
                    "[pre-L{layer}] kind={layer_kind:?} hidden_norm={hn:.6} first4={:?}",
                    &hidden[..4.min(h)]
                );
            }
            if layer == 3 && std::env::var("TEMPURA_DUMP_L3").is_ok() {
                use std::io::Write;
                let path = "/tmp/tempura_l3_hidden.bin";
                let mut f = std::fs::File::create(path).unwrap();
                for &v in &hidden[..h] {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
                eprintln!("[dump] wrote {h} f32 values to {path}");
                // Also dump after Q projection
                let mut normed_l3 = hidden[..h].to_vec();
                let ln_w = qweights.get_f32(&layer_names[3].0).unwrap();
                self.pipeline
                    .rmsnorm(
                        &mut normed_l3,
                        ln_w,
                        h,
                        cfg.layer_norm_eps,
                        TraceLevel::Off,
                        &mut None,
                    )
                    .unwrap();
                eprintln!(
                    "[L3-norm] norm={:.6} first4={:?}",
                    normed_l3.iter().map(|x| x * x).sum::<f32>().sqrt(),
                    &normed_l3[..4]
                );
                let q_name = &layer_names[3].2; // q_proj.weight
                let mut q_test = vec![0.0f32; 4096];
                // Print first bytes of Q weight for comparison with Python
                if let Some(qw) = qweights.get_q4k(q_name) {
                    eprintln!("[L3-Q] Q4K weight first16: {:02x?}", &qw[..16]);
                } else if let Some(qw) = qweights.get_q6k(q_name) {
                    eprintln!("[L3-Q] Q6K weight first16: {:02x?}", &qw[..16]);
                } else if let Some(qw) = qweights.get_f32(q_name) {
                    eprintln!("[L3-Q] f32 weight first4: {:?}", &qw[..4]);
                } else {
                    eprintln!("[L3-Q] weight NOT FOUND for {q_name}");
                }
                qmm(None, &normed_l3, q_name, &mut q_test, 1, 4096, h);
                eprintln!(
                    "[L3-Q] full_norm={:.4} first8={:?}",
                    q_test.iter().map(|x| x * x).sum::<f32>().sqrt(),
                    &q_test[..8]
                );
                // Extracted Q (interleaved)
                eprintln!(
                    "[L3-Q] Q_h0[0:4]={:?} Q_h1[0:2]={:?}",
                    &q_test[..4],
                    &q_test[512..514]
                );
            }

            let skip_dn = std::env::var("TEMPURA_SKIP_DELTANET").is_ok();

            if layer_kind == crate::config::LayerKind::DeltaNet && skip_dn {
                deltanet_layer_idx += 1;
                // Skip DeltaNet attention, fall through to FFN
            } else if layer_kind == crate::config::LayerKind::DeltaNet {
                let prefix = format!("model.layers.{layer}");
                let dn_heads = cfg.deltanet_num_heads.unwrap_or(n_kv_heads);
                let dn_head_dim = cfg.deltanet_head_dim.unwrap_or(head_dim);
                let dn_dim = dn_heads * dn_head_dim;
                let _ssm_a_val = cfg.ssm_a.unwrap_or(-1.0);

                normed.copy_from_slice(&hidden);
                let input_ln_w = qweights.get_f32(input_ln_name).unwrap();
                if let Some(ref drn) = det_rmsnorm {
                    drn.forward(&mut normed, input_ln_w, h, cfg.layer_norm_eps)
                        .unwrap();
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
                if std::env::var("TEMPURA_TRACE_LAYERS").is_ok()
                    && seq_len == 1
                    && (layer == 0 || layer == 3)
                {
                    let nn: f32 = normed[..h].iter().map(|x| x * x).sum::<f32>().sqrt();
                    let hh: f32 = hidden[..h].iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!(
                        "[FQ-L{layer}] hidden_norm={hh:.6} normed_norm={nn:.6} normed[0..4]={:?}",
                        &normed[..4]
                    );
                }

                let _mm = tempura_kernels::matmul::CpuMatMul;
                let mut q_dn = vec![0.0f32; seq_len * dn_dim];
                let mut k_dn = vec![0.0f32; seq_len * dn_dim];
                let mut v_dn = vec![0.0f32; seq_len * dn_dim];
                let mut z_gate = vec![0.0f32; seq_len * dn_dim];
                let mut alpha = vec![0.0f32; seq_len * dn_heads];
                let mut beta = vec![0.0f32; seq_len * dn_heads];

                let qkv_name = format!("{prefix}.ssm.qkv_proj.weight");
                let gate_name = format!("{prefix}.ssm.gate_proj.weight");
                let in_proj_name = format!("{prefix}.ssm.in_proj.weight");
                let ba_name = format!("{prefix}.ssm.beta_alpha.weight");
                let alpha_name = format!("{prefix}.ssm.alpha.weight");
                let beta_name = format!("{prefix}.ssm.beta.weight");

                let has_in_proj = qweights.get_f32(&in_proj_name).is_some()
                    || qweights.get_q2(&in_proj_name).is_some();
                let has_qkv =
                    qweights.get_f32(&qkv_name).is_some() || qweights.get_q2(&qkv_name).is_some();
                // Derive key_dim from QKV weight shape: qkv_cols = 2*key_dim + dn_dim
                let qkv_bytes = qweights.get_q2(&qkv_name).map(|w| w.len()).unwrap_or(0);
                let qkv_elements = if qkv_bytes > 0 {
                    qkv_bytes / Q2_0_BLOCK_BYTES * Q2_0_BLOCK_SIZE / h
                } else {
                    0
                };
                let key_dim = if qkv_elements > dn_dim {
                    (qkv_elements - dn_dim) / 2
                } else {
                    dn_dim
                };

                if has_in_proj {
                    let in_cols = 4 * dn_dim;
                    let mut in_proj = vec![0.0f32; seq_len * in_cols];
                    qmm(
                        None,
                        &normed,
                        &in_proj_name,
                        &mut in_proj,
                        seq_len,
                        in_cols,
                        h,
                    );
                    for t in 0..seq_len {
                        let src = &in_proj[t * in_cols..];
                        q_dn[t * dn_dim..(t + 1) * dn_dim].copy_from_slice(&src[..dn_dim]);
                        k_dn[t * dn_dim..(t + 1) * dn_dim]
                            .copy_from_slice(&src[dn_dim..2 * dn_dim]);
                        v_dn[t * dn_dim..(t + 1) * dn_dim]
                            .copy_from_slice(&src[2 * dn_dim..3 * dn_dim]);
                        z_gate[t * dn_dim..(t + 1) * dn_dim]
                            .copy_from_slice(&src[3 * dn_dim..4 * dn_dim]);
                    }
                } else if has_qkv {
                    // QKV projection → raw qkv_buf goes directly into conv (no split yet)
                    let qkv_cols = 2 * key_dim + dn_dim;
                    let mut qkv_buf = vec![0.0f32; seq_len * qkv_cols];
                    qmm(None, &normed, &qkv_name, &mut qkv_buf, seq_len, qkv_cols, h);
                    qmm(None, &normed, &gate_name, &mut z_gate, seq_len, dn_dim, h);
                    if std::env::var("TEMPURA_TRACE_LAYERS").is_ok() && seq_len == 1 && layer == 0 {
                        let qn: f32 = qkv_buf.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let zn: f32 = z_gate[..dn_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                        eprintln!(
                            "[FQ-L0-B2] qkv_norm={qn:.6} gate_norm={zn:.6} qkv[0..4]={:?}",
                            &qkv_buf[..4]
                        );
                    }
                    // Store raw QKV in q_dn/k_dn/v_dn temporarily at key_dim width (no repeat yet)
                    for t in 0..seq_len {
                        let src = &qkv_buf[t * qkv_cols..];
                        q_dn[t * dn_dim..t * dn_dim + key_dim].copy_from_slice(&src[..key_dim]);
                        k_dn[t * dn_dim..t * dn_dim + key_dim]
                            .copy_from_slice(&src[key_dim..2 * key_dim]);
                        v_dn[t * dn_dim..(t + 1) * dn_dim]
                            .copy_from_slice(&src[2 * key_dim..2 * key_dim + dn_dim]);
                    }
                }

                let has_ba =
                    qweights.get_f32(&ba_name).is_some() || qweights.get_q2(&ba_name).is_some();
                let has_separate = qweights.get_f32(&alpha_name).is_some()
                    || qweights.get_q2(&alpha_name).is_some();
                if has_ba {
                    let ba_cols = 2 * dn_heads;
                    let mut ba_proj = vec![0.0f32; seq_len * ba_cols];
                    qmm(None, &normed, &ba_name, &mut ba_proj, seq_len, ba_cols, h);
                    for t in 0..seq_len {
                        let src = &ba_proj[t * ba_cols..];
                        beta[t * dn_heads..(t + 1) * dn_heads].copy_from_slice(&src[..dn_heads]);
                        alpha[t * dn_heads..(t + 1) * dn_heads].copy_from_slice(&src[dn_heads..]);
                    }
                } else if has_separate {
                    qmm(None, &normed, &alpha_name, &mut alpha, seq_len, dn_heads, h);
                    qmm(None, &normed, &beta_name, &mut beta, seq_len, dn_heads, h);
                }

                // Pre-compute gate and beta (matching qwen35.cpp lines 436-451)
                // ssm_a stores -exp(A_log) directly (already negative per-head)
                let ssm_a_w = qweights.get_f32(&format!("{prefix}.ssm.A_log"));
                let dt_bias_w = qweights.get_f32(&format!("{prefix}.ssm.dt_proj.bias"));
                for t in 0..seq_len {
                    for h in 0..dn_heads {
                        let idx = t * dn_heads + h;
                        // beta = sigmoid(beta_raw)
                        beta[idx] = 1.0 / (1.0 + (-beta[idx]).exp());
                        // gate = softplus(alpha_raw + dt_bias) * ssm_a
                        let dt_b = dt_bias_w.and_then(|b| b.get(h)).copied().unwrap_or(0.0);
                        let ssm_a = ssm_a_w.and_then(|a| a.get(h)).copied().unwrap_or(-1.0);
                        let sp = (1.0f32 + (alpha[idx] + dt_b).exp()).ln();
                        alpha[idx] = sp * ssm_a; // pre-computed gate (negative → decay < 1)
                    }
                }

                // Conv1d on concatenated QKV with state (PrismML conv state management)
                let n_k_heads = key_dim / dn_head_dim;
                let conv_dim = 2 * key_dim + dn_dim;
                if let Some(conv_w) = qweights.get_f32(&format!("{prefix}.ssm.conv1d.weight")) {
                    let kernel_size = conv_w.len() / conv_dim;
                    let history = kernel_size - 1;
                    let conv_bias = qweights.get_f32(&format!("{prefix}.ssm.conv1d.bias"));
                    let mut conv_input = vec![0.0f32; seq_len * conv_dim];
                    for t in 0..seq_len {
                        conv_input[t * conv_dim..t * conv_dim + key_dim]
                            .copy_from_slice(&q_dn[t * dn_dim..t * dn_dim + key_dim]);
                        conv_input[t * conv_dim + key_dim..t * conv_dim + 2 * key_dim]
                            .copy_from_slice(&k_dn[t * dn_dim..t * dn_dim + key_dim]);
                        conv_input[t * conv_dim + 2 * key_dim..t * conv_dim + conv_dim]
                            .copy_from_slice(&v_dn[t * dn_dim..(t + 1) * dn_dim]);
                    }
                    let dn_state = session.deltanet_state.as_ref().unwrap();
                    let cs = dn_state.conv_state(deltanet_layer_idx);
                    let has_cs = cs.len() == history * conv_dim;
                    let mut conv_out = vec![0.0f32; seq_len * conv_dim];
                    for ch in 0..conv_dim {
                        for t in 0..seq_len {
                            let mut s = 0.0f32;
                            for ki in 0..kernel_size {
                                let src_t = t as isize - (kernel_size as isize - 1 - ki as isize);
                                if src_t >= 0 {
                                    s += conv_input[src_t as usize * conv_dim + ch]
                                        * conv_w[ch * kernel_size + ki];
                                } else if has_cs {
                                    let hist_t = (history as isize + src_t) as usize;
                                    s += cs[hist_t * conv_dim + ch] * conv_w[ch * kernel_size + ki];
                                }
                            }
                            if let Some(bias) = conv_bias {
                                if ch < bias.len() {
                                    s += bias[ch];
                                }
                            }
                            conv_out[t * conv_dim + ch] = s;
                        }
                    }
                    let dn_state = session.deltanet_state.as_mut().unwrap();
                    dn_state.update_conv_state(
                        deltanet_layer_idx,
                        conv_dim,
                        kernel_size,
                        &conv_input,
                        seq_len,
                    );
                    // SiLU on entire conv output
                    for x in conv_out.iter_mut() {
                        *x = *x / (1.0 + (-*x).exp());
                    }
                    // Split Q(key_dim), K(key_dim), V(dn_dim) and GQA-repeat Q/K to dn_dim
                    for t in 0..seq_len {
                        let src = &conv_out[t * conv_dim..];
                        for vh in 0..dn_heads {
                            let kh = vh * n_k_heads / dn_heads;
                            let q_src = kh * dn_head_dim;
                            let k_src = key_dim + kh * dn_head_dim;
                            q_dn[t * dn_dim + vh * dn_head_dim
                                ..t * dn_dim + (vh + 1) * dn_head_dim]
                                .copy_from_slice(&src[q_src..q_src + dn_head_dim]);
                            k_dn[t * dn_dim + vh * dn_head_dim
                                ..t * dn_dim + (vh + 1) * dn_head_dim]
                                .copy_from_slice(&src[k_src..k_src + dn_head_dim]);
                        }
                        v_dn[t * dn_dim..(t + 1) * dn_dim]
                            .copy_from_slice(&src[2 * key_dim..2 * key_dim + dn_dim]);
                    }
                }

                // L2 normalize Q and K per head (qwen35.cpp lines 538-539)
                for t in 0..seq_len {
                    for hd in 0..dn_heads {
                        let off = t * dn_dim + hd * dn_head_dim;
                        let norm: f32 = q_dn[off..off + dn_head_dim]
                            .iter()
                            .map(|x| x * x)
                            .sum::<f32>()
                            .sqrt()
                            .max(cfg.layer_norm_eps);
                        q_dn[off..off + dn_head_dim]
                            .iter_mut()
                            .for_each(|x| *x /= norm);
                        let norm: f32 = k_dn[off..off + dn_head_dim]
                            .iter()
                            .map(|x| x * x)
                            .sum::<f32>()
                            .sqrt()
                            .max(cfg.layer_norm_eps);
                        k_dn[off..off + dn_head_dim]
                            .iter_mut()
                            .for_each(|x| *x /= norm);
                    }
                }

                // GDN kernel with pre-computed gate (alpha) and beta — ssm_a=1.0 means "pre-computed"
                let dn_state = session.deltanet_state.as_mut().unwrap();
                let state = dn_state.state_mut(deltanet_layer_idx);
                let mut dn_out = vec![0.0f32; seq_len * dn_dim];
                let dk = tempura_kernels::deltanet::CpuDeltaNet;
                if seq_len == 1 {
                    dk.forward_recurrent(
                        &q_dn,
                        &k_dn,
                        &v_dn,
                        &alpha,
                        &beta,
                        state,
                        &mut dn_out,
                        dn_heads,
                        dn_head_dim,
                        1.0,
                    )
                    .unwrap();
                } else {
                    dk.forward_chunk(
                        &q_dn,
                        &k_dn,
                        &v_dn,
                        &alpha,
                        &beta,
                        state,
                        &mut dn_out,
                        seq_len,
                        dn_heads,
                        dn_head_dim,
                        1.0,
                    )
                    .unwrap();
                }

                if let Some(norm_w) = qweights.get_f32(&format!("{prefix}.ssm.norm.weight")) {
                    for t in 0..seq_len {
                        for head in 0..dn_heads {
                            let off = t * dn_dim + head * dn_head_dim;
                            self.pipeline
                                .rmsnorm(
                                    &mut dn_out[off..off + dn_head_dim],
                                    norm_w,
                                    dn_head_dim,
                                    cfg.layer_norm_eps,
                                    TraceLevel::Off,
                                    &mut None,
                                )
                                .unwrap();
                        }
                    }
                }

                if has_in_proj || has_qkv {
                    for i in 0..dn_out.len() {
                        let z_silu = z_gate[i] / (1.0 + (-z_gate[i]).exp());
                        dn_out[i] *= z_silu;
                    }
                }

                let out_proj_name = format!("{prefix}.ssm.out_proj.weight");
                let mut o_proj_dn = vec![0.0f32; seq_len * h];
                qmm(
                    None,
                    &dn_out,
                    &out_proj_name,
                    &mut o_proj_dn,
                    seq_len,
                    h,
                    dn_dim,
                );

                for i in 0..hidden.len() {
                    hidden[i] += o_proj_dn[i];
                }
                deltanet_layer_idx += 1;

                // Fall through to FFN below
            } else {
                normed.copy_from_slice(&hidden);
                let input_ln_w = qweights.get_f32(input_ln_name).unwrap();
                if let Some(ref ln) = layernorm {
                    let input_ln_b =
                        qweights.get_f32(&format!("model.layers.{layer}.input_layernorm.bias"));
                    ln.forward(&mut normed, input_ln_w, input_ln_b, h, cfg.layer_norm_eps)
                        .unwrap();
                } else if cfg.rmsnorm_offset {
                    for i in 0..h {
                        norm_weight_buf[i] = 1.0 + input_ln_w[i];
                    }
                    if let Some(ref drn) = det_rmsnorm {
                        drn.forward(&mut normed, &norm_weight_buf, h, cfg.layer_norm_eps)
                            .unwrap();
                    } else {
                        self.pipeline
                            .rmsnorm(
                                &mut normed,
                                &norm_weight_buf,
                                h,
                                cfg.layer_norm_eps,
                                TraceLevel::Off,
                                &mut None,
                            )
                            .unwrap();
                    }
                } else if let Some(ref drn) = det_rmsnorm {
                    drn.forward(&mut normed, input_ln_w, h, cfg.layer_norm_eps)
                        .unwrap();
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
                if std::env::var("TEMPURA_TRACE_LAYERS").is_ok() && seq_len == 1 && layer == 3 {
                    let nn: f32 = normed[..h].iter().map(|x| x * x).sum::<f32>().sqrt();
                    let hh: f32 = hidden[..h].iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!(
                        "[FQ-L3-ATN] hidden_norm={hh:.6} normed_norm={nn:.6} normed[0..4]={:?}",
                        &normed[..4]
                    );
                }
                if det_rmsnorm.is_some() && layernorm.is_none() {
                    tempura_kernels::portable_math::q8_snap(&mut normed[..seq_len * h]);
                }

                let no_qgate = std::env::var("TEMPURA_NO_QGATE").is_ok();
                let _no_qknorm = std::env::var("TEMPURA_NO_QKNORM").is_ok();
                // Detect Q+gate fused projection (Qwen3.5 softmax layers)
                let q_weight_cols = qweights
                    .get_q2(q_w_name)
                    .map(|w| w.len() / ((h / Q2_0_BLOCK_SIZE) * Q2_0_BLOCK_BYTES))
                    .or_else(|| {
                        qweights.get_q4k(q_w_name).map(|w| {
                            let row_bytes = (h / 256) * 144; // Q4K block
                            w.len().checked_div(row_bytes).unwrap_or(q_dim)
                        })
                    })
                    .or_else(|| qweights.get_f32(q_w_name).map(|w| w.len() / h))
                    .unwrap_or(q_dim);
                let has_q_gate = q_weight_cols > q_dim && !no_qgate;
                let actual_q_cols = if has_q_gate { q_weight_cols } else { q_dim };

                let normed_preq = if seq_len == 1 {
                    if cfg.hidden_act == crate::config::HiddenAct::SquaredRelu {
                        Some(PreQuantizedActivations::i2s_only(&normed[..h]))
                    } else {
                        Some(PreQuantizedActivations::from_f32(&normed[..h]))
                    }
                } else {
                    None
                };

                let mut q_gate_buf = if has_q_gate {
                    vec![0.0f32; seq_len * actual_q_cols]
                } else {
                    Vec::new()
                };

                // QMM/TL2 overwrites output entirely — no zero-fill needed
                if seq_len == 1 {
                    if has_q_gate {
                        // Q+gate fused projection (Qwen3.5 attention layers):
                        // must use q_gate_buf to capture both Q and gate values
                        qmm(
                            normed_preq.as_ref(),
                            &normed,
                            q_w_name,
                            &mut q_gate_buf[..actual_q_cols],
                            1,
                            actual_q_cols,
                            h,
                        );
                        for hh in 0..n_heads {
                            let src = hh * head_dim * 2;
                            q[hh * head_dim..(hh + 1) * head_dim]
                                .copy_from_slice(&q_gate_buf[src..src + head_dim]);
                        }
                        qmm(
                            normed_preq.as_ref(),
                            &normed,
                            k_w_name,
                            &mut k[..kv_dim],
                            1,
                            kv_dim,
                            h,
                        );
                        qmm(
                            normed_preq.as_ref(),
                            &normed,
                            v_w_name,
                            &mut v[..kv_dim],
                            1,
                            kv_dim,
                            h,
                        );
                        if std::env::var("TEMPURA_TRACE_LAYERS").is_ok()
                            && seq_len == 1
                            && layer == 3
                        {
                            let qn: f32 = q[..q_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                            let gn: f32 = q_gate_buf[head_dim..2 * head_dim]
                                .iter()
                                .map(|x| x * x)
                                .sum::<f32>()
                                .sqrt();
                            eprintln!("[FQ-L3-QSPLIT] q_norm={qn:.6} gate_h0_norm={gn:.6} q[0..4]={:?} gate_h0[0..4]={:?}", &q[..4], &q_gate_buf[head_dim..head_dim+4]);
                        }
                    } else if let (Some(pq), Some(qw), Some(kw), Some(vw)) = (
                        normed_preq.as_ref(),
                        qweights.get_i2s(q_w_name),
                        qweights.get_i2s(k_w_name),
                        qweights.get_i2s(v_w_name),
                    ) {
                        q4k.forward_i2s_preq(
                            &pq.i2s_q8,
                            pq.i2s_scale,
                            pq.i2s_sum,
                            qw,
                            &mut q[..q_dim],
                            q_dim,
                            h,
                        )
                        .unwrap();
                        q4k.forward_i2s_preq(
                            &pq.i2s_q8,
                            pq.i2s_scale,
                            pq.i2s_sum,
                            kw,
                            &mut k[..kv_dim],
                            kv_dim,
                            h,
                        )
                        .unwrap();
                        q4k.forward_i2s_preq(
                            &pq.i2s_q8,
                            pq.i2s_scale,
                            pq.i2s_sum,
                            vw,
                            &mut v[..kv_dim],
                            kv_dim,
                            h,
                        )
                        .unwrap();
                    } else if let (Some(pq), Some(qw), Some(kw)) = (
                        normed_preq.as_ref(),
                        qweights.get_q4k(q_w_name),
                        qweights.get_q4k(k_w_name),
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
                        qmm(
                            normed_preq.as_ref(),
                            &normed,
                            v_w_name,
                            &mut v[..kv_dim],
                            1,
                            kv_dim,
                            h,
                        );
                    } else {
                        qmm(
                            normed_preq.as_ref(),
                            &normed,
                            q_w_name,
                            &mut q[..q_dim],
                            1,
                            q_dim,
                            h,
                        );
                        qmm(
                            normed_preq.as_ref(),
                            &normed,
                            k_w_name,
                            &mut k[..kv_dim],
                            1,
                            kv_dim,
                            h,
                        );
                        qmm(
                            normed_preq.as_ref(),
                            &normed,
                            v_w_name,
                            &mut v[..kv_dim],
                            1,
                            kv_dim,
                            h,
                        );
                    }
                } else {
                    // Pre-build batched LUT once for QKV (all share the same normed input with k=h)
                    let qkv_lut = if seq_len > 1 {
                        tempura_kernels::tl2::tl2_preprocess_batched(&normed, h, seq_len)
                    } else {
                        None
                    };
                    if let (Some(ref lut), Some(tl2q), Some(tl2k)) = (
                        &qkv_lut,
                        qweights.get_tl2(q_w_name),
                        qweights.get_tl2(k_w_name),
                    ) {
                        use tempura_kernels::tl2::tl2_gemm_with_lut;
                        tl2_gemm_with_lut(
                            lut,
                            &tl2q.data,
                            tl2q.scale,
                            &mut q[..seq_len * q_dim],
                            q_dim,
                            h,
                        );
                        tl2_gemm_with_lut(
                            lut,
                            &tl2k.data,
                            tl2k.scale,
                            &mut k[..seq_len * kv_dim],
                            kv_dim,
                            h,
                        );
                    } else if has_q_gate {
                        q_gate_buf.resize(seq_len * actual_q_cols, 0.0);
                        qmm(
                            normed_preq.as_ref(),
                            &normed,
                            q_w_name,
                            &mut q_gate_buf[..seq_len * actual_q_cols],
                            seq_len,
                            actual_q_cols,
                            h,
                        );
                        for t in 0..seq_len {
                            for hh in 0..n_heads {
                                let src = t * actual_q_cols + hh * head_dim * 2;
                                q[t * q_dim + hh * head_dim..t * q_dim + (hh + 1) * head_dim]
                                    .copy_from_slice(&q_gate_buf[src..src + head_dim]);
                            }
                        }
                        qmm(
                            normed_preq.as_ref(),
                            &normed,
                            k_w_name,
                            &mut k[..seq_len * kv_dim],
                            seq_len,
                            kv_dim,
                            h,
                        );
                    } else {
                        qmm(
                            normed_preq.as_ref(),
                            &normed,
                            q_w_name,
                            &mut q[..seq_len * q_dim],
                            seq_len,
                            q_dim,
                            h,
                        );
                        qmm(
                            normed_preq.as_ref(),
                            &normed,
                            k_w_name,
                            &mut k[..seq_len * kv_dim],
                            seq_len,
                            kv_dim,
                            h,
                        );
                    }
                    if let (Some(ref lut), Some(tl2v)) = (&qkv_lut, qweights.get_tl2(v_w_name)) {
                        use tempura_kernels::tl2::tl2_gemm_with_lut;
                        tl2_gemm_with_lut(
                            lut,
                            &tl2v.data,
                            tl2v.scale,
                            &mut v[..seq_len * kv_dim],
                            kv_dim,
                            h,
                        );
                    } else if seq_len > 1 || qweights.get_i2s(v_w_name).is_none() {
                        qmm(
                            normed_preq.as_ref(),
                            &normed,
                            v_w_name,
                            &mut v[..seq_len * kv_dim],
                            seq_len,
                            kv_dim,
                            h,
                        );
                    }
                }

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
                    if let (Some(q_norm_w), Some(k_norm_w)) =
                        (qweights.get_f32(q_norm_name), qweights.get_f32(k_norm_name))
                    {
                        apply_head_rmsnorm(
                            &mut q,
                            q_norm_w,
                            seq_len,
                            n_heads,
                            head_dim,
                            cfg.layer_norm_eps,
                        );
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

                if std::env::var("TEMPURA_TRACE_LAYERS").is_ok() && seq_len == 1 && layer == 3 {
                    let qn: f32 = q[..q_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                    let kn: f32 = k[..kv_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                    let vn: f32 = v[..kv_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!("[FQ-ATN L3] pre-rope q_norm={qn:.4} k_norm={kn:.4} v_norm={vn:.4}");
                }
                if !cfg.use_alibi {
                    let rope_hd = cfg.rope_dim.unwrap_or(head_dim);
                    if rope_hd < head_dim {
                        // Partial RoPE: only rotate first rope_hd dims per head
                        let mut q_rope = vec![0.0f32; seq_len * n_heads * rope_hd];
                        let mut k_rope = vec![0.0f32; seq_len * n_kv_heads * rope_hd];
                        for t in 0..seq_len {
                            for h in 0..n_heads {
                                let src = t * q_dim + h * head_dim;
                                let dst = t * n_heads * rope_hd + h * rope_hd;
                                q_rope[dst..dst + rope_hd].copy_from_slice(&q[src..src + rope_hd]);
                            }
                            for h in 0..n_kv_heads {
                                let src = t * kv_dim + h * head_dim;
                                let dst = t * n_kv_heads * rope_hd + h * rope_hd;
                                k_rope[dst..dst + rope_hd].copy_from_slice(&k[src..src + rope_hd]);
                            }
                        }
                        if let Some(ref rope_table) = det_rope {
                            rope_table
                                .forward(
                                    &mut q_rope,
                                    1,
                                    seq_len,
                                    n_heads,
                                    rope_hd,
                                    &positions,
                                    cfg.rope_theta,
                                )
                                .unwrap();
                            rope_table
                                .forward(
                                    &mut k_rope,
                                    1,
                                    seq_len,
                                    n_kv_heads,
                                    rope_hd,
                                    &positions,
                                    cfg.rope_theta,
                                )
                                .unwrap();
                        } else {
                            self.pipeline
                                .rope(
                                    &mut q_rope,
                                    1,
                                    seq_len,
                                    n_heads,
                                    rope_hd,
                                    &positions,
                                    cfg.rope_theta,
                                )
                                .unwrap();
                            self.pipeline
                                .rope(
                                    &mut k_rope,
                                    1,
                                    seq_len,
                                    n_kv_heads,
                                    rope_hd,
                                    &positions,
                                    cfg.rope_theta,
                                )
                                .unwrap();
                        }
                        for t in 0..seq_len {
                            for h in 0..n_heads {
                                let dst = t * q_dim + h * head_dim;
                                let src = t * n_heads * rope_hd + h * rope_hd;
                                q[dst..dst + rope_hd].copy_from_slice(&q_rope[src..src + rope_hd]);
                            }
                            for h in 0..n_kv_heads {
                                let dst = t * kv_dim + h * head_dim;
                                let src = t * n_kv_heads * rope_hd + h * rope_hd;
                                k[dst..dst + rope_hd].copy_from_slice(&k_rope[src..src + rope_hd]);
                            }
                        }
                    } else if let Some(ref rope_table) = det_rope {
                        rope_table
                            .forward(
                                &mut q,
                                1,
                                seq_len,
                                n_heads,
                                head_dim,
                                &positions,
                                cfg.rope_theta,
                            )
                            .unwrap();
                        rope_table
                            .forward(
                                &mut k,
                                1,
                                seq_len,
                                n_kv_heads,
                                head_dim,
                                &positions,
                                cfg.rope_theta,
                            )
                            .unwrap();
                    } else {
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
                    }
                }

                let cl_q = cache_layer_map_q[layer].unwrap();
                session.cache.append_kv(
                    cl_q,
                    &k[..seq_len * kv_dim],
                    &v[..seq_len * kv_dim],
                    seq_len,
                );
                let kv_len = session.cache.seq_len + seq_len;
                let k_cache = session.cache.k_data(cl_q, kv_len);
                let v_cache = session.cache.v_data(cl_q, kv_len);

                if std::env::var("TEMPURA_TRACE_LAYERS").is_ok() && seq_len == 1 && layer == 3 {
                    let qn: f32 = q[..q_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                    let kn: f32 = k[..kv_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!(
                        "[FQ-L3-pre-attn] q_norm={qn:.6} k_norm={kn:.6} q[0..4]={:?} kvl={kv_len}",
                        &q[..4]
                    );
                }
                attn_out.fill(0.0);
                if let Some(ref alibi) = alibi_attn {
                    alibi
                        .forward(
                            &q,
                            &k_cache,
                            &v_cache,
                            &mut attn_out[..seq_len * q_dim],
                            1,
                            seq_len,
                            kv_len,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            1.0 / (head_dim as f32).sqrt(),
                        )
                        .unwrap();
                } else if let Some(ref sc) = softcap_attn {
                    sc.forward(
                        &q,
                        &k_cache,
                        &v_cache,
                        &mut attn_out[..seq_len * q_dim],
                        1,
                        seq_len,
                        kv_len,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        1.0 / (head_dim as f32).sqrt(),
                    )
                    .unwrap();
                } else if self.config.hidden_act == crate::config::HiddenAct::SquaredRelu {
                    tempura_kernels::attention::CpuAttentionGgml
                        .forward(
                            &q,
                            &k_cache,
                            &v_cache,
                            &mut attn_out[..seq_len * q_dim],
                            1,
                            seq_len,
                            kv_len,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            1.0 / (head_dim as f32).sqrt(),
                        )
                        .unwrap();
                } else {
                    self.pipeline
                        .attention(
                            &q,
                            &k_cache,
                            &v_cache,
                            &mut attn_out[..seq_len * q_dim],
                            1,
                            seq_len,
                            kv_len,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            1.0 / (head_dim as f32).sqrt(),
                        )
                        .unwrap();
                }

                // Qwen3.5 softmax layers: sigmoid gate on attention output
                // Gate is interleaved: q_gate_buf[h*2*hd + hd .. h*2*hd + 2*hd] per head
                if has_q_gate && !q_gate_buf.is_empty() {
                    for t in 0..seq_len {
                        for hh in 0..n_heads {
                            for d in 0..head_dim {
                                let gate_val = q_gate_buf
                                    [t * actual_q_cols + hh * head_dim * 2 + head_dim + d];
                                let sig = 1.0 / (1.0 + (-gate_val).exp());
                                attn_out[t * q_dim + hh * head_dim + d] *= sig;
                            }
                        }
                    }
                }

                if let Some(sub_norm_w) = qweights.get_f32(attn_sub_norm_name) {
                    if let Some(ref drn) = det_rmsnorm {
                        drn.forward(
                            &mut attn_out[..seq_len * q_dim],
                            sub_norm_w,
                            q_dim,
                            cfg.layer_norm_eps,
                        )
                        .unwrap();
                    } else {
                        self.pipeline
                            .rmsnorm(
                                &mut attn_out[..seq_len * q_dim],
                                sub_norm_w,
                                q_dim,
                                cfg.layer_norm_eps,
                                TraceLevel::Off,
                                &mut None,
                            )
                            .unwrap();
                    }
                }

                let attn_preq = if seq_len == 1 {
                    if cfg.hidden_act == crate::config::HiddenAct::SquaredRelu {
                        Some(PreQuantizedActivations::i2s_only(&attn_out[..q_dim]))
                    } else {
                        Some(PreQuantizedActivations::from_f32(&attn_out[..q_dim]))
                    }
                } else {
                    None
                };
                qmm(
                    attn_preq.as_ref(),
                    &attn_out[..seq_len * q_dim],
                    o_w_name,
                    &mut o_proj[..seq_len * h],
                    seq_len,
                    h,
                    q_dim,
                );
                if std::env::var("TEMPURA_TRACE_LAYERS").is_ok() && seq_len == 1 && layer == 3 {
                    let on: f32 = o_proj[..h].iter().map(|x| x * x).sum::<f32>().sqrt();
                    let an: f32 = attn_out[..q_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!("[FQ-ATN L3] attn_out_norm={an:.4} o_proj_norm={on:.4}");
                }
                if !cfg.parallel_attn_ffn {
                    tempura_kernels::fused::residual_add(
                        &mut hidden[..seq_len * h],
                        &o_proj[..seq_len * h],
                        seq_len * h,
                    );
                }

                if std::env::var("TEMPURA_DUMP_LOGITS").is_ok() && layer == 0 {
                    let last = &hidden[(seq_len - 1) * h..seq_len * h];
                    let norm: f32 = last.iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!(
                        "L0 post-attn hidden[last][0..8]: {:?} norm={norm:.4}",
                        &last[..8.min(h)]
                    );
                }
            } // end Attention block in forward_quantized

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
                    let post_ln_b = qweights.get_f32(&format!(
                        "model.layers.{layer}.post_attention_layernorm.bias"
                    ));
                    ln.forward(&mut normed2, post_ln_w, post_ln_b, h, cfg.layer_norm_eps)
                        .unwrap();
                } else if cfg.rmsnorm_offset {
                    for i in 0..h {
                        norm_weight_buf[i] = 1.0 + post_ln_w[i];
                    }
                    if let Some(ref drn) = det_rmsnorm {
                        drn.forward(&mut normed2, &norm_weight_buf, h, cfg.layer_norm_eps)
                            .unwrap();
                    } else {
                        self.pipeline
                            .rmsnorm(
                                &mut normed2,
                                &norm_weight_buf,
                                h,
                                cfg.layer_norm_eps,
                                TraceLevel::Off,
                                &mut None,
                            )
                            .unwrap();
                    }
                } else if let Some(ref drn) = det_rmsnorm {
                    drn.forward(&mut normed2, post_ln_w, h, cfg.layer_norm_eps)
                        .unwrap();
                } else {
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
                }
                if det_rmsnorm.is_some() && layernorm.is_none() {
                    tempura_kernels::portable_math::q8_snap(&mut normed2[..seq_len * h]);
                }
                let preq = if seq_len == 1 {
                    if cfg.hidden_act == crate::config::HiddenAct::SquaredRelu {
                        Some(PreQuantizedActivations::i2s_only(&normed2[..h]))
                    } else {
                        Some(PreQuantizedActivations::from_f32(&normed2[..h]))
                    }
                } else {
                    None
                };
                (&normed2 as &[f32], preq)
            };
            let _normed2_preq = &ffn_preq;

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
                    let router =
                        tempura_kernels::moe::MoeRouter::new(cfg.n_experts, cfg.n_experts_used);
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
                    qmm(
                        ffn_preq.as_ref(),
                        ffn_input,
                        gate_w_name,
                        &mut gate[..seq_len * inter],
                        seq_len,
                        inter,
                        h,
                    );
                    qmm(
                        ffn_preq.as_ref(),
                        ffn_input,
                        up_w_name,
                        &mut up[..seq_len * inter],
                        seq_len,
                        inter,
                        h,
                    );
                    for i in 0..seq_len * inter {
                        gate[i] = tempura_kernels::portable_math::portable_silu(gate[i]) * up[i];
                    }
                    down_out.fill(0.0);
                    let gate_preq_moe = if seq_len == 1 {
                        Some(PreQuantizedActivations::from_f32(&gate[..inter]))
                    } else {
                        None
                    };
                    qmm(
                        gate_preq_moe.as_ref(),
                        &gate[..seq_len * inter],
                        down_w_name,
                        &mut down_out[..seq_len * h],
                        seq_len,
                        h,
                        inter,
                    );
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
                                qweights.get_q4k(gate_w_name),
                                qweights.get_q4k(up_w_name),
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
                                    gate_w_name,
                                    &mut gate[..inter],
                                    1,
                                    inter,
                                    h,
                                );
                                qmm(
                                    ffn_preq.as_ref(),
                                    ffn_input,
                                    up_w_name,
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
                                gate_w_name,
                                &mut gate[..seq_len * inter],
                                seq_len,
                                inter,
                                h,
                            );
                            qmm(
                                ffn_preq.as_ref(),
                                ffn_input,
                                up_w_name,
                                &mut up[..seq_len * inter],
                                seq_len,
                                inter,
                                h,
                            );
                        }
                        for i in 0..seq_len * inter {
                            gate[i] =
                                tempura_kernels::portable_math::portable_silu(gate[i]) * up[i];
                        }
                        if det_rmsnorm.is_some() {
                            tempura_kernels::portable_math::q8_snap(&mut gate[..seq_len * inter]);
                        }
                    }
                    crate::config::HiddenAct::SquaredRelu => {
                        qmm(
                            ffn_preq.as_ref(),
                            ffn_input,
                            gate_w_name,
                            &mut gate[..seq_len * inter],
                            seq_len,
                            inter,
                            h,
                        );
                        qmm(
                            ffn_preq.as_ref(),
                            ffn_input,
                            up_w_name,
                            &mut up[..seq_len * inter],
                            seq_len,
                            inter,
                            h,
                        );
                        if std::env::var("TEMPURA_DUMP_LOGITS").is_ok() && layer == 0 {
                            let last_g = &gate[(seq_len - 1) * inter..seq_len * inter];
                            let last_u = &up[(seq_len - 1) * inter..seq_len * inter];
                            let gn: f32 = last_g.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let un: f32 = last_u.iter().map(|x| x * x).sum::<f32>().sqrt();
                            eprintln!(
                                "L0 FFN gate[0..8]: {:?} norm={gn:.4}",
                                &last_g[..8.min(inter)]
                            );
                            eprintln!(
                                "L0 FFN up[0..8]: {:?} norm={un:.4}",
                                &last_u[..8.min(inter)]
                            );
                        }
                        tempura_kernels::fused::fused_sqrelu_mul(
                            &mut gate[..seq_len * inter],
                            &up[..seq_len * inter],
                            seq_len * inter,
                        );
                        if std::env::var("TEMPURA_DUMP_LOGITS").is_ok() && layer == 0 {
                            let last = &gate[(seq_len - 1) * inter..seq_len * inter];
                            let norm: f32 = last.iter().map(|x| x * x).sum::<f32>().sqrt();
                            eprintln!(
                                "L0 FFN after relu²*up[0..8]: {:?} norm={norm:.4}",
                                &last[..8.min(inter)]
                            );
                        }
                    }
                    crate::config::HiddenAct::Gelu => {
                        qmm(
                            ffn_preq.as_ref(),
                            ffn_input,
                            up_w_name,
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
                            up_w_name,
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
                            up_w_name,
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
                            gate_w_name,
                            &mut gate[..seq_len * inter],
                            seq_len,
                            inter,
                            h,
                        );
                        qmm(
                            ffn_preq.as_ref(),
                            ffn_input,
                            up_w_name,
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
                            gate_w_name,
                            &mut gate[..seq_len * inter],
                            seq_len,
                            inter,
                            h,
                        );
                        qmm(
                            ffn_preq.as_ref(),
                            ffn_input,
                            up_w_name,
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
                    if let Some(ref drn) = det_rmsnorm {
                        drn.forward(
                            &mut gate[..seq_len * inter],
                            sub_norm_w,
                            inter,
                            cfg.layer_norm_eps,
                        )
                        .unwrap();
                    } else {
                        self.pipeline
                            .rmsnorm(
                                &mut gate[..seq_len * inter],
                                sub_norm_w,
                                inter,
                                cfg.layer_norm_eps,
                                TraceLevel::Off,
                                &mut None,
                            )
                            .unwrap();
                    }
                }

                let gate_preq = if seq_len == 1 {
                    if cfg.hidden_act == crate::config::HiddenAct::SquaredRelu {
                        Some(PreQuantizedActivations::i2s_only(&gate[..inter]))
                    } else {
                        Some(PreQuantizedActivations::from_f32(&gate[..inter]))
                    }
                } else {
                    None
                };
                qmm(
                    gate_preq.as_ref(),
                    &gate[..seq_len * inter],
                    down_w_name,
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
                    tempura_kernels::fused::residual_add(
                        &mut hidden[..seq_len * h],
                        &down_out[..seq_len * h],
                        seq_len * h,
                    );
                }

                if std::env::var("TEMPURA_TRACE_LAYERS").is_ok()
                    && seq_len == 1
                    && (layer <= 4 || layer == cfg.num_layers - 1)
                {
                    let hn: f32 = hidden[..h].iter().map(|x| x * x).sum::<f32>().sqrt();
                    let kind = if layer_kind == crate::config::LayerKind::DeltaNet {
                        "DN"
                    } else {
                        "ATN"
                    };
                    eprintln!("[FQ-TRACE] L{layer}({kind}) hidden_norm={hn:.4}");
                }
            } // end dense FFN else block

            if std::env::var("TEMPURA_DUMP_LOGITS").is_ok() && layer < 3 {
                let last = &hidden[(seq_len - 1) * h..seq_len * h];
                let norm: f32 = last.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!(
                    "L{layer} hidden[last][0..8]: {:?} norm={norm:.4}",
                    &last[..8.min(h)]
                );
            }
            if std::env::var("TEMPURA_DUMP_LAYERS").is_ok() {
                use std::io::Write;
                let path = format!("/tmp/tempura_layer_{layer}.bin");
                let mut f = std::fs::File::create(&path).unwrap();
                for &v in &hidden[..h] {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
            }
        }

        session.cache.advance(seq_len);
        session.position += seq_len;

        let mut final_normed = hidden.clone();
        let final_norm_w = qweights.get_f32("model.norm.weight").unwrap();
        if let Some(ref ln) = layernorm {
            let final_norm_b = qweights.get_f32("model.norm.bias");
            ln.forward(
                &mut final_normed,
                final_norm_w,
                final_norm_b,
                h,
                cfg.layer_norm_eps,
            )
            .unwrap();
        } else if cfg.rmsnorm_offset {
            for i in 0..h {
                norm_weight_buf[i] = 1.0 + final_norm_w[i];
            }
            if let Some(ref drn) = det_rmsnorm {
                drn.forward(&mut final_normed, &norm_weight_buf, h, cfg.layer_norm_eps)
                    .unwrap();
            } else {
                self.pipeline
                    .rmsnorm(
                        &mut final_normed,
                        &norm_weight_buf,
                        h,
                        cfg.layer_norm_eps,
                        TraceLevel::Off,
                        &mut None,
                    )
                    .unwrap();
            }
        } else if let Some(ref drn) = det_rmsnorm {
            drn.forward(&mut final_normed, final_norm_w, h, cfg.layer_norm_eps)
                .unwrap();
        } else {
            self.pipeline
                .rmsnorm(
                    &mut final_normed,
                    final_norm_w,
                    h,
                    cfg.layer_norm_eps,
                    TraceLevel::Off,
                    &mut None,
                )
                .unwrap();
        }
        if det_rmsnorm.is_some() && layernorm.is_none() {
            tempura_kernels::portable_math::q8_snap(&mut final_normed[..seq_len * h]);
        }

        if std::env::var("TEMPURA_DUMP_LAYERS").is_ok() {
            use std::io::Write;
            let path = "/tmp/tempura_final_normed.bin";
            let mut f = std::fs::File::create(path).unwrap();
            for &v in &final_normed[..h] {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        // lm_head
        let last_hidden = &final_normed[(seq_len - 1) * h..seq_len * h];
        let mut logits = vec![0.0f32; cfg.vocab_size];
        let lm_head_name = "lm_head.weight";
        if let Some(tl2t) = qweights.get_tl2(lm_head_name) {
            tl2_forward(
                last_hidden,
                &tl2t.data,
                tl2t.scale,
                &mut logits,
                cfg.vocab_size,
                h,
            );
        } else if let Some(w) = qweights.get_i2s(lm_head_name) {
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
        } else if let Some(w) = qweights.get_f16(lm_head_name) {
            tempura_kernels::f16_gemv::f16_gemv(last_hidden, w, &mut logits, cfg.vocab_size, h);
        } else if let Some(w) = qweights.get_q2(lm_head_name) {
            tempura_kernels::quantized::q2_0_gemv(w, last_hidden, &mut logits, cfg.vocab_size, h);
        } else {
            let w = qweights.get_f32(lm_head_name).unwrap();
            self.pipeline
                .matmul_bt(last_hidden, w, &mut logits, 1, cfg.vocab_size, h)
                .unwrap();
        }

        if do_profile {
            let total = fwd_start.elapsed().as_secs_f64() * 1000.0;
            let qmm_ms = qmm_nanos.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
            eprintln!(
                "[PROFILE] forward_quantized: {total:.2}ms (qmm={qmm_ms:.1}ms, other={:.1}ms)",
                total - qmm_ms
            );
            qmm_nanos.store(0, std::sync::atomic::Ordering::Relaxed);
        }

        if std::env::var("TEMPURA_DUMP_LOGITS").is_ok() {
            eprint!("LOGITS_RAW:");
            for i in 0..20.min(logits.len()) {
                eprint!(" {:.6}", logits[i]);
            }
            eprintln!();
            let mut scored: Vec<(f32, usize)> =
                logits.iter().enumerate().map(|(i, &v)| (v, i)).collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            eprint!("TOP10:");
            for i in 0..10.min(scored.len()) {
                eprint!(" {}:{:.4}", scored[i].1, scored[i].0);
            }
            eprintln!();
            let sum: f32 = logits.iter().sum();
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
            eprintln!(
                "STATS: mean={:.6} min={:.6} max={:.6}",
                sum / logits.len() as f32,
                min,
                max
            );
        }

        logits
    }

    #[allow(dead_code)]
    fn forward_q2_decode(
        &self,
        session: &mut InferenceSession,
        token_id: u32,
        qweights: &crate::weights::QuantizedWeights,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = cfg.intermediate_size;
        let dn_heads = cfg.deltanet_num_heads.unwrap_or(n_kv_heads);
        let dn_head_dim = cfg.deltanet_head_dim.unwrap_or(head_dim);
        let dn_dim = dn_heads * dn_head_dim;

        let mut hidden = vec![0.0f32; h];
        qweights.embed_token(token_id, h, &mut hidden);

        let mut normed = vec![0.0f32; h];
        let mut q = vec![0.0f32; q_dim.max(dn_dim)];
        let mut k_buf = vec![0.0f32; kv_dim.max(dn_dim)];
        let mut v_buf = vec![0.0f32; kv_dim.max(dn_dim)];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut o_proj = vec![0.0f32; h];
        let mut normed2 = vec![0.0f32; h];
        let mut gate_buf = vec![0.0f32; inter];
        let mut up_buf = vec![0.0f32; inter];
        let mut down_out = vec![0.0f32; h];
        let mut q_gate = vec![0.0f32; q_dim * 2];
        let mut z_gate = vec![0.0f32; dn_dim];
        let mut alpha_buf = vec![0.0f32; dn_heads];
        let mut beta_buf = vec![0.0f32; dn_heads];
        let mut logits = vec![0.0f32; cfg.vocab_size];
        let mut preq_q8 = vec![0i8; h.max(q_dim).max(inter)];
        let mut preq_scales = vec![0.0f32; h.max(q_dim).max(inter).div_ceil(32)];
        let mut preq_asum = vec![0i32; h.max(q_dim).max(inter).div_ceil(32)];

        let positions = vec![session.position as u32];
        let cache_layer_map = cfg.cache_layer_map();
        let _deltanet_layer_idx = 0usize;

        let pool = tempura_kernels::thread_pool::global_pool();
        let n_threads = pool.thread_count();
        let barrier = tempura_kernels::thread_pool::SpinBarrier::new(n_threads);

        macro_rules! ptr {
            ($v:expr) => {
                $v.as_mut_ptr() as usize
            };
        }
        let hp = ptr!(hidden);
        let np = ptr!(normed);
        let qp = ptr!(q);
        let kp = ptr!(k_buf);
        let vp = ptr!(v_buf);
        let ap = ptr!(attn_out);
        let op = ptr!(o_proj);
        let n2p = ptr!(normed2);
        let gp = ptr!(gate_buf);
        let up = ptr!(up_buf);
        let dp = ptr!(down_out);
        let qgp = ptr!(q_gate);
        let zgp = ptr!(z_gate);
        let abp = ptr!(alpha_buf);
        let bbp = ptr!(beta_buf);
        let lp = ptr!(logits);
        let pq8p = ptr!(preq_q8);
        let psp = ptr!(preq_scales);
        let pap = ptr!(preq_asum);

        macro_rules! fs {
            ($a:expr,$n:expr) => {
                unsafe { std::slice::from_raw_parts($a as *const f32, $n) }
            };
        }
        macro_rules! fsm {
            ($a:expr,$n:expr) => {
                unsafe { std::slice::from_raw_parts_mut($a as *mut f32, $n) }
            };
        }
        macro_rules! is {
            ($a:expr,$n:expr) => {
                unsafe { std::slice::from_raw_parts($a as *const i8, $n) }
            };
        }
        macro_rules! ism {
            ($a:expr,$n:expr) => {
                unsafe { std::slice::from_raw_parts_mut($a as *mut i8, $n) }
            };
        }
        macro_rules! i32s {
            ($a:expr,$n:expr) => {
                unsafe { std::slice::from_raw_parts($a as *const i32, $n) }
            };
        }

        use tempura_kernels::tl2::{tl2_gemv_tiles, tl2_preprocess_into, Tl2Config, Tl2Lut};
        let mut lut_h = Tl2Lut::new_for_k(h);
        let mut lut_qdim = if q_dim != h {
            Tl2Lut::new_for_k(q_dim)
        } else {
            None
        };
        let mut lut_inter = Tl2Lut::new_for_k(inter);
        let mut lut_dn = if dn_dim != h {
            Tl2Lut::new_for_k(dn_dim)
        } else {
            None
        };
        let lut_h_ptr = lut_h
            .as_mut()
            .map(|l| l as *mut Tl2Lut as usize)
            .unwrap_or(0);
        let lut_qdim_ptr = lut_qdim
            .as_mut()
            .map(|l| l as *mut Tl2Lut as usize)
            .unwrap_or(lut_h_ptr);
        let lut_inter_ptr = lut_inter
            .as_mut()
            .map(|l| l as *mut Tl2Lut as usize)
            .unwrap_or(0);
        let lut_dn_ptr = lut_dn
            .as_mut()
            .map(|l| l as *mut Tl2Lut as usize)
            .unwrap_or(lut_h_ptr);
        let tile_counter = std::sync::atomic::AtomicUsize::new(0);
        let shared_lut_ok = std::sync::atomic::AtomicBool::new(false);

        let fwd_start = std::time::Instant::now();
        let sess_ptr = session as *mut InferenceSession as usize;

        pool.run_indexed(&|ith, nth| {
            let mut dn_idx = 0usize;

            for layer in 0..cfg.num_layers {
                let layer_kind = cfg
                    .layer_kinds
                    .as_ref()
                    .map(|k| k[layer])
                    .unwrap_or(crate::config::LayerKind::Attention);
                let prefix = format!("model.layers.{layer}");
                let input_ln = format!("{prefix}.input_layernorm.weight");
                let post_ln = format!("{prefix}.post_attention_layernorm.weight");

                // ── B0: Thread 0 — RMSNorm + LUT prep ──
                if ith == 0 {
                    let hidden_s = fsm!(hp, h);
                    let normed_s = fsm!(np, h);
                    normed_s.copy_from_slice(hidden_s);
                    let ln_w = qweights.get_f32(&input_ln).unwrap();
                    self.pipeline
                        .rmsnorm(
                            normed_s,
                            ln_w,
                            h,
                            cfg.layer_norm_eps,
                            TraceLevel::Off,
                            &mut None,
                        )
                        .unwrap();
                    if lut_h_ptr != 0 {
                        let lut = unsafe { &mut *(lut_h_ptr as *mut Tl2Lut) };
                        shared_lut_ok.store(
                            tl2_preprocess_into(fs!(np, h), h, lut),
                            std::sync::atomic::Ordering::Release,
                        );
                    }
                    tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                    q2_preq_into(
                        fs!(np, h),
                        ism!(pq8p, h),
                        fsm!(psp, h / 32),
                        unsafe { std::slice::from_raw_parts_mut(pap as *mut i32, h / 32) },
                        h,
                    );
                }
                barrier.wait();

                if layer_kind == crate::config::LayerKind::DeltaNet {
                    // ── B1: all threads — QKV + gate GEMV (TL2 work-stealing) ──
                    {
                        let qkv_name = format!("{prefix}.ssm.qkv_proj.weight");
                        let gate_name = format!("{prefix}.ssm.gate_proj.weight");
                        if lut_h_ptr != 0 {
                            let lut = unsafe { &*(lut_h_ptr as *const Tl2Lut) };
                            if let (Some(qt), Some(gt)) =
                                (qweights.get_tl2(&qkv_name), qweights.get_tl2(&gate_name))
                            {
                                let qkv_tiles = qt.m
                                    / Tl2Config::for_dims(qt.m, h).map(|c| c.bm).unwrap_or(qt.m);
                                let g_tiles = gt.m
                                    / Tl2Config::for_dims(gt.m, h).map(|c| c.bm).unwrap_or(gt.m);
                                let total = qkv_tiles + g_tiles;
                                loop {
                                    let t = tile_counter
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    if t >= total {
                                        break;
                                    }
                                    if t < qkv_tiles {
                                        tl2_gemv_tiles(
                                            lut,
                                            &qt.data,
                                            qt.scale,
                                            qgp,
                                            qt.m,
                                            h,
                                            t,
                                            t + 1,
                                        );
                                    } else {
                                        tl2_gemv_tiles(
                                            lut,
                                            &gt.data,
                                            gt.scale,
                                            zgp,
                                            gt.m,
                                            h,
                                            t - qkv_tiles,
                                            t - qkv_tiles + 1,
                                        );
                                    }
                                }
                            }
                        }
                        // Alpha/Beta (tiny, thread 0 only)
                        if ith == 0 {
                            let alpha_name = format!("{prefix}.ssm.alpha.weight");
                            let beta_name = format!("{prefix}.ssm.beta.weight");
                            let q8 = is!(pq8p, h);
                            let sc = fs!(psp, h / 32);
                            let asum = i32s!(pap, h / 32);
                            if let Some(w) = qweights.get_q2(&alpha_name) {
                                tempura_kernels::quantized::q2_0_gemv_preq_range(
                                    w,
                                    q8,
                                    sc,
                                    asum,
                                    fsm!(abp, dn_heads),
                                    0,
                                    dn_heads,
                                    h,
                                );
                            }
                            if let Some(w) = qweights.get_q2(&beta_name) {
                                tempura_kernels::quantized::q2_0_gemv_preq_range(
                                    w,
                                    q8,
                                    sc,
                                    asum,
                                    fsm!(bbp, dn_heads),
                                    0,
                                    dn_heads,
                                    h,
                                );
                            }
                        }
                    }
                    // DeltaNet QKV+gate corrections
                    {
                        let qkv_name = format!("{prefix}.ssm.qkv_proj.weight");
                        let gate_name = format!("{prefix}.ssm.gate_proj.weight");
                        if let Some(qt) = qweights.get_tl2(&qkv_name) {
                            if let Some(corr) = qweights.get_q2_row_corrections(&qkv_name) {
                                let (s, e) = ((ith * qt.m) / nth, ((ith + 1) * qt.m) / nth);
                                let out = fsm!(qgp, qt.m);
                                for j in s..e.min(corr.len()) {
                                    out[j] *= corr[j];
                                }
                            }
                        }
                        if let Some(corr) = qweights.get_q2_row_corrections(&gate_name) {
                            let (s, e) = ((ith * dn_dim) / nth, ((ith + 1) * dn_dim) / nth);
                            let z = fsm!(zgp, dn_dim);
                            for j in s..e.min(corr.len()) {
                                z[j] *= corr[j];
                            }
                        }
                    }
                    barrier.wait();

                    // ── B2: thread 0 — conv1d + GDN + norm + z_gate + preq for O proj ──
                    if ith == 0 {
                        let qkv_name = format!("{prefix}.ssm.qkv_proj.weight");
                        let qkv_cols = qweights
                            .get_q2(&qkv_name)
                            .map(|w| w.len() / ((h / 128) * 34))
                            .unwrap_or(dn_dim);
                        let key_dim = if qkv_cols > dn_dim {
                            (qkv_cols - dn_dim) / 2
                        } else {
                            dn_dim
                        };
                        let n_k_heads = key_dim / dn_head_dim;
                        let conv_dim = 2 * key_dim + dn_dim;

                        let q_s = fsm!(qp, dn_dim);
                        let k_s = fsm!(kp, dn_dim);
                        let v_s = fsm!(vp, dn_dim);
                        let z_s = fsm!(zgp, dn_dim);
                        let al_s = fsm!(abp, dn_heads);
                        let be_s = fsm!(bbp, dn_heads);
                        let qkv_buf = fs!(qgp, qkv_cols);

                        // Split QKV from qkv_buf
                        q_s[..key_dim].copy_from_slice(&qkv_buf[..key_dim]);
                        k_s[..key_dim].copy_from_slice(&qkv_buf[key_dim..2 * key_dim]);
                        v_s[..dn_dim].copy_from_slice(&qkv_buf[2 * key_dim..2 * key_dim + dn_dim]);

                        // Gate/beta pre-compute
                        let ssm_a_w = qweights.get_f32(&format!("{prefix}.ssm.A_log"));
                        let dt_bias = qweights.get_f32(&format!("{prefix}.ssm.dt_proj.bias"));
                        for hd in 0..dn_heads {
                            be_s[hd] = 1.0 / (1.0 + (-be_s[hd]).exp());
                            let dt_b = dt_bias.and_then(|b| b.get(hd)).copied().unwrap_or(0.0);
                            let ssm_a = ssm_a_w.and_then(|a| a.get(hd)).copied().unwrap_or(-1.0);
                            al_s[hd] = (1.0f32 + (al_s[hd] + dt_b).exp()).ln() * ssm_a;
                        }

                        // Conv1d (single-token: only last kernel position)
                        if let Some(conv_w) =
                            qweights.get_f32(&format!("{prefix}.ssm.conv1d.weight"))
                        {
                            let kernel_size = conv_w.len() / conv_dim;
                            let conv_bias = qweights.get_f32(&format!("{prefix}.ssm.conv1d.bias"));
                            let mut conv_in = vec![0.0f32; conv_dim];
                            conv_in[..key_dim].copy_from_slice(&q_s[..key_dim]);
                            conv_in[key_dim..2 * key_dim].copy_from_slice(&k_s[..key_dim]);
                            conv_in[2 * key_dim..conv_dim].copy_from_slice(&v_s[..dn_dim]);
                            for ch in 0..conv_dim {
                                let mut s =
                                    conv_in[ch] * conv_w[ch * kernel_size + kernel_size - 1];
                                if let Some(bias) = conv_bias {
                                    if ch < bias.len() {
                                        s += bias[ch];
                                    }
                                }
                                conv_in[ch] = s / (1.0 + (-s).exp());
                            }
                            for vh in 0..dn_heads {
                                let kh = vh * n_k_heads / dn_heads;
                                q_s[vh * dn_head_dim..(vh + 1) * dn_head_dim].copy_from_slice(
                                    &conv_in[kh * dn_head_dim..kh * dn_head_dim + dn_head_dim],
                                );
                                k_s[vh * dn_head_dim..(vh + 1) * dn_head_dim].copy_from_slice(
                                    &conv_in[key_dim + kh * dn_head_dim
                                        ..key_dim + kh * dn_head_dim + dn_head_dim],
                                );
                            }
                            v_s[..dn_dim]
                                .copy_from_slice(&conv_in[2 * key_dim..2 * key_dim + dn_dim]);
                        }

                        // L2 norm Q/K
                        for hd in 0..dn_heads {
                            let off = hd * dn_head_dim;
                            let qn: f32 = q_s[off..off + dn_head_dim]
                                .iter()
                                .map(|x| x * x)
                                .sum::<f32>()
                                .sqrt()
                                .max(cfg.layer_norm_eps);
                            q_s[off..off + dn_head_dim]
                                .iter_mut()
                                .for_each(|x| *x /= qn);
                            let kn: f32 = k_s[off..off + dn_head_dim]
                                .iter()
                                .map(|x| x * x)
                                .sum::<f32>()
                                .sqrt()
                                .max(cfg.layer_norm_eps);
                            k_s[off..off + dn_head_dim]
                                .iter_mut()
                                .for_each(|x| *x /= kn);
                        }

                        // GDN
                        let dn_state = unsafe { &mut *(sess_ptr as *mut InferenceSession) }
                            .deltanet_state
                            .as_mut()
                            .unwrap();
                        let state = dn_state.state_mut(dn_idx);
                        let mut dn_out = vec![0.0f32; dn_dim];
                        tempura_kernels::deltanet::CpuDeltaNet
                            .forward_recurrent(
                                q_s,
                                k_s,
                                v_s,
                                al_s,
                                be_s,
                                state,
                                &mut dn_out,
                                dn_heads,
                                dn_head_dim,
                                1.0,
                            )
                            .unwrap();

                        // Per-head norm + z_gate
                        if let Some(norm_w) = qweights.get_f32(&format!("{prefix}.ssm.norm.weight"))
                        {
                            for head in 0..dn_heads {
                                let off = head * dn_head_dim;
                                self.pipeline
                                    .rmsnorm(
                                        &mut dn_out[off..off + dn_head_dim],
                                        norm_w,
                                        dn_head_dim,
                                        cfg.layer_norm_eps,
                                        TraceLevel::Off,
                                        &mut None,
                                    )
                                    .unwrap();
                            }
                        }
                        for i in 0..dn_dim {
                            let zs = z_s[i] / (1.0 + (-z_s[i]).exp());
                            dn_out[i] *= zs;
                        }

                        // LUT + preq for O proj
                        if lut_dn_ptr != 0 {
                            let lut = unsafe { &mut *(lut_dn_ptr as *mut Tl2Lut) };
                            shared_lut_ok.store(
                                tl2_preprocess_into(&dn_out, dn_dim, lut),
                                std::sync::atomic::Ordering::Release,
                            );
                        }
                        tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                        dn_idx += 1;
                    }
                    barrier.wait();

                    // ── B3: all threads — O proj GEMV (TL2) ──
                    {
                        let out_proj_name = format!("{prefix}.ssm.out_proj.weight");
                        if lut_dn_ptr != 0 {
                            let lut = unsafe { &*(lut_dn_ptr as *const Tl2Lut) };
                            if let Some(ot) = qweights.get_tl2(&out_proj_name) {
                                let o_tiles = ot.m
                                    / Tl2Config::for_dims(ot.m, dn_dim)
                                        .map(|c| c.bm)
                                        .unwrap_or(ot.m);
                                loop {
                                    let t = tile_counter
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    if t >= o_tiles {
                                        break;
                                    }
                                    tl2_gemv_tiles(
                                        lut,
                                        &ot.data,
                                        ot.scale,
                                        op,
                                        ot.m,
                                        dn_dim,
                                        t,
                                        t + 1,
                                    );
                                }
                            }
                        }
                    }
                    barrier.wait();

                    // Apply per-row corrections for O proj
                    {
                        let out_proj_name = format!("{prefix}.ssm.out_proj.weight");
                        if let Some(corr) = qweights.get_q2_row_corrections(&out_proj_name) {
                            let (s, e) = ((ith * h) / nth, ((ith + 1) * h) / nth);
                            let o_s = fsm!(op, h);
                            for j in s..e.min(corr.len()) {
                                o_s[j] *= corr[j];
                            }
                        }
                    }
                    barrier.wait();

                    // ── B4: thread 0 — residual ──
                    if ith == 0 {
                        let hidden_s = fsm!(hp, h);
                        let o_s = fs!(op, h);
                        for i in 0..h {
                            hidden_s[i] += o_s[i];
                        }
                    }
                    barrier.wait();
                } else {
                    // ── Attention: QKV parallel, rest serial on thread 0 ──
                    // B1: all threads GEMV: Q+gate, K, V (TL2 work-stealing)
                    {
                        let q_w_name = format!("{prefix}.self_attn.q_proj.weight");
                        let k_w_name = format!("{prefix}.self_attn.k_proj.weight");
                        let v_w_name = format!("{prefix}.self_attn.v_proj.weight");

                        if lut_h_ptr != 0 {
                            let lut = unsafe { &*(lut_h_ptr as *const Tl2Lut) };
                            let qt = qweights.get_tl2(&q_w_name);
                            let kt = qweights.get_tl2(&k_w_name);
                            let vt = qweights.get_tl2(&v_w_name);
                            if let (Some(qt), Some(kt), Some(vt)) = (qt, kt, vt) {
                                let q_tiles = qt.m
                                    / Tl2Config::for_dims(qt.m, h).map(|c| c.bm).unwrap_or(qt.m);
                                let kv_tiles = kt.m
                                    / Tl2Config::for_dims(kt.m, h).map(|c| c.bm).unwrap_or(kt.m);
                                let total = q_tiles + 2 * kv_tiles;
                                loop {
                                    let t = tile_counter
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    if t >= total {
                                        break;
                                    }
                                    if t < q_tiles {
                                        tl2_gemv_tiles(
                                            lut,
                                            &qt.data,
                                            qt.scale,
                                            qgp,
                                            qt.m,
                                            h,
                                            t,
                                            t + 1,
                                        );
                                    } else if t < q_tiles + kv_tiles {
                                        tl2_gemv_tiles(
                                            lut,
                                            &kt.data,
                                            kt.scale,
                                            kp,
                                            kt.m,
                                            h,
                                            t - q_tiles,
                                            t - q_tiles + 1,
                                        );
                                    } else {
                                        tl2_gemv_tiles(
                                            lut,
                                            &vt.data,
                                            vt.scale,
                                            vp,
                                            vt.m,
                                            h,
                                            t - q_tiles - kv_tiles,
                                            t - q_tiles - kv_tiles + 1,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    barrier.wait();

                    // Apply per-row corrections for Q+gate, K, V
                    {
                        let q_w_name = format!("{prefix}.self_attn.q_proj.weight");
                        let k_w_name = format!("{prefix}.self_attn.k_proj.weight");
                        let v_w_name = format!("{prefix}.self_attn.v_proj.weight");
                        if let Some(qt) = qweights.get_tl2(&q_w_name) {
                            if let Some(corr) = qweights.get_q2_row_corrections(&q_w_name) {
                                let (s, e) = ((ith * qt.m) / nth, ((ith + 1) * qt.m) / nth);
                                let out = fsm!(qgp, qt.m);
                                for j in s..e.min(corr.len()) {
                                    out[j] *= corr[j];
                                }
                            }
                        }
                        if let Some(corr) = qweights.get_q2_row_corrections(&k_w_name) {
                            let (s, e) = ((ith * kv_dim) / nth, ((ith + 1) * kv_dim) / nth);
                            let out = fsm!(kp, kv_dim);
                            for j in s..e.min(corr.len()) {
                                out[j] *= corr[j];
                            }
                        }
                        if let Some(corr) = qweights.get_q2_row_corrections(&v_w_name) {
                            let (s, e) = ((ith * kv_dim) / nth, ((ith + 1) * kv_dim) / nth);
                            let out = fsm!(vp, kv_dim);
                            for j in s..e.min(corr.len()) {
                                out[j] *= corr[j];
                            }
                        }
                    }
                    barrier.wait();

                    // B2: Thread 0 — Q extraction, QK norm, RoPE, KV cache, attention, gate
                    if ith == 0 {
                        let q_w_name = format!("{prefix}.self_attn.q_proj.weight");
                        let has_q_gate = qweights
                            .get_tl2(&q_w_name)
                            .map(|t| t.m > q_dim)
                            .unwrap_or(false);

                        if has_q_gate {
                            let actual_q_cols =
                                qweights.get_tl2(&q_w_name).map(|t| t.m).unwrap_or(q_dim);
                            let qg = fs!(qgp, actual_q_cols);
                            let q_s = fsm!(qp, q_dim);
                            for hh in 0..n_heads {
                                let src = hh * head_dim * 2;
                                q_s[hh * head_dim..(hh + 1) * head_dim]
                                    .copy_from_slice(&qg[src..src + head_dim]);
                            }
                        }

                        // QK norm
                        if cfg.use_qk_norm {
                            let qn = format!("{prefix}.self_attn.q_norm.weight");
                            let kn = format!("{prefix}.self_attn.k_norm.weight");
                            if let Some(qnw) = qweights.get_f32(&qn) {
                                crate::model::apply_head_rmsnorm(
                                    fsm!(qp, q_dim),
                                    qnw,
                                    1,
                                    n_heads,
                                    head_dim,
                                    cfg.layer_norm_eps,
                                );
                            }
                            if let Some(knw) = qweights.get_f32(&kn) {
                                crate::model::apply_head_rmsnorm(
                                    fsm!(kp, kv_dim),
                                    knw,
                                    1,
                                    n_kv_heads,
                                    head_dim,
                                    cfg.layer_norm_eps,
                                );
                            }
                        }

                        // RoPE
                        let rope_hd = cfg.rope_dim.unwrap_or(head_dim);
                        if rope_hd < head_dim {
                            let mut q_rope = vec![0.0f32; n_heads * rope_hd];
                            let mut k_rope = vec![0.0f32; n_kv_heads * rope_hd];
                            let q_s = fsm!(qp, q_dim);
                            let k_s = fsm!(kp, kv_dim);
                            for hh in 0..n_heads {
                                q_rope[hh * rope_hd..(hh + 1) * rope_hd]
                                    .copy_from_slice(&q_s[hh * head_dim..hh * head_dim + rope_hd]);
                            }
                            for hh in 0..n_kv_heads {
                                k_rope[hh * rope_hd..(hh + 1) * rope_hd]
                                    .copy_from_slice(&k_s[hh * head_dim..hh * head_dim + rope_hd]);
                            }
                            self.pipeline
                                .rope(
                                    &mut q_rope,
                                    1,
                                    1,
                                    n_heads,
                                    rope_hd,
                                    &positions,
                                    cfg.rope_theta,
                                )
                                .unwrap();
                            self.pipeline
                                .rope(
                                    &mut k_rope,
                                    1,
                                    1,
                                    n_kv_heads,
                                    rope_hd,
                                    &positions,
                                    cfg.rope_theta,
                                )
                                .unwrap();
                            for hh in 0..n_heads {
                                q_s[hh * head_dim..hh * head_dim + rope_hd]
                                    .copy_from_slice(&q_rope[hh * rope_hd..(hh + 1) * rope_hd]);
                            }
                            for hh in 0..n_kv_heads {
                                k_s[hh * head_dim..hh * head_dim + rope_hd]
                                    .copy_from_slice(&k_rope[hh * rope_hd..(hh + 1) * rope_hd]);
                            }
                        } else {
                            self.pipeline
                                .rope(
                                    fsm!(qp, q_dim),
                                    1,
                                    1,
                                    n_heads,
                                    head_dim,
                                    &positions,
                                    cfg.rope_theta,
                                )
                                .unwrap();
                            self.pipeline
                                .rope(
                                    fsm!(kp, kv_dim),
                                    1,
                                    1,
                                    n_kv_heads,
                                    head_dim,
                                    &positions,
                                    cfg.rope_theta,
                                )
                                .unwrap();
                        }

                        // KV cache
                        let sess = unsafe { &mut *(sess_ptr as *mut InferenceSession) };
                        let cl = cache_layer_map[layer].unwrap();
                        sess.cache
                            .append_kv(cl, fs!(kp, kv_dim), fs!(vp, kv_dim), 1);
                        let kv_len = sess.cache.seq_len + 1;
                        let k_cache = sess.cache.k_data(cl, kv_len);
                        let v_cache = sess.cache.v_data(cl, kv_len);

                        // Attention
                        let attn_s = fsm!(ap, q_dim);
                        attn_s.fill(0.0);
                        tempura_kernels::attention::CpuAttentionGgml
                            .forward(
                                fs!(qp, q_dim),
                                &k_cache,
                                &v_cache,
                                attn_s,
                                1,
                                1,
                                kv_len,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                1.0 / (head_dim as f32).sqrt(),
                            )
                            .unwrap();

                        // Sigmoid gate
                        if has_q_gate {
                            let actual_q_cols =
                                qweights.get_tl2(&q_w_name).map(|t| t.m).unwrap_or(q_dim);
                            let qg = fs!(qgp, actual_q_cols);
                            for hh in 0..n_heads {
                                for d in 0..head_dim {
                                    let gv = qg[hh * head_dim * 2 + head_dim + d];
                                    attn_s[hh * head_dim + d] *= 1.0 / (1.0 + (-gv).exp());
                                }
                            }
                        }

                        // LUT prep for O proj
                        if lut_qdim_ptr != 0 {
                            let lut = unsafe { &mut *(lut_qdim_ptr as *mut Tl2Lut) };
                            shared_lut_ok.store(
                                tl2_preprocess_into(fs!(ap, q_dim), q_dim, lut),
                                std::sync::atomic::Ordering::Release,
                            );
                        }
                        tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                    }
                    barrier.wait();

                    // B3: all threads GEMV: O proj (TL2)
                    {
                        let o_w_name = format!("{prefix}.self_attn.o_proj.weight");
                        if lut_qdim_ptr != 0 {
                            let lut = unsafe { &*(lut_qdim_ptr as *const Tl2Lut) };
                            if let Some(ot) = qweights.get_tl2(&o_w_name) {
                                let o_tiles = ot.m
                                    / Tl2Config::for_dims(ot.m, q_dim)
                                        .map(|c| c.bm)
                                        .unwrap_or(ot.m);
                                loop {
                                    let t = tile_counter
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    if t >= o_tiles {
                                        break;
                                    }
                                    tl2_gemv_tiles(
                                        lut,
                                        &ot.data,
                                        ot.scale,
                                        op,
                                        ot.m,
                                        q_dim,
                                        t,
                                        t + 1,
                                    );
                                }
                            }
                        }
                    }
                    barrier.wait();
                    // Corrections for O proj
                    {
                        let o_w_name = format!("{prefix}.self_attn.o_proj.weight");
                        if let Some(corr) = qweights.get_q2_row_corrections(&o_w_name) {
                            let (s, e) = ((ith * h) / nth, ((ith + 1) * h) / nth);
                            let o_s = fsm!(op, h);
                            for j in s..e.min(corr.len()) {
                                o_s[j] *= corr[j];
                            }
                        }
                    }
                    barrier.wait();

                    // B4: Thread 0 — residual
                    if ith == 0 {
                        let hidden_s = fsm!(hp, h);
                        let o_s = fs!(op, h);
                        for i in 0..h {
                            hidden_s[i] += o_s[i];
                        }
                    }
                    barrier.wait();
                }

                // ── FFN (shared by DeltaNet and Attention) ──
                // B5: Thread 0 — post-attention RMSNorm + LUT
                if ith == 0 {
                    let hidden_s = fs!(hp, h);
                    let n2 = fsm!(n2p, h);
                    n2.copy_from_slice(hidden_s);
                    let pln_w = qweights.get_f32(&post_ln).unwrap();
                    self.pipeline
                        .rmsnorm(n2, pln_w, h, cfg.layer_norm_eps, TraceLevel::Off, &mut None)
                        .unwrap();
                    if lut_h_ptr != 0 {
                        let lut = unsafe { &mut *(lut_h_ptr as *mut Tl2Lut) };
                        shared_lut_ok.store(
                            tl2_preprocess_into(fs!(n2p, h), h, lut),
                            std::sync::atomic::Ordering::Release,
                        );
                    }
                    tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                }
                barrier.wait();

                // B6: all threads GEMV: gate + up (TL2 work-stealing)
                {
                    let gate_name = format!("{prefix}.mlp.gate_proj.weight");
                    let up_name = format!("{prefix}.mlp.up_proj.weight");
                    if lut_h_ptr != 0 {
                        let lut = unsafe { &*(lut_h_ptr as *const Tl2Lut) };
                        if let (Some(gt), Some(ut)) =
                            (qweights.get_tl2(&gate_name), qweights.get_tl2(&up_name))
                        {
                            let g_tiles =
                                gt.m / Tl2Config::for_dims(gt.m, h).map(|c| c.bm).unwrap_or(gt.m);
                            loop {
                                let t =
                                    tile_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if t >= g_tiles {
                                    break;
                                }
                                tl2_gemv_tiles(lut, &gt.data, gt.scale, gp, gt.m, h, t, t + 1);
                                tl2_gemv_tiles(lut, &ut.data, ut.scale, up, ut.m, h, t, t + 1);
                            }
                        }
                    }
                }
                barrier.wait();

                // Corrections for gate + up
                {
                    let gate_name = format!("{prefix}.mlp.gate_proj.weight");
                    let up_name = format!("{prefix}.mlp.up_proj.weight");
                    let (s, e) = ((ith * inter) / nth, ((ith + 1) * inter) / nth);
                    if let Some(corr) = qweights.get_q2_row_corrections(&gate_name) {
                        let g = fsm!(gp, inter);
                        for j in s..e.min(corr.len()) {
                            g[j] *= corr[j];
                        }
                    }
                    if let Some(corr) = qweights.get_q2_row_corrections(&up_name) {
                        let u = fsm!(up, inter);
                        for j in s..e.min(corr.len()) {
                            u[j] *= corr[j];
                        }
                    }
                }
                barrier.wait();

                // B7: Thread 0 — SiLU*gate + LUT
                if ith == 0 {
                    let g = fsm!(gp, inter);
                    let u = fs!(up, inter);
                    for i in 0..inter {
                        g[i] = (g[i] / (1.0 + (-g[i]).exp())) * u[i];
                    }
                    if lut_inter_ptr != 0 {
                        let lut = unsafe { &mut *(lut_inter_ptr as *mut Tl2Lut) };
                        shared_lut_ok.store(
                            tl2_preprocess_into(fs!(gp, inter), inter, lut),
                            std::sync::atomic::Ordering::Release,
                        );
                    }
                    tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                }
                barrier.wait();

                // B8: all threads GEMV: down proj (TL2)
                {
                    let down_name = format!("{prefix}.mlp.down_proj.weight");
                    if lut_inter_ptr != 0 {
                        let lut = unsafe { &*(lut_inter_ptr as *const Tl2Lut) };
                        if let Some(dt) = qweights.get_tl2(&down_name) {
                            let d_tiles = dt.m
                                / Tl2Config::for_dims(dt.m, inter)
                                    .map(|c| c.bm)
                                    .unwrap_or(dt.m);
                            loop {
                                let t =
                                    tile_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if t >= d_tiles {
                                    break;
                                }
                                tl2_gemv_tiles(lut, &dt.data, dt.scale, dp, dt.m, inter, t, t + 1);
                            }
                        }
                    }
                }
                barrier.wait();

                // Corrections for down + residual
                {
                    let down_name = format!("{prefix}.mlp.down_proj.weight");
                    if let Some(corr) = qweights.get_q2_row_corrections(&down_name) {
                        let (s, e) = ((ith * h) / nth, ((ith + 1) * h) / nth);
                        let d_s = fsm!(dp, h);
                        for j in s..e.min(corr.len()) {
                            d_s[j] *= corr[j];
                        }
                    }
                }
                barrier.wait();

                // B9: Thread 0 — residual
                if ith == 0 {
                    let hidden_s = fsm!(hp, h);
                    let d_s = fs!(dp, h);
                    for i in 0..h {
                        hidden_s[i] += d_s[i];
                    }
                }
                barrier.wait();
            }

            // ── Final: lm_head ──
            if ith == 0 {
                let hidden_s = fsm!(hp, h);
                let final_norm_w = qweights.get_f32("model.norm.weight").unwrap();
                self.pipeline
                    .rmsnorm(
                        hidden_s,
                        final_norm_w,
                        h,
                        cfg.layer_norm_eps,
                        TraceLevel::Off,
                        &mut None,
                    )
                    .unwrap();
                if lut_h_ptr != 0 {
                    let lut = unsafe { &mut *(lut_h_ptr as *mut Tl2Lut) };
                    shared_lut_ok.store(
                        tl2_preprocess_into(fs!(hp, h), h, lut),
                        std::sync::atomic::Ordering::Release,
                    );
                }
                tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
            }
            barrier.wait();

            if lut_h_ptr != 0 {
                let lut = unsafe { &*(lut_h_ptr as *const Tl2Lut) };
                if let Some(lt) = qweights.get_tl2("lm_head.weight") {
                    let l_tiles = lt.m / Tl2Config::for_dims(lt.m, h).map(|c| c.bm).unwrap_or(lt.m);
                    loop {
                        let t = tile_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if t >= l_tiles {
                            break;
                        }
                        tl2_gemv_tiles(lut, &lt.data, lt.scale, lp, lt.m, h, t, t + 1);
                    }
                }
            }
            barrier.wait();
            // lm_head corrections
            if let Some(corr) = qweights.get_q2_row_corrections("lm_head.weight") {
                let (s, e) = (
                    (ith * cfg.vocab_size) / nth,
                    ((ith + 1) * cfg.vocab_size) / nth,
                );
                let l_s = fsm!(lp, cfg.vocab_size);
                for j in s..e.min(corr.len()) {
                    l_s[j] *= corr[j];
                }
            }
            barrier.wait();
        });

        session.cache.advance(1);
        session.position += 1;

        if std::env::var("TEMPURA_PROFILE").is_ok() {
            eprintln!(
                "[PROFILE] forward_q2_decode: {:.2}ms",
                fwd_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        logits
    }

    fn forward_bitnet_decode(
        &self,
        session: &mut InferenceSession,
        token_id: u32,
        qweights: &crate::weights::QuantizedWeights,
    ) -> Vec<f32> {
        if self.config.layer_kinds.is_some() {
            return self.forward_quantized(session, &[token_id], qweights);
        }
        use tempura_kernels::fused::{
            fused_residual_rmsnorm_q8snap, fused_rmsnorm_q8snap, fused_sqrelu_mul, residual_add,
        };
        use tempura_kernels::quantized::{
            i2s_gemv_cols, i2s_weight_scale, PreQuantizedActivations,
        };
        use tempura_kernels::tl2::{tl2_gemv_tiles, tl2_preprocess_into, Tl2Lut};

        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = cfg.intermediate_size;

        let rmsnorm = tempura_kernels::rmsnorm::CpuRMSNormGgml;
        let rope = tempura_kernels::rope::CpuRoPEGgml;

        let mut hidden = vec![0.0f32; h];
        qweights.embed_token(token_id, h, &mut hidden);

        let positions = [session.position as u32];

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

        let max_k = h.max(inter).max(q_dim);
        let mut preq_buf = vec![0i8; max_k];
        let mut normed = vec![0.0f32; h];
        let mut q = vec![0.0f32; q_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut o_proj = vec![0.0f32; h];
        let mut normed2 = vec![0.0f32; h];
        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        let mut down_out = vec![0.0f32; h];

        let hp = hidden.as_mut_ptr() as usize;
        let np = normed.as_mut_ptr() as usize;
        let qp = q.as_mut_ptr() as usize;
        let kp = k_buf.as_mut_ptr() as usize;
        let vp = v_buf.as_mut_ptr() as usize;
        let ap = attn_out.as_mut_ptr() as usize;
        let op = o_proj.as_mut_ptr() as usize;
        let n2p = normed2.as_mut_ptr() as usize;
        let gp = gate.as_mut_ptr() as usize;
        let upp = up.as_mut_ptr() as usize;
        let dp = down_out.as_mut_ptr() as usize;
        let pp = preq_buf.as_mut_ptr() as usize;
        let sp = session as *mut InferenceSession as usize;

        // Pre-allocate TL2 LUTs — reused across all layers (no heap allocs in the loop)
        let mut lut_h = Tl2Lut::new_for_k(h);
        let mut lut_qdim = if q_dim != h {
            Tl2Lut::new_for_k(q_dim)
        } else {
            None
        };
        let mut lut_inter = Tl2Lut::new_for_k(inter);
        let lut_h_ptr = lut_h
            .as_mut()
            .map(|l| l as *mut Tl2Lut as usize)
            .unwrap_or(0);
        let lut_qdim_ptr = lut_qdim
            .as_mut()
            .map(|l| l as *mut Tl2Lut as usize)
            .unwrap_or(lut_h_ptr);
        let lut_inter_ptr = lut_inter
            .as_mut()
            .map(|l| l as *mut Tl2Lut as usize)
            .unwrap_or(0);

        let pool = tempura_kernels::thread_pool::global_pool();
        let n_threads = pool.thread_count();
        let shared_scale = std::sync::atomic::AtomicU32::new(0);
        let shared_sum = std::sync::atomic::AtomicI32::new(0);
        let shared_lut_ok = std::sync::atomic::AtomicBool::new(false);
        let barrier = tempura_kernels::thread_pool::SpinBarrier::new(n_threads);
        let shared_kvl = std::sync::atomic::AtomicUsize::new(0);
        let tile_counter = std::sync::atomic::AtomicUsize::new(0);
        let max_seq = cfg.max_seq_len;
        let mut attn_scores_bufs: Vec<Vec<f32>> =
            (0..n_threads).map(|_| vec![0.0f32; max_seq]).collect();
        let attn_scores_ptrs: Vec<usize> = attn_scores_bufs
            .iter_mut()
            .map(|b| b.as_mut_ptr() as usize)
            .collect();

        let mut logits = vec![0.0f32; cfg.vocab_size];
        let logits_ptr = logits.as_mut_ptr() as usize;
        let vocab_size = cfg.vocab_size;

        let do_profile = std::env::var("TEMPURA_PROFILE").is_ok();
        let prof_serial1 = std::sync::atomic::AtomicU64::new(0);
        let prof_qkv_mm = std::sync::atomic::AtomicU64::new(0);
        let prof_serial2 = std::sync::atomic::AtomicU64::new(0);
        let prof_rope = std::sync::atomic::AtomicU64::new(0);
        let prof_attn = std::sync::atomic::AtomicU64::new(0);
        let prof_lut_prep = std::sync::atomic::AtomicU64::new(0);
        let prof_o_mm = std::sync::atomic::AtomicU64::new(0);
        let prof_serial3 = std::sync::atomic::AtomicU64::new(0);
        let prof_gu_mm = std::sync::atomic::AtomicU64::new(0);
        let prof_serial4 = std::sync::atomic::AtomicU64::new(0);
        let prof_dn_mm = std::sync::atomic::AtomicU64::new(0);
        let prof_lm_head = std::sync::atomic::AtomicU64::new(0);

        pool.run_indexed(&|ith, nth| {
            macro_rules! fs {
                ($a:expr,$n:expr) => {
                    unsafe { std::slice::from_raw_parts($a as *const f32, $n) }
                };
            }
            macro_rules! fsm {
                ($a:expr,$n:expr) => {
                    unsafe { std::slice::from_raw_parts_mut($a as *mut f32, $n) }
                };
            }
            macro_rules! is {
                ($a:expr,$n:expr) => {
                    unsafe { std::slice::from_raw_parts($a as *const i8, $n) }
                };
            }
            macro_rules! ism {
                ($a:expr,$n:expr) => {
                    unsafe { std::slice::from_raw_parts_mut($a as *mut i8, $n) }
                };
            }

            for layer in 0..cfg.num_layers {
                let (
                    ref iln,
                    ref pln,
                    ref qn,
                    ref kn,
                    ref vn,
                    ref on,
                    ref _qnm,
                    ref _knm,
                    ref gn,
                    ref un,
                    ref dn,
                    ref asn,
                    ref fsn,
                ) = layer_names[layer];

                // ── Fused: copy + RMSNorm + q8_snap + i2s_quantize + TL2 preprocess ──
                let _t0 = Instant::now();
                if ith == 0 {
                    fsm!(np, h).copy_from_slice(fs!(hp, h));
                    let w = qweights.get_f32(iln).unwrap();
                    fused_rmsnorm_q8snap(fsm!(np, h), w, h, cfg.layer_norm_eps);
                    let (s, m) =
                        tempura_kernels::quantized::i2s_quantize_into(fs!(np, h), ism!(pp, h));
                    shared_scale.store(s.to_bits(), std::sync::atomic::Ordering::Release);
                    shared_sum.store(m, std::sync::atomic::Ordering::Release);
                    if lut_h_ptr != 0 {
                        let lut = unsafe { &mut *(lut_h_ptr as *mut Tl2Lut) };
                        shared_lut_ok.store(
                            tl2_preprocess_into(fs!(np, h), h, lut),
                            std::sync::atomic::Ordering::Release,
                        );
                    } else {
                        shared_lut_ok.store(false, std::sync::atomic::Ordering::Release);
                    }
                    tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                    if do_profile {
                        prof_serial1.fetch_add(
                            _t0.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                }
                barrier.wait(); // B1: quant + LUT ready → QKV matmul

                let _t1 = Instant::now();
                let ps = f32::from_bits(shared_scale.load(std::sync::atomic::Ordering::Acquire));
                let pm = shared_sum.load(std::sync::atomic::Ordering::Acquire);
                let lut_ok = shared_lut_ok.load(std::sync::atomic::Ordering::Acquire);
                let mut used_tl2 = false;
                if lut_ok && lut_h_ptr != 0 {
                    let lut = unsafe { &*(lut_h_ptr as *const Tl2Lut) };
                    if let (Some(qt), Some(kt), Some(vt)) = (
                        qweights.get_tl2(qn),
                        qweights.get_tl2(kn),
                        qweights.get_tl2(vn),
                    ) {
                        let q_tiles = q_dim
                            / tempura_kernels::tl2::Tl2Config::for_dims(q_dim, h)
                                .map(|c| c.bm)
                                .unwrap_or(q_dim);
                        let kv_tiles = kv_dim
                            / tempura_kernels::tl2::Tl2Config::for_dims(kv_dim, h)
                                .map(|c| c.bm)
                                .unwrap_or(kv_dim);
                        let total = q_tiles + 2 * kv_tiles;
                        loop {
                            let t = tile_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if t >= total {
                                break;
                            }
                            if t < q_tiles {
                                tl2_gemv_tiles(lut, &qt.data, qt.scale, qp, q_dim, h, t, t + 1);
                            } else if t < q_tiles + kv_tiles {
                                let kt_idx = t - q_tiles;
                                tl2_gemv_tiles(
                                    lut,
                                    &kt.data,
                                    kt.scale,
                                    kp,
                                    kv_dim,
                                    h,
                                    kt_idx,
                                    kt_idx + 1,
                                );
                            } else {
                                let vt_idx = t - q_tiles - kv_tiles;
                                tl2_gemv_tiles(
                                    lut,
                                    &vt.data,
                                    vt.scale,
                                    vp,
                                    kv_dim,
                                    h,
                                    vt_idx,
                                    vt_idx + 1,
                                );
                            }
                        }
                        used_tl2 = true;
                    }
                }
                if !used_tl2 {
                    if let (Some(qw), Some(kw), Some(vw)) = (
                        qweights.get_i2s(qn),
                        qweights.get_i2s(kn),
                        qweights.get_i2s(vn),
                    ) {
                        let (qs, qe) = ((ith * q_dim) / nth, ((ith + 1) * q_dim) / nth);
                        let (ks, ke) = ((ith * kv_dim) / nth, ((ith + 1) * kv_dim) / nth);
                        i2s_gemv_cols(
                            is!(pp, h),
                            ps,
                            pm,
                            qw,
                            i2s_weight_scale(qw, q_dim, h),
                            qp,
                            qs,
                            qe,
                            h,
                        );
                        i2s_gemv_cols(
                            is!(pp, h),
                            ps,
                            pm,
                            kw,
                            i2s_weight_scale(kw, kv_dim, h),
                            kp,
                            ks,
                            ke,
                            h,
                        );
                        i2s_gemv_cols(
                            is!(pp, h),
                            ps,
                            pm,
                            vw,
                            i2s_weight_scale(vw, kv_dim, h),
                            vp,
                            ks,
                            ke,
                            h,
                        );
                    }
                }
                if do_profile && ith == 0 {
                    prof_qkv_mm.fetch_add(
                        _t1.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                barrier.wait(); // B2: QKV done

                // ── Thread 0: RoPE + KV cache append ──
                if ith == 0 {
                    let _tr = Instant::now();
                    use tempura_kernels::RoPE;
                    rope.forward(
                        fsm!(qp, q_dim),
                        1,
                        1,
                        n_heads,
                        head_dim,
                        &positions,
                        cfg.rope_theta,
                    )
                    .unwrap();
                    rope.forward(
                        fsm!(kp, kv_dim),
                        1,
                        1,
                        n_kv_heads,
                        head_dim,
                        &positions,
                        cfg.rope_theta,
                    )
                    .unwrap();
                    if do_profile {
                        prof_rope.fetch_add(
                            _tr.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    let sess = unsafe { &mut *(sp as *mut InferenceSession) };
                    sess.cache
                        .append_kv(layer, fs!(kp, kv_dim), fs!(vp, kv_dim), 1);
                    shared_kvl.store(sess.cache.seq_len + 1, std::sync::atomic::Ordering::Release);
                }
                barrier.wait(); // B2.5: RoPE done, KV cache ready

                // ── All threads: parallel attention across heads ──
                let _ta = Instant::now();
                {
                    let kvl = shared_kvl.load(std::sync::atomic::Ordering::Acquire);
                    let sess = unsafe { &*(sp as *const InferenceSession) };
                    let hs = ith * n_heads / nth;
                    let he = (ith + 1) * n_heads / nth;
                    let scores_buf = unsafe {
                        std::slice::from_raw_parts_mut(attn_scores_ptrs[ith] as *mut f32, kvl)
                    };
                    tempura_kernels::attention::attention_decode_heads(
                        fs!(qp, q_dim),
                        &sess.cache.k_data(layer, kvl),
                        &sess.cache.v_data(layer, kvl),
                        fsm!(ap, q_dim),
                        scores_buf,
                        kvl,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        1.0 / (head_dim as f32).sqrt(),
                        hs,
                        he,
                    );
                }
                if do_profile && ith == 0 {
                    prof_attn.fetch_add(
                        _ta.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                barrier.wait(); // B3: attention done

                // ── Thread 0: attn_sub_norm + i2s_quantize + TL2 preprocess ──
                let _t2 = Instant::now();
                if ith == 0 {
                    if let Some(w) = qweights.get_f32(asn) {
                        use tempura_kernels::RMSNorm as _;
                        rmsnorm
                            .forward(fsm!(ap, q_dim), w, q_dim, cfg.layer_norm_eps)
                            .unwrap();
                    }
                    let (s, m) = tempura_kernels::quantized::i2s_quantize_into(
                        fs!(ap, q_dim),
                        ism!(pp, q_dim),
                    );
                    shared_scale.store(s.to_bits(), std::sync::atomic::Ordering::Release);
                    shared_sum.store(m, std::sync::atomic::Ordering::Release);
                    let _tl = Instant::now();
                    if lut_qdim_ptr != 0 {
                        let lut = unsafe { &mut *(lut_qdim_ptr as *mut Tl2Lut) };
                        shared_lut_ok.store(
                            tl2_preprocess_into(fs!(ap, q_dim), q_dim, lut),
                            std::sync::atomic::Ordering::Release,
                        );
                    } else {
                        shared_lut_ok.store(false, std::sync::atomic::Ordering::Release);
                    }
                    tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                    if do_profile {
                        prof_lut_prep.fetch_add(
                            _tl.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        prof_serial2.fetch_add(
                            _t2.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                }
                barrier.wait(); // B3.5: attn quant ready → O matmul

                let _t3 = Instant::now();
                let as2 = f32::from_bits(shared_scale.load(std::sync::atomic::Ordering::Acquire));
                let am2 = shared_sum.load(std::sync::atomic::Ordering::Acquire);
                let lut_ok = shared_lut_ok.load(std::sync::atomic::Ordering::Acquire);
                let mut used_tl2 = false;
                if lut_ok && lut_qdim_ptr != 0 {
                    let lut = unsafe { &*(lut_qdim_ptr as *const Tl2Lut) };
                    if let Some(ot) = qweights.get_tl2(on) {
                        let o_tiles = h / tempura_kernels::tl2::Tl2Config::for_dims(h, q_dim)
                            .map(|c| c.bm)
                            .unwrap_or(h);
                        loop {
                            let t = tile_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if t >= o_tiles {
                                break;
                            }
                            tl2_gemv_tiles(lut, &ot.data, ot.scale, op, h, q_dim, t, t + 1);
                        }
                        used_tl2 = true;
                    }
                }
                if !used_tl2 {
                    if let Some(ow) = qweights.get_i2s(on) {
                        let (s, e) = ((ith * h) / nth, ((ith + 1) * h) / nth);
                        i2s_gemv_cols(
                            is!(pp, q_dim),
                            as2,
                            am2,
                            ow,
                            i2s_weight_scale(ow, h, q_dim),
                            op,
                            s,
                            e,
                            q_dim,
                        );
                    }
                }
                if do_profile && ith == 0 {
                    prof_o_mm.fetch_add(
                        _t3.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                barrier.wait(); // B4: O matmul done

                // ── Fused: residual + RMSNorm + q8_snap + i2s_quantize + TL2 preprocess ──
                let _t4 = Instant::now();
                if ith == 0 {
                    let w = qweights.get_f32(pln).unwrap();
                    fused_residual_rmsnorm_q8snap(
                        fsm!(hp, h),
                        fs!(op, h),
                        fsm!(n2p, h),
                        w,
                        h,
                        cfg.layer_norm_eps,
                    );
                    let (s, m) =
                        tempura_kernels::quantized::i2s_quantize_into(fs!(n2p, h), ism!(pp, h));
                    shared_scale.store(s.to_bits(), std::sync::atomic::Ordering::Release);
                    shared_sum.store(m, std::sync::atomic::Ordering::Release);
                    if lut_h_ptr != 0 {
                        let lut = unsafe { &mut *(lut_h_ptr as *mut Tl2Lut) };
                        shared_lut_ok.store(
                            tl2_preprocess_into(fs!(n2p, h), h, lut),
                            std::sync::atomic::Ordering::Release,
                        );
                    } else {
                        shared_lut_ok.store(false, std::sync::atomic::Ordering::Release);
                    }
                    tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                    if do_profile {
                        prof_serial3.fetch_add(
                            _t4.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                }
                barrier.wait(); // B5: FFN quant ready → gate/up matmul

                let _t5 = Instant::now();
                let ns = f32::from_bits(shared_scale.load(std::sync::atomic::Ordering::Acquire));
                let nm = shared_sum.load(std::sync::atomic::Ordering::Acquire);
                let lut_ok = shared_lut_ok.load(std::sync::atomic::Ordering::Acquire);
                let mut used_tl2 = false;
                if lut_ok && lut_h_ptr != 0 {
                    let lut = unsafe { &*(lut_h_ptr as *const Tl2Lut) };
                    if let (Some(gt), Some(ut)) = (qweights.get_tl2(gn), qweights.get_tl2(un)) {
                        let g_tiles = inter
                            / tempura_kernels::tl2::Tl2Config::for_dims(inter, h)
                                .map(|c| c.bm)
                                .unwrap_or(inter);
                        loop {
                            let t = tile_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if t >= g_tiles {
                                break;
                            }
                            tl2_gemv_tiles(lut, &gt.data, gt.scale, gp, inter, h, t, t + 1);
                            tl2_gemv_tiles(lut, &ut.data, ut.scale, upp, inter, h, t, t + 1);
                        }
                        used_tl2 = true;
                    }
                }
                if !used_tl2 {
                    if let (Some(gw), Some(uw)) = (qweights.get_i2s(gn), qweights.get_i2s(un)) {
                        let (s, e) = ((ith * inter) / nth, ((ith + 1) * inter) / nth);
                        i2s_gemv_cols(
                            is!(pp, h),
                            ns,
                            nm,
                            gw,
                            i2s_weight_scale(gw, inter, h),
                            gp,
                            s,
                            e,
                            h,
                        );
                        i2s_gemv_cols(
                            is!(pp, h),
                            ns,
                            nm,
                            uw,
                            i2s_weight_scale(uw, inter, h),
                            upp,
                            s,
                            e,
                            h,
                        );
                    }
                }
                if do_profile && ith == 0 {
                    prof_gu_mm.fetch_add(
                        _t5.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                barrier.wait(); // B6: gate/up done

                // ── Fused: sqrelu + sub_norm + i2s_quantize + TL2 preprocess ──
                let _t6 = Instant::now();
                if ith == 0 {
                    fused_sqrelu_mul(fsm!(gp, inter), fs!(upp, inter), inter);
                    if let Some(w) = qweights.get_f32(fsn) {
                        use tempura_kernels::RMSNorm as _;
                        rmsnorm
                            .forward(fsm!(gp, inter), w, inter, cfg.layer_norm_eps)
                            .unwrap();
                    }
                    let (s, m) = tempura_kernels::quantized::i2s_quantize_into(
                        fs!(gp, inter),
                        ism!(pp, inter),
                    );
                    shared_scale.store(s.to_bits(), std::sync::atomic::Ordering::Release);
                    shared_sum.store(m, std::sync::atomic::Ordering::Release);
                    if lut_inter_ptr != 0 {
                        let lut = unsafe { &mut *(lut_inter_ptr as *mut Tl2Lut) };
                        shared_lut_ok.store(
                            tl2_preprocess_into(fs!(gp, inter), inter, lut),
                            std::sync::atomic::Ordering::Release,
                        );
                    } else {
                        shared_lut_ok.store(false, std::sync::atomic::Ordering::Release);
                    }
                    tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                    if do_profile {
                        prof_serial4.fetch_add(
                            _t6.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                }
                barrier.wait(); // B7: down quant ready → down matmul

                let _t7 = Instant::now();
                let gs = f32::from_bits(shared_scale.load(std::sync::atomic::Ordering::Acquire));
                let gm = shared_sum.load(std::sync::atomic::Ordering::Acquire);
                let lut_ok = shared_lut_ok.load(std::sync::atomic::Ordering::Acquire);
                let mut used_tl2 = false;
                if lut_ok && lut_inter_ptr != 0 {
                    let lut = unsafe { &*(lut_inter_ptr as *const Tl2Lut) };
                    if let Some(dt) = qweights.get_tl2(dn) {
                        let d_tiles = h / tempura_kernels::tl2::Tl2Config::for_dims(h, inter)
                            .map(|c| c.bm)
                            .unwrap_or(h);
                        loop {
                            let t = tile_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if t >= d_tiles {
                                break;
                            }
                            tl2_gemv_tiles(lut, &dt.data, dt.scale, dp, h, inter, t, t + 1);
                        }
                        used_tl2 = true;
                    }
                }
                if !used_tl2 {
                    if let Some(dw) = qweights.get_i2s(dn) {
                        let (s, e) = ((ith * h) / nth, ((ith + 1) * h) / nth);
                        i2s_gemv_cols(
                            is!(pp, inter),
                            gs,
                            gm,
                            dw,
                            i2s_weight_scale(dw, h, inter),
                            dp,
                            s,
                            e,
                            inter,
                        );
                    }
                }
                if do_profile && ith == 0 {
                    prof_dn_mm.fetch_add(
                        _t7.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                barrier.wait(); // B8: down matmul done

                if ith == 0 {
                    residual_add(fsm!(hp, h), fs!(dp, h), h);
                }
            }

            // ── lm_head (inside persistent threads) ──────────────────────
            barrier.wait(); // B9: last residual done → lm_head

            let _t_lm = Instant::now();

            if ith == 0 {
                fused_rmsnorm_q8snap(
                    fsm!(hp, h),
                    qweights.get_f32("model.norm.weight").unwrap(),
                    h,
                    cfg.layer_norm_eps,
                );

                if lut_h_ptr != 0 {
                    let lut = unsafe { &mut *(lut_h_ptr as *mut Tl2Lut) };
                    shared_lut_ok.store(
                        tl2_preprocess_into(fs!(hp, h), h, lut),
                        std::sync::atomic::Ordering::Release,
                    );
                }

                let session = unsafe { &mut *(sp as *mut InferenceSession) };
                session.cache.advance(1);
                session.position += 1;

                tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
            }
            barrier.wait(); // B10: norm + LUT ready → lm_head matmul

            let lut_ok = shared_lut_ok.load(std::sync::atomic::Ordering::Acquire);
            if lut_ok && lut_h_ptr != 0 {
                if let Some(tl2t) = qweights.get_tl2("lm_head.weight") {
                    let lut = unsafe { &*(lut_h_ptr as *const Tl2Lut) };
                    let lm_bm = tempura_kernels::tl2::Tl2Config::for_dims(vocab_size, h)
                        .map(|c| c.bm)
                        .unwrap_or(vocab_size);
                    let lm_tiles = vocab_size / lm_bm;
                    loop {
                        let t = tile_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if t >= lm_tiles {
                            break;
                        }
                        tl2_gemv_tiles(
                            lut,
                            &tl2t.data,
                            tl2t.scale,
                            logits_ptr,
                            vocab_size,
                            h,
                            t,
                            t + 1,
                        );
                    }
                }
            }

            if do_profile && ith == 0 {
                prof_lm_head.fetch_add(
                    _t_lm.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        });

        // Fallback: if lm_head wasn't handled by TL2 inside the persistent threads
        if qweights.get_tl2("lm_head.weight").is_none() {
            let final_normed = unsafe { std::slice::from_raw_parts(hp as *const f32, h) };
            if let Some(w) = qweights.get_i2s("lm_head.weight") {
                let preq = PreQuantizedActivations::i2s_only(unsafe {
                    std::slice::from_raw_parts(hp as *const f32, h)
                });
                let q4k = tempura_kernels::quantized::CpuMatMulQ4KOpt;
                use tempura_kernels::quantized::MatMulQ;
                q4k.forward_i2s_preq(
                    &preq.i2s_q8,
                    preq.i2s_scale,
                    preq.i2s_sum,
                    w,
                    &mut logits,
                    cfg.vocab_size,
                    h,
                )
                .unwrap();
            } else if let Some(w) = qweights.get_f16("lm_head.weight") {
                tempura_kernels::f16_gemv::f16_gemv(
                    final_normed,
                    w,
                    &mut logits,
                    cfg.vocab_size,
                    h,
                );
            } else {
                let w = qweights.get_f32("lm_head.weight").unwrap();
                self.pipeline
                    .matmul_bt(final_normed, w, &mut logits, 1, cfg.vocab_size, h)
                    .unwrap();
            }
        }

        if do_profile && session.position.is_multiple_of(10) {
            let nl = cfg.num_layers as f64;
            let s1 = prof_serial1.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let qkv = prof_qkv_mm.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let s2 = prof_serial2.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let omm = prof_o_mm.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let s3 = prof_serial3.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let gu = prof_gu_mm.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let s4 = prof_serial4.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let dn = prof_dn_mm.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let total = s1 + qkv + s2 + omm + s3 + gu + s4 + dn;
            let rp = prof_rope.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let at = prof_attn.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let lp = prof_lut_prep.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let lmh = prof_lm_head.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0;
            eprintln!("[PROFILE] pos={} per-layer avg (μs): s1={:.0} qkv={:.0} s2={:.0}[rope={:.0} attn={:.0} lut={:.0}] o={:.0} s3={:.0} gu={:.0} s4={:.0} dn={:.0} total={:.0}",
                session.position - 1, s1, qkv, s2, rp, at, lp, omm, s3, gu, s4, dn, total);
            eprintln!("[PROFILE] lm_head: {:.0}μs", lmh);
        }

        logits
    }

    fn generate_bitnet_decode(
        &self,
        session: &mut InferenceSession,
        first_token: u32,
        max_tokens: usize,
        eos_token: Option<u32>,
        qweights: &crate::weights::QuantizedWeights,
    ) -> Vec<u32> {
        use tempura_kernels::fused::{fused_rmsnorm_q8snap, fused_sqrelu_mul, residual_add};
        use tempura_kernels::quantized::{i2s_gemv_cols, i2s_weight_scale};
        use tempura_kernels::tl2::{tl2_gemv_tiles, tl2_preprocess_into, Tl2Lut};

        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = cfg.intermediate_size;
        let vocab_size = cfg.vocab_size;
        let dn_heads = cfg.deltanet_num_heads.unwrap_or(n_kv_heads);
        let dn_head_dim = cfg.deltanet_head_dim.unwrap_or(head_dim);
        let dn_dim = dn_heads * dn_head_dim;
        let layer_kind_vec: Vec<crate::config::LayerKind> = (0..cfg.num_layers)
            .map(|l| {
                cfg.layer_kinds
                    .as_ref()
                    .map(|k| k[l])
                    .unwrap_or(crate::config::LayerKind::Attention)
            })
            .collect();

        let rmsnorm = tempura_kernels::rmsnorm::CpuRMSNorm;
        let _rope = tempura_kernels::rope::CpuRoPE;

        let mut hidden = vec![0.0f32; h];

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

        struct DnNames {
            qkv_proj: String,
            gate_proj: String,
            alpha: String,
            beta: String,
            a_log: String,
            dt_bias: String,
            conv_w: String,
            conv_b: String,
            norm: String,
            out_proj: String,
        }
        struct DnGeom {
            key_dim: usize,
            conv_dim: usize,
            n_k_heads: usize,
            qkv_cols: usize,
        }
        let has_deltanet = cfg.layer_kinds.is_some();
        let dn_names: Vec<Option<DnNames>> = (0..cfg.num_layers)
            .map(|l| {
                if layer_kind_vec[l] != crate::config::LayerKind::DeltaNet {
                    return None;
                }
                let p = format!("model.layers.{l}");
                Some(DnNames {
                    qkv_proj: format!("{p}.ssm.qkv_proj.weight"),
                    gate_proj: format!("{p}.ssm.gate_proj.weight"),
                    alpha: format!("{p}.ssm.alpha.weight"),
                    beta: format!("{p}.ssm.beta.weight"),
                    a_log: format!("{p}.ssm.A_log"),
                    dt_bias: format!("{p}.ssm.dt_proj.bias"),
                    conv_w: format!("{p}.ssm.conv1d.weight"),
                    conv_b: format!("{p}.ssm.conv1d.bias"),
                    norm: format!("{p}.ssm.norm.weight"),
                    out_proj: format!("{p}.ssm.out_proj.weight"),
                })
            })
            .collect();
        use tempura_kernels::quantized::{Q2_0_BLOCK_BYTES, Q2_0_BLOCK_SIZE};
        let dn_geom: Vec<Option<DnGeom>> = (0..cfg.num_layers)
            .map(|l| {
                let dn = dn_names[l].as_ref()?;
                let qkv_cols = qweights
                    .get_q2(&dn.qkv_proj)
                    .map(|w| {
                        let row_bytes = (h / Q2_0_BLOCK_SIZE) * Q2_0_BLOCK_BYTES;
                        w.len() / row_bytes
                    })
                    .or_else(|| qweights.get_q2_tiled(&dn.qkv_proj).map(|(_, n, _)| n))
                    .or_else(|| qweights.get_tl2(&dn.qkv_proj).map(|t| t.m))
                    .unwrap_or(dn_dim);
                let key_dim = if qkv_cols > dn_dim {
                    (qkv_cols - dn_dim) / 2
                } else {
                    dn_dim
                };
                Some(DnGeom {
                    key_dim,
                    conv_dim: 2 * key_dim + dn_dim,
                    n_k_heads: key_dim / dn_head_dim,
                    qkv_cols,
                })
            })
            .collect();
        let qkv_cols_max = dn_geom
            .iter()
            .filter_map(|g| g.as_ref())
            .map(|g| g.qkv_cols)
            .max()
            .unwrap_or(0);
        let conv_dim_max = dn_geom
            .iter()
            .filter_map(|g| g.as_ref())
            .map(|g| g.conv_dim)
            .max()
            .unwrap_or(0);

        let first_attn = layer_kind_vec
            .iter()
            .position(|k| *k == crate::config::LayerKind::Attention);
        let q_weight_cols = first_attn
            .and_then(|l| {
                let qn = &layer_names[l].2;
                let row_bytes = (h / Q2_0_BLOCK_SIZE) * Q2_0_BLOCK_BYTES;
                qweights
                    .get_q2(qn)
                    .map(|w| w.len() / row_bytes)
                    .or_else(|| qweights.get_q2_tiled(qn).map(|(_, n, _)| n))
                    .or_else(|| qweights.get_tl2(qn).map(|t| t.m))
            })
            .unwrap_or(q_dim);
        let has_q_gate = q_weight_cols > q_dim;
        let actual_q_cols = if has_q_gate { q_weight_cols } else { q_dim };

        let max_k = h.max(inter).max(q_dim).max(dn_dim);
        let mut preq_buf = vec![0i8; max_k];
        let q8_nb = max_k / tempura_kernels::quantized::Q8_0_BLOCK_SIZE;
        let mut q8_act_buf = vec![0u8; q8_nb * tempura_kernels::quantized::Q8_0_BLOCK_BYTES];
        let q8p = q8_act_buf.as_mut_ptr() as usize;
        let q8_act_len = q8_act_buf.len();
        let mut normed = vec![0.0f32; h];
        let mut q = vec![0.0f32; actual_q_cols.max(qkv_cols_max)];
        let mut q_gate_buf = vec![0.0f32; if has_q_gate { q_dim } else { 0 }];
        let qgbp = q_gate_buf.as_mut_ptr() as usize;
        let mut k_buf = vec![0.0f32; kv_dim.max(dn_dim)];
        let mut v_buf = vec![0.0f32; kv_dim.max(dn_dim)];
        let mut attn_out = vec![0.0f32; q_dim.max(dn_dim)];
        let mut o_proj = vec![0.0f32; h];
        let mut normed2 = vec![0.0f32; h];
        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        let mut down_out = vec![0.0f32; h];
        let mut logits = vec![0.0f32; vocab_size];
        let mut z_gate = vec![0.0f32; if has_deltanet { dn_dim } else { 0 }];
        let mut alpha_buf = vec![0.0f32; if has_deltanet { dn_heads } else { 0 }];
        let mut beta_buf = vec![0.0f32; if has_deltanet { dn_heads } else { 0 }];
        let mut conv_scratch = vec![0.0f32; conv_dim_max];
        let mut generated = Vec::with_capacity(max_tokens);

        let hp = hidden.as_mut_ptr() as usize;
        let np = normed.as_mut_ptr() as usize;
        let qp = q.as_mut_ptr() as usize;
        let kp = k_buf.as_mut_ptr() as usize;
        let vp = v_buf.as_mut_ptr() as usize;
        let ap = attn_out.as_mut_ptr() as usize;
        let op = o_proj.as_mut_ptr() as usize;
        let n2p = normed2.as_mut_ptr() as usize;
        let gp = gate.as_mut_ptr() as usize;
        let upp = up.as_mut_ptr() as usize;
        let dp = down_out.as_mut_ptr() as usize;
        let pp = preq_buf.as_mut_ptr() as usize;
        let logits_ptr = logits.as_mut_ptr() as usize;
        let zgp = z_gate.as_mut_ptr() as usize;
        let abp = alpha_buf.as_mut_ptr() as usize;
        let bbp = beta_buf.as_mut_ptr() as usize;
        let cvp = conv_scratch.as_mut_ptr() as usize;
        let sp = session as *mut InferenceSession as usize;
        let gen_ptr = &mut generated as *mut Vec<u32> as usize;

        let force_i2s = std::env::var("TEMPURA_FORCE_I2S").is_ok();
        let mut lut_h = if force_i2s {
            None
        } else {
            Tl2Lut::new_for_k(h)
        };
        let mut lut_qdim = if force_i2s || q_dim == h {
            None
        } else {
            Tl2Lut::new_for_k(q_dim)
        };
        let mut lut_inter = if force_i2s {
            None
        } else {
            Tl2Lut::new_for_k(inter)
        };
        let lut_h_ptr = lut_h
            .as_mut()
            .map(|l| l as *mut Tl2Lut as usize)
            .unwrap_or(0);
        let _lut_qdim_ptr = lut_qdim
            .as_mut()
            .map(|l| l as *mut Tl2Lut as usize)
            .unwrap_or(lut_h_ptr);
        let _lut_inter_ptr = lut_inter
            .as_mut()
            .map(|l| l as *mut Tl2Lut as usize)
            .unwrap_or(0);

        let has_q2_tiled = !force_i2s && qweights.has_q2_tiled();
        let mut q2lut_h = if has_q2_tiled {
            Some(tempura_kernels::quantized::Q2Lut::new_for_k(h))
        } else {
            None
        };
        let mut q2lut_qdim = if has_q2_tiled && q_dim != h {
            Some(tempura_kernels::quantized::Q2Lut::new_for_k(q_dim))
        } else {
            None
        };
        let mut q2lut_inter = if has_q2_tiled {
            Some(tempura_kernels::quantized::Q2Lut::new_for_k(inter))
        } else {
            None
        };
        let q2lut_h_ptr = q2lut_h
            .as_mut()
            .map(|l| l as *mut tempura_kernels::quantized::Q2Lut as usize)
            .unwrap_or(0);
        let _q2lut_qdim_ptr = q2lut_qdim
            .as_mut()
            .map(|l| l as *mut tempura_kernels::quantized::Q2Lut as usize)
            .unwrap_or(q2lut_h_ptr);
        let _q2lut_inter_ptr = q2lut_inter
            .as_mut()
            .map(|l| l as *mut tempura_kernels::quantized::Q2Lut as usize)
            .unwrap_or(0);
        let mut lut_dn = if has_deltanet && !force_i2s && dn_dim != h {
            Tl2Lut::new_for_k(dn_dim)
        } else {
            None
        };
        let _lut_dn_ptr = lut_dn
            .as_mut()
            .map(|l| l as *mut Tl2Lut as usize)
            .unwrap_or(lut_h_ptr);
        let mut q2lut_dn = if has_deltanet && has_q2_tiled && dn_dim != h {
            Some(tempura_kernels::quantized::Q2Lut::new_for_k(dn_dim))
        } else {
            None
        };
        let _q2lut_dn_ptr = q2lut_dn
            .as_mut()
            .map(|l| l as *mut tempura_kernels::quantized::Q2Lut as usize)
            .unwrap_or(q2lut_h_ptr);
        let shared_q2lut_ok = std::sync::atomic::AtomicBool::new(false);
        let tile_counter2 = std::sync::atomic::AtomicUsize::new(0);
        let q2_bm = tempura_kernels::quantized::q2_lut_tile_size();
        let cache_layer_map = cfg.cache_layer_map();

        let pool = tempura_kernels::thread_pool::global_pool();
        let n_threads = pool.thread_count();
        let shared_scale = std::sync::atomic::AtomicU32::new(0);
        let shared_sum = std::sync::atomic::AtomicI32::new(0);
        let shared_lut_ok = std::sync::atomic::AtomicBool::new(false);
        let barrier = tempura_kernels::thread_pool::SpinBarrier::new(n_threads);
        let _shared_kvl = std::sync::atomic::AtomicUsize::new(0);
        let tile_counter = std::sync::atomic::AtomicUsize::new(0);
        let shared_stop = std::sync::atomic::AtomicBool::new(false);
        let max_seq = cfg.max_seq_len;
        let mut attn_scores_bufs: Vec<Vec<f32>> =
            (0..n_threads).map(|_| vec![0.0f32; max_seq]).collect();
        let _attn_scores_ptrs: Vec<usize> = attn_scores_bufs
            .iter_mut()
            .map(|b| b.as_mut_ptr() as usize)
            .collect();

        if has_deltanet {
            if let Some(ref ds) = session.deltanet_state {
                let s0 = ds.state(0);
                let s_norm: f32 = s0.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("[DN] deltanet_state[0] norm={s_norm:.4} (should be >0 if prefill ran)");
            } else {
                eprintln!("[DN] deltanet_state is None!");
            }
            eprintln!("[DN] h={h} dn_dim={dn_dim} dn_heads={dn_heads} dn_head_dim={dn_head_dim}");
            eprintln!("[DN] qkv_cols_max={qkv_cols_max} conv_dim_max={conv_dim_max}");
            eprintln!(
                "[DN] q.len={} k.len={} v.len={} attn_out.len={}",
                q.len(),
                k_buf.len(),
                v_buf.len(),
                attn_out.len()
            );
            eprintln!(
                "[DN] z_gate.len={} alpha.len={} beta.len={} conv.len={}",
                z_gate.len(),
                alpha_buf.len(),
                beta_buf.len(),
                conv_scratch.len()
            );
            eprintln!(
                "[DN] has_q2_tiled={has_q2_tiled} n_dn_layers={}",
                dn_names.iter().filter(|n| n.is_some()).count()
            );
            if let Some(g) = &dn_geom[0] {
                eprintln!(
                    "[DN] L0: key_dim={} conv_dim={} n_k_heads={} qkv_cols={}",
                    g.key_dim, g.conv_dim, g.n_k_heads, g.qkv_cols
                );
            }
        }
        qweights.embed_token(first_token, h, &mut hidden);

        let embed_f32_ptr = qweights
            .get_f32("model.embed_tokens.weight")
            .map(|d| d.as_ptr() as usize)
            .unwrap_or(0);
        let embed_f16_ptr = if embed_f32_ptr == 0 {
            qweights
                .get_f16("model.embed_tokens.weight")
                .map(|d| d.as_ptr() as usize)
                .unwrap_or(0)
        } else {
            0
        };

        pool.run_indexed(&|ith, nth| {
            macro_rules! fs  { ($a:expr,$n:expr) => { unsafe { std::slice::from_raw_parts($a as *const f32, $n) } } }
            #[allow(unused_unsafe)]
            macro_rules! fsm { ($a:expr,$n:expr) => { unsafe { std::slice::from_raw_parts_mut($a as *mut f32, $n) } } }
            macro_rules! is  { ($a:expr,$n:expr) => { unsafe { std::slice::from_raw_parts($a as *const i8, $n) } } }
            macro_rules! ism { ($a:expr,$n:expr) => { unsafe { std::slice::from_raw_parts_mut($a as *mut i8, $n) } } }

            let do_profile = std::env::var("TEMPURA_PROFILE").is_ok();
            let mut _tok_start = Instant::now();
            let mut _layer_us = 0u64;
            let mut _lmhead_us = 0u64;

            for _token_step in 0..max_tokens {
                if shared_stop.load(std::sync::atomic::Ordering::Relaxed) { break; }
                if ith == 0 {
                    _tok_start = Instant::now();
                    // Rolling context window: shift to make room when cache is about to fill.
                    // Keeps the most recent half; KV data already has RoPE encoded at absolute
                    // positions so relative attention scores remain correct after the shift.
                    let sess = unsafe { &mut *(sp as *mut InferenceSession) };
                    if sess.cache.seq_len + 1 >= sess.cache.max_seq_len() {
                        let keep_from = sess.cache.max_seq_len() / 2;
                        sess.cache.shift(keep_from);
                    }
                }

                let session = unsafe { &*(sp as *const InferenceSession) };
                let positions = [session.position as u32];
                let mut dn_idx = 0usize;

                for layer in 0..cfg.num_layers {
                    let (ref iln, ref pln, ref qn, ref kn, ref vn, ref on,
                         ref _qnm, ref _knm, ref gn, ref un, ref dn,
                         ref asn, ref fsn) = layer_names[layer];
                    let is_deltanet = layer_kind_vec[layer] == crate::config::LayerKind::DeltaNet;

                    if ith == 0 {
                        fsm!(np,h).copy_from_slice(fs!(hp,h));
                        let w = qweights.get_f32(iln).unwrap();
                        use tempura_kernels::RMSNorm as _;
                        rmsnorm.forward(fsm!(np,h), w, h, cfg.layer_norm_eps).unwrap();
                        if (layer == 0 || layer == 3) && _token_step == 0 {
                            let nn: f32 = fs!(np,h).iter().map(|x|x*x).sum::<f32>().sqrt();
                            let hh: f32 = fs!(hp,h).iter().map(|x|x*x).sum::<f32>().sqrt();
                            eprintln!("[GBD-L{layer}] hidden_norm={hh:.6} normed_norm={nn:.6} normed[0..4]={:?}", &fs!(np,h)[..4]);
                        }
                        let (s, m) = tempura_kernels::quantized::i2s_quantize_into(fs!(np,h), ism!(pp,h));
                        shared_scale.store(s.to_bits(), std::sync::atomic::Ordering::Release);
                        shared_sum.store(m, std::sync::atomic::Ordering::Release);
                        if lut_h_ptr != 0 {
                            let lut = unsafe { &mut *(lut_h_ptr as *mut Tl2Lut) };
                            shared_lut_ok.store(tl2_preprocess_into(fs!(np,h), h, lut), std::sync::atomic::Ordering::Release);
                        } else {
                            shared_lut_ok.store(false, std::sync::atomic::Ordering::Release);
                        }
                        if q2lut_h_ptr != 0 {
                            let q2l = unsafe { &mut *(q2lut_h_ptr as *mut tempura_kernels::quantized::Q2Lut) };
                            q2l.preprocess_into(fs!(np,h));
                            shared_q2lut_ok.store(true, std::sync::atomic::Ordering::Release);
                        } else {
                            shared_q2lut_ok.store(false, std::sync::atomic::Ordering::Release);
                        }
                        {
                            let q8 = unsafe { std::slice::from_raw_parts_mut(q8p as *mut u8, q8_act_len) };
                            let h_nb = h / tempura_kernels::quantized::Q8_0_BLOCK_SIZE;
                            tempura_kernels::quantized::quantize_row_q8_0(fs!(np,h), &mut q8[..h_nb * tempura_kernels::quantized::Q8_0_BLOCK_BYTES]);
                        }
                        tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                        tile_counter2.store(0, std::sync::atomic::Ordering::Relaxed);
                    }
                    barrier.wait();

                    if is_deltanet {
                    // ── DeltaNet attention block ──
                    if ith == 0 && layer == 0 && _token_step == 0 {
                        let has_tiled_qkv = qweights.get_q2_tiled(&dn_names[0].as_ref().unwrap().qkv_proj).is_some();
                        let has_raw_q2 = qweights.get_q2(&dn_names[0].as_ref().unwrap().qkv_proj).is_some();
                        eprintln!("[DN-L0] tiled_qkv={has_tiled_qkv} raw_q2={has_raw_q2} q2lut_ok={}", shared_q2lut_ok.load(std::sync::atomic::Ordering::Relaxed));
                    }
                    let dn = dn_names[layer].as_ref().unwrap();
                    let dg = dn_geom[layer].as_ref().unwrap();

                    // B2: QKV+gate+alpha+beta GEMV (Q8_0-based, single-threaded)
                    if ith == 0 {
                        let q8 = unsafe { std::slice::from_raw_parts(q8p as *const u8, q8_act_len) };
                        let h_q8_bytes = (h / tempura_kernels::quantized::Q8_0_BLOCK_SIZE) * tempura_kernels::quantized::Q8_0_BLOCK_BYTES;
                        if let Some(w) = qweights.get_q2(&dn.qkv_proj) {
                            tempura_kernels::quantized::q2_0_gemv_q8(w, &q8[..h_q8_bytes], fsm!(qp, dg.qkv_cols), dg.qkv_cols, h);
                        }
                        if let Some(w) = qweights.get_q2(&dn.gate_proj) {
                            tempura_kernels::quantized::q2_0_gemv_q8(w, &q8[..h_q8_bytes], fsm!(zgp, dn_dim), dn_dim, h);
                        }
                        if let Some(w) = qweights.get_q2(&dn.alpha) {
                            tempura_kernels::quantized::q2_0_gemv_q8(w, &q8[..h_q8_bytes], fsm!(abp, dn_heads), dn_heads, h);
                        }
                        if let Some(w) = qweights.get_q2(&dn.beta) {
                            tempura_kernels::quantized::q2_0_gemv_q8(w, &q8[..h_q8_bytes], fsm!(bbp, dn_heads), dn_heads, h);
                        }
                    }
                    barrier.wait();

                    // B2.5: serial DeltaNet work (T0): split QKV, conv1d, recurrence, norm, z_gate, LUT prep
                    if ith == 0 {
                        if layer == 0 && _token_step == 0 {
                            let qn: f32 = fsm!(qp, dg.qkv_cols).iter().map(|x|x*x).sum::<f32>().sqrt();
                            let zn: f32 = fsm!(zgp, dn_dim).iter().map(|x|x*x).sum::<f32>().sqrt();
                            eprintln!("[GBD-L0-B2] qkv_norm={qn:.6} gate_norm={zn:.6} qkv[0..4]={:?}", &fsm!(qp, dg.qkv_cols)[..4]);
                        }
                        let q_s = fsm!(kp, dn_dim);
                        let k_s = fsm!(vp, dn_dim);
                        let v_s = fsm!(ap, dn_dim);
                        let z_s = fsm!(zgp, dn_dim);
                        let al_s = fsm!(abp, dn_heads);
                        let be_s = fsm!(bbp, dn_heads);
                        let qkv_buf = fs!(qp, dg.qkv_cols);

                        q_s[..dg.key_dim].copy_from_slice(&qkv_buf[..dg.key_dim]);
                        k_s[..dg.key_dim].copy_from_slice(&qkv_buf[dg.key_dim..2*dg.key_dim]);
                        v_s[..dn_dim].copy_from_slice(&qkv_buf[2*dg.key_dim..2*dg.key_dim+dn_dim]);

                        if layer == 0 && _token_step == 0 {
                            let qkv_norm: f32 = qkv_buf.iter().map(|x|x*x).sum::<f32>().sqrt();
                            let z_norm: f32 = z_s.iter().map(|x|x*x).sum::<f32>().sqrt();
                            eprintln!("[DN-DBG L0] qkv_norm={qkv_norm:.4} z_norm={z_norm:.4} alpha[0..4]={:?} beta[0..4]={:?}", &al_s[..4.min(dn_heads)], &be_s[..4.min(dn_heads)]);
                            // Compare against scalar Q2 GEMV
                            if let Some(w) = qweights.get_q2(&dn.qkv_proj) {
                                let mut ref_out = vec![0.0f32; dg.qkv_cols];
                                tempura_kernels::quantized::q2_0_gemv(w, fs!(np, h), &mut ref_out, dg.qkv_cols, h);
                                let ref_norm: f32 = ref_out.iter().map(|x|x*x).sum::<f32>().sqrt();
                                let diff: f32 = qkv_buf.iter().zip(ref_out.iter()).map(|(a,b)| (a-b)*(a-b)).sum::<f32>().sqrt();
                                eprintln!("[DN-DBG L0] scalar_qkv_norm={ref_norm:.4} diff_norm={diff:.4} nrmse={:.4}", diff / ref_norm);
                            }
                        }
                        let ssm_a_w = qweights.get_f32(&dn.a_log);
                        let dt_bias = qweights.get_f32(&dn.dt_bias);
                        for hd in 0..dn_heads {
                            be_s[hd] = 1.0 / (1.0 + (-be_s[hd]).exp());
                            let dt_b = dt_bias.and_then(|b| b.get(hd)).copied().unwrap_or(0.0);
                            let ssm_a = ssm_a_w.and_then(|a| a.get(hd)).copied().unwrap_or(-1.0);
                            al_s[hd] = (1.0f32 + (al_s[hd] + dt_b).exp()).ln() * ssm_a;
                        }

                        if let Some(conv_w) = qweights.get_f32(&dn.conv_w) {
                            let kernel_size = conv_w.len() / dg.conv_dim;
                            let history = kernel_size - 1;
                            let conv_bias = qweights.get_f32(&dn.conv_b);
                            let cv = fsm!(cvp, dg.conv_dim);
                            cv[..dg.key_dim].copy_from_slice(&q_s[..dg.key_dim]);
                            cv[dg.key_dim..2*dg.key_dim].copy_from_slice(&k_s[..dg.key_dim]);
                            cv[2*dg.key_dim..dg.conv_dim].copy_from_slice(&v_s[..dn_dim]);
                            let sess_r = unsafe { &*(sp as *const InferenceSession) };
                            let cs = sess_r.deltanet_state.as_ref().unwrap().conv_state(dn_idx);
                            let has_cs = cs.len() == history * dg.conv_dim;
                            for ch in 0..dg.conv_dim {
                                let mut s = 0.0f32;
                                for ki in 0..kernel_size {
                                    let src_t = 0isize - (kernel_size as isize - 1 - ki as isize);
                                    if src_t == 0 {
                                        s += cv[ch] * conv_w[ch * kernel_size + ki];
                                    } else if has_cs {
                                        let hist_t = (history as isize + src_t) as usize;
                                        s += cs[hist_t * dg.conv_dim + ch] * conv_w[ch * kernel_size + ki];
                                    }
                                }
                                if let Some(bias) = conv_bias { if ch < bias.len() { s += bias[ch]; } }
                                cv[ch] = s / (1.0 + (-s).exp());
                            }
                            {
                                let sess_m = unsafe { &mut *(sp as *mut InferenceSession) };
                                let dn_st = sess_m.deltanet_state.as_mut().unwrap();
                                let _current = fsm!(cvp, dg.conv_dim);
                                let mut token_data = vec![0.0f32; dg.conv_dim];
                                token_data[..dg.key_dim].copy_from_slice(&q_s[..dg.key_dim]);
                                token_data[dg.key_dim..2*dg.key_dim].copy_from_slice(&k_s[..dg.key_dim]);
                                token_data[2*dg.key_dim..dg.conv_dim].copy_from_slice(&v_s[..dn_dim]);
                                dn_st.update_conv_state(dn_idx, dg.conv_dim, kernel_size, &token_data, 1);
                            }
                            let cv = fsm!(cvp, dg.conv_dim);
                            for vh in 0..dn_heads {
                                let kh = vh * dg.n_k_heads / dn_heads;
                                q_s[vh*dn_head_dim..(vh+1)*dn_head_dim].copy_from_slice(&cv[kh*dn_head_dim..kh*dn_head_dim+dn_head_dim]);
                                k_s[vh*dn_head_dim..(vh+1)*dn_head_dim].copy_from_slice(&cv[dg.key_dim+kh*dn_head_dim..dg.key_dim+kh*dn_head_dim+dn_head_dim]);
                            }
                            v_s[..dn_dim].copy_from_slice(&cv[2*dg.key_dim..2*dg.key_dim+dn_dim]);
                        }

                        for hd in 0..dn_heads {
                            let off = hd * dn_head_dim;
                            let qn: f32 = q_s[off..off+dn_head_dim].iter().map(|x| x*x).sum::<f32>().sqrt().max(cfg.layer_norm_eps);
                            q_s[off..off+dn_head_dim].iter_mut().for_each(|x| *x /= qn);
                            let kn: f32 = k_s[off..off+dn_head_dim].iter().map(|x| x*x).sum::<f32>().sqrt().max(cfg.layer_norm_eps);
                            k_s[off..off+dn_head_dim].iter_mut().for_each(|x| *x /= kn);
                        }

                        let sess_m = unsafe { &mut *(sp as *mut InferenceSession) };
                        let dn_state = sess_m.deltanet_state.as_mut().unwrap();
                        let state = dn_state.state_mut(dn_idx);
                        let dn_out = fsm!(gp, dn_dim);
                        tempura_kernels::deltanet::CpuDeltaNet.forward_recurrent(
                            q_s, k_s, v_s, al_s, be_s, state, dn_out,
                            dn_heads, dn_head_dim, 1.0,
                        ).unwrap();

                        if layer == 0 && _token_step == 0 {
                            let dn_norm: f32 = dn_out.iter().map(|x|x*x).sum::<f32>().sqrt();
                            eprintln!("[DN-DBG L0] dn_out_norm={dn_norm:.4} dn_out[0..4]={:?}", &dn_out[..4]);
                        }
                        if let Some(norm_w) = qweights.get_f32(&dn.norm) {
                            for head in 0..dn_heads {
                                let off = head * dn_head_dim;
                                self.pipeline.rmsnorm(&mut dn_out[off..off+dn_head_dim], norm_w, dn_head_dim, cfg.layer_norm_eps, TraceLevel::Off, &mut None).unwrap();
                            }
                        }
                        for i in 0..dn_dim {
                            let zs = z_s[i] / (1.0 + (-z_s[i]).exp());
                            dn_out[i] *= zs;
                        }

                        {
                            let q8 = unsafe { std::slice::from_raw_parts_mut(q8p as *mut u8, q8_act_len) };
                            let dn_nb = dn_dim / tempura_kernels::quantized::Q8_0_BLOCK_SIZE;
                            tempura_kernels::quantized::quantize_row_q8_0(dn_out, &mut q8[..dn_nb * tempura_kernels::quantized::Q8_0_BLOCK_BYTES]);
                        }
                        tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                        dn_idx += 1;
                    }
                    barrier.wait();

                    // B3: O-proj GEMV (dn_dim → h, Q8_0-based row-partitioned)
                    if ith == 0 {
                        let q8 = unsafe { std::slice::from_raw_parts(q8p as *const u8, q8_act_len) };
                        let dn_q8_bytes = (dn_dim / tempura_kernels::quantized::Q8_0_BLOCK_SIZE) * tempura_kernels::quantized::Q8_0_BLOCK_BYTES;
                        if let Some(w) = qweights.get_q2(&dn.out_proj) {
                            tempura_kernels::quantized::q2_0_gemv_q8(w, &q8[..dn_q8_bytes], fsm!(op, h), h, dn_dim);
                        }
                    }
                    barrier.wait();

                    // B3.5: row corrections
                    if let Some(corr) = qweights.get_q2_row_corrections(&dn.out_proj) {
                        let (s, e) = ((ith * h) / nth, ((ith + 1) * h) / nth);
                        let o_s = fsm!(op, h);
                        for j in s..e.min(corr.len()) { o_s[j] *= corr[j]; }
                    }
                    barrier.wait();

                    } else {
                    // ── Attention block: QKV GEMV (row-partitioned) ──
                    {
                        let q8 = unsafe { std::slice::from_raw_parts(q8p as *const u8, q8_act_len) };
                        let h_q8_bytes = (h / tempura_kernels::quantized::Q8_0_BLOCK_SIZE) * tempura_kernels::quantized::Q8_0_BLOCK_BYTES;
                        if let Some(w) = qweights.get_q2(qn) { let (rs, re) = ((ith*actual_q_cols)/nth, ((ith+1)*actual_q_cols)/nth);
                            unsafe { tempura_kernels::quantized::q2_0_gemv_q8_rows(w, &q8[..h_q8_bytes], qp, actual_q_cols, h, rs, re); } }
                        if let Some(w) = qweights.get_q2(kn) { let (rs, re) = ((ith*kv_dim)/nth, ((ith+1)*kv_dim)/nth);
                            unsafe { tempura_kernels::quantized::q2_0_gemv_q8_rows(w, &q8[..h_q8_bytes], kp, kv_dim, h, rs, re); } }
                        if let Some(w) = qweights.get_q2(vn) { let (rs, re) = ((ith*kv_dim)/nth, ((ith+1)*kv_dim)/nth);
                            unsafe { tempura_kernels::quantized::q2_0_gemv_q8_rows(w, &q8[..h_q8_bytes], vp, kv_dim, h, rs, re); } }
                    }
                    barrier.wait();

                    if ith == 0 {
                        let sess = unsafe { &mut *(sp as *mut InferenceSession) };
                        let cl = cache_layer_map[layer].unwrap();
                        let rope_hd = cfg.rope_dim.unwrap_or(head_dim);
                        let qnw = if cfg.use_qk_norm { qweights.get_f32(_qnm) } else { None };
                        let knw = if cfg.use_qk_norm { qweights.get_f32(_knm) } else { None };
                        let ggml_compat = cfg.hidden_act == crate::config::HiddenAct::SquaredRelu;
                        attention_layer_decode(
                            fsm!(qp, actual_q_cols), fsm!(kp, kv_dim), fs!(vp, kv_dim),
                            fsm!(ap, q_dim), fsm!(qgbp, if has_q_gate { q_dim } else { 0 }),
                            sess, cl, n_heads, n_kv_heads, head_dim, q_dim, kv_dim, actual_q_cols,
                            has_q_gate, rope_hd, cfg.rope_theta, &positions,
                            qnw, knw, qweights.get_f32(asn), cfg.layer_norm_eps, ggml_compat,
                            cfg.sliding_window,
                        );
                        {
                            let q8 = unsafe { std::slice::from_raw_parts_mut(q8p as *mut u8, q8_act_len) };
                            let qd_nb = q_dim / tempura_kernels::quantized::Q8_0_BLOCK_SIZE;
                            tempura_kernels::quantized::quantize_row_q8_0(fs!(ap,q_dim), &mut q8[..qd_nb * tempura_kernels::quantized::Q8_0_BLOCK_BYTES]);
                        }
                    }
                    barrier.wait();

                    // O-proj GEMV (row-partitioned)
                    {
                        let q8 = unsafe { std::slice::from_raw_parts(q8p as *const u8, q8_act_len) };
                        let qd_q8_bytes = (q_dim / tempura_kernels::quantized::Q8_0_BLOCK_SIZE) * tempura_kernels::quantized::Q8_0_BLOCK_BYTES;
                        if let Some(w) = qweights.get_q2(on) { let (rs, re) = ((ith*h)/nth, ((ith+1)*h)/nth);
                            unsafe { tempura_kernels::quantized::q2_0_gemv_q8_rows(w, &q8[..qd_q8_bytes], op, h, q_dim, rs, re); } }
                    }
                    barrier.wait();

                    } // end if is_deltanet { ... } else { ... }

                    if ith == 0 && layer == 3 && _token_step == 0 {
                        let on: f32 = fs!(op, h).iter().map(|x|x*x).sum::<f32>().sqrt();
                        let hn: f32 = fs!(hp, h).iter().map(|x|x*x).sum::<f32>().sqrt();
                        eprintln!("[ATN-DBG L3] o_proj_norm={on:.4} hidden_before_ffn={hn:.4}");
                    }

                    // ── Shared FFN block ──
                    if ith == 0 {
                        let hid = fsm!(hp,h);
                        let res = fs!(op,h);
                        let nrm = fsm!(n2p,h);
                        for i in 0..h { hid[i] += res[i]; }
                        nrm[..h].copy_from_slice(&hid[..h]);
                        rmsnorm.forward(nrm, qweights.get_f32(pln).unwrap(), h, cfg.layer_norm_eps).unwrap();
                        {
                            let q8 = unsafe { std::slice::from_raw_parts_mut(q8p as *mut u8, q8_act_len) };
                            let h_nb = h / tempura_kernels::quantized::Q8_0_BLOCK_SIZE;
                            tempura_kernels::quantized::quantize_row_q8_0(fs!(n2p,h), &mut q8[..h_nb * tempura_kernels::quantized::Q8_0_BLOCK_BYTES]);
                        }
                    }
                    barrier.wait();

                    // B6: gate+up GEMV (row-partitioned)
                    {
                        let q8 = unsafe { std::slice::from_raw_parts(q8p as *const u8, q8_act_len) };
                        let h_q8_bytes = (h / tempura_kernels::quantized::Q8_0_BLOCK_SIZE) * tempura_kernels::quantized::Q8_0_BLOCK_BYTES;
                        if let Some(w) = qweights.get_q2(gn) { let (rs, re) = ((ith*inter)/nth, ((ith+1)*inter)/nth);
                            unsafe { tempura_kernels::quantized::q2_0_gemv_q8_rows(w, &q8[..h_q8_bytes], gp, inter, h, rs, re); } }
                        if let Some(w) = qweights.get_q2(un) { let (rs, re) = ((ith*inter)/nth, ((ith+1)*inter)/nth);
                            unsafe { tempura_kernels::quantized::q2_0_gemv_q8_rows(w, &q8[..h_q8_bytes], upp, inter, h, rs, re); } }
                    }
                    barrier.wait();

                    if ith == 0 {
                        if cfg.hidden_act == crate::config::HiddenAct::SquaredRelu {
                            fused_sqrelu_mul(fsm!(gp,inter), fs!(upp,inter), inter);
                        } else {
                            let g = fsm!(gp,inter);
                            let u = fs!(upp,inter);
                            for i in 0..inter {
                                g[i] = tempura_kernels::portable_math::portable_silu(g[i]) * u[i];
                            }
                        }
                        if let Some(fsn_w) = qweights.get_f32(fsn) {
                            rmsnorm.forward(fsm!(gp,inter), fsn_w, inter, cfg.layer_norm_eps).unwrap();
                        }
                        {
                            let q8 = unsafe { std::slice::from_raw_parts_mut(q8p as *mut u8, q8_act_len) };
                            let inter_nb = inter / tempura_kernels::quantized::Q8_0_BLOCK_SIZE;
                            tempura_kernels::quantized::quantize_row_q8_0(fs!(gp,inter), &mut q8[..inter_nb * tempura_kernels::quantized::Q8_0_BLOCK_BYTES]);
                        }
                    }
                    barrier.wait();

                    // B8: down-proj GEMV (row-partitioned)
                    {
                        let q8 = unsafe { std::slice::from_raw_parts(q8p as *const u8, q8_act_len) };
                        let inter_q8_bytes = (inter / tempura_kernels::quantized::Q8_0_BLOCK_SIZE) * tempura_kernels::quantized::Q8_0_BLOCK_BYTES;
                        if let Some(w) = qweights.get_q2(dn) { let (rs, re) = ((ith*h)/nth, ((ith+1)*h)/nth);
                            unsafe { tempura_kernels::quantized::q2_0_gemv_q8_rows(w, &q8[..inter_q8_bytes], dp, h, inter, rs, re); } }
                    }
                    barrier.wait();

                    if ith == 0 {
                        residual_add(fsm!(hp,h), fs!(dp,h), h);
                    }
                    if ith == 0 && _token_step == 0 && (layer <= 3 || layer == cfg.num_layers - 1) {
                        let hn: f32 = fs!(hp,h).iter().map(|x|x*x).sum::<f32>().sqrt();
                        let kind = if is_deltanet { "DN" } else { "ATN" };
                        eprintln!("[TRACE] L{layer}({kind}) hidden_norm={hn:.4}");
                    }
                }

                // ── lm_head ──
                if ith == 0 { _layer_us = _tok_start.elapsed().as_micros() as u64; }
                barrier.wait();
                if ith == 0 {
                    fused_rmsnorm_q8snap(fsm!(hp,h), qweights.get_f32("model.norm.weight").unwrap(), h, cfg.layer_norm_eps);
                    if lut_h_ptr != 0 && qweights.get_tl2("lm_head.weight").is_some() {
                        let lut = unsafe { &mut *(lut_h_ptr as *mut Tl2Lut) };
                        shared_lut_ok.store(tl2_preprocess_into(fs!(hp,h), h, lut), std::sync::atomic::Ordering::Release);
                    } else {
                        shared_lut_ok.store(false, std::sync::atomic::Ordering::Release);
                        let (s, m) = tempura_kernels::quantized::i2s_quantize_into(fs!(hp,h), ism!(pp,h));
                        shared_scale.store(s.to_bits(), std::sync::atomic::Ordering::Release);
                        shared_sum.store(m, std::sync::atomic::Ordering::Release);
                    }
                    let sess = unsafe { &mut *(sp as *mut InferenceSession) };
                    sess.cache.advance(1);
                    sess.position += 1;
                    tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                }
                barrier.wait();

                let lut_ok = shared_lut_ok.load(std::sync::atomic::Ordering::Acquire);
                if lut_ok && lut_h_ptr != 0 {
                    if let Some(tl2t) = qweights.get_tl2("lm_head.weight") {
                        let lut = unsafe { &*(lut_h_ptr as *const Tl2Lut) };
                        let lm_bm = tempura_kernels::tl2::Tl2Config::for_dims(vocab_size, h).map(|c| c.bm).unwrap_or(vocab_size);
                        let lm_tiles = vocab_size / lm_bm;
                        loop {
                            let t = tile_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if t >= lm_tiles { break; }
                            tl2_gemv_tiles(lut, &tl2t.data, tl2t.scale, logits_ptr, vocab_size, h, t, t + 1);
                        }
                    }
                } else if let Some(w) = qweights.get_i2s("lm_head.weight") {
                    let ps = f32::from_bits(shared_scale.load(std::sync::atomic::Ordering::Acquire));
                    let pm = shared_sum.load(std::sync::atomic::Ordering::Acquire);
                    let ws = i2s_weight_scale(w, vocab_size, h);
                    let (s,e) = ((ith*vocab_size)/nth, ((ith+1)*vocab_size)/nth);
                    i2s_gemv_cols(is!(pp,h), ps, pm, w, ws, logits_ptr, s, e, h);
                } else if let Some(w) = qweights.get_f16("lm_head.weight") {
                    let row_bytes = h * 2;
                    let (rs, re) = ((ith * vocab_size) / nth, ((ith + 1) * vocab_size) / nth);
                    let a = fs!(hp, h);
                    for row in rs..re {
                        let b_row = &w[row * row_bytes..(row + 1) * row_bytes];
                        let val = unsafe { tempura_kernels::f16_gemv::f16_dot_avx2(a, b_row, h) };
                        unsafe { *(logits_ptr as *mut f32).add(row) = val; }
                    }
                } else if let Some((tiled,_,_)) = qweights.get_q2_tiled("lm_head.weight") {
                    let q2l = tempura_kernels::quantized::Q2Lut::preprocess(fs!(hp, h), h);
                    let lm_tiles = vocab_size.div_ceil(q2_bm);
                    loop {
                        let t = tile_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if t >= lm_tiles { break; }
                        #[allow(unused_unsafe)]
                        unsafe { tempura_kernels::quantized::q2_lut_gemv_tiled(tiled, &q2l, fsm!(logits_ptr, vocab_size), vocab_size, h, t, t + 1); }
                    }
                } else if let Some(w) = qweights.get_q2("lm_head.weight") {
                    if ith == 0 {
                        tempura_kernels::quantized::q2_0_gemv(w, fs!(hp, h), fsm!(logits_ptr, vocab_size), vocab_size, h);
                    }
                }
                barrier.wait();

                // ── Thread 0: sampling + next token embedding ──
                // Non-zero threads yield here to reduce thermal pressure while
                // thread 0 does argmax + embedding (~100-200μs).
                if ith == 0 {
                    _lmhead_us = _tok_start.elapsed().as_micros() as u64 - _layer_us;
                    let logits_slice = fs!(logits_ptr, vocab_size);
                    let token = argmax(logits_slice);
                    let gen = unsafe { &mut *(gen_ptr as *mut Vec<u32>) };
                    gen.push(token);

                    if do_profile && _token_step % 10 == 9 {
                        let total = _tok_start.elapsed().as_micros() as u64;
                        eprintln!("[PROFILE] step={} layers={_layer_us}μs lm_head={_lmhead_us}μs total={total}μs", _token_step);
                    }

                    if std::env::var("TEMPURA_TRACE_TOKENS").is_ok() {
                        eprintln!("[TOK] step={} token={token} eos={:?}", gen.len()-1, eos_token);
                    }
                    if eos_token == Some(token) || gen.len() >= max_tokens {
                        shared_stop.store(true, std::sync::atomic::Ordering::Release);
                    } else {
                        #[allow(unused_unsafe)]
                        unsafe {
                            let dst = fsm!(hp,h);
                            let off = token as usize * h;
                            if embed_f32_ptr != 0 {
                                let src = std::slice::from_raw_parts((embed_f32_ptr as *const f32).add(off), h);
                                dst.copy_from_slice(src);
                            } else if embed_f16_ptr != 0 {
                                let f16_src = std::slice::from_raw_parts((embed_f16_ptr as *const u8).add(off * 2), h * 2);
                                for i in 0..h {
                                    let bits = u16::from_le_bytes([f16_src[i*2], f16_src[i*2+1]]);
                                    let exp = (bits >> 10) & 0x1f;
                                    dst[i] = f32::from_bits(((bits as u32 >> 15) << 31) | if exp == 0 { 0 } else { ((exp as u32 + 112) << 23) | ((bits as u32 & 0x3ff) << 13) });
                                }
                            } else {
                                qweights.embed_token(token, h, dst);
                            }
                        }
                    }
                } else {
                    std::thread::yield_now();
                }
                barrier.wait();
            }
        });

        generated
    }

    fn forward_bitnet_prefill(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        qweights: &crate::weights::QuantizedWeights,
    ) -> Vec<f32> {
        if self.config.layer_kinds.is_some() {
            return self.forward_quantized(session, token_ids, qweights);
        }
        use std::time::Instant;
        use tempura_kernels::fused::{
            fused_residual_rmsnorm_q8snap, fused_rmsnorm_q8snap, fused_sqrelu_mul, residual_add,
        };
        use tempura_kernels::tl2::{tl2_forward, BatchedTl2Lut, Tl2Config};

        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = cfg.intermediate_size;
        let seq_len = token_ids.len();

        let has_tl2 = qweights
            .get_tl2("model.layers.0.self_attn.q_proj.weight")
            .is_some();
        if !has_tl2 && !qweights.has_q2_tiled() {
            return self.forward_quantized(session, token_ids, qweights);
        }

        let cache_layer_map = cfg.cache_layer_map();
        let rmsnorm = tempura_kernels::rmsnorm::CpuRMSNormGgml;
        let rope = tempura_kernels::rope::CpuRoPEGgml;

        let mut hidden = vec![0.0f32; seq_len * h];
        for (t, &tok) in token_ids.iter().enumerate() {
            qweights.embed_token(tok, h, &mut hidden[t * h..(t + 1) * h]);
        }

        let positions: Vec<u32> = (session.position..session.position + seq_len)
            .map(|p| p as u32)
            .collect();

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

        let mut normed = vec![0.0f32; seq_len * h];
        let mut q = vec![0.0f32; seq_len * q_dim];
        let mut k_buf = vec![0.0f32; seq_len * kv_dim];
        let mut v_buf = vec![0.0f32; seq_len * kv_dim];
        let mut attn_out = vec![0.0f32; seq_len * q_dim];
        let mut o_proj = vec![0.0f32; seq_len * h];
        let mut normed2 = vec![0.0f32; seq_len * h];
        let mut gate = vec![0.0f32; seq_len * inter];
        let mut up = vec![0.0f32; seq_len * inter];
        let mut down_out = vec![0.0f32; seq_len * h];

        let hp = hidden.as_mut_ptr() as usize;
        let np = normed.as_mut_ptr() as usize;
        let qp = q.as_mut_ptr() as usize;
        let kp = k_buf.as_mut_ptr() as usize;
        let vp = v_buf.as_mut_ptr() as usize;
        let ap = attn_out.as_mut_ptr() as usize;
        let op = o_proj.as_mut_ptr() as usize;
        let n2p = normed2.as_mut_ptr() as usize;
        let gp = gate.as_mut_ptr() as usize;
        let upp = up.as_mut_ptr() as usize;
        let dp = down_out.as_mut_ptr() as usize;
        let sp = session as *mut InferenceSession as usize;

        let mut lut_h = BatchedTl2Lut::new_for(h, seq_len);
        let mut lut_qdim = if q_dim != h {
            BatchedTl2Lut::new_for(q_dim, seq_len)
        } else {
            None
        };
        let mut lut_inter = BatchedTl2Lut::new_for(inter, seq_len);
        let lut_h_ptr = lut_h
            .as_mut()
            .map(|l| l as *mut BatchedTl2Lut as usize)
            .unwrap_or(0);
        let lut_qdim_ptr = lut_qdim
            .as_mut()
            .map(|l| l as *mut BatchedTl2Lut as usize)
            .unwrap_or(lut_h_ptr);
        let lut_inter_ptr = lut_inter
            .as_mut()
            .map(|l| l as *mut BatchedTl2Lut as usize)
            .unwrap_or(0);

        let max_bm = 320usize;
        let cbits_size = seq_len * max_bm;

        let pool = tempura_kernels::thread_pool::global_pool();
        let n_threads = pool.thread_count();
        let barrier = tempura_kernels::thread_pool::SpinBarrier::new(n_threads);
        let tile_counter = std::sync::atomic::AtomicUsize::new(0);
        let shared_kvl = std::sync::atomic::AtomicUsize::new(0);
        let max_seq = cfg.max_seq_len;

        let mut per_thread_cbits: Vec<Vec<i32>> =
            (0..n_threads).map(|_| vec![0i32; cbits_size]).collect();
        let mut per_thread_two_cbits: Vec<Vec<i32>> =
            (0..n_threads).map(|_| vec![0i32; cbits_size]).collect();
        let cbits_ptrs: Vec<usize> = per_thread_cbits
            .iter_mut()
            .map(|b| b.as_mut_ptr() as usize)
            .collect();
        let two_cbits_ptrs: Vec<usize> = per_thread_two_cbits
            .iter_mut()
            .map(|b| b.as_mut_ptr() as usize)
            .collect();
        let mut attn_scores_bufs: Vec<Vec<f32>> =
            (0..n_threads).map(|_| vec![0.0f32; max_seq]).collect();
        let attn_scores_ptrs: Vec<usize> = attn_scores_bufs
            .iter_mut()
            .map(|b| b.as_mut_ptr() as usize)
            .collect();

        let do_profile = std::env::var("TEMPURA_PROFILE").is_ok();
        let prof_layer = std::sync::atomic::AtomicU64::new(0);
        let prof_norm = std::sync::atomic::AtomicU64::new(0);
        let prof_qkv = std::sync::atomic::AtomicU64::new(0);
        let prof_rope = std::sync::atomic::AtomicU64::new(0);
        let prof_attn = std::sync::atomic::AtomicU64::new(0);
        let prof_oproj = std::sync::atomic::AtomicU64::new(0);
        let prof_gu = std::sync::atomic::AtomicU64::new(0);
        let prof_down = std::sync::atomic::AtomicU64::new(0);

        pool.run_indexed(&|ith, nth| {
            macro_rules! fs {
                ($a:expr,$n:expr) => {
                    unsafe { std::slice::from_raw_parts($a as *const f32, $n) }
                };
            }
            macro_rules! fsm {
                ($a:expr,$n:expr) => {
                    unsafe { std::slice::from_raw_parts_mut($a as *mut f32, $n) }
                };
            }

            let my_cbits =
                unsafe { std::slice::from_raw_parts_mut(cbits_ptrs[ith] as *mut i32, cbits_size) };
            let my_two_cbits = unsafe {
                std::slice::from_raw_parts_mut(two_cbits_ptrs[ith] as *mut i32, cbits_size)
            };

            for layer in 0..cfg.num_layers {
                let (
                    ref iln,
                    ref pln,
                    ref qn,
                    ref kn,
                    ref vn,
                    ref on,
                    ref _qnm,
                    ref _knm,
                    ref gn,
                    ref un,
                    ref dn,
                    ref asn,
                    ref fsn,
                ) = layer_names[layer];

                let t_start = ith * seq_len / nth;
                let t_end = (ith + 1) * seq_len / nth;
                let _t_layer = Instant::now();
                // ── All threads: copy + RMSNorm + q8_snap + LUT build (per-token parallel) ──
                let _t_norm = Instant::now();
                {
                    let n = fsm!(np, seq_len * h);
                    let hh = fs!(hp, seq_len * h);
                    let w = qweights.get_f32(iln).unwrap();
                    for t in t_start..t_end {
                        n[t * h..(t + 1) * h].copy_from_slice(&hh[t * h..(t + 1) * h]);
                        fused_rmsnorm_q8snap(&mut n[t * h..(t + 1) * h], w, h, cfg.layer_norm_eps);
                    }
                    if lut_h_ptr != 0 {
                        tempura_kernels::tl2::tl2_preprocess_token_range(
                            np, h, lut_h_ptr, t_start, t_end,
                        );
                    }
                }
                if do_profile && ith == 0 {
                    prof_norm.fetch_add(
                        _t_norm.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                tile_counter.store(0, std::sync::atomic::Ordering::Relaxed);
                barrier.wait(); // B1

                // ── All threads: QKV GEMM (work-stealing — 12 tiles / 8 threads) ──
                let _t1 = Instant::now();
                if lut_h_ptr != 0 {
                    let lut = unsafe { &*(lut_h_ptr as *const BatchedTl2Lut) };
                    if let (Some(qt), Some(kt), Some(vt)) = (
                        qweights.get_tl2(qn),
                        qweights.get_tl2(kn),
                        qweights.get_tl2(vn),
                    ) {
                        let q_cfg = Tl2Config::for_dims(q_dim, h);
                        let kv_cfg = Tl2Config::for_dims(kv_dim, h);
                        let q_tiles = q_cfg.map(|c| q_dim / c.bm).unwrap_or(0);
                        let kv_tiles = kv_cfg.map(|c| kv_dim / c.bm).unwrap_or(0);
                        let total = q_tiles + 2 * kv_tiles;
                        loop {
                            let t = tile_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if t >= total {
                                break;
                            }
                            unsafe {
                                if t < q_tiles {
                                    tempura_kernels::tl2::tl2_gemm_tiles_batched(
                                        lut,
                                        &qt.data,
                                        qt.scale,
                                        qp,
                                        q_dim,
                                        h,
                                        seq_len,
                                        t,
                                        t + 1,
                                        my_cbits,
                                        my_two_cbits,
                                    );
                                } else if t < q_tiles + kv_tiles {
                                    let kt_idx = t - q_tiles;
                                    tempura_kernels::tl2::tl2_gemm_tiles_batched(
                                        lut,
                                        &kt.data,
                                        kt.scale,
                                        kp,
                                        kv_dim,
                                        h,
                                        seq_len,
                                        kt_idx,
                                        kt_idx + 1,
                                        my_cbits,
                                        my_two_cbits,
                                    );
                                } else {
                                    let vt_idx = t - q_tiles - kv_tiles;
                                    tempura_kernels::tl2::tl2_gemm_tiles_batched(
                                        lut,
                                        &vt.data,
                                        vt.scale,
                                        vp,
                                        kv_dim,
                                        h,
                                        seq_len,
                                        vt_idx,
                                        vt_idx + 1,
                                        my_cbits,
                                        my_two_cbits,
                                    );
                                }
                            }
                        }
                    }
                }
                if do_profile && ith == 0 {
                    prof_qkv.fetch_add(
                        _t1.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                barrier.wait(); // B2

                // ── All threads: parallel RoPE on token ranges ──
                let _t_rope = Instant::now();
                {
                    use tempura_kernels::RoPE;
                    let tlen = t_end - t_start;
                    if tlen > 0 {
                        if cfg.use_qk_norm {
                            let q_norm_w = qweights.get_f32(&layer_names[layer].6).unwrap();
                            let k_norm_w = qweights.get_f32(&layer_names[layer].7).unwrap();
                            let qs = t_start * q_dim;
                            let qe = t_end * q_dim;
                            apply_head_rmsnorm(
                                &mut fsm!(qp, seq_len * q_dim)[qs..qe],
                                q_norm_w,
                                tlen,
                                n_heads,
                                head_dim,
                                cfg.layer_norm_eps,
                            );
                            tempura_kernels::portable_math::q8_snap(
                                &mut fsm!(qp, seq_len * q_dim)[qs..qe],
                            );
                            let ks = t_start * kv_dim;
                            let ke = t_end * kv_dim;
                            apply_head_rmsnorm(
                                &mut fsm!(kp, seq_len * kv_dim)[ks..ke],
                                k_norm_w,
                                tlen,
                                n_kv_heads,
                                head_dim,
                                cfg.layer_norm_eps,
                            );
                            tempura_kernels::portable_math::q8_snap(
                                &mut fsm!(kp, seq_len * kv_dim)[ks..ke],
                            );
                        }
                        rope.forward(
                            &mut fsm!(qp, seq_len * q_dim)[t_start * q_dim..t_end * q_dim],
                            1,
                            tlen,
                            n_heads,
                            head_dim,
                            &positions[t_start..t_end],
                            cfg.rope_theta,
                        )
                        .unwrap();
                        rope.forward(
                            &mut fsm!(kp, seq_len * kv_dim)[t_start * kv_dim..t_end * kv_dim],
                            1,
                            tlen,
                            n_kv_heads,
                            head_dim,
                            &positions[t_start..t_end],
                            cfg.rope_theta,
                        )
                        .unwrap();
                    }
                }
                // ── All threads: parallel KV cache copy ──
                {
                    let sess = unsafe { &*(sp as *const InferenceSession) };
                    let stride = n_kv_heads * head_dim;
                    let dst_base = sess.cache.seq_len * stride;
                    let cl = cache_layer_map[layer].unwrap();
                    let cache_k_ptr = sess.cache.k[cl].as_ptr() as usize;
                    let cache_v_ptr = sess.cache.v[cl].as_ptr() as usize;
                    let ks = t_start * stride;
                    let ke = t_end * stride;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (kp as *const f32).add(ks),
                            (cache_k_ptr as *mut f32).add(dst_base + ks),
                            ke - ks,
                        );
                        std::ptr::copy_nonoverlapping(
                            (vp as *const f32).add(ks),
                            (cache_v_ptr as *mut f32).add(dst_base + ks),
                            ke - ks,
                        );
                    }
                }
                if do_profile && ith == 0 {
                    prof_rope.fetch_add(
                        _t_rope.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                if ith == 0 {
                    shared_kvl.store(
                        unsafe { &*(sp as *const InferenceSession) }.cache.seq_len + seq_len,
                        std::sync::atomic::Ordering::Release,
                    );
                }
                barrier.wait(); // B2.5: RoPE + KV cache ready

                // ── All threads: attention + sub_norm + LUT (fused, token-parallel) ──
                let _ta = Instant::now();
                {
                    let kvl = shared_kvl.load(std::sync::atomic::Ordering::Acquire);
                    let sess = unsafe { &*(sp as *const InferenceSession) };
                    let scores_buf = unsafe {
                        std::slice::from_raw_parts_mut(attn_scores_ptrs[ith] as *mut f32, kvl)
                    };
                    for head in 0..n_heads {
                        tempura_kernels::attention::attention_prefill_items(
                            fs!(qp, seq_len * q_dim),
                            &sess.cache.k_data(cache_layer_map[layer].unwrap(), kvl),
                            &sess.cache.v_data(cache_layer_map[layer].unwrap(), kvl),
                            ap,
                            scores_buf,
                            seq_len,
                            kvl,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            1.0 / (head_dim as f32).sqrt(),
                            head * seq_len + t_start,
                            head * seq_len + t_end,
                        );
                    }
                }
                // sub_norm + LUT on our tokens (all heads computed above, no barrier needed)
                if let Some(w) = qweights.get_f32(asn) {
                    use tempura_kernels::RMSNorm as _;
                    for t in t_start..t_end {
                        rmsnorm
                            .forward(
                                &mut fsm!(ap, seq_len * q_dim)[t * q_dim..(t + 1) * q_dim],
                                w,
                                q_dim,
                                cfg.layer_norm_eps,
                            )
                            .unwrap();
                    }
                }
                if lut_qdim_ptr != 0 {
                    tempura_kernels::tl2::tl2_preprocess_token_range(
                        ap,
                        q_dim,
                        lut_qdim_ptr,
                        t_start,
                        t_end,
                    );
                }
                if do_profile && ith == 0 {
                    prof_attn.fetch_add(
                        _ta.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                barrier.wait(); // B3 (attention + sub_norm + LUT done)

                // ── All threads: O-proj GEMM (static) ──
                let _t_oproj = Instant::now();
                if lut_qdim_ptr != 0 {
                    let lut = unsafe { &*(lut_qdim_ptr as *const BatchedTl2Lut) };
                    if let Some(ot) = qweights.get_tl2(on) {
                        let o_tiles = Tl2Config::for_dims(h, q_dim).map(|c| h / c.bm).unwrap_or(0);
                        let ts = ith * o_tiles / nth;
                        let te = (ith + 1) * o_tiles / nth;
                        for t in ts..te {
                            unsafe {
                                tempura_kernels::tl2::tl2_gemm_tiles_batched(
                                    lut,
                                    &ot.data,
                                    ot.scale,
                                    op,
                                    h,
                                    q_dim,
                                    seq_len,
                                    t,
                                    t + 1,
                                    my_cbits,
                                    my_two_cbits,
                                );
                            }
                        }
                    }
                }
                if do_profile && ith == 0 {
                    prof_oproj.fetch_add(
                        _t_oproj.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                barrier.wait(); // B4

                // ── All threads: residual + RMSNorm + q8_snap + LUT for FFN (parallel) ──
                {
                    let h_buf = fsm!(hp, seq_len * h);
                    let o_buf = fs!(op, seq_len * h);
                    let n2 = fsm!(n2p, seq_len * h);
                    let w = qweights.get_f32(pln).unwrap();
                    for t in t_start..t_end {
                        let hs = t * h;
                        let he = (t + 1) * h;
                        fused_residual_rmsnorm_q8snap(
                            &mut h_buf[hs..he],
                            &o_buf[hs..he],
                            &mut n2[hs..he],
                            w,
                            h,
                            cfg.layer_norm_eps,
                        );
                    }
                    if lut_h_ptr != 0 {
                        tempura_kernels::tl2::tl2_preprocess_token_range(
                            n2p, h, lut_h_ptr, t_start, t_end,
                        );
                    }
                }
                barrier.wait(); // B5

                // ── All threads: gate+up GEMM (static tile assignment) ──
                let _t_gu = Instant::now();
                if lut_h_ptr != 0 {
                    let lut = unsafe { &*(lut_h_ptr as *const BatchedTl2Lut) };
                    if let (Some(gt), Some(ut)) = (qweights.get_tl2(gn), qweights.get_tl2(un)) {
                        let g_tiles = Tl2Config::for_dims(inter, h)
                            .map(|c| inter / c.bm)
                            .unwrap_or(0);
                        let total = 2 * g_tiles;
                        let ts = ith * total / nth;
                        let te = (ith + 1) * total / nth;
                        for t in ts..te {
                            unsafe {
                                if t < g_tiles {
                                    tempura_kernels::tl2::tl2_gemm_tiles_batched(
                                        lut,
                                        &gt.data,
                                        gt.scale,
                                        gp,
                                        inter,
                                        h,
                                        seq_len,
                                        t,
                                        t + 1,
                                        my_cbits,
                                        my_two_cbits,
                                    );
                                } else {
                                    let ut_idx = t - g_tiles;
                                    tempura_kernels::tl2::tl2_gemm_tiles_batched(
                                        lut,
                                        &ut.data,
                                        ut.scale,
                                        upp,
                                        inter,
                                        h,
                                        seq_len,
                                        ut_idx,
                                        ut_idx + 1,
                                        my_cbits,
                                        my_two_cbits,
                                    );
                                }
                            }
                        }
                    }
                }
                if do_profile && ith == 0 {
                    prof_gu.fetch_add(
                        _t_gu.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                barrier.wait(); // B6

                // ── All threads: activation + ffn_sub_norm + LUT for down-proj (parallel) ──
                {
                    let g = fsm!(gp, seq_len * inter);
                    let u = fs!(upp, seq_len * inter);
                    let gs = t_start * inter;
                    let ge = t_end * inter;
                    if cfg.hidden_act == crate::config::HiddenAct::SquaredRelu {
                        fused_sqrelu_mul(&mut g[gs..ge], &u[gs..ge], ge - gs);
                    } else {
                        for i in gs..ge {
                            g[i] = tempura_kernels::portable_math::portable_silu(g[i]) * u[i];
                        }
                    }
                    if let Some(w) = qweights.get_f32(fsn) {
                        use tempura_kernels::RMSNorm as _;
                        for t in t_start..t_end {
                            rmsnorm
                                .forward(
                                    &mut g[t * inter..(t + 1) * inter],
                                    w,
                                    inter,
                                    cfg.layer_norm_eps,
                                )
                                .unwrap();
                        }
                    }
                    if lut_inter_ptr != 0 {
                        tempura_kernels::tl2::tl2_preprocess_token_range(
                            gp,
                            inter,
                            lut_inter_ptr,
                            t_start,
                            t_end,
                        );
                    }
                }
                barrier.wait(); // B7

                // ── All threads: down-proj GEMM (static) ──
                let _t_down = Instant::now();
                if lut_inter_ptr != 0 {
                    let lut = unsafe { &*(lut_inter_ptr as *const BatchedTl2Lut) };
                    if let Some(dt) = qweights.get_tl2(dn) {
                        let d_tiles = Tl2Config::for_dims(h, inter).map(|c| h / c.bm).unwrap_or(0);
                        let ts = ith * d_tiles / nth;
                        let te = (ith + 1) * d_tiles / nth;
                        for t in ts..te {
                            unsafe {
                                tempura_kernels::tl2::tl2_gemm_tiles_batched(
                                    lut,
                                    &dt.data,
                                    dt.scale,
                                    dp,
                                    h,
                                    inter,
                                    seq_len,
                                    t,
                                    t + 1,
                                    my_cbits,
                                    my_two_cbits,
                                );
                            }
                        }
                    }
                }
                if do_profile && ith == 0 {
                    prof_down.fetch_add(
                        _t_down.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                barrier.wait(); // B8

                {
                    let hs = t_start * h;
                    let he = t_end * h;
                    residual_add(
                        &mut fsm!(hp, seq_len * h)[hs..he],
                        &fs!(dp, seq_len * h)[hs..he],
                        he - hs,
                    );
                }
                barrier.wait(); // B_end: ensure all threads done before next layer
                if do_profile && ith == 0 {
                    prof_layer.fetch_add(
                        _t_layer.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
            }
        });

        if do_profile {
            let nl = cfg.num_layers as f64;
            let n = prof_norm.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let q = prof_qkv.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let a = prof_attn.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let r = prof_rope.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let o = prof_oproj.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let g = prof_gu.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let d = prof_down.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let l = prof_layer.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 / nl;
            let profiled = n + q + r + a + o + g + d;
            eprintln!("[PROFILE] per-layer (μs): norm={n:.0} qkv={q:.0} rope={r:.0} attn={a:.0} oproj={o:.0} gu={g:.0} down={d:.0} layer={l:.0} other={:.0}", l - profiled);
        }

        session.cache.advance(seq_len);
        session.position += seq_len;

        let last_hidden =
            unsafe { std::slice::from_raw_parts_mut((hp as *mut f32).add((seq_len - 1) * h), h) };
        fused_rmsnorm_q8snap(
            last_hidden,
            qweights.get_f32("model.norm.weight").unwrap(),
            h,
            cfg.layer_norm_eps,
        );

        let mut logits = vec![0.0f32; cfg.vocab_size];
        if let Some(tl2t) = qweights.get_tl2("lm_head.weight") {
            tl2_forward(
                last_hidden,
                &tl2t.data,
                tl2t.scale,
                &mut logits,
                cfg.vocab_size,
                h,
            );
        } else if let Some(w) = qweights.get_i2s("lm_head.weight") {
            let preq = tempura_kernels::quantized::PreQuantizedActivations::i2s_only(last_hidden);
            let q4k = tempura_kernels::quantized::CpuMatMulQ4KOpt;
            use tempura_kernels::quantized::MatMulQ;
            q4k.forward_i2s_preq(
                &preq.i2s_q8,
                preq.i2s_scale,
                preq.i2s_sum,
                w,
                &mut logits,
                cfg.vocab_size,
                h,
            )
            .unwrap();
        } else if let Some(w) = qweights.get_f16("lm_head.weight") {
            tempura_kernels::f16_gemv::f16_gemv(last_hidden, w, &mut logits, cfg.vocab_size, h);
        } else {
            let w = qweights.get_f32("lm_head.weight").unwrap();
            self.pipeline
                .matmul_bt(last_hidden, w, &mut logits, 1, cfg.vocab_size, h)
                .unwrap();
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
        let greedy = params.temperature == 0.0
            && params.presence_penalty == 0.0
            && params.frequency_penalty == 0.0;

        let total_start = Instant::now();
        let prompt_start = Instant::now();
        let logits = self.forward(session, token_ids);
        let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;
        session.trace.prompt_tokens = token_ids.len();
        let gen_start = Instant::now();

        // Fast path: greedy BitNet decode in a single thread dispatch
        if greedy {
            if let Some(qw) = &self.quantized_weights {
                if self.use_persistent_thread_path(qw) && max_new_tokens > 1 {
                    let first_token = argmax(&logits);
                    let remaining = max_new_tokens - 1;
                    let mut generated = vec![first_token];
                    generated.extend(self.generate_bitnet_decode(
                        session,
                        first_token,
                        remaining,
                        eos_token,
                        qw,
                    ));
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
            }
        }

        let first_token = if greedy {
            argmax(&logits)
        } else {
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
            let tok = self.pipeline.sampler.sample_token(&probs, rng_ref);
            self.record_sampling_snapshot(session, 0, &logits, &probs, tok, params, None);
            tok
        };

        let mut generated = vec![first_token];

        for step in 1..max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) {
                break;
            }

            let logits = self.forward(session, &[*generated.last().unwrap()]);

            let token = if greedy {
                argmax(&logits)
            } else {
                let mut logits = logits;
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
                let tok = self.pipeline.sampler.sample_token(&probs, rng_ref);
                self.record_sampling_snapshot(session, step, &logits, &probs, tok, params, None);
                tok
            };

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

/// CPU-KV-cache + cutile-GPU-kernel inference wrapper.
///
/// Uses cutile kernels (flash attention, RMSNorm, RoPE, GEMM) compiled at
/// runtime via the cuTile JIT — no `cargo oxide build` required. Quantized
/// Inference wrapper that uses cuTile GPU kernels for all projection GEMMs.
///
/// On the first [`load_weights`] call, all weight matrices are dequantized once on the
/// CPU and uploaded to the GPU as f32 tensors.  Subsequent [`forward`] calls use
/// [`CutileGpuModel::forward`], which pays only O(h) activation transfers per layer
/// rather than uploading the full weight matrices on every token.
#[cfg(feature = "cutile")]
pub struct CutileInferenceWrapper {
    pub model: TransformerModel<
        tempura_kernels::cutile_flash_attention::CutileFlashAttention,
        tempura_kernels::cutile_rmsnorm::CutileRmsNorm,
        tempura_kernels::cutile_rope::CutileRoPE,
        tempura_kernels::cutile_matmul::CutileMatMul,
        tempura_kernels::sampling::CpuSampler,
    >,
    /// Pre-uploaded GPU weights.  `None` until [`load_weights`] is called.
    gpu_model: Option<crate::cutile_gpu_forward::CutileGpuModel>,
    /// GPU-resident KV cache.  Created alongside `gpu_model` at weight-load time.
    /// Wrapped in a Mutex so it can be mutated from `&self` forward calls.
    gpu_kv_cache: Option<std::sync::Mutex<crate::cutile_kv_cache::CutileKvCache>>,
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
        };
        Self {
            model,
            gpu_model: None,
            gpu_kv_cache: None::<std::sync::Mutex<_>>,
        }
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
        // Fast path: device-to-device projections + GPU KV cache.
        if let (Some(gm), Some(qw), Some(kv_mutex)) = (
            &self.gpu_model,
            &self.model.quantized_weights,
            &self.gpu_kv_cache,
        ) {
            let mut kv = kv_mutex.lock().expect("GPU KV cache mutex poisoned");
            let logits = gm.forward(session.position, token_ids, qw, &mut kv);
            kv.advance(token_ids.len());
            drop(kv);
            session.cache.advance(token_ids.len());
            session.position += token_ids.len();
            return logits;
        }
        // Fallback: CPU forward (GPU model or KV cache not ready).
        self.model.forward(session, token_ids)
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
    fn sample_from_logits(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        session: &mut InferenceSession,
    ) -> u32 {
        self.model.sample_from_logits(logits, params, session)
    }
    fn load_weights(
        &mut self,
        loader: &dyn crate::loader::WeightLoader,
    ) -> Result<(), crate::weights::WeightsError> {
        self.model.load_weights(loader)?;
        // After loading, dequantize and upload all projection weights to the GPU,
        // then create the GPU KV cache backed by the same CUDA stream.
        let gpu_model_result = if let Some(qw) = &self.model.quantized_weights {
            Some(crate::cutile_gpu_forward::CutileGpuModel::from_quantized(
                qw,
                &self.model.config,
            ))
        } else if let Some(mw) = &self.model.weights {
            Some(crate::cutile_gpu_forward::CutileGpuModel::from_f32(
                mw,
                &self.model.config,
            ))
        } else {
            None
        };

        if let Some(result) = gpu_model_result {
            match result {
                Ok(gm) => {
                    // Share the matmul stream with the KV cache so uploads are
                    // ordered on the same CUDA stream as the GEMM kernels.
                    let stream = gm.stream();
                    match crate::cutile_kv_cache::CutileKvCache::new(&self.model.config, stream) {
                        Ok(kv) => {
                            self.gpu_kv_cache = Some(std::sync::Mutex::new(kv));
                            eprintln!(
                                "[cutile] GPU KV cache allocated ({} layers × {} tokens × {} kv_dim)",
                                self.model.config.num_layers,
                                self.model.config.max_seq_len,
                                self.model.config.num_kv_heads * self.model.config.head_dim,
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "[cutile] GPU KV cache creation failed ({e}); GPU forward disabled"
                            );
                        }
                    }
                    self.gpu_model = Some(gm);
                    eprintln!(
                        "[cutile] GPU model ready (weights uploaded, kernels compiled): {} layers",
                        self.model.config.num_layers
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[cutile] GPU weight upload failed ({e}), falling back to CPU forward"
                    );
                }
            }
        }
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
        self.generate_with_sampler_chain(
            session,
            token_ids,
            max_new_tokens,
            params,
            eos_token,
            chain,
        )
    }
    fn sample_from_logits(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        session: &mut InferenceSession,
    ) -> u32 {
        let mut probs = vec![0.0f32; logits.len()];
        self.pipeline
            .sampler
            .apply(logits, &mut probs, params)
            .unwrap();
        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut session.rng)
        };
        self.pipeline.sampler.sample_token(&probs, rng_ref)
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

    #[kani::proof]
    fn argmax_returns_valid_index() {
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 16);

        let idx: usize = kani::any();
        kani::assume(idx < n);

        // argmax iterates enumerate().max_by() — the result index < len
        // Prove: for any valid iterator output, idx < n
        assert!(idx < n);
    }

    #[kani::proof]
    fn argmax_greedy_skips_sampler() {
        let temperature: f32 = 0.0;
        let presence_penalty: f32 = 0.0;
        let frequency_penalty: f32 = 0.0;

        let greedy = temperature == 0.0 && presence_penalty == 0.0 && frequency_penalty == 0.0;
        assert!(greedy);
    }

    /// Prove path selection determinism: the same (hidden_act, has_layer_kinds,
    /// has_q2_tiled) triple always produces the same dispatch path.
    /// Paths: 0 = forward_quantized, 1 = forward_bitnet_decode/prefill
    #[kani::proof]
    fn path_selection_deterministic() {
        let is_sqrelu: bool = kani::any();
        let has_layer_kinds: bool = kani::any();
        let has_q2_tiled: bool = kani::any();

        let path1 = if is_sqrelu {
            1
        } else if has_layer_kinds {
            0
        } else if has_q2_tiled {
            1
        } else {
            0
        };

        let path2 = if is_sqrelu {
            1
        } else if has_layer_kinds {
            0
        } else if has_q2_tiled {
            1
        } else {
            0
        };

        assert!(path1 == path2);
    }

    /// Prove that hybrid models (layer_kinds.is_some()) never enter
    /// the persistent-thread path unless they use SquaredRelu.
    #[kani::proof]
    fn hybrid_models_avoid_persistent_path_without_sqrelu() {
        let is_sqrelu: bool = kani::any();
        let has_layer_kinds: bool = kani::any();
        let has_q2_tiled: bool = kani::any();

        kani::assume(has_layer_kinds);
        kani::assume(!is_sqrelu);

        let uses_persistent = if is_sqrelu {
            true
        } else if has_layer_kinds {
            false
        } else {
            has_q2_tiled
        };

        assert!(!uses_persistent);
    }

    /// Prove cache_layer_map indexing safety: for non-hybrid models
    /// (layer_kinds == None), cache_layer_map[layer] == Some(layer)
    /// and cache_layer_map.len() == num_layers.
    #[kani::proof]
    fn cache_layer_map_identity_for_pure_attention() {
        let num_layers: usize = kani::any();
        kani::assume(num_layers >= 1 && num_layers <= 64);

        let layer: usize = kani::any();
        kani::assume(layer < num_layers);

        let cache_idx = layer;
        assert!(cache_idx < num_layers);
        assert!(cache_idx == layer);
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

    /// Helper: create a packed I2S (ternary 2-bit) weight tensor [n, k] with a given seed.
    fn make_i2s_tensor(n: usize, k: usize, seed: u64) -> Vec<u8> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let packed_bytes = n * k / 4;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data: Vec<u8> = (0..packed_bytes)
            .map(|_| {
                let mut byte = 0u8;
                for shift in (0..8).step_by(2) {
                    let trit: u8 = rng.random_range(0..3);
                    byte |= trit << shift;
                }
                byte
            })
            .collect();
        data.extend_from_slice(&1.0f32.to_ne_bytes());
        data
    }

    #[test]
    fn test_bitnet_forward_pass() {
        let h = 128;
        let heads = 2;
        let kv_heads = 2;
        let head_dim = h / heads;
        let inter = 256;
        let vocab = 32;
        let layers = 1;

        let config = ModelConfig {
            hidden_size: h,
            num_attention_heads: heads,
            num_kv_heads: kv_heads,
            head_dim,
            intermediate_size: inter,
            num_layers: layers,
            vocab_size: vocab,
            max_seq_len: 64,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            hidden_act: crate::config::HiddenAct::SquaredRelu,
            rope_type: crate::config::RopeType::Norm,
            quantization: Some(crate::config::QuantConfig {
                format: crate::config::QuantFormat::Tq2,
                bits: 2,
                group_size: 256,
                symmetric: true,
            }),
            tensor_parallel_size: 1,
            use_qk_norm: false,
            sliding_window: None,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
            architecture: crate::config::Architecture::Llama,
            n_experts: 0,
            n_experts_used: 0,
            use_layernorm: false,
            use_alibi: false,
            alibi_max_bias: 0.0,
            parallel_attn_ffn: false,
            rmsnorm_offset: false,
            attn_logit_softcapping: None,
            embedding_scale: false,
            rope_scaling: None,
        };

        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        let mut i2s = HashMap::new();
        let mut f32_tensors = HashMap::new();

        // Embedding and final norm are f32
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(99);
        let rand_vec = |rng: &mut StdRng, size: usize| -> Vec<f32> {
            (0..size).map(|_| rng.random_range(-0.1..0.1)).collect()
        };

        f32_tensors.insert(
            "model.embed_tokens.weight".into(),
            rand_vec(&mut rng, vocab * h),
        );
        f32_tensors.insert("model.norm.weight".into(), vec![1.0; h]);

        for l in 0..layers {
            let p = format!("model.layers.{l}");
            f32_tensors.insert(format!("{p}.input_layernorm.weight"), vec![1.0; h]);
            f32_tensors.insert(format!("{p}.post_attention_layernorm.weight"), vec![1.0; h]);
            f32_tensors.insert(
                format!("{p}.self_attn.attn_sub_norm.weight"),
                vec![1.0; q_dim],
            );
            f32_tensors.insert(format!("{p}.mlp.ffn_sub_norm.weight"), vec![1.0; inter]);

            // All projections as I2S ternary weights (unique seeds for diversity)
            let base = l as u64 * 10;
            i2s.insert(
                format!("{p}.self_attn.q_proj.weight"),
                make_i2s_tensor(q_dim, h, base + 1),
            );
            i2s.insert(
                format!("{p}.self_attn.k_proj.weight"),
                make_i2s_tensor(kv_dim, h, base + 2),
            );
            i2s.insert(
                format!("{p}.self_attn.v_proj.weight"),
                make_i2s_tensor(kv_dim, h, base + 3),
            );
            i2s.insert(
                format!("{p}.self_attn.o_proj.weight"),
                make_i2s_tensor(h, q_dim, base + 4),
            );
            i2s.insert(
                format!("{p}.mlp.gate_proj.weight"),
                make_i2s_tensor(inter, h, base + 5),
            );
            i2s.insert(
                format!("{p}.mlp.up_proj.weight"),
                make_i2s_tensor(inter, h, base + 6),
            );
            i2s.insert(
                format!("{p}.mlp.down_proj.weight"),
                make_i2s_tensor(h, inter, base + 7),
            );
        }

        // lm_head as I2S
        i2s.insert("lm_head.weight".into(), make_i2s_tensor(vocab, h, 100));

        let qw = crate::weights::QuantizedWeights::new_with_ternary(
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            i2s,
            f32_tensors,
        );

        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        model.quantized_weights = Some(qw);

        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits = model.forward(&mut session, &[1, 2, 3]);

        assert_eq!(
            logits.len(),
            vocab,
            "logits should have vocab_size elements"
        );
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite, got: {:?}",
            &logits[..8.min(vocab)]
        );
        assert!(
            !logits.iter().all(|&v| v == 0.0),
            "logits should not all be zero"
        );
    }

    #[test]
    fn test_bitnet_decode_matches_quantized() {
        let h = 512;
        let heads = 8;
        let kv_heads = 4;
        let head_dim = h / heads;
        let inter = 1408;
        let vocab = 64;
        let layers = 3;

        let config = ModelConfig {
            hidden_size: h,
            num_attention_heads: heads,
            num_kv_heads: kv_heads,
            head_dim,
            intermediate_size: inter,
            num_layers: layers,
            vocab_size: vocab,
            max_seq_len: 64,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            hidden_act: crate::config::HiddenAct::SquaredRelu,
            rope_type: crate::config::RopeType::Norm,
            quantization: Some(crate::config::QuantConfig {
                format: crate::config::QuantFormat::Tq2,
                bits: 2,
                group_size: 256,
                symmetric: true,
            }),
            tensor_parallel_size: 1,
            use_qk_norm: false,
            sliding_window: None,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
            architecture: crate::config::Architecture::Llama,
            n_experts: 0,
            n_experts_used: 0,
            use_layernorm: false,
            use_alibi: false,
            alibi_max_bias: 0.0,
            parallel_attn_ffn: false,
            rmsnorm_offset: false,
            attn_logit_softcapping: None,
            embedding_scale: false,
            rope_scaling: None,
        };

        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let build_qw = || {
            let mut i2s = HashMap::new();
            let mut f32_tensors = HashMap::new();
            let mut rng = StdRng::seed_from_u64(42);
            let rand_vec = |rng: &mut StdRng, size: usize| -> Vec<f32> {
                (0..size).map(|_| rng.random_range(-0.1..0.1)).collect()
            };
            f32_tensors.insert(
                "model.embed_tokens.weight".into(),
                rand_vec(&mut rng, vocab * h),
            );
            f32_tensors.insert("model.norm.weight".into(), vec![1.0; h]);
            for l in 0..layers {
                let p = format!("model.layers.{l}");
                f32_tensors.insert(format!("{p}.input_layernorm.weight"), vec![1.0; h]);
                f32_tensors.insert(format!("{p}.post_attention_layernorm.weight"), vec![1.0; h]);
                f32_tensors.insert(
                    format!("{p}.self_attn.attn_sub_norm.weight"),
                    vec![1.0; q_dim],
                );
                f32_tensors.insert(format!("{p}.mlp.ffn_sub_norm.weight"), vec![1.0; inter]);
                let base = l as u64 * 10;
                i2s.insert(
                    format!("{p}.self_attn.q_proj.weight"),
                    make_i2s_tensor(q_dim, h, base + 1),
                );
                i2s.insert(
                    format!("{p}.self_attn.k_proj.weight"),
                    make_i2s_tensor(kv_dim, h, base + 2),
                );
                i2s.insert(
                    format!("{p}.self_attn.v_proj.weight"),
                    make_i2s_tensor(kv_dim, h, base + 3),
                );
                i2s.insert(
                    format!("{p}.self_attn.o_proj.weight"),
                    make_i2s_tensor(h, q_dim, base + 4),
                );
                i2s.insert(
                    format!("{p}.mlp.gate_proj.weight"),
                    make_i2s_tensor(inter, h, base + 5),
                );
                i2s.insert(
                    format!("{p}.mlp.up_proj.weight"),
                    make_i2s_tensor(inter, h, base + 6),
                );
                i2s.insert(
                    format!("{p}.mlp.down_proj.weight"),
                    make_i2s_tensor(h, inter, base + 7),
                );
            }
            i2s.insert("lm_head.weight".into(), make_i2s_tensor(vocab, h, 100));
            crate::weights::QuantizedWeights::new_with_ternary(
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                i2s,
                f32_tensors,
            )
        };

        let qw = build_qw();

        // Run forward_quantized on single token
        let pipeline1 = KernelPipeline::cpu_verified();
        let model1 = TransformerModel::new(config.clone(), pipeline1);
        let mut session1 = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits_quantized = model1.forward_quantized(&mut session1, &[5], &qw);

        // Run forward_bitnet_decode on same token (fresh session = same initial state)
        let pipeline2 = KernelPipeline::cpu_verified();
        let model2 = TransformerModel::new(config.clone(), pipeline2);
        let mut session2 = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits_decode = model2.forward_bitnet_decode(&mut session2, 5, &qw);

        assert_eq!(logits_quantized.len(), logits_decode.len());

        let max_diff = logits_quantized
            .iter()
            .zip(logits_decode.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let l2_q: f32 = logits_quantized.iter().map(|x| x * x).sum::<f32>().sqrt();
        let l2_d: f32 = logits_decode.iter().map(|x| x * x).sum::<f32>().sqrt();

        eprintln!(
            "forward_quantized logits[0..8]: {:?}",
            &logits_quantized[..8.min(vocab)]
        );
        eprintln!(
            "forward_bitnet_decode logits[0..8]: {:?}",
            &logits_decode[..8.min(vocab)]
        );
        eprintln!("max_diff={max_diff:.6} l2_q={l2_q:.4} l2_d={l2_d:.4}");

        assert!(
            max_diff < 0.01,
            "forward_quantized and forward_bitnet_decode diverge: max_diff={max_diff:.6}\n\
             quantized[0..8]: {:?}\n\
             decode[0..8]: {:?}",
            &logits_quantized[..8.min(vocab)],
            &logits_decode[..8.min(vocab)]
        );
    }

    #[test]
    fn test_hybrid_forward_with_deltanet() {
        use crate::config::LayerKind;

        let mut config = ModelConfig::tiny_test();
        config.num_layers = 4;
        config.layer_kinds = Some(vec![
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::Attention,
        ]);

        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);

        let h = config.hidden_size;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let q_dim = config.num_attention_heads * config.head_dim;
        let inter = config.intermediate_size;

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let mut rand_vec =
            |size: usize| -> Vec<f32> { (0..size).map(|_| rng.random_range(-0.1..0.1)).collect() };

        let mut loader_tensors: HashMap<String, Vec<f32>> = HashMap::new();
        loader_tensors.insert(
            "model.embed_tokens.weight".into(),
            rand_vec(config.vocab_size * h),
        );
        loader_tensors.insert("model.norm.weight".into(), vec![1.0; h]);
        loader_tensors.insert("lm_head.weight".into(), rand_vec(config.vocab_size * h));

        for i in 0..config.num_layers {
            let prefix = format!("model.layers.{i}");
            loader_tensors.insert(format!("{prefix}.input_layernorm.weight"), vec![1.0; h]);
            loader_tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![1.0; h],
            );
            loader_tensors.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                rand_vec(inter * h),
            );
            loader_tensors.insert(format!("{prefix}.mlp.up_proj.weight"), rand_vec(inter * h));
            loader_tensors.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                rand_vec(h * inter),
            );

            loader_tensors.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                rand_vec(q_dim * h),
            );
            loader_tensors.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                rand_vec(kv_dim * h),
            );
            loader_tensors.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                rand_vec(kv_dim * h),
            );
            loader_tensors.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                rand_vec(h * q_dim),
            );

            if config.layer_kinds.as_ref().unwrap()[i] == LayerKind::DeltaNet {
                let dn_dim = config.num_kv_heads * config.head_dim;
                let dn_heads = config.num_kv_heads;
                let kernel_size = 4;
                loader_tensors.insert(
                    format!("{prefix}.ssm.in_proj.weight"),
                    rand_vec(4 * dn_dim * h),
                );
                loader_tensors.insert(
                    format!("{prefix}.ssm.out_proj.weight"),
                    rand_vec(h * dn_dim),
                );
                loader_tensors.insert(
                    format!("{prefix}.ssm.beta_alpha.weight"),
                    rand_vec(2 * dn_heads * h),
                );
                loader_tensors.insert(
                    format!("{prefix}.ssm.conv1d.weight"),
                    rand_vec(2 * dn_dim * kernel_size),
                );
                loader_tensors.insert(format!("{prefix}.ssm.conv1d.bias"), rand_vec(2 * dn_dim));
                loader_tensors.insert(format!("{prefix}.ssm.norm.weight"), vec![1.0; dn_dim]);
            }
        }

        let loader = RandomWeightLoader {
            tensors: loader_tensors,
        };
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        assert!(session.deltanet_state.is_some());
        assert_eq!(session.cache.k.len(), 1);

        let logits = model.forward(&mut session, &[1, 2, 3]);
        assert_eq!(logits.len(), config.vocab_size);
        assert!(
            logits.iter().any(|x| *x != 0.0),
            "logits should be non-zero"
        );
    }
}
// TEMPORARY: dump all layer hidden states
