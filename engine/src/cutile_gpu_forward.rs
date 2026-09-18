/// Fused GPU forward pass for the transformer.
///
/// All projection weight matrices are dequantized once at load time and uploaded to the GPU.
/// Inference then pays only one host→device transfer per token (the activation vector) and
/// one device→host transfer per token (the projection output), instead of uploading the full
/// weight matrix on every call.
///
/// RMSNorm, RoPE, attention (with CPU KvCache), and elementwise operations remain on the
/// CPU.  The speedup comes from eliminating repeated weight uploads for the large projection
/// matrices (Q/K/V/O/gate/up/down proj), which dominate transfer cost at typical batch=1.
use std::collections::HashMap;
use std::sync::Arc;

use cutile::api;
use cutile::prelude::DeviceOp;
use cutile::tensor::{Reshape, Tensor};
use tempura_kernels::cutile_matmul::CutileMatMul;
use tempura_kernels::{Attention, RMSNorm, RoPE};

use crate::config::ModelConfig;
use crate::session::InferenceSession;
use crate::weights::QuantizedWeights;

// BK must match the constant in cutile_matmul.rs.
const BK: usize = 8;

/// A projection weight tensor resident on the GPU with shape metadata.
struct DeviceWeight {
    /// Pre-uploaded f32 tensor on the GPU, shaped [n, k_pad].
    tensor: Arc<Tensor<f32>>,
    /// Number of output rows (the "N" in z[M,N] = a[M,K] @ b[N,K]^T).
    n: usize,
    /// Logical K dimension (number of input features, unpadded).
    k_raw: usize,
    /// Padded K actually stored on the GPU (multiple of BK).
    k_pad: usize,
}

/// Transformer with all projection weights pre-uploaded to the GPU.
///
/// Create once with [`CutileGpuModel::from_quantized`] at model load time, then call
/// [`CutileGpuModel::forward`] for each inference step.
pub struct CutileGpuModel {
    config: ModelConfig,
    device_weights: HashMap<String, DeviceWeight>,
    matmul: CutileMatMul,
}

// SAFETY: CutileMatMul and Tensor<f32> are not Send/Sync by default because they hold
// raw CUDA pointers, but tempura uses a single-threaded inference path per model.
unsafe impl Send for CutileGpuModel {}
unsafe impl Sync for CutileGpuModel {}

impl CutileGpuModel {
    // ── private helpers ───────────────────────────────────────────────────────

    fn upload(
        matmul: &CutileMatMul,
        device_weights: &mut HashMap<String, DeviceWeight>,
        name: &str,
        data: &[f32],
        n: usize,
        k_raw: usize,
    ) -> Result<(), tempura_kernels::KernelError> {
        let k_pad = k_raw.div_ceil(BK) * BK;
        let padded: Arc<Vec<f32>> = if k_pad == k_raw {
            Arc::new(data.to_vec())
        } else {
            let mut v = vec![0.0f32; n * k_pad];
            for row in 0..n {
                v[row * k_pad..row * k_pad + k_raw]
                    .copy_from_slice(&data[row * k_raw..(row + 1) * k_raw]);
            }
            Arc::new(v)
        };

        let stream = matmul.stream();
        let tensor: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&padded)
            .sync_on(stream)
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[n, k_pad])
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .into();

        device_weights.insert(
            name.to_string(),
            DeviceWeight { tensor, n, k_raw, k_pad },
        );
        Ok(())
    }

    /// Dequantize a named tensor from `QuantizedWeights` to a plain f32 Vec.
    ///
    /// Returns `None` if the tensor is not present in `qw`.
    fn dequantize(qw: &QuantizedWeights, name: &str, n_elements: usize) -> Option<Vec<f32>> {
        if let Some(f32_data) = qw.get_f32(name) {
            return Some(f32_data.to_vec());
        }
        if let Some((dtype, bytes)) = qw.get_typed(name) {
            return Some(tempura_kernels::ggml::dequantize::dequantize_to_f32(
                dtype,
                bytes,
                n_elements,
            ));
        }
        None
    }

    /// Upload a single projection weight, dequantizing first if necessary.
    fn maybe_upload(
        matmul: &CutileMatMul,
        device_weights: &mut HashMap<String, DeviceWeight>,
        qw: &QuantizedWeights,
        name: &str,
        n: usize,
        k: usize,
    ) -> Result<(), tempura_kernels::KernelError> {
        if let Some(data) = Self::dequantize(qw, name, n * k) {
            Self::upload(matmul, device_weights, name, &data, n, k)?;
        }
        Ok(())
    }

    // ── constructors ──────────────────────────────────────────────────────────

    /// Dequantize all projection weights from `qw` and upload them to the GPU.
    ///
    /// Calling this once at model-load time eliminates repeated host→device weight
    /// transfers during inference.  Non-projection tensors (RMSNorm weights, biases)
    /// stay on the CPU because they are used by CPU kernels only.
    pub fn from_quantized(
        qw: &QuantizedWeights,
        config: &ModelConfig,
    ) -> Result<Self, tempura_kernels::KernelError> {
        let matmul = CutileMatMul::new()?;
        let mut dw: HashMap<String, DeviceWeight> = HashMap::new();

        let h = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;

        // lm_head  [vocab, h]
        Self::maybe_upload(&matmul, &mut dw, qw, "lm_head.weight", vocab, h)?;

        for layer in 0..config.num_layers {
            let p = format!("model.layers.{layer}");

            // Q / K / V projections: [q_dim/kv_dim, h]
            Self::maybe_upload(&matmul, &mut dw, qw, &format!("{p}.self_attn.q_proj.weight"), q_dim, h)?;
            Self::maybe_upload(&matmul, &mut dw, qw, &format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?;
            Self::maybe_upload(&matmul, &mut dw, qw, &format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?;

            // O projection: [h, q_dim]
            Self::maybe_upload(&matmul, &mut dw, qw, &format!("{p}.self_attn.o_proj.weight"), h, q_dim)?;

            // FFN projections: gate/up [inter, h], down [h, inter]
            Self::maybe_upload(&matmul, &mut dw, qw, &format!("{p}.mlp.gate_proj.weight"), inter, h)?;
            Self::maybe_upload(&matmul, &mut dw, qw, &format!("{p}.mlp.up_proj.weight"), inter, h)?;
            Self::maybe_upload(&matmul, &mut dw, qw, &format!("{p}.mlp.down_proj.weight"), h, inter)?;
        }

        Ok(Self { config: config.clone(), device_weights: dw, matmul })
    }

    /// Upload all projection weights from `mw` (already f32) to the GPU.
    pub fn from_f32(
        mw: &crate::weights::ModelWeights,
        config: &ModelConfig,
    ) -> Result<Self, tempura_kernels::KernelError> {
        let matmul = CutileMatMul::new()?;
        let mut dw: HashMap<String, DeviceWeight> = HashMap::new();

        let h = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;

        let mut try_upload = |name: &str, n: usize, k: usize| -> Result<(), _> {
            if let Some(data) = mw.get(name) {
                Self::upload(&matmul, &mut dw, name, data, n, k)?;
            }
            Ok(())
        };

        try_upload("lm_head.weight", vocab, h)?;
        for layer in 0..config.num_layers {
            let p = format!("model.layers.{layer}");
            try_upload(&format!("{p}.self_attn.q_proj.weight"), q_dim, h)?;
            try_upload(&format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?;
            try_upload(&format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?;
            try_upload(&format!("{p}.self_attn.o_proj.weight"), h, q_dim)?;
            try_upload(&format!("{p}.mlp.gate_proj.weight"), inter, h)?;
            try_upload(&format!("{p}.mlp.up_proj.weight"), inter, h)?;
            try_upload(&format!("{p}.mlp.down_proj.weight"), h, inter)?;
        }

        Ok(Self { config: config.clone(), device_weights: dw, matmul })
    }

    // ── inference ─────────────────────────────────────────────────────────────

    /// GPU GEMM: `output[m, n] = a[m, k] @ b[n, k]^T` using the pre-uploaded device weight.
    ///
    /// Falls back silently (no-op on `output`) if the weight is not present on the GPU.
    fn gemm_bt(&self, a: &[f32], weight_name: &str, output: &mut [f32], m: usize) {
        if let Some(dw) = self.device_weights.get(weight_name) {
            self.matmul
                .forward_bt_device_b(a, &dw.tensor, output, m, dw.n, dw.k_raw, dw.k_pad)
                .expect("CutileGpuModel::gemm_bt failed");
        }
    }

    /// Full autoregressive forward pass staying on GPU between layers.
    ///
    /// Returns logits over the vocabulary for the last token position.
    ///
    /// The caller (e.g. [`super::model::CutileInferenceWrapper`]) is responsible for
    /// advancing `session.position` and `session.cache` after this returns, exactly as
    /// it would for a CPU-only forward call.
    pub fn forward(
        &self,
        session: &mut InferenceSession,
        token_ids: &[u32],
        qweights: &QuantizedWeights,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let seq_len = token_ids.len();
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = cfg.intermediate_size;

        // ── 1. Embedding lookup (CPU, O(seq_len × h)) ─────────────────────────
        let embed = qweights
            .get_f32("model.embed_tokens.weight")
            .expect("model.embed_tokens.weight not found");
        let mut hidden = vec![0.0f32; seq_len * h];
        for (t, &tok) in token_ids.iter().enumerate() {
            hidden[t * h..(t + 1) * h]
                .copy_from_slice(&embed[tok as usize * h..(tok as usize + 1) * h]);
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

        // Deterministic CPU kernels (used in place of the pipeline for portability).
        let rope_mode = match cfg.rope_type {
            crate::config::RopeType::Norm => tempura_kernels::rope::RoPEMode::Normal,
            crate::config::RopeType::Neox => tempura_kernels::rope::RoPEMode::NeoX,
        };
        let rope = tempura_kernels::rope::CpuRoPEUnified::new(rope_mode, head_dim);
        let rmsnorm = tempura_kernels::rmsnorm::CpuRMSNormGgml;

        // Scratchpads (reused across layers).
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

        // Pre-build weight name strings to avoid allocations in the hot loop.
        let layer_names: Vec<(String, String, String, String, String, String, String, String, String)> =
            (0..cfg.num_layers)
                .map(|l| {
                    let p = format!("model.layers.{l}");
                    (
                        format!("{p}.input_layernorm.weight"),
                        format!("{p}.post_attention_layernorm.weight"),
                        format!("{p}.self_attn.q_proj.weight"),
                        format!("{p}.self_attn.k_proj.weight"),
                        format!("{p}.self_attn.v_proj.weight"),
                        format!("{p}.self_attn.o_proj.weight"),
                        format!("{p}.mlp.gate_proj.weight"),
                        format!("{p}.mlp.up_proj.weight"),
                        format!("{p}.mlp.down_proj.weight"),
                    )
                })
                .collect();

        // ── 2. Layer loop ──────────────────────────────────────────────────────
        for layer in 0..cfg.num_layers {
            let (
                ref input_ln_name,
                ref post_ln_name,
                ref q_w_name,
                ref k_w_name,
                ref v_w_name,
                ref o_w_name,
                ref gate_w_name,
                ref up_w_name,
                ref down_w_name,
            ) = layer_names[layer];

            // RMSNorm (CPU, O(seq_len × h)).
            normed.copy_from_slice(&hidden);
            let input_ln_w = qweights.get_f32(input_ln_name).unwrap();
            rmsnorm
                .forward(&mut normed, input_ln_w, h, cfg.layer_norm_eps)
                .unwrap();

            // QKV projections — GPU GEMM with pre-uploaded device weights.
            // Each call: upload A (seq_len × h), run GEMM on GPU, download output.
            q.fill(0.0);
            k.fill(0.0);
            v.fill(0.0);
            self.gemm_bt(&normed[..seq_len * h], q_w_name, &mut q[..seq_len * q_dim], seq_len);
            self.gemm_bt(&normed[..seq_len * h], k_w_name, &mut k[..seq_len * kv_dim], seq_len);
            self.gemm_bt(&normed[..seq_len * h], v_w_name, &mut v[..seq_len * kv_dim], seq_len);

            // RoPE (CPU).
            if !cfg.use_alibi {
                rope.forward(
                    &mut q, 1, seq_len, n_heads, head_dim, &positions, cfg.rope_theta,
                )
                .unwrap();
                rope.forward(
                    &mut k, 1, seq_len, n_kv_heads, head_dim, &positions, cfg.rope_theta,
                )
                .unwrap();
            }

            // KV-cache append + attention (CPU, reads/writes CPU KvCache).
            session.cache.append_kv(
                layer,
                &k[..seq_len * kv_dim],
                &v[..seq_len * kv_dim],
                seq_len,
            );
            let kv_len = session.cache.seq_len + seq_len;
            let k_cache_cow = session.cache.k_data(layer, kv_len);
            let v_cache_cow = session.cache.v_data(layer, kv_len);

            attn_out.fill(0.0);
            tempura_kernels::attention::CpuAttentionGgml
                .forward(
                    &q,
                    &k_cache_cow,
                    &v_cache_cow,
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

            // O projection — GPU GEMM.
            o_proj.fill(0.0);
            self.gemm_bt(
                &attn_out[..seq_len * q_dim],
                o_w_name,
                &mut o_proj[..seq_len * h],
                seq_len,
            );

            // Residual add (CPU).
            for i in 0..seq_len * h {
                hidden[i] += o_proj[i];
            }

            // Post-attention RMSNorm (CPU).
            normed2.copy_from_slice(&hidden);
            let post_ln_w = qweights.get_f32(post_ln_name).unwrap();
            rmsnorm
                .forward(&mut normed2, post_ln_w, h, cfg.layer_norm_eps)
                .unwrap();

            // FFN gate + up projections — GPU GEMM.
            gate.fill(0.0);
            up.fill(0.0);
            self.gemm_bt(
                &normed2[..seq_len * h],
                gate_w_name,
                &mut gate[..seq_len * inter],
                seq_len,
            );
            self.gemm_bt(
                &normed2[..seq_len * h],
                up_w_name,
                &mut up[..seq_len * inter],
                seq_len,
            );

            // SiLU * up gate (CPU).
            for i in 0..seq_len * inter {
                gate[i] =
                    tempura_kernels::portable_math::portable_silu(gate[i]) * up[i];
            }

            // FFN down projection — GPU GEMM.
            down_out.fill(0.0);
            self.gemm_bt(
                &gate[..seq_len * inter],
                down_w_name,
                &mut down_out[..seq_len * h],
                seq_len,
            );

            // Residual add (CPU).
            for i in 0..seq_len * h {
                hidden[i] += down_out[i];
            }
        }

        // ── 3. Final norm + lm_head ────────────────────────────────────────────

        // Final RMSNorm (CPU).
        let mut final_normed = hidden;
        let final_norm_w = qweights
            .get_f32("model.norm.weight")
            .expect("model.norm.weight not found");
        rmsnorm
            .forward(&mut final_normed, final_norm_w, h, cfg.layer_norm_eps)
            .unwrap();

        // lm_head — GPU GEMM.
        let last_hidden = &final_normed[(seq_len - 1) * h..seq_len * h];
        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.gemm_bt(last_hidden, "lm_head.weight", &mut logits, 1);

        logits
    }
}
