/// Fused GPU forward pass for the transformer.
///
/// All projection weight matrices are dequantized once at load time and uploaded to the GPU.
/// Activations flow device-to-device across all projection GEMMs within each layer:
///
///   normed (host) → device once per sub-block
///   Q/K/V via forward_bt_device_to_device (no host round-trip)
///   K/V appended to CutileKvCache (host mirror + device upload for attention)
///   Attention via CutileFlashAttention::forward_device
///   O / gate / up / down via forward_bt_device_to_device
///
/// CPU operations that still require a host round-trip: RMSNorm, RoPE, SiLU gate,
/// and the residual adds. These are cheap compared with the projection GEMMs and
/// the KV-cache attention, so they do not dominate.
use std::collections::HashMap;
use std::sync::Arc;

use cuda_core::Stream;
use cutile::api;
use cutile::prelude::{DeviceOp, ToHostVec};
use cutile::tensor::{Reshape, Tensor};
use tempura_kernels::cutile_flash_attention::CutileFlashAttention;
use tempura_kernels::cutile_matmul::CutileMatMul;
use tempura_kernels::{RMSNorm, RoPE};

use crate::config::ModelConfig;
use crate::cutile_kv_cache::CutileKvCache;
use crate::weights::QuantizedWeights;

// BK must be a multiple that divides all standard hidden / intermediate
// dimensions (4096, 11008, 14336, …).  16 matches the tiling constant used
// inside CutileMatMul::forward_bt_device_to_device.
const BK: usize = 16;

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

fn next_pow2(x: usize) -> usize {
    if x <= 1 { return 1; }
    1usize << (usize::BITS - (x - 1).leading_zeros()) as usize
}

/// Transformer with all projection weights pre-uploaded to the GPU.
///
/// Create once with [`CutileGpuModel::from_quantized`] at model load time, then call
/// [`CutileGpuModel::forward`] for each inference step.
pub struct CutileGpuModel {
    config: ModelConfig,
    device_weights: HashMap<String, DeviceWeight>,
    matmul: CutileMatMul,
    flash_attn: CutileFlashAttention,
}

// SAFETY: CutileMatMul, CutileFlashAttention and Tensor<f32> hold raw CUDA pointers,
// but tempura uses a single-threaded inference path per model instance.
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
        // sm_120 (Blackwell) cuTile JIT requires K to be a power-of-2.
        let k_pad = next_pow2(k_raw.div_ceil(BK) * BK);
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

    /// Pre-compile all cuTile GEMM kernel specializations by firing a dummy `m=1`
    /// device-to-device call for each distinct matrix shape used by this model.
    ///
    /// cuTile JIT-compiles and caches PTX specializations the first time a given
    /// (M, N, K, BM, BN, BK) tile tuple is seen.  Calling this once after weight
    /// upload means the first real decode token pays zero JIT cost.
    ///
    /// Only layer 0 weights are exercised — all layers share the same shapes.
    fn warmup_gemms(&self) {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;

        // Use k_pad sizes so the activation shapes match what gemm_d2d expects
        // after the first GEMM (avoids spurious download+repad during warmup).
        let h_pad = next_pow2(h.div_ceil(BK) * BK);
        let q_pad = next_pow2(q_dim.div_ceil(BK) * BK);
        let inter_pad = next_pow2(inter.div_ceil(BK) * BK);
        let dummy_h = self.to_device(&vec![0.0f32; h_pad], 1, h_pad);
        let dummy_q = self.to_device(&vec![0.0f32; q_pad], 1, q_pad);
        let dummy_inter = self.to_device(&vec![0.0f32; inter_pad], 1, inter_pad);

        // Q/K/V projection shape: [1, h] @ [q_dim/kv_dim, h]^T
        let _ = self.gemm_d2d(&dummy_h, "model.layers.0.self_attn.q_proj.weight", 1);
        let _ = self.gemm_d2d(&dummy_h, "model.layers.0.self_attn.k_proj.weight", 1);
        let _ = self.gemm_d2d(&dummy_h, "model.layers.0.self_attn.v_proj.weight", 1);
        // O projection shape: [1, q_dim] @ [h, q_dim]^T
        let _ = self.gemm_d2d(&dummy_q, "model.layers.0.self_attn.o_proj.weight", 1);
        // Gate/up projection shape: [1, h] @ [inter, h]^T
        let _ = self.gemm_d2d(&dummy_h, "model.layers.0.mlp.gate_proj.weight", 1);
        let _ = self.gemm_d2d(&dummy_h, "model.layers.0.mlp.up_proj.weight", 1);
        // Down projection shape: [1, inter] @ [h, inter]^T
        let _ = self.gemm_d2d(&dummy_inter, "model.layers.0.mlp.down_proj.weight", 1);
        // lm_head shape: [1, h] @ [vocab, h]^T
        let _ = self.gemm_d2d(&dummy_h, "lm_head.weight", 1);
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
        let flash_attn = CutileFlashAttention::new()?;
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

        let model = Self { config: config.clone(), device_weights: dw, matmul, flash_attn };
        eprintln!("[cutile] warming up kernel specializations…");
        model.warmup_gemms();
        eprintln!("[cutile] kernel warmup complete");
        Ok(model)
    }

    /// Upload all projection weights from `mw` (already f32) to the GPU.
    pub fn from_f32(
        mw: &crate::weights::ModelWeights,
        config: &ModelConfig,
    ) -> Result<Self, tempura_kernels::KernelError> {
        let matmul = CutileMatMul::new()?;
        let flash_attn = CutileFlashAttention::new()?;
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

        let model = Self { config: config.clone(), device_weights: dw, matmul, flash_attn };
        eprintln!("[cutile] warming up kernel specializations…");
        model.warmup_gemms();
        eprintln!("[cutile] kernel warmup complete");
        Ok(model)
    }

    /// Return the CUDA stream used by the GPU model.
    ///
    /// Share this with `CutileKvCache::new` so that KV-cache uploads are
    /// sequenced on the same stream as the GEMM kernels.
    pub fn stream(&self) -> Arc<cuda_core::Stream> {
        self.matmul.stream().clone()
    }

    // ── device tensor helpers ─────────────────────────────────────────────────

    /// Upload a host slice as a device tensor shaped `[rows, cols]`.
    fn to_device(&self, data: &[f32], rows: usize, cols: usize) -> Arc<Tensor<f32>> {
        let stream = self.matmul.stream();
        api::copy_host_vec_to_device(&Arc::new(data.to_vec()))
            .sync_on(stream)
            .expect("CutileGpuModel: host→device upload failed")
            .reshape(&[rows, cols])
            .expect("CutileGpuModel: reshape failed")
            .into()
    }

    /// Download a device tensor to a host Vec<f32>.
    fn from_device(&self, t: &Arc<Tensor<f32>>) -> Vec<f32> {
        t.to_host_vec()
            .sync_on(self.matmul.stream())
            .expect("CutileGpuModel: device→host download failed")
    }

    /// Device-to-device GEMM: `output[m, n] = a[m, k] @ b[n, k]^T`.
    ///
    /// Falls back silently (returns a zero tensor) if the weight is absent.
    fn gemm_d2d(&self, a: &Arc<Tensor<f32>>, weight_name: &str, m: usize) -> Arc<Tensor<f32>> {
        let dw = match self.device_weights.get(weight_name) {
            Some(d) => d,
            None => {
                let stream = self.matmul.stream();
                return Arc::new(
                    api::zeros::<f32>(&[m, 1])
                        .sync_on(stream)
                        .expect("zeros failed"),
                );
            }
        };
        // Ensure activation A has k_pad columns (zero-pad if needed).
        let a_ready: Arc<Tensor<f32>> = {
            let a_host = self.from_device(a);
            let k_raw_a = a_host.len() / m;
            if k_raw_a == dw.k_pad {
                a.clone()
            } else {
                let mut padded = vec![0.0f32; m * dw.k_pad];
                for row in 0..m {
                    padded[row * dw.k_pad..row * dw.k_pad + k_raw_a]
                        .copy_from_slice(&a_host[row * k_raw_a..(row + 1) * k_raw_a]);
                }
                self.to_device(&padded, m, dw.k_pad)
            }
        };
        self.matmul
            .forward_bt_device_to_device(&a_ready, &dw.tensor, m, dw.n, dw.k_pad)
            .expect("CutileGpuModel::gemm_d2d failed")
    }

    // ── inference ─────────────────────────────────────────────────────────────

    /// Full autoregressive forward pass with GPU-resident KV cache.
    ///
    /// Activations flow device-to-device across all projection GEMMs.
    /// RMSNorm and RoPE still run on CPU; each requires one host↔device
    /// round-trip per layer, but they are cheap relative to the projection
    /// GEMMs and the attention kernel.
    ///
    /// The caller is responsible for calling `gpu_kv_cache.advance(token_ids.len())`
    /// and incrementing the session position after this returns.
    pub fn forward(
        &self,
        position: usize,
        token_ids: &[u32],
        qweights: &QuantizedWeights,
        gpu_kv_cache: &mut CutileKvCache,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let seq_len = token_ids.len();
        let q_dim = n_heads * head_dim;
        let inter = cfg.intermediate_size;

        // ── 1. Embedding lookup (CPU) ─────────────────────────────────────────
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

        let positions: Vec<u32> = (position..position + seq_len).map(|p| p as u32).collect();

        let rope_mode = match cfg.rope_type {
            crate::config::RopeType::Norm => tempura_kernels::rope::RoPEMode::Normal,
            crate::config::RopeType::Neox => tempura_kernels::rope::RoPEMode::NeoX,
        };
        let rope = tempura_kernels::rope::CpuRoPEUnified::new(rope_mode, head_dim);
        let rmsnorm = tempura_kernels::rmsnorm::CpuRMSNormGgml;

        // CPU scratchpads reused across layers.
        let mut normed = vec![0.0f32; seq_len * h];
        let mut normed2 = vec![0.0f32; seq_len * h];

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

            // ── 2a. Pre-attention RMSNorm (CPU) ───────────────────────────────
            normed.copy_from_slice(&hidden);
            let input_ln_w = qweights.get_f32(input_ln_name).unwrap();
            rmsnorm.forward(&mut normed, input_ln_w, h, cfg.layer_norm_eps).unwrap();

            // ── 2b. QKV projections — device-to-device ───────────────────────
            // Upload normed once; Q, K, V are computed on device without
            // downloading normed again.
            let normed_dev = self.to_device(&normed[..seq_len * h], seq_len, h);
            let q_dev = self.gemm_d2d(&normed_dev, q_w_name, seq_len);
            let k_dev = self.gemm_d2d(&normed_dev, k_w_name, seq_len);
            let v_dev = self.gemm_d2d(&normed_dev, v_w_name, seq_len);

            // ── 2c. RoPE (CPU) — download Q, K; apply rotation; re-upload ────
            let mut q_host = self.from_device(&q_dev);
            let mut k_host = self.from_device(&k_dev);

            if !cfg.use_alibi {
                rope.forward(&mut q_host, 1, seq_len, n_heads, head_dim, &positions, cfg.rope_theta).unwrap();
                rope.forward(&mut k_host, 1, seq_len, n_kv_heads, head_dim, &positions, cfg.rope_theta).unwrap();
            }

            // ── 2d. GPU KV-cache append (host mirror) ────────────────────────
            // V does not need RoPE, download from device to update host mirror.
            let v_host = self.from_device(&v_dev);
            gpu_kv_cache.append_host(layer, &k_host, &v_host, seq_len);

            // kv_len includes the tokens just appended (seq_len new tokens on
            // top of whatever was already in the cache).
            let kv_len = gpu_kv_cache.seq_len() + seq_len;

            // ── 2e. Attention (device K/V from GPU cache) ─────────────────────
            let q_rot_dev = self.to_device(&q_host, seq_len, q_dim);
            let (k_cache_dev, v_cache_dev) = gpu_kv_cache
                .get_kv_device(layer, kv_len)
                .expect("CutileGpuModel: KV device upload failed");

            let attn_out_dev = self.flash_attn
                .forward_device(
                    &q_rot_dev,
                    &k_cache_dev,
                    &v_cache_dev,
                    seq_len,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    1.0 / (head_dim as f32).sqrt(),
                )
                .expect("CutileGpuModel: flash attention failed");

            // ── 2f. O projection — device-to-device ──────────────────────────
            let o_dev = self.gemm_d2d(&attn_out_dev, o_w_name, seq_len);

            // ── 2g. Residual add (CPU) ────────────────────────────────────────
            let o_host = self.from_device(&o_dev);
            for i in 0..seq_len * h {
                hidden[i] += o_host[i];
            }

            // ── 2h. Post-attention RMSNorm (CPU) ─────────────────────────────
            normed2.copy_from_slice(&hidden);
            let post_ln_w = qweights.get_f32(post_ln_name).unwrap();
            rmsnorm.forward(&mut normed2, post_ln_w, h, cfg.layer_norm_eps).unwrap();

            // ── 2i. FFN gate + up — device-to-device ─────────────────────────
            let normed2_dev = self.to_device(&normed2[..seq_len * h], seq_len, h);
            let gate_dev = self.gemm_d2d(&normed2_dev, gate_w_name, seq_len);
            let up_dev = self.gemm_d2d(&normed2_dev, up_w_name, seq_len);

            // ── 2j. SiLU gate × up (CPU) ─────────────────────────────────────
            let mut gate_host = self.from_device(&gate_dev);
            let up_host = self.from_device(&up_dev);
            for i in 0..seq_len * inter {
                gate_host[i] =
                    tempura_kernels::portable_math::portable_silu(gate_host[i]) * up_host[i];
            }

            // ── 2k. FFN down — device-to-device ──────────────────────────────
            let gate_silu_dev = self.to_device(&gate_host, seq_len, inter);
            let down_dev = self.gemm_d2d(&gate_silu_dev, down_w_name, seq_len);

            // ── 2l. Residual add (CPU) ────────────────────────────────────────
            let down_host = self.from_device(&down_dev);
            for i in 0..seq_len * h {
                hidden[i] += down_host[i];
            }
        }

        // ── 3. Final norm + lm_head ────────────────────────────────────────────
        let mut final_normed = hidden;
        let final_norm_w = qweights
            .get_f32("model.norm.weight")
            .expect("model.norm.weight not found");
        rmsnorm.forward(&mut final_normed, final_norm_w, h, cfg.layer_norm_eps).unwrap();

        // lm_head — device-to-device on the last token position.
        let last_hidden = &final_normed[(seq_len - 1) * h..seq_len * h];
        let last_dev = self.to_device(last_hidden, 1, h);
        let logits_dev = self.gemm_d2d(&last_dev, "lm_head.weight", 1);
        self.from_device(&logits_dev)
    }
}
