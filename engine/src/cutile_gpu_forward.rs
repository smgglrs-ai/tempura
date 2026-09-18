/// Fused GPU forward pass for the transformer.
///
/// All projection weight matrices are dequantized once at load time and uploaded to the GPU.
/// Activations flow device-to-device across all projection GEMMs within each layer.
///
/// Key invariant: whenever `gemm_d2d` is called the activation tensor MUST already have
/// exactly `dw.k_pad` columns.  Callers ensure this by uploading activations pre-padded
/// (via `to_device_padded`) so that `gemm_d2d` never has to download, re-pad, and
/// re-upload the activation.
///
/// The two remaining explicit pad steps per layer are:
///   - normed   → h_pad   (one `to_device_padded` call, used for Q, K, V)
///   - normed2  → h_pad   (one `to_device_padded` call, used for gate, up)
///   - gate×up  → inter_pad (one `to_device_padded` call, used for down)
///   - attn_out → q_pad   (one from_device + one to_device_padded, for O projection)
///
/// CPU operations that still require a host round-trip: RMSNorm, RoPE, SiLU gate,
/// and the residual adds.
use std::cell::RefCell;
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
    /// Padded K actually stored on the GPU (multiple of BK, power-of-2).
    k_pad: usize,
}

fn next_pow2(x: usize) -> usize {
    if x <= 1 { return 1; }
    1usize << (usize::BITS - (x - 1).leading_zeros()) as usize
}

// Scratch-pool slot indices — one pre-allocated decode output buffer per projection.
// Each slot is reclaimed between layers because the previous caller's Arc is dropped
// at the end of each loop body before the next layer's gemm_d2d_slot call.
const SLOT_Q: usize = 0;
const SLOT_K: usize = 1;
const SLOT_V: usize = 2;
const SLOT_O: usize = 3;
const SLOT_GATE: usize = 4;
const SLOT_UP: usize = 5;
const SLOT_DOWN: usize = 6;
const SLOT_LOGITS: usize = 7;
const NUM_SLOTS: usize = 8;

/// Transformer with all projection weights pre-uploaded to the GPU.
///
/// Create once with [`CutileGpuModel::from_quantized`] at model load time, then call
/// [`CutileGpuModel::forward`] for each inference step.
pub struct CutileGpuModel {
    config: ModelConfig,
    /// next_pow2(hidden_size.div_ceil(BK) * BK) — padded column count for h-wide activations.
    h_pad: usize,
    /// next_pow2(q_dim.div_ceil(BK) * BK) — padded column count for attention output (o_proj).
    q_pad: usize,
    /// next_pow2(intermediate_size.div_ceil(BK) * BK) — padded column count for FFN activations.
    inter_pad: usize,
    device_weights: HashMap<String, DeviceWeight>,
    matmul: CutileMatMul,
    flash_attn: CutileFlashAttention,
    /// Pre-allocated decode output scratch buffers indexed by SLOT_* constants.
    ///
    /// Eliminates the `api::zeros` allocation inside `forward_bt_device_to_device`
    /// for the 7 projection GEMMs + lm_head per decode token (m=1 only).
    /// Single-threaded inference guarantees each slot's Arc has strong_count == 1
    /// when the next layer begins, so `Arc::try_unwrap` can reclaim the Tensor
    /// without re-allocating device memory.
    scratch: RefCell<[Option<Arc<Tensor<f32>>>; NUM_SLOTS]>,
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
        let dummy_h = self.to_device(&vec![0.0f32; self.h_pad], 1, self.h_pad);
        let dummy_q = self.to_device(&vec![0.0f32; self.q_pad], 1, self.q_pad);
        let dummy_inter = self.to_device(&vec![0.0f32; self.inter_pad], 1, self.inter_pad);

        // Q/K/V projection shape: [1, h_pad] @ [q_dim/kv_dim, h_pad]^T
        let _ = self.gemm_d2d(&dummy_h, "model.layers.0.self_attn.q_proj.weight", 1);
        let _ = self.gemm_d2d(&dummy_h, "model.layers.0.self_attn.k_proj.weight", 1);
        let _ = self.gemm_d2d(&dummy_h, "model.layers.0.self_attn.v_proj.weight", 1);
        // O projection shape: [1, q_pad] @ [h, q_pad]^T
        let _ = self.gemm_d2d(&dummy_q, "model.layers.0.self_attn.o_proj.weight", 1);
        // Gate/up projection shape: [1, h_pad] @ [inter, h_pad]^T
        let _ = self.gemm_d2d(&dummy_h, "model.layers.0.mlp.gate_proj.weight", 1);
        let _ = self.gemm_d2d(&dummy_h, "model.layers.0.mlp.up_proj.weight", 1);
        // Down projection shape: [1, inter_pad] @ [h, inter_pad]^T
        let _ = self.gemm_d2d(&dummy_inter, "model.layers.0.mlp.down_proj.weight", 1);
        // lm_head shape: [1, h_pad] @ [vocab, h_pad]^T
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

        let h_pad = next_pow2(h.div_ceil(BK) * BK);
        let q_pad = next_pow2(q_dim.div_ceil(BK) * BK);
        let inter_pad = next_pow2(inter.div_ceil(BK) * BK);

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

        let model = Self {
            config: config.clone(),
            h_pad,
            q_pad,
            inter_pad,
            device_weights: dw,
            matmul,
            flash_attn,
            scratch: RefCell::new(std::array::from_fn(|_| None)),
        };
        eprintln!("[cutile] warming up kernel specializations…");
        model.warmup_gemms();
        model.preallocate_scratch();
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

        let h_pad = next_pow2(h.div_ceil(BK) * BK);
        let q_pad = next_pow2(q_dim.div_ceil(BK) * BK);
        let inter_pad = next_pow2(inter.div_ceil(BK) * BK);

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

        let model = Self {
            config: config.clone(),
            h_pad,
            q_pad,
            inter_pad,
            device_weights: dw,
            matmul,
            flash_attn,
            scratch: RefCell::new(std::array::from_fn(|_| None)),
        };
        eprintln!("[cutile] warming up kernel specializations…");
        model.warmup_gemms();
        model.preallocate_scratch();
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

    /// Upload a host slice to a device tensor shaped `[rows, padded_cols]`, zero-padding
    /// each row from `real_cols` to `padded_cols` on the CPU before the transfer.
    ///
    /// When `padded_cols == real_cols` no temporary allocation is made.
    fn to_device_padded(
        &self,
        data: &[f32],
        rows: usize,
        real_cols: usize,
        padded_cols: usize,
    ) -> Arc<Tensor<f32>> {
        if padded_cols == real_cols {
            return self.to_device(data, rows, padded_cols);
        }
        let mut padded = vec![0.0f32; rows * padded_cols];
        for row in 0..rows {
            padded[row * padded_cols..row * padded_cols + real_cols]
                .copy_from_slice(&data[row * real_cols..(row + 1) * real_cols]);
        }
        self.to_device(&padded, rows, padded_cols)
    }

    /// Download a device tensor to a host Vec<f32>.
    fn from_device(&self, t: &Arc<Tensor<f32>>) -> Vec<f32> {
        t.to_host_vec()
            .sync_on(self.matmul.stream())
            .expect("CutileGpuModel: device→host download failed")
    }

    /// Device-to-device GEMM: `output[m, n] = a[m, k_pad] @ b[n, k_pad]^T`.
    ///
    /// The caller MUST ensure `a` has exactly `dw.k_pad` columns.  No download
    /// or re-padding is performed here — the activation must already be padded.
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
        self.matmul
            .forward_bt_device_to_device(a, &dw.tensor, m, dw.n, dw.k_pad)
            .expect("CutileGpuModel::gemm_d2d failed")
    }

    /// Pre-allocate decode output scratch buffers for each projection slot.
    ///
    /// Called once after construction.  Each slot holds a 1-D flat tensor of
    /// n elements (the projection output width for m=1 decode).  On the first
    /// decode forward pass these buffers are zeroed and reused via
    /// `forward_bt_d2d_into` instead of calling `api::zeros` per GEMM.
    fn preallocate_scratch(&self) {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;

        let stream = self.matmul.stream();
        let alloc = |n: usize| -> Option<Arc<Tensor<f32>>> {
            Some(Arc::new(
                api::zeros::<f32>(&[n])
                    .sync_on(stream)
                    .expect("preallocate_scratch: device allocation failed"),
            ))
        };

        let mut pool = self.scratch.borrow_mut();
        pool[SLOT_Q] = alloc(q_dim);
        pool[SLOT_K] = alloc(kv_dim);
        pool[SLOT_V] = alloc(kv_dim);
        pool[SLOT_O] = alloc(h);
        pool[SLOT_GATE] = alloc(inter);
        pool[SLOT_UP] = alloc(inter);
        pool[SLOT_DOWN] = alloc(h);
        pool[SLOT_LOGITS] = alloc(vocab);
    }

    /// Device-to-device GEMM reusing a pre-allocated scratch buffer for the
    /// decode path (m == 1).
    ///
    /// When m == 1 and the pool slot holds a uniquely-owned `Arc<Tensor<f32>>`,
    /// reclaims the tensor, zeroes it in-place via `forward_bt_d2d_into`, and
    /// stores the result back so the next layer can reuse it.  Falls back to a
    /// fresh allocation for prefill (m > 1) or if the Arc is shared.
    fn gemm_d2d_slot(
        &self,
        a: &Arc<Tensor<f32>>,
        weight_name: &str,
        m: usize,
        slot: usize,
    ) -> Arc<Tensor<f32>> {
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

        // Pooled path: decode only.  Try to reclaim the pre-allocated scratch
        // if the previous layer's caller has dropped its reference.
        if m == 1 {
            let maybe_tensor = {
                let mut pool = self.scratch.borrow_mut();
                match pool[slot].take() {
                    Some(arc) => match Arc::try_unwrap(arc) {
                        Ok(tensor) => Some(tensor),
                        Err(arc_back) => {
                            // Caller still holds a reference — return the Arc
                            // to the pool and fall through to allocation.
                            pool[slot] = Some(arc_back);
                            None
                        }
                    },
                    None => None,
                }
            };

            if let Some(tensor) = maybe_tensor {
                if let Ok(result) =
                    self.matmul
                        .forward_bt_d2d_into(a, &dw.tensor, tensor, m, dw.n, dw.k_pad)
                {
                    self.scratch.borrow_mut()[slot] = Some(result.clone());
                    return result;
                }
            }
        }

        // Fallback: prefill (m > 1) or pool miss.
        self.matmul
            .forward_bt_device_to_device(a, &dw.tensor, m, dw.n, dw.k_pad)
            .expect("CutileGpuModel::gemm_d2d_slot failed")
    }

    // ── inference ─────────────────────────────────────────────────────────────

    /// Full autoregressive forward pass with GPU-resident KV cache.
    ///
    /// Activations flow device-to-device across all projection GEMMs.
    /// RMSNorm and RoPE still run on CPU; each requires one host↔device
    /// round-trip per layer, but they are cheap relative to the projection
    /// GEMMs and the attention kernel.
    ///
    /// All activations uploaded to the GPU are pre-padded to the required k_pad
    /// columns so that `gemm_d2d` never has to download and re-pad.  The only
    /// explicit pad step that involves a download is for the attention output
    /// (GPU-side, q_dim wide) before the O projection (expects q_pad columns).
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
            // Upload normed pre-padded to h_pad so gemm_d2d needs no round-trip.
            // All three Q/K/V GEMMs reuse the same device tensor.
            let normed_dev = self.to_device_padded(&normed[..seq_len * h], seq_len, h, self.h_pad);
            let q_dev = self.gemm_d2d_slot(&normed_dev, q_w_name, seq_len, SLOT_Q);
            let k_dev = self.gemm_d2d_slot(&normed_dev, k_w_name, seq_len, SLOT_K);
            let v_dev = self.gemm_d2d_slot(&normed_dev, v_w_name, seq_len, SLOT_V);

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
            // Flash attention does not require power-of-2 padding; upload q_dim wide.
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
            // attn_out_dev has q_dim columns; o_proj weight expects q_pad columns.
            // Pad once explicitly so gemm_d2d needs no download.
            let attn_padded_dev = if self.q_pad == q_dim {
                attn_out_dev
            } else {
                let attn_host = self.from_device(&attn_out_dev);
                self.to_device_padded(&attn_host, seq_len, q_dim, self.q_pad)
            };
            let o_dev = self.gemm_d2d_slot(&attn_padded_dev, o_w_name, seq_len, SLOT_O);

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
            // Upload normed2 pre-padded to h_pad; reused for both gate and up.
            let normed2_dev = self.to_device_padded(&normed2[..seq_len * h], seq_len, h, self.h_pad);
            let gate_dev = self.gemm_d2d_slot(&normed2_dev, gate_w_name, seq_len, SLOT_GATE);
            let up_dev = self.gemm_d2d_slot(&normed2_dev, up_w_name, seq_len, SLOT_UP);

            // ── 2j. SiLU gate × up (CPU) ─────────────────────────────────────
            let mut gate_host = self.from_device(&gate_dev);
            let up_host = self.from_device(&up_dev);
            for i in 0..seq_len * inter {
                gate_host[i] =
                    tempura_kernels::portable_math::portable_silu(gate_host[i]) * up_host[i];
            }

            // ── 2k. FFN down — device-to-device ──────────────────────────────
            // Upload gate×silu pre-padded to inter_pad so gemm_d2d needs no download.
            let gate_silu_dev = self.to_device_padded(&gate_host, seq_len, inter, self.inter_pad);
            let down_dev = self.gemm_d2d_slot(&gate_silu_dev, down_w_name, seq_len, SLOT_DOWN);

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
        // Pre-pad to h_pad so gemm_d2d needs no download.
        let last_hidden = &final_normed[(seq_len - 1) * h..seq_len * h];
        let last_dev = self.to_device_padded(last_hidden, 1, h, self.h_pad);
        let logits_dev = self.gemm_d2d_slot(&last_dev, "lm_head.weight", 1, SLOT_LOGITS);
        self.from_device(&logits_dev)
    }
}
