//! EAGLE-3 speculative decoding head (speculators format, e.g.
//! RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3).
//!
//! EAGLE-3 (arXiv 2503.01840) drafts tokens with a single decoder layer
//! conditioned on a FUSION of three of the TARGET model's hidden states
//! (layers 2 / n/2 / n-3, the official SafeAILab convention -- the stash
//! happens on the activations ENTERING those layers) plus the draft
//! token's own embedding.  The head keeps its own small autoregressive
//! feature-KV; drafting beyond the last verified position feeds the
//! head's own pre-norm output hidden forward (the `fc` fusion runs ONLY
//! on target-provided hidden states -- the official
//! `hidden_states.shape[-1] != inputs_embeds.shape[-1]` bypass).
//!
//! This module holds the CHECKPOINT side: config parsing, safetensors
//! loading (raw bytes -- the embed table stays bf16 on the host, I64/BOOL
//! vocab-mapping tensors are kept native) and the host f32 -> Q6_K
//! quantizer that lets the head reuse the engine's pooled oxide Q6K
//! GEMVs (the head is draft-only: its numerics influence acceptance
//! length, never the emitted tokens).  The GPU forward lives in
//! `cutile_gpu_forward.rs` under the `eagle_*` methods; the solo decode
//! cycle lives in `server/src/engine_task.rs`.

use std::collections::HashMap;
use std::path::Path;

/// Q6_K block geometry (matches the kernels crate's GGUF layout).
pub const Q6K_QK: usize = 256;
pub const Q6K_BLOCK_BYTES: usize = 210;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eagle3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub draft_vocab_size: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub norm_before_residual: bool,
}

impl Eagle3Config {
    /// Parse the speculators `config.json` from `dir`.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("eagle: reading config.json: {e}"))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("eagle: parsing config.json: {e}"))?;
        let tlc = v
            .get("transformer_layer_config")
            .ok_or("eagle: config.json missing transformer_layer_config")?;
        let g = |k: &str| -> Result<usize, String> {
            tlc.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("eagle: config missing {k}"))
        };
        let num_attention_heads = g("num_attention_heads")?;
        let num_key_value_heads = g("num_key_value_heads")?;
        let head_dim = tlc
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(g("hidden_size")? / num_attention_heads);
        Ok(Self {
            hidden_size: g("hidden_size")?,
            intermediate_size: g("intermediate_size")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            draft_vocab_size: v
                .get("draft_vocab_size")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or("eagle: config missing draft_vocab_size")?,
            vocab_size: g("vocab_size")?,
            rope_theta: tlc
                .get("rope_theta")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(500000.0),
            rms_norm_eps: tlc
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(1e-5),
            norm_before_residual: v
                .get("norm_before_residual")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        })
    }

    /// The three TARGET layers whose ENTERING activations feed the fusion
    /// `fc` (official EAGLE-3: {2, n/2, n-3}).
    pub fn tap_layers(num_layers: usize) -> [usize; 3] {
        if let Ok(v) = std::env::var("TEMPURA_EAGLE_TAP_LAYERS") {
            let idx: Vec<usize> = v
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if idx.len() == 3 {
                return [idx[0], idx[1], idx[2]];
            }
        }
        [2, num_layers / 2, num_layers.saturating_sub(3)]
    }
}

/// Host-side EAGLE-3 head weights, loaded from a speculators safetensors
/// checkpoint.  Matrices are f32 row-major [n, k]; `embed_bf16` keeps the
/// raw bf16 bytes ([vocab, hidden]) for cheap per-step row gathers.
pub struct Eagle3Weights {
    pub cfg: Eagle3Config,
    pub fc: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub embed_bf16: Vec<u8>,
    /// norm name -> weight (input_layernorm / hidden_norm /
    /// post_attention_layernorm / final norm).
    pub norms: HashMap<String, Vec<f32>>,
    /// draft id -> target id offset (target = draft + d2t[draft]).
    pub d2t: Vec<i64>,
}

fn bf16_f32(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

impl Eagle3Weights {
    /// Load `model.safetensors` from `dir` (mmap; the 1GB embed table is
    /// never fully materialized as f32).
    pub fn load(dir: &Path) -> Result<Self, String> {
        let cfg = Eagle3Config::from_dir(dir)?;
        let path = dir.join("model.safetensors");
        let file = std::fs::File::open(&path).map_err(|e| format!("eagle: open {path:?}: {e}"))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| format!("eagle: mmap {path:?}: {e}"))?;
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| format!("eagle: safetensors deserialize: {e}"))?;

        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;

        let mut mat = |name: &str, want: usize| -> Result<Vec<f32>, String> {
            let view = st
                .tensor(name)
                .map_err(|e| format!("eagle: tensor {name}: {e}"))?;
            let bytes = view.data();
            if bytes.len() != want * 2 {
                return Err(format!(
                    "eagle: tensor {name}: {} bytes, expected {} (bf16)",
                    bytes.len(),
                    want * 2
                ));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        };

        let fc = mat("fc.weight", h * 3 * h)?;
        let q_proj = mat("layers.0.self_attn.q_proj.weight", q_dim * 2 * h)?;
        let k_proj = mat("layers.0.self_attn.k_proj.weight", kv_dim * 2 * h)?;
        let v_proj = mat("layers.0.self_attn.v_proj.weight", kv_dim * 2 * h)?;
        let o_proj = mat("layers.0.self_attn.o_proj.weight", h * q_dim)?;
        let gate_proj = mat("layers.0.mlp.gate_proj.weight", inter * h)?;
        let up_proj = mat("layers.0.mlp.up_proj.weight", inter * h)?;
        let down_proj = mat("layers.0.mlp.down_proj.weight", h * inter)?;
        let lm_head = mat("lm_head.weight", cfg.draft_vocab_size * h)?;

        let embed_view = st
            .tensor("embed_tokens.weight")
            .map_err(|e| format!("eagle: embed_tokens: {e}"))?;
        let embed_bf16 = embed_view.data().to_vec();
        if embed_bf16.len() != cfg.vocab_size * h * 2 {
            return Err(format!(
                "eagle: embed_tokens: {} bytes, expected {}",
                embed_bf16.len(),
                cfg.vocab_size * h * 2
            ));
        }

        let mut norms = HashMap::new();
        for (field, name) in [
            ("input_layernorm", "layers.0.input_layernorm.weight"),
            ("hidden_norm", "layers.0.hidden_norm.weight"),
            ("post_attention_layernorm", "layers.0.post_attention_layernorm.weight"),
            ("norm", "norm.weight"),
        ] {
            norms.insert(field.to_string(), mat(name, h)?);
        }

        // d2t: I64 [draft_vocab].  (t2d is not needed on the GPU path --
        // greedy proposals map draft->target only.)
        let d2t_view = st
            .tensor("d2t")
            .map_err(|e| format!("eagle: d2t: {e}"))?;
        let d2t: Vec<i64> = d2t_view
            .data()
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
        if d2t.len() != cfg.draft_vocab_size {
            return Err(format!(
                "eagle: d2t len {} != draft_vocab {}",
                d2t.len(),
                cfg.draft_vocab_size
            ));
        }

        Ok(Self {
            cfg,
            fc,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
            lm_head,
            embed_bf16,
            norms,
            d2t,
        })
    }

    /// Gather one embedding row (bf16 -> f32) for `token`.
    pub fn embed_row(&self, token: u32) -> Vec<f32> {
        Self::embed_row_raw(&self.embed_bf16, token, self.cfg.hidden_size)
    }

    /// Raw-bytes embed row gather (shared by the GPU head's per-step
    /// staging).
    pub(crate) fn embed_row_raw(embed_bf16: &[u8], token: u32, h: usize) -> Vec<f32> {
        let off = token as usize * h * 2;
        embed_bf16[off..off + h * 2]
            .chunks_exact(2)
            .map(|c| bf16_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }
}

/// Quantize an f32 row-major [n, k] matrix to GGUF Q6_K blocks
/// (210 bytes per 256-element block, byte-exact inverse of the kernels
/// crate's `q6k_dot_block` layout).  Draft-only numerics: the scale
/// search is the straightforward per-sub-block amax round, not llama.cpp's
/// exact iteration -- acceptance-quality, not bit-compatibility, is the
/// contract.
pub fn quantize_q6k_host(w: &[f32], n: usize, k: usize) -> Vec<u8> {
    assert_eq!(w.len(), n * k);
    assert!(k % Q6K_QK == 0, "eagle: k={k} not a multiple of {Q6K_QK}");
    let k_blocks = k / Q6K_QK;
    let mut out = vec![0u8; n * k_blocks * Q6K_BLOCK_BYTES];
    for row in 0..n {
        for blk in 0..k_blocks {
            let src = &w[row * k + blk * Q6K_QK..row * k + (blk + 1) * Q6K_QK];
            let dst_off = (row * k_blocks + blk) * Q6K_BLOCK_BYTES;
            quantize_q6k_block(src, &mut out[dst_off..dst_off + Q6K_BLOCK_BYTES]);
        }
    }
    out
}

/// Quantize one 256-element block into the 210-byte GGUF Q6_K layout
/// (the byte-exact inverse of `q6k_dot_block`'s decode).
fn quantize_q6k_block(x: &[f32], dst: &mut [u8]) {
    // GGUF Q6_K scheme: per-16-element sub-scales (float) are themselves
    // quantized -- d = max_sub_scale / 31 (f16), stored 6-bit scale s_l =
    // round(sub_scale_l / d), element value = d * s_l * q with 6-bit
    // signed q in [-32, 31].
    let at = |buf: &[u8], i: usize| buf[i];
    let mut put = |buf: &mut [u8], i: usize, v: u8| buf[i] = v;
    // Sub-block amax -> float sub-scales.
    let mut sub_amax = [0.0f32; 16];
    for si in 0..16 {
        let half = si / 8;
        let rest = si % 8;
        let base = half * 128 + (rest / 2) * 32 + (rest % 2) * 16;
        let mut amax_g = 0.0f32;
        for t in 0..16 {
            amax_g = amax_g.max(x[base + t].abs());
        }
        sub_amax[si] = amax_g;
    }
    let max_scale = sub_amax.iter().fold(0.0f32, |a, v| a.max(*v)) / 31.0;
    let d = half::f16::from_f32(max_scale).to_f32();
    dst[208..210].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };

    for si in 0..16usize {
        // si = half*8 + (pos%32)/16 + 2*(pos/32); invert:
        let half = si / 8;
        let rest = si % 8;
        let base = half * 128 + (rest / 2) * 32 + (rest % 2) * 16;
        let sub_scale = sub_amax[si] / 31.0;
        let sc = (sub_scale * id).round().clamp(1.0, 63.0) as u32;
        dst[192 + si] = sc as u8;
        let yid = if sc > 0 { 1.0 / (d * sc as f32) } else { 0.0 };
        for t in 0..16 {
            let j = base + t;
            let v = ((x[j] * yid).round() as i32 + 32).clamp(0, 63) as u32;
            let half = j / 128;
            let pos = j % 128;
            let l = pos % 32;
            let lo = v & 0xF;
            let hi = (v >> 4) & 3;
            let (ql_lo, ql_hi) = (half * 64 + l, half * 64 + l + 32);
            let qh_i = 128 + half * 32 + l;
            match pos / 32 {
                0 => {
                    put(dst, ql_lo, (at(dst, ql_lo) & 0xF0) | lo as u8);
                    put(dst, qh_i, (at(dst, qh_i) & 0xFC) | hi as u8);
                }
                1 => {
                    put(dst, ql_hi, (at(dst, ql_hi) & 0xF0) | lo as u8);
                    put(dst, qh_i, (at(dst, qh_i) & 0xF3) | (hi as u8) << 2);
                }
                2 => {
                    put(dst, ql_lo, (at(dst, ql_lo) & 0x0F) | (lo as u8) << 4);
                    put(dst, qh_i, (at(dst, qh_i) & 0xCF) | (hi as u8) << 4);
                }
                _ => {
                    put(dst, ql_hi, (at(dst, ql_hi) & 0x0F) | (lo as u8) << 4);
                    put(dst, qh_i, (at(dst, qh_i) & 0x3F) | (hi as u8) << 6);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q6k_roundtrip_error_bounded() {
        // The layout round-trip must reproduce each element within one
        // quantization step of its scale.
        let k = Q6K_QK;
        let n = 4;
        let mut w = Vec::with_capacity(n * k);
        for i in 0..n * k {
            w.push(((i as f32 * 0.37) % 1.0 - 0.5) * 0.02);
        }
        let packed = quantize_q6k_host(&w, n, k);
        assert_eq!(packed.len(), n * Q6K_BLOCK_BYTES);
        for row in 0..n {
            let block = &packed[row * Q6K_BLOCK_BYTES..(row + 1) * Q6K_BLOCK_BYTES];
            let ql = &block[0..128];
            let qh = &block[128..192];
            let scales = &block[192..208];
            let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();
            let mut max_err = 0.0f32;
            let mut max_gate = 0.0f32;
            for j in 0..k {
                let (half, pos) = (j / 128, j % 128);
                let l = pos % 32;
                let q = match pos / 32 {
                    0 => ((ql[half * 64 + l] & 0xF) | ((qh[half * 32 + l] & 3) << 4)),
                    1 => (ql[half * 64 + l + 32] & 0xF) | (((qh[half * 32 + l] >> 2) & 3) << 4),
                    2 => (ql[half * 64 + l] >> 4) | (((qh[half * 32 + l] >> 4) & 3) << 4),
                    _ => (ql[half * 64 + l + 32] >> 4) | (((qh[half * 32 + l] >> 6) & 3) << 4),
                };
                let si = half * 8 + (pos % 32) / 16 + 2 * ((pos % 128) / 32);
                let v = (q as i32 - 32) as f32 * d * (scales[si] as i8 as f32);
                let err = (v - w[row * k + j]).abs();
                max_err = max_err.max(err);
                // The element's own quantization step is d * sc.
                max_gate = max_gate.max(d * (scales[si] as i8 as f32));


            }
            // One step (plus rounding slack) per element.
            assert!(
                max_err <= max_gate + 1e-9,
                "row {row}: max err {max_err} > max step {max_gate}"
            );
        }
    }

    #[test]
    fn tap_layers_default() {
        assert_eq!(Eagle3Config::tap_layers(32), [2, 16, 29]);
    }
}
