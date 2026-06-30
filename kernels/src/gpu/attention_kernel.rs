use cuda_device::{kernel, ptx_asm, thread, DisjointSlice};
use cuda_host::cuda_module;

#[inline(always)]
fn portable_exp(x: f32) -> f32 {
    const LN2_HI: f32 = 0.693145752;
    const LN2_LO: f32 = 1.428606820e-6;
    const LOG2E: f32 = 1.442695041;
    if x > 88.7 { return f32::INFINITY; }
    if x < -87.3 { return 0.0; }
    let n = (x * LOG2E + 0.5).floor();
    let ni = n as i32;
    let r = (x - n * LN2_HI) - n * LN2_LO;
    let mut p = 0.001388888f32;
    p = p * r + 0.008333360;
    p = p * r + 0.041666653;
    p = p * r + 0.166666672;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    p * f32::from_bits(((ni + 127) as u32) << 23)
}

#[inline(always)]
fn f32_to_f16_rte(x: f32) -> f32 {
    let h: u16;
    let result: f32;
    unsafe {
        ptx_asm!("cvt.rn.f16.f32 %0, %1;", out("=h") h, in("f") x);
        ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") result, in("h") h);
    }
    result
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Naive GPU attention: one thread per output element.
    /// Each thread redundantly computes softmax for its (b,s,h) group.
    /// Correct but not optimal — Flash Attention is a future optimization.
    #[kernel]
    pub fn attention_forward(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut output: DisjointSlice<f32>,
        batch: u32,
        seq_len: u32,
        kv_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();

        let hd = head_dim as usize;
        let nh = num_heads as usize;
        let nkv = num_kv_heads as usize;
        let sl = seq_len as usize;
        let kvl = kv_len as usize;

        let total = batch as usize * sl * nh * hd;
        if i >= total {
            return;
        }

        let d = i % hd;
        let remaining = i / hd;
        let h = remaining % nh;
        let remaining2 = remaining / nh;
        let s = remaining2 % sl;
        let b = remaining2 / sl;

        let heads_per_group = nh / nkv;
        let kv_h = h / heads_per_group;

        let q_base = ((b * sl + s) * nh + h) * hd;

        let cache_offset = kvl - sl;
        let causal_limit = cache_offset + s;

        // Compute dot products and online softmax
        let mut max_score = f32::NEG_INFINITY;
        let mut kv_s = 0usize;

        // First pass: find max score (with causal mask)
        while kv_s < kvl {
            if kv_s <= causal_limit {
                let k_base = ((b * kvl + kv_s) * nkv + kv_h) * hd;
                let mut dot = 0.0f32;
                let mut dd = 0usize;
                while dd < hd {
                    dot += q[q_base + dd] * k[k_base + dd];
                    dd += 1;
                }
                let score = dot * scale;
                if score > max_score {
                    max_score = score;
                }
            }
            kv_s += 1;
        }

        // Second pass: compute exp and sum (with causal mask)
        let mut exp_sum = 0.0f32;
        kv_s = 0;
        while kv_s < kvl {
            if kv_s <= causal_limit {
                let k_base = ((b * kvl + kv_s) * nkv + kv_h) * hd;
                let mut dot = 0.0f32;
                let mut dd = 0usize;
                while dd < hd {
                    dot += q[q_base + dd] * k[k_base + dd];
                    dd += 1;
                }
                let score = dot * scale;
                exp_sum += portable_exp(score - max_score);
            }
            kv_s += 1;
        }

        // Third pass: weighted sum of values for this d (with causal mask)
        let mut val = 0.0f32;
        kv_s = 0;
        while kv_s < kvl {
            if kv_s <= causal_limit {
                let k_base = ((b * kvl + kv_s) * nkv + kv_h) * hd;
                let mut dot = 0.0f32;
                let mut dd = 0usize;
                while dd < hd {
                    dot += q[q_base + dd] * k[k_base + dd];
                    dd += 1;
                }
                let score = dot * scale;
                let weight = portable_exp(score - max_score) / exp_sum;

                let v_base = ((b * kvl + kv_s) * nkv + kv_h) * hd;
                val += weight * v[v_base + d];
            }
            kv_s += 1;
        }

        if let Some(elem) = output.get_mut(idx) {
            *elem = val;
        }
    }

    /// Naive GPU attention with f16-truncated Q·K dot products.
    /// Same structure as attention_forward but Q/K values are rounded
    /// to f16 before multiply, accumulation stays f32.
    #[kernel]
    pub fn attention_forward_f16(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut output: DisjointSlice<f32>,
        batch: u32,
        seq_len: u32,
        kv_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();

        let hd = head_dim as usize;
        let nh = num_heads as usize;
        let nkv = num_kv_heads as usize;
        let sl = seq_len as usize;
        let kvl = kv_len as usize;

        let total = batch as usize * sl * nh * hd;
        if i >= total {
            return;
        }

        let d = i % hd;
        let remaining = i / hd;
        let h = remaining % nh;
        let remaining2 = remaining / nh;
        let s = remaining2 % sl;
        let b = remaining2 / sl;

        let heads_per_group = nh / nkv;
        let kv_h = h / heads_per_group;

        let q_base = ((b * sl + s) * nh + h) * hd;

        let cache_offset = kvl - sl;
        let causal_limit = cache_offset + s;

        let mut max_score = f32::NEG_INFINITY;
        let mut kv_s = 0usize;

        while kv_s < kvl {
            if kv_s <= causal_limit {
                let k_base = ((b * kvl + kv_s) * nkv + kv_h) * hd;
                let mut dot = 0.0f32;
                let mut dd = 0usize;
                while dd < hd {
                    dot += f32_to_f16_rte(q[q_base + dd]) * f32_to_f16_rte(k[k_base + dd]);
                    dd += 1;
                }
                let score = dot * scale;
                if score > max_score {
                    max_score = score;
                }
            }
            kv_s += 1;
        }

        let mut exp_sum = 0.0f32;
        kv_s = 0;
        while kv_s < kvl {
            if kv_s <= causal_limit {
                let k_base = ((b * kvl + kv_s) * nkv + kv_h) * hd;
                let mut dot = 0.0f32;
                let mut dd = 0usize;
                while dd < hd {
                    dot += f32_to_f16_rte(q[q_base + dd]) * f32_to_f16_rte(k[k_base + dd]);
                    dd += 1;
                }
                let score = dot * scale;
                exp_sum += portable_exp(score - max_score);
            }
            kv_s += 1;
        }

        let mut val = 0.0f32;
        kv_s = 0;
        while kv_s < kvl {
            if kv_s <= causal_limit {
                let k_base = ((b * kvl + kv_s) * nkv + kv_h) * hd;
                let mut dot = 0.0f32;
                let mut dd = 0usize;
                while dd < hd {
                    dot += f32_to_f16_rte(q[q_base + dd]) * f32_to_f16_rte(k[k_base + dd]);
                    dd += 1;
                }
                let score = dot * scale;
                let weight = portable_exp(score - max_score) / exp_sum;

                let v_base = ((b * kvl + kv_s) * nkv + kv_h) * hd;
                val += weight * v[v_base + d];
            }
            kv_s += 1;
        }

        if let Some(elem) = output.get_mut(idx) {
            *elem = val;
        }
    }
}
