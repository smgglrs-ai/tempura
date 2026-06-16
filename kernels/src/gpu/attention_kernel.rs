use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;

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

        // Compute dot products and online softmax
        let mut max_score = f32::NEG_INFINITY;
        let mut kv_s = 0usize;

        // First pass: find max score
        while kv_s < kvl {
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
            kv_s += 1;
        }

        // Second pass: compute exp and sum
        let mut exp_sum = 0.0f32;
        kv_s = 0;
        while kv_s < kvl {
            let k_base = ((b * kvl + kv_s) * nkv + kv_h) * hd;
            let mut dot = 0.0f32;
            let mut dd = 0usize;
            while dd < hd {
                dot += q[q_base + dd] * k[k_base + dd];
                dd += 1;
            }
            let score = dot * scale;
            exp_sum += (score - max_score).exp();
            kv_s += 1;
        }

        // Third pass: weighted sum of values for this d
        let mut val = 0.0f32;
        kv_s = 0;
        while kv_s < kvl {
            let k_base = ((b * kvl + kv_s) * nkv + kv_h) * hd;
            let mut dot = 0.0f32;
            let mut dd = 0usize;
            while dd < hd {
                dot += q[q_base + dd] * k[k_base + dd];
                dd += 1;
            }
            let score = dot * scale;
            let weight = (score - max_score).exp() / exp_sum;

            let v_base = ((b * kvl + kv_s) * nkv + kv_h) * hd;
            val += weight * v[v_base + d];
            kv_s += 1;
        }

        if let Some(elem) = output.get_mut(idx) {
            *elem = val;
        }
    }
}
