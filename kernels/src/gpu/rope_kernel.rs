use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// RoPE kernel: one thread per output element.
    ///
    /// Reads from x_in, writes to x_out (same layout, not in-place).
    /// Thread i maps to element i of the output. For each element,
    /// determines whether it's in the first or second half of head_dim
    /// and applies the appropriate cos/sin rotation.
    #[kernel]
    pub fn rope_forward(
        x_in: &[f32],
        positions: &[u32],
        mut x_out: DisjointSlice<f32>,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();

        let hd = head_dim as usize;
        let half = hd / 2;
        let total = seq_len as usize * num_heads as usize * hd;

        if i >= total {
            return;
        }

        let d_local = i % hd;
        let remaining = i / hd;
        let h = remaining % num_heads as usize;
        let s = remaining / num_heads as usize;

        let pos = positions[s] as f32;
        let offset = (s * num_heads as usize + h) * hd;

        if d_local < half {
            let d = d_local;
            let freq = pos / theta.powf(2.0 * d as f32 / hd as f32);
            let cos_val = freq.cos();
            let sin_val = freq.sin();
            let x0 = x_in[offset + d];
            let x1 = x_in[offset + d + half];
            if let Some(elem) = x_out.get_mut(idx) {
                *elem = x0 * cos_val - x1 * sin_val;
            }
        } else {
            let d = d_local - half;
            let freq = pos / theta.powf(2.0 * d as f32 / hd as f32);
            let cos_val = freq.cos();
            let sin_val = freq.sin();
            let x0 = x_in[offset + d];
            let x1 = x_in[offset + d + half];
            if let Some(elem) = x_out.get_mut(idx) {
                *elem = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}
