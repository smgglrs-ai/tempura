use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Pass 1: compute sum of squares per row.
    /// One thread per row, sequential reduction within each row.
    /// For hidden_size <= 4096 this is fast enough; warp reductions
    /// can optimize later.
    #[kernel]
    pub fn rmsnorm_sum_sq(
        x: &[f32],
        mut ss_out: DisjointSlice<f32>,
        hidden_size: u32,
    ) {
        let idx = thread::index_1d();
        let row = idx.get();

        let h = hidden_size as usize;
        let start = row * h;

        if start + h > x.len() {
            return;
        }

        let mut sum = 0.0f32;
        let mut i = 0usize;
        while i < h {
            let v = x[start + i];
            sum += v * v;
            i += 1;
        }

        if let Some(elem) = ss_out.get_mut(idx) {
            *elem = sum;
        }
    }

    /// Pass 2: apply normalization in-place.
    /// One thread per element. Reads the precomputed sum-of-squares
    /// for its row and applies x[i] = x[i] * inv_rms * weight[i].
    #[kernel]
    pub fn rmsnorm_apply(
        x_in: &[f32],
        weight: &[f32],
        ss: &[f32],
        mut x_out: DisjointSlice<f32>,
        hidden_size: u32,
        eps: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();

        let h = hidden_size as usize;
        if i >= x_in.len() {
            return;
        }

        let row = i / h;
        let col = i % h;

        let rms = ((ss[row] / hidden_size as f32) + eps).sqrt();
        let inv_rms = 1.0 / rms;

        if let Some(elem) = x_out.get_mut(idx) {
            *elem = x_in[i] * inv_rms * weight[col];
        }
    }
}
