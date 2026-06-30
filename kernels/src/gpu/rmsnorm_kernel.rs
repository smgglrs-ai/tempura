use cuda_device::{kernel, thread, warp, DisjointSlice};
use cuda_host::cuda_module;

const WARP_SIZE: usize = 32;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Pass 1 (warp-parallel): compute sum of squares per row.
    /// One warp (32 threads) per row. Each lane sums a stripe of
    /// hidden_size/32 elements, then butterfly reduction via shuffle_xor.
    /// All lanes write the final sum to distinct indices in ss_out
    /// (ss_out has num_tokens * 32 elements; pass 2 reads at stride 32).
    #[kernel]
    pub fn rmsnorm_sum_sq(x: &[f32], mut ss_out: DisjointSlice<f32>, hidden_size: u32) {
        let idx = thread::index_1d();
        let global_id = idx.get();

        let h = hidden_size as usize;
        let row = global_id / WARP_SIZE;
        let lane = warp::lane_id() as usize;
        let start = row * h;

        if start + h > x.len() {
            return;
        }

        let mut sum = 0.0f32;
        let mut i = lane;
        while i < h {
            let v = x[start + i];
            sum += v * v;
            i += WARP_SIZE;
        }

        sum += warp::shuffle_xor_f32(sum, 16);
        sum += warp::shuffle_xor_f32(sum, 8);
        sum += warp::shuffle_xor_f32(sum, 4);
        sum += warp::shuffle_xor_f32(sum, 2);
        sum += warp::shuffle_xor_f32(sum, 1);

        if let Some(elem) = ss_out.get_mut(idx) {
            *elem = sum;
        }
    }

    /// Deterministic sum-of-squares: one thread per row, sequential
    /// accumulation matching CPU order. ss_out has 1 element per row.
    #[kernel]
    pub fn rmsnorm_sum_sq_det(x: &[f32], mut ss_out: DisjointSlice<f32>, hidden_size: u32) {
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

    /// Pass 2: apply normalization.
    /// One thread per element. Reads the precomputed sum-of-squares
    /// for its row (at stride ss_stride in ss buffer) and applies normalization.
    #[kernel]
    pub fn rmsnorm_apply(
        x_in: &[f32],
        weight: &[f32],
        ss: &[f32],
        mut x_out: DisjointSlice<f32>,
        hidden_size: u32,
        ss_stride: u32,
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

        let rms = ((ss[row * ss_stride as usize] / hidden_size as f32) + eps).sqrt();
        let inv_rms = 1.0 / rms;

        if let Some(elem) = x_out.get_mut(idx) {
            *elem = x_in[i] * inv_rms * weight[col];
        }
    }
}
