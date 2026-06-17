use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn vector_add(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= a.len() {
            return;
        }
        if let Some(elem) = out.get_mut(idx) {
            *elem = a[i] + b[i];
        }
    }

    #[kernel]
    pub fn silu_mul(gate: &[f32], up: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= gate.len() {
            return;
        }
        let g = gate[i];
        let silu = g / (1.0 + (-g).exp());
        if let Some(elem) = out.get_mut(idx) {
            *elem = silu * up[i];
        }
    }

    /// Extract a column range from a row-major matrix.
    /// src: [rows, total_cols], dst: [rows, slice_cols]
    /// Copies src[row, col_offset..col_offset+slice_cols] for each row.
    #[kernel]
    pub fn slice_columns(
        src: &[f32],
        mut dst: DisjointSlice<f32>,
        rows: u32,
        total_cols: u32,
        col_offset: u32,
        slice_cols: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let total = rows as usize * slice_cols as usize;
        if i >= total {
            return;
        }
        let row = i / slice_cols as usize;
        let col = i % slice_cols as usize;
        let src_idx = row * total_cols as usize + col_offset as usize + col;
        if let Some(elem) = dst.get_mut(idx) {
            *elem = src[src_idx];
        }
    }
}
