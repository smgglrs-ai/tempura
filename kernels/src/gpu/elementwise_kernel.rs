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
}
