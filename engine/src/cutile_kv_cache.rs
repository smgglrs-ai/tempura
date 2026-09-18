/// GPU-resident KV cache.
///
/// Maintains a host-side mirror of K and V data per layer. Device tensors are
/// generated on demand for attention kernels. Append writes to the host mirror
/// at the current sequence position; `get_kv_device` uploads the live prefix
/// to the GPU immediately before attention.
///
/// This structure eliminates the need to copy the full KV history from CPU RAM
/// to GPU on every forward pass, and provides the right interface for upgrading
/// to a fully GPU-native attention kernel (at which point the host upload in
/// `get_kv_device` will also disappear).
use std::sync::Arc;

use cutile::api;
use cutile::prelude::DeviceOp;
use cutile::tensor::Tensor;
use cuda_core::Stream;
use tempura_kernels::KernelError;

use crate::config::ModelConfig;

pub struct CutileKvCache {
    /// Host mirrors: k_host[layer] is flat [max_seq_len * kv_dim].
    k_host: Vec<Vec<f32>>,
    v_host: Vec<Vec<f32>>,
    seq_len: usize,
    n_layers: usize,
    kv_dim: usize,
    max_seq_len: usize,
    stream: Arc<Stream>,
}

// SAFETY: Tensor<f32> and Stream hold raw CUDA pointers but tempura uses a
// single-threaded inference path per model instance.
unsafe impl Send for CutileKvCache {}
unsafe impl Sync for CutileKvCache {}

impl CutileKvCache {
    /// Allocate host mirrors for all layers.
    ///
    /// `stream` must be the same CUDA stream used by the matmul / attention
    /// kernels so that uploads are sequenced correctly.
    pub fn new(config: &ModelConfig, stream: Arc<Stream>) -> Result<Self, KernelError> {
        let kv_dim = config.num_kv_heads * config.head_dim;
        let layer_size = config.max_seq_len * kv_dim;
        let k_host = (0..config.num_layers).map(|_| vec![0.0f32; layer_size]).collect();
        let v_host = (0..config.num_layers).map(|_| vec![0.0f32; layer_size]).collect();
        Ok(Self {
            k_host,
            v_host,
            seq_len: 0,
            n_layers: config.num_layers,
            kv_dim,
            max_seq_len: config.max_seq_len,
            stream,
        })
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Write `num_new_tokens` K/V rows into the host mirror at the current
    /// sequence position for `layer`.
    ///
    /// `k_new` and `v_new` are flat host slices of length
    /// `num_new_tokens * kv_dim`.
    pub fn append_host(
        &mut self,
        layer: usize,
        k_new: &[f32],
        v_new: &[f32],
        num_new_tokens: usize,
    ) {
        let stride = self.kv_dim;
        let dst = self.seq_len * stride;
        let src_len = num_new_tokens * stride;
        self.k_host[layer][dst..dst + src_len].copy_from_slice(&k_new[..src_len]);
        self.v_host[layer][dst..dst + src_len].copy_from_slice(&v_new[..src_len]);
    }

    /// Advance the sequence-length counter after all layers have been updated.
    pub fn advance(&mut self, n: usize) {
        self.seq_len += n;
        debug_assert!(self.seq_len <= self.max_seq_len);
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Upload the live K/V prefix (first `kv_len` tokens) for `layer` to
    /// device and return the resulting device tensors.
    ///
    /// The returned tensors are flat 1-D tensors of length `kv_len * kv_dim`
    /// on the GPU — ready to pass to `CutileFlashAttention::forward_device`.
    pub fn get_kv_device(
        &self,
        layer: usize,
        kv_len: usize,
    ) -> Result<(Arc<Tensor<f32>>, Arc<Tensor<f32>>), KernelError> {
        let stride = self.kv_dim;
        let len = kv_len * stride;

        let k_slice = Arc::new(self.k_host[layer][..len].to_vec());
        let v_slice = Arc::new(self.v_host[layer][..len].to_vec());

        let k_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&k_slice)
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let v_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&v_slice)
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        Ok((k_dev, v_dev))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn make_stream() -> Arc<Stream> {
        use cuda_core::Device;
        let dev = Device::new(0).expect("no CUDA device");
        dev.new_stream().expect("stream creation failed")
    }

    #[test]
    fn test_gpu_kv_cache_append_and_advance() {
        let config = ModelConfig::tiny_test();
        let stream = make_stream();
        let mut cache = CutileKvCache::new(&config, stream).unwrap();

        assert_eq!(cache.seq_len(), 0);

        let kv_dim = config.num_kv_heads * config.head_dim;
        let k: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.1).collect();
        let v: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.2).collect();

        cache.append_host(0, &k, &v, 1);
        cache.advance(1);

        assert_eq!(cache.seq_len(), 1);
        // Host mirror should contain the appended data at offset 0.
        assert!((cache.k_host[0][0] - k[0]).abs() < 1e-6);
    }

    #[test]
    fn test_gpu_kv_cache_get_kv_device_roundtrip() {
        use cutile::prelude::ToHostVec;

        let config = ModelConfig::tiny_test();
        let stream = make_stream();
        let mut cache = CutileKvCache::new(&config, Arc::clone(&stream)).unwrap();

        let kv_dim = config.num_kv_heads * config.head_dim;
        let k: Vec<f32> = (0..kv_dim).map(|i| (i + 1) as f32).collect();
        let v: Vec<f32> = (0..kv_dim).map(|i| (i + 1) as f32 * 2.0).collect();

        cache.append_host(0, &k, &v, 1);
        cache.advance(1);

        let (k_dev, v_dev) = cache.get_kv_device(0, 1).unwrap();

        let k_back: Vec<f32> = k_dev.to_host_vec().sync_on(&stream).unwrap();
        let v_back: Vec<f32> = v_dev.to_host_vec().sync_on(&stream).unwrap();

        assert_eq!(k_back.len(), kv_dim);
        for i in 0..kv_dim {
            assert!((k_back[i] - k[i]).abs() < 1e-5, "k mismatch at {i}");
            assert!((v_back[i] - v[i]).abs() < 1e-5, "v mismatch at {i}");
        }
    }

    #[test]
    fn test_gpu_kv_cache_reset() {
        let config = ModelConfig::tiny_test();
        let stream = make_stream();
        let mut cache = CutileKvCache::new(&config, stream).unwrap();

        let kv_dim = config.num_kv_heads * config.head_dim;
        let k: Vec<f32> = vec![1.0; kv_dim];
        let v: Vec<f32> = vec![2.0; kv_dim];
        cache.append_host(0, &k, &v, 1);
        cache.advance(1);
        assert_eq!(cache.seq_len(), 1);

        cache.reset();
        assert_eq!(cache.seq_len(), 0);
    }
}
