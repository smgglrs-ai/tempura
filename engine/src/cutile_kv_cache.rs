/// GPU-resident KV cache.
///
/// Maintains a host-side mirror of K and V data per layer (for prefill path)
/// and a GPU-side per-head cache (for decode GPU attention path).
///
/// GPU layout: `k_device_heads[layer][kv_h]` is a flat 1-D tensor of
/// `gpu_max_seq_len * head_dim` f32 elements.  Consecutive tokens for head
/// `kv_h` are stored at `[seq * head_dim .. (seq+1) * head_dim]`.
///
/// `append_device` updates both the host mirror (via a D→H copy) and the
/// per-head GPU tensors (via async D→D `cuMemcpyDtoDAsync`) so that:
///   - Decode attention can read K/V directly from GPU without any H2D
///     re-upload of the growing history.
///   - Prefill can continue to use `get_kv_host_slices` + `forward_batched`.
use std::sync::Arc;

use cuda_core::Stream;
use cutile::api;
use cutile::prelude::{DeviceOp, ToHostVec};
use cutile::tensor::Tensor;
use tempura_kernels::KernelError;

use crate::config::ModelConfig;

/// Maximum number of tokens stored in the GPU-resident KV cache.
///
/// Set to 4096. For models with `max_seq_len > GPU_KV_MAX_SEQ_LEN` the GPU
/// cache covers the first 4096 positions; sequences beyond that fall back to
/// the CPU-attention path (which reads from the host mirror).
pub const GPU_KV_MAX_SEQ_LEN: usize = 4096;

pub struct CutileKvCache {
    /// Host mirrors: k_host[layer] is flat [max_seq_len * kv_dim].
    k_host: Vec<Vec<f32>>,
    v_host: Vec<Vec<f32>>,
    seq_len: usize,
    n_layers: usize,
    kv_dim: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    stream: Arc<Stream>,

    /// Per-head GPU tensors: k_device_heads[layer][kv_h] is flat
    /// [gpu_max_seq_len * head_dim].  Empty when `gpu_max_seq_len == 0`.
    k_device_heads: Vec<Vec<Arc<Tensor<f32>>>>,
    v_device_heads: Vec<Vec<Arc<Tensor<f32>>>>,
    gpu_max_seq_len: usize,
}

// SAFETY: Tensor<f32> and Stream hold raw CUDA pointers but tempura uses a
// single-threaded inference path per model instance.
unsafe impl Send for CutileKvCache {}
unsafe impl Sync for CutileKvCache {}

impl CutileKvCache {
    /// Allocate host mirrors and GPU per-head tensors for all layers.
    ///
    /// `stream` must be the same CUDA stream used by the matmul / attention
    /// kernels so that uploads are sequenced correctly.
    pub fn new(config: &ModelConfig, stream: Arc<Stream>) -> Result<Self, KernelError> {
        let n_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let layer_size = config.max_seq_len * kv_dim;
        let k_host = (0..config.num_layers)
            .map(|_| vec![0.0f32; layer_size])
            .collect();
        let v_host = (0..config.num_layers)
            .map(|_| vec![0.0f32; layer_size])
            .collect();

        let gpu_max_seq_len = config.max_seq_len.min(GPU_KV_MAX_SEQ_LEN);
        let head_layer_size = gpu_max_seq_len * head_dim;

        let mut k_device_heads: Vec<Vec<Arc<Tensor<f32>>>> =
            Vec::with_capacity(config.num_layers);
        let mut v_device_heads: Vec<Vec<Arc<Tensor<f32>>>> =
            Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            let mut k_layer: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(n_kv_heads);
            let mut v_layer: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(n_kv_heads);
            for _ in 0..n_kv_heads {
                let k_t: Arc<Tensor<f32>> = api::zeros::<f32>(&[head_layer_size])
                    .sync_on(&stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .into();
                let v_t: Arc<Tensor<f32>> = api::zeros::<f32>(&[head_layer_size])
                    .sync_on(&stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .into();
                k_layer.push(k_t);
                v_layer.push(v_t);
            }
            k_device_heads.push(k_layer);
            v_device_heads.push(v_layer);
        }

        Ok(Self {
            k_host,
            v_host,
            seq_len: 0,
            n_layers: config.num_layers,
            kv_dim,
            n_kv_heads,
            head_dim,
            max_seq_len: config.max_seq_len,
            stream,
            k_device_heads,
            v_device_heads,
            gpu_max_seq_len,
        })
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Returns `true` if there is a GPU-resident KV cache and the current
    /// sequence length fits within it.
    pub fn has_device_kv(&self) -> bool {
        self.gpu_max_seq_len > 0 && self.seq_len < self.gpu_max_seq_len
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

    /// Append `num_new_tokens` K/V tokens from device tensors for `layer`.
    ///
    /// This method updates both the host mirror (via a D→H download) and the
    /// per-head GPU tensors (via async D→D copies).  Callers must follow up
    /// with `advance(num_new_tokens)` after processing all layers.
    ///
    /// `k_new` is expected to be a flat tensor of `num_new_tokens * kv_dim`
    /// elements in interleaved layout `[num_new_tokens, n_kv_heads, head_dim]`.
    ///
    /// The D→D copies are enqueued on the cache's CUDA stream and complete
    /// before any subsequent operation on that stream, so no explicit sync is
    /// needed between `append_device` and the flash attention kernel.
    pub fn append_device(
        &mut self,
        layer: usize,
        k_new: &Arc<Tensor<f32>>,
        v_new: &Arc<Tensor<f32>>,
        num_new_tokens: usize,
    ) -> Result<(), KernelError> {
        use cuda_core::sys::CUdeviceptr;

        // Update host mirror (D→H).
        let k_host_new: Vec<f32> = k_new
            .to_host_vec()
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let v_host_new: Vec<f32> = v_new
            .to_host_vec()
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        self.append_host(layer, &k_host_new, &v_host_new, num_new_tokens);

        // GPU per-head D→D scatter (if the GPU cache has capacity).
        if self.seq_len + num_new_tokens > self.gpu_max_seq_len {
            return Ok(());
        }

        let head_dim = self.head_dim;
        let n_kv_heads = self.n_kv_heads;
        let f32_bytes = std::mem::size_of::<f32>() as u64;

        // k_new layout (interleaved): token t, kv-head h → offset (t * n_kv_heads + h) * head_dim
        let k_src_base: CUdeviceptr = k_new.device_pointer().cu_deviceptr();
        let v_src_base: CUdeviceptr = v_new.device_pointer().cu_deviceptr();

        for t in 0..num_new_tokens {
            let dst_tok = self.seq_len + t;
            let dst_offset_elems = dst_tok * head_dim;

            for kv_h in 0..n_kv_heads {
                let src_offset_elems = (t * n_kv_heads + kv_h) * head_dim;
                let src_k: CUdeviceptr = k_src_base + (src_offset_elems as u64 * f32_bytes);
                let src_v: CUdeviceptr = v_src_base + (src_offset_elems as u64 * f32_bytes);

                let dst_k: CUdeviceptr =
                    self.k_device_heads[layer][kv_h].device_pointer().cu_deviceptr()
                        + (dst_offset_elems as u64 * f32_bytes);
                let dst_v: CUdeviceptr =
                    self.v_device_heads[layer][kv_h].device_pointer().cu_deviceptr()
                        + (dst_offset_elems as u64 * f32_bytes);

                unsafe {
                    cuda_core::memcpy_dtod_async::<f32>(dst_k, src_k, head_dim, &self.stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
                    cuda_core::memcpy_dtod_async::<f32>(dst_v, src_v, head_dim, &self.stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
                }
            }
        }

        Ok(())
    }

    /// Advance the sequence-length counter after all layers have been updated.
    pub fn advance(&mut self, n: usize) {
        self.seq_len += n;
        debug_assert!(self.seq_len <= self.max_seq_len);
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Return host slices for the live K/V prefix (first `kv_len` tokens) for
    /// `layer`, directly from the host mirror.
    ///
    /// These slices can be passed to `CutileFlashAttention::forward` without
    /// any device upload.  Avoids the upload + immediate re-download cycle that
    /// occurs when calling `get_kv_device` followed by `forward_device`.
    pub fn get_kv_host_slices(&self, layer: usize, kv_len: usize) -> (&[f32], &[f32]) {
        let len = kv_len * self.kv_dim;
        (&self.k_host[layer][..len], &self.v_host[layer][..len])
    }

    /// Return the GPU tensor for KV-head `kv_h` in `layer`.
    ///
    /// The tensor holds `gpu_max_seq_len * head_dim` elements; only the first
    /// `seq_len * head_dim` elements are valid (written by `append_device`).
    /// Flash attention kernels receive `kv_len` as a runtime parameter and
    /// iterate only over valid positions.
    pub fn k_device_head(&self, layer: usize, kv_h: usize) -> Arc<Tensor<f32>> {
        Arc::clone(&self.k_device_heads[layer][kv_h])
    }

    pub fn v_device_head(&self, layer: usize, kv_h: usize) -> Arc<Tensor<f32>> {
        Arc::clone(&self.v_device_heads[layer][kv_h])
    }

    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn gpu_max_seq_len(&self) -> usize {
        self.gpu_max_seq_len
    }

    /// Upload the live K/V prefix (first `kv_len` tokens) for `layer` to
    /// device and return the resulting device tensors.
    ///
    /// The returned tensors are flat 1-D tensors of length `kv_len * kv_dim`
    /// on the GPU — ready to pass to `CutileFlashAttention::forward_device`.
    #[allow(clippy::type_complexity)]
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

    #[test]
    fn test_gpu_kv_cache_append_device_per_head() {
        use cutile::prelude::ToHostVec;

        let config = ModelConfig::tiny_test();
        let stream = make_stream();
        let mut cache = CutileKvCache::new(&config, Arc::clone(&stream)).unwrap();

        let n_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Create a device tensor with known values.
        let k_vals: Vec<f32> = (0..kv_dim).map(|i| (i + 1) as f32).collect();
        let v_vals: Vec<f32> = (0..kv_dim).map(|i| (i + 1) as f32 * 2.0).collect();
        let k_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(k_vals.clone()))
            .sync_on(&stream)
            .unwrap()
            .into();
        let v_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(v_vals.clone()))
            .sync_on(&stream)
            .unwrap()
            .into();

        cache.append_device(0, &k_dev, &v_dev, 1).unwrap();
        cache.advance(1);

        // Check host mirror.
        for i in 0..kv_dim {
            assert!((cache.k_host[0][i] - k_vals[i]).abs() < 1e-5);
        }

        // Check GPU per-head tensors.
        for kv_h in 0..n_kv_heads {
            let k_head = cache.k_device_head(0, kv_h);
            let k_head_host: Vec<f32> = k_head.to_host_vec().sync_on(&stream).unwrap();
            // First head_dim elements of k_head_host should match k_vals[kv_h*head_dim..]
            for d in 0..head_dim {
                let expected = k_vals[kv_h * head_dim + d];
                assert!(
                    (k_head_host[d] - expected).abs() < 1e-5,
                    "kv_h={kv_h} d={d}: expected={expected} got={}",
                    k_head_host[d]
                );
            }
        }
    }
}
