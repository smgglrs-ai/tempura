/// GPU-resident KV cache.
///
/// Maintains a host-side mirror of K and V data per layer (for CPU attention
/// and prefill flash attention) and a GPU-side flat cache (for future GPU
/// decode attention).
///
/// GPU layout: `k_device[layer]` is a flat 1-D tensor of
/// `gpu_max_seq_len * kv_dim` f32 elements in interleaved layout
/// `[gpu_max_seq_len, n_kv_heads, head_dim]`.  This matches the host mirror
/// layout exactly, enabling a single D→D `cuMemcpyDtoDAsync` per new token
/// per K or V per layer (2 copies total per layer), vs. the n_kv_heads × 2
/// scatter copies that a per-head-contiguous layout would require.
///
/// The GPU cache is populated by `append_device` and is reserved for a future
/// GPU-side decode attention kernel.  Current decode attention reads from the
/// host mirror via `get_kv_host_slices`.
use std::sync::Arc;

use cuda_core::Stream;
use cutile::api;
use cutile::prelude::{DeviceOp, ToHostVec};
use cutile::tensor::Tensor;
use tempura_kernels::KernelError;

use crate::config::ModelConfig;

/// Maximum number of tokens stored in the GPU-resident KV cache.
///
/// Set to 4096.  For models with `max_seq_len > GPU_KV_MAX_SEQ_LEN` the GPU
/// cache covers the first 4096 positions; sequences beyond that skip the GPU
/// cache update (host mirror remains authoritative).
pub const GPU_KV_MAX_SEQ_LEN: usize = 4096;

pub struct CutileKvCache {
    /// Host mirrors: k_host[layer] is flat [max_seq_len * kv_dim].
    k_host: Vec<Vec<f32>>,
    v_host: Vec<Vec<f32>>,
    seq_len: usize,
    n_layers: usize,
    kv_dim: usize,
    max_seq_len: usize,
    stream: Arc<Stream>,

    /// GPU-resident KV cache: k_device[layer] is flat [gpu_max_seq_len * kv_dim].
    /// Layout matches the host mirror: interleaved [token, kv_head, head_dim].
    /// Populated by `append_device`; empty when `gpu_max_seq_len == 0`.
    k_device: Vec<Arc<Tensor<f32>>>,
    v_device: Vec<Arc<Tensor<f32>>>,
    gpu_max_seq_len: usize,
    /// True when `append_device_gpu_only` has been called at least once since the
    /// last `sync_host_from_device`.  Signals that the host mirror is stale for
    /// the tokens appended without a D→H update.
    host_stale: bool,
}

// SAFETY: Tensor<f32> and Stream hold raw CUDA pointers but tempura uses a
// single-threaded inference path per model instance.
unsafe impl Send for CutileKvCache {}
unsafe impl Sync for CutileKvCache {}

impl CutileKvCache {
    /// Allocate host mirrors and GPU flat tensors for all layers.
    ///
    /// `stream` must be the same CUDA stream used by the matmul / attention
    /// kernels so that GPU operations are sequenced correctly.
    pub fn new(config: &ModelConfig, stream: Arc<Stream>) -> Result<Self, KernelError> {
        let kv_dim = config.num_kv_heads * config.head_dim;
        let layer_size = config.max_seq_len * kv_dim;
        let k_host = (0..config.num_layers)
            .map(|_| vec![0.0f32; layer_size])
            .collect();
        let v_host = (0..config.num_layers)
            .map(|_| vec![0.0f32; layer_size])
            .collect();

        let gpu_max_seq_len = config.max_seq_len.min(GPU_KV_MAX_SEQ_LEN);
        let gpu_layer_size = gpu_max_seq_len * kv_dim;

        let mut k_device: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(config.num_layers);
        let mut v_device: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let k_t: Arc<Tensor<f32>> = api::zeros::<f32>(&[gpu_layer_size])
                .sync_on(&stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .into();
            let v_t: Arc<Tensor<f32>> = api::zeros::<f32>(&[gpu_layer_size])
                .sync_on(&stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .into();
            k_device.push(k_t);
            v_device.push(v_t);
        }

        Ok(Self {
            k_host,
            v_host,
            seq_len: 0,
            n_layers: config.num_layers,
            kv_dim,
            max_seq_len: config.max_seq_len,
            stream,
            k_device,
            v_device,
            gpu_max_seq_len,
            host_stale: false,
        })
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Returns `true` if the GPU-resident KV cache is available and has capacity
    /// for the current sequence position.
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
    /// Updates both the host mirror (via D→H) and the flat GPU tensor (via
    /// async D→D `cuMemcpyDtoDAsync`).  Callers must follow with
    /// `advance(num_new_tokens)` after processing all layers.
    ///
    /// `k_new` must hold `num_new_tokens * kv_dim` elements in the same
    /// interleaved layout as the host mirror.
    ///
    /// The D→D copies are enqueued on the cache stream and complete before any
    /// subsequent operation on the same stream, so no explicit sync is needed
    /// between `append_device` and downstream GPU operations.
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

        // GPU D→D: copy k_new/v_new into the flat GPU cache at the current
        // sequence offset.  One copy per K and one per V (not per head).
        // The GPU layout matches the host: [seq, kv_dim] interleaved.
        if self.seq_len + num_new_tokens > self.gpu_max_seq_len {
            return Ok(());
        }

        let num_elems = num_new_tokens * self.kv_dim;
        let offset_elems = self.seq_len * self.kv_dim;
        let f32_bytes = std::mem::size_of::<f32>() as u64;

        let k_src: CUdeviceptr = k_new.device_pointer().cu_deviceptr();
        let v_src: CUdeviceptr = v_new.device_pointer().cu_deviceptr();

        let k_dst: CUdeviceptr =
            self.k_device[layer].device_pointer().cu_deviceptr() + offset_elems as u64 * f32_bytes;
        let v_dst: CUdeviceptr =
            self.v_device[layer].device_pointer().cu_deviceptr() + offset_elems as u64 * f32_bytes;

        unsafe {
            cuda_core::memcpy_dtod_async::<f32>(k_dst, k_src, num_elems, &self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            cuda_core::memcpy_dtod_async::<f32>(v_dst, v_src, num_elems, &self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }

        Ok(())
    }

    /// Returns `true` if the host mirror is stale (i.e., `append_device_gpu_only`
    /// has been called since the last `sync_host_from_device`).
    pub fn is_host_stale(&self) -> bool {
        self.host_stale
    }

    /// GPU-only KV append: skips the D→H host-mirror update entirely.
    ///
    /// For GPU decode paths where attention runs entirely on device, the host
    /// mirror is never read — maintaining it burns 2 × n_layers D→H sync calls
    /// (one per K, one per V per layer) per decode token, ~56 blocking transfers
    /// for a 28-layer model.  Skipping the D→H update eliminates these.
    ///
    /// The host mirror becomes stale.  Callers MUST call `sync_host_from_device`
    /// before switching to the CPU attention fallback (long-context or prefill).
    pub fn append_device_gpu_only(
        &mut self,
        layer: usize,
        k_new: &Arc<Tensor<f32>>,
        v_new: &Arc<Tensor<f32>>,
        num_new_tokens: usize,
    ) -> Result<(), KernelError> {
        use cuda_core::sys::CUdeviceptr;

        if self.seq_len + num_new_tokens > self.gpu_max_seq_len {
            return Ok(());
        }

        let num_elems = num_new_tokens * self.kv_dim;
        let offset_elems = self.seq_len * self.kv_dim;
        let f32_bytes = std::mem::size_of::<f32>() as u64;

        let k_src: CUdeviceptr = k_new.device_pointer().cu_deviceptr();
        let v_src: CUdeviceptr = v_new.device_pointer().cu_deviceptr();

        let k_dst: CUdeviceptr =
            self.k_device[layer].device_pointer().cu_deviceptr() + offset_elems as u64 * f32_bytes;
        let v_dst: CUdeviceptr =
            self.v_device[layer].device_pointer().cu_deviceptr() + offset_elems as u64 * f32_bytes;

        unsafe {
            cuda_core::memcpy_dtod_async::<f32>(k_dst, k_src, num_elems, &self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            cuda_core::memcpy_dtod_async::<f32>(v_dst, v_src, num_elems, &self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }

        self.host_stale = true;
        Ok(())
    }

    /// Rebuild the host mirror from the GPU cache for all layers up to `kv_len`.
    ///
    /// Must be called before switching to the CPU attention fallback after a
    /// sequence of `append_device_gpu_only` calls.  Performs one D→H copy per K
    /// and V per layer (blocking, on the shared CUDA stream).
    pub fn sync_host_from_device(&mut self, kv_len: usize) -> Result<(), KernelError> {
        let len = kv_len * self.kv_dim;
        if kv_len == 0 || kv_len > self.gpu_max_seq_len {
            return Ok(());
        }
        for layer in 0..self.n_layers {
            let k_host: Vec<f32> = self.k_device[layer].clone()
                .to_host_vec()
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            let v_host: Vec<f32> = self.v_device[layer].clone()
                .to_host_vec()
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            self.k_host[layer][..len].copy_from_slice(&k_host[..len]);
            self.v_host[layer][..len].copy_from_slice(&v_host[..len]);
        }
        self.host_stale = false;
        Ok(())
    }

    /// Advance the sequence-length counter after all layers have been updated.
    pub fn advance(&mut self, n: usize) {
        self.seq_len += n;
        debug_assert!(self.seq_len <= self.max_seq_len);
    }

    /// Advance the host-side seq_len counter without submitting GPU work.
    ///
    /// Used by the Opt 5 single-capture CUDA graph path: on steps 2+, the GPU
    /// KV append runs inside the cached graph, but the host counter must still
    /// be incremented so subsequent host-side seq_len queries are correct.
    pub fn advance_host_seq_len(&mut self, n: usize) {
        self.seq_len += n;
        debug_assert!(self.seq_len <= self.max_seq_len);
    }

    /// Roll back the KV cache to `target_seq_len` without touching device memory.
    ///
    /// The K/V data at positions ≥ target_seq_len remains in device memory but is
    /// considered logically overwritten on the next append.  This is safe because
    /// `append_device_gpu_only` always writes at `seq_len` (which we reset here),
    /// so subsequent appends will overwrite the stale positions.
    ///
    /// Used by speculative decoding to discard rejected draft tokens.
    pub fn rollback(&mut self, target_seq_len: usize) {
        debug_assert!(target_seq_len <= self.seq_len, "rollback target > seq_len");
        self.seq_len = target_seq_len;
        self.host_stale = true;
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        self.host_stale = false;
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

    /// Return the GPU tensor for K in `layer`.
    ///
    /// The tensor holds `gpu_max_seq_len * kv_dim` elements in interleaved
    /// layout `[gpu_max_seq_len, n_kv_heads, head_dim]`; only the first
    /// `seq_len * kv_dim` elements are valid.  Flash attention kernels receive
    /// `kv_len` as a runtime parameter and iterate only over valid positions.
    pub fn k_device(&self, layer: usize) -> Arc<Tensor<f32>> {
        Arc::clone(&self.k_device[layer])
    }

    pub fn v_device(&self, layer: usize) -> Arc<Tensor<f32>> {
        Arc::clone(&self.v_device[layer])
    }

    /// Return both K and V GPU device tensors for `layer`.
    ///
    /// Each tensor is flat `[gpu_max_seq_len * kv_dim]` in interleaved
    /// `[token, kv_head, head_dim]` layout.  Only the first `seq_len * kv_dim`
    /// elements are valid; the rest are zero-initialised.
    pub fn kv_device_tensors(&self, layer: usize) -> (Arc<Tensor<f32>>, Arc<Tensor<f32>>) {
        (Arc::clone(&self.k_device[layer]), Arc::clone(&self.v_device[layer]))
    }

    pub fn gpu_max_seq_len(&self) -> usize {
        self.gpu_max_seq_len
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_dim
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
    fn test_gpu_kv_cache_append_device_roundtrip() {
        use cutile::prelude::ToHostVec;

        let config = ModelConfig::tiny_test();
        let stream = make_stream();
        let mut cache = CutileKvCache::new(&config, Arc::clone(&stream)).unwrap();

        let kv_dim = config.num_kv_heads * config.head_dim;
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

        // Verify host mirror.
        for i in 0..kv_dim {
            assert!(
                (cache.k_host[0][i] - k_vals[i]).abs() < 1e-5,
                "host k mismatch at {i}"
            );
            assert!(
                (cache.v_host[0][i] - v_vals[i]).abs() < 1e-5,
                "host v mismatch at {i}"
            );
        }

        // Verify GPU flat tensor (sync stream first to ensure D→D completed).
        unsafe {
            stream.synchronize().unwrap();
        }
        let k_gpu: Vec<f32> = cache.k_device(0).to_host_vec().sync_on(&stream).unwrap();
        let v_gpu: Vec<f32> = cache.v_device(0).to_host_vec().sync_on(&stream).unwrap();
        for i in 0..kv_dim {
            assert!(
                (k_gpu[i] - k_vals[i]).abs() < 1e-5,
                "GPU k mismatch at {i}: expected={} got={}",
                k_vals[i],
                k_gpu[i]
            );
            assert!(
                (v_gpu[i] - v_vals[i]).abs() < 1e-5,
                "GPU v mismatch at {i}: expected={} got={}",
                v_vals[i],
                v_gpu[i]
            );
        }
    }
}
