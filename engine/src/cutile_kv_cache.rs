/// GPU-resident KV cache.
///
/// Maintains a host-side mirror of K and V data per layer (for CPU attention
/// and prefill flash attention) and a GPU-side flat cache (for future GPU
/// decode attention).
///
/// GPU layout: `k_device[layer]` is a flat 1-D tensor of
/// `gpu_max_seq_len * kv_dim` f32 elements in interleaved layout
/// `[gpu_max_seq_len, n_kv_heads, head_dim]`.  This matches the host mirror
/// layout exactly, enabling a single D->D `cuMemcpyDtoDAsync` per new token
/// per K or V per layer (2 copies total per layer), vs. the n_kv_heads x 2
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
    /// the tokens appended without a D->H update.
    host_stale: bool,
    /// SHARED page pool (p20 step 4b-2): when set, this cache holds a
    /// CONTIGUOUS RUN of `run_pages` pages starting at `base_page`
    /// inside the pool (k_device/v_device point at the POOL tensors),
    /// grown on demand with headroom via extend-in-place-or-realloc.
    /// `None` = dedicated buffers (all pre-pool behavior, base 0).
    ///
    /// SINGLE-CLIENT INVARIANT: the wired configuration (one shared
    /// session cache per pool) keeps `base_page == 0` structurally --
    /// `kv_device_tensors`/`paged_attn_inputs` debug_assert it.  Adding a
    /// second client (concurrent sessions / GPU batch decode) requires
    /// making the raw-pointer call sites base-offset aware (the ~25
    /// uniform `kv_device_tensors(...).device_pointer()` extractions --
    /// see the 4b-2 notes in AGENTS.md).
    pool: Option<Arc<std::sync::Mutex<GpuKvPagePool>>>,
    base_page: usize,
    run_pages: usize,
    /// Bumped on every capacity change (see [`Self::capacity_gen`]).
    capacity_gen: usize,
    /// Identity block table for the paged attention kernel, rank-2
    /// [max_blocks, 1] (rank-1 gather views do not JIT -- see
    /// cutile_flash_attn_heads_paged).  The GPU pool is contiguous, so
    /// logical block i IS physical page i.  Lazily allocated once;
    /// contents never change, so the pointer is CUDA-graph stable.
    block_table_dev: Option<Arc<Tensor<i32>>>,
    /// Device scalar holding the current kv_len for the paged attention
    /// kernel (read at EXECUTION time, not launch time -- CUDA-graph
    /// safe, the kv_scatter_write device-offset pattern).  Lazily
    /// allocated once; the VALUE is updated via async H2D before each
    /// paged launch.
    kv_len_dev: Option<Arc<Tensor<i32>>>,
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
            block_table_dev: None,
            kv_len_dev: None,
            pool: None,
            base_page: 0,
            run_pages: 0,
            capacity_gen: 0,
        })
    }

    /// Pooled mode: hold a run of `initial_tokens` (rounded up to pages,
    /// with growth headroom) from a shared [`GpuKvPagePool`].
    ///
    /// `k_device`/`v_device` point at the pool tensors; `gpu_max_seq_len`
    /// is the RUN capacity and grows via [`Self::ensure_capacity`].  The
    /// host mirrors stay at `max_seq_len` (unchanged semantics).  See the
    /// `pool` field docs for the single-client base-0 invariant.
    pub fn new_pooled(
        config: &ModelConfig,
        pool: Arc<std::sync::Mutex<GpuKvPagePool>>,
        initial_tokens: usize,
        stream: Arc<Stream>,
    ) -> Result<Self, KernelError> {
        let kv_dim = config.num_kv_heads * config.head_dim;
        let layer_size = config.max_seq_len * kv_dim;
        let k_host = (0..config.num_layers)
            .map(|_| vec![0.0f32; layer_size])
            .collect();
        let v_host = (0..config.num_layers)
            .map(|_| vec![0.0f32; layer_size])
            .collect();

        let block_size = tempura_kernels::cutile_flash_attention::PAGED_BLOCK_SIZE;
        let need_pages = initial_tokens.div_ceil(block_size);
        // Growth headroom: at least +256 tokens so a fresh sequence does
        // not realloc on its second chunk.
        let alloc_pages = need_pages.max(initial_tokens.div_ceil(block_size) + 16);
        let (base_page, run_pages, k_device, v_device, gpu_max_seq_len) = {
            let mut p = pool
                .lock()
                .map_err(|_| KernelError::InvalidConfig("pool mutex poisoned".into()))?;
            let run = p
                .alloc_pages(alloc_pages)
                .ok_or_else(|| KernelError::InvalidConfig("KV pool exhausted".into()))?;
            let k_device = p.k_pool_tensors();
            let v_device = p.v_pool_tensors();
            (run.start, run.pages, k_device, v_device, run.pages * block_size)
        };

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
            block_table_dev: None,
            kv_len_dev: None,
            pool: Some(pool),
            base_page,
            run_pages,
            capacity_gen: 0,
        })
    }

    /// Byte offset of the run's token 0 within the pool tensors (0 in
    /// dedicated mode).
    fn kv_base_byte_offset(&self) -> u64 {
        let block_size = tempura_kernels::cutile_flash_attention::PAGED_BLOCK_SIZE;
        (self.base_page * block_size * self.kv_dim) as u64 * std::mem::size_of::<f32>() as u64
    }

    /// Grow the run to hold at least `tokens` tokens (pooled mode only).
    ///
    /// Extends in place when the adjacent pages are free (keeps baked
    /// CUDA-graph pointers stable); otherwise reallocates (D2D copy of
    /// the live prefix per layer, free the old run -- callers' captured
    /// graphs detect the pointer change and re-capture).  Growth uses a
    /// 2x factor to amortize.
    pub fn ensure_capacity(&mut self, tokens: usize) -> Result<(), KernelError> {
        let Some(pool_arc) = self.pool.clone() else {
            return Ok(());
        };
        let block_size = tempura_kernels::cutile_flash_attention::PAGED_BLOCK_SIZE;
        let need_pages = tokens.div_ceil(block_size);
        if need_pages <= self.run_pages {
            return Ok(());
        }
        let mut pool = pool_arc
            .lock()
            .map_err(|_| KernelError::InvalidConfig("pool mutex poisoned".into()))?;
        // Headroom: 2x the current run, at least +256 tokens.
        let target_pages = need_pages
            .max(self.run_pages * 2)
            .max(need_pages + 256 / block_size);
        if pool.try_extend_run(self.base_page, self.run_pages, target_pages) {
            self.run_pages = target_pages;
            self.gpu_max_seq_len = target_pages * block_size;
            self.capacity_gen += 1;
            // The cached block table covers the OLD capacity (it is
            // built lazily from gpu_max_seq_len at first use) -- drop
            // it so the next paged attention call rebuilds it at the
            // new size.  (Found by long-ctx ppl: the stale 48-entry
            // table made the kernel read OOB page ids -> ppl 2718.)
            self.block_table_dev = None;
            return Ok(());
        }
        // Reallocate: alloc a bigger run, copy the live prefix, free old.
        let new_run = pool
            .alloc_pages(target_pages)
            .ok_or_else(|| KernelError::InvalidConfig("KV pool exhausted on growth".into()))?;
        let old_base = self.base_page;
        let f32_bytes = std::mem::size_of::<f32>() as u64;
        let live_elems = self.seq_len * self.kv_dim;
        for layer in 0..self.n_layers {
            let k_src = self.k_device[layer].device_pointer().cu_deviceptr()
                + (old_base * block_size * self.kv_dim) as u64 * f32_bytes;
            let v_src = self.v_device[layer].device_pointer().cu_deviceptr()
                + (old_base * block_size * self.kv_dim) as u64 * f32_bytes;
            let k_dst = self.k_device[layer].device_pointer().cu_deviceptr()
                + (new_run.start * block_size * self.kv_dim) as u64 * f32_bytes;
            let v_dst = self.v_device[layer].device_pointer().cu_deviceptr()
                + (new_run.start * block_size * self.kv_dim) as u64 * f32_bytes;
            if live_elems > 0 {
                unsafe {
                    cuda_core::memcpy_dtod_async::<f32>(k_dst, k_src, live_elems, &self.stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
                    cuda_core::memcpy_dtod_async::<f32>(v_dst, v_src, live_elems, &self.stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
                }
            }
        }
        pool.free_pages_alloc(crate::gpu_page_pool::Run {
            start: old_base,
            pages: self.run_pages,
        });
        self.base_page = new_run.start;
        self.run_pages = new_run.pages;
        self.gpu_max_seq_len = new_run.pages * block_size;
        self.capacity_gen += 1;
        // The block table contents changed (new base) -- drop the cached
        // device table so the next paged attention call rebuilds it.
        self.block_table_dev = None;
        Ok(())
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
        match &self.pool {
            None => self.gpu_max_seq_len > 0 && self.seq_len < self.gpu_max_seq_len,
            // Pooled: the run can grow into free pool pages (the growth
            // happens inside the append; this gate must see the POTENTIAL,
            // not the current run size -- found via long-ctx ppl 2026-11-12:
            // the initial 768-token run made every long prefill fall to the
            // host-staged path before any growth could happen).
            Some(pool) => match pool.lock() {
                Ok(p) => {
                    self.seq_len < self.gpu_max_seq_len
                        || p.free_pages() > 0
                        || p.allocated_pages() == 0
                }
                Err(_) => false,
            },
        }
    }

    /// Returns `true` if `append_device_gpu_only` can accept `n` new tokens
    /// without exceeding the GPU cache capacity.  Unlike `has_device_kv`
    /// (which only guarantees room for one more token), this covers multi-
    /// token prefill chunks.
    pub fn can_append_device(&self, n: usize) -> bool {
        match &self.pool {
            None => self.gpu_max_seq_len > 0 && self.seq_len + n <= self.gpu_max_seq_len,
            // Pooled: growable while the pool has (or can free) enough
            // pages for the post-append length.
            Some(pool) => match pool.lock() {
                Ok(p) => {
                    let block_size =
                        tempura_kernels::cutile_flash_attention::PAGED_BLOCK_SIZE;
                    let need_pages = (self.seq_len + n).div_ceil(block_size);
                    need_pages <= self.run_pages + p.free_pages()
                }
                Err(_) => false,
            },
        }
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
    /// Updates both the host mirror (via D->H) and the flat GPU tensor (via
    /// async D->D `cuMemcpyDtoDAsync`).  Callers must follow with
    /// `advance(num_new_tokens)` after processing all layers.
    ///
    /// `k_new` must hold `num_new_tokens * kv_dim` elements in the same
    /// interleaved layout as the host mirror.
    ///
    /// The D->D copies are enqueued on the cache stream and complete before any
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

        // Update host mirror (D->H).
        let k_host_new: Vec<f32> = k_new
            .to_host_vec()
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let v_host_new: Vec<f32> = v_new
            .to_host_vec()
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        self.append_host(layer, &k_host_new, &v_host_new, num_new_tokens);

        // GPU D->D: copy k_new/v_new into the flat GPU cache at the current
        // sequence offset.  One copy per K and one per V (not per head).
        // The GPU layout matches the host: [seq, kv_dim] interleaved.
        // Pooled mode grows the run instead of silently skipping.
        if self.pool.is_some() {
            self.ensure_capacity(self.seq_len + num_new_tokens)?;
        }
        if self.seq_len + num_new_tokens > self.gpu_max_seq_len {
            return Ok(());
        }

        let num_elems = num_new_tokens * self.kv_dim;
        let offset_elems = self.seq_len * self.kv_dim;
        let f32_bytes = std::mem::size_of::<f32>() as u64;
        let base_off = self.kv_base_byte_offset();

        let k_src: CUdeviceptr = k_new.device_pointer().cu_deviceptr();
        let v_src: CUdeviceptr = v_new.device_pointer().cu_deviceptr();

        let k_dst: CUdeviceptr = self.k_device[layer].device_pointer().cu_deviceptr()
            + base_off
            + offset_elems as u64 * f32_bytes;
        let v_dst: CUdeviceptr = self.v_device[layer].device_pointer().cu_deviceptr()
            + base_off
            + offset_elems as u64 * f32_bytes;

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

    /// GPU-only KV append: skips the D->H host-mirror update entirely.
    ///
    /// For GPU decode paths where attention runs entirely on device, the host
    /// mirror is never read -- maintaining it burns 2 x n_layers D->H sync calls
    /// (one per K, one per V per layer) per decode token, ~56 blocking transfers
    /// for a 28-layer model.  Skipping the D->H update eliminates these.
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

        // Pooled mode grows the run instead of silently skipping.
        if self.pool.is_some() {
            self.ensure_capacity(self.seq_len + num_new_tokens)?;
        }
        if self.seq_len + num_new_tokens > self.gpu_max_seq_len {
            return Ok(());
        }

        let num_elems = num_new_tokens * self.kv_dim;
        let offset_elems = self.seq_len * self.kv_dim;
        let f32_bytes = std::mem::size_of::<f32>() as u64;
        let base_off = self.kv_base_byte_offset();

        let k_src: CUdeviceptr = k_new.device_pointer().cu_deviceptr();
        let v_src: CUdeviceptr = v_new.device_pointer().cu_deviceptr();

        let k_dst: CUdeviceptr = self.k_device[layer].device_pointer().cu_deviceptr()
            + base_off
            + offset_elems as u64 * f32_bytes;
        let v_dst: CUdeviceptr = self.v_device[layer].device_pointer().cu_deviceptr()
            + base_off
            + offset_elems as u64 * f32_bytes;

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
    /// sequence of `append_device_gpu_only` calls.  Performs one D->H copy per K
    /// and V per layer (blocking, on the shared CUDA stream).
    pub fn sync_host_from_device(&mut self, kv_len: usize) -> Result<(), KernelError> {
        let len = kv_len * self.kv_dim;
        if kv_len == 0 || kv_len > self.gpu_max_seq_len {
            return Ok(());
        }
        let base_off = self.kv_base_byte_offset();
        let f32_bytes = std::mem::size_of::<f32>() as u64;
        for layer in 0..self.n_layers {
            // Offset D2H (raw memcpy): pooled mode must NOT download the
            // whole pool tensor (to_host_vec) -- the pool can be tens of
            // GB while the live prefix is small.
            let k_src = self.k_device[layer].device_pointer().cu_deviceptr() + base_off;
            let v_src = self.v_device[layer].device_pointer().cu_deviceptr() + base_off;
            unsafe {
                let rc = cuda_core::sys::cuMemcpyDtoHAsync_v2(
                    self.k_host[layer].as_mut_ptr() as *mut _,
                    k_src,
                    len * f32_bytes as usize,
                    self.stream.cu_stream(),
                );
                if rc != 0 {
                    return Err(KernelError::InvalidConfig(format!(
                        "sync_host k D2H rc={rc}"
                    )));
                }
                let rc = cuda_core::sys::cuMemcpyDtoHAsync_v2(
                    self.v_host[layer].as_mut_ptr() as *mut _,
                    v_src,
                    len * f32_bytes as usize,
                    self.stream.cu_stream(),
                );
                if rc != 0 {
                    return Err(KernelError::InvalidConfig(format!(
                        "sync_host v D2H rc={rc}"
                    )));
                }
            }
        }
        self.stream
            .synchronize()
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        self.host_stale = false;
        Ok(())
    }

    /// Mark the host KV mirror as stale after a device-side scatter-write.
    ///
    /// Called when the scatter-write kernel updates the GPU cache directly
    /// (bypassing `append_device_gpu_only`).  The host mirror must be rebuilt
    /// via `sync_host_from_device` before any CPU attention fallback.
    pub fn mark_host_stale(&mut self) {
        self.host_stale = true;
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
    /// The K/V data at positions >= target_seq_len remains in device memory but is
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
        // Single-client invariant (see the `pool` field docs): pooled
        // tensors are handed out only while the run sits at pool base.
        // A second pool client must make the raw-pointer call sites
        // base-offset aware first.
        debug_assert!(self.pool.is_none() || self.base_page == 0);
        (
            Arc::clone(&self.k_device[layer]),
            Arc::clone(&self.v_device[layer]),
        )
    }

    /// Inputs for the paged prefill attention kernel (p20): zero-copy
    /// rank-2 pool views, the identity block table, and the kv_len device
    /// scalar.  The kv_len VALUE is uploaded via async H2D on this call
    /// (stream-ordered before the launch that consumes it); the buffer
    /// pointer is allocated once and never changes, so captured CUDA
    /// graphs replay correctly as the cache grows.
    ///
    /// Pool view layout: `[-1, head_dim]` rows = `token * n_kv_heads +
    /// kv_head` (the same interleaved bytes as the rank-1 cache, viewed).
    /// Requires `head_dim` to divide `kv_dim` and `layer` to be valid.
    pub fn paged_attn_inputs(
        &mut self,
        layer: usize,
        head_dim: usize,
        kv_len: usize,
    ) -> Result<
        (
            Arc<Tensor<f32>>,
            Arc<Tensor<f32>>,
            Arc<Tensor<i32>>,
            Arc<Tensor<i32>>,
        ),
        KernelError,
    > {
        use cutile::tensor::Reshape;

        if head_dim == 0 || self.kv_dim % head_dim != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "paged_attn_inputs: head_dim {head_dim} must divide kv_dim {}",
                self.kv_dim
            )));
        }
        // Single-client invariant (see the `pool` field docs): the
        // identity table is only valid while the run sits at pool base.
        debug_assert!(self.pool.is_none() || self.base_page == 0);
        // View rows from the TENSOR's actual element count: in pooled
        // mode the tensor is the whole pool (bigger than the run).
        let tensor_elems = self.k_device[layer]
            .shape()
            .iter()
            .map(|&d| d as usize)
            .product::<usize>();
        let rows = tensor_elems / head_dim;
        let k_pool = self.k_device[layer]
            .reshape(&[rows, head_dim])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let v_pool = self.v_device[layer]
            .reshape(&[rows, head_dim])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        if self.block_table_dev.is_none() {
            let max_blocks = self
                .gpu_max_seq_len
                .div_ceil(tempura_kernels::cutile_flash_attention::PAGED_BLOCK_SIZE);
            let table: Vec<i32> = (0..max_blocks as i32).collect();
            let t = api::copy_host_vec_to_device(&Arc::new(table))
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .reshape(&[max_blocks, 1])
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            self.block_table_dev = Some(Arc::new(t));
        }
        if self.kv_len_dev.is_none() {
            let t = api::copy_host_vec_to_device(&Arc::new(vec![kv_len as i32]))
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            self.kv_len_dev = Some(Arc::new(t));
        }
        // Update the kv_len scalar: 4 bytes, async, execution-time read.
        let kv_len_i = kv_len as i32;
        let kv_len_dev = self.kv_len_dev.as_ref().unwrap();
        unsafe {
            cuda_core::memcpy_htod_async::<i32>(
                kv_len_dev.device_pointer().cu_deviceptr(),
                &kv_len_i as *const i32,
                1,
                &self.stream,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("paged_attn_inputs kv_len H2D: {e}")))?;

        Ok((
            k_pool,
            v_pool,
            self.block_table_dev.as_ref().unwrap().clone(),
            kv_len_dev.clone(),
        ))
    }

    pub fn gpu_max_seq_len(&self) -> usize {
        self.gpu_max_seq_len
    }

    /// Token capacity of the (possibly pooled) GPU cache.
    pub fn gpu_capacity_tokens(&self) -> usize {
        self.gpu_max_seq_len
    }

    /// Monotonic generation counter, bumped on every capacity change
    /// (pooled growth).  Captured CUDA graphs compare it to detect
    /// in-place growth that pointer validation cannot see.
    pub fn capacity_gen(&self) -> usize {
        self.capacity_gen
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Upload the live K/V prefix (first `kv_len` tokens) for `layer` to
    /// device and return the resulting device tensors.
    ///
    /// The returned tensors are flat 1-D tensors of length `kv_len * kv_dim`
    /// on the GPU -- ready to pass to `CutileFlashAttention::forward_device`.
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

        // Verify GPU flat tensor (sync stream first to ensure D->D completed).
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

// ============================================================================
// Shared GPU KV page pool (p20 step 4b)
// ============================================================================

/// A shared GPU KV pool with page-run allocation.
///
/// Sequences hold CONTIGUOUS runs of `block_size`-token pages allocated
/// from this pool ([`PageRunAllocator`], Kani-verified in
/// gpu_page_pool.rs), so a sequence's block table is affine
/// (`base_page + i`) and every contiguity-assuming reader works from a
/// base-shifted pointer -- no kernel/PTX changes.  This is the
/// foundation for sharing one pool across batch sequences (each batch
/// sequence today costs a FULL max-size buffer -- 1024MiB/seq at 4096
/// ctx; a run costs only the pages it uses) and for prefix
/// sharing/eviction later.
///
/// NOT yet wired into `CutileKvCache`/the decode graphs (that migration
/// touches graph-captured base pointers -- recorded as the p20 step-4b
/// follow-up); validated directly by the "GPU page pool" section of
/// bench-decode-ops.
pub struct GpuKvPagePool {
    /// Per-layer flat pools `[total_pages * page_stride]`, interleaved
    /// `[page, token_in_page, kv_head, head_dim]`.
    k_pools: Vec<Arc<Tensor<f32>>>,
    v_pools: Vec<Arc<Tensor<f32>>>,
    alloc: crate::gpu_page_pool::PageRunAllocator,
    /// Elements per page (block_size * kv_dim).
    page_stride: usize,
    block_size: usize,
    kv_dim: usize,
    stream: Arc<Stream>,
}

unsafe impl Send for GpuKvPagePool {}
unsafe impl Sync for GpuKvPagePool {}

/// A contiguous page run allocated from a [`GpuKvPagePool`].
///
/// Holds everything the paged attention kernel needs: whole-pool rank-2
/// views, the run's affine block table (`base_page + i`, rank-2 `[pages,
/// 1]` -- rank-1 gather views do not JIT), and a kv_len device scalar
/// (execution-time read, CUDA-graph safe).  Freeing returns the pages to
/// the pool via [`GpuKvPagePool::free_run`].
pub struct GpuKvRun {
    base_page: usize,
    pages: usize,
    /// Per-layer rank-2 `[-1, head_dim]` views of the WHOLE pool (rows =
    /// token * n_kv_heads + kv_head; zero-copy `Reshape` views).
    k_views: Vec<Arc<Tensor<f32>>>,
    v_views: Vec<Arc<Tensor<f32>>>,
    block_table_dev: Arc<Tensor<i32>>,
    kv_len_dev: Arc<Tensor<i32>>,
    kv_dim: usize,
    block_size: usize,
    stream: Arc<Stream>,
}

unsafe impl Send for GpuKvRun {}
unsafe impl Sync for GpuKvRun {}

impl GpuKvPagePool {
    /// Allocate a pool of `total_pages` pages of `PAGED_BLOCK_SIZE`
    /// (16) tokens each, for `config`'s layer/kv geometry.
    pub fn new(
        config: &ModelConfig,
        total_pages: usize,
        stream: Arc<Stream>,
    ) -> Result<Self, KernelError> {
        Self::new_with_dims(
            config.num_layers,
            config.num_kv_heads * config.head_dim,
            total_pages,
            stream,
        )
    }

    /// Dimension-explicit constructor (tests / non-ModelConfig callers).
    pub fn new_with_dims(
        n_layers: usize,
        kv_dim: usize,
        total_pages: usize,
        stream: Arc<Stream>,
    ) -> Result<Self, KernelError> {
        let block_size = tempura_kernels::cutile_flash_attention::PAGED_BLOCK_SIZE;
        let page_stride = block_size * kv_dim;
        let layer_elems = total_pages * page_stride;
        let mut k_pools = Vec::with_capacity(n_layers);
        let mut v_pools = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let k: Arc<Tensor<f32>> = api::zeros::<f32>(&[layer_elems])
                .sync_on(&stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .into();
            let v: Arc<Tensor<f32>> = api::zeros::<f32>(&[layer_elems])
                .sync_on(&stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .into();
            k_pools.push(k);
            v_pools.push(v);
        }
        Ok(Self {
            k_pools,
            v_pools,
            alloc: crate::gpu_page_pool::PageRunAllocator::new(total_pages),
            page_stride,
            block_size,
            kv_dim,
            stream,
        })
    }

    pub fn free_pages(&self) -> usize {
        self.alloc.free_pages()
    }

    pub fn allocated_pages(&self) -> usize {
        self.alloc.allocated_pages()
    }

    /// Allocate `pages` contiguous pages (raw allocator access).
    pub fn alloc_pages(
        &mut self,
        pages: usize,
    ) -> Option<crate::gpu_page_pool::Run> {
        self.alloc.alloc(pages)
    }

    /// Free a raw run (coalescing).
    pub fn free_pages_alloc(&mut self, run: crate::gpu_page_pool::Run) {
        self.alloc.free(run);
    }

    /// Extend the run `[start, start+pages)` to `target_pages` in place
    /// when the adjacent pages are free.
    pub fn try_extend_run(&mut self, start: usize, pages: usize, target_pages: usize) -> bool {
        self.alloc.try_extend(start, pages, target_pages)
    }

    /// Per-layer pool tensors (Arc clones; the whole pool, NOT the run).
    pub fn k_pool_tensors(&self) -> Vec<Arc<Tensor<f32>>> {
        self.k_pools.clone()
    }

    pub fn v_pool_tensors(&self) -> Vec<Arc<Tensor<f32>>> {
        self.v_pools.clone()
    }

    pub fn total_pages(&self) -> usize {
        self.alloc.total_pages()
    }

    /// Allocate a contiguous run of `pages` pages and build its attention
    /// inputs.  `head_dim` must divide `kv_dim` (the rank-2 view rows are
    /// `token * n_kv_heads + kv_head`).
    pub fn alloc_run(
        &mut self,
        pages: usize,
        head_dim: usize,
    ) -> Option<GpuKvRun> {
        use cutile::tensor::Reshape;

        if head_dim == 0 || self.kv_dim % head_dim != 0 {
            return None;
        }
        let run = self.alloc.alloc(pages)?;
        let rows = self.alloc.total_pages() * self.block_size * self.kv_dim / head_dim;
        let k_views = self
            .k_pools
            .iter()
            .map(|t| t.reshape(&[rows, head_dim]).ok())
            .collect::<Option<Vec<_>>>()?;
        let v_views = self
            .v_pools
            .iter()
            .map(|t| t.reshape(&[rows, head_dim]).ok())
            .collect::<Option<Vec<_>>>()?;
        // Affine table: block i of this run is physical page base+i.
        let table: Vec<i32> = (0..pages).map(|i| (run.start + i) as i32).collect();
        let block_table_dev: Arc<Tensor<i32>> = api::copy_host_vec_to_device(&Arc::new(table))
            .sync_on(&self.stream)
            .ok()?
            .reshape(&[pages, 1])
            .ok()?
            .into();
        let kv_len_dev: Arc<Tensor<i32>> =
            api::copy_host_vec_to_device(&Arc::new(vec![0i32]))
                .sync_on(&self.stream)
                .ok()?
                .into();
        Some(GpuKvRun {
            base_page: run.start,
            pages,
            k_views,
            v_views,
            block_table_dev,
            kv_len_dev,
            kv_dim: self.kv_dim,
            block_size: self.block_size,
            stream: self.stream.clone(),
        })
    }

    /// Return a run's pages to the pool (coalescing).  The run's views
    /// may outlive the free (Arc-shared storage) but must not be used
    /// for attention after the pages are reused.
    pub fn free_run(&mut self, run: &GpuKvRun) {
        self.alloc.free(crate::gpu_page_pool::Run {
            start: run.base_page,
            pages: run.pages,
        });
    }
}

impl GpuKvRun {
    pub fn base_page(&self) -> usize {
        self.base_page
    }

    pub fn pages(&self) -> usize {
        self.pages
    }

    /// Token capacity of the run.
    pub fn capacity_tokens(&self) -> usize {
        self.pages * self.block_size
    }

    /// Element offset of `token` within the flat pool (for raw-pointer
    /// access by contiguity-assuming readers: base pointer + this offset
    /// * 4 bytes).
    pub fn token_elem_offset(&self, token: usize) -> usize {
        self.base_page * self.block_size * self.kv_dim + token * self.kv_dim
    }

    /// Upload `k`/`v` host slices (`n * kv_dim` elements each) as `n`
    /// tokens starting at `token_offset` within the run (async,
    /// stream-ordered).
    pub fn upload_tokens(
        &self,
        layer: usize,
        token_offset: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), KernelError> {
        let n = k.len() / self.kv_dim;
        debug_assert_eq!(v.len(), k.len());
        let base = self.token_elem_offset(token_offset);
        let kp = self.k_views[layer].device_pointer().cu_deviceptr();
        let vp = self.v_views[layer].device_pointer().cu_deviceptr();
        unsafe {
            cuda_core::memcpy_htod_async::<f32>(
                kp + (base as u64) * 4,
                k.as_ptr(),
                n * self.kv_dim,
                &self.stream,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("pool k upload: {e}")))?;
        unsafe {
            cuda_core::memcpy_htod_async::<f32>(
                vp + (base as u64) * 4,
                v.as_ptr(),
                n * self.kv_dim,
                &self.stream,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("pool v upload: {e}")))?;
        Ok(())
    }

    /// Attention inputs for `cutile_flash_attn_heads_paged` /
    /// `prefill_attention_device_paged`: (k view, v view, block table,
    /// kv_len scalar).  The kv_len VALUE is uploaded via async H2D
    /// (stream-ordered before the consuming launch); the scalar's pointer
    /// is stable across calls (CUDA-graph safe).
    pub fn paged_attn_inputs(
        &self,
        layer: usize,
        kv_len: usize,
    ) -> Result<
        (
            Arc<Tensor<f32>>,
            Arc<Tensor<f32>>,
            Arc<Tensor<i32>>,
            Arc<Tensor<i32>>,
        ),
        KernelError,
    > {
        let kv_len_i = kv_len as i32;
        unsafe {
            cuda_core::memcpy_htod_async::<i32>(
                self.kv_len_dev.device_pointer().cu_deviceptr(),
                &kv_len_i as *const i32,
                1,
                &self.stream,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("run kv_len H2D: {e}")))?;
        Ok((
            self.k_views[layer].clone(),
            self.v_views[layer].clone(),
            self.block_table_dev.clone(),
            self.kv_len_dev.clone(),
        ))
    }
}
