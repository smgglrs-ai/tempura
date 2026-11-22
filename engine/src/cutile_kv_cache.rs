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

/// Recoverable KV-pool exhaustion: what a failed run growth needed vs
/// what the pool had free.  Carried up to the batch scheduler, which
/// picks a swap victim ([`CutileKvCache::swap_out`]) and retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolExhausted {
    /// Pages the failing growth needed.
    pub need_pages: usize,
    /// Pages free in the pool at failure time.
    pub free_pages: usize,
}

/// [`CutileKvCache::try_ensure_capacity`] error: pool exhaustion is
/// RECOVERABLE (evict + retry); kernel errors are not.
#[derive(Debug)]
pub enum EnsureCapacityError {
    Exhausted(PoolExhausted),
    Kernel(KernelError),
}

/// Maximum number of tokens stored in the GPU-resident KV cache.
///
/// Set to 4096.  For models with `max_seq_len > GPU_KV_MAX_SEQ_LEN` the GPU
/// cache covers the first 4096 positions; sequences beyond that skip the GPU
/// cache update (host mirror remains authoritative).
pub const GPU_KV_MAX_SEQ_LEN: usize = 4096;

/// KV pool storage dtype (the p24 f16 pool + the e4m3 follow-on).
/// f16 = the default (128 KiB/token for Llama-8B GQA); e4m3 = opt-in
/// via TEMPURA_KV_DTYPE=e4m3 (64 KiB/token, vLLM's production KV
/// density; 3 mantissa bits -- quality-gated, see docs/status).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype {
    F32,
    F16,
    E4M3,
}

impl KvDtype {
    /// Serving-pool dtype from the environment: TEMPURA_KV_DTYPE
    /// (f32|f16|e4m3, default f16), with TEMPURA_KV_F32=1 as an f32
    /// alias (existing status references / bisect workflows use it).
    /// ONE parse point shared by model.rs new_kv_pool and eval.rs.
    pub fn from_env() -> Self {
        if std::env::var("TEMPURA_KV_F32").as_deref() == Ok("1") {
            return KvDtype::F32;
        }
        match std::env::var("TEMPURA_KV_DTYPE").as_deref() {
            Ok("f32") => KvDtype::F32,
            Ok("e4m3") => KvDtype::E4M3,
            // f16 and anything unset/unknown: the default.  An invalid
            // value falls back to the default rather than failing the
            // server -- the banner prints the resolved dtype.
            _ => KvDtype::F16,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            KvDtype::F32 => "f32",
            KvDtype::F16 => "f16",
            KvDtype::E4M3 => "e4m3",
        }
    }
}

/// Per-layer pool/device tensors, dtype-tagged (p24 f16 serving pool;
/// e4m3 follow-on).  Raw-pointer sites are dtype-ERASED (a pointer is a
/// pointer); only the byte-offset math and the typed readers branch on
/// the variant.
pub enum LayerTensors {
    F32(Vec<Arc<Tensor<f32>>>),
    F16(Vec<Arc<Tensor<half::f16>>>),
    E4M3(Vec<Arc<Tensor<u8>>>),
}

impl LayerTensors {
    pub fn kv_dtype(&self) -> KvDtype {
        match self {
            LayerTensors::F32(_) => KvDtype::F32,
            LayerTensors::F16(_) => KvDtype::F16,
            LayerTensors::E4M3(_) => KvDtype::E4M3,
        }
    }
    pub fn is_f16(&self) -> bool {
        matches!(self, LayerTensors::F16(_))
    }
    pub fn elem_size(&self) -> usize {
        match self {
            LayerTensors::F32(_) => std::mem::size_of::<f32>(),
            LayerTensors::F16(_) => std::mem::size_of::<half::f16>(),
            LayerTensors::E4M3(_) => std::mem::size_of::<u8>(),
        }
    }
    pub fn ptr(&self, layer: usize) -> u64 {
        match self {
            LayerTensors::F32(v) => v[layer].device_pointer().cu_deviceptr(),
            LayerTensors::F16(v) => v[layer].device_pointer().cu_deviceptr(),
            LayerTensors::E4M3(v) => v[layer].device_pointer().cu_deviceptr(),
        }
    }
    pub fn f32_tensor(&self, layer: usize) -> Option<Arc<Tensor<f32>>> {
        match self {
            LayerTensors::F32(v) => Some(Arc::clone(&v[layer])),
            LayerTensors::F16(_) | LayerTensors::E4M3(_) => None,
        }
    }
    pub fn f16_tensor(&self, layer: usize) -> Option<Arc<Tensor<half::f16>>> {
        match self {
            LayerTensors::F32(_) | LayerTensors::E4M3(_) => None,
            LayerTensors::F16(v) => Some(Arc::clone(&v[layer])),
        }
    }
    pub fn e4m3_tensor(&self, layer: usize) -> Option<Arc<Tensor<u8>>> {
        match self {
            LayerTensors::F32(_) | LayerTensors::F16(_) => None,
            LayerTensors::E4M3(v) => Some(Arc::clone(&v[layer])),
        }
    }

    /// A new LayerTensors sharing the SAME device tensors (Arc clones).
    /// Used by [`CutileKvCache::spec_row_views`].
    pub fn share(&self) -> LayerTensors {
        match self {
            LayerTensors::F32(v) => LayerTensors::F32(v.clone()),
            LayerTensors::F16(v) => LayerTensors::F16(v.clone()),
            LayerTensors::E4M3(v) => LayerTensors::E4M3(v.clone()),
        }
    }
}

/// Paged-attention pool views, dtype-tagged (see [`LayerTensors`]).
pub enum PagedAttnViews {
    F32(Arc<Tensor<f32>>, Arc<Tensor<f32>>),
    F16(Arc<Tensor<half::f16>>, Arc<Tensor<half::f16>>),
    E4M3(Arc<Tensor<u8>>, Arc<Tensor<u8>>),
}

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
    /// F16 variant = the p24 f16 serving pool (halved KV density).
    k_device: LayerTensors,
    v_device: LayerTensors,
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
    /// Bits-checksum of the host mirror's live prefix at swap_out --
    /// swap_in verifies its device readback against it (p24 A/B debug).
    swap_checksum: Option<u64>,
    /// Device scalar holding the current kv_len for the paged attention
    /// kernel (read at EXECUTION time, not launch time -- CUDA-graph
    /// safe, the kv_scatter_write device-offset pattern).  Lazily
    /// allocated once; the VALUE is updated via async H2D before each
    /// paged launch.
    kv_len_dev: Option<Arc<Tensor<i32>>>,
    /// Lazily-allocated f16 staging for the append path: the f32 source is
    /// cast into it (cutile_cast_f32_f16), then D2D-copied into the pool.
    /// Resized to exactly ceil(n/CAST_B)*CAST_B per use (stable within a
    /// phase: decode = kv_dim, prefill = chunk*kv_dim).
    f16_stage: Option<Arc<Tensor<half::f16>>>,
    /// Same staging contract for e4m3 pools (cutile_cast_f32_e4m3).
    e4m3_stage: Option<Arc<Tensor<u8>>>,
    /// Zero-padded f32 staging for the d2d_into_pool cast path when
    /// num_elems is not a CAST_B multiple (kv_dim-512 models, odd token
    /// counts).  See [`Self::padded_cast_src`].
    f32_stage: Option<Arc<Tensor<f32>>>,
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
            k_device: LayerTensors::F32(k_device),
            v_device: LayerTensors::F32(v_device),
            f16_stage: None,
            e4m3_stage: None,
            f32_stage: None,
            gpu_max_seq_len,
            host_stale: false,
            block_table_dev: None,
            swap_checksum: None,
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
            let (k_device, v_device) = match p.dtype() {
                KvDtype::F16 => (
                    LayerTensors::F16(p.k_pool_tensors_f16()),
                    LayerTensors::F16(p.v_pool_tensors_f16()),
                ),
                KvDtype::E4M3 => (
                    LayerTensors::E4M3(p.k_pool_tensors_e4m3()),
                    LayerTensors::E4M3(p.v_pool_tensors_e4m3()),
                ),
                KvDtype::F32 => (
                    LayerTensors::F32(p.k_pool_tensors()),
                    LayerTensors::F32(p.v_pool_tensors()),
                ),
            };
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
            swap_checksum: None,
            kv_len_dev: None,
            f16_stage: None,
            e4m3_stage: None,
            f32_stage: None,
            pool: Some(pool),
            base_page,
            run_pages,
            capacity_gen: 0,
        })
    }

    /// Token capacity of the current run (pooled mode; 0 = dedicated).
    pub fn run_pages(&self) -> usize {
        self.run_pages
    }

    /// Pages free in the shared pool (None = dedicated mode).  Lets the
    /// scheduler check swap-in headroom before attempting it.
    pub fn pool_free_pages(&self) -> Option<usize> {
        self.pool
            .as_ref()
            .and_then(|p| p.lock().ok().map(|p| p.free_pages()))
    }

    /// Pages a run must hold to cover `tokens` (page granularity).
    pub fn pages_for_tokens(tokens: usize) -> usize {
        tokens.div_ceil(tempura_kernels::cutile_flash_attention::PAGED_BLOCK_SIZE)
    }

    pub fn base_page(&self) -> usize {
        self.base_page
    }

    /// Byte offset of the run's token 0 within the pool tensors (0 in
    /// dedicated mode).
    fn kv_base_byte_offset(&self) -> u64 {
        let block_size = tempura_kernels::cutile_flash_attention::PAGED_BLOCK_SIZE;
        (self.base_page * block_size * self.kv_dim) as u64 * self.k_device.elem_size() as u64
    }

    /// Grow the run to hold at least `tokens` tokens (pooled mode only).
    ///
    /// Extends in place when the adjacent pages are free (keeps baked
    /// CUDA-graph pointers stable); otherwise reallocates (D2D copy of
    /// the live prefix per layer, free the old run -- callers' captured
    /// graphs detect the pointer change and re-capture).  Growth uses a
    /// 2x factor to amortize.
    pub fn ensure_capacity(&mut self, tokens: usize) -> Result<(), KernelError> {
        self.try_ensure_capacity(tokens).map_err(|e| match e {
            EnsureCapacityError::Exhausted(_) => {
                KernelError::InvalidConfig("KV pool exhausted on growth".into())
            }
            EnsureCapacityError::Kernel(e) => e,
        })
    }

    /// [`Self::ensure_capacity`] with a recoverable error: on pool
    /// exhaustion returns the pages needed and the pages free, so the
    /// caller (the batch scheduler) can pick a swap victim and retry.
    pub fn try_ensure_capacity(&mut self, tokens: usize) -> Result<(), EnsureCapacityError> {
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
            .map_err(|_| EnsureCapacityError::Exhausted(PoolExhausted { need_pages, free_pages: 0 }))?;
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
        let new_run = pool.alloc_pages(target_pages).ok_or(
            EnsureCapacityError::Exhausted(PoolExhausted {
                need_pages: target_pages,
                free_pages: pool.free_pages(),
            }),
        )?;
        let old_base = self.base_page;
        let elem = self.k_device.elem_size() as u64;
        let live_elems = self.seq_len * self.kv_dim;
        for layer in 0..self.n_layers {
            let k_src = self.k_device.ptr(layer)
                + (old_base * block_size * self.kv_dim) as u64 * elem;
            let v_src = self.v_device.ptr(layer)
                + (old_base * block_size * self.kv_dim) as u64 * elem;
            let k_dst = self.k_device.ptr(layer)
                + (new_run.start * block_size * self.kv_dim) as u64 * elem;
            let v_dst = self.v_device.ptr(layer)
                + (new_run.start * block_size * self.kv_dim) as u64 * elem;
            if live_elems > 0 {
                let copy = |dst: u64, src: u64| -> Result<(), EnsureCapacityError> {
                    let r = match self.k_device.kv_dtype() {
                        KvDtype::F16 => unsafe {
                            cuda_core::memcpy_dtod_async::<half::f16>(dst, src, live_elems, &self.stream)
                        },
                        KvDtype::E4M3 => unsafe {
                            cuda_core::memcpy_dtod_async::<u8>(dst, src, live_elems, &self.stream)
                        },
                        KvDtype::F32 => unsafe {
                            cuda_core::memcpy_dtod_async::<f32>(dst, src, live_elems, &self.stream)
                        },
                    };
                    r.map_err(|e| {
                        EnsureCapacityError::Kernel(KernelError::InvalidConfig(e.to_string()))
                    })
                };
                copy(k_dst, k_src)?;
                copy(v_dst, v_src)?;
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
        self.d2d_into_pool(layer, k_new, v_new, num_elems)?;

        Ok(())
    }

    /// Copy `num_elems` f32 K/V elements into the pool at the current
    /// offset (dtype-aware): f32 pools take a straight D2D; f16/e4m3
    /// pools cast through the staging tensor first
    /// (cutile_cast_f32_f16 / cutile_cast_f32_e4m3).
    /// Zero-padded f32 staging for the cast path in [`Self::d2d_into_pool`]:
    /// D2D `num_elems` elements from `src` into a cached `n_pad`-sized
    /// buffer.  The (possibly stale) pad tail is cast but never copied
    /// into the pool -- only `num_elems` elements are.
    fn padded_cast_src(
        &mut self,
        src: &Arc<Tensor<f32>>,
        num_elems: usize,
        n_pad: usize,
    ) -> Result<Arc<Tensor<f32>>, KernelError> {
        let st = match self.f32_stage.take() {
            Some(t) if t.size() == n_pad => t,
            _ => Arc::new(
                api::zeros::<f32>(&[n_pad])
                    .sync_on(&self.stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    .into(),
            ),
        };
        unsafe {
            cuda_core::memcpy_dtod_async::<f32>(
                st.device_pointer().cu_deviceptr(),
                src.device_pointer().cu_deviceptr(),
                num_elems,
                &self.stream,
            )
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }
        self.f32_stage = Some(Arc::clone(&st));
        Ok(st)
    }

    fn d2d_into_pool(        &mut self,
        layer: usize,
        k_new: &Arc<Tensor<f32>>,
        v_new: &Arc<Tensor<f32>>,
        num_elems: usize,
    ) -> Result<(), KernelError> {
        use cuda_core::sys::CUdeviceptr;

        let offset_elems = self.seq_len * self.kv_dim;
        let elem = self.k_device.elem_size() as u64;
        let base_off = self.kv_base_byte_offset();

        if self.k_device.kv_dtype() == KvDtype::F32 {
            let k_src: CUdeviceptr = k_new.device_pointer().cu_deviceptr();
            let v_src: CUdeviceptr = v_new.device_pointer().cu_deviceptr();
            let k_dst: CUdeviceptr = self.k_device.ptr(layer) + base_off + offset_elems as u64 * elem;
            let v_dst: CUdeviceptr = self.v_device.ptr(layer) + base_off + offset_elems as u64 * elem;
            unsafe {
                cuda_core::memcpy_dtod_async::<f32>(k_dst, k_src, num_elems, &self.stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
                cuda_core::memcpy_dtod_async::<f32>(v_dst, v_src, num_elems, &self.stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            }
            return Ok(());
        }

        // f16 / e4m3 pools: cast each source through the staging tensor,
        // then D2D the cast result into the pool.  The cast is
        // stream-ordered before the D2D; the staging is reused across
        // calls (exact-size per use).
        let n_pad = num_elems.div_ceil(tempura_kernels::cutile_elementwise::CAST_B)
            * tempura_kernels::cutile_elementwise::CAST_B;
        // The cast grid pads n up to CAST_B and the kernel validates the
        // SRC against n_pad: a src sized exactly num_elems fails when
        // num_elems is not a CAST_B multiple (kv_dim-512 models with odd
        // token counts -- e.g. the Llama-3.2-1B draft; kv_dim-1024 targets
        // are always multiples).  Route such srcs through a zero-padded
        // f32 staging D2D first; the stale pad tail is never copied out
        // into the pool (only num_elems elements are).
        let pad_needed = num_elems % tempura_kernels::cutile_elementwise::CAST_B != 0;
        let is_e4m3 = self.k_device.kv_dtype() == KvDtype::E4M3;
        if is_e4m3 {
            let mut stage = match self.e4m3_stage.take() {
                Some(t) if t.size() == n_pad => t,
                _ => Arc::new(
                    api::zeros::<u8>(&[n_pad])
                        .sync_on(&self.stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                        .into(),
                ),
            };
            for src in [k_new, v_new] {
                let cast_src = if pad_needed && src.size() < n_pad {
                    self.padded_cast_src(src, num_elems, n_pad)?
                } else {
                    Arc::clone(src)
                };
                stage = tempura_kernels::cutile_elementwise::cast_f32_to_e4m3_dev(
                    &cast_src, stage, num_elems, &self.stream,
                )
                .map_err(|e| {
                    KernelError::InvalidConfig(format!(
                        "d2d_into_pool e4m3 cast ({}): layer={layer} seq_len={} kv_dim={} \
                         num_elems={num_elems} src_size={} err={e:?}",
                        if Arc::ptr_eq(src, k_new) { "K" } else { "V" },
                        self.seq_len,
                        self.kv_dim,
                        src.size(),
                    ))
                })?;
                let dst: CUdeviceptr = if Arc::ptr_eq(src, k_new) {
                    self.k_device.ptr(layer)
                } else {
                    self.v_device.ptr(layer)
                } + base_off
                    + offset_elems as u64 * elem;
                unsafe {
                    cuda_core::memcpy_dtod_async::<u8>(
                        dst,
                        stage.device_pointer().cu_deviceptr(),
                        num_elems,
                        &self.stream,
                    )
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
                }
            }
            self.e4m3_stage = Some(stage);
        } else {
            let mut stage = match self.f16_stage.take() {
                Some(t) if t.size() == n_pad => t,
                _ => Arc::new(
                    api::zeros::<half::f16>(&[n_pad])
                        .sync_on(&self.stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                        .into(),
                ),
            };
            for src in [k_new, v_new] {
                let cast_src = if pad_needed && src.size() < n_pad {
                    self.padded_cast_src(src, num_elems, n_pad)?
                } else {
                    Arc::clone(src)
                };
                stage = tempura_kernels::cutile_elementwise::cast_f32_to_f16_dev(
                    &cast_src, stage, num_elems, &self.stream,
                )
                .map_err(|e| {
                    KernelError::InvalidConfig(format!(
                        "d2d_into_pool f16 cast ({}): layer={layer} seq_len={} kv_dim={} \
                         num_elems={num_elems} src_size={} err={e:?}",
                        if Arc::ptr_eq(src, k_new) { "K" } else { "V" },
                        self.seq_len,
                        self.kv_dim,
                        src.size(),
                    ))
                })?;
                let dst: CUdeviceptr = if Arc::ptr_eq(src, k_new) {
                    self.k_device.ptr(layer)
                } else {
                    self.v_device.ptr(layer)
                } + base_off
                    + offset_elems as u64 * elem;
                unsafe {
                    cuda_core::memcpy_dtod_async::<half::f16>(
                        dst,
                        stage.device_pointer().cu_deviceptr(),
                        num_elems,
                        &self.stream,
                    )
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
                }
            }
            self.f16_stage = Some(stage);
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
        self.d2d_into_pool(layer, k_new, v_new, num_elems)?;

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
        let elem = self.k_device.elem_size();
        let dtype = self.k_device.kv_dtype();
        // f16/e4m3 pools download into a raw-bits staging and convert on
        // the host via the SHARED e4m3/f16 helpers (the DRAM-tier mirror
        // stays f32 for now).
        let mut f16_tmp: Vec<u16> = if dtype == KvDtype::F16 { vec![0u16; len] } else { Vec::new() };
        let mut e4m3_tmp: Vec<u8> = if dtype == KvDtype::E4M3 { vec![0u8; len] } else { Vec::new() };
        for layer in 0..self.n_layers {
            // Offset D2H (raw memcpy): pooled mode must NOT download the
            // whole pool tensor (to_host_vec) -- the pool can be tens of
            // GB while the live prefix is small.
            let k_src = self.k_device.ptr(layer) + base_off;
            let v_src = self.v_device.ptr(layer) + base_off;
            if dtype == KvDtype::F16 {
                for (src, mirror) in [(k_src, &mut self.k_host), (v_src, &mut self.v_host)] {
                    unsafe {
                        let rc = cuda_core::sys::cuMemcpyDtoHAsync_v2(
                            f16_tmp.as_mut_ptr() as *mut _,
                            src,
                            len * elem,
                            self.stream.cu_stream(),
                        );
                        if rc != 0 {
                            return Err(KernelError::InvalidConfig(format!(
                                "sync_host f16 D2H rc={rc}"
                            )));
                        }
                    }
                    self.stream
                        .synchronize()
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
                    let dst = &mut mirror[layer][..len];
                    for (d, &h) in dst.iter_mut().zip(f16_tmp.iter()) {
                        *d = half::f16::from_bits(h).to_f32();
                    }
                }
                continue;
            }
            if dtype == KvDtype::E4M3 {
                for (src, mirror) in [(k_src, &mut self.k_host), (v_src, &mut self.v_host)] {
                    unsafe {
                        let rc = cuda_core::sys::cuMemcpyDtoHAsync_v2(
                            e4m3_tmp.as_mut_ptr() as *mut _,
                            src,
                            len * elem,
                            self.stream.cu_stream(),
                        );
                        if rc != 0 {
                            return Err(KernelError::InvalidConfig(format!(
                                "sync_host e4m3 D2H rc={rc}"
                            )));
                        }
                    }
                    self.stream
                        .synchronize()
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
                    let dst = &mut mirror[layer][..len];
                    for (d, &b) in dst.iter_mut().zip(e4m3_tmp.iter()) {
                        *d = tempura_kernels::e4m3::e4m3_bits_to_f32(b);
                    }
                }
                continue;
            }
            unsafe {
                let rc = cuda_core::sys::cuMemcpyDtoHAsync_v2(
                    self.k_host[layer].as_mut_ptr() as *mut _,
                    k_src,
                    len * elem,
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
                    len * elem,
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

    /// Speculative-decode verification views over THIS run (greedy spec
    /// verify): `seq_lens.len()` lightweight handles that all address the
    /// SAME pool run, with view r reporting `seq_lens[r]` -- so the batch
    /// decode path derives per-row KV write offsets (`seq_lens[r]`) and
    /// attention lengths (`seq_lens[r] + 1`) that make the rows act as one
    /// sequence's consecutive positions (row r attends over exactly
    /// `base + r + 1` tokens -- causal verification of a draft block,
    /// computed with the decode-class kernels, per-row bit-identical to
    /// plain single-token decode).
    ///
    /// The views own NOTHING: `pool = None` makes `try_ensure_capacity` a
    /// no-op and `run_pages = 0` makes `Drop` a no-op -- the owner must
    /// `ensure_capacity(base + rows)` BEFORE the call and advance/rollback
    /// its own `seq_len` after.  Host mirrors are empty and `host_stale`
    /// is false: the batch decode path never reads them (device attention
    /// only), and a fresh view therefore never triggers a mirror sync.
    pub fn spec_row_views(&self, seq_lens: &[usize]) -> Vec<CutileKvCache> {
        seq_lens
            .iter()
            .map(|&sl| CutileKvCache {
                k_host: vec![Vec::new(); self.n_layers],
                v_host: vec![Vec::new(); self.n_layers],
                seq_len: sl,
                n_layers: self.n_layers,
                kv_dim: self.kv_dim,
                max_seq_len: self.max_seq_len,
                stream: Arc::clone(&self.stream),
                k_device: self.k_device.share(),
                v_device: self.v_device.share(),
                gpu_max_seq_len: self.gpu_max_seq_len,
                host_stale: false,
                pool: None,
                base_page: self.base_page,
                run_pages: 0,
                capacity_gen: self.capacity_gen,
                block_table_dev: None,
                swap_checksum: None,
                kv_len_dev: None,
                f16_stage: None,
                e4m3_stage: None,
            f32_stage: None,
            })
            .collect()
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
    pub fn k_device(&self, layer: usize) -> Option<Arc<Tensor<f32>>> {
        self.k_device.f32_tensor(layer)
    }

    pub fn v_device(&self, layer: usize) -> Option<Arc<Tensor<f32>>> {
        self.v_device.f32_tensor(layer)
    }

    /// Return both K and V GPU device tensors for `layer`.
    ///
    /// Each tensor is flat `[gpu_max_seq_len * kv_dim]` in interleaved
    /// `[token, kv_head, head_dim]` layout.  Only the first `seq_len * kv_dim`
    /// elements are valid; the rest are zero-initialised.
    /// None for an f16 pool (callers must use the f16 kernels).
    pub fn kv_device_tensors(&self, layer: usize) -> Option<(Arc<Tensor<f32>>, Arc<Tensor<f32>>)> {
        // Single-client invariant (see the `pool` field docs): pooled
        // tensors are handed out only while the run sits at pool base.
        // A second pool client must make the raw-pointer call sites
        // base-offset aware first.
        debug_assert!(self.pool.is_none() || self.base_page == 0);
        Some((
            self.k_device.f32_tensor(layer)?,
            self.v_device.f32_tensor(layer)?,
        ))
    }

    /// Whether this cache's pool tensors hold f16 (p24 f16 serving pool).
    pub fn kv_f16(&self) -> bool {
        self.k_device.is_f16()
    }

    /// This cache's pool storage dtype.
    pub fn kv_dtype(&self) -> KvDtype {
        self.k_device.kv_dtype()
    }

    /// Whether this cache's pool tensors hold f32 (the only dtype the
    /// f32 attention/graph kernels can read).
    pub fn kv_f32(&self) -> bool {
        self.k_device.kv_dtype() == KvDtype::F32
    }

    /// Base-shifted raw K/V pointers for this cache's RUN (p20 4b-3):
    /// pool tensor base + run byte offset.  Multi-client batch paths use
    /// these; the whole-tensor `kv_device_tensors` pointers point at the
    /// POOL start and are only valid for base-0 runs.  Dtype-erased:
    /// f16 pools address f16 elements.
    pub fn kv_device_base_ptrs(&self, layer: usize) -> (u64, u64) {
        let off = self.kv_base_byte_offset();
        (self.k_device.ptr(layer) + off, self.v_device.ptr(layer) + off)
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
            PagedAttnViews,
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
        // View rows from the TENSOR's actual element count: in pooled
        // mode the tensor is the whole pool (bigger than the run).
        let tensor_elems = match &self.k_device {
            LayerTensors::F32(v) => v[layer]
                .shape()
                .iter()
                .map(|&d| d as usize)
                .product::<usize>(),
            LayerTensors::F16(v) => v[layer]
                .shape()
                .iter()
                .map(|&d| d as usize)
                .product::<usize>(),
            LayerTensors::E4M3(v) => v[layer]
                .shape()
                .iter()
                .map(|&d| d as usize)
                .product::<usize>(),
        };
        let rows = tensor_elems / head_dim;
        let views = match &self.k_device {
            LayerTensors::F32(_) => PagedAttnViews::F32(
                self.k_device
                    .f32_tensor(layer)
                    .unwrap()
                    .reshape(&[rows, head_dim])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
                self.v_device
                    .f32_tensor(layer)
                    .unwrap()
                    .reshape(&[rows, head_dim])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
            ),
            LayerTensors::F16(_) => PagedAttnViews::F16(
                self.k_device
                    .f16_tensor(layer)
                    .unwrap()
                    .reshape(&[rows, head_dim])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
                self.v_device
                    .f16_tensor(layer)
                    .unwrap()
                    .reshape(&[rows, head_dim])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
            ),
            LayerTensors::E4M3(_) => PagedAttnViews::E4M3(
                self.k_device
                    .e4m3_tensor(layer)
                    .unwrap()
                    .reshape(&[rows, head_dim])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
                self.v_device
                    .e4m3_tensor(layer)
                    .unwrap()
                    .reshape(&[rows, head_dim])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
            ),
        };

        // Affine table: block i of this run is physical page base+i.  The
        // identity table was only valid for base-0 runs -- with several
        // pooled sessions every non-base run attended to the FIRST run's
        // KV through it (identical mush across clients, found by the
        // p24 streaming A/B).
        if self.block_table_dev.is_none() {
            let max_blocks = self
                .gpu_max_seq_len
                .div_ceil(tempura_kernels::cutile_flash_attention::PAGED_BLOCK_SIZE);
            let table: Vec<i32> = (0..max_blocks)
                .map(|i| (self.base_page + i) as i32)
                .collect();
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
            views,
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

    // ========================================================================
    // Tiering: DRAM fallback (p20 phase 4 / the p24-kv-swap mechanism)
    // ========================================================================

    /// Swap OUT to the DRAM tier: sync the host mirror (the DRAM copy)
    /// with the live prefix, then free the GPU run back to the pool.
    ///
    /// The sequence's KV now lives ONLY in the host mirror; another
    /// sequence can use the GPU pages.  Call [`Self::swap_in`] before any
    /// further GPU attention on this cache.  Attention cannot read a
    /// partially-swapped sequence (it needs the full prefix every step),
    /// so the granularity is the WHOLE RUN -- see the p24-kv-swap notes.
    pub fn swap_out(&mut self) -> Result<(), KernelError> {
        let Some(pool_arc) = self.pool.clone() else {
            // Dedicated caches have no pool to return pages to; the host
            // mirror is maintained by append_device already.
            return self.sync_host_from_device(self.seq_len);
        };
        // Sync first (stream-ordered D2H of the live prefix at the run
        // base), THEN free -- the pages must not be reused before the
        // copy completes, which stream ordering guarantees on this
        // cache's stream.
        self.sync_host_from_device(self.seq_len)?;
        self.swap_checksum = Some(self.mirror_bits_checksum());
        let freed_pages = self.run_pages;
        let cs = self.swap_checksum.unwrap();
        eprintln!(
            "[swap] out: seq={} base={} pages={} mirror_cs={cs:016x}",
            self.seq_len, self.base_page, freed_pages
        );
        let mut pool = pool_arc
            .lock()
            .map_err(|_| KernelError::InvalidConfig("pool mutex poisoned".into()))?;
        pool.free_pages_alloc(crate::gpu_page_pool::Run {
            start: self.base_page,
            pages: self.run_pages,
        });
        self.base_page = 0;
        self.run_pages = 0;
        self.gpu_max_seq_len = 0;
        self.capacity_gen += 1;
        self.block_table_dev = None;
        Ok(())
    }

    /// Swap IN from the DRAM tier: allocate a fresh run and upload the
    /// host mirror's live prefix into it.  The block table is rebuilt on
    /// the next paged attention call (contents = the NEW run's affine
    /// table); captured graphs re-capture via the capacity-gen check
    /// (the pointers changed).
    pub fn swap_in(&mut self, tokens: usize) -> Result<(), KernelError> {
        let Some(pool_arc) = self.pool.clone() else {
            return Ok(()); // dedicated: nothing to do (mirror is live)
        };
        let block_size = tempura_kernels::cutile_flash_attention::PAGED_BLOCK_SIZE;
        let want = tokens.max(self.seq_len);
        let need_pages = want.div_ceil(block_size);
        let mut pool = pool_arc
            .lock()
            .map_err(|_| KernelError::InvalidConfig("pool mutex poisoned".into()))?;
        let run = pool
            .alloc_pages(need_pages)
            .ok_or_else(|| KernelError::InvalidConfig("KV pool exhausted on swap-in".into()))?;
        let f32_bytes = std::mem::size_of::<f32>() as u64;
        let live_elems = self.seq_len * self.kv_dim;
        let dtype = self.k_device.kv_dtype();
        // f16/e4m3 pools: the f32 mirror is quantized to the pool's bits
        // on the host via the SHARED helpers (matching the append path's
        // cast semantics -- e4m3 RNE + saturation + subnormal flush),
        // then uploaded.
        let mut f16_tmp: Vec<u16> = if dtype == KvDtype::F16 && live_elems > 0 {
            vec![0u16; live_elems]
        } else {
            Vec::new()
        };
        let mut e4m3_tmp: Vec<u8> = if dtype == KvDtype::E4M3 && live_elems > 0 {
            vec![0u8; live_elems]
        } else {
            Vec::new()
        };
        for layer in 0..self.n_layers {
            let elem = self.k_device.elem_size() as u64;
            let k_dst = self.k_device.ptr(layer)
                + (run.start * block_size * self.kv_dim) as u64 * elem;
            let v_dst = self.v_device.ptr(layer)
                + (run.start * block_size * self.kv_dim) as u64 * elem;
            if live_elems > 0 {
                if dtype == KvDtype::F16 {
                    for (dst, mirror) in [(k_dst, &self.k_host), (v_dst, &self.v_host)] {
                        for (d, &s) in f16_tmp.iter_mut().zip(mirror[layer][..live_elems].iter())
                        {
                            *d = half::f16::from_f32(s).to_bits();
                        }
                        unsafe {
                            cuda_core::sys::cuMemcpyHtoDAsync_v2(
                                dst,
                                f16_tmp.as_ptr() as *const _,
                                live_elems * elem as usize,
                                self.stream.cu_stream(),
                            );
                        }
                    }
                } else if dtype == KvDtype::E4M3 {
                    for (dst, mirror) in [(k_dst, &self.k_host), (v_dst, &self.v_host)] {
                        for (d, &s) in e4m3_tmp.iter_mut().zip(mirror[layer][..live_elems].iter())
                        {
                            *d = tempura_kernels::e4m3::f32_to_e4m3_bits(s);
                        }
                        unsafe {
                            cuda_core::sys::cuMemcpyHtoDAsync_v2(
                                dst,
                                e4m3_tmp.as_ptr() as *const _,
                                live_elems * elem as usize,
                                self.stream.cu_stream(),
                            );
                        }
                    }
                } else {
                    unsafe {
                        cuda_core::sys::cuMemcpyHtoDAsync_v2(
                            k_dst,
                            self.k_host[layer].as_ptr() as *const _,
                            live_elems * f32_bytes as usize,
                            self.stream.cu_stream(),
                        );
                        cuda_core::sys::cuMemcpyHtoDAsync_v2(
                            v_dst,
                            self.v_host[layer].as_ptr() as *const _,
                            live_elems * f32_bytes as usize,
                            self.stream.cu_stream(),
                        );
                    }
                }
            }
        }
        self.base_page = run.start;
        self.run_pages = run.pages;
        self.gpu_max_seq_len = run.pages * block_size;
        self.capacity_gen += 1;
        self.block_table_dev = None;
        self.host_stale = false;
        // Verify the uploaded prefix bit-for-bit against the swap_out
        // checksum (p24 A/B debug: the swapped sessions diverged by one
        // token at resume -- this pins whether the KV data survived).
        if let Some(cs_out) = self.swap_checksum {
            let cs_dev = self.device_prefix_bits_checksum(run.start, self.seq_len);
            eprintln!(
                "[swap] in: seq={} base={} pages={} mirror_cs={cs_out:016x} device_cs={cs_dev:016x} {}",
                self.seq_len,
                run.start,
                run.pages,
                if cs_out == cs_dev { "EXACT" } else { "MISMATCH" }
            );
        }
        Ok(())
    }

    /// Sum of the raw f32 bits of the host mirror's live prefix (all
    /// layers, K and V) -- a bit-exact fingerprint for swap verification.
    fn mirror_bits_checksum(&self) -> u64 {
        let live = self.seq_len * self.kv_dim;
        let mut cs: u64 = 0;
        for layer in 0..self.n_layers {
            for v in &self.k_host[layer][..live] {
                cs = cs.wrapping_add(v.to_bits() as u64);
            }
            for v in &self.v_host[layer][..live] {
                cs = cs.wrapping_add(v.to_bits() as u64);
            }
        }
        cs
    }

    /// Same fingerprint, read back from the device run at `base_page`.
    /// f16/e4m3 pools: the raw bits are converted back to f32 BEFORE
    /// summing, so the value matches [`Self::mirror_bits_checksum`] (the
    /// mirror holds the dequantized values; the e4m3 conversion is the
    /// SHARED reader -- exact for normals, subnormals flush, same as the
    /// mirror sync).
    fn device_prefix_bits_checksum(&self, base_page: usize, tokens: usize) -> u64 {
        let block_size = tempura_kernels::cutile_flash_attention::PAGED_BLOCK_SIZE;
        let elem = self.k_device.elem_size();
        let live = tokens * self.kv_dim;
        let dtype = self.k_device.kv_dtype();
        let mut buf32 = vec![0.0f32; live];
        let mut buf16 = if dtype == KvDtype::F16 { vec![0u16; live] } else { Vec::new() };
        let mut buf8 = if dtype == KvDtype::E4M3 { vec![0u8; live] } else { Vec::new() };
        let mut cs: u64 = 0;
        let base_off =
            (base_page * block_size * self.kv_dim) as u64 * elem as u64;
        for layer in 0..self.n_layers {
            for is_k in [true, false] {
                let src = if is_k {
                    self.k_device.ptr(layer) + base_off
                } else {
                    self.v_device.ptr(layer) + base_off
                };
                if dtype == KvDtype::F16 {
                    let rc = unsafe {
                        cuda_core::sys::cuMemcpyDtoHAsync_v2(
                            buf16.as_mut_ptr() as *mut _,
                            src,
                            live * elem,
                            self.stream.cu_stream(),
                        )
                    };
                    if rc != 0 {
                        return u64::MAX;
                    }
                    self.stream.synchronize().ok();
                    for &h in &buf16 {
                        cs = cs.wrapping_add(half::f16::from_bits(h).to_f32().to_bits() as u64);
                    }
                    continue;
                }
                if dtype == KvDtype::E4M3 {
                    let rc = unsafe {
                        cuda_core::sys::cuMemcpyDtoHAsync_v2(
                            buf8.as_mut_ptr() as *mut _,
                            src,
                            live * elem,
                            self.stream.cu_stream(),
                        )
                    };
                    if rc != 0 {
                        return u64::MAX;
                    }
                    self.stream.synchronize().ok();
                    for &b in &buf8 {
                        cs = cs.wrapping_add(
                            tempura_kernels::e4m3::e4m3_bits_to_f32(b).to_bits() as u64,
                        );
                    }
                    continue;
                }
                let rc = unsafe {
                    cuda_core::sys::cuMemcpyDtoHAsync_v2(
                        buf32.as_mut_ptr() as *mut _,
                        src,
                        live * elem,
                        self.stream.cu_stream(),
                    )
                };
                if rc != 0 {
                    return u64::MAX; // unreadable -- report as mismatch
                }
                self.stream.synchronize().ok();
                for v in &buf32 {
                    cs = cs.wrapping_add(v.to_bits() as u64);
                }
            }
        }
        cs
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

impl Drop for CutileKvCache {
    /// Free the run back to the pool (pooled mode).  Without this, a
    /// completed session's pages are lost forever when the pool outlives
    /// the session (the server's process-lifetime pool) -- dropping the
    /// cache only released the pool Arc.  `swap_out` zeroes `run_pages`,
    /// so a swapped-out cache drops without double-freeing.
    ///
    /// Stream ordering makes the free safe against in-flight kernels:
    /// every kernel touching this run was launched on the pool's stream,
    /// and any future writer of these pages is allocated and launched
    /// behind the free on the same stream (the swap_out argument).
    fn drop(&mut self) {
        if self.run_pages == 0 {
            return;
        }
        if let Some(pool_arc) = self.pool.clone() {
            if let Ok(mut pool) = pool_arc.lock() {
                pool.free_pages_alloc(crate::gpu_page_pool::Run {
                    start: self.base_page,
                    pages: self.run_pages,
                });
            }
        }
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
    fn test_spec_row_views_share_run_and_drop_clean() {
        // Spec-verify views: per-row seq_lens, shared run identity, and
        // Drop is a no-op (run_pages == 0) so the owner keeps its pages.
        let config = ModelConfig::tiny_test();
        let stream = make_stream();
        let kv_dim = config.num_kv_heads * config.head_dim;
        let pool = GpuKvPagePool::new_with_dims(
            config.num_layers,
            kv_dim,
            8,
            stream.clone(),
        )
        .unwrap();
        let pool = Arc::new(std::sync::Mutex::new(pool));
        let mut cache = CutileKvCache::new_pooled(&config, pool.clone(), 4, stream).unwrap();
        cache.advance(3);
        let free_before = pool.lock().unwrap().free_pages();
        let views = cache.spec_row_views(&[3, 4, 5]);
        assert_eq!(
            views.iter().map(|v| v.seq_len()).collect::<Vec<_>>(),
            vec![3, 4, 5]
        );
        assert!(views.iter().all(|v| v.base_page() == cache.base_page()));
        assert!(views.iter().all(|v| v.kv_dtype() == cache.kv_dtype()));
        assert!(views.iter().all(|v| v.run_pages() == 0));
        drop(views);
        // Views must not have returned the owner's pages to the pool.
        assert_eq!(pool.lock().unwrap().free_pages(), free_before);
        drop(cache);
        assert!(pool.lock().unwrap().free_pages() > free_before);
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
    /// `[page, token_in_page, kv_head, head_dim]`.  Exactly one of the
    /// f32/f16 vecs is non-empty (the p24 f16 serving pool halves KV
    /// density; the wrapper's own session cache stays f32).
    k_pools: Vec<Arc<Tensor<f32>>>,
    v_pools: Vec<Arc<Tensor<f32>>>,
    k_pools_f16: Vec<Arc<Tensor<half::f16>>>,
    v_pools_f16: Vec<Arc<Tensor<half::f16>>>,
    /// e4m3 pools: raw u8 e4m3 bits (TEMPURA_KV_DTYPE=e4m3).
    k_pools_e4m3: Vec<Arc<Tensor<u8>>>,
    v_pools_e4m3: Vec<Arc<Tensor<u8>>>,
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
        Self::new_with_dims_dtype(n_layers, kv_dim, total_pages, stream, KvDtype::F32)
    }

    /// [`Self::new_with_dims`] with an explicit dtype: F16 halves and
    /// E4M3 quarters the pool bytes (the p24 f16 serving pool + its e4m3
    /// follow-on; the wrapper's own session cache stays f32).
    pub fn new_with_dims_dtype(
        n_layers: usize,
        kv_dim: usize,
        total_pages: usize,
        stream: Arc<Stream>,
        dtype: KvDtype,
    ) -> Result<Self, KernelError> {
        let block_size = tempura_kernels::cutile_flash_attention::PAGED_BLOCK_SIZE;
        let page_stride = block_size * kv_dim;
        let layer_elems = total_pages * page_stride;
        let mut k_pools = Vec::new();
        let mut v_pools = Vec::new();
        let mut k_pools_f16 = Vec::new();
        let mut v_pools_f16 = Vec::new();
        let mut k_pools_e4m3 = Vec::new();
        let mut v_pools_e4m3 = Vec::new();
        for _ in 0..n_layers {
            match dtype {
                KvDtype::F16 => {
                    let k: Arc<Tensor<half::f16>> = api::zeros::<half::f16>(&[layer_elems])
                        .sync_on(&stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                        .into();
                    let v: Arc<Tensor<half::f16>> = api::zeros::<half::f16>(&[layer_elems])
                        .sync_on(&stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                        .into();
                    k_pools_f16.push(k);
                    v_pools_f16.push(v);
                }
                KvDtype::E4M3 => {
                    let k: Arc<Tensor<u8>> = api::zeros::<u8>(&[layer_elems])
                        .sync_on(&stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                        .into();
                    let v: Arc<Tensor<u8>> = api::zeros::<u8>(&[layer_elems])
                        .sync_on(&stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                        .into();
                    k_pools_e4m3.push(k);
                    v_pools_e4m3.push(v);
                }
                KvDtype::F32 => {
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
            }
        }
        Ok(Self {
            k_pools,
            v_pools,
            k_pools_f16,
            v_pools_f16,
            k_pools_e4m3,
            v_pools_e4m3,
            alloc: crate::gpu_page_pool::PageRunAllocator::new(total_pages),
            page_stride,
            block_size,
            kv_dim,
            stream,
        })
    }

    /// This pool's storage dtype.
    pub fn dtype(&self) -> KvDtype {
        if !self.k_pools.is_empty() {
            KvDtype::F32
        } else if !self.k_pools_f16.is_empty() {
            KvDtype::F16
        } else {
            KvDtype::E4M3
        }
    }

    /// Whether this pool stores f16 KV.
    pub fn is_f16(&self) -> bool {
        !self.k_pools_f16.is_empty()
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
    /// Empty for an f16 pool -- use [`Self::k_pool_tensors_f16`].
    pub fn k_pool_tensors(&self) -> Vec<Arc<Tensor<f32>>> {
        self.k_pools.clone()
    }

    pub fn v_pool_tensors(&self) -> Vec<Arc<Tensor<f32>>> {
        self.v_pools.clone()
    }

    /// Per-layer f16 pool tensors (empty for an f32/e4m3 pool).
    pub fn k_pool_tensors_f16(&self) -> Vec<Arc<Tensor<half::f16>>> {
        self.k_pools_f16.clone()
    }

    pub fn v_pool_tensors_f16(&self) -> Vec<Arc<Tensor<half::f16>>> {
        self.v_pools_f16.clone()
    }

    /// Per-layer e4m3 (u8) pool tensors (empty for f32/f16 pools).
    pub fn k_pool_tensors_e4m3(&self) -> Vec<Arc<Tensor<u8>>> {
        self.k_pools_e4m3.clone()
    }

    pub fn v_pool_tensors_e4m3(&self) -> Vec<Arc<Tensor<u8>>> {
        self.v_pools_e4m3.clone()
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
        // GpuKvRun carries f32 views -- f16/e4m3 pools use CutileKvCache::
        // new_pooled + the dtype twins instead.
        if self.dtype() != KvDtype::F32 {
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
