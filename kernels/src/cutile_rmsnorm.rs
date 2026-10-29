use std::sync::Arc;

fn next_pow2(x: usize) -> usize {
    if x <= 1 {
        return 1;
    }
    1usize << (usize::BITS - (x - 1).leading_zeros()) as usize
}

use crate::{KernelError, RMSNorm, Result, VerificationTier};
use cuda_core::{Device, Stream};
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Reshape, Tensor};

/// Pre-allocated scratch buffers for `forward_device_fused_scratch`.
///
/// Sized for a single token (seq_len=1 / decode path).  Pre-allocating
/// eliminates the three `api::zeros.sync_on` calls that `forward_device_fused`
/// issues on every invocation -- 168 blocking round-trips per decode token
/// (3 allocs x 2 norms x 28 layers) become zero after the first warm-up call.
///
/// `sum_t` and `scale_t` are reclaimed within the same call via
/// `Arc::try_unwrap` on the tensors returned by the scale and apply kernels.
/// `out_t` is returned to the caller as an `Arc`; it is reclaimed at the
/// start of the *next* call once the caller's reference has been dropped.
pub struct RmsnormScratch {
    pub sum_t: Option<Arc<Tensor<f32>>>,
    pub scale_t: Option<Arc<Tensor<f32>>>,
    pub out_t: Option<Arc<Tensor<f32>>>,
}

unsafe impl Send for RmsnormScratch {}
unsafe impl Sync for RmsnormScratch {}

impl RmsnormScratch {
    /// Pre-allocate with a single blocking `sync_on` per buffer -- called once
    /// at model construction.  `seq_len` should be 1 (decode).
    pub fn new(
        stream: &Arc<Stream>,
        seq_len: usize,
        h_pad: usize,
    ) -> std::result::Result<Self, KernelError> {
        let alloc = |n: usize| -> std::result::Result<Arc<Tensor<f32>>, KernelError> {
            Ok(Arc::new(
                api::zeros::<f32>(&[n])
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
            ))
        };
        Ok(Self {
            sum_t: Some(alloc(seq_len)?),
            scale_t: Some(alloc(seq_len)?),
            out_t: Some(alloc(seq_len * h_pad)?),
        })
    }
}

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// Fused RMSNorm: sum-of-squares + rsqrt scale + apply in ONE kernel.
    ///
    /// Replaces the 3-kernel chain (rmsnorm_sum_sq -> cutile_rmsnorm_scale
    /// -> rmsnorm_apply) which costs 3 kernel launches and 4 memory
    /// round-trips per call (× 65 instances per decode step = 195 of the
    /// ~930 graph nodes). The fused kernel computes everything in
    /// registers: load x and w, reduce sum(x^2), rsqrt, multiply.
    ///
    /// Bitwise-equal to the 3-kernel chain: same reduction (reduce_sum on
    /// the [H] tile), same scale formula (rsqrt(sum*inv_h + eps)), same
    /// apply (x * scale * w), same cutile rsqrt (ftz::Disabled).
    #[cutile::entry()]
    pub fn rmsnorm_fused<const H: i32>(
        out: &mut Tensor<f32, { [H] }>,
        x: &Tensor<f32, { [-1] }>,
        weight: &Tensor<f32, { [-1] }>,
        inv_h: f32,
        eps: f32,
    ) {
        let pid = program_id(0);
        let tile_x: Tile<f32, { [H] }> = x.partition(shape![H]).load([pid]);
        let tile_w: Tile<f32, { [H] }> = weight.partition(shape![H]).load([0i32]);

        // Pass 1: sum(x^2) -- same reduction as rmsnorm_sum_sq.
        let sq: Tile<f32, { [H] }> = tile_x * tile_x;
        let sum: Tile<f32, { [] }> = reduce_sum(sq, 0i32);
        let sum_tile: Tile<f32, { [1] }> = sum.reshape(shape![1]);

        // Pass 2: scale = rsqrt(sum/H + eps) -- same formula as
        // cutile_rmsnorm_scale.
        let mean_eps = sum_tile * broadcast_scalar(inv_h, shape![1])
            + broadcast_scalar(eps, shape![1]);
        let scale_tile: Tile<f32, { [1] }> = rsqrt(mean_eps, ftz::Disabled);

        // Pass 3: apply (x * scale * w) -- same as rmsnorm_apply.
        let s_broad: Tile<f32, { [H] }> =
            broadcast_scalar(tile_to_scalar(reduce_sum(scale_tile, 0i32)), shape![H]);
        out.store(tile_x * s_broad * tile_w);
    }

    /// RMSNorm pass 1: compute sum(x^2) per token row into a 1-element tile.
    ///
    /// One program instance per token. H = hidden_size.
    /// Avoids all f32 scalar arithmetic in the JIT -- sum is stored as a tile
    /// element; the CPU computes scale = 1/sqrt(mean+eps) between the two passes.
    #[cutile::entry()]
    pub fn rmsnorm_sum_sq<const H: i32>(
        sum_out: &mut Tensor<f32, { [1] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile_x: Tile<f32, { [H] }> = x.partition(shape![H]).load([pid]);
        let sq: Tile<f32, { [H] }> = tile_x * tile_x;
        let sum: Tile<f32, { [] }> = reduce_sum(sq, 0i32);
        sum_out.store(sum.reshape(shape![1]));
    }

    /// RMSNorm scale kernel: compute per-token scale = 1/sqrt(sum_sq/hidden_size + eps).
    ///
    /// One program instance per token. Runs entirely on device between pass 1
    /// (sum_sq) and pass 2 (apply), eliminating the CPU round-trip.
    /// `inv_h` = 1.0 / hidden_size is pre-computed on the host and passed as
    /// a scalar parameter to avoid integer-to-f32 casting inside the JIT.
    #[cutile::entry()]
    pub fn cutile_rmsnorm_scale(
        scale: &mut Tensor<f32, { [1] }>,
        sum_sq: &Tensor<f32, { [-1] }>,
        inv_h: f32,
        eps: f32,
    ) {
        let pid = program_id(0);
        let s: Tile<f32, { [1] }> = sum_sq.partition(shape![1]).load([pid]);
        let mean_eps = s * broadcast_scalar(inv_h, shape![1]) + broadcast_scalar(eps, shape![1]);
        scale.store(rsqrt(mean_eps, ftz::Disabled));
    }

    /// RMSNorm pass 2: apply GPU-computed per-token scale and weight.
    ///
    /// One program instance per token. H = hidden_size.
    /// `scale` carries one float per token: 1/sqrt(mean(x^2)+eps).
    /// Broadcasting is done via tile_to_scalar + broadcast_scalar, keeping
    /// all operations within cuTile tile ops -- no f32 scalar arithmetic.
    #[cutile::entry()]
    pub fn rmsnorm_apply<const H: i32>(
        out: &mut Tensor<f32, { [H] }>,
        x: &Tensor<f32, { [-1] }>,
        weight: &Tensor<f32, { [-1] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile_x: Tile<f32, { [H] }> = x.partition(shape![H]).load([pid]);
        let tile_w: Tile<f32, { [H] }> = weight.partition(shape![H]).load([0i32]);
        let tile_s: Tile<f32, { [1] }> = scale.partition(shape![1]).load([pid]);
        let s_broad: Tile<f32, { [H] }> =
            broadcast_scalar(tile_to_scalar(reduce_sum(tile_s, 0i32)), shape![H]);
        out.store(tile_x * s_broad * tile_w);
    }
}

/// GPU-accelerated RMSNorm using NVIDIA cuTile Rust (`cutile-rs`).
///
/// Two-pass approach: pass 1 computes sum(x^2) per token on GPU, pass 2 applies
/// the scale (computed on CPU from the downloaded sums) and weight on GPU.
/// All f32 scalar arithmetic (division, sqrt) runs on CPU between the two passes,
/// avoiding the cuTile JIT's restriction on f32 scalar arithmetic in kernels.
pub struct CutileRmsNorm {
    device: Arc<Device>,
    stream: Arc<Stream>,
}

unsafe impl Send for CutileRmsNorm {}
unsafe impl Sync for CutileRmsNorm {}

impl CutileRmsNorm {
    /// Initialise with the default CUDA device (ordinal 0).
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    /// Initialise with a specific CUDA device ordinal.
    pub fn with_device(ordinal: usize) -> Result<Self> {
        let device = Device::new(ordinal).map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let stream = device
            .new_stream()
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Self { device, stream })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// RMSNorm entirely on device.
    ///
    /// `x_dev` is a flat 1-D device tensor of `seq_len * h_pad` elements (any
    /// shape is accepted; data is treated as row-major `[seq_len, h_pad]`).
    /// `weight_dev` is a pre-uploaded 1-D tensor of `h_pad` elements.
    ///
    /// Pass 1 computes `sum(x^2)` per token on device.  Only `seq_len` f32 sums
    /// cross the PCIe bus (D->H).  The CPU computes `scale[t] = 1/sqrt(mean+eps)`
    /// and uploads `seq_len` f32 scales (H->D).  Pass 2 applies scale and weight
    /// entirely on device.
    ///
    /// Returns a `[seq_len, h_pad]` device tensor ready for projection GEMMs.
    pub fn forward_device(
        &self,
        stream: &Arc<Stream>,
        x_dev: Arc<Tensor<f32>>,
        weight_dev: &Arc<Tensor<f32>>,
        seq_len: usize,
        hidden_size: usize,
        h_pad: usize,
        eps: f32,
    ) -> Result<Arc<Tensor<f32>>> {
        // Pass 1: sum(x^2) per token -- one scalar per token on device.
        let sum_t = api::zeros::<f32>(&[seq_len])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1]);

        let (sum_t, x_t) = kernels::rmsnorm_sum_sq(sum_t, x_dev)
            .generics(vec![h_pad.to_string()])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // Download seq_len floats -- negligible PCIe traffic.
        let sum_sq_vec: Vec<f32> = sum_t
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // CPU: scale[t] = 1/sqrt(mean(x^2) + eps).
        let scale_vec: Vec<f32> = sum_sq_vec
            .iter()
            .map(|&s| 1.0f32 / (s / hidden_size as f32 + eps).sqrt())
            .collect();

        let scale_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(scale_vec))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        // Pass 2: apply scale and weight on device.
        let out_t = api::zeros::<f32>(&[seq_len * h_pad])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([h_pad]);

        let (out_t, _x, _w, _s) = kernels::rmsnorm_apply(out_t, x_t, weight_dev.clone(), scale_t)
            .generics(vec![h_pad.to_string()])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // Reshape flat output to [seq_len, h_pad] for direct use by projection GEMMs.
        let out_2d: Arc<Tensor<f32>> = out_t
            .unpartition()
            .reshape(&[seq_len, h_pad])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();
        Ok(out_2d)
    }

    /// Fully-fused RMSNorm entirely on device -- no CPU round-trip between passes.
    ///
    /// Same interface as `forward_device` but all three GPU kernels are submitted
    /// asynchronously (pass 1 -> scale -> pass 2) and a single stream synchronize
    /// covers them all, eliminating the two blocking `sync_on` calls that
    /// previously forced the CPU to wait for each pass before the next began.
    ///
    /// 112 blocking round-trips per decode token (2 RMSNorm x 28+ layers x 2 sync_on)
    /// become 3 async kernel launches + 1 synchronize.
    pub fn forward_device_fused(
        &self,
        stream: &Arc<Stream>,
        x_dev: Arc<Tensor<f32>>,
        weight_dev: &Arc<Tensor<f32>>,
        seq_len: usize,
        hidden_size: usize,
        h_pad: usize,
        eps: f32,
    ) -> Result<Arc<Tensor<f32>>> {
        let inv_h = 1.0f32 / hidden_size as f32;

        // Pass 1: sum(x^2) per token -- async.
        let sum_t = api::zeros::<f32>(&[seq_len])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1]);

        // SAFETY: tensors are accessed only after the final stream.synchronize().
        let (sum_t, x_t) = unsafe {
            kernels::rmsnorm_sum_sq(sum_t, x_dev)
                .generics(vec![h_pad.to_string()])
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };
        let sum_arc: Arc<Tensor<f32>> = Arc::new(sum_t.unpartition());

        // Scale kernel: scale[t] = 1/sqrt(sum_sq[t]/hidden_size + eps) -- async.
        let scale_t = api::zeros::<f32>(&[seq_len])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1]);

        let (scale_t, _sum, _inv_h, _eps) = unsafe {
            kernels::cutile_rmsnorm_scale(scale_t, sum_arc, inv_h, eps)
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };
        let scale_arc: Arc<Tensor<f32>> = Arc::new(scale_t.unpartition());

        // Pass 2: apply scale and weight on device -- async.
        let out_t = api::zeros::<f32>(&[seq_len * h_pad])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([h_pad]);

        let (out_t, _x, _w, _s) = unsafe {
            kernels::rmsnorm_apply(out_t, x_t, weight_dev.clone(), scale_arc)
                .generics(vec![h_pad.to_string()])
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };

        // Single sync covers sum_sq + scale + apply.
        // SAFETY: synchronize waits for all stream ops; no concurrent access.
        unsafe {
            stream
                .synchronize()
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }

        let out_2d: Arc<Tensor<f32>> = out_t
            .unpartition()
            .reshape(&[seq_len, h_pad])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();
        Ok(out_2d)
    }

    /// Fused RMSNorm reusing pre-allocated scratch buffers.
    ///
    /// Identical to `forward_device_fused` but avoids the three
    /// `api::zeros.sync_on` allocations per call by reclaiming device memory
    /// from `scratch`.  For decode (seq_len=1) this eliminates all per-token
    /// allocation overhead.  Prefill callers use it too, with a scratch whose
    /// buffers grow on the first (size-mismatched) call -- steady state is
    /// zero allocations at any seq_len.
    ///
    /// # Reclaim strategy
    ///
    /// `sum_t` and `scale_t` are reclaimed within this call via
    /// `Arc::try_unwrap` on the tensors the scale/apply kernels return.
    /// `out_t` is reclaimed at the start of the *next* call once the caller
    /// has dropped its reference; a pool miss falls back to `api::zeros.sync_on`.
    pub fn forward_device_fused_scratch(
        &self,
        stream: &Arc<Stream>,
        x_dev: Arc<Tensor<f32>>,
        weight_dev: &Arc<Tensor<f32>>,
        seq_len: usize,
        hidden_size: usize,
        h_pad: usize,
        eps: f32,
        scratch: &mut RmsnormScratch,
    ) -> Result<Arc<Tensor<f32>>> {
        let inv_h = 1.0f32 / hidden_size as f32;

        // -- Fused single-kernel path (default) --------------------------------
        // ONE kernel replaces the sum_sq -> scale -> apply chain: computes
        // everything in registers (bitwise-equal -- same reduction, same
        // rsqrt, same apply formula). Eliminates 2 kernel launches and 4
        // memory round-trips per call (× 65 instances per decode step).
        // TEMPURA_NO_FUSED_RMSNORM=1 bisects back to the 3-kernel chain.
        if std::env::var("TEMPURA_NO_FUSED_RMSNORM").as_deref() != Ok("1") {
            let n_out = seq_len * h_pad;
            let out_tensor = match scratch.out_t.take() {
                Some(arc) => match Arc::try_unwrap(arc) {
                    Ok(tensor) => {
                        if tensor.size() == n_out {
                            tensor
                                .reshape(&[n_out])
                                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                        } else {
                            api::zeros::<f32>(&[n_out])
                                .sync_on(stream)
                                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                        }
                    }
                    Err(arc_back) => {
                        scratch.out_t = Some(arc_back);
                        api::zeros::<f32>(&[n_out])
                            .sync_on(stream)
                            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    }
                },
                None => api::zeros::<f32>(&[n_out])
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
            };
            let out_part = out_tensor.partition([h_pad]);

            let (out_part_out, _x, _w, _ih, _ep) = unsafe {
                kernels::rmsnorm_fused(out_part, x_dev, weight_dev.clone(), inv_h, eps)
                    .generics(vec![h_pad.to_string()])
                    .async_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            };

            let out_2d = Arc::new(
                out_part_out
                    .unpartition()
                    .reshape(&[seq_len, h_pad])
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
            );
            scratch.out_t = Some(out_2d.clone());
            return Ok(out_2d);
        }

        // -- 3-kernel chain (TEMPURA_NO_FUSED_RMSNORM=1 bisect path) -----------

        // -- Reclaim or allocate sum_t -----------------------------------------
        // Size check: the scratch may be sized for a different seq_len
        // (decode-sized allocation vs a prefill chunk).  A stale smaller
        // buffer would partition to fewer programs than tokens -- silently
        // norming only the first rows.
        let sum_tensor = match scratch.sum_t.take() {
            Some(arc) => match Arc::try_unwrap(arc) {
                Ok(tensor) => {
                    if tensor.size() == seq_len {
                        tensor
                            .reshape(&[seq_len])
                            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    } else {
                        api::zeros::<f32>(&[seq_len])
                            .sync_on(stream)
                            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    }
                }
                Err(arc_back) => {
                    scratch.sum_t = Some(arc_back);
                    api::zeros::<f32>(&[seq_len])
                        .sync_on(stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                }
            },
            None => api::zeros::<f32>(&[seq_len])
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        };
        let sum_part = sum_tensor.partition([1]);

        // Pass 1: sum(x^2) per token -- async.
        // SAFETY: tensors are accessed only after stream.synchronize() below.
        let (sum_part_out, x_t) = unsafe {
            kernels::rmsnorm_sum_sq(sum_part, x_dev)
                .generics(vec![h_pad.to_string()])
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };
        let sum_arc: Arc<Tensor<f32>> = Arc::new(sum_part_out.unpartition());

        // -- Reclaim or allocate scale_t ---------------------------------------
        // Same size check as sum_t above.
        let scale_tensor = match scratch.scale_t.take() {
            Some(arc) => match Arc::try_unwrap(arc) {
                Ok(tensor) => {
                    if tensor.size() == seq_len {
                        tensor
                            .reshape(&[seq_len])
                            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    } else {
                        api::zeros::<f32>(&[seq_len])
                            .sync_on(stream)
                            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    }
                }
                Err(arc_back) => {
                    scratch.scale_t = Some(arc_back);
                    api::zeros::<f32>(&[seq_len])
                        .sync_on(stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                }
            },
            None => api::zeros::<f32>(&[seq_len])
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        };
        let scale_part = scale_tensor.partition([1]);

        // Scale kernel: scale[t] = 1/sqrt(sum_sq[t]/hidden_size + eps) -- async.
        let (scale_part_out, _sum_returned, _inv_h, _eps) = unsafe {
            kernels::cutile_rmsnorm_scale(scale_part, sum_arc, inv_h, eps)
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };
        // Reclaim sum_t: the kernel returns the input Arc; unwrap back to flat tensor.
        scratch.sum_t = match Arc::try_unwrap(_sum_returned) {
            Ok(tensor) => Some(Arc::new(tensor)),
            Err(arc_back) => Some(arc_back),
        };
        let scale_arc: Arc<Tensor<f32>> = Arc::new(scale_part_out.unpartition());

        // -- Reclaim or allocate out_t -----------------------------------------
        // The caller's Arc from the previous call should be dropped by now;
        // Reclaim or allocate out_t.  The pre-allocated scratch may have been
        // sized for a different seq_len (e.g. max_batch at construction vs. the
        // actual M at call time).  If element counts differ, discard and re-allocate.
        let n_out = seq_len * h_pad;
        let out_tensor = match scratch.out_t.take() {
            Some(arc) => match Arc::try_unwrap(arc) {
                Ok(tensor) => {
                    if tensor.size() == n_out {
                        tensor
                            .reshape(&[n_out])
                            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    } else {
                        // Pre-allocated for a different seq_len; drop and allocate fresh.
                        api::zeros::<f32>(&[n_out])
                            .sync_on(stream)
                            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    }
                }
                Err(arc_back) => {
                    scratch.out_t = Some(arc_back);
                    api::zeros::<f32>(&[n_out])
                        .sync_on(stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                }
            },
            None => api::zeros::<f32>(&[n_out])
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        };
        let out_part = out_tensor.partition([h_pad]);

        // Pass 2: apply scale and weight on device -- async.
        let (out_part_out, _x, _w, _s) = unsafe {
            kernels::rmsnorm_apply(out_part, x_t, weight_dev.clone(), scale_arc)
                .generics(vec![h_pad.to_string()])
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };
        // Reclaim scale_t from the returned Arc.
        scratch.scale_t = match Arc::try_unwrap(_s) {
            Ok(tensor) => Some(Arc::new(tensor)),
            Err(arc_back) => Some(arc_back),
        };

        // All three kernels (sum_sq + scale + apply) are now queued async on
        // the stream.  The caller is responsible for synchronising before any
        // CPU read.  GPU-side consumers on the same stream see the correct
        // ordering without an explicit CPU barrier here.
        let out_2d = Arc::new(
            out_part_out
                .unpartition()
                .reshape(&[seq_len, h_pad])
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        );
        // Store a clone so the next call can reclaim the device memory.
        scratch.out_t = Some(out_2d.clone());
        Ok(out_2d)
    }
}

impl RMSNorm for CutileRmsNorm {
    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Tested
    }

    fn forward(&self, x: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) -> Result<()> {
        if weight.len() != hidden_size {
            return Err(KernelError::DimensionMismatch {
                expected: hidden_size,
                got: weight.len(),
            });
        }

        let num_tokens = x.len() / hidden_size;
        if num_tokens == 0 {
            return Ok(());
        }

        // cuTile JIT on sm_120 requires tile/reduction dims to be power-of-2.
        // Pad hidden_size to next power-of-2; padded elements are zero so they
        // don't affect sum(x^2) or the weighted output.
        let h_pad = next_pow2(hidden_size);

        let stream = &self.stream;

        // Upload x (zero-padded to h_pad per token) and weight (zero-padded).
        let x_padded: Vec<f32> = if h_pad == hidden_size {
            x.to_vec()
        } else {
            let mut v = vec![0.0f32; num_tokens * h_pad];
            for t in 0..num_tokens {
                v[t * h_pad..t * h_pad + hidden_size]
                    .copy_from_slice(&x[t * hidden_size..(t + 1) * hidden_size]);
            }
            v
        };
        let w_padded: Vec<f32> = if h_pad == hidden_size {
            weight.to_vec()
        } else {
            let mut v = vec![0.0f32; h_pad];
            v[..hidden_size].copy_from_slice(weight);
            v
        };

        let x_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x_padded))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let w_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(w_padded))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        // Pass 1: compute sum(x^2) per token -- one element per token in sum_t.
        let sum_t = api::zeros::<f32>(&[num_tokens])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1]);

        let (sum_t, x_t) = kernels::rmsnorm_sum_sq(sum_t, x_t)
            .generics(vec![h_pad.to_string()])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let sum_sq_vec: Vec<f32> = sum_t
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // CPU: scale[i] = 1 / sqrt(mean(x^2) + eps) -- no JIT constraint here.
        let scale_vec: Vec<f32> = sum_sq_vec
            .iter()
            .map(|&s| 1.0f32 / (s / hidden_size as f32 + eps).sqrt())
            .collect();

        // Pass 2: apply scale and weight.
        let scale_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(scale_vec))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let out_t = api::zeros::<f32>(&[num_tokens * h_pad])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([h_pad]);

        let (out_t, _x, _w, _s) = kernels::rmsnorm_apply(out_t, x_t, w_t, scale_t)
            .generics(vec![h_pad.to_string()])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let host: Vec<f32> = out_t
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // Trim padded output back to hidden_size per token.
        if h_pad == hidden_size {
            x.copy_from_slice(&host);
        } else {
            for t in 0..num_tokens {
                x[t * hidden_size..(t + 1) * hidden_size]
                    .copy_from_slice(&host[t * h_pad..t * h_pad + hidden_size]);
            }
        }
        Ok(())
    }
}
