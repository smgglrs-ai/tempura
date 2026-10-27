use crate::{KernelError, Result};
use cuda_core::{Device, Stream};
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Reshape, Tensor};
use std::sync::Arc;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// Element-wise addition: out[i] = a[i] + b[i]
    #[cutile::entry()]
    pub fn cutile_vector_add<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile_a = a.partition(shape![B]).load([pid]);
        let tile_b = b.partition(shape![B]).load([pid]);
        out.store(tile_a + tile_b);
    }

    /// SiLU-gated MLP: out[i] = SiLU(gate[i]) * up[i]
    ///
    /// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    #[cutile::entry()]
    pub fn cutile_silu_mul<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        gate: &Tensor<f32, { [-1] }>,
        up: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile_g = gate.partition(shape![B]).load([pid]);
        let tile_up = up.partition(shape![B]).load([pid]);
        let neg_one = broadcast_scalar(-1.0_f32, shape![B]);
        let pos_one = broadcast_scalar(1.0_f32, shape![B]);
        let denom = pos_one + exp(tile_g * neg_one);
        // SiLU: x / (1 + exp(-x))
        let silu = tile_g * (pos_one / denom);
        out.store(silu * tile_up);
    }

    /// Q8 quantize-dequantize round-trip over 32-element blocks.
    ///
    /// Snaps each element to the nearest int8-representable value within
    /// its block, then restores to f32.  B must equal 32.
    ///
    ///   amax  = max |x_in[i]|
    ///   scale = amax / 127.0
    ///   q     = clamp(round(x / scale), -128, 127)
    ///   out   = q * scale
    #[cutile::entry()]
    pub fn cutile_q8_snap<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        x_in: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile = x_in.partition(shape![B]).load([pid]);
        let abs_tile = absf(tile);
        // reduce broadcasts the max back across all B elements (same output shape as input).
        let amax = reduce(abs_tile, 0i32, 0.0f32, |acc, x| max(acc, x));
        let eps = broadcast_scalar(1e-9_f32, shape![B]);
        let c127 = broadcast_scalar(127.0_f32, shape![B]);
        let inv_scale = c127 / (amax + eps);
        let scale = (amax + eps) / c127;
        // Quantise then dequantise: truncation toward zero is acceptable
        // since the error is at most 0.5 * scale per element.
        let q = tile * inv_scale;
        out.store(q * scale);
    }
}

/// Tile width for 1D elementwise and q8 kernels (32 elements = one Q8 block).
const B: usize = 32;

/// GPU-accelerated elementwise operations using NVIDIA cuTile Rust (`cutile-rs`).
///
/// Covers: vector add, SiLU-gated projection, and Q8 round-trip quantisation.
/// Column-slice extraction is performed on the host for architectural reasons
/// (the operation requires non-uniform column strides that do not map to
/// uniform cutile tile partitions).
pub struct CutileElementwise {
    device: Arc<Device>,
    stream: Arc<Stream>,
}

unsafe impl Send for CutileElementwise {}
unsafe impl Sync for CutileElementwise {}

impl CutileElementwise {
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

    /// Element-wise addition: `output[i] = a[i] + b[i]`.
    pub fn vector_add(&self, a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        let n = a.len();
        if b.len() != n || output.len() != n {
            return Err(KernelError::DimensionMismatch {
                expected: n,
                got: b.len(),
            });
        }
        let n_pad = n.div_ceil(B) * B;
        let (a_vec, b_vec) = pad_pair(a, b, n, n_pad);
        let stream = &self.stream;

        let out_t = api::zeros::<f32>(&[n_pad])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([B]);

        let a_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&a_vec)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let b_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&b_vec)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let (out_t, _a, _b) = kernels::cutile_vector_add(out_t, a_t, b_t)
            .generics(vec![B.to_string()])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let host: Vec<f32> = out_t
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        output.copy_from_slice(&host[..n]);
        Ok(())
    }

    /// SiLU-gated projection: `output[i] = SiLU(gate[i]) * up[i]`.
    pub fn silu_mul(&self, gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<()> {
        let n = gate.len();
        if up.len() != n || output.len() != n {
            return Err(KernelError::DimensionMismatch {
                expected: n,
                got: up.len(),
            });
        }
        let n_pad = n.div_ceil(B) * B;
        let (gate_vec, up_vec) = pad_pair(gate, up, n, n_pad);
        let stream = &self.stream;

        let out_t = api::zeros::<f32>(&[n_pad])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([B]);

        let gate_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&gate_vec)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let up_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&up_vec)
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let (out_t, _g, _u) = kernels::cutile_silu_mul(out_t, gate_t, up_t)
            .generics(vec![B.to_string()])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let host: Vec<f32> = out_t
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        output.copy_from_slice(&host[..n]);
        Ok(())
    }

    /// SiLU-gated projection operating entirely on device tensors.
    ///
    /// `gate` and `up` must already reside on the GPU (`seq_len x inter` elements
    /// each, flat 1-D).  The result is computed on the GPU and returned as a
    /// device tensor of `seq_len x inter` elements -- no host round-trip for the
    /// gate or up inputs.
    ///
    /// When `inter` is already a multiple of `B` (true for all standard models,
    /// e.g. 11008 = 344 x 32) the inputs are used directly; otherwise a small
    /// host-side zero-padding pass is inserted.
    ///
    /// PCIe transfers: 0 D->H + 0 H->D for the standard (B-aligned) path.
    /// Compare to the CPU path: 2 D->H (gate + up) + 1 H->D (silu result) = 3 transfers.
    pub fn silu_mul_device(
        &self,
        gate: &Arc<Tensor<f32>>,
        up: &Arc<Tensor<f32>>,
        seq_len: usize,
        inter: usize,
        stream: &Arc<Stream>,
    ) -> Result<Arc<Tensor<f32>>> {
        let n = seq_len * inter;
        let n_pad = n.div_ceil(B) * B;
        // Launch on the CALLER's stream: this struct owns a private stream,
        // and running the kernel there races with the producers of gate/up and
        // the consumers of the output, which all run on the caller's stream
        // with no cross-stream events (found via bench-batch fresh-cache
        // replay: allocation-history-dependent logits).

        let out_t = api::zeros::<f32>(&[n_pad])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([B]);

        let (gate_t, up_t): (Arc<Tensor<f32>>, Arc<Tensor<f32>>) = if n_pad == n {
            // Already B-aligned: pass device tensors directly -- 0 PCIe transfers.
            (gate.clone(), up.clone())
        } else {
            // Rare: inter not a multiple of B.  Download, pad, upload.
            let gate_host = gate
                .to_host_vec()
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            let up_host = up
                .to_host_vec()
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            let (gv, uv) = pad_pair(&gate_host, &up_host, n, n_pad);
            let gt = api::copy_host_vec_to_device(&gv)
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .into();
            let ut = api::copy_host_vec_to_device(&uv)
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .into();
            (gt, ut)
        };

        // SAFETY: kernel is queued async; GPU-side consumers on the same stream
        // execute after this kernel completes.  Caller syncs before CPU read.
        let (out_t, _g, _u) = unsafe {
            kernels::cutile_silu_mul(out_t, gate_t, up_t)
                .generics(vec![B.to_string()])
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };

        Ok(Arc::new(out_t.unpartition()))
    }

    /// SiLU-gated multiply reusing a pre-allocated output buffer.
    ///
    /// Identical to `silu_mul_device` but avoids the `api::zeros.sync_on` allocation
    /// (a full stream synchronize + cudaMallocAsync) by reclaiming the previous call's
    /// buffer via `Arc::try_unwrap`.  Falls back to a fresh allocation on a pool miss.
    ///
    /// Caller must synchronise the stream before any CPU read of the output.
    pub fn silu_mul_device_scratch(
        &self,
        gate: &Arc<Tensor<f32>>,
        up: &Arc<Tensor<f32>>,
        seq_len: usize,
        inter: usize,
        scratch: &mut Option<Arc<Tensor<f32>>>,
        stream: &Arc<Stream>,
    ) -> Result<Arc<Tensor<f32>>> {
        let n = seq_len * inter;
        let n_pad = n.div_ceil(B) * B;
        // Launch on the CALLER's stream (see silu_mul_device): this struct's
        // private stream has no events tying it to the surrounding ops.

        let out_tensor = match scratch.take() {
            Some(arc) => match Arc::try_unwrap(arc) {
                Ok(tensor) => {
                    if tensor.size() == n_pad {
                        tensor
                            .reshape(&[n_pad])
                            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    } else {
                        // Pre-allocated for a different seq_len; discard and re-allocate.
                        api::zeros::<f32>(&[n_pad])
                            .sync_on(stream)
                            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    }
                }
                Err(arc_back) => {
                    *scratch = Some(arc_back);
                    api::zeros::<f32>(&[n_pad])
                        .sync_on(stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                }
            },
            None => api::zeros::<f32>(&[n_pad])
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        };
        let out_t = out_tensor.partition([B]);

        let (gate_t, up_t): (Arc<Tensor<f32>>, Arc<Tensor<f32>>) = if n_pad == n {
            (gate.clone(), up.clone())
        } else {
            let gate_host = gate
                .to_host_vec()
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            let up_host = up
                .to_host_vec()
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            let (gv, uv) = pad_pair(&gate_host, &up_host, n, n_pad);
            let gt = api::copy_host_vec_to_device(&gv)
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .into();
            let ut = api::copy_host_vec_to_device(&uv)
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                .into();
            (gt, ut)
        };

        // SAFETY: kernel queued async; GPU stream ordering ensures correct execution.
        // Caller syncs before CPU read.
        let (out_t, _g, _u) = unsafe {
            kernels::cutile_silu_mul(out_t, gate_t, up_t)
                .generics(vec![B.to_string()])
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };

        let result = Arc::new(out_t.unpartition());
        *scratch = Some(result.clone());
        Ok(result)
    }

    /// Q8 round-trip quantisation: snaps each element to the nearest
    /// int8-representable value within its 32-element block.
    pub fn q8_snap(&self, data: &mut [f32]) -> Result<()> {
        let n = data.len();
        // Align to multiples of B (= 32 = Q8 block size).
        let n_pad = n.div_ceil(B) * B;
        let mut padded = data.to_vec();
        padded.resize(n_pad, 0.0f32);

        let stream = &self.stream;

        let out_t = api::zeros::<f32>(&[n_pad])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([B]);

        let in_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(padded))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let (out_t, _in) = kernels::cutile_q8_snap(out_t, in_t)
            .generics(vec![B.to_string()])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let host: Vec<f32> = out_t
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        data.copy_from_slice(&host[..n]);
        Ok(())
    }

    /// Extract a column slice from a row-major matrix.
    ///
    /// `src` has shape `[rows, total_cols]`; `dst` receives
    /// `src[row, col_offset..col_offset+slice_cols]` for every row.
    ///
    /// Implemented on the host: the operation involves non-uniform column
    /// strides that do not map cleanly to uniform cutile tile partitions.
    pub fn slice_columns(
        &self,
        src: &[f32],
        dst: &mut [f32],
        rows: usize,
        total_cols: usize,
        col_offset: usize,
        slice_cols: usize,
    ) -> Result<()> {
        if src.len() != rows * total_cols {
            return Err(KernelError::DimensionMismatch {
                expected: rows * total_cols,
                got: src.len(),
            });
        }
        if dst.len() != rows * slice_cols {
            return Err(KernelError::DimensionMismatch {
                expected: rows * slice_cols,
                got: dst.len(),
            });
        }
        for row in 0..rows {
            let src_start = row * total_cols + col_offset;
            let dst_start = row * slice_cols;
            dst[dst_start..dst_start + slice_cols]
                .copy_from_slice(&src[src_start..src_start + slice_cols]);
        }
        Ok(())
    }
}

/// Pad two slices to `n_pad` and return `Arc<Vec<f32>>` for each.
fn pad_pair(a: &[f32], b: &[f32], n: usize, n_pad: usize) -> (Arc<Vec<f32>>, Arc<Vec<f32>>) {
    if n_pad == n {
        (Arc::new(a.to_vec()), Arc::new(b.to_vec()))
    } else {
        let mut av = vec![0.0f32; n_pad];
        let mut bv = vec![0.0f32; n_pad];
        av[..n].copy_from_slice(a);
        bv[..n].copy_from_slice(b);
        (Arc::new(av), Arc::new(bv))
    }
}
