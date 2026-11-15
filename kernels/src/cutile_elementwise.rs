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

    /// In-place row-broadcast bias add: out[row, :] += bias[:].
    /// `out` is flat row-major [rows*N] partitioned by [N] (one program per
    /// row); `bias` is [N]. In-place via out.load/out.store (same pattern
    /// as the upstream `increment` test kernel). Rows beyond the valid
    /// region (m_pad padding) receive the bias too -- harmless: pad rows
    /// never enter the KV cache or downstream reads.
    #[cutile::entry()]
    pub fn cutile_bias_add_rows<const N: i32>(
        out: &mut Tensor<f32, { [N] }>,
        bias: &Tensor<f32, { [-1] }>,
    ) {
        let tile_b = bias.partition(shape![N]).load([0i32]);
        let x = out.load();
        out.store(x + tile_b);
    }

    /// Row-broadcast bias add for NON-power-of-two column counts (Qwen2
    /// q_dim 3584 = 2^9 * 7 -- cutile tile dims must be powers of two, so
    /// cutile_bias_add_rows cannot partition by [cols]). `out` is flat
    /// row-major [rows * cols] partitioned into B-element tiles (B a power
    /// of two dividing cols, TPR = cols / B tiles per row); program pid
    /// handles flat tile pid, whose bias tile within its row is pid % TPR.
    /// Same math as cutile_bias_add_rows -- bitwise-equal when TPR == 1.
    #[cutile::entry()]
    pub fn cutile_bias_add_rows_tiled<const B: i32, const TPR: i32>(
        out: &mut Tensor<f32, { [B] }>,
        bias: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile_b = bias.partition(shape![B]).load([pid % TPR]);
        let x = out.load();
        out.store(x + tile_b);
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

    /// f32 -> f16 element cast (p24 f16 KV pool): one program per B-element
    /// block.  Used by the KV append path -- the pool stores half the
    /// bytes; conversion happens in registers.
    #[cutile::entry()]
    pub fn cutile_cast_f32_f16<const B: i32>(
        out: &mut Tensor<half::f16, { [B] }>,
        src: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile: Tile<f32, { [B] }> = src.partition(shape![B]).load([pid]);
        let cast: Tile<half::f16, { [B] }> = convert_tile(tile);
        out.store(cast);
    }

    /// f16 -> f32 element cast (p24 f16 KV pool): one program per B-element
    /// block.  Used by the swap path to rebuild the f32 host mirror from
    /// an f16 pool (the DRAM tier keeps an f32 mirror for now).
    #[cutile::entry()]
    pub fn cutile_cast_f16_f32<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        src: &Tensor<half::f16, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile: Tile<half::f16, { [B] }> = src.partition(shape![B]).load([pid]);
        let cast: Tile<f32, { [B] }> = convert_tile(tile);
        out.store(cast);
    }

    /// f32 -> e4m3 element cast (e4m3 KV pool): one program per B-element
    /// block.  Tile-level port of `tempura_kernels::e4m3::f32_to_e4m3_bits`
    /// (the ONE shared implementation, which this mirrors bit-for-bit):
    /// RNE on the 20 dropped mantissa bits, saturation to +-448, subnormal
    /// flush, NaN -> 0x7F.  Storage is raw u8.
    #[cutile::entry()]
    pub fn cutile_cast_f32_e4m3<const B: i32>(
        out: &mut Tensor<u8, { [B] }>,
        src: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let x: Tile<f32, { [B] }> = src.partition(shape![B]).load([pid]);

        // NaN detect on the ORIGINAL bits (float min/max would swallow
        // NaN -- max(NaN, 448) = 448 on PTX).  abs > inf-bits <=> NaN;
        // both fit positive-i32 so the tile compare is sign-agnostic.
        let xb: Tile<u32, { [B] }> = bitcast(x);
        let m_abs: Tile<u32, { [B] }> = broadcast_scalar(0x7FFF_FFFF_u32, shape![B]);
        let m_inf: Tile<u32, { [B] }> = broadcast_scalar(0x7F80_0000_u32, shape![B]);
        let is_nan: Tile<bool, { [B] }> = gt_tile(andi(xb, m_abs), m_inf);

        // Saturate to +-448 (E4M3FN has no infinity).
        let lo: Tile<f32, { [B] }> = broadcast_scalar(-448.0_f32, shape![B]);
        let hi: Tile<f32, { [B] }> = broadcast_scalar(448.0_f32, shape![B]);
        let xc = min_tile(max_tile(x, lo), hi);

        let bits: Tile<u32, { [B] }> = bitcast(xc);
        // Sign bit via the top byte (bits >> 24 & 0x80) -- the DSL cannot
        // parse the u32 literal 0x8000_0000 (i32 literal overflow).
        let c_24: Tile<u32, { [B] }> = broadcast_scalar(24_u32, shape![B]);
        let c_80: Tile<u32, { [B] }> = broadcast_scalar(0x80_u32, shape![B]);
        let sign8: Tile<u32, { [B] }> = andi(shri(bits, c_24), c_80);
        let abs: Tile<u32, { [B] }> = andi(bits, m_abs);
        let c_23: Tile<u32, { [B] }> = broadcast_scalar(23_u32, shape![B]);
        let exp: Tile<u32, { [B] }> = shri(abs, c_23);
        let c_man: Tile<u32, { [B] }> = broadcast_scalar(0x007F_FFFF_u32, shape![B]);
        let man: Tile<u32, { [B] }> = andi(abs, c_man);

        // Normal iff f32 exponent >= 121 (2^-6, the smallest e4m3
        // normal); below that -> signed zero (subnormal flush).
        let c_120: Tile<u32, { [B] }> = broadcast_scalar(120_u32, shape![B]);
        let is_norm: Tile<bool, { [B] }> = gt_tile(exp, c_120);

        // RNE on the 20 dropped mantissa bits.
        let c_20: Tile<u32, { [B] }> = broadcast_scalar(20_u32, shape![B]);
        let kept: Tile<u32, { [B] }> = shri(man, c_20);
        let c_drop: Tile<u32, { [B] }> = broadcast_scalar(0x000F_FFFF_u32, shape![B]);
        let dropped: Tile<u32, { [B] }> = andi(man, c_drop);
        let c_half: Tile<u32, { [B] }> = broadcast_scalar(0x0008_0000_u32, shape![B]);
        let gt_half: Tile<bool, { [B] }> = gt_tile(dropped, c_half);
        let eq_half: Tile<bool, { [B] }> = eq_tile(dropped, c_half);
        let c_1: Tile<u32, { [B] }> = broadcast_scalar(1_u32, shape![B]);
        let odd: Tile<bool, { [B] }> = eq_tile(andi(kept, c_1), c_1);
        let round_up: Tile<bool, { [B] }> = gt_half | (eq_half & odd);
        let c_0: Tile<u32, { [B] }> = broadcast_scalar(0_u32, shape![B]);
        let kept2: Tile<u32, { [B] }> = kept + select(round_up, c_1, c_0);
        // Carry: kept2 == 8 -> mantissa 0, exponent +1.  (e2 = 16 is
        // unreachable: the 448 clamp caps the top binade -- see the
        // shared helper's docs.)
        let c_8: Tile<u32, { [B] }> = broadcast_scalar(8_u32, shape![B]);
        let carry: Tile<bool, { [B] }> = eq_tile(kept2, c_8);
        let man_f: Tile<u32, { [B] }> = select(carry, c_0, kept2);
        let e2f: Tile<u32, { [B] }> =
            (exp - c_120) + select(carry, c_1, c_0); // wraps on flush lanes; discarded
        let c_3: Tile<u32, { [B] }> = broadcast_scalar(3_u32, shape![B]);
        let body: Tile<u32, { [B] }> = ori(ori(shli(e2f, c_3, overflow::NoUnsignedWrap), man_f), sign8);
        let res: Tile<u32, { [B] }> = select(is_norm, body, sign8);
        let res: Tile<u32, { [B] }> = select(is_nan, broadcast_scalar(0x7F_u32, shape![B]), res);
        let c_ff: Tile<u32, { [B] }> = broadcast_scalar(0xFF_u32, shape![B]);
        let b8: Tile<u8, { [B] }> = trunci(andi(res, c_ff), overflow::NoUnsignedWrap);
        out.store(b8);
    }

    /// e4m3 -> f32 element cast (e4m3 KV pool): one program per B-element
    /// block.  Tile-level port of `tempura_kernels::e4m3::e4m3_bits_to_f32`
    /// (EXACT for normals; subnormals flush to signed zero; the exp==15
    /// NaN encodings map through the normal path -- unreachable from our
    /// writer).  Used by the swap path to rebuild the f32 host mirror.
    #[cutile::entry()]
    pub fn cutile_cast_e4m3_f32<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        src: &Tensor<u8, { [-1] }>,
    ) {
        let pid = program_id(0);
        let raw: Tile<u8, { [B] }> = src.partition(shape![B]).load([pid]);
        let w: Tile<u32, { [B] }> = exti(raw);
        let c_80: Tile<u32, { [B] }> = broadcast_scalar(0x80_u32, shape![B]);
        let c_3: Tile<u32, { [B] }> = broadcast_scalar(3_u32, shape![B]);
        let c_0f: Tile<u32, { [B] }> = broadcast_scalar(0x0F_u32, shape![B]);
        let c_07: Tile<u32, { [B] }> = broadcast_scalar(0x07_u32, shape![B]);
        let c_20: Tile<u32, { [B] }> = broadcast_scalar(20_u32, shape![B]);
        let c_23: Tile<u32, { [B] }> = broadcast_scalar(23_u32, shape![B]);
        let c_24: Tile<u32, { [B] }> = broadcast_scalar(24_u32, shape![B]);
        let c_120: Tile<u32, { [B] }> = broadcast_scalar(120_u32, shape![B]);
        let c_0: Tile<u32, { [B] }> = broadcast_scalar(0_u32, shape![B]);
        let sign: Tile<u32, { [B] }> = shli(andi(w, c_80), c_24, overflow::NoUnsignedWrap);
        let exp: Tile<u32, { [B] }> = andi(shri(w, c_3), c_0f);
        let man: Tile<u32, { [B] }> = shli(andi(w, c_07), c_20, overflow::NoUnsignedWrap);
        // f32 bits = sign | (exp + 120) << 23 | man << 20; exp == 0
        // (subnormal encodings) -> signed zero.
        let norm: Tile<u32, { [B] }> =
            ori(ori(sign, shli(exp + c_120, c_23, overflow::NoUnsignedWrap)), man);
        let bits: Tile<u32, { [B] }> = select(ne_tile(exp, c_0), norm, sign);
        let f: Tile<f32, { [B] }> = bitcast(bits);
        out.store(f);
    }

    /// GELU-gated MLP (Gemma GeGLU): out[i] = gelu(gate[i]) * up[i].
    /// gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)));
    /// tanh via 1 - 2/(exp(2y)+1) (overflow-safe: y -> inf gives 1).
    #[cutile::entry()]
    pub fn cutile_gelu_mul<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        gate: &Tensor<f32, { [-1] }>,
        up: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile_g = gate.partition(shape![B]).load([pid]);
        let tile_up = up.partition(shape![B]).load([pid]);
        let half: Tile<f32, { [B] }> = broadcast_scalar(0.5_f32, shape![B]);
        let one: Tile<f32, { [B] }> = broadcast_scalar(1.0_f32, shape![B]);
        let two: Tile<f32, { [B] }> = broadcast_scalar(2.0_f32, shape![B]);
        let sqrt_2_over_pi: Tile<f32, { [B] }> =
            broadcast_scalar(0.797_884_56_f32, shape![B]);
        let gelu_coef: Tile<f32, { [B] }> = broadcast_scalar(0.044_715_f32, shape![B]);
        // x + 0.044715 * x^3, scaled by sqrt(2/pi).
        let y = (tile_g + tile_g * tile_g * tile_g * gelu_coef) * sqrt_2_over_pi;
        let e = exp(y * two);
        let tanh_y = one - two / (e + one);
        let g = half * tile_g * (one + tanh_y);
        out.store(g * tile_up);
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

/// Tile width for the device-side SiLU-gated multiply: 256 f32 (1KB) per
/// program. The 32-element tiles moved only 384B per block; 256 keeps the
/// same math (elementwise -- bitwise-identical for any tile size) with
/// properly sized memory transactions.
const SILU_B: usize = 256;

/// Tile width for the f32<->f16 cast kernels (4KB reads / 2KB writes).
pub const CAST_B: usize = 1024;

/// Cast the first `n` elements of an f32 device tensor into an f16 device
/// tensor (p24 f16 KV pool append path).  `src` must hold at least
/// `n` rounded up to a CAST_B multiple; `dst` is RESIZED to that count
/// (the returned Arc replaces the caller's).  Async on `stream`.
pub fn cast_f32_to_f16_dev(
    src: &Arc<Tensor<f32>>,
    dst: Arc<Tensor<half::f16>>,
    n: usize,
    stream: &Arc<Stream>,
) -> Result<Arc<Tensor<half::f16>>> {
    let n_pad = n.div_ceil(CAST_B) * CAST_B;
    if src.size() < n_pad {
        return Err(KernelError::InvalidConfig(format!(
            "cast_f32_to_f16: src holds {} elements, need {n_pad}",
            src.size()
        )));
    }
    let dst_tensor = match Arc::try_unwrap(dst) {
        Ok(t) if t.size() == n_pad => t,
        Ok(t) => t
            .reshape(&[n_pad])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        Err(_) => {
            return Err(KernelError::InvalidConfig(
                "cast_f32_to_f16: dst Arc has live clones".into(),
            ))
        }
    };
    let dst_part = dst_tensor.partition([CAST_B]);
    // SAFETY: tensors are re-Arc'd from the launch return; accessed only
    // after subsequent stream ops.
    let (dst_out, _src) = unsafe {
        kernels::cutile_cast_f32_f16(dst_part, src.clone())
            .generics(vec![CAST_B.to_string()])
            .async_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
    };
    Ok(Arc::new(dst_out.unpartition()))
}

/// Cast the first `n` elements of an f16 device tensor into an f32 device
/// tensor (p24 f16 KV pool: rebuild the f32 host-mirror staging).  Same
/// contract as [`cast_f32_to_f16_dev`] with the dtypes swapped.
pub fn cast_f16_to_f32_dev(
    src: &Arc<Tensor<half::f16>>,
    dst: Arc<Tensor<f32>>,
    n: usize,
    stream: &Arc<Stream>,
) -> Result<Arc<Tensor<f32>>> {
    let n_pad = n.div_ceil(CAST_B) * CAST_B;
    if src.size() < n_pad {
        return Err(KernelError::InvalidConfig(format!(
            "cast_f16_to_f32: src holds {} elements, need {n_pad}",
            src.size()
        )));
    }
    let dst_tensor = match Arc::try_unwrap(dst) {
        Ok(t) if t.size() == n_pad => t,
        Ok(t) => t
            .reshape(&[n_pad])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        Err(_) => {
            return Err(KernelError::InvalidConfig(
                "cast_f16_to_f32: dst Arc has live clones".into(),
            ))
        }
    };
    let dst_part = dst_tensor.partition([CAST_B]);
    // SAFETY: see cast_f32_to_f16_dev.
    let (dst_out, _src) = unsafe {
        kernels::cutile_cast_f16_f32(dst_part, src.clone())
            .generics(vec![CAST_B.to_string()])
            .async_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
    };
    Ok(Arc::new(dst_out.unpartition()))
}

/// Cast the first `n` elements of an f32 device tensor into an e4m3
/// (u8) device tensor (e4m3 KV pool append path).  Same contract as
/// [`cast_f32_to_f16_dev`] with e4m3 storage; the kernel mirrors the
/// shared `tempura_kernels::e4m3::f32_to_e4m3_bits` bit-for-bit.
pub fn cast_f32_to_e4m3_dev(
    src: &Arc<Tensor<f32>>,
    dst: Arc<Tensor<u8>>,
    n: usize,
    stream: &Arc<Stream>,
) -> Result<Arc<Tensor<u8>>> {
    let n_pad = n.div_ceil(CAST_B) * CAST_B;
    if src.size() < n_pad {
        return Err(KernelError::InvalidConfig(format!(
            "cast_f32_to_e4m3: src holds {} elements, need {n_pad}",
            src.size()
        )));
    }
    let dst_tensor = match Arc::try_unwrap(dst) {
        Ok(t) if t.size() == n_pad => t,
        Ok(t) => t
            .reshape(&[n_pad])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        Err(_) => {
            return Err(KernelError::InvalidConfig(
                "cast_f32_to_e4m3: dst Arc has live clones".into(),
            ))
        }
    };
    let dst_part = dst_tensor.partition([CAST_B]);
    // SAFETY: tensors are re-Arc'd from the launch return; accessed only
    // after subsequent stream ops.
    let (dst_out, _src) = unsafe {
        kernels::cutile_cast_f32_e4m3(dst_part, src.clone())
            .generics(vec![CAST_B.to_string()])
            .async_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
    };
    Ok(Arc::new(dst_out.unpartition()))
}

/// Cast the first `n` elements of an e4m3 (u8) device tensor into an f32
/// device tensor (e4m3 KV pool: rebuild the f32 host-mirror staging).
/// Same contract as [`cast_f32_to_f16_dev`] with the dtypes swapped.
pub fn cast_e4m3_to_f32_dev(
    src: &Arc<Tensor<u8>>,
    dst: Arc<Tensor<f32>>,
    n: usize,
    stream: &Arc<Stream>,
) -> Result<Arc<Tensor<f32>>> {
    let n_pad = n.div_ceil(CAST_B) * CAST_B;
    if src.size() < n_pad {
        return Err(KernelError::InvalidConfig(format!(
            "cast_e4m3_to_f32: src holds {} elements, need {n_pad}",
            src.size()
        )));
    }
    let dst_tensor = match Arc::try_unwrap(dst) {
        Ok(t) if t.size() == n_pad => t,
        Ok(t) => t
            .reshape(&[n_pad])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        Err(_) => {
            return Err(KernelError::InvalidConfig(
                "cast_e4m3_to_f32: dst Arc has live clones".into(),
            ))
        }
    };
    let dst_part = dst_tensor.partition([CAST_B]);
    // SAFETY: see cast_f32_to_e4m3_dev.
    let (dst_out, _src) = unsafe {
        kernels::cutile_cast_e4m3_f32(dst_part, src.clone())
            .generics(vec![CAST_B.to_string()])
            .async_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
    };
    Ok(Arc::new(dst_out.unpartition()))
}

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
        let device = Arc::new(Device::new(ordinal).map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
        let stream = device
            .primary_context()
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
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
        let n_pad = n.div_ceil(SILU_B) * SILU_B;
        // Launch on the CALLER's stream: this struct owns a private stream,
        // and running the kernel there races with the producers of gate/up and
        // the consumers of the output, which all run on the caller's stream
        // with no cross-stream events (found via bench-batch fresh-cache
        // replay: allocation-history-dependent logits).

        let out_t = api::zeros::<f32>(&[n_pad])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([SILU_B]);

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
                .generics(vec![SILU_B.to_string()])
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
        let n_pad = n.div_ceil(SILU_B) * SILU_B;
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
                        if std::env::var("TEMPURA_DEBUG_GELU").is_ok() {
                            eprintln!(
                                "[gelu] scratch size mismatch: {} != n_pad {n_pad}",
                                tensor.size()
                            );
                        }
                        api::zeros::<f32>(&[n_pad])
                            .sync_on(stream)
                            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    }
                }
                Err(arc_back) => {
                    if std::env::var("TEMPURA_DEBUG_GELU").is_ok() {
                        eprintln!("[gelu] scratch not uniquely owned");
                    }
                    *scratch = Some(arc_back);
                    api::zeros::<f32>(&[n_pad])
                        .sync_on(stream)
                        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                }
            },
            None => {
                if std::env::var("TEMPURA_DEBUG_GELU").is_ok() {
                    eprintln!("[gelu] scratch EMPTY (n_pad {n_pad})");
                }
                api::zeros::<f32>(&[n_pad])
                    .sync_on(stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            }
        };
        let out_t = out_tensor.partition([SILU_B]);

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
                .generics(vec![SILU_B.to_string()])
                .async_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };

        let result = Arc::new(out_t.unpartition());
        *scratch = Some(result.clone());
        Ok(result)
    }

    /// GELU-gated multiply (Gemma GeGLU) reusing a pre-allocated output
    /// buffer -- identical buffer lifecycle to `silu_mul_device_scratch`,
    /// with the SiLU swapped for tanh-approximation GELU.
    pub fn gelu_mul_device_scratch(
        &self,
        gate: &Arc<Tensor<f32>>,
        up: &Arc<Tensor<f32>>,
        seq_len: usize,
        inter: usize,
        scratch: &mut Option<Arc<Tensor<f32>>>,
        stream: &Arc<Stream>,
    ) -> Result<Arc<Tensor<f32>>> {
        let n = seq_len * inter;
        let n_pad = n.div_ceil(SILU_B) * SILU_B;

        let out_tensor = match scratch.take() {
            Some(arc) => match Arc::try_unwrap(arc) {
                Ok(tensor) => {
                    if tensor.size() == n_pad {
                        tensor
                            .reshape(&[n_pad])
                            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
                    } else {
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
        let out_t = out_tensor.partition([SILU_B]);

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

        // SAFETY: kernel queued async; GPU stream ordering ensures correct
        // execution. Caller syncs before CPU read.
        let (out_t, _g, _u) = unsafe {
            kernels::cutile_gelu_mul(out_t, gate_t, up_t)
                .generics(vec![SILU_B.to_string()])
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
        let n_pad = n.div_ceil(SILU_B) * SILU_B;
        let mut padded = data.to_vec();
        padded.resize(n_pad, 0.0f32);

        let stream = &self.stream;

        let out_t = api::zeros::<f32>(&[n_pad])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([SILU_B]);

        let in_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(padded))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let (out_t, _in) = kernels::cutile_q8_snap(out_t, in_t)
            .generics(vec![SILU_B.to_string()])
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

#[cfg(test)]
mod gelu_tests {
    use super::*;

    fn cpu_gelu(x: f32) -> f32 {
        // ggml gelu_f32 formula (ops_llama::gelu_f32 equivalent)
        let sqrt_2_over_pi = 0.797_884_560_8_f32;
        let c = 0.044_715_f32;
        0.5 * x * (1.0 + (sqrt_2_over_pi * x * (1.0 + c * x * x)).tanh())
    }

    /// cutile_gelu_mul vs the CPU formula on random data: catches constant,
    /// operand-order, and tile-math errors (uniform data cannot).
    #[test]
    fn gelu_mul_matches_cpu_formula() {
        use cutile::api;
        let el = CutileElementwise::new().expect("init");
        let stream = el.stream().clone();
        let n = 4096usize;
        let mut rng: u32 = 0x12345678;
        let next = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng >> 8) as f32 / 8388608.0) - 3.0 // [-3, 3) + fat tails via cubes below
        };
        let gate: Vec<f32> = (0..n).map(|_| next() * next() * 4.0).collect();
        let up: Vec<f32> = (0..n).map(|_| next()).collect();
        let gate_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(gate.clone()))
            .sync_on(&stream)
            .expect("upload gate")
            .into();
        let up_t: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(up.clone()))
            .sync_on(&stream)
            .expect("upload up")
            .into();
        let out = el
            .gelu_mul_device_scratch(&gate_t, &up_t, 1, n, &mut None, &stream)
            .expect("gelu mul");
        let got: Vec<f32> = Arc::try_unwrap(out)
            .expect("unique")
            .to_host_vec()
            .sync_on(&stream)
            .expect("download");
        let mut max_err = 0.0f32;
        for i in 0..n {
            let want = cpu_gelu(gate[i]) * up[i];
            let err = (got[i] - want).abs();
            assert!(got[i].is_finite(), "NaN/inf at {i}");
            max_err = max_err.max(err);
        }
        assert!(max_err < 1e-4, "max err {max_err}");
        eprintln!("gelu_mul max err {max_err:.2e}");
    }
}
