use crate::Result;
pub use crate::RoPE;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
use cuda_core::DeviceBuffer;

#[cfg(feature = "cpu")]
pub struct CpuRoPE;

#[cfg(feature = "cpu")]
impl RoPE for CpuRoPE {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Proven
    }

    fn forward(
        &self,
        x: &mut [f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        positions: &[u32],
        theta: f32,
    ) -> Result<()> {
        let half_dim = head_dim / 2;

        for b in 0..batch {
            for s in 0..seq_len {
                let pos = positions[b * seq_len + s] as f32;
                for h in 0..num_heads {
                    let offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                    for d in 0..half_dim {
                        let freq = pos / theta.powf(2.0 * d as f32 / head_dim as f32);
                        let cos_val = freq.cos();
                        let sin_val = freq.sin();

                        let x0 = x[offset + d];
                        let x1 = x[offset + d + half_dim];
                        x[offset + d] = x0 * cos_val - x1 * sin_val;
                        x[offset + d + half_dim] = x0 * sin_val + x1 * cos_val;
                    }
                }
            }
        }

        Ok(())
    }
}

/// GGML-compatible RoPE: adjacent pairs, multiplicative theta accumulation.
/// Translated from ggml.c:14943-15066 (non-NeoX mode, ext_factor=0).
#[cfg(feature = "cpu")]
pub struct CpuRoPEGgml;

#[cfg(feature = "cpu")]
impl RoPE for CpuRoPEGgml {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Tested
    }

    fn forward(
        &self,
        x: &mut [f32],
        _batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        positions: &[u32],
        freq_base: f32,
    ) -> Result<()> {
        // C: const float theta_scale = powf(freq_base, -2.0f/n_dims);
        let theta_scale: f32 = freq_base.powf(-2.0 / head_dim as f32);

        for s in 0..seq_len {
            // C: const int64_t p = pos[i2];
            let p = positions[s] as f32;

            // C: ggml_rope_cache_init — simplified for ext_factor=0, freq_scale=1, no freq_factors
            // C: float theta = theta_base;  (theta_base = p)
            // C: for (i0 = 0; i0 < ne0; i0 += 2) { cos = cosf(theta); sin = sinf(theta); theta *= theta_scale; }
            let mut theta: f32 = p;

            for i0 in (0..head_dim).step_by(2) {
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                // C: for (i1 = 0; i1 < ne1; i1++) — iterate over heads
                for h in 0..num_heads {
                    let offset = (s * num_heads + h) * head_dim;
                    // C: const float x0 = src[0]; const float x1 = src[1];
                    let x0 = x[offset + i0];
                    let x1 = x[offset + i0 + 1];
                    // C: dst_data[0] = x0*cos_theta - x1*sin_theta;
                    // C: dst_data[1] = x0*sin_theta + x1*cos_theta;
                    x[offset + i0] = x0 * cos_theta - x1 * sin_theta;
                    x[offset + i0 + 1] = x0 * sin_theta + x1 * cos_theta;
                }

                // C: theta *= theta_scale;
                theta *= theta_scale;
            }
        }

        Ok(())
    }
}

/// f16-native RoPE: reads f16 input, applies rotation with f32 sin/cos,
/// writes f16 output. For research into full f16 inference pipelines.
/// Enable with TEMPURA_F16_PIPE=1.
#[cfg(feature = "cpu")]
pub fn rope_forward_f16(
    x_f16: &mut [u16],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    positions: &[u32],
    freq_base: f32,
) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("f16c") {
        unsafe { rope_forward_f16_f16c(x_f16, seq_len, num_heads, head_dim, positions, freq_base); }
        return;
    }
    rope_forward_f16_scalar(x_f16, seq_len, num_heads, head_dim, positions, freq_base);
}

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "f16c")]
unsafe fn rope_forward_f16_f16c(
    x_f16: &mut [u16], seq_len: usize, num_heads: usize,
    head_dim: usize, positions: &[u32], freq_base: f32,
) {
    use std::arch::x86_64::*;
    let theta_scale: f32 = freq_base.powf(-2.0 / head_dim as f32);
    for s in 0..seq_len {
        let p = positions[s] as f32;
        let mut theta: f32 = p;
        for i0 in (0..head_dim).step_by(2) {
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            for h in 0..num_heads {
                let offset = (s * num_heads + h) * head_dim;
                let h0 = _mm_set1_epi16(x_f16[offset + i0] as i16);
                let h1 = _mm_set1_epi16(x_f16[offset + i0 + 1] as i16);
                let x0 = _mm_cvtss_f32(_mm_cvtph_ps(h0));
                let x1 = _mm_cvtss_f32(_mm_cvtph_ps(h1));
                let r0 = x0 * cos_theta - x1 * sin_theta;
                let r1 = x0 * sin_theta + x1 * cos_theta;
                x_f16[offset + i0] = _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(r0), _MM_FROUND_TO_NEAREST_INT), 0) as u16;
                x_f16[offset + i0 + 1] = _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(r1), _MM_FROUND_TO_NEAREST_INT), 0) as u16;
            }
            theta *= theta_scale;
        }
    }
}

#[cfg(feature = "cpu")]
fn rope_forward_f16_scalar(
    x_f16: &mut [u16], seq_len: usize, num_heads: usize,
    head_dim: usize, positions: &[u32], freq_base: f32,
) {
    let theta_scale: f32 = freq_base.powf(-2.0 / head_dim as f32);
    for s in 0..seq_len {
        let p = positions[s] as f32;
        let mut theta: f32 = p;
        for i0 in (0..head_dim).step_by(2) {
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            for h in 0..num_heads {
                let offset = (s * num_heads + h) * head_dim;
                let x0 = f16_to_f32_hw(x_f16[offset + i0]);
                let x1 = f16_to_f32_hw(x_f16[offset + i0 + 1]);
                x_f16[offset + i0] = f32_to_f16_hw(x0 * cos_theta - x1 * sin_theta);
                x_f16[offset + i0 + 1] = f32_to_f16_hw(x0 * sin_theta + x1 * cos_theta);
            }
            theta *= theta_scale;
        }
    }
}

/// Fused f32→RoPE→f16: reads f32, applies rotation, writes f16.
#[cfg(feature = "cpu")]
pub fn rope_f32_to_f16(
    x: &[f32],
    out_f16: &mut [u16],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    positions: &[u32],
    freq_base: f32,
) {
    let theta_scale: f32 = freq_base.powf(-2.0 / head_dim as f32);

    for s in 0..seq_len {
        let p = positions[s] as f32;
        let mut theta: f32 = p;

        for i0 in (0..head_dim).step_by(2) {
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            for h in 0..num_heads {
                let offset = (s * num_heads + h) * head_dim;
                let x0 = x[offset + i0];
                let x1 = x[offset + i0 + 1];
                out_f16[offset + i0] = f32_to_f16_hw(x0 * cos_theta - x1 * sin_theta);
                out_f16[offset + i0 + 1] = f32_to_f16_hw(x0 * sin_theta + x1 * cos_theta);
            }

            theta *= theta_scale;
        }
    }
}

#[cfg(feature = "cpu")]
#[inline(always)]
fn f16_to_f32_hw(bits: u16) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") {
            return unsafe {
                use std::arch::x86_64::*;
                let h = _mm_set1_epi16(bits as i16);
                _mm_cvtss_f32(_mm_cvtph_ps(h))
            };
        }
    }
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    if exp == 0 { f32::from_bits(sign << 31) }
    else if exp == 31 { f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13)) }
    else { f32::from_bits((sign << 31) | ((exp + 112) << 23) | (frac << 13)) }
}

#[cfg(feature = "cpu")]
#[inline(always)]
fn f32_to_f16_hw(v: f32) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") {
            return unsafe {
                use std::arch::x86_64::*;
                let xv = _mm_set_ss(v);
                _mm_extract_epi16(_mm_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT), 0) as u16
            };
        }
    }
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let frac = (bits >> 13) & 0x3ff;
    if exp <= 0 { sign as u16 }
    else if exp >= 31 { (sign | 0x7c00) as u16 }
    else { (sign | ((exp as u32) << 10) | frac) as u16 }
}

#[cfg(feature = "cpu")]
pub struct CpuRoPETable {
    pub table: Vec<f32>,
    pub head_dim: usize,
}

#[cfg(feature = "cpu")]
impl CpuRoPETable {
    pub fn new(max_pos: usize, head_dim: usize, theta: f32) -> Self {
        Self { table: build_cos_sin_table(max_pos, head_dim, theta), head_dim }
    }
}

#[cfg(feature = "cpu")]
impl RoPE for CpuRoPETable {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Tested
    }

    fn forward(
        &self,
        x: &mut [f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        positions: &[u32],
        _theta: f32,
    ) -> Result<()> {
        let half = head_dim / 2;
        for b in 0..batch {
            for s in 0..seq_len {
                let pos = positions[b * seq_len + s] as usize;
                for h in 0..num_heads {
                    let offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                    for d in 0..half {
                        let idx = (pos * half + d) * 2;
                        let cos_val = self.table[idx];
                        let sin_val = self.table[idx + 1];
                        let x0 = x[offset + d];
                        let x1 = x[offset + d + half];
                        x[offset + d] = x0 * cos_val - x1 * sin_val;
                        x[offset + d + half] = x0 * sin_val + x1 * cos_val;
                    }
                }
            }
        }
        Ok(())
    }
}

pub fn build_cos_sin_table(max_pos: usize, head_dim: usize, theta: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let mut table = vec![0.0f32; max_pos * half * 2];
    for pos in 0..max_pos {
        for d in 0..half {
            let freq = pos as f64 / (theta as f64).powf(2.0 * d as f64 / head_dim as f64);
            let cos_val = freq.cos() as f32;
            let sin_val = freq.sin() as f32;
            table[(pos * half + d) * 2] = cos_val;
            table[(pos * half + d) * 2 + 1] = sin_val;
        }
    }
    table
}

/// Rotation mode for RoPE — determines how dimension pairs are formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoPEMode {
    /// Adjacent pairs: (x[0],x[1]), (x[2],x[3]), ...
    Normal,
    /// Split-half pairs: (x[0],x[n/2]), (x[1],x[n/2+1]), ...
    NeoX,
}

/// Unified RoPE matching ggml's algorithm for both Normal and NeoX modes.
///
/// Supports partial n_dims, per-dimension freq_factors (LLaMA 3.1),
/// linear frequency scaling, and YaRN context extension.
///
/// Uses multiplicative theta accumulation: theta_scale = freq_base^(-2/n_dims),
/// theta starts at `position`, then theta *= theta_scale each step.
#[cfg(feature = "cpu")]
pub struct CpuRoPEUnified {
    pub mode: RoPEMode,
    /// Number of dimensions to rotate (must be even, <= head_dim).
    pub n_dims: usize,
    /// Per-dimension frequency factors (one per dimension pair).
    /// When set, each dimension's theta is divided by freq_factors[i].
    /// Used by LLaMA 3.1 for extended context.
    pub freq_factors: Option<Vec<f32>>,
    /// Global frequency scale (default 1.0). Simple linear scaling.
    pub freq_scale: f32,
    /// YaRN extrapolation factor (0.0 = disabled).
    pub ext_factor: f32,
    /// YaRN attention scaling factor (default 1.0).
    pub attn_factor: f32,
    /// YaRN fast beta (default 32.0).
    pub beta_fast: f32,
    /// YaRN slow beta (default 1.0).
    pub beta_slow: f32,
    /// Original context length before extension.
    pub original_max_position: usize,
}

#[cfg(feature = "cpu")]
impl CpuRoPEUnified {
    pub fn new(mode: RoPEMode, n_dims: usize) -> Self {
        Self {
            mode,
            n_dims,
            freq_factors: None,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 4096,
        }
    }
}

#[cfg(feature = "cpu")]
impl RoPE for CpuRoPEUnified {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Tested
    }

    fn forward(
        &self,
        x: &mut [f32],
        _batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        positions: &[u32],
        freq_base: f32,
    ) -> Result<()> {
        let n_dims = self.n_dims;
        debug_assert!(n_dims <= head_dim);
        debug_assert!(n_dims % 2 == 0);

        let theta_scale: f32 = freq_base.powf(-2.0 / n_dims as f32);
        let half_n = n_dims / 2;

        // YaRN correction dimensions (precomputed, independent of position)
        let corr_dims = if self.ext_factor != 0.0 {
            yarn_corr_dims(
                n_dims,
                self.original_max_position,
                freq_base,
                self.beta_fast,
                self.beta_slow,
            )
        } else {
            [0.0, 0.0]
        };

        for s in 0..seq_len {
            let p = positions[s] as f32;

            let mut theta: f32 = p;
            let mut cos_cache = vec![0.0f32; half_n];
            let mut sin_cache = vec![0.0f32; half_n];
            for i in 0..half_n {
                let ff = self
                    .freq_factors
                    .as_ref()
                    .map_or(1.0, |ff| ff[i]);
                let theta_extrap = theta / ff;

                let (cos_val, sin_val) = if self.ext_factor != 0.0 {
                    rope_yarn(
                        theta_extrap,
                        self.freq_scale,
                        &corr_dims,
                        i * 2,
                        self.ext_factor,
                        self.attn_factor,
                    )
                } else if self.freq_scale != 1.0 {
                    let scaled = self.freq_scale * theta_extrap;
                    (scaled.cos(), scaled.sin())
                } else {
                    (theta_extrap.cos(), theta_extrap.sin())
                };

                cos_cache[i] = cos_val;
                sin_cache[i] = sin_val;
                theta *= theta_scale;
            }

            for h in 0..num_heads {
                let offset = (s * num_heads + h) * head_dim;

                match self.mode {
                    RoPEMode::Normal => {
                        for i in 0..half_n {
                            let i0 = i * 2;
                            let cos_val = cos_cache[i];
                            let sin_val = sin_cache[i];
                            let x0 = x[offset + i0];
                            let x1 = x[offset + i0 + 1];
                            x[offset + i0] = x0 * cos_val - x1 * sin_val;
                            x[offset + i0 + 1] = x0 * sin_val + x1 * cos_val;
                        }
                    }
                    RoPEMode::NeoX => {
                        for i in 0..half_n {
                            let cos_val = cos_cache[i];
                            let sin_val = sin_cache[i];
                            let x0 = x[offset + i];
                            let x1 = x[offset + i + half_n];
                            x[offset + i] = x0 * cos_val - x1 * sin_val;
                            x[offset + i + half_n] = x0 * sin_val + x1 * cos_val;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// YaRN ramp function: linear interpolation between low and high correction dims.
/// Returns 1.0 when i0/2 < low, 0.0 when i0/2 > high, linear ramp between.
fn yarn_ramp(low: f32, high: f32, i0: i64) -> f32 {
    let y = (i0 as f32 / 2.0 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// YaRN: interpolate between scaled and extrapolated theta, apply magnitude correction.
/// Matches ggml rope_yarn().
fn rope_yarn(
    theta_extrap: f32,
    freq_scale: f32,
    corr_dims: &[f32; 2],
    i0: usize,
    ext_factor: f32,
    mscale: f32,
) -> (f32, f32) {
    let theta_interp = freq_scale * theta_extrap;
    let ramp_mix = yarn_ramp(corr_dims[0], corr_dims[1], i0 as i64) * ext_factor;
    let theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
    let mscale = mscale * (1.0 + 0.1 * (1.0 / freq_scale).ln());
    (theta.cos() * mscale, theta.sin() * mscale)
}

/// Compute YaRN correction dimensions from model parameters.
/// Matches ggml ggml_rope_yarn_corr_dims().
fn yarn_corr_dim(n_dims: usize, n_ctx_orig: usize, n_rot: f32, base: f32) -> f32 {
    n_dims as f32 * (n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

fn yarn_corr_dims(
    n_dims: usize,
    n_ctx_orig: usize,
    freq_base: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> [f32; 2] {
    let start = yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base).floor();
    let end = yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base).ceil();
    [start.max(0.0), end.min(n_dims as f32 - 1.0)]
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
pub struct CudaRoPE {
    gpu: crate::gpu::GpuContext,
    module: crate::gpu::rope_kernel::kernels::LoadedModule,
    cos_sin_table_dev: std::sync::Mutex<Option<DeviceBuffer<f32>>>,
    pub deterministic: bool,
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
unsafe impl Send for CudaRoPE {}
#[cfg(all(feature = "cuda", not(feature = "cutile")))]
unsafe impl Sync for CudaRoPE {}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
impl CudaRoPE {
    pub fn new() -> Result<Self> {
        let gpu = crate::gpu::GpuContext::new(0)?;
        let module = crate::gpu::rope_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(Self { gpu, module, cos_sin_table_dev: std::sync::Mutex::new(None), deterministic: false })
    }

    pub fn with_context(ctx: std::sync::Arc<cuda_core::CudaContext>) -> Result<Self> {
        let gpu = crate::gpu::GpuContext::from_context(ctx);
        let module = crate::gpu::rope_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(Self { gpu, module, cos_sin_table_dev: std::sync::Mutex::new(None), deterministic: false })
    }

    pub fn upload_cos_sin_table(&self, table: &[f32]) -> crate::Result<()> {
        let stream = self.gpu.stream();
        let dev = DeviceBuffer::from_host(&stream, table)
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        *self.cos_sin_table_dev.lock().unwrap() = Some(dev);
        Ok(())
    }
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
impl CudaRoPE {
    pub fn forward_device(
        &self,
        x_in: &DeviceBuffer<f32>,
        x_out: &mut DeviceBuffer<f32>,
        positions: &DeviceBuffer<u32>,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        theta: f32,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();
        let total = seq_len * num_heads * head_dim;
        let module = &self.module;

        if self.deterministic {
            let guard = self.cos_sin_table_dev.lock().unwrap();
            if let Some(ref table_dev) = *guard {
                module
                    .rope_forward_table(
                        &stream,
                        LaunchConfig::for_num_elems(total as u32),
                        x_in,
                        positions,
                        table_dev,
                        x_out,
                        seq_len as u32,
                        num_heads as u32,
                        head_dim as u32,
                    )
                    .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
                return Ok(());
            }
        }

        module
            .rope_forward(
                &stream,
                LaunchConfig::for_num_elems(total as u32),
                x_in,
                positions,
                x_out,
                seq_len as u32,
                num_heads as u32,
                head_dim as u32,
                theta,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
impl RoPE for CudaRoPE {
    fn forward(
        &self,
        x: &mut [f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        positions: &[u32],
        theta: f32,
    ) -> Result<()> {
        use cuda_core::LaunchConfig;

        let stream = self.gpu.stream();
        let total_elems = batch * seq_len * num_heads * head_dim;

        let x_dev = crate::gpu::to_device(&stream, x)?;
        let pos_dev = crate::gpu::to_device_u32(&stream, positions)?;
        let mut out_dev = crate::gpu::to_device(&stream, &vec![0.0f32; total_elems])?;

        let module = &self.module;

        for b in 0..batch {
            let b_offset = b * seq_len * num_heads * head_dim;
            let b_pos_offset = b * seq_len;
            let b_elems = seq_len * num_heads * head_dim;

            if batch == 1 {
                module
                    .rope_forward(
                        &stream,
                        LaunchConfig::for_num_elems(b_elems as u32),
                        &x_dev,
                        &pos_dev,
                        &mut out_dev,
                        seq_len as u32,
                        num_heads as u32,
                        head_dim as u32,
                        theta,
                    )
                    .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
            } else {
                let b_x = &mut x[b_offset..b_offset + b_elems];
                let b_pos = &positions[b_pos_offset..b_pos_offset + seq_len];
                let cpu = CpuRoPE;
                cpu.forward(b_x, 1, seq_len, num_heads, head_dim, b_pos, theta)?;
            }
        }

        if batch == 1 {
            crate::gpu::copy_back(&stream, &out_dev, x)?;
        }

        Ok(())
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn rope_indices_in_bounds() {
        let batch: usize = kani::any();
        let seq_len: usize = kani::any();
        let num_heads: usize = kani::any();
        let head_dim: usize = kani::any();

        kani::assume(batch >= 1 && batch <= 2);
        kani::assume(seq_len >= 1 && seq_len <= 4);
        kani::assume(num_heads >= 1 && num_heads <= 4);
        kani::assume(head_dim >= 2 && head_dim <= 8);
        kani::assume(head_dim % 2 == 0);

        let x_len = batch * seq_len * num_heads * head_dim;
        let pos_len = batch * seq_len;
        let half_dim = head_dim / 2;

        let b: usize = kani::any();
        let s: usize = kani::any();
        let h: usize = kani::any();
        let d: usize = kani::any();
        kani::assume(b < batch);
        kani::assume(s < seq_len);
        kani::assume(h < num_heads);
        kani::assume(d < half_dim);

        assert!(b * seq_len + s < pos_len);
        let offset = ((b * seq_len + s) * num_heads + h) * head_dim;
        assert!(offset + d < x_len);
        assert!(offset + d + half_dim < x_len);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn unified_no_panic() {
        let head_dim: usize = kani::any();
        kani::assume(head_dim >= 2 && head_dim <= 8);
        kani::assume(head_dim % 2 == 0);
        let n_dims = head_dim;
        let seq_len = 1;
        let num_heads = 1;
        let n = seq_len * num_heads * head_dim;

        let mut x = [0.0f32; 8];
        for i in 0..n {
            x[i] = kani::any();
            kani::assume(x[i].is_finite() && x[i] > -100.0 && x[i] < 100.0);
        }
        let pos: u32 = kani::any();
        kani::assume(pos <= 128);
        let positions = [pos];

        let rope = super::CpuRoPEUnified::new(super::RoPEMode::Normal, n_dims);
        let _ = rope.forward(&mut x[..n], 1, seq_len, num_heads, head_dim, &positions, 10000.0);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn unified_position_zero_identity() {
        let head_dim: usize = kani::any();
        kani::assume(head_dim >= 2 && head_dim <= 8);
        kani::assume(head_dim % 2 == 0);
        let n_dims = head_dim;
        let n = head_dim;

        let mut x = [0.0f32; 8];
        for i in 0..n {
            x[i] = kani::any();
            kani::assume(x[i].is_finite() && x[i] > -100.0 && x[i] < 100.0);
        }
        let original = x;
        let positions = [0u32];

        let rope = super::CpuRoPEUnified::new(super::RoPEMode::Normal, n_dims);
        let _ = rope.forward(&mut x[..n], 1, 1, 1, head_dim, &positions, 10000.0);

        // At position 0: theta=0, cos(0)=1, sin(0)=0 → identity
        for i in 0..n {
            assert!((x[i] - original[i]).abs() < 1e-5);
        }
    }

    #[kani::proof]
    fn gpu_rope_thread_decomposition_in_bounds() {
        let seq_len: usize = kani::any();
        let num_heads: usize = kani::any();
        let head_dim: usize = kani::any();

        kani::assume(seq_len >= 1 && seq_len <= 4);
        kani::assume(num_heads >= 1 && num_heads <= 4);
        kani::assume(head_dim >= 2 && head_dim <= 4);
        kani::assume(head_dim % 2 == 0);

        let total = seq_len * num_heads * head_dim;
        let x_len = total;

        let i: usize = kani::any();
        kani::assume(i < total);

        let hd = head_dim;
        let d_local = i % hd;
        let h = (i / hd) % num_heads;
        let s = i / hd / num_heads;

        assert!(s < seq_len);
        assert!(h < num_heads);
        assert!(d_local < head_dim);

        let offset = (s * num_heads + h) * hd;
        assert!(offset + d_local < x_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature = "cuda", not(feature = "cutile")))]
    #[test]
    fn test_cuda_rope_matches_cpu() {
        let cpu = CpuRoPE;
        let gpu = CudaRoPE::new().unwrap();

        let head_dim = 16;
        let seq_len = 4;
        let num_heads = 4;
        let theta = 10000.0f32;
        let n = seq_len * num_heads * head_dim;

        let x_orig: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let mut x_cpu = x_orig.clone();
        cpu.forward(
            &mut x_cpu, 1, seq_len, num_heads, head_dim, &positions, theta,
        )
        .unwrap();

        let mut x_gpu = x_orig.clone();
        gpu.forward(
            &mut x_gpu, 1, seq_len, num_heads, head_dim, &positions, theta,
        )
        .unwrap();

        for i in 0..n {
            let diff = (x_cpu[i] - x_gpu[i]).abs();
            assert!(
                diff < 1e-4,
                "CPU vs GPU mismatch at index {i}: cpu={}, gpu={}, diff={diff}",
                x_cpu[i],
                x_gpu[i]
            );
        }
    }

    #[cfg(all(feature = "cuda", not(feature = "cutile")))]
    #[test]
    fn test_cuda_rope_position_zero_identity() {
        let gpu = CudaRoPE::new().unwrap();
        let head_dim = 8;
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = original.clone();
        let positions = vec![0u32];

        gpu.forward(&mut x, 1, 1, 1, head_dim, &positions, 10000.0)
            .unwrap();

        for (i, (a, b)) in x.iter().zip(original.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "Position-zero should be identity, but index {i}: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn test_rope_position_zero_is_identity() {
        let rope = CpuRoPE;
        let head_dim = 4;
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = original.clone();
        let positions = vec![0u32];

        rope.forward(&mut x, 1, 1, 1, head_dim, &positions, 10000.0)
            .unwrap();

        // at position 0, cos=1 sin=0, so output should equal input
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_unified_normal_matches_ggml() {
        let ggml = CpuRoPEGgml;
        let unified = CpuRoPEUnified::new(RoPEMode::Normal, 16);
        let head_dim = 16;
        let seq_len = 4;
        let num_heads = 4;
        let freq_base = 10000.0f32;
        let n = seq_len * num_heads * head_dim;

        let x_orig: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 + 0.5).collect();
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let mut x_ggml = x_orig.clone();
        ggml.forward(&mut x_ggml, 1, seq_len, num_heads, head_dim, &positions, freq_base).unwrap();

        let mut x_unified = x_orig.clone();
        unified.forward(&mut x_unified, 1, seq_len, num_heads, head_dim, &positions, freq_base).unwrap();

        for i in 0..n {
            let diff = (x_ggml[i] - x_unified[i]).abs();
            assert!(
                diff < 1e-5,
                "Normal mode: unified vs ggml mismatch at {i}: ggml={}, unified={}, diff={diff}",
                x_ggml[i], x_unified[i]
            );
        }
    }

    #[test]
    fn test_unified_neox_matches_cpurope() {
        let neox = CpuRoPE;
        let unified = CpuRoPEUnified::new(RoPEMode::NeoX, 16);
        let head_dim = 16;
        let seq_len = 4;
        let num_heads = 4;
        let freq_base = 10000.0f32;
        let n = seq_len * num_heads * head_dim;

        let x_orig: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 + 0.5).collect();
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let mut x_neox = x_orig.clone();
        neox.forward(&mut x_neox, 1, seq_len, num_heads, head_dim, &positions, freq_base).unwrap();

        let mut x_unified = x_orig.clone();
        unified.forward(&mut x_unified, 1, seq_len, num_heads, head_dim, &positions, freq_base).unwrap();

        for i in 0..n {
            let diff = (x_neox[i] - x_unified[i]).abs();
            assert!(
                diff < 1e-4,
                "NeoX mode: unified vs CpuRoPE mismatch at {i}: neox={}, unified={}, diff={diff}",
                x_neox[i], x_unified[i]
            );
        }
    }

    #[test]
    fn test_unified_partial_n_dims_passthrough() {
        let head_dim = 8;
        let n_dims = 4; // only rotate first 4 of 8 dims
        let unified = CpuRoPEUnified::new(RoPEMode::NeoX, n_dims);

        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = original.clone();
        let positions = vec![3u32]; // non-zero position to actually rotate

        unified.forward(&mut x, 1, 1, 1, head_dim, &positions, 10000.0).unwrap();

        // Dimensions n_dims..head_dim should be unchanged
        for i in n_dims..head_dim {
            assert_eq!(
                x[i], original[i],
                "Passthrough dimension {i} was modified: expected {}, got {}",
                original[i], x[i]
            );
        }

        // Rotated dimensions should differ (position != 0)
        let rotated_changed = (0..n_dims).any(|i| (x[i] - original[i]).abs() > 1e-6);
        assert!(rotated_changed, "Rotated dimensions should change at non-zero position");
    }

    #[test]
    fn test_unified_partial_n_dims_normal_passthrough() {
        let head_dim = 8;
        let n_dims = 4;
        let unified = CpuRoPEUnified::new(RoPEMode::Normal, n_dims);

        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = original.clone();
        let positions = vec![5u32];

        unified.forward(&mut x, 1, 1, 1, head_dim, &positions, 10000.0).unwrap();

        for i in n_dims..head_dim {
            assert_eq!(x[i], original[i], "Passthrough dim {i} modified");
        }
        let rotated_changed = (0..n_dims).any(|i| (x[i] - original[i]).abs() > 1e-6);
        assert!(rotated_changed, "Rotated dimensions should change at non-zero position");
    }

    #[test]
    fn test_unified_position_zero_identity() {
        for mode in [RoPEMode::Normal, RoPEMode::NeoX] {
            let unified = CpuRoPEUnified::new(mode, 8);
            let head_dim = 8;
            let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let mut x = original.clone();

            unified.forward(&mut x, 1, 1, 1, head_dim, &[0], 10000.0).unwrap();

            for (i, (a, b)) in x.iter().zip(original.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "Position-zero identity failed for {mode:?} at {i}: got {a}, expected {b}"
                );
            }
        }
    }

    #[test]
    fn test_unified_position_zero_identity_with_passthrough() {
        let unified = CpuRoPEUnified::new(RoPEMode::NeoX, 4);
        let head_dim = 8;
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = original.clone();

        unified.forward(&mut x, 1, 1, 1, head_dim, &[0], 10000.0).unwrap();

        for (i, (a, b)) in x.iter().zip(original.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Position-zero+passthrough identity failed at {i}: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn test_unified_freq_factors_none_regression() {
        // freq_factors=None should produce identical results to the basic constructor
        let base = CpuRoPEUnified::new(RoPEMode::Normal, 16);
        let mut with_none = CpuRoPEUnified::new(RoPEMode::Normal, 16);
        with_none.freq_factors = None;

        let head_dim = 16;
        let seq_len = 4;
        let num_heads = 2;
        let freq_base = 10000.0f32;
        let n = seq_len * num_heads * head_dim;

        let x_orig: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 + 0.5).collect();
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let mut x_base = x_orig.clone();
        base.forward(&mut x_base, 1, seq_len, num_heads, head_dim, &positions, freq_base).unwrap();

        let mut x_none = x_orig.clone();
        with_none.forward(&mut x_none, 1, seq_len, num_heads, head_dim, &positions, freq_base).unwrap();

        for i in 0..n {
            assert_eq!(x_base[i], x_none[i], "freq_factors=None diverged at {i}");
        }
    }

    #[test]
    fn test_unified_freq_factors_all_ones_is_identity() {
        // freq_factors all 1.0 should produce same result as no freq_factors
        let base = CpuRoPEUnified::new(RoPEMode::NeoX, 8);
        let mut with_ones = CpuRoPEUnified::new(RoPEMode::NeoX, 8);
        with_ones.freq_factors = Some(vec![1.0; 4]); // 8/2 = 4 pairs

        let head_dim = 8;
        let x_orig: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let positions = vec![5u32];

        let mut x_base = x_orig.clone();
        base.forward(&mut x_base, 1, 1, 1, head_dim, &positions, 10000.0).unwrap();

        let mut x_ones = x_orig.clone();
        with_ones.forward(&mut x_ones, 1, 1, 1, head_dim, &positions, 10000.0).unwrap();

        for i in 0..head_dim {
            assert!(
                (x_base[i] - x_ones[i]).abs() < 1e-6,
                "freq_factors=[1.0] diverged at {i}: base={}, ones={}",
                x_base[i], x_ones[i]
            );
        }
    }

    #[test]
    fn test_unified_freq_factors_changes_output() {
        let mut rope = CpuRoPEUnified::new(RoPEMode::Normal, 8);
        rope.freq_factors = Some(vec![2.0, 0.5, 1.0, 3.0]); // 4 pairs

        let head_dim = 8;
        let x_orig: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let positions = vec![3u32];

        let mut x_ff = x_orig.clone();
        rope.forward(&mut x_ff, 1, 1, 1, head_dim, &positions, 10000.0).unwrap();

        let base = CpuRoPEUnified::new(RoPEMode::Normal, 8);
        let mut x_base = x_orig.clone();
        base.forward(&mut x_base, 1, 1, 1, head_dim, &positions, 10000.0).unwrap();

        let any_diff = (0..head_dim).any(|i| (x_ff[i] - x_base[i]).abs() > 1e-6);
        assert!(any_diff, "freq_factors should change the output");
    }

    #[test]
    fn test_unified_freq_scale() {
        let mut rope = CpuRoPEUnified::new(RoPEMode::Normal, 8);
        rope.freq_scale = 0.5;

        let head_dim = 8;
        let x_orig: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let positions = vec![4u32];

        let mut x_scaled = x_orig.clone();
        rope.forward(&mut x_scaled, 1, 1, 1, head_dim, &positions, 10000.0).unwrap();

        let base = CpuRoPEUnified::new(RoPEMode::Normal, 8);
        let mut x_base = x_orig.clone();
        base.forward(&mut x_base, 1, 1, 1, head_dim, &positions, 10000.0).unwrap();

        let any_diff = (0..head_dim).any(|i| (x_scaled[i] - x_base[i]).abs() > 1e-6);
        assert!(any_diff, "freq_scale=0.5 should change the output");
    }

    #[test]
    fn test_unified_yarn_ext_factor_changes_output() {
        let mut rope = CpuRoPEUnified::new(RoPEMode::Normal, 16);
        rope.ext_factor = 1.0;
        rope.freq_scale = 0.25; // 4x context extension
        rope.original_max_position = 4096;

        let head_dim = 16;
        let x_orig: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let positions = vec![100u32];

        let mut x_yarn = x_orig.clone();
        rope.forward(&mut x_yarn, 1, 1, 1, head_dim, &positions, 10000.0).unwrap();

        let base = CpuRoPEUnified::new(RoPEMode::Normal, 16);
        let mut x_base = x_orig.clone();
        base.forward(&mut x_base, 1, 1, 1, head_dim, &positions, 10000.0).unwrap();

        let any_diff = (0..head_dim).any(|i| (x_yarn[i] - x_base[i]).abs() > 1e-6);
        assert!(any_diff, "YaRN with ext_factor=1 should change output");
    }

    #[test]
    fn test_unified_yarn_preserves_l2_direction() {
        // YaRN with attn_factor applies magnitude scaling, so L2 norm will change,
        // but the output should still be finite and non-zero
        let mut rope = CpuRoPEUnified::new(RoPEMode::NeoX, 8);
        rope.ext_factor = 1.0;
        rope.freq_scale = 0.5;
        rope.original_max_position = 2048;

        let head_dim = 8;
        let x_orig: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let positions = vec![50u32];

        let mut x = x_orig.clone();
        rope.forward(&mut x, 1, 1, 1, head_dim, &positions, 10000.0).unwrap();

        let l2: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(l2.is_finite(), "YaRN output should be finite");
        assert!(l2 > 0.0, "YaRN output should be non-zero");
    }

    #[test]
    fn test_yarn_corr_dims_basic() {
        let dims = super::yarn_corr_dims(128, 4096, 10000.0, 32.0, 1.0);
        assert!(dims[0] >= 0.0);
        assert!(dims[1] >= dims[0]);
        assert!(dims[1] <= 127.0);
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_rotation_preserves_l2_norm(
                head_dim in (2usize..8usize).prop_filter("must be even", |d| d % 2 == 0),
                position in 0u32..128u32,
            ) {
                let rope = CpuRoPE;
                let mut data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
                let l2_before: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
                rope.forward(&mut data, 1, 1, 1, head_dim, &[position], 10000.0).unwrap();
                let l2_after: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
                prop_assert!((l2_before - l2_after).abs() < 1e-4,
                    "L2 changed: {l2_before} -> {l2_after}");
            }

            #[test]
            fn prop_position_zero_is_identity(
                head_dim in (2usize..8usize).prop_filter("must be even", |d| d % 2 == 0),
            ) {
                let rope = CpuRoPE;
                let original: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
                let mut data = original.clone();
                rope.forward(&mut data, 1, 1, 1, head_dim, &[0], 10000.0).unwrap();
                for (i, (a, b)) in original.iter().zip(data.iter()).enumerate() {
                    prop_assert!((a - b).abs() < 1e-5,
                        "position 0 should be identity, diff at {i}: {a} vs {b}");
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub type CudaRoPE = crate::cutile_rope::CutileRoPE;
