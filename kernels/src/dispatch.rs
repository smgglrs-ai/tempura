/// Kernel dispatch tables -- select and cache the right Q4K/Q6K variant per layer.
///
/// `GpuArch` is gone; callers pass `&GpuCapabilities` everywhere so that
/// `GpuCapabilities::features()` is the single source of truth for GPU ISA
/// feature checks.

use crate::gpu_capabilities::GpuCapabilities;

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
use crate::gpu::GpuContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKernelVariant {
    Naive,
    Coop,
    CoopMultiWarp,
}

#[derive(Debug, Clone, Copy)]
pub struct Q4KKernelConfig {
    pub variant: QuantKernelVariant,
    pub n_warps_per_col: usize,
}

impl Q4KKernelConfig {
    pub fn select(caps: &GpuCapabilities, n: usize, k: usize) -> Self {
        let blocks_per_row = k / 256;

        if caps.is_blackwell() {
            Self::blackwell(n, blocks_per_row)
        } else if caps.cc() >= 800 {
            // Ampere and Hopper
            Self::ampere_hopper(n, blocks_per_row)
        } else {
            Self::generic(n, blocks_per_row)
        }
    }

    fn blackwell(_n: usize, _blocks_per_row: usize) -> Self {
        Self {
            variant: QuantKernelVariant::CoopMultiWarp,
            n_warps_per_col: 8,
        }
    }

    fn ampere_hopper(n: usize, blocks_per_row: usize) -> Self {
        if n >= 32768 || blocks_per_row >= 16 {
            Self {
                variant: QuantKernelVariant::CoopMultiWarp,
                n_warps_per_col: 4,
            }
        } else {
            Self {
                variant: QuantKernelVariant::Coop,
                n_warps_per_col: 1,
            }
        }
    }

    fn generic(_n: usize, blocks_per_row: usize) -> Self {
        if blocks_per_row >= 8 {
            Self {
                variant: QuantKernelVariant::Coop,
                n_warps_per_col: 1,
            }
        } else {
            Self {
                variant: QuantKernelVariant::Naive,
                n_warps_per_col: 1,
            }
        }
    }
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
pub struct GpuKernelDispatcher {
    pub caps: GpuCapabilities,
    pub q4k: crate::quantized::CudaMatMulQ4K,
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
impl GpuKernelDispatcher {
    pub fn new() -> crate::Result<Self> {
        let q4k = crate::quantized::CudaMatMulQ4K::new()?;
        let ctx: &GpuContext = q4k.gpu();
        let (major, minor) = ctx.context().compute_capability().unwrap_or((0, 0));
        let caps = GpuCapabilities::from_cc(major as u32, minor as u32);
        Ok(Self { caps, q4k })
    }

    pub fn gemv_q4k(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        let config = Q4KKernelConfig::select(&self.caps, n, k);
        match config.variant {
            QuantKernelVariant::Naive => self.q4k.forward_q4k_device(a, b, output, 1, n, k),
            QuantKernelVariant::Coop => self.q4k.gemv_q4k_coop_device(a, b, output, n, k),
            QuantKernelVariant::CoopMultiWarp => {
                self.q4k
                    .gemv_q4k_multirow_device(a, b, output, n, k, config.n_warps_per_col)
            }
        }
    }

    pub fn gemv_q6k(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        self.q4k.forward_q6k_device(a, b, output, 1, n, k)
    }
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
pub struct CachedDispatch {
    q4k: crate::quantized::CudaMatMulQ4K,
    nwarps: Vec<u8>,
    index: std::collections::HashMap<String, usize>,
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
impl CachedDispatch {
    pub fn new(caps: &GpuCapabilities, projections: &[(&str, usize, usize)]) -> crate::Result<Self> {
        let q4k = crate::quantized::CudaMatMulQ4K::new()?;
        let mut nwarps = Vec::with_capacity(projections.len());
        let mut index = std::collections::HashMap::with_capacity(projections.len());
        for (i, (name, n, k)) in projections.iter().enumerate() {
            nwarps.push(Q4KKernelConfig::select(caps, *n, *k).n_warps_per_col as u8);
            index.insert(name.to_string(), i);
        }
        Ok(Self { q4k, nwarps, index })
    }

    pub fn from_model_dims(
        caps: &GpuCapabilities,
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
    ) -> crate::Result<Self> {
        let q4k = crate::quantized::CudaMatMulQ4K::new()?;
        let h = hidden_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let inter = intermediate_size;

        let mut nwarps = Vec::new();
        let mut index = std::collections::HashMap::new();
        for layer in 0..num_layers {
            let prefix = format!("model.layers.{layer}");
            for (proj, n, k) in [
                ("self_attn.q_proj.weight", q_dim, h),
                ("self_attn.k_proj.weight", kv_dim, h),
                ("self_attn.v_proj.weight", kv_dim, h),
                ("self_attn.o_proj.weight", h, h),
                ("mlp.gate_proj.weight", inter, h),
                ("mlp.up_proj.weight", inter, h),
                ("mlp.down_proj.weight", h, inter),
            ] {
                let idx = nwarps.len();
                nwarps.push(Q4KKernelConfig::select(caps, n, k).n_warps_per_col as u8);
                index.insert(format!("{prefix}.{proj}"), idx);
            }
        }
        let idx = nwarps.len();
        nwarps.push(Q4KKernelConfig::select(caps, vocab_size, h).n_warps_per_col as u8);
        index.insert("lm_head.weight".into(), idx);

        Ok(Self { q4k, nwarps, index })
    }

    #[inline]
    pub fn gemv(
        &self,
        name: &str,
        a: &cuda_core::DeviceBuffer<f32>,
        b: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        let nw = match self.index.get(name) {
            Some(&idx) => self.nwarps[idx] as usize,
            None => 8,
        };
        self.q4k.gemv_q4k_multirow_device(a, b, output, n, k, nw)
    }

    #[inline(always)]
    pub fn gemv_idx(
        &self,
        idx: usize,
        a: &cuda_core::DeviceBuffer<f32>,
        b: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        self.q4k
            .gemv_q4k_multirow_device(a, b, output, n, k, self.nwarps[idx] as usize)
    }

    pub fn layer_base_idx(&self, layer: usize) -> usize {
        layer * 7 // 7 projections per layer
    }

    pub fn lm_head_idx(&self) -> usize {
        self.nwarps.len() - 1
    }

    pub fn q4k(&self) -> &crate::quantized::CudaMatMulQ4K {
        &self.q4k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_selection_blackwell() {
        let caps = GpuCapabilities::from_cc(12, 0); // RTX 5090 sm_120

        // down_proj: n=4096, k=12288 (48 blocks)
        let cfg = Q4KKernelConfig::select(&caps, 4096, 12288);
        assert_eq!(cfg.variant, QuantKernelVariant::CoopMultiWarp);
        assert_eq!(cfg.n_warps_per_col, 8);

        // gate_proj: n=12288, k=4096 (16 blocks)
        let cfg = Q4KKernelConfig::select(&caps, 12288, 4096);
        assert_eq!(cfg.variant, QuantKernelVariant::CoopMultiWarp);
        assert_eq!(cfg.n_warps_per_col, 8);

        // lm_head: n=151936, k=4096 (16 blocks)
        let cfg = Q4KKernelConfig::select(&caps, 151936, 4096);
        assert_eq!(cfg.variant, QuantKernelVariant::CoopMultiWarp);
        assert_eq!(cfg.n_warps_per_col, 8);

        // k_proj: n=1024, k=4096
        let cfg = Q4KKernelConfig::select(&caps, 1024, 4096);
        assert_eq!(cfg.n_warps_per_col, 8);
    }

    #[test]
    fn test_kernel_selection_ampere() {
        let caps = GpuCapabilities::from_cc(8, 0); // A100

        // Small n + small blocks_per_row (k=2048 -> 8 blocks) -> Coop.
        let cfg = Q4KKernelConfig::select(&caps, 4096, 2048);
        assert_eq!(cfg.variant, QuantKernelVariant::Coop);

        // blocks_per_row >= 16 (k=4096) -> CoopMultiWarp even for moderate n.
        let cfg = Q4KKernelConfig::select(&caps, 4096, 4096);
        assert_eq!(cfg.variant, QuantKernelVariant::CoopMultiWarp);
        assert_eq!(cfg.n_warps_per_col, 4);

        // Large n -> CoopMultiWarp regardless of blocks_per_row.
        let cfg = Q4KKernelConfig::select(&caps, 32768, 2048);
        assert_eq!(cfg.variant, QuantKernelVariant::CoopMultiWarp);
        assert_eq!(cfg.n_warps_per_col, 4);
    }

    #[test]
    fn test_kernel_selection_pascal() {
        let caps = GpuCapabilities::from_cc(6, 1); // sm_61

        let cfg = Q4KKernelConfig::select(&caps, 4096, 4096);
        assert_eq!(cfg.variant, QuantKernelVariant::Coop);

        // Very small k -> Naive
        let cfg = Q4KKernelConfig::select(&caps, 512, 512);
        assert_eq!(cfg.variant, QuantKernelVariant::Naive);
    }

    #[test]
    fn test_all_projections_get_config() {
        let caps = GpuCapabilities::from_cc(12, 0);
        let projections = [
            ("q_proj", 4096, 4096),
            ("k_proj", 1024, 4096),
            ("v_proj", 1024, 4096),
            ("o_proj", 4096, 4096),
            ("gate_proj", 12288, 4096),
            ("up_proj", 12288, 4096),
            ("down_proj", 4096, 12288),
            ("lm_head", 151936, 4096),
        ];
        for (name, n, k) in projections {
            let cfg = Q4KKernelConfig::select(&caps, n, k);
            assert!(cfg.n_warps_per_col >= 1, "{name}: invalid nwarps");
        }
    }
}
