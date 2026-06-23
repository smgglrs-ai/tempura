#[cfg(feature = "cuda")]
use crate::gpu::GpuContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArch {
    Pascal,
    Turing,
    Ampere,
    Hopper,
    Blackwell,
    Unknown,
}

impl GpuArch {
    #[cfg(feature = "cuda")]
    pub fn detect(ctx: &GpuContext) -> Self {
        let (major, minor) = ctx.context().compute_capability().unwrap_or((0, 0));
        Self::from_sm(major, minor)
    }

    pub fn from_sm(major: i32, _minor: i32) -> Self {
        match major {
            6 => GpuArch::Pascal,
            7 => GpuArch::Turing,
            8 => GpuArch::Ampere,
            9 => GpuArch::Hopper,
            10 | 11 | 12 => GpuArch::Blackwell,
            _ => GpuArch::Unknown,
        }
    }

    pub fn has_dp4a(&self) -> bool {
        !matches!(self, GpuArch::Pascal | GpuArch::Unknown)
    }
}

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
    pub fn select(arch: GpuArch, n: usize, k: usize) -> Self {
        let blocks_per_row = k / 256;

        match arch {
            GpuArch::Blackwell => Self::blackwell(n, blocks_per_row),
            GpuArch::Hopper | GpuArch::Ampere => Self::ampere_hopper(n, blocks_per_row),
            _ => Self::generic(n, blocks_per_row),
        }
    }

    fn blackwell(_n: usize, _blocks_per_row: usize) -> Self {
        Self {
            variant: QuantKernelVariant::CoopMultiWarp,
            n_warps_per_col: 8,
        }
    }

    fn ampere_hopper(n: usize, blocks_per_row: usize) -> Self {
        if n >= 32768 {
            Self {
                variant: QuantKernelVariant::CoopMultiWarp,
                n_warps_per_col: 4,
            }
        } else if blocks_per_row >= 16 {
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

#[cfg(feature = "cuda")]
pub struct GpuKernelDispatcher {
    pub arch: GpuArch,
    pub q4k: crate::quantized::CudaMatMulQ4K,
}

#[cfg(feature = "cuda")]
impl GpuKernelDispatcher {
    pub fn new() -> crate::Result<Self> {
        let q4k = crate::quantized::CudaMatMulQ4K::new()?;
        let arch = GpuArch::detect(q4k.gpu());
        Ok(Self { arch, q4k })
    }

    pub fn gemv_q4k(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        let config = Q4KKernelConfig::select(self.arch, n, k);
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

#[cfg(feature = "cuda")]
pub struct CachedDispatch {
    q4k: crate::quantized::CudaMatMulQ4K,
    nwarps: Vec<u8>,
    index: std::collections::HashMap<String, usize>,
}

#[cfg(feature = "cuda")]
impl CachedDispatch {
    pub fn new(arch: GpuArch, projections: &[(&str, usize, usize)]) -> crate::Result<Self> {
        let q4k = crate::quantized::CudaMatMulQ4K::new()?;
        let mut nwarps = Vec::with_capacity(projections.len());
        let mut index = std::collections::HashMap::with_capacity(projections.len());
        for (i, (name, n, k)) in projections.iter().enumerate() {
            nwarps.push(Q4KKernelConfig::select(arch, *n, *k).n_warps_per_col as u8);
            index.insert(name.to_string(), i);
        }
        Ok(Self { q4k, nwarps, index })
    }

    pub fn from_model_dims(
        arch: GpuArch,
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
                nwarps.push(Q4KKernelConfig::select(arch, n, k).n_warps_per_col as u8);
                index.insert(format!("{prefix}.{proj}"), idx);
            }
        }
        let idx = nwarps.len();
        nwarps.push(Q4KKernelConfig::select(arch, vocab_size, h).n_warps_per_col as u8);
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
    fn test_arch_detection() {
        assert_eq!(GpuArch::from_sm(12, 0), GpuArch::Blackwell);
        assert_eq!(GpuArch::from_sm(9, 0), GpuArch::Hopper);
        assert_eq!(GpuArch::from_sm(8, 0), GpuArch::Ampere);
        assert_eq!(GpuArch::from_sm(7, 5), GpuArch::Turing);
        assert_eq!(GpuArch::from_sm(6, 1), GpuArch::Pascal);
    }

    #[test]
    fn test_dp4a_support() {
        assert!(!GpuArch::Pascal.has_dp4a());
        assert!(GpuArch::Turing.has_dp4a());
        assert!(GpuArch::Ampere.has_dp4a());
        assert!(GpuArch::Blackwell.has_dp4a());
    }

    #[test]
    fn test_kernel_selection_blackwell() {
        // down_proj: n=4096, k=12288 (48 blocks)
        let cfg = Q4KKernelConfig::select(GpuArch::Blackwell, 4096, 12288);
        assert_eq!(cfg.variant, QuantKernelVariant::CoopMultiWarp);
        assert_eq!(cfg.n_warps_per_col, 8);

        // gate_proj: n=12288, k=4096 (16 blocks)
        let cfg = Q4KKernelConfig::select(GpuArch::Blackwell, 12288, 4096);
        assert_eq!(cfg.variant, QuantKernelVariant::CoopMultiWarp);
        assert_eq!(cfg.n_warps_per_col, 8);

        // lm_head: n=151936, k=4096 (16 blocks) — large N, 8 warps
        let cfg = Q4KKernelConfig::select(GpuArch::Blackwell, 151936, 4096);
        assert_eq!(cfg.variant, QuantKernelVariant::CoopMultiWarp);
        assert_eq!(cfg.n_warps_per_col, 8);

        // k_proj: n=1024, k=4096 — 8 warps on Blackwell
        let cfg = Q4KKernelConfig::select(GpuArch::Blackwell, 1024, 4096);
        assert_eq!(cfg.n_warps_per_col, 8);
    }

    #[test]
    fn test_kernel_selection_generic() {
        let cfg = Q4KKernelConfig::select(GpuArch::Pascal, 4096, 4096);
        assert_eq!(cfg.variant, QuantKernelVariant::Coop);
    }

    #[test]
    fn test_all_projections_get_config() {
        let arch = GpuArch::Blackwell;
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
            let cfg = Q4KKernelConfig::select(arch, n, k);
            assert!(cfg.n_warps_per_col >= 1, "{name}: invalid nwarps");
        }
    }
}
