/// GPU capability discovery for kernel-variant selection.
///
/// Queried once at kernel-loader construction time and cached.  The capabilities
/// drive which PTX variant is loaded -- there is no runtime branching after the
/// constructor returns.
///
/// # Usage
///
/// ```rust,ignore
/// let caps = GpuCapabilities::query(stream.device().cu_device());
/// let feats = caps.features();
/// if feats.contains(GpuFeature::DP4A) { /* use dp4a path */ }
/// let best = KernelVariant::select_best(&variants, &caps)?;
/// ```

use bitflags::bitflags;

bitflags! {
    /// PTX ISA / CUDA hardware feature flags -- one bit per distinct feature.
    ///
    /// Authoritative CC thresholds from CUDA 13.3 PTX ISA and sm_*_intrinsics.h.
    /// Use `GpuCapabilities::features()` to obtain the flag set for a device.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GpuFeature: u64 {
        // -- Available on RTX 5090 (sm_120) ------------------------------
        /// cc >= 610 -- __dp4a / `dp4a.s32.s32` (CUDA `sm_61_intrinsics.h`).
        /// Pascal GP102/GP104/GP106/GP108; NOT GP100 (sm_60).
        const DP4A          = 1 << 0;
        /// cc >= 610 -- `dp2a.lo` / `dp2a.hi` 2-element INT8 dot.
        const DP2A          = 1 << 1;
        /// cc >= 750 -- `mma.sync.m16n8k8` FP16/INT8 tensor cores (Turing MMA).
        const TURING_MMA    = 1 << 2;
        /// cc >= 800 -- `cp.async.cg.shared.global` (PTX ISA 7.0, Ampere).
        const CP_ASYNC      = 1 << 3;
        /// cc >= 800 -- `ldmatrix.sync.aligned` (PTX ISA 7.5, Ampere).
        const LDMATRIX      = 1 << 4;
        /// cc >= 800 -- `redux.sync.add.f32` hardware warp reduction (PTX ISA 7.0).
        const REDUX_SYNC    = 1 << 5;
        /// cc >= 800 -- BF16 tensor cores / `hfma2.rn bf16x2` (Ampere).
        const BF16_MMA      = 1 << 6;
        /// cc >= 800 -- Ampere `m16n8k16` INT4/TF32 MMA variants.
        const AMPERE_MMA    = 1 << 7;
        /// cc >= 900 -- `elect.sync` single-thread-per-warp election (Hopper).
        const ELECT_SYNC    = 1 << 8;
        /// cc >= 900, sm_90a -- `wgmma.mma_async` warp-group MMA (Hopper).
        const WGMMA         = 1 << 9;
        /// cc >= 900 -- `cp.async.bulk` / TMA tensor memory access (Hopper).
        const TMA           = 1 << 10;
        /// cc >= 900 -- `setmaxnreg` dynamic register limit per warp (Hopper).
        const SETMAXNREG    = 1 << 11;
        /// cc >= 1000 -- thread-block cluster launch (`cluster.arrive`, etc.).
        const CLUSTER       = 1 << 12;
        /// cc >= 1200 -- direct sm_120 PTX codegen target (Blackwell consumer).
        const SM120_ISA     = 1 << 13;
        /// cc >= 1200 -- Blackwell FP16/BF16 MMA variants (sm_120 tensor cores).
        const BLACKWELL_MMA = 1 << 14;
        /// sm_120a sub-variant -- FP4 block-scaled MMA `mxf4.block_scale.m16n8k64.e2m1`.
        /// Detected separately (requires cuModuleLoadData probe); not set by `features()`.
        const MXFP4_MMA     = 1 << 15;
        // -- Datacenter / special only ------------------------------------
        /// cc == 1210 -- DGX Spark `prefetch.global.L2` (NOT RTX 5090).
        const L2_PREFETCH   = 1 << 16;
        /// cc_major == 10, sm_100a/sm_103a -- TCGEN05 TMEM tensor core (NOT sm_120).
        const TCGEN05       = 1 << 17;
        /// cc >= 900 -- `multimem.ld.reduce` NVLink multi-GPU ops.
        const MULTIMEM      = 1 << 18;
        // -- Micro-optimisations (all CC, cost-free) -----------------------
        /// all CC -- `ld.global.nc` non-coherent load (bypass L1 for scan-once reads).
        const LD_NC         = 1 << 19;
        /// all CC -- `prmt.b32` byte permutation (nibble extraction, shuffle tricks).
        const PRMT          = 1 << 20;
    }
}

/// Cached GPU hardware capabilities, queried once per device at init time.
#[derive(Debug, Clone, Copy)]
pub struct GpuCapabilities {
    /// Compute capability major version (e.g. 12 for sm_120).
    pub cc_major: u32,
    /// Compute capability minor version (e.g. 0 for sm_120).
    pub cc_minor: u32,
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
    /// Maximum shared memory per block (bytes).
    pub max_shared_mem_per_block: usize,
    /// Maximum registers per block.
    pub max_registers_per_block: u32,
}

impl GpuCapabilities {
    /// Query device capabilities via the CUDA driver API.
    ///
    /// Returns a zeroed struct (cc=0) if the driver calls fail.
    #[cfg(feature = "cutile")]
    pub fn query(cu_device: cuda_core::sys::CUdevice) -> Self {
        let mut cc_major = 0i32;
        let mut cc_minor = 0i32;
        let mut sm_count = 0i32;
        let mut max_smem = 0i32;
        let mut max_regs = 0i32;

        unsafe {
            use cuda_core::sys::{
                cuDeviceGetAttribute,
                CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as CC_MAJOR,
                CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as CC_MINOR,
                CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT     as SM_COUNT,
                CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK as MAX_SMEM,
                CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK  as MAX_REGS,
            };
            let _ = cuDeviceGetAttribute(&mut cc_major, CC_MAJOR, cu_device);
            let _ = cuDeviceGetAttribute(&mut cc_minor, CC_MINOR, cu_device);
            let _ = cuDeviceGetAttribute(&mut sm_count, SM_COUNT, cu_device);
            let _ = cuDeviceGetAttribute(&mut max_smem,  MAX_SMEM, cu_device);
            let _ = cuDeviceGetAttribute(&mut max_regs,  MAX_REGS, cu_device);
        }

        Self {
            cc_major: cc_major.max(0) as u32,
            cc_minor: cc_minor.max(0) as u32,
            sm_count: sm_count.max(0) as u32,
            max_shared_mem_per_block: max_smem.max(0) as usize,
            max_registers_per_block: max_regs.max(0) as u32,
        }
    }

    /// Build capabilities from known (major, minor) without a live CUDA context.
    /// Useful in tests and offline tooling.
    pub fn from_cc(cc_major: u32, cc_minor: u32) -> Self {
        Self { cc_major, cc_minor, sm_count: 0, max_shared_mem_per_block: 0, max_registers_per_block: 0 }
    }

    /// Compute capability as a single comparable integer.
    ///
    /// The encoding is `major * 100 + minor * 10`, so sm_61 -> 610, sm_120 -> 1200.
    #[inline]
    pub fn cc(&self) -> u32 {
        self.cc_major * 100 + self.cc_minor * 10
    }

    /// Full PTX ISA feature set available on this device.
    ///
    /// Derived entirely from `cc()`.  Sub-variant features that require a probe
    /// (`MXFP4_MMA`) are NOT set here -- callers must probe separately.
    pub fn features(&self) -> GpuFeature {
        // LD_NC and PRMT are available on all CC.
        let mut f = GpuFeature::LD_NC | GpuFeature::PRMT;
        let cc = self.cc();

        // dp4a: Pascal GP102+ (sm_61). GP100 (sm_60) lacks it -- threshold is 610.
        if cc >= 610  { f |= GpuFeature::DP4A  | GpuFeature::DP2A; }
        if cc >= 750  { f |= GpuFeature::TURING_MMA; }
        if cc >= 800  { f |= GpuFeature::CP_ASYNC | GpuFeature::LDMATRIX
                            | GpuFeature::REDUX_SYNC | GpuFeature::BF16_MMA
                            | GpuFeature::AMPERE_MMA; }
        if cc >= 900  { f |= GpuFeature::ELECT_SYNC | GpuFeature::WGMMA
                            | GpuFeature::TMA | GpuFeature::SETMAXNREG
                            | GpuFeature::MULTIMEM; }
        if cc >= 1000 { f |= GpuFeature::CLUSTER; }
        if cc >= 1200 { f |= GpuFeature::SM120_ISA | GpuFeature::BLACKWELL_MMA; }

        // Datacenter Blackwell (sm_100 / sm_103) -- cc_major == 10.
        // NOT RTX 5090 (sm_120, cc_major == 12).
        if self.cc_major == 10 { f |= GpuFeature::TCGEN05; }

        // DGX Spark (sm_121 -> cc() == 1210).
        if cc == 1210 { f |= GpuFeature::L2_PREFETCH; }

        f
    }

    /// True when the device supports dp4a (cc >= 610, Pascal GP102+ and later).
    ///
    /// NOTE: GP100 (sm_60, cc = 600) does NOT support dp4a.
    #[inline]
    pub fn has_dp4a(&self) -> bool {
        self.cc() >= 610
    }

    /// True for Blackwell consumer GPUs (sm_120, RTX 5090 etc.).
    #[inline]
    pub fn is_blackwell_consumer(&self) -> bool {
        self.cc_major == 12
    }

    /// True for Blackwell datacenter GPUs (sm_100/sm_103, B100/B200/GX10).
    #[inline]
    pub fn is_blackwell_datacenter(&self) -> bool {
        self.cc_major == 10
    }

    /// True for any Blackwell-generation device (sm_120 or sm_100).
    #[inline]
    pub fn is_blackwell(&self) -> bool {
        self.cc_major == 12 || self.cc_major == 10
    }

    /// True when the device is Hopper (cc_major == 9).
    #[inline]
    pub fn is_hopper(&self) -> bool {
        self.cc_major == 9
    }

    /// True when the device is Ampere (cc_major == 8).
    #[inline]
    pub fn is_ampere(&self) -> bool {
        self.cc_major == 8
    }
}

impl Default for GpuCapabilities {
    fn default() -> Self {
        Self {
            cc_major: 0,
            cc_minor: 0,
            sm_count: 0,
            max_shared_mem_per_block: 0,
            max_registers_per_block: 0,
        }
    }
}

/// A kernel variant with a minimum compute-capability requirement.
///
/// Used by kernel loaders to build a list of loaded variants, then select the
/// best one at construction time.  After construction the selected `CUfunction`
/// is stored directly -- no `Option` overhead on the hot path.
#[cfg(feature = "cutile")]
#[derive(Clone)]
pub struct KernelVariant {
    /// Resolved CUDA function handle.
    pub cu_fn: cuda_core::sys::CUfunction,
    /// Minimum cc() value required (e.g. 1200 for sm_120, 800 for sm_80).
    pub min_cc: u32,
    /// Human-readable name for diagnostics.
    pub name: &'static str,
}

#[cfg(feature = "cutile")]
impl KernelVariant {
    /// Select the highest-cc variant whose `min_cc <= caps.cc()`.
    ///
    /// Returns `None` when no variant is compatible (should never happen if the
    /// caller always includes a baseline sm_80 variant from the oxide PTX).
    pub fn select_best<'a>(
        variants: &'a [KernelVariant],
        caps: &GpuCapabilities,
    ) -> Option<&'a KernelVariant> {
        let device_cc = caps.cc();
        variants
            .iter()
            .filter(|v| device_cc >= v.min_cc)
            .max_by_key(|v| v.min_cc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cc_encoding() {
        let rtx5090 = GpuCapabilities::from_cc(12, 0);
        assert_eq!(rtx5090.cc(), 1200);

        let a100 = GpuCapabilities::from_cc(8, 0);
        assert_eq!(a100.cc(), 800);

        let sm61 = GpuCapabilities::from_cc(6, 1);
        assert_eq!(sm61.cc(), 610);

        let sm60 = GpuCapabilities::from_cc(6, 0);
        assert_eq!(sm60.cc(), 600);
    }

    #[test]
    fn test_dp4a_threshold_610() {
        // GP100 (sm_60, cc=600) does NOT have dp4a.
        let gp100 = GpuCapabilities::from_cc(6, 0);
        assert!(!gp100.has_dp4a());
        assert!(!gp100.features().contains(GpuFeature::DP4A));

        // GP102/GP104 (sm_61, cc=610) DO have dp4a per sm_61_intrinsics.h.
        let gp102 = GpuCapabilities::from_cc(6, 1);
        assert!(gp102.has_dp4a());
        assert!(gp102.features().contains(GpuFeature::DP4A));

        // Turing (sm_75, cc=750) has dp4a.
        let turing = GpuCapabilities::from_cc(7, 5);
        assert!(turing.has_dp4a());

        // RTX 5090 (sm_120, cc=1200) has dp4a.
        let rtx5090 = GpuCapabilities::from_cc(12, 0);
        assert!(rtx5090.has_dp4a());
    }

    #[test]
    fn test_blackwell_consumer_vs_datacenter() {
        let rtx5090 = GpuCapabilities::from_cc(12, 0);
        assert!(rtx5090.is_blackwell());
        assert!(rtx5090.is_blackwell_consumer());
        assert!(!rtx5090.is_blackwell_datacenter());

        // sm_100 (B100/B200 datacenter)
        let b200 = GpuCapabilities::from_cc(10, 0);
        assert!(b200.is_blackwell());
        assert!(!b200.is_blackwell_consumer());
        assert!(b200.is_blackwell_datacenter());
        assert!(b200.features().contains(GpuFeature::TCGEN05));
        assert!(!rtx5090.features().contains(GpuFeature::TCGEN05));
    }

    #[test]
    fn test_feature_accumulation() {
        let ampere = GpuCapabilities::from_cc(8, 0);
        let f = ampere.features();
        assert!(f.contains(GpuFeature::DP4A));
        assert!(f.contains(GpuFeature::TURING_MMA));
        assert!(f.contains(GpuFeature::CP_ASYNC));
        assert!(f.contains(GpuFeature::LDMATRIX));
        assert!(f.contains(GpuFeature::AMPERE_MMA));
        assert!(!f.contains(GpuFeature::WGMMA));
        assert!(!f.contains(GpuFeature::SM120_ISA));

        let rtx5090 = GpuCapabilities::from_cc(12, 0);
        let f2 = rtx5090.features();
        assert!(f2.contains(GpuFeature::SM120_ISA));
        assert!(f2.contains(GpuFeature::BLACKWELL_MMA));
        assert!(!f2.contains(GpuFeature::TCGEN05));
        assert!(!f2.contains(GpuFeature::L2_PREFETCH));

        // All CC have LD_NC and PRMT.
        let volta = GpuCapabilities::from_cc(7, 0);
        assert!(volta.features().contains(GpuFeature::LD_NC));
        assert!(volta.features().contains(GpuFeature::PRMT));
    }

    #[test]
    fn test_dgx_spark_l2_prefetch() {
        let dgx_spark = GpuCapabilities::from_cc(12, 1); // sm_121
        assert!(dgx_spark.features().contains(GpuFeature::L2_PREFETCH));

        let rtx5090 = GpuCapabilities::from_cc(12, 0); // sm_120
        assert!(!rtx5090.features().contains(GpuFeature::L2_PREFETCH));
    }
}
