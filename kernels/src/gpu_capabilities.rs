/// GPU capability discovery for kernel-variant selection.
///
/// Queried once at kernel-loader construction time and cached.  The capabilities
/// drive which PTX variant is loaded — there is no runtime branching after the
/// constructor returns.
///
/// # Usage
///
/// ```rust,ignore
/// let caps = GpuCapabilities::query(stream.device().cu_device());
/// // Select the best kernel variant for this device:
/// let best = KernelVariant::select_best(&variants, &caps)?;
/// ```
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

    /// Compute capability as a single comparable integer (e.g. 1200 for sm_120).
    #[inline]
    pub fn cc(&self) -> u32 {
        self.cc_major * 100 + self.cc_minor * 10
    }

    /// True when the device is Blackwell (cc_major == 12).
    #[inline]
    pub fn is_blackwell(&self) -> bool {
        self.cc_major == 12
    }

    /// True when the device is Ampere (cc_major == 8).
    #[inline]
    pub fn is_ampere(&self) -> bool {
        self.cc_major == 8
    }

    /// True when the device supports dp4a (Turing / sm_75 and newer).
    #[inline]
    pub fn has_dp4a(&self) -> bool {
        self.cc_major >= 7 && !(self.cc_major == 7 && self.cc_minor == 0)
        // sm_70 (Volta) lacks dp4a; sm_75 (Turing) introduced it.
        // Simpler: cc >= 750 → dp4a.
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
/// Used by kernel loaders to build a compile-time-empty list, then select the
/// best loaded variant at construction time.  After construction the selected
/// `CUfunction` is stored directly — no `Option` overhead on the hot path.
#[derive(Clone)]
pub struct KernelVariant {
    /// Resolved CUDA function handle.
    pub cu_fn: cuda_core::sys::CUfunction,
    /// Minimum cc() value required (e.g. 1200 for sm_120, 800 for sm_80).
    pub min_cc: u32,
    /// Human-readable name for diagnostics.
    pub name: &'static str,
}

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
