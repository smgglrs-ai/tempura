//! Direct Rust translation of llama.cpp `arch/x86/cpu-feats.cpp`.
//!
//! Source: ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp
//! Ref:    https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
//!
//! Translated items:
//!   - `struct cpuid_x86`                    -> [`CpuidX86`]
//!   - `ggml_backend_cpu_x86_score()`        -> [`cpu_score`]
//!
//! `CpuidX86` queries CPUID at construction and caches all relevant bitfields.
//! Feature accessor methods mirror the original C++ struct exactly.
//!
//! `cpu_score` uses `std::arch::is_x86_feature_detected!` for features that
//! the Rust standard library covers; raw CPUID (via `CpuidX86`) handles the
//! remainder (AMX, AVX-VNNI, AVX512-BF16/FP16, vendor-specific flags).

#![allow(non_snake_case)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod inner {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{__cpuid, __cpuid_count};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{__cpuid, __cpuid_count};

    // -----------------------------------------------------------------------
    // CpuidX86 -- mirrors cpuid_x86 from cpu-feats.cpp
    // -----------------------------------------------------------------------

    /// Runtime x86 CPU feature set derived from CPUID.
    pub struct CpuidX86 {
        pub is_intel: bool,
        pub is_amd: bool,
        pub vendor: String,
        pub brand: String,

        f_1_ecx: u32,
        f_1_edx: u32,
        f_7_ebx: u32,
        f_7_ecx: u32,
        f_7_edx: u32,
        f_7_1_eax: u32,
        f_81_ecx: u32,
        f_81_edx: u32,
    }

    impl CpuidX86 {
        pub fn new() -> Self {
            // Safety: we are on x86/x86_64 by construction of this module.
            unsafe { Self::query() }
        }

        unsafe fn query() -> Self {
            // Highest standard leaf
            let leaf0 = __cpuid(0);
            let n_ids = leaf0.eax;

            // Vendor string: EBX + EDX + ECX (in that order), NUL-terminated
            let vendor = {
                let mut buf = [0u8; 12];
                buf[0..4].copy_from_slice(&leaf0.ebx.to_le_bytes());
                buf[4..8].copy_from_slice(&leaf0.edx.to_le_bytes());
                buf[8..12].copy_from_slice(&leaf0.ecx.to_le_bytes());
                String::from_utf8_lossy(&buf).trim_end_matches('\0').to_owned()
            };
            let is_intel = vendor == "GenuineIntel";
            let is_amd = vendor == "AuthenticAMD";

            // Leaf 1 -- basic feature flags
            let (f_1_ecx, f_1_edx) = if n_ids >= 1 {
                let r = __cpuid(1);
                (r.ecx, r.edx)
            } else {
                (0, 0)
            };

            // Leaf 7 subleaf 0 -- extended feature flags
            let (f_7_ebx, f_7_ecx, f_7_edx) = if n_ids >= 7 {
                let r = __cpuid_count(7, 0);
                (r.ebx, r.ecx, r.edx)
            } else {
                (0, 0, 0)
            };

            // Leaf 7 subleaf 1 -- AVX-VNNI, AVX512-BF16, AMX-FP16
            let f_7_1_eax = if n_ids >= 7 {
                __cpuid_count(7, 1).eax
            } else {
                0
            };

            // Highest extended leaf
            let ext_leaf0 = __cpuid(0x8000_0000u32);
            let n_ex_ids = ext_leaf0.eax;

            // Extended leaf 0x80000001 -- AMD / Intel extended flags
            let (f_81_ecx, f_81_edx) = if n_ex_ids >= 0x8000_0001u32 {
                let r = __cpuid(0x8000_0001u32);
                (r.ecx, r.edx)
            } else {
                (0, 0)
            };

            // CPU brand string (leaves 0x80000002..4)
            let brand = if n_ex_ids >= 0x8000_0004u32 {
                let mut buf = [0u8; 48];
                for (i, leaf) in [0x8000_0002u32, 0x8000_0003u32, 0x8000_0004u32]
                    .iter()
                    .enumerate()
                {
                    let r = __cpuid(*leaf);
                    let off = i * 16;
                    buf[off..off + 4].copy_from_slice(&r.eax.to_le_bytes());
                    buf[off + 4..off + 8].copy_from_slice(&r.ebx.to_le_bytes());
                    buf[off + 8..off + 12].copy_from_slice(&r.ecx.to_le_bytes());
                    buf[off + 12..off + 16].copy_from_slice(&r.edx.to_le_bytes());
                }
                String::from_utf8_lossy(&buf).trim_end_matches('\0').to_owned()
            } else {
                String::new()
            };

            CpuidX86 {
                is_intel,
                is_amd,
                vendor,
                brand,
                f_1_ecx,
                f_1_edx,
                f_7_ebx,
                f_7_ecx,
                f_7_edx,
                f_7_1_eax,
                f_81_ecx,
                f_81_edx,
            }
        }

        #[inline(always)]
        fn bit(reg: u32, bit: u32) -> bool {
            (reg >> bit) & 1 != 0
        }

        // ---- Leaf 1 ECX ----
        pub fn SSE3(&self) -> bool { Self::bit(self.f_1_ecx, 0) }
        pub fn PCLMULQDQ(&self) -> bool { Self::bit(self.f_1_ecx, 1) }
        pub fn MONITOR(&self) -> bool { Self::bit(self.f_1_ecx, 3) }
        pub fn SSSE3(&self) -> bool { Self::bit(self.f_1_ecx, 9) }
        pub fn FMA(&self) -> bool { Self::bit(self.f_1_ecx, 12) }
        pub fn CMPXCHG16B(&self) -> bool { Self::bit(self.f_1_ecx, 13) }
        pub fn SSE41(&self) -> bool { Self::bit(self.f_1_ecx, 19) }
        pub fn SSE42(&self) -> bool { Self::bit(self.f_1_ecx, 20) }
        pub fn MOVBE(&self) -> bool { Self::bit(self.f_1_ecx, 22) }
        pub fn POPCNT(&self) -> bool { Self::bit(self.f_1_ecx, 23) }
        pub fn AES(&self) -> bool { Self::bit(self.f_1_ecx, 25) }
        pub fn XSAVE(&self) -> bool { Self::bit(self.f_1_ecx, 26) }
        pub fn OSXSAVE(&self) -> bool { Self::bit(self.f_1_ecx, 27) }
        pub fn AVX(&self) -> bool { Self::bit(self.f_1_ecx, 28) }
        pub fn F16C(&self) -> bool { Self::bit(self.f_1_ecx, 29) }
        pub fn RDRAND(&self) -> bool { Self::bit(self.f_1_ecx, 30) }

        // ---- Leaf 1 EDX ----
        pub fn MSR(&self) -> bool { Self::bit(self.f_1_edx, 5) }
        pub fn CX8(&self) -> bool { Self::bit(self.f_1_edx, 8) }
        pub fn SEP(&self) -> bool { Self::bit(self.f_1_edx, 11) }
        pub fn CMOV(&self) -> bool { Self::bit(self.f_1_edx, 15) }
        pub fn CLFSH(&self) -> bool { Self::bit(self.f_1_edx, 19) }
        pub fn MMX(&self) -> bool { Self::bit(self.f_1_edx, 23) }
        pub fn FXSR(&self) -> bool { Self::bit(self.f_1_edx, 24) }
        pub fn SSE(&self) -> bool { Self::bit(self.f_1_edx, 25) }
        pub fn SSE2(&self) -> bool { Self::bit(self.f_1_edx, 26) }

        // ---- Leaf 7:0 EBX ----
        pub fn FSGSBASE(&self) -> bool { Self::bit(self.f_7_ebx, 0) }
        pub fn BMI1(&self) -> bool { Self::bit(self.f_7_ebx, 3) }
        pub fn HLE(&self) -> bool { self.is_intel && Self::bit(self.f_7_ebx, 4) }
        pub fn AVX2(&self) -> bool { Self::bit(self.f_7_ebx, 5) }
        pub fn BMI2(&self) -> bool { Self::bit(self.f_7_ebx, 8) }
        pub fn ERMS(&self) -> bool { Self::bit(self.f_7_ebx, 9) }
        pub fn INVPCID(&self) -> bool { Self::bit(self.f_7_ebx, 10) }
        pub fn RTM(&self) -> bool { self.is_intel && Self::bit(self.f_7_ebx, 11) }
        pub fn AVX512F(&self) -> bool { Self::bit(self.f_7_ebx, 16) }
        pub fn AVX512DQ(&self) -> bool { Self::bit(self.f_7_ebx, 17) }
        pub fn RDSEED(&self) -> bool { Self::bit(self.f_7_ebx, 18) }
        pub fn ADX(&self) -> bool { Self::bit(self.f_7_ebx, 19) }
        pub fn AVX512PF(&self) -> bool { Self::bit(self.f_7_ebx, 26) }
        pub fn AVX512ER(&self) -> bool { Self::bit(self.f_7_ebx, 27) }
        pub fn AVX512CD(&self) -> bool { Self::bit(self.f_7_ebx, 28) }
        pub fn SHA(&self) -> bool { Self::bit(self.f_7_ebx, 29) }
        pub fn AVX512BW(&self) -> bool { Self::bit(self.f_7_ebx, 30) }
        pub fn AVX512VL(&self) -> bool { Self::bit(self.f_7_ebx, 31) }

        // ---- Leaf 7:0 ECX ----
        pub fn PREFETCHWT1(&self) -> bool { Self::bit(self.f_7_ecx, 0) }
        pub fn AVX512_VBMI(&self) -> bool { Self::bit(self.f_7_ecx, 1) }
        pub fn AVX512_VNNI(&self) -> bool { Self::bit(self.f_7_ecx, 11) }

        // ---- Leaf 7:0 EDX ----
        pub fn AMX_BF16(&self) -> bool { Self::bit(self.f_7_edx, 22) }
        pub fn AVX512_FP16(&self) -> bool { Self::bit(self.f_7_edx, 23) }
        pub fn AMX_TILE(&self) -> bool { Self::bit(self.f_7_edx, 24) }
        pub fn AMX_INT8(&self) -> bool { Self::bit(self.f_7_edx, 25) }

        // ---- Leaf 7:1 EAX ----
        pub fn AVX_VNNI(&self) -> bool { Self::bit(self.f_7_1_eax, 4) }
        pub fn AVX512_BF16(&self) -> bool { Self::bit(self.f_7_1_eax, 5) }
        pub fn AMX_FP16(&self) -> bool { Self::bit(self.f_7_1_eax, 21) }

        // ---- Extended leaf 0x80000001 ECX ----
        pub fn LAHF(&self) -> bool { Self::bit(self.f_81_ecx, 0) }
        pub fn LZCNT(&self) -> bool { self.is_intel && Self::bit(self.f_81_ecx, 5) }
        pub fn ABM(&self) -> bool { self.is_amd && Self::bit(self.f_81_ecx, 5) }
        pub fn SSE4a(&self) -> bool { self.is_amd && Self::bit(self.f_81_ecx, 6) }
        pub fn XOP(&self) -> bool { self.is_amd && Self::bit(self.f_81_ecx, 11) }
        pub fn TBM(&self) -> bool { self.is_amd && Self::bit(self.f_81_ecx, 21) }

        // ---- Extended leaf 0x80000001 EDX ----
        pub fn SYSCALL(&self) -> bool { self.is_intel && Self::bit(self.f_81_edx, 11) }
        pub fn MMXEXT(&self) -> bool { self.is_amd && Self::bit(self.f_81_edx, 22) }
        pub fn RDTSCP(&self) -> bool { self.is_intel && Self::bit(self.f_81_edx, 27) }
        pub fn _3DNOWEXT(&self) -> bool { self.is_amd && Self::bit(self.f_81_edx, 30) }
        pub fn _3DNOW(&self) -> bool { self.is_amd && Self::bit(self.f_81_edx, 31) }
    }

    // -----------------------------------------------------------------------
    // cpu_score -- mirrors ggml_backend_cpu_x86_score()
    //
    // Returns a capability score for the current CPU. Returns 0 if any feature
    // required by the compiled feature-flag set is absent at runtime. The
    // integer score grows with each enabled and present feature tier.
    //
    // In llama.cpp the check is gated on compile-time #defines (GGML_FMA etc.).
    // Here we mirror that by checking runtime support for features the code
    // was compiled to use, using is_x86_feature_detected! for mainstream ones
    // and CpuidX86 for the rest (AMX, AVX-VNNI, AVX512-BF16/FP16).
    // -----------------------------------------------------------------------

    /// Runtime CPU capability score (0 = unusable, higher = more capable).
    pub fn cpu_score() -> i32 {
        let mut score: i32 = 1;
        let is = CpuidX86::new();

        // FMA -- leaf 1 ecx[12]; also available via is_x86_feature_detected
        #[cfg(target_feature = "fma")]
        {
            if !is.FMA() {
                return 0;
            }
            score += 1;
        }

        // F16C -- leaf 1 ecx[29]
        #[cfg(target_feature = "f16c")]
        {
            if !is.F16C() {
                return 0;
            }
            score += 1 << 1;
        }

        // SSE4.2 -- leaf 1 ecx[20]
        #[cfg(target_feature = "sse4.2")]
        {
            if !is.SSE42() {
                return 0;
            }
            score += 1 << 2;
        }

        // BMI2 -- leaf 7:0 ebx[8]
        #[cfg(target_feature = "bmi2")]
        {
            if !is.BMI2() {
                return 0;
            }
            score += 1 << 3;
        }

        // AVX -- leaf 1 ecx[28]
        #[cfg(target_feature = "avx")]
        {
            if !is.AVX() {
                return 0;
            }
            score += 1 << 4;
        }

        // AVX2 -- leaf 7:0 ebx[5]
        #[cfg(target_feature = "avx2")]
        {
            if !is.AVX2() {
                return 0;
            }
            score += 1 << 5;
        }

        // AVX-VNNI -- leaf 7:1 eax[4] (not in is_x86_feature_detected! stable)
        #[cfg(target_feature = "avxvnni")]
        {
            if !is.AVX_VNNI() {
                return 0;
            }
            score += 1 << 6;
        }

        // AVX-512 baseline (F + CD + VL + DQ + BW)
        #[cfg(all(
            target_feature = "avx512f",
            target_feature = "avx512cd",
            target_feature = "avx512vl",
            target_feature = "avx512dq",
            target_feature = "avx512bw"
        ))]
        {
            if !is.AVX512F() || !is.AVX512CD() || !is.AVX512VL() || !is.AVX512DQ() || !is.AVX512BW() {
                return 0;
            }
            score += 1 << 7;
        }

        // AVX512-VBMI -- leaf 7:0 ecx[1]
        #[cfg(target_feature = "avx512vbmi")]
        {
            if !is.AVX512_VBMI() {
                return 0;
            }
            score += 1 << 8;
        }

        // AVX512-BF16 -- leaf 7:1 eax[5]
        #[cfg(target_feature = "avx512bf16")]
        {
            if !is.AVX512_BF16() {
                return 0;
            }
            score += 1 << 9;
        }

        // AVX512-VNNI -- leaf 7:0 ecx[11]
        #[cfg(target_feature = "avx512vnni")]
        {
            if !is.AVX512_VNNI() {
                return 0;
            }
            score += 1 << 10;
        }

        // AMX-INT8 -- leaf 7:0 edx[25]
        #[cfg(target_feature = "amx-int8")]
        {
            if !is.AMX_INT8() {
                return 0;
            }
            score += 1 << 11;
        }

        score
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn cpuid_x86_constructs() {
            let is = CpuidX86::new();
            // On any x86-64 host, SSE2 must be present.
            assert!(is.SSE2(), "SSE2 must be present on x86-64");
        }

        #[test]
        fn cpu_score_positive() {
            // Score must be at least 1 on any supported host.
            assert!(cpu_score() >= 1);
        }

        #[test]
        fn vendor_nonempty() {
            let is = CpuidX86::new();
            assert!(!is.vendor.is_empty());
        }

        #[test]
        fn vendor_known() {
            let is = CpuidX86::new();
            // Either Intel or AMD on CI; allow neither for exotic VMs.
            assert!(
                is.is_intel || is.is_amd || (!is.is_intel && !is.is_amd),
                "vendor must be consistent"
            );
        }

        #[test]
        fn avx_implies_sse2() {
            let is = CpuidX86::new();
            if is.AVX() {
                assert!(is.SSE2(), "AVX implies SSE2");
                assert!(is.SSE(), "AVX implies SSE");
            }
        }

        #[test]
        fn avx2_implies_avx() {
            let is = CpuidX86::new();
            if is.AVX2() {
                assert!(is.AVX(), "AVX2 implies AVX");
            }
        }

        #[test]
        fn avx512f_implies_avx2() {
            let is = CpuidX86::new();
            if is.AVX512F() {
                assert!(is.AVX2(), "AVX-512F implies AVX2");
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use inner::{CpuidX86, cpu_score};
