/// Warp-level matrix multiply-accumulate wrappers for sm_120/sm_121.
///
/// These wrap PTX `mma.sync.aligned` instructions via `ptx_asm!`, covering
/// the full precision landscape on consumer Blackwell GPUs:
///
/// - **Tier 1** (sm_80+): f16, bf16, tf32 — standard MMA
/// - **Tier 2** (sm_120): `kind::f8f6f4` — FP8, FP6, FP4 non-block-scaled
/// - **Tier 3** (sm_120a): `kind::mxf8f6f4` — MX block-scaled FP8/FP6
/// - **Tier 4** (sm_120a): `kind::mxf4nvf4` — NVFP4 block-scaled
///
/// All variants use the m16n8kN geometry with:
/// - D (output): 4 × f32 per thread
/// - A (input): packed fragments as u32
/// - B (input): packed fragments as u32
/// - C (accumulator): 4 × f32 per thread
///
/// Block-scaled variants (Tier 3-4) add scale factor operands.
pub mod tier1;
pub mod tier2;
pub mod tier3;
pub mod tier4;
