//! CPU-side FP8 (e4m3) prefill-path simulation.
//!
//! Purpose (2026-11-18, mid-precision prefill investigation): quantify,
//! BEFORE any GPU kernel work, the quality effect of the planned
//! e4m3-activation x e4m3-weight GEMM class -- the middle ground between
//! TEMPURA_PREFILL_PRECISION=exact (2508 tok/s pp128, MMLU 62.8% /
//! 98.0% agreement) and the MXFP4 default (7790 tok/s, 48.8% / 58.0% --
//! quality-FAILING). The hypothesis under test: the quality failure is the
//! E2M1 activation quantization (0-1 mantissa bits, rel~0.1 logit band)
//! plus the Q6K->MXFP4 weight double-quantization; e4m3 (3 mantissa bits,
//! own exponent) should shrink the band ~4-16x.
//!
//! The simulation runs INSIDE the real CPU quantized forward: every
//! Q4K/Q6K projection rounds its activations and/or dequantized weights
//! through e4m3 (the shared bit-exact helpers in crate::e4m3 -- the same
//! encoding the e4m3 KV pool uses) and then computes the GEMM in f32.
//! That is the numerics class of a Blackwell e4m3 x e4m3 mma with f32
//! accumulation, modulo sum order (which the MMLU choice metric is
//! robust to by design).
//!
//! This is NOT an optimized kernel and is default-off: the dispatch in
//! crate::quantized::matmulq_dispatch() returns the real
//! CpuMatMulQ4KOpt unless TEMPURA_FP8_SIM selects a simulation mode.
//! tools/fp8-sim drives it over the MMLU sample and reports the
//! choice-agreement / logit-band vs the unmodified CPU reference.

use crate::e4m3::f32_to_e4m3_bits;
use crate::quantized::{f16_to_f32, q4k_dequant_block, CpuMatMulQ4KOpt};
use crate::{
    KernelError, MatMulQ, Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE, Q6K_BLOCK_BYTES, Q6K_BLOCK_SIZE, Result,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Hardware-faithful e4m3 decode: subnormals preserved (man * 2^-9),
/// unlike the KV-cache helper (which flushes them -- a KV convention).
/// 0x7F/0xFF are the e4m3fn NaN encodings.
#[inline]
pub fn hw_e4m3_bits_to_f32(b: u8) -> f32 {
    let neg = b & 0x80 != 0;
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as i32;
    if exp == 15 && man == 7 {
        return f32::NAN;
    }
    let v = if exp == 0 {
        man as f32 * 2.0f32.powi(-9)
    } else {
        (1.0 + man as f32 / 8.0) * 2.0f32.powi(exp - 7)
    };
    if neg {
        -v
    } else {
        v
    }
}

/// Hardware-faithful e4m3 encode: RNE onto the full e4m3 grid INCLUDING
/// subnormals (steps of 2^-9 below the 2^-6 min normal), saturation to
/// +-448, NaN -> 0x7F. The normal range delegates to the shared
/// `f32_to_e4m3_bits` (RNE, exact there).
#[inline]
pub fn hw_f32_to_e4m3_bits(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let x = x.clamp(-448.0, 448.0);
    let ax = x.abs();
    if ax < 2.0f32.powi(-6) {
        let sign = if x.is_sign_negative() { 0x80u8 } else { 0x00 };
        // subnormal grid: RNE onto multiples of 2^-9 in [0, 8)
        let steps = ax * 512.0;
        let f = steps.floor();
        let frac = steps - f;
        let q = if frac > 0.5 || (frac == 0.5 && (f as i32) % 2 == 1) {
            f + 1.0
        } else {
            f
        };
        if q >= 8.0 {
            return sign | 0x08; // rounded up to the min normal
        }
        return sign | q as u8;
    }
    f32_to_e4m3_bits(x)
}

/// e4m3 round-trip (hardware semantics: subnormals preserved): the operand
/// value a Blackwell FP8 mma would see.
#[inline]
pub fn e4m3_roundtrip(x: f32) -> f32 {
    hw_e4m3_bits_to_f32(hw_f32_to_e4m3_bits(x))
}

/// FTZ variant: the KV-cache helper convention (subnormals flush to signed
/// zero). Selected with the FP8_SIM_FTZ mode bit -- relevant if the GPU
/// kernel ends up FTZ (e.g. an oxide-dialect conversion that reuses the
/// shared helpers).
#[inline]
pub fn e4m3_roundtrip_ftz(x: f32) -> f32 {
    crate::e4m3::e4m3_bits_to_f32(f32_to_e4m3_bits(x))
}

// -- mode bits ----------------------------------------------------------------

pub const FP8_SIM_ACT: u8 = 1; // round activations through e4m3
pub const FP8_SIM_WEIGHT: u8 = 2; // round dequantized weights through e4m3
pub const FP8_SIM_WSCALE: u8 = 4; // per-tensor scale on the weight side
pub const FP8_SIM_AROWSCALE: u8 = 8; // per-row scale on the activation side
pub const FP8_SIM_FTZ: u8 = 16; // KV-style subnormal flush instead of hw semantics
pub const FP8_SIM_NOOP: u8 = 32; // activate the Sim dispatch WITHOUT any quantization
// (path-switch control: measures the generic-route vs x8-route divergence)

static FP8_SIM_MODE: AtomicU8 = AtomicU8::new(u8::MAX); // MAX = not yet parsed

fn parse_env_mode() -> u8 {
    let Ok(spec) = std::env::var("TEMPURA_FP8_SIM") else {
        return 0;
    };
    let mut mode = 0u8;
    for part in spec.split('+') {
        match part.trim() {
            "off" | "" => {}
            "act" => mode |= FP8_SIM_ACT,
            "weight" => mode |= FP8_SIM_WEIGHT,
            "wscale" => mode |= FP8_SIM_WSCALE,
            "arowscale" => mode |= FP8_SIM_AROWSCALE,
            "ftz" => mode |= FP8_SIM_FTZ,
            "noop" => mode |= FP8_SIM_NOOP,
            other => {
                eprintln!("[fp8_sim] ignoring unknown TEMPURA_FP8_SIM part {other:?}");
            }
        }
    }
    mode
}

/// Current simulation mode (env TEMPURA_FP8_SIM, cached; overridable
/// in-process for harnesses).
pub fn fp8_sim_mode() -> u8 {
    let m = FP8_SIM_MODE.load(Ordering::Relaxed);
    if m != u8::MAX {
        return m;
    }
    let parsed = parse_env_mode();
    FP8_SIM_MODE.store(parsed, Ordering::Relaxed);
    parsed
}

/// In-process override (tools/fp8-sim): avoids env mutation in a
/// multithreaded process.
pub fn fp8_sim_set_mode(mode: u8) {
    FP8_SIM_MODE.store(mode, Ordering::Relaxed);
}

// -- weight conversion cache ---------------------------------------------------
//
// Keyed on (pointer, byte length, conversion config). Weight tensors are
// owned by the model's QuantizedWeights and stable for the process lifetime;
// the pointer key is therefore safe in the harness/single-model context this
// simulation targets. Do NOT reuse this pattern for engine production paths
// with tensor churn.

struct WeightCache(OnceLock<Mutex<HashMap<(usize, usize, u8), Arc<Vec<f32>>>>>);

impl WeightCache {
    fn get() -> &'static Mutex<HashMap<(usize, usize, u8), Arc<Vec<f32>>>> {
        static CACHE: WeightCache = WeightCache(OnceLock::new());
        CACHE.0.get_or_init(|| Mutex::new(HashMap::new()))
    }
}

fn dequant_rows(b: &[u8], n: usize, k: usize, q6k: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; n * k];
    let out_ptr = out.as_mut_ptr();
    let b_ptr = b.as_ptr();
    let (block_bytes, block_size) = if q6k {
        (Q6K_BLOCK_BYTES, Q6K_BLOCK_SIZE)
    } else {
        (Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE)
    };
    let bpr = k / block_size;
    crate::inference_pool::par_run(n, |start, end| {
        let mut tmp = [0.0f32; Q6K_BLOCK_SIZE]; // Q6K block >= Q4K block
        for j in start..end {
            let row = unsafe { std::slice::from_raw_parts_mut(out_ptr.add(j * k), k) };
            for blk in 0..bpr {
                let block = unsafe {
                    std::slice::from_raw_parts(
                        b_ptr.add((j * bpr + blk) * block_bytes),
                        block_bytes,
                    )
                };
                let dst = &mut row[blk * block_size..(blk + 1) * block_size];
                if q6k {
                    q6k_dequant_block(block, dst);
                } else {
                    q4k_dequant_block(block, &mut tmp);
                    dst.copy_from_slice(&tmp[..block_size]);
                }
            }
        }
    });
    out
}

/// Q6_K block -> f32, the q6k_dot_block layout verbatim (per-16-element
/// scales, raw 6-bit values offset by 32).
pub fn q6k_dequant_block(block: &[u8], output: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    for half in 0..2 {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let y_off = half * 128;

        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

            let sc0 = scales[half * 8 + is] as i8 as f32;
            let sc1 = scales[half * 8 + is + 2] as i8 as f32;
            let sc2 = scales[half * 8 + is + 4] as i8 as f32;
            let sc3 = scales[half * 8 + is + 6] as i8 as f32;

            output[y_off + l] = d * sc0 * q1 as f32;
            output[y_off + l + 32] = d * sc1 * q2 as f32;
            output[y_off + l + 64] = d * sc2 * q3 as f32;
            output[y_off + l + 96] = d * sc3 * q4 as f32;
        }
    }
}

#[inline]
fn rt(x: f32, ftz: bool) -> f32 {
    if ftz {
        e4m3_roundtrip_ftz(x)
    } else {
        e4m3_roundtrip(x)
    }
}

/// Apply the e4m3 round-trip in place. With `per_tensor_scale`, the values
/// are first scaled so the tensor absmax lands mid-range in e4m3 (224 of
/// 448) -- lifting small magnitudes out of the subnormal range, where e4m3
/// loses mantissa bits. The scale is divided back out, so the result stays
/// in the original units.
fn apply_e4m3(buf: &mut [f32], per_tensor_scale: bool, ftz: bool) {
    if per_tensor_scale {
        let mut amax = 0.0f32;
        for &v in buf.iter() {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        if amax > 0.0 && amax.is_finite() {
            let s = 224.0 / amax;
            for v in buf.iter_mut() {
                *v = rt(*v * s, ftz) / s;
            }
            return;
        }
    }
    for v in buf.iter_mut() {
        *v = rt(*v, ftz);
    }
}

/// Per-row (per-token) activation scale variant: each row's absmax lands at
/// 224 before quantization. Only meaningful for activations.
fn apply_e4m3_rows(buf: &mut [f32], k: usize, ftz: bool) {
    for row in buf.chunks_mut(k) {
        let mut amax = 0.0f32;
        for &v in row.iter() {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        if amax > 0.0 && amax.is_finite() {
            let s = 224.0 / amax;
            for v in row.iter_mut() {
                *v = rt(*v * s, ftz) / s;
            }
        } else {
            for v in row.iter_mut() {
                *v = rt(*v, ftz);
            }
        }
    }
}

fn converted_weights(b: &[u8], n: usize, k: usize, q6k: bool, mode: u8) -> Arc<Vec<f32>> {
    let quant = mode & FP8_SIM_WEIGHT != 0;
    let wscale = mode & FP8_SIM_WSCALE != 0;
    let ftz = mode & FP8_SIM_FTZ != 0;
    let cfg = (quant as u8)
        | ((wscale as u8) << 1)
        | ((q6k as u8) << 2)
        | ((ftz as u8) << 3);
    let key = (b.as_ptr() as usize, b.len(), cfg);
    let cache = WeightCache::get();
    if let Ok(map) = cache.lock() {
        if let Some(hit) = map.get(&key) {
            return hit.clone();
        }
    }
    let mut w = dequant_rows(b, n, k, q6k);
    if quant {
        apply_e4m3(&mut w, wscale, ftz);
    }
    let arc = Arc::new(w);
    if let Ok(mut map) = cache.lock() {
        map.insert(key, arc.clone());
    }
    arc
}

/// f32 GEMM with B in [n, k] row-major (the CpuMatMul::forward_b_transposed
/// layout), Rayon-parallel via the gemm crate.
fn gemm_bt(a: &[f32], b: &[f32], output: &mut [f32], m: usize, n: usize, k: usize) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            output.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            k as isize,
            1,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }
}

/// The simulated quantized GEMM. `mode` comes from fp8_sim_mode(); a mode
/// of 0 must never reach here (the dispatch returns the real kernel).
fn sim_gemm(
    a: &[f32],
    b_quantized: &[u8],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    q6k: bool,
    mode: u8,
) -> Result<()> {
    if a.len() != m * k {
        return Err(KernelError::DimensionMismatch {
            expected: m * k,
            got: a.len(),
        });
    }
    if output.len() != m * n {
        return Err(KernelError::DimensionMismatch {
            expected: m * n,
            got: output.len(),
        });
    }
    let (block_bytes, block_size) = if q6k {
        (Q6K_BLOCK_BYTES, Q6K_BLOCK_SIZE)
    } else {
        (Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE)
    };
    if !k.is_multiple_of(block_size) {
        return Err(KernelError::InvalidConfig(format!(
            "k={k} must be divisible by block size {block_size}"
        )));
    }
    let blocks_per_row = k / block_size;
    let expected_b_bytes = n * blocks_per_row * block_bytes;
    if b_quantized.len() != expected_b_bytes {
        return Err(KernelError::DimensionMismatch {
            expected: expected_b_bytes,
            got: b_quantized.len(),
        });
    }

    let weights = converted_weights(b_quantized, n, k, q6k, mode);

    let mut act_scratch: Vec<f32> = Vec::new();
    let a_used: &[f32] = if mode & FP8_SIM_ACT != 0 {
        let ftz = mode & FP8_SIM_FTZ != 0;
        act_scratch = a.to_vec();
        if mode & FP8_SIM_AROWSCALE != 0 {
            apply_e4m3_rows(&mut act_scratch, k, ftz);
        } else {
            for v in act_scratch.iter_mut() {
                *v = rt(*v, ftz);
            }
        }
        &act_scratch
    } else {
        a
    };

    gemm_bt(a_used, &weights, output, m, n, k);
    Ok(())
}

/// Simulation MatMulQ: implements the Q4K/Q6K entries with the e4m3
/// round-trips; every other format delegates to the real
/// CpuMatMulQ4KOpt unchanged.
pub struct Fp8SimMatMulQ;

impl MatMulQ for Fp8SimMatMulQ {
    fn forward_b_transposed_q4k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        sim_gemm(a, b_quantized, output, m, n, k, false, fp8_sim_mode())
    }

    fn forward_b_transposed_q6k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        sim_gemm(a, b_quantized, output, m, n, k, true, fp8_sim_mode())
    }

    fn forward_b_transposed_tq2(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        CpuMatMulQ4KOpt.forward_b_transposed_tq2(a, b_quantized, output, m, n, k)
    }

    fn forward_b_transposed_i2s(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        CpuMatMulQ4KOpt.forward_b_transposed_i2s(a, b_quantized, output, m, n, k)
    }

    fn forward_i2s_preq(
        &self,
        preq_q8: &[i8],
        preq_scale: f32,
        preq_sum: i32,
        b_quantized: &[u8],
        output: &mut [f32],
        n: usize,
        k: usize,
    ) -> Result<()> {
        CpuMatMulQ4KOpt.forward_i2s_preq(
            preq_q8,
            preq_scale,
            preq_sum,
            b_quantized,
            output,
            n,
            k,
        )
    }

    fn forward_i2s_lut(
        &self,
        a: &[f32],
        b_transposed: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        CpuMatMulQ4KOpt.forward_i2s_lut(a, b_transposed, output, m, n, k)
    }
}

// The e4m3 round-trip error is a per-operand property (half-ulp of a
// 3-mantissa format = 2^-4 worst case, proptest-gated in crate::e4m3);
// the dot-level bound is data-dependent (cancellative sums), so the
// equivalence claim is carried by the runtime comparison: the
// tools/fp8-sim MMLU/logit-band harness vs the unmodified CPU reference.
impl crate::VerifiedEquivalent<CpuMatMulQ4KOpt> for Fp8SimMatMulQ {
    fn max_relative_error() -> f64 {
        0.0625
    }
    fn proof_name() -> &'static str {
        "proptest:e4m3_roundtrip_bounds + dualpipeline:tools/fp8-sim (MMLU agreement \
         + logit band vs unmodified CPU reference)"
    }
    fn verification_tier() -> crate::VerificationTier {
        crate::VerificationTier::Tested
    }
}

/// Engine-facing dispatch: the real kernel unless a simulation mode is
/// active. Default path cost is one atomic load (plus a one-time env parse).
pub enum MatMulQDispatch {
    Real(CpuMatMulQ4KOpt),
    Sim(Fp8SimMatMulQ),
}

impl MatMulQ for MatMulQDispatch {
    fn forward_b_transposed_q4k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        match self {
            MatMulQDispatch::Real(kk) => kk.forward_b_transposed_q4k(a, b_quantized, output, m, n, k),
            MatMulQDispatch::Sim(kk) => kk.forward_b_transposed_q4k(a, b_quantized, output, m, n, k),
        }
    }

    fn forward_b_transposed_q6k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        match self {
            MatMulQDispatch::Real(kk) => kk.forward_b_transposed_q6k(a, b_quantized, output, m, n, k),
            MatMulQDispatch::Sim(kk) => kk.forward_b_transposed_q6k(a, b_quantized, output, m, n, k),
        }
    }

    fn forward_b_transposed_tq2(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        match self {
            MatMulQDispatch::Real(kk) => kk.forward_b_transposed_tq2(a, b_quantized, output, m, n, k),
            MatMulQDispatch::Sim(kk) => kk.forward_b_transposed_tq2(a, b_quantized, output, m, n, k),
        }
    }

    fn forward_b_transposed_i2s(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        match self {
            MatMulQDispatch::Real(kk) => kk.forward_b_transposed_i2s(a, b_quantized, output, m, n, k),
            MatMulQDispatch::Sim(kk) => kk.forward_b_transposed_i2s(a, b_quantized, output, m, n, k),
        }
    }

    fn forward_i2s_preq(
        &self,
        preq_q8: &[i8],
        preq_scale: f32,
        preq_sum: i32,
        b_quantized: &[u8],
        output: &mut [f32],
        n: usize,
        k: usize,
    ) -> Result<()> {
        match self {
            MatMulQDispatch::Real(kk) => {
                kk.forward_i2s_preq(preq_q8, preq_scale, preq_sum, b_quantized, output, n, k)
            }
            MatMulQDispatch::Sim(kk) => {
                kk.forward_i2s_preq(preq_q8, preq_scale, preq_sum, b_quantized, output, n, k)
            }
        }
    }

    fn forward_i2s_lut(
        &self,
        a: &[f32],
        b_transposed: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        match self {
            MatMulQDispatch::Real(kk) => kk.forward_i2s_lut(a, b_transposed, output, m, n, k),
            MatMulQDispatch::Sim(kk) => kk.forward_i2s_lut(a, b_transposed, output, m, n, k),
        }
    }
}

pub fn matmulq_dispatch() -> MatMulQDispatch {
    if fp8_sim_mode() == 0 {
        MatMulQDispatch::Real(CpuMatMulQ4KOpt)
    } else {
        MatMulQDispatch::Sim(Fp8SimMatMulQ)
    }
}

// Inherent-method delegation: forward_quantized also uses CpuMatMulQ4KOpt's
// non-trait methods. The m==1/decode-class entries (preq, fused) stay on the
// real kernel in Sim mode too (the fp8 target is the PREFILL projections);
// forward_prefill_q4kx8 DECLINES so callers fall back to the generic
// qmm_indexed route, which the simulation covers.
impl MatMulQDispatch {
    pub fn forward_q4k_preq(
        &self,
        preq: &crate::quantized::PreQuantizedActivations,
        a_f32: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        n: usize,
    ) -> Result<()> {
        CpuMatMulQ4KOpt.forward_q4k_preq(preq, a_f32, b_quantized, output, n)
    }

    pub fn forward_q6k_preq(
        &self,
        preq: &crate::quantized::PreQuantizedActivations,
        a_f32: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        n: usize,
    ) -> Result<()> {
        CpuMatMulQ4KOpt.forward_q6k_preq(preq, a_f32, b_quantized, output, n)
    }

    pub fn forward_prefill_q4kx8(
        &self,
        a: &[f32],
        b_repacked: &[crate::ggml::blocks::BlockQ4Kx8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> bool {
        match self {
            MatMulQDispatch::Real(kk) => kk.forward_prefill_q4kx8(a, b_repacked, output, m, n, k),
            MatMulQDispatch::Sim(_) => false, // force the simulated generic route
        }
    }

    pub fn fused_q4k_preq_2(
        &self,
        preq: &crate::quantized::PreQuantizedActivations,
        b1: &[u8],
        out1: &mut [f32],
        n1: usize,
        b2: &[u8],
        out2: &mut [f32],
        n2: usize,
    ) {
        CpuMatMulQ4KOpt.fused_q4k_preq_2(preq, b1, out1, n1, b2, out2, n2)
    }

    pub fn fused_q4k_preq_3(
        &self,
        preq: &crate::quantized::PreQuantizedActivations,
        b1: &[u8],
        out1: &mut [f32],
        n1: usize,
        b2: &[u8],
        out2: &mut [f32],
        n2: usize,
        b3: &[u8],
        out3: &mut [f32],
        n3: usize,
    ) {
        CpuMatMulQ4KOpt.fused_q4k_preq_3(preq, b1, out1, n1, b2, out2, n2, b3, out3, n3)
    }

    pub fn fused_q6k_preq_2(
        &self,
        a_f32: &[f32],
        b1: &[u8],
        out1: &mut [f32],
        n1: usize,
        b2: &[u8],
        out2: &mut [f32],
        n2: usize,
        k: usize,
    ) {
        CpuMatMulQ4KOpt.fused_q6k_preq_2(a_f32, b1, out1, n1, b2, out2, n2, k)
    }

    pub fn fused_q6k_preq_3(
        &self,
        a_f32: &[f32],
        b1: &[u8],
        out1: &mut [f32],
        n1: usize,
        b2: &[u8],
        out2: &mut [f32],
        n2: usize,
        b3: &[u8],
        out3: &mut [f32],
        n3: usize,
        k: usize,
    ) {
        CpuMatMulQ4KOpt.fused_q6k_preq_3(a_f32, b1, out1, n1, b2, out2, n2, b3, out3, n3, k)
    }

    pub fn fused_q4k_2_f16_1_preq(
        &self,
        preq: &crate::quantized::PreQuantizedActivations,
        b1: &[u8],
        out1: &mut [f32],
        n1: usize,
        b2: &[u8],
        out2: &mut [f32],
        n2: usize,
        f16_bytes: &[u8],
        out3: &mut [f32],
        n3: usize,
        k: usize,
        a_f32: &[f32],
    ) {
        CpuMatMulQ4KOpt.fused_q4k_2_f16_1_preq(
            preq, b1, out1, n1, b2, out2, n2, f16_bytes, out3, n3, k, a_f32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_gemm_matches_cpu_q4k_within_e4m3_band() {
        // random native-GGUF Q4K blocks (uniform data cannot catch layout
        // bugs -- the standing rule); the sim must agree with the real
        // CpuMatMulQ4KOpt within the e4m3 operand band, gated
        // cancellation-aware: |got - ref| <= 0.07 * sum|terms| + 1e-6.
        use crate::quantized::{q4k_dot_block, CpuMatMulQ4KOpt, MatMulQ};
        let (m, n, k) = (4usize, 64usize, 256usize);
        let bpr = k / Q4K_BLOCK_SIZE;
        let mut b = vec![0u8; n * bpr * Q4K_BLOCK_BYTES];
        let mut x = 0xdeadbeefu32;
        for blk in b.chunks_mut(Q4K_BLOCK_BYTES) {
            // realistic magnitudes: d=0.01, dmin=0.001 (weights land in
            // roughly +-0.5 like real Llama Q4K tensors)
            blk[0..2].copy_from_slice(
                &crate::ggml::helpers::f32_to_f16(0.01).to_le_bytes(),
            );
            blk[2..4].copy_from_slice(
                &crate::ggml::helpers::f32_to_f16(0.001).to_le_bytes(),
            );
            for byte in blk[4..].iter_mut() {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                *byte = (x >> 24) as u8;
            }
        }
        // random activations in a post-RMSNorm-ish range
        let mut a = vec![0.0f32; m * k];
        for v in a.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((x >> 8) as i32 as f32 / 8388608.0) - 1.0;
        }

        let mut ref_out = vec![0.0f32; m * n];
        CpuMatMulQ4KOpt
            .forward_b_transposed_q4k(&a, &b, &mut ref_out, m, n, k)
            .unwrap();

        let saved = fp8_sim_mode();
        fp8_sim_set_mode(FP8_SIM_ACT | FP8_SIM_WEIGHT | FP8_SIM_WSCALE);
        let mut sim_out = vec![0.0f32; m * n];
        Fp8SimMatMulQ
            .forward_b_transposed_q4k(&a, &b, &mut sim_out, m, n, k)
            .unwrap();
        fp8_sim_set_mode(saved);

        let mut worst = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                // cancellation-aware reference: sum of |a * w| per output
                let mut abssum = 0.0f64;
                for blk in 0..bpr {
                    let block = &b[(j * bpr + blk) * Q4K_BLOCK_BYTES..][..Q4K_BLOCK_BYTES];
                    let a_slice = &a[i * k + blk * Q4K_BLOCK_SIZE..][..Q4K_BLOCK_SIZE];
                    // |terms| via the same dot on sign-flipped activations is
                    // awkward; approximate with |dot| terms using dequant
                    let mut deq = [0.0f32; Q4K_BLOCK_SIZE];
                    q4k_dequant_block(block, &mut deq);
                    for t in 0..Q4K_BLOCK_SIZE {
                        abssum += (a_slice[t].abs() * deq[t].abs()) as f64;
                    }
                }
                let diff = (sim_out[i * n + j] - ref_out[i * n + j]).abs() as f64;
                let gate = 0.07 * abssum + 1e-6;
                let ratio = diff / gate;
                if ratio > worst {
                    worst = ratio;
                }
            }
        }
        assert!(
            worst <= 1.0,
            "sim vs real max err/gate = {worst} (layout bug if >> 1)"
        );
    }

    #[test]
    fn q6k_dequant_matches_dot_block() {
        // dequant must reproduce q6k_dot_block's value convention: dotting
        // the dequantized block against a one-hot activation picks out the
        // element q6k_dot_block computes for the same index.
        use crate::quantized::q6k_dot_block;
        let mut block = vec![0u8; Q6K_BLOCK_BYTES];
        // deterministic pseudo-random block
        let mut x = 0x12345678u32;
        for b in block.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (x >> 24) as u8;
        }
        block[208] = 0x00;
        block[209] = 0x3c; // d = 1.0
        let mut deq = vec![0.0f32; Q6K_BLOCK_SIZE];
        q6k_dequant_block(&block, &mut deq);
        for hot in [0usize, 1, 63, 64, 100, 200, 255] {
            let mut a = vec![0.0f32; Q6K_BLOCK_SIZE];
            a[hot] = 1.0;
            let dot = q6k_dot_block(&block, &a);
            assert!(
                (dot - deq[hot]).abs() <= 1e-5 * deq[hot].abs().max(1.0),
                "hot={hot}: dot={dot} deq={}",
                deq[hot]
            );
        }
    }

    #[test]
    fn e4m3_roundtrip_is_identity_on_grid() {
        // every finite encoding (incl. subnormals) round-trips to the same
        // byte; the NaN encodings 0x7f/0xff are excluded (the writer
        // normalizes NaN to 0x7f)
        for b in 0u16..=255 {
            let byte = b as u8;
            if byte & 0x7F == 0x7F {
                continue; // NaN encodings
            }
            let v = hw_e4m3_bits_to_f32(byte);
            let back = hw_f32_to_e4m3_bits(v);
            assert_eq!(back, byte, "b={b:#04x} v={v:e}");
        }
    }

    #[test]
    fn hw_e4m3_decode_known_values() {
        assert_eq!(hw_e4m3_bits_to_f32(0x00), 0.0);
        assert_eq!(hw_e4m3_bits_to_f32(0x80), -0.0);
        assert_eq!(hw_e4m3_bits_to_f32(0x01), 2.0f32.powi(-9)); // min subnormal
        assert_eq!(hw_e4m3_bits_to_f32(0x07), 7.0 * 2.0f32.powi(-9));
        assert_eq!(hw_e4m3_bits_to_f32(0x08), 2.0f32.powi(-6)); // min normal
        assert_eq!(hw_e4m3_bits_to_f32(0x38), 1.0);
        assert_eq!(hw_e4m3_bits_to_f32(0x7E), 448.0); // max finite
        assert!(hw_e4m3_bits_to_f32(0x7F).is_nan());
    }

    #[test]
    fn hw_e4m3_encode_edges() {
        assert_eq!(hw_f32_to_e4m3_bits(0.0), 0x00);
        assert_eq!(hw_f32_to_e4m3_bits(-0.0), 0x80);
        // RNE tie between subnormal 6 and 7 -> even (6)
        assert_eq!(hw_f32_to_e4m3_bits(6.5 * 2.0f32.powi(-9)), 0x06);
        // tie between 7 and 8(min normal) -> even (8 = min normal)
        assert_eq!(hw_f32_to_e4m3_bits(7.5 * 2.0f32.powi(-9)), 0x08);
        assert_eq!(hw_f32_to_e4m3_bits(1e30), 0x7E);
        assert!(hw_f32_to_e4m3_bits(f32::NAN) == 0x7F);
    }

    #[test]
    fn roundtrip_rel_error_bounded() {
        // every finite f32 in the representable range round-trips with
        // worst-case relative error 2^-4 (half-ulp of a 3-mantissa format)
        let mut x = 0.1f32;
        let mut worst = 0.0f32;
        for i in 0..20000 {
            x = (x * 1.0001 + 0.0003 * (i as f32 % 7.0)).fract().abs() * 400.0;
            if x == 0.0 || !x.is_finite() {
                continue;
            }
            let back = e4m3_roundtrip(x);
            let rel = ((back - x) / x).abs();
            if rel > worst {
                worst = rel;
            }
        }
        assert!(worst <= 0.0625 + 1e-6, "worst rel err {worst}");
    }
}

