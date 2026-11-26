//! CPU-side KV-cache precision-class simulation (TurboQuant go/no-go,
//! 2026-11-23 round).
//!
//! Purpose: quantify, BEFORE any GPU pool/kernel work, the quality effect
//! of storing K/V in the CpuPolarQuant class (group-wise min/max + b-bit
//! codes -- the turboquant.rs scheme) instead of f16/e4m3. The KV density
//! ladder so far: f16 passes (ppl ratio 1.0003), e4m3 passes (0.995); the
//! question is whether 2-4 bits + group scales hold the MMLU/ppl gates.
//!
//! The simulation runs INSIDE the real CPU quantized forward: at the KV
//! storage boundary (right before `append_kv` in the engine's attention
//! paths) every (token, head) K/V row of `head_dim` elements is round-
//! tripped through the EXACT CpuPolarQuant arithmetic (same min/max fold,
//! same range guard, same rounding). The cache therefore stores, and every
//! later attention read sees, precisely the values a polar-quant pool
//! would serve -- the exact serving semantics of quantize-on-write.
//!
//! This is NOT an optimized path and is default-off (KV_SIM_BITS == 0):
//! the engine hook is a single atomic load when disabled. tools/fp8-sim
//! drives the sweep (bits x group_size) over the MMLU sample + WikiText
//! ppl and reports accuracy / choice agreement / logit band / ppl ratio
//! vs the unmodified reference.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

static KV_SIM_BITS: AtomicU8 = AtomicU8::new(0);
static KV_SIM_GROUP: AtomicUsize = AtomicUsize::new(0);

/// Activate the simulation: K/V rows round-trip through the polar-quant
/// class with `bits` code bits and groups of `group_size` elements
/// (within each `head_dim` row, exactly like `CpuPolarQuant`).
pub fn kv_sim_set(bits: u8, group_size: usize) {
    assert!(bits >= 1 && bits <= 8, "kv sim bits must be 1..=8");
    assert!(group_size > 0, "kv sim group_size must be > 0");
    KV_SIM_BITS.store(bits, Ordering::Relaxed);
    KV_SIM_GROUP.store(group_size, Ordering::Relaxed);
}

pub fn kv_sim_off() {
    KV_SIM_BITS.store(0, Ordering::Relaxed);
}

#[inline]
pub fn kv_sim_enabled() -> bool {
    KV_SIM_BITS.load(Ordering::Relaxed) != 0
}

#[inline]
pub fn kv_sim_config() -> Option<(u8, usize)> {
    let bits = KV_SIM_BITS.load(Ordering::Relaxed);
    if bits == 0 {
        None
    } else {
        Some((bits, KV_SIM_GROUP.load(Ordering::Relaxed)))
    }
}

/// Round-trip ONE group of values through the CpuPolarQuant arithmetic.
/// Mirrors turboquant.rs `compress` + `decompress` exactly (same min/max
/// fold, same `range < 1e-30` guard, same rounding) so the simulation
/// cannot drift from the real compressor.
#[inline]
fn roundtrip_group(group: &mut [f32], bits: u8) {
    let levels = (1u32 << bits) - 1;
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &v in group.iter() {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }
    let range = max_val - min_val;
    if range < 1e-30 {
        for v in group.iter_mut() {
            *v = min_val;
        }
        return;
    }
    let levels_f = levels as f32;
    for v in group.iter_mut() {
        let normalized = (*v - min_val) / range;
        let q = (normalized * levels_f).round().min(levels_f) as u8;
        *v = min_val + (q as f32 / levels_f) * range;
    }
}

/// In-place polar-quant round-trip over a flat K (or V) buffer laid out
/// as consecutive `head_dim`-element rows (one row per (token, kv_head)):
/// the granularity a polar-quant KV pool would quantize at.
pub fn kv_sim_roundtrip(buf: &mut [f32], head_dim: usize) {
    let Some((bits, group_size)) = kv_sim_config() else {
        return;
    };
    debug_assert!(head_dim > 0 && buf.len() % head_dim == 0);
    for row in buf.chunks_mut(head_dim) {
        for group in row.chunks_mut(group_size) {
            roundtrip_group(group, bits);
        }
    }
}

/// Owned variant for call sites holding `&[f32]`.
pub fn kv_sim_roundtrip_owned(src: &[f32], head_dim: usize) -> Vec<f32> {
    let mut out = src.to_vec();
    kv_sim_roundtrip(&mut out, head_dim);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turboquant::{CompressedKV, KvCacheCompressor};

    // The simulation must be BIT-IDENTICAL to the real compressor's
    // round-trip: any drift invalidates the go/no-go.
    #[test]
    fn sim_matches_cpu_polar_quant_roundtrip() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(7);
        let head_dim = 128;
        let data: Vec<f32> = (0..head_dim * 17)
            .map(|_| rng.random_range(-3.0..3.0))
            .collect();

        for (bits, group) in [(2u8, 32usize), (3, 32), (3, 64), (4, 128), (2, 128)] {
            let pq = crate::turboquant::CpuPolarQuant::new(bits, group).unwrap();
            // CpuPolarQuant quantizes per head_dim vector.
            let mut expected = Vec::with_capacity(data.len());
            for row in data.chunks(head_dim) {
                let c: CompressedKV = pq.compress(row, head_dim).unwrap();
                expected.extend(pq.decompress(&c).unwrap());
            }

            let mut got = data.clone();
            kv_sim_set(bits, group);
            kv_sim_roundtrip(&mut got, head_dim);
            kv_sim_off();

            assert_eq!(got.len(), expected.len());
            for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "bits={bits} group={group}: mismatch at {i}: sim={a} pq={b}"
                );
            }
        }
    }

    // quantize -> dequantize -> quantize must be IDEMPOTENT for stored
    // values (the swap/mirror requantization requirement): a value that
    // already round-tripped must round-trip to itself bit-exactly.
    #[test]
    fn roundtrip_is_idempotent() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(11);
        let head_dim = 128;
        let data: Vec<f32> = (0..head_dim * 8)
            .map(|_| rng.random_range(-3.0..3.0))
            .collect();
        for (bits, group) in [(2u8, 32usize), (3, 64), (4, 128)] {
            let mut once = data.clone();
            kv_sim_set(bits, group);
            kv_sim_roundtrip(&mut once, head_dim);
            let mut twice = once.clone();
            kv_sim_roundtrip(&mut twice, head_dim);
            kv_sim_off();
            for (i, (a, b)) in once.iter().zip(&twice).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "bits={bits} group={group}: requantization not idempotent at {i}"
                );
            }
        }
    }

    #[test]
    fn off_is_identity() {
        let mut buf = vec![1.0f32, -2.5, 3.0, 0.0];
        kv_sim_off();
        kv_sim_roundtrip(&mut buf, 2);
        assert_eq!(buf, vec![1.0f32, -2.5, 3.0, 0.0]);
    }
}
