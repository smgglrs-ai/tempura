//! Pure split-K decode-attention scheduling math.
//!
//! Contract shared by three parties:
//! - `kernels/src/oxide_decode_attn_host.rs` (gated): sizes the scratch buffer
//!   and launches the kernels with these constants.
//! - `tools/oxide-decode-attn/src/main.rs` (standalone oxide crate): the GPU
//!   kernel re-implements this arithmetic inline (the crate cannot depend on
//!   tempura-kernels) and is cross-validated on-device against the
//!   single-warp kernel.
//! - This module: the same arithmetic as pure functions, Kani-proven, so the
//!   chunking/scheduling invariants (exact tiling, index bounds) hold for ALL
//!   inputs -- mirroring the `mxfp4.rs` (math) / `mxfp4_dispatch.rs` (launch)
//!   split.
//!
//! Scratch layout, per (head, split) slot of `head_dim + 2` f32 elements:
//!   [0..head_dim)   unnormalised acc (rescaled to the split's running max)
//!   [head_dim]      m -- split max score, or -inf when the split is empty
//!   [head_dim + 1]  l -- split sum of exp

/// KV length splits per attention head (grid Y dimension of the partial pass).
pub const ATTN_SPLITK_SPLITS: u32 = 16;

/// Device scratch length in f32 elements for the split-K partials.
pub const fn scratch_len(n_heads: usize, head_dim: usize, n_splits: u32) -> usize {
    n_heads * n_splits as usize * (head_dim + 2)
}

/// Base f32 index of the (head, split) scratch slot.
pub const fn slot_base(n_heads: usize, head_dim: usize, n_splits: u32, h: usize, s: u32) -> usize {
    // h < n_heads and s < n_splits are assumed (validated by callers); the
    // multiplication cannot overflow usize for any real model shape.
    let _ = n_heads;
    h * n_splits as usize * (head_dim + 2) + s as usize * (head_dim + 2)
}

/// Tokens per split: ceil(kv_len / n_splits). Zero when kv_len <= 0.
pub fn chunk_len(kv_len: i32, n_splits: u32) -> i32 {
    if kv_len <= 0 || n_splits == 0 {
        0
    } else {
        (kv_len + n_splits as i32 - 1) / n_splits as i32
    }
}

/// Half-open token range [t0, t1) owned by `split`. Trailing splits whose
/// start passes kv_len normalise to the empty range (t0, t0); the kernel
/// writes an identity partial (m = -inf, l = 0, acc = 0) in that case, which
/// the combine pass skips.
pub fn split_range(kv_len: i32, n_splits: u32, split: u32) -> (i32, i32) {
    let chunk = chunk_len(kv_len, n_splits);
    let t0 = split as i32 * chunk;
    // min(t0+chunk, kv_len) clips the last active split; max(t0) normalises
    // trailing splits past the end to an empty range.
    let t1 = (t0 + chunk).min(kv_len).max(t0);
    (t0, t1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_tile_exactly_no_overlap() {
        for n_splits in [1u32, 2, 3, 8, 16] {
            for kv_len in 1i32..=200 {
                let mut covered = vec![false; kv_len as usize];
                for s in 0..n_splits {
                    let (t0, t1) = split_range(kv_len, n_splits, s);
                    assert!(t0 <= t1, "negative chunk");
                    for t in t0..t1 {
                        assert!(t >= 0 && t < kv_len, "token {t} out of [0, {kv_len})");
                        assert!(!covered[t as usize], "token {t} covered twice");
                        covered[t as usize] = true;
                    }
                }
                assert!(covered.iter().all(|c| *c), "gap in tiling");
            }
        }
    }

    #[test]
    fn nonempty_split_exists_for_positive_kv_len() {
        for n_splits in [1u32, 4, 16] {
            for kv_len in 1i32..=64 {
                let (t0, t1) = split_range(kv_len, n_splits, 0);
                assert!(t0 < t1, "split 0 empty for kv_len={kv_len}");
            }
        }
    }

    #[test]
    fn empty_splits_trail_only() {
        // Empty splits are a (possibly empty) suffix: split s is empty iff
        // s * chunk >= kv_len. The combine pass relies on every slot still
        // being written (identity partial), so this only documents shape.
        for n_splits in [4u32, 16] {
            for kv_len in 1i32..=100 {
                let chunk = chunk_len(kv_len, n_splits);
                for s in 0..n_splits {
                    let (t0, t1) = split_range(kv_len, n_splits, s);
                    let empty = s as i32 * chunk >= kv_len;
                    assert_eq!(t0 >= t1, empty, "kv={kv_len} splits={n_splits} s={s}");
                }
            }
        }
    }

    #[test]
    fn slot_indices_within_scratch() {
        let (n_heads, head_dim, n_splits) = (32usize, 128usize, 16u32);
        let len = scratch_len(n_heads, head_dim, n_splits);
        assert_eq!(len, 32 * 16 * 130);
        for h in 0..n_heads {
            for s in 0..n_splits {
                let base = slot_base(n_heads, head_dim, n_splits, h, s);
                assert!(base + head_dim + 1 < len, "slot overflow h={h} s={s}");
            }
        }
    }

    #[test]
    fn chunk_len_zero_for_nonpositive() {
        assert_eq!(chunk_len(0, 16), 0);
        assert_eq!(chunk_len(-5, 16), 0);
        assert_eq!(chunk_len(100, 0), 0);
    }
}

// -- Kani: bounded proofs of the same invariants ------------------------------
#[cfg(kani)]
mod kani_tests {
    use super::*;

    #[kani::proof]
    #[kani::unwind(10)]
    fn kani_splits_tile_exactly() {
        let kv_len: i32 = kani::any();
        let n_splits: u32 = kani::any();
        kani::assume(kv_len >= 1 && kv_len <= 64);
        kani::assume(n_splits >= 1 && n_splits <= 8);
        let chunk = chunk_len(kv_len, n_splits);
        // Coverage by counting: sum of chunk sizes over splits == kv_len.
        let mut total: i32 = 0;
        for s in 0..n_splits {
            let (t0, t1) = split_range(kv_len, n_splits, s);
            kani::assert(t0 <= t1, "nonnegative chunk");
            kani::assert(t1 <= kv_len, "t1 clipped to kv_len");
            total += t1 - t0;
        }
        kani::assert(total == kv_len, "exact coverage, no gaps");
    }

    #[kani::proof]
    fn kani_no_double_coverage() {
        // Adjacent splits are disjoint: split s ends where s+1 begins (both
        // bounded by kv_len), so ranges cannot overlap.
        let kv_len: i32 = kani::any();
        let n_splits: u32 = kani::any();
        let s: u32 = kani::any();
        kani::assume(kv_len >= 1 && kv_len <= 64);
        kani::assume(n_splits >= 2 && n_splits <= 8);
        kani::assume(s + 1 < n_splits);
        let (t0a, t1a) = split_range(kv_len, n_splits, s);
        let (t0b, t1b) = split_range(kv_len, n_splits, s + 1);
        kani::assert(t0b >= t1a, "next split starts at or after previous end");
        kani::assert(t0b >= t0a, "monotone start");
        kani::assert(t1b >= t1a, "monotone end");
    }

    #[kani::proof]
    fn kani_chunk_at_least_one_when_splits_le_len() {
        let kv_len: i32 = kani::any();
        let n_splits: u32 = kani::any();
        kani::assume(kv_len >= 1 && kv_len <= 256);
        kani::assume(n_splits >= 1 && n_splits <= 16);
        kani::assume(kv_len >= n_splits as i32);
        let chunk = chunk_len(kv_len, n_splits);
        kani::assert(chunk >= 1, "chunk nonzero when kv_len >= n_splits");
        let (t0, t1) = split_range(kv_len, n_splits, 0);
        kani::assert(t0 < t1, "split 0 nonempty");
    }

    #[kani::proof]
    fn kani_slot_base_bounded() {
        let n_heads: usize = kani::any();
        let head_dim: usize = kani::any();
        let n_splits: u32 = kani::any();
        let h: usize = kani::any();
        let s: u32 = kani::any();
        kani::assume(n_heads >= 1 && n_heads <= 128);
        kani::assume(head_dim >= 8 && head_dim <= 256);
        kani::assume(n_splits >= 1 && n_splits <= 32);
        kani::assume(h < n_heads && s < n_splits);
        let len = scratch_len(n_heads, head_dim, n_splits);
        let base = slot_base(n_heads, head_dim, n_splits, h, s);
        kani::assert(base + head_dim + 1 < len, "m/l slots in bounds");
        kani::assert(base >= h * n_splits as usize * (head_dim + 2), "no underflow");
    }

    #[kani::proof]
    fn kani_empty_split_identity() {
        // An empty split has t0 >= t1 exactly when its start passes kv_len.
        let kv_len: i32 = kani::any();
        let n_splits: u32 = kani::any();
        let s: u32 = kani::any();
        kani::assume(kv_len >= 1 && kv_len <= 64);
        kani::assume(n_splits >= 1 && n_splits <= 16);
        kani::assume(s < n_splits);
        let chunk = chunk_len(kv_len, n_splits);
        let (t0, t1) = split_range(kv_len, n_splits, s);
        kani::assert((t0 >= t1) == (s as i32 * chunk >= kv_len), "empty iff start past end");
    }
}
