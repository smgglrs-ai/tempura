/*
 * oxide-gemv-q4k: Hand-optimized Q4_K decode GEMV kernel via rustc-codegen-cuda.
 *
 * Build:  cargo oxide build --manifest-path tools/oxide-gemv-q4k/Cargo.toml
 * The compiled binary embeds PTX in the `.oxart` ELF section.
 * The tempura-kernels build.rs extracts that PTX and writes it to OUT_DIR.
 *
 * Q4_K packed GPU format (42 u32 per block, 256 elements per block):
 *   u32[0]      = f32 bits of d (super-scale, f16 -> f32 at upload)
 *   u32[1]      = f32 bits of dmin (super-min, f16 -> f32 at upload)
 *   u32[2+j]    = sc[2*j] | (sc[2*j+1]<<8) for j=0..3 (group scale pairs, 8-bit each)
 *   u32[6+j]    = m[2*j]  | (m[2*j+1]<<8)  for j=0..3 (group min pairs, 8-bit each)
 *   u32[10..42] = 32 u32s of packed qs bytes (128 bytes = 256 nibbles, 4-bit each)
 *
 * Dequant for element `within` in a block:
 *   j       = within / 64                    -- which j-pair (0..3)
 *   in_j    = within % 64
 *   is_hi   = in_j >= 32
 *   l       = in_j % 32
 *   sc_group = 2*j + is_hi  (0..7)
 *   sc = (packed[base + 2 + j] >> ((sc_group%2)*8)) & 0xFF  as f32
 *   m  = (packed[base + 6 + j] >> ((sc_group%2)*8)) & 0xFF  as f32
 *   nibble = lo nibble if !is_hi, hi nibble if is_hi  (from qs[j*32+l])
 *   weight = d * sc * nibble - dmin * m
 *
 * v2 optimization (shfl-based scale sharing):
 *   One warp (32 lanes) processes 256 elements of one block per iteration.
 *   Each lane handles elements: l, l+32, l+64, ..., l+224 within the block.
 *   - d, dmin: lane 0 loads once; all lanes receive via shfl.
 *   - sc/m:    lanes 0..3 each load one u32 (2 scale/min values); broadcast via shfl.
 *   - qs:      each lane loads one u32 per j-group (contains lo+hi nibbles for that lane).
 *   The 4 j-groups are fully unrolled, each processing 2 elements per lane.
 */

use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, ptx_asm, thread, warp};
// NOTE: dp4a_s32 requires the upstream oxide backend (cuda-oxide/crates/rustc-codegen-cuda)
// which includes dp4a in its intrinsic catalog. The fork backend (crates/rustc-codegen-cuda)
// does not. Build q4k_gemv_decode_q8 only when the upstream backend is available.
use cuda_device::dotprod::dp4a_s32;
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_core::simt::LaunchConfig;

const WARPS_PER_BLOCK: u32 = 4;
const WARPS_PER_BLOCK_KPAR: u32 = 8;

/// f16 bits (low 16 of u32) -> f32, device-safe bit arithmetic.
#[inline(always)]
fn f16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15) << 31;
    let exp  = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;
    let bits = if exp == 0 {
        sign | (mant << 13)
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> f16 bits stored in low 16 of u16, device-safe.
#[inline(always)]
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 31) << 15;
    let exp  = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0xFF {
        return (sign | 0x7C00 | (mant >> 13)) as u16;
    }
    let exp16 = exp - 127 + 15;
    if exp16 <= 0 { return sign as u16; }
    if exp16 >= 31 { return (sign | 0x7C00) as u16; }
    (sign | ((exp16 as u32) << 10) | (mant >> 13)) as u16
}

/// Butterfly warp-reduce max over all 32 lanes (abs, for quantization scale).
#[inline(always)]
fn warp_reduce_max_f32(mut v: f32) -> f32 {
    let t = warp::shuffle_xor_f32(v, 16); if t > v { v = t; }
    let t = warp::shuffle_xor_f32(v, 8);  if t > v { v = t; }
    let t = warp::shuffle_xor_f32(v, 4);  if t > v { v = t; }
    let t = warp::shuffle_xor_f32(v, 2);  if t > v { v = t; }
    let t = warp::shuffle_xor_f32(v, 1);  if t > v { v = t; }
    v
}

/// Inline 6-bit scale unpack for raw GGUF Q4K format.
/// ps0 = packed u32 of bytes ps[0..4], ps1 = ps[4..8], ps2 = ps[8..12].
/// mw = j-group index (0..3).
/// Returns (sc_lo, sc_hi, m_lo, m_hi) as f32 for that j-group.
#[inline(always)]
fn unpack_scales_6bit(ps0: u32, ps1: u32, ps2: u32, mw: u32) -> (f32, f32, f32, f32) {
    // ps[i]: byte i of the 12-byte scale section
    // j=0: sc[0],sc[1],m[0],m[1] -> ps[0],ps[1],ps[4],ps[5]
    // j=1: sc[2],sc[3],m[2],m[3] -> ps[2],ps[3],ps[6],ps[7]
    // j=2: sc[4],sc[5],m[4],m[5] -> 6-bit unpack from ps[0..2] and ps[8..10]
    // j=3: sc[6],sc[7],m[6],m[7] -> 6-bit unpack from ps[2..4] and ps[10..12]
    let (sc_lo, sc_hi, m_lo, m_hi) = if mw < 2 {
        let shift = mw * 16;  // j=0: bytes 0,1,4,5 (shift=0); j=1: bytes 2,3,6,7 (shift=16)
        let a = (ps0 >> shift) & 0x3F;
        let b = (ps0 >> (shift + 8)) & 0x3F;
        let c = (ps1 >> shift) & 0x3F;
        let d = (ps1 >> (shift + 8)) & 0x3F;
        (a as f32, b as f32, c as f32, d as f32)
    } else {
        // j=2 (mw=2) or j=3 (mw=3)
        let k = mw - 2;  // 0 or 1
        let shift_lo = k * 16;      // j=2: byte 0,1,4,5 of ps0/ps1; j=3: byte 2,3,6,7
        let shift_hi = k * 16;      // same shift for ps2
        // ps0 bytes 0,1 (k=0) or 2,3 (k=1) hold lower 6 bits of sc[2k], sc[2k+1]
        let ps0_lo = (ps0 >> shift_lo) & 0xFF;       // ps[2k]
        let ps0_hi = (ps0 >> (shift_lo + 8)) & 0xFF; // ps[2k+1]
        let ps1_lo = (ps1 >> shift_lo) & 0xFF;       // ps[4+2k]
        let ps1_hi = (ps1 >> (shift_lo + 8)) & 0xFF; // ps[5+2k]
        let ps2_lo = (ps2 >> shift_hi) & 0xFF;       // ps[8+2k]
        let ps2_hi = (ps2 >> (shift_hi + 8)) & 0xFF; // ps[9+2k]
        // sc[4+2k] = ((ps[8+2k] & 0x0F) | ((ps[2k] >> 6) << 4)) & 0x3F
        let a = ((ps2_lo & 0x0F) | ((ps0_lo >> 6) << 4)) & 0x3F;
        let b = ((ps2_hi & 0x0F) | ((ps0_hi >> 6) << 4)) & 0x3F;
        // m[4+2k] = ((ps[8+2k] >> 4) | ((ps[4+2k] >> 6) << 4)) & 0x3F
        let c = ((ps2_lo >> 4) | ((ps1_lo >> 6) << 4)) & 0x3F;
        let d = ((ps2_hi >> 4) | ((ps1_hi >> 6) << 4)) & 0x3F;
        (a as f32, b as f32, c as f32, d as f32)
    };
    (sc_lo, sc_hi, m_lo, m_hi)
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Q4_K GEMV for M=1 decode: out[N] = dequant(packed[N, K_BLOCKS, 42]) x x[K].
    ///
    /// Grid:  (ceil(N / WARPS_PER_BLOCK),) blocks -- each warp owns one output neuron.
    /// Block: (WARPS_PER_BLOCK * 32,) threads.
    ///
    /// Fallback kernel: per-element stride-32 loop, no shfl.
    #[kernel]
    pub fn q4k_gemv_decode(
        mut out: DisjointSlice<f32>,
        x: &[f32],
        packed: &[u32],
        n: u32,
        k_blocks: u32,
    ) {
        let tid     = thread::threadIdx_x();
        let bid     = thread::blockIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;
        let n_out   = bid * WARPS_PER_BLOCK + warp_id;

        if n_out >= n {
            return;
        }

        let mut acc = 0.0_f32;
        let k = k_blocks * 256;

        let mut elem = lane;
        while elem < k {
            let blk    = elem / 256;
            let within = elem % 256;

            let base = (n_out * k_blocks + blk) * 42;

            let d    = f32::from_bits(packed[base as usize]);
            let dmin = f32::from_bits(packed[(base + 1) as usize]);

            let j     = within / 64;
            let in_j  = within % 64;
            let is_hi = in_j >= 32;
            let l     = in_j % 32;

            let qs_byte_idx = j * 32 + l;
            let qs_word_idx = qs_byte_idx / 4;
            let qs_byte_pos = qs_byte_idx % 4;
            let qs_word = packed[(base + 10 + qs_word_idx) as usize];
            let qs_byte = (qs_word >> (qs_byte_pos * 8)) & 0xFF;
            let nibble = if is_hi { qs_byte >> 4 } else { qs_byte & 0xF };

            let sc_group = 2 * j + is_hi as u32;
            let sc_pair_word = packed[(base + 2 + sc_group / 2) as usize];
            let m_pair_word  = packed[(base + 6 + sc_group / 2) as usize];
            let sc_shift = (sc_group % 2) * 8;
            let sc_u32 = (sc_pair_word >> sc_shift) & 0xFF;
            let m_u32  = (m_pair_word  >> sc_shift) & 0xFF;

            let w = d * (sc_u32 as f32) * (nibble as f32) - dmin * (m_u32 as f32);

            acc += w * x[elem as usize];

            elem += 32;
        }

        acc = warp::reduce_sum_f32(acc);
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(n_out as usize) = acc;
            }
        }
    }

    /// Q4_K GEMV v2: shfl-based scale/min sharing + multi-warp K-parallelism.
    ///
    /// Optimization over v1:
    ///   - d, dmin: loaded once by lane 0, broadcast via shfl.
    ///   - sc, m:   lanes 0..3 load 4 u32s; broadcast via shfl per j-group.
    ///   - qs:      one u32 load per lane per j-group.
    ///   - 4 j-groups fully unrolled; 2 accumulation steps per group per lane.
    ///   - Multi-warp K-parallelism: all WARPS_PER_BLOCK warps cooperate on one
    ///     output neuron, each covering a stride-WARPS_PER_BLOCK subset of K blocks.
    ///     After intra-warp butterfly reduce, cross-warp reduce via shared memory.
    ///
    /// Grid:  N blocks (one block per output neuron).
    /// Block: WARPS_PER_BLOCK * 32 threads.
    /// Shared memory: WARPS_PER_BLOCK * 4 bytes (f32 partial sums).
    #[kernel]
    pub fn q4k_gemv_dp4a_base(
        mut out: DisjointSlice<f32>,
        x: &[f32],
        packed: &[u32],
        n: u32,
        k_blocks: u32,
    ) {
        let tid     = thread::threadIdx_x();
        let bid     = thread::blockIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;
        let n_out   = bid;  // one output neuron per block

        if n_out >= n {
            return;
        }

        let mut acc = 0.0_f32;
        let qs_shift = (lane % 4) * 8;

        // Each warp covers a stride-WARPS_PER_BLOCK subset of K blocks.
        // Load restructuring: all 8 weight loads (d, dmin, sc, m, 4xqs) are
        // independent of each other -- hoist them ALL before any shuffle so the
        // GPU can issue them in parallel (one DRAM round-trip instead of three).
        // Software prefetch: pre-load next block's data during current block's
        // compute phase to hide DRAM latency across iterations.

        // Prime the prefetch registers with block warp_id data.
        // Guard: warps with warp_id >= k_blocks have no valid first block --
        // use a safe fallback address (block 0 of the same neuron) so Blackwell
        // doesn't trap on an out-of-bounds global load in a predicated instruction.
        let mut blk = warp_id;
        let init_valid = blk < k_blocks;
        let mut pre_base = if init_valid { (n_out * k_blocks + blk) * 42 } else { n_out * k_blocks * 42 };
        let mut pre_d_raw    = if init_valid && lane == 0 { packed[pre_base as usize] } else { 0 };
        let mut pre_dmin_raw = if init_valid && lane == 0 { packed[(pre_base + 1) as usize] } else { 0 };
        let mut pre_sc_raw   = if init_valid && lane < 4  { packed[(pre_base + 2 + lane) as usize] } else { 0 };
        let mut pre_m_raw    = if init_valid && lane < 4  { packed[(pre_base + 6 + lane) as usize] } else { 0 };
        let mut pre_qs_j0 = if init_valid { packed[(pre_base + 10 + lane / 4) as usize] } else { 0 };
        let mut pre_qs_j1 = if init_valid { packed[(pre_base + 10 + 8 + lane / 4) as usize] } else { 0 };
        let mut pre_qs_j2 = if init_valid { packed[(pre_base + 10 + 16 + lane / 4) as usize] } else { 0 };
        let mut pre_qs_j3 = if init_valid { packed[(pre_base + 10 + 24 + lane / 4) as usize] } else { 0 };

        while blk < k_blocks {
            // Move prefetched registers into current-block locals.
            let d_raw    = pre_d_raw;
            let dmin_raw = pre_dmin_raw;
            let sc_raw   = pre_sc_raw;
            let m_raw    = pre_m_raw;
            let qs_j0 = pre_qs_j0;
            let qs_j1 = pre_qs_j1;
            let qs_j2 = pre_qs_j2;
            let qs_j3 = pre_qs_j3;

            // Issue next-block loads NOW (during this block's compute) to hide latency.
            let next_blk = blk + WARPS_PER_BLOCK;
            if next_blk < k_blocks {
                pre_base  = (n_out * k_blocks + next_blk) * 42;
                pre_d_raw    = if lane == 0 { packed[pre_base as usize] } else { 0 };
                pre_dmin_raw = if lane == 0 { packed[(pre_base + 1) as usize] } else { 0 };
                pre_sc_raw   = if lane < 4  { packed[(pre_base + 2 + lane) as usize] } else { 0 };
                pre_m_raw    = if lane < 4  { packed[(pre_base + 6 + lane) as usize] } else { 0 };
                pre_qs_j0 = packed[(pre_base + 10 + lane / 4) as usize];
                pre_qs_j1 = packed[(pre_base + 10 + 8 + lane / 4) as usize];
                pre_qs_j2 = packed[(pre_base + 10 + 16 + lane / 4) as usize];
                pre_qs_j3 = packed[(pre_base + 10 + 24 + lane / 4) as usize];
            }

            // Compute: shuffle -> FMA using current-block data.
            // All loads above are already in-flight or completed.
            let d    = f32::from_bits(warp::shuffle(d_raw, 0));
            let dmin = f32::from_bits(warp::shuffle(dmin_raw, 0));

            // j=0
            {
                let sc_pair = warp::shuffle(sc_raw, 0);
                let m_pair  = warp::shuffle(m_raw, 0);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j0 >> qs_shift) & 0xFF;
                let nibble_lo = (qs_byte & 0xF) as f32;
                let nibble_hi = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nibble_lo - dmin * m_lo) * x[(blk * 256 + lane) as usize];
                acc += (d * sc_hi * nibble_hi - dmin * m_hi) * x[(blk * 256 + 32 + lane) as usize];
            }
            // j=1
            {
                let sc_pair = warp::shuffle(sc_raw, 1);
                let m_pair  = warp::shuffle(m_raw, 1);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j1 >> qs_shift) & 0xFF;
                let nibble_lo = (qs_byte & 0xF) as f32;
                let nibble_hi = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nibble_lo - dmin * m_lo) * x[(blk * 256 + 64 + lane) as usize];
                acc += (d * sc_hi * nibble_hi - dmin * m_hi) * x[(blk * 256 + 96 + lane) as usize];
            }
            // j=2
            {
                let sc_pair = warp::shuffle(sc_raw, 2);
                let m_pair  = warp::shuffle(m_raw, 2);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j2 >> qs_shift) & 0xFF;
                let nibble_lo = (qs_byte & 0xF) as f32;
                let nibble_hi = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nibble_lo - dmin * m_lo) * x[(blk * 256 + 128 + lane) as usize];
                acc += (d * sc_hi * nibble_hi - dmin * m_hi) * x[(blk * 256 + 160 + lane) as usize];
            }
            // j=3
            {
                let sc_pair = warp::shuffle(sc_raw, 3);
                let m_pair  = warp::shuffle(m_raw, 3);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j3 >> qs_shift) & 0xFF;
                let nibble_lo = (qs_byte & 0xF) as f32;
                let nibble_hi = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nibble_lo - dmin * m_lo) * x[(blk * 256 + 192 + lane) as usize];
                acc += (d * sc_hi * nibble_hi - dmin * m_hi) * x[(blk * 256 + 224 + lane) as usize];
            }

            blk += WARPS_PER_BLOCK;
        }

        // Intra-warp butterfly reduce; lane 0 of each warp holds the partial sum.
        acc = warp::reduce_sum_f32(acc);

        // Cross-warp reduce: lane 0 of each warp writes its partial sum to smem,
        // then warp 0 reads all partials and produces the final result.
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        if lane == 0 {
            unsafe { *smem.add(warp_id as usize) = acc; }
        }
        thread::sync_threads();

        if warp_id == 0 {
            acc = if lane < WARPS_PER_BLOCK {
                unsafe { *smem.add(lane as usize) }
            } else {
                0.0_f32
            };
            // N-step XOR butterfly for any power-of-2 WARPS_PER_BLOCK.
            let mut delta = WARPS_PER_BLOCK / 2;
            while delta >= 1 {
                acc += warp::shuffle_xor_f32(acc, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe { *out.get_unchecked_mut(n_out as usize) = acc; }
            }
        }
    }

    /// Batched Q4_K GEMV: out[M, N] = dequant(packed[N, K_BLOCKS, 42]) x x[M, K].
    ///
    /// Grid:  (N, M, 1) -- blockIdx.x = output neuron, blockIdx.y = input row.
    /// Block: WARPS_PER_BLOCK * 32 threads.
    /// Shared memory: WARPS_PER_BLOCK * 4 bytes.
    ///
    /// Key benefit: all M blocks for the same blockIdx.x access the same weight
    /// data. When M blocks run concurrently (RTX 5090 has 170 SMs), the L2 cache
    /// serves the weight data Mx -- effective bandwidth amortization.
    #[kernel]
    pub fn q4k_gemv_dp4a_base_batch(
        mut out: DisjointSlice<f32>,  // [M * N] row-major
        x: *const f32,                // [M * K] row-major
        packed: &[u32],               // [N * K_BLOCKS * 42]
        m: u32,
        n: u32,
        k_blocks: u32,
    ) {
        let tid     = thread::threadIdx_x();
        let n_out   = thread::blockIdx_x();
        let row_idx = thread::blockIdx_y();
        let warp_id = tid / 32;
        let lane    = tid % 32;

        if n_out >= n || row_idx >= m { return; }

        // Pointer to this row's K activations.
        let x_row = unsafe { x.add((row_idx * k_blocks * 256) as usize) };
        let qs_shift = (lane % 4) * 8;

        let mut acc = 0.0_f32;

        // Prime prefetch registers.
        let mut blk = warp_id;
        let mut pre_base = (n_out * k_blocks + blk) * 42;
        let mut pre_d_raw    = if lane == 0 { packed[pre_base as usize] } else { 0 };
        let mut pre_dmin_raw = if lane == 0 { packed[(pre_base + 1) as usize] } else { 0 };
        let mut pre_sc_raw   = if lane < 4  { packed[(pre_base + 2 + lane) as usize] } else { 0 };
        let mut pre_m_raw    = if lane < 4  { packed[(pre_base + 6 + lane) as usize] } else { 0 };
        let mut pre_qs_j0 = packed[(pre_base + 10 + lane / 4) as usize];
        let mut pre_qs_j1 = packed[(pre_base + 10 + 8 + lane / 4) as usize];
        let mut pre_qs_j2 = packed[(pre_base + 10 + 16 + lane / 4) as usize];
        let mut pre_qs_j3 = packed[(pre_base + 10 + 24 + lane / 4) as usize];

        while blk < k_blocks {
            let d_raw    = pre_d_raw;
            let dmin_raw = pre_dmin_raw;
            let sc_raw   = pre_sc_raw;
            let m_raw    = pre_m_raw;
            let qs_j0 = pre_qs_j0;
            let qs_j1 = pre_qs_j1;
            let qs_j2 = pre_qs_j2;
            let qs_j3 = pre_qs_j3;

            let next_blk = blk + WARPS_PER_BLOCK;
            if next_blk < k_blocks {
                pre_base     = (n_out * k_blocks + next_blk) * 42;
                pre_d_raw    = if lane == 0 { packed[pre_base as usize] } else { 0 };
                pre_dmin_raw = if lane == 0 { packed[(pre_base + 1) as usize] } else { 0 };
                pre_sc_raw   = if lane < 4  { packed[(pre_base + 2 + lane) as usize] } else { 0 };
                pre_m_raw    = if lane < 4  { packed[(pre_base + 6 + lane) as usize] } else { 0 };
                pre_qs_j0 = packed[(pre_base + 10 + lane / 4) as usize];
                pre_qs_j1 = packed[(pre_base + 10 + 8 + lane / 4) as usize];
                pre_qs_j2 = packed[(pre_base + 10 + 16 + lane / 4) as usize];
                pre_qs_j3 = packed[(pre_base + 10 + 24 + lane / 4) as usize];
            }

            let d    = f32::from_bits(warp::shuffle(d_raw, 0));
            let dmin = f32::from_bits(warp::shuffle(dmin_raw, 0));

            // j=0
            {
                let sc_pair = warp::shuffle(sc_raw, 0);
                let m_pair  = warp::shuffle(m_raw, 0);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j0 >> qs_shift) & 0xFF;
                let nl = (qs_byte & 0xF) as f32;
                let nh = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nl - dmin * m_lo) * unsafe { *x_row.add((blk * 256 + lane) as usize) };
                acc += (d * sc_hi * nh - dmin * m_hi) * unsafe { *x_row.add((blk * 256 + 32 + lane) as usize) };
            }
            // j=1
            {
                let sc_pair = warp::shuffle(sc_raw, 1);
                let m_pair  = warp::shuffle(m_raw, 1);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j1 >> qs_shift) & 0xFF;
                let nl = (qs_byte & 0xF) as f32;
                let nh = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nl - dmin * m_lo) * unsafe { *x_row.add((blk * 256 + 64 + lane) as usize) };
                acc += (d * sc_hi * nh - dmin * m_hi) * unsafe { *x_row.add((blk * 256 + 96 + lane) as usize) };
            }
            // j=2
            {
                let sc_pair = warp::shuffle(sc_raw, 2);
                let m_pair  = warp::shuffle(m_raw, 2);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j2 >> qs_shift) & 0xFF;
                let nl = (qs_byte & 0xF) as f32;
                let nh = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nl - dmin * m_lo) * unsafe { *x_row.add((blk * 256 + 128 + lane) as usize) };
                acc += (d * sc_hi * nh - dmin * m_hi) * unsafe { *x_row.add((blk * 256 + 160 + lane) as usize) };
            }
            // j=3
            {
                let sc_pair = warp::shuffle(sc_raw, 3);
                let m_pair  = warp::shuffle(m_raw, 3);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j3 >> qs_shift) & 0xFF;
                let nl = (qs_byte & 0xF) as f32;
                let nh = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nl - dmin * m_lo) * unsafe { *x_row.add((blk * 256 + 192 + lane) as usize) };
                acc += (d * sc_hi * nh - dmin * m_hi) * unsafe { *x_row.add((blk * 256 + 224 + lane) as usize) };
            }

            blk += WARPS_PER_BLOCK;
        }

        acc = warp::reduce_sum_f32(acc);
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        if lane == 0 {
            unsafe { *smem.add(warp_id as usize) = acc; }
        }
        thread::sync_threads();
        if warp_id == 0 {
            acc = if lane < WARPS_PER_BLOCK { unsafe { *smem.add(lane as usize) } } else { 0.0_f32 };
            let mut delta = WARPS_PER_BLOCK / 2;
            while delta >= 1 {
                acc += warp::shuffle_xor_f32(acc, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe { *out.get_unchecked_mut((row_idx * n + n_out) as usize) = acc; }
            }
        }
    }

    /// Colocated batched Q4_K GEMV: all M rows in ONE block -> same SM -> L2 broadcast.
    ///
    /// Block: (WARPS_PER_BLOCK * 32 * M, 1, 1) -- flat encoding of (row, warp, lane).
    ///   overall_warp = tid / 32
    ///   row     = overall_warp / WARPS_PER_BLOCK   (0..M)
    ///   warp_id = overall_warp % WARPS_PER_BLOCK   (0..WARPS_PER_BLOCK)
    ///   n_out   = blockIdx.x * WARPS_PER_BLOCK + warp_id
    ///
    /// Grid: (ceil(N / WARPS_PER_BLOCK), 1, 1)
    ///
    /// Key: M warps with the same warp_id but different row values all run on the SAME SM.
    /// They all access packed[n_out, blk, ...] simultaneously -> GPU L2 cache broadcasts
    /// the weight data to all M warps with a single DRAM fetch -> Mx bandwidth amortization.
    ///
    /// smem: [M * WARPS_PER_BLOCK] f32 for cross-warp partial sum reduction.
    #[kernel]
    pub unsafe fn q4k_gemm_colocated(
        mut out: DisjointSlice<f32>,   // [m * n] row-major
        x: *const f32,                  // [m * k] row-major, k = k_blocks * 256
        packed: &[u32],                 // [n * k_blocks * 42]
        m: u32,
        n: u32,
        k_blocks: u32,
    ) {
        let tid          = thread::threadIdx_x();
        let overall_warp = tid / 32;
        let lane         = tid % 32;
        let row          = overall_warp / WARPS_PER_BLOCK;   // which activation row
        let warp_id      = overall_warp % WARPS_PER_BLOCK;   // which K-stripe
        let n_out        = thread::blockIdx_x() * WARPS_PER_BLOCK + warp_id;

        if n_out >= n || row >= m { return; }

        // Pointer to activation row `row`.
        let x_row: *const f32 = x.add((row * k_blocks * 256) as usize);
        let qs_shift = (lane % 4) * 8;

        let mut acc = 0.0_f32;

        // K-loop: warp_id strides WARPS_PER_BLOCK (same K-parallelism as v2).
        // All M rows process the same base pointer for weight loads -> L2 broadcast.
        let mut blk = warp_id;
        let mut pre_base     = (n_out * k_blocks + blk) * 42;
        let mut pre_d_raw    = if lane == 0 { packed[pre_base as usize] } else { 0 };
        let mut pre_dmin_raw = if lane == 0 { packed[(pre_base + 1) as usize] } else { 0 };
        let mut pre_sc_raw   = if lane < 4  { packed[(pre_base + 2 + lane) as usize] } else { 0 };
        let mut pre_m_raw    = if lane < 4  { packed[(pre_base + 6 + lane) as usize] } else { 0 };
        let mut pre_qs_j0 = packed[(pre_base + 10          + lane / 4) as usize];
        let mut pre_qs_j1 = packed[(pre_base + 10 + 8      + lane / 4) as usize];
        let mut pre_qs_j2 = packed[(pre_base + 10 + 16     + lane / 4) as usize];
        let mut pre_qs_j3 = packed[(pre_base + 10 + 24     + lane / 4) as usize];

        while blk < k_blocks {
            let d_raw    = pre_d_raw;
            let dmin_raw = pre_dmin_raw;
            let sc_raw   = pre_sc_raw;
            let m_raw    = pre_m_raw;
            let qs_j0 = pre_qs_j0;
            let qs_j1 = pre_qs_j1;
            let qs_j2 = pre_qs_j2;
            let qs_j3 = pre_qs_j3;

            let next_blk = blk + WARPS_PER_BLOCK;
            if next_blk < k_blocks {
                pre_base      = (n_out * k_blocks + next_blk) * 42;
                pre_d_raw     = if lane == 0 { packed[pre_base as usize] } else { 0 };
                pre_dmin_raw  = if lane == 0 { packed[(pre_base + 1) as usize] } else { 0 };
                pre_sc_raw    = if lane < 4  { packed[(pre_base + 2 + lane) as usize] } else { 0 };
                pre_m_raw     = if lane < 4  { packed[(pre_base + 6 + lane) as usize] } else { 0 };
                pre_qs_j0 = packed[(pre_base + 10          + lane / 4) as usize];
                pre_qs_j1 = packed[(pre_base + 10 + 8      + lane / 4) as usize];
                pre_qs_j2 = packed[(pre_base + 10 + 16     + lane / 4) as usize];
                pre_qs_j3 = packed[(pre_base + 10 + 24     + lane / 4) as usize];
            }

            let d    = f32::from_bits(warp::shuffle(d_raw, 0));
            let dmin = f32::from_bits(warp::shuffle(dmin_raw, 0));

            // j=0
            {
                let sc_pair = warp::shuffle(sc_raw, 0);
                let m_pair  = warp::shuffle(m_raw, 0);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j0 >> qs_shift) & 0xFF;
                let nl = (qs_byte & 0xF) as f32;
                let nh = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nl - dmin * m_lo) * *x_row.add((blk * 256 + lane) as usize);
                acc += (d * sc_hi * nh - dmin * m_hi) * *x_row.add((blk * 256 + 32 + lane) as usize);
            }
            // j=1
            {
                let sc_pair = warp::shuffle(sc_raw, 1);
                let m_pair  = warp::shuffle(m_raw, 1);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j1 >> qs_shift) & 0xFF;
                let nl = (qs_byte & 0xF) as f32;
                let nh = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nl - dmin * m_lo) * *x_row.add((blk * 256 + 64 + lane) as usize);
                acc += (d * sc_hi * nh - dmin * m_hi) * *x_row.add((blk * 256 + 96 + lane) as usize);
            }
            // j=2
            {
                let sc_pair = warp::shuffle(sc_raw, 2);
                let m_pair  = warp::shuffle(m_raw, 2);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j2 >> qs_shift) & 0xFF;
                let nl = (qs_byte & 0xF) as f32;
                let nh = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nl - dmin * m_lo) * *x_row.add((blk * 256 + 128 + lane) as usize);
                acc += (d * sc_hi * nh - dmin * m_hi) * *x_row.add((blk * 256 + 160 + lane) as usize);
            }
            // j=3
            {
                let sc_pair = warp::shuffle(sc_raw, 3);
                let m_pair  = warp::shuffle(m_raw, 3);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j3 >> qs_shift) & 0xFF;
                let nl = (qs_byte & 0xF) as f32;
                let nh = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nl - dmin * m_lo) * *x_row.add((blk * 256 + 192 + lane) as usize);
                acc += (d * sc_hi * nh - dmin * m_hi) * *x_row.add((blk * 256 + 224 + lane) as usize);
            }

            blk += WARPS_PER_BLOCK;
        }

        // Intra-warp reduce.
        acc = warp::reduce_sum_f32(acc);

        // Cross-warp reduce: smem[row * WARPS_PER_BLOCK + warp_id] = partial sum.
        // Each (row, warp_id) has its own smem slot.
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        let smem_idx = (row * WARPS_PER_BLOCK + warp_id) as usize;
        if lane == 0 {
            *smem.add(smem_idx) = acc;
        }
        thread::sync_threads();

        // For each row: warp_id==0 reduces WARPS_PER_BLOCK partials and writes output.
        if warp_id == 0 {
            // Read this row's WARPS_PER_BLOCK partials (lanes 0..WARPS_PER_BLOCK hold them).
            acc = if lane < WARPS_PER_BLOCK {
                *smem.add((row * WARPS_PER_BLOCK + lane) as usize)
            } else {
                0.0_f32
            };
            let mut delta = WARPS_PER_BLOCK / 2;
            while delta >= 1 {
                acc += warp::shuffle_xor_f32(acc, delta);
                delta /= 2;
            }
            if lane == 0 {
                *out.get_unchecked_mut((row * n + n_out) as usize) = acc;
            }
        }
    }

    /// Smem-tiled batched Q4_K GEMM: true Mx bandwidth amortization.
    ///
    /// Same structure as q4k_gemm_colocated but row-0 warps explicitly load
    /// weight K-block data into shared memory before compute. All M rows then
    /// read weights from smem (no DRAM) during FMA, eliminating (M-1)/M of
    /// weight DRAM traffic.
    ///
    /// smem layout:
    ///   [0 .. WARPS_PER_BLOCK*42)         = weight u32 data (42 u32 per warp)
    ///   [WARPS_PER_BLOCK*42 .. +M*WPB)    = partial-sum reduction slots (f32)
    ///
    /// Block: WARPS_PER_BLOCK * 32 * M threads (flat 1D encoding).
    ///   overall_warp = tid / 32
    ///   row     = overall_warp / WARPS_PER_BLOCK
    ///   warp_id = overall_warp % WARPS_PER_BLOCK
    ///   n_out   = blockIdx.x * WARPS_PER_BLOCK + warp_id
    ///
    /// Grid: (ceil(N / WARPS_PER_BLOCK), 1, 1)
    /// shared_mem_bytes = (WARPS_PER_BLOCK * 42 + M * WARPS_PER_BLOCK) * 4
    #[kernel]
    pub unsafe fn q4k_gemm_smem(
        mut out: DisjointSlice<f32>,   // [m * n] row-major
        x: *const f32,                  // [m * k] row-major, k = k_blocks * 256
        packed: &[u32],                 // [n * k_blocks * 42]
        m: u32,
        n: u32,
        k_blocks: u32,
    ) {
        let tid          = thread::threadIdx_x();
        let overall_warp = tid / 32;
        let lane         = tid % 32;
        let row          = overall_warp / WARPS_PER_BLOCK;
        let warp_id      = overall_warp % WARPS_PER_BLOCK;
        let n_out        = thread::blockIdx_x() * WARPS_PER_BLOCK + warp_id;

        if n_out >= n || row >= m { return; }

        let x_row: *const f32 = x.add((row * k_blocks * 256) as usize);
        let qs_shift = (lane % 4) * 8;

        // smem: first section holds weight u32 data (42 u32s per warp),
        //       second section holds partial-sum reduction slots.
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        let smem_w: *mut u32 = smem as *mut u32;   // weight staging
        let smem_p_base = (WARPS_PER_BLOCK * 42) as usize;  // index of partials in f32 slots

        let mut acc = 0.0_f32;

        // K-loop: warp_id strides WARPS_PER_BLOCK (same K-parallelism as v2).
        let mut blk = warp_id;
        while blk < k_blocks {
            let base = (n_out * k_blocks + blk) * 42;
            let smem_blk = (warp_id * 42) as usize;

            // -- Phase 1: row=0 warps load 42 u32s for this K-block into smem --
            // 42 u32s loaded by 32 lanes in two passes (32 + 10 = 42).
            if row == 0 {
                *smem_w.add(smem_blk + lane as usize) = packed[(base + lane) as usize];
                if lane < 10 {
                    *smem_w.add(smem_blk + 32 + lane as usize) = packed[(base + 32 + lane) as usize];
                }
            }

            thread::sync_threads();

            // -- Phase 2: all rows compute using smem weight data --------------
            let d_src    = if lane == 0 { *smem_w.add(smem_blk) } else { 0 };
            let dmin_src = if lane == 0 { *smem_w.add(smem_blk + 1) } else { 0 };
            let d    = f32::from_bits(warp::shuffle(d_src, 0));
            let dmin = f32::from_bits(warp::shuffle(dmin_src, 0));

            let sc_raw = if lane < 4 { *smem_w.add(smem_blk + 2 + lane as usize) } else { 0 };
            let m_raw  = if lane < 4 { *smem_w.add(smem_blk + 6 + lane as usize) } else { 0 };

            // qs words: lanes 0..31 each read their own qs word from smem
            let qs_j0 = *smem_w.add(smem_blk + 10 + (lane / 4) as usize);
            let qs_j1 = *smem_w.add(smem_blk + 10 + 8 + (lane / 4) as usize);
            let qs_j2 = *smem_w.add(smem_blk + 10 + 16 + (lane / 4) as usize);
            let qs_j3 = *smem_w.add(smem_blk + 10 + 24 + (lane / 4) as usize);

            // j=0
            {
                let sc_pair = warp::shuffle(sc_raw, 0);
                let m_pair  = warp::shuffle(m_raw, 0);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j0 >> qs_shift) & 0xFF;
                let nl = (qs_byte & 0xF) as f32;
                let nh = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nl - dmin * m_lo) * *x_row.add((blk * 256 + lane) as usize);
                acc += (d * sc_hi * nh - dmin * m_hi) * *x_row.add((blk * 256 + 32 + lane) as usize);
            }
            // j=1
            {
                let sc_pair = warp::shuffle(sc_raw, 1);
                let m_pair  = warp::shuffle(m_raw, 1);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j1 >> qs_shift) & 0xFF;
                let nl = (qs_byte & 0xF) as f32;
                let nh = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nl - dmin * m_lo) * *x_row.add((blk * 256 + 64 + lane) as usize);
                acc += (d * sc_hi * nh - dmin * m_hi) * *x_row.add((blk * 256 + 96 + lane) as usize);
            }
            // j=2
            {
                let sc_pair = warp::shuffle(sc_raw, 2);
                let m_pair  = warp::shuffle(m_raw, 2);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j2 >> qs_shift) & 0xFF;
                let nl = (qs_byte & 0xF) as f32;
                let nh = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nl - dmin * m_lo) * *x_row.add((blk * 256 + 128 + lane) as usize);
                acc += (d * sc_hi * nh - dmin * m_hi) * *x_row.add((blk * 256 + 160 + lane) as usize);
            }
            // j=3
            {
                let sc_pair = warp::shuffle(sc_raw, 3);
                let m_pair  = warp::shuffle(m_raw, 3);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_byte = (qs_j3 >> qs_shift) & 0xFF;
                let nl = (qs_byte & 0xF) as f32;
                let nh = ((qs_byte >> 4) & 0xF) as f32;
                acc += (d * sc_lo * nl - dmin * m_lo) * *x_row.add((blk * 256 + 192 + lane) as usize);
                acc += (d * sc_hi * nh - dmin * m_hi) * *x_row.add((blk * 256 + 224 + lane) as usize);
            }

            thread::sync_threads();  // guard smem before next blk's row-0 write

            blk += WARPS_PER_BLOCK;
        }

        // Intra-warp reduce.
        acc = warp::reduce_sum_f32(acc);

        // Cross-warp reduce via smem (same as colocated).
        let smem_idx = smem_p_base + (row * WARPS_PER_BLOCK + warp_id) as usize;
        if lane == 0 {
            *smem.add(smem_idx) = acc;
        }
        thread::sync_threads();

        if warp_id == 0 {
            acc = if lane < WARPS_PER_BLOCK {
                *smem.add(smem_p_base + (row * WARPS_PER_BLOCK + lane) as usize)
            } else {
                0.0_f32
            };
            let mut delta = WARPS_PER_BLOCK / 2;
            while delta >= 1 {
                acc += warp::shuffle_xor_f32(acc, delta);
                delta /= 2;
            }
            if lane == 0 {
                *out.get_unchecked_mut((row * n + n_out) as usize) = acc;
            }
        }
    }

    /// Quantize f32 activations to Q8_1 packed format on GPU.
    ///
    /// Each warp (32 lanes) handles one 32-element Q8_1 block:
    ///   out_qs[blk * 8 .. blk * 8 + 8] = 8 u32s of packed i8 (4 per u32)
    ///   out_ds[blk]                      = f16 scale d as u16 in low 16 bits
    ///
    /// Grid:  (ceil(K/32) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK blocks
    /// Block: WARPS_PER_BLOCK * 32 threads
    #[kernel]
    pub fn quantize_q8_1(
        mut out_qs: DisjointSlice<u32>,  // [K/4] packed i8 activations
        mut out_ds: DisjointSlice<u32>,  // [K/32] f16 scale (u16 in low bits)
        x: &[f32],
        k: u32,
    ) {
        let tid     = thread::threadIdx_x();
        let bid     = thread::blockIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;
        let q8_blk  = bid * WARPS_PER_BLOCK + warp_id;  // Q8_1 block index
        let elem    = q8_blk * 32 + lane;

        let xi  = if elem < k { x[elem as usize] } else { 0.0_f32 };
        let axi = if xi < 0.0 { -xi } else { xi };

        let amax = warp_reduce_max_f32(axi);
        let d = if amax == 0.0 { 1.0_f32 } else { amax / 127.0_f32 };

        // Quantize: round to nearest, clamp to [-127, 127]
        let qf = xi / d;
        let rounded = (qf + if qf >= 0.0 { 0.5 } else { -0.5 }) as i32;
        let qi = if rounded > 127 { 127 } else if rounded < -127 { -127 } else { rounded };
        let qi_u8 = (qi as u32) & 0xFF;

        // Pack 4 lanes per u32: lane l writes with 3 shuffles to collect neighbours.
        let qi0 = warp::shuffle(qi_u8, (lane / 4) * 4 + 0);
        let qi1 = warp::shuffle(qi_u8, (lane / 4) * 4 + 1);
        let qi2 = warp::shuffle(qi_u8, (lane / 4) * 4 + 2);
        let qi3 = warp::shuffle(qi_u8, (lane / 4) * 4 + 3);
        let packed_qi = qi0 | (qi1 << 8) | (qi2 << 16) | (qi3 << 24);

        // Lane 0 of each 4-group writes one packed u32 to out_qs.
        if lane % 4 == 0 {
            let out_word_idx = q8_blk * 8 + lane / 4;
            // Guard: only write if the first element of this word is in range.
            if q8_blk * 32 + (lane / 4) * 4 < k {
                unsafe { *out_qs.get_unchecked_mut(out_word_idx as usize) = packed_qi; }
            }
        }
        // Lane 0 writes the f16 scale.
        if lane == 0 {
            unsafe {
                *out_ds.get_unchecked_mut(q8_blk as usize) =
                    f32_to_f16_bits(d) as u32;
            }
        }
    }

    /// Q4_K GEMV v2: K-loop parallelism + mini-warp dp4a (combined Opt 7+8).
    ///
    /// Two levels of parallelism:
    ///   K-level:       WARPS_PER_BLOCK warps stride across K blocks
    ///                  (warp w handles blocks w, w+4, w+8, ...)
    ///   Element-level: 4 mini-warps of 8 lanes each handle one j-group (64 elements)
    ///                  via proper 4-element dp4a (sub_l loads 1 nibble u32 + 1 Q8 u32)
    ///
    /// Reduction:
    ///   1. After K loop: 3-step XOR butterfly within each mini-warp (8 lanes -> sub_l 0).
    ///   2. sub_l==0 writes to smem[warp_id*4 + mw].
    ///   3. sync_threads (unconditional).
    ///   4. Warp 0 lane 0..3 reads all warp contributions for j-group j, sums -> smem[j].
    ///   5. sync_threads (unconditional).
    ///   6. Warp 0 lane 0 sums smem[0..4] and writes out[n_out].
    ///
    /// Grid:  N blocks (one output neuron per block; all warps cooperate on K).
    /// Block: WARPS_PER_BLOCK * 32 threads.
    /// Shared memory: WARPS_PER_BLOCK * 4 * 4 = 64 bytes.
    #[kernel]
    pub fn q4k_gemv_dp4a_q8_v2(
        mut out: DisjointSlice<f32>,
        x_qs: &[u32],    // [K/4] packed i8 activations (4 per u32)
        x_ds: &[u32],    // [K/32] Q8_1 f16 scales as u16 in low bits
        packed: &[u32],  // [N, K_BLOCKS, 42] Q4K packed weights
        n: u32,
        k_blocks: u32,
    ) {
        let tid     = thread::threadIdx_x();
        let bid     = thread::blockIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;
        let n_out   = bid;  // one output neuron per BLOCK; all warps cooperate on K
        if n_out >= n { return; }

        let mw        = lane / 8;
        let sub_l     = lane % 8;
        let mini_mask = 0xFF_u32 << (mw * 8);
        let base_lane = mw * 8;

        let mut acc = 0.0_f32;

        // K-loop: warp_id strides across K blocks (K-level parallelism).
        let mut blk = warp_id;
        while blk < k_blocks {
            let base = (n_out * k_blocks + blk) * 42;

            let d_raw    = if lane == 0 { packed[base as usize] } else { 0 };
            let dmin_raw = if lane == 0 { packed[(base + 1) as usize] } else { 0 };
            let d    = f32::from_bits(warp::shuffle(d_raw, 0));
            let dmin = f32::from_bits(warp::shuffle(dmin_raw, 0));

            let sc_raw = if sub_l == 0 { packed[(base + 2 + mw) as usize] } else { 0 };
            let m_raw  = if sub_l == 0 { packed[(base + 6 + mw) as usize] } else { 0 };
            let sc_raw_bc = warp::shuffle_sync(mini_mask, sc_raw, base_lane);
            let m_raw_bc  = warp::shuffle_sync(mini_mask, m_raw,  base_lane);
            let sc_lo = (sc_raw_bc & 0xFF) as f32;
            let sc_hi = ((sc_raw_bc >> 8) & 0xFF) as f32;
            let m_lo  = (m_raw_bc & 0xFF) as f32;
            let m_hi  = ((m_raw_bc >> 8) & 0xFF) as f32;

            let q8_blk_lo = blk * 8 + mw * 2;
            let q8_blk_hi = blk * 8 + mw * 2 + 1;
            let d8_lo = f16_to_f32(x_ds[q8_blk_lo as usize] & 0xFFFF);
            let d8_hi = f16_to_f32(x_ds[q8_blk_hi as usize] & 0xFFFF);

            let qs_word = packed[(base + 10 + mw * 8 + sub_l) as usize];
            let v_lo = qs_word & 0x0F0F0F0F;
            let v_hi = (qs_word >> 4) & 0x0F0F0F0F;

            let u_lo = x_qs[(q8_blk_lo * 8 + sub_l) as usize];
            let u_hi = x_qs[(q8_blk_hi * 8 + sub_l) as usize];

            let sumi_d_lo = dp4a_s32(v_lo, u_lo, 0);
            let sumi_d_hi = dp4a_s32(v_hi, u_hi, 0);
            let sumi_m_lo = dp4a_s32(0x01010101, u_lo, 0);
            let sumi_m_hi = dp4a_s32(0x01010101, u_hi, 0);

            acc += d8_lo * (d * sc_lo * (sumi_d_lo as f32) - dmin * m_lo * (sumi_m_lo as f32))
                 + d8_hi * (d * sc_hi * (sumi_d_hi as f32) - dmin * m_hi * (sumi_m_hi as f32));

            blk += WARPS_PER_BLOCK;
        }

        // Stage 1: 3-step XOR butterfly within mini-warp (8 lanes -> sub_l 0 holds sum).
        acc += warp::shuffle_xor_f32_sync(mini_mask, acc, 4);
        acc += warp::shuffle_xor_f32_sync(mini_mask, acc, 2);
        acc += warp::shuffle_xor_f32_sync(mini_mask, acc, 1);

        // Stage 2: write each warp's j-group partial to smem[warp_id*4 + mw].
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        if sub_l == 0 {
            unsafe { *smem.add((warp_id * 4 + mw) as usize) = acc; }
        }
        thread::sync_threads(); // unconditional -- all warps reach this

        // Stage 3: warp 0 lane j accumulates j-th j-group across all warps -> smem[j].
        // Warps 1-3 are idle but must reach the next sync.
        if warp_id == 0 && lane < 4 {
            let j = lane;
            let mut total = 0.0_f32;
            let mut w = 0u32;
            while w < WARPS_PER_BLOCK {
                total += unsafe { *smem.add((w * 4 + j) as usize) };
                w += 1;
            }
            unsafe { *smem.add(j as usize) = total; }
        }
        thread::sync_threads(); // unconditional

        if warp_id == 0 && lane == 0 {
            let total = unsafe {
                *smem.add(0) + *smem.add(1) + *smem.add(2) + *smem.add(3)
            };
            unsafe { *out.get_unchecked_mut(n_out as usize) = total; }
        }
    }

    /// Q4_K GEMV v3: VDR=2 double-block K-loop (llama.cpp access pattern).
    ///
    /// Same mini-warp dp4a structure as v2, but each K-loop iteration loads and
    /// processes TWO consecutive Q4K blocks (blk, blk+1) before any computation.
    /// This doubles concurrent DRAM requests per warp, targeting llama.cpp's 58%
    /// DRAM throughput (vs 32% in v2).
    ///
    /// K-loop: warp w starts at 2*w, strides WARPS_PER_BLOCK*2.
    /// Tail loop: handles any remaining block when k_blocks is odd.
    ///
    /// Grid:  N blocks (one output neuron per block; all warps cooperate on K).
    /// Block: WARPS_PER_BLOCK * 32 threads.
    /// Shared memory: WARPS_PER_BLOCK * 4 * 4 = 64 bytes.
    #[kernel]
    pub fn q4k_gemv_dp4a_q8(
        mut out: DisjointSlice<f32>,
        x_qs: &[u32],    // [K/4] packed i8 activations (4 per u32)
        x_ds: &[u32],    // [K/32] Q8_1 f16 scales as u16 in low bits
        packed: &[u32],  // [N, K_BLOCKS, 42] Q4K packed weights
        n: u32,
        k_blocks: u32,
    ) {
        let tid     = thread::threadIdx_x();
        let bid     = thread::blockIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;
        let n_out   = bid;
        if n_out >= n { return; }

        let mw        = lane / 8;
        let sub_l     = lane % 8;
        let mini_mask = 0xFF_u32 << (mw * 8);
        let base_lane = mw * 8;

        let mut acc = 0.0_f32;

        // -- Main double-block K-loop (VDR=2) ----------------------------------
        let mut blk = warp_id * 2;
        while blk + 1 < k_blocks {
            let base0 = (n_out * k_blocks + blk) * 42;
            let base1 = (n_out * k_blocks + blk + 1) * 42;

            // Prefetch next iteration's weight blocks (1 loop step = WARPS_PER_BLOCK*2 blocks ahead).
            // Hides DRAM latency for the next pair of blocks against current FMAs.
            let blk_pf = blk + WARPS_PER_BLOCK * 2;
            if blk_pf + 1 < k_blocks {
                let base_pf0 = ((n_out * k_blocks + blk_pf) * 42) as usize;
                let base_pf1 = ((n_out * k_blocks + blk_pf + 1) * 42) as usize;
                unsafe {
                    let ptr_pf0 = packed.as_ptr().add(base_pf0) as u64;
                    let ptr_pf1 = packed.as_ptr().add(base_pf1) as u64;
                    ptx_asm!("prefetch.global.L2 [%0];", in("l") ptr_pf0, clobber("memory"));
                    ptx_asm!("prefetch.global.L2 [%0];", in("l") ptr_pf1, clobber("memory"));
                }
            }

            // Phase 1: Issue ALL loads for BOTH blocks before any shuffle.
            let d_raw0    = if lane == 0 { packed[base0 as usize] } else { 0 };
            let dmin_raw0 = if lane == 0 { packed[(base0 + 1) as usize] } else { 0 };
            let sc_raw0   = if sub_l == 0 { packed[(base0 + 2 + mw) as usize] } else { 0 };
            let m_raw0    = if sub_l == 0 { packed[(base0 + 6 + mw) as usize] } else { 0 };
            let qs0       = packed[(base0 + 10 + mw * 8 + sub_l) as usize];
            let q8b0_lo   = blk * 8 + mw * 2;
            let q8b0_hi   = q8b0_lo + 1;
            let u0_lo     = x_qs[(q8b0_lo * 8 + sub_l) as usize];
            let u0_hi     = x_qs[(q8b0_hi * 8 + sub_l) as usize];

            let d_raw1    = if lane == 0 { packed[base1 as usize] } else { 0 };
            let dmin_raw1 = if lane == 0 { packed[(base1 + 1) as usize] } else { 0 };
            let sc_raw1   = if sub_l == 0 { packed[(base1 + 2 + mw) as usize] } else { 0 };
            let m_raw1    = if sub_l == 0 { packed[(base1 + 6 + mw) as usize] } else { 0 };
            let qs1       = packed[(base1 + 10 + mw * 8 + sub_l) as usize];
            let q8b1_lo   = (blk + 1) * 8 + mw * 2;
            let q8b1_hi   = q8b1_lo + 1;
            let u1_lo     = x_qs[(q8b1_lo * 8 + sub_l) as usize];
            let u1_hi     = x_qs[(q8b1_hi * 8 + sub_l) as usize];

            // Phase 2: Compute block 0.
            let d0    = f32::from_bits(warp::shuffle(d_raw0, 0));
            let dmin0 = f32::from_bits(warp::shuffle(dmin_raw0, 0));
            let sc0_bc = warp::shuffle_sync(mini_mask, sc_raw0, base_lane);
            let m0_bc  = warp::shuffle_sync(mini_mask, m_raw0,  base_lane);
            let d8_0_lo = f16_to_f32(x_ds[q8b0_lo as usize] & 0xFFFF);
            let d8_0_hi = f16_to_f32(x_ds[q8b0_hi as usize] & 0xFFFF);
            let v0_lo = qs0 & 0x0F0F0F0F;
            let v0_hi = (qs0 >> 4) & 0x0F0F0F0F;
            acc += d8_0_lo * (d0 * ((sc0_bc & 0xFF) as f32) * (dp4a_s32(v0_lo, u0_lo, 0) as f32)
                             - dmin0 * ((m0_bc & 0xFF) as f32) * (dp4a_s32(0x01010101, u0_lo, 0) as f32))
                 + d8_0_hi * (d0 * (((sc0_bc >> 8) & 0xFF) as f32) * (dp4a_s32(v0_hi, u0_hi, 0) as f32)
                             - dmin0 * (((m0_bc >> 8) & 0xFF) as f32) * (dp4a_s32(0x01010101, u0_hi, 0) as f32));

            // Phase 3: Compute block 1.
            let d1    = f32::from_bits(warp::shuffle(d_raw1, 0));
            let dmin1 = f32::from_bits(warp::shuffle(dmin_raw1, 0));
            let sc1_bc = warp::shuffle_sync(mini_mask, sc_raw1, base_lane);
            let m1_bc  = warp::shuffle_sync(mini_mask, m_raw1,  base_lane);
            let d8_1_lo = f16_to_f32(x_ds[q8b1_lo as usize] & 0xFFFF);
            let d8_1_hi = f16_to_f32(x_ds[q8b1_hi as usize] & 0xFFFF);
            let v1_lo = qs1 & 0x0F0F0F0F;
            let v1_hi = (qs1 >> 4) & 0x0F0F0F0F;
            acc += d8_1_lo * (d1 * ((sc1_bc & 0xFF) as f32) * (dp4a_s32(v1_lo, u1_lo, 0) as f32)
                             - dmin1 * ((m1_bc & 0xFF) as f32) * (dp4a_s32(0x01010101, u1_lo, 0) as f32))
                 + d8_1_hi * (d1 * (((sc1_bc >> 8) & 0xFF) as f32) * (dp4a_s32(v1_hi, u1_hi, 0) as f32)
                             - dmin1 * (((m1_bc >> 8) & 0xFF) as f32) * (dp4a_s32(0x01010101, u1_hi, 0) as f32));

            blk += WARPS_PER_BLOCK * 2;
        }

        // -- Tail: single-block fallback for odd k_blocks ----------------------
        while blk < k_blocks {
            let base = (n_out * k_blocks + blk) * 42;
            let d_raw    = if lane == 0 { packed[base as usize] } else { 0 };
            let dmin_raw = if lane == 0 { packed[(base + 1) as usize] } else { 0 };
            let d    = f32::from_bits(warp::shuffle(d_raw, 0));
            let dmin = f32::from_bits(warp::shuffle(dmin_raw, 0));
            let sc_raw = if sub_l == 0 { packed[(base + 2 + mw) as usize] } else { 0 };
            let m_raw  = if sub_l == 0 { packed[(base + 6 + mw) as usize] } else { 0 };
            let sc_bc = warp::shuffle_sync(mini_mask, sc_raw, base_lane);
            let m_bc  = warp::shuffle_sync(mini_mask, m_raw,  base_lane);
            let q8_lo = blk * 8 + mw * 2;
            let q8_hi = q8_lo + 1;
            let d8_lo = f16_to_f32(x_ds[q8_lo as usize] & 0xFFFF);
            let d8_hi = f16_to_f32(x_ds[q8_hi as usize] & 0xFFFF);
            let qs    = packed[(base + 10 + mw * 8 + sub_l) as usize];
            let v_lo  = qs & 0x0F0F0F0F;
            let v_hi  = (qs >> 4) & 0x0F0F0F0F;
            let u_lo  = x_qs[(q8_lo * 8 + sub_l) as usize];
            let u_hi  = x_qs[(q8_hi * 8 + sub_l) as usize];
            acc += d8_lo * (d * ((sc_bc & 0xFF) as f32) * (dp4a_s32(v_lo, u_lo, 0) as f32)
                           - dmin * ((m_bc & 0xFF) as f32) * (dp4a_s32(0x01010101, u_lo, 0) as f32))
                 + d8_hi * (d * (((sc_bc >> 8) & 0xFF) as f32) * (dp4a_s32(v_hi, u_hi, 0) as f32)
                           - dmin * (((m_bc >> 8) & 0xFF) as f32) * (dp4a_s32(0x01010101, u_hi, 0) as f32));
            blk += WARPS_PER_BLOCK * 2;
        }

        // -- Reduction: identical to v2 -----------------------------------------
        acc += warp::shuffle_xor_f32_sync(mini_mask, acc, 4);
        acc += warp::shuffle_xor_f32_sync(mini_mask, acc, 2);
        acc += warp::shuffle_xor_f32_sync(mini_mask, acc, 1);

        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        if sub_l == 0 {
            unsafe { *smem.add((warp_id * 4 + mw) as usize) = acc; }
        }
        thread::sync_threads();

        if warp_id == 0 && lane < 4 {
            let j = lane;
            let mut total = 0.0_f32;
            let mut w = 0u32;
            while w < WARPS_PER_BLOCK {
                total += unsafe { *smem.add((w * 4 + j) as usize) };
                w += 1;
            }
            unsafe { *smem.add(j as usize) = total; }
        }
        thread::sync_threads();

        if warp_id == 0 && lane == 0 {
            let total = unsafe {
                *smem.add(0) + *smem.add(1) + *smem.add(2) + *smem.add(3)
            };
            unsafe { *out.get_unchecked_mut(n_out as usize) = total; }
        }
    }

    /// K-parallel Q4K+Q8_1 GEMV: 8 warps per block for 100% SM occupancy on large projections.
    ///
    /// Identical to q4k_gemv_decode_q8_v3 but with WARPS_PER_BLOCK_KPAR=8 warps per block.
    /// ncu showed q4k_gemv_decode_q8_v3 on Q/O projections [4096x4096] at 68.8% occupancy:
    ///   4096 blocks x 4 warps = 16384 warps / 170 SMs -> 11 blocks/SM x 4 warps = 44 warps/SM.
    /// With 8 warps/block:
    ///   4096 blocks x 8 warps = 32768 warps / 170 SMs -> capped at 64 warps/SM = 100%.
    ///
    /// For k_blocks=16: each warp handles exactly 2 k_blocks (warp_id*2, warp_id*2+1).
    /// Smem: 8 warps x 4 mw slots = 32 f32 slots = 128 bytes (Blackwell predicated-smem safe).
    ///
    /// Grid:  (N, 1, 1) -- one output neuron per block.
    /// Block: WARPS_PER_BLOCK_KPAR * 32 = 256 threads.
    /// Shared memory: 32 * 4 = 128 bytes.
    #[kernel]
    pub fn q4k_gemv_dp4a_kpar8(
        mut out: DisjointSlice<f32>,
        x_qs: &[u32],    // [K/4] packed i8 activations (4 per u32)
        x_ds: &[u32],    // [K/32] Q8_1 f16 scales as u16 in low bits
        packed: &[u32],  // [N, K_BLOCKS, 42] Q4K packed weights
        n: u32,
        k_blocks: u32,
    ) {
        let tid     = thread::threadIdx_x();
        let bid     = thread::blockIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;
        let n_out   = bid;
        if n_out >= n { return; }

        let mw        = lane / 8;
        let sub_l     = lane % 8;
        let mini_mask = 0xFF_u32 << (mw * 8);
        let base_lane = mw * 8;

        let mut acc = 0.0_f32;

        // -- Main double-block K-loop (VDR=2) ----------------------------------
        let mut blk = warp_id * 2;
        while blk + 1 < k_blocks {
            let base0 = (n_out * k_blocks + blk) * 42;
            let base1 = (n_out * k_blocks + blk + 1) * 42;

            // Prefetch next iteration's weight blocks (stride=WARPS_PER_BLOCK_KPAR*2=16 blocks).
            let blk_pf = blk + WARPS_PER_BLOCK_KPAR * 2;
            if blk_pf + 1 < k_blocks {
                let base_pf0 = ((n_out * k_blocks + blk_pf) * 42) as usize;
                let base_pf1 = ((n_out * k_blocks + blk_pf + 1) * 42) as usize;
                unsafe {
                    let ptr_pf0 = packed.as_ptr().add(base_pf0) as u64;
                    let ptr_pf1 = packed.as_ptr().add(base_pf1) as u64;
                    ptx_asm!("prefetch.global.L2 [%0];", in("l") ptr_pf0, clobber("memory"));
                    ptx_asm!("prefetch.global.L2 [%0];", in("l") ptr_pf1, clobber("memory"));
                }
            }

            // Phase 1: Issue ALL loads for BOTH blocks before any shuffle.
            let d_raw0    = if lane == 0 { packed[base0 as usize] } else { 0 };
            let dmin_raw0 = if lane == 0 { packed[(base0 + 1) as usize] } else { 0 };
            let sc_raw0   = if sub_l == 0 { packed[(base0 + 2 + mw) as usize] } else { 0 };
            let m_raw0    = if sub_l == 0 { packed[(base0 + 6 + mw) as usize] } else { 0 };
            let qs0       = packed[(base0 + 10 + mw * 8 + sub_l) as usize];
            let q8b0_lo   = blk * 8 + mw * 2;
            let q8b0_hi   = q8b0_lo + 1;
            let u0_lo     = x_qs[(q8b0_lo * 8 + sub_l) as usize];
            let u0_hi     = x_qs[(q8b0_hi * 8 + sub_l) as usize];

            let d_raw1    = if lane == 0 { packed[base1 as usize] } else { 0 };
            let dmin_raw1 = if lane == 0 { packed[(base1 + 1) as usize] } else { 0 };
            let sc_raw1   = if sub_l == 0 { packed[(base1 + 2 + mw) as usize] } else { 0 };
            let m_raw1    = if sub_l == 0 { packed[(base1 + 6 + mw) as usize] } else { 0 };
            let qs1       = packed[(base1 + 10 + mw * 8 + sub_l) as usize];
            let q8b1_lo   = (blk + 1) * 8 + mw * 2;
            let q8b1_hi   = q8b1_lo + 1;
            let u1_lo     = x_qs[(q8b1_lo * 8 + sub_l) as usize];
            let u1_hi     = x_qs[(q8b1_hi * 8 + sub_l) as usize];

            // Phase 2: Compute block 0.
            let d0    = f32::from_bits(warp::shuffle(d_raw0, 0));
            let dmin0 = f32::from_bits(warp::shuffle(dmin_raw0, 0));
            let sc0_bc = warp::shuffle_sync(mini_mask, sc_raw0, base_lane);
            let m0_bc  = warp::shuffle_sync(mini_mask, m_raw0,  base_lane);
            let d8_0_lo = f16_to_f32(x_ds[q8b0_lo as usize] & 0xFFFF);
            let d8_0_hi = f16_to_f32(x_ds[q8b0_hi as usize] & 0xFFFF);
            let v0_lo = qs0 & 0x0F0F0F0F;
            let v0_hi = (qs0 >> 4) & 0x0F0F0F0F;
            acc += d8_0_lo * (d0 * ((sc0_bc & 0xFF) as f32) * (dp4a_s32(v0_lo, u0_lo, 0) as f32)
                             - dmin0 * ((m0_bc & 0xFF) as f32) * (dp4a_s32(0x01010101, u0_lo, 0) as f32))
                 + d8_0_hi * (d0 * (((sc0_bc >> 8) & 0xFF) as f32) * (dp4a_s32(v0_hi, u0_hi, 0) as f32)
                             - dmin0 * (((m0_bc >> 8) & 0xFF) as f32) * (dp4a_s32(0x01010101, u0_hi, 0) as f32));

            // Phase 3: Compute block 1.
            let d1    = f32::from_bits(warp::shuffle(d_raw1, 0));
            let dmin1 = f32::from_bits(warp::shuffle(dmin_raw1, 0));
            let sc1_bc = warp::shuffle_sync(mini_mask, sc_raw1, base_lane);
            let m1_bc  = warp::shuffle_sync(mini_mask, m_raw1,  base_lane);
            let d8_1_lo = f16_to_f32(x_ds[q8b1_lo as usize] & 0xFFFF);
            let d8_1_hi = f16_to_f32(x_ds[q8b1_hi as usize] & 0xFFFF);
            let v1_lo = qs1 & 0x0F0F0F0F;
            let v1_hi = (qs1 >> 4) & 0x0F0F0F0F;
            acc += d8_1_lo * (d1 * ((sc1_bc & 0xFF) as f32) * (dp4a_s32(v1_lo, u1_lo, 0) as f32)
                             - dmin1 * ((m1_bc & 0xFF) as f32) * (dp4a_s32(0x01010101, u1_lo, 0) as f32))
                 + d8_1_hi * (d1 * (((sc1_bc >> 8) & 0xFF) as f32) * (dp4a_s32(v1_hi, u1_hi, 0) as f32)
                             - dmin1 * (((m1_bc >> 8) & 0xFF) as f32) * (dp4a_s32(0x01010101, u1_hi, 0) as f32));

            blk += WARPS_PER_BLOCK_KPAR * 2;
        }

        // -- Tail: single-block fallback for odd k_blocks ----------------------
        while blk < k_blocks {
            let base = (n_out * k_blocks + blk) * 42;
            let d_raw    = if lane == 0 { packed[base as usize] } else { 0 };
            let dmin_raw = if lane == 0 { packed[(base + 1) as usize] } else { 0 };
            let d    = f32::from_bits(warp::shuffle(d_raw, 0));
            let dmin = f32::from_bits(warp::shuffle(dmin_raw, 0));
            let sc_raw = if sub_l == 0 { packed[(base + 2 + mw) as usize] } else { 0 };
            let m_raw  = if sub_l == 0 { packed[(base + 6 + mw) as usize] } else { 0 };
            let sc_bc = warp::shuffle_sync(mini_mask, sc_raw, base_lane);
            let m_bc  = warp::shuffle_sync(mini_mask, m_raw,  base_lane);
            let q8_lo = blk * 8 + mw * 2;
            let q8_hi = q8_lo + 1;
            let d8_lo = f16_to_f32(x_ds[q8_lo as usize] & 0xFFFF);
            let d8_hi = f16_to_f32(x_ds[q8_hi as usize] & 0xFFFF);
            let qs    = packed[(base + 10 + mw * 8 + sub_l) as usize];
            let v_lo  = qs & 0x0F0F0F0F;
            let v_hi  = (qs >> 4) & 0x0F0F0F0F;
            let u_lo  = x_qs[(q8_lo * 8 + sub_l) as usize];
            let u_hi  = x_qs[(q8_hi * 8 + sub_l) as usize];
            acc += d8_lo * (d * ((sc_bc & 0xFF) as f32) * (dp4a_s32(v_lo, u_lo, 0) as f32)
                           - dmin * ((m_bc & 0xFF) as f32) * (dp4a_s32(0x01010101, u_lo, 0) as f32))
                 + d8_hi * (d * (((sc_bc >> 8) & 0xFF) as f32) * (dp4a_s32(v_hi, u_hi, 0) as f32)
                           - dmin * (((m_bc >> 8) & 0xFF) as f32) * (dp4a_s32(0x01010101, u_hi, 0) as f32));
            blk += WARPS_PER_BLOCK_KPAR * 2;
        }

        // -- Reduction: 8-warp cross-warp reduce via shared memory --------------
        // Mini-warp intra-reduce (XOR 4, 2, 1 within the 8-lane mini-warp).
        acc += warp::shuffle_xor_f32_sync(mini_mask, acc, 4);
        acc += warp::shuffle_xor_f32_sync(mini_mask, acc, 2);
        acc += warp::shuffle_xor_f32_sync(mini_mask, acc, 1);

        // smem layout: [warp_id * 4 + mw] for warp_id 0..7, mw 0..3 -> 32 slots = 128 bytes.
        // 128 bytes satisfies Blackwell predicated-smem safety (max speculative index = 31).
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        if sub_l == 0 {
            unsafe { *smem.add((warp_id * 4 + mw) as usize) = acc; }
        }
        thread::sync_threads();

        if warp_id == 0 && lane < 4 {
            let j = lane;
            let mut total = 0.0_f32;
            let mut w = 0u32;
            while w < WARPS_PER_BLOCK_KPAR {
                total += unsafe { *smem.add((w * 4 + j) as usize) };
                w += 1;
            }
            unsafe { *smem.add(j as usize) = total; }
        }
        thread::sync_threads();

        if warp_id == 0 && lane == 0 {
            let total = unsafe {
                *smem.add(0) + *smem.add(1) + *smem.add(2) + *smem.add(3)
            };
            unsafe { *out.get_unchecked_mut(n_out as usize) = total; }
        }
    }

    /// Fused gate+up Q4K GEMV with inline SiLU: out[i] = silu(gate_dot[i]) * up_dot[i]
    ///
    /// Processes gate_proj and up_proj weight matrices in a single kernel launch,
    /// sharing the Q8_1 activation load between both GEMVs and applying SiLU inline.
    /// Halves activation DRAM traffic and kernel launches for FFN layers vs two GEMVs.
    ///
    /// Grid:  (N, 1, 1) -- one block per output neuron
    /// Block: (WARPS_PER_BLOCK * 32, 1, 1) -- 4 warps
    /// Smem:  32 f32 slots (128 bytes): [0..16] gate partials, [16..32] up partials
    #[kernel]
    pub fn q4k_fused_silu_dp4a(
        mut out: DisjointSlice<f32>,
        gate_packed: &[u32],  // [N, K_BLOCKS, 42] Q4K gate_proj weights
        up_packed:   &[u32],  // [N, K_BLOCKS, 42] Q4K up_proj weights
        x_qs: &[u32],         // [K/4] Q8_1 quantized activations (4 int8 per u32)
        x_ds: &[u32],         // [K/32] Q8_1 f16 block scales
        n: u32,
        k_blocks: u32,
    ) {
        let tid     = thread::threadIdx_x();
        let bid     = thread::blockIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;
        let n_out   = bid;
        if n_out >= n { return; }

        let mw        = lane / 8;
        let sub_l     = lane % 8;
        let mini_mask = 0xFF_u32 << (mw * 8);
        let base_lane = mw * 8;

        let mut gate_acc = 0.0_f32;
        let mut up_acc   = 0.0_f32;

        // -- Main double-block K-loop (VDR=2): mirrors q4k_gemv_decode_q8_v3 ------
        let mut blk = warp_id * 2;
        while blk + 1 < k_blocks {
            let base0 = (n_out * k_blocks + blk) * 42;
            let base1 = (n_out * k_blocks + blk + 1) * 42;

            // Phase 1: Issue ALL loads for BOTH blocks and BOTH weight matrices.
            // The activation loads (u0_lo, u0_hi, u1_lo, u1_hi) are shared.
            let gd_raw0    = if lane == 0 { gate_packed[base0 as usize] } else { 0 };
            let gdmin_raw0 = if lane == 0 { gate_packed[(base0 + 1) as usize] } else { 0 };
            let gsc_raw0   = if sub_l == 0 { gate_packed[(base0 + 2 + mw) as usize] } else { 0 };
            let gm_raw0    = if sub_l == 0 { gate_packed[(base0 + 6 + mw) as usize] } else { 0 };
            let gqs0       = gate_packed[(base0 + 10 + mw * 8 + sub_l) as usize];

            let ud_raw0    = if lane == 0 { up_packed[base0 as usize] } else { 0 };
            let udmin_raw0 = if lane == 0 { up_packed[(base0 + 1) as usize] } else { 0 };
            let usc_raw0   = if sub_l == 0 { up_packed[(base0 + 2 + mw) as usize] } else { 0 };
            let um_raw0    = if sub_l == 0 { up_packed[(base0 + 6 + mw) as usize] } else { 0 };
            let uqs0       = up_packed[(base0 + 10 + mw * 8 + sub_l) as usize];

            let q8b0_lo = blk * 8 + mw * 2;
            let q8b0_hi = q8b0_lo + 1;
            let u0_lo   = x_qs[(q8b0_lo * 8 + sub_l) as usize];
            let u0_hi   = x_qs[(q8b0_hi * 8 + sub_l) as usize];

            let gd_raw1    = if lane == 0 { gate_packed[base1 as usize] } else { 0 };
            let gdmin_raw1 = if lane == 0 { gate_packed[(base1 + 1) as usize] } else { 0 };
            let gsc_raw1   = if sub_l == 0 { gate_packed[(base1 + 2 + mw) as usize] } else { 0 };
            let gm_raw1    = if sub_l == 0 { gate_packed[(base1 + 6 + mw) as usize] } else { 0 };
            let gqs1       = gate_packed[(base1 + 10 + mw * 8 + sub_l) as usize];

            let ud_raw1    = if lane == 0 { up_packed[base1 as usize] } else { 0 };
            let udmin_raw1 = if lane == 0 { up_packed[(base1 + 1) as usize] } else { 0 };
            let usc_raw1   = if sub_l == 0 { up_packed[(base1 + 2 + mw) as usize] } else { 0 };
            let um_raw1    = if sub_l == 0 { up_packed[(base1 + 6 + mw) as usize] } else { 0 };
            let uqs1       = up_packed[(base1 + 10 + mw * 8 + sub_l) as usize];

            let q8b1_lo = (blk + 1) * 8 + mw * 2;
            let q8b1_hi = q8b1_lo + 1;
            let u1_lo   = x_qs[(q8b1_lo * 8 + sub_l) as usize];
            let u1_hi   = x_qs[(q8b1_hi * 8 + sub_l) as usize];

            // Phase 2: Compute gate and up for block 0 -- D deferred to post-sumf (llama.cpp pattern).
            // Accumulate sumf_d (dot terms) and sumf_m (min terms) without the super-block scale D,
            // then apply D once: gate_acc += gd * sumf_d - gdmin * sumf_m.
            let d8_0_lo = f16_to_f32(x_ds[q8b0_lo as usize] & 0xFFFF);
            let d8_0_hi = f16_to_f32(x_ds[q8b0_hi as usize] & 0xFFFF);

            let gd0    = f32::from_bits(warp::shuffle(gd_raw0, 0));
            let gdmin0 = f32::from_bits(warp::shuffle(gdmin_raw0, 0));
            let gsc0   = warp::shuffle_sync(mini_mask, gsc_raw0, base_lane);
            let gm0    = warp::shuffle_sync(mini_mask, gm_raw0,  base_lane);
            let gv0_lo = gqs0 & 0x0F0F0F0F;
            let gv0_hi = (gqs0 >> 4) & 0x0F0F0F0F;
            let gsfd0 = d8_0_lo * ((gsc0 & 0xFF) as f32) * (dp4a_s32(gv0_lo, u0_lo, 0) as f32)
                       + d8_0_hi * (((gsc0 >> 8) & 0xFF) as f32) * (dp4a_s32(gv0_hi, u0_hi, 0) as f32);
            let gsfm0 = d8_0_lo * ((gm0 & 0xFF) as f32) * (dp4a_s32(0x01010101, u0_lo, 0) as f32)
                       + d8_0_hi * (((gm0 >> 8) & 0xFF) as f32) * (dp4a_s32(0x01010101, u0_hi, 0) as f32);
            gate_acc += gd0 * gsfd0 - gdmin0 * gsfm0;

            let ud0    = f32::from_bits(warp::shuffle(ud_raw0, 0));
            let udmin0 = f32::from_bits(warp::shuffle(udmin_raw0, 0));
            let usc0   = warp::shuffle_sync(mini_mask, usc_raw0, base_lane);
            let um0    = warp::shuffle_sync(mini_mask, um_raw0,  base_lane);
            let uv0_lo = uqs0 & 0x0F0F0F0F;
            let uv0_hi = (uqs0 >> 4) & 0x0F0F0F0F;
            let usfd0 = d8_0_lo * ((usc0 & 0xFF) as f32) * (dp4a_s32(uv0_lo, u0_lo, 0) as f32)
                       + d8_0_hi * (((usc0 >> 8) & 0xFF) as f32) * (dp4a_s32(uv0_hi, u0_hi, 0) as f32);
            let usfm0 = d8_0_lo * ((um0 & 0xFF) as f32) * (dp4a_s32(0x01010101, u0_lo, 0) as f32)
                       + d8_0_hi * (((um0 >> 8) & 0xFF) as f32) * (dp4a_s32(0x01010101, u0_hi, 0) as f32);
            up_acc += ud0 * usfd0 - udmin0 * usfm0;

            // Phase 3: Compute gate and up for block 1 -- D deferred to post-sumf.
            let d8_1_lo = f16_to_f32(x_ds[q8b1_lo as usize] & 0xFFFF);
            let d8_1_hi = f16_to_f32(x_ds[q8b1_hi as usize] & 0xFFFF);

            let gd1    = f32::from_bits(warp::shuffle(gd_raw1, 0));
            let gdmin1 = f32::from_bits(warp::shuffle(gdmin_raw1, 0));
            let gsc1   = warp::shuffle_sync(mini_mask, gsc_raw1, base_lane);
            let gm1    = warp::shuffle_sync(mini_mask, gm_raw1,  base_lane);
            let gv1_lo = gqs1 & 0x0F0F0F0F;
            let gv1_hi = (gqs1 >> 4) & 0x0F0F0F0F;
            let gsfd1 = d8_1_lo * ((gsc1 & 0xFF) as f32) * (dp4a_s32(gv1_lo, u1_lo, 0) as f32)
                       + d8_1_hi * (((gsc1 >> 8) & 0xFF) as f32) * (dp4a_s32(gv1_hi, u1_hi, 0) as f32);
            let gsfm1 = d8_1_lo * ((gm1 & 0xFF) as f32) * (dp4a_s32(0x01010101, u1_lo, 0) as f32)
                       + d8_1_hi * (((gm1 >> 8) & 0xFF) as f32) * (dp4a_s32(0x01010101, u1_hi, 0) as f32);
            gate_acc += gd1 * gsfd1 - gdmin1 * gsfm1;

            let ud1    = f32::from_bits(warp::shuffle(ud_raw1, 0));
            let udmin1 = f32::from_bits(warp::shuffle(udmin_raw1, 0));
            let usc1   = warp::shuffle_sync(mini_mask, usc_raw1, base_lane);
            let um1    = warp::shuffle_sync(mini_mask, um_raw1,  base_lane);
            let uv1_lo = uqs1 & 0x0F0F0F0F;
            let uv1_hi = (uqs1 >> 4) & 0x0F0F0F0F;
            let usfd1 = d8_1_lo * ((usc1 & 0xFF) as f32) * (dp4a_s32(uv1_lo, u1_lo, 0) as f32)
                       + d8_1_hi * (((usc1 >> 8) & 0xFF) as f32) * (dp4a_s32(uv1_hi, u1_hi, 0) as f32);
            let usfm1 = d8_1_lo * ((um1 & 0xFF) as f32) * (dp4a_s32(0x01010101, u1_lo, 0) as f32)
                       + d8_1_hi * (((um1 >> 8) & 0xFF) as f32) * (dp4a_s32(0x01010101, u1_hi, 0) as f32);
            up_acc += ud1 * usfd1 - udmin1 * usfm1;

            blk += WARPS_PER_BLOCK * 2;
        }

        // -- Tail: single-block fallback for odd k_blocks ----------------------
        while blk < k_blocks {
            let base = (n_out * k_blocks + blk) * 42;

            let gd_raw    = if lane == 0 { gate_packed[base as usize] } else { 0 };
            let gdmin_raw = if lane == 0 { gate_packed[(base + 1) as usize] } else { 0 };
            let gsc_raw   = if sub_l == 0 { gate_packed[(base + 2 + mw) as usize] } else { 0 };
            let gm_raw    = if sub_l == 0 { gate_packed[(base + 6 + mw) as usize] } else { 0 };
            let gqs       = gate_packed[(base + 10 + mw * 8 + sub_l) as usize];

            let ud_raw    = if lane == 0 { up_packed[base as usize] } else { 0 };
            let udmin_raw = if lane == 0 { up_packed[(base + 1) as usize] } else { 0 };
            let usc_raw   = if sub_l == 0 { up_packed[(base + 2 + mw) as usize] } else { 0 };
            let um_raw    = if sub_l == 0 { up_packed[(base + 6 + mw) as usize] } else { 0 };
            let uqs       = up_packed[(base + 10 + mw * 8 + sub_l) as usize];

            let gd    = f32::from_bits(warp::shuffle(gd_raw, 0));
            let gdmin = f32::from_bits(warp::shuffle(gdmin_raw, 0));
            let gsc   = warp::shuffle_sync(mini_mask, gsc_raw, base_lane);
            let gm    = warp::shuffle_sync(mini_mask, gm_raw,  base_lane);

            let ud    = f32::from_bits(warp::shuffle(ud_raw, 0));
            let udmin = f32::from_bits(warp::shuffle(udmin_raw, 0));
            let usc   = warp::shuffle_sync(mini_mask, usc_raw, base_lane);
            let um    = warp::shuffle_sync(mini_mask, um_raw,  base_lane);

            let q8_lo = blk * 8 + mw * 2;
            let q8_hi = q8_lo + 1;
            let d8_lo = f16_to_f32(x_ds[q8_lo as usize] & 0xFFFF);
            let d8_hi = f16_to_f32(x_ds[q8_hi as usize] & 0xFFFF);
            let u_lo  = x_qs[(q8_lo * 8 + sub_l) as usize];
            let u_hi  = x_qs[(q8_hi * 8 + sub_l) as usize];

            let gv_lo = gqs & 0x0F0F0F0F;
            let gv_hi = (gqs >> 4) & 0x0F0F0F0F;
            let gsfd = d8_lo * ((gsc & 0xFF) as f32) * (dp4a_s32(gv_lo, u_lo, 0) as f32)
                      + d8_hi * (((gsc >> 8) & 0xFF) as f32) * (dp4a_s32(gv_hi, u_hi, 0) as f32);
            let gsfm = d8_lo * ((gm & 0xFF) as f32) * (dp4a_s32(0x01010101, u_lo, 0) as f32)
                      + d8_hi * (((gm >> 8) & 0xFF) as f32) * (dp4a_s32(0x01010101, u_hi, 0) as f32);
            gate_acc += gd * gsfd - gdmin * gsfm;

            let uv_lo = uqs & 0x0F0F0F0F;
            let uv_hi = (uqs >> 4) & 0x0F0F0F0F;
            let usfd = d8_lo * ((usc & 0xFF) as f32) * (dp4a_s32(uv_lo, u_lo, 0) as f32)
                      + d8_hi * (((usc >> 8) & 0xFF) as f32) * (dp4a_s32(uv_hi, u_hi, 0) as f32);
            let usfm = d8_lo * ((um & 0xFF) as f32) * (dp4a_s32(0x01010101, u_lo, 0) as f32)
                      + d8_hi * (((um >> 8) & 0xFF) as f32) * (dp4a_s32(0x01010101, u_hi, 0) as f32);
            up_acc += ud * usfd - udmin * usfm;

            blk += WARPS_PER_BLOCK * 2;
        }

        // -- Mini-warp butterfly reduction for both accumulators ----------------
        gate_acc += warp::shuffle_xor_f32_sync(mini_mask, gate_acc, 4);
        gate_acc += warp::shuffle_xor_f32_sync(mini_mask, gate_acc, 2);
        gate_acc += warp::shuffle_xor_f32_sync(mini_mask, gate_acc, 1);
        up_acc += warp::shuffle_xor_f32_sync(mini_mask, up_acc, 4);
        up_acc += warp::shuffle_xor_f32_sync(mini_mask, up_acc, 2);
        up_acc += warp::shuffle_xor_f32_sync(mini_mask, up_acc, 1);

        // -- Cross-warp reduction via shared memory -----------------------------
        // smem layout: [0..16] gate (warp_id*4+mw), [16..32] up (16+warp_id*4+mw)
        // Total 32 f32 slots = 128 bytes (same as v3 Blackwell safety budget).
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        if sub_l == 0 {
            unsafe {
                *smem.add((warp_id * 4 + mw) as usize)      = gate_acc;
                *smem.add((16 + warp_id * 4 + mw) as usize) = up_acc;
            }
        }
        thread::sync_threads();

        if warp_id == 0 && lane < 4 {
            let j = lane;
            let mut gtotal = 0.0_f32;
            let mut utotal = 0.0_f32;
            let mut w = 0u32;
            while w < WARPS_PER_BLOCK {
                gtotal += unsafe { *smem.add((w * 4 + j) as usize) };
                utotal += unsafe { *smem.add((16 + w * 4 + j) as usize) };
                w += 1;
            }
            unsafe {
                *smem.add(j as usize)        = gtotal;
                *smem.add((16 + j) as usize) = utotal;
            }
        }
        thread::sync_threads();

        if warp_id == 0 && lane == 0 {
            let gate_val = unsafe { *smem.add(0) + *smem.add(1) + *smem.add(2) + *smem.add(3) };
            let up_val   = unsafe { *smem.add(16) + *smem.add(17) + *smem.add(18) + *smem.add(19) };
            // SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
            let sigmoid = 1.0_f32 / (1.0_f32 + f32::exp(-gate_val));
            let out_val = gate_val * sigmoid * up_val;
            unsafe { *out.get_unchecked_mut(n_out as usize) = out_val; }
        }
    }

    /// q4k_gemv_decode_q8_v3r -- raw GGUF 36-u32 format variant of v3.
    /// 36 u32 (144 bytes) per block vs 42 u32 (168 bytes) -> 14% less DRAM.
    #[kernel]
    pub fn q4k_gemv_dp4a_q8_raw(
        mut out: DisjointSlice<f32>,
        x_qs: &[u32],
        x_ds: &[u32],
        packed: &[u32],   // [N, K_BLOCKS, 36] raw GGUF format
        n: u32,
        k_blocks: u32,
    ) {
        let tid     = thread::threadIdx_x();
        let bid     = thread::blockIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;
        let n_out   = bid;
        if n_out >= n { return; }

        let mw        = lane / 8;
        let sub_l     = lane % 8;
        let mini_mask = 0xFF_u32 << (mw * 8);
        let base_lane = mw * 8;

        let mut acc = 0.0_f32;

        let mut blk = warp_id * 2;
        while blk + 1 < k_blocks {
            let base0 = (n_out * k_blocks + blk) * 36;
            let base1 = (n_out * k_blocks + blk + 1) * 36;

            // Prefetch next iteration's weight blocks into L2 (1 loop step ahead).
            let blk_pf = blk + WARPS_PER_BLOCK * 2;
            if blk_pf + 1 < k_blocks {
                let base_pf0 = ((n_out * k_blocks + blk_pf) * 36) as usize;
                let base_pf1 = ((n_out * k_blocks + blk_pf + 1) * 36) as usize;
                unsafe {
                    let ptr_pf0 = packed.as_ptr().add(base_pf0) as u64;
                    let ptr_pf1 = packed.as_ptr().add(base_pf1) as u64;
                    ptx_asm!("prefetch.global.L2 [%0];", in("l") ptr_pf0, clobber("memory"));
                    ptx_asm!("prefetch.global.L2 [%0];", in("l") ptr_pf1, clobber("memory"));
                }
            }

            // -- Phase 1: ALL loads for BOTH blocks upfront -----------------
            // u32[0] = f16 d | (f16 dmin << 16)
            let dd0 = if lane == 0 { packed[base0 as usize] } else { 0 };
            // u32[1..4] = 12 bytes of 6-bit packed scales
            let ps0_0 = if sub_l == 0 { packed[(base0 + 1) as usize] } else { 0 };
            let ps1_0 = if sub_l == 0 { packed[(base0 + 2) as usize] } else { 0 };
            let ps2_0 = if sub_l == 0 { packed[(base0 + 3) as usize] } else { 0 };
            // u32[4+mw*8+sub_l] = nibble word (j-group mw, lane sub_l)
            let qs0   = packed[(base0 + 4 + mw * 8 + sub_l) as usize];
            let q8b0_lo = blk * 8 + mw * 2;
            let q8b0_hi = q8b0_lo + 1;
            let u0_lo = x_qs[(q8b0_lo * 8 + sub_l) as usize];
            let u0_hi = x_qs[(q8b0_hi * 8 + sub_l) as usize];

            let dd1 = if lane == 0 { packed[base1 as usize] } else { 0 };
            let ps0_1 = if sub_l == 0 { packed[(base1 + 1) as usize] } else { 0 };
            let ps1_1 = if sub_l == 0 { packed[(base1 + 2) as usize] } else { 0 };
            let ps2_1 = if sub_l == 0 { packed[(base1 + 3) as usize] } else { 0 };
            let qs1   = packed[(base1 + 4 + mw * 8 + sub_l) as usize];
            let q8b1_lo = (blk + 1) * 8 + mw * 2;
            let q8b1_hi = q8b1_lo + 1;
            let u1_lo = x_qs[(q8b1_lo * 8 + sub_l) as usize];
            let u1_hi = x_qs[(q8b1_hi * 8 + sub_l) as usize];

            // -- Phase 2: Compute block 0 ------------------------------------
            let d0    = f16_to_f32(warp::shuffle(dd0, 0) & 0xFFFF);
            let dmin0 = f16_to_f32((warp::shuffle(dd0, 0) >> 16) & 0xFFFF);
            // Broadcast ps0/ps1/ps2 for block 0 to all sub_l within mini-warp
            let ps0_0_bc = warp::shuffle_sync(mini_mask, ps0_0, base_lane);
            let ps1_0_bc = warp::shuffle_sync(mini_mask, ps1_0, base_lane);
            let ps2_0_bc = warp::shuffle_sync(mini_mask, ps2_0, base_lane);
            // Inline 6-bit scale unpack for j-group mw
            let (sc_lo_0, sc_hi_0, m_lo_0, m_hi_0) =
                unpack_scales_6bit(ps0_0_bc, ps1_0_bc, ps2_0_bc, mw);
            let d8_0_lo = f16_to_f32(x_ds[q8b0_lo as usize] & 0xFFFF);
            let d8_0_hi = f16_to_f32(x_ds[q8b0_hi as usize] & 0xFFFF);
            let v0_lo = qs0 & 0x0F0F0F0F;
            let v0_hi = (qs0 >> 4) & 0x0F0F0F0F;
            acc += d8_0_lo * (d0 * sc_lo_0 * (dp4a_s32(v0_lo, u0_lo, 0) as f32)
                             - dmin0 * m_lo_0 * (dp4a_s32(0x01010101, u0_lo, 0) as f32))
                 + d8_0_hi * (d0 * sc_hi_0 * (dp4a_s32(v0_hi, u0_hi, 0) as f32)
                             - dmin0 * m_hi_0 * (dp4a_s32(0x01010101, u0_hi, 0) as f32));

            // -- Phase 3: Compute block 1 ------------------------------------
            let d1    = f16_to_f32(warp::shuffle(dd1, 0) & 0xFFFF);
            let dmin1 = f16_to_f32((warp::shuffle(dd1, 0) >> 16) & 0xFFFF);
            let ps0_1_bc = warp::shuffle_sync(mini_mask, ps0_1, base_lane);
            let ps1_1_bc = warp::shuffle_sync(mini_mask, ps1_1, base_lane);
            let ps2_1_bc = warp::shuffle_sync(mini_mask, ps2_1, base_lane);
            let (sc_lo_1, sc_hi_1, m_lo_1, m_hi_1) =
                unpack_scales_6bit(ps0_1_bc, ps1_1_bc, ps2_1_bc, mw);
            let d8_1_lo = f16_to_f32(x_ds[q8b1_lo as usize] & 0xFFFF);
            let d8_1_hi = f16_to_f32(x_ds[q8b1_hi as usize] & 0xFFFF);
            let v1_lo = qs1 & 0x0F0F0F0F;
            let v1_hi = (qs1 >> 4) & 0x0F0F0F0F;
            acc += d8_1_lo * (d1 * sc_lo_1 * (dp4a_s32(v1_lo, u1_lo, 0) as f32)
                             - dmin1 * m_lo_1 * (dp4a_s32(0x01010101, u1_lo, 0) as f32))
                 + d8_1_hi * (d1 * sc_hi_1 * (dp4a_s32(v1_hi, u1_hi, 0) as f32)
                             - dmin1 * m_hi_1 * (dp4a_s32(0x01010101, u1_hi, 0) as f32));

            blk += WARPS_PER_BLOCK * 2;
        }

        // Tail single-block
        while blk < k_blocks {
            let base = (n_out * k_blocks + blk) * 36;
            let dd    = if lane == 0 { packed[base as usize] } else { 0 };
            let ps0_v = if sub_l == 0 { packed[(base + 1) as usize] } else { 0 };
            let ps1_v = if sub_l == 0 { packed[(base + 2) as usize] } else { 0 };
            let ps2_v = if sub_l == 0 { packed[(base + 3) as usize] } else { 0 };
            let qs    = packed[(base + 4 + mw * 8 + sub_l) as usize];
            let q8_lo = blk * 8 + mw * 2;
            let q8_hi = q8_lo + 1;
            let u_lo  = x_qs[(q8_lo * 8 + sub_l) as usize];
            let u_hi  = x_qs[(q8_hi * 8 + sub_l) as usize];
            let d    = f16_to_f32(warp::shuffle(dd, 0) & 0xFFFF);
            let dmin = f16_to_f32((warp::shuffle(dd, 0) >> 16) & 0xFFFF);
            let ps0_bc = warp::shuffle_sync(mini_mask, ps0_v, base_lane);
            let ps1_bc = warp::shuffle_sync(mini_mask, ps1_v, base_lane);
            let ps2_bc = warp::shuffle_sync(mini_mask, ps2_v, base_lane);
            let (sc_lo, sc_hi, m_lo, m_hi) = unpack_scales_6bit(ps0_bc, ps1_bc, ps2_bc, mw);
            let d8_lo = f16_to_f32(x_ds[q8_lo as usize] & 0xFFFF);
            let d8_hi = f16_to_f32(x_ds[q8_hi as usize] & 0xFFFF);
            let v_lo  = qs & 0x0F0F0F0F;
            let v_hi  = (qs >> 4) & 0x0F0F0F0F;
            acc += d8_lo * (d * sc_lo * (dp4a_s32(v_lo, u_lo, 0) as f32)
                           - dmin * m_lo * (dp4a_s32(0x01010101, u_lo, 0) as f32))
                 + d8_hi * (d * sc_hi * (dp4a_s32(v_hi, u_hi, 0) as f32)
                           - dmin * m_hi * (dp4a_s32(0x01010101, u_hi, 0) as f32));
            blk += WARPS_PER_BLOCK * 2;
        }

        // Same reduction as v3
        acc += warp::shuffle_xor_f32_sync(mini_mask, acc, 4);
        acc += warp::shuffle_xor_f32_sync(mini_mask, acc, 2);
        acc += warp::shuffle_xor_f32_sync(mini_mask, acc, 1);
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        if sub_l == 0 {
            unsafe { *smem.add((warp_id * 4 + mw) as usize) = acc; }
        }
        thread::sync_threads();
        if warp_id == 0 && lane < 4 {
            let j = lane;
            let mut total = 0.0_f32;
            let mut w = 0u32;
            while w < WARPS_PER_BLOCK { total += unsafe { *smem.add((w * 4 + j) as usize) }; w += 1; }
            unsafe { *smem.add(j as usize) = total; }
        }
        thread::sync_threads();
        if warp_id == 0 && lane == 0 {
            let total = unsafe { *smem.add(0) + *smem.add(1) + *smem.add(2) + *smem.add(3) };
            unsafe { *out.get_unchecked_mut(n_out as usize) = total; }
        }
    }

    /// Q4_K GEMV with Q8_1-quantized activations and proper 4-element dp4a.
    ///
    /// Mini-warp structure: the 32-lane warp is divided into 4 mini-warps of 8 lanes.
    /// Each mini-warp handles one j-group (64 elements = 32 lo + 32 hi).
    /// Each sub-lane within a mini-warp processes 4 consecutive same-scale elements,
    /// enabling proper 4-element dp4a packing.
    ///
    /// Mini-warp mw (lanes 8*mw .. 8*mw+7) -> j-group mw:
    ///   sub_l 0..7: each loads one 4-nibble u32 and one Q8_1 u32, calls dp4a_s32.
    ///   After K loop: 3-step XOR reduce within mini-warp (all 8 lanes get the sum).
    ///   Cross-mini-warp reduce via shared memory -> write output.
    ///
    /// Grid:  ceil(N / WARPS_PER_BLOCK) blocks (one output neuron per warp).
    /// Block: WARPS_PER_BLOCK * 32 threads.
    /// Shared memory: WARPS_PER_BLOCK * 4 * sizeof(f32) = 64 bytes.
    #[kernel]
    pub fn q4k_gemv_decode_q8(
        mut out: DisjointSlice<f32>,
        x_qs: &[u32],    // [K/4] packed i8 activations (4 per u32)
        x_ds: &[u32],    // [K/32] Q8_1 f16 scales as u16 in low bits
        packed: &[u32],  // [N, K_BLOCKS, 42] Q4K packed weights
        n: u32,
        k_blocks: u32,
    ) {
        let tid     = thread::threadIdx_x();
        let bid     = thread::blockIdx_x();
        let warp_id = tid / 32;
        let lane    = tid % 32;
        let n_out   = bid * WARPS_PER_BLOCK + warp_id;
        if n_out >= n { return; }

        // Mini-warp decomposition: 4 groups of 8 lanes, one per j-group.
        let mw        = lane / 8;      // 0..3 -- j-group this lane handles
        let sub_l     = lane % 8;      // 0..7 -- position within mini-warp
        let mini_mask = 0xFF_u32 << (mw * 8);  // participation mask for this mini-warp
        let base_lane = mw * 8;        // global lane ID of sub_l=0 in this mini-warp

        let mut acc = 0.0_f32;

        let mut blk = 0u32;
        while blk < k_blocks {
            let base = (n_out * k_blocks + blk) * 42;

            // d and dmin: lane 0 (sub_l=0 of mw=0) loads, broadcasts to all 32 lanes.
            let d_raw    = if lane == 0 { packed[base as usize] } else { 0 };
            let dmin_raw = if lane == 0 { packed[(base + 1) as usize] } else { 0 };
            let d    = f32::from_bits(warp::shuffle(d_raw, 0));
            let dmin = f32::from_bits(warp::shuffle(dmin_raw, 0));

            // Scale pair for j-group mw: sub_l==0 of each mini-warp loads and broadcasts.
            let sc_raw = if sub_l == 0 { packed[(base + 2 + mw) as usize] } else { 0 };
            let m_raw  = if sub_l == 0 { packed[(base + 6 + mw) as usize] } else { 0 };
            let sc_raw_bc = warp::shuffle_sync(mini_mask, sc_raw, base_lane);
            let m_raw_bc  = warp::shuffle_sync(mini_mask, m_raw,  base_lane);
            let sc_lo = (sc_raw_bc & 0xFF) as f32;
            let sc_hi = ((sc_raw_bc >> 8) & 0xFF) as f32;
            let m_lo  = (m_raw_bc & 0xFF) as f32;
            let m_hi  = ((m_raw_bc >> 8) & 0xFF) as f32;

            // Q8_1 blocks for this j-group and K-block:
            //   j-group mw covers Q4K elements [mw*64 .. mw*64+63] within the block,
            //   split into lo Q8_1 block (blk*8 + mw*2) and hi (blk*8 + mw*2 + 1).
            let q8_blk_lo = blk * 8 + mw * 2;
            let q8_blk_hi = blk * 8 + mw * 2 + 1;

            let d8_lo = f16_to_f32(x_ds[q8_blk_lo as usize] & 0xFFFF);
            let d8_hi = f16_to_f32(x_ds[q8_blk_hi as usize] & 0xFFFF);

            // Nibble word: covers 4 consecutive lo+hi nibble pairs for this sub-lane.
            // j-group mw nibbles start at packed word base+10+mw*8; sub_l selects which.
            let qs_word = packed[(base + 10 + mw * 8 + sub_l) as usize];
            let v_lo = qs_word & 0x0F0F0F0F;            // 4 lo nibbles as packed bytes
            let v_hi = (qs_word >> 4) & 0x0F0F0F0F;     // 4 hi nibbles as packed bytes

            // Q8_1 packed i8 values (4 per u32) for this sub-lane's 4 elements.
            let u_lo = x_qs[(q8_blk_lo * 8 + sub_l) as usize];
            let u_hi = x_qs[(q8_blk_hi * 8 + sub_l) as usize];

            // dp4a: 4 nibblexq8 products per instruction.
            // nibble 0..15 fits signed i8; q8 values are signed i8.
            let sumi_d_lo = dp4a_s32(v_lo, u_lo, 0);
            let sumi_d_hi = dp4a_s32(v_hi, u_hi, 0);
            // Sum of q8 values for the dmin*m*sum(q8) min-term correction.
            let sumi_m_lo = dp4a_s32(0x01010101, u_lo, 0);
            let sumi_m_hi = dp4a_s32(0x01010101, u_hi, 0);

            // Mini-warp XOR butterfly reduce (3 steps for 8 lanes: delta 4, 2, 1).
            // After reduce all 8 lanes in the mini-warp hold the total sum.
            let mut sd_lo = sumi_d_lo as u32;
            let mut sd_hi = sumi_d_hi as u32;
            let mut sm_lo = sumi_m_lo as u32;
            let mut sm_hi = sumi_m_hi as u32;

            sd_lo = sd_lo.wrapping_add(warp::shuffle_xor_sync(mini_mask, sd_lo, 4));
            sd_lo = sd_lo.wrapping_add(warp::shuffle_xor_sync(mini_mask, sd_lo, 2));
            sd_lo = sd_lo.wrapping_add(warp::shuffle_xor_sync(mini_mask, sd_lo, 1));
            sd_hi = sd_hi.wrapping_add(warp::shuffle_xor_sync(mini_mask, sd_hi, 4));
            sd_hi = sd_hi.wrapping_add(warp::shuffle_xor_sync(mini_mask, sd_hi, 2));
            sd_hi = sd_hi.wrapping_add(warp::shuffle_xor_sync(mini_mask, sd_hi, 1));
            sm_lo = sm_lo.wrapping_add(warp::shuffle_xor_sync(mini_mask, sm_lo, 4));
            sm_lo = sm_lo.wrapping_add(warp::shuffle_xor_sync(mini_mask, sm_lo, 2));
            sm_lo = sm_lo.wrapping_add(warp::shuffle_xor_sync(mini_mask, sm_lo, 1));
            sm_hi = sm_hi.wrapping_add(warp::shuffle_xor_sync(mini_mask, sm_hi, 4));
            sm_hi = sm_hi.wrapping_add(warp::shuffle_xor_sync(mini_mask, sm_hi, 2));
            sm_hi = sm_hi.wrapping_add(warp::shuffle_xor_sync(mini_mask, sm_hi, 1));

            // All 8 lanes now hold the same reduced sums. Scale and accumulate.
            acc += d8_lo * (d * sc_lo * (sd_lo as i32 as f32) - dmin * m_lo * (sm_lo as i32 as f32))
                 + d8_hi * (d * sc_hi * (sd_hi as i32 as f32) - dmin * m_hi * (sm_hi as i32 as f32));

            blk += 1;
        }

        // Cross-mini-warp reduce: collect one f32 per mini-warp via shared memory,
        // then lane 0 of each warp sums its 4 j-group contributions and writes output.
        // smem layout: slot = warp_id * 4 + mw  (WARPS_PER_BLOCK * 4 f32 slots total).
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        if sub_l == 0 {
            unsafe { *smem.add((warp_id * 4 + mw) as usize) = acc; }
        }
        thread::sync_threads();

        if sub_l == 0 && mw == 0 {
            let base_slot = (warp_id * 4) as usize;
            let total = unsafe {
                *smem.add(base_slot + 0) + *smem.add(base_slot + 1)
                + *smem.add(base_slot + 2) + *smem.add(base_slot + 3)
            };
            unsafe { *out.get_unchecked_mut(n_out as usize) = total; }
        }
    }
}

// -- Host-side correctness test ------------------------------------------------

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();

    // Small problem: N=32 outputs, K_BLOCKS=1 (K=256).
    let n: usize = 32;
    let k_blocks: usize = 1;
    let k = k_blocks * 256;

    // Activation: x[i] = 1.0.
    let x_host: Vec<f32> = vec![1.0_f32; k];

    // Build packed weights:
    //   d = 1.0, dmin = 0.0
    //   scales: all groups sc[g] = 1 (u8), m[g] = 0 (u8)
    //   qs: all nibbles = 3 -> packed byte = 0x33
    // Expected weight per element: 1.0 * 1.0 * 3 - 0.0 * 0.0 = 3.0
    // Expected output: sum over 256 elements of 3.0 * 1.0 = 768.0
    let packed_u32s = n * k_blocks * 42;
    let mut packed_host = vec![0u32; packed_u32s];
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 42;
            packed_host[base]     = 1.0_f32.to_bits();
            packed_host[base + 1] = 0.0_f32.to_bits();
            // sc[g]=1, m[g]=0 for all 8 groups (stored as pairs of bytes).
            // u32[2+j] = 1 | (1<<8) = 0x00000101
            // u32[6+j] = 0
            for j in 0..4usize {
                packed_host[base + 2 + j] = 0x00000101_u32;
                packed_host[base + 6 + j] = 0x00000000_u32;
            }
            // nibble=3: lo=3, hi=3, packed byte = 0x33 -> u32 = 0x33333333
            for w in 0..32usize {
                packed_host[base + 10 + w] = 0x33333333_u32;
            }
        }
    }

    let x_dev        = DeviceBuffer::from_host(&stream, &x_host).expect("upload x");
    let mut p_dev    = DeviceBuffer::from_host(&stream, &packed_host).expect("upload packed");
    let mut out_dev  = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc out");
    let mut out_dev2 = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc out2");

    let module = kernels::load(&ctx).expect("load oxide q4k module");

    let wpb = WARPS_PER_BLOCK as usize;
    // v1 (fallback): ceil(N/WARPS_PER_BLOCK) blocks, no shared memory.
    let grid_v1 = (n + wpb - 1) / wpb;
    let cfg = LaunchConfig {
        grid_dim:  (grid_v1 as u32, 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    // v2/q8 (multi-warp): N blocks (one per output neuron), smem for cross-warp reduce.
    let cfg_v2 = LaunchConfig {
        grid_dim:  (n as u32, 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        // Need 32 f32 slots (128 bytes): the reduction reads smem[lane] for all
        // 32 lanes of warp 0 on Blackwell -- predicated but LDSMEM still executes
        // for lanes 4..31 and traps if smem is smaller than 32 f32s.
        shared_mem_bytes: 32 * 4,
    };

    // -- Test 1: v1 and v2 both pass for nibble=3,sc=1,d=1,dmin=0 --------------
    unsafe {
        module.q4k_gemv_decode(
            stream.as_ref(), cfg, &mut out_dev, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v1 launch");

    unsafe {
        module.q4k_gemv_dp4a_base(
            stream.as_ref(), cfg_v2, &mut out_dev2, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v2 launch");

    let result1 = out_dev.to_host_vec(&stream).expect("download v1");
    let result2 = out_dev2.to_host_vec(&stream).expect("download v2");
    let expected = 768.0_f32;

    let max_err1 = result1.iter().map(|v| (v - expected).abs()).fold(0.0_f32, f32::max);
    let max_err2 = result2.iter().map(|v| (v - expected).abs()).fold(0.0_f32, f32::max);

    if max_err1 < 1.0 {
        println!(
            "oxide-gemv-q4k v1 test 1 (nibble=3,sc=1,d=1,dmin=0): PASSED (expected={}, max_err={:.2e})",
            expected, max_err1
        );
    } else {
        eprintln!(
            "oxide-gemv-q4k v1 test 1 FAILED (expected={}, max_err={:.4}, result[0]={})",
            expected, max_err1, result1[0]
        );
        std::process::exit(1);
    }

    if max_err2 < 1.0 {
        println!(
            "oxide-gemv-q4k v2 test 1 (nibble=3,sc=1,d=1,dmin=0): PASSED (expected={}, max_err={:.2e})",
            expected, max_err2
        );
    } else {
        eprintln!(
            "oxide-gemv-q4k v2 test 1 FAILED (expected={}, max_err={:.4}, result[0]={})",
            expected, max_err2, result2[0]
        );
        std::process::exit(1);
    }

    let diff12 = result1.iter().zip(result2.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f32, f32::max);
    if diff12 < 1.0 {
        println!("oxide-gemv-q4k v1/v2 test 1 agreement: PASSED (max_diff={:.2e})", diff12);
    } else {
        eprintln!("oxide-gemv-q4k v1/v2 test 1 DISAGREEMENT (max_diff={:.4})", diff12);
        std::process::exit(1);
    }

    // -- Test 2: d=2.0, sc=3, nibble=4, dmin=0.5, m=2 -------------------------
    // weight = 2.0 * 3.0 * 4.0 - 0.5 * 2.0 = 24.0 - 1.0 = 23.0
    // sum over 256 elements = 256 * 23.0 = 5888.0
    let mut packed_host2 = vec![0u32; packed_u32s];
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 42;
            packed_host2[base]     = 2.0_f32.to_bits();
            packed_host2[base + 1] = 0.5_f32.to_bits();
            for j in 0..4usize {
                packed_host2[base + 2 + j] = 0x00000303_u32;
                packed_host2[base + 6 + j] = 0x00000202_u32;
            }
            for w in 0..32usize {
                packed_host2[base + 10 + w] = 0x44444444_u32;
            }
        }
    }
    p_dev.copy_from_host(&stream, &packed_host2).expect("re-upload");

    let zeros = vec![0.0_f32; n];
    out_dev.copy_from_host(&stream, &zeros).expect("zero out");
    out_dev2.copy_from_host(&stream, &zeros).expect("zero out2");

    unsafe {
        module.q4k_gemv_decode(
            stream.as_ref(), cfg, &mut out_dev, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v1 launch 2");

    unsafe {
        module.q4k_gemv_dp4a_base(
            stream.as_ref(), cfg_v2, &mut out_dev2, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v2 launch 2");

    let result1b = out_dev.to_host_vec(&stream).expect("download v1 t2");
    let result2b = out_dev2.to_host_vec(&stream).expect("download v2 t2");
    let expected2 = 5888.0_f32;

    let max_err1b = result1b.iter().map(|v| (v - expected2).abs()).fold(0.0_f32, f32::max);
    let max_err2b = result2b.iter().map(|v| (v - expected2).abs()).fold(0.0_f32, f32::max);

    if max_err1b < 1.0 {
        println!(
            "oxide-gemv-q4k v1 test 2 (d=2,sc=3,nibble=4,dmin=0.5,m=2): PASSED (expected={}, max_err={:.2e})",
            expected2, max_err1b
        );
    } else {
        eprintln!(
            "oxide-gemv-q4k v1 test 2 FAILED (expected={}, max_err={:.4}, result[0]={})",
            expected2, max_err1b, result1b[0]
        );
        std::process::exit(1);
    }

    if max_err2b < 1.0 {
        println!(
            "oxide-gemv-q4k v2 test 2 (d=2,sc=3,nibble=4,dmin=0.5,m=2): PASSED (expected={}, max_err={:.2e})",
            expected2, max_err2b
        );
    } else {
        eprintln!(
            "oxide-gemv-q4k v2 test 2 FAILED (expected={}, max_err={:.4}, result[0]={})",
            expected2, max_err2b, result2b[0]
        );
        std::process::exit(1);
    }

    let diff12b = result1b.iter().zip(result2b.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f32, f32::max);
    if diff12b < 1.0 {
        println!("oxide-gemv-q4k v1/v2 test 2 agreement: PASSED (max_diff={:.2e})", diff12b);
    } else {
        eprintln!("oxide-gemv-q4k v1/v2 test 2 DISAGREEMENT (max_diff={:.4})", diff12b);
        std::process::exit(1);
    }

    // -- Test 3: multi-block (K_BLOCKS=4) --------------------------------------
    // weight = 1.0 * 2.0 * 5.0 - 0.0 = 10.0, sum = 256*4 * 10.0 = 10240.0
    let k_blocks3: usize = 4;
    let k3 = k_blocks3 * 256;
    let x3: Vec<f32> = vec![1.0_f32; k3];
    let packed_u32s3 = n * k_blocks3 * 42;
    let mut ph3 = vec![0u32; packed_u32s3];
    for row in 0..n {
        for blk in 0..k_blocks3 {
            let base = (row * k_blocks3 + blk) * 42;
            ph3[base]     = 1.0_f32.to_bits();
            ph3[base + 1] = 0.0_f32.to_bits();
            for j in 0..4usize {
                ph3[base + 2 + j] = 0x00000202_u32;
                ph3[base + 6 + j] = 0x00000000_u32;
            }
            for w in 0..32usize {
                ph3[base + 10 + w] = 0x55555555_u32;
            }
        }
    }
    let expected3 = 10240.0_f32;

    let x_dev3    = DeviceBuffer::from_host(&stream, &x3).expect("upload x3");
    let p_dev3    = DeviceBuffer::from_host(&stream, &ph3).expect("upload p3");
    let mut od1_3 = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc od1_3");
    let mut od2_3 = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc od2_3");

    let grid3_v1 = (n + wpb - 1) / wpb;
    let cfg3 = LaunchConfig {
        grid_dim:  (grid3_v1 as u32, 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg3_v2 = LaunchConfig {
        grid_dim:  (n as u32, 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: WARPS_PER_BLOCK * 4,
    };

    unsafe {
        module.q4k_gemv_decode(
            stream.as_ref(), cfg3, &mut od1_3, &x_dev3, &p_dev3, n as u32, k_blocks3 as u32,
        )
    }.expect("v1 launch 3");

    unsafe {
        module.q4k_gemv_dp4a_base(
            stream.as_ref(), cfg3_v2, &mut od2_3, &x_dev3, &p_dev3, n as u32, k_blocks3 as u32,
        )
    }.expect("v2 launch 3");

    let r1_3 = od1_3.to_host_vec(&stream).expect("download v1 t3");
    let r2_3 = od2_3.to_host_vec(&stream).expect("download v2 t3");

    let e1_3 = r1_3.iter().map(|v| (v - expected3).abs()).fold(0.0_f32, f32::max);
    let e2_3 = r2_3.iter().map(|v| (v - expected3).abs()).fold(0.0_f32, f32::max);
    let d3   = r1_3.iter().zip(r2_3.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f32, f32::max);

    if e1_3 < 1.0 {
        println!("oxide-gemv-q4k v1 test 3 (multi-block): PASSED (expected={}, max_err={:.2e})", expected3, e1_3);
    } else {
        eprintln!("oxide-gemv-q4k v1 test 3 FAILED (max_err={:.4}, result[0]={})", e1_3, r1_3[0]);
        std::process::exit(1);
    }
    if e2_3 < 1.0 {
        println!("oxide-gemv-q4k v2 test 3 (multi-block): PASSED (expected={}, max_err={:.2e})", expected3, e2_3);
    } else {
        eprintln!("oxide-gemv-q4k v2 test 3 FAILED (max_err={:.4}, result[0]={})", e2_3, r2_3[0]);
        std::process::exit(1);
    }
    if d3 < 1.0 {
        println!("oxide-gemv-q4k v1/v2 test 3 agreement: PASSED (max_diff={:.2e})", d3);
    } else {
        eprintln!("oxide-gemv-q4k v1/v2 test 3 DISAGREEMENT (max_diff={:.4})", d3);
        std::process::exit(1);
    }

    // -- v3r correctness test: isolated raw GGUF 36-u32 vs packed 42-u32 ------
    // Creates data directly without going through the engine upload path.
    // d=1.0, dmin=0.0, sc[i]=1, m[i]=0, nibbles=3, Q8_1 q=1 d8=1.0
    // Expected per-neuron output: 768.0 (same as existing v1/v2 test 1)
    {
        let n: usize = 32;
        let k_blocks: usize = 1;

        let k_q8_blocks = k_blocks * 256 / 32;  // = 8
        // Q8_1: all i8 = 1 packed as 0x01010101, scale f16(1.0) = 0x3C00
        let q8_qs_v: Vec<u32> = vec![0x01010101u32; k_q8_blocks * 8];
        let q8_ds_v: Vec<u32> = vec![0x00003C00u32; k_q8_blocks];

        // Packed 42-u32 for v3
        let packed_v3_h: Vec<u32> = {
            let mut p = vec![0u32; n * k_blocks * 42];
            for row in 0..n {
                let base = (row * k_blocks) * 42;
                p[base]   = 1.0f32.to_bits();
                p[base+1] = 0.0f32.to_bits();
                for j in 0..4usize { p[base+2+j] = 0x00000101; }
                for j in 0..4usize { p[base+6+j] = 0x00000000; }
                for w in 0..32usize { p[base+10+w] = 0x33333333; }
            }
            p
        };

        // Raw GGUF 36-u32 for v3r
        // ps[0..4]=[1,1,1,1] -> sc[0..3]=1; ps[4..8]=[0,0,0,0] -> m[0..3]=0;
        // ps[8..12]=[1,1,1,1] -> sc[4..7]=1, m[4..7]=0 (via 6-bit unpack)
        let packed_v3r_h: Vec<u32> = {
            let mut p = vec![0u32; n * k_blocks * 36];
            for row in 0..n {
                let base = (row * k_blocks) * 36;
                p[base]   = 0x00003C00u32;  // f16(1.0) | (f16(0.0) << 16)
                p[base+1] = 0x01010101u32;  // ps[0..4] = [1,1,1,1]
                p[base+2] = 0x00000000u32;  // ps[4..8] = [0,0,0,0]
                p[base+3] = 0x01010101u32;  // ps[8..12] = [1,1,1,1]
                for w in 0..32usize { p[base+4+w] = 0x33333333; }
            }
            p
        };

        let x_qs_dev  = DeviceBuffer::from_host(&stream, &q8_qs_v).expect("q8_qs upload");
        let x_ds_dev  = DeviceBuffer::from_host(&stream, &q8_ds_v).expect("q8_ds upload");
        let pv3_dev   = DeviceBuffer::from_host(&stream, &packed_v3_h).expect("v3 upload");
        let pv3r_dev  = DeviceBuffer::from_host(&stream, &packed_v3r_h).expect("v3r upload");
        let mut out_v3  = DeviceBuffer::<f32>::zeroed(&stream, n).expect("out_v3");
        let mut out_v3r = DeviceBuffer::<f32>::zeroed(&stream, n).expect("out_v3r");

        let smem_v3 = WARPS_PER_BLOCK * 4 * 4;
        let cfg_v3 = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: smem_v3,
        };
        unsafe {
            module.q4k_gemv_dp4a_q8(
                stream.as_ref(), cfg_v3, &mut out_v3,
                &x_qs_dev, &x_ds_dev, &pv3_dev, n as u32, k_blocks as u32,
            )
        }.expect("v3 launch");
        unsafe {
            module.q4k_gemv_dp4a_q8_raw(
                stream.as_ref(), cfg_v3, &mut out_v3r,
                &x_qs_dev, &x_ds_dev, &pv3r_dev, n as u32, k_blocks as u32,
            )
        }.expect("v3r launch");

        let r_v3  = out_v3.to_host_vec(&stream).expect("v3 download");
        let r_v3r = out_v3r.to_host_vec(&stream).expect("v3r download");

        let expected_v3r = 768.0f32;
        let e_v3  = r_v3.iter().map(|v| (v - expected_v3r).abs()).fold(0.0f32, f32::max);
        let e_v3r = r_v3r.iter().map(|v| (v - expected_v3r).abs()).fold(0.0f32, f32::max);
        let diff  = r_v3.iter().zip(r_v3r.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);

        println!("v3  vs expected={}: err={:.4}  first={}", expected_v3r, e_v3,  r_v3[0]);
        println!("v3r vs expected={}: err={:.4}  first={}", expected_v3r, e_v3r, r_v3r[0]);
        println!("v3 vs v3r max_diff={:.6}", diff);

        if e_v3 > 1.0 {
            eprintln!("v3 FAILED (expected={}, max_err={:.4}, first={})", expected_v3r, e_v3, r_v3[0]);
            std::process::exit(1);
        }
        if e_v3r > 1.0 {
            eprintln!("v3r FAILED (expected={}, max_err={:.4}, first={})", expected_v3r, e_v3r, r_v3r[0]);
            eprintln!("  Bug is in the KERNEL (not engine upload): v3r mismatches on direct data");
            eprintln!("  v3 first 8:  {:?}", &r_v3[..8]);
            eprintln!("  v3r first 8: {:?}", &r_v3r[..8]);
            std::process::exit(1);
        }
        if diff > 1.0 {
            eprintln!("v3 vs v3r MISMATCH (max_diff={:.4})", diff);
            eprintln!("  v3 first 8:  {:?}", &r_v3[..8]);
            eprintln!("  v3r first 8: {:?}", &r_v3r[..8]);
            std::process::exit(1);
        }
        println!("oxide-gemv-q4k v3 vs v3r agreement: PASSED (max_diff={:.2e})", diff);
    }

    // -- v3r paired-loop test: k_blocks=16 exercises the double-block paired loop --
    // k_blocks=1 only tests the tail. Real models use k_blocks=16 (4096/256).
    // d=1.0, dmin=0.0, sc[i]=1, m[i]=0, nibbles=3, Q8_1 q=1 d8=1.0
    // Expected: 16 blocks x 256 elements x 3.0 = 12288.0 per neuron
    {
        let n: usize = 32;
        let k_blocks: usize = 16;
        let k = k_blocks * 256;
        let k_q8_blocks = k / 32;

        let q8_qs_v: Vec<u32> = vec![0x01010101u32; k_q8_blocks * 8];
        let q8_ds_v: Vec<u32> = vec![0x00003C00u32; k_q8_blocks];

        // Packed 42-u32 for v3 reference
        let packed_v3_h: Vec<u32> = {
            let mut p = vec![0u32; n * k_blocks * 42];
            for row in 0..n {
                for blk in 0..k_blocks {
                    let base = (row * k_blocks + blk) * 42;
                    p[base]   = 1.0f32.to_bits();
                    p[base+1] = 0.0f32.to_bits();
                    for j in 0..4usize { p[base+2+j] = 0x00000101; }
                    for j in 0..4usize { p[base+6+j] = 0x00000000; }
                    for w in 0..32usize { p[base+10+w] = 0x33333333; }
                }
            }
            p
        };

        // Raw GGUF 36-u32 for v3r
        let packed_v3r_h: Vec<u32> = {
            let mut p = vec![0u32; n * k_blocks * 36];
            for row in 0..n {
                for blk in 0..k_blocks {
                    let base = (row * k_blocks + blk) * 36;
                    p[base]   = 0x00003C00u32;  // f16(1.0) | (f16(0.0) << 16)
                    p[base+1] = 0x01010101u32;  // ps[0..4]=[1,1,1,1] -> sc[0..3]=1
                    p[base+2] = 0x00000000u32;  // ps[4..8]=[0,0,0,0] -> m[0..3]=0
                    p[base+3] = 0x01010101u32;  // ps[8..12]=[1,1,1,1] -> sc[4..7]=1, m[4..7]=0
                    for w in 0..32usize { p[base+4+w] = 0x33333333; }
                }
            }
            p
        };

        let x_qs_dev  = DeviceBuffer::from_host(&stream, &q8_qs_v).expect("q8_qs upload kb16");
        let x_ds_dev  = DeviceBuffer::from_host(&stream, &q8_ds_v).expect("q8_ds upload kb16");
        let pv3_dev   = DeviceBuffer::from_host(&stream, &packed_v3_h).expect("v3 upload kb16");
        let pv3r_dev  = DeviceBuffer::from_host(&stream, &packed_v3r_h).expect("v3r upload kb16");
        let mut out_v3  = DeviceBuffer::<f32>::zeroed(&stream, n).expect("out_v3 kb16");
        let mut out_v3r = DeviceBuffer::<f32>::zeroed(&stream, n).expect("out_v3r kb16");

        let smem_bytes = WARPS_PER_BLOCK * 4 * 4;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        unsafe {
            module.q4k_gemv_dp4a_q8(
                stream.as_ref(), cfg, &mut out_v3,
                &x_qs_dev, &x_ds_dev, &pv3_dev, n as u32, k_blocks as u32,
            )
        }.expect("v3 launch kb16");
        unsafe {
            module.q4k_gemv_dp4a_q8_raw(
                stream.as_ref(), cfg, &mut out_v3r,
                &x_qs_dev, &x_ds_dev, &pv3r_dev, n as u32, k_blocks as u32,
            )
        }.expect("v3r launch kb16");

        let r_v3  = out_v3.to_host_vec(&stream).expect("v3 download kb16");
        let r_v3r = out_v3r.to_host_vec(&stream).expect("v3r download kb16");

        let expected = (k_blocks * 256) as f32 * 3.0;  // 12288.0 for k_blocks=16
        let e_v3  = r_v3.iter().map(|v| (v - expected).abs()).fold(0.0f32, f32::max);
        let e_v3r = r_v3r.iter().map(|v| (v - expected).abs()).fold(0.0f32, f32::max);
        let diff  = r_v3.iter().zip(r_v3r.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);

        println!("v3  k_blocks=16 vs expected={}: err={:.4}  first={}", expected, e_v3,  r_v3[0]);
        println!("v3r k_blocks=16 vs expected={}: err={:.4}  first={}", expected, e_v3r, r_v3r[0]);
        println!("v3 vs v3r k_blocks=16 max_diff={:.6}", diff);

        if e_v3 > 1.0 {
            eprintln!("v3 k_blocks=16 FAILED (expected={}, max_err={:.4}, first={})", expected, e_v3, r_v3[0]);
            std::process::exit(1);
        }
        if e_v3r > 1.0 {
            eprintln!("v3r k_blocks=16 FAILED (expected={}, max_err={:.4}, first={})", expected, e_v3r, r_v3r[0]);
            eprintln!("  Bug is in the paired LOOP of v3r kernel (not the tail loop)");
            eprintln!("  v3  first 8: {:?}", &r_v3[..8]);
            eprintln!("  v3r first 8: {:?}", &r_v3r[..8]);
            std::process::exit(1);
        }
        if diff > 1.0 {
            eprintln!("v3 vs v3r k_blocks=16 MISMATCH (max_diff={:.4})", diff);
            eprintln!("  v3  first 8: {:?}", &r_v3[..8]);
            eprintln!("  v3r first 8: {:?}", &r_v3r[..8]);
            std::process::exit(1);
        }
        println!("oxide-gemv-q4k v3 vs v3r k_blocks=16 (paired loop): PASSED (max_diff={:.2e})", diff);
    }

    // -- fused gate+up SiLU test: k_blocks=16 --------------------------------
    // Both gate and up weights: d=1.0, dmin=0.0, sc=1, m=0, nibbles=3 -> GEMV=12288.0
    // SiLU(12288) ~= 12288 (sigmoid(large positive) ~= 1.0)
    // Expected output ~= 12288 * 1.0 * 12288 = 150994944.0
    {
        let n: usize = 32;
        let k_blocks: usize = 16;
        let k = k_blocks * 256;
        let k_q8_blocks = k / 32;

        let q8_qs_v: Vec<u32> = vec![0x01010101u32; k_q8_blocks * 8];
        let q8_ds_v: Vec<u32> = vec![0x00003C00u32; k_q8_blocks];  // f16(1.0)

        let packed_gate_h: Vec<u32> = {
            let mut p = vec![0u32; n * k_blocks * 42];
            for row in 0..n {
                for blk in 0..k_blocks {
                    let base = (row * k_blocks + blk) * 42;
                    p[base]   = 1.0f32.to_bits();       // d = 1.0
                    p[base+1] = 0.0f32.to_bits();       // dmin = 0.0
                    for j in 0..4usize { p[base+2+j] = 0x00000101; }  // sc=1 for all groups
                    for j in 0..4usize { p[base+6+j] = 0x00000000; }  // m=0
                    for w in 0..32usize { p[base+10+w] = 0x33333333; } // nibbles=3
                }
            }
            p
        };
        // up weights identical to gate for this test
        let packed_up_h = packed_gate_h.clone();

        let x_qs_dev  = DeviceBuffer::from_host(&stream, &q8_qs_v).expect("fused q8_qs");
        let x_ds_dev  = DeviceBuffer::from_host(&stream, &q8_ds_v).expect("fused q8_ds");
        let gate_dev  = DeviceBuffer::from_host(&stream, &packed_gate_h).expect("fused gate");
        let up_dev    = DeviceBuffer::from_host(&stream, &packed_up_h).expect("fused up");
        let mut out_fused = DeviceBuffer::<f32>::zeroed(&stream, n).expect("fused out");

        let smem_fused = WARPS_PER_BLOCK * 4 * 4 * 2;  // 32 f32 = 128 bytes
        let cfg_fused = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: smem_fused,
        };
        unsafe {
            module.q4k_fused_silu_dp4a(
                stream.as_ref(), cfg_fused, &mut out_fused,
                &gate_dev, &up_dev, &x_qs_dev, &x_ds_dev,
                n as u32, k_blocks as u32,
            )
        }.expect("fused silu launch");

        let r_fused = out_fused.to_host_vec(&stream).expect("fused download");

        // Each GEMV = 16 blocks x 256 elements x 3.0 = 12288.0
        // silu(12288) * 12288: sigmoid(12288) ~= 1.0 -> expected ~= 12288^2 = 150994944
        let gemv_val = (k_blocks * 256) as f32 * 3.0;
        let sigmoid_approx = 1.0_f32 / (1.0_f32 + (-gemv_val).exp());
        let expected_fused = gemv_val * sigmoid_approx * gemv_val;
        let e_fused = r_fused.iter().map(|v| (v - expected_fused).abs()).fold(0.0f32, f32::max);

        println!("fused k_blocks=16 vs expected={:.2}: err={:.4}  first={:.2}",
                 expected_fused, e_fused, r_fused[0]);

        // Tolerance: 1.0 absolute (same as other tests) but proportionally it's
        // very tight (12288^2 ~= 1.5e8, so 1.0 is < 1e-7 relative).
        // For sigmoid(large) ~= 1.0, the rounding error is in FP32 dp4a accumulation.
        // Use 1% relative tolerance: 1.5e6.
        let tol = expected_fused * 0.01;
        if e_fused > tol {
            eprintln!("fused FAILED (expected={:.2}, max_err={:.4}, tol={:.4}, first={:.2})",
                      expected_fused, e_fused, tol, r_fused[0]);
            std::process::exit(1);
        }
        println!("oxide-gemv-q4k fused_silu_mul k_blocks=16: PASSED (max_err={:.2e})", e_fused);
    }

    // -- v3_kpar test: k_blocks=16, 8 warps per block ------------------------
    // Same inputs as the v3 k_blocks=16 test: expected=12288.0 per neuron.
    // Verifies the kpar kernel produces bit-identical results to v3.
    {
        let n: usize = 32;
        let k_blocks: usize = 16;
        let k = k_blocks * 256;
        let k_q8_blocks = k / 32;

        let q8_qs_v: Vec<u32> = vec![0x01010101u32; k_q8_blocks * 8];
        let q8_ds_v: Vec<u32> = vec![0x00003C00u32; k_q8_blocks];  // f16(1.0)

        let packed_v3_h: Vec<u32> = {
            let mut p = vec![0u32; n * k_blocks * 42];
            for row in 0..n {
                for blk in 0..k_blocks {
                    let base = (row * k_blocks + blk) * 42;
                    p[base]   = 1.0f32.to_bits();
                    p[base+1] = 0.0f32.to_bits();
                    for j in 0..4usize { p[base+2+j] = 0x00000101; }
                    for j in 0..4usize { p[base+6+j] = 0x00000000; }
                    for w in 0..32usize { p[base+10+w] = 0x33333333; }
                }
            }
            p
        };

        let x_qs_dev  = DeviceBuffer::from_host(&stream, &q8_qs_v).expect("kpar q8_qs upload");
        let x_ds_dev  = DeviceBuffer::from_host(&stream, &q8_ds_v).expect("kpar q8_ds upload");
        let pv3_dev   = DeviceBuffer::from_host(&stream, &packed_v3_h).expect("kpar v3 upload");
        let mut out_kpar = DeviceBuffer::<f32>::zeroed(&stream, n).expect("out_kpar");

        // kpar: 8 warps x 32 lanes = 256 threads/block; smem = 32 f32 = 128 bytes.
        let cfg_kpar = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (WARPS_PER_BLOCK_KPAR * 32, 1, 1),
            shared_mem_bytes: 32 * 4,  // 128 bytes (Blackwell predicated-smem safe)
        };
        unsafe {
            module.q4k_gemv_dp4a_kpar8(
                stream.as_ref(), cfg_kpar, &mut out_kpar,
                &x_qs_dev, &x_ds_dev, &pv3_dev, n as u32, k_blocks as u32,
            )
        }.expect("v3_kpar launch");

        let r_kpar = out_kpar.to_host_vec(&stream).expect("kpar download");
        let expected = (k_blocks * 256) as f32 * 3.0;  // 12288.0 for k_blocks=16
        let e_kpar = r_kpar.iter().map(|v| (v - expected).abs()).fold(0.0f32, f32::max);

        println!("v3_kpar k_blocks=16 vs expected={}: err={:.4}  first={}", expected, e_kpar, r_kpar[0]);

        if e_kpar > 1.0 {
            eprintln!("v3_kpar FAILED (expected={}, max_err={:.4}, first={})", expected, e_kpar, r_kpar[0]);
            std::process::exit(1);
        }
        println!("oxide-gemv-q4k v3_kpar k_blocks=16: PASSED (max_err={:.2e})", e_kpar);
    }

    println!("oxide-gemv-q4k: all tests PASSED");
}
