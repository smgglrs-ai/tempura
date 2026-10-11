/*
 * oxide-gemv-q4k: Hand-optimized Q4_K decode GEMV kernel via rustc-codegen-cuda.
 *
 * Build:  cargo oxide build --manifest-path tools/oxide-gemv-q4k/Cargo.toml
 * The compiled binary embeds PTX in the `.oxart` ELF section.
 * The tempura-kernels build.rs extracts that PTX and writes it to OUT_DIR.
 *
 * Q4_K packed GPU format (42 u32 per block, 256 elements per block):
 *   u32[0]      = f32 bits of d (super-scale, f16 → f32 at upload)
 *   u32[1]      = f32 bits of dmin (super-min, f16 → f32 at upload)
 *   u32[2+j]    = sc[2*j] | (sc[2*j+1]<<8) for j=0..3 (group scale pairs, 8-bit each)
 *   u32[6+j]    = m[2*j]  | (m[2*j+1]<<8)  for j=0..3 (group min pairs, 8-bit each)
 *   u32[10..42] = 32 u32s of packed qs bytes (128 bytes = 256 nibbles, 4-bit each)
 *
 * Dequant for element `within` in a block:
 *   j       = within / 64                    — which j-pair (0..3)
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
 *   Each lane handles elements: l, l+32, l+64, …, l+224 within the block.
 *   - d, dmin: lane 0 loads once; all lanes receive via shfl.
 *   - sc/m:    lanes 0..3 each load one u32 (2 scale/min values); broadcast via shfl.
 *   - qs:      each lane loads one u32 per j-group (contains lo+hi nibbles for that lane).
 *   The 4 j-groups are fully unrolled, each processing 2 elements per lane.
 */

use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, thread, warp};
// NOTE: dp4a_s32 requires the upstream oxide backend (cuda-oxide/crates/rustc-codegen-cuda)
// which includes dp4a in its intrinsic catalog. The fork backend (crates/rustc-codegen-cuda)
// does not. Build q4k_gemv_decode_q8 only when the upstream backend is available.
use cuda_device::dotprod::dp4a_s32;
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_core::simt::LaunchConfig;

const WARPS_PER_BLOCK: u32 = 4;

/// f16 bits (low 16 of u32) → f32, device-safe bit arithmetic.
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

/// f32 → f16 bits stored in low 16 of u16, device-safe.
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

#[cuda_module]
mod kernels {
    use super::*;

    /// Q4_K GEMV for M=1 decode: out[N] = dequant(packed[N, K_BLOCKS, 42]) × x[K].
    ///
    /// Grid:  (ceil(N / WARPS_PER_BLOCK),) blocks — each warp owns one output neuron.
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
    pub fn q4k_gemv_decode_v2(
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

        // Each warp covers a stride-WARPS_PER_BLOCK subset of K blocks.
        let mut blk = warp_id;
        while blk < k_blocks {
            let base = (n_out * k_blocks + blk) * 42;

            // d and dmin: lane 0 loads, all lanes receive via full-warp shfl.
            let d_raw    = if lane == 0 { packed[base as usize] } else { 0 };
            let dmin_raw = if lane == 0 { packed[(base + 1) as usize] } else { 0 };
            let d    = f32::from_bits(warp::shuffle(d_raw, 0));
            let dmin = f32::from_bits(warp::shuffle(dmin_raw, 0));

            let sc_raw = if lane < 4 { packed[(base + 2 + lane) as usize] } else { 0 };
            let m_raw  = if lane < 4 { packed[(base + 6 + lane) as usize] } else { 0 };

            // j=0
            {
                let sc_pair = warp::shuffle(sc_raw, 0);
                let m_pair  = warp::shuffle(m_raw, 0);
                let sc_lo = (sc_pair & 0xFF) as f32;
                let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                let m_lo  = (m_pair & 0xFF) as f32;
                let m_hi  = ((m_pair >> 8) & 0xFF) as f32;
                let qs_word = packed[(base + 10 + lane / 4) as usize];
                let qs_shift = (lane % 4) * 8;
                let qs_byte = (qs_word >> qs_shift) & 0xFF;
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
                let qs_word = packed[(base + 10 + 8 + lane / 4) as usize];
                let qs_shift = (lane % 4) * 8;
                let qs_byte = (qs_word >> qs_shift) & 0xFF;
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
                let qs_word = packed[(base + 10 + 16 + lane / 4) as usize];
                let qs_shift = (lane % 4) * 8;
                let qs_byte = (qs_word >> qs_shift) & 0xFF;
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
                let qs_word = packed[(base + 10 + 24 + lane / 4) as usize];
                let qs_shift = (lane % 4) * 8;
                let qs_byte = (qs_word >> qs_shift) & 0xFF;
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
            // Reduce 4 lanes (WARPS_PER_BLOCK=4): 2-step XOR butterfly.
            acc += warp::shuffle_xor_f32(acc, 2);
            acc += warp::shuffle_xor_f32(acc, 1);
            if lane == 0 {
                unsafe { *out.get_unchecked_mut(n_out as usize) = acc; }
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

    /// Q4_K GEMV with Q8_1-quantized activations and proper 4-element dp4a.
    ///
    /// Mini-warp structure: the 32-lane warp is divided into 4 mini-warps of 8 lanes.
    /// Each mini-warp handles one j-group (64 elements = 32 lo + 32 hi).
    /// Each sub-lane within a mini-warp processes 4 consecutive same-scale elements,
    /// enabling proper 4-element dp4a packing.
    ///
    /// Mini-warp mw (lanes 8*mw .. 8*mw+7) → j-group mw:
    ///   sub_l 0..7: each loads one 4-nibble u32 and one Q8_1 u32, calls dp4a_s32.
    ///   After K loop: 3-step XOR reduce within mini-warp (all 8 lanes get the sum).
    ///   Cross-mini-warp reduce via shared memory → write output.
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
        let mw        = lane / 8;      // 0..3 — j-group this lane handles
        let sub_l     = lane % 8;      // 0..7 — position within mini-warp
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

            // dp4a: 4 nibble×q8 products per instruction.
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

// ── Host-side correctness test ────────────────────────────────────────────────

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
    //   qs: all nibbles = 3 → packed byte = 0x33
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
            // nibble=3: lo=3, hi=3, packed byte = 0x33 → u32 = 0x33333333
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
        shared_mem_bytes: WARPS_PER_BLOCK * 4,
    };

    // ── Test 1: v1 and v2 both pass for nibble=3,sc=1,d=1,dmin=0 ──────────────
    unsafe {
        module.q4k_gemv_decode(
            stream.as_ref(), cfg, &mut out_dev, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v1 launch");

    unsafe {
        module.q4k_gemv_decode_v2(
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

    // ── Test 2: d=2.0, sc=3, nibble=4, dmin=0.5, m=2 ─────────────────────────
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
        module.q4k_gemv_decode_v2(
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

    // ── Test 3: multi-block (K_BLOCKS=4) ──────────────────────────────────────
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
        module.q4k_gemv_decode_v2(
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

    println!("oxide-gemv-q4k: all tests PASSED");
}
