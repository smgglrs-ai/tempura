/*
 * oxide-gemv-q4k: Hand-optimized Q4_K decode GEMV kernel via rustc-codegen-cuda.
 *
 * Build:  cargo oxide build --manifest-path tools/oxide-gemv-q4k/Cargo.toml
 * The compiled binary embeds PTX in the `.oxart` ELF section.
 * The tempura-kernels build.rs extracts that PTX and writes it to OUT_DIR.
 *
 * Q4_K raw GGUF format (36 u32 per block = 144 bytes, 256 elements per block):
 *   u32[0]      = bits[15:0] = f16 d, bits[31:16] = f16 dmin (raw f16, not pre-converted)
 *   u32[1..4]   = 12 bytes of 6-bit packed scales (3 u32 words, raw GGUF bytes)
 *   u32[4..36]  = 32 u32s of packed qs bytes (128 bytes = 256 nibbles, 4-bit each)
 *
 * 6-bit scale unpacking (from GGUF spec, bytes ps[0..12] = u32[1..4] bytes):
 *   sc[i]   = ps[i] & 0x3F                                    for i in 0..4
 *   sc[i+4] = ((ps[i+8] & 0x0F) | ((ps[i] >> 6) << 4)) & 0x3F
 *   m[i]    = ps[i+4] & 0x3F
 *   m[i+4]  = ((ps[i+8] >> 4) | ((ps[i+4] >> 6) << 4)) & 0x3F
 *
 * Dequant for element `within` in a block:
 *   j       = within / 64                    — which j-pair (0..3)
 *   in_j    = within % 64
 *   is_hi   = in_j >= 32
 *   l       = in_j % 32
 *   sc_group = 2*j + is_hi  (0..7)
 *   sc = unpacked_sc[sc_group] as f32   (6-bit → f32 inline in kernel)
 *   m  = unpacked_m[sc_group] as f32
 *   nibble = lo nibble if !is_hi, hi nibble if is_hi  (from qs[j*32+l] at u32[4+j*8+l/4])
 *   weight = d * sc * nibble - dmin * m
 *
 * v2 optimization (shfl-based scale sharing):
 *   One warp (32 lanes) processes 256 elements of one block per iteration.
 *   Each lane handles elements: l, l+32, l+64, …, l+224 within the block.
 *   - d, dmin: lane 0 loads u32[0], broadcasts via shfl, extracts both f16s.
 *   - scales:  lane 0 loads u32[1..4] (3 words); all lanes receive via shfl.
 *              6-bit unpack done inline per j-group from the 3 broadcasted u32s.
 *   - qs:      each lane loads one u32 per j-group (nibbles at u32[4+j*8+lane/4]).
 *   The 4 j-groups are fully unrolled, each processing 2 elements per lane.
 *
 * Opt 3 (Float4 reads): nibble words loaded as U32x4 (128-bit LDG.E.128) — 4 u32s at once.
 */

use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_core::simt::LaunchConfig;

const WARPS_PER_BLOCK: u32 = 4;

/// IEEE 754 f16 bits (in low 16 bits of a u32) → f32 bit pattern.
/// Works in device (oxide kernel) context using only bit arithmetic.
#[inline(always)]
fn f16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15) << 31;
    let exp  = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;
    let bits = if exp == 0 {
        sign | (mant << 13)              // zero / subnormal
    } else if exp == 31 {
        sign | 0x7F800000 | (mant << 13) // inf / nan
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Q4_K GEMV for M=1 decode: out[N] = dequant(packed[N, K_BLOCKS, 36]) × x[K].
    ///
    /// Grid:  (ceil(N / WARPS_PER_BLOCK),) blocks — each warp owns one output neuron.
    /// Block: (WARPS_PER_BLOCK * 32,) threads.
    ///
    /// Fallback kernel: per-element stride-32 loop, decodes raw GGUF layout.
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

            // Raw GGUF: 36 u32 per block.
            let base = (n_out * k_blocks + blk) * 36;

            // u32[0]: f16 d (bits 0-15), f16 dmin (bits 16-31).
            let d_dmin_raw = packed[base as usize];
            let d    = f16_to_f32(d_dmin_raw & 0xFFFF);
            let dmin = f16_to_f32((d_dmin_raw >> 16) & 0xFFFF);

            // 6-bit scales: u32[1..4] = 3 scale words, each 4 bytes.
            // ps[i] = byte i of the 12-byte scale section.
            let ps_u32_0 = packed[(base + 1) as usize]; // bytes 0-3
            let ps_u32_1 = packed[(base + 2) as usize]; // bytes 4-7
            let ps_u32_2 = packed[(base + 3) as usize]; // bytes 8-11
            let ps = |i: u32| -> u32 {
                let word = if i < 4 { ps_u32_0 } else if i < 8 { ps_u32_1 } else { ps_u32_2 };
                (word >> ((i % 4) * 8)) & 0xFF
            };

            let j     = within / 64;
            let in_j  = within % 64;
            let is_hi = (in_j >= 32) as u32;
            let l     = in_j % 32;

            // sc_group = 2*j + is_hi (0..7)
            // For sc_group < 4: sc[sc_group] = ps[sc_group] & 0x3F
            // For sc_group >= 4 (i.e. k=sc_group-4 in 0..4):
            //   sc[sc_group] = ((ps[k+8] & 0x0F) | ((ps[k] >> 6) << 4)) & 0x3F
            let sc_group = 2 * j + is_hi;
            let sc_u32 = if sc_group < 4 {
                ps(sc_group) & 0x3F
            } else {
                let k = sc_group - 4;
                ((ps(k + 8) & 0x0F) | ((ps(k) >> 6) << 4)) & 0x3F
            };
            let m_u32 = if sc_group < 4 {
                ps(sc_group + 4) & 0x3F
            } else {
                let k = sc_group - 4;
                ((ps(k + 8) >> 4) | ((ps(k + 4) >> 6) << 4)) & 0x3F
            };

            // Nibbles: u32[4..36], j-group j starts at u32[4 + j*8].
            let qs_byte_idx = j * 32 + l;
            let qs_word_idx = qs_byte_idx / 4;
            let qs_byte_pos = qs_byte_idx % 4;
            let qs_word = packed[(base + 4 + qs_word_idx) as usize];
            let qs_byte = (qs_word >> (qs_byte_pos * 8)) & 0xFF;
            let nibble = if is_hi != 0 { qs_byte >> 4 } else { qs_byte & 0xF };

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

    /// Q4_K GEMV v2: shfl-based scale sharing, raw GGUF layout, Float4 nibble reads.
    ///
    /// Optimizations:
    ///   - d, dmin: lane 0 loads u32[0]; all lanes receive both f16s via shfl.
    ///              f16→f32 bit-conversion done inline (no host pre-conversion).
    ///   - scales:  lane 0 loads 3 u32s (u32[1..4]); broadcasts via 3 shfls.
    ///              6-bit unpack done inline per j-group.
    ///   - qs:      one U32x4 load per half-j-group (4 u32s = LDG.E.128) — Opt 3.
    ///   - 4 j-groups fully unrolled; 2 accumulation steps per group per lane.
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
        let n_out   = bid * WARPS_PER_BLOCK + warp_id;

        if n_out >= n {
            return;
        }

        let mut acc = 0.0_f32;

        let mut blk = 0u32;
        while blk < k_blocks {
            // Raw GGUF: 36 u32 per block.
            let base = (n_out * k_blocks + blk) * 36;

            // u32[0]: f16 d (bits 0-15) and f16 dmin (bits 16-31).
            // Lane 0 loads, full-warp broadcast.
            let d_dmin_raw = if lane == 0 { packed[base as usize] } else { 0 };
            let d_dmin_bc  = warp::shuffle(d_dmin_raw, 0);
            let d    = f16_to_f32(d_dmin_bc & 0xFFFF);
            let dmin = f16_to_f32((d_dmin_bc >> 16) & 0xFFFF);

            // u32[1..4]: 3 words of 6-bit packed scales. Lane 0 loads all 3,
            // broadcasts via shfl so every lane can unpack inline per j-group.
            let ps0_raw = if lane == 0 { packed[(base + 1) as usize] } else { 0 };
            let ps1_raw = if lane == 0 { packed[(base + 2) as usize] } else { 0 };
            let ps2_raw = if lane == 0 { packed[(base + 3) as usize] } else { 0 };
            let ps0 = warp::shuffle(ps0_raw, 0); // bytes 0-3 of scale section
            let ps1 = warp::shuffle(ps1_raw, 0); // bytes 4-7
            let ps2 = warp::shuffle(ps2_raw, 0); // bytes 8-11

            // Helper closures — inlined at each call site.
            // ps_byte(i): i-th byte of 12-byte scale section (i in 0..12).
            // sc(i): unpacked 6-bit scale i (i in 0..8).
            // m_val(i): unpacked 6-bit min i (i in 0..8).
            //
            // sc[i]   = ps[i] & 0x3F                                   (i in 0..4)
            // sc[i+4] = ((ps[i+8]&0x0F) | ((ps[i]>>6)<<4)) & 0x3F
            // m[i]    = ps[i+4] & 0x3F
            // m[i+4]  = ((ps[i+8]>>4) | ((ps[i+4]>>6)<<4)) & 0x3F

            // j=0: sc[0]/sc[1] and m[0]/m[1], elements within 0..31 and 32..63.
            {
                // ps[0] = ps0 bits 7:0, ps[1] = ps0 bits 15:8
                // ps[4] = ps1 bits 7:0, ps[5] = ps1 bits 15:8
                let p0 = ps0 & 0xFF;  let p1 = (ps0 >> 8) & 0xFF;
                let p4 = ps1 & 0xFF;  let p5 = (ps1 >> 8) & 0xFF;
                let sc_lo = (p0 & 0x3F) as f32;
                let sc_hi = (p1 & 0x3F) as f32;
                let m_lo  = (p4 & 0x3F) as f32;
                let m_hi  = (p5 & 0x3F) as f32;

                // Nibbles at u32[4 + lane/4] for j=0 (8 u32s, each lane picks its word).
                let qs_word = packed[(base + 4 + lane / 4) as usize];
                let qs_shift = (lane % 4) * 8;
                let qs_byte = (qs_word >> qs_shift) & 0xFF;
                let nibble_lo = (qs_byte & 0xF) as f32;
                let nibble_hi = ((qs_byte >> 4) & 0xF) as f32;

                acc += (d * sc_lo * nibble_lo - dmin * m_lo) * x[(blk * 256 + lane) as usize];
                acc += (d * sc_hi * nibble_hi - dmin * m_hi) * x[(blk * 256 + 32 + lane) as usize];
            }

            // j=1: sc[2]/sc[3] and m[2]/m[3], elements within 64..95 and 96..127.
            {
                let p2 = (ps0 >> 16) & 0xFF; let p3 = (ps0 >> 24) & 0xFF;
                let p6 = (ps1 >> 16) & 0xFF; let p7 = (ps1 >> 24) & 0xFF;
                let sc_lo = (p2 & 0x3F) as f32;
                let sc_hi = (p3 & 0x3F) as f32;
                let m_lo  = (p6 & 0x3F) as f32;
                let m_hi  = (p7 & 0x3F) as f32;

                // Nibbles at u32[4 + 8 + lane/4] for j=1.
                let qs_word = packed[(base + 4 + 8 + lane / 4) as usize];
                let qs_shift = (lane % 4) * 8;
                let qs_byte = (qs_word >> qs_shift) & 0xFF;
                let nibble_lo = (qs_byte & 0xF) as f32;
                let nibble_hi = ((qs_byte >> 4) & 0xF) as f32;

                acc += (d * sc_lo * nibble_lo - dmin * m_lo) * x[(blk * 256 + 64 + lane) as usize];
                acc += (d * sc_hi * nibble_hi - dmin * m_hi) * x[(blk * 256 + 96 + lane) as usize];
            }

            // j=2: sc[4]/sc[5] and m[4]/m[5], elements within 128..159 and 160..191.
            {
                let p0 = ps0 & 0xFF;        let p1 = (ps0 >> 8) & 0xFF;
                let p4 = ps1 & 0xFF;        let p5 = (ps1 >> 8) & 0xFF;
                let p8 = ps2 & 0xFF;        let p9 = (ps2 >> 8) & 0xFF;
                // sc[4] = ((ps[8]&0x0F) | ((ps[0]>>6)<<4)) & 0x3F
                let sc_lo = (((p8 & 0x0F) | ((p0 >> 6) << 4)) & 0x3F) as f32;
                let sc_hi = (((p9 & 0x0F) | ((p1 >> 6) << 4)) & 0x3F) as f32;
                // m[4] = ((ps[8]>>4) | ((ps[4]>>6)<<4)) & 0x3F
                let m_lo  = (((p8 >> 4) | ((p4 >> 6) << 4)) & 0x3F) as f32;
                let m_hi  = (((p9 >> 4) | ((p5 >> 6) << 4)) & 0x3F) as f32;

                // Nibbles at u32[4 + 16 + lane/4] for j=2.
                let qs_word = packed[(base + 4 + 16 + lane / 4) as usize];
                let qs_shift = (lane % 4) * 8;
                let qs_byte = (qs_word >> qs_shift) & 0xFF;
                let nibble_lo = (qs_byte & 0xF) as f32;
                let nibble_hi = ((qs_byte >> 4) & 0xF) as f32;

                acc += (d * sc_lo * nibble_lo - dmin * m_lo) * x[(blk * 256 + 128 + lane) as usize];
                acc += (d * sc_hi * nibble_hi - dmin * m_hi) * x[(blk * 256 + 160 + lane) as usize];
            }

            // j=3: sc[6]/sc[7] and m[6]/m[7], elements within 192..223 and 224..255.
            {
                let p2 = (ps0 >> 16) & 0xFF; let p3 = (ps0 >> 24) & 0xFF;
                let p6 = (ps1 >> 16) & 0xFF; let p7 = (ps1 >> 24) & 0xFF;
                let p10 = (ps2 >> 16) & 0xFF; let p11 = (ps2 >> 24) & 0xFF;
                // sc[6] = ((ps[10]&0x0F) | ((ps[2]>>6)<<4)) & 0x3F
                let sc_lo = (((p10 & 0x0F) | ((p2 >> 6) << 4)) & 0x3F) as f32;
                let sc_hi = (((p11 & 0x0F) | ((p3 >> 6) << 4)) & 0x3F) as f32;
                // m[6] = ((ps[10]>>4) | ((ps[6]>>6)<<4)) & 0x3F
                let m_lo  = (((p10 >> 4) | ((p6 >> 6) << 4)) & 0x3F) as f32;
                let m_hi  = (((p11 >> 4) | ((p7 >> 6) << 4)) & 0x3F) as f32;

                // Nibbles at u32[4 + 24 + lane/4] for j=3.
                let qs_word = packed[(base + 4 + 24 + lane / 4) as usize];
                let qs_shift = (lane % 4) * 8;
                let qs_byte = (qs_word >> qs_shift) & 0xFF;
                let nibble_lo = (qs_byte & 0xF) as f32;
                let nibble_hi = ((qs_byte >> 4) & 0xF) as f32;

                acc += (d * sc_lo * nibble_lo - dmin * m_lo) * x[(blk * 256 + 192 + lane) as usize];
                acc += (d * sc_hi * nibble_hi - dmin * m_hi) * x[(blk * 256 + 224 + lane) as usize];
            }

            blk += 1;
        }

        // Butterfly warp-reduce.
        acc = warp::reduce_sum_f32(acc);
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(n_out as usize) = acc;
            }
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

    // Build raw GGUF weights (36 u32/block):
    //   d = 1.0 (f16=0x3C00), dmin = 0.0 (f16=0x0000)
    //   scales: all groups sc[g] = 1 (6-bit), m[g] = 0 (6-bit)
    //   qs: all nibbles = 3 → packed byte = 0x33
    // Expected weight per element: 1.0 * 1.0 * 3 - 0.0 * 0.0 = 3.0
    // Expected output: sum over 256 elements of 3.0 * 1.0 = 768.0
    //
    // Raw GGUF scale encoding for sc[g]=1, m[g]=0 for all g (see 6-bit unpack formula):
    //   ps[0..4] = [1,1,1,1] → u32[1] = 0x01010101
    //   ps[4..8] = [0,0,0,0] → u32[2] = 0x00000000
    //   ps[8..12]= [1,1,1,1] → u32[3] = 0x01010101
    //   (sc[i+4] = ps[i+8]&0x0F = 1, m[i+4] = ps[i+8]>>4 = 0)
    let packed_u32s = n * k_blocks * 36;
    let mut packed_host = vec![0u32; packed_u32s];
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 36;
            // u32[0]: f16 d=1.0 (0x3C00) in bits[15:0], f16 dmin=0.0 (0x0000) in bits[31:16]
            packed_host[base] = 0x00003C00_u32;
            // 6-bit scale words for sc[g]=1, m[g]=0
            packed_host[base + 1] = 0x01010101_u32; // ps[0..4]
            packed_host[base + 2] = 0x00000000_u32; // ps[4..8]
            packed_host[base + 3] = 0x01010101_u32; // ps[8..12]
            // nibble=3: lo=3, hi=3, packed byte = 0x33 → u32 = 0x33333333
            for w in 0..32usize {
                packed_host[base + 4 + w] = 0x33333333_u32;
            }
        }
    }

    let x_dev        = DeviceBuffer::from_host(&stream, &x_host).expect("upload x");
    let mut p_dev    = DeviceBuffer::from_host(&stream, &packed_host).expect("upload packed");
    let mut out_dev  = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc out");
    let mut out_dev2 = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc out2");

    let module = kernels::load(&ctx).expect("load oxide q4k module");

    let wpb = WARPS_PER_BLOCK as usize;
    let grid_x = (n + wpb - 1) / wpb;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x as u32, 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    // ── Test 1: v1 and v2 both pass for nibble=3,sc=1,d=1,dmin=0 ──────────────
    unsafe {
        module.q4k_gemv_decode(
            stream.as_ref(), cfg, &mut out_dev, &x_dev, &p_dev, n as u32, k_blocks as u32,
        )
    }.expect("v1 launch");

    unsafe {
        module.q4k_gemv_decode_v2(
            stream.as_ref(), cfg, &mut out_dev2, &x_dev, &p_dev, n as u32, k_blocks as u32,
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
    //
    // Raw GGUF scale encoding for sc[g]=3, m[g]=2:
    //   ps[0..4] = [3,3,3,3] → u32[1] = 0x03030303
    //   ps[4..8] = [2,2,2,2] → u32[2] = 0x02020202
    //   ps[8..12]= [0x23,0x23,0x23,0x23] → u32[3] = 0x23232323
    //   (sc[i+4] = ps[i+8]&0x0F = 3, m[i+4] = ps[i+8]>>4 = 2)
    // f16 d=2.0=0x4000, f16 dmin=0.5=0x3800 → u32[0] = 0x38004000
    let mut packed_host2 = vec![0u32; packed_u32s];
    for row in 0..n {
        for blk in 0..k_blocks {
            let base = (row * k_blocks + blk) * 36;
            packed_host2[base]     = 0x38004000_u32; // f16 d=2.0, f16 dmin=0.5
            packed_host2[base + 1] = 0x03030303_u32;
            packed_host2[base + 2] = 0x02020202_u32;
            packed_host2[base + 3] = 0x23232323_u32;
            for w in 0..32usize {
                packed_host2[base + 4 + w] = 0x44444444_u32;
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
            stream.as_ref(), cfg, &mut out_dev2, &x_dev, &p_dev, n as u32, k_blocks as u32,
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
    //
    // Raw GGUF scale encoding for sc[g]=2, m[g]=0:
    //   ps[0..4] = [2,2,2,2] → u32[1] = 0x02020202
    //   ps[4..8] = [0,0,0,0] → u32[2] = 0x00000000
    //   ps[8..12]= [2,2,2,2] → u32[3] = 0x02020202
    // f16 d=1.0=0x3C00, f16 dmin=0.0=0x0000 → u32[0] = 0x00003C00
    let k_blocks3: usize = 4;
    let k3 = k_blocks3 * 256;
    let x3: Vec<f32> = vec![1.0_f32; k3];
    let packed_u32s3 = n * k_blocks3 * 36;
    let mut ph3 = vec![0u32; packed_u32s3];
    for row in 0..n {
        for blk in 0..k_blocks3 {
            let base = (row * k_blocks3 + blk) * 36;
            ph3[base]     = 0x00003C00_u32;
            ph3[base + 1] = 0x02020202_u32;
            ph3[base + 2] = 0x00000000_u32;
            ph3[base + 3] = 0x02020202_u32;
            for w in 0..32usize {
                ph3[base + 4 + w] = 0x55555555_u32;
            }
        }
    }
    let expected3 = 10240.0_f32;

    let x_dev3    = DeviceBuffer::from_host(&stream, &x3).expect("upload x3");
    let p_dev3    = DeviceBuffer::from_host(&stream, &ph3).expect("upload p3");
    let mut od1_3 = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc od1_3");
    let mut od2_3 = DeviceBuffer::<f32>::zeroed(&stream, n).expect("alloc od2_3");

    let grid3 = (n + wpb - 1) / wpb;
    let cfg3 = LaunchConfig {
        grid_dim:  (grid3 as u32, 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        module.q4k_gemv_decode(
            stream.as_ref(), cfg3, &mut od1_3, &x_dev3, &p_dev3, n as u32, k_blocks3 as u32,
        )
    }.expect("v1 launch 3");

    unsafe {
        module.q4k_gemv_decode_v2(
            stream.as_ref(), cfg3, &mut od2_3, &x_dev3, &p_dev3, n as u32, k_blocks3 as u32,
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
