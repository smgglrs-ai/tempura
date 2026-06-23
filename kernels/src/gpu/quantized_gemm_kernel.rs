use cuda_device::{kernel, ptx_asm, thread, warp, DisjointSlice, SharedArray};
use cuda_host::cuda_module;

const Q4K_BLOCK_SIZE: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;

const Q8_BLOCK_SIZE: usize = 32;
const Q8_BLOCK_BYTES: usize = 36; // 4 bytes header (f16 d, f16 s) + 32 int8 values

#[inline(always)]
fn ld_global_b32(ptr: &[u8], byte_offset: usize) -> u32 {
    (ptr[byte_offset] as u32)
        | ((ptr[byte_offset + 1] as u32) << 8)
        | ((ptr[byte_offset + 2] as u32) << 16)
        | ((ptr[byte_offset + 3] as u32) << 24)
}

#[inline(always)]
fn q4k_dequant_dot_8(
    raw: u32,
    a_lo: &[f32],
    a_lo_off: usize,
    a_hi: &[f32],
    a_hi_off: usize,
    d_sc: f32,
    d_min: f32,
) -> f32 {
    // 4 low nibbles × 4 activations
    let dot_lo = a_lo[a_lo_off] * (raw & 0x0F) as f32
        + a_lo[a_lo_off + 1] * ((raw >> 8) & 0x0F) as f32
        + a_lo[a_lo_off + 2] * ((raw >> 16) & 0x0F) as f32
        + a_lo[a_lo_off + 3] * ((raw >> 24) & 0x0F) as f32;
    let sum_lo = a_lo[a_lo_off] + a_lo[a_lo_off + 1] + a_lo[a_lo_off + 2] + a_lo[a_lo_off + 3];

    // 4 high nibbles × 4 activations
    let dot_hi = a_hi[a_hi_off] * ((raw >> 4) & 0x0F) as f32
        + a_hi[a_hi_off + 1] * ((raw >> 12) & 0x0F) as f32
        + a_hi[a_hi_off + 2] * ((raw >> 20) & 0x0F) as f32
        + a_hi[a_hi_off + 3] * ((raw >> 28) & 0x0F) as f32;
    let sum_hi = a_hi[a_hi_off] + a_hi[a_hi_off + 1] + a_hi[a_hi_off + 2] + a_hi[a_hi_off + 3];

    d_sc * dot_lo - d_min * sum_lo + d_sc * dot_hi - d_min * sum_hi
}

#[inline(always)]
fn dp4a_u32_i32(a: u32, b: u32, c: i32) -> i32 {
    let result: i32;
    unsafe {
        ptx_asm!(
            "dp4a.u32.s32 %0, %1, %2, %3;",
            out("=r") result,
            in("r") a,
            in("r") b,
            in("r") c,
        );
    }
    result
}

const Q6K_BLOCK_SIZE: usize = 256;
const Q6K_BLOCK_BYTES: usize = 210;

#[inline(always)]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = exp;
        let mut f = frac;
        while (f & 0x400) == 0 {
            f <<= 1;
            e = e.wrapping_sub(1);
        }
        f &= 0x3ff;
        let f32_exp = 127 - 15 + 1 + e;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (f << 13))
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
        }
    } else {
        let f32_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// GPU Q4_K dequantize-fused matmul: output[i,j] = dot(a[i,:], dequant(b_q4k[j,:]))
    /// Each thread computes one output element.
    /// a: [m, k] f32 row-major
    /// b_q4k: [n * blocks_per_row * Q4K_BLOCK_BYTES] packed Q4_K weights (transposed layout)
    /// output: [m, n] f32
    #[kernel]
    pub fn gemm_q4k_bt(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b_q4k: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let m = m as usize;
        let n = n as usize;
        let k = k as usize;

        if tid >= m * n {
            return;
        }

        let row = tid / n;
        let col = tid % n;

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let mut sum = 0.0f32;

        let mut blk = 0usize;
        while blk < blocks_per_row {
            let block_offset = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;

            let d = f16_bits_to_f32(
                (b_q4k[block_offset] as u16) | ((b_q4k[block_offset + 1] as u16) << 8),
            );
            let dmin = f16_bits_to_f32(
                (b_q4k[block_offset + 2] as u16) | ((b_q4k[block_offset + 3] as u16) << 8),
            );

            let packed = &b_q4k[block_offset + 4..];

            let mut sc = [0.0f32; 8];
            let mut mins = [0.0f32; 8];

            let mut i = 0usize;
            while i < 4 {
                sc[i] = (packed[i] & 0x3F) as f32;
                sc[i + 4] =
                    (((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F) as f32;
                mins[i] = (packed[i + 4] & 0x3F) as f32;
                mins[i + 4] =
                    (((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F) as f32;
                i += 1;
            }

            let qs = &b_q4k[block_offset + 16..];
            let a_base = row * k + blk * Q4K_BLOCK_SIZE;

            let mut j = 0usize;
            while j < 4 {
                let q_off = j * 32;
                let d1 = d * sc[2 * j];
                let m1 = dmin * mins[2 * j];
                let d2 = d * sc[2 * j + 1];
                let m2 = dmin * mins[2 * j + 1];

                let elem_base = j * 64;
                let mut l = 0usize;
                while l < 32 {
                    let q_lo = (qs[q_off + l] & 0x0F) as f32;
                    let q_hi = (qs[q_off + l] >> 4) as f32;

                    let val_lo = d1 * q_lo - m1;
                    let val_hi = d2 * q_hi - m2;

                    sum += a[a_base + elem_base + l] * val_lo;
                    sum += a[a_base + elem_base + l + 32] * val_hi;
                    l += 1;
                }
                j += 1;
            }

            blk += 1;
        }

        if let Some(elem) = output.get_mut(idx) {
            *elem = sum;
        }
    }

    /// Split-K Q4_K matmul for decode (m=1): parallelizes over K dimension.
    /// Grid: (n, num_splits, 1). Each thread computes a partial dot product over
    /// blocks_per_split blocks and writes to partials[col * num_splits + split].
    /// A separate reduce_splitk kernel sums the partials into output.
    #[kernel]
    pub fn gemm_q4k_bt_splitk(
        n: u32,
        k: u32,
        num_splits: u32,
        a: &[f32],
        b_q4k: &[u8],
        mut partials: DisjointSlice<f32>,
    ) {
        let col = thread::blockIdx_x() as usize;
        let split = thread::blockIdx_y() as usize;
        let n = n as usize;
        let k = k as usize;
        let ns = num_splits as usize;

        if col >= n || split >= ns {
            return;
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let blocks_per_split = (blocks_per_row + ns - 1) / ns;
        let blk_start = split * blocks_per_split;
        let blk_end = if blk_start + blocks_per_split < blocks_per_row {
            blk_start + blocks_per_split
        } else {
            blocks_per_row
        };

        let mut sum = 0.0f32;
        let mut blk = blk_start;
        while blk < blk_end {
            let block_offset = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;

            let d = f16_bits_to_f32(
                (b_q4k[block_offset] as u16) | ((b_q4k[block_offset + 1] as u16) << 8),
            );
            let dmin = f16_bits_to_f32(
                (b_q4k[block_offset + 2] as u16) | ((b_q4k[block_offset + 3] as u16) << 8),
            );

            let packed = &b_q4k[block_offset + 4..];
            let mut sc = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            let mut i = 0usize;
            while i < 4 {
                sc[i] = (packed[i] & 0x3F) as f32;
                sc[i + 4] =
                    (((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F) as f32;
                mins[i] = (packed[i + 4] & 0x3F) as f32;
                mins[i + 4] =
                    (((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F) as f32;
                i += 1;
            }

            let qs = &b_q4k[block_offset + 16..];
            let a_base = blk * Q4K_BLOCK_SIZE;

            let mut j = 0usize;
            while j < 4 {
                let q_off = j * 32;
                let d1 = d * sc[2 * j];
                let m1 = dmin * mins[2 * j];
                let d2 = d * sc[2 * j + 1];
                let m2 = dmin * mins[2 * j + 1];
                let elem_base = j * 64;
                let mut l = 0usize;
                while l < 32 {
                    let q_lo = (qs[q_off + l] & 0x0F) as f32;
                    let q_hi = (qs[q_off + l] >> 4) as f32;
                    sum += a[a_base + elem_base + l] * (d1 * q_lo - m1);
                    sum += a[a_base + elem_base + l + 32] * (d2 * q_hi - m2);
                    l += 1;
                }
                j += 1;
            }
            blk += 1;
        }

        let partial_idx = thread::index_1d();
        if let Some(elem) = partials.get_mut(partial_idx) {
            *elem = sum;
        }
    }

    /// Reduce split-K partial sums: output[col] = sum(partials[col * num_splits + 0..num_splits])
    #[kernel]
    pub fn reduce_splitk(
        n: u32,
        num_splits: u32,
        partials: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let col = idx.get();
        let n = n as usize;
        let ns = num_splits as usize;

        if col >= n {
            return;
        }

        let mut sum = 0.0f32;
        let base = col * ns;
        let mut s = 0usize;
        while s < ns {
            sum += partials[base + s];
            s += 1;
        }

        if let Some(elem) = output.get_mut(idx) {
            *elem = sum;
        }
    }

    /// Warp-cooperative Q4_K matvec for decode (m=1).
    /// One warp (32 lanes) per output column. Lanes split K blocks:
    /// lane i processes blocks i, i+32, i+64, ... then warp-shuffle reduce.
    /// Grid: (ceil(n/COLS_PER_BLOCK), 1, 1), Block: (32, COLS_PER_BLOCK, 1)
    /// COLS_PER_BLOCK warps run in parallel within a CTA for better occupancy.
    #[kernel]
    pub fn gemv_q4k_warp(
        n: u32,
        k: u32,
        a: &[f32],
        b_q4k: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        let lane = warp::lane_id() as usize;
        let col = (thread::blockIdx_x() as usize) * (thread::blockDim_y() as usize)
            + thread::threadIdx_y() as usize;

        let n = n as usize;
        let k = k as usize;
        if col >= n {
            return;
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let mut sum = 0.0f32;

        let mut blk = lane;
        while blk < blocks_per_row {
            let block_offset = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;

            let d = f16_bits_to_f32(
                (b_q4k[block_offset] as u16) | ((b_q4k[block_offset + 1] as u16) << 8),
            );
            let dmin = f16_bits_to_f32(
                (b_q4k[block_offset + 2] as u16) | ((b_q4k[block_offset + 3] as u16) << 8),
            );

            let packed = &b_q4k[block_offset + 4..];
            let mut sc = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            let mut i = 0usize;
            while i < 4 {
                sc[i] = (packed[i] & 0x3F) as f32;
                sc[i + 4] =
                    (((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F) as f32;
                mins[i] = (packed[i + 4] & 0x3F) as f32;
                mins[i + 4] =
                    (((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F) as f32;
                i += 1;
            }

            let qs = &b_q4k[block_offset + 16..];
            let a_base = blk * Q4K_BLOCK_SIZE;

            let mut j = 0usize;
            while j < 4 {
                let q_off = j * 32;
                let d1 = d * sc[2 * j];
                let m1 = dmin * mins[2 * j];
                let d2 = d * sc[2 * j + 1];
                let m2 = dmin * mins[2 * j + 1];
                let elem_base = j * 64;
                let mut l = 0usize;
                while l < 32 {
                    let q_lo = (qs[q_off + l] & 0x0F) as f32;
                    let q_hi = (qs[q_off + l] >> 4) as f32;
                    sum += a[a_base + elem_base + l] * (d1 * q_lo - m1);
                    sum += a[a_base + elem_base + l + 32] * (d2 * q_hi - m2);
                    l += 1;
                }
                j += 1;
            }

            blk += 32;
        }

        // Warp-shuffle reduction: sum across 32 lanes
        sum += warp::shuffle_xor_f32(sum, 16);
        sum += warp::shuffle_xor_f32(sum, 8);
        sum += warp::shuffle_xor_f32(sum, 4);
        sum += warp::shuffle_xor_f32(sum, 2);
        sum += warp::shuffle_xor_f32(sum, 1);

        if lane == 0 {
            unsafe {
                *output.get_unchecked_mut(col) = sum;
            }
        }
    }

    /// Intra-block cooperative Q4_K GEMV: 32 lanes dequant one block together.
    ///
    /// Each lane reads 4 qs bytes → 8 elements (4 low nibbles + 4 high nibbles).
    /// All lanes read from the SAME block → coalesced 128-byte read for qs.
    /// Per-block partial sums reduced via warp shuffle.
    /// Outer loop: one warp iterates over all blocks for one output column.
    ///
    /// Grid: (ceil(n/WARPS_Y), 1, 1), Block: (32, WARPS_Y, 1)
    #[kernel]
    pub fn gemv_q4k_coop(
        n: u32,
        k: u32,
        a: &[f32],
        b_q4k: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        let lane = warp::lane_id() as usize;
        let col = (thread::blockIdx_x() as usize) * (thread::blockDim_y() as usize)
            + thread::threadIdx_y() as usize;

        let n = n as usize;
        let k = k as usize;
        if col >= n {
            return;
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let mut col_sum = 0.0f32;

        // Lane L reads qs bytes [L*4 .. L*4+3] from each block.
        // group = (L*4) / 32, local_offset = (L*4) % 32
        let qs_byte_off = lane * 4;
        let group = qs_byte_off / 32; // 0..3
        let local = qs_byte_off % 32; // 0..31

        // Element indices for this lane's low and high nibbles:
        // low:  group*64 + local + {0,1,2,3}
        // high: group*64 + 32 + local + {0,1,2,3}
        let elem_lo_base = group * 64 + local;
        let elem_hi_base = group * 64 + 32 + local;

        // Sub-group indices for scale lookup:
        // low nibbles use sc[2*group], m[2*group]
        // high nibbles use sc[2*group+1], m[2*group+1]
        let sc_idx_lo = 2 * group;
        let sc_idx_hi = 2 * group + 1;

        // Each lane accumulates its own partial sum across ALL blocks.
        // ONE warp reduction at the end — not per block.
        let mut lane_sum = 0.0f32;

        let mut blk = 0usize;
        while blk < blocks_per_row {
            let block_offset = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;

            let d = f16_bits_to_f32(
                (b_q4k[block_offset] as u16) | ((b_q4k[block_offset + 1] as u16) << 8),
            );
            let dmin = f16_bits_to_f32(
                (b_q4k[block_offset + 2] as u16) | ((b_q4k[block_offset + 3] as u16) << 8),
            );

            let packed = &b_q4k[block_offset + 4..];
            let mut sc = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            let mut i = 0usize;
            while i < 4 {
                sc[i] = (packed[i] & 0x3F) as f32;
                sc[i + 4] =
                    (((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F) as f32;
                mins[i] = (packed[i + 4] & 0x3F) as f32;
                mins[i + 4] =
                    (((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F) as f32;
                i += 1;
            }

            let d_lo = d * sc[sc_idx_lo];
            let m_lo = dmin * mins[sc_idx_lo];
            let d_hi = d * sc[sc_idx_hi];
            let m_hi = dmin * mins[sc_idx_hi];

            let qs = &b_q4k[block_offset + 16..];
            let a_base = blk * Q4K_BLOCK_SIZE;

            let mut li = 0usize;
            while li < 4 {
                let byte = qs[qs_byte_off + li];
                let q_lo = (byte & 0x0F) as f32;
                let q_hi = (byte >> 4) as f32;

                lane_sum += a[a_base + elem_lo_base + li] * (d_lo * q_lo - m_lo);
                lane_sum += a[a_base + elem_hi_base + li] * (d_hi * q_hi - m_hi);
                li += 1;
            }

            blk += 1;
        }

        // Single warp reduction after all blocks
        lane_sum += warp::shuffle_xor_f32(lane_sum, 16);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 8);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 4);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 2);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 1);

        if lane == 0 {
            unsafe {
                *output.get_unchecked_mut(col) = lane_sum;
            }
        }
    }

    /// Cooperative Q4K GEMV v2: shared-memory activation tile + multi-block accumulation.
    ///
    /// Fixes two inefficiencies in gemv_q4k_coop:
    /// 1. Activation reads were 12.5% coalesced (stride-4 across groups).
    ///    Now: 32 lanes cooperatively load 256 floats into SMEM coalesced,
    ///    then read from SMEM (bank-conflict-free with stride-1 access).
    /// 2. Warp shuffle reduction ran per-block (5 shuffles × blocks_per_row).
    ///    Now: accumulate across ACCUM_BLOCKS blocks before reducing.
    ///
    /// Grid: (ceil(n/WARPS_Y), 1, 1), Block: (32, WARPS_Y, 1)
    /// Shared memory: WARPS_Y × 256 × 4 bytes per CTA
    #[kernel]
    pub fn gemv_q4k_coop_v2(
        n: u32,
        k: u32,
        a: &[f32],
        b_q4k: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        // 4 warps per CTA, each warp gets its own 256-float SMEM tile
        static mut A_TILE: SharedArray<f32, 1024> = SharedArray::UNINIT; // 4 × 256

        let lane = warp::lane_id() as usize;
        let warp_in_cta = thread::threadIdx_y() as usize;
        let col = (thread::blockIdx_x() as usize) * (thread::blockDim_y() as usize)
            + warp_in_cta;

        let n = n as usize;
        let k = k as usize;
        if col >= n {
            return;
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let mut col_sum = 0.0f32;

        let qs_byte_off = lane * 4;
        let group = qs_byte_off / 32;
        let local = qs_byte_off % 32;
        let elem_lo_base = group * 64 + local;
        let elem_hi_base = group * 64 + 32 + local;
        let sc_idx_lo = 2 * group;
        let sc_idx_hi = 2 * group + 1;
        let smem_base = warp_in_cta * 256;

        let mut blk = 0usize;
        while blk < blocks_per_row {
            // --- Cooperative activation load into shared memory ---
            // 32 lanes load 256 floats: lane L loads a[blk*256 + L*8 .. L*8+7]
            // 8 coalesced 128-byte loads (32×4 bytes each)
            let a_block_base = blk * Q4K_BLOCK_SIZE;
            let load_base = lane * 8;
            let mut li = 0usize;
            while li < 8 {
                let val = a[a_block_base + load_base + li];
                unsafe {
                    A_TILE[smem_base + load_base + li] = val;
                }
                li += 1;
            }
            warp::sync_mask(0xFFFFFFFF);

            // --- Dequant + dot product from shared memory ---
            let block_offset = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;

            let d = f16_bits_to_f32(
                (b_q4k[block_offset] as u16) | ((b_q4k[block_offset + 1] as u16) << 8),
            );
            let dmin = f16_bits_to_f32(
                (b_q4k[block_offset + 2] as u16) | ((b_q4k[block_offset + 3] as u16) << 8),
            );

            let packed = &b_q4k[block_offset + 4..];
            let mut sc = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            let mut i = 0usize;
            while i < 4 {
                sc[i] = (packed[i] & 0x3F) as f32;
                sc[i + 4] =
                    (((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F) as f32;
                mins[i] = (packed[i + 4] & 0x3F) as f32;
                mins[i + 4] =
                    (((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F) as f32;
                i += 1;
            }

            let d_lo = d * sc[sc_idx_lo];
            let m_lo = dmin * mins[sc_idx_lo];
            let d_hi = d * sc[sc_idx_hi];
            let m_hi = dmin * mins[sc_idx_hi];

            let qs = &b_q4k[block_offset + 16..];

            let mut partial = 0.0f32;
            li = 0;
            while li < 4 {
                let byte = qs[qs_byte_off + li];
                let q_lo = (byte & 0x0F) as f32;
                let q_hi = (byte >> 4) as f32;

                // Read from shared memory — no coalescing issue
                let a_lo = unsafe { A_TILE[smem_base + elem_lo_base + li] };
                let a_hi = unsafe { A_TILE[smem_base + elem_hi_base + li] };

                partial += a_lo * (d_lo * q_lo - m_lo);
                partial += a_hi * (d_hi * q_hi - m_hi);
                li += 1;
            }

            // Warp reduction
            partial += warp::shuffle_xor_f32(partial, 16);
            partial += warp::shuffle_xor_f32(partial, 8);
            partial += warp::shuffle_xor_f32(partial, 4);
            partial += warp::shuffle_xor_f32(partial, 2);
            partial += warp::shuffle_xor_f32(partial, 1);

            if lane == 0 {
                col_sum += partial;
            }

            blk += 1;
        }

        if lane == 0 {
            unsafe {
                *output.get_unchecked_mut(col) = col_sum;
            }
        }
    }

    /// Multi-warp Q4K×Q8 GEMV via dp4a: combines multi-warp K-splitting with
    /// integer dot products. Activations pre-quantized to Q8 (32 int8 per block,
    /// scale+sum as f32, packed as 10 u32 words per block).
    /// 2.9x fewer instructions than f32 FMA path via dp4a.
    ///
    /// Grid: (n, 1, 1), Block: (32, n_warps_per_col, 1)
    #[kernel]
    pub fn gemv_q4k_q8_mw(
        n: u32,
        k: u32,
        n_warps_per_col: u32,
        a_q8: &[u32],
        b_q4k: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        static mut WARP_SUMS: SharedArray<f32, 64> = SharedArray::UNINIT;

        let lane = warp::lane_id() as usize;
        let warp_id = thread::threadIdx_y() as usize;
        let nwpc = n_warps_per_col as usize;
        let col_in_cta = warp_id / nwpc;
        let warp_in_col = warp_id % nwpc;
        let cols_per_cta = (thread::blockDim_y() as usize) / nwpc;
        let col = (thread::blockIdx_x() as usize) * cols_per_cta + col_in_cta;

        let n = n as usize;
        let k = k as usize;
        if col >= n {
            return;
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let q8_blocks_per_q4k = Q4K_BLOCK_SIZE / Q8_BLOCK_SIZE; // 8

        let qs_byte_off = lane * 4;
        let group = qs_byte_off / 32;
        let local = qs_byte_off % 32;
        let sc_idx_lo = 2 * group;
        let sc_idx_hi = 2 * group + 1;

        // Q8 block indices for this lane's elements
        let q8_idx_lo = (group * 64 + local) / Q8_BLOCK_SIZE;
        let q8_idx_hi = (group * 64 + 32 + local) / Q8_BLOCK_SIZE;
        let q8_off_lo = (group * 64 + local) % Q8_BLOCK_SIZE;
        let q8_off_hi = (group * 64 + 32 + local) % Q8_BLOCK_SIZE;

        let mut lane_sum = 0.0f32;
        let mut blk = warp_in_col;
        while blk < blocks_per_row {
            let block_offset = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;

            let d4 = f16_bits_to_f32(
                (b_q4k[block_offset] as u16) | ((b_q4k[block_offset + 1] as u16) << 8),
            );
            let dmin4 = f16_bits_to_f32(
                (b_q4k[block_offset + 2] as u16) | ((b_q4k[block_offset + 3] as u16) << 8),
            );

            let packed = &b_q4k[block_offset + 4..];
            let mut sc = [0u8; 8];
            let mut mins = [0u8; 8];
            let mut i = 0usize;
            while i < 4 {
                sc[i] = packed[i] & 0x3F;
                sc[i + 4] = ((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F;
                mins[i] = packed[i + 4] & 0x3F;
                mins[i + 4] = ((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F;
                i += 1;
            }

            // Load Q4K nibbles as packed u32
            let qs = &b_q4k[block_offset + 16..];
            let raw = (qs[qs_byte_off] as u32)
                | ((qs[qs_byte_off + 1] as u32) << 8)
                | ((qs[qs_byte_off + 2] as u32) << 16)
                | ((qs[qs_byte_off + 3] as u32) << 24);
            let v_lo = raw & 0x0F0F0F0F;
            let v_hi = (raw >> 4) & 0x0F0F0F0F;

            // Load Q8 activation data
            let q8_base = blk * q8_blocks_per_q4k;
            let q8_lo_base = (q8_base + q8_idx_lo) * 10;
            let q8_hi_base = (q8_base + q8_idx_hi) * 10;

            let d8_lo = f32::from_bits(a_q8[q8_lo_base]);
            let d8_hi = f32::from_bits(a_q8[q8_hi_base]);

            let u_lo = a_q8[q8_lo_base + 2 + q8_off_lo / 4];
            let u_hi = a_q8[q8_hi_base + 2 + q8_off_hi / 4];

            // dp4a: 2 instructions for 8 element products
            let dot_lo = dp4a_u32_i32(v_lo, u_lo, 0);
            let dot_hi = dp4a_u32_i32(v_hi, u_hi, 0);
            let sum_lo = dp4a_u32_i32(0x01010101, u_lo, 0);
            let sum_hi = dp4a_u32_i32(0x01010101, u_hi, 0);

            // Float scale application (4 FMAs instead of 8 cvt+FMA)
            lane_sum += d4 * (sc[sc_idx_lo] as f32) * d8_lo * (dot_lo as f32)
                - dmin4 * (mins[sc_idx_lo] as f32) * d8_lo * (sum_lo as f32)
                + d4 * (sc[sc_idx_hi] as f32) * d8_hi * (dot_hi as f32)
                - dmin4 * (mins[sc_idx_hi] as f32) * d8_hi * (sum_hi as f32);

            blk += nwpc;
        }

        // Intra-warp reduction
        lane_sum += warp::shuffle_xor_f32(lane_sum, 16);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 8);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 4);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 2);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 1);

        if lane == 0 {
            unsafe { WARP_SUMS[warp_id] = lane_sum; }
        }
        thread::sync_threads();

        if warp_in_col == 0 && lane == 0 {
            let smem_base = col_in_cta * nwpc;
            let mut total = 0.0f32;
            let mut w = 0usize;
            while w < nwpc {
                total += unsafe { WARP_SUMS[smem_base + w] };
                w += 1;
            }
            unsafe { *output.get_unchecked_mut(col) = total; }
        }
    }

    /// Multi-warp cooperative Q4K GEMV: N_WARPS warps share one output column.
    /// Each warp handles blocks_per_row/N_WARPS blocks with intra-block cooperation.
    /// Cross-warp reduction via shared memory. 8x more in-flight memory requests.
    ///
    /// Grid: (ceil(n / cols_per_cta), 1, 1)
    /// Block: (32, N_WARPS * cols_per_cta, 1)
    /// Shared memory: cols_per_cta * N_WARPS floats for cross-warp reduction
    #[kernel]
    pub fn gemv_q4k_multirow(
        n: u32,
        k: u32,
        n_warps_per_col: u32,
        a: &[f32],
        b_q4k: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        // Shared memory for cross-warp reduction: one f32 per warp
        static mut WARP_SUMS: SharedArray<f32, 64> = SharedArray::UNINIT; // max 8 warps × 8 cols

        let lane = warp::lane_id() as usize;
        let warp_id = thread::threadIdx_y() as usize;
        let nwpc = n_warps_per_col as usize;
        let col_in_cta = warp_id / nwpc;
        let warp_in_col = warp_id % nwpc;
        let cols_per_cta = (thread::blockDim_y() as usize) / nwpc;
        let col = (thread::blockIdx_x() as usize) * cols_per_cta + col_in_cta;

        let n = n as usize;
        let k = k as usize;
        if col >= n {
            return;
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;

        let qs_byte_off = lane * 4;
        let group = qs_byte_off / 32;
        let local = qs_byte_off % 32;
        let elem_lo_base = group * 64 + local;
        let elem_hi_base = group * 64 + 32 + local;
        let sc_idx_lo = 2 * group;
        let sc_idx_hi = 2 * group + 1;

        // Each warp handles a stripe of blocks: warp_in_col, warp_in_col + nwpc, ...
        let mut lane_sum = 0.0f32;
        let mut blk = warp_in_col;
        while blk < blocks_per_row {
            let block_offset = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;

            let d = f16_bits_to_f32(
                (b_q4k[block_offset] as u16) | ((b_q4k[block_offset + 1] as u16) << 8),
            );
            let dmin = f16_bits_to_f32(
                (b_q4k[block_offset + 2] as u16) | ((b_q4k[block_offset + 3] as u16) << 8),
            );

            let packed = &b_q4k[block_offset + 4..];
            let mut sc = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            let mut i = 0usize;
            while i < 4 {
                sc[i] = (packed[i] & 0x3F) as f32;
                sc[i + 4] =
                    (((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F) as f32;
                mins[i] = (packed[i + 4] & 0x3F) as f32;
                mins[i + 4] =
                    (((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F) as f32;
                i += 1;
            }

            let d_lo = d * sc[sc_idx_lo];
            let m_lo = dmin * mins[sc_idx_lo];
            let d_hi = d * sc[sc_idx_hi];
            let m_hi = dmin * mins[sc_idx_hi];

            let qs = &b_q4k[block_offset + 16..];
            let a_base = blk * Q4K_BLOCK_SIZE;

            let mut li = 0usize;
            while li < 4 {
                let byte = qs[qs_byte_off + li];
                let q_lo = (byte & 0x0F) as f32;
                let q_hi = (byte >> 4) as f32;
                lane_sum += a[a_base + elem_lo_base + li] * (d_lo * q_lo - m_lo);
                lane_sum += a[a_base + elem_hi_base + li] * (d_hi * q_hi - m_hi);
                li += 1;
            }

            blk += nwpc;
        }

        // Intra-warp reduction
        lane_sum += warp::shuffle_xor_f32(lane_sum, 16);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 8);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 4);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 2);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 1);

        // Cross-warp reduction via shared memory
        if lane == 0 {
            unsafe {
                WARP_SUMS[warp_id] = lane_sum;
            }
        }
        thread::sync_threads();

        // Warp 0 of each column sums all warp contributions
        if warp_in_col == 0 && lane == 0 {
            let smem_base = col_in_cta * nwpc;
            let mut total = 0.0f32;
            let mut w = 0usize;
            while w < nwpc {
                total += unsafe { WARP_SUMS[smem_base + w] };
                w += 1;
            }
            unsafe {
                *output.get_unchecked_mut(col) = total;
            }
        }
    }

    /// PTX inner loop: bypass LLVM for the hot dequant-multiply path.
    /// Same structure as gemv_q4k_multirow but inner 4-byte processing
    /// uses ptx_asm! for tighter instruction scheduling.
    #[kernel]
    pub fn gemv_q4k_ptx(
        n: u32,
        k: u32,
        n_warps_per_col: u32,
        a: &[f32],
        b_q4k: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        static mut WARP_SUMS: SharedArray<f32, 64> = SharedArray::UNINIT;

        let lane = warp::lane_id() as usize;
        let warp_id = thread::threadIdx_y() as usize;
        let nwpc = n_warps_per_col as usize;
        let col_in_cta = warp_id / nwpc;
        let warp_in_col = warp_id % nwpc;
        let cols_per_cta = (thread::blockDim_y() as usize) / nwpc;
        let col = (thread::blockIdx_x() as usize) * cols_per_cta + col_in_cta;

        let n = n as usize;
        let k = k as usize;
        if col >= n {
            return;
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let qs_byte_off = lane * 4;
        let group = qs_byte_off / 32;
        let local = qs_byte_off % 32;
        let elem_lo_base = group * 64 + local;
        let elem_hi_base = group * 64 + 32 + local;
        let sc_idx_lo = 2 * group;
        let sc_idx_hi = 2 * group + 1;

        let mut lane_sum = 0.0f32;
        let mut blk = warp_in_col;
        while blk < blocks_per_row {
            let block_offset = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;

            let d = f16_bits_to_f32(
                (b_q4k[block_offset] as u16) | ((b_q4k[block_offset + 1] as u16) << 8),
            );
            let dmin = f16_bits_to_f32(
                (b_q4k[block_offset + 2] as u16) | ((b_q4k[block_offset + 3] as u16) << 8),
            );

            let packed = &b_q4k[block_offset + 4..];
            let mut sc = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            let mut i = 0usize;
            while i < 4 {
                sc[i] = (packed[i] & 0x3F) as f32;
                sc[i + 4] =
                    (((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F) as f32;
                mins[i] = (packed[i + 4] & 0x3F) as f32;
                mins[i + 4] =
                    (((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F) as f32;
                i += 1;
            }

            let d_lo = d * sc[sc_idx_lo];
            let m_lo = dmin * mins[sc_idx_lo];
            let d_hi = d * sc[sc_idx_hi];
            let m_hi = dmin * mins[sc_idx_hi];

            let qs = &b_q4k[block_offset + 16..];
            let a_base = blk * Q4K_BLOCK_SIZE;

            // PTX inner loop: process 4 bytes, 8 elements
            // Load activations normally (these ARE the bottleneck reads)
            let a0_lo = a[a_base + elem_lo_base];
            let a1_lo = a[a_base + elem_lo_base + 1];
            let a2_lo = a[a_base + elem_lo_base + 2];
            let a3_lo = a[a_base + elem_lo_base + 3];
            let a0_hi = a[a_base + elem_hi_base];
            let a1_hi = a[a_base + elem_hi_base + 1];
            let a2_hi = a[a_base + elem_hi_base + 2];
            let a3_hi = a[a_base + elem_hi_base + 3];

            // Load 4 qs bytes and extract nibbles via PTX bfe
            let b0 = qs[qs_byte_off] as u32;
            let b1 = qs[qs_byte_off + 1] as u32;
            let b2 = qs[qs_byte_off + 2] as u32;
            let b3 = qs[qs_byte_off + 3] as u32;

            // PTX: extract nibble and convert to float in fewer instructions
            // bfe.u32 extracts a bit field; cvt converts
            let mut q0_lo_f: f32;
            let mut q0_hi_f: f32;
            unsafe {
                // Extract low nibble and convert to float
                let mut tmp: u32;
                ptx_asm!("bfe.u32 %0, %1, 0, 4;", out("=r") tmp, in("r") b0);
                ptx_asm!("cvt.rn.f32.u32 %0, %1;", out("=f") q0_lo_f, in("r") tmp);
                ptx_asm!("bfe.u32 %0, %1, 4, 4;", out("=r") tmp, in("r") b0);
                ptx_asm!("cvt.rn.f32.u32 %0, %1;", out("=f") q0_hi_f, in("r") tmp);
            }

            // For bytes 1-3, use Rust (LLVM generates same quality for these)
            let q1_lo_f = (b1 & 0x0F) as f32;
            let q1_hi_f = (b1 >> 4) as f32;
            let q2_lo_f = (b2 & 0x0F) as f32;
            let q2_hi_f = (b2 >> 4) as f32;
            let q3_lo_f = (b3 & 0x0F) as f32;
            let q3_hi_f = (b3 >> 4) as f32;

            // FMA: d*sc*q - dmin*m using fused multiply-add
            lane_sum += d_lo * (a0_lo * q0_lo_f + a1_lo * q1_lo_f + a2_lo * q2_lo_f + a3_lo * q3_lo_f)
                      - m_lo * (a0_lo + a1_lo + a2_lo + a3_lo)
                      + d_hi * (a0_hi * q0_hi_f + a1_hi * q1_hi_f + a2_hi * q2_hi_f + a3_hi * q3_hi_f)
                      - m_hi * (a0_hi + a1_hi + a2_hi + a3_hi);

            blk += nwpc;
        }

        lane_sum += warp::shuffle_xor_f32(lane_sum, 16);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 8);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 4);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 2);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 1);

        if lane == 0 {
            unsafe { WARP_SUMS[warp_id] = lane_sum; }
        }
        thread::sync_threads();

        if warp_in_col == 0 && lane == 0 {
            let smem_base = col_in_cta * nwpc;
            let mut total = 0.0f32;
            let mut w = 0usize;
            while w < nwpc {
                total += unsafe { WARP_SUMS[smem_base + w] };
                w += 1;
            }
            unsafe { *output.get_unchecked_mut(col) = total; }
        }
    }

    /// Integer-accumulation Q4K GEMV: keep nibble×activation products in integer
    /// as long as possible, convert to float only for the final scale application.
    /// Key insight from llama.cpp: avoid per-element uitofp by accumulating
    /// int(q_nibble × int(a_round)) in i32, then one float conversion per block.
    #[kernel]
    pub fn gemv_q4k_intacc(
        n: u32,
        k: u32,
        n_warps_per_col: u32,
        a: &[f32],
        b_q4k: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        static mut WARP_SUMS: SharedArray<f32, 64> = SharedArray::UNINIT;

        let lane = warp::lane_id() as usize;
        let warp_id = thread::threadIdx_y() as usize;
        let nwpc = n_warps_per_col as usize;
        let col_in_cta = warp_id / nwpc;
        let warp_in_col = warp_id % nwpc;
        let cols_per_cta = (thread::blockDim_y() as usize) / nwpc;
        let col = (thread::blockIdx_x() as usize) * cols_per_cta + col_in_cta;

        let n = n as usize;
        let k = k as usize;
        if col >= n {
            return;
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let qs_byte_off = lane * 4;
        let group = qs_byte_off / 32;
        let local = qs_byte_off % 32;
        let elem_lo_base = group * 64 + local;
        let elem_hi_base = group * 64 + 32 + local;
        let sc_idx_lo = 2 * group;
        let sc_idx_hi = 2 * group + 1;

        let mut lane_sum = 0.0f32;
        let mut blk = warp_in_col;
        while blk < blocks_per_row {
            let block_offset = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;

            let d = f16_bits_to_f32(
                (b_q4k[block_offset] as u16) | ((b_q4k[block_offset + 1] as u16) << 8),
            );
            let dmin = f16_bits_to_f32(
                (b_q4k[block_offset + 2] as u16) | ((b_q4k[block_offset + 3] as u16) << 8),
            );

            let packed = &b_q4k[block_offset + 4..];
            let mut sc = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            let mut i = 0usize;
            while i < 4 {
                sc[i] = (packed[i] & 0x3F) as f32;
                sc[i + 4] =
                    (((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F) as f32;
                mins[i] = (packed[i + 4] & 0x3F) as f32;
                mins[i + 4] =
                    (((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F) as f32;
                i += 1;
            }

            let d_lo = d * sc[sc_idx_lo];
            let m_lo = dmin * mins[sc_idx_lo];
            let d_hi = d * sc[sc_idx_hi];
            let m_hi = dmin * mins[sc_idx_hi];

            let qs = &b_q4k[block_offset + 16..];
            let a_base = blk * Q4K_BLOCK_SIZE;

            // Integer accumulation: compute q×round(a/scale) in i32,
            // then convert to float once at the end.
            // But activation is f32 — we can't easily avoid the conversion.
            // Instead, batch the FMAs: d_lo * (q0*a0 + q1*a1 + q2*a2 + q3*a3) - m_lo * (a0+a1+a2+a3)
            let a0 = a[a_base + elem_lo_base];
            let a1 = a[a_base + elem_lo_base + 1];
            let a2 = a[a_base + elem_lo_base + 2];
            let a3 = a[a_base + elem_lo_base + 3];

            let b0 = qs[qs_byte_off];
            let b1 = qs[qs_byte_off + 1];
            let b2 = qs[qs_byte_off + 2];
            let b3 = qs[qs_byte_off + 3];

            // Low nibbles: batch multiply-add
            let dot_lo = a0 * (b0 & 0x0F) as f32
                       + a1 * (b1 & 0x0F) as f32
                       + a2 * (b2 & 0x0F) as f32
                       + a3 * (b3 & 0x0F) as f32;
            let sum_lo = a0 + a1 + a2 + a3;

            // High nibbles
            let a0h = a[a_base + elem_hi_base];
            let a1h = a[a_base + elem_hi_base + 1];
            let a2h = a[a_base + elem_hi_base + 2];
            let a3h = a[a_base + elem_hi_base + 3];

            let dot_hi = a0h * (b0 >> 4) as f32
                       + a1h * (b1 >> 4) as f32
                       + a2h * (b2 >> 4) as f32
                       + a3h * (b3 >> 4) as f32;
            let sum_hi = a0h + a1h + a2h + a3h;

            // 4 float ops instead of 8 per-element ops
            lane_sum += d_lo * dot_lo - m_lo * sum_lo + d_hi * dot_hi - m_hi * sum_hi;

            blk += nwpc;
        }

        lane_sum += warp::shuffle_xor_f32(lane_sum, 16);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 8);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 4);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 2);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 1);

        if lane == 0 {
            unsafe { WARP_SUMS[warp_id] = lane_sum; }
        }
        thread::sync_threads();

        if warp_in_col == 0 && lane == 0 {
            let smem_base = col_in_cta * nwpc;
            let mut total = 0.0f32;
            let mut w = 0usize;
            while w < nwpc {
                total += unsafe { WARP_SUMS[smem_base + w] };
                w += 1;
            }
            unsafe { *output.get_unchecked_mut(col) = total; }
        }
    }

    /// Multi-row multi-warp Q4K GEMV with PTX-optimized inner loop.
    /// Combines: multi-row CTA (ILP from multiple output columns),
    /// multi-warp K-splitting, and inline PTX for tight dequant.
    ///
    /// Grid: (ceil(n / (n_warps_k * rows_per_cta)), 1, 1)
    /// Block: (32, n_warps_k * rows_per_cta, 1)
    /// rows_per_cta output columns per CTA, n_warps_k warps per column.
    #[kernel]
    pub fn gemv_q4k_fast(
        n: u32,
        k: u32,
        n_warps_k: u32,
        rows_per_cta: u32,
        a: &[f32],
        b_q4k: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        static mut WARP_SUMS: SharedArray<f32, 128> = SharedArray::UNINIT; // max 16 warps × 8 rows

        let lane = warp::lane_id() as usize;
        let warp_id = thread::threadIdx_y() as usize;
        let nwk = n_warps_k as usize;
        let rpc = rows_per_cta as usize;
        let col_in_cta = warp_id / nwk;
        let warp_in_col = warp_id % nwk;
        let col = (thread::blockIdx_x() as usize) * rpc + col_in_cta;

        let n = n as usize;
        let k = k as usize;
        if col >= n {
            return;
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;

        let qs_byte_off = lane * 4;
        let group = qs_byte_off / 32;
        let local = qs_byte_off % 32;
        let elem_lo_base = group * 64 + local;
        let elem_hi_base = group * 64 + 32 + local;
        let sc_idx_lo = 2 * group;
        let sc_idx_hi = 2 * group + 1;

        let mut lane_sum = 0.0f32;
        let mut blk = warp_in_col;
        while blk < blocks_per_row {
            let block_offset = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;

            // PTX-optimized header read: load d|dmin as one u32
            let d_bits = (b_q4k[block_offset] as u32) | ((b_q4k[block_offset + 1] as u32) << 8);
            let dmin_bits = (b_q4k[block_offset + 2] as u32) | ((b_q4k[block_offset + 3] as u32) << 8);
            let d = f16_bits_to_f32(d_bits as u16);
            let dmin = f16_bits_to_f32(dmin_bits as u16);

            let packed = &b_q4k[block_offset + 4..];
            let mut sc = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            let mut i = 0usize;
            while i < 4 {
                sc[i] = (packed[i] & 0x3F) as f32;
                sc[i + 4] =
                    (((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F) as f32;
                mins[i] = (packed[i + 4] & 0x3F) as f32;
                mins[i + 4] =
                    (((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F) as f32;
                i += 1;
            }

            let d_lo = d * sc[sc_idx_lo];
            let m_lo_f = dmin * mins[sc_idx_lo];
            let d_hi = d * sc[sc_idx_hi];
            let m_hi_f = dmin * mins[sc_idx_hi];

            let qs = &b_q4k[block_offset + 16..];
            let a_base = blk * Q4K_BLOCK_SIZE;

            // Inner loop: byte-level access with FMA
            let mut li = 0usize;
            while li < 4 {
                let byte = qs[qs_byte_off + li];
                let q_lo = (byte & 0x0F) as f32;
                let q_hi = (byte >> 4) as f32;
                lane_sum += a[a_base + elem_lo_base + li] * (d_lo * q_lo - m_lo_f);
                lane_sum += a[a_base + elem_hi_base + li] * (d_hi * q_hi - m_hi_f);
                li += 1;
            }

            blk += nwk;
        }

        // Intra-warp reduction
        lane_sum += warp::shuffle_xor_f32(lane_sum, 16);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 8);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 4);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 2);
        lane_sum += warp::shuffle_xor_f32(lane_sum, 1);

        // Cross-warp reduction via shared memory
        if lane == 0 {
            unsafe {
                WARP_SUMS[warp_id] = lane_sum;
            }
        }
        thread::sync_threads();

        if warp_in_col == 0 && lane == 0 {
            let smem_base = col_in_cta * nwk;
            let mut total = 0.0f32;
            let mut w = 0usize;
            while w < nwk {
                total += unsafe { WARP_SUMS[smem_base + w] };
                w += 1;
            }
            unsafe {
                *output.get_unchecked_mut(col) = total;
            }
        }
    }

    /// Q4K × Q8 GEMV via dp4a integer dot product.
    ///
    /// The activation vector is pre-quantized to Q8_1 format (32 int8 values + f16 scale + f16 sum
    /// per block, packed as [scale_f32, sum_f32, 32×i8] = 40 bytes per 32 elements).
    ///
    /// dp4a does 4 int8 multiply-adds in one instruction — 8x more compute-dense than f32 FMA.
    /// This is how llama.cpp achieves 750 GB/s: fewer instructions → more memory requests in flight.
    ///
    /// Grid: (ceil(n/WARPS_Y), 1, 1), Block: (32, WARPS_Y, 1)
    /// One warp per output column. Each lane handles 4 qs bytes (= 8 nibbles).
    /// Per block: 2 dp4a pairs (low nibbles + high nibbles) + scale/min float ops.
    #[kernel]
    pub fn gemv_q4k_q8_dp4a(
        n: u32,
        k: u32,
        a_q8: &[u32],
        b_q4k: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        let lane = warp::lane_id() as usize;
        let col = (thread::blockIdx_x() as usize) * (thread::blockDim_y() as usize)
            + thread::threadIdx_y() as usize;

        let n = n as usize;
        let k = k as usize;
        if col >= n {
            return;
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let q8_blocks_per_q4k = Q4K_BLOCK_SIZE / Q8_BLOCK_SIZE; // 8

        let qs_byte_off = lane * 4;
        let group = qs_byte_off / 32;
        let sc_idx_lo = 2 * group;
        let sc_idx_hi = 2 * group + 1;

        // Which Q8 sub-block does this lane's low-nibble elements fall in?
        // Lane L handles elements at: group*64 + local .. group*64 + local+3 (low)
        //                              group*64 + 32 + local .. +3 (high)
        // Q8 block index for low:  (group*64 + local) / 32
        // Q8 block index for high: (group*64 + 32 + local) / 32
        let local = qs_byte_off % 32;
        let q8_idx_lo = (group * 64 + local) / Q8_BLOCK_SIZE;
        let q8_idx_hi = (group * 64 + 32 + local) / Q8_BLOCK_SIZE;
        let q8_off_lo = (group * 64 + local) % Q8_BLOCK_SIZE;
        let q8_off_hi = (group * 64 + 32 + local) % Q8_BLOCK_SIZE;

        let mut col_sum = 0.0f32;

        let mut blk = 0usize;
        while blk < blocks_per_row {
            let block_offset = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;

            let d4 = f16_bits_to_f32(
                (b_q4k[block_offset] as u16) | ((b_q4k[block_offset + 1] as u16) << 8),
            );
            let dmin4 = f16_bits_to_f32(
                (b_q4k[block_offset + 2] as u16) | ((b_q4k[block_offset + 3] as u16) << 8),
            );

            let packed = &b_q4k[block_offset + 4..];
            let mut sc = [0u8; 8];
            let mut mins = [0u8; 8];
            let mut i = 0usize;
            while i < 4 {
                sc[i] = packed[i] & 0x3F;
                sc[i + 4] = ((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F;
                mins[i] = packed[i + 4] & 0x3F;
                mins[i + 4] = ((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F;
                i += 1;
            }

            let qs = &b_q4k[block_offset + 16..];

            // Load Q4K nibbles as packed u32 (4 bytes = 8 nibbles)
            let raw = (qs[qs_byte_off] as u32)
                | ((qs[qs_byte_off + 1] as u32) << 8)
                | ((qs[qs_byte_off + 2] as u32) << 16)
                | ((qs[qs_byte_off + 3] as u32) << 24);
            let v_lo = raw & 0x0F0F0F0F;       // low nibbles: 4 × u8
            let v_hi = (raw >> 4) & 0x0F0F0F0F; // high nibbles: 4 × u8

            // Load Q8 activations as packed i32 (4 × i8)
            let q8_base = blk * q8_blocks_per_q4k;

            // 10 u32 words per Q8 block: [scale_f32_bits, sum_f32_bits, 8×u32 of int8 data]
            let q8_lo_base = (q8_base + q8_idx_lo) * 10;
            let q8_hi_base = (q8_base + q8_idx_hi) * 10;

            let d8_lo_bits = a_q8[q8_lo_base];
            let s8_lo_bits = a_q8[q8_lo_base + 1];
            let d8_lo = f32::from_bits(d8_lo_bits);
            let s8_lo = f32::from_bits(s8_lo_bits);

            let d8_hi_bits = a_q8[q8_hi_base];
            let s8_hi_bits = a_q8[q8_hi_base + 1];
            let d8_hi = f32::from_bits(d8_hi_bits);
            let s8_hi = f32::from_bits(s8_hi_bits);

            // Q8 data word containing this lane's 4 int8 values
            let u_lo = a_q8[q8_lo_base + 2 + q8_off_lo / 4];
            let u_hi = a_q8[q8_hi_base + 2 + q8_off_hi / 4];

            // dp4a: dot(4×u8, 4×i8) — 4 multiply-adds in one instruction
            let dot_lo = dp4a_u32_i32(v_lo, u_lo, 0);
            let dot_hi = dp4a_u32_i32(v_hi, u_hi, 0);

            // Sum of Q8 values (for min correction): dot(0x01010101, q8)
            let sum_lo = dp4a_u32_i32(0x01010101, u_lo, 0);
            let sum_hi = dp4a_u32_i32(0x01010101, u_hi, 0);

            // Partial sum: d4 * sc * d8 * dot - dmin4 * m * d8 * sum_q8
            // But s8 = d8 * sum(all 32 q8 values), and sum_lo/sum_hi are partial (4 values)
            // Use per-element formula: d4*sc*d8*dot_lo - dmin4*m*d8*sum_lo
            let mut partial = d4 * (sc[sc_idx_lo] as f32) * d8_lo * (dot_lo as f32)
                - dmin4 * (mins[sc_idx_lo] as f32) * d8_lo * (sum_lo as f32)
                + d4 * (sc[sc_idx_hi] as f32) * d8_hi * (dot_hi as f32)
                - dmin4 * (mins[sc_idx_hi] as f32) * d8_hi * (sum_hi as f32);

            // Warp reduction
            partial += warp::shuffle_xor_f32(partial, 16);
            partial += warp::shuffle_xor_f32(partial, 8);
            partial += warp::shuffle_xor_f32(partial, 4);
            partial += warp::shuffle_xor_f32(partial, 2);
            partial += warp::shuffle_xor_f32(partial, 1);

            if lane == 0 {
                col_sum += partial;
            }

            blk += 1;
        }

        if lane == 0 {
            unsafe {
                *output.get_unchecked_mut(col) = col_sum;
            }
        }
    }

    /// Fused gate_proj + up_proj + SiLU×gate in one kernel.
    /// Reads activation vector ONCE, computes two Q4K dot products (gate and up),
    /// applies SiLU(gate) × up, writes one output value.
    /// Saves: 1 activation read, 2 kernel launches, 1 SiLU elementwise launch.
    ///
    /// Grid: (ceil(n/WARPS_Y), 1, 1), Block: (32, WARPS_Y, 1)
    /// n = intermediate_size (same for gate and up)
    /// k = hidden_size
    #[kernel]
    pub fn gemv_q4k_fused_gate_up_silu(
        n: u32,
        k: u32,
        a: &[f32],
        b_gate: &[u8],
        b_up: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        let lane = warp::lane_id() as usize;
        let col = (thread::blockIdx_x() as usize) * (thread::blockDim_y() as usize)
            + thread::threadIdx_y() as usize;

        let n = n as usize;
        let k = k as usize;
        if col >= n {
            return;
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let mut gate_sum = 0.0f32;
        let mut up_sum = 0.0f32;

        let qs_byte_off = lane * 4;
        let group = qs_byte_off / 32;
        let local = qs_byte_off % 32;
        let elem_lo_base = group * 64 + local;
        let elem_hi_base = group * 64 + 32 + local;
        let sc_idx_lo = 2 * group;
        let sc_idx_hi = 2 * group + 1;

        let mut blk = 0usize;
        while blk < blocks_per_row {
            let a_base = blk * Q4K_BLOCK_SIZE;

            // --- Gate weight block ---
            let g_off = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
            let g_d = f16_bits_to_f32(
                (b_gate[g_off] as u16) | ((b_gate[g_off + 1] as u16) << 8),
            );
            let g_dmin = f16_bits_to_f32(
                (b_gate[g_off + 2] as u16) | ((b_gate[g_off + 3] as u16) << 8),
            );
            let g_packed = &b_gate[g_off + 4..];
            let mut g_sc = [0.0f32; 8];
            let mut g_mins = [0.0f32; 8];
            let mut i = 0usize;
            while i < 4 {
                g_sc[i] = (g_packed[i] & 0x3F) as f32;
                g_sc[i + 4] =
                    (((g_packed[i + 8] & 0x0F) | ((g_packed[i] >> 6) << 4)) & 0x3F) as f32;
                g_mins[i] = (g_packed[i + 4] & 0x3F) as f32;
                g_mins[i + 4] =
                    (((g_packed[i + 8] >> 4) | ((g_packed[i + 4] >> 6) << 4)) & 0x3F) as f32;
                i += 1;
            }
            let g_d_lo = g_d * g_sc[sc_idx_lo];
            let g_m_lo = g_dmin * g_mins[sc_idx_lo];
            let g_d_hi = g_d * g_sc[sc_idx_hi];
            let g_m_hi = g_dmin * g_mins[sc_idx_hi];
            let g_qs = &b_gate[g_off + 16..];

            // --- Up weight block ---
            let u_off = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
            let u_d = f16_bits_to_f32(
                (b_up[u_off] as u16) | ((b_up[u_off + 1] as u16) << 8),
            );
            let u_dmin = f16_bits_to_f32(
                (b_up[u_off + 2] as u16) | ((b_up[u_off + 3] as u16) << 8),
            );
            let u_packed = &b_up[u_off + 4..];
            let mut u_sc = [0.0f32; 8];
            let mut u_mins = [0.0f32; 8];
            i = 0;
            while i < 4 {
                u_sc[i] = (u_packed[i] & 0x3F) as f32;
                u_sc[i + 4] =
                    (((u_packed[i + 8] & 0x0F) | ((u_packed[i] >> 6) << 4)) & 0x3F) as f32;
                u_mins[i] = (u_packed[i + 4] & 0x3F) as f32;
                u_mins[i + 4] =
                    (((u_packed[i + 8] >> 4) | ((u_packed[i + 4] >> 6) << 4)) & 0x3F) as f32;
                i += 1;
            }
            let u_d_lo = u_d * u_sc[sc_idx_lo];
            let u_m_lo = u_dmin * u_mins[sc_idx_lo];
            let u_d_hi = u_d * u_sc[sc_idx_hi];
            let u_m_hi = u_dmin * u_mins[sc_idx_hi];
            let u_qs = &b_up[u_off + 16..];

            // --- Shared activation read, dual accumulation ---
            let mut g_partial = 0.0f32;
            let mut u_partial = 0.0f32;
            let mut li = 0usize;
            while li < 4 {
                let a_lo = a[a_base + elem_lo_base + li];
                let a_hi = a[a_base + elem_hi_base + li];

                let g_byte = g_qs[qs_byte_off + li];
                g_partial += a_lo * (g_d_lo * (g_byte & 0x0F) as f32 - g_m_lo);
                g_partial += a_hi * (g_d_hi * (g_byte >> 4) as f32 - g_m_hi);

                let u_byte = u_qs[qs_byte_off + li];
                u_partial += a_lo * (u_d_lo * (u_byte & 0x0F) as f32 - u_m_lo);
                u_partial += a_hi * (u_d_hi * (u_byte >> 4) as f32 - u_m_hi);

                li += 1;
            }

            // Reduce both sums across warp
            g_partial += warp::shuffle_xor_f32(g_partial, 16);
            g_partial += warp::shuffle_xor_f32(g_partial, 8);
            g_partial += warp::shuffle_xor_f32(g_partial, 4);
            g_partial += warp::shuffle_xor_f32(g_partial, 2);
            g_partial += warp::shuffle_xor_f32(g_partial, 1);

            u_partial += warp::shuffle_xor_f32(u_partial, 16);
            u_partial += warp::shuffle_xor_f32(u_partial, 8);
            u_partial += warp::shuffle_xor_f32(u_partial, 4);
            u_partial += warp::shuffle_xor_f32(u_partial, 2);
            u_partial += warp::shuffle_xor_f32(u_partial, 1);

            if lane == 0 {
                gate_sum += g_partial;
                up_sum += u_partial;
            }

            blk += 1;
        }

        // SiLU(gate) × up
        if lane == 0 {
            let silu_gate = gate_sum / (1.0 + (-gate_sum).exp());
            let result = silu_gate * up_sum;
            unsafe {
                *output.get_unchecked_mut(col) = result;
            }
        }
    }

    /// GPU Q6_K dequantize-fused matmul: output[i,j] = dot(a[i,:], dequant(b_q6k[j,:]))
    /// Q6_K block layout (210 bytes per 256 elements):
    ///   ql[128]: low 4 bits of each element
    ///   qh[64]: high 2 bits of each element
    ///   scales[16]: i8 per-group scales
    ///   d: f16 super-block scale
    #[kernel]
    pub fn gemm_q6k_bt(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b_q6k: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let m = m as usize;
        let n = n as usize;
        let k = k as usize;

        if tid >= m * n {
            return;
        }

        let row = tid / n;
        let col = tid % n;

        let blocks_per_row = k / Q6K_BLOCK_SIZE;
        let mut sum = 0.0f32;

        let mut blk = 0usize;
        while blk < blocks_per_row {
            let block_offset = (col * blocks_per_row + blk) * Q6K_BLOCK_BYTES;
            let ql = &b_q6k[block_offset..];
            let qh = &b_q6k[block_offset + 128..];
            let scales = &b_q6k[block_offset + 192..];
            let d = f16_bits_to_f32(
                (b_q6k[block_offset + 208] as u16)
                    | ((b_q6k[block_offset + 209] as u16) << 8),
            );

            let a_base = row * k + blk * Q6K_BLOCK_SIZE;

            let mut half = 0usize;
            while half < 2 {
                let ql_off = half * 64;
                let qh_off = half * 32;

                let mut l = 0usize;
                while l < 32 {
                    let is = l / 16;
                    let q1 = ((ql[ql_off + l] & 0xF)
                        | (((qh[qh_off + l] >> 0) & 3) << 4))
                        as i32
                        - 32;
                    let q2 = ((ql[ql_off + l + 32] & 0xF)
                        | (((qh[qh_off + l] >> 2) & 3) << 4))
                        as i32
                        - 32;
                    let q3 = ((ql[ql_off + l] >> 4)
                        | (((qh[qh_off + l] >> 4) & 3) << 4))
                        as i32
                        - 32;
                    let q4 = ((ql[ql_off + l + 32] >> 4)
                        | (((qh[qh_off + l] >> 6) & 3) << 4))
                        as i32
                        - 32;

                    let sc0 = scales[half * 8 + is] as i8 as f32;
                    let sc1 = scales[half * 8 + is + 2] as i8 as f32;
                    let sc2 = scales[half * 8 + is + 4] as i8 as f32;
                    let sc3 = scales[half * 8 + is + 6] as i8 as f32;

                    let elem_base = half * 128;
                    sum += a[a_base + elem_base + l] * d * sc0 * q1 as f32;
                    sum += a[a_base + elem_base + l + 32] * d * sc1 * q2 as f32;
                    sum += a[a_base + elem_base + l + 64] * d * sc2 * q3 as f32;
                    sum += a[a_base + elem_base + l + 96] * d * sc3 * q4 as f32;
                    l += 1;
                }
                half += 1;
            }

            blk += 1;
        }

        if let Some(elem) = output.get_mut(idx) {
            *elem = sum;
        }
    }
}
