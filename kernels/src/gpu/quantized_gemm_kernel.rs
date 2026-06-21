use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;

const Q4K_BLOCK_SIZE: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;

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
