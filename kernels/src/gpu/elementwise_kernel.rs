use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;

/// Portable exp(x) producing bit-identical results on CPU and GPU.
/// Cody-Waite reduction + degree-6 minimax polynomial.
/// No FMA dependency: all operations are explicit add/multiply.
#[inline(always)]
fn portable_exp(x: f32) -> f32 {
    const LN2_HI: f32 = 0.693145752; // upper bits of ln(2)
    const LN2_LO: f32 = 1.428606820e-6; // ln(2) - LN2_HI
    const LOG2E: f32 = 1.442695041;
    // Minimax polynomial coefficients for exp(r)-1 on [-ln2/2, ln2/2]
    const P1: f32 = 1.0;
    const P2: f32 = 0.5;
    const P3: f32 = 0.166666672;
    const P4: f32 = 0.041666653;
    const P5: f32 = 0.008333360;
    const P6: f32 = 0.001388888;

    if x > 88.7 {
        return f32::INFINITY;
    }
    if x < -87.3 {
        return 0.0;
    }

    // Range reduction: x = n*ln(2) + r
    let n = (x * LOG2E + 0.5).floor();
    let ni = n as i32;
    // Cody-Waite: subtract in two parts for exact reduction
    let r = (x - n * LN2_HI) - n * LN2_LO;

    // Horner evaluation: 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120 + r/720)))))
    let mut p = P6;
    p = p * r + P5;
    p = p * r + P4;
    p = p * r + P3;
    p = p * r + P2;
    p = p * r + P1;
    p = p * r + 1.0;

    // Scale by 2^n via bit manipulation
    let scale = f32::from_bits(((ni + 127) as u32) << 23);
    p * scale
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn vector_add(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= a.len() {
            return;
        }
        if let Some(elem) = out.get_mut(idx) {
            *elem = a[i] + b[i];
        }
    }

    #[kernel]
    pub fn silu_mul(gate: &[f32], up: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= gate.len() {
            return;
        }
        let g = gate[i];
        let silu = g / (1.0 + portable_exp(-g));
        if let Some(elem) = out.get_mut(idx) {
            *elem = silu * up[i];
        }
    }

    /// Q8 round-trip: quantize each 32-element block to int8, dequantize back.
    /// One thread per block. Snaps values to 255-level grid.
    /// x is passed as read-only input + writable output (same buffer, split access).
    #[kernel]
    pub fn q8_snap(x_in: &[f32], mut x_out: DisjointSlice<f32>, num_blocks: u32) {
        let idx = thread::index_1d();
        let blk = idx.get();
        if blk >= num_blocks as usize {
            return;
        }
        let start = blk * 32;

        let mut amax = 0.0f32;
        let mut i = 0usize;
        while i < 32 {
            let val = x_in[start + i];
            let a = if val < 0.0 { -val } else { val };
            if a > amax { amax = a; }
            i += 1;
        }

        let scale = amax / 127.0;
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };

        i = 0;
        while i < 32 {
            let val = x_in[start + i];
            let mut q = (val * inv).round();
            if q < -128.0 { q = -128.0; }
            if q > 127.0 { q = 127.0; }
            unsafe { *x_out.get_unchecked_mut(start + i) = (q as i8) as f32 * scale; }
            i += 1;
        }
    }

    /// Quantize f32 activations to packed Q8 format for DP4A matmul.
    /// One thread per 32-element block.
    /// Output layout per block: [scale_f32_bits, sum_f32_bits, 8×u32 packed i8]
    /// = 10 u32 words per block.
    /// x_in: [num_blocks * 32] f32 activations
    /// q8_out: [num_blocks * 10] packed u32
    #[kernel]
    pub fn quantize_q8(x_in: &[f32], mut q8_out: DisjointSlice<u32>, num_blocks: u32) {
        let idx = thread::index_1d();
        let blk = idx.get();
        if blk >= num_blocks as usize {
            return;
        }
        let start = blk * 32;
        let out_base = blk * 10;

        let mut amax = 0.0f32;
        let mut i = 0usize;
        while i < 32 {
            let val = x_in[start + i];
            let a = if val < 0.0 { -val } else { val };
            if a > amax { amax = a; }
            i += 1;
        }

        let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv = if d > 0.0 { 127.0 / amax } else { 0.0 };

        let mut qs = [0i8; 32];
        let mut sum = 0i32;
        i = 0;
        while i < 32 {
            let val = x_in[start + i];
            let mut q = (val * inv).round();
            if q < -128.0 { q = -128.0; }
            if q > 127.0 { q = 127.0; }
            let qi = q as i8;
            qs[i] = qi;
            sum += qi as i32;
            i += 1;
        }

        unsafe {
            *q8_out.get_unchecked_mut(out_base) = d.to_bits();
            *q8_out.get_unchecked_mut(out_base + 1) = (d * sum as f32).to_bits();

            let mut chunk = 0usize;
            while chunk < 8 {
                let word = (qs[chunk * 4] as u8 as u32)
                    | ((qs[chunk * 4 + 1] as u8 as u32) << 8)
                    | ((qs[chunk * 4 + 2] as u8 as u32) << 16)
                    | ((qs[chunk * 4 + 3] as u8 as u32) << 24);
                *q8_out.get_unchecked_mut(out_base + 2 + chunk) = word;
                chunk += 1;
            }
        }
    }

    /// Extract a column range from a row-major matrix.
    /// src: [rows, total_cols], dst: [rows, slice_cols]
    /// Copies src[row, col_offset..col_offset+slice_cols] for each row.
    #[kernel]
    pub fn slice_columns(
        src: &[f32],
        mut dst: DisjointSlice<f32>,
        rows: u32,
        total_cols: u32,
        col_offset: u32,
        slice_cols: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let total = rows as usize * slice_cols as usize;
        if i >= total {
            return;
        }
        let row = i / slice_cols as usize;
        let col = i % slice_cols as usize;
        let src_idx = row * total_cols as usize + col_offset as usize + col;
        if let Some(elem) = dst.get_mut(idx) {
            *elem = src[src_idx];
        }
    }
}
