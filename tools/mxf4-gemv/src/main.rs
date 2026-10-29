// SM120a MXFP4 tensor-core GEMM: hello-world correctness test + tiled batch kernel.
//
// --- Hello-world kernel ---
// One warp issues one m16n8k64 MXFP4 MMA using the SM120a hardware instruction
// mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32 (block-scaled MXFP4).
//
// Test case: A = all 1.0 E2M1, B = all 1.0 E2M1, scale_a = scale_b = 1.0.
// Expected: every f32 accumulator = 64.0 (sum of 64 products of 1.0 x 1.0).
//
// --- Tiled batch kernel ---
// mxf4_batch_gemm_tiled: stream-K persistent grid, 4 warps/block.
// Grid: (SM_COUNT=170, 1, 1). Block: (128, 1, 1).
// Each warp independently computes one m16xn8 output tile over all k64 tiles.
// Accumulates via mma_m16n8k64_mxf4_f32 tensor cores (SM120a).
//
// A fragment layout (m16n8k64 MXFP4, row-major A, PTX ISA layout):
//   Thread lane t holds 4 u32 registers from A[M16][K64] E2M1 tile:
//     reg[0]: row=t/4,   cols (t%4)*8 ..(t%4)*8+7   (K[0:32] portion)
//     reg[1]: row=t/4+8, cols (t%4)*8 ..(t%4)*8+7
//     reg[2]: row=t/4,   cols (t%4)*8+32..(t%4)*8+39 (K[32:64] portion)
//     reg[3]: row=t/4+8, cols (t%4)*8+32..(t%4)*8+39
//   Each u32 = 8 packed E2M1 nibbles (lo=even nibble, hi=odd).
//
// B fragment layout (m16n8k64 MXFP4, B stored K-major as [N][K/2] bytes):
//   Thread lane t holds 2 u32 registers. B[n][k] stored as n-rows, K/2 bytes/row.
//   Kernel expects B as [N x K/2] bytes (n rows, K/2 bytes/row = K nibbles/row).
//   reg[j]: 8 nibbles from n-col (tile_n*8 + t/4), k-rows starting at (t%4)*8+32*j.
//   Each u32 = one contiguous 4-byte slice (8 nibbles in k-dimension, same n-col).
//
// Scale format: ctm_rs::mma_mxf4 uses byte_id=0, thread_id=0 (broadcast from
// thread 0). All warp lanes use thread-0's scale. => One scale pair per (m16 or
// n8) per k64 tile. a_scales: [M_tiles][k64_tiles][2] u8 (lo=K[0:32], hi=K[32:64]).
//
// Build:
//   cd tools/mxf4-gemv
//   CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide build --arch sm_120a
// Run:
//   ./target/release/mxf4-gemv

use ctm_rs::mma::{Accumulator16x8, Mxf4A, Mxf4B, ScalePairA, ScalePairB, mma_mxf4};
use ctm_rs::{cuda_module, float, kernel, thread};
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};

// UE8M0 byte for scale = 1.0: biased exponent 127 = 0x7F.
// Packed as u16: low byte = scale for K[0..32], high byte = scale for K[32..64].
const SCALE_1_0_PAIR: u16 = 0x7F7F;

// E2M1 nibble for 1.0: sign=0, exp=01 (bias 1 => exponent 0), mantissa=0.
// Encoded as 0b0010 = 0x2. Eight nibbles per u32 = 0x2222_2222.
const MXF4_ALL_ONES: u32 = 0x2222_2222;

// One m16n8k64 MMA tile: 32 lanes x 4 accumulators = 128 f32 output elements.
const OUT_ELEMS: usize = 128;

#[cuda_module]
mod kernels {
    use super::*;

    // silu(x) = x / (1 + exp(-x)), exp via the ex2 approximation.
    fn silu_mul(x: f32, up: f32) -> f32 {
        let e = float::ex2_approx_f32(x * -1.4426950408889634_f32);
        x * (1.0_f32 / (1.0_f32 + e)) * up
    }


    // One warp computes one m16n8k64 MXFP4 MMA tile (hello-world correctness test).
    //
    // # Safety
    // Launch with grid=(1,1,1) block=(32,1,1). `out` writable for 128 f32 elements.
    #[kernel]
    pub unsafe fn mxf4_all_ones(out: *mut f32) {
        let lane = thread::threadIdx_x();

        let a = Mxf4A::from_registers([MXF4_ALL_ONES; 4]);
        let b = Mxf4B::from_registers([MXF4_ALL_ONES; 2]);
        let scale_a = ScalePairA::from_bits(SCALE_1_0_PAIR);
        let scale_b = ScalePairB::from_bits(SCALE_1_0_PAIR);

        let c = unsafe { mma_mxf4(Accumulator16x8::zero(), a, b, scale_a, scale_b) };

        let regs = c.registers();
        let base = unsafe { out.add(lane as usize * 4) };
        unsafe {
            base.write(regs[0]);
            base.add(1).write(regs[1]);
            base.add(2).write(regs[2]);
            base.add(3).write(regs[3]);
        }
    }

    // Tiled MXFP4 batch GEMM: C[M][N] = A[M][K] x B[K][N] (f32 output).
    //
    // Grid: (ceil(n_tiles/4), m_tiles, 1) -- ONE TILE PER WARP. Block:
    // (128, 1, 1) = 4 warps; warp w of block (bx, by) computes tile
    // (tile_m = by, tile_n = bx*4 + w). Tile size: m16 x n8. Accumulates
    // over all k64_tiles using SM120a MMA.
    //
    // HISTORY: this kernel shipped with a wrong hand-derived accumulator
    // store mapping (row = lane%16 + 8*(j/2) overflowed the m16 tile into
    // the cudaMallocAsync pool, columns 4-7 were never written). The OOB
    // rows silently corrupted neighboring tensors at small grids and raised
    // CUDA error 700 at ~896+ blocks, which we misdiagnosed for a day as a
    // cutlass-mlir block-count miscompile (an earlier persistent tile-walk
    // variant was likewise misread as a loop-carry miscompilation). The fix
    // is the canonical cuda-device mma_frag::acc_coords mapping; test 4
    // validates against a random-data CPU reference, which uniform-data
    // tests fundamentally cannot (any store permutation "passes" and NaN
    // comparisons are false).
    //
    // A packing: row-major [M x K/2] u8, 2 E2M1 nibbles per byte.
    //   a_packed[m][k/2] contains nibbles for (m, k) and (m, k+1).
    //
    // B packing: row-major-by-n [N x K/2] u8 (N rows, K/2 bytes/row).
    //   b_packed[n][k/2] contains nibbles for column n at k-rows k, k+1.
    //   This is the transposed/column-major-in-K format needed by the hardware MMA.
    //   Each lane's B registers are 8 consecutive nibbles in the K dimension.
    //
    // a_scales: [M_tiles][k64_tiles][2] u8. Scale pair for each (m16, k64) block.
    // b_scales: [N_tiles][k64_tiles][2] u8. Scale pair for each (n8, k64) block.
    //   lo byte = UE8M0 for K[0:32], hi byte = UE8M0 for K[32:64].
    //   Only thread 0's scale is used (broadcast via byte_id=0, thread_id=0).
    //
    // Accumulator lane mapping (PTX ISA m16n8 layout):
    //   reg[j]: row = (lane%16) + 8*(j/2), col = tile_n*8 + (lane/16)*2 + (j%2)
    //
    // # Safety
    // All pointers are device addresses. m, n must be multiples of 16, 8.
    // k64_tiles >= 1.
    #[kernel]
    pub unsafe fn mxf4_batch_gemm_tiled(
        a_packed: *const u8,
        a_scales: *const u8,
        b_packed: *const u8,
        b_scales: *const u8,
        c: *mut f32,
        m: u32,
        n: u32,
        k64_tiles: u32,
    ) {
        let tid = thread::threadIdx_x();
        let bid_x = thread::blockIdx_x();
        let bid_y = thread::blockIdx_y();
        let warp_id = tid / 32;
        let lane = (tid % 32) as usize;

        let m_tiles = m / 16;
        let n_tiles = n / 8;

        let tile_m = bid_y;
        let tile_n = bid_x * 4 + warp_id;
        if tile_n >= n_tiles {
            return;
        }
            let mut acc = Accumulator16x8::zero();

            let mut k = 0u32;
            while k < k64_tiles {
                // --- Load A fragment from global memory ---
                // A[M][K/2] row-major. Each row has k64_tiles * 32 bytes.
                // Fragment mapping for lane t, reg j (PTX m16n8k64 row-major A):
                //   row   = (t/4) + 8*(j%2)        -- rows 0..7 (j%2=0) or 8..15 (j%2=1)
                //   k_u32 = (t%4) + 4*(j/2)         -- u32 within k64: 0..3 (j/2=0) or 4..7 (j/2=1)
                // Each u32 = 8 packed E2M1 nibbles for 8 consecutive k-columns.
                let a_ptr = a_packed as *const u32;
                // Row stride in u32s: (K/2 bytes) / 4 = k64_tiles * 8 u32s per row.
                let a_row_u32 = (k64_tiles * 8) as usize;
                let k_u32_base = (k * 8) as usize; // first u32 of this k64 tile in the row
                let base_m = (tile_m * 16) as usize;

                let row0 = base_m + lane / 4;
                let row1 = base_m + lane / 4 + 8;
                let col_lo = lane % 4;       // u32 idx within k64 for K[0:31]
                let col_hi = (lane % 4) + 4; // u32 idx within k64 for K[32:63]

                let a_reg = [
                    unsafe { *a_ptr.add(row0 * a_row_u32 + k_u32_base + col_lo) },
                    unsafe { *a_ptr.add(row1 * a_row_u32 + k_u32_base + col_lo) },
                    unsafe { *a_ptr.add(row0 * a_row_u32 + k_u32_base + col_hi) },
                    unsafe { *a_ptr.add(row1 * a_row_u32 + k_u32_base + col_hi) },
                ];
                let a_frag = Mxf4A::from_registers(a_reg);

                // --- Load B fragment from global memory ---
                // B[N][K/2] row-major (N rows = n-columns, K/2 bytes per row).
                // Fragment mapping for lane t, reg j (column-major B, K-rows x N-cols):
                //   n_col = tile_n*8 + t/4           -- which n-column (0..7 within n8 tile)
                //   k_u32 = (t%4) + 4*j              -- u32 index in row (for k-rows k0..k0+7)
                //     where k0 = k64_tile*32 + (t%4)*4 (k-row byte = k64_tile*32 + (t%4)*4)
                //   k64 tile occupies bytes [k64_tile*32 .. k64_tile*32+32] of each n row.
                //   col_lo selects first k32, col_hi selects second k32 within k64.
                let b_ptr = b_packed as *const u32;
                let b_row_u32 = (k64_tiles * 8) as usize; // u32s per n-row
                let n_col = (tile_n * 8) as usize + lane / 4;
                let bk_base = (k * 8) as usize; // first u32 of k64 tile within n-row
                let bk_lo = lane % 4;     // u32 for k-rows (t%4)*8..(t%4)*8+7 in k-group 0
                let bk_hi = (lane % 4) + 4; // k-group 1 (shifted by 32 k-rows)

                let b_reg = [
                    unsafe { *b_ptr.add(n_col * b_row_u32 + bk_base + bk_lo) },
                    unsafe { *b_ptr.add(n_col * b_row_u32 + bk_base + bk_hi) },
                ];
                let b_frag = Mxf4B::from_registers(b_reg);

                // --- Load scales (thread-0 scale is broadcast to all lanes) ---
                // a_scales: [M_tiles][k64_tiles * 2] u8 (lo then hi for each k64)
                let a_scale_base = (tile_m * k64_tiles * 2 + k * 2) as usize;
                let a_lo = unsafe { *a_scales.add(a_scale_base) };
                let a_hi = unsafe { *a_scales.add(a_scale_base + 1) };
                let scale_a = ScalePairA::from_bits((a_lo as u16) | ((a_hi as u16) << 8));

                let b_scale_base = (tile_n * k64_tiles * 2 + k * 2) as usize;
                let b_lo = unsafe { *b_scales.add(b_scale_base) };
                let b_hi = unsafe { *b_scales.add(b_scale_base + 1) };
                let scale_b = ScalePairB::from_bits((b_lo as u16) | ((b_hi as u16) << 8));

                // Issue SM120a m16n8k64 MXFP4 MMA.
                acc = unsafe { mma_mxf4(acc, a_frag, b_frag, scale_a, scale_b) };

                k += 1;
            }

            // --- Store accumulator to C ---
            // C[M][N] f32 row-major. Canonical m16n8 accumulator layout
            // (cuda-device mma_frag::acc_coords):
            //   row = lane/4 + 8*(j>>1)  in [0,16)
            //   col = 2*(lane%4) + (j&1) in [0,8)
            // The original hand-derived mapping (row = lane%16 + 8*(j/2),
            // col = (lane/16)*2 + j%2) overflowed the tile: rows up to 23
            // wrote past the tile into the cudaMallocAsync pool (silent
            // neighbor corruption at small grids, illegal access at ~896+
            // blocks) and columns 4-7 were never written. Validated against
            // a random-data CPU reference; the uniform-data tests could not
            // catch it (any store permutation "passes", and NaN comparisons
            // are false).
            let c_row_base = (tile_m * 16) as usize;
            let c_col_base = (tile_n * 8) as usize;
            let n_usize = n as usize;
            let regs = acc.registers();

            for j in 0..4usize {
                let row = c_row_base + lane / 4 + 8 * (j >> 1);
                let col = c_col_base + 2 * (lane % 4) + (j & 1);
                unsafe {
                    *c.add(row * n_usize + col) = regs[j];
                }
            }
    }

    // Register-blocked variant: each warp computes an m16 x n32 tile (FOUR
    // consecutive n8 tiles). The A fragment -- and its 4 global loads -- is
    // shared by all 4 MMAs of a k64 step, cutting the per-MMA load count
    // from ~10 (the m16n8 kernel: 4 A + 2 B + 4 scale byte-loads per single
    // MMA) to ~4.5, and the A re-read traffic across n-tiles drops 4x.
    // The m16n8 kernel's inner loop is load-issue-bound at 111-127 TFLOPS
    // (nsys, prefill m=128 shapes); this attacks exactly that ratio.
    //
    // Grid: (ceil(n_tiles/16), m_tiles, 1); block (128,1,1) = 4 warps; warp
    // w computes n-tiles [bx*16 + w*4, +4). n must be a multiple of 8; a
    // partial trailing group (n_tiles % 4 != 0) is handled by clamping the
    // invalid tiles' B/scale addresses to tile 0 (Blackwell traps on OOB
    // even under predication) and skipping their stores.
    //
    // Numerics: identical MMA instruction and k-order -> bitwise-equal to
    // mxf4_batch_gemm_tiled for the same inputs (validated by test 4's
    // random-data reference + a direct cross-kernel comparison).
    #[kernel]
    pub unsafe fn mxf4_batch_gemm_tiled_rb4(
        a_packed: *const u8,
        a_scales: *const u8,
        b_packed: *const u8,
        b_scales: *const u8,
        c: *mut f32,
        m: u32,
        n: u32,
        k64_tiles: u32,
    ) {
        let tid = thread::threadIdx_x();
        let bid_x = thread::blockIdx_x();
        let bid_y = thread::blockIdx_y();
        let warp_id = tid / 32;
        let lane = (tid % 32) as usize;

        let m_tiles = m / 16;
        let n_tiles = n / 8;

        let tile_m = bid_y;
        let tile_n0 = bid_x * 16 + warp_id * 4;
        if tile_n0 >= n_tiles {
            return;
        }

        // Per-j tile indices, clamped to a valid tile for address formation.
        let tn = [
            tile_n0 as usize,
            (tile_n0 + 1) as usize,
            (tile_n0 + 2) as usize,
            (tile_n0 + 3) as usize,
        ];
        let n_clamp = (n_tiles - 1) as usize;
        let tnc = [
            tn[0].min(n_clamp),
            tn[1].min(n_clamp),
            tn[2].min(n_clamp),
            tn[3].min(n_clamp),
        ];
        let j_valid = [
            tn[0] < n_tiles as usize,
            tn[1] < n_tiles as usize,
            tn[2] < n_tiles as usize,
            tn[3] < n_tiles as usize,
        ];

        let mut acc0 = Accumulator16x8::zero();
        let mut acc1 = Accumulator16x8::zero();
        let mut acc2 = Accumulator16x8::zero();
        let mut acc3 = Accumulator16x8::zero();

        let mut k = 0u32;
        while k < k64_tiles {
            let a_ptr = a_packed as *const u32;
            let a_row_u32 = (k64_tiles * 8) as usize;
            let k_u32_base = (k * 8) as usize;
            let base_m = (tile_m * 16) as usize;

            let row0 = base_m + lane / 4;
            let row1 = base_m + lane / 4 + 8;
            let col_lo = lane % 4;
            let col_hi = (lane % 4) + 4;

            let a_reg = [
                unsafe { *a_ptr.add(row0 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row1 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row0 * a_row_u32 + k_u32_base + col_hi) },
                unsafe { *a_ptr.add(row1 * a_row_u32 + k_u32_base + col_hi) },
            ];
            let a_frag = Mxf4A::from_registers(a_reg);

            let b_ptr = b_packed as *const u32;
            let b_row_u32 = (k64_tiles * 8) as usize;
            let bk_base = (k * 8) as usize;
            let bk_lo = lane % 4;
            let bk_hi = (lane % 4) + 4;

            let b_scale_ptr = b_scales;
            let bs_base = (k * 2) as usize;

            // A scale pair (shared by all 4 MMAs).
            let a_scale_base = (tile_m * k64_tiles * 2 + k * 2) as usize;
            let a_lo = unsafe { *a_scales.add(a_scale_base) };
            let a_hi = unsafe { *a_scales.add(a_scale_base + 1) };
            let scale_a = ScalePairA::from_bits((a_lo as u16) | ((a_hi as u16) << 8));

            // NOTE: unrolled by hand -- ctm-rs arrays of frags did not
            // index cleanly; 4 explicit blocks keep the register mapping
            // identical to the validated m16n8 kernel.
            let b_reg0 = [
                unsafe { *b_ptr.add((tnc[0] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[0] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag0 = Mxf4B::from_registers(b_reg0);
            let b_lo0 = unsafe { *b_scale_ptr.add(tnc[0] * k64_tiles as usize * 2 + bs_base) };
            let b_hi0 = unsafe { *b_scale_ptr.add(tnc[0] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b0 = ScalePairB::from_bits((b_lo0 as u16) | ((b_hi0 as u16) << 8));
            acc0 = unsafe { mma_mxf4(acc0, a_frag, b_frag0, scale_a, scale_b0) };

            let b_reg1 = [
                unsafe { *b_ptr.add((tnc[1] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[1] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag1 = Mxf4B::from_registers(b_reg1);
            let b_lo1 = unsafe { *b_scale_ptr.add(tnc[1] * k64_tiles as usize * 2 + bs_base) };
            let b_hi1 = unsafe { *b_scale_ptr.add(tnc[1] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b1 = ScalePairB::from_bits((b_lo1 as u16) | ((b_hi1 as u16) << 8));
            acc1 = unsafe { mma_mxf4(acc1, a_frag, b_frag1, scale_a, scale_b1) };

            let b_reg2 = [
                unsafe { *b_ptr.add((tnc[2] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[2] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag2 = Mxf4B::from_registers(b_reg2);
            let b_lo2 = unsafe { *b_scale_ptr.add(tnc[2] * k64_tiles as usize * 2 + bs_base) };
            let b_hi2 = unsafe { *b_scale_ptr.add(tnc[2] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b2 = ScalePairB::from_bits((b_lo2 as u16) | ((b_hi2 as u16) << 8));
            acc2 = unsafe { mma_mxf4(acc2, a_frag, b_frag2, scale_a, scale_b2) };

            let b_reg3 = [
                unsafe { *b_ptr.add((tnc[3] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[3] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag3 = Mxf4B::from_registers(b_reg3);
            let b_lo3 = unsafe { *b_scale_ptr.add(tnc[3] * k64_tiles as usize * 2 + bs_base) };
            let b_hi3 = unsafe { *b_scale_ptr.add(tnc[3] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b3 = ScalePairB::from_bits((b_lo3 as u16) | ((b_hi3 as u16) << 8));
            acc3 = unsafe { mma_mxf4(acc3, a_frag, b_frag3, scale_a, scale_b3) };

            k += 1;
        }

        // Stores: canonical m16n8 accumulator mapping (see the m16n8 kernel's
        // comment), guarded per j for the ragged n tail.
        let c_row_base = (tile_m * 16) as usize;
        let n_usize = n as usize;
        let regs0 = acc0.registers();
        let regs1 = acc1.registers();
        let regs2 = acc2.registers();
        let regs3 = acc3.registers();
        for j in 0..4usize {
            let row = c_row_base + lane / 4 + 8 * (j >> 1);
            let sub = 2 * (lane % 4) + (j & 1);
            if j_valid[0] {
                unsafe { *c.add(row * n_usize + tn[0] * 8 + sub) = regs0[j]; }
            }
            if j_valid[1] {
                unsafe { *c.add(row * n_usize + tn[1] * 8 + sub) = regs1[j]; }
            }
            if j_valid[2] {
                unsafe { *c.add(row * n_usize + tn[2] * 8 + sub) = regs2[j]; }
            }
            if j_valid[3] {
                unsafe { *c.add(row * n_usize + tn[3] * 8 + sub) = regs3[j]; }
            }
        }
    }

    // m32 register-blocked variant: each warp computes an m32 x n32 tile
    // (TWO m16 halves x FOUR n8 tiles = 8 MMAs per k64 step). In addition
    // to rb4's A-fragment sharing across n-tiles, the B fragments are
    // shared across the two m-halves -- the B re-read traffic across
    // m-tiles halves (m_tiles/2 instead of m_tiles), which is the binding
    // limit for the large-n GEMMs after rb4: gate/up [128,14336,4096] re-
    // reads the 29MB B matrix 8x = 235MB per GEMM, ~3.3 TB/s of L2 traffic
    // at 106us. Grid: (ceil(n_tiles/16), m/32); block (128,1,1) = 4 warps;
    // warp w computes n-tiles [bx*16 + w*4, +4) for m rows [by*32, +32).
    // m must be a multiple of 32 (the dispatch gates on it). Bitwise-equal
    // to mxf4_batch_gemm_tiled for the same inputs (same MMA, same k
    // order; validated by test 4's random-data reference + the cross-kernel
    // bit comparison).
    #[kernel]
    pub unsafe fn mxf4_batch_gemm_tiled_rb4m32(
        a_packed: *const u8,
        a_scales: *const u8,
        b_packed: *const u8,
        b_scales: *const u8,
        c: *mut f32,
        m: u32,
        n: u32,
        k64_tiles: u32,
    ) {
        let tid = thread::threadIdx_x();
        let bid_x = thread::blockIdx_x();
        let bid_y = thread::blockIdx_y();
        let warp_id = tid / 32;
        let lane = (tid % 32) as usize;

        let n_tiles = n / 8;

        let tile_m0 = bid_y * 2;
        let tile_n0 = bid_x * 16 + warp_id * 4;
        if tile_n0 >= n_tiles {
            return;
        }

        let tn = [
            tile_n0 as usize,
            (tile_n0 + 1) as usize,
            (tile_n0 + 2) as usize,
            (tile_n0 + 3) as usize,
        ];
        let n_clamp = (n_tiles - 1) as usize;
        let tnc = [
            tn[0].min(n_clamp),
            tn[1].min(n_clamp),
            tn[2].min(n_clamp),
            tn[3].min(n_clamp),
        ];
        let j_valid = [
            tn[0] < n_tiles as usize,
            tn[1] < n_tiles as usize,
            tn[2] < n_tiles as usize,
            tn[3] < n_tiles as usize,
        ];

        // 8 accumulators: [m-half][n-tile].
        let mut a00 = Accumulator16x8::zero();
        let mut a01 = Accumulator16x8::zero();
        let mut a02 = Accumulator16x8::zero();
        let mut a03 = Accumulator16x8::zero();
        let mut a10 = Accumulator16x8::zero();
        let mut a11 = Accumulator16x8::zero();
        let mut a12 = Accumulator16x8::zero();
        let mut a13 = Accumulator16x8::zero();

        let a_ptr = a_packed as *const u32;
        let a_row_u32 = (k64_tiles * 8) as usize;
        let b_ptr = b_packed as *const u32;
        let b_row_u32 = (k64_tiles * 8) as usize;

        let mut k = 0u32;
        while k < k64_tiles {
            let k_u32_base = (k * 8) as usize;
            let row0 = (tile_m0 * 16) as usize + lane / 4;
            let row1 = row0 + 8;
            let row2 = row0 + 16;
            let row3 = row0 + 24;
            let col_lo = lane % 4;
            let col_hi = (lane % 4) + 4;

            // A fragments for both m16 halves.
            let a_reg_lo = [
                unsafe { *a_ptr.add(row0 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row1 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row0 * a_row_u32 + k_u32_base + col_hi) },
                unsafe { *a_ptr.add(row1 * a_row_u32 + k_u32_base + col_hi) },
            ];
            let a_frag_lo = Mxf4A::from_registers(a_reg_lo);
            let a_reg_hi = [
                unsafe { *a_ptr.add(row2 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row3 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row2 * a_row_u32 + k_u32_base + col_hi) },
                unsafe { *a_ptr.add(row3 * a_row_u32 + k_u32_base + col_hi) },
            ];
            let a_frag_hi = Mxf4A::from_registers(a_reg_hi);

            let bk_base = (k * 8) as usize;
            let bk_lo = lane % 4;
            let bk_hi = (lane % 4) + 4;
            let bs_base = (k * 2) as usize;

            // A scale pairs (one per m-half; shared across the 4 n-tiles).
            let as_lo = (tile_m0 * k64_tiles * 2 + k * 2) as usize;
            let as_hi = ((tile_m0 + 1) * k64_tiles * 2 + k * 2) as usize;
            let sa_lo = unsafe { *a_scales.add(as_lo) };
            let sa_hi = unsafe { *a_scales.add(as_lo + 1) };
            let scale_a_lo =
                ScalePairA::from_bits((sa_lo as u16) | ((sa_hi as u16) << 8));
            let sb_lo = unsafe { *a_scales.add(as_hi) };
            let sb_hi = unsafe { *a_scales.add(as_hi + 1) };
            let scale_a_hi =
                ScalePairA::from_bits((sb_lo as u16) | ((sb_hi as u16) << 8));

            // Per-n-tile: B fragment + scale pair, shared by both m-halves.
            let b_reg0 = [
                unsafe { *b_ptr.add((tnc[0] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[0] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag0 = Mxf4B::from_registers(b_reg0);
            let b0_lo = unsafe { *b_scales.add(tnc[0] * k64_tiles as usize * 2 + bs_base) };
            let b0_hi = unsafe { *b_scales.add(tnc[0] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b0 = ScalePairB::from_bits((b0_lo as u16) | ((b0_hi as u16) << 8));
            a00 = unsafe { mma_mxf4(a00, a_frag_lo, b_frag0, scale_a_lo, scale_b0) };
            a10 = unsafe { mma_mxf4(a10, a_frag_hi, b_frag0, scale_a_hi, scale_b0) };

            let b_reg1 = [
                unsafe { *b_ptr.add((tnc[1] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[1] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag1 = Mxf4B::from_registers(b_reg1);
            let b1_lo = unsafe { *b_scales.add(tnc[1] * k64_tiles as usize * 2 + bs_base) };
            let b1_hi = unsafe { *b_scales.add(tnc[1] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b1 = ScalePairB::from_bits((b1_lo as u16) | ((b1_hi as u16) << 8));
            a01 = unsafe { mma_mxf4(a01, a_frag_lo, b_frag1, scale_a_lo, scale_b1) };
            a11 = unsafe { mma_mxf4(a11, a_frag_hi, b_frag1, scale_a_hi, scale_b1) };

            let b_reg2 = [
                unsafe { *b_ptr.add((tnc[2] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[2] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag2 = Mxf4B::from_registers(b_reg2);
            let b2_lo = unsafe { *b_scales.add(tnc[2] * k64_tiles as usize * 2 + bs_base) };
            let b2_hi = unsafe { *b_scales.add(tnc[2] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b2 = ScalePairB::from_bits((b2_lo as u16) | ((b2_hi as u16) << 8));
            a02 = unsafe { mma_mxf4(a02, a_frag_lo, b_frag2, scale_a_lo, scale_b2) };
            a12 = unsafe { mma_mxf4(a12, a_frag_hi, b_frag2, scale_a_hi, scale_b2) };

            let b_reg3 = [
                unsafe { *b_ptr.add((tnc[3] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[3] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag3 = Mxf4B::from_registers(b_reg3);
            let b3_lo = unsafe { *b_scales.add(tnc[3] * k64_tiles as usize * 2 + bs_base) };
            let b3_hi = unsafe { *b_scales.add(tnc[3] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b3 = ScalePairB::from_bits((b3_lo as u16) | ((b3_hi as u16) << 8));
            a03 = unsafe { mma_mxf4(a03, a_frag_lo, b_frag3, scale_a_lo, scale_b3) };
            a13 = unsafe { mma_mxf4(a13, a_frag_hi, b_frag3, scale_a_hi, scale_b3) };

            k += 1;
        }

        // Stores: canonical m16n8 accumulator mapping, 8 tiles.
        let c_row_base = (tile_m0 * 16) as usize;
        let n_usize = n as usize;
        let r0 = a00.registers();
        let r1 = a01.registers();
        let r2 = a02.registers();
        let r3 = a03.registers();
        let r4 = a10.registers();
        let r5 = a11.registers();
        let r6 = a12.registers();
        let r7 = a13.registers();
        for j in 0..4usize {
            let sub = 2 * (lane % 4) + (j & 1);
            let row_lo = c_row_base + lane / 4 + 8 * (j >> 1);
            let row_hi = row_lo + 16;
            if j_valid[0] {
                unsafe { *c.add(row_lo * n_usize + tn[0] * 8 + sub) = r0[j]; }
                unsafe { *c.add(row_hi * n_usize + tn[0] * 8 + sub) = r4[j]; }
            }
            if j_valid[1] {
                unsafe { *c.add(row_lo * n_usize + tn[1] * 8 + sub) = r1[j]; }
                unsafe { *c.add(row_hi * n_usize + tn[1] * 8 + sub) = r5[j]; }
            }
            if j_valid[2] {
                unsafe { *c.add(row_lo * n_usize + tn[2] * 8 + sub) = r2[j]; }
                unsafe { *c.add(row_hi * n_usize + tn[2] * 8 + sub) = r6[j]; }
            }
            if j_valid[3] {
                unsafe { *c.add(row_lo * n_usize + tn[3] * 8 + sub) = r3[j]; }
                unsafe { *c.add(row_hi * n_usize + tn[3] * 8 + sub) = r7[j]; }
            }
        }
    }

    // m32 x n16 variant: each warp computes an m32 x n16 tile
    // (TWO m16 halves x TWO n8 tiles = 4 MMAs per k64 step). Finer grid
    // than rb4m32 (2x the warps) for the small-n/long-k shapes whose
    // rb4m32 grid underfills the SMs and cannot hide DRAM latency:
    // down_proj [128,4096,14336] ran 162.7us in-pipeline vs 114 isolated
    // (nsys, 2026-10-29) at 512 warps = 3/SM.
    // Grid: (ceil(n_tiles/8), m/32); block (128,1,1) = 4 warps; warp w
    // computes n-tiles [bx*8 + w*2, +2) for m rows [by*32, +32). m must be
    // a multiple of 32. Bitwise-equal to mxf4_batch_gemm_tiled (same MMA,
    // same k-order; validated by the random-data reference + cross-kernel
    // bit comparison).
    #[kernel]
    pub unsafe fn mxf4_batch_gemm_tiled_rb2m32(
        a_packed: *const u8,
        a_scales: *const u8,
        b_packed: *const u8,
        b_scales: *const u8,
        c: *mut f32,
        m: u32,
        n: u32,
        k64_tiles: u32,
    ) {
        let tid = thread::threadIdx_x();
        let bid_x = thread::blockIdx_x();
        let bid_y = thread::blockIdx_y();
        let warp_id = tid / 32;
        let lane = (tid % 32) as usize;

        let n_tiles = n / 8;

        let tile_m0 = bid_y * 2;
        let tile_n0 = bid_x * 8 + warp_id * 2;
        if tile_n0 >= n_tiles {
            return;
        }

        let tn = [
            tile_n0 as usize,
            (tile_n0 + 1) as usize,
        ];
        let n_clamp = (n_tiles - 1) as usize;
        let tnc = [
            tn[0].min(n_clamp),
            tn[1].min(n_clamp),
        ];
        let j_valid = [
            tn[0] < n_tiles as usize,
            tn[1] < n_tiles as usize,
        ];

        // 8 accumulators: [m-half][n-tile].
        let mut a00 = Accumulator16x8::zero();
        let mut a01 = Accumulator16x8::zero();

        let mut a10 = Accumulator16x8::zero();
        let mut a11 = Accumulator16x8::zero();


        let a_ptr = a_packed as *const u32;
        let a_row_u32 = (k64_tiles * 8) as usize;
        let b_ptr = b_packed as *const u32;
        let b_row_u32 = (k64_tiles * 8) as usize;

        let mut k = 0u32;
        while k < k64_tiles {
            let k_u32_base = (k * 8) as usize;
            let row0 = (tile_m0 * 16) as usize + lane / 4;
            let row1 = row0 + 8;
            let row2 = row0 + 16;
            let row3 = row0 + 24;
            let col_lo = lane % 4;
            let col_hi = (lane % 4) + 4;

            // A fragments for both m16 halves.
            let a_reg_lo = [
                unsafe { *a_ptr.add(row0 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row1 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row0 * a_row_u32 + k_u32_base + col_hi) },
                unsafe { *a_ptr.add(row1 * a_row_u32 + k_u32_base + col_hi) },
            ];
            let a_frag_lo = Mxf4A::from_registers(a_reg_lo);
            let a_reg_hi = [
                unsafe { *a_ptr.add(row2 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row3 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row2 * a_row_u32 + k_u32_base + col_hi) },
                unsafe { *a_ptr.add(row3 * a_row_u32 + k_u32_base + col_hi) },
            ];
            let a_frag_hi = Mxf4A::from_registers(a_reg_hi);

            let bk_base = (k * 8) as usize;
            let bk_lo = lane % 4;
            let bk_hi = (lane % 4) + 4;
            let bs_base = (k * 2) as usize;

            // A scale pairs (one per m-half; shared across the 4 n-tiles).
            let as_lo = (tile_m0 * k64_tiles * 2 + k * 2) as usize;
            let as_hi = ((tile_m0 + 1) * k64_tiles * 2 + k * 2) as usize;
            let sa_lo = unsafe { *a_scales.add(as_lo) };
            let sa_hi = unsafe { *a_scales.add(as_lo + 1) };
            let scale_a_lo =
                ScalePairA::from_bits((sa_lo as u16) | ((sa_hi as u16) << 8));
            let sb_lo = unsafe { *a_scales.add(as_hi) };
            let sb_hi = unsafe { *a_scales.add(as_hi + 1) };
            let scale_a_hi =
                ScalePairA::from_bits((sb_lo as u16) | ((sb_hi as u16) << 8));

            // Per-n-tile: B fragment + scale pair, shared by both m-halves.
            let b_reg0 = [
                unsafe { *b_ptr.add((tnc[0] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[0] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag0 = Mxf4B::from_registers(b_reg0);
            let b0_lo = unsafe { *b_scales.add(tnc[0] * k64_tiles as usize * 2 + bs_base) };
            let b0_hi = unsafe { *b_scales.add(tnc[0] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b0 = ScalePairB::from_bits((b0_lo as u16) | ((b0_hi as u16) << 8));
            a00 = unsafe { mma_mxf4(a00, a_frag_lo, b_frag0, scale_a_lo, scale_b0) };
            a10 = unsafe { mma_mxf4(a10, a_frag_hi, b_frag0, scale_a_hi, scale_b0) };

            let b_reg1 = [
                unsafe { *b_ptr.add((tnc[1] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[1] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag1 = Mxf4B::from_registers(b_reg1);
            let b1_lo = unsafe { *b_scales.add(tnc[1] * k64_tiles as usize * 2 + bs_base) };
            let b1_hi = unsafe { *b_scales.add(tnc[1] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b1 = ScalePairB::from_bits((b1_lo as u16) | ((b1_hi as u16) << 8));
            a01 = unsafe { mma_mxf4(a01, a_frag_lo, b_frag1, scale_a_lo, scale_b1) };
            a11 = unsafe { mma_mxf4(a11, a_frag_hi, b_frag1, scale_a_hi, scale_b1) };

            k += 1;
        }

        // Stores: canonical m16n8 accumulator mapping, 4 tiles.
        let c_row_base = (tile_m0 * 16) as usize;
        let n_usize = n as usize;
        let r0 = a00.registers();
        let r1 = a01.registers();

        let r4 = a10.registers();
        let r5 = a11.registers();

        for j in 0..4usize {
            let sub = 2 * (lane % 4) + (j & 1);
            let row_lo = c_row_base + lane / 4 + 8 * (j >> 1);
            let row_hi = row_lo + 16;
            if j_valid[0] {
                unsafe { *c.add(row_lo * n_usize + tn[0] * 8 + sub) = r0[j]; }
                unsafe { *c.add(row_hi * n_usize + tn[0] * 8 + sub) = r4[j]; }
            }
            if j_valid[1] {
                unsafe { *c.add(row_lo * n_usize + tn[1] * 8 + sub) = r1[j]; }
                unsafe { *c.add(row_hi * n_usize + tn[1] * 8 + sub) = r5[j]; }
            }
        }
    }
    // rb4m32 + SiLU epilogue (the UP projection): C = silu(gate) * up,
    // where gate (the GATE projection's output, same [m, n] layout) is
    // read in the store epilogue -- eliminates the separate silu_mul
    // kernel's read of the up matrix (14.7MB per layer at m=128) and its
    // kernel boundary. silu via ex2_approx (the same approx family the
    // q4k fused-silu decode kernel uses) -- NOT bitwise-equal to the
    // cutile silu_mul's exp (gated by bench-quality). Base tiling:
    // m32 x n32 per warp (TWO m16 halves x FOUR n8 tiles). In addition
    // to rb4's A-fragment sharing across n-tiles, the B fragments are
    // shared across the two m-halves -- the B re-read traffic across
    // m-tiles halves (m_tiles/2 instead of m_tiles), which is the binding
    // limit for the large-n GEMMs after rb4: gate/up [128,14336,4096] re-
    // reads the 29MB B matrix 8x = 235MB per GEMM, ~3.3 TB/s of L2 traffic
    // at 106us. Grid: (ceil(n_tiles/16), m/32); block (128,1,1) = 4 warps;
    // warp w computes n-tiles [bx*16 + w*4, +4) for m rows [by*32, +32).
    // m must be a multiple of 32 (the dispatch gates on it). Bitwise-equal
    // to mxf4_batch_gemm_tiled for the same inputs (same MMA, same k
    // order; validated by test 4's random-data reference + the cross-kernel
    // bit comparison).
    #[kernel]
    pub unsafe fn mxf4_batch_gemm_tiled_rb4m32_silu(
        a_packed: *const u8,
        a_scales: *const u8,
        b_packed: *const u8,
        b_scales: *const u8,
        c_gate: *const f32,
        c: *mut f32,
        m: u32,
        n: u32,
        k64_tiles: u32,
    ) {
        let tid = thread::threadIdx_x();
        let bid_x = thread::blockIdx_x();
        let bid_y = thread::blockIdx_y();
        let warp_id = tid / 32;
        let lane = (tid % 32) as usize;

        let n_tiles = n / 8;

        let tile_m0 = bid_y * 2;
        let tile_n0 = bid_x * 16 + warp_id * 4;
        if tile_n0 >= n_tiles {
            return;
        }

        let tn = [
            tile_n0 as usize,
            (tile_n0 + 1) as usize,
            (tile_n0 + 2) as usize,
            (tile_n0 + 3) as usize,
        ];
        let n_clamp = (n_tiles - 1) as usize;
        let tnc = [
            tn[0].min(n_clamp),
            tn[1].min(n_clamp),
            tn[2].min(n_clamp),
            tn[3].min(n_clamp),
        ];
        let j_valid = [
            tn[0] < n_tiles as usize,
            tn[1] < n_tiles as usize,
            tn[2] < n_tiles as usize,
            tn[3] < n_tiles as usize,
        ];

        // 8 accumulators: [m-half][n-tile].
        let mut a00 = Accumulator16x8::zero();
        let mut a01 = Accumulator16x8::zero();
        let mut a02 = Accumulator16x8::zero();
        let mut a03 = Accumulator16x8::zero();
        let mut a10 = Accumulator16x8::zero();
        let mut a11 = Accumulator16x8::zero();
        let mut a12 = Accumulator16x8::zero();
        let mut a13 = Accumulator16x8::zero();

        let a_ptr = a_packed as *const u32;
        let a_row_u32 = (k64_tiles * 8) as usize;
        let b_ptr = b_packed as *const u32;
        let b_row_u32 = (k64_tiles * 8) as usize;

        let mut k = 0u32;
        while k < k64_tiles {
            let k_u32_base = (k * 8) as usize;
            let row0 = (tile_m0 * 16) as usize + lane / 4;
            let row1 = row0 + 8;
            let row2 = row0 + 16;
            let row3 = row0 + 24;
            let col_lo = lane % 4;
            let col_hi = (lane % 4) + 4;

            // A fragments for both m16 halves.
            let a_reg_lo = [
                unsafe { *a_ptr.add(row0 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row1 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row0 * a_row_u32 + k_u32_base + col_hi) },
                unsafe { *a_ptr.add(row1 * a_row_u32 + k_u32_base + col_hi) },
            ];
            let a_frag_lo = Mxf4A::from_registers(a_reg_lo);
            let a_reg_hi = [
                unsafe { *a_ptr.add(row2 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row3 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row2 * a_row_u32 + k_u32_base + col_hi) },
                unsafe { *a_ptr.add(row3 * a_row_u32 + k_u32_base + col_hi) },
            ];
            let a_frag_hi = Mxf4A::from_registers(a_reg_hi);

            let bk_base = (k * 8) as usize;
            let bk_lo = lane % 4;
            let bk_hi = (lane % 4) + 4;
            let bs_base = (k * 2) as usize;

            // A scale pairs (one per m-half; shared across the 4 n-tiles).
            let as_lo = (tile_m0 * k64_tiles * 2 + k * 2) as usize;
            let as_hi = ((tile_m0 + 1) * k64_tiles * 2 + k * 2) as usize;
            let sa_lo = unsafe { *a_scales.add(as_lo) };
            let sa_hi = unsafe { *a_scales.add(as_lo + 1) };
            let scale_a_lo =
                ScalePairA::from_bits((sa_lo as u16) | ((sa_hi as u16) << 8));
            let sb_lo = unsafe { *a_scales.add(as_hi) };
            let sb_hi = unsafe { *a_scales.add(as_hi + 1) };
            let scale_a_hi =
                ScalePairA::from_bits((sb_lo as u16) | ((sb_hi as u16) << 8));

            // Per-n-tile: B fragment + scale pair, shared by both m-halves.
            let b_reg0 = [
                unsafe { *b_ptr.add((tnc[0] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[0] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag0 = Mxf4B::from_registers(b_reg0);
            let b0_lo = unsafe { *b_scales.add(tnc[0] * k64_tiles as usize * 2 + bs_base) };
            let b0_hi = unsafe { *b_scales.add(tnc[0] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b0 = ScalePairB::from_bits((b0_lo as u16) | ((b0_hi as u16) << 8));
            a00 = unsafe { mma_mxf4(a00, a_frag_lo, b_frag0, scale_a_lo, scale_b0) };
            a10 = unsafe { mma_mxf4(a10, a_frag_hi, b_frag0, scale_a_hi, scale_b0) };

            let b_reg1 = [
                unsafe { *b_ptr.add((tnc[1] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[1] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag1 = Mxf4B::from_registers(b_reg1);
            let b1_lo = unsafe { *b_scales.add(tnc[1] * k64_tiles as usize * 2 + bs_base) };
            let b1_hi = unsafe { *b_scales.add(tnc[1] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b1 = ScalePairB::from_bits((b1_lo as u16) | ((b1_hi as u16) << 8));
            a01 = unsafe { mma_mxf4(a01, a_frag_lo, b_frag1, scale_a_lo, scale_b1) };
            a11 = unsafe { mma_mxf4(a11, a_frag_hi, b_frag1, scale_a_hi, scale_b1) };

            let b_reg2 = [
                unsafe { *b_ptr.add((tnc[2] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[2] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag2 = Mxf4B::from_registers(b_reg2);
            let b2_lo = unsafe { *b_scales.add(tnc[2] * k64_tiles as usize * 2 + bs_base) };
            let b2_hi = unsafe { *b_scales.add(tnc[2] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b2 = ScalePairB::from_bits((b2_lo as u16) | ((b2_hi as u16) << 8));
            a02 = unsafe { mma_mxf4(a02, a_frag_lo, b_frag2, scale_a_lo, scale_b2) };
            a12 = unsafe { mma_mxf4(a12, a_frag_hi, b_frag2, scale_a_hi, scale_b2) };

            let b_reg3 = [
                unsafe { *b_ptr.add((tnc[3] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[3] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag3 = Mxf4B::from_registers(b_reg3);
            let b3_lo = unsafe { *b_scales.add(tnc[3] * k64_tiles as usize * 2 + bs_base) };
            let b3_hi = unsafe { *b_scales.add(tnc[3] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b3 = ScalePairB::from_bits((b3_lo as u16) | ((b3_hi as u16) << 8));
            a03 = unsafe { mma_mxf4(a03, a_frag_lo, b_frag3, scale_a_lo, scale_b3) };
            a13 = unsafe { mma_mxf4(a13, a_frag_hi, b_frag3, scale_a_hi, scale_b3) };

            k += 1;
        }

        // Stores: canonical m16n8 accumulator mapping, 8 tiles.
        let c_row_base = (tile_m0 * 16) as usize;
        let n_usize = n as usize;
        let r0 = a00.registers();
        let r1 = a01.registers();
        let r2 = a02.registers();
        let r3 = a03.registers();
        let r4 = a10.registers();
        let r5 = a11.registers();
        let r6 = a12.registers();
        let r7 = a13.registers();
        for j in 0..4usize {
            let sub = 2 * (lane % 4) + (j & 1);
            let row_lo = c_row_base + lane / 4 + 8 * (j >> 1);
            let row_hi = row_lo + 16;
            if j_valid[0] {
                unsafe { *c.add(row_lo * n_usize + tn[0] * 8 + sub) = silu_mul(unsafe { *c_gate.add(row_lo * n_usize + tn[0] * 8 + sub) }, r0[j]); }
                unsafe { *c.add(row_hi * n_usize + tn[0] * 8 + sub) = silu_mul(unsafe { *c_gate.add(row_hi * n_usize + tn[0] * 8 + sub) }, r4[j]); }
            }
            if j_valid[1] {
                unsafe { *c.add(row_lo * n_usize + tn[1] * 8 + sub) = silu_mul(unsafe { *c_gate.add(row_lo * n_usize + tn[1] * 8 + sub) }, r1[j]); }
                unsafe { *c.add(row_hi * n_usize + tn[1] * 8 + sub) = silu_mul(unsafe { *c_gate.add(row_hi * n_usize + tn[1] * 8 + sub) }, r5[j]); }
            }
            if j_valid[2] {
                unsafe { *c.add(row_lo * n_usize + tn[2] * 8 + sub) = silu_mul(unsafe { *c_gate.add(row_lo * n_usize + tn[2] * 8 + sub) }, r2[j]); }
                unsafe { *c.add(row_hi * n_usize + tn[2] * 8 + sub) = silu_mul(unsafe { *c_gate.add(row_hi * n_usize + tn[2] * 8 + sub) }, r6[j]); }
            }
            if j_valid[3] {
                unsafe { *c.add(row_lo * n_usize + tn[3] * 8 + sub) = silu_mul(unsafe { *c_gate.add(row_lo * n_usize + tn[3] * 8 + sub) }, r3[j]); }
                unsafe { *c.add(row_hi * n_usize + tn[3] * 8 + sub) = silu_mul(unsafe { *c_gate.add(row_hi * n_usize + tn[3] * 8 + sub) }, r7[j]); }
            }
        }
    }


    // rb2m32 + residual-add epilogue (the DOWN projection): C = acc + res,
    // where res (the residual hidden state, same [m, n] layout and row
    // stride) is read in the store epilogue -- eliminates the separate
    // vector_add kernel and its re-read of the down output. Requires
    // m == m_pad (res only covers the valid rows) and h_pad == n.
    // Base tiling: m32 x n16 per warp.
    // (TWO m16 halves x TWO n8 tiles = 4 MMAs per k64 step). Finer grid
    // than rb4m32 (2x the warps) for the small-n/long-k shapes whose
    // rb4m32 grid underfills the SMs and cannot hide DRAM latency:
    // down_proj [128,4096,14336] ran 162.7us in-pipeline vs 114 isolated
    // (nsys, 2026-10-29) at 512 warps = 3/SM.
    // Grid: (ceil(n_tiles/8), m/32); block (128,1,1) = 4 warps; warp w
    // computes n-tiles [bx*8 + w*2, +2) for m rows [by*32, +32). m must be
    // a multiple of 32. Bitwise-equal to mxf4_batch_gemm_tiled (same MMA,
    // same k-order; validated by the random-data reference + cross-kernel
    // bit comparison).
    #[kernel]
    pub unsafe fn mxf4_batch_gemm_tiled_rb2m32_add(
        a_packed: *const u8,
        a_scales: *const u8,
        b_packed: *const u8,
        b_scales: *const u8,
        c_res: *const f32,
        c: *mut f32,
        m: u32,
        n: u32,
        k64_tiles: u32,
    ) {
        let tid = thread::threadIdx_x();
        let bid_x = thread::blockIdx_x();
        let bid_y = thread::blockIdx_y();
        let warp_id = tid / 32;
        let lane = (tid % 32) as usize;

        let n_tiles = n / 8;

        let tile_m0 = bid_y * 2;
        let tile_n0 = bid_x * 8 + warp_id * 2;
        if tile_n0 >= n_tiles {
            return;
        }

        let tn = [
            tile_n0 as usize,
            (tile_n0 + 1) as usize,
        ];
        let n_clamp = (n_tiles - 1) as usize;
        let tnc = [
            tn[0].min(n_clamp),
            tn[1].min(n_clamp),
        ];
        let j_valid = [
            tn[0] < n_tiles as usize,
            tn[1] < n_tiles as usize,
        ];

        // 8 accumulators: [m-half][n-tile].
        let mut a00 = Accumulator16x8::zero();
        let mut a01 = Accumulator16x8::zero();

        let mut a10 = Accumulator16x8::zero();
        let mut a11 = Accumulator16x8::zero();


        let a_ptr = a_packed as *const u32;
        let a_row_u32 = (k64_tiles * 8) as usize;
        let b_ptr = b_packed as *const u32;
        let b_row_u32 = (k64_tiles * 8) as usize;

        let mut k = 0u32;
        while k < k64_tiles {
            let k_u32_base = (k * 8) as usize;
            let row0 = (tile_m0 * 16) as usize + lane / 4;
            let row1 = row0 + 8;
            let row2 = row0 + 16;
            let row3 = row0 + 24;
            let col_lo = lane % 4;
            let col_hi = (lane % 4) + 4;

            // A fragments for both m16 halves.
            let a_reg_lo = [
                unsafe { *a_ptr.add(row0 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row1 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row0 * a_row_u32 + k_u32_base + col_hi) },
                unsafe { *a_ptr.add(row1 * a_row_u32 + k_u32_base + col_hi) },
            ];
            let a_frag_lo = Mxf4A::from_registers(a_reg_lo);
            let a_reg_hi = [
                unsafe { *a_ptr.add(row2 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row3 * a_row_u32 + k_u32_base + col_lo) },
                unsafe { *a_ptr.add(row2 * a_row_u32 + k_u32_base + col_hi) },
                unsafe { *a_ptr.add(row3 * a_row_u32 + k_u32_base + col_hi) },
            ];
            let a_frag_hi = Mxf4A::from_registers(a_reg_hi);

            let bk_base = (k * 8) as usize;
            let bk_lo = lane % 4;
            let bk_hi = (lane % 4) + 4;
            let bs_base = (k * 2) as usize;

            // A scale pairs (one per m-half; shared across the 4 n-tiles).
            let as_lo = (tile_m0 * k64_tiles * 2 + k * 2) as usize;
            let as_hi = ((tile_m0 + 1) * k64_tiles * 2 + k * 2) as usize;
            let sa_lo = unsafe { *a_scales.add(as_lo) };
            let sa_hi = unsafe { *a_scales.add(as_lo + 1) };
            let scale_a_lo =
                ScalePairA::from_bits((sa_lo as u16) | ((sa_hi as u16) << 8));
            let sb_lo = unsafe { *a_scales.add(as_hi) };
            let sb_hi = unsafe { *a_scales.add(as_hi + 1) };
            let scale_a_hi =
                ScalePairA::from_bits((sb_lo as u16) | ((sb_hi as u16) << 8));

            // Per-n-tile: B fragment + scale pair, shared by both m-halves.
            let b_reg0 = [
                unsafe { *b_ptr.add((tnc[0] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[0] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag0 = Mxf4B::from_registers(b_reg0);
            let b0_lo = unsafe { *b_scales.add(tnc[0] * k64_tiles as usize * 2 + bs_base) };
            let b0_hi = unsafe { *b_scales.add(tnc[0] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b0 = ScalePairB::from_bits((b0_lo as u16) | ((b0_hi as u16) << 8));
            a00 = unsafe { mma_mxf4(a00, a_frag_lo, b_frag0, scale_a_lo, scale_b0) };
            a10 = unsafe { mma_mxf4(a10, a_frag_hi, b_frag0, scale_a_hi, scale_b0) };

            let b_reg1 = [
                unsafe { *b_ptr.add((tnc[1] * 8 + lane / 4) * b_row_u32 + bk_base + bk_lo) },
                unsafe { *b_ptr.add((tnc[1] * 8 + lane / 4) * b_row_u32 + bk_base + bk_hi) },
            ];
            let b_frag1 = Mxf4B::from_registers(b_reg1);
            let b1_lo = unsafe { *b_scales.add(tnc[1] * k64_tiles as usize * 2 + bs_base) };
            let b1_hi = unsafe { *b_scales.add(tnc[1] * k64_tiles as usize * 2 + bs_base + 1) };
            let scale_b1 = ScalePairB::from_bits((b1_lo as u16) | ((b1_hi as u16) << 8));
            a01 = unsafe { mma_mxf4(a01, a_frag_lo, b_frag1, scale_a_lo, scale_b1) };
            a11 = unsafe { mma_mxf4(a11, a_frag_hi, b_frag1, scale_a_hi, scale_b1) };

            k += 1;
        }

        // Stores: canonical m16n8 accumulator mapping, 4 tiles.
        let c_row_base = (tile_m0 * 16) as usize;
        let n_usize = n as usize;
        let r0 = a00.registers();
        let r1 = a01.registers();

        let r4 = a10.registers();
        let r5 = a11.registers();

        for j in 0..4usize {
            let sub = 2 * (lane % 4) + (j & 1);
            let row_lo = c_row_base + lane / 4 + 8 * (j >> 1);
            let row_hi = row_lo + 16;
            if j_valid[0] {
                unsafe { *c.add(row_lo * n_usize + tn[0] * 8 + sub) = r0[j] + unsafe { *c_res.add(row_lo * n_usize + tn[0] * 8 + sub) }; }
                unsafe { *c.add(row_hi * n_usize + tn[0] * 8 + sub) = r4[j] + unsafe { *c_res.add(row_hi * n_usize + tn[0] * 8 + sub) }; }
            }
            if j_valid[1] {
                unsafe { *c.add(row_lo * n_usize + tn[1] * 8 + sub) = r1[j] + unsafe { *c_res.add(row_lo * n_usize + tn[1] * 8 + sub) }; }
                unsafe { *c.add(row_hi * n_usize + tn[1] * 8 + sub) = r5[j] + unsafe { *c_res.add(row_hi * n_usize + tn[1] * 8 + sub) }; }
            }
        }
    }
}

// Pack a uniform E2M1 nibble value into an 8-nibble u32.
fn _pack_u32(nibble: u8) -> u32 {
    let b = (nibble & 0xF) | ((nibble & 0xF) << 4);
    (b as u32) | ((b as u32) << 8) | ((b as u32) << 16) | ((b as u32) << 24)
}

// Build [rows x K/2] byte row-major packed E2M1 matrix, uniform nibble.
fn build_uniform_e2m1(rows: usize, k: usize, nibble: u8) -> Vec<u8> {
    let b = (nibble & 0xF) | ((nibble & 0xF) << 4);
    vec![b; rows * k / 2]
}

// Build [tiles][k64_tiles][2] UE8M0 scale array, uniform value.
fn build_uniform_scales(tiles: usize, k64_tiles: usize, ue8m0: u8) -> Vec<u8> {
    vec![ue8m0; tiles * k64_tiles * 2]
}

fn main() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    let dev_name = ctx.device_name().expect("device name");
    println!("device: {dev_name:?}  sm_{major}{minor}");

    if major < 12 {
        eprintln!("MXFP4 tensor cores require SM120+ (RTX 5090 or newer).");
        std::process::exit(1);
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded CUDA module");

    // --- Test 1: hello-world (one warp, all-ones) ---
    {
        let poison = vec![f32::NAN; OUT_ELEMS];
        let out_dev = DeviceBuffer::from_host(&stream, &poison).expect("alloc output");

        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let out_ptr = out_dev.cu_deviceptr() as *mut f32;
        unsafe { module.mxf4_all_ones(&stream, config, out_ptr) }.expect("hello-world launch");
        stream.synchronize().expect("synchronize");

        let result = out_dev.to_host_vec(&stream).expect("copy to host");
        let expected = 64.0_f32;
        let mut pass = true;
        for (i, &v) in result.iter().enumerate() {
            if v != expected {
                eprintln!("  hello-world FAIL out[{i}] = {v} (expected {expected})");
                pass = false;
            }
        }
        if pass {
            println!(
                "PASS hello-world: all {OUT_ELEMS} accumulators = {expected:.1} \
                 (64 x E2M1(1.0) x E2M1(1.0) x scale(1.0))"
            );
        } else {
            eprintln!("FAIL hello-world.");
            std::process::exit(1);
        }
    }

    // --- Test 2: tiled batch GEMM (single m16n8k64 tile, all-ones) ---
    // M=16, N=8, K=64, one k64 tile.
    // A = all E2M1(1.0), B = all E2M1(1.0), scales = UE8M0(1.0 = 0x7F).
    // Expected: C = 64.0 everywhere (128 f32 outputs).
    {
        let m: u32 = 16;
        let n: u32 = 8;
        let k: u32 = 64;
        let k64_tiles: u32 = 1;
        let nibble_one: u8 = 0x2; // E2M1 1.0
        let scale_one: u8 = 0x7F; // UE8M0 1.0

        // A: [16 x 32] bytes = 16 rows, 64/2=32 bytes/row.
        let a_data = build_uniform_e2m1(m as usize, k as usize, nibble_one);
        // B: [8 x 32] bytes = 8 n-cols, 64/2=32 bytes/row (K nibbles per n-col).
        let b_data = build_uniform_e2m1(n as usize, k as usize, nibble_one);
        // Scales: 1 m16-tile x 1 k64-tile x 2 bytes.
        let a_scales = build_uniform_scales(1, 1, scale_one);
        let b_scales = build_uniform_scales(1, 1, scale_one);

        let a_dev = DeviceBuffer::from_host(&stream, &a_data).expect("alloc A");
        let b_dev = DeviceBuffer::from_host(&stream, &b_data).expect("alloc B");
        let a_sc_dev = DeviceBuffer::from_host(&stream, &a_scales).expect("alloc a_scales");
        let b_sc_dev = DeviceBuffer::from_host(&stream, &b_scales).expect("alloc b_scales");

        let c_size = (m * n) as usize;
        let c_poison = vec![f32::NAN; c_size];
        let c_dev = DeviceBuffer::from_host(&stream, &c_poison).expect("alloc C");

        // One block of 128 threads (4 warps). Warp 0 handles tile (0,0).
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        let a_ptr = a_dev.cu_deviceptr() as *const u8;
        let a_sc_ptr = a_sc_dev.cu_deviceptr() as *const u8;
        let b_ptr = b_dev.cu_deviceptr() as *const u8;
        let b_sc_ptr = b_sc_dev.cu_deviceptr() as *const u8;
        let c_ptr = c_dev.cu_deviceptr() as *mut f32;

        unsafe {
            module
                .mxf4_batch_gemm_tiled(
                    &stream, config, a_ptr, a_sc_ptr, b_ptr, b_sc_ptr, c_ptr, m, n, k64_tiles,
                )
                .expect("tiled launch");
        }
        stream.synchronize().expect("synchronize");

        let result = c_dev.to_host_vec(&stream).expect("copy C");
        let expected = 64.0_f32;
        let mut pass = true;
        for (i, &v) in result.iter().enumerate() {
            if v.is_nan() || (v - expected).abs() > 1e-3 {
                eprintln!("  tiled FAIL C[{i}] = {v} (expected {expected})");
                pass = false;
            }
        }
        if pass {
            println!(
                "PASS tiled: M={m} N={n} K={k} all {c_size} outputs = {expected:.1}"
            );
        } else {
            eprintln!("FAIL tiled: incorrect output.");
            std::process::exit(1);
        }
    }

    // --- Test 3: multi-tile GEMM (m=32, n=16: four m16n8 tiles) ---
    // Regression for the loop-carry miscompilation: when a warp had to walk
    // more than one tile, ~half the output tiles were never written (left as
    // the NaN poison below) and others stored garbage. One-tile-per-warp on
    // the 2D grid must cover every tile; any unwritten element stays NaN and
    // fails the check.
    {
        let m: u32 = 32;
        let n: u32 = 16;
        let k: u32 = 64;
        let k64_tiles: u32 = 1;
        let nibble_one: u8 = 0x2;
        let scale_one: u8 = 0x7F;

        let a_data = build_uniform_e2m1(m as usize, k as usize, nibble_one);
        let b_data = build_uniform_e2m1(n as usize, k as usize, nibble_one);
        let a_scales = build_uniform_scales((m / 16) as usize, 1, scale_one);
        let b_scales = build_uniform_scales((n / 8) as usize, 1, scale_one);

        let a_dev = DeviceBuffer::from_host(&stream, &a_data).expect("alloc A");
        let b_dev = DeviceBuffer::from_host(&stream, &b_data).expect("alloc B");
        let a_sc_dev = DeviceBuffer::from_host(&stream, &a_scales).expect("alloc a_scales");
        let b_sc_dev = DeviceBuffer::from_host(&stream, &b_scales).expect("alloc b_scales");

        let c_size = (m * n) as usize;
        let c_poison = vec![f32::NAN; c_size];
        let c_dev = DeviceBuffer::from_host(&stream, &c_poison).expect("alloc C");

        let config = LaunchConfig {
            grid_dim: ((n / 8).div_ceil(4), m / 16, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        let a_ptr = a_dev.cu_deviceptr() as *const u8;
        let a_sc_ptr = a_sc_dev.cu_deviceptr() as *const u8;
        let b_ptr = b_dev.cu_deviceptr() as *const u8;
        let b_sc_ptr = b_sc_dev.cu_deviceptr() as *const u8;
        let c_ptr = c_dev.cu_deviceptr() as *mut f32;

        unsafe {
            module
                .mxf4_batch_gemm_tiled(
                    &stream, config, a_ptr, a_sc_ptr, b_ptr, b_sc_ptr, c_ptr, m, n, k64_tiles,
                )
                .expect("tiled multi-tile launch");
        }
        stream.synchronize().expect("synchronize");

        let result = c_dev.to_host_vec(&stream).expect("copy C");
        let expected = 64.0_f32;
        let mut pass = true;
        for (i, &v) in result.iter().enumerate() {
            if v.is_nan() || (v - expected).abs() > 1e-3 {
                eprintln!("  tiled multi-tile FAIL C[{i}] = {v} (expected {expected})");
                pass = false;
            }
        }
        if pass {
            println!(
                "PASS tiled multi-tile: M={m} N={n} K={k} all {c_size} outputs = {expected:.1}"
            );
        } else {
            eprintln!("FAIL tiled multi-tile: unwritten or incorrect tiles.");
            std::process::exit(1);
        }
    }

    // --- Test 4: random-data reference validation across shapes ------------
    // Uniform-data tests cannot validate fragment lane mappings (any store
    // permutation "passes" and NaN comparisons are false); this test uses
    // random E2M1 nibbles with uniform 1.0 scales so the CPU reference is a
    // plain dot product. It exists because the original hand-derived store
    // mapping was wrong (row = lane%16 + 8 overflowed the m16 tile into the
    // cudaMallocAsync pool -- silent neighbor corruption at small grids,
    // illegal access at ~896+ blocks -- misdiagnosed for a day as a
    // cutlass-mlir block-count miscompile). Includes the shapes that used
    // to fault plus the engine shape, and a bitwise double-launch check.
    {
        let scale_one: u8 = 0x7F;
        let mut seed = 0x12345678u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        const E2M1_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for &(m, n, k) in [
            (16u32, 8u32, 64u32),
            (16, 4096, 4096),
            (32, 4096, 4096),
            (96, 4096, 64),
            (128, 16, 64),
            (128, 512, 64),
            (112, 4096, 64),   // used to fault (896 blocks)
            (128, 4096, 64),   // used to fault (1024 blocks)
            (128, 4096, 4096), // engine prefill shape
        ]
        .iter()
        {
            let k64_tiles: u32 = k / 64;
            let a_data: Vec<u8> = (0..(m as usize * k as usize / 2))
                .map(|_| (rnd() & 0xFF) as u8)
                .collect();
            let b_data: Vec<u8> = (0..(n as usize * k as usize / 2))
                .map(|_| (rnd() & 0xFF) as u8)
                .collect();
            let a_scales = build_uniform_scales((m / 16) as usize, k64_tiles as usize, scale_one);
            let b_scales = build_uniform_scales((n / 8) as usize, k64_tiles as usize, scale_one);

            // CPU reference: decode nibbles, plain dot (scales are 1.0).
            let decode = |buf: &[u8], row: usize, kk: usize, rowlen_k: usize| -> f32 {
                let byte = buf[row * (rowlen_k / 2) + kk / 2];
                let nib = if kk % 2 == 0 { byte & 0xF } else { byte >> 4 };
                let mag = E2M1_MAG[(nib & 0x7) as usize];
                if nib & 8 != 0 { -mag } else { mag }
            };
            let mut reference = vec![0.0f32; (m * n) as usize];
            for mi in 0..m as usize {
                for ni in 0..n as usize {
                    let mut sum = 0.0f32;
                    for kk in 0..k as usize {
                        sum += decode(&a_data, mi, kk, k as usize)
                            * decode(&b_data, ni, kk, k as usize);
                    }
                    reference[mi * n as usize + ni] = sum;
                }
            }

            let a_dev = DeviceBuffer::from_host(&stream, &a_data).expect("alloc A");
            let b_dev = DeviceBuffer::from_host(&stream, &b_data).expect("alloc B");
            let a_sc_dev = DeviceBuffer::from_host(&stream, &a_scales).expect("alloc a_scales");
            let b_sc_dev = DeviceBuffer::from_host(&stream, &b_scales).expect("alloc b_scales");

            let c_size = (m * n) as usize;
            let launch = |c_dev: &DeviceBuffer<f32>| {
                let config = LaunchConfig {
                    grid_dim: ((n / 8).div_ceil(4), m / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    module
                        .mxf4_batch_gemm_tiled(
                            &stream,
                            config,
                            a_dev.cu_deviceptr() as *const u8,
                            a_sc_dev.cu_deviceptr() as *const u8,
                            b_dev.cu_deviceptr() as *const u8,
                            b_sc_dev.cu_deviceptr() as *const u8,
                            c_dev.cu_deviceptr() as *mut f32,
                            m,
                            n,
                            k64_tiles,
                        )
                        .expect("tiled launch")
                }
            };

            let c_dev = DeviceBuffer::from_host(&stream, &vec![f32::NAN; c_size]).expect("alloc C");
            launch(&c_dev);
            stream.synchronize().expect("synchronize");
            let r1 = c_dev.to_host_vec(&stream).expect("copy C");

            // rb4: same reference, PLUS a bitwise cross-check vs the m16n8
            // kernel (identical MMA and k-order -> must be bit-equal).
            let launch_rb4 = |c_dev: &DeviceBuffer<f32>| {
                let config = LaunchConfig {
                    grid_dim: ((n / 8).div_ceil(16), m / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    module
                        .mxf4_batch_gemm_tiled_rb4(
                            &stream,
                            config,
                            a_dev.cu_deviceptr() as *const u8,
                            a_sc_dev.cu_deviceptr() as *const u8,
                            b_dev.cu_deviceptr() as *const u8,
                            b_sc_dev.cu_deviceptr() as *const u8,
                            c_dev.cu_deviceptr() as *mut f32,
                            m,
                            n,
                            k64_tiles,
                        )
                        .expect("rb4 launch")
                }
            };
            let c_rb = DeviceBuffer::from_host(&stream, &vec![f32::NAN; c_size]).expect("alloc C_rb");
            launch_rb4(&c_rb);
            stream.synchronize().expect("synchronize rb4");
            let rr = c_rb.to_host_vec(&stream).expect("copy C_rb");
            let rb_nan = rr.iter().filter(|v| v.is_nan()).count();
            let rb_over = rr
                .iter()
                .zip(&reference)
                .filter(|(got, want)| {
                    !(**got).is_nan()
                        && ((**got) - (**want)).abs() / (**want).abs().max(1.0) > 1e-3
                })
                .count();
            let rb_xdiff = r1
                .iter()
                .zip(&rr)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();

            // rb4m32 (m32n32 per warp): requires m % 32 == 0.
            let (m32_nan, m32_over, m32_xdiff) = if m % 32 == 0 {
                let launch_rb4m32 = |c_dev: &DeviceBuffer<f32>| {
                    let config = LaunchConfig {
                        grid_dim: ((n / 8).div_ceil(16), m / 32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        module
                            .mxf4_batch_gemm_tiled_rb4m32(
                                &stream,
                                config,
                                a_dev.cu_deviceptr() as *const u8,
                                a_sc_dev.cu_deviceptr() as *const u8,
                                b_dev.cu_deviceptr() as *const u8,
                                b_sc_dev.cu_deviceptr() as *const u8,
                                c_dev.cu_deviceptr() as *mut f32,
                                m,
                                n,
                                k64_tiles,
                            )
                            .expect("rb4m32 launch")
                    }
                };
                let c_m32 =
                    DeviceBuffer::from_host(&stream, &vec![f32::NAN; c_size]).expect("alloc C_m32");
                launch_rb4m32(&c_m32);
                stream.synchronize().expect("synchronize rb4m32");
                let rm = c_m32.to_host_vec(&stream).expect("copy C_m32");
                (
                    rm.iter().filter(|v| v.is_nan()).count(),
                    rm.iter()
                        .zip(&reference)
                        .filter(|(got, want)| {
                            !(**got).is_nan()
                                && ((**got) - (**want)).abs() / (**want).abs().max(1.0) > 1e-3
                        })
                        .count(),
                    rm.iter()
                        .zip(&r1)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count(),
                )
            } else {
                (0, 0, 0)
            };

            // rb2m32 (m32n16/warp, finer grid): also requires m % 32 == 0.
            let (r2m32_nan, r2m32_over, r2m32_xdiff) = if m % 32 == 0 {
                let launch_rb2m32 = |c_dev: &DeviceBuffer<f32>| {
                    let config = LaunchConfig {
                        grid_dim: ((n / 8).div_ceil(8), m / 32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        module
                            .mxf4_batch_gemm_tiled_rb2m32(
                                &stream,
                                config,
                                a_dev.cu_deviceptr() as *const u8,
                                a_sc_dev.cu_deviceptr() as *const u8,
                                b_dev.cu_deviceptr() as *const u8,
                                b_sc_dev.cu_deviceptr() as *const u8,
                                c_dev.cu_deviceptr() as *mut f32,
                                m,
                                n,
                                k64_tiles,
                            )
                            .expect("rb2m32 launch")
                    }
                };
                let c_r2 =
                    DeviceBuffer::from_host(&stream, &vec![f32::NAN; c_size]).expect("alloc C_r2");
                launch_rb2m32(&c_r2);
                stream.synchronize().expect("synchronize rb2m32");
                let rm = c_r2.to_host_vec(&stream).expect("copy C_r2");
                (
                    rm.iter().filter(|v| v.is_nan()).count(),
                    rm.iter()
                        .zip(&reference)
                        .filter(|(got, want)| {
                            !(**got).is_nan()
                                && ((**got) - (**want)).abs() / (**want).abs().max(1.0) > 1e-3
                        })
                        .count(),
                    rm.iter()
                        .zip(&r1)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count(),
                )
            } else {
                (0, 0, 0)
            };

            // Epilogue variants (m % 32 == 0): silu (C = silu(gate) * up) and
            // residual-add (C = up + res). Host references use f64 libm exp;
            // the kernel's ex2 approx differs by ~1ulp on the sigmoid, so the
            // tolerance is relative 1e-3 (same as the GEMM reference).
            let (ep_silu_bad, ep_add_bad) = if m % 32 == 0 {
                let c_size_us = c_size;
                let gate_f32: Vec<f32> = (0..c_size_us)
                    .map(|i| ((rnd() % 2000) as f32) / 1000.0 - 1.0)
                    .collect();
                let res_f32: Vec<f32> = (0..c_size_us)
                    .map(|i| ((rnd() % 2000) as f32) / 1000.0 - 1.0)
                    .collect();
                let gate_dev = DeviceBuffer::from_host(&stream, &gate_f32).expect("alloc gate");
                let res_dev = DeviceBuffer::from_host(&stream, &res_f32).expect("alloc res");

                let launch_ep = |silu: bool, c_dev: &DeviceBuffer<f32>, aux: &DeviceBuffer<f32>| {
                    let config = LaunchConfig {
                        grid_dim: (
                            if silu { (n / 8).div_ceil(16) } else { (n / 8).div_ceil(8) },
                            m / 32,
                            1,
                        ),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    if silu {
                        unsafe {
                            module
                                .mxf4_batch_gemm_tiled_rb4m32_silu(
                                    &stream, config,
                                    a_dev.cu_deviceptr() as *const u8,
                                    a_sc_dev.cu_deviceptr() as *const u8,
                                    b_dev.cu_deviceptr() as *const u8,
                                    b_sc_dev.cu_deviceptr() as *const u8,
                                    aux.cu_deviceptr() as *const f32,
                                    c_dev.cu_deviceptr() as *mut f32,
                                    m, n, k64_tiles,
                                )
                                .expect("rb4m32_silu launch")
                        }
                    } else {
                        unsafe {
                            module
                                .mxf4_batch_gemm_tiled_rb2m32_add(
                                    &stream, config,
                                    a_dev.cu_deviceptr() as *const u8,
                                    a_sc_dev.cu_deviceptr() as *const u8,
                                    b_dev.cu_deviceptr() as *const u8,
                                    b_sc_dev.cu_deviceptr() as *const u8,
                                    aux.cu_deviceptr() as *const f32,
                                    c_dev.cu_deviceptr() as *mut f32,
                                    m, n, k64_tiles,
                                )
                                .expect("rb2m32_add launch")
                        }
                    }
                };

                let c_s = DeviceBuffer::from_host(&stream, &vec![f32::NAN; c_size_us])
                    .expect("alloc C_silu");
                launch_ep(true, &c_s, &gate_dev);
                let c_a = DeviceBuffer::from_host(&stream, &vec![f32::NAN; c_size_us])
                    .expect("alloc C_add");
                launch_ep(false, &c_a, &res_dev);
                stream.synchronize().expect("sync ep");
                let rs = c_s.to_host_vec(&stream).expect("copy C_silu");
                let ra = c_a.to_host_vec(&stream).expect("copy C_add");

                let silu_ref = |x: f32| -> f64 {
                    let xd = x as f64;
                    xd / (1.0 + (-xd).exp())
                };
                let sb = rs
                    .iter()
                    .zip(&reference)
                    .zip(&gate_f32)
                    .filter(|((got, want), g)| {
                        let expect = silu_ref(**g) * (**want as f64);
                        !(**got).is_nan() && ((**got) as f64 - expect).abs() > 1e-3 * expect.abs().max(1.0)
                    })
                    .count();
                let ab = ra
                    .iter()
                    .zip(&reference)
                    .zip(&res_f32)
                    .filter(|((got, want), r)| (**got) != (**want) + (**r))
                    .count();
                (sb, ab)
            } else {
                (0, 0)
            };

            let nan = r1.iter().filter(|v| v.is_nan()).count();
            let over = r1
                .iter()
                .zip(&reference)
                .filter(|(got, want)| {
                    !(**got).is_nan()
                        && ((**got) - (**want)).abs() / (**want).abs().max(1.0) > 1e-3
                })
                .count();

            let c2_dev = DeviceBuffer::from_host(&stream, &vec![f32::NAN; c_size]).expect("alloc C2");
            launch(&c2_dev);
            stream.synchronize().expect("synchronize 2");
            let r2 = c2_dev.to_host_vec(&stream).expect("copy C2");
            let det_diff = r1.iter().zip(&r2).position(|(x, y)| x.to_bits() != y.to_bits());

            println!(
                "shape M={m} N={n} K={k}: blocks={} nan={nan}/{c_size} ref_over_tol={over} \
                 det_first_diff={det_diff:?} rb4[nan={rb_nan} ref_over_tol={rb_over} \
                 xdiff_vs_m16n8={rb_xdiff}] rb4m32[nan={m32_nan} ref_over_tol={m32_over} \
                 xdiff_vs_m16n8={m32_xdiff}] rb2m32[nan={r2m32_nan} ref_over_tol={r2m32_over} \
                 xdiff_vs_m16n8={r2m32_xdiff}] epilogues[silu_bad={ep_silu_bad} \
                 add_bad={ep_add_bad}] {}",
                (n / 8).div_ceil(4) * (m / 16),
                if nan == 0 && over == 0 && det_diff.is_none() && rb_nan == 0 && rb_over == 0
                    && rb_xdiff == 0 && m32_nan == 0 && m32_over == 0 && m32_xdiff == 0
                    && r2m32_nan == 0 && r2m32_over == 0 && r2m32_xdiff == 0
                    && ep_silu_bad == 0 && ep_add_bad == 0
                { "PASS" } else { "FAIL" }
            );
        }
    }

    // --- Benchmark: engine prefill shapes, m16n8 vs rb4 -----------------------
    {
        println!("\n--- benchmark (m=128 prefill shapes) ---");
        for &(m, n, k) in [
            (128u32, 1024u32, 4096u32),   // k/v proj
            (128, 4096, 4096),            // q/o proj
            (128, 14336, 4096),           // gate/up proj
            (128, 4096, 14336),           // down proj
        ]
        .iter()
        {
            let k64_tiles: u32 = k / 64;
            let a_data: Vec<u8> = (0..(m as usize * k as usize / 2))
                .map(|i| (i as u8).wrapping_mul(31))
                .collect();
            let b_data: Vec<u8> = (0..(n as usize * k as usize / 2))
                .map(|i| (i as u8).wrapping_mul(17))
                .collect();
            let scale_one: u8 = 0x7F;
            let a_scales = build_uniform_scales((m / 16) as usize, k64_tiles as usize, scale_one);
            let b_scales = build_uniform_scales((n / 8) as usize, k64_tiles as usize, scale_one);
            let a_dev = DeviceBuffer::from_host(&stream, &a_data).expect("alloc A");
            let b_dev = DeviceBuffer::from_host(&stream, &b_data).expect("alloc B");
            let a_sc_dev = DeviceBuffer::from_host(&stream, &a_scales).expect("alloc a_scales");
            let b_sc_dev = DeviceBuffer::from_host(&stream, &b_scales).expect("alloc b_scales");
            let c_dev =
                DeviceBuffer::from_host(&stream, &vec![0.0f32; (m * n) as usize]).expect("alloc C");

            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let bench = |label: &str, grid_x: u32, variant: u8| {
                let grid_y = if variant >= 2 { m / 32 } else { m / 16 };
                let config = LaunchConfig {
                    grid_dim: (grid_x, grid_y, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let launch_one = || {
                    if variant == 1 {
                        unsafe {
                            module
                                .mxf4_batch_gemm_tiled_rb4(
                                    &stream, config,
                                    a_dev.cu_deviceptr() as *const u8,
                                    a_sc_dev.cu_deviceptr() as *const u8,
                                    b_dev.cu_deviceptr() as *const u8,
                                    b_sc_dev.cu_deviceptr() as *const u8,
                                    c_dev.cu_deviceptr() as *mut f32,
                                    m, n, k64_tiles,
                                )
                                .expect("bench rb4 launch")
                        }
                    } else if variant == 2 {
                        unsafe {
                            module
                                .mxf4_batch_gemm_tiled_rb4m32(
                                    &stream, config,
                                    a_dev.cu_deviceptr() as *const u8,
                                    a_sc_dev.cu_deviceptr() as *const u8,
                                    b_dev.cu_deviceptr() as *const u8,
                                    b_sc_dev.cu_deviceptr() as *const u8,
                                    c_dev.cu_deviceptr() as *mut f32,
                                    m, n, k64_tiles,
                                )
                                .expect("bench rb4m32 launch")
                        }
                    } else if variant == 3 {
                        unsafe {
                            module
                                .mxf4_batch_gemm_tiled_rb2m32(
                                    &stream, config,
                                    a_dev.cu_deviceptr() as *const u8,
                                    a_sc_dev.cu_deviceptr() as *const u8,
                                    b_dev.cu_deviceptr() as *const u8,
                                    b_sc_dev.cu_deviceptr() as *const u8,
                                    c_dev.cu_deviceptr() as *mut f32,
                                    m, n, k64_tiles,
                                )
                                .expect("bench rb2m32 launch")
                        }
                    } else {
                        unsafe {
                            module
                                .mxf4_batch_gemm_tiled(
                                    &stream, config,
                                    a_dev.cu_deviceptr() as *const u8,
                                    a_sc_dev.cu_deviceptr() as *const u8,
                                    b_dev.cu_deviceptr() as *const u8,
                                    b_sc_dev.cu_deviceptr() as *const u8,
                                    c_dev.cu_deviceptr() as *mut f32,
                                    m, n, k64_tiles,
                                )
                                .expect("bench launch")
                        }
                    }
                };
                for _ in 0..3 {
                    launch_one();
                    stream.synchronize().expect("sync");
                }
                let iters = 50u32;
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    launch_one();
                }
                stream.synchronize().expect("bench sync");
                let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
                let tflops = flops / (us * 1e-6) / 1e12;
                println!("  {label:<10} M={m} N={n} K={k}: {us:8.1} us  {tflops:6.1} TFLOPS");
            };
            bench("m16n8", (n / 8).div_ceil(4), 0);
            bench("rb4", (n / 8).div_ceil(16), 1);
            bench("rb4m32", (n / 8).div_ceil(16), 2);
            bench("rb2m32", (n / 8).div_ceil(8), 3);
        }
    }
}
