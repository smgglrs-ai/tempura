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
use ctm_rs::{cuda_module, kernel, thread};
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
    // Grid: (ceil(n_tiles/4), m_tiles, 1) -- ONE TILE PER WARP, no flat-index
    // decomposition and no multi-iteration tile loop. Block: (128, 1, 1) = 4 warps;
    // warp w of block (bx, by) computes tile (tile_m = by, tile_n = bx*4 + w).
    // Tile size: m16 x n8. Accumulates over all k64_tiles using SM120a MMA.
    //
    // WHY one tile per warp: the cutlass-mlir codegen miscompiles loop-carried
    // multi-variable tile walks -- when total tiles exceed the warp slots
    // (m >= 32), ~half the output tiles were left unwritten (stale memory) and
    // the rest stored garbage from mis-carried registers, giving NaN-free
    // run-to-run drift at m=128 while m=16 (single iteration per warp, no
    // carry) stayed bitwise-stable. A 2D grid expresses the tile coordinates
    // directly from blockIdx, so nothing is carried across iterations. At
    // m=16 the warp-to-tile assignment is identical to the previous flat
    // slot-indexed kernel, keeping batch-decode outputs bit-identical.
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
            // C[M][N] f32 row-major. m16n8 accumulator PTX layout:
            //   reg[j]: row = (lane%16) + 8*(j/2), col = (lane/16)*2 + (j%2) [within tile]
            let c_row_base = (tile_m * 16) as usize;
            let c_col_base = (tile_n * 8) as usize;
            let n_usize = n as usize;
            let regs = acc.registers();

            let r0 = c_row_base + lane % 16;
            let r1 = c_row_base + lane % 16 + 8;
            let c0 = c_col_base + (lane / 16) * 2;
            let c1 = c_col_base + (lane / 16) * 2 + 1;

            unsafe {
                *c.add(r0 * n_usize + c0) = regs[0];
                *c.add(r0 * n_usize + c1) = regs[1];
                *c.add(r1 * n_usize + c0) = regs[2];
                *c.add(r1 * n_usize + c1) = regs[3];
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
            if (v - expected).abs() > 1e-3 {
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
            if (v - expected).abs() > 1e-3 {
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

    // --- Test 4: shape bisection + engine-shape GEMM -----------------------
    // Uniform 1.0 data with 1.0 scales: every output must be exactly K. The
    // engine's isolation probe showed run-varying divergence at the prefill
    // shape (m=128, n=4096, k=4096) which smaller shapes do not reproduce;
    // bisect which dimension triggers it, then verify the engine shape
    // end-to-end including a bitwise double-launch determinism check.
    {
        let nibble_one: u8 = 0x2;
        let scale_one: u8 = 0x7F;
        for &(m, n, k) in [
            (112u32, 4096u32, 64u32),
            (128u32, 4088u32, 64u32),
            (128u32, 4104u32, 64u32),
            (128u32, 4096u32, 64u32),
        ]
        .iter()
        {
            let k64_tiles: u32 = k / 64;
            let a_data = build_uniform_e2m1(m as usize, k as usize, nibble_one);
            let b_data = build_uniform_e2m1(n as usize, k as usize, nibble_one);
            let a_scales = build_uniform_scales((m / 16) as usize, k64_tiles as usize, scale_one);
            let b_scales = build_uniform_scales((n / 8) as usize, k64_tiles as usize, scale_one);

            let a_dev = DeviceBuffer::from_host(&stream, &a_data).expect("alloc A");
            let b_dev = DeviceBuffer::from_host(&stream, &b_data).expect("alloc B");
            let a_sc_dev = DeviceBuffer::from_host(&stream, &a_scales).expect("alloc a_scales");
            let b_sc_dev = DeviceBuffer::from_host(&stream, &b_scales).expect("alloc b_scales");

            let c_size = (m * n) as usize;
            let expected = k as f32;
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
            let bad = r1.iter().filter(|v| (**v - expected).abs() > 1e-2).count();
            let first = r1.iter().position(|v| (*v - expected).abs() > 1e-2);

            let c2_dev = DeviceBuffer::from_host(&stream, &vec![f32::NAN; c_size]).expect("alloc C2");
            launch(&c2_dev);
            stream.synchronize().expect("synchronize 2");
            let r2 = c2_dev.to_host_vec(&stream).expect("copy C2");
            let det_diff = r1.iter().zip(&r2).position(|(x, y)| x.to_bits() != y.to_bits());

            println!(
                "shape M={m} N={n} K={k}: wrong={bad}/{c_size} first={first:?} \
                 det_first_diff={det_diff:?} {}",
                if bad == 0 && det_diff.is_none() { "PASS" } else { "FAIL" }
            );
        }
    }
}
