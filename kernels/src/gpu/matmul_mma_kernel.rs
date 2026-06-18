use cuda_device::{kernel, ptx_asm, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;

const CTA_M: usize = 32;
const CTA_N: usize = 32;
const MMA_K: usize = 16;

const SMEM_A: usize = CTA_M * MMA_K; // 512
const SMEM_B: usize = CTA_N * MMA_K; // 512

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Tiled GEMM using mma.sync m16n8k16: C = A × Bᵀ
    /// A: [m, k] row-major, B: [n, k] row-major (transposed), C: [m, n] row-major
    ///
    /// Launch: blockDim=(32,1,1), gridDim=(ceil(N/32), ceil(M/32), 1)
    /// One warp per CTA. 8 MMA ops per k-tile cover the full 32×32 output.
    #[kernel]
    pub fn sgemm_mma_32x32(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32>,
    ) {
        static mut A_TILE: SharedArray<f32, SMEM_A> = SharedArray::UNINIT;
        static mut B_TILE: SharedArray<f32, SMEM_B> = SharedArray::UNINIT;

        let lane = thread::threadIdx_x() as usize;

        let row_start = thread::blockIdx_y() as usize * CTA_M;
        let col_start = thread::blockIdx_x() as usize * CTA_N;

        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;

        let mut acc_t0 = [0.0f32; 4];
        let mut acc_t1 = [0.0f32; 4];
        let mut acc_t2 = [0.0f32; 4];
        let mut acc_t3 = [0.0f32; 4];
        let mut acc_b0 = [0.0f32; 4];
        let mut acc_b1 = [0.0f32; 4];
        let mut acc_b2 = [0.0f32; 4];
        let mut acc_b3 = [0.0f32; 4];

        let num_k_tiles = k_size.div_ceil(MMA_K);
        let group_id = lane >> 2;
        let tid = lane & 3;

        let mut kt = 0usize;
        while kt < num_k_tiles {
            let k_base = kt * MMA_K;

            let mut load_idx = 0usize;
            while load_idx < 16 {
                let elem = lane * 16 + load_idx;
                let row = elem / MMA_K;
                let col = elem % MMA_K;

                unsafe {
                    let a_row = row_start + row;
                    let a_col = k_base + col;
                    if a_row < m_size && a_col < k_size {
                        A_TILE[elem] = a[a_row * k_size + a_col];
                    } else {
                        A_TILE[elem] = 0.0;
                    }

                    let b_row = col_start + row;
                    let b_col = k_base + col;
                    if b_row < n_size && b_col < k_size {
                        B_TILE[elem] = b[b_row * k_size + b_col];
                    } else {
                        B_TILE[elem] = 0.0;
                    }
                }

                load_idx += 1;
            }

            thread::sync_threads();

            let a_base_top = ((group_id * MMA_K + tid * 2) * 4) as u32;
            let a_base_bot = (((group_id + 16) * MMA_K + tid * 2) * 4) as u32;
            let a_stride8 = (8 * MMA_K * 4) as u32;

            let b_base_0 = ((group_id * MMA_K + tid * 2) * 4) as u32;
            let b_base_1 = (((8 + group_id) * MMA_K + tid * 2) * 4) as u32;
            let b_base_2 = (((16 + group_id) * MMA_K + tid * 2) * 4) as u32;
            let b_base_3 = (((24 + group_id) * MMA_K + tid * 2) * 4) as u32;

            let c_t0 = acc_t0;
            let c_t1 = acc_t1;
            let c_t2 = acc_t2;
            let c_t3 = acc_t3;
            let c_b0 = acc_b0;
            let c_b1 = acc_b1;
            let c_b2 = acc_b2;
            let c_b3 = acc_b3;

            unsafe {
                let a_smem = A_TILE.as_ptr();
                let b_smem = B_TILE.as_ptr();
                mma_from_smem(&mut acc_t0, a_smem, a_base_top, a_stride8, b_smem, b_base_0, c_t0[0], c_t0[1], c_t0[2], c_t0[3]);
                mma_from_smem(&mut acc_t1, a_smem, a_base_top, a_stride8, b_smem, b_base_1, c_t1[0], c_t1[1], c_t1[2], c_t1[3]);
                mma_from_smem(&mut acc_t2, a_smem, a_base_top, a_stride8, b_smem, b_base_2, c_t2[0], c_t2[1], c_t2[2], c_t2[3]);
                mma_from_smem(&mut acc_t3, a_smem, a_base_top, a_stride8, b_smem, b_base_3, c_t3[0], c_t3[1], c_t3[2], c_t3[3]);
                mma_from_smem(&mut acc_b0, a_smem, a_base_bot, a_stride8, b_smem, b_base_0, c_b0[0], c_b0[1], c_b0[2], c_b0[3]);
                mma_from_smem(&mut acc_b1, a_smem, a_base_bot, a_stride8, b_smem, b_base_1, c_b1[0], c_b1[1], c_b1[2], c_b1[3]);
                mma_from_smem(&mut acc_b2, a_smem, a_base_bot, a_stride8, b_smem, b_base_2, c_b2[0], c_b2[1], c_b2[2], c_b2[3]);
                mma_from_smem(&mut acc_b3, a_smem, a_base_bot, a_stride8, b_smem, b_base_3, c_b3[0], c_b3[1], c_b3[2], c_b3[3]);
            }

            thread::sync_threads();
            kt += 1;
        }

        // Write accumulators to global C
        // acc[0]: (groupID, tid*2), acc[1]: (groupID, tid*2+1)
        // acc[2]: (groupID+8, tid*2), acc[3]: (groupID+8, tid*2+1)
        //
        // SAFETY: each thread writes to unique (row, col) positions determined by
        // its lane_id within the warp. groupID and tid partition the 16×8 output
        // tile without overlap, and the CTA-level row/col offsets are block-unique.
        unsafe {
            write_acc(&mut c, &acc_t0, row_start, col_start, 0, 0, group_id, tid, m_size, n_size);
            write_acc(&mut c, &acc_t1, row_start, col_start, 0, 8, group_id, tid, m_size, n_size);
            write_acc(&mut c, &acc_t2, row_start, col_start, 0, 16, group_id, tid, m_size, n_size);
            write_acc(&mut c, &acc_t3, row_start, col_start, 0, 24, group_id, tid, m_size, n_size);
            write_acc(&mut c, &acc_b0, row_start, col_start, 16, 0, group_id, tid, m_size, n_size);
            write_acc(&mut c, &acc_b1, row_start, col_start, 16, 8, group_id, tid, m_size, n_size);
            write_acc(&mut c, &acc_b2, row_start, col_start, 16, 16, group_id, tid, m_size, n_size);
            write_acc(&mut c, &acc_b3, row_start, col_start, 16, 24, group_id, tid, m_size, n_size);
        }
    }

    /// MMA via shared memory pointer: loads A/B fragments from SMEM
    /// inside the asm block to prevent LLVM register aliasing.
    ///
    /// a_base_off: byte offset of A[a_r0, tid*2] from SMEM base
    /// a_stride8: 8 * MMA_K * sizeof(f32) = byte stride for +8 rows
    /// b_base_off: byte offset of B[b_n, tid*2] from SMEM base
    #[inline(always)]
    fn mma_from_smem(
        d: &mut [f32; 4],
        smem_a: *const f32,
        a_base_off: u32,
        a_stride8: u32,
        smem_b: *const f32,
        b_base_off: u32,
        c0: f32, c1: f32, c2: f32, c3: f32,
    ) {
        let mut d0: f32;
        let mut d1: f32;
        let mut d2: f32;
        let mut d3: f32;
        unsafe {
            ptx_asm!(
                "{{ \
                   .reg .b32 %%ra<4>, %%rb<2>; \
                   .reg .f32 %%fa<2>; \
                   .reg .u64 %%addr; \
                   .reg .u32 %%off8; \
                   \
                   cvt.u64.u32 %%addr, %6; \
                   add.u64 %%addr, %4, %%addr; \
                   ld.f32 %%fa0, [%%addr]; \
                   add.u64 %%addr, %%addr, 4; \
                   ld.f32 %%fa1, [%%addr]; \
                   cvt.rn.f16x2.f32 %%ra0, %%fa1, %%fa0; \
                   \
                   cvt.u64.u32 %%addr, %6; \
                   add.u64 %%addr, %4, %%addr; \
                   add.u64 %%addr, %%addr, 32; \
                   ld.f32 %%fa0, [%%addr]; \
                   add.u64 %%addr, %%addr, 4; \
                   ld.f32 %%fa1, [%%addr]; \
                   cvt.rn.f16x2.f32 %%ra1, %%fa1, %%fa0; \
                   \
                   add.u32 %%off8, %6, %7; \
                   cvt.u64.u32 %%addr, %%off8; \
                   add.u64 %%addr, %4, %%addr; \
                   ld.f32 %%fa0, [%%addr]; \
                   add.u64 %%addr, %%addr, 4; \
                   ld.f32 %%fa1, [%%addr]; \
                   cvt.rn.f16x2.f32 %%ra2, %%fa1, %%fa0; \
                   \
                   cvt.u64.u32 %%addr, %%off8; \
                   add.u64 %%addr, %4, %%addr; \
                   add.u64 %%addr, %%addr, 32; \
                   ld.f32 %%fa0, [%%addr]; \
                   add.u64 %%addr, %%addr, 4; \
                   ld.f32 %%fa1, [%%addr]; \
                   cvt.rn.f16x2.f32 %%ra3, %%fa1, %%fa0; \
                   \
                   cvt.u64.u32 %%addr, %8; \
                   add.u64 %%addr, %5, %%addr; \
                   ld.f32 %%fa0, [%%addr]; \
                   add.u64 %%addr, %%addr, 4; \
                   ld.f32 %%fa1, [%%addr]; \
                   cvt.rn.f16x2.f32 %%rb0, %%fa1, %%fa0; \
                   \
                   cvt.u64.u32 %%addr, %8; \
                   add.u64 %%addr, %5, %%addr; \
                   add.u64 %%addr, %%addr, 32; \
                   ld.f32 %%fa0, [%%addr]; \
                   add.u64 %%addr, %%addr, 4; \
                   ld.f32 %%fa1, [%%addr]; \
                   cvt.rn.f16x2.f32 %%rb1, %%fa1, %%fa0; \
                   \
                   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                   {%0,%1,%2,%3}, {%%ra0,%%ra1,%%ra2,%%ra3}, {%%rb0,%%rb1}, {%9,%10,%11,%12}; \
                 }}",
                out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
                in("l") smem_a as u64,
                in("l") smem_b as u64,
                in("r") a_base_off, in("r") a_stride8,
                in("r") b_base_off,
                in("f") c0, in("f") c1, in("f") c2, in("f") c3,
            );
        }
        d[0] = d0;
        d[1] = d1;
        d[2] = d2;
        d[3] = d3;
    }

    /// SAFETY: caller must ensure (row_start + row_offset + groupID{+8}, col_start + col_offset + tid*2{+1})
    /// positions are unique per thread across all write_acc calls for this CTA.
    #[inline(always)]
    unsafe fn write_acc(
        c: &mut DisjointSlice<f32>,
        acc: &[f32; 4],
        row_start: usize,
        col_start: usize,
        row_offset: usize,
        col_offset: usize,
        group_id: usize,
        tid: usize,
        m_size: usize,
        n_size: usize,
    ) {
        let r0 = row_start + row_offset + group_id;
        let r1 = row_start + row_offset + group_id + 8;
        let c0 = col_start + col_offset + tid * 2;
        let c1 = col_start + col_offset + tid * 2 + 1;

        if r0 < m_size && c0 < n_size {
            *c.get_unchecked_mut(r0 * n_size + c0) = acc[0];
        }
        if r0 < m_size && c1 < n_size {
            *c.get_unchecked_mut(r0 * n_size + c1) = acc[1];
        }
        if r1 < m_size && c0 < n_size {
            *c.get_unchecked_mut(r1 * n_size + c0) = acc[2];
        }
        if r1 < m_size && c1 < n_size {
            *c.get_unchecked_mut(r1 * n_size + c1) = acc[3];
        }
    }
}
