use cuda_device::{kernel, ptx_asm, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;

const CTA_M: usize = 32;
const CTA_N: usize = 32;
const MMA_K: usize = 16;

const SMEM_A: usize = CTA_M * MMA_K; // 512
const SMEM_B: usize = CTA_N * MMA_K; // 512

#[inline(always)]
fn pack_f16x2(lo: f32, hi: f32) -> u32 {
    let result: u32;
    unsafe {
        ptx_asm!(
            "cvt.rn.f16x2.f32 %0, %2, %1;",
            out("=r") result,
            in("f") lo,
            in("f") hi,
        );
    }
    result
}

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

            // Pack A fragments from SMEM
            let at_r0 = group_id;
            let at_r1 = group_id + 8;
            let ab_r0 = group_id + 16;
            let ab_r1 = group_id + 24;

            let a_frag_top: [u32; 4];
            let a_frag_bot: [u32; 4];
            unsafe {
                // PTX ISA Fig.78: reg order is {a0,a1}=top/K0, {a2,a3}=bot/K0,
                // {a4,a5}=top/K1, {a6,a7}=bot/K1. Pack swaps: odd col first.
                a_frag_top = [
                    pack_f16x2(A_TILE[at_r0 * MMA_K + tid * 2 + 1], A_TILE[at_r0 * MMA_K + tid * 2]),
                    pack_f16x2(A_TILE[at_r1 * MMA_K + tid * 2 + 1], A_TILE[at_r1 * MMA_K + tid * 2]),
                    pack_f16x2(A_TILE[at_r0 * MMA_K + tid * 2 + 9], A_TILE[at_r0 * MMA_K + tid * 2 + 8]),
                    pack_f16x2(A_TILE[at_r1 * MMA_K + tid * 2 + 9], A_TILE[at_r1 * MMA_K + tid * 2 + 8]),
                ];
                a_frag_bot = [
                    pack_f16x2(A_TILE[ab_r0 * MMA_K + tid * 2 + 1], A_TILE[ab_r0 * MMA_K + tid * 2]),
                    pack_f16x2(A_TILE[ab_r1 * MMA_K + tid * 2 + 1], A_TILE[ab_r1 * MMA_K + tid * 2]),
                    pack_f16x2(A_TILE[ab_r0 * MMA_K + tid * 2 + 9], A_TILE[ab_r0 * MMA_K + tid * 2 + 8]),
                    pack_f16x2(A_TILE[ab_r1 * MMA_K + tid * 2 + 9], A_TILE[ab_r1 * MMA_K + tid * 2 + 8]),
                ];
            }

            // Pack B fragments for each column group
            let mut b_frags: [[u32; 2]; 4] = [[0; 2]; 4];
            let mut ni = 0usize;
            while ni < 4 {
                let n = ni * 8 + group_id;
                unsafe {
                    b_frags[ni] = [
                        pack_f16x2(B_TILE[n * MMA_K + tid * 2 + 1], B_TILE[n * MMA_K + tid * 2]),
                        pack_f16x2(B_TILE[n * MMA_K + tid * 2 + 9], B_TILE[n * MMA_K + tid * 2 + 8]),
                    ];
                }
                ni += 1;
            }

            // Issue 8 MMA ops via compiler intrinsic (proper register allocation)
            unsafe {
                cuda_device::mma_sync::mma_sync_m16n8k16_f32_f16(&mut acc_t0, &a_frag_top, &b_frags[0]);
                cuda_device::mma_sync::mma_sync_m16n8k16_f32_f16(&mut acc_t1, &a_frag_top, &b_frags[1]);
                cuda_device::mma_sync::mma_sync_m16n8k16_f32_f16(&mut acc_t2, &a_frag_top, &b_frags[2]);
                cuda_device::mma_sync::mma_sync_m16n8k16_f32_f16(&mut acc_t3, &a_frag_top, &b_frags[3]);
                cuda_device::mma_sync::mma_sync_m16n8k16_f32_f16(&mut acc_b0, &a_frag_bot, &b_frags[0]);
                cuda_device::mma_sync::mma_sync_m16n8k16_f32_f16(&mut acc_b1, &a_frag_bot, &b_frags[1]);
                cuda_device::mma_sync::mma_sync_m16n8k16_f32_f16(&mut acc_b2, &a_frag_bot, &b_frags[2]);
                cuda_device::mma_sync::mma_sync_m16n8k16_f32_f16(&mut acc_b3, &a_frag_bot, &b_frags[3]);
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
