use super::blocks::{BlockQ4K, BlockQ4Kx8};

/// Repack `n_rows` Q4K blocks (from a weight matrix with `bpr` blocks per row)
/// into `BlockQ4Kx8` groups. `n_rows` must be divisible by 8.
///
/// Returns a `Vec<BlockQ4Kx8>` with `(n_rows/8) * bpr` entries. Each entry
/// corresponds to one group of 8 weight rows × one K-dimension block.
///
/// The layout exactly matches llama.cpp's `make_block_q4_Kx8` with
/// `blck_size_interleave = 8`:
/// - `qs`: 128 groups of 8 bytes, cycling through rows 0-7 at each source offset
/// - `scales`: 4 sub-block pairs × 24 bytes, encoding 8 rows' scale/min data
///   using the same 6-bit packed format as standard Q4K but across rows
pub fn repack_q4k_to_q4kx8(src: &[BlockQ4K], n_rows: usize, bpr: usize) -> Vec<BlockQ4Kx8> {
    assert_eq!(n_rows % 8, 0, "n_rows={n_rows} must be divisible by 8");
    assert_eq!(
        src.len(),
        n_rows * bpr,
        "src.len()={} != n_rows*bpr={}",
        src.len(),
        n_rows * bpr
    );

    let n_groups = n_rows / 8;
    let mut out: Vec<BlockQ4Kx8> = Vec::with_capacity(n_groups * bpr);

    for grp in 0..n_groups {
        for blk in 0..bpr {
            let mut dst = BlockQ4Kx8 {
                d: [0u16; 8],
                dmin: [0u16; 8],
                scales: [0u8; 96],
                qs: [0u8; 1024],
            };

            // Collect pointers to the 8 source Q4K blocks for this row group
            let rows: [&BlockQ4K; 8] = std::array::from_fn(|r| &src[(grp * 8 + r) * bpr + blk]);

            // Copy d and dmin per row
            for r in 0..8 {
                dst.d[r] = rows[r].d;
                dst.dmin[r] = rows[r].dmin;
            }

            // Interleave qs in 8-byte chunks (blck_size_interleave=8).
            // end = QK_K * 4 / blck_size_interleave = 256 * 4 / 8 ... but QK_K/2 = 128 bytes per row,
            // so end = 128 * 8 / 8 = 128 chunks.
            // Chunk i: source row = i % 8, source byte offset = (i / 8) * 8.
            for i in 0..128usize {
                let src_row = i % 8;
                let src_byte = (i / 8) * 8;
                let dst_byte = i * 8;
                dst.qs[dst_byte..dst_byte + 8]
                    .copy_from_slice(&rows[src_row].qs[src_byte..src_byte + 8]);
            }

            // Pack scales (96 bytes = 4 sub-block pairs × 24 bytes).
            // Each 12-byte group encodes one Q4K scale index across all 8 rows,
            // using the same 6-bit format as standard Q4K, applied row-wise.
            //
            // First 4 groups (i=0..3): Q4K scale indices 0..3
            let mut s = [0u8; 8];
            let mut m = [0u8; 8];

            for i in 0..4usize {
                for r in 0..8 {
                    s[r] = rows[r].scales[i] & 63;
                    m[r] = rows[r].scales[i + 4] & 63;
                }
                dst.scales[i * 12] = (s[0] & 63) | ((s[4] & 48) << 2);
                dst.scales[i * 12 + 1] = (s[1] & 63) | ((s[5] & 48) << 2);
                dst.scales[i * 12 + 2] = (s[2] & 63) | ((s[6] & 48) << 2);
                dst.scales[i * 12 + 3] = (s[3] & 63) | ((s[7] & 48) << 2);
                dst.scales[i * 12 + 4] = (m[0] & 63) | ((m[4] & 48) << 2);
                dst.scales[i * 12 + 5] = (m[1] & 63) | ((m[5] & 48) << 2);
                dst.scales[i * 12 + 6] = (m[2] & 63) | ((m[6] & 48) << 2);
                dst.scales[i * 12 + 7] = (m[3] & 63) | ((m[7] & 48) << 2);
                dst.scales[i * 12 + 8] = (s[4] & 15) | ((m[4] & 15) << 4);
                dst.scales[i * 12 + 9] = (s[5] & 15) | ((m[5] & 15) << 4);
                dst.scales[i * 12 + 10] = (s[6] & 15) | ((m[6] & 15) << 4);
                dst.scales[i * 12 + 11] = (s[7] & 15) | ((m[7] & 15) << 4);
            }

            // Next 4 groups (i=0..3 offset +48): Q4K scale indices 4..7
            for i in 0..4usize {
                for r in 0..8 {
                    s[r] = ((rows[r].scales[i] & 0xC0) >> 2) | (rows[r].scales[i + 8] & 0x0F);
                    m[r] = ((rows[r].scales[i + 4] & 0xC0) >> 2)
                        | ((rows[r].scales[i + 8] & 0xF0) >> 4);
                }
                dst.scales[i * 12 + 48] = (s[0] & 63) | ((s[4] & 48) << 2);
                dst.scales[i * 12 + 49] = (s[1] & 63) | ((s[5] & 48) << 2);
                dst.scales[i * 12 + 50] = (s[2] & 63) | ((s[6] & 48) << 2);
                dst.scales[i * 12 + 51] = (s[3] & 63) | ((s[7] & 48) << 2);
                dst.scales[i * 12 + 52] = (m[0] & 63) | ((m[4] & 48) << 2);
                dst.scales[i * 12 + 53] = (m[1] & 63) | ((m[5] & 48) << 2);
                dst.scales[i * 12 + 54] = (m[2] & 63) | ((m[6] & 48) << 2);
                dst.scales[i * 12 + 55] = (m[3] & 63) | ((m[7] & 48) << 2);
                dst.scales[i * 12 + 56] = (s[4] & 15) | ((m[4] & 15) << 4);
                dst.scales[i * 12 + 57] = (s[5] & 15) | ((m[5] & 15) << 4);
                dst.scales[i * 12 + 58] = (s[6] & 15) | ((m[6] & 15) << 4);
                dst.scales[i * 12 + 59] = (s[7] & 15) | ((m[7] & 15) << 4);
            }

            out.push(dst);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::helpers::f16_to_f32;

    fn make_test_q4k(d: f32, dmin: f32, qs_fill: u8, scales_fill: u8) -> BlockQ4K {
        let mut b = BlockQ4K {
            d: crate::ggml::helpers::f32_to_f16(d),
            dmin: crate::ggml::helpers::f32_to_f16(dmin),
            scales: [0u8; 12],
            qs: [qs_fill; 128],
        };
        // Fill scales with valid 6-bit values using the same pattern as standard Q4K
        for i in 0..8 {
            let sc = scales_fill & 63;
            let mn = (scales_fill.wrapping_add(1)) & 63;
            if i < 4 {
                b.scales[i] = sc | ((mn & 3) << 6);
                b.scales[i + 4] = mn | ((sc & 3) << 6);
            }
        }
        b.scales[8] = (scales_fill & 0x0F) | ((scales_fill.wrapping_add(1) & 0x0F) << 4);
        b.scales[9] = b.scales[8];
        b.scales[10] = b.scales[8];
        b.scales[11] = b.scales[8];
        b
    }

    #[test]
    fn repack_size_and_layout() {
        assert_eq!(std::mem::size_of::<BlockQ4Kx8>(), crate::ggml::blocks::BLOCK_Q4KX8_SIZE);

        let bpr = 2usize;
        let n_rows = 8usize;
        let src: Vec<BlockQ4K> = (0..n_rows * bpr)
            .map(|i| make_test_q4k(1.0, 0.5, (i as u8) | 0x11, (i as u8) & 0x3F))
            .collect();

        let packed = repack_q4k_to_q4kx8(&src, n_rows, bpr);
        assert_eq!(packed.len(), (n_rows / 8) * bpr);

        // Verify d values are correctly copied
        for blk in 0..bpr {
            let p = &packed[blk];
            for r in 0..8 {
                let src_d = src[r * bpr + blk].d;
                assert_eq!(p.d[r], src_d, "d mismatch at row={r}, blk={blk}");
                let src_dmin = src[r * bpr + blk].dmin;
                assert_eq!(p.dmin[r], src_dmin, "dmin mismatch at row={r}, blk={blk}");
            }
        }
    }

    #[test]
    fn repack_qs_interleave_correct() {
        let bpr = 1usize;
        let n_rows = 8usize;
        // Give each row a distinct fill pattern
        let src: Vec<BlockQ4K> = (0..n_rows)
            .map(|r| {
                let mut b = BlockQ4K {
                    d: 0x3C00u16,
                    dmin: 0,
                    scales: [0u8; 12],
                    qs: [(r as u8) * 17; 128],
                };
                // Distinct per-row byte values
                for k in 0..128usize {
                    b.qs[k] = ((r * 128 + k) & 0xFF) as u8;
                }
                b
            })
            .collect();

        let packed = repack_q4k_to_q4kx8(&src, n_rows, bpr);
        let p = &packed[0];

        // Verify interleaving: chunk i covers row (i%8), bytes (i/8)*8..(i/8)*8+8
        for i in 0..128usize {
            let row = i % 8;
            let byte_off = (i / 8) * 8;
            let dst_off = i * 8;
            for k in 0..8usize {
                assert_eq!(
                    p.qs[dst_off + k],
                    src[row].qs[byte_off + k],
                    "qs mismatch at chunk={i}, k={k}: row={row}, byte_off={byte_off}"
                );
            }
        }
    }

    #[test]
    fn repack_d_values_match() {
        let bpr = 3usize;
        let n_rows = 16usize;
        let src: Vec<BlockQ4K> = (0..n_rows * bpr)
            .map(|i| make_test_q4k((i as f32) * 0.01 + 0.1, 0.05, 0x33, 10))
            .collect();

        let packed = repack_q4k_to_q4kx8(&src, n_rows, bpr);
        assert_eq!(packed.len(), (n_rows / 8) * bpr);

        for grp in 0..(n_rows / 8) {
            for blk in 0..bpr {
                let p = &packed[grp * bpr + blk];
                for r in 0..8 {
                    let src_blk = &src[(grp * 8 + r) * bpr + blk];
                    assert_eq!(p.d[r], src_blk.d);
                    assert_eq!(p.dmin[r], src_blk.dmin);
                    let pd = f16_to_f32(p.d[r]);
                    let sd = f16_to_f32(src_blk.d);
                    assert!((pd - sd).abs() < 1e-4, "d f16 round-trip mismatch");
                }
            }
        }
    }
}
