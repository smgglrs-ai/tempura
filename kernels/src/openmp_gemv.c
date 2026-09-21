#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef float (*DotFnQ4K)(const void *weight_row, const void *q8k, size_t blocks_per_row);
typedef float (*DotFnQ6K)(const void *weight_row, const void *q8k, size_t blocks_per_row);
typedef float (*BlockDotFn)(const uint8_t *w_blk, const uint8_t *q8k_blk);

#define N_TILE_Q4K 64
#define N_TILE_Q6K 48

void tempura_omp_gemv_q4k(
    const uint8_t *weights,
    const void    *q8k,
    float         *output,
    size_t         n,
    size_t         blocks_per_row,
    size_t         block_bytes,
    DotFnQ4K       dot_fn
) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < (int)n; j++) {
        const uint8_t *weight_row = weights + (size_t)j * blocks_per_row * block_bytes;
        output[j] = dot_fn(weight_row, q8k, blocks_per_row);
    }
}

// K-tiled GEMM for Q4K: loop order (j_tile ∥, kb, i, j).
// Weight blocks for the j_tile at k_block kb (tile_n×q4k_block_bytes ≈ 9 KB) are
// loaded from memory once and reused for all m input rows — saving (m-1) extra fetches.
void tempura_omp_gemm_tiled_q4k(
    const uint8_t *weights,
    const uint8_t *q8k_rows,
    float         *output,
    size_t         m,
    size_t         n,
    size_t         bpr,
    size_t         q4k_block_bytes,
    size_t         q8k_block_bytes,
    BlockDotFn     dot_fn
) {
    #pragma omp parallel for schedule(static)
    for (int j_start = 0; j_start < (int)n; j_start += N_TILE_Q4K) {
        int tile_n = (j_start + N_TILE_Q4K < (int)n) ? N_TILE_Q4K : (int)n - j_start;

        // Zero output tile (scattered writes into [m][n] output)
        for (int i = 0; i < (int)m; i++)
            memset(output + i * n + j_start, 0, (size_t)tile_n * sizeof(float));

        // K-block outer loop: N_TILE_Q4K × q4k_block_bytes ≈ 9 KB stays in L1D
        for (size_t kb = 0; kb < bpr; kb++) {
            for (int i = 0; i < (int)m; i++) {
                const uint8_t *q8k_blk = q8k_rows
                    + (size_t)i * bpr * q8k_block_bytes
                    + kb * q8k_block_bytes;
                for (int j = 0; j < tile_n; j++) {
                    const uint8_t *w_blk = weights
                        + (size_t)(j_start + j) * bpr * q4k_block_bytes
                        + kb * q4k_block_bytes;
                    output[(size_t)i * n + (size_t)(j_start + j)] += dot_fn(w_blk, q8k_blk);
                }
            }
        }
    }
}

// K-tiled GEMM for Q6K (same structure, different tile size due to larger blocks)
void tempura_omp_gemm_tiled_q6k(
    const uint8_t *weights,
    const uint8_t *q8k_rows,
    float         *output,
    size_t         m,
    size_t         n,
    size_t         bpr,
    size_t         q6k_block_bytes,
    size_t         q8k_block_bytes,
    BlockDotFn     dot_fn
) {
    #pragma omp parallel for schedule(static)
    for (int j_start = 0; j_start < (int)n; j_start += N_TILE_Q6K) {
        int tile_n = (j_start + N_TILE_Q6K < (int)n) ? N_TILE_Q6K : (int)n - j_start;

        for (int i = 0; i < (int)m; i++)
            memset(output + i * n + j_start, 0, (size_t)tile_n * sizeof(float));

        for (size_t kb = 0; kb < bpr; kb++) {
            for (int i = 0; i < (int)m; i++) {
                const uint8_t *q8k_blk = q8k_rows
                    + (size_t)i * bpr * q8k_block_bytes
                    + kb * q8k_block_bytes;
                for (int j = 0; j < tile_n; j++) {
                    const uint8_t *w_blk = weights
                        + (size_t)(j_start + j) * bpr * q6k_block_bytes
                        + kb * q6k_block_bytes;
                    output[(size_t)i * n + (size_t)(j_start + j)] += dot_fn(w_blk, q8k_blk);
                }
            }
        }
    }
}

void tempura_omp_gemv_q6k(
    const uint8_t *weights,
    const void    *q8k,
    float         *output,
    size_t         n,
    size_t         blocks_per_row,
    size_t         block_bytes,
    DotFnQ6K       dot_fn
) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < (int)n; j++) {
        const uint8_t *weight_row = weights + (size_t)j * blocks_per_row * block_bytes;
        output[j] = dot_fn(weight_row, q8k, blocks_per_row);
    }
}
