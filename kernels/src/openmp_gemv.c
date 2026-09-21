#include <stdint.h>
#include <stddef.h>

typedef float (*DotFnQ4K)(const void *weight_row, const void *q8k, size_t blocks_per_row);
typedef float (*DotFnQ6K)(const void *weight_row, const void *q8k, size_t blocks_per_row);

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
