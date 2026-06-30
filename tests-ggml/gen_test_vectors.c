// Generate test vectors by calling ggml's quantize/dequantize directly.
// Outputs binary files that Rust tests compare against.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Direct function declarations from ggml-quants.h
void quantize_row_q8_0_ref(const float *x, void *y, int64_t k);
void quantize_row_q8_K_ref(const float *x, void *y, int64_t k);
void quantize_row_q4_0_ref(const float *x, void *y, int64_t k);
void quantize_row_q4_K_ref(const float *x, void *y, int64_t k);
void quantize_row_q6_K_ref(const float *x, void *y, int64_t k);
void dequantize_row_q4_0(const void *x, float *y, int64_t k);
void dequantize_row_q4_K(const void *x, float *y, int64_t k);
void dequantize_row_q6_K(const void *x, float *y, int64_t k);
void dequantize_row_q8_0(const void *x, float *y, int64_t k);

static void write_file(const char *name, const void *data, size_t len) {
    FILE *f = fopen(name, "wb");
    if (!f) { perror(name); exit(1); }
    fwrite(data, 1, len, f);
    fclose(f);
    printf("  wrote %s (%zu bytes)\n", name, len);
}

static void gen_activations(float *buf, int n, int seed) {
    for (int i = 0; i < n; i++) {
        float phase = (float)(i + seed) * 0.1f;
        buf[i] = sinf(phase) * 0.5f + 0.1f;
    }
}

int main() {
    printf("Generating ggml test vectors...\n");

    // ── Q4_0 ───────────────────────────────────────────────────────────
    {
        const int k = 256;
        float src[256];
        gen_activations(src, k, 42);

        int nb = k / 32;
        size_t q4_bytes = nb * 18;
        void *q4_data = calloc(1, q4_bytes);
        quantize_row_q4_0_ref(src, q4_data, k);

        float deq[256];
        dequantize_row_q4_0(q4_data, deq, k);

        size_t q8_bytes = nb * 34;
        void *q8_data = calloc(1, q8_bytes);
        quantize_row_q8_0_ref(src, q8_data, k);

        float deq_q8[256];
        dequantize_row_q8_0(q8_data, deq_q8, k);

        // f32 reference dot: dequant(q4) · dequant(q8)
        float ref_dot = 0.0f;
        for (int i = 0; i < k; i++) ref_dot += deq[i] * deq_q8[i];

        write_file("q4_0_src.bin", src, k * sizeof(float));
        write_file("q4_0_quantized.bin", q4_data, q4_bytes);
        write_file("q4_0_dequantized.bin", deq, k * sizeof(float));
        write_file("q4_0_q8_0_activations.bin", q8_data, q8_bytes);
        write_file("q4_0_ref_dot.bin", &ref_dot, sizeof(float));
        printf("  Q4_0 ref_dot = %.6f\n", ref_dot);

        free(q4_data); free(q8_data);
    }

    // ── Q4_K ───────────────────────────────────────────────────────────
    {
        const int k = 256;
        float src[256];
        gen_activations(src, k, 17);

        size_t q4k_bytes = 144;
        void *q4k_data = calloc(1, q4k_bytes);
        quantize_row_q4_K_ref(src, q4k_data, k);

        float deq[256];
        dequantize_row_q4_K(q4k_data, deq, k);

        size_t q8k_bytes = 292;
        void *q8k_data = calloc(1, q8k_bytes);
        quantize_row_q8_K_ref(src, q8k_data, k);

        write_file("q4k_src.bin", src, k * sizeof(float));
        write_file("q4k_quantized.bin", q4k_data, q4k_bytes);
        write_file("q4k_dequantized.bin", deq, k * sizeof(float));
        write_file("q4k_q8k_activations.bin", q8k_data, q8k_bytes);
        printf("  Q4_K written\n");

        free(q4k_data); free(q8k_data);
    }

    // ── Q6_K ───────────────────────────────────────────────────────────
    {
        const int k = 256;
        float src[256];
        gen_activations(src, k, 99);

        size_t q6k_bytes = 210;
        void *q6k_data = calloc(1, q6k_bytes);
        quantize_row_q6_K_ref(src, q6k_data, k);

        float deq[256];
        dequantize_row_q6_K(q6k_data, deq, k);

        size_t q8k_bytes = 292;
        void *q8k_data = calloc(1, q8k_bytes);
        quantize_row_q8_K_ref(src, q8k_data, k);

        write_file("q6k_src.bin", src, k * sizeof(float));
        write_file("q6k_quantized.bin", q6k_data, q6k_bytes);
        write_file("q6k_dequantized.bin", deq, k * sizeof(float));
        write_file("q6k_q8k_activations.bin", q8k_data, q8k_bytes);
        printf("  Q6_K written\n");

        free(q6k_data); free(q8k_data);
    }

    printf("Done.\n");
    return 0;
}
