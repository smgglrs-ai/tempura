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
void quantize_row_iq4_nl(const float *x, void *y, int64_t k);
void quantize_row_iq4_xs(const float *x, void *y, int64_t k);
void dequantize_row_iq4_nl(const void *x, float *y, int64_t k);
void dequantize_row_iq4_xs(const void *x, float *y, int64_t k);

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

    // ── IQ4_NL ─────────────────────────────────────────────────────────
    // Non-linear 4-bit lookup table quantization, 32 elements per block.
    // Block layout: { ggml_half d; uint8_t qs[16]; } = 18 bytes.
    {
        const int k = 256;
        float src[256];
        gen_activations(src, k, 13);

        // 256/32 = 8 blocks, 8 * 18 = 144 bytes
        const size_t iq4nl_bytes = 144;
        void *iq4nl_data = calloc(1, iq4nl_bytes);
        quantize_row_iq4_nl(src, iq4nl_data, k);

        float deq[256];
        dequantize_row_iq4_nl(iq4nl_data, deq, k);

        write_file("iq4nl_src.bin", src, k * sizeof(float));
        write_file("iq4nl_quantized.bin", iq4nl_data, iq4nl_bytes);
        write_file("iq4nl_dequantized.bin", deq, k * sizeof(float));
        printf("  IQ4_NL written\n");

        free(iq4nl_data);
    }

    // ── IQ4_XS ─────────────────────────────────────────────────────────
    // Cross-scale 4-bit quantization, 256 elements per block.
    // Block layout: { ggml_half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; }
    // = 2 + 2 + 4 + 128 = 136 bytes.
    {
        const int k = 256;
        float src[256];
        gen_activations(src, k, 77);

        // 256/256 = 1 block, 136 bytes
        const size_t iq4xs_bytes = 136;
        void *iq4xs_data = calloc(1, iq4xs_bytes);
        quantize_row_iq4_xs(src, iq4xs_data, k);

        float deq[256];
        dequantize_row_iq4_xs(iq4xs_data, deq, k);

        write_file("iq4xs_src.bin", src, k * sizeof(float));
        write_file("iq4xs_quantized.bin", iq4xs_data, iq4xs_bytes);
        write_file("iq4xs_dequantized.bin", deq, k * sizeof(float));
        printf("  IQ4_XS written\n");

        free(iq4xs_data);
    }

    // ── RoPE reference values ───────────────────────────────────────────
    // Standard NeoX-style RoPE: freq = pos / base^(2i/d), rotates pairs (x[i], x[i+d/2]).
    // These cos/sin values are the ground truth the Rust kernel must match.
    {
        const int n_pos = 8, head_dim = 64;
        const float base = 10000.0f;
        // cos and sin for each (pos, dim_pair) — head_dim values per position
        float rope_cos[8 * 64], rope_sin[8 * 64];
        const int half_dim = head_dim / 2;
        for (int pos = 0; pos < n_pos; pos++) {
            for (int i = 0; i < half_dim; i++) {
                float theta = (float)pos / powf(base, 2.0f * (float)i / (float)head_dim);
                float c = cosf(theta), s = sinf(theta);
                rope_cos[pos * head_dim + i]            = c;
                rope_cos[pos * head_dim + i + half_dim] = c;
                rope_sin[pos * head_dim + i]            = s;
                rope_sin[pos * head_dim + i + half_dim] = s;
            }
        }
        write_file("rope_cos_ref.bin", rope_cos, sizeof(rope_cos));
        write_file("rope_sin_ref.bin", rope_sin, sizeof(rope_sin));
        printf("  RoPE reference (n_pos=%d, head_dim=%d, base=%.0f)\n", n_pos, head_dim, base);
    }

    printf("Done.\n");
    return 0;
}
