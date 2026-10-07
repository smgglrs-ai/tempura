# llama.cpp Lookup Tables and Precomputed Constants — Hot Inference Path Audit

Analysis of every lookup table and precomputed constant array used in hot
inference paths in llama.cpp (`ggml/src/ggml-cpu/`).  We already translated
`ggml_table_f32_f16`; this document finds everything else.

---

## 1. `ggml_table_f32_f16`

**File:** `ggml/src/ggml-cpu/ggml-cpu.c:81`

- **Size:** 65,536 entries × 4 bytes = 256 KB
- **Computes:** f16 → f32 for all 65,536 IEEE 754 half-precision bit patterns
- **Populated at init:** `ggml_table_f32_f16[i] = ggml_fp16_to_fp32(i)` (loop over 0..65535)
- **Used in:** `ggml_lookup_fp16_to_fp32()` (macro `GGML_CPU_FP16_TO_FP32`) — called on every
  quantized block's scale fields (`x[i].d`, `x[i].dmin`) in every `ggml_vec_dot_q*_K_q8_K`
  function.  Also in softmax mask application and dequantization.
- **Assembly pattern (x86):** `mov eax, [scale_ptr]; movss xmm0, [ggml_table_f32_f16 + rax*4]`
- **Status in Tempura:** TRANSLATED — `fp16_to_fp32()` in `simd_mappings_llama.rs`
  uses F16C hardware (`_cvtsh_ss`) when available, falls back to `half` crate.
  The 256 KB table itself is not materialized; hardware instruction is faster.

---

## 2. `ggml_table_gelu_f16`

**File:** `ggml/src/ggml-cpu/vec.cpp:6` (extern declared in `vec.h:62`)

- **Size:** 65,536 entries × 2 bytes = 128 KB
- **Computes:** `GELU(x)` for all f16 inputs, stored back as f16
- **Populated at init:** `ggml_table_gelu_f16[i] = fp32_to_fp16(ggml_gelu_f32(fp16_to_fp32(i)))`
- **Used in:**
  - `ggml_vec_gelu_f32` (vec.h:975) — main GELU activation on every FFN gate
  - `ggml_vec_gelu_backprop_f32` (vec.h:1425) — backward pass
  - All paths use `y[i] = ggml_table_gelu_f16[i16[i]]` where `i16` is the raw
    bit pattern of the f16-cast input
- **Assembly pattern:** `movzx eax, word ptr [input_f16+i*2]; movzx ecx, word ptr [ggml_table_gelu_f16+rax*2]`
- **Status in Tempura:** MISSING — we have no gelu LUT; forward pass stubs use scalar `ggml_gelu_f32`.
  **Impact:** Every FFN layer calls GELU; at batch size 1 this is ~2× slower than the table path.

---

## 3. `ggml_table_gelu_quick_f16`

**File:** `ggml/src/ggml-cpu/vec.cpp:9`

- **Size:** 65,536 entries × 2 bytes = 128 KB
- **Computes:** Quick-GELU approximation `x * sigmoid(-1.702x)` for all f16 inputs
  - `GELU_QUICK_COEF = -1.702f` (vec.h:964)
  - `ggml_gelu_quick_f32(x) = x / (1 + expf(-1.702 * x))`
- **Used in:** `ggml_vec_gelu_quick_f32` (vec.h:1024/1034), GLU gate variants (vec.h:1476/1491)
- **Status in Tempura:** MISSING — same situation as `ggml_table_gelu_f16`.

---

## 4. `ggml_table_f32_e8m0_half`

**File:** `ggml/src/ggml-cpu/ggml-cpu.c:84`

- **Size:** 256 entries × 4 bytes = 1 KB
- **Computes:** E8M0 (8-bit exponent-only float, used in MXFP4 format) → f32
  - `ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i)` = `powf(2.0f, (i - 127) / 2.0f)` (half-exponent)
- **Used in:** `ggml_vec_dot_mxfp4_q8_0` (quants.c:316) — the MXFP4 scale dequantization
- **Accessed via macro:** `GGML_CPU_E8M0_TO_FP32_HALF(x)` = `ggml_table_f32_e8m0_half[(uint8_t)(x)]`
- **Status in Tempura:** MISSING — MXFP4 format not yet translated, table not needed yet.
  Add when MXFP4 `vec_dot` is implemented.

---

## 5. `ggml_table_f32_ue4m3`

**File:** `ggml/src/ggml-cpu/ggml-cpu.c:87`

- **Size:** 256 entries × 4 bytes = 1 KB
- **Computes:** UE4M3 (4-bit unsigned exponent, 3-bit mantissa, NVFP4 format) → f32
  - `ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i)`
- **Used in:** `ggml_vec_dot_nvfp4_q8_0` (quants.c:347) — NVFP4 scale dequantization
  - Note: the per-block scale still uses `ggml_ue4m3_to_fp32()` directly for one value,
    but the LUT is used for the element-level decode.
- **Accessed via macro:** `GGML_CPU_UE4M3_TO_FP32(x)` = `ggml_table_f32_ue4m3[(uint8_t)(x)]`
- **Status in Tempura:** MISSING — NVFP4 format not yet translated.

---

## 6. `get_scale_shuffle_k4` / `K_SHUFFLE_K4`

**File:** `ggml/src/ggml-cpu/arch/x86/quants.c:528-538`

- **Size:** 256 bytes (static const, inline in function)
- **Computes:** PSHUFB control bytes that broadcast Q4_K/Q6_K scale lane `i` to all
  positions needed by the 32-lane AVX2 maddubs inner loop
- **Pattern:** Row `i` (32 bytes) repeats byte pair `{2i, 2i+1}` sixteen times
- **Used in:** `ggml_vec_dot_q4_K_q8_K` AVX2 path (inner loop `j < QK_K/64`),
  `ggml_vec_dot_q6_K_q8_K` AVX2 path
- **Assembly pattern:** `vpshufb ymm, ymm_scales, [K_SHUFFLE_K4 + i*32]`
- **Status in Tempura:** TRANSLATED — `K_SHUFFLE_K4: [u8; 256]` in `quants_llama.rs:118`
  and `quants_llama_avx.rs:1187`.

---

## 7. `get_scale_shuffle_q3k` / `K_SHUFFLE_Q3K`

**File:** `ggml/src/ggml-cpu/arch/x86/quants.c:519-525`

- **Size:** 128 bytes (static const, inline in function)
- **Computes:** PSHUFB control bytes that broadcast Q3_K scale pairs to their 8
  corresponding positions across a 32-lane vector
- **Pattern:** Row `i` (32 bytes) repeats byte pair `{2i, 2i+1}` sixteen times,
  but only 4 distinct rows (0..3)
- **Used in:** `ggml_vec_dot_q3_K_q8_K` AVX2 path
- **Status in Tempura:** TRANSLATED — `K_SHUFFLE_Q3K: [u8; 128]` in `quants_llama.rs:100`.

---

## 8. `get_scale_shuffle` / `K_SHUFFLE` (Q2_K / IQ formats)

**File:** `ggml/src/ggml-cpu/arch/x86/quants.c:541-551`

- **Size:** 128 bytes (static const, inline in function)
- **Computes:** 128-bit PSHUFB mask that broadcasts a single byte-lane scale `i`
  to 8 consecutive 16-bit positions (for 16-bit accumulation with `_mm_madd_epi16`)
- **Pattern:** Row `i` (16 bytes) = `{i, i, i, i, i, i, i, i}` (broadcast)
- **Used in:** `ggml_vec_dot_q2_K_q8_K` AVX path, `ggml_vec_dot_q6_K_q8_K` SSE4 fallback
- **Status in Tempura:** TRANSLATED — `K_SHUFFLE: [u8; 128]` in `quants_llama.rs:144`
  and `quants_llama_avx.rs:186`.

---

## 9. `kmask1`, `kmask2`, `kmask3` (scale unpacking masks)

**File:** `ggml/src/ggml-cpu/arch/x86/quants.c:2051-2053` (and quants.c:709-711, 784-786)

- **Size:** 3 × 4 bytes = 12 bytes (compile-time constants, not arrays)
- **Values:**
  - `kmask1 = 0x3f3f3f3f` — lower 6 bits of each byte
  - `kmask2 = 0x0f0f0f0f` — lower 4 bits of each byte
  - `kmask3 = 0x03030303` — lower 2 bits of each byte
- **Used in:** Scale byte-unpacking in every K-quant block at the start of each outer
  loop iteration: Q4_K, Q5_K, Q6_K, Q2_K, Q3_K.  The 12-byte `scales` field packs
  6-bit scales into a compact format that must be expanded to 8-bit per scale before
  the inner loop.
- **Assembly pattern:** `and eax, 0x3f3f3f3f` — single instruction per word
- **Status in Tempura:** TRANSLATED — defined as local constants at the top of each
  matching `vec_dot_*` function.

---

## 10. `ggml_v_expf` — Polynomial Exp Approximation

**File:** `ggml/src/ggml-cpu/vec.h:1172` (AVX512), `1215` (AVX2), `1278` (SSE2)

- **Type:** Inline SIMD function, not a table — polynomial approximation using Cody-Waite
  range reduction and a degree-4 minimax polynomial
- **Constants (embedded in every call site via `_mm256_set1_ps`, not hoisted):**

  | Name       | Hex float literal  | Decimal value     | Purpose                    |
  |------------|--------------------|-------------------|----------------------------|
  | `r`        | `0x1.8p23f`        | 12582912.0        | Rounding bias for int cast |
  | `log2e`    | `0x1.715476p+0f`   | 1.4426950216…     | log₂(e) for range reduction|
  | `ln2_hi`   | `0x1.62e4p-1f`     | 0.693359375       | High part of ln2           |
  | `ln2_lo`   | `0x1.7f7d1cp-20f`  | 1.428606765e-6    | Low part of ln2 (error)    |
  | `p1`       | `0x1.ffffecp-1f`   | 0.999999821…      | Poly coefficient           |
  | `p2`       | `0x1.fffdb6p-2f`   | 0.499996185…      | Poly coefficient           |
  | `p3`       | `0x1.555e66p-3f`   | 0.166656464…      | Poly coefficient           |
  | `p4`       | `0x1.573e2ep-5f`   | 0.041917745…      | Poly coefficient           |
  | `p5`       | `0x1.0e4020p-7f`   | 0.008369981…      | Poly coefficient           |

- **Used in:** `ggml_vec_soft_max_f32` (vec.cpp:531) — the innermost loop computing `exp(x - max)`
  for every attention score.  Also `ggml_v_silu` for FFN SiLU activation.
- **Critical property:** These are `_mm256_set1_ps(const)` inside a loop — GCC/Clang hoists them
  to registers before the loop (broadcast once).  If we use `_mm256_set1_ps` inside the Rust unsafe
  block in a loop, `rustc` + LLVM also hoists them provided the function is `#[inline(always)]`.
- **Status in Tempura:** TRANSLATED — `ggml_v_expf` in `vec_llama.rs` (or `ops_llama_simd.rs`).
  Verify that all 9 constants match the hex float literals exactly — any rounding difference
  causes >1 ULP divergence from llama.cpp.

---

## 11. `ksigns64`

**File:** `ggml/src/ggml-common.h:524`

- **Size:** 128 entries × 8 bytes = 1 KB
- **Computes:** Sign-expansion masks for IQ2_XS / IQ3_XXS — maps a 7-bit sign index to
  a 64-bit mask where each byte is either `0x00` (positive) or `0xFF` (negative).
  Used to flip 8 `int8` grid values simultaneously via AND.
- **Used in:** GPU paths only (SYCL `mmvq.cpp:416`, `512`).  CPU path uses
  `ksigns_iq2xs` (128 bytes, separate table) with the `kmask_iq2xs` bitmask array.
- **Status in Tempura:** NOT NEEDED for CPU — `KSIGNS_IQ2XS` covers the CPU path.
  Only needed if/when GPU kernels are translated.

---

## 12. `kmask_iq2xs` / `KMASK_IQ2XS`

**File:** `ggml/src/ggml-common.h:509`

- **Size:** 8 entries × 1 byte = 8 bytes
- **Computes:** Bitmask array `{1, 2, 4, 8, 16, 32, 64, 128}` — used to test individual
  bits of a sign byte, determining whether to negate each of 8 grid values
- **Used in:** `ggml_vec_dot_iq2_xs_q8_K` scalar path (quants.c:937, 976, etc.)
- **Status in Tempura:** TRANSLATED — `KMASK_IQ2XS: [u8; 8]` in `iq_tables.rs:3`.

---

## 13. `ksigns_iq2xs` / `KSIGNS_IQ2XS`

**File:** `ggml/src/ggml-common.h:513`

- **Size:** 128 entries × 1 byte = 128 bytes
- **Computes:** Maps a 7-bit sign index → packed sign byte used with `kmask_iq2xs`
  to negate individual grid elements
- **Used in:** `ggml_vec_dot_iq2_xs_q8_K` and `ggml_vec_dot_iq3_xxs_q8_K` scalar paths
- **Status in Tempura:** TRANSLATED — `KSIGNS_IQ2XS: [u8; 128]` in `iq_tables.rs:5`.

---

## 14. IQ Grid Tables

All six IQ grid tables are translated in `iq_tables.rs`. Summary:

| Table              | llama.cpp name    | Size                 | Status in Tempura    |
|--------------------|-------------------|----------------------|----------------------|
| `IQ2XXS_GRID`      | `iq2xxs_grid`     | 256 × u64 = 2 KB     | TRANSLATED           |
| `IQ2XS_GRID`       | `iq2xs_grid`      | 512 × u64 = 4 KB     | TRANSLATED           |
| `IQ2S_GRID`        | `iq2s_grid`       | 1024 × u64 = 8 KB    | TRANSLATED           |
| `IQ3XXS_GRID`      | `iq3xxs_grid`     | 256 × u32 = 1 KB     | TRANSLATED           |
| `IQ3S_GRID`        | `iq3s_grid`       | 512 × u32 = 2 KB     | TRANSLATED           |
| `IQ1S_GRID`        | `iq1s_grid`       | 2048 × u64 = 16 KB   | TRANSLATED           |
| `KVALUES_IQ4NL`    | `kvalues_iq4nl`   | 16 × i8 = 16 bytes   | TRANSLATED           |
| `KVALUES_MXFP4`    | `kvalues_fp4`     | 16 × i8 = 16 bytes   | TRANSLATED           |
| `iq1s_grid_gpu`    | `iq1s_grid_gpu`   | 2048 × u32 = 8 KB    | NOT NEEDED (GPU only)|

---

## 15. `bytes_from_bits_32` / `bytes_from_nibbles_32` Helper Masks

**File:** `ggml/src/ggml-cpu/arch/x86/quants.c:76-95`

These are not arrays but inline constants embedded in helper functions:

- **`bytes_from_bits_32`:**
  - `shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000)` — routes each of 4 input bytes to 8 output bytes
  - `bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe)` — used with `_mm256_or_si256` + `_mm256_cmpeq_epi8` to expand bits to 0x00/0xFF bytes
  - Used in: Q3_K, Q5_0, Q5_1, Q5_K high-bit extraction

- **`bytes_from_nibbles_32`:**
  - `lowMask = _mm256_set1_epi8(0xF)` — isolates low/high nibbles
  - Used in: Q4_0, Q4_1, Q4_K nibble extraction

- **Status in Tempura:** TRANSLATED — both helpers in `quants_llama.rs:22-45`.

---

## What Is Missing From Our Translation

Priority order (highest impact first):

### Priority 1: GELU / Quick-GELU tables (128 KB each)

`ggml_table_gelu_f16` and `ggml_table_gelu_quick_f16` are used on every FFN layer's
activation.  Without them, we compute `expf` per-element in the scalar fallback, which
is 10-50× slower than a table lookup.

**What to add in `unary_ops_llama.rs`:**

```rust
// 128 KB each — lazily initialized via std::sync::OnceLock
static GELU_TABLE_F16: OnceLock<Box<[u16; 65536]>> = OnceLock::new();
static GELU_QUICK_TABLE_F16: OnceLock<Box<[u16; 65536]>> = OnceLock::new();

fn gelu_table() -> &'static [u16; 65536] {
    GELU_TABLE_F16.get_or_init(|| {
        let mut t = Box::new([0u16; 65536]);
        for i in 0..65536u32 {
            let f = fp16_to_fp32(i as u16);
            t[i as usize] = fp32_to_fp16(ggml_gelu_f32(f));
        }
        t
    })
}
```

Then `ggml_vec_gelu_f32`: `y[i] = fp16_to_fp32(gelu_table()[as_u16(x[i])])`.

### Priority 2: `ggml_table_f32_e8m0_half` and `ggml_table_f32_ue4m3`

Required for MXFP4 and NVFP4 format support.  Each is 1 KB — trivial to add.
Defer until those formats are implemented.

### Priority 3: Verify `ggml_v_expf` polynomial constants are byte-exact

The hex float literals in `ggml_v_expf` must match exactly.  Any deviation introduces
systematic bias across all softmax outputs.  Run the DualPipeline softmax comparison
test to verify.

---

## Constants Loaded Once Before Hot Loops (Not Tables)

These are scalar constants computed once per block in the outer loop body, before
the inner AVX2 loop over 64-element sub-blocks.  They are NOT reloaded in the inner loop.

| Constant | Where | Value | Notes |
|----------|-------|-------|-------|
| `d` | Q4K outer loop | `y[i].d * fp16_to_fp32(x[i].d)` | Scale product, hoisted |
| `dmin` | Q4K outer loop | `-y[i].d * fp16_to_fp32(x[i].dmin)` | Min-scale product |
| `m4` | Q4K prologue | `_mm256_set1_epi8(0xF)` | Nibble mask, truly constant |
| `mins_and_scales` | Q4K outer | `_mm256_cvtepu8_epi16(...)` | 8 scale+min values per block |
| `scales` | Q4K outer | `MM256_SET_M128I(sc128, sc128)` | Broadcast to 256-bit for shuffles |

The `m4` nibble mask (`0x0F` byte broadcast) is the only one that could be a
global constant — GCC/Clang CSE it out of the loop anyway, but in Rust it should
be defined once per function call with `_mm256_set1_epi8(0x0F)` outside the loop.

Our translation already does this correctly in `quants_llama.rs` and `quants_llama_avx.rs`.

---

## Summary Table

| Table / Constant              | llama.cpp source        | Size     | Hot path             | Tempura status  |
|-------------------------------|-------------------------|----------|----------------------|-----------------|
| `ggml_table_f32_f16`          | ggml-cpu.c:81           | 256 KB   | All K-quant scales   | TRANSLATED (F16C hw) |
| `ggml_table_gelu_f16`         | vec.cpp:6               | 128 KB   | Every FFN GELU       | **MISSING**     |
| `ggml_table_gelu_quick_f16`   | vec.cpp:9               | 128 KB   | Quick-GELU activations| **MISSING**    |
| `ggml_table_f32_e8m0_half`    | ggml-cpu.c:84           | 1 KB     | MXFP4 scale decode   | Not needed yet  |
| `ggml_table_f32_ue4m3`        | ggml-cpu.c:87           | 1 KB     | NVFP4 scale decode   | Not needed yet  |
| `K_SHUFFLE_K4`                | x86/quants.c:528        | 256 B    | Q4K/Q6K inner loop   | TRANSLATED      |
| `K_SHUFFLE_Q3K`               | x86/quants.c:519        | 128 B    | Q3K inner loop       | TRANSLATED      |
| `K_SHUFFLE`                   | x86/quants.c:541        | 128 B    | Q2K/Q6K SSE path     | TRANSLATED      |
| `kmask1/2/3`                  | x86/quants.c:2051       | 12 B     | K-quant scale unpack | TRANSLATED      |
| `ggml_v_expf` polynomial      | vec.h:1172/1215/1278    | 9 scalars| Softmax inner loop   | TRANSLATED (verify constants) |
| `ksigns64`                    | ggml-common.h:524       | 1 KB     | GPU IQ2/IQ3 only     | NOT NEEDED (CPU) |
| `KMASK_IQ2XS`                 | ggml-common.h:509       | 8 B      | IQ2_XS scalar        | TRANSLATED      |
| `KSIGNS_IQ2XS`                | ggml-common.h:513       | 128 B    | IQ2_XS / IQ3_XXS     | TRANSLATED      |
| All IQ grid tables            | ggml-common.h:560-1650  | ~33 KB   | IQ2/IQ3/IQ1 formats  | TRANSLATED      |
