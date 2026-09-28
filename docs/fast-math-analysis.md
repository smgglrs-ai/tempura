# Fast-Math Analysis: llama.cpp vs Tempura

**Branch:** llama-cpu-translation  
**Date:** 2026-09-28

This document catalogues every fast approximation for expensive math functions
(`exp`, `tanh`, `sigmoid`, `gelu`, `silu`) in llama.cpp's CPU inference path,
then compares them with tempura's current implementations.

---

## 1. llama.cpp Fast-Math Inventory

### 1.1 `ggml_v_expf` — Polynomial Approximation (Primary Fast Path)

**Source:** `ggml/src/ggml-cpu/vec.h`, lines 1133–1320

This is the central fast-exp in llama.cpp. It is an ARM Limited optimized
routine ported to x86 and RISC-V. Exists for every SIMD ISA:

| ISA        | Width      | Header guard                        |
|------------|------------|-------------------------------------|
| AVX-512    | 16 floats  | `__AVX512F__ && __AVX512DQ__`       |
| AVX2 + FMA | 8 floats   | `__AVX2__ && __FMA__`               |
| SSE2       | 4 floats   | `__SSE2__`                          |
| ARM SVE    | variable   | `__ARM_FEATURE_SVE && __aarch64__`  |
| ARM NEON   | 4 floats   | `__ARM_NEON && __aarch64__`         |
| RISC-V V   | variable   | `__riscv_v_intrinsic`               |

**Algorithm (AVX2 example, `ggml_v_expf(__m256 x)`):**

1. Cody-Waite range reduction: decompose `x = n * ln2 + b` with two-constant
   subtraction (`0x1.62e4p-1` hi, `0x1.7f7d1cp-20` lo).
2. Exponent reconstruction: shift the integer part of `n` into the IEEE-754
   exponent field (`slli_epi32(z, 23)`).
3. Degree-4 polynomial on `b` and `b²`:

   ```
   j = P4*b³ + P3*b² + P2*b + P1 + P0*b   (Horner, two-interleaved chains)
   ```

   Constants: `P0=0x1.ffffecp-1`, `P1=0x1.fffdb6p-2`, `P2=0x1.555e66p-3`,
   `P3=0x1.573e2ep-5`, `P4=0x1.0e4020p-7`.
4. Combine: `result = fmadd(j, k, k)` where `k = 2^n` from the bit-shift.
5. Overflow guard: `|n| > 126` triggers a two-step scaling path to handle
   numbers near ±∞.

**Accuracy:** max error 1.45358 + 0.5 ULPs (from ARM Limited specification).

**Throughput:** 8 floats per iteration (AVX2), no libm call.

---

### 1.2 Softmax — Uses `ggml_v_expf`

**Source:** `ggml/src/ggml-cpu/vec.cpp`, line 531 (`ggml_vec_soft_max_f32`)

```c
// AVX2 path (8 floats/iter):
__m256 val = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
                                        _mm256_set1_ps(max)));

// Scalar tail (1 float/iter, libm):
float val = expf(x[i] - max);
```

`ggml_vec_soft_max_f32` is the only exp-using function called from the main
attention path (`ggml_compute_forward_soft_max_f32`, `ops.cpp:5675`).

**Compiled instruction count (libggml-cpu.so, AVX2 build):** 139 instructions
for the full `ggml_vec_soft_max_f32` function.

---

### 1.3 SiLU — Uses `ggml_v_expf`

**Source:** `ggml/src/ggml-cpu/vec.h`, lines 1199–1262 (`ggml_v_silu`)
**Source:** `ggml/src/ggml-cpu/vec.cpp`, line 380 (`ggml_vec_silu_f32`)

```c
// AVX2 path:
__m256 ggml_v_silu(__m256 x) {
    const __m256 neg_x = _mm256_sub_ps(zero, x);
    const __m256 exp_neg_x = ggml_v_expf(neg_x);   // <-- fast polynomial
    return _mm256_div_ps(x, _mm256_add_ps(one, exp_neg_x));
}
```

Uses the same `ggml_v_expf` polynomial. No table. 8 floats per iteration.

---

### 1.4 GELU — Precomputed F16 Table (128 KB)

**Source:** `ggml/src/ggml-cpu/vec.h`, lines 61–65, 987–1008

```c
// 128 KB precomputed table: one entry per possible f16 bit pattern
extern ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// Lookup for f32 inputs (GGML_GELU_FP16 always defined):
void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) { y[i] = 0.0f; continue; }
        if (x[i] >= 10.0f)  { y[i] = x[i]; continue; }
        ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);  // F16C vcvtps2ph
        uint16_t t; memcpy(&t, &fp16, sizeof(t));
        y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[t]);
    }
}
```

**Cost per element:** f32→f16 conversion (1 F16C instruction) + array index +
f16→f32 conversion (1 F16C instruction). Zero exp/tanh calls at inference time.

**Scalar formula (reference only):**
`gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * x * (1 + 0.044715 * x²)))`

---

### 1.5 GELU Quick — Precomputed F16 Table (128 KB)

**Source:** `ggml/src/ggml-cpu/vec.h`, lines 1028–1043

Same table strategy as GELU, with a separate 65536-entry table:

```c
extern ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];
```

**Scalar formula:** `gelu_quick(x) = x * sigmoid(1.702 * x) = x / (1 + exp(-1.702 * x))`

The table is built at startup from this formula. During inference: table lookup only.

---

### 1.6 tanh — No Fast Approximation in Main CPU Path

**Source:** `ggml/src/ggml-cpu/vec.h`, line 909

```c
inline static void ggml_vec_tanh_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) y[i] = tanhf(x[i]);  // standard libm
}
```

`tanhf` is only called for the `GGML_UNARY_OP_TANH` graph node (standalone op,
not inside an activation). In the flash attention path (`ops.cpp:8752`) logit
softcapping uses `tanhf(s)` directly — no approximation.

GELU avoids tanh entirely by using the precomputed table.

---

## 2. Tempura's Current Implementations

### 2.1 `ggml_exp()` — Scalar Polynomial (Correct, Not Vectorized)

**Source:** `kernels/src/portable_math.rs`, line 43

Tempura has an exact scalar translation of `ggml_v_expf` AVX2 with the same
constants and polynomial structure:

```rust
pub fn ggml_exp(x: f32) -> f32 {
    const R: f32 = 12582912.0;           // 0x1.8p23
    const LOG2E: f32 = f32::from_bits(0x3fb8aa3b);  // 0x1.715476p+0
    const LN2_HI: f32 = f32::from_bits(0x3f317200); // 0x1.62e4p-1
    const LN2_LO: f32 = f32::from_bits(0x35bfbe8e); // 0x1.7f7d1cp-20
    // ... same P0–P4 constants, same Horner evaluation ...
}
```

**Gap:** runs 1 float at a time. No `std::arch` SIMD. Same math, 8× worse
throughput on AVX2, 16× worse on AVX-512.

Also has a second, independent exp: `portable_exp()` (line 5) — degree-6
Cody-Waite with different constants. This is less accurate than `ggml_exp` and
should not be used in the attention hotpath.

---

### 2.2 Softmax — Scalar `ggml_exp`, No SIMD

**Source:** `kernels/src/attention.rs`, lines 151, 241, 331, 586, 746, 820, 891

```rust
let val = crate::portable_math::ggml_exp(*sc - max_score);
```

Mathematically matches llama.cpp. No SIMD.

**Estimated cost:** 8× slower than llama.cpp AVX2 softmax for long sequences
(each attention head iterates over `kv_len` scores scalar).

---

### 2.3 SiLU — Rust Standard `.exp()` (Libm, Not `ggml_exp`)

**Source:** `kernels/src/ggml/ops_llama.rs`, line 40

```rust
pub fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())   // Rust f32::exp() → libm expf
}
```

This uses Rust's standard library `f32::exp()`, which calls libm `expf`.
`portable_math.rs` has `portable_silu()` using `portable_exp()` (degree-6),
but `ops_llama::silu_f32` does not call it.

**Gap vs llama.cpp:**
- llama.cpp: `ggml_v_silu` processes 8 floats with `ggml_v_expf` polynomial
- tempura: 1 float via libm `expf` (no polynomial approximation, full precision)

---

### 2.4 GELU — Rust Standard `.tanh()` (No Table)

**Source:** `kernels/src/ggml/ops_llama.rs`, line 21

```rust
pub fn gelu_f32(x: f32) -> f32 {
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}
```

Uses Rust's standard `f32::tanh()`. No table lookup.

**Gap vs llama.cpp:** llama.cpp's `ggml_vec_gelu_f32` with `GGML_GELU_FP16`
is a single f16 array index — zero exp/tanh cost. Tempura pays full libm tanh
per element.

`activations.rs::gelu()` uses `portable_tanh()` which calls `portable_exp()`
(two calls for `exp(2x)`). Still no table.

---

### 2.5 GELU Quick — Rust Standard `.exp()` (No Table)

**Source:** `kernels/src/ggml/ops_llama.rs`, line 27

```rust
pub fn gelu_quick_f32(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-GELU_QUICK_COEF * x).exp()))
}
```

Uses libm `expf` via Rust `.exp()`. No table.

---

## 3. Gap Summary

| Function       | llama.cpp                                    | Tempura                              | Gap                                  |
|----------------|----------------------------------------------|--------------------------------------|--------------------------------------|
| `exp` (vector) | `ggml_v_expf`: 4-term polynomial, 8×/iter    | `ggml_exp`: same polynomial, scalar  | 8× throughput (no SIMD)              |
| Softmax        | `ggml_v_expf` vectorized + f64 sum           | `ggml_exp` scalar + f64 sum          | 8× throughput in exp loop            |
| SiLU           | `ggml_v_silu`: 8 floats/iter, polynomial exp | `silu_f32`: libm expf, scalar        | SIMD gap + slower exp (libm vs poly) |
| GELU           | 128 KB f16 table lookup, zero exp/tanh       | libm tanh, scalar                    | Many cycles vs 2 F16C instructions   |
| GELU Quick     | 128 KB f16 table lookup, zero exp            | libm expf, scalar                    | Many cycles vs 2 F16C instructions   |
| tanh (raw)     | libm tanhf                                   | libm f32::tanh()                     | Parity (neither approximates)        |

---

## 4. Recommended Fast-Math Ports

In priority order for inference throughput:

### 4.1 GELU / GELU Quick Table (Highest ROI)

Port `ggml_table_gelu_f16` and `ggml_table_gelu_quick_f16`:

```rust
// In a new file: kernels/src/gelu_table.rs
pub static GELU_F16_TABLE: OnceLock<[u16; 65536]> = OnceLock::new();

pub fn gelu_table_f32(x: f32) -> f32 {
    if x <= -10.0 { return 0.0; }
    if x >=  10.0 { return x; }
    let fp16_bits = f32_to_f16_bits(x);  // F16C or software
    let out_bits  = GELU_F16_TABLE.get().unwrap()[fp16_bits as usize];
    f16_bits_to_f32(out_bits)
}
```

Complexity: medium. Requires a 128 KB initialization pass at model load time.
Use F16C instructions (`vcvtps2ph` / `vcvtph2ps`) for the conversions.

### 4.2 SIMD `ggml_v_expf` for Softmax (Most Critical for Decode Throughput)

Port `ggml_v_expf` to Rust `std::arch` for x86 (AVX2/AVX-512) and ARM NEON.
Wrap the existing `ggml_exp` scalar in an `#[cfg(target_feature = "avx2")]` AVX2
version that processes 8 floats at once. Replace the scalar loop in all
softmax paths in `attention.rs`.

```rust
#[cfg(target_feature = "avx2")]
unsafe fn ggml_v_expf_avx2(x: __m256) -> __m256 { ... }

pub fn vec_soft_max_f32(y: &mut [f32], x: &[f32], max: f32) -> f64 {
    let mut i = 0;
    let mut sum = 0f64;
    // AVX2 path: 8 floats/iter
    #[cfg(target_feature = "avx2")]
    while i + 7 < x.len() { ... }
    // scalar tail
    for j in i..x.len() { ... }
    sum
}
```

### 4.3 `silu_f32` — Use `ggml_exp` Instead of Libm

Immediate fix, zero complexity:

```rust
// ops_llama.rs
pub fn silu_f32(x: f32) -> f32 {
    x / (1.0 + crate::portable_math::ggml_exp(-x))
}
```

Switches from libm `expf` to the polynomial approximation, matching
llama.cpp's scalar fallback. Should improve SiLU accuracy for bit-identical
matching and reduce latency on CPUs without AVX2.

### 4.4 Unify SiLU Implementations

`portable_math::portable_silu` and `ops_llama::silu_f32` use different exp
functions. Consolidate: `ops_llama::silu_f32` should call `portable_math::ggml_exp`.

---

## 5. Implementation Notes

**The f16 table lookup is the same class of optimization as `ggml_table_f32_f16`:**
- `ggml_table_f32_f16`: indexed by f32 bits (256 KB) → convert f16→f32 at the cost of a lookup
- `ggml_table_gelu_f16`: indexed by f16 bits (128 KB) → compute GELU at the cost of a lookup

Both trade startup memory for zero-cost per-element computation. Both use the
observation that the f16 domain has only 65536 distinct values.

**`ggml_v_expf` and `ggml_exp` are the same algorithm:** the scalar `ggml_exp`
in `portable_math.rs` is already the correct translation. The work for the SIMD
version is to wrap the same constants and polynomial in `_mm256_*` intrinsics.
The constants are already in the codebase; only the loop structure changes.

**Kani / Verus coverage:** `ggml_v_expf` polynomial constants are provable by
Verus (pure integer/bit arithmetic on the exponent field). A new Kani harness
should bound-check the index `((ni + 127) as u32) << 23` for the scale factor.
The GELU table initialization loop is a straightforward Kani target for bounds.
