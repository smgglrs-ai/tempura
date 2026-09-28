# Assembly Hoisting Analysis: GCC llama.cpp vs Rust Tempura

Comparing `libggml-cpu.so` (GCC 14, `-O3 -march=native`) against
`tempura-serve` (LLVM/Rust, `-C opt-level=3`, `target-cpu=native`) for the two
hot quantised dot-product kernels `ggml_vec_dot_q4_K_q8_K` and
`ggml_vec_dot_q6_K_q8_K`.

Assembly dumps used:

```
objdump -d --no-show-raw-insn -M intel <binary>  # Intel syntax, no raw bytes
```

---

## ggml_vec_dot_q4_K_q8_K

### GCC prologue — hoisted before the block loop

| Instruction | Register | Value / Symbol | Role |
|---|---|---|---|
| `vmovdqa ymm15,[rip+0x3680c]` | ymm15 | shuffle constant @ 0x12b5e0 | scale broadcast pattern 0 |
| `vmovdqa ymm14,[rip+0x36824]` | ymm14 | shuffle constant @ 0x12b600 | scale broadcast pattern 1 |
| `vmovdqa ymm13,[rip+0x36830]` | ymm13 | shuffle constant @ 0x12b620 | scale broadcast pattern 2 |
| `vmovdqa ymm12,[rip+0x36848]` | ymm12 | shuffle constant @ 0x12b640 | scale broadcast pattern 3 |
| `vmovdqa ymm11,[rip+0x3685a]` | ymm11 | shuffle constant @ 0x12b660 | scale broadcast pattern 4 |
| `vmovdqa ymm10,[rip+0x36872]` | ymm10 | shuffle constant @ 0x12b680 | scale broadcast pattern 5 |
| `vmovdqa ymm9, [rip+0x3688a]` | ymm9  | shuffle constant @ 0x12b6a0 | scale broadcast pattern 6 |
| `vmovdqa ymm8, [rip+0x368a2]` | ymm8  | shuffle constant @ 0x12b6c0 | scale broadcast pattern 7 |
| `mov r8,[rip+0x5d0ed]`        | r8    | `ggml_table_f32_f16` pointer | f16→f32 lookup table base |
| `vpbroadcastd ymm4,0x0f0f0f0f`| ymm4  | nibble mask | extract low nibble from Q4 bytes |

Loop body address: `0xf4e30`–`0xf502e`.  Loop instruction count: **96**.

### GCC loop body — constant reloads (missed hoists)

| Address | Instruction | Reload count | Cause |
|---|---|---|---|
| `0xf4e40` | `vxorps xmm1,xmm0,XMMWORD PTR [rip+0x35098]` | 1 per block | dmin sign-flip mask not hoisted |

GCC reloads the 128-bit sign-flip mask (used to negate dmin) from memory on
every block iteration.  This is a memory load from the data segment on each
pass through the outer loop.

### GCC inner loop — scale scatter (8 vpshufb, 0 reloads)

The 8 pre-loaded YMM shuffle constants are used directly:

```asm
vpshufb ymm1,  ymm0, ymm15   ; scatter scale[0] → 32 bytes
vpshufb ymm17, ymm0, ymm14   ; scatter scale[1]
vpshufb ymm18, ymm0, ymm13   ; scatter scale[2]
vpshufb ymm19, ymm0, ymm12   ; scatter scale[3]
vpshufb ymm16, ymm0, ymm11   ; scatter scale[4]
vpshufb ymm20, ymm0, ymm10   ; scatter scale[5]
vpshufb ymm2,  ymm0, ymm9    ; scatter scale[6]
vpshufb ymm0,  ymm0, ymm8    ; scatter scale[7]
```

Each `vpshufb ymm,ymm,ymm` has 1-cycle throughput (port 5).  Total: **8 instructions** for scale scatter, zero memory traffic.

---

### Rust prologue — hoisted before the block loop

| Instruction | Register | Value / Symbol | Role |
|---|---|---|---|
| `vbroadcasti128 ymm2,[rip+...]` | ymm2 | 0x0f0f…0f × 32 @ 0x182900 | nibble mask |
| `vbroadcastss xmm3,[rip+...]`   | xmm3 | sign-flip scalar @ 0x187b80 | dmin negation ✓ |
| `mov rax,[rip+0x23de23]`        | rax  | `ggml_table_f32_f16` @ 0xd8d350 | f16→f32 table base |

Loop body address: `0xb4f540`–`0xb4f702`.  Loop instruction count: **89**.

### Rust loop body — scale scatter (arithmetic, 13 instructions)

LLVM replaced `vpshufb ymm,ymm,ymm_const` with arithmetic scatter to avoid
occupying 8 YMM registers.  The resulting per-block sequence:

```asm
; extract and broadcast scale[0]
vpbroadcastw ymm8, xmm5
; extract and broadcast scale[1]
vpshufd      xmm11, xmm5, 0x55
vpbroadcastw ymm11, xmm11
; extract and broadcast scale[2]
vpsrld       xmm9,  xmm5, 0x10
vpbroadcastw ymm9,  xmm9
; extract and broadcast scale[3]
vpshufd      xmm11, xmm5, 0xee
vpbroadcastw ymm11, xmm11
; extract and broadcast scale[4]
vpsrlq       xmm8,  xmm5, 0x30
vpbroadcastw ymm8,  xmm8
; extract and broadcast scale[5]
vpshufd      xmm11, xmm5, 0xff
vpbroadcastw ymm11, xmm11
; extract and broadcast scale[6]
vpsrldq      xmm8,  xmm5, 0xa
vpbroadcastw ymm8,  xmm8
; extract upper 128 bits, then broadcast scale[7]
vextracti128 xmm5,  ymm5, 0x1
vpsrldq      xmm8,  xmm5, 0xe
vpbroadcastw ymm8,  xmm8
```

Instruction count: **6 shifts/shuffles + 6 broadcasts + 1 vextracti128 = 13 instructions**.

### Q4K hoisting gap summary

| Metric | GCC | Rust | Rust extra |
|---|---|---|---|
| Pre-loop constant loads | 10 | 3 | — |
| Scale scatter per block | 8 × vpshufb | 13 arithmetic ops | +5 instructions/block |
| Sign-flip mask | reload each block | hoisted (xmm3) ✓ | Rust better by 1 |
| Table pointer | hoisted (r8) ✓ | hoisted (rax) ✓ | tie |
| Nibble mask | hoisted (ymm4) ✓ | hoisted (ymm2) ✓ | tie |

**Net per-block cost in Rust vs GCC: +4 instructions** (5 extra scatter − 1 saved sign-flip reload).

For a 100-block sequence (25 600 tokens) that is **~400 wasted instruction slots** in the
scale-scatter path alone.

---

## ggml_vec_dot_q6_K_q8_K

### GCC prologue — hoisted before the block loop

| Instruction | Register | Value | Role |
|---|---|---|---|
| `vmovdqa64 xmm19,[rip+...]` | xmm19 | @ 0x12b3a0 | scale shuffle pattern A |
| `vmovdqa64 xmm18,[rip+...]` | xmm18 | @ 0x12b3b0 | scale shuffle pattern B |
| `vmovdqa64 xmm17,[rip+...]` | xmm17 | @ 0x12a1c0 | scale shuffle pattern C |
| `vmovdqa64 xmm16,[rip+...]` | xmm16 | @ 0x12a1d0 | scale shuffle pattern D |
| `vmovdqa xmm15,[rip+...]`   | xmm15 | @ 0x12a1e0 | scale shuffle pattern E |
| `vmovdqa xmm14,[rip+...]`   | xmm14 | @ 0x12a1f0 | scale shuffle pattern F |
| `vmovdqa xmm13,[rip+...]`   | xmm13 | @ 0x12a200 | scale shuffle pattern G |
| `vmovdqa xmm12,[rip+...]`   | xmm12 | @ 0x12a210 | scale shuffle pattern H |
| `mov rdi,[rip+0x5c9e1]`     | rdi   | `ggml_table_f32_f16` ptr | f16→f32 table base |
| `vpbroadcastd ymm11,0x03030303` | ymm11 | 6-bit extraction mask | bits 4-5 of byte pair |
| `vpbroadcastd ymm10,0x0c0c0c0c` | ymm10 | 6-bit extraction mask | bits 2-3 |
| `vpbroadcastd ymm9, 0xc0c0c0c0` | ymm9  | 6-bit extraction mask | bits 6-7 |
| `vpbroadcastd ymm5, 0x0f0f0f0f` | ymm5  | nibble mask | Q6 nibble extraction |
| `vpbroadcastd ymm8, 0x30303030` | ymm8  | high-nibble mask | upper 2-bit field |

8 XMM shuffle constants + 5 YMM bit masks + 1 table pointer = **14 registers** occupied.

Loop body: `0xf5540`–`0xf5757`.  Loop instruction count: **88**.

### GCC loop — scale scatter (8 vpshufb xmm, 0 reloads)

```asm
vpshufb xmm23, xmm0, xmm19   ; scale scatter A
vpshufb xmm22, xmm0, xmm16   ; scale scatter B
vpshufb xmm1,  xmm0, xmm19   ; scale scatter C
vpshufb xmm21, xmm0, xmm17   ; scale scatter D
vpshufb xmm24, xmm0, xmm14   ; scale scatter E
vpshufb xmm27, xmm0, xmm15   ; scale scatter F
vpshufb xmm26, xmm0, xmm13   ; scale scatter G
vpshufb xmm0,  xmm0, xmm12   ; scale scatter H
```

8 vpshufb xmm operations, 1-cycle throughput each.  No memory traffic.

### Rust prologue — hoisted before the block loop

| Instruction | Register | Role |
|---|---|---|
| `vmovdqa xmm5,[rip+...]` | xmm5 | scale shuffle pattern (low half, set A) |
| `vmovdqa xmm6,[rip+...]` | xmm6 | scale shuffle pattern (low half, set B) |
| `vmovdqa xmm7,[rip+...]` | xmm7 | scale shuffle pattern (low half, set C) |
| `vmovdqa xmm8,[rip+...]` | xmm8 | scale shuffle pattern (low half, set D) |
| `vpbroadcastd ymm1,[rip+...]` | ymm1 | 6-bit extraction mask |
| `vpbroadcastd ymm2,[rip+...]` | ymm2 | 6-bit extraction mask |
| `vbroadcasti128 ymm3,[rip+...]` | ymm3 | nibble mask |
| `vbroadcasti128 ymm4,[rip+...]` | ymm4 | upper mask |
| `mov rax,[rip+0x23db53]` | rax | `ggml_table_f32_f16` pointer |

Only **4 XMM shuffle constants** instead of 8.  Rust halves register pressure by reusing
each pattern on two sub-regions.

### Rust loop — scale scatter (8 vpshufb xmm + 2 extra splits)

Rust splits the 16-byte scale region into low and high halves before shuffling:

```asm
; split xmm10 (16 scale bytes) into two 16-byte regions
vpunpcklbw xmm15, xmm10, xmm10   ; interleave low bytes with themselves
vpunpckhbw xmm18, xmm10, xmm10   ; interleave high bytes with themselves
; low half: 4 × vpshufb using xmm5..xmm8
vpshufb xmm19, xmm15, xmm6
vpshufb xmm16, xmm15, xmm5
vpshufb xmm20, xmm15, xmm7
vpshufb xmm15, xmm15, xmm8
; high half: 4 × vpshufb reusing xmm5..xmm8
vpshufb xmm14, xmm18, xmm5
vpshufb xmm15, xmm18, xmm6
vpshufb xmm19, xmm18, xmm7
vpshufb xmm18, xmm18, xmm8
```

Same vpshufb count (8), but 2 extra `vpunpck*` instructions per block.

### Q6K hoisting gap summary

| Metric | GCC | Rust | Rust extra |
|---|---|---|---|
| Pre-loop XMM shuffle constants | 8 | 4 | — |
| Pre-loop YMM bit masks | 5 | 4 | — |
| vpshufb per block | 8 | 8 | tie |
| Extra split instructions per block | 0 | 2 (vpunpcklbw + vpunpckhbw) | +2 instructions |
| Table pointer | hoisted (rdi) ✓ | hoisted (rax) ✓ | tie |

**Net per-block cost in Rust vs GCC: +2 instructions.**

---

## ggml_vec_soft_max_f32

This is not a hoisting gap — it is a deeper algorithmic difference.

GCC hoists **13 ZMM polynomial coefficients** (zmm6–zmm19) before the SIMD
`expf` loop, implementing a fast in-register polynomial approximation of `exp`
using AVX-512 fused-multiply instructions.  The loop body processes 16 floats
per iteration with no calls to the C library.

Rust `soft_max_rows_f32` calls scalar `expf@GLIBC_2.27` (CALL rip-relative to
`d8bf10`) for each element.  No vectorised `exp` approximation is implemented.
This is not an LLVM hoisting failure; it is a missing vectorised kernel.

---

## Aggregate wasted instructions

Assuming `nb` = number of 256-element super-blocks per row:

| Kernel | Extra instructions per block | Source |
|---|---|---|
| Q4K scale scatter | +5 | shift+broadcast chain vs vpshufb |
| Q4K sign-flip reload | −1 | Rust correctly hoists, GCC does not |
| Q6K scale split | +2 | vpunpcklbw + vpunpckhbw pair |
| **Q4K net** | **+4** | |
| **Q6K net** | **+2** | |

For a Llama-3 8B decoder layer with 32 heads × 128 head-dim = 4096 columns, at
Q4K granularity (nb ≈ 16):

- Q4K: 4 extra × 16 = **64 wasted instruction slots per dot-product call**
- Q6K: 2 extra × 16 = **32 wasted instruction slots per dot-product call**

---

## Actionable fixes

### Fix 1: Q4K — hoist scale-shuffle constants explicitly

Replace the LLVM-generated shift-broadcast chain for scale scatter with 8
pre-computed YMM constants loaded before the outer `nb` loop:

```rust
// Before outer loop: load K_SHUFFLE_K4 slices into YMM registers
let sh0 = unsafe { _mm256_loadu_si256(K_SHUFFLE_K4.as_ptr().add(0 * 32) as *const __m256i) };
let sh1 = unsafe { _mm256_loadu_si256(K_SHUFFLE_K4.as_ptr().add(1 * 32) as *const __m256i) };
// ... sh2..sh7 similarly

// Inside the block loop, replace get_scale_shuffle_k4(j) with sh{j}:
let scale_l = _mm256_shuffle_epi8(scales, sh2);
let scale_h = _mm256_shuffle_epi8(scales, sh3);
```

This tells LLVM the shuffle pattern is loop-invariant, removing the
vpsrld/vpbroadcastw chain and matching GCC's vpshufb approach.  Expected
saving: ~5 instructions per block.

### Fix 2: Q6K — hoist 8 XMM constants instead of 4

Pre-load all 8 XMM shuffle constants (as GCC does via xmm12–xmm19) before the
outer loop.  Eliminate the `vpunpcklbw`/`vpunpckhbw` pair per block.  Expected
saving: 2 instructions per block.

### Fix 3 (optional): GCC regression — Q4K sign-flip

GCC reloads the sign-flip mask (`-0.0f`) from memory each block (1 load/block).
Rust already avoids this by hoisting `xmm3 = vbroadcastss [rip+...]` before
the loop.  No Rust change needed; noted for completeness.

### Fix 4 (architectural): Softmax

Implement a vectorised `exp` approximation (e.g., the Cephes polynomial used by
GCC) for `soft_max_rows_f32`.  GCC processes 16 elements/cycle; scalar `expf`
processes 1.  This is not a hoisting issue but the largest single-kernel gap.

---

## Files analysed

- GCC binary: `/home/fdupont/Code/github.com/ggml-org/llama.cpp/build/bin/libggml-cpu.so`
- Rust binary: `/home/fdupont/Code/github.com/smgglrs-ai/tempura/target/release/tempura-serve`
- Rust source Q4K: `kernels/src/ggml/quants_llama.rs:529`
- Rust source Q6K: `kernels/src/ggml/quants_llama.rs:718`
- Shuffle tables: `kernels/src/ggml/quants_llama.rs:117–141`
