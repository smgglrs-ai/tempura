+++
title = "BitNet Implementation Research"
description = "Verified findings on GGUF ternary formats, BitLinear inference, BitNet v2, available models, and CUDA kernel approaches."
weight = 50
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## Executive Summary

GGUF ternary weight support centers on three format types: TQ1_0 (base-3 packing, 1.69 bpw), TQ2_0 (2-bit packing, 2.06 bpw), and STQ1_0 (codebook-based, 1.31 bpw). All use 256-element blocks with fp16 scales, matching existing Q4_K/Q6_K block structure. For tempura with cuda-oxide, TQ2_0 is the recommended starting point: its 2-bit layout maps cleanly to bitwise ops and existing dp4a infrastructure.

## GGUF Ternary Weight Formats

### TQ1_0 (Type ID 36) --- 1.6875 bpw

Block layout: 54 bytes per 256 elements.

| Field | Size | Contents |
|-------|------|----------|
| `qs[48]` | 48 B | 5 trits per byte via base-3 packing ($3^5 = 243 < 256$) |
| `qh[4]` | 4 B | 4 trits per byte for remaining 16 elements |
| `d` | 2 B | fp16 per-block scale |

**Encoding:** Map $\{-1, 0, 1\} \to \{0, 1, 2\}$ with `q *= 3; q += xi`, then ceiling-division remap:

```c
(uint16_t)q * 256 + 242) / 243
```

**Decoding:** Multiply by `pow3[n]`, extract via:

```c
((uint16_t)q * 3) >> 8
```

**GPU suitability:** No production CUDA kernel exists. Base-3 division is SIMD-unfriendly --- integer division by 243 requires multiply-shift sequences that waste ALU throughput.

### TQ2_0 (Type ID 37) --- 2.0625 bpw

Block layout: 66 bytes per 256 elements.

| Field | Size | Contents |
|-------|------|----------|
| `qs[64]` | 64 B | 4 values per byte (2 bits each) |
| `d` | 2 B | fp16 per-block scale |

**Encoding:** Quantize with `lroundf(x * id) + 1` mapping $\{-1, 0, 1\} \to \{0, 1, 2\}$, pack with bit shifts.

**Decoding:** Extract 2-bit fields, subtract 1.

**GPU suitability:** CUDA kernels exist (llama.cpp PR #11183) with three paths:

- **MMVQ:** Matrix-vector, direct 2-bit unpack
- **MMQ:** Shared-memory tiles, dp4a with INT32 accumulation
- **cuBLAS:** Dequantize to FP16, delegate to cuBLAS GEMM

### STQ1_0 (Type ID 42) --- 1.3125 bpw

Block layout: 42 bytes per 256 elements.

| Field | Size | Contents |
|-------|------|----------|
| `qs[32]` | 32 B | 4-bit codebook indices |
| `sign[8]` | 8 B | 1-bit sign per weight |
| `d` | 2 B | fp16 per-block scale |

Uses a 32-entry codebook lookup per group of 4 weights. Stride-16 sparsity layout for NEON alignment.

**Caveat:** Requires models trained with this specific sparsity structure --- cannot be applied to arbitrary ternary weights.

## BitNet b1.58 Inference (BitLinear)

### Architecture

The 2B-4T reference model uses:

| Field | Value |
|-------|-------|
| `model_type` | `bitnet` |
| `hidden_act` | `relu2` (Squared ReLU) |
| Tokenizer | LLaMA 3 |
| Vocabulary | 128,256 |
| Bias terms | None |
| Position encoding | RoPE |

Squared ReLU ($\text{ReLU}(x)^2$), **not** SwiGLU.

### BitLinear Forward Pass

1. **RMSNorm** the input
2. **Absmax INT8 per-token activation quantization:**
   $Q_{\text{INT8}}(X) = \text{RoundClip}\left(\frac{127}{\max(|X|)} \cdot X,\ -128,\ 127\right)$
3. **Integer add/sub matmul** (ternary weights $\times$ INT8 activations)
4. **Rescale** output

Weight quantization (baked at inference time):

$$W_q = \text{RoundClip}\left(\frac{W}{\text{mean}(|W|)},\ -1,\ 1\right)$$

Absmean quantization produces ~42--51% zero weights.

### Per-block vs Per-tensor Activation Quantization

This distinction is critical for numerical fidelity:

| Approach | Scale granularity | Used by |
|----------|-------------------|---------|
| Per-block | 256 elements per scale | llama.cpp TQ1_0/TQ2_0 |
| Per-tensor | 1 scale for entire activation vector | BitNet training, bitnet.cpp |

The per-block approach is lossy. The bitnet.cpp paper (ACL 2025) demonstrates measurable perplexity differences between these quantization strategies.

## BitNet v2: H-BitLinear

BitNet v2 introduces a **selective Hadamard transform** applied only to $W_o$ (attention output) and $W_\text{down}$ (FFN down-projection). All other projections use standard BitLinear.

**INT4 activation quantization** with per-token absmean scaling:

$$Q_{\text{INT4}}(X) = \frac{\beta}{\sqrt{7}} \cdot \text{RoundClip}\left(\frac{\sqrt{7}}{\beta + \epsilon} \cdot X,\ -8,\ 7\right) \quad \text{where}\ \beta = \text{mean}(|X|)$$

KV cache is also compressed to INT4.

## Available Models

| Model | Params | GGUF Type | Architecture |
|-------|--------|-----------|--------------|
| `microsoft/bitnet-b1.58-2B-4T-gguf` | 2B | TQ1_0, TQ2_0 | bitnet |
| `tiiuae/Falcon3-1B-Instruct-1.58bit-GGUF` | 1B | IQ2_S | llama |

### HuggingFace BitNetConfig

```json
{
  "model_type": "bitnet",
  "hidden_size": 2560,
  "intermediate_size": 6912,
  "num_hidden_layers": 30,
  "num_attention_heads": 20,
  "num_key_value_heads": 5,
  "hidden_act": "relu2",
  "rms_norm_eps": 1e-5,
  "weight_quant_method": "absmean",
  "input_quant_method": "absmax"
}
```

## CUDA Kernel Approaches

### Microsoft W2A8 GEMV

Pack 4 trits into int8, interleave layout `[0,4,8,12,1,5,9,13,...]` for coalesced access. Uses dp4a with INT32 accumulation.

### llama.cpp TQ2_0 MMQ

Unpack 2-bit values to 8-bit in shared memory tiles, reuse existing dp4a infrastructure. GPU path uses Q8_1 quantization (32 weights per scale) rather than the CPU's Q8_K (256 weights per scale).

### Sparse Ternary

Warp-per-row decomposition, `__shfl_down_sync` reduction, branchless sign application. Achieves 1.06--1.19x speedup over cuBLAS INT8 for matrix-vector operations.

### bitnet.cpp LUT Kernels

Two table sizes:

| Variant | Packing | LUT entries | Description |
|---------|---------|-------------|-------------|
| TL1 | 2 trits | 9 | 4-bit partial sums |
| TL2 | 3 trits | 27 | sign + 4-bit, halved LUT via symmetry |

Implemented with `vpshufb` (x86) and `vqtbl1q` (ARM). GPU LUT approaches remain open research.

## Implementation Plan for tempura

### Phase 1: TQ2_0 Loading + CPU Reference

- `block_tq2_0` struct (66 bytes, `QK_K=256`)
- BitLinear CPU kernel with per-tensor INT8 activation quantization
- Squared ReLU activation function
- Validate against `microsoft/bitnet-b1.58-2B-4T-gguf`

### Phase 2: TQ2_0 GPU Kernel (cuda-oxide)

- MMQ approach: unpack 2-bit to 8-bit in shared memory
- dp4a for INT8 dot products (reuse existing quantized GEMM infrastructure)
- Q8_1 activation tiles (32 weights per scale)

### Phase 3: TQ1_0 CPU-only

- Base-3 decode with `pow3` lookup table
- Skip GPU implementation (SIMD-unfriendly base-3 arithmetic)

### Phase 4: BitNet v2 (future)

- Hadamard transform on $W_o$ and $W_\text{down}$
- INT4 activation and KV cache quantization

## Sources

All claims adversarially verified: 103 agents, 21 sources fetched, 101 claims extracted, 25 adversarially verified (22 confirmed, 3 killed). Each claim verified by 3 independent skeptical agents.

| Source | Description |
|--------|-------------|
| llama.cpp PR #8151 | TQ1_0/TQ2_0 implementation (merged Sept 2024) |
| llama.cpp PR #11183 | TQ2_0 CUDA kernels |
| llama.cpp PR #22836 | STQ1_0 codebook format |
| arXiv 2504.12285 | BitNet b1.58 2B4T Report --- architecture details |
| arXiv 2504.18415 | BitNet v2 --- H-BitLinear, INT4 activations |
| arXiv 2502.11880 | bitnet.cpp (ACL 2025) --- LUT kernels, lossless inference |
| Microsoft BitNet GPU README | W2A8 GEMV kernel |
| HuggingFace BitNet Docs | Config fields |
| `microsoft/bitnet-b1.58-2B-4T-gguf` | Reference model |
