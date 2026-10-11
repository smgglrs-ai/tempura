---
title: "oxide GEMV — Next Optimizations"
description: "Concrete next steps to close the 7-9× gap to llama.cpp on RTX 5090"
weight: 10
---

# oxide GEMV — Next Optimizations

**Current state** (2026-10-11, RTX 5090 Blackwell sm_120):

| Model | Format | tok/s decode | ITL P50 | Gap vs llama.cpp |
|-------|--------|-------------|---------|-----------------|
| Bonsai-1.7B | Q2_0 | 30 | 33ms | 7× |
| Llama-3.1-8B | Q4K+Q6K | 24 | 41ms | 8.7× |
| llama.cpp ref | — | 209 | 5ms | 1× |

All oxide kernels live in `tools/oxide-gemv*/src/main.rs`. Build instructions are in CLAUDE.md ("GPU Kernels" section) and `bench-gpu-rtx5090.json`.

---

## Roofline position

RTX 5090: 1.79 TB/s memory bandwidth, ~120 TFLOPS FP32.

**Q2_0 Bonsai-1.7B**: K=2048, N=2048 per projection, 7 GEMVs × 28 layers.
- Weight bandwidth per token: 7 × 28 × 2048 × (34/128) bytes ≈ 211 MB
- Theoretical at 1.79 TB/s: 211 MB / 1790 GB/s = **0.118ms → 43 tok/s ceiling**
- Current: 33ms = 30 tok/s → **70% of bandwidth limit**. Close to roof.

**Q4K Llama-3.1-8B**: K=4096 avg, N varies (1024–14336), 7 GEMVs × 32 layers.
- Weight bandwidth per token: ~17 GB (packed 168 bytes/block format = 17% bloat)
- Theoretical at 1.79 TB/s: 17 GB / 1790 GB/s = **9.5ms → 105 tok/s ceiling**
- Current: 41ms = 24 tok/s → **23% of bandwidth limit**. Large headroom.

The Q4K efficiency is much lower because of packed format overhead and poor ILP in the nibble extraction loop. Q2_0 is near the bandwidth ceiling — dual-output warp is its main remaining lever.

---

## Optimization 1: Raw quantized bytes on GPU (HIGH IMPACT — Q4K VRAM + bandwidth)

**Problem**: The Q4K packed GPU format uses 168 bytes/block vs 144 bytes in the raw GGUF format — 17% more bandwidth and VRAM.

```
Current: [N, K_BLOCKS, 42 u32] = 168 bytes/block
GGUF raw: [d f16, dmin f16, scales 12B, qs 128B] = 144 bytes/block
```

With 168 bytes: Llama-3.1-8B → 17 GB VRAM. With 144 bytes: → ~5 GB VRAM.

**Fix**:
1. Remove the pack step in `engine/src/cutile_gpu_forward.rs::maybe_upload_q4k`
2. Upload raw GGUF bytes directly (`cudaMemcpy` from the mmap region via `qw.get_q4k(name)`)
3. Update `tools/oxide-gemv-q4k/src/main.rs` to decode the raw GGUF layout inline:

```rust
// Raw GGUF Q4K block layout (144 bytes = 36 u32):
//   u32[0]    = f16 d bits + f16 dmin bits (packed into one u32)
//   u32[1..4] = 12 bytes of packed 6-bit scales (4 u32s)
//   u32[4..36]= 32 u32s of nibbles (256 nibbles, 8 per u32)
```

The `q4k_unpack_scales` function in `kernels/src/cutile_quantized_gemm.rs` already shows the 6-bit scale unpacking logic. Port it to the device kernel.

**Expected impact**: Llama-3.1-8B VRAM 17 GB → ~5 GB, bandwidth reduced by 17%, decode 24 → ~28 tok/s.

---

## Optimization 2: Two-output-per-warp (MEDIUM IMPACT — compute)

**Problem**: Each warp makes a complete pass through `x[]` to compute ONE output neuron. Adjacent output neurons read the same activation — if they shared the pass, we'd get 2× arithmetic intensity for the same bandwidth cost.

**Fix** in `tools/oxide-gemv-q2k/src/main.rs` (and Q4K/Q6K):

```rust
// Split warp: lanes 0-15 → output n, lanes 16-31 → output n+1
let n_out_0 = base + 2 * warp_id;
let n_out_1 = n_out_0 + 1;
let lane = threadIdx_x() % 32;
let my_half = lane / 16;  // 0 = first output, 1 = second
let my_lane = lane % 16;

// Both halves accumulate independently; reduce over half-warp
let mut acc_0 = 0.0f32;
let mut acc_1 = 0.0f32;

// Scale broadcasts: lane 0 → all (existing); same scale serves both outputs
// Nibbles: packed[n_out_0 * K_BLOCKS + blk * ...] vs packed[n_out_1 * ...]
// Different rows → lanes 0-15 load row n_out_0, lanes 16-31 load row n_out_1

// Warp-level reduce over 16 lanes (half-warp reduce)
acc_0 = reduce_sum_f32_half(acc_0);  // shfl over 16 lanes
acc_1 = reduce_sum_f32_half(acc_1);

if my_lane == 0 {
    out[n_out_0] = acc_0;
    out[n_out_1] = acc_1;
}
```

Grid becomes `ceil(N/2) / WARPS_PER_BLOCK` blocks. Halves kernel launch overhead with the same memory traffic.

**Expected impact**: Q2_0 30 → 35-40 tok/s; Q4K 24 → 28-32 tok/s.

**Prerequisite**: `reduce_sum_f32` over 16 lanes (half-warp). Use shfl_down with mask `0x0000FFFF` for lanes 0-15 and `0xFFFF0000` for lanes 16-31. Check `cuda-oxide/crates/cuda-device/src/warp.rs` for available shfl variants.

---

## Optimization 3: Float4 reads for Q4K nibble words (MEDIUM IMPACT — bandwidth ILP)

**Problem**: 32 nibble u32 words loaded one-at-a-time. 32 scalar `ld.global.b32` instructions vs 8 `ld.global.v4.b32` (128-bit, LDG.E.128).

**Fix**: Cast `*const u32` to `*const U32x4` where `U32x4 = CuSimd<u32, 4>`:

```rust
// In tools/oxide-gemv-q4k/src/main.rs device kernel:
// Load 4 nibble words at once
let qs_base = packed.add((base + 4 + word_group * 4) as usize) as *const CuSimd<u32, 4>;
let v4 = *qs_base;  // LDG.E.128 — 128 bits = 4 u32s = 32 nibbles
let w0 = v4.x; let w1 = v4.y; let w2 = v4.z; let w3 = v4.w;
// Process w0..w3 together (32 nibbles, 4 groups of 8)
```

`CuSimd<u32, 4>` should be available via `cuda-device/src/cusimd.rs` (check if `U32x4` is defined alongside `Float4`). If not, define it as `repr(C)` struct of 4 u32s with the `Vector` trait from `cuda-device/src/vector.rs`.

**Expected impact**: ~25% Q4K bandwidth improvement → ~30 tok/s.

---

## Optimization 4: Q4K/Q6K prefill oxide GEMM (HIGH IMPACT — prefill throughput)

**Problem**: Prefill (seq_len > 1) uses cuTile JIT → 7 tok/s for Llama-3.1-8B prefill. llama.cpp achieves 7335 tok/s pp128 — a 1200× gap.

**Fix**: Implement a batched oxide GEMM for M tokens:

```rust
// tools/oxide-gemv-q4k/src/main.rs — add:
#[kernel]
pub unsafe fn q4k_gemm_prefill(
    out: *mut f32,         // [M, N]
    x: *const f32,         // [M, K] 
    packed: *const u32,    // [N, K_BLOCKS, 42]
    m: u32, n: u32, k_blocks: u32,
) {
    // Block handles TM × TN tile of output
    // Each thread computes one output: out[token_idx, neuron_idx]
    // Inner loop over K blocks with shfl scale sharing
}
```

Grid: `(ceil(N/TN), ceil(M/TM))`. TM=4, TN=32 is a reasonable start.

**Expected impact**: Prefill 7 → 100+ tok/s (approach llama.cpp territory).

---

## Optimization 5: CUDA graph with device-side kv_len (LOW-MEDIUM — latency)

**Problem**: `cuGraphExecUpdate_v2` in `engine/src/cutile_gpu_forward.rs::forward_decode_with_graph` costs ~200μs per step re-patching kv_len. Saves only 0.3ms total vs full dispatch.

**Fix**:
1. Add `d_kv_len: Arc<Tensor<i32>>` field to `CutileGpuModel`
2. Modify `cutile_decode_attn` in `kernels/src/cutile_flash_attention.rs` to read kv_len from device:
   ```rust
   // In the kernel:
   let kv_len: i32 = tile_to_scalar(kv_len_dev.partition(shape![1]).load([0i32]));
   ```
3. Before each `cuGraphLaunch`: write `kv_len_host` to `d_kv_len` via tiny H→D memcpy (< 1μs)
4. Graph captured once; no `cuGraphExecUpdate_v2` needed

**Expected impact**: ~0.5ms/token → ~35 tok/s at current Q2_0 performance. More meaningful as other kernels improve.

**Reference**: `engine/src/cutile_gpu_forward.rs::forward_decode_with_graph` — the infrastructure exists, just needs kv_len moved to device.

---

## Optimization 6: Occupancy tuning (LOW — quick experiment)

Change `WARPS_PER_BLOCK` in each oxide kernel:

```rust
// tools/oxide-gemv-q4k/src/main.rs
const WARPS_PER_BLOCK: u32 = 4;  // try 2, 4, 8
```

Rebuild and benchmark. sm_120 has large register files; `WARPS_PER_BLOCK=2` may increase registers-per-warp and reduce register spilling, improving the inner loop.

**Expected impact**: 5-15% depending on register pressure profile.

---

## Implementation priority

1. **Raw bytes on GPU** (Opt 1) — biggest VRAM savings, straightforward to implement
2. **Two-output warp** (Opt 2) — 2× arithmetic intensity, pure kernel change
3. **Q4K prefill GEMM** (Opt 4) — closes the massive prefill gap
4. **Float4 nibbles** (Opt 3) — bandwidth optimization for Q4K
5. **Device kv_len** (Opt 5) — enables true single-capture CUDA graphs
6. **Occupancy tuning** (Opt 6) — quick experiment, 10 minutes

Implementing Opts 1-3 together should push Q4K Llama-3.1-8B from 24 → 60+ tok/s, reducing the gap to ~3× vs llama.cpp.

---

## How to rebuild oxide kernels after changes

```sh
export CUDA_OXIDE_BACKEND=~/Code/github.com/NVLabs/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
WDIR=/home/fdupont/Code/github.com/smgglrs-ai/tempura

for crate in oxide-gemv oxide-gemv-q4k oxide-gemv-q6k; do
  cd $WDIR/tools/$crate && cargo oxide build && cd ../..
done

for d in $WDIR/target/release/build/tempura-kernels-*/out; do
  cp $WDIR/tools/oxide-gemv/oxide_gemv.ptx         $d/oxide_q2_gemv.ptx  2>/dev/null
  cp $WDIR/tools/oxide-gemv-q4k/oxide_gemv_q4k.ptx $d/oxide_q4k_gemv.ptx 2>/dev/null
  cp $WDIR/tools/oxide-gemv-q6k/oxide_gemv_q6k.ptx $d/oxide_q6k_gemv.ptx 2>/dev/null
done

cargo run --manifest-path $WDIR/tools/bench-gpu/Cargo.toml --release \
  --bin bench-real-model -- /data/models/Ternary-Bonsai-1.7B-Q2_0.gguf
cargo run --manifest-path $WDIR/tools/bench-gpu/Cargo.toml --release \
  --bin bench-real-model -- ~/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```
