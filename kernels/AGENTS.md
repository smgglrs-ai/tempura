# GPU Kernels (cuda-oxide)

Build recipes and load-bearing quirks for the cuda-oxide kernel crates. This file is loaded automatically when working under kernels/.

We use upstream cuda-oxide main at `~/Code/github.com/NVLabs/cuda-oxide`
(switched from the fork 2026-10-29: no tempura kernel used the fork-only
features — mma.sync, invariant.load, multi-output ptx_asm — and upstream
gained libm-to-libdevice, u64 cooperative groups, and #[nounroll]).

**`cargo oxide build` uses a
cached backend `.so` from `~/.cargo/cuda-oxide/`** — point it at the local
upstream build:

```sh
# Build the codegen backend from upstream (once, or after upstream changes)
cd ~/Code/github.com/NVLabs/cuda-oxide/cuda-oxide/crates/rustc-codegen-cuda
T=~/.rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu
PATH=$T/bin:$PATH LD_LIBRARY_PATH=$T/lib RUSTC_BOOTSTRAP=1 $T/bin/cargo build --release
# PATH must put the toolchain's bin first -- see the E0463 note in the
# cargo oxide build section below (distro /usr/bin/rustc shadows it).

# Set this in every shell that runs cargo oxide build
export CUDA_OXIDE_BACKEND=~/Code/github.com/NVLabs/cuda-oxide/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
```

The UPSTREAM backend (`cuda-oxide/cuda-oxide/crates/rustc-codegen-cuda`) is
required for oxide-decode-attn (ex2_approx_f32) and the dp4a crates. Its
current build links the **1.99.1** toolchain's rustc driver (the upstream
tree pins channel "1.99.1"), and `cargo oxide` passes `-Z codegen-backend`,
which stable rejects — so:

```sh
export CUDA_OXIDE_BACKEND=~/Code/github.com/NVLabs/cuda-oxide/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
T=~/.rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu
export LD_LIBRARY_PATH=$T/lib:$LD_LIBRARY_PATH
PATH=$T/bin:$PATH RUSTC_BOOTSTRAP=1 RUSTUP_TOOLCHAIN=1.99.1 cargo oxide build
# If the backend .so is rebuilt from a different toolchain, match
# LD_LIBRARY_PATH/RUSTUP_TOOLCHAIN to that toolchain (the .so dlopens a
# versioned librustc_driver-*.so).
# PATH=$T/bin FIRST is load-bearing: with LD_LIBRARY_PATH set to the
# toolchain's lib, rustc MUST be that same toolchain's rustc. If /usr/bin/rustc
# (distro build) shadows it, every rustc_private crate fails E0463 ("can't
# find crate for rustc_driver") because the distro rustc loads the mismatched
# librustc_driver .so. Same applies to building the backend itself
# (2026-11-10, found when rebuilding after the upstream pull).
```

Build and benchmark GPU kernels:

```sh
# cuTile kernels (standard cargo build — JIT compiled at runtime)
cargo build -p tempura-bench-gpu --release --bin bench-real-model

# oxide hand-optimized GEMV kernels (require cargo oxide build)
# Build all oxide GEMV crates:
for crate in oxide-gemv oxide-gemv-q4k oxide-gemv-q6k oxide-decode-attn; do
  cd tools/$crate && cargo oxide build && cd ../..
done

# Copy PTX to build output directories:
for d in target/release/build/tempura-kernels-*/out; do
  cp tools/oxide-gemv/oxide_gemv.ptx       $d/oxide_q2_gemv.ptx  2>/dev/null
  cp tools/oxide-gemv-q4k/oxide_gemv_q4k.ptx $d/oxide_q4k_gemv.ptx 2>/dev/null
  cp tools/oxide-gemv-q6k/oxide_gemv_q6k.ptx $d/oxide_q6k_gemv.ptx 2>/dev/null
  cp tools/oxide-decode-attn/oxide_decode_attn.ptx $d/oxide_decode_attn.ptx 2>/dev/null
done

# mxf4-gemv (MXFP4 tensor-core GEMM cubin): DIFFERENT recipe from the crates
# above — upstream backend (nested tree), cutlass-mlir device backend, and the
# managed CUTLASS compiler. The cubin is include_bytes!()d by tempura-kernels,
# so a normal cargo build picks it up after this step.
#
# WARNING (2026-10-28 stale-cubin incident): any change to
# tools/mxf4-gemv/src/main.rs MUST be followed by this rebuild AND a commit
# of the regenerated mxf4_gemv.cubin in the SAME commit. The standalone
# `mxf4-gemv` binary compiles from source and will pass while the engine
# runs the checked-in binary -- a stale cubin shipped the pre-fix
# accumulator store mapping for a day and produced allocation-history-
# dependent garbage prefill outputs. Regression test: bench-decode-ops
# "mxfp4 quant+gemm composition" (bit-exact, real quant scales, padded m).
cd tools/mxf4-gemv
export CUDA_OXIDE_BACKEND=~/Code/github.com/NVLabs/cuda-oxide/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
export CUDA_OXIDE_CUTLASS_COMPILER=~/.cargo/cuda-oxide/toolchains/cutlass/4.7.0/x86_64-cu13/lib/libCutlassCompiler.so
export LD_LIBRARY_PATH=~/.rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu/lib:$LD_LIBRARY_PATH
RUSTC_BOOTSTRAP=1 RUSTUP_TOOLCHAIN=1.99.1 CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide build --arch sm_120a
# The standalone binary validates the kernel against a random-data CPU
# reference (uniform-data tests cannot catch fragment-mapping bugs) across
# all shapes incl. the engine's (128, 4096, 4096):
LD_LIBRARY_PATH=~/.rustup/toolchains/nightly-2026-10-11-x86_64-unknown-linux-gnu/lib:$LD_LIBRARY_PATH \
  ./target/release/mxf4-gemv

# From the laptop, scripts/gpu-remote.sh runs all of this on the GPU host:
#   scripts/gpu-remote.sh oxide
#   scripts/gpu-remote.sh bench real-model Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Benchmark (oxide-gemv feature already in tools/bench-gpu/Cargo.toml):
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release \
  --bin bench-real-model -- /path/to/model.gguf
```

Current results (RTX 5090, see `bench-gpu-rtx5090.json`):
- Llama-3.1-8B Q4K single-sequence: **171 tok/s decode**, ITL 5.83 ms
      (1.22x gap vs llama.cpp 209 tok/s; outputs verified NaN-free +
      deterministic, 2026-10-27 post-fix. A 2026-10-28 retraction (outputs
      numerically wrong vs CPU/llama.cpp) was resolved the same day: the
      sm_120 PTX shuffle-encoding bug -- see open work's fixed item.
      bench-quality now shows 100% argmax agreement vs the CPU reference,
      64/64 identical tokens free-running. 2026-10-28 fused-silu raw-format
      port: 134 -> 140 tok/s; lm_head GEMV BK=128 -> 512 (709 -> 1475 GB/s):
      140 -> 171 tok/s; 2026-10-29 the register-exchange rewrite of the
      sm120 q6k kernel fixed the kb=16 residual and made it default-ON
      (169 tok/s); 2026-10-29 the lm_head upload switched from dequantized
      f32 (2.1GB read per token) to the GGUF's native Q6_K via the kpar
      GEMV: 204 tok/s, ITL 4.89 ms, bench-quality teacher-forced PASS 90.6%
      unchanged, +1.3GB VRAM; 2026-10-29 the dual-block direct-load q6k
      variant (no cp.async/barriers, 2x MLP, L2 prefetch) closed the q6k
      down_proj bandwidth gap: 196 -> 204 tok/s, bitwise-identical logits;
      the Q6K v_proj moved to the kpar path (206) and cuGraphUpload after
      instantiation removed the 122-139us serialized launch cost:
      210 tok/s; 2026-10-29 the fused RMSNorm (3 kernels -> 1, 195 -> 65
      graph nodes): **217 tok/s -- BEATING llama.cpp (ref 209)**, ITL
      4.61 ms, teacher-forced PASS 89.1%)
- Llama-3.1-8B Q4K prefill: **7700 tok/s pp128**, TTFT P50 16.5 ms
      (BEATING llama.cpp 7335 by 1.05x; logits det=OK, NaN-free; A/B numerics vs
      exact-dequant max |logit| rel 0.1331 -- the MXFP4 band including the
      Q6K->MXFP4 down_proj double-quantization (lossier than the Q4K path's
      0.1015; bench-quality teacher-forced PASS 89.1%, all mismatches
      near-ties). 2026-10-28 stack: on-device attention (408 -> 815) +
      scratch RoPE with a cos/sin cache (815 -> 1305) + pooled prefill
      norms/adds/SiLU (1305 -> 1365) + Q6K down_proj via MXFP4
      (1365 -> 5126, nsys-driven) + flash BR=16 tile tuning (5126 -> 5485,
      bitwise-neutral) + lm_head BK=512 + last-row d2d (5485 -> 5816).
      Kernel-level verification bitwise-exact
      vs the CPU causal reference for the attention/RoPE/norm paths, see
      open work's prefill item. Decode at 171 tok/s)
- Llama-3.1-8B Q4K batch decode (2026-10-27, post-store-mapping-fix, NO
      bisect switches; verified = NaN scan every step + final-step
      fingerprints + fresh-cache replay bitwise identical; every M>=2
      fingerprint re-baselined this day — earlier MXFP4 batch outputs were
      computed with the broken accumulator store mapping and retracted):
      **M=1 198, M=2 244, M=4 407, M=5 466, M=8 606, M=16 796 tok/s**
      (re-baselined 2026-10-29 after the lm_head upload switched from
      dequantized f32 to the GGUF's native Q6_K + an MXFP4 tiled copy for
      the M-row batch GEMM -- the f32 lm_head re-read 2.1GB per batch step;
      was M=1 174, M=2 136, M=4 243, M=5 290, M=8 407, M=16 602 on
      2026-10-28; all M re-verified NaN-free + fresh-cache replay bitwise).
      (re-baselined 2026-10-28 after down_proj moved to MXFP4 -- was
      M=1 135, M=2 133, M=4 224, M=5 253, M=8 332, M=16 444 on Q6K batch
      GEMV; M=1 rose again 141 -> 174 with the lm_head BK=512 fix; all M
      re-verified NaN-free + fresh-cache replay bitwise).
      M>=2 uses the generalized M-row q6k batch kernel + MXFP4 tensor-core
      Q/K/O/gate/up (m_valid-masked activation quant); M=1 routes to the
      single-seq CUDA graph path.
- MXFP4 M=64: 55.6 TFLOPS; register-blocked rb4 variant (m16n32/warp,
      large-n grids only): 141.6 TFLOPS at (128, 14336, 4096); rb4m32
      (m32n32/warp, B re-reads halved): 259.9 TFLOPS isolated /
      ~67us in-pipeline on the same shape; rb2m32 (m32n16/warp, finer
      grid for small-n/long-k): 154.1 TFLOPS at (128, 4096, 14336)

If you see `Symbol cuda_device__mma_sync__... not found`, the backend is stale.
Rebuild it from upstream and set `CUDA_OXIDE_BACKEND`.

FIXED (2026-11-17): the hand-written q4k_mma_batch / q4k_streamk_batch PTX
kernels were numerically WRONG from their introduction until 2026-11-16
(misparse the Q4K sub-block nibble/scale layout AND used an invented
mma.sync fragment mapping -- 12-31% rel error vs the per-row kpar8 GEMV
on random data; found by bench-probe-q4k-batch). Both are now FIXED and
probe-gated at <= 1e-5 max rel error (measured 2.8e-9..7.5e-8, pure f32
sum-order noise): true m16n8k32 fragment layout (tid=lane&3, gid=lane>>2;
D = the mxf4 acc_coords mapping), byte group mw>>1 with hi-select on odd
mw, raw nibble / raw 6-bit scales, and the mma kernel uses the streamk
8-warp geometry. They are NOT bit-identical to the per-row kpar8 GEMV:
dispatch them only where the batch composition is fixed per capture (the
equal-position graph paths) or uniformly across ALL m (the
TEMPURA_Q4K_BATCH_KERNEL selector routes every m through them; verified
composition-invariant + ppl-clean, but costs 47-88% at m=1 -- see the
2026-11-17 status entry; default stays kpar8). The correct bit-identical
general-M Q4K batch kernel for the no-graph path remains
`q4k_gemv_sm120_kpar8_batch` in kernels/src/q4k_gemv_v4_ptx.rs.
Do not re-enable any new batch kernel without running that probe first.
(The OLD `q4k_batch_sm120` two-phase kernel remains DISABLED: its PTX is
M=2-hardcoded and was never rewritten -- has_sm120_batch() stays false.)

NEW (2026-11-17d): two LARGE-m tensor-core batch kernels for the
TEMPURA_PREFILL_PRECISION=exact prefill route -- `q4k_mma128_batch_sm120`
(kernels/src/q4k_mma128_batch_sm120_ptx.rs; m <= 128 = 8 statically-unrolled
m16 tiles, register accumulators, weights read ONCE per launch; per-row
numerics BITWISE-equal to q4k_mma_batch_sm120 by construction -- same
fragment mapping, warp stripe, and f32 accumulation order; probe arm in
bench-probe-q4k-batch checks both) and `q6k_mma128_batch_sm120`
(kernels/src/q6k_mma128_batch_sm120_ptx.rs; same skeleton, Q6K parsing with
a 2-mma k16 split for the per-16-element scales -- each mma zeroes one B
k16 half so each D is one sub-block's exact s32 sum, scaled by its own
(d*sc) in f32; Q8_1 ACTIVATIONS = llama.cpp's Q6_K x Q8_K class, NOT the
f32-activation class of the q6k sm120 batch GEMV -- never mix them on a
path that compares against f32-act baselines). Debug lessons baked into
the q6k probe: (a) a naive rel-to-result gate against the CPU f64
reference FALSE-POSITIVES on cancellative sums (f32 rounding is ~1e-7 x
sum|terms| but can exceed 1e-5 rel-to-result) -- the gate is
|got - ref| <= 1e-5 * sum|terms| + 1e-6; (b) the reference's h/qq/l decode
must derive from the WITHIN-BLOCK element (elem & 255), not the global
element index.

NEW (2026-11-19): `q4k_mmq_batch_sm120` (kernels/src/
q4k_mmq_batch_sm120_ptx.rs, PTX generated by ptx()) -- the llama.cpp-mmq-
structured exact-prefill kernel and the DEFAULT for the exact route's
m>16 projections: one CTA covers 128 neurons x 32 tokens, operands staged
through ONE 29696-byte dynamic-smem buffer via cp.async (3 CTAs/SM
co-reside and cover each other's staging latency -- the double-buffered
variant measured SLOWER at 1 CTA/SM, and J=64 slower still from the grid
tail; see the 2026-11-19 status entry), weights staged as RAW nibble
words with the planes unpacked at A-fragment load, and the weight scale
products (d*sc)/(dmin*mv) computed as f16 (mul.f16) at staging --
llama.cpp mmq's numerics class. NOT bit-identical to mma128/kpar8 (the
f16 roundings): bench-probe-q4k-batch's mmq arm gates it vs the per-row
kpar8 reference at the class band (measured 3-8e-5, gate 2e-3) AND vs a
CPU mirror of the exact op sequence (ulp-level, gate 1e-5; the mirror
must use the oxide quantizer's TRUNCATING f32_to_f16_bits for d8 and RNE
for the mul.f16 scale products -- tempura's shared f32_to_f16 truncates
and cannot model mul.f16). End-to-end gates: MMLU 96.8% agreement,
ppl 8.0863. Fallback/bisect: TEMPURA_Q4K_PREFILL_KERNEL=mma128.

NEW (2026-11-18): `q4k_mma128w_batch_sm120` (kernels/src/q4k_mma128w_batch_sm120_ptx.rs,
PTX generated by `ptx()` -- OnceLock, not a const str) -- the WIDE-neuron
mma128 variant: 64 neurons/block (8 warps x 8 neurons, WARP-OWNED -- no mw
stripe, warp-private header smem synced with a CTA bar per k-block
(bar.warp.sync was tried first but racecheck cannot model warp-scope
barriers and flags the header handoff as hazards -- the CTA bar is
racecheck-visible and measured cost-neutral),
direct-store epilogue, no cross-warp reduction; grid.y splits the row range
via the row_span param so small-N shapes fill the SMs). Numerics: same
per-element fp op sequence as mma128 but SEQUENTIAL (kb, mw) accumulation --
NOT bitwise vs mma16/mma128; probe-gated <= 1e-5 vs kpar8 (measured
1e-8..4e-7; probe arm `mma128w`, runs when n % 64 == 0). MEASURED VERDICT
(2026-11-18): a WASH vs mma128 at every prefill shape (best grid_y swept
1/2/4/8; host heuristic targets ~256 blocks) -- the wide layout removes the
activation L2 re-reading (8x) and the per-kb CTA bars, and neither was the
bottleneck; the kernel is bound by the per-mma f32 scale chain + A-fragment
global loads, which are layout-independent. Opt-in via
TEMPURA_Q4K_PREFILL_KERNEL=wide; default stays mma128. Also 2026-11-18:
quantize_q8_1 (tools/oxide-gemv-q4k/src/main.rs, the oxide PTX) now stashes
the block's exact integer
q8 rowsum in the HIGH 16 bits of the ds word (llama.cpp's Q8_1 ds layout);
all other consumers read only the low f16 via b16 loads and are unaffected;
q4k_mma128 reads it for the min term (bit-identical numerics, ~40 fewer
ops/tile; the 2026-11-17d follow-up (a)).

WARNING (2026-11-16, Q6K): the two-phase `q6k_gemv_sm120_batch` (the
equal-position graph-path kernel) is NOT bit-identical to the per-row
kpar8/kpar8_db GEMVs (max rel err 6e-5..2.4e-4 on random data -- its
cross-warp reduction is a 3-step lane butterfly, not the per-row
kernels' sequential thread-0 sum; see bench-probe-q6k-batch, which
prints the comparison). It stays on the graph paths where the captured
composition is fixed; NEVER route a ragged/no-graph path through it.
The correct general-M Q6K batch kernel for the no-graph path is
`q6k_gemv_sm120_kpar8_db_batch` in kernels/src/q6k_gemv_v4_ptx.rs (the
kpar8_db warp structure with an m-loop; bit-identical per row to the
per-row db GEMV, probe-gated).

FIXED (2026-11-17): `q4k_gemv_sm120_dp4a_kpar8_dual_block` (kernels/src/
q4k_gemv_v4_ptx.rs) read past the Q8_1 activation scratch on ragged
k-block tails (e.g. down_proj kb=56) since its introduction: the B-side
clamp selected A's FULL q8_blk_lo (already containing the mw*2 offset)
and added mw*2 again, indexing blkA*8+mw*4 -- up to 6 slots past
act_ds/act_qs at the last block. The loaded values were discarded by
the predicated accumulate (all numerics gates passed while the reads
were live); under compute-sanitizer the read faulted (error 719). The
q6k db kernels are immune (they branch-skip block B instead of
clamping). REGRESSION GATE: `bash scripts/sanitizer-gate.sh` (this
host, repo root) -- memcheck over both probes + bench-decode-ops,
racecheck/initcheck on the probes. Its one allowlist: racecheck's
hazards on the q6k kpar8_db/db_batch release/acquire smem handoff
(red.release.cta counter + ld.acquire.cta poll -- racecheck cannot
model scoped-atomics ordering; the pattern is the 2026-10-29 FIX for
the real cross-warp race and is determinism-gated). If you edit those
kernels, re-audit the hazard offsets against the new SASS rather than
extending the list. Any new PTX kernel or oxide host scratch-reuse
path should land with this gate green -- and check scratch reuse sites
against the ACTUAL n/k of every caller, not just the engine's
(oxide_q4k_gemv_host.rs now enforces out-scratch >= n at every reuse).
