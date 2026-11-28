# tempura — current status & open work

Verified gaps and open items. Do not assume these are done.
Archive completed items to docs/history/.

## Latest session status

**STATUS 2026-11-27 (EAGLE-3 verify-cost round — the wall DECOMPOSED and
it is NOT what the recorded follow-ups assumed: the verify already
replays a CUDA graph, the "~13+2m ms no-graph batch-path fixed cost"
theory is RETIRED, one real ENGINE BUG found and fixed (a plain m=1
forward ran after EVERY spec cycle — 40% of "spec" tokens were plain
steps), the true wall is the kpar8-class batch GEMV's m-scaling
(per-CTA compute-bound, ~2.9x at m=4, structural under the verify's
byte-identity contract), and the honest verdict is 0.80x solo at K=4 /
0.88x at K=3 — all gates green, the profile table + the negative
results recorded): (1) THE METHOD (differential nsys, the round's core
deliverable).  The eagle gate's solo 400-token leg profiled ON (eagle)
vs OFF (plain) with --cuda-graph-trace=node; kernel windows segmented
by the verify's q4k_kpar8_batch signature.  THE FIRST FINDING killed
the plan's premise: the serve log shows `[m2p_graph] captured m=4
dtype=1` — the verify ALREADY routes through the (m,dtype)-keyed m>=2
position-synced ragged CUDA graph (option (a) of the design space was
already built; a dedicated verify graph was moot).  (2) THE PROFILE
TABLE (fishtank, RTX 5090, Llama-3.1-8B Q4_K_M, ~290-token avg ctx,
m=4 verify vs the plain m=1 graph step for the same weights):  Q4K
GEMVs 8.66 ms vs 2.97 (gate/up 85 vs 28.1 us per launch, q/o 35.2 vs
12.3, k/v 9.6 vs 3.9 — 2.5-3.0x);  Q6K GEMVs 2.61 vs 1.21 (down 98.8
vs 45.7, lm_head 802 vs 388 — 2.1-2.2x);  attention 3.41 vs 3.44 ms
(28-188 us per launch, ctx-proportional — the oxide batch attention
amortizes m rows PERFECTLY, the one component that does);  small
kernels ~1.0 vs ~0.9;  GPU total ~14.8 vs ~7.9 ms;  host-side ~5 vs
~4.7.  CONCLUSION: no launch-overhead component at all — the delta is
ENTIRELY the GEMV m-scaling.  (3) THE ENGINE BUG the profile flushed
out: one FULL plain m=1 no-graph target forward (per-row GEMVs + 512KB
logits download + argmax) per verify cycle, interleaved
[draft][verify][plain-step] every tick.  Root cause: `eagle_step_gpu`
returned `false` both for "could not run" AND "ran but the request
continues", and the caller's `else if plain_step_gpu(...)` treated
every successful cycle as a failure — taking a plain step (which also
emitted a token + resynced the head, keeping the cycle arithmetic
alive).  ~40% of the run's tokens came from plain steps.  FIXED:
`eagle_step_gpu -> Option<bool>` (Some(done) = ran; None = could not
run -> plain fallback).  Outputs byte-identical (the gate proves it —
the plain step's token equals the next verify's row-0 argmax by
construction).  NOTE the honest throughput accounting: the bug was
accidentally HELPFUL on the wall-clock metric (the hybrid tick was
cheaper per token than a pure spec cycle), so the fix moves the solo
number 66.0 -> 59.7 tok/s at K=4 while making every measurement honest
— pure-spec cycles are what EAGLE actually delivers.  (4) THE GEMM WALL
ATTACKED and characterized: a ROWS-PARALLEL variant of the batch kpar8
GEMV (`q4k_gemv_sm120_kpar8_batch_rp`, kernels/src/q4k_gemv_v4_ptx.rs —
grid (m, n), each CTA runs the exact single-row kpar8 warp structure
for its row, register accumulator, 128 B smem; bit-identical per row BY
CONSTRUCTION, probe-gated bit-exact at every shape x m in
bench-probe-q4k-batch arm 1b).  MEASURED WASH at m=4: gate/up 94.7 us
vs the m-loop's 85 (the m sibling CTAs' weight re-reads do NOT
L2-share — Blackwell CTA scheduling/L2 partitioning defeats the
adjacency assumption), k/v 7.7 vs 9.6, q/o wash.  The LESSON: the
m-loop kernel is per-thread COMPUTE-bound (the kpar8 GEMV at m=1 is
already at ~1.04 TB/s effective with compute hidden under bandwidth;
m rows multiply the per-thread dependent chain ~linearly in ANY
per-row-bit-identical organization) — rows-across-CTAs just multiplies
the HBM traffic instead.  The only structural escape is the
tensor-core class (mma128/mmq), which is BARRED for the verify: its
rows' argmaxes are the emitted tokens and must stay byte-identical to
plain decode's per-row kpar8 numerics (the eagle-equivalence gate).
The rp kernel stays in-tree opt-in via TEMPURA_Q4K_BATCH_KERNEL=rp
(the ffma/h2-epilogue-wash precedent).  (5) THE NUMBERS (final tree,
eagle-equivalence gate legs): OFF 74.4 tok/s solo; ON 59.7 at K=4
(mean_emitted 1.75, hist [110,50,17,22]/199, draft 1.6 ms/cyc, verify
17.8, post 0.37 — cycle 20.1 ms engine) = 0.80x;  K sweep (byte-
identical at every K): K=3 65.6 tok/s (best: mean 1.64, verify 14.6,
cycle 16.2 ms = 9.9 ms/tok engine vs plain's 9.05) = 0.88x, K=6 49.4.
Steady-state per-token: plain 9.05 ms (attn 3.44 + GEMV 4.2 + small
1.4) vs spec 11.5 at K=4 / 9.9 at K=3.  The 400-token leg also carries
a ~1.7 s FIRST-PREFILL warmup spike (cuTile JIT, COMMON to OFF and ON;
eagle adds ~0.26 s: the m2p captures) — 30% of the leg, so the
leg-metric ratio understates steady-state slightly for both sides.
(6) THE VERDICT: the 1.2x bar is NOT met and the wall is now precisely
attributed — the verify pays the m-fold GEMV cost that plain decode
spreads over m tokens, and at acceptance 1.64-1.75 (55% first-token
rejection — the linear-chain head, unchanged from 2026-11-26b) the
arithmetic cannot close: break-even needs mean_emitted >= ~2.2 at K=4
GEMM costs, or the GEMM m-scaling broken without breaking
bit-identity.  The recorded levers stand: DRAFT TREES (the EAGLE-3
reference's acceptance lever — a 60-node tree verify at large m would
ALSO change the GEMM calculus by amortizing into the compute-efficient
large-m regime, but requires the same bit-identity answer per row), a
higher-acceptance head conditioning, or relaxing the verify's numerics
contract (a deliberate, gated decision — e.g. accepting the mma128
class for BOTH the verify and a matched plain path — which trades
byte-identity for speed and was not taken this round).  (7) WHAT
LANDED: the Option<bool> cycle-contract fix (server/src/
engine_task.rs), the rp kernel + host loader + probe arm (opt-in), the
retired fixed-cost theory, this profile table.  (8) GATES ALL GREEN on
the final tree: eagle-equivalence PASS (OFF vs ON solo byte-identical
1686/1686); spec-equivalence PASS (solo + c1..c6 byte-identical, multi
OFF 475.4 tok/s); bench-probe-q4k-batch PASS (all arms incl. the NEW
1b rp arm, bit-identical at every shape x m); sanitizer-gate PASS
(memcheck 3/3 + racecheck + initcheck over the probes incl. the rp
kernel, 0 errors); local cargo test --workspace (exclude bench-gpu,
CPU laptop) 936 passed / 0 failed.  NOT rerun (no default-path
numerics change: the rp kernel is opt-in, the m-loop dispatch is
untouched, the eagle fix changes only the eagle path): MMLU, pooled
ppl, ab-repro, kv-swap A/B, smoke, restart-var — the 2026-11-22/26b
batteries stand.**

**STATUS 2026-11-26b (EAGLE-3 speculative decoding on the SOLO path — the
recorded lever from the speculation arc's honest negative: stage-0
availability GO, the head implemented end-to-end and GATE-GREEN
(byte-identical solo outputs), drafting IS nearly-free as hypothesized
(1.5 ms/cycle), but the solo throughput verdict is an HONEST NEGATIVE at
0.87x — the verify's batch-path fixed cost (~13+2m ms, not the m=1
graph's 5.2 ms) plus a linear-chain acceptance of 1.88/token (vs the 1B
draft's 2.79 on the same gate) leave the cycle above the plain step; the
correctness machinery + two gates ship): (1) STAGE 0 (availability): a
pretrained EAGLE-3 head for Llama-3.1-8B-Instruct EXISTS and is open —
RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3 (speculators format:
config.json + 1.9 GB model.safetensors BF16, Apache-2.0-licensed base);
architecture fully documented in-repo (eagle3.py) and cross-checked
against the official SafeAILab code (modeling_llama_kv.py): fc(3H->H)
over the CONCAT of the target's hidden states entering layers {2, 16,
29}, ONE decoder layer with 2x-width q/k/v on [embed, fused], own
embed_tokens (128256x4096), lm_head over a 32000 draft vocab with
d2t/t2d mapping tensors, norm_before_residual=true, rope_theta 500000.
The official yuhuili/EAGLE3-LLaMA3.1-Instruct-8B also exists but ships
only a pickle .bin (the RedHat safetensors one is loadable directly).
Checkpoint pulled to the fishtank model cache
(~/.cache/tempura/models/eagle3-llama3.1-8b).  (2) THE IMPLEMENTATION
(all landed, cutile-gated): engine/src/eagle.rs (config parse, raw
safetensors reader that keeps the embed table bf16 on the host and the
I64/BOOL vocab maps native, and a host f32->Q6_K quantizer that is the
byte-exact inverse of the kernels crate's q6k_dot_block layout —
roundtrip-test-gated); the head rides the TARGET's CutileGpuModel kernel
machinery under an eagle.* weight namespace (Q6K matrices through the
oxide kpar GEMV + batch GEMM dispatch, f32 norms, shared cutile
rmsnorm/rope/argmax), so zero new GPU kernels were needed; hidden-state
TAPS (3 device buffers) written at the top of the layer loop in FIVE
forward paths (prefill layer loop, m=1 decode inner — captured in the
m1 CUDA graph, generic batch tail, ragged graph body, equal-position
batch graph body) — a no-op when no head is loaded (the DEFAULT path is
unchanged, single Option check); the solo cycle in server
(engine_task.rs): head prefill per target-prefill chunk (teacher-forced
batched rows), K draft steps (the head's own pre-norm output chains
forward — NO fc on drafted rows, per the official
shape[-1]!=embed_dim bypass), ONE verify through the existing v1
forward_spec_verify_greedy/spec_row_views machinery, longest-prefix
acceptance, post-verify teacher resync over the verify's tap rows (the
feature re-anchoring), plain-step resyncs keep head state alive through
multi-client phases; TEMPURA_NO_EAGLE bisects off, TEMPURA_SPEC_K,
TEMPURA_EAGLE_TAP_LAYERS, TEMPURA_EAGLE_POS_SHIFT, TEMPURA_EAGLE_F32
(quantization bisect) are the knobs.  Flag: --eagle-head <dir>.  (3)
THE BRING-UP BUGS (all found by the teacher-forced accuracy probe —
row r's logits vs prompt[r+1] — which is the debug tool that mattered):
(a) the m=1 kpar GEMV returns the SHARED kpar output scratch — running
q/k/v (and gate/up) before copying results out handed every consumer
the LAST GEMV's values (q==k==v, attention garbage; fixed with
interleaved copy-out staging); (b) append_device_gpu_only does not
advance the host seq_len — batching the advance left every head row
attending a STALE kv_len (0 during the whole prefill: the head never
saw its own KV); (c) rope_rows_graph_ready gathers the cos/sin row for
the position in d_positions (NOT its hint argument) — the head now
uploads its own position first; (d) the initial alignment (token
shifted by one, fc-on-carry) was wrong — the correct EAGLE alignment is
head row p = (token_p, h_p) predicting token_{p+1}, anchor-conditioned
drafting, post-verify re-anchoring; (e) the fc input needs pow2 K
padding (12288->16384) for the Q6K GEMV reshape.  (4) THE NUMBERS
(fishtank, RTX 5090, Llama-3.1-8B Q4_K_M, the eagle-equivalence gate's
solo 400-token leg): OFF 75.8 tok/s client-side vs ON 66.0 (0.87x);
engine telemetry: draft 1.5 ms/cycle (the EAGLE structural goal —
nearly-free drafting — ACHIEVED: ~0.4 ms/head-step at Q6K), verify
19.6 ms for 5 rows, mean_emitted 1.88 (hist [51,24,9,15]/99, K=4).
Acceptance sweep: K=3 1.71 (69.1 tok/s — best wall), K=6 1.81 (worse —
verify rows dominate), f32 head 1.81 acceptance == Q6K 1.88
(quantization is NOT the limiter), tap layers {3,17,30} 1.83 /
{1,15,28} 1.69 / reversed order 1.69 (the {2,16,29} entering-layer
convention is best), pos_shift 0/1 identical.  (5) THE ATTRIBUTION
(where the cycle goes, and why it loses): the verify's ~13+2m ms fixed
+ per-row cost matches v1's solo measurements exactly — the m>=2
ragged/generic batch path (launch-bound at small m, per-row attention
at ~450-token contexts) is ~4x the m=1 CUDA graph's 5.2 ms plain step,
so a cycle emitting 1.88 tokens at 21.1 ms loses to plain's ~13.2
ms/token client-side.  The acceptance gap vs the 1B draft (1.88 vs
2.79 on the same prompt) is unexplained after exhausting the
alignment/quantization/tap-layer matrix — the remaining suspects are
the linear-chain (no draft TREE — EAGLE-3's published 4-5 accepted
lengths use 60-node trees) and a residual conditioning difference;
follow-ups recorded: draft trees, a dedicated small-m verify CUDA graph
(the m1-graph analog for the verify shape, collapsing the 13 ms fixed
cost), re-testing the V1 token-shift alignment now that the head
forward is fixed.  (6) GATES: scripts/eagle-equivalence.sh (NEW, the
permanent EAGLE gate): solo OFF vs ON byte-identical 1686/1686 bytes
PASS; scripts/spec-equivalence.sh (the v1/v2 standing gate on the final
tree): solo + c1..c6 ALL byte-identical PASS (multi OFF 488.7 tok/s);
local cargo test --workspace (bench-gpu excluded, CPU laptop) 936
passed / 0 failed (931 + the eagle q6k-roundtrip + tap-layers tests).
NOT rerun (no kernel/PTX change; the default path is untouched — the
taps are dead code without --eagle-head): MMLU, pooled ppl, probes,
sanitizer-gate, ab-repro.  (7) VRAM: the head is ~332 MB Q6K on device
+ a 33 MB 1-layer KV pool + 1.5 MB taps — trivial against the 8B's
budget.**

**STATUS 2026-11-26 (TurboQuant KV-density round — stage-1 go/no-go:
NEGATIVE RESULT, simulation-first per the FP8-round precedent; the
CpuPolarQuant class does NOT pass the KV quality gates at any useful
density, so e4m3 REMAINS THE KV DENSITY FLOOR and no GPU pool/kernel
work was done; the simulation harness landed as the durable output,
all gates green): (1) THE HYPOTHESIS UNDER TEST (the Bonsai density
thesis — KV cache is the memory bottleneck on edge): group-wise
min/max polar quant (turboquant.rs CpuPolarQuant: per-group min/max
f32 scales + b-bit uniform codes) at 2-4 bits/elem + group scales
lands between e4m3 (1 B/elem, ppl ratio 0.995 PASS) and nothing —
the ladder rung below e4m3. (2) THE METHOD — quantify BEFORE kernel
work: kernels/src/kv_quant_sim.rs injects the EXACT CpuPolarQuant
round-trip at the KV storage boundary (quantize-on-write,
dequantize-on-read — the exact serving semantics of a polar-quant
pool) inside the real CPU quantized forward, default-off via a single
atomic load. Unit-gated: bit-IDENTICAL to the real compressor's
compress+decompress (all sweep configs), and the round-trip is
IDEMPOTENT (quantize->dequantize->quantize is a fixed point — the
swap/mirror requantization requirement). tools/fp8-sim gained
TEMPURA_KV_SIM_SWEEP mode: reference pass (sim off) + per-config
(bits x group) passes, reporting MMLU accuracy / choice agreement /
logit band / WikiText-2 ppl ratio + the pool density (B/elem,
KiB/token for the 32-layer x 2 x 1024 GQA pool). (3) THE LADDER
(fishtank CPU, Llama-3.1-8B-Instruct Q4_K_M, 120q MMLU + 8x512-ctx
ppl legs, gates: agreement >= 95% and/or ppl ratio <= 1.01):
b2g32 32 KiB/tok agr 28.3% ppl 3.53 | b2g64 24 KiB/tok 24.2% 10.58 |
b2g128 20 KiB/tok 18.3% 58.12 | b3g32 40 KiB/tok 81.7% 1.060 |
b3g64 32 KiB/tok 74.2% 1.147 | b3g128 28 KiB/tok 70.0% 1.231 |
b4g32 48 KiB/tok 90.8% 1.0194 | b4g64 40 KiB/tok 88.3% 1.0207 |
b4g128 36 KiB/tok 85.0% 1.0357. NO CONFIG PASSES. (4) THE VERDICT
and why it is structural, not tunable: the best config (4:32) fails
BOTH gates (90.8% agreement vs 95%; ppl ratio 1.0194 vs 1.01 — the
same 90.8% agreement the e4m3-ACTIVATION negative measured, a
coincidence of cliffs) AND buys only 1.33x density over e4m3 (48 vs
64 KiB/token); matching e4m3's 1 B/elem would need 4:16, strictly
worse than e4m3 at the same bytes. Quality scales monotonically
with B/elem across the whole ladder and the min/max polar class is
dominated by e4m3's floating exponent at every density — the
per-group min/max fold is outlier-sensitive exactly where K/V
distributions (post-RoPE K, V) have heavy tails. The floor stands:
f16 (128 KiB/tok, default) -> e4m3 (64, opt-in) -> nothing passing
below. A future rung requires a different scheme (outlier-aware or
non-uniform scales — e.g. per-group f16 scale+offset with clipping,
or KV-aware rotation as in KIVI/NoPE-style analyses), not this one.
(5) WHAT LANDED: the harness (kv_quant_sim.rs + the model.rs storage-
boundary hooks on every CPU attention path — decode, prefill paged +
plain, BitNet persistent-thread, multi-token, eval — each a no-op
atomic load when off) + the fp8-sim sweep mode; committed 2a7ef71.
(6) GATES: local cargo test --workspace (bench-gpu excluded, CPU
laptop) ALL PASS incl. the 3 new kv_quant_sim tests (bit-identity,
idempotence, off-identity); the sweep ran single-instance-per-core-
set (3 parallel processes pinned via taskset, ~6 GB quantized
weights each — no f32 weight cache, unlike the e4m3 sim, so the
no-parallel-instances OOM rule does not bind). 250q confirmation of
the best config: agr 89.6%, ppl ratio 1.0194 (ppl 8.2624 vs the
exact-prefill f16-KV 8.1048 baseline; logit band max |d| 2.0361,
rel-to-peak 0.0984) — verdict unchanged, both gates fail. (7) FOLLOW-UPS RECORDED: if KV density below
64 KiB/token ever matters (Bonsai-class edge), the candidate schemes
are outlier-aware scales or per-head format selection (e4m3 for
outlier-heavy heads, polar for flat ones) — both need this same
stage-1 harness first (it is config-agnostic: swap roundtrip_group).**

**STATUS 2026-11-26 (zeolite — the Intel NPU stack spun off from tempura is
COMPLETE through its 7-stage 1:1 translation plan: a DROP-IN
libze_intel_npu.so.1 replacement for Lunar Lake NPU 4000 where the only
trusted binaries are the firmware and the kernel driver, both l0diff
differential gates GREEN, Definition of Done checked — ~33k lines of cited
Rust, 160 tests, 14 commits in the SIBLING repo ../zeolite; recorded here
because the arc started as tempura's NPU investigation): (1) THE ARC. The
NPU stubs (npu-intel/npu-qualcomm features) needed ground truth: recon of
intel/linux-npu-driver (UMD C++) + openvinotoolkit/npu_compiler_elf
(vpux_elf core, hpi_component, loader) — the UMD is the entire user-space
stack below the Level-Zero API. Phase A proved PURE-RUST /dev/accel
submission: hand-built ioctls + firmware commands drove DMA workloads on
the real NPU with no Intel code linked. (2) THE BLOB EMITTER + FIRMWARE
CONTRACTS discovered on hardware: DMA_MEM user buffers (not
DMA_MEM_IMPORT), the WLM structure requirement (descriptor->buffers->tasks
layout), queue-destroy-before-wait ABORTS jobs (the KMD call order is
load-bearing), 1D width/dim_size_1 aliasing in DMA descriptors,
task-storage classes, TILE_CONFIG fuse-mask INVERSION, and the MI note
11.4.10 contract. (3) THE DIFFERENTIAL METHOD: the real Intel driver as
ORACLE — l0diff runs the same workload against both drivers and
byte-compares; gdb byte-dumps of driver-internal buffers anchored the
emitter when no source was trusted. (4) THE 7-STAGE PLAN (TRANSLATION_PLAN.
md, hard rules: NO STAND-INS — if a 1:1 translation exists, use it; LINE
CITATIONS on every function; PIN EVERY LAYOUT (static_assert -> Rust
size/offset test); NO KMD-CALL REORDERING; BYTE-DIFF GATES at every
boundary where C++ produces bytes; GREEN AT EVERY COMMIT; judgment calls
reported, never hidden). Stage results: (1) vpu_driver leaf (ioctls,
device, BO) 1:1; (2) command/cmdbuf/job 1:1; (3) context + command queues
1:1; (4) stand-ins DELETED — vpux_hpi/vpux_loader/elf_parser wired into
the exec path; (5) vpux_elf writer/reader 1:1 (blob flows through
elf::writer); (6) tracing 1:1; (7) full Level-Zero API surface + cdylib
(tools/zeolite-l0). (5) THE FINAL STATE: drop-in driver — l0diff gate 1
(copy workload) VERIFIED field-identical on a hand-emitted WLM blob; gate
2 (golden profiling blob) EXECUTES with profiling; 160 tests green.
Naming: ze = Level Zero prefix; the cuda-oxide parallel holds exactly —
trusted infrastructure below (firmware + KMD, like rustc/CPU), everything
above it translated and gated. NOTHING in tempura depends on zeolite;
the relationship is prospective (a future npu-intel backend for edge
inference — Bonsai-class ternary on Lunar Lake). OPEN ITEMS (in flight,
in ../zeolite): DPU compute tasks (the emitter authors DMA tasks today;
DPU work is in another worktree) and a Kani/Verus pass over the
translation (same verification-first discipline as tempura — index
safety, struct-layout proofs, the packing/bitmap roundtrips).**

**STATUS 2026-11-23 (speculative decoding v2 — the cross-session BATCHED
cycle is LANDED and GATE-GREEN, and the throughput verdict is a
CONFIRMED HONEST NEGATIVE at serving batch scale, with the per-component
attribution that closes the question for this hardware/model pair; plus
TWO engine bugs the batched draft flushed out, both fixed): (1) THE
DESIGN. `spec_batch_step_gpu` (server/src/engine_task.rs) replaces v1's
per-session cycle: (0) resident/greedy/decode-ready filter, (1) lazy
draft-state materialization + prompt prefill per session, (2) draft-run
resync to the target base, (3) K BATCHED draft steps — every live
session's draft model runs its next proposal in ONE forward per step
(m = live sessions; a session that stops drafting is swap-removed to the
tail and keeps its proposals), (4+5) ONE multi-session verify:
`CutileInferenceWrapper::spec_verify_greedy_multi` (trait
`forward_spec_verify_greedy_multi`) builds spec_row_views for EVERY
session — session i contributes j_i rows over its OWN run (row r writes
K/V at base_i+r, attends exactly base_i+r+1 — the v1 causal pattern
generalized to MIXED sessions) — and runs ONE forward_batch_decode_
greedy over the whole set: the batch attention/scatter/rope take
per-row kv_lens, write offsets and run base pointers, so rows of
different sessions compose exactly like rows of one; (6) per-session
accept/rollback/emit (v1 logic verbatim). Fallbacks: <2 proposals ->
one plain m=1 step; target growth failure -> defer to a later tick;
unsupported verify shape -> plain steps. DraftCache.kv became an Option
so the batched draft can take() every live run into a contiguous slice.
(2) THE GATE (unchanged, PASS on the final tree, 2026-11-23):
scripts/spec-equivalence.sh — solo 1686 bytes + c1..c6 1277-1444 bytes,
ALL byte-identical ON vs OFF. Acceptance: mean_emitted 2.80, hist
[117,90,54,225]/486 session-cycles (46% full acceptance). (3) THE
NUMBERS (honest): solo 0.72x (71.7 -> 51.9 tok/s), multi-client 0.35x
(453.1 -> 159.8 tok/s; v1 per-session was 0.27x — batching bought
1.3x, not the win). The v2 bars (solo 1.3x, multi 1.5x) are NOT met.
(4) THE ATTRIBUTION (per-ENGINE-cycle telemetry — fixed this round, see
(6b) — m=6 steady state: draft 38 ms = 4 batched 1B steps x ~9.5 ms,
verify 75.6 ms for 24 rows = ~3.15 ms/row). THE WALL: a verify row
costs the same as a plain decode row (~3.5 ms at m=6 / ~250-token
contexts) and does NOT amortize with batch size — the weight traffic
amortizes (m-loop kernels read weights once per launch) but the per-row
work (attention, activation quant, row compute) is per accepted-token,
and the DRAFT adds K full model passes on top. Plain batched decode at
m=6 ALREADY amortizes the target's weights across sessions (3.15
ms/tok) — the classic speculation win condition (memory-bound solo
decode) simply does not exist at serving batch sizes on a 5090.
Measured-and-reverted along the way: chunking the verify at
max_batch_graph to reach the ragged graphs is WORSE (81.4 vs 75.6 ms —
each chunk re-reads all the weights); the single unchunked call is the
default. Floor math for the record: K=4 draft passes (4 x 2.3 GB for
the 1B incl. its f32 lm_head) + verify weight traffic ~= 9.7 GB/cycle
vs plain's 4.9 GB/tick — a theoretical ceiling of ~1.1x at zero per-row
cost. (5) THE NEXT LEVERS (recorded, in order of leverage): (a) a
CHEAPER DRAFT — the 1B is too big relative to the 8B (2.3 GB effective
per pass: 1.2 GB weights + the 1.05 GB f32 lm_head read per row);
quantizing the draft's f32 lm_head to Q6K at load (0.26 GB, routed
through the pooled Q6K batch kernel) and/or a 0.5B draft caps the
draft phase near ~1 ms/step; (b) EAGLE/Medusa-style draft heads on the
TARGET's hidden state — removes the K-separate-model-passes floor
entirely (the only structural route to a win at batch scale); (c)
bigger K only pays after (a)/(b). Speculation stays OPT-IN
(--draft-model unset = off). (6) THE TWO BUGS THE BATCHED DRAFT FOUND:
(a) f32 weights at m>1 on the batch path took gemm_d2d_slot's
ALLOCATING tiled prefill GEMM — eager-legal, but a DriverError 900
PANIC inside a CUDA-graph capture ("operation not permitted when
stream is capturing", engine thread dead, HTTP 503s). The 1B draft's
f32 lm_head (in its m>=2 ragged-graph capture) was the first model to
hit it. Fixed in gemm_any_d2d_batch: m>1 f32 weights now route through
a POOLED tiled GEMM (forward_bt_d2d_into, weight matrix read ONCE for
all m rows — the per-row fallback re-read it m times, ~5 GB of
avoidable traffic at m=6) with the pooled per-row GEMV loop as
fallback; the a_row pool scratch is now rank-2 ([1, row_stride] —
cutile_gemv_coalesced expects a rank-2 x; the old rank-1 buffer
panicked the GEMV). Numerics note: the tiled arm is a different
sum-order class than the m=1 slot GEMV — acceptable because the only
current m>1 f32 batch consumer is the SPEC DRAFT (draft numerics
influence acceptance length, never the emitted tokens; no target model
has an f32 weight on the batch path). (b) v1's spec telemetry divided
draft/verify micros by the per-SESSION cycle count — the per-engine-
cycle costs were understated ~4x (the "draft 3.6ms, verify 20.7ms"
v1 attribution was per session-note, not per engine cycle); SpecStats
now counts engine_cycles separately. (7) GATES ON THE FINAL TREE:
spec-equivalence PASS (solo + 6 clients byte-identical); kv-swap-ab-
mini PASS (A1==A2==B + eviction); nonstream-route-smoke PASS;
kv-restart-var PASS (no cross-restart variance); sanitizer-gate PASS
(memcheck over bench-probe-q4k-batch / bench-probe-q6k-batch /
bench-decode-ops + racecheck/initcheck on the probes, 0 errors);
local cargo test 931 passed / 0 failed (--exclude tempura-bench-gpu,
CPU-only laptop). NOT rerun (no kernel/PTX change; the 2026-11-22
battery stands): MMLU, pooled ppl, ab-repro.**

**STATUS 2026-11-22b (speculative decoding wired into the serving stack:
the equivalence property is LANDED and GATE-GREEN — greedy spec decoding
is byte-identical to plain greedy decode, solo AND 6-client, made
executable as scripts/spec-equivalence.sh — but the throughput verdict
is an HONEST NEGATIVE at v1 per-session scope: solo 0.75x, multi-client
0.27x, so speculation stays OPT-IN via --draft-model; plus one
PRE-EXISTING f16-pool cast bug found and fixed by the draft model):
(1) THE DESIGN. The verify is the round's soul: the target verifies
[last_emitted, d1..d_{K-1}] in ONE batch-decode-shaped forward over the
session's OWN pooled run via CutileKvCache::spec_row_views — K
lightweight handles on one run, view r reporting seq_len base+r so the
existing batch machinery derives per-row KV write offsets (base+r) and
attention lens (base+r+1): row r writes its K/V at offset base+r and
attends over exactly the first base+r+1 tokens — CAUSAL verification of
a draft block computed with the SAME decode-class kernels the plain
batch path dispatches (ragged m>=2 graph or generic no-graph; the
m-loop batch GEMVs, batched rope, per-row-offset scatter, oxide batch
attention, GPU argmax — every per-row bit-identity the composition-
invariance gates already prove). Views own nothing (pool=None,
run_pages=0 -> Drop no-op); the owner pre-grows and rolls back. The
engine step (server/src/engine_task.rs spec_step_gpu): lazy draft-state
materialization + prompt prefill, K sequential m=1 greedy draft steps
through the draft's own pool (any numerics — the draft only influences
ACCEPTANCE LENGTH, never the output), the one verify, accept the
longest prefix where d_i == row_i's argmax, emit the target's argmax at
the first mismatch (no bonus token on full acceptance — K proposals
from K draft feeds, both KV runs roll back to base+emitted, uniform).
Greedy only (v1): sampled sessions take the plain batch step;
spec-eligible sessions are partitioned out of plain_batch_step_gpu
(extracted unchanged from decode_step_gpu). Draft model:
--draft-model <gguf> (server flag, loaded after the target so VRAM
order is target-weights -> target-pool -> draft-weights -> draft-pool;
vocab-size hard-check — same tokenizer family required).
Llama-3.2-1B-Instruct-Q4_K_M (bartowski) pulled to the fishtank model
cache as the draft for Llama-3.1-8B. Telemetry: TEMPURA_ENGINE_DEBUG_
TICKS prints cycles / mean_emitted / hist / draft-ms + verify-ms per
cycle; TEMPURA_SPEC_K bisects K (default 4, measured best 3).
(2) THE GATE. scripts/spec-equivalence.sh (the permanent regression
gate): run OFF (no draft) vs run ON (--draft-model), 1 solo client +
6 concurrent streaming clients, greedy — PASS = every client
byte-identical. RESULT (2026-11-22, fishtank, final tree): ALL MATCH —
solo 1686 bytes, c1..c6 1277-1444 bytes each, AND a 3-repetition solo
rerun all byte-identical. Acceptance: mean_emitted 2.81-2.91, K=4 hist
[156,133,85,314]/688 cycles (46% full acceptance), K=3 2.43, K=2 1.80 —
a 1B drafting for an 8B accepts as hoped; the EQUIVALENCE is what the
round delivered. (3) THE THROUGHPUT NEGATIVE (recorded, not hidden):
solo OFF 5.4-5.6 s vs ON K=3 7.2-7.5 s per 400 tok (0.75x); multi 6
clients OFF 459.5 vs ON 123.6 tok/s (0.27x — per-session spec cycles
SERIALIZE in the engine loop while the plain batch amortizes one
forward across all sessions). Cycle attribution (the new telemetry):
draft m=1 step ~3.6 ms (the 1B's m=1 graph replay is NOT 8x cheaper —
launch/graph-fixed costs dominate), verify m=4 ~20.7 ms / m=3 ~15.2 /
m=2 ~14.3; K=3 attributes 10.6 ms/tok vs the plain path's ~13.8 — but
wall clock still loses (per-cycle engine overheads + the draft run's
growth/re-capture amortized in). The structural levers for a future
win, in order: (a) BATCH the draft steps across sessions (the same
ragged-batch machinery, m = n_sessions, one launch instead of K
sequential m=1 replays), (b) batch the verify across sessions (m =
n_sessions x K rows over per-session runs — the scatter/attention
already take per-row lens/offsets/ptrs), (c) a cheaper draft decode
path. v1 stays per-session by scope; the numbers say do not flip
anything on. (4) THE PRE-EXISTING BUG THE DRAFT FOUND:
d2d_into_pool's f16/e4m3 cast (engine/src/cutile_kv_cache.rs) pads
num_elems up to CAST_B for the grid and validates the SRC against
n_pad — a src sized exactly num_elems failed whenever num_elems was
not a CAST_B multiple. kv_dim-1024 targets are always multiples (the
target serving path is BIT-UNCHANGED — the pad branch never fires);
the Llama-3.2-1B draft (kv_dim 512, 73-token prompt) is the first
model to hit it: engine-thread panic 'cast_f32_to_f16: src holds 37376
elements, need 37888'. Fixed with CutileKvCache::padded_cast_src — a
cached zero-padded f32 staging D2D before the cast (pad tail is cast
but never copied into the pool); enriched the cast error with
layer/seq_len/kv_dim/src-size context. (5) GATES ON THE FINAL TREE
(spec OFF default — the battery the diff structurally touches):
kv-swap-ab-mini 3-client A/B A1==A2==B byte-identical with eviction
exercised PASS; nonstream-route-smoke PASS; kv-restart-var PASS; local
cargo test 931 passed / 0 failed (--exclude tempura-bench-gpu,
CPU-only laptop). NOT rerun this round (no kernel/PTX change; the
d2d pad branch cannot fire at kv_dim 1024): MMLU, pooled ppl, probes,
ab-repro, sanitizer-gate — the 2026-11-22 battery stands for those.
(6) FOLLOW-UPS RECORDED: multi-session batched draft+verify (the
throughput lever), sampled speculation (stochastic acceptance — v1 is
greedy; the A/B and smoke loads are greedy so the gates cover v1),
e4m3-pool spec run (the gate ran f16; the scatter twins are dtype-
gated so e4m3 should work but is unmeasured).**

**STATUS 2026-11-22 (differential-profiling round: the llama.cpp prefill
gap CLOSED -- exact mode 5618 -> 7962 tok/s pp128 (1.42x, vs llama.cpp's
7335 reference we are now 1.08x FASTER at reference quality, and faster
than our own opt-in MXFP4 path at 7783, retiring the quality/speed fork
for Q4_K_M), the DIFFERENTIAL PROFILE TABLE built (the round's method:
per-kernel nsys on BOTH engines, same GPU + model, delta per kernel --
not lever-guessing), TWO root causes found and fixed (the J=32 weight
restaging + a 6-way smem bank-conflict on the fragment loads), both
fixes PORTED to q6k_mmq, the ffma/h2 epilogue washes re-measured at
J=128 (still washes), all gates green incl. the flip-tree pending-gate
battery): (1) THE METHOD AND THE TABLE. llama.cpp was rebuilt with CUDA
on fishtank (build-cuda, checkout 418f905; llama-bench pp128 = 9752 +/-
525 tok/s -- the NEWER checkout is faster than the recorded 7335
reference; all gap ratios below cite the 7335 record) and profiled with
nsys alongside our bench-real-model exact. Per 128-token step, GPU kernel
time: llama 12.9 ms vs ours 22.9. Per kernel pair (llama / ours-before):
Q4K GEMMs 8.5 / 15.8 (q 29 / 66 us, k 9.8 / 15, v 21 / 15, o 21 / 66,
gate 61 / 142, up 60 / 142 -- llama splits each mul_mat_q into kernel +
stream_k_fixup ~10 us); Q6K down 1.9 / 3.3 (incl. our lm_head-last-row
kpar8_db 0.43); flash attn 0.99 / 1.39; quantize 0.34 / 0.79 (both
quantize per projection, 7/layer); silu 0.24 / 0.16 (we WIN); norms
0.21 / 0.23; add 0.16 / 0.17; rope 0.18 / 0.27. SUM OF DELTAS ~ 9.5 ms
~= the 9.4 ms gap; ~75% of it in the Q4K GEMMs, concentrated in gate/up
(+6.4 ms) and q/o (+2.0). (2) ROOT CAUSE A -- THE J (TOKENS-PER-CTA)
TILE ASPECT. llama's mmq config at pp128 is I=128 neurons x J=128 TOKENS
per CTA: the weight tile is staged into smem ONCE per 128 tokens. Ours
was J=32 -> grid.y=4 CTAs each restage the same weights (4x the cp.async
smem traffic + 4x the per-k-block CTA barriers; the k-split grid.z
multiplies the CTAs further). The 2026-11-19 J=64 failure was grid-tail,
which the k-split now fixes. LANDED: the kernel generator is
J-parameterized (kernels/src/q4k_mmq_batch_sm120_ptx.rs build_entry:
jt unroll J/8, J/2 accumulators, a_qs staging J/4 words/thread, a_ds
staging looped J/32 with a J-dependent group stride, token_base =
blockIdx.y << log2 J, per-J dynamic smem) with new entries
q4k_mmq_batch_sm120_j64 (38912 -> 42496 B after (3)) and _j128 (57344 ->
61952); the host loads them from the SAME module and forward_mmq_* take
a j param. J IS NUMERICS-NEUTRAL (per-element accumulation order is
J-independent; the probe's existing CPU mirrors gate the J arms
UNCHANGED -- measured ulp-level/bitm-identical) and the engine picks
(J, S) by n ONLY (composition invariance; the (J,S) table is calibrated
on a J x S sweep in bench-probe-prefill-tc: n<=1024 -> (64,8), n<=4096
-> (64,5), else (64,5); TEMPURA_Q4K_MMQ_J=32|64|128 bisects,
TEMPURA_Q4K_MMQ_SPLIT=0 still bisects S). MEASURED (isolated per-shape):
gate/up 0.146 -> 0.117 (J=128 S=3), q/o 0.049 -> 0.045 (J=64 S=5); END-
TO-END 5618 -> 6209 tok/s, logits fingerprint UNCHANGED (the S values
per shape did not change in this step). (3) ROOT CAUSE B -- SMEM BANK
CONFLICTS (the one the wash-measurements could not see). ncu on the j128
gate/up kernel: 6-way AVERAGE conflicts on the shared loads, 29.1M
conflicts / 37.2M wavefronts = 78% (est. speedup 67%), stalls 43% on
MIO-scoreboard latency, 96 regs, 0 spills. CAUSE: every smem row stride
was a multiple of 32 words (w_qs 128 B/neuron, a_qs 256 B/token, w_dm
32 B/neuron), so the 8 neurons/tokens a warp's fragment load touches all
map to the SAME bank set (bank = (row*32 + off) % 32 = off). FIX: PADDED
STRIDES -- w_qs 144 B (bank = 4*neuron + j*8 + tid4 = lane + const, a
per-instruction BIJECTION), a_qs 272 B (bank = 4*token + lane offset),
w_dm 36 B; pad words are never written or read. MEASURED (isolated):
gate/up 0.117 -> 0.091, q/o 0.045 -> 0.031 (BEATS llama's 0.030+0.010
fixup), k 0.018 -> 0.014, down-shape 0.155 -> 0.088. END-TO-END 6209 ->
7350 tok/s = PARITY with the 7335 reference; fingerprint shifted
(expected: the gate/up S changed 3 -> 5 with the table update -- the
same validated split sum-order class, probe-gated). (4) THE Q6K PORT:
the same two treatments in q6k_mmq_batch_sm120_ptx.rs (J entries
q6k_mmq_batch_sm120_j64/_j128; padded strides w_raw 192 -> 196 B
(bank = 17*neuron + c, distinct for the 8 fragment neurons), w_dsc
32 -> 36 B, a_qs 272 B; forward_mmq_* take j; engine default J=64,
TEMPURA_Q6K_MMQ_J bisects). Probe arm 4c (j64/j128 x S=1/2 at every
shape x m): BIT-IDENTICAL to the unchanged CPU mirrors (0.000e0).
MEASURED: 7350 -> 7962 tok/s end-to-end (the J=32 bisect with padding
alone: 7513 -- the padding bought +163, J=64 +449); logits fingerprint
UNCHANGED from (3). THE MXFP4 OPT-IN PATH MEASURED UNCHANGED at 7783
tok/s det=OK -- exact is now the fastest AND the reference-quality
route; the fork's remaining value for Q4_K_M is zero (the selector stays
for quality-insensitive bulk prefill / other formats). (5) THE WASHES
RE-MEASURED AT J=128: the 2026-11-21c epilogue restructures (ffma 12-op
chain, h2 f16x2 packed pos) were washes at J=32 with 3 CTAs/SM; at
J=128/1 CTA the issue regime differs, so they were rebuilt as
q4k_mmq_batch_sm120_ffma_j128/_h2_j128 entries and swept: STILL WASHES
(gate/up J=128 S=3: scalar 0.117, ffma 0.118, h2 0.119 ms). The kernel
is bound by smem/load latency (the ncu stall profile), not fp issue --
consistent with (3) being the win. (6) BRING-UP BUGS (both caught by the
gates, both one-liners): the a_ds staging loop's token stride was
i*256 instead of i*32 (shl 8 vs shl 5 -- silent smem OOB writes past the
buffer at J>=64, compute-sanitizer memcheck localized it to
q4k_mmq_batch_sm120_j128+0x3b40 = the ds zfill STS with BUF_A_DS base;
at J=64 it corrupted nothing visible and PASSED the m=1 probe -- the
probe now covers all J arms at every m, and J=128's m=128 arm would
have caught it too); the q6k host loader tuple arity / a stray j param
on forward_mma128 (compile errors). (7) GATES ALL GREEN ON THE FINAL
TREE (this battery doubles as the 2026-11-21d default-flip validation
the previous entry left pending -- it ran on the flipped tree at 7962):
probe-q4k-batch PASS (all mmq arms incl. the NEW 5c J=64/128 x S=1/2
arms at every shape x m, ulp-level vs the unchanged mirrors + class
band vs kpar); probe-q6k-batch PASS (arm 4c bit-identical to the
mirror); bench-real-model exact 7962 tok/s pp128 det=OK (fingerprint
170452491303490), decode 216; mxfp4 opt-in 7783 det=OK; bench-ab-repro
solo/repeat/interleave/composition IDENTICAL; nonstream-route-smoke
PASS; kv-restart-var PASS; 6-client kv-swap A/B PASS on f16 AND e4m3
(byte-identical, eviction exercised); sanitizer-gate PASS (memcheck +
racecheck + initcheck, 0 hazards -- the J variants + padded strides are
inside the probe coverage); MMLU exact GPU 62.4% (CPU 61.6%, llama 62.4%), GPU/CPU choice
agreement 96.8% (gate 95%) PASS; pooled ppl 8 chunks 8.0877
(exact-class baseline 8.0863-8.0909, GPU/CPU ratio 0.9979, 0 NaN)
PASS; local cargo test 931 passed / 0 failed
(929 + the new PTX/geometry tests; --exclude tempura-bench-gpu on the
CPU-only laptop per the documented caveat). (7b) THE FINAL PROFILE (nsys, the after-column of the table): per step
q4k_mmq_j64 9.32 ms (was 15.78; llama 8.5 incl. fixups), q6k_mmq_j64
2.00 + reduce 0.08 (was 3.33; llama 1.9), lm_head-last-row 0.44, flash
attn 1.40 (llama 0.99), quantize 0.83 (llama 0.34), rope 0.26 (0.18),
norms 0.21, add 0.16, silu 0.18 (0.24 -- we win), q4k split-reduce 0.79.
(8) WHAT REMAINS (the honest
differential, recorded for the next round, all sub-1 ms/step): quantize
0.79 vs 0.34 ms (their quantize_mmq_q8_1 is ~2.3x faster per launch --
vectorization/layout), attention 1.39 vs 0.99 (our paged flash attn vs
their flash_attn_ext_f16 + fixup), lm_head-last-row 0.43 (kpar8_db at
714 GB/s), rope 0.27 vs 0.18. vs the CURRENT llama.cpp checkout
(9752 tok/s) we are at 0.82x -- the remaining levers above are worth
~1.5-2 ms/step (~8500-9000 tok/s optimistic). (9) THE DECISION: exact
stays the DEFAULT prefill route at 1.08x the 7335 record; the mmq J=64
variants + padded strides are the default kernels; the MXFP4 selector
stays opt-in.**

**STATUS 2026-11-21e (THE DEFAULT-FLIP DECISION, executed: exact is now the
DEFAULT prefill route at 5584 tok/s pp128 = 1.31x of llama.cpp's 7335 at
reference quality; MXFP4 demoted to opt-in via TEMPURA_PREFILL_PRECISION
=mxfp4; the decision criterion was the fork's remaining value at the final
number, NOT the retired 6000-tok/s proxy bar; the last lever round ran the
q6k k-split (+395 tok/s, LANDED) and micro-experimented the two remaining
epilogue levers (both measured WASHES, skipped with numbers recorded);
GLU-fused writeback skipped on ROI -- see (5)): (1) THE LEVERS THIS ROUND.
(i) q6k mmq grid.z k-split (commit 054a304): the q4k pattern ported to
q6k_mmq_batch_sm120 -- ctaid.z k-block ranges, partials[z*m*n], fixed-order
q6k_mmq_split_reduce, splits==1 = the unchanged unsplit path; host
forward_mmq_split_batch_d2d_rows (NOTE: the kernel entry now takes 9
params, the unsplit host launch passes splits=1/partials=0 -- an early
7-param build faulted every mmq launch with error 700); engine heuristic
n-ONLY (slots<256 -> S=round(640/slots) clamp [1,8] -- down_proj n=4096
-> S=5; slots>=256 = lm_head stays UNSPLIT: at n=128256/k_blocks=16 the
S*128*128256*4 B partials traffic dominates any fill gain);
TEMPURA_Q6K_MMQ_SPLIT=0 bisects. Probe arm 4b (S=2/S=4 vs the split CPU
mirror) measured BIT-IDENTICAL + the 2e-3*abssum class band vs mma128.
MEASURED: exact 5189 -> 5584 tok/s pp128 (+7.6% -- the down_proj split
bought ~1.7 ms/step, more than the isolated 0.05 ms sweep delta, because
the split also relieves the q4k gate/up CTAs' co-residency competition),
det=OK, decode 218 unchanged, llama.cpp gap 1.4x -> 1.31x, logits
fingerprint re-baselined to 170432912025587 (the q6k sum-order class
change, expected). (ii) FMA-FORM epilogue (acc += d8*fma(D, ax, ay*rs),
12 fp ops per (g,jt) vs 18) and (iii) F16X2 packed pos products (two
mul.f16x2 + cvt unpacks; mul.f16x2 REQUIRES .b32 operand regs -- ptxas
"arguments mismatch" on .u32): both WASHES at every prefill shape
(projections-only tok/s: scalar 8104, ffma 8079 (-0.3%), h2 8028 (-0.9%))
-- the f16 pipe does not dual-issue under the cvt wall on sm_120 AND the
kernel is not fp-issue-bound at all (a 33% fp-op cut moves nothing; the
binding constraint is staging/latency/co-residency). Variants stay
in-tree opt-in via TEMPURA_Q4K_MMQ_EPILOGUE=ffma|h2 (measurement-only,
different numerics class, NOT mirror-gated); default stays scalar.
(2) THE FLIP RATIONALE (the deliverable). The 6000 bar was a proxy; the
real question: is the quality/speed fork still worth keeping at the final
number? At 5584: exact is within 1.31x of llama.cpp (under the ~1.35x
criterion) with BETTER quality than llama.cpp's own default path (MMLU
exact 62.4-63.6% / agreement 96.4-96.8%, ppl ratio ~0.998); MXFP4 is
7682 tok/s (1.38x over exact) but quality-FAILING for eval-class work
(MMLU 48.8%, agreement 58.0%). The fork's remaining value is a 1.38x
speed option for quality-insensitive bulk prefill -- narrow, and it costs
every default user reference quality on eval-class workloads. DECISION:
flip. prefill_precision_exact() now defaults true;
TEMPURA_PREFILL_PRECISION=mxfp4 restores the fast path (code + selector
KEPT -- the right choice for bulk work, just not the default). (3) GLU
FUSION SKIPPED (recorded, not attempted past design): the bar was met
without it; the honest estimate was ~0.6-0.9 ms/step for a full
restructure (interleaved gate/up staging or 32-acc double-compute, smem
co-residency risk at 49 KB/CTA), and the adjacent prior is NEGATIVE (the
2026-10-29 MXFP4 epilogue-read fusion measured a regression from
scattered 32B transactions, 7257 -> 6994). If prefill speed matters
later, the levers are recorded here and in kernels/AGENTS.md. (4) GATES
GREEN on the final tree: probe-q6k-batch PASS (all arms incl. the new
S=2/S=4 split arms, split mirror bit-identical); probe-q4k-batch PASS
(unchanged arms); sanitizer-gate PASS (memcheck 3/3 + racecheck 2/2 +
initcheck 2/2, 0 errors/0 hazards outside the documented db-handoff
allowlist -- the q6k split kernel + reduce entry are inside the probe
coverage; one host reboot mid-run lost the first attempt, the rerun
completed clean); bench-real-model exact 5584 tok/s pp128 det=OK, decode
218; the MXFP4 path measured unchanged pre-flip at 7682 det=OK; local
cargo test 930 passed / 0 failed (929 + the q6k split-reduce PTX test).
(5) GATES PENDING FOLLOW-UP (honest gaps -- the flip changes only the
DEFAULT env routing, and the exact route is the same code path these
gates exercised green at 5189 under TEMPURA_PREFILL_PRECISION=exact on
2026-11-21, but the q6k-split sum-order change means they should be
re-run on the flipped tree): bench-ab-repro, the 6-client kv-swap A/B on
f16 AND e4m3, nonstream-route-smoke, kv-restart-var, MMLU, pooled ppl
(re-baseline expected in the 8.086-8.089 exact-class band). (6) The
kernels/AGENTS.md blocks: 2026-11-21b (q6k split), 2026-11-21c (the
epilogue washes -- do not rebuild without new evidence), 2026-11-21d
(the flip).**

**STATUS 2026-11-21 (q4k mmq grid.z k-split -- the first of the three
documented levers LANDED: exact mode 4381 -> 5189 tok/s pp128 (+18.5%,
llama.cpp gap 1.7x -> 1.4x), the within-kernel q4k_mmq breakdown measured
BEFORE choosing (gate/up 64% of the kernel, q/o+k 36% with k_proj running
at 19% SM fill -- 0.059 ms at n=1024, the same latency as n=4096), an
S-sweep per shape recorded, the 6000-tok/s DEFAULT-FLIP BAR NOT MET (needs
~3.3 ms more; the remaining levers are each sub-2.5 ms even optimistically)
so exact stays opt-in and MXFP4 stays default, TWO bring-up bugs caught by
the probe (writeback split-branch polarity inverted; the splits param
register %r7 clobbered by the warp_id decomposition -- warps != 1 wrote
partials through the null pointer), and all gates green): (1) THE
WITHIN-KERNEL BREAKDOWN (bench-probe-prefill-tc, mmq variant added, m=128
per call): q 0.059 + k 0.059 + o 0.059 + gate 0.158 + up 0.158 = 0.493
ms/layer -> 15.8 ms/step isolated (the engine-profile 20.3 ms includes
cache-cold activations + interleaving); gate/up (448 CTAs, 210 GB/s
effective) is issue/latency-bound NOT fill-bound, while k_proj (32 CTAs =
19% of 170 SMs, 40 GB/s) and q/o (128 CTAs = 75%) are fill-limited -- the
k-split lever targets exactly those. (2) THE KERNEL (all in
kernels/src/q4k_mmq_batch_sm120_ptx.rs, same module): two new params
(splits u32, partials ptr); CTA z = ctaid.z owns k-blocks [z*kb/S,
(z+1)*kb/S) (u32 div, S=1 covers [0,kb) -- the unsplit path numerically
UNCHANGED); writeback branches on splits==1 to the direct out store, else
to partials[z*m*n + token*n + neuron]; NEW entry q4k_mmq_split_reduce in
the same module combines out[i] = p0 + p1 + ... in ASCENDING z (fixed
order, deterministic; f32 adds; grid ceil(m*n/256)). NUMERICS: per-split
accumulation identical (same (kb,g,jt) order in range); the combine changes
only the f32 SUM ORDER -- the established mma128-class band, gated by NEW
probe arms (S=2 and S=4 at every shape x m vs a SPLIT CPU mirror:
mmq_cpu_mirror_split models per-split accumulation + the ascending-z
combine; measured max rel 1.5e-8..6e-8 vs mirror at every shape, class
band vs kpar unchanged at ~5e-5). Host:
forward_mmq_split_batch_d2d_rows (oxide_q4k_gemv_host) + has_mmq_split_batch
+ sm_count accessor; splits<=1 delegates to the unsplit method. ENGINE
(gemm_q4k_prefill_batch Mmq arm): S depends ONLY on n, NEVER the runtime m
-- a smaller final chunk must not change a token's sum order (composition
invariance, validated by bench-ab-repro composition); slots computed at the
fixed 128-row chunk; rule calibrated on the fishtank S-sweep: slots < 256
-> S = round(640/slots) (k_proj 32 slots -> 8, q/o 128 -> 5), else S = 3
(gate/up 448 -> 3), clamped [1,8] and <= k_blocks. TEMPURA_Q4K_MMQ_SPLIT=0
bisects off (S=1 everywhere). (3) THE MEASURED MILESTONES (fishtank,
bench-probe-prefill-tc m=128, then bench-real-model exact pp128): k_proj
0.059 -> 0.019 ms (3.1x, S=8), q/o 0.060 -> 0.049 (-18%, S=5), gate/up
0.157 -> 0.146 (-7%, S=3; the partials+reduce traffic ~cancels the fill
gain below S=3 -- q/o at S=2/3 was a WASH, the sweep decides per shape);
per-layer 0.497 -> 0.416 ms. END-TO-END exact: 4381 -> 5012 tok/s with
S=round(640/slots) only, -> 5189 tok/s with gate/up S=3 included (decode
218 unchanged, logits det=OK, fingerprint re-baselined to 170432546484652
-- the sum-order class change, expected). Default MXFP4 path UNTOUCHED:
7682 tok/s det=OK (unchanged band). (4) THE S-WEEP DATA (SWEEP=1 mode in
bench-probe-prefill-tc): k_proj best S>=4 (0.019-0.020), q/o best S=5
(0.049, noisy plateau 0.049-0.061), gate/up best S=3 (0.146 vs 0.157 S=1),
down-shape [4096,14336] (Q6K side, NOT wired -- the q6k_mmq kernel has the
same 128-CTA fill problem) best S=5 (0.158 vs 0.206 -- a further ~0.5
ms/step is available by porting the split to q6k_mmq for down/v_proj).
(5) THE REMAINING LEVERS, honestly NOT attempted this session (the bar
needs ~3.3 ms more; f16x2 ~2 ms optimistic + GLU ~0.8 ms would land ~5900,
short of 6000 -- better to bank the gated win than half-land two): (i)
f16x2 scale pipes: paper analysis says the win is NOT free -- the products
must reach the f32 accumulators, so every mul.f16x2 saved pays an unpack
cvt.f32.f16 pair (5 packed muls + 12 unpack cvts vs the current 10 muls +
4 cvts: a wash on issue count, the gain exists only if the f16 pipe
dual-issues under the cvt wall -- MEASURE a variant entry before believing
it); a cheaper epilogue restructure exists: acc += d8 * fma(D, ax, ay*rs)
(12 fp ops vs 14, fewer roundings, different class -- same probe-mirror
workflow); (ii) GLU-fused writeback: needs interleaved gate/up weight
staging (llama.cpp chunks-of-8) so one CTA tile holds (gate[i], up[i])
pairs + the silu math op-identical to the standalone kernel; saves the
silu launch + ~30 MB/layer of intermediate traffic (~0.6-0.9 ms/step);
(iii) the q6k-side split (see (4)). (6) GATES ALL GREEN on the final tree:
probe-q4k-batch PASS (28 mmq arms + 56 NEW split arms S=2/S=4 at every
shape x m incl. k=14336, ulp-level vs the split mirror); probe-q6k-batch
PASS (unchanged); sanitizer-gate PASS (memcheck 3/3 + racecheck 2/2 +
initcheck, 0 errors 0 hazards -- the split kernel + reduce entry are
inside the probe coverage); bench-real-model exact 5189 tok/s pp128
det=OK, decode 218; default MXFP4 7682 tok/s det=OK (unchanged);
bench-ab-repro exact solo/repeat/interleave/composition IDENTICAL;
6-client kv-swap A/B PASS on f16 AND e4m3 (byte-identical, eviction
exercised, both under TEMPURA_PREFILL_PRECISION=exact); nonstream-route-
smoke PASS (exact); kv-restart-var PASS (exact); MMLU exact 62.4% GPU /
61.6% CPU / 62.4% llama, agreement 96.8% PASS (gate 95%); pooled ppl
8 chunks 8.0909, GPU/CPU ratio 0.9983, 0 NaN PASS (exact-class baseline
8.0863-8.0889, ratio 1.0005 -- the split is quality-neutral); local cargo
test 929 passed / 0 failed. (7) THE DECISION: exact stays OPT-IN at 1.4x
from llama.cpp; the k-split is the exact route's DEFAULT (bisect:
TEMPURA_Q4K_MMQ_SPLIT=0); the quality/speed fork REMAINS (MXFP4 7682 vs
exact 5189) and the 6000-tok/s flip bar is carried forward with the (i)-(iii)
levers above.**

**STATUS 2026-11-20 (q6k mmq port: the llama.cpp-mmq treatment ported to the
Q6K exact-prefill projections via the NEW q6k_mmq_batch_sm120 kernel, the
q6k stack 6.9 -> 4.64 ms/step (1.49x), exact mode 4101 -> 4381 tok/s pp128
(llama.cpp gap 1.8x -> 1.7x), the numerics class extended to the Q6K side
and RE-VALIDATED (MMLU 63.6%/96.4% agreement, ppl 8.0889 vs the 8.0863
baseline), the 6000-tok/s DEFAULT-FLIP BAR NOT MET so exact stays opt-in
and MXFP4 stays default, a long divergence hunt resolved as a PROBE-GATE
BUG (the mirror's abssum dropped the |.| -- the kernel was correct), a
borrow-free SWAR s8 unpack landed (8 ops per 4-element word instead of
~28), and the re-profiled step breakdown recorded (q4k_mmq 20.3 ms = 71%
of the step -- it is now the entire game); all gates green): (1) THE
KERNEL (kernels/src/q6k_mmq_batch_sm120_ptx.rs, PTX generated by ptx()):
the q4k_mmq_batch_sm120 skeleton ported to Q6K's packed 53-u32 format --
one CTA covers 128 neurons x 32 tokens, ONE 37888-byte dynamic-smem
staging buffer via cp.async (w_raw 24576 B raw ql+qh + w_dsc 4096 B f16
scale pairs + a_qs 8192 + a_ds 1024; 2 CTAs/SM co-reside -- the q4k
co-residency lesson), weights staged RAW and unpacked to s8 A-fragment
words at load with a BORROW-FREE SWAR: per 4-element word,
v = ((ql >> sh) & 0x0F0F0F0F | (((qh >> qsh) & 0x03030303) << 4))
+ 0x60606060 ^ 0x80808080 -- per-byte x + 96 never carries (x <= 63) and
the bit-7 toggle maps x + 96 to the two's-complement encoding of x - 32
for both halves of the range; f16 (d*sc) pre-extraction per (neuron,
sub-block) computed as ONE f32 product rounded ONCE to f16 (cvt.rn.f16.f32
-- no pre-rounding of d, so arbitrary f32 d values match the CPU mirror
exactly; real GGUF Q6K d is f16-exact anyway); the 2-mma k16 split for
the per-16-element scales (mma1 B = {b0, 0}, mma2 B = {0, b1}, each D the
exact s32 sum of one sub-block). 56 registers, 0 spills. Host:
forward_mmq_batch_d2d_rows (oxide_q6k_gemv_host.rs, cuFuncSetAttribute at
load); ENGINE: gemm_q6k_prefill_mma128 routes through mmq when loaded and
n % 128 == 0 (down_proj 4096, lm_head 128256 both qualify);
TEMPURA_Q6K_PREFILL_KERNEL=mma128 bisects back to the exact-int kernel.
(2) THE MEASURED RESULT (fishtank, RTX 5090, Llama-3.1-8B Q4_K_M, exact
mode, nsys on bench-real-model pp128, 28 steps): q6k_mmq 4.64 ms/step
(was 6.9 with mma128 -- 1.49x; the mma128 skeleton's warp-strided gw loop
left 7/8 of the warps idle per group and its ~80-op scalar B assembly ran
per tile), q4k_mmq 20.3 ms (71%), flash attention 1.39, quantize_q8_1
0.79, rope/norms/silu/add ~0.8. End-to-end: exact 4309-4381 tok/s pp128
(was 4073-4101), decode unchanged 206-217, logits det=OK; default MXFP4
path unchanged (7758 tok/s, det=OK). The 4381 number vs the 6000 bar:
NOT MET -- the q6k port bought 7% because the q6k side was only 22% of
the step; the step is now 71% q4k_mmq, whose remaining levers carry the
flip: (i) f16x2 scale products on the separate f16 pipe for the q4k/q6k
epilogues (the consumer-Blackwell SASS check showed f32x2 FFMA splits to
scalar -- the f16 pipe is the only dual-issue escape), (ii) GLU-fused
writeback (gate/up + SiLU in the GEMM epilogue, llama.cpp has_fusion,
~0.3 ms of silu + intermediate traffic visible in the profile), (iii)
k-split grid fill for the 128-CTA shapes (down_proj n/128 x m/32 = 128
CTAs on 170 SMs = 75% fill; needs a deterministic reduction), (iv) the
q6k kernel's own grid fill (same 75% problem). (3) THE NUMERICS GO/NO-GO:
the f16 (d*sc) class on the Q6K side PASSES the quality gates -- MMLU
exact GPU 63.6% (CPU 61.6%, llama 62.4%), GPU/CPU agreement 96.4% (gate
95%); pooled ppl 8 chunks 8.0889 (exact-class baseline 8.0863, ratio
1.0003; GPU/CPU ratio 0.9980, 0 NaN). Both within the run-to-run band of
the 2026-11-19 measurements (62.8%/96.8%/8.0863) -- the q6k scale
roundings are quality-neutral exactly as the q4K side's were. (4) THE
PROBE ARM (bench-probe-q6k-batch arm 4) and its two lessons: gates are
(a) BITWISE vs q6k_mmq_cpu_mirror -- a CPU model of the EXACT op sequence
(quantize model incl. the oxide TRUNCATING f16 d8 converter, the single
product->f16 RNE, exact int32 sub-block dots, fma epilogue in (kb, gw,
sub) order) and (b) the class band vs the mma128 output, cancellation-
aware |got - ref| <= 2e-3 * abssum + 1e-6 (measured err/gate 0.02-0.06 at
every shape x m incl. k=14336 and the lm_head n=128256). LESSON ONE: the
mirror's abssum initially accumulated SIGNED terms (a dropped |.|), which
collapses the gate denominator to ~|acc| on cancellative sums and blew
the band up to err/gate 1e2-1e6 -- a three-hour divergence hunt (debug
f64 scans, determinism relaunches, per-sub decompositions) that chased a
kernel bug that did not exist; the kernel was bit-identical to the mirror
from the first product-rounding fix. Keep |.| on abssum terms. LESSON
TWO: the probe's random d must be f16-EXACT like real GGUF data (a random
f32 d exposed a d pre-rounding difference between mul.f16 and the mirror
that cannot occur in production; fixed in build_block AND made moot by
the product-then-round kernel semantics). (5) GATES ALL GREEN on the
final tree: probe-q6k-batch PASS (28/28 mmq arms, bit-identical mirror);
probe-q4k-batch PASS (unchanged); sanitizer-gate PASS (memcheck +
racecheck + initcheck over both probes and bench-decode-ops, 0 hazards --
the new kernel is inside the probe coverage); bench-real-model exact
4309-4381 tok/s det=OK, decode 206-217; default MXFP4 7758 tok/s det=OK
(unchanged); MMLU exact 63.6%/96.4% PASS; pooled ppl exact 8.0889 ratio
0.9980 PASS; bench-ab-repro solo/repeat/interleave/composition IDENTICAL;
6-client kv-swap A/B PASS on f16 AND e4m3 (byte-identical, eviction
exercised); nonstream-route-smoke PASS; kv-restart-var PASS; local cargo
test 929 passed / 0 failed (928 + the q6k_mmq ptx test). (6) THE
DECISION: exact stays OPT-IN (TEMPURA_PREFILL_PRECISION=exact) at 1.7x
from llama.cpp; within the exact route, q6k mmq is the DEFAULT kernel;
the quality/speed fork REMAINS (MXFP4 7758 vs exact 4381) and the
6000-tok/s flip bar is carried forward with the (i)-(iv) levers, now
concentrated in the q4k_mmq epilogue and the grid-fill problem.**

**STATUS 2026-11-19 (mmq-class int8 prefill restructure: the llama.cpp-mmq
port EXECUTED -- new q4k_mmq_batch_sm120 kernel (128 neurons x 32 tokens
per CTA, cp.async smem staging, f16 scale pre-extraction), exact mode
2732 -> 4101 tok/s pp128 (1.5x, llama.cpp gap 2.7x -> 1.8x), the f16-scale
numerics class VALIDATED (MMLU 62.8%/96.8% agreement, ppl 8.0863 vs the
exact-class 8.0869), the 6000-tok/s DEFAULT-FLIP BAR NOT MET so exact
stays opt-in and MXFP4 stays default, the fp8-sim harness extended with
the q8act/mmq simulation arms, an honest negative on the CPU-side mmq MMLU
sim (7h projected runtime, killed -- the kernel-level gates carried the
go/no-go), and FOUR kernel-iteration measurements recorded (single-buffer
register staging 3635, cp.async double-buffer 3938, J=64 3318 -- grid
tail, single-buffer + 3 CTAs/SM 4101 -- the winner); all gates green):
(1) THE NUMERICS GO/NO-GO: the mmq class (f16 (d*sc)/(dmin*mv) weight
scale products via mul.f16 at staging -- llama.cpp's load-time half2
pre-extraction; everything else the mma128 class: raw nibbles, exact int
q8 rowsum, f32 epilogue in kb order) PASSES the quality gates:
MMLU exact GPU 62.8% (CPU 61.6%), choice agreement 96.8% (the exact-class
level, gate 95%); pooled ppl 8 chunks 8.0863 (exact-class baseline
8.086885, delta 0.0006; GPU/CPU ratio 0.9977, 0 NaN). The fp8-sim harness
(kernels/src/fp8_sim.rs) gained the "q8act" (Q8_1 activations + f32 scale
chain -- the exact-GPU control class) and "mmq" (= q8act + the f16
(d*sc)/(dmin*mv) roundings; the min term keeps the exact int rowsum,
matching the kernel -- slightly tighter than llama.cpp's f16 sum field)
simulation arms with a
proptest-gated kernel-level band test; the CPU-side MMLU run of the mmq
arm measured ~100 s/question scalar (7 h for 250 q -- the sim_gemm_q8
path is a per-neuron scalar loop, unlike the e4m3 sim's gemm-crate
route) and was killed by PID; the class evidence therefore comes from
the real kernel's MMLU/ppl (stronger evidence than a simulation anyway).
(2) THE KERNEL (kernels/src/q4k_mmq_batch_sm120_ptx.rs, PTX generated by
ptx()): grid (n/128, ceil(m/32)), block 256 (8 warps, warp owns 16
neurons), ONE 29696-byte DYNAMIC-smem staging buffer (w_qs raw nibble
words 16 KiB + w_dm f16 scale half2s 4 KiB + a_qs q8 8 KiB + a_ds
group-major ds words 1 KiB; 3 CTAs co-reside per SM at this size),
per-k-block schedule: bar -> cp.async staging -> w_dm f16 products ->
wait_group -> bar -> 256 mma.sync.m16n8k32 (8 g x 4 jt token tiles) ->
f32 epilogue (pos = ax*d8, tb = d8*rs, min = ay*tb, acc = fma(D,pos,acc)
+ min, kb order). The nibble planes are unpacked AT A-FRAGMENT LOAD
(both planes share the raw word; ~6 bit-ops per (warp, group) amortized
over 4 mmas). Host: forward_mmq_batch_d2d_rows (oxide_q4k_gemv_host),
cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) at load (the q6k batch
pattern). ENGINE DEFAULT FLIP: TEMPURA_Q4K_PREFILL_KERNEL now defaults to
mmq for the exact route's m>16 projections (mma128 still selectable via
=mma128, kpar8/wide/streamk unchanged); the GLOBAL default stays MXFP4
(the 6000 bar was not met -- measured 4101). (3) THE ITERATION HISTORY
(all probe-gated before benchmarking; each step's lesson recorded):
(a) v1 register staging (single buffer, static 45 KiB expanded-weight
smem): 3635 tok/s, ncu showed 20-30% SM compute utilization, 8-13.8
achieved warps -- LATENCY-bound, the k-block staging serialized against
compute; (b) v2 cp.async double buffer (59392 B dynamic, raw weights):
3938 tok/s -- the pipeline helped 1 CTA/SM but the SM smem (~100 KiB)
caps co-residency at 1 CTA, losing v1's 2; (c) J=64 (8 token tiles,
77824 B): 3318 tok/s -- the grid tail (224 CTAs = 1.32 waves at 1 CTA/SM)
dominates, REVERTED; (d) FINAL: single 29696-B buffer + cp.async + 3
CTAs/SM -- the co-resident CTAs cover each other's staging latency the
way the software pipeline could not at 1 CTA: 4073-4101 tok/s. The
remaining 1.8x gap to llama.cpp's 7335: ncu shows the kernel still at
~30% compute / 6.2 warp-cycles-per-instruction with 24 warps/SM -- the
per-mma f32 epilogue chain (~22 scalar fp/cvt ops per mma; f32x2 packed
ops COMPILE on sm_120 but ptxas SPLITS them to scalar FFMA -- consumer
Blackwell lacks the datacenter dual-f32 pipe, verified by SASS dump)
plus the J=32-only 4-mma A-fragment amortization (J=128 needs 54 KiB/
buffer, unaffordable at 2+ CTAs). The follow-up levers, in priority
order: (i) f16x2 scale products on the SEPARATE f16 pipe (the cvt wall
must be profiled first), (ii) GLU fusion into the writeback (llama.cpp's
has_fusion, ~0.8 ms/step of silu + intermediate traffic), (iii) the q6k
side (6.9 ms/step untouched -- same skeleton, 2-mma k16 split),
(iv) k-split grid.z for the q/o shapes (128 CTAs < 170 SMs, 75% fill).
(4) BRING-UP BUGS (all caught by the probe/sanitizer, the gates doing
their job): word-vs-byte unit confusion in the activation staging and
writeback addressing (three separate instances -- misaligned-address
716, then a 4x row scramble in the a_ds path that produced the
characteristic "output token t = true token 4t" pattern, isolated by a
constant-per-token activation experiment + an in-kernel smem dump);
goff not reset per k-block (B-fragment addresses walked out of the
buffer after kb 1); the g>=4 6-bit scale unpack leaked bits from the
adjacent byte (shr without a 0xC0 mask before the <<4); J=64's
token_base left at y*32 (tokens 96-127 never computed). The probe arm
gates the kernel TWO ways: vs the per-row kpar8 reference at the class
band (measured 2.8e-5..8.3e-5, gate 2e-3) and vs a CPU mirror of the
EXACT op sequence (mmq_cpu_mirror -- models the GPU quantizer including
the oxide f32_to_f16_bits TRUNCATING converter and mul.f16's RNE; ulp-
level agreement 1.5e-8..3e-8, gate 1e-5 -- bit-identity is not claimed:
a residual ptxas contraction nuance keeps it at last-ulp). (5) GATES
ALL GREEN on the final tree: probe-q4k-batch PASS (28/28 mmq arms at
every shape x m in {1,2,4,6,16,64,128} incl. k=14336); probe-q6k-batch
PASS; bench-real-model exact 4073-4101 tok/s pp128 / decode 199-206,
logits NaN-free det=OK (fingerprint 170501989534200); bench-real-model
default (MXFP4) 7756 tok/s, det=OK (unchanged -- the default path is
untouched); MMLU exact 62.8%/96.8% PASS; pooled ppl exact 8.0863 ratio
0.9977 PASS; bench-ab-repro solo/repeat/interleave/composition IDENTICAL;
6-client kv-swap A/B PASS on f16 AND e4m3 (byte-identical, eviction
exercised); nonstream-route-smoke PASS; kv-restart-var PASS;
sanitizer-gate PASS (memcheck + racecheck + initcheck over both probes
and bench-decode-ops, 0 hazards -- the new kernel is inside the probe
coverage); local cargo test 928 passed / 0 failed (926 + the
q4k_mmq ptx test + the mmq-sim band test). (6) THE DECISIONS: exact
stays OPT-IN (TEMPURA_PREFILL_PRECISION=exact) at 1.8x from llama.cpp;
within the exact route, mmq is the new default kernel; the quality/speed
fork REMAINS (MXFP4 7756 vs exact 4101) and the 6000-tok/s flip bar is
carried forward with the (i)-(iv) levers above.**

**STATUS 2026-11-18b (int8 tensor-core prefill: the kernel-efficiency gap
to llama.cpp's mmq measured, ONE landed win (Q8_1 rowsum precompute,
bit-identical, exact mode 2508 -> 2732 tok/s pp128), ONE refuted
hypothesis (the wide-neuron restructure measured a WASH -- probe-gated,
landed opt-in), the full llama.cpp mmq structural analysis recorded, and
the DEFAULT-FLIP DECISION: NOT MET, exact stays opt-in at 2.8x from
llama.cpp; all gates green): (1) THE PROFILE (fishtank, RTX 5090,
Llama-3.1-8B-Instruct Q4_K_M, exact mode, nsys on bench-real-model pp128,
after the rowsum win, 2730 tok/s = 46.9 ms/step): q4k_mma128 35.9 ms/step
(77%), q6k_mma128 6.9 ms (14%), flash attention 1.4 ms (3%),
quantize_q8_1 0.8 ms, rope/norms/silu/add ~0.8 ms. The Q4K mma128 stack
is the entire game: at gate/up (n=14336, k=4096) m=128 it runs 0.287 ms
= ~230 GB/s effective weight read vs the 5090's 1792 GB/s DRAM -- the
kernel is NOT DRAM-bound; the issue-floor estimate (45 warp-instructions
per mma.sync) is ~65-100 us, so it is not purely issue-bound either --
it is bound by the per-mma scalar chain's latency/structure. (2) THE
LLAMA.CPP MMQ STUDY (the existence proof at 7335 tok/s; read from the
fishtank checkout, ggml/src/ggml-cuda/mmq.*): their block is I=128
NEURONS x J tokens (J up to 128), 256 threads, occupancy 1, K_vram=256,
stream-K with a tile-efficiency-aware grid (pure tiling when >= 90%).
The structural differences that explain the gap, in estimated order:
(a) their mma loop is ~2 FFMA per output element -- the Q4_K loader
PRE-EXTRACTS (d*sc, dmin*mv) as half2 per (neuron, sub-block) at load
time and the activation Q8_1 layout embeds (d, d*sum(q)) as half2, so
the per-element work is (dmA.x*dsB.x)*C + dmA.y*dsB.y; OURS is ~7 fp ops
per element (8 scale-factor muls per mma + per-element mul/mul/sub/add
+ cvts) with f32 scales -- a numerics-class choice (f16 scale rounding
vs our f32) that their reference quality carries but our <= 1e-5
probe gate does not; (b) A/B fragments come from SMEM via ldmatrix
(cooperative coalesced staging of both operand tiles) vs our per-thread
global loads with per-tile address recomputation; (c) FUSED gate/up+GLU:
weight rows interleaved in chunks of 8 so one C tile holds up+gate pairs
and the SiLU-GLU is applied in the writeback -- removes the separate
silu_mul kernel and one output round-trip on the LARGEST GEMM; (d) the
Q8_1 mmq layout's partial-sum field (s = d*sum(q)) is EXACTLY the rowsum
precompute we adopted this session (see (3)); (e) column-tile-innermost
block order for L2 sharing of the weight tile. (3) THE LANDED WIN -- the
2026-11-17d follow-up (a): quantize_q8_1 (tools/oxide-gemv-q4k) now
computes the block's exact integer q8 rowsum (warp butterfly, |sum| <=
32*127 < 2^12) and stashes it in the HIGH 16 bits of the ds word
(llama.cpp's Q8_1 ds layout); every other consumer reads only the low
f16 via b16 loads (verified by grep over all ds readers) so all decode /
kpar8 / mma16 / streamk numerics are UNCHANGED; q4k_mma128 (and the new
wide kernel) read the full ds word -- the min term's rowsum becomes one
b16 load instead of ~38 scalar ops + 4 shuffles per (row, sub-block,
m16 tile). Probe-verified bit-identical (mma128 arm still bitwise vs
chunked mma16). MEASURED: Q4K projection stack projected 4241 -> 4735
tok/s pp128; end-to-end exact mode 2508 -> 2732 tok/s (llama.cpp 7335 =
2.7x, was 2.9x); decode unchanged (217); MMLU exact 62.8%/agreement
98.0% PASS (quality-neutral as constructed); default MXFP4 path
unchanged (7771 tok/s, ppl at baseline). (4) THE REFUTED HYPOTHESIS --
the wide-neuron restructure q4k_mma128w_batch_sm120
(kernels/src/q4k_mma128w_batch_sm120_ptx.rs, PTX generated by ptx() --
OnceLock + format!-unrolled tiles): the mma128 block covers only 8
neurons, so the activation A-fragments are re-read from L2 once per 8
neurons (at m=128 that is ~14x the weight bytes; gate/up ~917 MB L2 vs
66 MB weights) plus a CTA bar per (kb, 8-neuron group) -- the
llama.cpp-I=128 reading suggested widening. The wide variant: 64
neurons/block, each WARP owns 8 neurons (no mw stripe, warp-private
header smem synced with a CTA bar per k-block (a bar.warp.sync variant
was rejected: racecheck cannot model warp-scope barriers and the
sanitizer gate flagged the header handoff), direct-store epilogue -- the cross-warp 8-partial smem reduction is GONE), grid.y
row-splitting (row_span param) so small-N shapes fill the SMs; the
accumulator budget (32 f32 = 8 tiles x 4 D) stays because warp-owned
accumulators are non-redundant. NUMERICS: sequential (kb, mw)
accumulation in one warp -- NOT bitwise vs mma16/mma128 (the fp op
sequence per element is identical); probe arm mma128w gates <= 1e-5 vs
kpar8, measured 1e-8..4e-7 across all shapes x m in {1..128}.
MEASURED (grid_y swept 1/2/4/8 at every prefill shape): a WASH -- at
the best grid_y per shape: q/o 0.103 vs mma128's 0.102 ms, k_proj 0.059
vs 0.047, gate/up 0.287 vs 0.293, down-shape 0.353 vs 0.358. The
activation-L2-re-reading hypothesis is REFUTED: removing 8x of L2
traffic and all CTA bars does not move the kernel -- the bound is the
per-mma f32 scale chain + A-fragment load structure, which are
layout-independent. VALIDATION-PROCESS NOTE (the gate doing its job,
twice): the first version synced the header handoff with bar.warp.sync
-- racecheck cannot model warp-scope barriers and the sanitizer gate
FAILED it (55 hazards); switching to the CTA bar exposed the REAL bug
racecheck had also been pointing at: the MW_DONE bar of the original
mma128 was dropped in the restructure, so k-block kb+1's header store
could overwrite the warp smem tables while other warps still read
kb's -- a genuine cross-iteration data race that the numerics probe
passed anyway (warps near-lockstep; passing was luck, not correctness).
Fixed with the double-bar-per-kb structure (bar after header + bar at
MW_DONE, verbatim the original's); racecheck now 0 hazards / 0 warnings
and the perf is unchanged (the wash numbers hold). LANDED as opt-in
(TEMPURA_Q4K_PREFILL_KERNEL=wide; host heuristic targets ~256 blocks;
TEMPURA_Q4K_WIDE_GY overrides grid.y) with the probe arm + docs -- the
correct-and-available variant for follow-up work on the scale chain.
Bring-up bug (fixed same day, found by the probe): the tile row
computation dropped the +gid (row_lo = row0 + t*16 -- m=1 passed by
accident, every other row read row t*16's data; 0.03-0.3 rel err,
m-1 of m rows bad). (5) THE DEFAULT-FLIP DECISION: the bar was >= ~6000
tok/s pp128 at reference quality; measured 2732 (best of mma128 /
mma128w at every shape). THE FLIP DOES NOT HAPPEN: the default stays
MXFP4 (7771 tok/s), exact stays the documented opt-in for
quality-sensitive workloads (eval harnesses, log-prob scoring) at 2.8x.
The remaining-gap analysis for any future attempt, in priority order
per the mmq study: (a) restructure the scale chain to the 2-FFMA form
(load-time (d*sc)/(dmin*mv) pre-extraction -- NOTE this means f16 scale
precision, llama.cpp's exact class, a numerics-class change our current
1e-5 probe gate would need to re-baseline against MMLU/ppl, not
bit-identity); (b) smem staging of the A tile per k-block + ldmatrix
fragment loads (kills the per-tile address arithmetic AND the global
latency); (c) fused gate/up+GLU writeback (llama.cpp's has_fusion,
interleaved 8-row chunks); (d) Q6K side same treatment (6.9 ms/step,
same skeleton). (6) GATES ALL GREEN on the final tree: probe-q4k-batch
PASS (all arms incl. the new mma128w at every shape x m); probe-q6k-batch
PASS; bench-real-model exact 2732 tok/s pp128 (2721-2732 across runs)
/ decode 217, logits
NaN-free det=OK; bench-real-model default 7771 tok/s pp128 (unchanged),
det=OK; MMLU exact GPU 62.8% / CPU 61.6% / llama 62.4%, GPU/CPU
agreement 98.0% PASS (gate 85%), GPU/llama 95.6%; pooled ppl 8 chunks:
default f16 8.683812 / e4m3 8.628168 -- EXACTLY the standing baselines
(8.6838 / 8.6282; the rowsum change is numerics-neutral on the default
path, doubly confirmed), exact f16 8.086885 (the 2026-11-17d exact
baseline 8.0869; ratio 0.9978 vs CPU, 0 NaN); bench-ab-repro
solo/repeat/interleave/composition checksums IDENTICAL;
6-client kv-swap A/B PASS on f16 AND e4m3 (byte-identical, eviction
exercised); nonstream-route-smoke PASS; kv-restart-var PASS;
sanitizer-gate PASS (the changed quantize kernel is inside its memcheck
coverage); local cargo test 926 passed / 0 failed (925 + the new
q4k_mma128w ptx test).**
**STATUS 2026-11-18 (mid-precision FP8 prefill: NEGATIVE RESULT, simulation-first
per the stop rule — an e4m3-activation x e4m3-weight prefill GEMM class does NOT
pass the MMLU agreement gate (90.8% vs the 95% gate; exact 96.8-98.0%, MXFP4
58.0%), so NO GPU kernel work was done; the e4m3 ACTIVATION quantization is the
dominant cost (act-only 89.6%); the CPU-side simulation harness landed as
tools/fp8-sim + kernels/src/fp8_sim.rs, default-off and numerics-identical on
the default path; all gates green): (1) THE HYPOTHESIS UNDER TEST (the
2026-11-17c/d follow-up): the MXFP4 quality failure is the E2M1 activation
quant (0-1 mantissa bits, rel~0.1 logit band) + the Q6K->MXFP4 weight
double-quant; e4m3 activations (3 mantissa bits + own exponent) with
Q4K/Q6K->e4m3 weights should shrink the band enough to pass MMLU at a speed
approaching MXFP4 (Blackwell e4m3 x e4m3 mma), eliminating the 2508-vs-7790
tok/s quality fork. (2) THE METHOD — quantify BEFORE kernel work: a CPU-side
simulation INSIDE the real quantized forward (kernels/src/fp8_sim.rs): every
Q4K/Q6K prefill projection rounds its activations and/or dequantized weights
through a HARDWARE-FAITHFUL e4m3 round-trip (subnormals preserved — the
KV-cache helpers flush them, a KV convention, NOT mma semantics) then computes
the GEMM in f32 — the exact numerics class of an e4m3 x e4m3 mma with f32
accumulation, modulo sum order. Activation through engine/model.rs's
forward_quantized via a new MatMulQDispatch (default = pure delegation to
CpuMatMulQ4KOpt; TEMPURA_FP8_SIM selects the sim; the x8 repacked prefill
kernels decline in sim mode so projections fall back to the generic
qmm_indexed route the sim covers; decode-class preq/fused paths stay real —
fp8 targets prefill only). Mode bits: act / weight / wscale (per-tensor weight
scale, absmax -> 224) / arowscale / ftz / noop (route-switch control).
tools/fp8-sim (new workspace crate) runs the 250q MMLU harness twice on one
model (reference pass = mode OFF, bit-identical to stock) and reports
accuracy, choice agreement, and the logit band. TWO findings baked into the
harness: (a) plain e4m3 weight quantization is CATASTROPHIC without a
per-tensor scale — e4m3's min normal is 2^-6 = 0.0156 and real Q4K weight
magnitudes straddle it, so unscaled small weights land in the subnormal range
where e4m3 has 0-2 mantissa bits (the earlier "no block scales needed, the
exponent suffices" reading is WRONG; a per-TENSOR scale is mandatory — still
no block scales); (b) the CPU forward's x8-repacked route and the generic
qmm route differ by a band of 0.021 (noop control, 100% agreement, 12q) — a
pre-existing Q8Kx4-vs-Q8K quantization-class difference between the two
shipping routes, small but nonzero; the noop control now exists to measure
it. (3) THE NUMBERS (fishtank, Llama-3.1-8B-Instruct Q4_K_M, 250q MMLU,
mode act+weight+wscale): reference 61.6%, sim 64.0% (flips are near-ties both
ways — accuracy moves are noise), choice agreement 90.8% = 23/250 flips FAIL
(gate 95%), logit band: max |delta| 1.42, mean 0.169, max rel to peak |logit|
0.074 vs the MXFP4 GPU band 0.10-0.13 — only ~1.5x better, NOT the hoped
8-16x. ATTRIBUTION (48q each): act-only (e4m3 acts, exact-dequant weights)
agreement 89.6%, band 0.060 — the ACTIVATION quantization alone fails the
gate and is essentially the whole combined band; weight-only+wscale (e4m3
weights, exact acts) 95.8%, band 0.053 — borderline. (4) THE CONCLUSION: an
fp8 mma path REQUIRES e4m3 activations (mixed-operand mma does not exist),
and e4m3 activations alone fail the MMLU gate — the mid-precision FP8 prefill
is DEAD as a quality-passing default. The quality cliff sits between 3
mantissa bits (e4m3, fails at ~2% RMS per-element) and 7 (Q8_1, passes at
98% agreement — the exact-mode class); no FP8 format has more than 3. A
secondary (unmeasured, moot after the quality fail) speed premise was also
doubtful: e4m3 weights are 1.78x the Q4K bytes (weight traffic DOUBLES vs
the MXFP4 path's 4.58 bits/elem) and the m16n8k32 mma rate equals int8's —
the fp8 path would have to win on parse-ALU alone. The simulation-first rule
saved the kernel work. (5) WHAT LANDED anyway (the harness is the durable
output): kernels/src/fp8_sim.rs (Fp8SimMatMulQ + MatMulQDispatch +
hardware-faithful e4m3 encode/decode helpers + 6 unit tests incl. a
random-data kernel-level gate vs CpuMatMulQ4KOpt with a
cancellation-aware 0.07*sum|terms| bound and realistic Q4K block
magnitudes); model.rs's three `let q4k = CpuMatMulQ4KOpt` bindings ->
matmulq_dispatch(); tools/fp8-sim (MMLU agreement + logit-band harness);
f16_to_f32 made pub(crate). Default-path delta = one atomic load + one-time
env parse per forward (mode 0 delegates verbatim — numerics-identical by
construction). Fp8SimMatMulQ carries a VerifiedEquivalent impl (bound 2^-4
per operand, tier Tested, carried by the runtime comparison) but is
deliberately NOT in the kernel-verify registry (that gate enforces <= 5%
for OPTIMIZED kernels; this is an intentional-approximation simulation).
OPERATIONAL NOTE: the harness caches converted weights as f32 (~32 GB for
the 8B model) — NEVER run multiple fp8-sim instances concurrently (four
parallel runs OOM'd on the 125 GB fishtank; run sequentially). (6) GATES ALL
GREEN on the final tree: local cargo test (workspace, bench-gpu excluded —
no CUDA toolkit on the laptop) 925 passed / 0 failed (919 + 6 new); GPU-side
battery on fishtank (my changes are CPU-forward-only, but the standing gates
ran): probe-q4k-batch PASS; probe-q6k-batch PASS; bench-ab-repro checksums
IDENTICAL; 6-client kv-swap A/B PASS on f16 AND e4m3 (byte-identical,
eviction exercised); nonstream-route-smoke PASS; kv-restart-var PASS; pooled
ppl 8 chunks f16 8.6838 / e4m3 8.6282 — EXACTLY the standing baselines,
0 NaN; sanitizer-gate PASS. (7) FOLLOW-UP RECORDED for any future
mid-precision attempt: the remaining design space is (a) activation formats
with >= ~5 mantissa bits (Q8_1-class — that IS the exact mode's mma128
route, so "mid-precision" collapses into it), or (b) MXFP4-with-wider-scales
on the WEIGHT side only with Q8_1 activations — but that pairing has no
single-mma operand format either; the x8-vs-generic 0.021 route band (2b
above) is a separate small pre-existing observation worth knowing when
comparing CPU paths.**

**STATUS 2026-11-17d (exact-mode prefill ACCELERATED 7.1x: two NEW LARGE-m
INT8 tensor-core kernels -- q4k_mma128_batch_sm120 + q6k_mma128_batch_sm120
-- route both TEMPURA_PREFILL_PRECISION=exact prefill sides with the weight
matrix read ONCE per 128-row launch; exact mode 355 -> 2508 tok/s pp128
(TTFT 360 -> 51 ms, 2.9x from llama.cpp's 7335), MMLU 62.8% / GPU-CPU
agreement 98.0% (PASS, was 96.8% -- the Q6K-side Q8_1 activation class
change is quality-NEUTRAL), exact ppl 8.0869 (old exact baseline 8.0889,
0.02% = noise; default-path ppl EXACTLY at baselines); all gates green):
(1) THE MEASUREMENT LADDER (fishtank, RTX 5090, Llama-3.1-8B-Instruct
Q4_K_M, probe-prefill-tc + bench-real-model pp128, exact mode): the
2026-11-17c exact route cost 355 tok/s because BOTH sides re-read the
weight matrix once per 16-row chunk at prefill m (Q4K kpar8 batch ~90-128
GB/s effective; Q6K sm120 batch ~95-106 GB/s). nsys on the chunked-mma
route (876 tok/s): Q4K mma 68.5 ms/step + Q6K sm120_batch 71.8 ms/step +
~4 ms misc. The FIXED tensor-core kernels (2026-11-16f) at m=16 measured
3.9-4.5x kpar8 (350-446 GB/s) -- routing exact prefill through chunked
mma gave 876 tok/s. The structural fix: the m16n8k32 tile is the only
16-row constraint -- an IN-KERNEL m-tile loop reads weights once for all
128 rows. (2) THE KERNELS (kernels/src/q4k_mma128_batch_sm120_ptx.rs,
q6k_mma128_batch_sm120_ptx.rs): 8 STATICALLY-UNROLLED m16 tile bodies
with REGISTER accumulators (f100-f131), per-(kb,mw) weight/header loads
hoisted out of the tile loop, per-tile smem reduction reusing the
q4k_mma_batch_sm120 warp geometry (grid ceil(n/8), block 256, stripe
(kb*8+gw)%8, TRUE fragment mapping). The Q4K variant's per-row numerics
are BITWISE-EQUAL to q4k_mma_batch_sm120 by construction (same
accumulation order) -- probe-verified. The Q6K variant handles Q6K's
PER-16-ELEMENT scales with a 2-MMA K16 SPLIT: B zeroed on one k16 half
per mma, so each D is ONE sub-block's exact s32 sum, scaled by its own
(d*sc) in f32 (no min term; weight byte = raw6-32 as exact s8). Q6K
mma128 takes Q8_1 ACTIVATIONS (llama.cpp's Q6_K x Q8_K class) -- NOT the
f32-activation class of the q6k sm120 batch GEMV; the quantize goes
through the q4k oxide module's quantize_q8_1_window (now public; the
quantize kernel lives in the q4k PTX). ptxas: 64 regs, 0 spills, 33152 B
static smem. DEBUGGING NOTES (both kernels probe-gated; the q6k one took
three fixes found by a one-hot-activation B-fragment dump): the B weight
u32 base forgot the x53 block stride (every thread read neuron 0 at
kb=0); and the CPU f64 probe reference had TWO bugs of its own -- h/qq/l
decoded from the GLOBAL element index instead of the within-block index
(reads past the block for kb>=1), and a naive rel-to-result gate that
false-positives on cancellative sums (f32 rounding is ~1e-7 x
sum|terms| but can exceed 1e-5 rel-to-result) -- the q6k probe's mma128
arm now gates |got-ref| <= 1e-5*sum|terms| + 1e-6 (measured headroom:
max err/gate ~5e-3; layout bugs misplace whole sub-blocks, >= 1e-3 x
abssum). (3) THE NUMBERS: exact mode 2508 tok/s pp128 / TTFT P50 51 ms
(ladder: 355 kpar8 -> 876 chunked mma -> 1086 q4k-mma128 -> 2508 both
sides; llama.cpp 7335 = 2.9x gap, was 20.6x); decode UNCHANGED (217
tok/s); logits NaN-free, det=OK. MMLU (250q, exact mode): GPU 62.8%,
CPU 61.6%, llama 62.4%; GPU/CPU agreement 98.0% PASS (gate 85%), GPU/llama
95.6% -- the Q6K Q8_1-activation class change did not move accuracy
(62.8% both before and after) and IMPROVED agreement. PPL (pooled 8
chunks): default f16 8.6838 / e4m3 8.6282 -- EXACTLY the pre-change
baselines (the default path is numerics-neutral: the only default-path
delta is env-var reads); exact f16 8.0869 vs the 2026-11-17c exact
baseline 8.0889 (ratio 0.998 vs llama.cpp 8.0192) -- the class change is
within noise, no re-baseline needed. (4) THE DEFAULT-FLIP DECISION:
exact at 2508 tok/s is still 3.1x below the MXFP4 default's 7790 -- the
~2x default-flip bar is NOT met, so the default stays MXFP4 (the
all-or-nothing choice is now 2508 vs 7790, and the 2026-11-17c follow-up
(d) -- a mid-precision prefill keeping most of the quality at serving
speed -- remains the interesting direction). Kernel selection under
exact mode: TEMPURA_Q4K_PREFILL_KERNEL=kpar8|mma|streamk (default mma,
which uses mma128 for m>16 when loaded), TEMPURA_Q6K_PREFILL_KERNEL=sm120
(default mma128); both bisect switches restore the 2026-11-17c routes.
(5) KERNEL-SHAPE NUMBERS (probe-prefill-tc, final tree): mma128 at
m=128 -- q/o_proj 0.214 ms, k_proj 0.055 ms, gate/up 0.330 ms, down-shape
0.395 ms -- vs the chunked mma's 0.597/0.135/0.597(x2)/0.713: ~1.8-2.5x
per shape at m=128 and ~5.2x on gate/up (weights once vs 8x). Projected
Q4K-projection stack: ~4241 tok/s pp128 (kpar8 547, chunked mma 2270).
(6) GATES ALL GREEN on the final tree: probe-q4k-batch PASS (mma128 arm:
<= 1e-5 vs kpar per-row AND bitwise vs chunked mma16, all shapes x m in
{1,2,4,6,16,64,128}); probe-q6k-batch PASS (mma128 arm: 4 shapes x 7 m
values, max err/gate ~5e-3); bench-ab-repro solo/repeat/interleave/
composition checksums IDENTICAL; 6-client kv-swap A/B PASS on f16 AND
e4m3 (byte-identical, eviction exercised); nonstream-route-smoke PASS;
kv-restart-var PASS; sanitizer-gate PASS (memcheck/racecheck/initcheck
clean on both probes incl. the mma128 arms, only the documented kpar8_db
handoff allowlist); local cargo test green (919).**
FOLLOW-UPS (open): (a) the mma128 kernels' remaining gap to llama.cpp is
compute-side (the per-tile f32 scaling chains + Q4K rowsum butterflies
run per (kb,gw,tile)); a quantize-side precomputed per-32-byte rowsum
(stash in the x_ds high bits, ignored by existing kernels) would cut the
Q4K min-term ALU;
      [RESOLVED 2026-11-18b: the rowsum stash landed (llama.cpp's Q8_1
      ds layout, bit-identical, exact mode 2508 -> 2732 tok/s); the
      REMAINING gap was profiled and analyzed against llama.cpp's mmq --
      see the newer status entry above for the structural analysis and
      the refuted wide-neuron hypothesis.] (b) the mid-precision prefill (2026-11-17c follow-up
d) is now measured from both ends: 2508 exact vs 7790 MXFP4; (c) the
mma128 kernels are candidates for the equal-position graph paths too
(uniform routing would be needed -- see the 2026-11-17 entry's
composition-invariance note); not measured here.

## Latest session status

**STATUS 2026-11-17c (the MMLU precision gap CLOSED: TEMPURA_PREFILL_PRECISION
=exact quality mode landed -- the MXFP4 prefill opt-out the 2026-11-13
measurement called for; MMLU GPU 48.8% -> 62.8%, GPU/CPU agreement 58.0% ->
96.8% PASS; the old TEMPURA_NO_MXFP4 capture-panic is CONFIRMED GONE;
default stays MXFP4 at 7790 tok/s pp128 -- the exact path costs 355, a
documented 20.6x opt-in trade; all default-path gates green, ppl EXACTLY at
baseline): (1) THE RE-MEASURED BASELINE on current main (fishtank, RTX
5090, Llama-3.1-8B-Instruct Q4_K_M, 250 questions, 0-shot, bench-mmlu):
tempura CPU 61.6%, llama.cpp 62.4% (agreement 94.8%), tempura GPU 48.8%
(agreement vs CPU 58.0%) -- the 2026-11-13 gap (51.2/58.4) PERSISTS
through all the serving-path rewrites (slightly worse GPU; the f16/e4m3
pools, no-graph exact decode, and batched RoPE did not touch the prefill
projections, as expected). NOTE: the CPU reference itself flips ~2/250
choices between runs (61.6% vs 62.4% across two runs of the same CPU code
path -- near-tie choices under threaded CPU reduction order; llama
agreement 94.8% both times), so single-run CPU accuracy carries a ~±0.8pt
band. (2) THE OLD BLOCKER IS GONE: TEMPURA_NO_MXFP4=1 no longer panics --
the follow-up (6) capture panic required the prefill CUDA graph, which has
been off by default since the cuda-oxide v0.5.0 retirement-thread fix
(TEMPURA_PREFILL_GRAPH=1 opts back in). The switch now runs cleanly -- on
the 9 tok/s host-dequant prefill fallback (per-call weight download +
dequant + re-upload), unusable as a quality mode. (3) THE MODE --
TEMPURA_PREFILL_PRECISION=exact (engine/src/cutile_gpu_forward.rs):
gemm_mxfp4_prefill_ep returns None at the top (every dispatch site,
including the fused-epilogue call sites, falls back), and Q4K prefill
projections route through the NEW gemm_q4k_prefill_batch -- the
bit-identical-per-row q4k_gemv_sm120_kpar8_batch serving kernel
(probe-gated), chunked over 16-row windows, reusing the nograph Q8_1
scratch pool -- instead of the per-call host-dequant f32 GEMM. Q6K keeps
the exact sm120 batch GEMV; layout-incompatible activations (h_pad != h
models) keep the host-dequant fallback. Default (unset/"mxfp4") is
numerics-UNCHANGED: the only default-path delta is an env-var read.
MEASURED (bench-mmlu, exact mode): tempura GPU 62.8%, CPU 62.4%, llama
62.4%; GPU/CPU agreement 96.8%, GPU/llama 95.6% -- PASS at the 85% gate
(was 58.0% FAIL). The GPU phase is 89 s for 250 questions (vs 21 s mxfp4).
(4) THE TRADE-OFF (bench-real-model, pp128): default 7790 tok/s prefill /
TTFT P50 16.4 ms; exact 355 tok/s / TTFT P50 360 ms; decode UNCHANGED
(218/217 tok/s -- decode never touches the prefill GEMMs). 20.6x vs
llama.cpp's 7335 pp128 -- FAR outside the ~2x default-flip bar, so exact
stays an OPT-IN for quality-sensitive workloads (eval harnesses, log-prob
scoring); serving keeps MXFP4 prefill. The higher-precision-tiling stretch
goal (wider block scales / FP8 activations) was NOT needed: the opt-out
fully closes the measured gap. (5) GATES ALL GREEN on the final tree
(default path; commit 855ae90 + docs): probe-q4k-batch PASS;
probe-q6k-batch PASS; bench-decode-ops 97 PASS / 0 FAIL; bench-ab-repro
solo/repeat/interleave/m=2/m=3 checksums IDENTICAL; 6-client kv-swap A/B
PASS on f16 AND e4m3; nonstream-route-smoke PASS; kv-restart-var PASS;
pooled ppl 8 chunks f16 8.6838 / e4m3 8.6282 -- EXACTLY the pre-change
baselines, 0 NaN (the default path is numerics-neutral by construction,
doubly confirmed); sanitizer-gate PASS; local cargo test green (919).
Exact-mode pooled f16 ppl (documentation, opt-in path): 8.0889 vs
llama.cpp 8.0192 = ratio 0.9980, 0 NaN -- the MXFP4 prefill band was
nearly the ENTIRE pooled-ppl gap to llama.cpp (default f16 8.6838,
ratio 1.0714; e4m3 8.6282, ratio 1.0646): exact-mode GPU ppl is
essentially at llama.cpp parity.
FOLLOW-UPS (open): (a) if exact-mode prefill throughput ever matters, the
mma batch kernel (probe-gated 1e-5, sum-order-only divergence) is an
exact-class tensor-core route for large-m prefill -- the 355 tok/s kpar8
route re-reads the weight matrix once per 16-row chunk; (b) the CPU
eval-reference run-to-run flip band (±0.8pt on 250 questions) is worth a
determinism pass if MMLU ever becomes a release gate; (c) h_pad != h
models (Qwen) get the host-dequant exact fallback, not the kpar8 batch
route (compact-activation guard) -- fine for correctness, slow if anyone
runs exact-mode evals on them; (d) the exact-mode ppl parity (0.9980 vs
llama.cpp) re-opens the case for a MID-precision prefill (FP8
activations or wider MXFP4 block scales) that keeps most of the quality
gain at a serving-usable speed -- the all-or-nothing choice is 355 vs
7790 tok/s.**

**STATUS 2026-11-17b (the kpar8 OOB-read debt PAID: compute-sanitizer
root-caused and fixed THREE pre-existing out-of-bounds sites in the default
serving path, all numerics-neutral -- probes bit-exact, ppl EXACTLY at
baseline; a permanent sanitizer gate now locks it; all gates green): (1)
THE HUNT -- compute-sanitizer memcheck on bench-probe-q4k-batch reproduced
the follow-up from the tensor-core session: 2-byte OOB reads in
q4k_gemv_sm120_dp4a_kpar8_dual_block, warp 7 only, 1/17 bytes past the
1792-byte act_ds scratch (k=14336, 448 u32 slots). SASS walk (ptxas +
cuobjdump on the embedded PTX) pinned the instruction: the B-side
ragged-tail clamp. The PTX clamped invalid blkB's activation index to
A's FULL q8_blk_lo (%r26 = blkA*8 + mw*2) and then added mw*2 AGAIN ->
B indexed blkA*8 + mw*4; at the last k-block (blkA=55, k_blocks=56) that
is slots 448/452 of 448 -- past the end. The same wrong index fed the
B-side act_qs loads (up to ~1.1 KB past, invisible to memcheck only
because they landed inside a larger mempool allocation). The loaded
values are discarded by the @%p10-predicated accumulate -- which is why
every output gate (bit-identity, ppl) passed for weeks while the reads
were live: the exact "benign OOB is NOT benign" hazard (under memcheck
the read faulted with error 719 -- the allocation-boundary class). FIX:
clamp to blkA*8 and add mw*2 once (kernels/src/q4k_gemv_v4_ptx.rs).
q6k is structurally immune: kpar8_db/db_batch BRANCH-SKIP block B past
k_blocks (no clamped loads) -- probe-q6k-batch was memcheck-clean from
the start. (2) TWO MORE SITES the first fault had been masking --
memcheck on bench-decode-ops (previously unreachable: the probe died at
the dual_block 719 before the later sections): (a) an OOB WRITE --
q4k_fused_silu_sm120_dp4a writing out[] past a 16 KB tensor: the host
scratch reuse in forward_sm120_fused_silu_d2d (and its oxide twin, plus
the three q8_out_scratch GEMV paths) never checked that the pre-allocated
out scratch fits n; bench-decode-ops pre-allocates for q_proj sizes then
calls the fused path with n=intermediate=14336. The ENGINE sizes
correctly today (max_n=inter) -- the fix (oxide_q4k_gemv_host.rs) makes
the invariant enforced at every reuse site instead of assumed. (b) a
harness bug: the q6k determinism-churn loop fed a 7777-element junk
activation to forward_kpar_d2d with kb=56 (reads kb*256=14336) --
sized to kb*256. (3) THE GATE -- scripts/sanitizer-gate.sh (fishtank,
repo root): memcheck on bench-probe-q4k-batch + bench-probe-q6k-batch +
bench-decode-ops, racecheck + initcheck on the probes (SAN_GATE_FAST=1
skips those); FAILS on any sanitizer error OR app-level FAIL. One
documented allowlist: racecheck reports the q6k kpar8_db/db_batch
cross-warp reduction handoff (st.shared partials -> red.release.cta
counter -> ld.acquire.cta poll -> thread-0 reads) as Write/Read hazards
-- racecheck cannot model scoped-atomics ordering; that pattern IS the
2026-10-29 fix for the real race and is determinism-gated (0/8 stress
runs). Hazards in any other kernel fail the gate. (4) GATES ALL GREEN
on the final tree (5702d9f): sanitizer-gate PASS (memcheck 0 errors on
all three bins, racecheck 0 hazards q4k / handoff-only q6k, initcheck 0
errors); probe-q4k-batch PASS; probe-q6k-batch PASS; bench-decode-ops
97 PASS / 0 FAIL (also under memcheck); bench-ab-repro
solo/repeat/interleave/m=2/m=3 checksums IDENTICAL; 6-client kv-swap
A/B PASS on f16 (6/6 byte-identical, 5 swaps) AND e4m3 (6/6, 4 swaps),
m1/m2p graph captures evidenced for both dtypes; pooled ppl 8 chunks
f16 8.6838 / e4m3 8.6282 -- EXACTLY the pre-change baselines, 0 NaN
(both doubly proving numerics-neutrality of the dual_block fix); local
cargo test green (919). NOTE: the e4m3 ppl's CPU reference phase takes
~30 min (2040 positions through the CPU model) -- the first e4m3
attempt was killed as "hung" at 29 min, it was not (GPU part: 16.7 s);
let it run. FOLLOW-UPS (open): the racecheck allowlist is
kernel+offset-specific (q6k_gemv_sm120_kpar8_db+0x17a0,
db_batch+0x36c0) -- if those kernels are edited, re-audit the hazards
against the new SASS, do not blindly extend the list; sanitizer-gate
does not yet cover the oxide-decode-attn crates' self-tests or
bench-nograph (candidates for the next extension).**

**STATUS 2026-11-17 (the retracted Q4K tensor-core batch kernels are
FIXED -- all three root causes found, probe-gated at 1e-5, measured
3e-8 sum-order noise; the serving-path routing EVALUATED end to end:
uniform mma is composition-invariant and ppl-clean and wins +30-34% at
m=4-6 but costs 47-88% at m=1, so the DEFAULT stays the bit-identical
kpar8 batch -- the selector TEMPURA_Q4K_BATCH_KERNEL=mma|streamk lands
as the documented opt-in; all gates green, ppl EXACTLY at baseline on
the default path): (1) THE FIX -- q4k_mma_batch_sm120 and
q4k_streamk_batch_sm120 (kernels/src/*_ptx.rs) had THREE stacked bugs,
found by iterating bench-probe-q4k-batch (now a HARD gate: every batch
kernel must match the per-row kpar8 reference at <= 1e-5 max rel err
on random native-GGUF blocks; launch-failure of any kernel is also a
FAIL -- a silently-unloaded kernel is how the original slipped): (a)
FRAGMENT MAPPING -- both kernels decomposed threads as row=lane%16 /
khalf=lane/16 (A), neuron=lane%8 / kgroup=lane/8 (B), row=lane%16 /
colgroup=lane/16 (D); NONE of that matches the real m16n8k32 layout.
The docstring claimed nvcc confirmation -- it was not (the mxf4-gemv
fragment lesson, again: uniform data and NaN checks cannot catch it).
Rewrote the compute core to the TRUE layout (tid=lane&3,
gid=lane>>2): A a0..a3 = rows {gid,gid+8} x k-halves {tid*4, 16+tid*4};
B b0/b1 = col gid, same k-halves; D d_j = (gid + 8*(j>>1),
2*tid + (j&1)) -- the mxf4 acc_coords mapping. The per-row min term is
now a 2-step XOR butterfly over the 4-lane group with per-row d8
scales. (b) Q4K NIBBLE LAYOUT -- byte group mw%4 with hi-select mw>=4
is wrong; sub-blocks 2j/2j+1 share qs[32j..32j+31] (lo/hi nibbles) --
byte group mw>>1, hi-select on ODD mw. (c) DEQUANT -- the kernels
applied nibble-8 and sc/32; the kpar8 convention is the RAW nibble
0..15 and RAW 6-bit sc/mv. Additionally: q4k_mma_batch_sm120 was
converted from 4 warps (%4 stripe) to the streamk-proven 8-warp
geometry (%8 stripe, 8-partial reduction, block 256) -- with 4 warps
it failed DETERMINISTICALLY on odd D columns only (probe dump: every
col 2k+1 wrong, every col 2k right) while byte-identical per-thread
code passed in streamk; the mw-skip divergence pattern of the %4
stripe is the identified difference (microarchitectural mechanism not
fully root-caused -- documented, not hand-waved). A bar.sync at
MW_DONE now protects the per-kb sync header reload. PROBE RESULT:
both kernels PASS at 2.8e-9..7.5e-8 max rel err across m in
{1,2,4,6} x shapes (64,1),(64,2),(4096,16),(1024,16),(14336,16),
(4096,56) -- pure f32 sum-order noise (mma.sync int8 products are
exact). has_mma_batch/has_streamk_batch re-enabled (uses_raw_gguf-
gated): the equal-position f32 batch graph paths can dispatch them
again (composition is fixed per capture -- the q6k two-phase
precedent). (2) THE ROUTING EVALUATION -- TEMPURA_Q4K_BATCH_KERNEL
selects the no-graph batch kernel (unset = kpar8, today's path).
m>=2: gemm_any_d2d_batch dispatches by selector. m=1:
gemm_any_d2d_slot's Q4K arm routes through the batch kernel too (the
kernels are row-independent -> m=1 numerics == m>=2 numerics), with
the (k,1)/(slot,n,1) scratch PRE-WARMED in preallocate_scratch
(allocation during the m=1 graph capture is error 900); the KV-stream
fork is skipped under the selector (its K/V submissions are per-row
kpar on stream_kv -- before that skip, ab-repro showed m=1 vs m>=2
diverging at step 1 by 2.1e-2 logits; after it, solo/m=2/m=3
checksums are IDENTICAL). MEASURED (fishtank, RTX 5090,
Llama-3.1-8B Q4_K_M, f16 pool, bench-nograph, before = same tree
default): short ctx m=1 5.2->9.8, m=2 8.7->10.6, m=3 10.8->11.2,
m=4 13.1->12.0, m=5 15.9->13.1, m=6 19.2->14.3 ms (313->419 tok/s,
+34%); long ctx m=1 7.6->11.2, m=2 11.0->12.8, m=3 12.9->13.4,
m=4 15.2->14.3, m=5 17.9->15.2, m=6 21.2->16.4 ms (283->367, +30%).
stream-K at serving m: 23-27 ms/step -- REJECTED (SM_COUNT-persistent
grid + sequential per-block n8 tiles target large batches). PPL under
uniform mma: pooled 8 chunks f16 8.6765 vs 8.6838 baseline (ratio
0.9992, 0 NaN) -- PASSES the <= 1.005 gate. VERDICT: default stays
kpar8. Composition invariance forbids m-thresholded dispatch (m=1
solo must stay bit-identical to the same session batched), so
switching means ALL m -- and uniform mma costs 191->102 tok/s at
m=1 short-ctx to buy +8-34% at m=4-6. Not taken; the selector is the
documented opt-in for m>=4-pinned profiles. (3) GATES ALL GREEN on
the final tree (default path -- selector off; the ppl numbers are
EXACTLY the pre-change baselines, confirming zero default-path
numerics change): probe-q4k-batch PASS (all three kernels);
probe-q6k-batch PASS; bench-decode-ops 97 PASS lines / 0 FAIL;
bench-ab-repro solo/repeat/interleave/m=2/m=3 checksums IDENTICAL;
pooled ppl 8 chunks f16 8.6838 / e4m3 8.6282 (baselines exact, 0
NaN); nonstream-route-smoke PASS; kv-restart-var PASS; 6-client
kv-swap A/B PASS on f16 AND e4m3 (6/6 byte-identical each, 4 swap
cycles in run B); local cargo test green (919). FOLLOW-UPS (open):
(a) compute-sanitizer on the probe flagged PRE-EXISTING 2-byte OOB
reads in q4k_gemv_sm120_dp4a_kpar8_dual_block (the per-row reference
kernel, threads reading past a buffer edge -- deterministic and
bit-consistent with the batch twin, but OOB is OOB; root-cause and
fix);
      [RESOLVED 2026-11-17b: root-caused (B-side ragged-tail clamp
      double-added mw*2), fixed numerics-neutral, plus two more masked
      sites (fused-silu out-scratch reuse, bench churn buffer) and a
      permanent sanitizer gate -- see the newer status entry above.]
      (b) the mma kernel's small-m cost is the m16-tile floor (it
computes 16 rows of A regardless) -- a small-m-specialized
tensor-core variant that is numerics-compatible across m (same sum
order, fewer wasted tiles) could flip the routing verdict; (c) the
stream-K kernel needs a grid/tile rework before it is useful at
serving m (or re-target it at prefill M>=32, its design point).**

**STATUS 2026-11-16e (BATCHED RoPE for the serving decode path -- the
per-request rope chain is GONE: one cutile_rope_rows_batch launch per
tensor per layer replaces m x 2 rope launches + 3m extraction slices +
2m copy-outs + m appends + m q_batch concat copies, in BOTH the generic
no-graph path AND the m>=2 ragged graph body (shared per-m buffers,
op-for-op identical); m=6 step 21.4 -> 18.9 ms short-ctx / 23.9 -> 21.2
ms long-ctx = 280 -> 317 / 251 -> 283 tok/s (+13%); m=1 UNCHANGED; all
gates green, ppl EXACTLY at baseline): (1) THE KERNEL --
`cutile_rope_rows_batch` (kernels/src/cutile_rope.rs, cuTile DSL): grid
2*m*n_heads programs, the cutile_rope_interleaved layout with the
(seq,head) pair drawn from a batch row.  Each program reads its row's
Q/K elements DIRECTLY from the strided projection output (row r at
tile r*SRC_TILES, SRC_TILES = q_pad|kv_pad / half -- no extraction
copies), rotates with the per-row GATHERED cos/sin table rows
([m*half] buffers refreshed per step by d2d_gather_row reading a
device [m] positions array -- the graph-stable convention), and stores
its tile at pid in the contiguous interleaved output.  The math is the
SAME tile expressions (`x0*cos - x1*sin` / `x0*sin + x1*cos`) the
per-request cutile_rope_from_table path uses, on the SAME table rows
-- bit-identical per element by construction, and probe-gated (new
bench-decode-ops section "batched rope rows": m=4, DISTINCT per-row
positions, adversarial stride row_dim+half with NaN garbage in the pad
tails, vs the per-request table rope as ground truth: 0 mismatched
elements on both the Q (32-head) and K (8-head GQA) shapes; also
exercises the reclaimed-z second call).  Host wrapper
`CutileRoPE::forward_rows_batch` uses the try_unwrap reclaim pattern
(stable buffer address per m -- CUDA-graph capturable).  Power-of-two
halves only; the route gates on tile_geometry(head_dim).is_none() and
stride % half == 0 (Llama/Qwen head_dim 128 fine).  (2) THE WIRING --
a per-m RopeBatchState (ng.rope_batch: positions array, write-offset
array, cos/sin row buffers, reclaimed q_z [m*q_dim] / k_z [m*kv_dim]
rope outputs, v_batch [m*kv_dim] extraction staging, [m] u64 scatter
src-ptr arrays, host staging) SHARED by both routes so the captured
graph body and the generic path run the same kernels on the same
buffers.  Per step: one H2D refresh of positions (f32-bitcast) /
per-row write offsets (each session's kv_len) / scatter src ptrs, plus
2m gather launches (host-launched on the generic route, captured
IN-graph on the ragged route -- replays re-gather at execution time).
Per layer: 2 rope launches (Q -> q_z which IS the attention Q input,
K -> k_z), m V D2D rows into v_batch, and the KV append through the
per-row-offset scatter twins (kv_scatter_write_rows{,_f16,_e4m3} --
the same cast bit-math as the per-request append's cast + D2D,
probe-gated; replaces the m append_device_gpu_only calls).  The K-side
scatter fusion (rope storing straight into the pool at
base + row_offset*kv_dim) was NOT forced: keeping the scatter
separate keeps the cast bit-math provably the twins' verbatim and the
code clean.  The m=1 graph route keeps its own path (numbers must not
regress -- they don't).  TEMPURA_NO_BATCH_ROPE=1 bisects back to the
per-request chain.  RaggedGraphState shed its rope fields (the body
now uses the shared ng.rope_batch state).  (3) ROUTE GATE (computed
per step, model-constant except positions): m >= 2, batch attention
twins present for the uniform pool dtype, all-device KV, positions in
table range, pow2 half, strides divisible -- anything else keeps the
old chain, and the m>=2 graph route additionally requires it so the
captured body never diverges from the generic path.  (4) NUMBERS
(fishtank, RTX 5090, Llama-3.1-8B Q4_K_M, f16 pool, bench-nograph
graph-routed, before = same tree with TEMPURA_NO_BATCH_ROPE=1):
short ctx (48 steps): m=1 5.2 -> 5.2 (UNCHANGED), m=2 9.7 -> 8.6,
m=3 12.2 -> 10.7, m=4 14.8 -> 13.0, m=5 18.1 -> 15.7, m=6 21.4 -> 18.9
ms = 280 -> 317 tok/s.  Long ctx (400 steps, kv ~440): m=1 7.6 -> 7.6,
m=2 11.9 -> 10.9, m=3 14.3 -> 12.9, m=4 17.0 -> 15.2, m=5 20.1 ->
18.1, m=6 23.9 -> 21.2 ms = 251 -> 283 tok/s.  ~2.5-2.7 ms/step at
m=6 -- the single biggest kernel-side win since the Q4K batch GEMV.
(5) THE UPDATED BREAKDOWN (TEMPURA_ENGINE_DEBUG_TICKS, 6-client f16
serving load): the step is now ~100% forward (prep < 1 us, greedy
sample+emit ~9 us, forward ~21 ms at m=6 long-ctx) and forward is
kernel-bound on the batch GEMV weight traffic (~3 GB Q4K + ~1.65 GB
Q6K + 431 MB lm_head per step) + the oxide batch attention -- the
rope/copy chain is no longer a visible cost.  The structural next
levers are GEMV/attention-side: fusing RMSNorm into the batch GEMVs,
a QKV-fused batch GEMV (one weight sweep for q+k+v), or attention
kernel improvements -- not launch overhead and not the rope.  (6)
GATES ALL GREEN on the final tree: new bench-decode-ops "batched rope
rows" PASS (bit-exact, both shapes + z-reclaim); bench-decode-ops
full suite PASS (97 PASS lines, incl. scatter-rows twins);
bench-probe-q4k-batch + bench-probe-q6k-batch PASS; bench-ab-repro
solo/repeat/interleave/composition checksums IDENTICAL under BOTH
TEMPURA_NO_M2P_GRAPH=1 and graph routing, and the full checksum
vector byte-identical between the two (graph-vs-generic); 6-client
kv-swap A/B PASS on f16 AND e4m3 (byte-identical per client, 4 swap
cycles in run B, [m2p_graph] captured m=2..6 on BOTH dtypes in every
serving log); pooled ppl 8 chunks f16 8.6838 / e4m3 8.628168 --
EXACTLY the pre-change baselines, 0 NaN; nonstream-route-smoke PASS;
kv-restart-var PASS; local cargo test green (919).  VALIDATION-PROCESS
NOTE: the first probe run's "z reclaim FAIL" was a PROBE bug (the
first output Arc was held alive across the second call, blocking
try_unwrap by design); the wrapper was right.  FOLLOW-UPS (open): the
V extraction is still m D2D copies per layer (small; a strided-copy
kernel or folding V into the rope launch would remove m launches per
layer); the batched rope is pow2-half-only (Phi-3.5 head_dim 96 keeps
the per-request chain and skips the m2p graph route); the m=1 route's
long-ctx step (7.6 ms) remains kpar-attention-bound.**

**STATUS 2026-11-16d (position-synced RAGGED batch CUDA graphs for m>=2
serving decode -- the composition-invariance blocker solved by making
every per-step value device-read; m=2..6 step 21.6 -> 21.3 ms short-ctx /
23.8 -> 23.5 ms long-ctx at m=6 -- a MODEST, honest win: the m>=2 step is
kernel-bound, the m=1 route's 2x came from being launch-bound; all gates
green, ppl EXACTLY at baseline): (1) THE DESIGN -- m>=2 batch steps with
ragged per-session positions on f16/e4m3 pools now route through
(m, dtype)-keyed CUDA graphs (forward_batch_decode_greedy's new arm ->
forward_batch_ragged_graph; TEMPURA_NO_M2P_GRAPH=1 bisects; f32 pools
keep the existing paths).  EVERY per-step value the captured graph
consumes is read from DEVICE memory at execution time: per-row positions
(the in-graph d2d_gather_row cos/sin row gathers read
d_positions_batch[r]), per-row KV write offsets (each session's kv_len),
per-row kv_lens, and the per-layer K/V run base pointers (one
[layers][m] H->D upload whose stride m matches the graphs' baked
layer*m*8 offsets) -- so the graphs are batch-COMPOSITION-INDEPENDENT:
any m sessions of the same pool dtype replay the same graph.  Session
churn, ragged positions, and pool growth NEVER re-capture and need NO
run-pointer validation (the base arrays are refreshed per step; the
equal-position graphs' baked-values assumption is untouched because they
are f32-only and this route is f16/e4m3-only).  (2) NEW KERNELS --
kv_scatter_write_rows{,_f16,_e4m3} (kernels/src/kv_scatter_write.rs):
grid (ceil(kv_dim/B), m) batch KV scatter with per-row src pointer /
cache base / write offset all read from device arrays; the cast bit-math
is the single-seq twins' verbatim.  Probe-gated in bench-decode-ops
("kv scatter write rows"): batch vs per-row twins BIT-EXACT at m=4 on
adversarial per-row data (RNE ties / saturation / NaN lanes landing on
different elements per row) with distinct per-row offsets -- f32/f16/
e4m3 all 0 mismatched elements.  (The first probe run failed f32 with 3
mismatches: a PROBE bug -- the source tensors dropped before the
kernels read them, cuda-oxide's deferred free recycled the memory;
keeping them alive fixed it.  The kernels were right.)  (3)
BIT-IDENTITY -- the captured body (forward_batch_ragged_graph_body)
mirrors the generic no-graph m>=2 path op-for-op (the SAME
gemm_any_d2d_batch m-loop batch GEMVs, the same per-(site,m) rmsnorm
scratch, the oxide batch attention, silu/add/pad helpers, the same
residual ping-pong) with exactly the three device-read substitutions
the m=1 route already proved bit-identical: table RoPE via gathered
rows, scatter twins for the append (same cast bit-math as the generic
cast + D2D), device-read scalars.  The first step per (m, dtype) runs
the body once FOR REAL (the step's compute + warm-up of every per-m
pool -- allocation inside capture is DriverError 900), syncs, then
captures the same op sequence (recorded, not executed; the warm-up
result is the step's output).  The generic tail's greedy-argmax/
download block was extracted verbatim into finish_batch_outputs and is
shared.  (4) NUMBERS (fishtank, RTX 5090, Llama-3.1-8B Q4_K_M, f16
pool, bench-nograph, before = same tree with TEMPURA_NO_M2P_GRAPH=1):
short ctx (48 steps): m=1 5.2 -> 5.2 (UNCHANGED -- own route), m=2
9.7 -> 9.4, m=3 12.2 -> 11.8, m=4 14.9 -> 14.4, m=5 18.0 -> 17.5, m=6
21.6 -> 21.3 ms = 278 -> 281 tok/s.  Long ctx (400 steps, kv ~440):
m=1 7.6 -> 7.6, m=2 11.9 -> 11.6, m=3 14.3 -> 13.9, m=4 16.9 -> 16.6,
m=5 20.2 -> 19.8, m=6 23.8 -> 23.5 ms = 252.5 -> 254.9 tok/s.  THE
HONEST BREAKDOWN: the win is ~0.3-0.5 ms/step (~2%), NOT the m=1
route's 2x -- because the m>=2 generic step is KERNEL-bound: the host
enqueues the ~1500 launches asynchronously and runs ahead of the GPU
(unlike m=1, whose step was launch-latency-bound with the host as the
bottleneck).  The graph removes the residual un-hidden host time only;
the step is now squarely bound by the batch GEMV weight traffic +
attention kernel time (the structural next levers are kernel-side:
fusing the per-row rope/copy-out chain, a batched rope, or attention/
GEMM improvements -- not launch overhead).  (5) GATES ALL GREEN on the
final tree (md5-verified code, [m2p_graph] capture evidence in every
serving log): 6-client kv-swap A/B PASS on f16 AND e4m3 (byte-identical
per client; 5/4 swap-out lines in run B; [m2p_graph] captured m=2..6 on
BOTH dtypes in BOTH runs -- the churn/cache-transition exercise);
bench-ab-repro solo/repeat/interleave/m=2/m=3 checksums IDENTICAL and --
the graph-vs-generic gate -- the FULL checksum vector is byte-identical
between a TEMPURA_NO_M2P_GRAPH=1 run and a graph-routed run; pooled ppl
8 chunks f16 8.6838 / e4m3 8.6282 EXACTLY the pre-change baselines, 0
NaN (m=1 route, doubly proving no m=1 perturbation); nonstream-route-
smoke PASS; kv-restart-var PASS; probe-q4k-batch + probe-q6k-batch
PASS; bench-decode-ops PASS (incl. the new rows probes); local cargo
test green (919).  VALIDATION-PROCESS NOTE: two early ppl runs (8.6790,
both dtypes identical) were run WITHOUT TEMPURA_PPL_POOLED=1 (the
wrapper-session path, dtype-independent -- not a regression) and on a
stale binary missing the oxide-decode-attn PTX in its build out dir;
the gate runs above were re-run on the rebuilt binary with the PTX
re-copied.  FOLLOW-UPS (open): the m>=2 step is kernel-bound -- the
next structural lever is a batched RoPE (today m x 2 rope launches + m
q_rot + m k_rot D->D copy-outs per layer inside the graph) and/or
fusing the row-slice/copy chain; the graph replay itself is not the
bottleneck.**
      [RESOLVED 2026-11-16e: the batched RoPE landed -- see the newer
      status entry above; m=6 21.4 -> 18.9 short / 23.9 -> 21.2 long.]

**STATUS 2026-11-16c (m=1 serving decode routed through the single-seq
CUDA graph -- the composition-invariance blocker solved with a
bit-identity-pinned capture; single-client decode 10.4 -> 5.2 ms/step
short-ctx (96 -> 191 tok/s), 9.0 -> 7.6 ms at 400-step contexts; m>=2
UNCHANGED; all gates green): (1) THE BLOCKER, SOLVED -- the engine never
routed m=1 ticks to the single-seq graph path because a session decoded
ALONE had to produce BIT-IDENTICAL logits to the same session inside a
batch.  The route (forward_batch_decode_greedy's m=1 arm ->
forward_decode_with_graph_pooled) captures a graph whose composition is
PINNED to the generic path's per-op kernels via a new
DecodeInnerMode::PooledM1 in forward_decode_inner: (a) attention = the
per-request KPRA twins on run base pointers dispatched by pool dtype
(NEVER splitk -- different reduce order; the m>=2 oxide batch kernel is
itself bit-identical to kpar, dae173b); (b) projections =
gemm_any_d2d_slot per-row GEMVs (the same function the m=1 arm of
gemm_any_d2d_batch dispatches; the multistream KV fork uses the same
kernels with separate scratch); (c) fused gate+up SiLU is Wrapper-only
(separate GEMVs + silu_mul in PooledM1); (d) KV append = NEW
kv_scatter_write_f16/_e4m3 dtype twins (same in-kernel cast bit-math as
the generic append's cutile_cast kernels -- the f16 twin is cvt.rn.f16.
f32, the e4m3 twin is a scalar PTX port of the shared
e4m3::f32_to_e4m3_bits; probe-gated in bench-decode-ops against cast
kernel + CPU references on adversarial RNE-tie/saturation/NaN data);
(e) RoPE = the graph-stable table gather (the generic path's seq_len==1
fast path reads the SAME precomputed table -- forward_device_scratch
falls into from_table at pos < table_max_pos, which the route gates
on).  Graphs are cached per RUN (key: layer-0 K base ptr + dtype tag),
validated against every layer's run base pointers each call (realloc/
swap re-captures), NOT keyed on capacity_gen (all per-step scalars are
device-read; extend-in-place growth replays).  Bisect:
TEMPURA_NO_M1_GRAPH=1.  (2) NUMBERS (fishtank, RTX 5090,
Llama-3.1-8B Q4_K_M, f16 pool, bench-nograph, before = same tree with
TEMPURA_NO_M1_GRAPH=1): 48-step ctx m=1 10.4 -> 5.2 ms = 95.9 -> 191.0
tok/s (2.0x); 400-step ctx (kv -> ~440) m=1 9.0 -> 7.6 ms = 111.0 ->
131.0; m=2..6 UNCHANGED at both context lengths (m=6 long 23.9 ->
23.8, short 21.6 -> 21.9 -- noise).  Short-ctx m=1 lands at the
wrapper graph's ~4.6 ms step; the long-ctx residual (7.6 vs the
wrapper's splitk-equipped 4.6) is the kpar-vs-splitk attention
difference + unfused gate/up SiLU -- the PRICE of bit-identity
(splitk's different reduce order and the fused kernel are
composition-forbidden).  (3) GATES ALL GREEN on the final tree, graph
capture evidenced by [m1_graph] lines in every serving log: 6-client
kv-swap A/B PASS on f16 AND e4m3 (6/6 byte-identical each, 4 swaps in
run B, multiple per-run graph captures); bench-ab-repro
solo/repeat/interleave/m=2/m=3 checksums all IDENTICAL (solo is m=1
throughout -- the graph-vs-batch bit-identity gate); pooled ppl 8
chunks f16 8.6838 / e4m3 8.6282 -- EXACTLY the pre-change baselines,
0 NaN (the ppl path decodes at m=1 through the new route, doubly
proving bit-identity); nonstream-route-smoke PASS; kv-restart-var
PASS; probe-q4k-batch + probe-q6k-batch PASS; bench-decode-ops
scatter twins PASS; local cargo test green (919).  VALIDATION-PROCESS
NOTE: the first A/B + ab-repro runs (16:43-17:01 remote) predated the
final engine edit and ran without the working route -- all gates were
RE-RUN on the md5-verified final tree; only runs with [m1_graph]
capture evidence count.  FOLLOW-UPS (open): the long-ctx m=1 step
(7.6 ms) is kpar-attention-bound -- a bit-identical splitk variant
(same reduce order, more blocks) or a graph-capturable position-
synced BATCH graph (m>=2 through graphs at all) is the structural
next lever; the pooled-graph cache clears wholesale at >32 entries
(re-capture is cheap but a smarter eviction is trivial if it ever
shows up in a profile).**

**STATUS 2026-11-16b (the Q6K projections join the bit-identical batch
party + the sampled-session sampler defused; engine step 26.8 -> 23.6 ms
at m=6 = 224 -> 254 tok/s aggregate (short-ctx 24.6 -> 21.6); sample+emit
for SAMPLED sessions 10.4 ms -> 0.87 ms; all gates green): (1) THE Q6K
BATCH KERNEL -- `q6k_gemv_sm120_kpar8_db_batch` (kernels/src/
q6k_gemv_v4_ptx.rs, own PTX module): the dual-block direct-load kpar8_db
warp structure (the per-row variant the no-graph path DISPATCHES to by
default via select_kpar_fn) VERBATIM with an m-loop over activation
rows, the exact q4k_gemv_sm120_kpar8_batch recipe. Each thread loads its
weight-block words once per K-loop iteration (dual blocks A/A+8,
back-to-back ld.global.nc, L2 prefetch, no smem/cp.async/barriers in the
loop) and computes all m rows' FMAs against them: the Q6K weights
(v_proj + down_proj + the quantized lm_head in Q4_K_M -- ~1.65 GB/layer-
set + 431 MB lm_head per STEP x m) are read ONCE per batch instead of
once per request. f5 = d*sc*(q6-32) is computed once per block-step
(row-independent) and reused across rows; per-row accumulators live in
local memory (lacc[64], m <= 16); the 5-step butterfly + the cross-warp
handoff (lane-0 partials, red.release counter, thread-0 SEQUENTIAL sum)
are db-verbatim, so each output row is BITWISE-EQUAL to a per-row db
launch. VALIDATION: new bench-probe-q6k-batch (random native-GGUF 53-u32
blocks -- random f32 d, random scale bytes, random ql/qh words -- vs
the per-row kpar dispatch as ground truth): bit-exact at m in {1,2,4,6}
on (1024,16) v_proj, (4096,16), (4096,56) down, AND (128256,16)
lm_head-shaped probes. FIRST-TRY CLEAN (no bring-up bugs -- the q4k
recipe transplanted). (2) THE EXISTING two-phase q6k_gemv_sm120_batch
(graph path) PROBED: NOT bit-identical to per-row (max rel err
6e-5..2.4e-4 -- its cross-warp reduce is a 3-step lane butterfly, not
the sequential thread-0 sum; a reduce-ORDER difference, not a layout
misparse -- numerically fine, just not composition-invariant). Verdict:
it STAYS on the equal-position graph paths (captured composition is
fixed per graph) and must NEVER be routed to a ragged path; documented
in kernels/AGENTS.md. (3) ENGINE WIRING -- gemm_any_d2d_batch's new Q6K
arm mirrors the Q4K arm: compact [m*k] activations (shared
q4k_batch_act pool, keyed (slot,k,m)), output in the gemm_out pool,
16-row chunking, m == 1 keeps the per-row kpar GEMV (probe-proven
bit-identity at m=1 -> m-dependent DISPATCH, no m-dependent NUMERICS).
The lm_head dispatch in the no-graph path goes through
gemm_any_d2d_batch (line ~10765), so the quantized lm_head picks the
batch kernel up for free; output handling is the existing [m*n] gemm_out
buffer + download_logits (the oversized-tensor invariant unchanged; the
greedy GPU-argmax path reads the same buffer). Bisect:
TEMPURA_NO_SM120_Q6K_BATCH=1 reverts to per-row GEMVs. NUMBERS
(fishtank, RTX 5090, Llama-3.1-8B Q4_K_M, f16 pool, bench-nograph,
before = main a78913d same day): 400-step contexts: m=1 9.0 -> 9.0 ms
(unchanged, per-row kept), m=2 12.6 -> 11.8, m=3 15.7 -> 14.3,
m=4 19.1 -> 17.0, m=5 22.7 -> 20.2, m=6 26.8 -> 23.6 ms =
223.9 -> 253.8 tok/s aggregate (+13%). Short ctx (48 steps): m=6
24.6 -> 21.6 ms = 244 -> 277 tok/s (+13%); m=1 10.5 -> 10.3
(unchanged). (4) THE SAMPLED-SESSION SAMPLER -- measured first
(scripts/measure-sample-cost.sh: 6-client temp=0.8 streaming load under
TEMPURA_ENGINE_DEBUG_TICKS): sample+emit p50 10.37 ms / p90 10.76 ms
per step at m=6 (298/300 ticks at m=6) -- the CpuSampler's FULL-VOCAB
SORT (128k x m, twice when top_p < 1.0) plus O(v) softmax/sample walks.
FIX (CPU-side, no GPU kernel): a fused decode-path sampler --
Sampler::sample_from_logits (new provided trait method, kernel-api) +
CpuSampler::sample_fused (kernels/src/sampling.rs): temperature (same
l/T division), min_p, top-k via a bounded MIN-HEAP (one streaming pass,
O(n log k), deterministic lower-index tie-break), softmax over the k
candidates in descending order, top-p, and the RNG walk over the kept
candidates in INDEX order (equivalent to the full path's whole-array
walk -- excluded entries are zero). Same r -> same token as the full
path (unit test: fused == apply+sample_token on random logits, 5 param
configs x 20 trials); greedy rows and typical_p < 1.0 fall back to the
full path. Kani: topk_heap_selection_in_bounds (heap indexing, no
OOB, sortedness). MEASURED AFTER: sample+emit p50 10.37 -> 0.87 ms
(p90 0.89) at m=6/temp=0.8 -- 12x, and UNDER the 2 ms build/skip bar.
(5) THE GPU-SIDE SAMPLER (option b: GPU temperature-softmax + top-k,
download top-k, CPU top-p + RNG): SKIPPED ON MEASUREMENT. The download
is NOT the cost (m x 513 KB at ~20 GB/s ~ 150 us, inside forward --
forward p50 23.7 ms unchanged by either sampler); the cost was the CPU
sort, now eliminated host-side. A GPU top-k kernel would save at most
the remaining ~0.7 ms of O(n) CPU passes (scale + NaN scan + heap
compare) minus kernel launch + its own costs -- under the 2 ms bar by
the task's own rounding-error criterion. Do not build it without a
load that shows sampled sessions dominating again. GATES ALL GREEN on
the final tree: probe-q4k-batch PASS; probe-q6k-batch PASS (db_batch
bit-identical at every shape x m); 6-client kv-swap A/B PASS on f16 AND
e4m3 (byte-identical, swaps exercised); pooled ppl 8 chunks f16 8.6838 /
e4m3 8.6282 -- BIT-IDENTICAL to the pre-change baselines (the batch
kernel is numerically exact vs per-row; the ppl path decodes at m=1 and
never dispatches it, doubly unchanged); nonstream-route-smoke PASS;
kv-restart-var PASS; bench-ab-repro solo/repeat/interleave/m=2/m=3
checksums all IDENTICAL; local cargo test green (919, +3 sampler
tests). FOLLOW-UPS (open): m=1 decode (~9 ms) is now the latency-bound
per-row GEMV floor (position-synced batch graphs are the structural
fix); the O(n) sampler passes could fuse into one pass (~0.4 ms more)
if ever worth it.**

**STATUS 2026-11-16 (no-graph serving decode: the Q4K projections now read
weights ONCE per batch instead of once per request -- engine step 31.1 ->
26.8 ms at m=6 = 193 -> 224 tok/s aggregate (short-ctx 208 -> 244); the
batch kernel is BIT-IDENTICAL per row to the per-row kpar8 GEMV, so all
numerics gates pass at ratio 1.0000; ALSO FOUND + DISABLED: the stream-K
and MMA batch kernels are numerically WRONG; all gates green): (1) THE
KERNEL -- `q4k_gemv_sm120_kpar8_batch` (new entry in
kernels/src/q4k_gemv_v4_ptx.rs, own PTX module): the validated kpar8 warp
structure VERBATIM with an m-loop over activation rows. Each thread loads
its weight-block data once per K-loop iteration and computes all m rows'
dp4a partials against it; per-row accumulators live in local memory (PTX
cannot index registers; the loads/stores do not alter arithmetic). Every
per-row operation -- dp4a inputs, term multiply/subtract order, the
per-block accumulate order, the K-loop block order (blk = warp_id, +8),
the 3-step XOR-bfly reduction (c=0x181F) and the cross-warp/final sums --
is identical to a per-row kpar8 launch, so each output row is
BITWISE-EQUAL to running the per-row GEMV on that row alone (the oxide
BATCH decode-attention approach, applied to GEMVs). Grid (n,1,1), block
(256,1,1), static smem 2048 B (16 rows x the kpar8 128 B slots), m <= 16
per launch. VALIDATION: new bench-probe-q4k-batch (random native-GGUF
blocks -- random 6-bit cross-byte-packed scales, random nibbles, per the
mxf4 lesson that uniform data cannot catch layout bugs -- vs the per-row
sm120 kpar8 as ground truth): bit-exact at m in {1,2,4,6} on all four
projection shapes (4096x4096, 1024x4096, 14336x4096, 4096x14336).
(Bring-up bug, fixed: the reduction's smem slot address double-multiplied
the row offset -- slot units vs bytes -- corrupting rows at m>=2 and
writing past the 2 KB buffer at m>=6, caught by the probe before any
engine wiring.) (2) THE ENGINE PATH -- gemm_any_d2d_batch dispatches Q4K
weights (q/k/o/gate/up in Q4_K_M) through
forward_sm120_kpar_batch_d2d_rows for every m >= 2, chunking m > 16 over
16-row windows (chunking cannot change per-row numerics). Compact [m*k]
activations via the cached nograph pool when row_stride != k; Q8_1
scratch keyed (k, min(m,16)); output in the existing gemm_out pool.
m == 1 KEEPS the per-row kpar GEMV: the probe verified bit-identity
between the batch kernel at m=1 and the per-row path, so this
m-dependent DISPATCH introduces no m-dependent NUMERICS (the one case
the no-m-dependent-dispatch rule allows) and saves ~7 us/projection of
batch-kernel overhead. Bisect: TEMPURA_NO_SM120_KPAR_BATCH=1 reverts to
per-row GEMVs. (3) THE BROKEN KERNELS -- the original plan (route the
no-graph path through the EXISTING oxide stream-K / MMA batch kernels,
which read weights once and are general-M) was implemented first and
REJECTED on evidence: the ppl gate exploded (pooled f16 1.89e6 vs 8.68
baseline, 0 NaN). Root cause via the probe: BOTH kernels misparse the
Q4K sub-block layout -- they pair weight nibbles with the wrong
activation elements (sub-block mw reads lo-nibbles of bytes 32*mw, but
the true GGUF layout -- confirmed by the kpar8 kernel AND the engine's
CPU dequant -- maps sub-block 2j to lo-nibbles of bytes 32j and sub-block
2j+1 to HI-nibbles of the SAME bytes) and use a wrong dequant convention
(raw nibble 0..15 with NO -8 offset and NO /32 scale factor). 12-31% rel
error vs the kpar8 reference on random data at every shape and m. They
were never numerically validated: only NaN/determinism checks ever ran
on them (the exact mxf4-gemv lesson), and on Q4_K_M they were dormant in
the graph path (all Q4K projections route to MXFP4). All three
has_sm120_batch/has_mma_batch/has_streamk_batch now return false with
pointers to the probe; all-Q4K models thereby fall back from the (wrong)
Q4K batch graph to the no-graph path -- which now has the correct fast
batch kernel. Follow-up recorded: fix or delete the streamk/mma PTX;
wire the kpar8 batch kernel into the graph path. The 2026-11-12
streamk graph-path batch numbers (M=1..16) and all pre-2026-11-12
mma_batch graph numbers were computed with these wrong kernels and are
RETRACTED as quality-unverified (they were NaN-free + deterministic --
internally consistent garbage).
      [RESOLVED 2026-11-17: both kernels FIXED (three stacked bugs --
      fragment mapping, nibble layout, dequant convention) and
      probe-gated at 1e-5; has_* re-enabled; routing evaluated, default
      stays kpar8 -- see the newer status entry above.] (4) GRAPH-PATH PPL RE-MEASURED CLEANLY
(the unexplained "7.5 vs 5.4" note from 2026-11-14): pooled f32 pool +
TEMPURA_PPL_M2=1 (m=2 equal-position dummy member -> MXFP4 batch graph)
8 chunks = 11.3059 vs the f32 no-graph baseline 8.6799 = ratio 1.3025 --
the MXFP4 activation-quantization approximation, ~stable across chunk
counts (was 1.39x at 1-2 chunks). Serving never uses this path (f16/e4m3
pools -> no-graph, exact). NUMBERS (fishtank, RTX 5090, Llama-3.1-8B
Q4_K_M, f16 pool, bench-nograph, before = main 3d3172b same day):
400-step contexts: m=1 9.0 -> 8.9 ms (unchanged, per-row kept), m=2
13.0 -> 12.6, m=3 17.4 -> 15.7, m=4 21.9 -> 19.1, m=5 26.5 -> 22.7,
m=6 31.1 -> 26.8 ms = 193.2 -> 223.9 tok/s aggregate (+16%). Short ctx
(48 steps): m=6 28.8 -> 24.6 ms = 208 -> 244 tok/s (+17%); m=1 10.5 ->
10.6 (unchanged). The weight-traffic model predicted more (m x
weight_read -> 1 x weight_read on ~3 GB of Q4K weights = ~15 GB saved at
m=6): the measured reality is that the per-row kpar8 GEMV is
latency-bound, not bandwidth-bound, so removing 5/6 of the weight READS
saves less than the raw bytes suggest (the batch kernel adds m x dp4a +
activation-load work per thread, serialized in the same latency chains).
The remaining no-graph costs: the Q6K projections (v/down + lm_head in
Q4_K_M) still run per-row GEMVs (~1.65 GB/layer-set + 431 MB lm_head per
STEP x m -- a bit-identical q6k batch kernel is the same recipe); the
m=1 step (~9 ms) is now dominated by the same latency-bound per-row
GEMVs. GATES ALL GREEN on the final tree: 6-client kv-swap A/B PASS on
f16 AND e4m3 (byte-identical, swaps exercised); pooled ppl 8 chunks f16
8.6838 / e4m3 8.6282 -- BIT-IDENTICAL to the pre-change baselines (ratio
1.0000, 0 NaN; the batch kernel is numerically exact vs per-row);
nonstream-route-smoke PASS; kv-restart-var PASS; bench-ab-repro
solo/repeat/interleave/m=2/m=3 checksums all IDENTICAL; local cargo test
green (916).**

**STATUS 2026-11-15b (the two remaining no-graph serving cost centers
closed: GPU greedy argmax for the batch sampler + the oxide BATCH
decode-attention kernel; engine step 47 -> 31 ms at m=6 = 192 tok/s
aggregate; all gates green): (1) THE SAMPLER -- sample_and_emit cost
11 ms of the 47 ms step at m=6: forward_batch_decode downloaded FULL
logits rows (m x 128256 f32 = ~513 KB/row) and CpuSampler::apply
SORTED the whole vocab on the CPU before the argmax.  FIX: sessions
that are pure greedy (temperature == 0, no presence/frequency
penalties -- new SamplingParams::is_greedy in kernel-api; top-k/top-p/
min-p never move the argmax) now run the EXISTING GPU greedy-argmax
kernel (greedy_argmax.rs, the u64 packed-key atomicMax whose tie-break
matches max_by -- forward_greedy's kernel) on the DEVICE logits row
before any download.  New InferenceModel::forward_batch_with_kvs_greedy
returns BatchDecodeOutput::Token(u32) for greedy rows / Logits(Vec) for
sampling rows; CutileGpuModel::forward_batch_decode_greedy launches the
argmax per greedy row on the [m x vocab] lm_head output (GreedyArgmax::
launch_ptr raw-pointer variant, cached [2m] u32 packed scratch), and
all-greedy batches (the serving default) download 8*m bytes instead of
m*vocab f32 -- the logits never reach the host.  Mixed batches dispatch
per row.  The engine path never needs logprobs so this is always safe
there.  MEASURED (TEMPURA_ENGINE_DEBUG_TICKS, 6-client streaming load):
sample+emit 11 ms -> 7.5 MICROseconds per step at m=6.  Semantics note:
the CPU chain argmaxes post-softmax probs; exp is monotonic so the
argmax agrees except in a ~6e-8-relative near-tie window -- the
divergence forward_greedy already ships with; all gates compare
new-vs-new so determinism is unaffected.  (2) THE BATCH ATTENTION
KERNEL -- oxide_decode_attn_batch (+ f16/e4m3 twins) now EXISTS in
tools/oxide-decode-attn: grid (n_heads, m), 256 threads, one block per
(request, head) pair running the kpar warp structure verbatim,
parameterized per block by the request's K/V run base addresses (read
from a device [m] u64 array -- the u64->pointer cast compiles fine in
the oxide dialect), its kv_len (device [m] i32 array), and its Q row of
the [m*n_heads*head_dim] batch Q tensor.  Per-request independent ->
composition invariant; per-block math bit-identical to a per-request
kpar launch (crate self-test: random per-request data, m in {2,3,6},
bit-identical for all three dtypes; memcheck/racecheck/initcheck
clean).  Host wrapper: forward_batch_d2d_ptrs{,_f16,_e4m3} on
pre-uploaded device arrays (the old forward_batch_d2d alloc'd per
call); forward_batch_graph_d2d keeps the 11-param ABI with the kpar
grid/smem.  Engine no-graph wiring: cached per-m device arrays +
grow-only host staging Vecs (kv_len_stage lesson), 3 tiny main-stream
H2Ds per layer, kernel writes straight into attn_out (per-request
out_r buffers + m scatter D2Ds per layer gone).  Dispatch m >= 2 only
(at m=1 the direct launch is a wash), uniform pool dtype; per-request
kpar stays the fallback.  NUMBERS (fishtank, RTX 5090, Llama-3.1-8B
Q4_K_M, f16 pool, bench-nograph 400-step contexts): engine step at m=6
46.6 -> 31.2 ms = 129 -> 192 tok/s aggregate (47 -> 31 ms vs yesterday
counting the sampler fix; short-ctx m=6 32.9 -> 28.7 ms; m=1
unchanged, direct path kept).  GATES ALL GREEN on the final tree:
6-client kv-swap A/B PASS on f16 AND e4m3 (6/6 byte-identical, swaps
exercised); pooled ppl 8 chunks f16 8.6838 / e4m3 8.6282 --
BIT-IDENTICAL to the pre-change baselines (the batch kernel is
numerically exact vs per-request); nonstream-route-smoke PASS;
kv-restart-var PASS (no cross-restart variance); bench-ab-repro
solo/repeat/interleave/m=2/m=3 checksums all IDENTICAL; local cargo
test green (916).  DEBUGGING NOTE (oxide crate): during bring-up the
batch self-test failed intermittently at m=3 (~50% of runs, always row
2) with BOTH the reference kpar and the batch kernel producing
garbage; the failures disappeared after a cargo-clean rebuild and
never recurred (40+ runs, 3 sanitizers x3, full suite x10).  Root
cause was never positively identified -- most plausibly stale
build-artifact mixing during the remote source-sed bisection (the
first "0.68 s" builds were cached).  If the batch self-test ever
flakes again: clean-rebuild the crate FIRST, then suspect the k_ptrs
indirection.  FOLLOW-UPS (open): (a) Option B -- batch GEMVs via the
general-M streamk/mma kernels (weights read once per projection
instead of m times; the per-row GEMV floor at m=6 is ~29 GB of weight
traffic per step -- now the DOMINANT cost at ~25 ms of the 31 ms
step); (b) m=1 serving decode is ~110 tok/s vs the single-seq graph's
217 (composition invariance forbids routing m=1 there; position-
synced batch graphs are the structural fix); (c) the sampler's CPU
path (non-greedy sessions) still downloads full logits rows -- a GPU
softmax/top-k would extend the same trick to sampled sessions.**

**STATUS 2026-11-15 (ragged no-graph serving decode 4.3-8.4x faster; TWO
root causes found -- a missing-feature server build and the sync-per-alloc
no-graph path; all gates green): the dominant serving perf gap is closed
to ~130 tok/s aggregate at m=6 (was ~10-30).  (1) THE SERVER HAS NEVER
HAD OXIDE KERNELS: `cargo build -p tempura --features cutile` wires
cli->server->engine `cutile` but NOT `oxide-gemv` (feature unification
via bench-gpu masked it for every benchmark) -- every Q4K projection in
SERVING ran the slow cuTile `cutile_q4k_gemv` fallback: nsys on the live
6-client server showed it at 92% of GPU time (2394 instances, 427us avg),
0.52 s/step at m=6, GPU "100% SM / 1% memory" -- the exact signature of
the documented "50x gap".  FIX: tempura-server's cutile/cuda features
now enable tempura-engine/oxide-gemv; kernels/build.rs gained
ensure_ptx_file (extract-from-binary -> copy the crate-root PTX left by
`gpu-remote.sh oxide` -> write an EMPTY file so include_bytes! still
compiles and runtime try_load returns None -- a fresh fingerprinted
OUT_DIR previously embedded whatever stale PTX happened to be there);
scripts/kv-swap-ab-validate.sh copies the PTX after its server build.
ALONE this took the live 6-client server from 521 -> 62 ms/step.
(2) THE NO-GRAPH PATH SYNC-PER-ALLOC: every `api::zeros(...).sync_on(s)`
is a FULL cuStreamSynchronize + alloc; the ragged batch path did ~3000
of them per step at m=6 (rmsnorm 3+trailing-sync x2 x32 layers, gemm
out+row allocs x7 x32 x(1+m), per-request q/k/v/rope/attn buffers,
`&mut None` residual adds, silu, and an inter->inter_pad HOST
round-trip: D2H+pad+H2D per layer -- Llama 14336->16384).  FIX: a
NoGraphBatchScratch pool on CutileGpuModel caches every buffer by
(slot,n,m)/(site,m)/m -- gemm out rows, activation rows, per-request
q_r/k_r/v_r/q_rot/attn_out, rmsnorm scratch (forward_device_fused_
scratch, bitwise-identical kernels), silu/GeGLU scratch, device-side
inter padding, and ping-pong residual-add buffers.  Numerics-neutral:
every buffer is fully overwritten in stream order or its pad tail
provably stays zero.  (3) ATTENTION: the per-request fallback used the
BASE single-warp kernel (step time grew linearly with kv_len: 86 ms
short-ctx -> 199 ms at kv~440, m=6); it now dispatches the K-PAR kernel
(+ new forward_kpar_d2d_ptrs_f16/_e4m3 twins -- the e4m3 kpar twin is
now loaded from the PTX too).  Per-request independent -> composition
invariant; used for every m.  NUMBERS (RTX 5090, Llama-3.1-8B Q4_K_M,
f16 pool, bench-nograph = new permanent bench, m=1..6): engine step
400-step contexts (kv up to ~440) 199.3 -> 46.6 ms at m=6 = 30.1 ->
128.9 tok/s aggregate (4.3x); short ctx 85.9 -> 32.9 ms = 182 tok/s
(2.6x); m=1 27.6 -> 9.0 ms = 36 -> 111 tok/s.  LIVE SERVER (6 clients,
512 tokens, engine-thread timing): forward 521 -> 62 ms/step; the
6-client kv-swap A/B wall time 285 s -> 34 s per run (8.4x).  GATES ALL
GREEN on the final tree: bench-ab-repro (m=2 AND m=3 batch checksums
bit-identical to solo, plus interleave S1); pooled ppl 8 chunks f16
8.6838 (baseline 8.6827, ratio 1.0001) / e4m3 8.6282 (baseline 8.6366,
ratio 0.999), 0 NaN both; FULL 6-client kv-swap A/B PASS on f16 AND
( first time) e4m3 -- 6/6 byte-identical, 3-4 swaps, mirror checksums
exact; nonstream-route-smoke PASS; kv-restart-var PASS (3 clients x 199
steps, texts + all logits checksums identical across restarts); local
cargo test green (867).  NOTE: the earlier "0.5-0.7 s/step" baseline was
PARTLY the missing-oxide bug (the A/B servers always built without
oxide-gemv); like-for-like on main with oxide present, the engine-step
improvement is 4.3x long-ctx / 2.6x short-ctx, and the end-to-end
serving gain vs yesterday's tree is 8.4x.  FOLLOW-UPS (open):
(a) oxide_decode_attn_batch is MISSING FROM THE PTX -- the kernel was
never written in tools/oxide-decode-attn/src/main.rs (has_batch() =
false, both graph and no-graph paths fall back to per-request
launches); writing it (+ f16/e4m3 twins) would collapse the m attention
launches per layer; (b) sample_and_emit costs 11 ms/step at m=6 (sampler
chain over 128k logits x6 -- CPU); (c) Option B (batch GEMVs via the
general-M streamk/mma kernels, weights read once per projection instead
of m times) is the next engine-side lever -- the per-row GEMV floor at
m=6 is ~29 GB of weight traffic per step; (d) m=1 serving decode is
111 tok/s vs the single-seq graph's 217 (composition invariance forbids
routing m=1 there; position-synced batch graphs are the structural
fix).**

**STATUS 2026-11-15 (6-client A/B flakiness ROOT-CAUSED + FIXED; the gate is
green on f16 main again): the cross-restart/batch-composition divergence was
a NULL-stream kv_len H2D racing the batch attention kernels.  In
forward_batch_decode's per-request attention fallback
(engine/src/cutile_gpu_forward.rs), each request writes its kv_len to the
shared d_kv_len device scalar before launching its attention kernel.  The
c94572d fix for the pooled-ppl NaN replaced the (stack-unsafe) async H2D
with memcpy_htod_sync -- but a synchronous cuMemcpyHtoD runs on the NULL
stream, and EVERY cuda-core stream is created CU_STREAM_NON_BLOCKING, so
the NULL-stream copy has NO ordering vs the engine's main stream.  At
m >= 2 the loop's next request's H2D could execute while the previous
request's attention kernel was still in flight; that kernel then read the
NEIGHBOR's kv_len and attended over the wrong length (neighboring pooled
run's rows -- wrong-but-plausible values, small logit deltas, no NaN).
This explains the full symptom set: only multi-client flows diverge
(single-client m=1 never has a second H2D in the tick), run A varies
across server restarts (race fires per timing/GPU queue state), the first
divergence lands at the decode step where the batch first becomes m >= 2
(verified with per-step [logits-cs] fingerprints: the session whose early
steps never overlapped a batch-mate was bit-identical for all 199 steps),
the 2026-11-14 PASS was luck (pre-c94572d the copy was at least
stream-ordered; its stack-lifetime bug fired rarely), and why
TEMPURA_NO_SM120_Q4K_BATCH=1 "didn't fix it" (kernel bisects cannot fix a
host-side copy race).  FIX: the per-request kv_len H2D is now a
main-stream memcpy_htod_async from a model-owned per-request staging slot
(kv_len_stage: RefCell<Vec<f32>> -- stable source address even if the
driver defers a pageable async copy, sized to the batch).  HUNT TOOLING
landed with it: tools/bench-gpu bench-ab-repro (deterministic single-
process interleave/composition scenarios; found the race fires only at
m >= 2 and variably per run), scripts/kv-restart-var.sh (2-server-restart
3-client variance gate with per-session first-divergent-step reporting),
scripts/kv-swap-ab-mini.sh (fast 3-client A/B).  VALIDATION (fishtank,
RTX 5090, Llama-3.1-8B-Instruct-Q4_K_M, f16 pool): bench-ab-repro 3x runs
-- m=2 AND m=3 batch checksums bit-identical to the m=1 solo baseline
(pre-fix: m>=2 diverged at a per-run-random step); kv-restart-var
(3 clients x 100 tokens, two fresh servers): all 3 texts + all 297
per-step logits checksums IDENTICAL across restarts; FULL 6-client
kv-swap A/B PASS (6/6 byte-identical, 3 swaps, bit-checksums exact);
nonstream-route-smoke PASS; local cargo test green (867 tests).  The
e4m3 re-run of the 6-client gate is now unblocked (same fix applies --
the f16/e4m3 twins share the launch path).  Also fixed en route
(hygiene, 231f6df): the sm120 m==2 Q4K batch GEMV dispatch removed from
gemm_any_d2d_batch.  FOLLOW-UP (open, minor): scripts/kv-swap-ab-mini.sh
run B with a very small pool (700 tokens) exposed an admission livelock
-- requeued admissions never trigger eviction, so 2 of 3 clients starved
until request timeout; root-cause when next touching the scheduler (the
full gate's 3072-token pool does not hit it).**

**STATUS 2026-11-15 (e4m3/FP8 KV pool landed, opt-in TEMPURA_KV_DTYPE=e4m3;
f16 stays default): the serving pool can now store KV as e4m3 (u8 bits,
64 KiB/token for Llama-8B GQA -- 4x f32 density, vLLM's production KV
format), mirroring the f16 p24 blueprint end to end.  Piece map: ONE
shared e4m3 bit-math implementation (kernels/src/e4m3.rs -- RNE write,
+-448 saturation, NaN->0x7F, subnormals FLUSHED to signed zero on BOTH
sides so round-trips and cross-path checksums stay exact; the oxide tool
include!s it and compiles it to PTX); cutile_cast_f32_e4m3/_e4m3_f32
append/mirror kernels (tile-level ports, two DSL quirks found: u32
literals >= 2^31 break the DSL's i32 literal parser, and shli/trunci
with overflow::None omit the overflow attribute bytecode serialization
requires -- use NoUnsignedWrap); cutile_flash_attn_heads_paged_e4m3
prefill twin; oxide decode-attn e4m3 twins (base/kpar/splitk, *const
u8); engine KvDtype enum refactor (ONE env parse point; TEMPURA_KV_F32=1
stays the f32 alias); dtype-aware staging append, DRAM-mirror syncs,
swap checksums, dispatch.  GATES: (1) oxide self-tests all green (f16
AND e4m3, all variants x kv_len {3,13,100}, exact by construction);
(2) nonstream-route-smoke e4m3 PASS (banner, byte-identical, coherent);
(3) ppl gate: pooled WikiText-2 8 chunks e4m3 8.6366 vs f16 8.6827
(ratio 0.995, gate <= 1.02), 0 NaN -- e4m3 stays OPT-IN; (4) f16
regression smoke PASS on the refactored code; (5) local cargo tests
green.  THE 6-CLIENT kv-swap A/B DID NOT PASS for e4m3 (5/6 mismatch)
-- but this is NOT an e4m3 issue: THE SAME GATE FAILS ON MAIN (bf681c3)
WITH THE DEFAULT F16 POOL (4/6 mismatch, artifacts /tmp/kv-swap-ab.*
on fishtank).  Root-cause evidence: first divergences occur BEFORE any
swap (token ~60 of ~500; swaps happen at seq ~500), server A alone
produces different texts across restarts, and the e4m3 run's 3 swaps
all had mirror/device bit-checksums EXACT with 0 NaN -- the swap
mechanism is bit-faithful; the mismatch is concurrent-batch composition
nondeterminism (suspect: gemm_any_d2d_batch's m==2 -> sm120 Q4K batch
kernel vs per-row GEMV at other m -- the 07f74ce neighborhood; also
matches the m=1/m=2 ppl deltas recorded 2026-11-15).  FOLLOW-UP (open):
root-cause the batch-composition token divergence and restore the
6-client A/B gate to green on f16 main (candidates: m==2 sm120 batch
GEMV vs per-row GEMV numerics; TEMPURA_NO_SM120_Q4K=1 bisect;
per-token logprob diff between m=1 and m=2 pooled runs via
TEMPURA_PPL_DUMP_LOGPROBS); then re-run the 6-client A/B for e4m3.**
      [RESOLVED same day: the real cause was the NULL-stream d_kv_len H2D
      racing the batch attention kernels (c94572d's memcpy_htod_sync has no
      ordering vs the non-blocking main stream) -- see the newer status
      entry above; the m==2 dispatch was also removed as hygiene (231f6df),
      but it was not the cause.]

**STATUS 2026-11-15 (pooled-ppl NaN ROOT-CAUSED + FIXED; f16 KV pool is
now the DEFAULT): the m=1 batch-decode NaN in perplexity_pooled was the
Q4K sm120 batch GEMV called with m != 2 — that PTX kernel is M=2
HARDCODED (q4k_batch_sm120_ptx.rs: "param_m_rows not read"; it
unconditionally reads a row-1 activation base and writes row-1 outputs
at out[n..2n]).  gemm_any_d2d_batch forwarded ANY m to it; at m=1 the
row-1 read ran past the q8 quantize scratch and the row-1 WRITE landed
n f32 elements past the [n] output tensor — silent neighbor corruption
in the cudaMallocAsync pool.  That explains the full symptom set:
deterministic per run (fixed allocation layout), build-dependent NaN
counts (2032/2040 recorded, 508/508 after a rebuild), f32/f16 count
drift (different pool layouts), and the "clean" server on chat data
(the corrupted 16KB happened to land in free pool memory there — its
coherent text was luck, not correctness).  The composition-invariance
work removing the m==1 single-seq special case is what first routed
m=1 into the kernel.  FIX (commit 07f74ce, three layers):
oxide_q4k_gemv_host refuses m != 2 in forward_sm120_batch_d2d(_into)
with a clear error; gemm_any_d2d_batch dispatches to it only at m == 2
(other m take the exact per-row GEMV fallback); TEMPURA_NO_SM120_Q4K_
BATCH=1 bisects the dispatch.  Hunt tooling landed with it:
TEMPURA_PPL_NAN_PROBE=1 (per-step NaN tuples, zero GPU work),
TEMPURA_PPL_M2=1 (dummy second batch member — this probe pinned
m=1-specific), and pooled mode now honors collect_logprobs so
TEMPURA_PPL_DUMP_LOGPROBS works for batch-vs-single diffs.
VALIDATION: pooled ppl m=1 f32 5.3993@1c / 6.1737@2c, f16 6.1644@2c,
m=2 f16 5.3922@1c == m=1 f16 (composition-invariant), DECODE_SINGLE
5.3894@1c (matches the pre-hunt record) — all 0 NaN; nonstream-route-
smoke PASS; local cargo test green.  THE f16 GATE PASSED — pooled f32
8.6799 vs f16 8.6827 at 8 chunks (ratio 1.0003, gate <=1.01, 0 NaN
both) — so TEMPURA_KV_F16's default is FLIPPED ON in new_kv_pool
(engine/src/model.rs; TEMPURA_KV_F32=1 restores the f32 pool; commit
follows 07f74ce); the smoke re-ran green on the f16 pool (banner
"512 pages, f16", byte-identical buffered vs SSE).  Follow-ups
recorded: (a) TEMPURA_NO_MXFP4=1 is NOT a clean bisect for the
no-graph batch path — it computes wrong logits at ANY m (m=1:
4.8857 vs 5.3993 correct) even after the sm120 fix; root-cause when
next needed; (b) the m>=2 f32 equal-position batch GRAPH path scores
ppl 7.5 vs 5.4 exact — that is the documented MXFP4 tensor-core
approximation (activation quantization ~0.1 rel logit delta), not a
regression, and serving never uses it (ragged / f16 -> no-graph, which
is exact and composition-invariant); (c) NEXT: FP8/TurboQuant on the
pool (bigger win; attention kernels need dequant epilogues), and
optionally a bench-decode-ops f16-pool section.**

**STATUS 2026-11-15: ppl gate for the f16 default-flip BLOCKED on an open
harness bug (commit c94572d landed the tooling + one real fix).
perplexity_pooled (TEMPURA_PPL_POOLED=1) scores the corpus through the
serving flow (pool + forward_with_kv prefill + m=1 forward_batch_with_kvs
decode) — and it NaNs DETERMINISTICALLY in the m=1 batch decode: identical
counts every run (2032/2040 scored positions f32, 2027/2040 f16), BOTH
dtypes (NOT f16-specific), wiki corpus.  The same batch path serves
correctly in the server (f16 A/B + smoke PASS, byte-identical, bit-exact
swaps); the wrapper-mode ppl on the same corpus is clean (7.9754@32c);
pooled + TEMPURA_PPL_DECODE_SINGLE (single-seq decode on the same pool) is
clean (0 NaN).  FIXED en route (real bug regardless): the per-request
kv_len H2D in the batch attention fallback raced its loop-iteration stack
slot (async copy deferring past the iteration scope -> garbage kv_len ->
OOB KV scans); now a synchronous 4-byte copy.  NEXT STEPS for the hunt:
(a) the NaN is deterministic + data-dependent — diff per-position logprobs
between pooled-batch and pooled-single runs (TEMPURA_PPL_DUMP_LOGPROBS) to
find the first divergent position and whether it correlates with specific
tokens; (b) try m=2 batches (a dummy second cache) — discriminates
m=1-specific vs general batch-path; (c) suspicion list: the oxide BASE
single-warp attention kernel (only the batch fallback uses it — the
single-seq paths use kpar/splitk; the server used the same base kernel
successfully on CHAT data, so corpus data may matter), probe contamination
under cuda-oxide v0.5.0's retirement-thread lease (extra D2H downloads
inside the layer loop), or an m=1 edge in the generic batch path
(post-m==1-removal) that chat-shaped data never exercised.  The f16
default flip stays BLOCKED until this resolves; server-side f16 remains
fully validated.**
**STATUS 2026-11-14 (f16 KV pool COMPLETE, pieces 1+2 landed): the
serving pool can store KV as f16 (TEMPURA_KV_F16=1, commits aaacbba +
023dba7) — halved KV density (128 KiB/token for Llama-8B GQA; the f32
pool's 256 KiB/token made a 131k context = 32GB, unreachable on a
32GB card; f16 is llama.cpp's default density).  Everything validated
end-to-end on fishtank: nonstream-route-smoke PASS (f16 pool banner,
byte-identical buffered vs SSE, coherent text) AND the full streaming
A/B PASS (6/6 byte-identical between a 20000-token f16 pool and an
eviction-forced 3072-token f16 pool; 3 swaps with mirror/device
bit-checksums EXACT — the quantizing DRAM-mirror roundtrip is exact).
Piece map: oxide f16 decode-attention twins (exact integer f16→f32
in-kernel, correspondence-tested at ≤2e-7); cutile_cast_f32_f16 kernel
(staged append conversion); cutile_flash_attn_heads_paged_f16 prefill
twin (f16 tiles + convert_tile); LayerTensors dtype-tagged pool
tensors (dtype-erased pointers, elem_size byte math); graph paths +
the wrapper's own session cache stay f32 (gated).  REMAINING before
f16 becomes the DEFAULT: (a) ppl quality gate vs the f32 pool
(bench-perplexity with the flag on; the MXFP4 lesson — gate on ppl,
not logit correlation; f16 KV is llama.cpp's default so the
expectation is a clean pass); (b) optionally extend bench-decode-ops
with an f16-pool section (kernel-level, random data); (c) flip the
default + retire the f32 serving path once (a) passes.  Then FP8/
TurboQuant on the pool as the follow-on (bigger win, attention
kernels need dequant epilogues).  A quality note: the f16 pool's
greedy text differs from the f32 pool's (expected — KV rounding
shifts near-tie logits); both are valid greedy continuations.**
**STATUS 2026-11-14 (f16 KV pool, piece 1 landed): the oxide decode
attention now has f16 twins (commit aaacbba) — single-warp/kpar/splitk
kernels reading K/V as f16 bits with an exact in-kernel integer
f16→f32 (normals, subnormals, inf/NaN), validated on fishtank against
the f32 kernel on dequant(the same bits) at max_diff ≤ 2e-7 across all
variants × kv_len {3,13,100} (random data — uniform data cannot catch
conversion/index bugs).  Host wrapper: has_f16() +
forward_d2d_ptrs_f16; PTX regenerated + committed.  WHY: the serving
pool stores KV as raw f32 (256 KiB/token for Llama-8B GQA) — 131k
context = 32GB, unreachable on a 32GB card; f16 halves it (llama.cpp's
default density) and halves the DRAM swap tier.  REMAINING for the f16
pool: (a) GpuKvPagePool/CutileKvCache pool tensors Tensor<f16> (byte
offsets ×2, kv_base_byte_offset), (b) append path converts f32→f16
(cutile convert kernel into a per-cache f16 staging tensor, then D2D),
(c) host mirror f16 (halves DRAM tier; swap checksums adapt), (d) the
cutile paged PREFILL attention loads Tile<f16> + convert_tile (op
verified to exist in upstream cutile), (e) the equal-position batch
CUDA-graph paths + kv_scatter_write gated OFF for f16 pools (f32
kernels; serving is ragged/no-graph today anyway), (f) gates: the
streaming A/B (byte-identical vs the f32 pool is NOT expected — f16
rounds KV; the gate becomes A-vs-A f16 and a ppl ratio check vs the
f32 pool, TEMPURA_PPL_MAX_RATIO), (g) then FP8/TurboQuant on the pool
as the follow-on.**
**STATUS 2026-11-14 (later): non-streaming chat now routes through the
ENGINE (commit 4626aee) -- plain requests (n=1, no logprobs/guided/
sampler-chain) submit an EngineRequest and buffer tokens through the
same scheduler/pool/swap path as streaming; the feature-rich direct
path (logprobs/guided/n>1/sampler chains) stays on the HTTP-thread
generate() and retains its documented hazard (concurrent GPU requests
interleave the wrapper's SHARED session KV cache -- the engine route
is the safe one).  Validated by scripts/nonstream-route-smoke.sh on
fishtank: identical byte-for-byte output for the same greedy prompt
via buffered and SSE responses (re-proving composition invariance
across requests).  NEXT FOLLOW-UPS, in order: (1) release the
wrapper's shared session KV cache in server mode (a quarter of VRAM
sits idle once the serving pool exists; the serving pool is sized
AFTER it, so pools are ~25% smaller than they could be) -- DESIGN: the
&mut window before Arc::new in run_server can drop it, but the direct
path needs a lazy fallback (generate() should own a LOCAL dedicated
CutileKvCache via gm.forward when the shared one is absent; forward()
with no cache currently falls to the CPU path which panics on GPU
models with freed CPU weights); (2) position-synced batch CUDA graphs
(reading per-row positions from a device array at execution time, the
d_kv_len pattern extended) -- the ragged no-graph path is correct but
~1-2 s/step, the dominant serving perf gap; (3) NIXL transfers + NVMe
tier; (4) batch-decode CUDA-graph QKV bias (Qwen2/Phi).**
**STATUS 2026-11-14: p24-kv-swap VALIDATION GATE PASSED — the streaming
A/B is green (6/6 clients byte-identical between a 20000-token pool run
and a 3072-token eviction-forced run; 3 swap-out/swap-in cycles in the
small-pool run; per-prompt coherent, unique text per client).  All work
LANDED as signed-off commits (engine foundations + batch-path fixes /
scheduler policy / TLA+ / engine_task rewire / docs+scripts).  The A/B
scripts/kv-swap-ab-validate.sh + kv-swap-ab-client.py are the permanent
regression gate (run on fishtank via gpu-remote; needs
--max-seq-len 24576 so the pool env override is not clamped, and
--request-timeout-secs 2400 — ragged no-graph decode runs ~1-2 s/step).
THE BUG CHAIN the A/B flushed out (7 real bugs — the gate did exactly
what it existed for; 1-3 and 7 were latent in code that had never run):
(1) engine_task re-insert loop used kvs.remove(i) inside enumerate() —
removals shift the vec, any 2-sequence batch panicked the engine thread
(remove-index 1 < len 1); (2) cuMemGetInfo_v2 fails rc=201 on the
engine thread (no current context — cuda-oxide ops bind internally, a
bare driver call does not) so pool sizing saw 0 free VRAM — fixed with
CutileGpuModel::free_vram_bytes() which binds the primary context
first; (3) the no-graph batch path called gemm_d2d("lm_head.weight")
directly — quantized lm_heads live in the q4k/q6k maps it cannot see,
so it silently returned ZEROS (now routes through lm_head dispatch /
per-row GEMVs); (4) CutileKvCache::paged_attn_inputs built an IDENTITY
block table, valid only for base-0 runs — with several pooled sessions
every non-base run's prefill attended to the FIRST run's KV (affine
base+i table now, matching GpuKvPagePool::alloc_run); (5) the no-graph
batch path advanced host seq_len once PER LAYER (+32 tokens/step
runaway — pool exhaustion plus layer-skewed KV garbage; the graph
path's comment literally warns about this); (6) attention kv_len in the
no-graph path excluded the just-appended token (seq_len, not
seq_len+1); (7) THE BIG ONE: gemm_any_d2d_batch's M-sequential fallback
concatenated per-row GEMV results that all ALIAS the shared slot
scratch in a loop AFTER all GEMVs — every projection returned m copies
of the LAST row's result, and q_rot_parts aliased the shared rope
scratch so every request attended with the last request's Q: the whole
batch computed ONE shared sequence (all clients generated the same
text, diverging only via sampling penalties).  Dead code until p24 —
ragged batches never ran before (bench-batch uses the equal-position
CUDA-graph path).  Fixes copy each row's result out INSIDE the loop,
before the next GEMV/RoPE can overwrite the scratch.  ALSO: batch
COMPOSITION invariance — the m==1 decode branch (single-seq graph,
exact kernels) vs m>=2 (generic path) made a session's logits depend
on who else was decoding; the m==1 special case is removed and the
batch lm_head dispatches per-row for any m, so logits are
composition-invariant (required for the byte-identical A/B, and the
right determinism posture anyway); sample_and_emit treats a CLOSED
response channel (client gone / request timeout) as session completion
(the 300s default timeout used to leave the engine spinning millions
of no-op ticks on dead sessions); eviction retries with only the
growing sequence protected when the whole batch is protected (a full
pool otherwise deadlocks with zero swaps — safe because replay-time
pointer validation re-captures graphs and the no-graph path bakes
nothing).  DIAGNOSTICS that earned their keep, now permanent:
TEMPURA_ENGINE_DEBUG_TICKS=1 (per-50-tick session/pool/run state +
per-step logits bits-checksums) and unconditional [swap] out/in lines
with host-mirror/device-readback bit-checksums (EXACT verified on every
swap).  Follow-ups: the no-graph ragged decode path is CORRECT but
SLOW (~1-2 s/step) — position-synced batch CUDA graphs are the p24-
overlap perf work; route non-streaming chat through the engine (arch
gap — it still bypasses the scheduler on the HTTP thread); release the
wrapper's shared gpu_kv_cache once the serving pool exists; NIXL
transfers + NVMe tier; batch-decode CUDA-graph QKV bias (Qwen2/Phi).
NOTE: no new Kani/Verus harnesses for the cutile-gated fixes (cargo
kani is featureless — GPU paths cannot run under it); the A/B gate +
bit-checksums + bench-decode-ops are the validation for these.**
**STATUS 2026-11-13 (later session): p24-kv-swap wiring + FULL SERVER
INTEGRATION -- code complete, locally verified, remote GPU validation ONE
STEP SHORT (details below; previous status preserved underneath).  All
work is UNCOMMITTED in the working tree and synced to fishtank as
wip/fdupont-thinkpadx1carbongen13.  Delivered this session: (1) engine --
Drop-frees-pooled-runs (fixed a page leak for process-lifetime pools),
try_ensure_capacity with recoverable PoolExhausted{need_pages,free_pages},
new_kv_pool/new_session_kv_cache/forward_with_kv/forward_batch_with_kvs
(cfg-cutile methods on the InferenceModel trait, default None = CPU path);
(2) server/batch.rs -- SequenceState::Swapped + LRU/priority
select_swap_victim(exclude)/mark_swapped/mark_resident/touch/requeue_active/
mark_decoding; swap PRESERVES state (unlike preempt->recompute); 6 unit
tests + a Kani harness (verified locally; NOTE kani CI only covers
kernels+engine, server harnesses are manual-only); (3) tla/
BatchScheduler.tla -- SwapOut/SwapIn actions, swapped-first admission
(swapped > preempted > pending), NoDoubleResident + SwapDisjointCompleted
invariants, SwappedEventuallyResumes liveness -- TLC CLEAN (134 states),
and TLC caught two real spec bugs (swapped ids re-enqueueable; swap-thrash
livelock needs SF(Complete) -- weak fairness cannot break the
SwapOut/SwapIn 2-cycle; the exclude-set in the Rust policy is what
prevents it in the implementation); (4) server/src/engine_task.rs
REWRITTEN -- the engine loop now drives admission/stepping through the
shared BatchScheduler + per-session pooled KV runs: priority swap-ins
(+256-token margin), chunked prefill via forward_with_kv, batched decode
via forward_batch_with_kvs with pre-growth + scheduler-driven eviction
(ensure_capacity_with_eviction: PoolExhausted -> victim -> swap_out ->
mark_swapped -> retry; the whole decode batch is protected from
eviction -- swapping a mid-batch member orphans baked graph pointers);
admission gated on pool headroom (ADMISSION_HEADROOM_TOKENS=768) to
prevent admit/requeue spinning; CPU models keep the old path unchanged;
AppState.scheduler switched tokio->std Mutex, now SHARED with the engine
thread (previously the scheduler was gauges/429-only and the engine loop
ignored it entirely); 69/69 server + 230/230 engine tests green;
remote cutile/cuda builds green on fishtank.  (5) TWO PRE-EXISTING BUGS
FOUND+FIXED: (a) the server's GGUF path injected quantized_weights
WITHOUT load_weights, so gpu_model was never built -- the tempura SERVER
has NEVER done GPU inference on GGUF models (silently CPU; found because
my pool banners never printed).  Fix: CutileInferenceWrapper::
try_upload_gpu() extracted from load_weights, called by the server's GGUF
branch.  Requires --device cuda (defaults to cpu).  (b) SamplingParams
gained Clone.  THE ONE REMAINING VALIDATION GATE (blocked at launch, not
at failure): the server's NON-STREAMING chat path calls
state.model.generate() DIRECTLY on the HTTP thread -- it BYPASSES the
engine channel/scheduler entirely (pre-existing architecture; only
"stream": true requests reach EngineTask).  All curl tests so far were
non-streaming -> they exercised the bypass, not the new loop.  NEEDED:
streaming A/B -- run A TEMPURA_KV_POOL_TOKENS=20000 (no eviction) vs run
B =3072 (forced eviction), 6 concurrent "stream":true clients, greedy
temp=0; PASS = byte-identical streamed text + "swapped out" lines in B's
serve.log.  WRITE THE SCRIPT TO A FILE AND SCP IT -- the long inline
gpu-remote run one-liner repeatedly mangled the tool call (quoting
layers); kill leftover servers BY PID before re-running (preflight
refuses otherwise).  AFTER validation: land as 5 signed-off commits
(engine foundations / scheduler policy / TLA+ / engine_task rewire /
AGENTS.md).  FOLLOW-UPS: route non-streaming chat through the engine
(arch gap); the wrapper's shared gpu_kv_cache (quarter VRAM) is wasted
once the serving pool exists -- release it; batch CUDA-graph fast path
requires equal positions (ragged falls to the validated no-graph path --
correct but slower; position-sync is p24-overlap work); NIXL transfers +
NVMe tier still open.**
**STATUS 2026-11-13: p20 (paged attention + KV page pool + tiering) is
COMPLETE end-to-end — tempura BEATS llama.cpp on both flagship GPU
benchmarks (prefill 7757-7797 vs 7335 pp128; decode 217-220 vs 209) with
20k-token GPU contexts (VRAM-sized pool, was 4096) and multi-client +
DRAM-tiering KV.  All session commits CI-green (kani.yml).  NEXT SESSION
PRIORITIES, in order (full rationale in the p20/p24 entries below):
(1) p24-kv-swap WIRING — the swap_out/swap_in primitives are validated
(swap granularity = whole runs); remaining: BatchScheduler LRU/priority
eviction calling swap_out on pool exhaustion (extend TLA+
BatchScheduler.tla with the swap transitions), NIXL transfers (do NOT
reinvent protocols -- the author's kv-mesh knows the plugin model),
NVMe capacity tier.  (2) MMLU GPU gap: MXFP4 prefill precision opt-out
for quality-sensitive workloads, or higher-precision tiling (the
rel-0.10 logit band -> ~40% choice-flip on 4-way near-ties is
quantified; CPU matches llama at 94.8% agreement)
 [RESOLVED 2026-11-17c: TEMPURA_PREFILL_PRECISION=exact landed,
 MMLU GPU 48.8 -> 62.8%, agreement 96.8% -- see the newer status
 entry].  (3) p24-overlap
scheduling (CPU prepares step N+1 while GPU runs N -- the step loop is
synchronous).  (4) the small broken edges: cutile_decode_attn (corr
0.66, dead code), oxide PTX attention softcap >4k ctx (Gemma-2), batch
QKV bias (Qwen2/Phi batch decode).  NOTE for remote workflows: kill
stale ssh watchers by PID (their command lines contain 'tempura'/
'bench-' and trip the gpu-remote preflight pgrep -- hit 3x this
session); the remote's cuda-oxide is at e1f6df0ad, VALIDATED
(bench-decode-ops 82/82, 7720 pp128, batch streamk M=16 809, all
within noise of a0a4dc78a).**

## Open work (verified gaps — do not assume these are done)

- [ ] **Native paged attention kernel (planned: .lean/plan.yml
      p20-paged-attention-kernel). NEXT UP (decided 2026-11-11, after the
      cuda-oxide update to a0a4dc78a absorbed + CI made green): the item is
      now UNBLOCKED -- upstream cute-rs gained SM120a CuTeDSL parity with
      a paged KV scatter reference and TiledCopy/TiledMma primitives
      (#1686/#1672/#1734 in cuda-oxide; see cute-rs-preview/).
      STEP 0 DONE (2026-11-11, bench-decode-ops "paged gather view probe"
      + bench-probe-gather): the cutile tile-level gather primitive the
      kernel needs WORKS -- GatherScatterView + load_view_raw gathers a
      [BC, D] KV tile with fully computed sparse row indices (token ->
      divi block_idx -> page-id gather -> page*bs+remi offset -> *n_kv_heads
      +kv_head de-interleave) BIT-EXACT vs a CPU reference on shuffled
      page mappings with ragged tails (4 shape configs). Throughput in
      isolation: 0.42-0.48x the dense load_pipelined path (1223 GB/s at
      kv=4096 / 1499 GB/s = 84% DRAM peak at kv=16384; shuffled pages cost
      nothing; the Some(4) latency hint is a no-op) -- a net WIN in-pipeline
      is still expected because the gather deletes the K/V gather/scatter
      round trips (an extra write+read of all of K/V per layer) and 4 of 5
      per-layer kernel launches. DSL CONSTRAINT found: RANK-1
      GatherScatterViews fail to JIT ("Failed to compile type") for both
      i32 and f32; rank-2 views compile for both -- the block table is
      passed as rank-2 [-1, 1] (isolated by bench-probe-gather variants
      A-F, kept as the regression record). Design decisions locked
      (2026-11-11 design review): build in cutile (not cute-rs -- the
      paging is expressible with gather views, cute-rs stays an algorithm
      reference only); block_size=16 stays (pow2 -> shifts, BC=32 spans 2
      pages, per-column resolution in the sparse index tile); kv_len read
      from a device scalar (d_kv_len pattern -- upstream cute-rs bakes it
      as a launch scalar, and the prefill graph's (seq_len, positions) key
      only pins it under an unenforced invariant); v1 = gather-view K/V
      with identity table over the existing contiguous pool (isolates
      kernel work from allocator work), Q gather + out scatter retained;
      then drop the remaining gathers via scatter views; reverse KV
      traversal + FIRST/LAST-only masks as a later perf lever. The cute-rs
      PageRing smem staging is unnecessary for cutile (the loader computes
      coordinates itself).
      STEP 1 DONE (2026-11-12): CPU paged reference in
      kernels/src/attention.rs -- paged_flash_attention (same tile loop,
      causal mask, and online-softmax order as flash_attention; the inner
      K/V loads resolve through paged_row_offset: block_table[token/bs] ->
      (page*bs + token%bs)*stride + kv_h*head_dim) + CpuPagedAttention
      (forward_paged; not an Attention impl -- the pool/table are extra
      inputs). BITWISE-identical to flash_attention batch=1 with identity
      AND shuffled page tables (test_paged_attention_identity_and_shuffled_
      match_flash; ragged kv_len vs FLASH_BC and block_size -- uniform/
      identity data cannot catch index bugs). Kani:
      paged_row_offset_in_bounds + paged_row_offset_identity_table (both
      verified; the CI plan job greps #[kani::proof] so they are
      self-enrolling). Verus: verus-verified/src/paged_attention.rs, 13
      proofs (offset in bounds for ALL inputs, identity-table collapse to
      the contiguous offset, distinct-tokens-distinct-rows within a page
      -- the write-safety side). VerifiedEquivalent<CpuAttention> row in
      the kernel-verify registry (error bound 0.0 -- bitwise; cites the
      Verus lemma). The CPU reference's paged_row_offset mirrors the
      step-0 GPU gather's index chain exactly.
      STEP 2 DONE (2026-11-12): GPU kernel v1 LANDED, at parity with the
      gather path. cutile_flash_attn_heads_paged (kernels/src/
      cutile_flash_attention.rs): same mma tiling/masks/softmax order as
      cutile_flash_attn_heads, K/V tiles gathered from the interleaved
      pool through the block table; kv_len read from a device i32 scalar
      at EXECUTION time (kv_len_dev; cache_offset derived in-kernel --
      graph-safe, the kv_scatter_write pattern); block table rank-2
      [-1, 1] (rank-1 gather views do not JIT). block_size == BC == 16
      (one page per tile): the page id is a single SCALAR table load and
      pool_row is affine -- no per-column divi/remi in the hot loop
      (Kani: paged_row_offset_affine_one_page_per_tile). Manual software
      pipelining (gather loads carry no latency hint): K(j+1) AND V(j+1)
      prefetched before tile j's compute, branch-free clamp on the last
      tile. Host: prefill_attention_device_paged (Q gather + out scatter
      retained; k/v head-major scratch DEAD) + CutileKvCache::
      paged_attn_inputs (zero-copy rank-2 pool views via Reshape,
      lazily-allocated identity block table [gpu_max_seq_len/16, 1],
      kv_len scalar updated via async H2D). Engine dispatch at the
      prefill branch; TEMPURA_NO_PAGED_ATTN=1 bisects to the gather path.
      PERF TUNING HISTORY (bench-real-model pp128, Llama-3.1-8B Q4_K_M,
      same-binary A/B): 7161 (naive per-tile gather chain, BC=32) ->
      7265 (K(j+1) prefetch) -> 7468 (BC=16: smaller tiles give the
      prefetch more overlap granularity -- the DENSE path preferred
      BC=32, gathers have different economics; BC=64 6914, BC=8 7256) ->
      7525 (V(j+1) prefetch too). Gather-path bisect on the same binary:
      7555. PAGED AT PARITY (0.4%, within noise; both beat llama.cpp
      7335). Decode unchanged 219. The paged advantage grows with
      context length (the eliminated K/V gather/scatter round trips are
      O(kv_len) per layer; at kv=128 they were only ~6us/layer).
      CORRECTNESS: bench-decode-ops "paged attn" section -- max_abs=0
      vs the CPU paged reference on 8 configs (4 shapes x identity/
      shuffled tables, ragged kv, cache-offset history) + bitwise
      determinism re-runs; bench-real-model NaN-free, det=OK, prefill
      logits fingerprint IDENTICAL to the gather path (bits_sum
      216043403379854, argmax 233); bench-quality teacher-forced gate
      PASS. NOTE: DSL has NO closures in kernels (standalone closure
      values are rejected -- the gather chains are inlined per site).
      ALSO FOUND + FIXED the same day (see the [x] item below): the
      cuTile Q4K GEMV warmup crash (DriverError 719) was a tempura
      format-mismatch bug (raw 36-u32 weights fed to the 42-u32 cuTile
      kernel) surfaced by cuda-oxide a0a4dc78a's new partition bounds
      checks -- not an upstream regression. Remaining
      sequence: (3) kill the Q gather + out scatter via scatter views
      (~5.7us/layer more), (4) real page allocation (non-identity
      tables + page-granular CutileKvCache refactor).
      STEP 3 DONE (2026-11-12): the paged kernel is now FULLY DIRECT --
      zero gather/scatter kernels, zero head-major scratch.  Q is
      gathered from the interleaved tensor through a padding::Zero view
      (pad rows of the last BR tile read as zeros -- the old gather
      kernel's zero-fill semantics for free); the output is stored
      DIRECTLY to the interleaved [seq_len, q_dim] buffer through a
      scatter view whose padding::Zero MASKS out-of-bounds stores (pad
      rows carry OOB sparse indices that the padding mode drops -- the
      output tensor stays exactly [seq_len, q_dim], keeping the
      caller's zero-copy reshape path intact).  Grid is EXPLICIT
      (.grid(...) -- no mutable output partition to infer it); only
      out_flat remains pooled.  DSL dead-ends found and recorded:
      Tensor::as_ptr is HOST-ONLY; i32->i64 scalar casts and
      PointerTile::reshape do not lower (three masked-pointer-store
      attempts abandoned -- see the NOTE in the kernel's store block).
      RESULT: prefill 7797 tok/s pp128 (7525 step-2 -> +3.6%) -- BEATS
      the gather path (7555 same-binary bisect) by 3.2% and the
      2026-10-29 7700 baseline; decode 220 unchanged; logits
      fingerprint IDENTICAL to the gather path; bench-decode-ops 8/8
      bit-exact + deterministic; bench-quality teacher-forced gate PASS.
      llama.cpp pp128 reference: 7335 (gap 0.9x in our favor).
      Remaining: (4) real page allocation (non-identity tables +
      page-granular CutileKvCache refactor -- the memory-tier wins).
      STEP 4a DONE (2026-11-12): the CPU paged hot path no longer
      materializes -- model.rs calls CpuPagedAttention::forward_paged
      directly on the PagedKvCache pools + block tables (the two
      k_contiguous/v_contiguous gathers PER LAYER are gone; softcap
      models included via a softcap param on paged_flash_attention,
      same tanh site as naive_attention_ggml_softcap and the GPU
      kernel).  Gates: test_paged_forward_matches_contiguous (1e-5)
      and test_paged_generate_matches_contiguous (identical tokens)
      both PASS with the flash-order paged attention (vs naive
      CpuAttention on the contiguous side); new
      test_paged_attention_softcap_matches_softcap_reference.  Also
      fixed two more pre-existing cfg(test) breaks invisible to kani
      CI: model.rs test configs missing final_logit_softcapping
      (Gemma b0166a7), and a stale spm_tokenizer expectation
      ([4,19,2,19,5] -> [4,19,2,5]: the post-special-token space is
      consumed by the piece's own ▁ prefix -- b0166a7's raw-piece
      matching, validated in the 5-family matrix).  STILL OPEN under
      step 4: the GPU side (page-granular CutileKvCache allocator,
      non-identity tables on the GPU pool -- the oxide decode
      attention PTX reads the pool contiguously, so per-sequence
      base-page offsets or table-aware PTX kernels are the design
      decision; prefix sharing/eviction on top).
      STEP 4b-1 DONE (2026-11-12): the GPU page-POOL MECHANISM, validated
      end-to-end (production wiring is the follow-up).  engine/src/
      gpu_page_pool.rs (UNGATED -- cargo kani is featureless, the
      cutile_quantized_gemm lesson): PageRunAllocator -- fixed-capacity
      array free list ([Run; 128]), BEST-FIT alloc, coalescing free via
      absorb-then-append (swap-remove only, NO memmove).  GpuKvPagePool
      + GpuKvRun (cutile_kv_cache.rs, gated): shared per-layer pool
      tensors, run handles with affine block tables (base+i), per-run
      kv_len device scalar, whole-pool rank-2 views for the paged
      kernel, upload_tokens at base-shifted offsets.  Sequences hold
      CONTIGUOUS runs so every contiguity-assuming reader (oxide decode
      PTX etc.) works from a base-shifted pointer -- no PTX changes.
      VALIDATED: bench-decode-ops "GPU page pool" -- two concurrent runs
      (B at base 8: NON-ZERO table indirection through the real kernel;
      C reusing A's freed hole), both bit-exact (max_abs=0) vs the CPU
      paged reference over a whole-pool mirror, conservation holds;
      Kani: alloc_free_conservation_and_well_formed (154s) +
      free_rejected_out_of_bounds (16s) -- free-list sorted/disjoint/
      non-adjacent/in-bounds + page conservation + no-overlap for
      symbolic request sizes; Verus gpu_page_pool.rs: 4 proofs
      (well-formedness, conservation, sorted-insert, run disjointness).
      KANI LESSONS (2026-11-12, the Vec-bomb class again): (1) Vec
      insert/remove with symbolic positions made CBMC OOM/heap-corrupt
      at 20+ GB on a 125 GB host -- fixed-capacity arrays with
      swap-remove only; (2) iterator-adapter folds over runtime-length
      slices (.iter().map().sum()) are the same bomb -- plain indexed
      while loops; (3) loops over a structurally-bounded length
      (free_len <= 128) still fail unwind(N) unless a PROVABLE concrete
      bound is asserted before the loop (harness asserts free_len <= 8
      from the concrete op sequence -- CBMC then prunes the unwinding).
      REMAINING for 4b: wire the pool into CutileKvCache/batch decode
      (per-seq 1024MiB full-size buffers -> runs from one shared pool;
      the batch graphs' per-seq base pointers update per replay already
      so base-page offsets compose), then prefix sharing/eviction.
      4b-2 DESIGN NOTES (mapped 2026-11-12, ready to execute): pooled
      mode is for the BATCH caches ONLY (single-seq sessions keep
      dedicated buffers, base 0 -- zero changes to their paths).  The
      ~15 batch call sites are UNIFORM raw-pointer extractions
      (kv_device_tensors(l).{0,1}.device_pointer().cu_deviceptr() --
      oxide attn, kv_scatter_write destinations, batch_attn_all_k_ptrs
      arrays, capture-time pointer validation): add a base-offset-aware
      accessor (kv_device_base_ptrs) and update them mechanically.
      Sites that pass Arc<Tensor> INTO kernels (gather-path prefill,
      cutile decode fallback) must NOT see pooled tensors -- batch is
      decode-only, so this does not arise there.  GROWTH: runs must
      allocate with HEADROOM (e.g. max(2x needed, +256 tokens)) --
      run reallocation changes the pointers baked into captured batch
      graphs, triggering re-capture; without headroom that would fire
      every 16 decode steps.  The existing "KV cache ptrs changed --
      re-capturing" validation/replay machinery handles realloc
      transparently.  sync_host_from_device must become base-offset
      aware (offset D2H, not whole-pool to_host_vec).  Gate:
      bench-batch (the batch fingerprints) + bench-real-model +
      bench-decode-ops after the wiring.
      STEP 4b-2 DONE (2026-11-13): the pool is wired into PRODUCTION --
      CutileInferenceWrapper's shared session cache now holds a run from
      a VRAM-sized GpuKvPagePool (quarter of free VRAM, >= the old
      4096-token default; TEMPURA_KV_POOL_TOKENS override,
      TEMPURA_NO_KV_POOL=1 bisect).  CutileKvCache pooled mode:
      new_pooled, ensure_capacity (extend-in-place via allocator
      try_extend, else realloc-with-D2D-copy; 2x growth headroom),
      base-offset appends, offset-D2H host sync (whole-pool to_host_vec
      would be catastrophic), growth-aware capacity gates.  THE GPU
      CONTEXT CAP IS LIFTED: Llama-8B now runs 20k-token contexts on
      GPU (was 4096) -- validated ppl 6.92@512 / 8.1681@4096
      (bit-identical to the dedicated no-pool path) / 7.0569@8192 (MORE
      context = better ppl, as expected); real-model 7757 pp128 / 219
      decode det=OK identical fingerprint; bench-quality 90.6% PASS.
      THE BUG HUNT (three bugs, all found by ppl@8192 = 2716 garbage):
      (1) block_table_dev staleness on growth -- the lazily-built table
      snapshots the capacity at first use; growth must invalidate it
      (fixed; defensive -- the gate bug below meant the table path
      wasn't even live in the failing runs, which is why the fix was
      bit-identical no-op); (2) the capacity GATES
      (can_append_device/has_device_kv) check the CURRENT run size, not
      the growth POTENTIAL -- a 4096-token single forward against the
      initial 768-token run fell to host-staged paths before any
      growth could happen (fixed: pooled gates consult pool free
      pages); DecodeGraph gained captured_capacity_gen because
      extend-in-place keeps pointers while outgrowing capture-time
      scratch; (3) THE ACTUAL KILLER, PRE-EXISTING: the RoPE cos/sin
      table was precomputed to max_seq_len.min(4096) -- positions
      >= 4096 read past the table -> garbage rotations.  The prefill
      half of a ppl chunk ends at exactly position 4095; every scored
      decode position >= 4096 was rotated with garbage.  Latent since
      the GPU paths were built (nothing ran past 4096 before the cap
      lift); fixed by precomputing the FULL context (~67MB VRAM at
      131k).  Also added: oxide decode attn long-kv regression
      (bench-decode-ops, kv 4097/8192 bit-exact -- the kernels were
      never the problem).  DIAGNOSTIC PLAYBOOK THAT WORKED: the ppl
      bisect matrix (pool@512 / no-pool@4096 / pool@4096) cleared the
      pool in three runs; the kernel-level long-kv test cleared the
      kernels; the position-4095 boundary in the ppl chunking pointed
      at position-indexed tables.
      STEP 4b-3 + TIERING DONE (2026-11-13): PHASE 4 COMPLETE.
      MULTI-CLIENT: new_batch_kv_caches allocates n caches as RUNS from
      one shared VRAM-sized pool (per-seq cost = pages USED, not a full
      max-size buffer); the batch raw-pointer sites (~15) now use
      CutileKvCache::kv_device_base_ptrs (run-base pointers) --
      kv_scatter_write destinations, batch_attn_all_k_ptrs arrays,
      capture validation, kpar/batch attention fallbacks (new
      forward_{d2d,kpar}_ptrs wrapper variants; the cuTile flash
      fallback documents its whole-tensor limitation -- broken kernel
      anyway).  Growth hooks: forward_batch_decode grows every run
      BEFORE graph replay (extend-in-place keeps baked pointers; realloc
      re-captures via pointer validation); BatchDecodeGraph gained
      captured_capacity_gens (extend evades pointer checks).
      VALIDATED: bench-batch all M (1..16) verified NaN-free +
      deterministic at parity (M=16 803 vs 809 dedicated -- within
      noise).
      TIERING (DRAM fallback, the p24-kv-swap mechanism):
      CutileKvCache::swap_out (sync mirror to the DRAM tier, free the
      GPU run back to the pool) + swap_in (alloc a fresh run --
      possibly a different base -- upload the mirror back).  Whole-RUN
      granularity (attention reads the full prefix every step;
      page-granularity tiering just moves the read cost to PCIe).
      VALIDATED (bench-decode-ops "KV tiering"): B filled + attention
      bit-exact -> swap_out (pool fully recovered) -> client D
      allocates from the freed pages -> swap_in at a DIFFERENT base
      (20) -> attention bit-exact again vs the same reference.  90/90
      sections PASS.  NOT YET: scheduler-driven eviction policy
      (LRU/priority) + NIXL transfers + NVMe tier -- the p24-kv-swap
      wiring on top of these primitives; prefix sharing/eviction.
      Baselines:
      prefill PARITY at pp128 (7525 paged vs 7555 gather, 2026-11-12;
      the 2026-10-29 7700 baseline reads ~7555 on current hw/cuda-oxide
      state), decode ITL unchanged, gather/scatter kernels: ALL GONE
      from the paged prefill chain (Q gather, K/V gathers, out scatter --
      one kernel launch per layer). What
      remains un-paged: the CPU forward path still materializes
      PagedKvCache::k_contiguous()/v_contiguous() before running standard
      contiguous attention (model.rs -- the CPU paged reference from step
      1 can replace it), batch-decode CUDA graphs still run oxide
      attention over contiguous caches, and the GPU pool is still the
      contiguous CutileKvCache viewed through an identity table (real
      page-granular allocation is step 4).
- [x] **cuTile Q4K GEMV fallback crash under cuda-oxide a0a4dc78a --
      ROOT-CAUSED and FIXED (2026-11-12, same day)**: NOT a cuda-oxide
      regression -- a tempura bug the update surfaced.  compute-sanitizer
      on a standalone repro (bench-probe-q4k -- the kernel is BIT-EXACT
      on synthetic 42-u32 data) + memcheck on bench-real-model pinned
      it: when the oxide kernel's uses_raw_gguf is set, Q4K weights are
      uploaded in the RAW GGUF 36-u32 block format
      (maybe_upload_q4k_raw), but the cuTile cutile_q4k_gemv fallback
      kernel indexes with 42-u32 strides -- every program with
      pid_n >= 6*n/7 reads past the tensor (memcheck: first traps at
      blocks 3512-3534 of 4096 = 4096*6/7).  Older cuda-oxide silently
      read OOB garbage (harmless only because the warmup discards
      results); a0a4dc78a added partition-access bounds checks that
      TRAP -> DriverError 719, which poisoned the context (bench-real-
      model was fully blocked on main).  FIX: DeviceWeightQ4K now
      carries uses_raw_gguf (set at upload); the constructor warmup
      skips raw-format entries (the cuTile kernel cannot consume them,
      and gemm_q4k_d2d prefers oxide for those weights anyway) and a
      debug_assert in gemm_q4k_d2d's cuTile fallback keeps the formats
      from ever mixing at dispatch.  Validated both ways on fishtank:
      raw config 7790 tok/s det=OK; PTX-load-forced-fail (repacked
      42-u32) config runs the cuTile warmup cleanly ("kernel warmup
      complete", 7800 tok/s prefill, logits det=OK -- decode 12 tok/s
      expected with no oxide kernels).  LESSON: the
      stale-cubin/silent-garbage class strikes again -- a warmup that
      discards results cannot catch a format mismatch; the format is
      now part of the weight struct.  (The streamk PTX load failure
      observed in the same logs was a SEPARATE tempura bug, also fixed
      2026-11-12 -- see the sm_120a PTX note under GPU inference.)
      cuda-oxide e1f6df0ad (2026-11-12 pull) VALIDATED on fishtank:
      bench-decode-ops 82/82, real-model 7720 pp128 / 217 decode,
      det=OK, identical logits fingerprint; batch-with-streamk
      M=16 809 -- all within noise of the a0a4dc78a numbers.
- [x] q6k_gemv_sm120_kpar8 (single-seq Q6K GEMV, renamed from
      ..._mbar_kpar8): FIXED and DEFAULT ON since 2026-10-29
      (register-exchange rewrite; TEMPURA_NO_SM120_Q6K=1 opts out).
      The kb=16 residual race is closed: 0/81920 at EVERY %8 shape
      including kb=16, kb=56 real weights 5e-6 vs CPU, 8/8
      DETERMINISTIC under churn, bench-quality teacher-forced PASS
      90.6%, 169 tok/s decode (ITL 5.92 ms). The fix, and the two
      hardware truths found on the way (SASS + a synthetic-weights
      micro harness that drives the real kernel, a permanent
      bench-decode-ops section):
        1. The kb=16 residual was CROSS-LANE smem reads: lanes read
           words other lanes copied via cp.async, and no CTA barrier
           fully ordered that publication. The fix removes the
           dependency class: each lane reads ONLY the words it copied
           itself (round 1: lane l copies word l; round 2: lanes 0..20
           copy words 32+l), and cross-lane values (d, scales, ql/qh
           words) arrive via shfl.sync.idx REGISTER exchange (registers
           carry no proxy semantics; the shuffle's data dependence on
           the source lane's ld.shared forces that load to complete).
           12 ld.shared -> 2 predicated ld.shared + 12 shuffles.
        2. The CTA bar.sync x3 at the loop tail is LOAD-BEARING, not
           belt-and-suspenders: with only [cp.async.wait_group 0;
           fence.proxy.async] (per-thread), iteration >= 2 read
           UNWRITTEN slots at kb=13/16/40 with BOTH wait placements
           (tail and rotated loop-head), SASS audited correct in both --
           MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S does not publish LDGSTS
           data to subsequent same-thread LDS on this hw/ptxas without
           a CTA barrier. (Earlier findings stand: ptxas ELIDES warp-
           uniform mbarrier arrive+try_wait pairs; bar.sync lowers to
           DEFER_BLOCKING -- the 2nd bar forces the 1st to complete,
           the 3rd pins it; the cross-warp reduce uses [st.shared;
           membar.cta; red.release.cta counter bump / ld.acquire poll;
           sequential 8-partial sum on tid 0] -- no shuffles.)
       Consequence of (2): the k_blocks % 8 == 0 host gate STAYS (the
       CTA loop barrier requires uniform warp trip counts). Run the
       micro harness: `bench-decode-ops` (section "sm120_kpar8 micro
       (REAL kernel, synthetic weights)").
- [x] **lm_head was dequantized to f32 at load (2.1GB) although the GGUF
      stores it Q6_K (435MB) -- FIXED 2026-10-29** (nsys
      `--cuda-graph-trace=node` per-kernel profile of one decode step: the
      f32 lm_head GEMV alone was 1.24ms = 21% of the 5.92ms ITL). lm_head
      now goes through the same Q4_K -> Q6_K -> Q2_0 -> f32 upload priority
      as the layer projections (try_proj); decode/prefill m=1 dispatch the
      exact Q6K kpar8 GEMV (result mirrored into SLOT_LOGITS with a D2D
      copy so the graph callers keep finding it there); m>1 batch paths get
      an MXFP4 tiled copy (435MB -> 109MB). Decode 169 -> 196 tok/s
      (ITL 5.92 -> 5.11 ms, gap 1.07x), prefill 5866 -> 6033, batch
      M=2 136 -> 244 / M=16 602 -> 796 (the batch lm_head GEMM re-read
      2.1GB f32 per step), +1.3GB VRAM. Verified: probe/prefill argmax
      IDENTICAL to the f32 path, bits_sum within ULP noise; bench-quality
      teacher-forced PASS 90.6% UNCHANGED (same mismatch list); bench-batch
      all M NaN-free + fresh-cache replay bitwise (fingerprints
      re-baselined for the MXFP4 lm_head).
- [x] **q6k sm120 kpar8 bandwidth gap FIXED (dual-block direct-load
      variant, 2026-10-29)**: the cp.async kernel ran down_proj
      [4096,14336] at ~0.86 TB/s while the q4k dual_block kernel does the
      same shape at ~1.25 TB/s -- the cp.async design's load-bearing CTA
      bar.sync x3 per iteration (21 per CTA) locksteps all 8 warps: every
      warp waits for the slowest warp's prefetch+compute before issuing
      its next, exposing DRAM latency with no way to skew.
      `q6k_gemv_sm120_kpar8_db` (second entry in the same module) adopts
      the q4k recipe with the register-exchange consume: weights load
      DIRECTLY into registers via ld.global.nc (2 warp-instructions per
      212-byte block -- no smem, no cp.async, no proxy fences, NO barriers
      in the loop), dual-block (blocks A and A+8 per iteration, 2x MLP),
      prefetch.global.L2 16 blocks ahead, running x pointers.
      Accumulation order unchanged -> bitwise-equal output (probe
      fingerprint identical; micro harness 0/81920 at every kb including
      non-multiples of 8 -- the _db variant needs no k_blocks % 8 gate).
      Decode 196 -> 204 tok/s (ITL 5.11 -> 4.89 ms, gap to llama.cpp
      1.025x). TEMPURA_NO_Q6K_DB=1 bisects back to the cp.async variant.
- [x] **Single-sequence GPU decode gap vs llama.cpp CLOSED -- BEATING:
      217 vs 209 tok/s (1.04x in our favor; 2026-10-29).** Final change:
      fused RMSNorm (3 kernels -> 1, 195 -> 65 graph nodes).
      graph instantiation at all three capture sites (single decode +
      both batch graphs) -- the ~930-node decode graph's cuGraphLaunch
      was 122-139us of SERIALIZED host time per step (GPU idle; nsys
      runtime trace); pre-staging the graph on the device took ITL
      4.86 -> 4.77 ms (206 -> 210 tok/s), logits bitwise-unchanged.
      Cumulative decode timeline on this item: 140 (BK=512 lm_head) ->
      169 (q6k sm120 register exchange, default ON) -> 196 (lm_head via
      the GGUF's native Q6_K instead of 2.1GB dequantized f32) -> 204
      (q6k dual-block direct-load GEMV) -> 206 (Q6K v_proj via kpar) ->
      210 (cuGraphUpload). bench-quality teacher-forced PASS 89.1%
      throughout. Remaining micro-optimizations (not a gap anymore):
      logits DtoH 39-49us/step and the CPU argmax are GONE for greedy
      decoding since 2026-10-29: `forward_greedy` (InferenceModel trait,
      CutileInferenceWrapper override) runs the decode graph + a GPU
      argmax kernel (kernels/src/greedy_argmax.rs -- packed
      (ordered-key<<32)|index u64 atomicMax reduction whose tie-break
      exactly matches Rust max_by's last-max-wins; validated by a
      permanent bench-quality section: generate vs free-run manual
      continuation, 64/64 MATCH) and downloads 4 bytes instead of 513KB.
      The same change fixed a pre-existing bug: CutileInferenceWrapper::
      generate delegated to the CPU model's generate, which runs the CPU
      forward and PANICS on GPU-loaded models whose CPU projection
      weights were freed -- it is now a GPU-native loop. Remaining: ~40us
      residual launch cost (node-count reduction: rmsnorm is 3 kernels x
      65, the q/k rope chain 6 kernels x 32).
      NOTE: this GGUF (bartowski Llama-3.1-8B Q4_K_M) has a mixed layout
      -- 16 layers with Q6K v+down_proj, 16 with Q4K -- verified from the
      GGUF tensor-type census (193 Q4K + 33 Q6K) and the trace.
- [x] **Quality benchmark suite:** implementation-robust quality
      benchmarks + multi-model validation COMPLETE (5-family matrix,
      2026-11-10; follow-ups below).
      The free-running greedy comparison showed all implementations
      (including llama.cpp's own GPU) produce repetition loops for
      open-ended prompts -- quantized greedy decoding operates at the
      edge of chaos, and small numerical differences determine whether
      you get a coherent story or a loop. The teacher-forced 89.1%
      agreement is the right internal gate, but external validation
      needs benchmarks that are insensitive to argmax flips:
      (a) [x] **Perplexity comparison** (WikiText-2): engine/src/eval.rs
          + tools/bench-gpu/src/bench_perplexity.rs (bench-perplexity).
          Replicates llama.cpp's perplexity algorithm exactly (n_ctx
          chunks, BOS REPLACING the chunk-start token, positions [n_ctx/2, n_ctx-2]
          scored, f64 log-softmax NLL) but scores via prefill-half +
          incremental decode (forward returns last-position logits
          only; causal attention makes per-position logits identical).
          Corpus: ~/.cache/tempura/eval/wikitext-2-raw/wiki.test.raw
          (from https://wikitext.smerity.com/wikitext-2-raw-v1.zip).
          Reports tempura GPU / tempura CPU / llama.cpp (CPU
          llama-perplexity, -ngl 0) side by side; gate on the GPU/CPU
          ppl ratio (TEMPURA_PPL_MAX_RATIO, default 1.10). Knobs:
          TEMPURA_PPL_CTX (512), TEMPURA_PPL_CHUNKS (32; -1 = full
          corpus), TEMPURA_NO_LLAMA, LLAMA_PPL.
          THE 5-FAMILY MATRIX IS COMPLETE (2026-11-10): every CPU and
          GPU side PASSes the 1.10 gate. Full bug narratives live in
          the commit messages (each family flushed out real bugs the
          others masked); the rules they taught are extracted below.
            Llama-3.1-8B   CPU 7.31    / llama 7.09    = 1.03x    GPU PASS
            Mistral-7B     CPU 5.96    / llama 5.96    = 1.003x   GPU 5.98
            Qwen2.5-7B     CPU 6.4567  / llama 6.2968  = 1.025x   GPU 6.6770
            Phi-3.5-mini   CPU 6.0753  / llama 5.9609  = 1.019x   GPU 6.3598
            Gemma-2-9B     CPU 7.7312  / llama 7.7263  = 1.0006x  GPU 7.6835
          Reports: bench-perplexity{,-mistral7b,-qwen25-7b,-phi35-mini,
          -gemma2-9b}.json. Bug-chain commits: Llama rope pairing +
          harness (2026-10-30 lineage), Mistral 1f44ac1, Qwen 442ab23 +
          145e105, Phi 7df8c19, Gemma b0166a7.
          MXFP4 BAND POLICY: the default GPU config runs Q4K/Q6K
          projections through MXFP4 double-quantization (logit-level
          probe corr 0.82-0.93 vs the exact path's ~0.998) -- accepted
          when the ppl gate passes (Llama 1.09x, Qwen 1.034x, Phi
          1.047x, Gemma 0.994x). ppl, not logit corr, is the criterion.
          Exact path: TEMPURA_NO_MXFP4=1 (works in single-forward probe
          mode; NOT for repeated prefills -- see follow-up (6)).
          QUALITY-MODEL INVARIANTS (each line prevents a real, fixed
          bug -- do not "simplify" these away; the narratives live in
          the commits referenced above):
            * RoPE pairing: the GGUF converter permutes Q/K rows into
              adjacent-pair order for NORM-rope arches (llama, llama4,
              deepseek, granite, smollm3, mistral3/4, command-r, ...)
              and leaves split-half order for NEOX arches (qwen*,
              phi*, gemma*, falcon, gptneox, bitnet). Wrong pairing =
              progressive positional corruption that 1-2-token probes
              CANNOT catch (corr degrades with distance; llama.cpp's
              rope-type table in src/llama-model.cpp is the source).
            * cutile tile dims must be powers of two. Non-pow2 model
              dims need tiling or zero-padding: Qwen q bias 3584 ->
              cutile_bias_add_rows_tiled (B = largest pow2 divisor,
              clamped 512); Phi head_dim 96 -> cutile_rope_*_tiled;
              flash attention head_dim zero-pad in the host-staged
              paths (EXACT: pad columns contribute 0 to QK^T, pad
              output columns are never scattered).
            * GGUF key spellings: {arch}.attn_logit_softcapping and
              {arch}.final_logit_softcapping (no "attention." infix);
              tokenizer.ggml.add_space_prefix (SPM dummy prefix;
              gemma GGUFs carry 0).
            * The GGUF converter bakes Gemma's (1+w) norm offset into
              every norm.weight: rmsnorm_offset stays FALSE for
              GGUF-loaded gemma (plain rms_norm x w), TRUE for raw HF
              safetensors weights.
            * Gemma-2 has FOUR norms per layer: post_attention_norm
              and post_ffw_norm norm the attention/FFN OUTPUTS before
              the residual adds (mapped HF names:
              self_attn.post_attention_norm / mlp.post_ffw_norm).
            * Gemma FFN is GeGLU (tanh-approx GELU gating), not SiLU:
              dispatch by cfg.hidden_act; the fused-silu GEMV path and
              the MXFP4 Silu epilogue are SiLU-only and must stay
              gated off for GeGLU.
            * SPM tokenization must match llama.cpp bit-for-bit:
              USER_DEFINED (type 4) pieces merge as text AND match as
              RAW-text prefixes BEFORE whitespace escaping (gemma's
              "\n" and "  " pieces); the merge map has no type filter.
              A wrong tokenizer yields high ppl with a PERFECT probe
              corr -- always diff the token count against llama
              (llama_logitprobe tokenize mode).
            * CUDA graph capture: ANY allocation, D2H download, or
              cuTile JIT inside capture = DriverError 900. Pre-size
              scratch in the constructors/warmup_gemms; keep prefill-
              sized and decode-sized scratch SEPARATE; forward_*_into
              CONSUMES its out tensor on failure -- never hand it the
              decode slot pool for m>1 callers.
            * cutile_decode_attn IS BROKEN (corr 0.66 vs the oxide
              kernel's 0.9973 on identical inputs; never exercised --
              all validated models use oxide attn in the decode graph).
              Do not route decode through it until fixed. The oxide
              PTX lacks attention softcapping (inert at ppl context
              lengths; matters beyond ~4k ctx where scores exceed
              Gemma-2's 50 cap).
            * The m=1 Q6K projections return the SHARED kpar scratch
              (pre-allocated at max(inter, vocab) for lm_head) whenever n
              fits -- consumers must tolerate OVERSIZED rank-1 tensors and
              read by explicit element count (add_device_scratch's
              contract). Exact-count reshapes on these tensors panic
              (found by the bench-real-model probe 2026-11-10; the probe's
              catch_unwind mislabels any failure as "unsupported model
              shape" and exits 0 -- a silently-skipped bench is the
              symptom, check for "OK -- forward pass works").
            * Uniform-data kernel tests cannot catch fragment-mapping,
              store-permutation, or pairing bugs: validate every new
              kernel against a random-data CPU reference.
          CROSS-IMPLEMENTATION DEBUGGING PLAYBOOK (llama.cpp is the
          reference, never a reimplementation; /tmp artifacts on
          fishtank, recreate if wiped):
            * TEMPURA_PPL_PROBE=<ids>: exact token ids through both
              sides, dump last-position logits (.cpu.bin/.gpu.bin);
              compare corr + argmax vs /tmp/llama-logitprobe output
              (modes: last position, `all` per-position, `batch` one
              prompt per line, `tokenize` for tokenization diffs;
              needs LD_LIBRARY_PATH=~/Code/github.com/llama.cpp/
              build/bin).
            * TEMPURA_PPL_PROBE_ALL=1: per-position DECODE logits for
              both sides (.cpu.dec.bin/.gpu.dec.bin). A prefill-vs-
              decode disagreement at the same position localizes
              decode-path bugs (this found the GeGLU / decode-attn /
              scratch bugs on Gemma).
            * TEMPURA_PPL_DUMP_TOKENS / TEMPURA_PPL_DUMP_KV:
              tokenization and K-cache dumps for byte-level diffs.
            * TEMPURA_DEBUG_NAN=1: per-op NaN + bits_sum trace; also
              forces eager execution (D2H during capture is illegal).
            * Order of operations: fix CPU to reference quality FIRST
              (corr 0.99+ vs llama at multiple lengths), then
              GPU-vs-CPU.
            * Kill remote benches by PID or full command line, NEVER
              pkill by binary name.
          * REMAINING follow-ups: (1) [x] bench-quality's llama.cpp arbiter
              now renders the prompt through the model's chat template
              (both sides see identical templated text; post-fix: free-
              running text coherent, prefill argmax MATCH, llama.cpp
              teacher-forced agreement 73.4%, GPU/CPU 82.8% PASS);
              (2) re-characterize free-running text quality
              post-fix (greedy on this quantized instruct model still
              repetition-loops with both tempura and llama.cpp's GPU --
              edge of chaos -- but tempura with the template now
              produces grammatical, on-theme text where the pre-fix
              output was degenerate); (3) [x] bench-batch re-baselined
              (M=1 223, M=2 243, M=4 401 tok/s, NaN-free, det=OK,
              sensible fingerprints); (4) validate the qwen/gemma/phi
              NeoX mappings and the config.rs HF-path conventions
              against llama.cpp the same way; (5) the graph-prefill path
              (TEMPURA_GRAPH_PREFILL=1) is still broken and off;
              (6) allocation/download during CUDA-graph capture panics
              the TEMPURA_NO_MXFP4 / TEMPURA_NO_GPU_PREFILL prefill
              paths (found via MMLU, cutile_gpu_forward.rs:~4837)
              [RESOLVED 2026-11-17c: the prefill graph has been off by
              default since the cuda-oxide v0.5.0 retirement-thread fix
              (TEMPURA_PREFILL_GRAPH=1 opts in) -- TEMPURA_NO_MXFP4=1
              now runs without panic (on the slow host-dequant
              fallback), and the quality mode is
              TEMPURA_PREFILL_PRECISION=exact; see the newer status
              entry];
              (7) batch-decode CUDA graphs (m>=2) have no QKV bias
              support -- bias models (Qwen2/Phi) compute wrong batch
              logits until the capture sites apply the same
              apply_qkv_bias helper;
              (8) CI status of the 2026-11-09 pushes (76b8eac..
              current) unverified -- no gh auth on fishtank; check
              kani.yml via web or auth gh before building on top;
              (9) next-session priorities, in order (updated 2026-11-10:
              (a), (b), (c) all DONE -- the 5-family matrix is complete):
              (a) [x] Qwen prefill perf; (b) [x] Phi GPU; (c) [x]
              Gemma-2-9B ppl (10 bugs, see the Gemma entry); (d)
              batch-decode CUDA graphs (m>=2) QKV bias support; (e) the
              remaining Qwen prefill gap (3x) and Phi prefill gap
              (host-staged attention); (f) Gemma decode 103 tok/s (2.0x
              gap); (g) FIX cutile_decode_attn (broken: corr 0.66 vs
              oxide's 0.9973 on identical inputs -- every validated model
              uses the oxide kernel, so this path is dead code that
              silently corrupts if ever taken); (h) oxide PTX attention
              softcap for >4k contexts.
      (b) [x] **MMLU-style log-prob evaluation**: bench-mmlu
          (tools/bench-gpu/src/bench_mmlu.rs). Per question: prefill the
          standard harness prompt, read the log-probs of the four
          answer-letter tokens (" A".." D") from the last-position logits,
          argmax. Compares CHOICE agreement (robust to argmax noise).
          Dataset: stratified 250-question sample of cais/mmlu test at
          ~/.cache/tempura/eval/mmlu/mmlu_sample.jsonl. llama_logitprobe
          gained a batch mode (one prompt per line, KV cleared between
          prompts) so the reference runs without reloading the model.
          Knobs: TEMPURA_MMLU_MIN_AGREE (0.85), TEMPURA_MMLU_NO_CPU,
          TEMPURA_NO_LLAMA, LLAMA_LOGITPROBE.
          RESULT (Llama-3.1-8B Q4_K_M, 250 questions, 0-shot):
            tempura CPU  61.6%   llama.cpp 62.4%   agreement 94.8%
            tempura GPU  51.2%   agreement vs CPU 58.4%
          CPU is at reference quality. The GPU's larger drop vs its ppl
          gap (9%) is the MXFP4 prefill band concentrated: MMLU prompts
          are single prefills whose ENTIRE KV is MXFP4-written (ppl
          diluted the error across 256 exact-decode positions per
          chunk), and 4-way near-tie choices amplify a rel-0.1 logit
          band into ~40% choice flips. Quantified perf/precision
          tradeoff -- candidate follow-up: MXFP4 prefill opt-out for
          quality-sensitive workloads, or higher-precision tiling.
          NOTE: the GPU-side bisect switches are broken for this
          workload (TEMPURA_NO_MXFP4=1 / TEMPURA_NO_GPU_PREFILL=1 panic
          during CUDA graph capture -- see follow-up (6)). Open work.
          [RESOLVED 2026-11-17c: TEMPURA_PREFILL_PRECISION=exact landed;
          re-baselined on 2026-11-17 main the mxfp4 GPU is 48.8%
          (agreement 58.0%), exact-mode GPU 62.8% (agreement 96.8%,
          PASS) at 355 tok/s pp128 vs 7790 default; the capture-panic
          is gone with the default-off prefill graph -- see the newer
          status entry above.]
      (c) [x] **Self-consistency generation** (GSM8K-style):
          bench-selfconsistency (tools/bench-gpu/src/bench_selfconsistency.rs,
          2026-11-04; REFERENCE RUN 2026-11-06 on fishtank). K samples per
          problem at T=0.7 (seeded, deterministic per sample: sample k uses
          base_seed+k), answer extraction
          (engine/src/eval.rs extract_answer: "Answer:"/"answer is"/"####"
          markers, last-number fallback, char-boundary-safe for non-ASCII
          completions) + majority() vote (canonical per-f64 number format so
          "3.50"/"3.5" aggregate). Sides: tempura GPU, tempura CPU
          (TEMPURA_SC_CPU=1), llama.cpp via llama-server
          (TEMPURA_SC_LLAMA_URL, -ngl 0 CPU llama-server works for the
          arbiter). Dataset: deterministic 102-problem sample of GSM8K test
          at ~/.cache/tempura/eval/gsm8k/gsm8k_sample.jsonl, built by
          scripts/prep-gsm8k.sh (source: openai/grade-school-math GitHub).
          Knobs: TEMPURA_SC_PROBLEMS/_SAMPLES/_MAX_TOKENS/_SEED/_MAX_GAP,
          TEMPURA_SC_NO_GPU. RESULT (2026-11-06, Llama-3.1-8B Q4_K_M,
          102 problems x 5 samples, T=0.7): tempura GPU 63.7%,
          llama.cpp (CPU llama-server arbiter) 52.0% -- gap 11.8 pts,
          PASS at the 15-pt gate (tempura AHEAD of the reference).
          llama phase ~2.6 h at ~10 tok/s CPU decode; report at
          bench-selfconsistency.json. Self-consistency voting behaves as
          designed: no repetition-loop pathology, coherent multi-sample
          agreement on arithmetic. bench-gpu now has a default-on `cutile` feature
          so CPU-only hosts build with `cargo build -p tempura-bench-gpu
          --release --no-default-features --bin bench-selfconsistency` (the
          CUDA toolkit was previously required to build the crate at all;
          the remote workflow is unchanged). Validated end-to-end CPU-only
          (Qwen2.5-0.5B smoke run); the reference run (Llama-3.1-8B Q4_K_M,
          full 102 problems x 5 samples) is pending the GPU host.
      (d) [x] **Multi-model validation**: COMPLETE 2026-11-10 (see the
          matrix in (a)). Each family flushed out real bugs the others
          masked: Mistral (SPM tokenizer), Qwen (QKV biases x3, staticpar
          race, capture-time pad alloc, f16 subnormals, h_pad striding),
          Phi (fused qkv/gate_up splits, silu+sequential, Q5K-as-Q4K
          catch-all, rope mscale, non-pow2 head_dim), Gemma (softcapping,
          norm offset, four-norm layers, tokenizer x3, GeGLU, broken
          cutile_decode_attn). Bug narratives: commit messages
          1f44ac1..b0166a7.
      Current quality matrix for reference (Llama-3.1-8B Q4_K_M,
      teacher-forced, 64 positions):
        tempura CPU vs GPU:    89.1%  (internal gate -- PASSES)
        llama.cpp CPU vs GPU:  76.6%  (reference impl's own gap)
        tempura vs llama.cpp:  ~20%   (systematic impl difference)
      Free-running text quality (6 diverse prompts):
        llama.cpp CPU:   6/6 coherent (100%)
        llama.cpp GPU:   3/6 coherent (50%)
        tempura:         0/6 coherent (repetition loops)
      The repetition is inherent to quantized greedy decoding, not a
      tempura bug (llama.cpp's GPU has the same problem). Temperature
      > 0 or self-consistency sampling would resolve it.
- [x] **Prefill on GPU: 7700 tok/s pp128, TTFT P50 16.5 ms -- BEATING
      llama.cpp (ref 7335, 1.05x in our favor; 2026-10-29).**
      Lever-by-lever history
      below (nsys-
      driven: on-device attention + scratch RoPE + pooled norms/adds/SiLU +
      Q6K down_proj via MXFP4 + flash BR=16 + lm_head BK=512/last-row d2d)
      vs llama.cpp 7335 (1.01x gap -- parity; 17.8x the 408 tok/s host-staged
      baseline). Six changes landed 2026-10-28:
      (a) The prefill attention is fully on-device: 3
      `cutile_gather_heads` (de-interleave Q/K/V into head-major padded
      buffers) + ONE `cutile_flash_attn_heads` launch for all heads (the
      audited `cutile_flash_attn_head` math with multi-head tiling) +
      `cutile_scatter_heads` back to interleaved -- 5 async kernels per
      layer, zero PCIe, zero host syncs, scratch pooled
      (`PrefillAttnScratch`, RopeScratch-style take/try_unwrap). 408 ->
      815 tok/s. Prefill KV append uses `append_device_gpu_only` (the
      staleness guard before the layer loop now syncs the host mirror
      whenever THIS call needs host attention -- this also fixes a latent
      stale-mirror bug where a prefill chunk after GPU-only decode read a
      mirror missing the GPU-only region). TEMPURA_NO_GPU_ATTN=1 bisects
      back to the host-staged path.
      (b) Prefill RoPE left the `forward_device` fallback (which per layer
      recomputed cos/sin on the CPU, did 2 blocking H2D uploads, 5 blocking
      allocs and a FULL stream synchronize) for dedicated prefill rope
      scratches: `RopeScratch::ensure_capacity` grows the buffers to the
      chunk size once, and a (positions, theta)-keyed cos/sin cache makes
      layers 2..N upload-free -- 815 -> 1305 tok/s, logits fingerprint
      BITWISE-identical (pure overhead removal). The prefill scratches are
      separate fields because ensure_capacity reallocates and the
      decode/graph scratches are pointer-baked into captured CUDA graphs.
      (c) Prefill norms/residuals/SiLU left the allocating paths for
      dedicated prefill scratch pools: forward_device_fused_scratch for both
      RMSNorms (sum_t/scale_t gained missing size checks -- a stale smaller
      buffer would silently norm only the first rows), add_device_scratch
      for both residual adds (gained the same size check; add_device
      removed), silu_mul_device_scratch for the FFN. Eliminates ~288
      blocking api::zeros.sync_on allocs per 32-layer chunk (3 per norm x2,
      1 per add x2, 1 per SiLU); 1305 -> 1365 tok/s, logits fingerprint
      still bitwise-identical.
      (d) nsys showed q6k_gemv_sm120_batch at 77% of prefill GPU time: the
      kernel caps at 16 rows per launch (smem reduction area), so an m=128
      chunk launches 8x per layer, each re-reading the full 47MB down_proj
      weight matrix at ~40% DRAM efficiency (~2.3ms/layer, ~74ms of the
      94ms TTFT). Fix: Q6K down_proj is now converted to MXFP4 tiled at
      load (maybe_upload_mxfp4_tiled_from_q6k -- the same double-
      quantization the Q4K projections already use, but LOSSIER: Q6K is
      6.5-bit) and prefill dispatch tries gemm_mxfp4_prefill before the
      q6k batch GEMV. +~1GB VRAM for the 32 MXFP4 down_proj copies.
      1365 -> 5126 tok/s, TTFT 94 -> 25 ms. PRECISION TRADEOFF (accepted):
      prefill A/B rel 0.1015 -> 0.1331; bench-quality teacher-forced PASS
      89.1% (57/64, all 7 mismatches cpu_margin <= 0.65 near-ties), prefill
      last-position argmax MISMATCH vs CPU (margin 0.264 near-tie; was
      MATCH). Escape hatches: decode (m=1) keeps the exact Q6K kpar GEMV;
      TEMPURA_NO_MXFP4=1 skips the conversion (down_proj falls back to the
      q6k batch path, prefill ~1365 tok/s, exact); TEMPURA_NO_GPU_PREFILL=1
      keeps the host-dequant prefill. Batch decode M>=2 also switched
      down_proj to MXFP4 (the batch path already preferred it when present)
      -- fingerprints RE-BASELINED 2026-10-28 and batch throughput ROSE:
      M=2 133->136, M=4 224->243, M=5 253->290, M=8 332->407,
      M=16 444->602 tok/s (all M re-verified NaN-free + fresh-cache replay
      bitwise). The capture-pre-size blocks in both batch graph captures
      now pre-size sum_t/scale_t and the add scratches to exact-m as well
      (the new size checks in forward_device_fused_scratch /
      add_device_scratch realloc on mismatch, which is illegal inside
      stream capture -- found by bench-batch, DriverError 900).
      (e) Flash-attn tile tuning: the multi-head kernel was CTA-count-bound
      at prefill sizes (grid 128 CTAs on 170 SMs at seq=128, running at
      ~2.7 TFLOPS / 13% of prefill GPU time). A (BR, BC) sweep (now a
      permanent bench-decode-ops section -- BR/BC are launch-time generics,
      one build sweeps all tuples) found BR=16 ~1.9x faster at both seq=128
      (48 vs 91us) and seq=512 (400 vs 754us); BR=8 overcorrects, larger
      BC strictly worse. BR 32 -> 16; BITWISE-neutral (per-row math
      unchanged -- identical logits fingerprint and identical bench-quality
      verdict). 5126 -> 5485 tok/s.
      (f) lm_head GEMV BK=512 + final-norm last-row d2d: the f32 lm_head
      projection ran cutile_gemv_coalesced at BK=128 = 709 GB/s (2.6ms per
      prefill, 11% of TTFT, and ~1.5ms of EVERY decode token). A BK sweep
      on the real weights (permanent bench-decode-ops section) found BK=512
      at 1475 GB/s (82% of peak, 2.08x); the d2d wrappers pick 512 when
      k_pad allows, 128 otherwise. The prefill final-norm also stopped
      downloading the whole [seq, h_pad] hidden state to extract the last
      row (blocking sync + PCIe) -- one D->D row copy now. 5485 -> 5816
      tok/s prefill, TTFT 23.4 -> 22.1 ms; DECODE 139 -> 171 tok/s
      (ITL 7.15 -> 5.83 ms -- the decode gap item below). Not bitwise
      (reduction order), but det=OK, NaN-free, bench-quality PASS 89.1%
      unchanged, argmax identical.
      (g) MXFP4 GEMM register-blocked variant (2026-10-29): the m16n8
      kernel issues ~10 global loads per MMA (load-issue-bound at
      111-127 TFLOPS on the m=128 prefill shapes; GEMMs = 75% of prefill
      GPU time, nsys). mxf4_batch_gemm_tiled_rb4 gives each warp an
      m16 x n32 tile -- 4 MMAs share each A-fragment load (per-MMA load
      count ~4.5, A re-read traffic /4). BITWISE-equal output (same MMA,
      same k-order; validated by the tool's random-data CPU reference AND
      a cross-kernel bit comparison, plus the in-engine prefill logits
      fingerprint staying identical). Wins only on large-n grids (it
      quarters the block count): gate/up [128,14336,4096] 122.4 -> 106.2 us
      (141.6 TFLOPS) but loses at n<=4096 -- the host dispatches rb4 only
      when ceil(n_tiles/16) * m_tiles >= 512 (RB4_MIN_BLOCKS; m=16 batch
      decode never qualifies). 6084 -> 6255 tok/s, TTFT 21.0 -> 20.4 ms.
      A first cut of rb4 dropped the `+ lane/4` from the B row index -- the
      cross-kernel bit comparison caught it instantly (uniform-data tests
      could not have).
      (h) m32 register blocking (2026-10-29): after (g), the large-n GEMMs
      were B-re-read-bound across m-tiles (gate/up re-read the 29MB B
      matrix 8x = 235MB per GEMM of L2/DRAM traffic). rb4m32 gives each
      warp an m32 x n32 tile (2 m16 halves x 4 n8 tiles = 8 MMAs per
      k-step; B fragments shared across the m-halves -> B re-reads /2).
      BITWISE-equal (same validation chain as rb4). Isolated (L2-hot)
      benchmark: gate/up 123.3 -> 57.8 us (259.9 TFLOPS); in-pipeline
      ~67us (B streams from DRAM per layer). Dispatched when m % 32 == 0
      and its grid >= RB4M32_MIN_BLOCKS (128): gate/up/down/q/o at m=128
      qualify, k/v (n=1024) stays m16n8, m=16 batch decode stays m16n8.
      6255 -> 6481 tok/s, TTFT 20.4 -> 19.7 ms (1.13x gap).
      (i) rb2m32 (m32n16/warp, 2026-10-29): the down_proj GEMM
      [128,4096,14336] ran 162.7us IN-PIPELINE vs 114.6 isolated -- its
      rb4m32 grid (128 blocks = 512 warps = 3/SM) cannot hide the DRAM
      latency of the streamed B matrix. rb2m32 doubles the warp count for
      the same m32 tiling (2 n-tiles per warp instead of 4): down
      114.6 -> 97.5 us (154.1 TFLOPS) and q/o 35.0 -> 30.4 us isolated;
      gate/up keeps rb4m32 (57.5 vs 65.3), k/v keeps m16n8. Dispatch
      tiers: rb4m32 (grid >= 256) > rb2m32 (grid >= 128) > rb4 (m%32!=0
      large-n) > m16n8. 6481 -> 6865 tok/s, TTFT 19.7 -> 18.6 ms
      (1.07x gap). Remaining GEMM structure: ~12ms of the 18.6ms;
      non-GEMM overhead ~5ms (flash 1.5, silu 0.9, adds/rope/gather/
      scatter/rmsnorm/quant ~2.0, host ~0.6).
      (j) Elementwise tile width (2026-10-29): the residual adds and the
      SiLU-mul ran on 32-element cutile tiles -- 384B moved per program
      (2x128B reads + 128B store), 16K tiny blocks per [128,4096] add at
      ~690 GB/s = half of DRAM peak (nsys). B 32 -> 256 (1KB tiles;
      elementwise -> bitwise-identical for any tile size; the q8_snap
      kernel keeps B=32 -- one Q8 block). 6865 -> 7257 tok/s, TTFT
      18.6 -> 17.6 ms (1.01x gap).
      (k) GEMM-epilogue fusion experiment (2026-10-29): rb4m32_silu (UP
      projection: C = silu(gate) * up, gate read in the store epilogue)
      and rb2m32_add (DOWN: C = acc + residual) -- both validated (tool
      test 4: silu within 1e-3 of the f64 reference, add exact) but
      MEASURED AS A REGRESSION in-pipeline: prefill 7257 -> 6994 tok/s.
      The epilogue reads gate/res at the m16n8 accumulator's scattered
      store addresses (~32B transactions) -- that costs more than the
      eliminated silu_mul/vector_add kernels. Dispatch is default-OFF
      (TEMPURA_FUSED_EPILOGUE=1 opts in); the kernels + validation stay
      in tools/mxf4-gemv for a future attempt that stages the epilogue
      read through coalesced smem. Post-revert: 7304 tok/s, TTFT
      17.5 ms, 1.004x gap -- prefill parity.
      (l) Fused RoPE-on-interleaved (2026-10-29): the prefill Q/K rope ran
      as a 3-kernel chain (deinterleave -> rope -> interleave) -- 6 memory
      round-trips and 3 kernel launches per Q/K per layer, ~15us/layer
      (nsys). cutile_rope_interleaved does it in ONE kernel (reads the
      interleaved [num_sh*head_dim] tensor, applies the rotation, writes
      interleaved; same indexed loads, same formula, same store positions
      -> bitwise-equal). TEMPURA_NO_FUSED_ROPE=1 bisects back to the
      3-kernel chain. 7304 -> 7473 tok/s, TTFT 17.5 -> 17.1 ms --
      BEATING llama.cpp 7335 (1.02x in our favor). Logits fingerprint
      bitwise-identical, bench-quality PASS 89.1%.
      (m) Fused RMSNorm (2026-10-29): the rmsnorm ran as a 3-kernel chain
      (sum_sq -> scale -> apply) -- 195 of the ~930 decode graph nodes
      (65 instances x 3 kernels), each a tiny kernel with a full memory
      round-trip between passes. rmsnorm_fused does it in ONE kernel
      (load x/w, reduce sum(x^2), rsqrt, multiply -- all in registers;
      same reduction, same rsqrt formula, same apply -> bitwise-equal).
      TEMPURA_NO_FUSED_RMSNORM=1 bisects back to the 3-kernel chain.
      Decode 210 -> 217 tok/s (graph 930 -> ~800 nodes, launch + dispatch
      overhead drops); prefill 7473 -> 7562 tok/s. Logits fingerprint
      bitwise-identical, bench-quality PASS 89.1%.
      (n) Prefill CUDA graph (2026-10-29): the prefill chunk's layer loop
      (~800 kernel launches, ~0.6ms of host dispatch per chunk) is
      captured into a CUDA graph on the SECOND call with the same
      (seq_len, positions) key and replayed on subsequent calls. The
      first call runs eagerly (all scratch buffers get allocated/grown);
      the second captures; the third+ replays. The final norm + lm_head
      stay OUTSIDE the graph (their D2H download is not capture-safe).
      Requires seq_len % 16 == 0 (m_pad == m: no trim alloc during
      capture); TEMPURA_NO_PREFILL_GRAPH=1 bisects to eager. The
      last-row buffer (final norm input) is pre-allocated with a stable
      address. 7562 -> 7700 tok/s, TTFT 16.9 -> 16.5 ms. Logits
      fingerprint bitwise-identical, det=OK, bench-quality PASS 89.1%.
      llama.cpp cross-implementation comparison (teacher-forced,
      2026-10-29): tempura CPU 20.3%, tempura GPU 18.8% agreement with
      llama.cpp's greedy choices -- an implementation difference (SIMD
      accumulation order vs scalar f32), not a quality issue; the
      internal GPU-vs-CPU 89.1% is the gate. llama.cpp's OWN CPU-vs-GPU
      teacher-forced agreement: 76.6% (49/64) -- tempura's 89.1% is
      BETTER than the reference implementation's internal consistency.
      llama.cpp free-running CPU-vs-GPU diverges at token 0 (different
      first token; the model is in near-tie territory immediately).
      Verified 2026-10-28: bench-decode-ops "prefill attn dev" 8/8, "rope
      prefill scratch" 4/4 and "rmsnorm prefill scratch" (grow/reuse/
      shrink/odd-seq) 4/4 all
      bitwise-exact vs their references; bench-real-model NaN-free, det=OK,
      A/B vs host-dequant rel 0.1015 (the documented MXFP4 band); decode
      unchanged at 139-140 tok/s. Remaining levers: gemm dispatch host
      overhead in the chunk loop, the gather/scatter copies (~3 extra
      passes over Q/K/V activations -- fusible into the flash kernel or the
      RoPE/GEMM epilogues), BR/BC tile tuning for long prompts
- [ ] Validate hybrid DeltaNet pipeline against real Qwen3.6 / Bonsai 2
      weights (last attempt: "architecture incompatible" — diagnostics added,
      commit 9c6362d)
- [x] **m=1 GPU decode path produced wrong logits -- FOUND and FIXED
      (2026-10-28, bench-quality + bench-decode-ops)**. Root cause: the
      hand-written sm_120 PTX kernels used `shfl.sync.bfly/idx` with
      c=0x71F for their 8-lane mini-warp reductions; the correct width-8
      encoding is c=0x181F (verified via `nvcc --ptx` on sm_120). With
      0x71F, mini-warp 3 (lanes 24-31) collapsed onto lane 24: the
      reduction returned 8x lane-24's partial and dropped sub_l 1-7, so
      j-group 3 of every Q4_K GEMV contributed garbage whenever its qs
      bytes were non-uniform (~110% relative logit error). Fixed in
      q4k_gemv_v4_ptx.rs, q4k_batch_sm120_ptx.rs, q4k_fused_silu_v4_ptx.rs
      (18 shuffle sites). Uniform-data tests cannot catch this class --
      the one-hot element sweep in bench-decode-ops is the regression
      test. Post-fix: bench-quality shows 100% argmax agreement (64/64
      tokens, free-running greedy, identical continuations) vs the CPU
      reference; 134 tok/s decode unchanged. Every decode op now has a
      real-weights CPU-math validation in bench-decode-ops: RoPE table +
      graph-gather paths, KV scatter-write, q6k kpar GEMV, q4k sm120 kpar
      GEMV, oxide decode attention (splitk + kpar, incl. dirty
      4096-row cache), rmsnorm fused scratch, silu_mul.
- [x] q4k_gemv_dp4a_q8_raw (sm_80 Rust-codegen fallback GEMV): WRONG --
      FOUND and FIXED (2026-10-28). Root cause: the crate's device-side
      f16_to_f32 returned f32-DENORMAL bits for f16-subnormal inputs
      (exp==0 path: sign | mant<<13), so every Q4_K weight block whose f16
      super-scale d fell just below 2^-14 (extremely common -- quantizers
      emit d right around the subnormal boundary) lost its scale term and
      only the -dmin*m contribution survived: 414/4096 output rows off by
      ~0.1-0.4 (additive error, invisible on large outputs -- "correct on
      large, wrong on small"). The sm_120 PTX kernels dequant d/dmin via the
      hardware cvt.f32.f16 and were never affected. Also fixed
      f32_to_f16_bits (used by the SHARED quantize_q8_1 kernel): it flushed
      subnormal f16 Q8_1 activation scales to zero, silently dropping any
      32-element activation block with amax < ~0.0078 -- now rounds to
      nearest subnormal. Post-fix: bench-decode-ops q8_raw rel 0.0003
      (was 0.333), 0/4096 bad rows, one-hot sweeps (tail + main loop,
      incl. negative d) clean; 134 tok/s decode unchanged. Regression
      tests: the q8_raw real-weights census + crafted one-hot sweeps +
      per-block isolation in bench-decode-ops. NOTE: the PTX is a
      gitignored build artifact -- kernel-source changes to
      tools/oxide-gemv-q4k require `cargo oxide build` + the out-dir copy
      loop (see the GPU Kernels section) on every clone.
- [x] **Short-prompt GPU prefill wrong -- FOUND and FIXED (2026-10-28, three
      distinct bugs, found by bench-quality + bench-decode-ops)**:
      1. The embedded mxf4_gemv.cubin was STALE: the 2026-10-27 accumulator
         store-mapping fix (11d2d7c) changed tools/mxf4-gemv/src/main.rs but
         the cubin was never rebuilt/committed -- the engine ran the pre-fix
         kernel (rows written past the m16 tile into the cudaMallocAsync
         pool; outputs allocation-history dependent) while the standalone
         tool compiled the fixed source and passed. Symlink-style trap: the
         tool validates SOURCE, the engine runs the CHECKED-IN BINARY.
         Lesson recorded below in the mxf4 build recipe. Post-rebuild the
         bench-decode-ops quant+GEMM composition test is bit-exact at every
         m (the old cubin failed at ALL m including 128; uniform-scale and
         permutation-cancelling random-nibble tests cannot catch it).
      2. cutile_flash_attn_head computed num_kv_tiles = kv_len / BC
         (truncating): kv_len=12 gave 0 tiles (all-NaN logits from 0/0) and
         kv_len=85 silently dropped keys 64..84 for the last rows. Fixed to
         ceil_div -- this also cut the m=86 host-dequant prefill delta from
         1.77 to 0.34 and the teacher-forced decode delta from 1.87 to 0.37.
      3. The GGUF tokenizer defaulted add_bos=false when the
         tokenizer.ggml.add_bos_token key is absent -- Llama-3.1 GGUFs ship
         bos_token_id without that key, so prompts lost their BOS
         (generation degraded, worst on short prompts). from_gguf_metadata
         now defaults BOS-on for the llama/mistral/gemma families (llama.cpp
         LLM_ARCH semantics); llama.cpp-arbiter verified.
      Post-fix: fast prefill argmax-MATCH at every prompt length (m=13: 3.43,
      m=86: 1.91, both vs the CPU reference -- the intrinsic MXFP4 4-bit
      noise band, matching the documented pp128 A/B rel 0.10); short-prompt
      CPU and GPU continuations are identical text.
- [x] Generation-quality validation of the MXFP4/decode paths (2026-10-28):
      bench-quality (greedy GPU-vs-CPU continuation, per-position argmax
      agreement, teacher forcing, NaN-aware diffs, GPU decode-vs-prefill
      self-test) + bench-decode-ops (per-op real-weights CPU-math checks +
      one-hot element sweeps) + a llama.cpp arbiter run. Result after the
      shuffle fix: 100% argmax agreement, 64/64 identical tokens
      free-running (85-token prompt, Llama-3.1-8B Q4_K_M, logits max_abs
      4.08 / avg 2.17 -- same band as the prefill's own CPU delta).
      MODE CHANGE (2026-10-28, later): teacher forcing is now the DEFAULT
      (TEMPURA_QUALITY_FREE=1 restores free-running). The free-running
      verdict is a coin flip -- post-BOS-fix prompts land the model in
      near-tie territory (first divergence at a margin-0.014 top1/top2 gap)
      and every position after the flip "disagrees" with logit deltas of
      10-20 that measure different contexts, not kernel health. Same code,
      same day: free-running 15.6% FAIL while teacher-forced PASSes at
      92.2% with max_abs 4.27 / avg 2.44 (the documented band). Use
      teacher-forced agreement + the logit-delta band as the quality gate;
      use free-running only to eyeball continuation text.
      Cross-impl note: the CPU reference diverges from llama.cpp after ~2
      greedy tokens (near-tie argmax flips, quantized-impl accumulation
      differences) -- llama.cpp text output remains the ground truth for
      "is the model thinking correctly".
- [ ] ARM NEON/SVE optimized CPU kernels (optimized paths are x86-only today)
- [ ] NPU backends: `npu-intel`/`npu-qualcomm` are todo!() stubs (OpenVINO /
      QNN SDK required)
- [ ] Disaggregated serving over network — traits + local worker impl only
      (planned: .lean/plan.yml p24-pd-disaggregation-network; bind to NIXL,
      do NOT reinvent transfer protocols. Prior art, not an integration
      dependency: the author's kv-mesh
      ~/Code/gitlab.cee.redhat.com/fdupont/kv-mesh knows the NIXL plugin
      model, but its G3 NVMe tier is a capacity tier, not the PD hot path)
- [ ] Serving-technique gaps vs vLLM/SGLang (planned as .lean/plan.yml
      Phase 24, filed 2026-11-04): p24-overlap-scheduling (CPU prepares
      step N+1 while GPU runs step N — the step loop is synchronous
      today), p24-kv-swap-preemption (preemption requeues/recomputes; no
      swap-to-CPU-DRAM path), p24-eagle-speculative (EAGLE draft heads
      distinct from the done Medusa work), p24-orchestration-surface
      (engine-side KV export/import + prefix digests so EXTERNAL routers
      can front tempura — engines and orchestrators are different layers;
      prior art, consumer-side, in
      ~/Code/gitlab.cee.redhat.com/fdupont/kv-topology-aware-router-extension),
      p24-tensor-parallelism-gpu (real multi-GPU TP — p5's done-claim is
      CPU-foundation only, cuda-oxide has no NCCL). Also re-audited:
      p22-multi-lora was marked done but the server has NO lora wiring
      (engine-side LoraRegistry only — now in_progress), and
      p22-radix-attention was a corrupted entry claiming done+pending
      (hash-based prefix cache only — now pending/0).
- [x] **Kani suite made actually runnable + 3 real bugs found (2026-11-05,
      local install of Kani 0.67.0 + rustup; CI's job had been silently red
      since 2026-06-15 because the gemm->pulp dep contains global ASM --
      fixed with -Z unstable-options --ignore-global-asm in kani.yml).**
      Bugs found by the now-running harnesses:
      1. **dequantize_row_q5_k had a REAL out-of-bounds**: the translation
         advanced qh_offset by 32 when is==4 and read qh[32..64] on the
         32-byte qh array -- any Q5_K model would panic at dequant (the
         path was never exercised: no Q5K cross-validation vectors, no Q5K
         model tests). Rewritten C-faithfully (u1/u2 bit masks shifting
         left 2 per 64-group, llama.cpp ggml-quants.c:1741). The Q5K
         vec_dot was always correct (different code path).
      2. **Tl2Config::for_dims accepted odd k**, but the two-LUT packing is
         pair-based (two_n = m*two_k/2): odd two_k (= k%96) with odd m
         silently truncates the last weight column. for_dims now rejects
         odd k (all real BitNet dims are even).
      3. Three harnesses used symbolic-length Vec allocation (allreduce,
         speculative x2) -- CBMC unwinds allocation internals without
         bound (2h / 10GB+ each). Rewritten with concrete lengths.
      Plus: Kani 0.67/CBMC 6.8 FALSE POSITIVE confirmed with a minimal
      probe -- the inserted nan-check reports "NaN on division" for
      1.0/(1.0 + f32::INFINITY) (IEEE: 1/+inf = 0). Affects any harness
      whose math divides by a saturating denominator (gelu_quick bounded
      to the polynomial range; fused_sqrelu bounded so the square stays
      finite). Same class: the full-block dequant_*_no_panic family used a
      symbolic u16 f16 scale -- a symbolic index into the 65536-entry
      f16_to_f32 table -- costing 12-20 GB of SAT per harness; the scales
      are now concrete (f16 1.0) with the payload fully symbolic (the
      indexing safety these harnesses exist for), bringing each from
      OOM/15min to 1-46s. The arbitrary-scale decode itself is covered by
      f16_to_f32_no_panic. Harness bug fixes: tq2/i2s/tl2 trit-range
      assertions claimed <= 1 but the decode of arbitrary bytes yields
      {-1,0,1,2} (encoder data is ternary -- covered by the roundtrip
      harnesses); q8_0_quant_in_range / tl2_per_tensor_quant_positive
      allowed subnormal amax (127/subnormal overflows to inf; production
      safe via saturating casts); kani_no_double_coverage evaluated s+1 on
      an unconstrained u32 (overflow before the assume). 11 harnesses in
      cutile_quantized_gemm.rs were behind the cutile feature gate and had
      NEVER run anywhere (cargo kani is featureless) -- the pure host-side
      block layout/dequant helpers moved to the ungated
      kernels/src/gpu_block_layout.rs (single definition, re-imported by
      the gated module). alibi_slopes_* and adaptive_p Kani harnesses
      (powf / exp-ln chains -- CBMC bit-blasts float transcendentals)
      converted to unit tests per the float-property convention.
      Local driver: /tmp/opencode/kani-batched.sh-style per-harness runs
      with ulimit + timeout (one exploding harness must not OOM the host).
      Second round (2026-11-06, full-suite drive to 198/220 then the rest):
      4. moe.rs kani_proofs was NESTED inside #[cfg(test)] mod tests --
         compiled out for cargo kani (which builds without cfg(test)), so
         the 4 router/softmax harnesses had never run anywhere. Unwedged,
         they revealed route() is softmax+Vec (unverifiable in Kani) --
         now a randomized unit test (router_properties_random).
      5. kani_splits_tile_exactly assertion was over-strong: trailing
         splits normalize to the EMPTY range (t0,t0) with t0 > kv_len --
         safe (identity partial), but "t1 <= kv_len" fails; the invariant
         is "within [0,kv_len] OR empty".
      6. q8_k_bsums_partition had a SYMBOLIC loop bound (nb = k/16 from
         kani::any k) -- CBMC unwound 5000+ iterations for a 16-trip loop;
         #[kani::unwind(20)] fixed it (15 min -> 0.15 s). Same lesson as
         the symbolic-length Vec bombs: bound loops explicitly.
      7. f16_to_f32_finite_for_normal: symbolic u16 index into the 64K
         table -> exhaustive unit test over all 65536 bits (stronger).
      8. The 9 sampler/norm full-execution harnesses (dry, xtc, typical,
         mirostat_v2, repetition_penalty x2, sampler/layernorm/rmsnorm
         full_execution) -- exp/ln/powf + Vec, several with symbolic-length
         Vecs -- converted to randomized/exhaustive unit tests per the
         float-property convention.
- [x] **NVFP4 complete on CPU (2026-11-04)** — was "enum + block size
      only" per the stale note; the reality was worse: partially translated
      paths with THREE divergent UE4M3 decoders and four real bugs, all
      fixed against llama.cpp master (ggml-quants.c/ggml-impl.h fetched and
      diffed): (1) `dequantize.rs` had a private arithmetic UE4M3 decode
      with a subnormal bug (m/8 * 2^-126 instead of man * 2^-9 — the exact
      bug class llama.cpp fixed in commit 4f232be: transformer amax/6 lands
      in the subnormal range) and WITHOUT the kvalues half-convention
      (x0.5), making `vec_dot_nvfp4_q8_0` and the AVX path return 2x the
      true dot; (2) `dequantize_row_nvfp4` used an adjacent-pairs nibble
      layout while both vec_dot variants (and llama.cpp) use SPLIT layout
      (byte j of a 16-elem sub-block: lo = element j, hi = element j+8);
      (3) `dequantize_row_mxfp4` had the same two bugs (adjacent-pairs +
      full-scale E8M0 decode instead of half); (4) `quants_llama_scalar`
      carried a third raw-convention UE4M3 copy. Now: single source of
      truth `helpers::ue4m3_to_f32` (build.rs-generated 256-entry table,
      exact ggml_ue4m3_to_fp32 port incl. subnormals, 0/0x7F -> 0,
      0xFF quirk), new `helpers::f32_to_ue4m3` encoder port (subnormal
      man*2^-9 path, 448 saturation, NaN/negative -> 0), new
      `quantize_row_nvfp4` (C-faithful: per-16 amax/6 -> UE4M3, quantize
      against the DECODED scale, best_index_mxfp4 over KVALUES_MXFP4,
      split packing). NOTE the format floor: without NVFP4's second-level
      per-tensor f32 global scale, sub-blocks with amax < 6*2^-9 ~= 0.0117
      encode to zero (llama.cpp single-level quantize_row_nvfp4_ref has
      the same floor; real checkpoints carry the tensor-level scale in a
      separate tensor). Verification: exhaustive 128-byte UE4M3
      decode->encode round-trip (all canonical bytes; encoder saturates at
      0x78..=0x7F by construction), quantize->dequant rel-L2 gates
      (0.30 normal / 0.60 subnormal-scale data), vec_dot vs f64
      dequantized-both-sides reference (catches the 2x class), generic vs
      scalar agreement, Kani harnesses (ue4m3_encode_no_panic,
      nvfp4_quantize_no_panic — CI), Verus module
      verus-verified/src/nvfp4.rs (UE4M3 field extraction + compose
      round-trip + encode range + split-layout index arithmetic + nibble
      packing round-trip + kvalues index safety; in run-verus.sh MODULES —
      no local Verus binary on the laptop, first run pending). GGUF type
      id 40 verified against upstream ggml.h. Follow-up: C
      cross-validation vectors via tests-ggml/gen_test_vectors.c next time
      a llama.cpp build is available; GPU NVFP4 kernels (Blackwell
      cvt.rn.satfinite.e2m1x2) remain open.
- [ ] Multimodal input (Bonsai 2 is text+image; no vision encoder path)
- [ ] TQ1_0 GPU kernel (CPU-only today)

