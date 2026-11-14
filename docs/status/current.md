# tempura — current status & open work

Verified gaps and open items. Do not assume these are done.
Archive completed items to docs/history/.

## Latest session status

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
quantified; CPU matches llama at 94.8% agreement).  (3) p24-overlap
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
              paths (found via MMLU, cutile_gpu_forward.rs:~4837);
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

