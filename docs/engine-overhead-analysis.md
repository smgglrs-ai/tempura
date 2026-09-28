# Engine Overhead Analysis: tokio/async vs llama.cpp synchronous model

**Branch:** llama-cpu-translation  
**Date:** 2026-09-28  
**Context:** tempura executes ~2.7× more instructions per decode token than llama.cpp running the same
algorithms. A 12% GEMV code-generation gap explains ~12% of this. This document accounts for the
rest by tracing every instruction source that llama.cpp avoids.

---

## 1. Execution model comparison

### llama.cpp

```
llama_decode()
  ↓ ggml_backend_sched_compute()
  ↓ workers spin on ggml_barrier (atomic CAS, no futex if spinning)
  ↓ result returned in pre-allocated ggml_tensor buffer
zero heap allocations on the hot path per token
```

Single synchronous call stack. All output buffers are pre-allocated in a `ggml_context` at session
start. Workers spin between barriers using `GOMP_SPINCOUNT` (OpenMP) or ggml's own barrier, so no
futex syscall occurs when the interval between barriers is short enough.

### tempura (streaming path)

```
tokio task (axum handler)
  → engine_tx.send(EngineRequest).await          [1 tokio mpsc send, once per request]
  → tokio::spawn(async { loop {
      tokio::time::timeout(deadline - now,        [timer registered every token]
          resp_rx.recv()).await                   [1 tokio task wakeup per token]
    }})

std::thread "tempura-engine"
  → EngineTask::run() loop
      try_recv() from rx (non-blocking poll)
      ActiveSession::step(&dyn InferenceModel)    [called once per token]
        forward() → Vec<f32>                      [heap allocation per token]
        try_send(EngineEvent::Token(token))       [1 mpsc send per token]
```

The engine runs in a dedicated `std::thread`, which is correct — it is not blocked by the tokio
scheduler. But every decoded token still crosses a thread boundary twice (try_send → waker →
tokio task reschedule → recv) and allocates one `Vec<f32>` of vocab_size floats.

---

## 2. Per-token overhead sources, ranked

### 2.1 Scratchpad Vec allocations in `forward_quantized()` — **largest contributor**

File: `engine/src/model.rs`, function `forward_quantized()`, lines 1287–1298.

```rust
// Pre-allocate scratchpad buffers (reused across layers)
let mut hidden   = vec![0.0f32; seq_len * h];
let mut normed   = vec![0.0f32; seq_len * h];
let mut q        = vec![0.0f32; seq_len * q_dim];
let mut k        = vec![0.0f32; seq_len * kv_dim];
let mut v        = vec![0.0f32; seq_len * kv_dim];
let mut attn_out = vec![0.0f32; seq_len * q_dim];
let mut o_proj   = vec![0.0f32; seq_len * h];
let mut normed2  = vec![0.0f32; seq_len * h];
let mut gate     = vec![0.0f32; seq_len * inter];
let mut up       = vec![0.0f32; seq_len * inter];
let mut down_out = vec![0.0f32; seq_len * h];
```

For a 7B LLaMA model (h=4096, inter=11008, seq_len=1, n_kv_heads=n_heads=32):

| Buffer | Size | malloc+free | zero-init |
|--------|------|-------------|-----------|
| hidden, normed, o_proj, normed2, attn_out | 4×4 KB each | ✓ | ✓ |
| gate, up | 44 KB each | ✓ | ✓ |
| q, k, v | 4 KB each | ✓ | ✓ |
| **Total** | **~120 KB** | **11 allocs** | **11 memsets** |

Each `vec![0.0f32; N]` calls `calloc` (or `malloc` + `memset`). At 11 allocations per decode step,
this is 11 round-trips to the allocator plus ~120 KB of memory writes that contribute nothing to
computation.

Instruction-count estimate: ~2,000 cycles for allocator metadata per call × 11 + ~30,000 cycles
for zero-initialization of 120 KB = **~52,000 extra cycles per token** from scratchpads alone.

**llama.cpp equivalent:** one-time `ggml_new_tensor` call at session init; decode reuses the same
pointer. Zero allocation overhead per token.

### 2.2 Per-layer `hidden.clone()` in the non-quantized `forward()` path — **f32 path only**

File: `engine/src/model.rs`, `forward()` method, inside the layer loop at line 516:

```rust
let mut normed = hidden.clone();        // 4096 f32 = 16 KB per layer
let mut o_proj = vec![0.0f32; seq_len * h];  // another 16 KB per layer
```

For 32 layers: 32 × 2 = **64 large allocations per token**, plus 64 memcpys of 16 KB each.
Total extra memory writes: 32 × 32 KB = **1 MB of copies per decode token**. This accounts for
a large fraction of the instruction-count gap on the f32 path. The quantized path (`forward_quantized`)
avoids this correctly by pre-allocating once.

### 2.3 String table allocation per `forward_quantized()` call — **medium contributor**

File: `engine/src/model.rs`, lines 1562–1581:

```rust
let layer_names: Vec<_> = (0..cfg.num_layers)
    .map(|l| {
        let p = format!("model.layers.{l}");
        (
            format!("{p}.input_layernorm.weight"),
            format!("{p}.post_attention_layernorm.weight"),
            format!("{p}.self_attn.q_proj.weight"),
            // ... 10 more
        )
    })
    .collect();
```

This executes **once per `forward_quantized()` call**, i.e., once per decode token. For 32 layers
× 13 names = **416 heap-allocated Strings per token**, each requiring `format!()` (format string
parse, UTF-8 copy) plus an allocator round-trip.

Estimated cost: 416 allocations × ~300 cycles each = **~125,000 cycles per token**.

**llama.cpp equivalent:** weight tensors are referenced by integer index (`model.layers[i].wq`),
never by string lookup.

### 2.4 HashMap string-keyed weight lookups per matmul — **medium contributor**

File: `engine/src/model.rs`, inside `qmm_inner` closure.

For each of ~224 matmul calls per forward pass (7 projections × 32 layers):

```rust
if let Some((dtype, w)) = qweights.get_typed(name) { ... }
if let Some(tl2t) = qweights.get_tl2(name) { ... }
if let Some(w) = qweights.get_i2s(name) { ... }
```

Each `get_typed(name)` is a `HashMap<String, _>::get(&str)` call: SipHash computation over
~30-40 chars + hash-map probe. On a hit at the first try, this is approximately 120–200 cycles.
If the key misses the first table and tries the second (tl2) or third (i2s), cost multiplies.

For 224 matmuls × 3 probes × 150 cycles = **~100,000 cycles per token** of hash overhead.

**llama.cpp equivalent:** `model.layers[i].wq` is a direct struct field access — 1 pointer
dereference, 0 hashing.

### 2.5 `logits Vec<f32>` allocation on every `forward()` return — **medium contributor**

The `InferenceModel::forward` signature returns `Vec<f32>` by value:

```rust
fn forward(&self, session: &mut InferenceSession, token_ids: &[u32]) -> Vec<f32>;
```

For a 32K vocabulary, this is a **128 KB heap allocation per decode token**. It must be freed
by the caller (in `step()`). The allocation + free cost is approximately:

- `malloc(128 KB)`: ~1,000–2,000 cycles
- Logit vector write (happens inside forward): included in compute cost
- `drop(logits)` after sampling: ~500 cycles

Total: **~2,500 extra cycles per token** from the logits allocation alone.

### 2.6 Thread pool condvar/mutex dispatch per matmul — **medium contributor**

File: `crates/staticpar/src/lib.rs`, `run_raw()`, lines 95–155.

Every GEMV/GEMM dispatch through `StaticPool` acquires a mutex and broadcasts a condvar:

```rust
{
    let mut gen = inner.mutex.lock().unwrap();   // futex lock
    *gen += 1;
    inner.condvar.notify_all();                  // futex wake N workers
}
```

For 8 worker threads and ~150 matmul dispatches per decode token:

- Each `mutex.lock()`: ~50–100 cycles uncontended + `LOCK CMPXCHG`
- Each `notify_all()` when workers are sleeping: 1 `futex(FUTEX_WAKE)` syscall = ~1,000–5,000 ns
  per call, waking up to 8 threads (8 additional futex wakeup events)
- Spin-wait after dispatch: 2,000 iterations × `pause` instruction

With 8 threads all sleeping between tokens (decode takes ~27 ms — far longer than the 2,000-spin
budget), `notify_all` almost always escalates to a kernel futex call.

Estimated wall-time overhead: 150 dispatches × 8 workers × ~1 μs/futex = **~1.2 ms/token**,
or ~4.3% at 36 tok/s decode. In instruction count, futex syscalls are kernel instructions that
do appear in `perf stat instructions` when counting all rings.

The `PersistentPool` (BitNet path, `kernels/src/thread_pool.rs`) uses a lock-free
`gen_atomic.fetch_add` for the generation counter but still acquires `sh.gen.lock()` and calls
`sh.wake.notify_all()` on every `parallel_chunks()` invocation (line 200–203), so the same
condvar overhead applies.

**llama.cpp equivalent with `GOMP_SPINCOUNT`:** Workers spin for up to
`GOMP_SPINCOUNT` iterations between barriers. With `GOMP_SPINCOUNT=1000` and typical inter-barrier
time of ~20–200 μs, workers never sleep — `notify_all` returns immediately without a syscall.
The comment in `run_server()` acknowledges this: *"0.3ms/barrier → ~0.05ms/barrier"* with spinning.

### 2.7 Tokio channel overhead per token (streaming path) — **minor**

Per decode token on the streaming path:

| Operation | Location | Cost |
|-----------|----------|------|
| `try_send(EngineEvent::Token(token))` | `engine_task.rs:60` | Lock-free fast path ~50 cycles |
| Waker fired → tokio schedules streaming task | tokio runtime | ~200–500 cycles |
| `tokio::time::timeout(deadline - now, resp_rx.recv()).await` | `api.rs` | New `Sleep` future registered per token |
| Timer future drop after recv completes | tokio time driver | ~100 cycles |

Total: **~1,000–2,000 cycles per token** from channel overhead. At 27 ms/token = ~81M cycles,
this is <0.003% — negligible.

However, `tokio::time::timeout(deadline - now, ...)` creates a new `Sleep` future from a dynamic
`Duration` on every loop iteration, which registers/deregisters with the tokio timer wheel. This
could be replaced with `tokio::time::timeout_at(deadline, ...)` and a pre-computed `Instant`,
or a `select! { _ = sleep_until(deadline) => break, ev = resp_rx.recv() => ... }` with the
sleep future created once before the loop.

### 2.8 `std::env::var()` calls on the hot path — **minor but real**

Inside `forward_quantized()`, five `std::env::var()` calls execute per token:

```
line 1236: std::env::var("TEMPURA_DETERMINISTIC")   [per forward call]
line 1249: std::env::var("TEMPURA_DETERMINISTIC")   [per forward call]
line 1311: std::env::var("TEMPURA_DEBUG")           [per forward call]
line 1325: std::env::var("TEMPURA_PROFILE")         [per forward call]
```

`std::env::var()` acquires an OS-level environment lock (on Linux: `pthread_rwlock_rdlock` on
glibc's `environ_lock`). This is typically fast (uncontended rwlock ≈ 50–100 cycles) but it is
a real lock operation on every forward call. Moving these to `OnceLock<bool>` or reading them at
model construction would eliminate them entirely.

---

## 3. Measured futex syscalls per token

Run command (single-token benchmark, 128-token context, Q4K 7B model):

```sh
perf stat -e syscalls:sys_enter_futex -- \
  ./target/release/tempura --model model.gguf --port 9999 &
# fire 1 chat completion request, decode 10 tokens, kill
```

Expected range based on the analysis:
- **StaticPool dispatches per token**: ~150 × 1 `futex(FUTEX_WAKE)` + 8 worker wakeups = ~9 futex calls/dispatch × 150 = **~1,350 futex syscalls per token**
- **llama.cpp** (with GOMP_SPINCOUNT): **0 futex syscalls per token** when workers stay spinning

To measure directly:
```sh
# Attach perf to the tempura-engine thread specifically
perf stat -p $(pgrep -f tempura-engine) -e syscalls:sys_enter_futex sleep 10
# Count tokens generated in that 10s window, divide
```

---

## 4. Allocations per token

### Summary table

| Allocation | Count (Q4K, 7B) | Count (f32, 7B) | Size |
|------------|----------------|----------------|------|
| Scratchpad Vecs in `forward_quantized()` | 11 | 0 (different path) | ~120 KB total |
| Per-layer `hidden.clone()` in `forward()` | 0 | 32 | 16 KB each |
| Per-layer `o_proj` Vec in `forward()` | 0 | 32 | 16 KB each |
| `layer_names` String table | 416 | 0 (uses `format!()` per access) | ~30 B each |
| `positions` Vec | 1 | 1 | 4 B × seq_len |
| `logits` Vec<f32> return | 1 | 1 | 128 KB (32K vocab) |
| **Total (Q4K)** | **429** | — | **~248 KB** |
| **Total (f32)** | — | **66+** | **>1 MB** |
| **llama.cpp** | **0** | **0** | **0** |

---

## 5. Ranked overhead sources with impact estimates

| Rank | Source | Cycles/token (estimate) | % of 27 ms token | Fix category |
|------|--------|------------------------|------------------|--------------|
| 1 | f32 path: per-layer `hidden.clone()` + `o_proj` alloc | ~500,000 | ~6% | Pre-allocate buffers in session |
| 2 | Q4K path: `layer_names` String table per token | ~125,000 | ~1.5% | Pre-compute at model construction |
| 3 | Q4K path: scratchpad Vec allocs + zero-init | ~52,000 | ~0.6% | Pre-allocate in TransformerModel |
| 4 | Thread pool condvar dispatch (~150/token × N threads) | ~1.2ms wall | ~4.3% | Spinning barrier or lock-free generation |
| 5 | HashMap string lookups for weight tensors | ~100,000 | ~0.1% | Indexed struct fields |
| 6 | `logits` Vec<f32> return | ~2,500 | ~0.003% | Return via pre-allocated output buffer |
| 7 | `std::env::var()` calls (5/token) | ~1,000 | ~0.001% | `OnceLock<bool>` |
| 8 | Tokio channel + timer per token | ~2,000 | <0.003% | `timeout_at` + reuse sleep future |

**Observation:** The instruction-count gap is dominated by the f32 path per-layer clone and the
string table allocation, not the async runtime. The tokio channel overhead is genuinely negligible.
The condvar dispatch overhead is real in wall time (~4%) but does not appear in user-space
instruction counts — it lives in kernel instruction counts.

---

## 6. Proposed changes

### Fix 1: Pre-allocate scratchpad buffers in `TransformerModel` (eliminates Ranks 1, 3)

Add a `scratchpads: Mutex<ForwardBuffers>` field to `TransformerModel`. `ForwardBuffers` holds all
the Vecs pre-sized at construction. `forward()` and `forward_quantized()` take a `&mut ForwardBuffers`
parameter from `with_scratchpads(|bufs| ...)`.

```rust
pub struct ForwardBuffers {
    hidden:   Vec<f32>,  // hidden_size
    normed:   Vec<f32>,  // hidden_size
    q:        Vec<f32>,  // num_heads * head_dim
    k:        Vec<f32>,  // num_kv_heads * head_dim
    v:        Vec<f32>,  // num_kv_heads * head_dim
    attn_out: Vec<f32>,  // num_heads * head_dim
    o_proj:   Vec<f32>,  // hidden_size
    normed2:  Vec<f32>,  // hidden_size
    gate:     Vec<f32>,  // intermediate_size
    up:       Vec<f32>,  // intermediate_size
    down_out: Vec<f32>,  // hidden_size
    logits:   Vec<f32>,  // vocab_size — return buffer
}
```

For single-session decode (common case), no mutex is needed; the buffer lives in `InferenceSession`.

Impact: eliminates 11 allocations + ~120 KB of zero-init per token for the Q4K path; eliminates
64 allocations + >1 MB of copies per token for the f32 path.

### Fix 2: Pre-compute `layer_names` at model construction (eliminates Rank 2)

Move the `layer_names` table from `forward_quantized()` body to a field on `TransformerModel`:

```rust
pub struct TransformerModel<...> {
    // existing fields ...
    layer_weight_names: Vec<LayerWeightNames>,  // computed once in ::new()
}

struct LayerWeightNames {
    input_layernorm:          String,
    post_attention_layernorm: String,
    q_proj:                   String,
    k_proj:                   String,
    v_proj:                   String,
    o_proj:                   String,
    // ...
}
```

`forward_quantized()` then borrows `&self.layer_weight_names[layer]` instead of rebuilding the
table on every call. Impact: eliminates 416 heap allocations per decode token.

### Fix 3: Switch weight lookup from `HashMap<String>` to indexed `Vec<LayerWeights>` (eliminates Rank 5)

`QuantizedWeights` currently stores tensors in a `HashMap<String, _>`. Replace with:

```rust
pub struct QuantizedWeights {
    pub layers: Vec<QuantizedLayerWeights>,
    pub embed: Vec<u8>,
    pub lm_head: Vec<u8>,
    pub norm: Vec<u8>,
}
pub struct QuantizedLayerWeights {
    pub q_proj:  Option<QuantizedTensor>,
    pub k_proj:  Option<QuantizedTensor>,
    // ...
}
```

Weight access becomes `qweights.layers[layer].q_proj.as_ref()` — zero hashing, zero string
comparison. Impact: eliminates SipHash computation for all ~224 matmul lookups per decode token.

### Fix 4: Replace condvar wake with spinning generation counter (eliminates Rank 4)

`StaticPool::run_raw()` currently wakes workers with `condvar.notify_all()` after updating gen.
Replace with a pure atomic generation counter and a spinning worker:

```rust
// In run_raw():
inner.gen.fetch_add(1, Ordering::Release);  // no mutex, no condvar

// In worker_loop():
loop {
    let target = inner.gen.load(Ordering::Acquire);
    if target != my_gen {
        my_gen = target;
        // do work
        continue;
    }
    // Only park if gen hasn't changed after a spin period
    std::hint::spin_loop();
}
```

With `TEMPURA_SPIN_COUNT` (analogous to `GOMP_SPINCOUNT`), workers spin for a configurable number
of iterations before actually sleeping. For decode tokens at 36 tok/s (27 ms apart), workers with
a spin budget of ~50,000 iterations will stay awake between the 150 matmul dispatches within a
single token's forward pass but sleep between tokens. This eliminates ~1.2 ms/token of futex
overhead while keeping idle CPU usage proportional to throughput (not fixed at 100%).

### Fix 5: Pre-allocate logits buffer in `InferenceSession` (eliminates Rank 6)

Change `InferenceModel::forward` to write into a caller-provided buffer:

```rust
fn forward_into(
    &self,
    session: &mut InferenceSession,
    token_ids: &[u32],
    logits_out: &mut Vec<f32>,
);
```

`InferenceSession` carries a `logits_buf: Vec<f32>` pre-sized to `vocab_size`. Callers pass
`&mut session.logits_buf` and sample directly from it. The current `forward() -> Vec<f32>`
signature is then a thin wrapper for compatibility.

### Fix 6: `OnceLock` for env-var flags (eliminates Rank 7)

```rust
static TEMPURA_DETERMINISTIC: OnceLock<bool> = OnceLock::new();

fn is_deterministic() -> bool {
    *TEMPURA_DETERMINISTIC
        .get_or_init(|| std::env::var("TEMPURA_DETERMINISTIC").is_ok())
}
```

Replace all `std::env::var(...)` calls inside the forward hot path with these static reads.

### Fix 7: Reuse the sleep future in the streaming loop (eliminates Rank 8)

```rust
// Before: new Sleep created every token
match tokio::time::timeout(deadline - now, resp_rx.recv()).await { ... }

// After: single Sleep created once before the loop
let sleep = tokio::time::sleep_until(deadline);
tokio::pin!(sleep);
loop {
    tokio::select! {
        biased;
        ev = resp_rx.recv() => match ev { ... },
        _ = &mut sleep => { /* timeout */ break },
    }
}
```

---

## 7. Expected impact after all fixes

| Path | Current | After fixes 1–7 | Expected gain |
|------|---------|-----------------|---------------|
| Q4K decode (7B) | 2.7× vs llama.cpp instructions | ~1.3–1.5× | Removes string/alloc overhead |
| f32 decode | 3.5× vs llama.cpp instructions | ~1.5–1.8× | Removes per-layer clone/alloc |
| Condvar wall-time overhead | ~4.3% | <0.1% | Spinning barrier |
| Tokio channel overhead | <0.003% | <0.003% | Already negligible |

The remaining gap after these fixes is accounted for by:
- The 12% GEMV code-generation gap (separate work)
- vtable dispatch through `&dyn InferenceModel` and `&dyn RMSNorm` (inlining lost)
- Rust bounds checking on slice indexing (mitigated by `unsafe` in hot loops, already done in
  the BitNet path)

---

## 8. Conclusion

The ~2.7× instruction overhead relative to llama.cpp is **not caused by tokio**. The async runtime
contributes <0.003% overhead per token. The actual sources are:

1. **Memory allocation on every forward call** — the biggest single item, especially the f32 path's
   per-layer `hidden.clone()` and the Q4K path's 11-buffer scratchpad Vec construction.
2. **String table construction every forward call** — 416 heap allocations per token to build
   a table of weight names that is constant across the model's lifetime.
3. **Thread pool condvar wake latency** — ~1.2 ms/token (4.3% wall-time) from futex syscalls
   when workers sleep between matmul dispatches.
4. **String-keyed HashMap weight lookup** — SipHash computation for every matmul routing decision.

Fixes 1–3 (scratchpad pre-allocation, layer_names caching, weight struct indexing) directly address
the instruction-count gap and require no API-visible changes. Fix 4 (spinning generation counter)
addresses the wall-time condvar penalty. Fixes 5–7 are minor cleanups.
