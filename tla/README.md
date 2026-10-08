# TLA+ Formal Specifications

Formal models of tempura's runtime state machines, checked with TLC.

## BatchScheduler

**File:** `BatchScheduler.tla` / `BatchScheduler.cfg`

Formal model of the batch scheduler (`server/src/batch.rs`).

### What it models

Three queues matching the Rust implementation:

| Queue | Rust field | Priority |
|-------|-----------|----------|
| `preempted` | `BatchScheduler::preempted` | Highest (drained first) |
| `queue` | `BatchScheduler::pending` | Normal |
| `running` | `BatchScheduler::active` | Currently executing |

Each running sequence holds exactly one KV-cache slot.

### Verified properties

**Safety invariants** (hold in every reachable state):

- `KvUsageInvariant` — `kv_slots_used` always equals `|running|`
- `KvBoundInvariant` — `kv_slots_used ≤ KV_SLOTS_MAX`
- `NoDoubleSchedule` — no request id appears in both `running` and `queue`/`preempted`
- `CompletedNeverRunning` — completed requests never re-enter any queue
- `BatchSizeBound` — `|running| ≤ MAX_BATCH_SIZE`

**Liveness property** (holds under weak fairness of Admit and Complete):

- `AllEventuallyComplete` — every enqueued request eventually completes

### Key modelling decisions

- `Preempt` is guarded by `FreeSlots = 0 ∧ |running| < MAX_BATCH_SIZE ∧ queue/preempted non-empty`, matching the real code: preemption only fires inside `admit_new_requests` when a slot must be freed to admit a waiting higher-priority request.
- `Done` is a stuttering action that prevents TLC from reporting deadlock when all requests have completed.
- Priority within preemption is non-deterministic in the model (any running request can be evicted); the invariants hold regardless of eviction choice.

### TLC model size

With `REQ_IDS = {r1, r2, r3}`, `KV_SLOTS_MAX = 2`, `MAX_BATCH_SIZE = 2`:

```
149 states generated, 77 distinct states found
Model checking completed. No error has been found.
```

### Running TLC

Requirements: Java ≥ 11, `tla2tools.jar`.

```bash
cd tla/

# Parse only (SANY)
java -cp ~/.local/share/tlaplus/tla2tools.jar tla2sany.SANY BatchScheduler.tla

# Full model check (safety + liveness)
tlc -config BatchScheduler.cfg -workers 4 BatchScheduler.tla

# Or without the wrapper script:
java -cp ~/.local/share/tlaplus/tla2tools.jar tlc2.TLC \
    -config BatchScheduler.cfg -workers 4 BatchScheduler.tla
```

Download `tla2tools.jar` from https://github.com/tlaplus/tlaplus/releases if not installed.
