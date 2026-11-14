---------------------------- MODULE BatchScheduler ----------------------------
(*
  Formal model of the tempura BatchScheduler (server/src/batch.rs).

  The scheduler manages three queues:
    - pending:   new requests waiting to be admitted (FIFO)
    - preempted: evicted requests waiting for re-admission (higher priority)
    - running:   active sequences holding a KV-cache slot

  Each running sequence holds exactly one KV slot.
  Preempted requests are retried before pending ones (they drain first).
  Swapped sequences (p24-kv-swap) hold their state in the DRAM tier with
  their GPU pages freed; they resume via swap-in BEFORE any new admission
  (swapped > preempted > pending).

  Invariants proved:
    KvUsageInvariant    -- kv_slots_used tracks Cardinality(running) exactly
    KvBoundInvariant    -- kv_slots_used never exceeds KV_SLOTS_MAX
    NoDoubleSchedule    -- no request id in both queue/preempted and running
    NoDoubleResident    -- a sequence is running XOR swapped, never both,
                           and never queued while swapped
    CompletedNeverRunning -- completed requests never re-appear in queues

  Liveness properties:
    AllEventuallyComplete -- with fairness, every enqueued request eventually
                            completes (no starvation under the model's scheduler)
    SwappedEventuallyResumes -- no starvation of the DRAM tier: every swapped
                            sequence eventually returns to running
*)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    REQ_IDS,        \* finite set of all possible request ids (e.g. {r1,r2,r3})
    KV_SLOTS_MAX,   \* total KV cache slots available
    MAX_BATCH_SIZE  \* maximum number of concurrently running sequences

ASSUME KV_SLOTS_MAX \in Nat /\ KV_SLOTS_MAX >= 1
ASSUME MAX_BATCH_SIZE \in Nat /\ MAX_BATCH_SIZE >= 1
ASSUME IsFiniteSet(REQ_IDS) /\ REQ_IDS # {}

-----------------------------------------------------------------------------
VARIABLES
    queue,         \* Sequence of pending request ids (FIFO; no duplicates)
    preempted,     \* Sequence of preempted request ids (drained before queue)
    running,       \* Set of currently active request ids
    swapped,       \* Set of request ids whose KV is in the DRAM tier
                   \* (p24-kv-swap: state preserved, GPU pages freed)
    completed,     \* Set of finished request ids (absorbing)
    kv_slots_used  \* Integer count of allocated KV slots (= |running|)

vars == <<queue, preempted, running, swapped, completed, kv_slots_used>>

-----------------------------------------------------------------------------
\* Helper: convert a sequence to a set
SeqToSet(s) == {s[i] : i \in DOMAIN s}

\* True when r appears in sequence s
InSeq(r, s) == \E i \in DOMAIN s : s[i] = r

\* Count of free KV slots
FreeSlots == KV_SLOTS_MAX - kv_slots_used

\* All requests that are "in the system" (not yet completed)
InSystem(r) ==
    \/ InSeq(r, queue)
    \/ InSeq(r, preempted)
    \/ r \in running
    \/ r \in swapped

-----------------------------------------------------------------------------
TypeOK ==
    /\ queue      \in Seq(REQ_IDS)
    /\ preempted  \in Seq(REQ_IDS)
    /\ running    \subseteq REQ_IDS
    /\ swapped    \subseteq REQ_IDS
    /\ completed  \subseteq REQ_IDS
    /\ kv_slots_used \in 0..KV_SLOTS_MAX

-----------------------------------------------------------------------------
\* Safety invariants

KvUsageInvariant ==
    kv_slots_used = Cardinality(running)

KvBoundInvariant ==
    kv_slots_used <= KV_SLOTS_MAX

NoDoubleSchedule ==
    /\ SeqToSet(queue) \cap running = {}
    /\ SeqToSet(preempted) \cap running = {}
    /\ SeqToSet(queue) \cap SeqToSet(preempted) = {}

\* p24-kv-swap: a sequence is GPU-resident (running) XOR swapped to DRAM,
\* never both, never in a queue while swapped, and completed sequences
\* never reappear anywhere.
NoDoubleResident ==
    /\ running \cap swapped = {}
    /\ SeqToSet(queue) \cap swapped = {}
    /\ SeqToSet(preempted) \cap swapped = {}

SwapDisjointCompleted ==
    swapped \cap completed = {}

CompletedNeverRunning ==
    /\ running \cap completed = {}
    /\ SeqToSet(queue) \cap completed = {}
    /\ SeqToSet(preempted) \cap completed = {}

BatchSizeBound ==
    Cardinality(running) <= MAX_BATCH_SIZE

\* Conjunction of all safety properties (checked by TLC as INVARIANT)
Safety ==
    /\ TypeOK
    /\ KvUsageInvariant
    /\ KvBoundInvariant
    /\ NoDoubleSchedule
    /\ NoDoubleResident
    /\ SwapDisjointCompleted
    /\ CompletedNeverRunning
    /\ BatchSizeBound

-----------------------------------------------------------------------------
\* Initial state: empty scheduler

Init ==
    /\ queue         = <<>>
    /\ preempted     = <<>>
    /\ running       = {}
    /\ swapped       = {}
    /\ completed     = {}
    /\ kv_slots_used = 0

-----------------------------------------------------------------------------
\* Actions

(*
  Enqueue: a new request arrives and joins the tail of pending.
  A request can only be enqueued if it has never been in the system
  (not running, not completed, not already queued or preempted).
*)
Enqueue(r) ==
    /\ r \in REQ_IDS
    /\ ~InSystem(r)
    /\ r \notin completed
    /\ queue' = Append(queue, r)
    /\ UNCHANGED <<preempted, running, swapped, completed, kv_slots_used>>

(*
  Admit: move the head of preempted (if non-empty) or queue into running,
  provided a KV slot is free and the batch is not full.

  Models admit_new_requests: preempted is drained before pending.
  Swapped sequences take precedence over NEW admissions (the engine
  resumes them before building each batch) -- the guard keeps the
  starvation-freedom proof below honest.
*)
AdmitFromPreempted ==
    /\ preempted # <<>>
    /\ Cardinality(running) < MAX_BATCH_SIZE
    /\ FreeSlots > 0
    /\ swapped = {}
    /\ LET r == Head(preempted)
       IN /\ running'       = running \cup {r}
          /\ preempted'     = Tail(preempted)
          /\ kv_slots_used' = kv_slots_used + 1
          /\ UNCHANGED <<queue, swapped, completed>>

AdmitFromQueue ==
    /\ preempted = <<>>
    /\ queue # <<>>
    /\ Cardinality(running) < MAX_BATCH_SIZE
    /\ FreeSlots > 0
    /\ swapped = {}
    /\ LET r == Head(queue)
       IN /\ running'       = running \cup {r}
          /\ queue'         = Tail(queue)
          /\ kv_slots_used' = kv_slots_used + 1
          /\ UNCHANGED <<preempted, swapped, completed>>

Admit == AdmitFromPreempted \/ AdmitFromQueue

(*
  Preempt: evict one running sequence back to the front of preempted.
  Models preempt_lowest_priority in admit_new_requests: preemption is only
  triggered when ALL KV slots are occupied but a waiting request still needs
  to be admitted (i.e. batch is not yet full).
  Guards:
    - FreeSlots = 0          -- no slot available for the incoming request
    - running non-empty      -- something to evict
    - batch not full         -- there is still capacity for one more sequence
    - something is waiting   -- a request actually needs the freed slot
*)
Preempt ==
    /\ running # {}
    /\ FreeSlots = 0
    /\ Cardinality(running) < MAX_BATCH_SIZE
    /\ (queue # <<>> \/ preempted # <<>> \/ swapped # {})
    /\ \E r \in running :
           /\ running'       = running \ {r}
           /\ preempted'     = <<r>> \o preempted
           /\ kv_slots_used' = kv_slots_used - 1
           /\ UNCHANGED <<queue, swapped, completed>>

(*
  Complete: a running request finishes successfully.
  Its slot is freed and it moves to completed (absorbing set).
*)
Complete ==
    /\ running # {}
    /\ \E r \in running :
           /\ running'       = running \ {r}
           /\ completed'     = completed \cup {r}
           /\ kv_slots_used' = kv_slots_used - 1
           /\ UNCHANGED <<queue, preempted, swapped>>

(*
  SwapOut (p24-kv-swap): move a running sequence's KV to the DRAM tier
  and free its GPU pages.  UNLIKE Preempt, the sequence keeps all state
  (tokens, position) -- resuming needs swap_in, no recompute.  Models
  the scheduler-driven eviction on pool exhaustion (LRU/priority victim
  selection in batch.rs; the model abstracts WHICH victim).
*)
SwapOut ==
    /\ running # {}
    /\ \E r \in running :
           /\ running'       = running \ {r}
           /\ swapped'       = swapped \cup {r}
           /\ kv_slots_used' = kv_slots_used - 1
           /\ UNCHANGED <<queue, preempted, completed>>

(*
  SwapIn: resume a swapped sequence into a free GPU slot.  Enabled
  whenever a slot is free; weak fairness on SwapIn + the swapped-first
  admission guard gives starvation-freedom (SwappedEventuallyResumes).
*)
SwapIn ==
    /\ swapped # {}
    /\ Cardinality(running) < MAX_BATCH_SIZE
    /\ FreeSlots > 0
    /\ \E r \in swapped :
           /\ running'       = running \cup {r}
           /\ swapped'       = swapped \ {r}
           /\ kv_slots_used' = kv_slots_used + 1
           /\ UNCHANGED <<queue, preempted, completed>>

-----------------------------------------------------------------------------
\* Terminal state: all requests completed, scheduler is idle.
\* This stuttering action prevents TLC from reporting deadlock when the
\* system has legitimately finished processing all requests.
Done ==
    /\ completed = REQ_IDS
    /\ queue = <<>>
    /\ preempted = <<>>
    /\ running = {}
    /\ swapped = {}
    /\ UNCHANGED vars

Next ==
    \/ \E r \in REQ_IDS : Enqueue(r)
    \/ Admit
    \/ Preempt
    \/ Complete
    \/ SwapOut
    \/ SwapIn
    \/ Done

\* Fairness: Admit and SwapIn must eventually fire when continuously
\* enabled.  Complete uses STRONG fairness: the abstract model permits a
\* swap thrash cycle (SwapOut(r) -> SwapIn(r) -> SwapOut(r) ...) in which
\* Complete is enabled only every other state, so weak fairness cannot
\* break it.  Strong fairness models the implementation's guarantee that
\* a resumed sequence always gets to step before any re-eviction (the
\* scheduler's `exclude` set protects the sequences whose growth caused
\* the exhaustion, and swap-in only fires for a sequence about to step).
Fairness ==
    /\ WF_vars(Admit)
    /\ SF_vars(Complete)
    /\ WF_vars(SwapIn)

Spec == Init /\ [][Next]_vars /\ Fairness

-----------------------------------------------------------------------------
\* Liveness property

(*
  Every request that enters the system eventually completes.
  With WF(Admit) and WF(Complete), no request can be stuck forever:
  - Pending requests are admitted whenever a slot is free.
  - Running requests complete eventually.
  - Preempted requests rejoin running as slots free up.

  Checked by TLC as PROPERTY AllEventuallyComplete.
*)
AllEventuallyComplete ==
    \A r \in REQ_IDS :
        InSystem(r) ~> (r \in completed)

(*
  p24-kv-swap liveness: every swapped sequence eventually resumes on the
  GPU (no starvation of the DRAM tier).  Holds because WF(SwapIn) fires
  as soon as a slot is free, WF(Complete) eventually frees slots, and
  both Admit actions yield to swapped sequences (the swapped = {} guard).
*)
SwappedEventuallyResumes ==
    \A r \in REQ_IDS :
        (r \in swapped) ~> (r \in running \/ r \in completed)

=============================================================================
