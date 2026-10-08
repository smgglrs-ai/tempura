---------------------------- MODULE BatchScheduler ----------------------------
(*
  Formal model of the tempura BatchScheduler (server/src/batch.rs).

  The scheduler manages three queues:
    - pending:   new requests waiting to be admitted (FIFO)
    - preempted: evicted requests waiting for re-admission (higher priority)
    - running:   active sequences holding a KV-cache slot

  Each running sequence holds exactly one KV slot.
  Preempted requests are retried before pending ones (they drain first).

  Invariants proved:
    KvUsageInvariant    -- kv_slots_used tracks Cardinality(running) exactly
    KvBoundInvariant    -- kv_slots_used never exceeds KV_SLOTS_MAX
    NoDoubleSchedule    -- no request id in both queue/preempted and running
    CompletedNeverRunning -- completed requests never re-appear in queues

  Liveness property:
    AllEventuallyComplete -- with fairness, every enqueued request eventually
                            completes (no starvation under the model's scheduler)
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
    completed,     \* Set of finished request ids (absorbing)
    kv_slots_used  \* Integer count of allocated KV slots (= |running|)

vars == <<queue, preempted, running, completed, kv_slots_used>>

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

-----------------------------------------------------------------------------
TypeOK ==
    /\ queue      \in Seq(REQ_IDS)
    /\ preempted  \in Seq(REQ_IDS)
    /\ running    \subseteq REQ_IDS
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
    /\ CompletedNeverRunning
    /\ BatchSizeBound

-----------------------------------------------------------------------------
\* Initial state: empty scheduler

Init ==
    /\ queue         = <<>>
    /\ preempted     = <<>>
    /\ running       = {}
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
    /\ UNCHANGED <<preempted, running, completed, kv_slots_used>>

(*
  Admit: move the head of preempted (if non-empty) or queue into running,
  provided a KV slot is free and the batch is not full.

  Models admit_new_requests: preempted is drained before pending.
*)
AdmitFromPreempted ==
    /\ preempted # <<>>
    /\ Cardinality(running) < MAX_BATCH_SIZE
    /\ FreeSlots > 0
    /\ LET r == Head(preempted)
       IN /\ running'       = running \cup {r}
          /\ preempted'     = Tail(preempted)
          /\ kv_slots_used' = kv_slots_used + 1
          /\ UNCHANGED <<queue, completed>>

AdmitFromQueue ==
    /\ preempted = <<>>
    /\ queue # <<>>
    /\ Cardinality(running) < MAX_BATCH_SIZE
    /\ FreeSlots > 0
    /\ LET r == Head(queue)
       IN /\ running'       = running \cup {r}
          /\ queue'         = Tail(queue)
          /\ kv_slots_used' = kv_slots_used + 1
          /\ UNCHANGED <<preempted, completed>>

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
    /\ (queue # <<>> \/ preempted # <<>>)
    /\ \E r \in running :
           /\ running'       = running \ {r}
           /\ preempted'     = <<r>> \o preempted
           /\ kv_slots_used' = kv_slots_used - 1
           /\ UNCHANGED <<queue, completed>>

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
           /\ UNCHANGED <<queue, preempted>>

-----------------------------------------------------------------------------
\* Terminal state: all requests completed, scheduler is idle.
\* This stuttering action prevents TLC from reporting deadlock when the
\* system has legitimately finished processing all requests.
Done ==
    /\ completed = REQ_IDS
    /\ queue = <<>>
    /\ preempted = <<>>
    /\ running = {}
    /\ UNCHANGED vars

Next ==
    \/ \E r \in REQ_IDS : Enqueue(r)
    \/ Admit
    \/ Preempt
    \/ Complete
    \/ Done

\* Weak fairness: Admit and Complete must eventually fire when continuously enabled.
\* This prevents the scheduler from indefinitely ignoring a ready-to-run request.
Fairness ==
    /\ WF_vars(Admit)
    /\ WF_vars(Complete)

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

=============================================================================
