------------------------ MODULE SpeculativeDecoding -------------------------
(*
  Formal model of tempura's speculative decoding draft/verify protocol.
  Implementation: engine/src/speculative.rs (SpeculativeDecoder::generate)

  Protocol structure (one round):
    1. DRAFTING   -- draft model generates up to MAX_DRAFT tokens, each
                     consuming one draft KV slot.
    2. VERIFYING  -- target model scores draft tokens sequentially; stops
                     at the first rejected token and appends a corrected one.
    3. ACCEPTING  -- non-deterministically accept k ∈ {1..draft_len+1} tokens.
                     The +1 is the bonus/corrected token: the implementation
                     guarantees len(accepted) ≥ 1 in every round (see line 203
                     and lines 164/178 in speculative.rs).
    4. DONE       -- accepted_count ≥ MAX_TOKENS.

  NOTE – Probabilistic acceptance: The actual acceptance criterion uses
  modified rejection sampling (target_prob / draft_prob ratio) to preserve
  the target model's output distribution. This is probabilistic and cannot
  be directly modelled in TLA+. We model it as a non-deterministic choice
  of k ∈ {1..draft_len+1}, capturing every possible acceptance outcome.
  Safety invariants are verified for all non-deterministic choices.

  KV-cache model: draft and target sessions maintain separate caches, but
  we model a unified pool of TOTAL_KV_SLOTS slots to bound total memory.
  draft_slots are freed (rollback) after every round; accepted_slots grow
  monotonically with accepted_count.
*)

EXTENDS Naturals, Sequences

CONSTANTS
    MAX_DRAFT,       \* max draft tokens per round  (e.g. 2)
    TOTAL_KV_SLOTS,  \* unified KV slot budget       (e.g. 8)
    MAX_TOKENS       \* generation budget             (e.g. 3)

ASSUME MAX_DRAFT \in Nat /\ MAX_DRAFT >= 1
ASSUME TOTAL_KV_SLOTS \in Nat /\ TOTAL_KV_SLOTS >= MAX_DRAFT + 2
ASSUME MAX_TOKENS \in Nat /\ MAX_TOKENS >= 1

-----------------------------------------------------------------------------
VARIABLES
    phase,           \* "Drafting" | "Verifying" | "Accepting" | "Done"
    draft_len,       \* draft tokens generated this round (0..MAX_DRAFT)
    accepted_count,  \* total tokens accepted and emitted (monotone)
    accepted_slots,  \* KV slots held by accepted sequence (target session)
    draft_slots      \* KV slots held by current drafts   (draft session)

vars == <<phase, draft_len, accepted_count, accepted_slots, draft_slots>>

Phases == {"Drafting", "Verifying", "Accepting", "Done"}

FreeSlots == TOTAL_KV_SLOTS - accepted_slots - draft_slots

-----------------------------------------------------------------------------
TypeOK ==
    /\ phase          \in Phases
    /\ draft_len      \in 0..MAX_DRAFT
    /\ accepted_count \in 0..MAX_TOKENS
    /\ accepted_slots \in 0..TOTAL_KV_SLOTS
    /\ draft_slots    \in 0..TOTAL_KV_SLOTS

-----------------------------------------------------------------------------
\* Safety invariants

\* Unified KV pool is never exceeded.
SlotConservation ==
    accepted_slots + draft_slots <= TOTAL_KV_SLOTS

\* Draft buffer never exceeds configured limit.
DraftBounded ==
    draft_len <= MAX_DRAFT

\* Draft tokens and draft slots are in sync.
DraftSlotsConsistent ==
    draft_slots = draft_len

\* Draft context is zeroed in terminal and inter-round states.
NoDraftAfterDone ==
    phase = "Done" => draft_len = 0

\* accepted_slots tracks accepted_count (one slot per accepted token,
\* plus one for the prompt token consumed at init).
AcceptedSlotsConsistent ==
    accepted_slots = accepted_count + 1

Safety ==
    /\ TypeOK
    /\ SlotConservation
    /\ DraftBounded
    /\ DraftSlotsConsistent
    /\ NoDraftAfterDone
    /\ AcceptedSlotsConsistent

-----------------------------------------------------------------------------
\* Initial state: prompt has been run through both models (1 slot each used
\* for the prompt KV entry in the target session); draft session is empty.
Init ==
    /\ phase          = "Drafting"
    /\ draft_len      = 0
    /\ accepted_count = 0
    /\ accepted_slots = 1   \* one slot: the prompt token in target session
    /\ draft_slots    = 0

-----------------------------------------------------------------------------
\* Actions

\* Generate one draft token (extend draft KV cache by 1).
DraftOne ==
    /\ phase = "Drafting"
    /\ draft_len < MAX_DRAFT
    /\ accepted_count < MAX_TOKENS           \* budget remains
    /\ FreeSlots >= 1                        \* room for one draft slot
    /\ draft_len'      = draft_len + 1
    /\ draft_slots'    = draft_slots + 1
    \* Transition to Verifying when draft buffer is full.
    /\ phase' = IF draft_len + 1 = MAX_DRAFT THEN "Verifying" ELSE "Drafting"
    /\ UNCHANGED <<accepted_count, accepted_slots>>

\* Optionally stop drafting early (EOS token, budget pressure).
\* Requires at least one draft token to have been generated.
DraftEarlyStop ==
    /\ phase = "Drafting"
    /\ draft_len >= 1
    /\ phase' = "Verifying"
    /\ UNCHANGED <<draft_len, accepted_count, accepted_slots, draft_slots>>

\* Target model has scored all draft tokens; move to Accepting phase.
VerifyAll ==
    /\ phase = "Verifying"
    /\ phase' = "Accepting"
    /\ UNCHANGED <<draft_len, accepted_count, accepted_slots, draft_slots>>

\* Accept k tokens (k ∈ {1..draft_len+1}).
\*
\* The implementation always produces len(accepted) >= 1 per round:
\*   - If any draft is rejected: accepted = prefix_accepted ++ [corrected_token]
\*   - If all drafts accepted:  accepted = draft_tokens ++ [bonus_token]
\* Hence k >= 1 is guaranteed; we model k ∈ {1..draft_len+1}.
\*
\* Effect:
\*   - accepted_count advances by k
\*   - accepted_slots grows by k (one slot per new token in target KV cache)
\*   - draft_slots freed (draft session rolled back)
\*   - draft_len reset to 0 (ready for next round)
Accept(k) ==
    /\ phase = "Accepting"
    /\ k \in 1..(draft_len + 1)
    /\ accepted_count + k <= MAX_TOKENS      \* within budget
    /\ accepted_slots + k <= TOTAL_KV_SLOTS  \* target session has room
    /\ accepted_count' = accepted_count + k
    /\ accepted_slots' = accepted_slots + k
    /\ draft_slots'    = 0
    /\ draft_len'      = 0
    /\ phase' = IF accepted_count + k >= MAX_TOKENS
                THEN "Done"
                ELSE "Drafting"

\* Clamp acceptance to the remaining budget when a full round would overshoot.
\* (Matches the budget guard at speculative.rs:95.)
AcceptPartialBudget ==
    /\ phase = "Accepting"
    /\ accepted_count < MAX_TOKENS
    /\ LET remaining == MAX_TOKENS - accepted_count
           max_k     == draft_len + 1
           k         == IF max_k <= remaining THEN max_k ELSE remaining
       IN
       /\ k >= 1
       /\ accepted_slots + k <= TOTAL_KV_SLOTS
       /\ accepted_count' = accepted_count + k
       /\ accepted_slots' = accepted_slots + k
       /\ draft_slots'    = 0
       /\ draft_len'      = 0
       /\ phase'          = "Done"

\* Stuttering is allowed (TLA+ convention); explicit Done is absorbing.
Stutter == phase = "Done" /\ UNCHANGED vars

-----------------------------------------------------------------------------
Next ==
    \/ DraftOne
    \/ DraftEarlyStop
    \/ VerifyAll
    \/ \E k \in 1..(MAX_DRAFT + 1) : Accept(k)
    \/ AcceptPartialBudget
    \/ Stutter

-----------------------------------------------------------------------------
\* Weak fairness: every continuously enabled action is eventually taken.
\* This models the scheduler always making forward progress.
Fairness ==
    /\ WF_vars(DraftOne)
    /\ WF_vars(DraftEarlyStop)
    /\ WF_vars(VerifyAll)
    /\ \A k \in 1..(MAX_DRAFT + 1) : WF_vars(Accept(k))
    /\ WF_vars(AcceptPartialBudget)

Spec == Init /\ [][Next]_vars /\ Fairness

-----------------------------------------------------------------------------
\* Temporal / liveness properties

\* accepted_count is monotonically non-decreasing.
AcceptedCountMonotone ==
    [][accepted_count' >= accepted_count]_vars

\* accepted_slots is monotonically non-decreasing.
AcceptedSlotsMonotone ==
    [][accepted_slots' >= accepted_slots]_vars

\* Generation always terminates: the phase eventually reaches Done.
\* Guaranteed because each Accepting step advances accepted_count by ≥ 1
\* and MAX_TOKENS is finite.
EventuallyDone == <>(phase = "Done")

=============================================================================
\* Modification History
\* Created for tempura speculative decoding verification.
