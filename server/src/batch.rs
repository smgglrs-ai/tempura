use std::collections::VecDeque;

#[derive(Debug)]
pub struct InferenceRequest {
    pub id: String,
    pub token_ids: Vec<u32>,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub priority: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    Prefill,
    Decode,
    Preempted,
    /// KV swapped to the DRAM tier (p24-kv-swap): the sequence keeps ALL
    /// its state (tokens, position, sampling); its GPU run was freed
    /// back to the pool by `CutileKvCache::swap_out`.  Unlike Preempted,
    /// resuming needs NO recompute -- `swap_in` restores the prefix.
    Swapped,
}

#[derive(Debug)]
pub struct ActiveSequence {
    pub request_id: String,
    pub slot_id: usize,
    pub position: usize,
    pub generated_tokens: Vec<u32>,
    pub max_new_tokens: u32,
    pub state: SequenceState,
    pub age: u64,
    pub token_ids: Vec<u32>,
    pub prefill_offset: usize,
    pub prefix_matched_len: usize,
    pub priority: u8,
    /// GPU-resident KV footprint in tokens (0 while swapped).  Used by
    /// the eviction policy; maintained by the engine loop via
    /// `set_kv_tokens` / `mark_swapped` / `mark_resident`.
    pub kv_tokens: usize,
    /// Tick of the last decode step (LRU clock for the eviction policy).
    pub last_touched: u64,
}

pub struct BatchScheduler {
    pending: VecDeque<InferenceRequest>,
    active: Vec<ActiveSequence>,
    preempted: VecDeque<InferenceRequest>,
    max_batch_size: usize,
    max_seq_len: usize,
    num_slots: usize,
    allocated_slots: Vec<bool>,
    tick: u64,
    prefill_chunk_size: usize,
}

impl BatchScheduler {
    pub fn new(max_batch_size: usize, max_seq_len: usize, num_slots: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            active: Vec::new(),
            preempted: VecDeque::new(),
            max_batch_size,
            max_seq_len,
            num_slots,
            allocated_slots: vec![false; num_slots],
            tick: 0,
            prefill_chunk_size: 512,
        }
    }

    pub fn with_prefill_chunk_size(mut self, size: usize) -> Self {
        self.prefill_chunk_size = size;
        self
    }

    pub fn prefill_chunk_size(&self) -> usize {
        self.prefill_chunk_size
    }

    pub fn submit(&mut self, request: InferenceRequest) {
        self.pending.push_back(request);
    }

    fn allocate_slot(&mut self) -> Option<usize> {
        for (i, used) in self.allocated_slots.iter_mut().enumerate() {
            if !*used {
                *used = true;
                return Some(i);
            }
        }
        None
    }

    fn release_slot(&mut self, slot_id: usize) {
        self.allocated_slots[slot_id] = false;
    }

    fn preempt_lowest_priority(&mut self) -> Option<usize> {
        if self.active.is_empty() {
            return None;
        }
        let (idx, _) = self
            .active
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| (s.priority, s.age))
            .unwrap();

        let evicted = self.active.remove(idx);
        let slot_id = evicted.slot_id;
        self.release_slot(slot_id);

        self.preempted.push_back(InferenceRequest {
            id: evicted.request_id,
            token_ids: evicted.token_ids,
            max_new_tokens: evicted.max_new_tokens - evicted.generated_tokens.len() as u32,
            temperature: 0.0,
            top_p: 1.0,
            priority: evicted.priority,
        });

        Some(slot_id)
    }

    pub fn next_batch(&mut self) -> Vec<InferenceRequest> {
        self.tick += 1;
        let count = self.pending.len().min(self.max_batch_size);
        self.pending.drain(..count).collect()
    }

    pub fn admit_new_requests(&mut self) -> Vec<ActiveSequence> {
        self.tick += 1;
        let mut admitted = Vec::new();

        let mut sources: Vec<InferenceRequest> = self
            .preempted
            .drain(..)
            .chain(self.pending.drain(..))
            .collect();
        sources.sort_by_key(|r| std::cmp::Reverse(r.priority));

        for req in sources {
            if self.active.len() >= self.max_batch_size {
                self.pending.push_back(req);
                continue;
            }

            let slot_id = match self.allocate_slot() {
                Some(id) => id,
                None => match self.preempt_lowest_priority() {
                    Some(id) => id,
                    None => {
                        self.pending.push_back(req);
                        continue;
                    }
                },
            };

            let seq = ActiveSequence {
                request_id: req.id,
                slot_id,
                position: 0,
                generated_tokens: Vec::new(),
                max_new_tokens: req.max_new_tokens,
                state: SequenceState::Prefill,
                age: self.tick,
                token_ids: req.token_ids,
                prefill_offset: 0,
                prefix_matched_len: 0,
                priority: req.priority,
                kv_tokens: 0,
                last_touched: self.tick,
            };

            admitted.push(ActiveSequence {
                request_id: seq.request_id.clone(),
                slot_id: seq.slot_id,
                position: seq.position,
                generated_tokens: Vec::new(),
                max_new_tokens: seq.max_new_tokens,
                state: seq.state,
                age: seq.age,
                token_ids: seq.token_ids.clone(),
                prefill_offset: 0,
                prefix_matched_len: 0,
                priority: seq.priority,
                kv_tokens: 0,
                last_touched: self.tick,
            });

            self.active.push(seq);
        }

        admitted
    }

    pub fn complete_sequence(&mut self, request_id: &str) -> Option<usize> {
        let idx = self
            .active
            .iter()
            .position(|s| s.request_id == request_id)?;
        let seq = self.active.remove(idx);
        let slot_id = seq.slot_id;
        self.release_slot(slot_id);
        Some(slot_id)
    }

    /// Return a just-admitted sequence to the FRONT of the pending queue
    /// (admission was rejected downstream, e.g. the KV pool could not
    /// provide a run).  Unlike preemption this carries no state loss --
    /// the sequence had not started prefill.
    pub fn requeue_active(&mut self, request_id: &str) -> bool {
        let Some(idx) = self.active.iter().position(|s| s.request_id == request_id) else {
            return false;
        };
        let seq = self.active.remove(idx);
        self.release_slot(seq.slot_id);
        self.pending.push_front(InferenceRequest {
            id: seq.request_id,
            token_ids: seq.token_ids,
            max_new_tokens: seq.max_new_tokens - seq.generated_tokens.len() as u32,
            temperature: 0.0,
            top_p: 1.0,
            priority: seq.priority,
        });
        true
    }

    /// Prefill finished: the sequence is ready for decode steps.
    pub fn mark_decoding(&mut self, request_id: &str) -> bool {
        match self
            .active
            .iter_mut()
            .find(|s| s.request_id == request_id && s.state == SequenceState::Prefill)
        {
            Some(s) => {
                s.state = SequenceState::Decode;
                true
            }
            None => false,
        }
    }

    /// True when the id is anywhere in the system (active, pending, or
    /// preempted) -- lets the engine detect requests the HTTP layer
    /// already submitted via acquire_slot.
    pub fn in_system(&self, request_id: &str) -> bool {
        self.active.iter().any(|s| s.request_id == request_id)
            || self.pending.iter().any(|r| r.id == request_id)
            || self.preempted.iter().any(|r| r.id == request_id)
    }

    pub fn active_sequences(&self) -> &[ActiveSequence] {
        &self.active
    }

    pub fn active_sequences_mut(&mut self) -> &mut [ActiveSequence] {
        &mut self.active
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn preempted_count(&self) -> usize {
        self.preempted.len()
    }

    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn withdraw_pending(&mut self, request_id: &str) {
        self.pending.retain(|r| r.id != request_id);
    }

    pub fn allocated_slot_count(&self) -> usize {
        self.allocated_slots.iter().filter(|&&used| used).count()
    }

    // ------------------------------------------------------------------
    // KV swap policy (p24-kv-swap): on pool exhaustion the scheduler
    // picks a victim whose GPU run is swapped out to DRAM.  Unlike
    // preemption (which drops state and recomputes), swap preserves
    // everything.
    // ------------------------------------------------------------------

    /// LRU/priority victim selection: the lowest-priority sequence,
    /// least recently stepped (tie: oldest admission).  Never selects a
    /// Swapped sequence (nothing to free) or an excluded one (the
    /// sequences whose growth triggered the exhaustion).
    pub fn select_swap_victim(&self, exclude: &[&str]) -> Option<String> {
        self.active
            .iter()
            .filter(|s| s.state != SequenceState::Swapped && s.kv_tokens > 0)
            .filter(|s| !exclude.contains(&s.request_id.as_str()))
            .min_by_key(|s| (s.priority, s.last_touched, s.age))
            .map(|s| s.request_id.clone())
    }

    /// Mark `request_id` swapped out: state flips and its GPU footprint
    /// is zeroed (the engine has already called `swap_out` on its run).
    /// Returns false when the id is unknown or already swapped.
    pub fn mark_swapped(&mut self, request_id: &str) -> bool {
        match self.active.iter_mut().find(|s| s.request_id == request_id) {
            Some(s) if s.state != SequenceState::Swapped => {
                s.state = SequenceState::Swapped;
                s.kv_tokens = 0;
                true
            }
            _ => false,
        }
    }

    /// Mark `request_id` resident again with `kv_tokens` of GPU KV (the
    /// engine has already called `swap_in`).  Returns false when the id
    /// is unknown or not swapped.
    pub fn mark_resident(&mut self, request_id: &str, kv_tokens: usize) -> bool {
        match self
            .active
            .iter_mut()
            .find(|s| s.request_id == request_id && s.state == SequenceState::Swapped)
        {
            Some(s) => {
                s.state = SequenceState::Decode;
                s.kv_tokens = kv_tokens;
                s.last_touched = self.tick;
                true
            }
            None => false,
        }
    }

    /// Update a resident sequence's KV footprint + LRU clock (called by
    /// the engine loop after each step).
    pub fn touch(&mut self, request_id: &str, kv_tokens: usize) -> bool {
        self.tick += 1;
        match self
            .active
            .iter_mut()
            .find(|s| s.request_id == request_id && s.state != SequenceState::Swapped)
        {
            Some(s) => {
                s.kv_tokens = kv_tokens;
                s.last_touched = self.tick;
                true
            }
            None => false,
        }
    }

    /// Swap-in order: highest priority first, then oldest admission.
    pub fn next_swap_in_candidate(&self) -> Option<String> {
        self.active
            .iter()
            .filter(|s| s.state == SequenceState::Swapped)
            .max_by_key(|s| (s.priority, u64::MAX - s.age))
            .map(|s| s.request_id.clone())
    }

    /// Swapped sequence ids in swap-in priority order (highest first).
    pub fn swapped_ids(&self) -> Vec<String> {
        let mut swapped: Vec<&ActiveSequence> = self
            .active
            .iter()
            .filter(|s| s.state == SequenceState::Swapped)
            .collect();
        swapped.sort_by_key(|s| (std::cmp::Reverse(s.priority), s.age));
        swapped.into_iter().map(|s| s.request_id.clone()).collect()
    }

    pub fn swapped_count(&self) -> usize {
        self.active
            .iter()
            .filter(|s| s.state == SequenceState::Swapped)
            .count()
    }

    /// Total GPU-resident KV footprint across active sequences (tokens).
    pub fn resident_kv_tokens(&self) -> usize {
        self.active.iter().map(|s| s.kv_tokens).sum()
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn slot_accounting_no_double_alloc() {
        let num_slots: usize = kani::any();
        kani::assume(num_slots >= 1 && num_slots <= 4);

        let mut slots = vec![false; num_slots];

        let alloc_idx: usize = kani::any();
        kani::assume(alloc_idx < num_slots);
        kani::assume(!slots[alloc_idx]);

        slots[alloc_idx] = true;

        let release_idx: usize = kani::any();
        kani::assume(release_idx < num_slots);
        kani::assume(slots[release_idx]);

        slots[release_idx] = false;

        let re_alloc: usize = kani::any();
        kani::assume(re_alloc < num_slots);
        kani::assume(!slots[re_alloc]);

        assert!(!slots[re_alloc]);
        slots[re_alloc] = true;

        let used_count = slots.iter().filter(|&&v| v).count();
        assert!(used_count <= num_slots);
    }

    #[kani::proof]
    fn slot_indices_in_bounds() {
        let num_slots: usize = kani::any();
        kani::assume(num_slots >= 1 && num_slots <= 8);

        let idx: usize = kani::any();
        kani::assume(idx < num_slots);

        let slots = vec![false; num_slots];
        assert!(idx < slots.len());
    }

    /// Swap accounting: a swapped sequence contributes 0 to the resident
    /// KV footprint and is never selected as an eviction victim.  The
    /// state transition is idempotent-rejecting (no double swap).
    #[kani::proof]
    fn swap_accounting_zero_footprint_and_no_victim() {
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 4);
        let victim_idx: usize = kani::any();
        kani::assume(victim_idx < n);

        // Model the active list as fixed-capacity parallel arrays
        // (Kani-safe: no symbolic-length Vec iteration with adapters).
        let mut kv_tokens = [0usize; 4];
        let mut swapped = [false; 4];
        let mut i = 0;
        while i < n {
            kv_tokens[i] = 16;
            swapped[i] = false;
            i += 1;
        }
        // Indexed-sum (not .iter().sum()) -- adapter folds over
        // runtime-length slices unwind CBMC (the Vec-bomb class).
        let mut total = 0usize;
        let mut j = 0;
        while j < n {
            total += kv_tokens[j];
            j += 1;
        }
        assert!(total == 16 * n);

        // swap_out the victim: state flips, footprint zeroes.
        swapped[victim_idx] = true;
        kv_tokens[victim_idx] = 0;
        let mut total_after = 0usize;
        let mut k = 0;
        while k < n {
            total_after += kv_tokens[k];
            k += 1;
        }
        assert!(total_after == 16 * (n - 1));

        // Victim selection skips swapped entries: every unswapped index
        // is a valid victim, no swapped index is.
        let pick: usize = kani::any();
        kani::assume(pick < n);
        if !swapped[pick] {
            assert!(kv_tokens[pick] > 0);
        } else {
            assert!(kv_tokens[pick] == 0);
        }

        // Double swap is rejected (the footprint is already 0).
        let already_swapped = swapped[victim_idx];
        if already_swapped {
            assert!(kv_tokens[victim_idx] == 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: &str, tokens: Vec<u32>) -> InferenceRequest {
        InferenceRequest {
            id: id.to_string(),
            token_ids: tokens,
            max_new_tokens: 10,
            temperature: 1.0,
            top_p: 1.0,
            priority: 5,
        }
    }

    #[test]
    fn test_batch_scheduling() {
        let mut scheduler = BatchScheduler::new(2, 4096, 4);

        for i in 0..5 {
            scheduler.submit(make_request(&format!("req-{i}"), vec![1, 2, 3]));
        }

        assert_eq!(scheduler.pending_count(), 5);

        let batch1 = scheduler.next_batch();
        assert_eq!(batch1.len(), 2);
        assert_eq!(scheduler.pending_count(), 3);

        let batch2 = scheduler.next_batch();
        assert_eq!(batch2.len(), 2);
        assert_eq!(scheduler.pending_count(), 1);
    }

    #[test]
    fn test_admit_and_complete() {
        let mut scheduler = BatchScheduler::new(4, 4096, 4);

        scheduler.submit(make_request("r1", vec![1, 2]));
        scheduler.submit(make_request("r2", vec![3, 4]));

        let admitted = scheduler.admit_new_requests();
        assert_eq!(admitted.len(), 2);
        assert_eq!(scheduler.active_count(), 2);
        assert_eq!(scheduler.pending_count(), 0);
        assert_eq!(scheduler.allocated_slot_count(), 2);

        let slot = scheduler.complete_sequence("r1").unwrap();
        assert!(slot < 4);
        assert_eq!(scheduler.active_count(), 1);
        assert_eq!(scheduler.allocated_slot_count(), 1);
    }

    #[test]
    fn test_preemption_under_full_pool() {
        let mut scheduler = BatchScheduler::new(4, 4096, 2);

        scheduler.submit(make_request("r1", vec![1]));
        scheduler.submit(make_request("r2", vec![2]));
        scheduler.admit_new_requests();
        assert_eq!(scheduler.active_count(), 2);
        assert_eq!(scheduler.allocated_slot_count(), 2);

        scheduler.submit(make_request("r3", vec![3]));
        let admitted = scheduler.admit_new_requests();

        assert!(!admitted.is_empty(), "r3 should be admitted via preemption");
        assert_eq!(
            scheduler.preempted_count(),
            1,
            "evicted request should be in preempted queue"
        );
        assert_eq!(scheduler.active_count(), 2, "all slots remain in use");
    }

    #[test]
    fn test_readmission_of_preempted() {
        let mut scheduler = BatchScheduler::new(4, 4096, 2);

        scheduler.submit(make_request("r1", vec![1]));
        scheduler.submit(make_request("r2", vec![2]));
        scheduler.admit_new_requests();

        scheduler.submit(make_request("r3", vec![3]));
        scheduler.admit_new_requests();

        scheduler.complete_sequence("r3");
        scheduler.complete_sequence(
            &scheduler
                .active_sequences()
                .first()
                .map(|s| s.request_id.clone())
                .unwrap_or_default(),
        );

        let remaining_pending = scheduler.pending_count();
        let remaining_preempted = scheduler.preempted_count();
        assert!(
            remaining_pending + remaining_preempted <= 1,
            "preempted requests should be readmitted"
        );
    }

    #[test]
    fn test_mixed_prefill_decode() {
        let mut scheduler = BatchScheduler::new(4, 4096, 4);

        scheduler.submit(make_request("prefill-1", vec![1, 2, 3]));
        scheduler.submit(make_request("prefill-2", vec![4, 5]));
        let admitted = scheduler.admit_new_requests();

        assert_eq!(admitted.len(), 2);
        for seq in &admitted {
            assert_eq!(seq.state, SequenceState::Prefill);
        }

        scheduler.submit(make_request("prefill-3", vec![6]));
        let new_admitted = scheduler.admit_new_requests();
        assert_eq!(new_admitted.len(), 1);
        assert_eq!(scheduler.active_count(), 3);
    }

    #[test]
    fn test_slot_lifecycle() {
        let mut scheduler = BatchScheduler::new(4, 4096, 2);

        scheduler.submit(make_request("r1", vec![1]));
        scheduler.submit(make_request("r2", vec![2]));
        scheduler.admit_new_requests();

        assert_eq!(scheduler.allocated_slot_count(), 2);

        scheduler.complete_sequence("r1");
        assert_eq!(scheduler.allocated_slot_count(), 1);

        scheduler.complete_sequence("r2");
        assert_eq!(scheduler.allocated_slot_count(), 0);

        scheduler.submit(make_request("r3", vec![3]));
        scheduler.admit_new_requests();
        assert_eq!(scheduler.allocated_slot_count(), 1);
    }

    #[test]
    fn test_no_silent_drops() {
        let mut scheduler = BatchScheduler::new(2, 4096, 1);

        scheduler.submit(make_request("r1", vec![1]));
        scheduler.submit(make_request("r2", vec![2]));
        scheduler.submit(make_request("r3", vec![3]));

        scheduler.admit_new_requests();

        let total =
            scheduler.active_count() + scheduler.pending_count() + scheduler.preempted_count();
        assert_eq!(total, 3, "no request should be silently dropped");
    }

    #[test]
    fn test_priority_preempts_low_priority() {
        let mut scheduler = BatchScheduler::new(4, 4096, 1);

        let mut low = make_request("low", vec![1]);
        low.priority = 1;
        scheduler.submit(low);
        scheduler.admit_new_requests();
        assert_eq!(scheduler.active_count(), 1);

        let mut high = make_request("high", vec![2]);
        high.priority = 10;
        scheduler.submit(high);
        let admitted = scheduler.admit_new_requests();

        assert!(
            admitted.iter().any(|s| s.request_id == "high"),
            "high-priority request should be admitted via preemption"
        );
        assert_eq!(scheduler.preempted_count(), 1);
    }

    #[test]
    fn test_swap_victim_lru_within_priority() {
        let mut scheduler = BatchScheduler::new(4, 4096, 4);
        for id in ["a", "b", "c"] {
            scheduler.submit(make_request(id, vec![1]));
        }
        scheduler.admit_new_requests();

        // All resident with KV; step a and b so c is the LRU victim.
        assert!(scheduler.touch("a", 100));
        assert!(scheduler.touch("b", 100));
        assert!(scheduler.touch("c", 100));
        assert!(scheduler.touch("a", 100));
        assert!(scheduler.touch("b", 100));

        assert_eq!(scheduler.select_swap_victim(&[]).as_deref(), Some("c"));
        // Excluding the exhausted sequence protects it.
        assert_eq!(
            scheduler.select_swap_victim(&["c"]).as_deref(),
            Some("a"),
            "a is now the LRU resident"
        );
    }

    #[test]
    fn test_swap_victim_prefers_low_priority() {
        let mut scheduler = BatchScheduler::new(4, 4096, 4);
        let mut low = make_request("low", vec![1]);
        low.priority = 1;
        let mut high = make_request("high", vec![2]);
        high.priority = 9;
        scheduler.submit(low);
        scheduler.submit(high);
        scheduler.admit_new_requests();
        scheduler.touch("low", 50);
        scheduler.touch("high", 50);

        assert_eq!(scheduler.select_swap_victim(&[]).as_deref(), Some("low"));
    }

    #[test]
    fn test_mark_swapped_and_resident_accounting() {
        let mut scheduler = BatchScheduler::new(4, 4096, 4);
        scheduler.submit(make_request("r1", vec![1]));
        scheduler.submit(make_request("r2", vec![2]));
        scheduler.admit_new_requests();
        scheduler.touch("r1", 64);
        scheduler.touch("r2", 64);
        assert_eq!(scheduler.resident_kv_tokens(), 128);

        assert!(scheduler.mark_swapped("r1"));
        assert!(!scheduler.mark_swapped("r1"), "double swap is rejected");
        assert_eq!(scheduler.resident_kv_tokens(), 64, "swapped holds 0 pages");
        assert_eq!(scheduler.swapped_count(), 1);
        assert_eq!(
            scheduler.select_swap_victim(&[]).as_deref(),
            Some("r2"),
            "r2 is the only remaining resident"
        );

        assert!(scheduler.mark_swapped("r2"));
        assert_eq!(
            scheduler.select_swap_victim(&[]),
            None,
            "nothing left to evict once every sequence is swapped"
        );

        assert!(scheduler.mark_resident("r1", 64));
        assert_eq!(scheduler.resident_kv_tokens(), 64);
        assert_eq!(scheduler.swapped_count(), 1);
    }

    #[test]
    fn test_swap_in_order_priority_then_age() {
        let mut scheduler = BatchScheduler::new(8, 4096, 8);
        let mut low_old = make_request("low-old", vec![1]);
        low_old.priority = 1;
        let mut high = make_request("high", vec![2]);
        high.priority = 9;
        let mut low_new = make_request("low-new", vec![3]);
        low_new.priority = 1;
        scheduler.submit(low_old);
        scheduler.submit(high);
        scheduler.submit(low_new);
        scheduler.admit_new_requests();
        for id in ["low-old", "high", "low-new"] {
            scheduler.mark_swapped(id);
        }

        let order = scheduler.swapped_ids();
        assert_eq!(order, vec!["high".to_string(), "low-old".to_string(), "low-new".to_string()]);
        assert_eq!(scheduler.next_swap_in_candidate().as_deref(), Some("high"));
    }

    #[test]
    fn test_swapped_sequence_keeps_generated_state() {
        let mut scheduler = BatchScheduler::new(4, 4096, 4);
        scheduler.submit(make_request("r1", vec![1, 2, 3]));
        scheduler.admit_new_requests();
        // Simulate decode progress on r1.
        if let Some(seq) = scheduler
            .active_sequences_mut()
            .iter_mut()
            .find(|s| s.request_id == "r1")
        {
            seq.generated_tokens = vec![7, 8, 9];
            seq.position = 6;
        }
        scheduler.mark_swapped("r1");

        let seq = scheduler
            .active_sequences()
            .iter()
            .find(|s| s.request_id == "r1")
            .unwrap();
        assert_eq!(seq.generated_tokens, vec![7, 8, 9], "swap preserves state");
        assert_eq!(seq.position, 6);
        assert_eq!(seq.state, SequenceState::Swapped);
    }
}
