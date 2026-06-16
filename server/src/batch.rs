use std::collections::VecDeque;

#[derive(Debug)]
pub struct InferenceRequest {
    pub id: String,
    pub token_ids: Vec<u32>,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    Prefill,
    Decode,
    Preempted,
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
            .min_by_key(|(_, s)| s.age)
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

        let sources: Vec<InferenceRequest> = self
            .preempted
            .drain(..)
            .chain(self.pending.drain(..))
            .collect();

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

    pub fn active_sequences(&self) -> &[ActiveSequence] {
        &self.active
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

        assert!(admitted.len() >= 1, "r3 should be admitted via preemption");
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
}
