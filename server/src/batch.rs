use std::collections::VecDeque;

#[derive(Debug)]
pub struct InferenceRequest {
    pub id: String,
    pub token_ids: Vec<u32>,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
}

#[derive(Debug)]
pub struct InferenceResult {
    pub id: String,
    pub output_tokens: Vec<u32>,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Copy)]
pub enum FinishReason {
    Stop,
    Length,
}

pub struct BatchScheduler {
    pending: VecDeque<InferenceRequest>,
    max_batch_size: usize,
    max_seq_len: usize,
}

impl BatchScheduler {
    pub fn new(max_batch_size: usize, max_seq_len: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            max_batch_size,
            max_seq_len,
        }
    }

    pub fn submit(&mut self, request: InferenceRequest) {
        self.pending.push_back(request);
    }

    /// Select the next batch of requests to process together.
    ///
    /// Continuous batching: requests can join mid-generation. This stub
    /// implementation simply pops up to max_batch_size from the queue.
    pub fn next_batch(&mut self) -> Vec<InferenceRequest> {
        let count = self.pending.len().min(self.max_batch_size);
        self.pending.drain(..count).collect()
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_scheduling() {
        let mut scheduler = BatchScheduler::new(2, 4096);

        for i in 0..5 {
            scheduler.submit(InferenceRequest {
                id: format!("req-{}", i),
                token_ids: vec![1, 2, 3],
                max_new_tokens: 10,
                temperature: 1.0,
                top_p: 1.0,
            });
        }

        assert_eq!(scheduler.pending_count(), 5);

        let batch1 = scheduler.next_batch();
        assert_eq!(batch1.len(), 2);
        assert_eq!(scheduler.pending_count(), 3);

        let batch2 = scheduler.next_batch();
        assert_eq!(batch2.len(), 2);
        assert_eq!(scheduler.pending_count(), 1);
    }
}
