use oxide_engine::trace::InferenceTrace;
use std::collections::VecDeque;

pub struct TraceStore {
    traces: VecDeque<InferenceTrace>,
    max_traces: usize,
}

impl TraceStore {
    pub fn new(max_traces: usize) -> Self {
        Self {
            traces: VecDeque::with_capacity(max_traces),
            max_traces,
        }
    }

    pub fn insert(&mut self, trace: InferenceTrace) {
        if self.traces.len() >= self.max_traces {
            self.traces.pop_front();
        }
        self.traces.push_back(trace);
    }

    pub fn get(&self, trace_id: &str) -> Option<&InferenceTrace> {
        self.traces.iter().find(|t| t.trace_id == trace_id)
    }

    pub fn list(&self) -> Vec<TraceSummary> {
        self.traces
            .iter()
            .map(|t| TraceSummary {
                trace_id: t.trace_id.clone(),
                model_id: t.model_id.clone(),
                level: t.level,
                prompt_tokens: t.prompt_tokens,
                generated_tokens: t.generated_tokens,
            })
            .collect()
    }

    pub fn remove(&mut self, trace_id: &str) -> bool {
        let before = self.traces.len();
        self.traces.retain(|t| t.trace_id != trace_id);
        self.traces.len() < before
    }
}

#[derive(Debug, serde::Serialize)]
pub struct TraceSummary {
    pub trace_id: String,
    pub model_id: String,
    pub level: oxide_engine::trace::TraceLevel,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxide_engine::trace::{InferenceTrace, TraceLevel};

    fn make_trace(id: &str) -> InferenceTrace {
        InferenceTrace::new(id.to_string(), TraceLevel::Summary, "test".to_string(), 0)
    }

    #[test]
    fn test_insert_and_get() {
        let mut store = TraceStore::new(10);
        store.insert(make_trace("t1"));
        assert!(store.get("t1").is_some());
        assert!(store.get("t2").is_none());
    }

    #[test]
    fn test_bounded_eviction() {
        let mut store = TraceStore::new(2);
        store.insert(make_trace("t1"));
        store.insert(make_trace("t2"));
        store.insert(make_trace("t3"));
        assert!(store.get("t1").is_none());
        assert!(store.get("t2").is_some());
        assert!(store.get("t3").is_some());
    }

    #[test]
    fn test_remove() {
        let mut store = TraceStore::new(10);
        store.insert(make_trace("t1"));
        assert!(store.remove("t1"));
        assert!(!store.remove("t1"));
        assert!(store.get("t1").is_none());
    }

    #[test]
    fn test_list() {
        let mut store = TraceStore::new(10);
        store.insert(make_trace("t1"));
        store.insert(make_trace("t2"));
        let list = store.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].trace_id, "t1");
    }
}
