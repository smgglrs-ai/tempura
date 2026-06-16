use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::ModelConfig;
use crate::kv_cache::KvCache;
use crate::trace::{InferenceTrace, TraceLevel};

pub struct InferenceSession {
    pub cache: KvCache,
    pub trace: InferenceTrace,
    pub trace_level: TraceLevel,
    pub position: usize,
    pub rng: StdRng,
    pub seed: u64,
}

impl InferenceSession {
    pub fn new(config: &ModelConfig, trace_level: TraceLevel, seed: u64) -> Self {
        Self {
            cache: KvCache::new(config),
            trace: InferenceTrace::new(String::new(), trace_level, String::new(), seed),
            trace_level,
            position: 0,
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }

    pub fn with_cache(cache: KvCache, trace_level: TraceLevel, seed: u64) -> Self {
        Self {
            cache,
            trace: InferenceTrace::new(String::new(), trace_level, String::new(), seed),
            trace_level,
            position: 0,
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }

    pub fn take_cache(self) -> KvCache {
        self.cache
    }

    pub fn reset(&mut self) {
        self.cache.reset();
        self.position = 0;
        self.rng = StdRng::seed_from_u64(self.seed);
        self.trace = InferenceTrace::new(
            self.trace.trace_id.clone(),
            self.trace_level,
            self.trace.model_id.clone(),
            self.seed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_new() {
        let config = ModelConfig::tiny_test();
        let session = InferenceSession::new(&config, TraceLevel::Summary, 42);

        assert_eq!(session.position, 0);
        assert_eq!(session.seed, 42);
        assert_eq!(session.trace_level, TraceLevel::Summary);
        assert_eq!(session.trace.level, TraceLevel::Summary);
    }

    #[test]
    fn test_session_reset() {
        let config = ModelConfig::tiny_test();
        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);

        session.position = 10;
        session.trace.generated_tokens = 5;
        session.reset();

        assert_eq!(session.position, 0);
        assert_eq!(session.trace.generated_tokens, 0);
    }
}
