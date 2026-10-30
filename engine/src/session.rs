use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::ModelConfig;
use crate::deltanet_state::DeltaNetState;
use crate::kv_cache::{KvCache, PagedKvCache};
use crate::trace::{InferenceTrace, TraceLevel};

#[derive(Default)]
pub struct ForwardScratch {
    pub hidden: Vec<f32>,
    pub normed: Vec<f32>,
    pub normed2: Vec<f32>,
    pub q_buf: Vec<f32>,
    pub k_buf: Vec<f32>,
    pub v_buf: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub gate_buf: Vec<f32>,
    pub up_buf: Vec<f32>,
    pub mlp_out: Vec<f32>,
}

pub struct InferenceSession {
    pub cache: KvCache,
    pub paged_cache: Option<PagedKvCache>,
    pub deltanet_state: Option<DeltaNetState>,
    pub trace: InferenceTrace,
    pub trace_level: TraceLevel,
    pub position: usize,
    pub rng: StdRng,
    pub seed: u64,
    pub scratch: ForwardScratch,
}

impl InferenceSession {
    pub fn new(config: &ModelConfig, trace_level: TraceLevel, seed: u64) -> Self {
        let deltanet_state = if config.num_deltanet_layers() > 0 {
            Some(DeltaNetState::new(config))
        } else {
            None
        };
        Self {
            cache: KvCache::new(config),
            paged_cache: None,
            deltanet_state,
            trace: InferenceTrace::new(String::new(), trace_level, String::new(), seed),
            trace_level,
            position: 0,
            rng: StdRng::seed_from_u64(seed),
            seed,
            scratch: ForwardScratch::default(),
        }
    }

    pub fn new_paged(
        config: &ModelConfig,
        total_pages: usize,
        block_size: usize,
        trace_level: TraceLevel,
        seed: u64,
    ) -> Self {
        let deltanet_state = if config.num_deltanet_layers() > 0 {
            Some(DeltaNetState::new(config))
        } else {
            None
        };
        Self {
            cache: KvCache::new(config),
            paged_cache: Some(PagedKvCache::new(config, total_pages, block_size)),
            deltanet_state,
            trace: InferenceTrace::new(String::new(), trace_level, String::new(), seed),
            trace_level,
            position: 0,
            rng: StdRng::seed_from_u64(seed),
            seed,
            scratch: ForwardScratch::default(),
        }
    }

    pub fn with_cache(cache: KvCache, trace_level: TraceLevel, seed: u64) -> Self {
        Self {
            cache,
            paged_cache: None,
            deltanet_state: None,
            trace: InferenceTrace::new(String::new(), trace_level, String::new(), seed),
            trace_level,
            position: 0,
            rng: StdRng::seed_from_u64(seed),
            seed,
            scratch: ForwardScratch::default(),
        }
    }

    pub fn uses_paged_cache(&self) -> bool {
        self.paged_cache.is_some()
    }

    pub fn take_cache(self) -> KvCache {
        self.cache
    }

    pub fn reset(&mut self) {
        self.cache.reset();
        if let Some(ref mut paged) = self.paged_cache {
            paged.reset();
        }
        if let Some(ref mut dn) = self.deltanet_state {
            dn.reset();
        }
        self.position = 0;
        self.rng = StdRng::seed_from_u64(self.seed);
        self.trace = InferenceTrace::new(
            self.trace.trace_id.clone(),
            self.trace_level,
            self.trace.model_id.clone(),
            self.seed,
        );
    }

    pub fn to_transfer(&self) -> SessionTransfer {
        SessionTransfer {
            cache: self.cache.to_transfer(),
            position: self.position,
            seed: self.seed,
            last_logits: Vec::new(),
        }
    }

    pub fn to_transfer_with_logits(&self, logits: Vec<f32>) -> SessionTransfer {
        SessionTransfer {
            cache: self.cache.to_transfer(),
            position: self.position,
            seed: self.seed,
            last_logits: logits,
        }
    }

    pub fn from_transfer(
        transfer: SessionTransfer,
        config: &ModelConfig,
        trace_level: TraceLevel,
    ) -> Self {
        let cache = KvCache::from_transfer(transfer.cache, config);
        Self {
            cache,
            paged_cache: None,
            deltanet_state: None,
            trace: InferenceTrace::new(String::new(), trace_level, String::new(), transfer.seed),
            trace_level,
            position: transfer.position,
            rng: StdRng::seed_from_u64(transfer.seed),
            seed: transfer.seed,
            scratch: ForwardScratch::default(),
        }
    }
}

impl tempura_kernel_api::VerifiedSession for InferenceSession {
    fn position(&self) -> usize {
        self.position
    }
    fn cache_seq_len(&self) -> usize {
        self.cache.seq_len
    }
    fn max_seq_len(&self) -> usize {
        self.cache.max_seq_len()
    }
}

use vstd::prelude::*;

verus! {

proof fn session_new_consistent(max_seq_len: nat)
    requires max_seq_len >= 1,
    ensures 0nat == 0nat && 0nat <= max_seq_len,
{
}

proof fn session_advance_consistent(pos: nat, cache_sl: nat, max_sl: nat, n: nat)
    requires pos == cache_sl, pos <= max_sl, n >= 1, pos + n <= max_sl,
    ensures pos + n == cache_sl + n && pos + n <= max_sl,
{
}

proof fn session_reset_consistent()
    ensures 0nat == 0nat,
{
}

} // verus!

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionTransfer {
    pub cache: crate::kv_cache::KvCacheTransfer,
    pub position: usize,
    pub seed: u64,
    pub last_logits: Vec<f32>,
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
