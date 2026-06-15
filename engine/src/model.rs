use crate::config::ModelConfig;
use crate::kv_cache::KvCache;
use crate::weights::ModelWeights;
use oxide_kernels::{Attention, LayerNorm, RoPE, Sampler, SamplingParams};

pub struct TransformerModel<A, L, R, S> {
    pub config: ModelConfig,
    pub cache: KvCache,
    pub attention: A,
    pub layernorm: L,
    pub rope: R,
    pub sampler: S,
    pub weights: Option<ModelWeights>,
}

impl<A, L, R, S> TransformerModel<A, L, R, S>
where
    A: Attention,
    L: LayerNorm,
    R: RoPE,
    S: Sampler,
{
    pub fn new(config: ModelConfig, attention: A, layernorm: L, rope: R, sampler: S) -> Self {
        let cache = KvCache::new(&config);
        Self {
            config,
            cache,
            attention,
            layernorm,
            rope,
            sampler,
            weights: None,
        }
    }

    pub fn load_weights(
        &mut self,
        loader: &dyn crate::loader::WeightLoader,
    ) -> Result<(), crate::weights::WeightsError> {
        self.weights = Some(ModelWeights::load(loader, &self.config)?);
        Ok(())
    }

    /// Run a forward pass for a single token and return logits.
    ///
    /// This is a structural placeholder — the actual implementation requires
    /// weight tensors, embedding lookup, and the full layer loop.
    pub fn forward(&mut self, _token_ids: &[u32], _positions: &[u32]) -> Vec<f32> {
        // TODO: implement actual forward pass:
        // 1. embedding lookup
        // 2. for each layer:
        //    a. layernorm
        //    b. qkv projection
        //    c. rope
        //    d. attention (with kv-cache)
        //    e. residual + layernorm
        //    f. feed-forward (gate_proj, up_proj, down_proj with SiLU)
        //    g. residual
        // 3. final layernorm
        // 4. lm_head projection -> logits
        vec![0.0; self.config.vocab_size]
    }

    pub fn sample(
        &self,
        logits: &[f32],
        params: &SamplingParams,
    ) -> oxide_kernels::Result<Vec<f32>> {
        let mut probs = vec![0.0; logits.len()];
        self.sampler.apply(logits, &mut probs, params)?;
        Ok(probs)
    }

    pub fn reset(&mut self) {
        self.cache.reset();
    }
}

pub use oxide_kernels::sampling::SamplingParams as SampleParams;
