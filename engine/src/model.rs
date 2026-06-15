use crate::config::ModelConfig;
use crate::session::InferenceSession;
use crate::weights::ModelWeights;
use oxide_kernels::{Attention, LayerNorm, RoPE, Sampler, SamplingParams};

pub struct TransformerModel<A, L, R, S> {
    pub config: ModelConfig,
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
        Self {
            config,
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

    pub fn forward(&self, session: &mut InferenceSession, _token_ids: &[u32]) -> Vec<f32> {
        let _ = session;
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
}

pub use oxide_kernels::sampling::SamplingParams as SampleParams;
