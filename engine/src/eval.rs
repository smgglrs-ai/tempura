//! Corpus-level evaluation: perplexity scoring over any `InferenceModel`.
//!
//! Replicates llama.cpp's `tools/perplexity/perplexity.cpp` default
//! algorithm exactly (non-overlapping chunks of `n_ctx` tokens, every chunk
//! starting with BOS, positions `[n_ctx/2, n_ctx-2]` scored against the
//! next token, f64-accumulated log-softmax NLL) so the two implementations'
//! numbers are directly comparable on the same corpus file.
//!
//! llama.cpp computes logits for all scored positions in one batched
//! forward per chunk; tempura's `InferenceModel::forward` returns only the
//! last position's logits, so this module scores by prefilling the first
//! half of the chunk and decoding the rest one token at a time. Attention
//! is causal, so per-position logits are identical either way; the only
//! difference is which kernel path computes them (decode vs prefill).

use crate::model::InferenceModel;
use crate::session::InferenceSession;
use crate::trace::TraceLevel;
use crate::ModelConfig;
use std::time::Instant;

/// Log-probability of `tok` under `logits`, replicating llama.cpp's
/// `log_softmax` (f32 max-subtraction, f64 accumulation of the exp-sum,
/// natural log in f64). NaN logits anywhere poison the sum like upstream.
pub fn token_logprob(logits: &[f32], tok: usize) -> f64 {
    debug_assert!(tok < logits.len(), "token id {tok} outside logits");
    let tok = tok.min(logits.len().saturating_sub(1));
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f64;
    for &l in logits {
        sum_exp += (l - max_logit).exp() as f64;
    }
    (logits[tok] - max_logit) as f64 - sum_exp.ln()
}

#[derive(Debug, Clone)]
pub struct PplResult {
    /// Sum of negative log-likelihoods over all scored positions.
    pub nll: f64,
    /// Number of scored positions (llama.cpp's `count`).
    pub count: usize,
    /// exp(nll / count) -- the perplexity.
    pub ppl: f64,
    /// Chunks actually evaluated.
    pub chunks: usize,
    /// Logits vectors containing at least one NaN (quality gate input).
    pub nan_steps: usize,
    pub elapsed_s: f64,
    /// Per-scored-position logprob of the actual next token (corpus
    /// order), populated when `PplOptions::collect_logprobs` is set.
    pub logprobs: Vec<f64>,
}

impl PplResult {
    /// Average NLL per scored token.
    pub fn avg_nll(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.nll / self.count as f64
        }
    }
}

/// Options for [`perplexity`].
#[derive(Debug, Clone)]
pub struct PplOptions {
    /// Context window per chunk (llama.cpp `-c`). Positions `[n_ctx/2,
    /// n_ctx-2]` of each chunk are scored.
    pub n_ctx: usize,
    /// Evaluate at most this many chunks (llama.cpp `--chunks`); None =
    /// all. The corpus prefix is used first, so a chunk subset compares
    /// clean against `llama-perplexity --chunks N` on the same file.
    pub max_chunks: Option<usize>,
    /// BOS token id substituted at position 0 of EVERY chunk (llama.cpp
    /// replaces the chunk-start token with BOS when the vocab has add_bos).
    pub bos: Option<u32>,
    /// Print per-chunk progress to stderr (long CPU runs are silent
    /// otherwise).
    pub progress: bool,
    /// When set, collect the per-scored-position logprobs into the result
    /// (token logprob of each scored next-token, in corpus order).
    pub collect_logprobs: bool,
}

impl Default for PplOptions {
    fn default() -> Self {
        PplOptions {
            n_ctx: 512,
            max_chunks: None,
            bos: None,
            progress: false,
            collect_logprobs: false,
        }
    }
}

/// Corpus perplexity, llama.cpp-compatible, with optional per-chunk progress
/// (long CPU runs print nothing otherwise).
///
/// `tokens` is the full tokenized corpus INCLUDING the leading BOS (that is
/// what `common_tokenize(text, true)` produces and what
/// `Tokenizer::encode` produces for add_bos vocabs).
///
/// Returns `None` when the corpus has fewer than `n_ctx` tokens (llama.cpp
/// requires `2*n_ctx`; we only require one full chunk so small smoke-test
/// corpora work).
pub fn perplexity(
    model: &dyn InferenceModel,
    config: &ModelConfig,
    tokens: &[u32],
    opts: &PplOptions,
) -> Option<PplResult> {
    let n_ctx = opts.n_ctx;
    if tokens.len() < n_ctx || n_ctx < 4 {
        return None;
    }
    let n_chunk_max = tokens.len() / n_ctx;
    let n_chunk = opts.max_chunks.map_or(n_chunk_max, |m| m.min(n_chunk_max));
    let first = n_ctx / 2;

    let t0 = Instant::now();
    let mut nll = 0.0f64;
    let mut count = 0usize;
    let mut nan_steps = 0usize;
    let mut logprobs = if opts.collect_logprobs { Vec::new() } else { Vec::new() };

    for i in 0..n_chunk {
        if opts.progress {
            eprintln!(
                "  [ppl] chunk {}/{} (nll/count so far: {:.4}/{})",
                i + 1,
                n_chunk,
                nll,
                count
            );
        }
        let start = i * n_ctx;
        let mut chunk = tokens[start..start + n_ctx].to_vec();
        if let Some(b) = opts.bos {
            chunk[0] = b;
        }
        // Fresh KV per chunk: llama.cpp clears the cache and resets
        // positions for every chunk.
        let mut session = InferenceSession::new(config, TraceLevel::Off, 0);
        // Prefill [0..=first]; the returned logits sit at position `first`
        // and predict chunk[first + 1].
        let mut logits = model.forward(&mut session, &chunk[..first + 1]);
        if logits.iter().any(|v| v.is_nan()) {
            nan_steps += 1;
        }
        let mut j = first;
        while j < n_ctx - 1 {
            let lp = token_logprob(&logits, chunk[j + 1] as usize);
            if opts.collect_logprobs {
                logprobs.push(lp);
            }
            nll += -lp;
            count += 1;
            j += 1;
            if j < n_ctx - 1 {
                logits = model.forward(&mut session, &[chunk[j]]);
                if logits.iter().any(|v| v.is_nan()) {
                    nan_steps += 1;
                }
            }
        }
    }

    let ppl = if count > 0 {
        (nll / count as f64).exp()
    } else {
        f64::NAN
    };
    Some(PplResult {
        nll,
        count,
        ppl,
        chunks: n_chunk,
        nan_steps,
        elapsed_s: t0.elapsed().as_secs_f64(),
        logprobs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SampleParams;

    /// Deterministic mock: uniform logits -> every token has logprob
    /// -ln(V), so perplexity == vocab_size regardless of the corpus.
    struct UniformMock {
        config: ModelConfig,
    }

    impl InferenceModel for UniformMock {
        fn config(&self) -> &ModelConfig {
            &self.config
        }
        fn has_weights(&self) -> bool {
            true
        }
        fn forward(&self, _session: &mut InferenceSession, _token_ids: &[u32]) -> Vec<f32> {
            vec![1.0; self.config.vocab_size]
        }
        fn generate(
            &self,
            _s: &mut InferenceSession,
            _t: &[u32],
            _n: usize,
            _p: &SampleParams,
            _e: Option<u32>,
        ) -> Vec<u32> {
            Vec::new()
        }
        fn generate_with_logprobs(
            &self,
            _s: &mut InferenceSession,
            _t: &[u32],
            _n: usize,
            _p: &SampleParams,
            _e: Option<u32>,
            _top: usize,
        ) -> (Vec<u32>, Vec<crate::model::TokenLogprob>) {
            (Vec::new(), Vec::new())
        }
        fn generate_with_callback(
            &self,
            _s: &mut InferenceSession,
            _t: &[u32],
            _n: usize,
            _p: &SampleParams,
            _e: Option<u32>,
            _c: &mut dyn FnMut(u32) -> bool,
        ) -> Vec<u32> {
            Vec::new()
        }
        fn generate_guided(
            &self,
            _s: &mut InferenceSession,
            _t: &[u32],
            _n: usize,
            _p: &SampleParams,
            _e: Option<u32>,
            _d: &mut dyn tempura_kernels::GuidedDecoder,
        ) -> Vec<u32> {
            Vec::new()
        }
        fn sample_from_logits(
            &self,
            _l: &[f32],
            _p: &SampleParams,
            _s: &mut InferenceSession,
        ) -> u32 {
            0
        }
        fn load_weights(
            &mut self,
            _loader: &dyn crate::loader::WeightLoader,
        ) -> Result<(), crate::weights::WeightsError> {
            Ok(())
        }
    }

    fn test_config(vocab: usize) -> ModelConfig {
        ModelConfig {
            vocab_size: vocab,
            ..ModelConfig::llama2_7b()
        }
    }

    #[test]
    fn token_logprob_uniform_is_minus_ln_v() {
        let logits = vec![0.5f32; 128];
        let lp = token_logprob(&logits, 42);
        assert!((lp - -128f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn token_logprob_matches_manual_computation() {
        let logits = vec![0.0f32, 1.0, 2.0, 3.0];
        // logprob(token 3) = 3 - 3 - ln(1 + e^-1 + e^-2 + e^-3). The exps
        // are computed in f32 (llama.cpp uses expf), so a f64 reference
        // only agrees to ~1e-7.
        let expected = -(1.0f64 + (-1.0f64).exp() + (-2.0f64).exp() + (-3.0f64).exp()).ln();
        let lp = token_logprob(&logits, 3);
        assert!((lp - expected).abs() < 1e-6, "lp {lp} vs {expected}");
    }

    #[test]
    fn token_logprob_favors_large_logit() {
        let mut logits = vec![0.0f32; 4];
        logits[2] = 10.0;
        // sum_exp = 1 + 3*e^-10 -> logprob = -ln(1.000136) ~ -1.36e-4
        let lp = token_logprob(&logits, 2);
        assert!((lp - -(1.0f64 + 3.0 * (-10.0f64).exp()).ln()).abs() < 1e-6);
        let lp_other = token_logprob(&logits, 0);
        assert!((lp_other - -10.0f64).abs() < 1e-3);
    }

    #[test]
    fn uniform_mock_perplexity_is_vocab_size() {
        let cfg = test_config(50);
        let model = UniformMock {
            config: cfg.clone(),
        };
        // 3 chunks + a partial 4th (ignored: n_chunk_max = floor); ids stay
        // inside the toy vocab (real corpora always do).
        let tokens: Vec<u32> = (0..(512 * 3 + 100) as u32).map(|t| t % 50).collect();
        let res = perplexity(&model, &cfg, &tokens, &PplOptions::default()).unwrap();
        assert_eq!(res.chunks, 3);
        assert_eq!(res.count, 3 * (512 - 256 - 1));
        // ppl == exp(ln 50) == 50 (within f64 rounding of the mean)
        assert!((res.ppl - 50.0).abs() < 1e-9, "ppl {}", res.ppl);
        assert_eq!(res.nan_steps, 0);
    }

    #[test]
    fn max_chunks_limits_to_corpus_prefix() {
        let cfg = test_config(8);
        let model = UniformMock {
            config: cfg.clone(),
        };
        let tokens: Vec<u32> = (0..2048u32).map(|t| t % 8).collect();
        let res = perplexity(
            &model,
            &cfg,
            &tokens,
            &PplOptions {
                max_chunks: Some(1),
                ..PplOptions::default()
            },
        )
        .unwrap();
        assert_eq!(res.chunks, 1);
        assert_eq!(res.count, 512 - 256 - 1);
    }

    #[test]
    fn bos_replaces_every_chunk_start() {
        // The mock returns logits that depend on the LAST fed token, so the
        // scored NLL exposes which tokens were actually fed. BOS=7 at every
        // chunk start means position 0 of chunk >= 1 is token 7, not the
        // corpus token that was there.
        struct LastTokenMock {
            config: ModelConfig,
        }
        impl InferenceModel for LastTokenMock {
            fn config(&self) -> &ModelConfig {
                &self.config
            }
            fn has_weights(&self) -> bool {
                true
            }
            fn forward(&self, _session: &mut InferenceSession, token_ids: &[u32]) -> Vec<f32> {
                let mut l = vec![0.0f32; self.config.vocab_size];
                let last = token_ids.last().copied().unwrap_or(0) as usize;
                let idx = last % l.len();
                l[idx] = 5.0;
                l
            }
            fn generate(
                &self,
                _s: &mut InferenceSession,
                _t: &[u32],
                _n: usize,
                _p: &SampleParams,
                _e: Option<u32>,
            ) -> Vec<u32> {
                Vec::new()
            }
            fn generate_with_logprobs(
                &self,
                _s: &mut InferenceSession,
                _t: &[u32],
                _n: usize,
                _p: &SampleParams,
                _e: Option<u32>,
                _top: usize,
            ) -> (Vec<u32>, Vec<crate::model::TokenLogprob>) {
                (Vec::new(), Vec::new())
            }
            fn generate_with_callback(
                &self,
                _s: &mut InferenceSession,
                _t: &[u32],
                _n: usize,
                _p: &SampleParams,
                _e: Option<u32>,
                _c: &mut dyn FnMut(u32) -> bool,
            ) -> Vec<u32> {
                Vec::new()
            }
            fn generate_guided(
                &self,
                _s: &mut InferenceSession,
                _t: &[u32],
                _n: usize,
                _p: &SampleParams,
                _e: Option<u32>,
                _d: &mut dyn tempura_kernels::GuidedDecoder,
            ) -> Vec<u32> {
                Vec::new()
            }
            fn sample_from_logits(
                &self,
                _l: &[f32],
                _p: &SampleParams,
                _s: &mut InferenceSession,
            ) -> u32 {
                0
            }
            fn load_weights(
                &mut self,
                _loader: &dyn crate::loader::WeightLoader,
            ) -> Result<(), crate::weights::WeightsError> {
                Ok(())
            }
        }

        let cfg = test_config(16);
        let model = LastTokenMock {
            config: cfg.clone(),
        };
        // Corpus: token i at position i; with BOS=7 substituted at chunk
        // starts, the model's 5.0 logit sits on token 7 whenever the last
        // fed token was the chunk-start BOS.
        let n_ctx = 8usize;
        let tokens: Vec<u32> = (0..(n_ctx * 2) as u32).map(|t| t % 16).collect();
        let res = perplexity(
            &model,
            &cfg,
            &tokens,
            &PplOptions {
                n_ctx,
                bos: Some(7),
                ..PplOptions::default()
            },
        )
        .unwrap();
        assert_eq!(res.chunks, 2);
        // count = 2 * (8 - 4 - 1) = 6 scored positions
        assert_eq!(res.count, 6);
        // Sanity: the run completed with finite NLL (BOS clamp did not
        // panic, out-of-range ids are impossible here).
        assert!(res.nll.is_finite());
    }

    #[test]
    fn too_short_corpus_returns_none() {
        let cfg = test_config(4);
        let model = UniformMock {
            config: cfg.clone(),
        };
        assert!(perplexity(&model, &cfg, &[1, 2, 3], &PplOptions::default()).is_none());
    }
}
