use std::collections::HashMap;
use std::sync::Arc;
use tempura_engine::session::InferenceSession;
use tempura_engine::trace::TraceLevel;
use tempura_engine::InferenceModel;
use tempura_kernels::SamplingParams;
use tokio::sync::mpsc;

const PREFILL_CHUNK_SIZE: usize = 512;

#[derive(Debug)]
pub enum EngineEvent {
    Token(u32),
    Done,
}

pub struct EngineRequest {
    pub id: String,
    pub token_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub params: SamplingParams,
    pub eos_token: Option<u32>,
    pub seed: u64,
    pub response_tx: mpsc::Sender<EngineEvent>,
}

struct ActiveSession {
    session: InferenceSession,
    params: SamplingParams,
    generated: Vec<u32>,
    max_new_tokens: usize,
    eos_token: Option<u32>,
    prefill_tokens: Vec<u32>,
    prefill_offset: usize,
    prefill_done: bool,
    response_tx: mpsc::Sender<EngineEvent>,
}

impl ActiveSession {
    fn step(&mut self, model: &dyn InferenceModel) -> bool {
        if !self.prefill_done {
            let start = self.prefill_offset;
            let end = (start + PREFILL_CHUNK_SIZE).min(self.prefill_tokens.len());
            let chunk = self.prefill_tokens[start..end].to_vec();
            let logits = model.forward(&mut self.session, &chunk);
            self.prefill_offset = end;

            if self.prefill_offset >= self.prefill_tokens.len() {
                self.prefill_done = true;
                let token = model.sample_from_logits(&logits, &self.params, &mut self.session);
                self.generated.push(token);
                let _ = self.response_tx.try_send(EngineEvent::Token(token));
                if self.generated.len() >= self.max_new_tokens || self.eos_token == Some(token) {
                    return true;
                }
            }
            return false;
        }

        let last_token = *self.generated.last().unwrap();
        let mut logits = model.forward(&mut self.session, &[last_token]);
        self.params.apply_penalties(&mut logits, &self.generated);
        let token = model.sample_from_logits(&logits, &self.params, &mut self.session);
        self.generated.push(token);
        let _ = self.response_tx.try_send(EngineEvent::Token(token));

        self.generated.len() >= self.max_new_tokens || self.eos_token == Some(token)
    }
}

pub struct EngineTask {
    model: Arc<dyn InferenceModel>,
    rx: mpsc::Receiver<EngineRequest>,
    thread_pool: rayon::ThreadPool,
}

impl EngineTask {
    pub fn new(model: Arc<dyn InferenceModel>, rx: mpsc::Receiver<EngineRequest>) -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("tempura-compute-{i}"))
            .build()
            .expect("failed to build rayon thread pool");
        Self { model, rx, thread_pool }
    }

    pub fn run(mut self) {
        let rt = tokio::runtime::Handle::current();
        let mut sessions: HashMap<String, ActiveSession> = HashMap::new();

        loop {
            // Drain incoming requests without blocking.
            loop {
                match self.rx.try_recv() {
                    Ok(req) => {
                        let session =
                            InferenceSession::new(self.model.config(), TraceLevel::Off, req.seed);
                        sessions.insert(
                            req.id.clone(),
                            ActiveSession {
                                session,
                                params: req.params,
                                generated: Vec::new(),
                                max_new_tokens: req.max_new_tokens,
                                eos_token: req.eos_token,
                                prefill_tokens: req.token_ids,
                                prefill_offset: 0,
                                prefill_done: false,
                                response_tx: req.response_tx,
                            },
                        );
                    }
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        if sessions.is_empty() {
                            return;
                        }
                        break;
                    }
                }
            }

            if sessions.is_empty() {
                match rt.block_on(self.rx.recv()) {
                    Some(req) => {
                        let session =
                            InferenceSession::new(self.model.config(), TraceLevel::Off, req.seed);
                        sessions.insert(
                            req.id.clone(),
                            ActiveSession {
                                session,
                                params: req.params,
                                generated: Vec::new(),
                                max_new_tokens: req.max_new_tokens,
                                eos_token: req.eos_token,
                                prefill_tokens: req.token_ids,
                                prefill_offset: 0,
                                prefill_done: false,
                                response_tx: req.response_tx,
                            },
                        );
                    }
                    None => return,
                }
                continue;
            }

            let ids: Vec<String> = sessions.keys().cloned().collect();
            let mut completed = Vec::new();

            for id in &ids {
                let entry = sessions.get_mut(id).unwrap();
                let done = self.thread_pool.install(|| entry.step(self.model.as_ref()));
                if done {
                    completed.push(id.clone());
                }
            }

            for id in &completed {
                if let Some(entry) = sessions.remove(id) {
                    let _ = entry.response_tx.try_send(EngineEvent::Done);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempura_engine::pipeline::KernelPipeline;
    use tempura_engine::{ModelConfig, TransformerModel};

    type TestModel = TransformerModel<
        tempura_kernels::attention::CpuAttention,
        tempura_kernels::rmsnorm::CpuRMSNorm,
        tempura_kernels::rope::CpuRoPE,
        tempura_kernels::matmul::CpuMatMul,
        tempura_kernels::sampling::CpuSampler,
    >;

    fn make_model() -> Arc<dyn InferenceModel> {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        Arc::new(TestModel::new(config, pipeline))
    }

    #[tokio::test]
    async fn test_engine_processes_batch() {
        let model = make_model();
        let vocab = model.config().vocab_size;
        let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(16);
        let task = EngineTask::new(model, engine_rx);

        tokio::task::spawn_blocking(move || task.run());

        let mut results = Vec::new();
        for i in 0..2u64 {
            let (resp_tx, mut resp_rx) = mpsc::channel::<EngineEvent>(32);
            engine_tx
                .send(EngineRequest {
                    id: format!("req-{i}"),
                    token_ids: vec![1u32, 2, 3],
                    max_new_tokens: 3,
                    params: SamplingParams {
                        temperature: 0.0,
                        ..Default::default()
                    },
                    eos_token: None,
                    seed: i,
                    response_tx: resp_tx,
                })
                .await
                .unwrap();

            let mut tokens = Vec::new();
            while let Some(event) = resp_rx.recv().await {
                match event {
                    EngineEvent::Token(t) => {
                        assert!((t as usize) < vocab, "token {t} out of vocab range {vocab}");
                        tokens.push(t);
                    }
                    EngineEvent::Done => break,
                }
            }
            assert_eq!(tokens.len(), 3, "expected 3 tokens for req-{i}");
            results.push(tokens);
        }
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_chunked_prefill_matches_full_prefill() {
        // A prompt shorter than PREFILL_CHUNK_SIZE produces the same first token
        // as one that would be split if slightly longer (deterministic sampling).
        let model = make_model();
        let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(4);
        let task = EngineTask::new(Arc::clone(&model), engine_rx);
        tokio::task::spawn_blocking(move || task.run());

        let prompt: Vec<u32> = (0..10).map(|i| i % 4).collect();

        let (resp_tx, mut resp_rx) = mpsc::channel::<EngineEvent>(8);
        engine_tx
            .send(EngineRequest {
                id: "short".into(),
                token_ids: prompt.clone(),
                max_new_tokens: 1,
                params: SamplingParams {
                    temperature: 0.0,
                    ..Default::default()
                },
                eos_token: None,
                seed: 42,
                response_tx: resp_tx,
            })
            .await
            .unwrap();

        let mut token_from_engine = None;
        while let Some(ev) = resp_rx.recv().await {
            match ev {
                EngineEvent::Token(t) => token_from_engine = Some(t),
                EngineEvent::Done => break,
            }
        }

        // Compare against direct forward on the same model.
        let mut session = InferenceSession::new(model.config(), TraceLevel::Off, 42);
        let logits = model.forward(&mut session, &prompt);
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let expected = model.sample_from_logits(&logits, &params, &mut session);

        assert_eq!(token_from_engine, Some(expected));
    }
}
