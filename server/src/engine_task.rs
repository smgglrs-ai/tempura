use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tempura_engine::session::InferenceSession;
use tempura_engine::trace::TraceLevel;
use tempura_engine::InferenceModel;
use tempura_kernels::SamplingParams;
use tokio::sync::mpsc;

use crate::batch::{BatchScheduler, InferenceRequest, SequenceState};

const PREFILL_CHUNK_SIZE: usize = 512;
/// Admission headroom (tokens): a new session's initial run (512) plus
/// one growth step.  Below this the engine stops admitting until the
/// pool drains -- prevents admit/requeue spinning when the pool is full.
#[cfg(feature = "cutile")]
const ADMISSION_HEADROOM_TOKENS: usize = 768;

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

/// Per-request DRAFT model state for speculative decoding (greedy):
/// the draft's own pooled KV run + a session mirror for position
/// bookkeeping through `forward_with_kv` (prompt prefill).  `kv` is an
/// Option so a batched draft step can `take()` every live run into a
/// contiguous slice for `forward_batch_with_kvs_greedy` and put it back.
#[cfg(feature = "cutile")]
struct DraftCache {
    session: InferenceSession,
    kv: Option<tempura_engine::cutile_kv_cache::CutileKvCache>,
    prefilled: bool,
}

/// Per-request EAGLE-3 head state: the head's own 1-layer pooled KV run,
/// the raw 3-layer carry (target hidden at position head_len), and the
/// prefill/failed bookkeeping.  `failed` latches: a session whose head
/// state could not be maintained (stale taps, unsupported forward path)
/// stops speculating for the rest of the request -- plain decode.
#[cfg(feature = "cutile")]
struct EagleSessionState {
    kv: tempura_engine::cutile_kv_cache::CutileKvCache,
    anchor: std::sync::Arc<cutile::tensor::Tensor<f32>>,
    prefilled: bool,
    failed: bool,
}

/// Acceptance-length telemetry for the speculative path: how many tokens
/// each verify cycle actually emitted (index 0 = rejection at the first
/// draft token, index k = full acceptance).  TEMPURA_ENGINE_DEBUG_TICKS
/// prints the histogram + mean every 50 ticks -- the throughput story
/// (mean ~2-3 for a 1B drafting for an 8B) needs it.
#[cfg(feature = "cutile")]
#[derive(Default)]
struct SpecStats {
    cycles: u64,
    /// emitted-per-cycle histogram: hist[i] = cycles that emitted i+1
    /// tokens (index k-1 = full acceptance of k drafts).
    hist: Vec<u64>,
    tokens_emitted: u64,
    /// Wall time in the draft-propose phase vs the verify forward (the
    /// throughput attribution: which side a future optimization must
    /// attack).  Microseconds.  Accumulated per ENGINE cycle (one
    /// spec_batch_step_gpu call), unlike `cycles`/`hist` which count
    /// per-SESSION notes.
    draft_us: u64,
    verify_us: u64,
    /// Engine spec cycles (spec_batch_step_gpu calls) -- the correct
    /// denominator for draft_us/verify_us.
    engine_cycles: u64,
}

#[cfg(feature = "cutile")]
impl SpecStats {
    fn note(&mut self, emitted: usize, k: usize) {
        self.cycles += 1;
        self.tokens_emitted += emitted as u64;
        if self.hist.len() < k {
            self.hist.resize(k, 0);
        }
        self.hist[emitted - 1] += 1;
    }
    fn mean(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            self.tokens_emitted as f64 / self.cycles as f64
        }
    }
    fn add_timing(&mut self, draft_us: u64, verify_us: u64) {
        self.draft_us += draft_us;
        self.verify_us += verify_us;
        self.engine_cycles += 1;
    }
}

impl ActiveSession {
    /// Emit an already-sampled token: append it, push it to the client,
    /// and return true when the request is finished.  Shared by
    /// `sample_and_emit` (CPU sampling) and the GPU greedy-argmax path
    /// (token produced on-device, no logits download).
    fn emit_token(&mut self, token: u32) -> bool {
        self.generated.push(token);
        // A closed response channel (client gone / request timed out)
        // retires the session -- otherwise the engine spins on a dead
        // request forever (found by the p24 A/B: the 300s request
        // timeout left the sessions stepping into a dropped receiver,
        // millions of no-op ticks).  A merely FULL channel drops the
        // token but keeps the request alive.
        let client_gone = match self.response_tx.try_send(EngineEvent::Token(token)) {
            Ok(()) => false,
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => false,
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => true,
        };
        self.generated.len() >= self.max_new_tokens
            || self.eos_token == Some(token)
            || client_gone
    }

    /// Sample the next token from `logits` (with penalties), emit it, and
    /// return true when the request is finished.  Shared by the CPU
    /// per-session path and the GPU batch path.
    fn sample_and_emit(&mut self, model: &dyn InferenceModel, logits: Vec<f32>) -> bool {
        let mut logits = logits;
        self.params.apply_penalties(&mut logits, &self.generated);
        let token = model.sample_from_logits(&logits, &self.params, &mut self.session);
        self.emit_token(token)
    }

    /// CPU-model step (chunked prefill + decode).  Unchanged from the
    /// pre-p24 loop.
    fn step(&mut self, model: &dyn InferenceModel) -> bool {
        if !self.prefill_done {
            let start = self.prefill_offset;
            let end = (start + PREFILL_CHUNK_SIZE).min(self.prefill_tokens.len());
            let chunk = self.prefill_tokens[start..end].to_vec();
            let logits = model.forward(&mut self.session, &chunk);
            self.prefill_offset = end;

            if self.prefill_offset >= self.prefill_tokens.len() {
                self.prefill_done = true;
                return self.sample_and_emit(model, logits);
            }
            return false;
        }

        let last_token = *self.generated.last().unwrap();
        let logits = model.forward(&mut self.session, &[last_token]);
        self.sample_and_emit(model, logits)
    }
}

/// Parse TEMPURA_CCD_CORES env var.
/// "0-15"  -> (16, Some(0..=15))   pin to CCD0
/// "16-31" -> (16, Some(16..=31))  pin to CCD1
/// "0-31"  -> (32, Some(0..=31))   pin all, no migration
/// unset   -> (available, None)    no pinning (default)
fn parse_ccd_cores_env() -> (usize, Option<Vec<usize>>) {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    match std::env::var("TEMPURA_CCD_CORES").ok().as_deref() {
        Some(s) => {
            let parts: Vec<&str> = s.splitn(2, '-').collect();
            if parts.len() == 2 {
                if let (Ok(start), Ok(end)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>())
                {
                    let cores: Vec<usize> = (start..=end).collect();
                    let n = cores.len();
                    return (n, Some(cores));
                }
            }
            (available, None)
        }
        None => (available, None),
    }
}

pub struct EngineTask {
    model: Arc<dyn InferenceModel>,
    rx: mpsc::Receiver<EngineRequest>,
    /// Shared with AppState (Prometheus gauges).  Lock is held ONLY for
    /// scheduler operations, never across model forwards.
    scheduler: Arc<Mutex<BatchScheduler>>,
    /// Full request payloads; the scheduler queues carry only the
    /// lightweight InferenceRequest projection.
    payloads: HashMap<String, EngineRequest>,
    /// Process-shared GPU KV page pool (p24-kv-swap).  None = CPU model,
    /// pool creation failed, or TEMPURA_NO_KV_POOL=1 -- the engine runs
    /// the per-session CPU path.
    #[cfg(feature = "cutile")]
    kv_pool: Option<Arc<Mutex<tempura_engine::cutile_kv_cache::GpuKvPagePool>>>,
    #[cfg(feature = "cutile")]
    kv_pool_tried: bool,
    /// Per-session pooled KV runs, keyed by request id.  Kept OUT of
    /// ActiveSession so a decode batch can move them into a contiguous
    /// slice for forward_batch_decode (and so dropping a session frees
    /// its run regardless of the sessions map).
    #[cfg(feature = "cutile")]
    kv_caches: HashMap<String, tempura_engine::cutile_kv_cache::CutileKvCache>,
    /// Draft model for speculative decoding (--draft-model; None =
    /// speculation off, the default).
    #[cfg(feature = "cutile")]
    draft_model: Option<Arc<dyn InferenceModel>>,
    /// The draft's own KV page pool (dims come from the draft config).
    /// Created lazily AFTER the target pool so pool sizing sees the
    /// target's allocation first (VRAM order matters on one card).
    #[cfg(feature = "cutile")]
    draft_kv_pool: Option<Arc<Mutex<tempura_engine::cutile_kv_cache::GpuKvPagePool>>>,
    #[cfg(feature = "cutile")]
    draft_kv_pool_tried: bool,
    /// Per-request draft state (lazy: materialized on the session's first
    /// decode-ready tick).
    #[cfg(feature = "cutile")]
    draft_caches: HashMap<String, DraftCache>,
    /// EAGLE-3 head sessions (see eagle.rs; solo decode path only).
    #[cfg(feature = "cutile")]
    eagle_states: HashMap<String, EagleSessionState>,
    /// The head's shared 1-layer KV pool.
    #[cfg(feature = "cutile")]
    eagle_kv_pool: Option<Arc<Mutex<tempura_engine::cutile_kv_cache::GpuKvPagePool>>>,
    #[cfg(feature = "cutile")]
    eagle_kv_pool_tried: bool,
    #[cfg(feature = "cutile")]
    spec_stats: SpecStats,
}

impl EngineTask {
    /// `draft_model` is only used on the cutile (GPU pooled) path; on CPU
    /// builds it is ignored (speculation is GPU-serving v1).
    pub fn new(
        model: Arc<dyn InferenceModel>,
        draft_model: Option<Arc<dyn InferenceModel>>,
        rx: mpsc::Receiver<EngineRequest>,
        scheduler: Arc<Mutex<BatchScheduler>>,
    ) -> Self {
        // inference_pool (staticpar) is initialized lazily on first GEMV call;
        // CCD pinning is handled via TEMPURA_CCD_CORES env var read by parse_ccd_cores_env
        // which is kept for future use but no longer wires a Rayon pool here.
        let _ = parse_ccd_cores_env();
        let _ = &draft_model;
        Self {
            model,
            #[cfg(feature = "cutile")]
            draft_model,
            rx,
            scheduler,
            payloads: HashMap::new(),
            #[cfg(feature = "cutile")]
            kv_pool: None,
            #[cfg(feature = "cutile")]
            kv_pool_tried: false,
            #[cfg(feature = "cutile")]
            kv_caches: HashMap::new(),
            #[cfg(feature = "cutile")]
            draft_kv_pool: None,
            #[cfg(feature = "cutile")]
            draft_kv_pool_tried: false,
            #[cfg(feature = "cutile")]
            draft_caches: HashMap::new(),
            #[cfg(feature = "cutile")]
            eagle_states: HashMap::new(),
            #[cfg(feature = "cutile")]
            eagle_kv_pool: None,
            #[cfg(feature = "cutile")]
            eagle_kv_pool_tried: false,
            #[cfg(feature = "cutile")]
            spec_stats: SpecStats::default(),
        }
    }

    /// Create the shared KV pool on first use (lazily: the model may not
    /// be fully loaded at construction).  Honors TEMPURA_NO_KV_POOL.
    #[cfg(feature = "cutile")]
    fn ensure_kv_pool(&mut self) {
        if self.kv_pool_tried {
            return;
        }
        self.kv_pool_tried = true;
        if std::env::var("TEMPURA_NO_KV_POOL").as_deref() == Ok("1") {
            return;
        }
        self.kv_pool = self.model.new_kv_pool();
        if self.kv_pool.is_some() {
            eprintln!("[engine] multi-session GPU KV pool enabled (p24-kv-swap)");
        }
    }

    #[cfg(not(feature = "cutile"))]
    fn ensure_kv_pool(&mut self) {}

    /// Speculative decode depth K (number of draft tokens per verify
    /// cycle).  TEMPURA_SPEC_K bisects; default 4.
    #[cfg(feature = "cutile")]
    fn spec_k(&self) -> usize {
        std::env::var("TEMPURA_SPEC_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|k| (2..=16).contains(k))
            .unwrap_or(4)
    }

    /// Create the DRAFT's KV pool on first use (after the target pool --
    /// VRAM order).  Honors TEMPURA_NO_KV_POOL.
    #[cfg(feature = "cutile")]
    fn ensure_draft_kv_pool(&mut self) {
        if self.draft_kv_pool_tried || self.draft_model.is_none() {
            return;
        }
        self.draft_kv_pool_tried = true;
        if std::env::var("TEMPURA_NO_KV_POOL").as_deref() == Ok("1") {
            return;
        }
        self.draft_kv_pool = self
            .draft_model
            .as_ref()
            .and_then(|m| m.new_kv_pool());
        if self.draft_kv_pool.is_some() {
            eprintln!(
                "[engine] speculative decoding enabled (draft pool up, K={})",
                self.spec_k()
            );
        }
    }

    #[cfg(feature = "cutile")]
    fn pool_free_pages(&self) -> usize {
        self.kv_pool
            .as_ref()
            .and_then(|p| p.lock().ok().map(|p| p.free_pages()))
            .unwrap_or(0)
    }

    #[cfg(feature = "cutile")]
    fn pages_for(tokens: usize) -> usize {
        tempura_engine::cutile_kv_cache::CutileKvCache::pages_for_tokens(tokens)
    }

    /// Swap out one scheduler-selected LRU/priority victim (not in
    /// `protect`).  Returns true when a victim was swapped (or stale
    /// accounting was cleaned up) and the caller should retry.
    #[cfg(feature = "cutile")]
    fn evict_one(&mut self, protect: &[String]) -> bool {
        let protect_refs: Vec<&str> = protect.iter().map(|s| s.as_str()).collect();
        let victim = {
            let Ok(sched) = self.scheduler.lock() else {
                return false;
            };
            sched.select_swap_victim(&protect_refs)
        };
        let Some(victim) = victim else {
            return false;
        };
        match self.kv_caches.get_mut(&victim) {
            Some(vkv) => {
                let live = vkv.seq_len();
                let freed = vkv.run_pages();
                match vkv.swap_out() {
                    Ok(()) => {
                        if let Ok(mut sched) = self.scheduler.lock() {
                            sched.mark_swapped(&victim);
                        }
                        eprintln!(
                            "[engine] swapped out {victim} ({live} tokens, {freed} \
                             pages) to the DRAM tier"
                        );
                        true
                    }
                    Err(e) => {
                        eprintln!("[engine] swap_out failed for {victim}: {e}");
                        false
                    }
                }
            }
            None => {
                // Victim has no GPU run (stale accounting) --
                // mark it swapped so it stops being selected.
                if let Ok(mut sched) = self.scheduler.lock() {
                    sched.mark_swapped(&victim);
                }
                true
            }
        }
    }

    /// Ensure the run for `id` can hold `tokens`; on pool exhaustion,
    /// swap out a scheduler-selected victim (never one of `protect`) and
    /// retry.  Returns false when no victim can satisfy the growth (the
    /// caller defers this sequence to a later tick).
    #[cfg(feature = "cutile")]
    fn ensure_capacity_with_eviction(
        &mut self,
        id: &str,
        tokens: usize,
        protect: &[String],
    ) -> bool {
        loop {
            let Some(kv) = self.kv_caches.get_mut(id) else {
                return false;
            };
            match kv.try_ensure_capacity(tokens) {
                Ok(()) => return true,
                Err(tempura_engine::cutile_kv_cache::EnsureCapacityError::Kernel(e)) => {
                    eprintln!("[engine] KV run growth kernel failure for {id}: {e}");
                    return false;
                }
                Err(tempura_engine::cutile_kv_cache::EnsureCapacityError::Exhausted(info)) => {
                    if !self.evict_one(protect) {
                        eprintln!(
                            "[engine] KV pool exhausted (need {} pages, {} free) and no \
                             eviction candidate -- deferring {id}",
                            info.need_pages, info.free_pages
                        );
                        return false;
                    }
                }
            }
        }
    }

    /// Resume swapped sequences while the pool has headroom (priority
    /// order).  The +256-token margin matches the fresh-run headroom so
    /// a resumed sequence does not immediately re-exhaust the pool.
    #[cfg(feature = "cutile")]
    fn try_swap_ins(&mut self, sessions: &HashMap<String, ActiveSession>) {
        let candidates = {
            let Ok(sched) = self.scheduler.lock() else {
                return;
            };
            sched.swapped_ids()
        };
        for id in candidates {
            let Some(entry) = sessions.get(&id) else {
                continue;
            };
            let want = {
                let Some(kv) = self.kv_caches.get(&id) else {
                    continue;
                };
                entry.session.position.max(kv.seq_len()) + 256
            };
            if self.pool_free_pages() < Self::pages_for(want) {
                break; // pool still tight; higher-priority candidates first
            }
            let Some(kv) = self.kv_caches.get_mut(&id) else {
                continue;
            };
            match kv.swap_in(want) {
                Ok(()) => {
                    let tokens = kv.seq_len();
                    if let Ok(mut sched) = self.scheduler.lock() {
                        sched.mark_resident(&id, tokens);
                    }
                }
                Err(e) => {
                    eprintln!("[engine] swap_in failed for {id}: {e}");
                    break;
                }
            }
        }
    }

    /// One chunk of prefill for `id` on its pooled run (GPU mode).
    /// Returns true when the request finished.
    #[cfg(feature = "cutile")]
    fn prefill_step_gpu(
        &mut self,
        sessions: &mut HashMap<String, ActiveSession>,
        id: &str,
    ) -> bool {
        let (start, end) = {
            let entry = sessions.get(id).expect("prefill session");
            let start = entry.prefill_offset;
            let end = (start + PREFILL_CHUNK_SIZE).min(entry.prefill_tokens.len());
            (start, end)
        };
        let chunk_len = end - start;
        if chunk_len == 0 {
            return false;
        }
        // Growth with eviction BEFORE the forward (the internal growth
        // inside forward then becomes a no-op -- its failure path panics).
        let seq_len = self.kv_caches.get(id).map(|kv| kv.seq_len()).unwrap_or(0);
        if !self.ensure_capacity_with_eviction(id, seq_len + chunk_len, &[id.to_string()]) {
            return false; // deferred to a later tick
        }
        let (chunk, done_prefill) = {
            let entry = sessions.get_mut(id).expect("prefill session");
            let chunk = entry.prefill_tokens[start..end].to_vec();
            entry.prefill_offset = end;
            (chunk, entry.prefill_offset >= entry.prefill_tokens.len())
        };
        let kv = self.kv_caches.get_mut(id).expect("kv");
        let entry = sessions.get_mut(id).expect("prefill session");
        let logits = match self.model.forward_with_kv(&mut entry.session, &chunk, kv) {
            Some(logits) => logits,
            None => {
                // Not a pooled GPU model after all -- CPU fallback.
                let logits = self.model.forward(&mut entry.session, &chunk);
                entry.prefill_done = done_prefill;
                if done_prefill {
                    return entry.sample_and_emit(self.model.as_ref(), logits);
                }
                return false;
            }
        };
        entry.prefill_done = done_prefill;
        if done_prefill {
            if let Ok(mut sched) = self.scheduler.lock() {
                sched.mark_decoding(id);
            }
        }
        // LRU clock + footprint (a mid-prefill session is an eviction
        // candidate -- its prefix lives in the run).
        if let Ok(mut sched) = self.scheduler.lock() {
            let tokens = self.kv_caches.get(id).map(|kv| kv.seq_len()).unwrap_or(0);
            sched.touch(id, tokens);
        }
        let finished = if done_prefill {
            entry.sample_and_emit(self.model.as_ref(), logits)
        } else {
            false
        };
        // EAGLE head prefill for this chunk (consumes the chunk's taps
        // immediately -- the next forward overwrites them).  Runs after
        // the entry borrow ends (the hook takes sessions immutably).
        #[cfg(feature = "cutile")]
        {
            self.eagle_prefill_hook(sessions, id, start, chunk_len, done_prefill);
        }
        finished
    }

    /// One batched decode step over the decode-ready resident sessions
    /// (GPU mode).  Returns the ids that finished.
    #[cfg(feature = "cutile")]
    fn decode_step_gpu(&mut self, sessions: &mut HashMap<String, ActiveSession>) -> Vec<String> {
        // Decode-ready: prefill finished, KV resident (not swapped).
        let ready: Vec<String> = match self.scheduler.lock() {
            Ok(sched) => sched
                .active_sequences()
                .iter()
                .filter(|s| s.state == SequenceState::Decode)
                .filter(|s| {
                    sessions
                        .get(&s.request_id)
                        .map(|e| e.prefill_done)
                        .unwrap_or(false)
                })
                .map(|s| s.request_id.clone())
                .collect(),
            Err(_) => Vec::new(),
        };
        if ready.is_empty() {
            return vec![];
        }
        let max_batch = {
            let Ok(sched) = self.scheduler.lock() else {
                return vec![];
            };
            sched.max_batch_size()
        };
        let batch: Vec<String> = ready.into_iter().take(max_batch).collect();

        // Pre-grow every run (eviction on exhaustion; the whole batch is
        // protected first -- swapping out a mid-batch member orphans its
        // baked CUDA-graph pointers).
        let mut stepped: Vec<String> = Vec::with_capacity(batch.len());
        for id in &batch {
            let seq_len = self.kv_caches.get(id).map(|kv| kv.seq_len()).unwrap_or(0);
            if self.ensure_capacity_with_eviction(id, seq_len + 1, &batch) {
                stepped.push(id.clone());
            } else {
                // All active sequences sit in this batch: no unprotected
                // victim exists.  Retry allowing batch-member victims --
                // safe because replay-time pointer validation re-captures
                // graphs after a swap, and the ragged no-graph path bakes
                // nothing.  Without this a full pool deadlocks the whole
                // batch (found by the p24 streaming A/B: run B admitted
                // 4/6 sessions and stalled with zero swaps).
                if self.ensure_capacity_with_eviction(id, seq_len + 1, &[id.clone()]) {
                    stepped.push(id.clone());
                } else {
                    break; // keep batch order stable; defer the rest
                }
            }
        }
        // A retry may have swapped out a batch member -- drop it from
        // this tick's batch (it resumes via swap-in).
        if !stepped.is_empty() {
            let swapped: Vec<String> = match self.scheduler.lock() {
                Ok(s) => s
                    .active_sequences()
                    .iter()
                    .filter(|seq| seq.state == SequenceState::Swapped)
                    .map(|seq| seq.request_id.clone())
                    .collect(),
                Err(_) => Vec::new(),
            };
            stepped.retain(|id| !swapped.contains(id));
        }
        // -- Speculative split (greedy) -----------------------------------
        // Decode-ready PURE-GREEDY sessions speculate when a draft model
        // is loaded: ALL of them run one BATCHED spec cycle (K batched
        // draft steps across sessions, then ONE multi-session verify --
        // see spec_batch_step_gpu).  Everything else -- sampled sessions
        // included, speculation is greedy-only -- takes the plain batch
        // step below.
        let (spec_ids, stepped): (Vec<String>, Vec<String>) = if self.draft_model.is_some() {
            stepped.into_iter().partition(|id| {
                sessions
                    .get(id)
                    .map(|e| e.params.is_greedy())
                    .unwrap_or(false)
            })
        } else {
            (Vec::new(), stepped)
        };

        let mut completed = Vec::new();
        // -- EAGLE-3 solo cycle split ---------------------------------------
        // The structural accounting (docs/status 2026-11-23): a verify row
        // costs ~a plain decode row and does not amortize at serving batch
        // sizes, so speculation cannot win there.  The SOLO case is
        // different (the m=1 graph reads the full weights once per token)
        // -- EAGLE's nearly-free drafting plus one k-row verify reading
        // the weights ONCE is a genuine win there.  Solo only: exactly
        // one greedy decode-ready session with a live head state runs the
        // EAGLE cycle; multi-client batches keep plain decode (with head
        // resync maintenance so a session re-enters speculation when it
        // becomes solo again).
        #[cfg(feature = "cutile")]
        let mut eagle_id: Option<String> = None;
        #[cfg(feature = "cutile")]
        {
            if std::env::var("TEMPURA_EAGLE_DEBUG").is_ok() {
                let greedy = stepped
                    .iter()
                    .filter(|id| {
                        sessions.get(*id).map(|e| e.params.is_greedy()).unwrap_or(false)
                    })
                    .count();
                eprintln!(
                    "[eagle-split] stepped={} greedy={} active={} draft={}",
                    stepped.len(),
                    greedy,
                    self.model.eagle_active(),
                    self.draft_model.is_some()
                );
            }
            if self.draft_model.is_none()
                && self.model.eagle_active()
                && std::env::var("TEMPURA_NO_EAGLE").as_deref() != Ok("1")
            {
                let greedy_solo = stepped
                    .iter()
                    .filter(|id| {
                        sessions
                            .get(*id)
                            .map(|e| e.params.is_greedy())
                            .unwrap_or(false)
                    })
                    .count();
                if greedy_solo == 1 {
                    let id = stepped
                        .iter()
                        .find(|id| {
                            sessions
                                .get(*id)
                                .map(|e| e.params.is_greedy())
                                .unwrap_or(false)
                        })
                        .unwrap()
                        .clone();
                    match self.eagle_states.get(&id) {
                        Some(es) if es.prefilled && !es.failed => {
                            eagle_id = Some(id);
                        }
                        other => {
                            if std::env::var("TEMPURA_EAGLE_DEBUG").is_ok() {
                                eprintln!(
                                    "[eagle] solo session {id} not eligible: {:?}",
                                    other.map(|e| (e.prefilled, e.failed, e.kv.seq_len()))
                                );
                            }
                        }
                    }
                }
            }
        }
        #[cfg(feature = "cutile")]
        let stepped: Vec<String> = match &eagle_id {
            Some(id) => stepped.into_iter().filter(|s| s != id).collect(),
            None => stepped,
        };
        if !stepped.is_empty() {
            completed.extend(self.plain_batch_step_gpu(sessions, stepped));
        }
        if !spec_ids.is_empty() {
            completed.extend(self.spec_batch_step_gpu(sessions, spec_ids));
        }
        #[cfg(feature = "cutile")]
        if let Some(id) = eagle_id {
            if self.eagle_step_gpu(sessions, &id) {
                completed.push(id);
            } else if self.plain_step_gpu(sessions, &id) {
                // The cycle could not run (deferred/capacity); plain step
                // + head resync keep the session alive and in sync.
                completed.push(id);
            }
        }
        completed
    }

    /// The plain (non-speculative) batched decode step over `stepped`
    /// (extracted unchanged from decode_step_gpu when the speculative
    /// split was added).  Returns the ids that finished.
    #[cfg(feature = "cutile")]
    fn plain_batch_step_gpu(
        &mut self,
        sessions: &mut HashMap<String, ActiveSession>,
        stepped: Vec<String>,
    ) -> Vec<String> {
        let dbg_timing = std::env::var("TEMPURA_ENGINE_DEBUG_TICKS").is_ok();
        let t_prep = std::time::Instant::now();

        let mut toks = Vec::with_capacity(stepped.len());
        let mut pos = Vec::with_capacity(stepped.len());
        for id in &stepped {
            let entry = sessions.get(id).expect("decode session");
            toks.push(*entry.generated.last().expect("decode has a last token"));
            pos.push(entry.session.position as u32);
        }
        // Move the stepped runs into a contiguous slice (HashMap values
        // cannot be borrowed &mut several at once); re-inserted below.
        let mut kvs: Vec<tempura_engine::cutile_kv_cache::CutileKvCache> = stepped
            .iter()
            .map(|id| self.kv_caches.remove(id).expect("kv"))
            .collect();
        if dbg_timing {
            eprintln!("[tick-timing] m={} prep={:?}", stepped.len(), t_prep.elapsed());
        }
        let t_fwd = std::time::Instant::now();
        // Pure-greedy sessions (temperature 0, no penalties -- the A/B and
        // smoke workloads) get their token from the GPU greedy-argmax
        // kernel: 4 bytes downloaded instead of a 513KB logits row, and no
        // CPU softmax/top-k over 128k floats x m (was ~11 ms/step at m=6).
        // The engine path never needs logprobs, so greedy rows never
        // materialize logits on the host.
        let greedy: Vec<bool> = stepped
            .iter()
            .map(|id| sessions.get(id).expect("decode session").params.is_greedy())
            .collect();
        let all_outputs = match self
            .model
            .forward_batch_with_kvs_greedy(&toks, &pos, &mut kvs, &greedy)
        {
            Some(l) => l,
            None => {
                // kvs was built in stepped order; zip (not remove(i) --
                // removals shift the remaining elements) restores each run
                // to its own session.
                for (id, kv) in stepped.iter().zip(kvs) {
                    self.kv_caches.insert(id.clone(), kv);
                }
                return vec![];
            }
        };
        if dbg_timing {
            eprintln!("[tick-timing] forward={:?}", t_fwd.elapsed());
        }
        // EAGLE head maintenance: every eagle session in this plain batch
        // resyncs its head row from its tap row (the head must track every
        // target position to stay speculation-eligible when it next
        // becomes solo).  Tap rows are valid for exactly this batch's m.
        #[cfg(feature = "cutile")]
        {
            let tap_rows = self.model.eagle_tap_rows();
            for (i, id) in stepped.iter().enumerate() {
                let Some(es) = self.eagle_states.get_mut(id) else {
                    continue;
                };
                if es.failed || !es.prefilled {
                    continue;
                }
                if tap_rows <= i
                    || !self.model.eagle_resync_plain(
                        &mut es.kv,
                        &es.anchor,
                        toks[i],
                        i,
                        pos[i] as usize,
                    )
                {
                    es.failed = true;
                }
            }
        }
        let t_sample = std::time::Instant::now();

        let mut completed = Vec::new();
        // remove(i) inside this loop would shift kvs after every removal
        // (found by the streaming A/B: removal index 1 < len 1 panicked the
        // engine thread on the first 2-sequence batch).
        for (i, (id, kv)) in stepped.iter().zip(kvs).enumerate() {
            let fresh = self.kv_caches.insert(id.clone(), kv).is_none();
            debug_assert!(fresh, "kv re-inserted over an existing entry");
            let entry = sessions.get_mut(id).expect("decode session");
            // forward_batch_decode is stateful: it advanced the KV run.
            // Mirror the wrapper's single-seq bookkeeping.
            entry.session.cache.advance(1);
            entry.session.position += 1;
            if std::env::var("TEMPURA_ENGINE_DEBUG_TICKS").is_ok() {
                // Per-step output fingerprint: diffing run A vs run B logs
                // pinpoints the first divergent decode step (p24 A/B hunt).
                // Greedy rows have no host logits -- fingerprint the token.
                match &all_outputs[i] {
                    tempura_engine::BatchDecodeOutput::Logits(logits) => {
                        let cs: u64 = logits
                            .iter()
                            .fold(0u64, |a, v| a.wrapping_add(v.to_bits() as u64));
                        eprintln!(
                            "[logits-cs] {} step={} cs={cs:016x}",
                            &id[id.len().saturating_sub(6)..],
                            entry.generated.len()
                        );
                    }
                    tempura_engine::BatchDecodeOutput::Token(t) => {
                        eprintln!(
                            "[logits-cs] {} step={} tok={t}",
                            &id[id.len().saturating_sub(6)..],
                            entry.generated.len()
                        );
                    }
                }
            }
            let done = match &all_outputs[i] {
                tempura_engine::BatchDecodeOutput::Token(token) => entry.emit_token(*token),
                tempura_engine::BatchDecodeOutput::Logits(logits) => {
                    entry.sample_and_emit(self.model.as_ref(), logits.clone())
                }
            };
            if done {
                completed.push(id.clone());
            }
        }

        // LRU clock + footprint for the eviction policy.
        if dbg_timing {
            eprintln!("[tick-timing] sample+emit={:?}", t_sample.elapsed());
        }
        {
            let Ok(mut sched) = self.scheduler.lock() else {
                return completed;
            };
            for id in &stepped {
                let tokens = self.kv_caches.get(id).map(|kv| kv.seq_len()).unwrap_or(0);
                sched.touch(id, tokens);
            }
        }
        completed
    }

    /// Lazy draft-state materialization + prompt prefill for `id`.
    /// Returns true when the draft run is prefilled and in sync with the
    /// target's prompt.  On failure the draft state is dropped and the
    /// session stops speculating (plain decode).
    #[cfg(feature = "cutile")]
    fn ensure_draft_state(
        &mut self,
        draft_model: &Arc<dyn InferenceModel>,
        sessions: &HashMap<String, ActiveSession>,
        id: &str,
    ) -> bool {
        if !self.draft_caches.contains_key(id) {
            self.ensure_draft_kv_pool();
            let dc = self
                .draft_kv_pool
                .clone()
                .and_then(|pool| draft_model.new_session_kv_cache(&pool, 256))
                .map(|kv| DraftCache {
                    session: InferenceSession::new(draft_model.config(), TraceLevel::Off, 0),
                    kv: Some(kv),
                    prefilled: false,
                });
            let Some(dc) = dc else {
                return false;
            };
            self.draft_caches.insert(id.to_string(), dc);
        }
        let dc = self.draft_caches.get_mut(id).expect("draft cache");
        if dc.prefilled {
            return true;
        }
        let entry = sessions.get(id).expect("spec decode session");
        let prompt_len = entry.prefill_tokens.len();
        let mut done = true;
        {
            let kv = dc.kv.as_mut().expect("draft kv");
            for chunk in entry.prefill_tokens.chunks(PREFILL_CHUNK_SIZE) {
                if draft_model
                    .forward_with_kv(&mut dc.session, chunk, kv)
                    .is_none()
                {
                    done = false;
                    break;
                }
            }
        }
        if !done || dc.kv.as_ref().unwrap().seq_len() != prompt_len {
            // Draft prefill failed -- stop speculating on this session
            // entirely (fall back to plain decode).
            self.draft_caches.remove(id);
            return false;
        }
        dc.prefilled = true;
        true
    }

    /// One BATCHED speculative decode cycle across all spec-eligible
    /// sessions (v2 -- v1's per-session cycle serialized in the engine
    /// loop, 0.27x at 6 clients):
    ///
    ///   1. (lazy, per session) prefill the draft with the session's
    ///      prompt;
    ///   2. the draft proposes K tokens for EVERY session in K batched
    ///      forwards (m = live sessions, one launch chain per step
    ///      instead of K per session);
    ///   3. the target VERIFIES all sessions in ONE batch-decode-shaped
    ///      forward: session i contributes j_i = K rows over its own run
    ///      (row r writes K/V at offset base_i + r, attends exactly
    ///      base_i + r + 1 tokens -- causal), through the same per-row
    ///      machinery the plain batch path dispatches;
    ///   4. per session: accept the longest prefix where d_i == row_i's
    ///      argmax, emit the target's argmax at the first mismatch, roll
    ///      BOTH KV runs back to base + emitted.
    ///
    /// By construction the emitted token sequence per session is
    /// IDENTICAL to plain greedy decode (the equivalence gate,
    /// scripts/spec-equivalence.sh); the draft's numerics only influence
    /// acceptance length.  Returns the ids that finished.  Sessions that
    /// cannot grow their target run are deferred to a later tick (v1
    /// semantics); sessions whose draft chain broke early (< 2
    /// proposals) fall back to one plain m=1 step.
    #[cfg(feature = "cutile")]
    fn spec_batch_step_gpu(
        &mut self,
        sessions: &mut HashMap<String, ActiveSession>,
        spec_ids: Vec<String>,
    ) -> Vec<String> {
        let Some(draft_model) = self.draft_model.clone() else {
            return Vec::new();
        };
        let mut completed = Vec::new();

        // (0) Still-resident, greedy, decode-ready filter.  A plain-batch
        // eviction (or this tick's earlier steps) may have swapped a
        // session out after the decode-ready list was built --
        // speculating on a swapped run would read freed pages.
        let mut act: Vec<String> = Vec::with_capacity(spec_ids.len());
        {
            let Ok(sched) = self.scheduler.lock() else {
                return completed;
            };
            let decode_ids: std::collections::HashSet<&str> = sched
                .active_sequences()
                .iter()
                .filter(|s| s.state == SequenceState::Decode)
                .map(|s| s.request_id.as_str())
                .collect();
            for id in spec_ids {
                if decode_ids.contains(id.as_str())
                    && sessions.get(&id).map(|e| e.params.is_greedy()).unwrap_or(false)
                    && self.kv_caches.contains_key(&id)
                {
                    act.push(id);
                }
            }
        }
        if act.is_empty() {
            return completed;
        }

        // (1) Lazy draft-state materialization + prompt prefill.
        let mut ready: Vec<String> = Vec::with_capacity(act.len());
        for id in act {
            if self.ensure_draft_state(&draft_model, sessions, &id) {
                ready.push(id);
            } else if self.plain_step_gpu(sessions, &id) {
                completed.push(id);
            }
        }
        if ready.is_empty() {
            return completed;
        }

        let k = self.spec_k();
        let n = ready.len();
        let dbg_timing = std::env::var("TEMPURA_ENGINE_DEBUG_TICKS").is_ok();
        let t_prep = std::time::Instant::now();

        // (2) Resync every draft run to its target's base: a previous
        // tick's fallback (plain step after a failed verify/draft) leaves
        // the draft ahead of the target -- roll it back so it re-feeds
        // from the actually-emitted token (stale K/V rows are
        // overwritten).  Then collect the feeds (each session's last
        // emitted token) and bases.
        let mut feeds = vec![0u32; n];
        for (i, id) in ready.iter().enumerate() {
            let entry = sessions.get(id).expect("spec decode session");
            feeds[i] = *entry.generated.last().expect("decode has a last token");
            let base = self
                .kv_caches
                .get(id)
                .map(|kv| kv.seq_len())
                .unwrap_or(0);
            debug_assert_eq!(base, sessions.get(id).unwrap().session.position);
            let dc = self.draft_caches.get_mut(id).expect("draft cache");
            let dkv = dc.kv.as_mut().expect("draft kv");
            if dkv.seq_len() > base {
                dkv.rollback(base);
            }
            debug_assert_eq!(dkv.seq_len(), base);
        }

        // (3) K BATCHED draft steps.  Live draft runs are held contiguous
        // in `dkvs[..m]` (owner[s] maps slot -> session); a session that
        // stops drafting (pool tight, non-token row) is swap-removed to
        // the tail and keeps whatever proposals it already collected.
        let mut dkvs: Vec<tempura_engine::cutile_kv_cache::CutileKvCache> =
            Vec::with_capacity(n);
        for id in &ready {
            dkvs.push(
                self.draft_caches
                    .get_mut(id)
                    .expect("draft cache")
                    .kv
                    .take()
                    .expect("draft kv"),
            );
        }
        let mut owner: Vec<usize> = (0..n).collect();
        let mut drafting: Vec<bool> = vec![true; n];
        let mut drafts: Vec<Vec<u32>> = vec![Vec::with_capacity(k); n];
        let t_draft = std::time::Instant::now();
        let mut m = n;
        'draft: for _step in 0..k {
            let mut s = 0;
            while s < m {
                let next = dkvs[s].seq_len() + 1;
                if dkvs[s].try_ensure_capacity(next).is_err() {
                    // draft pool tight: this session keeps fewer proposals
                    let sess = owner[s];
                    drafting[sess] = false;
                    m -= 1;
                    dkvs.swap(s, m);
                    owner.swap(s, m);
                } else {
                    s += 1;
                }
            }
            if m == 0 {
                break;
            }
            let toks: Vec<u32> = (0..m).map(|s| feeds[owner[s]]).collect();
            let pos: Vec<u32> = dkvs[..m].iter().map(|kv| kv.seq_len() as u32).collect();
            let greedy = vec![true; m];
            let out = draft_model.forward_batch_with_kvs_greedy(
                &toks,
                &pos,
                &mut dkvs[..m],
                &greedy,
            );
            match out {
                Some(rows) => {
                    for (s, row) in rows.into_iter().enumerate() {
                        let sess = owner[s];
                        match row {
                            tempura_engine::BatchDecodeOutput::Token(t) => {
                                drafts[sess].push(t);
                                feeds[sess] = t;
                            }
                            // unreachable with greedy rows; stop drafting
                            // for this session (keeps its proposals)
                            tempura_engine::BatchDecodeOutput::Logits(_) => {
                                drafting[sess] = false;
                            }
                        }
                    }
                    let mut s = 0;
                    while s < m {
                        if !drafting[owner[s]] {
                            m -= 1;
                            dkvs.swap(s, m);
                            owner.swap(s, m);
                        } else {
                            s += 1;
                        }
                    }
                    if m == 0 {
                        break 'draft;
                    }
                }
                // Nothing advanced on None -- every session keeps its
                // (possibly empty) proposal set; short ones fall back
                // to a plain step below.
                None => break 'draft,
            }
        }
        let draft_us = t_draft.elapsed().as_micros() as u64;
        if dbg_timing {
            eprintln!(
                "[spec-timing] m={m}/{n} prep={:?} draft={:?}",
                t_prep.elapsed(),
                t_draft.elapsed()
            );
        }
        // Restore every draft run to its DraftCache (slots m..n hold the
        // retired ones).
        for (s, kv) in dkvs.into_iter().enumerate() {
            let id = &ready[owner[s]];
            self.draft_caches.get_mut(id).expect("draft cache").kv = Some(kv);
        }

        // (4) Partition: sessions with >= 2 proposals verify; the rest
        // take one plain m=1 step (deferred to after the verify so the
        // verify set cannot be evicted by a fallback's growth).
        let mut verify: Vec<(String, Vec<u32>)> = Vec::with_capacity(n);
        let mut fallback: Vec<String> = Vec::new();
        for (i, id) in ready.iter().enumerate() {
            if drafts[i].len() >= 2 {
                verify.push((id.clone(), std::mem::take(&mut drafts[i])));
            } else {
                fallback.push(id.clone());
            }
        }

        // (5) The batched verify.  Pre-grow every target run for its
        // whole block first (eviction-aware, the whole verify set
        // protected); growth failures defer to a later tick.
        let protect: Vec<String> = verify.iter().map(|(id, _)| id.clone()).collect();
        let t_pregrow = std::time::Instant::now();
        let mut grown: Vec<(String, Vec<u32>)> = Vec::with_capacity(verify.len());
        for (id, dr) in verify {
            let base = self.kv_caches.get(&id).map(|kv| kv.seq_len()).unwrap_or(0);
            if self.ensure_capacity_with_eviction(&id, base + dr.len(), &protect) {
                grown.push((id, dr));
            }
        }
        // A retry may have swapped a verify member out -- drop it from
        // this cycle (it resumes via swap-in; its draft run resyncs next
        // tick).
        if !grown.is_empty() {
            let swapped: Vec<String> = match self.scheduler.lock() {
                Ok(s) => s
                    .active_sequences()
                    .iter()
                    .filter(|seq| seq.state == SequenceState::Swapped)
                    .map(|seq| seq.request_id.clone())
                    .collect(),
                Err(_) => Vec::new(),
            };
            grown.retain(|(id, _)| !swapped.contains(id));
        }

        let mut verify_us = 0u64;
        if !grown.is_empty() {
            let t_verify = std::time::Instant::now();
            let ids: Vec<String> = grown.iter().map(|(id, _)| id.clone()).collect();
            let token_lists: Vec<Vec<u32>> = grown
                .iter()
                .map(|(id, dr)| {
                    let entry = sessions.get(id).expect("spec decode session");
                    let last = *entry.generated.last().expect("decode has a last token");
                    let j = dr.len();
                    let mut v = Vec::with_capacity(j);
                    v.push(last);
                    v.extend_from_slice(&dr[..j - 1]);
                    v
                })
                .collect();
            let mut kvs: Vec<tempura_engine::cutile_kv_cache::CutileKvCache> = grown
                .iter()
                .map(|(id, _)| self.kv_caches.remove(id).expect("target kv"))
                .collect();
            let argmaxes = self
                .model
                .forward_spec_verify_greedy_multi(&token_lists, &mut kvs);
            match argmaxes {
                Some(per_session) => {
                    debug_assert_eq!(per_session.len(), kvs.len());
                    for (((id, dr), kv), argmaxes) in
                        grown.into_iter().zip(kvs).zip(per_session.into_iter())
                    {
                        // accept the longest matching prefix; emit the
                        // target's argmax at the first mismatch
                        let j = dr.len();
                        let base = kv.seq_len() - j; // the verify advanced by j
                        let mut a = 0;
                        while a < j && argmaxes[a] == dr[a] {
                            a += 1;
                        }
                        let emitted: Vec<u32> = if a == j {
                            dr.clone()
                        } else {
                            let mut v = dr[..a].to_vec();
                            v.push(argmaxes[a]);
                            v
                        };
                        let n_em = emitted.len();
                        let mut kv = kv;
                        kv.rollback(base + n_em);
                        self.kv_caches.insert(id.clone(), kv);
                        if let Some(dc) = self.draft_caches.get_mut(&id) {
                            if let Some(dkv) = dc.kv.as_mut() {
                                dkv.rollback(base + n_em);
                            }
                        }
                        let entry = sessions.get_mut(&id).expect("spec decode session");
                        entry.session.cache.advance(n_em);
                        entry.session.position += n_em;
                        self.spec_stats.note(n_em, k);
                        let mut done = false;
                        for tok in emitted {
                            if entry.emit_token(tok) {
                                done = true;
                                break;
                            }
                        }
                        if done {
                            completed.push(id.clone());
                        }
                        if let Ok(mut sched) = self.scheduler.lock() {
                            let tokens =
                                self.kv_caches.get(&id).map(|kv| kv.seq_len()).unwrap_or(0);
                            sched.touch(&id, tokens);
                        }
                    }
                }
                None => {
                    // Unsupported shape -- plain steps for everyone.
                    for (id, kv) in ids.into_iter().zip(kvs) {
                        self.kv_caches.insert(id.clone(), kv);
                        fallback.push(id);
                    }
                }
            }
            verify_us = t_verify.elapsed().as_micros() as u64;
            if dbg_timing {
                eprintln!(
                    "[spec-timing] rows={} pregrow={:?} verify={:?}",
                    token_lists.iter().map(|t| t.len()).sum::<usize>(),
                    t_pregrow.elapsed(),
                    t_verify.elapsed()
                );
            }
        }
        self.spec_stats.add_timing(draft_us, verify_us);

        for id in &fallback {
            if self.plain_step_gpu(sessions, id) {
                completed.push(id.clone());
            }
        }
        completed
    }

    /// Plain single-session m=1 decode step (the speculation fallback:
    /// identical to what the plain batch would do for this session alone).
    #[cfg(feature = "cutile")]
    fn plain_step_gpu(&mut self, sessions: &mut HashMap<String, ActiveSession>, id: &str) -> bool {
        let base = self.kv_caches.get(id).map(|kv| kv.seq_len()).unwrap_or(0);
        if !self.ensure_capacity_with_eviction(id, base + 1, &[id.to_string()]) {
            return false;
        }
        let last = {
            let entry = sessions.get(id).expect("decode session");
            *entry.generated.last().expect("decode has a last token")
        };
        let kv = self.kv_caches.get_mut(id).expect("target kv");
        let out = self
            .model
            .forward_batch_with_kvs_greedy(&[last], &[base as u32], std::slice::from_mut(kv), &[true]);
        let Some(rows) = out else {
            return false;
        };
        let entry = sessions.get_mut(id).expect("decode session");
        entry.session.cache.advance(1);
        entry.session.position += 1;
        // EAGLE head maintenance for the fallback step (m=1: tap row 0).
        #[cfg(feature = "cutile")]
        {
            let tap_rows = self.model.eagle_tap_rows();
            if let Some(es) = self.eagle_states.get_mut(id) {
                if !es.failed && es.prefilled {
                    if tap_rows < 1
                        || !self
                            .model
                            .eagle_resync_plain(&mut es.kv, &es.anchor, last, 0, base)
                    {
                        es.failed = true;
                    }
                }
            }
        }
        match rows.into_iter().next() {
            Some(tempura_engine::BatchDecodeOutput::Token(t)) => entry.emit_token(t),
            Some(tempura_engine::BatchDecodeOutput::Logits(logits)) => {
                entry.sample_and_emit(self.model.as_ref(), logits)
            }
            None => false,
        }
    }

    // -- EAGLE-3 solo speculative cycle -----------------------------------

    /// Lazily materialize the head's shared KV pool + a session's head
    /// state.  False = the session cannot speculate (head state missing
    /// or latched-failed).
    #[cfg(feature = "cutile")]
    fn ensure_eagle_state(&mut self, id: &str, initial_tokens: usize) -> bool {
        if let Some(es) = self.eagle_states.get(id) {
            return !es.failed;
        }
        if !self.eagle_kv_pool_tried {
            self.eagle_kv_pool_tried = true;
            self.eagle_kv_pool = self.model.eagle_new_kv_pool();
            if self.eagle_kv_pool.is_none() {
                eprintln!("[eagle] head KV pool unavailable; speculation disabled");
            }
        }
        let Some(pool) = self.eagle_kv_pool.clone() else {
            return false;
        };
        let Some((kv, anchor)) = self.model.eagle_new_session(&pool, initial_tokens) else {
            return false;
        };
        self.eagle_states.insert(
            id.to_string(),
            EagleSessionState {
                kv,
                anchor,
                prefilled: false,
                failed: false,
            },
        );
        true
    }

    /// Head prefill for one target-prefill chunk: consumes the chunk's
    /// taps right after the target forward wrote them.  The final chunk
    /// processes one row fewer (row L-1 becomes the first draft step,
    /// conditioned on the carry).
    #[cfg(feature = "cutile")]
    fn eagle_prefill_hook(
        &mut self,
        sessions: &HashMap<String, ActiveSession>,
        id: &str,
        chunk_start: usize,
        chunk_len: usize,
        done: bool,
    ) {
        if !self.model.eagle_active()
            || std::env::var("TEMPURA_NO_EAGLE").as_deref() == Ok("1")
        {
            return;
        }
        let prompt: Vec<u32> = {
            let entry = sessions.get(id).expect("prefill session");
            entry.prefill_tokens.clone()
        };
        if !self.ensure_eagle_state(id, prompt.len() + 16) {
            return;
        }
        // Head rows for this chunk: rows chunk_start..chunk_start+chunk_len,
        // row p input (prompt[p], h_p) -- NO token shift under the V2
        // alignment (the head's row p predicts token_{p+1}).
        let prompt_tokens: Vec<u32> = prompt[chunk_start..chunk_start + chunk_len].to_vec();
        let es = self.eagle_states.get_mut(id).expect("eagle state");
        if !self.model.eagle_prefill_chunk(
            &mut es.kv,
            &es.anchor,
            &prompt_tokens,
            chunk_start,
            chunk_len,
        ) {
            // Taps stale (CPU-fallback prefill, oversized chunk) -- the
            // head cannot be prefilled; this session never speculates.
            eprintln!(
                "[eagle] prefill hook FAILED for {id}: chunk_start={chunk_start} chunk_len={chunk_len} rows={} tap_rows={}",
                prompt_tokens.len(),
                self.model.eagle_tap_rows()
            );
            es.failed = true;
            return;
        }
        if done {
            es.prefilled = true;
        }
    }

    /// One EAGLE-3 speculative decode cycle for the SOLO session `id`:
    /// K head draft steps, ONE k-row target verify over the session's run
    /// (the v1 `forward_spec_verify_greedy` machinery), longest-prefix
    /// acceptance, head resync + carry refresh.  By construction the
    /// emitted sequence is IDENTICAL to plain greedy decode (the
    /// spec-equivalence gate); the head's numerics only influence the
    /// acceptance length.  Returns true when the request finished.
    /// False = the cycle could not run (caller falls back to a plain
    /// step) OR ran but the request continues.
    #[cfg(feature = "cutile")]
    fn eagle_step_gpu(&mut self, sessions: &mut HashMap<String, ActiveSession>, id: &str) -> bool {
        let Some(base) = self.kv_caches.get(id).map(|kv| kv.seq_len()) else {
            return false;
        };
        if base == 0 {
            return false;
        }
        // Head-state invariant: head rows 0..base-1 (row p = (token_p,
        // h_p)); the anchor holds row base-1's output hidden.
        {
            let Some(es) = self.eagle_states.get(id) else {
                eprintln!("[eagle-step] {id}: no state");
                return false;
            };
            if es.failed || !es.prefilled || es.kv.seq_len() != base {
                eprintln!(
                    "[eagle-step] {id}: bad state prefilled={} failed={} head_len={} base={base}",
                    es.prefilled,
                    es.failed,
                    es.kv.seq_len()
                );
                if !es.failed && es.kv.seq_len() != base {
                    // Stale (swap/re-admission lost the run alignment) --
                    // latch failed; plain decode for the rest.
                    self.eagle_states.get_mut(id).unwrap().failed = true;
                }
                return false;
            }
        }
        let last = {
            let entry = sessions.get(id).expect("eagle session");
            *entry.generated.last().expect("decode has a last token")
        };
        let k = self.spec_k();
        // Pre-grow the target run for the whole verify block (eviction
        // protects this session) and the head run for K draft rows.
        if !self.ensure_capacity_with_eviction(id, base + k, &[id.to_string()]) {
            return false;
        }
        {
            let es = self.eagle_states.get_mut(id).expect("eagle state");
            if es.kv.try_ensure_capacity(base + k).is_err() {
                return false;
            }
        }
        let t_draft = std::time::Instant::now();
        let proposals = {
            let es = self.eagle_states.get_mut(id).expect("eagle state");
            match self
                .model
                .eagle_draft_cycle(&mut es.kv, &es.anchor, last, base, k)
            {
                Some(p) => p,
                None => {
                    eprintln!("[eagle-step] {id}: draft cycle returned None");
                    es.failed = true;
                    return false;
                }
            }
        };
        let draft_us = t_draft.elapsed().as_micros() as u64;
        if proposals.len() < 2 {
            // Not enough proposals to be worth a verify -- plain step.
            return false;
        }
        let j = proposals.len();
        let mut token_list = Vec::with_capacity(j);
        token_list.push(last);
        token_list.extend_from_slice(&proposals[..j - 1]);
        let mut kv = self.kv_caches.remove(id).expect("target kv");
        let t_verify = std::time::Instant::now();
        let argmaxes = self.model.forward_spec_verify_greedy(&token_list, &mut kv);
        let argmaxes = match argmaxes {
            Some(a) => a,
            None => {
                self.kv_caches.insert(id.to_string(), kv);
                return false;
            }
        };
        let verify_us = t_verify.elapsed().as_micros() as u64;
        self.spec_stats.add_timing(draft_us, verify_us);
        if std::env::var("TEMPURA_EAGLE_TRACE").is_ok() && base < 50 {
            eprintln!("[eagle] verify base={base}: argmaxes={:?} proposals={:?}", &argmaxes[..j.min(4)], &proposals[..j.min(4)]);
        }

        // Accept the longest prefix; emit the target's argmax at the
        // first mismatch (no bonus token on full acceptance -- K proposals
        // from K draft feeds, both KV runs roll back to base + emitted,
        // uniform).
        let mut a = 0;
        while a < j && argmaxes[a] == proposals[a] {
            a += 1;
        }
        let emitted: Vec<u32> = if a == j {
            proposals.clone()
        } else {
            let mut v = proposals[..a].to_vec();
            v.push(argmaxes[a]);
            v
        };
        let n_em = emitted.len();
        kv.rollback(base + n_em);
        self.kv_caches.insert(id.to_string(), kv);
        // Head resync (rows base-1..base+n_em-2 with the TRUE
        // conditioning) + carry refresh (verify tap row n_em-1).
        {
            let es = self.eagle_states.get_mut(id).expect("eagle state");
            if !self.model.eagle_post_verify(
                &mut es.kv,
                &es.anchor,
                last,
                &emitted,
                base,
                j,
            ) {
                es.failed = true;
            }
        }
        let entry = sessions.get_mut(id).expect("eagle session");
        entry.session.cache.advance(n_em);
        entry.session.position += n_em;
        self.spec_stats.note(n_em, k);
        let mut done = false;
        for tok in emitted {
            if entry.emit_token(tok) {
                done = true;
                break;
            }
        }
        if let Ok(mut sched) = self.scheduler.lock() {
            let tokens = self.kv_caches.get(id).map(|kv| kv.seq_len()).unwrap_or(0);
            sched.touch(id, tokens);
        }
        done
    }

    pub fn run(mut self) {
        let mut sessions: HashMap<String, ActiveSession> = HashMap::new();
        let debug_ticks = std::env::var("TEMPURA_ENGINE_DEBUG_TICKS").is_ok();
        let mut tick: u64 = 0;

        loop {
            tick += 1;
            // -- 1. Drain incoming requests into the payload map. ---------
            // The HTTP layer's acquire_slot already submitted+admitted
            // most requests; submit here only when the scheduler does
            // not know the id yet (direct channel users / tests).
            loop {
                match self.rx.try_recv() {
                    Ok(req) => {
                        let known = self
                            .scheduler
                            .lock()
                            .map(|s| s.in_system(&req.id))
                            .unwrap_or(true);
                        if !known {
                            if let Ok(mut sched) = self.scheduler.lock() {
                                sched.submit(InferenceRequest {
                                    id: req.id.clone(),
                                    token_ids: req.token_ids.clone(),
                                    max_new_tokens: req.max_new_tokens as u32,
                                    temperature: req.params.temperature,
                                    top_p: req.params.top_p,
                                    priority: 5,
                                });
                            }
                        }
                        self.payloads.insert(req.id.clone(), req);
                    }
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                }
            }

            // -- 2. Admit pending (requeued/preempted/direct) requests, ----
            //        gated on pool headroom in GPU mode to prevent
            //        admit/requeue spinning.  Then materialize a session
            //        for every ACTIVE sequence that has a payload but no
            //        session yet (covers both api-side acquire_slot
            //        admissions and the admit call above).
            {
                let headroom_ok = {
                    #[cfg(feature = "cutile")]
                    {
                        self.kv_pool.is_none()
                            || self.pool_free_pages()
                                >= Self::pages_for(ADMISSION_HEADROOM_TOKENS)
                            // Idle-pool floor: a pool SMALLER than the
                            // headroom constant (e.g. 700 tokens) would
                            // otherwise never admit anything -- requeued
                            // requests livelock because eviction only
                            // triggers on growth exhaustion, never on
                            // admission.  With no active sequences the
                            // only requirement is room for one fresh
                            // initial run (256 tokens + the +16-page
                            // growth margin = 32 pages).
                            || (self
                                .scheduler
                                .lock()
                                .map(|s| s.active_count() == 0)
                                .unwrap_or(false)
                                && self.pool_free_pages() >= Self::pages_for(512))
                    }
                    #[cfg(not(feature = "cutile"))]
                    {
                        true
                    }
                };
                if headroom_ok {
                    if let Ok(mut sched) = self.scheduler.lock() {
                        let _ = sched.admit_new_requests();
                    }
                }
            }
            let has_active = self
                .scheduler
                .lock()
                .map(|s| s.active_count() > 0)
                .unwrap_or(false);
            if has_active {
                self.ensure_kv_pool();
            }
            let active_ids: Vec<String> = match self.scheduler.lock() {
                Ok(sched) => sched
                    .active_sequences()
                    .iter()
                    .filter(|s| {
                        self.payloads.contains_key(&s.request_id)
                            && !sessions.contains_key(&s.request_id)
                    })
                    .map(|s| s.request_id.clone())
                    .collect(),
                Err(_) => Vec::new(),
            };
            for id in active_ids {
                let req = self.payloads.get(&id).expect("payload checked above");
                let session = InferenceSession::new(self.model.config(), TraceLevel::Off, req.seed);
                sessions.insert(
                    id.clone(),
                    ActiveSession {
                        session,
                        params: req.params.clone(),
                        generated: Vec::new(),
                        max_new_tokens: req.max_new_tokens,
                        eos_token: req.eos_token,
                        prefill_tokens: req.token_ids.clone(),
                        prefill_offset: 0,
                        prefill_done: false,
                        response_tx: req.response_tx.clone(),
                    },
                );
                #[cfg(feature = "cutile")]
                if self.kv_pool.is_some() {
                    // 256-token initial runs (32 pages incl. margin): small
                    // enough that pool pressure forces growth -> eviction
                    // (512-token initial runs already hold prompt+512
                    // decode tokens and never grow -- the p24 A/B run B
                    // showed zero swaps because of this).
                    let mut run = self
                        .model
                        .new_session_kv_cache(self.kv_pool.as_ref().unwrap(), 256);
                    if run.is_none() {
                        // Pool too tight to admit: EVICT an LRU active
                        // session (admission failure is also an eviction
                        // trigger -- otherwise a full pool of
                        // non-growing sessions livelocks requeued
                        // admissions) and retry once.
                        if self.evict_one(&[]) {
                            run = self
                                .model
                                .new_session_kv_cache(self.kv_pool.as_ref().unwrap(), 256);
                        }
                    }
                    match run {
                        Some(kv) => {
                            self.kv_caches.insert(id.clone(), kv);
                        }
                        None => {
                            // Still too tight -- requeue (the
                            // payload stays parked until a later admit).
                            eprintln!("[engine] KV pool full at admission of {id}; requeued");
                            sessions.remove(&id);
                            if let Ok(mut sched) = self.scheduler.lock() {
                                sched.requeue_active(&id);
                            }
                        }
                    }
                }
            }

            let gpu_mode = {
                #[cfg(feature = "cutile")]
                {
                    self.kv_pool.is_some()
                }
                #[cfg(not(feature = "cutile"))]
                {
                    false
                }
            };

            let mut completed: Vec<String> = Vec::new();

            if gpu_mode {
                // -- 3a. GPU multi-session path (p24-kv-swap). -------------
                #[cfg(feature = "cutile")]
                {
                    self.try_swap_ins(&sessions);

                    // Prefill chunks (per session; the batch API is
                    // decode-shaped).  One chunk per session per tick.
                    let prefill_ids: Vec<String> = match self.scheduler.lock() {
                        Ok(sched) => sched
                            .active_sequences()
                            .iter()
                            .filter(|s| {
                                sessions
                                    .get(&s.request_id)
                                    .map(|e| !e.prefill_done)
                                    .unwrap_or(false)
                            })
                            .map(|s| s.request_id.clone())
                            .collect(),
                        Err(_) => Vec::new(),
                    };
                    for id in prefill_ids {
                        if self.prefill_step_gpu(&mut sessions, &id) {
                            completed.push(id);
                        }
                    }

                    // Batched decode.
                    completed.extend(self.decode_step_gpu(&mut sessions));
                }
            } else {
                // -- 3b. CPU path: per-session stepping (unchanged). -------
                let ids: Vec<String> = sessions.keys().cloned().collect();
                for id in &ids {
                    let entry = sessions.get_mut(id).unwrap();
                    let done = entry.step(self.model.as_ref());
                    if done {
                        completed.push(id.clone());
                    }
                }
            }

            // -- 4. Retire finished requests. ------------------------------
            if debug_ticks && tick % 50 == 0 {
                #[cfg(feature = "cutile")]
                let free = self.pool_free_pages();
                #[cfg(not(feature = "cutile"))]
                let free = 0;
                #[cfg(feature = "cutile")]
                let spec = {
                    let s = &self.spec_stats;
                    if s.cycles > 0 {
                        format!(
                            " | spec: cycles={} (engine={}) mean_emitted={:.2} hist={:?} \
                             draft={:.1}ms/eng-cyc verify={:.1}ms/eng-cyc",
                            s.cycles,
                            s.engine_cycles,
                            s.mean(),
                            s.hist,
                            s.draft_us as f64 / 1000.0 / s.engine_cycles as f64,
                            s.verify_us as f64 / 1000.0 / s.engine_cycles as f64,
                        )
                    } else {
                        String::new()
                    }
                };
                #[cfg(not(feature = "cutile"))]
                let spec = String::new();
                let per_sess: Vec<String> = sessions
                    .iter()
                    .map(|(id, e)| {
                        #[cfg(feature = "cutile")]
                        let kv = self.kv_caches.get(id).map(|k| {
                            format!("kv_seq={} base={} pages={}", k.seq_len(), k.base_page(), k.run_pages())
                        }).unwrap_or_else(|| "kv=-".into());
                        #[cfg(not(feature = "cutile"))]
                        let kv = String::new();
                        format!(
                            "{}[pf={} off={} gen={} {}]",
                            &id[id.len().saturating_sub(6)..],
                            e.prefill_done,
                            e.prefill_offset,
                            e.generated.len(),
                            kv
                        )
                    })
                    .collect();
                eprintln!(
                    "[engine-tick {tick}] sess={} free_pages={}{} | {}",
                    sessions.len(),
                    free,
                    spec,
                    per_sess.join(" ")
                );
            }
            for id in &completed {
                if let Some(entry) = sessions.remove(id) {
                    let _ = entry.response_tx.try_send(EngineEvent::Done);
                }
                #[cfg(feature = "cutile")]
                {
                    self.kv_caches.remove(id); // Drop frees the run
                    self.draft_caches.remove(id); // Drop frees the draft run
                    self.eagle_states.remove(id); // Drop frees the head run
                }
                self.payloads.remove(id);
                if let Ok(mut sched) = self.scheduler.lock() {
                    sched.complete_sequence(id);
                }
            }

            // -- 5. Idle: block for the next request. ----------------------
            let idle = sessions.is_empty()
                && self
                    .scheduler
                    .lock()
                    .map(|s| {
                        s.pending_count() == 0
                            && s.preempted_count() == 0
                            && s.active_count() == 0
                    })
                    .unwrap_or(true)
                && self.payloads.is_empty();
            if idle {
                match self.rx.blocking_recv() {
                    Some(req) => {
                        let known = self
                            .scheduler
                            .lock()
                            .map(|s| s.in_system(&req.id))
                            .unwrap_or(true);
                        if !known {
                            if let Ok(mut sched) = self.scheduler.lock() {
                                sched.submit(InferenceRequest {
                                    id: req.id.clone(),
                                    token_ids: req.token_ids.clone(),
                                    max_new_tokens: req.max_new_tokens as u32,
                                    temperature: req.params.temperature,
                                    top_p: req.params.top_p,
                                    priority: 5,
                                });
                            }
                        }
                        self.payloads.insert(req.id.clone(), req);
                    }
                    None => return,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn make_task(model: Arc<dyn InferenceModel>) -> (mpsc::Sender<EngineRequest>, EngineTask) {
        let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(16);
        let scheduler = Arc::new(Mutex::new(BatchScheduler::new(4, 4096, 4)));
        let task = EngineTask::new(model, None, engine_rx, scheduler);
        (engine_tx, task)
    }

    #[tokio::test]
    async fn test_engine_processes_batch() {
        let model = make_model();
        let vocab = model.config().vocab_size;
        let (engine_tx, task) = make_task(model);

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
        let (engine_tx, task) = make_task(model.clone());

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

    #[tokio::test]
    async fn test_multi_session_scheduler_no_drops() {
        // Several concurrent requests all complete; the scheduler-driven
        // loop neither drops nor duplicates requests.
        let model = make_model();
        let (engine_tx, task) = make_task(model);

        tokio::task::spawn_blocking(move || task.run());

        let mut rxs = Vec::new();
        for i in 0..4u64 {
            let (resp_tx, resp_rx) = mpsc::channel::<EngineEvent>(64);
            engine_tx
                .send(EngineRequest {
                    id: format!("m{i}"),
                    token_ids: vec![1, 2, 3],
                    max_new_tokens: 2,
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
            rxs.push(resp_rx);
        }

        for (i, mut rx) in rxs.into_iter().enumerate() {
            let mut tokens = 0;
            while let Some(ev) = rx.recv().await {
                match ev {
                    EngineEvent::Token(_) => tokens += 1,
                    EngineEvent::Done => break,
                }
            }
            assert_eq!(tokens, 2, "session m{i} produced {tokens} tokens");
        }
    }

    #[tokio::test]
    async fn test_disconnect_drains_remaining_sessions() {
        // After the last sender drops, in-flight sessions still finish.
        let model = make_model();
        let (engine_tx, task) = make_task(model);

        let (resp_tx, mut resp_rx) = mpsc::channel::<EngineEvent>(64);
        engine_tx
            .send(EngineRequest {
                id: "drain".into(),
                token_ids: vec![1, 2],
                max_new_tokens: 2,
                params: SamplingParams {
                    temperature: 0.0,
                    ..Default::default()
                },
                eos_token: None,
                seed: 7,
                response_tx: resp_tx,
            })
            .await
            .unwrap();
        drop(engine_tx);

        tokio::task::spawn_blocking(move || task.run());

        let mut tokens = 0;
        while let Some(ev) = resp_rx.recv().await {
            match ev {
                EngineEvent::Token(_) => tokens += 1,
                EngineEvent::Done => break,
            }
        }
        assert_eq!(tokens, 2);
    }
}
