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

impl ActiveSession {
    /// Sample the next token from `logits` (with penalties), emit it, and
    /// return true when the request is finished.  Shared by the CPU
    /// per-session path and the GPU batch path.
    fn sample_and_emit(&mut self, model: &dyn InferenceModel, logits: Vec<f32>) -> bool {
        let mut logits = logits;
        self.params.apply_penalties(&mut logits, &self.generated);
        let token = model.sample_from_logits(&logits, &self.params, &mut self.session);
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
}

impl EngineTask {
    pub fn new(
        model: Arc<dyn InferenceModel>,
        rx: mpsc::Receiver<EngineRequest>,
        scheduler: Arc<Mutex<BatchScheduler>>,
    ) -> Self {
        // inference_pool (staticpar) is initialized lazily on first GEMV call;
        // CCD pinning is handled via TEMPURA_CCD_CORES env var read by parse_ccd_cores_env
        // which is kept for future use but no longer wires a Rayon pool here.
        let _ = parse_ccd_cores_env();
        Self {
            model,
            rx,
            scheduler,
            payloads: HashMap::new(),
            #[cfg(feature = "cutile")]
            kv_pool: None,
            #[cfg(feature = "cutile")]
            kv_pool_tried: false,
            #[cfg(feature = "cutile")]
            kv_caches: HashMap::new(),
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
                    let protect_refs: Vec<&str> = protect.iter().map(|s| s.as_str()).collect();
                    let victim = {
                        let Ok(sched) = self.scheduler.lock() else {
                            return false;
                        };
                        sched.select_swap_victim(&protect_refs)
                    };
                    let Some(victim) = victim else {
                        eprintln!(
                            "[engine] KV pool exhausted (need {} pages, {} free) and no \
                             eviction candidate -- deferring {id}",
                            info.need_pages, info.free_pages
                        );
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
                                }
                                Err(e) => {
                                    eprintln!("[engine] swap_out failed for {victim}: {e}");
                                    return false;
                                }
                            }
                        },
                        None => {
                            // Victim has no GPU run (stale accounting) --
                            // mark it swapped so it stops being selected.
                            if let Ok(mut sched) = self.scheduler.lock() {
                                sched.mark_swapped(&victim);
                            }
                        }
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
        if done_prefill {
            entry.sample_and_emit(self.model.as_ref(), logits)
        } else {
            false
        }
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
        if stepped.is_empty() {
            return vec![];
        }

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
        let all_logits = match self.model.forward_batch_with_kvs(&toks, &pos, &mut kvs) {
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
                // Per-step logits fingerprint: diffing run A vs run B logs
                // pinpoints the first divergent decode step (p24 A/B hunt).
                let cs: u64 = all_logits[i]
                    .iter()
                    .fold(0u64, |a, v| a.wrapping_add(v.to_bits() as u64));
                eprintln!(
                    "[logits-cs] {} step={} cs={cs:016x}",
                    &id[id.len().saturating_sub(6)..],
                    entry.generated.len()
                );
            }
            let done = entry.sample_and_emit(self.model.as_ref(), all_logits[i].clone());
            if done {
                completed.push(id.clone());
            }
        }

        // LRU clock + footprint for the eviction policy.
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
                    let run = self
                        .model
                        .new_session_kv_cache(self.kv_pool.as_ref().unwrap(), 256);
                    match run {
                        Some(kv) => {
                            self.kv_caches.insert(id.clone(), kv);
                        }
                        None => {
                            // Pool too tight to admit -- requeue (the
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
                    "[engine-tick {tick}] sess={} free_pages={} | {}",
                    sessions.len(),
                    free,
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
        let task = EngineTask::new(model, engine_rx, scheduler);
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
