use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

struct Inner {
    // Condvar-based wake for both direct and session modes.
    mutex:    Mutex<u64>,
    condvar:  Condvar,
    // Session mode: workers spin between dispatches within a session.
    session_active: AtomicBool,  // true = workers enter spinning phase
    dispatch_gen:   AtomicU64,   // incremented per dispatch within session
    session_end:    AtomicBool,  // signal workers to exit session spin
    awake_count:    AtomicUsize, // workers currently in session spin
    // Per-dispatch completion.
    done:     AtomicUsize,
    shutdown: AtomicBool,
    // Per-worker pre-assigned ranges.
    ranges: Box<[u64]>,
    fn_ptr: AtomicU64,
    ctx:    AtomicU64,
}

/// Static-partition thread pool with Condvar park/wake.
///
/// Supports two dispatch modes:
/// - `run()`: one condvar wake + immediate work per call (simple, ~10μs overhead)
/// - `start_session()`: one condvar wake, then spinning barriers between dispatches
///   (matches llama.cpp ggml_threadpool: ~0.1μs per dispatch within session)
pub struct StaticPool {
    inner:     Arc<Inner>,
    n_workers: usize,
    _threads:  Vec<JoinHandle<()>>,
}

/// A decode session. Workers wake once at creation and spin between dispatches.
/// Drop returns workers to condvar sleep.
pub struct Session<'a> {
    inner:        &'a Inner,
    n_workers:    usize,
    dispatch_gen: u64,
}

impl StaticPool {
    pub fn new(n_workers: usize) -> Self {
        let ranges: Box<[u64]> = vec![0u64; n_workers].into_boxed_slice();
        let inner = Arc::new(Inner {
            mutex:          Mutex::new(0u64),
            condvar:        Condvar::new(),
            session_active: AtomicBool::new(false),
            dispatch_gen:   AtomicU64::new(0),
            session_end:    AtomicBool::new(false),
            awake_count:    AtomicUsize::new(0),
            done:           AtomicUsize::new(0),
            shutdown:       AtomicBool::new(false),
            ranges,
            fn_ptr:         AtomicU64::new(0),
            ctx:            AtomicU64::new(0),
        });
        let mut threads = Vec::with_capacity(n_workers);
        for k in 0..n_workers {
            let ic = Arc::clone(&inner);
            threads.push(
                std::thread::Builder::new()
                    .name(format!("staticpar-worker-{k}"))
                    .spawn(move || worker_loop(k, ic))
                    .expect("failed to spawn staticpar worker"),
            );
        }
        Self { inner, n_workers, _threads: threads }
    }

    pub fn n_threads(&self) -> usize { self.n_workers }

    /// Run `f(start, end)` for each static partition of `0..n`.
    /// Calling thread participates. One condvar wake per call.
    pub fn run<F: Fn(usize, usize) + Sync>(&self, n: usize, f: &F) {
        if n == 0 { return; }
        unsafe fn trampoline<F: Fn(usize, usize)>(s: usize, e: usize, ctx: *const ()) {
            (*{ ctx as *const F })(s, e)
        }
        unsafe { self.run_raw(n, trampoline::<F>, f as *const F as *const ()) }
    }

    /// Raw dispatch — condvar-based direct mode.
    ///
    /// # Safety
    /// `func` callable concurrently with non-overlapping (start, end).
    /// `ctx` valid from any thread for the duration of the call.
    pub unsafe fn run_raw(
        &self,
        n: usize,
        func: unsafe fn(usize, usize, *const ()),
        ctx: *const (),
    ) {
        if n == 0 { return; }
        let inner = &self.inner;
        let total = self.n_workers + 1;
        let base = n / total;
        let rem  = n % total;
        for k in 0..self.n_workers {
            let start = k * base + k.min(rem);
            let end   = start + base + usize::from(k < rem);
            unsafe { *inner.ranges.as_ptr().cast_mut().add(k) = (start as u64) << 32 | end as u64; }
        }
        let caller_start = self.n_workers * base + self.n_workers.min(rem);

        inner.fn_ptr.store(func as *const () as usize as u64, Ordering::Relaxed);
        inner.ctx.store(ctx as usize as u64, Ordering::Relaxed);
        inner.done.store(0, Ordering::Relaxed);
        // session_active = false → workers will use direct mode.
        inner.session_active.store(false, Ordering::Release);
        {
            let mut gen = inner.mutex.lock().unwrap();
            *gen += 1;
            inner.condvar.notify_all();
        }
        if caller_start < n { unsafe { func(caller_start, n, ctx) }; }
        let mut spins = 0usize;
        while inner.done.load(Ordering::Acquire) < self.n_workers {
            if spins < 2_000 { spins += 1; std::hint::spin_loop(); }
            else { std::thread::yield_now(); }
        }
    }

    /// Wake workers once for a session (one condvar notify_all).
    /// All subsequent dispatches via `session.run_raw()` use spinning barriers.
    pub fn start_session(&self) -> Session<'_> {
        let inner = &self.inner;
        inner.session_end.store(false, Ordering::Release);
        let dgen = inner.dispatch_gen.load(Ordering::Relaxed);
        // session_active = true → workers enter spin phase after waking.
        inner.session_active.store(true, Ordering::Release);
        {
            let mut gen = inner.mutex.lock().unwrap();
            *gen += 1;
            inner.condvar.notify_all();
        }
        // Wait until all workers are awake and spinning.
        while inner.awake_count.load(Ordering::Acquire) < self.n_workers {
            std::hint::spin_loop();
        }
        Session { inner, n_workers: self.n_workers, dispatch_gen: dgen }
    }
}

impl<'a> Session<'a> {
    /// Dispatch via spinning barrier — no condvar, workers already awake.
    ///
    /// # Safety
    /// Same as `StaticPool::run_raw`.
    pub unsafe fn run_raw(
        &mut self,
        n: usize,
        func: unsafe fn(usize, usize, *const ()),
        ctx: *const (),
    ) {
        if n == 0 { return; }
        let inner = self.inner;
        let total = self.n_workers + 1;
        let base = n / total;
        let rem  = n % total;
        for k in 0..self.n_workers {
            let start = k * base + k.min(rem);
            let end   = start + base + usize::from(k < rem);
            unsafe { *inner.ranges.as_ptr().cast_mut().add(k) = (start as u64) << 32 | end as u64; }
        }
        let caller_start = self.n_workers * base + self.n_workers.min(rem);
        inner.fn_ptr.store(func as *const () as usize as u64, Ordering::Relaxed);
        inner.ctx.store(ctx as usize as u64, Ordering::Relaxed);
        inner.done.store(0, Ordering::Release);
        // Spinning barrier: workers see this immediately.
        self.dispatch_gen += 1;
        inner.dispatch_gen.store(self.dispatch_gen, Ordering::Release);
        if caller_start < n { unsafe { func(caller_start, n, ctx) }; }
        let mut spins = 0usize;
        while inner.done.load(Ordering::Acquire) < self.n_workers {
            if spins < 4_000 { spins += 1; std::hint::spin_loop(); }
            else { std::thread::yield_now(); }
        }
    }

    pub fn run<F: Fn(usize, usize) + Sync>(&mut self, n: usize, f: F) {
        unsafe fn trampoline<F: Fn(usize, usize)>(s: usize, e: usize, c: *const ()) {
            (*{ c as *const F })(s, e)
        }
        unsafe { self.run_raw(n, trampoline::<F>, &f as *const F as *const ()) }
    }
}

impl<'a> Drop for Session<'a> {
    fn drop(&mut self) {
        let inner = self.inner;
        inner.session_end.store(true, Ordering::Release);
        inner.dispatch_gen.fetch_add(1, Ordering::Release);
        while inner.awake_count.load(Ordering::Acquire) > 0 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for StaticPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Relaxed);
        // Also bump dispatch_gen in case workers are in session-spin phase.
        self.inner.session_end.store(true, Ordering::Release);
        self.inner.dispatch_gen.fetch_add(1, Ordering::Release);
        let mut gen = self.inner.mutex.lock().unwrap();
        *gen += 1;
        self.inner.condvar.notify_all();
    }
}

fn worker_loop(worker_idx: usize, inner: Arc<Inner>) {
    let mut last_gen = 0u64;
    loop {
        // ── PHASE 1: Park until woken ──────────────────────────────────────
        let current_gen = {
            let guard = inner.mutex.lock().unwrap();
            let guard = inner.condvar
                .wait_while(guard, |gen| {
                    !inner.shutdown.load(Ordering::Relaxed) && *gen == last_gen
                })
                .unwrap();
            *guard
        };
        if inner.shutdown.load(Ordering::Relaxed) { return; }
        last_gen = current_gen;

        if inner.session_active.load(Ordering::Acquire) {
            // ── SESSION MODE: spin between dispatches ──────────────────────
            let mut last_dgen = inner.dispatch_gen.load(Ordering::Acquire);
            inner.awake_count.fetch_add(1, Ordering::Release);

            loop {
                if inner.session_end.load(Ordering::Relaxed) { break; }
                if inner.shutdown.load(Ordering::Relaxed) {
                    inner.awake_count.fetch_sub(1, Ordering::Release);
                    return;
                }
                let dgen = inner.dispatch_gen.load(Ordering::Acquire);
                if dgen != last_dgen {
                    last_dgen = dgen;
                    if inner.session_end.load(Ordering::Relaxed) { break; }
                    let packed = unsafe { *inner.ranges.as_ptr().add(worker_idx) };
                    let start = (packed >> 32) as usize;
                    let end   = (packed & 0xFFFF_FFFF) as usize;
                    if start < end {
                        type Fn = unsafe fn(usize, usize, *const ());
                        let f: Fn = unsafe { std::mem::transmute(inner.fn_ptr.load(Ordering::Relaxed) as usize) };
                        unsafe { f(start, end, inner.ctx.load(Ordering::Relaxed) as *const ()) };
                    }
                    inner.done.fetch_add(1, Ordering::Release);
                } else {
                    std::hint::spin_loop();
                }
            }
            inner.awake_count.fetch_sub(1, Ordering::Release);
        } else {
            // ── DIRECT MODE: do work immediately ──────────────────────────
            let packed = unsafe { *inner.ranges.as_ptr().add(worker_idx) };
            let start = (packed >> 32) as usize;
            let end   = (packed & 0xFFFF_FFFF) as usize;
            if start < end {
                type Fn = unsafe fn(usize, usize, *const ());
                let f: Fn = unsafe { std::mem::transmute(inner.fn_ptr.load(Ordering::Relaxed) as usize) };
                unsafe { f(start, end, inner.ctx.load(Ordering::Relaxed) as *const ()) };
            }
            inner.done.fetch_add(1, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI64;
    use std::sync::Arc;

    #[test]
    fn test_sum_is_correct() {
        let pool = StaticPool::new(4);
        let n = 1024usize;
        let sums: Arc<Vec<AtomicI64>> = Arc::new((0..n).map(|i| AtomicI64::new(i as i64)).collect());
        let s2 = Arc::clone(&sums);
        pool.run(n, &|start, end| {
            for i in start..end { assert_eq!(s2[i].load(Ordering::Relaxed), i as i64); }
        });
        let got: i64 = sums.iter().map(|v| v.load(Ordering::Relaxed)).sum();
        assert_eq!(got, (0..n as i64).sum::<i64>());
    }

    #[test]
    fn test_no_overlap() {
        let pool = StaticPool::new(8);
        let n = 256usize;
        let counts: Arc<Vec<AtomicI64>> = Arc::new((0..n).map(|_| AtomicI64::new(0)).collect());
        let c2 = Arc::clone(&counts);
        pool.run(n, &|start, end| {
            for i in start..end { c2[i].fetch_add(1, Ordering::Relaxed); }
        });
        for (i, c) in counts.iter().enumerate() {
            assert_eq!(c.load(Ordering::Relaxed), 1, "item {i} wrong count");
        }
    }

    #[test]
    fn test_empty_n() {
        let pool = StaticPool::new(4);
        pool.run(0, &|_, _| panic!("should not be called"));
    }

    #[test]
    fn test_n_less_than_threads() {
        let pool = StaticPool::new(8);
        let n = 3usize;
        let counts: Arc<Vec<AtomicI64>> = Arc::new((0..n).map(|_| AtomicI64::new(0)).collect());
        let c2 = Arc::clone(&counts);
        pool.run(n, &|start, end| {
            for i in start..end { c2[i].fetch_add(1, Ordering::Relaxed); }
        });
        assert!(counts.iter().all(|v| v.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn test_repeated_calls() {
        let pool = StaticPool::new(4);
        for round in 0..50u64 {
            let sum = std::sync::atomic::AtomicU64::new(0);
            pool.run(100, &|start, end| {
                let local: u64 = (start..end).map(|i| i as u64).sum();
                sum.fetch_add(local, Ordering::Relaxed);
            });
            assert_eq!(sum.load(Ordering::Relaxed), (0..100u64).sum::<u64>(), "round {round}");
        }
    }

    #[test]
    fn test_session_sum_correct() {
        let pool = StaticPool::new(4);
        let n = 512usize;
        let counts: Arc<Vec<AtomicI64>> = Arc::new((0..n).map(|_| AtomicI64::new(0)).collect());
        {
            let mut session = pool.start_session();
            for _ in 0..20 {
                let c2 = Arc::clone(&counts);
                session.run(n, |start, end| {
                    for i in start..end { c2[i].fetch_add(1, Ordering::Relaxed); }
                });
            }
        }
        for (i, c) in counts.iter().enumerate() {
            assert_eq!(c.load(Ordering::Relaxed), 20, "item {i}");
        }
    }

    #[test]
    fn test_session_then_run() {
        let pool = StaticPool::new(4);
        {
            let mut s = pool.start_session();
            let sum = AtomicI64::new(0);
            s.run(64, |start, end| {
                sum.fetch_add((start..end).map(|i| i as i64).sum::<i64>(), Ordering::Relaxed);
            });
        }
        let sum2 = std::sync::atomic::AtomicU64::new(0);
        pool.run(64, &|start, end| {
            sum2.fetch_add((start..end).map(|i| i as u64).sum::<u64>(), Ordering::Relaxed);
        });
        assert_eq!(sum2.load(Ordering::Relaxed), (0..64u64).sum::<u64>());
    }
}
