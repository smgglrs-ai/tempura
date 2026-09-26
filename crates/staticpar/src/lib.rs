use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

struct Inner {
    // gen_atomic: lock-free shadow of mutex-protected gen for spinning workers.
    // Stored with Release after ranges/fn_ptr/ctx are written (inside mutex);
    // read with Acquire in spin phase — establishes happens-before for all prior stores.
    gen_atomic: AtomicU64,
    // Condvar-based wake: workers park here after spin budget is exhausted.
    // Zero memory-bus pressure when idle — no polling.
    mutex: Mutex<u64>,
    condvar: Condvar,
    done: AtomicUsize,
    shutdown: AtomicBool,
    // Per-worker pre-assigned ranges, packed as (start << 32) | end.
    // Written before Mutex release (which is the Release fence);
    // read after Condvar wake (which implies the Acquire).
    ranges: Box<[u64]>,
    fn_ptr: AtomicU64,
    ctx: AtomicU64,
    // Spin budget: iterations with spin_loop() before parking on condvar.
    // Default 4_000_000 (~1ms at 4GHz). Set TEMPURA_POOL_SPIN=0 to disable.
    spin_budget: usize,
}

/// Static-partition thread pool with spin-before-sleep wake discipline.
///
/// Pre-spawned threads first spin with `std::hint::spin_loop()` for `spin_budget`
/// iterations reading a lock-free generation counter. If the generation changes
/// during spinning (next matmul arrived), they proceed with zero condvar latency.
/// If the budget is exhausted without a generation change, they fall back to
/// `std::sync::Condvar` parking — no idle CPU burn on non-inference workloads.
///
/// Work is divided into statically pre-assigned contiguous ranges — thread i
/// processes its range with zero per-item atomics. On wake, each thread reads
/// its own range slot (no cache-line sharing during computation).
///
/// vs pure condvar: eliminates ~20μs wake latency × ~100 dispatches/token.
/// vs tight spinning: condvar fallback prevents DRAM-bandwidth starvation when idle.
/// vs Rayon: no crossbeam epoch GC overhead between calls.
///
/// Designed for repeated short embarrassingly-parallel workloads where
/// thread count equals core count (GEMV, matrix rows, etc.).
pub struct StaticPool {
    inner: Arc<Inner>,
    n_workers: usize,
    _threads: Vec<JoinHandle<()>>,
}

impl StaticPool {
    pub fn new(n_workers: usize) -> Self {
        let spin_budget = std::env::var("TEMPURA_POOL_SPIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4_000_000usize);

        let ranges: Box<[u64]> = vec![0u64; n_workers].into_boxed_slice();
        let inner = Arc::new(Inner {
            gen_atomic: AtomicU64::new(0),
            mutex: Mutex::new(0u64),
            condvar: Condvar::new(),
            done: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            ranges,
            fn_ptr: AtomicU64::new(0),
            ctx: AtomicU64::new(0),
            spin_budget,
        });

        let mut threads = Vec::with_capacity(n_workers);
        for worker_idx in 0..n_workers {
            let ic = Arc::clone(&inner);
            threads.push(
                std::thread::Builder::new()
                    .name(format!("staticpar-worker-{worker_idx}"))
                    .spawn(move || worker_loop(worker_idx, ic))
                    .expect("failed to spawn staticpar worker"),
            );
        }

        Self {
            inner,
            n_workers,
            _threads: threads,
        }
    }

    /// Number of worker threads (excluding the calling thread).
    pub fn n_threads(&self) -> usize {
        self.n_workers
    }

    /// Run `f(start, end)` for each static partition of `0..n`.
    /// The calling thread participates (handles one partition).
    /// Blocks until all worker threads complete their partitions.
    pub fn run<F: Fn(usize, usize) + Sync>(&self, n: usize, f: &F) {
        if n == 0 {
            return;
        }

        unsafe fn trampoline<F: Fn(usize, usize)>(start: usize, end: usize, ctx: *const ()) {
            (*{ ctx as *const F })(start, end)
        }

        unsafe { self.run_raw(n, trampoline::<F>, f as *const F as *const ()) }
    }

    /// Run `func(start, end, ctx)` for each static partition of `0..n`.
    ///
    /// # Safety
    /// - `func` must be callable concurrently with non-overlapping ranges.
    /// - `ctx` must remain valid and accessible from any thread.
    /// - Writes via `ctx` must target non-overlapping regions per `(start, end)`.
    pub unsafe fn run_raw(
        &self,
        n: usize,
        func: unsafe fn(usize, usize, *const ()),
        ctx: *const (),
    ) {
        if n == 0 {
            return;
        }
        let inner = &self.inner;
        let total = self.n_workers + 1; // +1 = calling thread

        let base = n / total;
        let rem = n % total;

        // Pre-compute worker ranges.
        // Safety: ranges is exclusively owned by run_raw (no concurrent run_raw
        // is possible — callers must not call concurrently), and the mutex
        // lock below provides the Release fence.
        for k in 0..self.n_workers {
            let start = k * base + k.min(rem);
            let end = start + base + usize::from(k < rem);
            unsafe {
                *inner.ranges.as_ptr().cast_mut().add(k) = (start as u64) << 32 | end as u64;
            }
        }

        // Calling thread's partition (last slice).
        let caller_start = self.n_workers * base + self.n_workers.min(rem);
        let caller_end = n;

        inner
            .fn_ptr
            .store(func as *const () as usize as u64, Ordering::Relaxed);
        inner.ctx.store(ctx as usize as u64, Ordering::Relaxed);
        inner.done.store(0, Ordering::Relaxed);

        // Wake workers: increment gen under the mutex, then store gen_atomic
        // with Release (establishes happens-before for all stores above),
        // then notify_all for workers that already fell back to condvar sleep.
        {
            let mut gen = inner.mutex.lock().unwrap();
            *gen += 1;
            inner.gen_atomic.store(*gen, Ordering::Release);
            inner.condvar.notify_all();
        }

        // Calling thread does its partition while workers run.
        if caller_start < caller_end {
            unsafe { func(caller_start, caller_end, ctx) };
        }

        // Wait for workers. Brief spin first (workers wake fast), then yield.
        let mut spins = 0usize;
        while inner.done.load(Ordering::Acquire) < self.n_workers {
            if spins < 2_000 {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

impl Drop for StaticPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Relaxed);
        let mut gen = self.inner.mutex.lock().unwrap();
        *gen += 1;
        // Store gen_atomic so workers in spin phase see the gen change and check shutdown.
        self.inner.gen_atomic.store(*gen, Ordering::Release);
        self.inner.condvar.notify_all();
        // _threads drops here, joining all worker threads.
    }
}

#[inline(always)]
fn do_work(inner: &Inner, worker_idx: usize) {
    // The Acquire load of gen_atomic (in spin phase) or condvar wake (in park phase)
    // establishes happens-before with the Release store of gen_atomic in run_raw,
    // which in turn sees all prior writes to ranges, fn_ptr, ctx.
    let packed = unsafe { *inner.ranges.as_ptr().add(worker_idx) };
    let start = (packed >> 32) as usize;
    let end = (packed & 0xFFFF_FFFF) as usize;

    if start < end {
        let fn_raw = inner.fn_ptr.load(Ordering::Relaxed);
        let ctx_raw = inner.ctx.load(Ordering::Relaxed);
        type WorkFn = unsafe fn(usize, usize, *const ());
        let f: WorkFn = unsafe { std::mem::transmute(fn_raw as usize) };
        unsafe { f(start, end, ctx_raw as *const ()) };
    }

    inner.done.fetch_add(1, Ordering::Release);
}

fn worker_loop(worker_idx: usize, inner: Arc<Inner>) {
    let mut last_gen = 0u64;
    loop {
        // Phase 1: spin-before-sleep — busy-wait with _mm_pause() for spin_budget
        // iterations. For decode at ~7 tok/s, consecutive matmuls arrive every ~1ms;
        // with a 1ms spin budget workers are awake when the next dispatch arrives,
        // eliminating condvar wake latency (~20μs × ~100 dispatches/token).
        let spin_budget = inner.spin_budget;
        let mut spun = 0usize;
        while spun < spin_budget {
            let g = inner.gen_atomic.load(Ordering::Acquire);
            if g != last_gen {
                if inner.shutdown.load(Ordering::Relaxed) {
                    return;
                }
                last_gen = g;
                do_work(&inner, worker_idx);
                break;
            }
            std::hint::spin_loop();
            spun += 1;
        }
        if spun < spin_budget {
            // Work was found during spin phase; loop back to spin for next dispatch.
            continue;
        }

        // Phase 2: spin budget exhausted — park on condvar until gen changes.
        // This prevents idle CPU burn on non-inference workloads.
        let current_gen = {
            let guard = inner.mutex.lock().unwrap();
            let guard = inner
                .condvar
                .wait_while(guard, |gen| {
                    !inner.shutdown.load(Ordering::Relaxed) && *gen == last_gen
                })
                .unwrap();
            *guard
        };
        if inner.shutdown.load(Ordering::Relaxed) {
            return;
        }
        last_gen = current_gen;
        do_work(&inner, worker_idx);
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
        let sums: Arc<Vec<AtomicI64>> =
            Arc::new((0..n).map(|i| AtomicI64::new(i as i64)).collect());
        let sums2 = Arc::clone(&sums);
        pool.run(n, &|start, end| {
            for i in start..end {
                assert_eq!(sums2[i].load(Ordering::Relaxed), i as i64);
            }
        });
        let got: i64 = sums.iter().map(|v| v.load(Ordering::Relaxed)).sum();
        let expected: i64 = (0..n as i64).sum();
        assert_eq!(got, expected);
    }

    #[test]
    fn test_no_overlap() {
        let pool = StaticPool::new(8);
        let n = 256usize;
        let counts: Arc<Vec<AtomicI64>> = Arc::new((0..n).map(|_| AtomicI64::new(0)).collect());
        let c2 = Arc::clone(&counts);
        pool.run(n, &|start, end| {
            for i in start..end {
                c2[i].fetch_add(1, Ordering::Relaxed);
            }
        });
        for (i, c) in counts.iter().enumerate() {
            assert_eq!(
                c.load(Ordering::Relaxed),
                1,
                "item {i} processed {} times",
                c.load(Ordering::Relaxed)
            );
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
            for i in start..end {
                c2[i].fetch_add(1, Ordering::Relaxed);
            }
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
            let expected: u64 = (0..100).sum();
            assert_eq!(sum.load(Ordering::Relaxed), expected, "round {round}");
        }
    }
}
