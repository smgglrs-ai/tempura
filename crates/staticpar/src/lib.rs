use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

struct Inner {
    // Condvar-based wake: workers park until gen changes.
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
}

/// Static-partition thread pool with Condvar park/wake.
///
/// Pre-spawned threads park between `run()` calls via `std::sync::Condvar`.
/// Work is divided into statically pre-assigned contiguous ranges — thread i
/// processes its range with zero per-item atomics. On wake, each thread reads
/// its own range slot (no cache-line sharing during computation).
///
/// vs spinning: eliminates idle memory-bus pressure from polling a shared atomic.
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
        let ranges: Box<[u64]> = vec![0u64; n_workers].into_boxed_slice();
        let inner = Arc::new(Inner {
            mutex: Mutex::new(0u64),
            condvar: Condvar::new(),
            done: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            ranges,
            fn_ptr: AtomicU64::new(0),
            ctx: AtomicU64::new(0),
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

        // Wake workers: increment gen under the mutex, then notify_all.
        // The mutex lock provides the Release fence for the stores above.
        {
            let mut gen = inner.mutex.lock().unwrap();
            *gen += 1;
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
        self.inner.condvar.notify_all();
        // _threads drops here, joining all worker threads.
    }
}

fn worker_loop(worker_idx: usize, inner: Arc<Inner>) {
    let mut last_gen = 0u64;
    loop {
        // Park until gen changes or shutdown.
        let current_gen = {
            let guard = inner.mutex.lock().unwrap();
            // wait_while parks atomically — no busy-polling, zero memory bus pressure.
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

        // Read our pre-assigned range. The mutex unlock above (from Condvar::wait_while)
        // provides the Acquire barrier, ensuring we see the range written before notify_all.
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
