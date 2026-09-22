use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

struct Inner {
    gen:      AtomicU64,
    fn_ptr:   AtomicU64,
    ctx:      AtomicU64,
    done:     AtomicUsize,
    shutdown: AtomicBool,
    // Per-worker pre-assigned ranges, packed as (start << 32) | end.
    // Written by caller before gen.fetch_add(Release); read by worker
    // after gen.load(Acquire) — the gen fence provides the happens-before.
    // AtomicU64 used for shared access without UnsafeCell; store/load use
    // Relaxed since gen Release/Acquire is the synchronization point.
    ranges:   Box<[AtomicU64]>,
}

/// Static-partition spin-wait thread pool.
///
/// Pre-spawned threads spin between `run()` calls. Work is divided into
/// statically pre-assigned contiguous ranges — thread i processes its
/// range without any atomic counter in the hot path. Zero fetch_add
/// during computation; no crossbeam epoch GC.
///
/// Designed for repeated short embarrassingly-parallel workloads where
/// thread count equals core count (GEMV, matrix rows, etc.).
pub struct StaticPool {
    inner:     Arc<Inner>,
    n_workers: usize,
    _threads:  Vec<JoinHandle<()>>,
}

impl StaticPool {
    pub fn new(n_workers: usize, spin_budget: u64) -> Self {
        let ranges: Box<[AtomicU64]> = (0..n_workers)
            .map(|_| AtomicU64::new(0))
            .collect();
        let inner = Arc::new(Inner {
            gen:      AtomicU64::new(0),
            fn_ptr:   AtomicU64::new(0),
            ctx:      AtomicU64::new(0),
            done:     AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            ranges,
        });

        let mut threads = Vec::with_capacity(n_workers);
        for worker_idx in 0..n_workers {
            let ic = Arc::clone(&inner);
            threads.push(
                std::thread::Builder::new()
                    .name(format!("staticpar-worker-{worker_idx}"))
                    .spawn(move || worker_loop(worker_idx, ic, spin_budget))
                    .expect("failed to spawn staticpar worker"),
            );
        }

        Self { inner, n_workers, _threads: threads }
    }

    /// Number of worker threads (excluding the calling thread).
    pub fn n_threads(&self) -> usize {
        self.n_workers
    }

    /// Run `f(start, end)` for each static partition of `0..n`.
    /// The calling thread participates (handles one partition).
    /// Blocks until all worker threads complete their partitions.
    pub fn run<F: Fn(usize, usize) + Sync>(&self, n: usize, f: &F) {
        if n == 0 { return; }

        unsafe fn trampoline<F: Fn(usize, usize)>(
            start: usize, end: usize, ctx: *const (),
        ) {
            (*{ ctx as *const F })(start, end)
        }

        unsafe { self.run_raw(n, trampoline::<F>, f as *const F as *const ()) }
    }

    /// Run `func(start, end, ctx)` for each static partition of `0..n`.
    ///
    /// # Safety
    /// - `func` must be callable concurrently with non-overlapping ranges.
    /// - `ctx` must remain valid and the data must be accessible from any thread.
    /// - Writes via `ctx` must target non-overlapping regions per `(start, end)`.
    pub unsafe fn run_raw(
        &self,
        n: usize,
        func: unsafe fn(usize, usize, *const ()),
        ctx: *const (),
    ) {
        if n == 0 { return; }
        let inner = &self.inner;
        let total  = self.n_workers + 1; // +1 = calling thread

        // Static partition: ceil(n/total) items per thread.
        let base = n / total;
        let rem  = n % total;

        // Pre-compute and store each worker's range BEFORE gen increment.
        // gen.fetch_add(Release) below provides the Release fence that
        // makes these stores visible to workers after gen.load(Acquire).
        for k in 0..self.n_workers {
            let start = k * base + k.min(rem);
            let end   = start + base + usize::from(k < rem);
            inner.ranges[k].store((start as u64) << 32 | end as u64, Ordering::Relaxed);
        }

        // Calling thread's partition (last slice, index = n_workers).
        let caller_start = self.n_workers * base + self.n_workers.min(rem);
        let caller_end   = n;

        // Publish fn/ctx (Relaxed — workers read after Acquire on gen).
        inner.fn_ptr.store(func as *const () as usize as u64, Ordering::Relaxed);
        inner.ctx.store(ctx as usize as u64, Ordering::Relaxed);
        inner.done.store(0, Ordering::Relaxed);

        // Release fence: makes all stores above visible to workers.
        inner.gen.fetch_add(1, Ordering::Release);

        // Calling thread does its partition concurrently with workers.
        if caller_start < caller_end {
            unsafe { func(caller_start, caller_end, ctx) };
        }

        // Spin-wait for workers.
        while inner.done.load(Ordering::Acquire) < self.n_workers {
            std::hint::spin_loop();
        }
    }
}

impl Drop for StaticPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Relaxed);
        self.inner.gen.fetch_add(1, Ordering::Release);
    }
}

fn worker_loop(worker_idx: usize, inner: Arc<Inner>, spin_budget: u64) {
    let mut last_gen = 0u64;
    loop {
        // Spin until gen changes (new job) or shutdown.
        let mut spins = 0u64;
        loop {
            let g = inner.gen.load(Ordering::Acquire);
            if g != last_gen { last_gen = g; break; }
            if inner.shutdown.load(Ordering::Relaxed) { return; }
            if spins < spin_budget {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
                spins = 0;
            }
        }

        if inner.shutdown.load(Ordering::Relaxed) { return; }

        // Read our pre-assigned range — no atomic counter, no cache-line sharing.
        let packed = inner.ranges[worker_idx].load(Ordering::Relaxed);
        let start = (packed >> 32) as usize;
        let end   = (packed & 0xFFFF_FFFF) as usize;

        if start < end {
            let fn_raw  = inner.fn_ptr.load(Ordering::Relaxed);
            let ctx_raw = inner.ctx.load(Ordering::Relaxed);
            type Fn = unsafe fn(usize, usize, *const ());
            let f: Fn = unsafe { std::mem::transmute(fn_raw as usize) };
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
        let pool = StaticPool::new(4, 10_000);
        let n = 1024usize;
        let sums: Arc<Vec<AtomicI64>> = Arc::new((0..n).map(|i| AtomicI64::new(i as i64)).collect());
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
        let pool = StaticPool::new(8, 10_000);
        let n = 256usize;
        let counts: Arc<Vec<AtomicI64>> = Arc::new((0..n).map(|_| AtomicI64::new(0)).collect());
        let c2 = Arc::clone(&counts);
        pool.run(n, &|start, end| {
            for i in start..end {
                c2[i].fetch_add(1, Ordering::Relaxed);
            }
        });
        for (i, c) in counts.iter().enumerate() {
            assert_eq!(c.load(Ordering::Relaxed), 1,
                "item {i} processed {} times", c.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn test_empty_n() {
        let pool = StaticPool::new(4, 10_000);
        pool.run(0, &|_, _| panic!("should not be called"));
    }

    #[test]
    fn test_n_less_than_threads() {
        let pool = StaticPool::new(8, 10_000);
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
        let pool = StaticPool::new(4, 10_000);
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
