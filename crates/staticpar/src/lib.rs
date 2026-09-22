use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

struct Inner {
    gen:      AtomicU64,
    fn_ptr:   AtomicU64,
    ctx:      AtomicU64,
    n:        AtomicUsize,
    chunk:    AtomicUsize,
    work_ctr: AtomicUsize,
    done:     AtomicUsize,
    shutdown: AtomicBool,
}

/// Static-partition spin-wait thread pool.
///
/// Pre-spawned threads spin between `run()` calls (up to `spin_budget` iterations
/// before yielding). Work is divided into equal partitions — thread i processes
/// `[i*chunk .. (i+1)*chunk)`. No work-stealing, no crossbeam epoch GC.
///
/// Designed for repeated short embarrassingly-parallel workloads where thread
/// count equals core count (GEMV, matrix multiplication over output neurons).
pub struct StaticPool {
    inner:       Arc<Inner>,
    n_workers:   usize,
    _threads:    Vec<JoinHandle<()>>,
}

impl StaticPool {
    pub fn new(n_workers: usize, spin_budget: u64) -> Self {
        let inner = Arc::new(Inner {
            gen:      AtomicU64::new(0),
            fn_ptr:   AtomicU64::new(0),
            ctx:      AtomicU64::new(0),
            n:        AtomicUsize::new(0),
            chunk:    AtomicUsize::new(1),
            work_ctr: AtomicUsize::new(0),
            done:     AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        });

        let mut threads = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let ic = Arc::clone(&inner);
            threads.push(
                std::thread::Builder::new()
                    .name("staticpar-worker".into())
                    .spawn(move || worker_loop(ic, spin_budget))
                    .expect("failed to spawn staticpar worker"),
            );
        }

        Self { inner, n_workers, _threads: threads }
    }

    pub fn n_threads(&self) -> usize {
        self.n_workers
    }

    /// Run `f(start, end)` for each partition of `0..n`.
    /// The calling thread participates. Blocks until all threads complete.
    ///
    /// `f` must be `Sync` — it is called concurrently from multiple threads
    /// with non-overlapping `[start, end)` ranges. The pool guarantees all
    /// threads complete before this function returns.
    pub fn run<F: Fn(usize, usize) + Sync>(&self, n: usize, f: &F) {
        if n == 0 { return; }

        unsafe fn trampoline<F: Fn(usize, usize)>(
            start: usize, end: usize, ctx: *const (),
        ) {
            (*{ ctx as *const F })(start, end)
        }

        let total_threads = self.n_workers + 1; // +1 for calling thread
        let chunk = ((n + total_threads - 1) / total_threads).max(1);

        self.inner.fn_ptr.store(trampoline::<F> as usize as u64, Ordering::Relaxed);
        self.inner.ctx.store(f as *const F as usize as u64, Ordering::Relaxed);
        self.inner.n.store(n, Ordering::Relaxed);
        self.inner.chunk.store(chunk, Ordering::Relaxed);
        self.inner.work_ctr.store(0, Ordering::Relaxed);
        self.inner.done.store(0, Ordering::Release);

        self.inner.gen.fetch_add(1, Ordering::Release);

        // Calling thread also claims work
        loop {
            let start = self.inner.work_ctr.fetch_add(chunk, Ordering::Relaxed);
            if start >= n { break; }
            let end = (start + chunk).min(n);
            (f)(start, end);
        }

        // Spin-wait for all workers
        while self.inner.done.load(Ordering::Acquire) < self.n_workers {
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

fn worker_loop(inner: Arc<Inner>, spin_budget: u64) {
    let mut last_gen = 0u64;
    loop {
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

        let n       = inner.n.load(Ordering::Acquire);
        let chunk   = inner.chunk.load(Ordering::Relaxed);
        let fn_raw  = inner.fn_ptr.load(Ordering::Acquire);
        let ctx_raw = inner.ctx.load(Ordering::Acquire);

        if fn_raw != 0 {
            type Fn = unsafe fn(usize, usize, *const ());
            let f: Fn = unsafe { std::mem::transmute(fn_raw as usize) };
            let ctx = ctx_raw as *const ();
            loop {
                let start = inner.work_ctr.fetch_add(chunk, Ordering::Relaxed);
                if start >= n { break; }
                let end = (start + chunk).min(n);
                unsafe { f(start, end, ctx) };
            }
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
        // verify each element set to its index
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
}
