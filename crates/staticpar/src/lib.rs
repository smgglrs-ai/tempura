use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::Arc;

const CHUNK: usize = 4;

struct Inner {
    gen: AtomicU64,
    work_ctr: AtomicUsize,
    done: AtomicUsize,
    fn_ptr: AtomicU64,
    ctx: AtomicU64,
    total_n: AtomicUsize,
    n_workers: usize,
    shutdown: AtomicBool,
    spin_budget: u64,
}

unsafe impl Sync for Inner {}
unsafe impl Send for Inner {}

pub struct StaticPool {
    inner: Arc<Inner>,
    _threads: Vec<std::thread::JoinHandle<()>>,
}

impl StaticPool {
    pub fn new(n_threads: usize, spin_budget: u64) -> Self {
        let inner = Arc::new(Inner {
            gen: AtomicU64::new(0),
            work_ctr: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            fn_ptr: AtomicU64::new(0),
            ctx: AtomicU64::new(0),
            total_n: AtomicUsize::new(0),
            n_workers: n_threads,
            shutdown: AtomicBool::new(false),
            spin_budget,
        });
        let mut threads = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            let shared = inner.clone();
            threads.push(
                std::thread::Builder::new()
                    .name("staticpar-worker".into())
                    .spawn(move || worker_loop(&shared))
                    .expect("spawn"),
            );
        }
        Self {
            inner,
            _threads: threads,
        }
    }

    pub fn n_threads(&self) -> usize {
        self.inner.n_workers
    }

    pub fn run<F: Fn(usize, usize) + Sync>(&self, n: usize, f: F) {
        if n == 0 {
            return;
        }

        unsafe extern "C" fn trampoline<F: Fn(usize, usize)>(
            start: usize,
            end: usize,
            ctx: *const (),
        ) {
            (*(ctx as *const F))(start, end);
        }

        let inner = &self.inner;
        inner.fn_ptr.store(trampoline::<F> as unsafe extern "C" fn(usize, usize, *const ()) as usize as u64, Release);
        inner.ctx.store(&f as *const F as u64, Release);
        inner.total_n.store(n, Release);
        inner.work_ctr.store(0, Release);
        inner.done.store(0, Release);
        inner.gen.fetch_add(1, Release);

        loop {
            let start = inner.work_ctr.fetch_add(CHUNK, Relaxed);
            if start >= n {
                break;
            }
            let end = (start + CHUNK).min(n);
            f(start, end);
        }

        while inner.done.load(Acquire) < inner.n_workers {
            std::hint::spin_loop();
        }
    }
}

impl Drop for StaticPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Release);
        self.inner.gen.fetch_add(1, Release);
    }
}

fn worker_loop(inner: &Inner) {
    let mut last_gen = 0u64;
    loop {
        let mut spin = 0u64;
        loop {
            if inner.shutdown.load(Relaxed) {
                return;
            }
            let g = inner.gen.load(Acquire);
            if g != last_gen {
                last_gen = g;
                break;
            }
            if spin < inner.spin_budget {
                spin += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
                spin = 0;
            }
        }
        if inner.shutdown.load(Relaxed) {
            return;
        }
        let n = inner.total_n.load(Acquire);
        loop {
            let start = inner.work_ctr.fetch_add(CHUNK, Relaxed);
            if start >= n {
                break;
            }
            let end = (start + CHUNK).min(n);
            // Safety: fn_ptr and ctx are written with Release before gen is incremented,
            // and f's lifetime is bounded by StaticPool::run's stack frame — run
            // spin-waits until all workers finish before returning.
            let func: unsafe extern "C" fn(usize, usize, *const ()) =
                unsafe { std::mem::transmute(inner.fn_ptr.load(Acquire) as usize) };
            let ctx = inner.ctx.load(Acquire) as *const ();
            unsafe { func(start, end, ctx) };
        }
        inner.done.fetch_add(1, Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering::*};

    #[test]
    fn test_sum_parallel() {
        let pool = StaticPool::new(4, 10_000);
        let sum = AtomicI64::new(0);
        pool.run(1000, |start, end| {
            let local: i64 = (start..end).map(|i| i as i64).sum();
            sum.fetch_add(local, Relaxed);
        });
        assert_eq!(sum.load(Relaxed), (0..1000i64).sum::<i64>());
    }

    #[test]
    fn test_multiple_calls() {
        let pool = StaticPool::new(8, 10_000);
        for _ in 0..100 {
            let sum = AtomicI64::new(0);
            pool.run(256, |start, end| {
                let v: i64 = (start..end).map(|i| i as i64).sum();
                sum.fetch_add(v, Relaxed);
            });
            assert_eq!(sum.load(Relaxed), (0..256i64).sum::<i64>());
        }
    }

    #[test]
    fn test_zero_items() {
        let pool = StaticPool::new(4, 1000);
        pool.run(0, |_, _| panic!("should not be called"));
    }

    #[test]
    fn test_fewer_items_than_threads() {
        let pool = StaticPool::new(8, 1000);
        let count = AtomicUsize::new(0);
        pool.run(3, |start, end| {
            count.fetch_add(end - start, Relaxed);
        });
        assert_eq!(count.load(Relaxed), 3);
    }

    #[test]
    fn test_single_thread() {
        let pool = StaticPool::new(1, 1000);
        let sum = AtomicI64::new(0);
        pool.run(64, |start, end| {
            sum.fetch_add((start..end).map(|i| i as i64).sum::<i64>(), Relaxed);
        });
        assert_eq!(sum.load(Relaxed), (0..64i64).sum::<i64>());
    }

    #[test]
    fn test_correctness_large() {
        let pool = StaticPool::new(16, 100_000);
        let sum = AtomicI64::new(0);
        pool.run(4096, |start, end| {
            sum.fetch_add((start..end).map(|i| i as i64).sum::<i64>(), Relaxed);
        });
        assert_eq!(sum.load(Relaxed), (0..4096i64).sum::<i64>());
    }
}
