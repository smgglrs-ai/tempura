//! Persistent thread pool — translated from ggml's threading model.
//! Workers sleep via Condvar when idle (zero CPU). Per-worker generation tracking
//! prevents the stale-signal race that caused deadlocks in spin-wait variants.

use std::sync::atomic::{AtomicUsize, Ordering, fence};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

// ── SpinBarrier (translated from ggml_barrier) ─────────────────────────────

pub struct SpinBarrier {
    n_threads: i32,
    n_barrier: std::sync::atomic::AtomicI32,
    n_barrier_passed: std::sync::atomic::AtomicI32,
}

impl SpinBarrier {
    pub fn new(n_threads: usize) -> Self {
        Self {
            n_threads: n_threads as i32,
            n_barrier: std::sync::atomic::AtomicI32::new(0),
            n_barrier_passed: std::sync::atomic::AtomicI32::new(0),
        }
    }

    #[inline]
    pub fn wait(&self) {
        if self.n_threads <= 1 { return; }
        let n_passed = self.n_barrier_passed.load(Ordering::Relaxed);
        let n = self.n_barrier.fetch_add(1, Ordering::SeqCst);
        if n == self.n_threads - 1 {
            self.n_barrier.store(0, Ordering::Relaxed);
            self.n_barrier_passed.fetch_add(1, Ordering::SeqCst);
            return;
        }
        while self.n_barrier_passed.load(Ordering::Relaxed) == n_passed {
            std::hint::spin_loop();
        }
        fence(Ordering::SeqCst);
    }
}

// ── PersistentPool ─────────────────────────────────────────────────────────

#[repr(align(64))]
struct CacheLine(AtomicUsize);

struct Shared {
    gen: Mutex<usize>,
    gen_atomic: AtomicUsize,
    wake: Condvar,
    chunk_counter: AtomicUsize,
    n_chunks: AtomicUsize,
    chunk_size: AtomicUsize,
    total: AtomicUsize,
    stop: std::sync::atomic::AtomicBool,
    n_workers: usize,
    work_ptr: AtomicUsize,
    work_vtable: AtomicUsize,
    worker_done_gen: Vec<CacheLine>,
}

pub struct PersistentPool {
    shared: Arc<Shared>,
    _threads: Vec<JoinHandle<()>>,
}

impl PersistentPool {
    pub fn new(n_threads: usize) -> Self {
        let n_workers = n_threads.saturating_sub(1);
        let mut wdg = Vec::with_capacity(n_workers);
        for _ in 0..n_workers { wdg.push(CacheLine(AtomicUsize::new(0))); }

        let shared = Arc::new(Shared {
            gen: Mutex::new(0),
            gen_atomic: AtomicUsize::new(0),
            wake: Condvar::new(),
            chunk_counter: AtomicUsize::new(0),
            n_chunks: AtomicUsize::new(0),
            chunk_size: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
            stop: std::sync::atomic::AtomicBool::new(false),
            n_workers,
            work_ptr: AtomicUsize::new(0),
            work_vtable: AtomicUsize::new(0),
            worker_done_gen: wdg,
        });

        let mut threads = Vec::with_capacity(n_workers);
        for i in 0..n_workers {
            let sh = Arc::clone(&shared);
            threads.push(thread::Builder::new()
                .name(format!("pool-{i}"))
                .spawn(move || worker_loop(&sh, i))
                .unwrap());
        }

        Self { shared, _threads: threads }
    }

    #[inline]
    pub fn thread_count(&self) -> usize {
        self.shared.n_workers + 1
    }

    /// Dispatch an indexed computation: each of N threads calls f(ith, nth).
    /// Uses the persistent pool — no thread spawn/join overhead.
    #[inline]
    pub fn run_indexed<F: Fn(usize, usize) + Sync>(&self, f: &F) {
        let nth = self.shared.n_workers + 1;
        if nth <= 1 {
            f(0, 1);
            return;
        }
        self.parallel_chunks(nth, 1, &|start, _end| {
            f(start, nth);
        });
    }

    #[inline]
    pub fn parallel_chunks<F: Fn(usize, usize) + Sync>(&self, total: usize, chunk_size: usize, f: &F) {
        let n_chunks = (total + chunk_size - 1) / chunk_size;
        if n_chunks <= 1 || self.shared.n_workers == 0 {
            f(0, total);
            return;
        }

        let trait_ref: &dyn Fn(usize, usize) = f;
        let raw: [usize; 2] = unsafe { std::mem::transmute(trait_ref) };

        let sh = &self.shared;
        sh.work_ptr.store(raw[0], Ordering::Relaxed);
        sh.work_vtable.store(raw[1], Ordering::Relaxed);
        sh.total.store(total, Ordering::Relaxed);
        sh.chunk_size.store(chunk_size, Ordering::Relaxed);
        sh.n_chunks.store(n_chunks, Ordering::Relaxed);
        sh.chunk_counter.store(0, Ordering::Relaxed);

        let gen = sh.gen_atomic.fetch_add(1, Ordering::AcqRel) + 1;
        { let mut g = sh.gen.lock().unwrap(); *g = gen; }
        sh.wake.notify_all();

        steal_chunks(sh);

        // Wait for all workers to finish THIS generation
        'wait: loop {
            for i in 0..sh.n_workers {
                if sh.worker_done_gen[i].0.load(Ordering::Acquire) < gen {
                    std::hint::spin_loop();
                    continue 'wait;
                }
            }
            break;
        }
    }
}

impl Drop for PersistentPool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.gen_atomic.fetch_add(1, Ordering::Release);
        { let mut g = self.shared.gen.lock().unwrap(); *g += 1; }
        self.shared.wake.notify_all();
        for t in self._threads.drain(..) { let _ = t.join(); }
    }
}

#[inline]
fn steal_chunks(sh: &Shared) {
    let n_chunks = sh.n_chunks.load(Ordering::Relaxed);
    let chunk_size = sh.chunk_size.load(Ordering::Relaxed);
    let total = sh.total.load(Ordering::Relaxed);
    let data = sh.work_ptr.load(Ordering::Relaxed);
    let vtable = sh.work_vtable.load(Ordering::Relaxed);
    let f: &dyn Fn(usize, usize) = unsafe { std::mem::transmute([data, vtable]) };
    loop {
        let c = sh.chunk_counter.fetch_add(1, Ordering::Relaxed);
        if c >= n_chunks { break; }
        let start = c * chunk_size;
        let end = (start + chunk_size).min(total);
        f(start, end);
    }
}

fn worker_loop(sh: &Shared, id: usize) {
    let mut my_gen = 0usize;
    loop {
        {
            let mut g = sh.gen.lock().unwrap();
            while *g == my_gen {
                if sh.stop.load(Ordering::Relaxed) { return; }
                g = sh.wake.wait(g).unwrap();
            }
            my_gen = *g;
        }
        if sh.stop.load(Ordering::Relaxed) { return; }
        steal_chunks(sh);
        sh.worker_done_gen[id].0.store(my_gen, Ordering::Release);
    }
}

static POOL: std::sync::OnceLock<PersistentPool> = std::sync::OnceLock::new();

pub fn global_pool() -> &'static PersistentPool {
    POOL.get_or_init(|| {
        let n = thread::available_parallelism().map(|p| p.get().min(8)).unwrap_or(1);
        PersistentPool::new(n)
    })
}

// ── parallel_compute ───────────────────────────────────────────────────────

pub struct ComputeParams<'a> {
    pub ith: usize,
    pub nth: usize,
    pub barrier: &'a SpinBarrier,
    pub current_chunk: &'a AtomicUsize,
}

pub fn parallel_compute<F: Fn(&ComputeParams) + Sync + Send>(n_threads: usize, f: F) {
    if n_threads <= 1 {
        let barrier = SpinBarrier::new(1);
        let chunk = AtomicUsize::new(0);
        f(&ComputeParams { ith: 0, nth: 1, barrier: &barrier, current_chunk: &chunk });
        return;
    }
    let barrier = SpinBarrier::new(n_threads);
    let current_chunk = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for t in 1..n_threads {
            let br = &barrier; let cc = &current_chunk; let fr = &f;
            s.spawn(move || {
                fr(&ComputeParams { ith: t, nth: n_threads, barrier: br, current_chunk: cc });
            });
        }
        f(&ComputeParams { ith: 0, nth: n_threads, barrier: &barrier, current_chunk: &current_chunk });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrier() {
        let b = SpinBarrier::new(4);
        let count = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..3 { s.spawn(|| { count.fetch_add(1, Ordering::Relaxed); b.wait(); }); }
            count.fetch_add(1, Ordering::Relaxed); b.wait();
            assert_eq!(count.load(Ordering::Relaxed), 4);
        });
    }

    #[test]
    fn test_pool_sum() {
        let pool = PersistentPool::new(4);
        let data: Vec<i64> = (0..10000).collect();
        let total = AtomicUsize::new(0);
        pool.parallel_chunks(data.len(), 500, &|start, end| {
            total.fetch_add(data[start..end].iter().sum::<i64>() as usize, Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), (0i64..10000).sum::<i64>() as usize);
    }

    #[test]
    fn test_pool_repeated() {
        let pool = PersistentPool::new(4);
        for _ in 0..200 {
            let count = AtomicUsize::new(0);
            pool.parallel_chunks(64, 8, &|_, _| { count.fetch_add(1, Ordering::Relaxed); });
            assert_eq!(count.load(Ordering::Relaxed), 8);
        }
    }

    #[test]
    fn test_pool_single() {
        let pool = PersistentPool::new(8);
        let r = AtomicUsize::new(0);
        pool.parallel_chunks(1, 1, &|_, _| { r.store(42, Ordering::Relaxed); });
        assert_eq!(r.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn test_global_pool() {
        let count = AtomicUsize::new(0);
        global_pool().parallel_chunks(100, 10, &|_, _| { count.fetch_add(1, Ordering::Relaxed); });
        assert_eq!(count.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_run_indexed() {
        let pool = PersistentPool::new(4);
        let nth = pool.thread_count();
        assert_eq!(nth, 4);
        let seen = (0..nth).map(|_| std::sync::atomic::AtomicBool::new(false)).collect::<Vec<_>>();
        pool.run_indexed(&|ith, n| {
            assert_eq!(n, nth);
            assert!(ith < n);
            seen[ith].store(true, Ordering::Relaxed);
        });
        for i in 0..nth {
            assert!(seen[i].load(Ordering::Relaxed), "thread {i} not executed");
        }
    }

    #[test]
    fn test_run_indexed_with_barrier() {
        let pool = PersistentPool::new(4);
        let nth = pool.thread_count();
        let barrier = SpinBarrier::new(nth);
        let phase1 = AtomicUsize::new(0);
        let phase2 = AtomicUsize::new(0);
        pool.run_indexed(&|_ith, _n| {
            phase1.fetch_add(1, Ordering::Relaxed);
            barrier.wait();
            assert_eq!(phase1.load(Ordering::Relaxed), nth);
            phase2.fetch_add(1, Ordering::Relaxed);
            barrier.wait();
            assert_eq!(phase2.load(Ordering::Relaxed), nth);
        });
    }
}
