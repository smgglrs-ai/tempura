//! Persistent thread pool with atomic chunk stealing.
//! Workers park via condvar between ops, wake via notify_all, steal chunks via atomic fetch_add.
//! One condvar wake per parallel_chunks call — not per chunk.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

struct PoolShared {
    chunk_counter: AtomicUsize,
    n_chunks: AtomicUsize,
    chunk_size: AtomicUsize,
    total: AtomicUsize,
    done_count: AtomicUsize,
    stop: AtomicBool,
    n_workers: usize,
    work_ptr: AtomicUsize,
    work_vtable: AtomicUsize,
    gen: Mutex<usize>,
    wake: Condvar,
}

pub struct ChunkPool {
    shared: Arc<PoolShared>,
    _threads: Vec<std::thread::JoinHandle<()>>,
}

impl ChunkPool {
    pub fn new(n_threads: usize) -> Self {
        let n_workers = n_threads.saturating_sub(1);
        let shared = Arc::new(PoolShared {
            chunk_counter: AtomicUsize::new(0),
            n_chunks: AtomicUsize::new(0),
            chunk_size: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
            done_count: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            n_workers,
            work_ptr: AtomicUsize::new(0),
            work_vtable: AtomicUsize::new(0),
            gen: Mutex::new(0),
            wake: Condvar::new(),
        });

        let mut threads = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let sh = Arc::clone(&shared);
            threads.push(std::thread::spawn(move || worker(&sh)));
        }

        Self { shared, _threads: threads }
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
        sh.done_count.store(0, Ordering::Release);

        {
            let mut g = sh.gen.lock().unwrap();
            *g += 1;
        }
        sh.wake.notify_all();

        steal_chunks(sh);

        // Spin-wait for workers (they're already awake, just finishing their chunks)
        while sh.done_count.load(Ordering::Acquire) < sh.n_workers {
            std::hint::spin_loop();
        }
    }
}

impl Drop for ChunkPool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        {
            let mut g = self.shared.gen.lock().unwrap();
            *g += 1;
        }
        self.shared.wake.notify_all();
        for t in self._threads.drain(..) {
            let _ = t.join();
        }
    }
}

#[inline]
fn steal_chunks(sh: &PoolShared) {
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

fn worker(sh: &PoolShared) {
    let mut my_gen = 0usize;
    loop {
        // Wait for new generation
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
        sh.done_count.fetch_add(1, Ordering::Release);
    }
}

static POOL: std::sync::OnceLock<ChunkPool> = std::sync::OnceLock::new();

pub fn global_pool() -> &'static ChunkPool {
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|p| p.get().min(8))
            .unwrap_or(1);
        ChunkPool::new(n)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_sum() {
        let pool = ChunkPool::new(4);
        let data: Vec<i64> = (0..10000).collect();
        let total = AtomicUsize::new(0);
        pool.parallel_chunks(data.len(), 500, &|start, end| {
            let s: usize = data[start..end].iter().sum::<i64>() as usize;
            total.fetch_add(s, Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), (0i64..10000).sum::<i64>() as usize);
    }

    #[test]
    fn test_pool_single_element() {
        let pool = ChunkPool::new(8);
        let result = std::sync::atomic::AtomicI32::new(0);
        pool.parallel_chunks(1, 1, &|_, _| { result.store(42, Ordering::Relaxed); });
        assert_eq!(result.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn test_pool_repeated_calls() {
        let pool = ChunkPool::new(4);
        for _ in 0..100 {
            let count = AtomicUsize::new(0);
            pool.parallel_chunks(64, 8, &|_, _| { count.fetch_add(1, Ordering::Relaxed); });
            assert_eq!(count.load(Ordering::Relaxed), 8);
        }
    }

    #[test]
    fn test_global_pool() {
        let count = AtomicUsize::new(0);
        global_pool().parallel_chunks(100, 10, &|_, _| { count.fetch_add(1, Ordering::Relaxed); });
        assert_eq!(count.load(Ordering::Relaxed), 10);
    }
}
