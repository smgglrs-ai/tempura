use std::sync::OnceLock;
use staticpar::StaticPool;

static POOL: OnceLock<StaticPool> = OnceLock::new();

pub fn get() -> &'static StaticPool {
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let spin = std::env::var("TEMPURA_POOL_SPIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50_000u64);
        // n-1 worker threads; calling thread is the nth participant — total = n, no oversubscription
        StaticPool::new(n.saturating_sub(1).max(1), spin)
    })
}

pub fn n_threads() -> usize {
    get().n_threads() + 1
}
