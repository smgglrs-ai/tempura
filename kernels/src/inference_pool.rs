use std::sync::OnceLock;
use staticpar::StaticPool;

static POOL: OnceLock<StaticPool> = OnceLock::new();

pub fn get() -> &'static StaticPool {
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        StaticPool::new(n.saturating_sub(1).max(1))
    })
}

pub fn n_threads() -> usize {
    get().n_threads() + 1
}

/// Parallel run for closures that capture raw pointers.
///
/// Uses `StaticPool::run_raw` to bypass the `Sync` bound on the closure.
///
/// # Safety
/// - All captured raw pointers must remain valid for the duration of the call.
/// - Each `(start, end)` partition must write to non-overlapping memory.
pub fn par_run<F: Fn(usize, usize)>(n: usize, f: F) {
    unsafe fn trampoline<F: Fn(usize, usize)>(start: usize, end: usize, ctx: *const ()) {
        (*{ ctx as *const F })(start, end)
    }
    unsafe { get().run_raw(n, trampoline::<F>, &f as *const F as *const ()) };
}
