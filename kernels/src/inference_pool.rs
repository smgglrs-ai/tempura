use staticpar::StaticPool;
use std::sync::OnceLock;

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

/// Trait for kernel job types that can be dispatched through the static pool.
/// Each concrete type `J` generates ONE stable function pointer `trampoline::<J>`,
/// so the CPU branch predictor learns a single target for all dispatches of that kernel type.
pub trait KernelJob: Sync {
    fn exec_range(&self, start: usize, end: usize);
}

/// Parallel dispatch using a concrete job type — single stable function pointer.
/// All calls using the same `J` share `trampoline::<J>` → good branch prediction.
pub fn par_run_job<J: KernelJob>(n: usize, job: &J) {
    unsafe fn trampoline<J: KernelJob>(start: usize, end: usize, ctx: *const ()) {
        (*{ ctx as *const J }).exec_range(start, end)
    }
    unsafe { get().run_raw(n, trampoline::<J>, job as *const J as *const ()) }
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
