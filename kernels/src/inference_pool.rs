use std::cell::Cell;
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

// Thread-local flag: non-zero dispatch_gen means we're inside a session.
// The session pointer is valid for the duration of with_session's call.
thread_local! {
    static SESSION_PTR: Cell<*mut staticpar::Session<'static>> =
        const { Cell::new(std::ptr::null_mut()) };
}

/// Run `f` inside a StaticPool session: workers wake once at entry,
/// then use spinning atomic barriers between all `par_run()` calls inside `f`.
/// Workers return to condvar sleep when `f` returns.
///
/// Matches llama.cpp's ggml_threadpool: one cond_broadcast per token,
/// ~0.1μs spinning barriers between matmul ops (vs ~10μs condvar per op).
pub fn with_session<R>(f: impl FnOnce() -> R) -> R {
    let pool = get();
    let mut session = pool.start_session();

    // Safety: session lives for the duration of this function.
    // SESSION_PTR is only accessed from this thread.
    let session_ptr: *mut staticpar::Session<'static> =
        unsafe { std::mem::transmute(&mut session as *mut staticpar::Session<'_>) };

    SESSION_PTR.with(|cell| cell.set(session_ptr));
    let result = f();
    SESSION_PTR.with(|cell| cell.set(std::ptr::null_mut()));
    drop(session);
    result
}

/// Parallel run for closures that capture raw pointers.
///
/// If called within `with_session()`, uses the session's spinning barrier
/// (no condvar, ~0.1μs overhead). Otherwise falls back to condvar-based wake.
///
/// # Safety
/// - All captured raw pointers must remain valid for the duration of the call.
/// - Each `(start, end)` partition must write to non-overlapping memory.
pub fn par_run<F: Fn(usize, usize)>(n: usize, f: F) {
    SESSION_PTR.with(|cell| {
        let ptr = cell.get();
        if !ptr.is_null() {
            // Inside a session — use spinning barrier.
            // Safety: pointer is valid (set by with_session on same thread).
            let session: &mut staticpar::Session<'_> =
                unsafe { &mut *(ptr as *mut staticpar::Session<'_>) };
            unsafe fn trampoline_s<F: Fn(usize, usize)>(start: usize, end: usize, ctx: *const ()) {
                (*{ ctx as *const F })(start, end)
            }
            unsafe { session.run_raw(n, trampoline_s::<F>, &f as *const F as *const ()) };
            return;
        }
        // Outside a session — use condvar-based wake.
        unsafe fn trampoline<F: Fn(usize, usize)>(start: usize, end: usize, ctx: *const ()) {
            (*{ ctx as *const F })(start, end)
        }
        unsafe { get().run_raw(n, trampoline::<F>, &f as *const F as *const ()) };
    });
}
