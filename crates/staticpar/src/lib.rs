//! Static-partition spin-wait thread pool.
//!
//! Zero external dependencies. Pre-spawned threads spin between jobs.
//! Designed for repeated short embarrassingly-parallel workloads where
//! work count is known upfront (GEMV, matrix multiplication rows, etc.).

/// A fixed-size thread pool with spin-wait between jobs.
///
/// Threads never park between `run()` calls (below the spin budget).
/// Work is distributed via a static partition — thread i processes
/// items [i*chunk .. (i+1)*chunk). No work-stealing, no epoch GC.
pub struct StaticPool {
    _inner: (),   // replaced by implementation
}

impl StaticPool {
    /// Create a pool with `n_threads` pre-spawned threads.
    /// Threads spin for up to `spin_budget` iterations between jobs
    /// before yielding. Use `u64::MAX` for always-spinning, `0` to
    /// always yield immediately.
    pub fn new(n_threads: usize, spin_budget: u64) -> Self {
        let _ = (n_threads, spin_budget);
        Self { _inner: () }
    }

    /// Number of worker threads (not counting the calling thread).
    pub fn n_threads(&self) -> usize { 0 }

    /// Run `f(start, end)` for each partition of `0..n`.
    /// The calling thread participates (handles partition 0).
    /// Blocks until all threads complete.
    ///
    /// # Safety contract
    /// `f` must be `Sync` (shared across threads). The pool waits for
    /// all threads before returning, so `f`'s lifetime is bounded by
    /// this call — no escaping references.
    pub fn run<F: Fn(usize, usize) + Sync>(&self, n: usize, f: F) {
        let _ = (n, f);
    }
}

impl Drop for StaticPool {
    fn drop(&mut self) {}
}
