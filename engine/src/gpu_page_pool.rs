//! GPU KV page-run allocator (p20 step 4b) -- pure integer logic, ungated
//! so `cargo kani` (featureless) can verify it.
//!
//! A [`PageRunAllocator`] hands out CONTIGUOUS runs of pages from a fixed
//! page universe.  Sequences hold runs, so a sequence's block table is
//! affine (`base_page + i`) and every reader that assumes contiguity
//! works from a base-shifted pointer -- no kernel/PTX changes.  Full
//! per-page fragmentation (non-contiguous runs) is deliberately NOT
//! supported here; that would require table-aware decode kernels (see
//! the p20 notes in AGENTS.md).
//!
//! VERIFICATION-FRIENDLY DESIGN (the 2026-11-12 lesson, same class as
//! the moe.rs route() precedent): the free list is a FIXED-CAPACITY
//! array manipulated only via in-place edits and swap-remove -- no
//! sorted-order shifting (Vec insert/remove memmove made CBMC OOM/heap-
//! corrupt at 20+ GB even on a 125 GB host).  Consequences:
//! - the list is UNSORTED; `alloc` is any-fit (scan order), which is
//!   equivalent for run allocation;
//! - `free` coalesces by repeated merge scans until no adjacent pair
//!   remains (maximal coalescing -- the Verus-modeled invariant).
//!
//! Invariants (Kani-verified below):
//! - runs are within `[0, total_pages)` and non-empty
//! - free runs are pairwise disjoint and non-adjacent (maximal
//!   coalescing)
//! - conservation: free pages + allocated pages == total_pages

/// Maximum free-list entries.  Entries are bounded by live allocated
/// runs + 1 (each `free` coalesces before taking a new slot); a pool
/// serving more concurrently-fragmented runs than this panics with a
/// clear message.  128 is far beyond any realistic serving topology.
pub const MAX_FREE_RUNS: usize = 128;

/// A contiguous run of pages: `[start, start + pages)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    pub start: usize,
    pub pages: usize,
}

/// Any-fit run allocator with coalescing free, over `total_pages`.
#[derive(Debug, Clone)]
pub struct PageRunAllocator {
    free: [Run; MAX_FREE_RUNS],
    free_len: usize,
    total_pages: usize,
    /// Pages currently allocated (for conservation checks).
    allocated: usize,
}

impl PageRunAllocator {
    /// New allocator with the whole universe free.
    pub fn new(total_pages: usize) -> Self {
        let mut free = [Run {
            start: 0,
            pages: 0,
        }; MAX_FREE_RUNS];
        free[0] = Run {
            start: 0,
            pages: total_pages,
        };
        Self {
            free,
            free_len: usize::from(total_pages > 0),
            total_pages,
            allocated: 0,
        }
    }

    pub fn total_pages(&self) -> usize {
        self.total_pages
    }

    pub fn free_pages(&self) -> usize {
        // Plain indexed loop (verification-friendly: iterator-adapter
        // folds over runtime-length slices make CBMC explode).
        let mut sum = 0usize;
        let mut i = 0;
        while i < self.free_len {
            sum += self.free[i].pages;
            i += 1;
        }
        sum
    }

    pub fn allocated_pages(&self) -> usize {
        self.allocated
    }

    /// Allocate a contiguous run of `pages` pages (BEST-FIT: the
    /// smallest free run that fits, earliest-scanned on ties -- prefers
    /// holes over the big tail run).  Returns `None` if no free run
    /// fits.  Consumed runs are removed with swap-remove; a partially
    /// consumed run shrinks in place.
    pub fn alloc(&mut self, pages: usize) -> Option<Run> {
        if pages == 0 {
            return None;
        }
        let mut best: Option<usize> = None;
        let mut i = 0;
        while i < self.free_len {
            if self.free[i].pages >= pages
                && best.is_none_or(|b| self.free[i].pages < self.free[b].pages)
            {
                best = Some(i);
            }
            i += 1;
        }
        let idx = best?;
        let run = Run {
            start: self.free[idx].start,
            pages,
        };
        let remainder = self.free[idx].pages - pages;
        if remainder == 0 {
            // Swap-remove (the list is unsorted).
            self.free[idx] = self.free[self.free_len - 1];
            self.free_len -= 1;
        } else {
            self.free[idx] = Run {
                start: self.free[idx].start + pages,
                pages: remainder,
            };
        }
        self.allocated += pages;
        Some(run)
    }

    /// Free a previously allocated run, coalescing with adjacent free
    /// runs until none remain adjacent (maximal coalescing).
    pub fn free(&mut self, mut run: Run) {
        if run.pages == 0 || run.start + run.pages > self.total_pages {
            return;
        }
        self.allocated -= run.pages;

        // Absorb every free run adjacent to `run` (swap-removing each
        // from the list), extending `run`, until nothing is adjacent.
        // Then append the fully coalesced run once.
        loop {
            let mut absorbed: Option<usize> = None;
            let mut i = 0;
            while i < self.free_len {
                // free[i] ends exactly where run starts.
                if self.free[i].start + self.free[i].pages == run.start {
                    run = Run {
                        start: self.free[i].start,
                        pages: self.free[i].pages + run.pages,
                    };
                    absorbed = Some(i);
                    break;
                }
                // free[i] starts exactly where run ends.
                if self.free[i].start == run.start + run.pages {
                    run = Run {
                        start: run.start,
                        pages: run.pages + self.free[i].pages,
                    };
                    absorbed = Some(i);
                    break;
                }
                i += 1;
            }
            match absorbed {
                Some(i) => {
                    self.free[i] = self.free[self.free_len - 1];
                    self.free_len -= 1;
                }
                None => break,
            }
        }

        assert!(
            self.free_len < MAX_FREE_RUNS,
            "PageRunAllocator free-list overflow: more than {MAX_FREE_RUNS} \
             concurrently fragmented runs"
        );
        self.free[self.free_len] = run;
        self.free_len += 1;
    }

    /// Free-list state (for tests/verification).
    pub fn free_runs(&self) -> &[Run] {
        &self.free[..self.free_len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_sequential_then_free_coalesces() {
        let mut a = PageRunAllocator::new(64);
        let r0 = a.alloc(16).unwrap();
        let r1 = a.alloc(16).unwrap();
        let r2 = a.alloc(16).unwrap();
        assert_eq!(r0.start, 0);
        assert_eq!(r1.start, 16);
        assert_eq!(r2.start, 32);
        assert_eq!(a.free_pages(), 16);

        // Free r1: hole at [16, 32).
        a.free(r1);
        assert_eq!(a.free_runs().len(), 2);

        // A 16-page alloc fits either the hole or the (equally sized)
        // tail run -- best-fit ties break to the earliest scanned.
        let r3 = a.alloc(16).unwrap();
        assert_eq!(r3.pages, 16);
        assert!(r3.start == 16 || r3.start == 48, "r3={r3:?}");

        // Free everything -> one coalesced run covering the universe.
        a.free(r0);
        a.free(r2);
        a.free(r3);
        assert_eq!(a.free_runs().len(), 1);
        assert_eq!(a.free_runs()[0].start, 0);
        assert_eq!(a.free_runs()[0].pages, 64);
        assert_eq!(a.allocated_pages(), 0);
    }

    #[test]
    fn triple_fragmentation_coalesces() {
        let mut a = PageRunAllocator::new(48);
        let r0 = a.alloc(16).unwrap();
        let r1 = a.alloc(16).unwrap();
        let r2 = a.alloc(16).unwrap();
        // Free r0 and r2 (r1 keeps [16,32) live): two holes.
        a.free(r0);
        a.free(r2);
        assert_eq!(a.free_runs().len(), 2);
        // Free r1: everything must coalesce back to one run.
        a.free(r1);
        assert_eq!(a.free_runs().len(), 1);
        assert_eq!(a.free_runs()[0].pages, 48);
    }

    #[test]
    fn oversize_alloc_fails() {
        let mut a = PageRunAllocator::new(10);
        assert!(a.alloc(11).is_none());
        assert_eq!(a.free_pages(), 10);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Free runs are pairwise disjoint, non-adjacent, in bounds, and
    /// non-empty.
    fn well_formed(a: &PageRunAllocator) -> bool {
        let n = a.free_len;
        let mut i = 0;
        while i < n {
            if a.free[i].pages == 0 || a.free[i].start + a.free[i].pages > a.total_pages() {
                return false;
            }
            let mut j = i + 1;
            while j < n {
                let (x, y) = (a.free[i], a.free[j]);
                let (lo, hi) = if x.start <= y.start { (x, y) } else { (y, x) };
                // Disjoint AND non-adjacent (maximal coalescing).
                if lo.start + lo.pages >= hi.start {
                    return false;
                }
                j += 1;
            }
            i += 1;
        }
        true
    }

    /// Provable bound on the free-list length for the harness's concrete
    /// operation sequences (each `free` grows the list by at most one
    /// slot; the harness performs at most two).  Asserting this BEFORE
    /// each loop-bearing call lets CBMC bound the loop unwinding --
    /// without it, `free_len` is structurally <= MAX_FREE_RUNS (128) and
    /// every loop over it fails the unwind(20) bound.
    fn bounded(a: &PageRunAllocator) {
        assert!(a.free_len <= 8);
    }

    #[kani::proof]
    #[kani::unwind(20)]
    fn alloc_free_conservation_and_well_formed() {
        // Concrete universe (per the symbolic-length lesson); symbolic
        // request sizes exercise the best-fit/split/coalesce paths.  The
        // free list is a fixed-capacity array with swap-remove only --
        // no memmove loops for CBMC to explode on.
        let total: usize = 16;
        let mut a = PageRunAllocator::new(total);
        bounded(&a);
        assert!(well_formed(&a));
        assert!(a.free_pages() + a.allocated_pages() == total);

        let want1: usize = kani::any();
        kani::assume(want1 >= 1 && want1 <= total);
        let r1 = a.alloc(want1);
        assert!(r1.is_some());
        let r1 = r1.unwrap();
        assert!(r1.start + r1.pages <= total);
        bounded(&a);
        assert!(well_formed(&a));
        assert!(a.free_pages() + a.allocated_pages() == total);

        let want2: usize = kani::any();
        kani::assume(want2 >= 1 && want2 <= total);
        let r2 = a.alloc(want2);
        if let Some(r2) = r2 {
            // Distinct allocated runs never overlap.
            let (lo, hi) = if r1.start <= r2.start { (r1, r2) } else { (r2, r1) };
            assert!(lo.start + lo.pages <= hi.start);
            bounded(&a);
            assert!(well_formed(&a));
            assert!(a.free_pages() + a.allocated_pages() == total);
            a.free(r2);
            bounded(&a);
            assert!(well_formed(&a));
            assert!(a.free_pages() + a.allocated_pages() == total);
        }
        a.free(r1);
        bounded(&a);
        assert!(well_formed(&a));
        assert!(a.free_pages() + a.allocated_pages() == total);
        // After freeing everything the universe is one run again.
        assert!(a.free_runs().len() == 1);
        assert!(a.free_runs()[0].start == 0);
        assert!(a.free_runs()[0].pages == total);
    }

    #[kani::proof]
    #[kani::unwind(20)]
    fn free_rejected_out_of_bounds() {
        let mut a = PageRunAllocator::new(8);
        bounded(&a);
        let r = a.alloc(4).unwrap();
        a.free(Run {
            start: 4,
            pages: 100,
        });
        // The bogus free must not have corrupted state.
        bounded(&a);
        assert!(a.free_pages() + a.allocated_pages() == 8);
        assert!(well_formed(&a));
    }
}
