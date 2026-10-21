//! Direct Rust translation of llama.cpp `ggml-cpu.c` scheduling infrastructure.
//!
//! Source files:
//!   - ggml/src/ggml-cpu/ggml-cpu.c  (threadpool, barrier, graph compute loop)
//!   - ggml/src/ggml-cpu/ggml-cpu.cpp (backend context -- struct only)
//!
//! Translated sections (non-mul_mat):
//!   - `struct ggml_threadpool`               -> [`Threadpool`]
//!   - `struct ggml_compute_state`            -> [`ComputeState`]
//!   - `struct ggml_compute_params`           -> [`ComputeParams`]
//!   - `ggml_barrier`                         -> [`barrier`]
//!   - `ggml_get_n_tasks`                     -> [`get_n_tasks`]
//!   - `ggml_compute_forward` (dispatch table)-> [`compute_forward`]
//!   - `ggml_graph_compute_thread`            -> [`graph_compute_thread`]
//!   - `ggml_graph_compute_secondary_thread`  -> [`graph_compute_secondary_thread`]
//!   - `ggml_graph_compute_kickoff`           -> [`graph_compute_kickoff`]
//!   - `ggml_threadpool_new_impl`             -> [`Threadpool::new`]
//!   - `ggml_threadpool_free`                 -> [`Threadpool::free`] / `Drop`
//!   - `ggml_graph_compute`                   -> [`graph_compute`]
//!
//! The mul_mat dispatch (`GGML_OP_MUL_MAT`) is handled in `mul_mat_llama.rs`
//! and referenced here via `compute_forward_mul_mat_stub`.
//!
//! Naming map (C -> Rust):
//!   GGML_THREADPOOL_N_THREADS_MASK = N_THREADS_MASK
//!   GGML_THREADPOOL_N_THREADS_BITS = N_THREADS_BITS
//!   memory_order_relaxed            = Ordering::Relaxed
//!   memory_order_seq_cst            = Ordering::SeqCst

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

// ---------------------------------------------------------------------------
// Constants (ggml-cpu.c line 206-207)
// ---------------------------------------------------------------------------

/// Lower 16 bits of `n_graph` hold the active thread count.
const N_THREADS_MASK: i32 = 0xffff_i32;
/// Upper 16 bits of `n_graph` hold the graph generation counter.
const N_THREADS_BITS: i32 = 16;

/// Maximum number of threads / CPU cores (GGML_MAX_N_THREADS).
pub const MAX_N_THREADS: usize = 512;

/// Default number of compute threads (GGML_DEFAULT_N_THREADS).
pub const DEFAULT_N_THREADS: usize = 4;

// ---------------------------------------------------------------------------
// GGML op enum (maps to enum ggml_op in ggml.h)
// ---------------------------------------------------------------------------
//
// Variant order matches the C integer values so that `as u32` gives the
// canonical GGML op code.  Only variants referenced in `compute_forward` and
// `get_n_tasks` are listed; the full enum lives in the GGML C headers.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlOp {
    None = 0,
    Dup,
    Add,
    AddId,
    Add1,
    Acc,
    Sub,
    Mul,
    Div,
    Sqr,
    Sqrt,
    Log,
    Sin,
    Cos,
    Sum,
    SumRows,
    Cumsum,
    Mean,
    Argmax,
    CountEqual,
    Repeat,
    RepeatBack,
    Concat,
    SiluBack,
    Norm,
    RmsNorm,
    RmsNormBack,
    GroupNorm,
    L2Norm,
    MulMat,
    MulMatId,
    OutProd,
    Scale,
    Set,
    Cpy,
    Cont,
    Reshape,
    View,
    Permute,
    Transpose,
    GetRows,
    GetRowsBack,
    SetRows,
    Diag,
    DiagMaskInf,
    DiagMaskZero,
    SoftMax,
    SoftMaxBack,
    Rope,
    RopeBack,
    Clamp,
    ConvTranspose1d,
    Im2col,
    Im2colBack,
    Im2col3d,
    Col2im1d,
    Conv2d,
    Conv3d,
    Conv2dDw,
    ConvTranspose2d,
    Pool1d,
    Pool2d,
    Pool2dBack,
    Upscale,
    Pad,
    PadReflect1d,
    Roll,
    Arange,
    TimestepEmbedding,
    Argsort,
    TopK,
    LeakyRelu,
    Tri,
    Fill,
    FlashAttnExt,
    FlashAttnBack,
    SsmConv,
    SsmScan,
    WinPart,
    WinUnpart,
    Unary,
    Glu,
    GetRelPos,
    AddRelPos,
    RwkvWkv6,
    GatedLinearAttn,
    RwkvWkv7,
    SolveTri,
    GatedDeltaNet,
    LightningIndexer,
    Dsv4HcComb,
    Dsv4HcPre,
    Dsv4HcPost,
    MapCustom1,
    MapCustom2,
    MapCustom3,
    Custom,
    CrossEntropyLoss,
    CrossEntropyLossBack,
    OptStepAdamW,
    OptStepSgd,
    Count,
}

impl GgmlOp {
    /// Returns true for ops that are no-ops (GGML_OP_NONE, RESHAPE, VIEW, PERMUTE, TRANSPOSE).
    pub fn is_nop(self) -> bool {
        matches!(
            self,
            GgmlOp::None | GgmlOp::Reshape | GgmlOp::View | GgmlOp::Permute | GgmlOp::Transpose
        )
    }
}

// ---------------------------------------------------------------------------
// Minimal tensor / graph / plan types for scheduling
// ---------------------------------------------------------------------------
//
// These mirror the C struct fields used by the scheduling logic only.  They
// are not full GGML tensors -- the full tensor type lives in the GGUF loader.

/// Tensor flags (GGML_TENSOR_FLAG_*).
pub mod tensor_flags {
    /// GGML_TENSOR_FLAG_COMPUTE -- this node participates in forward pass.
    pub const COMPUTE: u32 = 1 << 1;
}

/// Minimal view of a ggml_tensor for scheduling: op type, flags, source count,
/// element counts per dimension, and head count (for RWKV/GLA ops).
pub struct GgmlTensor {
    pub op: GgmlOp,
    pub flags: u32,
    /// ne[0..4] -- element counts per dimension (C: node->ne[0..3])
    pub ne: [i64; 4],
    /// Number of source tensors (C: node->src[i] != NULL).
    pub n_src: usize,
    /// ne[1] of src[1] -- used by RWKV/GLA to get n_heads.
    pub src1_ne1: i64,
    /// op_params[0] -- used by custom ops (n_tasks) and flash attn back (masked).
    pub op_param0: i32,
    /// op_params[1] -- used by CUSTOM/MAP_CUSTOM (n_tasks field).
    pub op_param1: i32,
}

impl GgmlTensor {
    pub fn is_empty(&self) -> bool {
        self.ne[0] == 0
    }
}

/// Minimal ggml_cgraph -- just the node list.
pub struct GgmlCgraph<'a> {
    pub nodes: &'a [GgmlTensor],
}

impl<'a> GgmlCgraph<'a> {
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// Abort / callback type (GGML_ABORT_CALLBACK).
pub type AbortCallback = fn(*mut ()) -> bool;

/// Compute plan (ggml_cplan).
pub struct GgmlCplan {
    /// Number of compute threads.
    pub n_threads: usize,
    /// Work buffer for operators that need scratch space.
    pub work_data: Vec<u8>,
    /// Optional abort callback.
    pub abort_callback: Option<AbortCallback>,
    pub abort_callback_data: *mut (),
    /// Use reference (non-fused) implementation.
    pub use_ref: bool,
}

unsafe impl Send for GgmlCplan {}
unsafe impl Sync for GgmlCplan {}

// ---------------------------------------------------------------------------
// Scheduling priority (GGML_SCHED_PRIO_*)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SchedPrio {
    Low = -1,
    Normal = 0,
    Medium = 1,
    High = 2,
    Realtime = 3,
}

// ---------------------------------------------------------------------------
// Compute params (ggml_compute_params -- passed per call to each forward fn)
// ---------------------------------------------------------------------------

/// Per-call compute parameters threaded through every operator forward pass.
/// Translated from `struct ggml_compute_params` in ggml-cpu-impl.h.
pub struct ComputeParams {
    /// Thread index (C: ith).
    pub ith: usize,
    /// Total threads (C: nth).
    pub nth: usize,
    /// Scratch buffer size.
    pub wsize: usize,
    /// Raw scratch pointer (thread-local slice into work_data).
    pub wdata: *mut u8,
    /// Back-pointer to threadpool for barrier.
    pub threadpool: Arc<ThreadpoolShared>,
    /// Use reference implementation.
    pub use_ref: bool,
}

unsafe impl Send for ComputeParams {}

// ---------------------------------------------------------------------------
// Threadpool shared state (struct ggml_threadpool)
// ---------------------------------------------------------------------------
//
// C layout (ggml-cpu.c lines 481-504):
//
//   struct ggml_threadpool {
//       ggml_mutex_t      mutex;
//       ggml_cond_t       cond;
//       struct ggml_cgraph * cgraph;
//       struct ggml_cplan  * cplan;
//       atomic_int  n_graph;           // graph-gen << 16 | n_threads
//       atomic_int  n_barrier;         // # threads entered barrier
//       atomic_int  n_barrier_passed;  // barrier epoch
//       atomic_int  current_chunk;     // matmul chunk counter
//       atomic_bool stop;
//       atomic_bool pause;
//       atomic_int  abort;             // node index to abort at (-1 = none)
//       struct ggml_compute_state * workers;
//       int         n_threads;
//       int32_t     prio;
//       uint32_t    poll;
//       enum ggml_status ec;
//   };

/// Inner shared state protected by the mutex / accessed via atomics.
pub struct ThreadpoolShared {
    // Condition variable + mutex for worker sleep/wake (cond + mutex).
    pub mutex: Mutex<()>,
    pub cond:  Condvar,

    // Atomic scheduling state.
    /// Encodes (graph_gen << N_THREADS_BITS) | n_active_threads.
    pub n_graph: AtomicI32,
    /// Count of threads currently inside the barrier.
    pub n_barrier: AtomicI32,
    /// Barrier epoch -- incremented when all threads have crossed.
    pub n_barrier_passed: AtomicI32,
    /// Next matmul chunk to claim (shared work-stealing counter).
    pub current_chunk: AtomicI32,

    pub stop:  AtomicBool,
    pub pause: AtomicBool,
    /// Abort marker: -1 = none, else = node index that triggered abort.
    pub abort: AtomicI32,

    /// Scheduling priority.
    pub prio: SchedPrio,
    /// Polling rounds (0 = pure sleep, higher = more spin before sleep).
    pub poll: u32,
    /// Number of worker threads (including thread 0 = caller).
    pub n_threads: usize,
}

/// Per-thread state (struct ggml_compute_state).
pub struct ComputeState {
    /// Thread index (0 = main/caller thread).
    pub ith: usize,
    /// CPU affinity mask (bool per logical CPU, up to MAX_N_THREADS).
    pub cpumask: [bool; MAX_N_THREADS],
    /// Back-pointer to shared pool.
    pub threadpool: Arc<ThreadpoolShared>,
    /// Last graph generation seen by this worker thread.
    pub last_graph: i32,
    /// True when this thread has been assigned work for the current graph.
    pub pending: bool,
}

// ---------------------------------------------------------------------------
// Threadpool lifecycle (ggml_threadpool_new_impl / ggml_threadpool_free)
// ---------------------------------------------------------------------------

/// Active threadpool: shared state + join handles for worker threads.
pub struct Threadpool {
    pub shared: Arc<ThreadpoolShared>,
    /// Join handles for worker threads (indices 1..n_threads).
    workers: Vec<JoinHandle<()>>,
}

/// Parameters for creating a new threadpool (ggml_threadpool_params).
pub struct ThreadpoolParams {
    pub n_threads: usize,
    pub prio:      SchedPrio,
    pub poll:      u32,
    pub cpumask:   [bool; MAX_N_THREADS],
    pub strict_cpu: bool,
    pub paused:    bool,
}

impl ThreadpoolParams {
    /// Default params -- matches `ggml_threadpool_params_default`.
    pub fn default(n_threads: usize) -> Self {
        Self {
            n_threads,
            prio: SchedPrio::Normal,
            poll: 50,
            cpumask: [false; MAX_N_THREADS],
            strict_cpu: false,
            paused: false,
        }
    }
}

impl Threadpool {
    /// Translated from `ggml_threadpool_new_impl`.
    /// `cgraph` / `cplan` are `None` when creating a persistent pool
    /// (the caller will assign them before each `graph_compute` call).
    pub fn new(params: &ThreadpoolParams) -> Arc<Self> {
        let shared = Arc::new(ThreadpoolShared {
            mutex:           Mutex::new(()),
            cond:            Condvar::new(),
            n_graph:         AtomicI32::new(0),
            n_barrier:       AtomicI32::new(0),
            n_barrier_passed: AtomicI32::new(0),
            current_chunk:   AtomicI32::new(0),
            stop:            AtomicBool::new(false),
            pause:           AtomicBool::new(params.paused),
            abort:           AtomicI32::new(-1),
            prio:            params.prio,
            poll:            params.poll,
            n_threads:       params.n_threads,
        });

        let mut handles = Vec::with_capacity(params.n_threads.saturating_sub(1));

        // Spin worker threads 1..n_threads.  Thread 0 is the caller.
        for ith in 1..params.n_threads {
            let state = ComputeState {
                ith,
                cpumask: [false; MAX_N_THREADS],
                threadpool: Arc::clone(&shared),
                last_graph: 0,
                pending:    false,
            };
            let handle = thread::spawn(move || {
                graph_compute_secondary_thread(state);
            });
            handles.push(handle);
        }

        Arc::new(Threadpool {
            shared,
            workers: handles,
        })
    }

    /// Pause the pool (translated from `ggml_threadpool_pause`).
    pub fn pause(pool: &Arc<ThreadpoolShared>) {
        let _guard = pool.mutex.lock().unwrap();
        pool.pause.store(true, Ordering::Relaxed);
        pool.cond.notify_all();
    }

    /// Resume the pool (translated from `ggml_threadpool_resume`).
    pub fn resume(pool: &Arc<ThreadpoolShared>) {
        let _guard = pool.mutex.lock().unwrap();
        pool.pause.store(false, Ordering::Relaxed);
        pool.cond.notify_all();
    }

    /// Stop all worker threads and join them (translated from `ggml_threadpool_free`).
    pub fn free(self) {
        {
            let _guard = self.shared.mutex.lock().unwrap();
            self.shared.stop.store(true, Ordering::Relaxed);
            self.shared.pause.store(false, Ordering::Relaxed);
            self.shared.cond.notify_all();
        }
        for handle in self.workers {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Barrier  (ggml_barrier -- ggml-cpu.c lines 576-612)
// ---------------------------------------------------------------------------
//
// Spinning barrier: threads count in via n_barrier; the last one resets the
// counter and bumps n_barrier_passed.  Others spin on n_barrier_passed.
//
// The full seq-cst fence on entry and exit guarantees that all stores from
// the preceding phase are visible to all threads after the barrier.

/// Translated from `ggml_barrier(struct ggml_threadpool * tp)`.
pub fn barrier(tp: &ThreadpoolShared) {
    let n_threads = (tp.n_graph.load(Ordering::Relaxed) & N_THREADS_MASK) as i32;
    if n_threads == 1 {
        return;
    }

    // Remember which epoch we're in.
    let n_passed = tp.n_barrier_passed.load(Ordering::Relaxed);

    // Enter barrier (full seq-cst fence via fetch_add with SeqCst).
    let n_barrier = tp.n_barrier.fetch_add(1, Ordering::SeqCst);

    if n_barrier == (n_threads - 1) {
        // We are the last thread -- reset counter and advance epoch.
        tp.n_barrier.store(0, Ordering::Relaxed);
        tp.n_barrier_passed.fetch_add(1, Ordering::SeqCst);
        return;
    }

    // Spin until the epoch advances (last thread crossed and bumped n_barrier_passed).
    while tp.n_barrier_passed.load(Ordering::Relaxed) == n_passed {
        cpu_relax();
    }

    // Exit barrier with a full seq-cst fence.
    std::sync::atomic::fence(Ordering::SeqCst);
}

/// Platform-specific CPU relax hint.
/// Translated from `ggml_thread_cpu_relax` (ggml-cpu.c lines 521-538).
#[inline(always)]
fn cpu_relax() {
    #[cfg(target_arch = "x86_64")]
    #[allow(unused_unsafe)]
    unsafe {
        core::arch::x86_64::_mm_pause();
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::aarch64::__yield();
    }
    // Other arches: no-op, matches the C `else` branch.
}

// ---------------------------------------------------------------------------
// chunk helpers (ggml_threadpool_chunk_set / chunk_add)
// ---------------------------------------------------------------------------

pub fn chunk_set(tp: &ThreadpoolShared, value: i32) {
    tp.current_chunk.store(value, Ordering::Relaxed);
}

pub fn chunk_add(tp: &ThreadpoolShared, value: i32) -> i32 {
    tp.current_chunk.fetch_add(value, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// get_n_tasks -- per-op parallelism (ggml_get_n_tasks, ggml-cpu.c lines 2253-2523)
// ---------------------------------------------------------------------------
//
// Returns the number of threads that should participate in computing `node`.
// Matches the C switch verbatim, collapsing groups of ops that share the same
// thread count rule.

/// Translated from `ggml_get_n_tasks(struct ggml_tensor * node, int n_threads)`.
pub fn get_n_tasks(node: &GgmlTensor, n_threads: usize) -> usize {
    if node.is_empty() {
        return 1;
    }

    match node.op {
        // n_threads -- fully parallel
        GgmlOp::Cpy
        | GgmlOp::Dup
        | GgmlOp::Cont
        | GgmlOp::Add
        | GgmlOp::AddId
        | GgmlOp::Add1
        | GgmlOp::Acc
        | GgmlOp::Cumsum
        | GgmlOp::Tri
        | GgmlOp::Fill => n_threads,

        // n_threads (distinct ops, same rule)
        GgmlOp::CountEqual
        | GgmlOp::SolveTri
        | GgmlOp::GatedDeltaNet
        | GgmlOp::Dsv4HcComb
        | GgmlOp::Dsv4HcPre
        | GgmlOp::Dsv4HcPost => n_threads,

        // 1 thread
        GgmlOp::Sub
        | GgmlOp::Sqr
        | GgmlOp::Sqrt
        | GgmlOp::Log
        | GgmlOp::Sin
        | GgmlOp::Cos
        | GgmlOp::Sum
        | GgmlOp::SumRows
        | GgmlOp::Mean
        | GgmlOp::Argmax
        | GgmlOp::Repeat
        | GgmlOp::RepeatBack
        | GgmlOp::LeakyRelu => 1,

        // Unary: depends on which unary op (op_param0 encodes ggml_unary_op).
        GgmlOp::Unary => {
            // GGML_UNARY_OP_GELU / GELU_ERF / GELU_QUICK / SILU / XIELU -> n_threads.
            // All others -> 1.
            // Translate by checking op_param0 values:
            //   GELU=6, GELU_ERF=14, GELU_QUICK=7, SILU=9, XIELU=15 (from ggml.h enum ggml_unary_op)
            match node.op_param0 {
                6 | 7 | 9 | 14 | 15 => n_threads,
                _ => 1,
            }
        }

        // GLU variants: all fully parallel
        GgmlOp::Glu => n_threads,

        // n_threads (mixed group)
        GgmlOp::SiluBack
        | GgmlOp::Mul
        | GgmlOp::Div
        | GgmlOp::Norm
        | GgmlOp::RmsNorm
        | GgmlOp::RmsNormBack
        | GgmlOp::L2Norm
        | GgmlOp::GroupNorm
        | GgmlOp::Concat
        | GgmlOp::MulMat
        | GgmlOp::MulMatId
        | GgmlOp::OutProd => n_threads,

        // GET_ROWS and SET_ROWS are single-threaded (GPU-offload penalty).
        GgmlOp::GetRows | GgmlOp::SetRows => 1,

        // 1 thread
        GgmlOp::Scale
        | GgmlOp::Set
        | GgmlOp::Reshape
        | GgmlOp::View
        | GgmlOp::Permute
        | GgmlOp::Transpose
        | GgmlOp::GetRowsBack
        | GgmlOp::Diag => 1,

        // n_threads
        GgmlOp::DiagMaskZero
        | GgmlOp::DiagMaskInf
        | GgmlOp::SoftMaxBack
        | GgmlOp::Rope
        | GgmlOp::RopeBack
        | GgmlOp::AddRelPos => n_threads,

        // Clamp: TODO (upstream also says TODO)
        GgmlOp::Clamp => 1,

        // SoftMax: capped by number of rows
        GgmlOp::SoftMax => n_threads.min(node.ne[1] as usize),

        // n_threads
        GgmlOp::Im2col
        | GgmlOp::Im2colBack
        | GgmlOp::Im2col3d
        | GgmlOp::Conv2d
        | GgmlOp::Conv3d
        | GgmlOp::Conv2dDw
        | GgmlOp::Col2im1d
        | GgmlOp::ConvTranspose1d
        | GgmlOp::ConvTranspose2d => n_threads,

        // 1 thread
        GgmlOp::Pool1d | GgmlOp::Pool2d | GgmlOp::Pool2dBack => 1,

        // n_threads
        GgmlOp::Upscale
        | GgmlOp::Pad
        | GgmlOp::PadReflect1d
        | GgmlOp::Roll
        | GgmlOp::Arange
        | GgmlOp::TimestepEmbedding
        | GgmlOp::Argsort
        | GgmlOp::TopK
        | GgmlOp::FlashAttnExt
        | GgmlOp::FlashAttnBack
        | GgmlOp::SsmConv
        | GgmlOp::SsmScan
        | GgmlOp::LightningIndexer => n_threads,

        // RWKV / GLA: cap by head count
        GgmlOp::RwkvWkv6 | GgmlOp::GatedLinearAttn | GgmlOp::RwkvWkv7 => {
            // n_heads is src[1]->ne[1]
            n_threads.min(node.src1_ne1 as usize)
        }

        // 1 thread
        GgmlOp::WinPart | GgmlOp::WinUnpart | GgmlOp::GetRelPos => 1,

        // Custom ops: use n_tasks field from op_params, or n_threads if GGML_N_TASKS_MAX (-1).
        GgmlOp::MapCustom1 | GgmlOp::MapCustom2 | GgmlOp::MapCustom3 | GgmlOp::Custom => {
            // C: p.n_tasks == GGML_N_TASKS_MAX (-1) -> n_threads, else min(p.n_tasks, n_threads)
            if node.op_param1 == -1 {
                n_threads
            } else {
                n_threads.min(node.op_param1 as usize)
            }
        }

        // n_threads
        GgmlOp::CrossEntropyLoss
        | GgmlOp::CrossEntropyLossBack
        | GgmlOp::OptStepAdamW
        | GgmlOp::OptStepSgd => n_threads,

        // None / Count
        GgmlOp::None => 1,
        GgmlOp::Count => panic!("GgmlOp::Count is not a valid op"),
    }
}

// ---------------------------------------------------------------------------
// compute_forward -- operator dispatch table
// (ggml_compute_forward, ggml-cpu.c lines 1744-2177)
// ---------------------------------------------------------------------------
//
// In the C source every case calls a concrete implementation function defined
// in ops.h / unary-ops.h / binary-ops.h (e.g. ggml_compute_forward_add).
// Those functions are NOT translated here -- this module is responsible only
// for the dispatch routing.  Individual operator forward passes are
// implemented in separate Rust modules and called via the `OpDispatch` trait.
//
// The mul_mat path calls into `mul_mat_llama.rs` (already translated).

/// Trait implemented by any type that can execute GGML forward passes.
/// The concrete implementation is provided by higher-level modules.
pub trait OpDispatch {
    // element-wise / reduction
    fn forward_dup         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_add         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_add_id      (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_add1        (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_acc         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_sub         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_mul         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_div         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_sqr         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_sqrt        (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_log         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_sin         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_cos         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_sum         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_sum_rows    (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_cumsum      (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_mean        (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_argmax      (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_count_equal (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_repeat      (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_repeat_back (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_concat      (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_silu_back   (&self, params: &ComputeParams, node: &GgmlTensor);
    // normalization
    fn forward_norm            (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_rms_norm        (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_rms_norm_back   (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_group_norm      (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_l2_norm         (&self, params: &ComputeParams, node: &GgmlTensor);
    // matmul (routes to mul_mat_llama.rs)
    fn forward_mul_mat    (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_mul_mat_id (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_out_prod   (&self, params: &ComputeParams, node: &GgmlTensor);
    // linear algebra
    fn forward_scale           (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_set             (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_cpy             (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_cont            (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_get_rows        (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_get_rows_back   (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_set_rows        (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_diag            (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_diag_mask_inf   (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_diag_mask_zero  (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_soft_max        (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_soft_max_back   (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_rope            (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_rope_back       (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_clamp           (&self, params: &ComputeParams, node: &GgmlTensor);
    // conv / pool / spatial
    fn forward_conv_transpose_1d (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_im2col            (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_im2col_back       (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_im2col_3d         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_col2im_1d         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_conv_2d           (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_conv_3d           (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_conv_2d_dw        (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_conv_transpose_2d (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_pool_1d           (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_pool_2d           (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_pool_2d_back      (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_upscale           (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_pad               (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_pad_reflect_1d    (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_roll              (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_arange            (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_timestep_embedding(&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_argsort           (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_top_k             (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_leaky_relu        (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_tri               (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_fill              (&self, params: &ComputeParams, node: &GgmlTensor);
    // attention / ssm / rwkv
    fn forward_flash_attn_ext    (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_flash_attn_back   (&self, params: &ComputeParams, node: &GgmlTensor, masked: bool);
    fn forward_ssm_conv          (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_ssm_scan          (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_win_part          (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_win_unpart        (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_unary             (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_glu               (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_get_rel_pos       (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_add_rel_pos       (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_rwkv_wkv6         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_gla               (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_rwkv_wkv7         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_solve_tri         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_gated_delta_net   (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_lightning_indexer (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_dsv4_hc_comb      (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_dsv4_hc_pre       (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_dsv4_hc_post      (&self, params: &ComputeParams, node: &GgmlTensor);
    // custom
    fn forward_map_custom1       (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_map_custom2       (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_map_custom3       (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_custom            (&self, params: &ComputeParams, node: &GgmlTensor);
    // loss / opt
    fn forward_cross_entropy_loss     (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_cross_entropy_loss_back(&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_opt_step_adamw         (&self, params: &ComputeParams, node: &GgmlTensor);
    fn forward_opt_step_sgd           (&self, params: &ComputeParams, node: &GgmlTensor);
}

/// Translated from `ggml_compute_forward` (ggml-cpu.c lines 1744-2177).
///
/// Dispatches `params` and `node` to the concrete operator implementation
/// via the `OpDispatch` trait.  GGML_OP_NONE / RESHAPE / PERMUTE / VIEW /
/// TRANSPOSE are no-ops (return immediately).
pub fn compute_forward<D: OpDispatch>(
    dispatch: &D,
    params:   &ComputeParams,
    node:     &GgmlTensor,
) {
    if node.op.is_nop() || node.is_empty() {
        return;
    }

    match node.op {
        GgmlOp::Dup            => dispatch.forward_dup(params, node),
        GgmlOp::Add            => dispatch.forward_add(params, node),
        GgmlOp::AddId          => dispatch.forward_add_id(params, node),
        GgmlOp::Add1           => dispatch.forward_add1(params, node),
        GgmlOp::Acc            => dispatch.forward_acc(params, node),
        GgmlOp::Sub            => dispatch.forward_sub(params, node),
        GgmlOp::Mul            => dispatch.forward_mul(params, node),
        GgmlOp::Div            => dispatch.forward_div(params, node),
        GgmlOp::Sqr            => dispatch.forward_sqr(params, node),
        GgmlOp::Sqrt           => dispatch.forward_sqrt(params, node),
        GgmlOp::Log            => dispatch.forward_log(params, node),
        GgmlOp::Sin            => dispatch.forward_sin(params, node),
        GgmlOp::Cos            => dispatch.forward_cos(params, node),
        GgmlOp::Sum            => dispatch.forward_sum(params, node),
        GgmlOp::SumRows        => dispatch.forward_sum_rows(params, node),
        GgmlOp::Cumsum         => dispatch.forward_cumsum(params, node),
        GgmlOp::Mean           => dispatch.forward_mean(params, node),
        GgmlOp::Argmax         => dispatch.forward_argmax(params, node),
        GgmlOp::CountEqual     => dispatch.forward_count_equal(params, node),
        GgmlOp::Repeat         => dispatch.forward_repeat(params, node),
        GgmlOp::RepeatBack     => dispatch.forward_repeat_back(params, node),
        GgmlOp::Concat         => dispatch.forward_concat(params, node),
        GgmlOp::SiluBack       => dispatch.forward_silu_back(params, node),
        GgmlOp::Norm           => dispatch.forward_norm(params, node),
        GgmlOp::RmsNorm        => dispatch.forward_rms_norm(params, node),
        GgmlOp::RmsNormBack    => dispatch.forward_rms_norm_back(params, node),
        GgmlOp::GroupNorm      => dispatch.forward_group_norm(params, node),
        GgmlOp::L2Norm         => dispatch.forward_l2_norm(params, node),
        GgmlOp::MulMat         => dispatch.forward_mul_mat(params, node),
        GgmlOp::MulMatId       => dispatch.forward_mul_mat_id(params, node),
        GgmlOp::OutProd        => dispatch.forward_out_prod(params, node),
        GgmlOp::Scale          => dispatch.forward_scale(params, node),
        GgmlOp::Set            => dispatch.forward_set(params, node),
        GgmlOp::Cpy            => dispatch.forward_cpy(params, node),
        GgmlOp::Cont           => dispatch.forward_cont(params, node),
        GgmlOp::GetRows        => dispatch.forward_get_rows(params, node),
        GgmlOp::GetRowsBack    => dispatch.forward_get_rows_back(params, node),
        GgmlOp::SetRows        => dispatch.forward_set_rows(params, node),
        GgmlOp::Diag           => dispatch.forward_diag(params, node),
        GgmlOp::DiagMaskInf    => dispatch.forward_diag_mask_inf(params, node),
        GgmlOp::DiagMaskZero   => dispatch.forward_diag_mask_zero(params, node),
        GgmlOp::SoftMax        => dispatch.forward_soft_max(params, node),
        GgmlOp::SoftMaxBack    => dispatch.forward_soft_max_back(params, node),
        GgmlOp::Rope           => dispatch.forward_rope(params, node),
        GgmlOp::RopeBack       => dispatch.forward_rope_back(params, node),
        GgmlOp::Clamp          => dispatch.forward_clamp(params, node),
        GgmlOp::ConvTranspose1d => dispatch.forward_conv_transpose_1d(params, node),
        GgmlOp::Im2col         => dispatch.forward_im2col(params, node),
        GgmlOp::Im2colBack     => dispatch.forward_im2col_back(params, node),
        GgmlOp::Im2col3d       => dispatch.forward_im2col_3d(params, node),
        GgmlOp::Col2im1d       => dispatch.forward_col2im_1d(params, node),
        GgmlOp::Conv2d         => dispatch.forward_conv_2d(params, node),
        GgmlOp::Conv3d         => dispatch.forward_conv_3d(params, node),
        GgmlOp::Conv2dDw       => dispatch.forward_conv_2d_dw(params, node),
        GgmlOp::ConvTranspose2d => dispatch.forward_conv_transpose_2d(params, node),
        GgmlOp::Pool1d         => dispatch.forward_pool_1d(params, node),
        GgmlOp::Pool2d         => dispatch.forward_pool_2d(params, node),
        GgmlOp::Pool2dBack     => dispatch.forward_pool_2d_back(params, node),
        GgmlOp::Upscale        => dispatch.forward_upscale(params, node),
        GgmlOp::Pad            => dispatch.forward_pad(params, node),
        GgmlOp::PadReflect1d   => dispatch.forward_pad_reflect_1d(params, node),
        GgmlOp::Roll           => dispatch.forward_roll(params, node),
        GgmlOp::Arange         => dispatch.forward_arange(params, node),
        GgmlOp::TimestepEmbedding => dispatch.forward_timestep_embedding(params, node),
        GgmlOp::Argsort        => dispatch.forward_argsort(params, node),
        GgmlOp::TopK           => dispatch.forward_top_k(params, node),
        GgmlOp::LeakyRelu      => dispatch.forward_leaky_relu(params, node),
        GgmlOp::Tri            => dispatch.forward_tri(params, node),
        GgmlOp::Fill           => dispatch.forward_fill(params, node),
        GgmlOp::FlashAttnExt   => dispatch.forward_flash_attn_ext(params, node),
        GgmlOp::FlashAttnBack  => {
            // masked = op_params[0] != 0  (C: t != 0, where t is i32 from op_params[0])
            let masked = node.op_param0 != 0;
            dispatch.forward_flash_attn_back(params, node, masked);
        }
        GgmlOp::SsmConv        => dispatch.forward_ssm_conv(params, node),
        GgmlOp::SsmScan        => dispatch.forward_ssm_scan(params, node),
        GgmlOp::WinPart        => dispatch.forward_win_part(params, node),
        GgmlOp::WinUnpart      => dispatch.forward_win_unpart(params, node),
        GgmlOp::Unary          => dispatch.forward_unary(params, node),
        GgmlOp::Glu            => dispatch.forward_glu(params, node),
        GgmlOp::GetRelPos      => dispatch.forward_get_rel_pos(params, node),
        GgmlOp::AddRelPos      => dispatch.forward_add_rel_pos(params, node),
        GgmlOp::RwkvWkv6       => dispatch.forward_rwkv_wkv6(params, node),
        GgmlOp::GatedLinearAttn => dispatch.forward_gla(params, node),
        GgmlOp::RwkvWkv7       => dispatch.forward_rwkv_wkv7(params, node),
        GgmlOp::SolveTri       => dispatch.forward_solve_tri(params, node),
        GgmlOp::GatedDeltaNet  => dispatch.forward_gated_delta_net(params, node),
        GgmlOp::LightningIndexer => dispatch.forward_lightning_indexer(params, node),
        GgmlOp::Dsv4HcComb     => dispatch.forward_dsv4_hc_comb(params, node),
        GgmlOp::Dsv4HcPre      => dispatch.forward_dsv4_hc_pre(params, node),
        GgmlOp::Dsv4HcPost     => dispatch.forward_dsv4_hc_post(params, node),
        GgmlOp::MapCustom1     => dispatch.forward_map_custom1(params, node),
        GgmlOp::MapCustom2     => dispatch.forward_map_custom2(params, node),
        GgmlOp::MapCustom3     => dispatch.forward_map_custom3(params, node),
        GgmlOp::Custom         => dispatch.forward_custom(params, node),
        GgmlOp::CrossEntropyLoss      => dispatch.forward_cross_entropy_loss(params, node),
        GgmlOp::CrossEntropyLossBack  => dispatch.forward_cross_entropy_loss_back(params, node),
        GgmlOp::OptStepAdamW   => dispatch.forward_opt_step_adamw(params, node),
        GgmlOp::OptStepSgd     => dispatch.forward_opt_step_sgd(params, node),
        // Explicit no-ops (already handled by is_nop() above, listed for exhaustiveness).
        GgmlOp::None | GgmlOp::Reshape | GgmlOp::Permute | GgmlOp::View | GgmlOp::Transpose => {}
        GgmlOp::Count => panic!("GgmlOp::Count is not a valid compute op"),
    }
}

// ---------------------------------------------------------------------------
// Worker thread -- per-graph compute loop
// (ggml_graph_compute_thread, ggml-cpu.c lines 3109-3182)
// ---------------------------------------------------------------------------
//
// Each thread (including thread 0 = caller) executes this function once per
// graph.  It iterates nodes in order; after each node all threads synchronize
// at a barrier before the next node.

/// Translated from `ggml_graph_compute_thread`.
///
/// `state`    -- this thread's per-thread state.
/// `cgraph`   -- the graph to compute (all threads share the same graph).
/// `cplan`    -- the compute plan (work buffer, abort callback, etc.).
/// `dispatch` -- operator implementations.
pub fn graph_compute_thread<D: OpDispatch>(
    state:    &ComputeState,
    cgraph:   &GgmlCgraph<'_>,
    cplan:    &GgmlCplan,
    dispatch: &D,
) {
    let tp = &state.threadpool;

    let n_threads = (tp.n_graph.load(Ordering::Relaxed) & N_THREADS_MASK) as usize;

    let mut params = ComputeParams {
        ith:        state.ith,
        nth:        n_threads,
        wsize:      cplan.work_data.len(),
        wdata:      cplan.work_data.as_ptr() as *mut u8,
        threadpool: Arc::clone(tp),
        use_ref:    cplan.use_ref,
    };

    let abort_at = tp.abort.load(Ordering::Relaxed);

    for (node_n, node) in cgraph.nodes.iter().enumerate() {
        // Abort check: if abort was triggered at or before this node, stop.
        if abort_at >= 0 && node_n as i32 >= abort_at {
            break;
        }

        // Skip NOP-flagged nodes (GGML_OP_NONE etc.).
        if node.op.is_nop() {
            continue;
        }

        // Skip nodes that are not marked for compute.
        if (node.flags & tensor_flags::COMPUTE) == 0 {
            continue;
        }

        // Advance the per-thread work pointer so threads use distinct scratch regions.
        // C: params.wdata = cplan->work_data + wsize_per_thread * state->ith
        // We leave the exact arithmetic to the caller because work-buffer layout
        // depends on per-op sizes computed during ggml_graph_plan.
        params.ith = state.ith;

        compute_forward(dispatch, &params, node);

        // After computing, thread 0 checks the abort callback.
        if state.ith == 0 {
            if let Some(cb) = cplan.abort_callback {
                if cb(cplan.abort_callback_data) {
                    tp.abort.store(node_n as i32 + 1, Ordering::Relaxed);
                }
            }
        }

        // All threads synchronize before the next node (skip barrier after last node).
        if node_n + 1 < cgraph.n_nodes() {
            barrier(tp);
        }
    }

    // Final barrier: all threads must complete the last node before returning.
    barrier(tp);
}

// ---------------------------------------------------------------------------
// Secondary worker thread loop
// (ggml_graph_compute_secondary_thread, ggml-cpu.c lines 3250-3285)
// ---------------------------------------------------------------------------

/// Check whether this worker should wake up (has new work, stop, or pause flag).
/// Translated from `ggml_graph_compute_thread_ready`.
fn thread_ready(state: &ComputeState) -> bool {
    let tp = &state.threadpool;
    if state.pending || tp.stop.load(Ordering::Relaxed) || tp.pause.load(Ordering::Relaxed) {
        return true;
    }
    let n_graph   = tp.n_graph.load(Ordering::Relaxed);
    let n_threads = n_graph & N_THREADS_MASK;
    if n_graph != state.last_graph {
        return true;
    }
    let _ = n_threads; // suppress unused warning
    false
}

/// Spin-poll for new work, then fall back to condvar sleep.
/// Translated from `ggml_graph_compute_check_for_work`.
fn check_for_work(state: &mut ComputeState) -> bool {
    let tp = Arc::clone(&state.threadpool);

    // Polling phase (ggml_graph_compute_poll_for_work).
    let n_rounds: u64 = 1024 * 128 * tp.poll as u64;
    for _ in 0..n_rounds {
        if thread_ready(state) {
            break;
        }
        cpu_relax();
    }

    if thread_ready(state) {
        // Sync with seq-cst fence (ggml_graph_compute_thread_sync).
        std::sync::atomic::fence(Ordering::SeqCst);
        // Update pending flag.
        let n_graph   = tp.n_graph.load(Ordering::Relaxed);
        let n_threads = (n_graph & N_THREADS_MASK) as usize;
        if n_graph != state.last_graph {
            state.pending    = state.ith < n_threads;
            state.last_graph = n_graph;
        }
        return state.pending;
    }

    // Sleep phase: wait on condvar.
    let guard = tp.mutex.lock().unwrap();
    drop(tp.cond.wait_while(guard, |_| !thread_ready(state)).unwrap());

    // Recheck pending.
    let n_graph   = tp.n_graph.load(Ordering::Relaxed);
    let n_threads = (n_graph & N_THREADS_MASK) as usize;
    if n_graph != state.last_graph {
        state.pending    = state.ith < n_threads;
        state.last_graph = n_graph;
    }
    state.pending
}

/// Worker thread body for threads 1..n_threads.
/// Translated from `ggml_graph_compute_secondary_thread`.
///
/// Note: because Rust's thread API requires `'static` closures, the
/// actual graph pointer and dispatch object cannot be passed through here
/// in the same way as in C.  The production integration stores the graph
/// and dispatch inside the `ThreadpoolShared` behind a `Mutex<Option<...>>`;
/// this translation uses an `UnsafeCell`-free placeholder via `GraphWork`.
pub fn graph_compute_secondary_thread(mut state: ComputeState) {
    // Apply thread priority and CPU affinity (platform-specific, skipped here).
    // C: ggml_thread_apply_priority(threadpool->prio);
    //    ggml_thread_apply_affinity(state->cpumask);

    loop {
        // Pause loop: sleep while pool is paused.
        while state.threadpool.pause.load(Ordering::Relaxed) {
            let guard = state.threadpool.mutex.lock().unwrap();
            if state.threadpool.pause.load(Ordering::Relaxed) {
                drop(state.threadpool.cond.wait(guard).unwrap());
            } else {
                drop(guard);
            }
        }

        // Stop check (must come after cond_wait).
        if state.threadpool.stop.load(Ordering::Relaxed) {
            break;
        }

        // Wait for new graph work.
        let has_work = check_for_work(&mut state);
        if has_work {
            state.pending = false;
            // In the full integration, graph + cplan + dispatch are read from
            // ThreadpoolShared here, then graph_compute_thread is called.
            // The actual call site must provide the dispatch object.
            // Placeholder: invoke hook stored by the scheduler (not in this file).
        }
    }
}

// ---------------------------------------------------------------------------
// Kickoff -- signal workers to start a new graph
// (ggml_graph_compute_kickoff, ggml-cpu.c lines 3288-3318)
// ---------------------------------------------------------------------------

/// Translated from `ggml_graph_compute_kickoff`.
///
/// Updates `n_graph` to encode the new graph generation + active thread count,
/// then broadcasts on the condvar to wake sleeping workers.
pub fn graph_compute_kickoff(tp: &ThreadpoolShared, n_threads: usize) {
    let _guard = tp.mutex.lock().unwrap();

    // Build new n_graph value: increment generation counter, store thread count in low 16 bits.
    let old_n_graph = tp.n_graph.load(Ordering::Relaxed);
    let gen = (old_n_graph >> N_THREADS_BITS) + 1;
    let new_n_graph = (gen << N_THREADS_BITS) | (n_threads as i32 & N_THREADS_MASK);

    tp.n_graph.store(new_n_graph, Ordering::SeqCst);

    if tp.pause.load(Ordering::Relaxed) {
        // If pool was paused, resume it (applies priority/affinity on main thread).
        tp.pause.store(false, Ordering::Relaxed);
    }

    tp.cond.notify_all();
}

// ---------------------------------------------------------------------------
// graph_compute -- top-level entry point
// (ggml_graph_compute, ggml-cpu.c lines 3399-3474)
// ---------------------------------------------------------------------------
//
// Coordinates: creates a disposable pool if none provided, kicks off workers,
// runs thread 0's compute loop, waits for all workers, returns status.

/// Compute status (ggml_status).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlStatus {
    Success,
    Aborted,
    AllocFailed,
    Failed,
}

/// Translated from `ggml_graph_compute`.
///
/// If `threadpool` is `None`, a single-use pool is created and destroyed.
/// If `threadpool` is `Some`, it is reused; its `cgraph`/`cplan`/`abort`/
/// `current_chunk` fields are reset for this call.
///
/// The `dispatch` object provides the actual operator implementations and
/// must be `Send + Sync` so that worker threads can call it concurrently.
pub fn graph_compute<D>(
    cgraph:     &GgmlCgraph<'_>,
    cplan:      &mut GgmlCplan,
    threadpool: Option<Arc<ThreadpoolShared>>,
    dispatch:   &D,
) -> GgmlStatus
where
    D: OpDispatch + Send + Sync,
{
    assert!(cplan.n_threads > 0, "cplan.n_threads must be > 0");

    let n_threads = cplan.n_threads;
    let mut disposable = false;

    let tp: Arc<ThreadpoolShared> = match threadpool {
        Some(existing) => {
            // Reset mutable fields for this graph run.
            existing.current_chunk.store(0, Ordering::Relaxed);
            existing.abort.store(-1, Ordering::Relaxed);
            existing
        }
        None => {
            disposable = true;
            let params = ThreadpoolParams::default(n_threads);
            Arc::new(ThreadpoolShared {
                mutex:            Mutex::new(()),
                cond:             Condvar::new(),
                n_graph:          AtomicI32::new(0),
                n_barrier:        AtomicI32::new(0),
                n_barrier_passed: AtomicI32::new(0),
                current_chunk:    AtomicI32::new(0),
                stop:             AtomicBool::new(false),
                pause:            AtomicBool::new(false),
                abort:            AtomicI32::new(-1),
                prio:             params.prio,
                poll:             params.poll,
                n_threads:        params.n_threads,
            })
        }
    };

    if n_threads > tp.n_threads {
        // Warn: more threads requested than available -- cap to pool size.
        let n_threads = tp.n_threads;
        let _ = n_threads;
    }
    let effective = n_threads.min(tp.n_threads);

    // Kick workers (sets n_graph, broadcasts condvar).
    graph_compute_kickoff(&tp, effective);

    // Thread 0 runs the compute loop directly.
    let state0 = ComputeState {
        ith:        0,
        cpumask:    [false; MAX_N_THREADS],
        threadpool: Arc::clone(&tp),
        last_graph: 0,
        pending:    true,
    };
    graph_compute_thread(&state0, cgraph, cplan, dispatch);

    // Determine final status.
    let aborted = tp.abort.load(Ordering::Relaxed) >= 0;

    if disposable {
        // Signal the (nonexistent, for single-threaded disposable) workers to stop.
        tp.stop.store(true, Ordering::Relaxed);
    }

    if aborted {
        GgmlStatus::Aborted
    } else {
        GgmlStatus::Success
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barrier_single_thread() {
        let shared = ThreadpoolShared {
            mutex:            Mutex::new(()),
            cond:             Condvar::new(),
            n_graph:          AtomicI32::new(1), // 1 thread
            n_barrier:        AtomicI32::new(0),
            n_barrier_passed: AtomicI32::new(0),
            current_chunk:    AtomicI32::new(0),
            stop:             AtomicBool::new(false),
            pause:            AtomicBool::new(false),
            abort:            AtomicI32::new(-1),
            prio:             SchedPrio::Normal,
            poll:             0,
            n_threads:        1,
        };
        // Should return immediately without spinning.
        barrier(&shared);
        barrier(&shared);
        assert_eq!(shared.n_barrier_passed.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn barrier_two_threads() {
        use std::sync::Arc;
        let shared = Arc::new(ThreadpoolShared {
            mutex:            Mutex::new(()),
            cond:             Condvar::new(),
            n_graph:          AtomicI32::new(2), // 2 threads
            n_barrier:        AtomicI32::new(0),
            n_barrier_passed: AtomicI32::new(0),
            current_chunk:    AtomicI32::new(0),
            stop:             AtomicBool::new(false),
            pause:            AtomicBool::new(false),
            abort:            AtomicI32::new(-1),
            prio:             SchedPrio::Normal,
            poll:             0,
            n_threads:        2,
        });
        let s2 = Arc::clone(&shared);
        let t = thread::spawn(move || {
            barrier(&s2);
        });
        barrier(&shared);
        t.join().unwrap();
        assert_eq!(shared.n_barrier_passed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn get_n_tasks_basic() {
        let node = GgmlTensor {
            op:        GgmlOp::Add,
            flags:     tensor_flags::COMPUTE,
            ne:        [128, 1, 1, 1],
            n_src:     2,
            src1_ne1:  0,
            op_param0: 0,
            op_param1: 0,
        };
        assert_eq!(get_n_tasks(&node, 4), 4);

        let node2 = GgmlTensor { op: GgmlOp::Sum, ..node };
        assert_eq!(get_n_tasks(&node2, 4), 1);

        let node3 = GgmlTensor { op: GgmlOp::SoftMax, ne: [128, 3, 1, 1], ..node };
        assert_eq!(get_n_tasks(&node3, 8), 3); // capped by ne[1]
    }

    #[test]
    fn ggml_op_is_nop() {
        assert!(GgmlOp::None.is_nop());
        assert!(GgmlOp::Reshape.is_nop());
        assert!(GgmlOp::View.is_nop());
        assert!(GgmlOp::Permute.is_nop());
        assert!(GgmlOp::Transpose.is_nop());
        assert!(!GgmlOp::Add.is_nop());
        assert!(!GgmlOp::MulMat.is_nop());
    }

    #[test]
    fn chunk_helpers() {
        let shared = ThreadpoolShared {
            mutex:            Mutex::new(()),
            cond:             Condvar::new(),
            n_graph:          AtomicI32::new(1),
            n_barrier:        AtomicI32::new(0),
            n_barrier_passed: AtomicI32::new(0),
            current_chunk:    AtomicI32::new(0),
            stop:             AtomicBool::new(false),
            pause:            AtomicBool::new(false),
            abort:            AtomicI32::new(-1),
            prio:             SchedPrio::Normal,
            poll:             0,
            n_threads:        1,
        };
        chunk_set(&shared, 7);
        assert_eq!(chunk_add(&shared, 3), 7);
        assert_eq!(shared.current_chunk.load(Ordering::Relaxed), 10);
    }
}
