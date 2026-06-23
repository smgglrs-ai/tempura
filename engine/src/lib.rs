pub mod architecture;
pub mod batch_invariant;
pub mod config;
pub mod disaggregated;
pub mod dual_pipeline;
pub mod gguf_loader;
#[cfg(feature = "cuda")]
pub mod gpu_scratchpad;
pub mod kv_cache;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod schema;
pub mod session;
pub mod speculative;
pub mod tokenizer;
pub mod tool_call;
pub mod trace;
pub mod trace_diff;
pub mod weights;

pub use config::{ConfigError, ModelConfig, QuantConfig, QuantFormat};
#[cfg(feature = "cuda")]
pub use gpu_scratchpad::GpuScratchpad;
#[cfg(feature = "cuda")]
pub use kv_cache::GpuKvCache;
pub use kv_cache::{
    BlockTable, Fp8PagedKvCache, KvCache, KvCachePool, PageAllocator, PagedKvCache, PoolError,
    PrefixCacheTable, PrefixMatch, SharedPrefixCache, DEFAULT_BLOCK_SIZE,
};
#[cfg(feature = "cuda")]
pub use model::GpuInferenceWrapper;
#[cfg(feature = "cuda")]
pub use model::GpuQuantizedInferenceWrapper;
pub use model::{InferenceModel, TokenLogprob, TopLogprob, TransformerModel};
pub use session::InferenceSession;
pub use tokenizer::{ChatMessageInput, ChatTemplate};
pub use trace::{
    InferenceTrace, LayerSnapshot, NormStats, SamplingSnapshot, TimingInfo, TraceDecodeError,
    TraceLevel, TRACE_FORMAT_VERSION, TRACE_MAGIC,
};
pub use trace_diff::{diff_traces, DivergenceLevel, TraceDiff};
#[cfg(feature = "cuda")]
pub use weights::GpuWeights;
pub use weights::{ModelWeights, QuantizedWeights};
