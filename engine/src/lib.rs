pub mod architecture;
pub mod batch_invariant;
pub mod config;
#[cfg(feature = "cutile")]
pub mod cutile_gpu_forward;
#[cfg(feature = "cutile")]
pub mod cutile_kv_cache;
pub mod deltanet_state;
pub mod disaggregated;
pub mod dual_pipeline;
pub mod gguf_loader;
pub mod kv_cache;
pub mod loader;
pub mod lora;
pub mod model;
pub mod moe;
pub mod multi_token;
pub mod pipeline;
pub mod prefix_cache;
pub mod schema;
pub mod session;
pub mod speculative;
pub mod tokenizer;
pub mod tool_call;
pub mod trace;
pub mod trace_diff;
pub mod weights;

pub use config::{ConfigError, ModelConfig, QuantConfig, QuantFormat};
#[cfg(feature = "cutile")]
pub use cutile_gpu_forward::CutileGpuModel;
pub use kv_cache::{
    BlockTable, Fp8PagedKvCache, KvCache, KvCachePool, PageAllocator, PagedKvCache, PoolError,
    PrefixCacheTable, PrefixMatch, SharedPrefixCache, DEFAULT_BLOCK_SIZE,
};
#[cfg(feature = "cutile")]
pub use model::CutileInferenceWrapper;
pub use model::{InferenceModel, TokenLogprob, TopLogprob, TransformerModel};
pub use session::InferenceSession;
pub use tokenizer::{ChatMessageInput, ChatTemplate};
pub use trace::{
    EdgeMetrics, InferenceTrace, LayerSnapshot, NormStats, SamplingSnapshot, TimingInfo,
    TraceDecodeError, TraceLevel, TRACE_FORMAT_VERSION, TRACE_MAGIC,
};
pub use trace_diff::{diff_traces, DivergenceLevel, TraceDiff};
pub use weights::{ModelWeights, QuantizedWeights};
