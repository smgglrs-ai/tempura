#![allow(dead_code, unused_imports)]
mod api;
mod batch;
mod sse;
mod trace_store;

use clap::Parser;
use oxide_engine::pipeline::KernelPipeline;
use oxide_engine::tokenizer::Tokenizer;
use oxide_engine::{ModelConfig, TransformerModel};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "oxide-serve", about = "Rust-native LLM inference server")]
struct Args {
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    #[arg(long, default_value_t = 8080)]
    port: u16,

    #[arg(long, help = "Path to model weights (safetensors)")]
    model: Option<String>,

    #[arg(long, help = "Path to tokenizer.json")]
    tokenizer: Option<String>,

    #[arg(long, default_value_t = 32, help = "Maximum batch size")]
    max_batch_size: usize,

    #[arg(long, default_value_t = 4096, help = "Maximum sequence length")]
    max_seq_len: usize,

    #[arg(long, default_value_t = 100, help = "Maximum stored traces")]
    max_traces: usize,
}

pub type CpuModel = TransformerModel<
    oxide_kernels::attention::CpuAttention,
    oxide_kernels::rmsnorm::CpuRMSNorm,
    oxide_kernels::rope::CpuRoPE,
    oxide_kernels::matmul::CpuMatMul,
    oxide_kernels::sampling::CpuSampler,
>;

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<CpuModel>,
    pub tokenizer: Option<Arc<Tokenizer>>,
    pub model_id: String,
    pub scheduler: Arc<Mutex<batch::BatchScheduler>>,
    pub trace_store: Arc<Mutex<trace_store::TraceStore>>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let config = ModelConfig::llama2_7b();
    let pipeline = KernelPipeline::cpu_verified();
    let mut model = TransformerModel::new(config, pipeline);

    let model_id = if let Some(ref model_path) = args.model {
        let loader =
            oxide_engine::loader::SafetensorsLoader::from_file(Path::new(model_path)).unwrap();
        model.load_weights(&loader).unwrap();
        tracing::info!("Loaded model weights from {}", model_path);
        model_path.clone()
    } else {
        tracing::warn!("No --model provided, running with unloaded weights (stub responses)");
        "oxide-serve-unloaded".to_string()
    };

    let tokenizer = args.tokenizer.as_ref().map(|path| {
        let tok = Tokenizer::from_file(Path::new(path)).unwrap();
        tracing::info!("Loaded tokenizer from {}", path);
        Arc::new(tok)
    });

    let scheduler = batch::BatchScheduler::new(args.max_batch_size, args.max_seq_len);
    let traces = trace_store::TraceStore::new(args.max_traces);
    let state = AppState {
        model: Arc::new(model),
        tokenizer,
        model_id,
        scheduler: Arc::new(Mutex::new(scheduler)),
        trace_store: Arc::new(Mutex::new(traces)),
    };

    let app = api::router()
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Starting oxide-serve on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
