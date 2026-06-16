#![allow(dead_code, unused_imports)]
mod api;
mod batch;
pub mod replay;
mod sse;
mod trace_store;

use clap::Parser;
use metrics_exporter_prometheus::PrometheusHandle;
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

    #[arg(
        long,
        default_value = "llama2-7b",
        help = "Model config preset: llama2-7b, micro-llama, tiny-test"
    )]
    model_config: String,

    #[arg(long, default_value_t = 32, help = "Maximum batch size")]
    max_batch_size: usize,

    #[arg(long, default_value_t = 4096, help = "Maximum sequence length")]
    max_seq_len: usize,

    #[arg(long, default_value_t = 100, help = "Maximum stored traces")]
    max_traces: usize,

    #[arg(
        long,
        default_value_t = 30,
        help = "Graceful shutdown timeout in seconds"
    )]
    shutdown_timeout: u64,

    #[arg(
        long,
        default_value_t = 300,
        help = "Default request timeout in seconds"
    )]
    request_timeout_secs: u64,

    #[arg(
        long,
        help = "OTLP gRPC endpoint for OpenTelemetry export (e.g. http://localhost:4317)"
    )]
    otlp_endpoint: Option<String>,
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
    pub metrics_handle: PrometheusHandle,
    pub request_timeout: std::time::Duration,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    init_tracing(&args.otlp_endpoint);

    let model_path = args.model.as_deref().map(Path::new);
    let is_model_dir = model_path.is_some_and(|p| p.is_dir());

    let config = if is_model_dir {
        let config_path = model_path.unwrap().join("config.json");
        if config_path.exists() {
            let c = ModelConfig::from_hf_config(&config_path).unwrap();
            tracing::info!("Loaded model config from {}", config_path.display());
            c
        } else {
            tracing::warn!("No config.json in model directory, using preset");
            match args.model_config.as_str() {
                "micro-llama" => ModelConfig::micro_llama(),
                "tiny-test" => ModelConfig::tiny_test(),
                _ => ModelConfig::llama2_7b(),
            }
        }
    } else {
        match args.model_config.as_str() {
            "micro-llama" => ModelConfig::micro_llama(),
            "tiny-test" => ModelConfig::tiny_test(),
            _ => ModelConfig::llama2_7b(),
        }
    };
    let pipeline = KernelPipeline::cpu_verified();
    let mut model = TransformerModel::new(config, pipeline);

    let model_id = if let Some(model_p) = model_path {
        if is_model_dir {
            let safetensors: Vec<_> = std::fs::read_dir(model_p)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
                .collect();
            if let Some(entry) = safetensors.first() {
                let loader =
                    oxide_engine::loader::SafetensorsLoader::from_file(&entry.path()).unwrap();
                model.load_weights(&loader).unwrap();
                tracing::info!("Loaded model weights from {}", entry.path().display());
            }
            model_p.display().to_string()
        } else {
            let loader = oxide_engine::loader::SafetensorsLoader::from_file(model_p).unwrap();
            model.load_weights(&loader).unwrap();
            tracing::info!("Loaded model weights from {}", model_p.display());
            model_p.display().to_string()
        }
    } else {
        tracing::warn!("No --model provided, running with unloaded weights (stub responses)");
        "oxide-serve-unloaded".to_string()
    };

    let tokenizer = if is_model_dir {
        let tok_path = model_path.unwrap().join("tokenizer.json");
        if tok_path.exists() {
            let tok = Tokenizer::from_file(&tok_path).unwrap();
            tracing::info!("Loaded tokenizer from {}", tok_path.display());
            Some(Arc::new(tok))
        } else {
            args.tokenizer.as_ref().map(|path| {
                let tok = Tokenizer::from_file(Path::new(path)).unwrap();
                tracing::info!("Loaded tokenizer from {}", path);
                Arc::new(tok)
            })
        }
    } else {
        args.tokenizer.as_ref().map(|path| {
            let tok = Tokenizer::from_file(Path::new(path)).unwrap();
            tracing::info!("Loaded tokenizer from {}", path);
            Arc::new(tok)
        })
    };

    let prometheus_builder = metrics_exporter_prometheus::PrometheusBuilder::new();
    let metrics_handle = prometheus_builder.install_recorder().unwrap();

    let scheduler =
        batch::BatchScheduler::new(args.max_batch_size, args.max_seq_len, args.max_batch_size);
    let traces = trace_store::TraceStore::new(args.max_traces);
    let state = AppState {
        model: Arc::new(model),
        tokenizer,
        model_id,
        scheduler: Arc::new(Mutex::new(scheduler)),
        trace_store: Arc::new(Mutex::new(traces)),
        metrics_handle,
        request_timeout: std::time::Duration::from_secs(args.request_timeout_secs),
    };

    let app = api::router()
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Starting oxide-serve on {}", addr);

    let shutdown_timeout = std::time::Duration::from_secs(args.shutdown_timeout);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown_timeout))
        .await
        .unwrap();
}

async fn shutdown_signal(timeout: std::time::Duration) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install SIGINT handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received SIGINT, starting graceful shutdown (timeout: {}s)", timeout.as_secs());
        }
        _ = terminate => {
            tracing::info!("Received SIGTERM, starting graceful shutdown (timeout: {}s)", timeout.as_secs());
        }
    }
}

fn init_tracing(_otlp_endpoint: &Option<String>) {
    #[cfg(feature = "otlp")]
    {
        if let Some(endpoint) = _otlp_endpoint {
            use opentelemetry::trace::TracerProvider;
            use opentelemetry_otlp::WithExportConfig;
            use tracing_subscriber::layer::SubscriberExt;

            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_tonic()
                .with_endpoint(endpoint)
                .build()
                .expect("failed to create OTLP exporter");

            let provider = opentelemetry_sdk::trace::TracerProvider::builder()
                .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
                .with_resource(opentelemetry_sdk::Resource::new(vec![
                    opentelemetry::KeyValue::new("service.name", "oxide-serve"),
                    opentelemetry::KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                ]))
                .build();

            let tracer = provider.tracer("oxide-serve");
            let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

            let subscriber = tracing_subscriber::Registry::default()
                .with(EnvFilter::from_default_env())
                .with(tracing_subscriber::fmt::layer())
                .with(otel_layer);

            tracing::subscriber::set_global_default(subscriber)
                .expect("failed to set global subscriber");

            tracing::info!("OpenTelemetry enabled, exporting to {}", endpoint);
            return;
        }
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
}
