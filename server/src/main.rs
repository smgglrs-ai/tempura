#![allow(dead_code, unused_imports)]
mod api;
mod auth;
mod batch;
pub mod engine_task;
pub mod replay;
mod sse;
mod trace_store;

use clap::Parser;
use metrics_exporter_prometheus::PrometheusHandle;
use std::path::Path;
use std::sync::Arc;
use tempura_engine::pipeline::KernelPipeline;
use tempura_engine::tokenizer::Tokenizer;
use tempura_engine::{ModelConfig, TransformerModel};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "tempura-serve", about = "Rust-native LLM inference server")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Start the inference server (default)
    Serve(Box<ServeArgs>),
    /// Download a model from HuggingFace
    Pull {
        /// HuggingFace repo ID (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)
        repo_id: String,
    },
    /// List downloaded models
    List,
    /// Inspect a model (show config, format, architecture)
    Inspect {
        /// Path to model file or directory
        path: String,
    },
}

#[derive(clap::Args, Debug)]
struct ServeArgs {
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    #[arg(long, default_value_t = 8080)]
    port: u16,

    #[arg(long, help = "Path to model weights (safetensors or GGUF)")]
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

    #[arg(long, default_value = "cpu", help = "Device: cpu or cuda")]
    device: String,

    #[arg(
        long,
        default_value = "auto",
        help = "Attention mode: auto, naive-f32, naive-f16, flash-f32, flash-f16"
    )]
    attn_mode: String,

    #[arg(
        long,
        help = "API key required on all non-health requests (overrides TEMPURA_API_KEY env var)"
    )]
    api_key: Option<String>,
}

pub type CpuModel = TransformerModel<
    tempura_kernels::attention::CpuAttention,
    tempura_kernels::rmsnorm::CpuRMSNorm,
    tempura_kernels::rope::CpuRoPE,
    tempura_kernels::matmul::CpuMatMul,
    tempura_kernels::sampling::CpuSampler,
>;

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<dyn tempura_engine::InferenceModel>,
    pub tokenizer: Option<Arc<Tokenizer>>,
    pub model_id: String,
    pub scheduler: Arc<Mutex<batch::BatchScheduler>>,
    pub trace_store: Arc<Mutex<trace_store::TraceStore>>,
    pub metrics_handle: PrometheusHandle,
    pub request_timeout: std::time::Duration,
    pub api_key: Option<String>,
    pub engine_tx: Arc<tokio::sync::mpsc::Sender<engine_task::EngineRequest>>,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Pull { repo_id }) => {
            tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::from_default_env())
                .init();
            let store = tempura_model::store::ModelStore::open();
            match tempura_model::hub::pull(&store, &repo_id) {
                Ok(entry) => {
                    println!("Downloaded {} to {}", entry.repo_id, entry.path.display());
                    println!(
                        "  Format: {}, Size: {:.1} MB",
                        entry.format,
                        entry.size_bytes as f64 / 1_048_576.0
                    );
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
            return;
        }
        Some(Command::List) => {
            let store = tempura_model::store::ModelStore::open();
            let models = store.list();
            if models.is_empty() {
                println!(
                    "No models installed. Use `tempura-serve pull <repo-id>` to download one."
                );
            } else {
                println!("{:<40} {:<15} {:>10}", "NAME", "FORMAT", "SIZE");
                println!("{}", "-".repeat(67));
                for m in &models {
                    let size_mb = m.size_bytes as f64 / 1_048_576.0;
                    let size_str = if size_mb > 1024.0 {
                        format!("{:.1} GB", size_mb / 1024.0)
                    } else {
                        format!("{:.0} MB", size_mb)
                    };
                    println!("{:<40} {:<15} {:>10}", m.name, m.format, size_str);
                }
            }
            return;
        }
        Some(Command::Inspect { path }) => {
            match tempura_model::inspect::inspect(Path::new(&path)) {
                Ok(info) => print!("{info}"),
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
            return;
        }
        Some(Command::Serve(args)) => {
            run_server(*args).await;
        }
        None => {
            run_server(ServeArgs {
                host: "0.0.0.0".into(),
                port: 8080,
                model: None,
                tokenizer: None,
                model_config: "llama2-7b".into(),
                max_batch_size: 32,
                max_seq_len: 4096,
                max_traces: 100,
                shutdown_timeout: 30,
                request_timeout_secs: 300,
                otlp_endpoint: None,
                device: "cpu".into(),
                attn_mode: "auto".into(),
                api_key: None,
            })
            .await;
        }
    }
}

async fn run_server(args: ServeArgs) {
    #[cfg(feature = "openmp")]
    {
        // Allow GOMP_SPINCOUNT to be overridden; default 1000 lets threads spin
        // between the ~150 #pragma omp parallel for regions per decode token,
        // eliminating condvar wake-up latency (~0.3ms/barrier → ~0.05ms/barrier).
        if std::env::var("GOMP_SPINCOUNT").is_err() {
            std::env::set_var("GOMP_SPINCOUNT", "1000");
        }
    }
    init_tracing(&args.otlp_endpoint);

    let api_key = args
        .api_key
        .clone()
        .or_else(|| std::env::var("TEMPURA_API_KEY").ok());
    if api_key.is_some() {
        tracing::info!("API key authentication enabled");
    }

    let use_gpu = (args.device == "cuda" || args.device == "cutile") && cfg!(feature = "cutile");
    if (args.device == "cuda" || args.device == "cutile") && !cfg!(feature = "cutile") {
        eprintln!(
            "Error: --device {} requires building with --features cutile (or cuda)",
            args.device
        );
        std::process::exit(1);
    }
    tracing::info!(
        "Using {} kernels",
        if use_gpu {
            "CUDA GPU (cuTile)"
        } else {
            "CPU reference"
        }
    );

    let model_path = args.model.as_deref().map(Path::new);
    let is_model_dir = model_path.is_some_and(|p| p.is_dir());
    let is_gguf = model_path.is_some_and(|p| {
        if p.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            return true;
        }
        if p.is_file() {
            if let Ok(mut f) = std::fs::File::open(p) {
                let mut magic = [0u8; 4];
                use std::io::Read;
                if f.read_exact(&mut magic).is_ok() {
                    return &magic == b"GGUF";
                }
            }
        }
        false
    });

    let build_model = |config: &ModelConfig,
                       loader: &dyn tempura_engine::loader::WeightLoader|
     -> Arc<dyn tempura_engine::InferenceModel> {
        #[cfg(feature = "cutile")]
        if use_gpu {
            use tempura_engine::InferenceModel;
            let mut wrapper = tempura_engine::CutileInferenceWrapper::new(config.clone());
            wrapper.load_weights(loader).unwrap();
            tracing::info!("cuTile GPU inference ready");
            return Arc::new(wrapper);
        }
        let pipeline = KernelPipeline::cpu_native();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        model.load_weights(loader).unwrap();
        Arc::new(model)
    };

    let (config, model_id) = if is_gguf {
        let model_p = model_path.unwrap();
        let gguf = tempura_engine::gguf_loader::GgufLoader::from_file_quantized(model_p).unwrap();
        let mut config = gguf.config.clone();
        let metadata = gguf.metadata.clone();
        tracing::info!("Loaded GGUF model from {}", model_p.display());

        let tokenizer = if let Some(path) = args.tokenizer.as_ref() {
            let tok = Tokenizer::from_file(Path::new(path)).unwrap();
            tracing::info!("Loaded tokenizer from {}", path);
            Some(Arc::new(tok))
        } else {
            match Tokenizer::from_gguf_metadata(&metadata) {
                Ok(tok) => {
                    tracing::info!("Extracted tokenizer from GGUF metadata");
                    Some(Arc::new(tok))
                }
                Err(e) => {
                    tracing::warn!("Could not extract tokenizer from GGUF metadata: {e}");
                    None
                }
            }
        };

        let qweights = gguf.to_quantized_weights();
        config.max_seq_len = config.max_seq_len.min(args.max_seq_len);
        tracing::info!("Using max_seq_len={}", config.max_seq_len);

        #[allow(unused_mut)]
        let mut model: Arc<dyn tempura_engine::InferenceModel> = {
            #[cfg(feature = "cutile")]
            if use_gpu {
                let mut wrapper = tempura_engine::CutileInferenceWrapper::new(config.clone());
                wrapper.model.quantized_weights = Some(qweights);
                tracing::info!("cuTile GPU quantized inference ready");
                Arc::new(wrapper)
            } else {
                let pipeline = KernelPipeline::cpu_native();
                let mut model = TransformerModel::new(config.clone(), pipeline);
                model.quantized_weights = Some(qweights);
                tracing::info!("Using CPU quantized weights");
                Arc::new(model)
            }

            #[cfg(not(feature = "cutile"))]
            {
                let pipeline = KernelPipeline::cpu_native();
                let mut model = TransformerModel::new(config.clone(), pipeline);
                model.quantized_weights = Some(qweights);
                tracing::info!("Using CPU quantized weights");
                Arc::new(model)
            }
        };

        let prometheus_builder = metrics_exporter_prometheus::PrometheusBuilder::new();
        let metrics_handle = prometheus_builder.install_recorder().unwrap();

        let scheduler =
            batch::BatchScheduler::new(args.max_batch_size, args.max_seq_len, args.max_batch_size);
        let traces = trace_store::TraceStore::new(args.max_traces);

        let (engine_tx, engine_rx) =
            tokio::sync::mpsc::channel::<engine_task::EngineRequest>(args.max_batch_size * 4);
        let engine_model = model.clone();
        std::thread::Builder::new()
            .name("tempura-engine".to_string())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                engine_task::EngineTask::new(engine_model, engine_rx).run();
            })
            .expect("failed to spawn engine thread");

        let state = AppState {
            model,
            tokenizer,
            model_id: model_p.display().to_string(),
            scheduler: Arc::new(Mutex::new(scheduler)),
            trace_store: Arc::new(Mutex::new(traces)),
            metrics_handle,
            request_timeout: std::time::Duration::from_secs(args.request_timeout_secs),
            api_key: api_key.clone(),
            engine_tx: Arc::new(engine_tx),
        };

        let app = api::router()
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                auth::auth_middleware,
            ))
            .layer(CorsLayer::permissive())
            .with_state(state);

        let addr = format!("{}:{}", args.host, args.port);
        tracing::info!("Starting tempura-serve on {}", addr);

        let shutdown_timeout = std::time::Duration::from_secs(args.shutdown_timeout);
        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal(shutdown_timeout))
            .await
            .unwrap();
        return;
    } else if is_model_dir {
        let config_path = model_path.unwrap().join("config.json");
        let config = if config_path.exists() {
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
        };
        (config, model_path.unwrap().display().to_string())
    } else if let Some(model_p) = model_path {
        let config = match args.model_config.as_str() {
            "micro-llama" => ModelConfig::micro_llama(),
            "tiny-test" => ModelConfig::tiny_test(),
            _ => ModelConfig::llama2_7b(),
        };
        (config, model_p.display().to_string())
    } else {
        let config = match args.model_config.as_str() {
            "micro-llama" => ModelConfig::micro_llama(),
            "tiny-test" => ModelConfig::tiny_test(),
            _ => ModelConfig::llama2_7b(),
        };
        tracing::warn!("No --model provided, running with unloaded weights (stub responses)");
        (config, "tempura-serve-unloaded".to_string())
    };

    let model: Arc<dyn tempura_engine::InferenceModel> = if let Some(model_p) = model_path {
        if is_model_dir {
            let loader = tempura_engine::loader::SafetensorsLoader::from_dir(model_p).unwrap();
            tracing::info!("Loaded model weights from {}", model_p.display());
            build_model(&config, &loader)
        } else {
            let loader = tempura_engine::loader::SafetensorsLoader::from_file(model_p).unwrap();
            tracing::info!("Loaded model weights from {}", model_p.display());
            build_model(&config, &loader)
        }
    } else {
        let pipeline = KernelPipeline::cpu_native();
        Arc::new(TransformerModel::new(config.clone(), pipeline))
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

    let (engine_tx, engine_rx) =
        tokio::sync::mpsc::channel::<engine_task::EngineRequest>(args.max_batch_size * 4);
    let engine_model = model.clone();
    std::thread::Builder::new()
        .name("tempura-engine".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            engine_task::EngineTask::new(engine_model, engine_rx).run();
        })
        .expect("failed to spawn engine thread");

    let state = AppState {
        model,
        tokenizer,
        model_id,
        scheduler: Arc::new(Mutex::new(scheduler)),
        trace_store: Arc::new(Mutex::new(traces)),
        metrics_handle,
        request_timeout: std::time::Duration::from_secs(args.request_timeout_secs),
        api_key,
        engine_tx: Arc::new(engine_tx),
    };

    let app = api::router()
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Starting tempura-serve on {}", addr);

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
                    opentelemetry::KeyValue::new("service.name", "tempura-serve"),
                    opentelemetry::KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                ]))
                .build();

            let tracer = provider.tracer("tempura-serve");
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
