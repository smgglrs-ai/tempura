#![allow(dead_code, unused_imports)]
mod api;
mod batch;
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
    Serve(ServeArgs),
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
}

pub type CpuModel = TransformerModel<
    tempura_kernels::attention::CpuAttention,
    tempura_kernels::rmsnorm::CpuRMSNorm,
    tempura_kernels::rope::CpuRoPE,
    tempura_kernels::matmul::CpuMatMul,
    tempura_kernels::sampling::CpuSampler,
>;

#[cfg(feature = "cuda")]
pub type CudaModel = TransformerModel<
    tempura_kernels::attention::CudaAttention,
    tempura_kernels::rmsnorm::CudaRMSNorm,
    tempura_kernels::rope::CudaRoPE,
    tempura_kernels::matmul::CudaMatMul,
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
            run_server(args).await;
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
            })
            .await;
        }
    }
}

async fn run_server(args: ServeArgs) {
    init_tracing(&args.otlp_endpoint);

    if args.device == "cuda" && !cfg!(feature = "cuda") {
        eprintln!("Error: --device cuda requires building with --features cuda");
        std::process::exit(1);
    }
    let use_cuda = args.device == "cuda" && cfg!(feature = "cuda");
    tracing::info!(
        "Using {} kernels",
        if use_cuda {
            "CUDA GPU"
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
        #[cfg(feature = "cuda")]
        if use_cuda {
            let pipeline = KernelPipeline::cuda();
            let mut model = TransformerModel::new(config.clone(), pipeline);
            model.load_weights(loader).unwrap();

            let ctx = model.pipeline.matmul.gpu().context().clone();
            let gpu_weights = tempura_engine::GpuWeights::upload_from_with_concat(
                model.weights.as_ref().unwrap(),
                config,
                &ctx,
            )
            .expect("GPU weight upload failed");
            ctx.bind_to_thread().unwrap();
            let stream = ctx.default_stream();
            let gpu_cache = tempura_engine::GpuKvCache::new(config, &stream)
                .expect("GPU KV cache allocation failed");

            let scratchpad = tempura_engine::GpuScratchpad::new(config, &stream)
                .expect("GPU scratchpad allocation failed");

            tracing::info!("Uploaded weights and KV cache to GPU");
            return Arc::new(tempura_engine::GpuInferenceWrapper {
                model,
                gpu_weights,
                gpu_cache: std::sync::Mutex::new(gpu_cache),
                scratchpad: std::sync::Mutex::new(scratchpad),
            });
        }
        let pipeline = KernelPipeline::cpu_verified();
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

        let q2_shapes = gguf.q2_tensor_shapes();
        let mut qweights = gguf.to_quantized_weights();
        if !q2_shapes.is_empty() && std::env::var("TEMPURA_NO_TL2").is_err() {
            eprintln!("[Q2_0] Repacking {} tensors to TL2 (LUT)...", q2_shapes.len());
            let t0 = std::time::Instant::now();
            qweights.build_q2_tl2(&q2_shapes);
            eprintln!("[Q2_0] TL2 repack done in {:.1}s", t0.elapsed().as_secs_f64());
        }
        config.max_seq_len = config.max_seq_len.min(args.max_seq_len);
        tracing::info!("Using max_seq_len={}", config.max_seq_len);

        #[allow(unused_mut)]
        let mut model: Arc<dyn tempura_engine::InferenceModel> = {
            #[cfg(feature = "cuda")]
            if use_cuda {
                let q4k_matmul = tempura_kernels::quantized::CudaMatMulQ4K::new().unwrap();
                let ctx = q4k_matmul.gpu().context().clone();
                ctx.bind_to_thread().unwrap();

                let attention =
                    tempura_kernels::attention::CudaAttention::with_context(ctx.clone()).unwrap();
                let mut rmsnorm =
                    tempura_kernels::rmsnorm::CudaRMSNorm::with_context(ctx.clone()).unwrap();
                let mut rope =
                    tempura_kernels::rope::CudaRoPE::with_context(ctx.clone()).unwrap();
                let matmul =
                    tempura_kernels::matmul::CudaMatMul::with_context(ctx.clone()).unwrap();
                eprintln!("[gpu] uploading quantized weights...");
                let gpu_weights =
                    tempura_engine::GpuWeights::upload_from_quantized(&qweights, &ctx)
                        .expect("GPU quantized weight upload failed");
                eprintln!("[gpu] weights uploaded: {} q + {} f32", gpu_weights.quantized_count(), gpu_weights.f32_count());

                ctx.bind_to_thread().unwrap();
                let stream_ref = ctx.default_stream();
                eprintln!("[gpu] allocating KV cache (max_seq_len={})...", config.max_seq_len);
                let gpu_cache = tempura_engine::GpuKvCache::new(&config, &stream_ref)
                    .expect("GPU KV cache allocation failed");
                eprintln!("[gpu] KV cache allocated, allocating scratchpad...");
                let scratchpad = tempura_engine::GpuScratchpad::new(&config, &stream_ref)
                    .expect("GPU scratchpad allocation failed");
                eprintln!("[gpu] scratchpad allocated");

                let embed = qweights
                    .get_f32("model.embed_tokens.weight")
                    .unwrap()
                    .to_vec();
                tracing::info!(
                    "Uploaded quantized weights to GPU ({} Q4K + {} Q6K + {} f32 tensors)",
                    gpu_weights.quantized_count(),
                    0, // Q6K counted in quantized_count
                    gpu_weights.f32_count(),
                );

                let attn_mode: tempura_kernels::AttentionMode = args.attn_mode.parse()
                    .unwrap_or_else(|e| { eprintln!("invalid --attn-mode: {e}"); std::process::exit(1); });
                if attn_mode == tempura_kernels::AttentionMode::NaiveF32
                    || std::env::var("TEMPURA_DETERMINISTIC").is_ok()
                {
                    rmsnorm.deterministic = true;
                    rope.deterministic = true;
                    let table = tempura_kernels::rope::build_cos_sin_table(
                        config.max_seq_len, config.head_dim, config.rope_theta,
                    );
                    rope.upload_cos_sin_table(&table).expect("upload RoPE table");
                    eprintln!("[gpu] deterministic RMSNorm + RoPE table enabled");
                }
                eprintln!("[gpu] attention mode: {attn_mode}");
                let cpu_model = if std::env::var("TEMPURA_CPU_OPS").is_ok() {
                    eprintln!("[gpu] CPU_OPS: using CPU forward path for bit-identical output");
                    eprintln!("[gpu] CPU_OPS: reloading weights for CPU model...");
                    let gguf2 = tempura_engine::gguf_loader::GgufLoader::from_file_quantized(model_p).unwrap();
                    let qw2 = gguf2.to_quantized_weights();
                    drop(gguf2);
                    let pipeline = KernelPipeline::cpu_verified();
                    let mut model = TransformerModel::new(config.clone(), pipeline);
                    model.quantized_weights = Some(qw2);
                    Some(std::sync::Mutex::new(model))
                } else {
                    None
                };
                Arc::new(tempura_engine::GpuQuantizedInferenceWrapper {
                    config: config.clone(),
                    attention,
                    rmsnorm,
                    rope,
                    matmul,
                    q4k_matmul,
                    sampler: tempura_kernels::sampling::CpuSampler,
                    gpu_weights,
                    gpu_cache: std::sync::Mutex::new(gpu_cache),
                    scratchpad: std::sync::Mutex::new(scratchpad),
                    embed_host: embed,
                    attn_mode,
                    cpu_model,
                })
            } else {
                let pipeline = KernelPipeline::cpu_verified();
                let mut model = TransformerModel::new(config.clone(), pipeline);
                model.quantized_weights = Some(qweights);
                tracing::info!("Using CPU quantized weights");
                Arc::new(model)
            }

            #[cfg(not(feature = "cuda"))]
            {
                let pipeline = KernelPipeline::cpu_verified();
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
        let state = AppState {
            model,
            tokenizer,
            model_id: model_p.display().to_string(),
            scheduler: Arc::new(Mutex::new(scheduler)),
            trace_store: Arc::new(Mutex::new(traces)),
            metrics_handle,
            request_timeout: std::time::Duration::from_secs(args.request_timeout_secs),
        };

        let app = api::router()
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
        let pipeline = KernelPipeline::cpu_verified();
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
    let state = AppState {
        model,
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
