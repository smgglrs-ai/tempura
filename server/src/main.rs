#![allow(dead_code, unused_imports)]
mod api;
mod batch;
mod sse;

use clap::Parser;
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
}

#[derive(Clone)]
pub struct AppState {
    pub scheduler: Arc<Mutex<batch::BatchScheduler>>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let scheduler = batch::BatchScheduler::new(args.max_batch_size, args.max_seq_len);
    let state = AppState {
        scheduler: Arc::new(Mutex::new(scheduler)),
    };

    let app = api::router()
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Starting oxide-serve on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
