use clap::{Parser, Subcommand};
use std::path::Path;
use tempura_server::ServeArgs;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "tempura",
    about = "Rust-native LLM inference server",
    version,
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the inference server
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

#[tokio::main]
async fn main() {
    match Cli::parse().command {
        Command::Serve(args) => {
            tempura_server::run_server(*args).await;
        }
        Command::Pull { repo_id } => {
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
        }
        Command::List => {
            let store = tempura_model::store::ModelStore::open();
            let models = store.list();
            if models.is_empty() {
                println!("No models installed. Use `tempura pull <repo-id>` to download one.");
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
        }
        Command::Inspect { path } => {
            match tempura_model::inspect::inspect(Path::new(&path)) {
                Ok(info) => print!("{info}"),
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}
