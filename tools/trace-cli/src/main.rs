mod tui;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tempura_engine::trace::InferenceTrace;

#[derive(Parser)]
#[command(name = "tempura-trace", about = "TUI viewer for tempura inference traces")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    View {
        #[arg(help = "Path to .tempura-trace or .json trace file")]
        file: PathBuf,
        #[arg(long, help = "Filter by layer range (e.g., 10-15)")]
        layers: Option<String>,
        #[arg(long, help = "Filter by token range (e.g., 30-50)")]
        tokens: Option<String>,
    },
    Diff {
        #[arg(help = "First trace file")]
        file_a: PathBuf,
        #[arg(help = "Second trace file")]
        file_b: PathBuf,
        #[arg(long, default_value = "1e-4", help = "Divergence threshold")]
        threshold: f32,
    },
    Export {
        #[arg(help = "Path to trace file")]
        file: PathBuf,
        #[arg(long, value_parser = ["json", "csv"], default_value = "json")]
        format: String,
        #[arg(long, short, help = "Output file path")]
        output: Option<PathBuf>,
    },
    Info {
        #[arg(help = "Path to trace file")]
        file: PathBuf,
    },
}

fn load_trace(path: &PathBuf) -> Result<InferenceTrace> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;

    if data.len() >= 5 && &data[0..4] == b"TMPT" {
        InferenceTrace::from_binary(&data).map_err(|e| anyhow::anyhow!("binary decode: {e}"))
    } else {
        serde_json::from_slice(&data).with_context(|| "JSON decode")
    }
}

fn parse_range(s: &str) -> Option<(usize, usize)> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() == 2 {
        let start = parts[0].parse().ok()?;
        let end = parts[1].parse().ok()?;
        Some((start, end))
    } else {
        let val = s.parse().ok()?;
        Some((val, val))
    }
}

fn cmd_info(path: &PathBuf) -> Result<()> {
    let trace = load_trace(path)?;
    println!("Trace: {}", trace.trace_id);
    println!("Model: {}", trace.model_id);
    println!("Level: {:?}", trace.level);
    println!("Seed:  {}", trace.seed);
    println!("Prompt tokens:    {}", trace.prompt_tokens);
    println!("Generated tokens: {}", trace.generated_tokens);
    println!("Layers recorded:  {}", trace.layers.len());
    println!("Sampling steps:   {}", trace.sampling_steps.len());
    if let Some(ref timing) = trace.timing {
        println!("Timing:");
        println!("  Total:      {:.1}ms", timing.total_ms);
        println!("  Prompt:     {:.1}ms", timing.prompt_ms);
        println!("  Generation: {:.1}ms", timing.generation_ms);
        println!("  Tokens/s:   {:.1}", timing.tokens_per_second);
    }
    Ok(())
}

fn cmd_view(path: &PathBuf, layers: &Option<String>, tokens: &Option<String>) -> Result<()> {
    let trace = load_trace(path)?;
    let layer_range = layers.as_ref().and_then(|s| parse_range(s));
    let token_range = tokens.as_ref().and_then(|s| parse_range(s));

    if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        tui::run_tui(&trace, layer_range, token_range)?;
    } else {
        print_trace_text(&trace, layer_range, token_range);
    }
    Ok(())
}

fn print_trace_text(
    trace: &InferenceTrace,
    layer_range: Option<(usize, usize)>,
    token_range: Option<(usize, usize)>,
) {
    println!("=== Trace {} ===", trace.trace_id);
    println!(
        "Model: {} | Prompt: {} | Generated: {} | Level: {:?}",
        trace.model_id, trace.prompt_tokens, trace.generated_tokens, trace.level
    );
    println!();

    if !trace.layers.is_empty() {
        println!("--- Layers ---");
        for snap in &trace.layers {
            if let Some((start, end)) = layer_range {
                if snap.layer_index < start || snap.layer_index > end {
                    continue;
                }
            }
            print!("  Layer {:>3}", snap.layer_index);
            if let Some(ref pre) = snap.pre_norm {
                print!("  pre_norm: mean={:.4} var={:.4}", pre.mean, pre.variance);
            }
            if let Some(ref d) = snap.duration {
                print!("  {:.2}ms", d.as_secs_f64() * 1000.0);
            }
            println!();
        }
    }

    if !trace.sampling_steps.is_empty() {
        println!("\n--- Sampling ---");
        for step in &trace.sampling_steps {
            if let Some((start, end)) = token_range {
                if step.step < start || step.step > end {
                    continue;
                }
            }
            print!("  Step {:>3}: token={:<6}", step.step, step.selected_token);
            if !step.top_logits.is_empty() {
                let top = &step.top_logits[0];
                print!(" top_logit=({}, {:.2})", top.0, top.1);
            }
            if let Some(masked) = step.tokens_masked {
                print!(" masked={masked}");
            }
            println!();
        }
    }
}

fn cmd_diff(file_a: &PathBuf, file_b: &PathBuf, threshold: f32) -> Result<()> {
    let trace_a = load_trace(file_a)?;
    let trace_b = load_trace(file_b)?;

    let diff = tempura_engine::diff_traces(&trace_a, &trace_b, threshold);
    let json = serde_json::to_string_pretty(&diff)?;
    println!("{json}");
    Ok(())
}

fn cmd_export(path: &PathBuf, format: &str, output: &Option<PathBuf>) -> Result<()> {
    let trace = load_trace(path)?;

    let content = match format {
        "json" => serde_json::to_string_pretty(&trace)?,
        "csv" => {
            let mut lines = vec!["layer_index,pre_norm_mean,pre_norm_var,duration_ms".to_string()];
            for snap in &trace.layers {
                let mean = snap.pre_norm.as_ref().map(|n| n.mean).unwrap_or(0.0);
                let var = snap.pre_norm.as_ref().map(|n| n.variance).unwrap_or(0.0);
                let dur = snap
                    .duration
                    .as_ref()
                    .map(|d| d.as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);
                lines.push(format!("{},{:.6},{:.6},{:.3}", snap.layer_index, mean, var, dur));
            }
            lines.join("\n")
        }
        _ => anyhow::bail!("unsupported format: {format}"),
    };

    if let Some(out) = output {
        std::fs::write(out, &content)?;
        println!("Exported to {}", out.display());
    } else {
        println!("{content}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Commands::View {
            file,
            layers,
            tokens,
        } => cmd_view(file, layers, tokens),
        Commands::Diff {
            file_a,
            file_b,
            threshold,
        } => cmd_diff(file_a, file_b, *threshold),
        Commands::Export {
            file,
            format,
            output,
        } => cmd_export(file, format, output),
        Commands::Info { file } => cmd_info(file),
    }
}
