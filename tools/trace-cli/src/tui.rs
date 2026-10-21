use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table},
    Terminal,
};
use std::io;
use tempura_engine::trace::InferenceTrace;

enum Tab {
    Layers,
    Sampling,
}

struct AppState {
    tab: Tab,
    scroll: usize,
    layer_range: Option<(usize, usize)>,
    token_range: Option<(usize, usize)>,
}

pub fn run_tui(
    trace: &InferenceTrace,
    layer_range: Option<(usize, usize)>,
    token_range: Option<(usize, usize)>,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = AppState {
        tab: Tab::Layers,
        scroll: 0,
        layer_range,
        token_range,
    };

    loop {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(0),
                    Constraint::Length(1),
                ])
                .split(f.area());

            let header = format!(
                " {} | model={} | prompt={} gen={} | {:?}",
                trace.trace_id,
                trace.model_id,
                trace.prompt_tokens,
                trace.generated_tokens,
                trace.level,
            );
            let header_widget = Paragraph::new(header).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("tempura-trace"),
            );
            f.render_widget(header_widget, chunks[0]);

            match state.tab {
                Tab::Layers => render_layers(f, chunks[1], trace, &state),
                Tab::Sampling => render_sampling(f, chunks[1], trace, &state),
            }

            let tab_label = match state.tab {
                Tab::Layers => "[1] Layers  2  Sampling",
                Tab::Sampling => " 1  Layers [2] Sampling",
            };
            let footer = Line::from(vec![
                Span::styled(tab_label, Style::default().fg(Color::Cyan)),
                Span::raw("  |  ?????? scroll  q quit"),
            ]);
            f.render_widget(Paragraph::new(footer), chunks[2]);
        })?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Char('1') => {
                        state.tab = Tab::Layers;
                        state.scroll = 0;
                    }
                    KeyCode::Char('2') => {
                        state.tab = Tab::Sampling;
                        state.scroll = 0;
                    }
                    KeyCode::Down | KeyCode::Char('j') => state.scroll += 1,
                    KeyCode::Up | KeyCode::Char('k') => {
                        state.scroll = state.scroll.saturating_sub(1)
                    }
                    KeyCode::PageDown => state.scroll += 10,
                    KeyCode::PageUp => state.scroll = state.scroll.saturating_sub(10),
                    KeyCode::Home => state.scroll = 0,
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

fn render_layers(
    f: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    trace: &InferenceTrace,
    state: &AppState,
) {
    let header = Row::new(vec![
        "Layer",
        "Pre-norm mean",
        "Pre-norm var",
        "Post-norm mean",
        "Duration",
    ])
    .style(
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Yellow),
    );

    let rows: Vec<Row> = trace
        .layers
        .iter()
        .filter(|s| {
            if let Some((start, end)) = state.layer_range {
                s.layer_index >= start && s.layer_index <= end
            } else {
                true
            }
        })
        .skip(state.scroll)
        .map(|snap| {
            let pre_mean = snap
                .pre_norm
                .as_ref()
                .map(|n| format!("{:.6}", n.mean))
                .unwrap_or_else(|| "-".into());
            let pre_var = snap
                .pre_norm
                .as_ref()
                .map(|n| format!("{:.6}", n.variance))
                .unwrap_or_else(|| "-".into());
            let post_mean = snap
                .post_norm
                .as_ref()
                .map(|n| format!("{:.6}", n.mean))
                .unwrap_or_else(|| "-".into());
            let dur = snap
                .duration
                .as_ref()
                .map(|d| format!("{:.2}ms", d.as_secs_f64() * 1000.0))
                .unwrap_or_else(|| "-".into());
            Row::new(vec![
                format!("{}", snap.layer_index),
                pre_mean,
                pre_var,
                post_mean,
                dur,
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(6),
            Constraint::Length(14),
            Constraint::Length(14),
            Constraint::Length(14),
            Constraint::Length(10),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title("Layers"));

    f.render_widget(table, area);
}

fn render_sampling(
    f: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    trace: &InferenceTrace,
    state: &AppState,
) {
    let header = Row::new(vec!["Step", "Token", "Top Logit", "Top Prob", "Masked"]).style(
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Yellow),
    );

    let rows: Vec<Row> = trace
        .sampling_steps
        .iter()
        .filter(|s| {
            if let Some((start, end)) = state.token_range {
                s.step >= start && s.step <= end
            } else {
                true
            }
        })
        .skip(state.scroll)
        .map(|step| {
            let top_logit = step
                .top_logits
                .first()
                .map(|(id, val)| format!("{}={:.2}", id, val))
                .unwrap_or_else(|| "-".into());
            let top_prob = step
                .top_probs
                .first()
                .map(|(id, val)| format!("{}={:.4}", id, val))
                .unwrap_or_else(|| "-".into());
            let masked = step
                .tokens_masked
                .map(|m| format!("{}", m))
                .unwrap_or_else(|| "-".into());
            Row::new(vec![
                format!("{}", step.step),
                format!("{}", step.selected_token),
                top_logit,
                top_prob,
                masked,
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(6),
            Constraint::Length(8),
            Constraint::Length(16),
            Constraint::Length(16),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title("Sampling"));

    f.render_widget(table, area);
}
