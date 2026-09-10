//! `acpsub` binary: `serve` (MCP over stdio), `agents`, and `transcript`.

use std::fmt::Write as _;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use acpsub::{AppState, Config, RenderOptions, build_tools, default_config_path, render};
use clap::{Parser, Subcommand};
use tracing::error;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Any ACP agent as a named, resumable subagent over MCP.
#[derive(Parser)]
#[command(name = "acpsub", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the subagent tools as an MCP server over stdio. This is what an
    /// orchestrating agent runs; logs go to stderr and `--log-file`.
    Serve {
        /// Config file path (default ~/.config/acpsub/config.toml).
        #[arg(long)]
        config: Option<PathBuf>,
        /// Also write logs to this file.
        #[arg(long)]
        log_file: Option<PathBuf>,
    },
    /// Print the configured agents.
    Agents {
        /// Config file path (default ~/.config/acpsub/config.toml).
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Render a subagent's transcript.
    Transcript {
        /// Subagent name.
        name: String,
        /// Start at record N (skip the first N records).
        #[arg(long)]
        from: Option<usize>,
        /// Show only the last N records.
        #[arg(long)]
        tail: Option<usize>,
        /// Do not clip long values.
        #[arg(long)]
        full: bool,
        /// Include thinking chunks.
        #[arg(long)]
        thinking: bool,
        /// Config file path (default ~/.config/acpsub/config.toml).
        #[arg(long)]
        config: Option<PathBuf>,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Serve { config, log_file } => serve(config, log_file).await,
        Command::Agents { config } => agents(config),
        Command::Transcript {
            name,
            from,
            tail,
            full,
            thinking,
            config,
        } => transcript(&name, config, from, tail, full, thinking),
    }
}

/// Load the config from `--config` or the default path.
fn load_config(path: Option<PathBuf>) -> acpsub::Result<Config> {
    let path = path
        .or_else(default_config_path)
        .ok_or_else(|| acpsub::Error::NoHome("~/.config/acpsub/config.toml".to_string()))?;
    Config::load(&path)
}

/// `acpsub serve`: MCP over stdio; diagnostics on stderr and `--log-file`.
async fn serve(
    config: Option<PathBuf>,
    log_file: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("acpsub=info,warn"));
    let stderr_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);
    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(stderr_layer);
    if let Some(path) = log_file {
        let file = std::fs::File::create(&path)?;
        registry
            .with(
                tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .with_writer(LogFile(std::sync::Mutex::new(file))),
            )
            .init();
    } else {
        registry.init();
    }

    let config = load_config(config)?;
    let state = AppState::new(config)?;
    let tools = build_tools(state)?;
    let mut server = aither_mcp::McpServer::stdio(tools, "acpsub", env!("CARGO_PKG_VERSION"));
    server.run().await.map_err(|error| {
        error!(%error, "MCP server failed");
        error
    })?;
    Ok(())
}

/// `acpsub agents`: the configured agents, one line each.
fn agents(config: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(config)?;
    let mut out = String::new();
    for (name, agent) in &config.agents {
        let mut line = format!("{name}\t{} {}", agent.command, agent.args.join(" "));
        if let Some(mode) = &agent.mode {
            let _ = write!(line, "\tmode={mode}");
        }
        if !agent.config.is_empty() {
            let options = agent
                .config
                .iter()
                .map(|(id, value)| {
                    let value = match value {
                        acpsub::config::ConfigValue::Select(v) => v.clone(),
                        acpsub::config::ConfigValue::Toggle(v) => v.to_string(),
                    };
                    format!("{id}={value}")
                })
                .collect::<Vec<_>>()
                .join(",");
            let _ = write!(line, "\tconfig={{{options}}}");
        }
        if agent.allow_outside_cwd {
            line.push_str("\tallow_outside_cwd");
        }
        if let Some(permission) = agent.permission {
            let name = serde_json::to_value(permission)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            let _ = write!(line, "\tpermission={name}");
        }
        out.push_str(&line);
        out.push('\n');
    }
    std::io::stdout().write_all(out.as_bytes())?;
    Ok(())
}

/// `acpsub transcript <name>`: render the JSONL transcript.
fn transcript(
    name: &str,
    config: Option<PathBuf>,
    from: Option<usize>,
    tail: Option<usize>,
    full: bool,
    thinking: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(config)?;
    let path = config.defaults.transcript_dir.join(format!("{name}.jsonl"));
    let text =
        std::fs::read_to_string(&path).map_err(|_| acpsub::Error::NoTranscript(path.clone()))?;
    let rendered = render(
        &text,
        &RenderOptions {
            from: from.unwrap_or(0),
            tail,
            full,
            thinking,
        },
    );
    std::io::stdout().write_all(rendered.as_bytes())?;
    Ok(())
}

/// A `MakeWriter` that appends log lines to a file behind a mutex.
struct LogFile(std::sync::Mutex<std::fs::File>);

impl std::io::Write for &LogFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("log file poisoned").write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().expect("log file poisoned").flush()
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogFile {
    type Writer = &'a Self;

    fn make_writer(&'a self) -> Self::Writer {
        self
    }
}
