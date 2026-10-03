//! `acpsub` binary: `serve` (MCP over stdio), `daemon` (MCP over a Unix
//! socket), the client commands (`spawn`, `send`, `wait`, …), `agents`,
//! and `transcript`.

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use acpsub::config::PermissionPolicy;
use acpsub::{AppState, Config, RenderOptions, build_tools, default_config_path, render};
use aither_mcp::protocol::{CallToolResult, Content};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::{Value, json};
use tracing::error;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Any ACP agent as a resumable subagent: an MCP server, a daemon, or the
/// CLI that drives the daemon.
#[derive(Parser)]
#[command(name = "acpsub", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Connection options every daemon client command accepts.
#[derive(Args)]
struct DaemonConn {
    /// Daemon socket path (default ~/.local/share/acpsub/daemon.sock).
    /// When no daemon answers, one is started automatically.
    #[arg(long)]
    socket: Option<PathBuf>,
    /// Config file path (default ~/.config/acpsub/config.toml); forwarded
    /// to the daemon when it has to be started for this call.
    #[arg(long)]
    config: Option<PathBuf>,
}

impl DaemonConn {
    /// Resolve the socket path.
    fn socket_path(&self) -> acpsub::Result<PathBuf> {
        self.socket
            .clone()
            .or_else(|| std::env::var_os("ACPSUB_SOCKET").map(PathBuf::from))
            .or_else(acpsub::daemon::default_socket_path)
            .ok_or_else(|| acpsub::Error::NoHome("~/.local/share/acpsub/daemon.sock".to_string()))
    }
}

/// A prompt given inline or from a file (`--prompt-file -` reads stdin).
#[derive(Args)]
struct PromptArg {
    /// The prompt text.
    #[arg(long, conflicts_with = "prompt_file")]
    prompt: Option<String>,
    /// Read the prompt from this file; `-` reads stdin.
    #[arg(long)]
    prompt_file: Option<PathBuf>,
}

impl PromptArg {
    /// Resolve the prompt text.
    fn text(&self) -> acpsub::Result<Option<String>> {
        match (&self.prompt, &self.prompt_file) {
            (Some(text), None) => Ok(Some(text.clone())),
            (None, Some(path)) => {
                let text = if path.as_os_str() == "-" {
                    let mut text = String::new();
                    std::io::stdin()
                        .read_to_string(&mut text)
                        .map_err(|source| {
                            acpsub::Error::io("cannot read prompt from stdin", source)
                        })?;
                    text
                } else {
                    std::fs::read_to_string(path).map_err(|source| {
                        acpsub::Error::io(format!("cannot read {}", path.display()), source)
                    })?
                };
                Ok(Some(text))
            }
            _ => Ok(None),
        }
    }
}

/// `send`'s delivery policy while a turn is running.
#[derive(Clone, Copy, ValueEnum)]
enum SendPolicy {
    /// Error unless the subagent is idle/done/cancelled.
    Try,
    /// Park the prompt on the subagent's FIFO queue.
    Queued,
    /// Inject the prompt into the running turn.
    Steer,
}

/// A permission policy flag value.
#[derive(Clone, Copy, ValueEnum)]
enum Permission {
    /// Auto-approve permission requests.
    Allow,
    /// Auto-reject permission requests.
    Deny,
    /// Queue requests for `permit`.
    Ask,
}

impl Permission {
    /// The tool argument value.
    const fn policy(self) -> PermissionPolicy {
        match self {
            Self::Allow => PermissionPolicy::Allow,
            Self::Deny => PermissionPolicy::Deny,
            Self::Ask => PermissionPolicy::Ask,
        }
    }
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
    /// Serve the subagent tools over a Unix socket as a standalone daemon.
    /// Holds the live ACP sessions; the client commands below connect to it.
    /// Stays resident across coordinator restarts; reaps sessions whose
    /// owning coordinator pid (`--owner` at spawn) has exited.
    Daemon {
        /// Socket path (default ~/.local/share/acpsub/daemon.sock).
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Config file path (default ~/.config/acpsub/config.toml).
        #[arg(long)]
        config: Option<PathBuf>,
        /// Log file (default <socket dir>/daemon.log).
        #[arg(long)]
        log_file: Option<PathBuf>,
    },
    /// Spawn a subagent: start the agent, open a session in --cwd, send the
    /// first prompt. Returns immediately with the session id.
    Spawn {
        /// Session working directory.
        #[arg(long)]
        cwd: PathBuf,
        /// Model to run (e.g. swe-2-high). Required when the agent advertises
        /// a `model` option; omit it for an agent that advertises none.
        #[arg(long)]
        model: Option<String>,
        /// Session mode to activate (e.g. bypass). Required when the agent
        /// advertises session modes; omit it for an agent that advertises none.
        #[arg(long)]
        mode: Option<String>,
        /// Configured agent key; falls back to the default.
        #[arg(long)]
        agent: Option<String>,
        /// Extra session config options as key=value (booleans accepted).
        #[arg(long = "set", value_name = "KEY=VALUE")]
        config_options: Vec<String>,
        /// Permission policy for this subagent.
        #[arg(long)]
        permission: Option<Permission>,
        /// Owning coordinator pid; the daemon reaps the session if it dies
        /// (default: `$ACPSUB_OWNER`).
        #[arg(long)]
        owner: Option<u32>,
        #[command(flatten)]
        prompt: PromptArg,
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Adopt an existing session id as a live subagent via session/load.
    Adopt {
        /// Session id to take over.
        session_id: String,
        /// Model to run; omit it for an agent that advertises no `model` option.
        #[arg(long)]
        model: Option<String>,
        /// Session mode to activate; omit it for an agent that advertises no
        /// session modes.
        #[arg(long)]
        mode: Option<String>,
        /// Configured agent key.
        #[arg(long)]
        agent: Option<String>,
        /// Session working directory (usually discovered).
        #[arg(long)]
        cwd: Option<PathBuf>,
        /// Permission policy override.
        #[arg(long)]
        permission: Option<Permission>,
        /// Owning coordinator pid.
        #[arg(long)]
        owner: Option<u32>,
        #[command(flatten)]
        prompt: PromptArg,
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Send a prompt to a live subagent.
    Send {
        /// Session id.
        session_id: String,
        /// Delivery policy while a turn is running.
        #[arg(long)]
        policy: SendPolicy,
        #[command(flatten)]
        prompt: PromptArg,
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Block until the subagent's turn ends, a permission needs an answer,
    /// or the turn outlives --expect. Designed to run in the background:
    /// exit means "check the result".
    Wait {
        /// Session id.
        session_id: String,
        /// Expected turn duration in seconds; an overrun returns early with
        /// `state: "overrun"` for investigation.
        #[arg(long)]
        expect: u64,
        /// The wait's own deadline in seconds. When it passes with the
        /// turn still running, wait returns `state: "running"` with an
        /// activity `digest` of the window (shape documented in the
        /// README's `wait` entry) instead of dying silent under a
        /// background-task kill limit — set it just below that limit.
        /// Overrun results carry the same digest.
        #[arg(long)]
        max_wait: Option<u64>,
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Block until the first of several subagents leaves `running`.
    WaitAny {
        /// Session ids to watch.
        session_ids: Vec<String>,
        /// Expected turn duration in seconds.
        #[arg(long)]
        expect: u64,
        /// The wait's own deadline in seconds, as in `wait`: on expiry the
        /// result is the longest-running watched turn's `state: "running"`
        /// with its activity `digest`.
        #[arg(long)]
        max_wait: Option<u64>,
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Report a subagent's state.
    Status {
        /// Session id.
        session_id: String,
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Return a turn's reply (latest turn by default).
    Result {
        /// Session id.
        session_id: String,
        /// 1-based turn number.
        #[arg(long)]
        turn: Option<u64>,
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Cancel a subagent's running turn.
    Cancel {
        /// Session id.
        session_id: String,
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Answer a pending permission request from `wait`'s
    /// `pending_permission` field.
    Permit {
        /// Session id.
        session_id: String,
        /// Pending permission request id (perm-N).
        #[arg(long)]
        request: String,
        /// Option id to select (one of the request's options).
        #[arg(long)]
        option: String,
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// List subagents: live ones and registered (closed but resumable).
    List {
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Close a subagent (registry entry stays; `forget` removes it).
    Close {
        /// Session id.
        session_id: String,
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Close a subagent and remove its registry entry.
    Forget {
        /// Session id.
        session_id: String,
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Restart the daemon: it drains in-flight turns, closes every live
    /// session marked for resume, and exits; the next daemon — started by
    /// this call when none answers — re-adopts them under the same ids.
    /// Returns once the new daemon finished resuming and prints the
    /// resume report.
    Restart {
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Call the daemon's `agents` tool: configured agents plus the live
    /// sessions' reported agent info, modes, and config options.
    AgentsLive {
        #[command(flatten)]
        conn: DaemonConn,
    },
    /// Print the configured agents.
    Agents {
        /// Config file path (default ~/.config/acpsub/config.toml).
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Render a subagent's transcript.
    Transcript {
        /// Session id.
        session_id: String,
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
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<ExitCode, Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Serve { config, log_file } => {
            serve(config, log_file).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Daemon {
            config,
            socket,
            log_file,
        } => {
            daemon(config, socket, log_file).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Agents { config } => {
            agents(config)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Transcript {
            session_id,
            from,
            tail,
            full,
            thinking,
            config,
        } => {
            transcript(&session_id, config, from, tail, full, thinking)?;
            Ok(ExitCode::SUCCESS)
        }
        command => call(command).await,
    }
}

/// Run a daemon client command: connect, call the matching tool, print the
/// result.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per subcommand; every arm is a thin flag-to-JSON mapping"
)]
async fn call(command: Command) -> Result<ExitCode, Box<dyn std::error::Error>> {
    if let Command::Restart { conn } = command {
        return restart(conn).await;
    }
    let (conn, tool, arguments) = match command {
        Command::Spawn {
            cwd,
            model,
            mode,
            agent,
            config_options,
            permission,
            owner,
            prompt,
            conn,
        } => (
            conn,
            "spawn",
            spawn_args(
                &cwd,
                model.as_deref(),
                mode.as_deref(),
                agent.as_deref(),
                &config_options,
                permission,
                owner,
                &prompt,
            )?,
        ),
        Command::Adopt {
            session_id,
            model,
            mode,
            agent,
            cwd,
            permission,
            owner,
            prompt,
            conn,
        } => (
            conn,
            "adopt",
            adopt_args(
                &session_id,
                model.as_deref(),
                mode.as_deref(),
                agent.as_deref(),
                cwd.as_deref(),
                permission,
                owner,
                &prompt,
            )?,
        ),
        Command::Send {
            session_id,
            policy,
            prompt,
            conn,
        } => {
            let prompt = prompt
                .text()?
                .ok_or("send needs --prompt or --prompt-file")?;
            let policy = match policy {
                SendPolicy::Try => "try",
                SendPolicy::Queued => "queued",
                SendPolicy::Steer => "steer",
            };
            (
                conn,
                "send",
                json!({"session_id": session_id, "prompt": prompt, "policy": policy}),
            )
        }
        Command::Wait {
            session_id,
            expect,
            max_wait,
            conn,
        } => {
            let mut args = json!({"session_id": session_id, "expect_secs": expect});
            set_if(&mut args, "max_wait_secs", max_wait);
            (conn, "wait", args)
        }
        Command::WaitAny {
            session_ids,
            expect,
            max_wait,
            conn,
        } => {
            let mut args = json!({"session_ids": session_ids, "expect_secs": expect});
            set_if(&mut args, "max_wait_secs", max_wait);
            (conn, "wait_any", args)
        }
        Command::Status { session_id, conn } => (conn, "status", json!({"session_id": session_id})),
        Command::Result {
            session_id,
            turn,
            conn,
        } => {
            let mut args = json!({"session_id": session_id});
            set_if(&mut args, "turn", turn);
            (conn, "result", args)
        }
        Command::Cancel { session_id, conn } => (conn, "cancel", json!({"session_id": session_id})),
        Command::Permit {
            session_id,
            request,
            option,
            conn,
        } => (
            conn,
            "permit",
            json!({"session_id": session_id, "request_id": request, "option_id": option}),
        ),
        Command::List { conn } => (conn, "list", json!({})),
        Command::Close { session_id, conn } => (conn, "close", json!({"session_id": session_id})),
        Command::Forget { session_id, conn } => (conn, "forget", json!({"session_id": session_id})),
        Command::AgentsLive { conn } => (conn, "agents", json!({})),
        _ => unreachable!("non-client commands are handled before `call`"),
    };
    let socket = conn.socket_path()?;
    let result = acpsub::client::call(&socket, tool, arguments, conn.config.as_deref()).await?;
    Ok(print_result(result))
}

/// Build the `spawn` tool arguments from the CLI flags.
#[expect(clippy::too_many_arguments, reason = "mirrors the clap surface")]
fn spawn_args(
    cwd: &Path,
    model: Option<&str>,
    mode: Option<&str>,
    agent: Option<&str>,
    config_options: &[String],
    permission: Option<Permission>,
    owner: Option<u32>,
    prompt: &PromptArg,
) -> Result<Value, Box<dyn std::error::Error>> {
    let prompt = prompt
        .text()?
        .ok_or("spawn needs --prompt or --prompt-file")?;
    let mut args = json!({
        "cwd": cwd,
        "prompt": prompt,
    });
    set_if(&mut args, "model", model);
    set_if(&mut args, "mode", mode);
    set_if(&mut args, "agent", agent);
    if !config_options.is_empty() {
        args["config"] = parse_config_options(config_options)?;
    }
    set_if(&mut args, "permission", permission.map(Permission::policy));
    set_if(&mut args, "owner", owner.or_else(owner_env));
    Ok(args)
}

/// Build the `adopt` tool arguments from the CLI flags.
#[expect(clippy::too_many_arguments, reason = "mirrors the clap surface")]
fn adopt_args(
    session_id: &str,
    model: Option<&str>,
    mode: Option<&str>,
    agent: Option<&str>,
    cwd: Option<&Path>,
    permission: Option<Permission>,
    owner: Option<u32>,
    prompt: &PromptArg,
) -> Result<Value, Box<dyn std::error::Error>> {
    let mut args = json!({
        "session_id": session_id,
    });
    set_if(&mut args, "model", model);
    set_if(&mut args, "mode", mode);
    set_if(&mut args, "agent", agent);
    set_if(&mut args, "cwd", cwd);
    set_if(&mut args, "prompt", prompt.text()?);
    set_if(&mut args, "permission", permission.map(Permission::policy));
    set_if(&mut args, "owner", owner.or_else(owner_env));
    Ok(args)
}

/// Insert `key` into `args` when the value is present.
fn set_if(args: &mut Value, key: &str, value: Option<impl serde::Serialize>) {
    if let Some(value) = value {
        args[key] = serde_json::to_value(value).expect("serializable arg");
    }
}

/// Parse `--set KEY=VALUE` pairs into session config options; `true`/`false`
/// become toggle values, anything else a select id.
fn parse_config_options(pairs: &[String]) -> acpsub::Result<Value> {
    let mut options = serde_json::Map::new();
    for pair in pairs {
        let (key, value) = pair.split_once('=').ok_or_else(|| {
            acpsub::Error::io(
                format!("--set expects KEY=VALUE, got '{pair}'"),
                std::io::Error::other("bad --set syntax"),
            )
        })?;
        let value = match value {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            other => Value::String(other.to_string()),
        };
        options.insert(key.to_string(), value);
    }
    Ok(Value::Object(options))
}

/// `acpsub restart`: ask the daemon to drain, wait for its exit — the
/// socket refusing connections is the signal the drain completed — then
/// let the auto-started replacement resume every marked session. The
/// `daemon/resumed` answer only exists once resuming finished, so it is
/// the report this prints.
async fn restart(conn: DaemonConn) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let socket = conn.socket_path()?;
    if async_net::unix::UnixStream::connect(&socket).await.is_err() {
        return Err(format!("no acpsub daemon at {}", socket.display()).into());
    }
    let ack =
        acpsub::client::call(&socket, "daemon/drain", json!({}), conn.config.as_deref()).await?;
    if ack.is_error {
        return Ok(print_result(ack));
    }
    acpsub::client::wait_for_next_daemon(&socket).await?;
    let report =
        acpsub::client::call(&socket, "daemon/resumed", json!({}), conn.config.as_deref()).await?;
    Ok(print_result(report))
}

/// The owning coordinator pid from the environment.
fn owner_env() -> Option<u32> {
    std::env::var("ACPSUB_OWNER")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Print a tool result: JSON pretty-printed when it parses, raw text
/// otherwise; tool errors go to stderr with a failure exit.
fn print_result(result: CallToolResult) -> ExitCode {
    let mut text = String::new();
    for item in result.content {
        if let Content::Text(item) = item {
            text.push_str(&item.text);
        }
    }
    if result.is_error {
        eprintln!("{text}");
        return ExitCode::FAILURE;
    }
    match serde_json::from_str::<Value>(text.trim()) {
        Ok(value) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&value).expect("Value serializes")
            );
        }
        Err(_) => println!("{text}"),
    }
    ExitCode::SUCCESS
}

/// Resolve the config path from `--config` or the default location.
fn config_path(path: Option<PathBuf>) -> acpsub::Result<PathBuf> {
    path.or_else(default_config_path)
        .ok_or_else(|| acpsub::Error::NoHome("~/.config/acpsub/config.toml".to_string()))
}

/// Load the config from `--config` or the default path.
fn load_config(path: Option<PathBuf>) -> acpsub::Result<Config> {
    Config::load(&config_path(path)?)
}

/// Shared tracing setup for `serve` and `daemon`: stderr plus an optional
/// log file.
fn init_logging(log_file: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
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
    Ok(())
}

/// `acpsub serve`: MCP over stdio; diagnostics on stderr and `--log-file`.
async fn serve(
    config: Option<PathBuf>,
    log_file: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    init_logging(log_file)?;
    let config_path = config_path(config)?;
    let config = Config::load(&config_path)?;
    let state = AppState::new(config, config_path)?;
    let tools = build_tools(state)?;
    let mut server = aither_mcp::McpServer::stdio(tools, "acpsub", env!("CARGO_PKG_VERSION"));
    server.run().await.map_err(|error| {
        error!(%error, "MCP server failed");
        error
    })?;
    Ok(())
}

/// `acpsub daemon`: MCP over a Unix socket; sessions outlive the caller and
/// are reaped when their owning coordinator pid dies.
async fn daemon(
    config: Option<PathBuf>,
    socket: Option<PathBuf>,
    log_file: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let socket = socket
        .or_else(acpsub::daemon::default_socket_path)
        .ok_or_else(|| acpsub::Error::NoHome("~/.local/share/acpsub/daemon.sock".to_string()))?;
    // A detached daemon has no useful stderr: default the log file to
    // daemon.log beside the socket.
    let log_file = log_file.or_else(|| socket.parent().map(|dir| dir.join("daemon.log")));
    init_logging(log_file)?;
    let config_path = config_path(config)?;
    let config = Config::load(&config_path)?;
    let state = AppState::new(config, config_path)?;
    acpsub::daemon::run(state, &socket).await?;
    Ok(())
}

/// `acpsub agents`: the configured agents, one line each.
fn agents(config: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(config)?;
    let mut out = String::new();
    for (name, agent) in &config.agents {
        let mut line = format!("{name}\t{} {}", agent.command, agent.args.join(" "));
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

/// `acpsub transcript <session_id>`: render the JSONL transcript.
fn transcript(
    session_id: &str,
    config: Option<PathBuf>,
    from: Option<usize>,
    tail: Option<usize>,
    full: bool,
    thinking: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(config)?;
    let path = config
        .defaults
        .transcript_dir
        .join(format!("{session_id}.jsonl"));
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
