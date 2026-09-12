# acpsub

Any [ACP](https://agentclientprotocol.com) agent as a named, resumable subagent
over [MCP](https://modelcontextprotocol.io).

`acpsub` serves a set of MCP tools that let an orchestrating agent spawn
long-lived subagents backed by ACP agents (`devin acp`,
`claude-code-acp`, ...). Each subagent keeps its own ACP session, transcript
file, and a registry entry, so sessions survive server restarts via
`session/load`.

## Install

```sh
cargo install acpsub
```

## Wire it into your orchestrating agent

Claude Code:

```sh
claude mcp add --scope user acpsub -- acpsub serve
```

Any other MCP client: run `acpsub serve` and speak MCP over stdio. Logs go to
stderr (stdout is the MCP channel); add `--log-file PATH` for a copy.

## Configuration

`~/.config/acpsub/config.toml` (override with `--config PATH`):

```toml
[defaults]
permission = "allow"                       # allow | deny | ask
transcript_dir = "~/.local/share/acpsub/transcripts"
registry = "~/.local/share/acpsub/registry.json"

[agents.devin]
command = "devin"
args = ["acp"]
mode = "bypass"                            # session/set_mode after session/new
config = { model = "swe-2-max" }           # session/set_config_option id → value
allow_outside_cwd = false                  # refuse fs/* paths outside cwd

[agents.claude]
command = "npx"
args = ["-y", "@zed-industries/claude-code-acp"]
mode = "bypassPermissions"
```

`~` expands in all paths. See `config.example.toml`.

A missing config file is an error naming the path — create it first. `agents`
with no `[agents.*]` entries serves no agents.

## Tools

| tool | arguments | behaviour |
|---|---|---|
| `spawn` | `name, agent, cwd, prompt, mode?, config?, permission?, replace?` | Start the process (`session/load` for a resumable registered name), `initialize`, `session/new`, `set_mode`, `set_config_option`, send the prompt. Returns `{name, session_id, agent, state}` at once. |
| `send` | `name, prompt` | Next turn on the same session. Errors unless the subagent is `idle`, `done`, or `cancelled`. |
| `wait` | `name, timeout_secs` (default 600, max 3600) | Block until the turn ends, a permission is needed, or the timeout. Returns `{state, stop_reason?, reply, tool_calls, elapsed_secs, pending_permission?}`. |
| `wait_any` | `names, timeout_secs` | First of them to leave `running`. |
| `status` | `name` | State, session id, cwd, agent, turns, transcript path. |
| `result` | `name, turn?` | The reply of the last (or nth) turn. |
| `cancel` | `name` | Answer pending permissions `cancelled`, send `session/cancel`. |
| `permit` | `name, request_id, option_id` | Answer a queued `ask`-policy permission request. |
| `transcript` | `name, from?, tail?, full?, thinking?` | Rendered transcript (same output as the CLI). |
| `list` | — | Live and registered subagents with state. |
| `close` | `name` | End the process; keep the registry entry (still resumable). |
| `forget` | `name` | Remove the registry entry (and the process if live). |
| `agents` | — | Configured agents; for initialised ones their `agentInfo`, modes, config options. |

## Model

- A **subagent** = `name` (`[A-Za-z0-9._-]+`) + one process + one ACP session +
  a transcript file. States: `idle | running | needs_permission |
  done(stop_reason) | cancelled | failed`.
- **Registry** (`registry.json`): `name → {agent, session_id, cwd, created,
  last_turn, turns}`, written atomically. `spawn` of a registered name whose
  agent advertised `loadSession` runs `session/load`; otherwise it errors
  (`replace=true` forgets the old entry instead).
- **Transcript**: every `session/update`, prompt, and turn end is appended to
  `<transcript_dir>/<name>.jsonl` as a verbatim JSON record. A turn's `reply`
  is that turn's concatenated `agent_message_chunk` text.
- **Permissions**: `permission = allow|deny` answers requests by option kind;
  `ask` queues the request and flips the state to `needs_permission` until
  `permit` answers it.
- **cwd boundary**: `fs/*` and `terminal/*` requests are refused outside the
  subagent's `cwd` unless the agent has `allow_outside_cwd = true`.

## CLI

```sh
acpsub serve [--config PATH] [--log-file PATH]   # MCP over stdio
acpsub agents [--config PATH]                    # list configured agents
acpsub transcript <name> [--from N] [--tail N] [--full] [--thinking]
```

`acpsub transcript` renders a session's JSONL file:

```text
=== [turn 1] USER
Reply with the single word pong.
--- ASSISTANT
pong
--- CALL execute cargo build (tc-1) [completed]
    /tmp/project/src/main.rs
--- PLAN
    [completed] reproduce
    [in_progress] fix
=== [turn 1] END end_turn
```

Long values are clipped to 400 chars unless `--full`; `--tail N` shows the
last N records, `--from N` skips the first N, `--thinking` includes
`agent_thought_chunk` records.

## Development

```sh
cargo check --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo nextest run
cargo doc --no-deps
```

Python-dependent tests skip themselves when `python3` is absent.

## Releases

`release-plz` runs on pushes to `main` (`.github/workflows/release.yml`). The
crate's crates.io Trusted Publishing configuration names this workflow, so
publishing authenticates with GitHub OIDC and no token secret is needed.
