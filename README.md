# acpsub

Any [ACP](https://agentclientprotocol.com) agent as a resumable subagent
over [MCP](https://modelcontextprotocol.io).

`acpsub` serves a set of MCP tools that let an orchestrating agent spawn
long-lived subagents backed by ACP agents (`devin acp`,
`claude-code-acp`, ...). Each subagent keeps its own ACP session, transcript
file, and a registry entry keyed by its session id, so sessions survive
server restarts and externally created sessions can be adopted via
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
agent = "devin"                            # default backend for spawn/adopt
permission = "allow"                       # allow | deny | ask
transcript_dir = "~/.local/share/acpsub/transcripts"
registry = "~/.local/share/acpsub/registry.json"

[agents.devin]
command = "devin"
args = ["acp"]
mode = "bypass"                            # session/set_mode after session/new
config = { model = "swe-2-max" }           # session/set_config_option id → value
allow_outside_cwd = false                  # refuse fs/* paths outside cwd
# sessions_db = "~/Library/Application Support/devin/sessions.db"

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
| `spawn` | `cwd, prompt, agent?, mode?, config?, permission?` | Start the process, `initialize`, `session/new`, `set_mode`, `set_config_option`, send the prompt. Returns `{session_id, agent, state}` at once — the session id is the handle for every other call. |
| `adopt` | `session_id, agent?, cwd?, prompt?, permission?` | Take over an existing ACP session via `session/load` — a registered (closed) session, or an external one such as a Devin session created elsewhere. Returns `{session_id, agent, state}`; with `prompt` the first turn starts immediately. |
| `send` | `session_id, prompt` | Next turn on the same session. Errors unless the subagent is `idle`, `done`, or `cancelled`. |
| `wait` | `session_id, timeout_secs` (default 600, min 60, max 3600; prefer 300–1800) | Block until the turn ends, a permission is needed, or the timeout. Returns `{state, stop_reason?, reply, tool_calls, elapsed_secs, pending_permission?}`. |
| `wait_any` | `session_ids, timeout_secs` | First of them to leave `running`. |
| `status` | `session_id` | State, cwd, agent, turns, transcript path — instant check. |
| `result` | `session_id, turn?` | The reply of the last (or nth) turn. |
| `cancel` | `session_id` | Answer pending permissions `cancelled`, send `session/cancel`. |
| `permit` | `session_id, request_id, option_id` | Answer a queued `ask`-policy permission request. |
| `transcript` | `session_id, from?, tail?, full?, thinking?` | Rendered transcript (same output as the CLI). |
| `list` | — | Live and registered subagents with state. |
| `close` | `session_id` | End the process; keep the registry entry (still adoptable). |
| `forget` | `session_id` | Remove the registry entry (and the process if live). |
| `agents` | — | Configured agents; for initialised ones their `agentInfo`, modes, config options. |

`agent` may be omitted anywhere it appears: the `[defaults].agent` setting
wins, then a single configured agent is used implicitly; with several agents
and no default the call errors asking for one.

`adopt` resolves `agent` and `cwd` from the registry for known session ids;
for unknown ones it asks the agent's session database (`sessions_db`) for the
session's working directory, and errors asking for an explicit `cwd` when the
lookup finds nothing.

The wait is event-driven: a long `timeout_secs` costs nothing while the
subagent runs and returns early on any state change, but every expiry costs
the orchestrator a model turn to re-issue the wait on a still-`running`
result. Prefer 300–1800s (5–30 min), staying under the 30-min prompt-cache
TTL. Timeouts under 60s are rejected — use `status` for an instant check.

## Model

- A **subagent** = one process + one ACP session + a transcript file,
  addressed by its `session_id`. States: `idle | running | needs_permission |
  done(stop_reason) | cancelled | failed | closed`.
- **Registry** (`registry.json`): `session_id → {agent, cwd, created,
  last_turn, turns}`, written atomically. `adopt` on a registered (or
  externally created) session id whose agent advertised `loadSession` runs
  `session/load` to resume it.
- **Transcript**: every `session/update`, prompt, and turn end is appended to
  `<transcript_dir>/<session_id>.jsonl` as a verbatim JSON record. A turn's
  `reply` is that turn's concatenated `agent_message_chunk` text.
- **Permissions**: `permission = allow|deny` answers requests by option kind;
  `ask` queues the request and flips the state to `needs_permission` until
  `permit` answers it.
- **cwd boundary**: `fs/*` and `terminal/*` requests are refused outside the
  subagent's `cwd` unless the agent has `allow_outside_cwd = true`.

## CLI

```sh
acpsub serve [--config PATH] [--log-file PATH]   # MCP over stdio
acpsub agents [--config PATH]                    # list configured agents
acpsub transcript <session_id> [--from N] [--tail N] [--full] [--thinking]
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
