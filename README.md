# acpsub

Any [ACP](https://agentclientprotocol.com) agent as a resumable subagent —
over [MCP](https://modelcontextprotocol.io) or over the built-in CLI.

`acpsub` serves a set of MCP tools that let an orchestrating agent spawn
long-lived subagents backed by ACP agents (`devin acp`,
`claude-code-acp`, ...). The tool set is served two ways: `acpsub serve`
speaks MCP over stdio for an MCP host, and `acpsub daemon` serves the same
tools over a Unix socket for the `acpsub` CLI. Each subagent keeps its own
ACP session, transcript file, and a registry entry keyed by its session id,
so sessions survive restarts and externally created sessions can be adopted
via `session/load`.

## Install

Prebuilt binaries for macOS, Linux, and Windows ship on every GitHub
release; the installer puts `acpsub` on your `PATH`:

```sh
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/lexoliu/acpsub/releases/latest/download/acpsub-installer.sh | sh
```

From source instead:

```sh
cargo install acpsub --locked
```

## Wire it into your orchestrating agent

**CLI (daemon)**: the client commands below start `acpsub daemon`
automatically when the socket is silent — nothing to wire up. `daemon` keeps
running after the caller exits; it reaps a session whose `--owner` pid has
died, so a subagent cannot outlive its coordinator as an orphan.

```sh
acpsub spawn --cwd . --model swe-2-high --mode bypass --prompt "Fix the typo in README" --owner $PPID
acpsub wait <session_id> --expect 600   # blocks until the turn ends —
                                      # run it in the background
```

**MCP**: an orchestrating agent that prefers tool calls over a CLI adds the
stdio server — `claude mcp add --scope user acpsub -- acpsub serve` for
Claude Code, or `acpsub serve` directly for any other MCP client. In this
mode a blocking `wait` holds the caller's turn; the CLI path exists to avoid
that. Logs go to stderr (stdout is the MCP channel); add `--log-file PATH`
for a copy.

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
allow_outside_cwd = false                  # refuse fs/* paths outside cwd
# sessions_db = "~/Library/Application Support/devin/sessions.db"

[agents.claude]
command = "npx"
args = ["-y", "@zed-industries/claude-code-acp"]

[agents.codex]
command = "codex-acp"
steer = "folded"                         # folds a mid-turn prompt into the
                                         # running turn and never answers it
```

`~` expands in all paths. See `config.example.toml`.

`[agents.<name>]` only describes how to launch the agent — session options
are never configured there. `spawn`/`adopt` take `model` and `mode`
arguments that pair with what the agent advertises: each is required when
the agent advertises a `model` config option / session modes, and an error
when passed to an agent that advertises none (no `set_mode` or
`set_config_option` call is made then). The result reports the model and
mode the agent accepted — `null` for an agent without them.

A missing config file is an error naming the path — create it first. `agents`
with no `[agents.*]` entries serves no agents. The file is re-read on every
`spawn`/`adopt`/`agents` call, so edits take effect without a restart and a
file that fails to parse is a tool error, never a stale fallback.

## Tools

| tool | arguments | behaviour |
|---|---|---|
| `spawn` | `cwd, prompt, model?, mode?, agent?, config?, permission?, owner?` | Start the process, `initialize`, `session/new`, `set_mode`, `set_config_option`, send the prompt. `model`/`mode` are required when the agent advertises a `model` option / session modes and must be omitted when it does not (the calls are skipped). Returns `{session_id, agent, state, model, mode}` at once — the session id is the handle for every other call. `owner` (a pid) lets the daemon reap the subagent when its coordinator exits. While a rate limit holds in the agent's quota scope the prompt parks instead: `{state: "rate_limited", resume_at, reason, position}`. |
| `adopt` | `session_id, model?, mode?, agent?, cwd?, prompt?, permission?, owner?` | Take over an existing ACP session via `session/load` — a registered (closed) session, a live `exited` one (the dead runtime is reaped, no `close` needed), or an external one such as a Devin session created elsewhere. Same `model`/`mode` rule as `spawn`. Returns `{session_id, agent, state, model, mode}`; with `prompt` the first turn starts immediately. |
| `send` | `session_id, prompt, policy` | Prompt the session; `policy` (required — nothing is queued or injected implicitly) says what a `running`/`needs_permission` subagent does with it: `try` errors unless `idle`/`done`/`cancelled`/`failed` (the original behaviour); `queued` parks it FIFO and fires it as the next turn when the current one ends — dropped if that turn is cancelled or fails, and on `cancel`/`close`/`forget` — returning `{state: "queued", position}`; `steer` injects it into the running turn as a second `session/prompt` (mid-turn steering — agents that support it fold the text into the active task, agents that do not surface an error; agents configured `steer = "folded"` never answer it and the steer settles folded at the turn's end), returning `{state: "steered", turn}`. On a promptable subagent all three just start the turn: `{state: "running"}`. Under a rate limit in the session's quota scope, any policy's prompt parks and returns `{state: "rate_limited", resume_at, reason, position}`. |
| `wait` | `session_id, expect_secs, max_wait_secs?` | Block until the turn ends, a permission is needed, or the turn has run longer than `expect_secs` — required, measured from the turn's recorded start, so re-issuing a wait never extends it. Returns `{state, turn, stop_reason?, reply, tool_calls, queued, elapsed_secs, pending_permission?, digest?, resume_at?, reason?}`; a turn past its budget reports `state: "overrun"` with the turn's `elapsed_secs` and `latest_tool_call`. `max_wait_secs` bounds the wait call itself: when it passes first, the result is `state: "running"` with an activity `digest` (below). `overrun` results carry the same digest. A bound turn that ends rate-limited does not end the wait: it rebinds to the resume's continuation turn and freezes while parked — the parked seconds never count against `expect_secs` — and a parked session reports `state: "rate_limited"` with `resume_at` and the provider's `reason`. |
| `wait_any` | `session_ids, expect_secs, max_wait_secs?` | First of them to leave `running` — a turn end, a permission request, or an `overrun`. On a `max_wait_secs` timeout, the longest-running watched turn's `running` state with its `digest`. |
| `status` | `session_id` | State, cwd, agent, model, mode (`null` for agents without them), turns, queued prompts, transcript path — instant check; a parked session also reports `resume_at` and `reason`. |
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

The wait is event-driven and returns early on any state change, so
`expect_secs` is an honest expectation of the turn's duration, not a cap
on how long you are willing to block. When a turn outlives it the result
is `overrun` with the turn's elapsed time and latest tool call — an
overrun is a signal to investigate (`status`, `transcript`, a `send`
steer), never to re-wait with a larger number. For an instant check use
`status`.

`max_wait_secs` is the cap on how long the wait call itself blocks —
measured from when the wait began, unlike `expect_secs`. It exists
because a backgrounded `wait` can outlive its host's task limit: set it
just below that limit and the wait returns `state: "running"` with an
activity `digest` instead of dying silent. The digest covers the window
since the wait began, and every value is measured from the tool calls'
recorded `started_at`/`ended_at` — nothing is classified from command
text:

```json
"digest": {
  "window_secs": 60.1,          // length of the measured window
  "tool_calls": 6,              // calls whose recorded lifetime overlaps it
  "tool_call_share": 0.83,      // fraction of window wall time inside calls
                                //   (> 1 when calls overlap)
  "longest_tool_calls": [       // up to 5, by in-window duration
    {"id": "tc-7", "title": "run sleep 300", "duration_secs": 60.1}
  ],
  "repeated_titles": [          // every title seen 3+ times
    {"title": "run check", "count": 3}
  ],
  "secs_since_last_edit": 42.7, // since the last edit/delete/move call
                                //   finished; ~0 while one runs; null
                                //   when none overlaps the window
  "latest_tool_call": {"id": "tc-7", "title": "run sleep 300", ...}
}
```

## Model

- A **subagent** = one process + one ACP session + a transcript file,
  addressed by its `session_id`. States: `idle | running | needs_permission |
  rate_limited(resume_at) | done(stop_reason) | cancelled | failed | exited |
  closed`. `failed` means
  the last turn errored while the agent process stays live — `send` resumes
  the same ACP session with a new turn. `exited` means the process is gone —
  the husk stays listed until `adopt` reaps it (no `close` needed) or
  `close`/`forget` removes it.
- **Rate limits**: when a turn's `session/prompt` fails with a provider's
  structured rate-limit error (e.g. Devin's `-32010` carrying
  `errorKind: "unavailable"` + `retryable: true`), the session parks in
  `rate_limited` until the reset the message's `(at HH:MM UTC)` clause
  names; a report with no readable clause fails the turn with the
  original error, never a guessed delay. The quota binds the agent
  config key — every session of one config runs under its single
  provider login — so every prompt `send`/`spawn`/`adopt` would send in
  the scope parks instead. At the reset each parked session resumes on
  its own in the same ACP session — `session/load` when its process is
  gone — on a fixed continuation prompt, staggered; prompts parked
  while limited run after the continuation. `wait` treats the pause as
  frozen: the parked seconds do not count against `expect_secs`.
- **Registry** (`registry.json`): `session_id → {agent, cwd, created,
  last_turn, turn_started?, turns, model?, mode?, owner?,
  rate_limited?}`, written
  atomically. `turn_started` is
  set while a turn is in flight — it is the base `expect_secs` measures
  from; `rate_limited` keeps a parked session's `{resume_at, reason}`
  across a daemon restart. `adopt` on a registered (or
  externally created) session id whose agent advertised `loadSession` runs
  `session/load` to resume it.
- **Transcript**: every `session/update`, prompt, and turn end is appended to
  `<transcript_dir>/<session_id>.jsonl` as a verbatim JSON record. A turn's
  `reply` is that turn's concatenated `agent_message_chunk` text.
- **Permissions**: `permission = allow|deny` answers requests by option kind;
  `ask` queues the request and flips the state to `needs_permission` until
  `permit` answers it.
- **Prompt queue**: `send` with `policy: "queued"` parks prompts FIFO while a
  turn runs; each fires as the next turn when the current one ends (the state
  stays `running` through the handoff). A `cancelled` or `failed` turn — and
  `cancel`/`close`/`forget` — drops the queue. `policy: "steer"` instead
  sends a second `session/prompt` mid-turn; agents that support mid-turn
  injection (e.g. `devin acp`) steer the active task with it. On an agent
  that serializes concurrent prompts rather than injecting them, a steer
  that lands behind the turn's end becomes a turn of its own — acpsub tracks
  it from the `session/update` traffic and holds the queue until it ends.
  Some agents fold a mid-turn prompt into the running turn but never answer
  its request (codex-acp); declare `steer = "folded"` for them so a steer
  still pending when its target turn ends settles as folded — a
  `steer_end: folded` transcript record — and stops blocking the session,
  while one that landed behind the turn's end still runs and resolves as a
  turn of its own.
- **cwd boundary**: `fs/*` and `terminal/*` requests are refused outside the
  subagent's `cwd` unless the agent has `allow_outside_cwd = true`.

## CLI

```sh
acpsub serve [--config PATH] [--log-file PATH]   # MCP over stdio
acpsub daemon [--socket PATH] [--config PATH] [--log-file PATH]
                                                # MCP over a Unix socket
acpsub agents [--config PATH]                    # list configured agents
acpsub transcript <session_id> [--from N] [--tail N] [--full] [--thinking]
```

Every tool in the table above is also a client subcommand that talks to the
daemon (`--socket PATH` overrides the default `~/.local/share/acpsub/
daemon.sock`; `ACPSUB_SOCKET` works too). When the socket is silent the
client spawns `acpsub daemon` itself, in its own process group with a pid
file at `<socket>.pid`:

```sh
acpsub spawn   --cwd DIR [--model M] [--mode M] [--agent K] [--set K=V]...
              [--permission allow|deny|ask] [--owner PID]
              --prompt TEXT | --prompt-file FILE   # FILE of `-` reads stdin
acpsub adopt   <session_id> [--model M] [--mode M] [--agent K] [--cwd DIR]
              [--prompt TEXT | --prompt-file FILE] [--permission P] [--owner PID]
acpsub send    <session_id> --policy try|queued|steer
              --prompt TEXT | --prompt-file FILE
acpsub wait    <session_id> --expect SECS [--max-wait SECS]
acpsub wait-any <session_id>... --expect SECS [--max-wait SECS]
acpsub status  <session_id>
acpsub result  <session_id> [--turn N]
acpsub cancel  <session_id>
acpsub permit  <session_id> --request REQ --option OPT
acpsub list
acpsub close   <session_id>
acpsub forget  <session_id>
acpsub agents-live                              # the daemon's `agents` tool
```

Client output is the tool's JSON result on stdout; a tool error exits 1 with
the message on stderr. `wait` is meant for a background task: the process
exit is the completion signal, so an orchestrating agent that backgrounds
`acpsub wait` is woken the moment the turn ends instead of holding a tool
call open.

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
